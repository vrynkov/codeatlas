"""Runs the whole question set and prints a scorecard.

Run from backend/ (with the venv active):
    python -m eval.run_eval requests
    python -m eval.run_eval codeatlas                  # a repo the models have never seen (see README)
    python -m eval.run_eval requests --clear-lessons   # test WITHOUT self-improvement, as a baseline

Questions come from eval/questions_<repo>.json if that file exists, else eval/questions.json.
Compare against "the same model with NO search at all" with:  python -m eval.baseline <repo>
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

from app.agents.llm import CRITIC_MODEL_CHAIN, MODEL_CHAIN

from .harness import run_one, score_item

QUESTIONS = Path(__file__).parent / "questions.json"


def questions_for(repo: str) -> list[dict]:
    """Each repo can have its own question set: eval/questions_<repo>.json (falls back to the
    original requests set in questions.json)."""
    own = Path(__file__).parent / f"questions_{repo}.json"
    return json.loads((own if own.exists() else QUESTIONS).read_text(encoding="utf-8"))
RESULTS_DIR = Path(__file__).parent / "results"


QUOTA_MARKER = "exhausted for today"  # from llm.py's MultiModelChatGroq once every model is spent
LONG_LIMIT_MARKER = "rate limit hit; try again in"  # from react_agent.py, for a non-daily long wait


def _crashed_before_routing(row: dict) -> bool:
    """True if the run failed so early that the Supervisor never got to pick a route.
    These are NOT routing mistakes and must not be scored as one."""
    return bool(row.get("error")) and not row["route"]


def run(repo: str, questions: list[dict]) -> list[dict]:
    print(f"Writer model order : {' -> '.join(MODEL_CHAIN)} "
          f"(falls forward automatically if one hits its limit)\n"
          f"Critic model order : {' -> '.join(CRITIC_MODEL_CHAIN)} (a different model grades the answer)\n",
          file=sys.stderr)
    rows = []
    last_model = None
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['id']}: {item['question'][:60]}", file=sys.stderr)
        r = run_one(repo, item)
        row = score_item(item, r, repo)
        rows.append(row)
        if row.get("model") and row["model"] != last_model:
            arrow = "->" if last_model else "using"
            print(f"    [MODEL] {arrow} {row['model']}", file=sys.stderr)
            last_model = row["model"]
        if _crashed_before_routing(row):
            tag = "CRASHED"
        else:
            tag = "OK " if row["route_ok"] else "MISROUTE"
        extra = (f" score={'BROKEN' if row.get('critic_failed') else row.get('critic_score')}"
                 f" passed={row.get('passed')} file_hit={row.get('file_hit')}") if "critic_score" in row else ""
        print(f"    -> {tag} route={row['route']}{extra}" + (f"  ({row['error'][:80]})" if row.get("error") else ""),
              file=sys.stderr)
        if QUOTA_MARKER in (row.get("error") or "") or LONG_LIMIT_MARKER in (row.get("error") or ""):
            print(f"\nStopping early - rate limit exhausted: {row['error'][:420]}\n"
                  f"({i}/{len(questions)} questions completed before this happened.)", file=sys.stderr)
            break
    return rows


def _rate(rows: list[dict], key: str):
    vals = [bool(r.get(key)) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list[dict]) -> dict:
    # A question that crashed before the Supervisor ran (e.g. the daily quota ran out) never got a
    # real routing decision, so it must not count as a routing mistake - just skip it for that metric.
    routed = [r for r in rows if not _crashed_before_routing(r)]
    # EVERY research question that got routed counts below - including a misroute, a broken review or
    # a mid-run error. (Earlier versions only counted rows with a readable critic score, which quietly
    # dropped the failures from the averages and made the numbers look better than they were.)
    research = [r for r in routed if "critic_score" in r]
    scored = [r for r in research if r.get("critic_score") is not None]
    n, n_routed = len(rows), len(routed)
    cites = sum(r.get("citations", 0) for r in research)
    return {
        "n_questions": n,
        "n_crashed": n - n_routed,
        "models_used": dict(Counter(r["model"] for r in rows if r.get("model"))),
        "any_model_switch": any(r.get("model_switched") for r in rows),
        "routing_accuracy": sum(r["route_ok"] for r in routed) / n_routed if n_routed else None,
        "n_research": len(research),
        "pass_rate": sum(bool(r.get("passed")) for r in research) / len(research) if research else None,
        "critic_failures": sum(bool(r.get("critic_failed")) for r in research),
        "critic_same_model": sum(bool(r.get("critic_same_model")) for r in research),
        "avg_critic_score": sum(r["critic_score"] for r in scored) / len(scored) if scored else None,
        "n_scored": len(scored),
        "retry_rate": sum(bool(r.get("needed_retry")) for r in research) / len(research) if research else None,
        "file_hit_rate": _rate(research, "file_hit"),
        "file_mention_rate": _rate(research, "file_mentioned"),
        "keyword_hit_rate": _rate(research, "keyword_hit"),
        "citations": cites,
        "valid_citation_rate": sum(r.get("valid_citations", 0) for r in research) / cites if cites else None,
        "invented_paths": sum(r.get("invented_paths", 0) for r in research),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def print_summary(title: str, s: dict) -> None:
    print(f"\n=== {title} ===")
    if s.get("models_used"):
        counts = ", ".join(f"{m}: {n}" for m, n in s["models_used"].items())
        flag = "  (!) answers came from more than one model - see per-row detail before quoting an overall score" \
               if len(s["models_used"]) > 1 else ""
        print(f"Model(s) used      : {counts}{flag}")
    if s.get("n_crashed"):
        print(f"(!) {s['n_crashed']} question(s) crashed before a route was chosen - excluded below, not counted as misrouted")
    if s["routing_accuracy"] is None:
        print("Routing accuracy   : n/a (every question crashed before routing)")
    else:
        print(f"Routing accuracy   : {s['routing_accuracy']:.0%}  ({s['n_questions'] - s['n_crashed']} questions)")
    if s.get("n_research"):
        print(f"Passed review      : {_pct(s.get('pass_rate'))}  ({s['n_research']} research questions; "
              f"a broken review, misroute or error counts as FAIL)")
        if s.get("critic_failures"):
            print(f"(!) Broken reviews : {s['critic_failures']}  (critic returned no usable score - counted as fail)")
        if s.get("critic_same_model"):
            print(f"(!) Self-graded    : {s['critic_same_model']} answer(s) were reviewed by the SAME model that "
                  f"wrote them (critic model was rate limited) - those scores are less trustworthy")
        if s["avg_critic_score"] is not None:
            print(f"Avg critic score   : {s['avg_critic_score']:.2f} / 5  (over the {s['n_scored']} readable reviews)")
        print(f"Needed a retry     : {_pct(s['retry_rate'])}")
        print(f"Cited expected file: {_pct(s['file_hit_rate'])}  (precise file:line citation)")
        print(f"Mentioned expected file: {_pct(s['file_mention_rate'])}  (named the file anywhere, any format)")
        print(f"Mentioned keyword  : {_pct(s['keyword_hit_rate'])}")
        if s.get("citations"):
            print(f"Citations that exist: {_pct(s['valid_citation_rate'])}  ({s['citations']} file:line citations "
                  f"checked against the real files - no AI involved)")
        if "invented_paths" in s:
            print(f"Invented file paths: {s['invented_paths']}  (made-up files the app caught and labelled "
                  f"'not found' - not counted in the line above)")
    if s["errors"]:
        print(f"Errors             : {s['errors']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    repo = sys.argv[1]
    if "--clear-lessons" in sys.argv:
        from app.memory.lessons import LessonStore
        n = LessonStore(repo).clear()
        print(f"Cleared {n} existing lesson(s) for a clean baseline.", file=sys.stderr)

    items = questions_for(repo)
    rows = run(repo, items)
    summary = summarize(rows)
    print_summary(f"CodeAtlas eval - {repo}", summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{repo}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nSaved detailed results to {out}")
