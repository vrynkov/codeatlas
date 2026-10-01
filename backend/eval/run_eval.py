"""Runs the whole question set and prints a scorecard.

Run from backend/ (with the venv active):
    python -m eval.run_eval requests
    python -m eval.run_eval requests --clear-lessons   # test WITHOUT self-improvement, as a baseline
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

from app.agents.llm import MODEL_CHAIN

from .harness import run_one, score_item

QUESTIONS = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent / "results"


QUOTA_MARKER = "exhausted for today"  # from llm.py's MultiModelChatGroq once every model is spent
LONG_LIMIT_MARKER = "rate limit hit; try again in"  # from react_agent.py, for a non-daily long wait


def _crashed_before_routing(row: dict) -> bool:
    """True if the run failed so early that the Supervisor never got to pick a route.
    These are NOT routing mistakes and must not be scored as one."""
    return bool(row.get("error")) and not row["route"]


def run(repo: str, questions: list[dict]) -> list[dict]:
    print(f"Model order today: {' -> '.join(MODEL_CHAIN)} "
          f"(falls forward automatically if one hits its daily limit)\n", file=sys.stderr)
    rows = []
    last_model = None
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['id']}: {item['question'][:60]}", file=sys.stderr)
        r = run_one(repo, item)
        row = score_item(item, r)
        rows.append(row)
        if row.get("model") and row["model"] != last_model:
            arrow = "->" if last_model else "using"
            print(f"    [MODEL] {arrow} {row['model']}", file=sys.stderr)
            last_model = row["model"]
        if _crashed_before_routing(row):
            tag = "CRASHED"
        else:
            tag = "OK " if row["route_ok"] else "MISROUTE"
        extra = f" score={row.get('critic_score')} file_hit={row.get('file_hit')}" if "critic_score" in row else ""
        print(f"    -> {tag} route={row['route']}{extra}" + (f"  ({row['error'][:80]})" if row.get("error") else ""),
              file=sys.stderr)
        if QUOTA_MARKER in (row.get("error") or "") or LONG_LIMIT_MARKER in (row.get("error") or ""):
            print(f"\nStopping early - rate limit exhausted: {row['error'][:420]}\n"
                  f"({i}/{len(questions)} questions completed before this happened.)", file=sys.stderr)
            break
    return rows


def summarize(rows: list[dict]) -> dict:
    # A question that crashed before the Supervisor ran (e.g. the daily quota ran out) never got a
    # real routing decision, so it must not count as a routing mistake - just skip it for that metric.
    routed = [r for r in rows if not _crashed_before_routing(r)]
    research = [r for r in rows if r.get("critic_score") is not None]
    n, n_routed = len(rows), len(routed)
    return {
        "n_questions": n,
        "n_crashed": n - n_routed,
        "models_used": dict(Counter(r["model"] for r in rows if r.get("model"))),
        "any_model_switch": any(r.get("model_switched") for r in rows),
        "routing_accuracy": sum(r["route_ok"] for r in routed) / n_routed if n_routed else None,
        "n_research": len(research),
        "avg_critic_score": sum(r["critic_score"] for r in research) / len(research) if research else None,
        "retry_rate": sum(r["needed_retry"] for r in research) / len(research) if research else None,
        "file_hit_rate": sum(bool(r["file_hit"]) for r in research if r["file_hit"] is not None) / max(1, sum(1 for r in research if r["file_hit"] is not None)),
        "file_mention_rate": sum(bool(r.get("file_mentioned")) for r in research if r.get("file_mentioned") is not None) / max(1, sum(1 for r in research if r.get("file_mentioned") is not None)),
        "keyword_hit_rate": sum(bool(r["keyword_hit"]) for r in research if r["keyword_hit"] is not None) / max(1, sum(1 for r in research if r["keyword_hit"] is not None)),
        "errors": sum(1 for r in rows if r.get("error")),
    }


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
    if s["avg_critic_score"] is not None:
        print(f"Avg critic score   : {s['avg_critic_score']:.2f} / 5  ({s['n_research']} research questions)")
        print(f"Needed a retry     : {s['retry_rate']:.0%}")
        print(f"Cited expected file: {s['file_hit_rate']:.0%}  (precise file:line citation)")
        print(f"Mentioned expected file: {s['file_mention_rate']:.0%}  (named the file anywhere, any format)")
        print(f"Mentioned keyword  : {s['keyword_hit_rate']:.0%}")
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

    items = json.loads(QUESTIONS.read_text())
    rows = run(repo, items)
    summary = summarize(rows)
    print_summary(f"CodeAtlas eval - {repo}", summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{repo}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nSaved detailed results to {out}")
