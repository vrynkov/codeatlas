"""The "no search" baseline: does CodeAtlas actually beat just asking the model?

It asks the SAME research questions to the SAME writer model, but with NO search, NO code and NO
critic - the model only gets the repo's name and must answer from memory. Then it scores both with
the same outside, no-AI checks (expected file named? keyword mentioned? do the cited file:line
locations really exist?) and prints them side by side.

How to read the result:
  - On a FAMOUS repo (requests), the baseline will look surprisingly good: the model has memorized
    it. That's exactly why a famous repo alone is a weak test.
  - On a repo the model has never seen (codeatlas itself), the baseline should fall apart - that
    gap is what the search + agents are really worth.

Cheap: one model call per question (vs ~10+ for the full team).

Run from backend/ (the repo must be indexed first, so its files are on disk for the citation check):
    python -m eval.baseline codeatlas
    python -m eval.baseline requests
"""
import json
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import active_model_name, get_llm
from app.agents.react_agent import invoke_with_recovery
from app.indexing.indexer import repo_source_url

from .harness import _FILE_RE, RunResult, check_citations, cost_summary, measured, score_item
from .run_eval import LONG_LIMIT_MARKER, QUOTA_MARKER, RESULTS_DIR, questions_for

BASELINE_PROMPT = """You are answering a question about the GitHub repository {url}.
You CANNOT see its code. Answer only from what you already know about it.
Use the same format a code assistant would:
1. A direct answer in 1-2 sentences.
2. How it works, in plain language.
3. "Where to look": the files and lines, written as `path/to/file.py:START-END`.
If you don't know, say so honestly. Stay under about 250 words."""

METRICS = [("file_mention_rate", "Named the expected file"),
           ("file_hit_rate", "Cited expected file (file:line)"),
           ("keyword_hit_rate", "Mentioned the keyword"),
           ("valid_citation_rate", "Citations that really exist")]


def run_baseline(repo: str, items: list[dict]) -> list[dict]:
    llm = get_llm()
    system = BASELINE_PROMPT.format(url=repo_source_url(repo) or f"named '{repo}'")
    rows = []
    for i, item in enumerate(items, 1):
        print(f"[baseline {i}/{len(items)}] {item['id']}", file=sys.stderr)
        r = RunResult(id=item["id"], question=item["question"], route="research")
        with measured(r):
            try:
                r.final = invoke_with_recovery(llm, [SystemMessage(system), HumanMessage(item["question"])]).content
                r.models_used.append(active_model_name())
            except Exception as e:  # noqa: BLE001
                r.error = f"{type(e).__name__}: {e}"
        r.cited_files = sorted({f.split("/")[-1] for f in _FILE_RE.findall(r.final)})
        rows.append(score_item(item, r, repo))
        if QUOTA_MARKER in r.error or LONG_LIMIT_MARKER in r.error:
            print("Stopping early - rate limit exhausted.", file=sys.stderr)
            break
    return rows


def outside_metrics(rows: list[dict], repo: str) -> dict:
    """Only the checks that need no AI - so the baseline and the team are judged identically."""
    def rate(key):
        vals = [bool(r.get(key)) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    total = valid = 0
    for r in rows:
        c = check_citations(r.get("answer", ""), repo)  # recomputed, so older team results work too
        total += c["citations"]
        valid += c["valid_citations"]
    return {"n": len(rows), "file_mention_rate": rate("file_mentioned"), "file_hit_rate": rate("file_hit"),
            "keyword_hit_rate": rate("keyword_hit"), "citations": total,
            "valid_citation_rate": valid / total if total else None,
            "invented_paths": sum(r.get("invented_paths", 0) for r in rows), **cost_summary(rows)}


def latest_team_results(repo: str) -> tuple[str, list[dict]] | None:
    files = sorted(p for p in RESULTS_DIR.glob(f"{repo}-2*.json")
                   if "baseline" not in p.name and "rescored" not in p.name)
    if not files:
        return None
    rows = json.loads(files[-1].read_text())["rows"]
    return files[-1].name, [r for r in rows if "critic_score" in r]


def _pct(x) -> str:
    return "  n/a" if x is None else f"{x:5.0%}"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    repo = sys.argv[1]
    items = [q for q in questions_for(repo) if q["expected_route"] == "research"]
    rows = run_baseline(repo, items)
    base = outside_metrics(rows, repo)

    team = latest_team_results(repo)
    team_m = outside_metrics(team[1], repo) if team else None
    print(f"\n=== {repo}: CodeAtlas team vs. the same model with NO search ===")
    if team:
        print(f"(team numbers from {team[0]}; run `python -m eval.run_eval {repo}` for fresh ones)")
    print(f"{'':34}{'team':>8}{'no search':>11}")
    for key, label in METRICS:
        print(f"{label:34}{_pct(team_m[key]) if team_m else '    -':>8}{_pct(base[key]):>11}")
    print(f"{'Questions':34}{team_m['n'] if team_m else '-':>8}{base['n']:>11}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{repo}-baseline-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": base, "rows": rows}, indent=2))
    print(f"\nSaved to {out}")
