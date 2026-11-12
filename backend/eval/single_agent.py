"""Team vs ONE agent: would a single ReAct agent have done the job just as well, for less?

The team spends several model calls per question (Supervisor routes, Researcher digs, Explainer
rewrites, Critic grades, sometimes a second round). The simplest setup that can still read the
code is ONE ReAct agent (app/agents/react_agent.py) that searches and writes the answer itself.
This script runs the SAME research questions through that one agent and scores it with the SAME
outside, no-AI checks as the team and the no-search baseline - plus tokens and seconds per question.

What is kept the same, so the comparison is fair:
  - same questions (the research ones; there's no Supervisor, so the 2 chat questions are skipped)
  - same model chain, same tools (search_code, get_file), same step limit (MAX_AGENT_STEPS)
  - the same plain-code citation filter the team applies in finalize() (_fix_citation_paths), so
    an invented path is labelled "(not found in this repo)" here too. It's code, not an agent,
    so giving it to both sides isolates what the extra AGENTS add.
What is different, on purpose: no lessons memory, no Explainer rewrite, no Critic, no retry.
So also run the team with `--no-lessons`, otherwise it gets hints from earlier runs.

Run from backend/ (the repo must be indexed first):
    python -m eval.single_agent codeatlas
Then print the README table for all three setups with:
    python -m eval.compare_setups codeatlas
"""
import json
import sys
import time

from langgraph.errors import GraphRecursionError

from app.agents.llm import active_model_name
from app.agents.react_agent import ask
from app.agents.team import _fix_citation_paths

from .baseline import METRICS, _pct, outside_metrics
from .harness import _FILE_RE, RunResult, measured, score_item
from .run_eval import LONG_LIMIT_MARKER, QUOTA_MARKER, RESULTS_DIR, questions_for


def run_single(repo: str, item: dict, llm=None) -> RunResult:
    """One question through react_agent.ask(): the last message without a tool call is the answer."""
    r = RunResult(id=item["id"], question=item["question"], route="research")  # no Supervisor: always research
    with measured(r):
        try:
            for node, msg in ask(repo, item["question"], llm=llm):
                if node == "agent" and not getattr(msg, "tool_calls", None):
                    r.final = msg.content
        except GraphRecursionError:
            r.error = "GraphRecursionError: the agent used all its steps without answering"
        except Exception as e:  # noqa: BLE001  one crash shouldn't stop the run
            r.error = f"{type(e).__name__}: {e}"
    r.final = _fix_citation_paths(r.final, repo)
    r.models_used.append(getattr(llm, "last_model", None) or active_model_name())
    r.cited_files = sorted({f.split("/")[-1] for f in _FILE_RE.findall(r.final)})
    return r


def run_all(repo: str, items: list[dict], llm=None) -> list[dict]:
    rows = []
    for i, item in enumerate(items, 1):
        print(f"[single agent {i}/{len(items)}] {item['id']}", file=sys.stderr)
        r = run_single(repo, item, llm)
        row = score_item(item, r, repo)
        rows.append(row)
        print(f"    -> file_hit={row.get('file_hit')} tokens={row['tokens']} {row['seconds']:.0f}s "
              f"model={row['model']}" + (f"  ({row['error'][:80]})" if row["error"] else ""), file=sys.stderr)
        if QUOTA_MARKER in r.error or LONG_LIMIT_MARKER in r.error:
            print("Stopping early - rate limit exhausted.", file=sys.stderr)
            break
    return rows


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    repo = sys.argv[1]
    items = [q for q in questions_for(repo) if q["expected_route"] == "research"]
    rows = run_all(repo, items)
    s = outside_metrics(rows, repo)
    s["models_used"] = {m: sum(r["model"] == m for r in rows) for m in {r["model"] for r in rows}}
    s["errors"] = sum(bool(r["error"]) for r in rows)

    print(f"\n=== {repo}: ONE ReAct agent (no team) ===")
    for key, label in METRICS:
        print(f"{label:34}{_pct(s[key]):>8}")
    print(f"{'Invented paths caught':34}{s['invented_paths']:>8}")
    if s["avg_tokens"] is not None:
        print(f"{'Tokens / question':34}{s['avg_tokens']:>8.0f}")
        print(f"{'Model calls / question':34}{s['avg_llm_calls']:>8.1f}")
        print(f"{'Seconds / question':34}{s['avg_seconds']:>8.0f}  ({s['avg_llm_seconds']:.0f}s model busy)")
    print(f"{'Questions':34}{s['n']:>8}   models: {s['models_used']}   errors: {s['errors']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{repo}-single-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": s, "rows": rows}, indent=2))
    print(f"\nSaved to {out}\nNext: python -m eval.compare_setups {repo}")
