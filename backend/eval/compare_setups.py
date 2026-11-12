"""Prints ONE table comparing three setups on the same questions, ready to paste into the README:

    Same model, no search  (eval/baseline.py)
    One ReAct agent        (eval/single_agent.py)
    Full team              (eval/run_eval.py)

No model calls, no quota: it only reads the latest saved result files (or the ones you name).

Honesty rules built in:
  - Only questions that EVERY setup finished are compared. A question cut off because the daily
    quota ran out is dropped from all three, never counted as a fail for just one of them.
  - Team vs one agent is also shown question by question ("only the team got it right: 2"), which
    is what matters with so few questions.
  - It warns when the team used saved lessons, when an old file has no token counts, and when the
    two setups' answers came from different models.

Run from backend/:
    python -m eval.compare_setups codeatlas
    python -m eval.compare_setups codeatlas --team eval/results/codeatlas-20261112-101500.json
"""
import json
import sys
from pathlib import Path

from .baseline import outside_metrics
from .harness import cost_summary
from .run_eval import LONG_LIMIT_MARKER, QUOTA_MARKER, RESULTS_DIR


def _latest(pattern: str, exclude: tuple = ()) -> Path | None:
    files = sorted(p for p in RESULTS_DIR.glob(pattern) if not any(x in p.name for x in exclude))
    return files[-1] if files else None


def find_files(repo: str, argv: list[str]) -> dict[str, Path | None]:
    files = {"baseline": _latest(f"{repo}-baseline-2*.json"),
             "single": _latest(f"{repo}-single-2*.json"),
             "team": _latest(f"{repo}-2*.json", exclude=("rescored",))}
    for name in files:  # --team / --single / --baseline FILE overrides the "latest" choice
        flag = f"--{name}"
        if flag in argv and argv.index(flag) + 1 < len(argv):
            files[name] = Path(argv[argv.index(flag) + 1])
    return files


def load(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if "file_hit" in r]  # research questions only (chat rows have none)
    return data.get("summary", {}), rows


def _quota_cut(row: dict) -> bool:
    err = row.get("error") or ""
    return QUOTA_MARKER in err or LONG_LIMIT_MARKER in err


def common_ids(setups: dict[str, list[dict]]) -> list[str]:
    """Question ids that every setup actually got to answer (in the first setup's order)."""
    done = [{r["id"] for r in rows if not _quota_cut(r)} for rows in setups.values()]
    first = next(iter(setups.values()))
    return [r["id"] for r in first if all(r["id"] in d for d in done)]


def setup_metrics(rows: list[dict], repo: str) -> dict:
    m = outside_metrics(rows, repo)
    m.update(cost_summary(rows))
    return m


def paired(team: list[dict], single: list[dict], key: str = "file_hit") -> dict:
    """Question by question: who got it right?"""
    s_by_id = {r["id"]: r for r in single}
    out = {"both": 0, "only_team": 0, "only_single": 0, "neither": 0, "different_model": 0}
    for t in team:
        s = s_by_id[t["id"]]
        a, b = bool(t.get(key)), bool(s.get(key))
        out["both" if a and b else "only_team" if a else "only_single" if b else "neither"] += 1
        out["different_model"] += t.get("model") != s.get("model")
    return out


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _num(x, fmt="{:,.0f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def table(metrics: dict[str, dict]) -> str:
    labels = {"baseline": "Same model, no search", "single": "One ReAct agent", "team": "Full team"}
    lines = ["| Setup | Cited right file | Real citations | Tokens / question | Model calls / question "
             "| Seconds / question (model busy) |",
             "|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        secs = "n/a" if m["avg_seconds"] is None else f"{m['avg_seconds']:.0f} ({m['avg_llm_seconds']:.0f})"
        lines.append(f"| {labels[name]} | {_pct(m['file_hit_rate'])} | {_pct(m['valid_citation_rate'])} "
                     f"({m['citations']}) | {_num(m['avg_tokens'])} | {_num(m['avg_llm_calls'], '{:.1f}')} "
                     f"| {secs} |")
    return "\n".join(lines)


def main(repo: str, argv: list[str]) -> None:
    files = find_files(repo, argv)
    missing = [n for n, p in files.items() if p is None or not p.exists()]
    if missing:
        sys.exit(f"No result file for: {', '.join(missing)}. Run eval.baseline / eval.single_agent / "
                 f"eval.run_eval for {repo} first (see the README).")
    loaded = {n: load(p) for n, p in files.items()}
    ids = common_ids({n: rows for n, (_, rows) in loaded.items()})
    if not ids:
        sys.exit("No question was finished by all three setups - nothing fair to compare.")
    rows = {n: [r for r in loaded[n][1] if r["id"] in ids] for n in files}
    metrics = {n: setup_metrics(rows[n], repo) for n in files}

    print(f"Files: " + ", ".join(f"{n}={p.name}" for n, p in files.items()))
    print(f"Compared on the {len(ids)} research question(s) all three setups finished.\n")
    print(table(metrics))

    p = paired(rows["team"], rows["single"])
    print(f"\nTeam vs one agent, question by question (cited the right file):\n"
          f"  both right: {p['both']}   only the team: {p['only_team']}   "
          f"only the one agent: {p['only_single']}   neither: {p['neither']}")
    t, s = metrics["team"]["avg_tokens"], metrics["single"]["avg_tokens"]
    if t and s:
        print(f"  The team used {t / s:.1f}x the tokens of the one agent.")

    print("\nRead this before quoting it:")
    print(f"  - {len(ids)} questions: one question = {1 / len(ids):.0%}. A gap of 1-2 questions is noise "
          f"(identical team runs have moved 20 points before).")
    team_summary = loaded["team"][0]
    if team_summary.get("lessons_used") is not False:
        print("  (!) The team run may have used saved lessons (hints from earlier runs of these questions). "
              "Re-run it with --no-lessons for a fair comparison.")
    for n in files:
        if metrics[n]["n_measured"] < len(ids):
            print(f"  (!) {n}: only {metrics[n]['n_measured']}/{len(ids)} rows have token counts "
                  f"(older result file) - re-run it to get the cost columns.")
    if p["different_model"]:
        print(f"  (!) On {p['different_model']} question(s) the team and the one agent were answered by "
              f"DIFFERENT models (quota fallback) - those rows compare models, not setups.")
    for n in files:
        models = sorted({r.get("model") for r in rows[n] if r.get("model")})
        print(f"  {n} answer model(s): {', '.join(models) or 'unknown'}")
    print("  - Seconds include rate-limit waits, which hit the team harder (more calls per minute). "
          "The number in brackets is time the model was actually working.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
