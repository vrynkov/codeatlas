"""Measures whether the self-improvement loop (Phase 4) actually helps, with a number.

It runs the SAME research questions twice on the SAME repo:
  Pass 1 (baseline): lessons are cleared first, so the team starts with a blank slate.
  Pass 2 (improved): run right after, so it can use whatever lessons Pass 1 produced
                      (a "correction" lesson is saved automatically whenever a retry was needed).
A fair comparison needs the SAME questions both times; that's why this script exists
instead of just running run_eval.py twice.

Run from backend/:
    python -m eval.compare_lessons requests
"""
import json
import sys
from pathlib import Path

from .harness import run_one, score_item
from .run_eval import LONG_LIMIT_MARKER, QUESTIONS, QUOTA_MARKER, RESULTS_DIR, print_summary, summarize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.memory.lessons import LessonStore  # noqa: E402


def run_pass(repo: str, items: list[dict], label: str) -> list[dict]:
    rows = []
    for i, item in enumerate(items, 1):
        print(f"[{label} {i}/{len(items)}] {item['id']}", file=sys.stderr)
        row = score_item(item, run_one(repo, item))
        rows.append(row)
        if QUOTA_MARKER in (row.get("error") or "") or LONG_LIMIT_MARKER in (row.get("error") or ""):
            print(f"\nStopping {label} pass early: daily quota exhausted "
                  f"({i}/{len(items)} completed).", file=sys.stderr)
            break
    return rows


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    repo = sys.argv[1]
    items = [q for q in json.loads(QUESTIONS.read_text()) if q["expected_route"] == "research"]

    cleared = LessonStore(repo).clear()
    print(f"Cleared {cleared} lesson(s) for a clean baseline.\n", file=sys.stderr)

    before = run_pass(repo, items, "BEFORE")
    after = run_pass(repo, items, "AFTER")  # same questions; any lessons from BEFORE are now available

    s_before, s_after = summarize(before), summarize(after)
    print_summary("BEFORE (no lessons)", s_before)
    print_summary("AFTER (with lessons from the first pass)", s_after)

    print("\n=== Change ===")
    # Whether a bigger number is good depends on the metric: higher critic scores and higher
    # citation accuracy are better, but a LOWER retry rate is better (fewer do-overs needed).
    LOWER_IS_BETTER = {"retry_rate"}
    for key in ("avg_critic_score", "retry_rate", "file_hit_rate"):
        b, a = s_before.get(key), s_after.get(key)
        if b is not None and a is not None:
            if a == b:
                verdict = "same"
            else:
                improved = (a < b) if key in LOWER_IS_BETTER else (a > b)
                verdict = "better" if improved else "worse"
            print(f"{key:18s}: {b:.2f} -> {a:.2f}  ({verdict})")

    n_before, n_after = len(before), len(after)
    if n_before != n_after:
        print(f"\n(!) BEFORE covered {n_before} question(s) but AFTER only completed {n_after} before "
              f"stopping (see the note above) - treat this comparison as suggestive, not conclusive, "
              f"since it's not comparing the same number of questions on both sides.")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{repo}-lesson-comparison.json"
    out.write_text(json.dumps({"before": s_before, "after": s_after,
                               "before_rows": before, "after_rows": after}, indent=2))
    print(f"\nSaved to {out}")
