"""Turns a saved comparison JSON into a bar chart PNG for the README.

Run from backend/ AFTER compare_lessons.py has produced eval/results/<repo>-lesson-comparison.json:
    python -m eval.make_chart requests
"""
import json
import sys
from pathlib import Path

from .run_eval import RESULTS_DIR

METRICS = [("avg_critic_score", "Avg critic score (1-5 scale)", 5),
           ("retry_rate", "Needed a retry (%)", 1),
           ("file_hit_rate", "Cited expected file (%)", 1)]


def make_chart(repo: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save a PNG (works on any machine, free)
    import matplotlib.pyplot as plt

    data = json.loads((RESULTS_DIR / f"{repo}-lesson-comparison.json").read_text())
    before, after = data["before"], data["after"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(9, 3.2))
    for ax, (key, label, scale) in zip(axes, METRICS):
        b, a = before[key], after[key]  # plot the RAW value - `scale` only picks the axis range
                                        # and text format below, it must never divide the bar height
        bars = ax.bar(["Before", "After"], [b, a], color=["#94a3b8", "#22c55e"])
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1.15 if scale == 1 else 5.5)
        for rect, v in zip(bars, [b, a]):
            text = f"{v:.0%}" if scale == 1 else f"{v:.1f}"
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.03 * ax.get_ylim()[1],
                    text, ha="center", fontsize=9)
    fig.suptitle(f"CodeAtlas self-improvement on {repo}: before vs. after lessons", fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / f"{repo}-lesson-comparison.png"
    fig.savefig(out, dpi=150)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = make_chart(sys.argv[1])
    print(f"Saved chart to {path}")
