"""Re-scores an EXISTING results file with the current eval/questions.json ground truth and
the current citation-extraction regex - no new API calls, no quota spent. Useful whenever a
mistake in the ground truth or the extraction logic gets fixed after a run already happened
(exactly what happened here: a wrong expected_files entry, and a regex that missed a common
markdown citation style).

Run from backend/:
    python -m eval.rescore eval/results/requests-20260928-141615.json
"""
import json
import sys
from pathlib import Path

from .harness import _FILE_RE
from .run_eval import QUESTIONS, print_summary, summarize


def rescore(path: str) -> None:
    data = json.loads(Path(path).read_text())
    items = {q["id"]: q for q in json.loads(QUESTIONS.read_text())}
    rows = []
    for row in data["rows"]:
        item = items.get(row["id"])
        if not item or item["expected_route"] != "research" or "answer" not in row:
            rows.append(row)  # chat rows, or rows saved before the "answer" field existed
            continue
        cited = sorted({f.split("/")[-1] for f in _FILE_RE.findall(row["answer"])})
        exp_files = item.get("expected_files", [])
        row = {**row,
               "cited_files": cited,
               "file_hit": any(f in cited for f in exp_files) if exp_files else None,
               "file_mentioned": any(f.lower() in row["answer"].lower() for f in exp_files) if exp_files else None}
        rows.append(row)
    print_summary(f"RESCORED - {path}", summarize(rows))
    out = Path(path).with_name(Path(path).stem + "-rescored.json")
    out.write_text(json.dumps({"summary": summarize(rows), "rows": rows}, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rescore(sys.argv[1])
