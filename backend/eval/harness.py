"""Runs one question through the team and pulls out the numbers we care about,
by reading the same progress events the terminal and the MCP server already print.
No extra AI calls needed: we reuse what the team already produces."""
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.agents.llm import MODEL_CHAIN
from app.agents.team import NOT_FOUND_LABEL, run_team
from app.config import PASS_SCORE, REPOS_DIR


@dataclass
class RunResult:
    id: str
    question: str
    route: str = ""
    final: str = ""
    critic_scores: list = field(default_factory=list)  # one entry per review: an int, or None = BROKEN review
    critic_models: list = field(default_factory=list)  # which model did each review
    cited_files: list = field(default_factory=list)
    models_used: list = field(default_factory=list)  # models switched TO during this question, in order
    error: str = ""

    @property
    def final_model(self) -> str:
        """Whichever model actually produced this question's answer: the last one switched to,
        or the default (first in the chain, i.e. GROQ_MODEL) if no switch happened at all."""
        return self.models_used[-1] if self.models_used else MODEL_CHAIN[0]

    @property
    def attempts(self) -> int:
        return max(len(self.critic_scores), 1 if self.route == "research" else 0)

    @property
    def needed_retry(self) -> bool:
        return self.attempts > 1

    @property
    def last_score(self) -> int | None:
        return self.critic_scores[-1] if self.critic_scores else None

    @property
    def critic_failed(self) -> bool:
        """The final review was broken (the critic returned no usable score)."""
        return bool(self.critic_scores) and self.critic_scores[-1] is None


_FILE_RE = re.compile(r"([\w./-]+\.\w{1,4})`?(?::\d|\s*\(?lines?\s+\d)", re.I)  # file.py:123 OR
                                                                                # "`file.py` lines 123" OR
                                                                                # "file.py (lines 123)"
_SCORE_RE = re.compile(r"score (\d)/5")


def run_one(repo: str, item: dict, llm=None) -> RunResult:
    r = RunResult(id=item["id"], question=item["question"])
    try:
        for ev in run_team(repo, item["question"], llm=llm):
            if ev["agent"] == "supervisor" and ev["type"] == "route":
                r.route = ev["text"].split()[0].strip()
            elif ev["agent"] == "critic" and ev["type"] == "review":
                m = _SCORE_RE.search(ev["text"])
                r.critic_scores.append(int(m.group(1)) if m else None)  # unreadable = broken, not skipped
            elif ev["agent"] == "critic" and ev["type"] == "review_failed":
                r.critic_scores.append(None)
            elif ev["agent"] == "critic" and ev["type"] == "model":
                r.critic_models.append(ev["text"])
            elif ev["agent"] == "model" and ev["type"] in ("switch", "info"):
                r.models_used.append(ev["text"])
            elif ev["agent"] == "team" and ev["type"] == "answer":
                r.final = ev["text"]
    except Exception as e:  # noqa: BLE001  a crash on one question shouldn't stop the eval run
        r.error = f"{type(e).__name__}: {e}"
    r.cited_files = sorted({f.split("/")[-1] for f in _FILE_RE.findall(r.final)})
    return r


_CITE_RE = re.compile(r"([\w.\-/]+\.\w{1,4}):(\d+)(?:[-‑–—](\d+))?")


def check_citations(answer: str, repo: str, repo_dir: Path | None = None) -> dict:
    """An outside, no-AI check: for every `path:LINE` or `path:START-END` citation in the answer,
    does that file exist in the repo, and does it really have those lines? A model answering from
    memory (no search) tends to invent plausible paths and line numbers - this catches that."""
    root = repo_dir or (REPOS_DIR / repo)
    if not root.exists():
        return {"citations": 0, "valid_citations": 0, "invalid_citations": []}
    files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    total, valid, bad = 0, 0, []
    for path, start, end in _CITE_RE.findall(answer):
        total += 1
        matches = [f for f in files if f == path or f.endswith("/" + path)]
        ok = False
        if len(matches) == 1:
            try:
                n_lines = len((root / matches[0]).read_text(encoding="utf-8", errors="replace").splitlines())
                s, e = int(start), int(end or start)
                ok = 1 <= s <= e <= n_lines
            except OSError:
                ok = False
        valid += ok
        if not ok:
            bad.append(f"{path}:{start}" + (f"-{end}" if end else ""))
    return {"citations": total, "valid_citations": valid, "invalid_citations": bad}


def score_item(item: dict, r: RunResult, repo: str | None = None) -> dict:
    """Compares one run against the question's expectations. Two INDEPENDENT checks on purpose:
    the Critic score comes from inside the pipeline; the keyword/file checks are an outside judge,
    so a bug in the Critic's own prompt can't make the eval look better than it really is.

    Two file-citation signals, kept separate and both reported (never silently pick the
    flattering one): `file_hit` is strict - the file appears as a precise `file.py:123`
    citation the way search results/get_file format them. `file_mentioned` is loose - the
    filename appears ANYWHERE in the answer, even in prose with no line number. A big gap
    between the two usually means the model named the right file conversationally rather
    than in the strict citation format, not that it got the wrong file."""
    out = {"id": r.id, "route_ok": r.route == item["expected_route"], "route": r.route,
           "model": r.final_model, "model_switched": r.final_model != MODEL_CHAIN[0], "error": r.error,
           "answer": r.final}  # saved in full so a low score can actually be diagnosed later
    if item["expected_route"] != "research":
        return out
    text_low = r.final.lower()
    exp_files = item.get("expected_files", [])
    exp_kw = item.get("expected_keywords", [])
    writer = r.final_model
    out.update({
        "critic_score": r.last_score,
        # A broken review is a FAIL, never a pass, and the row stays in every average.
        "critic_failed": r.critic_failed,
        "passed": (not r.error) and r.route == "research" and r.last_score is not None and r.last_score >= PASS_SCORE,
        "critic_model": r.critic_models[-1] if r.critic_models else None,
        # The critic is meant to be a DIFFERENT model from the writer; flag when fallback made them equal.
        "critic_same_model": bool(r.critic_models) and r.critic_models[-1] == writer,
        "attempts": r.attempts,
        "needed_retry": r.needed_retry,
        "file_hit": any(f in r.cited_files for f in exp_files) if exp_files else None,
        "file_mentioned": any(f.lower() in text_low for f in exp_files) if exp_files else None,
        "keyword_hit": any(k.lower() in text_low for k in exp_kw) if exp_kw else None,
        "cited_files": r.cited_files,
        # Made-up file paths the app caught and labelled before showing the answer. They never
        # reach the citation check above, so they're counted here to keep that 100% honest.
        "invented_paths": r.final.count(NOT_FOUND_LABEL),
        **(check_citations(r.final, repo) if repo else {}),
    })
    return out
