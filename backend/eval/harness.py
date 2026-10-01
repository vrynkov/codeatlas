"""Runs one question through the team and pulls out the numbers we care about,
by reading the same progress events the terminal and the MCP server already print.
No extra AI calls needed: we reuse what the team already produces."""
import re
from dataclasses import dataclass, field

from app.agents.llm import MODEL_CHAIN
from app.agents.team import run_team


@dataclass
class RunResult:
    id: str
    question: str
    route: str = ""
    final: str = ""
    critic_scores: list = field(default_factory=list)
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
                if m:
                    r.critic_scores.append(int(m.group(1)))
            elif ev["agent"] == "model" and ev["type"] in ("switch", "info"):
                r.models_used.append(ev["text"])
            elif ev["agent"] == "team" and ev["type"] == "answer":
                r.final = ev["text"]
    except Exception as e:  # noqa: BLE001  a crash on one question shouldn't stop the eval run
        r.error = f"{type(e).__name__}: {e}"
    r.cited_files = sorted({f.split("/")[-1] for f in _FILE_RE.findall(r.final)})
    return r


def score_item(item: dict, r: RunResult) -> dict:
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
    out.update({
        "critic_score": r.last_score,
        "attempts": r.attempts,
        "needed_retry": r.needed_retry,
        "file_hit": any(f in r.cited_files for f in exp_files) if exp_files else None,
        "file_mentioned": any(f.lower() in text_low for f in exp_files) if exp_files else None,
        "keyword_hit": any(k.lower() in text_low for k in exp_kw) if exp_kw else None,
        "cited_files": r.cited_files,
    })
    return out
