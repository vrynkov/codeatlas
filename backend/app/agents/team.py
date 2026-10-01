"""The multi-agent team, orchestrated with LangGraph.

    START -> supervisor --research--> researcher -> explainer -> critic --pass--> finalize -> END
                  |                       ^                        |
                  +--chat--> chat -> END  +------ retry (with the critic's advice) ------+

Design rule: the LLM decides WHAT to do (route, review). Plain code decides the loop control
(max attempts, pass score). That keeps free models from looping forever.

Run from the backend/ folder:
    python -m app.agents.team requests "How are retries handled?"
"""
import json
import re
import sys
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from ..config import GROQ_API_KEY, MAX_AGENT_STEPS, MAX_ATTEMPTS, MAX_EVIDENCE_CHARS, PASS_SCORE
from ..memory.lessons import LessonStore, lesson_from_retry, record_feedback
from .llm import active_model_name
from .prompts import CHAT_PROMPT, CRITIC_PROMPT, EXPLAINER_PROMPT, RESEARCHER_PROMPT, SUPERVISOR_PROMPT
from .react_agent import build_graph, invoke_with_recovery


class TeamState(TypedDict):
    question: str
    route: str
    attempts: int
    findings: str
    evidence: list      # pieces of code the researcher looked at
    files: list          # files cited, for the lesson we may save later
    queries: list         # search_code queries the researcher tried, for the lesson
    draft: str
    score: int
    advice: str         # the critic's tips for the next attempt
    first_advice: str    # the critic's advice on attempt 1, kept even after a passing retry
    final: str
    lessons_used: list


# ---------- helpers ----------
def _emit(agent: str, kind: str, text: str = ""):
    """Send a progress event to whoever is listening (the terminal now, the web UI later)."""
    get_stream_writer()({"agent": agent, "type": kind, "text": text})


def _ask_json(llm, system: str, user: str):
    """Ask for JSON and parse it. Returns None if the model did not give valid JSON."""
    text = invoke_with_recovery(llm, [SystemMessage(system), HumanMessage(user)]).content
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _evidence_text(evidence: list) -> str:
    return "\n\n".join(evidence)[:MAX_EVIDENCE_CHARS]


# ---------- the graph ----------
def build_team_graph(repo_name: str, llm):
    def supervisor(state: TeamState):
        _emit("model", "info", active_model_name())  # which model we're STARTING this question with
        data = _ask_json(llm, SUPERVISOR_PROMPT, state["question"]) or {}
        route = data.get("route") if data.get("route") in ("research", "chat") else "research"
        _emit("supervisor", "route", f"{route}  ({data.get('reason', 'default')})")
        return {"route": route}

    def chat(state: TeamState):
        reply = invoke_with_recovery(llm, [SystemMessage(CHAT_PROMPT), HumanMessage(state["question"])]).content
        _emit("model", "info", active_model_name())  # which model actually answered, for the eval to record
        _emit("chat", "answer", reply)
        return {"final": reply}

    def researcher(state: TeamState):
        task = state["question"]
        lessons_used = list(state["lessons_used"])
        if state["attempts"] == 0:  # first attempt only: recall relevant past lessons
            hits = LessonStore(repo_name).recall(state["question"])
            if hits:
                tips = "\n".join(f"- ({h['kind']}) {h['lesson']}" for h in hits)
                task += f"\n\nHINTS from past questions (verify, don't just trust):\n{tips}"
                lessons_used = [h["lesson"] for h in hits]
                _emit("memory", "recall", f"using {len(hits)} past lesson(s)")
        if state["advice"]:  # second attempt: include the critic's feedback
            task += ("\n\nA reviewer checked your previous findings and gave this advice. "
                     f"Investigate it specifically:\n{state['advice']}")
        graph = build_graph(repo_name, llm, RESEARCHER_PROMPT)
        findings, evidence = "", list(state["evidence"])
        files, queries = set(state["files"]), list(state["queries"])
        try:
            for update in graph.stream({"messages": [HumanMessage(task)]},
                                       {"recursion_limit": MAX_AGENT_STEPS * 2 + 1}, stream_mode="updates"):
                for node, data in update.items():
                    for msg in data["messages"]:
                        if node == "agent" and getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                _emit("researcher", "act", f"{tc['name']}({tc['args']})")
                                if tc["name"] == "search_code":
                                    queries.append(str(tc["args"].get("query", "")))
                                elif tc["name"] == "get_file":
                                    files.add(str(tc["args"].get("path", "")))
                        elif node == "tools":
                            evidence.append(msg.content)
                            _emit("researcher", "observe", f"{len(msg.content)} characters")
                        else:
                            findings = msg.content
        except GraphRecursionError:
            findings = findings or "(the researcher ran out of steps)"
        except Exception as e:  # noqa: BLE001  keep going with whatever evidence we have
            _emit("researcher", "error", f"{type(e).__name__}: {str(e)[:150]}")
            findings = findings or "(the researcher failed; answer only from the evidence)"
        return {"findings": findings, "evidence": evidence, "attempts": state["attempts"] + 1,
                "files": sorted(f for f in files if f), "queries": queries, "lessons_used": lessons_used}

    def explainer(state: TeamState):
        _emit("explainer", "write", "writing the answer")
        user = (f"QUESTION:\n{state['question']}\n\nFINDINGS:\n{state['findings']}\n\n"
                f"EVIDENCE:\n{_evidence_text(state['evidence'])}")
        draft = invoke_with_recovery(llm, [SystemMessage(EXPLAINER_PROMPT), HumanMessage(user)]).content
        return {"draft": draft}

    def critic(state: TeamState):
        user = (f"QUESTION:\n{state['question']}\n\nEVIDENCE:\n{_evidence_text(state['evidence'])}\n\n"
                f"ANSWER:\n{state['draft']}")
        data = _ask_json(llm, CRITIC_PROMPT, user)
        if data is None:  # the critic failed to answer properly: don't block the user
            _emit("critic", "review", "no valid review returned; accepting the answer")
            return {"score": PASS_SCORE, "advice": "", "first_advice": state["first_advice"]}
        try:
            score = int(data.get("score", PASS_SCORE))
        except (TypeError, ValueError):
            score = PASS_SCORE
        advice = data.get("advice", "") or ""
        problems = (data.get("unsupported_claims") or []) + (data.get("missing") or [])
        if problems:
            advice += " Problems: " + "; ".join(str(p) for p in problems)
        _emit("critic", "review", f"score {score}/5" + (f" - {advice[:160]}" if score < PASS_SCORE else ""))
        first_advice = state["first_advice"] or (advice.strip() if state["attempts"] == 1 else "")
        return {"score": score, "advice": advice.strip(), "first_advice": first_advice}

    def after_critic(state: TeamState):
        if state["score"] >= PASS_SCORE or state["attempts"] >= MAX_ATTEMPTS:
            return "finalize"
        _emit("supervisor", "retry", "answer not good enough; sending the researcher back with feedback")
        return "researcher"

    def finalize(state: TeamState):
        final = state["draft"]
        summary = {"question": state["question"], "files": state["files"], "queries": state["queries"],
                   "first_advice": state["first_advice"]}
        if state["score"] < PASS_SCORE:
            final += (f"\n\n> Confidence: low (reviewer score {state['score']}/5). "
                      f"Could not fully verify: {state['advice'][:300]}")
        elif state["attempts"] > 1 and state["first_advice"]:
            # it needed a retry but then passed: worth remembering for next time, automatically
            store = LessonStore(repo_name)
            store.add(state["question"], lesson_from_retry(summary), "correction")
            _emit("memory", "save", "saved a correction lesson (this question needed a second look)")
        _emit("model", "info", active_model_name())  # which model actually answered, for the eval to record
        _emit("team", "answer", final)
        return {"final": final}

    g = StateGraph(TeamState)
    for name, fn in [("supervisor", supervisor), ("chat", chat), ("researcher", researcher),
                     ("explainer", explainer), ("critic", critic), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["route"], {"research": "researcher", "chat": "chat"})
    g.add_edge("chat", END)
    g.add_edge("researcher", "explainer")
    g.add_edge("explainer", "critic")
    g.add_conditional_edges("critic", after_critic, {"finalize": "finalize", "researcher": "researcher"})
    g.add_edge("finalize", END)
    return g.compile()


def run_team(repo_name: str, question: str, llm=None):
    """Runs the whole team and yields progress events (dicts with agent / type / text)."""
    if llm is None:
        from .llm import get_llm
        llm = get_llm()
    app = build_team_graph(repo_name, llm)
    start = {"question": question, "route": "", "attempts": 0, "findings": "", "evidence": [],
             "files": [], "queries": [], "draft": "", "score": 0, "advice": "", "first_advice": "",
             "final": "", "lessons_used": []}
    yield from app.stream(start, {"recursion_limit": 30}, stream_mode="custom")


def submit_feedback(repo_name: str, question: str, files: list[str], positive: bool, note: str = "") -> str:
    """Call this when the user clicks thumbs up/down on an answer (see finalize()'s summary)."""
    return record_feedback(LessonStore(repo_name), {"question": question, "files": files}, positive, note)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if not GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is missing. Put it in backend/.env")
    for ev in run_team(sys.argv[1], sys.argv[2]):
        if ev["type"] == "answer":
            print(f"\n=== ANSWER ===\n{ev['text']}")
        else:
            print(f"[{ev['agent'].upper():<10}] {ev['type']}: {ev['text']}")
