"""A ReAct agent built by hand with LangGraph, so you can see every moving part.

ReAct = Reason + Act. The loop:  think -> call a tool -> read the result -> think again -> ... -> answer.

Run from the backend/ folder:
    python -m app.agents.react_agent requests "How are retries handled?"
"""
import json
import sys
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ..config import GROQ_API_KEY, MAX_AGENT_STEPS, MAX_EVIDENCE_CHARS
from .rate_limits import is_connection_error, is_daily_limit, is_rate_limit, suggested_wait
from ..tools.repo_tools import build_tools

SYSTEM_PROMPT = """You are CodeAtlas, an expert software engineer who answers questions about ONE code repository.

Work like this (ReAct):
1. THINK about what you need to find out.
2. Use a tool: search_code to find relevant places, get_file to read code in full.
3. search_code only shows short previews. Use get_file to read the real code before you answer.
   If a search was not useful, NEVER repeat it: change the words, change the folder, or read a file instead.
4. When you have enough evidence, write the answer.

Tools (use ONLY these two, with exactly these parameter names):
- search_code(query, path)            -> `path` is optional, e.g. "src/requests/"
- get_file(path, start_line, end_line)
Never invent other tools (there is no repo_browser, open_file, etc.).

Rules:
- Base every claim on code you actually saw. Never guess.
- Cite sources as `path/to/file.py:START-END`.
- If you cannot find the answer, say so honestly and explain what you tried.
- Keep answers clear and short. Explain like a friendly senior colleague."""


TOOL_NUDGE = ("Your last tool call was invalid. Use only search_code(query, path) "
              "and get_file(path, start_line, end_line), with exactly those parameter names.")
NO_TOOLS_NUDGE = "Do not call any tools. Reply with plain text only."


MAX_AUTO_WAIT = 30  # seconds. Groq's per-MINUTE limits clear fast, so a short wait is worth it.
                    # A per-DAY limit ("TPD") can say "try again in 24m" - waiting that long inside a
                    # loop would freeze the whole run, so we fail fast instead. NOTE: llm.get_llm()
                    # already retries a daily-limit error on a DIFFERENT model automatically, so by
                    # the time we see one here, every configured model has already been exhausted.


def invoke_with_recovery(llm_with_tools, messages, tries: int = 3, nudge: str = TOOL_NUDGE):
    """Free-tier models sometimes fail in predictable ways. Instead of crashing, we recover:
    - the model invents a tool that doesn't exist -> tell it so and let it try again
    - a short rate limit (per-minute)               -> wait, then try again
    - a long rate limit (per-day quota, ALL configured models exhausted) -> stop right away with
      a clear message, instead of silently retrying for a minute and then failing anyway"""
    extra = []
    for attempt in range(tries):
        try:
            return llm_with_tools.invoke(messages + extra)
        except Exception as e:  # noqa: BLE001
            text = str(e)
            last = attempt == tries - 1
            if "tool_use_failed" in text and not last:
                print("[RECOVER] model called a tool that doesn't exist; asking it to retry", file=sys.stderr)
                extra = [HumanMessage(nudge)]
                continue
            if is_rate_limit(text):
                wait = suggested_wait(text)
                daily = is_daily_limit(text)
                if daily or (wait is not None and wait > MAX_AUTO_WAIT):
                    mins = f"{wait / 60:.1f} min" if wait else "a while"
                    kind = "daily (all configured models exhausted)" if daily else "long (non-daily)"
                    raise RuntimeError(f"Groq {kind} rate limit hit; try again in ~{mins}. "
                                       f"[Groq said: {text[:400]}]") from e
                if not last:
                    print(f"[RECOVER] rate limit hit; waiting {wait or 20:.0f} seconds", file=sys.stderr)
                    time.sleep(wait or 20)
                    continue
            if is_connection_error(text) and not last:
                print("[RECOVER] connection hiccup reaching Groq; retrying shortly", file=sys.stderr)
                time.sleep(3)
                continue
            raise


def _key(tc) -> tuple:
    """A comparable fingerprint of a tool call (empty arguments are ignored)."""
    args = {k: v for k, v in tc["args"].items() if v not in ("", None)}
    return tc["name"], json.dumps(args, sort_keys=True)


def _seen_calls(messages) -> set:
    """All tool calls the agent has already made."""
    return {_key(tc) for m in messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}


def _is_repeat(reply, messages) -> bool:
    seen = _seen_calls(messages)
    return any(_key(tc) in seen for tc in (reply.tool_calls or []))


FORCED_SYSTEM = """You are CodeAtlas, an expert software engineer. You have run out of research steps.
Using ONLY the EVIDENCE below, write your best answer to the QUESTION now, in plain text.
Cite sources as `path/to/file.py:START-END`. Say clearly what you could not confirm.
Do not call any tools and do not ask for more steps."""


def force_answer(llm, messages):
    """Last resort. We give the model a fresh, tool-free prompt with the evidence collected so far.
    (Sending the old tool-call history without tools makes some models try to call tools anyway.)"""
    question = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    evidence = "\n\n".join(m.content for m in messages if isinstance(m, ToolMessage))[-MAX_EVIDENCE_CHARS:]
    prompt = [SystemMessage(FORCED_SYSTEM), HumanMessage(f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}")]
    try:
        return invoke_with_recovery(llm, prompt, nudge=NO_TOOLS_NUDGE)
    except Exception as e:  # noqa: BLE001
        print(f"[LIMIT]   could not write a final answer ({type(e).__name__})", file=sys.stderr)
        return AIMessage(content="(The researcher ran out of steps and could not summarize; "
                                 "see the evidence it gathered.)")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # the conversation so far; new messages get appended


def build_graph(repo_name: str, llm=None, system_prompt: str = SYSTEM_PROMPT):
    tools = build_tools(repo_name)
    if llm is None:
        from .llm import get_llm

        llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)  # tells the model which tools exist

    def agent(state: AgentState):
        """The 'Reason' step: the model looks at the conversation and decides: call a tool, or answer."""
        msgs = [SystemMessage(system_prompt)] + state["messages"]
        steps_used = sum(isinstance(m, AIMessage) for m in state["messages"])

        # Out of steps? Don't crash: ask for the best possible answer, with tools switched off.
        if steps_used >= MAX_AGENT_STEPS - 1:
            print("[LIMIT]   step limit reached; asking for a final answer", file=sys.stderr)
            return {"messages": [force_answer(llm, state["messages"])]}

        reply = invoke_with_recovery(llm_with_tools, msgs)

        # Same tool call as before? That is a loop. Nudge once, then force an answer.
        if reply.tool_calls and _is_repeat(reply, state["messages"]):
            print("[NUDGE]   repeated call blocked; asking the model to try something different", file=sys.stderr)
            nudge = HumanMessage("You already ran that exact call and have its result above. Do NOT repeat it. "
                                 "Either read code with get_file, search with different words, or answer now.")
            reply = invoke_with_recovery(llm_with_tools, msgs + [nudge])
            if reply.tool_calls and _is_repeat(reply, state["messages"]):
                print("[LIMIT]   still looping; asking for a final answer", file=sys.stderr)
                reply = force_answer(llm, state["messages"])
        return {"messages": [reply]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))  # the 'Act' step: runs whichever tool the model asked for
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool requested? -> "tools", otherwise -> END
    graph.add_edge("tools", "agent")  # after acting, go back to reasoning
    return graph.compile()


def ask(repo_name: str, question: str, llm=None):
    """Runs the agent and yields each step so we can print (later: stream to the UI) the trace."""
    app = build_graph(repo_name, llm)
    config = {"recursion_limit": MAX_AGENT_STEPS * 2 + 1}
    for update in app.stream(
        {"messages": [HumanMessage(question)]}, config, stream_mode="updates"
    ):
        for node, data in update.items():
            for msg in data["messages"]:
                yield node, msg


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # avoids Windows console encoding errors
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if not GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is missing. Put it in backend/.env (see README).")
    repo, question = sys.argv[1], sys.argv[2]
    try:
        for node, msg in ask(repo, question):
            if node == "agent" and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    print(f"\n[ACT]     {tc['name']}({tc['args']})")
            elif node == "tools":
                print(f"[OBSERVE] got {len(msg.content)} characters back")
            else:
                print(f"\n=== ANSWER ===\n{msg.content}")
    except GraphRecursionError:
        print("\nThe agent used all its steps without finishing. Try a more specific question.")
