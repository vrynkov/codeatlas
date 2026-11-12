"""The ReAct loop must not resend every old tool result in full on every step."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.react_agent import trim_history


def _conversation(n_results: int, size: int = 2000) -> list:
    msgs = [HumanMessage("question")]
    for i in range(n_results):
        msgs.append(AIMessage(content="", tool_calls=[{"name": "get_file", "args": {"path": f"f{i}.py"}, "id": f"c{i}"}]))
        msgs.append(ToolMessage(content=str(i) * size, tool_call_id=f"c{i}"))
    return msgs


def test_only_the_newest_results_are_sent_in_full():
    msgs = _conversation(4)
    sent = trim_history(msgs, keep_full=2, max_chars=100)
    tools = [m for m in sent if isinstance(m, ToolMessage)]
    assert [len(t.content) > 2000 - 1 for t in tools] == [False, False, True, True]
    assert tools[0].content.startswith("0" * 100) and "2000 characters in full" in tools[0].content
    assert [t.tool_call_id for t in tools] == ["c0", "c1", "c2", "c3"]  # pairing with tool calls is kept


def test_the_state_itself_is_not_changed():
    msgs = _conversation(3)
    trim_history(msgs, keep_full=1, max_chars=50)
    assert all(len(m.content) == 2000 for m in msgs if isinstance(m, ToolMessage))


def test_trimming_really_cuts_what_gets_resent():
    """Added up over all 8 steps (each step resends the history so far), the saving is large."""
    msgs = _conversation(8)
    full = sent = 0
    for step in range(1, 9):
        history = msgs[: 1 + 2 * step]
        full += sum(len(m.content) for m in history)
        sent += sum(len(m.content) for m in trim_history(history))
    assert sent < full * 0.6  # ~43% less here; the real saving depends on result sizes


def test_short_results_and_other_messages_are_left_alone():
    msgs = _conversation(3, size=50)
    assert trim_history(msgs, keep_full=0, max_chars=400) == msgs
