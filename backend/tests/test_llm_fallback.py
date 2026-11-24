"""Model fallback across Groq's free-tier limits - with fake clients, so no real Groq calls."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import llm

DAILY = "Error code: 429 - rate_limit_exceeded: Limit 200000, Used 199999, tokens per day (TPD). try again in 7m30s"
MINUTE = "Error code: 429 - rate_limit_exceeded: tokens per minute (TPM). Please try again in 15.2s"


class FakeClient:
    def __init__(self, model, failures, log):
        self.model, self.failures, self.log = model, failures, log

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kw):
        self.log.append(self.model)
        err = self.failures.get(self.model)
        if err:
            raise Exception(err)
        return AIMessage(content=f"answer from {self.model}")


@pytest.fixture
def fake_groq(monkeypatch):
    """Returns (failures, log): set failures[model] = error text to make that model fail."""
    failures, log = {}, []
    monkeypatch.setattr(llm, "_make_client", lambda model, temperature: FakeClient(model, failures, log))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)  # never really wait in tests
    return failures, log


def test_daily_limit_falls_to_next_model_and_is_remembered_by_name(fake_groq):
    failures, log = fake_groq
    a, b, _ = llm.MODEL_CHAIN
    failures[a] = DAILY
    out = llm.get_llm().invoke([HumanMessage("hi")])
    assert out.content == f"answer from {b}"
    assert a in llm._STATE["dead"]
    assert json.loads(llm._STATE_FILE.read_text())["dead"] == [a]   # saved by NAME, not list position
    log.clear()
    llm.get_llm().invoke([HumanMessage("again")])
    assert log == [b]  # the dead model isn't even tried again today


def test_critic_chain_skips_the_same_dead_model(fake_groq):
    """The writer and critic chains are in a different ORDER - a dead model must be skipped by both."""
    failures, log = fake_groq
    llm._STATE["dead"].add(llm.CRITIC_MODEL_CHAIN[0])
    out = llm.get_critic_llm().invoke([HumanMessage("grade this")])
    assert log[0] == llm.CRITIC_MODEL_CHAIN[1]
    assert out.content == f"answer from {llm.CRITIC_MODEL_CHAIN[1]}"


def test_short_limit_reroutes_without_marking_dead(fake_groq):
    failures, log = fake_groq
    a, b, _ = llm.MODEL_CHAIN
    failures[a] = MINUTE  # 15s wait > MIN_WAIT_TO_REROUTE: worth trying another model now
    assert llm.get_llm().invoke([HumanMessage("hi")]).content == f"answer from {b}"
    assert llm._STATE["dead"] == set()  # next question should still try the best model first


def test_real_errors_are_not_swallowed(fake_groq):
    failures, _ = fake_groq
    failures[llm.MODEL_CHAIN[0]] = "Error code: 401 - invalid api key"
    with pytest.raises(Exception, match="401"):
        llm.get_llm().invoke([HumanMessage("hi")])


def test_all_models_dead_gives_a_clear_message(fake_groq):
    llm._STATE["dead"].update(llm.MODEL_CHAIN)
    with pytest.raises(RuntimeError, match="exhausted for today"):
        llm.get_llm().invoke([HumanMessage("hi")])


def test_default_critic_starts_on_a_different_model_than_the_writer():
    assert llm.CRITIC_MODEL_CHAIN[0] != llm.MODEL_CHAIN[0]
    assert set(llm.CRITIC_MODEL_CHAIN) == set(llm.MODEL_CHAIN)


def test_critic_calls_never_change_which_model_is_reported_as_the_answerer(fake_groq):
    before = llm.active_model_name()
    critic = llm.get_critic_llm()
    critic.invoke([HumanMessage("grade")])
    assert critic.last_model == llm.CRITIC_MODEL_CHAIN[0]
    assert llm.active_model_name() == before


def test_old_state_file_with_list_positions_is_ignored(tmp_path, monkeypatch):
    f = tmp_path / "state.json"
    f.write_text(json.dumps({"date": llm._today(), "active": 1, "dead": [0]}))
    monkeypatch.setattr(llm, "_STATE_FILE", f)
    assert llm._load_state() == {"dead": set()}


def test_avoiding_moves_the_writers_model_to_the_end():
    critic = llm.MultiModelChatGroq(models=["a", "b", "c"], report=False)
    assert critic.avoiding("a").models == ["b", "c", "a"]
    assert critic.avoiding("unknown").models == ["a", "b", "c"]
    assert critic.avoiding(None) is critic
    assert critic.avoiding("b").report is False  # still never reported as "who answered"


def test_critic_avoids_the_writer_even_after_the_writer_fell_back(fake_groq):
    failures, log = fake_groq
    first, second, last = llm.MODEL_CHAIN
    failures[first] = failures[second] = MINUTE  # the writer gets pushed to the last model...
    llm.get_llm().invoke([HumanMessage("write")])
    assert llm.active_model_name() == last
    failures.clear()
    assert llm.CRITIC_MODEL_CHAIN[0] == last     # ...which is exactly where the critic would start
    critic = llm.get_critic_llm().avoiding(llm.active_model_name())
    critic.invoke([HumanMessage("grade")])
    assert critic.last_model != last             # so it grades with a different model instead


def test_a_smaller_writer_is_graded_by_the_strongest_model():
    strongest, middle, small = llm.MODEL_CHAIN
    critic = llm.get_critic_llm()
    assert critic.avoiding(small).models[0] == strongest   # small model wrote -> big model grades
    assert critic.avoiding(middle).models[0] == strongest
    assert critic.avoiding(strongest).models[0] != strongest  # big model wrote -> never self-graded
    assert critic.avoiding(strongest).models[-1] == strongest


TOO_LARGE = ("Error code: 429 - Request too large for model `qwen` on output tokens per minute (OTPM): "
             "Limit 1000, Requested 2048. rate_limit_exceeded")
SHORT_21S = "Error code: 429 - rate_limit_exceeded: tokens per minute (TPM). Please try again in 21s"


def test_too_large_on_the_last_model_waits_for_the_skipped_one(fake_groq, monkeypatch):
    """The 2026-11-20 crash: the first model was used up for the day, the second said "wait 21s",
    the third refused the Explainer's request as too large. It must wait for the second, not crash."""
    failures, log = fake_groq
    a, b, c = llm.MODEL_CHAIN
    llm._STATE["dead"].add(a)
    failures[b], failures[c] = SHORT_21S, TOO_LARGE
    waits = []

    def fake_sleep(seconds):
        waits.append(seconds)
        failures.pop(b, None)  # after the wait, b's per-minute bucket has room again
    monkeypatch.setattr(llm.time, "sleep", fake_sleep)

    out = llm.get_llm().invoke([HumanMessage("explain")])
    assert out.content == f"answer from {b}"
    assert log == [b, c, b] and waits == [21]


def test_goes_back_only_once_then_gives_the_clear_error(fake_groq):
    failures, log = fake_groq
    a, b, c = llm.MODEL_CHAIN
    llm._STATE["dead"].add(a)
    failures[b], failures[c] = SHORT_21S, TOO_LARGE  # b never recovers in this test
    with pytest.raises(RuntimeError, match="couldn't handle this specific request"):
        llm.get_llm().invoke([HumanMessage("explain")])
    assert log == [b, c, b, c]  # one trip back, no endless loop


def test_a_too_large_model_is_not_waited_for(fake_groq):
    """"Too large" never clears by waiting, so a model that said it is never the one we go back to."""
    failures, log = fake_groq
    a, b, c = llm.MODEL_CHAIN
    failures[a], failures[b], failures[c] = TOO_LARGE, TOO_LARGE, TOO_LARGE
    with pytest.raises(RuntimeError, match="couldn't handle this specific request") as err:
        llm.get_llm().invoke([HumanMessage("explain")])
    assert log == [a, b, c]
    # Our "likely too large" is a guess: Groq's own words must be in the error, so a run can be diagnosed.
    assert "Groq said: " + TOO_LARGE[:40] in str(err.value)


def test_a_per_minute_wait_just_over_30s_is_waited_not_a_crash(monkeypatch):
    # Real eval 2026-11-20: "TPM: Limit 8000 ... Please try again in 31.3125s" crashed the question,
    # because anything over 30s was treated as a long limit. A per-minute limit always clears in a minute.
    from langchain_core.messages import AIMessage, HumanMessage
    from app.agents import react_agent
    slept = []
    monkeypatch.setattr(react_agent.time, "sleep", lambda s: slept.append(s))

    class Throttled:
        calls = 0
        def invoke(self, messages):
            Throttled.calls += 1
            if Throttled.calls == 1:
                raise Exception("Error code: 429 - rate_limit_exceeded: Rate limit reached for model "
                                "`openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000, Used 6349, "
                                "Requested 5826. Please try again in 31.3125s.")
            return AIMessage(content="ok")

    assert react_agent.invoke_with_recovery(Throttled(), [HumanMessage("q")]).content == "ok"
    assert len(slept) == 1 and 31 <= slept[0] <= 65
