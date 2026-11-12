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
