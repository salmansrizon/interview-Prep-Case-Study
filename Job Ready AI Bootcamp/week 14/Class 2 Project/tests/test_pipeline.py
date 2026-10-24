"""
Offline tests for the multi-agent research team. Run: pytest tests -q

HIGHLIGHTS: কোনো network বা Ollama লাগে না (ADR 0006)। একটা scripted fake
``BaseLLM`` CrewAI-র ReAct format-এ ("Thought: ...\\nFinal Answer: ...") উত্তর দেয়
— তাই পুরো crew, guardrail retry আর Flow-এর router সত্যিই চলে, শুধু model নকল।
"""
from __future__ import annotations

import config  # noqa: F401 - telemetry env vars before crewai
import pytest
from crewai import Process
from crewai.llms.base_llm import BaseLLM

from config import get_config
from src.crew.team import MODES, build_crew, report_guardrail, research_guardrail
from src.flow.research_flow import run_research
from src.llm.factory import build_llm, ollama_status
from src.report import save_report
from src.tools.corpus_search import Corpus, CorpusSearchTool, _cached_corpus, tokens
from src.verify.citations import cited_ids, sentences, support, verify

CORPUS_DIR = str(get_config().data.corpus_dir)
CORPUS = _cached_corpus(CORPUS_DIR)


class FakeLLM(BaseLLM):
    """Replays scripted answers in CrewAI's ReAct text format; records how often it was called."""

    def __init__(self, answers, **kw):
        super().__init__(model="fake", **kw)
        self._answers = list(answers)
        self.calls = 0

    def call(self, messages, *args, **kwargs):
        self.calls += 1
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        return f"Thought: I now know the final answer\nFinal Answer: {answer}"

    def supports_function_calling(self) -> bool:
        return False


NOTES = ("- MAST grouped 14 failure modes into three categories [multi_agent_systems:4]\n"
         "- Adding a verification step was one of the most effective fixes [multi_agent_systems:4]\n"
         "- Independent agents amplified errors about 17 times [multi_agent_systems:3]\n"
         "- Agent loops need a maximum number of steps and repeat detection [react_agents:3]")
REPORT = ("# Why Multi-Agent Systems Fail\n\n## Failure modes\n"
          "The MAST taxonomy grouped 14 failure modes into system design, inter-agent misalignment and task "
          "verification [multi_agent_systems:4]. Independent agents amplified errors about 17 times versus about "
          "4 times with a central orchestrator [multi_agent_systems:3].\n\n## Fixes\n"
          "Adding a verification step was one of the most effective fixes [multi_agent_systems:4]. Agent loops "
          "need hard stop conditions such as a maximum number of steps [react_agents:3].")


# ── corpus + BM25 tool ────────────────────────────────────────────────
def test_corpus_has_stable_section_ids():
    assert len(CORPUS.passages) >= 30
    assert CORPUS.by_id["rag:3"].title == "Guardrails"
    assert all(":" in p.id for p in CORPUS.passages)


def test_bm25_ranks_relevant_passage_first():
    top, score = CORPUS.search("MAST failure taxonomy traces", k=3)[0]
    assert top.id == "multi_agent_systems:4" and score > 0
    assert CORPUS.search("zebra giraffe", k=3) == []


def test_tokens_drop_stopwords():
    assert tokens("The cost of the RAG pipeline") == ["cost", "rag", "pipeline"]


def test_tool_output_has_citation_ids():
    tool = CorpusSearchTool(corpus_dir=CORPUS_DIR, top_k=2)
    out = tool._run("multi-agent failure verification")
    assert out.startswith("[multi_agent_systems:") and out.count("\n\n[") == 1   # exactly top_k passages
    assert tool._run("zebra giraffe").startswith("NO_RESULTS")


def test_corpus_from_tmp_dir(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n## One\nsolar storage costs fell\n\n## Two\nwind is cheap\n")
    c = Corpus(tmp_path)
    assert [p.id for p in c.passages] == ["a:1", "a:2"] and c.search("solar")[0][0].id == "a:1"


# ── deterministic verification ────────────────────────────────────────
def test_cited_ids_and_sentences():
    assert cited_ids("x [rag:1] y [tool_calling:3]") == ["rag:1", "tool_calling:3"]
    assert sentences("# Title\n\n## Sec\nFirst real sentence here now. Second one is here too.") == [
        "First real sentence here now.", "Second one is here too."]


def test_support_score():
    p = CORPUS.by_id["react_agents:2"].text
    assert support("none of ReAct's failures were hallucinations [react_agents:2]", [p]) >= 0.8
    assert support("ReAct was invented by Google in 1999 for robots [react_agents:2]", [p]) < 0.45


def test_verify_flags_invalid_weak_uncited_and_extra_ids():
    text = ("Independent agents amplified errors about 17 times [multi_agent_systems:3]. "
            "Bananas are a great source of potassium for athletes [multi_agent_systems:3]. "
            "A sentence without any citation at all here. "
            "Ollama default temperature is 0.8 [local_llms_ollama:2]. "
            "Fake fact from a fake place [rag:15].")
    v = verify(text, CORPUS, notes=NOTES)
    s = v.summary()
    assert s["supported"] == 2 and s["weak"] == 1 and s["uncited"] == 1
    assert s["invalid_ids"] == ["rag:15"] and s["not_in_notes"] == ["local_llms_ollama:2"]
    assert "do not exist" in v.feedback() and "not supported" in v.feedback()


def test_verify_good_report():
    s = verify(REPORT, CORPUS, notes=NOTES).summary()
    assert s["citation_precision"] == 1.0 and not s["invalid_ids"] and not s["not_in_notes"]


# ── guardrails ────────────────────────────────────────────────────────
class Out:
    def __init__(self, raw):
        self.raw = raw


def test_research_guardrail():
    check = research_guardrail(CORPUS_DIR)
    assert check(Out(NOTES)) == (True, NOTES)
    ok, msg = check(Out(NOTES.replace("[react_agents:3]", "[rag:15]")))
    assert not ok and "rag:15" in msg
    ok, msg = check(Out("- only one fact [rag:1]"))
    assert not ok and "at least 4 bullet points" in msg


def test_report_guardrail_checks_format_ids_and_notes():
    class NotesTask:
        output = Out(NOTES)

    check = report_guardrail(CORPUS_DIR, NotesTask())
    assert check(Out(REPORT))[0]
    assert "two '## Section'" in check(Out("no headings [rag:1] [rag:2] [rag:3] [rag:4]"))[1]
    extra = REPORT + "\nOllama default temperature is 0.8 [local_llms_ollama:2]."
    assert "not in the research notes" in check(Out(extra))[1]


# ── crew wiring ───────────────────────────────────────────────────────
def test_crew_wiring_grounded_vs_tool_mode():
    c = build_crew("sequential", llm=FakeLLM(["x"]), passages="[rag:1] text")
    assert c.process == Process.sequential and len(c.tasks) == 3
    assert c.tasks[1].context == [c.tasks[0]] and c.tasks[2].context == c.tasks[:2]
    assert c.agents[0].tools == [] and "[rag:1] text" in c.tasks[0].description
    t = build_crew("tool", llm=FakeLLM(["x"]))
    assert [tool.name for tool in t.agents[0].tools] == ["search_corpus"]
    assert len(build_crew("single", llm=FakeLLM(["x"])).tasks) == 1
    assert build_crew("hierarchical", llm=FakeLLM(["x"])).process == Process.hierarchical
    with pytest.raises(ValueError):
        build_crew("nope")
    assert set(MODES) == {"sequential", "single", "tool", "hierarchical"}


def test_braces_in_passages_are_escaped():
    c = build_crew("sequential", llm=FakeLLM(["x"]), passages="use {topic} literally")
    assert "{topic} literally" not in c.tasks[0].description


def test_full_crew_runs_offline_with_fake_llm():
    llm = FakeLLM([NOTES, REPORT, REPORT])
    out = build_crew("sequential", llm=llm, passages="[multi_agent_systems:4] ...").kickoff(inputs={"topic": "x"})
    assert out.raw.startswith("# Why Multi-Agent") and len(out.tasks_output) == 3 and llm.calls == 3


def test_guardrail_retry_then_success():
    bad = "- one fact [rag:1]"
    llm = FakeLLM([bad, NOTES, REPORT, REPORT])
    out = build_crew("sequential", llm=llm, passages="...").kickoff(inputs={"topic": "x"})
    assert out.tasks_output[0].raw == NOTES and llm.calls == 4


# ── flow ──────────────────────────────────────────────────────────────
def test_flow_publishes_verified_report():
    state = run_research("Why multi-agent systems fail", llm=FakeLLM([NOTES, REPORT, REPORT]))
    assert state.outcome == "published" and state.rounds == 1
    assert state.verification["citation_precision"] == 1.0
    assert "multi_agent_systems:4" in state.passages


def test_flow_escalates_when_verification_keeps_failing():
    unsupported = ("# Report\n\n## A\nBananas cure every bug in distributed robots on Mars [multi_agent_systems:4]. "
                   "Purple elephants write better code than people [multi_agent_systems:3].\n\n## B\n"
                   "Quantum toasters remove all hallucinations instantly [multi_agent_systems:4]. "
                   "Agent loops need a maximum number of steps [react_agents:3].")
    state = run_research("x", llm=FakeLLM([NOTES, unsupported, unsupported]))
    assert state.outcome == "needs_human" and state.rounds == get_config().crew.flow_max_rounds


def test_flow_survives_guardrail_exhaustion():
    state = run_research("x", llm=FakeLLM(["- one fact [rag:1]"]))
    assert state.outcome == "needs_human" and "error" in state.verification


# ── misc ──────────────────────────────────────────────────────────────
def test_llm_factory_and_status():
    llm = build_llm()
    assert llm.model == "llama3.1:8b" and llm.temperature == 0.1
    ok, msg = ollama_status(base_url="http://127.0.0.1:1")
    assert not ok and "not reachable" in msg


def test_save_report(tmp_path):
    p = save_report("# Hi", "My Topic!", tmp_path)
    assert p.read_text(encoding="utf-8") == "# Hi\n" and p.name.startswith("my-topic-")
