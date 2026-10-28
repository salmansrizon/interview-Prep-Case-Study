"""
The assistant pipeline: KB loading, ingestion sync, guardrail ORDER, abstention, citation check, Chroma client choice.

HIGHLIGHTS: সবচেয়ে গুরুত্বপূর্ণ test হলো ক্রম — PII + emergency একসাথে থাকলে আগে redaction, তারপর
escalation, আর retrieval বা LLM একবারও ডাকা হয় না। ``ScriptedLLM.calls`` আর ``CountingEmbedder.calls``
দিয়ে প্রমাণ করা হয় কোন ধাপ চলেনি।
"""
import logging
import shutil
from unittest.mock import MagicMock

import pytest

from conftest import KB_DIR, ScriptedLLM
from src.assistant import HealthcareAssistant
from src.backend import HashEmbedder, build_assistant
from src.guardrails import red_flags, scope
from src.guardrails.grounding import NOT_IN_KB
from src.rag.components import ChromaStore, LLMUnavailable
from src.rag.kb import DISCLAIMER, load_kb
from src.settings import Settings


# ── knowledge base ────────────────────────────────────────────────────
def test_kb_sections_have_stable_ids_and_stay_short():
    sections = load_kb(KB_DIR)
    assert len({s.file for s in sections}) == 30 and len(sections) >= 100
    assert len({s.id for s in sections}) == len(sections)                     # unique ids
    assert sections[0].id.startswith("kb:") and all(s.id.count(":") == 2 for s in sections)
    assert max(len(s.text) for s in sections) <= 500                          # authoring rule: short sections
    assert all(DISCLAIMER in p.read_text(encoding="utf-8") for p in KB_DIR.glob("*.md"))
    assert all(DISCLAIMER not in s.text for s in sections)                    # header is not indexed


def test_ingest_is_idempotent_and_removes_stale_sections(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    for name in ("fever.md", "dehydration.md"):
        shutil.copy(KB_DIR / name, kb / name)
    a = HealthcareAssistant(HashEmbedder(), ChromaStore(str(tmp_path / "db"), "sync_test"), None)
    first, again = a.ingest(kb), a.ingest(kb)
    assert first.sections == again.sections == a.store.count() and again.removed == 0
    (kb / "fever.md").unlink()                                                 # page withdrawn from the handbook
    third = a.ingest(kb)
    assert third.removed == 5 and a.store.count() == third.sections == 4
    assert [p["file"] for p in a.store.pages()] == ["dehydration.md"]


# ── guardrail order ───────────────────────────────────────────────────
def test_emergency_with_pii_is_redacted_then_escalated_without_retrieval_or_llm(assistant, caplog):
    with caplog.at_level(logging.INFO, logger="healthcare.assistant"):
        r = assistant.ask("I'm calling from +880 1812 345678, I have chest pain right now")
    assert r.route == "escalated" and r.rule == "chest_pain" and r.answer == red_flags.EMERGENCY_MESSAGE
    assert r.redaction.counts == {"PHONE": 1} and "[PHONE]" in r.redaction.text
    assert assistant.embedder.calls == 0 and assistant.llm.calls == 0 and r.retrieved == [] and r.citations == []
    assert "1812" not in caplog.text and "[PHONE]" in caplog.text               # logs only see redacted text


def test_llm_only_ever_sees_redacted_question(assistant):
    r = assistant.ask("My email is rahim@example.com. How do I prepare an ORS packet?")
    assert r.route == "answered" and r.redaction.counts == {"EMAIL": 1}
    assert "rahim@example.com" not in assistant.llm.prompts[0] and "[EMAIL]" in assistant.llm.prompts[0]


def test_refusal_offers_related_handbook_sections_without_llm(assistant):
    r = assistant.ask("Can I take ibuprofen with my blood pressure medicine?")
    assert r.route == "refused" and r.rule == "individual_dosing" and assistant.llm.calls == 0
    assert r.answer == scope.REFUSAL_MESSAGE + scope.RELATED_NOTE
    assert 1 <= len(r.citations) <= 2 and [n for n, _ in r.citations] == list(range(1, len(r.citations) + 1))
    none_related = assistant.ask("How many tablets of xyzzol should I take?")      # nothing within the threshold
    assert none_related.answer == scope.REFUSAL_MESSAGE and none_related.citations == []


def test_abstains_below_threshold_without_calling_llm(assistant):
    assistant.max_distance = 0.01
    r = assistant.ask("How do I prepare an ORS packet?")
    assert r.route == "abstained" and r.rule == "below_threshold" and r.answer == NOT_IN_KB
    assert assistant.llm.calls == 0 and r.retrieved                       # retrieval ran; its distances are kept


def test_answered_with_verified_citations_and_timings(assistant):
    r = assistant.ask("How do I prepare an ORS packet?")
    assert r.route == "answered" and r.answer.endswith("[1].") and assistant.llm.calls == 1
    assert [n for n, _ in r.citations] == [1] and r.citations[0][1].id.startswith("kb:dehydration:")
    assert {"redact_ms", "rules_ms", "embed_ms", "search_ms", "llm_ms", "total_ms"} <= set(r.timings_ms)


@pytest.mark.parametrize("reply,rule", [("ORS replaces water and salts.", "no_valid_citation"),
                                        ("ORS replaces water and salts [9].", "no_valid_citation"),
                                        ("NOT_IN_KB", "model_abstained")])
def test_uncited_or_invented_citation_answers_become_abstentions(assistant, reply, rule):
    assistant.llm = ScriptedLLM(reply)
    r = assistant.ask("How do I prepare an ORS packet?")
    assert r.route == "abstained" and r.rule == rule and r.answer == NOT_IN_KB and r.citations == []


def test_invented_citation_is_removed_and_flagged(assistant):
    assistant.llm = ScriptedLLM("Use the water amount on the packet [1]. Never add sugar [8].")
    r = assistant.ask("How do I prepare an ORS packet?")
    assert r.route == "answered" and "[8]" not in r.answer and r.flags == ["invalid_citation_removed"]


def test_llm_down_breaks_answers_but_not_safety_routes(assistant):
    assistant.llm = ScriptedLLM(fail=True)
    with pytest.raises(LLMUnavailable):
        assistant.ask("How do I prepare an ORS packet?")
    assert assistant.ask("My son swallowed bleach").route == "escalated"
    assert assistant.ask("Do I have diabetes?").route == "refused"
    assert assistant.ask("What is the capital of France?").route == "abstained"


def test_prompt_numbers_passages_and_puts_question_last(assistant):
    hits = assistant.retrieve("ORS packet water")
    p = HealthcareAssistant.prompt("q?", hits)
    assert p.startswith("PASSAGES:\n[1] (") and p.endswith("QUESTION: q?")


# ── Chroma: local folder vs server (Week 18) ─────────────────────────
def _patched_chromadb(monkeypatch):
    import chromadb
    persistent, http = MagicMock(name="PersistentClient"), MagicMock(name="HttpClient")
    monkeypatch.setattr(chromadb, "PersistentClient", persistent)
    monkeypatch.setattr(chromadb, "HttpClient", http)
    return persistent, http


def test_chroma_uses_persistent_client_when_host_empty(monkeypatch, tmp_path):
    persistent, http = _patched_chromadb(monkeypatch)
    ChromaStore(str(tmp_path), "kb", host="").client()
    assert persistent.call_args.kwargs["path"] == str(tmp_path) and not http.called
    assert persistent.call_args.kwargs["settings"].anonymized_telemetry is False


def test_chroma_uses_http_client_when_host_set_via_env(monkeypatch):
    persistent, http = _patched_chromadb(monkeypatch)
    for k, v in {"HC_BACKEND": "real", "HC_CHROMA_HOST": "chroma", "HC_CHROMA_PORT": "8001",
                 "HC_OLLAMA_HOST": "http://ollama:11434", "HC_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"}.items():
        monkeypatch.setenv(k, v)
    a = build_assistant(Settings())                       # real backend: everything lazy, nothing loads here
    a.store.client()
    kw = http.call_args.kwargs
    assert (kw["host"], kw["port"], kw["settings"].anonymized_telemetry) == ("chroma", 8001, False)
    assert not persistent.called
    assert a.llm.host == "http://ollama:11434" and a.embedder.model_name.endswith("all-MiniLM-L6-v2")
