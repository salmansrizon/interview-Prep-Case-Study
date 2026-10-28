"""
HTTP contract with the fake backend: lifespan auto-ingest, /ready, routes over HTTP, structured errors, privacy.

HIGHLIGHTS: ``TestClient(app)`` আসল lifespan চালায় — fake backend-এ temp Chroma তৈরি হয় আর
``HC_AUTO_INGEST`` handbook index করে, ঠিক যেমন Week 18-এর নতুন container-এ হবে।
"""
import pytest
from fastapi.testclient import TestClient

from conftest import ScriptedLLM
from src.api.main import app
from src.guardrails import red_flags


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ready_and_documents_after_auto_ingest(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["backend"] == "fake" and r.json()["indexed_sections"] >= 100
    pages = client.get("/v1/documents").json()["pages"]
    assert len(pages) == 30 and {"file", "page", "sections"} == set(pages[0])


def test_ask_answered_contract(client):
    r = client.post("/v1/ask", json={"question": "How do I prepare an ORS packet?"}, headers={"X-Request-ID": "t-1"})
    body = r.json()
    assert r.status_code == 200 and r.headers["x-request-id"] == "t-1" and body["request_id"] == "t-1"
    assert body["route"] == "answered" and body["citations"][0]["n"] == 1 and body["citations"][0]["id"].startswith("kb:")
    assert body["disclaimer"].startswith("Educational, fictional") and body["redaction_count"] == 0
    assert "total_ms" in body["timings_ms"]


def test_emergency_with_pii_over_http(client):
    body = client.post("/v1/ask", json={"question": "My number is 01712-345678 and my chest pain is getting worse"}).json()
    assert body["route"] == "escalated" and body["rule"] == "chest_pain" and body["answer"] == red_flags.EMERGENCY_MESSAGE
    assert body["redaction_count"] == 1 and "[PHONE]" in body["redacted_question"] and body["citations"] == []
    assert "01712" not in str(body)                                         # raw PII never comes back


def test_refused_and_abstained_over_http(client):
    refused = client.post("/v1/ask", json={"question": "Can I double my insulin dose?"}).json()
    assert refused["route"] == "refused" and refused["rule"] == "individual_dosing"
    abstained = client.post("/v1/ask", json={"question": "What is the capital of France?"}).json()
    assert abstained["route"] == "abstained" and abstained["citations"] == []


def test_validation_errors_are_structured_and_never_echo_input(client):
    r = client.post("/v1/ask", json={"question": ""})
    assert r.status_code == 422 and r.json()["error"] == "validation_error" and r.json()["request_id"]
    long_q = "My phone is 01712-345678. " + "x" * 1100
    r = client.post("/v1/ask", json={"question": long_q})
    assert r.status_code == 422 and "01712" not in r.text


def test_llm_down_is_503_but_emergencies_still_answer(client):
    a = client.app.state.assistant
    real_llm, a.llm = a.llm, ScriptedLLM(fail=True)
    try:
        r = client.post("/v1/ask", json={"question": "How do I prepare an ORS packet?"})
        assert r.status_code == 503 and r.json()["error"] == "llm_unavailable"
        assert client.post("/v1/ask", json={"question": "She is unconscious"}).json()["route"] == "escalated"
        assert client.get("/ready").status_code == 503
    finally:
        a.llm = real_llm


def test_reingest_endpoint_is_idempotent(client):
    body = client.post("/v1/ingest").json()
    assert body["pages"] == 30 and body["removed"] == 0
