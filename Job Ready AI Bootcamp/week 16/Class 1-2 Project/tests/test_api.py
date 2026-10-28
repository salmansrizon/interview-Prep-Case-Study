"""
HTTP-level tests: API contract with the FakeBackend, plus the RealBackend end to end with fakes inside.

HIGHLIGHTS: ``dependency_overrides`` দিয়ে route-এর backend বদলানো হয় — Ollama/FinBERT কিছুই লাগে না।
শেষ দুটো test আসল ``RealBackend`` (আসল SQLite, আসল Flow, আসল PDF) চালায়, শুধু LLM আর sentiment নকল।
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from api.backend import FakeBackend, RealBackend
from api.main import app, get_backend
from api.settings import Settings
from src.risk.metrics import assess
from tests.fakes import FakeLLM, FakeSentiment, brief_for, memo_for


@contextmanager
def client_for(backend):
    app.dependency_overrides[get_backend] = lambda: backend
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client():
    with client_for(FakeBackend()) as c:
        yield c


def test_health_ready_and_tickers(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["backend"] == "fake" and r.headers["x-request-id"]
    assert client.get("/v1/tickers").json()["tickers"][0]["ticker"] == "ACME"


def test_risk_contract_and_structured_404(client):
    body = client.get("/v1/risk/acme").json()                       # ticker is upper-cased
    assert body["risk_score"] == 28.1 and body["band"] == "low" and set(body["metrics"]) == {
        "volatility", "max_drawdown", "var_95", "sentiment"}
    assert body["news"][0]["id"] == "news:ACME:7" and body["request_id"]
    r = client.get("/v1/risk/ZZZZ")
    assert r.status_code == 404 and r.json()["error"] == "unknown_ticker" and r.json()["request_id"]


def test_validation_errors_are_structured(client):
    r = client.get("/v1/risk/not a ticker!")
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert client.post("/v1/analyze/ACME", params={"mode": "hierarchical"}).status_code == 422


def test_memo_404_until_analyzed_then_pdf(client):
    r = client.get("/v1/memo/ACME")
    assert r.status_code == 404 and r.json()["error"] == "memo_not_found"
    a = client.post("/v1/analyze/ACME").json()
    assert a["outcome"] == "published" and a["memo_available"] and a["risk"]["risk_score"] == 28.1
    pdf = client.get("/v1/memo/ACME")
    assert pdf.headers["content-type"] == "application/pdf" and pdf.content.startswith(b"%PDF")
    assert 'filename="ACME-memo.pdf"' in pdf.headers["content-disposition"]


def test_busy_analysis_slot_returns_503_with_retry_after(client):
    app.state.gate = asyncio.Semaphore(0)                           # every slot taken
    app.state.settings.queue_timeout_s = 0.05
    r = client.post("/v1/analyze/ACME")
    assert r.status_code == 503 and r.json()["error"] == "overloaded" and r.headers["retry-after"] == "1"


def test_real_backend_end_to_end_with_fakes(tmp_path):
    backend = RealBackend(Settings(db_path=tmp_path / "m.db", output_dir=tmp_path / "out"), classifier=FakeSentiment())
    risk, news = assess(backend.store, "MRDN", backend.classifier)
    backend.llm = FakeLLM([brief_for(news), memo_for(risk.to_dict(), news[-1]["id"])])
    with client_for(backend) as c:
        assert c.get("/ready").status_code == 200
        r = c.get("/v1/risk/mrdn").json()
        assert r["risk_score"] == risk.score and r["band"] == "low" and len(r["news"]) == 6
        a = c.post("/v1/analyze/MRDN").json()
        assert a["outcome"] == "published" and a["verification"]["ok"] and a["risk"]["risk_score"] == risk.score
        pdf = c.get("/v1/memo/MRDN")
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


def test_real_backend_without_ollama(tmp_path):
    backend = RealBackend(Settings(db_path=tmp_path / "m.db", ollama_host="http://127.0.0.1:1"),
                          classifier=FakeSentiment())
    with client_for(backend) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and "not reachable" in r.json()["detail"]
        assert c.get("/v1/risk/HELX").status_code == 200                       # deterministic path needs no LLM
        a = c.post("/v1/analyze/HELX")
        assert a.status_code == 503 and a.json()["error"] == "llm_unavailable"
        assert c.post("/v1/analyze/NOPE").json()["error"] == "unknown_ticker"   # 404 wins over 503
