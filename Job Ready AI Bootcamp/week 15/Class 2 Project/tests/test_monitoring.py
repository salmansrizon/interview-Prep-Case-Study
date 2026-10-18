"""Monitoring tests — Ollama/Docker/Langfuse/network কিছুই লাগে না। tmp SQLite (tempfile)।"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.backend import FakeBackend
from app.langfuse_obs import make_langfuse_sink
from app.main import app, get_service
from app.pricing import equivalent_cost
from app.quality import drift_report, grounding_score
from app.service import RagService
from app.usage import pop_usage, record_usage

PDF = b"%PDF-1.4 fake"


@pytest.fixture
def ctx(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("RAG_MONITOR_DB", os.path.join(d, "m.db"))
        backend = FakeBackend()
        svc = RagService(backend)
        app.dependency_overrides[get_service] = lambda: svc
        with TestClient(app) as client:
            mon = app.state.monitor
            svc.observers.append(mon.observer)
            yield client, backend, mon
        mon._db.close()  # Windows: file lock ছাড়তে
        app.dependency_overrides.clear()


def test_pricing_and_usage_capture():
    assert equivalent_cost("small-cloud", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert equivalent_cost("local", 10, 10) == 0
    record_usage({"prompt_eval_count": 12, "eval_count": 5})
    assert pop_usage() == {"prompt_tokens": 12, "completion_tokens": 5} and pop_usage() is None


def test_grounding_score():
    assert grounding_score("leave policy gives twenty days", ["the leave policy gives twenty days"], True)[0] == 1.0
    assert grounding_score("bitcoin moon rocket", ["leave policy"], True)[0] == 0.0
    assert grounding_score("This information was not found in the documents.", [], False) == (1.0, "correct_refusal")
    assert grounding_score("This information was not found in the documents.", ["x"], True)[1] == "refused_with_context"


def test_drift_report():
    base = [{"latency_s": 0.2, "quality": 0.9}] * 5
    assert drift_report(base, 5)["status"] == "insufficient_data"
    slow = base + [{"latency_s": 0.6, "quality": 0.5}] * 5
    r = drift_report(slow, 5)
    assert r["status"] == "drift" and len(r["alerts"]) == 2
    assert drift_report(base + base, 5)["status"] == "ok"


def test_ask_is_logged_and_stats(ctx):
    c, backend, mon = ctx
    c.post("/upload", files={"file": ("hr.pdf", PDF, "application/pdf")})
    assert "usage" not in c.post("/ask", json={"question": "leave days"}).json()  # API-তে leak নয়
    backend.answer_override, backend.context_override = "invented nonsense words", ["unrelated text"]
    c.post("/ask", json={"question": "second"})
    s = c.get("/stats").json()
    assert s["queries"] == 2 and s["prompt_tokens"] > 0 and s["equivalent_cost_usd"] > 0
    assert s["quality_avg"] == pytest.approx(0.25)  # একটা grounded, একটা নয়
    assert s["recent"][0]["question"] == "second" and s["drift"]["status"] == "insufficient_data"
    assert "leave days" in c.get("/dashboard").text


def test_refusal_and_error_logged(ctx):
    c, _, mon = ctx
    c.post("/ask", json={"question": "no docs yet"})  # guardrail: found=False
    assert mon.stats()["refusals"] == 1
    assert mon.stats()["errors"] == 0


def test_langfuse_disabled_without_env(monkeypatch):
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    assert make_langfuse_sink() is None


def test_langfuse_sink_with_stub_client(ctx):
    c, _, mon = ctx
    stub = MagicMock()
    mon.sinks.append(make_langfuse_sink(stub))
    c.post("/upload", files={"file": ("hr.pdf", PDF, "application/pdf")})
    c.post("/ask", json={"question": "leave"})
    kw = stub.start_observation.call_args.kwargs
    assert kw["usage_details"]["input"] > 0 and kw["as_type"] == "generation"
    stub.start_observation.return_value.score_trace.assert_called_once()
