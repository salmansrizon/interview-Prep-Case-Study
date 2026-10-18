"""API tests — Ollama/Chroma/Week 12 কিছুই লাগে না। FakeBackend + dependency_overrides।"""

import pytest
from fastapi.testclient import TestClient

from app.backend import FakeBackend
from app.errors import LLMUnavailable
from app.main import app, get_service
from app.service import RagService

PDF = b"%PDF-1.4 fake"


@pytest.fixture
def ctx():
    svc = RagService(FakeBackend())
    records = []
    svc.observers.append(records.append)  # Class 30 seam এভাবেই কাজ করবে
    app.dependency_overrides[get_service] = lambda: svc
    with TestClient(app) as client:  # with = lifespan চলে
        yield client, svc, records
    app.dependency_overrides.clear()


def test_health_and_docs(ctx):
    c, *_ = ctx
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/docs").status_code == 200
    assert "/ask" in c.get("/openapi.json").json()["paths"]


def test_upload_then_ask_with_citation(ctx):
    c, _, records = ctx
    r = c.post("/upload", files={"file": ("hr.pdf", PDF, "application/pdf")})
    assert r.status_code == 201 and r.json()["chunks_indexed"] == 3
    r = c.post("/ask", json={"question": "leave?"})
    body = r.json()
    assert body["found"] and body["citations"][0]["source"] == "hr.pdf"
    assert body["request_id"] and r.headers["X-Request-ID"] == body["request_id"]
    assert [x.op for x in records] == ["ingest", "ask"] and records[-1].ok


def test_not_found_guardrail_path(ctx):
    c, *_ = ctx
    assert c.post("/ask", json={"question": "anything"}).json()["found"] is False


def test_validation_error_is_structured(ctx):
    c, *_ = ctx
    r = c.post("/ask", json={"question": ""})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_upload_rejects_non_pdf_and_bad_pdf(ctx):
    c, *_ = ctx
    assert c.post("/upload", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415
    r = c.post("/upload", files={"file": ("a.pdf", b"not pdf", "application/pdf")})
    assert r.status_code == 422 and r.json()["error"] == "bad_document"


def test_llm_down_maps_to_503_and_observer_sees_failure(ctx):
    c, svc, records = ctx

    def boom(*a):
        raise LLMUnavailable("ollama not running")

    svc.backend.ask = boom
    r = c.post("/ask", json={"question": "hi"})
    assert r.status_code == 503 and r.json()["error"] == "llm_unavailable"
    assert records[-1].ok is False and records[-1].error == "LLMUnavailable"


def test_unexpected_error_is_500_without_leak(ctx):
    c, svc, _ = ctx
    svc.backend.sources = lambda: 1 / 0
    c2 = TestClient(app, raise_server_exceptions=False)
    r = c2.get("/sources")
    assert r.status_code == 500 and "division" not in r.text


def test_stream_returns_full_answer(ctx):
    c, *_ = ctx
    c.post("/upload", files={"file": ("hr.pdf", PDF, "application/pdf")})
    r = c.post("/ask/stream", json={"question": "leave?"})
    assert r.text == "Fake answer to: leave?"


def test_slow_ask_does_not_block_health():
    """threadpool প্রমাণ: ০.৫s-এর /ask চলাকালীন /health দ্রুত ফেরে।"""
    import asyncio
    import time

    import httpx

    svc = RagService(FakeBackend(delay=0.5))
    app.dependency_overrides[get_service] = lambda: svc

    async def go():
        t = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=t, base_url="http://t") as cl:
            slow = asyncio.create_task(cl.post("/ask", json={"question": "q"}))
            await asyncio.sleep(0.05)
            t0 = time.perf_counter()
            await cl.get("/health")
            health_s = time.perf_counter() - t0
            await slow
            return health_s

    try:
        assert asyncio.run(go()) < 0.3
    finally:
        app.dependency_overrides.clear()
