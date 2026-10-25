"""
HTTP-level tests with the FakeBackend: contract, errors, auth, SSE, backpressure, cache.

HIGHLIGHTS: ``RAG_BACKEND=fake`` — Ollama/Chroma/model কিছুই লাগে না। Service-কে
``dependency_overrides`` দিয়ে test-এর নিজের settings-এ বানানো হয়।
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.backend import FakeBackend
from app.main import app, get_service
from app.service import RagService

PDF = b"%PDF-1.4 fake pdf bytes"


def client_with(svc: RagService, api_key: str | None = None) -> TestClient:
    app.dependency_overrides[get_service] = lambda: svc
    c = TestClient(app)
    c.__enter__()
    app.state.settings.api_key = api_key
    return c


@pytest.fixture
def svc():
    s = RagService(FakeBackend(), max_concurrent=2, queue_timeout_s=1.0, cache_size=8)
    yield s
    app.dependency_overrides.clear()


@pytest.fixture
def client(svc):
    c = client_with(svc)
    yield c
    c.__exit__(None, None, None)


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["backend"] == "fake"


def test_upload_validation_and_idempotency(client):
    r = client.post("/v1/documents", files={"file": ("hr.pdf", PDF, "application/pdf")})
    assert r.status_code == 201 and r.json()["new_chunks"] == 3 and r.headers["x-request-id"]
    assert client.post("/v1/documents", files={"file": ("hr.pdf", PDF, "application/pdf")}).json()["new_chunks"] == 0
    assert client.post("/v1/documents", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415
    bad = client.post("/v1/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert bad.status_code == 422 and bad.json()["error"] == "bad_document"
    assert client.get("/v1/documents").json() == {"sources": ["hr.pdf"]}


def test_upload_size_limit(client):
    client.app.state.settings.max_upload_mb = 0
    r = client.post("/v1/documents", files={"file": ("big.pdf", PDF, "application/pdf")})
    client.app.state.settings.max_upload_mb = 20
    assert r.status_code == 413


def test_ask_contract(client):
    r = client.post("/v1/ask", json={"question": "anything?"})
    assert r.status_code == 200 and r.json()["found"] is False and r.json()["citations"] == []
    client.post("/v1/documents", files={"file": ("hr.pdf", PDF, "application/pdf")})
    body = client.post("/v1/ask", json={"question": "leave?"}).json()
    assert body["found"] and body["citations"][0]["source"] == "hr.pdf" and body["citations"][0]["n"] == 1
    assert body["usage"]["completion_tokens"] == 12 and "llm_ms" in body["timings_ms"]


def test_validation_errors_are_structured(client):
    r = client.post("/v1/ask", json={"question": ""})
    assert r.status_code == 422 and r.json()["error"] == "validation_error" and r.json()["request_id"]
    assert client.post("/v1/ask", json={"question": "q", "top_k": 99}).status_code == 422


def test_cache_hit_and_invalidation_on_new_document(client, svc):
    client.post("/v1/documents", files={"file": ("hr.pdf", PDF, "application/pdf")})
    assert client.post("/v1/ask", json={"question": "Leave?"}).json()["cached"] is False
    assert client.post("/v1/ask", json={"question": "  leave? "}).json()["cached"] is True   # normalized key
    client.post("/v1/documents", files={"file": ("new.pdf", PDF + b"2", "application/pdf")})
    assert client.post("/v1/ask", json={"question": "leave?"}).json()["cached"] is False      # corpus changed


def test_api_key(svc):
    c = client_with(svc, api_key="s3cret")
    try:
        assert c.get("/v1/documents").status_code == 401
        assert c.get("/v1/documents", headers={"X-API-Key": "wrong"}).json()["error"] == "unauthorized"
        assert c.get("/v1/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200          # ops endpoints stay open for the orchestrator
    finally:
        app.state.settings.api_key = None
        c.__exit__(None, None, None)


def test_sse_stream(client, svc):
    client.post("/v1/documents", files={"file": ("hr.pdf", PDF, "application/pdf")})
    with client.stream("POST", "/v1/ask/stream", json={"question": "leave?"}) as s:
        assert s.headers["content-type"].startswith("text/event-stream")
        assert s.headers["x-accel-buffering"] == "no"
        text = "".join(s.iter_text())
    events = [b.split("\n") for b in text.strip().split("\n\n")]
    names = [e[0].removeprefix("event: ") for e in events]
    assert names[0] == "citations" and names[-1] == "done" and set(names[1:-1]) == {"token"}
    done = json.loads(events[-1][1].removeprefix("data: "))
    assert done["found"] and done["answer"].startswith("Fake answer")
    assert svc.in_flight == 0                                  # slot released after the stream


def test_backpressure_returns_503_with_retry_after():
    slow = RagService(FakeBackend(delay=0.6), max_concurrent=1, queue_timeout_s=0.1, cache_size=0)
    slow.backend.docs["hr.pdf"] = "x"
    app.dependency_overrides[get_service] = lambda: slow

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            app.state.settings = __import__("app.settings", fromlist=["get_settings"]).get_settings()
            return await asyncio.gather(*[ac.post("/v1/ask", json={"question": f"q{i}"}) for i in range(3)])

    try:
        responses = asyncio.run(run())
    finally:
        app.dependency_overrides.clear()
    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 503, 503]
    rejected = next(r for r in responses if r.status_code == 503)
    assert rejected.headers["retry-after"] == "1" and rejected.json()["error"] == "overloaded"


def test_observers_get_call_records(client, svc):
    seen = []
    svc.observers.append(seen.append)
    client.post("/v1/documents", files={"file": ("hr.pdf", PDF, "application/pdf")})
    client.post("/v1/ask", json={"question": "leave?"})
    assert [r.op for r in seen] == ["ingest", "ask"] and all(r.ok for r in seen)


def test_broken_observer_never_breaks_requests(client, svc):
    svc.observers.append(lambda rec: 1 / 0)
    assert client.post("/v1/ask", json={"question": "q?"}).status_code == 200


def test_stream_client_leaving_early_releases_slot(client, svc):
    seen = []
    svc.observers.append(seen.append)
    client.post("/v1/documents", files={"file": ("hr.pdf", PDF, "application/pdf")})
    with client.stream("POST", "/v1/ask/stream", json={"question": "leave?"}) as s:
        next(s.iter_lines())                                    # read one line, then hang up
    assert svc.in_flight == 0 and seen[-1].op == "stream"
