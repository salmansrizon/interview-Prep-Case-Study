"""Monitoring tests: quality signals, SQLite summary + SLO burn rates, Prometheus, traces, judge, endpoints.

HIGHLIGHTS: সব offline — FakeBackend, in-memory span exporter, fake judge client। Prometheus registry
process-জুড়ে একটাই, তাই test-এ মান না, *পার্থক্য* (delta) মাপা হয়।
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.backend import FakeBackend
from app.main import app, get_service
from app.rag.components import Hit, Usage
from app.rag.engine import NOT_FOUND, AskResult
from app.service import CallRecord, RagService
from app.telemetry import judge
from app.telemetry.metrics import REGISTRY
from app.telemetry.monitor import Monitor
from app.telemetry.quality import assess
from app.telemetry.store import RequestLog
from app.telemetry.tracing import make_tracer

PDF = b"%PDF-1.4 fake"
HITS = [Hit("c1", "Employees get 12 paid sick days per year.", "hb.pdf", 1, 15.0),
        Hit("c2", "Up to 5 unused days can be carried over.", "hb.pdf", 1, 18.0)]


def result(answer, **kw):
    return AskResult(answer, found=kw.pop("found", True), hits=kw.pop("hits", HITS),
                     timings_ms={"embed_ms": 10.0, "search_ms": 1.0, "llm_ms": 1500.0},
                     usage=kw.pop("usage", Usage(prompt_tokens=400, completion_tokens=20, prefill_ms=60,
                                                 decode_ms=1000, ttft_ms=80)), **kw)


def metric(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ── quality signals ───────────────────────────────────────────────────
def test_signals_good_answer():
    s = assess(result("Employees get 12 paid sick days per year [1]."), cached=False, ok=True, num_predict=512)
    assert (s.outcome, s.citations, s.citation_valid, s.numbers_supported) == ("answered", 1, True, True)
    assert s.lexical_support == 1.0 and not s.truncated


def test_signals_catch_invented_number_bad_citation_and_truncation():
    s = assess(result("Employees get 15 sick days [7]."), cached=False, ok=True, num_predict=20)
    assert not s.numbers_supported and s.unsupported_numbers == "15"
    assert not s.citation_valid and s.truncated                  # [7] with 2 passages; 20 tokens == num_predict
    assert not assess(result("Employees get 12 sick days."), cached=False, ok=True, num_predict=512).citation_valid


@pytest.mark.parametrize("kw,expected", [
    ({"cached": True, "ok": True}, "cached"), ({"cached": False, "ok": False}, "error")])
def test_signals_outcomes(kw, expected):
    assert assess(result("x [1]"), num_predict=512, **kw).outcome == expected
    assert assess(result(NOT_FOUND, found=False), cached=False, ok=True, num_predict=512).outcome == "not_found"
    assert assess(result("down", degraded=True), cached=False, ok=True, num_predict=512).outcome == "degraded"


# ── store: summary, SLO, PII ──────────────────────────────────────────
def rows_for(log, n, *, bad_every=0, slow_every=0, now=None):
    now = now or time.time()
    for i in range(n):
        bad = bool(bad_every) and i % bad_every == 0
        log.add({"ts": now - 60 + i * 0.1, "request_id": f"r{i}", "op": "ask", "ok": int(not bad),
                 "error": "Overloaded" if bad else None, "outcome": "error" if bad else "answered",
                 "latency_s": 12.0 if slow_every and i % slow_every == 0 else 2.0, "queue_wait_s": 0.0,
                 "llm_ms": 1500.0, "prompt_tokens": 400, "completion_tokens": 20, "citation_valid": 1,
                 "numbers_supported": 0 if i == 1 else 1, "question": f"q{i}"})
    return now


def test_summary_rates_and_percentiles(tmp_path):
    log = RequestLog(tmp_path / "m.db")
    now = rows_for(log, 10, bad_every=5)
    s = log.summary(3600, now=now + 1)
    assert s["requests"] == 10 and s["outcomes"]["error"] == 2 and s["rates"]["error"] == 0.2
    assert s["latency_s"]["p50"] == 2.0 and s["tokens"]["prompt"] == 4000
    assert s["quality"]["numbers_supported_rate"] == pytest.approx(7 / 8, abs=1e-3)


def test_slo_burn_rate_pages_on_fast_burn(tmp_path):
    log = RequestLog(tmp_path / "m.db")
    now = rows_for(log, 100, bad_every=5)                       # 20% errors vs 1% budget -> burn 20x
    slo = log.summary(3600, now=now + 1)["slo"]["availability"]
    assert slo["burn_rate"]["5m"] == 20.0 and "page_fast" in slo["alerts"]
    healthy = RequestLog(tmp_path / "h.db")
    now = rows_for(healthy, 100)
    assert healthy.summary(3600, now=now + 1)["slo"]["availability"]["alerts"] == []


def test_latency_slo(tmp_path):
    log = RequestLog(tmp_path / "m.db")
    now = rows_for(log, 100, slow_every=2)                      # 50% slow vs 5% budget -> 10x: ticket, not page
    lat = log.summary(3600, now=now + 1)["slo"]["latency"]
    assert lat["burn_rate"]["1h"] == 10.0 and lat["alerts"] == ["page_slow"]


def test_pii_mode_hashes_text(tmp_path):
    log = RequestLog(tmp_path / "m.db", log_text=False)
    log.add({"ts": time.time(), "op": "ask", "ok": 1, "question": "my salary is 5000", "answer": "a"})
    q = log.recent(1)[0]["question"]
    assert q.startswith("sha256:") and "salary" not in q


# ── HTTP + Prometheus + feedback + dashboard ──────────────────────────
@pytest.fixture
def ctx(tmp_path):
    svc = RagService(FakeBackend(), max_concurrent=2, queue_timeout_s=1.0, cache_size=8)
    app.dependency_overrides[get_service] = lambda: svc
    with TestClient(app) as client:
        mon = Monitor(RequestLog(tmp_path / "m.db"), model="llama3.1:8b", temperature=0.2, num_predict=512)
        app.state.monitor = mon
        svc.observers.append(mon.observe)
        client.post("/v1/documents", files={"file": ("hr.pdf", PDF, "application/pdf")})
        yield client, svc, mon
    app.dependency_overrides.clear()


def test_requests_are_logged_and_summarised(ctx):
    client, _, mon = ctx
    rid = client.post("/v1/ask", json={"question": "leave?"}).headers["x-request-id"]
    client.post("/v1/ask", json={"question": "leave?"})                  # cache hit
    with client.stream("POST", "/v1/ask/stream", json={"question": "sick days?"}) as s:
        "".join(s.iter_text())
    st = client.get("/v1/stats").json()
    assert st["requests"] == 3 and st["outcomes"]["answered"] == 2 and st["outcomes"]["cached"] == 1
    row = [r for r in mon.log.recent(10) if r["request_id"] == rid][0]
    assert row["prompt_tokens"] == 120 and row["citation_valid"] == 1 and row["chunk_ids"]
    assert [r["op"] for r in mon.log.recent(10)] == ["stream", "ask", "ask"]


def test_prometheus_metrics(ctx):
    client, _, _ = ctx
    before = metric("rag_http_requests_total", method="POST", route="/v1/ask", status="200")
    tokens = metric("rag_tokens_total", kind="completion")
    client.post("/v1/ask", json={"question": "metrics?"})
    assert metric("rag_http_requests_total", method="POST", route="/v1/ask", status="200") == before + 1
    assert metric("rag_tokens_total", kind="completion") == tokens + 12
    text = client.get("/metrics").text
    assert 'rag_stage_duration_seconds_bucket{le="64.0",stage="llm"}' in text and "rag_llm_in_flight" in text
    client.get("/v1/documents/nope")                                     # unknown path
    assert 'route="unmatched"' in client.get("/metrics").text            # raw path never becomes a label


def test_503_is_logged_for_ask_and_stream(tmp_path):
    slow = RagService(FakeBackend(delay=0.5), max_concurrent=1, queue_timeout_s=0.05, cache_size=0)
    slow.backend.docs["hr.pdf"] = "x"
    mon = Monitor(RequestLog(tmp_path / "m.db"), model="m", temperature=0, num_predict=512)
    slow.observers.append(mon.observe)
    app.dependency_overrides[get_service] = lambda: slow

    async def run():
        from app.settings import get_settings
        app.state.settings, app.state.monitor = get_settings(), mon
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
            return await asyncio.gather(ac.post("/v1/ask", json={"question": "a"}),
                                        ac.post("/v1/ask", json={"question": "b"}),
                                        ac.post("/v1/ask/stream", json={"question": "c"}))
    try:
        codes = sorted(r.status_code for r in asyncio.run(run()))
    finally:
        app.dependency_overrides.clear()
    assert codes == [200, 503, 503]
    errors = [r for r in mon.log.recent(10) if r["outcome"] == "error"]
    assert {r["op"] for r in errors} == {"ask", "stream"} and all(r["error"] == "Overloaded" for r in errors)


def test_feedback(ctx):
    client, _, mon = ctx
    rid = client.post("/v1/ask", json={"question": "fb?"}).headers["x-request-id"]
    assert client.post("/v1/feedback", json={"request_id": rid, "vote": -1, "comment": "wrong"}).status_code == 204
    assert client.post("/v1/feedback", json={"request_id": "nope", "vote": 1}).status_code == 404
    assert client.post("/v1/feedback", json={"request_id": rid, "vote": 5}).status_code == 422
    assert client.get("/v1/stats").json()["feedback"] == {"up": 0, "down": 1}


def test_dashboard_escapes_user_text(ctx):
    client, _, _ = ctx
    client.post("/v1/ask", json={"question": "<script>alert(1)</script>"})
    html = client.get("/v1/dashboard").text
    assert "RAG API Monitor" in html and "<script>alert" not in html and "&lt;script&gt;" in html


def test_disconnect_is_not_an_error(ctx):
    client, _, mon = ctx
    with client.stream("POST", "/v1/ask/stream", json={"question": "bye?"}) as s:
        next(s.iter_lines())
    assert mon.log.recent(1)[0]["outcome"] == "disconnected"
    assert client.get("/v1/stats").json()["rates"]["error"] == 0.0


# ── traces ────────────────────────────────────────────────────────────
def test_trace_tree_and_genai_attributes(tmp_path):
    exp = InMemorySpanExporter()
    mon = Monitor(RequestLog(tmp_path / "m.db"), model="llama3.1:8b", temperature=0.2, num_predict=512,
                  tracer=make_tracer(None, exporter=exp), trace_content=True)
    mon.observe(CallRecord("ask", "rid1", 1.6, True, None, result("12 days [1]."), 0.05, question="sick?"))
    mon.observe(CallRecord("ask", "rid2", 0.1, True, None, result("down", degraded=True), question="q"))
    root = next(s for s in exp.get_finished_spans() if s.attributes.get("rag.request_id") == "rid1")
    kids = [s for s in exp.get_finished_spans() if s.parent and s.parent.span_id == root.context.span_id]
    assert sorted(k.name for k in kids) == ["chat llama3.1:8b", "embed", "queue.wait", "retrieve"]
    chat = next(k for k in kids if k.name.startswith("chat"))
    assert chat.attributes["gen_ai.provider.name"] == "ollama" and chat.attributes["gen_ai.usage.input_tokens"] == 400
    assert chat.attributes["openinference.span.kind"] == "LLM" and root.attributes["input.value"] == "sick?"
    assert chat.kind.name == "CLIENT" and next(k for k in kids if k.name == "embed").kind.name == "INTERNAL"
    assert all(root.start_time <= k.start_time and k.end_time <= root.end_time for k in kids)
    degraded_chat = [s for s in exp.get_finished_spans() if s.name.startswith("chat") and not s.status.is_ok]
    assert len(degraded_chat) == 1


def test_tracing_off_without_endpoint():
    assert make_tracer(None) is None


def test_trace_content_is_opt_in(tmp_path):
    exp = InMemorySpanExporter()
    mon = Monitor(RequestLog(tmp_path / "m.db"), model="m", temperature=0, num_predict=512,
                  tracer=make_tracer(None, exporter=exp))                       # default: no content
    mon.observe(CallRecord("ask", "r", 1.0, True, None, result("12 days [1]."), question="my salary?"))
    assert all("input.value" not in s.attributes and "output.value" not in s.attributes for s in exp.get_finished_spans())


def test_cold_start_counter(tmp_path):
    before = metric("rag_llm_cold_starts_total")
    mon = Monitor(RequestLog(tmp_path / "m.db"), model="m", temperature=0, num_predict=512)
    cold = Usage(prompt_tokens=10, completion_tokens=2, load_ms=3600, decode_ms=100)
    mon.observe(CallRecord("ask", "c", 4.0, True, None, result("12 days [1].", usage=cold), question="q"))
    mon.observe(CallRecord("ask", "w", 1.0, True, None, result("12 days [1]."), question="q"))
    assert metric("rag_llm_cold_starts_total") == before + 1


# ── offline judge ─────────────────────────────────────────────────────
class FakeJudgeClient:
    """Answers like the live 8B judge: one claim per answer; "15" unsupported; "cashed" = invented quote."""

    def __init__(self):
        self.prompts = []

    def chat(self, model, messages, format, options):
        self.prompts.append(messages[1]["content"])
        answer = messages[1]["content"].split("ANSWER:")[1]
        if "cashed" in answer:      # judge claims support but invents the evidence -> quote check catches it
            claim = {"claim": "cashed out", "evidence": "Sick days can be cashed out.", "supported": True}
        else:
            claim = {"claim": answer.strip(), "evidence": "Employees get 12 paid sick days per year.",
                     "supported": "15" not in answer}
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps({"claims": [claim]})))


class FakeStore:
    def texts(self, ids):
        return {h.chunk_id: h.text for h in HITS if h.chunk_id in ids}


def test_judge_writes_verdicts_back(tmp_path):
    log = RequestLog(tmp_path / "m.db")
    mon = Monitor(log, model="m", temperature=0, num_predict=512)
    mon.observe(CallRecord("ask", "good", 1.0, True, None, result("12 sick days [1]."), question="sick?"))
    mon.observe(CallRecord("ask", "bad", 1.0, True, None, result("15 sick days [1]."), question="sick?"))
    mon.observe(CallRecord("ask", "inconsistent", 1.0, True, None, result("12 days, cashed out [1]."), question="s?"))
    client = FakeJudgeClient()
    out = {r["id"]: r["supported"] for r in judge.run(log, FakeStore(), client, "m", limit=10)}
    assert out == {1: True, 2: False, 3: False}          # 3: judge said true, but its quote is invented
    assert "[1] Employees get 12" in client.prompts[0]
    s = log.summary(3600)["quality"]
    assert s["judged"] == 3 and s["judge_supported_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert judge.run(log, FakeStore(), client, "m", limit=10) == []          # nothing left to judge
