"""Monitor = one observer on ``RagService.observers``: CallRecord -> signals -> SQLite + Prometheus + trace.

HIGHLIGHTS: Class 1-এর service একটা seam রেখে গিয়েছিল — প্রতিটা call শেষে ``CallRecord``।
Monitoring পুরোটা এখানে, RAG core বা route-এ কোনো বদল নেই। একটা observer ভাঙলে request ভাঙে না
(service ``_emit`` exception গিলে log করে)। Quality signal একবারই হিসাব হয়, তিন জায়গায় একই মান যায়।
"""
from __future__ import annotations

import time

from . import metrics as m
from .quality import assess
from .store import RequestLog
from .tracing import emit_trace


class Monitor:
    def __init__(self, log: RequestLog, *, model: str, temperature: float, num_predict: int, tracer=None,
                 trace_content: bool = False):
        self.log, self.tracer, self.trace_content = log, tracer, trace_content
        self.model, self.temperature, self.num_predict = model, temperature, num_predict

    def row(self, rec) -> dict:
        res = rec.result if rec.op in ("ask", "stream") else None
        sig = assess(res, cached=rec.cached, ok=rec.ok, num_predict=self.num_predict) if rec.op != "ingest" else None
        t = getattr(res, "timings_ms", None) or {}
        fresh = res is not None and not rec.cached          # cached = no LLM work happened now
        u = res.usage if fresh and res.usage else None
        return {
            "ts": time.time(), "request_id": rec.request_id, "op": rec.op, "ok": int(rec.ok), "error": rec.error,
            "outcome": "disconnected" if rec.error == "Disconnected" else
                       (sig.outcome if sig else ("ok" if rec.ok else "error")),
            "latency_s": round(rec.latency_s, 4), "queue_wait_s": round(rec.queue_wait_s, 4),
            "embed_ms": t.get("embed_ms") if fresh else None, "search_ms": t.get("search_ms") if fresh else None,
            "llm_ms": t.get("llm_ms") if fresh else None,
            "ttft_ms": round(u.ttft_ms, 1) if u and u.ttft_ms else None,
            "prefill_ms": round(u.prefill_ms, 1) if u else None, "decode_ms": round(u.decode_ms, 1) if u else None,
            "load_ms": round(u.load_ms, 1) if u else None,
            "prompt_tokens": u.prompt_tokens if u else None, "completion_tokens": u.completion_tokens if u else None,
            "tokens_per_s": u.tokens_per_s if u else None,
            "top_distance": round(res.hits[0].distance, 2) if fresh and res.hits else None,
            "citations": sig.citations if sig else None,
            "citation_valid": _int(sig.citation_valid) if sig else None,
            "numbers_supported": _int(sig.numbers_supported) if sig else None,
            "unsupported_numbers": sig.unsupported_numbers if sig else None,
            "lexical_support": sig.lexical_support if sig else None,
            "truncated": int(sig.truncated) if sig else None,
            "model": self.model, "question": getattr(rec, "question", None),
            "answer": res.answer if res is not None else None,
            "chunk_ids": ",".join(h.chunk_id for h in res.hits) if fresh and res.hits else None,
        }

    def observe(self, rec) -> None:
        row = self.row(rec)
        self.log.add(row)
        self._metrics(row)
        if self.tracer is not None:
            emit_trace(self.tracer, row, model=self.model, temperature=self.temperature,
                       max_tokens=self.num_predict, capture_content=self.trace_content)

    @staticmethod
    def _metrics(r: dict) -> None:
        if r["op"] == "ingest":
            return
        m.ANSWERS.labels(r["op"], r["outcome"]).inc()
        m.QUEUE_WAIT.observe(r["queue_wait_s"])
        for stage in ("embed", "search", "llm"):
            if r[f"{stage}_ms"] is not None:
                m.STAGE.labels(stage).observe(r[f"{stage}_ms"] / 1000)
        for k in ("prefill", "decode"):
            if r[f"{k}_ms"]:
                m.STAGE.labels(k).observe(r[f"{k}_ms"] / 1000)
        if r["ttft_ms"]:
            m.TTFT.observe(r["ttft_ms"] / 1000)
        if r["prompt_tokens"] is not None:
            m.TOKENS.labels("prompt").inc(r["prompt_tokens"])
            m.TOKENS.labels("completion").inc(r["completion_tokens"])
        if r["tokens_per_s"]:
            m.DECODE_TPS.observe(r["tokens_per_s"])
        if (r["load_ms"] or 0) > 500:            # warm calls report a few ms; seconds = model was (re)loaded
            m.COLD_STARTS.inc()
        if r["outcome"] == "answered":
            for flag, bad in (("bad_citation", r["citation_valid"] == 0),
                              ("unsupported_number", r["numbers_supported"] == 0), ("truncated", r["truncated"])):
                if bad:
                    m.QUALITY_FLAGS.labels(flag).inc()
            if r["lexical_support"] is not None:
                m.LEXICAL.observe(r["lexical_support"])


def _int(b: bool | None):
    return None if b is None else int(b)
