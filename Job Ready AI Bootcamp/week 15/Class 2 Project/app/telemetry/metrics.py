"""Prometheus metrics — the four golden signals, measured per RAG stage.

HIGHLIGHTS:
* **Buckets:** prometheus-client-এর default bucket 10 s-এ শেষ। Local LLM-এর উত্তর প্রায়ই 10 s পার হয় →
  সব `+Inf`-এ পড়ে, p95 অর্থহীন। তাই LLM-আকারের bucket: 64 s, 120 s পর্যন্ত।
* **Cardinality:** label-এ শুধু ছোট, সীমিত মান (route template, status, outcome)। কখনো user id,
  প্রশ্ন, request id, raw URL না — প্রতিটা নতুন মান = নতুন time series = Prometheus-এর memory শেষ।
* **Naming:** base unit (`_seconds`, না ms), counter `_total`, prefix `rag_`।
* নিজস্ব ``CollectorRegistry`` — test-এ বারবার import করলে "Duplicated timeseries" error হয় না।

| Golden signal | Metric |
|---|---|
| Latency | `rag_request_duration_seconds`, `rag_stage_duration_seconds{stage}`, `rag_ttft_seconds` |
| Traffic | `rag_http_requests_total`, `rag_tokens_total{kind}` |
| Errors | `rag_http_requests_total{status=~"5.."}`, `rag_answers_total{outcome}`, `rag_quality_flags_total{flag}` |
| Saturation | `rag_llm_in_flight`, `rag_queue_wait_seconds` |
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

LLM_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 120)
FAST_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)

HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY)
HTTP_DURATION = Histogram("rag_request_duration_seconds", "HTTP request duration (until headers for streams)",
                          ["route"], buckets=LLM_BUCKETS, registry=REGISTRY)
STAGE = Histogram("rag_stage_duration_seconds", "Time per RAG stage", ["stage"],
                  buckets=FAST_BUCKETS + LLM_BUCKETS[3:], registry=REGISTRY)
TTFT = Histogram("rag_ttft_seconds", "Time to first token (streaming)", buckets=LLM_BUCKETS, registry=REGISTRY)
QUEUE_WAIT = Histogram("rag_queue_wait_seconds", "Wait for an LLM slot", buckets=LLM_BUCKETS, registry=REGISTRY)
TOKENS = Counter("rag_tokens_total", "LLM tokens", ["kind"], registry=REGISTRY)            # kind=prompt|completion
DECODE_TPS = Histogram("rag_decode_tokens_per_second", "Decode speed", buckets=(5, 10, 15, 20, 25, 30, 40, 60, 100),
                       registry=REGISTRY)
ANSWERS = Counter("rag_answers_total", "Answers by outcome", ["op", "outcome"], registry=REGISTRY)
QUALITY_FLAGS = Counter("rag_quality_flags_total", "Answers failing a cheap quality check", ["flag"],
                        registry=REGISTRY)                    # flag=bad_citation|unsupported_number|truncated
LEXICAL = Histogram("rag_lexical_support_ratio", "Share of answer words found in the passages",
                    buckets=(0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), registry=REGISTRY)
FEEDBACK = Counter("rag_feedback_total", "User feedback", ["vote"], registry=REGISTRY)    # vote=up|down
COLD_STARTS = Counter("rag_llm_cold_starts_total", "LLM calls that had to (re)load the model", registry=REGISTRY)
IN_FLIGHT = Gauge("rag_llm_in_flight", "Requests holding an LLM slot right now", registry=REGISTRY)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
