"""OpenTelemetry traces: one trace per request, child spans per RAG stage, exported over OTLP/HTTP.

Point it at any OTLP backend — nothing here is vendor-specific:
  Phoenix (1 process, local):  RAG_OTLP_ENDPOINT=http://localhost:6006/v1/traces
  Langfuse (self-hosted):      RAG_OTLP_ENDPOINT=http://localhost:3000/api/public/otel/v1/traces
                               RAG_OTLP_HEADERS="Authorization=Basic <base64(public_key:secret_key)>"
No endpoint -> tracing is off (zero overhead).

HIGHLIGHTS:
* **Post-hoc spans:** service layer প্রতিটা stage-এর আসল সময় মেপে রাখে (embed, search, llm, queue wait)।
  Request শেষে সেই মাপ থেকে explicit start/end time দিয়ে span বানানো হয় — RAG core-এ tracing code
  ঢোকাতে হয় না। Ceiling: stage-গুলো পরপর ধরে নেওয়া হয় (এখানে সত্যিই পরপর চলে)।
* **Two attribute vocabularies:** ``gen_ai.*`` = OpenTelemetry GenAI semantic conventions (status:
  Development — নাম বদলাতে পারে, তাই সব নাম এই একটা file-এ)। ``openinference.*`` / ``llm.*`` /
  ``input.value`` = Phoenix-এর UI যেটা পড়ে span-কে LLM/RETRIEVER/CHAIN হিসেবে দেখাতে।
* ``gen_ai.provider.name`` পুরোনো ``gen_ai.system``-এর বদলি; "ollama" তালিকায় নেই, custom মান চলে।
* **Content opt-in:** spec বলে prompt/answer text default-এ capture না (PII)। ``RAG_TRACE_CONTENT=true`` দিলে
  তবেই প্রশ্ন/উত্তর span-এ যায়।
"""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

NS = 1_000_000


def make_tracer(endpoint: str | None, headers: str | None = None, *, exporter: SpanExporter | None = None,
                service_name: str = "rag-api"):
    """Return a tracer, or None when tracing is off. ``exporter`` = tests (InMemorySpanExporter)."""
    if exporter is None and not endpoint:
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        hdrs = dict(kv.split("=", 1) for kv in headers.split(",")) if headers else None
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=hdrs)))
    return provider.get_tracer("rag-api.telemetry")


def emit_trace(tracer, row: dict, *, model: str, temperature: float, max_tokens: int, capture_content: bool) -> None:
    """Build the span tree for one finished request from its log row."""
    end = int(row["ts"] * 1e9)
    start = end - int(row["latency_s"] * 1e9)
    root = tracer.start_span(f"rag.{row['op']}", start_time=start, attributes=_clean({
        "openinference.span.kind": "CHAIN", "rag.request_id": row["request_id"], "rag.outcome": row["outcome"],
        "rag.queue_wait_s": row["queue_wait_s"],
        "input.value": row.get("question") if capture_content else None,          # content = opt-in (PII)
        "output.value": row.get("answer") if capture_content else None}))
    ctx = trace.set_span_in_context(root)
    t = start + int((row["queue_wait_s"] or 0) * 1e9)
    if row["queue_wait_s"]:
        tracer.start_span("queue.wait", context=ctx, start_time=start).end(end_time=t)
    for name, ms, attrs in (
        ("embed", row["embed_ms"], {"openinference.span.kind": "EMBEDDING", "gen_ai.operation.name": "embeddings"}),
        ("retrieve", row["search_ms"], {"openinference.span.kind": "RETRIEVER", "gen_ai.operation.name": "retrieval",
                                        "db.system.name": "chromadb", "rag.top_distance": row["top_distance"]}),
        (f"chat {model}", row["llm_ms"], {
            "openinference.span.kind": "LLM", "gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama",
            "gen_ai.request.model": model, "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens, "gen_ai.usage.input_tokens": row["prompt_tokens"],
            "gen_ai.usage.output_tokens": row["completion_tokens"], "llm.model_name": model,
            "llm.token_count.prompt": row["prompt_tokens"], "llm.token_count.completion": row["completion_tokens"],
            "rag.ttft_ms": row["ttft_ms"], "rag.prefill_ms": row["prefill_ms"]}),
    ):
        if ms is None:
            continue
        kind = SpanKind.CLIENT if name.startswith("chat") else SpanKind.INTERNAL     # spec: inference span = CLIENT
        span = tracer.start_span(name, context=ctx, kind=kind, start_time=t, attributes=_clean(attrs))
        t += int(ms * NS)
        if name.startswith("chat") and row["outcome"] == "degraded":
            span.set_status(Status(StatusCode.ERROR, "LLM unavailable"))
        span.end(end_time=min(t, end))
    if not row["ok"]:
        root.set_status(Status(StatusCode.ERROR, row["error"] or "error"))
    root.end(end_time=end)


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}
