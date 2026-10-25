"""FastAPI app (Class 1 service) + monitoring: /metrics, /v1/stats, /v1/dashboard, /v1/feedback.

Run:  uvicorn app.main:app --port 8000      ->  http://localhost:8000/docs
      RAG_BACKEND=fake uvicorn app.main:app  (no Ollama/Chroma needed)

HIGHLIGHTS:
* ``lifespan`` (not the removed ``on_event``) creates heavy resources once per process.
* Every endpoint is ``async def`` and pushes blocking work to the threadpool via the service layer.
* ``/v1/...`` = versioned API: prompt, model or schema changes alter behaviour, so clients pin a version.
* ``/health`` = liveness (process up), ``/ready`` = readiness (Ollama reachable + model pulled).
* SSE via FastAPI's built-in ``fastapi.sse`` (0.135+): events ``citations`` → ``token``... → ``done``.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.sse import EventSourceResponse, format_sse_event
from starlette.background import BackgroundTask

from .backend import make_backend
from .telemetry import dashboard
from .telemetry import metrics as tm
from .telemetry.monitor import Monitor
from .telemetry.store import RequestLog
from .telemetry.tracing import make_tracer
from .errors import ServiceError, Unauthorized
from .schemas import (AskRequest, AskResponse, Citation, ErrorResponse, FeedbackRequest, HealthResponse,
                      ReadyResponse, SourcesResponse, UploadResponse, Usage)
from .service import RagService
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = app.state.settings = get_settings()
    app.state.service = None
    app.state.monitor = Monitor(RequestLog(s.monitor_db, log_text=s.log_text), model=s.model,
                                temperature=s.temperature, num_predict=s.num_predict,
                                tracer=make_tracer(s.otlp_endpoint, s.otlp_headers), trace_content=s.trace_content)

    def make_service() -> RagService:
        svc = RagService(make_backend(s), max_concurrent=s.max_concurrent_llm, queue_timeout_s=s.queue_timeout_s,
                         cache_size=s.cache_size, observers=[app.state.monitor.observe])
        tm.IN_FLIGHT.set_function(lambda: svc.in_flight)      # saturation, read at scrape time
        return svc

    app.state.make_service = make_service
    if s.warmup and s.backend == "real":
        # Pay the cold start (embedding model load, Chroma open, Ollama model load) BEFORE the first user
        # does. Measured with a cold Ollama: first request TTFT 8.4 s without this, ~2.5 s with it.
        app.state.service = svc = app.state.make_service()
        await run_in_threadpool(svc.backend.warmup)
    yield


app = FastAPI(title="rag-api", version="2.0.0", lifespan=lifespan,
              description="Local RAG Q&A as a service: upload PDFs, ask questions, get cited (optionally streamed) "
                          "answers. Runs on Ollama llama3.1:8b + ChromaDB, fully offline.")


def get_service(request: Request) -> RagService:
    st = request.app.state
    if st.service is None:
        st.service = st.make_service()
    return st.service


_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(request: Request, key: str | None = Security(_api_key)) -> None:
    expected = request.app.state.settings.api_key
    if expected and key != expected:
        raise Unauthorized("Missing or invalid X-API-Key header")


Svc = Annotated[RagService, Depends(get_service)]
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_key)])


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.1f}"
    # Label = route TEMPLATE ("/v1/ask"), never the raw path: raw paths are unbounded -> cardinality blow-up.
    route = getattr(request.scope.get("route"), "path", "unmatched")
    tm.HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
    tm.HTTP_DURATION.labels(route).observe(elapsed)
    return response


def _err(request: Request, status: int, error: str, detail: str, headers=None) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    return _err(request, exc.status_code, exc.code, str(exc), exc.headers)


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return _err(request, exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    return _err(request, 422, "validation_error", f"{'.'.join(map(str, first['loc']))}: {first['msg']}")


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    return _err(request, 500, "internal_error", "Unexpected server error")   # never leak a stack trace


def _citations(hits) -> list[Citation]:
    return [Citation(n=i, source=h.source, page=h.page, chunk_id=h.chunk_id, distance=round(h.distance, 2),
                     snippet=h.text[:200]) for i, h in enumerate(hits, 1)]


def _usage(u) -> Usage | None:
    return None if u is None else Usage(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                                        tokens_per_s=u.tokens_per_s, ttft_ms=u.ttft_ms, load_ms=round(u.load_ms, 1),
                                        prefill_ms=round(u.prefill_ms, 1))


# ── ops ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(status="ok")          # liveness: never touches Ollama or Chroma


@app.get("/ready", response_model=ReadyResponse, tags=["ops"], responses={503: {"model": ErrorResponse}})
async def ready(request: Request, svc: Svc):
    ok, msg = await svc.ready()
    body = ReadyResponse(status="ready" if ok else "not_ready", backend=svc.backend.name, llm=msg,
                         indexed_documents=len(await svc.sources()) if ok else 0)
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)


@app.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics():
    """Prometheus scrape target. Plain route (not ``make_asgi_app`` mount -> no 307 redirect to /metrics/).
    Keep it on an internal network: it reveals traffic, not user data."""
    body, ctype = tm.render()
    return Response(body, media_type=ctype)


# ── monitoring ────────────────────────────────────────────────────────
@v1.get("/stats", tags=["monitoring"])
async def stats(request: Request, window_s: int = 3600):
    s = request.app.state.settings
    return request.app.state.monitor.log.summary(window_s, slo_availability=s.slo_availability,
                                                 slo_latency_s=s.slo_latency_s,
                                                 slo_latency_target=s.slo_latency_target)


@v1.get("/dashboard", tags=["monitoring"], response_class=HTMLResponse)
async def dash(request: Request, window_s: int = 3600):
    log = request.app.state.monitor.log
    return dashboard.render(await stats(request, window_s), log.recent(25))


@v1.post("/feedback", tags=["monitoring"], status_code=204, responses={404: {"model": ErrorResponse}})
async def feedback(body: FeedbackRequest, request: Request):
    """Thumbs up/down for an answer, by its request_id. Use it to FIND bad traces, not as a quality score:
    only a few users vote, and mostly when angry."""
    if not request.app.state.monitor.log.feedback(body.request_id, body.vote, body.comment):
        raise HTTPException(404, f"Unknown request_id {body.request_id!r}")
    tm.FEEDBACK.labels("up" if body.vote == 1 else "down").inc()
    return Response(status_code=204)


# ── documents ─────────────────────────────────────────────────────────
@v1.get("/documents", response_model=SourcesResponse, tags=["documents"])
async def list_documents(svc: Svc):
    return SourcesResponse(sources=await svc.sources())


@v1.post("/documents", response_model=UploadResponse, status_code=201, tags=["documents"],
         responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def upload(request: Request, svc: Svc, file: UploadFile = File(...)):
    name = (file.filename or "document.pdf").replace("/", "_").replace("\\", "_")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(415, "Only .pdf files are supported")
    limit = request.app.state.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)                 # read limit+1: detect "too big" without loading it all
    if len(data) > limit:
        raise HTTPException(413, f"File larger than {request.app.state.settings.max_upload_mb} MB")
    res = await svc.ingest(name, data, request_id=request.state.request_id)
    return UploadResponse(filename=name, pages=res.pages, chunks=res.chunks, new_chunks=res.new_chunks,
                          request_id=request.state.request_id)


# ── questions ─────────────────────────────────────────────────────────
@v1.post("/ask", response_model=AskResponse, tags=["qa"],
         responses={503: {"model": ErrorResponse}, 401: {"model": ErrorResponse}})
async def ask(body: AskRequest, request: Request, svc: Svc):
    res, cached = await svc.ask(body.question, body.source, body.top_k, request_id=request.state.request_id)
    return AskResponse(answer=res.answer, found=res.found, degraded=res.degraded, cached=cached,
                       citations=_citations(res.hits) if res.found else [], timings_ms=res.timings_ms,
                       usage=_usage(res.usage), request_id=request.state.request_id)


# NOTE: no response_class=EventSourceResponse here — with that, FastAPI expects the endpoint itself to be a
# generator. We return the response object ourselves so the slot can be claimed (or 503 raised) first.
@v1.post("/ask/stream", tags=["qa"],
         responses={200: {"content": {"text/event-stream": {}}, "description": "SSE stream"},
                    503: {"model": ErrorResponse}})
async def ask_stream(body: AskRequest, request: Request, svc: Svc):
    """Server-Sent Events: `citations` (once) → `token` (many) → `done` (answer, found, timings, usage).

    The LLM slot is claimed before the response starts, so overload is a normal HTTP 503."""
    handle = await svc.open_stream(body.question, body.source, body.top_k, request_id=request.state.request_id)

    async def events():
        async for kind, value in handle.events():
            if await request.is_disconnected():       # stop generating for a client that left
                break
            if kind == "hits":
                yield sse("citations", [c.model_dump() for c in _citations(value)])
            elif kind == "token":
                yield sse("token", value)
            elif kind == "done":
                yield sse("done", {
                    "answer": value.answer, "found": value.found, "degraded": value.degraded,
                    "timings_ms": value.timings_ms, "usage": (_usage(value.usage).model_dump() if value.usage else None),
                    "request_id": request.state.request_id})

    return EventSourceResponse(events(), background=BackgroundTask(handle.close),
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def sse(event: str, data) -> bytes:
    """One Server-Sent Event: `event: <name>` + `data: <json>` + blank line."""
    return format_sse_event(event=event, data_str=json.dumps(data, ensure_ascii=False))


app.include_router(v1)
