"""FastAPI service for the healthcare assistant: /health, /ready, /v1/ask, /v1/documents, /v1/ingest.

Run:  uvicorn src.api.main:app --port 8000          ->  http://localhost:8000/docs
      HC_BACKEND=fake uvicorn src.api.main:app      (no Ollama, no model download)

HIGHLIGHTS:
* ``lifespan``-এ assistant একবার তৈরি হয়; ``HC_AUTO_INGEST=true`` আর collection খালি হলে bundled
  handbook index হয়। তাই নতুন container (Week 18) এক command-এই কাজ করে, আলাদা ingest ধাপ লাগে না।
* Blocking কাজ (embedding, Chroma, Ollama) ``run_in_threadpool``-এ — event loop আর /health আটকায় না।
* Privacy: middleware শুধু method, path, status, সময় log করে — request body কখনো না। Validation
  error-এর detail-এ শুধু field আর message যায়, input-এর text না।
* Ollama বন্ধ থাকলে শুধু ``answered`` route-এর প্রশ্নে 503 ``llm_unavailable``; emergency, refusal আর
  abstention LLM ছাড়াই উত্তর দেয়।
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..assistant import HealthcareAssistant
from ..backend import build_assistant
from ..rag.components import LLMUnavailable
from ..rag.kb import DISCLAIMER
from ..settings import get_settings
from .schemas import (AskRequest, AskResponse, Citation, DocumentsResponse, ErrorResponse, HealthResponse,
                      IngestResponse, PageInfo, ReadyResponse)

log = logging.getLogger("healthcare.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = app.state.settings = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    a = app.state.assistant = build_assistant(s)
    if s.auto_ingest:
        try:
            if await run_in_threadpool(a.store.count) == 0:
                res = await run_in_threadpool(a.ingest, s.kb_dir)
                log.info("auto-ingested %s pages / %s sections from %s", res.pages, res.sections, s.kb_dir)
        except Exception as exc:  # noqa: BLE001 - e.g. Chroma server not up yet; /ready reports it
            log.warning("auto-ingest skipped: %s", type(exc).__name__)
    yield


app = FastAPI(title="healthcare-assistant", version="1.0.0", lifespan=lifespan,
              description="Privacy-first healthcare assistant over a local clinic handbook: PII redaction, "
                          "Red-Flag Escalation, diagnosis/dosing refusal and Abstention, on Ollama llama3.1:8b + "
                          "ChromaDB. Educational, fictional clinic handbook; not medical advice.")


def get_assistant(request: Request) -> HealthcareAssistant:
    return request.app.state.assistant


Assistant = Annotated[HealthcareAssistant, Depends(get_assistant)]
v1 = APIRouter(prefix="/v1")


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    log.info("%s %s %s %.1fms rid=%s", request.method, request.url.path, response.status_code, ms, rid)  # no body
    return response


def _err(request: Request, status: int, error: str, detail: str) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(LLMUnavailable)
async def llm_down_handler(request: Request, exc: LLMUnavailable):
    return _err(request, 503, "llm_unavailable", str(exc))


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return _err(request, exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0]           # loc + msg only: never echo the (possibly PII-containing) input
    return _err(request, 422, "validation_error", f"{'.'.join(map(str, first['loc']))}: {first['msg']}")


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.error("unhandled %s rid=%s", type(exc).__name__, getattr(request.state, "request_id", None))
    return _err(request, 500, "internal_error", "Unexpected server error")


# ── ops ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(status="ok")                 # liveness: never touches Ollama or Chroma


@app.get("/ready", response_model=ReadyResponse, tags=["ops"], responses={503: {"model": ErrorResponse}})
async def ready(request: Request, a: Assistant):
    ok, msg = await run_in_threadpool(a.llm.ping)
    try:
        n = await run_in_threadpool(a.store.count)
    except Exception as exc:  # noqa: BLE001 - Chroma server down
        n, ok, msg = 0, False, f"{msg}; chroma unavailable: {type(exc).__name__}"
    ok = ok and n > 0
    body = ReadyResponse(status="ready" if ok else "not_ready", backend=request.app.state.settings.backend,
                         llm=msg if n else f"{msg}; knowledge base not indexed", indexed_sections=n)
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)


# ── knowledge base ────────────────────────────────────────────────────
@v1.get("/documents", response_model=DocumentsResponse, tags=["knowledge base"])
async def documents(a: Assistant):
    return DocumentsResponse(pages=[PageInfo(**p) for p in await run_in_threadpool(a.store.pages)])


@v1.post("/ingest", response_model=IngestResponse, tags=["knowledge base"])
async def ingest(request: Request, a: Assistant):
    """Re-index the handbook from HC_KB_DIR (server-side files only; no uploads)."""
    res = await run_in_threadpool(a.ingest, request.app.state.settings.kb_dir)
    return IngestResponse(pages=res.pages, sections=res.sections, removed=res.removed)


# ── questions ─────────────────────────────────────────────────────────
@v1.post("/ask", response_model=AskResponse, tags=["qa"], responses={503: {"model": ErrorResponse}})
async def ask(body: AskRequest, request: Request, a: Assistant):
    r = await run_in_threadpool(a.ask, body.question)
    return AskResponse(
        route=r.route, answer=r.answer, rule=r.rule, flags=r.flags, disclaimer=DISCLAIMER, timings_ms=r.timings_ms,
        citations=[Citation(n=n, id=h.id, page=h.page, section=h.title, distance=round(h.distance, 3),
                            snippet=h.text[:240]) for n, h in r.citations],
        redaction_count=r.redaction.total, redactions=r.redaction.counts, redacted_question=r.redaction.text,
        request_id=request.state.request_id)


app.include_router(v1)
