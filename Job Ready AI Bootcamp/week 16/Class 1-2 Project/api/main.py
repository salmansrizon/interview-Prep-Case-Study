"""FastAPI app for the Offline Market Intelligence Team.

Run:  uvicorn api.main:app --port 8000          ->  http://localhost:8000/docs
      MKT_BACKEND=fake uvicorn api.main:app     (no Ollama/FinBERT needed; canned answers)

HIGHLIGHTS:
* ``/v1/risk/{ticker}`` = শুধু deterministic code (SQLite → FinBERT → math), কোনো LLM call নেই — দ্রুত,
  বারবার একই উত্তর। ``/v1/analyze/{ticker}`` = পুরো CrewAI Flow (মিনিট খানেক), তাই ``POST``
  (side effect: PDF লেখা, SQLite-এ index)।
* Analysis gate: একটা ``asyncio.Semaphore`` — একসাথে ``max_concurrent_analyses``-টা crew। Slot
  ``queue_timeout_s``-এ না পেলে 503 + ``Retry-After`` (Week 15-এর backpressure)।
* Blocking backend call ``run_in_threadpool``-এ — event loop (আর ``/health``) মুক্ত থাকে।
* সব error-এর একই আকার ``{error, detail, request_id}``; stack trace কখনো বাইরে যায় না।
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .backend import make_backend
from .errors import ServiceError, Unavailable
from .schemas import AnalyzeResponse, ErrorResponse, HealthResponse, ReadyResponse, RiskResponse, TickersResponse
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = app.state.settings = get_settings()
    app.state.backend = None                       # built lazily on first use (tests override get_backend)
    app.state.gate = asyncio.Semaphore(s.max_concurrent_analyses)
    yield


app = FastAPI(title="market-intel-api", version="1.0.0", lifespan=lifespan,
              description="Offline Market Intelligence Team: deterministic Risk Scores from a local snapshot, "
                          "explained by a local CrewAI team (News Scout Agent + Risk Analyst Agent) and published "
                          "as a PDF memo. Educational project, synthetic data, not investment advice.")


def get_backend(request: Request):
    st = request.app.state
    if st.backend is None:
        st.backend = make_backend(st.settings)
    return st.backend


Ticker = Annotated[str, Path(pattern=r"^[A-Za-z0-9.^\-]{1,12}$", examples=["ACME"])]
Backend = Annotated[object, Depends(get_backend)]
v1 = APIRouter(prefix="/v1")


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
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


# ── ops ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(status="ok")             # liveness: never touches SQLite, FinBERT or Ollama


@app.get("/ready", response_model=ReadyResponse, tags=["ops"], responses={503: {"model": ErrorResponse}})
async def ready(backend: Backend):
    ok, msg = await run_in_threadpool(backend.ready)
    body = ReadyResponse(status="ready" if ok else "not_ready", backend=backend.name, detail=msg,
                         tickers=len(await run_in_threadpool(backend.tickers)))
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)


# ── market ────────────────────────────────────────────────────────────
@v1.get("/tickers", response_model=TickersResponse, tags=["market"])
async def tickers(backend: Backend):
    return TickersResponse(tickers=await run_in_threadpool(backend.tickers))


@v1.get("/risk/{ticker}", response_model=RiskResponse, tags=["market"],
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def risk(ticker: Ticker, request: Request, backend: Backend):
    """Deterministic Risk Score: metrics + FinBERT sentiment from the local store. No LLM."""
    body = await run_in_threadpool(backend.risk, ticker.upper())
    return RiskResponse(**body, request_id=request.state.request_id)


@v1.post("/analyze/{ticker}", response_model=AnalyzeResponse, tags=["agents"],
         responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def analyze(ticker: Ticker, request: Request, backend: Backend,
                  mode: Literal["grounded", "tool"] = "grounded"):
    """Full Flow: Risk Score (code) -> News Scout Agent -> Risk Analyst Agent -> verify (code) -> PDF or needs_human."""
    s, gate = request.app.state.settings, request.app.state.gate
    try:
        await asyncio.wait_for(gate.acquire(), timeout=s.queue_timeout_s)
    except TimeoutError:
        raise Unavailable(f"An analysis is already running; retry in {s.queue_timeout_s:.0f}s", "overloaded",
                          {"Retry-After": str(max(1, int(s.queue_timeout_s)))}) from None
    try:
        body = await run_in_threadpool(backend.analyze, ticker.upper(), mode)
    finally:
        gate.release()
    return AnalyzeResponse(**body, request_id=request.state.request_id)


@v1.get("/memo/{ticker}", tags=["agents"], response_class=Response,
        responses={200: {"content": {"application/pdf": {}}}, 404: {"model": ErrorResponse}})
async def memo(ticker: Ticker, backend: Backend):
    """Latest published PDF investment memo for the ticker."""
    name, data = await run_in_threadpool(backend.memo_pdf, ticker.upper())
    return Response(data, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{name}"'})


app.include_router(v1)
