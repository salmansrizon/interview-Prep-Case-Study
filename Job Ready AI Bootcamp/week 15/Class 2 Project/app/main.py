"""FastAPI app: routes + error handling + request-id/timing middleware.

চালানো:  uvicorn app.main:app --port 8000   →  http://localhost:8000/docs
"""

import asyncio
import html
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError

from .backend import make_backend
from .errors import ServiceError
from .schemas import (AskRequest, AskResponse, ErrorResponse, HealthResponse,
                      SourcesResponse, UploadResponse)
from .langfuse_obs import make_langfuse_sink
from .monitor import Monitor
from .service import RagService
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # HIGHLIGHTS: heavy resource (Chroma client, embedding model) একবার বানাই, প্রতি request-এ না।
    # Tests dependency_overrides দিয়ে get_service বদলায়, তাই real backend শুধু প্রথম
    # ব্যবহারে lazily তৈরি হয় (get_service) — import/startup-এ Ollama/Chroma লাগে না।
    app.state.settings = get_settings()
    app.state.service = None
    # HIGHLIGHTS: Monitor (SQLite) সবসময় চালু; Langfuse শুধু env থাকলে sink হিসেবে যুক্ত হয়
    s = app.state.settings
    app.state.monitor = Monitor(s.monitor_db, s.model_name, s.price_key)
    if (sink := make_langfuse_sink()) is not None:
        app.state.monitor.sinks.append(sink)
    yield


app = FastAPI(
    title="rag-api",
    version="1.0.0",
    description="Week 12 RAG Q&A bot as a web service. Upload PDFs, ask questions, get cited answers.",
    lifespan=lifespan,
)


def get_service(request: Request) -> RagService:
    st = request.app.state
    if st.service is None:
        st.service = RagService(make_backend(st.settings))
        st.service.observers.append(st.monitor.observer)  # <-- Class 30 seam
    return st.service


Svc = Annotated[RagService, Depends(get_service)]


@app.middleware("http")
async def request_context(request: Request, call_next):
    # HIGHLIGHTS: request_id + total latency header — Class 30 এখানে access log যোগ করতে পারে
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def _err(request: Request, status: int, error: str, detail: str) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    return _err(request, exc.status_code, exc.code, str(exc))


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return _err(request, exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    loc = ".".join(str(p) for p in first["loc"])
    return _err(request, 422, "validation_error", f"{loc}: {first['msg']}")


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    # stack trace client-কে leak না করে generic message
    return _err(request, 500, "internal_error", "Unexpected server error")


ERRORS = {503: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request, svc: Svc):
    # HIGHLIGHTS: async + threadpool, তাই লম্বা /ask চলাকালীনও /health দ্রুত উত্তর দেয়
    srcs = await svc.sources()
    return HealthResponse(status="ok", backend=request.app.state.settings.backend, indexed_documents=len(srcs))


@app.get("/sources", response_model=SourcesResponse, tags=["documents"])
async def sources(svc: Svc):
    return SourcesResponse(sources=await svc.sources())


@app.post("/ask", response_model=AskResponse, responses=ERRORS, tags=["qa"])
async def ask(body: AskRequest, request: Request, svc: Svc):
    t0 = time.perf_counter()
    rid = request.state.request_id
    res = await svc.ask(body.question, body.source, body.top_k, request_id=rid)
    return res.model_copy(update={"request_id": rid, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})


@app.post("/ask/stream", tags=["qa"])
async def ask_stream(body: AskRequest, request: Request, svc: Svc):
    """Word-by-word streaming (text/plain)।

    Honest limitation: Week 12 `generate()` non-streaming, তাই আগে পুরো উত্তর তৈরি হয়,
    তারপর stream হয়। আসল token streaming-এর জন্য ollama `stream=True` দরকার (lecture দেখুন)।
    """
    res = await svc.ask(body.question, body.source, body.top_k, request_id=request.state.request_id)

    async def gen():
        for word in re.findall(r"\S+\s*", res.answer):
            yield word
            await asyncio.sleep(0)  # event loop-কে অন্যদের সুযোগ দেওয়া

    return StreamingResponse(gen(), media_type="text/plain")


@app.post("/upload", response_model=UploadResponse, status_code=201, responses=ERRORS, tags=["documents"])
async def upload(request: Request, svc: Svc, file: UploadFile = File(...)):
    name = file.filename or "document.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(415, "Only .pdf files are supported")
    mb = request.app.state.settings.max_upload_mb
    data = await file.read(mb * 1024 * 1024 + 1)  # limit+1 পড়ি: বড় হলে ধরা যায়, পুরোটা memory-তে না নিয়ে
    if len(data) > mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {mb} MB")
    n = await svc.ingest(name, data, request_id=request.state.request_id)
    return UploadResponse(filename=name, chunks_indexed=n, request_id=request.state.request_id)


@app.get("/stats", tags=["ops"])
async def stats(request: Request, drift_window: int = 20):
    """প্রতি query-র token/cost/latency/quality-র সারাংশ + drift check (SQLite থেকে)।"""
    return request.app.state.monitor.stats(drift_window)


@app.get("/dashboard", response_class=HTMLResponse, tags=["ops"])
async def dashboard(request: Request):
    st = request.app.state.monitor.stats()
    rows = "".join(
        f"<tr><td>{html.escape(str(r['question'])[:60])}</td><td>{r['latency_s']}</td>"
        f"<td>{r['prompt_tokens']}/{r['completion_tokens']}</td><td>{(r['cost_usd'] or 0):.6f}</td>"
        f"<td>{r['quality']} ({r['quality_label']})</td></tr>" for r in st["recent"])
    head = (f"queries={st['queries']} errors={st['errors']} cost=${st['equivalent_cost_usd']} (illustrative) "
            f"p95={st['latency_s']['p95']} quality_avg={st['quality_avg']} drift={st['drift']['status']}")
    return (f"<h3>rag-api monitor</h3><p>{html.escape(head)}</p><table border=1 cellpadding=4>"
            f"<tr><th>question</th><th>latency s</th><th>tokens in/out</th><th>equiv. $</th><th>quality</th></tr>{rows}</table>")
