"""Service layer — every request passes through here. The monitoring seam for Class 30.

HIGHLIGHTS:
* **Threadpool:** blocking backend call ``run_in_threadpool``-এ যায়; event loop মুক্ত থাকে।
* **Concurrency gate (backpressure):** একটা ``asyncio.Semaphore`` ঠিক করে একসাথে কয়টা
  request LLM-এ ঢুকবে। Slot ``queue_timeout_s``-এর মধ্যে না পেলে ``Overloaded`` →
  HTTP 503 + ``Retry-After``। অসীম queue-র বদলে দ্রুত, সৎ ব্যর্থতা — client পরে চেষ্টা করে।
  Ollama নিজেও default-এ model-প্রতি একটাই request চালায়, বাকিগুলো queue করে।
* **Exact-match cache:** (normalized question, source, top_k, corpus version) → answer।
  নতুন document এলে corpus version বাড়ে, পুরোনো cache নিজে থেকেই অকেজো হয়।
  Semantic cache ইচ্ছে করে নেই — ভুল "কাছাকাছি" hit আত্মবিশ্বাসী ভুল উত্তর দেয়।
* **Observers:** প্রতিটা call শেষে ``CallRecord`` — Class 30 শুধু observer যোগ করে।
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool

from .errors import Overloaded

log = logging.getLogger("rag_api.service")


@dataclass
class CallRecord:
    op: str                     # ask | stream | ingest | sources
    request_id: str | None
    latency_s: float
    ok: bool
    error: str | None
    result: Any                 # AskResult / IngestResult / list
    queue_wait_s: float = 0.0
    cached: bool = False


Observer = Callable[[CallRecord], None]


class RagService:
    def __init__(self, backend, *, max_concurrent: int = 2, queue_timeout_s: float = 30.0,
                 cache_size: int = 256, observers: list[Observer] | None = None):
        self.backend = backend
        self.gate = asyncio.Semaphore(max_concurrent)
        self.queue_timeout_s = queue_timeout_s
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple, Any] = OrderedDict()
        self.corpus_version = 0
        self.in_flight = 0
        self.observers: list[Observer] = observers if observers is not None else []

    # ── helpers ───────────────────────────────────────────────────────
    def _emit(self, rec: CallRecord) -> None:
        for obs in self.observers:
            try:
                obs(rec)
            except Exception:  # a broken observer must never break a request
                log.exception("observer failed")

    async def _acquire(self) -> float:
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(self.gate.acquire(), timeout=self.queue_timeout_s)
        except TimeoutError:
            raise Overloaded(f"All LLM slots busy for {self.queue_timeout_s:.0f}s; retry later",
                             retry_after_s=max(1, int(self.queue_timeout_s))) from None
        self.in_flight += 1
        return time.perf_counter() - t0

    def _release(self) -> None:
        self.in_flight -= 1
        self.gate.release()

    def _key(self, question: str, source, top_k) -> tuple:
        return (" ".join(question.lower().split()), source, top_k, self.corpus_version)

    # ── operations ────────────────────────────────────────────────────
    async def ask(self, question, source=None, top_k=None, request_id=None):
        key = self._key(question, source, top_k)
        if self.cache_size and key in self._cache:
            self._cache.move_to_end(key)
            res = self._cache[key]
            self._emit(CallRecord("ask", request_id, 0.0, True, None, res, cached=True))
            return res, True
        t0, err, res, waited = time.perf_counter(), None, None, 0.0
        try:
            waited = await self._acquire()
            try:
                res = await run_in_threadpool(self.backend.ask, question, source, top_k)
            finally:
                self._release()
            if self.cache_size and not res.degraded:
                self._cache[key] = res
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return res, False
        except Exception as exc:
            err = type(exc).__name__
            raise
        finally:
            self._emit(CallRecord("ask", request_id, time.perf_counter() - t0, err is None, err, res, waited))

    async def open_stream(self, question, source=None, top_k=None, request_id=None) -> "StreamHandle":
        """Claim an LLM slot NOW (so Overloaded becomes a real HTTP 503 before any byte is sent),
        then hand back a handle whose ``events()`` streams the answer. ``close()`` is idempotent and
        is called both by the stream's ``finally`` and by a response background task, so the slot is
        released even if the client disconnects before streaming starts."""
        waited = await self._acquire()
        return StreamHandle(self, question, source, top_k, request_id, waited)

    async def ingest(self, filename, data, request_id=None):
        t0, err, res = time.perf_counter(), None, None
        try:
            res = await run_in_threadpool(self.backend.ingest, filename, data)
            if res.new_chunks:
                self.corpus_version += 1          # invalidates the answer cache
            return res
        except Exception as exc:
            err = type(exc).__name__
            raise
        finally:
            self._emit(CallRecord("ingest", request_id, time.perf_counter() - t0, err is None, err, res))

    async def sources(self) -> list[str]:
        return await run_in_threadpool(self.backend.sources)

    async def ready(self) -> tuple[bool, str]:
        return await run_in_threadpool(self.backend.ready)


class StreamHandle:
    def __init__(self, svc: RagService, question, source, top_k, request_id, waited: float):
        self.svc, self.args, self.request_id, self.waited = svc, (question, source, top_k), request_id, waited
        self.t0, self.final, self.error, self._closed = time.perf_counter(), None, None, False

    async def events(self) -> AsyncIterator[tuple[str, object]]:
        try:
            async for kind, value in iterate_in_threadpool(self.svc.backend.stream(*self.args)):
                if kind == "done":
                    self.final = value
                yield kind, value
        except Exception as exc:
            self.error = type(exc).__name__
            raise
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.svc._release()
        self.svc._emit(CallRecord("stream", self.request_id, time.perf_counter() - self.t0,
                                  self.error is None and self.final is not None, self.error or
                                  (None if self.final is not None else "Disconnected"), self.final, self.waited))
