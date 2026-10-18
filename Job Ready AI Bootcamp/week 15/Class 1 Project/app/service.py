"""Service layer — প্রতিটা request এখান দিয়েই যায়। Class 30 (monitoring)-এর extension seam।

HIGHLIGHTS — THE SEAM:
``RagService._run()`` হলো একমাত্র জায়গা যেখানে (১) blocking backend call threadpool-এ
যায়, (২) latency মাপা হয়, (৩) সব ``observers`` কে ``CallRecord`` দেওয়া হয়।
Class 30 শুধু ``service.observers.append(my_logger)`` করবে (token/cost/latency
logging) — route বা backend ছুঁতে হবে না।

কেন threadpool? ``async def`` endpoint-এর ভেতরে blocking call (Ollama ~ কয়েক সেকেন্ড)
সরাসরি চালালে পুরো event loop আটকে যায় — অন্য সব request, এমনকি /health-ও, অপেক্ষা
করে। ``run_in_threadpool`` blocking কাজ worker thread-এ পাঠায়, loop মুক্ত থাকে।
(Default threadpool = AnyIO-র ৪০ thread; এটাই concurrency-র ceiling।)
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi.concurrency import run_in_threadpool

log = logging.getLogger("rag_api.service")


@dataclass
class CallRecord:
    """একটা backend call-এর ফলাফল — monitoring-এর কাঁচামাল।"""

    op: str  # "ask" | "ingest" | "sources"
    request_id: str | None
    latency_s: float
    ok: bool
    error: str | None
    result: Any  # AskResponse / int / list — Class 30 এখান থেকে answer/citation/token আকার বের করতে পারবে


Observer = Callable[[CallRecord], None]


class RagService:
    def __init__(self, backend, observers: list[Observer] | None = None):
        self.backend = backend
        self.observers: list[Observer] = observers if observers is not None else []

    async def _run(self, op: str, request_id: str | None, fn, *args) -> Any:
        t0 = time.perf_counter()
        result, error = None, None
        try:
            result = await run_in_threadpool(fn, *args)
            return result
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            rec = CallRecord(op, request_id, time.perf_counter() - t0, error is None, error, result)
            for obs in self.observers:
                try:
                    obs(rec)
                except Exception:  # observer ভাঙলে request যেন না ভাঙে
                    log.exception("observer failed")

    async def ask(self, question, source, top_k, request_id=None):
        return await self._run("ask", request_id, self.backend.ask, question, source, top_k)

    async def ingest(self, filename, data, request_id=None) -> int:
        return await self._run("ingest", request_id, self.backend.ingest, filename, data)

    async def sources(self, request_id=None) -> list[str]:
        return await self._run("sources", request_id, self.backend.sources)
