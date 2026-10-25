"""Concurrency sweep: throughput, latency percentiles, 503s and (with --stream) time-to-first-token.

Modes:
  python benchmark/bench.py --fake                         # in-process, no Ollama (FakeBackend, 0.2s per answer)
  python benchmark/bench.py --fake --queue-timeout 0.5     # small queue -> watch 503s appear
  python benchmark/bench.py --url http://localhost:8000 --levels 1,2,4 --requests 8 --stream   # real server

For each concurrency level it sends N requests and reports:
  req/s     = completed (200) requests / wall seconds       -> capacity
  p50/p95/p99 = per-request latency of the 200s             -> user experience
  ttft p50  = time to first `token` event (only with --stream)
  503       = requests rejected by backpressure (fast, honest failure instead of an endless queue)

HIGHLIGHTS: throughput grows until the LLM slots (``RAG_MAX_CONCURRENT_LLM``) are full; after that the
"knee" — throughput flat, latency grows with the queue, then 503s once the wait passes
``RAG_QUEUE_TIMEOUT_S``. Real mode adds "(request i)" to every question so the answer cache never hits.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = ["How many sick days do employees get?", "How many characters must a password have?",
             "How many days per week can I work remotely?", "How much is the home-office allowance?"]


def pct(xs: list[float], p: float) -> float:
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else float("nan")


async def one(client: httpx.AsyncClient, i: int, stream: bool) -> tuple[int, float, float | None]:
    body = {"question": f"{QUESTIONS[i % len(QUESTIONS)]} (request {i})"}
    t0 = time.perf_counter()
    if not stream:
        r = await client.post("/v1/ask", json=body, timeout=600)
        return r.status_code, time.perf_counter() - t0, None
    ttft = None
    async with client.stream("POST", "/v1/ask/stream", json=body, timeout=600) as r:
        async for line in r.aiter_lines():
            if ttft is None and line == "event: token":
                ttft = time.perf_counter() - t0
        return r.status_code, time.perf_counter() - t0, ttft


async def level(client, concurrency: int, total: int, stream: bool) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(i):
        async with sem:
            return await one(client, i, stream)

    t0 = time.perf_counter()
    res = await asyncio.gather(*(guarded(i) for i in range(total)))
    wall = time.perf_counter() - t0
    ok = sorted(lat for code, lat, _ in res if code == 200)
    ttfts = sorted(t for code, _, t in res if code == 200 and t is not None)
    return {"conc": concurrency, "rps": len(ok) / wall, "p50": pct(ok, .5), "p95": pct(ok, .95), "p99": pct(ok, .99),
            "ttft": statistics.median(ttfts) if ttfts else float("nan"),
            "n503": sum(code == 503 for code, _, _ in res), "other": sum(code not in (200, 503) for code, _, _ in res)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--requests", type=int, default=32, help="requests per level")
    ap.add_argument("--stream", action="store_true", help="use /v1/ask/stream and measure TTFT")
    ap.add_argument("--slots", type=int, default=2, help="--fake only: max concurrent LLM calls")
    ap.add_argument("--queue-timeout", type=float, default=30.0, help="--fake only: seconds to wait for a slot")
    ap.add_argument("--api-key")
    a = ap.parse_args()
    if not (a.url or a.fake):
        ap.error("pass --url or --fake")

    headers = {"X-API-Key": a.api_key} if a.api_key else {}
    if a.fake:
        from app.backend import FakeBackend
        from app.main import app, get_service
        from app.service import RagService
        from app.settings import get_settings

        svc = RagService(FakeBackend(delay=0.2), max_concurrent=a.slots, queue_timeout_s=a.queue_timeout,
                         cache_size=0)
        svc.backend.docs["employee_handbook.pdf"] = "x"
        app.dependency_overrides[get_service] = lambda: svc
        app.state.settings = get_settings()          # ASGITransport does not run the lifespan
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bench", headers=headers)
    else:
        client = httpx.AsyncClient(base_url=a.url, headers=headers)

    print(f"{'conc':>5} {'req/s':>7} {'p50 s':>7} {'p95 s':>7} {'p99 s':>7} {'ttft s':>7} {'503':>5} {'other':>5}")
    async with client:
        for c in (int(x) for x in a.levels.split(",")):
            r = await level(client, c, a.requests, a.stream)
            print(f"{r['conc']:>5} {r['rps']:>7.2f} {r['p50']:>7.2f} {r['p95']:>7.2f} {r['p99']:>7.2f} "
                  f"{r['ttft']:>7.2f} {r['n503']:>5} {r['other']:>5}")


if __name__ == "__main__":
    asyncio.run(main())
