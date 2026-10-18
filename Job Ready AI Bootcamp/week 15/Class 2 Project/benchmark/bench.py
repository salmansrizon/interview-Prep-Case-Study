"""Throughput-vs-latency concurrency sweep।

দুই mode:
  python benchmark/bench.py --fake                  # in-process, Ollama ছাড়া (০.২s নকল LLM latency)
  python benchmark/bench.py --url http://localhost:8000   # চলমান server-এর বিরুদ্ধে (real Ollama)

প্রতিটা concurrency level-এ N টা /ask পাঠিয়ে মাপে:
  throughput = completed requests / wall seconds        (সিস্টেম প্রতি সেকেন্ডে কত করতে পারে)
  latency p50/p95 = একটা request-এর নিজের অপেক্ষার সময় (ব্যবহারকারীর অভিজ্ঞতা)
HIGHLIGHTS: concurrency বাড়ালে throughput বাড়ে যতক্ষণ না worker/threadpool (বা Ollama
নিজে) saturate হয়; তার পর throughput থেমে যায় কিন্তু latency (queueing) বাড়তেই থাকে —
এই "knee" খোঁজাই benchmark-এর উদ্দেশ্য।
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def one(client, i):
    t0 = time.perf_counter()
    r = await client.post("/ask", json={"question": f"question {i}"}, timeout=300)
    return time.perf_counter() - t0, r.status_code


async def level(client, concurrency, total):
    sem = asyncio.Semaphore(concurrency)

    async def guarded(i):
        async with sem:
            return await one(client, i)

    t0 = time.perf_counter()
    res = await asyncio.gather(*(guarded(i) for i in range(total)))
    wall = time.perf_counter() - t0
    lats = sorted(x[0] for x in res)
    errs = sum(1 for x in res if x[1] != 200)
    return total / wall, statistics.median(lats), lats[int(0.95 * (len(lats) - 1))], errs


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--levels", default="1,2,4,8,16,32,64")
    ap.add_argument("--requests", type=int, default=64, help="total requests per level")
    a = ap.parse_args()
    if not (a.url or a.fake):
        ap.error("--url অথবা --fake দিন")

    if a.fake:
        from app.backend import FakeBackend
        from app.main import app, get_service
        from app.service import RagService

        svc = RagService(FakeBackend(delay=0.2))
        app.dependency_overrides[get_service] = lambda: svc
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    else:
        client = httpx.AsyncClient(base_url=a.url)

    print(f"{'conc':>5} {'req/s':>8} {'p50 s':>8} {'p95 s':>8} {'errors':>6}")
    async with client:
        for c in [int(x) for x in a.levels.split(",")]:
            rps, p50, p95, errs = await level(client, c, a.requests)
            print(f"{c:>5} {rps:>8.1f} {p50:>8.3f} {p95:>8.3f} {errs:>6}")


if __name__ == "__main__":
    asyncio.run(main())
