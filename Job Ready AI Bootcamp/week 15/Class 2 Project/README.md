# rag-api + Monitoring (Class 30)

Class 29-এর `rag-api`-র self-contained কপি + প্রতি query-র **token, cost, latency, quality score** tracking। Class 1 folder runtime-এ লাগে না (Week 12 pipeline আগের মতোই import হয়)।

## Architecture

```
/ask -> RagService._run() -> CallRecord -> service.observers
                                              |
                                  Monitor.observer (app/monitor.py)
                                   |- metrics: tokens, equivalent cost, grounding score
                                   |- SQLite  data/monitor.db   (সবসময়, source of truth)
                                   '- sinks -> Langfuse (শুধু env থাকলে)
GET /stats      JSON summary + drift      GET /dashboard   ছোট HTML table
```

| File | Role |
| --- | --- |
| `app/usage.py` | `ollama.chat` wrap করে `prompt_eval_count`/`eval_count` ধরে (Week 12 অক্ষত) |
| `app/pricing.py` | **illustrative** equivalent cloud price table (`RAG_PRICE_TABLE_JSON` দিয়ে override) |
| `app/quality.py` | grounding-overlap score + "not found" guardrail detection + rolling-window drift check |
| `app/monitor.py` | observer, SQLite log, `/stats` aggregation |
| `app/langfuse_obs.py` | optional Langfuse sink; env/SDK না থাকলে `None` (graceful) |
| `docker-compose.yml`, `.env.example` | self-hosted Langfuse |

## Run (Docker ছাড়া, SQLite only)

```bash
pip install -r requirements.txt
RAG_BACKEND=fake uvicorn app.main:app --port 8000     # fake backend; real: ollama pull llama3.1:8b, RAG_BACKEND=real
curl -X POST localhost:8000/ask -H "content-type: application/json" -d '{"question":"leave policy?"}'
curl localhost:8000/stats      # বা browser-এ /dashboard
```

## Langfuse (self-hosted) যোগ করা

```bash
cp .env.example .env                 # secret বদলান
docker compose up -d                 # ~2-3 মিনিট; UI http://localhost:3000 (admin@example.com / .env-এর password)
export LANGFUSE_PUBLIC_KEY=pk-lf-local-demo LANGFUSE_SECRET_KEY=sk-lf-local-demo LANGFUSE_HOST=http://localhost:3000
uvicorn app.main:app --port 8000     # এখন প্রতি /ask Langfuse-এ generation + "grounding" score
docker compose down                  # (-v দিলে data মুছে যায়)
```

## Tests

```bash
python -m pytest -q       # 16 tests; Ollama/Docker/Langfuse/network লাগে না
```

Windows-এ `pytest tmp_path` PermissionError দিতে পারে, তাই monitoring tests `tempfile` ব্যবহার করে।

## সীমাবদ্ধতা (honest)

* **Cost illustrative**: local model-এর আসল খরচ ০; table-এর দাম গোল সংখ্যা, আসল দাম নয়।
* **Quality score proxy**: token overlap — paraphrase-এ কম দেখায়, context ভুল হলে ভুলও "grounded"। Ground truth নয়।
* **Drift check সরল**: প্রথম N বনাম শেষ N query-র mean (default N=20); statistical test নয়।
* Langfuse span-এর duration প্রায় শূন্য (observer request শেষে চলে); আসল latency `metadata.latency_s`-এ।
* Streaming `/ask/stream` পথেও `ask` log হয়, কিন্তু ইউজার-দেখা stream-latency আলাদা মাপা হয় না।
* SQLite এক-process; multi-worker হলে Postgres/Langfuse ব্যবহার করুন।
* **Unverified**: Langfuse stack চালানো হয়নি (শুধু `docker compose config` pass); image tag `:4` ও `langfuse-worker` image official docs/ctx7 থেকে নেওয়া, pull করে দেখা হয়নি। Langfuse SDK real server-এর বিপরীতে চালানো হয়নি (stub client দিয়ে test)। Real Ollama token capture-ও আসল model-এ চালানো হয়নি।
