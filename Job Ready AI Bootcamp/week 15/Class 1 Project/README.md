# rag-api — Week 12 RAG Q&A Bot as a FastAPI Service

`/ask`, `/ask/stream`, `/upload`, `/sources`, `/health` endpoints; Pydantic v2 schemas; structured errors; Swagger at `/docs`.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000        # real: Week 12 pipeline + Ollama (ollama pull llama3.1:8b)
RAG_BACKEND=fake uvicorn app.main:app   # Ollama ছাড়া demo (০.২s নকল latency)
```

Open http://localhost:8000/docs. Windows PowerShell-এ env: `$env:RAG_BACKEND="fake"`.

```bash
curl -F "file=@hr.pdf" localhost:8000/upload
curl -X POST localhost:8000/ask -H "content-type: application/json" -d '{"question":"leave policy?"}'
```

## Test and benchmark

```bash
python -m pytest -q                       # 9 tests, Ollama/Chroma লাগে না
python benchmark/bench.py --fake          # in-process concurrency sweep
python benchmark/bench.py --url http://localhost:8000   # চলমান server (real Ollama)
```

## Layout

| File | Role |
| --- | --- |
| `app/main.py` | routes, middleware (request id + `X-Process-Time-Ms`), error handlers |
| `app/service.py` | **`RagService`** — প্রতিটা request-এর একমাত্র প্রবেশপথ (extension seam) |
| `app/backend.py` | `RealBackend` (Week 12 adapter) / `FakeBackend` (tests, benchmark) |
| `app/schemas.py` | Pydantic request/response models |
| `app/errors.py` | domain errors -> HTTP status |
| `app/settings.py` | env config (`RAG_` prefix: `WEEK12_DIR`, `CHROMA_DIR`, `MAX_UPLOAD_MB`, `BACKEND`) |

## How Week 12 is reused (decision)

Week 12 কোড **import করা হয়, কপি না**। `RealBackend.__init__` Week 12 root (`../../week 12/Class 2 Project`, override: `RAG_WEEK12_DIR`) `sys.path`-এর **শেষে** যোগ করে, কারণ Week 12 `config`/`src.*` top-level নামে import করে, আর তার streamlit `app.py` আমাদের `app/` package-এর সাথে নাম-সংঘর্ষ করবে (append করায় আমাদেরটা আগে resolve হয়)। Week 12-এর Chroma path override করে `./data/chroma_db`-এ আনা হয় (singleton `_config_instance` replace), তাই Week 12 কিছুই বদলায় না। Tradeoff: Week 12 folder সরালে path ভাঙবে; production-এ একে installable package করা উচিত।

## Extension seam for Class 30 (monitoring)

**Seam:** `app/service.py` -> `RagService._run()`. প্রতিটা `ask`/`ingest`/`sources` call এখান দিয়ে যায়, এবং শেষে `CallRecord(op, request_id, latency_s, ok, error, result)` প্রতিটা `service.observers` callable-কে দেওয়া হয় (observer ভাঙলে request ভাঙে না)।

```python
def log_cost(rec):   # Class 30
    if rec.op == "ask" and rec.ok:
        ...  # rec.result.answer থেকে token আনুমানিক গণনা, rec.latency_s, rec.request_id
service.observers.append(log_cost)   # get_service() (app/main.py)-এ RagService বানানোর পর
```

HTTP-স্তরের metric (status, total time) চাইলে `app/main.py::request_context` middleware। Token সংখ্যা আসল Ollama response-এ (`prompt_eval_count`, `eval_count`) আছে, কিন্তু Week 12 `generate()` তা ফেরত দেয় না — Class 30-কে `RealBackend.ask`-এ সেটা surface করতে হবে (Week 12 না ছুঁয়ে আলাদা wrapper)।

## Known limits

* `/ask/stream` fake-streaming (পুরো উত্তর তৈরির পর শব্দ ধরে পাঠায়); Week 12 `generate()` non-streaming।
* Concurrency ceiling = AnyIO threadpool (default 40) এবং Ollama-র নিজস্ব parallelism।
* `/upload` synchronous (index শেষ হলে response); বড় PDF-এ background job ভালো।
* Auth/rate-limit নেই (local-only)।
