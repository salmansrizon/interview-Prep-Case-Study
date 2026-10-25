# 📈 RAG API Monitoring: Metrics, Traces, Quality & SLOs (Prometheus + OpenTelemetry + Ollama)

## সহজ ভাষায় Project Overview

Class 1-এর RAG API চলছে। এখন প্রশ্ন: **চলার সময় কী হচ্ছে?** কত দ্রুত উত্তর দিচ্ছে, কত token যাচ্ছে, উত্তরগুলো কি document-এর সাথে মেলে, আর কখন রাত তিনটায় কাউকে ঘুম থেকে তুলতে হবে। এই project Class 1-এর service-এর একটা self-contained কপি নিয়ে তার উপর পুরো monitoring stack বসায় — কোনো RAG code না বদলে, শুধু Class 1-এর রেখে যাওয়া observer seam-এ।

### কোন Problem Solve করে?

LLM app-এর সবচেয়ে বিপজ্জনক ব্যর্থতা HTTP 200 দেয়: ভুল সংখ্যা, বানানো দাবি, ভুল citation, ভুল করে "not found"। সাধারণ API monitoring (status code, latency) এগুলো দেখে না। এই project তিন স্তরে দেখে — প্রতিটা request-এ সস্তা check, offline LLM judge, আর মানুষের feedback — আর তার সাথে latency/availability-র SLO burn-rate alert।

### কেন এই tool-গুলো?

**Prometheus** = industry-standard metrics + alerting (single binary)। **OpenTelemetry** = vendor-neutral trace format — একই span Phoenix, Langfuse বা যেকোনো OTLP backend-এ যায়। **SQLite** = per-request log, Docker ছাড়াই। **Phoenix** = এক process-এ local trace UI।

> **Run করার নিয়ম:** venv, `pip install -r requirements.txt`, `ollama pull llama3.1:8b`। Prometheus আর Phoenix optional — না থাকলেও `/metrics`, `/v1/stats`, `/v1/dashboard` সব কাজ করে।

---

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.162-009688.svg)
![Prometheus](https://img.shields.io/badge/prometheus-3.15-e6522c.svg)
![OpenTelemetry](https://img.shields.io/badge/opentelemetry-1.45-425cc7.svg)
![Ollama](https://img.shields.io/badge/ollama-llama3.1%3A8b-black.svg)
![Tests](https://img.shields.io/badge/tests-43%20passing-green.svg)

## ✨ Features

- 🔌 **Observer seam** — `RagService` প্রতিটা call শেষে `CallRecord` দেয়; `Monitor.observe` একবার signal হিসাব করে SQLite + Prometheus + OTel-এ পাঠায়। Observer ভাঙলেও request ভাঙে না।
- 📊 **Prometheus `/metrics`** — চার golden signal, RAG stage ধরে: `rag_stage_duration_seconds{stage=embed|search|llm|prefill|decode}`, `rag_ttft_seconds`, `rag_tokens_total`, `rag_answers_total{outcome}`, `rag_llm_in_flight`, `rag_queue_wait_seconds`, `rag_llm_cold_starts_total`। LLM-আকারের bucket (120 s পর্যন্ত), label-এ route template (`unmatched` অজানা path-এ)।
- 🧾 **SQLite request log** — প্রতিটা request-এর latency breakdown, token, prefill/decode, top distance, chunk ids, quality flag, judge verdict, feedback। WAL mode; `RAG_LOG_TEXT=false` দিলে প্রশ্ন/উত্তর শুধু hash।
- ✅ **Cheap quality signals (প্রতি request)** — citation valid, numbers supported, lexical support, truncated।
- ⚖️ **Offline LLM judge** — `python -m app.telemetry.judge`: claim → passage থেকে হুবহু evidence → verdict, আর code-এ quote check। Request path-এর বাইরে।
- 🔭 **OpenTelemetry traces** — request-প্রতি span tree (queue → embed → retrieve → `chat llama3.1:8b` as CLIENT span), GenAI semantic-convention attribute + OpenInference kind; প্রশ্ন/উত্তর text opt-in (`RAG_TRACE_CONTENT`); OTLP/HTTP-এ Phoenix বা Langfuse।
- 🚨 **SLO + burn rate** — availability (99%) আর latency (95% < 10 s); 5m/30m/1h/6h burn rate, SRE Workbook-এর multiwindow alert — `/v1/stats`-এ আর `monitoring/alerts.yml`-এ (promtool unit test সহ)।
- 👍 **Feedback** — `POST /v1/feedback {request_id, vote}`; খারাপ trace খোঁজার জন্য, quality score হিসেবে না।
- 🖥️ **`/v1/dashboard`** — একটা server-rendered page: tiles, burn-rate table, recent requests (flags, judge, feedback); প্রতি 10 s refresh; সব user text escape করা।
- 🧪 **43 offline tests** — Class 1-এর 24 + monitoring-এর 19 (quality, SLO math, PII, metrics, 503 logging, XSS, trace tree, opt-in content, cold starts, judge)।

## 🚀 Quick Start

```bash
cd "Job Ready AI Bootcamp/week 15/Class 2 Project"
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama serve

uvicorn app.main:app --port 8000
curl -F "file=@samples/employee_handbook.pdf" localhost:8000/v1/documents
curl -X POST localhost:8000/v1/ask -H "content-type: application/json" -d '{"question":"How many sick days do employees get?"}'

curl localhost:8000/metrics | grep ^rag_          # Prometheus text
curl localhost:8000/v1/stats                      # JSON summary + SLO burn rates
open http://localhost:8000/v1/dashboard           # HTML dashboard
python -m app.telemetry.judge --limit 20          # offline judge over the log
pytest -q                                          # 43 tests, Ollama লাগে না
```

**Traces → Phoenix (local, 1 process):**

```bash
pip install arize-phoenix && phoenix serve        # আলাদা venv-এ রাখা ভালো; UI http://localhost:6006
RAG_OTLP_ENDPOINT=http://localhost:6006/v1/traces RAG_TRACE_CONTENT=true uvicorn app.main:app --port 8000
```

**Metrics → Prometheus:**

```bash
prometheus --config.file=monitoring/prometheus.yml          # UI http://localhost:9090
promtool check config monitoring/prometheus.yml
promtool test rules monitoring/alerts_test.yml              # alert-এর unit test
```

**Langfuse (optional, Docker):** `docker compose up -d`, তারপর `RAG_OTLP_ENDPOINT=http://localhost:3000/api/public/otel/v1/traces` আর `RAG_OTLP_HEADERS="Authorization=Basic <base64(pk:sk)>"` — এই repo-তে Docker ছাড়া বানানো, তাই **live verify করা হয়নি**।

## 📁 Project Structure

```
Class 2 Project/
├── app/
│   ├── main.py, service.py, backend.py, schemas.py, errors.py, settings.py   # Class 1 service (+ /metrics, stats, dashboard, feedback)
│   ├── rag/                     # self-contained RAG core (same as Class 1, + ChromaStore.texts for the judge)
│   └── telemetry/
│       ├── monitor.py           # the observer: CallRecord -> row -> SQLite + Prometheus + trace
│       ├── quality.py           # cheap per-request signals (citations, numbers, lexical, truncation)
│       ├── store.py             # SQLite log, summary, SLO burn rates
│       ├── metrics.py           # Prometheus registry + metric definitions
│       ├── tracing.py           # OTel spans from measured stage timings (GenAI + OpenInference attrs)
│       ├── judge.py             # offline LLM judge CLI
│       └── dashboard.py         # HTML page
├── monitoring/
│   ├── prometheus.yml           # scrape config
│   ├── alerts.yml               # recording rules + multiwindow burn-rate alerts + quality tickets
│   └── alerts_test.yml          # promtool unit tests for the alerts
├── notebooks/01_monitoring.ipynb   # live walkthrough
├── docker-compose.yml           # optional self-hosted Langfuse (unverified here)
├── benchmark/, samples/, tests/, Dockerfile, requirements.txt
```

## 🏗️ Architecture

```
request ─▶ middleware ─▶ rag_http_requests_total{method,route,status}, rag_request_duration_seconds
              │
              ▼
          RagService ── CallRecord(op, latency, queue_wait, result, question) ──▶ observers
                                                                                   │
                                                                     Monitor.observe(rec)
                                                    ┌──────────────────┬──────────┴──────────┐
                                           quality.assess()     metrics (Prometheus)   tracing (OTel)
                                                    │                  │                      │
                                          SQLite request log      GET /metrics          OTLP/HTTP
                                           │        │                  │                      │
                                  /v1/stats  /v1/dashboard        Prometheus ──▶ alerts.yml   Phoenix / Langfuse
                                           │
                               judge CLI (offline) writes verdicts back
```

## 🧪 Quality: Three Tiers

| Tier | Signal | কখন | খরচ | কী ধরে | কী ধরে না |
|---|---|---|---|---|---|
| 1 | outcome, citation_valid, numbers_supported, lexical_support, truncated | প্রতি request | ~০ | ভুল `[n]`, passage-এ অনুপস্থিত সংখ্যা, কাটা উত্তর | অন্য জায়গায় থাকা সংখ্যা ("15:00"), বানানো দাবি |
| 2 | NLI/HHEM groundedness | — (এখানে নেই) | CPU model | unsupported claim | — |
| 3 | LLM judge (claim → evidence → verdict + quote check) | offline, sampled | ~৪ s/উত্তর | বানানো দাবি, ভুল সংখ্যা | self-preference bias (একই model) |
| — | user 👍/👎 | ঐচ্ছিক | ০ | কোন trace দেখতে হবে | বেশিরভাগ user vote দেয় না |

## 📊 Live Results (MacBook, Ollama `llama3.1:8b`, Phoenix 20.26.0, Prometheus 3.15.1)

| Check | Result |
|---|---|
| ১২টা request (৮ on-topic, ২ off-topic, ১ repeat, ১ stream) | 8 answered, 3 not_found, 1 cached; citation valid 100%, numbers supported 100% |
| Latency (ask) | p50 1.25 s, p95 2.39 s; stream TTFT 190 ms; decode p50 19.5 tok/s; LLM ≈98% of time |
| Not-found review | **"Is MFA required for the VPN?" → not found (distance 27.06 > 26.0)**, যদিও handbook-এ আছে; "multi-factor authentication" লিখলে 17.77 → answered। ছোট embedding model acronym মেলায় না |
| Cheap checks vs planted errors | "18 sick days" ধরা পড়ে; **"15 sick days" পার পায়** (অন্য passage-এ "15:00"); "cash out" দাবিও পার পায় |
| Offline judge, final design | **11/11 ঠিক** (৮টা আসল উত্তর supported; "15 days" আর "cash out" unsupported; "twelve" paraphrase supported); ≈3.9 s/উত্তর। Notebook run: 12/12 |
| Phoenix | 43 span, 13 trace; LLM / RETRIEVER / EMBEDDING / CHAIN kind চেনে; GenAI token attribute দেখায় |
| Prometheus | scrape `up=1`; `histogram_quantile` llm p50 1.25 s; ১২টা rule healthy; `promtool test rules` pass |
| 503 burst (queue timeout 0.2 s, 3 wave × 6) | 15/18 → 503; Prometheus 5m error ratio 0.50 → `RagErrorBudgetFastBurn` + `SlowBurn` **pending**; `/v1/stats` burn 60× → `page_fast`, `page_slow` |

### Judge design: তিনটা চেষ্টা, মাপা ফল

| Design (একই llama3.1:8b) | ফল (১১টা উত্তর, ২টা ভুল planted) |
|---|---|
| `{"supported": bool, "unsupported_claim": str}` | "cash out" উত্তরে `supported: true` অথচ claim-এ সেটাই লেখা — নিজের সাথে অসঙ্গত |
| আগে unsupported claim-এর তালিকা, পরে verdict | সব claim-ই তালিকায় → ১১/১১ fail |
| **claim → হুবহু evidence → verdict, + code-এ quote check** | **১১/১১ ঠিক** |

শিক্ষা: ছোট judge-এ prompt/schema-র গঠন ফলাফল ০ থেকে ১০০%-এ নিয়ে যায় — judge-কেও eval করতে হয়।

## ⚙️ Configuration (Class 1-এর সব `RAG_` var, আর)

| Variable | Default | মানে |
|---|---|---|
| `RAG_MONITOR_DB` | `data/monitor.db` | SQLite request log |
| `RAG_LOG_TEXT` | `true` | `false` = প্রশ্ন/উত্তর শুধু SHA-256 hash (PII) |
| `RAG_OTLP_ENDPOINT` | — | OTLP/HTTP traces URL; না দিলে tracing বন্ধ |
| `RAG_OTLP_HEADERS` | — | `k=v,k2=v2` (যেমন Langfuse-এর Basic auth) |
| `RAG_TRACE_CONTENT` | `false` | `true` = প্রশ্ন/উত্তর span-এ (OTel spec: content opt-in, PII) |
| `RAG_SLO_AVAILABILITY` | `0.99` | 503 সহ ব্যর্থতা ≤1% |
| `RAG_SLO_LATENCY_S` / `RAG_SLO_LATENCY_TARGET` | `10` / `0.95` | ৯৫% উত্তর < 10 s |

## 🔌 Endpoints Added

| Method | Path | কী দেয় |
|---|---|---|
| GET | `/metrics` | Prometheus text (auth নেই — internal network-এ রাখুন) |
| GET | `/v1/stats?window_s=3600` | outcomes, rates, p50/p95, TTFT, tokens, GPU-seconds, quality, feedback, SLO burn + alerts |
| GET | `/v1/dashboard` | HTML (API key লাগলে `X-API-Key`) |
| POST | `/v1/feedback` `{request_id, vote: 1\|-1, comment?}` | `204`; অজানা id → `404` |

## 🧭 Design Decisions

- **Post-hoc spans** — service আগেই প্রতিটা stage-এর সময় মাপে; request শেষে সেই মাপ থেকে explicit timestamp দিয়ে span। RAG core-এ tracing code নেই। Ceiling: stage-গুলো পরপর ধরা হয় (এখানে সত্যিই পরপর)।
- **`/metrics` plain route** — `make_asgi_app()` mount করলে `/metrics` → 307 → `/metrics/`; plain route-এ সেই ফাঁদ নেই।
- **`or vector(0)` recording rule-এ** — কখনো 5xx না হলে 5xx series-ই থাকে না, ratio *খালি* হয় (0 না)। Live run-এ এটা ধরা পড়েছিল।
- **Disconnect ≠ error** — client stream মাঝপথে ছাড়লে outcome `disconnected`, availability SLO-তে গোনা হয় না। Stream-এর 503 (slot না পাওয়া)-ও log হয়।
- **Judge offline** — Ollama-য় একটাই slot; request-এর ভেতরে judge = user latency দ্বিগুণ।
- **Quality → ticket, কখনো page না** — sample ছোট, noise বেশি; page শুধু user-এর ব্যথায় (availability/latency burn)।
- **একই model judge** — course-এ একটাই model; self-preference bias স্বীকার করে score-কে trend হিসেবে পড়া।

## ⚠️ Known Limits

- SQLite আর in-process Prometheus registry এক process-এর; একাধিক replica-য় Postgres/ClickHouse আর প্রতিটা replica scrape (Prometheus যোগ করে) লাগবে।
- Observer event loop-এ চলে, SQLite write কয়েক ms block করে; বেশি traffic-এ queue + background writer।
- `numbers_supported` শুধু দেখে সংখ্যাটা passage-এ *কোথাও* আছে কিনা; কথায় লেখা সংখ্যা দেখে না।
- Judge-এর ১১/১১ একটা ছোট sample-এ; নিজের data-য় মানুষের label-এর সাথে মিলিয়ে দেখুন।
- Langfuse compose আর OTLP endpoint এই repo-তে live চালানো হয়নি (Docker daemon ছিল না)।
- `/metrics`-এ auth নেই; public internet-এ খোলা রাখবেন না।
