# 📈 Offline Market Intelligence Team (CrewAI + FinBERT + Ollama)

## সহজ ভাষায় Project Overview

দুই agent-এর একটা local team: **News Scout Agent** local snapshot থেকে একটা ticker-এর জরুরি financial news বাছে (প্রতিটা headline-এর id আর FinBERT sentiment সহ), আর **Risk Analyst Agent** সেই ticker-এর **Risk Score** ব্যাখ্যা করে একটা investment memo লেখে। Risk Score নিজে (0-100) পুরোটা **code** হিসাব করে — volatility, drawdown, VaR আর news sentiment থেকে, documented bound আর weight দিয়ে। শেষে code agent-দের লেখা যাচাই করে (id আসল? সংখ্যা মেলে?) আর হয় একটা PDF memo publish করে, নয়তো মানুষের কাছে পাঠায়। দাম, news, sentiment, agent, PDF — সব local; internet লাগে শুধু optional `refresh`-এ।

`Course Module.pdf` (Week 16, Class 31 & 32) এদের "Researcher Agent" আর "Analyst Agent" বলে; এই project `projects.md`-এর নাম ব্যবহার করে।

### কোন Problem Solve করে?

Financial research-এ LLM-এর সবচেয়ে বিপজ্জনক ভুল হলো আত্মবিশ্বাসী ভুল সংখ্যা: "volatility 18%" যেখানে আসলে 31%। ছোট local model সংখ্যা হিসাব বা কপি করতে প্রায়ই ভুল করে, আর tool ডাকার বদলে নিজে result বানিয়ে ফেলে (Week 14-এর live finding)। এই project ভাগটা স্পষ্ট রাখে: **সংখ্যা code-এ, ব্যাখ্যা agent-এ, আর যাচাই আবার code-এ।** Agent Risk Score কখনো set বা adjust করে না।

### কেন Offline Snapshot?

Live market data প্রতিদিন বদলায়, API key/rate limit লাগে, আর class-এ সবার সংখ্যা আলাদা আসে। তাই project-এর সাথে একটা versioned **Offline Snapshot** আসে: ৮টা **fictional** company-র এক বছরের synthetic দাম (seeded geometric Brownian motion, দুটো crash event) আর হাতে-লেখা fictional headline। সবার মেশিনে একই সংখ্যা, test করা যায়। চাইলে `refresh` real ticker একই SQLite store-এ আনে।

> **Run করার নিয়ম:** venv বানিয়ে dependency install (CrewAI ভারী; FinBERT-এর জন্য `torch` আর `transformers`), `ollama pull llama3.1:8b`, তারপর Quick Start। প্রথম `risk`/`analyze`-এ FinBERT একবার (~440 MB) download হয়।

> ⚠️ **Educational project, synthetic data, not investment advice.**

---

![Python](https://img.shields.io/badge/python-3.11--3.13-blue.svg)
![CrewAI](https://img.shields.io/badge/crewai-1.15-ff5a50.svg)
![Ollama](https://img.shields.io/badge/ollama-llama3.1%3A8b-black.svg)
![FinBERT](https://img.shields.io/badge/sentiment-ProsusAI%2Ffinbert-yellow.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.162-009688.svg)
![Tests](https://img.shields.io/badge/tests-31%20passing-green.svg)

## ✨ Features

- 🗂️ **Offline Snapshot** — ৮টা fictional ticker × ২৫২ trading day, ৫৩টা fictional headline (`news:ACME:3`)। `data/make_snapshot.py` একই seed-এ byte-by-byte একই file দেয় (test আছে)।
- 🗄️ **Local data persistence** — stdlib `sqlite3`: `tickers`, `prices`, `news`, `sentiment` (FinBERT cache), `memos` (PDF index)। প্রথম run-এ snapshot নিজে load হয়।
- 📰 **FinBERT sentiment** — `ProsusAI/finbert`, lazy load (প্রথম cache miss-এ), per-headline cache (key = model + headline hash)।
- 🧮 **Deterministic Risk Score** — volatility, max drawdown, 95% historical VaR, recency-weighted net sentiment → 0-100 + low/medium/high band। Stdlib math, numpy লাগে না। শেষ দামের তারিখের পরের news বাদ (look-ahead bias নেই)।
- 🧑‍🤝‍🧑 **News Scout Agent → Risk Analyst Agent** — `Process.sequential`, explicit `context`, delegation বন্ধ, `max_iter=6`, `memory=False`।
- 🛡️ **Guardrails** — brief-এ বানানো id বা কম citation, memo-তে section না থাকলে task feedback সহ আবার চলে।
- ✅ **Code verification** — cited id আসল কিনা, memo-র Risk Score/volatility/drawdown/VaR computed মানের সাথে rounding-এর মধ্যে মেলে কিনা, required section আছে কিনা।
- 🌊 **CrewAI Flow** — `load_data` → `score_sentiment` → `compute_risk_score` (code) → `run_crew` (agents, ১ বার retry) → `gate` → `publish` (PDF + SQLite) | `needs_human`।
- ⚖️ **দুটো mode** — `grounded` (default: সংখ্যা task-এ বসানো) আর `tool` (Risk Analyst নিজে `get_risk_metrics` / `get_news_sentiment` ডাকে — failure demo)।
- 📄 **PDF investment memo** — fpdf2: disclaimer, metrics table, Risk Score, price chart (`polyline`, matplotlib ছাড়া), news brief, analyst explanation।
- 🌐 **FastAPI + Streamlit + CLI** — `/v1/risk` (LLM ছাড়া), `/v1/analyze`, `/v1/memo` (PDF); UI শুধু HTTP দিয়ে API ডাকে।
- 🔒 **Offline by design** — telemetry-র তিনটা flag crewai import-এর আগে; network শুধু optional `refresh`-এ।
- 🧪 **31 offline tests** — fake `BaseLLM`, fake sentiment, temp SQLite, `FakeBackend`; Ollama, FinBERT বা network লাগে না।

## 🚀 Quick Start

```bash
cd "Job Ready AI Bootcamp/week 16/Class 1-2 Project"
python -m venv venv && source venv/bin/activate          # Windows: venv\Scripts\activate ; Python 3.11-3.13
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU torch is enough for FinBERT
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama serve

python cli.py snapshot                 # load data/snapshot into data/market.db (also happens automatically)
python cli.py risk HELX                # deterministic Risk Score (FinBERT, no LLM)
python cli.py analyze ACME             # agents + verification + PDF in output/
python cli.py analyze ACME --mode tool # Risk Analyst must call tools itself (failure demo)

uvicorn api.main:app --port 8000       # http://localhost:8000/docs
streamlit run ui.py                    # UI calls the API over HTTP (MKT_API_URL, default http://localhost:8000)
MKT_BACKEND=fake uvicorn api.main:app  # canned answers: UI demo without Ollama/FinBERT

pytest -q                              # 31 tests, no Ollama / model download / network
```

Windows PowerShell-এ env var: `$env:MKT_BACKEND="fake"`। Sandbox-এ pytest-এর temp folder blocked থাকলে: `pytest -q --basetemp=.pytest_tmp`।

Optional live refresh (একমাত্র network step; বাকি সব এর পরেও offline চলে):

```bash
pip install yfinance feedparser
python cli.py refresh AAPL MSFT --period 1y      # real prices + RSS headlines into the SAME SQLite store
python cli.py risk AAPL
```

```bash
curl localhost:8000/v1/tickers
curl localhost:8000/v1/risk/HELX
curl -X POST "localhost:8000/v1/analyze/ACME?mode=grounded"
curl -o ACME-memo.pdf localhost:8000/v1/memo/ACME
```

## 📁 Project Structure

```
Class 1-2 Project/
├── api/                          # FastAPI service
│   ├── main.py                   # routes, request id, structured errors, analysis gate (503 + Retry-After)
│   ├── backend.py                # RealBackend (src/) / FakeBackend (tests, UI demo)
│   ├── schemas.py                # response models (= OpenAPI contract)
│   ├── errors.py                 # ServiceError -> 404 / 422 / 503
│   └── settings.py               # pydantic-settings, env prefix MKT_
├── ui.py                         # thin Streamlit UI (httpx -> API)
├── cli.py                        # snapshot | refresh | risk TICKER | analyze TICKER
├── config.py                     # telemetry flags (before crewai) + Risk Score bounds/weights
├── data/
│   ├── make_snapshot.py          # seeded GBM prices + authored fictional news
│   └── snapshot/                 # prices.csv, news.json, manifest.json (committed, versioned)
├── notebooks/01_exploration.ipynb# snapshot → metrics → sentiment → score → verifier → crew → PDF
├── src/
│   ├── store/db.py               # SQLite store (5 tables), snapshot loader
│   ├── store/refresh.py          # optional yfinance + RSS ingestion
│   ├── sentiment/finbert.py      # lazy FinBERT + per-headline cache
│   ├── risk/metrics.py           # vol, drawdown, VaR, net sentiment, Risk Score
│   ├── crew/team.py              # agents, tasks, guardrails, grounded | tool
│   ├── crew/tools.py             # get_risk_metrics, get_news_sentiment (BaseTool)
│   ├── flow/market_flow.py       # Flow: code → crew → verify → publish | needs_human
│   ├── verify/memo_check.py      # ids, quoted numbers, sections
│   ├── memo/pdf.py               # fpdf2 investment memo
│   └── llm/factory.py            # LLM("ollama/llama3.1:8b") + health check
└── tests/                        # test_pipeline.py (24), test_api.py (7), fakes.py
```

## 🔧 Architecture

```
                      ┌──── optional: cli.py refresh (yfinance + RSS) — the only network step ────┐
data/snapshot/ ──load──▶ SQLite data/market.db  (tickers · prices · news · sentiment · memos) ◀──┘
                                   │
ui.py ──HTTP──▶ api/main.py ──┬── GET  /v1/risk/{t}     → assess(): store → FinBERT → metrics (no LLM)
cli.py ───────────────────────┴── POST /v1/analyze/{t}  → MarketFlow ↓
┌──────────────────────────────── MarketFlow (CrewAI Flow, typed state) ─────────────────────────────────┐
│ @start load_data          ticker + news from SQLite                                          (code)    │
│ @listen score_sentiment   FinBERT per headline, cached in SQLite                             (code)    │
│ @listen compute_risk_score vol · drawdown · VaR · net sentiment → 0-100 Risk Score + band     (code)    │
│ @listen run_crew          for round in 1..2:                                                           │
│                             Crew(sequential, memory off):                                              │
│                               News Scout Agent ── 3-6 bullets [news:ACME:n] ── guardrail: ids, ≥2 cites │
│                               Risk Analyst Agent ─ memo sections ───────────── guardrail: sections, ids │
│                             verify_memo(): ids exist · quoted numbers match · sections   (code)       │
│                             ok? break : feedback → next round                                          │
│ @router gate ──"ok"──▶ publish: PDF (fpdf2) → output/ + memos row     ──"escalate"──▶ needs_human     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

`grounded` mode-এ Risk Analyst-এর task-এ code-এর সংখ্যা বসানো থাকে; `tool` mode-এ থাকে না, agent-কে tool ডাকতে হয়। দুই mode-এই tool বা task — সংখ্যা আসে একই code থেকে।

## 🧮 Risk Score Formula

প্রতিটা metric linear ভাবে 0-100 risk point-এ যায়: `points = clamp((value − lo) / (hi − lo), 0, 1) × 100`। তারপর `Risk Score = Σ weight × points` (weight-এর যোগফল 1)।

| Metric | কীভাবে মাপা | 0 points (lo) | 100 points (hi) | Weight |
|---|---|---|---|---|
| Annualized volatility | daily log return-এর sample stdev × √252 | 10% | 70% | 0.35 |
| Max drawdown | running peak থেকে সবচেয়ে বড় পতন | 5% | 60% | 0.25 |
| 95% historical VaR (1-day) | daily simple return-এর 5th percentile (linear interpolation), positive loss | 1% | 6% | 0.25 |
| Net news sentiment | Σ w·(positive − negative) / Σ w, w = 0.5^(age_days / 30); `as_of`-এর পরের news বাদ; news না থাকলে 0 | +1 | −1 | 0.15 |

Band: `< 35` low, `< 65` medium, বাকি high। সব মান `config.py: RiskConfig`-এ।

Snapshot-এর price metric (deterministic — সবার মেশিনে একই; sentiment আর তাই চূড়ান্ত Risk Score FinBERT-এর উপর নির্ভর করে):

| Ticker | Company (fictional) | Volatility | Max drawdown | VaR 95% |
|---|---|---|---|---|
| MRDN | Meridian Pantry Foods | 14.0% | 11.1% | 1.57% |
| KRYO | Kryo Cold Chain Logistics | 18.1% | 18.9% | 1.88% |
| ZEPH | Zephyr Wind Power | 23.4% | 24.4% | 2.59% |
| ACME | Acme Robotics | 27.9% | 18.3% | 2.51% |
| VALT | Valtara Payments | 30.4% | 53.9% | 3.28% |
| LUMA | Luma Optics | 37.5% | 44.4% | 3.63% |
| QBIT | Qubit Compute | 55.6% | 47.1% | 5.96% |
| HELX | Helix Biotherapeutics | 66.1% | 65.4% | 4.52% |

## 🤖 Model Choices

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`llama3.1:8b`** (course standard, ADR 0006) | দুই agent-এর crew, ছোট memo | Laptop-এ চলে, offline | CrewAI-র ভেতরে tool নির্ভরযোগ্যভাবে ডাকে না — তাই default `grounded` |
| `llama3.2:3b` (fallback) | 8 GB RAM | হালকা, দ্রুত | সংখ্যা কপি আর format-এ আরও বেশি ভুল |
| বড় local model (70B) | Tool mode নির্ভরযোগ্য করা | Instruction following ভালো | ~40 GB+ memory |
| Cloud API | Production research desk | সবচেয়ে নির্ভরযোগ্য | Offline না — data বাইরে যায় |

| Sentiment model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`ProsusAI/finbert`** (chosen, ADR 0009) | Financial headline | Finance domain, native 3-class, probability দেয় | ~440 MB, শুধু English |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` (Week 10) | Tweet / review | 3-class, পরিচিত | Tweet domain — finance phrasing-এ domain shift |
| `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | হালকা financial sentiment | ছোট, দ্রুত | Community model, calibration যাচাই করতে হয় |
| `llama3.1:8b` zero-shot | Label-ছাড়া নতুন domain | Extra model লাগে না | ধীর, non-deterministic, calibrated probability নেই — deterministic score-এ বসে না |

## ⚙️ Configuration

API settings (`MKT_` env vars, `api/settings.py`):

| Variable | Default | মানে |
|---|---|---|
| `MKT_BACKEND` | `real` | `fake` = canned উত্তর (tests, Ollama/FinBERT ছাড়া UI demo) |
| `MKT_DB_PATH` | `data/market.db` | SQLite store (git-ignored); CLI আর notebook-ও এটা মানে |
| `MKT_OUTPUT_DIR` | `output/` | PDF memo (git-ignored) |
| `MKT_SNAPSHOT_DIR` | `data/snapshot` | Offline Snapshot |
| `MKT_OLLAMA_HOST` / `MKT_MODEL` | `http://localhost:11434` / `llama3.1:8b` | Ollama |
| `MKT_MAX_CONCURRENT_ANALYSES` | `1` | একসাথে কয়টা crew run; একটা Ollama-য় বেশি দিলে শুধু queue লুকায় |
| `MKT_QUEUE_TIMEOUT_S` | `5` | slot-এর অপেক্ষা, তারপর 503 + `Retry-After` |
| `MKT_API_URL` (UI) | `http://localhost:8000` | Streamlit UI কোন API ডাকবে |

Method settings (`config.py`):

| Setting | Default | কেন |
|---|---|---|
| `llm.temperature` | `0.1` | সংখ্যা কপি আর format কম ভাঙে |
| `crew.max_iter` | `6` | CrewAI default ২৫ — ছোট model-এ লম্বা loop |
| `crew.guardrail_retries` | `2` | Guardrail fail হলে task-এর retry |
| `crew.flow_max_rounds` | `2` | Verification fail হলে পুরো crew-এর একবার retry (hard cap) |
| `crew.news_items` | `8` | News Scout-কে সর্বশেষ কয়টা headline (context ছোট) |
| `risk.bounds` / `risk.weights` / `risk.band_cuts` | উপরের table | Risk Score-এর policy |
| `risk.half_life_days` | `30` | পুরোনো news-এর weight কত দ্রুত কমে |
| `verify.required_sections` | Risk Summary, Key Drivers, News Impact | Memo-র কাঠামো |

## 🔌 API

| Method | Path | Success | Errors |
|---|---|---|---|
| GET | `/health` | `200 {"status":"ok"}` | — |
| GET | `/ready` | `200 {status, backend, detail, tickers}` | `503` store খালি / Ollama নেই |
| GET | `/v1/tickers` | `200 {"tickers": [...]}` | — |
| GET | `/v1/risk/{ticker}` | `200 {risk_score, band, metrics, components, weights, news[]}` | `404 unknown_ticker`, `422`, `503 sentiment_unavailable` |
| POST | `/v1/analyze/{ticker}?mode=grounded\|tool` | `200 {outcome, rounds, risk, brief, memo, verification, memo_available}` | `404`, `422`, `503 llm_unavailable / overloaded` |
| GET | `/v1/memo/{ticker}` | `200 application/pdf` (সর্বশেষ published memo) | `404 memo_not_found` |

সব error-এর একই আকার: `{error, detail, request_id}`; প্রতিটা response-এ `X-Request-ID` আর `X-Process-Time-Ms`।

## 📏 What to Measure on Your Machine

এই project যে machine-এ বানানো, সেখানে Ollama বা GPU ছিল না, তাই কোনো live সংখ্যা এখানে নেই (ADR 0008)। নিজে মেপে এই table ভরুন:

| মাপুন | কীভাবে | কী দেখবেন |
|---|---|---|
| FinBERT label কতটা ঠিক | `python cli.py risk HELX` বা notebook §4 | ৫৩টা headline-এ নিজের label বনাম FinBERT; "explores strategic alternatives"-এর মতো নিরপেক্ষ শব্দের খারাপ খবর |
| Cold vs warm sentiment | প্রথম `risk` বনাম দ্বিতীয় `risk` (একই ticker) | প্রথমবার model load + download; পরে SQLite cache থেকে ~ms |
| Grounded publish rate | `analyze` প্রতিটা ticker-এ ৩ বার | কত run `published`, কত `needs_human`; কয়টায় round 2 লাগল |
| Tool mode failure | `analyze ACME --mode tool` ৩ বার | Agent কি tool ডেকেছে? `wrong_numbers`-এ কী এসেছে? |
| Latency আর token | CLI output-এর seconds, Flow state-এর `token_usage` | এক analysis-এ কত সেকেন্ড আর token |
| Verifier false alarm | needs_human memo-গুলো পড়ুন | সত্যিই ভুল সংখ্যা, নাকি "volatility contributes 35%" ধরনের false alarm |

## ⚠️ Known Limits

- **Not investment advice.** Educational project। Risk Score একটা শেখানোর জন্য বানানো formula — কোনো regulatory বা industry standard না।
- **Synthetic data.** Snapshot-এর সব company, ticker, দাম আর headline বানানো; কোনো ticker real listing-এর সাথে মিলে গেলে কাকতালীয়। Synthetic calendar-এ exchange holiday নেই।
- **FinBERT domain caveats** — analyst-style English sentence-এ trained; slang, sarcasm, অন্য ভাষা, আর "strategic alternatives"-এর মতো নিরপেক্ষ শব্দের খারাপ খবরে ভুল করতে পারে। Headline-এর বাইরে context দেখে না।
- **Historical metrics পিছনে তাকায়** — volatility, drawdown, VaR অতীতের দাম থেকে; এক বছরের data-য় tail risk কম ধরা পড়ে, আর ভবিষ্যতের crash আগে দেখে না।
- **Number check একটা proxy** — metric নামের পরে ~২৫ character-এর মধ্যে প্রথম সংখ্যা মেলায়। "volatility contributes 35%" false alarm দেয় (এক round retry), আর metric নাম ছাড়া লেখা ভুল সংখ্যা ধরে না। কঠোর করতে: structured JSON memo।
- **Agent-রা as_of-এর পরের headline দেখতে পারে** — Risk Score সেগুলো বাদ দেয়, কিন্তু refresh-এর real ticker-এ News Scout সেগুলো brief-এ আনতে পারে।
- Yahoo-র per-ticker RSS URL অনানুষ্ঠানিক; বদলালে `--rss` দিয়ে অন্য feed দিন।
- Single Ollama — একসাথে একটাই analysis (gate), প্রতিটা ১-৩ মিনিট।
- CrewAI Flow console-এ বড় banner ছাপে; Windows console-এ emoji encoding warning এড়াতে `PYTHONIOENCODING=utf-8`।
