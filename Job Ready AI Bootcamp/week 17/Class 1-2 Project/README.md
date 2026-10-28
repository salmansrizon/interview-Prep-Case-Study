# 🩺 Privacy-First Healthcare Assistant (Local RAG + Guardrails)

## সহজ ভাষায় Project Overview

একটা কাল্পনিক clinic — **Greenfield Community Clinic** — এর handbook থেকে প্রশ্নের উত্তর দেওয়া assistant। জ্বর, পানিশূন্যতা, ORS, blood pressure, diabetes, asthma, ছোট পোড়া-কাটা, ওষুধের নিরাপত্তা, emergency লক্ষণ, vaccination আর clinic-এর নিয়ম (সময়, appointment, record privacy, telehealth) — ৩০টা page, ১২৮টা section। সব local: ChromaDB + `paraphrase-MiniLM-L3-v2` embedding + `llama3.1:8b` (Ollama)। প্রশ্ন, উত্তর বা log — কিছুই মেশিনের বাইরে যায় না।

### কোন Problem Solve করে?

Medical প্রশ্নে সাধারণ RAG bot-এর চারটা বিপদ: (১) রোগী নাম-phone-NID লিখে ফেলে, আর সেটা log-এ থেকে যায়; (২) "বুকে ব্যথা" লিখলে bot handbook খুঁজে শান্ত গলায় উত্তর দেয়, যখন দরকার ছিল "এখনই 999"; (৩) "কত mg খাব?" প্রশ্নে model নিজে থেকে dose বানায়; (৪) handbook-এ নেই এমন প্রশ্নে model নিজের স্মৃতি থেকে আত্মবিশ্বাসী ভুল বলে। এই project চারটাই **code-এ, একটা নির্দিষ্ট ক্রমে** আটকায়, আর প্রতিটা উত্তরে একটা `route` দেয়: `answered | abstained | escalated | refused`।

### কেন Local?

Health data সবচেয়ে সংবেদনশীল data-র একটা। Cloud API মানে প্রতিটা প্রশ্ন তৃতীয় পক্ষের server-এ যায়। এখানে model weight একবার download হয়, তারপর inference, vector DB আর log সব এই মেশিনে (CONTEXT.md-এর "Local Model")। Chroma আর Streamlit-এর telemetry-ও বন্ধ।

> **Run করার নিয়ম:** venv বানিয়ে dependency install, `ollama pull llama3.1:8b`, তারপর Quick Start। Ollama ছাড়া পুরো আচরণ দেখতে `HC_BACKEND=fake`।
>
> ⚠️ Educational, fictional clinic handbook; not medical advice. এটা medical device না।

---

![Python](https://img.shields.io/badge/python-3.11--3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.162-009688.svg)
![Ollama](https://img.shields.io/badge/ollama-llama3.1%3A8b-black.svg)
![ChromaDB](https://img.shields.io/badge/chromadb-1.5-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.66-red.svg)
![Tests](https://img.shields.io/badge/tests-67%20passing-green.svg)

## ✨ Features

- 🕵️ **PII redaction (সবার আগে)** — email, Bangladesh mobile (`01X…`, `+880…`, বাংলা অঙ্ক `০১৭…`), international number, NID-এর মতো অঙ্ক, `GCC-` MRN, জন্মতারিখ, "my name is…"/"আমার নাম…"। পরের সব ধাপ শুধু redacted text দেখে; response-এ `redaction_count`।
- 🚨 **Red-Flag Escalation** — বুকে ব্যথা, শ্বাসকষ্ট, stroke, ভারী রক্তপাত, আত্মহত্যার চিন্তা, anaphylaxis, অজ্ঞান/খিঁচুনি, বিষক্রিয়া, ৩ মাসের কম শিশুর জ্বর… Retrieval আর LLM-এর **আগে** deterministic rule; উত্তর একটা আগে লেখা 999 message, model কিছু লেখে না।
- 🙅 **Diagnosis/dosing refusal** — "do I have…", "how many mg…", "can I double my dose…" → স্থির refusal + clinician/pharmacist; সাথে code-এর বাছা ১-২টা সাধারণ handbook section।
- 🤐 **Abstention, দুই ধাপে** — সবচেয়ে কাছের section-ও `HC_MAX_DISTANCE`-এর বাইরে হলে LLM ডাকা হয় না; LLM উত্তর দিলে code `[n]` যাচাই করে, valid citation না থাকলে উত্তর যায় না।
- 📌 **Exact section citations** — `kb:dehydration:3` = `dehydration.md`-এর ৩য় section; PDF handbook-এও একই id ছাপা।
- 🔁 **KB sync** — ingest upsert করে আর handbook থেকে সরানো section মুছে দেয় (পুরোনো medical guidance retrievable থাকা বিপজ্জনক)।
- 🐳 **Week 18-ready** — UI শুধু HTTP-তে API ডাকে (`HC_API_URL`); Chroma local folder বা server (`HC_CHROMA_HOST`); startup-এ খালি collection নিজে থেকে index (`HC_AUTO_INGEST`); single-service `Dockerfile`।
- 📏 **Eval set** — ৪২টা labelled প্রশ্ন; route accuracy, `unsafe_answers`, `missed_escalations`, retrieval hit-rate, PII recall; safety metric fail হলে `cli.py eval` exit code 2।
- 🧪 **67 offline tests** — real FastAPI + real Chroma (temp dir); শুধু embedder আর LLM নকল।

## 🚀 Quick Start

```bash
cd "Job Ready AI Bootcamp/week 17/Class 1-2 Project"
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama serve

uvicorn src.api.main:app --port 8000                   # http://localhost:8000/docs  (startup-এ KB index হয়)
streamlit run ui.py                                    # http://localhost:8501  (HC_API_URL=http://localhost:8000)
python cli.py ask "How do I prepare an ORS packet?"
python cli.py eval                                     # route accuracy + safety metrics
pytest -q                                              # 67 tests, Ollama/model download লাগে না
```

Ollama বা model ছাড়া demo: `HC_BACKEND=fake uvicorn src.api.main:app --port 8000` (Windows PowerShell: `$env:HC_BACKEND="fake"`)।

```bash
curl localhost:8000/ready
curl -X POST localhost:8000/v1/ask -H "content-type: application/json" \
     -d '{"question": "My phone is 01712-345678. What are the clinic opening hours?"}'
curl -X POST localhost:8000/v1/ask -H "content-type: application/json" -d '{"question": "I have chest pain"}'
python samples/make_handbook.py                        # data/kb -> samples/clinic_handbook.pdf
```

Docker (শুধু API; Ollama host-এ চলে। Docker daemon ছাড়া লেখা, build করে verify করা হয়নি — ADR 0008):

```bash
docker build -t healthcare-api .
docker run -p 8000:8000 -e HC_OLLAMA_HOST=http://host.docker.internal:11434 healthcare-api
```

## 📁 Project Structure

```
Class 1-2 Project/
├── src/
│   ├── assistant.py          # HealthcareAssistant.ask(): the guardrail ORDER lives here  ← start reading
│   ├── guardrails/
│   │   ├── pii.py            # 1. regex redaction (+ Bangla digits), counts
│   │   ├── red_flags.py      # 2. emergency rules + fixed 999 messages
│   │   ├── scope.py          # 3. personal diagnosis / dosing refusal
│   │   └── grounding.py      # 4. abstention message + [n] citation verification
│   ├── rag/
│   │   ├── kb.py             # data/kb/*.md -> sections, ids kb:<page>:<n>
│   │   └── components.py     # Embedder (lazy), ChromaStore (Persistent | Http), OllamaLLM
│   ├── api/
│   │   ├── main.py           # FastAPI: /health /ready /v1/ask /v1/documents /v1/ingest, auto-ingest
│   │   └── schemas.py        # Pydantic contract (route, citations, redactions)
│   ├── evaluation/runner.py  # route accuracy, safety misses, hit-rate, PII recall
│   ├── backend.py            # real (MiniLM + Chroma + Ollama) | fake (keyword embedder + extractive LLM)
│   └── settings.py           # pydantic-settings, env prefix HC_, telemetry off
├── data/
│   ├── kb/*.md               # 30 handbook pages = 128 sections (source of truth)
│   └── eval/questions.jsonl  # 42 labelled questions (route + expected section)
├── samples/                  # make_handbook.py -> clinic_handbook.pdf (printable, same ids)
├── notebooks/01_exploration.ipynb   # KB -> distances -> threshold -> each guardrail -> eval
├── tests/                    # 67 offline tests
├── ui.py                     # Streamlit, talks to the API over HTTP only
├── cli.py                    # ingest | ask | eval
├── .streamlit/config.toml    # gatherUsageStats = false
├── Dockerfile                # API image: CPU torch, embedding model baked in, KB inside, non-root
└── requirements.txt          # pinned, tested versions
```

## 🏗️ Architecture

```
 Streamlit ui.py ──HTTP (HC_API_URL)──▶ FastAPI  /v1/ask   (middleware: request id, timing; logs path, never body)
                                           │
                                           ▼  run_in_threadpool
 ┌──────────────────────── HealthcareAssistant.ask(question) ───────────────────────────┐
 │ 1. PII redaction ── raw text stops here; everything below sees "[PHONE]", "[NAME]"   │
 │        │                                                                             │
 │ 2. Red-Flag rules ─ match ─▶ route=escalated  (fixed 999 message; no retrieval/LLM)  │
 │        │                                                                             │
 │ 3. Diagnosis/dosing rules ─ match ─▶ retrieve ─▶ route=refused                       │
 │        │                             (fixed refusal + ≤2 related sections; no LLM)   │
 │        ▼                                                                             │
 │    embed (MiniLM) ─▶ Chroma top-k (cosine)                                           │
 │        │                                                                             │
 │ 4a. best distance > HC_MAX_DISTANCE ─▶ route=abstained  (no LLM)                     │
 │        │                                                                             │
 │    llama3.1:8b (Ollama) on numbered passages, "answer only from passages, cite [n]"  │
 │        │                                                                             │
 │ 4b. code checks [n]: none valid / NOT_IN_KB ─▶ route=abstained                       │
 │        │            invented [n] ─▶ removed + flags                                  │
 │        ▼                                                                             │
 │    route=answered + citations (kb:<page>:<n>, distance, snippet)                     │
 └──────────────────────────────────────────────────────────────────────────────────────┘
      log: route, rule, redaction counts, REDACTED question, total_ms
```

| Route | কখন | LLM ডাকা হয়? | Response-এ |
|---|---|---|---|
| `escalated` | emergency rule মিলেছে | না (retrieval-ও না) | স্থির 999 message, `rule` (যেমন `chest_pain`) |
| `refused` | ব্যক্তিগত diagnosis বা dose | না | স্থির refusal + ≤২টা সাধারণ section |
| `abstained` | KB-তে নেই, বা model valid citation দেয়নি | প্রথম ক্ষেত্রে না | "handbook-এ পাইনি", `rule` = `below_threshold` / `no_valid_citation` / `model_abstained` |
| `answered` | grounded, যাচাই-করা citation সহ | হ্যাঁ | উত্তর + citation |

## ⚙️ Configuration (`HC_` env vars)

| Variable | Default | মানে |
|---|---|---|
| `HC_BACKEND` | `real` | `fake` = keyword embedder + extractive LLM; Ollama/model download লাগে না |
| `HC_OLLAMA_HOST` | `http://localhost:11434` | Ollama server (Week 18: `http://ollama:11434`) |
| `HC_MODEL` | `llama3.1:8b` | course-এর একমাত্র LLM |
| `HC_TEMPERATURE` / `HC_NUM_CTX` / `HC_NUM_PREDICT` | `0.1` / `4096` / `400` | generation options |
| `HC_LLM_TIMEOUT_S` | `120` | এর বেশি হলে 503 `llm_unavailable` |
| `HC_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | বদলালে `cli.py ingest --reset` + threshold আবার মাপুন |
| `HC_CHROMA_HOST` | *(empty)* | খালি = local `PersistentClient(HC_CHROMA_DIR)`; set = `HttpClient(host, port)` |
| `HC_CHROMA_PORT` | `8000` | Chroma server port |
| `HC_CHROMA_DIR` | `data/chroma_db` | local index (git-ignored) |
| `HC_COLLECTION` | `clinic_kb` | Chroma collection (cosine space) |
| `HC_KB_DIR` | `data/kb` | handbook Markdown |
| `HC_AUTO_INGEST` | `true` | startup-এ collection খালি হলে KB index |
| `HC_TOP_K` | `4` | প্রতি প্রশ্নে কয়টা section |
| `HC_MAX_DISTANCE` | `0.56` | abstention threshold (cosine) — notebook §3-এ calibrate করা |
| `HC_LOG_LEVEL` | `INFO` | log level |
| `HC_API_URL` *(UI only)* | `http://localhost:8000` | Streamlit কোন API ডাকবে |

## 🔌 API

| Method | Path | Success | Errors |
|---|---|---|---|
| GET | `/health` | `200 {"status":"ok"}` (liveness, কিছু ছোঁয় না) | — |
| GET | `/ready` | `200 {status, backend, llm, indexed_sections}` | `503` Ollama/model নেই, Chroma নেই, বা KB index হয়নি |
| GET | `/v1/documents` | `200 {"pages": [{file, page, sections}]}` — যা আসলে index-এ আছে | — |
| POST | `/v1/ingest` | `200 {pages, sections, removed}` — `HC_KB_DIR` আবার index (upload নেই) | — |
| POST | `/v1/ask` `{question}` (১-১০০০ char) | `200 {route, answer, rule, citations[], redaction_count, redactions, redacted_question, flags, disclaimer, timings_ms, request_id}` | `422` (input echo হয় না), `503 llm_unavailable` (শুধু answered পথে) |

সব error-এর একই আকার: `{error, detail, request_id}`। প্রতিটা response-এ `X-Request-ID` আর `X-Process-Time-Ms`।

## 🧠 Model Choices

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **llama3.1:8b** via Ollama (chosen) | passage থেকে grounded উত্তর | Offline, ভালো instruction following, course standard (ADR 0006) | Medical training নেই — তাই শুধু passage থেকে, citation code-এ যাচাই |
| BioMistral-7B | Medical QA research | PubMed-এ further pre-trained | Safety tuning দুর্বল, Ollama-য় official না |
| Meditron-7B | Clinical guideline QA | Clinical guideline + PubMed | Llama 2 base, chat tuning কম |
| llama3.2:3b | 8 GB RAM মেশিন | দ্রুত, হালকা | Citation format বেশি ভাঙে |

| Model (embedding) | Use Case | Advantage | Limitation |
|---|---|---|---|
| **paraphrase-MiniLM-L3-v2** (chosen) | ছোট English handbook | ~61 MB, CPU-তে দ্রুত (ADR 0004) | Acronym ("ORS") আর medical synonym দুর্বল |
| all-MiniLM-L6-v2 | General English retrieval | ভালো quality | একটু ধীর |
| pritamdeka/S-PubMedBert-MS-MARCO | Biomedical retrieval | PubMed vocabulary, retrieval-tuned | ~440 MB; সাধারণ মানুষের প্রশ্নে লাভ অনিশ্চিত |
| paraphrase-multilingual-MiniLM-L12-v2 | বাংলা প্রশ্ন | Cross-lingual | বড়; threshold আবার মাপতে হয় |

## 🔒 Privacy Guarantees (আর তাদের সীমা)

**যা নিশ্চিত:**
- কোনো cloud API call নেই। LLM (Ollama), embedding (sentence-transformers), vector DB (Chroma) — সব local। Model weight একবার download, তারপর offline (Dockerfile-এ model image-এর ভেতরে, `HF_HUB_OFFLINE=1`)।
- Telemetry বন্ধ: Chroma `anonymized_telemetry=False` (দুই client-এই), `HF_HUB_DISABLE_TELEMETRY=1`, Streamlit `gatherUsageStats=false`।
- Redaction সবার আগে। Log, embedding, Chroma query, LLM prompt আর response — সবাই শুধু redacted প্রশ্ন দেখে। Raw প্রশ্ন কোথাও লেখা হয় না; HTTP log-এ শুধু method, path, status, সময়।
- Validation error input echo করে না; unhandled error-এ stack trace বা message যায় না।
- প্রশ্ন সংরক্ষণ হয় না: কোনো cache, database বা history নেই।

**যা নিশ্চিত না:**
- **Regex PII redaction অসম্পূর্ণ।** ঠিকানা, ছোট হাতের নাম ("i am rahim"), বানান ভুল, অচেনা ID format বাদ পড়ে। Production-এ NER model (যেমন Presidio) + regex + মানুষের review লাগে।
- **এটা HIPAA বা বাংলাদেশের কোনো health-data আইনের compliance না।** Access control, audit log, encryption at rest, data retention policy — কিছুই নেই।
- **এটা medical device না** আর clinician-এর বিকল্প না। Handbook কাল্পনিক, clinician-reviewed না।
- Uvicorn/OS-level log, browser history, বা Streamlit server-এর memory-তে কী থাকে তা এই code নিয়ন্ত্রণ করে না।
- `/v1/ingest`-এ auth নেই; network-এ খোলা রাখবেন না।

## 📏 What to Measure on Your Machine

এই repo যেখানে লেখা, সেখানে Ollama ছিল না (ADR 0008), তাই LLM-এর সংখ্যা মাপা হয়নি। আসল embedder দিয়ে threshold calibrate করা হয়েছে (notebook §3): answerable প্রশ্নের সবচেয়ে কাছের section ≤০.৫৪ (একটা acronym ব্যতিক্রম, ০.৬২), not-in-KB ≥০.৫৯ → `0.56`।

```bash
python cli.py eval --out output/eval_report.json
```

| Metric | কেন | লক্ষ্য |
|---|---|---|
| `unsafe_answers` | escalate/refuse হওয়ার কথা, কিন্তু model-লেখা উত্তর গেছে | **০** (না হলে exit 2) |
| `missed_escalations` | emergency ধরা পড়েনি | **০** |
| `per_route_recall.answered` | false abstention কত — বেশি হলে threshold বা embedding দেখুন | যত বেশি তত ভালো |
| `retrieval_hit_rate` | প্রত্যাশিত section top-k-এ | ≥০.৯ |
| `citation_hit_rate` | llama3.1:8b সঠিক section cite করছে কিনা | মাপুন |
| rows-এ `rule=no_valid_citation` | model কত ঘন ঘন `[n]` ভুলে যায় | কম; বেশি হলে prompt/temperature |
| `pii_recall` | PII-সহ প্রশ্নে redaction হয়েছে | ১.০ |
| `timings_ms.llm_ms` বনাম বাকি | সময় কোথায় যায় (embed+search কয়েক ms) | — |

নিজের প্রশ্ন যোগ করুন (`data/eval/questions.jsonl`), বিশেষ করে যেগুলোতে ভুল হয়েছে।

## 🧭 Design Decisions

- **Markdown section = chunk।** Handbook আমরাই লিখি, তাই section ছোট রাখা (≤৪৫০ char, test যাচাই করে) — chunker লাগে না, citation হুবহু। PDF শুধু মানুষের জন্য, একই source থেকে বানানো।
- **Cosine distance।** ০-১ scale-এ threshold বোঝা সহজ, vector-এর দৈর্ঘ্যে নির্ভর করে না।
- **Threshold নিরাপদ দিকে।** ০.৬৫ দিলে acronym প্রশ্নটা বাঁচত, কিন্তু malaria-র প্রশ্ন dengue passage নিয়ে LLM-এ যেত। Abstain করা নিরাপদ ভুল।
- **Refusal-এর সাথে related section।** খালি "না" বলার চেয়ে সাধারণ তথ্য (যেমন OTC label পড়ার নিয়ম) দেখানো বেশি সাহায্য করে; লেখা code-এর, তাই dose ঢোকার পথ নেই।
- **Ollama down = 503, degraded উত্তর না।** Week 15 passage ফেরত দিত; medical প্রসঙ্গে কাঁচা passage-কে উত্তর হিসেবে দেখানো বিভ্রান্তিকর। Safety route-গুলো LLM ছাড়াই চলে।
- **Cache নেই।** PII-সহ প্রশ্ন cache-এ রাখা privacy ঝুঁকি, আর গতি এই project-এর লক্ষ্য না।
- **Fake backend-এ আসল guardrail।** শুধু model দুটো নকল, তাই test আর demo আসল route আচরণ দেখায়।

## 🐳 Notes for Week 18 (containers)

- API: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`, health `GET /health`, readiness `GET /ready`।
- UI: `streamlit run ui.py --server.port 8501 --server.address 0.0.0.0`, শুধু `streamlit` + `httpx` লাগে, env `HC_API_URL=http://api:8000`।
- Chroma server: `HC_CHROMA_HOST=chroma`, `HC_CHROMA_PORT=8000`; server image-এর version client-এর (`chromadb==1.5.9`) সাথে মেলান, আর সেখানেও `ANONYMIZED_TELEMETRY=FALSE`।
- Ollama: `HC_OLLAMA_HOST=http://ollama:11434`; model একবার `ollama pull llama3.1:8b` (volume-এ রাখুন)।
- নতুন stack-এ `HC_AUTO_INGEST=true` খালি collection নিজে index করে; আলাদা ingest ধাপ লাগে না।

## ⚠️ Known Limits

- **Keyword rule।** Red flag আর refusal negation বোঝে না ("no chest pain" → escalate) আর তথ্যের প্রশ্নেও escalate করে ("heart attack-এর ঝুঁকি?")। এটা ইচ্ছাকৃত: emergency মিসের চেয়ে অকারণ 999 সস্তা। পরোক্ষ dosing প্রশ্ন ("is 11 high for me?") বাদ পড়তে পারে।
- **Citation check শুধু নম্বর দেখে**, বাক্যটা সত্যিই passage-এ আছে কিনা না। Ceiling: sentence-level support check বা NLI।
- **English-only retrieval।** বাংলা প্রশ্নে emergency rule কাজ করে, কিন্তু retrieval দুর্বল — বাংলা-মেশানো প্রশ্নের distance বেশি (multilingual embedding দরকার)।
- **Section id অবস্থান-ভিত্তিক।** মাঝখানে section ঢোকালে পরের id আর eval label সরে যায়; নতুন section page-এর শেষে যোগ করুন।
- Placeholder (`[NAME]`, `[PHONE]`) embedding-কে off-topic দিকে টানে, তাই embedding-এর আগে সেগুলো বাদ দেওয়া হয় (`strip_placeholders`); তবুও PII-ভারী প্রশ্নের distance একটু বেশি থাকে (p05 ≈০.৫৪, threshold-এর কাছে)।
- Prompt injection: passage-কে "data" বলা হয়েছে; handbook আমাদের লেখা, তাই ঝুঁকি কম, কিন্তু অন্য source যোগ করলে আবার ভাবতে হবে।
- এক process, কোনো concurrency gate নেই (Week 15-এর semaphore ইচ্ছে করে বাদ); বেশি user হলে সেটা ফিরিয়ে আনুন।
