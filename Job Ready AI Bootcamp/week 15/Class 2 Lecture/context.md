# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: LLM Observability & Monitoring — Metrics, Traces, Quality আর SLO**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

Class 29-এ RAG bot একটা service হয়েছে। Deploy-এর পরের প্রশ্নগুলো: কত দ্রুত উত্তর দিচ্ছে, কোন ধাপে সময় যাচ্ছে, কত token যাচ্ছে, উত্তরগুলো কি সত্যিই document থেকে আসছে, গত সপ্তাহের চেয়ে আজ খারাপ কিনা, আর কখন কাউকে রাত তিনটায় ডাকতে হবে। সাধারণ web monitoring (status code, latency) এর অর্ধেক দেখে — কারণ LLM-এর সবচেয়ে বিপজ্জনক ব্যর্থতা **HTTP 200 দেয়**: ভুল সংখ্যা, বানানো দাবি, ভুল citation, ভুল করে "not found"। এই class-এ শিখব কীভাবে এগুলো মাপতে হয় — আর মাপার যন্ত্রকেও (LLM judge) কীভাবে যাচাই করতে হয়।

## শেখার সহজ Workflow

Monitoring বনাম observability → metrics/traces/logs → golden signal stage ধরে → Ollama-র free telemetry → Prometheus (bucket, cardinality) → OpenTelemetry GenAI trace → tool বাছাই → quality-র তিন স্তর → LLM judge আর তার bias → drift আর feedback → SLO আর burn-rate alert।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** প্রতিটা metric-এর আগে জিজ্ঞেস করুন — "এটা লাল হলে আমি কী করব?" উত্তর না থাকলে metric-টা dashboard-এর শোভা, monitoring না।

---

## 1. Topic: LLM Observability & Monitoring

এই ক্লাসে আমরা শিখব:

* **Metrics, traces, logs** — কোনটা কোন প্রশ্নের উত্তর দেয়।
* **Golden signals, stage ধরে** — latency, traffic, errors (implicit সহ), saturation।
* **Prometheus** — naming, histogram, LLM-আকারের bucket, cardinality, PromQL।
* **OpenTelemetry GenAI semantic conventions** — `chat llama3.1:8b` span, `gen_ai.*` attribute, maturity।
* **Tool landscape ২০২৬** — Phoenix, Langfuse v5, OpenLIT, Prometheus + Grafana।
* **Quality monitoring** — reference-free proxy, citation check, RAGAS faithfulness, LLM-as-judge ও তার bias।
* **Drift, feedback, PII**।
* **SLO + multiwindow burn-rate alert** — কখন page, কখন ticket।

**Project Goal (Class 2 Project — non-portfolio, full project per ADR 0007):** Class 1-এর RAG API-র উপর পুরো monitoring stack, কোনো RAG code না বদলে: Prometheus `/metrics`, SQLite per-request log, প্রতি request-এ সস্তা quality check, OpenTelemetry trace (Phoenix/Langfuse), offline LLM judge, `/v1/stats` + SLO burn rate, HTML dashboard, feedback endpoint, আর promtool-tested alert rules।

---

## 2. Why It Is Related

Class 29-এর service প্রতিটা call শেষে একটা `CallRecord` বানিয়ে observer-দের দেয় — সেটাই এই class-এর প্রবেশপথ (seam)। Week 12-এর guardrail (`max_distance`), Week 13-এর tool-call log, Week 14-এর citation verification — সবগুলোই আসলে monitoring signal ছিল, কিন্তু এক এক জায়গায় ছড়ানো। এখানে সেগুলো এক জায়গায়: একটা observer, তিনটা output (SQLite, Prometheus, trace)।

---

## 3. How It Works

### 3.1 Monitoring বনাম Observability; Metrics, Traces, Logs

**Monitoring** = আগে থেকে জানা প্রশ্নের উত্তর ("error rate কত?")। **Observability** = অজানা প্রশ্নের উত্তর খুঁজে বের করার ক্ষমতা ("মঙ্গলবার দুপুরে শুধু HR প্রশ্নগুলো কেন ধীর?")। তিন ধরনের data:

| Data | কোন প্রশ্নের উত্তর | বৈশিষ্ট্য | Project-এ |
|---|---|---|---|
| **Metrics** | "কিছু ভুল? কতটা?" | aggregate, সস্তা, কখনো sample হয় না, alert-এর জন্য | Prometheus `/metrics` |
| **Traces** | "*এই* request-এর সময় কোথায় গেল?" | per-request causal tree, sample করা যায় | OTel → Phoenix |
| **Logs / per-request rows** | "ঠিক কী হয়েছিল?" | বিস্তারিত, context-সহ | SQLite `requests` table |

Request ID / trace ID তিনটাকে বাঁধে: metric-এ spike দেখে → log-এ সেই সময়ের row → trace-এ সেই request-এর stage।

### 3.2 Golden Signals, Stage ধরে

Google SRE book-এর চার golden signal: **latency, traffic, errors, saturation**। LLM RAG-এ প্রতিটার নতুন রূপ:

| Signal | LLM RAG-এ | Project metric |
|---|---|---|
| Latency | end-to-end + TTFT, **stage ধরে** (embed, search, prefill, decode); সফল আর ব্যর্থ আলাদা | `rag_stage_duration_seconds{stage}`, `rag_ttft_seconds` |
| Traffic | req/s, **token/s** | `rag_http_requests_total`, `rag_tokens_total{kind}` |
| Errors | 5xx/timeout **+ implicit**: not found, truncation, ভুল citation, অসমর্থিত সংখ্যা | `rag_answers_total{outcome}`, `rag_quality_flags_total{flag}` |
| Saturation | in-flight, queue wait, context-window fill, GPU | `rag_llm_in_flight`, `rag_queue_wait_seconds` |

SRE book নিজেই "implicit failure" (সফল response, ভুল content)-কে error গোনে — LLM monitoring-এর কেন্দ্র এটাই। আর বলে average না, latency-কে bucket-এ গুনতে, কারণ average tail লুকায়।

### 3.3 Ollama-র Free Telemetry

Ollama প্রতিটা `/api/chat` response-এ দেয়: `prompt_eval_count`, `eval_count`, `done_reason`, আর nanosecond-এ `total_duration`, `load_duration`, `prompt_eval_duration`, `eval_duration`। কোনো library ছাড়াই:

| Metric | সূত্র | কেন |
|---|---|---|
| Decode speed | `eval_count / (eval_duration/1e9)` | GPU contention-এ কমে |
| Prefill speed | `prompt_eval_count / (prompt_eval_duration/1e9)` | লম্বা prompt কীভাবে TTFT বাড়ায় |
| Server-side TTFT | ≈ `load_duration + prompt_eval_duration` | client-side TTFT-এ HTTP + queue যোগ হয় |
| Cost proxy | `(prompt_eval_duration + eval_duration)/1e9` = GPU-seconds | local model-এর "bill" |
| Cold start | `load_duration` বড় | ৫ মিনিট idle-এর পর model unload |

**দুটো ফাঁদ (project-এ মাপা):** (১) `prompt_eval_duration` শুধু *uncached* token গোনে — আমাদের data-য় নতুন prompt-এ prefill ≈১.৭-২.৪ s, prefix আগে দেখা থাকলে ≈৫০ ms; TTFT-এর বণ্টন তাই **bimodal**, গড় অর্থহীন। (২) warm call-এও `load_duration` কয়েক ms থাকে; project ৫০০ ms-এর বেশি হলে `rag_llm_cold_starts_total` গোনে। Option (যেমন `num_ctx`) বদলালেও Ollama model আবার load করে — lecture notebook-এ ৩.৬ s দেখা গেছে।

### 3.4 Prometheus: Naming, Histogram, Bucket, Cardinality

* **Naming:** base unit (`_seconds`, ms না), counter-এর শেষে `_total`, এক শব্দের prefix (`rag_`)।
* **Histogram, summary না:** summary instance-জুড়ে যোগ করা যায় না। p95 = `histogram_quantile(0.95, sum by (le) (rate(x_bucket[5m])))`।
* **Bucket — সবচেয়ে সাধারণ LLM ভুল:** `prometheus_client`-এর default bucket **10 s-এ শেষ**। ৩০-৬০ s-এর local generation `+Inf`-এ পড়ে, p95 অর্থহীন। Project: `(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 120)`। OTel GenAI spec-এর নিজের inference bucket 81.92 s পর্যন্ত — spec-লেখকরাও দশ সেকেন্ডের call ধরে নেন।
* **Cardinality:** প্রতিটা আলাদা label মান = নতুন time series। User id, প্রশ্ন, document id, trace id label-এ **কখনো না** — সেগুলো trace/log-এ। Project route-এর **template** (`/v1/ask`) label করে; অজানা path হয় `unmatched` (notebook-এ ৩টা random path → একটাই নতুন label)।
* **`/metrics`-এর 307 ফাঁদ:** `app.mount("/metrics", make_asgi_app())` → `GET /metrics` 307 → `/metrics/`। Project plain route দেয়, redirect নেই।
* **Absent series ফাঁদ (project-এ live ধরা পড়েছে):** কখনো 5xx না হলে 5xx series-ই নেই → error ratio *খালি*, 0 না। Recording rule-এ `(sum(rate(...5xx...)) or vector(0)) / sum(rate(...))`।

### 3.5 OpenTelemetry Traces আর GenAI Semantic Conventions

একটা trace = একটা request; span = প্রতিটা ধাপ:

```
rag.ask                  2463 ms   CHAIN
  embed                    12 ms   EMBEDDING   gen_ai.operation.name=embeddings
  retrieve                  1 ms   RETRIEVER   gen_ai.operation.name=retrieval
  chat llama3.1:8b       2450 ms   LLM         (SpanKind.CLIENT)
```

**GenAI semantic conventions (October 2026):**

* Status এখনো **Development**; মে ২০২৬-এ আলাদা repo-তে (`open-telemetry/semantic-conventions-genai`), কোনো tagged release নেই।
* Inference span-এর নাম `{gen_ai.operation.name} {gen_ai.request.model}` → `chat llama3.1:8b`, kind `CLIENT`।
* Required: `gen_ai.operation.name`, `gen_ai.provider.name` (পুরোনো `gen_ai.system`-এর বদলি)। Recommended: `gen_ai.request.model`, `gen_ai.usage.input_tokens/output_tokens`, `gen_ai.response.finish_reasons`।
* `ollama` well-known provider না, কিন্তু spec বলে custom মান চলে।
* **Prompt/answer content opt-in** — default-এ capture না (PII)। Project: `RAG_TRACE_CONTENT=true` দিলে তবেই।
* **Metric নাম এখনো বদলাচ্ছে:** repo-র `main`-এ `gen_ai.client.operation.*` → `gen_ai.client.inference.*`, কিন্তু released Python package (`opentelemetry-semantic-conventions` 0.66b1) এখনো পুরোনো নাম। শিক্ষা: **attribute নাম শেখান (স্থিতিশীল), metric নাম এক module-এ রাখুন** (বদলালে এক লাইন)।

**Project-এর design — post-hoc spans:** service layer আগেই প্রতিটা stage-এর আসল সময় মাপে (`timings_ms`, `queue_wait_s`)। Request শেষে observer সেই মাপ থেকে explicit start/end time দিয়ে span বানায় — RAG core-এ একটাও tracing লাইন নেই। Ceiling: stage-গুলো পরপর ধরা হয় (এখানে সত্যিই পরপর)। Phoenix 20.26.0-এ live: ৪৩ span, ১৩ trace, LLM/RETRIEVER/EMBEDDING/CHAIN kind চেনে। Phoenix এই kind পড়ে OpenInference attribute (`openinference.span.kind`) থেকে — তাই project দুটো vocabulary-ই লেখে।

### 3.6 Observer Seam

```python
def make_service():
    svc = RagService(make_backend(s), ..., observers=[app.state.monitor.observe])
    tm.IN_FLIGHT.set_function(lambda: svc.in_flight)        # saturation, scrape-এর সময় পড়া হয়
    return svc

class Monitor:
    def observe(self, rec: CallRecord):
        row = self.row(rec)          # quality.assess() একবার
        self.log.add(row)            # SQLite
        self._metrics(row)           # Prometheus
        if self.tracer: emit_trace(self.tracer, row, ...)   # OTel
```

Service `_emit` প্রতিটা observer-এর exception গিলে log করে — monitoring ভাঙলে request ভাঙে না (test আছে)। Stream-এর 503 (slot না পাওয়া) আর client disconnect-ও record হয়; disconnect `outcome=disconnected`, availability SLO-তে error না।

---

## 4. Details & Valid Points

### 4.1 Tool Landscape (October 2026)

| Tool | পুরো local? | Footprint | সবচেয়ে ভালো | সতর্কতা |
|---|---|---|---|---|
| **Prometheus + Grafana** | হ্যাঁ | ২টা container / binary | golden signal, PromQL, alerting | per-request prompt debug নেই |
| **Arize Phoenix 20.26.0** (course default) | হ্যাঁ, air-gap সম্ভব | ১ process, SQLite, OTLP 6006/4317 | সবচেয়ে হালকা LLM trace UI + eval | license এই research-এ নিশ্চিত না |
| **Langfuse** (SDK 5.0.4, server ≥ 4.59) | হ্যাঁ, কিন্তু ভারী | ৬ service, সুপারিশ ৪ core / 16 GiB | prompt management, score, session, dataset, online eval | Compose production-grade না; কিছু EE feature |
| OpenLIT 1.45 | হ্যাঁ | platform + collector | এক লাইনে instrumentation, Ollama + GPU | port/license verify হয়নি |
| LangSmith | মূলত SaaS | — | LangChain/LangGraph | self-host শর্ত verify হয়নি |
| Helicone | self-host proxy | gateway | base URL বদলে cost/latency log | বর্তমান architecture verify হয়নি |

**সিদ্ধান্ত:** 8B model + 16 GiB observability stack বেশিরভাগ student laptop-এ আঁটে না → Phoenix default, Langfuse optional। সবাই OTLP নেয়, তাই `gen_ai.*` দিয়ে একবার instrument করলে পছন্দ বদলানো config-এর ব্যাপার। Project-এর `docker-compose.yml` (Langfuse) এই repo-তে live চালানো হয়নি (Docker daemon ছিল না)।

**Langfuse SDK-র ২০২৪→২০২৬ পরিবর্তন** (পুরোনো course material এখন ভুল): v4 "observations-first" — `start_as_current_span/generation` → একটাই `start_as_current_observation(as_type=...)`; user/session id → `propagate_attributes()`; আর default-এ শুধু LLM-জাতীয় span export (FastAPI/httpx span হারিয়ে যায়, `should_export_span` দিয়ে ফেরান)। v5: `set_current_trace_io()` নেই (trace I/O root observation থেকে), batch gzip।

### 4.2 Quality Monitoring: তিন স্তর

Live traffic-এ gold answer নেই → **reference-free** metric। RAGAS **faithfulness** = উত্তরের claim-গুলোর কত ভাগ context সমর্থন করে (LLM claim-এ ভাঙে, প্রতিটা যাচাই করে, অনুপাত)। TruLens-এর **RAG triad**: context relevance (retrieval ভুল), groundedness (অসমর্থিত দাবি), answer relevance (প্রশ্নের পাশ কাটানো)। Reference-based metric (context recall, correctness) nightly golden-set-এ, live traffic-এ না।

| Tier | Signal | কখন | খরচ |
|---|---|---|---|
| 1 | outcome, citation presence/validity, numbers supported, lexical support, truncation, top-1 distance | প্রতি request | ~০ |
| 2 | NLI/HHEM groundedness (CPU classifier), embedding relevance | প্রতি request বা sample | CPU model |
| 3 | LLM judge | offline, sample | ১টা LLM call |
| — | user 👍/👎 | ঐচ্ছিক | ০ |

**Project-এ মাপা, সস্তা check-এর সীমা** (আসল hits, ইচ্ছাকৃত ভুল উত্তর):

| Planted answer | citation_valid | numbers_supported | ধরা পড়ল? |
|---|---|---|---|
| "18 sick days [1]" | ✅ | ❌ (18) | হ্যাঁ |
| "12 days [9]" | ❌ | ✅ | হ্যাঁ |
| "12 days" (citation নেই) | ❌ | ✅ | হ্যাঁ |
| **"15 sick days [1]"** | ✅ | ✅ | **না** — অন্য passage-এ "15:00" (core hours) |
| **"…unused sick days can be cashed out [1]"** | ✅ | ✅ | **না** — সংখ্যা ঠিক, শব্দ মেলে |

সস্তা check দরকারি কিন্তু যথেষ্ট না — শুধু প্রমাণ করে সংখ্যাটা context-এ *কোথাও* আছে।

ALCE benchmark citation-কে NLI দিয়ে মাপে (recall: cited passage-গুলো মিলে দাবি entail করে কিনা; precision: প্রতিটা citation দরকারি কিনা); সেরা model-ও ELI5-এ ৫০% সময় পূর্ণ citation support দেয় না। NLI judge-এর মানুষের সাথে মিল κ ≈ 0.70 (recall), 0.53 (precision)।

### 4.3 LLM-as-Judge: কাজের, কিন্তু biased — আর prompt-এর গঠনে সংবেদনশীল

Zheng et al.: GPT-4 judge মানুষের সাথে "৮০%-এর বেশি" মেলে (মানুষ-মানুষের সমান), কিন্তু **position, verbosity, self-enhancement bias**। Panickssery et al.: LLM নিজের লেখা উত্তর পছন্দ করে, আর নিজেকে চেনার ক্ষমতার সাথে এই পক্ষপাত রৈখিকভাবে বাড়ে। এই course-এ judge আর উত্তরদাতা **একই model** (llama3.1:8b) → self-preference এড়ানো অসম্ভব; সীমাবদ্ধতা হিসেবে বলতে হবে, score পড়তে হবে trend হিসেবে।

**Project-এ তিনটা judge design মাপা হয়েছে** (৮টা আসল উত্তর + ২টা ভুল + ১টা সঠিক paraphrase):

| Design | ফল |
|---|---|
| `{"supported": bool, "unsupported_claim": str}` | "cash out"-এ `supported: true` অথচ claim-এ সেটাই লেখা — নিজের সাথে অসঙ্গত |
| আগে "unsupported claim-এর তালিকা", পরে verdict | সব claim-ই তালিকায় → **০/১১** |
| **claim → passage থেকে হুবহু evidence → verdict** + code-এ quote check | **১১/১১** (notebook-এ ১২/১২) |

শিক্ষা: (১) একটা criterion, কারণ আগে রায় পরে (JSON field-এর ক্রম = generation-এর ক্রম), temperature 0, structured output (Ollama `format=<JSON schema>`)। (২) Judge-এর "evidence" code-এ যাচাই করা যায় — বানানো quote ধরা পড়ে। (৩) **Judge-কেও eval করতে হয়**: ৫০-২০০টা trace নিজে label করে judge-এর সাথে মেলান, বিশেষ করে failure class-এর recall। (৪) Judge request-এর **বাইরে** — Ollama-য় একটা slot, ভেতরে চললে user latency দ্বিগুণ। (৫) Judge-এর JSON parse failure গুনুন — fail হওয়া verdict চুপচাপ বাদ পড়লে গড় উপরে ওঠে।

### 4.4 Not-Found Rate: দুই দিকেই ব্যর্থ হয়

* **বাড়লে:** retrieval বা index ভেঙেছে, বা threshold বেশি কড়া। Project-এ live: "Is MFA required for the VPN?" → not found (distance 27.06 > 26.0), অথচ handbook-এ আছে; "multi-factor authentication" লিখলে 17.77 → answered। ছোট embedding model acronym মেলায় না। Fix: threshold, ভালো embedding, hybrid search (BM25 acronym-এ exact match), acronym expansion।
* **প্রায় শূন্যে নামলে:** ইচ্ছে করে উত্তরহীন প্রশ্নের একটা canary set-এ model আর "জানি না" বলছে না — hallucinate করছে।

Monitoring-এর কাজ সিদ্ধান্ত নেওয়া না, সিদ্ধান্তের data দেওয়া: not-found row-গুলো distance-সহ চোখে দেখা।

### 4.5 Drift আর Version Stamp

Drift মানে derived feature-এর বণ্টন বদলানো। Evidently-র default: numeric column-এ ১০০০ row পর্যন্ত KS test (p < 0.05), তার বেশি হলে Wasserstein ≥ 0.1; categorical-এ chi-square, তারপর Jensen–Shannon ≥ 0.1; text drift-এ domain classifier ROC AUC > 0.55। RAG-এ তিন দল feature: input (প্রশ্নের দৈর্ঘ্য, ভাষা, topic centroid থেকে দূরত্ব), retrieval (top-1 distance বণ্টন, খালি retrieval rate), output (উত্তরের দৈর্ঘ্য, refusal rate, citation, judge score)। ছোট volume-এ p-value noisy, বড় volume-এ তুচ্ছ পরিবর্তনেও লাল — effect size পছন্দ করুন। **সবচেয়ে নীরব ব্যর্থতা:** query আর index আলাদা embedding model-এ — cosine score আসে, অর্থহীন। Embedding model-এর নাম collection metadata-য় রাখুন, startup-এ assert; প্রতিটা trace-এ model digest, prompt version, index version।

### 4.6 User Feedback

সন্তুষ্টির সবচেয়ে কাছের signal, আবার সবচেয়ে biased: অসন্তুষ্টরা বেশি vote দেয়, সাবলীল ভুল উত্তর 👍 পায়, user source যাচাই করে না। Feedback **faithfulness মাপে না** — এর সেরা ব্যবহার কোন trace মানুষ দেখবে সেটা বাছা আর judge calibrate করার label। Project: `POST /v1/feedback {request_id, vote}` → SQLite row-এ, আর `rag_feedback_total{vote}`।

### 4.7 SLO আর Multiwindow Burn-Rate Alert

**SLO** = লক্ষ্য (যেমন ৯৯% request সফল); বাকি ১% = **error budget**। **Burn rate** = budget কত দ্রুত খরচ হচ্ছে; 1× = ঠিক SLO period-এ শেষ। SRE Workbook (৩০ দিনের period):

| Severity | Long window | Short window | Burn | Budget খরচ |
|---|---|---|---|---|
| Page | 1 h | 5 m | 14.4× | ২% |
| Page | 6 h | 30 m | 6× | ৫% |
| Ticket | 3 d | 6 h | 1× | ১০% |

দুটো window-ই threshold পার হলে তবেই alert: লম্বা window নিশ্চিত করে এটা আসল সমস্যা (precision), ছোট window নিশ্চিত করে এটা *এখনো* চলছে (fix-এর পর দ্রুত reset)।

Project-এর `monitoring/alerts.yml`:

```yaml
- record: rag:request_errors:ratio_rate1h
  expr: (sum(rate(rag_http_requests_total{route=~"/v1/ask.*",status=~"5.."}[1h])) or vector(0))
        / sum(rate(rag_http_requests_total{route=~"/v1/ask.*"}[1h]))
- alert: RagErrorBudgetFastBurn
  expr: rag:request_errors:ratio_rate1h > (14.4 * 0.01) and rag:request_errors:ratio_rate5m > (14.4 * 0.01)
  for: 2m
  labels: {severity: page}
```

`promtool test rules monitoring/alerts_test.yml` — synthetic series দিয়ে unit test: ২০% error → page, ১% → চুপ। **Live (Prometheus 3.15.1):** queue timeout 0.2 s করে ৩ wave × ৬ request → ১৫/১৮টা 503 → Prometheus 5m error ratio 0.50 → `RagErrorBudgetFastBurn` আর `SlowBurn` **pending**; `/v1/stats` নিজে একই হিসাবে burn 60× → `page_fast`, `page_slow`।

**Quality → ticket, কখনো page না।** Classroom API = low traffic: ঘণ্টায় ১০টা request-এ একটা failure = ১০% → বিশাল burn rate। Judge আবার sample করে, denominator আরও ছোট। তাই quality SLI (judged groundedness, not-found rate, unsupported number) ticket-tier, ১-৩ দিনের window; drift dashboard-এ, pager-এ না।

### 4.8 Local Model-এর Cost

Token-প্রতি bill নেই → resource time মাপুন: **GPU-seconds** = prefill + decode duration। × amortized GPU-hour দাম = টাকা; × মাপা watt = energy (macOS-এ `powermetrics`, NVIDIA-তে `nvidia-smi` দিয়ে নিজের সংখ্যা মাপুন — llama3.1:8b-এর প্রকাশিত energy-per-token সংখ্যা research-এ পাওয়া যায়নি)। Project-এর `/v1/stats` `gpu_seconds` দেয় (live ১২ request-এ ১৪.৬ s)।

### 4.9 PII আর Sampling

* Content default-এ capture না (OTel opt-in); metric label-এ কখনো user data না; log-এ চাইলে hash (project: `RAG_LOG_TEXT=false` → SHA-256)।
* Redaction স্তরে: regex (email, phone, card) → PII detector → tool-এর mask।
* Head sampling (`TraceIdRatioBased`) সস্তা কিন্তু বিরল failing trace হারাতে পারে; tail sampling (Collector-এ) সব error আর ধীর trace রাখে। Metric কখনো sample হয় না। Classroom volume-এ সব trace রাখুন।

---

## 5. Related Analogy: হাসপাতালের ICU

* **Monitor-এর vital signs** (heart rate, BP, SpO₂) = **metrics** — সবসময়, সস্তা, সংখ্যায়; সীমা পার হলে alarm।
* **রোগীর chart** (কখন কী ওষুধ, কী ঘটল) = **logs**।
* **রোগীর পুরো যাত্রা** (emergency → X-ray → OT → ward, প্রতিটা ধাপে কত সময়) = **trace**।
* **Lab test** (রক্ত পরীক্ষা, দামি, sample-এ, পরে ফল) = **LLM judge**।
* **রোগীর অভিযোগ** = **user feedback** — গুরুত্বপূর্ণ, কিন্তু সব রোগী বলে না, আর "ভালো লাগছে" মানেই সুস্থ না।
* **Alarm threshold** — প্রতিটা ছোট ওঠানামায় নার্সকে ডাকলে নার্স alarm উপেক্ষা করা শেখে (alert fatigue); তাই **burn rate** — সমস্যা টানা আর বড় হলে তবেই।
* **Thermometer-এর scale ৩৮°-এ শেষ** = **10 s-এ শেষ হওয়া bucket** — ৪০° জ্বর আর ৪২° জ্বর দুটোই "৩৮+"।
* **Lab machine-কেও calibrate** করতে হয় = **judge-কে মানুষের label দিয়ে যাচাই**।

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
|---|---|---|
| Default 10 s bucket | ধীর generation `+Inf`-এ, p95 মিথ্যা | 64-120 s পর্যন্ত bucket |
| User id / প্রশ্ন / trace id label-এ | cardinality বিস্ফোরণ, PII | trace/log-এ রাখুন |
| Mean latency বা summary | tail লুকায়, যোগ করা যায় না | histogram quantile |
| শুধু end-to-end latency | retrieval বনাম prefill বনাম decode আলাদা করা যায় না | stage ধরে মাপুন |
| শুধু 5xx-কে error ধরা | ভুল content (200) অদৃশ্য | implicit error: not found, truncation, citation, number |
| TTFT-এর গড় | prefix cache-এ bimodal | percentile, cache hit আলাদা |
| `app.mount("/metrics")` | 307 redirect | plain route বা `metrics_path: /metrics/` |
| 5xx series নেই ধরে নেওয়া | error ratio খালি, alert কখনো evaluate হয় না | `or vector(0)` |
| Prompt content default-এ trace-এ | PII | opt-in |
| Judge request path-এ | latency দ্বিগুণ, judge-এর outage = API outage | offline, sample |
| Judge-কে বিশ্বাস | self-preference, prompt-গঠনে ০/১১ থেকে ১১/১১ | মানুষের label-এ calibrate, evidence quote check |
| সস্তা number check-কে যথেষ্ট ভাবা | "15" পার পায় কারণ "15:00" আছে | judge-এর সাথে স্তরে স্তরে |
| Quality signal-এ page | low volume-এ noise, alert fatigue | ticket tier |
| Client disconnect-কে error গোনা | availability SLO মিথ্যা খারাপ | আলাদা outcome |
| Langfuse v4+ default | FastAPI span হারায় | `should_export_span` |
| Version stamp নেই | `ollama pull`-এর পর step change ব্যাখ্যাহীন | model digest, prompt/index version trace-এ |

---

## 7. Interview Q&A

1. **Golden signal-গুলো LLM RAG service-এ কীভাবে খাটাবেন?** Latency = end-to-end + TTFT, stage ধরে (embed, retrieve, prefill, decode), সফল আর ব্যর্থ আলাদা। Traffic = req/s আর token/s। Errors = 5xx/timeout **আর** implicit: not found, truncation, ভুল citation, parse failure। Saturation = in-flight, queue wait, context fill, GPU। Symptom-এ (p95, error ratio) page, stage metric দিয়ে কারণ খুঁজুন।
2. **TTFT কেন total latency থেকে আলাদা?** TTFT = queue + model load + prefill → prompt-এর দৈর্ঘ্যে বাড়ে; total-এ decode যোগ → output-এর দৈর্ঘ্যে বাড়ে। Streaming user TTFT অনুভব করে। Prefix cache-এর কারণে TTFT bimodal — আমাদের data-য় ২ s বনাম ০.১ s।
3. **Histogram না summary? p95 কীভাবে?** Histogram — summary instance-জুড়ে যোগ হয় না। `histogram_quantile(0.95, sum by (le) (rate(x_bucket[5m])))`। Accuracy bucket-এর প্রস্থে; SLO-র কাছে ঘন bucket, আর সবচেয়ে ধীর generation-এর পরে পর্যন্ত (local 8B-এ ৬৪ s+)।
4. **User id বা প্রশ্ন label-এ রাখলে সমস্যা কী?** প্রতিটা মান নতুন time series → Prometheus-এর memory/storage বিস্ফোরণ, আর PII। Per-request detail trace/log-এ, metric থেকে exemplar দিয়ে লিঙ্ক।
5. **OTel GenAI semantic conventions কী, কতটা পরিপক্ব?** LLM telemetry-র shared vocabulary: CLIENT span `chat llama3.1:8b`, `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, usage token; content opt-in। এখনো Development, মে ২০২৬ থেকে আলাদা repo, tagged release নেই, metric rename চলছে যা released Python package-এ নেই। লাভ portability — Phoenix, Langfuse, OpenLIT সবাই OTLP নেয়।
6. **Self-hosted model-এর cost কীভাবে?** GPU-seconds = prefill + decode duration (Ollama দেয়); × amortized GPU-hour বা মাপা watt। Token/request-ও track করুন, কারণ সেটাই দুটোকে চালায়।
7. **Metrics, traces, logs — কোনটা কোন প্রশ্ন?** Metrics: "কিছু ভুল? কতটা?" — সস্তা, aggregate, alert। Traces: "এই request-এর সময় কোথায়?" — causal tree, sample করা যায়। Logs: "ঠিক কী ঘটল?"। Trace id log-এ আর exemplar metric-এ তিনটাকে বাঁধে।
8. **Local setup-এ Langfuse না Phoenix?** Phoenix: এক process, SQLite, OTLP 6006/4317, air-gap, laptop-এ আঁটে। Langfuse: prompt management, session, score, dataset, online eval — কিন্তু ৬ service, ৪ core/16 GiB সুপারিশ, Compose production-grade না। Team-এর ঐ feature লাগলে Langfuse, হালকা local tracing-এ Phoenix।
9. **Traffic স্থির, p95 4 s থেকে 20 s — কীভাবে debug করবেন?** Stage histogram: retrieval না LLM? LLM হলে prefill বনাম decode। Prefill বাড়লে prompt বড় হয়েছে (বেশি/বড় chunk) → input token/request দেখুন। Decode বাড়লে লম্বা output বা GPU contention → output token আর saturation। `load_duration` দিয়ে cold start বাদ দিন। তারপর ধীর trace আর তার log।
10. **Ground truth ছাড়া RAG ভালো উত্তর দিচ্ছে কিনা কীভাবে জানবেন?** স্তরে: সব traffic-এ সস্তা heuristic (not-found rate, distance, citation validity, number support); CPU proxy (NLI/HHEM); sampled LLM judge, মানুষের label-এ calibrate; feedback দিয়ে review-এর trace বাছাই; nightly golden set-এ reference-based metric।
11. **Faithfulness, answer relevance, context precision?** Faithfulness = উত্তরের দাবির কত ভাগ context সমর্থন করে (generator মাপে)। Answer relevance = প্রশ্নের উত্তর দিল কিনা। Context precision = প্রাসঙ্গিক chunk উপরে কিনা (retriever মাপে)। Faithful উত্তরও অপ্রাসঙ্গিক হতে পারে, প্রাসঙ্গিক উত্তরও বানানো হতে পারে।
12. **LLM judge-এর bias কী, কীভাবে কমাবেন?** Position (দুই ক্রমে চালিয়ে consistent রায় রাখুন), verbosity (দৈর্ঘ্যকে পুরস্কার না), self-preference (অন্য model family — এক local model-এ অসম্ভব, সীমাবদ্ধতা বলুন), দুর্বল reasoning (এক criterion, কারণ আগে, temperature 0)। সবসময় মানুষের label-এ যাচাই। আমাদের project-এ শুধু schema-র গঠন বদলে ০/১১ থেকে ১১/১১ — তাই judge-কেও eval।
13. **Judge request path-এ কেন না?** Local Ollama-য় একটা slot; judge ভেতরে = user latency দ্বিগুণ, আর judge ভাঙলে API ভাঙে। Path-এ শুধু সস্তা, blocking-যোগ্য check (size/rate limit, schema, citation validity); judge, drift, aggregation async।
14. **Not-found rate কোন দুই দিকে ব্যর্থ?** বাড়লে retrieval/index ভাঙা বা threshold কড়া (আমাদের MFA acronym উদাহরণ)। উত্তরহীন canary প্রশ্নে প্রায় শূন্যে নামলে model abstain করা ছেড়ে hallucinate করছে।
15. **LLM API-র SLO আর alert design করুন।** SLI: availability (non-5xx, timeout = bad), latency/TTFT, sampled groundedness, format validity। Multiwindow burn-rate: 1h+5m-এ 14.4× page, 6h+30m-এ 6× page, 3d+6h-এ 1× ticket। Quality SLI sample-এ ছোট volume → শুধু ticket। Recording rule-এ `or vector(0)` যাতে error না থাকলে ratio 0 হয়, খালি না।
16. **👎 rate 3% — quality ভালো?** অবশ্যই না। Feedback বিরল, self-selected; user groundedness যাচাই করে না; সাবলীল ভুল উত্তর 👍 পায়। Feedback দিয়ে review বাছাই আর judge calibrate; quality মাপুন sampled judged groundedness আর human audit দিয়ে।
17. **কোন embedding-জনিত ব্যর্থতা সবচেয়ে নীরব?** Query আর index আলাদা embedding model-এ — distance আসে, অর্থহীন; error নেই। Collection metadata-য় model নাম, startup-এ assert, trace-এ index version।

---

## Achievement: RAG API Monitoring (Class 30 Project)

**`Class 2 Project/`** — Class 1 service-এর self-contained কপি + `app/telemetry/`: observer (`Monitor`), Prometheus `/metrics` (stage histogram 120 s পর্যন্ত, route-template label, cold start, saturation), SQLite request log (PII mode), প্রতি request-এ সস্তা quality signal, OTel trace (GenAI + OpenInference, CLIENT chat span, content opt-in) Phoenix/Langfuse-এ, offline LLM judge (claim → evidence → verdict + quote check), `/v1/stats` + SLO burn rate, `/v1/dashboard`, `/v1/feedback`, `monitoring/alerts.yml` + promtool unit test। ৪৩টা offline test। Live: Phoenix, Prometheus, 503-burst alert, judge ১১/১১। এই ক্লাসের companion notebook **`monitoring_lecture.ipynb`** সম্পূর্ণ self-contained — project folder ছাড়াই চলে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Bucket Trap:** default bucket-এর একটা histogram-এ ২, ৮, ১২, ৩০, ৪৫ s observe করুন। `histogram_quantile(0.95)` কত বলে? LLM bucket-এ কত?
2. **Cardinality Bill:** `rag_http_requests_total`-এ `question` label যোগ করলে ১০,০০০ জন user, প্রত্যেকে ২০টা আলাদা প্রশ্ন — কয়টা time series? Prometheus-এর memory-তে এর মানে কী?
3. **Judge Field Order:** `Claim` model-এ `supported`-কে `evidence`-এর আগে আনুন। Planted উত্তরগুলোতে ফল বদলায়?
4. **Canary Set:** handbook-এ নেই এমন ১০টা প্রশ্ন লিখুন যেগুলো শুনতে handbook-এর মতো ("How many days of bereavement leave?")। কয়টা "not found" হয়? বাকিগুলোতে উত্তর কী?
5. **Burn Rate Math:** ৯৯% SLO, ঘণ্টায় ১২টা request, একটা 503। 1h burn rate কত? এতে page হওয়া উচিত? Low traffic-এ কী করবেন?
6. **Drift Drill:** দুই সপ্তাহের প্রশ্নের দৈর্ঘ্যের বণ্টন বানান (একটায় গড় ৪০ অক্ষর, অন্যটায় ৯০)। KS test আর Wasserstein distance কী বলে?

---

## References

* Google SRE Book — Monitoring Distributed Systems: https://sre.google/sre-book/monitoring-distributed-systems/ · SRE Workbook — Alerting on SLOs: https://sre.google/workbook/alerting-on-slos/
* Ollama API (chat/generate timing fields) — https://docs.ollama.com/api/chat · https://docs.ollama.com/api/generate · Ollama FAQ — https://docs.ollama.com/faq
* OpenTelemetry GenAI semantic conventions repo — https://github.com/open-telemetry/semantic-conventions-genai · client inference spans — https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md · attribute registry — https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md · metrics — https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md
* PyPI: opentelemetry-semantic-conventions — https://pypi.org/project/opentelemetry-semantic-conventions/ · opentelemetry-instrumentation-logging — https://pypi.org/project/opentelemetry-instrumentation-logging/ · prometheus-client — https://pypi.org/project/prometheus-client/
* Prometheus naming — https://prometheus.io/docs/practices/naming/ · Histograms and summaries — https://prometheus.io/docs/practices/histograms/
* Arize Phoenix self-hosting — https://arize.com/docs/phoenix/self-hosting · configuration — https://arize.com/docs/phoenix/self-hosting/configuration
* Langfuse SDK overview — https://langfuse.com/docs/observability/sdk/overview · v3→v4 — https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4 · v4→v5 — https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5 · self-hosting — https://langfuse.com/self-hosting · LLM-as-a-judge — https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge
* OpenLIT — https://docs.openlit.io/latest/sdk/overview
* RAGAS paper — https://arxiv.org/abs/2309.15217 · faithfulness — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/ · context precision — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/ · TruLens RAG triad — https://www.trulens.org/getting_started/core_concepts/rag_triad/
* ALCE (citation evaluation) — https://arxiv.org/abs/2305.14627
* Zheng et al., Judging LLM-as-a-Judge — https://arxiv.org/abs/2306.05685 · Panickssery et al., LLM evaluators recognize and favor their own generations — https://arxiv.org/abs/2404.13076
* Evidently data drift — https://docs.evidentlyai.com/metrics/customize_data_drift · DeepEval + Ollama — https://deepeval.com/integrations/models/ollama
* Course research report: `reports/LLM observability and monitoring.md`
