# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **Monitoring & Observability — Token/Cost Tracking, Drift ও Hallucination Monitoring, আর Langfuse**। Technical term English-এ, explanation Bangla-তে।

## কেন এই Topic দরকার?

Class 29-এ আমরা RAG bot-কে service বানিয়েছি। কিন্তু deploy করার পর প্রশ্ন আসে: প্রতিটা query-তে কত token গেল, কত সময় লাগল, উত্তর কি আসলেই document থেকে এসেছে, আর গত সপ্তাহের চেয়ে আজ কি খারাপ হচ্ছে? Log না থাকলে এসব প্রশ্নের উত্তর অনুমান। এই class-এর **Achievement**: প্রতিটা query-র cost, time আর quality score track করা একটা dashboard।

## শেখার Workflow

Problem বোঝা → মাপার জিনিস ঠিক করা (token, cost, latency, quality) → seam-এ observer বসানো → drift check → Langfuse দিয়ে dashboard। প্রতিটা section-এ তিনটা প্রশ্ন: **কোন problem solve করে? কীভাবে কাজ করে? বিকল্পের তুলনায় কখন better?**

---

## 1. Monitoring বনাম Observability

* **Monitoring** জানা প্রশ্নের উত্তর দেয়: latency কত, error কয়টা, cost কত। আগে থেকে ঠিক করা metric।
* **Observability** অজানা প্রশ্নের উত্তর দেয়: এই নির্দিষ্ট query-টা কেন ভুল উত্তর দিল? এর জন্য প্রতিটা request-এর পুরো trace (input, retrieved context, output, token) সংরক্ষিত থাকতে হয়।

LLM app-এ আলাদা সমস্যা হলো উত্তরটা "ঠিক কিনা" বলা কঠিন। Server 200 OK দিলেও উত্তর hallucinated হতে পারে। তাই ৩টা স্তর মাপি: **operational** (latency, error), **economic** (token, cost), **quality** (grounding, refusal rate)।

## 2. Token ও Cost Tracking

Token হলো LLM-এর billing ও latency-র একক। Ollama প্রতিটা response-এ দেয় `prompt_eval_count` (input) আর `eval_count` (output)। সমস্যা: Week 12-এর `generate()` শুধু text ফেরত দেয়, count হারিয়ে যায়।

**সমাধান:** `app/usage.py` `ollama.chat`-কে wrap করে count একটা `threading.local`-এ রাখে। `RealBackend.ask` সেটা `AskResponse.usage`-এ বসায় (`exclude=True`, তাই API response বদলায় না)। Week 12 কোড একটুও বদলায়নি।

**Cost:** local model-এর খরচ শূন্য। তাই আমরা "equivalent cloud cost" হিসাব করি: `(prompt × input_price + completion × output_price) / 1M`। এটা **illustrative** (গোল সংখ্যার price table); আসল দাম provider-এর pricing page থেকে নিতে হবে। এর কাজ: model বদলালে বা cloud-এ গেলে আনুমানিক বিল আগে থেকে বোঝা।

### Token আসে কোথা থেকে (reference)

| পদ্ধতি | Use case | Advantage | Limitation |
| --- | --- | --- | --- |
| Provider response field (chosen) | Ollama/OpenAI সরাসরি | সঠিক, ফ্রি | client code-এ ধরতে হয় |
| Tokenizer দিয়ে গোনা (tiktoken) | response-এ count না থাকলে | provider-নিরপেক্ষ | model-ভেদে সামান্য ভুল |
| Word count × ফ্যাক্টর | দ্রুত অনুমান | কোনো dependency নেই | Bangla-তে অনেক ভুল |
| Proxy/gateway (Helicone, LiteLLM) | code না বদলে | সব app একসাথে | আলাদা process, latency যোগ |

## 3. Quality Score ও Hallucination Proxy

Ground truth ছাড়া hallucination সরাসরি মাপা যায় না, তাই সস্তা proxy ব্যবহার করি:

1. **Grounding overlap:** উত্তরের অর্থবহ শব্দের কত শতাংশ retrieved context-এ আছে। কম মানে model নিজে বানিয়েছে।
2. **Refusal detection:** উত্তরে "not found in the documents" থাকলে এবং retrieval আসলেই কিছু পায়নি (`found=False`), সেটা **correct refusal** (score 1.0)। Context থাকা সত্ত্বেও refuse করলে সন্দেহজনক (0.5)।

সীমা: paraphrase করা সঠিক উত্তরেও score কমে; context নিজে ভুল হলে ভুল উত্তরও grounded দেখায়। তাই এটা alert-এর সংকেত, চূড়ান্ত রায় নয়।

| Method | Use case | Advantage | Limitation |
| --- | --- | --- | --- |
| Token overlap (chosen) | প্রতি query-তে cheap check | ফ্রি, offline | semantic বোঝে না |
| Embedding similarity | paraphrase সহ | semantic ধরে | embedding call লাগে |
| LLM-as-judge | সূক্ষ্ম faithfulness | মানুষের কাছাকাছি | খরচ, judge নিজেও ভুল করে |
| RAGAS / DeepEval | পুরো evaluation suite | অনেক metric | dependency + judge LLM |
| Human review | golden set তৈরি | সবচেয়ে নির্ভরযোগ্য | ধীর |

## 4. Drift Monitoring

**Drift** মানে সময়ের সাথে আচরণ বদলে যাওয়া: নতুন ধরনের প্রশ্ন, document বদল, model আপডেট, বা hardware ধীর হওয়া। আমাদের সরল পদ্ধতি: প্রথম N query = **baseline**, সর্বশেষ N query = **recent window**; latency ৫০%-এর বেশি বাড়লে বা quality ০.১৫-এর বেশি কমলে alert। `ponytail` comment-এ লেখা আছে: ছোট window-এর mean তুলনা, সত্যিকারের statistical test (KS test, PSI) পরে।

Drift-এর ধরন: **data drift** (ইনপুট প্রশ্নের বণ্টন বদলায়), **concept drift** (সঠিক উত্তরের সংজ্ঞা বদলায়, যেমন নীতি আপডেট), **performance drift** (latency/quality নামে)। আমরা শুধু performance drift সরাসরি মাপি; বাকি দুটো quality score কমার মাধ্যমে পরোক্ষে ধরা পড়ে।

## 5. Seam-এ Observer: কোড কীভাবে বসেছে

Class 29-এ `RagService._run()` প্রতিটা call-এর শেষে `CallRecord` বানিয়ে `service.observers`-এর প্রতিটা callable-কে দেয়। আমরা শুধু `Monitor.observer` যোগ করেছি (`get_service()`-এ); route বা backend-এ monitoring-এর কোড নেই।

```
CallRecord -> Monitor.metrics() -> SQLite insert -> sinks (Langfuse)
```

SQLite হলো **source of truth** (Docker ছাড়াও চলে), Langfuse optional sink। কারণ: observability tool নিজে ভেঙে গেলে আমাদের হিসাব হারানো চলবে না, আর tests-এ network লাগবে না। Sink ভাঙলেও request ভাঙে না।

## 6. Langfuse

Langfuse একটা open-source LLM observability platform: trace, token/cost, score, prompt management, dashboard। Self-host করলে data নিজের machine-এই থাকে।

| Tool | Use case | Advantage | Limitation |
| --- | --- | --- | --- |
| Langfuse (chosen) | self-host, open source | data নিজের কাছে, cost + score UI | compose-এ ৬টা service |
| LangSmith | LangChain ecosystem | গভীর integration | closed SaaS |
| Arize Phoenix | local eval + OpenTelemetry | হালকা | team feature কম |
| Helicone | proxy-based | code বদলায় না | deep trace কম |
| SQLite log (fallback) | zero infra | সহজ | UI/alert নেই |

Langfuse-এর মূল ধারণা: **trace** (একটা request) → ভেতরে **observation** (span বা generation) → trace-এ **score**। আমরা প্রতি `/ask`-এ একটা `generation` observation (model, input, output, `usage_details`, `cost_details`) আর একটা `grounding` score পাঠাই।

```python
gen = client.start_observation(name="rag-ask", as_type="generation", model=..., usage_details={"input": 120, "output": 40})
gen.score_trace(name="grounding", value=0.83)
gen.end()
```

Env না থাকলে `make_langfuse_sink()` `None` ফেরত দেয় এবং app স্বাভাবিক চলে। সীমা: observer request-এর পরে চলে, তাই Langfuse-এ span duration প্রায় শূন্য, আসল latency `metadata.latency_s`-এ।

## 7. Docker Compose Primer (Docker Week 18-এ পড়ানো হবে, তাই এখানে সংক্ষেপে)

* **Image** = প্যাকেজ করা software; **container** = সেই image-এর চলমান instance।
* **Compose file** (`docker-compose.yml`) একসাথে অনেক container-এর নকশা: `services` (কী কী চলবে), `ports` (`host:container`), `environment`, `volumes` (data টিকিয়ে রাখা), `depends_on`।
* Service-গুলো একই network-এ নাম ধরে একে অপরকে পায় (`postgres`, `redis`), তাই `localhost` নয়।
* Commands: `docker compose config` (file ঠিক কিনা), `up -d` (background-এ চালু), `logs -f`, `down` (বন্ধ; `-v` দিলে volume মুছে যায়)।

Langfuse v3-এর ৬টা service: `langfuse-web` (UI + API), `langfuse-worker` (async ingestion), `postgres` (metadata), `clickhouse` (trace analytics), `redis` (queue), `minio` (S3-compatible blob store)। কেন এত ভারী? trace লেখা বেশি, পড়া analytics-ভিত্তিক, তাই আলাদা কাজের জন্য আলাদা store।

## 8. কখন কী ব্যবহার করবেন

* ছোট prototype / offline: শুধু SQLite + `/stats`।
* Team, trace ঘাঁটা, prompt version: Langfuse self-host।
* Compliance-এ data বাইরে যাবে না: self-host (Langfuse) বা Phoenix।
* Cloud API ব্যবহার করলে: provider-এর usage field দিয়ে আসল cost, equivalent নয়।

## Interview প্রশ্ন

1. Monitoring আর observability-র পার্থক্য কী?
2. Ground truth ছাড়া hallucination কীভাবে track করবেন? এই পদ্ধতির দুর্বলতা কী?
3. Observer pattern কেন route-এর ভেতরে logging-এর চেয়ে ভালো?
4. Langfuse down থাকলে আপনার service কী করবে?
5. Data drift আর concept drift-এর পার্থক্য কী?

## Project

`week 15/Class 2 Project/` — `README.md` দেখুন। `python -m pytest -q` (16 tests, Ollama/Docker লাগে না)। Langfuse stack এই lecture লেখার সময় চালানো হয়নি (শুধু `docker compose config` যাচাই)।
