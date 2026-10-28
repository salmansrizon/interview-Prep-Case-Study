# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Privacy-First Healthcare RAG — Local Vector DB, Local Model, PII Redaction আর Abstention**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

> ⚠️ এই lecture আর project-এর clinic handbook **educational, fictional; not medical advice**। Regulation অংশটা ধারণা বোঝানোর জন্য, legal advice না।

## কেন এই Topic দরকার?

Week 12-এ আমরা employee handbook-এর ওপর RAG বানিয়েছি, Week 15-এ সেটাকে API বানিয়েছি। Healthcare-এ একই pipeline বসালেই কাজ শেষ না। একজন রোগী লেখেন: "আমার নাম করিম, ফোন ০১৭১২..., আমার তিন দিন ধরে জ্বর আর গায়ে র‍্যাশ — কী করব?" এই এক লাইনে নাম, ফোন নম্বর আর একটা স্বাস্থ্য-তথ্য একসাথে আছে। প্রশ্নটা cloud API-তে গেলে একটা third party সেটা পায়; log-এ গেলে যে কেউ log পড়ে সে পায়; আর উত্তর ভুল হলে মানুষটা ভুল সিদ্ধান্ত নেয়। Healthcare-এ তাই তিনটা প্রশ্ন আগে: **data কোথায় যায়, কোথায় থেকে যায়, আর KB-তে উত্তর না থাকলে assistant কী বলে।** এই ক্লাসে পুরো system local-এ বানানোর পদ্ধতি শিখব, আর শিখব "local" ঠিক কী guarantee করে আর কী করে না।

## শেখার সহজ Workflow

Healthcare আলাদা কেন → regulation-এর ধারণা (HIPAA, GDPR, Bangladesh, WHO) → threat model: data কোথায় leak হয় → local stack-এর guarantee আর সীমা → authored KB → section ধরে chunking → stable id → embedding আর cosine distance → model choice → Abstention threshold calibration → PII redaction (regex বনাম NER) → redaction-এর জায়গা: log আর embedding-এর আগে।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** প্রতিটা component-এর পাশে লিখুন "এখান থেকে data বাইরে যেতে পারে কিনা, আর disk-এ কী থেকে যায়।" Notebook চালানোর আগে লিখুন কোন প্রশ্ন abstain করবে বলে আপনার ধারণা — তারপর মেলান।

---

## 1. Topic: Privacy-First Healthcare RAG

এই ক্লাসে আমরা শিখব:

* **Healthcare কেন আলাদা** — sensitivity, ক্ষতির সম্ভাবনা, regulation।
* **Threat model** — cloud API, telemetry, log, cache, embedding, model memorization: data কোথায় কোথায় leak হয়।
* **Local stack** — ChromaDB + `paraphrase-MiniLM-L3-v2` + `llama3.1:8b` (Ollama): কী guarantee দেয়, কোথায় থামে।
* **Clinical document chunking** — section ধরে chunk, `doc#section-slug` id, stale id মোছা।
* **Embedding আর distance** — cosine space, normalized vector, `max_seq_length`-এর ফাঁদ।
* **Abstention** — threshold calibration, near-topic প্রশ্ন, overlap হলে কী করবেন।
* **PII redaction** — regex বনাম NER (Presidio), false positive/negative, Bangladeshi phone format, redaction-এর সঠিক জায়গা।

**Project Goal (Class 1-2 Project — Portfolio Project 16: Local Knowledge-Grounded Healthcare Assistant):** একটা কাল্পনিক clinic-এর নিজে-লেখা handbook (২৫-৪০ পাতা Markdown + generated PDF) থেকে প্রশ্নের উত্তর, পুরোটা নিজের মেশিনে: ChromaDB, local embedding, `llama3.1:8b`। এই ক্লাস, Class 33, বানায় **private RAG core** — KB, chunking, Chroma index, PII redaction আর Abstention threshold। পরের Class 34 তার ওপর safety layer (Red-Flag Escalation, refusal, citation check), evaluation আর FastAPI + Streamlit serving যোগ করবে।

---

## 2. Why It Is Related

Week 10-এ embedding আর cosine similarity, Week 12-এ ChromaDB, chunking আর RAG, Week 15-এ `max_distance` দিয়ে "not found" guardrail — এই ক্লাস সবগুলোকে একটা কঠিন domain-এ নিয়ে আসে। Week 14-এ CrewAI-র telemetry বন্ধ করতে তিনটা env flag লেগেছিল; এখানে একই mindset Chroma, Hugging Face আর Streamlit-এর জন্য: **library-র লুকানো network default আগে খুঁজুন।** Week 15-এর threshold calibration (chunk size বনাম distance gap) এখানে Abstention-এর ভিত্তি, কিন্তু এবার `cosine` space-এ, যাতে সংখ্যাগুলো পড়া যায়। Class 34 এই core-এর সামনে rule-based safety layer বসাবে, আর Week 18 পুরো stack-কে `docker compose up`-এ চালাবে।

---

## 3. How It Works

### 3.1 Healthcare কেন আলাদা

| দিক | সাধারণ handbook RAG | Healthcare RAG |
|---|---|---|
| **Data sensitivity** | ছুটির নিয়ম জানতে চাওয়া | প্রশ্ন নিজেই একটা health record: রোগ, ওষুধ, গর্ভাবস্থা, মানসিক স্বাস্থ্য। Password বদলানো যায়, diagnosis বদলানো যায় না |
| **ভুলের দাম** | ভুল ছুটির হিসাব | ভুল আশ্বাসে চিকিৎসা দেরি, ভুল dose, emergency-তে সময় নষ্ট |
| **"জানি না"-র মূল্য** | বিরক্তিকর | প্রায়ই সবচেয়ে নিরাপদ উত্তর — Abstention একটা correct outcome |
| **Regulation** | সাধারণ privacy আইন | health data-র জন্য আলাদা, কড়া নিয়ম (HIPAA, GDPR Art. 9) |
| **User** | কর্মী, অফিস সময়ে | ভীত, অসুস্থ, হয়তো রাত ৩টায়, হয়তো Bangla-য় |

### 3.2 Regulation-এর ধারণা (legal advice না)

উদ্দেশ্য আইন মুখস্থ করা না; design-এর সিদ্ধান্তে কোন ধারণাগুলো আসে সেটা চেনা।

**HIPAA (United States)** — রোগী-সংক্রান্ত identifiable তথ্য হলো **PHI**। নিয়মগুলো খাটে covered entity (hospital, clinic, insurer) আর তাদের **business associate**-এর ওপর — যে vendor PHI process করে। Cloud LLM vendor-কে PHI পাঠালে সাধারণত একটা **Business Associate Agreement (BAA)** লাগে। De-identification-এর দুটো পথ: **Safe Harbor** (১৮ ধরনের identifier সরানো — নাম, ফোন, email, MRN, ব্যক্তির সাথে যুক্ত তারিখ (বছর বাদে), ছোট এলাকার ঠিকানা ইত্যাদি) আর **Expert Determination** (একজন বিশেষজ্ঞ মেপে বলেন re-identification-এর ঝুঁকি খুব কম)। আরেকটা কাজের ধারণা **minimum necessary** — কাজের জন্য যতটুকু দরকার, তার বেশি না।

**GDPR (European Union)** — "data concerning health" একটা **special category** (Art. 9): explicit consent বা নির্দিষ্ট আইনি ভিত্তি ছাড়া process নিষেধ। Art. 5-এর নীতি: data minimisation, purpose limitation, storage limitation। উঁচু ঝুঁকির processing-এ **DPIA** (Art. 35)। আর একটা ফাঁদ: **pseudonymised data এখনো personal data** (Recital 26) — নাম সরিয়ে একটা id বসালেই GDPR-এর বাইরে যাওয়া যায় না।

**Bangladesh** — personal data protection-এর আলাদা, পূর্ণাঙ্গ আইন কয়েক বছর ধরে খসড়া ও ordinance পর্যায়ে বারবার বদলেছে, cyber security আইনও ২০১৮ থেকে কয়েকবার বদলেছে। কোনো product চালুর আগে বর্তমান অবস্থা একজন আইনজীবী বা প্রতিষ্ঠানের compliance team-এর কাছে যাচাই করুন। আইনের বাইরেও চিকিৎসকের রোগীর গোপনীয়তা রক্ষার পেশাগত দায় আছে — clinic-এর জন্য বানানো tool সেই দায় ভাঙতে পারে না।

**WHO (2021), Ethics and governance of AI for health** — ছয়টা নীতি: মানুষের autonomy রক্ষা; মানুষের কল্যাণ, নিরাপত্তা আর জনস্বার্থ; transparency, explainability আর intelligibility; responsibility আর accountability; inclusiveness আর equity; responsive আর sustainable AI। আমাদের design-এ এগুলোর সরাসরি রূপ: citation (transparency), fixed emergency message (safety), Bangla প্রশ্ন eval-এ রাখা (inclusiveness), প্রতিটা সিদ্ধান্তের `rule` log (accountability)।

> **Valid Point:** "আমরা local, তাই HIPAA/GDPR-এর ঝামেলা নেই" — ভুল। Local মানে third-party processor কম, কিন্তু access control, audit log, retention, breach procedure, consent — এগুলো process-এর প্রশ্ন, software-এর একার না।

### 3.3 Threat Model: Data কোথায় Leak হয়

| Leak point | কীভাবে ঘটে | Project-এ প্রতিকার |
|---|---|---|
| **Cloud LLM / embedding API** | প্রতিটা প্রশ্ন, retrieved passage সহ, vendor-এর server-এ; vendor-এর log আর retention policy আপনার নিয়ন্ত্রণে না | Ollama (`llama3.1:8b`) আর local sentence-transformers; কোনো API key নেই |
| **Telemetry** | Library নিজে usage data পাঠায়: Chroma-র anonymized telemetry, Hugging Face Hub-এর usage telemetry, Streamlit-এর usage stats (`browser.gatherUsageStats`) | Import-এর আগে env var, Chroma `Settings(anonymized_telemetry=False)`, Streamlit config-এ বন্ধ |
| **Model download** | প্রথম run-এ Hub থেকে weight আসে; পরে প্রতিবার version check | একবার download, তারপর `HF_HUB_OFFLINE=1` — Local Model-এর সংজ্ঞাই এটা |
| **Application log** | `logger.info(f"question: {q}")` — raw PII disk-এ, log shipper হয়ে আরও দূরে | Request-এর প্রথম লাইনে redact; log-এ শুধু redacted লেখা আর PII-র count |
| **Traceback / crash report** | Exception message-এ request body | Error handler শুধু request id আর error type log করে |
| **Cache** | Exact answer cache-এর key = প্রশ্ন; browser history; Streamlit session | Redacted প্রশ্নে cache key; healthcare-এ semantic cache না |
| **Vector store** | Patient note embed করলে embedding থেকে লেখা অনেকটা ফেরত আনা যায় (Morris et al. 2023 clinical note থেকে পূর্ণ নাম উদ্ধার দেখিয়েছে); Chroma folder OneDrive-এ sync হয় | KB-তে কোনো patient data নেই (authored); query embedding store হয় না; `data/` sync folder-এর বাইরে |
| **Model memorization** | Patient data দিয়ে fine-tune করলে model হুবহু অংশ মনে রাখতে পারে (Carlini et al. 2021) | Fine-tuning নেই; জ্ঞান KB-তে, model-এ না — RAG-এর privacy সুবিধা |
| **Network exposure** | `OLLAMA_HOST=0.0.0.0` দিলে Ollama পুরো network-এ খোলা, কোনো auth নেই; API-র port public | Ollama `127.0.0.1`-এ; API-র সামনে auth/TLS (Week 18) |
| **Device** | Laptop চুরি, unencrypted disk, shared Windows account | Disk encryption, আলাদা user — software-এর বাইরের কাজ, কিন্তু threat model-এ লিখতে হয় |

### 3.4 Local Stack: Guarantee আর সীমা

```
User question (raw, may contain PII)
      │
      ▼
[1] redact()  ── regex: email, phone (+880 / 01X), MRN, DOB, ID-number, "my name is X"
      │                     -> log + cache + everything below see only redacted text
      ▼
[2] embed (paraphrase-MiniLM-L3-v2, local CPU, weights cached on disk)
      │
      ▼
[3] ChromaDB PersistentClient (local folder, cosine space, telemetry off)
      │
      ├── best distance > threshold ──► Abstention: fixed message, NO LLM call
      ▼
[4] Ollama llama3.1:8b on 127.0.0.1:11434 → grounded answer with [n] citations
```

**Guarantee:** inference-এর সময় প্রশ্ন বা passage কোনো third party দেখে না; প্রথম download-এর পরে internet ছাড়াই চলে; retention আপনার নিয়ন্ত্রণে।

**সীমা:**

* **Local ≠ compliant।** Access control, audit, consent, retention policy আলাদা কাজ।
* **Local ≠ সঠিক।** `llama3.1:8b` hallucinate করে; local হওয়া উত্তরকে সত্য বানায় না — grounding আর Abstention লাগে।
* **Local ≠ নিরাপদ device।** Disk, OS account, backup, screenshot।
* **একবারের download একটা supply chain।** পরিচিত publisher-এর model, `safetensors`/GGUF, version pin।
* **Network-এ দিলে আর "local" না।** দুজন user একই server ব্যবহার করলেই auth, TLS, user-ভিত্তিক isolation লাগে।

**যাচাই করুন, বিশ্বাস না:** Wi-Fi বন্ধ করে (প্রথম download-এর পরে) eval চালান; Windows Resource Monitor-এর Network tab-এ দেখুন python আর ollama কোথায় connect করছে।

### 3.5 Authored Knowledge Base

Project-এর KB একটা কাল্পনিক clinic-এর নিজে-লেখা handbook, প্রতিটা পাতায় "educational, fictional; not medical advice"। কেন real data বা scraped লেখা না:

* **PHI শূন্য** — repo public করা যায়, demo-তে দেখানো যায়।
* **License পরিষ্কার** — নিজের লেখা।
* **Ground truth জানা** — প্রতিটা eval প্রশ্নের expected section আমরা জানি।
* **ইচ্ছাকৃত ফাঁক** — কিছু বিষয় (যেমন malaria-র চিকিৎসা) বাদ, near-topic Abstention পরীক্ষার জন্য।
* **গঠন নিয়ন্ত্রিত** — প্রতিটা document `#` title, ছোট ছোট `##` section, আর প্রতিটা বিষয়ে একটা "কখন clinician দেখাবেন" অংশ।

### 3.6 Section ধরে Chunking

Fixed-size chunk clinical লেখায় বিপজ্জনক: "এই লক্ষণগুলো থাকলে হাসপাতালে যান: ..." list মাঝখানে কাটা পড়লে retrieval অর্ধেক list আনে, আর উত্তর পড়ে মনে হয় বাকি লক্ষণ গুরুত্বহীন। নিয়ম: **একটা `##` section = একটা chunk**, আর chunk-এর শুরুতে `Title - Heading`।

```python
def chunk_by_section(doc_id, markdown):
    title, chunks = "", []
    for block in re.split(r"(?m)^(?=#{1,2} )", markdown):
        head, _, body = block.strip().partition("\n")
        if head.startswith("# "):
            title = head[2:].strip()
        elif head.startswith("## "):
            heading = head[3:].strip()
            text = f"{title} - {heading}\n{body.strip()}"           # heading-ও embedding-এ ঢোকে
            chunks.append({"id": f"{doc_id}#{slug(heading)}", "text": text,
                           "meta": {"doc": doc_id, "section": heading,
                                    "sha": hashlib.sha256(text.encode()).hexdigest()[:12]}})
    return chunks
```

**`max_seq_length`-এর ফাঁদ:** `paraphrase-MiniLM-L3-v2`-এর `max_seq_length` **128 token** — section-এর প্রথম ~১০০ শব্দই embedding-এ ঢোকে, বাকিটা চুপচাপ কাটা পড়ে। লম্বা section-এর শেষের "কখন হাসপাতালে যাবেন" অংশ তাহলে retrieval-এ অদৃশ্য। তাই section ছোট রাখুন, অথবা লম্বা section paragraph ধরে ভাগ করে প্রতিটা ভাগে heading prefix দিন (`doc#section-slug-2`)।

### 3.7 Stable ID

| Id scheme | নতুন section ঢোকালে | একটা শব্দ বদলালে | Citation/eval label-এ |
|---|---|---|---|
| Positional (`chunk_0`, `chunk_1`) | পরের সব id সরে যায় | একই | ভাঙে, আর নিঃশব্দে ভুল লেখা দেখায় |
| Content hash (Week 15) | একই | id বদলায় | Typo fix-এও ভাঙে |
| **`doc#section-slug`** + `sha` metadata | একই | id একই, `sha` বদলায় | স্থির; `sha` বলে কোনটা re-embed করতে হবে |

Notebook-এ handbook v2-তে (একটা section যোগ, একটা শব্দ বদল, একটা section সরানো) positional scheme-এ ৭টার মধ্যে ৭টা id বদলেছে, content hash-এ ১টা, slug-এ ০টা।

**Stale id মুছতেই হবে।** `upsert` নতুন বা বদলানো chunk লেখে, কিন্তু KB থেকে সরানো section vector store-এ থেকে যায় — সরানো (হয়তো পুরোনো বা ভুল) medical লেখা retrieval-এ আসতেই থাকে। Re-ingest-এ: `stale = old_ids - new_ids` → `collection.delete(ids=stale)`।

### 3.8 Embedding আর Distance

Week 15-এ Chroma-র default `l2` (squared L2) space ছিল, MiniLM-এর normalize না করা vector-এ — distance 25-26, যার কোনো স্বাভাবিক স্কেল নেই। এখানে:

```python
client = chromadb.PersistentClient(path="data/chroma_db", settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection("clinic_kb", configuration={"hnsw": {"space": "cosine"}})
vectors = model.encode(texts, normalize_embeddings=True).tolist()
```

* **Cosine distance = 1 − cosine similarity।** 0 = একই দিক, ১-এর কাছে = সম্পর্কহীন। Normalized vector-এ cosine, dot product আর L2 একই ক্রম দেয়; শুধু সংখ্যার স্কেল আলাদা।
* **Space collection তৈরির সময়ই ঠিক হয়।** পরে বদলাতে নতুন collection বানিয়ে re-embed।
* **Threshold model-নির্ভর।** একই KB-তে অন্য embedding model মানে নতুন বণ্টন, নতুন calibration।
* **Query-ও একই model দিয়ে।** Index একটা model-এ, query আরেকটায় — distance অর্থহীন। Model-এর নাম collection metadata-তে রাখুন।

### 3.9 Model Choice

**Embedding model:**

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| `sentence-transformers/paraphrase-MiniLM-L3-v2` (chosen) | ছোট local KB, শুধু CPU | ~61 MB, 384-dim, দ্রুত; course standard (ADR 0004) | General English; `max_seq_length` 128; medical synonym-এ দুর্বল |
| `sentence-transformers/all-MiniLM-L6-v2` | General semantic search | ভালো quality, `max_seq_length` 256 | একটু বড় ও ধীর; তবু general domain |
| `pritamdeka/S-PubMedBert-MS-MARCO` | Biomedical retrieval | PubMedBERT + MS MARCO; medical vocabulary আর synonym | BERT-base (768-dim, ~110M param), CPU-তে কয়েক গুণ ধীর; research লেখার ভাষা, রোগীর সাধারণ ভাষা না |
| `NeuML/pubmedbert-base-embeddings` | Medical literature search | PubMed-এ tune করা embedding | একই আকার; literature-এর জন্য, lay প্রশ্নে যাচাই করতে হয় |
| `intfloat/multilingual-e5-small` | Bangla + English প্রশ্ন | এক model-এ বহু ভাষা, `max_seq_length` 512 | `query: ` / `passage: ` prefix লাগে; MiniLM-L3-এর চেয়ে ভারী |
| Cloud embedding API | Managed scale | নিজের compute লাগে না | প্রতিটা প্রশ্ন বাইরে যায় — privacy-first-এর বিপরীত |

**LLM:**

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| `llama3.1:8b` via Ollama (chosen) | Grounded উত্তর, instruction মানা | Course standard (ADR 0006); Ollama-তে ~4.9 GB; ভালো instruction following | Medical-tuned না; KB-র বাইরের জ্ঞান মেশাতে পারে → grounding prompt + Abstention + citation check |
| `llama3.2:3b` | ৮ GB RAM-এর মেশিন | দ্রুত, ছোট | Citation format আর "not in KB" নির্দেশ মানায় দুর্বল |
| BioMistral-7B | Medical QA research | Mistral-এর ওপর PubMed Central Open Access দিয়ে আরও pretraining | Model card নিজেই production বা পেশাগত health কাজে ব্যবহার করতে নিষেধ করে; official Ollama library-তে নেই |
| Meditron-7B (Ollama `meditron`) | Medical research | Llama 2-এর ওপর clinical guideline আর PubMed দিয়ে pretraining | Context মাত্র 2K token; model card ব্যাপক alignment আর real-world trial ছাড়া medical application-এ deploy করতে নিষেধ করে |
| Cloud frontier model | সর্বোচ্চ quality | সবচেয়ে শক্তিশালী reasoning | Data বাইরে যায়; BAA/DPA লাগে; প্রতি call খরচ |

> **Valid Point:** Medical-tuned model মানেই নিরাপদ না। RAG-এ আমরা চাই model **KB থেকে** উত্তর দিক, নিজের স্মৃতি থেকে না। যে model বেশি medicine "জানে", সে KB-তে উত্তর না থাকলেও আত্মবিশ্বাসী উত্তর দিতে বেশি প্রলুব্ধ — Abstention কঠিন হয়। Model বাছাই করুন eval-এ মেপে: route accuracy, citation validity, "not in KB" মানা।

### 3.10 Abstention Threshold Calibration

Abstention: সবচেয়ে কাছের chunk-ও threshold-এর বাইরে → LLM না ডেকে fixed message: *"I don't have information about this in the clinic handbook. Please ask a clinician or the front desk."* এতে দুটো লাভ: model-কে KB-র বাইরের প্রশ্নে "সাহায্য" করার সুযোগই দেওয়া হয় না, আর laptop-এ এক-দুই সেকেন্ডের LLM call বাঁচে।

পদ্ধতি:

1. **On-topic set** — KB-তে উত্তর আছে এমন প্রশ্ন, রোগীর ভাষায় (paraphrase সহ)।
2. **Off-topic set** — দুই ধরনের: একদম অন্য বিষয় ("capital of France") আর **near-topic** (একই domain, KB-তে নেই: malaria-র চিকিৎসা, clinic-এর parking)। দ্বিতীয়টাই আসল পরীক্ষা।
3. প্রতিটা প্রশ্নের **best distance** মাপুন। On-topic-এর সবচেয়ে খারাপ < off-topic-এর সবচেয়ে ভালো হলে মাঝে একটা gap — threshold সেখানে।
4. **Overlap হলে** — healthcare-এ abstain-এর দিকে: threshold off-topic-এর সবচেয়ে ভালোর একটু নিচে। কিছু on-topic প্রশ্ন abstain করবে; সেটা ঠিক করুন data দিয়ে (chunking, section ভাগ, KB-তে লেখা যোগ), threshold সরিয়ে না।
5. **Model, chunking বা KB বদলালেই আবার calibrate।**

Notebook-এ (lexical fallback দিয়ে) on-topic worst 0.742, off-topic best 0.828 (near-topic parking প্রশ্ন), threshold 0.785। এই সংখ্যা fallback-এর; real embedding-এ নিজে মাপতে হবে। আর notebook দেখায় fallback-এর অন্ধতা: "My stools are watery, what should I drink?" আর একই প্রশ্ন Bangla-য় — দুটোরই উত্তর KB-তে আছে, তবু দুটোই abstain। এটা retrieval-এর দোষ, threshold-এর না; semantic বা multilingual embedding লাগে।

Abstention-এর দ্বিতীয় স্তর prompt-এ: threshold পেরোলেও passage প্রশ্নের উত্তর না দিলে model একটা নির্দিষ্ট sentinel (`NOT_IN_KB`) ফেরত দেবে, আর code সেটাকে Abstention হিসেবে ধরবে (Class 34)।

### 3.11 PII Redaction: Regex বনাম NER

কী মুছব: সরাসরি identifier — নাম, ফোন, email, MRN, জন্মতারিখ, NID/passport-ধরনের সংখ্যা। লক্ষণ বা রোগের নাম মুছব না — KB খুঁজতে সেগুলোই লাগে।

```python
PII_PATTERNS = [   # order matters: specific first, generic last
    ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    ("MRN",   r"(?P<keep>(?i:\bMRN|\bmedical record (?:number|no\.?))[\s:#-]*)[A-Za-z0-9][A-Za-z0-9-]{3,}"),
    ("DOB",   rf"(?P<keep>(?i:\bDOB|\bdate of birth|\bborn(?: on)?)[\s:]*){DATE}"),
    ("PHONE", rf"(?<![\d+])(?:\+?88{SEP})?0{SEP}1[3-9](?:{SEP}\d){{8}}(?!\d)"),   # 01712345678, +880 1712-345678
    ("PHONE", r"(?<![\d+])\+\d{1,3}(?:[\s-]?\d){6,12}(?!\d)"),
    ("ID",    r"(?<!\d)\d(?:[\s-]?\d){7,16}(?!\d)"),                                 # NID: 10/13/17 digits
    ("NAME",  r"(?P<keep>(?i:\bmy name is|\bthis is|\bi am called)\s+)[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"),
    ("NAME",  r"(?P<keep>আমার নাম\s+)[^\s,।]+"),
]
```

**Bangladesh-এর format:** mobile নম্বর `01` + operator digit (`3`-`9`) + ৮ digit = ১১ digit; international রূপে `+880 1XXX-XXXXXX` (শুরুর `0` বাদ)। মানুষ space বা dash দেয় (`017 1234 5678`, `01712-345678`), আর Bangla digit-এ লেখে (`০১৭১২৩৪৫৬৭৮`)। তাই আগে `str.translate` দিয়ে Bengali digit → ASCII, তারপর pattern। আরেকটা সূক্ষ্ম ব্যাপার: Python-এর `\d` str pattern-এ যেকোনো Unicode digit মেলায়, Bengali digit-ও — normalize না করলে কিছু pattern মেলে, কিছু মেলে না, আর বোঝা যায় না কেন।

| | Regex (project) | NER — Microsoft Presidio (regex + spaCy NER + context word) |
|---|---|---|
| **কী ধরে** | নির্দিষ্ট আকার: phone, email, MRN, ID, trigger-সহ নাম | তার ওপর trigger ছাড়া নাম, ঠিকানা, প্রতিষ্ঠান |
| **Speed / dependency** | Microsecond, শুধু stdlib | spaCy model (`en_core_web_lg` কয়েকশো MB), ধীর |
| **Explainable** | প্রতিটা match একটা লেখা pattern | Model score; কেন ধরল বা ছাড়ল বোঝা কঠিন |
| **Bangla** | নিজে pattern লিখলে | Default model English; Bangla নামে দুর্বল, আলাদা model/recognizer লাগে |
| **ব্যর্থতা** | FN: ছোট হাতের নাম, trigger ছাড়া নাম; FP: "This is Urgent" | FN: অপরিচিত নাম; FP: রোগ বা ওষুধের নামকে PERSON ভাবা |

Notebook-এর ১৯টা test-এর মধ্যে ১৫টা ঠিক, আর বাকি চারটা ইচ্ছা করে রাখা: দুটো FN ("my name is rahim", "call my son Karim"), একটা FP ("This is Urgent"), আর একটা ভুল label (ফাঁক দেওয়া phone `[ID]` হিসেবে মুছেছে — value তবু লুকানো)। **FN আর FP-এর দাম আলাদা:** redaction-এ FN মানে PII log-এ পৌঁছায় — একটা privacy breach; FP মানে প্রশ্নের একটা শব্দ হারায় (সাধারণত retrieval-এ প্রভাব নেই)। তাই redaction-এ FP-এর দিকে ঝুঁকুন, আর FN-গুলো মেপে লিখে রাখুন — "আমাদের redactor trigger ছাড়া নাম ধরে না" একটা documented limit।

Presidio-তে Bangladeshi phone যোগ করা এক লাইনের কাজ:

```python
bd_phone = PatternRecognizer(supported_entity="BD_PHONE", patterns=[Pattern("bd_mobile", BD_PHONE_REGEX, 0.7)],
                             context=["phone", "mobile", "call", "number"])     # কাছের শব্দ score বাড়ায়
analyzer = AnalyzerEngine(); analyzer.registry.add_recognizer(bd_phone)
```

### 3.12 Redaction-এর জায়গা: Log আর Embedding-এর আগে

Redaction request-এর **প্রথম লাইন**। তার পরে log, cache key, red-flag rule, retrieval, LLM prompt, trace — সবাই শুধু redacted লেখা দেখে। Log-এ value না, শুধু count: `{"q": "My name is [NAME], [PHONE]. How do I mix ORS ...", "pii": {"NAME": 1, "PHONE": 1}, "route": "answered"}` — কতজন PII লিখছে মাপা যায়, কে লিখছে না।

দুটো সূক্ষ্মতা notebook-এ দেখা যায়:

* **নাম-নম্বর retrieval-এ কাজে লাগে না** — redaction-এর আগে-পরে একই top chunk।
* **কিন্তু placeholder-ও text।** `[PHONE]` retriever-এর কাছে "phone" শব্দ; notebook-এ একটা প্রশ্নের top chunk বদলে "Book ... by phone" লেখা section হয়ে গেছে। খোঁজার আগে placeholder বাদ দেওয়া যায়, আর eval set-এ PII-সহ প্রশ্ন রাখতেই হয়।

আর যদি কখনো patient note embed করতে হয় (আমাদের project-এ না): **embed-এর আগে redact**, কারণ embedding থেকে লেখা, এমনকি নাম, অনেকটা ফেরত আনা যায়।

---

## 4. Details & Valid Points

### 4.1 Offline/Privacy Switch-এর তালিকা

| Component | Switch | কখন |
|---|---|---|
| ChromaDB | `ANONYMIZED_TELEMETRY=False` env, বা `Settings(anonymized_telemetry=False)` | Import/client তৈরির আগে; দুটোই দিন |
| Hugging Face Hub | `HF_HUB_DISABLE_TELEMETRY=1` | সবসময় |
| Hugging Face Hub | `HF_HUB_OFFLINE=1` | প্রথম model download-এর **পরে** — model না থাকলে নিঃশব্দে download না করে error |
| Streamlit | `.streamlit/config.toml` → `[browser] gatherUsageStats = false` | UI চালানোর আগে |
| Ollama | `OLLAMA_HOST=127.0.0.1:11434` (default); কখনো `0.0.0.0` না | Server চালানোর সময় |
| Application | Log formatter শুধু redacted text; request body কখনো log না | সবসময় |

### 4.2 Abstention কেন "সফলতা"

Abstention-কে failure গুনলে team threshold বাড়িয়ে "answer rate" বাড়াতে চাইবে — আর KB-র বাইরের প্রশ্নে model-এর স্মৃতি থেকে medical উত্তর আসবে। তাই eval-এ off-topic প্রশ্নের **expected route = abstained**; সেখানে abstain করা সঠিক উত্তর। Metric দুটো আলাদা রাখুন: on-topic প্রশ্নে কত ভাগ answered (helpfulness), off-topic প্রশ্নে কত ভাগ abstained (safety)।

### 4.3 Hybrid Retrieval আর Bangla

Lexical (BM25/TF-IDF) নির্দিষ্ট শব্দে ভালো ("ORS", "MRN"), semantic paraphrase-এ ("watery stools" ≈ diarrhoea)। Healthcare KB-তে দুটো একসাথে, অর্থাৎ hybrid retrieval, প্রায়ই সবচেয়ে ভালো। Bangla-র জন্য তিনটা পথ: multilingual embedding (`multilingual-e5-small`), KB-র Bangla সংস্করণ, বা প্রশ্নকে English-এ অনুবাদ (local model দিয়েই — cloud translation API আবার data বাইরে পাঠায়)। যেটাই নিন, eval-এ Bangla প্রশ্ন আলাদা করে মাপুন।

### 4.4 "Local Model" মানে কী

Course-এর glossary অনুযায়ী Local Model = weight একবার download, তারপর পুরোটা নিজের মেশিনে, আর কোনো network call না। Sovereign AI মানে inference-এর জন্য Cloud API Call এড়ানো, internet পুরো এড়ানো না। Healthcare-এ এই পার্থক্য interview-তে স্পষ্ট বলুন: "model download একবার, publisher আর version pin করা; প্রশ্ন কখনো বাইরে যায় না।"

---

## 5. Related Analogy: গ্রামের Clinic-এর তালাবদ্ধ আলমারি

একটা ছোট clinic। রোগীর ফাইল আর clinic-এর নিজের নিয়মের বই — দুটোই আছে।

| Concept | Clinic Analogy |
|---|---|
| **Cloud LLM API** | প্রতিটা রোগীর প্রশ্ন courier-এ শহরের অচেনা অফিসে পাঠিয়ে উত্তর আনা — courier কী কপি রাখে, জানেন না |
| **Local stack** | নিয়মের বই clinic-এর নিজের তালাবদ্ধ আলমারিতে; উত্তর clinic-এর ভেতরেই |
| **Telemetry** | Courier না, কিন্তু আলমারির কোম্পানি প্রতি মাসে "শুধু হিসাবের জন্য" একটা খাতা নিয়ে যায় — জেনে নিন খাতায় কী লেখা |
| **PII redaction** | Reception-এ প্রশ্নের কাগজ থেকে নাম-নম্বর কালো কালিতে ঢেকে তারপর ভেতরে পাঠানো |
| **Log-এ redacted text** | রেজিস্টার খাতায় "একজন রোগী ORS নিয়ে জানতে চাইলেন", নাম না |
| **Section ধরে chunk** | নিয়মের বইয়ের প্রতিটা অধ্যায় আলাদা কার্ডে — ছেঁড়া অর্ধেক পাতা না |
| **Stable id** | প্রতিটা কার্ডে স্থায়ী নম্বর ("Dengue-২"); নতুন কার্ড যোগ হলেও পুরোনো নম্বর বদলায় না |
| **Stale id মোছা** | বাতিল নিয়মের কার্ড আলমারি থেকে সরানো — নইলে কেউ একদিন পুরোনো নিয়ম পড়ে শোনাবে |
| **Abstention** | Receptionist সৎভাবে বলেন "এটা আমাদের বইয়ে নেই, ডাক্তারকে জিজ্ঞেস করুন" — আন্দাজে না |
| **Threshold calibration** | কতটা মিললে "বইয়ে আছে" বলবেন, সেটা কয়েকটা জানা প্রশ্ন দিয়ে আগে পরীক্ষা করে ঠিক করা |
| **Model memorization** | Receptionist-কে রোগীর ফাইল মুখস্থ করানো — তারপর অন্য কারো সামনে মুখ ফসকে বলে ফেলার ঝুঁকি |

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
|---|---|---|
| "Local, তাই compliant" | Access, audit, retention, consent বাদ পড়ে | Local = কম third party; compliance আলাদা কাজ |
| Telemetry switch import-এর পরে | Library ততক্ষণে default পড়ে ফেলেছে | Env var সবার আগে + code-এ `Settings(...)` |
| `HF_HUB_OFFLINE=1` প্রথম run-এর আগে | Model download হয় না, error | একবার download, তারপর offline flag |
| Raw প্রশ্ন log বা cache-এ | PHI disk-এ, log shipper-এ | Request-এর প্রথম লাইনে redact |
| Fixed-size chunk clinical লেখায় | "কখন হাসপাতালে যাবেন" list অর্ধেক | `##` section = chunk, heading prefix |
| লম্বা section | `max_seq_length` 128-এ শেষ অংশ embedding-এ নেই | Section ছোট, বা paragraph ধরে ভাগ |
| Positional chunk id | Section যোগ হলে সব citation আর eval label ভুল লেখা দেখায় | `doc#section-slug` + `sha` metadata |
| Stale id না মোছা | সরানো medical লেখা retrieval-এ আসতে থাকে | `old_ids - new_ids` → `delete` |
| Default `l2` space আর normalize না করা vector | Distance-এর স্কেল অর্থহীন, threshold বোঝা কঠিন | `{"hnsw": {"space": "cosine"}}` + `normalize_embeddings=True` |
| অন্য project থেকে threshold কপি | Model/chunking বদলালে বণ্টন বদলায় | নিজের KB-তে on/off/near-topic দিয়ে calibrate |
| Off-topic set-এ শুধু অসম্পর্কিত প্রশ্ন | Threshold খুব ঢিলা, near-topic প্রশ্নে medical hallucination | Near-topic প্রশ্ন রাখুন |
| Abstention-কে failure গোনা | Threshold বাড়িয়ে KB-র বাইরের উত্তর | Off-topic-এর expected route = abstained |
| Bengali digit normalize না করা | `০১৭১২...` redact হয় না | `str.translate` আগে |
| শুধু English test | Bangla প্রশ্নে retrieval আর redaction অন্ধ | Eval আর PII test-এ Bangla রাখুন |
| Medical-tuned model = নিরাপদ ধরে নেওয়া | স্মৃতি থেকে আত্মবিশ্বাসী উত্তর, Abstention কঠিন | Eval-এ মেপে বাছাই |
| `OLLAMA_HOST=0.0.0.0` | Auth-হীন LLM পুরো network-এ | `127.0.0.1`; বাইরের access শুধু API দিয়ে |

---

## 7. Interview Q&A

1. **Healthcare RAG সাধারণ RAG থেকে আলাদা কেন?** তিনটা কারণ: প্রশ্ন নিজেই sensitive health data; ভুল উত্তরে শারীরিক ক্ষতি হতে পারে; health data-র জন্য কড়া regulation। তাই design-এ যোগ হয়: local inference, request-এর শুরুতে PII redaction, KB-তে না থাকলে Abstention, citation, আর (পরের ক্লাসে) emergency-র জন্য rule-based escalation।
2. **"পুরোটা local" — কী guarantee করে, কী করে না?** Inference-এর সময় কোনো third party প্রশ্ন দেখে না, প্রথম download-এর পরে offline চলে, retention নিজের হাতে। Guarantee করে না: compliance (access control, audit, consent), সঠিকতা (local model-ও hallucinate করে), device security, আর network-এ খুললে isolation। আর telemetry, log, cache নিজে বন্ধ না করলে local stack-ও data ছড়ায়।
3. **একটা RAG system-এ PHI কোথায় কোথায় leak হতে পারে?** Cloud LLM/embedding API, library telemetry, application log আর traceback, answer cache আর browser history, vector store (embedding inversion, sync folder), fine-tuning করা model-এর memorization, খোলা Ollama port, আর device নিজে। প্রতিটার জন্য একটা প্রতিকার threat model table-এ লিখে রাখি।
4. **Clinical document কীভাবে chunk করবেন, আর কেন?** Section ধরে: একটা heading-এর পুরো লেখা এক chunk, শুরুতে title + heading। Fixed-size cut একটা warning-sign list অর্ধেক করে দিতে পারে, আর অর্ধেক list বিপজ্জনক। Embedding model-এর `max_seq_length` দেখি (MiniLM-L3-এ 128) — section তার চেয়ে লম্বা হলে paragraph ধরে ভাগ করে heading prefix দিই।
5. **Chunk id কেমন হওয়া উচিত?** `doc#section-slug`, আর content-এর hash metadata-তে। Positional id section যোগ হলেই সরে যায়, content-hash id typo fix-এও বদলায় — দুটোতেই citation আর eval label ভাঙে। Re-ingest-এ পুরোনো id-র মধ্যে যেগুলো নতুন build-এ নেই সেগুলো delete করি, নইলে সরানো লেখা retrieve হতে থাকে।
6. **Abstention threshold কীভাবে ঠিক করবেন?** On-topic, off-topic আর near-topic প্রশ্নের best distance মাপি; on-topic-এর সবচেয়ে খারাপ আর off-topic-এর সবচেয়ে ভালোর মাঝখানে threshold। Overlap হলে healthcare-এ abstain-এর দিকে ঝুঁকি, আর overlap ঠিক করি data দিয়ে (chunking, KB)। Model, chunking বা KB বদলালেই আবার calibrate করি। Threshold পেরোলেও model-এর `NOT_IN_KB` sentinel দ্বিতীয় স্তর।
7. **L2 না cosine — Chroma-তে কোনটা, কেন?** Normalized embedding + `cosine` space। Cosine distance 0 থেকে ১-এর আশেপাশে, তাই threshold পড়া আর ব্যাখ্যা করা সহজ; normalized vector-এ ক্রম L2-এর সাথে একই। Space collection তৈরির সময় ঠিক হয়, পরে বদলাতে re-embed লাগে।
8. **PII redaction-এ regex না NER?** দুটোই, স্তরে। Regex দ্রুত, explainable, dependency-হীন, আর নির্দিষ্ট format-এ (Bangladeshi phone, MRN, NID) নির্ভরযোগ্য। NER (Presidio + spaCy) trigger ছাড়া নাম ধরে, কিন্তু ভারী, English-কেন্দ্রিক, আর রোগ/ওষুধের নামে FP দেয়। যেটাই হোক, একটা labeled test set-এ FN আর FP মেপে limit document করি। Redaction-এ FN-এর দাম বেশি, তাই FP-এর দিকে ঝুঁকি।
9. **Redaction pipeline-এর কোথায় বসবে?** Request-এর প্রথম লাইনে, log, cache, rule, retrieval আর prompt-এর আগে। Log-এ value না, শুধু entity count। Patient note কখনো index করতে হলে embed-এর আগেই redact, কারণ embedding থেকে নামসহ লেখা অনেকটা উদ্ধার করা যায়।
10. **BioMistral বা Meditron কেন default না?** দুটোরই model card পেশাগত health ব্যবহারে নিষেধ বা সতর্কতা দেয়; Meditron-7B-এর context 2K token, RAG passage-এর জন্য ছোট; BioMistral official Ollama library-তে নেই। আর RAG-এ আমরা চাই model KB থেকে উত্তর দিক — যে model বেশি medicine জানে, সে স্মৃতি থেকে উত্তর দিতে বেশি প্রলুব্ধ। বাছাই eval-এ মেপে: route accuracy, citation validity।

---

## Achievement: Local Knowledge-Grounded Healthcare Assistant (Portfolio Project 16)

**`../Class 1-2 Project/`** — Week 17-এর Shared Project, দুই ক্লাস মিলে একটা system। **এই ক্লাস, Class 33, বানায় private RAG core:** কাল্পনিক clinic-এর authored handbook (Markdown + generated PDF, প্রতিটা পাতায় disclaimer), section-ভিত্তিক chunking আর stable id, telemetry বন্ধ ChromaDB index (`cosine`), `paraphrase-MiniLM-L3-v2` embedding, `llama3.1:8b` (Ollama) generation, request-এর শুরুতে regex PII redaction, আর calibrated Abstention threshold। Class 34 এর ওপর safety layer, evaluation আর FastAPI + Streamlit যোগ করে। এই ক্লাসের companion notebook **`privacy_first_rag_lecture.ipynb`** সম্পূর্ণ self-contained — project folder ছাড়াই চলে (optional Chroma আর Presidio cell বাদে সব cell offline)।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Threat Model Audit:** Project-এর প্রতিটা component (Streamlit, FastAPI, Chroma, sentence-transformers, Ollama) চালিয়ে Resource Monitor-এ network connection দেখুন। কোনটা প্রথম run-এ বাইরে যায়, কোনটা প্রতিবার?
2. **Near-Topic Hunt:** KB-র বিষয়ের কাছাকাছি কিন্তু KB-তে নেই এমন ১০টা প্রশ্ন লিখুন। কয়টা আপনার threshold পেরিয়ে LLM-এ পৌঁছায়? সেগুলোর জন্য কী বদলাবেন — threshold না KB?
3. **Truncation Test:** একটা section-এ ২০০ শব্দ লিখুন, শেষ বাক্যে একটা অনন্য তথ্য রাখুন। শুধু সেই তথ্য নিয়ে প্রশ্ন করলে section retrieve হয়? `max_seq_length` ব্যাখ্যা করুন।
4. **Redactor Red Team:** একজন বন্ধুকে ১০টা বাক্য লিখতে বলুন যেখানে PII লুকানো আছে (Banglish, ফাঁক দেওয়া নম্বর, "ami Rahim")। আপনার redactor কয়টা ধরে? FN-গুলো কি pattern দিয়ে ঠিক করা উচিত, নাকি limit হিসেবে লিখে রাখা?
5. **Embedding Swap:** `multilingual-e5-small`-এ বদলে (prefix সহ) Bangla প্রশ্নের retrieval মাপুন। Threshold কত বদলাল? English প্রশ্নে কি ক্ষতি হলো?
6. **Stale Content Drill:** একটা section KB থেকে মুছে শুধু `upsert` দিয়ে re-ingest করুন। সেই section-এর প্রশ্ন এখনো উত্তর পায়? তারপর stale id delete যোগ করে আবার দেখুন।

---

## References

* Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020) — https://arxiv.org/abs/2005.11401
* HHS, Guidance Regarding Methods for De-identification of PHI (Safe Harbor, Expert Determination) — https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html · HIPAA Privacy Rule — https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html
* GDPR, Regulation (EU) 2016/679 — https://eur-lex.europa.eu/eli/reg/2016/679/oj · Art. 9 (special categories) — https://gdpr-info.eu/art-9-gdpr/
* WHO, "Ethics and governance of artificial intelligence for health" (2021) — https://www.who.int/publications/i/item/9789240029200
* Microsoft Presidio docs — https://microsoft.github.io/presidio/ · Analyzer — https://microsoft.github.io/presidio/analyzer/ · Anonymizer — https://microsoft.github.io/presidio/anonymizer/
* Chroma docs: configure collections (HNSW space) — https://docs.trychroma.com/docs/collections/configure · Query and get — https://docs.trychroma.com/docs/querying-collections/query-and-get · Migration (client settings, telemetry flag) — https://docs.trychroma.com/docs/overview/migration
* Model cards — https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2 · https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 · https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO · https://huggingface.co/NeuML/pubmedbert-base-embeddings · https://huggingface.co/intfloat/multilingual-e5-small · https://huggingface.co/BioMistral/BioMistral-7B · https://huggingface.co/epfl-llm/meditron-7b
* Labrak et al., "BioMistral" (2024) — https://arxiv.org/abs/2402.10373 · Chen et al., "MEDITRON-70B" (2023) — https://arxiv.org/abs/2311.16079
* Ollama: llama3.1 — https://ollama.com/library/llama3.1 · meditron — https://ollama.com/library/meditron · FAQ (host binding, offline) — https://docs.ollama.com/faq
* Carlini et al., "Extracting Training Data from Large Language Models" (USENIX Security 2021) — https://arxiv.org/abs/2012.07805
* Morris et al., "Text Embeddings Reveal (Almost) As Much As Text" (EMNLP 2023) — https://arxiv.org/abs/2310.06816
