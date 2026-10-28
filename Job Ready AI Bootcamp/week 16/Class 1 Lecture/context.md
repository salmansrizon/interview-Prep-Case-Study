# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Market Data, Risk Metrics আর Financial News Sentiment — Offline Snapshot থেকে Risk-এর কাঁচামাল**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

"একটা agent stock-এর খবর খুঁজবে আর risk বলে দেবে" — শুনতে সহজ। কিন্তু agent-কে যদি সরাসরি internet আর একটা LLM দিয়ে ছেড়ে দেন, তিনটা সমস্যা একসাথে আসে: আজকের উত্তর কাল আর মেলে না (data বদলেছে), model নিজের মন থেকে "volatility 32%" বানিয়ে বলে, আর কেউ যাচাই করতে পারে না সংখ্যাটা কোথা থেকে এল। Finance-এ একটা ভুল সংখ্যা মানে ভুল সিদ্ধান্ত।

তাই এই সপ্তাহের system দুই ভাগে ভাঙা। এই ক্লাসে (Class 31) **data layer**: market data কী, local-এ কীভাবে রাখতে হয়, risk-এর তিনটা সংখ্যা (volatility, drawdown, VaR) code-এ কীভাবে মাপতে হয়, আর financial headline-এর sentiment একটা finance-specific classifier দিয়ে কীভাবে সংখ্যায় আনতে হয়। পরের ক্লাসে (Class 32) agent-রা এই সংখ্যাগুলো **ব্যাখ্যা** করবে — বানাবে না।

## শেখার সহজ Workflow

OHLCV → simple বনাম log return → Offline Snapshot আর ingestion → SQLite local store → volatility (√252) → max drawdown → VaR আর Expected Shortfall → ফাঁদ (look-ahead, survivorship, fat tail, split) → FinBERT classifier → caching → সময়ের সাথে sentiment aggregate।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** প্রতিটা metric হিসাব করার আগে নিজেকে জিজ্ঞেস করুন — "এই সংখ্যাটা *কোন তারিখ পর্যন্ত* data দেখে বানানো, আর সেই তারিখে আমি কি এটা সত্যিই জানতে পারতাম?" Finance data-র বেশিরভাগ bug এই প্রশ্নেই ধরা পড়ে।

---

## 1. Topic: Market Data, Risk Metrics আর Financial News Sentiment

এই ক্লাসে আমরা শিখব:

* **Market data-র ভাষা** — OHLCV, close বনাম adjusted close, trading day, simple আর log return।
* **Offline Snapshot pattern** — data একবার ingest, তারপর সব analysis local; "real-time" মানে ingestion, analysis না।
* **SQLite local persistence** — schema, primary key, idempotent upsert, as-of query, snapshot version।
* **তিনটা risk metric** — annualized volatility, maximum drawdown, 95% historical VaR (সাথে Expected Shortfall) — intuition, formula, হাতে-কলমে সংখ্যা, আর ফাঁদ।
* **FinBERT** — classifier আর LLM-এর পার্থক্য, probability, domain shift, batching, caching।
* **Sentiment aggregation** — headline থেকে ticker-এর একটা net sentiment, recency weight আর as-of filter সহ।

**Project Goal (Class 1-2 Project — Portfolio Project 15: Offline Market Intelligence Team):** একটা দুই-agent system — **News Scout Agent** local snapshot থেকে একটা ticker-এর খবর বের করে sentiment সহ brief লেখে, **Risk Analyst Agent** code-এ হিসাব করা 0-100 **Risk Score** ব্যাখ্যা করে, আর শেষে একটা PDF investment memo। পুরোটা offline, `llama3.1:8b`-এ। (`Course Module.pdf` এদের "Researcher Agent" আর "Analyst Agent" বলে; এই course-এ উপরের নাম দুটোই canonical।) এই ক্লাস বানায় তার data layer।

---

## 2. Why It Is Related

Week 10-এ `cardiffnlp/twitter-roberta-base-sentiment-latest` দিয়ে review-এর sentiment মেপেছিলাম — এখানে একই pipeline, কিন্তু finance-এর ভাষায় train করা model (domain shift-এর সমস্যাটা এবার সরাসরি দেখব)। Week 12-এ ChromaDB-তে local persistence শিখেছি; এখানে structured data-র জন্য SQLite — vector না, table। Week 14-এর CrewAI crew আর Flow পরের ক্লাসে এই data-র ওপর বসবে, আর Week 14-এর মূল শিক্ষা "যা code পারে, code করুক" এখানে নিয়ম হয়ে যায়: সব সংখ্যা code-এ। Week 15-এর FastAPI pattern দিয়ে project-টা API + thin Streamlit UI হিসেবে serve হয়।

---

## 3. How It Works

### 3.1 Market data-র ভাষা: OHLCV

প্রতিটা trading day-এর জন্য একটা row: **Open** (দিনের প্রথম দাম), **High**, **Low**, **Close** (শেষ দাম), **Volume** (কত share হাতবদল)। Risk metric সাধারণত শুধু close দিয়ে হয়।

* **Adjusted close** — split আর dividend-এর জন্য ঠিক করা দাম। 2-for-1 split-এ share দ্বিগুণ, দাম অর্ধেক — investor-এর কোনো ক্ষতি নেই, কিন্তু raw close-এ সেটা −50% "crash"। Return সবসময় adjusted price থেকে।
* **Trading day** — exchange শুধু কাজের দিনে খোলা, ছুটি বাদ; বছরে ~252 দিন। Weekend-এর জন্য দাম forward-fill করে return বানালে শূন্য-return-এর নকল দিন ঢোকে, volatility কৃত্রিমভাবে কমে যায়।
* **Timezone আর timestamp** — "close" একটা নির্দিষ্ট মুহূর্ত (যেমন 16:00 local)। News-এর timestamp সেই মুহূর্তের আগে না পরে, সেটা পরে গুরুত্বপূর্ণ হবে (§3.9)।

### 3.2 Simple বনাম Log Return

Price নিজে তুলনাযোগ্য না, return তুলনাযোগ্য:

* **Simple return** Rₜ = Pₜ / Pₜ₋₁ − 1 — একদিনে কত % লাভ/ক্ষতি। Portfolio-তে asset জুড়ে weighted যোগ করা যায়, তাই VaR-এর টাকার ক্ষতি এতেই মাপি।
* **Log return** rₜ = ln(Pₜ / Pₜ₋₁) = ln(1 + Rₜ) — **সময় জুড়ে যোগ হয়**: ln(P₂/P₀) = r₁ + r₂। তাই volatility আর statistics-এর জন্য।

**হাতে-কলমে:** 100 → 110 → 99। Simple: +10%, −10%, যোগফল 0% — অথচ আসলে 1% ক্ষতি। Log: +0.0953, −0.1054, যোগফল −0.0101 = ln(0.99) — ঠিক। ছোট move-এ দুটো প্রায় সমান; বড় move-এ ফাঁক বাড়ে (notebook-এ ORBQ-এর crash দিনে ~৩ percentage point)।

### 3.3 Offline Snapshot — কেন, আর কীভাবে

**Offline Snapshot** = বাইরের data-র একটা bundled, versioned local copy। Project default-এ শুধু এটাই পড়ে; agent আর analysis চলে local store-এর বিরুদ্ধে। একটা optional `refresh` ধাপ live source থেকে snapshot নতুন করতে পারে — সেটাই **একমাত্র** network ধাপ, আর downstream কিছুই তার ওপর নির্ভর করে না।

কেন এভাবে:

| কারণ | মানে |
|---|---|
| **Reproducibility** | একই snapshot = একই Risk Score = একই memo। ছয় মাস পরে audit-এ ঠিক সেই সংখ্যা আবার বানানো যায়। |
| **Sovereignty** | Analysis-এর সময় কোনো data বাইরে যায় না, কোনো cloud API নেই। Sensitive portfolio-র প্রশ্ন বাইরে leak হয় না। |
| **Testability** | Test-এ network লাগে না; fixed data-য় assert লেখা যায় ("NOVX-এর drawdown 50.78%")। |
| **Stability** | Free data source ভাঙে, rate-limit করে, format বদলায় — demo-র দিন সেটা হতে দেওয়া যায় না। |

"Real-time market data extraction" (syllabus-এর ভাষা) এখানে মানে **ingestion**-এর অংশ: refresh যত দ্রুত চালান, data তত নতুন। কিন্তু analysis সবসময় একটা নির্দিষ্ট, নামওয়ালা snapshot-এর ওপর — memo-তে "as of 2026-10-14 16:00, snapshot cf0a46a2a5c0" লেখা থাকে, "real-time" না।

Refresh-এর sketch (project-এর `refresh` command-এর মতো; এই lecture-এর notebook এটা চালায় না):

```python
import feedparser, yfinance as yf                    # only the refresh step imports these

def refresh(con, tickers, feeds):
    for t in tickers:
        df = yf.download(t, period="1y", auto_adjust=True, progress=False)   # adjusted prices, stated explicitly
        upsert_prices(con, to_rows(t, df))                                  # idempotent: same day = same row
    for url in feeds:
        for e in feedparser.parse(url).entries:
            upsert_news(con, stable_id(e.link), e.title, e.published)       # stable id -> agents can cite it
    write_meta(con, version=content_hash(con), as_of=last_close(con), source="yfinance+rss")
```

দুটো সতর্কতা: yfinance একটা unofficial library (Yahoo-র সাথে যুক্ত না, Yahoo-র শর্তে ব্যক্তিগত/গবেষণা ব্যবহার); RSS feed-এরও নিজস্ব শর্ত আছে। তাই repo-তে bundled snapshot **কাল্পনিক** ticker আর **নিজে লেখা** headline দিয়ে বানানো — কারো data redistribute হয় না, আর কোনো আসল company নিয়ে ভুল "risk" দাবিও হয় না।

### 3.4 SQLite Local Store

SQLite = একটা file-এ পুরো database; Python-এর stdlib-এ (`sqlite3`), server নেই, copy করলেই backup। Schema-র মূল সিদ্ধান্তগুলো:

```sql
CREATE TABLE prices (ticker TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL, low REAL,
                     close REAL NOT NULL, volume INTEGER, PRIMARY KEY (ticker, date));
CREATE TABLE news (id TEXT PRIMARY KEY, ticker TEXT NOT NULL, published_at TEXT NOT NULL,
                   source TEXT, headline TEXT NOT NULL);
CREATE TABLE sentiment_cache (text_hash TEXT NOT NULL, model TEXT NOT NULL, label TEXT,
                              p_pos REAL, p_neg REAL, p_neu REAL, PRIMARY KEY (text_hash, model));
CREATE TABLE snapshot_meta (key TEXT PRIMARY KEY, value TEXT);
```

* **Primary key `(ticker, date)`** — একই দিন দুবার থাকতে পারে না। Duplicate আটকানো app code-এর দায়িত্ব না, DB constraint-এর।
* **Upsert** — `INSERT ... ON CONFLICT(ticker, date) DO UPDATE SET ...`। Refresh দুবার চালালে row দ্বিগুণ হয় না (**idempotent**), আর provider কোনো দিনের দাম সংশোধন করলে নতুন মান বসে। Notebook-এ: দুই refresh-এর পরেও 780 row; `pandas.to_sql(if_exists="append")` ব্যবহার করলে primary key `IntegrityError` দিয়ে আটকায় (pandas 3-এ সেটা `pd.errors.DatabaseError`-এ মোড়ানো থাকে)।
* **Stable news id** (`NOVX-02`) — agent পরে এই id cite করবে, verifier মিলিয়ে দেখবে। Id কখনো row number না (refresh-এ বদলে যায়)।
* **Date ISO text** (`2026-10-14`, `2026-10-14T16:00`) — string হিসেবে sort করলেও সময়ের ক্রম ঠিক থাকে, তাই `WHERE date <= ?` সরাসরি as-of query।
* **`PRAGMA journal_mode=WAL`** — একজন writer (refresh) চলাকালীন অন্যরা (API, UI) পড়তে পারে।
* **Snapshot version = content hash** — একই data = একই version; একটা দামও বদলালে নতুন version। Memo-তে এটা লেখা থাকে, যাতে memo-টা হুবহু আবার বানানো যায়।

### 3.5 Annualized Volatility

Volatility = return কতটা ছড়ানো — দাম কতটা কাঁপে:

σ_daily = √( Σ (rₜ − r̄)² / (n − 1) ),  σ_annual = σ_daily × √252

**হাতে-কলমে:** ৫টা daily return +1%, −2%, +0.5%, +1.5%, −1%। Mean 0। বর্গের যোগ 1 + 4 + 0.25 + 2.25 + 1 = 8.5 (%²)। ÷ (n−1) = ÷4 → 2.125 → σ_daily = 1.458%। × √252 (≈15.87) → **σ_annual ≈ 23.1%**।

* **√252 কেন:** return যদি দিন-থেকে-দিন independent আর একই distribution-এর (i.i.d.) হয়, variance যোগ হয় — ২৫২ দিনের variance = 252σ², তাই std = √252·σ। এটা একটা assumption (§4.1)।
* **`ddof`-এর ফাঁদ:** `pandas.Series.std()` default `ddof=1` (sample), `numpy.std()` default `ddof=0` (population)। উপরের ৫টা সংখ্যায় 1.458% বনাম 1.304% — একই data, দুই উত্তর, আর ১০% পার্থক্য। ২৫০টা observation-এ পার্থক্য ছোট (~০.২%), কিন্তু test আর project-এর সংখ্যা না মিলিয়ে দিতে যথেষ্ট।
* **Log return দিয়ে**, simple দিয়ে না — দৈনিক ছোট move-এ পার্থক্য তুচ্ছ, কিন্তু convention একটাই রাখুন।

### 3.6 Maximum Drawdown

DDₜ = Pₜ / max(P₀…Pₜ) − 1,  MDD = minₜ DDₜ (সাধারণত ধনাত্মক % হিসেবে বলা হয়)

**হাতে-কলমে:** দাম 100, 120, 90, 110, 130, 104। Running max 100, 120, 120, 120, 130, 130। Drawdown 0, 0, **−25%**, −8.3%, 0, −20%। MDD = 25%।

Volatility উপরে-নিচে দুই দিকের নড়াচড়া সমান গোনে; drawdown শুধু peak থেকে পতন — investor যেটা আসলে অনুভব করে ("আমার সর্বোচ্চ মূল্যের এক-চতুর্থাংশ হারিয়েছিলাম")। Drawdown **path-dependent**: একই return-গুলো অন্য ক্রমে এলে drawdown বদলায়, volatility বদলায় না। আর এটা sample-এর দৈর্ঘ্যের ওপর নির্ভর করে — data যত লম্বা, খারাপ সময় পাওয়ার সম্ভাবনা তত বেশি।

### 3.7 Value at Risk আর Expected Shortfall

* **Historical VaR (95%, 1-day)** = daily simple return-এর 5th percentile, loss হিসেবে ধনাত্মক: VaR₉₅ = −Quantile₀.₀₅(R)। মানে: "ঐতিহাসিকভাবে ১০০ দিনের ৯৫ দিনে একদিনের ক্ষতি এর বেশি হয়নি।" বাকি ৫ দিনে **কত বেশি** — VaR বলে না।
* **Expected Shortfall** (ES, CVaR) = ওই সবচেয়ে খারাপ ৫% দিনের গড় ক্ষতি: ES₉₅ = −E[R | R ≤ Quantile₀.₀₅]। Tail-এর ভেতরের ছবি।
* **Parametric (normal) VaR** = −(μ − z·σ), z₀.₉₅ = 1.645, z₀.₉₉ = 2.326। দ্রুত, কিন্তু ধরে নেয় return normal।

**হাতে-কলমে:** σ_daily = 1.5%, μ ≈ 0 হলে normal VaR₉₅ ≈ 1.645 × 1.5% ≈ 2.47%। ১,০০,০০০ টাকার position-এ: "৯৫% দিনে ক্ষতি ~২,৪৭০ টাকার কম"। Notebook-এর NOVX: historical VaR₉₅ 3.03% (৩,০৩০ টাকা), কিন্তু ES₉₅ 4.44% (৪,৪৪০ টাকা) — খারাপ দিনগুলো VaR-এর চেয়ে প্রায় দেড় গুণ খারাপ।

**Sample size:** ২৫৯টা return-এর 5% মানে মাত্র ~১৩টা tail দিন; 99%-এ ~৩টা। একটা দিন বেশি বা কম এলেই VaR₉৯ লাফায়। এজন্য project ৯৫% ব্যবহার করে, ৯৯% না।

RiskMetrics (J.P. Morgan, 1994; Technical Document 1996) VaR-কে industry standard বানিয়েছিল — সেখানে volatility-র estimate exponentially-weighted (daily data-য় λ = 0.94), যাতে সাম্প্রতিক দিনের ওজন বেশি। আমাদের project সরল equal-weight historical পদ্ধতি নেয় — বোঝা আর test করা সহজ, কিন্তু regime বদলে ধীরে সাড়া দেয়।

### 3.8 FinBERT — LLM না, Classifier

FinBERT (Araci, 2019) = BERT-কে financial text-এ আরও pre-train করে Financial PhraseBank-এ (Malo et al., 2014: ~৪,৮০০টা financial news বাক্য, positive/negative/neutral label) fine-tune করা। `ProsusAI/finbert` তার HuggingFace version।

**কীভাবে কাজ করে:** headline → WordPiece token → BERT encoder → `[CLS]` vector → একটা linear layer → তিনটা logit → softmax → `positive`, `negative`, `neutral`-এর probability (যোগফল 1)। Config-এ `id2label = {0: positive, 1: negative, 2: neutral}`, সর্বোচ্চ 512 token।

```python
from transformers import pipeline
finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)   # ~440 MB, first run only
out = finbert(headlines, batch_size=8, truncation=True)
# out[i] = [{"label": "negative", "score": ...}, {"label": "neutral", ...}, {"label": "positive", ...}]
```

* **`top_k=None`** — তিনটা label-এরই probability; শুধু সেরাটা নিলে net sentiment (p_pos − p_neg) বানানো যায় না।
* **`batch_size`** — অনেক headline একসাথে model-এ; প্রতিটা batch সবচেয়ে লম্বা headline পর্যন্ত pad হয়, তাই লম্বা-ছোট মেশানো batch-এ অপচয়। আপনার machine-এ batch size বদলে মাপুন।
* **`truncation=True`** — 512 token-এর বেশি text-এ error না দিয়ে কেটে দেয়। Headline-এ সমস্যা নেই; পুরো article দিলে শুরুর অংশই শুধু পড়া হয়।
* **Offline:** প্রথম call-এ download, তারপর local cache। `HF_HUB_OFFLINE=1` set করলে নিশ্চিত — আর কখনো network-এ যাবে না।

**Classifier বনাম LLM — কেন এখানে classifier:**

| | FinBERT (classifier) | LLM zero-shot (`llama3.1:8b`) |
|---|---|---|
| Output | তিনটা probability, সবসময় একই format | একটা শব্দ বা বাক্য — parse করতে হয় |
| Determinism | একই input-এ একই output (sampling নেই) | Temperature 0-তেও নিশ্চয়তা নেই; prompt-এ একটা শব্দ বদলালে label বদলায় |
| গতি | CPU-তে একটা batch-এ অনেক headline | প্রতি headline-এ একটা generation |
| Risk Score-এ ব্যবহার | p_pos − p_neg সরাসরি একটা সংখ্যা | "positive" শব্দকে সংখ্যায় আনতে নিজের নিয়ম লাগে |
| Test | Fixed input → fixed output, assert সহজ | Mock ছাড়া test কঠিন |
| দুর্বলতা | শুধু train-করা ৩ label; context-এর বাইরে কিছু বোঝে না | ব্যাখ্যা দিতে পারে, নতুন label চাইলে prompt বদলালেই হয় |

Probability-কে "নিশ্চয়তা" ভাববেন না: আধুনিক neural network প্রায়ই **overconfident** — ভুল উত্তরেও 0.95 দেয় (Guo et al., 2017)। তবু probability deterministic আর graded, তাই একটা deterministic score-এর input হিসেবে LLM-এর label-এর চেয়ে অনেক ভালো (ADR 0009)।

#### Model Reference Table

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`ProsusAI/finbert`** (chosen, ADR 0009) | Financial headline-এর 3-class sentiment | Financial text-এ further pre-train + Financial PhraseBank fine-tune; native positive/negative/neutral; CPU-তেও চলে | ~440 MB; 2019-এর BERT-base, 512 token; ছোট বাক্যে train, লম্বা article-এ না; শুধু English |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` (Week 10) | Tweet / social media sentiment | Native 3-class, students আগে থেকে চেনে | Tweet domain — "guidance cut", "beats estimates" জাতীয় finance ভাষায় domain shift |
| `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | হালকা finance sentiment | DistilRoBERTa — ছোট, দ্রুত; Financial PhraseBank-এ fine-tune | Model card-এর ~0.98 accuracy একই dataset-এর split-এ মাপা — আশাবাদী; community model |
| `llama3.1:8b` zero-shot (Ollama) | আলাদা model ছাড়া label | নতুন download নেই; কারণ ব্যাখ্যা করতে পারে | ধীর, non-deterministic, calibrated probability নেই, unit-test কঠিন |

**Domain shift কেন এত বড় কথা:** Loughran & McDonald (2011) দেখিয়েছেন সাধারণ-উদ্দেশ্যের sentiment dictionary (Harvard GI) যে শব্দগুলোকে negative বলে, তার প্রায় তিন-চতুর্থাংশ financial filing-এ আসলে negative না — "liability", "tax", "cost", "capital" finance-এ নিরপেক্ষ শব্দ। উল্টো দিকে "guidance cut", "miss", "downgrade" সাধারণ text-এ নিরীহ, finance-এ খারাপ। Tweet-এ train করা model এই ভাষা চেনে না।

**Caching:** cache key = (headline-এর `sha256`, model name)। একই headline দ্বিতীয়বার কখনো model-এ যায় না (notebook-এ দ্বিতীয় run-এ 0 call)। Model বা তার version বদলালে key বদলায় — পুরোনো model-এর score ভুল করে নতুনের নামে চলে না।

### 3.9 সময়ের সাথে Sentiment Aggregate

একটা ticker-এর অনেক headline → একটা সংখ্যা:

sᵢ = p_pos,ᵢ − p_neg,ᵢ ∈ [−1, 1],  wᵢ = 0.5^(ageᵢ / h),  S = Σ wᵢ sᵢ / Σ wᵢ

* **Recency weight** — h = half-life। Project-এ h = ৩০ দিন: এক মাসের পুরোনো খবরের ওজন অর্ধেক। ছোট h নতুন খবরে দ্রুত সাড়া দেয়, কিন্তু একটা headline-এই দোলে।
* **As-of filter** — `published_at > as_of` হলে বাদ। Market close 16:00-এ, আর একটা খারাপ খবর 18:05-এ এলে, আজকের close-এর score-এ সেটা ঢোকা মানে ভবিষ্যৎ দেখা।
* **খবর না থাকলে** S = 0 (neutral), আর headline সংখ্যাটা memo-তে দেখান — "৩টা headline থেকে" আর "৩০টা থেকে" একই ভরসার না।

**হাতে-কলমে:** as-of থেকে ২ দিন আগের s = +0.80, ১০ দিন আগের s = −0.70, ১ দিন আগের s = −0.10। h = 30: weight 0.955, 0.794, 0.977 → S ≈ +0.04। h = 7: weight 0.820, 0.371, 0.906 → S ≈ +0.15। সাধারণ গড় 0.00। একই খবর, তিন উত্তর — h একটা documented policy, লুকানো constant না।

---

## 4. Details & Valid Points

### 4.1 Metric-এর ফাঁদ — বিস্তারিত

| ফাঁদ | কী হয় | উদাহরণ | প্রতিকার |
|---|---|---|---|
| **Look-ahead bias** | যে তারিখের risk বলছেন, তার পরের data হিসাবে ঢোকে | Notebook: ORBQ-এর drawdown "as of 2026-09-03" — শুধু তখন পর্যন্ত data-য় 31.5%, পুরো file থেকে 41.5% (পরের crash ঢুকে গেছে) | প্রতিটা query-তে `date <= as_of`; news-এ timestamp-সহ filter; full-sample normalization না |
| **Survivorship bias** | Universe-এ শুধু যারা টিকে আছে; delist/দেউলিয়া company গায়েব | আজকের index-এর stock দিয়ে ১০ বছরের backtest — ঝরে পড়ারা নেই, তাই risk কম, return বেশি দেখায় (Brown et al., 1992) | Point-in-time universe; delisted ticker রাখা |
| **Fat tail** | Return-এর tail normal-এর চেয়ে মোটা; normal VaR বড় ক্ষতি ছোট দেখায় | Notebook: NOVX (Student-t shock) VaR₉₉ historical 5.16% বনাম normal 4.20% | Historical VaR + ES; kurtosis দেখান |
| **একটা extreme দিন** | একটা crash σ ফুলিয়ে দেয়, বাকি বছর আড়াল হয় | ORBQ: normal VaR₉₉ (7.11%) historical-এর (6.25%) চেয়েও বড়, অথচ crash-এর আকার শুধু ES (6.89%) আর kurtosis (11.6)-এ দেখা যায় | একাধিক metric একসাথে; robust estimate |
| **√252 assumption** | Volatility clustering (শান্ত সপ্তাহের পরে শান্ত, ঝড়ের পরে ঝড় — Cont, 2001) আর autocorrelation-এ variance সময়ের সাথে সরল যোগ হয় না | ঝড়ের মাসের daily σ × √252 পুরো বছরকে ঝড়ো দেখায় | Annualized সংখ্যাকে "scale", "ভবিষ্যদ্বাণী" না; দীর্ঘ horizon সরাসরি মাপুন |
| **Estimation error** | এক বছরের data-য় volatility estimate-এর নিজস্ব error ~σ/√(2n) ≈ ৪-৫% | Notebook: generate করা σ 0.18, estimate 0.167 | Memo-তে দুই দশমিকের বেশি না; confidence interval বা range |
| **Unadjusted split** | Split দেখায় crash | Notebook: KTLX drawdown 10.5% → 55.2% | Adjusted price; split factor |
| **ddof অমিল** | pandas আর numpy ভিন্ন সংখ্যা দেয় | ৫টা return-এ 1.458% বনাম 1.304% | এক জায়গায় define, test-এ pin |
| **Forward-filled weekend** | নকল 0-return দিন volatility কমায় | Calendar day-এ resample | শুধু trading day |

### 4.2 VaR-এর সীমা আর Expected Shortfall

Artzner et al. (1999) দেখিয়েছেন VaR **coherent** risk measure না — বিশেষ করে subadditive না: দুটো position মিলিয়ে VaR আলাদা দুটোর যোগফলের চেয়ে বেশি হতে পারে, অর্থাৎ VaR কখনো diversification-কে শাস্তি দেয়। Expected Shortfall coherent, আর tail-এর আকার দেখে। এই কারণেই Basel Committee-র market risk framework (FRTB) internal model-এ 99% VaR থেকে **97.5% Expected Shortfall**-এ সরেছে। আমাদের Risk Score VaR নেয় (সহজ, ব্যাখ্যা করা যায়), কিন্তু memo-তে ES-ও দেখানো ভালো অভ্যাস — দুটোকে একসাথে score-এ দিলে একই tail দুবার গোনা হয়।

### 4.3 Sentiment-এর সীমা

* **Sentiment ≠ ভবিষ্যদ্বাণী।** Tetlock (2007): media-র pessimism বাজারে নিম্নমুখী চাপের পূর্বাভাস দেয়, যা পরে fundamentals-এর দিকে ফিরে আসে; অস্বাভাবিক উঁচু বা নিচু pessimism-এ trading volume বাড়ে। মানে sentiment একটা ঝুঁকির সংকেত, দামের দিকনির্দেশ না।
* **Headline বনাম article।** Headline ছোট, নাটকীয়; FinBERT ছোট বাক্যেই train করা — headline-এর জন্য ভালো মিল, পুরো article-এর জন্য না।
* **Negation আর মিশ্র খবর।** "not expected to miss", "costs rise but margins hold" — word list পুরোপুরি ব্যর্থ হয় (notebook Section 9); contextual model-এর অন্তত সুযোগ আছে, কিন্তু নিশ্চিত হতে হলে নিজের headline-এ মাপতে হয়।
* **Entity confusion।** একই নামের দুটো company, বা headline-এ প্রতিযোগীর খবর ("ORBQ-এর rival recall করেছে") — sentiment ঠিক, ticker ভুল। Ingestion-এ ticker mapping সাবধানে।
* **Syndication।** একই খবর পাঁচটা wire-এ এলে পাঁচ গুণ ওজন পায়। Normalized headline-এর hash দিয়ে dedupe।
* **কম headline।** ৩টা headline-এর গড় ভঙ্গুর; `S × n / (n + k)` জাতীয় shrinkage S-কে 0-এর দিকে টানে।

### 4.4 Metric Summary

| Metric | Formula | কী বলে | কী লুকায় |
|---|---|---|---|
| Annualized volatility | std(log r) × √252 | দাম কতটা কাঁপে | দিক (উপর/নিচ সমান), tail, path |
| Max drawdown | min(P / running max − 1) | Peak থেকে সবচেয়ে খারাপ পতন | কত দ্রুত, কতদিন; sample-দৈর্ঘ্যে নির্ভর |
| Historical VaR₉₅ | −Q₀.₀₅(simple R) | সাধারণ খারাপ দিনের সীমা | সীমার বাইরের দিন কত খারাপ |
| ES₉₅ | −mean(R ≤ Q₀.₀₅) | সবচেয়ে খারাপ ৫% দিনের গড় | ছোট sample-এ noisy |
| Net sentiment | Σwᵢ(p_pos − p_neg) / Σwᵢ | সাম্প্রতিক খবরের সুর | খবরের গুরুত্ব, সত্যতা, entity |

### 4.5 এই lecture-এর notebook-এর সংখ্যা (seed = 10, কাল্পনিক ticker)

| Ticker | Annual vol | Max drawdown | VaR₉₅ (1-day) | ES₉₅ | Net sentiment (fallback scorer) |
|---|---|---|---|---|---|
| KTLX (Kite Logistics) | 16.7% | 10.5% | 1.69% | 2.02% | +0.08 |
| NOVX (Nova Grid Energy) | 27.3% | 50.8% | 3.03% | 4.44% | −0.04 |
| ORBQ (Orbiq Semiconductor) | 50.2% | 41.5% | 4.63% | 6.89% | +0.11 (recall খবর as-of-এর পরে, বাদ) |

Sentiment column FinBERT-এর না, notebook-এর teaching fallback-এর — FinBERT চালালে বদলাবে। এই table-ই Class 32-এর Risk Score-এর input।

---

## 5. Related Analogy: আবহাওয়া অফিসের লগবই

> একটা আবহাওয়া অফিস প্রতিদিন তাপমাত্রা আর বৃষ্টি একটা বাঁধানো লগবইয়ে লেখে, আর পুরোনো পাতা কখনো ছেঁড়ে না।

| Concept | আবহাওয়া অফিস |
|---|---|
| **Offline Snapshot** | বাঁধানো লগবই — বিশ্লেষণ হয় বই থেকে, জানালা দিয়ে আকাশ দেখে না |
| **Refresh / ingestion** | প্রতি সন্ধ্যায় নতুন পাতায় আজকের মাপ লেখা — একমাত্র সময় যখন বাইরে তাকানো হয় |
| **Snapshot version** | বইয়ের edition নম্বর — রিপোর্টে লেখা থাকে কোন edition থেকে |
| **Volatility** | দিন-থেকে-দিন তাপমাত্রা কতটা লাফায় |
| **Max drawdown** | মৌসুমের সবচেয়ে গরম দিন থেকে সবচেয়ে বড় পতন |
| **VaR₉₅** | "১০০ দিনের ৯৫ দিনে বৃষ্টি ৫০ mm-এর কম" — বাকি ৫ দিন কেমন, বলে না |
| **Expected Shortfall** | ওই ৫টা ঝড়ের দিনের গড় বৃষ্টি |
| **Fat tail** | ঘূর্ণিঝড় — "স্বাভাবিক" অঙ্কে যত বিরল হওয়ার কথা, তার চেয়ে অনেক বেশি আসে |
| **Look-ahead bias** | কালকের খবরের কাগজ দেখে আজকের পূর্বাভাস লেখা |
| **Survivorship bias** | শুধু যে station-গুলো এখনো চালু, তাদের রেকর্ড দিয়ে "অঞ্চলের আবহাওয়া শান্ত" বলা — ঝড়ে ভেঙে যাওয়াগুলো বাদ |
| **FinBERT বনাম সাধারণ model** | অভিজ্ঞ আবহাওয়াবিদ বনাম সাধারণ সাংবাদিক — "নিম্নচাপ" শব্দটা দুজনের কাছে আলাদা মানে |
| **Sentiment cache** | একই বার্তা আবার এলে আগের ব্যাখ্যা খাতা থেকে, নতুন করে বিশ্লেষণ না |

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
|---|---|---|
| Analysis-এর সময় live API call | প্রতি run-এ আলাদা সংখ্যা, rate-limit, test-এ network | Snapshot পড়ুন; network শুধু refresh-এ |
| Raw close থেকে return | Split-এ নকল crash, dividend-এ নকল পতন | Adjusted price (`auto_adjust` স্পষ্ট), বা split factor |
| Weekend forward-fill | নকল 0-return, কম volatility | শুধু trading day |
| `ddof` না মেলানো | pandas আর numpy-র সংখ্যা আলাদা | এক function, test-এ pin |
| Full-file থেকে "as of" metric | Look-ahead; backtest মিথ্যা ভালো | প্রতিটা query-তে `<= as_of` |
| News-এর শুধু date, সময় না | Close-এর পরের খবর আজকের score-এ | Timestamp রাখুন, close-এর সাথে তুলনা |
| `to_sql(if_exists="append")` দিয়ে refresh | Duplicate row বা `IntegrityError` | Primary key + `ON CONFLICT DO UPDATE` |
| Row number-কে news id | Refresh-এ id বদলায়, পুরোনো citation ভুল জায়গায় যায় | Stable id (link-এর hash বা `TICKER-NN`) |
| Normal VaR-কে tail-এর সত্য ভাবা | Fat tail-এ বড় ক্ষতি ছোট দেখায় | Historical VaR + ES + kurtosis |
| 99% VaR এক বছরের data-য় | ২-৩টা দিন থেকে একটা "risk number" | 95%, বা লম্বা history, বা ES |
| Twitter-model finance-এ | "guidance cut" neutral, "liability" negative | FinBERT; নিজের headline-এ যাচাই |
| `top_k` ছাড়া pipeline | শুধু সেরা label — net sentiment অসম্ভব | `top_k=None` |
| Cache key-তে model নেই | Model বদলালেও পুরোনো score | Key = (text hash, model name/version) |
| Syndicated duplicate | একই খবর বহুগুণ ওজন | Normalized headline hash দিয়ে dedupe |
| Sentiment-কে signal ভাবা | "Positive news → কিনুন" | Sentiment risk-এর একটা input মাত্র; কোনো advice না |
| Memo-তে "real-time" দাবি | পাঠক ভাবে এখনকার data | "As of" timestamp + snapshot version |

---

## 7. Interview Q&A

1. **Simple আর log return-এর পার্থক্য কী, কোনটা কখন?** Simple = P₁/P₀ − 1, asset জুড়ে weighted যোগ হয় → portfolio আর VaR-এর টাকার ক্ষতি। Log = ln(P₁/P₀), সময় জুড়ে যোগ হয় → volatility, statistics, multi-period। ছোট move-এ প্রায় সমান; বড় move-এ ফাঁক বাড়ে। একটা pipeline-এ একটা convention, document করা।
2. **Volatility annualize করতে √252 কেন, আর কখন ভুল?** i.i.d. return-এ variance সময়ের সাথে যোগ হয়: ২৫২ দিনের variance = 252σ², তাই std × √252। Volatility clustering আর autocorrelation থাকলে খাটে না — তখন annualized সংখ্যা একটা scale, ভবিষ্যদ্বাণী না।
3. **Historical VaR ৯৫% = 3% মানে কী, আর কী বলে না?** ঐতিহাসিকভাবে ৯৫% দিনে একদিনের ক্ষতি 3%-এর কম ছিল। বাকি ৫% দিনে কত খারাপ — বলে না; সেটা Expected Shortfall। আর এটা অতীতের distribution, ভবিষ্যতের guarantee না।
4. **VaR-এর বদলে বা পাশে ES কেন?** VaR coherent না (subadditive না — Artzner et al.), tail-এর আকার দেখে না। ES tail-এর গড়, coherent; Basel FRTB internal model-এ 97.5% ES ব্যবহার করে। অসুবিধা: ছোট sample-এ আরও noisy।
5. **Look-ahead bias কী, risk pipeline-এ কোথায় লুকায়?** এমন data ব্যবহার যা সেই সময়ে জানা সম্ভব ছিল না। লুকায়: full-sample normalization, "as of" metric পুরো file থেকে, close-এর পরে প্রকাশিত খবর, পরে সংশোধিত দাম, আজকের universe দিয়ে অতীত বিচার, অর্থাৎ survivorship। প্রতিকার: সব query-তে `as_of`, timestamp-সহ news।
6. **Live API না ডেকে snapshot কেন? "Real-time" তাহলে কোথায়?** Reproducibility (একই input → একই score → audit করা যায়), sovereignty (analysis-এ কিছু বাইরে যায় না), testability (network ছাড়া test), stability (free source ভাঙে)। Real-time = ingestion-এর গতি; analysis সবসময় একটা নামওয়ালা, versioned snapshot-এ।
7. **SQLite-এ idempotent ingestion কীভাবে?** Natural primary key `(ticker, date)`, news-এর stable id, `INSERT ... ON CONFLICT DO UPDATE`। Refresh দুবার = একই row সংখ্যা; সংশোধিত দাম নতুন মান পায়। Snapshot version = content hash, memo-তে লেখা।
8. **Financial headline-এ Twitter-sentiment model কেন না?** Domain shift: finance-এর ভাষা ("guidance cut", "beats estimates", "liability") সাধারণ text থেকে আলাদা; Loughran-McDonald দেখিয়েছেন সাধারণ dictionary-র "negative" শব্দের বড় অংশ finance-এ নিরপেক্ষ। FinBERT financial text-এ pre-train আর Financial PhraseBank-এ fine-tune — native 3-class।
9. **Sentiment-এ LLM না classifier — কীভাবে বাছবেন?** Score-এর input হলে classifier: deterministic, দ্রুত, batch-যোগ্য, probability → সরাসরি সংখ্যা, test সহজ। LLM ধীর, non-deterministic, calibrated probability নেই — তবে ব্যাখ্যা বা নতুন label লাগলে কাজে আসে। Classifier-এর probability-ও overconfident হতে পারে, তাই নিজের data-য় যাচাই।
10. **অনেক headline থেকে একটা ticker-এর sentiment কীভাবে?** sᵢ = p_pos − p_neg, recency weight 0.5^(age/half-life), weighted গড়; as-of-এর পরের খবর বাদ; duplicate dedupe; কম headline-এ shrinkage; headline সংখ্যা দেখানো। Half-life একটা documented policy।
11. **FinBERT production-এ দ্রুত আর offline কীভাবে রাখবেন?** Model একবার load (process-প্রতি), headline batch-এ, `truncation=True`, score cache (text hash + model), `HF_HUB_OFFLINE=1`, test-এ classifier mock। Batch size আর throughput নিজের machine-এ মাপুন।
12. **Risk metric-এর সংখ্যা কতটা বিশ্বাস করবেন?** এগুলোও estimate: এক বছরে volatility-র error ~৪-৫%, VaR₉₅ মাত্র ~১৩টা দিন থেকে। তাই memo-তে যুক্তিসঙ্গত precision, একাধিক metric, আর data-র সময়সীমা স্পষ্ট।

---

## Achievement: Offline Market Intelligence Team (Portfolio Project 15)

**`../Class 1-2 Project/`** — এই সপ্তাহের Shared Project (দুই ক্লাসের একটাই project)। **এই ক্লাস (Class 31) বানায় তার data layer:** bundled Offline Snapshot (কাল্পনিক ticker-এর synthetic price + নিজে লেখা headline, `data/snapshot/`), optional `refresh` (yfinance + RSS — একমাত্র network ধাপ), SQLite local store, deterministic risk metric (annualized volatility, max drawdown, 95% historical VaR), আর FinBERT headline sentiment (locally cached) থেকে recency-weighted net sentiment। Class 32 এর ওপর বসায় Risk Score, News Scout Agent → Risk Analyst Agent crew, CrewAI Flow, verification আর PDF memo। Project-এর snapshot-এর ticker আর সংখ্যা এই notebook-এর থেকে আলাদা; formula আর design একই। এই ক্লাসের companion notebook **`market_data_sentiment_lecture.ipynb`** সম্পূর্ণ self-contained — project folder ছাড়াই চলে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Precision-এর সততা:** Notebook-এ `SEED` বদলে পাঁচবার চালান। NOVX-এর annual vol-এর range কত? Memo-তে "27.25%" লেখা কি সৎ, নাকি "~27%"?
2. **Fat tail মাপুন:** NOVX-এর Student-t `df=3` থেকে `df=30` করুন। VaR₉₉ historical আর normal-এর ফাঁক কী হয়? Kurtosis?
3. **Look-ahead শিকার:** এমন একটা bug লিখুন যেখানে rolling 20-দিনের volatility আজকের return-ও ধরে, অথচ সিদ্ধান্ত হয় দিনের শুরুতে। `.shift(1)` কোথায় বসবে?
4. **Shrinkage:** Net sentiment-এ `S × n / (n + 3)` যোগ করুন। কোন ticker সবচেয়ে বেশি বদলায়? কেন?
5. **Domain shift নিজে দেখুন:** FinBERT আর `cardiffnlp/twitter-roberta-base-sentiment-latest` দুটোই ২০টা নিজের লেখা finance headline-এ চালান (negation, "guidance", "beats", "liability" সহ)। কোথায় দুটো ভিন্ন? কোনটা ঠিক?
6. **Syndication:** একই headline তিনটা wire-এর নামে তিনবার ঢোকান। Net sentiment কী হয়? Normalized-text hash দিয়ে dedupe লিখুন।
7. **Survivorship চিন্তা:** আপনার snapshot-এ শুধু আজকের তিনটা ticker। গত বছর দেউলিয়া হওয়া একটা চতুর্থ company থাকলে "গড় drawdown" কী হতো?

---

## References

* Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063 — https://arxiv.org/abs/1908.10063
* `ProsusAI/finbert` model card — https://huggingface.co/ProsusAI/finbert
* Malo, P., Sinha, A., Korhonen, P., Wallenius, J., Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology* 65(4), 782-796. (Financial PhraseBank)
* `cardiffnlp/twitter-roberta-base-sentiment-latest` — https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest · `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` — https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
* Loughran, T., McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance* 66(1), 35-65.
* Tetlock, P. C. (2007). Giving Content to Investor Sentiment: The Role of Media in the Stock Market. *Journal of Finance* 62(3), 1139-1168.
* Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML — https://arxiv.org/abs/1706.04599
* J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document*, 4th ed.
* Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*, 3rd ed. McGraw-Hill.
* Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). Coherent Measures of Risk. *Mathematical Finance* 9(3), 203-228.
* Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk* — https://www.bis.org/bcbs/publ/d457.htm
* Mandelbrot, B. (1963). The Variation of Certain Speculative Prices. *Journal of Business* 36(4), 394-419.
* Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance* 1(2), 223-236.
* Brown, S. J., Goetzmann, W., Ibbotson, R. G., Ross, S. A. (1992). Survivorship Bias in Performance Studies. *Review of Financial Studies* 5(4), 553-580.
* López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. (backtest আর look-ahead-এর ফাঁদ)
* Hugging Face Transformers pipelines — https://huggingface.co/docs/transformers/main_classes/pipelines · `huggingface_hub` environment variables (`HF_HUB_OFFLINE`) — https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
* Python `sqlite3` — https://docs.python.org/3/library/sqlite3.html · SQLite UPSERT — https://www.sqlite.org/lang_upsert.html · SQLite WAL — https://www.sqlite.org/wal.html
* pandas `Series.std` (`ddof`) — https://pandas.pydata.org/docs/reference/api/pandas.Series.std.html
* yfinance (unofficial, not affiliated with Yahoo) — https://github.com/ranaroussi/yfinance · feedparser — https://feedparser.readthedocs.io/
* ADR 0002 (self-contained), 0008 (static verification), 0009 (FinBERT) — `docs/adr/`
