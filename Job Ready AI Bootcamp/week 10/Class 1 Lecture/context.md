# সহজ ভাষায় Lecture Overview

এই lecture-এর বিষয় **NLP Basics & Transformers: Tokenization, Word Embeddings, Attention**। Technical term English-এ, ব্যাখ্যা Bangla-তে। সঙ্গে থাকা notebook-এ আগে ছোট, offline হিসাব; তারপর চাইলে pretrained model দিয়ে বাস্তব উদাহরণ।

## কেন এই Topic দরকার?

একই দক্ষতা CV-তে **"ML"** আর job description-এ **"machine learning"** লেখা থাকতে পারে। শুধু শব্দ মেলালে প্রাসঙ্গিক CV বাদ পড়বে। টেক্সট কীভাবে token, vector, contextual representation, তারপর similarity score হয়—এই ধাপগুলো বুঝলে ফলাফল ব্যাখ্যা ও যাচাই করা যায়।

## শেখার সহজ Workflow

Job description ও CV → tokenization → embedding → attention/context → sentence embedding → cosine similarity → ranking।

**ক্লাস শেষে পারবেন:** (১) ছোট corpus-এ BPE merge দেখাতে, (২) static ও contextual embedding-এর পার্থক্য বলতে, (৩) attention weight হিসাবের অর্থ বুঝতে, (৪) cosine score দিয়ে CV rank করতে এবং score-এর সীমা ব্যাখ্যা করতে।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?** এই প্রশ্নগুলোর উত্তর দিতে পারলে topic-টি শুধু মুখস্থ নয়, সত্যি বোঝা হয়েছে।

> **Practical mindset:** Example code run করার আগে expected input এবং output লিখে নিন। Run করার পরে result expectation-এর সাথে compare করুন এবং ভুল হলে কোন pipeline step-এ সমস্যা হয়েছে তা isolate করুন।

---

## 1. Topic: NLP Basics & Transformers — Tokenization, Word Embeddings, Attention

কম্পিউটার সংখ্যা বোঝে, শব্দ বোঝে না। **Natural Language Processing (NLP)**-এর প্রথম কাজই হলো টেক্সটকে সংখ্যায় রূপান্তর করা, এমনভাবে যেন সেই সংখ্যাগুলো শব্দের **অর্থ** ধরে রাখে। এই ক্লাসে আমরা তিনটা ধাপে এগোব, যেগুলো আসলে একে অপরের ওপর বিল্ড হয়েছে:

* **Tokenization (BPE)** — টেক্সটকে ছোট ছোট টুকরায় (token) ভাগ করা।
* **Word Embeddings (Word2Vec/GloVe)** — প্রতিটা শব্দকে একটা সংখ্যার ভেক্টরে রূপান্তর করা, যেখানে কাছাকাছি অর্থের শব্দ কাছাকাছি অবস্থানে থাকে।
* **Transformer Attention** — একটা বাক্যের প্রতিটা শব্দ অন্য শব্দগুলোর সাথে কতটা প্রাসঙ্গিক (relevant), তা হিসাব করা।

**Project Goal:** একটা **Semantic CV-to-Job Matcher** বানানো — যেটা কোনো জব ডেসক্রিপশনের সাথে একগুচ্ছ সিভি (CV) কতটা "মিলে যায়" তা বের করবে, শুধু কমন শব্দ গুনে নয়, বরং **অর্থ** বুঝে।

---

## 2. Why It Is Related

Week 2-এ আমরা শিখেছিলাম **Dot Product** এবং **Cosine Similarity** দিয়ে দুটো ভেক্টর কতটা "একই দিকে" নির্দেশ করে তা মাপা যায় (`week 2/Class 1 Lecture`-এর NumPy লেকচার মনে করুন)। NLP-তে সেই একই গণিত ব্যবহার হয় — শুধু ভেক্টরগুলো এখন সংখ্যার বদলে **শব্দ বা বাক্যের অর্থ** ধারণ করে।

* **Keyword Search** (traditional): "machine learning" শব্দটা CV-তে literally আছে কিনা চেক করে। "ML" বা "deep learning" লেখা থাকলে miss করে যাবে, যদিও অর্থ একই।
* **Semantic Search** (এই ক্লাসের টার্গেট): "machine learning", "ML", "deep learning" — এই তিনটাকেই কাছাকাছি ভেক্টর হিসেবে চেনে, কারণ ভেক্টরগুলো শব্দের **অর্থ** এনকোড করে, বানান নয়।

আজকের প্রায় সব বড় এআই সিস্টেম — ChatGPT, সার্চ ইঞ্জিন, রেকোমেন্ডেশন সিস্টেম — এই একই ভিত্তির ওপর দাঁড়িয়ে: **টেক্সটকে অর্থ-সংরক্ষণকারী ভেক্টরে রূপান্তর করা, তারপর ভেক্টরের মধ্যে সাদৃশ্য মাপা।**

---

## 3. How It Works

### 3.1 Tokenization — টেক্সটকে টুকরা করা

মডেল সরাসরি বাক্য পড়তে পারে না — আগে এটাকে ছোট ছোট **token**-এ ভাঙতে হয়। সবচেয়ে সহজ উপায় হলো শব্দ ধরে ধরে ভাঙা, কিন্তু এতে সমস্যা আছে: "running", "runner", "runs" — প্রতিটাকে আলাদা টোকেন ধরলে ভোকাবুলারি বিশাল হয়ে যায়, আবার নতুন শব্দ (out-of-vocabulary) এলে মডেল পুরোপুরি আটকে যায়।

**Byte-Pair Encoding (BPE)** এই সমস্যার সমাধান করে সাব-ওয়ার্ড (sub-word) লেভেলে ভেঙে:

```
"unhappiness" ──▶ ["un", "happi", "ness"]  (illustration; exact split vocabulary-নির্ভর)
```

BPE ট্রেনিংয়ে সবচেয়ে ঘনঘন পাশাপাশি আসা token pair ধাপে ধাপে merge হয়। ফলে সাধারণ অংশগুলো বড় token হতে পারে, বিরল শব্দ ছোট অংশে থাকে। **BERT-এর WordPiece** একই subword লক্ষ্য পূরণ করে, কিন্তু তার vocabulary শেখার নিয়ম BPE-এর মতো নয়। Notebook-এ আগে BPE-এর ছোট উদাহরণ, পরে ঐচ্ছিক BERT tokenizer দেখা যাবে।

### 3.2 Word Embeddings — অর্থকে সংখ্যায় ধরা

একবার টোকেন হয়ে গেলে, প্রতিটা টোকেনকে একটা **fixed-size ভেক্টর** (যেমন ৫০ বা ৩০০ ডাইমেনশনের সংখ্যার লিস্ট) দিয়ে রিপ্রেজেন্ট করা হয় — একে বলে **Embedding**। এই ভেক্টরগুলো এমনভাবে ট্রেইন করা হয় যেন কাছাকাছি অর্থের শব্দ কাছাকাছি ভেক্টরে বসে:

```
vector("king")  - vector("man") + vector("woman")  ≈  vector("queen")
```

* **Word2Vec** — একটা শব্দ তার আশেপাশের শব্দ (context) দেখে প্রেডিক্ট করার চেষ্টা করে ট্রেইন হয়; এই প্রক্রিয়ায় ভেক্টরগুলো নিজে থেকেই অর্থ ধারণ করতে শেখে।
* **GloVe** — পুরো কর্পাসে কোন শব্দ কোন শব্দের কাছাকাছি কতবার এসেছে (co-occurrence statistics) তার ওপর ভিত্তি করে ভেক্টর তৈরি করে।

**সীমাবদ্ধতা:** Word2Vec/GloVe-তে প্রতিটা শব্দের **একটাই** ভেক্টর থাকে; "bank" (নদীর তীর) আর "bank" (আর্থিক প্রতিষ্ঠান) একই vector পায়। Contextual model-এ বাক্য অনুসারে representation বদলায়।

### 3.3 Transformer Attention — প্রসঙ্গ বোঝা

Transformer একটা বাক্যের প্রতিটা শব্দের ভেক্টরকে **বাক্যের বাকি শব্দগুলোর প্রেক্ষাপটে** নতুন করে হিসাব করে — একে বলে **Self-Attention**। "bank" শব্দটা "river"-এর কাছে থাকলে একরকম ভেক্টর পাবে, "money"-এর কাছে থাকলে সম্পূর্ণ ভিন্ন ভেক্টর পাবে।

```
Query (আমি কী খুঁজছি) · Key (প্রতিটা শব্দ কী অফার করছে) ──▶ Attention Score
Attention Score → Softmax → Weighted sum of Value vectors
```

এর সূত্র $\operatorname{softmax}(QK^T/\sqrt{d_k})V$। $QK^T$ token pair-এর learned compatibility score; $\sqrt{d_k}$ দিয়ে scale করলে softmax স্থিতিশীল থাকে। Softmax-এ প্রতিটি query-এর weight-গুলোর যোগফল ১; সেই weight দিয়ে $V$-এর weighted sum নেওয়া হয়। **Attention weight একা কোনো মডেলের সিদ্ধান্তের সম্পূর্ণ ব্যাখ্যা নয়।** Notebook-এ ছোট NumPy উদাহরণে ধাপগুলো দেখা যাবে।

### 3.4 Sentence Embeddings — পুরো বাক্যকে একটা ভেক্টরে ধরা

আমাদের প্রজেক্টের জন্য দরকার একটা শব্দ না, বরং পুরো CV বা জব ডেসক্রিপশনের একটা ভেক্টর। **Sentence-Transformers** মডেলগুলো (BERT-এর ওপর ভিত্তি করে তৈরি) একটা পুরো প্যারাগ্রাফ নিয়ে একটামাত্র ফিক্সড-সাইজ ভেক্টর আউটপুট দেয়, যেটাতে পুরো টেক্সটের অর্থ সংকুচিত হয়ে থাকে। দুটো টেক্সটের ভেক্টরের মধ্যে **Cosine Similarity** বের করলেই বোঝা যায় তারা অর্থগতভাবে কতটা কাছাকাছি।

---

## 4. Details & Valid Points

### 4.1 Model Reference Table — কোন এমবেডিং মডেল কখন ব্যবহার করবেন

| Model | Use Case | Advantage | Limitation |
| --- | --- | --- | --- |
| **`paraphrase-MiniLM-L3-v2`** (আমাদের প্রজেক্টে ব্যবহৃত) | দ্রুত, লোকাল সেমান্টিক সিমিলারিটি | ছোট (~৬১MB), CPU-তেই ফাস্ট | বড় মডেলের তুলনায় সামান্য কম নির্ভুল |
| `all-MiniLM-L6-v2` | জেনারেল-পারপাস সেন্টেন্স এমবেডিং | ভালো অ্যাকুরেসি, তবু ছোট | L3 ভ্যারিয়েন্টের চেয়ে ~৫০% বড়, একটু ধীর |
| `all-mpnet-base-v2` | হাই-অ্যাকুরেসি সেমান্টিক সার্চ | এই ফ্যামিলিতে সবচেয়ে নির্ভুল | অনেক ভারী (~৪২০MB), CPU-তে ধীর |
| OpenAI `text-embedding-3-small` | ক্লাউড-বেসড এমবেডিং | লোকাল কম্পিউট লাগে না | লোকাল না — প্রতি কলে খরচ, ডেটা থার্ড-পার্টিতে যায় |

> **Valid Point:** ছোট মডেল মানেই "খারাপ" না — একটা পোর্টফোলিও-স্কেল CV ম্যাচারে `paraphrase-MiniLM-L3-v2`-এর অ্যাকুরেসি যথেষ্ট, আর CPU-তেই রিয়েল-টাইম রেসপন্স পাওয়া যায়। প্রোডাকশনে ইউজার বেশি এবং অ্যাকুরেসি critical হলে `all-mpnet-base-v2`-এর দিকে যাওয়া যুক্তিসঙ্গত।

### 4.2 "Local Model" মানে কী — Sovereign AI-এর সাথে সামঞ্জস্যপূর্ণ কীভাবে?

`paraphrase-MiniLM-L3-v2` প্রথমবার HuggingFace Hub থেকে ডাউনলোড করতে ইন্টারনেট লাগে। প্রয়োজনীয় model files cache-এ থাকলে inference লোকালি করা যায়; cache মুছে গেলে বা নতুন environment-এ আবার download লাগবে। CV-এর ব্যক্তিগত তথ্য নিয়ে কাজের আগে কোথায় inference হচ্ছে, ডেটা কোথায় সংরক্ষিত হচ্ছে এবং কাদের access আছে—এসবও যাচাই করতে হবে।

### 4.3 Why Dot Product, Not Euclidean Distance, for Text Similarity?

Cosine Similarity $\frac{A\cdot B}{\|A\|\|B\|}$ ভেক্টরের **দিক** মাপে; ভেক্টরের magnitude বাড়লেও স্কোর বদলায় না। কিন্তু text length সরাসরি vector magnitude নয়, আর বড় CV-তে অনেক অপ্রাসঙ্গিক বিষয় থাকলে অর্থ সংকুচিত করার সময় দরকারি তথ্য হারাতে পারে। **স্কোর হলো model-এর similarity estimate, যোগ্যতা বা নিয়োগের সিদ্ধান্ত নয়।** Zero vector-এর cosine undefined; বাস্তব pipeline-এ এই edge case সামলাতে হয়।

---

## 5. Related Analogy: The Restaurant Recommendation System

> কল্পনা করুন একটা রেস্টুরেন্ট রেকোমেন্ডেশন অ্যাপ, যা আপনার পছন্দের সাথে মিলিয়ে রেস্টুরেন্ট সাজেস্ট করে।

| NLP Concept | Restaurant Analogy |
| --- | --- |
| **Tokenization (BPE)** | মেনুর প্রতিটা ডিশকে উপকরণে ভেঙে ফেলা ("চিকেন বিরিয়ানি" → চিকেন + বিরিয়ানি + মসলা), যাতে নতুন ডিশও চেনা উপকরণ দিয়ে বোঝা যায়। |
| **Word Embedding (একটা শব্দ = একটা ভেক্টর)** | প্রতিটা উপকরণের একটা "ফ্লেভার প্রোফাইল" (ঝাল, মিষ্টি, টক — সংখ্যায়) থাকা, যাতে "মরিচ" আর "কাঁচামরিচ"-এর প্রোফাইল কাছাকাছি হয়। |
| **Transformer Attention** | একটা ডিশের স্বাদ বোঝার সময় শুধু একটা উপকরণ না, বরং কোন উপকরণ কোন উপকরণের সাথে মিলে রান্না হয়েছে, সেই পুরো কম্বিনেশন বিবেচনা করা। |
| **Sentence Embedding** | পুরো একটা ডিশের (শুধু একটা উপকরণ না) সামগ্রিক "স্বাদ-প্রোফাইল" একটা সিঙ্গেল কার্ডে সামারাইজ করা। |
| **Cosine Similarity** | দুটো ডিশের স্বাদ-প্রোফাইল কার্ড পাশাপাশি রেখে মেলানো — কতটা মিল আছে তার একটা স্কোর বের করা, ঠিক দুইটা রেসিপির স্বাদ কতটা কাছাকাছি তা বলার মতো। |

**এই অ্যানালজিটা কেন কাজ করে:** ঠিক যেমন দুটো ডিশ ভিন্ন নাম হলেও একই রকম স্বাদের হতে পারে (উপকরণের কম্বিনেশন কাছাকাছি হলে), একটা CV আর একটা জব ডেসক্রিপশন ভিন্ন শব্দ ব্যবহার করেও একই অর্থ বহন করতে পারে — এমবেডিং আর কসাইন সিমিলারিটি ঠিক এই "গভীর মিল" ধরার জন্যই তৈরি।

---

## Achievement: Semantic CV-to-Job Matcher (Preview)

সম্পূর্ণ অ্যাপ **`Class 1 Project/`** ফোল্ডারে আছে — job description এবং একাধিক CV নিয়ে `paraphrase-MiniLM-L3-v2` দিয়ে embedding তৈরি করে cosine similarity অনুযায়ী rank করে। কম্প্যানিয়ন notebook-এ offline BPE, toy vector, attention ও cosine demo চালানো যায়। BERT tokenizer, GloVe এবং pretrained sentence embedding demo চালাতে `RUN_PRETRAINED = True`, অতিরিক্ত dependency ও প্রথমবার internet লাগবে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **BPE Merge Order**: BPE ট্রেইন করার সময় সবচেয়ে ফ্রিকোয়েন্ট ক্যারেক্টার-পেয়ার আগে মার্জ হয় — একটা খুব ছোট কর্পাস (যেমন ৫টা বাক্য) দিয়ে ম্যানুয়ালি ২-৩টা মার্জ স্টেপ ট্রেস করে দেখুন প্রথম কোন পেয়ারগুলো মার্জ হবে।
2. **Word2Vec Analogy Failure**: `king - man + woman ≈ queen` মতো এনালজি সবসময় কাজ করে না। এমন একটা এনালজি ভাবুন যেটা GloVe-তে ফেল করতে পারে (যেমন সাংস্কৃতিকভাবে নির্দিষ্ট কোনো সম্পর্ক), এবং কেন সেটা ফেল করতে পারে ব্যাখ্যা করুন।
3. **Attention vs. GloVe**: "The bank raised interest rates" আর "We sat by the river bank" — এই দুই বাক্যে "bank" শব্দটার এমবেডিং GloVe-তে কী হবে, আর Transformer-এ কী হবে? পার্থক্যটা ঠিক কোথায় তৈরি হয়?
