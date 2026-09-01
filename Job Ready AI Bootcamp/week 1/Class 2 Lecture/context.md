
# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **Week 1 — Class 2: Advanced Python for AI**। এখানে technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

Real-world AI system বানাতে শুধু library function call জানলেই হয় না। Input data কোথা থেকে আসে, algorithm কীভাবে decision নেয়, কোন limitation আছে এবং output কীভাবে validate করতে হয়—এই পুরো flow বোঝা দরকার। এই lecture সেই problem-solving mindset তৈরি করবে।

## শেখার সহজ Workflow

Problem বোঝা → Core concept ও intuition → Step-by-step workflow → Practical example → কখন ব্যবহার করবেন বা করবেন না।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?** এই প্রশ্নগুলোর উত্তর দিতে পারলে topic-টি শুধু মুখস্থ নয়, সত্যি বোঝা হয়েছে।

> **Practical mindset:** Example code run করার আগে expected input এবং output লিখে নিন। Run করার পরে result expectation-এর সাথে compare করুন এবং ভুল হলে কোন pipeline step-এ সমস্যা হয়েছে তা isolate করুন।

---

## Week 1 — Class 2: Advanced Python for AI

## 1. List Comprehensions

### 📌 Topic Overview

List Comprehension হলো পাইথনের একটা শর্টকাট লেখার নিয়ম। একটা লিস্ট থেকে নতুন একটা লিস্ট বানাতে চাইলে — সেটা মাত্র এক লাইনেই করা যায়।

> সহজ কথায়: আগে আপনাকে একটা খালি লিস্ট বানাতে হতো, তারপর `for` লুপ চালাতে হতো, তারপর প্রতিবার `.append()` করতে হতো। এই ৩-৪ লাইনের কাজটাই List Comprehension দিয়ে এক লাইনে হয়ে যায়।

**তুলনা করে দেখুন:**

```python
# পুরোনো নিয়ম (৩ লাইন)
squares = []
for n in [1, 2, 3, 4]:
    squares.append(n * n)

# List Comprehension (১ লাইন)
squares = [n * n for n in [1, 2, 3, 4]]
# দুটোর ফলাফলই: [1, 4, 9, 16]
```

### Why It Is Related

AI-এর কাজে ডেটা মানেই কোনো না কোনো **লিস্ট** — প্রশ্নের লিস্ট (prompts), শব্দের লিস্ট (tokens), সংখ্যার লিস্ট (embeddings), অথবা সার্ভারের উত্তরের লিস্ট (API responses)।

এই লিস্টগুলোকে বারবার বদলাতে (transform) বা ছেঁকে নিতে (filter) হয়। এই কাজের জন্য List Comprehension-ই সবচেয়ে বেশি ব্যবহৃত টুল — কারণ এটা দ্রুত চলে আর কোড ছোট থাকে।

### How It Works

গঠনটা এমন:

```python
[expression for item in iterable if condition]
```

তিনটা অংশ:

* **`for item in iterable`** — লুপ। কোন জিনিসগুলোর উপর কাজ হবে (`iterable` মানে যেকোনো লিস্ট, স্ট্রিং বা এমন কিছু যার উপর লুপ চালানো যায়)।
* **`expression`** — প্রতিটা আইটেম নিয়ে কী করবেন। এখানেই হিসাব বা পরিবর্তনের কাজটা হয়।
* **`if condition`** (না দিলেও চলে) — ছাঁকনি। শর্ত মিললে তবেই আইটেমটা নতুন লিস্টে ঢুকবে।

পাইথন এই কাজটা ভেতরে ভেতরে C ভাষার গতিতে চালায়। তাই সাধারণ `for` লুপে `.append()` করার চেয়ে এটা বেশি দ্রুত।

---

### Production Examples & Code Snippets

```python
# 1. সহজ পরিবর্তন: সব লেখা ছোট হাতের অক্ষরে আনা
tokens = ["LLaMA", "Ollama", "PyTorch", "FastAPI"]
clean_tokens = [t.lower() for t in tokens]
# Output: ['llama', 'ollama', 'pytorch', 'fastapi']

# 2. ছাঁকনি: শুধু ভালো স্কোরগুলো রাখা
confidence_scores = [0.89, 0.32, 0.95, 0.12, 0.78]
high_confidence = [score for score in confidence_scores if score > 0.5]
# Output: [0.89, 0.95, 0.78]

# 3. লিস্টের ভেতরের লিস্ট সমান করা (২ স্তর থেকে ১ স্তরে)
paragraphs = [["Hello", "World"], ["AI", "Is", "Awesome"]]
flattened_tokens = [token for sentence in paragraphs for token in sentence]
# Output: ['Hello', 'World', 'AI', 'Is', 'Awesome']
```

> 💡 **Core Industry Insight:**
> গতির চেয়ে **কোড পড়তে পারাটা (readability)** বেশি জরুরি। List comprehension সুন্দর দেখায় যতক্ষণ সেটা এক লাইনে শেষ হয়। লাইন যদি ৩-৪ লাইন লম্বা হয়ে যায়, তখন সাধারণ `for` লুপ লেখাই ভালো। মনে রাখবেন — কোড একবার লেখা হয়, কিন্তু পড়া হয় ১০০ বার।

### You Can Think it This way

একটা **হাইলাইটার লাগানো ফটোকপি মেশিন** ভাবুন:

* **সাধারণ `for` লুপ:** আপনি একটা পেজ কপি করলেন, হেঁটে টেবিলে গেলেন, মার্কার দিয়ে দাগ দিলেন, আবার মেশিনে ফিরে পরের পেজ দিলেন। এভাবে ১০০ বার।
* **List Comprehension:** এমন এক স্মার্ট মেশিন, যেটা কপি করার সময়ই নিজে থেকে দাগ দিয়ে দেয় আর শেষে পুরো স্তূপটা একবারে বের করে দেয়। কম খাটুনি, বেশি গতি।

---

## 2. Lambda Functions

### 📌 Topic Overview

Lambda Function হলো নাম ছাড়া ছোট্ট একটা ফাংশন, যেটা এক লাইনেই লেখা যায়। `lambda` কিওয়ার্ড দিয়ে কোডের ভেতরেই সরাসরি বানিয়ে ফেলা যায়।

> সহজ কথায়: সাধারণ ফাংশন বানাতে `def` লিখতে হয়, নাম দিতে হয়, `return` লিখতে হয়। Lambda-তে এসবের দরকার নেই। খুব ছোট আর একবারই দরকার — এমন কাজের জন্য এটা ব্যবহার করা হয়।

**তুলনা করে দেখুন:**

```python
# সাধারণ ফাংশন
def double(x):
    return x * 2

# একই কাজ lambda দিয়ে
double = lambda x: x * 2
```

### Why It Is Related

AI-এর কাজে অনেক ছোট ছোট একবারের কাজ থাকে। যেমন: *"স্কোর অনুযায়ী সাজাও"*, *"খালি উত্তরগুলো বাদ দাও"*, বা *"লেবেলগুলোকে সংখ্যায় বদলাও"*।

এত ছোট কাজের জন্য প্রতিবার আলাদা `def` ফাংশন লিখলে কোড অকারণে বড় ও এলোমেলো হয়ে যায়। `lambda` দিয়ে ঠিক যেখানে দরকার, সেখানেই লজিকটা লিখে ফেলা যায়।

### How It Works

```python
lambda arguments: expression
```

* নাম ছাড়াই একটা ফাংশন তৈরি করে।
* এটাকে ভেরিয়েবলে রাখা যায়, অথবা অন্য ফাংশনের ভেতরে পাঠিয়ে দেওয়া যায়।
* ভেতরে **শুধু একটা এক্সপ্রেশন** লেখা যায় — কোনো লুপ, কোনো ভেরিয়েবল অ্যাসাইন, বা একাধিক লাইন লেখা যাবে না।

---

### Production Examples & Code Snippets

```python
# ডিকশনারির লিস্টকে নির্দিষ্ট key অনুযায়ী সাজানো
model_outputs = [
    {"model": "llama3.2", "latency": 120, "score": 0.88},
    {"model": "gpt-4o", "latency": 450, "score": 0.95},
    {"model": "qwen2.5", "latency": 90, "score": 0.82}
]

# latency অনুযায়ী সাজানো (কম থেকে বেশি)
# lambda x: x['latency'] মানে — প্রতিটা আইটেম থেকে latency-র মানটা নাও, সেটা দিয়ে সাজাও
fastest_models = sorted(model_outputs, key=lambda x: x['latency'])

# score অনুযায়ী সাজানো (বেশি থেকে কম)
best_models = sorted(model_outputs, key=lambda x: x['score'], reverse=True)
```

> 💡 **Core Industry Insight:**
> Lambda একটা কাজের টুল, সব জায়গায় জোর করে বসানোর নিয়ম নয়। লজিক এক লাইনের বেশি বড় হলে নাম দিয়ে ঠিকমতো `def` ফাংশন লিখুন। PEP 8 (পাইথনের কোড লেখার অফিসিয়াল নিয়ম) এবং আপনার টিমের সিনিয়ররা খুশি হবেন। লক্ষ্য কোড ছোট করা নয় — কোড সহজ করা।

### You Can Think it This way

Lambda হলো **একবার ব্যবহারের কফি ফিল্টার**:

* এক কাপ কফি দরকার। ফিল্টার নিলেন, কফি বানালেন, ফিল্টার ফেলে দিলেন।
* এই ফিল্টারের নিশ্চয়ই কোনো নাম দেবেন না ("CoffeeFilterVersion1"), বা ধুয়ে বছরের পর বছর ড্রয়ারে রাখবেন না।
* কিন্তু স্থায়ী, বারবার ব্যবহারের ফিল্টার লাগলে (মানে বড়, কয়েক ধাপের লজিক) — তখন নাম দিয়ে ঠিকমতো `def` ফাংশন বানাবেন।

---

## 3. Handling JSON/YAML for Model Configs

### 📌 Topic Overview

পাইথন দিয়ে JSON আর YAML ফাইল পড়া, লেখা ও বদলানো। আজকের AI সিস্টেমে সেটিংস রাখার জন্য এই দুটো ফরম্যাটই সবচেয়ে বেশি ব্যবহৃত হয়।

### Why It Is Related

আজকের AI সিস্টেম কোড দিয়ে নয়, **সেটিংস ফাইল (config file) দিয়ে চালানো হয়**। মডেলের সেটিংস, RAG পাইপলাইনের সার্চ সেটিংস, API-এর রিট্রাই নিয়ম — সব থাকে config ফাইলে।

এগুলো যদি পাইথন কোডের ভেতরে সরাসরি লিখে রাখা হয়, তাহলে ছোট্ট একটা মান বদলাতেও নতুন গিট কমিট আর কোড রিভিউ লাগবে। আলাদা config ফাইল থাকলে সোর্স কোডে হাত না দিয়েই যে কেউ পরীক্ষা-নিরীক্ষা করতে পারে।

### How It Works

* **JSON (JavaScript Object Notation):** কড়া নিয়মের, মেশিনের জন্য বানানো ফরম্যাট। প্রতিটা key অবশ্যই ডাবল-কোটেশনের (`" "`) ভেতরে থাকতে হবে। কোনো কমেন্ট (`#`) লেখা যায় না। এক সার্ভার থেকে আরেক সার্ভারে ডেটা পাঠানোর জন্য সবচেয়ে ভালো।
* **YAML (YAML Ain't Markup Language):** মানুষের পড়ার জন্য অনেক সহজ ফরম্যাট। এখানে কমেন্ট লেখা যায়, কয়েক লাইনের লেখা রাখা যায়। Docker Compose, Kubernetes, GitHub Actions আর ML পরীক্ষার সেটিংসে এটাই স্ট্যান্ডার্ড।

**পাশাপাশি দেখুন:**

```json
{ "model": "llama3.2", "temperature": 0.7 }
```

```yaml
model: "llama3.2"
temperature: 0.7  # কমেন্ট লেখা যায়, JSON-এ যায় না
```

---

### Production Code: Loading & Validating Configs

```python
import json
import yaml

# 📝 1. YAML config নিরাপদভাবে পড়া (ইন্ডাস্ট্রি স্ট্যান্ডার্ড নিয়ম)
def load_ai_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        # safe_load ব্যবহার করলে ফাইলের ভেতর লুকানো কোড চলতে পারে না
        return yaml.safe_load(file)

# উদাহরণ config.yaml ফাইলের ভেতরে যা থাকবে:
# model: "llama3.2"
# temperature: 0.7  # উত্তর কতটা সৃজনশীল হবে
# max_tokens: 512

# 📝 2. রান করার সময়ের হিসাবগুলো JSON ফাইলে সেভ করা
def save_session_log(metrics: dict, log_path: str):
    with open(log_path, 'w') as file:
        json.dump(metrics, file, indent=4)  # indent=4 দিলে ফাইলটা পড়তে সুবিধা হয়
```

> 🛑 **Security Warning:**
> পাইথনে কখনোই সাধারণ `yaml.load()` ব্যবহার করবেন না। এটা একটা মারাত্মক নিরাপত্তা ঝুঁকি — আক্রমণকারী YAML ফাইলের ভেতরে ক্ষতিকর পাইথন কোড লুকিয়ে রাখতে পারে, আর ফাইলটা পড়ার সময়ই সেই কোড আপনার সার্ভারে চলে যাবে। সবসময় **`yaml.safe_load()`** ব্যবহার করবেন।

### You Can Think it This way

JSON আর YAML-কে **ব্লুপ্রিন্ট বনাম স্টিকি নোট** হিসেবে ভাবুন:

* **JSON** হলো ইঞ্জিনিয়ারের আঁকা নিখুঁত **নকশা (ব্লুপ্রিন্ট)**। একদম নির্দিষ্ট, কোনো হিজিবিজি চলবে না। দরজা ৩৬ ইঞ্চি লেখা থাকলে ৩৬ ইঞ্চিই হতে হবে।
* **YAML** হলো সেই নকশার উপর সাঁটা **স্টিকি নোট**, যেখানে মানুষের ভাষায় লেখা — *"৫ নম্বর রুমের রঙটা একটু হালকা কোরো, ক্লায়েন্ট বলেছে"*। এটা মানুষের বোঝার মতো বাড়তি তথ্য বহন করতে পারে, যেটা JSON পারে না।

---

## 4. Error Handling for API Failures

### 📌 Topic Overview

**Resilient code** (টেকসই কোড) মানে এমনভাবে কোড লেখা, যাতে বাইরের AI API (যেমন OpenAI, Anthropic, বা নিজের কম্পিউটারে চলা Ollama) কল করার সময় নেটওয়ার্ক কেটে গেলে, সার্ভার দেরি করলে, বা রিকোয়েস্টের সীমা পার হয়ে গেলেও আপনার অ্যাপ বন্ধ হয়ে না যায়।

> সহজ কথায়: API ব্যবহার করার সময় নেটওয়ার্ক কেটে যাওয়া, সার্ভার ডাউন থাকা বা রিকোয়েস্টের চাপ বেশি হওয়া — এসব খুবই স্বাভাবিক ঘটনা। ভালো কোড এই সমস্যাগুলো নিজে থেকেই সামলাতে পারে এবং সাময়িক সমস্যায় আবার চেষ্টা করে ঠিক হয়ে যায়।

**দুটো পদ্ধতি পাশাপাশি:**

* **দুর্বল পদ্ধতি (নতুনরা যেভাবে লেখে):**

```python
# কোনো সুরক্ষা ছাড়াই সরাসরি API কল
# নেটওয়ার্ক এক সেকেন্ডের জন্য কেটে গেলেই পুরো অ্যাপ বন্ধ হয়ে যাবে!
response = requests.post("https://api.openai.com/v1/chat/completions", json=payload)
data = response.json()
```

* **টেকসই পদ্ধতি (প্রোডাকশনে যেভাবে লেখা হয়):**

```python
# try/except আর আবার-চেষ্টা (retry) দিয়ে কোড সুরক্ষিত করা
import time

for attempt in range(3):
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, timeout=10)
        response.raise_for_status()  # সার্ভার এরর পাঠালে এখানেই ধরা পড়বে
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
        time.sleep(2)  # ২ সেকেন্ড অপেক্ষা করে আবার চেষ্টা করবে
```

এখানে API ফেইল করলেই কোড হাল ছেড়ে দেয় না। `try/except` দিয়ে সমস্যাটা ধরে ফেলে, তারপর আবার চেষ্টা করে। প্রোডাকশনে ৯৯.৯% আপটাইম রাখতে হলে এভাবেই কোড লিখতে হয়।

### Why It Is Related

আপনার AI অ্যাপ ঠিক ততটাই মজবুত, যতটা এর সবচেয়ে দুর্বল নেটওয়ার্ক কলটা।

OpenAI, Anthropic বা আপনার নিজের মডেল সার্ভার — যেকোনো সময় ডাউন হতে পারে। Rate limit (429), Gateway Timeout (504), Connection Reset — এগুলো ব্যতিক্রম নয়, এগুলো প্রোডাকশনের **স্বাভাবিক অবস্থা**। প্রথম এররেই যদি কোড বন্ধ হয়ে যায়, তাহলে সেই কোড প্রোডাকশনের জন্য তৈরি নয়।

### How It Works & Resiliency Strategies

**১. দুই ধরনের এরর — সাময়িক বনাম স্থায়ী:**

* *সাময়িক এরর (নিজে থেকেই ঠিক হয়ে যায়):* 500 (সার্ভারের ভেতরের সমস্যা), 502/503/504 (গেটওয়ে), Connection Timeout, 429 (রিকোয়েস্টের সীমা পার)। এগুলোতে **আবার চেষ্টা করতে হবে (Retry)**।
* *স্থায়ী এরর (কোড বা ডেটা না বদলালে ঠিক হবে না):* 400 (ভুল রিকোয়েস্ট), 401 (লগইন নেই), 403 (অনুমতি নেই), 404 (খুঁজে পাওয়া যায়নি)। এগুলোতে বারবার চেষ্টা করা অর্থহীন — সাথে সাথে **থেমে যেতে হবে (Fail Fast)**।

**২. Exponential Backoff (ধাপে ধাপে অপেক্ষা বাড়ানো):** প্রতিবার চেষ্টার মাঝে অপেক্ষার সময় দ্বিগুণ করা (১ সেকেন্ড, তারপর ২, তারপর ৪, তারপর ৮)। এতে অলরেডি চাপে থাকা সার্ভারের উপর আপনি বারবার হাতুড়ির বাড়ি মারতে থাকেন না — তাকে সামলে ওঠার সময় দেন।

---

### Production Code: Resilient API Call with Retries & Backoff

```python
import time
import requests
from requests.exceptions import RequestException

def call_llm_api_resilient(url: str, payload: dict, max_retries: int = 3) -> dict:
    backoff_time = 1  # প্রথমে ১ সেকেন্ড অপেক্ষা

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=10)  # ১০ সেকেন্ডের বেশি অপেক্ষা নয়

            # সার্ভার 4xx বা 5xx পাঠালে এখানে এরর উঠবে
            response.raise_for_status()
            return response.json()  # সফল! কাজ শেষ

        except requests.exceptions.HTTPError as http_err:
            status_code = response.status_code
            # স্থায়ী এরর হলে বারবার চেষ্টা করে লাভ নেই
            if status_code in [400, 401, 403, 404]:
                print(f"Permanent Error: {status_code}. Aborting.")
                raise http_err

            print(f"Transient HTTP Error {status_code}. Retrying in {backoff_time}s...")

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
            print(f"Network glitch caught: {net_err}. Retrying in {backoff_time}s...")

        # অপেক্ষার সময় বাড়ানোর লজিক
        time.sleep(backoff_time)
        backoff_time *= 2  # প্রতিবার অপেক্ষা দ্বিগুণ

    raise Exception("Max retries exceeded. API call failed.")
```

> 💡 **Core Industry Insight:**
> স্টুডেন্ট প্রজেক্টে আমরা এরর হ্যান্ডলিং বাদ দিই, কারণ "API তো কাজ করেই!" কিন্তু প্রোডাকশনে "কাজ করেই" মানে হলো "যেকোনো মুহূর্তে বন্ধ হয়ে যাবে"। জুনিয়র আর সিনিয়র ইঞ্জিনিয়ারের আসল তফাত কোড সফলভাবে চালানোয় নয় — **কোড ফেইল করলে সিস্টেম কীভাবে সামলে নেয়**, তাতে।

### You Can Think it This way

API এরর হ্যান্ডলিংকে **খারাপ আবহাওয়ায় গাড়ি চালানোর** সাথে মেলান:

* **কোনো এরর হ্যান্ডলিং নেই:** সিটবেল্ট ছাড়া বরফের রাস্তায় ঘণ্টায় ৭০ মাইলে গাড়ি চালানো। রাস্তা সোজা থাকলে ঠিক আছে, কিন্তু চাকা একটু পিছলালেই শেষ (কোড ক্র্যাশ)।
* **শুধু Try/Except:** এটা সিটবেল্টের মতো। আপনি ছিটকে বাইরে পড়বেন না, কিন্তু গাড়ি অ্যাক্সিডেন্ট করবেই (ইউজার এরর স্ক্রিন দেখবে)।
* **Retry + Exponential Backoff:** এটা গাড়ির **ABS ব্রেক আর ট্র্যাকশন কন্ট্রোল**। গাড়ি নিজেই বুঝতে পারে চাকা পিছলাচ্ছে, তাই নিজে থেকে ব্রেক পাম্প করে ধীরে ধীরে রাস্তার উপর দখল ফিরিয়ে আনে।

---
