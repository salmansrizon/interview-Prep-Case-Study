
## Week 1 — Class 2: Advanced Python for AI

## 1. List Comprehensions

### 📌 Topic Overview

List Comprehension হলো একটি চমৎকার Pythonic syntax, যার মাধ্যমে আপনি মাত্র একটি একক এবং সহজে পড়া যায় এমন এক্সপ্রেশনের (single, readable expression) সাহায্যে যেকোনো iterable-কে ট্রান্সফর্ম এবং ফিল্টার করে সম্পূর্ণ নতুন একটি লিস্ট তৈরি করে ফেলতে পারেন।
> সহজ কথায়, ট্র্যাডিশনাল for লুপ লিখে, একটা খালি লিস্ট বানিয়ে, সেখানে বারবার .append() করার যে দীর্ঘ প্রসেস—সেটিকে এক লাইনে নিয়ে আসার নামই হলো List Comprehension।

### Why It Is Related

AI Engineering-এ ডেটা মানেই হলো কোনো না কোনো কিছুর লিস্ট: **A list of prompts, a list of tokens, a list of embeddings, কিংবা a list of API responses**। আপনাকে অনবরত এই লিস্টগুলো ট্রান্সফর্ম, ফিল্টার বা নরমালাইজ করতে হবে। List comprehensions হলো এর স্ট্যান্ডার্ড টুল, কারণ এটি সুপার-ফাস্ট, কনসাইজ এবং গ্লোবাল AI প্র্যাকটিশনারদের কাছে বহুল ব্যবহৃত।

### How It Works

একটি standard list comprehension-এর স্ট্রাকচার দেখতে এমন হয়:

```python
[expression for item in iterable if condition]

```

* **`for item in iterable`**: লুপটি ড্রাইভ করে।
* **`expression`**: প্রতিটা আইটেমের আউটপুট কেমন হবে (এখানে ক্যালকুলেশন বা ট্রান্সফর্মেশন করা যায়)।
* **`if condition` (Optional)**: একটি ফিল্টার; শুধুমাত্র কন্ডিশন ম্যাচ করা আইটেমগুলোই নতুন লিস্টে জায়গা পাবে।

Python এই অপারেশনটিকে ইন্টারনালি **C-speed**-এ রান করায়, যার কারণে এটি সাধারণ `for` লুপে `.append()` করার চেয়ে অনেক বেশি ফাস্ট।

---

### Production Examples & Code Snippets

```python
# 1. Basic Transformation: Token lowercasing
tokens = ["LLaMA", "Ollama", "PyTorch", "FastAPI"]
clean_tokens = [t.lower() for t in tokens]
# Output: ['llama', 'ollama', 'pytorch', 'fastapi']

# 2. Filtering Logits/Scores: Keeping high confidence predictions
confidence_scores = [0.89, 0.32, 0.95, 0.12, 0.78]
high_confidence = [score for score in confidence_scores if score > 0.5]
# Output: [0.89, 0.95, 0.78]

# 3. Flattening a 2D List (Matrix to 1D): Common in text processing
paragraphs = [["Hello", "World"], ["AI", "Is", "Awesome"]]
flattened_tokens = [token for sentence in paragraphs for token in sentence]
# Output: ['Hello', 'World', 'AI', 'Is', 'Awesome']

```

> 💡 **Core Industry Insight (Valid Point):**
> মাইক্রো-অপ্টিমাইজেশনের চেয়ে **Readability (পঠনযোগ্যতা)** অনেক বেশি গুরুত্বপূর্ণ। একটি list comprehension তখনই সুন্দর দেখায় যখন এটি এক লাইনে শেষ হয়। যদি কোড ৩-৪ লাইন বড় হয়ে যায়, তবে প্রথাগত `for` লুপ বা আলাদা হেল্পার ফাংশন ব্যবহার করাই শ্রেয়। কোড একবার লেখা হয়, কিন্তু পড়া হয় ১০০ বার।

### You Can Think it This way 

জিনিসটিকে একটি **Built-in Highlighter যুক্ত ফটোকপি মেশিন** হিসেবে চিন্তা করুন:

* একটি ট্র্যাডিশনাল `for` লুপ হলো: আপনি একটা করে পেজ কপি করছেন, হেঁটে টেবিলে যাচ্ছেন, মার্কার দিয়ে হাইলাইট করছেন, আবার মেশিনে এসে পরের পেজ দিচ্ছেন। এভাবে ১০০ বার করছেন।
* আর **List Comprehension** হলো এমন এক স্মার্ট ফটোকপি মেশিন যা কপি করার সময়ই *অটোমেটিক্যালি হাইলাইট করে* একবারে ফিনিশড পেজের স্ট্যাক বের করে দিচ্ছে। কম খাটনি, ফাস্ট কাজ!

---

## 2. Lambda Functions

### 📌 Topic Overview

Lambda Functions হলো এমন এক ধরণের anonymous (নামহীন) এবং single-expression (এক লাইনে সীমাবদ্ধ) ফাংশন, যা পাইথনের lambda কিওয়ার্ড ব্যবহার করে কোডের ভেতরেই সরাসরি (inline) ডিফাইন বা তৈরি করা যায়।
> সহজ কথায়, সাধারণ ফাংশন তৈরি করার জন্য আমাদের যেভাবে def কিওয়ার্ড ব্যবহার করে, ফাংশনের একটা নাম দিয়ে, তারপর return লিখতে হয়—ল্যাম্বডাতে সেই ঝামেলা নেই। এটি মূলত তাৎক্ষণিক বা ওয়ান-টাইম ব্যবহারের জন্য কোনো নাম ছাড়াই ছোটখাটো লজিক লিখে ফেলার একটি স্মার্ট উপায়।

### Why It Is Related

AI পাইপলাইনগুলো ওয়ান-অফ (একবার ব্যবহার্য) ট্রান্সফর্মেশনে ভরপুর থাকে। যেমন: *"Sort by confidence score," "filter out empty responses,"* অথবা *"map labels to integers."* এই ছোট ছোট কাজের জন্য প্রতিবার একটা করে ফুল `def` ফাংশন ডিক্লেয়ার করলে কোড নোংরা ও জট পাকিয়ে যায়। `lambda` আপনাকে ঠিক সেই জায়গাতেই লজিক লেখার সুবিধা দেয় যেখানে এটি ব্যবহার হচ্ছে।

### How It Works

```python
lambda arguments: expression

```

* কোনো নাম ছাড়াই একটি ফাংশন অবজেক্ট তৈরি করে।
* এটিকে ভেরিয়েবলে অ্যাসাইন করা যায় অথবা অন্য কোনো ফাংশনের আর্গুমেন্ট হিসেবে পাস করা যায়।
* এটি **Strictly restricted to a single expression** (এর ভেতর কোনো লুপ, অ্যাসাইনমেন্ট বা মাল্টিপল স্টেটমেন্ট লেখা যায় না)।

---

### Production Examples & Code Snippets

```python
# 1. Sorting a list of dictionaries by a specific key (Ubiquitous in LLM routing)
model_outputs = [
    {"model": "llama3.2", "latency": 120, "score": 0.88},
    {"model": "gpt-4o", "latency": 450, "score": 0.95},
    {"model": "qwen2.5", "latency": 90, "score": 0.82}
]

# Sort by latency (Lowest to Highest)
fastest_models = sorted(model_outputs, key=lambda x: x['latency'])

# Sort by score (Highest to Lowest)
best_models = sorted(model_outputs, key=lambda x: x['score'], reverse=True)

```

> 💡 **Core Industry Insight (Valid Point):**
> ল্যাম্বডা একটি কাজের টুল, কোনো ধর্ম বা নিয়ম নয় যে সব জায়গায় জোর করে বসাতে হবে। লজিক যদি এক লাইনের চেয়ে বেশি বড় হয়, তবে দয়া করে একটি নামসহ প্রপার `def` ফাংশন লিখুন। PEP 8 স্ট্যান্ডার্ড এবং আপনার টিমের সিনিয়রেরা আপনাকে ধন্যবাদ জানাবে। আমাদের লক্ষ্য কোড ছোট করা নয়, কোড সহজ করা।

### You Can Think it This way 

Lambda একটি **ডিসপোজেবল ওয়ান-টাইম কফি ফিল্টার (Disposable Coffee Filter)** হিসেবে চিন্তা করুন:

* আপনার এখন এক কাপ কফি দরকার। আপনি ফিল্টারটি নিলেন, কফি বানালেন, এবং ফিল্টারটি ডাস্টবিনে ফেলে দিলেন।
* আপনি নিশ্চয়ই এই ফিল্টারটার একটা নাম দেবেন না ("CoffeeFilterVersion1"), বা এটাকে ধুয়ে বছরের পর বছর ড্রয়ারে যত্ন করে রাখবেন না।
* তবে আপনার যদি একটি পার্মানেন্ট, রিইউজেবল গোল্ড ফিল্টার লাগে (কমপ্লেক্স, মাল্টি-স্টেপ লজিক), তখন আপনি একটি প্রপার নামসহ `def` ফাংশন তৈরি করবেন।

---

## 3. Handling JSON/YAML for Model Configs

### 📌 Topic Overview

Using Python to read, write, and manipulate JSON and YAML — modern AI ইনফ্রাস্ট্রাকচারের সবচেয়ে ডমিন্যান্ট দুটি কনফিগারেশন ফরম্যাট।

### Why It Is Related

মডার্ন AI সিস্টেমগুলো কোড দিয়ে নয়, বরং **Config ফাইল দিয়ে ড্রাইভ করা হয়**। একটি মডেলের hyperparameters, একটা RAG pipeline-এর retrieval settings, কিংবা কোনো API-এর retry policy — এই সবকিছু কনফিগ ফাইলে থাকে। এগুলো পাইথন কোডের ভেতর হার্ডকোড করে রাখলে প্রতিটা ছোট টুইকের জন্য নতুন গিট কমিট এবং কোড রিভিউ লাগবে। কনফিগ ফাইল থাকলে ডেটা সায়েন্টিস্ট ও ইঞ্জিনিয়াররা সোর্স কোড না ছুঁয়েই এক্সপেরিমেন্ট করতে পারেন।

### How It Works

* **JSON (JavaScript Object Notation):** একটি স্ট্রিক্ট এবং মেশিন-ফ্রেন্ডলি ফরম্যাট। এর প্রতিটা Key অবশ্যই ডাবল-কোটেশনের (`" "`) ভেতর হতে হবে। এতে কোনো কমেন্ট (`#`) লেখা যায় না। এটি এপিআই পে-লোড এবং মাইক্রোসার্ভিসের মধ্যে যোগাযোগের জন্য বেস্ট।
* **YAML (YAML Ain't Markup Language):** এটি মানুষের পড়ার জন্য অত্যন্ত ফ্রেন্ডলি একটি ফরম্যাট (JSON-এর সুপারসেট)। এটি কমেন্ট, মাল্টি-লাইন স্ট্রিং সাপোর্ট করে। Docker Compose, Kubernetes, GitHub Actions এবং ML experiment কনফিগারেশনে এটি স্ট্যান্ডার্ড।

---

###  Production Code: Loading & Validating Configs

```python
import json
import yaml

# 📝 1. Safe Loading YAML Config (The Industry Standard Pattern)
def load_ai_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        # ALWAYS use safe_load to prevent arbitrary code execution vulnerabilities
        return yaml.safe_load(file)

# Hypothetical config.yaml content:
# model: "llama3.2"
# temperature: 0.7  # Controls creativity
# max_tokens: 512

# 📝 2. Exporting Runtime Metrics to JSON
def save_session_log(metrics: dict, log_path: str):
    with open(log_path, 'w') as file:
        json.dump(metrics, file, indent=4)  # indent=4 makes it human-readable

```

> 🛑 **Security Warning (Valid Point):**
> পাইথনে কখনো সাধারণ `yaml.load()` ব্যবহার করবেন না; এটি একটি অত্যন্ত মারাত্মক সিকিউরিটি ভালনারেবিলিটি (CVE class-vulnerability) তৈরি করে, যার মাধ্যমে হ্যাকাররা YAML ফাইলের ভেতরে ম্যালিশিয়াস পাইথন কোড ঢুকিয়ে আপনার সার্ভার এক্সিকিউট করিয়ে নিতে পারে। সবসময় **`yaml.safe_load()`** ব্যবহার করবেন।

### You Can Think it This way 

JSON এবং YAML-কে **ব্লু-প্রিন্ট বনাম স্টিকি নোট (Blueprints vs. Sticky Notes)** হিসেবে চিন্তা করুন:

* **JSON** হলো সিভিল ইঞ্জিনিয়ারদের দেওয়া নিখুঁত **আর্কিটেকচারাল ব্লুপ্রিন্ট**। এটি একদম সুনির্দিষ্ট, স্ট্যান্ডার্ড এবং এখানে কোনো হিজিবিজি কাটার সুযোগ নেই। দরজা যদি ৩৬ ইঞ্চি বলা থাকে, তবে ৩৬ ইঞ্চিই হতে হবে।
* **YAML** হলো সেই ব্লুপ্রিন্টের ওপর মারা **স্টিকি নোট**, যেখানে মানুষের ভাষায় লেখা থাকে—*"৫ নম্বর রুমের রঙের শেডটা একটু হালকা করিও, ক্লায়েন্ট রিকোয়েস্ট করেছে"*। এটি মানুষের বোঝার জন্য চমৎকার context ক্যারি করে, যা ব্লুপ্রিন্ট ফরম্যাট (JSON) নিজে পারে না।

---

## 4. Error Handling for API Failures

### 📌 Topic Overview

**Resilient Code Writing** হলো এমনভাবে কোড লেখার একটি টেকনিক বা আর্কিটেকচার, যা এক্সটার্নাল AI APIs (যেমন: OpenAI, Anthropic, বা লোকাল Ollama endpoints) কল করার সময় বিভিন্ন ধরণের **network failures, timeouts, rate limits, এবং server errors** হওয়া সত্ত্বেও আপনার অ্যাপ্লিকেশনকে ক্র্যাশ হতে দেয় না, বরং সিস্টেমকে টিকিয়ে রাখে।

> সহজ কথায়, ক্লাউড বা লোকাল এপিআই ব্যবহার করার সময় নেটওয়ার্ক ড্রপ করা, সার্ভার ডাউন থাকা কিংবা রিকোয়েস্টের চাপ বেশি হওয়া (Rate limit) খুবই স্বাভাবিক ঘটনা। প্রোডাকশন-গ্রেড এআই অ্যাপ্লিকেশনের কোড এমন হতে হবে যাতে এই সমস্যাগুলো সে নিজে থেকেই হ্যান্ডেল করতে পারে এবং সাময়িক এরর খেলেও স্মার্টলি রিকভার করতে পারে।

**যেমন:**

* **Fragile Approach (ভঙ্গুর বা জুনিয়র কোডারদের নিয়ম):**
```python
# কোনো এরর হ্যান্ডেলিং ছাড়া সরাসরি API কল
# নেটওয়ার্ক এক সেকেন্ডের জন্য ড্রপ করলেই পুরো অ্যাপ ক্র্যাশ করবে!
response = requests.post("https://api.openai.com/v1/chat/completions", json=payload)
data = response.json()

```


* **Resilient Approach (স্মার্ট ও প্রোডাকশন-রেডি নিয়ম):**

```python
    # try/except ব্লক এবং রিট্রাই মেকানিজম ব্যবহার করে কোডকে সুরক্ষিত করা
    import time
    
    for attempt in range(3):
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, timeout=10)
            response.raise_for_status() # কোনো ৪xx বা ৫xx এরর থাকলে এক্সেপশন রেইজ করবে
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2) # ২ সেকেন্ড পর আবার চেষ্টা করবে
    ```

এখানে কোডটি এপিআই ফেইল করা মাত্রই হাল ছেড়ে দেয় না, বরং `try/except` দিয়ে এররটি ক্যাচ করে এবং ব্যাকগ্রাউন্ডে রিট্রাই লুপের মাধ্যমে পুনরায় চেষ্টা করে। প্রোডাকশনে ৯৯.৯% আপটাইম (Uptime) নিশ্চিত করার জন্য এই ধরণের রেসিলিয়েন্ট কোড রাইটিং প্যাটার্ন অ্যাপ্লাই করা অত্যন্ত জরুরি।

```

### Why It Is Related

আপনার এআই অ্যাপ্লিকেশনটি ঠিক ততটুকুই স্ট্রং, যতটুকু এর সবচেয়ে দুর্বল নেটওয়ার্ক কলটি। OpenAI, Anthropic বা আপনার নিজস্ব মডেল সার্ভার যেকোনো সময় ডাউন হতে পারে বা এরর দিতে পারে। Rate limits (429), Gateway Timeouts (504), Connection Resets এগুলো এক্সেপশন নয়—এগুলো প্রোডাকশনের **স্বাভাবিক অপারেটিং কন্ডিশন**। প্রথম এরর খেয়েই যদি আপনার কোড ক্র্যাশ করে, তবে সেই কোড প্রোডাকশন-গ্রেড নয়।

### How It Works & Resiliency Strategies

1. **Transient vs. Permanent Errors:**
* *Transient Errors (সাময়িক সমস্যা - রিলিজ হলে ঠিক হয়ে যায়):* 500 (Internal Server Error), 502/503/504 (Gateways), Connection Timeouts, 429 (Rate Limits)। এগুলোতে **Retry** করতে হবে।
* *Permanent Errors (স্থায়ী সমস্যা - কোড বা ডাটা চেঞ্জ না করলে ঠিক হবে না):* 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden), 404 (Not Found)। এগুলোতে রিট্রাই করা বোকামি, সরাসরি **Fail Fast** করতে হবে।


2. **Exponential Backoff:** প্রতি রিট্রাই-এর মাঝে ওয়েটিং টাইম ক্রমান্বয়ে বাড়িয়ে দেওয়া (যেমন: 1s $\rightarrow$ 2s $\rightarrow$ 4s $\rightarrow$ 8s)। এতে করে অলরেডি চাপে থাকা সার্ভারের ওপর আপনার ক্লায়েন্ট হাতুড়ির মতো আঘাত করা বন্ধ করে।

---

### Production Code: Resilient API Call with Retries & Backoff

```python
import time
import requests
from requests.exceptions import RequestException

def call_llm_api_resilient(url: str, payload: dict, max_retries: int = 3) -> dict:
    backoff_time = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=10) # 10s timeout budget
            
            # Raises HTTPError if status is 4xx or 5xx
            response.raise_for_status()
            return response.json()  # Success path!
            
        except requests.exceptions.HTTPError as http_err:
            status_code = response.status_code
            # If it's a permanent error, don't waste time retrying
            if status_code in [400, 401, 403, 404]:
                print(f"Permanent Error: {status_code}. Aborting.")
                raise http_err
            
            print(f"Transient HTTP Error {status_code}. Retrying in {backoff_time}s...")
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
            print(f"Network glitch caught: {net_err}. Retrying in {backoff_time}s...")
        
        # Backoff logic
        time.sleep(backoff_time)
        backoff_time *= 2  # Exponentially double the wait time
        
    raise Exception("Max retries exceeded. API call failed.")

```

> 💡 **Core Industry Insight (Valid Point):**
> স্টুডেন্ট লাইফের প্রোজেক্টে আমরা এরর হ্যান্ডেলিং স্কিপ করি কারণ "এপিআই তো কাজ করেই!" কিন্তু প্রোডাকশনে "কাজ করেই" শব্দটির মানে হলো "যেকোনো মুহূর্তে ক্র্যাশ করবে"। একজন জুনিয়র আর একজন সিনিয়র এআই ইঞ্জিনিয়ারের মূল তফাত কোড সাকসেসফুলি রান করায় নয়, বরং **কোড ফেইল করলে সিস্টেম কীভাবে রিকভার করে** তার ওপর নির্ভর করে।

### You Can Think it This way 

এপিআই এরর হ্যান্ডেলিংকে **খারাপ আবহাওয়ায় গাড়ি ড্রাইভ করার (Driving in bad weather)** সাথে তুলনা করুন:

* **No Error Handling:** সিটবেল্ট ছাড়া বরফের রাস্তায় ৭০ মাইল স্পিডে গাড়ি চালানো। রাস্তা সোজা থাকলে আপনি ঠিক আছেন, কিন্তু একটু চাকার স্লিপ মানেই স্পট ডেড (কোড ক্র্যাশ)।
* **Basic Try/Except:** এটি সিটবেল্টের মতো। আপনাকে উইন্ডশিল্ড ভেঙে বাইরে ছিটকে যেতে দেবে না, তবে গাড়ি এক্সিডেন্ট করবেই (ইউজার এরর স্ক্রিন দেখবে)।
* **Retry with Exponential Backoff:** এটি হলো আপনার গাড়ির **Anti-lock Brakes (ABS) + Traction Control**। গাড়ি বুঝতে পারে চাকা স্লিপ কাটছে, তাই সে নিজে থেকেই ব্রেক পাম্প করে ধীরে ধীরে গাড়ির গ্রিপ ফিরিয়ে আনে।

---

## 🎯 Class 2 এ আমরা যা যা শিখলাম 

| Achievement | Knowledge Applied |
| --- | --- |
| **Parse raw data into clean format** | List comprehensions, lambda, dict transformations |
| **Handle JSON/YAML configs** | `json.load`, `yaml.safe_load`, production schema validation |
| **Survive API failures** | `try/except`, retry loops, exponential backoff, transient classification |

---
