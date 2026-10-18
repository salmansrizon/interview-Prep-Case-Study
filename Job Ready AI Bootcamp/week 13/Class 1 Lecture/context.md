# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **Tool Calling & Autonomous Logic — AI-কে "Hands" দেওয়া**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

LLM শুধু text বানাতে পারে। সে নির্ভুল অঙ্ক কষতে পারে না, তোমার ফাইল দেখতে পারে না, আজকের তারিখও জানে না। Tool calling সেই ফাঁক পূরণ করে: model সিদ্ধান্ত নেয় কোন function ডাকতে হবে, আর আমাদের Python code সেটা চালায়। এখান থেকেই agent-এর শুরু।

## শেখার সহজ Workflow

Problem বোঝা → Tool schema → Agent loop → Safety (sandbox) → Model তুলনা।

প্রতিটা section-এ তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? কখন ব্যবহার করবেন না?**

> **Practical mindset:** Run করার আগে ভাবুন model কোন tool ডাকবে আর কী argument দেবে। তারপর trace মিলিয়ে দেখুন।

---

## 1. Topic: Tool Calling & Autonomous Logic

এই ক্লাসে শিখব:

* **Function calling** — model text-এর বদলে একটা structured request পাঠায়: tool-এর নাম আর arguments।
* **Local models** — `llama3.1:8b` (Ollama), তুলনায় `mistral:7b` ও `llama3.2:3b`।
* **Python tools** — `calculate`, `list_files`, `read_file`।
* **Agent loop** — model ডাকে, tool চলে, result ফেরত যায়, উত্তর তৈরি না হওয়া পর্যন্ত চলে।

**Project Goal:** এমন একটা AI যে অঙ্ক কষতে পারে এবং একটা sandbox directory-র ফাইল দেখে প্রশ্নের উত্তর দিতে পারে, পুরোটা local।

---

## 2. Why It Is Related

Week 11-এ Ollama দিয়ে prompting আর Week 12-এ RAG শিখেছি; RAG-এ retrieval ছিল একটা নির্দিষ্ট ধাপ যা আমরা নিজে ডাকতাম। Tool calling-এ সিদ্ধান্তটা model নেয়: আদৌ কিছু ডাকবে কিনা, কোনটা, কী দিয়ে। Retrieval নিজেই একটা tool হতে পারে। পরের ক্লাসগুলোর agent আর multi-agent system এই loop-এর ওপরেই দাঁড়িয়ে আছে।

---

## 3. How It Works

### 3.1 Model আসলে কী করে

Model কখনো function চালায় না। সে শুধু JSON-এর মতো একটা output দেয়:

```json
{"name": "calculate", "arguments": {"expression": "(12+8)*3"}}
```

চালায় আমাদের code। এটাই সবচেয়ে গুরুত্বপূর্ণ নিরাপত্তা-সীমা: model শুধু অনুরোধ করে, সিদ্ধান্ত নিয়ে চালানোর দায়িত্ব আমাদের।

### 3.2 Tool schema হলো prompt

Model function দেখে না, দেখে JSON schema: নাম, `description`, parameters। `description` যত পরিষ্কার, model তত ঠিক জায়গায় tool ডাকে। "Use for ANY arithmetic" লিখলে ছোট model-ও অঙ্ক নিজে না কষে tool ডাকে।

### 3.3 Agent loop

```
messages = [system, user]
repeat max_steps বার:
    reply = model(messages, tools)
    tool_calls না থাকলে -> reply.content-ই উত্তর, থামো
    প্রতিটা call চালাও -> result messages-এ role="tool" দিয়ে যোগ করো
```

দুটো থামার শর্ত থাকা জরুরি: model আর tool না ডাকলে, অথবা `max_steps` শেষ হলে। দ্বিতীয়টা ছাড়া ছোট model একই tool বারবার ডেকে চিরকাল ঘুরতে পারে।

### 3.4 Safe tool লেখা

* **Calculator:** `eval()` কখনো না। Python-এর `ast` দিয়ে শুধু সংখ্যা, operator আর কয়েকটা function allow করা হয়।
* **File tools:** path `resolve()` করে দেখা হয় ফলাফল sandbox-এর ভেতরে কিনা। `../secret.txt`, absolute path, symlink সবই আটকায়। শুধু read, কোনো write নেই।
* **Error হলো string:** tool exception না ছুড়ে `"Error: ..."` ফেরত দেয়, যাতে model পরের turn-এ নিজে শুধরে নিতে পারে।

### 3.5 Model তুলনা

| Model | Use Case | Advantage | Limitation |
| --- | --- | --- | --- |
| `llama3.1:8b` (chosen) | Local tool calling, agents | Native tool support, argument ভালো | ~5GB RAM, CPU-তে ধীর |
| `mistral:7b` | General chat + tool calling | দ্রুত, instruction following ভালো | Multi-step-এ tool call মাঝে মাঝে বাদ পড়ে |
| `llama3.2:3b` | Low-RAM laptop, দ্রুত demo | ~2GB, দ্রুত | জটিল বা একাধিক tool-এ নির্ভরযোগ্যতা কম |
| OpenAI `gpt-4o-mini` | Cloud tool calling | সবচেয়ে নির্ভরযোগ্য | Local না, খরচ লাগে, Sovereign AI ভাঙে |

কিছু model Ollama-তে tool support ঘোষণা করে না; সেগুলোতে `tools=` দিলে error আসে। তাই model বদলানোর আগে `ollama show <model>` দিয়ে capabilities দেখে নিন।

### 3.6 কখন tool calling ব্যবহার করবেন না

নির্দিষ্ট, সবসময় একই ক্রমের কাজ হলে সরাসরি code লিখুন; model-কে সিদ্ধান্ত নিতে দেওয়া বাড়তি খরচ ও ঝুঁকি। যেখানে প্রশ্ন অনির্দিষ্ট আর কোন tool লাগবে আগে থেকে জানা নেই, সেখানেই agent মানায়।

---

## 4. Project: Math & File Inspector

`week 13/Class 1 Project/` দেখুন। Test পুরোটাই fake LLM দিয়ে চলে, Ollama লাগে না। Ollama বন্ধ থাকলে app friendly error দেখায়।

## 5. Brain Teaser

১. `read_file("notes.txt; rm -rf /")` দিলে কী ঘটে? কেন নিরাপদ?
২. `max_steps` না থাকলে কোন failure ঘটতে পারে?
৩. কেন `temperature=0` রাখা হয়েছে?
