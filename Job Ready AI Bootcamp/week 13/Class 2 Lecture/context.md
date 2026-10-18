# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: The ReAct Loop — Reasoning + Acting, Task Decomposition, and Failure Handling**। এখানে technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

Real-world AI system বানাতে শুধু একবার LLM call করলেই হয় না। "সবচেয়ে সস্তা ফ্লাইট খুঁজে আমার নামে বুক করো"—এই কাজে আগে খুঁজতে হবে, ফলাফল দেখে বেছে নিতে হবে, তারপর বুক করতে হবে, আর মাঝপথে কিছু ভুল হলে সামলাতে হবে। এই পুরো চক্রটা বোঝা দরকার। এই lecture সেই problem-solving mindset তৈরি করবে।

## শেখার সহজ Workflow

Problem বোঝা → Core concept ও intuition → Step-by-step workflow → Practical example → কখন ব্যবহার করবেন বা করবেন না।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?** এই প্রশ্নগুলোর উত্তর দিতে পারলে topic-টি শুধু মুখস্থ নয়, সত্যি বোঝা হয়েছে।

> **Practical mindset:** Example code run করার আগে expected input এবং output লিখে নিন। Run করার পরে result expectation-এর সাথে compare করুন এবং ভুল হলে কোন loop step-এ সমস্যা হয়েছে তা isolate করুন।

---

## 1. Topic: The ReAct Loop — Reasoning + Acting, Task Decomposition, and Failure Handling

Class 25-এ আমরা শিখেছি LLM কীভাবে tool call করে। কিন্তু একটা tool call দিয়ে বাস্তব কাজ শেষ হয় না। **ReAct** (Reason + Act) হলো এমন একটা loop যেখানে model বারবার তিনটা কাজ করে: **Thought** (কী করা দরকার ভাবা), **Action** (একটা tool চালানো), **Observation** (tool-এর ফলাফল দেখা)। এই চক্র চলতে থাকে যতক্ষণ না model **Final Answer** দেয়। এই ক্লাসে আমরা শিখব:

* **Reasoning and Acting loop** — loop-টা নিজের হাতে লেখা, কোনো framework ছাড়া।
* **Multi-step task decomposition** — একটা বড় কাজকে model নিজে ছোট ছোট step-এ ভাঙে।
* **Failure management** — retry, max-steps, tool error, আর loop detection।

**Project Goal:** একটা Travel Agent, যে local flights database (JSON) থেকে ফ্লাইট খোঁজে এবং একটা ফ্লাইট (simulated) বুক করে। LLM চলে Ollama-তে `llama3.1:8b`, আর `mistral:7b` ও `llama3.2:3b`-এর সাথে তুলনা করা হয়।

---

## 2. Why It Is Related

Week 12-এর RAG bot ছিল একমুখী pipeline: retrieve করো, তারপর generate করো। Step-এর ক্রম আগে থেকেই code-এ ঠিক করা ছিল। Agent-এ সেই ক্রম ঠিক করে model নিজে, আর প্রতিটা Observation দেখে পরের সিদ্ধান্ত বদলাতে পারে। এই স্বাধীনতার দাম হলো অনিশ্চয়তা: model ভুল tool বেছে নিতে পারে, ভাঙা JSON দিতে পারে, বা একই কাজ বারবার করতে থাকতে পারে। তাই ReAct loop আর তার failure handling একসাথে শিখতে হয়। পরের ক্লাসগুলোতে multi-agent system বানাতে এই loop-ই ভিত্তি হবে।

---

## 3. How It Works

### 3.1 Loop-এর গঠন

```
task ──► [LLM] ──► Thought + Action ──► [আমাদের code tool চালায]
            ▲                                   │
            └────────── Observation ◄───────────┘
                (Final Answer না আসা পর্যন্ত)
```

গুরুত্বপূর্ণ কথা: LLM কখনো নিজে tool চালায় না। সে শুধু লেখে "আমি `search_flights` চালাতে চাই এই argument দিয়ে"। চালায় আমাদের Python code, আর ফলাফল text হিসেবে conversation-এ ফেরত যায়। এই কারণেই validation আমাদের দায়িত্ব।

আমাদের project-এ প্রতি turn-এ model একটা JSON দেয়:

```json
{"thought": "আগে DAC থেকে CXB খুঁজি", "action": "search_flights",
 "action_input": {"origin": "DAC", "destination": "CXB", "date": "2026-11-05"}}
```

কাজ শেষ হলে দেয় `{"thought": "...", "final_answer": "..."}`। Ollama-র `format: "json"` option output-কে valid JSON রাখতে সাহায্য করে।

### 3.2 Task Decomposition

"সবচেয়ে সস্তা DAC to CXB ফ্লাইট Rahim-এর নামে বুক করো"—এই কাজ model নিজে ভাঙে:

1. `search_flights` চালাও (তারিখ ও route দিয়ে)।
2. Observation থেকে দাম অনুযায়ী সবচেয়ে সস্তা আর যেটায় seat আছে সেটা বাছো।
3. `book_flight` চালাও।
4. Booking reference দিয়ে Final Answer লেখো।

Decomposition-এর মান নির্ভর করে model-এর ওপর। বড় model পুরো পরিকল্পনা ঠিক রাখে, ছোট model মাঝপথে হারিয়ে যায় বা step বাদ দেয়। System prompt-এ "আগে search, তারপর book" নিয়ম স্পষ্ট লিখে দেওয়া এই ঝুঁকি কমায়।

### 3.3 Failure Handling: চারটা লাগাম

```
১. max_steps      → loop-এর সর্বোচ্চ চক্র। শেষ হলে status = "max_steps"।
২. LLM retry      → Ollama বন্ধ বা timeout হলে সীমিতবার আবার চেষ্টা, তারপর "llm_failed"।
৩. Error as data  → ভাঙা JSON বা tool error exception হয়ে crash করে না।
                    "ERROR: ..." Observation হয়ে model-কে শুধরানোর সুযোগ দেয়।
৪. Loop detection → একই action + একই input বারবার হলে "loop_detected" বলে থামানো।
```

কেন "error as data"? ধরুন model sold-out ফ্লাইট বুক করতে চাইল। Exception ছুঁড়ে program থামালে user কিছু পায় না। কিন্তু Observation-এ "Flight BG102 is sold out. Pick another flight." ফেরত দিলে model পরের step-এ অন্য ফ্লাইট বেছে নেয়। এই self-correction ReAct-এর সবচেয়ে বড় শক্তি।

### 3.4 Safety: Booking একটা side effect

Search নিরাপদ, কিন্তু booking পরিবর্তন ঘটায়। তাই `book_flight` tool-এর ভেতরেই নিয়ম: passenger name বাধ্যতামূলক, flight আছে কিনা, seat বাকি আছে কিনা, একই passenger একই ফ্লাইটে দুবার বুক করতে পারবে না। Prompt-এ "দুবার বুক কোরো না" লিখে ভরসা করা চলে না। নিয়মটা code-এ থাকতে হবে।

---

## 4. Details & Valid Points

### 4.1 LLM Model Reference Table (ReAct দৃষ্টিকোণ)

| Model | Use Case | Advantage | Limitation |
| --- | --- | --- | --- |
| **`llama3.1:8b`** (আমাদের project-এ default) | Multi-step reasoning + tool use | JSON format ভালো মানে, tool use-এর জন্য train করা | ~16GB RAM লাগে, 8GB মেশিনে ধীর |
| `mistral:7b` | দ্রুত general-purpose agent | Size কাছাকাছি, latency প্রায়ই কম | Format মাঝে মাঝে ভাঙে, JSON-এর বাইরে extra text মেশায় |
| `llama3.2:3b` | Low-RAM মেশিন, সহজ task | ছোট ও দ্রুত | বেশি step-এ হারিয়ে যায়, ভুল argument বানায়, loop-এ আটকায় |
| Cloud (GPT-4 / Claude) | সবচেয়ে নির্ভরযোগ্য agent | Planning ও tool use সেরা | Local না, খরচ আছে, data বাইরে যায় |

> **Valid Point:** Model যত ছোট, failure handling তত বেশি কাজে লাগে। `llama3.2:3b`-এ আপনি retry আর loop detection বেশি সক্রিয় দেখবেন। এটাই তুলনা করার আসল শিক্ষা: "কোন model সবচেয়ে ভালো" নয়, বরং "কোন model-এ কোন failure কত ঘন ঘন হয়"।

### 4.2 Hand-written Loop বনাম Framework

নিজে লিখলে আপনি দেখেন messages list-এ ঠিক কী যাচ্ছে, কখন থামছে, কোন error কোথায় ধরা পড়ছে। LangChain বা LangGraph-এর মতো framework এই কাজ ঢেকে রাখে। শেখার জন্য আগে hand-written, production-এ দরকারমতো framework।

### 4.3 কিছু Pitfall

* **Context বড় হতে থাকে:** প্রতি step-এ messages বাড়ে। বেশি step মানে ধীর ও দামি call, আর ছোট model-এর context ভরে যায়।
* **Temperature:** agent-এ `temperature=0` রাখুন। এলোমেলো output মানে এলোমেলো tool call।
* **`max_steps` খুব কম বা খুব বেশি:** আমাদের task-এ ঠিক দরকার ৩-৪ step, তাই limit ৮ যুক্তিসঙ্গত।
* **Loop detection-এর সীমা:** এটা শুধু হুবহু একই action ধরে। ঘুরিয়ে একই কাজ করলে (যেমন প্রতিবার max_price একটু বদলে) ধরা পড়ে না, সেখানে max_steps শেষ ভরসা।

---

## 5. Related Analogy: ট্র্যাভেল এজেন্সির নতুন কর্মী

> কল্পনা করুন একজন নতুন কর্মী কাউন্টারে বসে, আর আপনি তাকে বললেন "আমার জন্য সস্তা কক্সবাজার ফ্লাইট বুক করো।"

| Concept | Analogy |
| --- | --- |
| **Thought** | কর্মী মনে মনে ভাবে "আগে দেখি আজ কী কী ফ্লাইট আছে।" |
| **Action** | কম্পিউটারে search চালায়। |
| **Observation** | স্ক্রিনে ফলাফল দেখে: দুটো ফ্লাইট, একটা sold out। |
| **Final Answer** | আপনাকে বলে "বুক হয়ে গেছে, ref BK0001।" |
| **Tool error** | Sold out ফ্লাইটে বুক করতে গেলে সিস্টেম "না" বলে। ভালো কর্মী থেমে যায় না, অন্য ফ্লাইট দেখে। |
| **max_steps** | ম্যানেজার বলে দিয়েছে "১০ মিনিটের বেশি লাগলে আমাকে ডাকো।" |
| **Loop detection** | কর্মী একই ভুল search বারবার করছে দেখলে ম্যানেজার এসে থামিয়ে দেয়। |

---

## Achievement: Travel Agent (Hand-written ReAct Loop)

এই ক্লাসের কোড **`Class 2 Project/`** ফোল্ডারে আছে: `data/flights.json` (১০টা ফ্লাইট), `src/tools.py` (search ও simulated booking), `src/react_agent.py` (পুরো loop আর চার লাগাম), `src/llm_client.py` (Ollama), `app.py` (CLI, `--compare` দিয়ে তিন model-এর তুলনা)। Tests-এ Ollama লাগে না, একটা scripted fake LLM দিয়ে loop, failure handling আর booking যাচাই হয়।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Model Comparison**: একই task তিন model-এ চালান (`python app.py --compare "..."`)। কোন model কত step নিল, কোনটা কোথায় ভুল করল, তার একটা ছোট table বানান।
2. **Break the Agent**: এমন task দিন যেখানে কোনো ফ্লাইট নেই (যেমন DAC to JSR, নভেম্বরের ১০ তারিখ)। Agent কী করে? Final Answer সৎ হয়, নাকি বানিয়ে বলে?
3. **Loop Detection-এর ফাঁক**: এমন একটা fake LLM script লিখুন যা প্রতিবার `max_price` সামান্য বদলে search করে। Loop detection কি ধরে? কোন লাগাম শেষে থামায়?
4. **নতুন Tool**: `cancel_booking(booking_ref)` tool যোগ করুন। কোন কোন validation লাগবে, আর test কীভাবে লিখবেন?
