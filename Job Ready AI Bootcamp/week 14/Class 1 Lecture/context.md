# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Orchestration with LangGraph — State, Nodes/Edges, Conditional Edges, Human-in-the-Loop**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে, যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

একটা LLM call দিয়ে সহজ কাজ হয়। কিন্তু real agent-এর কাজ হলো: লেখো, যাচাই করো, ভুল হলে আবার চেষ্টা করো, দরকার হলে মানুষের অনুমতি নাও। এই "loop আর branch" যদি সাধারণ `while` আর `if` দিয়ে লেখা হয়, তাহলে state ছড়িয়ে যায়, মাঝপথে থেমে আবার শুরু করা যায় না। LangGraph এই flow-কে একটা graph হিসেবে আঁকতে দেয়, আর state, retry, pause/resume সব সামলায়।

## শেখার সহজ Workflow

Problem বোঝা → Core concept ও intuition → Step-by-step workflow → Practical example → কখন ব্যবহার করবেন বা করবেন না।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** Graph আঁকার আগে কাগজে node আর তীর (edge) এঁকে নিন। কোন node কোন state key পড়ে আর কোনটা লেখে, সেটা লিখে ফেললে অর্ধেক bug আগেই ধরা পড়ে।

---

## 1. Topic: Orchestration with LangGraph

Week 11-এ আমরা Ollama দিয়ে একটা prompt পাঠিয়ে উত্তর পেয়েছি, week 12-এ সেই prompt-এর সামনে retrieval জুড়েছি। এখন প্রশ্ন: একাধিক ধাপ, loop আর মানুষের সিদ্ধান্ত কীভাবে একসাথে চালাব? LangGraph-এ আমরা:

* **State** একটা `TypedDict` হিসেবে ঠিক করি, যেটা পুরো graph-এর shared memory।
* **Node** হলো সাধারণ Python function: state পড়ে, state-এর অংশ update করে ফেরত দেয়।
* **Edge** ঠিক করে পরে কোন node চলবে; **conditional edge** state দেখে runtime-এ পথ বেছে নেয়।
* **interrupt + checkpointer** দিয়ে graph মাঝপথে থামিয়ে মানুষের উত্তরের জন্য অপেক্ষা করি।

**Project Goal:** "Self-Correcting Writer" (Project 11): Python function লেখে, subprocess-এ test চালায়, fail করলে error সহ আবার লেখে, শেষে মানুষের approval নেয়।

---

## 2. Why It Is Related

আগের ক্লাসগুলোতে আমরা একক LLM call আর RAG pipeline বানিয়েছি, যেগুলো সরল সরলরেখার flow। Agent মানেই feedback loop: output যাচাই করে ফল ইনপুটে ফেরত দেওয়া। পরের ক্লাসে (Class 28) আমরা একাধিক agent নিয়ে multi-agent system বানাব; সেখানে প্রতিটা agent আসলে এই ক্লাসের graph-এর একটা node বা subgraph। তাই state আর routing ভালোভাবে বুঝে নিলে পরের সব orchestration সহজ লাগবে।

---

## 3. How It Works

### 3.1 State আর Reducer

```python
import operator
from typing import Annotated, TypedDict

class WriterState(TypedDict, total=False):
    task: str
    code: str
    attempts: int
    history: Annotated[list[str], operator.add]   # reducer
```

Node যখন `{"code": "..."}` ফেরত দেয়, LangGraph শুধু ওই key-টা overwrite করে। কিন্তু `history`-তে reducer (`operator.add`) আছে, তাই node `{"history": ["attempt 1"]}` ফেরত দিলে সেটা আগের list-এর সাথে জোড়া লাগে। Reducer না থাকলে প্রতিবার পুরো list মুছে নতুনটা বসত। Chat message-এর জন্য বিল্ট-ইন reducer হলো `add_messages` (এটা message ID দেখে update বা append করে)।

### 3.2 Node, Edge আর Conditional Edge

```python
g = StateGraph(WriterState)
g.add_node("write_code", write_code)
g.add_node("run_tests", run_tests_node)
g.add_edge(START, "write_code")
g.add_edge("write_code", "run_tests")
g.add_conditional_edges("run_tests", route_after_tests, ["write_code", "approval", "done", "give_up"])
graph = g.compile(checkpointer=InMemorySaver())
```

`route_after_tests(state)` একটা সাধারণ function, যেটা পরের node-এর নাম string হিসেবে ফেরত দেয়। আমাদের নিয়ম: test pass হলে approval বা done; fail হলে `attempts < max_attempts` থাকলে আবার `write_code`, না থাকলে `give_up`। এই `max_attempts` ছাড়া loop কখনো থামবে না, তাই এটা hard limit।

### 3.3 Human-in-the-Loop: interrupt + checkpointer

```python
from langgraph.types import interrupt, Command

def approval(state):
    decision = interrupt({"question": "Approve?", "code": state["code"]})
    return {"approved": bool(decision)}

config = {"configurable": {"thread_id": "run-1"}}
out = graph.invoke(start_state, config)          # approval-এ এসে থামে
out["__interrupt__"][0].value                    # মানুষকে দেখানোর payload
graph.invoke(Command(resume=True), config)       # একই thread_id-তে resume
```

`interrupt()` চললে graph-এর পুরো state checkpointer-এ সেভ হয় আর run থেমে যায়। পরে একই `thread_id` দিয়ে `Command(resume=value)` পাঠালে node শুরু থেকে আবার চলে, কিন্তু এবার `interrupt()` ওই value ফেরত দেয়। তাই interrupt-এর আগে side-effect (যেমন file লেখা বা API call) রাখলে তা দুবার চলতে পারে; এটা সাবধানে লিখতে হয়। Checkpointer ছাড়া interrupt কাজ করে না।

### 3.4 Generated code নিরাপদে চালানো

LLM-এর লেখা code সরাসরি `exec` করা বিপজ্জনক। আমরা আলাদা subprocess-এ `python -I` দিয়ে চালাই, temp directory-তে, timeout সহ। Test fail হলে stderr-এর শেষ অংশ পরের prompt-এ ঢুকিয়ে দিই; এটাই self-correction-এর জ্বালানি।

---

## 4. Details & Valid Points

### 4.1 Code-generation Model Reference Table

| Model | Use Case | Advantage | Limitation |
| --- | --- | --- | --- |
| **`llama3.1:8b`** (আমাদের default) | General-purpose, ছোট Python function লেখা | Instruction-following ভালো, 16GB RAM-এ comfortable, week 11/12-এর সাথে মেলে | কঠিন algorithm-এ ভুল করে, তাই retry loop দরকার |
| `mistral:7b` | দ্রুত inference, structured output | হালকা ও দ্রুত | Code quality llama3.1-এর কাছাকাছি, আলাদা সুবিধা কম |
| `qwen2.5-coder:7b` (code-specialised) | শুধু code লেখা ও fix করা | এই size-এ code benchmark-এ সাধারণত সেরা | General প্রশ্নে দুর্বল, আলাদা `ollama pull` লাগে |
| `llama3.2:3b` | কম RAM (8GB) machine | খুব ছোট ও দ্রুত | বেশি ভুল করে, retry বেশি লাগে |

> **Valid Point:** `llama3.1:8b` বেছে নেওয়ার কারণ: course জুড়ে একই local model ব্যবহার করলে setup এক থাকে, আর self-correction loop মডেলের ছোটখাটো ভুল পুষিয়ে দেয়। শুধু code-এর কাজ হলে `qwen2.5-coder` ভালো বিকল্প; model বদলাতে `config.py`-র এক লাইন বদলালেই হয়।

### 4.2 Checkpointer-এর ধরন

| Checkpointer | কখন | সীমাবদ্ধতা |
| --- | --- | --- |
| `InMemorySaver` | শেখা, test | Process বন্ধ হলে state হারায় |
| SQLite/Postgres saver | Production, দীর্ঘ approval অপেক্ষা | আলাদা package ও setup লাগে |

### 4.3 কখন LangGraph লাগবে না

Flow যদি সরল (একবার call, একবার parse), সাধারণ function যথেষ্ট। Loop, branch, pause/resume বা একাধিক agent লাগলে তবেই graph-এর সুবিধা পাওয়া যায়।

---

## 5. Related Analogy: The Restaurant Kitchen Pass

> কল্পনা করুন একটা রেস্তোরাঁর রান্নাঘর। অর্ডার একটা slip হয়ে স্টেশন থেকে স্টেশনে ঘোরে।

| Concept | Kitchen Analogy |
| --- | --- |
| **State** | Order slip: এতে অর্ডার, এ পর্যন্ত কী রান্না হয়েছে, কী বদলাতে হবে সব লেখা থাকে। |
| **Node** | একেকটা স্টেশন (grill, plating): slip পড়ে নিজের অংশ যোগ করে। |
| **Reducer** | Slip-এর "notes" ঘর: নতুন নোট আগেরগুলো মুছে না, নিচে জুড়ে যায়। |
| **Conditional edge** | Head chef চেখে দেখেন: ঠিক থাকলে পরিবেশন, নইলে আবার grill-এ। |
| **Retry limit** | তিনবার ফেরত গেলে অর্ডার বাতিল, যাতে খাবার চিরকাল ঘুরতে না থাকে। |
| **interrupt + checkpointer** | Customer-এর অনুমতি দরকার হলে slip আলাদা তাকে রাখা; উত্তর এলে ঠিক সেই জায়গা থেকে আবার শুরু। |

---

## Achievement: Self-Correcting Writer (Project 11)

কোড **`Class 1 Project/`** ফোল্ডারে আছে: `src/graph.py` (graph), `src/sandbox.py` (subprocess runner), `src/llm.py` (Ollama wrapper), `app.py` (CLI)। Test গুলো fake LLM দিয়ে চলে, তাই Ollama ছাড়াই `pytest` pass করে। Ollama বন্ধ থাকলে app একটা পরিষ্কার বার্তা দেখায়।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Reducer Experiment**: `history` থেকে `operator.add` সরিয়ে চালান। Output-এ কী বদলায়? কেন?
2. **Retry Limit**: `max_attempts=1` আর `10` দিয়ে একই কঠিন task চালান। কোনটা কখন ভালো?
3. **Reject Path**: Approval-এ `False` দিলে code কি আবার লেখা উচিত? Rejected হলে মানুষের feedback state-এ রেখে `write_code`-এ ফেরত পাঠানোর edge যোগ করুন।
4. **Model Swap**: `config.py`-তে `qwen2.5-coder:7b` আর `llama3.2:3b` দিয়ে চালান; প্রতিটায় কত attempt লাগল?
