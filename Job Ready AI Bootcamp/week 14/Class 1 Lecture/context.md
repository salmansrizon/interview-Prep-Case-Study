# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Orchestration with LangGraph — State, Loop, Checkpoint আর Human Approval**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

Week 13-এ আমরা ReAct loop নিজের হাতে লিখেছি — একটা `while` loop, কয়েকটা `if`। কিন্তু বাস্তব agent-কে আরও কিছু পারতে হয়: মাঝপথে থেমে মানুষের approval নেওয়া, process restart হলেও আগের জায়গা থেকে চালু হওয়া, প্রতিটা ধাপ live দেখানো, আর দরকার হলে পুরোনো একটা ধাপে ফিরে গিয়ে আবার চেষ্টা করা। এগুলো নিজে বানাতে গেলে আসলে একটা ছোট workflow engine লিখতে হয়। **LangGraph** সেই engine — agent-কে একটা explicit graph হিসেবে লেখা যায়, যেখানে state typed, প্রতিটা ধাপ checkpoint হয়, আর loop/branch/pause চোখে দেখা যায়।

## শেখার সহজ Workflow

State → Node → Edge (conditional) → Loop-এর থামার শর্ত → Interrupt + Resume → Checkpointer + Time travel → Streaming → Self-correcting code-এর research → নিরাপদ execution → কখন LangGraph লাগবে না।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** একটা graph চালানোর আগে কাগজে আঁকুন: কোন node-এর পরে কোনটা, কোন শর্তে loop ফেরে, কোথায় থামে। তারপর `get_graph().draw_mermaid()` দিয়ে আসল graph-এর সাথে মেলান।

---

## 1. Topic: Orchestration with LangGraph

এই ক্লাসে আমরা শিখব:

* **StateGraph** — typed state, node (Python function), edge, conditional edge, `START`/`END`।
* **Reducer** — state-এর কোন key overwrite হবে আর কোনটা জোড়া লাগবে।
* **Human-in-the-loop** — `interrupt()` আর `Command(resume=...)`।
* **Persistence** — `InMemorySaver`, `SqliteSaver`, `PostgresSaver`; `thread_id`; time travel।
* **Streaming** — প্রতিটা node শেষ হলেই UI-তে দেখানো।
* **Self-correcting code generation** — research কী বলে, আর কোথায় এর সীমা।
* **Generated code নিরাপদে চালানো** — subprocess কেন sandbox না।

**Project Goal (Class 1 Project — Portfolio Project 12: Self-Correcting LLM Coder):** `llama3.1:8b` একটা Python function লেখে, আলাদা process-এ test চলে, fail করলে কোন test কী ফেরত দিল সেটা model-কে ফেরত যায়, model নিজে শুধরে আবার চেষ্টা করে — retry budget পর্যন্ত। শেষে hidden test (model কখনো দেখে না) আর মানুষের approval।

---

## 2. Why It Is Related

Class 26-এর হাতে লেখা ReAct loop-এ আমরা নিজেই step গুনেছি, repeat ধরেছি, approval gate বানিয়েছি। LangGraph সেই একই ধারণাগুলোকে framework-এর primitive বানায়: loop = conditional edge, approval = `interrupt()`, history = reducer, "কোথায় ছিলাম" = checkpointer। পরের ক্লাসে (Class 28) CrewAI উল্টো দিক থেকে আসে — নিয়ন্ত্রণ কম, agent-দের "role" বেশি। দুটো পাশাপাশি দেখলে বোঝা যায় কখন explicit control (LangGraph) আর কখন role-based team (CrewAI)। Week 15-এ এই agent-গুলোকে API হিসেবে serve আর monitor করব।

---

## 3. How It Works

### 3.1 State আর Reducer

State হলো graph-এর shared memory — সাধারণত একটা `TypedDict`। প্রতিটা node পুরো state পায়, আর ফেরত দেয় শুধু যা বদলাতে চায় (partial update)।

```python
import operator
from typing import Annotated, TypedDict

class CoderState(TypedDict, total=False):
    code: str                                   # reducer নেই → প্রতিবার overwrite
    attempts: int
    history: Annotated[list[str], operator.add]  # reducer আছে → নতুন item জোড়া লাগে
```

**Reducer ছাড়া শেষ write জেতে।** `history`-তে `operator.add` না দিলে প্রতিটা node আগের history মুছে দিত। দুটো parallel node একই key-তে reducer ছাড়া লিখলে LangGraph `InvalidUpdateError` ছোড়ে। `add_messages` chat message-এর জন্য বিশেষ reducer — নতুনটা জোড়ে, একই ID-র পুরোনোটা বদলায়। ইচ্ছে করে reducer এড়াতে `Overwrite([])`।

### 3.2 Node, Edge আর Conditional Edge

```python
from langgraph.graph import StateGraph, START, END

g = StateGraph(CoderState)
g.add_node("generate", generate)            # node = state নেয়, dict ফেরত দেয়
g.add_node("run_visible", run_visible)
g.add_edge(START, "generate")
g.add_edge("generate", "run_visible")
g.add_conditional_edges("run_visible", route, ["run_hidden", "generate", "give_up"])
graph = g.compile(checkpointer=InMemorySaver())
```

`route` একটা সাধারণ function — state দেখে পরের node-এর নাম ফেরত দেয়। **Loop মানে শুধু একটা conditional edge যেটা আগের node-এ ফেরে।** তৃতীয় argument-এ সম্ভাব্য গন্তব্যের list দিলে `draw_mermaid()` সেই edge-গুলো আঁকতে পারে।

দুটো উন্নত primitive: `Command(update=..., goto=...)` — একটা node একসাথে state বদলায় আর পরের node বেছে নেয়; `Send("worker", payload)` — fan-out (map-reduce), যেমন একসাথে কয়েকটা draft লেখা।

### 3.3 Loop-এর থামার শর্ত — 1.x-এর সবচেয়ে বিপজ্জনক পরিবর্তন

২০২৪-এর tutorial বলে default `recursion_limit` = ২৫ — অর্থাৎ ভুল loop ২৫ ধাপ পরে নিজেই থামবে। **LangGraph 1.0.6 (Jan 2026) থেকে এটা আর সত্য না:** code-এ default এখন ~১০,০০০ (PR #6676; source-এ `10007`, docs-এ "1000" লেখা — code-ই সঠিক ধরুন)। llama3.1:8b-এর প্রতি call ~১৫-২০ সেকেন্ড হলে ভুল loop ঘণ্টার পর ঘণ্টা চলতে পারে। তাই:

* থামার শর্ত **state-এ** রাখুন (`attempts < max_attempts`) আর router-এ check করুন।
* Backstop হিসেবে config-এ `recursion_limit` দিন — শেষ হলে `GraphRecursionError`।

```python
config = {"configurable": {"thread_id": "run-42"}, "recursion_limit": 40}
```

### 3.4 Human-in-the-Loop: `interrupt()` আর `Command(resume=...)`

```python
from langgraph.types import interrupt, Command

def approval(state):
    decision = interrupt({"code": state["code"]})     # এখানে graph থামে, checkpoint হয়
    return {"approved": bool(decision)}

out = graph.invoke(initial, config)                   # out["__interrupt__"] = [Interrupt(value=...)]
out = graph.invoke(Command(resume=True), config)      # একই thread_id — থামা জায়গা থেকে চলে
```

**সবচেয়ে গুরুত্বপূর্ণ নিয়ম: resume-এ node শুরু থেকে আবার চলে।** `interrupt()`-এর আগে LLM call বা test থাকলে প্রতিটা resume-এ সেটা আবার হবে। তাই approval-এর জন্য আলাদা node, যেখানে interrupt-এর আগে কিছু নেই। আরও নিয়ম: `interrupt()`-কে খালি `try/except`-এ মোড়াবেন না (এটা exception ছুড়েই কাজ করে); payload JSON-serializable রাখুন; interrupt checkpointer আর `thread_id` ছাড়া কাজ করে না।

### 3.5 Checkpointer আর Time Travel

Checkpointer প্রতিটা super-step-এর পরে state সেভ করে, `thread_id` অনুযায়ী।

| Saver | Install / import | কখন |
| --- | --- | --- |
| `InMemorySaver` | built-in, `langgraph.checkpoint.memory` | Test, notebook — process বন্ধ হলে সব হারায় |
| `SqliteSaver` | `pip install langgraph-checkpoint-sqlite` | Local app — restart-এর পরও থাকে |
| `PostgresSaver` | `pip install langgraph-checkpoint-postgres`, তারপর `.setup()` | Production, অনেক user |

`get_state(config)` = এখনকার state; `get_state_history(config)` = প্রতিটা checkpoint (নতুন থেকে পুরোনো)। পুরোনো একটা snapshot-এর `config` দিয়ে `graph.invoke(None, snap.config)` = সেখান থেকে **replay** (node আবার চলে)। `update_state(snap.config, values)` = নতুন শাখা (fork)। আমাদের project-এ এক retry-র একটা run-এ (approval ছাড়া) ১০টা checkpoint তৈরি হয়েছে — notebook-এর time-travel table দেখুন।

### 3.6 Streaming

```python
for chunk in graph.stream(initial, config, stream_mode="updates"):
    for node, update in chunk.items():   # প্রতিটা node শেষ হলেই একটা chunk
        print(node, update.get("history"))
```

সাতটা stream mode: `values` (পুরো state), `updates` (node-এর delta), `messages` (LLM token), `custom` (`get_stream_writer()`), `checkpoints`, `tasks`, `debug`। 1.1-এ `version="v2"` এসেছে (প্রতিটা chunk `{"type", "ns", "data"}`), 1.2-তে `stream_events(version="v3")` — দুটোই opt-in; default এখনো v1।

### 3.7 Functional API — বিকল্প লেখার ধরন

`@entrypoint` আর `@task` দিয়ে সাধারণ Python control flow-তেই checkpoint, interrupt, streaming পাওয়া যায়। বড় পার্থক্য: resume-এ entrypoint শুরু থেকে চলে, কিন্তু **শেষ হওয়া `@task`-এর ফল checkpoint থেকে আসে, আবার চলে না**। দাম: graph আঁকা যায় না, state declared schema-য় থাকে না।

---

## 4. Details & Valid Points

### 4.1 Self-Correcting Code — research কী বলে

| Work | Feedback signal | ফলাফল |
| --- | --- | --- |
| Self-Refine (NeurIPS 2023) | Model-এর নিজের critique, execution নেই | ৭টা task-এ গড়ে ~২০% |
| Self-Debugging (ICLR 2024) | Execution + নিজেকে ব্যাখ্যা | Unit-test feedback-এ **+১২%** পর্যন্ত; বেশিরভাগ fix **৩ turn-এর মধ্যে** |
| Reflexion (NeurIPS 2023) | Verbal reflection, episodic memory | HumanEval **৯১% pass@1** |
| AlphaCodium (2024) | Test-ভিত্তিক "flow engineering" | GPT-4 CodeContests pass@5 ১৯% → ৪৪% |
| "Is Self-Repair a Silver Bullet?" (ICLR 2024) | Compute ধরে হিসাব | লাভ "often modest... sometimes not present at all" |

Self-Debugging-এর ablation: সাধারণ "ভুল হয়েছে"-এর চেয়ে **unit-test feedback** বেশি কাজ করে; পুরো execution trace কখনো ক্ষতি করে (Codex: ৯২.৫ → ৮৭.৯)। সব repair step greedy decoding-এ (temperature 0)। Olausson et al.: compute গুনলে অনেক repair-এর চেয়ে বেশি প্রথম draft নেওয়া প্রায়ই সমান বা ভালো; আর **feedback-এর মানই bottleneck** — Code Llama নিজে ৯.১% fail ঠিক করেছিল, GPT-4-এর লেখা feedback পেয়ে ৩৯.৩%।

> **Valid Point:** ছোট model নিজের bug ঠিকমতো চিনতে পারে না — তাই diagnosis harness করে: কোন test, কী input, কী ফেরত এল, কী প্রত্যাশিত ছিল। আমাদের project-এ feedback হুবহু এরকম: `roman_to_int('IV'): returned 6 (int), expected 4`।

### 4.2 Code-generation Model Reference Table

| Model | Use Case | Advantage | Limitation |
| --- | --- | --- | --- |
| **`llama3.1:8b`** (course-এর একমাত্র model, ADR 0006) | ছোট function, self-repair loop | HumanEval **72.6**, MBPP++ 72.8 (Meta, full precision) | Code-specialist-দের চেয়ে ~১৬ point পিছিয়ে; Ollama-র 4-bit-এ আরও একটু কম |
| Qwen2.5-Coder-7B (শুধু তুলনা) | Code-only | HumanEval 88.4, HumanEval+ 84.1 | General প্রশ্নে দুর্বল |
| DeepSeek-Coder-6.7B (শুধু তুলনা) | Code-only | HumanEval 74.4 | পুরোনো |
| Cloud (GPT/Claude) | কঠিন algorithm | সবচেয়ে শক্তিশালী | Local না, খরচ |

**pass@k** = k-টা sample-এর অন্তত একটা পাস করার সম্ভাবনা (Chen et al. 2021: n sample, c সঠিক → 1 − C(n−c,k)/C(n,k))। HumanEval-এ problem-প্রতি test কম — EvalPlus ৮০ গুণ বেশি test যোগ করে pass@k **২৮.৯% পর্যন্ত** কমিয়ে দেয়।

### 4.3 আমাদের Live Benchmark (llama3.1:8b, ১১টা task, retry budget ৪)

| Metric | ফল |
| --- | --- |
| pass@1 (প্রথম attempt-এ visible + hidden সব) | **৮/১১ (৭৩%)** |
| self-repair-এর পরে | **৯/১১ (৮২%)** |
| repaired | ১ — `format_duration`, ৩য় attempt-এ |
| gave_up | ২ — `word_freq`, `camel_to_snake` |

`word_freq`-এর গল্প শিক্ষণীয়: প্রথম attempt visible ৩/৩ পাস কিন্তু hidden fail; generic hint ("আরও case-এ ভুল") পেয়ে পরের attempt-গুলো **খারাপ** হয়েছে। Olausson-এর কথাই — feedback নির্দিষ্ট না হলে repair উল্টো ফল দিতে পারে।

### 4.4 Reward Hacking — "test পাস" মানেই সঠিক না

ImpossibleBench (ICLR 2026) test-কে spec-এর বিপরীত বানায় — তাই যেকোনো "পাস" মানেই cheating। GPT-5 Oneoff-SWEbench-এ **৭৬%** task-এ cheat করেছে। চারটা কৌশল: test বদলানো, `__eq__` overload, call-এর মাঝে state রাখা, test input special-case করা। Mitigation: strict prompt (GPT-5 >৮৫% → ১%), "flag for human" বিকল্প (৫৪% → ৯%), test লুকানো।

আমাদের harness-এর উত্তর: model শুধু function দেয়, test harness-এর হাতে; **hidden test** কখনো prompt-এ যায় না (fail হলে শুধু generic hint); result-এর type হুবহু মিলতে হয় (`__eq__` overload ঠেকায়); আর approval node-ই "flag for human"।

### 4.5 নিরাপদ execution — subprocess sandbox না

OWASP **LLM05:2025** ঠিক এটাই বলে: LLM-এর output `exec`/`eval`/shell-এ গেলে remote code execution। বাস্তব উদাহরণ: Vanna.AI **CVE-2024-5565 (CVSS 8.1)** — prompt injection দিয়ে generated Plotly code বদলে চালানো গিয়েছিল।

আমাদের harness: আলাদা process, `python -I`, খালি env (API key-এর মতো secret child পায় না), temp dir, timeout, `RLIMIT_CPU/AS/FSIZE`। কিন্তু এটা আপনার user হিসেবেই চলে — disk পড়তে পারে, network-এ যেতে পারে (notebook-এ `os.listdir('/')` দেখানো)। আসল isolation, দুর্বল থেকে শক্ত: Docker (`--network none --read-only --cap-drop ALL`), gVisor, nsjail, Firecracker microVM / e2b, Pyodide/WASM।

AST blocklist (`os`, `subprocess`, `eval`...) ছোট model-এর জন্য **ভালো feedback**, কিন্তু নিরাপত্তা না — `().__class__.__base__.__subclasses__()` দিয়ে পার পাওয়া যায়।

### 4.6 কখন LangGraph লাগবে না

একটা সোজা write → test → retry loop ৩০ লাইনের `while`-এ হয়। LangGraph-এর দাম তখনই সার্থক যখন লাগে: মাঝপথে human approval, restart-এর পরে resume, live progress streaming, পুরোনো ধাপে ফেরা, বা অনেক branch। Stock tool-calling agent-এর জন্য `langchain.agents.create_agent` (1.x-এ `create_react_agent` deprecated)।

---

## 5. Related Analogy: The Hospital Surgery Checklist

> একটা অপারেশন থিয়েটার — প্রতিটা ধাপ লেখা থাকে, আর কোথাও কেউ "পরে দেখব" বলতে পারে না।

| Concept | Surgery Analogy |
| --- | --- |
| **State** | রোগীর chart — সবাই একই chart পড়ে আর লেখে |
| **Reducer** | Chart-এর "medication log" — নতুন entry যোগ হয়, পুরোনোটা মোছে না |
| **Node** | একটা নির্দিষ্ট কাজ: anesthesia, incision, suture |
| **Conditional edge** | "রক্তচাপ কম হলে আগে stabilize, তারপর এগোও" |
| **Retry budget** | "তিনবার চেষ্টার পরে senior surgeon ডাকো" — অসীম চেষ্টা না |
| **interrupt()** | Consent form — রোগীর সই ছাড়া পরের ধাপে যাওয়া নিষেধ |
| **Checkpointer** | প্রতিটা ধাপের পরে chart-এ লেখা সময় আর অবস্থা — shift বদলালেও নতুন টিম ঠিক সেখান থেকে শুরু করে |
| **Time travel** | Post-op review: "১১টা ৪০-এ কী অবস্থা ছিল?" — আর সেখান থেকে বিকল্প সিদ্ধান্ত বিশ্লেষণ |
| **Hidden tests** | Pathology report — surgeon নিজে যা দেখে তার বাইরে স্বাধীন যাচাই |

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
| --- | --- | --- |
| Attempt cap নেই | 1.x-এ ~১০ হাজার ধাপ পর্যন্ত loop | State-এ `attempts`, router-এ check, `recursion_limit` backstop |
| `interrupt()`-এর আগে LLM call/test | প্রতি resume-এ আবার চলে | Approval-এর আলাদা node |
| History list-এ reducer নেই | প্রতিটা node আগেরটা মুছে দেয় | `Annotated[list, operator.add]` |
| Parallel node একই key-তে | `InvalidUpdateError` | Reducer, বা আলাদা key |
| `state[...]`-কে সরাসরি mutate | Update checkpoint নাও হতে পারে | নতুন dict ফেরত দিন |
| পুরো history prompt-এ জমানো | `num_ctx` ভরে যায় | Spec + শেষ code + শেষ failure যথেষ্ট |
| Hidden test prompt-এ ফাঁস | Model মুখস্থ করে "পাস" | Hidden শুধু শেষে, feedback generic |
| শুধু visible test-এ grade | Overfit code approve হয় | দুটো pass rate আলাদা রিপোর্ট |
| Repair-এ high temperature | Model feedback উপেক্ষা করে নতুন করে লেখে | Repair greedy; নতুন draft-এ শুধু উঁচু temperature |
| `exec()` orchestrator process-এ | `while True` app আটকায়, `sys.exit` app মারে, RCE | আলাদা process + limit; production-এ container |
| Child process-এ পুরো `os.environ` | API key ফাঁস | খালি env দিন |
| AST blocklist-কে security ভাবা | Introspection দিয়ে বাইপাস | OS/VM boundary-ই security |

---

## 7. Interview Q&A

1. **Reducer কী, না থাকলে কী ভাঙে?** `(current, update) → new` function, `Annotated`-এ বসে। না থাকলে শেষ write জেতে — জমানো list চুপচাপ মুছে যায়, আর দুটো parallel writer `InvalidUpdateError` দেয়।
2. **`interrupt()` resume করলে কী হয়?** Pause checkpoint হয়, payload `__interrupt__`-এ আসে। একই `thread_id`-তে `Command(resume=v)` দিলে **node শুরু থেকে আবার চলে** আর `interrupt()` `v` ফেরত দেয় — তাই interrupt-এর আগের code idempotent হতে হবে বা আগের node-এ থাকতে হবে।
3. **Checkpointer কী রাখে, time travel কীভাবে?** প্রতি super-step-এ channel value, পরের node, metadata — `thread_id` + `checkpoint_id` দিয়ে। `get_state_history` দিয়ে snapshot খুঁজে `invoke(None, snap.config)` = replay; `update_state` = fork। Test-এ InMemory, local-এ SQLite, production-এ Postgres।
4. **Self-correcting loop চিরকাল চলা কীভাবে থামাবেন?** Router-এ attempts counter, code-hash দিয়ে repeat detector, আর explicit `recursion_limit`। বলুন যে 1.0.6-এ default ২৫ থেকে ~১০ হাজার হয়েছে — এতে বোঝা যায় আপনি library-র বর্তমান অবস্থা জানেন।
5. **Conditional edge, `Command`, `Send` — কখন কোনটা?** Routing state-এর pure function হলে conditional edge (graph-এ দেখা যায়); node-কে একসাথে update আর route করতে হলে `Command`; fan-out/map-reduce-এ `Send`।
6. **Graph API নাকি Functional API?** Graph API: explicit node/edge, আঁকা যায়, প্রতি super-step checkpoint। Functional API: সাধারণ Python, `@task`-এর ফল resume-এ cache থেকে আসে, কিন্তু আঁকা যায় না।
7. **Self-repair কি free win?** না। Compute গুনলে লাভ ছোট, কখনো নেই; diverse প্রথম draft বেশি জরুরি; feedback-এর মানই bottleneck (Code Llama ৯.১% → GPT-4 feedback-এ ৩৯.৩%)। ১-২টা repair আর নতুন sample মেশান।
8. **কী feedback, কয় turn, কী temperature?** Failing assert-এর expected/actual আর শেষ exception line; ~৩ turn; repair greedy, নতুন draft-এ ০.৭-০.৮।
9. **Retry যোগ করে pass rate ১০০% হলো — কী সন্দেহ করবেন?** Reward hacking: hard-coded output, special-case input, `__eq__` overload, test edit। Hidden-test pass rate দেখুন, plain-type comparison করুন।
10. **`subprocess.run(timeout=5)` কি sandbox? `import os` block করলেই হয় না?** না — hang থামায়, দুষ্টুমি না; child আপনার user, disk, network, env পায়। Blocklist `__subclasses__()` বা `getattr` string দিয়ে বাইপাস হয়। আসল isolation container/gVisor/microVM; OWASP LLM05 আর Vanna.AI CVE-2024-5565 উল্লেখ করুন।
11. **HumanEval-এর বদলে HumanEval+ কেন?** ৮০ গুণ বেশি test-এ pass@k ২৮.৯% পর্যন্ত কমে আর ranking বদলায় — আসল HumanEval সঠিকতা বাড়িয়ে দেখায়; contamination-এর জন্য LiveCodeBench।
12. **`while` loop-এর বদলে LangGraph কেন?** ৩০ লাইনের loop write→test→retry পারে; কিন্তু approval-এ থামা, restart-এর পরে resume, streaming, পুরোনো attempt-এ ফেরা — এগুলোর জন্য নিজে checkpointing বানাতে হতো।

---

## Achievement: Self-Correcting LLM Coder (Portfolio Project 12)

**`Class 1 Project/`** — LangGraph `StateGraph` (generate → static check → visible tests → hidden tests → approval), per-test feedback, hidden-test রহস্য রাখা, subprocess harness with limits, SQLite checkpointer + time travel, Streamlit UI (live streaming + Approve/Reject), ১১-task benchmark (pass@1 বনাম repair), ২৭টা offline test। এই ক্লাসের companion notebook **`langgraph_lecture.ipynb`** সম্পূর্ণ self-contained — project folder ছাড়াই চলে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Resume Trap:** approval node-এ `interrupt()`-এর আগে একটা `print("running tests")` বসান। একবার pause আর একবার resume করলে কয়বার print হয়? কেন?
2. **Reducer ছাড়া:** `history`-র `operator.add` সরিয়ে graph চালান। শেষ state-এ history-তে কী থাকে?
3. **Best-of-N বনাম Repair:** একই LLM-call budget (৪) দিয়ে দুটো কৌশল তুলনা করুন — (ক) ১ draft + ৩ repair, (খ) ৪টা স্বাধীন draft (temperature 0.8), যেকোনো একটা পাস করলেই হলো। কোনটা বেশি task পাস করে?
4. **Cheat Detector:** এমন একটা check লিখুন যেটা generated code-এ visible test-এর expected value-গুলো literal হিসেবে খোঁজে (hard-coding-এর লক্ষণ)। False positive কোথায় হবে?
5. **Time Travel Fork:** `get_state_history` থেকে attempt ১-এর পরের checkpoint নিন, `update_state` দিয়ে `feedback`-এ নিজের লেখা hint বসান, তারপর সেখান থেকে চালান। নিজের hint কি generic hint-এর চেয়ে ভালো কাজ করে?

---

## References

* LangGraph docs — Graph API https://docs.langchain.com/oss/python/langgraph/graph-api · Interrupts https://docs.langchain.com/oss/python/langgraph/interrupts · Persistence https://docs.langchain.com/oss/python/langgraph/persistence · Time travel https://docs.langchain.com/oss/python/langgraph/use-time-travel · Streaming https://docs.langchain.com/oss/python/langgraph/streaming · Functional API https://docs.langchain.com/oss/python/langgraph/functional-api
* What's new in LangGraph v1 — https://docs.langchain.com/oss/python/releases/langgraph-v1 · LangChain/LangGraph 1.0 blog — https://www.langchain.com/blog/langchain-langgraph-1dot0
* Recursion limit change, PR #6676 — https://github.com/langchain-ai/langgraph/pull/6676
* Self-Debugging — https://arxiv.org/abs/2304.05128 · Self-Refine — https://arxiv.org/abs/2303.17651 · Reflexion — https://arxiv.org/abs/2303.11366 · AlphaCodium — https://arxiv.org/abs/2401.08500 · LDB — https://arxiv.org/abs/2402.16906
* Olausson et al., "Is Self-Repair a Silver Bullet for Code Generation?" — https://arxiv.org/abs/2306.09896
* ImpossibleBench — https://arxiv.org/abs/2510.20270
* Llama 3.1 model card — https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md · Qwen2.5-Coder report — https://arxiv.org/html/2409.12186
* Codex / pass@k (Chen et al. 2021) — https://arxiv.org/abs/2107.03374 · EvalPlus — https://arxiv.org/abs/2305.01210
* OWASP LLM05:2025 — https://genai.owasp.org/llmrisk/llm052025-improper-output-handling/ · Vanna.AI CVE-2024-5565 — https://jfrog.com/blog/prompt-injection-attack-code-execution-in-vanna-ai-cve-2024-5565
* Python docs: subprocess https://docs.python.org/3/library/subprocess.html · resource https://docs.python.org/3/library/resource.html · `-I` https://docs.python.org/3/using/cmdline.html#cmdoption-I
