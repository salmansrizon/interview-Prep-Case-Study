# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: The ReAct Loop — Reasoning + Acting, আর সেটাকে নির্ভরযোগ্য বানানোর engineering**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

Class 25-এ আমরা একটা tool call কীভাবে কাজ করে শিখেছি। কিন্তু বাস্তব কাজ এক ধাপে শেষ হয় না: "সবচেয়ে সস্তা ফ্লাইট বুক করো" মানে আগে খোঁজা, ফলাফল দেখে সিদ্ধান্ত, তারপর বুক — পরের ধাপ নির্ভর করে আগের ধাপের ফলাফলের ওপর। **ReAct** সেই loop-এর নাম। আর ছোট local model-এ loop-টা নিজে থেকে নির্ভরযোগ্য না — তাই এই ক্লাসের আসল বিষয় হলো loop-এর চারপাশের code: কখন থামবে, ভুল হলে কী হবে, আর কীভাবে মাপব এটা কতটা কাজ করে।

## শেখার সহজ Workflow

Paper-এর idea → Thought/Action/Observation → Text ReAct বনাম native tool calling → Loop-এর পাঁচটা exit → Deterministic tool → Validation ও error recovery → Human approval → Evaluation (pass^k) → ReAct-এর উত্তরসূরি pattern।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** প্রতিটা loop চালানোর আগে লিখে ফেলুন সেটা কোন কারণে থামবে (`final`, `loop`, `max_steps`...)। Agent debug করা মানে "কেন থামল" প্রশ্নের উত্তর খোঁজা।

---

## 1. Topic: The ReAct Loop

এই ক্লাসে শিখব:

* **ReAct paper (Yao et al., 2022, ICLR 2023)** — reasoning trace আর action interleave করা; কোথায় জেতে, কোথায় হারে।
* **দুই implementation** — classic text ReAct (`Thought:` / `Action:` / `Observation:`, regex parse) আর ২০২৬-এর default native tool calling।
* **Loop engineering** — typed stop reason, repeat detection, deadline, error budget, deterministic fallback।
* **Deterministic tool, mostly-reproducible LLM** — কেন `temperature=0` যথেষ্ট না।
* **Agent evaluation** — end state, trajectory, pass^k (τ-bench), BFCL।
* **ReAct-এর পরে** — Plan-and-Execute, ReWOO, Reflexion, LATS, CodeAct।

**Project Goal (Class 2 Project — Portfolio Project 11: Deterministic Function-Calling Travel Agent):** একটা conversational travel agent যে local flight database-এ খোঁজে আর **আপনার approval নিয়ে** simulated booking করে — framework ছাড়া হাতে লেখা ReAct loop, সাথে একটা evaluation harness যেটা সৎভাবে মাপে agent কতটা কাজ করে।

---

## 2. Why It Is Related

Class 25-এর tool loop-ই ReAct-এর কঙ্কাল — model tool ডাকে, result দেখে, আবার ডাকে। সেখানে আমরা দেখেছি llama3.1:8b একই ব্যর্থ call বারবার করে `max_steps` পর্যন্ত ঘোরে (prompt-injection experiment)। এই ক্লাস সেই সমস্যার পূর্ণ সমাধান দেয়। Week 14-এ LangGraph এই loop-কে একটা graph বানায় (state, node, conditional edge), আর CrewAI একাধিক ReAct agent-কে একটা team বানায় — দুটোই বোঝা সহজ হয় যদি নিচের loop নিজের হাতে লেখা থাকে। Week 15-এ এই agent-কে FastAPI service আর monitoring-এর আওতায় আনব।

---

## 3. How It Works

### 3.1 Paper-এর মূল idea

ReAct ("Synergizing Reasoning and Acting in Language Models") LLM-কে বলে **reasoning trace আর task action পালাক্রমে** লিখতে। Reasoning দিয়ে model plan বানায়, track করে, exception সামলায়; action দিয়ে বাইরে থেকে তথ্য আনে, যা reasoning-কে মাটিতে রাখে (grounding)। Paper-এর Wikipedia environment-এ action ছিল তিনটা: `search[entity]`, `lookup[string]`, `finish[answer]`; prompt-এ মাত্র ৬টা (HotpotQA) বা ৩টা (FEVER) example।

```
Question: Colorado orogeny-র eastern sector কোন এলাকায় বিস্তৃত, তার elevation range কত?
Thought 1: Colorado orogeny search করি, eastern sector খুঁজি।
Action 1: search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building...
Thought 2: Eastern sector-এর কথা নেই। lookup করি।
Action 2: lookup[eastern sector]
Observation 2: ...extends into the High Plains...
Thought 3: High Plains-এর elevation লাগবে।
Action 3: search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions...   ← disambiguation!
Thought 4: United States-এর High Plains search করি।
Action 4: search[High Plains (United States)]
Observation 4: ...rises in elevation from around 1,800 to 7,000 ft...
Action 5: finish[1,800 to 7,000 ft]
```

Step 3-এ disambiguation page এসেছিল — model observation দেখে plan বদলেছে। এই **adaptivity**-ই loop-এর কারণ।

### 3.2 ফলাফল — খ্যাতির চেয়ে বিনয়ী

| Benchmark (PaLM-540B) | ReAct | তুলনা |
| --- | --- | --- |
| HotpotQA (EM) | 27.4 | CoT 29.4 — **ReAct একা সামান্য খারাপ**; ReAct→CoT-SC hybrid 35.1 |
| FEVER (acc) | 60.9 | CoT 56.3; CoT-SC→ReAct 64.6 |
| ALFWorld (success) | **71%** | Act-only 45%, BUTLER 37% |
| WebShop (success) | **40.0%** | Act-only 30.1% |

**কাজের নিয়ম:** পরের ধাপ যখন environment-এর সদ্য-ফেরত-আসা ফলাফলের ওপর নির্ভর করে (interactive task), ReAct তখনই জেতে।

### 3.3 Paper-এর error analysis = loop engineering-এর ভিত্তি

| HotpotQA failure | ReAct | CoT |
| --- | --- | --- |
| Hallucination | **0%** | **56%** |
| Reasoning error (repetitive loop সহ) | **47%** | 16% |
| অকেজো search result | 23% | — |

Grounding hallucination মুছে দেয়, কিন্তু নতুন failure আনে: **agent আটকে যায়** — একই কাজ বারবার, বা খারাপ observation থেকে আর উঠতে পারে না। Authors সন্দেহ করেন greedy decoding repetition বাড়ায়। নিচের প্রতিটা guard (step cap, repeat detection, compact observation, error budget) এই table থেকে এসেছে।

### 3.4 Text ReAct বনাম Native Tool Calling

| | Text ReAct (২০২২) | Native tool calls (২০২৬ default) | Code actions (CodeAct) |
| --- | --- | --- | --- |
| Action format | `Action:` line, regex parse | JSON `tool_calls`, args object | Sandbox-এ Python snippet |
| নকল Observation | সম্ভব — `stop=["\nObservation:"]` লাগে | অসম্ভব — result আসে `role: tool` দিয়ে | শুধু interpreter output |
| Thought দেখা যায় | স্পষ্ট `Thought:` | Optional content | Comment / code-এর আগে text |
| Model support | যেকোনো model | Tool-capable template | শক্তিশালী code model |

Text mode-এ সবচেয়ে বড় বিপদ: model নিজেই `Observation: ...` লিখে ফেলে — মানে tool result **বানিয়ে** নেয়। Ollama-র `options={"stop": ["\nObservation:"]}` দিলে generation সেখানেই থামে, আর আসল observation আমরা বসাই। (আমাদের live test-এ llama3.1 chat API-তে নিজেই `Action Input`-এর পরে থেমেছিল — template turn শেষে `<|eot_id|>` বসায়। তবু stop রাখুন: অন্য model, template বা raw completion-এ এটাই একমাত্র সুরক্ষা।)

### 3.5 Loop-এর পাঁচটা exit, প্রতিটা typed

Anthropic-এর ভাষায় agent হলো "LLMs using tools based on environmental feedback in a loop", আর "stopping conditions (such as a maximum number of iterations)" রাখা স্বাভাবিক। Ollama কোনো guard দেয় না — আমাদের লিখতে হয়:

```python
while llm_calls < MAX_STEPS:                                  # ২. max_steps
    if clock() - start > DEADLINE: return done("timeout")     # ৪. timeout
    reply = llm(messages, tools=TOOLS)
    if not reply.tool_calls: return done("final", reply.content)   # ১. final
    for call in reply.tool_calls:
        key = (call.name, json.dumps(call.args, sort_keys=True))   # key-এর ক্রম model বদলায়!
        if seen[key] == 1: obs = reminder(previous_result)    # প্রথম repeat: মনে করিয়ে দাও
        if seen[key] >= 2: return done("loop")                # ৩. loop
        obs = tools.execute(call.name, call.args)             # কখনো raise করে না
        errors = errors + 1 if is_model_mistake(obs) else 0
    if errors >= MAX_ERRORS: return done("error_budget")      # ৫. error budget
```

Typed status থাকলে test বলতে পারে **কেন** থামল, শুধু থামল কিনা না। থামার পর উত্তরটা আরেকটা LLM call না — **deterministic fallback**, শেষ সফল observation থেকে বানানো ("Nothing was booked. Flights I found: BG101 08:00 6500 BDT…")।

| Framework | Cap | Default | Breach হলে |
| --- | --- | --- | --- |
| OpenAI Agents SDK | `max_turns` | 10 (JS SDK-তে confirmed) | `MaxTurnsExceeded` / `error_handlers` |
| LangGraph / `create_agent` | `recursion_limit` | 25 (third-party sources) | `GRAPH_RECURSION_LIMIT` |
| BFCL v3 benchmark | — | 20 step / turn | Force-terminate, incorrect |
| Ollama | নেই | — | **আপনি লিখবেন** |

### 3.6 Deterministic tool — হ্যাঁ; deterministic LLM — না

Tools পুরোপুরি deterministic বানানো যায়: pure function, total-order sort `(price, departs, id)` যাতে সমান দামের ফ্লাইটের ক্রম কখনো না বদলায়, `uuid4()`-এর বদলে input থেকে derived booking ref (`BK-BG101-RAHIM`), in-memory state যেটা test শেষে assert করা যায়।

LLM শুধু "mostly reproducible"। Ollama-র default temperature **0.8** — set না করলে agent random sample করে। কিন্তু `temperature=0`-ও guarantee না: Thinking Machines দেখিয়েছে একই prompt-এ ১০০০ বার temperature-0 চালিয়ে **৮০টা আলাদা output** — কারণ floating-point non-associativity + server load-এ batch size বদলানো। একক local Ollama-তে batch variance কম, কিন্তু Ollama version, quantization, CPU/GPU, `num_ctx` বদলালে output বদলাতে পারে। Agent-এ একটা token-এর পার্থক্য step 2-এ observation বদলায়, step 3-এ trajectory শাখা বদলায়। তাই: **unit test-এ LLM সবসময় mock; live behaviour statistically মাপুন।**

### 3.7 Trust boundary — validation, error as observation

প্রতিটা tool call untrusted input। Executor:

1. নাম registry dict-এ খোঁজে — model-এর string-এ কখনো `getattr`/`eval` না; না পেলে valid নাম + `difflib` "did you mean"।
2. Arguments string হলে JSON decode (text mode, OpenAI-style)।
3. Pydantic validate — lax mode `"6600"`-কে `6600` বানায়, `extra="forbid"` বানানো field ধরে, validator `"dac"`-কে `"DAC"` করে আর `"5 Nov"` date reject করে।
4. ব্যর্থ হলে **exception না** — `{"ok": false, "error": "invalid_arguments", "details": [...], "hint": "...ask the user if unsure"}`।

Temperature 0-এ একই prompt আবার পাঠালে model সম্ভবত একই ভুল call করবে। Error observation-ই সেই **নতুন তথ্য** যা পরের output বদলায়।

### 3.8 Human-in-the-loop — code-এ, prompt-এ না

`book_flight` side-effecting আর irreversible। BFCL দেখেছে সেরা model-ও প্রায়ই "fail to explore the current state before performing actions"। তাই:

* Executor book করার আগে injected `approve(details)` ডাকে — CLI-তে `input()`, UI-তে Approve বাটন, test-এ `lambda d: True`।
* Tool নিজের precondition নিজে check করে: flight আছে, seat আছে, একই passenger আগে বুক হয়নি (idempotent)।
* Prompt-এ "আগে confirm করো" বলা যায়, কিন্তু **enforce করতে পারে শুধু code**।

### 3.9 llama3.1 template-এর রহস্য — আমাদের project-এর সবচেয়ে বড় শিক্ষা

Live eval-এ native mode প্রথমে "I will book BG101 for Rahim" লিখে থেমে যেত, `book_flight` ডাকত না। `ollama show llama3.1:8b --template` দেখে কারণ পাওয়া গেল:

1. Tool list দেখানো হয় **শুধু শেষ user message-এর পাশে।** Tool result আসার পর শেষ message আর user না — model তার tools দেখতেই পায় না।
2. Tool দেখানো হলে template বলে **"respond with a JSON for a function call"** — model তখন plain text-এ থামতে পারে না।

শুধু "continue" nudge দিলে model অসীম tool ডাকে (eval 0/8)। সমাধান — smolagents-এর `ToolCallingAgent`-এর মতো একটা **`final_answer` tool**, আর প্রতিটা observation-এর পর একটা nudge যেটা **মূল request** আর **already-done তালিকা** দেয় (template nudge-কে `Question:` হিসেবে দেখায়, তাই request না দিলে model সেটাকেই প্রশ্ন ভাবে)। কয়েকটা prompt rule ("শুধু চাইলে বুক করো", "date বানিয়ো না") আর এই workaround মিলিয়ে native mode 4/8 থেকে **6/8**-এ উঠেছে। এটাই agent engineering — framework-এর ভেতরে লুকিয়ে থাকা এই কাজগুলোই আসলে agent চালায়।

---

## 4. Details & Valid Points

### 4.1 Evaluation — তিন স্তর

| স্তর | কী মাপে | উদাহরণ |
| --- | --- | --- |
| **Outcome** | Run শেষে database state = expected? | τ-bench: শেষ DB state বনাম annotated goal state |
| **Trajectory** | ঠিক tool, ঠিক args, নিষিদ্ধ call নেই | BFCL v3 multi-turn (state + response check); `agentevals` strict/unordered/subset/superset |
| **Reliability** | k বার চালালে **সবগুলো** সফল? (pass^k) | τ-bench: gpt-4o retail pass^8 < 25% |

Bootcamp scale-এ: ১০-২০টা golden task, expected end state সহ; metric — success rate, invalid-call rate, step সংখ্যা, stop-reason distribution, policy violation (না বলে বুক, sold-out বুক)। আমাদের harness-এ approval gate-এ একটা **simulated user** বসে — শুধু request-এ লেখা নামের booking approve করে, তাই বানানো "John Doe" আটকায়।

> **Valid Point:** τ-bench 2.0-এর score 1.x leaderboard-এর সাথে তুলনীয় না (tau2-bench repo, v1.0.1 2026-07-15)। আর llama3.1:8b-এর কোনো published τ/BFCL-multi-turn agent score পাওয়া যায়নি — নিজের eval set-ই সবচেয়ে বিশ্বাসযোগ্য প্রমাণ।

### 4.2 আমাদের Live Eval (llama3.1:8b, ৮টা golden task)

| Mode | Success | Invalid-call rate | উল্লেখযোগ্য failure |
| --- | --- | --- | --- |
| native + final_answer + nudge | **6/8** | 0.14–0.19 | sold-out → loop; নাম ছাড়া "John Doe" বুকের চেষ্টা |
| text (classic ReAct) | 4/8 | 0.48 | শুধু-খোঁজার প্রশ্নে বুক; sold-out-এ অন্য রুট বুক |
| native, শুধু nudge (final_answer ছাড়া) | 0/8 | — | Template-এর জোরে অসীম tool call |

`temperature=0`-এ একই task বারবার প্রায় একই trajectory দেয় — তাই এখানে pass^k ≈ pass^1। Reliability মাপতে একই task-এর ভিন্ন phrasing যোগ করুন।

### 4.3 Context growth — নীরব ঘাতক

প্রতিটা step-এ একটা assistant message + প্রতিটা call-এ একটা tool message যোগ হয়; আমাদের trace-এ `prompt_tokens` প্রতি step-এ ~৩০০ করে বাড়ে (722 → 1049 → 1294)। Ollama-র docs default context নিয়ে নিজেই পরস্পরবিরোধী (FAQ: 4096, Modelfile: 2048) — তাই `num_ctx` সবসময় explicitly দিন, observation compact রাখুন (৫টা ফ্লাইট × ৫টা field), আর trim করলে assistant `tool_calls` message আর তার tool result **জোড়ায়** রাখুন।

### 4.4 ReAct-এর উত্তরসূরি — প্রত্যেকে একটা দুর্বলতা সারায়

| Pattern | Loop shape | কোন ReAct দুর্বলতা সারায় | কখন বাছবেন |
| --- | --- | --- | --- |
| ReAct | Thought → Action → Observation | — (baseline) | পরের ধাপ আগের ফলাফলে নির্ভর |
| Plan-and-Execute | Plan → execute → re-plan | প্রতি ধাপে বড় model call | লম্বা, ভাগ করা যায় এমন task |
| ReWOO | `#E1, #E2` variable-সহ plan → workers → solver | Token খরচ (৫× কম, HotpotQA +৪) | Tool chain আগে থেকে অনুমেয় |
| LLMCompiler | DAG plan → parallel | Serial latency (৩.৬× দ্রুত দাবি) | অনেক স্বাধীন call |
| Reflexion | Trial → মূল্যায়ন → verbal reflection → retry | Attempt-এর মাঝে শেখা নেই (HumanEval 91%) | Verifier আছে (unit test) — Week 14-এর self-correcting coder |
| LATS | ReAct trajectory-র ওপর MCTS | একটাই greedy trajectory | খরচের চেয়ে accuracy জরুরি |
| CodeAct | Python code-ই action | এক step-এ একটাই rigid action (+২০% success) | অনেক tool compose করা, sandbox আছে |

> **Valid Point:** Reflexion আর LATS ReAct-কে **প্রতিস্থাপন করে না, মোড়ায়** — ভেতরে একটা ReAct loop লাগে। Anthropic-এর ভাষায় ReAct/CodeAct হলো "agent", Plan-and-Execute/ReWOO "orchestrator-worker workflow", Reflexion "evaluator-optimizer"। আমাদের travel agent-এর path ছোট আর একমাত্র irreversible action gated — এর কোনো upgrade লাগে না: "find the simplest solution possible"।

### 4.5 ২০২৬-এর framework API — পুরনো tutorial সাবধান

* LangGraph-এর `create_react_agent` **v1-এ deprecated** → `langchain.agents.create_agent(model, tools, system_prompt, middleware=...)`; custom state schema এখন শুধু `TypedDict`।
* OpenAI Agents SDK: output final হয় যখন "text output... and there are no tool calls"; `tool_use_behavior` default `"run_llm_again"` = classic ReAct।
* smolagents: `ToolCallingAgent` (JSON call, নিরাপদ) বনাম `CodeAgent` (Python action, sandbox লাগে: e2b/docker)।

---

## 5. Related Analogy: The Detective's Notebook

> একজন detective (LLM) একটা কেস সমাধান করছেন, পকেটে একটা নোটবুক।

| Concept | Detective Analogy |
| --- | --- |
| **Thought** | নোটবুকে লেখা: "দারোয়ান বলেছে রাত ১০টায় গেট বন্ধ ছিল — তাহলে পেছনের দরজা দেখি।" |
| **Action** | পেছনের দরজায় গিয়ে দেখা — বাস্তব জগতে একটা কাজ। |
| **Observation** | "দরজার তালা ভাঙা" — যা পাওয়া গেল, অনুমান না। |
| **Hallucination (CoT-এর সমস্যা)** | Desk থেকে না উঠে গল্প বানিয়ে কেস "সমাধান" করা। |
| **Reasoning loop (ReAct-এর সমস্যা)** | একই দারোয়ানকে একই প্রশ্ন বারবার জিজ্ঞেস করা। |
| **Repeat guard** | পার্টনার বলে: "এটা তো আগেই জিজ্ঞেস করেছ, উত্তর ছিল ১০টা।" |
| **max_steps / deadline** | Chief বলেন: "৪৮ ঘণ্টার মধ্যে রিপোর্ট দাও।" |
| **Approval gate** | গ্রেপ্তারের আগে magistrate-এর warrant — detective নিজে যতই নিশ্চিত হোক। |
| **Deterministic fallback** | সময় শেষ হলে যা প্রমাণ পাওয়া গেছে তাই নিয়ে সৎ রিপোর্ট — "কেস অসমাপ্ত, এই সূত্রগুলো পেয়েছি।" |

---

## 6. Common Pitfalls

| Pitfall | Guard |
| --- | --- |
| Step cap নেই | `MAX_STEPS` + typed `max_steps` exit |
| একই search ৮ বার | `(name, sorted-JSON args)` fingerprint; reminder, তারপর stop |
| Tool exception ছোড়ে | Executor `{"ok": false, ...}` observation দেয় |
| Arguments বিশ্বাস করা | Registry + Pydantic (`extra="forbid"`, validator, pattern) |
| Assistant `tool_calls` message ছাড়া tool result | আগে assistant message; trim করলে জোড়ায় |
| Default `num_ctx` + verbose tool JSON | `num_ctx` set, compact observation, `prompt_eval_count` দেখুন |
| Temperature ভুলে যাওয়া (default 0.8) | `options={"temperature": 0, "seed": 42}` |
| Tool-এ `now()`, `uuid4()`, unstable sort | Injected clock, derived ID, total-order sort |
| Side effect শুধু prompt দিয়ে gated | Code-level `approve()` + precondition + idempotency |
| Text ReAct-এ stop sequence নেই | `stop=["\nObservation:"]` |
| শুধু final answer grade করা | End-state + trajectory + pass^k |
| Unit test-এ live LLM | Scripted fake LLM |
| Model-এর template না দেখা | `ollama show <model> --template` — tool কোথায়, কীভাবে দেখানো হয় |

---

## 7. Interview Q&A

1. **ReAct loop কী, কখন থামে?** Reasoning আর tool action পালাক্রমে; প্রতিটা tool result observation হিসেবে পরের call-কে প্রভাবিত করে। থামে যখন model tool ছাড়া উত্তর দেয় — আর hard guard: step cap, repeat detector, deadline, error budget।
2. **ReAct কিছু task-এ CoT-কে হারায়, কিছুতে না — কেন?** Grounding hallucination মুছে দেয় (ReAct failure-এর 0% বনাম CoT-এর 56%), কিন্তু reasoning loop আর খারাপ search-এ আটকায়। Interactive task-এ বড় জয় (ALFWorld 71% বনাম 45%), QA-তে সামান্য — সেখানে CoT-SC fallback hybrid সেরা।
3. **Prompt-based ReAct নাকি native tool calling?** Model support করলে native — args dict আসে, observation নকল করা যায় না। Text ReAct যেকোনো model-এ চলে, thought স্পষ্ট দেখায়, কিন্তু tolerant parser, parse-error observation আর `stop=["\nObservation:"]` লাগে।
4. **আটকে যাওয়া agent কীভাবে ধরবেন?** প্রতিটা call fingerprint `(tool, canonical JSON args)`; প্রথম repeat-এ আগের result মনে করিয়ে দিন, পরেরটায় fallback দিয়ে থামুন; সাথে total step cap।
5. **Temperature 0 + fixed seed = deterministic agent?** না। Floating-point non-associativity + load-নির্ভর batch size (১০০০ run-এ ৮০টা আলাদা output), version/quantization drift, আর step-এ step-এ divergence বাড়ে। Tool deterministic করুন, test-এ LLM mock করুন, live-এ pass^k মাপুন।
6. **Model `book_flight(passengers="two")` পাঠালে?** Pydantic `int_parsing`-এ ব্যর্থ; executor field-এর নাম সহ structured error দেয়; model পরের turn-এ `2` পাঠায়। পরপর ৩ বার ব্যর্থ হলে deterministic fallback।
7. **Hallucinated tool name?** Registry dict-এ lookup, কখনো `getattr`/`eval` না; `unknown_tool` + valid নাম + did-you-mean; error budget-এ গোনা।
8. **ভুল ফ্লাইট বুক হওয়া কীভাবে আটকাবেন?** Side-effect tool-এর আগে code-এ human approval; tool নিজের precondition check করে (model আগে search করেছে ধরে নেয় না); idempotent booking; schema-তে pattern/enum।
9. **এই agent কীভাবে evaluate করবেন?** Outcome = end state বনাম goal (τ-bench); trajectory match (strict/unordered/subset/superset); reliability = pass^k; সাথে invalid-call rate, step, stop-reason distribution।
10. **Demo-তে চলে, লম্বা session-এ local model-এ ভাঙে — কেন?** Context overflow — Ollama-র default window ছোট, প্রতিটা observation prompt ফোলায়। `num_ctx` set, compact observation, `prompt_eval_count` monitor, জোড়ায় trim।
11. **LLM ছাড়া agent unit-test কীভাবে?** LLM-কে callable হিসেবে inject; scripted fake Ollama-shaped message replay করে আর input রেকর্ড করে; assert: tool sequence, ফেরত-দেওয়া error observation, final state, stop reason।
12. **কখন Plan-and-Execute/ReWOO/Reflexion/LATS/CodeAct?** প্রতি-ধাপ LLM খরচ/latency সমস্যা হলে plan-first (ReWOO ৫× কম token); verifier থাকলে Reflexion; accuracy > খরচ হলে LATS; tool compose আর sandbox থাকলে CodeAct। নাহলে সবচেয়ে সরল সমাধান।
13. **LangChain-এর agent API-তে কী বদলেছে?** LangGraph v1-এ `create_react_agent` deprecated → `langchain.agents.create_agent`, middleware সহ; custom state এখন শুধু `TypedDict`।

---

## Achievement: Deterministic Function-Calling Travel Agent (Portfolio Project 11)

**`Class 2 Project/`** — হাতে লেখা ReAct loop দুই mode-এ (native + classic text), typed stop reason, Pydantic trust boundary, code-enforced approval gate, Streamlit chat UI (Approve/Reject বাটন আর trace viewer সহ), golden-task evaluation harness (end state + trajectory + policy + pass^k), ৩৯টা offline test, আর live output সহ exploration notebook। এই ক্লাসের companion notebook **`react_loop_lecture.ipynb`** সম্পূর্ণ self-contained — project folder ছাড়াই চলে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Exit Prediction:** `max_repeat_calls=1` আর `max_consecutive_errors=1` দিয়ে notebook-এর প্রতিটা scripted scenario চালানোর আগে লিখে ফেলুন কোন status আসবে। কয়টা মিলল?
2. **Stop Sequence:** Text mode-এ `stop` sequence সরিয়ে দিন। Model কি নিজের "Observation:" বানায়? বানানো observation আর আসল observation-এর পার্থক্য trace-এ কীভাবে ধরবেন?
3. **New Golden Task:** "Book BS401 for Rahim and Karim" যোগ করুন। Expected end state কী? Agent কি দুটো আলাদা `book_flight` call করে — parallel না sequential?
4. **Template Detective:** `qwen3:8b` pull করে `ollama show qwen3:8b --template` দেখুন। Tool list কোথায় বসে? `nudge_after_tools=False` দিয়ে qwen3-এ eval চালান — nudge কি এখনো দরকার?
5. **Paraphrase pass^k:** প্রতিটা golden task-এর ৩টা ভিন্ন phrasing লিখে evaluation চালান। Success rate কি ৬/৮-এর আশেপাশে থাকে, নাকি phrasing বদলালে ভেঙে পড়ে?

---

## References

* Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023) — https://arxiv.org/abs/2210.03629 · HTML — https://arxiv.org/html/2210.03629v3
* Anthropic, "Building effective agents" — https://www.anthropic.com/research/building-effective-agents
* Ollama tool calling — https://docs.ollama.com/capabilities/tool-calling · API — https://docs.ollama.com/api/chat · Modelfile options — https://docs.ollama.com/modelfile · FAQ — https://docs.ollama.com/faq
* Thinking Machines, "Defeating Nondeterminism in LLM Inference" — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
* τ-bench — https://arxiv.org/abs/2406.12045 · τ²-bench — https://arxiv.org/abs/2506.07982 · tau2-bench repo — https://github.com/sierra-research/tau2-bench
* BFCL v3 multi-turn — https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html
* LangChain agentevals — https://github.com/langchain-ai/agentevals · AgentBench — https://arxiv.org/abs/2308.03688
* Pydantic validation — https://pydantic.dev/docs/validation/latest/concepts/validation_decorator/
* LangGraph v1 release notes — https://docs.langchain.com/oss/python/releases/langgraph-v1 · LangChain agents — https://docs.langchain.com/oss/python/langchain/agents
* OpenAI Agents SDK, running agents — https://openai.github.io/openai-agents-python/running_agents/
* smolagents — https://huggingface.co/docs/smolagents/guided_tour
* Plan-and-Solve — https://arxiv.org/abs/2305.04091 · ReWOO — https://arxiv.org/abs/2305.18323 · Reflexion — https://arxiv.org/abs/2303.11366 · LATS — https://arxiv.org/abs/2310.04406 · CodeAct — https://arxiv.org/abs/2402.01030
* LangChain, "Planning agents" — https://www.langchain.com/blog/planning-agents
