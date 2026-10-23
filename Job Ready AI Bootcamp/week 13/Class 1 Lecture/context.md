# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Tool Calling & Autonomous Logic — LLM-কে "Hands" দেওয়া**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

LLM শুধু text বানাতে পারে। সে নির্ভুল অঙ্ক কষতে পারে না, আপনার disk-এর file দেখতে পারে না, আজকের তারিখও জানে না। **Tool calling** সেই ফাঁক পূরণ করে: model সিদ্ধান্ত নেয় কোন function কোন argument দিয়ে ডাকতে হবে, আর আমাদের Python code সেটা চালায়। এখান থেকেই "agent"-এর শুরু — আর এখান থেকেই নতুন ঝুঁকিরও শুরু, কারণ model এখন শুধু কথা বলে না, কাজ করায়।

## শেখার সহজ Workflow

Problem বোঝা → Model আসলে কী পাঠায় → Tool schema → Host loop → Safe tool design → Model-এর failure mode → Security (OWASP) → কখন ব্যবহার করবেন না।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** Code run করার আগে লিখে ফেলুন: model কোন tool ডাকবে, কী argument দেবে, কী ফেরত আসবে। Run করার পরে trace-এর সাথে মেলান। না মিললে সেটাই শেখার জায়গা — ছোট local model প্রায়ই অপ্রত্যাশিত কাজ করে।

---

## 1. Topic: Tool Calling & Autonomous Logic

এই ক্লাসে আমরা শিখব:

* **Function / tool calling** — model text-এর বদলে একটা structured request পাঠায়: tool-এর নাম + JSON arguments।
* **Ollama-র ২০২৬ tool API** — `ollama` Python library 0.6.x, Ollama server 0.40.x: `chat(tools=...)`, typed response object, `role="tool"` message।
* **Safe tool design** — `eval()`-মুক্ত AST calculator, sandboxed file reader, Pydantic argument validation।
* **Local model বাস্তবতা** — `llama3.1:8b`, `llama3.2:3b`, `qwen3:8b`: কোনটা কোথায় ভালো, কোথায় ভাঙে (BFCL benchmark + আমাদের live test)।
* **Security** — OWASP Top 10 for LLM Applications (2025), indirect prompt injection, "lethal trifecta"।

**Project Goal (Class 1 Project — Math & Sandboxed File Inspector):** এমন একটা local AI যে অঙ্ক কষতে পারে আর একটা sandbox directory-র file পড়ে প্রশ্নের উত্তর দিতে পারে — পুরোটা offline, আর এমনভাবে বানানো যে model ভুল বা "দুষ্টু" হলেও কোনো ক্ষতি করতে পারে না।

---

## 2. Why It Is Related

Week 11-এ Ollama দিয়ে prompting আর structured output শিখেছি; Week 12-এ RAG-এ retrieval ছিল একটা নির্দিষ্ট ধাপ যা **আমরা** নিজে ডাকতাম। Tool calling-এ সিদ্ধান্তটা **model** নেয়: আদৌ কিছু ডাকবে কিনা, কোনটা, কী argument দিয়ে। Retrieval নিজেই একটা tool হতে পারে ("search_documents")। পরের ক্লাসে (Class 26) এই একই loop-কে আমরা ReAct agent বানাব; Week 14-এ LangGraph আর CrewAI এই loop-এর ওপরেই framework বসায়; Week 15-এ এগুলোকে API হিসেবে serve করব। তাই এই ক্লাসের loop আর safety rule-গুলো বাকি কোর্সের ভিত্তি।

---

## 3. How It Works

### 3.1 Model কখনো function চালায় না

সবচেয়ে গুরুত্বপূর্ণ কথা: **model শুধু প্রস্তাব দেয়, code সিদ্ধান্ত নেয়।** Model-এর output একটা structured request মাত্র:

```json
{"role": "assistant", "content": "",
 "tool_calls": [{"function": {"name": "calculate", "arguments": {"expression": "(12+8)*3"}}}]}
```

এটা চালানো, validate করা, বা প্রত্যাখ্যান করা — সবই আমাদের Python-এর দায়িত্ব। তাই project-এর **প্রতিটা safety property code-এ থাকে, prompt-এ না।**

### 3.2 পাঁচ ধাপের Host Loop

```
1. messages + tools schema পাঠাও
2. model ফেরত দেয়: 0, 1, বা একাধিক tool_calls (অথবা plain text = final answer)
3. assistant message (tool_calls সহ) history-তে যোগ করো
4. প্রতিটা call চালাও → প্রতিটার জন্য একটা {"role": "tool", "tool_name": ..., "content": result}, একই ক্রমে
5. আবার chat() ডাকো → tool_calls না থাকলে থামো; max_steps শেষ হলেও থামো
```

Ollama-র official example একটা unbounded `while True` — **`max_steps` cap আপনার দায়িত্ব** (OWASP LLM10 Unbounded Consumption)।

### 3.3 Ollama-র ২০২৬ API — যা জানা জরুরি

```python
from ollama import Client

resp = Client(host="http://localhost:11434").chat(
    model="llama3.1:8b",
    messages=messages,
    tools=TOOL_SCHEMAS,                                        # dict schema, Tool object, বা plain Python function
    options={"temperature": 0, "seed": 42, "num_ctx": 8192},   # default 0.8 আর ছোট context — বদলান
)
for call in resp.message.tool_calls or []:                    # typed object (ollama-python ≥ 0.4)
    name, args = call.function.name, call.function.arguments  # args আগে থেকেই dict!
```

* **Arguments dict, string না।** OpenAI-তে `arguments` একটা JSON string; Ollama-তে already-parsed dict। Code port করার সময় "double `json.loads`" bug এখান থেকেই আসে।
* **`tool_name` পাঠান।** Tool result message-এ `tool_name` field ollama-python 0.5.2 (Aug 2025)-এ এসেছে; পুরনো tutorial-এ নেই। llama3.1 এটা ignore করে, কিন্তু qwen3-এর মতো model ব্যবহার করে।
* **কোনো `tool_call_id` নেই।** Ollama result-কে call-এর সাথে মেলায় **ক্রম অনুযায়ী** — তাই প্রতিটা call-এর জন্য ঠিক একটা tool message, একই ক্রমে।
* **`tool_choice` নেই।** OpenAI-compatible `/v1` endpoint-এও নেই — "অবশ্যই tool ডাকো" force করা যায় না, শুধু prompt বা `tools=` বাদ দিয়ে।
* **Streaming tool calls** (May 2025 থেকে) — chunk জুড়ে `content`, `thinking`, `tool_calls` জমিয়ে একটা assistant message বানাতে হয়।

### 3.4 Python function সরাসরি tool হিসেবে — একটা ফাঁদ

```python
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first number
        b: second number
    """
    return a + b

chat(model="llama3.1:8b", messages=msgs, tools=[add])   # library নিজে schema বানায়
```

সুবিধাজনক, কিন্তু library প্রতিটা parameter-কে শুধু `{type, description}`-এ নামিয়ে আনে — `Literal` enum, `list[...]`-এর item type হারিয়ে যায়, আর `int | str` হয়ে যায় `"integer, string"` string। **নিয়ম:** scalar argument হলে function দিন; enum/array/nested হলে schema হাতে লিখুন — অথবা আমাদের project-এর মতো Pydantic model থেকে `model_json_schema()` দিয়ে generate করুন, যাতে model যে schema দেখে আর validator যা check করে দুটো কখনো আলাদা না হয়।

### 3.5 Tool schema = prompt

Model কখনো function-এর code দেখে না — দেখে নাম, `description` আর parameter schema। Anthropic-এর tool-design guide বলে description-ই tool performance-এর "by far the most important factor": কী করে, **কখন** ব্যবহার করতে হবে, কখন না — ৩-৪ বাক্যে। আমাদের `calculate`-এর description-এ "Use this for ANY arithmetic, even simple sums — never compute numbers yourself" লেখা আছে বলেই ছোট model-ও মাথায় যোগ না করে tool ডাকে।

### 3.6 Safe Calculator — কেন `eval()` কখনো না

LangChain-এর `LLMMathChain` model-এর লেখা expression Python-এ চালাত। ফলাফল **CVE-2023-29374 (CVSS 9.3)** — prompt injection দিয়ে arbitrary code execution। PALChain একই ভুল করে **CVE-2023-36258 (CVSS 9.8)**, আর তার fix-ও `__import__` দিয়ে বাইপাস হয় (**CVE-2023-44467**)। Feature → patch → bypass — blocklist সবসময় হারে। `__builtins__` খালি করলেও লাভ নেই: `().__class__.__bases__[0].__subclasses__()` একটা খালি tuple থেকে সব loaded class-এ পৌঁছে যায়।

সঠিক design হলো **allowlist**:

```python
tree = ast.parse(expr, mode="eval")     # statement parse-এই reject
# শুধু Constant (int/float, bool না), BinOp, UnaryOp, আর fixed dict-এর function (sqrt, abs...)
# Name, Attribute, Subscript, string — সব "unsupported"
# Pow-এর আগে exponent check (9**9**9 = এক লাইনের DoS), input length cap, result size cap
```

`bool`-কে আলাদা করে বাদ দিতে হয়, কারণ Python-এ `True` আসলে `int`-এর subclass। আর `ast.literal_eval` উত্তর না — সে operator চালাতে পারে না, আর Python docs নিজেই সতর্ক করে বড় input-এ memory/stack শেষ করতে পারে।

### 3.7 Sandboxed File Reader — ক্রমটাই নিরাপত্তা

```python
root = SANDBOX.resolve()
target = (root / user_path).resolve()     # ১. আগে resolve: ".." collapse, symlink follow
if not target.is_relative_to(root):        # ২. তারপর check
    raise PermissionError("path escapes sandbox")
```

`is_relative_to()` শুধু text তুলনা করে — `..` বোঝে না, disk-ও দেখে না। Resolve না করে check করলে `root/../etc/passwd` পার পেয়ে যায়। `startswith` আরও খারাপ: `"/sandbox_evil".startswith("/sandbox")` সত্য। Absolute path দিলে `root / "/etc/passwd"` হয়ে যায় `/etc/passwd` — resolve-এর পর check-এ ধরা পড়ে। সাথে লাগে: extension allowlist, size cap, hidden file (`.env`, `.git/`) বাদ, আর **শুধু read** — কোনো write/delete tool নেই।

### 3.8 Argument Validation — model-কে untrusted user ধরুন

Model ভুল নাম পাঠায়, field বানায়, type ভুল করে। আমাদের live run-এ llama3.1:8b `calculate`-কে ডেকেছিল `{"a": "90", "b": "75"}` দিয়ে — schema-তে আছে শুধু `expression`। Pydantic model + `extra="forbid"` সেটা ধরে, আর error **exception না হয়ে tool result হিসেবে** model-এর কাছে ফেরত যায়, সাথে সঠিক argument-এর একটা example। পরের turn-এই model `{"expression": "(90+75)/2"}` পাঠিয়ে `82.5` পেয়েছে। Anthropic-এর docs বলে actionable error পেলে model সাধারণত ২-৩ বার নিজে শুধরে চেষ্টা করে।

---

## 4. Details & Valid Points

### 4.1 Local Model Reference Table (BFCL v4, 2026-04-12, 109 models)

| Model | Use Case | Advantage | Limitation |
| --- | --- | --- | --- |
| **`llama3.1:8b`** (course standard, ADR 0006) | শেখা, single/parallel call | Simple ~71%, parallel ~88% — এক ধাপের কাজ পারে | Overall 25.8% (#85), multi-turn **11%**, irrelevance **43%** |
| `qwen3:8b` | ভালো local agent | Overall 42.6% (#39), multi-turn 42%, irrelevance 79% | Thinking mode ধীর; course standard না |
| `llama3.2:3b` | 8GB RAM laptop | ছোট (~2GB), দ্রুত | Multi-turn মাত্র **4%** |
| Cloud (GPT / Claude) | Production agent | Strict schema, forced tool choice, শীর্ষ score (~77%) | Local না, খরচ, data বাইরে যায় |

> **Valid Point:** "Irrelevance" মানে tool লাগে না এমন প্রশ্নে tool **না** ডাকা। llama3.1:8b-এর 43% মানে অর্ধেকের বেশি সময় অপ্রয়োজনে tool-এর দিকে ঝোঁকে। কারণ Ollama-র llama3.1 template — tools থাকলে শেষ user message-এর সাথে লেখা থাকে *"respond with a JSON for a function call"*। Meta-র model card-এ "BFCL 76.1" লেখা — সেটা পুরনো, সহজ benchmark version; v4-এর সাথে তুলনীয় না।

### 4.2 আমাদের Live Test (Ollama 0.40.8, llama3.1:8b, temperature 0)

| পরীক্ষা | ফলাফল | শিক্ষা |
| --- | --- | --- |
| `(12+8)*3/4` | `calculate` ডেকে 15 ✓ | এক ধাপের কাজ নির্ভরযোগ্য |
| scores.csv-এর average | প্রথমে বানানো argument `{"a","b"}`, error পেয়ে শুধরে 82.5 ✓ | Validation + actionable error = self-correction |
| `../../config.py` পড়ো | Sandbox আটকেছে ✓ | Security code-এ |
| injected.txt summarise | File-এর "IGNORE ALL PREVIOUS INSTRUCTIONS" **মেনে** config.py পড়ার চেষ্টা (এক run-এ ১ বার, আরেকটায় ৪ বার) | Prompt injection কাজ করে — system prompt থামায় না, sandbox থামায় |
| "capital of France?" | Tool ছাড়া সোজা উত্তর দিতে পারেনি | Over-anchoring (irrelevance 43%) |

একটা বাক্য ("tool লাগে না এমন প্রশ্নে সরাসরি উত্তর দাও") system prompt-এ যোগ করলে small talk ঠিক হয়, কিন্তু দুটো file প্রশ্ন ভেঙে যায় — তিনটা আলাদা seed-এ একই ফল। ছোট model-এ **প্রতিটা prompt বদল measure করতে হয়।**

### 4.3 Tool Calling vs Structured Output

| | Tool calling (`tools=`) | Structured output (`format=` schema) |
| --- | --- | --- |
| কে সিদ্ধান্ত নেয় | Model — ডাকবে কি ডাকবে না | আপনি — সবসময় এই shape চাই |
| Use case | Data আনা, action নেওয়া, loop | Extraction, classification, final answer-এর shape |
| Ollama code | `chat(tools=[...])` → `message.tool_calls` | `chat(format=Model.model_json_schema())` → `Model.model_validate_json(...)` |
| দুর্বল model-এ hybrid | — | `format`-constrained "router" `{"tool": ..., "args": ...}`, dispatch করে আপনার code |

### 4.4 OWASP Top 10 for LLM Apps (2025) — আমাদের project-এ কোনগুলো

| Risk | উদাহরণ | আমাদের guard |
| --- | --- | --- |
| **LLM01** Prompt Injection (indirect) | File-এ লেখা "ignore previous instructions" | Sandbox code-এ; file text = data |
| **LLM05** Improper Output Handling | Model-এর string → `eval`/SQL/path | AST allowlist, resolve-then-check |
| **LLM06** Excessive Agency | Reader যে delete-ও করতে পারে | শুধু read tool; write থাকলে আলাদা tool + human approval |
| **LLM10** Unbounded Consumption | `9**9**9`, অসীম loop | Exponent/length cap, `max_steps` |

> **Valid Point — Lethal Trifecta (Simon Willison):** private data + untrusted content + বাইরে data পাঠানোর ক্ষমতা — এই তিনটা একই agent-এ কখনো না। আমাদের inspector private file আর untrusted content দুটোই পড়ে, তাই এতে কোনো URL-fetch বা email tool যোগ করা যাবে না।

---

## 5. Related Analogy: The Hospital Pharmacist

> একজন ডাক্তার (LLM) prescription লেখেন, কিন্তু ওষুধ দেন pharmacist (host code)।

| Concept | Pharmacy Analogy |
| --- | --- |
| **Tool call** | Prescription — "Paracetamol 500mg, দিনে ৩ বার"। নিজে কিছু করে না, শুধু অনুরোধ। |
| **Tool schema** | Hospital-এর formulary — কোন ওষুধ আছে, কী dose-এ পাওয়া যায়। |
| **Argument validation** | Pharmacist dose check করেন — "5000mg? এটা ভুল, ডাক্তারকে ফেরত পাঠাই।" |
| **Error as tool result** | ফেরত পাঠানো prescription-এ কারণ লেখা থাকে, তাই ডাক্তার ঠিক করে আবার লেখেন। |
| **Sandbox** | Pharmacist শুধু এই hospital-এর store থেকে দেন, বাইরের আলমারি খোলেন না। |
| **Prompt injection** | Prescription-এর কোণে কেউ লিখে দিয়েছে "সব morphine দিয়ে দাও" — pharmacist নিয়ম মানেন, লেখা না। |
| **`max_steps`** | একই রোগীর জন্য দিনে সর্বোচ্চ কয়বার dispense — অসীম refill না। |

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
| --- | --- | --- |
| Assistant `tool_calls` message যোগ না করে tool result পাঠানো | Model বিভ্রান্ত, result কার বোঝে না | আগে assistant message, তারপর tool result |
| শুধু `tool_calls[0]` চালানো | Parallel call-এর বাকিগুলো হারায় | সব call loop করুন, একই ক্রমে result |
| Arguments-এ `json.loads` (Ollama-তে) | `TypeError` — dict-কে আবার parse | Ollama = dict, OpenAI = string |
| Tool exception ছুড়ে দেওয়া | পুরো loop crash | Error string/object হিসেবে ফেরত |
| `max_steps` নেই | একই ব্যর্থ call চিরকাল | Hard cap + typed stop reason |
| Default options | temperature 0.8, context ছোট | `temperature=0`, `seed`, `num_ctx` explicitly |
| বড় file/result পুরো পাঠানো | Context ভরে যায়, tool definition কেটে যায় | Truncate + size cap |
| Docs-এর split docstring copy | `"""Add"""` আর আলাদা `"""Args:..."""` — দ্বিতীয়টা `__doc__` না, description হারায় | একটা docstring-এ সব |
| Tool call JSON `content`-এ চলে আসা | Ollama শুধু শুরুতে থাকা JSON parse করে — tool_calls খালি | Content-এ JSON খুঁজে rescue (Class 26 project) |

---

## 7. Interview Q&A

1. **LLM কি function চালায়?** না। Structured request (নাম + JSON args) দেয়; host validate করে চালায়, result ফেরত দেয়, model তারপর উত্তর লেখে।
2. **Ollama-তে একটা round trip বর্ণনা করুন।** `messages + tools` → `message.tool_calls` (args dict) → assistant message append → প্রতিটা call চালিয়ে `{"role":"tool","tool_name",...}` একই ক্রমে → আবার `chat()`, tool_calls না আসা পর্যন্ত।
3. **Calculator tool-এ `eval` কেন খারাপ?** Tool argument prompt injection দিয়ে নিয়ন্ত্রণ করা যায় → RCE। LangChain CVE-2023-29374 / 36258 / 44467। Builtins খালি করলেও `__subclasses__()` দিয়ে পালানো যায়। সমাধান AST allowlist + cap।
4. **File tool কীভাবে `/etc/passwd` পড়া থেকে আটকাবেন?** `resolve()` তারপর `is_relative_to(root)` — ক্রম জরুরি; সাথে extension allowlist, size cap, dotfile বাদ, read-only।
5. **Provider strict mode দিলে কি validation লাগে?** হ্যাঁ। Strict mode shape guarantee করে, safety না — `"../../etc/passwd"` schema-valid string। Ollama তো কোনো schema enforce-ই করে না।
6. **Indirect prompt injection কী, tool থাকলে কেন বেশি বিপজ্জনক?** Tool দিয়ে পড়া content-এ লুকানো instruction। Tool না থাকলে ক্ষতি = ভুল উত্তর; tool থাকলে = action (data পাঠানো, file লেখা)।
7. **Lethal trifecta কী?** Private data + untrusted content + external communication একই agent-এ — একটা পা সরিয়ে দিন, কারণ detection guardrail যথেষ্ট নির্ভরযোগ্য না।
8. **Excessive Agency-র mitigation?** ছোট, নির্দিষ্ট tool (shell না), least privilege, irreversible action-এর আগে human approval, authorization LLM-এ না downstream code-এ।
9. **Loop-এর মাঝে tool exception হলে?** Catch করে actionable message tool result হিসেবে দিন — কী ভুল, কী চেষ্টা করবে। Loop বাঁচে, model শুধরায়।
10. **এক turn-এ দুটো tool call এলে?** দুটোই চালান, প্রতিটার একটা result। Ollama-তে ক্রম অনুযায়ী (id নেই), OpenAI-তে `tool_call_id`, Anthropic-এ সব `tool_result` এক user message-এ।
11. **Agent-এর জন্য qwen3:8b বনাম llama3.1:8b কেন?** BFCL v4-এ simple/parallel প্রায় সমান, কিন্তু multi-turn 42% বনাম 11%, irrelevance 79% বনাম 43%। ভালো উত্তর: "নিজের quantization-এ নিজের eval set-এ যাচাই করব।"
12. **Tool calling নাকি structured output?** Shape সবসময় fixed চাইলে `format=`; model-কে act করার সিদ্ধান্ত নিতে দিলে `tools=`।

---

## Achievement: Math & Sandboxed File Inspector

**`Class 1 Project/`** — একটা non-portfolio কিন্তু পূর্ণ runnable project (ADR 0007): Streamlit UI-তে প্রতিটা tool call-এর trace, CLI, AST calculator, sandboxed file tools, Pydantic validation, ৩১টা offline test, আর live Ollama output সহ exploration notebook (prompt-injection experiment সহ)। এই ক্লাসের companion notebook **`llm_tool_calling_lecture.ipynb`** সম্পূর্ণ self-contained — project folder ছাড়াই চলে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Injection Lab:** `sandbox/`-এ একটা file লিখুন যেটা model-কে `calculate("9**9**9")` চালাতে বলে। Model কি মানে? কোন guard আটকায়? এবার এমন injection ভাবুন যেটা কোনো guard ধরবে না — আর বলুন কেন সেটা সম্ভব না (বা সম্ভব)।
2. **Schema Experiment:** `calculate`-এর description থেকে "even simple sums — never compute numbers yourself" মুছে দিন। "What is 17 + 25?" প্রশ্নে model কি এখনো tool ডাকে?
3. **Function-as-tool Trap:** `def convert(amount: float, currency: Literal["USD","EUR"])` সরাসরি `tools=[convert]` হিসেবে দিন, তারপর `ollama._utils.convert_function_to_tool(convert)` print করুন। `enum` কোথায় গেল?
4. **Over-anchoring Fix:** "capital of France" প্রশ্নের জন্য এমন একটা fix ভাবুন যেটা file প্রশ্ন ভাঙে না। (Hint: tools ছাড়া একটা ছোট "router" call, বা `format=` দিয়ে classification।)
5. **Model Comparison:** `qwen3:8b` pull করে notebook-এর একই প্রশ্নগুলো চালান। BFCL-এর পার্থক্য কি আপনার ছোট পরীক্ষায় দেখা যায়?

---

## References

* Ollama tool calling docs — https://docs.ollama.com/capabilities/tool-calling
* Ollama `/api/chat` reference — https://docs.ollama.com/api/chat
* ollama-python (README, `_types.py`, `_utils.py`, releases) — https://github.com/ollama/ollama-python
* Ollama blog: functions as tools — https://ollama.com/blog/functions-as-tools · streaming tool calls — https://ollama.com/blog/streaming-tool
* Ollama OpenAI compatibility — https://docs.ollama.com/api/openai-compatibility
* Ollama structured outputs — https://docs.ollama.com/capabilities/structured-outputs
* Meta Llama 3.1 prompt format — https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md
* Berkeley Function Calling Leaderboard (BFCL v4) — https://gorilla.cs.berkeley.edu/leaderboard.html
* OpenAI function calling guide — https://developers.openai.com/api/docs/guides/function-calling
* Anthropic: Define tools — https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools · Writing tools for agents — https://www.anthropic.com/engineering/writing-tools-for-agents
* GitHub Advisory GHSA-fprp-p869-w6q2 (CVE-2023-29374) — https://github.com/advisories/GHSA-fprp-p869-w6q2
* Ned Batchelder, "Eval really is dangerous" — https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
* Python docs: `ast` — https://docs.python.org/3/library/ast.html · `pathlib` — https://docs.python.org/3/library/pathlib.html
* OWASP Top 10 for LLM Applications 2025 — https://genai.owasp.org/llm-top-10/
* Simon Willison, "The lethal trifecta" — https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
