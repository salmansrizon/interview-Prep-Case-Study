# 🛠️ Tool-Calling Agent: Math & Sandboxed File Inspector

## সহজ ভাষায় Project Overview

এই project-এ একটা local LLM-কে "hands" দেওয়া হয়েছে। Model নিজে অঙ্ক কষে না, file-ও খোলে না — সে শুধু একটা structured request পাঠায় ("`calculate` ডাকো, `expression="(12+8)*3"` দিয়ে")। Python সেই request **validate করে**, চালায়, result ফেরত দেয়, আর model সেটা দেখে উত্তর লেখে। Model চলে Ollama-তে (`llama3.1:8b`), কোনো cloud API লাগে না।

### কোন Problem Solve করে?

LLM নির্ভুল অঙ্ক কষতে পারে না, আপনার disk-এর file দেখতে পারে না। Tool calling সেই ফাঁক পূরণ করে — কিন্তু model-কে "hands" দিলে ঝুঁকিও আসে: `eval()` calculator মানে remote code execution, অসতর্ক file tool মানে `/etc/passwd` পড়ে ফেলা। এই project দেখায় কীভাবে **reliability আর security একই engineering layer-এ** (host code) সমাধান হয়।

### কেন এই Approach ভালো?

- **Model শুধু প্রস্তাব দেয়, code সিদ্ধান্ত নেয়:** সব safety Python-এ, prompt-এ না।
- **Testable:** loop-এ LLM একটা injectable function, তাই ৩১টা test Ollama ছাড়াই চলে।
- **Explainable:** UI প্রতিটা tool call-এর trace দেখায় — কোন tool, কী argument, কী result, কত ms।
- **Honest:** live run-এ দেখা failure (invented argument, prompt injection) README আর notebook-এ লেখা আছে, লুকানো না।

> **Run করার নিয়ম:** আগে virtual environment তৈরি করে dependency install করুন, `ollama pull llama3.1:8b` করুন, তারপর নিচের Quick Start follow করুন।

---

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.40+-red.svg)
![Ollama](https://img.shields.io/badge/ollama-llama3.1%3A8b-black.svg)
![Pydantic](https://img.shields.io/badge/pydantic-v2-e92063.svg)
![Tests](https://img.shields.io/badge/tests-31%20passing-green.svg)

## ✨ Features

- 🧮 **AST-allowlist calculator** — `eval()` কখনো না। `__import__`, `().__class__.__subclasses__()`, `9**9**9`, string, bool সব reject; input length, exponent আর result size-এ cap।
- 📂 **Sandboxed read-only file tools** — `resolve()` তারপর `is_relative_to()`: `../`, absolute path, symlink escape, hidden file (`.env`), অননুমোদিত extension আর বিশাল file সব আটকায়।
- ✅ **Pydantic argument validation** — প্রতিটা tool-এর argument model থেকেই JSON schema generate হয় (schema আর validator কখনো আলাদা হয় না)। `extra="forbid"` model-এর বানানো field ধরে, আর error message-এ সঠিক argument-এর example দেয়।
- 🔁 **Correct tool loop** — parallel tool call সব চালায়, assistant message আগে আর tool result পরে (একই ক্রমে), `max_steps` guard।
- 🎛️ **Deterministic settings** — `temperature=0`, `seed=42`, `num_ctx=8192` explicitly (Ollama-র default 0.8 আর ছোট context tool calling-এর জন্য খারাপ)।
- 🖥️ **Streamlit UI + CLI** — UI-তে trace table, model selector, tool schema viewer; CLI দ্রুত terminal test-এর জন্য।
- 🧪 **Offline tests** — scripted fake LLM আর fake `ollama` module (ADR 0006)।

## 🚀 Quick Start

```bash
# 1. Project folder-এ যান
cd "Job Ready AI Bootcamp/week 13/Class 1 Project"

# 2. Virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. One-time model pull (~4.9GB)
ollama pull llama3.1:8b

# 5. Ollama server (চালু না থাকলে)
ollama serve

# 6a. Streamlit UI
streamlit run app.py

# 6b. অথবা CLI
python cli.py "What is (12+8)*3 divided by 4?"
python cli.py "What is the launch code in notes.txt?"
```

Ollama বন্ধ থাকলে app crash করে না — friendly message দেখায়।

### Tests ও notebook

```bash
pytest tests -q                                   # 31 tests, Ollama লাগে না
jupyter notebook notebooks/01_exploration.ipynb   # live Ollama run + prompt-injection experiment
```

## 📁 Project Structure

```
Class 1 Project/
├── app.py                      # Streamlit UI (শুধু src/ ডাকে, ollama import করে না)
├── cli.py                      # Terminal version of the same agent
├── config.py                   # Frozen dataclass config + model reference table
├── requirements.txt
├── sandbox/                    # Agent শুধু এই folder দেখতে পারে
│   ├── notes.txt
│   ├── scores.csv
│   └── injected.txt            # Indirect prompt-injection demo
├── notebooks/
│   └── 01_exploration.ipynb    # Cell-by-cell walkthrough, live Ollama outputs সহ
├── src/
│   ├── tools/
│   │   ├── calculator.py       # AST allowlist evaluator
│   │   ├── filesystem.py       # Sandboxed list_files / read_file
│   │   └── registry.py         # Pydantic arg models -> schemas + validating dispatcher
│   ├── llm/
│   │   └── client.py           # একমাত্র module যেটা `ollama` import করে
│   ├── agent/
│   │   └── loop.py             # Tool loop, max_steps guard, trace
│   └── utils/
│       └── logger.py           # Shared logging
└── tests/
    └── test_pipeline.py        # 31 offline tests
```

## 🔧 Architecture

```
  user question
       │
       ▼
┌──────────────────────────────────────────────┐
│ src/agent/loop.py  run_agent()               │
│ messages = [system, user]                    │
│ for step in 1..max_steps:                    │◄─────────────┐
└──────────────────────┬───────────────────────┘              │
                       ▼                                      │
┌──────────────────────────────────────────────┐              │
│ src/llm/client.py  chat(messages, TOOLS)     │              │
│ Ollama llama3.1:8b, temp=0, seed=42          │              │
└──────────────────────┬───────────────────────┘              │
                       ▼                                      │
              tool_calls in reply?                            │
             no │              │ yes                          │
                ▼              ▼                              │
        final answer   append assistant msg                   │
                       for each call (in order):              │
                ┌──────────────────────────────────────┐      │
                │ src/tools/registry.py  run_tool()     │      │
                │ 1. unknown tool? -> "Error: ..."      │      │
                │ 2. Pydantic validate (extra=forbid)   │      │
                │ 3. calculate | list_files | read_file │      │
                │    (sandbox + AST guards)             │      │
                └──────────────────┬───────────────────┘      │
                                   ▼                          │
                   append {"role":"tool", tool_name, result} ─┘
```

## ⚙️ Configuration

সব setting [`config.py`](config.py)-তে frozen dataclass হিসেবে।

| Setting | Default | কেন |
|---|---|---|
| `llm.model` | `llama3.1:8b` | Course standard (ADR 0006) |
| `llm.temperature` / `seed` | `0.0` / `42` | Tool argument-এ creativity চাই না; run reproducible হয় (তবে bit-exact না) |
| `llm.num_ctx` | `8192` | Ollama-র default context ছোট; history বাড়লে tool definition কেটে যেতে পারে |
| `agent.max_steps` | `5` | সর্বোচ্চ LLM call — infinite loop ও খরচ (OWASP LLM10) আটকায় |
| `safety.max_expression_chars` | `200` | Calculator DoS guard |
| `safety.max_exponent` | `100` | `9**9**9` আটকায় |
| `safety.max_file_chars` / `max_file_bytes` | `4000` / `100000` | Context flood আর বিশাল file আটকায় |
| `safety.allowed_extensions` | `.txt .csv .md .json` | Code/binary file পড়া বন্ধ |

## 🔒 Security Model

| Threat (OWASP LLM Top 10, 2025) | এখানে কীভাবে আটকানো |
|---|---|
| LLM05 Improper Output Handling (`eval` → RCE, CVE-2023-29374) | AST allowlist; কোনো `eval`/`exec` নেই |
| LLM06 Excessive Agency | শুধু read tool; write/delete tool নেই-ই |
| LLM01 Indirect Prompt Injection | Sandbox code-এ enforce করা; prompt শুধু সহায়ক |
| LLM10 Unbounded Consumption | `max_steps`, exponent/length/file-size cap |

## 🧪 Live Findings (llama3.1:8b, Ollama 0.40.8)

এগুলো notebook-এ live output সহ আছে:

- **Invented arguments:** model `calculate`-কে `{"a": "90", "b": "75"}` দিয়ে ডেকেছিল। `extra="forbid"` + example-সহ error message দেওয়ার পর পরের turn-এ সে নিজেই `{"expression": "(90+75)/2"}` পাঠিয়েছে।
- **Prompt injection কাজ করে:** `injected.txt`-এর "IGNORE ALL PREVIOUS INSTRUCTIONS" পড়ে model `../../config.py` পড়ার চেষ্টা করেছে, system prompt থাকা সত্ত্বেও — এক run-এ একবার, আরেক run-এ ৪ বার (তখন `max_steps` থামিয়েছে)। Sandbox প্রতিবার আটকেছে। শিক্ষা: **prompt নিরাপত্তা দেয় না, code দেয়।**
- **System prompt trade-off:** "tool লাগে না এমন প্রশ্নে সরাসরি উত্তর দাও" যোগ করলে small talk ঠিক হয়, কিন্তু file প্রশ্নে tool use ভেঙে যায় — ছোট model-এ প্রতিটা prompt বদল measure করতে হয়।

## ⚠️ Known Limits

- llama3.1:8b BFCL v4-এ multi-turn ~11% — একই call বারবার করলে শুধু `max_steps` থামায়; repeat detection Class 2-এ।
- `resolve()` আর `open()`-এর মাঝে symlink বদলানো (TOCTOU race) সম্ভব — production-এ OS/container level sandbox লাগে।
- Ollama-তে `tool_choice` নেই, তাই "অবশ্যই tool ডাকো" force করা যায় না — শুধু prompt।
