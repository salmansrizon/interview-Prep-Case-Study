# ✈️ Deterministic Function-Calling Travel Agent (Hand-written ReAct Loop)

## সহজ ভাষায় Project Overview

এই project-এ lecture-এর ReAct theory (Thought → Action → Observation) একটা কাজের, conversational travel agent-এ রূপ নিয়েছে। Agent একটা local flight database থেকে ফ্লাইট খোঁজে আর **আপনার approval নিয়ে** simulated booking করে। Loop-টা framework ছাড়া নিজের হাতে লেখা (`src/agent/react_agent.py`), LLM চলে Ollama-তে (`llama3.1:8b`)।

### কোন Problem Solve করে?

একটা 8B local model একা নির্ভরযোগ্য agent না — BFCL v4-এ llama3.1:8b-এর multi-turn accuracy মাত্র ~11%। এই project দেখায় model-এর চারপাশের **code** দিয়ে কীভাবে একটা দুর্বল model-কে কাজের agent বানানো যায়: validated tool call, typed stop reason, repeat guard, code-enforced approval gate, আর একটা evaluation harness যেটা সৎভাবে মাপে কতটা কাজ করে।

### "Deterministic" কোথায় সত্যি?

- **Tools: সম্পূর্ণ deterministic।** Pure function, total-order sort `(price, departs, id)`, input থেকে derived booking ref (`BK-BG101-RAHIM`), in-memory state।
- **LLM: শুধু "mostly" reproducible।** `temperature=0`, `seed=42` দেওয়া আছে, কিন্তু version/quantization/hardware বদলালে output বদলাতে পারে। তাই unit test-এ LLM সবসময় scripted fake, আর live behaviour মাপা হয় evaluation দিয়ে।

> **Run করার নিয়ম:** venv বানিয়ে dependency install, `ollama pull llama3.1:8b`, তারপর নিচের Quick Start।

---

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-chat-red.svg)
![Ollama](https://img.shields.io/badge/ollama-llama3.1%3A8b-black.svg)
![Pydantic](https://img.shields.io/badge/pydantic-v2-e92063.svg)
![Tests](https://img.shields.io/badge/tests-39%20passing-green.svg)

## ✨ Features

- 🔁 **Hand-written ReAct loop, দুই mode** — `native` (Ollama structured `tool_calls`) আর `text` (২০২২-এর আসল `Thought / Action / Action Input` format, regex parser + `stop=["Observation:"]`)।
- 🛑 **Typed stop reasons** — `final`, `max_steps`, `loop`, `timeout`, `error_budget`, `llm_failed`। থামলে উত্তর deterministic fallback (শেষ সফল observation থেকে), আরেকটা LLM call না।
- 🔍 **Repeat guard** — `(tool, sorted-JSON args)` fingerprint; প্রথম repeat-এ আগের result মনে করিয়ে দেয়, পরেরটায় থামে।
- 🛡️ **Trust boundary** — registry lookup + Pydantic (`extra="forbid"`, IATA/date validator), `did_you_mean` suggestion; `execute()` কখনো exception ছোড়ে না।
- ✋ **Human-in-the-loop, code-এ** — `book_flight`-এর আগে `approve(details)` gate। Streamlit-এ Approve/Reject বাটন — বুক করে code, model না।
- 🧩 **llama3.1 template workaround** — `final_answer` tool + progress-note nudge (নিচে বিস্তারিত)।
- 💬 **Conversational** — আগের turn-গুলো history হিসেবে যায়; "ওই রুটে আর কী ফ্লাইট ছিল?" কাজ করে।
- 📊 **Golden-task evaluation** — end-state + trajectory + policy check, simulated approver, pass^k।
- 🧪 **39 offline tests** — scripted fake LLM দিয়ে প্রতিটা exit, self-correction, approval, parser।

## 🚀 Quick Start

```bash
cd "Job Ready AI Bootcamp/week 13/Class 2 Project"
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama serve                                       # চালু না থাকলে

streamlit run app.py                               # chat UI

python cli.py "Book the cheapest DAC to CXB flight on 2026-11-05 for Rahim"   # approval চাইবে
python cli.py --mode text "..."                    # classic text ReAct
python cli.py --eval                               # golden tasks, native mode
python cli.py --eval --mode text -k 3              # text mode, pass^3

pytest tests -q                                    # 39 tests, Ollama লাগে না
```

## 📁 Project Structure

```
Class 2 Project/
├── app.py                       # Streamlit chat UI + approval buttons + trace viewer
├── cli.py                       # CLI: single task, --compare, --eval
├── config.py                    # Frozen dataclass config + model reference table
├── requirements.txt
├── data/
│   └── flights.json             # 12 flights (BG102 sold out, BG101/US103 price tie, QR701 last seat)
├── notebooks/
│   └── 01_exploration.ipynb     # Every exit, both modes, template deep-dive, live eval
├── src/
│   ├── agent/
│   │   └── react_agent.py       # run_agent(): the loop, parser, nudge, fallback
│   ├── tools/
│   │   └── travel.py            # TravelTools, Pydantic arg models, schemas, approval gate
│   ├── llm/
│   │   └── client.py            # The ONLY module that imports `ollama`
│   ├── evaluation/
│   │   └── harness.py           # Golden tasks, grade(), simulated_user, pass^k
│   └── utils/
│       └── logger.py
└── tests/
    └── test_pipeline.py         # 39 tests, scripted fake LLM
```

## 🔧 Architecture

```
 user message (+ chat history)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ run_agent()   while llm_calls < max_steps:                   │
│   deadline passed? ─────────────────────────────► timeout     │
│   LLM call (retry on transport error) ──────────► llm_failed  │
│      native: tools=[search, book, final_answer]               │
│      text:   stop=["Observation:"], parse Thought/Action      │
│   no call / final_answer / Final Answer ────────► final       │
│   for each action (in order):                                 │
│      same (tool,args) 2nd time → reminder; 3rd → loop         │
│      TravelTools.execute()  ── registry → Pydantic → tool     │
│          book_flight → approve(details)? ── human gate        │
│      append observation (role=tool | "Observation:")          │
│   too many invalid calls in a row ──────────────► error_budget│
│   native: append nudge = request + "already done" list        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼  answer + typed status + trace (thought, action, args, observation, prompt_tokens)
```

## 🧩 Why `final_answer` + a nudge? (llama3.1 template deep-dive)

`ollama show llama3.1:8b --template` দেখলে দুটো quirk:

1. **Tool list শুধু শেষ *user* message-এর পাশে দেখানো হয়।** Tool result আসার পর model আর tool দেখে না — তাই "Would you like to book BG101?" লিখে থেমে যায়, `book_flight` ডাকে না (nudge বন্ধ করে notebook-এ দেখানো আছে)।
2. **Tool দেখানো হলে template বলে "respond with a JSON for a function call"।** তাই শুধু একটা "continue" nudge দিলে model আর plain text-এ থামতে পারে না, চিরকাল tool ডাকে।

সমাধান (smolagents `ToolCallingAgent`-এর একই ধারণা): `final_answer` tool যোগ করা, আর প্রতিটা observation-এর পরে একটা user nudge যেটা **মূল request** আর **already-done তালিকা** দেয় (template nudge-কে `Question:` হিসেবে দেখায়, তাই request না দিলে model ভুলে যায়)।

## 📊 Live Evaluation (llama3.1:8b, Ollama 0.40.8, temperature 0)

`python cli.py --eval` — ৮টা golden task, simulated approver (শুধু request-এ লেখা নামের booking approve করে):

| Mode | Success | Invalid-call rate | Avg LLM calls | Failures |
|---|---|---|---|---|
| **native** | **6/8 (75%)** | 0.14–0.19 (দুই run) | 2.6–2.9 | sold_out → `loop`; missing_name → "John Doe" বুক করার চেষ্টা (approver আটকেছে, policy fail) |
| text | 4/8 (50%) | 0.48 | 4.4 | search-only প্রশ্নে বুক করার চেষ্টা, sold-out-এ অন্য রুট (BG201) বুক |

Prompt/loop বদলের সময় একই eval দিয়ে মাপা হয়েছে — যেমন শুধু nudge (final_answer ছাড়া) দিলে native 0/8-এ নেমে গিয়েছিল। **শিক্ষা:** ছোট model-এ প্রতিটা বদল measure করুন। `temperature=0`-এ একক local Ollama প্রায় হুবহু একই trajectory দেয়, তাই এখানে pass^k ≈ pass^1 — reliability মাপতে task-এর ভিন্ন phrasing যোগ করুন।

## ⚙️ Configuration

| Setting | Default | কেন |
|---|---|---|
| `llm.model` | `llama3.1:8b` | Course standard (ADR 0006) |
| `llm.temperature` / `seed` | `0.0` / `42` | Ollama default 0.8 — agent-এ randomness চাই না |
| `llm.num_ctx` / `num_predict` | `8192` / `512` | Default context ছোট; প্রতি step-এ prompt ~300 token বাড়ে |
| `agent.mode` | `native` | `text` = classic ReAct parsing |
| `agent.nudge_after_tools` | `True` | llama3.1 template workaround |
| `agent.max_steps` | `8` | সর্বোচ্চ LLM call |
| `agent.max_repeat_calls` | `2` | ২য়-বার reminder, ৩য়-বার `loop` |
| `agent.max_consecutive_errors` | `3` | `error_budget` |
| `agent.deadline_seconds` | `180` | `timeout` |
| `agent.max_results` | `5` | Compact observation |

## ⚠️ Known Limits

- Booking simulated, in-memory — app reset বা restart-এ মুছে যায়। Concurrent user-এর জন্য lock নেই।
- Repeat guard শুধু হুবহু একই args ধরে; ঘুরিয়ে-ফিরিয়ে repeat ধরে `max_steps`।
- Simulated approver শুধু নাম মেলায় — একই নামে ভুল ফ্লাইট (text mode-এর BG201) আটকায় না; আসল UI-তে human সেটা দেখে reject করবে।
- History শেষ ৮টা message পর্যন্ত; লম্বা conversation-এ trimming সরল।
