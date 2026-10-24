# 🔁 Self-Correcting LLM Coder (LangGraph + Ollama)

## সহজ ভাষায় Project Overview

একটা local LLM (`llama3.1:8b`) Python function লেখে, একটা আলাদা process-এ test চালানো হয়, fail করলে **কোন test কী ফেরত দিল আর কী প্রত্যাশিত ছিল** সেটা model-কে ফেরত দেওয়া হয়, আর model নিজের code শুধরে আবার চেষ্টা করে — retry budget শেষ না হওয়া পর্যন্ত। শেষে একটা **hidden test** set চলে যেটা model কখনো দেখে না, আর তারপর মানুষের approval। পুরো loop-টা একটা LangGraph `StateGraph`।

### কোন Problem Solve করে?

LLM-এর লেখা code প্রথমবারেই ঠিক হবে, এমন না — বিশেষ করে spec-এ খুঁটিনাটি থাকলে। আবার "test পাস করেছে" মানেই সঠিক না: model example hard-code করে test পাস করতে পারে (reward hacking)। এই project দেখায় কীভাবে (১) test feedback দিয়ে self-repair, (২) hidden test দিয়ে "মুখস্থ" ধরা, (৩) subprocess-এ সীমিত execution, আর (৪) checkpoint + human approval — সব একটা পরিষ্কার, testable graph-এ।

### কেন LangGraph?

Loop, branch আর pause (human approval) একটা explicit graph-এ থাকে — state typed, প্রতিটা step checkpoint হয় (SQLite), restart-এর পরও run দেখা যায়, আর পুরোনো checkpoint থেকে replay (time travel) করা যায়।

> **Run করার নিয়ম:** venv বানিয়ে dependency install, `ollama pull llama3.1:8b`, তারপর Quick Start।

---

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/langgraph-1.2-1c3c3c.svg)
![Ollama](https://img.shields.io/badge/ollama-llama3.1%3A8b-black.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.40+-red.svg)
![Tests](https://img.shields.io/badge/tests-27%20passing-green.svg)

## ✨ Features

- 🔁 **Self-repair loop** — generate → static check → visible tests → (fail হলে failing-test diff সহ retry) → hidden tests → approval।
- 🧪 **Per-test feedback** — "`roman_to_int('IV')`: returned 6 (int), expected 4"। Self-Debugging paper-এ unit-test feedback-ই সবচেয়ে বড় লাভ দিয়েছিল।
- 🙈 **Hidden tests** — model কখনো দেখে না; fail করলে শুধু একটা generic hint ("আরও case-এ ভুল, spec আবার পড়ো")। Example hard-code করা আর spec অর্ধেক বোঝা দুটোই ধরে।
- 🧱 **Sandboxed-ish execution** — আলাদা process, `python -I`, খালি env, timeout, `RLIMIT_CPU/AS/FSIZE` (POSIX)। আসল sandbox না — README-তে আর notebook-এ সেই সীমা দেখানো আছে।
- 🔍 **AST static check** — syntax, function-এর নাম, banned import/call, dunder access — চালানোর আগেই পরিষ্কার feedback।
- ✋ **Human-in-the-loop** — LangGraph `interrupt()` + `Command(resume=...)`; UI-তে Approve/Reject বাটন।
- 💾 **SQLite checkpointer + time travel** — run restart-এর পরও থাকে; `get_state_history` দিয়ে প্রতিটা checkpoint, পুরোনোটা থেকে replay।
- 📊 **Benchmark** — ১১টা task-এ pass@1 বনাম self-repair-এর পরে।
- 🧪 **27 offline tests** — fake LLM, কিন্তু real subprocess sandbox আর real SQLite।

## 🚀 Quick Start

```bash
cd "Job Ready AI Bootcamp/week 14/Class 1 Project"
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama serve

streamlit run app.py                 # UI: live graph steps, approval, time travel
python cli.py                        # list the 11 tasks
python cli.py duration               # solve one task (asks for approval at the end)
python cli.py --bench                # all tasks: pass@1 vs pass after repair
pytest tests -q                      # 27 tests, Ollama লাগে না
```

## 📁 Project Structure

```
Class 1 Project/
├── app.py                     # Streamlit: stream graph updates, approve/reject, time travel
├── cli.py                     # CLI: one task with approval, or --bench
├── config.py                  # Frozen dataclass config + model reference table
├── requirements.txt
├── data/                      # checkpoints.sqlite (created at runtime, git-ignored)
├── notebooks/
│   └── 01_exploration.ipynb   # Every path of the graph, time travel, live run + benchmark
├── src/
│   ├── graph/coder.py         # StateGraph: state, nodes, routing, interrupt, checkpointers
│   ├── sandbox/runner.py      # Subprocess test harness, per-test JSON results, limits
│   ├── sandbox/static_check.py# Code extraction + AST checks
│   ├── tasks/bank.py          # 11 coding tasks with visible + hidden tests
│   ├── evaluation/bench.py    # pass@1, pass after repair, hidden_fail, gave_up
│   ├── llm/client.py          # The ONLY module that imports `ollama`
│   └── utils/logger.py
└── tests/test_pipeline.py     # 27 tests
```

## 🔧 Architecture

```
START → generate ─► static_check ──fail──┐
                        │ ok             │   attempts < max ? → generate (with feedback)
                        ▼                │                  : → give_up (best attempt kept)
                   run_visible ──fail────┤
                        │ pass           │
                        ▼                │
                   run_hidden ───fail────┘   (generic hint only — hidden tests never leak)
                        │ pass
                        ▼
                   approval ── interrupt() ── Command(resume=True|False)
                        │
                 done ◄─┴─► rejected                (budget out after hidden fail → hidden_fail)

State (TypedDict): code, attempts, feedback, visible/hidden pass counts, best_code,
                   status, history  ← Annotated[list, operator.add] reducer
Checkpointer: InMemorySaver (tests) | SqliteSaver data/checkpoints.sqlite (app, cli)
```

## 📊 Live Benchmark (llama3.1:8b, temperature 0, retry budget 4)

`python cli.py --bench`, ১১টা task, একই ফল CLI আর notebook — দুই run-এ:

| Metric | Value |
|---|---|
| pass@1 (প্রথম attempt-এ visible + hidden সব) | **8/11 (73%)** |
| pass after self-repair | **9/11 (82%)** |
| repaired | 1 (`format_duration` — ৩য় attempt-এ) |
| gave_up | 2 (`word_freq`, `camel_to_snake`) |

পর্যবেক্ষণ: সহজ task প্রথমবারেই হয়; repair-এর লাভ spec-ভারী task-এ। `word_freq`-এ প্রথম attempt visible পাস করেছিল কিন্তু hidden fail; generic hint পেয়ে পরের attempt-গুলো **খারাপ** হয়েছে — "Is Self-Repair a Silver Bullet?"-এর মূল কথা: feedback-এর মানই bottleneck।

## ⚙️ Configuration

| Setting | Default | কেন |
|---|---|---|
| `llm.model` | `llama3.1:8b` | Course-এর একমাত্র model (ADR 0006) |
| `llm.temperature` / `seed` | `0.0` / `42` | Repair step greedy (Self-Debugging); Ollama default 0.8 |
| `llm.num_ctx` / `num_predict` | `8192` / `1024` | Retry prompt-এ আগের code + feedback থাকে |
| `agent.max_attempts` | `4` | Research: বেশিরভাগ fix প্রথম ~৩ turn-এ |
| `agent.recursion_limit` | `40` | LangGraph 1.x default আর ২৫ না (~১০ হাজার) — নিজে সীমা |
| `sandbox.timeout_seconds` | `10` | `while True` আটকায় |
| `sandbox.memory_mb` | `512` | `RLIMIT_AS` (Linux-এ কার্যকর, macOS-এ প্রায়ই না) |
| `sandbox.banned_imports` / `banned_calls` | os, subprocess, socket… / eval, exec, open… | Feedback-এর জন্য; নিরাপত্তার ভরসা না |

## ⚠️ Known Limits

- **Subprocess আসল sandbox না।** Generated code disk পড়তে পারে (notebook-এ `os.listdir('/')` দেখানো আছে); network বন্ধ না। Untrusted code production-এ Docker `--network none`, gVisor, বা Firecracker/e2b-তে চালান।
- **Static check বাইপাসযোগ্য** (`importlib`, string trick) — শুধু ভালো feedback-এর জন্য।
- **Hidden test ছোট** (task-প্রতি ৩টা) — EvalPlus দেখিয়েছে বেশি test-এ pass@k ২৮.৯% পর্যন্ত কমে।
- Live সংখ্যা Ollama version/hardware অনুযায়ী বদলাতে পারে।
