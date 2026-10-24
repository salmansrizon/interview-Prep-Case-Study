# 🧑‍🔬 Collaborative Multi-Agent Research Team (CrewAI + Ollama)

## সহজ ভাষায় Project Overview

একটা local research team: **Researcher** একটা local document collection থেকে fact বাছে (প্রতিটার source id সহ), **Writer** সেগুলো দিয়ে একটা structured Markdown report লেখে, **Reviewer** প্রতিটা বাক্য notes-এর সাথে মেলায়। তারপর **code** নিজে প্রতিটা citation যাচাই করে — id আসল কিনা, বাক্যটা সত্যিই সেই passage-এ আছে কিনা — আর report publish করে অথবা মানুষের কাছে পাঠায়। পুরোটা `llama3.1:8b`-এ, CrewAI Flow দিয়ে, কোনো cloud call ছাড়া।

### কোন Problem Solve করে?

Multi-agent team দেখতে আকর্ষণীয়, কিন্তু grounding ছাড়া এক agent-এর hallucination পরের agent আরও সুন্দর করে লেখে। এই project-এর পুরোনো সংস্করণে reviewer একটা **সত্য** claim ("vector database text handle করে না") "ঠিক" করে ভুল বানিয়েছিল — কারণ কারো কাছে যাচাইয়ের source ছিল না। এই সংস্করণ দেখায় কীভাবে ছোট model-এর team-কে নির্ভরযোগ্য বানাতে হয়: **retrieval আর verification code-এ, বিচারবুদ্ধির কাজ agent-এ।**

### কেন এভাবে ভাগ করা?

Live পরীক্ষায় llama3.1:8b CrewAI-র ভেতরে search tool **একবারও ডাকেনি** — tool-call JSON text হিসেবে লিখে নিজেই "search result" আর `[rag:15]` ধরনের বানানো id দিয়েছে। তাই default mode-এ Flow নিজে BM25 দিয়ে passage আনে আর researcher-এর task-এ বসায়। `tool` mode সেই failure দেখানোর জন্য রাখা আছে।

> **Run করার নিয়ম:** venv বানিয়ে dependency install (CrewAI ভারী — ~১৩৭ package), `ollama pull llama3.1:8b`, তারপর Quick Start।

---

![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)
![CrewAI](https://img.shields.io/badge/crewai-1.15-ff5a50.svg)
![Ollama](https://img.shields.io/badge/ollama-llama3.1%3A8b-black.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.40+-red.svg)
![Tests](https://img.shields.io/badge/tests-20%20passing-green.svg)

## ✨ Features

- 🔎 **Local corpus + BM25** — ৮টা English document (এই course-এর verified research থেকে), ৩৪টা passage, stable id (`rag:3`)। Embedding বা vector DB লাগে না।
- 🧑‍🤝‍🧑 **Researcher → Writer → Reviewer** — `Process.sequential`, explicit `context`, delegation বন্ধ, `max_iter=6`।
- 🛡️ **Citation guardrails** — research আর report task-এ code-level check: বানানো id, notes-এর বাইরের id, Markdown structure, ন্যূনতম citation। Fail হলে feedback সহ retry।
- ✅ **Code verification** — প্রতিটা বাক্য: `supported` / `weak` / `uncited` / `invalid_id`, support score (content-word overlap), citation precision।
- 🌊 **CrewAI Flow** — `retrieve` (code) → `run_team` (crew, retry-র hard cap ২) → `gate` (router) → `publish` | `needs_human`। Guardrail-exhaustion exception ধরে round fail গণ্য।
- ⚖️ **চারটা mode** — `sequential` (default), `single` (একই budget-এর baseline), `tool` (researcher নিজে tool ডাকে — failure demo), `hierarchical` (manager LLM — experiment)।
- 🔒 **Offline by design** — telemetry-র তিনটা flag import-এর আগে, memory/knowledge বন্ধ (নাহলে OpenAI embedder-এ যায়)।
- 🖥️ **Streamlit UI** — topic বাছুন, mode বাছুন, report + প্রতিটা বাক্যের verdict table + প্রতিটা agent-এর output।
- 🧪 **20 offline tests** — scripted fake `BaseLLM` পুরো crew, guardrail retry আর Flow router চালায়।

## 🚀 Quick Start

```bash
cd "Job Ready AI Bootcamp/week 14/Class 2 Project"
python -m venv venv && source venv/bin/activate      # Python 3.10-3.13
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama serve

streamlit run app.py
python cli.py "Why multi-agent systems fail and how to make them reliable"
python cli.py "Safe tool calling with local models" --mode single     # single-agent baseline
python cli.py "Safe tool calling with local models" --mode tool       # researcher calls the tool itself
python cli.py --search "multi-agent failure modes"                    # just the BM25 tool
pytest tests -q                                                       # 20 tests, no Ollama
```

Report-গুলো `output/`-এ (git-ignored)।

## 📁 Project Structure

```
Class 2 Project/
├── app.py                        # Streamlit UI (runs the Flow, shows verification)
├── cli.py                        # CLI: one topic, any mode, or --search
├── config.py                     # Telemetry flags (before crewai import) + dataclass config
├── requirements.txt
├── data/corpus/*.md              # 8 documents = 34 passages, ids like rag:3
├── notebooks/01_exploration.ipynb# Tool-mode failure, grounded run, multi vs single agent
├── src/
│   ├── tools/corpus_search.py    # Corpus + BM25 + CrewAI BaseTool
│   ├── crew/team.py              # Agents, tasks, guardrails, 4 modes
│   ├── flow/research_flow.py     # Flow: retrieve → team → gate → publish | needs_human
│   ├── verify/citations.py       # Deterministic citation verification
│   ├── llm/factory.py            # LLM("ollama/llama3.1:8b") + health check
│   ├── report.py                 # Save Markdown
│   └── utils/logger.py
└── tests/test_pipeline.py        # 20 tests with a fake BaseLLM
```

## 🔧 Architecture

```
topic
  │
  ▼
┌────────────────────────── ResearchFlow (CrewAI Flow, typed state) ─────────────────────────┐
│ @start retrieve        BM25 over data/corpus  (code)  → 6 passages "[id] (section) text"   │
│        │                                                                                   │
│        ▼                                                                                   │
│ @listen run_team       for round in 1..2:                                                  │
│                          Crew(sequential):                                                 │
│                            Researcher ── 5-8 cited bullets ── guardrail: ids exist, ≥4     │
│                            Writer ────── Markdown report ──── guardrail: structure, ids ⊆ notes│
│                            Reviewer ──── fixed report ─────── guardrail: same               │
│                          verify(report, corpus, notes)  (code)                             │
│                          pass? → break   else feedback → next round                        │
│        │                                                                                   │
│        ▼                                                                                   │
│ @router gate ──"ok"──► @listen publish        ──"escalate"──► @listen needs_human          │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

Publish rule: কোনো invalid বা notes-এর-বাইরের id নেই, ≥৪ citation, cited-কিন্তু-weak বাক্য ≤ cited-এর ২০%, uncited বাক্য ≤ মোটের ৫০%।

## 📊 Live Results (llama3.1:8b)

| Run | Outcome |
|---|---|
| পুরোনো crew, grounding ছাড়া | Reviewer একটা সত্য claim মুছে "সংশোধন" করেছে |
| `tool` mode (researcher নিজে tool ডাকে) | Tool **০ বার** ডাকা; বানানো id `[rag:5]...[rag:25]`; guardrail ধরেছে |
| `sequential` (grounded), CLI | **published**, ৯/৯ cited বাক্য supported, citation precision ১.০, ৬৫ s |
| `sequential`, Streamlit (প্রথম run) | needs_human (৭/১৩ বাক্য uncited — পুরোনো একক সীমায়); এর পরে uncited আর weak-এর সীমা আলাদা করা হয়েছে |
| `sequential`, Streamlit (শেষ নিয়মে) | **published**, ৭/৭ cited বাক্য supported, ৫৭ s |

**Multi-agent বনাম single-agent** (notebook Section 7, ৩টা topic, একই passage আর guardrail):

| Mode | গড় সময় | গড় token | Citation precision | গড় uncited বাক্য |
|---|---|---|---|---|
| `sequential` (৩ agent) | ৯২ s | ৭,৪৬৫ | ০.৯৫ | ১.৭ |
| `single` (১ agent) | ৬০ s | ১,০৭৬ | ১.০০ | ৪.০ |

Single agent দ্রুত আর ~৭ গুণ কম token-এ একই citation quality দিয়েছে; crew-এর একমাত্র সুবিধা কম uncited বাক্য। গবেষণার সাথে মেলে: একটা model-এ sequential role chain quality বাড়ায় না — এর মূল্য separation of concerns আর auditability।

## ⚙️ Configuration

| Setting | Default | কেন |
|---|---|---|
| `llm.model` | `llama3.1:8b` | Course-এর একমাত্র model |
| `llm.temperature` | `0.1` | Format কম ভাঙে |
| `crew.max_iter` | `6` | CrewAI default ২৫ — ছোট model-এ "Invalid Format" loop |
| `crew.guardrail_retries` | `2` | Guardrail fail হলে task-এর retry |
| `crew.flow_max_rounds` | `2` | Verification fail হলে পুরো crew-এর retry (hard cap) |
| `crew.retrieve_k` / `passage_chars` | `6` / `450` | Context ছোট রাখা (Ollama default ৪০৯৬ token) |
| `verify.min_support` | `0.45` | Sentence-এর content word-এর কত অংশ cited passage-এ |
| `verify.max_weak_ratio` / `max_uncited_ratio` | `0.2` / `0.5` | Weak = আসল বিপদ; uncited প্রায়ই transition |

## ⚠️ Known Limits

- **Support score একটা proxy** — content-word overlap; paraphrase কম score পায়, শব্দ মিলিয়ে ভুল দাবিও পার পেতে পারে। Production-এ NLI model বা quote-matching।
- Reviewer আর generator একই model — self-preference bias; তাই code check-ই মূল gate।
- Single Ollama server-এ সব request queue-তে — crew sequential, ১-৩ মিনিট/report।
- CrewAI Flow console-এ বড় banner ছাপে (বন্ধ করার documented option পাওয়া যায়নি)।
- Live ফল run থেকে run-এ বদলায়।
