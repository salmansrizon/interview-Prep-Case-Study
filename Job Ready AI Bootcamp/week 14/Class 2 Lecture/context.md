# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Agentic Workflows with CrewAI — Multi-Agent Team, আর কখন সেটা আসলে কাজে লাগে**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

গত ক্লাসে LangGraph-এ আমরা প্রতিটা ধাপ আর edge নিজে লিখেছি — পূর্ণ নিয়ন্ত্রণ। **CrewAI** উল্টো দিক থেকে আসে: আপনি একটা "টিম" বর্ণনা করেন — কে researcher, কে writer, কে reviewer, কার কী কাজ — আর framework তাদের কাজ ক্রমে চালায়। এটা দ্রুত বানানো যায়, পড়তে সহজ। কিন্তু প্রশ্ন থাকে: একটা ছোট local model-এ কয়েকটা "role" বানালেই কি ফল ভালো হয়? এই ক্লাসে আমরা CrewAI-এর API শিখব **আর** গবেষণা দিয়ে যাচাই করব কখন multi-agent সাহায্য করে, কখন শুধু খরচ বাড়ায়।

## শেখার সহজ Workflow

Agent / Task / Crew / Process → LLM (Ollama) → Custom tool → Guardrail → Flow → Telemetry/memory-র ফাঁদ → Multi-agent research (Anthropic, Cognition, Google, MAST) → Grounding আর citation verification → Local hardware-এর খরচ → কখন multi-agent লাগবে না।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** প্রতিটা crew run-এর আগে লিখে ফেলুন: কোন agent কোন তথ্য দেখবে, কোথা থেকে ভুল ঢুকতে পারে, আর কোন check সেটা ধরবে। Multi-agent-এ ভুল এক agent থেকে পরের agent-এ বাড়ে।

---

## 1. Topic: Agentic Workflows with CrewAI

এই ক্লাসে আমরা শিখব:

* **CrewAI 1.15-এর মূল abstraction** — `Agent`, `Task`, `Crew`, `Process`, `Flow`।
* **Local Ollama** — 1.12 থেকে native (LiteLLM লাগে না)।
* **Custom tool, guardrail, structured output** — 8B model-কে নির্ভরযোগ্য বানানোর অস্ত্র।
* **Offline/sovereign setup** — telemetry, memory, knowledge-এর লুকানো network default।
* **Multi-agent research** — কখন জেতে (parallel research), কখন হারে (sequential কাজ), কেন ভাঙে (MAST)।
* **Grounding আর evaluation** — citation, code-level verification, LLM-as-judge-এর bias।

**Project Goal (Class 2 Project — Portfolio Project 13: Collaborative Multi-Agent Research Team):** একটা local corpus থেকে Researcher fact সংগ্রহ করে, Writer একটা structured report লেখে, Reviewer প্রতিটা claim notes-এর সাথে মেলায় — আর শেষে code নিজে প্রতিটা citation যাচাই করে report publish করে বা মানুষের কাছে পাঠায়। পুরোটা `llama3.1:8b`-এ, কোনো cloud call ছাড়া।

---

## 2. Why It Is Related

Week 12-এর RAG retrieval এখানে research team-এর "চোখ"। Week 13-এর tool calling আর ReAct loop CrewAI-এর প্রতিটা agent-এর ভেতরে চলে (CrewAI agent = একটা ReAct loop)। Class 27-এর LangGraph দেখিয়েছে explicit control; CrewAI দেখায় role-based abstraction — আর আমরা দেখব কেন ছোট model-এ দুটো মিশিয়ে চলতে হয়: **deterministic shell (Flow/code), agentic core (crew)**। Week 15-এ এই team-কে API হিসেবে serve আর monitor করব।

---

## 3. How It Works

### 3.1 চারটা abstraction

```python
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434", temperature=0.1)

researcher = Agent(role="Senior Research Specialist",
                   goal="Collect accurate facts about {topic}, each with its citation id.",
                   backstory="You are meticulous and never add anything the documents do not say.",
                   llm=llm, allow_delegation=False, max_iter=6)
research = Task(description="Pick the facts relevant to {topic} from the source passages...",
                expected_output="5-8 bullet points, each ending with a citation id like [rag:2].",
                agent=researcher)
write = Task(description="Write a Markdown report...", expected_output="...", agent=writer,
             context=[research])                     # dependency: research-এর output এখানে ঢোকে
crew = Crew(agents=[researcher, writer, reviewer], tasks=[research, write, review],
            process=Process.sequential)
out = crew.kickoff(inputs={"topic": "Why multi-agent systems fail"})   # {topic} বসে যায়
print(out.raw, out.tasks_output[0].raw, out.token_usage)
```

* **Agent** — `role` (সে কে), `goal` (কী চায়), `backstory` (কেমন মানসিকতা) — তিনটাই system prompt-এ ঢোকে।
* **Task** — `description`, `expected_output`, `agent`, `context` (কোন আগের task-এর output দেখবে), `guardrail`, `output_pydantic`।
* **Crew** — agent + task + process। `kickoff` → `CrewOutput` (`raw`, `tasks_output`, `token_usage`)।
* **Process** — `sequential` (list-এর ক্রমে) বা `hierarchical` (একটা manager LLM delegate করে)।

### 3.2 1.15-এর default — ২০২৪-এর tutorial থেকে যা বদলেছে

| বিষয় | এখন |
| --- | --- |
| Version | CrewAI **1.15.29** (2026-10-23), Python ≥3.10,<3.14; 1.0 এসেছিল 2025-10-20 |
| Ollama | 1.12 থেকে **native** OpenAI-compatible client; LiteLLM optional (docs page এখনো পুরোনো কথা বলে) |
| Tool import | `from crewai.tools import BaseTool, tool` (`crewai_tools` না) |
| `Agent.max_iter` | default **25** — 8B model-এ ৫-৮ রাখুন |
| `Agent.allow_delegation` | default `False` |
| Guardrail retry | `guardrail_max_retries` default **3** |
| Memory | একটাই LanceDB `Memory` class; চালু করলে default **OpenAI** embedder |
| Install | base install ~১৩৭ package (chromadb, lancedb, onnxruntime, kubernetes...) |

### 3.3 Custom tool

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(..., description="A few keywords")

class CorpusSearchTool(BaseTool):
    name: str = "search_corpus"
    description: str = "Search the local documents. Returns passages starting with ids like [rag:2]."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        ...                                            # exception না, "NO_RESULTS" ফেরত দিন
```

Agent tool বাছে নাম, description আর argument schema দেখে। ছোট model-এর জন্য: একটা পরিষ্কার কাজ, একটাই flat `str` argument, sentinel return (`NO_RESULTS`)।

### 3.4 Guardrail — task-এর output code-এ যাচাই

```python
def check(output):                     # return annotation নেই — নিচে কারণ
    if not cited_ids(output.raw):
        return False, "Cite every fact with an id like [rag:2]."   # এই message সহ task আবার চলে
    return True, output.raw            # (True, output) না — 1.15-এ structured output হারায়

research = Task(..., guardrail=check, guardrail_max_retries=2)
```

দুটো ফাঁদ যা আমরা live পেয়েছি/research পেয়েছে: (১) CrewAI guardrail-এর return annotation যাচাই করে — file-এর শুরুতে `from __future__ import annotations` থাকলে annotation একটা string হয়ে যায় (`"tuple[bool, str]"`), আর CrewAI task বানাতেই `ValidationError` দেয় (আমাদের project-এ ঘটেছে; ওই import ছাড়া একই annotation গৃহীত হয়) — নিরাপদ পথ: annotation না দেওয়া; (২) `(True, output)` ফেরত দিলে `CrewOutput.pydantic` `None` হয়ে যায়, `(True, output.raw)` দিন। String guardrail-ও আছে (LLM দিয়ে বিচার) — কিন্তু তাতে একই 8B model নিজেকে grade করে।

### 3.5 Flow — deterministic shell

```python
from crewai.flow.flow import Flow, start, listen, router

class ResearchFlow(Flow[ResearchState]):        # typed state (Pydantic)
    @start()
    def retrieve(self): ...                     # code: BM25 retrieval
    @listen(retrieve)
    def run_team(self, passages): ...           # crew: research → write → review (+ capped retry)
    @router(run_team)
    def gate(self): return "ok" if self._passes() else "escalate"
    @listen("ok")
    def publish(self): ...
    @listen("escalate")
    def needs_human(self): ...
```

নিয়ম: **control predictable রাখতে হলে Flow** (routing, retry-র hard cap, persistence, human approval); **একটা ধাপের ভেতরে open-ended collaboration হলে Crew**। "Deterministic shell, agentic core।"

### 3.6 Offline/sovereign — তিনটা লুকানো network default

* **Telemetry default-এ চালু** — role name, tool name, LLM info, timing পাঠায়। বন্ধ করতে **তিনটা** flag, crewai import-এর **আগে**: `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, `CREWAI_TRACING_ENABLED=false` (tracing আলাদা switch)।
* **Memory/knowledge** — চালু করলে embedder না দিলে OpenAI-তে যায়; `Crew(memory=True)` বানাতে কোনো error হয় না, run-এর সময় fail করে। আমাদের project-এ দুটোই বন্ধ, retrieval নিজের tool-এ।
* **`crewai test`** — শুধু OpenAI দিয়ে চলে। Offline-এ নিজের harness লাগে: একটা ১০ লাইনের `BaseLLM` subclass যেটা `"Thought: ...\nFinal Answer: ..."` ফেরত দেয়, পুরো crew চালাতে পারে।

### 3.7 Live পর্যবেক্ষণ — llama3.1:8b CrewAI-র ভেতরে tool ডাকেনি

আমাদের প্রথম live run-এ researcher-কে `search_corpus` tool দেওয়া ছিল। Model tool **একবারও ডাকেনি** — tool-call JSON plain text হিসেবে লিখে, তারপর নিজেই "search result" বানিয়ে নিয়েছে: `[rag:15]`, `[rag:22]` ধরনের id (rag.md-তে মাত্র ৪টা section আছে)। Week 13-এর "leaked tool call" সমস্যাই, এবার framework-এর ভেতরে। Guardrail ধরেছে, কিন্তু retry শেষে CrewAI exception ছুড়ে পুরো Flow crash করেছিল।

সমাধান দুই স্তরে: (১) **retrieval code-এ** — Flow BM25 দিয়ে passage আনে, researcher শুধু বাছাই আর সংক্ষেপ করে (model যেটায় সবচেয়ে দুর্বল, সেটা code করে); (২) Flow guardrail-exhaustion exception ধরে round-টাকে fail গণ্য করে, পরের round-এ feedback সহ চেষ্টা করে, শেষে `needs_human`। `tool` mode project-এ পরীক্ষা হিসেবে রাখা আছে।

---

## 4. Details & Valid Points

### 4.1 Multi-agent গবেষণা — চারটা মূল ফল

| Source | কী পেয়েছে | শিক্ষা |
| --- | --- | --- |
| **Anthropic** research system (June 2025) | Orchestrator-worker (lead Opus 4 + parallel Sonnet 4 subagent + CitationAgent) single Opus 4-কে **৯০.২%** হারিয়েছে; ~**১৫×** chat token; BrowseComp-এ token usage একাই ৮০% variance ব্যাখ্যা করে | লাভ মূলত **parallel token খরচ** থেকে, "role-play" থেকে না |
| **Cognition**, "Don't Build Multi-Agents" | Agent-রা পুরো context না পেলে পরস্পরবিরোধী implicit সিদ্ধান্ত নেয় | Context শেয়ার করুন; single-threaded agent প্রায়ই ভালো |
| **Google/MIT** scaling study (Dec 2025, ১৮০ config) | Parallel-যোগ্য task-এ +৮০.৯%, **sequential task-এ −৩৯% থেকে −৭০%**; independent agent ভুল **১৭.২×** বাড়ায়, validating orchestrator **৪.৪×** | Task-এর ধরনই ঠিক করে multi-agent লাভ কিনা |
| **MAST** (Cemri et al. 2025) | ১৬০০+ trace, ৭ framework, ১৪ failure mode; failure rate ৪১-৮৬.৭% | ভুলের বড় অংশ design আর verification-এ |

> **Valid Point:** আমাদের Researcher → Writer → Reviewer crew একটা **sequential prompt chain**, একটাই model, একটাই GPU queue — Anthropic-এর parallel orchestrator-worker না। তাই এর সৎ দাবি "ভালো separation of concerns আর auditability", "বেশি quality" না — যতক্ষণ না একই budget-এর single-agent baseline অন্য কথা বলে। Project-এ `single` mode ঠিক সেই baseline — আর আমাদের live তুলনায় (৩টা topic) single agent ৬০ s আর ~১,০৭৬ token-এ citation precision ১.০০ পেয়েছে, ৩-agent crew ৯২ s আর ~৭,৪৬৫ token-এ ০.৯৫; crew-এর একমাত্র সুবিধা কম uncited বাক্য (গড় ১.৭ বনাম ৪.০)।

### 4.2 MAST failure mode → আমাদের crew

| Category (share) | প্রধান mode | llama3.1:8b-এ কেমন দেখায় | আমাদের mitigation |
| --- | --- | --- | --- |
| System design (~৪৪%) | Step repetition ১৫.৭%, termination না বোঝা ১২.৪%, task spec না মানা ১১.৮% | একই কাজ বারবার; format না মানা | `max_iter=6`, guardrail, Flow-এ retry-র hard cap |
| Inter-agent misalignment (~৩২%) | Reasoning-action mismatch ১৩.২% | "Cite করব" বলে cite না করা | Code-level citation guardrail |
| Task verification (~২৩.৫%) | ভুল verification ৯.১%, অসম্পূর্ণ ৮.২% | Reviewer "ঠিক আছে" বলে unsupported claim পাস করায় | Reviewer notes দেখে; code প্রতিটা বাক্যের support মাপে |

MAST-এর পরীক্ষায় একটা high-level **objective verification step** যোগ করে ChatDev-এ +১৫.৬% — সবচেয়ে বড় একক উন্নতি। আমাদের Flow-এর code verification ঠিক সেই জায়গা।

### 4.3 Grounding আর evaluation

Web search ছাড়া grounding মানে তিনটা জিনিস: local corpus থেকে retrieval, প্রতিটা claim-এ source id, আর verification **retrieved text-এর বিরুদ্ধে** (writer-এর গদ্যের বিরুদ্ধে না)। আমাদের metric:

* **Citation precision** — cited বাক্যের কত অংশ সত্যিই cited passage দিয়ে supported (code মাপে: content-word overlap)।
* **Uncited ratio** — কত বাক্য কোনো source ছাড়া (প্রায়ই transition — তাই আলাদা, ঢিলা সীমা)।
* **Invalid / not-in-notes id** — বানানো id বা notes-এ না থাকা id — শূন্য হতে হবে।

ALCE benchmark: সেরা model-ও ELI5-এ ৫০% সময় পূর্ণ citation support দেয় না। **LLM-as-judge bias** (Zheng et al.): position, verbosity, self-enhancement; self-preference model-এর নিজের লেখা চেনার ক্ষমতার সাথে রৈখিকভাবে বাড়ে (Panickssery et al.)। আমাদের reviewer আর generator একই llama3.1:8b — তাই যত check সম্ভব, code-এ।

### 4.4 একটা Ollama server = একটা queue

Ollama default-এ model-প্রতি **একবারে একটা request** (`OLLAMA_NUM_PARALLEL`), context **৪০৯৬ token** (CrewAI ধরে নেয় ~৭k — নিঃশব্দ truncation-এর ঝুঁকি; `OLLAMA_CONTEXT_LENGTH` দিয়ে বাড়ান)। Sequential crew-এর সময় = সব LLM call-এর যোগফল; আমাদের live run-এ grounded crew ~১-৩ মিনিট, ৯টা request, ~৯ হাজার token। "Parallel" research task single server-এ মূলত queue-তে অপেক্ষা করে।

### 4.5 Framework landscape ২০২৬

| Framework | পরের ধাপ কে ঠিক করে | State | একটা local 8B-এ |
| --- | --- | --- | --- |
| CrewAI 1.15 | নির্দিষ্ট ক্রম (sequential) বা manager LLM; Flow-এ code | Task `context`; optional Memory | Sequential, delegation ছাড়া ভালো; hierarchical ভঙ্গুর |
| LangGraph | Explicit edge; supervisor = tool call | Typed state + checkpointer | ভালো |
| OpenAI Agents SDK | Handoff / agents-as-tools বা code | Run context | Ollama-তে যাচাই করা হয়নি |
| Google ADK 2.0 | Template + graph workflow | Session state | Ollama option আছে |
| Microsoft Agent Framework / AG2 | Agent + graph / group chat | Thread | ছোট model-এ speaker selection কঠিন |

AutoGen ২০২৫-এর অক্টোবর থেকে maintenance mode-এ, উত্তরসূরি Microsoft Agent Framework (secondary source অনুযায়ী ২০২৬-এর এপ্রিলে 1.0)।

---

## 5. Related Analogy: The Newspaper Newsroom

> একটা পত্রিকার newsroom — কিন্তু সব সাংবাদিক একই মানুষ, শুধু আলাদা টুপি পরে।

| Concept | Newsroom Analogy |
| --- | --- |
| **Researcher agent** | Reporter — archive থেকে তথ্য আনে, প্রতিটার source লিখে রাখে |
| **Writer agent** | Feature writer — reporter-এর notes থেকে লেখা বানায় |
| **Reviewer agent** | Copy editor — প্রতিটা বাক্য notes-এর সাথে মেলায় |
| **Code verification** | Fact-checking desk যেটা একটা নিয়মের বই মানে — "source কোথায়? দেখাও" — কারো মতামতে না |
| **Hierarchical manager** | একজন editor-in-chief যিনি কাজ ভাগ করেন — কিন্তু অনভিজ্ঞ হলে ভুল লোককে ভুল কাজ দেন |
| **একই model সব role-এ** | একই মানুষ নিজের লেখা নিজে edit করছে — নিজের ভুল চোখে পড়ে কম (self-preference bias) |
| **Hallucination propagation** | Reporter একটা গুজব লিখল, writer সেটা সুন্দর করে লিখল, editor বানান ঠিক করল — গুজব এখন ছাপা খবর |
| **Single-agent baseline** | একজন অভিজ্ঞ সাংবাদিক একা পুরো খবর লেখেন — প্রায়ই দ্রুত, কখনো একই মানের |

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
| --- | --- | --- |
| ২০২৪-এর import আর LiteLLM ধরে নেওয়া | `ImportError` বা ভারী venv | `from crewai.tools import ...`; plain `crewai` install |
| 8B-এ hierarchical process | Manager delegation tool call-এ ব্যর্থ | Sequential, explicit `context`, delegation বন্ধ |
| ReAct format loop | "Invalid Format: I missed the 'Action:'..." | ছোট prompt, `max_iter` ৫-৮, কম temperature |
| Small model-কে tool দিয়ে retrieval | Tool না ডেকে result বানায় (আমাদের live run) | Retrieval code-এ, agent শুধু বাছাই করে |
| `from __future__ import annotations` সহ guardrail-এ return annotation | Annotation string হয়ে যায় → task বানাতেই `ValidationError` | Guardrail-এ annotation দেবেন না |
| Guardrail retry শেষে exception | পুরো run crash | Flow-এ catch, round fail, `needs_human` |
| Flow-এর ভেতর থেকে UI call | Worker thread-এ Streamlit call ব্যর্থ (আমাদের app-এ ঘটেছে) | Callback শুধু data জমায়, UI পরে দেখায় |
| Jupyter-এ `crew.kickoff()` | `RuntimeError: ... invoked synchronously from within a running event loop` (1.15) | Notebook-এ `await crew.kickoff_async()`; script-এ `kickoff()` |
| ৪০৯৬-token server context | Notes নিঃশব্দে কেটে যায় | `OLLAMA_CONTEXT_LENGTH` ৮-১৬k |
| Embedder ছাড়া memory/knowledge | Run-time-এ OpenAI key error বা data বাইরে | বন্ধ রাখুন, বা Ollama embedder (`model_name`) |
| Telemetry বন্ধ ধরে নেওয়া | Role/tool নাম বাইরে যায় | তিনটা env flag, import-এর আগে |
| Reviewer শুধু draft দেখে | Hallucination পাস | Reviewer-কে notes দিন (`context=[research, write]`) + code check |
| Revision loop-এ cap নেই | অসীম review cycle | Flow-এ ১-২ round |
| Baseline নেই | Crew-কে ভালো ধরে নেওয়া | একই budget-এর single agent |

---

## 7. Interview Q&A

1. **CrewAI-এর মূল abstraction কী?** Agent (role, goal, backstory, tools, llm), Task (description, expected_output, agent, context, schema, guardrail), Crew (team + process), Flow (typed state, `@start`/`@listen`/`@router` দিয়ে code-চালিত orchestration)।
2. **Sequential নাকি hierarchical — এখানে কেন sequential?** Hierarchical-এ manager LLM tool call দিয়ে delegate করে; 8B-এ সেই call প্রায়ই ভাঙে আর একটা-request-এর server-এ latency বাড়ায়। নির্দিষ্ট ক্রম + explicit context বেশি নির্ভরযোগ্য আর সস্তা।
3. **ছোট model থেকে নির্ভরযোগ্য structured output কীভাবে?** Flat field-এর `output_pydantic`, তারপর callable guardrail যেটা `(False, "নির্দিষ্ট fix")` দিয়ে retry করায়; সফল হলে `(True, out.raw)`।
4. **8B model-এর জন্য tool কীভাবে ডিজাইন করবেন?** একটা কাজ, একটা flat `str` argument, নিখুঁত description, exception-এর বদলে `NO_RESULTS`। আর যাচাই করুন model আসলেই tool ডাকছে কিনা — আমাদের run-এ ডাকেনি, result বানিয়েছে।
5. **CrewAI পুরোপুরি offline কীভাবে?** `LLM(model="ollama/llama3.1:8b")` (1.12+ native), memory/knowledge বন্ধ বা Ollama embedder, telemetry-র তিনটা flag import-এর আগে, `crewai test`-এর বদলে নিজের harness — আর network traffic দেখে নিশ্চিত হোন।
6. **Flow কখন, Crew কখন?** Control deterministic হতে হলে (routing, capped retry, persistence, approval) Flow; একটা ধাপের ভেতরে open-ended collaboration হলে Crew। "Deterministic shell, agentic core"।
7. **LLM ছাড়া crew test কীভাবে?** Tool `_run` সরাসরি test; `build_crew(llm)` factory-তে `FakeLLM(BaseLLM)` inject করে wiring, placeholder, guardrail retry, Flow router test; quality আলাদা eval set-এ।
8. **কখন single agent-এর চেয়ে multi-agent ভালো?** কাজ যখন স্বাধীন parallel অংশে ভাগ হয়, context isolation দরকার, বা role-গুলোর tool/prompt খুব আলাদা। Anthropic: +৯০.২% কিন্তু ~১৫× token; Google: parallel-এ +৮০.৯%, sequential-এ −৩৯ থেকে −৭০%।
9. **Anthropic আর Cognition-এর বিরোধ মেলান।** ভিন্ন task: research read-heavy ও parallel, একজন lead ফল মেলায়; coding write-heavy, অংশগুলোকে মিলতে হয়। দুই পক্ষেরই আসল কথা context engineering।
10. **MAST কী?** ১৬০০+ trace, ৭ framework, ১৪ failure mode, তিন ভাগ: system design (~৪৪%), inter-agent misalignment (~৩২%), task verification (~২৩.৫%)। শীর্ষ: step repetition ১৫.৭%, reasoning-action mismatch ১৩.২%, termination না বোঝা ১২.৪%।
11. **Reviewer hallucinated claim পাস করাচ্ছে — কী করবেন?** Reviewer-কে notes দিন (শুধু draft না), প্রতি বাক্যে verdict চান, id আর support-এর check code-এ নিন, revision round cap করুন। MAST-এ verification step-ই সবচেয়ে বড় একক fix (+১৫.৬%)।
12. **Generated report কীভাবে evaluate করবেন?** ~২০টা বাস্তব প্রশ্ন; rubric judge (factual accuracy, citation accuracy, completeness, source quality); ALCE-ধাঁচের citation precision/recall; key-point coverage; position swap, length normalize, ভিন্ন family-র judge; হাতে spot-check।
13. **LLM-as-judge-এর bias কী কী?** Position, verbosity, self-enhancement, সীমিত reasoning; self-preference নিজের লেখা চেনার ক্ষমতার সাথে বাড়ে; শক্তিশালী judge-ও মানুষের সাথে ~৮০% একমত।
14. **Ollama-তে crew ধীর আর truncate করছে — কেন?** Sequential call যোগ হয়, প্রতিটা ReAct step-এ prompt বাড়ে; Ollama default-এ একবারে একটা request আর ৪০৯৬ token context, CrewAI ধরে ~৭k। `OLLAMA_CONTEXT_LENGTH` বাড়ান, verbose reasoning কমান, `max_iter` নামান।
15. **CrewAI 0.x থেকে 1.x-এ কী বদলেছে?** Native provider SDK (LiteLLM optional), Ollama native 1.12-এ, monorepo + আলাদা `crewai-cli`, একক LanceDB Memory, guardrail list ও LLM guardrail, `akickoff`, Flow persistence, `@human_feedback`; `CodeInterpreterTool` বাদ।

---

## Achievement: Collaborative Multi-Agent Research Team (Portfolio Project 13)

**`Class 2 Project/`** — CrewAI Flow: code-এ BM25 retrieval → Researcher/Writer/Reviewer crew (citation guardrail সহ) → code-এ citation verification → publish বা `needs_human`; চারটা mode (`sequential`, `single` baseline, `tool` experiment, `hierarchical` experiment); ৮টা document-এর local corpus; Streamlit UI-তে প্রতিটা বাক্যের verdict; ২০টা offline test (fake `BaseLLM`)। এই ক্লাসের companion notebook **`crewai_lecture.ipynb`** সম্পূর্ণ self-contained — project folder ছাড়াই চলে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Tool নাকি Code:** `tool` mode আর `sequential` mode একই topic-এ তিনবার করে চালান। কোনটায় invalid citation id আসে? কেন?
2. **Reviewer-এর মূল্য:** Reviewer বাদ দিয়ে ২-agent crew বানান। Citation precision আর সময়ে কী পার্থক্য?
3. **Same-budget Baseline:** `single` mode-এ max_iter বাড়িয়ে তার token খরচ crew-এর সমান করুন। Quality কি crew-কে ধরে ফেলে?
4. **Self-preference পরীক্ষা:** একটা report-কে llama3.1:8b দিয়ে ১-১০ score করান, তারপর একই report-কে "অন্য model-এর লেখা" বলে score করান। Score কি বদলায়?
5. **Hierarchical:** `hierarchical` mode চালিয়ে manager-এর আচরণ দেখুন। Delegation tool call কি ঠিকমতো হয়?

---

## References

* CrewAI docs — Agents https://docs.crewai.com/en/concepts/agents · Tasks https://docs.crewai.com/en/concepts/tasks · Crews https://docs.crewai.com/en/concepts/crews · Flows https://docs.crewai.com/en/concepts/flows · LLMs https://docs.crewai.com/en/concepts/llms · Custom tools https://docs.crewai.com/en/learn/create-custom-tools · Memory https://docs.crewai.com/en/concepts/memory · Telemetry https://docs.crewai.com/en/telemetry · Testing https://docs.crewai.com/en/concepts/testing · Changelog https://docs.crewai.com/en/changelog
* CrewAI on PyPI — https://pypi.org/pypi/crewai/json · Release 1.0.0 — https://github.com/crewAIInc/crewAI/releases/tag/1.0.0
* Anthropic, "How we built our multi-agent research system" — https://www.anthropic.com/engineering/multi-agent-research-system · "Building effective agents" — https://www.anthropic.com/engineering/building-effective-agents
* Cognition, "Don't Build Multi-Agents" — https://cognition.com/blog/dont-build-multi-agents
* Google Research, scaling agent systems — https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/ · arXiv 2512.08296
* MAST, "Why Do Multi-Agent LLM Systems Fail?" — https://arxiv.org/abs/2503.13657
* Du et al., multi-agent debate — https://arxiv.org/abs/2305.14325 · Huang et al. — https://arxiv.org/abs/2310.01798 · "Stop Overvaluing Multi-Agent Debate" — https://arxiv.org/abs/2502.08788
* ALCE (citation evaluation) — https://arxiv.org/abs/2305.14627 · Zheng et al., LLM-as-judge — https://arxiv.org/abs/2306.05685 · Panickssery et al., self-preference — https://arxiv.org/abs/2404.13076
* LangChain multi-agent docs — https://docs.langchain.com/oss/python/langchain/multi-agent · OpenAI Agents SDK handoffs — https://openai.github.io/openai-agents-python/handoffs/
* Ollama FAQ (NUM_PARALLEL, context length) — https://docs.ollama.com/faq
