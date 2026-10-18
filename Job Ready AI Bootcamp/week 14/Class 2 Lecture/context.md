# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **Agentic Workflows with CrewAI — Roles, Multi-Agent Collaboration, Sequential vs Parallel Tasks**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে।

## কেন এই Topic দরকার?

একটা LLM-কে একবারে "research করো, report লেখো, ভুল ধরো" বললে সে তিনটা কাজই মাঝারি মানে করে। মানুষের টিমে যেমন কাজ ভাগ হয়, agent টিমেও তেমন ভাগ করলে প্রতিটা prompt ছোট ও focused হয়, আর output ভালো হয়। CrewAI সেই টিম বানানোর framework।

## শেখার সহজ Workflow

Problem বোঝা → Agent/Task/Crew-এর concept → Process (sequential, parallel, hierarchical) → Practical example → কখন কোনটা ব্যবহার করবেন।

> **Practical mindset:** Crew চালানোর আগে লিখে নিন কোন task কার output পড়বে। Output খারাপ হলে প্রথমে সেই dependency (context) দেখুন, model বদলানো তার পরে।

---

## 1. Topic: Agentic Workflows with CrewAI

এই ক্লাসে আমরা শিখব:

* **Agent** — `role`, `goal`, `backstory` দিয়ে সংজ্ঞায়িত একজন টিম-সদস্য।
* **Task** — `description` ও `expected_output` সহ একটা নির্দিষ্ট কাজ, যা একজন agent-কে দেওয়া হয়।
* **Crew** — agents + tasks + `process`; `kickoff()` দিয়ে চলে।
* **Process** — `sequential`, `hierarchical`, আর async task দিয়ে parallel execution।

---

## 2. Why It Is Related

Week 11-এ আমরা একটা LLM-কে prompt করা শিখেছি, Week 12-এ RAG দিয়ে তাকে তথ্য জোগানো। Agentic workflow হলো পরের ধাপ: একাধিক LLM-call একটা পরিকল্পিত কাঠামোতে বাঁধা, যেখানে এক ধাপের output পরের ধাপের input হয়।

```text
Topic → Researcher → (notes) → Writer → (draft) → Reviewer → report.md
```

Repo-র `Complete Agentic AI/Multi Agent Research Bot` একই ধারণা hand-rolled কোডে করেছে (Tavily search, Streamlit, cloud LLM)। এই ক্লাসের project তার framework-ভিত্তিক সংস্করণ: search ছাড়া, পুরোপুরি local, আর লক্ষ্য শুধু CrewAI-এর concept শেখা। এটা সেই project-এর copy নয়।

---

## 3. How It Works

### 3.1 Agent: role, goal, backstory

তিনটা field মিলে agent-এর system prompt তৈরি হয়। `role` বলে সে কে, `goal` বলে কী অর্জন করবে, `backstory` বলে কেমন মানসিকতায় কাজ করবে। Specific লিখলে output সরাসরি ভালো হয় ("Senior Research Specialist who flags uncertainty" বনাম শুধু "Researcher")।

```python
from crewai import Agent, LLM

llm = LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434", temperature=0.2)

researcher = Agent(
    role="Senior Research Specialist",
    goal="Collect accurate, well-organised facts about the topic.",
    backstory="You separate facts from speculation and flag anything uncertain.",
    llm=llm, allow_delegation=False,
)
```

### 3.2 Task এবং context

`expected_output` agent-কে বলে "শেষ হলে এটা দেখতে কেমন হবে"। `context=[অন্য task]` হলো dependency: ওই task-এর output এই task-এর prompt-এ যুক্ত হয়। `output_file` দিলে ফল সরাসরি ফাইলে লেখা হয়।

```python
from crewai import Task

research = Task(description="Research: vector databases", expected_output="15 bullet facts", agent=researcher)
write = Task(description="Write a Markdown report", expected_output="Full report", agent=writer, context=[research])
```

### 3.3 Crew এবং Process

```python
from crewai import Crew, Process

crew = Crew(agents=[researcher, writer], tasks=[research, write], process=Process.sequential)
result = crew.kickoff()      # result.raw = final text, result.tasks_output = প্রতিটা task-এর output
```

* **Sequential** — task গুলো তালিকার ক্রমে একটার পর একটা চলে। সবচেয়ে সহজ ও predictable।
* **Hierarchical** — `Process.hierarchical` আর `manager_llm=...` দিলে একজন manager কাজ ভাগ করে ও ফল যাচাই করে। Manager-কে agents list-এ দেওয়া হয় না।
* **Parallel (async)** — `Task(async_execution=True)` দিলে task background-এ শুরু হয়। যে task এদের output চায়, সে `context=[...]` দিয়ে অপেক্ষা করে। শেষ task async হতে পারে না।

### 3.4 Sequential বনাম Parallel বনাম Hierarchical

| Mode | কখন ব্যবহার করবেন | Trade-off |
|---|---|---|
| Sequential | ধাপগুলো একটা আরেকটার উপর নির্ভরশীল | ধীর, কিন্তু debug সহজ |
| Parallel (async) | স্বাধীন sub-task আছে (যেমন দুই দিকের research) | একটাই local model হলে বাস্তবে speed-up কম; বেশি memory |
| Hierarchical | কাজের ধাপ আগে থেকে জানা নেই | Manager-এর বাড়তি LLM-call; ছোট model-এ loop করতে পারে |

---

## 4. Details & Valid Points

### 4.1 Model Reference Table (Ollama, local)

আমরা **llama3.1:8b** বেছেছি, কারণ Crew-এ প্রতিটা agent নির্দিষ্ট format ধরে instruction মানতে হয়, আর 8B model সেটা মোটামুটি নির্ভরযোগ্যভাবে করে এবং laptop-এ চলে।

| Model | Use case | Trade-off |
|---|---|---|
| llama3.1:8b | Default; balanced | ~8GB RAM |
| mistral:7b | দ্রুত লেখা | Format মাঝে মাঝে ভাঙে |
| llama3.2:3b | খুব হালকা machine | Reviewer/manager role-এ দুর্বল |
| qwen2.5:7b | Structured output, multilingual | ধীর হতে পারে |
| llama3.1:70b | সেরা মান, hierarchical manager | ~40GB+ memory |

### 4.2 Multi-agent-এর সীমা

* প্রতিটা agent মানে আলাদা LLM-call, তাই latency ও cost বাড়ে। Single prompt যথেষ্ট হলে crew বানাবেন না।
* ছোট local model-এ agent-রা hallucinate করে বা format ভাঙে। Reviewer তাই research notes-এর বিপরীতে যাচাই করে।
* `verbose=True` দিয়ে agent-এর চিন্তা দেখা যায়; debug-এর সময় এটা সবচেয়ে কাজের।

---

## 5. Related Analogy: The Newsroom

একটা সংবাদ কক্ষে রিপোর্টার তথ্য আনে (Researcher), সাব-এডিটর লেখে (Writer), আর সম্পাদক ছাপার আগে ভুল কাটে (Reviewer)। Sequential হলো এই লাইন। Parallel হলো দুজন রিপোর্টার একসাথে দুই দিকের তথ্য আনছে। Hierarchical হলো বার্তা সম্পাদক নিজে ঠিক করছেন কে কী করবে।

---

## Achievement: Multi-Agent Report Team

`Class 2 Project/` ফোল্ডারে Researcher → Writer → Reviewer টিম আছে, যা `llama3.1:8b`-এ একটা topic-এর full report Markdown ফাইলে লেখে (`python app.py "Topic" --mode sequential|parallel|hierarchical`)। Ollama না থাকলে app পরিষ্কার message দিয়ে থামে, আর `pytest` Ollama ও network ছাড়াই চলে। Notebook: `crewai_lecture.ipynb`।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. Backstory বদলে Reviewer-কে "খুব নরম" করুন। Output কীভাবে বদলায়?
2. Parallel mode-এ writer-এর `context` থেকে একটা research task সরান। কী হারায়?
3. Sequential mode-এ `llama3.2:3b` দিয়ে চালিয়ে `llama3.1:8b`-এর সাথে report-এর মান তুলনা করুন।
4. কোন ধরনের কাজে hierarchical sequential-এর চেয়ে ভালো? একটা উদাহরণ ভাবুন।
5. চতুর্থ agent "Fact Checker" যোগ করুন এবং তার task wiring করুন।
