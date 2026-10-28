# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Job Readiness — GitHub Portfolio, AI Engineer Resume, LinkedIn আর Interview Strategy**। গত আঠারো সপ্তাহে আপনি ১৬টা Portfolio Project বানিয়েছেন। আজকের কাজ সেগুলো এমনভাবে সাজানো যাতে একজন recruiter ৩০ সেকেন্ডে বোঝেন আপনি কী পারেন, আর একজন interviewer ৪৫ মিনিট প্রশ্ন করেও আপনাকে আটকাতে না পারেন — কারণ আপনার লেখা প্রতিটা লাইন সত্যি, আর প্রতিটা সংখ্যা আপনি নিজে মেপেছেন।

Technical term English-এ রাখা হয়েছে, ব্যাখ্যা Bangla-তে। সাথে একটা `kit/` folder আছে — চারটা ready-to-fill file। সেগুলো English-এ, কারণ resume, README আর LinkedIn profile English-এই লেখা হয়।

## কেন এই Topic দরকার?

ভালো project থাকলেই interview আসে না। Ladders-এর eye-tracking study (2018) অনুযায়ী একজন recruiter প্রথমবার একটা resume-এ গড়ে মাত্র ~৭.৪ সেকেন্ড চোখ বোলান — সেটা পড়া না, ছাঁকা। GitHub profile-এও একই ছবি: pinned repo-র নাম, README-র প্রথম স্ক্রিন, একটা demo GIF — এর বাইরে খুব কম মানুষ যান। উল্টো দিকে, interview-তে resume-এর প্রতিটা সংখ্যা নিয়ে প্রশ্ন হয়: "এই 82% কোথায় মাপলেন? কোন hardware? কয়টা task?" বানানো বা ধার করা সংখ্যা ঠিক এখানেই ধরা পড়ে।

তাই এই ক্লাসের দুটো নিয়ম: **(১) ছাঁকনি পার হওয়ার মতো করে সাজান, (২) এমন কিছু লিখবেন না যা interview-তে নিজে ব্যাখ্যা করতে পারবেন না।**

## শেখার সহজ Workflow

Recruiter কীভাবে skim করেন → GitHub profile README আর ছয়টা pinned repo → per-project README anatomy → commit hygiene আর secrets → demo GIF → Resume (ATS, এক পাতা, XYZ bullet, সৎ metric) → কী দাবি করবেন না → LinkedIn (headline, About, Featured, Skills, project post) → Interview loop-এর ধাপ → STAR দিয়ে project-এর গল্প → Junior-level RAG/agent system design → Take-home → "Why local models?"

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটা কে পড়বে? কত সেকেন্ডে? পড়ে কী সিদ্ধান্ত নেবে?**

> **Practical mindset:** Resume-এ প্রতিটা bullet লেখার পর নিজেকে জিজ্ঞেস করুন — interviewer যদি বলেন "show me", আমি কি ৩০ সেকেন্ডে repo-তে সেই code বা সেই measurement দেখাতে পারব? না পারলে bullet বদলান।

---

## 1. Topic: Job Readiness for an AI Engineer

এই ক্লাসে আমরা শিখব:

* **GitHub portfolio** — profile README, pinned repo, প্রতিটা project-এর README-র কাঠামো, পরিষ্কার commit history, secret-মুক্ত repo, demo GIF।
* **AI Engineer resume** — ATS-এর নিয়ম, এক পাতা, XYZ formula-য় impact bullet, নিজের মাপা সংখ্যা, আর কী কখনো দাবি করবেন না।
* **LinkedIn** — headline formula, About, Featured, Skills, আর প্রথম project post।
* **Interview strategy** — AI engineer loop-এর ধাপ, STAR method-এ project-এর গল্প, junior-level system design, take-home, আর "why local models?"-এর উত্তর।

**Achievement (Class 40):** প্রত্যেকে বের হবেন একটা finalized AI Engineer resume, একটা polished LinkedIn profile আর ১৬টা project-সহ একটা GitHub portfolio নিয়ে। Kit:

| File | কী আছে |
| --- | --- |
| `kit/portfolio_map.md` | ১৬টা project → course folder → এক লাইনের pitch → key skill → pin order |
| `kit/portfolio_README_template.md` | GitHub profile README template + per-project README template |
| `kit/resume_template.md` | এক পাতার resume template + ১৬টা project-এর bullet bank |
| `kit/linkedin_checklist.md` | Headline formula, About template, Featured, Skills, first-post template |

---

## 2. Why It Is Related

পুরো course একটা গল্প বলে, আর আজ সেই গল্প লিখে ফেলার দিন। Week 2-এর NumPy similarity engine থেকে Week 12-এর RAG, Week 13-এর tool-calling agent, Week 14-এর LangGraph আর CrewAI, Week 15-এর FastAPI service, Week 16-এর market intelligence team আর Week 17-এর healthcare assistant — প্রতিটা ধাপে একটা জিনিস বারবার এসেছে: **model-এর চারপাশের code দিয়ে একটা দুর্বল local model-কে নির্ভরযোগ্য বানানো, আর সেটা মেপে দেখানো।** Resume-এ এটাই আপনার positioning: "আমি LLM দিয়ে এমন system বানাই যেখানে ভুল ধরা পড়ে, test হয়, আর মাপা যায়।" প্রতিটা course README-র "Known Limits" আর "Live Results" section-ই interview-তে আপনার সবচেয়ে শক্তিশালী উত্তরের কাঁচামাল।

---

## 3. How It Works

### 3.1 Recruiter-এর ৩০ সেকেন্ড — কে কী দেখেন

একটা application সাধারণত তিনটা ছাঁকনি পার হয়:

| ধাপ | কে | কতক্ষণ | কী দেখে |
| --- | --- | --- | --- |
| ATS | Software | স্বয়ংক্রিয় | Resume parse হয় কিনা; job description-এর keyword আছে কিনা |
| Recruiter screen | সাধারণত non-technical মানুষ | Resume-এ কয়েক সেকেন্ড, GitHub-এ ~৩০ সেকেন্ড | Title, skill-এর নাম, project-এর নাম, পরিষ্কার layout |
| Hiring manager / engineer | Technical মানুষ | কয়েক মিনিট | একটা repo খুলে README, architecture, test, code-এর মান |

GitHub profile-এ recruiter-এর ৩০ সেকেন্ড মোটামুটি এভাবে কাটে: নাম আর bio → profile README-র প্রথম স্ক্রিন → ছয়টা pinned repo-র নাম আর description → একটা repo খোলা → সেই README-র প্রথম স্ক্রিন (title, এক লাইনের pitch, GIF) → বের হয়ে যাওয়া। তাই **প্রথম স্ক্রিনেই উত্তর থাকতে হবে: এটা কী, কেন দরকারি, কোন tech।** Contribution graph-এর সবুজ ঘর বা star count এখানে খুব কম ওজন পায়; পরিষ্কার README অনেক বেশি পায়।

### 3.2 GitHub portfolio — profile README আর pinned repo

**Profile README:** আপনার username-এর নামেই একটা public repo বানান (username `rahim-ai` হলে repo-র নামও `rahim-ai`), root-এ একটা `README.md` রাখুন — GitHub সেটা profile page-এর একদম উপরে দেখায়। Repo private হলে বা নাম না মিললে দেখায় না। এখানে থাকবে: এক লাইনে আপনি কে, তিনটা featured project, বাকি ১৬টার একটা টেবিল, stack, আর যোগাযোগ। Template: `kit/portfolio_README_template.md`।

**Pinned repo:** Profile-এ সর্বোচ্চ **ছয়টা** repo বা gist pin করা যায়, আর pin করলে "Popular repositories"-এর জায়গায় সেগুলোই দেখায়। ১৬টার মধ্যে ছয়টা বাছতে হবে — বাকিগুলো profile README-র টেবিলে। কোন ছয়টা, সেটা নির্ভর করে কোন role-এ apply করছেন তার উপর (`kit/portfolio_map.md`-এ তিনটা set আছে)। LLM/AI engineer role-এর default:

1. #16 Local Knowledge-Grounded Healthcare Assistant — RAG + guardrail + eval + API
2. #14 Production-Ready RAG API Service
3. #15 Offline Market Intelligence Team
4. #13 Collaborative Multi-Agent Research Team
5. #12 Self-Correcting LLM Coder
6. #11 Deterministic Function-Calling Travel Agent

**এক project = এক repo।** Course-এর সব কিছু একটা বড় repo-তে, কিন্তু recruiter একটা repo খুলে একটাই জিনিস দেখতে চান, আর pin হয় repo ধরে। প্রতিটা project ADR 0002 অনুযায়ী self-contained — নিজের `requirements.txt`, নিজের README, অন্য week-এর কোনো import নেই — তাই folder কপি করে নতুন repo বানালেই চলে। Repo-র নাম হবে বর্ণনামূলক: `local-rag-api-fastapi`, `week15-class1` না। Description field-এ এক লাইন আর Topics-এ কয়েকটা tag (`rag`, `fastapi`, `ollama`) দিন — pinned card-এ এগুলোই দেখা যায়।

### 3.3 Per-project README anatomy

Course-এর README-গুলোর কাঠামোই template, কারণ সেগুলো ঠিক এই প্রশ্নগুলোর উত্তর দেয়, এই ক্রমে:

| Section | কোন প্রশ্নের উত্তর | কে পড়েন |
| --- | --- | --- |
| Title + এক লাইনের pitch | এটা কী? | Recruiter |
| Demo GIF / screenshot | সত্যিই চলে? | Recruiter |
| Badge (Python, framework, tests) | কোন stack? Test আছে? | Recruiter |
| Overview: problem আর কেন এই approach | কেন বানানো? | Hiring manager |
| Features | কী কী পারে? | সবাই |
| Quick Start | আমি কি ৫ মিনিটে চালাতে পারব? | Engineer |
| Project Structure | Code কোথায়? | Engineer |
| Architecture (ASCII diagram) | Data কোন পথে যায়? | Engineer |
| Configuration টেবিল | কোন knob, কেন সেই default? | Engineer |
| Model choice টেবিল | কেন এই model? | Engineer |
| Results / What to measure | কতটা ভালো? কোথায় মাপা? | Engineer |
| Design decisions | কোন trade-off নেওয়া হয়েছে? | Engineer |
| Known Limits | কোথায় ভাঙে? | Senior engineer — সবচেয়ে বেশি মুগ্ধ হন এখানেই |
| Credits | কোনটা আপনার, কোনটা course-এর? | সবাই |

Course README-র বাইরে দুটো জিনিস যোগ করবেন: **(১) Credits section** — "Built during Job Ready AI Bootcamp (Week 15). My additions: ..." — আর **(২) Results-এ আপনার নিজের মেশিনের সংখ্যা**, hardware আর version সহ। Course README-র "Live Results" টেবিল author-এর মেশিনের; আপনার repo-তে সেটা হুবহু রাখবেন না। নিজে চালিয়ে নিজের সংখ্যা বসান, অথবা "Reference run (course author)" শিরোনামে স্পষ্ট আলাদা রাখুন।

### 3.4 Commit hygiene আর secrets

Engineer repo খুললে history-ও দেখেন। একটা commit "final", তার পরেরটা "final2", তারপর "fix" — এটা অভ্যাসের সংকেত দেয়।

* **ছোট, অর্থপূর্ণ commit:** `feat: add distance guardrail before LLM call`, `test: cover 503 on queue timeout`। প্রতিটা commit একটা কাজ।
* **যা কখনো commit হবে না:** `venv/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `data/chroma_db/`, `output/`, `.env`, model weight। Course project-গুলোর `.gitignore` কপি করুন, `git add`-এর **আগে**। GitHub ১০০ MiB-এর বেশি বড় file push আটকায় আর ৫০ MiB-এর বেশিতে সতর্ক করে — `.keras`, `.pkl`, `.safetensors` repo-তে না রেখে README-তে লিখুন কীভাবে বানাতে বা নামাতে হয়।
* **Secret:** API key, token, password কখনো code-এ না। Week 15-এর `settings.py`-র মতো env var থেকে পড়ুন; repo-তে থাকবে শুধু `.env.example`, খালি value সহ। GitHub-এর **push protection for users** default-এ চালু থাকে আর public repo-তে চেনা ধরনের secret থাকলে push আটকায় — কিন্তু সব ধরনের secret চেনে না, তাই এটা শেষ বেড়া, প্রথম না। Push-এর আগে `gitleaks`-এর মতো scanner চালাতে পারেন।
* **ভুল করে secret push হয়ে গেলে:** **প্রথমে key rotate বা revoke করুন**, তারপর history পরিষ্কার করুন (`git filter-repo`)। History মুছলেও যে কেউ আগেই clone বা fork করেছে, তার কাছে key থেকে যায় — তাই rotate-ই আসল fix।

এই course-এর একটা সুবিধা: সব project local model-এ চলে, তাই cloud API key লাগেই না। Week 15-এর `RAG_API_KEY`-এর মতো জিনিস env var-এ থাকে, code-এ না।

### 3.5 Demo GIF আর screenshot

একটা ১০-২০ সেকেন্ডের GIF লম্বা README-র চেয়ে দ্রুত বলে "এটা চলে"। নিয়ম:

* একটা কাজ দেখান, পুরো app না: প্রশ্ন → citation-সহ উত্তর; বা red-flag প্রশ্ন → escalation message।
* README-র একদম উপরে, title-এর ঠিক নিচে।
* File ছোট রাখুন (কয়েক MB), `docs/demo.gif`-এ রেখে relative path দিন। Windows-এ ScreenToGif, macOS-এ Kap ভালো কাজ করে; CLI project-এর জন্য terminal recording।
* LLM-এর উত্তর ধীর হলে GIF কেটে ছোট করা চলে, কিন্তু README-তে আসল latency লিখে দিন। দ্রুত দেখানো আর দ্রুত দাবি করা এক জিনিস না।
* Streamlit app-এর screenshot-এ অর্থবহ sample data দেখান, ফাঁকা স্ক্রিন বা lorem ipsum না।

### 3.6 Resume — ATS আর এক পাতা

**ATS** (Applicant Tracking System) resume-কে text-এ ভেঙে field-এ সাজায়, তারপর recruiter keyword দিয়ে খোঁজেন। ATS জটিল layout ভালো বোঝে না, তাই:

* **এক column**, সাধারণ font, স্পষ্ট heading: `Summary`, `Skills`, `Projects`, `Experience`, `Education`।
* জরুরি তথ্য table, text box, header/footer, ছবি বা icon-এর ভেতরে রাখবেন না — অনেক parser সেগুলো হারিয়ে ফেলে।
* **Text-based PDF** (Word বা Google Docs থেকে export; scan করা না)। Job post `.docx` চাইলে সেটাই দিন। File-এর নাম `Rahim-Uddin-AI-Engineer.pdf`।
* Keyword: job description-এ যে শব্দ আছে আর **আপনি সত্যিই ব্যবহার করেছেন**, সেটা একই বানানে লিখুন — "RAG", "FastAPI", "LangGraph", "vector database"। যা ব্যবহার করেননি, সেটা keyword হিসেবেও না।
* ছবি: international বা remote role-এ দেবেন না; দেশের অনেক company ছবি চায় — job post দেখে ঠিক করুন।

**Junior-দের জন্য এক পাতা।** কারণ সোজা: recruiter-এর হাতে কয়েক সেকেন্ড, আর দুই পাতায় সেরা তিনটা project চাপা পড়ে যায়। ১৬টা project-এর সবগুলো resume-এ যাবে না — **৩-৫টা**, job-এর সাথে মিলিয়ে। বাকিগুলোর জন্য একটা লাইনই যথেষ্ট: "16 end-to-end AI projects: github.com/rahim-ai"।

### 3.7 Impact bullet — XYZ formula আর সৎ metric

Google-এর তৎকালীন People Operations প্রধান Laszlo Bock-এর formula: **"Accomplished [X] as measured by [Y], by doing [Z]"** — কী অর্জন করলেন, কীভাবে মাপা হলো, কী করে করলেন। দায়িত্বের তালিকা না, ফলাফল।

দুর্বল: `Worked on a RAG chatbot using ChromaDB.`

শক্তিশালী: `Built a citation-grounded PDF Q&A service (FastAPI, ChromaDB, llama3.1:8b) that skips the LLM when retrieval distance exceeds a calibrated threshold; answered [X]/[Y] on-topic test questions with page citations and returned "not found" for [Z]/[Z] off-topic ones.`

AI engineer-এর bullet-এ সংখ্যা আসে এসব জায়গা থেকে: accuracy বা F1 (classic ML), pass@1 বনাম repair-এর পরে (coding agent), golden-task success rate (agent), latency p50/p95, TTFT, req/s (serving), citation precision (RAG), test-এর সংখ্যা, token খরচ। **নিয়ম:** সংখ্যা তখনই লিখবেন যখন (১) আপনি নিজে মেপেছেন, (২) কোথায়-কীভাবে মাপা সেটা বলতে পারবেন (hardware, model version, dataset-এর আকার), আর (৩) সংখ্যাটার প্রেক্ষিত আছে। "7/7" লেখার আগে মনে রাখুন interviewer জিজ্ঞেস করবেন "মাত্র ৭টা প্রশ্ন?" — আর উত্তর হবে: "হ্যাঁ, এটা smoke eval, benchmark না; এর সীমা হলো..." — এই সততাই জেতায়।

`kit/resume_template.md`-এর bullet bank-এ প্রতিটা project-এর জন্য ২-৩টা bullet আছে, `[X]` placeholder সহ। যেখানে course README কোনো সংখ্যা report করেছে, সেটা "Course reference" হিসেবে পাশে লেখা — যাতে জানেন আপনার সংখ্যা কোন পরিসরে আসার কথা। কিন্তু resume-এ যাবে **শুধু আপনার নিজের run-এর সংখ্যা।**

### 3.8 কী দাবি করবেন না

এখানে একটা ভুল পুরো interview শেষ করে দিতে পারে।

| লোভনীয় দাবি | সমস্যা | সৎ বিকল্প |
| --- | --- | --- |
| "Deployed to production serving 1,000+ users" | কোনো user ছিল না | "Production-style service: health/readiness probes, backpressure (503 + Retry-After), Docker image, [X] tests" |
| "Reduced latency by 70%" — baseline ছাড়া | কীসের তুলনায়? | "Startup warmup cut cold first-request latency from [A] s to [B] s on my machine" |
| LLM project-এ "Built an LLM" বা "Trained a model" | Model ব্যবহার করেছেন, train করেননি | "Built an agent / RAG system on llama3.1:8b" |
| "Fine-tuned" | এই course-এর LLM project-এ fine-tuning নেই | যা করেছেন: prompting, retrieval, guardrail, tool design |
| "HIPAA-compliant healthcare assistant" | PII redaction মানে compliance না; কোনো audit হয়নি | "PII redaction, deterministic red-flag escalation and abstention guardrails" |
| #15-এ "Real-time market data" | এটা Offline Snapshot | "Agents run fully offline on a bundled snapshot of prices and news" |
| প্রেক্ষিত ছাড়া "99% accuracy" | MNIST-এ 99% সাধারণ ঘটনা | "[X]% test accuracy; compared 3 CNN variants and quantified the overfitting gap" |
| Course author-এর benchmark নিজের নামে | "আপনার GPU কী ছিল?" — উত্তর নেই | নিজে চালিয়ে নিজের সংখ্যা |
| ২০টা tool-এ "Expert" | যেকোনো একটায় গভীর প্রশ্নে ধরা | যা দিয়ে কিছু বানিয়েছেন, শুধু সেগুলো |
| Solo project-এ "Led a team" | Team ছিল না | "Designed and built" |

আর একটা বড় নিয়ম: **course-এর reference code-কে নিজের মৌলিক আবিষ্কার বলবেন না।** বলুন "Built during Job Ready AI Bootcamp", তারপর আপনি কী বুঝেছেন, কী বদলেছেন, কী মেপেছেন। "কোন অংশ আপনি নিজে লিখেছেন?" প্রশ্নটা আসবেই — সবচেয়ে ভালো উত্তর একটা নির্দিষ্ট extension যা আপনি নিজে করেছেন: একটা নতুন test, একটা নতুন guardrail, একটা benchmark। তাই pinned ছয়টা project-এর প্রতিটায় অন্তত একটা নিজের extension যোগ করুন।

### 3.9 LinkedIn — headline, About, Featured, Skills, project post

LinkedIn-কে resume-এর কপি ভাববেন না। এটা একটা search engine — recruiter keyword দিয়ে খোঁজেন, তারপর headline দেখে click করেন।

* **Headline:** ২২০ character পর্যন্ত লেখা যায়, কিন্তু search result আর comment-এ প্রথম ~৬০-৭০টাই দেখায়। তাই role আগে, তারপর প্রমাণ: `AI Engineer | RAG, Agents & LLM APIs on local models (Ollama, FastAPI, LangGraph) | 16 open-source projects`
* **About:** ২,৬০০ character পর্যন্ত, কিন্তু "see more"-এর আগে প্রথম দুই-তিন লাইনই দেখা যায়। প্রথম লাইনেই আপনি কী বানান; তারপর প্রমাণ হিসেবে ৩টা project, stack, কী খুঁজছেন, যোগাযোগ।
* **Featured:** profile README, সেরা repo, project write-up post, demo video — এই ৩-৪টা।
* **Skills:** যা দিয়ে আসলে কিছু বানিয়েছেন — Python, RAG, LLM, FastAPI, LangGraph, CrewAI, ChromaDB, scikit-learn, TensorFlow/Keras, OpenCV, Docker। সবচেয়ে প্রাসঙ্গিকগুলো উপরে।
* **Project post:** একটা project-এর গল্প — সমস্যা, একটা আকর্ষণীয় পর্যবেক্ষণ (যেমন "llama3.1:8b CrewAI-র ভেতরে search tool একবারও ডাকেনি, নিজেই result বানিয়েছে"), কীভাবে সমাধান করলেন, একটা GIF, repo link। "Excited to share..." দিয়ে না, পর্যবেক্ষণ দিয়ে শুরু করুন।

পুরো checklist আর first-post template: `kit/linkedin_checklist.md`।

### 3.10 AI engineer interview loop — ধাপগুলো

Company ভেদে বদলায়, তবে সাধারণ ছবি এরকম:

| ধাপ | কী হয় | প্রস্তুতি |
| --- | --- | --- |
| Recruiter screen (১৫-৩০ মিনিট) | Background, কেন এই role, salary, কবে join | ৯০ সেকেন্ডের self-intro; একটা project এক লাইনে |
| Technical screen | Python coding (list, dict, string, ছোট algorithm) বা practical LLM task | Python fundamentals, edge case, test লেখা |
| Take-home বা live build | ছোট RAG/agent/API, ২-৮ ঘণ্টা | 3.13 দেখুন |
| Project deep-dive | আপনার একটা project নিয়ে ৩০-৪৫ মিনিট | STAR গল্প (3.11), architecture আঁকা, প্রতিটা সংখ্যার উৎস |
| ML/LLM fundamentals | Embedding, chunking, temperature, overfitting, precision/recall, eval | আগের lecture-গুলোর Interview Q&A |
| System design | "একটা company-র HR policy bot design করুন" | 3.12-এর কাঠামো |
| Behavioral / hiring manager | Conflict, ব্যর্থতা, শেখা, team | ৩-৪টা STAR গল্প, অন্তত একটা ব্যর্থতার |

Startup-এ প্রায়ই take-home আর project deep-dive-ই মূল; বড় company-তে coding round বেশি ওজন পায়।

### 3.11 STAR দিয়ে project-এর গল্প

**STAR**-এর চারটা অংশ: **Situation** — কোন প্রেক্ষাপটে কাজটা; **Task** — আপনার দায়িত্ব কী ছিল; **Action** — আপনি নিজে কী করলেন (সবচেয়ে বড় অংশ); **Result** — কী হলো, মাপা ফল, আর কী শিখলেন। সব মিলিয়ে ২ মিনিটের মধ্যে।

উদাহরণ — #11 Travel Agent:

* **Situation:** একটা local 8B model একা নির্ভরযোগ্য agent না — course README অনুযায়ী BFCL v4-এ llama3.1:8b-এর multi-turn accuracy ~11%।
* **Task:** এমন একটা travel agent বানানো যেটা local flight database খোঁজে আর মানুষের approval ছাড়া কখনো booking করে না।
* **Action:** Framework ছাড়া নিজের ReAct loop; প্রতিটা tool call registry আর Pydantic (`extra="forbid"`) দিয়ে validate; typed stop reason (`final`, `loop`, `timeout`...); repeat guard; `book_flight`-এর আগে code-এ approval gate। তারপর `ollama show --template` পড়ে দেখলাম tool result আসার পর model আর tool তালিকা দেখে না — তাই `final_answer` tool আর একটা nudge যোগ করলাম। ৮টা golden task-এর eval harness বানালাম।
* **Result:** আমার run-এ native mode [X]/8 golden task পেরিয়েছে (course reference run: native 6/8, text 4/8)। শিক্ষা: শুধু nudge দিলে, `final_answer` ছাড়া, native mode reference run-এ 0/8-এ নেমে গিয়েছিল — ছোট model-এ প্রতিটা বদল মাপতে হয়।

Result-এ **negative result লুকাবেন না**, সেটাই প্রায়ই গল্পের সবচেয়ে ভালো অংশ। #13-এর গল্পে: "একই passage আর guardrail দিয়ে single-agent baseline ~৭ গুণ কম token-এ একই citation precision দিয়েছে; crew-এর লাভ শুধু কম uncited বাক্য।" এটা শুনলে interviewer বোঝেন আপনি hype না, measurement মানেন।

প্রতিটা pinned project-এর জন্য একটা STAR লিখে রাখুন, আর জোরে বলে সময় মাপুন।

### 3.12 Junior-level system design — RAG আর agent

Junior-এর কাছে কেউ planet-scale design চান না। চান পরিষ্কার কাঠামো, সঠিক প্রশ্ন, আর trade-off বোঝা। ধাপগুলো:

1. **Requirement জিজ্ঞেস করুন:** কতজন user, দিনে কত প্রশ্ন, কত latency সহ্য, data কত বড়, data কি machine-এর বাইরে যেতে পারবে, ভুল উত্তরের দাম কত।
2. **Back-of-envelope হিসাব:** ৫০০ কর্মী, দিনে মাথাপিছু ২টা প্রশ্ন = দিনে ১,০০০টা। ৮ ঘণ্টার অফিসে গড় ≈ ০.০৩৫ req/s; peak-এ ১০ গুণ ধরলে ≈ ০.৩৫ req/s। Week 15-এর course measurement-এ একটা Ollama আর একটা GPU ছোট উত্তরে ~০.৬ req/s-এ আটকেছিল — তাই শুরুতে একটা replica যথেষ্ট, কিন্তু headroom কম; queue আর 503 লাগবে।
3. **Ingestion:** PDF → page-aware chunk → embedding → vector DB; idempotent id (hash), metadata (source, page)।
4. **Query path:** embed → metadata filter সহ top-k retrieve → **guardrail** (distance threshold পার না হলে LLM ডাকাই হয় না, "not found") → prompt (passage আগে, প্রশ্ন শেষে, যাতে prefix cache কাজে লাগে) → LLM → citation।
5. **Serving:** FastAPI, `async` endpoint কিন্তু blocking কাজ threadpool-এ, concurrency gate, timeout, degraded mode, `/health` বনাম `/ready`।
6. **Evaluation:** ছোট golden set (on-topic, off-topic, adversarial), retrieval hit rate, citation precision, abstention rate।
7. **Monitoring:** latency (p50/p95, TTFT), token, error, quality flag; SLO।
8. **Scale বাড়লে কী বদলাবে:** replica আর load balancer, shared cache (Redis), continuous batching server (vLLM), tenant-ভিত্তিক collection, per-key rate limit।

**Agent design-এ** আরও চারটা জিনিস বলবেন: (১) tool schema আর validation — এটাই trust boundary; (২) loop-এর সীমা আর typed stop reason; (৩) side effect (booking, email, payment)-এর আগে human approval, code-এ, model-এ না; (৪) যা deterministic হতে পারে তা code-এ — #15-এর Risk Score code হিসাব করে, agent শুধু ব্যাখ্যা করে; #16-এর Red-Flag Escalation retrieval-এর আগেই rule দিয়ে ঠিক হয়, model সেটা লেখে না বা বদলাতে পারে না।

Whiteboard-এ আঁকার অনুশীলন করুন — course README-গুলোর Architecture section ঠিক সেই আকারের।

### 3.13 Take-home tips

* **শুরুতে প্রশ্ন করুন:** অস্পষ্ট কিছু থাকলে email-এ জিজ্ঞেস করুন; উত্তর না এলে README-তে assumption লিখে এগোন।
* **Timebox মানুন।** ৪ ঘণ্টা বললে ৪ ঘণ্টা — তিন দিন লাগিয়ে "perfect" করা প্রায়ই নেতিবাচক সংকেত।
* **আগে সবচেয়ে সরল working version**, তারপর উন্নতি। যা চাওয়া হয়নি, তা বানাবেন না।
* **README-ই আপনার কণ্ঠস্বর:** কীভাবে চালাতে হয়, design decision, trade-off, "আরও সময় পেলে কী করতাম", known limits।
* **Test:** অন্তত core logic-এর কয়েকটা test, আর LLM ছাড়া চলার মতো fake — course-এর `FakeBackend` বা `FakeLLM(BaseLLM)` pattern।
* **Clean clone থেকে চালিয়ে দেখুন** — নতুন venv, README-র Quick Start হুবহু।
* **Secret আর বড় file নেই**, `.gitignore` আছে।
* **AI assistant ব্যবহার:** company-র policy মানুন; ব্যবহার করলে জানিয়ে দিন, আর প্রতিটা লাইন ব্যাখ্যা করার মতো বুঝে নিন — পরের round-এ সেই code নিয়েই প্রশ্ন হবে।

### 3.14 "Why local models?" — উত্তরের কাঠামো

এই প্রশ্ন আসবেই, কারণ আপনার ১৬টা project-ই local। উত্তর তিন ভাগে:

1. **কেন local:** data machine-এর বাইরে যায় না (healthcare-এর PII, finance-এর research), request-প্রতি খরচ নেই, offline চলে, model version pin করা যায় তাই behaviour হঠাৎ বদলায় না।
2. **সৎ trade-off:** frontier cloud model-এর চেয়ে quality কম (BFCL-এ multi-turn ~11%); hardware লাগে; throughput কম (course measurement-এ একটা Ollama ~০.৬ req/s)। তাই নির্ভরযোগ্যতা এসেছে model-এর চারপাশের code থেকে — validation, guardrail, deterministic tool, eval।
3. **কখন cloud বেছে নেবেন:** data sensitive না, quality-ই মূল, traffic অনিয়মিত, team-এর GPU চালানোর সময় নেই। Architecture এমন যে LLM client একটা module-এ থাকে — বদলাতে হয় এক জায়গায়।

শেষ লাইন: "Local বনাম cloud একটা requirement-এর সিদ্ধান্ত, বিশ্বাসের না।"

Course-এর LLM বাছাই, interview-এ যেভাবে তুলনা করবেন:

| Model | Use Case | Advantage | Limitation |
| --- | --- | --- | --- |
| **`llama3.1:8b`** via Ollama (course standard, ADR 0006) | Local RAG, agent, tool calling, generation | Data machine-এ থাকে; ১৬ GB RAM-এ চলে; native tool calling; version pin করা যায় | Multi-turn tool use দুর্বল (BFCL ~11%); একটা server-এ একবারে একটা stream |
| `llama3.2:3b` via Ollama | ৮ GB RAM-এর machine, সহজ extraction বা classification | ছোট, দ্রুত | জটিল instruction আর tool call-এ আরও দুর্বল |
| `mistral:7b` via Ollama | Local generation, structured task | একই আকার, কিছু benchmark-এ দ্রুত | Fine-tune ecosystem ছোট; এই course-এ যাচাই করা হয়নি |
| Cloud API (GPT / Claude) | সর্বোচ্চ quality, অনিয়মিত traffic | State-of-the-art quality, নিজের hardware লাগে না | Data বাইরে যায়; request-প্রতি খরচ; provider model বদলালে behaviour বদলায় |
| একই open model, vLLM server-এ | অনেক concurrent user | Continuous batching-এ throughput অনেক বেশি | GPU server আর operations-এর কাজ লাগে |

---

## 4. Details & Valid Points

### 4.1 ১৬টা project → তিনটা track

Recruiter-এর কাছে ১৬টা আলাদা project একটা লম্বা তালিকা; তিনটা track একটা গল্প:

| Track | Project | যা প্রমাণ করে |
| --- | --- | --- |
| ML & Data foundations | #01 Vector Engine, #02 Real Estate Valuator, #03 Spam & Intent, #04 Segmentation | NumPy, scikit-learn, evaluation, explainability |
| Deep Learning, CV & NLP | #05 MNIST, #06 Privacy Shield, #07 CV Matcher, #08 Feedback Analyzer | Keras CNN, OpenCV, embedding, transformer pipeline |
| LLM Systems & Agents | #09 Brand Generator থেকে #16 Healthcare Assistant | Prompting, RAG, tool calling, LangGraph, CrewAI, FastAPI, guardrail, eval |

AI engineer role-এ তৃতীয় track সামনে থাকে; প্রথম দুটো দেখায় আপনার ভিত্তি আছে — model কেন কাজ করে বোঝেন, শুধু API ডাকেন না।

### 4.2 Numbers policy — reference বনাম আপনার measurement

Course README-তে তিন ধরনের সংখ্যা আছে:

* **Deterministic, reproducible** — যেমন #02-এর R² 0.8754 (fixed CSV আর সব জায়গায় `random_state=42`)। নিজে চালালেও একই আসবে; তবু লিখবেন "on a 5,000-row synthetic dataset" — data synthetic, সেটা লুকানো যাবে না। #03, #04-এর data-ও synthetic।
* **Hardware আর version নির্ভর** — #14-এর latency, #11-এর 6/8, #12-এর 9/11, #13-এর token। এগুলো author-এর মেশিনের। নিজে চালান; আলাদা এলে নিজেরটাই লিখুন, আর README-তে কারণ ব্যাখ্যা করুন (ভিন্ন GPU, ভিন্ন Ollama version) — এটা নিজেই একটা ভালো interview আলোচনা।
* **কোনো reference সংখ্যা নেই** — #15 আর #16 (ADR 0008: এই দুই সপ্তাহের project static verification-এ, course-এ live run করা হয়নি)। #05-এর README-তে শুধু "typical" সংখ্যা আছে, মাপা না। এগুলোয় resume-এর প্রতিটা সংখ্যা আপনার নিজের মেশিনের।

### 4.3 Course folder থেকে নিজের repo

* প্রতিটা project folder নতুন repo-তে কপি করুন, `.gitignore` সাথে নিয়ে; `__pycache__/` কপি করবেন না।
* README-র Credits section: "Built during Job Ready AI Bootcamp (Week N, Class M). My changes: ..."।
* README-র ভাষা: course README Bangla-English মিশ্র; public repo-র README English-এ লিখুন, কারণ recruiter পড়বেন। চাইলে নিচে একটা Bangla section রাখুন।
* অন্তত একটা নিজের extension: নতুন eval case, নিজের corpus, নতুন guardrail, Dockerfile, অথবা একটা CI workflow যেটা প্রতিটা push-এ `pytest` চালায় — এমন কিছু যা নিজে ব্যাখ্যা করতে পারবেন।
* যা নেই, সেটাই সবচেয়ে সহজ extension: #06 Privacy Shield-এ কোনো test নেই; #01-এ Streamlit view আর Euclidean distance নেই (`projects.md` এগুলো বলে, কিন্তু course build একটা CLI, cosine দিয়ে); #02-এ `pytest` suite নেই, শুধু একটা smoke test।

### 4.4 Application cadence

* প্রতিটা application-এ resume-এর Projects অংশ job description অনুযায়ী বদলান — RAG-ভারী role-এ #16/#14/#10, agent role-এ #11/#12/#13/#15, ML/CV role-এ #05/#06/#02।
* Referral, অথবা সরাসরি একজন engineer-কে ছোট আর নির্দিষ্ট message, cold application-এর চেয়ে বেশি সাড়া পায় — repo link দিন, তিন লাইনের বেশি না।
* প্রতিটা interview-এর পর দশ মিনিটে লিখে রাখুন কী প্রশ্ন এসেছিল আর কোথায় আটকেছেন — পরের সপ্তাহের প্রস্তুতি সেখান থেকেই।

---

## 5. Related Analogy: The Restaurant

> আপনার portfolio একটা রেস্তোরাঁ, আর প্রতিটা project একটা পদ। রান্না ভালো — কিন্তু মানুষ ঢোকেন সাইনবোর্ড আর মেনু দেখে।

| Concept | Restaurant Analogy |
| --- | --- |
| **LinkedIn headline** | রাস্তার সাইনবোর্ড — পথচারী দুই সেকেন্ডে ঠিক করেন ঢুকবেন কিনা |
| **Resume** | এক পাতার মেনু — সব রান্না না, আজকের সেরা ৪-৫টা পদ |
| **ATS** | দরজার দারোয়ান যিনি শুধু তালিকা মেলান — নাম ঠিক বানানে না থাকলে ঢুকতে দেন না |
| **GitHub profile README** | কাচের দেয়ালের ওপারে খোলা রান্নাঘর — পরিষ্কার হলে বিশ্বাস জন্মায় |
| **Pinned repo** | Chef's special — ছয়টা পদ যার জন্য আপনি পরিচিত হতে চান |
| **Demo GIF** | কাউন্টারে সাজানো আসল খাবার — বিজ্ঞাপনের ছবি না |
| **Known Limits** | "এই পদে বাদাম আছে" সতর্কবার্তা — সৎ রেস্তোরাঁ লুকায় না, আর অভিজ্ঞ ক্রেতা ঠিক এটাই খোঁজেন |
| **বানানো metric** | মেনুতে "শহরের সেরা বিরিয়ানি" — প্রথম কামড়েই ধরা পড়ে |
| **Project deep-dive** | Food critic রান্নাঘরে ঢুকে জিজ্ঞেস করেন — "এই মশলা কেন? কত তাপে? কতক্ষণ?" |

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
| --- | --- | --- |
| ১৬টা project এক repo-তে | Pin করা যায় না, recruiter হারিয়ে যান | এক project = এক repo, বর্ণনামূলক নাম |
| Repo-র নাম `week15-class1` | কেউ বোঝেন না এটা কী | `local-rag-api-fastapi` |
| README-তে শুধু install command | "এটা কী?"-এর উত্তর নেই | প্রথম স্ক্রিনে pitch আর GIF |
| Course-এর Live Results টেবিল হুবহু রাখা | "কোন GPU?"-এর উত্তর নেই | নিজের run, hardware সহ; নইলে "Reference run" লেবেল |
| `venv/`, `__pycache__/`, `chroma_db/` commit | Repo ভারী, অযত্নের ছাপ | `.gitignore` আগে, তারপর `git add` |
| `.env`-এ আসল key commit | Key ফাঁস | `.env.example`; ফাঁস হলে আগে rotate |
| দুই-column-এর সুন্দর resume template | ATS text উল্টাপাল্টা পড়ে | এক column, সাধারণ heading |
| দুই-তিন পাতার junior resume | সেরা project চাপা পড়ে | এক পাতা, ৩-৫টা project |
| দায়িত্বের bullet ("Worked on...") | কোনো ফলাফল নেই | XYZ: কী, কীভাবে মাপা, কী করে |
| বানানো বা ধার করা সংখ্যা | Deep-dive-এ ধরা পড়ে, বিশ্বাস শেষ | শুধু নিজের মাপা সংখ্যা, `[X]` পূরণ করে |
| "Production users" দাবি | মিথ্যা | "Production-style": test, probe, Docker |
| Skills-এ ৪০টা buzzword | কোনোটায় গভীরতা নেই | যা দিয়ে কিছু বানিয়েছেন |
| STAR-এ শুধু "আমরা" | আপনি কী করলেন বোঝা যায় না | "আমি" আর নির্দিষ্ট action |
| System design-এ সরাসরি tool-এর নাম | Requirement না জেনেই সমাধান | আগে প্রশ্ন, তারপর হিসাব, তারপর design |
| Take-home-এ over-engineering | সময় শেষ, মূল অংশ অসম্পূর্ণ | সরল version আগে, README-তে "next steps" |
| "Why local?"-এ শুধু প্রশংসা | একপেশে, অপরিণত শোনায় | সুবিধা, সৎ trade-off, আর কখন cloud |
| AI assistant-এর code না বুঝে জমা | পরের round-এ ব্যাখ্যা করতে পারেন না | প্রতিটা লাইন বুঝে, policy মেনে |

---

## 7. Interview Q&A

"Tell me about a project" ধাঁচের প্রশ্ন — উত্তরের কাঠামো আর এই course-এর project থেকে উদাহরণ।

1. **"Tell me about a project you're proud of."** ৯০ সেকেন্ড: এক লাইনে কী আর কেন → architecture-এর তিনটা ধাপ → সবচেয়ে কঠিন অংশ আর আপনার সিদ্ধান্ত → মাপা ফল → একটা limit। যেমন #14: "Week 12-এর RAG demo-কে একটা service বানিয়েছি। কঠিন অংশ ছিল খারাপ দিনের আচরণ — Ollama বন্ধ থাকলে degraded উত্তর, ভিড়ে 503 আর Retry-After। আমার benchmark-এ throughput ~[X] req/s-এ আটকেছে, কারণ একটাই GPU stream — তাই concurrency বাড়ালে শুধু latency বাড়ে।"
2. **"What was the hardest bug?"** একটা আসল ঘটনা বলুন। #11: llama3.1-এর chat template tool result আসার পরে tool তালিকা দেখায় না, তাই model booking না ডেকে প্রশ্ন করে থেমে যেত — `ollama show --template` পড়ে কারণ ধরেছি, `final_answer` tool আর nudge দিয়ে সমাধান। অথবা #13: guardrail-এর return annotation `from __future__ import annotations`-এর কারণে string হয়ে গিয়ে CrewAI task বানাতেই `ValidationError`।
3. **"How do you know it works?"** Test আর eval আলাদা: test (LLM ছাড়া, fake দিয়ে) wiring আর logic প্রমাণ করে; eval (আসল model দিয়ে) quality মাপে। #12-এ visible test-এর পাশে hidden test, যা model কখনো দেখে না — তাই hard-code করা উত্তর ধরা পড়ে।
4. **"How did you test code that calls an LLM?"** LLM client একটা module-এ বন্দি; test-এ scripted fake (`FakeLLM(BaseLLM)`, FastAPI-তে `RAG_BACKEND=fake`); ভারী import lazy — function-এর ভেতরে, module-এর মাথায় না — যাতে test-এ model download বা network না লাগে। Quality মাপা হয় আলাদা golden set-এ।
5. **"How do you handle hallucination?"** স্তরে স্তরে: retrieval distance threshold পার না হলে LLM-ই ডাকা হয় না (#10, #14); prompt-এ "শুধু context থেকে"; প্রতিটা উত্তরে citation; code-এ citation verification (#13); আর Abstention — "knowledge base-এ নেই" একটা সঠিক ফল, ব্যর্থতা না (#16)।
6. **"Walk me through the architecture."** Whiteboard-এ বাঁ থেকে ডানে: input → validation → core → guardrail → output, আর প্রতিটা তীরে কোন data যায়। README-র ASCII diagram মুখস্থ না, বুঝে আঁকুন।
7. **"Why this model? Why local?"** 3.14-এর কাঠামো: privacy, খরচ, offline, version pin → সৎ trade-off → কখন cloud। সাথে একটা নির্দিষ্ট উদাহরণ: #08-এ জনপ্রিয় DistilBERT SST-2-এর বদলে `cardiffnlp/twitter-roberta-base-sentiment-latest`, কারণ সেটায় আসল Neutral class আছে; #15-এ `ProsusAI/finbert`, কারণ "guidance cut"-এর মতো finance-এর ভাষা tweet-এ train করা model-এর জন্য domain shift (ADR 0009)।
8. **"What would you do differently, or at 10x scale?"** Known Limits থেকে উত্তর: #14-এ upload synchronous → background job; in-memory cache আর semaphore process-প্রতি আলাদা → Redis; per-key rate limit নেই; একটা Ollama → replica বা vLLM।
9. **"Tell me about a trade-off you made."** #14: exact-match cache বনাম semantic cache — semantic cache ভুল "কাছাকাছি" hit-এ আত্মবিশ্বাসী ভুল উত্তর দিতে পারে আর poison করা যায়, তাই exact cache, key-তে `corpus_version`। অথবা concurrency gate = 1, Ollama-র slot-এর সমান — queue আমাদের layer-এ থাকলে timeout আর 503 সম্ভব হয়।
10. **"Tell me about a result that surprised you."** #13: একই passage আর guardrail-এ single agent ~৭ গুণ কম token-এ (1,076 বনাম 7,465, course reference run) প্রায় একই citation precision দিয়েছে। Multi-agent-এর আসল মূল্য separation of concerns আর auditability, quality না। Negative result মেপে স্বীকার করা — এটা senior সংকেত।
11. **"Which part did you write yourself?"** সৎ উত্তর: "Course-এর reference implementation থেকে শুরু; আমি নিজে করেছি [নির্দিষ্ট extension], আর পুরো pipeline নিজে চালিয়ে [সংখ্যা] মেপেছি।" তারপর সেই extension-এর code খুলে দেখান।
12. **"Where does this number on your resume come from?"** Hardware, model আর version, dataset বা task-এর আকার, কতবার চালিয়েছেন, কোন command (`python cli.py --bench`)। Sample ছোট হলে নিজেই বলুন: "এটা ১১টা task-এর smoke benchmark, statistical claim না।"
13. **"Why does code compute the risk score instead of the LLM?"** (#15) LLM-এর দেওয়া সংখ্যা non-deterministic, যাচাই বা test করা যায় না। Volatility, drawdown, VaR আর FinBERT sentiment থেকে documented weight-এ code Risk Score বানায়; Risk Analyst Agent tool ডেকে সেটা আনে আর memo-তে ব্যাখ্যা করে — কিন্তু সংখ্যা বদলাতে পারে না।
14. **"How did you make the healthcare assistant safe?"** (#16) Retrieval বা model call-এর আগেই deterministic Red-Flag Escalation (যেমন chest pain) — আগে-লেখা নির্দিষ্ট message, যা model লেখে না বা বদলায় না; input-এ PII redaction; knowledge base সমর্থন না করলে Abstention; scope-এর বাইরের অনুরোধে refusal; আর একটা eval set যা প্রতিটা আচরণ মাপে। সাথে স্পষ্ট বলুন: এটা compliance-certified product না, একটা privacy-first prototype।
15. **"How would you make it production-ready?"** দুটো তালিকা দিন — যা আছে (test, health/ready probe, backpressure, Docker, monitoring seam) আর যা নেই (auth আর multi-tenant, rate limit, CI/CD, secret management, আসল user-এর eval, on-call)। কী নেই সেটা জানা, কী আছে সেটা দাবি করার মতোই গুরুত্বপূর্ণ।

---

## Achievement: Finalized AI Engineer Resume, Polished LinkedIn, 16-Project GitHub Portfolio (Class 40)

**`kit/`** — চারটা ready-to-fill file:

* `portfolio_map.md` — ১৬টা Portfolio Project → course folder → pitch → skill → pin order, তিনটা role-ভিত্তিক pin set সহ।
* `portfolio_README_template.md` — GitHub profile README আর per-project README template, course README-র section ধরে।
* `resume_template.md` — এক পাতার AI Engineer resume আর bullet bank: প্রতিটা project-এর ২-৩টা bullet, `[X]` placeholder, course reference সংখ্যা আলাদা করে চিহ্নিত।
* `linkedin_checklist.md` — headline formula, About template, Featured, Skills, first-post template।

ক্লাস শেষের checklist: profile README live; ছয়টা repo pin করা, প্রতিটায় GIF আর নিজের অন্তত একটা extension; resume এক পাতার PDF, প্রতিটা সংখ্যা নিজের মাপা; LinkedIn headline, About আর Featured আপডেট; প্রথম project post publish।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **৩০ সেকেন্ডের test:** একজন non-technical বন্ধুকে আপনার GitHub profile ৩০ সেকেন্ড দেখতে দিন, তারপর জিজ্ঞেস করুন "আমি কী বানাই?" উত্তর আপনার headline-এর সাথে না মিললে profile README-র প্রথম স্ক্রিন বদলান।
2. **Bullet rewrite:** নিজের লেখা তিনটা "Worked on..." bullet XYZ formula-য় লিখুন। প্রতিটা সংখ্যার উৎস (command, hardware, তারিখ) একটা আলাদা note-এ রাখুন।
3. **Secret audit:** প্রতিটা public repo-র পুরো history-তে secret scanner চালান। কিছু পেলে প্রথম কাজ কোনটা, আর কেন?
4. **Reproduce a number:** #12-এ `python cli.py --bench` অথবা #11-এ `python cli.py --eval` নিজের মেশিনে চালান। Course reference-এর সাথে মেলান; পার্থক্য এলে README-তে কারণ লিখুন।
5. **STAR timer:** তিনটা pinned project-এর STAR গল্প জোরে বলুন, timer চালিয়ে। ২ মিনিট পেরোলে Situation ছোট করুন, Action না।
6. **Mock design:** "৫০০ কর্মীর company-র HR policy bot, data বাইরে যাবে না" — ১৫ মিনিটে design করুন: প্রশ্ন, হিসাব, diagram, guardrail, eval, আর scale বাড়লে কী বদলাবে।
7. **What not to claim:** একজন সহপাঠীর resume পড়ে এমন একটা লাইন খুঁজুন যা interview-তে প্রমাণ করা যাবে না। নিজেরটাও তাঁকে দিন।

---

## References

* GitHub Docs — Managing your profile README: https://docs.github.com/en/account-and-profile/how-tos/profile-customization/managing-your-profile-readme
* GitHub Docs — Pinning items to your profile: https://docs.github.com/en/account-and-profile/how-tos/profile-customization/pinning-items-to-your-profile
* GitHub Docs — Security features (push protection for users): https://docs.github.com/en/code-security/getting-started/github-security-features
* GitHub Docs — About large files on GitHub: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
* HR Dive — Ladders eye-tracking study (2018), 7.4-second initial screen: https://www.hrdive.com/news/eye-tracking-study-shows-recruiters-look-at-resumes-for-7-seconds/541582/
* Laszlo Bock's XYZ resume formula (Business Insider summary): https://www.businessinsider.in/google-hr-boss-says-this-is-the-key-to-a-perfect-resume/articleshow/43924179.cms
* LinkedIn field limits (headline 220, About 2,600 characters): https://www.unilink.us/blog/linkedin-character-limits-2026
* Gitleaks — https://github.com/gitleaks/gitleaks · git-filter-repo — https://github.com/newren/git-filter-repo
* Course internal: `projects.md`; `docs/adr/0002`, `0006`, `0008`, `0009`, `0010`; প্রতিটা Portfolio Project-এর README (Live Results আর Known Limits section)।
