# LinkedIn Checklist for an AI Engineer

LinkedIn is a search engine first: recruiters search by keyword, then decide from your photo, name and headline whether to click. Fix the parts that show up in search before polishing the rest.

## 1. Basics (15 minutes)

- [ ] **Photo:** recent, face clearly visible, plain background, you alone.
- [ ] **Banner:** simple. Your name + "AI Engineer" + GitHub URL, or a clean architecture diagram from one of your projects. No stock "AI brain" art.
- [ ] **Custom URL:** `linkedin.com/in/firstname-lastname` (Edit public profile & URL). Use the same handle as GitHub if you can.
- [ ] **Location** set to where you want to work (or the city you're in for remote roles).
- [ ] **Contact info:** email, GitHub, portfolio link.
- [ ] **Open to work:** set job titles (AI Engineer, ML Engineer, LLM Engineer), locations and remote preference. Choose "recruiters only" if you are currently employed.

## 2. Headline (up to 220 characters; only the first ~60-70 show in search)

Put the role first, then proof. Pick one formula:

| Formula | Example |
|---|---|
| Role \| Specialty \| Proof | `AI Engineer \| RAG, Agents & LLM APIs on local models \| 16 open-source projects` |
| Role \| Stack \| Domain | `AI Engineer \| LangGraph · CrewAI · FastAPI · ChromaDB \| Privacy-first healthcare & finance AI` |
| Role who builds X that does Y | `AI Engineer building LLM systems that cite their sources and know when not to answer` |
| Student / graduate version | `Aspiring AI Engineer \| Job Ready AI Bootcamp graduate \| RAG · Agents · FastAPI \| Open to work` |

Rules
- [ ] Contains the exact job title you want (that's what recruiters type).
- [ ] 3-5 real keywords, no buzzword soup.
- [ ] No "Passionate", "Guru", "Ninja", "Visionary".

## 3. About section (up to 2,600 characters; the first 2-3 lines show before "see more")

Template (fill the brackets, keep it under ~1,500 characters):

```text
I build LLM systems that run on local models and can prove they work: [RAG assistants,
tool-calling agents and APIs] with tests, guardrails and measured evaluations.

What I've built (all on GitHub, all running offline on llama3.1:8b):
• [Healthcare RAG Assistant]: grounded medical Q&A; red-flag escalation and PII
  redaction run in code before the model; it abstains when the knowledge base has no answer.
• [RAG API Service]: async FastAPI with SSE streaming, backpressure (503 + Retry-After)
  and a degraded mode when the LLM is down. [My measured number, e.g. throughput/latency.]
• [Self-Correcting Coder]: a LangGraph loop that writes code, runs tests in a subprocess
  and repairs itself; hidden tests catch shortcuts. [My pass@1 → after-repair result.]

How I work: code around the model (validation, deterministic tools, guardrails), offline
tests with fake LLMs, honest "known limits" in every README.

Stack: Python · Ollama · LangGraph · CrewAI · ChromaDB · FastAPI · scikit-learn ·
TensorFlow/Keras · OpenCV · Docker · pytest

Looking for: [AI Engineer / ML Engineer] roles, [on-site in City / remote].
Reach me: [email] · github.com/[handle]
```

Rules
- [ ] The first line says what you build, not "Hello, I am...".
- [ ] Every number is your own measurement (same rule as the resume).
- [ ] Ends with what you want and how to reach you.

## 4. Featured (3-4 items)

- [ ] Your GitHub profile (link to `github.com/[handle]`).
- [ ] Your best repo (link, with the demo GIF as the thumbnail if possible).
- [ ] Your project write-up post (see section 7).
- [ ] Optional: a 1-2 minute demo video, or your resume PDF (only if you're fine with it being public).

## 5. Experience, Education, Projects

- [ ] **Education:** add Job Ready AI Bootcamp (dates, "20-week AI engineering program, 16 portfolio projects").
- [ ] **Projects section:** add your 4-6 pinned projects, each with a one-line pitch, the stack, and the repo link. Same wording as the resume, no new claims.
- [ ] **Experience:** keep non-AI jobs; rewrite one bullet each to show transferable skills (automation, data, customers, ownership).
- [ ] Never list the bootcamp as an employer or a job title.

## 6. Skills

- [ ] List only skills you used in a project you can show. Suggested core set: Python, Large Language Models (LLM), Retrieval-Augmented Generation (RAG), Prompt Engineering, LangGraph, CrewAI, FastAPI, ChromaDB, Vector Databases, scikit-learn, TensorFlow, Keras, Hugging Face Transformers, OpenCV, Docker, Git, pytest.
- [ ] Put the 5 most relevant to your target role at the top.
- [ ] Attach skills to the matching project or education entry where LinkedIn offers it.
- [ ] Remove skills you can't discuss for five minutes.

## 7. First post: announce one project

Post one project, not all 16. Lead with an observation, not "Excited to share". Attach the demo GIF or a screenshot. Put the repo link in the post (or in the first comment).

Template:

```text
[One surprising, specific observation from building it.]
Example: "My 8B local model was given a search tool and never called it once. It wrote the
tool call as plain text and then invented the search results, citation ids included."

So I built [project name] differently:
→ [Design decision 1, e.g. retrieval runs in code, the agents only select and summarize]
→ [Design decision 2, e.g. code verifies every citation before the report is published]
→ [Design decision 3, e.g. anything that fails verification goes to a human]

What I measured on my machine ([hardware], llama3.1:8b via Ollama):
• [Your number 1]
• [Your number 2, a negative result is fine, e.g. "a single agent matched the crew's
  citation quality with far fewer tokens"]

What I'd do next: [one honest limitation + the fix].

Code (runs fully offline): [repo link]
Built during Job Ready AI Bootcamp. [One line on what you added yourself.]

#AIEngineering #LLM #RAG #[Framework]
```

Post checklist
- [ ] Your own numbers, with hardware named.
- [ ] One visual (GIF, screenshot or architecture diagram).
- [ ] 3-5 hashtags, not 20.
- [ ] Reply to every comment within the first day.
- [ ] Follow-up posts: one project or one lesson every 1-2 weeks is plenty.

## 8. Weekly habits while job hunting

- [ ] Message 3-5 people a week (engineers or hiring managers at target companies): 3 lines, specific, one repo link, no attachment.
- [ ] Comment with substance on 2-3 posts from engineers you follow.
- [ ] After every interview, note the questions you got and where you got stuck.

## 9. Final consistency check

- [ ] Headline, About, resume Summary and GitHub profile README say the same thing in different lengths.
- [ ] Project names and numbers match across resume, LinkedIn and READMEs.
- [ ] All links open in a private browser window.
