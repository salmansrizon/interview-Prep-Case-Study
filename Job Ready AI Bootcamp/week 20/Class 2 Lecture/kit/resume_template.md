# AI Engineer Resume Template (one page)

How to use this file:

1. Copy **Part 1** into Google Docs or Word, fill it in, export a **text-based PDF** named `Firstname-Lastname-AI-Engineer.pdf`.
2. Pick **3-5 projects** from the **Part 2 bullet bank** that match the job description. Use 2-3 bullets each.
3. Replace every `[X]` with a number **you measured yourself**, and keep a note of how (command, hardware, versions, date). If you can't measure it, delete the number, not the honesty.
4. Run the **Part 3 checklist** before every application.

Numbers marked **Course reference** come from the course READMEs (the author's machine). They show the range to expect. They are **not** your results: never paste them into your resume as your own.

---

## Part 1: The one-page template

```markdown
# [FULL NAME]
[City, Country] · [email] · [phone] · linkedin.com/in/[handle] · github.com/[handle]

## Summary
AI Engineer who builds [RAG systems, tool-calling agents and LLM APIs] on local models
(Ollama, llama3.1:8b), with offline tests, guardrails and measured evaluations.
16 end-to-end projects from NumPy and scikit-learn to LangGraph, CrewAI and FastAPI.
Seeking [AI Engineer / ML Engineer] roles in [domain / location / remote].

## Skills
- **Languages:** Python[, SQL — only if you use it]
- **LLM & Agents:** Ollama, llama3.1:8b, prompt engineering, tool calling, ReAct, LangGraph, CrewAI
- **RAG & Retrieval:** ChromaDB, sentence-transformers, BM25, chunking, metadata filtering, citation verification
- **ML / DL:** scikit-learn, TensorFlow/Keras, HuggingFace Transformers, OpenCV, NumPy, pandas
- **Serving & Quality:** FastAPI, Server-Sent Events, Docker, pytest[, Prometheus, OpenTelemetry]
- **Tools:** Git, GitHub, Streamlit, [Linux]

## Projects
### [Project Name] | [Tech, Tech, Tech] | github.com/[handle]/[repo]
- [Bullet from the bank, numbers filled in]
- [Bullet]
- [Bullet]

### [Project Name] | [Tech, Tech] | github.com/[handle]/[repo]
- [Bullet]
- [Bullet]

### [Project Name] | [Tech, Tech] | github.com/[handle]/[repo]
- [Bullet]
- [Bullet]

[### 4th and 5th project, if the page still has room]

More: 16 end-to-end AI projects at github.com/[handle]

## Experience
### [Job Title] | [Company] | [City] | [Mon YYYY - Mon YYYY]
- [What you achieved, XYZ style. Non-AI jobs count: automation, data, teamwork, customers.]
- [Bullet]

## Education
- **[Degree]**, [University] | [Year]
- **Job Ready AI Bootcamp** | [Year] | 20-week AI engineering program, 16 portfolio projects

## [Optional: Certifications | Languages | Awards]
- [Only if relevant and true]
```

**Layout rules (ATS-safe)**

- One column. Standard headings exactly as above. No tables, text boxes, icons, skill bars or photos holding key information.
- Contact details in the body, not in the page header/footer.
- One page. If it spills over, cut the weakest project first, not the margins.
- Photo: leave it out for international/remote roles; some local employers ask for one, so follow the job post.
- Order for a fresh graduate: Summary → Skills → Projects → Experience → Education. If you have relevant AI work experience, put Experience before Projects.

**Bullet formula (XYZ):** *Accomplished [X] as measured by [Y], by doing [Z].* Start with a strong verb (Built, Designed, Reduced, Measured, Diagnosed, Hardened). One line, two at most.

---

## Part 2: Project bullet bank

For each project: a title line, 2-3 bullets (pick the ones you can defend), the course reference numbers where the README reports them, and the command to measure your own.

### 01. Mathematical Vector Engine
**Title line:** NumPy Vector Similarity Engine | Python, NumPy, PyYAML
- Built a loop-free semantic search engine in pure NumPy: L2-normalized document vectors once at index time so every query is a single matrix multiplication (cosine similarity) with top-k ranking via `argsort`.
- Implemented scaled dot-product attention with a causal mask from scratch and showed why the √d_k scaling matters (without it, softmax collapses onto a single token).
- Benchmarked vectorized NumPy against pure-Python loops: [X]× faster with identical results on my machine; completed [8]/8 vectorization exercises checked by an automated, loop-detecting grader.
- *Course reference:* speedup varies roughly 30x-200x by CPU, so measure your own. *Measure:* `python main.py`.

### 02. Local Real Estate Valuator
**Title line:** Explainable House Price Predictor | scikit-learn, pandas, Streamlit, Plotly
- Built an explainable price predictor (Linear Regression vs Ridge and Lasso) in a 6-page Streamlit dashboard that breaks every prediction into per-feature dollar contributions.
- Reached R² [X] and RMSE $[X] (MAPE [X]%) on a 5,000-row synthetic dataset; residual analysis and a Ridge/Lasso comparison showed no gain from regularization, so I kept the simpler model.
- Implemented gradient descent by hand to visualize learning-rate convergence vs divergence, and documented data leakage (scaler fit on train only) and multicollinearity effects on coefficients.
- *Course reference (deterministic: fixed CSV + `random_state=42`, should reproduce exactly):* R² 0.8754, RMSE $22,433, MAE $17,077, MAPE 6.2%. Always say the data is synthetic. *Measure:* Model Evaluation page, or `python train_model.py`.

### 03. Pure Offline Spam & Intent Classifier
**Title line:** Offline Spam & Intent Classifier | scikit-learn, TF-IDF, NLTK, Streamlit
- Built a fully offline text-classification app for spam detection and 6-class intent routing (purchase, support, inquiry, complaint, feedback, other) on TF-IDF unigram + bigram features.
- Trained and compared Naive Bayes, Linear SVM and KNN on precision, recall, F1 and confusion matrices; best model: [model] at F1 [X] on [N] held-out messages ([synthetic / your dataset]).
- Made preprocessing run without downloaded NLTK data through built-in fallbacks, covered by pytest.
- *Course reference:* no metrics reported (synthetic data). *Measure:* Train & Evaluate tab.

### 04. Customer Segmentation Engine
**Title line:** Customer Segmentation Engine | scikit-learn, mlxtend, Plotly, Streamlit
- Built an unsupervised analytics app combining K-Means customer segmentation, PCA (auto-selects components for a 95% variance target) and Apriori market-basket rules.
- Chose K using elbow, silhouette and Calinski-Harabasz scores rather than inertia alone: K=[X], silhouette [X] on [synthetic / your] customer data.
- Mined association rules filtered by support, confidence and lift (> [1.5]) for bundle ideas, with pytest covering preprocessing, K-Means, PCA and Apriori.
- *Course reference:* no metrics reported (synthetic data). *Measure:* K-Means and Market Basket tabs.

### 05. High-Accuracy MNIST Digit Classifier
**Title line:** MNIST CNN Optimization Lab | TensorFlow/Keras, Streamlit, pytest
- Trained and compared three CNN variants (baseline, dropout, and BatchNorm + dropout with EarlyStopping and ReduceLROnPlateau) under an isolated-variable design; the optimized model reached [X]% test accuracy (target > 98%).
- Quantified overfitting as the train-validation gap per variant ([X] pt baseline vs [X] pt optimized), shown as live curves in a Streamlit app.
- Shipped draw-to-predict (canvas or upload) with MNIST-style preprocessing and per-class confidence; a fast pytest suite checks shapes, layer presence and forward passes without training.
- *Course reference:* README gives only "typical" values (optimized ~99.0% validation), not a measured run. *Measure:* Train & Compare and Model Insights tabs on the full dataset, and note epochs and seed.

### 06. Real-Time Video Privacy Shield
**Title line:** Real-Time Face Privacy Shield | OpenCV, Streamlit
- Built a real-time webcam face-anonymization app (OpenCV Haar Cascade) with five modes (Gaussian blur, pixelate, solid mask, black bar, emoji) plus an eyes-only mode and feathered edges.
- Decoupled capture from processing with a threaded camera reader and frame queue; the live HUD showed [X] FPS at [1280×720] on my [laptop CPU].
- *Only if you did it:* Added pytest coverage for every anonymization mode on synthetic frames (the course build ships without tests).
- *Course reference:* no metrics reported. *Measure:* the live HUD; note resolution and CPU.

### 07. Semantic CV-to-Job Matcher
**Title line:** Semantic CV-to-Job Matcher | sentence-transformers, NumPy, Streamlit
- Built a semantic CV ranker that embeds a job description and multiple CVs (pasted text, PDF or DOCX) with `paraphrase-MiniLM-L3-v2` (384-dim) and ranks them by hand-written cosine similarity.
- Chose a ~61 MB CPU-friendly embedding model for real-time local ranking and documented the trade-off against larger models; the model loads lazily, so the test suite never downloads it.
- Covered text/PDF/DOCX extraction and ranking correctness with [X] pytest tests using hand-crafted vectors.
- *Course reference:* no ranking metrics reported. *Measure:* build a small labeled set (e.g. [N] CVs × [M] jobs) and report top-1 accuracy, or skip the number.

### 08. Local Customer Feedback Analyzer
**Title line:** Local Review Sentiment Analyzer | HuggingFace Transformers, pandas, Streamlit
- Built a local 3-class (positive / negative / neutral) review-sentiment analyzer with single, batch-paste and CSV-upload modes (review column auto-detected) and an aggregate dashboard.
- Chose `cardiffnlp/twitter-roberta-base-sentiment-latest` for its real Neutral class over the popular binary SST-2 default, and flagged predictions below 0.6 confidence for human review ([X]% of my [N]-review sample).
- Isolated the model behind one module with batched inference, so ingestion, aggregation and flagging are unit-tested without downloading it.
- *Course reference:* no accuracy reported. *Measure:* label [N] reviews yourself and report accuracy and the flagged share.

### 09. Zero-Shot Brand Content Generator
**Title line:** Local Brand Voice Generator | Ollama (llama3.1:8b), Streamlit, pytest
- Built an on-brand marketing-copy generator on a local LLM (llama3.1:8b via Ollama) with zero-shot and few-shot modes, an optional chain-of-thought trace, and separate temperature / top-p controls.
- Implemented a two-layer prompt-injection defense: delimited untrusted input with an instruction-hierarchy system prompt, then an output validator for persona switches and off-topic replies; it caught [X]/[N] attacks in my own test set.
- Kept the only Ollama import in one module so the full pytest suite runs with the LLM mocked.
- *Course reference:* no metrics reported. Never claim the defense is foolproof; the README itself says it isn't. *Measure:* write [N] attack prompts and use the "Try to Break It" tab.

### 10. Local Document Q&A Bot (RAG Engine)
**Title line:** Local PDF RAG Q&A | ChromaDB, sentence-transformers, Ollama, Streamlit
- Built a local RAG app that answers strictly from uploaded PDFs with source + page citations, using paragraph-aware chunking with overlap and a persistent ChromaDB index.
- Added a retrieval-distance guardrail that answers "not found" without calling the LLM, calibrated on my corpus (relevant ≈ [X]-[X], off-topic ≈ [X]-[X] L2 distance); refused [X]/[N] off-topic questions.
- Implemented per-document search as a ChromaDB `where` filter, tested against a real temporary ChromaDB with the LLM mocked.
- *Course reference:* on the course test corpus, relevant questions scored ~10-19 and off-topic ~30-37, so the threshold was set at 22.0. *Measure:* ask known-relevant and known-irrelevant questions on your PDFs.

### 11. Deterministic Function-Calling Travel Agent
**Title line:** ReAct Travel Agent with Tool Calling | Python, Ollama (llama3.1:8b), Pydantic, Streamlit
- Built a conversational flight-search and booking agent with a hand-written ReAct loop (native tool calls and classic text parsing) on llama3.1:8b, without an agent framework.
- Hardened a weak local model with code: Pydantic-validated tool calls (`extra="forbid"`), typed stop reasons, a repeat-call guard, and a code-enforced human approval gate before any booking.
- Built a golden-task evaluation harness (end state, trajectory, policy, pass^k): native mode solved [X]/8 tasks vs [X]/8 for text mode; [39] offline tests.
- Diagnosed a llama3.1 chat-template quirk (tools hidden after a tool result) and fixed it with a `final_answer` tool plus a progress nudge, measuring every prompt change with the same eval.
- *Course reference (Ollama 0.40.8, temperature 0):* native 6/8 (75%), text 4/8 (50%); nudge without `final_answer` dropped native to 0/8. *Measure:* `python cli.py --eval` and `python cli.py --eval --mode text`.

### 12. Self-Correcting LLM Coder
**Title line:** Self-Correcting LLM Coder | LangGraph, Ollama (llama3.1:8b), SQLite, Streamlit
- Built a LangGraph StateGraph in which llama3.1:8b writes Python, runs unit tests in an isolated subprocess (`python -I`, timeout, resource limits), and repairs its code from per-test failure feedback within a retry budget.
- Caught reward hacking with hidden tests the model never sees plus AST static checks before execution; added human approval via `interrupt()` / `Command(resume=...)` and SQLite checkpointing with time-travel replay.
- Benchmarked 11 coding tasks: pass@1 [X]/11 → [X]/11 after self-repair; [27] offline tests (fake LLM, real subprocess sandbox and SQLite).
- *Course reference (temperature 0, budget 4):* pass@1 8/11 (73%) → 9/11 (82%) after repair. *Measure:* `python cli.py --bench`.

### 13. Collaborative Multi-Agent Research Team
**Title line:** Grounded Multi-Agent Research Team | CrewAI, Ollama (llama3.1:8b), BM25, Streamlit
- Built a CrewAI Flow (retrieve → Researcher / Writer / Reviewer crew → gate → publish or escalate to a human) on llama3.1:8b over a local 34-passage corpus, fully offline (telemetry and memory disabled).
- Moved retrieval (BM25) and citation verification into code after observing the 8B model never called its search tool and invented citation ids; per-sentence verdicts (supported / weak / uncited / invalid id) gate publication.
- Ran a same-budget single-agent baseline: [your result, e.g. "[X]× fewer tokens at citation precision [X] vs [X]"], showing the crew's value is auditability rather than quality; [20] offline tests with a fake `BaseLLM`.
- *Course reference (3 topics):* single agent 60 s, ~1,076 tokens, citation precision 1.00 vs crew 92 s, ~7,465 tokens, 0.95. *Measure:* `python cli.py "<topic>" --mode single` vs default mode, same topics.

### 14. Production-Ready RAG API Service
**Title line:** RAG API Service | FastAPI, ChromaDB, Ollama (llama3.1:8b), SSE, Docker
- Turned a RAG prototype into an async FastAPI service with OpenAPI docs: validated PDF upload (413/415/422), idempotent hash-based ingestion, page-exact citations, and token streaming over Server-Sent Events.
- Designed for failure: `asyncio.Semaphore` backpressure (503 + Retry-After), a retrieval-only degraded mode when the LLM is down, `/health` vs `/ready` probes, optional API key, request IDs, and an exact-match cache invalidated by corpus version.
- Load-tested with a concurrency sweep: throughput plateaued at ~[X] req/s on one GPU stream (p95 [X] s, TTFT p50 [X] s), so I kept the gate equal to the LLM's parallel slots; [24] offline tests.
- *Optional, if you built it:* Added Prometheus metrics, OpenTelemetry traces, per-request quality flags and SLO burn-rate alerts through an observer seam without touching RAG code ([43] tests).
- *Course reference (MacBook, Ollama 0.40.8):* throughput ~0.6 req/s; cache hit ~3 ms vs ~1.3 s; startup warmup cut cold first-request TTFT 8.4 s → ~2.5 s; 7/7 on-topic answers cited, 2/2 off-topic refused with no LLM call. *Measure:* `python benchmark/bench.py --url http://localhost:8000 --levels 1,2,4 --stream`.

### 15. Offline Market Intelligence Team
**Title line:** Offline Market Intelligence Agents | CrewAI, Ollama (llama3.1:8b), FinBERT, FastAPI
- Built a two-agent CrewAI system on llama3.1:8b: a News Scout Agent surfaces ticker news from an Offline Snapshot with FinBERT sentiment, and a Risk Analyst Agent explains the ticker's risk in an auto-generated PDF investment memo, served over FastAPI.
- Kept the number out of the LLM: a deterministic 0-100 Risk Score computed in code from volatility, drawdown, VaR and news sentiment with documented weights; the agent calls tools and explains the score but can never set it.
- Chose `ProsusAI/finbert` (finance-domain, native 3-class) over a tweet-trained model and LLM zero-shot labeling for deterministic, testable sentiment; [X] offline tests with the LLM and classifier mocked.
- *Course reference:* none (static verification, ADR 0008). Every number here must be your own. Never write "real-time data" or "investment advice".

### 16. Local Knowledge-Grounded Healthcare Assistant
**Title line:** Private Healthcare RAG Assistant | ChromaDB, Ollama (llama3.1:8b), FastAPI[, Docker]
- Built a privacy-first medical Q&A assistant that grounds every answer in a local clinical knowledge base (ChromaDB + llama3.1:8b; no data leaves the machine), served over FastAPI.
- Layered safety in code: PII redaction, deterministic red-flag escalation (e.g. chest pain) returned before any retrieval or model call, refusal for out-of-scope requests, and abstention when the knowledge base doesn't support an answer.
- Built an evaluation set of answerable, unanswerable, red-flag and PII cases: [X]/[N] red flags escalated, [X]% correct abstentions, [X]% grounded answers on my machine.
- *Optional, if you did Week 18:* Packaged the full stack (UI + API + Chroma + Ollama) as a one-command `docker compose up` with an optional GPU override.
- *Course reference:* none (static verification, ADR 0008). Never write "HIPAA-compliant", "clinically validated" or "diagnoses".

---

## Part 3: Pre-send checklist

**Truth**
- [ ] Every number is from my own run, and I can name the command, hardware and versions.
- [ ] No course reference number appears as my result.
- [ ] Synthetic data is called synthetic.
- [ ] No "production users", "deployed at scale", "fine-tuned", "trained an LLM", "HIPAA-compliant", "real-time market data", "expert in".
- [ ] For each project on the page, I can open the repo and show the code behind each bullet in 30 seconds.

**Fit**
- [ ] 3-5 projects chosen for this job description; the strongest match is first.
- [ ] Keywords from the job description that I genuinely used appear with the same spelling.
- [ ] The summary names the role I'm applying for.

**Format**
- [ ] One page, one column, standard headings, text-based PDF.
- [ ] Links (GitHub, LinkedIn, repos) work and are spelled out as text.
- [ ] File name: `Firstname-Lastname-AI-Engineer.pdf`.
- [ ] Spell-checked; dates and tense consistent (past tense for finished work).

**Words to avoid, and what to say instead**

| Avoid | Say instead |
|---|---|
| "Responsible for", "Worked on", "Helped with" | Built, Designed, Measured, Diagnosed, Reduced |
| "Production-ready" (unqualified) | "Production-style: tests, health probes, backpressure, Docker" |
| "State-of-the-art" | The actual model name and why it fits |
| "Passionate about AI" | A project link |
| "Various", "multiple", "etc." | The specific things |
