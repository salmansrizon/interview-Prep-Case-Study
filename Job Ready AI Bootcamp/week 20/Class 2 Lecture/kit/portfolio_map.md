# Portfolio Map — 16 Portfolio Projects

Where each of the 16 `projects.md` Portfolio Projects lives in this course, what to call it on GitHub, how to pitch it in one line, and which six to pin.

Paths are relative to `Job Ready AI Bootcamp/`. Each project folder is self-contained (ADR 0002): copy it into its own repo with its `.gitignore`, skip `__pycache__/`, `venv/` and any `data/chroma_db/` or `output/` folder.

## The map

Pin order is for an **LLM / AI Engineer** application (GitHub allows six pins). Rows marked "table" go in your profile README's project table, in the order shown.

| # | Project | Course folder | Suggested repo name | One-line pitch | Key skills | Pin order |
|---|---|---|---|---|---|---|
| 01 | Mathematical Vector Engine | `week 2/Class 2 Project` | `numpy-vector-similarity-engine` | Loop-free semantic search and scaled dot-product attention in pure NumPy | NumPy vectorization, broadcasting, cosine similarity, matmul, attention + causal mask | table 16 |
| 02 | Local Real Estate Valuator | `week 5/Class 1 Project` | `explainable-house-price-predictor` | Linear-regression price predictor that breaks every estimate into per-feature dollar contributions | scikit-learn (Linear/Ridge/Lasso), metrics (RMSE, MAE, R², MAPE), residuals, gradient descent, Streamlit | table 14 |
| 03 | Pure Offline Spam & Intent Classifier | `week 6/Class 1 Project` | `offline-spam-intent-classifier` | 100% offline TF-IDF pipeline that flags spam and routes messages into 6 intents | TF-IDF (1-2 grams), Naive Bayes, Linear SVM, KNN, precision/recall/F1, Streamlit | table 13 |
| 04 | Customer Segmentation Engine | `week 6/Class 2 Project` | `customer-segmentation-engine` | Unsupervised K-Means personas, PCA compression and Apriori basket rules in one app | K-Means (elbow, silhouette), PCA, Apriori (support/confidence/lift), mlxtend, Plotly | table 15 |
| 05 | High-Accuracy MNIST Digit Classifier | `week 8/Class 2 Project` (baseline CNN: `week 8/Class 1 Project`) | `mnist-cnn-optimization-lab` | Three CNN variants compared side by side to show how regularization closes the overfitting gap (>98% target) | TensorFlow/Keras CNN, BatchNorm, Dropout, EarlyStopping, ReduceLROnPlateau, Streamlit canvas | table 11 |
| 06 | Real-Time Video Privacy Shield | `week 9/Class 2 Project` | `realtime-face-privacy-shield` | Live webcam app that detects and anonymizes faces in real time with five privacy modes | OpenCV (Haar Cascade), threaded capture + frame queue, blur/pixelate/mask, Streamlit | table 12 |
| 07 | Semantic CV-to-Job Matcher | `week 10/Class 1 Project` | `semantic-cv-job-matcher` | Ranks CVs (text/PDF/DOCX) against a job description by meaning, not keyword overlap | sentence-transformers (paraphrase-MiniLM-L3-v2), cosine similarity, pypdf, python-docx, lazy model loading | table 9 |
| 08 | Local Customer Feedback Analyzer | `week 10/Class 2 Project` | `local-review-sentiment-analyzer` | Local 3-class review sentiment with low-confidence flagging for human review | HuggingFace Transformers (twitter-roberta 3-class), batched inference, confidence thresholds, dashboard | table 10 |
| 09 | Zero-Shot Brand Content Generator | `week 11/Class 1 Project` | `local-brand-voice-generator` | On-brand marketing copy from a local LLM, hardened against prompt injection | Ollama llama3.1:8b, zero/few-shot, chain-of-thought, temperature/top-p, prompt-injection defense | table 8 |
| 10 | Local Document Q&A Bot (RAG Engine) | `week 12/Class 2 Project` | `local-pdf-rag-qa` | Answers strictly from uploaded PDFs with page citations and an honest "not found" | ChromaDB, chunking with overlap, metadata filters, distance guardrail, llama3.1:8b, Streamlit | table 7 |
| 11 | Deterministic Function-Calling Travel Agent | `week 13/Class 2 Project` | `react-travel-agent-tool-calling` | Hand-written ReAct agent that searches flights and books only after human approval | Tool calling, ReAct, Pydantic validation, stop reasons, repeat guard, human-in-the-loop, golden-task eval | **pin 6** |
| 12 | Self-Correcting LLM Coder | `week 14/Class 1 Project` | `self-correcting-llm-coder-langgraph` | LangGraph loop where an LLM writes code, runs tests in a subprocess and repairs itself | LangGraph StateGraph, subprocess sandbox, AST checks, hidden tests, interrupt(), SQLite checkpoints | **pin 5** |
| 13 | Collaborative Multi-Agent Research Team | `week 14/Class 2 Project` | `crewai-grounded-research-team` | Researcher/Writer/Reviewer crew whose every citation is verified by code before publishing | CrewAI Flow, BM25 retrieval, guardrails, citation verification, single-agent baseline, fake-LLM tests | **pin 4** |
| 14 | Production-Ready RAG API Service | `week 15/Class 1 Project` (+ monitoring: `week 15/Class 2 Project`) | `rag-api-service-fastapi` | Async FastAPI RAG service with streaming, backpressure and graceful degradation | FastAPI, SSE, asyncio.Semaphore, idempotent ingestion, caching, health/ready, Docker, benchmarking | **pin 2** |
| 15 | Offline Market Intelligence Team | `week 16/Class 1-2 Project` | `offline-market-intelligence-agents` | Two CrewAI agents turn an offline news + price snapshot into a deterministic risk score and a PDF memo | CrewAI, FinBERT sentiment, deterministic Risk Score (volatility, drawdown, VaR), PDF generation, FastAPI | **pin 3** |
| 16 | Local Knowledge-Grounded Healthcare Assistant | `week 17/Class 1-2 Project` (+ deployment: `week 18/Class 1-2 Project`) | `private-healthcare-rag-assistant` | Privacy-first medical Q&A grounded in a local knowledge base, with safety rules decided before the model runs | ChromaDB + llama3.1:8b RAG, PII redaction, red-flag escalation, refusal and abstention, eval set, FastAPI | **pin 1** |

## Alternative pin sets

Pick the set that matches the job description. Everything not pinned stays in the profile README table.

| Target role | Pin 1 | Pin 2 | Pin 3 | Pin 4 | Pin 5 | Pin 6 |
|---|---|---|---|---|---|---|
| **LLM / AI Engineer** (default) | #16 Healthcare RAG | #14 RAG API | #15 Market Intelligence | #13 Research Team | #12 Self-Correcting Coder | #11 Travel Agent |
| **ML Engineer** (classic + DL) | #14 RAG API | #05 MNIST | #08 Feedback Analyzer | #02 Real Estate | #03 Spam & Intent | #04 Segmentation |
| **Computer Vision / Applied ML** | #06 Privacy Shield | #05 MNIST | #07 CV Matcher | #10 PDF RAG | #14 RAG API | #02 Real Estate |

## What each project still lacks (easy personal extensions)

An extension you made yourself is your best answer to "which part did you write?".

| # | Gap in the course build | Extension idea |
|---|---|---|
| 01 | `projects.md` mentions Streamlit and Euclidean distance; the course build is a CLI using cosine only | Add a small Streamlit view and a Euclidean vs cosine comparison |
| 02 | Smoke test only (`test_app.py`), no pytest suite; synthetic data | Add pytest for `model_utils`; try a real public housing dataset |
| 03, 04 | Synthetic data | Run on a public dataset and report the metric change |
| 05 | README lists "typical" numbers, not measured ones | Record your own measured table (hardware, epochs, seed) |
| 06 | No tests | Unit-test `privacy_engine` modes on synthetic frames |
| 09 | Injection defense is heuristic | Build a small attack set and report blocked / slipped through |
| 10 | Threshold calibrated on the course corpus | Recalibrate on your own PDFs and document the distance gap |
| 11-14 | Live numbers are the author's machine | Re-run `--eval` / `--bench` / `bench.py` and publish your own table |
| 15, 16 | No reference live numbers at all (static verification, ADR 0008) | Run the eval on your machine and publish the first results table |
| all | No CI | A GitHub Actions workflow that runs `pytest` on every push |

## Optional extra repos (not Portfolio Projects)

These are full projects too, but not among the 16. Merge them into the related portfolio repo, or keep them as extra (unpinned) repos.

| Course folder | What it is | Suggestion |
|---|---|---|
| `week 13/Class 1 Project` | Tool-calling agent: math + sandboxed file inspector (31 tests) | Separate small repo, or skip |
| `week 15/Class 2 Project` | Prometheus + OpenTelemetry monitoring, quality flags, SLO burn rates for the #14 API (43 tests) | Merge into the #14 repo as the "Monitoring" section |
| `week 18/Class 1-2 Project` | One-command `docker compose up` for the #16 stack (UI + API + Chroma + Ollama), optional GPU override | Merge into the #16 repo as the "Deployment" section |
