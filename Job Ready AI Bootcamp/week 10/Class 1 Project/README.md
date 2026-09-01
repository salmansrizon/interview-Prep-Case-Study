# 🧭 Semantic CV-to-Job Matcher

## সহজ ভাষায় Project Overview

**🧭 Semantic CV-to-Job Matcher** project-এ lecture-এর theory-কে working software বা executable notebook-এ convert করা হয়েছে। লক্ষ্য শুধু final output দেখা নয়; input থেকে preprocessing, core logic/model, evaluation এবং output—পুরো pipeline বোঝা।

### কোন Problem Solve করে?

Manual বা disconnected workflow-কে repeatable code pipeline-এ আনে। এর ফলে একই process নতুন data-তে আবার চালানো, result compare করা, error trace করা এবং future feature add করা সহজ হয়।

### কীভাবে কাজ করে?

Input/Data → Validation ও Preprocessing → Core Algorithm/Model → Evaluation → UI, Report বা Saved Output। নিচের detailed section-গুলোতে project-specific command, feature এবং architecture দেওয়া আছে।

### কেন এই Approach ভালো?

- **Repeatable:** একই input দিলে একই workflow follow করে।
- **Testable:** প্রতিটি stage আলাদাভাবে verify করা যায়।
- **Explainable:** কোন step কী কাজ করছে তা code এবং output দিয়ে দেখা যায়।
- **Portfolio-ready:** শুধু notebook result নয়, setup, structure এবং usage-সহ complete project হিসেবে দেখানো যায়।

> **Run করার নিয়ম:** আগে virtual environment তৈরি করে dependency install করুন। তারপর README-এর Quick Start follow করুন, sample input দিয়ে smoke test করুন এবং expected metric/output-এর সাথে result compare করুন।

---

A production-grade semantic search app that ranks a batch of CVs against a job
description by **meaning**, not keyword overlap — built with
`sentence-transformers`, NumPy, and Streamlit, for **Week 10, Class 1: NLP
Basics & Transformers**.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-2.2+-orange.svg)
![Tests](https://img.shields.io/badge/tests-pytest-green.svg)

## ✨ Features

- 🧠 **Semantic matching, not keyword search** — "ML", "machine learning", and
  "deep learning" all land close together in embedding space, so a CV that
  never uses the JD's exact wording can still rank #1
- 📄 **Flexible input** — paste plain text or upload **PDF/DOCX** for both the
  job description and CVs, several CVs at once
- 📊 **Ranked results** — a sortable table + bar chart of cosine-similarity
  scores, with the top match highlighted
- 🪶 **Small, fast, local model** — `paraphrase-MiniLM-L3-v2` (~61MB) runs
  real-time on CPU, no GPU, no API key
- 🔌 **Offline after first run** — the model downloads once from the
  HuggingFace Hub, then every subsequent run is 100% local (see
  [Model Choice](#-model-choice--why-paraphrase-minilm-l3-v2) below)
- 🧮 **Hand-rolled cosine similarity** — the exact dot-product math from the
  Week 2 NumPy lecture, applied to sentence vectors instead of raw numbers
- ✅ **Tested** — fast pytest suite covering extraction (plain/PDF/DOCX) and
  ranking correctness with hand-crafted vectors, no real model load required
- 🧩 **Clean layered architecture** — `app.py` never touches
  `sentence-transformers` or file-parsing libraries directly; it only calls
  into `src/`

## 🚀 Quick Start

```bash
# 1. Navigate to the project folder
cd "Job Ready AI Bootcamp/week 10/Class 1 Project"

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. On the **Match** tab, paste (or
upload) a job description and one or more CVs, then click **Match** — the
first click will download the embedding model (~61MB, needs internet once);
every click after that runs fully offline.

### Run the tests

```bash
pytest tests/ -v
```

Tests never load the real model or touch the network — see
[`src/embeddings/service.py`](src/embeddings/service.py) for why lazy loading
makes that possible.

## 📁 Project Structure

```
Class 1 Project/
├── app.py                       # Streamlit entry point (UI only, no ML/parsing calls)
├── config.py                    # Dataclass-based configuration (model, upload limits, display)
├── requirements.txt
├── README.md
├── notebooks/
│   └── 01_exploration.ipynb     # Load model, embed toy examples, sanity-check ranking
├── src/
│   ├── embeddings/
│   │   └── service.py           # Lazy-loaded SentenceTransformer wrapper — embed()
│   ├── extraction/
│   │   └── text_extractor.py    # plain/PDF/DOCX bytes -> plain text
│   ├── matching/
│   │   └── ranker.py            # Hand-rolled cosine similarity + rank_cvs()
│   └── utils/
│       └── logger.py            # Shared structured logging
└── tests/
    └── test_pipeline.py         # Extraction + ranking correctness tests
```

## 🔧 Architecture

```
┌─────────────────────┐        ┌──────────────────────┐
│  Job Description      │        │   CV #1, #2, #3, ...   │
│  (paste or PDF/DOCX)  │        │  (paste or PDF/DOCX)   │
└──────────┬───────────┘        └───────────┬───────────┘
           │                                 │
           ▼                                 ▼
┌────────────────────────────────────────────────────────┐
│           src/extraction/text_extractor.py               │
│   plain text / PDF text-layer / DOCX paragraphs+tables    │
│              →  one plain-text string each                │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            src/embeddings/service.py — embed()            │
│   paraphrase-MiniLM-L3-v2 (lazy-loaded, cached singleton)  │
│         text  →  384-dim vector, per document              │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            src/matching/ranker.py — rank_cvs()             │
│     cosine_similarity(job_vec, cv_vec)  for each CV         │
│        (a · b) / (‖a‖ · ‖b‖)  →  sorted best-first          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                    app.py — Match tab                      │
│      Ranked table + bar chart, top match highlighted        │
└────────────────────────────────────────────────────────┘
```

## ⚙️ Configuration

All settings live in [`config.py`](config.py) as frozen dataclasses — one
place to change the model, upload limits, or display precision without
hunting through `app.py`/`src/`.

| Setting | Default | Description |
|---|---|---|
| `embedding.model_name` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | Which sentence-embedding model to load |
| `embedding.embedding_dim` | `384` | Output vector size (recorded, not discovered at runtime) |
| `embedding.encode_batch_size` | `16` | Batch size for `.encode()` |
| `upload.max_upload_size_mb` | `10` | UI-level guardrail for uploads |
| `upload.allowed_extensions` | `("pdf", "docx", "txt")` | Accepted upload formats |
| `matching.similarity_precision` | `4` | Decimal places shown in the results table |
| `matching.min_text_length_chars` | `20` | Below this, extraction is treated as probably junk |
| `app.page_title` / `page_icon` / `layout` | `"Semantic CV-to-Job Matcher"` / `"🧭"` / `"wide"` | Streamlit page settings |

## 🧠 Model Choice — Why `paraphrase-MiniLM-L3-v2`?

This project embeds text with
[`sentence-transformers/paraphrase-MiniLM-L3-v2`](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2)
(see `config.py`'s `EmbeddingConfig.model_name`, read by
[`src/embeddings/service.py`](src/embeddings/service.py)). It's a
**portfolio-scale CV matcher** — ranking a handful of CVs against one job
description — so CPU-only, real-time response matters more than the last
couple points of accuracy a heavier model would buy.

### Model Reference Table

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`paraphrase-MiniLM-L3-v2`** (used here) | Fast, local semantic similarity | Small (~61MB), fast on CPU | Slightly less accurate than larger models |
| `all-MiniLM-L6-v2` | General-purpose sentence embeddings | Good accuracy, still small | ~50% larger than the L3 variant, a bit slower |
| `all-mpnet-base-v2` | High-accuracy semantic search | Most accurate in this model family | Much heavier (~420MB), slow on CPU |
| OpenAI `text-embedding-3-small` | Cloud-based embeddings | No local compute required | Not local — per-call cost, data leaves the machine |

> A production system serving many concurrent users, where every extra point
> of accuracy matters, would be a reasonable place to upgrade to
> `all-mpnet-base-v2` — that's a one-line change in `config.py`'s
> `EmbeddingConfig`, not a code change anywhere else.

### "Local Model" — first run downloads once, then fully offline

`paraphrase-MiniLM-L3-v2` is downloaded from the HuggingFace Hub **the first
time it's used** (needs internet), then cached on disk
(`~/.cache/torch/sentence_transformers/`). Every run after that is **100%
local** — no network call, no per-request cost, unlike a cloud embedding API
(e.g. OpenAI's `text-embedding-3-small`) where every request leaves the
machine. This app makes that observable, not just claimed: the model is
never loaded at import time (see `src/embeddings/service.py`'s lazy-loading
`get_model()`), only the first time you click **Match** — so simply running
`pytest` or opening the app never silently triggers a download.

## 🛠️ Development

### Adding a new file format to extract

Add a new `extract_text_from_<format>()` function in
`src/extraction/text_extractor.py` and one branch in the `extract_text()`
dispatcher — `app.py` needs no changes since it only ever calls the
dispatcher.

### Swapping the embedding model

Change `EmbeddingConfig.model_name` (and `embedding_dim`, if it differs) in
`config.py`. Nothing else in `src/` or `app.py` hardcodes the model name.

## 📜 License

MIT License — feel free to use in your own projects!
