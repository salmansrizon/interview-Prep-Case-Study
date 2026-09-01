# 📚 Local Document Q&A Bot (RAG Engine)

## সহজ ভাষায় Project Overview

**📚 Local Document Q&A Bot (RAG Engine)** project-এ lecture-এর theory-কে working software বা executable notebook-এ convert করা হয়েছে। লক্ষ্য শুধু final output দেখা নয়; input থেকে preprocessing, core logic/model, evaluation এবং output—পুরো pipeline বোঝা।

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

A production-grade **Retrieval-Augmented Generation** app that answers
questions **strictly from PDFs you upload** — with source + page citations,
a real per-document search filter, and an honest "not found in the
documents" guardrail when nothing relevant is indexed. Built with
**ChromaDB**, `sentence-transformers/paraphrase-MiniLM-L3-v2`, **Ollama**
(`llama3.1:8b`), and Streamlit, for **Week 12, Class 2: RAG Pipelines**.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![ChromaDB](https://img.shields.io/badge/chromadb-local--persistent-orange.svg)
![Ollama](https://img.shields.io/badge/ollama-llama3.1%3A8b-black.svg)
![Tests](https://img.shields.io/badge/tests-pytest-green.svg)

## ✨ Features

- 📤 **Multi-PDF upload** — chunk and index any number of PDFs in one go,
  with a live progress indicator per file
- ✂️ **Recursive, paragraph-aware chunking with overlap** — never a naive
  fixed-size cut mid-sentence; overlap keeps context from being lost right
  at a chunk border (Class 1 Lecture, section 3.4)
- 🏷️ **Real metadata, not decoration** — every chunk stores `{source, page}`,
  and that metadata powers **two** actual features: citations on every
  answer, and a document-picker filter that constrains the ChromaDB `where`
  clause (not a cosmetic UI control that filters results after the fact)
- 🎯 **Document-scoped search** — pick "search all documents" or restrict
  retrieval to one specific uploaded PDF
- 🚫 **Two-layer "answer strictly from documents" guardrail** — (1) a
  similarity-distance threshold check that skips calling the LLM entirely
  when nothing relevant was retrieved, PLUS (2) an Open-Book-Exam-style
  system prompt instructing the model to say "not found" rather than guess
- 📖 **Citations on every answer** — which document, which page, shown
  alongside the generated answer
- 🏠 **100% local, offline after one-time setup** — no API key, no data
  leaves your machine, per this course's Sovereign AI philosophy
- ✅ **Tested with Ollama fully mocked** — retrieval/chunking/guardrail
  tests run against a real local ChromaDB; the LLM call is stubbed, per
  this environment's standing testing constraint (`docs/adr/0006`)
- 🧩 **Clean layered architecture** — `app.py` never imports `chromadb`,
  `sentence_transformers`, `ollama`, or `pypdf` directly; it only calls `src/`

## 🚀 Quick Start

```bash
# 1. Navigate to the project folder
cd "Job Ready AI Bootcamp/week 12/Class 2 Project"

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. One-time model pull for generation (needs internet; ~4.7GB download)
ollama pull llama3.1:8b
# On an 8GB-RAM machine, pull the smaller fallback instead:
# ollama pull llama3.2:3b   (then set LLMConfig.model in config.py to match)

# 5. Start the Ollama server (if not already running)
ollama serve

# 6. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Go to **📤 Upload & Index**, upload
one or more PDFs, then go to **💬 Ask** and ask a question. The first
indexing click downloads the embedding model (~61MB, needs internet once);
every run after that — indexing AND answering — is fully offline (once
Ollama is running locally).

### Run the tests

```bash
pytest tests/ -q
```

Tests never contact a real Ollama server — `src/llm/client.py`'s
`generate()` imports the `ollama` package *inside* the function body
specifically so a fake module can be substituted into `sys.modules['ollama']`
before it runs. Chunking and vector-store tests run against a **real,
temporary ChromaDB PersistentClient** with hand-crafted embedding vectors —
no model download needed, no mocking of ChromaDB itself, since it's already
100% local.

## 📁 Project Structure

```
Class 2 Project/
├── app.py                          # Streamlit entry point (UI only, no direct lib calls)
├── config.py                       # Dataclass-based configuration (models, chunking, guardrail)
├── requirements.txt
├── README.md
├── data/
│   └── chroma_db/                  # ChromaDB's on-disk persistent store (created at runtime)
├── notebooks/
│   └── 01_exploration.ipynb        # Extraction/chunking/indexing/retrieval walkthrough, LLM not run
├── src/
│   ├── extraction/
│   │   └── pdf_extractor.py        # Page-aware PDF text extraction (pypdf, no OCR fallback)
│   ├── chunking/
│   │   └── chunker.py              # Recursive/paragraph-aware chunking with overlap + page metadata
│   ├── embeddings/
│   │   └── service.py              # Lazy-loading singleton for paraphrase-MiniLM-L3-v2
│   ├── vectorstore/
│   │   └── store.py                # ChromaDB wrapper — index/query/list_indexed_sources
│   ├── llm/
│   │   └── client.py               # The ONLY module that imports `ollama` — generate()
│   ├── rag/
│   │   └── pipeline.py             # Orchestration: embed -> retrieve -> guardrail -> augment -> generate
│   └── utils/
│       └── logger.py               # Shared structured logging
└── tests/
    └── test_pipeline.py            # Chunking/retrieval/guardrail/citation tests, Ollama fully mocked
```

## 🔧 Architecture

```
   Multiple PDFs (upload)
          │
          ▼
┌──────────────────────────────────────────┐
│  src/extraction/pdf_extractor.py          │
│  extract_pages() -> [(page_number, text)] │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│  src/chunking/chunker.py                  │
│  chunk_page_texts() -> [Chunk(text,       │
│      source, page)] — recursive,          │
│      paragraph-aware, with overlap        │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│  src/embeddings/service.py                │
│  embed() -> vectors (paraphrase-MiniLM-L3-v2)│
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│  src/vectorstore/store.py                 │
│  index_chunks() -> ChromaDB (persisted    │
│      to ./data/chroma_db)                 │
└──────────────────────┬─────────────────────┘
                       │
        ═══════════════╪═══════════════  [ user asks a question ]
                       │
                       ▼
┌──────────────────────────────────────────┐
│  src/rag/pipeline.py — answer_question()  │
│  [1] embed the question                   │
│  [2] retrieve top-K via                   │
│      src/vectorstore/store.py::query()    │
│      (optional source_filter -> ChromaDB  │
│       `where` clause)                     │
│  [3] GUARDRAIL: best distance too high?   │
│      -> return "not found", skip LLM      │
│  [4] build augmented (Open-Book-Exam)     │
│      prompt with retrieved context        │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│  src/llm/client.py — generate()           │
│  ONLY module that imports `ollama`        │
│  model=llama3.1:8b, low temperature       │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│  app.py — Ask tab                         │
│  Answer + Citations (source, page)        │
│  or the honest "not found" message        │
└──────────────────────────────────────────┘
```

## ⚙️ Configuration

All settings live in [`config.py`](config.py) as frozen dataclasses.

| Setting | Default | Description |
|---|---|---|
| `embedding.model_name` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | Local embedding model (ADR 0004) |
| `embedding.embedding_dim` | `384` | Output vector dimensionality |
| `chunking.chunk_size_chars` | `700` | Target chunk size (chars) — mid-range of the lecture's 300-800 sweet spot |
| `chunking.chunk_overlap_chars` | `100` | ~14% overlap between consecutive chunks |
| `chunking.min_chunk_size_chars` | `40` | Discard leftover fragments smaller than this |
| `vectorstore.persist_directory` | `./data/chroma_db` | ChromaDB on-disk storage location |
| `vectorstore.collection_name` | `document_qa_chunks` | ChromaDB collection name |
| `llm.model` | `llama3.1:8b` | Default Ollama model (course standard, ADR 0006) |
| `llm.fallback_model` | `llama3.2:3b` | Low-RAM (8GB) fallback |
| `llm.default_temperature` | `0.2` | Low — RAG answers should be grounded, not creative |
| `retrieval.top_k` | `4` | Chunks retrieved per question (K=3-5 sweet spot, lecture section 4.3) |
| `retrieval.max_distance_threshold` | `22.0` | The "not found" guardrail's distance cutoff (see below) |

## 🚫 "Answers Strictly From Documents" — How It's Actually Enforced

This is **not** just prompt wording. It's two independent, testable layers:

1. **Similarity-threshold guardrail** (`src/rag/pipeline.py::answer_question`)
   — ChromaDB's vector search always returns *something* (its nearest
   neighbors), even for a completely off-topic question, because it has no
   concept of "nothing matches." Before ever calling the LLM, the pipeline
   checks the BEST retrieved chunk's distance against
   `config.retrieval.max_distance_threshold`. If it's too high, the app
   returns the honest "not found in the documents" message **without
   spending a single LLM call** — this is directly unit-tested in
   `tests/test_pipeline.py` with a synthetic high-distance result.
2. **System-prompt instruction** — even when relevant chunks ARE retrieved,
   the model is told explicitly (Open-Book-Exam framing, Class 2 Lecture
   section 3.2) to answer only from the given context and to say so plainly
   if the context doesn't actually contain the answer.

### Tuning the Guardrail Threshold

ChromaDB's default distance metric here is **L2 (squared Euclidean)** on
un-normalized `paraphrase-MiniLM-L3-v2` embeddings — not cosine distance.
`22.0` was picked empirically against this project's own test corpus:
clearly-relevant questions scored **~10-19**, clearly off-topic questions
scored **~30-37**, so `22.0` sits in the gap. This is a heuristic, not a
guarantee — a lightweight local embedding model can occasionally mis-score
an oddly-worded *relevant* question as "not found." That's the honest
trade-off of a small, fully-local model versus a larger cloud embedding API
(see the Model Reference Table below), and it's exactly the "Garbage In,
Garbage Out" caveat from Class 1 Lecture section 4.2 in action. If you index
a very different corpus, re-run a few known-relevant and known-irrelevant
questions and adjust `max_distance_threshold` accordingly.

## 🗂️ Vector Database Reference Table

(Adapted from Class 1 Lecture, section 4.1)

| Database | Use Case | Advantage | Limitation |
|---|---|---|---|
| **ChromaDB** (used here) | Local/small-to-medium scale projects | No server setup, Python-native | Limited performance at very large (10M+ vector) scale |
| FAISS | Research, very large scale, in-memory | Extremely fast, GPU support | Doesn't manage metadata/persistence itself — extra code needed |
| Pinecone | Production, managed cloud service | Scalable, managed infrastructure | Cloud-based — not local, subscription cost, data leaves the machine |
| Weaviate | Production, built-in hybrid search | Self-host or cloud, powerful filtering | More complex setup than ChromaDB |

> ChromaDB is the right choice here because it's fully local — saved to disk
> as files, no server or cloud account needed — matching this course's
> Sovereign AI philosophy. Moving to FAISS/Pinecone/Weaviate is reasonable
> once you outgrow portfolio scale, but ChromaDB is the best entry point for
> learning and this kind of project.

## 🧠 LLM Model Reference Table

(Adapted from Week 11 lecture, section 4.1)

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`llama3.1:8b`** (used here, default) | General-purpose text generation, instruction-following | Strong output quality, runs comfortably on 16GB RAM | Tight on 8GB RAM machines, can be slow there |
| `llama3.2:3b` (fallback) | Low-resource machines | Small, runs fine even on 8GB RAM | Less accurate than the 8B model, weaker on complex instructions |
| `mistral:7b` | Fast inference, code/structured tasks | Similar size to llama3.1, faster on some benchmarks | Smaller community/fine-tune ecosystem than Llama |
| OpenAI GPT-4 / Claude (Cloud API) | When you need best-in-class output quality | State-of-the-art benchmark performance | Not local — per-call cost, data leaves the machine, conflicts with Sovereign AI philosophy |

> `llama3.1:8b`'s instruction-following is what makes the "answer only from
> context, admit when you can't" guardrail reliable in practice — a weaker
> model is more likely to ignore that instruction and hallucinate anyway. An
> 8GB-RAM student should switch to `config.llm.fallback_model`
> (`llama3.2:3b`) — one line in `config.py`, no code changes needed anywhere
> else.

## 🧭 The "Open-Book Exam" Analogy

| RAG Concept | Open-Book Exam Analogy |
|---|---|
| **Vector DB (indexing)** | Indexing the whole book ahead of time (building a table of contents), so the right page is fast to find during the exam. |
| **Retrieval** | Reading the exam question and figuring out which 3-4 pages of the book are relevant. |
| **Augmentation** | Opening those pages on the desk before writing the answer. |
| **Generation** | Writing the answer from those open pages — not from memory. |
| **"Strictly from documents" guardrail** | The exam rule: "Only write what's in the book. Don't add outside knowledge." |
| **Metadata Filtering** | The instruction "only answer from Chapter 3" — narrowing focus to one section instead of the whole book. |
| **Citations** | Writing "per page 45..." next to the answer, so the grader (or user) can verify it. |

## 🛠️ Development

### Swapping the embedding or LLM model

Change `EmbeddingConfig.model_name` or `LLMConfig.model`/`fallback_model` in
`config.py`. Nothing else in `src/` or `app.py` hardcodes either model name.

### Adjusting chunk size / overlap

Change `ChunkingConfig.chunk_size_chars` / `chunk_overlap_chars` in
`config.py` — see the Class 1 Lecture Brain Teaser #1 ("Chunk Size
Experiment") for what to try.

### Adjusting the guardrail

Change `RetrievalConfig.max_distance_threshold` in `config.py` — see
"Tuning the Guardrail Threshold" above before picking a new value.

## 📜 License

MIT License — feel free to use in your own projects!
