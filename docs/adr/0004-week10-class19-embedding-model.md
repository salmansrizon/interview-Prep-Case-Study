# 0004. Semantic CV Matcher uses a local sentence-transformer model, not TF-IDF or a cloud API

## Status
Accepted

## Context
Project 07 (Semantic CV-to-Job Matcher, Class 19) is explicitly scoped by `Course Module.pdf` to match by *meaning*, not keyword overlap — ruling out TF-IDF/keyword-count approaches. A real sentence-embedding model is needed. This tensions against the course's repeated "100% local/offline/Sovereign AI" framing (see Projects 03, 10, 16), since any HuggingFace model requires a download from the Hub before it can run.

## Decision
Use `sentence-transformers/paraphrase-MiniLM-L3-v2` (the lighter 3-layer variant of the MiniLM family, ~61MB) loaded lazily inside the app/service layer on first use — not at import time, not in any setup/build script. The lecture explicitly teaches the distinction between "download the model weights once, then run fully offline forever" (this project, and the course's general Sovereign AI story) versus "call a cloud API on every single request" (what this project deliberately avoids) — framing the one-time download as consistent with, not a violation of, the local-AI value the course teaches elsewhere.

## Consequences
- `requirements.txt` for this project includes `sentence-transformers`, and the README must state the first-run download requirement plainly.
- The model choice is a genuine trade-off: lighter/faster than `all-MiniLM-L6-v2` at a small cost to embedding quality — acceptable for a portfolio-scale CV matcher, not necessarily for a production system with harder accuracy requirements.
- Any future project needing local embeddings should default to this same model unless there's a specific reason to deviate, for consistency across the course.
