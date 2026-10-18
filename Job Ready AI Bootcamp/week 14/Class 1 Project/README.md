# Self-Correcting Writer (Class 27 — LangGraph)

Python code লেখে, subprocess-এ test চালায়, fail করলে error সহ LLM-কে আবার পাঠায় (max `max_attempts` বার), শেষে চাইলে মানুষের approval নেয়।

```
START -> write_code -> run_tests --pass--> approval (interrupt) -> done | rejected
             ^             |--fail--> write_code (attempts < max) | give_up
```

## Run

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b        # model বদলাতে config.py
python app.py                  # demo task
python app.py "Write add(a,b)" --tests my_asserts.py --no-approval
pytest -q                      # Ollama ছাড়াই চলে (fake LLM)
```

## Structure

- `config.py` — model, retry limit, timeout, approval on/off + model reference table
- `src/graph.py` — `WriterState` (reducer সহ), nodes, conditional edges, `interrupt`
- `src/sandbox.py` — subprocess runner (timeout সহ), code fence extractor
- `src/llm.py` — Ollama wrapper (না থাকলে `OllamaUnavailable`)
- `tests/test_graph.py` — routing, retry limit, approval interrupt

সীমাবদ্ধতা: subprocess সত্যিকারের sandbox নয়; `InMemorySaver` process বন্ধ হলে state হারায় (production-এ SQLite/Postgres checkpointer)।
