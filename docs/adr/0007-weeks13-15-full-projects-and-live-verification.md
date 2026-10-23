# 0007. Weeks 13-15: full runnable projects for every class, verified live

## Status
Accepted (amends the Non-Portfolio Class Project rule in `CONTEXT.md`)

## Context
Weeks 13-15 cover tool calling, ReAct, LangGraph, CrewAI, FastAPI serving and LLM monitoring. Two of these six classes are not among the 16 `projects.md` portfolio entries: Week 13 Class 1 (Math & File Inspector) and Week 15 Class 2 (Monitoring). The Week 12 blanket rule says such classes get a notebook demo only.

The course owner asked for weeks 13-15 to be deepened to the depth of weeks 1-12, with deep research behind every topic and a **runnable project for every topic**. For agent topics, a notebook alone hides the parts students most need to see: an app that runs the loop, a test suite that mocks the LLM, and the failure modes of a real local model.

ADR 0006 also says LLM-dependent code is verified statically because build agents cannot reach Ollama. For this work, a local Ollama server (0.40.x with `llama3.1:8b`) was available.

## Decision
1. **Every class in weeks 13-15 gets a full project scaffold**, portfolio or not: README with architecture and config table, `app.py` (Streamlit for weeks 13-14, FastAPI for week 15, where serving is the topic), `config.py`, `requirements.txt`, `notebooks/01_exploration.ipynb`, `src/<domain>/` packages and `tests/test_pipeline.py`. Weeks 13-14 keep a `cli.py` as a second entry point.
2. **Projects stay self-contained (ADR 0002).** Week 15 projects no longer import Week 12's code via `sys.path`; each one carries its own trimmed RAG pipeline.
3. **Verify live when Ollama is available.** Every notebook (lecture and project) is executed top to bottom against local Ollama, with outputs saved. Every `pytest` suite stays fully mocked, so it runs without Ollama (ADR 0006 still holds for tests). Observed model failures are documented in READMEs and lectures, not hidden.
4. **Lecture notebooks are self-contained.** They do not import from the Project folder.

## Consequences
- The Non-Portfolio rule in `CONTEXT.md` now has a scoped exception for weeks 13-15. Weeks 1-12 are not retrofitted.
- Saved notebook outputs reflect one model, version and machine (Ollama 0.40.8, `llama3.1:8b`, temperature 0). Re-running can differ; the lectures explain why.
- More code to maintain in weeks 13-15. This is accepted in exchange for runnable, testable agent projects.
