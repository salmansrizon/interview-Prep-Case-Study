# 0006. Standardize on Ollama + llama3.1:8b as the course-wide local LLM

## Status
Accepted

## Context
Starting at Week 11 (Class 21, Project 09), and recurring through Weeks 13-14 and 16-17, multiple projects need a local, offline LLM for generation, tool-calling, and agentic reasoning. `Course Module.pdf` names "Mistral/Llama 3.1" explicitly for Week 13's function-calling achievement, and Week 1 already introduced Ollama as the local-LLM runtime. Without picking one default now, each future week would re-litigate model choice, producing inconsistent requirements/READMEs across the course.

## Decision
Standardize on **Ollama** as the local LLM runtime for every project needing generative text, and **`llama3.1:8b`** as the default model, with **`llama3.2:3b`** documented as the fallback for students on the course's stated minimum spec (8GB RAM — an 8B model quantized via Ollama's default Q4 is workable on 16GB but tight on 8GB). Every project using this must document `ollama pull <model>` as a prerequisite setup step, not something the app installs itself.

## Consequences
- Background build agents in this environment cannot reach a live Ollama server, so LLM-dependent code is verified statically (syntax/structure) and tests must mock/stub Ollama calls rather than hit a real local server — this is a standing constraint for every future week that uses this decision, not just Week 11.
- READMEs for these projects must state the model pull step plainly, matching the "one-time setup, then fully offline" framing already established for HuggingFace models in `docs/adr/0004`.
- If a future week's curriculum explicitly names a different model (e.g. a vision-language model), that's a scoped exception, not a reason to revisit this default for text generation.
