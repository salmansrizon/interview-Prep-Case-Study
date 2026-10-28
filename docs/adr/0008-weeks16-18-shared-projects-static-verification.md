# 0008. Weeks 16-18: one Shared Project per week, verified statically

## Status
Accepted (extends ADR 0007 to weeks 16-18; amends its live-verification rule)

## Context
From Week 16, `Course Module.pdf` pairs two classes under one achievement ("Class 31 & 32", "Class 33 & 34", "Class 35 & 36"). Until Week 15, every Class Session had its own achievement, Lecture and Project. Splitting one system into two staged Projects would force the second to copy the first (ADR 0002: Projects are self-contained). Writing one Lecture per week would break the one-Lecture-per-Class-Session mapping.

ADR 0007 says notebooks are executed live against Ollama when it is available. On the machine building weeks 16-18, Ollama is not installed, the Docker daemon is not running and there is no NVIDIA GPU.

## Decision
1. **Two Lectures + one Shared Project per week.** `Class 1 Lecture/` and `Class 2 Lecture/` (one per Class Session, each self-contained), plus one `Class 1-2 Project/` that holds the whole achievement. Lecture 1 teaches the first half of the system, Lecture 2 the second half.
2. **Full project depth (ADR 0007 decision 1) applies to weeks 16-18**, including Week 18, which is not a Portfolio Project.
3. **Weeks 16-17 projects use FastAPI + a thin Streamlit UI + `cli.py`.** Week 15 taught serving, and Week 18 needs a real UI + API + DB + LLM stack to containerize.
4. **Verification is static + mocked.** Every `pytest` suite is fully mocked and is run. Notebooks are written to run top to bottom, but are not executed against a live LLM. LLM-dependent cells say their output is illustrative. Docker files are linted/reviewed but not built or run here. GPU passthrough is documented but not tested.

## Consequences
- Weeks 16-18 notebooks have no saved live outputs, unlike weeks 13-15. Re-running them on a machine with Ollama (and committing the outputs) would bring them to the ADR 0007 standard; this ADR does not forbid that.
- READMEs carry no "Live Results" table with measured numbers; where one would go, they say what to measure and how.
- Week 18's Docker stack is unverified until someone runs `docker compose up` on a machine with the daemon running.
