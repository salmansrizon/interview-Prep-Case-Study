# Job Ready AI Bootcamp — Glossary

## Class Session
One live class in `Course Module.pdf` (e.g. "Class 17"). Maps to exactly one `Class N Lecture` folder and, where the curriculum assigns an achievement, one `Class N Project` folder under `Job Ready AI Bootcamp/week X/`.

## Lecture
The teaching material for one Class Session: a `context.md` (Bangla-humanized concept explainer) and, for code-heavy topics, a companion `.ipynb`. A Lecture must be self-contained — understandable and runnable without reading any other class's Lecture or Project.

## Project
The hands-on deliverable folder for one Class Session. A Project must be self-contained — runnable on its own (own README, own dependencies, own source tree) without requiring another class's Project folder to be present. See [[self-contained-class-artifacts]].

## Portfolio Project
One of the 16 entries in `projects.md`. Each maps 1:1 to a specific Class Session's achievement in `Course Module.pdf` and is written to interview-showcase quality (production-grade structure, README with architecture diagram, tests). Not every Class Session produces a Portfolio Project — some (e.g. EDA/statistics weeks) only produce a Lecture.

## Non-Portfolio Class Project
A Project belonging to a Class Session whose achievement is not one of the 16 `projects.md` entries. **Rule (settled at Week 12):** every one of the 16 `projects.md` entries gets a real, production-grade portfolio scaffold; every other Class Project gets a notebook demo only (no README/app.py/src/tests) — regardless of how technically close it is to an adjacent portfolio project. This is a blanket policy, not a case-by-case judgment call — don't re-litigate it per week. Matches the Week 3 Data Storyboard precedent. Week 4 Class 2 (Statistical Insight Report) predates this convention and got a full scaffold; that's a historical inconsistency, not the pattern to follow. First made explicit for Week 11 Class 22 (Resume-to-JSON), then generalized into this blanket rule at Week 12 (Class 23, vector DB indexing — direct prerequisite to Project 10, still notebook-only per this rule). **Exception (Weeks 13-15):** every class there, portfolio or not, gets a full runnable project, per `docs/adr/0007-weeks13-15-full-projects-and-live-verification.md`.

## Curriculum Deviation
A case where a Class Session's actual Lecture/Project content differs from what `Course Module.pdf` states for that class. Always recorded as an ADR at the time of the deviation, not left implicit. See `docs/adr/0003-week9-class1-curriculum-deviation-emoji-cam.md` for the first instance.

## Local Model
A model whose weights are downloaded once (typically from HuggingFace Hub, on first run) and then run entirely on the student's own machine with no further network calls — distinct from a Cloud API Call, which sends data to a third party on every request. This course's "Sovereign AI" positioning is about avoiding Cloud API Calls for inference, not about avoiding the internet entirely — a one-time model download is consistent with "local," a live API dependency is not. See `docs/adr/0004-week10-class19-embedding-model.md` for where this distinction first had to be made explicit.
