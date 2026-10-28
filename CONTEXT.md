# Job Ready AI Bootcamp — Glossary

## Class Session
One live class in `Course Module.pdf` (e.g. "Class 17"). Maps to exactly one `Class N Lecture` folder and, where the curriculum assigns an achievement, one `Class N Project` folder under `Job Ready AI Bootcamp/week X/`.

## Lecture
The teaching material for one Class Session: a `context.md` (Bangla-humanized concept explainer) and, for code-heavy topics, a companion `.ipynb`. A Lecture must be self-contained — understandable and runnable without reading any other class's Lecture or Project.

## Project
The hands-on deliverable folder for one Class Session. A Project must be self-contained — runnable on its own (own README, own dependencies, own source tree) without requiring another class's Project folder to be present. See [[self-contained-class-artifacts]].

## Shared Project
The single Project for a week whose two Class Sessions share one achievement in `Course Module.pdf` (Weeks 16–18, e.g. "Class 31 & 32"). Lives in a `Class 1-2 Project` folder. Each of that week's two Class Sessions still gets its own self-contained Lecture; the Shared Project is not split or duplicated across the two sessions. See `docs/adr/0008-weeks16-18-shared-projects-static-verification.md`.

## Offline Snapshot
A bundled, versioned local copy of external data (e.g. Week 16's market prices and news) that a Project reads by default. Its agents and analysis run only against the local store. An optional ingestion step may refresh the snapshot from live sources. That refresh is the only network-touching step, and nothing downstream needs it to run.

## Risk Score
Week 16's 0–100 per-ticker number, computed deterministically by code from fixed quantitative metrics (volatility, drawdown, VaR) and news sentiment using documented weights. It is computed as of the last price date: news dated after that is excluded (no look-ahead). An agent may choose which tools to call and explain the result, but never sets or adjusts the number itself.

## News Scout Agent
Week 16's first agent. It surfaces relevant financial news for a ticker from the Offline Snapshot and attaches sentiment. `Course Module.pdf` calls it the "Researcher Agent"; this is the canonical name.
_Avoid_: Researcher Agent

## Risk Analyst Agent
Week 16's second agent. It obtains a ticker's Risk Score through deterministic tools and explains it in the investment memo. `Course Module.pdf` calls it the "Analyst Agent"; this is the canonical name.
_Avoid_: Analyst Agent

## Abstention
Week 17's required response when the local knowledge base doesn't support an answer: the assistant says the information isn't in its knowledge base instead of answering from model memory. An abstention is a correct outcome, not a failure.

## Red-Flag Escalation
Week 17's fixed, pre-written "seek emergency care" response, returned when a query matches an emergency rule (e.g. chest pain, suicidal ideation). It is decided by deterministic rules before any retrieval or model call, so the model never words or overrides it.

## Portfolio Project
One of the 16 entries in `projects.md`. Each maps 1:1 to a specific Class Session's achievement in `Course Module.pdf` and is written to interview-showcase quality (production-grade structure, README with architecture diagram, tests). Not every Class Session produces a Portfolio Project — some (e.g. EDA/statistics weeks) only produce a Lecture.

## Non-Portfolio Class Project
A Project belonging to a Class Session whose achievement is not one of the 16 `projects.md` entries. **Rule (settled at Week 12):** every one of the 16 `projects.md` entries gets a real, production-grade portfolio scaffold; every other Class Project gets a notebook demo only (no README/app.py/src/tests) — regardless of how technically close it is to an adjacent portfolio project. This is a blanket policy, not a case-by-case judgment call — don't re-litigate it per week. Matches the Week 3 Data Storyboard precedent. Week 4 Class 2 (Statistical Insight Report) predates this convention and got a full scaffold; that's a historical inconsistency, not the pattern to follow. First made explicit for Week 11 Class 22 (Resume-to-JSON), then generalized into this blanket rule at Week 12 (Class 23, vector DB indexing — direct prerequisite to Project 10, still notebook-only per this rule). **Exception (Weeks 13-18):** every class there, portfolio or not, gets a full runnable project, per `docs/adr/0007-weeks13-15-full-projects-and-live-verification.md` and `docs/adr/0008-weeks16-18-shared-projects-static-verification.md`.

## Curriculum Deviation
A case where a Class Session's actual Lecture/Project content differs from what `Course Module.pdf` states for that class. Always recorded as an ADR at the time of the deviation, not left implicit. See `docs/adr/0003-week9-class1-curriculum-deviation-emoji-cam.md` for the first instance.

## Local Model
A model whose weights are downloaded once (typically from HuggingFace Hub, on first run) and then run entirely on the student's own machine with no further network calls — distinct from a Cloud API Call, which sends data to a third party on every request. This course's "Sovereign AI" positioning is about avoiding Cloud API Calls for inference, not about avoiding the internet entirely — a one-time model download is consistent with "local," a live API dependency is not. See `docs/adr/0004-week10-class19-embedding-model.md` for where this distinction first had to be made explicit.
