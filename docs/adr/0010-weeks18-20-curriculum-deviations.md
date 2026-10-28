# 0010. Curriculum deviations for weeks 18-20

## Status
Accepted

## Context
`Course Module.pdf` lists, for the last three weeks: Week 18 "Deployment to cloud server" with a "Single-Click Deployment"; Week 19 Capstone (project management, topic selection, planning); Week 20 Class 39 (showcase day) and Class 40 (GitHub portfolio, resume, interview strategy). The course owner chose a narrower scope for this round.

## Decision
1. **Week 18:** "Single-Click Deployment" means one local command (`docker compose up`) that starts the whole Week 17 Healthcare Assistant stack (UI + API + Chroma + Ollama), with an optional GPU override file. Cloud deployment is taught in the Lecture as theory (what changes on a VM, HTTPS, secrets, GPU VM options); there is no deploy script, and no cloud account is assumed.
2. **Week 19 is not built.** No Lecture, no Project, no templates.
3. **Week 20:** Class 39 (presentation day) gets no material. Class 40 gets one Lecture plus a kit: a GitHub portfolio README template, an AI Engineer resume template that maps all 16 Portfolio Projects to bullet points, and a LinkedIn checklist. No interview question bank.

## Consequences
- Students who need capstone planning material won't find it in this repo; Week 19 can be added later without changing anything else.
- Week 18's cloud section can't be followed hands-on from the repo alone.
