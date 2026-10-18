# 0001. Lecture and Project authoring template

## Status
Accepted

## Context
Weeks 1–7 of the bootcamp organically converged on a consistent structure for both Lecture and Project material, but it was never written down. Weeks 8–9 already show drift from it (missing lecture files, notebooks sitting in Project folders, a mislabeled folder). Without a written template, each new week (10–20) risks re-deriving or subtly diverging from the pattern, and reviewers have no baseline to check against.

## Decision
Codify the pattern observed in weeks 1–7 as the standard for all future weeks:

**Lecture** (`Class N Lecture/`):
- `context.md`: Topic → Why It Is Related → How It Works (tables/ASCII diagrams as needed) → Detailed Brief/Deep-dive → Analogy (industry-relevant, often a comparison table) → runnable code walkthrough → Brain Teasers/exercises. Prose is Bangla-humanized; technical terminology stays in English, not duplicated in both languages side by side.
- Companion `.ipynb` when the topic is code-heavy enough to warrant interactive, cell-by-cell practice (established in Week 2 for NumPy/Pandas); otherwise `context.md` alone is sufficient.

**Project** (`Class N Project/`), for classes that map to one of the 16 Portfolio Projects:
- `README.md` (badges, features, quick start, project structure tree, architecture ASCII diagram, config table)
- `app.py` (Streamlit UI)
- `config.py`, `requirements.txt`
- `notebooks/01_eda.ipynb`
- `src/{data,features,models,training,utils}/` with `__init__.py` per package
- `tests/test_pipeline.py`

Every Class Session's Lecture and Project must be self-contained per `docs/adr/0002-self-contained-class-artifacts.md`.

## Consequences
- New weeks have a checklist to build against and to review against.
- Deviating from this template (e.g., skipping the notebook, restructuring `src/`) should be a deliberate, called-out choice, not silent drift.
- Retrofitting weeks 8–9 to this template is tracked separately (see the Week 8/9 gap-resolution work).
