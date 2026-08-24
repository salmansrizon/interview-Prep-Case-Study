# 0001 — Workspace opened: probability & confidence, taught in Bangla

**Date:** 2026-08-12
**Status:** Active
**Source material:** `week 3/Class 2 Project/week3_class7_project.ipynb`

## Context

First session in this teaching workspace. The request: explain the Week 3 / Class 2
project notebook "like a 15 year old", in simple Bangla.

The notebook covers five topics in one pass: normal distribution, Z-scores, frequentist
probability, Bayesian probability, and ML confidence scores. That is far more than fits
in one lesson's working memory, so it was split into five lessons of one idea each.

## Decisions

1. **Bangla prose, English technical nouns.** Translating "posterior" into Bangla would
   make the notebook and the job interview harder to navigate later. Explanation in
   Bangla, vocabulary in English.
2. **Five lessons, not one.** Each is completable in roughly ten minutes and delivers a
   single win. They chain: shape → scale → counting uncertainty → updating belief →
   applying both to a model.
3. **Every lesson ends with retrieval, not summary.** A recall box that will not reveal
   the model answer until something has been written. Summaries build fluency; retrieval
   builds storage strength.
4. **The medical-test paradox leads with counting, not with the formula.** Imagining
   10,000 people produces the 50% answer far more reliably than plugging into Bayes.

## Non-obvious insight worth preserving

The pedagogical spine that emerged, and that later lessons should keep referring back to:

> Normal distribution gives data a **shape**. Z-score gives that shape a **ruler**.
> Frequentist **counts** to measure uncertainty. Bayesian **adds prior knowledge** to it.
> A confidence score is those last two applied to a model's own output.

Each topic answers a limitation of the one before. Taught as five disconnected topics —
which is how the notebook currently reads — that chain is invisible.

## Open

The mission is provisional (see `MISSION.md`): teaching Bangla-speaking students, or
learning this personally? Difficulty and format for lesson 6 depend on the answer.
