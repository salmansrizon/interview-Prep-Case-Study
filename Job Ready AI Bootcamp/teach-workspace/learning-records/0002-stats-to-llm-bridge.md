# 0002 — The stats → LLM bridge became the point of the course

**Date:** 2026-08-12
**Status:** Active
**Supersedes nothing. Extends:** [0001](0001-workspace-opened-bangla-stats.md)

## Context

Right after lessons 1–5 were delivered, the follow-up was: "also add how it's
implemented & used in LLM models."

That is not a small addition. It changes what the Week 3 material *is for*. The
notebook teaches probability as classical statistics. In an **AI Engineering**
bootcamp, the reason to learn it is that these exact operations run inside the
models the students will ship.

## Decision

Two new lessons, split by inside/outside the model — because they need different
mental models and would blow past working memory as one:

- **0006 — inside the model.** Weight init from a normal distribution, LayerNorm as
  literal z-score, RMSNorm, `/√d_k` in attention, outlier filtering in training data.
- **0007 — outside the model.** Next-token distribution, logits → softmax →
  temperature, top-k/top-p, logprob, perplexity, semantic entropy for hallucination
  detection, RLHF calibration collapse, benchmark confidence intervals.

New reference doc `reference/llm-stats-map.html` — a one-page concept-to-location map
plus NumPy implementations. This is the sheet worth printing before an interview.

## The insight that makes the bridge work

Three facts, each of which visibly changes how the earlier lessons read:

1. **LayerNorm *is* the z-score formula.** `γ(x−μ)/√(σ²+ε) + β`. Not analogous —
   identical, with two learned parameters bolted on. Every token, every layer.
   This single fact retroactively justifies lesson 2 better than any exam example.
2. **Softmax temperature is σ, the spread knob, wearing a different hat.** And the
   classic fix for miscalibration is called *temperature scaling* — the same knob,
   fitted on a validation set instead of chosen by vibe.
3. **The best hallucination detector is the law of large numbers.** Semantic entropy
   = run the prompt ten times, cluster answers by meaning, measure entropy. That is
   lesson 3's frequentist move applied to a language model.

## Consequence for how this material should be taught

The notebook's five topics currently end at "here is a confidence score." They should
end at "here is why your LLM sounds certain and is wrong." The calibration point is
the one with real engineering stakes:

> Base models are reasonably calibrated. RLHF degrades it — the model learns that
> humans dislike hedging. So a model's stated confidence is a style artifact, not a
> measurement. Measure it yourself: logprob, or semantic entropy.

## Accuracy notes

All numeric examples were computed, not estimated: softmax outputs at T = 1.0 / 2.0 /
0.3, and the MMLU-scale CI (±0.56% at n = 14,000, p = 0.87). Every citation URL was
fetched and title-checked. The "prompt/RAG context = prior" line is labelled an
analogy in the lesson, because it is one — RAG does not perform literal Bayesian
updating.

## Open

Mission still provisional (see `MISSION.md`). The LLM follow-up leans toward the
bootcamp being AI-engineering-focused rather than statistics-focused, but does not
settle whether the user is teaching this or learning it.
