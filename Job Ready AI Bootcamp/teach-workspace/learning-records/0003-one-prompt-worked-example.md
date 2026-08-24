# 0003 — One worked prompt beats eight topic explanations

**Date:** 2026-08-12
**Status:** Active
**Extends:** [0002](0002-stats-to-llm-bridge.md)

## Context

After lessons 1–7 the request was: "explain all this with a prompt example, visualize
if needed." Seven lessons had each explained a concept correctly and separately. What
was missing was a single artefact where every concept fires on the *same* input, in
order, with numbers attached.

## Decision

Lesson 0008 is one prompt traced through eight numbered steps — tokenize, forward pass,
logits, softmax, sample, perplexity, semantic entropy, calibration — with every earlier
lesson tagged at the step where it applies.

Two contrasting prompts, not one, running through the identical pipeline:

- **A:** "বাংলাদেশ স্বাধীন হয় কোন সালে?" — the model knows this.
- **B:** "ঢাকার ৩ নম্বর রোডের চায়ের দোকানের মালিকের নাম কী?" — unknowable, and the
  model will confabulate a name without hesitating.

## Why the pair is the whole trick

A single confident prompt demonstrates the mechanism but teaches nothing about
judgement. The pair makes every metric produce a *contrast* rather than a number:

| signal | A | B |
|---|---|---|
| top-1 probability | 0.99 | 0.26 |
| entropy (max 1.61) | 0.04 | 1.59 |
| top-p=0.9 keeps | 1/5 | 5/5 |
| perplexity | 1.13 | 8.53 |
| semantic entropy | 0.00 | 1.50 |

Every row separates cleanly, from the same pipeline, with no access to ground truth.
That is the engineering lesson: **you can flag a hallucination without knowing the
answer.**

## The knob-turning insight

The interactive lab makes one thing land that prose could not: setting temperature to
0.1 on prompt B does not make the model correct or honest. It makes the wrong answer
*more certain-looking*. Low temperature hides ignorance rather than removing it — and
that makes hallucinations more dangerous, not less.

## Build notes

- Figures are generated, not drawn: `assets/prompt_walkthrough.py`, runnable with
  `uv run --with matplotlib --with numpy`. matplotlib is not installed in the repo
  environment and was deliberately not added to it.
- The logits are hand-authored so a learner can verify the arithmetic by hand; the
  lesson says so in a callout. Everything downstream of the logits is really computed.
  Do not let this slip into implying real model measurements.
- Figure 1 needed its own mid-range logits. Prompt A is too lopsided for temperature to
  visibly change anything and prompt B is too flat — the knob only shows its work in
  between. Worth remembering when building similar visuals.
- `assets/softmax-lab.js` is a reusable component (temperature + top-p sliders, live
  bars, entropy readout). Any future lesson touching sampling should link it rather
  than rebuild it.

## Consequence for teaching

The bootcamp's Week 3 notebook ends at "here is a confidence score." This lesson
suggests the real ending is a routing function:

```python
if top1 >= 0.90 and sem_entropy < 0.5:  auto_accept()
elif top1 >= 0.60:                      human_review()
else:                                   refuse()
```

That is where statistics stops being coursework and becomes a production decision.
