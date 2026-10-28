# 0009. Week 16 news sentiment uses FinBERT, not the Week 10 model or the LLM

## Status
Accepted

## Context
Portfolio Project 15 (Offline Market Intelligence Team) requires "semantic sentiment analysis on financial news", and its Risk Score uses the sentiment as a numeric input. Three options were real: reuse Week 10's `cardiffnlp/twitter-roberta-base-sentiment-latest` (students already know it), ask `llama3.1:8b` to label sentiment zero-shot (no extra model), or use a finance-domain classifier.

## Decision
Use `ProsusAI/finbert`: a natively 3-class (positive/negative/neutral) model fine-tuned on financial text, run locally through the HuggingFace `pipeline`. This follows ADR 0005's rule: use a model that natively has the required classes.

## Considered Options
- **Twitter RoBERTa (Week 10):** tweet-domain; finance phrasing like "beats estimates" or "guidance cut" is a domain shift.
- **LLM zero-shot:** slow, non-deterministic, no calibrated probabilities to feed into a deterministic Risk Score, and hard to unit-test.

## Consequences
- One more one-time download (~440 MB), consistent with the "Local Model" definition (ADR 0004).
- Tests mock the classifier; the real model loads lazily on first use.
- The lecture and the loading code include the model comparison table (FinBERT vs Twitter RoBERTa vs DistilRoBERTa-financial vs LLM zero-shot).
