# 0005. Feedback Analyzer uses a native 3-class sentiment model, not a thresholded 2-class one

## Status
Accepted

## Context
Project 08 (Local Customer Feedback Analyzer, Class 20) is scoped by `Course Module.pdf` to label reviews Positive/Negative/**Neutral** — three classes. The most common off-the-shelf HuggingFace sentiment pipeline default (`distilbert-base-uncased-finetuned-sst-2-english`) is binary (Positive/Negative only); a "Neutral" label would have to be faked by thresholding on prediction confidence, which is a common real-world shortcut but methodologically arbitrary (the threshold is a guess, doesn't generalize across domains, and the model was never trained to recognize neutral sentiment).

## Decision
Use a genuinely 3-class pretrained model — `cardiffnlp/twitter-roberta-base-sentiment-latest` — which natively outputs Positive/Negative/Neutral probabilities, rather than deriving Neutral from a binary model's confidence score.

## Consequences
- Larger download (~500MB) than a DistilBERT binary model, but still a one-time local download consistent with `docs/adr/0004`'s framing.
- Model was trained on tweet-style text; review text (the project's actual domain) is different enough that the lecture/README should note this as a real-world domain-shift caveat, not silently assume perfect accuracy.
- Sets the pattern for this course: when a curriculum achievement specifies N classes, find a model that natively supports N classes rather than hacking a coarser model into producing the right number of labels.
