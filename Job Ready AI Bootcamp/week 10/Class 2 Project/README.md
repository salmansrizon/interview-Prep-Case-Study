# 💬 Local Customer Feedback Analyzer

A production-grade sentiment analysis application built with Python, HuggingFace Transformers, and Streamlit. Classifies customer reviews as Positive, Negative, or Neutral with confidence scores, flags low-confidence predictions for human review, and aggregates results into a dashboard — all running locally, no third-party API calls.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red.svg)
![transformers](https://img.shields.io/badge/transformers-4.40+-yellow.svg)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔤 **Single Review Analysis** | Paste one review, get an instant label + confidence score |
| 📋 **Multi-line Batch Paste** | Paste several reviews at once (one per line) for batch classification |
| 📁 **CSV Upload** | Upload a CSV of reviews; the review-text column is auto-detected |
| 🎯 **3-Class Sentiment** | Genuine Positive / Negative / Neutral output — no hacky thresholding on a binary model |
| ⚠️ **Confidence-Threshold Flagging** | Low-confidence predictions are flagged `needs_review` for a human to double-check |
| 📊 **Aggregate Dashboard** | Label-distribution bar/pie charts and a flagged-for-review count, computed from the session's analyzed batch |
| ⚡ **Batched Inference** | Reviews are passed to the model as a list, not one-at-a-time — see `src/sentiment/classifier.py` |
| 🧪 **Model-Free Test Suite** | `tests/test_pipeline.py` covers CSV ingestion, aggregation, and flagging logic without downloading the model |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. On the first real classification call, HuggingFace downloads the `cardiffnlp/twitter-roberta-base-sentiment-latest` model (~500MB) and caches it locally — subsequent runs load from cache.

### 3. Run the Tests

```bash
pytest tests/test_pipeline.py
```

No internet access or model download is required to run the test suite — see `tests/test_pipeline.py`'s HIGHLIGHTS comment for why.

## 📁 Project Structure

```
Class 2 Project/
├── app.py                        # Streamlit entry point (Overview / Analyze / Dashboard tabs)
├── config.py                     # Centralized config: model name, thresholds, paths, UI colors
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/
│   └── sample/
│       └── sample_reviews.csv    # Example CSV for the upload flow
├── notebooks/
│   └── 01_exploration.ipynb      # Hands-on exploration of the classifier + flagging logic
├── src/
│   ├── ingestion/
│   │   └── csv_loader.py         # CSV -> clean list of review strings (column auto-detect)
│   ├── sentiment/
│   │   └── classifier.py         # Lazy-loaded HuggingFace pipeline + confidence-flagging
│   ├── analytics/
│   │   └── aggregator.py         # SentimentResult list -> DataFrame + dashboard summary dict
│   └── utils/
│       └── logger.py             # Shared structured logging setup
└── tests/
    └── test_pipeline.py          # Ingestion / aggregation / flagging tests (no model needed)
```

## 🔧 Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│  Review Text │────▶│   Tokenize   │────▶│   Classify   │────▶│  Confidence    │────▶│   Aggregate    │
│ (1 / batch / │     │ (inside the  │     │ (RoBERTa     │     │  Check         │     │   Dashboard    │
│  CSV upload) │     │  pipeline)   │     │  3-class)    │     │ (< threshold?  │     │ (counts, %,    │
│              │     │              │     │              │     │  flag review)  │     │  flagged, chart│
└──────────────┘     └──────────────┘     └──────────────┘     └────────────────┘     └───────────────┘
      app.py         transformers'         src/sentiment/       src/sentiment/          src/analytics/
   (3 input modes)    pipeline() call      classifier.py         classifier.py            aggregator.py
```

Each stage lives in exactly one module: `app.py` never imports `transformers` or parses CSVs directly — it only calls into `src/ingestion`, `src/sentiment`, and `src/analytics`. This keeps the model/network dependency isolated to one file (`src/sentiment/classifier.py`) and lets everything else be unit-tested without it.

## ⚙️ Configuration

All knobs live in `config.py` (`AppConfig`, consumed by both `app.py` and `src/`):

| Setting | Default | Description |
|---------|---------|--------------|
| `model.model_name` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | HuggingFace checkpoint used for sentiment classification |
| `model.task` | `sentiment-analysis` | HuggingFace `pipeline()` task alias (works for this 3-class model too) |
| `analysis.confidence_threshold` | `0.6` | Predictions scoring below this are flagged `needs_review` |
| `analysis.batch_size` | `16` | Max reviews sent to the pipeline per batch chunk (bounds memory on large CSVs) |
| `analysis.max_review_chars` | `2000` | Guard against pathologically long pasted text |
| `ui.label_colors` | Positive=green, Negative=red, Neutral=gray | Color coding used in the results table |
| `ui.needs_review_color` | orange (`#f39c12`) | Highlight color for flagged rows |
| `sample_csv_path` | `data/sample/sample_reviews.csv` | Example file offered in the CSV upload flow |

## 🧠 Model Choice

### Model Reference Table

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`cardiffnlp/twitter-roberta-base-sentiment-latest`** (used here) | General 3-class sentiment (Positive/Negative/Neutral) | Genuine Neutral class — no arbitrary confidence-band hack needed | Trained on tweet-style text; some domain shift vs. formal product reviews |
| `distilbert-base-uncased-finetuned-sst-2-english` | Fast binary (Positive/Negative only) sentiment | Small, fast, extremely popular default | No Neutral class — forcing one in requires an unreliable score threshold |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 1–5 star rating prediction | Multilingual, rating-style output | Outputs a star scale, not Positive/Negative/Neutral — needs a mapping layer |
| OpenAI GPT (zero-shot prompting) | Flexible classification with custom categories | No fine-tuning needed, just change the prompt | Not local, per-call cost, data leaves your machine (third-party API) |

We picked `cardiffnlp/twitter-roberta-base-sentiment-latest` because the project spec explicitly wants three real classes, and the most popular default (DistilBERT SST-2) simply isn't trained to produce a Neutral class — bolting one on via an arbitrary confidence-band threshold is a hack layered on top of a model never trained for that distinction. "Which model is actually trained for this task" wins over "which model is the most popular default."

### Domain-Shift Caveat

`cardiffnlp/twitter-roberta-base-sentiment-latest` was fine-tuned on tweets — short, informal, emoji/hashtag-heavy text — not on product reviews, which are longer and more formal. This mismatch between training data and deployment data is called **domain shift**. The two domains share enough general sentiment vocabulary ("broke", "disappointed", "fantastic") that the model still performs well in practice, but don't expect tweet-benchmark-level accuracy on review text — and treat every prediction's confidence score, not just its label, as part of the answer. The standard production mitigation (out of scope for this project, but worth knowing) is collecting a sample of your own domain's labeled reviews and running a second, lighter fine-tuning pass on top of this checkpoint to close the gap.

## 🛠️ Development

### Adding a New Input Mode to the Analyze Tab

1. Add the new input widget in `app.py`'s Analyze tab.
2. Convert whatever the widget produces into a plain `List[str]` of review strings.
3. Pass that list to `src.sentiment.classifier.classify_in_batches()` — never call `transformers` directly from `app.py`.
4. Store the resulting `SentimentResult` list in `st.session_state` so the Dashboard tab can pick it up.

### Swapping the Sentiment Model

Change `ModelConfig.model_name` in `config.py` and update the Model Reference Table above — see `src/sentiment/classifier.py`'s `get_classifier()` docstring for the full reasoning behind the current choice and what changes if the label spelling differs (`_LABEL_MAP`).

## 📜 License

MIT License
