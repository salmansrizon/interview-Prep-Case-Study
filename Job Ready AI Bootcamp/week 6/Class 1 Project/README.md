# 🛡️ Spam & Intent Classifier

## সহজ ভাষায় Project Overview

**🛡️ Spam & Intent Classifier** project-এ lecture-এর theory-কে working software বা executable notebook-এ convert করা হয়েছে। লক্ষ্য শুধু final output দেখা নয়; input থেকে preprocessing, core logic/model, evaluation এবং output—পুরো pipeline বোঝা।

### কোন Problem Solve করে?

Manual বা disconnected workflow-কে repeatable code pipeline-এ আনে। এর ফলে একই process নতুন data-তে আবার চালানো, result compare করা, error trace করা এবং future feature add করা সহজ হয়।

### কীভাবে কাজ করে?

Input/Data → Validation ও Preprocessing → Core Algorithm/Model → Evaluation → UI, Report বা Saved Output। নিচের detailed section-গুলোতে project-specific command, feature এবং architecture দেওয়া আছে।

### কেন এই Approach ভালো?

- **Repeatable:** একই input দিলে একই workflow follow করে।
- **Testable:** প্রতিটি stage আলাদাভাবে verify করা যায়।
- **Explainable:** কোন step কী কাজ করছে তা code এবং output দিয়ে দেখা যায়।
- **Portfolio-ready:** শুধু notebook result নয়, setup, structure এবং usage-সহ complete project হিসেবে দেখানো যায়।

> **Run করার নিয়ম:** আগে virtual environment তৈরি করে dependency install করুন। তারপর README-এর Quick Start follow করুন, sample input দিয়ে smoke test করুন এবং expected metric/output-এর সাথে result compare করুন।

---

An offline Streamlit learning project that turns text into numbers with TF-IDF, trains three classification algorithms, compares their results, and predicts whether a new message is spam or what the writer intends to do.

This is the Class 1 project for Week 6: **Advanced Classification Models**.

## What Problem Does It Solve?

Computers do not naturally understand sentences. A support inbox may receive thousands of messages such as “I cannot log in,” “Where is my order?”, or “You won a prize.” Reading and routing each one manually is slow and inconsistent.

This app demonstrates the standard solution:

```text
Raw message → clean the text → convert words to TF-IDF numbers
            → train a classifier → evaluate it → classify new messages
```

It supports two tasks:

- **Spam detection:** classify a message as `spam` or `ham` (legitimate).
- **Intent classification:** classify a message as `purchase`, `support`, `inquiry`, `complaint`, `feedback`, or `other`.

## Learning Objectives

| Topic | What you learn | Why it matters |
|---|---|---|
| Text preprocessing | Lowercasing, URL/email removal, stop-word removal, and word normalization | Reduces noise before training |
| TF-IDF | Turns important words and two-word phrases into numeric features | ML models require numbers, not raw sentences |
| Naive Bayes | Makes a decision from class and word probabilities | Very fast and strong for text baselines |
| Linear SVM | Finds the widest boundary between classes | Often performs well on sparse, high-dimensional text |
| KNN | Lets nearby training examples vote | Gives an intuitive comparison with “lazy learning” |
| Evaluation | Accuracy, precision, recall, F1, and confusion matrix | Shows what kinds of mistakes the model makes |

## Algorithms in Plain English

### Naive Bayes

Naive Bayes asks, “How often do these words appear in spam compared with normal messages?” It combines those clues and chooses the more likely class. It is usually the best first baseline because training is fast and it works well even with many word features.

### Support Vector Machine (SVM)

Imagine spam and normal messages as dots on a map. SVM draws a dividing road and tries to keep the road as far as possible from the closest dots on both sides. A wider gap usually means a safer decision on unseen messages.

### K-Nearest Neighbors (KNN)

KNN stores the training examples. For a new message, it finds the `K` most similar examples and lets them vote. It is easy to understand, but prediction becomes slower as the dataset grows.

## Project Structure

```text
Class 1 Project/
├── app.py                         # Streamlit user interface
├── config.py                      # Paths, model defaults, and TF-IDF settings
├── requirements.txt               # Runtime and test dependencies
├── README.md                       # This guide
├── src/
│   ├── data/
│   │   ├── loader.py              # CSV loading and synthetic data generation
│   │   └── preprocessor.py        # Offline-safe text cleaning
│   ├── features/
│   │   └── vectorizer.py          # TF-IDF feature extraction
│   ├── models/
│   │   ├── base.py                # Shared classifier interface/evaluation
│   │   ├── naive_bayes.py
│   │   ├── svm.py
│   │   ├── knn.py
│   │   └── registry.py            # Model factory
│   ├── training/
│   │   └── trainer.py             # End-to-end training pipeline
│   └── utils/
│       └── logger.py
├── tests/
│   └── test_pipeline.py           # Preprocessing tests
└── notebooks/
    └── 01_eda.ipynb               # Exploratory notebook
```

The app creates `data/raw`, `data/processed`, and `data/models` when needed. Generated CSV files and trained `.pkl` artifacts are stored there.

## Requirements

- Python 3.10 or newer
- Internet access only while installing Python packages
- No API key, cloud account, or internet connection while using the app

The preprocessor uses built-in offline fallbacks when optional NLTK datasets are not installed. You do not need to run `nltk.download(...)`.

## Quick Start

From the repository root on Windows PowerShell:

```powershell
cd "week 6\Class 1 Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS or Linux, activate with `source .venv/bin/activate` instead. Open `http://localhost:8501` if the browser does not open automatically.

## How to Use the App

1. In the sidebar, choose **Spam Detection** or **Intent Classification**.
2. Choose Naive Bayes, SVM, or KNN and adjust its hyperparameters.
3. In **Data**, generate an offline dataset or upload a CSV.
4. In **Train & Evaluate**, train the selected model and inspect its metrics.
5. In **Predict**, enter a new message and review the predicted label and confidence.
6. Train other algorithms and use **Compare Models** to compare their scores.

### Upload Format

The CSV must contain these exact columns:

| Column | Meaning | Example |
|---|---|---|
| `text` | Raw message | `Where is my order?` |
| `label` | Correct category | `support` |

Example:

```csv
text,label
"Congratulations, claim your free prize",spam
"Can we meet at 3 PM?",ham
```

Each class needs enough rows for the stratified train/test split. For a reliable demo, use at least 50 examples per class.

## Important Settings

Edit `config.py` to change shared defaults:

| Setting | Default | Purpose |
|---|---:|---|
| `TEST_SIZE` | `0.2` | Uses 20% of the data for evaluation |
| `MAX_FEATURES` | `5000` | Maximum TF-IDF vocabulary size |
| `NGRAM_RANGE` | `(1, 2)` | Uses single words and two-word phrases |
| `NB_ALPHA` | `1.0` | Prevents zero probabilities in Naive Bayes |
| `SVM_C` | `1.0` | Balances a wide margin against training mistakes |
| `KNN_N_NEIGHBORS` | `5` | Number of neighbors that vote |

## Understanding the Metrics

- **Accuracy:** percentage of all predictions that were correct.
- **Precision:** when the model says “spam,” how often it is right.
- **Recall:** how much of the real spam the model catches.
- **F1:** one balanced score combining precision and recall.
- **Confusion matrix:** counts each correct and incorrect class pairing.

For spam filtering, accuracy alone can hide a bad model. If only 1% of messages are spam, predicting “not spam” every time gives 99% accuracy but catches no spam. Recall and F1 expose that problem.

## Run the Tests

```powershell
cd "week 6\Class 1 Project"
python -m pytest -q
python -m compileall app.py config.py src tests
```

The tests verify lowercase conversion, URL and punctuation removal, word normalization, and operation without downloaded NLTK data.

## Troubleshooting

- **`ModuleNotFoundError`:** activate the project virtual environment and rerun `python -m pip install -r requirements.txt`.
- **PowerShell blocks activation:** run `.venv\Scripts\python -m streamlit run app.py` and `.venv\Scripts\python -m pytest -q` without activating.
- **CSV error:** confirm the file has non-empty `text` and `label` columns.
- **Vocabulary is empty:** provide more varied examples; TF-IDF ignores terms that appear fewer than twice.
- **KNN neighbor error:** increase the training data or reduce `K`.

## Limitations

- Synthetic data is useful for learning, not a production-quality spam benchmark.
- SVM confidence values are normalized decision scores, not calibrated probabilities.
- Saved model files should only be loaded from trusted sources because pickle/joblib files can execute code.
- Real deployments need monitoring for changing vocabulary, class imbalance, abuse, and fairness.

## Tech Stack

Streamlit, scikit-learn, pandas, NumPy, NLTK, joblib, Matplotlib, and Seaborn.
