"""
Centralized configuration for the Spam & Intent Classifier.
Modify constants here to change behavior across the entire application.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# Auto-create directories
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model Defaults ────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Naive Bayes
NB_ALPHA = 1.0  # Laplace smoothing

# SVM
SVM_C = 1.0
SVM_KERNEL = "linear"  # 'linear' is fast and great for text

# KNN
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"  # 'uniform' or 'distance'

# ── Text Processing ───────────────────────────────────
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)  # Unigrams + Bigrams
MIN_DF = 2
MAX_DF = 0.95

# ── Labels ───────────────────────────────────────────
SPAM_LABELS = ["ham", "spam"]
INTENT_LABELS = ["purchase", "support", "inquiry", "complaint", "feedback", "other"]
