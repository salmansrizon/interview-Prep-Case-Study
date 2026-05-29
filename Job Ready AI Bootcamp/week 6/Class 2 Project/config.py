"""
Centralized configuration for the Customer Segmentation Engine.
Modify constants here to change behavior across the entire application.
"""

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

# ── Random State ──────────────────────────────────────
RANDOM_STATE = 42

# ── K-Means Defaults ──────────────────────────────────
KMEANS_DEFAULT_CLUSTERS = 5
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10

# ── PCA Defaults ──────────────────────────────────────
PCA_DEFAULT_COMPONENTS = 2
PCA_VARIANCE_THRESHOLD = 0.95  # For automatic component selection

# ── Market Basket Analysis Defaults ───────────────────
MBA_MIN_SUPPORT = 0.05
MBA_MIN_CONFIDENCE = 0.3
MBA_MIN_LIFT = 1.5
MBA_MAX_LEN = 3  # Maximum itemset length

# ── Visualization ─────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 150
COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
