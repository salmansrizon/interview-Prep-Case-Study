"""
Centralized configuration for the Neural Network Lab.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FIGURE_DPI = 150
COLOR_PALETTE = ["#6C5CE7", "#00B894", "#0984E3", "#E17055", "#FD79A8", "#FDCB6E"]
