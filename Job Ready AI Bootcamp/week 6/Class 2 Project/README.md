# 🎯 Customer Segmentation Engine

A production-grade, 100% offline unsupervised learning system built with
Streamlit and scikit-learn for **Unsupervised Learning Module**.

## Algorithms Covered
1. **K-Means Clustering** — Partition customers into distinct segments
2. **Principal Component Analysis (PCA)** — Reduce dimensions for visualization and compression
3. **Market Basket Analysis** — Discover product associations using Apriori algorithm

## Quick Start

```bash
# 1. Navigate to project folder
cd customer-segmentation-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

## Usage
1. **K-Means Tab**: Generate synthetic customers, adjust K, run clustering, view elbow method & silhouette analysis.
2. **PCA Tab**: Reduce dimensions, view explained variance, inspect feature loadings, project data to 2D.
3. **Market Basket Tab**: Generate transactions, mine association rules, visualize rule landscape & product network.

## Project Highlights
- **Modular Architecture**: Separate modules for data, features, models, and visualization.
- **3 Interactive Modules**: K-Means, PCA, and Market Basket Analysis in one app.
- **Synthetic Data Generator**: Creates realistic customer behavior with 5 hidden segments.
- **Evaluation Metrics**: Silhouette score, Calinski-Harabasz index, inertia (WCSS).
- **Association Rule Mining**: Support, confidence, lift, and conviction metrics.
- **Plotly Visualizations**: Interactive 3D scatter, network graphs, and animated charts.
- **Persistent Artifacts**: Models saved via joblib for reuse.

## File Structure
```
customer-segmentation-engine/
├── app.py                          # Main Streamlit app
├── config.py                       # Centralized configuration
├── requirements.txt                # Dependencies
├── src/
│   ├── data/
│   │   ├── loader.py               # Synthetic data generation
│   │   └── preprocessor.py         # Scaling & feature prep
│   ├── features/
│   │   └── engineering.py          # RFM features & selection
│   ├── models/
│   │   ├── kmeans_engine.py        # K-Means clustering
│   │   ├── pca_engine.py           # PCA dimensionality reduction
│   │   └── market_basket.py        # Apriori association rules
│   ├── visualization/
│   │   └── charts.py               # Static matplotlib charts
│   └── utils/
│       └── logger.py               # Logging utilities
├── tests/
│   └── test_pipeline.py            # Unit tests
└── notebooks/
    └── 01_eda.ipynb                # Exploratory analysis
```
