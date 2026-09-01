# 🎯 Customer Segmentation Engine

## সহজ ভাষায় Project Overview

**🎯 Customer Segmentation Engine** project-এ lecture-এর theory-কে working software বা executable notebook-এ convert করা হয়েছে। লক্ষ্য শুধু final output দেখা নয়; input থেকে preprocessing, core logic/model, evaluation এবং output—পুরো pipeline বোঝা।

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

An offline Streamlit learning project that finds useful structure in data that has no answer labels. It groups similar customers, compresses many columns into a smaller set, and discovers products that are often purchased together.

This is the Class 2 project for Week 6: **Unsupervised Learning**.

## What Problem Does It Solve?

Many datasets do not tell us the “correct answer.” A store knows what customers bought, but it may not know which customer types exist or which products belong together. Unsupervised learning searches for those hidden patterns.

This app answers three business questions:

- **K-Means:** What natural customer groups exist?
- **PCA:** Can we summarize many related columns without losing most of their information?
- **Market Basket Analysis:** Which products are genuinely associated and could support bundles or recommendations?

## Learning Objectives

| Topic | What you learn | Problem solved |
|---|---|---|
| Feature scaling | Put measurements such as dollars and counts on comparable scales | Prevents large-number columns from dominating distance |
| K-Means | Repeatedly assign points to their closest center | Finds compact customer segments without labels |
| Elbow and silhouette | Compare possible values of `K` | Helps avoid choosing the number of groups blindly |
| PCA | Rotate data toward its most informative directions | Reduces dimensions, noise, and visualization difficulty |
| Apriori | Prune product combinations that cannot be frequent | Makes basket-rule discovery practical |
| Support, confidence, lift | Measure frequency, reliability, and added value | Separates meaningful product links from popular-item coincidences |

## Algorithms in Plain English

### K-Means Clustering

Imagine placing `K` meeting points in a city. Every customer walks to the nearest point; then each point moves to the middle of its assigned customers. Assignment and movement repeat until the points stop changing. Each final meeting point represents one cluster.

K-Means is useful when groups are roughly compact and numeric distance is meaningful. It is better than manual segments because it uses all selected measurements together, but it still needs a human to interpret and name each group.

### Principal Component Analysis (PCA)

Think of photographing a long object. A photograph from the right angle preserves its length; a poor angle makes it look short. PCA finds the viewing angles that preserve the most variation in the data, then keeps only the most informative views.

PCA is useful for visualization, faster downstream models, and removing repeated information from correlated columns. The trade-off is that a component such as `PC1` is harder to explain than an original column.

### Market Basket Analysis

Market Basket Analysis counts combinations such as `{Milk, Bread} → {Butter}`. **Support** says how common the combination is, **confidence** says how often butter appears when milk and bread appear, and **lift** compares that result with butter’s normal popularity. Lift greater than 1 indicates a positive association.

## How Data Flows

```text
Customer CSV or synthetic data
        → remove IDs/text labels → fill missing numeric values → scale
        → K-Means clusters or PCA components → metrics and charts

Transaction baskets → one-hot product matrix → frequent itemsets
                    → association rules → filter by confidence and lift
```

## Project Structure

```text
Class 2 Project/
├── app.py                          # Streamlit interface for all three modules
├── config.py                       # Paths and algorithm defaults
├── requirements.txt               # Runtime and test dependencies
├── README.md                       # This guide
├── src/
│   ├── data/
│   │   ├── loader.py              # Customer and transaction generators
│   │   └── preprocessor.py        # Numeric selection, missing values, scaling
│   ├── features/
│   │   └── engineering.py         # RFM and engagement feature helpers
│   ├── models/
│   │   ├── kmeans_engine.py
│   │   ├── pca_engine.py
│   │   └── market_basket.py
│   ├── visualization/
│   │   └── charts.py
│   └── utils/
│       └── logger.py
├── tests/
│   └── test_pipeline.py
└── notebooks/
    └── 01_eda.ipynb
```

The app creates `data/raw`, `data/processed`, and `data/models` when required.

## Requirements

- Python 3.10 or newer
- Internet access only while installing packages
- No API key or network connection while running the app

## Quick Start

From the repository root on Windows PowerShell:

```powershell
cd "week 6\Class 2 Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS or Linux, activate with `source .venv/bin/activate`. Open `http://localhost:8501` if the browser does not open automatically.

## How to Use Each Module

### K-Means

1. Generate the offline customer data or upload a CSV.
2. Choose the number of clusters, initialization method, and iteration limit.
3. Run K-Means.
4. Check cluster sizes and profiles, then review inertia, silhouette, and Calinski-Harabasz scores.
5. Compare the elbow and silhouette charts before deciding whether `K` is sensible.

### PCA

1. Load customer data with at least two rows and two numeric feature columns.
2. Choose a component count or enable automatic selection for a 95% variance target.
3. Inspect explained variance, cumulative variance, feature loadings, and the 2D projection.

The app limits component choices to what the current dataset can support.

### Market Basket Analysis

1. Choose the number of transactions and products.
2. Adjust minimum support, confidence, lift, and maximum itemset length.
3. Generate and analyze the baskets.
4. Inspect the strongest rules and the product-association network.
5. Download the rule table as CSV.

If no rules appear, lower the thresholds one at a time. Start with support because an itemset must be frequent before Apriori can create rules from it.

## Upload Data Format

K-Means and PCA accept a CSV. Identifier columns containing `id` and columns containing `ground_truth` are excluded; other non-numeric columns are ignored. Missing numeric values are filled with each column’s mean.

Example:

```csv
customer_id,purchase_frequency,avg_order_value,total_spend,satisfaction_score
CUST_001,5,42.50,212.50,4.2
CUST_002,1,350.00,350.00,3.8
```

Use meaningful numeric behavior columns. A numeric ID that does not contain `id` can accidentally influence distance and should be removed before upload.

## Understanding the Results

| Metric | Better direction | Meaning |
|---|---|---|
| Inertia/WCSS | Lower for the same `K` | Points are closer to their cluster centers |
| Silhouette | Closer to 1 | Clusters are compact and separated |
| Calinski-Harabasz | Higher | Between-cluster separation is strong relative to within-cluster spread |
| Explained variance | Higher | Retained PCA components preserve more information |
| Lift | Greater than 1 | Products co-occur more than chance/popularity predicts |

Inertia always decreases as `K` increases, so it should not be used alone. Also check silhouette, cluster size, stability, and whether the segments make business sense.

## Important Settings

| Setting | Default | Purpose |
|---|---:|---|
| `RANDOM_STATE` | `42` | Reproducible generated data and models |
| `KMEANS_DEFAULT_CLUSTERS` | `5` | Initial customer group count |
| `PCA_VARIANCE_THRESHOLD` | `0.95` | Auto-selection information target |
| `MBA_MIN_SUPPORT` | `0.05` | Minimum transaction frequency |
| `MBA_MIN_CONFIDENCE` | `0.3` | Minimum rule reliability |
| `MBA_MIN_LIFT` | `1.5` | Minimum strength above chance |

## Run the Tests

```powershell
cd "week 6\Class 2 Project"
python -m pytest -q
python -m compileall app.py config.py src tests
```

The suite covers customer generation, preprocessing/scaling, K-Means, PCA, Market Basket Analysis, and enforcement of the requested product-catalog size.

## Troubleshooting

- **`ModuleNotFoundError`:** activate the virtual environment and install `requirements.txt` again.
- **PowerShell blocks activation:** call `.venv\Scripts\python` directly.
- **No numeric features:** upload a CSV containing at least two useful numeric columns.
- **No association rules:** reduce minimum support/confidence/lift or generate more transactions.
- **Slow basket analysis:** raise support, reduce maximum itemset length, or reduce products/transactions.
- **One huge or tiny cluster:** rescale/check outliers and try another `K`.

## Limitations

- Synthetic segments and associations are for education, not business evidence.
- K-Means prefers roughly spherical, similarly sized clusters and is sensitive to outliers.
- PCA is linear and may hide nonlinear patterns; components are less interpretable than original features.
- Association rules show co-occurrence, not causation.
- A production system also needs data validation, drift monitoring, privacy controls, and stakeholder review.

## Tech Stack

Streamlit, scikit-learn, pandas, NumPy, mlxtend, Plotly, Matplotlib, Seaborn, and joblib.
