# Module 1 — Lecture Reference Guide
## The Machine Learning Pipeline & Tools Refresher

> **How to use this file**  
> Each section maps 1-to-1 to a notebook cell.  
> For every decision you will find: **What we did · Why · Available alternatives · Common mistakes**

---

## Big Picture — What Are We Building?

We build an **end-to-end ML pipeline** that takes raw customer transaction data
and produces a live web app that predicts whether a new customer is a
**Low / Medium / High** spender.

```
Raw CSV (1 M rows)
      │
      ▼
 Sample 10 000 rows         ← Step 2
      │
      ▼
 Feature Engineering        ← Step 3   (create smarter numeric inputs)
      │
      ▼
 Imputation                 ← Step 4   (fill missing values)
      │
      ▼
 Target + Split             ← Step 5   (label creation, 80/20 split)
      │
      ▼
 Train 3 Models             ← Step 6   (LR / RF / GBM comparison)
      │
      ▼
 Evaluate Best Model        ← Step 7   (confusion matrix, report, confidence)
      │
      ▼
 Save 3 Artefacts           ← Step 8   (.joblib files)
      │
      ▼
 Streamlit Web App          ← Step 9   (live prediction UI)
```

---

## Step 1 — Imports & Global Settings

### What we do
Import every library and centralise all configuration into named constants.

### Why centralise config?
If `SAMPLE_SIZE` is buried inside a cell, a student changing it must hunt through
the notebook. One block at the top → one place to change, experiment, explain.

### Libraries explained

| Library | What it does | Why this one |
|---|---|---|
| `numpy` | N-dimensional arrays, maths | Foundation of all numeric ML in Python |
| `pandas` | Labelled tables (DataFrames) | Readable, column-named data manipulation |
| `matplotlib` | Base plotting engine | Universal — every other plot lib builds on it |
| `seaborn` | Statistical visualisations | Prettier defaults, built-in heatmaps |
| `sklearn` | Machine learning toolkit | Industry standard, consistent API |
| `joblib` | Fast object serialisation | Saves/loads ML objects (models, arrays) efficiently |

### `matplotlib.use('Agg')` — why?
`Agg` is a **non-interactive** backend that renders to files, not a window.
**Without it**, Jupyter may try to open an OS window popup and crash in headless
environments (servers, CI). Always set this before importing `pyplot`.

**Alternatives:**
- `TkAgg` — opens a Tk window (needs a display)
- `Qt5Agg` — Qt window
- `inline` (set via `%matplotlib inline` magic) — renders in notebook directly

### Config constants

```python
RANDOM_STATE = 42    # seed for reproducibility
SAMPLE_SIZE  = 10_000
OUTPUT_DIR   = "outputs"
DATA_PATH    = "Dataset/20260411/customer_spending_1M_2018_2025.csv"
```

**Why `RANDOM_STATE = 42`?**  
Any integer works. `42` is a convention in the ML community (from "The Hitchhiker's
Guide to the Galaxy"). The number itself does not matter — using the **same** number
across all calls (sampling, splitting, model init) guarantees identical results every run.

**`warnings.filterwarnings('ignore')`** — suppresses scikit-learn's convergence
warnings so students see the output they care about, not noise.
Never do this in production code; always investigate warnings there.

---

## Step 2 — Load Data & EDA

### What we do
1. Load the full 1 M-row CSV
2. Take a 10 000-row random sample
3. Inspect dtypes, missing values, numeric stats
4. Plot the distribution of `Amount_spent`

### Why EDA before modelling?
You cannot engineer good features or spot data issues if you haven't **looked** at
the data. EDA prevents "garbage in, garbage out."

### `pd.read_csv` — key parameters you may need

| Parameter | Purpose | Example |
|---|---|---|
| `dtype=` | Force column types on load | `dtype={'Age': float}` |
| `parse_dates=` | Auto-parse date columns | `parse_dates=['Transaction_date']` |
| `usecols=` | Load only specific columns (memory saving) | `usecols=['Age', 'Amount_spent']` |
| `chunksize=` | Read in chunks for huge files | `chunksize=100_000` |
| `na_values=` | Extra strings to treat as NaN | `na_values=['N/A', 'missing']` |

### Why sample instead of training on all 1 M rows?
- 1 M rows takes **minutes** to train; 10 k rows takes **seconds**
- Development speed matters during experimentation
- The statistical patterns are essentially the same at 10 k if the sample is random
- Use full data only for the final production model (Step 8 retrains on the full sample)

### `.describe()` output — what to look for
- `count` much lower than total rows → missing values
- `min` or `max` seems impossible (negative age, 0 spend) → data quality issue
- `std` very large relative to `mean` → heavy outliers exist
- `25%` equals `50%` → many identical values (possibly categorical encoded as number)

### Histogram — why visualise Amount_spent?
- Reveals whether the distribution is **normal**, **skewed**, or **bimodal**
- Informs binning strategy (Step 3): equal-width vs equal-frequency bins
- Shows outliers visually

---

## Step 3 — Feature Engineering

### What is Feature Engineering?
The process of using **domain knowledge** to create new input columns that are
more informative to a model than the raw columns. It is often the single biggest
lever to improve model performance.

### The `fitted_params` pattern — why a dictionary?
```python
fitted_params = {}
fitted_params['q25'] = df['Amount_spent'].quantile(0.25)
```
We must apply the **same thresholds** in the web app that we used during training.
Storing them in a dict and saving it with `joblib.dump` is the simplest approach
that requires **no class definition** — easy to explain, easy to debug.

**Alternative:** a scikit-learn `Pipeline` + custom `Transformer` class — more
elegant, but harder to read for beginners.

### Feature 1 — `amount_spent_segment`

**What:** bin `Amount_spent` into 4 labelled categories using quantile edges.

```python
bins   = [0, q25, q50, q75, float('inf')]
labels = ['Low_Spender', 'Mid_Spender', 'Upper-Mid_Spender', 'High_Spender']
df['amount_spent_segment'] = pd.cut(df['Amount_spent'], bins=bins, labels=labels,
                                     include_lowest=True)
```

**Why `pd.cut` with quantile edges (equal-frequency) instead of equal-width?**

| Approach | How bins are sized | Risk |
|---|---|---|
| Equal-width (`pd.cut` with a number) | Fixed ৳ range per bin | Skewed data → most rows in one bin |
| Equal-frequency (`pd.qcut` or manual quantiles) | Same number of rows per bin | Bins have different ৳ widths |

We use **equal-frequency** (quantile edges) so each bin has roughly the same number
of customers — a balanced encoding.

**`include_lowest=True`** — without this, the row with the exact minimum value
falls outside all bins and becomes `NaN`. Always set it.

**Why not use `pd.qcut` directly?**  
`pd.qcut` computes quantiles internally. We need to store the edges in `fitted_params`
so the app can reuse them — so we compute quantiles manually first.

**Available options for encoding categories:**
- `pd.cut` / `pd.qcut` → ordinal bins (what we use)
- One-hot encoding (`pd.get_dummies`) → binary columns per category
- Label encoding (`LabelEncoder`) → integer per category
- Target encoding (used in Feature 3) → replace with mean of target

### Feature 2 — `spending_per_age`

**What:** `Amount_spent / Age` — spend relative to life stage.

```python
df['spending_per_age'] = df['Amount_spent'] / (df['Age'] + 1e-6)
df['spending_per_age'] = df['spending_per_age'].clip(upper=max_ratio)
```

**Why `+ 1e-6`?**  
Prevents `ZeroDivisionError` if `Age == 0`. `1e-6` is small enough that it
doesn't meaningfully change the ratio for normal ages.

**Why clip?**  
A user could enter `Age = 18` and `Amount_spent = 3000`. If training data had
`Age_min = 15`, the max training ratio is `2999.98 / 15 ≈ 200`. Any inference
value above 200 was never seen in training — clipping keeps predictions reliable.

**Available transformation options:**
- Log transform: `np.log1p(x)` — good for right-skewed features
- Square root: `np.sqrt(x)` — milder compression
- Standardisation: `(x - mean) / std` — zero mean, unit variance
- Min-max scaling: `(x - min) / (max - min)` → range [0, 1]

### Feature 3 — `segment_target_encoded`

**What:** replace the text `Segment` column with that segment's mean `Amount_spent`.

```python
segment_means = df.groupby('Segment')['Amount_spent'].mean().to_dict()
df['segment_target_encoded'] = df['Segment'].map(segment_means).fillna(0.0)
```

**Why target encoding instead of one-hot?**

| Method | Result | Problem |
|---|---|---|
| One-hot encoding | 4 binary columns (Basic/Gold/Platinum/Silver) | High cardinality → many columns |
| Target encoding | 1 numeric column | Compact, captures ordinal relationship |

**Data leakage risk:** computing segment means on the full dataset (including test
rows) leaks information. Here we compute on the sample only, which acts as our
training universe — acceptable for a classroom pipeline.

**Solution in production:** compute means on the training fold only, inside
cross-validation.

**`fillna(0.0)`** — if the app receives a segment not seen during training,
it gets a mean of 0 (neutral) rather than `NaN` which would crash the model.

### Feature 4 — `days_since_start`

**What:** integer count of days from the earliest training date to each transaction.

```python
df['days_since_start'] = (df['Transaction_date'] - min_date).dt.days
```

**Why?**  
Captures **time trend**: do customers in 2024 spend differently from 2018?
An integer is much easier for a model to use than a raw date string.

**Why store `min_date`?**  
The app calculates `(user_date - min_date).days`. If we used "today" instead
of `min_date`, the value would change every day — inconsistent with training.

### Feature 5 — Calendar decomposition

```python
df['transaction_year']      = df['Transaction_date'].dt.year
df['transaction_month']     = df['Transaction_date'].dt.month
df['transaction_dayofweek'] = df['Transaction_date'].dt.dayofweek  # 0=Mon, 6=Sun
df['is_weekend']            = (df['transaction_dayofweek'] >= 5).astype(int)
```

**Why decompose a date into components?**  
A single date cannot be fed to a model. Breaking it into parts lets the model
independently learn:
- Year → long-term growth trend
- Month → seasonal patterns (holiday spending spikes)
- Day of week → weekly rhythm (weekend vs weekday)

**`astype(int)`** converts `True`/`False` booleans to `1`/`0`.
Almost all sklearn models require **numeric** input only.

### Feature 6 — Cyclical month encoding

```python
df['month_sin'] = np.sin(2 * np.pi * df['transaction_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['transaction_month'] / 12)
```

**The problem with raw month numbers:**  
Month 12 (December) and month 1 (January) are adjacent on a calendar,
but numerically `12 - 1 = 11` — they look far apart to a model.

**The fix — unit circle projection:**  
Project each month onto a circle using sine and cosine. Month 6 (June)
maps to angle `π` (180°). Month 12 maps to `2π` (360°) = same as 0° = January.
The model can now "see" that December and January are close.

```
Month  →  Angle          sin     cos
  1       30°            0.50    0.87
  6      180°            0.00   -1.00
 12      360°/0°        -0.00    1.00  ← same as month 1
```

**Why both sin AND cos?**  
Sin alone is ambiguous: `sin(30°) = sin(150°)`. With cos as well,
every month maps to a unique (sin, cos) pair on the circle.

---

## Step 4 — Imputation (Handle Missing Values)

### What is imputation?
Replacing missing (`NaN`) values with a computed substitute so the model
can process every row.

### Why can't we just drop rows with NaN?
- We may lose a significant fraction of data (e.g. `Referral` has 616 NaNs = 6 %)
- At prediction time (web app), the user might not provide all fields —
  we cannot "drop" a user's prediction request

### Why mean imputation?
```python
imputer = SimpleImputer(strategy='mean')
```

| Strategy | Replaces NaN with | Best for |
|---|---|---|
| `'mean'` | column average | Symmetric, continuous data |
| `'median'` | column middle value | Skewed data, outlier-resistant |
| `'most_frequent'` | most common value | Categorical or discrete data |
| `'constant'` | a fixed value you specify | When 0 or a sentinel makes sense |

**More advanced options:**
- `KNNImputer` — fills from k nearest neighbours (slower, more accurate)
- `IterativeImputer` — models each column with other columns (experimental)

### The fit/transform separation — critical concept

```python
imputer = SimpleImputer(strategy='mean')
X_imputed_array = imputer.fit_transform(X_raw)  # on training data
```

In the app:
```python
X_imputed = imputer.transform(X_input)  # NEVER fit again — use stored means
```

**If we called `fit` again in the app**, the means would be recomputed from just
one user row — producing nonsense. Always `fit` once on training data, then
only `transform` forever after.

### Why exclude `Transaction_ID` and `Amount_spent`?

| Column | Reason to exclude |
|---|---|
| `Transaction_ID` | Unique row identifier — no predictive signal; model would memorise IDs |
| `Amount_spent` | This **is** the target we are predicting — using it as input is cheating (data leakage) |
| `amount_spent_segment` | Categorical bins — non-numeric and already captured by other engineered features |

---

## Step 5 — Target Variable & Train/Test Split

### Creating the target label

```python
spend_q33 = df['Amount_spent'].quantile(0.33)
spend_q67 = df['Amount_spent'].quantile(0.67)
y = np.select([..., ...], ['Low', 'Medium'], default='High')
```

**Why 3 classes instead of 2?**  
Binary (High/Not-High) loses nuance. Three classes give actionable segments:
- `Low` → needs promotional nudge
- `Medium` → upgrade campaigns
- `High` → retention/loyalty focus

**Why percentiles instead of fixed ৳ thresholds?**
- Fixed thresholds are domain-specific and need domain expertise
- Percentiles automatically produce balanced class distributions on any dataset
- They generalise to different datasets without manual adjustment

**`np.select` vs `pd.cut` for target:**  
We use `np.select` because we need a 1-D numpy array `y` (not a Series with
category dtype) — required by sklearn's `fit(X, y)`.

### Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

**Parameters explained:**

| Parameter | Value | Why |
|---|---|---|
| `test_size=0.2` | 20% test | Standard; gives enough test rows for reliable metrics |
| `random_state` | 42 | Reproducible — same split every run |
| `stratify=y` | class labels | Each class appears proportionally in both halves |

**Why `stratify`?**  
Without it, random chance could put all "High" rows in the training set,
leaving the test set with mostly "Low". Stratification prevents this.

**Common split ratios:**
- 80/20 — standard for medium datasets (what we use)
- 70/30 — when test set needs to be larger for confidence
- 60/20/20 — train / validation / test (for hyperparameter tuning)
- K-fold cross-validation — no fixed split; more robust, slower

---

## Step 6 — Train Three Models & Compare

### Why train multiple models?
No single algorithm wins on every dataset. Different models make different
assumptions about data structure. Comparing empirically is the **only** reliable
way to select the best one.

### Model A — Logistic Regression

```python
lr_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
```

**How it works:**  
Finds a **linear decision boundary** (a hyperplane in N dimensions).
Each feature gets a weight; prediction = sigmoid of the weighted sum.

**Strengths:**
- Fastest to train
- Highly interpretable (weights show feature importance)
- Performs well when classes are linearly separable

**Weaknesses:**
- Struggles with non-linear relationships (e.g. spending peaks for both
  very young and very old customers — a U-shape is non-linear)
- Sensitive to feature scale; benefits from standardisation

**`max_iter=2000`:** the solver (`lbfgs` by default) needs iterations to converge.
With 1 000 (the default) it may warn of non-convergence. 2 000 usually fixes it.

**Other solvers:**
- `lbfgs` — default, good for multiclass
- `saga` — fast for large datasets, supports all penalties
- `liblinear` — only binary classification

### Model B — Random Forest

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
```

**How it works:**  
Trains `n_estimators` independent decision trees, each on a **random bootstrap**
sample of rows and a random subset of features. Final prediction = majority vote.

This is called **bagging** (Bootstrap AGGregation).

**Strengths:**
- Handles non-linear patterns naturally
- Robust to outliers and noisy features
- Built-in feature importance
- Little tuning needed for good results

**Weaknesses:**
- Slower than LR; large models use lots of RAM
- Less interpretable than a single decision tree
- Can overfit on very small datasets

**Key hyperparameters to tune:**
- `n_estimators` — more trees = better (but slower). 100–500 typical.
- `max_depth` — limits tree depth to prevent overfitting
- `min_samples_split` — minimum rows to split a node
- `max_features` — fraction of features considered at each split (`'sqrt'` default)

**`n_jobs=-1`:** use all available CPU cores in parallel → faster training.

### Model C — Gradient Boosting

```python
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
```

**How it works:**  
Builds trees **sequentially**. Each new tree is trained to predict the **residual
errors** of all previous trees. The "gradient" refers to minimising a loss function
via gradient descent in function space.

**Strengths:**
- Usually achieves the highest accuracy of the three
- Handles mixed feature types well
- Works well out of the box

**Weaknesses:**
- Slowest to train (sequential, cannot parallelise tree building)
- More hyperparameters to tune
- Can overfit if `n_estimators` is too high without regularisation

**Key hyperparameters:**
- `n_estimators` — number of trees (boosting rounds)
- `learning_rate` — shrinks each tree's contribution (lower = better generalisation, needs more trees)
- `max_depth` — keeps trees shallow (default=3); deeper = more complex, more overfit risk
- `subsample` — fraction of rows used per tree (< 1 introduces randomness)

**Modern alternatives to sklearn's GBM:**
- `XGBoost` — faster, widely used in competitions
- `LightGBM` — fastest for large datasets, uses leaf-wise growth
- `CatBoost` — best for categorical features with no preprocessing

### Evaluation metrics

```python
lr_acc = accuracy_score(y_test, lr_preds)
lr_auc = roc_auc_score(y_test, lr_proba, multi_class='ovr', average='macro')
```

**Accuracy:**  
`correct predictions / total predictions`  
Simple but misleading when classes are imbalanced.
If 90 % of rows are "High", a model that always predicts "High" gets 90 % accuracy
without learning anything.

**ROC-AUC (Receiver Operating Characteristic — Area Under Curve):**  
Measures how well the model **ranks** predictions. AUC = 1.0 is perfect;
AUC = 0.5 is random guessing.

`multi_class='ovr'` (One-vs-Rest): for each class, AUC is computed treating
it as positive and all others as negative, then averaged.

`average='macro'`: treats all classes equally regardless of size.
`'weighted'` would weight by class size — use when classes are imbalanced.

**When to use which metric:**

| Metric | Use when |
|---|---|
| Accuracy | Classes are balanced |
| ROC-AUC | Ranking quality matters; imbalanced classes |
| F1-score | You care equally about precision and recall |
| Precision | False positives are costly (e.g. spam filter) |
| Recall | False negatives are costly (e.g. disease detection) |

---

## Step 7 — Evaluate the Best Model in Depth

### A — Classification Report

```python
report_text = classification_report(y_test, y_pred, target_names=classes)
```

**What each column means:**

| Column | Formula | What it tells you |
|---|---|---|
| Precision | TP / (TP + FP) | Of all predicted as X, how many were actually X? |
| Recall | TP / (TP + FN) | Of all actual X, how many did we correctly predict? |
| F1-score | 2 × P × R / (P + R) | Harmonic mean — penalises extreme imbalance between P and R |
| Support | — | Count of actual rows of that class in the test set |

**`macro avg` vs `weighted avg`:**
- `macro avg` — simple average across classes (treats all classes equally)
- `weighted avg` — weighted by support (class size)

A model can have high `weighted avg` F1 but poor `macro avg` — it's doing great on
the big class and ignoring the small one.

### B — Confusion Matrix

```
              Predicted
              High  Low  Medium
Actual High  [788    0       0]
       Low   [  0  579      18]
       Medium[  0    0     615]
```

**How to read it:**
- **Diagonal** = correct predictions
- **Row = true class**, **Column = predicted class**
- Off-diagonal cell `[Low, Medium]` = 18 means 18 actual-Low rows were
  predicted as Medium (false negatives for Low, false positives for Medium)

**Why seaborn heatmap instead of printing a table?**  
Colour intensity makes errors visually obvious at a glance — much faster to
explain to an audience than reading numbers in a grid.

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ...)
```

- `annot=True` — print the number in every cell
- `fmt='d'` — format as integer (not `1.23e+02`)
- `cmap='Blues'` — light = low count, dark = high count

### C — Confidence Histogram

```python
max_confidence = y_prob.max(axis=1)  # highest class probability per row
```

`predict_proba` returns a `(n_rows, 3)` array. Each row sums to 1.0.
`max(axis=1)` picks the **largest** of the three per row — the model's
stated confidence in its winning prediction.

**What to look for:**
- Spike near 1.0 → model is very confident (usually good if correct)
- Spread across 0.4–0.9 → model is uncertain (may need more data or features)
- High confidence AND low accuracy → model is **overconfident** (calibration problem)

**Calibration:** well-calibrated model where confidence = 0.8 should be correct
80 % of the time. sklearn's `CalibratedClassifierCV` can fix poorly calibrated models.

---

## Step 8 — Save All Artefacts

### Why save three separate files instead of one?

| File | What it holds | Why separate |
|---|---|---|
| `final_best_model.joblib` | Trained classifier (all decision trees/weights) | Large; swapped out when you retrain |
| `fitted_params.joblib` | Dict of quantile edges, segment means, min_date, column list | Small; changes only if feature engineering changes |
| `imputer.joblib` | Column means from training | Changes if features change |

Keeping them separate means you can update just the model (e.g. retrain on more
data) without regenerating the feature params — and vice versa.

### Why `copy.deepcopy` before retraining?

```python
final_model = copy.deepcopy(best_model)
final_model.fit(X, y)   # ALL rows
```

`best_model` was trained on `X_train` (80 % of sample). We want a fresh model
trained on `X` (100 %). `deepcopy` gives a new unfitted object with the **same
hyperparameters** without disturbing the original.

**Why not just call `best_model.fit(X, y)` directly?**  
It would overwrite the weights we used for evaluation in Step 7, making the
step-by-step story inconsistent.

### Why retrain on 100 % of the sample?
During Step 6 we trained on 80 % to get an unbiased test evaluation.
Now that we know Gradient Boosting is the best algorithm, we "unlock" the held-out
20 % for training too. More data → more patterns learned → better generalisation.

### `joblib` vs `pickle`

| | `joblib` | `pickle` |
|---|---|---|
| Speed | Fast (uses memory-mapped arrays for numpy) | Slower for large arrays |
| File size | Smaller for numpy-heavy objects | Same |
| Compatibility | Python/sklearn only | Any Python object |

Use `joblib` for sklearn models and numpy arrays. Use `pickle` only for
pure-Python objects.

---

## Step 9 — Streamlit Web App

### How the app pipeline mirrors the notebook

```
Notebook Step 3          App
─────────────────        ────────────────────────────────────────
Compute q25/q50/q75  →  Load from fitted_params['q25', 'q50', 'q75']
pd.cut(Amount_spent) →  pd.cut(user_input, same bins)
segment_means.map()  →  segment_means.get(segment, 0.0)
days_since_start     →  (user_date - fitted_params['min_date']).days
month_sin/cos        →  np.sin/cos with same formula
```

**The golden rule:** every transformation in the app must use values from
`fitted_params`, never recomputed from the user's single row.

### Streamlit embedding in Jupyter

```python
threading.Thread(target=start_server, daemon=True).start()
time.sleep(4)
display(HTML('<iframe src="http://127.0.0.1:8501" ...>'))
```

- `daemon=True` — thread dies automatically when the kernel restarts
- `time.sleep(4)` — wait for the server to finish booting before the iframe renders
- The iframe shows the Streamlit UI inside the notebook output area

**Why `subprocess.Popen` instead of just importing streamlit?**  
Streamlit must run as its own process with its own event loop. Importing and
calling it from inside a Jupyter kernel would conflict with the kernel's event loop.

---

## Common Mistakes & How to Avoid Them

| Mistake | Symptom | Fix |
|---|---|---|
| Training imputer/scaler in the app | Predictions inconsistent with training | `fit` once in notebook, `transform` only in app |
| Including `Amount_spent` as a feature | Model scores 99 %+ (leakage) | Always drop the target before training |
| Including `Transaction_ID` | Model overfits to row IDs | Drop ID columns before training |
| Not stratifying the split | Classes imbalanced in test set | Add `stratify=y` |
| Comparing models on training accuracy | Picks the most overfit model | Always evaluate on held-out test set |
| Raw month number without cyclical encoding | December and January appear far apart | Use `month_sin` and `month_cos` |
| Recomputing quantiles/means in the app | App uses different thresholds than training | Save `fitted_params` and load in app |
| Forgetting to save the imputer | App crashes with `FileNotFoundError` | `joblib.dump(imputer, imputer_path)` |

---

## Glossary for Students

| Term | One-line definition |
|---|---|
| **Feature** | An input column the model uses to make predictions |
| **Target** | The column we are trying to predict (here: Low/Medium/High) |
| **NaN** | Not a Number — a missing value placeholder |
| **Imputation** | Filling missing values with a computed substitute |
| **Quantile** | Value below which X% of data falls (e.g. 25th quantile = Q1) |
| **Target encoding** | Replace a category label with the mean of the target for that category |
| **Overfitting** | Model memorises training data, fails on new data |
| **Data leakage** | Test-set information influences training → inflated accuracy |
| **Artefact** | A saved file produced by training (model, scaler, params) |
| **ROC-AUC** | Ranking quality metric; 1.0 = perfect, 0.5 = random |
| **Precision** | Of all predicted positives, what fraction are actually positive |
| **Recall** | Of all actual positives, what fraction did we correctly predict |
| **F1** | Harmonic mean of precision and recall |
| **Confusion matrix** | Grid showing actual vs predicted class counts |
| **Bagging** | Train many models on random subsets, average results (Random Forest) |
| **Boosting** | Train models sequentially, each fixing previous errors (GBM) |
| **Stratify** | Ensure class proportions are preserved in a split |
| **Serialisation** | Converting a Python object to bytes for disk storage (`joblib.dump`) |
| **Inference** | Using a trained model to make predictions on new data |

---

## Quick Reference — Key Numbers from Our Run

| Metric | Logistic Regression | Random Forest | Gradient Boosting |
|---|---|---|---|
| Accuracy | 81.1 % | 96.2 % | **98.1 %** |
| ROC-AUC | 0.920 | 0.997 | **0.999** |

**High / Low classes: near-perfect.** The model almost never confuses a High
spender with a Low spender — makes intuitive sense (very different amounts).

**Medium class: hardest.** Medium is adjacent to both Low and High in value,
so boundary cases are genuinely ambiguous.

**Average prediction confidence: 93.6 %** — the model is very sure of itself,
and the evaluation confirms that confidence is warranted.

---

*This file is the single source of truth for all teaching notes for Module 1.*  
*Update it whenever the notebook changes.*