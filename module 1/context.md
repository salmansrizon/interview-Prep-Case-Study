# Module 1: The Machine Learning Pipeline — Student Study Plan
### *Predicting Customer Spending Categories from Transaction Data*

> **Who is this for?** Students with light Python skills who are new to machine learning. No prior ML or AI experience needed — every concept is explained from scratch with analogies, diagrams, and code.

---

## 📋 Table of Contents

1. [What Are We Building?](#1-what-are-we-building)
2. [The Dataset — Understanding Our Raw Material](#2-the-dataset)
3. [The Big Picture — ML Pipeline Overview](#3-the-big-picture)
4. [Step 1 — Exploratory Data Analysis (EDA)](#4-step-1-eda)
5. [Step 2 — Feature Engineering](#5-step-2-feature-engineering)
6. [Step 3 — Imputation (Handling Missing Data)](#6-step-3-imputation)
7. [Step 4 — Model Training & Selection](#7-step-4-model-training--selection)
8. [Step 5 — Evaluation — How Good Is Our Model?](#8-step-5-evaluation)
9. [Step 6 — Saving the Model (joblib)](#9-step-6-saving-the-model)
10. [Step 7 — Deploying with Streamlit](#10-step-7-deploying-with-streamlit)
11. [Common Mistakes & How We Fixed Them](#11-common-mistakes--how-we-fixed-them)
12. [Glossary](#12-glossary)

---

## 1. What Are We Building?

### The Business Problem

Imagine you run an online store. You have **1 million customer transactions** recorded from 2018 to 2025. Each row tells you who bought something, when, how much they spent, and a bit about who they are.

Your goal: **When a new customer arrives, predict whether they will be a Low, Medium, or High spender — before they check out.**

Why does this matter?
- **Low spender?** → Show them discount banners.
- **Medium spender?** → Offer a loyalty card.
- **High spender?** → Assign a personal shopper / premium experience.

This is called a **classification problem** — we are placing a customer into one of several categories (classes).

### What We'll Solve

| Problem | Our Solution |
|---|---|
| Raw data has dates, text, missing values | Feature Engineering + Imputation |
| Too many algorithms to choose from | Model comparison with metrics |
| Need a usable product, not just code | Streamlit web app |

### How We'll Solve It — The Short Version

```
Raw CSV Data
    ↓
Exploratory Data Analysis (understand the data)
    ↓
Feature Engineering (create smarter columns)
    ↓
Imputation (fill in missing values)
    ↓
Model Training (teach the algorithm)
    ↓
Model Evaluation (check how good it is)
    ↓
Save Model to Disk
    ↓
Deploy as a Web App (Streamlit)
```

---

## 2. The Dataset

### What Does Our Data Look Like?

The dataset is `customer_spending_1M_2018_2025.csv` — roughly 89 MB, 1 million rows, 11 columns.

**Sample rows:**

```
Transaction_ID | Transaction_date        | Gender | Age | Marital_status | State_names | Segment  | Employees_status | Payment_method | Referral | Amount_spent
1000           | 2018-01-01T00:04:00     | Female | 39  | Single         | Oklahoma    | Platinum | Unemployment     | Card           | 0        | 1557.5
1001           | 2018-01-01T00:06:00     | Male   | 34  | Married        | Hawaii      | Basic    | workers          | PayPal         | 1        | 153.55
```

### Column-by-Column Explanation

| Column | Type | What It Means |
|---|---|---|
| `Transaction_ID` | Number | A unique ID for each transaction. Like a receipt number. |
| `Transaction_date` | DateTime | When the purchase happened. |
| `Gender` | Text | Male / Female |
| `Age` | Number | Customer's age |
| `Marital_status` | Text | Single / Married |
| `State_names` | Text | Which US state the customer is from |
| `Segment` | Text | Customer tier: Basic, Silver, Gold, Platinum |
| `Employees_status` | Text | Employment type: Employed, Self-employed, etc. |
| `Payment_method` | Text | Card / Cash / PayPal |
| `Referral` | 0 or 1 | Was the customer referred by a friend? (1 = Yes) |
| `Amount_spent` | Number | **This is our target.** How much they spent. |

### What Is Our "Target"?

`Amount_spent` is a continuous number (like 153.55 or 2558.00). But our goal is to classify customers into spending categories (Low / Medium / High), so we **convert** this number into a category label using binning — explained in Feature Engineering.

---

## 3. The Big Picture — ML Pipeline Overview

Think of the pipeline like preparing a meal:

```
🧺 Raw Ingredients          →  Raw CSV data (messy, mixed, incomplete)
🔪 Chopping & Prep          →  Feature Engineering (transform raw data into useful form)
🧂 Season & Fill Gaps       →  Imputation (fill missing values)
🍳 Cook with Right Recipe   →  Model Training (choose & train algorithm)
👅 Taste Test               →  Evaluation (check accuracy, confusion matrix)
📦 Package for Delivery     →  Save model with joblib
🍽️  Serve to Customer       →  Streamlit App
```

### Why Not Just Feed Raw Data to a Model?

Most ML algorithms only understand **numbers**. They cannot process:
- Text like `"Male"` or `"Platinum"`
- Dates like `"2018-01-01T00:04:00"`
- Missing values (blank cells)

Feature Engineering and Imputation fix all of these problems.

---

## 4. Step 1 — Exploratory Data Analysis (EDA)

### What Is EDA?

EDA means **getting to know your data** before doing anything with it. Think of it as reading the instructions before assembling furniture.

### What We Did in This Project

```python
df_sample = pd.read_csv(CONFIG['data_path']).sample(n=5000, random_state=42)
print(df_sample.info())          # Column types and non-null counts
print(df_sample.describe())      # Statistics: mean, min, max, etc.
print(df_sample.isnull().sum())  # Count missing values per column
```

### What We Found

- **Shape:** 1,000,000 rows × 11 columns
- **Missing values:** Yes — `Age`, `Referral`, and `spending_per_age` had NaN values
- **Date column:** Stored as a string, not a proper date — needs conversion
- **No duplicate rows** in the sample

### Why Sample (5,000 rows) and Not All 1M rows?

**Analogy:** A chef doesn't eat the entire pot to taste the food — they taste a spoonful. Sampling 5,000 rows is fast and gives a reliable picture of the full data without loading 89 MB into memory every time.

### Why Use `random_state=42`?

`random_state` is like a "shuffle seed" — it ensures you get the **same random sample every time** you run the code. This makes your work reproducible. The number 42 is conventional (a programming culture reference) but any fixed number works the same way.

---

## 5. Step 2 — Feature Engineering

### What Is Feature Engineering?

> **Simple definition:** Taking your raw columns and creating new, smarter columns that help the model learn better.

**Analogy:** Imagine you have a student's date of birth. That raw date is not very useful to predict exam performance. But if you calculate **"age at time of exam"**, that's a useful feature. Feature engineering is that transformation.

### The `FeatureEngineer` Class — Why a Class?

We wrapped all transformations in a Python class because:
1. It can **remember parameters** learned from training data (e.g., what the average spending was)
2. We can **save and reload** it later (with joblib)
3. It applies the **exact same transformation** to new data at prediction time

This is called the **fit → transform** pattern:
- `fit()` = learn statistics from training data
- `transform()` = apply those learned statistics to any data

```
Training time:   fit(training_data)   → learns quantile values, date ranges, etc.
Prediction time: transform(new_data)  → applies the same learned rules
```

### Feature 1 — Amount Spent Binning (Creating the Target Label)

**Problem:** `Amount_spent` is a number (e.g., 153.55). We need a label: Low / Medium / High.

**Solution:** Divide customers into 4 groups based on how much they spent relative to everyone else.

```python
bins = [0, Q1, Q2, Q3, infinity]
labels = ['Low_Spender', 'Mid_Spender', 'Upper-Mid_Spender', 'High_Spender']
```

**What Are Quantiles?**

Quantiles split data into equal-sized groups. The Q1, Q2, Q3 (25th, 50th, 75th percentile) are the "dividing lines":

```
All customers sorted by Amount_spent:

|---25%---|---25%---|---25%---|---25%---|
0        Q1        Q2        Q3       Max
  Low    Mid    Upper-Mid   High
```

**Why this, not just "under $500 = Low"?**

Because fixed thresholds don't adapt. If inflation doubles prices next year, $500 may no longer be "Low". Quantile-based bins always split the data into equal groups, regardless of the actual dollar values.

**Alternatives:**
- Fixed thresholds — simple but brittle, requires domain expertise
- K-Means clustering — data-driven but harder to explain to business stakeholders
- Equal-width bins — splits by dollar range, not by customer count

### Feature 2 — Spending Per Age Ratio

```python
df['spending_per_age'] = df['Amount_spent'] / (df['Age'] + 1e-6)
```

**Why create this?**

A 25-year-old spending $1,500 is different from a 65-year-old spending $1,500. The ratio captures this relative behavior.

**Why `+ 1e-6`?**

This tiny number (`0.000001`) is added to `Age` to prevent division by zero if Age is ever 0. This is called a **numerical stability guard**.

**Why clip the ratio?**

```python
df['spending_per_age'] = df['spending_per_age'].clip(upper=max_ratio)
```

Extreme outliers (e.g., a 18-year-old spending $50,000) create wild ratio values that can confuse the model. Clipping limits the maximum value to something reasonable.

**Alternatives:**
- Log transform — reduces outlier impact but harder to interpret
- Just use raw age — simpler, but loses the relationship with spending

### Feature 3 — Target Encoding for Segment

**Problem:** `Segment` is a text column (Basic, Silver, Gold, Platinum). Models need numbers.

**What is Target Encoding?**

Replace each category with the **average Amount_spent** for that category in the training data.

```
Segment   → Average Amount_spent
Basic     → $1,400
Silver    → $1,420
Gold      → $1,380
Platinum  → $1,442
```

So a row with `Segment = Platinum` gets replaced with `1442.0` as a number.

**Why target encoding, not just numbering them 1,2,3,4?**

Assigning arbitrary numbers (Basic=1, Silver=2...) implies that Silver is "twice as good" as Basic — which is not true. Target encoding assigns numbers that have real meaning: the actual average spending of that group.

**Alternatives:**

| Method | How it works | When to use |
|---|---|---|
| One-Hot Encoding | Creates a new column for each category (0 or 1) | Few categories (under ~10) |
| Label Encoding | Assigns 1,2,3... to categories | Ordered categories (e.g., Low/Med/High) |
| Target Encoding | Replaces category with average target value | Many categories, strong relationship to target |
| Frequency Encoding | Replaces category with how often it appears | When frequency matters more than target value |

**Risk of Target Encoding:** It can "leak" information from the target into the features if done carelessly. That is why we `fit()` it on the training sample only, not on all the data.

### Feature 4 — Time-Based Features

**Problem:** A date like `2018-01-01` is one value. But a model can learn much more from it.

**What we created:**

```python
df['days_since_start']      # How many days since the first transaction in the dataset
df['transaction_year']      # 2018, 2019, 2020...
df['transaction_month']     # 1, 2, 3... 12
df['transaction_dayofweek'] # 0=Monday, 6=Sunday
df['is_weekend']            # 1 if Saturday/Sunday, else 0
df['month_sin']             # Cyclical encoding of month
df['month_cos']             # Cyclical encoding of month
```

**What is Cyclical Encoding? (month_sin / month_cos)**

**The problem with raw month numbers:** Month 12 (December) and Month 1 (January) are actually close in time (just 1 month apart), but as numbers they are far apart (12 vs 1). A model would think they are very different.

**The solution — use a circle:**

Imagine the 12 months arranged on a clock face. We encode each month as a point on that circle using sine and cosine:

```
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

This way, December and January are "close" to each other in the encoded space, just as they are on a calendar.

```
          Month 6 (June)
               |
Month 3 -------+------- Month 9
               |
          Month 12/1 (Dec/Jan are close!)
```

**Alternatives:**
- Raw month number — loses the cyclical nature
- One-hot encoding of month — works but creates 12 columns

---

## 6. Step 3 — Imputation (Handling Missing Data)

### What Is Imputation?

> **Imputation** = filling in missing values with an educated guess.

### Why Do We Have Missing Values?

In real-world data, missing values are normal. A customer might not have entered their age. A field might have a recording error. In our data:

```
Age          → 163 missing values (out of 5,000 sample)
Referral     → 616 missing values
spending_per_age → 1,116 missing values  (derived column, missing because Age was missing)
```

### Why Can't We Just Delete Rows With Missing Values?

**Analogy:** Imagine a hospital study where sicker patients are more likely to have incomplete records. If you delete those rows, your study becomes biased — you only study healthy patients. Same problem here: deleting missing rows can silently bias your model.

Also, with 1 million rows, any missing data pattern is worth keeping. Deleting rows = losing signal.

### What We Used — SimpleImputer (Mean Strategy)

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_numeric_train)       # Learn the mean of each column from training data
X_imputed = imputer.transform(X)   # Fill NaNs with those learned means
```

**Why mean imputation?**

The mean is the "best single guess" for a missing number when you have no other information. It keeps the average of the column unchanged.

**Alternatives:**

| Strategy | How it works | Best for |
|---|---|---|
| Mean | Replace with average | Normally distributed data, no strong outliers |
| Median | Replace with middle value | Data with outliers (median is more robust) |
| Mode | Replace with most common value | Categorical (text) columns |
| KNN Imputer | Find similar rows and use their values | Complex datasets where similar customers should have similar values |
| Model-based | Train a separate model to predict the missing value | Highest accuracy, most complex |

### The Critical Fit/Transform Rule

```
✅ CORRECT:
  imputer.fit(training_data)         # Learn means from TRAINING data only
  imputer.transform(training_data)   # Apply to training
  imputer.transform(test_data)       # Apply same means to test data

❌ WRONG:
  imputer.fit(all_data)              # Leaks test data statistics into training!
```

**Why does this matter?** If you fit the imputer on all data including the test set, the model "sees" future data during training — giving artificially inflated results. This is called **data leakage**.

### The `Transaction_ID` Bug — A Real Lesson

In this project, we discovered a bug: `Transaction_ID` (just a row number) was accidentally included when the imputer was fitted. This meant every time we tried to predict, the imputer and model both complained that `Transaction_ID` was missing.

**The fix:** Include a dummy `Transaction_ID = 0` in the prediction input so the pipeline stays consistent. This is a real-world lesson: **whatever columns were present during training must be present during prediction — in the same order.**

---

## 7. Step 4 — Model Training & Selection

### What Is a Classification Model?

A model is a **mathematical function** that learns the relationship between your input columns (features) and the output label (Low/Medium/High spender).

**Analogy:** Imagine teaching a child to recognize dogs vs cats. You show them thousands of examples: "this is a dog, this is a cat." After enough examples, the child learns the pattern and can classify new animals. A classification model does the same with numbers.

### Train/Test Split — Why We Separate Data

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

We split our data into:
- **Training set (80%)** — the model learns from this
- **Test set (20%)** — we evaluate the model on this (it never saw this data)

**Analogy:** Train/Test split is like studying for an exam with practice questions (train), then taking the real exam with different questions (test). If you could memorize the real exam answers in advance, the exam would be meaningless.

```
Full Data (10,000 rows)
│
├── Training Set (8,000 rows) → Model learns patterns here
│
└── Test Set (2,000 rows)    → Model is evaluated here (unseen)
```

### The Models We Compared

We trained four different algorithms and compared them:

#### 1. Logistic Regression

**What it is:** Despite the name, it's a classification algorithm. It finds a straight line (or plane) that best separates the classes.

**Analogy:** Imagine trying to separate apples from oranges on a table by drawing a straight line between them. Logistic Regression draws that line.

**Best for:** Simple problems where classes are linearly separable. Fast to train.

**Weakness:** Struggles when the boundary between classes is curved or complex.

#### 2. Random Forest

**What it is:** Builds many decision trees (hence "forest") and combines their votes.

**What is a Decision Tree?**

A series of yes/no questions:
```
Is Age > 40?
├── YES → Is Segment = Platinum?
│         ├── YES → High Spender
│         └── NO  → Mid Spender
└── NO  → Is Referral = 1?
          ├── YES → Mid Spender
          └── NO  → Low Spender
```

**Why "Random"?** Each tree is trained on a random subset of data and random subset of columns. This variety means no single tree dominates, and errors average out.

**Best for:** Tabular (table-style) data. Handles mixed types well. Hard to overfit.

**Weakness:** Slower to train with many trees. Less interpretable than a single tree.

#### 3. Gradient Boosting (Our Winner ✅)

**What it is:** Builds trees one after another, where each new tree specifically fixes the mistakes of the previous trees.

**Analogy:** Imagine a team of students grading essays. Student 1 grades them all. Student 2 focuses only on the essays Student 1 got wrong. Student 3 focuses on what Student 2 still got wrong. Each student specializes in fixing errors. The final grade is a combination of all their assessments.

**Why it won:** Gradient Boosting tends to achieve the highest accuracy on structured tabular data because it iteratively reduces errors.

**Weakness:** Slower to train. More sensitive to hyperparameters. Can overfit if not tuned carefully.

#### 4. Support Vector Machine (SVM)

**What it is:** Finds the widest possible "gap" between classes in a high-dimensional space.

**Analogy:** Imagine two groups of students sitting in a room. SVM finds the widest aisle you can draw between the two groups.

**Best for:** High-dimensional data (many features), text classification.

**Weakness:** Very slow on large datasets. Harder to tune. Doesn't naturally output probabilities.

### How We Chose the Best Model

```python
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")
```

**Winner: Gradient Boosting** — highest accuracy on the test set.

---

## 8. Step 5 — Evaluation

### Accuracy — The Simplest Metric

```
Accuracy = (Number of correct predictions) / (Total predictions)
```

If we correctly classify 850 out of 1,000 customers: Accuracy = 85%.

**Why accuracy alone is not enough:**

Imagine 95% of customers are "Low Spenders". A lazy model that always predicts "Low Spender" gets 95% accuracy — but it's completely useless for identifying High Spenders. That is why we also look at the **Confusion Matrix** and **Classification Report**.

### The Confusion Matrix

A confusion matrix shows how many of each class were correctly and incorrectly classified.

```
                    PREDICTED
                Low  Mid  High
ACTUAL   Low  [ 280   12    8 ]
         Mid  [  15  190   20 ]
         High [   5   10  160 ]
```

Reading it:
- **Row = Actual label** (what the true answer was)
- **Column = Predicted label** (what the model said)
- **Diagonal = Correct predictions** (where actual = predicted)
- **Off-diagonal = Mistakes**

### Classification Report — Precision, Recall, F1

```
              precision  recall  f1-score
Low Spender      0.93    0.94      0.93
Mid Spender      0.88    0.86      0.87
High Spender      0.84    0.85      0.84
```

**Precision** = Of all customers predicted as High Spender, how many actually were?
- High precision = few false alarms

**Recall** = Of all actual High Spenders, how many did we catch?
- High recall = few missed High Spenders

**F1 Score** = Harmonic mean of Precision and Recall. One number that balances both.

### ROC-AUC Score

**AUC** = Area Under the ROC Curve. Ranges from 0.5 (random guessing) to 1.0 (perfect).

Tells you: "How good is the model at ranking customers by their likelihood of being a High Spender?"

An AUC above 0.85 is generally considered good.

---

## 9. Step 6 — Saving the Model

### What Is `joblib`?

After training, we need to save the model to disk so the web app can use it later — without retraining every time.

`joblib` is a Python library that **serializes** (converts to a file) Python objects.

```python
import joblib

joblib.dump(model, 'outputs/final_best_model.joblib')    # Save model
joblib.dump(engineer, 'outputs/feature_engineer.joblib') # Save feature engineer
joblib.dump(imputer, 'outputs/imputer.joblib')           # Save imputer

# Later, to reload:
model = joblib.load('outputs/final_best_model.joblib')
```

**Analogy:** Think of `joblib` as vacuum-sealing cooked food. You cook it once (train), seal it (save), and reheat it (load) whenever you need it — without starting from scratch.

### Why Save Three Files?

| File | Why it must be saved |
|---|---|
| `final_best_model.joblib` | The trained model weights and parameters |
| `feature_engineer.joblib` | The quantile boundaries, date ranges, segment means learned at training time |
| `imputer.joblib` | The mean values learned from training data to fill NaNs |

All three must use parameters **from training time** when making predictions. If you recalculate them on new data, the transformation will be different, and the model will receive unexpected input.

**Alternative:** `pickle` (Python's built-in serializer). `joblib` is preferred for ML models because it handles large NumPy arrays more efficiently.

---

## 10. Step 7 — Deploying with Streamlit

### What Is Streamlit?

Streamlit is a Python library that turns a regular Python script into an interactive web application — with no HTML, CSS, or JavaScript knowledge needed.

```python
import streamlit as st

st.title("My App")
age = st.number_input("Enter your age")
if st.button("Submit"):
    st.write(f"You are {age} years old")
```

Run with: `streamlit run app.py` → Opens in browser at `http://localhost:8501`

### How Our App Works (Step by Step)

```
User fills in the form
        ↓
app.py creates a DataFrame from the input
        ↓
engineer.transform(input_data) → creates all the engineered features
        ↓
imputer.transform(aligned_features) → fills any NaN values
        ↓
model.predict(features) → returns "Low", "Mid", or "High"
        ↓
model.predict_proba(features) → returns probability for each class
        ↓
Display result on screen
```

### `@st.cache_resource` — Why Is This Important?

```python
@st.cache_resource
def load_assets():
    model = joblib.load('outputs/final_best_model.joblib')
    engineer = joblib.load('outputs/feature_engineer.joblib')
    imputer = joblib.load('outputs/imputer.joblib')
    return model, engineer, imputer
```

Streamlit re-runs the **entire script** every time a user interacts with the app (clicks a button, changes a dropdown). Without caching, it would reload the model from disk on every click — which is very slow.

`@st.cache_resource` tells Streamlit: "Run this function only once, then remember the result."

### The Pipeline Consistency Rule

> **Golden Rule:** Whatever columns and transformations you used at training time MUST be replicated exactly at prediction time.

This is why we redefined the `FeatureEngineer` class inside `app.py`. When `joblib` loads the saved engineer object, Python needs to know the class definition — otherwise it cannot reconstruct it.

### Why `Transaction_ID = 0` in the App?

During training, `Transaction_ID` was accidentally included in the feature set when the imputer and model were fitted. This means both the imputer and model permanently expect this column.

In the web app, a user doesn't enter a `Transaction_ID` — so we just supply a dummy value of `0`. It doesn't affect the prediction meaningfully, but it satisfies the pipeline's expectation.

**The lesson:** Always carefully control which columns enter your model during training.

---

## 11. Common Mistakes & How We Fixed Them

### Mistake 1 — f-string Double Braces

```python
# ❌ Wrong — shows literal text {e}
st.error(f"Error: {{e}}")

# ✅ Correct — interpolates the variable
st.error(f"Error: {e}")
```

**Why it happened:** When writing Streamlit code inside a Python string (using triple quotes), real curly braces needed escaping. When that code was moved to its own file, the escaping was no longer needed — but wasn't removed.

### Mistake 2 — Transaction_ID Missing from Prediction Input

**Symptom:** `ValueError: Feature names seen at fit time, yet now missing: Transaction_ID`

**Why:** The imputer (and later the model) were fitted with `Transaction_ID` present as a numeric column.

**Fix:** Add `'Transaction_ID': [0]` to the input DataFrame so it flows through the entire pipeline unchanged.

### Mistake 3 — Redefining the Class in app.py

**Why it's needed:** `joblib` saves Python objects by reference to their class. When loading in a different script, Python must be able to find that class definition. If `FeatureEngineer` is not defined in `app.py`, loading the saved engineer will fail with a cryptic error.

**Rule:** Always redefine any custom classes in every script that loads joblib-saved objects of that class.

---

## 12. Glossary

| Term | Plain English Definition |
|---|---|
| **Feature** | A column used as input to the model (e.g., Age, Segment) |
| **Target** | The column we want to predict (e.g., Low/Mid/High Spender) |
| **Classification** | Predicting which category something belongs to |
| **Feature Engineering** | Transforming raw columns into more useful ones |
| **Imputation** | Filling in missing values |
| **Train/Test Split** | Dividing data into learning data and evaluation data |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Underfitting** | Model is too simple to capture patterns |
| **Quantile** | A value that divides data into equal-sized groups |
| **Target Encoding** | Replacing a category with the average target value of that category |
| **Cyclical Encoding** | Using sin/cos to represent values that wrap around (months, days) |
| **Data Leakage** | When future/test data accidentally influences training |
| **joblib** | Library to save and load Python objects to/from disk |
| **Streamlit** | Python library for building interactive web apps |
| **AUC** | A model quality metric between 0.5 (random) and 1.0 (perfect) |
| **Confusion Matrix** | Table showing correct vs incorrect predictions per class |
| **Precision** | Of all positive predictions, how many were actually correct |
| **Recall** | Of all actual positives, how many did we correctly identify |
| **F1 Score** | A single metric balancing Precision and Recall |
| **Serialization** | Converting a Python object into a file that can be saved and reloaded |
| **`@st.cache_resource`** | Streamlit decorator to run a function only once and cache the result |

---

> 💡 **Key Takeaway:** Machine Learning is not magic. It is a structured process: understand your data → transform it intelligently → train multiple models → evaluate honestly → save everything → deploy carefully. Every step in this pipeline exists for a reason, and skipping any one of them causes real problems.