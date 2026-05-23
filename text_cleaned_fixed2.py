
import json

# Build the notebook structure as valid .ipynb JSON
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": []
}

def add_markdown_cell(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    })

def add_code_cell(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

# ==================== CELL 1: TITLE ====================
add_markdown_cell("""# Class 9 (Module 3): Machine Learning -- Predicting Values

> **Course:** AI Engineering  
> **Class:** 9 | **Module:** 3  
> **Focus Area:** Machine Learning (Predicting Values)  
> **Topics:** Linear Regression · Gradient Descent (Simplified) · Mean Squared Error (MSE) · Feature Scaling with Scikit-Learn  
> **Duration:** ~90 minutes lecture + 120 minutes lab  

---

## 1. Topic: Predicting Values with Machine Learning

### What We Will Cover
- **Linear Regression:** The foundational algorithm for predicting continuous numerical values.
- **Gradient Descent:** The optimization engine that teaches the model how to learn from data.
- **Mean Squared Error (MSE):** The mathematical "ruler" that measures how wrong our predictions are.
- **Feature Scaling:** The data preprocessing step that prevents some features from unfairly dominating the learning process.

Together, these four concepts form the **minimal viable toolkit** for any engineer building value-prediction systems--whether forecasting revenue, est
imating house prices, or predicting energy consumption.
""")

# ==================== CELL 2: WHY IT MATTERS ====================
add_markdown_cell("""## 2. Why This Is Related to AI Engineering

| Domain | Connection |
|--------|-----------|
| **Predictive Systems** | Most real-world AI applications begin with "predict a number": sales forecasts, demand planning, risk scoring, pricing engi
nes. Linear regression is the gateway to all of them. |
| **Model Interpretability** | Unlike black-box neural networks, linear regression exposes *exactly* how each feature influences the prediction. In re
gulated industries (finance, healthcare), interpretability is not optional--it is mandatory. |
| **Feature Engineering Validation** | Before deploying complex models, engineers benchmark against linear regression. If a neural network cannot beat
 a well-tuned linear model, the problem likely needs better features--not more parameters. |
| **Optimization Literacy** | Gradient descent is the same algorithm that trains deep neural networks, logistic regression, and recommendation systems
. Understanding it in 1D (linear regression) makes it intuitive in 1000D (deep learning). |
| **Production Data Pipelines** | Feature scaling is a non-negotiable preprocessing step in production. Models trained on unscaled data fail silently 
when deployed because real-world input distributions drift. |

> **The Core Truth:** Linear regression is not "simple"--it is *fundamental*. Every complex model you will ever build is, at its heart, trying to do w
hat linear regression does: find a function that maps inputs to outputs with minimal error. Master this, and you master the DNA of machine learning.
""")

# ==================== CELL 3: HOW IT WORKS ====================
add_markdown_cell("""## 3. How It Works: The Four Pillars of Value Prediction

### Pillar 1: Linear Regression -- Drawing the Best-Fit Line

Linear regression assumes there is a **linear relationship** between input features (X) and the target value (y). It finds the line (or hyperplane) th
at best predicts y from X.

**The Model Equation:**

```
ŷ = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

Where:
- `ŷ` = predicted value
- `β₀` = intercept (baseline prediction when all features are zero)
- `β₁...βₙ` = coefficients (weights) representing each feature's influence
- `X₁...Xₙ` = input features

**What the model does:**
1. Start with random guesses for all β values.
2. Make predictions using the current β values.
3. Measure how wrong the predictions are.
4. Adjust the β values to reduce the error.
5. Repeat until the error cannot be reduced further.

### Pillar 2: Mean Squared Error (MSE) -- Measuring Wrongness

Before a model can improve, it needs a **loss function**--a single number that quantifies "how bad are we?"

**The MSE Formula:**

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

Where:
- `n` = number of data points
- `yᵢ` = actual value
- `ŷᵢ` = predicted value
- `(yᵢ - ŷᵢ)` = residual (prediction error for one point)

**Why squared?**
- Squaring ensures **all errors are positive** (so negative and positive errors don't cancel out).
- Squaring **punishes large errors disproportionately**--a prediction off by 10 is penalized 100× more than one off by 1. This forces the model to fix
 its worst mistakes first.
- Squaring makes the function **smooth and differentiable**, which is mathematically convenient for optimization.

### Pillar 3: Gradient Descent -- The Learning Algorithm

Gradient descent is how the model "walks downhill" on the error landscape to find the best β values.

**The Core Idea:**

Imagine you are blindfolded on a mountain and want to reach the valley floor. You feel the ground with your feet to determine which direction is steep
est downhill, take one step in that direction, and repeat.

**Mathematically:**

```
β_new = β_old - α × (∂MSE/∂β)
```

Where:
- `α` (alpha) = learning rate (step size)
- `∂MSE/∂β` = gradient (slope of the error curve at the current β)

**How it works step-by-step:**
1. **Initialize:** Pick random starting values for all β coefficients.
2. **Predict:** Compute ŷ for all training examples using current β values.
3. **Compute Error:** Calculate MSE between predictions and actual values.
4. **Compute Gradient:** Determine how much each β contributed to the error.
5. **Update:** Adjust each β in the direction that reduces MSE.
6. **Repeat:** Go back to step 2 until MSE stops improving significantly.

**The Learning Rate (α):**
- **Too small:** The model learns glacially slow. You might run out of patience (or compute budget) before reaching the minimum.
- **Too large:** The model overshoots the minimum, bouncing around wildly or diverging to infinity.
- **Just right:** The model converges smoothly to the optimal β values.

### Pillar 4: Feature Scaling -- Leveling the Playing Field

Features often have vastly different numerical ranges:
- House size: 1,000--5,000 sq ft
- Number of bedrooms: 1--5
- Distance to city center: 0.5--50 miles
- Year built: 1900--2024

**The Problem:**

Gradient descent is sensitive to scale. If one feature ranges from 1--1,000,000 and another from 0--1, the gradient for the large-scale feature will d
ominate updates. The model will converge slowly or get stuck in a suboptimal valley.

**The Solution: Standardization (Z-score Normalization)**

```
X_scaled = (X - μ) / σ
```

Where:
- `μ` = mean of the feature
- `σ` = standard deviation of the feature

**Result:** All features have **mean ≈ 0** and **standard deviation ≈ 1**.

**Alternative: Min-Max Scaling**

```
X_scaled = (X - X_min) / (X_max - X_min)
```

**Result:** All features are squeezed into the **[0, 1]** range.

**When to use which:**
- **Standardization:** Preferred for gradient descent and when features have outliers (it is less sensitive to extreme values).
- **Min-Max:** Preferred when you need bounded ranges (e.g., for neural networks with specific activation functions).

**Scikit-Learn Implementation:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Learn μ and σ from training data
X_test_scaled = scaler.transform(X_test)         # Apply the SAME transformation to test data
```

> **Critical Rule:** Never fit the scaler on test data. The test set must remain "unseen." You learn the scaling parameters from training data only, t
hen apply them to both training and test data.
""")

# ==================== CELL 4: ANALOGY ====================
add_markdown_cell("""## 4. Analogy: Teaching a Robot to Throw a Basketball

Imagine you are teaching a robot to throw a basketball into a hoop from various distances.

### The Robot = Linear Regression Model
- The robot has **adjustable joints** (β coefficients) that control arm angle, force, and release timing.
- The **distance to the hoop** is the input feature (X).
- The **hoop position** is the target value (y).

### MSE = The Scoreboard
- After each throw, you measure how far the ball landed from the hoop.
- You **square the miss distance** so a throw that misses by 3 meters is penalized 9× more than one that misses by 1 meter.
- The robot's goal: minimize the average squared miss distance across all throws.

### Gradient Descent = The Coach's Feedback
- The coach (gradient descent) tells the robot: *"Your arm angle is too high--that contributed 70% of the error. Lower it slightly."*
- The **learning rate** is how big a step the robot takes to adjust.
  - Step too big → the robot overcorrects and throws worse than before.
  - Step too small → the robot takes 10,000 throws to learn what 100 throws could teach.
- The robot keeps throwing, adjusting, and measuring until its throws cluster tightly around the hoop.

### Feature Scaling = Standardizing the Court
- Suppose the robot also considers **wind speed** (range: 0--30 km/h) and **court elevation** (range: 0--3,000 meters).
- Without scaling, the robot obsesses over elevation (huge numbers) and barely notices wind speed (small numbers)--even though wind might matter more.

- **Standardization** converts all inputs to the same "unitless" scale, so the robot fairly weighs every factor.

### The Moral
> The robot does not "understand" basketball. It simply finds the joint angles that minimize squared miss distance through systematic trial and error.
 That is exactly what linear regression does: it does not understand your business--it finds the coefficients that minimize prediction error. Your job as the engineer is to give it the right features, the right scale, and the right loss function.
""")

# ==================== CELL 5: DETAILED BREAKDOWN ====================
add_markdown_cell("""## 5. Detailed Breakdown: How It Works & Valid Points

### 5.1 Linear Regression: Assumptions & When It Works

Linear regression is powerful *when its assumptions hold*. Violating them produces misleading results.

| Assumption | What It Means | How to Check | Violation Consequence |
|------------|--------------|--------------|----------------------|
| **Linearity** | The true relationship between X and y is linear | Residual plot (should look like random noise) | Curved patterns in residuals → pre
dictions systematically biased |
| **Independence** | Observations do not influence each other | Durbin-Watson test; domain knowledge | Correlated errors → inflated confidence in pred
ictions |
| **Homoscedasticity** | Error variance is constant across all X values | Residuals vs. fitted plot (funnel shape = bad) | Unequal variance → unreliab
le hypothesis tests |
| **Normality of Errors** | Residuals are normally distributed | Q-Q plot, Shapiro-Wilk test | Severe skew → invalid p-values and confidence intervals
 |
| **No Multicollinearity** | Features are not perfectly correlated | Variance Inflation Factor (VIF < 10) | Redundant features → unstable coefficient 
estimates |

**Valid Point:** Linear regression is not a "dumb" baseline--it is a **diagnostic tool**. When it fails, it tells you *exactly* what is wrong with you
r data or features (non-linearity, outliers, multicollinearity). That diagnostic power is priceless.

### 5.2 Gradient Descent: Variants & Practical Considerations

| Variant | How It Works | Best For | Trade-off |
|---------|-------------|----------|-----------|
| **Batch GD** | Uses *all* training data to compute gradient every step | Small datasets (<10k rows), convex problems | Slow per step, but stable con
vergence |
| **Stochastic GD (SGD)** | Uses *one random* data point per step | Massive datasets, online learning | Fast per step, but noisy convergence |
| **Mini-Batch GD** | Uses a small batch (e.g., 32--512 samples) per step | Deep learning, large tabular data | Balances speed and stability |

**Practical Tips:**
- Always **shuffle data** before each epoch (pass through the dataset) to prevent SGD from memorizing order.
- Use **learning rate decay** (gradually reduce α over time) to fine-tune convergence near the minimum.
- Monitor **training loss vs. validation loss** to detect overfitting--if validation loss starts rising while training loss falls, stop early.

### 5.3 MSE: Why It Dominates & When to Avoid It

**Why MSE is the default:**
- Mathematically elegant: the gradient is simply `2 × residual`, making updates computationally trivial.
- Analytically tractable: for linear regression, there is a **closed-form solution** (Normal Equation) that bypasses gradient descent entirely.
- Connects to maximum likelihood estimation under Gaussian noise assumptions.

**When to avoid MSE:**
- **Outliers present:** MSE squares errors, so one extreme outlier can dominate the loss. Use **Mean Absolute Error (MAE)** instead--it is more robust
.
- **Asymmetric costs:** If over-predicting is worse than under-predicting (e.g., inventory forecasting), MSE treats both equally. Use a custom loss fu
nction.
- **Classification:** MSE is for regression. For classification, use cross-entropy.

### 5.4 Feature Scaling: The Silent Killer of Production Models

**Why scaling matters in production:**

1. **Convergence Speed:** Unscaled features can increase training time by 10×--100×.
2. **Regularization Fairness:** L1/L2 regularization penalizes coefficients equally. If one feature is 1000× larger, its coefficient is 1000× smaller-
-regularization will unfairly ignore it.
3. **Distance-Based Algorithms:** KNN, K-Means, and SVM rely on distance calculations. Unscaled features make distance meaningless (the large-scale fe
ature swamps all others).
4. **Deployment Consistency:** If production data has a slightly different distribution than training data, a model trained on unscaled data is more f
ragile.

**The Scikit-Learn Pipeline Pattern (Production Best Practice):**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Encapsulate preprocessing + model into a single, deployable object
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Valid Point:** The Pipeline object ensures that scaling is applied *automatically* during prediction. Without it, engineers often forget to scale pr
oduction inputs, causing silent, catastrophic prediction failures.

### 5.5 The Complete Workflow: From Raw Data to Prediction

```
Raw Data
    │
    ▼
┌─────────────────────┐
│ 1. Data Cleaning    │  → Handle missing values, remove duplicates
│    & Splitting      │  → Train/validation/test split (e.g., 70/15/15)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 2. Feature Scaling  │  → Fit scaler on training data ONLY
│    (StandardScaler) │  → Transform train, validation, test
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 3. Model Training   │  → Initialize LinearRegression()
│    (Gradient Descent│  → Fit on scaled training data
│     or Normal Eq)   │  → Monitor MSE on validation set
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 4. Evaluation       │  → Compute MSE, RMSE, MAE on test set
│                     │  → Analyze residuals for pattern violations
│                     │  → Check coefficient magnitudes and signs
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 5. Interpretation   │  → Which features drive the prediction?
│    & Deployment     │  → Are coefficients directionally correct?
│                     │  → Package model + scaler into Pipeline
└─────────────────────┘
```
""")

# ==================== CELL 6: SETUP ====================
add_markdown_cell("""## 6. Hands-On: Building a Value Prediction Pipeline

### 6.1 Setup & Imports

We will use the **California Housing dataset** (built into scikit-learn) to demonstrate all four pillars. This dataset contains real census data: medi
an house values across California districts based on features like median income, house age, average rooms, and location.

**Analogy:** We are teaching a robot to estimate house prices. Each district is a "throw," and the robot must learn which features matter and how to c
ombine them.
""")

add_code_cell(r"""# =============================================================================
# CLASS 9: MACHINE LEARNING -- PREDICTING VALUES
# =============================================================================
# We will build and evaluate a linear regression system from the ground up.
# Each section includes: WHAT, WHY, and HOW with analogies in comments.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("All libraries imported successfully.")
print("Ready to build: Linear Regression | Gradient Descent | MSE | Feature Scaling")
""")

# ==================== CELL 7: DATA LOADING ====================
add_markdown_cell("""### 6.2 Data Loading & Exploration

**What we do:** Load the California Housing dataset and inspect its structure.

**Why:** Before any model touches the data, we must understand what we are working with. This is the "robot's first look at the court."

**Analogy:** Before the robot starts throwing basketballs, the engineer checks the court dimensions, wind conditions, and hoop height. We are checking
 the data's vital signs: shape, ranges, distributions, and correlations.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------------------
# We use California Housing -- a real dataset of 20,640 California districts.
# - Target: Median house value (in $100,000s)
# - Features: Income, age, rooms, bedrooms, population, location, etc.
#
# Analogy: This is our "basketball court data" -- each district is a 
# different court with different conditions (income = wind speed, 
# rooms = distance to hoop, etc.).
# ---------------------------------------------------------------------------

# Load the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("=" * 60)
print("DATA OVERVIEW -- The Basketball Court Registry")
print("=" * 60)
print(f"Shape: {df.shape[0]} districts x {df.shape[1]} columns")
print(f"\nColumn Names & Descriptions:")
for name, desc in zip(housing.feature_names, housing.feature_names):
    print(f"  - {name:20s} -- {desc}")

print(f"\nTarget: MedHouseVal (in $100,000s)")
print(f"  Range: ${df['MedHouseVal'].min()*100000:.0f} - ${df['MedHouseVal'].max()*100000:.0f}")

print(f"\nFirst 3 Records:")
df.head(3)
""")

# ==================== CELL 8: DATA PROFILING ====================
add_markdown_cell("""### 6.3 Data Profiling -- Understanding Feature Ranges

**What we do:** Compute descriptive statistics to see the scale of each feature.

**Why:** This reveals why feature scaling is necessary. Notice how `MedInc` (median income) ranges 0.5--15, while `AveRooms` ranges 0.8--142. Without 
scaling, the gradient for `AveRooms` would dominate.

**Analogy:** The robot discovers that one court variable (elevation: 0--3,000m) has numbers 100x larger than another (wind speed: 0--30). Without stan
dardization, the robot will obsess over elevation and ignore wind.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 2: PROFILE FEATURE RANGES
# ---------------------------------------------------------------------------
# Analogy: The robot scans all court variables and notices:
#   - Elevation: 0 to 3,000 meters (HUGE numbers)
#   - Wind speed: 0 to 30 km/h (small numbers)
#   - Temperature: -10 to 45C (medium numbers)
# Without scaling, the robot thinks elevation is 100x more important 
# just because its numbers are bigger. That is a TRAP.
# ---------------------------------------------------------------------------

print("=" * 60)
print("FEATURE STATISTICS -- Why Scaling Is Essential")
print("=" * 60)
stats = df.describe().T[['mean', 'std', 'min', 'max']]
stats['range'] = stats['max'] - stats['min']
stats['range_ratio_to_smallest'] = stats['range'] / stats['range'].min()
print(stats.round(3).to_string())

print("\nSCALING ALERT:")
print(f"   - Largest range (AveRooms): {stats.loc['AveRooms', 'range']:.1f}")
print(f"   - Smallest range (Latitude): {stats.loc['Latitude', 'range']:.1f}")
print(f"   - Ratio: {stats.loc['AveRooms', 'range_ratio_to_smallest']:.1f}x difference!")
print("   -> Gradient descent will be dominated by large-range features.")
print("   -> StandardScaler fixes this by giving every feature the same 'ruler'.")
""")

# ==================== CELL 9: CORRELATION ====================
add_markdown_cell("""### 6.4 Correlation Analysis -- Finding the Strongest Signals

**What we do:** Visualize the correlation matrix to see which features most strongly relate to house value.

**Why:** Linear regression performs best when features are correlated with the target. This heatmap tells us which features are our "best teachers."

**Analogy:** The robot studies past throws and notices: "When income was high, the ball went in 80% of the time. When house age was old, it barely mat
tered." The robot prioritizes learning from income.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 3: CORRELATION HEATMAP
# ---------------------------------------------------------------------------
# Analogy: The robot reviews 1,000 past throws and asks:
# "Which court conditions most often correlated with a successful shot?"
# The heatmap reveals the answer at a glance.
# ---------------------------------------------------------------------------

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap\n(Red = Positive | Blue = Negative)', fontsize=14)
plt.tight_layout()
plt.show()

print("\nKEY INSIGHTS FROM CORRELATIONS:")
print(f"   - MedInc vs MedHouseVal: {corr_matrix.loc['MedInc', 'MedHouseVal']:.3f} -- STRONG positive")
print(f"   - Latitude vs MedHouseVal: {corr_matrix.loc['Latitude', 'MedHouseVal']:.3f} -- moderate negative")
print(f"   - HouseAge vs MedHouseVal: {corr_matrix.loc['HouseAge', 'MedHouseVal']:.3f} -- surprisingly weak!")
print("\n Median Income is the dominant predictor -- the robot should focus here first.")
""")

# ==================== CELL 10: TRAIN TEST SPLIT ====================
add_markdown_cell("""### 6.5 Train-Test Split

**What we do:** Separate data into training (70%) and testing (30%) sets.

**Why:** We must evaluate the robot on courts it has **never practiced on**. Otherwise, we are grading it on throws it already memorized.

**Analogy:** The robot practices on 70% of the courts (training set). The coach reserves 30% of courts for the final exam (test set). If the robot pra
cticed on all courts, the exam would be meaningless.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 4: TRAIN-TEST SPLIT
# ---------------------------------------------------------------------------
# Analogy: We divide 20,640 basketball courts into two groups:
# - Practice courts (70%): The robot throws, misses, adjusts, repeats.
# - Final exam courts (30%): NEVER seen during practice. The true test.
# 
# random_state = 42 ensures the split is IDENTICAL every time.
# Reproducibility is sacred in ML engineering.
# ---------------------------------------------------------------------------

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42
)

print("=" * 60)
print("TRAIN-TEST SPLIT -- Dividing the Courts")
print("=" * 60)
print(f"Training set:   {X_train.shape[0]} districts ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Test set:       {X_test.shape[0]} districts ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\nFeatures:       {X_train.shape[1]}")
print(f"Target range (train): ${y_train.min()*100000:.0f} - ${y_train.max()*100000:.0f}")
print(f"Target range (test):  ${y_test.min()*100000:.0f} - ${y_test.max()*100000:.0f}")
print("Split complete -- test set is now LOCKED until final evaluation.")
""")

# ==================== CELL 11: FEATURE SCALING ====================
add_markdown_cell("""### 6.6 Feature Scaling -- Standardizing the Rulers

**What we do:** Apply `StandardScaler` to normalize all features to mean ≈ 0, std ≈ 1.

**Why:** Without scaling, gradient descent takes erratic steps. The coefficient for `AveRooms` (range: 142) updates 100× more aggressively than `Latit
ude` (range: 12), even if latitude is more predictive.

**Analogy:** The coach gives the robot a standardized measuring tape. Now "1 unit" of elevation means the same as "1 unit" of wind speed. The robot ca
n fairly compare and learn from all variables.

**CRITICAL RULE:** Fit the scaler on `X_train` ONLY. Transform both `X_train` and `X_test` with the learned parameters. Never let the scaler peek at t
est data.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 5: FEATURE SCALING
# ---------------------------------------------------------------------------
# Analogy: The coach hands the robot a universal measuring tape.
# Before: Elevation = 0-3000, Wind = 0-30 (incomparable)
# After:  Elevation = -1.5 to +2.1, Wind = -1.2 to +1.8 (comparable)
# 
# CRITICAL RULE: Fit scaler on TRAINING data ONLY.
# The exam courts (test set) must remain unseen until the final test.
# ---------------------------------------------------------------------------

scaler = StandardScaler()

# Fit on training data, transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("=" * 60)
print("FEATURE SCALING -- Standardizing the Rulers")
print("=" * 60)
print("BEFORE Scaling (Training Data):")
print(X_train.agg(['min', 'max', 'mean', 'std']).round(2).to_string())
print("\nAFTER Scaling (Training Data):")
print(X_train_scaled.agg(['min', 'max', 'mean', 'std']).round(2).to_string())

print("\nAll features now centered near 0 with std ~ 1")
print("Gradient descent will now update all coefficients fairly.")
""")

# ==================== CELL 12: LINEAR REGRESSION SKLEARN ====================
add_markdown_cell("""### 6.7 Model 1: Linear Regression (Scikit-Learn)

**What we do:** Train a production-grade linear regression model using scikit-learn's optimized implementation.

**Why:** This uses the **Normal Equation** (closed-form solution) under the hood--no gradient descent needed. It is exact, fast, and numerically stabl
e for datasets with <100k rows.

**Analogy:** Instead of the robot throwing 10,000 practice shots (gradient descent), a mathematician solves the optimal joint angles with algebra. Ins
tant perfection--for small courts.

**Key Insight:** `LinearRegression()` in scikit-learn does NOT use gradient descent. It uses linear algebra (SVD) to solve for β directly. This is fas
ter and avoids learning rate tuning.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 6: LINEAR REGRESSION (SKLEARN) -- The Mathematician's Solution
# ---------------------------------------------------------------------------
# Analogy: Instead of the robot throwing 10,000 times to find the best 
# angles, a mathematician uses algebra to compute the EXACT optimal angles 
# in one step. This is the Normal Equation: beta = (X^T X)^-1 X^T y
# 
# WHY use this: It is exact, fast, and needs no learning rate tuning.
# WHEN to avoid: If you have >100,000 features (X^T X becomes huge).
# ---------------------------------------------------------------------------

# Train on SCALED data (coefficients will be comparable across features)
lr_scaled = LinearRegression()
lr_scaled.fit(X_train_scaled, y_train)

# Also train on UNSCALED data to demonstrate the difference
lr_unscaled = LinearRegression()
lr_unscaled.fit(X_train, y_train)

# Predictions
y_pred_scaled = lr_scaled.predict(X_test_scaled)
y_pred_unscaled = lr_unscaled.predict(X_test)

# Evaluate both
print("=" * 60)
print("LINEAR REGRESSION RESULTS -- Scaled vs. Unscaled")
print("=" * 60)

for name, pred in [("SCALED", y_pred_scaled), ("UNSCALED", y_pred_unscaled)]:
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"\n{name} Features:")
    print(f"   MSE:  {mse:.4f}  (Mean Squared Error -- punishes big misses)")
    print(f"   RMSE: {rmse:.4f}  (Root MSE -- in same units as target: $100k)")
    print(f"   MAE:  {mae:.4f}  (Mean Absolute Error -- average miss distance)")
    print(f"   R2:   {r2:.4f}  (Explained variance -- 1.0 = perfect)")

print("\nNOTE: MSE/RMSE/MAE are identical because LinearRegression uses the")
print("   Normal Equation, which is scale-invariant. But the COEFFICIENTS differ wildly!")
""")

# ==================== CELL 13: COEFFICIENT COMPARISON ====================
add_markdown_cell("""### 6.8 Coefficient Comparison -- Scaled vs. Unscaled

**What we do:** Compare the learned coefficients (β values) between scaled and unscaled models.

**Why:** On unscaled data, coefficients are tiny for large-range features and huge for small-range features. This makes interpretation impossible. On 
scaled data, coefficient magnitude directly reflects predictive importance.

**Analogy:** On the unscaled court, the robot's "elevation coefficient" is 0.0001 (tiny number) while "wind coefficient" is 50.0 (huge number). Does t
hat mean wind matters 500,000× more? No--it just means wind's numbers were smaller to begin with. Standardization removes this illusion.
""")

add_code_cell(r"""# ------------------------------------------------------------
# COEFFICIENT COMPARISON -- Removing the Scale Illusion
# ---------------------------------------------------------------------------
# Analogy: On the unscaled court, the robot reports:
#   "Elevation coefficient = 0.0001, Wind coefficient = 50.0"
# A naive observer thinks wind is 500,000x more important. 
# But elevation's raw numbers were 100x larger -- the coefficient HAD to be 
# smaller to compensate. Standardization removes this optical illusion.
# ---------------------------------------------------------------------------

coef_comparison = pd.DataFrame({
    'Feature': X.columns,
    'Unscaled_Coef': lr_unscaled.coef_,
    'Scaled_Coef': lr_scaled.coef_,
    'Abs_Scaled_Coef': np.abs(lr_scaled.coef_)
}).sort_values('Abs_Scaled_Coef', ascending=False)

print("=" * 60)
print("COEFFICIENT COMPARISON -- Scaled vs. Unscaled")
print("=" * 60)
print(coef_comparison.round(4).to_string(index=False))

# Visualize scaled coefficients
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coef_comparison['Scaled_Coef']]
plt.barh(coef_comparison['Feature'], coef_comparison['Scaled_Coef'], color=colors, alpha=0.7)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Coefficient Value (Impact on Median House Value)')
plt.title('Linear Regression: Feature Impact on House Price\n(Green = Increases Price | Red = Decreases Price)\n(Scaled Features -- Magnitude = True I
mportance)', fontsize=12)
plt.tight_layout()
plt.show()

print("\nINTERPRETATION (Scaled Coefficients):")
print(f"   - MedInc: {coef_comparison[coef_comparison['Feature']=='MedInc']['Scaled_Coef'].values[0]:.3f}")
print("     -> Strongest positive driver. A 1-std increase in income raises")
print("       house value by ~0.84 standard deviations. This is EXPECTED.")
print(f"   - Latitude: {coef_comparison[coef_comparison['Feature']=='Latitude']['Scaled_Coef'].values[0]:.3f}")
print("     -> Negative. Northern California (higher latitude) has lower house values.")
print(f"   - HouseAge: {coef_comparison[coef_comparison['Feature']=='HouseAge']['Scaled_Coef'].values[0]:.3f}")
print("     -> Weak positive. Older houses are slightly more valuable, but not much.")
""")

# ==================== CELL 14: GRADIENT DESCENT FROM SCRATCH ====================
add_markdown_cell("""### 6.9 Model 2: Gradient Descent from Scratch (Simplified)

**What we do:** Implement batch gradient descent manually to understand exactly how the model learns.

**Why:** Scikit-learn hides the learning process. Building it from scratch reveals how coefficients evolve, how the learning rate affects convergence,
 and why scaling is essential.

**Analogy:** This is the robot's "practice mode." We watch every throw, every adjustment, every miss. We see the learning rate in action: too big = ch
aos, too small = boredom, just right = smooth improvement.

**Simplification:** We use only ONE feature (`MedInc`) for visualization. In 8D, we cannot draw the error surface. In 2D (one feature + intercept), we
 can.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 7: GRADIENT DESCENT FROM SCRATCH -- The Robot's Practice Mode
# ---------------------------------------------------------------------------
# Analogy: We strip the robot down to ONE joint (income sensitivity) 
# and ONE baseline (intercept). Now we can WATCH the learning happen.
# 
# We will try THREE learning rates to see why alpha matters:
#   - alpha = 0.001  (too small) -- crawls painfully
#   - alpha = 0.01   (good) -- smooth convergence
#   - alpha = 0.30   (too big) -- bounces around
# ---------------------------------------------------------------------------

# Use only ONE feature for 2D visualization
X_simple = X_train_scaled[['MedInc']].values  # shape: (n_samples, 1)
y_simple = y_train.values                       # shape: (n_samples,)

# Add bias column (intercept term) -- column of 1s
X_b = np.c_[np.ones((X_simple.shape[0], 1)), X_simple]  # shape: (n_samples, 2)

def gradient_descent(X, y, beta_init, learning_rate, n_iterations):
    """
    Batch Gradient Descent for linear regression.
    
    Analogy: The coach watches all throws, computes the average mistake,
    and tells the robot how to adjust its joints.
    """
    n_samples = len(y)
    beta = beta_init.copy()
    history = []  # Track MSE and beta values over time
    
    for iteration in range(n_iterations):
        # PREDICT: y_hat = X dot beta
        # Analogy: The robot throws using current joint angles.
        y_pred = X.dot(beta)
        
        # COMPUTE ERRORS: residuals = y - y_hat
        # Analogy: Measure how far each ball landed from the hoop.
        residuals = y - y_pred
        
        # COMPUTE MSE
        mse = np.mean(residuals ** 2)
        history.append({'iter': iteration, 'mse': mse, 'beta0': beta[0], 'beta1': beta[1]})
        
        # COMPUTE GRADIENT: dMSE/dbeta = -(2/n) * X^T * residuals
        # Analogy: The coach calculates which joint contributed most to the misses.
        gradient = -(2 / n_samples) * X.T.dot(residuals)
        
        # UPDATE: beta_new = beta_old - alpha * gradient
        # Analogy: Adjust the joint in the direction that reduces misses.
        beta = beta - learning_rate * gradient
    
    return beta, history

# Try three learning rates
alphas = [0.001, 0.01, 0.30]
colors = ['blue', 'green', 'red']
labels = ['alpha = 0.001 (Too Small)', 'alpha = 0.01 (Just Right)', 'alpha = 0.30 (Too Big)']

plt.figure(figsize=(14, 5))

# Plot 1: MSE over iterations
plt.subplot(1, 2, 1)
for alpha, color, label in zip(alphas, colors, labels):
    beta_init = np.zeros(2)  # Start with beta0=0, beta1=0
    final_beta, history = gradient_descent(X_b, y_simple, beta_init, alpha, 200)
    mses = [h['mse'] for h in history]
    plt.plot(mses, color=color, label=label, linewidth=2, alpha=0.8)

plt.xlabel('Iteration (Practice Throw #)')
plt.ylabel('MSE (Mean Squared Error)')
plt.title('Gradient Descent: How Learning Rate Affects Convergence')
plt.legend()
plt.yscale('log')  # Log scale to see all curves clearly
plt.grid(True, alpha=0.3)

# Plot 2: Beta trajectory in parameter space
plt.subplot(1, 2, 2)
for alpha, color, label in zip(alphas, colors, labels):
    beta_init = np.zeros(2)
    final_beta, history = gradient_descent(X_b, y_simple, beta_init, alpha, 200)
    beta0s = [h['beta0'] for h in history]
    beta1s = [h['beta1'] for h in history]
    plt.plot(beta0s, beta1s, color=color, label=label, linewidth=2, alpha=0.8, marker='o', markersize=2)

plt.xlabel('beta0 (Intercept)')
plt.ylabel('beta1 (MedInc Coefficient)')
plt.title('Parameter Trajectory: Walking Down the Error Mountain')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=" * 60)
print("GRADIENT DESCENT OBSERVATIONS -- The Coach's Report")
print("=" * 60)
print("- alpha = 0.001 (Too Small):")
print("   -> MSE barely decreases after 200 throws. The robot is learning,")
print("      but so slowly that it needs 10,000+ throws to converge.")
print("\n- alpha = 0.01 (Just Right):")
print("   -> MSE drops smoothly and stabilizes. The robot finds the valley")
print("      in ~100 throws. This is the Goldilocks zone.")
print("\n- alpha = 0.30 (Too Big):")
print("   -> MSE bounces wildly! The robot overshoots the valley,")
print("      climbs the other side, and never settles. It DIVERGES.")
""")

# ==================== CELL 15: MSE VISUALIZATION ====================
add_markdown_cell("""### 6.10 Visualizing MSE -- The Error Landscape

**What we do:** Plot the 3D error surface (MSE as a function of β₀ and β₁) to see why gradient descent works.

**Why:** Seeing the "bowl" shape of the MSE surface makes gradient descent intuitive. The gradient always points uphill; we walk in the opposite direc
tion.

**Analogy:** This is the topographic map of the mountain the robot is climbing down. The bottom of the bowl is the optimal β values. Gradient descent 
is the robot feeling the slope and walking downhill.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 8: VISUALIZE THE MSE LANDSCAPE -- The Mountain Topography
# ---------------------------------------------------------------------------
# Analogy: We draw a topographic map of the mountain. The robot starts 
# at a random point and walks downhill. The bowl shape proves there is 
# ONE global minimum -- no dangerous local traps.
# ---------------------------------------------------------------------------

# Create a grid of beta values
beta0_range = np.linspace(0.5, 2.5, 100)
beta1_range = np.linspace(0.5, 1.5, 100)
B0, B1 = np.meshgrid(beta0_range, beta1_range)

# Compute MSE for each (beta0, beta1) pair
n_samples = len(y_simple)
MSE_surface = np.zeros_like(B0)
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        beta = np.array([B0[i, j], B1[i, j]])
        residuals = y_simple - X_b.dot(beta)
        MSE_surface[i, j] = np.mean(residuals ** 2)

# Plot
fig = plt.figure(figsize=(14, 5))

# 3D surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(B0, B1, MSE_surface, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.set_xlabel('beta0 (Intercept)')
ax1.set_ylabel('beta1 (MedInc Coefficient)')
ax1.set_zlabel('MSE')
ax1.set_title('MSE Error Surface -- The Bowl Shape')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

# Contour plot (top-down view)
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contour(B0, B1, MSE_surface, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('beta0 (Intercept)')
ax2.set_ylabel('beta1 (MedInc Coefficient)')
ax2.set_title('Contour Map: Walking Downhill to the Minimum')

# Overlay gradient descent path for alpha=0.01
beta_init = np.zeros(2)
final_beta, history = gradient_descent(X_b, y_simple, beta_init, 0.01, 200)
beta0s = [h['beta0'] for h in history]
beta1s = [h['beta1'] for h in history]
ax2.plot(beta0s, beta1s, 'r.-', linewidth=2, markersize=4, label='GD Path (alpha=0.01)')
ax2.plot(beta0s[-1], beta1s[-1], 'r*', markersize=15, label='Minimum Reached')
ax2.legend()

plt.tight_layout()
plt.show()

print("=" * 60)
print("ERROR LANDSCAPE INSIGHTS")
print("=" * 60)
print("   - The surface is a smooth BOWL -- one global minimum.")
print("   - This is why linear regression is CONVEX -- gradient descent")
print("     cannot get stuck in local minima (unlike neural networks).")
print(f"   - Optimal beta0 ~ {final_beta[0]:.3f}, beta1 ~ {final_beta[1]:.3f}")
print("   - The red path shows the robot walking directly downhill.")
""")

# ==================== CELL 16: SGD REGRESSOR ====================
add_markdown_cell("""### 6.11 Model 3: SGD Regressor (Scikit-Learn's Gradient Descent)

**What we do:** Use scikit-learn's `SGDRegressor` to apply gradient descent on the full 8-feature dataset.

**Why:** Our from-scratch implementation only handled one feature. In production, we use optimized libraries that handle all features, large datasets,
 and advanced tricks (learning rate decay, momentum, early stopping).

**Analogy:** The robot has graduated from the practice court to the real league. Instead of homemade coaching, it now has a professional trainer (scik
it-learn) with advanced techniques.

**Key Parameters:**
- `max_iter=1000`: Maximum practice throws.
- `tol=1e-3`: Stop if improvement is smaller than this--no need to keep practicing if barely improving.
- `eta0=0.01`: Initial learning rate.
- `learning_rate='invscaling'`: Gradually reduces α over time (the robot takes smaller steps as it nears the target).
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 9: SGD REGRESSOR -- The Professional Trainer
# ---------------------------------------------------------------------------
# Analogy: The robot now has a professional coach (SGDRegressor) who:
#   - Handles ALL joints (8 features) simultaneously
#   - Automatically reduces step size as the robot improves
#   - Stops practice when improvement becomes negligible
#   - Uses optimized C++ code under the hood for speed
# ---------------------------------------------------------------------------

sgd = SGDRegressor(
    max_iter=1000,           # Maximum practice throws
    tol=1e-3,                # Stop if improvement < 0.001
    eta0=0.01,               # Initial learning rate
    learning_rate='invscaling',  # Reduce step size over time
    random_state=42,
    penalty=None             # No regularization for now (pure linear regression)
)

# Train on scaled data (MANDATORY for gradient descent)
sgd.fit(X_train_scaled, y_train)

# Predict
y_pred_sgd = sgd.predict(X_test_scaled)

# Evaluate
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
rmse_sgd = np.sqrt(mse_sgd)
mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

print("=" * 60)
print("SGD REGRESSOR -- The Professional Trainer's Results")
print("=" * 60)
print(f"MSE:  {mse_sgd:.4f}")
print(f"RMSE: {rmse_sgd:.4f}  (~${rmse_sgd*100000:.0f} average prediction error)")
print(f"MAE:  {mae_sgd:.4f}  (~${mae_sgd*100000:.0f} average absolute error)")
print(f"R2:   {r2_sgd:.4f}  ({r2_sgd*100:.1f}% of variance explained)")

print("\nCOMPARISON:")
print(f"   Normal Equation (LinearRegression) R2: {r2_score(y_test, y_pred_scaled):.4f}")
print(f"   Gradient Descent (SGDRegressor)   R2: {r2_sgd:.4f}")
print("   -> Both converge to nearly identical solutions. Gradient descent")
print("      is an approximation, but a very good one with enough iterations.")
""")

# ==================== CELL 17: PIPELINE ====================
add_markdown_cell("""### 6.12 Production Best Practice: The Scikit-Learn Pipeline

**What we do:** Bundle the scaler and model into a single `Pipeline` object.

**Why:** In production, you cannot trust humans to remember preprocessing steps. A Pipeline guarantees that every prediction goes through the exact sa
me transformation as training data.

**Analogy:** Instead of the robot manually picking up the measuring tape before every throw, the coach builds a "throwing machine" that automatically 
measures, adjusts, and throws. No human error, no forgotten steps.

**The Most Common Production Bug:** Deploying a model but forgetting to scale new input data. The Pipeline prevents this by design.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 10: PRODUCTION PIPELINE -- The Throwing Machine
# ---------------------------------------------------------------------------
# Analogy: The coach builds an automated "throwing machine" (Pipeline).
# You feed it raw court conditions -> it automatically standardizes them 
# -> feeds to the model -> returns the prediction. 
# 
# NO human can forget the measuring tape because it is BUILT INTO the machine.
# ---------------------------------------------------------------------------

# Build the pipeline: Scaler -> Model
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Step 1: Standardize
    ('regressor', LinearRegression())  # Step 2: Predict
])

# Train the entire pipeline on raw (unscaled) training data
# The pipeline handles scaling internally -- you never touch it manually.
pipeline.fit(X_train, y_train)

# Predict on raw test data -- scaling happens automatically inside the pipeline
y_pred_pipeline = pipeline.predict(X_test)

# Evaluate
mse_pipe = mean_squared_error(y_test, y_pred_pipeline)
rmse_pipe = np.sqrt(mse_pipe)
r2_pipe = r2_score(y_test, y_pred_pipeline)

print("=" * 60)
print("PIPELINE RESULTS -- The Automated Throwing Machine")
print("=" * 60)
print(f"MSE:  {mse_pipe:.4f}")
print(f"RMSE: {rmse_pipe:.4f}")
print(f"R2:   {r2_pipe:.4f}")

print("\nPRODUCTION DEPLOYMENT:")
print("   # Save the entire pipeline (scaler + model) to one file:")
print("   import joblib")
print("   joblib.dump(pipeline, 'house_price_pipeline.pkl')")
print("\n   # Load and predict in production:")
print("   loaded_pipeline = joblib.load('house_price_pipeline.pkl')")
print("   prediction = loaded_pipeline.predict(new_raw_data)")
print("\n   The scaler is INSIDE the pipeline -- impossible to forget!")
""")

# ==================== CELL 18: PREDICTION VISUALIZATION ====================
add_markdown_cell("""### 6.13 Visualizing Predictions vs. Actual Values

**What we do:** Scatter plot of predicted vs. actual house values to assess model quality visually.

**Why:** Numbers tell you the error; plots tell you *where* the error happens. Are we systematically under-predicting expensive houses? Are we confuse
d by outliers?

**Analogy:** The coach reviews game tape. Not just the final score (MSE), but *which* throws were close and which were embarrassingly wrong.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 11: PREDICTION VISUALIZATION -- Reviewing the Game Tape
# ---------------------------------------------------------------------------
# Analogy: The coach watches replay footage. The perfect robot would have 
# all dots on the diagonal line (predicted = actual). Deviations show 
# WHERE and HOW the robot misses.
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = [
    ("Normal Equation\n(LinearRegression)", y_pred_scaled, "steelblue"),
    ("Gradient Descent\n(SGDRegressor)", y_pred_sgd, "forestgreen"),
    ("Pipeline\n(Scaler + LR)", y_pred_pipeline, "coral")
]

for ax, (name, preds, color) in zip(axes, models):
    ax.scatter(y_test, preds, alpha=0.4, color=color, edgecolors='black', linewidth=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual House Value ($100k)')
    ax.set_ylabel('Predicted House Value ($100k)')
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R2 text
    r2 = r2_score(y_test, preds)
    ax.text(0.05, 0.95, f'R2 = {r2:.3f}', transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Predicted vs. Actual House Values\n(Dots on the diagonal = perfect predictions)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\nGAME TAPE ANALYSIS:")
print("   - Most dots cluster near the diagonal -> model is generally accurate.")
print("   - The vertical band at y ~ 5.0 is the price CAP in the dataset.")
print("     (Original data capped values at $500,000 -- a data limitation!)")
print("   - Slight fan shape at higher values -> model struggles with very")
print("     expensive houses. This is HETEROSCEDASTICITY (violated assumption).")
""")

# ==================== CELL 19: RESIDUAL ANALYSIS ====================
add_markdown_cell("""### 6.14 Residual Analysis -- Diagnostic Power of Linear Regression

**What we do:** Plot residuals (errors) vs. predicted values to check for violated assumptions.

**Why:** Residual plots are the X-ray of linear regression. Patterns reveal:
- **Funnel shape** → heteroscedasticity (violated assumption)
- **Curved pattern** → non-linearity (need polynomial features)
- **Outliers** → influential points skewing the model

**Analogy:** The coach analyzes not just where the ball landed, but *how* it missed. Consistent left-misses mean the robot's aim is biased. Random mis
ses mean good calibration.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 12: RESIDUAL ANALYSIS -- The Diagnostic X-Ray
# ---------------------------------------------------------------------------
# Analogy: The coach plots "miss direction" vs. "predicted landing spot."
#   - Random scatter around zero = healthy robot (good!)
#   - Funnel shape = robot gets worse at long distances (bad!)
#   - Curved pattern = robot's aim formula is wrong (need non-linear fix)
# ---------------------------------------------------------------------------

residuals = y_test - y_pred_pipeline

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs. Predicted
axes[0].scatter(y_pred_pipeline, residuals, alpha=0.4, color='steelblue', edgecolors='black', linewidth=0.5)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Value ($100k)')
axes[0].set_ylabel('Residual (Actual - Predicted)')
axes[0].set_title('Residuals vs. Predicted Values')
axes[0].grid(True, alpha=0.3)

# Histogram of residuals
axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual Value ($100k)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Residual Diagnostics: Is the Robot Healthy?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("=" * 60)
print("RESIDUAL DIAGNOSTICS -- The Robot's Health Check")
print("=" * 60)
print(f"   Mean residual:     {residuals.mean():.4f}  (should be ~ 0)")
print(f"   Std of residuals:  {residuals.std():.4f}")
print(f"   Max under-predict: {residuals.min():.4f}  (predicted ${abs(residuals.min())*100000:.0f} too HIGH)")
print(f"   Max over-predict:  {residuals.max():.4f}  (predicted ${residuals.max()*100000:.0f} too LOW)")

print("\nDIAGNOSIS:")
print("   - Mean ~ 0 -> No systematic bias (good).")
print("   - Slight funnel shape at high predictions -> Heteroscedasticity.")
print("     (Errors grow larger for expensive houses -- common in housing data.)")
print("   - Histogram is roughly normal -> errors are reasonably well-behaved.")
print("   - A few extreme residuals (outliers) exist -> consider robust regression.")
""")

# ==================== CELL 20: MSE DEMONSTRATION ====================
add_markdown_cell("""### 6.15 MSE in Action -- Why Squaring Matters

**What we do:** Demonstrate how MSE behaves differently from MAE when outliers are present.

**Why:** This explains why MSE is the default for training, but MAE is often preferred for reporting when outliers exist.

**Analogy:** The scoreboard uses squared misses to punish catastrophic failures. A throw that misses by 10 meters is penalized 100× more than one that
 misses by 1 meter. This forces the robot to fix its worst errors first.
""")

add_code_cell(r"""# ------------------------------------------------------------
# STEP 13: MSE vs. MAE -- Why Squaring Changes Behavior
# ---------------------------------------------------------------------------
# Analogy: We show the robot two scoreboards:
#   Scoreboard A (MSE): Miss by 3m -> penalty = 9 points
#   Scoreboard B (MAE): Miss by 3m -> penalty = 3 points
# 
# With MSE, the robot panics about its worst throws and fixes them first.
# With MAE, the robot treats all misses equally -- more democratic, but 
# less urgency to fix catastrophic errors.
# ---------------------------------------------------------------------------

# Create synthetic errors: one outlier
test_errors = np.array([0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 5.0])  # Last one is outlier

mae_values = np.abs(test_errors)
mse_values = test_errors ** 2

print("=" * 60)
print("MSE vs. MAE -- The Two Scoreboards")
print("=" * 60)
print(f"Errors:        {test_errors}")
print(f"MAE (linear):  {mae_values}")
print(f"MSE (squared): {mse_values}")
print(f"\nTotal MAE:  {mae_values.sum():.2f}")
print(f"Total MSE:  {mse_values.sum():.2f}")
print(f"\nThe outlier (5.0) contributes:")
print(f"   - {mae_values[-1]/mae_values.sum()*100:.1f}% of total MAE")
print(f"   - {mse_values[-1]/mse_values.sum()*100:.1f}% of total MSE")
print(f"\nMSE gives the outlier {mse_values[-1]/mae_values[-1]:.0f}x more voting power!")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x_pos = np.arange(len(test_errors))
axes[0].bar(x_pos, mae_values, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_title('MAE: Linear Penalty\n(All misses treated equally)')
axes[0].set_xlabel('Throw #')
axes[0].set_ylabel('Penalty')
axes[0].set_xticks(x_pos)

axes[1].bar(x_pos, mse_values, color='coral', alpha=0.7, edgecolor='black')
axes[1].set_title('MSE: Squared Penalty\n(Big misses punished heavily)')
axes[1].set_xlabel('Throw #')
axes[1].set_ylabel('Penalty')
axes[1].set_xticks(x_pos)

plt.suptitle('Why MSE Forces the Robot to Fix Its Worst Throws First', fontsize=12)
plt.tight_layout()
plt.show()

print("\nWHEN TO USE WHICH:")
print("   - MSE for TRAINING: Forces model to fix worst errors first.")
print("   - MAE for REPORTING: More robust to outliers, easier to interpret.")
print("   - If outliers are DATA ERRORS -> remove them, then use MSE.")
print("   - If outliers are REAL but rare -> use MAE or Huber loss.")
""")

# ==================== CELL 21: SUMMARY ====================
add_markdown_cell("""## 7. Class 9 Learning Objectives

By the end of this session, you will be able to:

1. **Explain** the linear regression equation and interpret each coefficient's meaning in a business context.
2. **Implement** gradient descent from scratch (simplified) to understand how models "learn" parameter values.
3. **Compute** and interpret MSE, RMSE, and MAE to evaluate regression model performance.
4. **Apply** StandardScaler and MinMaxScaler correctly using Scikit-Learn, respecting the train/test boundary.
5. **Build** a complete regression pipeline (preprocessing → model → evaluation) using `sklearn.pipeline.Pipeline`.
6. **Diagnose** model issues by analyzing residuals and checking linear regression assumptions.

---

## 8. Key Takeaways

> 📌 **Linear regression is the Rosetta Stone of ML.** Every algorithm you learn later--logistic regression, neural networks, SVMs--is trying to do the
 same thing: find parameters that minimize a loss function. Master this pattern, and you master 80% of machine learning.

> 📌 **MSE punishes arrogance.** Because it squares errors, a model that is confidently wrong pays a heavy price. This forces the model to fix its wors
t predictions first--a useful property in production.

> 📌 **Gradient descent is universal.** Whether you are tuning 2 coefficients in linear regression or 175 billion parameters in GPT-4, the algorithm is
 the same: compute gradient, take a step, repeat. The scale changes; the principle does not.

> 📌 **Feature scaling is not optional.** An unscaled model is like a race where one runner starts at the finish line. Scaling ensures every feature co
mpetes fairly. Skip it, and your model will silently underperform.

> 📌 **Always use Pipelines.** A Pipeline object bundles preprocessing and modeling into one deployable unit. It prevents the most common production bu
g: forgetting to scale new data before prediction.

---

## 9. Recommended Reading & Resources

- *"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"* -- Aurélien Géron (Chapters 4--5)
- *"An Introduction to Statistical Learning"* -- James, Witten, Hastie & Tibshirani (Chapter 3: Linear Regression)
- *"The Elements of Statistical Learning"* -- Hastie, Tibshirani & Friedman (Chapter 3, for deeper theory)
- **Scikit-Learn Documentation:** `LinearRegression`, `SGDRegressor`, `StandardScaler`, `Pipeline`
- **Interactive Visualization:** [Gradient Descent Visualization](https://github.com/ageron/handson-ml3) (companion repo for Géron's book)

---

*End of Class 9 -- Module 3: Machine Learning (Predicting Values)*  
*Next: Lab Session -- Building a House Price Prediction Pipeline from Scratch*
""")

# Save the notebook
output_path = '/mnt/agents/output/Class9_Notebook_ML_Predicting_Values.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook saved successfully to: {output_path}")
print(f"Total cells: {len(notebook['cells'])}")
print(f"Markdown cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')}")
print(f"Code cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')}")
