# Week 2 — Class 4 Project: Data Audit Pipeline

## Achievement
Perform a complete audit on a "dirty" row dataset — detecting missing values, outliers, inconsistencies, and generating a comprehensive cleaning report.

---

## Production-Grade File Structure

```
week2_class4_project/
├── README.md                     # You are here
├── requirements.txt              # Pinned dependencies
├── config/
│   └── settings.yaml             # Audit rules, thresholds, dtype mappings
├── src/
│   ├── __init__.py               # Package marker
│   ├── data_auditor.py           # Core: audit + clean + report generation
│   └── utils/
│       ├── __init__.py
│       └── file_handler.py       # CSV/Excel/JSON I/O utilities
├── data/
│   ├── raw/                      # Input datasets
│   ├── clean/                    # Cleaned + merged outputs
│   └── audit_reports/            # JSON audit reports
└── main.py                       # Single entry point
```

---

## Step-by-Step Instructions

### Step 1: Set Up Environment

```bash
cd week2_class4_project
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# OR
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

---

### Step 2: Run the Audit Pipeline

```bash
python main.py
```

**What happens:**
1. **Creates dirty data** with intentional problems (NaN, outliers, inconsistent labels, wrong types, duplicates)
2. **Runs comprehensive audit** — missing stats, outlier counts, memory usage, per-column health
3. **Applies automated cleaning** — drops bad columns, imputes missing, caps outliers, standardizes labels, enforces dtypes
4. **GroupBy analysis** — segments by category and country
5. **Merge demonstration** — inner join + left join with secondary dataset
6. **Advanced analysis** — pivot tables, correlation matrix
7. **Saves everything** — clean CSVs + JSON audit reports

---

## Key Concepts Demonstrated

### DataFrame as the Universal Container
```python
df = pd.DataFrame(data)  # 2D labeled table
df['column']             # Series (1D column)
df.loc[0:5, 'col']       # Label-based slicing
```
**ANALOGY:** DataFrame = restaurant prep station. Series = single ingredient tray.

### Handling NaN
```python
df.isnull().sum()        # Count missing per column
df['col'].fillna(median) # Impute with median
df.dropna()              # Remove rows with any NaN
```
**ANALOGY:** NaN = empty salt shaker. Impute = borrow from neighbor. Drop = skip the dish.

### Outlier Detection (IQR)
```python
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df < Q1 - 1.5*IQR) | (df > Q3 + 1.5*IQR)]
```
**ANALOGY:** IQR = the "normal range" of a heartbeat monitor. Anything outside triggers an alarm.

### GroupBy (Split-Apply-Combine)
```python
df.groupby('category').agg({'income': 'mean', 'age': 'median'})
```
**ANALOGY:** School report cards. Split into classes, compute average per class, combine into summary table.

### Merge (SQL Joins)
```python
pd.merge(df1, df2, on='customer_id', how='left')
```
**ANALOGY:** Jigsaw puzzle. Left join = keep your puzzle, add matching pieces where they fit.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pandas` | Run `pip install -r requirements.txt` |
| `MemoryError` | Reduce `n` in `create_dirty_dataset()` (default: 200) |
| `KeyError during merge` | Check that both DataFrames have the join key column |
| `Dtype conversion failed` | Check `config/settings.yaml` dtype_mapping matches your data |

---

## Learning Checklist
- [ ] I can create a DataFrame and understand the difference from a Python dict.
- [ ] I can detect missing values and choose between drop/impute/remove.
- [ ] I can detect outliers using IQR and Z-score methods.
- [ ] I can standardize inconsistent categorical labels.
- [ ] I can use GroupBy to compute segment statistics.
- [ ] I can merge two datasets using inner, left, and outer joins.
- [ ] I can create pivot tables and correlation matrices.
- [ ] I understand why data cleaning is 80% of an AI engineer's job.
