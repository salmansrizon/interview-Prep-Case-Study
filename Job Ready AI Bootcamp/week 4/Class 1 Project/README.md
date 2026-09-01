# 🧪 Prompt A/B Test & Hypothesis Testing

## সহজ ভাষায় Project Overview

Prompt B-এর accuracy Prompt A-এর চেয়ে বেশি দেখালেই improvement real প্রমাণ হয় না; difference random chance থেকেও আসতে পারে। এই project **CLT, T-Test, P-Value, ANOVA এবং Correlation vs. Causation** ব্যবহার করে statistically defensible decision নিতে শেখায়।

## Project Pipeline

```text
Experiment Data → Assumption Check → Hypothesis Test
                → P-Value/Effect Interpretation → Business Decision
```

## Files

- `week4_class8_project.ipynb` — complete guided A/B-testing project
- `project.ipynb` — additional project notebook
- `test.py` — supporting executable check

## কীভাবে Run করবেন?

```powershell
cd "week 4\Class 1 Project"
python -m pip install jupyter numpy pandas scipy matplotlib seaborn
python -m jupyter notebook week4_class8_project.ipynb
```

## Validation Checklist

- Null এবং Alternative Hypothesis test-এর আগে define করা
- Test type data design-এর সাথে match করে
- P-Value-কে “H0 true হওয়ার probability” বলা হয় না
- Statistical significance-এর সাথে effect size/business value check করা
- Correlation থেকে unsupported causation claim করা হয় না
