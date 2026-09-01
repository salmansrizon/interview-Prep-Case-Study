# 📈 Statistics for AI: Probability & Confidence

## সহজ ভাষায় Project Overview

এই interactive notebook-এ **Normal Distribution, Z-Score, Frequentist Probability, Bayesian Probability, Confidence এবং Calibration** হাতে-কলমে দেখানো হয়েছে। Model score দেখেই blindly trust না করে uncertainty কীভাবে measure করতে হয় সেটাই project-এর মূল লক্ষ্য।

## কী শিখবেন?

- Bell Curve এবং Standardization
- Z-Score দিয়ে unusual value identify করা
- Frequentist বনাম Bayesian interpretation
- Confidence score build ও validate করা
- Model calibration কেন accuracy থেকে আলাদা

## কীভাবে Run করবেন?

```powershell
cd "week 3\Class 2 Project"
python -m pip install jupyter numpy pandas scipy matplotlib seaborn scikit-learn
python -m jupyter notebook week3_class7_project.ipynb
```

## Main File

- `week3_class7_project.ipynb` — probability theory, visual experiment এবং confidence-system demo

## Validation Checklist

- Distribution-এর area/probability valid range-এ থাকে
- Standardized data-এর mean প্রায় 0 এবং standard deviation প্রায় 1
- Confidence value 0–1 range-এর বাইরে যায় না
- Calibration chart-এর predicted confidence ও observed outcome compare করা হয়
