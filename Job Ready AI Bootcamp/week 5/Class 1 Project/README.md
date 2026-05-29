# 🏠 House Price Predictor — Production-Grade ML Dashboard

A complete Streamlit application for predicting real estate values using Linear Regression,
with interactive educational modules covering the full ML pipeline.

## 📚 Learning Objectives

| Topic | Description | Page |
|-------|-------------|------|
| **Linear Regression** | Mathematical foundation, coefficients, prediction formula | 📐 Model Theory |
| **Gradient Descent** | Optimization algorithm with interactive visualization | 📉 Gradient Descent |
| **MSE & Metrics** | Model evaluation with MSE, RMSE, MAE, R², residuals | 📏 Model Evaluation |
| **Feature Scaling** | StandardScaler, MinMaxScaler, why scaling matters | ⚖️ Feature Scaling |
| **Live Prediction** | Real-time price estimation with explainable AI | 🔮 Predict Price |

## 🏗️ Project Structure

```
houseprice_predictor/
├── app.py                          # Main entry point
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── data/
│   └── housing_data.csv            # 5,000 synthetic real estate records
├── models/
│   ├── linear_regression.pkl       # Trained model
│   ├── scaler.pkl                  # Feature scaler
│   ├── feature_names.json          # Feature list
│   ├── feature_importance.csv      # Coefficients
│   └── model_results.json          # Evaluation metrics
├── utils/
│   ├── data_loader.py              # Data I/O & preprocessing
│   ├── model_utils.py              # Prediction engine & explainability
│   └── visualizations.py           # Chart utilities
└── pages/
    ├── 1_Explore_Data.py           # EDA: distributions, correlations
    ├── 2_Model_Theory.py           # Linear Regression math & coefficients
    ├── 3_Gradient_Descent.py       # Interactive GD optimization demo
    ├── 4_Model_Evaluation.py       # MSE, RMSE, MAE, R², residuals
    ├── 5_Feature_Scaling.py        # StandardScaler with before/after
    └── 6_Predict_Price.py          # Live prediction with explanations
```

## 🚀 Quick Start

```bash
cd houseprice_predictor
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 🎯 Features

### 🔮 Live Price Prediction
- Input 16 house features via interactive sliders
- Get instant price estimate with confidence range
- See **feature contribution breakdown** (explainable AI)
- Compare your inputs to market averages

### 📐 Educational Modules
- **Interactive coefficient demo** — adjust slope/intercept, watch MSE change
- **Gradient Descent simulator** — control learning rate, see convergence
- **Before/after scaling** — visual comparison of feature distributions
- **Residual analysis** — validate model assumptions

### 📊 Data Exploration
- 5,000 synthetic real estate records
- 16 features: income, rooms, age, location, amenities, school ratings
- Interactive correlation heatmap
- Distribution analysis with box plots

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.8754 |
| RMSE | $22,433 |
| MAE | $17,077 |
| MAPE | ~6.2% |

## 🛠️ Tech Stack

- **Streamlit** — Web app framework
- **Scikit-Learn** — Linear Regression, StandardScaler, train/test split
- **Pandas/NumPy** — Data manipulation
- **Matplotlib/Seaborn** — Static visualizations
- **Plotly** — Interactive charts

## 📝 Dataset Features

| Feature | Description | Range |
|---------|-------------|-------|
| `median_income` | Block group median income ($10K) | 1.5 - 15.0 |
| `house_age` | Age of house in years | 1 - 51 |
| `avg_rooms` | Average rooms per household | 2 - 10 |
| `avg_bedrooms` | Average bedrooms per household | 0.5 - 3 |
| `population` | Block group population | 100 - 35,000 |
| `avg_occupancy` | Average household occupancy | 0.5 - 7 |
| `latitude` / `longitude` | Geographic coordinates | CA region |
| `lot_size_sqft` | Lot size in square feet | 1,000 - 50,000 |
| `garage_spaces` | Number of garage spaces | 0 - 4 |
| `has_pool` | Swimming pool (0/1) | 0 or 1 |
| `has_basement` | Basement (0/1) | 0 or 1 |
| `distance_to_city` | Miles to nearest city center | 0 - 80 |
| `school_rating` | Local school rating (1-10) | 1 - 10 |
| `property_tax_rate` | Annual property tax rate (%) | 0.5 - 2.5 |
| `year_built` | Year house was constructed | 1973 - 2023 |

## 👥 Authors

Data Science Education Team
