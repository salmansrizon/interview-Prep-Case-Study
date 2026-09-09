# Industrial Equipment Success Score Predictor

TensorFlow/Keras এবং Streamlit দিয়ে তৈরি একটি portfolio-oriented educational project। এটি synthetic industrial sensor ও maintenance data থেকে equipment-এর `success_score` (0–100) predict করার complete flow দেখায়:

```text
Synthetic data → Validation/preprocessing → Train/validation/test split
               → Keras regression model → Evaluation/report → Streamlit UI
```

## কী শিখবেন

- Reproducible synthetic tabular data তৈরি করা
- Numeric feature impute/scale এবং categorical feature one-hot encode করা
- Keras `Sequential` regression model build ও train করা
- Early stopping, model checkpoint এবং learning-rate reduction ব্যবহার করা
- Saved preprocessor ও model দিয়ে নতুন input predict করা
- Metric, residual এবং feature importance দিয়ে result validate করা

> এটি production deployment claim নয়। Synthetic data educational; real deployment-এর আগে domain validation, security, monitoring, drift detection এবং privacy controls দরকার।

## Data Contract

Raw dataset-এ ১৫টি column থাকে:

| Group | Count | Details |
| --- | ---: | --- |
| Identifier | 1 | `equipment_id` — model input নয় |
| Numeric predictors | 10 | Temperature, vibration, pressure, power, runtime, maintenance, errors, oil quality, load, ambient temperature |
| Categorical predictors | 3 | Equipment type, manufacturer, facility |
| Target | 1 | `success_score` |

অর্থাৎ raw predictor ১৩টি। Current fixed category catalog one-hot encoding-এর পরে model input width ২৬টি হয় (১০ numeric + ১৬ one-hot columns)। Training code processed array থেকে input width infer করে; নতুন category catalog হলে encoded width বদলাতে পারে।

Target-ও `StandardScaler` দিয়ে scale করা হয়। তাই model-এর raw prediction original 0–100 score নয়—user-facing result-এর আগে saved target scaler দিয়ে inverse-transform করতে হবে।

## Project Structure

```text
Class 2 Project/
├── config.yaml                # Paths, seed, model/training settings
├── run_pipeline.py            # End-to-end CLI orchestrator
├── src/
│   ├── data/                  # Generator, loader, preprocessing
│   ├── models/                # Builder, trainer, evaluator
│   └── utils/                 # Logging and persistence helpers
├── app/
│   ├── main.py                # Streamlit entry point
│   ├── components/            # Shared UI pieces
│   └── pages/                 # Home, explorer, training, prediction, analytics
├── notebooks/eda.ipynb        # Data assumptions and EDA
├── data/
│   ├── raw/                   # Generated CSV
│   ├── processed/             # Arrays, feature names, preprocessor
│   └── models/                # `.keras` models and training metadata
├── artifacts/                 # Evaluation plots/reports and pipeline manifest
├── logs/                      # Application/training logs
└── tests/                     # Generator, preprocessing and model tests
```

## Quick Start

Run commands from this `Class 2 Project` directory. If TensorFlow is unavailable for your newest Python release, use Python 3.13 (or another version supported by the TensorFlow release pip selects).

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

If installation fails with a Windows path-length error, clone/move the repository to a shorter path or create the virtual environment in a short path.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Recommended: Run the Complete Pipeline

```powershell
python run_pipeline.py
```

For a faster learning/smoke run:

```powershell
python run_pipeline.py --epochs 5
```

The orchestrator generates data, preprocesses it, trains the model, evaluates it and writes `artifacts/pipeline_manifest.json`.

### Run Individual Stages

```powershell
python -m src.data.generator
python -m src.data.preprocessor --input data/raw/equipment_data.csv
python -m src.models.trainer --epochs 20
python -m src.models.evaluator
```

Stage commands are useful for debugging, but they have dependencies: preprocessing needs raw data, training needs processed arrays, and evaluation needs saved model artifacts.

### Launch the App

```powershell
python -m streamlit run app/main.py
```

Open the local URL printed by Streamlit. Generate/train the pipeline first if the prediction and analytics pages report missing artifacts.

## Notebook Workflow

1. Start with `../Class 2 Lecture/Class14_TensorFlow_Keras_Lab.ipynb` for TensorFlow/Keras concepts and a small in-memory training flow.
2. Open `notebooks/eda.ipynb` for data distributions, quality checks, target relationships and encoded feature inspection.
3. Run the CLI pipeline only after the notebook assumptions make sense.
4. Use the Streamlit app to inspect the saved workflow interactively.

Notebook source is intentionally stored without execution outputs. Select the project environment as the Jupyter kernel and run cells top-to-bottom.

## Expected Artifacts

| Path | Purpose |
| --- | --- |
| `data/raw/equipment_data.csv` | Reproducible synthetic source data |
| `data/processed/X_*.npy`, `y_*.npy` | Train/validation/test arrays |
| `data/processed/preprocessor.joblib` | Numeric/categorical transforms and target scaler |
| `data/processed/feature_names.csv` | Encoded model-input names |
| `data/models/best_model.keras` | Best validation checkpoint |
| `data/models/final_model.keras` | Final restored/best-weight model |
| `data/models/training_history.json` | Loss and metric history |
| `artifacts/evaluation_report.json` | Evaluation metrics |
| `artifacts/pipeline_manifest.json` | End-to-end run summary |

## Validation

```powershell
python -m pytest -q
python -m compileall run_pipeline.py src app tests
```

Before trusting a result, also verify:

- Train, validation and test sets are distinct.
- Model input shape equals processed feature width.
- Validation loss does not diverge while training loss keeps falling.
- Predictions are inverse-transformed before interpreting the 0–100 score.
- MAE is reported with its scale clearly stated.
- Save/load round-trip produces materially identical predictions.

## Troubleshooting

- **`ModuleNotFoundError`:** use the virtual environment’s Python (`.\.venv\Scripts\python`) and reinstall requirements.
- **TensorFlow wheel not found:** use a supported Python version such as 3.13 and recreate the environment.
- **Windows filename/path too long:** use a shorter repository/environment path; partial installs should be recreated cleanly.
- **Missing raw/processed/model files:** run the earlier pipeline stage or use `python run_pipeline.py`.
- **Input shape mismatch:** inspect `data/processed/feature_names.csv`; do not assume a hard-coded width.
- **Prediction outside expected interpretation:** confirm target inverse-transform is applied.
- **Streamlit page error:** launch from the project root and confirm required artifacts exist.

## Limitations

- The target is generated from known synthetic rules, so strong performance does not prove real-world generalization.
- Category options and feature ranges are fixed by the generator.
- Correlation and feature importance do not establish causality.
- A neural network may be unnecessary for some tabular datasets; compare against linear and tree-based baselines.
- Docker and UI structure are learning scaffolds, not a complete production platform.

## Tech Stack

TensorFlow/Keras, pandas, NumPy, scikit-learn, Streamlit, Plotly, Matplotlib, Seaborn, Loguru, PyYAML, joblib and pytest.

## License

MIT
