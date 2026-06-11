# MIMIC-IV ICU Mortality Prediction

Machine learning project for predicting 48-hour ICU mortality using structured clinical features from the first 24 hours of an ICU stay.

The project is designed for healthcare analytics and clinical risk prediction workflows using MIMIC-IV style data. It trains an XGBoost classifier, evaluates mortality prediction performance, and exports reusable model and report artifacts.

## Project Highlights

- Predicts short-term ICU mortality from vitals, labs, demographics, and admission-level features.
- Builds a reusable feature table from raw local MIMIC-IV `hosp` and `icu` tables.
- Uses subject-level train-test splitting when patient identifiers are available.
- Handles missing values, categorical variables, and class imbalance.
- Reports Accuracy, ROC-AUC, PR-AUC, Precision, Recall, F1-score, and confusion matrix.
- Saves trained model and feature-importance outputs for later use.

## Folder Structure

```text
mimic-iv-mortality-pred-ml/
|-- data/
|   `-- README.md
|-- models/
|   `-- .gitkeep
|-- notebooks/
|   `-- README.md
|-- outputs/
|   `-- .gitkeep
|-- reports/
|   `-- figures/
|       `-- .gitkeep
|-- src/
|   |-- __init__.py
|   |-- feature_importance.py
|   |-- predict.py
|   |-- preprocess.py
|   `-- train.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Data

MIMIC-IV is a controlled-access clinical dataset. This repository does not include patient data. Prepare a de-identified feature table locally and place it at:

```text
data/mimic_icu_features.csv
```

You can generate this feature table from local MIMIC-IV directories:

```bash
python src/preprocess.py --mimic-hosp-dir /path/to/mimiciv/hosp --mimic-icu-dir /path/to/mimiciv/icu
```

The preprocessing script keeps the reusable MIMIC-IV logic from the assignment notebook:

- builds an ICU cohort with stays of at least 24 hours
- labels mortality between 24 and 48 hours from ICU admission
- adds age, gender, race, and insurance features
- extracts first-24-hour routine vital-sign and lab summaries
- creates a subject-level train/test split in `data_split`

The table should include one binary target column. Supported target names:

- `mortality_48h`
- `48_hour_mortality_flag`
- `mortality`
- `hospital_expire_flag`

Optional identifier columns such as `subject_id`, `hadm_id`, and `stay_id` are automatically excluded from model features. If `data_split` or `train` is present, the saved split is reused; otherwise, `subject_id` is used for a subject-level train-test split.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --data data/mimic_icu_features.csv
```

Training creates:

- `models/xgboost_mortality_model.joblib`
- `outputs/metrics.json`
- `outputs/feature_importance.csv`
- `reports/figures/confusion_matrix.png`
- `reports/figures/top_features.png`

## Predict

```bash
python src/predict.py --model models/xgboost_mortality_model.joblib --input data/mimic_icu_features.csv --output outputs/predictions.csv
```

## Feature Importance

```bash
python src/feature_importance.py --model models/xgboost_mortality_model.joblib
```

## Model Selection

XGBoost is used as the final model because it performs well on structured tabular healthcare data, captures non-linear relationships between labs, vitals, demographics, and admission features, and supports imbalance-aware training through `scale_pos_weight`.

## Metrics

The training script evaluates:

- Accuracy
- ROC-AUC
- PR-AUC / Average Precision
- Precision
- Recall
- F1-score
- Confusion matrix

## Disclaimer

This project is for educational and portfolio purposes only. It is not a clinical decision support system and should not be used for medical diagnosis or treatment decisions.
