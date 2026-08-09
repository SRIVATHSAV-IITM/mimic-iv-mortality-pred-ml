# MIMIC-IV ICU Mortality Prediction

Beginner-to-intermediate healthcare machine-learning project using a clear
24-hour landmark design. The project compares interpretable classical models
with a small optional PyTorch neural network while preventing patient leakage.

> Educational retrospective study only. This repository is not a medical
> device or clinical decision-support system.

## Clinical question

Among adults who are alive and still in the ICU 24 hours after ICU admission,
can information recorded during those first 24 hours predict **in-hospital
death after hour 24 and no later than hour 48**?

This wording is intentional. It is not general "mortality within 48 hours":
patients who die or leave the ICU before the 24-hour prediction landmark are
not part of this cohort.

## What this project teaches

- MIMIC-IV cohort construction from raw `hosp` and `icu` tables
- Landmark prediction and first-24-hour feature extraction
- Patient-level train/validation/test separation
- Missing-data handling learned from training data only
- Logistic Regression, Random Forest and XGBoost baselines
- Model and decision-threshold selection using validation PR-AUC/F1
- A compact two-hidden-layer PyTorch MLP
- ROC-AUC, PR-AUC, sensitivity, specificity, F1 and calibration
- Bootstrap confidence intervals
- Permutation importance and optional SHAP for tree models
- Saved models and batch prediction scripts
- Synthetic smoke tests, unit tests and GitHub Actions

## Dataset and access

The project targets [MIMIC-IV v3.1 on PhysioNet](https://physionet.org/content/mimiciv/).
MIMIC-IV is controlled-access. Complete the required credentialing and data-use
agreement before downloading it. Do not commit raw data, derived patient-level
tables, trained models or patient-level predictions to a public repository.

Expected local directories:

```text
/path/to/mimiciv/
|-- hosp/
|   |-- admissions.csv.gz
|   |-- patients.csv.gz
|   `-- labevents.csv.gz
`-- icu/
    |-- icustays.csv.gz
    `-- chartevents.csv.gz
```

## Project structure

```text
.
|-- src/
|   |-- common.py          # splitting, thresholds and metrics
|   |-- preprocess.py      # raw MIMIC-IV to named feature table
|   |-- train.py           # classical model comparison
|   |-- train_dl.py        # optional beginner-friendly PyTorch MLP
|   |-- predict.py         # classical inference
|   |-- predict_dl.py      # MLP inference
|   `-- explain.py         # permutation importance and optional SHAP
|-- scripts/
|   `-- generate_synthetic_data.py
|-- tests/
|-- data/README.md
|-- MODEL_CARD.md
|-- requirements*.txt
`-- .github/workflows/ci.yml
```

## 1. Installation

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the optional DL track:

```bash
pip install -r requirements-dl.txt
```

For SHAP:

```bash
pip install -r requirements-explain.txt
```

## 2. Verify the installation without MIMIC-IV

The synthetic generator exists only to test the software. Synthetic results
must not be reported as clinical performance.

```bash
python scripts/generate_synthetic_data.py
python src/train.py --data data/synthetic_features.csv --bootstrap-repetitions 20
```

## 3. Build the real feature table

```bash
python src/preprocess.py \
  --mimic-hosp-dir /path/to/mimiciv/hosp \
  --mimic-icu-dir /path/to/mimiciv/icu
```

The output includes one row per first ICU stay in a hospital admission. It
contains demographics plus mean, minimum, maximum and measurement count for a
small set of named vitals and laboratory measurements. Values outside broad
plausibility ranges are removed. See `outputs/data_dictionary.csv` after
preprocessing.

The preprocessing step assigns whole patients to approximately 70% training,
15% validation and 15% test partitions. A patient can never occur in more than
one partition.

## 4. Train classical models

```bash
python src/train.py --data data/mimic_icu_features.csv
```

The script:

1. Removes highly missing features using the training partition only.
2. Fits imputation, scaling and encoding using training data only.
3. Trains Logistic Regression, Random Forest and XGBoost.
4. Selects the model with the highest validation PR-AUC.
5. Chooses an F1-oriented threshold using validation data.
6. Evaluates the untouched test set once.

Generated local artifacts:

- `models/best_classical_model.joblib`
- `outputs/model_comparison.csv`
- `outputs/metrics.json`
- `outputs/test_predictions.csv`
- ROC, precision-recall, calibration and confusion-matrix figures

## 5. Train the beginner-friendly DL model

```bash
python src/train_dl.py --data data/mimic_icu_features.csv
```

The DL model is deliberately modest:

```text
processed features -> Dense(64) -> ReLU -> Dropout
                   -> Dense(32) -> ReLU -> Dropout
                   -> mortality logit
```

It uses weighted binary cross-entropy, mini-batches and early stopping on
validation PR-AUC. This teaches core PyTorch concepts without pretending that a
more complicated architecture is automatically better for tabular data.

Always compare `outputs/dl_metrics.json` with the classical model results. The
neural network should not be called superior unless the held-out results support
that claim.

## 6. Explain the classical model

Permutation importance works for every selected classical model:

```bash
python src/explain.py
```

Optional SHAP for Random Forest or XGBoost:

```bash
python src/explain.py --shap
```

Explanations describe model associations, not causal medical effects.

## 7. Predict on another compatible feature table

```bash
python src/predict.py \
  --input data/new_feature_table.csv \
  --output outputs/new_predictions.csv
```

For the MLP:

```bash
python src/predict_dl.py --input data/new_feature_table.csv
```

## 8. Run quality checks

```bash
pip install -r requirements-dev.txt
ruff check src scripts tests
pytest
```

## Variables

The intentionally small feature set includes:

- Age, gender, broad race group, insurance and admission type
- Heart rate, blood pressure, respiratory rate, oxygen saturation and temperature
- Albumin, anion gap, bicarbonate, bilirubin, creatinine, glucose, potassium,
  sodium, BUN, hematocrit, hemoglobin, INR, platelets and white-cell count

Measurement counts may partly reflect care processes. Demographic and insurance
variables may encode structural inequities. Subgroup evaluation and cautious
interpretation are required before any research conclusion.

## Reporting results honestly

This public repository intentionally contains no claimed MIMIC-IV result because
the controlled dataset is not available in CI. After running locally, report:

- Cohort size and mortality prevalence
- Exact MIMIC-IV version
- Patient-level split sizes
- Validation model comparison
- Test PR-AUC and ROC-AUC with 95% intervals
- Sensitivity, specificity, F1 and threshold
- Calibration/Brier score
- Classical-versus-MLP comparison
- Subgroup results and limitations

Do not headline accuracy for an imbalanced mortality outcome.

## Limitations

- Single-center retrospective data
- A narrow 24-to-48-hour mortality window
- Simple summary statistics discard within-day trajectories
- Missingness and measurement intensity can encode clinician behaviour
- No external or prospective validation
- No evidence that predictions improve patient outcomes

See [MODEL_CARD.md](MODEL_CARD.md) for intended use and safety boundaries.
