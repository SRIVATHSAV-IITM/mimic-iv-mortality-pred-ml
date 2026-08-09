# Model Card

## Model purpose

Educational retrospective modelling of a narrowly defined ICU landmark task:
predict in-hospital death during hours 24-48 after ICU admission among adults
who are alive and still in the ICU at hour 24.

## Intended use

- Learning healthcare data science and beginner/intermediate ML/DL
- Reproducible portfolio experiments on authorized local MIMIC-IV data
- Comparing classical tabular models with a compact neural network

## Out-of-scope use

- Clinical diagnosis, triage or treatment decisions
- Real-time deployment
- Predictions for a hospital or population not independently validated
- Claims of causal relationships
- Use on data obtained without the required MIMIC-IV agreement

## Models

- Logistic Regression baseline
- Random Forest baseline
- XGBoost boosted-tree model
- Optional two-hidden-layer PyTorch MLP

The classical model with the best validation PR-AUC is selected. The MLP is
reported separately and must be compared on the same held-out test partition.

## Data

MIMIC-IV v3.1 `patients`, `admissions`, `icustays`, `chartevents` and
`labevents`. Raw and derived patient-level data are excluded from Git.

## Evaluation

- Patient-level train/validation/test split
- PR-AUC and ROC-AUC
- Sensitivity, specificity and F1
- Balanced accuracy
- Brier score and calibration curve
- Patient-level bootstrap confidence intervals

## Risks and limitations

- Retrospective single-center data may not generalize.
- Race, insurance and measurement frequency may encode healthcare inequities.
- Simple summaries omit detailed temporal trajectories.
- The selected threshold depends on the validation population and use case.
- Feature attribution reflects model behaviour, not treatment effects.

## Human oversight

Outputs are for educational analysis only. They require clinical, statistical
and ethical review before any research interpretation and must never be used as
standalone clinical advice.
