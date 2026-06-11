"""Train an XGBoost model for ICU mortality prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


TARGET_CANDIDATES = [
    "mortality_48h",
    "48_hour_mortality_flag",
    "mortality",
    "hospital_expire_flag",
]

ID_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "icustay_id",
    "admission_id",
    "patient_id",
}


def find_target_column(dataframe: pd.DataFrame) -> str:
    for column in TARGET_CANDIDATES:
        if column in dataframe.columns:
            return column
    raise ValueError(
        "No supported target column found. Expected one of: "
        + ", ".join(TARGET_CANDIDATES)
    )


def clean_binary_target(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "t": 1,
        "yes": 1,
        "y": 1,
        "1": 1,
        "false": 0,
        "f": 0,
        "no": 0,
        "n": 0,
        "0": 0,
    }
    cleaned = normalized.map(mapping)
    if cleaned.isna().any():
        cleaned = pd.to_numeric(series, errors="coerce")

    if cleaned.isna().any():
        raise ValueError("Target column contains values that cannot be parsed as binary.")

    return cleaned.astype(int)


def split_by_subject(
    features: pd.DataFrame,
    target: pd.Series,
    subject_ids: pd.Series | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if subject_ids is None:
        stratify = target if target.value_counts().min() >= 2 else None
        return train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    subject_frame = pd.DataFrame({"subject_id": subject_ids, "target": target})
    subject_labels = subject_frame.groupby("subject_id")["target"].max()
    stratify = subject_labels if subject_labels.value_counts().min() >= 2 else None
    train_subjects, test_subjects = train_test_split(
        subject_labels.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_mask = subject_ids.isin(train_subjects)
    test_mask = subject_ids.isin(test_subjects)
    return features[train_mask], features[test_mask], target[train_mask], target[test_mask]


def build_pipeline(features: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=4,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "average_precision_pr_auc": average_precision_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True,
        ),
    }

    if y_test.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)

    return metrics


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_figures(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    importance: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions = model.predict(x_test)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title("ICU Mortality Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    top_features = importance.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_features.png", dpi=160)
    plt.close()


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    figures_dir = Path(args.figures_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(data_path)
    target_column = find_target_column(dataframe)
    target = clean_binary_target(dataframe[target_column])

    missing_fraction = dataframe.isna().mean()
    retained_columns = missing_fraction[missing_fraction <= args.missingness_threshold].index
    dataframe = dataframe[retained_columns]

    target = target.loc[dataframe.index]
    drop_columns = [column for column in ID_COLUMNS | {target_column} if column in dataframe.columns]
    features = dataframe.drop(columns=drop_columns)
    subject_ids = dataframe["subject_id"] if "subject_id" in dataframe.columns else None

    x_train, x_test, y_train, y_test = split_by_subject(
        features,
        target,
        subject_ids,
        args.test_size,
        args.random_state,
    )

    model = build_pipeline(x_train, y_train)
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)
    importance = get_feature_importance(model)
    save_figures(model, x_test, y_test, importance, figures_dir)

    bundle = {
        "model": model,
        "target_column": target_column,
        "feature_columns": list(features.columns),
        "identifier_columns": sorted(ID_COLUMNS),
        "metrics": metrics,
    }
    joblib.dump(bundle, models_dir / "xgboost_mortality_model.joblib")

    with (outputs_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    importance.to_csv(outputs_dir / "feature_importance.csv", index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['average_precision_pr_auc']:.4f}")
    print(f"Model saved to: {models_dir / 'xgboost_mortality_model.joblib'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ICU mortality prediction model.")
    parser.add_argument("--data", default="data/mimic_icu_features.csv", help="Input feature CSV.")
    parser.add_argument("--models-dir", default="models", help="Directory for model artifacts.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for metrics outputs.")
    parser.add_argument("--figures-dir", default="reports/figures", help="Directory for figures.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--missingness-threshold",
        type=float,
        default=0.2,
        help="Drop columns with missing fraction above this threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
