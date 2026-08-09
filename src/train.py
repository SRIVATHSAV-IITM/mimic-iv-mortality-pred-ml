"""Train and compare beginner-friendly ICU mortality classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    TARGET_COLUMN,
    bootstrap_intervals,
    choose_f1_threshold,
    classification_metrics,
    select_training_features,
    split_modeling_frame,
)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create train-fitted numeric and categorical preprocessing."""
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
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
        ],
        remainder="drop",
    )


def build_models(
    features: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict[str, Pipeline]:
    """Return interpretable baselines and one boosted-tree model."""
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    estimators: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        estimators["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=4,
        )
    except ImportError:
        print("xgboost is not installed; training the two scikit-learn baselines only.")

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(features)),
                ("classifier", estimator),
            ]
        )
        for name, estimator in estimators.items()
    }


def save_evaluation_figures(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=[0, 1])
    plt.title("Test-set confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("Test-set ROC curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, probabilities)
    plt.title("Test-set precision-recall curve")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    observed, predicted = calibration_curve(y_test, probabilities, n_bins=8, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(predicted, observed, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed mortality fraction")
    plt.title("Test-set calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=160)
    plt.close()


def train(args: argparse.Namespace) -> None:
    dataframe = pd.read_csv(args.data)
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Expected target column {TARGET_COLUMN!r}. Run src/preprocess.py first."
        )

    split = split_modeling_frame(dataframe, TARGET_COLUMN)
    selected_features = select_training_features(
        split.x_train,
        maximum_missing_fraction=args.maximum_missing_fraction,
    )
    x_train = split.x_train[selected_features]
    x_validation = split.x_validation.reindex(columns=selected_features)
    x_test = split.x_test.reindex(columns=selected_features)

    comparison_rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}

    for name, model in build_models(x_train, split.y_train, args.random_state).items():
        print(f"Training {name}...")
        model.fit(x_train, split.y_train)
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        threshold = choose_f1_threshold(split.y_validation, validation_probabilities)
        validation_metrics = classification_metrics(
            split.y_validation,
            validation_probabilities,
            threshold,
        )
        comparison_rows.append(
            {
                "model": name,
                "validation_pr_auc": float(
                    average_precision_score(split.y_validation, validation_probabilities)
                ),
                "validation_roc_auc": float(validation_metrics.get("roc_auc", np.nan)),
                "validation_f1": float(validation_metrics["f1"]),
                "selected_threshold": threshold,
            }
        )
        fitted_models[name] = model
        thresholds[name] = threshold

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "validation_pr_auc", ascending=False
    )
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted_models[best_name]
    best_threshold = thresholds[best_name]
    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    test_metrics = classification_metrics(split.y_test, test_probabilities, best_threshold)
    confidence_intervals = bootstrap_intervals(
        split.y_test,
        test_probabilities,
        best_threshold,
        repetitions=args.bootstrap_repetitions,
        random_state=args.random_state,
    )

    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    figures_dir = Path(args.figures_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "target_column": TARGET_COLUMN,
        "feature_columns": selected_features,
        "threshold": best_threshold,
        "outcome_definition": (
            "In-hospital death after the 24-hour ICU landmark and no later than "
            "48 hours after ICU admission."
        ),
    }
    joblib.dump(bundle, models_dir / "best_classical_model.joblib")

    report = {
        "best_model": best_name,
        "selection_metric": "validation_pr_auc",
        "selected_threshold": best_threshold,
        "selected_feature_count": len(selected_features),
        "split_sizes": {
            "train": len(x_train),
            "validation": len(x_validation),
            "test": len(x_test),
        },
        "test_metrics": test_metrics,
        "test_95_percent_bootstrap_intervals": confidence_intervals,
    }
    with (outputs_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    comparison.to_csv(outputs_dir / "model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "probability": test_probabilities,
            "prediction": (test_probabilities >= best_threshold).astype(int),
            "observed": split.y_test.to_numpy(),
        }
    ).to_csv(outputs_dir / "test_predictions.csv", index=False)
    save_evaluation_figures(
        split.y_test,
        test_probabilities,
        best_threshold,
        figures_dir,
    )

    print(comparison.to_string(index=False))
    print(f"Best model: {best_name}")
    print(f"Test PR-AUC: {test_metrics['pr_auc_average_precision']:.4f}")
    print(f"Test ROC-AUC: {test_metrics.get('roc_auc', float('nan')):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mimic_icu_features.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--maximum-missing-fraction", type=float, default=0.80)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
