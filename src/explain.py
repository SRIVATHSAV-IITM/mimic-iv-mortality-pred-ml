"""Explain the trained classical model with permutation importance and optional SHAP."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from common import TARGET_COLUMN, split_modeling_frame


def explain(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    dataframe = pd.read_csv(args.data)
    split = split_modeling_frame(dataframe, TARGET_COLUMN)
    x_test = split.x_test.reindex(columns=feature_columns)
    y_test = split.y_test
    if len(x_test) > args.maximum_rows:
        sampled_indices = x_test.sample(
            n=args.maximum_rows,
            random_state=args.random_state,
        ).index
        x_test = x_test.loc[sampled_indices]
        y_test = y_test.loc[sampled_indices]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=args.repeats,
        random_state=args.random_state,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "mean_pr_auc_decrease": result.importances_mean,
                "standard_deviation": result.importances_std,
            }
        )
        .sort_values("mean_pr_auc_decrease", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_dir / "permutation_importance.csv", index=False)

    top = importance.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["mean_pr_auc_decrease"], xerr=top["standard_deviation"])
    plt.xlabel("Decrease in PR-AUC after permutation")
    plt.title("Test-set permutation importance")
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_importance.png", dpi=160)
    plt.close()

    if args.shap:
        save_shap_summary(model, x_test, output_dir, args.shap_rows)

    print(f"Explanations saved to {output_dir}")


def save_shap_summary(
    model,
    features: pd.DataFrame,
    output_dir: Path,
    maximum_rows: int,
) -> None:
    """Save a SHAP summary for a fitted tree classifier."""
    try:
        import shap
    except ImportError as error:
        raise ImportError(
            "Install optional explanation dependencies with "
            "`pip install -r requirements-explain.txt`."
        ) from error

    classifier = model.named_steps["classifier"]
    classifier_name = classifier.__class__.__name__.lower()
    if "forest" not in classifier_name and "xgb" not in classifier_name:
        raise ValueError("Optional SHAP output currently supports Random Forest and XGBoost.")

    preprocessor = model.named_steps["preprocessor"]
    sample = features.head(maximum_rows)
    transformed = preprocessor.transform(sample)
    transformed_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    plt.figure()
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=transformed_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best_classical_model.joblib")
    parser.add_argument("--data", default="data/mimic_icu_features.csv")
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--maximum-rows", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--shap", action="store_true")
    parser.add_argument("--shap-rows", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    explain(parse_args())
