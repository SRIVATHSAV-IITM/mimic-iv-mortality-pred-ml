"""Export feature importances from a trained XGBoost mortality model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def export_feature_importance(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    model = bundle["model"]
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    importance = (
        pd.DataFrame(
            {
                "feature": preprocessor.get_feature_names_out(),
                "importance": classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    print(f"Feature importances saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model feature importances.")
    parser.add_argument("--model", default="models/xgboost_mortality_model.joblib")
    parser.add_argument("--output", default="outputs/feature_importance.csv")
    return parser.parse_args()


if __name__ == "__main__":
    export_feature_importance(parse_args())
