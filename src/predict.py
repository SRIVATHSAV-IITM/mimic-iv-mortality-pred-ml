"""Generate predictions with the saved classical mortality model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = float(bundle["threshold"])

    dataframe = pd.read_csv(args.input)
    missing = sorted(set(feature_columns) - set(dataframe.columns))
    if missing and not args.allow_missing_columns:
        raise ValueError(
            "Input is missing model features. Use --allow-missing-columns only when "
            f"missing values should be imputed. First missing columns: {missing[:10]}"
        )
    features = dataframe.reindex(columns=feature_columns)
    probabilities = model.predict_proba(features)[:, 1]

    result = dataframe.copy()
    result["mortality_probability"] = probabilities
    result["mortality_prediction"] = (probabilities >= threshold).astype(int)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}; decision threshold={threshold:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best_classical_model.joblib")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--allow-missing-columns", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
