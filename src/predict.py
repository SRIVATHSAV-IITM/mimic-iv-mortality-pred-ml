"""Run predictions with a trained ICU mortality model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    target_column = bundle.get("target_column")
    identifier_columns = set(bundle.get("identifier_columns", []))

    dataframe = pd.read_csv(input_path)
    drop_columns = [column for column in identifier_columns if column in dataframe.columns]
    if target_column in dataframe.columns:
        drop_columns.append(target_column)

    features = dataframe.drop(columns=drop_columns)
    result = dataframe.copy()
    result["mortality_prediction"] = model.predict(features)

    if hasattr(model, "predict_proba"):
        result["mortality_probability"] = model.predict_proba(features)[:, 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ICU mortality risk.")
    parser.add_argument("--model", default="models/xgboost_mortality_model.joblib")
    parser.add_argument("--input", required=True, help="Input feature CSV.")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
