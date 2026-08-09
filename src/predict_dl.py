"""Generate predictions with the saved PyTorch tabular MLP."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train_dl import TabularMLP, predict_probabilities


def predict(args: argparse.Namespace) -> None:
    preprocessing = joblib.load(args.preprocessor)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    feature_columns = preprocessing["feature_columns"]
    dataframe = pd.read_csv(args.input)
    missing = sorted(set(feature_columns) - set(dataframe.columns))
    if missing and not args.allow_missing_columns:
        raise ValueError(f"Input is missing model features: {missing[:10]}")

    features = dataframe.reindex(columns=feature_columns)
    transformed = np.asarray(
        preprocessing["preprocessor"].transform(features),
        dtype=np.float32,
    )
    model = TabularMLP(
        input_features=int(checkpoint["input_features"]),
        hidden_features=int(checkpoint["hidden_features"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    probabilities = predict_probabilities(model, transformed, args.batch_size)
    threshold = float(checkpoint["threshold"])

    result = dataframe.copy()
    result["mortality_probability"] = probabilities
    result["mortality_prediction"] = (probabilities >= threshold).astype(int)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}; decision threshold={threshold:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/tabular_mlp.pt")
    parser.add_argument("--preprocessor", default="models/dl_preprocessor.joblib")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/dl_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-missing-columns", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
