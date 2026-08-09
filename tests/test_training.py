from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import joblib
import pandas as pd

from scripts.generate_synthetic_data import generate
from train import train


def test_classical_training_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.csv"
    generate(300, 11).to_csv(data_path, index=False)
    args = Namespace(
        data=str(data_path),
        models_dir=str(tmp_path / "models"),
        outputs_dir=str(tmp_path / "outputs"),
        figures_dir=str(tmp_path / "figures"),
        maximum_missing_fraction=0.80,
        bootstrap_repetitions=10,
        random_state=11,
    )
    train(args)
    model_path = tmp_path / "models" / "best_classical_model.joblib"
    assert model_path.exists()
    bundle = joblib.load(model_path)
    assert bundle["model_name"] in {"logistic_regression", "random_forest", "xgboost"}
    assert 0 <= bundle["threshold"] <= 1
    metrics = pd.read_json(tmp_path / "outputs" / "metrics.json", typ="series")
    assert metrics["selection_metric"] == "validation_pr_auc"
