"""Train a small beginner-friendly PyTorch MLP on the same feature table."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (
    TARGET_COLUMN,
    bootstrap_intervals,
    choose_f1_threshold,
    classification_metrics,
    select_training_features,
    split_modeling_frame,
)
from train import build_preprocessor, save_evaluation_figures


class TabularMLP(nn.Module):
    """A compact two-hidden-layer network for tabular clinical features."""

    def __init__(self, input_features: int, hidden_features: int = 64, dropout: float = 0.25):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(_tensor(features)), batch_size=batch_size, shuffle=False)
    probabilities = []
    with torch.no_grad():
        for (batch,) in loader:
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probabilities)


def train_network(
    model: TabularMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
) -> tuple[TabularMLP, list[dict[str, float]]]:
    positives = max(float(y_train.sum()), 1.0)
    negatives = float(len(y_train) - y_train.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(_tensor(x_train), _tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_pr_auc = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_probabilities = predict_probabilities(model, x_validation, batch_size)
        validation_pr_auc = float(
            average_precision_score(y_validation, validation_probabilities)
        )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_pr_auc": validation_pr_auc,
            }
        )
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
            f"validation_pr_auc={validation_pr_auc:.4f}"
        )

        if validation_pr_auc > best_validation_pr_auc + 1e-4:
            best_validation_pr_auc = validation_pr_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model, history


def train(args: argparse.Namespace) -> None:
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    dataframe = pd.read_csv(args.data)
    split = split_modeling_frame(dataframe, TARGET_COLUMN)
    selected_features = select_training_features(
        split.x_train,
        maximum_missing_fraction=args.maximum_missing_fraction,
    )
    x_train_frame = split.x_train[selected_features]
    x_validation_frame = split.x_validation.reindex(columns=selected_features)
    x_test_frame = split.x_test.reindex(columns=selected_features)

    preprocessor = build_preprocessor(x_train_frame)
    x_train = np.asarray(preprocessor.fit_transform(x_train_frame), dtype=np.float32)
    x_validation = np.asarray(preprocessor.transform(x_validation_frame), dtype=np.float32)
    x_test = np.asarray(preprocessor.transform(x_test_frame), dtype=np.float32)

    model = TabularMLP(
        input_features=x_train.shape[1],
        hidden_features=args.hidden_features,
        dropout=args.dropout,
    )
    model, history = train_network(
        model,
        x_train,
        split.y_train.to_numpy(dtype=np.float32),
        x_validation,
        split.y_validation.to_numpy(dtype=np.float32),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    validation_probabilities = predict_probabilities(model, x_validation, args.batch_size)
    threshold = choose_f1_threshold(split.y_validation, validation_probabilities)
    test_probabilities = predict_probabilities(model, x_test, args.batch_size)
    test_metrics = classification_metrics(split.y_test, test_probabilities, threshold)
    confidence_intervals = bootstrap_intervals(
        split.y_test,
        test_probabilities,
        threshold,
        repetitions=args.bootstrap_repetitions,
        random_state=args.random_state,
    )

    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    figures_dir = Path(args.figures_dir) / "dl"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"preprocessor": preprocessor, "feature_columns": selected_features},
        models_dir / "dl_preprocessor.joblib",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_features": x_train.shape[1],
            "hidden_features": args.hidden_features,
            "dropout": args.dropout,
            "threshold": threshold,
            "outcome_definition": (
                "In-hospital death after the 24-hour ICU landmark and no later than "
                "48 hours after ICU admission."
            ),
        },
        models_dir / "tabular_mlp.pt",
    )

    report = {
        "model": "two_hidden_layer_tabular_mlp",
        "selection_metric": "validation_pr_auc",
        "selected_threshold": threshold,
        "selected_feature_count": len(selected_features),
        "transformed_feature_count": int(x_train.shape[1]),
        "epochs_completed": len(history),
        "test_metrics": test_metrics,
        "test_95_percent_bootstrap_intervals": confidence_intervals,
    }
    with (outputs_dir / "dl_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    pd.DataFrame(history).to_csv(outputs_dir / "dl_training_history.csv", index=False)

    history_frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_frame["epoch"], history_frame["training_loss"])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Weighted BCE loss")
    axes[1].plot(history_frame["epoch"], history_frame["validation_pr_auc"])
    axes[1].set(title="Validation PR-AUC", xlabel="Epoch", ylabel="PR-AUC")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_history.png", dpi=160)
    plt.close(fig)
    save_evaluation_figures(split.y_test, test_probabilities, threshold, figures_dir)

    print(f"Test PR-AUC: {test_metrics['pr_auc_average_precision']:.4f}")
    print(f"Test ROC-AUC: {test_metrics.get('roc_auc', float('nan')):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mimic_icu_features.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--maximum-missing-fraction", type=float, default=0.80)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
