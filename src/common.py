"""Shared data splitting and evaluation helpers.

The project intentionally keeps these helpers small and explicit so that a
beginner can follow the complete validation workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "mortality_24_48h"
SPLIT_COLUMN = "data_split"
ID_COLUMNS = {"subject_id", "hadm_id", "stay_id"}


@dataclass(frozen=True)
class DatasetSplit:
    """Container for a leakage-safe train/validation/test split."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def clean_binary_target(series: pd.Series) -> pd.Series:
    """Parse a target series into integers containing only zero and one."""
    if series.dtype == bool:
        cleaned = series.astype(int)
    else:
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

    if cleaned.isna().any() or not set(cleaned.unique()).issubset({0, 1}):
        raise ValueError("The mortality target must contain only binary values.")
    return cleaned.astype(int)


def _stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    return labels if len(counts) == 2 and counts.min() >= 2 else None


def assign_subject_splits(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign every subject to exactly one dataset split.

    Splitting subjects rather than rows prevents multiple ICU stays belonging
    to one patient from leaking across the model-development partitions.
    """
    if "subject_id" not in dataframe.columns:
        raise ValueError("A subject_id column is required for patient-level splitting.")
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1.")

    subject_labels = dataframe.groupby("subject_id")[target_column].max()
    development_subjects, test_subjects = train_test_split(
        subject_labels.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_or_none(subject_labels),
    )

    development_labels = subject_labels.loc[development_subjects]
    relative_validation_size = validation_size / (1.0 - test_size)
    train_subjects, validation_subjects = train_test_split(
        development_subjects,
        test_size=relative_validation_size,
        random_state=random_state + 1,
        stratify=_stratify_or_none(development_labels),
    )

    result = dataframe.copy()
    result[SPLIT_COLUMN] = ""
    result.loc[result["subject_id"].isin(train_subjects), SPLIT_COLUMN] = "train"
    result.loc[result["subject_id"].isin(validation_subjects), SPLIT_COLUMN] = "validation"
    result.loc[result["subject_id"].isin(test_subjects), SPLIT_COLUMN] = "test"
    if (result[SPLIT_COLUMN] == "").any():
        raise RuntimeError("Some subjects were not assigned to a split.")
    return result


def split_modeling_frame(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> DatasetSplit:
    """Create feature and target frames from a pre-split feature table."""
    required_splits = {"train", "validation", "test"}
    if SPLIT_COLUMN not in dataframe.columns:
        raise ValueError(f"Expected a {SPLIT_COLUMN!r} column.")
    present = set(dataframe[SPLIT_COLUMN].astype(str).str.lower().unique())
    if not required_splits.issubset(present):
        raise ValueError(f"Expected train, validation and test splits; found {sorted(present)}.")

    target = clean_binary_target(dataframe[target_column])
    drop_columns = [
        column
        for column in ID_COLUMNS | {SPLIT_COLUMN, target_column}
        if column in dataframe.columns
    ]
    features = dataframe.drop(columns=drop_columns)
    split_values = dataframe[SPLIT_COLUMN].astype(str).str.lower()

    return DatasetSplit(
        x_train=features.loc[split_values == "train"].copy(),
        x_validation=features.loc[split_values == "validation"].copy(),
        x_test=features.loc[split_values == "test"].copy(),
        y_train=target.loc[split_values == "train"].copy(),
        y_validation=target.loc[split_values == "validation"].copy(),
        y_test=target.loc[split_values == "test"].copy(),
    )


def select_training_features(
    x_train: pd.DataFrame,
    maximum_missing_fraction: float,
) -> list[str]:
    """Select features using training data only to prevent test leakage."""
    if not 0 <= maximum_missing_fraction < 1:
        raise ValueError("maximum_missing_fraction must be in [0, 1).")
    missing_fraction = x_train.isna().mean()
    selected = missing_fraction[missing_fraction <= maximum_missing_fraction].index.tolist()
    if not selected:
        raise ValueError("No features remain after applying the missingness threshold.")
    return selected


def choose_f1_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a probability threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    """Calculate discrimination, calibration and threshold metrics."""
    y_array = np.asarray(y_true, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    predictions = (probability_array >= threshold).astype(int)
    matrix = confusion_matrix(y_array, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0

    metrics: dict[str, float | list[list[int]]] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, predictions)),
        "precision": float(precision_score(y_array, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_array, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_array, predictions, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y_array, probability_array)),
        "brier_score": float(brier_score_loss(y_array, probability_array)),
        "confusion_matrix": matrix.tolist(),
    }
    if len(np.unique(y_array)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_array, probability_array))
    return metrics


def bootstrap_intervals(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    repetitions: int = 200,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """Return simple patient-level bootstrap 95% intervals for headline metrics."""
    y_array = np.asarray(y_true, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(random_state)
    collected: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc_average_precision": [],
        "f1": [],
        "recall_sensitivity": [],
        "specificity": [],
    }

    for _ in range(repetitions):
        indices = rng.integers(0, len(y_array), len(y_array))
        sampled_y = y_array[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_metrics = classification_metrics(
            sampled_y,
            probability_array[indices],
            threshold,
        )
        for name in collected:
            collected[name].append(float(sampled_metrics[name]))

    return {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for name, values in collected.items()
        if values
    }
