from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    TARGET_COLUMN,
    assign_subject_splits,
    choose_f1_threshold,
    classification_metrics,
    select_training_features,
)


def test_subject_splits_are_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(100), 2),
            TARGET_COLUMN: np.tile([0, 1], 100),
        }
    )
    result = assign_subject_splits(frame, random_state=7)
    subject_split_counts = result.groupby("subject_id")["data_split"].nunique()
    assert subject_split_counts.max() == 1
    assert set(result["data_split"]) == {"train", "validation", "test"}


def test_feature_selection_uses_missing_fraction() -> None:
    frame = pd.DataFrame(
        {
            "keep": [1.0, 2.0, np.nan, 4.0],
            "drop": [np.nan, np.nan, np.nan, 1.0],
        }
    )
    assert select_training_features(frame, 0.50) == ["keep"]


def test_threshold_and_metrics() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    threshold = choose_f1_threshold(target, probabilities)
    metrics = classification_metrics(target, probabilities, threshold)
    assert 0 <= threshold <= 1
    assert metrics["f1"] == 1.0
    assert metrics["specificity"] == 1.0
