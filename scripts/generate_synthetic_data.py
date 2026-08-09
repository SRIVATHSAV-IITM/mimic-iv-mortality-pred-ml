"""Generate fake data for installation checks only.

The generated values are not MIMIC-IV data and must never be reported as a
clinical experiment result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    age = rng.normal(64, 16, rows).clip(18, 91)
    heart_rate = rng.normal(88, 18, rows).clip(35, 180)
    systolic_bp = rng.normal(118, 24, rows).clip(55, 220)
    creatinine = rng.lognormal(mean=0.1, sigma=0.65, size=rows).clip(0.2, 12)
    oxygen_saturation = rng.normal(96, 3, rows).clip(70, 100)
    race = rng.choice(["WHITE", "BLACK", "ASIAN", "HISPANIC", "OTHER"], rows)
    gender = rng.choice(["F", "M"], rows)
    admission_type = rng.choice(["URGENT", "EMERGENCY", "ELECTIVE"], rows)

    log_odds = (
        -5.0
        + 0.035 * (age - 60)
        + 0.035 * (heart_rate - 85)
        - 0.020 * (systolic_bp - 110)
        + 0.45 * (creatinine - 1)
        - 0.12 * (oxygen_saturation - 95)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    outcome = rng.binomial(1, probability)
    # Guarantee enough positives for the small smoke-test partitions.
    if outcome.sum() < 20:
        outcome[np.argsort(probability)[-20:]] = 1

    frame = pd.DataFrame(
        {
            "subject_id": np.arange(10_000, 10_000 + rows),
            "hadm_id": np.arange(20_000, 20_000 + rows),
            "stay_id": np.arange(30_000, 30_000 + rows),
            "age": age,
            "gender": gender,
            "race": race,
            "insurance": rng.choice(["Medicare", "Medicaid", "Private"], rows),
            "admission_type": admission_type,
            "vital_heart_rate_mean": heart_rate,
            "vital_systolic_bp_mean": systolic_bp,
            "vital_oxygen_saturation_mean": oxygen_saturation,
            "lab_creatinine_mean": creatinine,
            "mortality_24_48h": outcome,
        }
    )
    missing_rows = rng.choice(rows, size=max(1, rows // 10), replace=False)
    frame.loc[missing_rows, "lab_creatinine_mean"] = np.nan

    indices = rng.permutation(rows)
    train_end = int(rows * 0.70)
    validation_end = int(rows * 0.85)
    frame["data_split"] = "test"
    frame.loc[indices[:train_end], "data_split"] = "train"
    frame.loc[indices[train_end:validation_end], "data_split"] = "validation"

    # Ensure each split has both classes for metrics and threshold selection.
    for split_name in ("train", "validation", "test"):
        split_indices = frame.index[frame["data_split"] == split_name]
        if frame.loc[split_indices, "mortality_24_48h"].sum() == 0:
            frame.loc[split_indices[-1], "mortality_24_48h"] = 1
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--output", default="data/synthetic_features.csv")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows, args.random_state).to_csv(output_path, index=False)
    print(f"Synthetic smoke-test data saved to {output_path}")


if __name__ == "__main__":
    main()
