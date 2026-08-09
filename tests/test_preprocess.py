from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import TARGET_COLUMN
from preprocess import build_base_cohort, build_feature_table, clean_race


def _write_tables(root: Path) -> tuple[Path, Path]:
    hosp = root / "hosp"
    icu = root / "icu"
    hosp.mkdir()
    icu.mkdir()

    pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "hadm_id": [11, 22, 33, 44],
            "stay_id": [111, 222, 333, 444],
            "intime": ["2120-01-01 00:00:00"] * 4,
            "outtime": [
                "2120-01-03 00:00:00",
                "2120-01-01 12:00:00",
                "2120-01-04 00:00:00",
                "2120-01-04 00:00:00",
            ],
            "los": [2.0, 0.5, 3.0, 3.0],
        }
    ).to_csv(icu / "icustays.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "hadm_id": [11, 22, 33, 44],
            "admittime": ["2119-12-31 20:00:00"] * 4,
            "dischtime": ["2120-01-05 00:00:00"] * 4,
            "deathtime": [
                "2120-01-02 06:00:00",
                None,
                "2120-01-04 00:00:00",
                None,
            ],
            "hospital_expire_flag": [1, 0, 1, 0],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", None],
            "insurance": ["Medicare"] * 4,
            "admission_type": ["EMERGENCY"] * 4,
        }
    ).to_csv(hosp / "admissions.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "gender": ["F", "M", "F", "M"],
            "anchor_age": [60, 61, 62, 63],
            "anchor_year": [2120] * 4,
        }
    ).to_csv(hosp / "patients.csv", index=False)
    return hosp, icu


def test_landmark_cohort_and_target(tmp_path: Path) -> None:
    hosp, icu = _write_tables(tmp_path)
    cohort = build_base_cohort(hosp, icu)
    # Subject 2 leaves before 24 hours and is excluded.
    assert set(cohort["subject_id"]) == {1, 3, 4}
    targets = cohort.set_index("subject_id")[TARGET_COLUMN].to_dict()
    assert targets == {1: 1, 3: 0, 4: 0}


def test_clean_race_groups_values() -> None:
    assert clean_race("HISPANIC/LATINO") == "HISPANIC"
    assert clean_race("DECLINED TO ANSWER") == "UNKNOWN"
    assert clean_race("AMERICAN INDIAN") == "OTHER"


def test_event_features_use_only_first_24_hours(tmp_path: Path) -> None:
    hosp, icu = _write_tables(tmp_path)
    cohort = build_base_cohort(hosp, icu)
    pd.DataFrame(
        {
            "stay_id": [111, 111, 333, 444],
            "charttime": [
                "2120-01-01 12:00:00",
                "2120-01-02 12:00:00",  # after the landmark and excluded
                "2120-01-01 10:00:00",
                "2120-01-01 08:00:00",
            ],
            "itemid": [220045, 220045, 223761, 220277],
            "valuenum": [100.0, 200.0, 98.6, 97.0],
        }
    ).to_csv(icu / "chartevents.csv", index=False)
    pd.DataFrame(
        {
            "hadm_id": [11, 33, 44],
            "charttime": ["2120-01-01 06:00:00"] * 3,
            "itemid": [50912, 50912, 50912],
            "valuenum": [1.2, 2.5, 0.8],
        }
    ).to_csv(hosp / "labevents.csv", index=False)

    features = build_feature_table(cohort, hosp, icu, chunk_size=2)
    subject_one = features.set_index("subject_id").loc[1]
    assert subject_one["vital_heart_rate_mean"] == 100.0
    assert subject_one["vital_heart_rate_count"] == 1.0
    assert subject_one["lab_creatinine_mean"] == 1.2
    # Fahrenheit is converted to Celsius and receives a readable name.
    subject_three = features.set_index("subject_id").loc[3]
    assert abs(subject_three["vital_temperature_c_mean"] - 37.0) < 1e-6
