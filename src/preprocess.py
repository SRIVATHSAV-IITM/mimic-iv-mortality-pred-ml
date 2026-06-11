"""Build ICU mortality modeling features from raw MIMIC-IV tables.

This module keeps the reusable MIMIC-IV workflow from the assignment notebook
and removes exploratory notebook-only analysis such as Iris regression, ad-hoc
plots, and display calls.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):
        return iterable


TARGET_COLUMN = "48_hour_mortality_flag"
DEFAULT_OUTPUT = Path("data/mimic_icu_features.csv")


def csv_path(parent: Path, table_name: str) -> Path:
    """Return the CSV path for a MIMIC table, accepting compressed files too."""
    candidates = [
        parent / f"{table_name}.csv",
        parent / f"{table_name}.csv.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {table_name}.csv or {table_name}.csv.gz in {parent}")


def to_datetime_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    datetime_columns = [column for column in dataframe.columns if column.endswith("time")]
    for column in datetime_columns:
        dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce")
    return dataframe


def clean_race(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).upper()
    replacements = [
        (r"ASIAN.*", "ASIAN"),
        (r"WHITE.*", "WHITE"),
        (r"BLACK.*", "BLACK"),
        (r"HISPANIC.*", "HISPANIC"),
        (r"UNABLE.*", "UNKNOWN"),
        (r"PATIENT.*", "UNKNOWN"),
        (r"PORT.*", "SOUTH AMERICAN"),
        (r"MULTIPLE.*", "OTHER"),
        (r"NATIVE.*", "OTHER"),
        (r"SOUTH AMERICAN.*", "OTHER"),
        (r"AMERICAN.*", "OTHER"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def extract_number(value: object) -> float:
    match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", str(value))
    return float(match.group()) if match else np.nan


def build_base_cohort(hosp_dir: Path, icu_dir: Path, min_icu_los_days: float) -> pd.DataFrame:
    icustays = pd.read_csv(csv_path(icu_dir, "icustays"))
    admissions = pd.read_csv(csv_path(hosp_dir, "admissions"))
    patients = pd.read_csv(csv_path(hosp_dir, "patients"))

    icustays = to_datetime_columns(icustays)
    admissions = to_datetime_columns(admissions)

    icustays = icustays[icustays["los"] >= min_icu_los_days].copy()
    icustays = icustays[icustays["hadm_id"].isin(admissions["hadm_id"])].copy()
    admissions = admissions[admissions["hadm_id"].isin(icustays["hadm_id"])].copy()
    patients = patients[patients["subject_id"].isin(admissions["subject_id"])].copy()

    cohort = icustays.merge(
        admissions[["hadm_id", "admittime", "deathtime", "hospital_expire_flag", "race", "insurance"]],
        on="hadm_id",
        how="left",
    )
    cohort = cohort.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    )

    hours_to_death = (cohort["deathtime"] - cohort["intime"]) / pd.Timedelta(hours=1)
    cohort[TARGET_COLUMN] = (
        (cohort["hospital_expire_flag"] == 1)
        & hours_to_death.between(24, 48, inclusive="both")
    ).astype(int)
    cohort["age"] = cohort["anchor_age"] + (cohort["intime"].dt.year - cohort["anchor_year"])
    cohort["age"] = cohort["age"].clip(upper=90)
    cohort["gender"] = (cohort["gender"] == "F").astype(int)
    cohort["race"] = cohort["race"].apply(clean_race)

    return cohort


def split_subjects(
    cohort: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    subject_labels = cohort.groupby("subject_id")[TARGET_COLUMN].max()
    stratify = subject_labels if subject_labels.value_counts().min() >= 2 else None
    train_subjects, test_subjects = train_test_split(
        subject_labels.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    cohort = cohort.copy()
    cohort["data_split"] = np.where(cohort["subject_id"].isin(train_subjects), "train", "test")
    return train_subjects, test_subjects, cohort


def routine_vital_itemids(icu_dir: Path) -> np.ndarray:
    d_items = pd.read_csv(csv_path(icu_dir, "d_items"))
    routine_vitals = d_items[
        (d_items["linksto"] == "chartevents")
        & (d_items["category"] == "Routine Vital Signs")
    ]
    return routine_vitals["itemid"].unique()


def load_chartevents(
    icu_dir: Path,
    cohort: pd.DataFrame,
    chunk_size: int,
) -> pd.DataFrame:
    stay_ids = set(cohort["stay_id"])
    itemids = set(routine_vital_itemids(icu_dir))
    chunks: list[pd.DataFrame] = []

    usecols = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "warning"]
    reader = pd.read_csv(csv_path(icu_dir, "chartevents"), usecols=usecols, chunksize=chunk_size)
    for chunk in tqdm(reader, desc="Filtering chartevents"):
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids)
            & chunk["itemid"].isin(itemids)
            & chunk["valuenum"].notna()
        ].copy()
        if not chunk.empty:
            chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
            chunk["warning"] = (
                chunk["warning"]
                .replace({"True": 1, "False": 0, True: 1, False: 0})
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
            )
            chunk = chunk[chunk["valuenum"].between(0, 300, inclusive="both")]
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    chartevents = pd.concat(chunks, ignore_index=True)
    return to_datetime_columns(chartevents)


def load_labevents(
    hosp_dir: Path,
    cohort: pd.DataFrame,
    chunk_size: int,
) -> pd.DataFrame:
    subject_ids = set(cohort["subject_id"])
    usecols = [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
        "flag",
        "priority",
        "comments",
    ]
    chunks: list[pd.DataFrame] = []

    reader = pd.read_csv(csv_path(hosp_dir, "labevents"), usecols=usecols, chunksize=chunk_size)
    for chunk in tqdm(reader, desc="Filtering labevents"):
        chunk = chunk[chunk["subject_id"].isin(subject_ids) & chunk["hadm_id"].notna()].copy()
        if chunk.empty:
            continue

        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        comment_values = chunk["comments"].apply(extract_number)
        chunk["valuenum"] = chunk["valuenum"].fillna(comment_values)
        chunk["abnormal"] = chunk["flag"].astype(str).str.lower().eq("abnormal").astype(int)
        chunk["priority"] = chunk["priority"].astype(str).str.upper().eq("STAT").astype(int)
        chunk = chunk.drop(columns=["flag", "comments"])
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "priority", "abnormal"])

    labevents = pd.concat(chunks, ignore_index=True)
    return to_datetime_columns(labevents)


def itemids_with_minimum_coverage(
    dataframe: pd.DataFrame,
    uid_column: str,
    key_column: str,
    minimum_coverage: float,
) -> set[int]:
    uid_count = dataframe[uid_column].nunique()
    if uid_count == 0:
        return set()

    coverage = dataframe.groupby(key_column)[uid_column].nunique() / uid_count
    return set(coverage[coverage >= minimum_coverage].index)


def mean_features_in_first_24h(
    dataframe: pd.DataFrame,
    intime: pd.Timestamp,
    time_column: str,
    key_column: str,
    value_column: str,
) -> dict[int, float]:
    if dataframe.empty:
        return {}

    window_end = intime + pd.Timedelta(hours=24)
    current = dataframe[dataframe[time_column].between(intime, window_end, inclusive="both")].copy()
    current[value_column] = pd.to_numeric(current[value_column], errors="coerce")
    current = current[current[value_column].notna()]
    grouped = current.groupby(key_column)[value_column].mean()
    return grouped.to_dict()


def build_feature_table(
    cohort: pd.DataFrame,
    chartevents: pd.DataFrame,
    labevents: pd.DataFrame,
    train_subjects: np.ndarray,
    minimum_coverage: float,
) -> pd.DataFrame:
    train_stay_ids = cohort.loc[cohort["subject_id"].isin(train_subjects), "stay_id"]
    vital_itemids = itemids_with_minimum_coverage(
        chartevents[chartevents["stay_id"].isin(train_stay_ids)],
        uid_column="stay_id",
        key_column="itemid",
        minimum_coverage=minimum_coverage,
    )
    lab_itemids = itemids_with_minimum_coverage(
        labevents[labevents["subject_id"].isin(train_subjects)],
        uid_column="subject_id",
        key_column="itemid",
        minimum_coverage=minimum_coverage,
    )

    chartevents = chartevents[chartevents["itemid"].isin(vital_itemids)].copy()
    labevents = labevents[labevents["itemid"].isin(lab_itemids)].copy()
    chartevents_by_stay = {
        stay_id: group for stay_id, group in chartevents.groupby("stay_id", sort=False)
    }
    labevents_by_subject = {
        subject_id: group for subject_id, group in labevents.groupby("subject_id", sort=False)
    }
    empty_chartevents = chartevents.iloc[0:0]
    empty_labevents = labevents.iloc[0:0]

    rows: list[dict[str, object]] = []
    cohort_records = cohort.to_dict("records")
    for stay in tqdm(cohort_records, desc="Building feature rows"):
        stay_chartevents = chartevents_by_stay.get(stay["stay_id"], empty_chartevents)
        subject_labevents = labevents_by_subject.get(stay["subject_id"], empty_labevents)
        vital_values = mean_features_in_first_24h(
            stay_chartevents,
            intime=stay["intime"],
            time_column="charttime",
            key_column="itemid",
            value_column="valuenum",
        )
        vital_warnings = mean_features_in_first_24h(
            stay_chartevents,
            intime=stay["intime"],
            time_column="charttime",
            key_column="itemid",
            value_column="warning",
        )
        lab_values = mean_features_in_first_24h(
            subject_labevents,
            intime=stay["intime"],
            time_column="charttime",
            key_column="itemid",
            value_column="valuenum",
        )
        lab_priorities = mean_features_in_first_24h(
            subject_labevents,
            intime=stay["intime"],
            time_column="charttime",
            key_column="itemid",
            value_column="priority",
        )
        lab_abnormal = mean_features_in_first_24h(
            subject_labevents,
            intime=stay["intime"],
            time_column="charttime",
            key_column="itemid",
            value_column="abnormal",
        )

        rows.append(
            {
                "subject_id": stay["subject_id"],
                "hadm_id": stay["hadm_id"],
                "stay_id": stay["stay_id"],
                "age": stay["age"],
                "gender": stay["gender"],
                "race": stay["race"],
                "insurance": stay["insurance"],
                "data_split": stay["data_split"],
                TARGET_COLUMN: stay[TARGET_COLUMN],
                **{f"vital_{key}": value for key, value in vital_values.items()},
                **{f"vital_warning_{key}": value for key, value in vital_warnings.items()},
                **{f"lab_{key}": value for key, value in lab_values.items()},
                **{f"lab_priority_{key}": value for key, value in lab_priorities.items()},
                **{f"lab_abnormal_{key}": value for key, value in lab_abnormal.items()},
            }
        )

    return pd.DataFrame(rows)


def preprocess(args: argparse.Namespace) -> None:
    hosp_dir = Path(args.mimic_hosp_dir)
    icu_dir = Path(args.mimic_icu_dir)
    output_path = Path(args.output)

    cohort = build_base_cohort(hosp_dir, icu_dir, args.min_icu_los_days)
    train_subjects, _, cohort = split_subjects(cohort, args.test_size, args.random_state)
    chartevents = load_chartevents(icu_dir, cohort, args.chunk_size)
    labevents = load_labevents(hosp_dir, cohort, args.chunk_size)
    features = build_feature_table(
        cohort,
        chartevents,
        labevents,
        train_subjects,
        args.minimum_coverage,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"Feature table saved to: {output_path}")
    print(f"Rows: {features.shape[0]}, columns: {features.shape[1]}")
    print(features[TARGET_COLUMN].value_counts().rename("count"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MIMIC-IV ICU mortality feature table.")
    parser.add_argument("--mimic-hosp-dir", required=True, help="Path to MIMIC-IV hosp table directory.")
    parser.add_argument("--mimic-icu-dir", required=True, help="Path to MIMIC-IV icu table directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output feature CSV.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="CSV chunk size.")
    parser.add_argument("--min-icu-los-days", type=float, default=1.0, help="Minimum ICU length of stay in days.")
    parser.add_argument("--minimum-coverage", type=float, default=0.9, help="Minimum train-set UID coverage for itemids.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Subject-level test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
