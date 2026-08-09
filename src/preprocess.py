"""Create a clinically interpretable MIMIC-IV landmark feature table.

The prediction time is 24 hours after ICU admission. The outcome is in-hospital
death after that landmark and no later than 48 hours after ICU admission.
Only the first ICU stay in each hospital admission is retained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import TARGET_COLUMN, assign_subject_splits

DEFAULT_OUTPUT = Path("data/mimic_icu_features.csv")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    minimum: float
    maximum: float
    scale: float = 1.0
    offset: float = 0.0


# Curated, commonly used MIMIC-IV variables. Fixed item IDs make the project
# easier to understand and ensure that every output feature has a clinical name.
VITAL_SPECS = {
    220045: FeatureSpec("heart_rate", 20, 250),
    220179: FeatureSpec("systolic_bp", 40, 300),
    220180: FeatureSpec("diastolic_bp", 20, 200),
    220181: FeatureSpec("mean_arterial_pressure", 20, 250),
    220210: FeatureSpec("respiratory_rate", 1, 80),
    220277: FeatureSpec("oxygen_saturation", 50, 100),
    223761: FeatureSpec("temperature_c", 25, 45, scale=5 / 9, offset=-32 * 5 / 9),
    223762: FeatureSpec("temperature_c", 25, 45),
}

LAB_SPECS = {
    50862: FeatureSpec("albumin", 0.5, 8),
    50868: FeatureSpec("anion_gap", 0, 60),
    50882: FeatureSpec("bicarbonate", 2, 60),
    50885: FeatureSpec("bilirubin_total", 0, 80),
    50912: FeatureSpec("creatinine", 0.1, 30),
    50931: FeatureSpec("glucose", 20, 1500),
    50971: FeatureSpec("potassium", 1, 10),
    50983: FeatureSpec("sodium", 90, 200),
    51006: FeatureSpec("blood_urea_nitrogen", 1, 300),
    51221: FeatureSpec("hematocrit", 5, 75),
    51222: FeatureSpec("hemoglobin", 2, 25),
    51237: FeatureSpec("inr", 0.5, 20),
    51265: FeatureSpec("platelet_count", 1, 2000),
    51301: FeatureSpec("white_blood_cell_count", 0.1, 500),
}


def csv_path(parent: Path, table_name: str) -> Path:
    """Find an uncompressed or gzip-compressed MIMIC-IV CSV table."""
    for suffix in (".csv", ".csv.gz"):
        candidate = parent / f"{table_name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {table_name}.csv[.gz] in {parent}")


def clean_race(value: object) -> str:
    """Collapse detailed race values into transparent broad groups."""
    if pd.isna(value):
        return "UNKNOWN"
    value_upper = str(value).upper()
    for group in ("WHITE", "BLACK", "ASIAN", "HISPANIC"):
        if value_upper.startswith(group):
            return group
    if "DECLINED" in value_upper or "UNABLE" in value_upper or "UNKNOWN" in value_upper:
        return "UNKNOWN"
    return "OTHER"


def build_base_cohort(hosp_dir: Path, icu_dir: Path) -> pd.DataFrame:
    """Build the 24-hour landmark cohort and mortality outcome."""
    icustays = pd.read_csv(csv_path(icu_dir, "icustays"))
    admissions = pd.read_csv(csv_path(hosp_dir, "admissions"))
    patients = pd.read_csv(csv_path(hosp_dir, "patients"))

    for frame, columns in (
        (icustays, ["intime", "outtime"]),
        (admissions, ["admittime", "dischtime", "deathtime"]),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    # One prediction per hospital admission keeps the unit of analysis clear.
    icustays = (
        icustays.dropna(subset=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
        .sort_values(["hadm_id", "intime"])
        .drop_duplicates("hadm_id", keep="first")
    )
    admission_columns = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
        "race",
        "insurance",
        "admission_type",
    ]
    cohort = icustays.merge(admissions[admission_columns], on=["subject_id", "hadm_id"], how="inner")
    cohort = cohort.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="inner",
    )

    cohort["landmark_time"] = cohort["intime"] + pd.Timedelta(hours=24)
    cohort["prediction_end_time"] = cohort["intime"] + pd.Timedelta(hours=48)

    # Patients must still be in the ICU and alive at the 24-hour landmark.
    observed_at_landmark = cohort["outtime"] >= cohort["landmark_time"]
    alive_at_landmark = cohort["deathtime"].isna() | (
        cohort["deathtime"] > cohort["landmark_time"]
    )
    cohort = cohort.loc[observed_at_landmark & alive_at_landmark].copy()

    cohort[TARGET_COLUMN] = (
        cohort["deathtime"].notna()
        & (cohort["deathtime"] > cohort["landmark_time"])
        & (cohort["deathtime"] <= cohort["prediction_end_time"])
    ).astype(int)
    cohort["age"] = cohort["anchor_age"] + (
        cohort["intime"].dt.year - cohort["anchor_year"]
    )
    cohort["age"] = cohort["age"].clip(lower=18, upper=91)
    cohort["race"] = cohort["race"].apply(clean_race)
    cohort["gender"] = cohort["gender"].fillna("UNKNOWN").astype(str)
    cohort["insurance"] = cohort["insurance"].fillna("UNKNOWN").astype(str)
    cohort["admission_type"] = cohort["admission_type"].fillna("UNKNOWN").astype(str)
    return cohort.reset_index(drop=True)


def _extract_event_features(
    table_path: Path,
    cohort: pd.DataFrame,
    specs: dict[int, FeatureSpec],
    join_column: str,
    chunk_size: int,
) -> pd.DataFrame:
    """Extract and aggregate selected measurements from one event table."""
    cohort_lookup = cohort[[join_column, "intime", "landmark_time"]].drop_duplicates(join_column)
    valid_ids = set(cohort_lookup[join_column])
    selected_itemids = set(specs)
    parts: list[pd.DataFrame] = []

    usecols = [join_column, "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(table_path, usecols=usecols, chunksize=chunk_size):
        chunk = chunk[
            chunk[join_column].isin(valid_ids)
            & chunk["itemid"].isin(selected_itemids)
            & chunk["valuenum"].notna()
        ].copy()
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.merge(cohort_lookup, on=join_column, how="inner")
        chunk = chunk[
            chunk["charttime"].between(
                chunk["intime"], chunk["landmark_time"], inclusive="both"
            )
        ].copy()
        if chunk.empty:
            continue

        chunk["feature"] = chunk["itemid"].map(lambda itemid: specs[int(itemid)].name)
        chunk["minimum"] = chunk["itemid"].map(lambda itemid: specs[int(itemid)].minimum)
        chunk["maximum"] = chunk["itemid"].map(lambda itemid: specs[int(itemid)].maximum)
        chunk["scale"] = chunk["itemid"].map(lambda itemid: specs[int(itemid)].scale)
        chunk["offset"] = chunk["itemid"].map(lambda itemid: specs[int(itemid)].offset)
        chunk["value"] = chunk["valuenum"] * chunk["scale"] + chunk["offset"]
        chunk = chunk[chunk["value"].between(chunk["minimum"], chunk["maximum"])]
        parts.append(chunk[[join_column, "feature", "value"]])

    if not parts:
        return pd.DataFrame(index=pd.Index([], name=join_column))

    events = pd.concat(parts, ignore_index=True)
    aggregated = events.groupby([join_column, "feature"])["value"].agg(
        mean="mean",
        minimum="min",
        maximum="max",
        count="count",
    )
    wide = aggregated.unstack("feature")
    wide.columns = [f"{feature}_{statistic}" for statistic, feature in wide.columns]
    return wide.sort_index(axis=1)


def build_feature_table(
    cohort: pd.DataFrame,
    hosp_dir: Path,
    icu_dir: Path,
    chunk_size: int,
) -> pd.DataFrame:
    """Combine demographics with first-24-hour vital and lab summaries."""
    vital_features = _extract_event_features(
        csv_path(icu_dir, "chartevents"),
        cohort,
        VITAL_SPECS,
        join_column="stay_id",
        chunk_size=chunk_size,
    ).add_prefix("vital_")
    lab_features = _extract_event_features(
        csv_path(hosp_dir, "labevents"),
        cohort,
        LAB_SPECS,
        join_column="hadm_id",
        chunk_size=chunk_size,
    ).add_prefix("lab_")

    static_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "age",
        "gender",
        "race",
        "insurance",
        "admission_type",
        TARGET_COLUMN,
    ]
    features = cohort[static_columns].copy()
    features = features.merge(vital_features, left_on="stay_id", right_index=True, how="left")
    features = features.merge(lab_features, left_on="hadm_id", right_index=True, how="left")
    return features


def write_data_dictionary(path: Path) -> None:
    rows = []
    for source, specs in (("chartevents", VITAL_SPECS), ("labevents", LAB_SPECS)):
        for itemid, spec in specs.items():
            rows.append(
                {
                    "source_table": source,
                    "itemid": itemid,
                    "feature": spec.name,
                    "plausible_minimum": spec.minimum,
                    "plausible_maximum": spec.maximum,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["source_table", "feature"]).to_csv(path, index=False)


def preprocess(args: argparse.Namespace) -> None:
    hosp_dir = Path(args.mimic_hosp_dir)
    icu_dir = Path(args.mimic_icu_dir)
    cohort = build_base_cohort(hosp_dir, icu_dir)
    features = build_feature_table(cohort, hosp_dir, icu_dir, args.chunk_size)
    features = assign_subject_splits(
        features,
        target_column=TARGET_COLUMN,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    write_data_dictionary(Path(args.data_dictionary))

    print(f"Feature table saved to {output_path}")
    print(f"Rows: {len(features):,}; columns: {features.shape[1]:,}")
    print(features.groupby("data_split")[TARGET_COLUMN].agg(["count", "sum", "mean"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-hosp-dir", required=True)
    parser.add_argument("--mimic-icu-dir", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--data-dictionary", default="outputs/data_dictionary.csv")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
