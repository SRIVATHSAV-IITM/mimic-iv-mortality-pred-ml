# Data Folder

Place your prepared MIMIC-IV ICU feature table here as:

```text
mimic_icu_features.csv
```

The file should contain one row per ICU stay or prediction unit, structured feature columns, and one binary mortality target column.

To generate it from raw MIMIC-IV tables, run:

```bash
python src/preprocess.py --mimic-hosp-dir /path/to/mimiciv/hosp --mimic-icu-dir /path/to/mimiciv/icu
```

Supported target column names:

- `mortality_48h`
- `48_hour_mortality_flag`
- `mortality`
- `hospital_expire_flag`

Common identifier columns such as `subject_id`, `hadm_id`, and `stay_id` can be included. They are used for splitting or excluded from model training.

Do not commit real MIMIC-IV data to this repository.
