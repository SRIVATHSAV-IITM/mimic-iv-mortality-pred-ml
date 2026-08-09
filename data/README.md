# Local data directory

MIMIC-IV is controlled-access. Do not commit raw or derived patient data.

After receiving authorized access, build the local feature table with:

```bash
python src/preprocess.py \
  --mimic-hosp-dir /path/to/mimiciv/hosp \
  --mimic-icu-dir /path/to/mimiciv/icu
```

This creates `data/mimic_icu_features.csv`, which remains ignored by Git.

For a software-only smoke test:

```bash
python scripts/generate_synthetic_data.py
```

`synthetic_features.csv` is fake and cannot support clinical claims.
