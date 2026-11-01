# Ingestion pipeline (Big Data Technologies, Group Assessment 1)

Loads the project's sources into a SQLite database that follows the data dictionary.

| File | Purpose |
|---|---|
| `ingest.py` | Extract, validate and load each source file in one transaction. Re-runs skip files already loaded (SHA-256). Bad records go to `ingest_reject`. |
| `validate.py` | Runs every named query in `sql/03_validation.sql` and prints the results. |
| `sql/01_schema.sql` | Tables, constraints, indexes. |
| `sql/02_generated.sql` | Datasets generated in SQL (forecast time grid, regimes, object types) and curated views. |
| `sql/03_validation.sql` | Validation checks and example downstream queries. |
| `results/` | Output of the last validation run. |

Sources (put them in `pipeline/data/raw/`, which is git-ignored):

- GP catalog, JSON: Space-Track `class/gp`
- Boxscore, CSV: Space-Track `class/boxscore`
- Conjunction alerts, CSV: `python main.py --full` in the repository root
- Public CDMs, JSON: Space-Track `class/cdm_public` (loader written, data pending)

```bash
python pipeline/ingest.py --db pipeline/db/orbital_sentinel.db \
  --gp pipeline/data/raw/gp_catalog_20261031.json \
  --boxscore pipeline/data/raw/boxscore_20261031.csv --boxscore-date 2026-10-31 \
  --alerts pipeline/data/raw/conjunction_alerts_20261031T161840Z.csv \
  --engine-run-id 20261031T161840Z --window-start "2026-10-31 16:18:40"
python pipeline/validate.py --db pipeline/db/orbital_sentinel.db
```

Only the Python standard library is needed.
