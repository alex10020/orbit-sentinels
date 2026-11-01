# Ingestion pipeline (Big Data Technologies, Group Assessment 1)

Loads the project's sources into a SQLite database that follows the data dictionary.

| File | Purpose |
|---|---|
| `ingest.py` | Extract, validate and load each source file in one transaction. Re-runs skip files already loaded (SHA256 hash). Bad records go to `ingest_reject`. |
| `validate.py` | Runs every named query in `sql/03_validation.sql` and prints the results. |
| `sql/01_schema.sql` | Tables, constraints, indexes. |
| `sql/02_generated.sql` | Datasets generated in SQL (forecast time grid, regimes, object types) and curated views. |
| `sql/03_validation.sql` | Validation checks and example downstream queries. |
| `export.py` | Writes the database as CSV files and as a SQL Server script for SSMS (and, with `--sqlite-dump`, as a SQLite SQL dump). |
| `run_all.sh` | Rebuilds the database from `data/raw/`, runs the checks and writes the exports. |
| `results/` | Output of the last validation run. |

Sources (put them in `pipeline/data/raw/`, which is git-ignored):

- GP catalog, JSON: Space-Track `class/gp`
- Boxscore, CSV: Space-Track `class/boxscore` (course use only, not basic SSA data)
- Conjunction alerts, CSV: `python main.py --full` in the repository root
- Public CDMs, JSON: Space-Track `class/cdm_public` (loader written, data pending, course use only)

```bash
python pipeline/ingest.py --db pipeline/db/orbital_sentinel.db \
  --gp pipeline/data/raw/gp_catalog_20261031.json \
  --boxscore pipeline/data/raw/boxscore_20261031.csv --boxscore-date 2026-10-31 \
  --alerts pipeline/data/raw/conjunction_alerts_20261031T161840Z.csv \
  --engine-run-id 20261031T161840Z --window-start "2026-10-31 16:18:40"
python pipeline/validate.py --db pipeline/db/orbital_sentinel.db
python pipeline/export.py --db pipeline/db/orbital_sentinel.db --out pipeline/db/export
```

Or all three in one go, with no Space Track account needed once the raw files
are in `data/raw/`:

```bash
bash pipeline/run_all.sh
```

### Opening the database

- **SQLite** (`db/orbital_sentinel.db`): open directly in DBeaver, DB Browser for
  SQLite or the `sqlite3` shell. `export.py --sqlite-dump` also writes it as a
  plain SQL dump.
- **SQL Server / SSMS**: copy `db/export/csv/` to a folder the SQL Server service
  can read, set `@csv_dir` at the top of `db/export/orbital_sentinel_mssql.sql`
  and run the script (SQL Server 2017 or newer). It creates the tables with the
  same constraints, loads the CSVs, rebuilds the SQL generated tables and views,
  and ends with a short validation.
- **CSV**: `db/export/csv/` has one file per table, for Excel or any other tool.

Only the Python standard library is needed.

Orbital data: U.S. Space Command, via Space-Track.org.
