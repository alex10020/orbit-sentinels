#!/usr/bin/env bash
# Rebuild the Orbital Sentinel database from the raw files in data/raw/,
# run the validation checks and write the exports (SQL Server script, CSV, SQLite dump).
# Run from the repository root:  bash pipeline/run_all.sh
# No Space Track account is needed: everything is loaded from the landing folder.
set -euo pipefail
cd "$(dirname "$0")"

SNAP=20261031
ENGINE_RUN=20261031T161840Z
WINDOW_START="2026-10-31 16:18:40"
DB=db/orbital_sentinel.db

mkdir -p db results
rm -f "$DB"

python3 ingest.py --db "$DB" \
  --gp data/raw/gp_catalog_${SNAP}.json \
  --boxscore data/raw/boxscore_${SNAP}.csv --boxscore-date 2026-10-31 \
  --alerts data/raw/conjunction_alerts_${ENGINE_RUN}.csv \
  --engine-run-id "$ENGINE_RUN" --window-start "$WINDOW_START"

python3 validate.py --db "$DB" --out results/validation_${SNAP}.txt
python3 export.py --db "$DB" --out db/export
