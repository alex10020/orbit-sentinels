"""Orbital Sentinel -- ingestion pipeline (extract -> validate -> load).

Moves the project's source files from the raw landing zone into a queryable
SQLite database that follows the data dictionary.

    python pipeline/ingest.py --db db/orbital_sentinel.db \
        --gp data/raw/gp_catalog.json \
        --boxscore data/raw/boxscore.csv \
        --alerts data/raw/conjunction_alerts.csv \
        --window-start "2026-10-31 16:18:40"

Design rules
  * Each source file is loaded in ONE transaction: it lands completely or not at all.
  * Every run is recorded in ingestion_run (file hash, counts, status); loading a
    file whose hash already succeeded is skipped, so re-runs are idempotent.
  * Records failing validation are not dropped silently: they go to ingest_reject
    with the reason and the raw record, so they can be inspected and replayed.
  * Database CHECK constraints are the second line of defence after Python checks.

Status (check-in): GP, boxscore and alert loaders are implemented and tested.
The CDM loader (load_cdm) is implemented but awaits a fresh cdm_public extract;
orchestration and the document store are still in design (see report §8).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OBJECT_TYPES = {"PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN", "TBA"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def to_float(v, name, required=True):
    if v is None or (isinstance(v, str) and not v.strip()):
        if required:
            raise ValueError(f"{name} is missing")
        return None
    return float(v)


def to_int(v, name, required=True):
    f = to_float(v, name, required)
    return None if f is None else int(f)


def norm_ts(v):
    """ISO-8601 'YYYY-MM-DDTHH:MM:SS.ffffff' -> 'YYYY-MM-DD HH:MM:SS.ffffff'."""
    return None if v is None else str(v).replace("T", " ")


def required_ts(v, name):
    if not v:
        raise ValueError(f"{name} is missing")
    return norm_ts(v)


class Run:
    """Context manager writing one ingestion_run row and owning the transaction."""

    def __init__(self, con, source, path: Path):
        self.con, self.source, self.path = con, source, path
        self.run_id = f"{source}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
        self.read = self.loaded = self.rejected = 0
        self.skip = False

    def __enter__(self):
        digest = sha256(self.path)
        done = self.con.execute(
            "SELECT run_id FROM ingestion_run WHERE source=? AND file_sha256=? AND status='succeeded'",
            (self.source, digest)).fetchone()
        self.con.execute("BEGIN")
        self.con.execute(
            "INSERT INTO ingestion_run(run_id,source,file_name,file_sha256,file_bytes,started_utc,status)"
            " VALUES (?,?,?,?,?,?,?)",
            (self.run_id, self.source, self.path.name, digest, self.path.stat().st_size, utcnow(),
             "skipped" if done else "running"))
        if done:
            self.skip = True
            print(f"[{self.source}] {self.path.name} already loaded by {done[0]} -- skipped")
        self.t0 = time.perf_counter()
        return self

    def reject(self, key, reason, raw):
        self.rejected += 1
        self.con.execute("INSERT INTO ingest_reject(run_id,record_key,reason,raw_record) VALUES (?,?,?,?)",
                         (self.run_id, key, reason, json.dumps(raw)[:4000]))

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.con.execute("ROLLBACK")
            self.con.execute(
                "INSERT INTO ingestion_run(run_id,source,file_name,file_sha256,file_bytes,started_utc,finished_utc,status)"
                " VALUES (?,?,?,?,?,?,?,'failed')",
                (self.run_id, self.source, self.path.name, "-", 0, utcnow(), utcnow()))
            self.con.commit()
            print(f"[{self.source}] FAILED and rolled back: {exc}")
            return False
        status = "skipped" if self.skip else "succeeded"
        self.con.execute(
            "UPDATE ingestion_run SET finished_utc=?, rows_read=?, rows_loaded=?, rows_rejected=?, status=?"
            " WHERE run_id=?",
            (utcnow(), self.read, self.loaded, self.rejected, status, self.run_id))
        self.con.execute("COMMIT")
        if not self.skip:
            print(f"[{self.source}] read {self.read:,}  loaded {self.loaded:,}  rejected {self.rejected:,}"
                  f"  in {time.perf_counter() - self.t0:.1f}s  (run {self.run_id})")
        return False


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def gp_row(r: dict, run_id: str) -> tuple:
    otype = (r.get("OBJECT_TYPE") or "").strip()
    if otype not in OBJECT_TYPES:
        raise ValueError(f"unknown OBJECT_TYPE {otype!r}")
    l1, l2 = r.get("TLE_LINE1") or "", r.get("TLE_LINE2") or ""
    if len(l1) != 69 or len(l2) != 69:
        raise ValueError("TLE line length != 69")
    return (
        to_int(r.get("GP_ID"), "GP_ID"), to_int(r.get("NORAD_CAT_ID"), "NORAD_CAT_ID"),
        r.get("OBJECT_NAME"), r.get("OBJECT_ID"), otype, r.get("COUNTRY_CODE"),
        r.get("LAUNCH_DATE"), r.get("SITE"), r.get("DECAY_DATE"), r.get("RCS_SIZE"),
        required_ts(r.get("EPOCH"), "EPOCH"),
        to_float(r.get("MEAN_MOTION"), "MEAN_MOTION"), to_float(r.get("ECCENTRICITY"), "ECCENTRICITY"),
        to_float(r.get("INCLINATION"), "INCLINATION"), to_float(r.get("RA_OF_ASC_NODE"), "RA_OF_ASC_NODE"),
        to_float(r.get("ARG_OF_PERICENTER"), "ARG_OF_PERICENTER"), to_float(r.get("MEAN_ANOMALY"), "MEAN_ANOMALY"),
        to_float(r.get("BSTAR"), "BSTAR", False), to_float(r.get("MEAN_MOTION_DOT"), "MEAN_MOTION_DOT", False),
        to_float(r.get("MEAN_MOTION_DDOT"), "MEAN_MOTION_DDOT", False),
        to_float(r.get("SEMIMAJOR_AXIS"), "SEMIMAJOR_AXIS", False), to_float(r.get("PERIOD"), "PERIOD", False),
        to_float(r.get("APOAPSIS"), "APOAPSIS", False), to_float(r.get("PERIAPSIS"), "PERIAPSIS", False),
        to_int(r.get("REV_AT_EPOCH"), "REV_AT_EPOCH", False), to_int(r.get("ELEMENT_SET_NO"), "ELEMENT_SET_NO", False),
        r.get("CLASSIFICATION_TYPE"), norm_ts(r.get("CREATION_DATE")), l1, l2, run_id,
    )


def load_gp(con, path: Path):
    with Run(con, "gp", path) as run:
        if run.skip:
            return
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("GP file is not a JSON array (Space-Track error envelope?)")
        sql = "INSERT OR REPLACE INTO gp_element_set VALUES (" + ",".join("?" * 31) + ")"
        for r in records:
            run.read += 1
            try:
                con.execute(sql, gp_row(r, run.run_id))
                run.loaded += 1
            except (ValueError, sqlite3.IntegrityError) as exc:
                run.reject(r.get("GP_ID"), str(exc), r)


BOX_INT_COLS = ["ORBITAL_TBA", "ORBITAL_PAYLOAD_COUNT", "ORBITAL_ROCKET_BODY_COUNT", "ORBITAL_DEBRIS_COUNT",
                "ORBITAL_TOTAL_COUNT", "DECAYED_PAYLOAD_COUNT", "DECAYED_ROCKET_BODY_COUNT",
                "DECAYED_DEBRIS_COUNT", "DECAYED_TOTAL_COUNT", "COUNTRY_TOTAL"]


def load_boxscore(con, path: Path, snapshot_date: str):
    with Run(con, "boxscore", path) as run:
        if run.skip:
            return
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = {"COUNTRY", "SPADOC_CD", *BOX_INT_COLS} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"boxscore header missing {sorted(missing)}")
            for r in reader:
                run.read += 1
                if r["SPADOC_CD"] == "ALL":           # grand-total row: kept out, used for reconciliation
                    run.reject("ALL", "aggregate row (not an entity)", r)
                    continue
                try:
                    vals = [to_int(r[c], c) for c in BOX_INT_COLS]
                    con.execute("INSERT OR REPLACE INTO country_boxscore VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (snapshot_date, r["SPADOC_CD"].strip(), r["COUNTRY"].strip(), *vals, run.run_id))
                    run.loaded += 1
                except (ValueError, sqlite3.IntegrityError) as exc:
                    run.reject(r.get("SPADOC_CD"), str(exc), r)


def load_alerts(con, path: Path, engine_run_id: str):
    with Run(con, "alerts", path) as run:
        if run.skip:
            return
        sql = ("INSERT INTO conjunction_alert(engine_run_id,tca_utc,norad_1,norad_2,name_1,name_2,type_1,type_2,"
               "miss_distance_km,relative_speed_km_s,altitude_km,x_km,y_km,z_km,refined,ingestion_run_id)"
               " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
        with path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                run.read += 1
                try:
                    datetime.strptime(r["tca_utc"], "%Y-%m-%d %H:%M:%S")
                    con.execute(sql, (
                        engine_run_id, r["tca_utc"], int(r["norad_1"]), int(r["norad_2"]), r["name_1"], r["name_2"],
                        r["type_1"], r["type_2"], float(r["miss_distance_km"]), float(r["relative_speed_km_s"]),
                        float(r["altitude_km"]), float(r["x_km"]), float(r["y_km"]), float(r["z_km"]),
                        1 if str(r.get("refined", "")).strip().lower() == "true" else 0, run.run_id))
                    run.loaded += 1
                except (ValueError, KeyError, sqlite3.IntegrityError) as exc:
                    run.reject(f'{r.get("norad_1")}-{r.get("norad_2")}', str(exc), r)


def load_cdm(con, path: Path):
    """cdm_public JSON -> cdm_public table. MIN_RNG stays in metres, as published."""
    with Run(con, "cdm", path) as run:
        if run.skip:
            return
        for r in json.loads(path.read_text(encoding="utf-8")):
            run.read += 1
            try:
                con.execute("INSERT OR REPLACE INTO cdm_public VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    to_int(r.get("CDM_ID"), "CDM_ID"), norm_ts(r.get("CREATED")), norm_ts(r.get("TCA")),
                    to_float(r.get("MIN_RNG"), "MIN_RNG"), to_float(r.get("PC"), "PC", False),
                    to_int(r.get("SAT_1_ID"), "SAT_1_ID"), r.get("SAT_1_NAME"), r.get("SAT1_OBJECT_TYPE"),
                    to_int(r.get("SAT_2_ID"), "SAT_2_ID"), r.get("SAT_2_NAME"), r.get("SAT2_OBJECT_TYPE"),
                    r.get("EMERGENCY_REPORTABLE"), run.run_id))
                run.loaded += 1
            except (ValueError, sqlite3.IntegrityError) as exc:
                run.reject(r.get("CDM_ID"), str(exc), r)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Load Orbital Sentinel sources into SQLite.")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--gp", type=Path)
    ap.add_argument("--boxscore", type=Path)
    ap.add_argument("--boxscore-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--alerts", type=Path)
    ap.add_argument("--engine-run-id", default="unknown")
    ap.add_argument("--cdm", type=Path)
    ap.add_argument("--window-start", required=True, help="forecast window start, UTC 'YYYY-MM-DD HH:MM:SS'")
    a = ap.parse_args(argv)

    a.db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(a.db, isolation_level=None)  # explicit BEGIN/COMMIT in Run
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript((HERE / "sql" / "01_schema.sql").read_text())

    if a.gp:
        load_gp(con, a.gp)
    if a.boxscore:
        load_boxscore(con, a.boxscore, a.boxscore_date)
    if a.alerts:
        load_alerts(con, a.alerts, a.engine_run_id)
    if a.cdm:
        load_cdm(con, a.cdm)

    gen = (HERE / "sql" / "02_generated.sql").read_text().replace(":window_start", f"'{a.window_start}'")
    con.executescript(gen)
    print("generated tables/views rebuilt for window start", a.window_start)
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
