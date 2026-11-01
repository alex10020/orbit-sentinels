-- Orbital Sentinel -- integration database schema (SQLite 3)
-- Layers: raw landing files (outside the DB) -> staging/curated tables below -> views.
-- Every loaded row carries the ingestion_run that produced it (lineage).

PRAGMA foreign_keys = ON;

-- --------------------------------------------------------------------------
-- Operational metadata
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_run (
    run_id         TEXT PRIMARY KEY,               -- e.g. gp_20261031T161840Z
    source         TEXT NOT NULL,                  -- gp | boxscore | alerts | cdm
    file_name      TEXT NOT NULL,
    file_sha256    TEXT NOT NULL,                  -- detects re-loads of the same file
    file_bytes     INTEGER NOT NULL,
    started_utc    TEXT NOT NULL,
    finished_utc   TEXT,
    rows_read      INTEGER DEFAULT 0,
    rows_loaded    INTEGER DEFAULT 0,
    rows_rejected  INTEGER DEFAULT 0,
    status         TEXT NOT NULL CHECK (status IN ('running','succeeded','failed','skipped'))
);

CREATE TABLE IF NOT EXISTS ingest_reject (
    reject_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES ingestion_run(run_id),
    record_key     TEXT,
    reason         TEXT NOT NULL,
    raw_record     TEXT                              -- original record as JSON, for replay
);

-- --------------------------------------------------------------------------
-- Source 1: Space-Track GP catalog (JSON, CCSDS OMM keywords)
-- One row per element set; GP_ID is Space-Track's unique element-set id.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gp_element_set (
    gp_id               INTEGER PRIMARY KEY,
    norad_cat_id        INTEGER NOT NULL,
    object_name         TEXT,
    object_id           TEXT,                        -- international designator YYYY-NNNP
    object_type         TEXT NOT NULL CHECK (object_type IN ('PAYLOAD','ROCKET BODY','DEBRIS','UNKNOWN','TBA')),
    country_code        TEXT,                        -- joins country_boxscore.spadoc_cd
    launch_date         TEXT,
    site                TEXT,
    decay_date          TEXT,                        -- NULL = still on orbit
    rcs_size            TEXT CHECK (rcs_size IS NULL OR rcs_size IN ('SMALL','MEDIUM','LARGE')),
    epoch_utc           TEXT NOT NULL,
    mean_motion_rev_day REAL NOT NULL CHECK (mean_motion_rev_day > 0),
    eccentricity        REAL NOT NULL CHECK (eccentricity >= 0 AND eccentricity < 1),
    inclination_deg     REAL NOT NULL CHECK (inclination_deg BETWEEN 0 AND 180),
    raan_deg            REAL NOT NULL CHECK (raan_deg >= 0 AND raan_deg < 360),
    arg_pericenter_deg  REAL NOT NULL CHECK (arg_pericenter_deg >= 0 AND arg_pericenter_deg < 360),
    mean_anomaly_deg    REAL NOT NULL CHECK (mean_anomaly_deg >= 0 AND mean_anomaly_deg < 360),
    bstar               REAL,
    mean_motion_dot     REAL,
    mean_motion_ddot    REAL,
    semimajor_axis_km   REAL,
    period_min          REAL,
    apoapsis_km         REAL,
    periapsis_km        REAL,
    rev_at_epoch        INTEGER,
    element_set_no      INTEGER,
    classification_type TEXT,
    creation_date_utc   TEXT,
    tle_line1           TEXT NOT NULL CHECK (length(tle_line1) = 69),
    tle_line2           TEXT NOT NULL CHECK (length(tle_line2) = 69),
    ingestion_run_id    TEXT NOT NULL REFERENCES ingestion_run(run_id)
);
CREATE INDEX IF NOT EXISTS ix_gp_norad  ON gp_element_set(norad_cat_id);
CREATE INDEX IF NOT EXISTS ix_gp_epoch  ON gp_element_set(epoch_utc);
CREATE INDEX IF NOT EXISTS ix_gp_country ON gp_element_set(country_code);

-- --------------------------------------------------------------------------
-- Source 2: Space-Track boxscore (CSV) -- objects per owner country
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS country_boxscore (
    snapshot_date              TEXT NOT NULL,
    spadoc_cd                  TEXT NOT NULL,
    country                    TEXT NOT NULL,
    orbital_tba                INTEGER NOT NULL CHECK (orbital_tba >= 0),
    orbital_payload_count      INTEGER NOT NULL CHECK (orbital_payload_count >= 0),
    orbital_rocket_body_count  INTEGER NOT NULL CHECK (orbital_rocket_body_count >= 0),
    orbital_debris_count       INTEGER NOT NULL CHECK (orbital_debris_count >= 0),
    orbital_total_count        INTEGER NOT NULL,
    decayed_payload_count      INTEGER NOT NULL CHECK (decayed_payload_count >= 0),
    decayed_rocket_body_count  INTEGER NOT NULL CHECK (decayed_rocket_body_count >= 0),
    decayed_debris_count       INTEGER NOT NULL CHECK (decayed_debris_count >= 0),
    decayed_total_count        INTEGER NOT NULL,
    country_total              INTEGER NOT NULL,
    ingestion_run_id           TEXT NOT NULL REFERENCES ingestion_run(run_id),
    PRIMARY KEY (snapshot_date, spadoc_cd),
    CHECK (orbital_total_count = orbital_tba + orbital_payload_count + orbital_rocket_body_count + orbital_debris_count),
    CHECK (decayed_total_count = decayed_payload_count + decayed_rocket_body_count + decayed_debris_count),
    CHECK (country_total = orbital_total_count + decayed_total_count)
);

-- --------------------------------------------------------------------------
-- Source 3: conjunction alerts produced by the Orbital Sentinel engine (CSV)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conjunction_alert (
    alert_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    engine_run_id        TEXT NOT NULL,              -- engine run that produced the CSV
    tca_utc              TEXT NOT NULL,
    norad_1              INTEGER NOT NULL,
    norad_2              INTEGER NOT NULL,
    name_1               TEXT,
    name_2               TEXT,
    type_1               TEXT,
    type_2               TEXT,
    miss_distance_km     REAL NOT NULL CHECK (miss_distance_km >= 0),
    relative_speed_km_s  REAL NOT NULL CHECK (relative_speed_km_s >= 0 AND relative_speed_km_s <= 16.5),
    altitude_km          REAL NOT NULL,
    x_km REAL, y_km REAL, z_km REAL,
    refined              INTEGER NOT NULL CHECK (refined IN (0,1)),
    ingestion_run_id     TEXT NOT NULL REFERENCES ingestion_run(run_id),
    CHECK (norad_1 <> norad_2),
    UNIQUE (engine_run_id, norad_1, norad_2)
);
CREATE INDEX IF NOT EXISTS ix_alert_miss ON conjunction_alert(miss_distance_km);
CREATE INDEX IF NOT EXISTS ix_alert_tca  ON conjunction_alert(tca_utc);

-- --------------------------------------------------------------------------
-- Source 4: public Conjunction Data Messages (JSON) -- table defined, loader in progress
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cdm_public (
    cdm_id              INTEGER PRIMARY KEY,
    created_utc         TEXT,
    tca_utc             TEXT NOT NULL,
    min_rng_m           REAL NOT NULL CHECK (min_rng_m >= 0),     -- published in metres
    pc                  REAL CHECK (pc IS NULL OR (pc >= 0 AND pc <= 1)),
    sat_1_id            INTEGER NOT NULL,
    sat_1_name          TEXT,
    sat1_object_type    TEXT,
    sat_2_id            INTEGER NOT NULL,
    sat_2_name          TEXT,
    sat2_object_type    TEXT,
    emergency_reportable TEXT,
    ingestion_run_id    TEXT NOT NULL REFERENCES ingestion_run(run_id)
);
