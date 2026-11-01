-- Orbital Sentinel -- programmatically generated datasets (pure SQL)
-- These tables are not extracted from any file: SQL generates them, and the
-- curated views below derive analysis-ready entities from the loaded sources.

-- 1. Forecast time grid: 72 h at 1-minute resolution = 4,320 rows, generated
--    with a recursive CTE. The engine samples exactly these instants.
DROP TABLE IF EXISTS forecast_time_grid;
CREATE TABLE forecast_time_grid (
    step_index   INTEGER PRIMARY KEY,
    t_utc        TEXT NOT NULL,
    hour_of_run  INTEGER NOT NULL
);
INSERT INTO forecast_time_grid (step_index, t_utc, hour_of_run)
WITH RECURSIVE steps(i) AS (
    SELECT 0
    UNION ALL
    SELECT i + 1 FROM steps WHERE i < 72 * 60 - 1
)
SELECT i,
       strftime('%Y-%m-%d %H:%M:%S', :window_start, '+' || i || ' minutes'),
       i / 60
FROM steps;

-- 2. Orbital regime reference table (generated, with altitude bounds).
DROP TABLE IF EXISTS orbital_regime;
CREATE TABLE orbital_regime (
    regime        TEXT PRIMARY KEY,
    min_perigee_km REAL,
    max_perigee_km REAL,
    description   TEXT
);
INSERT INTO orbital_regime VALUES
  ('LEO',      0,      2000,   'Low Earth orbit: perigee at or below 2,000 km (screened by the engine)'),
  ('MEO/HEO',  2000,   35000,  'Medium or highly elliptical orbits'),
  ('GEO band', 35000,  36500,  'Geostationary belt'),
  ('BEYOND',   36500,  NULL,   'Above GEO, lunar or escape trajectories');

-- 3. Object-type dimension (generated) to normalise labels across sources.
DROP TABLE IF EXISTS object_type_dim;
CREATE TABLE object_type_dim (object_type TEXT PRIMARY KEY, boxscore_bucket TEXT NOT NULL);
INSERT INTO object_type_dim VALUES
  ('PAYLOAD','PAYLOAD'), ('ROCKET BODY','ROCKET BODY'), ('DEBRIS','DEBRIS'),
  ('UNKNOWN','TBA'), ('TBA','TBA');

-- 4. Curated views ---------------------------------------------------------
DROP VIEW IF EXISTS v_space_object;
CREATE VIEW v_space_object AS
SELECT g.norad_cat_id, g.object_name, g.object_type, g.country_code, g.rcs_size,
       g.epoch_utc, g.decay_date, g.periapsis_km, g.apoapsis_km, g.inclination_deg,
       g.period_min,
       CASE WHEN g.decay_date IS NULL THEN 1 ELSE 0 END                     AS on_orbit,
       julianday(:window_start) - julianday(g.epoch_utc)                   AS epoch_age_days,
       (SELECT r.regime FROM orbital_regime r
         WHERE g.periapsis_km >  r.min_perigee_km
           AND (r.max_perigee_km IS NULL OR g.periapsis_km <= r.max_perigee_km)) AS regime
FROM gp_element_set g;

-- Objects the screening engine is allowed to propagate: on orbit, LEO perigee,
-- element set no older than 14 days, near-Earth (period < 225 min).
DROP VIEW IF EXISTS v_screenable_object;
CREATE VIEW v_screenable_object AS
SELECT * FROM v_space_object
WHERE regime = 'LEO' AND epoch_age_days <= 14 AND period_min < 225;
