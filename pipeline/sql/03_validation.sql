-- Orbital Sentinel -- validation and downstream queries
-- Run:  python pipeline/validate.py --db db/orbital_sentinel.db
-- Each query is preceded by a "-- name:" line; validate.py prints every result.

-- name: V1 load completeness (rows in DB vs rows read per run)
SELECT r.source, r.file_name, r.rows_read, r.rows_loaded, r.rows_rejected,
       CASE r.source WHEN 'gp' THEN (SELECT COUNT(*) FROM gp_element_set)
                     WHEN 'boxscore' THEN (SELECT COUNT(*) FROM country_boxscore)
                     WHEN 'alerts' THEN (SELECT COUNT(*) FROM conjunction_alert)
                     WHEN 'cdm' THEN (SELECT COUNT(*) FROM cdm_public) END AS rows_in_table
FROM ingestion_run r WHERE r.status = 'succeeded' ORDER BY r.started_utc;

-- name: V2 uniqueness of business keys
SELECT 'gp: NORAD ids with >1 element set' AS check_name,
       COUNT(*) AS violations FROM (SELECT norad_cat_id FROM gp_element_set GROUP BY 1 HAVING COUNT(*) > 1)
UNION ALL
SELECT 'alerts: duplicate unordered pairs per engine run',
       COUNT(*) FROM (SELECT engine_run_id, MIN(norad_1,norad_2), MAX(norad_1,norad_2)
                      FROM conjunction_alert GROUP BY 1,2,3 HAVING COUNT(*) > 1);

-- name: V3 completeness of optional GP attributes (share of NULLs)
SELECT ROUND(100.0*SUM(country_code IS NULL)/COUNT(*),1) AS pct_no_country,
       ROUND(100.0*SUM(rcs_size IS NULL)/COUNT(*),1)     AS pct_no_rcs,
       ROUND(100.0*SUM(launch_date IS NULL)/COUNT(*),1)  AS pct_no_launch_date,
       ROUND(100.0*SUM(periapsis_km IS NULL)/COUNT(*),1) AS pct_no_periapsis
FROM gp_element_set;

-- name: V4 referential integrity: every alert object exists in the catalog
SELECT COUNT(*) AS alert_rows,
       SUM(g1.norad_cat_id IS NULL) AS missing_object_1,
       SUM(g2.norad_cat_id IS NULL) AS missing_object_2
FROM conjunction_alert a
LEFT JOIN gp_element_set g1 ON g1.norad_cat_id = a.norad_1
LEFT JOIN gp_element_set g2 ON g2.norad_cat_id = a.norad_2;

-- name: V5 cross-source reconciliation: on-orbit objects, catalog vs boxscore
SELECT (SELECT COUNT(*) FROM gp_element_set WHERE decay_date IS NULL) AS gp_on_orbit,
       (SELECT SUM(orbital_total_count) FROM country_boxscore)       AS boxscore_on_orbit,
       (SELECT COUNT(*) FROM gp_element_set WHERE decay_date IS NULL)
     - (SELECT SUM(orbital_total_count) FROM country_boxscore)       AS difference;

-- name: V6 reconciliation by country (largest absolute differences)
WITH gp AS (
    SELECT country_code, COUNT(*) AS gp_on_orbit FROM gp_element_set
    WHERE decay_date IS NULL GROUP BY country_code)
SELECT b.spadoc_cd, b.orbital_total_count AS boxscore, COALESCE(gp.gp_on_orbit,0) AS gp,
       COALESCE(gp.gp_on_orbit,0) - b.orbital_total_count AS diff
FROM country_boxscore b LEFT JOIN gp ON gp.country_code = b.spadoc_cd
ORDER BY ABS(diff) DESC LIMIT 6;

-- name: V7 engine reconciliation: SQL screening filter vs objects the engine propagated (28,716)
SELECT COUNT(*) AS sql_screenable, 28716 AS engine_screened, COUNT(*) - 28716 AS difference
FROM v_screenable_object;

-- name: V8 physical plausibility of alerts (engine assumes closing speed <= 16 km/s)
SELECT COUNT(*) AS alerts, MIN(miss_distance_km) AS min_miss_km, MAX(miss_distance_km) AS max_miss_km,
       ROUND(MAX(relative_speed_km_s),3) AS max_rel_kms, SUM(relative_speed_km_s > 16) AS above_vmax,
       MIN(altitude_km) AS min_alt_km, MAX(altitude_km) AS max_alt_km
FROM conjunction_alert;

-- name: Q1 downstream: ten closest approaches with owner countries
SELECT a.tca_utc, a.name_1, g1.country_code AS c1, a.name_2, g2.country_code AS c2,
       ROUND(a.miss_distance_km*1000) AS miss_m, ROUND(a.relative_speed_km_s,2) AS rel_kms
FROM conjunction_alert a
JOIN gp_element_set g1 ON g1.norad_cat_id = a.norad_1
JOIN gp_element_set g2 ON g2.norad_cat_id = a.norad_2
ORDER BY a.miss_distance_km LIMIT 10;

-- name: Q2 downstream: sub-km encounters by owner pair
SELECT MIN(g1.country_code, g2.country_code) || ' x ' || MAX(g1.country_code, g2.country_code) AS owner_pair,
       COUNT(*) AS encounters_under_1km
FROM conjunction_alert a
JOIN gp_element_set g1 ON g1.norad_cat_id = a.norad_1
JOIN gp_element_set g2 ON g2.norad_cat_id = a.norad_2
WHERE a.miss_distance_km < 1
GROUP BY 1 ORDER BY 2 DESC LIMIT 8;

-- name: Q3 downstream: hourly encounter load using the SQL-generated time grid
SELECT t.hour_of_run, COUNT(a.alert_id) AS encounters,
       SUM(a.type_1 = 'DEBRIS' OR a.type_2 = 'DEBRIS') AS involving_debris
FROM (SELECT DISTINCT hour_of_run, MIN(t_utc) AS h0, MAX(t_utc) AS h1 FROM forecast_time_grid GROUP BY hour_of_run) t
LEFT JOIN conjunction_alert a ON a.tca_utc >= t.h0 AND a.tca_utc < datetime(t.h1, '+1 minute')
GROUP BY t.hour_of_run ORDER BY t.hour_of_run LIMIT 6;

-- name: Q4 downstream: catalog by regime and type (on-orbit objects)
SELECT regime, object_type, COUNT(*) AS objects
FROM v_space_object WHERE on_orbit = 1
GROUP BY 1, 2 ORDER BY 1, 3 DESC;
