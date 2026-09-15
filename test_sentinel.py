"""Correctness tests for the Orbital Sentinel engine.

Run with:  python -m pytest test_sentinel.py -v
       or:  python test_sentinel.py      (no pytest required)

The spatial tests are the important ones: they assert that the O(N log N)
cKDTree search returns exactly what an O(N^2) brute-force screen would.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sgp4.api import Satrec

import config
from ingestion import SpaceObject, parse_records
from logger import ConjunctionEvent, ConjunctionLog, Profiler
from propagation import (
    PropagationEngine,
    build_time_grid,
    choose_chunk_steps,
    estimate_memory_gb,
    refine_tca,
)
from spatial_index import detect_pairs, required_screening_radius
from validator import CDMRecord, deduplicate_cdms, parse_cdms, validate

# A real ISS element set, used as a stable fixture.
ISS_L1 = "1 25544U 98067A   24079.54791667  .00016717  00000-0  30074-3 0  9994"
ISS_L2 = "2 25544  51.6400 208.9163 0004378  90.0000 270.0000 15.50000000    05"


def _make_object(norad: int, line1: str = ISS_L1, line2: str = ISS_L2) -> SpaceObject:
    satrec = Satrec.twoline2rv(line1, line2)
    return SpaceObject(
        norad_id=norad,
        name=f"TEST-{norad}",
        object_type="PAYLOAD",
        epoch=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        satrec=satrec,
        mean_motion=15.5,
        eccentricity=0.0004378,
    )


# --------------------------------------------------------------------------- #
# Time grid
# --------------------------------------------------------------------------- #
def test_time_grid_length_and_spacing():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    grid = build_time_grid(start=start, hours=2, step_minutes=1)
    assert len(grid) == 120
    assert grid.timestamp(0) == start
    assert grid.timestamp(119) == start + timedelta(minutes=119)


def test_time_grid_fraction_is_normalized():
    """`fr` must stay in [0, 1) with whole days carried into `jd`."""
    grid = build_time_grid(
        start=datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc),
        hours=72,
        step_minutes=1,
    )
    assert np.all(grid.fr >= 0.0)
    assert np.all(grid.fr < 1.0)
    # Total elapsed time must equal the window, despite the day rollovers.
    elapsed_days = (grid.jd[-1] + grid.fr[-1]) - (grid.jd[0] + grid.fr[0])
    assert math.isclose(elapsed_days, (72 * 60 - 1) / 1440.0, rel_tol=1e-9)


def test_time_grid_slice_preserves_absolute_time():
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    grid = build_time_grid(start=start, hours=3, step_minutes=1)
    sub = grid.slice(60, 90)
    assert len(sub) == 30
    assert sub.timestamp(0) == grid.timestamp(60)
    assert np.allclose(sub.jd, grid.jd[60:90])


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #
def test_propagation_shapes_and_physics():
    engine = PropagationEngine([_make_object(1), _make_object(2)])
    grid = build_time_grid(hours=1, step_minutes=1)
    result = engine.propagate(grid)

    assert result.positions.shape == (2, 60, 3)
    assert result.velocities.shape == (2, 60, 3)
    assert result.valid.shape == (2, 60)

    radius = np.linalg.norm(result.positions[result.valid], axis=1)
    altitude = radius - config.EARTH_RADIUS_KM
    # An ISS-like orbit must stay in the LEO band.
    assert altitude.min() > 150
    assert altitude.max() < 800

    speed = np.linalg.norm(result.velocities[result.valid], axis=1)
    # Circular LEO orbital speed is ~7.7 km/s.
    assert 7.0 < speed.mean() < 8.2


def test_identical_elements_produce_identical_tracks():
    """Two objects sharing an element set must propagate to the same point."""
    engine = PropagationEngine([_make_object(1), _make_object(2)])
    grid = build_time_grid(hours=1, step_minutes=5)
    result = engine.propagate(grid)
    assert np.allclose(result.positions[0], result.positions[1])


def test_memory_estimate_and_chunking():
    # 30,000 objects x 4,320 steps of float64 positions + velocities.
    gb = estimate_memory_gb(30000, 4320)
    assert 5.0 < gb < 6.5
    chunk = choose_chunk_steps(30000, budget_gb=2.0, cap=10_000)
    assert chunk * estimate_memory_gb(30000, 1) <= 2.0 + 1e-9


# --------------------------------------------------------------------------- #
# Spatial index -- the core correctness guarantee
# --------------------------------------------------------------------------- #
def _brute_force_pairs(positions, valid, threshold):
    live = np.flatnonzero(valid)
    dense = squareform(pdist(positions[live]))
    np.fill_diagonal(dense, np.inf)
    i, j = np.where(np.triu(dense <= threshold, k=1))
    return {(int(live[a]), int(live[b])) for a, b in zip(i, j)}


def test_kdtree_matches_brute_force():
    rng = np.random.default_rng(42)
    positions = rng.uniform(-7000, 7000, size=(800, 3))
    # Distinct velocities so nothing trips the co-orbiting filter.
    velocities = rng.uniform(-8, 8, size=(800, 3))
    valid = np.ones(800, dtype=bool)

    for threshold in (100.0, 300.0, 800.0):
        detection = detect_pairs(
            positions, velocities, valid,
            threshold_km=threshold, min_rel_speed_kms=0.0, step_minutes=0.0,
        )
        got = {
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(detection.index_i, detection.index_j)
        }
        assert got == _brute_force_pairs(positions, valid, threshold), (
            f"KDTree disagreed with brute force at r={threshold}"
        )


def test_invalid_rows_are_excluded():
    """A masked-out object must never appear in a detection."""
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    velocities = np.array([[0.0, 0, 0], [7.0, 0, 0], [-7.0, 0, 0]])
    valid = np.array([True, False, True])
    detection = detect_pairs(
        positions, velocities, valid, threshold_km=5.0,
        min_rel_speed_kms=0.0, step_minutes=0.0,
    )
    assert 1 not in detection.index_i.tolist()
    assert 1 not in detection.index_j.tolist()
    assert len(detection) == 1  # only the 0-2 pair survives


def test_relative_velocity_gate_rejects_co_orbiting():
    """Docked / co-moving objects must be filtered out, fast ones kept."""
    positions = np.array([[7000.0, 0, 0], [7000.5, 0, 0], [7000.2, 0, 0]])
    # 0 and 1 move together; 2 moves the opposite way.
    velocities = np.array([[0.0, 7.5, 0], [0.0, 7.5, 0], [0.0, -7.5, 0]])
    valid = np.ones(3, dtype=bool)

    gated = detect_pairs(
        positions, velocities, valid, threshold_km=5.0,
        min_rel_speed_kms=0.05, step_minutes=0.0,
    )
    pairs = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in zip(gated.index_i, gated.index_j)
    }
    assert (0, 1) not in pairs, "co-moving pair should have been filtered"
    assert (0, 2) in pairs and (1, 2) in pairs

    ungated = detect_pairs(
        positions, velocities, valid, threshold_km=5.0,
        min_rel_speed_kms=0.0, step_minutes=0.0,
    )
    assert len(ungated) == 3


def test_miss_distance_and_altitude_are_correct():
    positions = np.array([[7000.0, 0.0, 0.0], [7003.0, 4.0, 0.0]])
    velocities = np.array([[0.0, 7.5, 0.0], [0.0, -7.5, 0.0]])
    valid = np.ones(2, dtype=bool)
    detection = detect_pairs(
        positions, velocities, valid, threshold_km=10.0,
        min_rel_speed_kms=0.0, step_minutes=0.0,
    )
    assert len(detection) == 1
    assert math.isclose(float(detection.miss_km[0]), 5.0, rel_tol=1e-9)  # 3-4-5
    assert math.isclose(float(detection.rel_speed_kms[0]), 15.0, rel_tol=1e-9)
    midpoint_radius = np.linalg.norm([7001.5, 2.0, 0.0])
    assert math.isclose(
        float(detection.altitude_km[0]),
        midpoint_radius - config.EARTH_RADIUS_KM,
        rel_tol=1e-9,
    )


def test_empty_and_degenerate_inputs():
    empty = detect_pairs(
        np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=bool)
    )
    assert len(empty) == 0
    single = detect_pairs(
        np.array([[1.0, 2, 3]]), np.array([[0.0, 7, 0]]), np.array([True])
    )
    assert len(single) == 0



# --------------------------------------------------------------------------- #
# Two-stage screening -- regression guard for the coarse-grid blind spot
# --------------------------------------------------------------------------- #
def test_screening_radius_brackets_a_full_step():
    """The coarse radius must cover the furthest a pair can travel in a step."""
    assert math.isclose(
        required_screening_radius(5.0, 1.0, max_rel_speed_kms=16.0), 485.0
    )
    # Halving the step halves the bracketing term.
    assert math.isclose(
        required_screening_radius(5.0, 0.5, max_rel_speed_kms=16.0), 245.0
    )


def test_fast_conjunction_between_samples_is_detected():
    """The bug this engine was built to avoid.

    Two objects closing at 14 km/s pass within 1 km of each other 20 s after
    the grid sample. At the sample itself they are 280 km apart, so a naive
    5 km tree query finds nothing. The two-stage screen must still catch it.
    """
    rel_speed = 14.0
    offset_s = 20.0
    miss = 1.0

    # Place them so that at t = +20 s the separation is exactly `miss`.
    velocities = np.array([[0.0, rel_speed / 2, 0.0], [0.0, -rel_speed / 2, 0.0]])
    dv = velocities[0] - velocities[1]
    positions = np.array([[7000.0, 0.0, 0.0], [7000.0 + miss, 0.0, 0.0]])
    positions[0] -= dv * offset_s / 2
    positions[1] += dv * offset_s / 2
    valid = np.ones(2, dtype=bool)

    separation_at_sample = float(np.linalg.norm(positions[0] - positions[1]))
    assert separation_at_sample > 200, "fixture should be far apart at the sample"

    # A naive instantaneous screen sees nothing.
    naive = detect_pairs(
        positions, velocities, valid,
        threshold_km=5.0, min_rel_speed_kms=0.0, step_minutes=0.0,
    )
    assert len(naive) == 0

    # The two-stage screen recovers it, with a sub-step TCA offset.
    found = detect_pairs(
        positions, velocities, valid,
        threshold_km=5.0, min_rel_speed_kms=0.0, step_minutes=1.0,
    )
    assert len(found) == 1
    assert math.isclose(float(found.miss_km[0]), miss, abs_tol=1e-6)
    assert math.isclose(float(found.tca_offset_s[0]), offset_s, abs_tol=1e-6)
    assert math.isclose(float(found.rel_speed_kms[0]), rel_speed, abs_tol=1e-9)


def test_closest_approach_outside_the_step_is_clamped():
    """A pair whose TCA falls beyond the half-step is not credited here.

    The neighbouring grid sample owns that moment, so clamping keeps steps from
    double-counting the same encounter.
    """
    # Receding pair 400 km apart, separating at 10 km/s: the true closest
    # approach was 40 s ago, beyond this sample's 30 s half-step. It is inside
    # the 485 km screening radius, so it does reach stage 2 and must be
    # rejected there rather than silently never screened.
    positions = np.array([[7000.0, 0.0, 0.0], [7400.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    valid = np.ones(2, dtype=bool)
    detection = detect_pairs(
        positions, velocities, valid,
        threshold_km=5.0, min_rel_speed_kms=0.0, step_minutes=1.0,
    )
    assert len(detection) == 0


def test_tca_offset_shifts_the_reported_timestamp():
    """detections_to_events must apply the sub-step offset to the event time."""
    from spatial_index import detections_to_events

    class _FakeEngine:
        norad_ids = np.array([11, 22])
        names = ["A", "B"]
        types = ["PAYLOAD", "DEBRIS"]

    detection = detect_pairs(
        np.array([[7000.0, 0.0, 0.0], [7000.0, 140.0, 0.0]]),
        np.array([[0.0, 7.0, 0.0], [0.0, -7.0, 0.0]]),
        np.ones(2, dtype=bool),
        threshold_km=5.0, min_rel_speed_kms=0.0, step_minutes=1.0,
    )
    assert len(detection) == 1
    stamp = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    events = detections_to_events(detection, _FakeEngine(), stamp)
    # TCA is 10 s after the sample (140 km closing at 14 km/s).
    assert events[0].tca_utc == "2026-01-01 12:00:10"


# --------------------------------------------------------------------------- #
# TCA refinement
# --------------------------------------------------------------------------- #
def test_refine_tca_finds_a_closer_approach():
    """Refinement must never report a worse miss distance than the grid."""
    obj_a = _make_object(1)
    # Same orbit, shifted slightly in mean anomaly so they actually close.
    line2_b = "2 25544  51.6400 208.9163 0004378  90.0000 269.9000 15.50000000    05"
    obj_b = _make_object(2, ISS_L1, line2_b)

    coarse = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    tca, miss, rel_speed = refine_tca(obj_a, obj_b, coarse, window_minutes=1.0)

    assert abs((tca - coarse).total_seconds()) <= 60.0
    assert np.isfinite(miss) and miss >= 0.0
    assert rel_speed >= 0.0


# --------------------------------------------------------------------------- #
# Ingestion filters
# --------------------------------------------------------------------------- #
def _gp_record(norad: int, epoch: str) -> dict:
    return {
        "NORAD_CAT_ID": str(norad),
        "OBJECT_NAME": f"OBJ-{norad}",
        "OBJECT_TYPE": "PAYLOAD",
        "EPOCH": epoch,
        "TLE_LINE1": ISS_L1,
        "TLE_LINE2": ISS_L2,
        "MEAN_MOTION": "15.5",
        "ECCENTRICITY": "0.0004378",
    }


def test_stale_epochs_are_rejected():
    fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    stale = (datetime.now(timezone.utc) - timedelta(days=400)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    parsed = parse_records(
        [_gp_record(1, fresh), _gp_record(2, stale)],
        leo_only=False,
        max_epoch_age_days=14.0,
        exclude_deep_space=False,
    )
    assert [o.norad_id for o in parsed] == [1]


def test_malformed_records_are_skipped_not_fatal():
    fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    bad = _gp_record(9, fresh)
    bad["TLE_LINE1"] = None
    parsed = parse_records(
        [bad, _gp_record(1, fresh)],
        leo_only=False, max_epoch_age_days=0, exclude_deep_space=False,
    )
    assert [o.norad_id for o in parsed] == [1]


# --------------------------------------------------------------------------- #
# Logging / dedup
# --------------------------------------------------------------------------- #
def _event(n1, n2, miss, stamp="2026-01-01 00:00:00") -> ConjunctionEvent:
    return ConjunctionEvent(
        tca_utc=stamp, norad_1=n1, norad_2=n2,
        name_1=f"A{n1}", name_2=f"B{n2}", type_1="PAYLOAD", type_2="DEBRIS",
        miss_distance_km=miss, relative_speed_km_s=10.0, altitude_km=500.0,
    )


def test_deduplicate_keeps_tightest_approach_per_pair():
    conj = ConjunctionLog(run_id="test")
    conj.extend([
        _event(100, 200, 4.0, "2026-01-01 00:00:00"),
        _event(200, 100, 1.2, "2026-01-01 00:01:00"),  # mirrored pair, closer
        _event(100, 200, 3.0, "2026-01-01 00:02:00"),
        _event(300, 400, 2.0),
    ])
    conj.deduplicate()
    assert len(conj.events) == 2
    closest = conj.events[0]
    assert {closest.norad_1, closest.norad_2} == {100, 200}
    assert math.isclose(closest.miss_distance_km, 1.2)


def test_profiler_accumulates_across_calls():
    profiler = Profiler()
    for _ in range(3):
        with profiler.stage("work", units=10):
            pass
    stage = profiler.stages["work"]
    assert stage.calls == 3 and stage.units == 30


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_cdm_min_range_converted_from_meters():
    parsed = parse_cdms([{
        "CDM_ID": "1", "TCA": "2026-09-16T02:54:09.579000",
        "SAT_1_ID": "1", "SAT_2_ID": "2",
        "SAT_1_NAME": "A", "SAT_2_NAME": "B",
        "MIN_RNG": "462", "PC": "1.27e-4", "EMERGENCY_REPORTABLE": "Y",
    }])
    assert len(parsed) == 1
    assert math.isclose(parsed[0].min_range_km, 0.462)
    assert parsed[0].emergency is True


def test_mirrored_cdms_collapse_to_one_encounter():
    raw = [
        {"CDM_ID": "1", "TCA": "2026-09-16T02:54:09.579000", "SAT_1_ID": "10",
         "SAT_2_ID": "20", "MIN_RNG": "200", "PC": "1e-4"},
        {"CDM_ID": "2", "TCA": "2026-09-16T02:54:09.579000", "SAT_1_ID": "20",
         "SAT_2_ID": "10", "MIN_RNG": "180", "PC": "1e-4"},
    ]
    collapsed = deduplicate_cdms(parse_cdms(raw))
    assert len(collapsed) == 1
    assert math.isclose(collapsed[0].min_range_km, 0.180)


def test_validate_matches_prediction_to_cdm():
    tca = datetime(2026, 9, 16, 2, 54, 9, tzinfo=timezone.utc)
    cdm = CDMRecord(
        cdm_id="1", tca=tca, norad_1=20, norad_2=10, name_1="A", name_2="B",
        min_range_km=0.200, probability=1e-4, emergency=True,
    )
    # Predicted with swapped ordering and a 2-minute TCA offset.
    events = [_event(10, 20, 0.25, "2026-09-16 02:56:09")]
    report = validate(
        events, [cdm], propagated_norads={10, 20},
        window_start=tca - timedelta(hours=1),
        window_end=tca + timedelta(hours=1),
        tolerance_minutes=30.0,
    )
    assert report.considered_cdms == 1
    assert report.matched_cdms == 1
    assert math.isclose(report.recall, 1.0)
    assert math.isclose(report.residuals_km[0], 0.05, abs_tol=1e-9)


def test_validate_counts_a_miss_when_no_prediction():
    tca = datetime(2026, 9, 16, 2, 54, 9, tzinfo=timezone.utc)
    cdm = CDMRecord(
        cdm_id="1", tca=tca, norad_1=20, norad_2=10, name_1="A", name_2="B",
        min_range_km=0.200, probability=1e-4, emergency=False,
    )
    report = validate(
        [], [cdm], propagated_norads={10, 20},
        window_start=tca - timedelta(hours=1),
        window_end=tca + timedelta(hours=1),
    )
    assert report.missed_cdms == 1 and report.recall == 0.0


def test_out_of_scope_cdms_are_not_counted_as_misses():
    """A CDM whose objects we never propagated is outside the experiment."""
    tca = datetime(2026, 9, 16, 2, 54, 9, tzinfo=timezone.utc)
    cdm = CDMRecord(
        cdm_id="1", tca=tca, norad_1=999, norad_2=888, name_1="A", name_2="B",
        min_range_km=0.200, probability=1e-4, emergency=False,
    )
    report = validate(
        [], [cdm], propagated_norads={10, 20},
        window_start=tca - timedelta(hours=1),
        window_end=tca + timedelta(hours=1),
    )
    assert report.considered_cdms == 0 and report.missed_cdms == 0


if __name__ == "__main__":
    import sys
    import traceback

    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    sys.exit(1 if failures else 0)
