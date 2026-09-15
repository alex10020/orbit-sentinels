"""3D spatial partitioning: cKDTree conjunction search over TEME state vectors.

Two-stage screening
-------------------
Querying the tree directly at the 5 km miss-distance threshold does not work on
a coarse time grid, and the failure is silent. Two objects closing at 14 km/s
travel 840 km between 1-minute samples, so the nearest grid sample to their true
closest approach can sit hundreds of km apart. A naive 5 km query therefore only
ever finds encounters slower than ~0.17 km/s -- which excludes essentially every
dangerous debris conjunction.

So the search runs in two stages:

1. **Coarse pass.** Query the tree at a *screening radius* large enough to
   bracket any encounter reachable within one time step:
   ``R = threshold + v_max * dt / 2`` (~485 km for a 1-minute step). Any pair
   that conjuncts inside this step must appear here.
2. **Fine pass.** For each candidate, solve for the closest approach assuming
   linear relative motion across the step -- a closed-form, fully vectorized
   NumPy operation -- and keep only pairs whose true miss distance falls inside
   the threshold.

Stage 2 reduces ~300,000 candidates per step to a handful, and the surviving
TCA estimates are already sub-step accurate. ``propagation.refine_tca`` then
polishes the closest ones with direct SGP4 evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
from scipy.spatial import cKDTree

import config
from logger import ConjunctionEvent, get_logger
from propagation import PropagationEngine, PropagationResult

log = get_logger("spatial")


def required_screening_radius(
    threshold_km: float | None = None,
    step_minutes: float | None = None,
    max_rel_speed_kms: float | None = None,
) -> float:
    """Tree radius that provably brackets any encounter within one time step.

    Each grid sample owns the half-step either side of it, so the worst-case
    offset from a true TCA is ``dt / 2``. A pair closing at ``max_rel_speed``
    can be no further than ``threshold + v_max * dt / 2`` from its counterpart
    at the nearest sample.
    """
    threshold_km = (
        config.CONJUNCTION_THRESHOLD_KM if threshold_km is None else threshold_km
    )
    step_minutes = config.TIME_STEP_MINUTES if step_minutes is None else step_minutes
    max_rel_speed_kms = (
        config.MAX_RELATIVE_SPEED_KM_S
        if max_rel_speed_kms is None
        else max_rel_speed_kms
    )
    return threshold_km + max_rel_speed_kms * (step_minutes * 60.0) / 2.0


@dataclass(slots=True)
class PairDetection:
    """Close-approach pairs found around a single time step.

    All arrays share length M. ``index_i`` / ``index_j`` address rows of the
    full object catalog. ``tca_offset_s`` is the signed offset from the grid
    sample to the estimated closest approach, and ``miss_km`` is the distance
    at that offset -- not at the grid sample itself.
    """

    index_i: np.ndarray
    index_j: np.ndarray
    miss_km: np.ndarray
    rel_speed_kms: np.ndarray
    altitude_km: np.ndarray
    tca_offset_s: np.ndarray
    midpoint_km: np.ndarray  # [M, 3] TEME position of the encounter

    def __len__(self) -> int:
        return int(self.index_i.shape[0])


def _empty_detection() -> PairDetection:
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=np.float64)
    return PairDetection(
        index_i=empty_i,
        index_j=empty_i.copy(),
        miss_km=empty_f,
        rel_speed_kms=empty_f.copy(),
        altitude_km=empty_f.copy(),
        tca_offset_s=empty_f.copy(),
        midpoint_km=np.empty((0, 3), dtype=np.float64),
    )


def detect_pairs(
    positions: np.ndarray,
    velocities: np.ndarray,
    valid: np.ndarray,
    threshold_km: float | None = None,
    min_rel_speed_kms: float | None = None,
    step_minutes: float | None = None,
    screening_radius_km: float | None = None,
) -> PairDetection:
    """Find every pair whose closest approach within this step is under threshold.

    ``positions`` and ``velocities`` are ``[N, 3]`` TEME arrays; ``valid`` is
    the ``[N]`` boolean mask of objects SGP4 propagated successfully.

    Pairs closing slower than ``min_rel_speed_kms`` are discarded: docked
    station modules, co-deployed payloads and duplicate element sets otherwise
    saturate the results with 0 km / 0 km/s "conjunctions" between what is
    physically a single structure.
    """
    threshold_km = (
        config.CONJUNCTION_THRESHOLD_KM if threshold_km is None else threshold_km
    )
    min_rel_speed_kms = (
        config.MIN_RELATIVE_SPEED_KM_S
        if min_rel_speed_kms is None
        else min_rel_speed_kms
    )
    step_minutes = config.TIME_STEP_MINUTES if step_minutes is None else step_minutes
    if screening_radius_km is None:
        screening_radius_km = required_screening_radius(threshold_km, step_minutes)

    # Compact to valid rows only: an invalid row would otherwise sit at whatever
    # garbage coordinate SGP4 left behind and pollute the tree.
    live_idx = np.flatnonzero(valid)
    if live_idx.size < 2:
        return _empty_detection()

    live_pos = np.ascontiguousarray(positions[live_idx])

    # -- Stage 1: coarse O(N log N) screen at the bracketing radius --------- #
    tree = cKDTree(live_pos)
    pairs = tree.query_pairs(r=screening_radius_km, output_type="ndarray")
    if pairs.shape[0] == 0:
        return _empty_detection()

    i_local, j_local = pairs[:, 0], pairs[:, 1]
    gi = live_idx[i_local]
    gj = live_idx[j_local]

    # -- Stage 2: closed-form closest approach, vectorized over candidates -- #
    dr = live_pos[i_local] - live_pos[j_local]
    dv = velocities[gi] - velocities[gj]

    rel_speed = np.linalg.norm(dv, axis=1)
    speed_sq = np.einsum("ij,ij->i", dv, dv)

    # t* minimizing |dr + dv*t| is -(dr.dv)/|dv|^2. Guard the degenerate
    # co-moving case, where the separation is constant and t* is undefined.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = -np.einsum("ij,ij->i", dr, dv) / speed_sq
    t_star = np.where(speed_sq > 0, t_star, 0.0)

    # Each sample owns the half-step either side, so consecutive steps tile the
    # timeline without gaps or double-counting.
    half_step_s = step_minutes * 60.0 / 2.0
    t_star = np.clip(t_star, -half_step_s, half_step_s)

    closest = dr + dv * t_star[:, None]
    miss = np.linalg.norm(closest, axis=1)

    keep = miss <= threshold_km
    if min_rel_speed_kms > 0:
        keep &= rel_speed >= min_rel_speed_kms
    if not keep.any():
        return _empty_detection()

    gi, gj = gi[keep], gj[keep]
    miss = miss[keep]
    rel_speed = rel_speed[keep]
    t_star = t_star[keep]

    # Midpoint altitude at the moment of closest approach.
    pos_i = live_pos[i_local[keep]] + velocities[gi] * t_star[:, None]
    pos_j = live_pos[j_local[keep]] + velocities[gj] * t_star[:, None]
    midpoint = 0.5 * (pos_i + pos_j)
    altitude = np.linalg.norm(midpoint, axis=1) - config.EARTH_RADIUS_KM

    return PairDetection(
        index_i=gi,
        index_j=gj,
        miss_km=miss,
        rel_speed_kms=rel_speed,
        altitude_km=altitude,
        tca_offset_s=t_star,
        midpoint_km=midpoint,
    )


def detections_to_events(
    detection: PairDetection,
    engine: PropagationEngine,
    timestamp: datetime,
) -> list[ConjunctionEvent]:
    """Attach catalog metadata to raw pair indices."""
    events: list[ConjunctionEvent] = []
    for k in range(len(detection)):
        i = int(detection.index_i[k])
        j = int(detection.index_j[k])
        tca = timestamp + timedelta(seconds=float(detection.tca_offset_s[k]))
        events.append(
            ConjunctionEvent(
                tca_utc=tca.strftime("%Y-%m-%d %H:%M:%S"),
                norad_1=int(engine.norad_ids[i]),
                norad_2=int(engine.norad_ids[j]),
                name_1=engine.names[i],
                name_2=engine.names[j],
                type_1=engine.types[i],
                type_2=engine.types[j],
                miss_distance_km=float(detection.miss_km[k]),
                relative_speed_km_s=float(detection.rel_speed_kms[k]),
                altitude_km=float(detection.altitude_km[k]),
                x_km=float(detection.midpoint_km[k, 0]),
                y_km=float(detection.midpoint_km[k, 1]),
                z_km=float(detection.midpoint_km[k, 2]),
            )
        )
    return events


def scan_result(
    result: PropagationResult,
    engine: PropagationEngine,
    step_offset: int = 0,
    threshold_km: float | None = None,
    min_rel_speed_kms: float | None = None,
    screening_radius_km: float | None = None,
) -> Iterator[tuple[int, datetime, PairDetection]]:
    """Walk every time step in a propagation chunk, yielding detections.

    Yields ``(absolute_step_index, utc_timestamp, detection)``. Steps with no
    close approach are skipped, which is the overwhelming majority of them.
    """
    for local_step in range(result.n_steps):
        pos, vel, valid = result.step(local_step)
        detection = detect_pairs(
            pos,
            vel,
            valid,
            threshold_km=threshold_km,
            min_rel_speed_kms=min_rel_speed_kms,
            step_minutes=result.grid.step_minutes,
            screening_radius_km=screening_radius_km,
        )
        if len(detection) == 0:
            continue
        absolute = step_offset + local_step
        yield absolute, result.grid.timestamp(local_step), detection


def query_statistics(
    positions: np.ndarray, valid: np.ndarray, threshold_km: float | None = None
) -> dict[str, float]:
    """Diagnostics for one time step: tree size and the brute-force it avoided."""
    threshold_km = (
        config.CONJUNCTION_THRESHOLD_KM if threshold_km is None else threshold_km
    )
    n_live = int(np.count_nonzero(valid))
    brute_force_pairs = n_live * (n_live - 1) / 2
    return {
        "objects": n_live,
        "brute_force_pairs": brute_force_pairs,
        "threshold_km": threshold_km,
    }
