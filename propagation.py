"""Vectorized SGP4 propagation via ``sgp4.api.SatrecArray``.

The whole point of this module is to never loop over objects in Python. One
call to :meth:`SatrecArray.sgp4` propagates every object across every requested
timestamp inside compiled C++, returning ``[N, T]`` and ``[N, T, 3]`` arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, Sequence

import numpy as np
from sgp4.api import SatrecArray, jday

import config
from ingestion import SpaceObject
from logger import get_logger

log = get_logger("propagate")


# --------------------------------------------------------------------------- #
# Time grid
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TimeGrid:
    """Julian-date arrays for the forecast window, plus their UTC equivalents."""

    start: datetime
    step_minutes: float
    jd: np.ndarray  # float64 [T] -- integer-ish Julian day
    fr: np.ndarray  # float64 [T] -- fraction of day, in [0, 1)

    def __len__(self) -> int:
        return int(self.jd.shape[0])

    def timestamp(self, index: int) -> datetime:
        """UTC datetime of time step ``index``."""
        return self.start + timedelta(minutes=self.step_minutes * index)

    def slice(self, lo: int, hi: int) -> "TimeGrid":
        """A sub-grid covering steps ``[lo, hi)``, keeping absolute timestamps."""
        return TimeGrid(
            start=self.timestamp(lo),
            step_minutes=self.step_minutes,
            jd=self.jd[lo:hi],
            fr=self.fr[lo:hi],
        )


def build_time_grid(
    start: datetime | None = None,
    hours: float | None = None,
    step_minutes: float | None = None,
) -> TimeGrid:
    """Build the Julian-date grid for a rolling forecast window."""
    start = start or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    hours = config.FORECAST_WINDOW_HOURS if hours is None else hours
    step_minutes = config.TIME_STEP_MINUTES if step_minutes is None else step_minutes

    steps = int(round(hours * 60.0 / step_minutes))
    if steps < 1:
        raise ValueError("The forecast window must contain at least one time step.")

    jd0, fr0 = jday(
        start.year,
        start.month,
        start.day,
        start.hour,
        start.minute,
        start.second + start.microsecond / 1e6,
    )

    # Keep `fr` inside [0, 1) and push whole days into `jd`. SGP4 accepts an
    # unnormalized fraction, but splitting this way preserves float64 precision
    # across a multi-day window.
    offsets = np.arange(steps, dtype=np.float64) * (step_minutes / 1440.0)
    fr = fr0 + offsets
    whole = np.floor(fr)
    jd = np.full(steps, jd0, dtype=np.float64) + whole
    fr = fr - whole

    log.info(
        "Time grid: %s -> %s | %d steps @ %.2f min",
        start.strftime("%Y-%m-%d %H:%M:%SZ"),
        (start + timedelta(minutes=step_minutes * (steps - 1))).strftime(
            "%Y-%m-%d %H:%M:%SZ"
        ),
        steps,
        step_minutes,
    )
    return TimeGrid(start=start, step_minutes=step_minutes, jd=jd, fr=fr)


# --------------------------------------------------------------------------- #
# Propagation engine
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PropagationResult:
    """State vectors for one chunk of the time grid.

    ``positions`` and ``velocities`` are ``[N, T, 3]`` in the TEME frame (km,
    km/s). ``valid`` is ``[N, T]`` and is False wherever SGP4 reported an error
    (decayed object, elements propagated too far past epoch, etc).
    """

    positions: np.ndarray
    velocities: np.ndarray
    valid: np.ndarray
    grid: TimeGrid

    @property
    def n_objects(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.positions.shape[1])

    @property
    def state_vectors(self) -> int:
        return self.n_objects * self.n_steps

    def step(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Positions, velocities and validity mask at one time step."""
        return (
            self.positions[:, index, :],
            self.velocities[:, index, :],
            self.valid[:, index],
        )


class PropagationEngine:
    """Wraps a ``SatrecArray`` built once and reused across the whole run."""

    def __init__(self, objects: Sequence[SpaceObject]) -> None:
        if not objects:
            raise ValueError("Cannot build a propagation engine with zero objects.")
        self.objects = list(objects)
        self.norad_ids = np.array([o.norad_id for o in self.objects], dtype=np.int64)
        self.names = [o.name for o in self.objects]
        self.types = [o.object_type for o in self.objects]

        log.info("Building SatrecArray over %d objects.", len(self.objects))
        self.sat_array = SatrecArray([o.satrec for o in self.objects])

    def __len__(self) -> int:
        return len(self.objects)

    def propagate(self, grid: TimeGrid) -> PropagationResult:
        """Propagate every object across every step of ``grid`` in one call."""
        errors, positions, velocities = self.sat_array.sgp4(grid.jd, grid.fr)

        # Error code 0 means a usable state vector; anything else is discarded.
        valid = errors == 0
        invalid = int(valid.size - np.count_nonzero(valid))
        if invalid:
            log.debug(
                "%d of %d state vectors returned an SGP4 error and were dropped.",
                invalid,
                valid.size,
            )
        return PropagationResult(
            positions=positions, velocities=velocities, valid=valid, grid=grid
        )

    def propagate_chunked(
        self, grid: TimeGrid, chunk_steps: int | None = None
    ) -> Iterator[tuple[int, PropagationResult]]:
        """Yield ``(offset, result)`` for successive slices of the time grid.

        Propagating all 4,320 steps of a 72-hour window at once would allocate
        ``N x T x 3 x 8`` bytes twice over (~14 GB at N=30,000). Chunking bounds
        peak memory while keeping every call fully vectorized.
        """
        chunk_steps = chunk_steps or config.TIME_CHUNK_STEPS
        total = len(grid)
        for lo in range(0, total, chunk_steps):
            hi = min(lo + chunk_steps, total)
            yield lo, self.propagate(grid.slice(lo, hi))


def estimate_memory_gb(n_objects: int, n_steps: int, dtype_bytes: int = 8) -> float:
    """Peak array memory for one propagation call: positions + velocities."""
    return 2 * n_objects * n_steps * 3 * dtype_bytes / (1024 ** 3)


def choose_chunk_steps(
    n_objects: int, budget_gb: float = 2.0, cap: int | None = None
) -> int:
    """Largest chunk size whose position+velocity arrays fit in ``budget_gb``."""
    per_step_gb = estimate_memory_gb(n_objects, 1)
    steps = max(1, int(budget_gb / per_step_gb)) if per_step_gb > 0 else 1
    cap = cap or config.TIME_CHUNK_STEPS
    return max(1, min(steps, cap))


# --------------------------------------------------------------------------- #
# Sub-step Time of Closest Approach refinement
# --------------------------------------------------------------------------- #
def _pair_separation(
    sat_a, sat_b, jd: float, fr: float
) -> tuple[float, float]:
    """Miss distance (km) and relative speed (km/s) for one pair at one instant."""
    err_a, r_a, v_a = sat_a.sgp4(jd, fr)
    err_b, r_b, v_b = sat_b.sgp4(jd, fr)
    if err_a != 0 or err_b != 0:
        return float("inf"), 0.0
    dr = np.subtract(r_a, r_b)
    dv = np.subtract(v_a, v_b)
    return float(np.linalg.norm(dr)), float(np.linalg.norm(dv))


def refine_tca(
    obj_a: SpaceObject,
    obj_b: SpaceObject,
    coarse_time: datetime,
    window_minutes: float | None = None,
    tolerance_seconds: float = 0.05,
) -> tuple[datetime, float, float]:
    """Resolve the true TCA below the coarse step size for a single pair.

    A 1-minute grid can miss the real closest approach by several km, because
    two LEO objects close at up to ~15 km/s. This brackets the coarse detection
    and runs a golden-section search on the separation function, which is
    smooth and unimodal inside a single encounter.

    Returns ``(tca_utc, miss_distance_km, relative_speed_km_s)``.
    """
    window_minutes = window_minutes or config.TIME_STEP_MINUTES
    sat_a, sat_b = obj_a.satrec, obj_b.satrec

    base = coarse_time.replace(tzinfo=coarse_time.tzinfo or timezone.utc)
    jd_base, fr_base = jday(
        base.year,
        base.month,
        base.day,
        base.hour,
        base.minute,
        base.second + base.microsecond / 1e6,
    )

    def separation(offset_seconds: float) -> float:
        fr = fr_base + offset_seconds / 86400.0
        whole = np.floor(fr)
        distance, _ = _pair_separation(sat_a, sat_b, jd_base + whole, fr - whole)
        return distance

    # Golden-section search over the bracketing window.
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    lo = -window_minutes * 60.0
    hi = window_minutes * 60.0
    c = hi - invphi * (hi - lo)
    d = lo + invphi * (hi - lo)
    f_c, f_d = separation(c), separation(d)

    while abs(hi - lo) > tolerance_seconds:
        if f_c < f_d:
            hi, d, f_d = d, c, f_c
            c = hi - invphi * (hi - lo)
            f_c = separation(c)
        else:
            lo, c, f_c = c, d, f_d
            d = lo + invphi * (hi - lo)
            f_d = separation(d)

    best_offset = (lo + hi) / 2.0
    fr = fr_base + best_offset / 86400.0
    whole = np.floor(fr)
    distance, rel_speed = _pair_separation(sat_a, sat_b, jd_base + whole, fr - whole)
    return base + timedelta(seconds=best_offset), distance, rel_speed
