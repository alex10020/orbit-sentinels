"""Multiprocessing worker pool for time-chunk parallel conjunction screening.

The 72-hour window splits cleanly along the time axis: every chunk is
independent, needs no state from its neighbours, and produces its own event
list. That makes the pipeline embarrassingly parallel across CPU cores.

Two constraints shape this module:

1. **``Satrec`` objects cannot be pickled.** So the parent cannot ship a built
   ``SatrecArray`` to the workers. Instead each worker receives the raw TLE
   lines once, through the pool initializer, and builds its own
   ``SatrecArray`` in its own address space. That cost is paid once per worker,
   not once per chunk.
2. **Windows uses ``spawn``, not ``fork``.** Workers re-import this module from
   scratch, so everything they touch must be importable at module level -- no
   closures, no lambdas. The caller must also guard its entry point with
   ``if __name__ == "__main__"``.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Sequence

import numpy as np
from sgp4.api import Satrec, SatrecArray

from logger import ConjunctionEvent, get_logger

log = get_logger("parallel")


# --------------------------------------------------------------------------- #
# Payloads exchanged with workers (all picklable)
# --------------------------------------------------------------------------- #
@dataclass
class CatalogPayload:
    """The catalog in a form that survives pickling to a worker process."""

    norad_ids: list[int]
    names: list[str]
    types: list[str]
    line1: list[str]
    line2: list[str]

    @classmethod
    def from_objects(cls, objects: Sequence[Any]) -> "CatalogPayload":
        return cls(
            norad_ids=[o.norad_id for o in objects],
            names=[o.name for o in objects],
            types=[o.object_type for o in objects],
            line1=[o.tle_line1 for o in objects],
            line2=[o.tle_line2 for o in objects],
        )

    def __len__(self) -> int:
        return len(self.norad_ids)


@dataclass
class ChunkTask:
    """One contiguous slice of the forecast window."""

    offset: int  # absolute index of the first step
    jd: np.ndarray
    fr: np.ndarray
    start_iso: str  # UTC timestamp of this chunk's first step
    step_minutes: float
    threshold_km: float
    min_rel_speed_kms: float

    @property
    def n_steps(self) -> int:
        return int(self.jd.shape[0])


# --------------------------------------------------------------------------- #
# Worker-side state
# --------------------------------------------------------------------------- #
# Each worker process holds exactly one rebuilt engine, created by the pool
# initializer and reused for every chunk that worker handles.
_WORKER_SAT_ARRAY: SatrecArray | None = None
_WORKER_META: CatalogPayload | None = None


def init_worker(payload: CatalogPayload) -> None:
    """Pool initializer: rebuild the SatrecArray inside this worker process."""
    global _WORKER_SAT_ARRAY, _WORKER_META
    satrecs = [
        Satrec.twoline2rv(l1, l2) for l1, l2 in zip(payload.line1, payload.line2)
    ]
    _WORKER_SAT_ARRAY = SatrecArray(satrecs)
    _WORKER_META = payload


def process_chunk(task: ChunkTask) -> list[dict[str, Any]]:
    """Propagate and screen one time chunk. Returns picklable event rows.

    Imported lazily so that a worker re-importing this module under ``spawn``
    does not pull in the whole pipeline before the initializer has run.
    """
    from spatial_index import detect_pairs

    if _WORKER_SAT_ARRAY is None or _WORKER_META is None:
        raise RuntimeError("Worker was not initialized via init_worker().")

    errors, positions, velocities = _WORKER_SAT_ARRAY.sgp4(task.jd, task.fr)
    valid = errors == 0

    start = datetime.fromisoformat(task.start_iso)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    meta = _WORKER_META
    rows: list[dict[str, Any]] = []

    for local_step in range(task.n_steps):
        detection = detect_pairs(
            positions[:, local_step, :],
            velocities[:, local_step, :],
            valid[:, local_step],
            threshold_km=task.threshold_km,
            min_rel_speed_kms=task.min_rel_speed_kms,
            step_minutes=task.step_minutes,
        )
        if len(detection) == 0:
            continue

        stamp = start + timedelta(minutes=task.step_minutes * local_step)
        for k in range(len(detection)):
            i = int(detection.index_i[k])
            j = int(detection.index_j[k])
            tca = stamp + timedelta(seconds=float(detection.tca_offset_s[k]))
            rows.append(
                {
                    "tca_utc": tca.strftime("%Y-%m-%d %H:%M:%S"),
                    "norad_1": meta.norad_ids[i],
                    "norad_2": meta.norad_ids[j],
                    "name_1": meta.names[i],
                    "name_2": meta.names[j],
                    "type_1": meta.types[i],
                    "type_2": meta.types[j],
                    "miss_distance_km": float(detection.miss_km[k]),
                    "relative_speed_km_s": float(detection.rel_speed_kms[k]),
                    "altitude_km": float(detection.altitude_km[k]),
                    "x_km": float(detection.midpoint_km[k, 0]),
                    "y_km": float(detection.midpoint_km[k, 1]),
                    "z_km": float(detection.midpoint_km[k, 2]),
                }
            )

    return rows


# --------------------------------------------------------------------------- #
# Parent-side driver
# --------------------------------------------------------------------------- #
def resolve_worker_count(requested: int | None = None) -> int:
    """How many worker processes to start.

    ``0`` or ``None`` means "all cores". The count is capped at the CPU count
    because this workload is CPU and memory-bandwidth bound, so oversubscribing
    only adds context switching.
    """
    available = os.cpu_count() or 1
    if not requested:
        return available
    return max(1, min(requested, available))


# A chunk must stay wide enough that one SGP4 call still covers many steps;
# splitting down to single steps throws away the vectorization the engine is
# built on and leaves only per-task pickling overhead.
MIN_CHUNK_STEPS = 8


def plan_chunks(
    n_steps: int,
    n_workers: int,
    memory_cap_steps: int,
    tasks_per_worker: int = 2,
) -> int:
    """Chunk size that keeps every worker busy without breaching the memory cap.

    Sizing chunks purely by memory leaves too few of them to distribute: a
    120-step cap over a 30-step window yields a single chunk, so one worker does
    everything while the rest idle. Aiming for a few tasks per worker also gives
    the pool something to rebalance with when chunks finish unevenly.
    """
    if n_steps <= 0 or n_workers <= 0:
        return max(1, memory_cap_steps)
    target = n_steps // (n_workers * tasks_per_worker)
    target = max(MIN_CHUNK_STEPS, target)
    return max(1, min(memory_cap_steps, target))


def worth_parallelizing(
    n_objects: int, n_steps: int, n_workers: int, min_state_vectors: int = 2_000_000
) -> bool:
    """Whether the job is big enough to repay process startup.

    Each worker rebuilds the whole SatrecArray from TLE text, which costs about
    a second at full catalog size. Below roughly a couple of million state
    vectors the single-process path finishes before a pool has even started.
    """
    if n_workers <= 1 or n_steps < 2:
        return False
    return n_objects * n_steps >= min_state_vectors


def build_tasks(
    grid,
    chunk_steps: int,
    threshold_km: float,
    min_rel_speed_kms: float,
) -> list[ChunkTask]:
    """Slice the forecast window into independent chunks."""
    tasks: list[ChunkTask] = []
    total = len(grid)
    for offset in range(0, total, chunk_steps):
        end = min(offset + chunk_steps, total)
        tasks.append(
            ChunkTask(
                offset=offset,
                jd=grid.jd[offset:end].copy(),
                fr=grid.fr[offset:end].copy(),
                start_iso=grid.timestamp(offset).isoformat(),
                step_minutes=grid.step_minutes,
                threshold_km=threshold_km,
                min_rel_speed_kms=min_rel_speed_kms,
            )
        )
    return tasks


def run_parallel(
    objects: Sequence[Any],
    grid,
    chunk_steps: int,
    threshold_km: float,
    min_rel_speed_kms: float,
    workers: int | None = None,
    show_progress: bool = True,
) -> tuple[list[ConjunctionEvent], int]:
    """Screen the whole window across a process pool.

    Returns ``(events, steps_completed)``. Chunks come back out of order --
    they are independent, and the event list is sorted downstream anyway.
    """
    n_workers = resolve_worker_count(workers)
    payload = CatalogPayload.from_objects(objects)
    tasks = build_tasks(grid, chunk_steps, threshold_km, min_rel_speed_kms)

    log.info(
        "Dispatching %d chunks of <=%d steps across %d worker processes "
        "(%d cores available).",
        len(tasks),
        chunk_steps,
        n_workers,
        os.cpu_count() or 1,
    )

    events: list[ConjunctionEvent] = []
    steps_done = 0

    with mp.Pool(
        processes=n_workers, initializer=init_worker, initargs=(payload,)
    ) as pool:
        results: Iterator[list[dict[str, Any]]] = pool.imap_unordered(
            process_chunk, tasks
        )
        results = _with_progress(results, tasks, show_progress)
        for rows in results:
            events.extend(ConjunctionEvent(**row) for row in rows)

    steps_done = sum(t.n_steps for t in tasks)
    return events, steps_done


def _with_progress(results, tasks, show_progress: bool):
    """Wrap the result iterator in a tqdm bar when tqdm is available."""
    if not show_progress:
        return results
    try:
        from tqdm import tqdm
    except ImportError:  # progress is a convenience, never a dependency
        return results
    return tqdm(
        results,
        total=len(tasks),
        desc="Screening chunks",
        unit="chunk",
        dynamic_ncols=True,
    )
