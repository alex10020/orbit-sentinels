"""Project Orbital Sentinel: conjunction forecast pipeline orchestrator.

Pipeline: Ingestion -> SatrecArray -> cKDTree -> Conjunction Log -> Metrics.

Examples
--------
    python main.py                       # 10-minute smoke test, first 1000 objects
    python main.py --full                # 72-hour window over the whole LEO catalog
    python main.py --hours 6 --max-objects 5000
    python main.py --full --refresh      # force a fresh catalog download
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

import config
from ingestion import catalog_breakdown, load_catalog
from logger import ConjunctionLog, Profiler, get_logger, memory_mb
from propagation import (
    PropagationEngine,
    build_time_grid,
    choose_chunk_steps,
    estimate_memory_gb,
    refine_tca,
)
from parallel import (
    plan_chunks,
    resolve_worker_count,
    run_parallel,
    worth_parallelizing,
)
from spatial_index import (
    detections_to_events,
    required_screening_radius,
    scan_result,
)
from validator import run_validation

log = get_logger("main")

# Defaults for the smoke test described in the spec.
DEMO_HOURS = 10 / 60
DEMO_MAX_OBJECTS = 1000


def _progress_bar(iterable, enabled: bool):
    """Wrap an iterable in tqdm when available; tqdm is never required."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable, desc="Screening chunks", unit="chunk", dynamic_ncols=True
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="orbital-sentinel",
        description="Vectorized LEO conjunction forecasting over a rolling window.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Run the full {config.FORECAST_WINDOW_HOURS}h forecast over the "
        "entire LEO catalog.",
    )
    parser.add_argument(
        "--hours", type=float, default=None,
        help="Forecast window length in hours (default: 10 minutes, or 72 with --full).",
    )
    parser.add_argument(
        "--step", type=float, default=config.TIME_STEP_MINUTES,
        help=f"Time step in minutes (default: {config.TIME_STEP_MINUTES}).",
    )
    parser.add_argument(
        "--max-objects", type=int, default=None,
        help="Cap the number of propagated objects (default: 1000, all with --full).",
    )
    parser.add_argument(
        "--threshold", type=float, default=config.CONJUNCTION_THRESHOLD_KM,
        help=f"Screening radius in km (default: {config.CONJUNCTION_THRESHOLD_KM}).",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force a fresh catalog download instead of using the local cache.",
    )
    parser.add_argument(
        "--all-regimes", action="store_true",
        help="Do not restrict the catalog to objects with a LEO perigee.",
    )
    parser.add_argument(
        "--min-rel-speed", type=float, default=config.MIN_RELATIVE_SPEED_KM_S,
        help="Discard pairs closing slower than this (km/s), which are docked "
             f"or co-orbiting rather than conjuncting (default: "
             f"{config.MIN_RELATIVE_SPEED_KM_S}).",
    )
    parser.add_argument(
        "--max-epoch-age", type=float, default=config.MAX_EPOCH_AGE_DAYS,
        help=f"Reject element sets older than this many days (default: "
             f"{config.MAX_EPOCH_AGE_DAYS:.0f}).",
    )
    parser.add_argument(
        "--include-deep-space", action="store_true",
        help="Keep objects SGP4 classes as deep-space (period >= 225 min).",
    )
    parser.add_argument(
        "--chunk-steps", type=int, default=None,
        help="Time steps per propagation chunk (default: sized to a 2 GB budget).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes for parallel screening (default: all cores; "
             "1 runs single-process).",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Force the single-process path.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--alerts-csv", default="conjunction_alerts.csv",
        help="Path for the final pandas alert table "
             "(default: conjunction_alerts.csv).",
    )
    parser.add_argument(
        "--no-refine", action="store_true",
        help="Skip sub-second TCA refinement of the flagged events.",
    )
    parser.add_argument(
        "--refine-top", type=int, default=200,
        help="How many of the closest events to refine (default: 200).",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip benchmarking against official Space-Track CDMs.",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Rows to show in the risk table (default: 20).",
    )
    args = parser.parse_args(argv)

    if args.hours is None:
        args.hours = config.FORECAST_WINDOW_HOURS if args.full else DEMO_HOURS
    if args.max_objects is None:
        args.max_objects = 0 if args.full else DEMO_MAX_OBJECTS
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profiler = Profiler()
    baseline_memory = memory_mb()

    log.info("=" * 72)
    log.info("PROJECT ORBITAL SENTINEL")
    log.info(
        "window=%.2f h | step=%.2f min | threshold=%.1f km | max_objects=%s | "
        "min_rel_speed=%.3f km/s | max_epoch_age=%.0f d",
        args.hours,
        args.step,
        args.threshold,
        args.max_objects or "all",
        args.min_rel_speed,
        args.max_epoch_age,
    )
    log.info("=" * 72)

    # -- Stage 1: ingestion ------------------------------------------------- #
    with profiler.stage("ingestion"):
        try:
            objects = load_catalog(
                force_refresh=args.refresh,
                leo_only=not args.all_regimes,
                max_objects=args.max_objects or None,
                max_epoch_age_days=args.max_epoch_age,
                exclude_deep_space=not args.include_deep_space,
            )
        except RuntimeError as exc:
            log.error("%s", exc)
            return 1

    if len(objects) < 2:
        log.error("Need at least two objects to search for conjunctions.")
        return 1
    log.info("Catalog breakdown: %s", catalog_breakdown(objects))

    # -- Stage 2: propagation engine ---------------------------------------- #
    with profiler.stage("satrec_array_build"):
        engine = PropagationEngine(objects)

    grid = build_time_grid(hours=args.hours, step_minutes=args.step)
    n_steps = len(grid)
    chunk_steps = args.chunk_steps or choose_chunk_steps(len(engine), budget_gb=2.0)
    chunk_steps = min(chunk_steps, n_steps)
    screening_radius = required_screening_radius(args.threshold, args.step)
    log.info(
        "Propagating %d objects x %d steps = %s state vectors "
        "(chunk=%d steps, ~%.2f GB peak).",
        len(engine),
        n_steps,
        f"{len(engine) * n_steps:,}",
        chunk_steps,
        estimate_memory_gb(len(engine), chunk_steps),
    )
    log.info(
        "Two-stage screen: coarse cKDTree radius %.0f km (brackets %.0f km/s "
        "closing over a %.2f-min step), then linear-TCA filter at %.1f km.",
        screening_radius,
        config.MAX_RELATIVE_SPEED_KM_S,
        args.step,
        args.threshold,
    )

    # -- Stages 3 & 4: propagate and search --------------------------------- #
    conjunctions = ConjunctionLog()
    total_pairs_examined = 0

    n_workers = resolve_worker_count(args.workers)
    use_parallel = (
        not args.no_parallel
        and args.workers != 1
        and worth_parallelizing(len(engine), n_steps, n_workers)
    )
    if use_parallel:
        # Memory bounds the chunk from above; worker count bounds it from below,
        # so every core gets work instead of one chunk going to one process.
        chunk_steps = plan_chunks(n_steps, n_workers, memory_cap_steps=chunk_steps)
        log.info("Parallel chunk size: %d steps.", chunk_steps)
    else:
        n_workers = 1

    if use_parallel:
        # Time chunks are independent, so the whole screen fans out across
        # cores. Workers rebuild their own SatrecArray from TLE lines because
        # Satrec objects cannot be pickled.
        with profiler.stage("parallel_screen", units=len(engine) * n_steps):
            events, steps_done = run_parallel(
                objects,
                grid,
                chunk_steps=chunk_steps,
                threshold_km=args.threshold,
                min_rel_speed_kms=args.min_rel_speed,
                workers=args.workers,
                show_progress=not args.no_progress,
            )
        conjunctions.extend(events)
        total_pairs_examined = len(events)
        avg_live = len(engine)
        brute_force_equivalent = avg_live * (avg_live - 1) / 2 * n_steps
    else:
        log.info("Running the single-process screening path.")
        steps_done = 0
        brute_force_equivalent = 0.0
        chunk_offsets = list(range(0, n_steps, chunk_steps))
        progress = _progress_bar(chunk_offsets, not args.no_progress)

        # Chunks are sliced explicitly rather than via propagate_chunked() so
        # the profiler brackets the actual SGP4 call, not a lazy generator step.
        for offset in progress:
            end_step = min(offset + chunk_steps, n_steps)
            sub_grid = grid.slice(offset, end_step)

            with profiler.stage(
                "propagation", units=len(engine) * (end_step - offset)
            ):
                result = engine.propagate(sub_grid)

            with profiler.stage("spatial_search", units=result.n_steps):
                for abs_step, stamp, detection in scan_result(
                    result,
                    engine,
                    step_offset=offset,
                    threshold_km=args.threshold,
                    min_rel_speed_kms=args.min_rel_speed,
                ):
                    conjunctions.extend(
                        detections_to_events(detection, engine, stamp)
                    )
                    total_pairs_examined += len(detection)

            steps_done += result.n_steps
            # Average live objects per step, converted to the number of
            # distance comparisons an O(N^2) screen would have needed.
            avg_live = np.count_nonzero(result.valid) / result.n_steps
            brute_force_equivalent += avg_live * (avg_live - 1) / 2 * result.n_steps

    log.info(
        "Screened %d steps; %d raw detections; RSS %.0f MB.",
        steps_done,
        len(conjunctions.events),
        memory_mb(),
    )

    raw_event_count = len(conjunctions.events)
    with profiler.stage("deduplication"):
        conjunctions.deduplicate()
    log.info(
        "Collapsed %d step-level detections into %d distinct encounters.",
        raw_event_count,
        len(conjunctions.events),
    )

    # -- Stage 5: sub-second TCA refinement --------------------------------- #
    if not args.no_refine and conjunctions.events:
        by_norad = {o.norad_id: o for o in objects}
        targets = conjunctions.closest(args.refine_top)
        log.info("Refining TCA for the %d closest encounters.", len(targets))
        with profiler.stage("tca_refinement", units=len(targets)):
            for event in targets:
                obj_a = by_norad.get(event.norad_1)
                obj_b = by_norad.get(event.norad_2)
                if obj_a is None or obj_b is None:
                    continue
                coarse = datetime.strptime(
                    event.tca_utc, "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
                try:
                    tca, miss, rel_speed = refine_tca(
                        obj_a, obj_b, coarse, window_minutes=args.step / 2.0
                    )
                except Exception as exc:  # a single bad pair must not kill the run
                    log.debug("Refinement failed for %d/%d: %s",
                              event.norad_1, event.norad_2, exc)
                    continue
                # The detection stage estimated this pair assuming linear
                # relative motion across the step. Refinement evaluates real
                # SGP4 geometry, so its answer supersedes the estimate whether
                # it comes out tighter or looser.
                if np.isfinite(miss):
                    event.tca_utc = tca.strftime("%Y-%m-%d %H:%M:%S")
                    event.miss_distance_km = miss
                    event.relative_speed_km_s = rel_speed
                    event.refined = True
        conjunctions.events.sort(key=lambda e: e.miss_distance_km)
        refined = sum(1 for e in conjunctions.events if e.refined)
        log.info(
            "Refined %d encounters to true SGP4 geometry at 0.05 s resolution.",
            refined,
        )

    # -- Stage 6: output ---------------------------------------------------- #
    conjunctions.print_table(limit=args.top)
    json_path, csv_path = conjunctions.write()
    with profiler.stage("alerts_csv"):
        alerts_path = conjunctions.write_alerts_csv(args.alerts_csv)

    # -- Stage 7: validation ------------------------------------------------ #
    validation_summary = None
    if not args.no_validate:
        with profiler.stage("validation"):
            report = run_validation(
                conjunctions.events,
                propagated_norads={o.norad_id for o in objects},
                window_start=grid.start,
                window_end=grid.start + timedelta(minutes=args.step * (n_steps - 1)),
                force_refresh=args.refresh,
            )
        if report is not None:
            report.report()
            validation_summary = report.summary()
            # Persist the per-CDM comparison, not just the aggregate metrics:
            # this is the evidence that the engine reproduces official alerts.
            detail_path = config.LOG_DIR / f"validation_{conjunctions.run_id}.json"
            detail_path.write_text(
                json.dumps(
                    {
                        "summary": validation_summary,
                        "matches": report.matches,
                        "misses": report.misses,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log.info("CDM comparison detail -> %s", detail_path.name)
            validation_summary["detail_file"] = str(detail_path)

    # -- Stage 8: performance ----------------------------------------------- #
    profiler.report()
    peak_memory = memory_mb()
    total_vectors = len(engine) * n_steps
    throughput = total_vectors / profiler.elapsed if profiler.elapsed else 0.0

    log.info("THROUGHPUT")
    log.info("  objects propagated        : %s", f"{len(engine):,}")
    log.info("  time steps                : %s", f"{n_steps:,}")
    log.info("  state vectors computed    : %s", f"{total_vectors:,}")
    log.info("  end-to-end rate           : %s state vectors/sec",
             f"{throughput:,.0f}")
    log.info("  brute-force pairs avoided : ~%s", f"{brute_force_equivalent:,.0f}")
    log.info("  pair detections examined  : %s", f"{total_pairs_examined:,}")
    log.info("  resident memory           : %.0f MB (baseline %.0f MB)",
             peak_memory, baseline_memory)
    log.info("  wall clock                : %.2f s", profiler.elapsed)

    run_summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "window_hours": args.hours,
        "step_minutes": args.step,
        "threshold_km": args.threshold,
        "screening_radius_km": screening_radius,
        "min_rel_speed_km_s": args.min_rel_speed,
        "max_epoch_age_days": args.max_epoch_age,
        "objects": len(engine),
        "time_steps": n_steps,
        "state_vectors": total_vectors,
        "state_vectors_per_second": round(throughput, 1),
        "raw_detections": raw_event_count,
        "distinct_encounters": len(conjunctions.events),
        "resident_memory_mb": round(peak_memory, 1),
        "wall_clock_seconds": round(profiler.elapsed, 2),
        "stages": profiler.summary(),
        "validation": validation_summary,
        "workers": n_workers,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "alerts_csv": str(alerts_path),
        },
    }
    summary_path = config.LOG_DIR / f"run_summary_{conjunctions.run_id}.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    log.info("Run summary -> %s", summary_path.name)
    return 0


if __name__ == "__main__":
    # Required before creating a Pool when the interpreter uses spawn (Windows)
    # or when this script is frozen into an executable.
    mp.freeze_support()
    sys.exit(main())
