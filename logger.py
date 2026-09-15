"""Structured logging, performance profiling and conjunction alert output."""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import config

_CONFIGURED = False


def get_logger(name: str = "sentinel") -> logging.Logger:
    """Return the shared logger, attaching console + file handlers once."""
    global _CONFIGURED
    root = logging.getLogger("sentinel")
    if not _CONFIGURED:
        root.setLevel(logging.INFO)
        root.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        root.addHandler(console)

        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_handler = logging.FileHandler(
            config.LOG_DIR / f"sentinel_{run_stamp}.log", encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        _CONFIGURED = True

    return root if name == "sentinel" else root.getChild(name)


# --------------------------------------------------------------------------- #
# Performance profiling
# --------------------------------------------------------------------------- #
@dataclass
class Stage:
    """Cumulative wall-clock and throughput for one pipeline stage."""

    name: str
    seconds: float = 0.0
    calls: int = 0
    units: float = 0.0  # e.g. state vectors propagated, pairs queried

    @property
    def rate(self) -> float:
        return self.units / self.seconds if self.seconds > 0 else 0.0


class Profiler:
    """Accumulates per-stage timings across the run."""

    def __init__(self) -> None:
        self.stages: dict[str, Stage] = {}
        self.started = time.perf_counter()

    @contextmanager
    def stage(self, name: str, units: float = 0.0) -> Iterator[Stage]:
        entry = self.stages.setdefault(name, Stage(name))
        t0 = time.perf_counter()
        try:
            yield entry
        finally:
            entry.seconds += time.perf_counter() - t0
            entry.calls += 1
            entry.units += units

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": s.name,
                "seconds": round(s.seconds, 3),
                "calls": s.calls,
                "units": s.units,
                "units_per_second": round(s.rate, 1),
                "share_pct": round(100 * s.seconds / self.elapsed, 1)
                if self.elapsed
                else 0.0,
            }
            for s in sorted(
                self.stages.values(), key=lambda s: s.seconds, reverse=True
            )
        ]

    def report(self, log: logging.Logger | None = None) -> None:
        log = log or get_logger("perf")
        log.info("-" * 72)
        log.info("PERFORMANCE PROFILE (total %.2fs)", self.elapsed)
        log.info(
            "%-22s %10s %7s %14s %9s",
            "stage",
            "seconds",
            "calls",
            "units/sec",
            "share",
        )
        for row in self.summary():
            log.info(
                "%-22s %10.3f %7d %14s %8.1f%%",
                row["stage"],
                row["seconds"],
                row["calls"],
                f"{row['units_per_second']:,.0f}" if row["units"] else "-",
                row["share_pct"],
            )
        log.info("-" * 72)


def memory_mb() -> float:
    """Resident set size in MB, or 0.0 when unavailable on this platform."""
    try:  # psutil is optional, never a hard dependency
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:  # Windows fallback via the Win32 API
        import ctypes
        import ctypes.wintypes as wt

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        # GetCurrentProcess returns the pseudo-handle -1; ctypes truncates a
        # bare Python int on 64-bit, so it must be wrapped as a real HANDLE.
        handle = wt.HANDLE(ctypes.windll.kernel32.GetCurrentProcess())
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# --------------------------------------------------------------------------- #
# Conjunction alert records
# --------------------------------------------------------------------------- #
@dataclass
class ConjunctionEvent:
    """One flagged close approach between two catalogued objects."""

    tca_utc: str
    norad_1: int
    norad_2: int
    name_1: str
    name_2: str
    type_1: str
    type_2: str
    miss_distance_km: float
    relative_speed_km_s: float
    altitude_km: float
    # TEME Cartesian midpoint of the encounter, in km. Carried through to the
    # CSV so visualize.py can plot each conjunction in 3D without re-running
    # the propagator.
    x_km: float = 0.0
    y_km: float = 0.0
    z_km: float = 0.0
    # Set when propagation.refine_tca() resolved the TCA below the step size.
    refined: bool = False

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


class ConjunctionLog:
    """Collects events in memory and writes them to JSON + CSV on close."""

    FIELDS = [
        "tca_utc",
        "norad_1",
        "norad_2",
        "name_1",
        "name_2",
        "type_1",
        "type_2",
        "miss_distance_km",
        "relative_speed_km_s",
        "altitude_km",
        "x_km",
        "y_km",
        "z_km",
        "refined",
    ]

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.events: list[ConjunctionEvent] = []
        self.log = get_logger("conj")

    def add(self, event: ConjunctionEvent) -> None:
        self.events.append(event)

    def extend(self, events: Iterable[ConjunctionEvent]) -> None:
        for event in events:
            self.add(event)

    def closest(self, limit: int = 20) -> list[ConjunctionEvent]:
        return sorted(self.events, key=lambda e: e.miss_distance_km)[:limit]

    def deduplicate(self) -> None:
        """Keep only the tightest approach per object pair.

        A single physical conjunction spans several consecutive time steps, so
        the raw event stream reports the same encounter repeatedly. Only the
        minimum miss distance is operationally meaningful.
        """
        best: dict[tuple[int, int], ConjunctionEvent] = {}
        for event in self.events:
            key = (min(event.norad_1, event.norad_2), max(event.norad_1, event.norad_2))
            if key not in best or event.miss_distance_km < best[key].miss_distance_km:
                best[key] = event
        self.events = sorted(best.values(), key=lambda e: e.miss_distance_km)

    def write(self) -> tuple[Path, Path]:
        json_path = config.LOG_DIR / f"conjunctions_{self.run_id}.json"
        csv_path = config.LOG_DIR / f"conjunctions_{self.run_id}.csv"

        rows = [
            e.as_row() for e in sorted(self.events, key=lambda e: e.miss_distance_km)
        ]
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        self.log.info("Wrote %d conjunction events -> %s", len(rows), csv_path.name)
        return json_path, csv_path

    def to_dataframe(self):
        """The alert table as a pandas DataFrame, closest approach first."""
        import pandas as pd

        rows = [
            e.as_row() for e in sorted(self.events, key=lambda e: e.miss_distance_km)
        ]
        frame = pd.DataFrame(rows, columns=self.FIELDS)
        if frame.empty:
            # Preserve the schema so downstream readers do not have to special
            # case an empty run.
            return frame.astype(
                {"norad_1": "int64", "norad_2": "int64", "refined": "bool"},
                errors="ignore",
            )
        return frame

    def write_alerts_csv(self, path: Path | str) -> Path:
        """Write the final filtered alerts to a local CSV via pandas."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_dataframe()
        frame.to_csv(path, index=False)
        self.log.info(
            "Wrote %d conjunction alerts -> %s", len(frame), path
        )
        return path

    def print_table(self, limit: int = 20) -> None:
        """Render the highest-risk conjunctions as an aligned console table."""
        if not self.events:
            self.log.info(
                "No conjunctions detected below the %.1f km threshold.",
                config.CONJUNCTION_THRESHOLD_KM,
            )
            return

        rows = self.closest(limit)
        self.log.info("=" * 108)
        self.log.info(
            "TOP %d HIGHEST-RISK CONJUNCTIONS (of %d flagged)", len(rows), len(self.events)
        )
        self.log.info(
            "%-21s %-26s %-26s %9s %9s %8s",
            "TCA (UTC)",
            "OBJECT 1",
            "OBJECT 2",
            "MISS km",
            "REL km/s",
            "ALT km",
        )
        self.log.info("-" * 108)
        for e in rows:
            self.log.info(
                "%-21s %-26s %-26s %9.3f %9.3f %8.0f",
                e.tca_utc,
                f"{e.norad_1} {e.name_1}"[:26],
                f"{e.norad_2} {e.name_2}"[:26],
                e.miss_distance_km,
                e.relative_speed_km_s,
                e.altitude_km,
            )
        self.log.info("=" * 108)
