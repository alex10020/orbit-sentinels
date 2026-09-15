"""Benchmark predicted conjunctions against official 18th SDS CDMs.

Ground truth comes from the Space-Track ``cdm_public`` class. Two caveats shape
how the metrics below should be read:

1. ``cdm_public`` exposes only the most recent ~100 messages, and only those the
   18th Space Defense Squadron publishes openly. It is a *truncated* sample of
   the real conjunction set, so a prediction with no matching CDM is not
   necessarily a false alarm. Reported precision is therefore a lower bound.
2. CDM ``MIN_RNG`` is expressed in METERS; predictions are in kilometers.

Recall restricted to CDM pairs whose objects both appear in the propagated
subset (see :func:`filter_relevant_cdms`) is the most defensible metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import numpy as np

import config
from ingestion import fetch_cdms
from logger import ConjunctionEvent, get_logger

log = get_logger("validate")


@dataclass(slots=True)
class CDMRecord:
    """One official Conjunction Data Message, normalized to km and UTC."""

    cdm_id: str
    tca: datetime
    norad_1: int
    norad_2: int
    name_1: str
    name_2: str
    min_range_km: float
    probability: float
    emergency: bool

    @property
    def pair(self) -> tuple[int, int]:
        return (min(self.norad_1, self.norad_2), max(self.norad_1, self.norad_2))


@dataclass
class ValidationReport:
    """Precision / recall / residuals for one forecast run."""

    considered_cdms: int = 0
    matched_cdms: int = 0
    missed_cdms: int = 0
    predictions: int = 0
    matched_predictions: int = 0
    unmatched_predictions: int = 0
    residuals_km: list[float] = field(default_factory=list)
    tca_residuals_minutes: list[float] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    misses: list[dict[str, Any]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.matched_cdms / self.considered_cdms if self.considered_cdms else 0.0

    @property
    def precision(self) -> float:
        return (
            self.matched_predictions / self.predictions if self.predictions else 0.0
        )

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> dict[str, Any]:
        res = np.array(self.residuals_km) if self.residuals_km else np.empty(0)
        tres = (
            np.array(self.tca_residuals_minutes)
            if self.tca_residuals_minutes
            else np.empty(0)
        )
        return {
            "considered_cdms": self.considered_cdms,
            "matched_cdms": self.matched_cdms,
            "missed_cdms": self.missed_cdms,
            "predictions": self.predictions,
            "matched_predictions": self.matched_predictions,
            "unmatched_predictions": self.unmatched_predictions,
            "recall": round(self.recall, 4),
            "precision_lower_bound": round(self.precision, 4),
            "f1_lower_bound": round(self.f1, 4),
            "miss_distance_residual_km": {
                "mean_abs": round(float(np.mean(np.abs(res))), 4) if res.size else None,
                "median_abs": round(float(np.median(np.abs(res))), 4)
                if res.size
                else None,
                "max_abs": round(float(np.max(np.abs(res))), 4) if res.size else None,
            },
            "tca_residual_minutes": {
                "mean_abs": round(float(np.mean(np.abs(tres))), 3)
                if tres.size
                else None,
                "max_abs": round(float(np.max(np.abs(tres))), 3) if tres.size else None,
            },
        }

    def report(self) -> None:
        s = self.summary()
        log.info("=" * 72)
        log.info("VALIDATION AGAINST OFFICIAL 18th SDS CDMs")
        log.info("-" * 72)
        log.info("CDMs in scope (both objects propagated) : %d", s["considered_cdms"])
        log.info("  matched by our predictions            : %d", s["matched_cdms"])
        log.info("  missed                                : %d", s["missed_cdms"])
        log.info("Our predictions                         : %d", s["predictions"])
        log.info("  corroborated by a CDM                 : %d", s["matched_predictions"])
        log.info("  uncorroborated                        : %d",
                 s["unmatched_predictions"])
        log.info("-" * 72)
        log.info("Recall                                  : %.3f", s["recall"])
        log.info("Precision (lower bound)                 : %.3f",
                 s["precision_lower_bound"])
        log.info("F1 (lower bound)                        : %.3f", s["f1_lower_bound"])
        mr = s["miss_distance_residual_km"]
        if mr["mean_abs"] is not None:
            log.info("Miss-distance residual |km|  mean/median/max: %.3f / %.3f / %.3f",
                     mr["mean_abs"], mr["median_abs"], mr["max_abs"])
        tr = s["tca_residual_minutes"]
        if tr["mean_abs"] is not None:
            log.info("TCA residual |min|           mean/max       : %.2f / %.2f",
                     tr["mean_abs"], tr["max_abs"])
        log.info("-" * 72)
        log.info("NOTE: cdm_public exposes only the ~100 most recent public messages,")
        log.info("so uncorroborated predictions are not necessarily false alarms.")
        log.info("=" * 72)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_tca(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_cdms(records: Iterable[dict[str, Any]]) -> list[CDMRecord]:
    """Normalize raw ``cdm_public`` JSON into :class:`CDMRecord` objects."""
    parsed: list[CDMRecord] = []
    for record in records:
        tca = _parse_tca(record.get("TCA"))
        if tca is None:
            continue
        try:
            norad_1 = int(record["SAT_1_ID"])
            norad_2 = int(record["SAT_2_ID"])
        except (KeyError, TypeError, ValueError):
            continue

        # MIN_RNG is published in meters.
        min_range_m = _to_float(record.get("MIN_RNG"))
        parsed.append(
            CDMRecord(
                cdm_id=str(record.get("CDM_ID", "")),
                tca=tca,
                norad_1=norad_1,
                norad_2=norad_2,
                name_1=(record.get("SAT_1_NAME") or "").strip(),
                name_2=(record.get("SAT_2_NAME") or "").strip(),
                min_range_km=min_range_m / 1000.0,
                probability=_to_float(record.get("PC"), 0.0),
                emergency=str(record.get("EMERGENCY_REPORTABLE", "")).upper() == "Y",
            )
        )
    log.info("Parsed %d CDMs.", len(parsed))
    return parsed


def deduplicate_cdms(cdms: Sequence[CDMRecord]) -> list[CDMRecord]:
    """Collapse mirrored and re-issued messages down to one per encounter.

    Space-Track publishes each conjunction twice with SAT_1/SAT_2 swapped, and
    re-issues a message whenever the solution is refined. Counting those as
    distinct events would inflate the recall denominator, so we key on the
    unordered pair plus the TCA minute and keep the tightest reported range.
    """
    best: dict[tuple[tuple[int, int], str], CDMRecord] = {}
    for cdm in cdms:
        key = (cdm.pair, cdm.tca.strftime("%Y-%m-%d %H:%M"))
        current = best.get(key)
        if current is None or cdm.min_range_km < current.min_range_km:
            best[key] = cdm
    collapsed = sorted(best.values(), key=lambda c: c.tca)
    if len(collapsed) != len(cdms):
        log.info(
            "Collapsed %d raw CDMs into %d distinct encounters.",
            len(cdms),
            len(collapsed),
        )
    return collapsed


def filter_relevant_cdms(
    cdms: Sequence[CDMRecord],
    propagated_norads: set[int],
    window_start: datetime,
    window_end: datetime,
    threshold_km: float | None = None,
) -> list[CDMRecord]:
    """Keep only CDMs our run could possibly have detected.

    A CDM is in scope when both objects were propagated, its TCA falls inside
    the forecast window, and its reported minimum range is within our screening
    threshold. Anything else is outside the experiment, not a miss.
    """
    threshold_km = (
        config.CONJUNCTION_THRESHOLD_KM if threshold_km is None else threshold_km
    )
    unique = deduplicate_cdms(cdms)
    relevant = [
        c
        for c in unique
        if c.norad_1 in propagated_norads
        and c.norad_2 in propagated_norads
        and window_start <= c.tca <= window_end
        and (np.isnan(c.min_range_km) or c.min_range_km <= threshold_km)
    ]
    log.info(
        "%d of %d distinct CDM encounters are in scope (both objects propagated, "
        "TCA in window, MIN_RNG <= %.1f km).",
        len(relevant),
        len(unique),
        threshold_km,
    )
    return relevant


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def validate(
    events: Sequence[ConjunctionEvent],
    cdms: Sequence[CDMRecord],
    propagated_norads: set[int],
    window_start: datetime,
    window_end: datetime,
    tolerance_minutes: float | None = None,
) -> ValidationReport:
    """Match predictions against in-scope CDMs and compute the metrics."""
    tolerance_minutes = (
        config.TCA_MATCH_TOLERANCE_MINUTES
        if tolerance_minutes is None
        else tolerance_minutes
    )
    tolerance = timedelta(minutes=tolerance_minutes)
    in_scope = filter_relevant_cdms(
        cdms, propagated_norads, window_start, window_end
    )

    # Index predictions by unordered NORAD pair.
    by_pair: dict[tuple[int, int], list[ConjunctionEvent]] = {}
    for event in events:
        key = (min(event.norad_1, event.norad_2), max(event.norad_1, event.norad_2))
        by_pair.setdefault(key, []).append(event)

    report = ValidationReport(
        considered_cdms=len(in_scope), predictions=len(events)
    )
    matched_event_ids: set[int] = set()

    for cdm in in_scope:
        candidates = by_pair.get(cdm.pair, [])
        best: ConjunctionEvent | None = None
        best_gap = tolerance
        for event in candidates:
            stamp = datetime.strptime(
                event.tca_utc, "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            gap = abs(stamp - cdm.tca)
            if gap <= best_gap:
                best, best_gap = event, gap

        if best is None:
            report.missed_cdms += 1
            report.misses.append(
                {
                    "cdm_id": cdm.cdm_id,
                    "pair": list(cdm.pair),
                    "names": [cdm.name_1, cdm.name_2],
                    "tca_utc": cdm.tca.strftime("%Y-%m-%d %H:%M:%S"),
                    "cdm_min_range_km": round(cdm.min_range_km, 4),
                    "probability": cdm.probability,
                }
            )
            continue

        report.matched_cdms += 1
        matched_event_ids.add(id(best))
        residual = best.miss_distance_km - cdm.min_range_km
        report.residuals_km.append(residual)
        report.tca_residuals_minutes.append(best_gap.total_seconds() / 60.0)
        report.matches.append(
            {
                "cdm_id": cdm.cdm_id,
                "pair": list(cdm.pair),
                "names": [cdm.name_1, cdm.name_2],
                "cdm_tca_utc": cdm.tca.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_tca_utc": best.tca_utc,
                "cdm_min_range_km": round(cdm.min_range_km, 4),
                "predicted_miss_km": round(best.miss_distance_km, 4),
                "residual_km": round(residual, 4),
                "tca_residual_minutes": round(best_gap.total_seconds() / 60.0, 3),
                "probability": cdm.probability,
            }
        )

    report.matched_predictions = len(matched_event_ids)
    report.unmatched_predictions = len(events) - report.matched_predictions
    return report


def run_validation(
    events: Sequence[ConjunctionEvent],
    propagated_norads: set[int],
    window_start: datetime,
    window_end: datetime,
    force_refresh: bool = False,
) -> ValidationReport | None:
    """Fetch CDMs and validate in one call. Returns None if no CDMs available."""
    raw = fetch_cdms(force_refresh=force_refresh)
    if not raw:
        log.warning("No CDMs available; skipping validation.")
        return None
    cdms = parse_cdms(raw)
    return validate(events, cdms, propagated_norads, window_start, window_end)


if __name__ == "__main__":
    cdms = parse_cdms(fetch_cdms())
    log.info("Loaded %d CDMs for inspection.", len(cdms))
    for c in sorted(cdms, key=lambda c: c.min_range_km)[:10]:
        log.info(
            "%s  %6d x %-6d  %-22s %-22s  min_rng=%6.3f km  Pc=%.2e %s",
            c.tca.strftime("%Y-%m-%d %H:%M:%S"),
            c.norad_1,
            c.norad_2,
            c.name_1[:22],
            c.name_2[:22],
            c.min_range_km,
            c.probability,
            "EMERGENCY" if c.emergency else "",
        )
