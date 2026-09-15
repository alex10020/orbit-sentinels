"""Space-Track ingestion: authenticated catalog download, caching and parsing.

The public entry point is :func:`load_catalog`, which returns a list of
:class:`SpaceObject` records ready for vectorized propagation. It transparently
prefers a fresh local cache over hitting the API, because Space-Track enforces
a strict request budget (~30/min, 300/hour).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
from sgp4.api import Satrec

import config
from logger import get_logger

log = get_logger("ingest")

# Mean motion (rev/day) -> semi-major axis, via Kepler's third law.
_MU_EARTH_KM3_S2 = 398600.4418


@dataclass(slots=True)
class SpaceObject:
    """A catalogued object: its SGP4 propagator plus identifying metadata."""

    norad_id: int
    name: str
    object_type: str
    epoch: str
    satrec: Satrec = field(repr=False)
    mean_motion: float = 0.0
    eccentricity: float = 0.0
    # The raw element set is retained because Satrec objects cannot be pickled:
    # multiprocessing workers rebuild their own SatrecArray from these lines.
    tle_line1: str = ""
    tle_line2: str = ""

    @property
    def semi_major_axis_km(self) -> float:
        """Semi-major axis derived from mean motion (Kepler's third law)."""
        if self.mean_motion <= 0:
            return 0.0
        n_rad_s = self.mean_motion * 2.0 * 3.141592653589793 / 86400.0
        return (_MU_EARTH_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)

    @property
    def perigee_altitude_km(self) -> float:
        a = self.semi_major_axis_km
        if a <= 0:
            return 0.0
        return a * (1.0 - self.eccentricity) - config.EARTH_RADIUS_KM

    @property
    def apogee_altitude_km(self) -> float:
        a = self.semi_major_axis_km
        if a <= 0:
            return 0.0
        return a * (1.0 + self.eccentricity) - config.EARTH_RADIUS_KM

    @property
    def epoch_datetime(self) -> datetime | None:
        """Element set epoch as a UTC datetime, or None if unparseable."""
        if not self.epoch:
            return None
        text = self.epoch.replace("T", " ").strip()[:26]
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @property
    def epoch_age_days(self) -> float:
        """Days since the element set epoch; +inf when the epoch is unusable."""
        stamp = self.epoch_datetime
        if stamp is None:
            return float("inf")
        return (datetime.now(timezone.utc) - stamp).total_seconds() / 86400.0

    @property
    def is_deep_space(self) -> bool:
        """True when SGP4 selected its deep-space model (period >= 225 min)."""
        return getattr(self.satrec, "method", "n") == "d"


class SpaceTrackClient:
    """Authenticated, rate-limited Space-Track session."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})
        self._authenticated = False
        self._last_request = 0.0

    # -- session management ------------------------------------------------ #
    def login(self) -> bool:
        """Authenticate the session. Returns False on credential failure."""
        if self._authenticated:
            return True
        config.require_credentials()

        log.info("Authenticating with Space-Track as %s", config.SPACETRACK_USER)
        try:
            response = self.session.post(
                config.API_LOGIN_URL,
                data={
                    "identity": config.SPACETRACK_USER,
                    "password": config.SPACETRACK_PASS,
                },
                timeout=60,
            )
        except requests.RequestException as exc:
            log.error("Login request failed: %s", exc)
            return False

        # Space-Track answers a bad login with HTTP 200 and a JSON error body.
        if response.status_code != 200:
            log.error("Login rejected: HTTP %s", response.status_code)
            return False
        body = response.text.strip()
        if body and "Failed" in body:
            log.error("Login rejected by Space-Track: %s", body[:200])
            return False

        self._authenticated = True
        log.info("Authenticated successfully.")
        return True

    def _throttle(self) -> None:
        """Space out requests to stay inside the published rate limit."""
        gap = time.monotonic() - self._last_request
        if gap < config.API_MIN_REQUEST_INTERVAL_S:
            time.sleep(config.API_MIN_REQUEST_INTERVAL_S - gap)
        self._last_request = time.monotonic()

    def get_json(self, url: str) -> list[dict[str, Any]] | None:
        """GET a Space-Track query URL, retrying on transient failures."""
        if not self.login():
            return None

        for attempt in range(1, config.API_MAX_RETRIES + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=config.API_TIMEOUT_S)
            except requests.RequestException as exc:
                log.warning("Request error (attempt %d): %s", attempt, exc)
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    log.error("Response was not valid JSON (%d bytes).",
                              len(response.content))
                    return None
                if isinstance(payload, dict):  # error envelope
                    log.error("Space-Track returned an error: %s",
                              str(payload)[:200])
                    return None
                return payload

            if response.status_code in (429, 500, 502, 503, 504):
                backoff = 2 ** attempt * 5
                log.warning("HTTP %s, backing off %ds (attempt %d).",
                            response.status_code, backoff, attempt)
                time.sleep(backoff)
                continue

            if response.status_code == 401:
                log.warning("Session expired, re-authenticating.")
                self._authenticated = False
                if not self.login():
                    return None
                continue

            log.error("Unexpected HTTP %s from Space-Track.", response.status_code)
            return None

        log.error("Giving up after %d attempts.", config.API_MAX_RETRIES)
        return None

    def logout(self) -> None:
        self.session.close()
        self._authenticated = False

    def __enter__(self) -> "SpaceTrackClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.logout()


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _cache_age_hours(path: Path) -> float:
    if not path.exists():
        return float("inf")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0


def _read_cache(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Cache at %s unreadable: %s", path.name, exc)
        return None
    return payload if isinstance(payload, list) else None


def _write_cache(path: Path, records: Sequence[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records), encoding="utf-8")
    tmp.replace(path)  # atomic, so an interrupted write cannot corrupt the cache
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info("Cached %d records -> %s (%.1f MB)", len(records), path.name, size_mb)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def fetch_raw_catalog(
    force_refresh: bool = False,
    client: SpaceTrackClient | None = None,
) -> list[dict[str, Any]]:
    """Return the raw GP catalog, preferring a fresh cache over the API.

    Falls back to a stale cache if the network or credentials fail, so an
    offline run still produces results.
    """
    age = _cache_age_hours(config.CATALOG_CACHE)
    if not force_refresh and age <= config.CATALOG_MAX_AGE_HOURS:
        cached = _read_cache(config.CATALOG_CACHE)
        if cached:
            log.info("Using cached catalog: %d objects (%.1f h old).",
                     len(cached), age)
            return cached

    log.info("Downloading active catalog from Space-Track (this takes ~30-90s).")
    owns_client = client is None
    client = client or SpaceTrackClient()
    try:
        records = client.get_json(config.API_QUERY_URL)
    finally:
        if owns_client:
            client.logout()

    if not records:
        stale = _read_cache(config.CATALOG_CACHE)
        if stale:
            log.warning("Download failed; falling back to stale cache "
                        "(%d objects, %.1f h old).", len(stale), age)
            return stale
        raise RuntimeError(
            "Could not download the catalog and no local cache is available. "
            "Check your network and the credentials in .env."
        )

    _write_cache(config.CATALOG_CACHE, records)
    return records


def fetch_cdms(
    force_refresh: bool = False,
    client: SpaceTrackClient | None = None,
) -> list[dict[str, Any]]:
    """Return recent public Conjunction Data Messages (18th SDS ground truth)."""
    age = _cache_age_hours(config.CDM_CACHE)
    if not force_refresh and age <= config.CATALOG_MAX_AGE_HOURS:
        cached = _read_cache(config.CDM_CACHE)
        if cached is not None:
            log.info("Using cached CDMs: %d messages (%.1f h old).",
                     len(cached), age)
            return cached

    log.info("Downloading public CDMs from Space-Track.")
    owns_client = client is None
    client = client or SpaceTrackClient()
    try:
        records = client.get_json(config.API_CDM_URL)
    finally:
        if owns_client:
            client.logout()

    if records is None:
        stale = _read_cache(config.CDM_CACHE)
        if stale is not None:
            log.warning("CDM download failed; using stale cache (%d messages).",
                        len(stale))
            return stale
        log.warning("CDM download failed and no cache exists; "
                    "validation will be skipped.")
        return []

    _write_cache(config.CDM_CACHE, records)
    return records


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_records(
    records: Iterable[dict[str, Any]],
    leo_only: bool = True,
    max_objects: int | None = None,
    max_epoch_age_days: float | None = None,
    exclude_deep_space: bool | None = None,
) -> list[SpaceObject]:
    """Convert raw GP JSON records into initialized :class:`SpaceObject` items.

    Records are dropped when the TLE is malformed, SGP4 init returns an error,
    the element set is staler than ``max_epoch_age_days``, SGP4 classes the
    orbit as deep-space, or (when ``leo_only``) perigee sits above the LEO
    ceiling. The epoch-age gate matters most: the catalog still lists
    interplanetary probes under their years-old launch-phase TLE, which SGP4
    will happily propagate into a physically meaningless LEO trajectory.
    """
    max_epoch_age_days = (
        config.MAX_EPOCH_AGE_DAYS if max_epoch_age_days is None else max_epoch_age_days
    )
    exclude_deep_space = (
        config.EXCLUDE_DEEP_SPACE if exclude_deep_space is None else exclude_deep_space
    )

    objects: list[SpaceObject] = []
    skipped_malformed = 0
    skipped_init = 0
    skipped_regime = 0
    skipped_stale = 0
    skipped_deep = 0

    for record in records:
        line1 = record.get("TLE_LINE1")
        line2 = record.get("TLE_LINE2")
        if not line1 or not line2:
            skipped_malformed += 1
            continue

        try:
            satrec = Satrec.twoline2rv(line1, line2)
        except Exception:
            skipped_malformed += 1
            continue

        # A non-zero error code here means the element set is unusable.
        if getattr(satrec, "error", 0) != 0:
            skipped_init += 1
            continue

        obj = SpaceObject(
            norad_id=int(_to_float(record.get("NORAD_CAT_ID"), -1)),
            name=(record.get("OBJECT_NAME") or "UNKNOWN").strip(),
            object_type=(record.get("OBJECT_TYPE") or "UNKNOWN").strip(),
            epoch=(record.get("EPOCH") or "").strip(),
            satrec=satrec,
            mean_motion=_to_float(record.get("MEAN_MOTION")),
            eccentricity=_to_float(record.get("ECCENTRICITY")),
            tle_line1=line1,
            tle_line2=line2,
        )

        if max_epoch_age_days and obj.epoch_age_days > max_epoch_age_days:
            skipped_stale += 1
            continue

        if exclude_deep_space and obj.is_deep_space:
            skipped_deep += 1
            continue

        if leo_only and obj.perigee_altitude_km > config.LEO_ALTITUDE_CEILING_KM:
            skipped_regime += 1
            continue

        objects.append(obj)
        if max_objects and len(objects) >= max_objects:
            break

    log.info(
        "Parsed %d objects (skipped: %d malformed, %d init-error, %d stale >%.0fd, "
        "%d deep-space, %d above %.0f km).",
        len(objects),
        skipped_malformed,
        skipped_init,
        skipped_stale,
        max_epoch_age_days,
        skipped_deep,
        skipped_regime,
        config.LEO_ALTITUDE_CEILING_KM,
    )
    return objects


def load_catalog(
    force_refresh: bool = False,
    leo_only: bool = True,
    max_objects: int | None = None,
    client: SpaceTrackClient | None = None,
    max_epoch_age_days: float | None = None,
    exclude_deep_space: bool | None = None,
) -> list[SpaceObject]:
    """Fetch (or load from cache) and parse the active catalog."""
    raw = fetch_raw_catalog(force_refresh=force_refresh, client=client)
    limit = max_objects if max_objects is not None else (config.MAX_OBJECTS or None)
    return parse_records(
        raw,
        leo_only=leo_only,
        max_objects=limit,
        max_epoch_age_days=max_epoch_age_days,
        exclude_deep_space=exclude_deep_space,
    )


def catalog_breakdown(objects: Sequence[SpaceObject]) -> dict[str, int]:
    """Count objects by OBJECT_TYPE, for the run summary."""
    counts: dict[str, int] = {}
    for obj in objects:
        counts[obj.object_type] = counts.get(obj.object_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


if __name__ == "__main__":
    catalog = load_catalog(max_objects=200)
    log.info("Breakdown: %s", catalog_breakdown(catalog))
    for item in catalog[:5]:
        log.info(
            "%6d %-24s %-8s perigee=%7.1f km apogee=%7.1f km",
            item.norad_id,
            item.name[:24],
            item.object_type,
            item.perigee_altitude_km,
            item.apogee_altitude_km,
        )
