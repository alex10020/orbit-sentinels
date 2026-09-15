"""Global configuration for Project Orbital Sentinel.

Secrets are read from ``.env`` (gitignored). Tunable parameters fall back to the
constants below when no environment override is present.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CATALOG_CACHE = DATA_DIR / "latest_catalog.json"
CDM_CACHE = DATA_DIR / "latest_cdms.json"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
SPACETRACK_USER = os.getenv("SPACETRACK_USER", "")
SPACETRACK_PASS = os.getenv("SPACETRACK_PASS", "")


def credentials_present() -> bool:
    """True when both Space-Track credentials are configured."""
    return bool(SPACETRACK_USER and SPACETRACK_PASS)


def require_credentials() -> None:
    """Raise a helpful error when ``.env`` is missing or incomplete."""
    if not credentials_present():
        raise RuntimeError(
            "Space-Track credentials missing. Copy .env.example to .env and set "
            "SPACETRACK_USER and SPACETRACK_PASS."
        )


# --------------------------------------------------------------------------- #
# Physics & detection parameters
# --------------------------------------------------------------------------- #
CONJUNCTION_THRESHOLD_KM = _env_float("CONJUNCTION_THRESHOLD_KM", 5.0)
FORECAST_WINDOW_HOURS = _env_int("FORECAST_WINDOW_HOURS", 72)
TIME_STEP_MINUTES = _env_int("TIME_STEP_MINUTES", 1)

# Objects above this altitude (km) are outside the LEO regime of interest.
LEO_ALTITUDE_CEILING_KM = _env_float("LEO_ALTITUDE_CEILING_KM", 2000.0)
EARTH_RADIUS_KM = 6378.137

# 0 / unset means "propagate the entire catalog".
MAX_OBJECTS = _env_int("MAX_OBJECTS", 0)

# SGP4 accuracy degrades quickly past epoch, and the catalog retains element
# sets for objects that escaped Earth orbit years ago (interplanetary probes
# still carry their stale launch-phase TLE). Discard anything older than this.
MAX_EPOCH_AGE_DAYS = _env_float("MAX_EPOCH_AGE_DAYS", 14.0)

# Exclude objects SGP4 classes as deep-space (orbital period >= 225 min).
EXCLUDE_DEEP_SPACE = os.getenv("EXCLUDE_DEEP_SPACE", "1").strip() not in ("0", "false", "False")

# Fastest physically plausible LEO closing speed: two objects in ~7.8 km/s
# orbits meeting head-on. Sets the coarse screening radius, which must bracket
# any encounter reachable within one time step (see spatial_index).
MAX_RELATIVE_SPEED_KM_S = _env_float("MAX_RELATIVE_SPEED_KM_S", 16.0)

# Pairs closing slower than this are co-orbiting, not conjuncting: docked
# station modules, co-deployed payloads and objects sharing an element set all
# register a near-zero miss distance at ~0 relative velocity. A genuine LEO
# conjunction between independent objects closes at hundreds of m/s to ~15 km/s.
MIN_RELATIVE_SPEED_KM_S = _env_float("MIN_RELATIVE_SPEED_KM_S", 0.05)

# Time steps propagated per chunk. Bounds peak memory: a chunk holds an
# [N, T, 3] float64 position array plus the matching velocity array.
TIME_CHUNK_STEPS = _env_int("TIME_CHUNK_STEPS", 120)

# --------------------------------------------------------------------------- #
# Space-Track API
# --------------------------------------------------------------------------- #
API_BASE_URL = "https://www.space-track.org"
API_LOGIN_URL = f"{API_BASE_URL}/ajaxauth/login"
API_QUERY_URL = (
    f"{API_BASE_URL}/basicspacedata/query/class/gp/DECAY_DATE/null-val"
    "/orderby/NORAD_CAT_ID%20asc/format/json"
)
API_CDM_URL = (
    f"{API_BASE_URL}/basicspacedata/query/class/cdm_public"
    "/limit/100/orderby/TCA%20desc/format/json"
)

# Space-Track rate limits clients to ~30 requests/min and 300/hour.
API_MIN_REQUEST_INTERVAL_S = _env_float("API_MIN_REQUEST_INTERVAL_S", 3.0)
API_TIMEOUT_S = _env_int("API_TIMEOUT_S", 300)
API_MAX_RETRIES = _env_int("API_MAX_RETRIES", 3)
USER_AGENT = "OrbitalSentinel/1.0 (academic SSA research)"

# Re-download the catalog when the cache is older than this.
CATALOG_MAX_AGE_HOURS = _env_float("CATALOG_MAX_AGE_HOURS", 8.0)

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
# A prediction matches a CDM when the object pair agrees and TCA is within this.
TCA_MATCH_TOLERANCE_MINUTES = _env_float("TCA_MATCH_TOLERANCE_MINUTES", 30.0)
