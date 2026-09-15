"""Evidence report: proof that the engine ran SGP4, NumPy and CDM validation.

The pipeline prints its results to the terminal and scatters them across
``logs/``. This gathers the evidence into one page you can actually read or
hand to someone else:

* **Engine evidence** -- runs a live vectorized SGP4 propagation and reports the
  real NumPy array shapes, dtypes and a sample state vector, so the claim
  "``SatrecArray`` + NumPy" is demonstrated rather than asserted.
* **CDM validation** -- recomputes the comparison against official 18th Space
  Defense Squadron messages and shows every matched encounter side by side:
  our predicted TCA and miss distance against theirs, with residuals.
* **Performance** -- the per-stage profile from the most recent run.

    python report.py --open

Reads ``conjunction_alerts.csv`` and the newest ``logs/run_summary_*.json``.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from ingestion import fetch_cdms, load_catalog
from logger import get_logger
from propagation import PropagationEngine, build_time_grid
from validator import parse_cdms, validate

log = get_logger("report")

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 16px 64px;
  background: #05070d; color: #dbe4f0;
  font: 15px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
header { padding: 48px 0 8px; border-bottom: 1px solid #1d2740; margin-bottom: 8px; }
h1 { margin: 0 0 6px; font-size: 30px; letter-spacing: -0.4px; color: #f2f6fc; }
.sub { color: #8fa6c4; font-size: 14px; }
h2 {
  margin: 44px 0 4px; font-size: 20px; color: #f2f6fc;
  padding-bottom: 8px; border-bottom: 1px solid #1d2740;
}
h3 { margin: 26px 0 8px; font-size: 15px; color: #9fb4d0; font-weight: 600; }
p.note { color: #8fa6c4; font-size: 13.5px; margin: 8px 0 0; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0 6px; }
.card {
  flex: 1 1 190px; background: #0d1422; border: 1px solid #1d2740;
  border-radius: 10px; padding: 14px 16px;
}
.card .k { color: #7d90ac; font-size: 11px; letter-spacing: .07em;
           text-transform: uppercase; }
.card .v { font-size: 23px; font-weight: 700; color: #f2f6fc; margin-top: 4px; }
.card .v.good { color: #6fe3a1; }
.card .v.warn { color: #ffb454; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }
th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid #172033; }
th { color: #8fa6c4; font-weight: 600; text-align: right;
     font-size: 11px; letter-spacing: .05em; text-transform: uppercase; }
td:first-child, th:first-child, td.l, th.l { text-align: left; }
tbody tr:hover { background: #0d1422; }
pre {
  background: #0d1422; border: 1px solid #1d2740; border-radius: 10px;
  padding: 14px 16px; overflow-x: auto; font-size: 12.5px; line-height: 1.55;
  color: #b8c9e0; font-family: "Cascadia Code", Consolas, monospace;
}
.tag {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 600;
}
.tag.ok { background: #10361f; color: #6fe3a1; }
.tag.miss { background: #3a1a1a; color: #ff7b72; }
.scroll { overflow-x: auto; }
footer { margin-top: 56px; color: #6d7f99; font-size: 12.5px;
         border-top: 1px solid #1d2740; padding-top: 16px; }
"""


# --------------------------------------------------------------------------- #
# Section 1: live engine evidence
# --------------------------------------------------------------------------- #
def engine_evidence(n_objects: int = 2000, n_steps: int = 60) -> dict:
    """Actually run the vectorized engine and capture what it produced."""
    objects = load_catalog(max_objects=n_objects)
    engine = PropagationEngine(objects)
    grid = build_time_grid(hours=n_steps / 60.0, step_minutes=1.0)

    started = time.perf_counter()
    result = engine.propagate(grid)
    elapsed = time.perf_counter() - started

    vectors = result.n_objects * result.n_steps
    live = np.flatnonzero(result.valid[:, 0])
    sample_index = int(live[0]) if live.size else 0
    position = result.positions[sample_index, 0, :]
    velocity = result.velocities[sample_index, 0, :]

    transcript = (
        f"engine        = SatrecArray({len(engine)} Satrec objects)\n"
        f"jd  array     = shape {grid.jd.shape}, dtype {grid.jd.dtype}\n"
        f"fr  array     = shape {grid.fr.shape}, dtype {grid.fr.dtype}\n"
        f"\n"
        f"errors, r, v  = sat_array.sgp4(jd, fr)\n"
        f"\n"
        f"errors        = shape {result.valid.shape}, "
        f"{int(np.count_nonzero(result.valid))} of {result.valid.size} "
        f"returned code 0\n"
        f"r (positions) = shape {result.positions.shape}, "
        f"dtype {result.positions.dtype}   # TEME km\n"
        f"v (velocities)= shape {result.velocities.shape}, "
        f"dtype {result.velocities.dtype}   # TEME km/s\n"
        f"\n"
        f"sample object : {engine.names[sample_index]} "
        f"(NORAD {int(engine.norad_ids[sample_index])})\n"
        f"  position r  = [{position[0]:12.4f}, {position[1]:12.4f}, "
        f"{position[2]:12.4f}] km\n"
        f"  velocity v  = [{velocity[0]:12.6f}, {velocity[1]:12.6f}, "
        f"{velocity[2]:12.6f}] km/s\n"
        f"  |r|         = {np.linalg.norm(position):.3f} km "
        f"-> altitude {np.linalg.norm(position) - config.EARTH_RADIUS_KM:.1f} km\n"
        f"  |v|         = {np.linalg.norm(velocity):.4f} km/s "
        f"(circular LEO is ~7.7)\n"
        f"\n"
        f"{vectors:,} state vectors in {elapsed:.3f} s "
        f"= {vectors / elapsed:,.0f} per second"
    )

    return {
        "objects": len(engine),
        "steps": result.n_steps,
        "vectors": vectors,
        "rate": vectors / elapsed if elapsed else 0.0,
        "dtype": str(result.positions.dtype),
        "shape": str(result.positions.shape),
        "transcript": transcript,
    }


# --------------------------------------------------------------------------- #
# Section 2: CDM validation
# --------------------------------------------------------------------------- #
def cdm_validation(alerts: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Recompute the official-CDM comparison from the alert table."""
    from logger import ConjunctionEvent

    events = [
        ConjunctionEvent(
            tca_utc=row.tca_utc,
            norad_1=int(row.norad_1),
            norad_2=int(row.norad_2),
            name_1=str(row.name_1),
            name_2=str(row.name_2),
            type_1=str(getattr(row, "type_1", "")),
            type_2=str(getattr(row, "type_2", "")),
            miss_distance_km=float(row.miss_distance_km),
            relative_speed_km_s=float(row.relative_speed_km_s),
            altitude_km=float(row.altitude_km),
        )
        for row in alerts.itertuples(index=False)
    ]

    catalog = load_catalog()
    propagated = {o.norad_id for o in catalog}

    stamps = pd.to_datetime(alerts["tca_utc"])
    window_start = stamps.min().to_pydatetime().replace(tzinfo=timezone.utc)
    window_end = stamps.max().to_pydatetime().replace(tzinfo=timezone.utc)

    cdms = parse_cdms(fetch_cdms())
    report = validate(
        events, cdms, propagated,
        window_start=window_start - timedelta(minutes=1),
        window_end=window_end + timedelta(minutes=1),
    )

    matches = pd.DataFrame(report.matches)
    misses = pd.DataFrame(report.misses)
    return report.summary(), matches, misses


def matches_table(matches: pd.DataFrame) -> str:
    if matches.empty:
        return "<p class='note'>No CDM encounters fell inside this run's window.</p>"

    rows = []
    for _, m in matches.iterrows():
        pair = m["pair"]
        names = m["names"]
        rows.append(
            "<tr>"
            f"<td class='l'>{html.escape(str(names[0]))} &times; "
            f"{html.escape(str(names[1]))}</td>"
            f"<td>{pair[0]} / {pair[1]}</td>"
            f"<td>{html.escape(str(m['cdm_tca_utc']))}</td>"
            f"<td>{html.escape(str(m['predicted_tca_utc']))}</td>"
            f"<td>{m['tca_residual_minutes'] * 60:.1f} s</td>"
            f"<td>{m['cdm_min_range_km']:.3f}</td>"
            f"<td>{m['predicted_miss_km']:.3f}</td>"
            f"<td>{m['residual_km']:+.3f}</td>"
            f"<td>{m['probability']:.2e}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr>"
        "<th class='l'>Objects</th><th>NORAD</th>"
        "<th>CDM TCA (official)</th><th>Our TCA</th><th>&Delta;t</th>"
        "<th>CDM miss (km)</th><th>Our miss (km)</th><th>Residual</th>"
        "<th>Pc</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


# --------------------------------------------------------------------------- #
# Section 3: performance
# --------------------------------------------------------------------------- #
def latest_run_summary() -> dict | None:
    files = sorted(glob.glob(str(config.LOG_DIR / "run_summary_*.json")))
    if not files:
        return None
    try:
        return json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def stages_table(summary: dict | None) -> str:
    if not summary or not summary.get("stages"):
        return "<p class='note'>No run summary found in logs/.</p>"
    rows = "".join(
        "<tr>"
        f"<td class='l'>{html.escape(str(s['stage']))}</td>"
        f"<td>{s['seconds']:.2f}</td>"
        f"<td>{s['share_pct']:.1f}%</td>"
        f"<td>{s['units_per_second']:,.0f}</td>"
        "</tr>"
        for s in summary["stages"]
    )
    return (
        "<div class='scroll'><table><thead><tr>"
        "<th class='l'>Stage</th><th>Seconds</th><th>Share</th>"
        "<th>Units/sec</th></tr></thead><tbody>" + rows + "</tbody></table></div>"
    )


def top_conjunctions_table(alerts: pd.DataFrame, limit: int) -> str:
    top = alerts.nsmallest(limit, "miss_distance_km")
    rows = "".join(
        "<tr>"
        f"<td class='l'>{html.escape(str(r.tca_utc))}</td>"
        f"<td class='l'>{html.escape(str(r.name_1))}</td>"
        f"<td class='l'>{html.escape(str(r.name_2))}</td>"
        f"<td>{r.miss_distance_km:.3f}</td>"
        f"<td>{r.relative_speed_km_s:.2f}</td>"
        f"<td>{r.altitude_km:,.0f}</td>"
        "</tr>"
        for r in top.itertuples(index=False)
    )
    return (
        "<div class='scroll'><table><thead><tr>"
        "<th class='l'>TCA (UTC)</th><th class='l'>Object 1</th>"
        "<th class='l'>Object 2</th><th>Miss (km)</th>"
        "<th>Rel (km/s)</th><th>Alt (km)</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table></div>"
    )


def card(key: str, value: str, tone: str = "") -> str:
    cls = f" {tone}" if tone else ""
    return (
        f"<div class='card'><div class='k'>{key}</div>"
        f"<div class='v{cls}'>{value}</div></div>"
    )


# --------------------------------------------------------------------------- #
def build_html(
    evidence: dict,
    summary: dict,
    matches: pd.DataFrame,
    misses: pd.DataFrame,
    alerts: pd.DataFrame,
    run: dict | None,
    top: int,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    recall = summary["recall"]
    residual = summary["miss_distance_residual_km"]
    tca_res = summary["tca_residual_minutes"]

    engine_cards = (
        card("Objects propagated", f"{evidence['objects']:,}")
        + card("State vectors", f"{evidence['vectors']:,}")
        + card("SGP4 rate", f"{evidence['rate']:,.0f}/s", "good")
        + card("Array dtype", evidence["dtype"])
    )

    val_cards = (
        card("CDMs in scope", str(summary["considered_cdms"]))
        + card("Matched", str(summary["matched_cdms"]),
               "good" if summary["matched_cdms"] else "warn")
        + card("Recall", f"{recall:.3f}", "good" if recall >= 0.9 else "warn")
        + card(
            "Median miss residual",
            f"{residual['median_abs']:.3f} km" if residual["median_abs"] is not None
            else "n/a",
        )
        + card(
            "Mean TCA residual",
            f"{tca_res['mean_abs'] * 60:.1f} s" if tca_res["mean_abs"] is not None
            else "n/a",
        )
    )

    run_cards = ""
    if run:
        run_cards = (
            card("Window", f"{run.get('window_hours', 0):.0f} h")
            + card("Objects", f"{run.get('objects', 0):,}")
            + card("State vectors", f"{run.get('state_vectors', 0):,}")
            + card("Wall clock", f"{run.get('wall_clock_seconds', 0):,.1f} s")
            + card("Workers", str(run.get("workers", 1)))
            + card("Peak RSS", f"{run.get('resident_memory_mb', 0):,.0f} MB")
        )

    miss_note = ""
    if not misses.empty:
        items = "".join(
            f"<li>{html.escape(str(m['names'][0]))} &times; "
            f"{html.escape(str(m['names'][1]))} "
            f"&mdash; CDM {m['cdm_min_range_km']:.3f} km at {m['tca_utc']}</li>"
            for _, m in misses.iterrows()
        )
        miss_note = f"<h3>Missed CDM encounters</h3><ul>{items}</ul>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Orbital Sentinel &mdash; Results</title>
<style>{CSS}</style></head><body><div class="wrap">

<header>
  <h1>Project Orbital Sentinel &mdash; Results</h1>
  <div class="sub">Generated {generated} &middot; {len(alerts):,} conjunction
  alerts in <code>conjunction_alerts.csv</code></div>
</header>

<h2>1 &nbsp;Engine evidence &mdash; SGP4 + NumPy</h2>
<p class="note">Run live when this page was generated. The transcript below is
the actual output of <code>sgp4.api.SatrecArray.sgp4()</code> called on NumPy
float64 arrays &mdash; shapes, dtypes and a real state vector, not a summary of
what the code intends to do.</p>
<div class="cards">{engine_cards}</div>
<pre>{html.escape(evidence['transcript'])}</pre>
<p class="note">A single <code>sgp4()</code> call returns the
<code>[N, T]</code> error matrix and the <code>[N, T, 3]</code> position and
velocity tensors. No Python loop touches an individual object, which is what
makes full-catalog propagation tractable.</p>

<h2>2 &nbsp;Validation against official CDMs</h2>
<p class="note">Ground truth is the 18th Space Defense Squadron
<code>cdm_public</code> feed. Our predictions come from public TLEs; theirs come
from high-precision ephemerides and operator data, so exact agreement is not
expected &mdash; the residual columns show how close we get.</p>
<div class="cards">{val_cards}</div>
{matches_table(matches)}
{miss_note}
<p class="note"><strong>Precision is deliberately not headlined.</strong>
<code>cdm_public</code> exposes only the ~100 most recent public messages, so
the denominator is truncated ground truth rather than a count of false alarms.
Recall &mdash; restricted to CDMs whose objects were both propagated and whose
TCA falls inside the window &mdash; is the defensible metric.</p>

<h2>3 &nbsp;Pipeline performance</h2>
<div class="cards">{run_cards}</div>
{stages_table(run)}

<h2>4 &nbsp;Highest-risk conjunctions</h2>
<p class="note">Closest {top} of {len(alerts):,} flagged encounters, after
deduplication and sub-second TCA refinement.</p>
{top_conjunctions_table(alerts, top)}

<footer>
Orbital Sentinel &middot; vectorized SGP4 propagation, cKDTree spatial
screening, CDM validation. Source data: Space-Track.org.
</footer>

</div></body></html>"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="report",
        description="Generate an evidence report for the pipeline results.",
    )
    parser.add_argument("--input", default="conjunction_alerts.csv")
    parser.add_argument("--output", default="results.html")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--evidence-objects", type=int, default=2000,
        help="Objects used for the live SGP4 demonstration (default: 2000).",
    )
    parser.add_argument("--open", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.input)
    if not source.exists():
        raise SystemExit(
            f"No alert file at {source}. Run the pipeline first: "
            "python main.py --full"
        )

    alerts = pd.read_csv(source)
    if alerts.empty:
        raise SystemExit(f"{source} has no rows to report on.")

    log.info("Running live SGP4 evidence pass.")
    evidence = engine_evidence(n_objects=args.evidence_objects)

    log.info("Recomputing CDM validation.")
    summary, matches, misses = cdm_validation(alerts)

    run = latest_run_summary()
    page = build_html(evidence, summary, matches, misses, alerts, run, args.top)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    log.info(
        "Wrote %s (%.0f KB) | recall %.3f on %d CDMs",
        output,
        output.stat().st_size / 1024,
        summary["recall"],
        summary["considered_cdms"],
    )

    if args.open:
        import webbrowser

        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
