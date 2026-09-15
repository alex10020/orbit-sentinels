# Project Orbital Sentinel

A vectorized orbital propagation and 3D spatial-indexing engine that forecasts
LEO satellite and debris conjunctions over a rolling 72-hour window.

The design target is throughput. Two bottlenecks decide whether a full-catalog
screen is feasible in Python, and both are pushed out of the interpreter:

| Bottleneck | Naive approach | What this engine does |
|---|---|---|
| **Physics** | Loop `Satrec.sgp4()` per object, per time step | One `SatrecArray.sgp4()` call propagates all *N* objects across all *T* timestamps inside compiled C++ |
| **Search** | O(N²) all-pairs distance (~400M comparisons *per time step* at N=28,000) | O(N log N) `scipy.spatial.cKDTree.query_pairs()` returns only pairs already inside the screening radius |

A third problem is less obvious than either of those, and it is the one that
actually decides whether the output is worth anything. See
[Two-stage screening](#two-stage-screening-the-bug-that-matters-most).

---

## Results from a full run

72-hour forecast, 1-minute steps, 5 km screening threshold, whole LEO catalog:

```
28,301 objects x 4,320 time steps = 122,260,320 state vectors
```

The top of the risk table is populated by exactly the objects operational SSA
cares about — Starlink shell crossings, and debris from the Fengyun-1C ASAT
test, the Iridium 33 / Cosmos 2251 collision, and the CZ-6A upper-stage
breakups:

```
TCA (UTC)             OBJECT 1                   OBJECT 2               MISS km  REL km/s   ALT km
2026-09-15 16:17:46   52491 STARLINK-3908        66900 STARLINK-36074     0.352     5.551      468
2026-09-15 15:51:47   54592 CZ-6A DEB            64523 KUIPER-00063       1.067    14.379      625
2026-09-15 15:53:47    6157 THORAD AGENA D DEB   54965 CZ-6A DEB          1.528    14.045      807
2026-09-15 15:49:47   33850 IRIDIUM 33 DEB       38725 FENGYUN 1C DEB     1.895    13.548      756
```

---

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on POSIX
pip install -r requirements.txt

cp .env.example .env           # then fill in your Space-Track credentials

python main.py                 # 10-minute smoke test, first 1,000 objects
python main.py --full          # full 72-hour forecast over the LEO catalog
python test_sentinel.py        # 21 correctness tests, no pytest required
```

### Useful invocations

```bash
python main.py --hours 6 --max-objects 5000     # medium run
python main.py --full --refresh                 # force a fresh catalog download
python main.py --full --threshold 10            # widen the screening radius
python main.py --full --no-refine --no-validate # fastest possible screen
```

Run `python main.py --help` for the full flag list.

---

## Architecture

```
orbital-sentinel/
├── config.py          Constants, .env loading, tunable thresholds
├── ingestion.py       Space-Track auth, rate-limited fetch, caching, TLE parsing
├── propagation.py     SatrecArray vectorized SGP4 + sub-second TCA refinement
├── spatial_index.py   cKDTree construction and pairwise conjunction query
├── validator.py       Benchmarking against official 18th SDS CDMs
├── logger.py          Profiling, alert tables, JSON/CSV output
├── main.py            Pipeline orchestrator and CLI
└── test_sentinel.py   Correctness tests
```

Data flows `Ingestion -> SatrecArray -> cKDTree -> Conjunction Log -> Metrics`.

### Outputs

Every run writes to `logs/` (gitignored):

- `conjunctions_<run>.csv` / `.json` — the ranked encounter list
- `run_summary_<run>.json` — throughput, per-stage profile, validation metrics
- `sentinel_<run>.log` — full run transcript

The catalog is cached to `data/latest_catalog.json` and reused for
`CATALOG_MAX_AGE_HOURS` (default 8), so repeated runs do not re-hit the API.
If a download fails, the engine falls back to a stale cache rather than dying.

---

## Two-stage screening: the bug that matters most

Querying the KDTree directly at the 5 km miss-distance threshold **does not
work on a 1-minute grid, and it fails silently.**

Two objects closing at 14 km/s travel 840 km between samples. The nearest grid
sample to their true closest approach can therefore be hundreds of km away from
it, and a 5 km query sees nothing at all. Working the arithmetic backwards, a
naive query on a 1-minute grid can only ever detect encounters slower than
about **0.17 km/s** — which excludes essentially every dangerous debris
conjunction, since those are exactly the high-relative-velocity ones.

This was not theoretical. The first full run of this engine produced 5,282
"conjunctions" and matched **zero** of the 20 official CDM encounters that were
in scope. Independently propagating those 20 pairs confirmed all of them were
detectable: TLE-derived separations of 0.1–3.7 km, all well inside the
threshold. The pipeline was blind to them purely because of grid sampling.

The fix is the standard SSA screening-volume approach, in two stages:

1. **Coarse pass.** Query the tree at a radius large enough to bracket any
   encounter reachable within one step: `R = threshold + v_max · dt/2`, which
   is **485 km** for a 1-minute step and a 16 km/s maximum closing speed. Any
   pair that conjuncts during the step must appear in this result.
2. **Fine pass.** For each candidate, solve for the closest approach assuming
   linear relative motion across the step. For relative position `Δr` and
   relative velocity `Δv`, the minimising offset is `t* = −(Δr·Δv)/|Δv|²`,
   clamped to the half-step that this sample owns — so consecutive steps tile
   the timeline without gaps or double-counting. Keep only pairs whose miss
   distance at `t*` is within the threshold.

Stage 2 is closed-form and fully vectorized in NumPy, which is what makes this
affordable: it reduces ~308,000 candidate pairs per step to a handful, with no
Python-level loop and no extra SGP4 calls.

**Result: recall against the in-scope CDMs went from 0/20 to 20/20.**

The surviving TCA estimates are already sub-step accurate. `refine_tca()` then
polishes the closest encounters against true SGP4 geometry (below).

---

## Physical filtering — why the raw output is wrong without it

An unfiltered 5 km screen over the catalog is dominated by results that are not
conjunctions at all. Three filters make the output meaningful; each is
configurable and each was added in response to a specific false-positive class
observed in real runs.

**1. Relative-velocity gate (`--min-rel-speed`, default 0.05 km/s).**
The single most important filter. Docked station modules, co-deployed payloads
and duplicate element sets sit permanently within metres of each other. Without
this gate the top of the table is ISS (ZVEZDA) vs ISS (DESTINY), CSS (TIANHE-1)
vs CSS (WENTIAN), and similar — reported at 0.000 km miss distance and
0.000 km/s relative speed, because they are the same physical structure. A
genuine LEO conjunction between independent objects closes at hundreds of m/s
to ~15 km/s. On a one-hour full-catalog run this filter alone collapsed 4,731
raw detections into 54 real ones.

**2. Epoch-age gate (`--max-epoch-age`, default 14 days).**
About 8.8% of the catalog carries element sets that are years stale. Deep-space
probes are the dangerous case: PSYCHE, LUCY and EUROPA CLIPPER all still list a
`DECAY_DATE` of null under their original launch-phase TLE (epoch age 700–1,800
days, perigee ~160 km). SGP4 will happily propagate those into a dense,
entirely fictitious LEO trajectory that collides with everything. SGP4 accuracy
also degrades quickly past epoch in general, so this doubles as the veracity
control.

**3. Deep-space exclusion (`--include-deep-space` to disable).**
Uses SGP4's own classification (`satrec.method == 'd'`, orbital period ≥ 225
minutes) to drop GEO, Molniya and lunar-transfer objects, which are outside the
LEO screening regime.

A note on the LEO filter: objects are admitted on **perigee**, not apogee, so
highly eccentric objects that dip through the LEO shell are correctly retained
even when their apogee is far above the 2,000 km ceiling.

---

## Time of Closest Approach refinement

The linear-motion solve in stage 2 of the screen is an approximation: it
ignores orbital curvature across the step. After deduplication, the closest
encounters are re-examined with a golden-section search on the true SGP4
separation function, which is smooth and unimodal within a single encounter.

This resolves the TCA to **0.05 s**. Because it evaluates real SGP4 geometry
rather than a linear extrapolation, its answer supersedes the estimate whether
it comes out tighter *or* looser — accepting it only when it improved the
number would bias every reported miss distance downward. Disable with
`--no-refine`.

---

## Validation against official CDMs

`validator.py` benchmarks predictions against Conjunction Data Messages from
the 18th Space Defense Squadron (`cdm_public`). Three details matter for
reading the metrics:

- **`MIN_RNG` is published in metres**, not kilometres. The validator converts.
- **CDMs are mirrored and re-issued.** Space-Track publishes each encounter
  twice with `SAT_1`/`SAT_2` swapped, plus a new message each time the solution
  is refined — 100 raw messages collapsed to 21 distinct encounters in testing.
  Counting those separately would inflate the recall denominator, so they are
  deduplicated on (unordered pair, TCA minute).
- **Precision is a lower bound, not a measurement.** `cdm_public` exposes only
  the ~100 most recent *public* messages. It is a truncated sample of the real
  conjunction set, so a prediction with no matching CDM is not necessarily a
  false alarm. Recall, restricted to CDMs whose objects were both propagated
  and whose TCA falls inside the forecast window, is the defensible metric.

---

## Performance notes

Memory is the binding constraint, not CPU. A full 72-hour window at once would
allocate `N x T x 3 x 8` bytes twice over — **5.8 GB** at 30,000 objects — so
propagation is chunked along the time axis, sized to a 2 GB budget by default
(`--chunk-steps`). Resident memory stays flat across the run regardless of
window length.

Measured on the reference machine: SGP4 propagation sustains ~3M state
vectors/sec. The spatial search, not the physics, dominates wall-clock time at
full catalog size — each time step rebuilds a fresh cKDTree over ~28,000 points.

---

## Security

`.env` is gitignored and contains live Space-Track credentials. `data/` and
`logs/` are gitignored too. Never commit any of them. If a credential is ever
exposed, rotate it at <https://www.space-track.org>.

---

## Further optimization

Not yet implemented, in rough order of expected payoff:

1. **Parallel time-chunk propagation** — chunks are independent, so
   `multiprocessing` can distribute the 72-hour window across cores. This is
   the largest remaining win: the coarse tree query now dominates wall-clock.
2. **float32 positions for the tree** — halves memory traffic in the search
   stage; 5 km resolution does not need float64 for the *screening* pass, only
   for the exact-distance and refinement passes.
3. **Conjunction-window interpolation** — replace the per-pair SGP4 calls in
   refinement with Hermite interpolation over the existing state vectors.
   A curvature-aware quadratic solve in stage 2 would also let the screening
   radius shrink.
4. **Apogee/perigee pre-screening** — two objects whose altitude bands do not
   overlap can never conjunct, which prunes candidate pairs before the tree.
