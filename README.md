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

72-hour forecast, 1-minute steps, 5 km screening threshold, whole LEO catalog,
across 22 CPU cores:

| | |
|---|---|
| Objects propagated | 28,298 |
| Time steps | 4,320 |
| State vectors computed | **122,247,360** |
| Wall clock | **98.5 s** (6.1x faster than single-process 602 s) |
| Screening throughput | 1.42M state vectors/sec across 22 cores |
| End-to-end rate | 1.24M state vectors/sec |
| Brute-force comparisons avoided | ~1.73 x 10^12 |
| Resident memory | 352 MB (parent), flat across the run |
| Distinct encounters within 5 km | 184,624 |

### Accuracy against official ground truth

Validated against 18th Space Defense Squadron CDMs, using only public TLEs
against their high-precision ephemerides:

| | |
|---|---|
| **Recall** | **20 / 20 = 1.000** |
| Miss-distance residual | median **0.228 km**, mean 0.576, max 2.83 |
| TCA residual | mean **0.6 s**, max 1.2 s |

The top of the risk table is populated by exactly the objects operational SSA
cares about -- Starlink shell crossings, and debris from the CZ-6A upper-stage
breakups:

```
TCA (UTC)             OBJECT 1                   OBJECT 2               MISS km  REL km/s   ALT km
2026-09-17 13:09:51   58744 STARLINK-31167       67996 STARLINK-36979     0.008     2.292      479
2026-09-17 04:14:47   46457 JILIN-01 GAOFEN 3E   54352 CZ-6A DEB          0.011    14.044      347
2026-09-18 13:59:29   54841 STARLINK-4659        63399 STARLINK-33531     0.016     2.330      487
2026-09-18 01:45:39   57716 STARLINK-30281       58121 STARLINK-30799     0.019     1.279      488
```

### Reading these numbers honestly

**Precision is reported as 0.000, and that number is meaningless.** The engine
flags 184,624 encounters and only 20 CDMs exist to corroborate them, because
`cdm_public` exposes just the ~100 most recent *public* messages. The
denominator is truncated ground truth, not a count of false alarms. Recall is
the defensible metric; see [Validation](#validation-against-official-cdms).

**184,624 encounters over 72 hours is physically correct, not a bug.** The
kinetic-theory estimate for a 5 km cross-section across 28,298 objects in the
LEO shell predicts ~10^5 events over this window -- the same order. The lesson
is that **5 km is a screening volume, not an alert threshold.** Operational SSA
ranks candidates by probability of collision (the CDMs here carry Pc ~ 10^-4),
not by raw miss distance. The ranked table is the actionable output; the full
list is the screen that feeds it.

---

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on POSIX
pip install -r requirements.txt

cp .env.example .env           # then fill in your Space-Track credentials

python main.py                 # 10-minute smoke test, first 1,000 objects
python main.py --full          # full 72-hour forecast over the LEO catalog
python visualize.py --open     # render index.html and open it
python test_sentinel.py        # 28 correctness tests, no pytest required
```

### Useful invocations

```bash
python main.py --hours 6 --max-objects 5000     # medium run
python main.py --full --refresh                 # force a fresh catalog download
python main.py --full --threshold 10            # widen the screening radius
python main.py --full --no-refine --no-validate # fastest possible screen
python main.py --full --workers 8               # cap the process pool
python main.py --full --no-parallel             # single-process path
python main.py --full --step 0.5                # faster AND more accurate
```

Run `python main.py --help` for the full flag list.

---

## Architecture

```
orbital-sentinel/
├── config.py          Constants, .env loading, tunable thresholds
├── ingestion.py       Space-Track auth, rate-limited fetch, caching, TLE parsing
├── propagation.py     SatrecArray vectorized SGP4 + sub-second TCA refinement
├── spatial_index.py   cKDTree construction and two-stage conjunction query
├── parallel.py        Multiprocessing pool over independent time chunks
├── validator.py       Benchmarking against official 18th SDS CDMs
├── logger.py          Profiling, alert tables, pandas/JSON/CSV output
├── main.py            Pipeline orchestrator and CLI
├── visualize.py       Plotly 3D globe -> index.html
└── test_sentinel.py   Correctness tests
```

Data flows `Ingestion -> SatrecArray -> cKDTree -> Conjunction Log -> Metrics`.

### Outputs

The headline artifact is **`conjunction_alerts.csv`** in the project root: the
final, filtered, deduplicated alert table written via pandas, sorted closest
approach first. It carries the TEME Cartesian position (`x_km`, `y_km`, `z_km`)
of each encounter so `visualize.py` can plot it without re-running the
propagator. Everything else lands in `logs/` (gitignored):

- `conjunctions_<run>.csv` / `.json` — the same events, timestamped per run
- `run_summary_<run>.json` — throughput, per-stage profile, validation metrics
- `sentinel_<run>.log` — full run transcript

`visualize.py` reads the alert CSV and writes a standalone **`index.html`**:
Earth as a shaded globe with a wireframe graticule, every conjunction plotted
at its true 3D position, coloured by miss distance, with hover labels naming
both objects. Plotly is inlined, so the page works offline with no server.

Because a full screen yields ~184,000 encounters and a browser will not enjoy
all of them at once, the default plots the closest 15,000 (`--limit 0` for
everything).

All storage is local. There is no database, cloud or otherwise, in the
pipeline.

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

## Parallel execution

Time chunks are independent -- each needs nothing from its neighbours and
produces its own event list -- so the screen fans out across cores with
`multiprocessing.Pool`. Measured on the full 72-hour window, 22 cores:

| | Wall clock |
|---|---|
| Single process (`--no-parallel`) | 602 s |
| 22 worker processes (default) | **98.5 s** |
| | **6.1x** |

Both paths produce **byte-identical** event lists; a test asserts this rather
than trusting it.

Three details were not optional:

**`Satrec` objects cannot be pickled.** The parent cannot ship a built
`SatrecArray` to a worker. Each worker instead receives the raw TLE text once
through the pool initializer and builds its own `SatrecArray` in its own
address space -- paid once per worker, not once per chunk.

**Chunk size needs a floor as well as a ceiling.** Memory bounds a chunk from
above (~120 steps), but sizing by memory alone gave a 30-step window a single
chunk, so one worker did everything while 21 idled. Sizing by worker count
alone was worse: it drove chunks down to *one step each*, which throws away the
vectorization the whole engine is built on and leaves only pickling overhead --
measured at zero speedup. Chunks are now sized to fill the pool, floored at 8
steps, and capped by the memory budget.

**Small jobs must skip the pool entirely.** Starting 22 workers that each parse
28,000 TLEs costs more than a short screen saves: a 30-step job took 19.8 s
through the pool and 1.0 s without it. Jobs below ~2M state vectors take the
single-process path automatically.

Why 6.1x and not 22x: worker startup is a fixed ~10 s, and the search is
memory-bandwidth bound, so cores contend for the same memory rather than
scaling linearly.

---

## Performance notes

**Memory is bounded by chunking, not by luck.** A full 72-hour window
propagated at once would allocate `N x T x 3 x 8` bytes twice over -- **5.8 GB**
at 30,000 objects -- so propagation is chunked along the time axis, sized to a
2 GB budget by default (`--chunk-steps`). Resident memory stays flat at ~468 MB
regardless of window length.

**The search dominates, not the physics.** In the single-process profile SGP4
propagation is 6.8% of wall clock and the spatial search is 92.1%. It is
memory-bandwidth bound on roughly 1.3 billion candidate pairs gathered across
the run, not compute bound -- which is why parallel scaling falls short of
linear.

**Finer time steps are cheaper, which is counterintuitive.** The screening
radius grows linearly with the step (`v_max * dt/2`), so candidate pairs per
step grow as roughly `dt^3`, while the number of steps only falls as `1/dt`.
Total work therefore *drops* as the step shrinks, until fixed per-step tree
construction takes over. Measured over a fixed forecast span:

| Step | Screening radius | Relative cost |
|---|---|---|
| 2.0 min | 965 km | 2.5x |
| 1.0 min | 485 km | 1.2x |
| **0.5 min** | **245 km** | **1.0x** (optimum) |
| 0.25 min | 125 km | 1.1x |

So `--step 0.5` is both faster than the 1-minute default *and* more accurate,
since it shortens the linear-motion extrapolation in stage 2. The default stays
at 1 minute to match the specification.

---

## Security

`.env` is gitignored and contains live Space-Track credentials. `data/` and
`logs/` are gitignored too. Never commit any of them. If a credential is ever
exposed, rotate it at <https://www.space-track.org>.

---

## Further optimization

Not yet implemented, in rough order of expected payoff:

1. ~~Parallel time-chunk propagation~~ — **done**, 6.1x on 22 cores. See
   [Parallel execution](#parallel-execution).
2. **float32 positions for the tree** — halves memory traffic in the search
   stage; 5 km resolution does not need float64 for the *screening* pass, only
   for the exact-distance and refinement passes.
3. **Conjunction-window interpolation** — replace the per-pair SGP4 calls in
   refinement with Hermite interpolation over the existing state vectors.
   A curvature-aware quadratic solve in stage 2 would also let the screening
   radius shrink.
4. **Apogee/perigee pre-screening** — two objects whose altitude bands do not
   overlap can never conjunct, which prunes candidate pairs before the tree.
