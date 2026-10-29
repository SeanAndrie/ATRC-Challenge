# ATRC Challenge — Abu Dhabi hotel demand and air arrivals

**Submission deliverable.** A transparent parametric demand chain
(`seats → PAX → P2P → New Arrivals → guest-nights`) and a hybrid LightGBM layer
that replaces the fixed capture ratio, built on the licensed competition data pack.
Start at [`docs/METHODS.md`](docs/METHODS.md).

## Submission at a glance

| Piece | Where |
| --- | --- |
| Methods, results and decisions (start here) | [`docs/METHODS.md`](docs/METHODS.md) |
| Deck outline, 90-second demo script, system diagram, planner playbook | [`docs/SUBMISSION_PACK.md`](docs/SUBMISSION_PACK.md) |
| Hybrid layer: parametric chain + LightGBM (Phases 3–4) | `notebooks/hybrid_layer.py` |
| Competition-pack loader | `pipeline/competition/` |
| Interactive simulator | `app/simulator.py` (Streamlit app) |

The models run **entirely on the licensed competition pack**. An earlier public-data
investigation (Phase 1) returned a negative result — air arrivals are not a usable
driver of hotel demand — and is retained as **background evidence**, not part of
the deliverable. Short version: a seasonal+trend model that never sees arrivals
scores 5.2% WMAPE on held-out years, while adding arrivals worsens it to 12.6%;
the elasticity is 0.170 ± 0.176. See
[Phase 1 background](docs/METHODS.md#phase-1--public-data-investigation-background).

## Layout

```
data/raw/          downloaded sources            (gitignored)
data/interim/      extracted PDF text + manifest (tracked)
data/processed/    14 analysis-ready tables      (tracked)
pipeline/paths.py  every filesystem path in the project
pipeline/dct/      DCT PDF parsing, by report era
pipeline/aviation/ passenger and ADS-B builders
pipeline/hotel_supply.py
pipeline/calendar_events.py   UAE event calendar (Ramadan, Eid, National Day, NYE)
pipeline/hybrid.py            canonical chain + LightGBM layer (notebook and simulator share it)
pipeline/simulator.py         Scenario/Simulator API for the Streamlit app
pipeline/sensitivity.py       empirically measured lever spreads + Sobol (sensitivity view 2)
pipeline/tests/    ten standalone diagnostics
notebooks/         four marimo notebooks
docs/              METHODS.md and the organisers' brief
tools/             one-off utilities + the OpenSky puller
outputs/           generated reports + scratch (gitignored);
                   outputs/models/simulator_full.pkl is tracked
```

No module hardcodes an absolute path. `pipeline/paths.py` owns them all, and
importing it also puts the pipeline directories on `sys.path`.

## Setup

```
uv sync                      # or: python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -c "import pandas"    # confirm the environment
```

Python 3.14. `pytest` is **not** installed — the diagnostics are plain scripts,
run them directly. `scipy` is present (pulled in by `statsmodels`), and the
competition hybrid model uses `lightgbm`.

`credentials.json` (gitignored) holds the OpenSky API credentials.

## Reproducing the data

```
.venv/bin/python pipeline/dct/dct_fetch_manifest.py
.venv/bin/python pipeline/dct/dct_download.py
.venv/bin/python pipeline/dct/dct_fingerprint.py
.venv/bin/python pipeline/dct/dct_extract_text.py
.venv/bin/python pipeline/dct/dct_parse.py
.venv/bin/python pipeline/hotel_supply.py
.venv/bin/python pipeline/aviation/build_annual_arrivals.py
.venv/bin/python pipeline/aviation/build_adsb_2025.py
.venv/bin/python pipeline/aviation/calibrate_adsb_2025.py
```

`data/raw/` is gitignored, so a clean clone starts empty. The **competition pack
must be placed in `data/raw/competition/`** — it is licensed for this competition
only and is never committed. Its expected files and checksums are in
[`docs/METHODS.md` §2.1](docs/METHODS.md#21-loader-and-integrity). The steps above
build the public-data tables, which are Phase 1 background only; the OpenSky pull
in particular is quota-limited and may not be repeatable.

## Checking the work

For the submission gate, run one command:

```
scripts/check.sh          # app logic + sensitivity + packaging pins + attribution guard
```

The individual diagnostics:

```
.venv/bin/python pipeline/dct/dct_validate.py        # reconciliation
.venv/bin/python pipeline/tests/test_market_link.py
.venv/bin/python pipeline/tests/test_aggregate_link.py
.venv/bin/python pipeline/tests/test_retarget_international.py
.venv/bin/python pipeline/tests/test_p2p_vs_pax.py
.venv/bin/python pipeline/tests/test_calendar_events.py
.venv/bin/python pipeline/tests/test_market_attribution.py   # capture > 1.0 in 5 markets
.venv/bin/python pipeline/tests/test_market_origin_validation.py   # pack region mix vs SCAD embarkation mix
.venv/bin/python pipeline/tests/test_simulator_package.py   # packaging gate; writes no report
.venv/bin/python pipeline/tests/test_sensitivity.py         # empirical lever spreads; writes no report
.venv/bin/python pipeline/tests/test_simulator_app.py       # app logic, charts, impact windows
.venv/bin/python pipeline/tests/crosscheck_markets.py
```

`dct_validate.py` sums each year's monthly series and compares it against the
annual total printed in the same report, then checks an external anchor: the
December 2017 DCT report states 4,875,499 guests, and the parsed months sum to
4,874,253 — a 0.026% difference.

The diagnostics write into `outputs/`. Each report is written fresh by its
script on every run and is gitignored; the commands above name the generators.
`test_simulator_package.py` writes no report: it re-fits the packaged model and
pins it to the notebook's numbers, so the simulator cannot silently drift from
the deliverable. `test_sensitivity.py` writes no report and pins the
empirical-sensitivity module on a synthetic panel (METHODS §3.4).
`test_simulator_app.py` writes no report and pins the app's pure logic (impact
windows, seasonal shape, market × season matrix, attribution range, efficiency
ratio, chart builders).

## Streamlit simulator

The parametric chain and the hybrid layer are packaged as an importable API for
the interactive front end. `app/simulator.py` is the working Streamlit app:

```python
from pipeline.simulator import Simulator, Scenario
sim = Simulator().fit()                       # trains only the requested window
sim.monthly(Scenario(seats_mult=1.2))         # aggregate view — safe by default
sim.market(Scenario(seats_mult=1.2))          # market view — flagged supply-side-only
sim.save()                                    # cache -> outputs/models/
```

Supply levers (`seats_mult`, `lf_mult`, `p2p_mult`) rewrite the supply frame, so
the hybrid's features and Poisson exposure respond non-linearly;
`capture_mult` / `season_mult` / `alos_mult` are post-hoc dials. The hybrid
path's guest-nights use the **learned ALOS** (a 5-seed regression on the model's
features), while `chain()` keeps the fixed per-market `ALOS[m]` so the
parametric layer stays interpretable. Fitting is lazy and per-window (`full` for
scenarios, `competition` for the final test-window replay); a fitted simulator
can be pickled with `save()`/`load()` to spare the refit.

The app adds scenario controls the brief asks for: a monthly **seasonal shape**,
a **seasonal market mix**, a **suspend market** structural zero, a route editor
with route-level frequency/closure changes, an **impact window** (full / first 12 months after change / last 12 months /
calendar year / season), a **market × season heatmap**, a **guests-per-1,000-seats**
ratio, and an **attribution range** that bounds each market's guest nights across
the last-departure, demand-mix and supply-mix allocations. Because of the
market-join defect in §4.5, `market()` rows carry
`attrs["data_status"] == "supply-side-only"` (and a matching `data_status`
column) and scenarios cannot invent new markets, zero a global supply lever, or
run past the event calendar's 2026-02 horizon. A market can be **suspended**
explicitly (a structural zero), and the hybrid forces zero arrivals when seats
are zero. See [`METHODS.md` §Reproducibility](docs/METHODS.md#reproducibility).

## Notebooks

```
.venv/bin/python -m marimo edit notebooks/hybrid_layer.py    # the deliverable
.venv/bin/python -m marimo edit notebooks/sensitivity_empirical.py # §3.4 empirical view
.venv/bin/python -m marimo edit notebooks/linkage_study.py   # Phase 1 evidence
.venv/bin/python -m marimo edit notebooks/flight_analysis.py # Phase 1 evidence
```

`hybrid_layer.py` is the submission: Part A is the parametric chain, Part B the
hybrid LightGBM layer, both scored on one panel in one run.
`sensitivity_empirical.py` is the §3.4 companion that measures each lever's actual
month-to-month spread and reruns the Sobol on those bounds.
`linkage_study.py` is
the main public-data analysis and `flight_analysis.py` is a read of the OpenSky
ADS-B extract; both are Phase 1 background kept as evidence, and neither is part
of the deliverable. The OpenSky puller is a marimo notebook too, but it lives in
`tools/fetch_opensky.py` because it is data acquisition, not analysis — it is the
only thing in the repo that spends OpenSky credits.

To run the deliverable end to end without a browser:

```
.venv/bin/python -m marimo export html --no-include-code notebooks/hybrid_layer.py -o out.html
```

That trains 15 LightGBM models, so it is not instant.

## The simulator app

```
.venv/bin/streamlit run app/simulator.py
```

Tabs: **Scenario** (levers, route edits, seasonal shape/mix, waterfall),
**By market & season** (heatmap, attribution range, guests per seat),
**Which levers matter** (a-priori + empirical Sobol), **Validation** (live final
test window and last-year holdout, monthly and seasonal weighted error),
**New origin (proxy)** (parametric range estimate for countries not in the 33
markets), and **Assumptions & limits**. The app loads
the tracked fitted artifact if present, else fits once (~1 min). For a
non-interactive smoke test use `.venv/bin/streamlit run app/simulator.py
--server.headless true`.



## Things to know before reading the numbers

- **The final test window has no `Guests` column.** Only New Arrivals can be
  scored out-of-sample on 2025-08 → 2026-02; guest-nights are validated in-sample
  and in the last-year holdout.
- **Capture is behavioural, not mechanical.** `New Arrivals / P2P` absorbs
  visiting-friends-and-relatives travel, booking behaviour and non-air hotel
  demand. Move it as a lever with its spread in view.
- **Purpose-of-travel is unmeasured.** No source publishes it at any frequency, so
  the visitor split inside capture is an assumption, never an estimate.
- **12 hotel-only markets** (Australia, Brazil, Czechia, Denmark, Finland, Mexico,
  Morocco, Norway, Pakistan, Romania, South Africa, Sweden) have no flight supply
  and are carried as a separately reported residual.
- **The panel is 1,405 market-months** (33 markets, 43 months). That is why the
  ML model is deliberately shallow.
- **If you touch the public-data tables** (`dct_kpis.csv` and friends): they carry a
  `unit` column that changes mid-series, 68 rows are `derived == 1`, ADS-B sees only
  48.9–61.5% of official arrivals, and 203 2025-Q1 legs are `is_local_leg`
  repositioning that must be filtered out.

## Background — what the public data showed (Phase 1)

Summarised in [`docs/METHODS.md` Phase 1](docs/METHODS.md#phase-1--public-data-investigation-background)
and the generated reports in `outputs/`. This is background evidence; the
submission itself is Phase 2–4.

| Question | Answer |
|---|---|
| Do arrivals predict hotel guests? | Not beyond seasonality. Elasticity 0.170 (se 0.176). |
| Market-level or aggregate? | Aggregate only, and weakly. Market-level conversion spreads 36x across 9 markets. |
| Are hotel guests internally consistent? | Yes. `guests x ALOS = guest_nights` to within 0.174%. |
| Is there a true international series? | Only 2024, and it comes from narrative prose. |
| Can the ex-UAE proxy replace it? | No. It overstates true international share by 12.8, 20.3 and 27.6 points in 2021-2023. |

## Competition deliverable — results

On the 2022-2026 pack the simple formula demand chain predicts New Arrivals at
28.4% weighted error on the last-year holdout (36.0% seasonal benchmark), and
the hybrid LightGBM layer cuts that to **21.3%** — **20.6%** on the unseen
2025-08 to 2026-02 test window (overall bias −0.1% vs +1.1%). Guest-nights
reach 20.8% weighted error with a learned ALOS (the value the simulator's hybrid
path ships — see [`METHODS.md` §Reproducibility](docs/METHODS.md#reproducibility)).
See [`docs/METHODS.md` §3–4](docs/METHODS.md#phase-3--parametric-layer).

### Read this before quoting any market-level number

The `market` key in every model here joins flight **departure country** to hotel
guest **nationality**. That equality is false, and it is falsified arithmetically:
capture (`New Arrivals / P2P`) **exceeds 1.0 in five markets** — CHINA 5.854,
CANADA 2.288, USA 1.990, RUSSIA 1.527, FRANCE 1.220 — meaning ~903,000 guests
could not have arrived the way the pack files them. Connecting passengers are
filed under the hub (India → Doha → AUH books as QATAR supply, INDIA demand).

- **Aggregate results are valid.** The error is zero-sum across hub and spoke
  markets and cancels in the national total. This is *why* the 20.6% vs 32.8%
  gap is real evidence, and the only reason.
- **Market-level attribution is not valid.** Per-market output, the simulator's
  market breakdown, and the `seats_mult` lever are all filed to the wrong
  market. Success criteria 01/05/06 depend on these.
- **`capture` is a calibration coefficient, not a conversion probability.**

Not fixable from this pack: `Residence (groups)` is constant and no flight-side
residence field exists. Guarded by
`pipeline/tests/test_market_attribution.py`; full analysis in
[`METHODS.md` §4.5](docs/METHODS.md#45-the-market-join-is-structurally-wrong-and-what-survives).

An official anchor independently confirms the direction at region level. Against
SCAD's own embarkation-region table (Key Transport Statistics 2025, t.6) the
pack's GCC share runs **+5.4 to +7.3 pp above SCAD's every year** (TV distance
~0.15–0.20), the pack files no departures for Australasia, South America or
`Other`, and *no coarser region grouping brings capture near 1.0*. Guarded by
`pipeline/tests/test_market_origin_validation.py`; the parser and auditable
country→region mapping live in `pipeline/aviation/scad_region_arrivals.py` — see
[`METHODS.md` §4.5](docs/METHODS.md#45-the-market-join-is-structurally-wrong-and-what-survives).

One implication was checked and is worth stating: the chain's 5.0 capture cap
**helps** the baseline (32.77% capped vs 36.10% uncapped), because China's
inflated coefficient is itself partly this misattribution. So the headline gap is
measured against the strongest version of the chain — see
[§4.6](docs/METHODS.md#46-the-capture-cap-helps-the-baseline).

The brief also names event calendars and public holidays as a lever, and says
supporting data would be supplied. **The pack contained none**, so
`pipeline/calendar_events.py` reconstructs the UAE-wide half from an inline
tabular Hijri conversion pinned to hand-checked windows. Ablated, it is worth
**+0.41 pp** on the holdout and **+0.25 pp** on the test window — real, consistent
in sign, small. Source-market public holidays are still not covered; see
[`docs/METHODS.md` §4.3](docs/METHODS.md#43-the-event-calendar-and-what-it-is-worth).

## Mapping to the success criteria

| # | Criterion | Where |
| --- | --- | --- |
| 01 | Working simulator, not a report | `pipeline/simulator.py` (`Scenario`/`Simulator`); `app/simulator.py` (Streamlit UI) |
| 02 | Transparent, adjustable chain | [`METHODS.md` Phase 3](docs/METHODS.md#phase-3--parametric-layer) — every ratio a lever |
| 03 | Validation with honest error metric | In-sample, last-year holdout, and final test window; weighted error reported |
| 04 | Sensitivity analysis | A-priori Sobol + empirical lever spreads ([`METHODS.md` §3.4](docs/METHODS.md#34-sensitivity)) |
| 05 | Granularity by market and season | 33-market panel, monthly season index, market × season heatmap, **attribution range** — market levels are bounded, not measured |
| 06 | Decision relevance | capture spread is the allocation lever; impact-window selector, guests per 1,000 seats, dynamic decision summary — **market levers misfiled, see the warning above** |
| 07 | Reproducibility | automated assertions; [`METHODS.md#reproducibility`](docs/METHODS.md#reproducibility) |

## Open work

- **Market attribution is the top blocker.** It is pinned and guarded
  (`test_market_attribution.py`), and the new region-level diagnostic
  (`test_market_origin_validation.py`) confirms it against SCAD's official
  embarkation table — but it is **not resolved**. Needs passenger
  residence/nationality on the flight side, or all output must be re-scoped to
  aggregate — which costs success criteria 01, 05 and 06. See
  [`METHODS.md` §4.5](docs/METHODS.md#45-the-market-join-is-structurally-wrong-and-what-survives).
- **Simulator is built.** The chain and the learned layer are packaged behind
  one API in `pipeline/hybrid.py` + `pipeline/simulator.py`
  (`Scenario`/`Simulator`, six global levers, seasonal shape/mix, route edits,
  market suspension, an attribution guard), and the notebook delegates to the
  same code. `app/simulator.py` is the interactive deliverable; its market
  breakdown is explicitly labelled `supply-side-only` and paired with an
  attribution range.
- **Serve the ML layer.** Done: `Simulator().fit()` trains the hybrid end to end
  and `market()`/`monthly()` score scenarios with the learned capture and ALOS.
- **52 air-only origins** have no measured hotel capture rate. OpenSky data in
  `data/raw/opensky/` is intended to close this and is not yet consumed; it is
  pulled by `tools/fetch_opensky.py`, which spends credits and so has not been
  re-run. Note this extends *supply* coverage and does nothing for attribution.
- **VFR share is unmeasured and no public source exists.** Any
  passenger-to-visitor conversion is therefore an explicit assumption, not an
  estimate. Do not let a simulator imply otherwise.

## Documentation

| File | Contents |
|---|---|
| `README.md` | this file — submission overview, setup, how to run |
| [`docs/METHODS.md`](docs/METHODS.md) | **start here** — methods, results, decision log, reproducibility |
| [`docs/SUBMISSION_PACK.md`](docs/SUBMISSION_PACK.md) | deck outline, demo script, system diagram, planner playbook |
| [`docs/competition-brief.md`](docs/competition-brief.md) | challenge framing (provided by the organisers) |

The seven generated diagnostics reports (`market_link_test.md`,
`aggregate_link_test.md`, `retarget_test.md`, `p2p_vs_pax_test.md`,
`calendar_events_test.md`, `market_attribution_test.md`,
`market_origin_validation_test.md`) live in `outputs/` and are **gitignored**.
Regenerate them with the diagnostics above; a clean clone will not have them.