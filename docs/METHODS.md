# Methods and process — ATRC Abu Dhabi flight-to-hotel challenge

**What this document is.** A single, end-to-end account of everything done on the
challenge so far: the question, the data, the engineering, every test, what was
found, what was decided, and what remains. It is written so that a judge can see
the reasoning and a developer can reproduce every number.

**How to read it.** The **submission deliverable is Phase 2–4** (the competition
pack, the parametric chain and the hybrid ML layer); Phase 1 is background
evidence for why the project switched datasets. The repository index is
[`../README.md`](../README.md).

| If you want | Go to |
| --- | --- |
| The one-paragraph answer | [Executive summary](#executive-summary) |
| The repository index | [`../README.md`](../README.md) |
| The competition deliverable chain | [Phase 3 — parametric layer](#phase-3--parametric-layer) |
| Why the original plan failed | [Phase 1 — public-data investigation](#phase-1--public-data-investigation-background) |
| How the competition pack changed the picture | [Phase 2 — competition data pack](#phase-2--competition-data-pack) |
| Each statistical test in detail | [Methods appendix](#methods-appendix) |
| Every design decision | [Decision log](#decision-log) |
| Exact commands and environment | [Reproducibility](#reproducibility) |
| What still has to be done | [Open work](#open-work-and-next-steps) |

---

## Executive summary

The challenge asks for an interactive simulator that predicts how changes in
flights (new routes, frequency, seats, load factor, market mix) change hotel
guest numbers in Abu Dhabi, broken down by source market and season, with the
conversion chain `seats → passengers → visitors → hotel guest-nights` made
explicit and every assumption adjustable
([brief](competition-brief.md)).

Work proceeded in three phases.

**Phase 1 (public data, 2026-09).** Using only public sources — 99 DCT monthly
hotel reports, SCAD transport statistics, and 2025 ADS-B/OpenSky flight data —
three independent tests showed a *negative result*: aggregate air arrivals are
not a usable driver of aggregate hotel demand, and the airborne origin mix cannot
allocate hotel guests by market. Total arrivals at AUH are dominated by
UAE-resident and GCC traffic that does not use hotels; the hotel-relevant
non-resident segment is a minority that moves on its own schedule. The monthly
input needed to fix this — non-resident passenger counts — is not published
after 2018. All of this is documented with evidence in
[Phase 1](#phase-1--public-data-investigation-background).

**Phase 2 (competition data pack, 2026-10).** The organisers supplied licensed
data that closes the key gap: the flight table now contains
`Total PAX`, `Total P2P` (point-to-point, i.e. terminating), `Total Transfer` and
`Total Transit`, and the hotel table contains `New Arrivals` alongside `Guests`.
This *refines rather than overturns* Phase 1: the old negative result came from a
contaminated denominator (all onboard passengers include transfers/transit) and
an origin-versus-nationality mismatch. With the terminating-passenger denominator
(P2P), the conversion chain becomes viable. On matched nationality×month panels,
P2P beats PAX as the inbound driver (R² 0.8774 vs 0.8625, tighter standard
error). See [§2.2](#22-gate-experiment--pax-vs-p2p).

**Phase 3 (parametric layer, 2026-10-03).** A transparent, auditable chain was
built, with supply stages measured off flight rows, demand stages estimated from
the joined panel (§4.5), and every ratio an override lever. It is
Part A of `notebooks/hybrid_layer.py`. Headline results:

| metric | value |
| --- | --- |
| Panel | 1,405 market-months, 33 markets, 43 months (2022-01 → 2025-07) |
| Global capture (`New Arrivals / P2P`) | **0.391** |
| Global ALOS (`Guests / New Arrivals`) | **3.602** |
| In-sample WMAPE, New Arrivals / guest-nights | **27.3% / 24.8%** |
| Leave-tail holdout WMAPE, New Arrivals | **28.4%** (seasonal naive **36.0%**) |
| Leave-tail holdout WMAPE, guest-nights | **25.6%** |
| Competition test window (2025-08 → 2026-02), New Arrivals | **32.8%**, aggregate within **+1.1%** |

The chain beats a seasonal-naive baseline on held-out data. At market-month
granularity error is still roughly a third of volume; at aggregate it is much
closer. Both numbers ship, because the brief asks for granularity *and* honesty.

Three things are deliberately *not* overclaimed: purpose-of-travel/VFR is
unmeasured and absorbed inside the capture rate; capture and ALOS are behavioural,
not mechanical; and guest-nights cannot be scored on the competition test window
because that frame has no `Guests` column. Details in
[Phase 3](#phase-3--parametric-layer).

**Phase 4 (ML layer, 2026-10-03).** A hybrid model keeps the exact supply stages
and replaces the fixed capture ratio with a **LightGBM** Poisson model using
`log(P2P)` as an exposure offset, i.e. a learned, market-aware capture
rate. It beats the parametric chain on every held-out metric: T4 New Arrivals
**21.3%** vs 28.4%, competition window **20.6%** vs 32.8% (aggregate bias −0.1%
vs +1.1%), and holdout guest-nights **20.8%** with a learned ALOS. `market`
dominates feature importance — the model is the regularised answer to the 36×
capture spread. See [Phase 4](#phase-4--machine-learning-layer).

**Phase 4b (event calendar, 2026-10-05).** The brief names event calendars and
public holidays as a lever and says supporting data would be supplied; **the pack
contained none**, so the requirement was being dropped silently.
`pipeline/calendar_events.py` reconstructs the UAE-wide half of it from an inline
tabular Hijri conversion, pinned to hand-checked windows. Four monthly
event-day counts are worth **+0.41 pp** on the leave-tail holdout and **+0.25 pp**
on the competition window — real, consistent in sign, and small. Source-market
public holidays remain uncovered. See [§4.3](#43-the-event-calendar-and-what-it-is-worth).

---

## The problem and the success criteria

From the [brief](competition-brief.md), evaluation is 40% technical accuracy and
modelling rigour, 20% creativity/practicality, 20% clarity, plus a 20% realism
weight. The seven success criteria map to the work as follows:

| # | Criterion | Where it is addressed |
| --- | --- | --- |
| 01 | A working simulator, not a static report | `pipeline/simulator.py` + `app/simulator.py` (Streamlit UI) with scenario levers; notebook Part A remains the audit trail |
| 02 | Transparent conversion chain, assumptions adjustable | [Phase 3](#phase-3--parametric-layer) |
| 03 | Validation against history with an honest error metric | T3/T4/T5, WMAPE on holdout and test window |
| 04 | Sensitivity analysis | T6 a-priori Sobol + empirically measured lever spreads (`pipeline/sensitivity.py`) |
| 05 | Granularity by market and season | 33-market panel, global monthly season index — **market attribution unvalidated, see §4.5** |
| 06 | Decision relevance | [Scope limits](#36-scope-limits-and-honest-positioning); capture spread is the allocation lever — **levers misfiled at market level, see §4.5** |
| 07 | Reproducibility | [Reproducibility](#reproducibility), pinned assertions T7 |

---

## Timeline

| Dates | Phase | Output |
| --- | --- | --- |
| 2026-09-28 | Challenge framing | [`competition-brief.md`](competition-brief.md) |
| 2026-09-30 | Public-data investigation | negative result — see [Phase 1](#phase-1--public-data-investigation-background) |
| 2026-10-01 | DCT parsing, data dictionary, diagnostics | `pipeline/dct/`, generated reports in `outputs/` |
| 2026-10-02 | Competition pack received, loader + audit, gate experiment | [Phase 2](#phase-2--competition-data-pack), `outputs/p2p_vs_pax_test.md` |
| 2026-10-03 | Parametric layer + tests | Part A of `notebooks/hybrid_layer.py` |
| 2026-10-03 | Hybrid ML layer (LightGBM) + tests | Part B of `notebooks/hybrid_layer.py` |
| 2026-10-05 | Phase 3 and 4 merged into one notebook; OpenSky puller moved to `tools/fetch_opensky.py` | `notebooks/hybrid_layer.py` |
| 2026-10-06 | Origin mix validated against SCAD's official embarkation table; region-level gap pinned | `pipeline/aviation/scad_region_arrivals.py`, `test_market_origin_validation.py` |
| 2026-10-09 | Chain + hybrid extracted to a canonical package; scenario API for the Streamlit app; notebook delegates | `pipeline/hybrid.py`, `pipeline/simulator.py`, `test_simulator_package.py` |
| 2026-10-10 | Simulator package hardened: lazy per-window fit, pickle persistence, immutable validated market multipliers, lever guards | `pipeline/simulator.py`, `pipeline/hybrid.py`, `pipeline/tests/test_simulator_package.py` |
| 2026-10-11 | Learned ALOS shipped in the simulator's hybrid path; T4 guest-nights replayed live | `pipeline/simulator.py`, `pipeline/hybrid.py`, `pipeline/tests/test_simulator_package.py` |
| 2026-10-11 | Empirical lever-spread sensitivity added as a second, labelled view; a-priori ranges kept as the simulator's decision space | `pipeline/sensitivity.py`, `notebooks/sensitivity_empirical.py`, `pipeline/tests/test_sensitivity.py` |

The repository has a single commit (`Initial commit: ATRC Abu Dhabi
flight-to-hotel analysis`); the working tree carries the Phase 2–4 work. Raw data
is gitignored; all analysis-ready tables and text extracts are tracked.

---

## Repository layout and data tiers

```
data/raw/          downloaded sources                (gitignored)
data/interim/      extracted PDF text + fingerprint   (tracked)
data/processed/    14 analysis-ready tables           (tracked)
pipeline/paths.py  every filesystem path in the project
pipeline/dct/      DCT PDF parsing, by report era
pipeline/aviation/ passenger, SCAD and ADS-B builders
pipeline/competition/  competition-pack loader + audit
pipeline/hotel_supply.py
pipeline/hybrid.py            canonical chain + LightGBM layer (shared by notebook and simulator)
pipeline/simulator.py         Scenario/Simulator API for the Streamlit app
pipeline/sensitivity.py       empirically measured lever spreads + Sobol (sensitivity view 2)
pipeline/tests/    ten standalone diagnostics
notebooks/         four marimo notebooks
docs/              METHODS.md + the organisers' brief
tools/             one-off utilities + the OpenSky puller (fetch_opensky.py)
app/               Streamlit simulator (stub)
outputs/           generated reports + scratch (gitignored)
```

Tiers: **raw** is gitignored because it is large and re-downloadable; **interim**
and **processed** are tracked so the analysis reproduces from a clean clone
without re-fetching. `data/raw/opensky/` is a quota-limited pull that may not be
repeatable. No module hardcodes an absolute path — everything resolves through
`pipeline/paths.py`, and importing it also puts the pipeline directories on
`sys.path`.

---

## Phase 1 — public-data investigation (background)

The original plan built the chain from public data alone. It failed for a
structural reason, and that negative result is what motivated switching to the
competition pack. It is recorded here briefly; the submission itself is Phase 2–4.

Sources: DCT monthly hotel reports (99 PDFs, 2015–2024, parsed by `pipeline/dct/`),
SCAD transport statistics, one-day-per-month ADS-B snapshots for 2025, and an
OpenSky pull. The decisive availability finding: SCAD publishes **no `Transport
Statistics` edition after 2018** and no resident/non-resident split at any
frequency, so monthly non-resident passenger counts simply do not exist for the
period of interest. The SCAD library was enumerated (1,850 documents) rather than
searched, and two API routes were tried and documented as dead ends.

Three tests, all negative:

- **Origin mix** does not predict hotel guest mix — apparent correlation
  (Pearson 0.73–0.96) is carried by India, and drops to **−0.041** without it;
  per-market conversion spans **36×**.
- **Aggregate monthly** elasticity of guests to arrivals is **0.170 (se 0.176)**,
  indistinguishable from zero.
- **Forecast:** a seasonal+trend baseline that never sees arrivals scores **5.2%**
  WMAPE, adding arrivals worsens it to **5.6%**, and arrivals alone score
  **12.6%**. A model without the flight data beats one with it.

The mechanism: total AUH arrivals blends UAE residents, GCC visitors and
non-resident travellers, and the hotel-relevant segment cannot be isolated. An
ex-UAE retargeting proxy does not fix it (it overstates true international share
by 12.8–27.6 points).

**Conclusion.** Do not build the simulator on total public arrivals. The
competition pack's `P2P` column supplies the terminating-passenger denominator
the public data lacked.

---

## Phase 2 — competition data pack

On 2026-10-02 the organisers' pack was installed under
`data/raw/competition/`, licensed for competition use only. Checksums were
verified lossless on move.

| File | Role | Bytes | md5 |
| --- | --- | ---: | --- |
| `flight_data.xlsx` | scheduled supply + realised passengers, route × date | 15,445,753 | `0d62b67543951c856a6f5b3233304891` |
| `data international_train.xlsx` | hotel stats by nationality, 2022-01 → 2025-07 (`Guests`, `New Arrivals`) | 1,688,395 | `2f130d1631b6c630076baf7e1d963806` |
| `data international_test.xlsx` | 2025-08 → 2026-02, `New Arrivals` only (no `Guests`) | 256,361 | `2820cdcd9604842e446bb0b056596948` |
| `data domestic_train.xlsx` | domestic counterpart (train) | 46,562 | `3371878432cbe0a6f0fc9efa373a9d7a` |
| `data domestic_test.xlsx` | domestic counterpart (test) | 16,190 | `3f92aba6c7897622f16df8e82a5eaf8d` |
| `Data_Dictionary.pdf` | field definitions | 537,842 | `51602e3fb9066e204a86e149a918e7f5` |

### 2.1 Loader and integrity

`pipeline/competition/load.py` is the single repair-aware loader
(`read_flights`, `flights_monthly`, `flights_monthly_agg`,
`read_international_train/test`, `read_domestic_train/test`). Repairs and checks
established by `pipeline/competition/audit.py`:

- **Identities hold on all 117,608 flight rows:**
  `Total PAX = Total P2P + Total Transfer + Total Transit` and
  `Load Factor = Total PAX / Total Seats`.
- `Average Weekly Frequency` is a monthly-decimal, repaired to weekly via **×31**.
- **1,213 blanks**, all confined to 2022 cabin/frequency columns; `PAX` and
  `Seats` are intact.
- **Grain break at 2023-01** (2022 is monthly, 2023+ is daily); all consumers
  normalise to `ym` (month period) through the loader.
- 12 hotel-only markets have **no flight supply** (Australia, Brazil, Czechia,
  Denmark, Finland, Mexico, Morocco, Norway, Pakistan, Romania, South Africa,
  Sweden) and are carried as a residual, never assigned synthetic capture.

### 2.2 Gate experiment — PAX vs P2P

`pipeline/tests/test_p2p_vs_pax.py` → `outputs/p2p_vs_pax_test.md`.
Log-log regression of hotel guests on the inbound passenger measure, with month
and nationality fixed effects, on an identical 1,405-row matched panel:

| model | coef (elasticity) | std err | t | R² | adj. R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAX | 0.4322 | 0.0282 | 15.324 | 0.8625 | 0.8547 |
| **P2P** | **0.4403** | **0.0213** | **20.632** | **0.8774** | **0.8705** |

P2P wins on explanatory power (+1.5pp R²) and precision. This is the evidence that
Phase 1's failure was a *contaminated denominator*, not a broken premise: strip
transfer/transit and matching nationality, and the inbound→hotel link is strong.

### 2.3 The measured panel and metrics

The panel is the inner join of monthly flight supply and monthly hotel stats on
`departure country == nationality × month`. **This equality is asserted, not
given, and it is false — see §4.5.** It is load-bearing for every number in this
document.

- **1,405 market-months, 33 markets, 43 months (2022-01 → 2025-07).**
- Coverage of pack P2P against SCAD AUH arrivals: **~77%** (2022 72.9%, 2023
  78.1%, 2024 77.3%).
- **Capture (`New Arrivals / P2P`)**: mean 0.83, median 0.53, highly
  heterogeneous — market means span ~35× (0.083–2.958 after capping extremes at
  5), matching Phase 1's 36× spread. Five markets exceed 1.0, which no
  conversion probability can.
- **ALOS (`Guests / New Arrivals`)**: ~3.60, anchoring the stock-flow step.

Why this matters: the pack supplies both `New Arrivals` and a terminating
denominator, so capture is now *estimated* rather than assumed from Phase 1 — and
the join it is computed from is structurally wrong for market-level attribution
(§4.5).

---

## Phase 3 — parametric layer

**Artifact:** Part A of `notebooks/hybrid_layer.py` (marimo) — self-contained
engine + tests + charts. The notebook is shared with Phase 4; Part A owns
`fit_params`, `simulate` and `wmape`, and Part B consumes them.

### 3.1 The chain

```
scheduled seats → PAX → P2P → New Arrivals → guest-nights
```

| stage | ratio | source |
| --- | --- | --- |
| seats | `Total Seats` | `flight_data.xlsx` |
| PAX | load factor `Total PAX / Total Seats` | flight rows |
| P2P | `Total P2P / Total PAX` (strips transfer + transit) | flight rows |
| New Arrivals | `capture[m] = Σ New Arrivals / Σ P2P`, per market, capped at 5.0 | joined panel |
| guest-nights | `ALOS[m] = Σ Guests / Σ New Arrivals` × global monthly season index | joined panel |

Every stage is a volume-weighted ratio exposed as an override lever; the supply
stages are measured off flight rows and the demand stages are estimated from the
joined panel (§4.5), so the whole thing stays auditable and explainable to a
non-technical planner.

### 3.2 Measured vs estimated vs assumed

- **Measured:** seats, load factor, and P2P share — all read directly off flight
  rows. This is the change from Phase 1: the pack supplies both `New Arrivals`
  and `P2P` bookkeeping on one row.
- **Estimated:** `capture[m]` and `ALOS[m]`, volume-weighted so small markets
  cannot outvote large ones. Capture is only observable *after* the panel join,
  and that join is structurally wrong for market-level attribution (§4.5).
  Global capture **0.391**; global ALOS **3.602**.
- **Assumed:** purpose-of-travel/VFR, which the brief lists as a lever but no
  source measures. It is absorbed inside the estimated capture ratio, and the
  simulator must not present the visitor split as measured.

### 3.3 Why empirical-Bayes shrinkage was rejected

The natural guard for thin markets is a Beta-binomial shrinker toward the global
mean. It was implemented, tested, and **rejected**: between-market capture
dispersion (0.085 → 5.854, median 0.554) exceeds the maximum variance a Beta
prior can represent, so the method-of-moments concentration goes negative. That
is real behavioural heterogeneity — Phase 1's 36× spread — not sampling noise. The
estimated rate is kept with the 5.0 cap as the only guard.

### 3.4 Sensitivity

Sensitivity is reported in **two labelled views** that answer different
questions. Both use the same chain algebra and the same SALib Sobol
decomposition; only the ranges differ.

**View 1 — decision space (a-priori).** Test T6 shocks the six levers over the
ranges the simulator exposes to a planner (capture 0.7–1.3; seats/frequency
0.8–1.2; load factor, P2P share, ALOS, season 0.9–1.1). These are deliberately
wider than observed — they are "how much room a planner gets". Every lever is
multiplicative, so all six have unit elasticity by construction, and the output
variance is dominated by the wider-bound levers: capture (ST 0.51) and
seats/frequency (0.23), the rest at 0.06–0.07. This view says *which lever is
**allowed** to move the number most*; it is what the UI's `LEVER_BOUNDS` encode.

**View 2 — empirical spread (what actually moved).** Added 2026-10-11. For each
lever, `pipeline/sensitivity.py` derives the month-level multiplier observed in
the panel, `u = z / baseline`:

| lever | month measure `z` | baseline |
| --- | --- | --- |
| seats / frequency | `Total_Seats` month-over-month | 1.0 |
| load factor | `Total_PAX / Total_Seats` | market seat-weighted mean |
| P2P share | `Total_P2P / Total_PAX` | market PAX-weighted mean |
| capture rate | `New Arrivals / Total_P2P` | `capture[m]` (capped, shipped) |
| ALOS | `Guests / New Arrivals` | `ALOS[m]` (shipped) |
| season | `season[month]` (12 values) | 1.0 |

The **volume-weighted 16th–84th percentile** becomes that lever's Sobol range
(uniform and winsorized kernels are reported for robustness). The measured middle
bands:

| lever | q16 | median | q84 |
| --- | ---: | ---: | ---: |
| capture rate | 0.665 | 0.970 | 1.285 |
| P2P share | 0.841 | 1.002 | 1.131 |
| load factor | 0.889 | 1.019 | 1.099 |
| ALOS | 0.883 | 0.980 | 1.090 |
| seats / frequency | 0.943 | 1.023 | 1.127 |
| season | 0.926 | 1.016 | 1.054 |

Sobol on these bounds gives capture **ST 0.62**, P2P share **0.14**, load factor
**0.073**, ALOS **0.070**, seats **0.050**, season **0.027**. The two views agree
that **capture dominates**; they disagree about second place. A-priori grants
seats a wide 0.8–1.2 and ranks it second, whereas measured seat supply only
wobbles about ±9% month-over-month, so **P2P share** (the widest measured supply
lever) takes its place. That disagreement is the point of measuring.
`notebooks/sensitivity_empirical.py` (E1–E4) and
`pipeline/tests/test_sensitivity.py` reproduce every number.

**What view 2 does not do.** It does not replace the a-priori ranges: the UI must
keep rooms a planner can press, and new-route / step-change scenarios
legitimately exceed the historical wobble. The empirical ranges are
informational — they sharpen the *reading* of criterion 04 and never overwrite
`LEVER_BOUNDS`. Both views stay labelled; the empirical one also inherits the
§4.5 caveat, since it measures movements of already-contaminated per-market
coefficients. (SALib note: the package import name is `SALib`, `saltelli` is
replaced by `Sobol.sample`, and the empirical wrapper seeds sampling and
analysis so its indices are reproducible.)

### 3.5 Results and the test suite

| test | target | result |
| --- | --- | --- |
| T3 in-sample, 1,405 market-months | New Arrivals / guest-nights | 27.3% / 24.8% WMAPE |
| T4 leave-tail holdout, 198 market-months | New Arrivals | **28.4%** (seasonal naive 36.0%) |
| T4 | guest-nights | 25.6% |
| T5 competition test window, 231 market-months | New Arrivals | 32.8%; aggregate **+1.1%** |
| T1/T2/T7 | integrity, chain arithmetic, pinned consistency | all pass |

T1 checks input identities, T2 checks the chain reproduces the measured supply
stages under unit levers, T3–T5 are the fit/holdout/test evaluations above, T6 is
sensitivity, and T7 pins the numbers quoted in this write-up so the notebook
fails loudly if any moves. The holdout beats seasonal-naive; the aggregate is
close; market-month error stays around a third of volume and is reported as such.

### 3.6 Scope limits and honest positioning

- **Out-of-sample guest-nights are unscoreable** (the competition test frame has
  no `Guests` column), so they are validated in-sample and on the leave-tail
  holdout only.
- **Season is one global monthly profile**; market-specific seasonality is pooled
  because most markets have only 43 months.
- **Capture and ALOS are behavioural**, absorbing VFR, booking behaviour and
  non-air hotel demand — levers to move with their spread in view, not constants.
- **12 hotel-only markets** carry no flight supply and are a separately reported
  residual.

---

## Phase 4 — machine-learning layer

**Artifact:** Part B of `notebooks/hybrid_layer.py` (marimo). Depends on
`lightgbm`.

Phase 3 and Phase 4 were developed as two notebooks and merged into one on
2026-10-05. The reason was duplication rather than tidiness: each carried its own
copy of `fit_params`, `simulate` and `wmape`, and the ML notebook recomputed the
entire parametric baseline (ML-1) purely to prove the two agreed. In one notebook
the baseline *is* the Part A result, so the head-to-head cannot drift. The
no-lever `simulate` stub that the ML copy carried was arithmetically identical to
the full engine at unit multipliers and was deleted.

### 4.1 Hybrid design

The parametric chain has five stages; three are arithmetic and reproduce the
measured data exactly (T2), so there is nothing to learn there. The ML layer
replaces **only** the capture step:

```
seats → PAX → P2P             [parametric chain, exact]
P2P  → New Arrivals           [LightGBM Poisson, offset log(P2P)]
New Arrivals → guest-nights    [learned ALOS by default; chain ALOS[m] is the parametric view]
```

The Poisson offset makes the model learn an intensity per terminating passenger —
a learned, market- and season-aware capture rate. That rate is a calibration
coefficient between two differently composed populations, not a conversion
probability (§4.5). Features are
**leak-free** (only quantities known at prediction time; no New-Arrivals-derived
feature is ever constructed):

| feature | type |
| --- | --- |
| `market` | native categorical (33 levels) |
| `month` | native categorical (12 levels) |
| `t` | months since 2022-01 (trend) |
| `log_seats` | `log1p(Total_Seats)` |
| `load_factor` | `Total_PAX / Total_Seats` |
| `p2p_share` | `Total_P2P / Total_PAX` |
| `transfer_share` | `(Transfer + Transit) / Total_PAX` |
| `routes`, `airlines` | route and carrier counts |
| `eid_days`, `ramadan_days`, `natday_days`, `nye_days` | UAE event-day counts per month (§4.3) |

Exposure offset: `log(Total_P2P)`. The competition test window supplies all of
these, so a forecast needs no target history.

### 4.2 Evaluation

Identical splits and metric as the chain. Configuration was chosen **a priori**,
not tuned on the reported holdouts: `num_leaves=15`, `min_child_samples=20`,
`learning_rate=0.03`, `feature_fraction=0.8`, `bagging_fraction=0.8`,
`lambda_l2=1.0`, single thread, deterministic. Iteration count is selected by
early stopping (patience 100, cap 2000) on an internal 6-month tail of each fit
window, then a **5-seed ensemble** (`0,1,2,42,2024`) is refit on the full fit
window and averaged on the rate scale -- each seed predicts `exp(raw score +
log(Total_P2P))` and the resulting intensities are meaned. (Averaging raw
scores instead would be a geometric-mean estimator.) Selected
iterations: **1466** (T4) and **1978** (T5), with the event calendar of §4.3;
**1334** and **1934** without it.

| metric | parametric chain | hybrid LightGBM |
| --- | ---: | ---: |
| T4 New Arrivals WMAPE (198 market-months) | 28.4% | **21.3%** (seeds 20.8–21.7) |
| T5 New Arrivals WMAPE (231 market-months) | 32.8% | **20.6%** (seeds 20.3–20.9) |
| T5 aggregate bias | +1.1% | **−0.1%** |
| T4 guest-nights WMAPE, chain ALOS | 25.6% | 23.3% |
| T4 guest-nights WMAPE, learned ALOS | — | **20.8%** (shipped in the simulator, `reproduce_holdout()`) |
| seasonal-naive New Arrivals baseline, T4 | 36.0% | — |

### 4.3 The event calendar, and what it is worth

The brief lists *"event calendars and public holidays in source markets"* among
the factors to exploit, and says supporting event data would be provided. **It
was not.** The eight-page data dictionary has no mention of events or holidays,
and the five files in the pack are all flight and hotel tables. The requirement
was therefore being dropped silently, which is the one way to fail it.

`pipeline/calendar_events.py` reconstructs the destination-side half of it. Four
features per market-month, each a count of event days clipped to that month:

| feature | window | why |
| --- | --- | --- |
| `eid_days` | Eid al-Fitr and Eid al-Adha, ±3 days around each | biggest measured uplift |
| `ramadan_days` | all days of Ramadan | large and *negative* |
| `natday_days` | 2 December ±3 days, plus the 5–8 December 2025 four-day weekend | |
| `nye_days` | 31 December and 1 January, ±3 days, unioned | overlapping windows, not summed |

**Why computed dates and not a holiday table.** UAE holidays follow moon
sighting, so declared dates can differ from any computed calendar by a day or
two. Islamic dates are derived from an inline tabular Hijri conversion — no new
dependency, since `holidays`, `convertdate` and `hijridate` are all absent — and
every date is pinned to a hand-checked window two to five days wide by
`pipeline/tests/test_calendar_events.py`. Because the features count days across
multi-day windows, a one-day shift cannot move them. A hand-typed table would
have needed extending by hand every year and would have failed silently when
stale; `monthly_features` instead raises if asked for a period outside
2022-01…2026-02.

Two aggregation traps are pinned by that test rather than left to inspection:
New Year's two windows overlap on 29–31 December, so days must be unioned
(4 in December, not 7); and Eid al-Adha 1444's window straddles 30 June, giving
5 days in June and 4 in July.

**Measured uplift**, against the same window one year either side so the baseline
carries the month-to-month level and only the local shape is compared:

| event (±3d) | uplift in national new arrivals |
| --- | ---: |
| Eid al-Adha | +18.4% |
| Eid al-Fitr | +11.2% |
| UAE National Day | +3.8% |
| New Year | +1.2% |
| Ramadan, mid-month | −3.3% (−12.4% at ±7d) |

Ramadan is negative and that is the finding, not an artefact: resident hotel
demand falls while outbound travel rises, and the target is inbound arrivals.

**The ablation (ML-7).** The feature set was fixed a priori from the brief's
wording and the uplifts above, evaluated once, and reported once. Both arms use
the identical protocol — seed-0 early stopping for the round count, then a
five-seed ensemble refit — so the columns differ only in their features.

| window | without events | with events | difference |
| --- | ---: | ---: | ---: |
| T4 leave-tail holdout | 21.69% | 21.28% | **+0.41 pp** |
| T5 competition window | 20.80% | 20.55% | **+0.25 pp** |

Real, consistent in sign across two independent windows, and small. That is the
honest summary, and it is smaller than the raw uplifts suggest for two reasons.
The four features are largely collinear with the `month` categorical — Eid al-Adha
fell in June or July every year of the fit window, which `month` already encodes —
so the genuine addition is only *Islamic drift*: Ramadan moved April → March →
February, so a month that contained it and one that did not share the same `month`
value. And with 1,405 rows and roughly five event months per feature, the panel
can support little more.

The four ML-6 pins were **not** re-baselined. Adding the calendar moved T4 by
−0.4 pp and T5 by −0.2 pp, both inside the existing tolerances, so the pins still
hold and were left alone. That is the outcome to want: had they needed loosening,
the first question would have been whether the features were chosen to fit them.

**Not covered.** Public holidays in the *source* markets — the other half of the
brief's wording — are absent. Those need a maintained holiday table per market for
at least India, Egypt, Saudi Arabia, Qatar, Turkey and the UK, which is a sourcing
exercise this does not fake.

### 4.4 Interpretation and scope limits

`market` dominates gain importance (`market` ≫ `p2p_share` > `log_seats` >
`routes` > `transfer_share` > `t` > `load_factor` > `month` > `airlines`).
Read that carefully, because `market` is the §4.5 join key: part of its gain is
**shrinking noise in a noisy raw ratio** and part is **shrinking a contaminated
signal**, so the importance partly measures the size of the misattribution term
rather than only the model's market awareness. ML-5 plots learned capture against
the chain's capped ratio; thin markets are pulled toward the centre, and the five
points beyond 1.0 are impossible for a probability.

Limits, mirrored in the notebook: the test window has no `Guests`, so guest-nights
are validated in-sample and on the leave-tail holdout only; the panel is small
(1,405 rows) so the model is shallow by design; capture stays behavioural; and the
12 hotel-only markets remain unmodelled.

### 4.5 The `market` join is structurally wrong, and what survives

`market` carries two incompatible meanings. On the supply side it is
`Departure Country Name` — the last departure point before AUH. On the demand side
it is `Nationality` — the guest's citizenship. The panel join asserts they are the
same object.

**Mechanism.** `P2P = Total PAX − Transfer − Transit` is computed *per flight
row*. An Indian national flying India → Doha → Abu Dhabi is Transfer/Transit on
the first leg and P2P on the second, where `Departure Country Name = QATAR`. So
supply lands on Qatar and demand lands on India. Stripping transfers does not
prevent this — it *relocates* connecting traffic onto the hub as if it originated
there.

**Falsification.** If the join were valid, capture could not exceed 1.0: you
cannot book more guests than terminating passengers.

| market | New Arrivals | own-departure P2P | capture |
|---|---:|---:|---:|
| CHINA | 563,639 | 96,283 | **5.854** |
| CANADA | 95,414 | 41,707 | **2.288** |
| UNITED STATES OF AMERICA | 326,985 | 164,351 | **1.990** |
| RUSSIAN FEDERATION | 511,393 | 334,858 | **1.527** |
| FRANCE | 238,730 | 195,689 | **1.220** |

**~903,000 guests** in these five markets could not have arrived the way the pack
files them. The cause is visible as a supply/demand mix divergence: hubs carry
supply in excess of demand (QATAR 4.7% vs 1.7%, AZERBAIJAN 2.5% vs 0.5%, EGYPT
8.5% vs 2.7%) while spokes carry the mirror image (UNITED KINGDOM 3.5% vs 8.5%,
GERMANY 1.7% vs 4.0%). It is a zero-sum transfer, not noise.

`pipeline/tests/test_market_attribution.py` pins this and guards it. Any *new*
market crossing 1.0 is a hard failure.

**What survives, and what does not.**

- **Aggregate is defensible.** `Σ P2P × capture = Σ New Arrivals` holds exactly,
  and the error is stationary over the fit window, so it cancels in the national
  total. This is *why* T5's −0.1% aggregate bias and the T5 WMAPE gap are real
  evidence, and it is the only reason they are.
- **Market-level attribution is not defensible.** Per-market outputs, the
  simulator's market breakdown, and the `seats_mult` lever are all filed to the
  wrong market. Nothing downstream of a market split should be quoted.
- **`capture` is a calibration coefficient, not a conversion probability.** It is
  a shrunk mapping between two differently composed populations.

**Not fixable from this pack.** `Residence (groups)` is constant
(`"International"`, 100% of rows) and no flight-side residence field exists.
Correcting the mapping requires external residence/nationality data. This is a
data limitation, not an implementation gap.

**A prior signal was missed.** The Phase 1 negative result was already in this
document (§2, "the airborne origin mix cannot allocate hotel guests by market",
with UAE-resident and GCC traffic not using hotels flagged as the mechanism), and
`test_market_link.py` had already measured the symptom on independent public data
— airborne seat share vs hotel guest share correlates 0.908 overall but −0.041
without India. Both were filed under "different dataset" rather than connected to
the competition join. The evidence was in the repo and read past.

**The official embarkation mix confirms it at region level.** SCAD publishes
arrivals at Zayed by *region of embarkation* (Key Transport Statistics 2025,
table 6) — the only independent break-down of where incoming passengers boarded.
`scad_region_arrivals.py` parses it and holds an auditable country→region
mapping whose ambiguous cases (Egypt→Africa, Türkiye→Western Europe, Caspian/
Central Asia→Asia) were resolved against SCAD's own bucket levels. Year-matched
to the pack (2022–2025):

- the pack's GCC share runs **+5.4 to +7.3 pp above SCAD's every year**, and the
  gap survives every country→region choice — it is the pack crediting
  Doha-connecting traffic to the GCC that SCAD resolves to true embarkation;
- total-variation distance between the two mixes is ~0.15–0.20, and the pack
  files **no departure at all** for Australasia, South America or the residual
  `Other` bucket (~2.1% of 2025 guests arrive from regions with zero
  own-region departures);
- guest nationality is *not* the same object as embarkation — ~15% of guests are
  Indian nationals against SCAD's ~32% Indian-Subcontinent share, which is the
  UAE-resident/GCC population mechanism (§2) again, not a supply shortfall.

Region re-aggregation does **not** rescue market meaning: capture on the 11 SCAD
regions still spans 0.16–2.32, and hub markets' supply is a single connecting
point (Qatar 100% Doha at capture 0.15). `test_market_origin_validation.py`
runs this as four diagnosis-only legs (supply vs SCAD, nationality vs SCAD,
within-market signal + hub exposure, restructure) and pins every headline.

### 4.6 The capture cap helps the baseline

The chain clips capture at 5.0. CHINA's raw 5.854 exceeds it. The natural guess
is that this clip conceals a defect and flatters the chain's score. **The data
says the opposite, and it is worth recording because the guess was wrong.**

| capture clip | T5 WMAPE | gap vs hybrid |
| --- | ---: | ---: |
| uncapped | 36.10% | +15.54 pp |
| **capped at 5.0** | **32.77%** | **+12.21 pp** |
| hybrid LightGBM | 20.55% | — |

Removing the cap *worsens* the chain by **3.33 pp**. The reason is §4.5: China's
inflated coefficient is partly fitting connecting traffic filed under China, so
extrapolating it forward is harmful, and clipping shrinks that contamination.

Two consequences. The capped chain is the **stronger, fairer** baseline, and the
headline 12.21 pp gap is measured against it — the conservative comparison. And
the claim that part of the hybrid's margin came from the baseline's guard being
wrong does not survive: the guard was working.

---

## Methods appendix

Each test is a standalone script (pytest is not installed) writing a markdown
report. Reproduce with the commands shown.

| Test | Script | Report | Design | Headline |
| --- | --- | --- | --- | --- |
| Market mix | `pipeline/tests/test_market_link.py` | `outputs/market_link_test.md` | Seat-weighted 2025 ADS-B shares vs DCT hotel shares, renormalised to common support (9 markets) | Correlation carried by India; −0.041 without it; 36× conversion spread |
| Aggregate monthly | `pipeline/tests/test_aggregate_link.py` | `outputs/aggregate_link_test.md` | log-log guests on arrivals + month + trend, leave-one-year-out WMAPE | Elasticity 0.170 (se 0.176); no-arrivals 5.2% beats full 5.6% |
| Annual | `pipeline/aviation/build_annual_arrivals.py` | — | Annual arrivals vs guests and ratio drift, 2015–2024 | Arrivals −8.6% while guests +25.1% (2015–2019); ratio 0.355→1.303 |
| Retarget international | `pipeline/tests/test_retarget_international.py` | `outputs/retarget_test.md` | ex-UAE proxy vs true international; bootstrap CIs + permutation p-values | Proxy overstates by 12.8/20.3/27.6pp; no forecastable link |
| PAX vs P2P gate | `pipeline/tests/test_p2p_vs_pax.py` | `outputs/p2p_vs_pax_test.md` | log-log on identical matched panel, month + nationality FE | P2P R² 0.8774 > PAX 0.8625, tighter se |
| Cross-market check | `pipeline/tests/crosscheck_markets.py` | — | Cross-source market reconciliation | — |
| UAE event calendar | `pipeline/tests/test_calendar_events.py` | `outputs/calendar_events_test.md` | JDN epoch + 3 Hijri anchors, 15 window pins, Ramadan→Eid = 30d, known month counts | All pass; overlapping New Year windows must be unioned, not summed |
| Market attribution | `pipeline/tests/test_market_attribution.py` | `outputs/market_attribution_test.md` | Paired-panel capture per market, supply vs demand mix divergence, lower bound on misattributed guests | Capture > 1.0 in 5 markets (CHINA 5.854); ~903k guests misfiled; identity gap exactly 0 |
| Origin-mix validation | `pipeline/tests/test_market_origin_validation.py` | `outputs/market_origin_validation_test.md` | Pack region mix vs SCAD embarkation mix (KTS 2025 t.6): supply, nationality, within-market signal, hub exposure | GCC gap +5.4–7.3 pp every year; TV 0.15–0.20; hub capture ~0.1; no coarser rescue |
| ML hybrid | `notebooks/hybrid_layer.py` Part B | — | LightGBM Poisson with offset, same T4/T5 splits | T4 21.3%, T5 20.6%, guest-nights 20.8%; events worth +0.41/+0.25pp |
| Simulator packaging | `pipeline/tests/test_simulator_package.py` | — | Re-fits the packaged model end to end; pins T5 WMAPE 20.8 ± 2.0 / bias −0.1%, lazy per-window fit, save/load round-trip, lever guards | ~75s; all pins pass |
| Empirical sensitivity | `notebooks/sensitivity_empirical.py` E1–E4 | — | Per-lever monthly multiplier `u = z/baseline`; volume-weighted 16–84th pct bounds over 3 kernels; Sobol on measured vs a-priori bounds | Capture dominates both views; measured #2 is P2P share, a-priori #2 is seats |
| Sensitivity module | `pipeline/tests/test_sensitivity.py` | — | Synthetic panel: lever series finite, three kernels ordered, `min_p2p` filters, Sobol responds to bounds, deterministic | All pass |

Each report is written fresh by its script on every run and gitignored; the
Script column above names the generator.

---

## Decision log

| Date | Decision | Record |
| --- | --- | --- |
| 2026-09-30 | Public flight→hotel link is not viable; do not build on total arrivals | [Phase 1](#phase-1--public-data-investigation-background) |
| 2026-09-30 | Monthly non-resident arrivals are not published after 2018; no substitute invented | [Phase 1](#phase-1--public-data-investigation-background) |
| 2026-10-01 | DCT unit contract, era routing, byte-hash duplicate rejection, YTD not consumed | `pipeline/dct/` (era routing, see [Methods appendix](#methods-appendix)) |
| 2026-10-02 | Competition pack preserved with checksums; licensed use only | [§2.1](#21-loader-and-integrity) |
| 2026-10-02 | P2P is the correct inbound denominator; old negative result refined, not overturned | [§2.2](#22-gate-experiment--pax-vs-p2p) |
| 2026-10-03 | Parametric layer built as a notebook-only engine; SALib for sensitivity | [§3.4](#34-sensitivity) |
| 2026-10-03 | Empirical-Bayes shrinkage rejected (dispersion exceeds Beta support) | [§3.3](#33-why-empirical-bayes-shrinkage-was-rejected) |
| 2026-10-03 | Capture/ALOS kept as behavioural estimated ratios with a 5.0 cap; 12 hotel-only markets kept as residual | [§3.2](#32-measured-vs-estimated-vs-assumed) |
| 2026-10-03 | Hybrid ML layer replaces the fixed conversion only, not the exact supply stages | [§4.1](#41-hybrid-design) |
| 2026-10-03 | LightGBM added; shallow a-priori config, 5-seed ensemble, iteration from internal tail (not the reported holdout) | [§4.2](#42-evaluation) |
| 2026-10-05 | Docs and notebooks consolidated into `README.md` + `METHODS.md` + three notebooks | [Layout](#repository-layout-and-data-tiers) |
| 2026-10-05 | Brief's event-calendar lever implemented UAE-wide; the pack shipped no event data, and source-market holidays stay uncovered | [§4.3](#43-the-event-calendar-and-what-it-is-worth) |
| 2026-10-05 | `market` join declared a structural defect, not a caveat: capture relabelled a calibration coefficient, market-level attribution withdrawn, guard test added | [§4.5](#45-the-market-join-is-structurally-wrong-and-what-survives) |
| 2026-10-05 | Capped chain confirmed as the *stronger* baseline (32.77% vs 36.10% uncapped); headline gap kept at the conservative 12.21 pp | [§4.6](#46-the-capture-cap-helps-the-baseline) |
| 2026-10-05 | No attempt made to re-derive the supply→nationality mapping; not identifiable from this pack without residence data | [§4.5](#45-the-market-join-is-structurally-wrong-and-what-survives) |
| 2026-10-06 | Origin mix validated against SCAD's official region-of-embarkation table (KTS 2025 t.6): pack files last-flight, not true embarkation (GCC +5–7 pp every year); no coarser region grouping rescues market-level attribution | [§4.5](#45-the-market-join-is-structurally-wrong-and-what-survives) |
| 2026-10-09 | Chain + hybrid extracted from the notebook into one canonical package (`pipeline/hybrid.py`), wrapped by a scenario API (`pipeline/simulator.py`) for the Streamlit UI; notebook delegates so there is a single model; packaging pinned by a ninth diagnostic | [Simulator package](#simulator-package-pipelinehybridpy--pipelinesimulatorpy) |
| 2026-10-10 | Simulator package hardened after a critical audit: lazy per-window fit (fit trains only the selected window), save()/load() pickle persistence, market_seats_mult made an immutable tuple validated against the 33 markets, zero-supply/unknown-market/out-of-horizon guards, NaN-safe featurize, data_status column; packaging diagnostic halved to ~75s | [Simulator package](#simulator-package-pipelinehybridpy--pipelinesimulatorpy) |
| 2026-10-11 | Learned ALOS wired into the simulator's hybrid path: `monthly()`/`market()` guest-nights now match the documented 20.8% (T4), replayed live by `reproduce_holdout()`; `chain()` keeps the fixed per-market `ALOS[m]` so the parametric layer stays interpretable; ALOS trained once, persisted, backward-safe on load | [§4.2](#42-evaluation) and [Simulator package](#simulator-package-pipelinehybridpy--pipelinesimulatorpy) |
| 2026-10-11 | Sensitivity split into two labelled views: a-priori decision ranges (T6 / `LEVER_BOUNDS`, untouched) and empirically measured lever spreads (volume-weighted 16–84th pct); empirics inform the reading and never overwrite the UI ranges | [§3.4](#34-sensitivity) |

---

## Assumptions and known defects

Carried forward, consolidated. The list below is authoritative; the competition
pack's own field definitions are in `Data_Dictionary.pdf` under
`data/raw/competition/`.

**Structural assumptions (the most important entry in this document):**

- **`market` conflates flight departure country with guest nationality.** The panel
  join asserts `Departure Country Name == Nationality`; it is false, and falsified
  by capture exceeding 1.0 in five markets (§4.5). Aggregate results remain valid
  because the error is zero-sum and cancels; **market-level attribution does not
  and must not be quoted**, including the simulator's market breakdown and the
  `seats_mult` lever. `capture` is a calibration coefficient, not a conversion
  probability. Not correctable from this pack — `Residence (groups)` is constant
  and no flight-side residence field exists. SCAD's official region-of-embarkation
  table (KTS 2025 t.6) reproduces the same reading at region level (pack GCC
  +5–7 pp every year; TV 0.15–0.20), so it is not country-level noise.

**Assumptions (not measured, disclosed):**

- Purpose-of-travel / VFR split — no source at any frequency; absorbed in capture.
- Passenger-to-visitor conversion — an explicit assumption, never an estimate.
- Seasonality is a single global monthly profile in the parametric layer.
- Volume-weighted capture/ALOS treat each market's aggregate as representative.

**Data defects (fixed or open):**

- Three parser defects in `era_key.py` corrupted the `ytd` grain (BN truncation, B
  scaled as M, MN kept as count); fixed 2026-10 with no analytical finding moved,
  because no consumer reads `ytd`.
- Source typo `2021-12` monthly total revenue (`676K` where ~600M is implied) is
  kept faithfully and flagged by `dct_validate.py`, not "fixed".
- `2015-02` and `2018-02` are byte-identical copies of a 2017-02 report and parse
  to zero KPI rows by design; 2023-11 and 2023-12 reports are unavailable.
- February 2015/2018 lost; 2018 annual carries two imputed months; 2020 annual
  arrivals absent entirely.
- 64 `derived == 1` rows are bar-chart readings for 2017-11 → 2021-03 and 2019
  quarters — any trend through 2019–2021 is partly chart-based.
- ADS-B covers only 48.9–61.5% of official arrivals and samples one day/month;
  203 Q1-2025 legs are `is_local_leg` and must be dropped.
- The 12 hotel-only markets and the grain break at 2023-01 are handled in the
  loader, not silently.
- The brief's event-calendar and public-holiday requirement has **no supporting
  data in the pack** — the data dictionary never mentions events. Covered
  UAE-wide by `pipeline/calendar_events.py`; **source-market public holidays are
  not covered** and remain open work.
- Event dates are computed from the tabular Islamic calendar, not declared. UAE
  holidays follow moon sighting, so computed dates can lead the official one by a
  day or two; every date is pinned to a two-to-five-day window by
  `test_calendar_events.py` and the features span multi-day windows. Coverage is
  2022-01 to 2026-02 and the module raises outside it rather than reporting zeros.
- The UAE changed its weekend from Fri–Sat to Sat–Sun in January 2025 and gave
  National Day a four-day weekend in December 2025. The regime change straddles
  the train/test split; the December 2025 extra days are encoded, the wider
  regime change is not a feature.

---

## Reproducibility

**Environment:** Python **3.14**, managed with `uv` (0.12.10). `pytest` is **not**
installed; `statsmodels`, `pandas`, `numpy`, `plotly`, `marimo`, `streamlit`,
`salib`, `pyopensky` and `lightgbm` are. `scipy` is present, pulled in
transitively by `statsmodels`. Dependencies are declared in `pyproject.toml`,
locked in `uv.lock`, and the project is marked non-package
(`[tool.uv] package = false`).

```
uv sync
.venv/bin/python -c "import pandas"        # confirm the environment
```

**Regenerate the public-data tables:**

```
.venv/bin/python pipeline/dct/dct_fetch_manifest.py
.venv/bin/python pipeline/dct/dct_download.py
.venv/bin/python pipeline/dct/dct_extract_text.py
.venv/bin/python pipeline/dct/dct_parse.py
.venv/bin/python pipeline/hotel_supply.py
.venv/bin/python pipeline/aviation/build_annual_arrivals.py
.venv/bin/python pipeline/aviation/build_adsb_2025.py
.venv/bin/python pipeline/aviation/calibrate_adsb_2025.py
```

**Check the work:**

```
.venv/bin/python pipeline/dct/dct_validate.py          # 2017 reconciles to 0.026%
.venv/bin/python pipeline/tests/test_market_link.py
.venv/bin/python pipeline/tests/test_aggregate_link.py
.venv/bin/python pipeline/tests/test_retarget_international.py
.venv/bin/python pipeline/tests/test_p2p_vs_pax.py
.venv/bin/python pipeline/tests/test_calendar_events.py
.venv/bin/python pipeline/tests/test_market_attribution.py   # capture > 1.0 in 5 markets; see METHODS 4.5
.venv/bin/python pipeline/tests/test_market_origin_validation.py  # pack region mix vs SCAD embarkation mix; see METHODS 4.5
.venv/bin/python pipeline/tests/test_simulator_package.py    # packaging gate; re-fits the simulator, writes no report
.venv/bin/python pipeline/tests/test_sensitivity.py         # empirical lever spreads; synthetic panel, no report
.venv/bin/python pipeline/tests/crosscheck_markets.py
cd data/processed && md5sum -c checksums.md5
```

**Run the hybrid layer (executes all cells — Part A's T2–T6 plus Part B's ML-2…ML-7;
the Part C assertions guard every pinned number for both models):**

```
.venv/bin/python -m marimo export html --no-include-code notebooks/hybrid_layer.py \
  -o /tmp/hybrid_layer.html
```

The model cells of the notebook delegate to `pipeline/hybrid.py` (see the
Simulator package subsection below), so this export *is* the canonical model and
the simulator scores the same numbers.

The empirical-sensitivity companion is fast (no LightGBM) and reproduces §3.4's
two views:

```
.venv/bin/python -m marimo export html --no-include-code notebooks/sensitivity_empirical.py \
  -o /tmp/sensitivity_empirical.html
```

This trains 30 LightGBM models: a seed-0 early-stopping pass plus a five-seed
refit for T4 and T5 (12), the same pair of arms again for the ML-7 ablation (12),
five for the learned ALOS, and one for the ML-5 importance plot. Part A on its
own is fast, but there is no way to skip Part B from inside the app.

`credentials.json` (gitignored) holds OpenSky API credentials, consumed only by
`tools/fetch_opensky.py` and `tools/measure_window_cost.py`.

> Note: the seven generated reports in `outputs/` (`market_link_test.md`,
> `aggregate_link_test.md`, `retarget_test.md`, `p2p_vs_pax_test.md`,
> `calendar_events_test.md`, `market_attribution_test.md`,
> `market_origin_validation_test.md`) are gitignored, so a
> clean clone must regenerate them before reading.

### Simulator package (`pipeline/hybrid.py` + `pipeline/simulator.py`)

The notebook is the submission; the imported package is how a simulator consumes
it. `pipeline/hybrid.py` is the **single canonical implementation** — the
notebook's Part A/B cells now delegate to it (thin cells that bind the same
names), so the notebook's pinned Part C numbers *are* the packaged model's
numbers, and `test_simulator_package.py` re-fits the package and pins it again
(for the T5 competition window, WMAPE within `20.8 ± 2.0` with aggregate bias
near `-0.1%`). The test takes ~90s: it fits the `full` window once, then the
`competition` window on demand, then replays the T4 holdout for guest-nights.

```
from pipeline.simulator import Simulator, Scenario, RouteEdit
sim = Simulator().fit()                     # trains only the requested window
sim.monthly(Scenario(seats_mult=1.2))       # safe aggregate view
sim.market(Scenario(seats_mult=1.2))        # market view, supply-side-only flag
sim.save()                                  # optional cache -> outputs/models/
```

Fitting is **lazy and per-window**: `Simulator(window=...)` selects the model
default scenarios are scored with, and `fit()` trains only that window
(`full` for scenario work, `competition` for `reproduce_competition_window`).
`save()`/`load()` pickle the fitted state to `outputs/models/` (gitignored) to
spare the ~1-minute refit on a Streamlit restart; no artifact is required, and a
fresh fit yields identical predictions.

The hybrid path's guest-nights use the **learned ALOS** — a 5-seed LightGBM
regression on the same leak-free features as the arrival model, fitted once on
the training panel — so `monthly()`/`market()` emit the guest-nights accuracy
this document quotes (§4.2, `20.8%` on T4), replayed live by
`reproduce_holdout()`. The parametric `chain()` keeps the fixed per-market
`ALOS[m]`, so the interpretable layer stays interpretable and the two paths
mirror the notebook's ML-4 head-to-head. `reproduce_competition_window()` replays
the T5 New-Arrivals protocol as before.

Lever semantics are documented in the `simulator.py` docstring: the three
supply levers re-enter the model's features *and* its Poisson exposure offset
pre-score (so the hybrid responds non-linearly); `capture_mult`, `season_mult`
and `alos_mult` are post-hoc dials. Simulation is bounded by the model's own
world: markets are a fixed 33-level categorical (new markets and
out-of-horizon route edits raise), supply levers must be positive (a zeroed
lever is rejected, not silently degenerated), and the horizon stops at the event
calendar's `COVERED_TO` = 2026-02. Because the §4.5 join is structurally wrong
at market level, `market()` returns rows flagged
`attrs["data_status"] == "supply-side-only"` **and** a constant `data_status`
column (the attr is lost by `groupby`/serialization; the column survives), and
the UI must not present market levels as measured; the aggregate views are the
defaults for a reason.

The Streamlit front end (`app/simulator.py`) is the interactive deliverable with
five tabs (Scenario, By market & season, Which levers matter, Validation,
Assumptions & limits), a decisions-first copy (`app/copy.py`), and a thin,
headlessly-testable logic layer (`app/logic.py`). The model side is complete,
deterministic (`deterministic=True`, fixed seeds, single thread) and
fit-on-demand from the competition pack — predictions are identical to the
notebook's. The app loads `outputs/models/simulator_full.pkl` if present,
otherwise fits once (`st.cache_resource`) and saves it.

---

## Artifact inventory

**Processed tables (`data/processed/`, 14):** `dct_kpis.csv`,
`dct_guests_nationality.csv`, `dct_hotel_supply.csv`, `dct_coverage.csv`,
`dct_cross_checks.csv`, `dct_notes_raw.csv`, `pax_movements.csv`,
`etihad_passengers.csv`, `pax_arrivals_annual.csv`, `omaa_arrivals_2025.csv`,
`monthly_summary.csv`, `markets_by_month.csv`, `aircraft_types.csv`,
`capture_rate.csv` (plus `checksums.md5`).

**Notebooks (marimo, three):** `hybrid_layer.py` (the deliverable — Part A the
parametric chain, Part B the hybrid LightGBM layer, Part C the shared pin block),
`linkage_study.py` (main public-data analysis, Phase 1 evidence),
`flight_analysis.py` (read of the OpenSky ADS-B extract, Phase 1 evidence).

**Tools:** `tools/fetch_opensky.py` (the OpenSky pull — the only thing in the repo
that spends credits), `tools/export_omaa_csv.py`,
`tools/measure_window_cost.py`.

**Pipeline:** `pipeline/dct/` (11 modules + `eras/` with 8 era parsers),
`pipeline/aviation/` (4 builders + `scad_region_arrivals.py` parser and
country→region mapping), `pipeline/competition/` (loader + audit),
`pipeline/hotel_supply.py`, `pipeline/calendar_events.py` (UAE event calendar,
§4.3), `pipeline/hybrid.py` (canonical chain + LightGBM layer, shared with the
notebook), `pipeline/simulator.py` (`Scenario`/`Simulator` API for the app),
`pipeline/paths.py`, `pipeline/tests/` (9 diagnostics).

**Docs:** [`../README.md`](../README.md) (entry point), this file
([`METHODS.md`](METHODS.md)), [`competition-brief.md`](competition-brief.md)
(supplied by the organisers), and the seven generated reports in `outputs/`.

**App:** `app/simulator.py` is a Streamlit stub, not yet built out by a UI
contributor. The model side it will call — `pipeline/simulator.py` — is
complete and guarded; see the Simulator package subsection under Reproducibility.

---

## Open work and next steps

1. **Do not ship a market-level view without attribution.** This is now the top
   blocker, ahead of the simulator. `pipeline/tests/test_market_attribution.py`
   pins the defect, and `test_market_origin_validation.py` confirms it against
   SCAD's official embarkation mix (pack GCC +5–7 pp every year at region level;
   no coarser grouping rescues capture), but nothing yet *resolves* it, and every
   market-split output is affected. Either (a) source passenger
   residence/nationality on the flight side, or (b) restrict all output to
   aggregate and re-scope success criteria 01, 05 and 06, which are rendered by
   market. Attempting to infer the mapping from the pack alone is not
   identifiable: a 33×45 origin-to-nationality matrix on 1,405 rows has more
   parameters than data.
2. **The interactive front end is implemented.** `app/simulator.py` (Streamlit) with
   tabs for Scenario, market/season, sensitivity, validation, and assumptions; the
   model side (`pipeline/hybrid.py` + `pipeline/simulator.py`) is complete with
   lazy per-window fit, save/load, and guards. The front end follows the guidance:
   aggregate views remain the default, and the market breakdown and per-market
   seat mix are explicit, clearly labelled `supply-side-only`. Periodically
   revisit (1) as a data fix if residence data becomes available.
3. **Measure capture for the 52 air-only origins.** OpenSky data is present in
   `data/raw/opensky/` but unconsumed; this would extend the chain beyond the 33
   matched markets. Re-pulling requires `tools/fetch_opensky.py` and spends
   credits. Note this extends *supply* coverage and does nothing for the
   attribution defect.
4. **Market-specific seasonality** once more than 43 months are available.
5. **Serve the ML layer.** Done: `Simulator().fit()` trains the hybrid end to
   end from the pack and `monthly()`/`market()` score scenarios with the learned
   capture; the remaining work is the UI that exposes it (item 2).
6. **Test-frame scoring beyond New Arrivals** is impossible without `Guests`; note
   it explicitly in the submission.
7. **Source-market public holidays.** The brief asks for them and the pack shipped
   none; only the UAE-wide events are modelled. Closing this needs a maintained
   holiday table per market for at least India, Egypt, Saudi Arabia, Qatar, Turkey
   and the UK, which is a sourcing problem rather than a modelling one. The
   ablation in §4.3 is the template: add the feature set, run both arms once, and
   report the difference without retuning anything else.
8. **Extend the pinned calendar** past 2026-02 before forecasting further out. The
   module raises rather than silently reporting zeros, so this is a loud failure,
   not a quiet one.

---

## Known stale references

None. Earlier drift in `README.md` (notebook/diagnostic counts, reproduce order,
the `scipy` note) and two `linkage_study.py` comments that pointed at `scripts/`
were corrected while consolidating the documentation.

---

## Glossary

- **PAX** — all passengers on board a flight, including transfers and transit.
- **P2P** — point-to-point; passengers whose journey terminates at Abu Dhabi.
- **Transfer / transit** — passengers connecting through AUH to elsewhere; no
  Abu Dhabi hotel stay.
- **Capture rate** — `New Arrivals / P2P`, per market. **A calibration
  coefficient, not a conversion probability** (§4.5): it exceeds 1.0 in five
  markets because the `market` key conflates departure country with nationality.
  It is the share of *own-departure* passengers matched by *that nationality's*
  guests, with connecting traffic filed under the hub.
- **ALOS** — average length of stay, `Guests / New Arrivals`.
- **Guests** — guest-nights present on a given day (a stock).
- **New Arrivals** — hotel check-ins (a flow).
- **WMAPE** — weighted mean absolute percentage error.
- **TVD** — total variation distance between two distributions, in percentage
  points.
- **ADS-B** — automatic dependent surveillance–broadcast; here, arriving-flight
  snapshots at AUH.
- **VFR** — visiting friends and relatives; non-hotel travel, unmeasured.
- **Common support** — the set of markets measured on both the air and hotel
  sides, over which shares are renormalised.
