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
built, with every stage a measured ratio and every ratio an override lever. It is
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
and replaces the fixed conversion ratio with a **LightGBM** Poisson model using
`log(P2P)` as an exposure offset, i.e. a learned, shrunk, market-aware capture
rate. It beats the parametric chain on every held-out metric: T4 New Arrivals
**21.7%** vs 28.4%, competition window **20.8%** vs 32.8% (aggregate bias −0.5%
vs +1.1%), and holdout guest-nights **21.0%** with a learned ALOS. `market`
dominates feature importance — the model is the regularised answer to the 36×
capture spread. See [Phase 4](#phase-4--machine-learning-layer).

---

## The problem and the success criteria

From the [brief](competition-brief.md), evaluation is 40% technical accuracy and
modelling rigour, 20% creativity/practicality, 20% clarity, plus a 20% realism
weight. The seven success criteria map to the work as follows:

| # | Criterion | Where it is addressed |
| --- | --- | --- |
| 01 | A working simulator, not a static report | `notebooks/hybrid_layer.py` Part A (levers + charts), `app/simulator.py` (stub, [open work](#open-work-and-next-steps)) |
| 02 | Transparent conversion chain, assumptions adjustable | [Phase 3](#phase-3--parametric-layer) |
| 03 | Validation against history with an honest error metric | T3/T4/T5, WMAPE on holdout and test window |
| 04 | Sensitivity analysis | T6 (SALib Sobol + one-at-a-time elasticities) |
| 05 | Granularity by market and season | 33-market panel, global monthly season index |
| 06 | Decision relevance | [Scope limits](#scope-limits-and-honest-positioning); capture spread is the allocation lever |
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
pipeline/tests/    five standalone diagnostics
notebooks/         three marimo notebooks
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
`departure country == nationality × month`:

- **1,405 market-months, 33 markets, 43 months (2022-01 → 2025-07).**
- Coverage of pack P2P against SCAD AUH arrivals: **~77%** (2022 72.9%, 2023
  78.1%, 2024 77.3%).
- **Capture (`New Arrivals / P2P`)**: mean 0.83, median 0.53, highly
  heterogeneous — market means span ~35× (0.083–2.958 after capping extremes at
  5), matching Phase 1's 36× spread.
- **ALOS (`Guests / New Arrivals`)**: ~3.60, anchoring the stock-flow step.

Why this matters: the pack supplies both `New Arrivals` and the terminating
denominator `P2P`, so capture is now *measured* rather than assumed — the exact
variable that was unavailable in Phase 1.

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

Every stage is a measured, volume-weighted ratio exposed as an override lever, so
the whole thing is auditable and explainable to a non-technical planner.

### 3.2 Measured vs estimated vs assumed

- **Measured:** seats, load factor, P2P share, and capture itself (because the
  pack supplies both `New Arrivals` and `P2P`). This is the change from Phase 1.
- **Estimated:** `capture[m]` and `ALOS[m]`, volume-weighted so small markets
  cannot outvote large ones. Global capture **0.391**; global ALOS **3.602**.
- **Assumed:** purpose-of-travel/VFR, which the brief lists as a lever but no
  source measures. It is absorbed inside the measured capture rate, and the
  simulator must not present the visitor split as measured.

### 3.3 Why empirical-Bayes shrinkage was rejected

The natural guard for thin markets is a Beta-binomial shrinker toward the global
mean. It was implemented, tested, and **rejected**: between-market capture
dispersion (0.085 → 5.854, median 0.554) exceeds the maximum variance a Beta
prior can represent, so the method-of-moments concentration goes negative. That
is real behavioural heterogeneity — Phase 1's 36× spread — not sampling noise. The
measured rate is kept with the 5.0 cap as the only guard.

### 3.4 Sensitivity

Test T6 runs a **SALib Sobol** global sensitivity over six levers plus
one-at-a-time +10% elasticities. Every lever is multiplicative, so all six have
unit elasticity by construction; the output variance is dominated by the
wider-bound levers — capture (0.7–1.3) and seats/frequency (0.8–1.2) — with load
factor, P2P share, ALOS and season (all 0.9–1.1) contributing proportionally
less. This is why the chain reports a **range**, not a point. (Note: the SALib
package import name is `SALib`, and `saltelli` is replaced by `Sobol.sample`.)

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
replaces **only** the behavioural conversion:

```
seats → PAX → P2P             [parametric chain, exact]
P2P  → New Arrivals           [LightGBM Poisson, offset log(P2P)]
New Arrivals → guest-nights    [chain ALOS[m] or a learned ALOS]
```

The Poisson offset makes the model learn an intensity per terminating passenger —
a learned, shrunk, market- and season-aware capture rate. Features are
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

Exposure offset: `log(Total_P2P)`. The competition test window supplies all of
these, so a forecast needs no target history.

### 4.2 Evaluation

Identical splits and metric as the chain. Configuration was chosen **a priori**,
not tuned on the reported holdouts: `num_leaves=15`, `min_child_samples=20`,
`learning_rate=0.03`, `feature_fraction=0.8`, `bagging_fraction=0.8`,
`lambda_l2=1.0`, single thread, deterministic. Iteration count is selected by
early stopping (patience 100, cap 2000) on an internal 6-month tail of each fit
window, then a **5-seed ensemble** (`0,1,2,42,2024`) is refit on the full fit
window and averaged on the raw-score scale before the exp link. Selected
iterations: **1334** (T4) and **1934** (T5).

| metric | parametric chain | hybrid LightGBM |
| --- | ---: | ---: |
| T4 New Arrivals WMAPE (198 market-months) | 28.4% | **21.7%** (seeds 21.1–22.1) |
| T5 New Arrivals WMAPE (231 market-months) | 32.8% | **20.8%** (seeds 20.4–21.2) |
| T5 aggregate bias | +1.1% | **−0.5%** |
| T4 guest-nights WMAPE, chain ALOS | 25.6% | 23.8% |
| T4 guest-nights WMAPE, learned ALOS | — | **21.0%** |
| seasonal-naive New Arrivals baseline, T4 | 36.0% | — |

### 4.3 Interpretation and scope limits

`market` dominates gain importance (`market` ≫ `p2p_share` > `log_seats` >
`routes` > `transfer_share` > `t` > `load_factor` > `month` > `airlines`) —
confirming that a shrunk per-market capture rate is the real improvement over the
raw ratio. ML-5 plots learned capture against the chain's capped ratio to show the
shrinkage directly; thin markets are pulled toward the centre.

Limits, mirrored in the notebook: the test window has no `Guests`, so guest-nights
are validated in-sample and on the leave-tail holdout only; the panel is small
(1,405 rows) so the model is shallow by design; capture stays behavioural; and the
12 hotel-only markets remain unmodelled.

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
| ML hybrid | `notebooks/hybrid_layer.py` Part B | — | LightGBM Poisson with offset, same T4/T5 splits | T4 21.7%, T5 20.8%, guest-nights 21.0% |

All report-generating diagnostics carry a `Generated by` line naming the script,
so report checksums are only comparable with that line excluded.

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
| 2026-10-03 | Capture/ALOS kept as behavioural measured ratios with a 5.0 cap; 12 hotel-only markets kept as residual | [§3.2](#32-measured-vs-estimated-vs-assumed) |
| 2026-10-03 | Hybrid ML layer replaces the fixed conversion only, not the exact supply stages | [§4.1](#41-hybrid-design) |
| 2026-10-03 | LightGBM added; shallow a-priori config, 5-seed ensemble, iteration from internal tail (not the reported holdout) | [§4.2](#42-evaluation) |

---

## Assumptions and known defects

Carried forward, consolidated. The list below is authoritative; the competition
pack's own field definitions are in `Data_Dictionary.pdf` under
`data/raw/competition/`.

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
.venv/bin/python pipeline/tests/crosscheck_markets.py
cd data/processed && md5sum -c checksums.md5
```

**Run the hybrid layer (executes all cells — Part A's T2–T6 plus Part B's ML-2…ML-5;
the Part C assertions guard every pinned number for both models):**

```
.venv/bin/python -m marimo export html --no-include-code notebooks/hybrid_layer.py \
  -o /tmp/hybrid_layer.html
```

This trains 15 LightGBM models (T4 and T5 at five seeds, plus five for the learned
ALOS), so it takes a while. Part A on its own is fast, but there is no way to skip
Part B from inside the app.

`credentials.json` (gitignored) holds OpenSky API credentials, consumed only by
`tools/fetch_opensky.py` and `tools/measure_window_cost.py`.

> Note: the four generated reports in `outputs/` (`market_link_test.md`,
> `aggregate_link_test.md`, `retarget_test.md`, `p2p_vs_pax_test.md`) are
> gitignored, so a clean clone must regenerate them before reading.

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
`pipeline/aviation/` (4 builders), `pipeline/competition/` (loader + audit),
`pipeline/hotel_supply.py`, `pipeline/paths.py`, `pipeline/tests/` (5 diagnostics).

**Docs:** [`../README.md`](../README.md) (entry point), this file
([`METHODS.md`](METHODS.md)), [`competition-brief.md`](competition-brief.md)
(supplied by the organisers), and the four generated reports in `outputs/`.

**App:** `app/simulator.py` is a 7-line Streamlit stub, not yet built out.

---

## Open work and next steps

1. **Build the interactive simulator.** The chain exists in Part A of
   `notebooks/hybrid_layer.py`; the Streamlit/FastAPI front end that exposes the
   six levers and renders by market and season does not. This is the most direct
   route to success criteria 01, 05 and 06.
2. **Measure capture for the 52 air-only origins.** OpenSky data is present in
   `data/raw/opensky/` but unconsumed; this would extend the chain beyond the 33
   matched markets. Re-pulling requires `tools/fetch_opensky.py` and spends
   credits.
3. **Market-specific seasonality** once more than 43 months are available.
4. **Serve the ML layer.** The hybrid is currently notebook-only; wire it in
   behind the simulator/API so the learned capture can drive day-to-day planning,
   not just the report.
5. **Test-frame scoring beyond New Arrivals** is impossible without `Guests`; note
   it explicitly in the submission.

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
- **Capture rate** — `New Arrivals / P2P`, per market; the share of terminating
  passengers who check into Abu Dhabi hotels.
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
