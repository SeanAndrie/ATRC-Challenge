# ATRC Flight-to-Hotel Demand Simulator

### From flight changes to explainable hotel-demand scenarios

**ATRC Challenge · DCT Abu Dhabi**  
An interactive decision-support prototype that connects air supply to hotel demand.

*One transparent conversion chain. One trained LightGBM hybrid. One place to test “what if?”*

**Headline result:** held-out test WMAPE **20.55%** vs **32.77%** for the parametric baseline, at **−0.14%** aggregate bias.

<!-- PAGE BREAK -->

# Contents

1. **Title** — Flight-to-Hotel Demand Simulator
2. **Contents** — The story in ten pages
3. **The challenge** — Two plans, two sets of numbers, one missing link
4. **The data** — What the competition pack does and does not tell us
5. **Baseline model** — An auditable seats-to-guest-nights chain
6. **LightGBM hybrid** — Learned, market-aware conversion
7. **Product & results** — The simulator, the numbers, the decision boundary
8. **Future 1** — Make market attribution decision-grade
9. **Future 2** — Turn the prototype into a planning workflow
10. **Thank you** — Pilot the next decision

<!-- PAGE BREAK -->

# The challenge: two plans that never meet

## Flight capacity and hotel demand are planned separately

Route launches, weekly frequency, aircraft gauge, load factors and seasonal schedules all move potential hotel demand — but today the aviation and hospitality teams use two separate forecasts and no shared model.

### The brief’s five questions

- A new route opens from a country that never flew here
- An airline runs more (or fewer) flights per week
- Airlines add seats — larger aircraft or extra flights
- A higher (or lower) share of seats gets filled
- The mix of countries flying in shifts with the seasons

### Our response

Pair an **explainable parametric chain** with a **LightGBM hybrid**, and expose both through an **interactive Streamlit simulator** that answers those questions in seconds.

**Scorecard:** technical rigour 40% · creativity 20% · practicality 20% · communication 20%.

<!-- PAGE BREAK -->

# The data: a strong supply signal, with one honest caveat

### The competition pack

- **117,608** flight records across **33** departure markets
- **1,405** matched market-months over **43** months (train: Jan 2022 – Jul 2025)
- **231** market-months in the held-out test window (Aug 2025 – Feb 2026)
- Flight identity holds: **PAX = P2P + transfer + transit**

### Why point-to-point (P2P) is the denominator

P2P strips out transfer and transit passengers. On the matched panel it explains hotel arrivals better than total PAX (**R² 0.877 vs 0.863**). Global capture **0.391**; global ALOS **3.60** nights per arrival.

### The caveat that shapes everything

The flight market is **the last departure country before AUH**; the hotel market is **guest nationality**. The join is not valid for market attribution, and twelve hotel-only markets (Australia, Brazil, …) carry no flight supply.

### Built to keep planning

Observed supply ends **2026-02**; the model holds the same calendar month one year forward to **2028-12** for forward scenarios — clearly marked as a baseline, not a schedule forecast.

<!-- PAGE BREAK -->

# Baseline model: every conversion visible

## Seats → PAX → P2P → New Arrivals → Guest-nights

1. **Seats** — scheduled capacity from flight records
2. **PAX** — seats × observed load factor
3. **P2P** — PAX × observed point-to-point share
4. **New Arrivals** — P2P × market capture (capped at 5.0), adjusted by monthly seasonality
5. **Guest-nights** — New Arrivals × average length of stay (ALOS)

Load factor and P2P share come straight from flight data; capture and ALOS are estimated from joined history. Every stage is inspectable and every ratio is a lever.

### Historical baseline

| Evaluation | New Arrivals WMAPE | Guest-nights WMAPE |
| --- | ---: | ---: |
| T4 holdout (Feb–Jul 2025) | **28.4%** | **25.6%** |
| T5 test (Aug 2025–Feb 2026) | **32.77%** | Not scoreable — test has no Guests target |

The T4 seasonal-naive arrivals benchmark is **36.0% WMAPE**. The capture cap helps the baseline (36.10% uncapped → 32.77% capped), so the headline comparison is against the stronger chain.

<!-- PAGE BREAK -->

# LightGBM hybrid: learn the behavioural conversion

The model keeps the measured **Seats → PAX → P2P** arithmetic and replaces the fixed capture step with a **Poisson LightGBM model using log(P2P) as its exposure offset**. Five deterministic seeds are averaged.

### Inputs known at prediction time — never the target

Market, month, time trend, seats, load factor, P2P share, transfer share, routes, airlines and UAE event-day features. No hotel target or target-derived feature enters a prediction, so the test window stays honest.

### T5 held-out result

| Model | New Arrivals WMAPE | Aggregate bias | Predicted arrivals |
| --- | ---: | ---: | ---: |
| Parametric chain | 32.77% | +1.14% | 1,610,907 |
| **LightGBM hybrid** | **20.55%** | **−0.14%** | **1,590,506** |
| Actual | — | — | 1,592,790 |

The five-seed range is **20.30–20.85%**. Guest-nights reach **20.8% WMAPE** on the T4 holdout with a learned ALOS; T5 guest-nights cannot be scored because the test pack omits Guests.

<!-- PAGE BREAK -->

# Product & results: from “what if?” to a comparable number

## The simulator answers the brief’s questions directly

- **New route / new country** — modelled on a chosen analog market and labelled as an estimate
- **Frequency, seats, aircraft** — signed seat edits with start/end months, plus flight↔seat calculators
- **Load factor** — global or per-market fill-rate dials
- **Seasonal mix** — per-market seat multipliers or winter-only route edits

Every scenario is scored by the hybrid and the chain against a no-change baseline: horizon Δ guest-nights and arrivals, monthly and seasonal views, one-at-a-time contribution bars, an auditable chain table, a supply-side-only market breakdown, a route explorer and a save/compare library.

### The result

On the competition test window the hybrid cuts New Arrivals WMAPE by **12.21 percentage points** against the stronger, capture-capped chain, with aggregate prediction within **0.14%** of actual.

### Decision boundary

**Compare scenarios on the aggregate; treat market levels as supply-side-only.** Capture exceeds 1.0 in five joined markets — the departure-country/nationality mismatch in plain sight. Occupancy is reported as guest-nights because the pack has no room-capacity data.

<!-- PAGE BREAK -->

# Future 1: make market attribution decision-grade

## Resolve the join before making market-level promises

**Data work**

- Link flights to passenger residence/nationality or true embarkation market
- Align the flight and hotel market definitions before fitting market capture
- Validate the mapping against SCAD/official embarkation totals
- Keep hotel-only origins visible as a residual until their supply is measured

**Model work**

- Re-estimate capture and ALOS on a validated market key
- Re-run the same held-out tests and report market-level error only after validation
- Extend source-market holiday coverage and refresh the event calendar

**Gate to proceed:** market capture and allocation reconcile with independent evidence; any residual is explicit, never synthetic.

<!-- PAGE BREAK -->

# Future 2: turn the prototype into a planning workflow

### From one-off scenarios to repeatable planning

- Ingest forward route schedules and airline capacity proposals as scenario inputs
- Expand outcomes from arrivals and guest-nights to occupancy and hotel capacity once room data exists
- Add uncertainty ranges, model monitoring and scheduled back-tests as new actuals arrive
- Move saved scenarios to durable shared storage for team review and audit
- Pilot one route or seasonal decision; compare forecasts with subsequent hotel results

### Practical rollout

1. Validate market mapping and pick a pilot decision
2. Agree data owners, refresh cadence and forecast horizon
3. Review scenarios with aviation and hotel planners
4. Measure accuracy, usefulness and residual coverage before expanding

<!-- PAGE BREAK -->

# Thank you

## Make flight and hotel planning one conversation

**ATRC Flight-to-Hotel Demand Simulator**  
Transparent assumptions · Held-out validation · Interactive scenarios

### The next step

Pilot the aggregate simulator on one route or seasonal planning question — and validate the market link before relying on market-level recommendations.

*Resolving the departure-country/nationality join is the single most valuable next milestone: it unlocks trustworthy origin-specific planning on top of a model that already predicts the national total within a fraction of a percent.*
