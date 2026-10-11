# ATRC Flight-to-Hotel Demand Simulator

### Turning flight decisions into hotel-demand answers — in seconds, with the working shown

**ATRC Challenge · DCT Abu Dhabi**  
A working decision-support prototype that converts air supply into hotel guest-nights.

*One transparent conversion chain. One learned conversion layer. One place to ask “what if?”*

| Held-out accuracy | National-total bias | Gain over parametric baseline | Scenario answer time |
| :---: | :---: | :---: | :---: |
| **20.55% WMAPE** | **−0.14%** | **12.21 pp** | **seconds** |

## 1. Executive summary

**The problem.** Abu Dhabi plans flights and hotels with two separate teams and two separate sets of numbers. Nothing connects them, so a basic question like *“what happens to hotel nights if this route adds two flights a week?”* has no quick, defensible answer.

**What we built.** An interactive simulator that runs every flight change through an explainable chain — **Seats → Passengers → Point-to-point arrivals → New hotel arrivals → Guest-nights**. A **LightGBM Poisson model** replaces the chain's one weak link: the fixed “share of arriving passengers who become hotel guests”. It learns that share by market, by season and around UAE events.

**How well it works.** On a held-out window the model never saw (Aug 2025 – Feb 2026), it predicts new hotel arrivals at **20.55% WMAPE**, against **32.77%** for the parametric chain. Its national total is **within 0.14%** of actual, and five random seeds all land between 20.30% and 20.85%.

**What a planner gets.** A Streamlit app with one page for each of the brief's three headline questions (new route, winter flight cut, guests per seat), plus a scenario lab, a validation page and a **plain-language chatbot**. Every number on every page, and in the chat, comes from the same model. Example answers from the prototype:

- A new **twice-weekly, 300-seat route** (modelled on UK-market behaviour, Nov 2026 – Oct 2027) adds **≈ 35,000 guest-nights** and **≈ 8,200 new hotel arrivals**.
- Cutting **7 weekly London flights** over winter 2026/27 costs **≈ 49,000 guest-nights**, about **1.4%** of national winter demand.

**What makes it trustworthy.** We found and documented a structural flaw in the data: flight “country” is not the same as hotel “country”. So the product leads with numbers that hold at the national level, and labels market-level views honestly.

<!-- PAGE BREAK -->

# 2. Intro / team overview

## Who we are

> *[Team name] — [university / organisation]*

| Member | Role | Owned |
| --- | --- | --- |
| *[Name]* | Data & modelling lead | Data-pack loader and audit, parametric chain, LightGBM hybrid, validation |
| *[Name]* | Product & simulator lead | Scenario API, Streamlit app, scenario library, deployment |
| *[Name]* | AI & experience lead | Chatbot agent, grounded tools, guardrails, presentation and video |

## What we delivered — a working prototype, not a concept

| Deliverable | What it is |
| --- | --- |
| **The model** | One canonical module (`pipeline/hybrid.py`): parametric chain + LightGBM hybrid, deterministic and reproducible |
| **The scenario engine** | `pipeline/simulator.py`: six global levers, signed and dated route edits, analog estimates for new origins, horizon to Dec 2028 |
| **The simulator app** | `app/streamlit_app.py`: a page per headline question, a scenario lab with sensitivity analysis and a save/compare library, and live back-test validation |
| **The chatbot** | *Ask the model* page: the brief's five what-if questions answered by tools that call the real model, with a chart per answer |
| **The evidence** | `docs/METHODS.md`: every test, every decision, every limitation, with dates |

## How we work

- **Measure before assuming.** Every ratio in the chain is either read off the data or estimated from it, and labelled as such.
- **Report the honest number.** We publish held-out error, not training fit. We state limits inside the product, not in an appendix.
- **One source of truth.** The notebook, the app and the chatbot all call the same model, so they can never disagree.

<!-- PAGE BREAK -->

# 3. The problem — and why it matters

## Two plans that never meet

Almost every international visitor reaches Abu Dhabi by air, but only some of them stay in a hotel. Others are transferring to another flight, returning residents, or visiting family. The link between flights and hotels is therefore real but **indirect**. What matters is the **conversion rate** from arriving passenger to hotel guest, and that rate changes with the country of origin and the season.

Today, route decisions and hotel planning run on separate forecasts. Five everyday questions have no quick answer:

1. A **new route** opens from a country that has never flown here.
2. An airline runs **more or fewer flights** per week.
3. Airlines add **seats**, through larger aircraft or extra flights.
4. A **higher or lower share** of those seats is filled.
5. The **mix of countries** flying in shifts with the seasons.

## Why it matters

| Stake | What goes wrong today |
| --- | --- |
| **Revenue** | Routes are supported without a quantified hotel return; incentives cannot be weighed against guest-nights |
| **Capacity** | Rooms, staff and events are planned against demand that a schedule change has already moved |
| **Early warning** | A winter frequency cut shows up in hotel figures months late, after the response window has closed |
| **Shared language** | Aviation talks in seats and frequencies, hospitality in guests and nights; with no conversion, the two cannot trade off |

## The hidden data catch

The two datasets define **“country” differently**. In the flight data it is **where the flight departed**; in the hotel data it is **the guest's nationality**. An Indian traveller connecting through Doha shows up as **Qatar** supply and **India** demand. A naive join quietly assigns hotel guests to the wrong countries. We measured that effect (section 4) and designed the product around it.

<!-- PAGE BREAK -->

# 4. Our solution — a detailed proposal

## Layer 1: a transparent parametric chain

| Stage | How it is computed | Status |
| --- | --- | --- |
| **Seats** | Scheduled capacity per route and month | Measured |
| **Passengers (PAX)** | Seats × load factor | Measured |
| **Point-to-point (P2P)** | PAX minus transfer and transit passengers | Measured |
| **New hotel arrivals** | P2P × market capture rate × seasonality | Estimated (global capture 0.391) |
| **Guest-nights** | New arrivals × average length of stay | Estimated (ALOS 3.60 nights) |

Every ratio is an adjustable lever, and every stage can be inspected in the app. **Why P2P and not total passengers:** on 1,405 matched market-months, P2P explains hotel demand better (**R² 0.877 vs 0.863**, tighter error). Transfer passengers never leave the airport.

## Layer 2: a learned conversion (the hybrid)

The first three stages are exact arithmetic, so there is nothing to learn there. The **only** step that we replace is the fixed capture rate. A **LightGBM Poisson model with log(P2P) as its exposure** learns the capture rate for each market and each month.

- **Features** are only those known at prediction time: market, month, trend, seats, load factor, P2P and transfer share, route and airline counts, and **UAE event days** (Eid, Ramadan, National Day, New Year).
- **No leakage.** No hotel figure, or anything derived from one, is ever used as an input.
- **Stability.** A five-seed ensemble, single-threaded and deterministic, so every published number reproduces exactly.
- **Event calendar.** The data pack did not include one, so we rebuilt it from a Hijri calendar conversion. It adds a small but consistent gain (+0.41 pp and +0.25 pp on two independent windows).

| Held-out test (Aug 2025 – Feb 2026) | WMAPE | Aggregate bias | Predicted | Actual |
| --- | ---: | ---: | ---: | ---: |
| Seasonal-naive (no flights) | 36.0%* | — | — | — |
| Parametric chain | 32.77% | +1.14% | 1,610,907 | 1,592,790 |
| **LightGBM hybrid** | **20.55%** | **−0.14%** | **1,590,506** | 1,592,790 |

<sub>*Seasonal-naive was scored on the earlier Feb – Jul 2025 holdout, where the chain scored 28.4% and the hybrid 21.3%.</sub>

## Which factors move hotel demand most (sensitivity)

Hybrid response, as the change in total guest-nights:

| Lever | Change | Δ guest-nights | Elasticity |
| --- | ---: | ---: | ---: |
| Capture rate (post-model dial) | +10% | +10.0% | 1.00 |
| Average length of stay | +5% | +5.0% | 1.00 |
| **Seats** | +10% | **+5.8%** | **0.58** |
| Load factor | +5% | +2.1% | 0.42 |
| P2P share | +5% | +2.4% | 0.47 |

**The planning insight.** Adding seats does not add hotel guests one-for-one. The model shows **diminishing returns to capacity**, consistent with the 0.44 elasticity in the econometric test. Conversion (who arrives, and whether they stay) moves the number more than raw seat count.

**What actually moves demand.** Equal ±10% shocks show what a planner *can* move. We also measured how far each ratio really moved month to month in 2022–2025, then ran a global Sobol analysis (SALib) under both sets of ranges:

| Lever | Share of variance, allowed ranges | Share of variance, measured swings |
| --- | ---: | ---: |
| **Hotel capture rate** | **51%** | **62%** |
| Transfer share (P2P) | 6% | **14%** |
| Load factor | 6% | 7% |
| Length of stay | 6% | 7% |
| Seats / frequency | 23% | 5% |
| Season | 6% | 3% |

Both views agree that **who becomes a hotel guest** dominates. Real seat supply only wobbles about ±9% month to month, so in practice transfer share matters more than capacity.

## The data catch, solved honestly

If flight and hotel “country” matched, no market could have more hotel guests than terminating passengers. **Five markets do**: China has a capture rate of 5.85, the US 1.99 and Russia 1.53, which means about 903,000 guests were filed against the wrong origin. The data pack cannot correct this, because it has no residence field. The errors do **cancel in the national total**, which is why the aggregate is accurate to 0.14%. **Our design response:**

- Lead with aggregate results.
- Label market views *supply-side-only*.
- Treat new origins as clearly labelled **analog estimates**.

<!-- PAGE BREAK -->

# 5. System design

## How the pieces fit

```
   ┌──────────────────────────────────────────────────────────────┐
   │  1. DATA            Competition pack                         │
   │     flights by route & date · hotel arrivals & nights        │
   │     → loader repairs + integrity audit (117,608 flight rows) │
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  2. FEATURES        Matched market × month panel             │
   │     supply ratios · routes · airlines · UAE event calendar   │
   │     1,405 market-months · 33 markets · horizon to 2028-12    │
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  3. MODEL           One canonical model                      │
   │     Seats → PAX → P2P   (exact, parametric)                  │
   │     P2P → New Arrivals  (LightGBM Poisson, 5 seeds)          │
   │     → Guest-nights      (length of stay)                     │
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  4. SCENARIO ENGINE                                          │
   │     user edits → validate → rewrite supply → forward-project │
   │     → score hybrid + chain → Δ vs no-change baseline         │
   └───────────────┬──────────────────────────────┬───────────────┘
                   ▼                              ▼
   ┌──────────────────────────────┐  ┌────────────────────────────┐
   │  5a. SIMULATOR APP           │  │  5b. CHATBOT AGENT         │
   │  3 question pages · lab      │  │  plain-language questions  │
   │  validation · library        │  │  8 tools → the same model  │
   └──────────────────────────────┘  └────────────────────────────┘
```

## Where each layer lives

| Layer | Module | Key design choice |
| --- | --- | --- |
| Data | `pipeline/competition/` | One repair-aware loader; accounting identities checked on every row |
| Features | `pipeline/hybrid.py`, `pipeline/calendar_events.py` | Only features known at prediction time; event days computed, not hand-typed |
| Model | `pipeline/hybrid.py` | Notebook, app and chatbot share one module, so their numbers never diverge |
| Engine | `pipeline/simulator.py` | Guardrails in the API: invalid markets, zeroed levers and out-of-horizon dates are rejected |
| Apps | `app/core.py`, `app/views/`, `app/agent.py` | Pages and chatbot share one set of scenario builders; the fitted model loads in about a second; no raw data is needed at runtime |

## What happens when a planner moves a lever

1. **Edit.** For example: *two extra 300-seat flights a week from the UK, Nov 2026 – Oct 2027* (+600 seats/week).
2. **Rewrite supply.** The edit changes seats on that route. PAX and P2P follow, using the route's observed load factor and P2P share.
3. **Rebuild features.** Supply ratios update, so the model responds non-linearly. Added seats join the market's busiest existing route: the model's route-count feature mostly encodes market size and would inflate a single new service several-fold.
4. **Score.** The hybrid and the chain both predict new arrivals and guest-nights for every month.
5. **Compare.** Δ against the no-change baseline, broken down by month, season, market and individual change.

<!-- PAGE BREAK -->

# 6. Intended impact

## The brief's own questions, answered by the prototype

| Question | Prototype answer |
| --- | --- |
| *How many extra hotel nights would a new twice-weekly flight generate?* | **≈ +35,000 guest-nights / +8,200 arrivals** over 12 months (600 seats/week, modelled on UK-market behaviour) |
| *How much would winter hotel demand suffer if an airline cut flights?* | **≈ −49,000 guest-nights** (−1.4% of national winter demand) for 7 fewer weekly London flights, Nov–Feb |
| *Which countries bring the most hotel guests per available seat?* | A live ranking on the **Guests per seat** page: hotel arrivals per 1,000 seats, a conversion-vs-stay chart and a month-by-market heatmap |

## For DCT planners

- **Seconds, not spreadsheets.** A route proposal becomes a one-row scenario with a quantified hotel return, before the negotiation rather than after.
- **Compare before committing.** Saved scenarios line up side by side, so incentive packages can be weighed against guest-nights.
- **Know which lever to pull.** Contribution bars rank each change. The sensitivity results show that conversion and length of stay move demand more than raw seats.
- **Earlier warning.** A schedule cut feeds straight into the expected winter guest-nights, while there is still time to respond with marketing, events or other carriers.

## For the organisation

- **Usable today at the national level.** The national total is within 0.14% on unseen months.
- **A shared language.** Aviation and hospitality teams read the same number, produced by the same auditable chain.
- **Institutional memory.** Every scenario is saved with its inputs and outputs, and every model decision is logged with its reason.

## For the visitor economy

Better-aligned flights and hotels mean fewer empty rooms after supply shocks and fewer turned-away guests at peaks. Abu Dhabi can also give airlines a faster, evidence-based answer when they ask the destination to support a new route.

<!-- PAGE BREAK -->

# 7. Feasibility and real-world implementation

## It runs today, on modest infrastructure

| Dimension | Reality |
| --- | --- |
| **Data** | Runs on the competition pack alone; no external feeds are required |
| **Compute** | Trains in about 1–2 minutes on a laptop CPU; no GPU; a light energy footprint |
| **Serving** | One saved model file; Streamlit on a small VM or Streamlit Community Cloud (private) |
| **Privacy** | Aggregated data only; no passenger-level records; the licensed pack never leaves the training environment |
| **Chatbot** | One API key (OpenRouter), with a configurable model; the default is low-cost |

## Rollout plan

| Phase | Timeline | Goal |
| --- | --- | --- |
| **Pilot** | Months 0–3 | One planning team; real route and seasonal questions; national-level outputs only |
| **Integrate** | Months 3–6 | Feed forward airline schedules automatically; quarterly re-fit with back-test before publishing |
| **Extend** | Months 6–12 | Add residence/nationality and room-capacity data, to unlock market-level and occupancy outputs |

## What would make it better

- **Forward airline schedules**, so the baseline after Feb 2026 becomes a real forecast instead of last year held forward.
- **Passenger residence or nationality on the flight side**, which fixes the market-attribution problem directly.
- **Hotel room capacity**, so winter impact is expressed as occupancy points as well as guest-nights.
- **Source-market public holidays** (India, Egypt, Saudi Arabia, the UK and others), extending the event calendar.

## Risks and how we manage them

| Risk | Response |
| --- | --- |
| Market attribution is wrong in the source data | National-level outputs first; market views labelled; attribution fix is milestone one |
| No observed data after Feb 2026 | Held-forward baseline, labelled observed / projected / scenario on every row |
| Model drift as networks change | Scheduled back-tests; a pinned replay of the test window on every release |
| Planners distrust black boxes | Explainable chain beside every ML answer; caveats inside the product; chatbot for non-analysts |

<!-- PAGE BREAK -->

# 8. Testing and demonstration

## Already validated

| Test | Result |
| --- | --- |
| Held-out test window (Aug 2025 – Feb 2026, 231 market-months) | Hybrid **20.55%** vs chain 32.77% WMAPE; bias **−0.14%**; five-seed range 20.30–20.85% |
| Leave-tail holdout (Feb – Jul 2025, 198 market-months) | New arrivals **21.3%** (chain 28.4%, seasonal-naive 36.0%); guest-nights **20.8%** |
| Data integrity | PAX = P2P + transfer + transit, and load factor = PAX / seats, hold on all 117,608 rows |
| Reproducibility | Deterministic seeds; pinned assertions fail loudly if any published number moves; a fresh process reproduces the test window from the saved model |

## If shortlisted: a 15-minute live demonstration

1. **Trust first.** Open the Validation page: predicted vs actual on the held-out window, national total within 0.14%.
2. **The judges' question.** Take a route question from the panel and answer it on the New route or Winter flight cut page in under a minute.
3. **Explain the answer.** The monthly chart, the funnel showing where the seats go, and, in the scenario lab, contribution bars and the sensitivity tornado.
4. **Same question, plain language.** Ask the chatbot; show its tool calls and the identical number.
5. **Limits, unprompted.** Show the market-attribution warning and explain what data would remove it.

## The 1–2 minute submission video

| Time | Shot |
| --- | --- |
| 0:00–0:15 | The problem: two plans, no link; the five questions |
| 0:15–0:35 | The chain and the hybrid, with the held-out accuracy result |
| 0:35–1:15 | Live: add a twice-weekly route → +35k guest-nights; cut winter flights → −49k |
| 1:15–1:35 | The chatbot answers the same question in plain language |
| 1:35–1:50 | Impact and next step |

## Pilot success criteria

- A planner builds a credible scenario in **under 3 minutes** with no training.
- **Back-test:** re-run a past route change and compare the prediction with what happened.
- **Comprehension:** hotel planners read market views correctly given their labels.
- **Grounding:** the chatbot never states a figure that its tools did not return, even under pressure.

<!-- PAGE BREAK -->

# 9. Visuals, prototype and supporting materials

## The simulator: New route page (mock-up of the live screen)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ✈ Overview   │ New route                                                 │
│ THE THREE    │ “How many extra hotel nights would a new twice-weekly     │
│ QUESTIONS    │   flight from a given country generate?”                  │
│ ▸ New route  │ ┌────────────────────────────────────────────────────────┐│
│   Winter cut │ │ Origin: United Kingdom  Flights/wk: 2  Aircraft: 300   ││
│   Guests per │ │ First month: 2026-11    Months operated: 12            ││
│   seat       │ └────────────────────────────────────────────────────────┘│
│ EXPLORE      │  Extra guest-nights  Extra arrivals  Per year  Per seat   │
│   Scenario   │     +34,831            +8,178        +34,831    1.11      │
│   lab        │                                                           │
│   Ask the    │  Extra guest-nights by month    Where the added seats go  │
│   model      │  ▃▃▅▇█▇▅▃▃▃▄▅                   Seats       31,286 ████   │
│ EVIDENCE     │                                 Passengers  27,891 ███    │
│   Validation │                                 Terminating  8,778 █      │
│              │                                 Arrivals     8,178 █      │
└──────────────┴───────────────────────────────────────────────────────────┘
```

## The chatbot (mock-up of the live screen)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Ask the model                                                            │
│                                                                          │
│ 👤 What if Etihad runs 7 fewer London flights a week this winter?        │
│                                                                          │
│ 🤖 About −49,000 guest-nights over Nov–Feb (≈ −11,000 hotel arrivals),   │
│    roughly 1.4% of national winter demand, or ≈ −0.7 occupancy points    │
│    assuming 34,000 rooms and 1.7 guests per room.                        │
│    ▁▂▂▁  Δ guest-nights by month                                         │
│    ▸ 1 model call: change_frequency(market="UNITED KINGDOM",             │
│        city="London", airline="Etihad", delta=−7, 2026-11 → 2027-02)     │
└──────────────────────────────────────────────────────────────────────────┘
```

## What each screen shows

| Page | What a planner sees |
| --- | --- |
| **Overview** | Headline accuracy; a Sankey diagram following a year of seats to hotel arrivals; live answers to the three questions |
| **New route** | Any country, or a new one priced on an analog; flights, aircraft and dates; guest-nights, arrivals and yield per seat; a ranking of all 33 origins for the same service |
| **Winter flight cut** | Pick a route and the flights cut; guest-nights lost, share of the season's demand, baseline vs scenario, and an indicative occupancy change |
| **Guests per seat** | Ranking by hotel arrivals or guest-nights per 1,000 seats; conversion vs length of stay; month-by-market heatmap |
| **Scenario lab** | Network levers, route changes and seasonal mix shifts combined; contribution by change; sensitivity three ways (equal ±10% shock, measured real-world swings, Sobol variance shares); save, compare and download scenarios |
| **Ask the model** | Plain-language questions, an inline chart per answer, and every model call shown |
| **Validation** | Hybrid vs fixed-ratio chain vs actual on the held-out window, by month and by market-month; what the model can and cannot say |

## Supporting materials

| Material | Location |
| --- | --- |
| Methods, results, decision log, limitations | `docs/METHODS.md` |
| End-to-end notebook (chain + hybrid, every published number) | `notebooks/hybrid_layer.py` |
| App and chatbot guide, deployment notes | `app/README.md` |
| Setup and project map | `README.md` |
| Organisers' brief | `docs/competition-brief.md` |

<!-- PAGE BREAK -->

# 10. Thank you

## Make flight and hotel planning one conversation

**ATRC Flight-to-Hotel Demand Simulator**  
Transparent chain · Learned conversion · Held-out validation · Interactive scenarios · Plain-language access

### Our ask

Pilot the simulator with one DCT planning team on a real route or seasonal decision. Pair it with residence data from the airport or the airlines, to unlock trustworthy market-level answers on top of a model that already gets the national total right to within a fraction of a percent.

### Key numbers at a glance

| Metric | Value |
| --- | ---: |
| Held-out new-arrivals WMAPE (hybrid) | **20.55%** |
| Same, parametric chain | 32.77% |
| National-total bias | **−0.14%** |
| Five-seed WMAPE range | 20.30–20.85% |
| Guest-nights WMAPE (holdout) | 20.8% |
| Seat elasticity of guest-nights | 0.58 |
| Training panel | 1,405 market-months · 33 markets · 43 months |
| Flight records audited | 117,608 |
| Scenario horizon | through Dec 2028 |
