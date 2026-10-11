"""Static prose for the Streamlit front end, kept as data so it is easy to audit.

Everything a planner reads here is deliberately blunt about what the numbers are
and are not. The market-attribution and VFR caveats are not boilerplate: success
criteria 01/05/06 depend on granular output, and the pack cannot support market
attribution (``docs/METHODS.md`` section 4.5).
"""

TITLE = "Flight → Hotel Demand Simulator"
SUBTITLE = (
    "Adjust flight and market factors and see Abu Dhabi hotel guest numbers "
    "update. Seats → passengers → visitors → guest nights, every assumption "
    "adjustable."
)

MARKET_WARNING = (
    "**Supply-side-only — not measured.** Flight ``country`` is the departure "
    "country; hotel ``country`` is guest nationality. The two are not the same, "
    "and capture exceeds 1.0 in five markets, so **market-level numbers below "
    "are filings, not measurements**. Only the national total is validated. Do "
    "not quote a single market's level."
)

CHAIN_CAPTION = (
    "_The parametric chain applies the six global dials. Per-market seat "
    "multipliers and the learned capture/ALOS live in the hybrid layer, which "
    "is what the headline numbers use._"
)

ASSUMPTIONS = """
### What this simulator is

A transparent conversion chain — **scheduled seats → arriving passengers (load
factor) → inbound visitors (excluding transfer/transit) → hotel guest nights
(capture rate × average length of stay)** — combined with a shallow LightGBM
layer that learns seasonality and source-market effects. You move the levers;
both layers re-score and the headline numbers update.

### Ground rules and limits

- **All data is aggregated.** No personal or passenger-level records exist here,
  and none are used.
- **The test window has no ``Guests`` column.** Guest nights are validated
  in-sample and on the leave-tail holdout; New Arrivals are validated
  out-of-sample on the competition window.
- **Capture is behavioural, not mechanical.** ``New Arrivals / P2P`` absorbs
  visiting-friends-and-relatives travel, booking behaviour and non-air hotel
  demand. It is a calibration coefficient, not a conversion probability.
- **Purpose-of-travel is unmeasured.** No source publishes it at any frequency,
  so the visitor split inside capture is an assumption, never an estimate.
- **VFR share is unmeasured and no public source exists.** Any
  passenger-to-visitor conversion is therefore an explicit assumption.
- **12 hotel-only markets** (Australia, Brazil, Czechia, Denmark, Finland,
  Mexico, Morocco, Norway, Pakistan, Romania, South Africa, Sweden) have no
  flight supply and are carried as a separately reported residual.
- **Source-market public holidays are out of scope.** The UAE-wide event
  calendar is baked into the features; the ``season`` dial is the seasonality
  control.

### The market-join caveat (read before quoting any market number)

The ``market`` key joins flight **departure country** to hotel guest
**nationality**. That equality is false, and arithmetically falsified: capture
(``New Arrivals / P2P``) exceeds 1.0 in five markets (CHINA 5.854, CANADA 2.288,
USA 1.990, RUSSIA 1.527, FRANCE 1.220) — more guests than terminating
passengers, which is impossible. Connecting passengers are filed under the hub
(India → Doha → AUH books as QATAR supply, INDIA demand).

- **Aggregate results are valid.** The error is zero-sum across hub and spoke
  markets and cancels in the national total.
- **Market-level attribution is not valid.** Per-market output, the market
  breakdown and the ``seats_mult`` lever are filed to the wrong market.
  Success criteria 01/05/06 depend on these, so they are shown only with this
  warning attached.

### What a DCT Abu Dhabi planner should do differently

1. **Allocate on capture, not on seats.** Sensitivity shows capture is the
   lever that moves guest nights most. Route and frequency decisions only
   matter through how many of those seats become hotel guests.
2. **Use the market view to start conversations, not to set budgets.** The
   ranking of markets by guest nights is informative at the top; the levels are
   not. Confirm any single-market conclusion with a residence-side data request.
3. **Plan on the aggregate range, not a point estimate.** The validated error is
   ~20.6% WMAPE on New Arrivals for the competition window — treat that as the
   honest spread and pair it with the levers' empirical ranges.
4. **Treat a new route's payoff as a range.** The parametric chain and the ML
   layer bracket it; where they disagree, widen the range rather than pick one.

### Validation with an honest error metric

- **Competition window (2025-08 → 2026-02), New Arrivals:** 20.6% WMAPE,
  aggregate bias −0.1% (seasonal-naive 32.8%).
- **Leave-tail holdout, guest nights:** ~20.8% WMAPE with a learned ALOS.
- All figures are replayed live in the **Validation** tab and pinned by the
  test suite, not asserted.
"""

DECISION_CALLOUT = (
    "**Decision relevance:** capture is the allocation lever. A planner should "
    "debate how many arriving passengers become guests — not just how many "
    "seats arrive — because that is what moves guest nights most."
)
