# Submission pack — deck, demo and playbook

This is the short, judge-facing layer on top of [`METHODS.md`](METHODS.md). It
gives the 10-page deck structure, a 90-second demo script, the system diagram,
and the one-page planner playbook. The numbers are backed by automated checks
and recomputed live in the app's **Validation** tab.

## The one-sentence pitch

**A defect-aware decision simulator: it links flights to hotel demand, tells a
planner what is measured versus assumed, and shows how wide the answer is
before a budget is committed.**

## 10-page deck outline

| Page | Content | Key number / visual |
| ---: | --- | --- |
| 1 | Problem: flights and hotels are planned separately; the link is indirect and country definitions differ | Brief's three questions |
| 2 | The data catch: departure country ≠ guest nationality; capture > 1.0 in five markets | CHINA 5.854, CANADA 2.288, USA 1.990, RUSSIA 1.527, FRANCE 1.220; ~903k guests |
| 3 | System design: two layers, one panel, one app | Mermaid diagram below |
| 4 | Simple formula chain: seats → PAX → P2P → arrivals → guest-nights; measured vs estimated vs assumed | Global capture 0.391; ALOS 3.602 |
| 5 | Hybrid layer: LightGBM learns the P2P → arrivals step; learned ALOS | Features, shallow model, 5-seed ensemble |
| 6 | Validation: holdouts and final test window, baselines, aggregate bias | Final test window 20.6% weighted error, −0.1% bias vs simple formula 32.8%; last-year holdout guest-nights 20.8% |
| 7 | What moves the number: a-priori vs empirical Sobol | Capture dominates; seats vs P2P runner-up |
| 8 | Simulator demo: levers, seasonal mix, route editor, impact window, market range | Screenshots / live demo |
| 9 | Decision playbook: what a DCT planner does differently | Capture, attribution range, data request |
| 10 | Limits, reproducibility and next data | The final test window has no `Guests`; 12 hotel-only / 52 air-only markets; residence/nationality request |

## 90-second demo script

**0–10s — the problem.** "Flights and hotels are planned by separate teams. We
built a simulator that connects them, and — just as important — tells you which
numbers are trustworthy."

**10–25s — the data catch.** "Flight country is where the plane departed; hotel
country is the guest's nationality. They are not the same. Capture exceeds one
in five markets, which is impossible. That is why the national total is the
validated number and the market split is shown as a range."

**25–55s — live demo.** "Here is the baseline. I add a twice-weekly route from an
existing market, change the seasonal mix, and switch the impact window to the
first 12 months after the change. Guest nights move by X%. The waterfall shows
every step: seats, passengers, terminating passengers, arrivals, guest nights.
The market tab shows the same total split three ways, so you can see how much the
market ranking depends on the join assumption."

**55–75s — validation.** "We tested on a last-year holdout and an unseen final
test window. The weighted error by market and month is about 20.6% against a
32.8% simple formula baseline, with overall bias −0.1%. The monthly aggregate is
closer still. Every figure is recomputed live here."

**75–90s — decision.** "So what? Debate capture — how many arriving passengers
become hotel guests — not just seats. Use the aggregate range for budgets. Use
the market range to start a residence-side data request, not to set a market
target."

## System diagram

```mermaid
flowchart LR
    A[Flight schedule<br/>seats, PAX, P2P, transfer/transit] --> C[Matched market-month panel<br/>1,405 rows · 33 markets]
    B[Hotel performance<br/>guests, arrivals, ALOS] --> C
    C --> D[Simple formula chain<br/>capture[m] · ALOS[m] · season]
    C --> E[Hybrid LightGBM<br/>P2P → arrivals + learned ALOS]
    D --> F[Scenario engine<br/>Scenario / Simulator]
    E --> F
    F --> G[Streamlit app<br/>levers · route edits · seasonal mix<br/>market range · validation]
    H[Attribution guard<br/>last-departure ≠ nationality] --> G
    I[Validation<br/>~20.6% · ~20.8%] --> G
```

**Guardrails.** Aggregate views are the default. Market rows carry
`data_status = supply-side-only`. Closed markets are structural zeros. The
horizon stops at 2026-02. A scenario cannot invent a market or zero a global
supply lever.

## One-page planner playbook

**What to do differently**

1. **Allocate on capture, not seats.** Sensitivity shows capture is the largest
   driver of guest nights. A route only matters through the share of its seats
   that becomes hotel guests.
2. **Use the aggregate range for budgets.** The typical weighted error is ~20.6%
   on New Arrivals in the final test window; a scenario delta smaller than that
   is within model noise.
3. **Use the market range for conversations, not targets.** The market tab shows
   a last-departure point plus a demand-mix/supply-mix range. A wide range means
   the market level is mostly assumption.
4. **Measure the right window.** A route starting in 2025-09 should be judged on
   its first 12 months, not on a 44-month total.
5. **Ask for the missing join key.** Residence/nationality on the flight side
   would turn the market range into a measurement. That is the single highest
   value data request.

**Three numbers to remember**

| Number | Meaning |
| --- | --- |
| **20.6%** | Final test window: New Arrivals weighted error, overall bias −0.1% |
| **20.8%** | Last-year holdout: guest-nights weighted error with learned ALOS |
| **5.854** | CHINA capture under the last-departure join — impossible, and the clearest proof of the defect |

**Three honest limits**

- The final test window has no `Guests` column, so guest-nights are validated in-sample and in the last-year holdout.
- 12 hotel-only markets have no flight supply; 52 air-only origins have no
  measured hotel capture.
- Purpose-of-travel / VFR is unmeasured; capture is a calibration coefficient,
  not a conversion probability.
