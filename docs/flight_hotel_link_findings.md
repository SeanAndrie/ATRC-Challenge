# Can airport data predict Abu Dhabi hotel demand?

**A negative result, with evidence.**

Prepared 2026-09-30. Reproduce every number below with:

```
.venv/bin/python pipeline/tests/test_market_link.py
.venv/bin/python pipeline/tests/test_aggregate_link.py
.venv/bin/python pipeline/aviation/build_annual_arrivals.py
```

## Summary

The original plan was a simulator converting AUH flight supply into Abu Dhabi
hotel guests, calibrated on DCT monthly data. Three independent tests say the
premise does not hold on available public data:

| test | sample | result |
| --- | --- | --- |
| Market mix | 2015-2023 DCT vs 2025 ADS-B | correlation carried entirely by one market; drops to -0.04 without India |
| Monthly aggregate | 46 months, 2015-2018 | elasticity 0.170 (t 0.97); arrivals add nothing to a seasonal+trend model |
| Annual aggregate | 2015-2024 | 2015-19 arrivals -8.6% while guests +25.1%; conversion ratio swings 0.355 to 1.303 |

The mechanism is the same in all three: total arrivals at AUH is dominated by
UAE-resident and GCC traffic that never occupies a hotel room, while the
non-resident leisure and business segment that does use hotels is small and
grows on its own schedule. Aircraft movements and hotel guest nights are moved
by different forces, and the flight series carries little or no information
about the hotel series once seasonality is accounted for.

Compounding this, the input series needed to fix the problem does not exist:
**monthly passenger counts after 2018 are not published by SCAD**, and the
monthly non-resident split is not published at any frequency. Details in
`docs/decisions/source_availability.md`.

## What was tested

### 1. Does the origin mix of arriving seats predict the origin mix of hotel guests?

This is the load-bearing assumption behind allocating guests by market from
flight data. Seat-weighted 2025 ADS-B arrivals, UAE domestic legs removed,
against DCT's printed guest shares for markets DCT publishes in at least six
months of some year.

Both shares are renormalised over the same nine markets, the ones measured on
both sides. The air side carries a long tail of 52 origins DCT never reports as
a hotel market; leaving those in one denominator and not the other would make a
total-variation distance between the two meaningless. Jordan is the mirror
case, a hotel market with no ADS-B row.

| DCT year | markets | Pearson | Spearman | top-5 overlap | TVD pp |
| --- | --- | --- | --- | --- | --- |
| 2015 | 9 | 0.728 | 0.433 | 3 | 40.3 |
| 2016 | 9 | 0.862 | 0.633 | 4 | 39.8 |
| 2022 | 7 | 0.920 | 0.357 | 3 | 37.5 |
| 2023 | 5 | 0.959 | 0.500 | 2 | 29.2 |

2018 (3 markets) and 2021 (2 markets) are left with too little common support
to report a correlation and are recorded rather than shown: a correlation on
three or four points is arithmetically defined and analytically worthless.

Those look encouraging. They are not robust:

| DCT year | markets, India removed | Pearson | Spearman |
| --- | --- | --- | --- |
| 2015 | 8 | 0.649 | 0.214 |
| 2016 | 8 | 0.443 | 0.476 |
| 2022 | 6 | **-0.041** | **-0.029** |

Per-market conversion, hotel share over air seat share, DCT 2015:

| market | hotel share % | air seat share % | ratio |
| --- | --- | --- | --- |
| United Kingdom | 4.75 | 14.63 | 0.32 |
| India | 4.46 | 48.09 | **0.09** |
| China | 2.56 | 0.77 | **3.32** |
| Philippines | 2.39 | 3.49 | 0.68 |
| United States | 2.38 | 10.65 | 0.22 |
| Germany | 2.28 | 9.62 | 0.24 |
| Saudi Arabia | 2.17 | 6.67 | 0.33 |
| Egypt | 2.09 | 1.19 | 1.75 |
| Pakistan | 1.75 | 4.89 | 0.36 |

A **36x spread** across nine markets. India is the largest air market by seats
and a minor hotel market; China is the reverse. This is the expected signature
of visiting-friends-and-relatives travel, which stays in homes, against leisure
and business travel, which uses hotels. A blended conversion would misallocate
guests by an order of magnitude in both directions.

### 2. Does aggregate monthly arrival volume predict aggregate monthly guests?

46 matched months (2015-2018, less the two Februaries lost to a misfiled source
file).

| measure | value |
| --- | --- |
| Pearson of levels | 0.099 |
| Pearson of year-on-year log growth (n=34) | 0.441 |
| Spearman of same | 0.324 |
| Elasticity, log-log with month effects and trend | 0.170 (se 0.176, t 0.97) |

The elasticity is not distinguishable from zero. The r-squared of 0.874 comes
entirely from the month dummies and trend.

The holdout comparison settles it. Leave-one-year-out WMAPE:

| held out | seasonal naive | **no arrivals** | arrivals only | full model |
| --- | --- | --- | --- | --- |
| 2015 | n/a | 4.1% | 14.9% | 3.7% |
| 2016 | n/a | 5.9% | 7.7% | 6.5% |
| 2017 | 9.9% | 4.4% | 11.1% | 4.8% |
| 2018 | 5.6% | 6.4% | 16.5% | 7.3% |
| **mean** | 7.8% | **5.2%** | 12.6% | 5.6% |

A model that never sees arrivals beats the model that includes them. Arrivals
alone are much worse than either.

Guests per arriving passenger also drifts hard, 0.356 in 2015 to 0.477 in 2018,
a 34% rise in three years, so no fixed conversion exists to calibrate.

### 3. Does it hold over a longer horizon?

Annual arrivals, 2015-2024 except 2020, from the Statistical Yearbook of Abu
Dhabi 2020 table 2.7.7 and Key Transport Statistics 2025 table 2.

| year | arrivals | hotel guests | guests per arrival |
| --- | --- | --- | --- |
| 2015 | 11,564,563 | 4,105,846 | 0.355 |
| 2016 | 12,139,014 | 4,440,314 | 0.366 |
| 2017 | 11,678,045 | 4,874,253 | 0.417 |
| 2018 | 10,633,563 | 5,104,753 | 0.480 |
| 2019 | 10,564,367 | 5,135,000 | 0.486 |
| 2021 | 2,515,116 | 3,277,977 | 1.303 |
| 2022 | 7,821,770 | 4,065,000 | 0.520 |
| 2023 | 11,117,530 | 5,504,000 | 0.495 |
| 2024 | 14,522,288 | 5,819,000 | 0.401 |

Between 2015 and 2019 air arrivals fell **8.6%** while hotel guests rose
**25.1%**. Pre-COVID growth correlation is 0.135, Spearman 0.000. From 2022 to
2024 arrivals rose 85.7% and guests 43.1%, and the ratio has turned downward.

The sharpest single case: **2018 to 2019, arrivals -0.7%, hotel guests +0.6%.**
An arrivals-keyed converter predicts the wrong sign of change.

A useful side effect: the Yearbook independently reproduces the 2016, 2017 and
2018 figures already transcribed by hand into `pax_movements.csv`, to the
passenger, which validates that earlier work.

## Why it fails

Total AUH arrivals is the wrong denominator. It mixes at least three populations
with very different relationships to hotels:

* **UAE residents** flying domestically and to the GCC, who are a large share of
  the traffic and mostly not seeking accommodation.
* **GCC visitors**, often staying in homes or visiting relatives, similarly
  over-represented in seat counts relative to hotel nights.
* **Non-resident leisure and business travellers**, the segment that actually
  fills rooms, and a minority of seats.

Because the segments have different shares and different growth rates, the
aggregate can move opposite to the hotel series. 2015-2019 is exactly that: a
declining seat count and a rising hotel count simultaneously.

The market-mix failure is the same mechanism in cross-section. India's share of
seats is inflated relative to its share of rooms, and China's is deflated, and
those distortions are large, persistent, and behavioural rather than
measurement error.

Plausible actual drivers, none available monthly from official sources: hotel
supply additions, events and conferences, hotel pricing, and source-market
income in the specific long-haul leisure markets that do stay in hotels.

## Data availability

| publication | monthly movements | monthly passengers | annual passengers | resident split |
| --- | --- | --- | --- | --- |
| Key Transport Statistics 2025 | yes | **no** | yes | **no** |
| Statistical Yearbook 2020 | yes | **no** | yes | **no** |
| Transport Statistics 2018 | yes | yes, 2015-2018 | yes | no |

The full SCAD publications library was enumerated rather than searched, 1,850
unique documents across all 22 pages of the download portlet. The detailed
*Transport Statistics* edition that carried monthly passengers in table 1.2
**ends at 2018**. The region breakdowns are geographic, not by residency.

2020 is absent from both usable publications: the Yearbook stops at 2019 and KTS
begins at 2021.

## What would make this work

In rough order of value:

1. **Monthly non-resident arrivals.** The single variable that would fix the
   mechanism. Not published by SCAD. Possible routes are airline schedules
   filtered to non-resident routes, or a residency-aware origin split from
   ADS-B that excludes the UAE, which `adsb_2025` partly provides but cannot
   currently distinguish residents from GCC visitors.
2. **Monthly hotel supply and events.** A plausible confounder that no test
   here controls for. It could absorb apparent arrivals effects.
3. **A longer monthly passenger series.** Would allow the structural-break and
   recovery tests that four pre-COVID years cannot support. Blocked on (1)'s
   publication gap.

## Recommendation

Do not build the flight-to-hotel simulator. If a model is wanted in this
notebook, the honest version predicts hotel guests from DCT's own signal
(seasonality, trend, events, supply) and uses ADS-B only as an independent
cross-check on whether flights corroborate the narrative, never as a driver.
The 5.2% WMAPE seasonal-plus-trend baseline is the benchmark to beat.

An alternative use of the same data is to change the question: rather than
predict what DCT already reports, forecast the gaps. 2020 nationality, the 2017
nationality table, and a 2025 nowcast are all places where flight data could
serve as a genuine input.

## Caveats

* The monthly tests cover 2015-2018 only, so no structural-break test is
  possible at monthly frequency. The annual test spans COVID and the recovery
  but rests on few pre-COVID growth observations.
* The annual 2018 guest total depends on interpolating the lost February.
* 2023 and 2024 use published annual figures because their monthly series is
  incomplete; 2023's published figure is the restatement carried in the 2024
  report, which disagrees with the sum of the 2023 monthly reports.
* The market-mix test compares 2015-2023 hotel shares against 2025 airborne
  shares. India in particular is not comparable across that span, since
  India-UAE capacity grew sharply after bilateral liberalisation, so part of
  the India-driven correlation is a genuine structural change.
* Market shares are renormalised over the nine markets measured on both sides.
  52 air-only origins and hotel-only Jordan are excluded from every statistic,
  which raises the correlations relative to the unnormalised version but makes
  the TVD and the conversion ratios meaningful. The renormalised TVD of 29-40pp
  is the honest distance between the two distributions; the earlier unnormalised
  figure of 12-16pp was artificially *deflated*, because the unmatched air-side
  mass was dropped from the comparison rather than counted as disagreement, and
  it is not used.
* ADS-B market shares inherit one-day-per-month sampling error. 39 of 62 markets
  have fewer than 20 observed legs and 42 are missing at least one sampled day.
  Direction survives this because the markets carrying the signal are the
  well-measured ones; the magnitude of the conversion-ratio spread does not, and
  it is not quoted as precise.
* Seat weights are typical-capacity estimates, not operator actuals.
* Samples are small: 5-9 markets per year, because DCT nationality coverage is
  thin in 2017, 2018 and 2021.
