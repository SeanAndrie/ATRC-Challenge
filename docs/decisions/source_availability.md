# AUH passenger data availability, and what it means for the simulator

Compiled 2026-09-30 while attempting to source monthly arrivals for 2019-2025.
Sources: `data/raw/scad/`. Reproduce with
`pipeline/aviation/build_annual_arrivals.py`.

## What was wanted

Monthly arriving passengers at Zayed International Airport, by month, for
2019-2025, so that arrivals could be tested as a driver of monthly hotel guests
over a period including COVID and the recovery.

## What the official sources actually contain

| Publication | Monthly movements | Monthly passengers | Annual passengers | Resident/non-resident |
| --- | --- | --- | --- | --- |
| Key Transport Statistics 2025 (table 1, 2) | yes, 2021-2025 | **no** | yes, 2021-2025 | **no** |
| Statistical Yearbook 2020 (table 2.7.6, 2.7.7) | yes, 2016-2019 | **no** | yes, 2016-2019 | **no** |
| Transport Statistics 2018 (table 1.2) | yes | yes, 2015-2018 | yes | no |

The monthly passenger series behind the existing 2015-2018 data came from the
detailed *Transport Statistics* annual XLSX. That publication ends at 2018.

## Exhaustive sweep of the SCAD publications library

Rather than guess at document URLs, the whole library was enumerated. The
`download-publications` portlet is server-rendered and paginates at 150 per
page, so every page was fetched and every document link extracted:

```
for s in $(seq 0 21); do
  curl -sSL "https://www.scad.gov.ae/download-publications?delta=150&start=$s"
done | grep -oE 'href="[^"]*documents/20122[^"]*"' | sed 's/?t=.*//' | sort -u
```

1,850 unique documents. Every transport-related file in the entire library:

| file | note |
| --- | --- |
| Statistics of Air Water Land transport 2013 en v1.pdf | |
| Transportation - AD Emirate - 2014 - EN.pdf | |
| Transport 2014-2015.xlsx | |
| Transport and Storage Statistics IA - en.pdf | |
| TransportIA.xlsx | |
| Economic Survey Result - Transport and Storage 2018 | |
| Transport Statistics 2018 Annual Yearly (en, both_v5) | last edition with monthly passengers |
| Key Transport Statistics 2025 (xlsx) | obtained separately; annual passengers only |

**No Transport Statistics edition after 2018 exists in the library.** Monthly
passengers for 2019 onward are therefore not available from SCAD, and this is a
settled finding rather than an unresolved search.

Two routes were tried and failed, both recorded so they are not repeated:

* The Liferay OAuth token endpoint redirects to an HTML interstitial, so
  client-credentials guest auth is disabled. The headless-delivery API returns
  403 for guests.
* A "Transport Statistics-Annually 2024" page exists at
  `https://scad.gov.ae/w/transport-statistics-annually-2024` but carries no
  attached document: no `/documents/` link of any group, and its only file-type
  strings come from Liferay's client-side extension-point template. It appears
  to be a stub, though one manual browser check is the one thing that could
  still overturn this.

Region splits in these publications are geographic (Indian Subcontinent, Western
Europe, GCC, and so on), not by residency. So the series that would most plausibly
explain hotel demand, non-resident arrivals, is not available from SCAD at all.

2020 is missing from both usable publications: the Yearbook stops at 2019 and
KTS starts at 2021.

## What was obtained instead

A complete annual arrivals series, 2015-2024 except 2020:

| year | arrivals | source |
| --- | --- | --- |
| 2015 | 11,564,563 | local `pax_movements.csv` |
| 2016 | 12,139,014 | Yearbook 2.7.7 |
| 2017 | 11,678,045 | Yearbook 2.7.7 |
| 2018 | 10,633,563 | Yearbook 2.7.7 |
| 2019 | 10,564,367 | Yearbook 2.7.7 |
| 2021 | 2,515,116 | KTS table 2 |
| 2022 | 7,821,770 | KTS table 2 |
| 2023 | 11,117,530 | KTS table 2 |
| 2024 | 14,522,288 | KTS table 2 |

The Yearbook independently reproduces the 2016, 2017 and 2018 figures already in
`pax_movements.csv` to the passenger, which validates the earlier hand
transcription.

## Why this does not rescue the simulator

Two independent tests, both negative.

**Monthly, 46 matched months (2015-2018).** Log-log regression of guests on
arrivals with month effects and trend gives an elasticity of 0.170, se 0.176,
t 0.97. Leave-one-year-out WMAPE is 5.2% for a model that never sees arrivals,
5.6% for one that does, and 12.6% for arrivals alone. See
`dct/aggregate_link_test.md`.

**Annual, 2015-2024.** Hotel guests per arriving passenger is not remotely
stable:

| year | arrivals | guests | guests per arrival |
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

Between 2015 and 2019 air arrivals fell 8.6% while hotel guests rose 25.1%. The
conversion ratio climbed 37% over those four years. Pre-COVID growth
correlation is 0.135, Spearman 0.000. From 2022 to 2024 arrivals rose 85.7% and
guests rose 43.1%, and the ratio is now falling rather than rising.

The clearest single case: 2018 to 2019, arrivals -0.7% and hotel guests +0.6%.
An arrivals-keyed converter predicts the wrong sign.

## Interpretation

Total arrivals is a poor driver of Abu Dhabi hotel demand because it is
dominated by UAE-resident and GCC traffic that does not use hotels, while the
non-resident leisure and business segment that does use hotels is small relative
to it and grows on its own schedule. This is the same mechanism that broke the
market-mix link, now visible in aggregate: the aircraft movements and the hotel
guests are being moved by different forces.

Hotel supply, events, and the resident/non-resident mix are the plausible
drivers. None is available monthly from the official sources.

## Caveats

* The annual 2018 guest total depends on interpolating the lost February.
* 2023 and 2024 use published annual figures because their monthly series is
  incomplete. 2023's published figure is the restatement carried in the 2024
  report, which does not agree with the sum of the 2023 monthly reports.
* The annual growth correlation rests on four pre-COVID observations and is
  indicative only.
* Monthly evidence covers 2015-2018 only, so no structural-break test is
  possible at monthly frequency.
