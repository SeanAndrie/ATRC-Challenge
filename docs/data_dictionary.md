# Data dictionary

Every table the pipeline publishes, what it means, where it came from, and what
is wrong with it.

This file was **reconstructed** after a repository reorganisation. See
[Provenance of this document](#provenance-of-this-document) at the end for
exactly what was lost and how the gaps were filled.

Regenerate everything with:

```
.venv/bin/python pipeline/dct/dct_parse.py
.venv/bin/python pipeline/hotel_supply.py
.venv/bin/python pipeline/aviation/build_annual_arrivals.py
.venv/bin/python pipeline/aviation/build_adsb_2025.py
.venv/bin/python pipeline/aviation/calibrate_adsb_2025.py
```

All paths in code resolve through `pipeline/paths.py`. No module hardcodes an
absolute path.

---

## Data tiers

| Tier | Location | Tracked | Holds |
|---|---|---|---|
| raw | `data/raw/` | no (gitignored) | downloaded PDFs, workbooks, ADS-B snapshots, OpenSky dumps, OurAirports |
| interim | `data/interim/` | yes | extracted PDF text plus a fingerprint manifest |
| processed | `data/processed/` | yes | the 14 analysis-ready tables below |

Raw is gitignored because it is large and re-downloadable. Everything
downstream of raw is tracked, so the analysis can be reproduced from a clean
clone without re-fetching anything. `data/raw/opensky/` in particular is a
quota-limited pull that cannot be regenerated on demand.

Current raw contents:

| Directory | Files | Notes |
|---|---|---|
| `data/raw/dct_pdfs/` | 101 | 99 DCT reports plus sidecars |
| `data/raw/scad/` | 2 | SCAD transport workbooks |
| `data/raw/adsb_snapshots/` | 12 | one per month of 2025 |
| `data/raw/opensky/` | 38 | 1 CSV (6,535 rows) + 37 Parquet |
| `data/raw/reference/` | 1 | OurAirports |

`data/interim/dct_text/` holds 99 extracted text files plus a fingerprint
manifest.

---

## Source provenance

**Abu Dhabi DCT (Department of Culture and Tourism)** — monthly and annual
hotel performance reports, 2015 through 2024. Parsed by `pipeline/dct/dct_parse.py`.

**SCAD (Statistics Centre of Abu Dhabi)** — transport statistics and
statistical yearbooks, used for passenger arrivals and movements:

- `https://scad.gov.ae/documents/20122/0/Publication_ar_PID_344_2025_Annual_Yearly_ar.xlsx/40063176-6ed9-e21c-8d81-a1804ac8a445`
- `https://scad.gov.ae/documents/20122/1043412/Statistical%20Yearbook%20of%20Abu%20Dhabi_2020_Annual_Yearly_en.pdf/68b923b8-e4f2-3fe5-f922-11f60f9ddff4`
- `https://scad.gov.ae/documents/20122/2311280/Transport%2520Statistics_2018_Annual_Yearly_en.xlsx/e189ae42-2214-35a7-cfe7-badc54fc4d21`

Exact per-row provenance is preserved in the `source_url` and `source_name`
columns of `pax_movements.csv` and `etihad_passengers.csv`.

**Abu Dhabi Airports / OMAA OpenSky** — ADS-B exchange snapshots for the 12
months of 2025. Built by `pipeline/aviation/build_adsb_2025.py`.

**Etihad** — published monthly, half-yearly and annual passenger figures,
2020-01 through 2026-02.

---

## Unit contract

`dct_kpis.csv` carries a `unit` column and **every consumer must respect it**.
The publisher is not consistent, so units legitimately vary by year.

| `unit` | Meaning | Where it appears |
|---|---|---|
| `count` | absolute headcount | guest counts 2015-01 to 2021-03, 2015-2016 revenue, derived totals |
| `thousands` | thousands | guest counts from 2021-04, guests and revenue 2022-2023 |
| `aed` | raw AED, no scaling | revenue 2015-2016 |
| `aed_million` | millions of AED | revenue from 2019 onward |
| `pct` | percent, 0-100 | occupancy |
| `days` | nights per stay | ALOS |

Two conventions are deliberate:

1. **Revenue is canonically `aed_million`, but 2015-2016 stay `aed`** exactly as
   printed. Those reports publish raw AED (for example `6,620,373,000`), and
   rewriting them would mean inventing a precision the source never had.
2. **Guest counts change unit mid-series**, from `count` to `thousands` at
   2021-04. Both labels are correct for what each report printed. Summing across
   the boundary without normalising produces differences of three orders of
   magnitude that look like catastrophic data loss.
   `pipeline/dct/dct_validate.py` converts via its `UNIT_FACTOR` map for
   exactly this reason.

Magnitude suffixes printed in the reports are normalised into the metric's own
unit by `_value()` in `pipeline/dct/eras/era_key.py`, using the `SCALERS` map.
`K`/`KK` become thousands, `M`/`MN` become millions, `B`/`BN` become billions
before conversion. A bare token with no suffix keeps its declared unit
untouched.

---

## DCT hotel KPIs — `dct_kpis.csv`

1,211 rows. Producer: `pipeline/dct/dct_parse.py`. Source: 99 DCT reports.

Long format: one row per (report, granularity, metric).

| Column | Type | Notes |
|---|---|---|
| `source_file` | str | report stem, e.g. `2021-09`, `2024-annual-2024` |
| `year`, `period` | int | calendar year; `period` is `01`-`12`, `Q1`-`Q4`, `annual`, or the YTD month |
| `granularity` | str | `month`, `ytd`, `quarter`, `year` |
| `segment` | str | always `total` |
| `metric` | str | see below |
| `unit` | str | see unit contract |
| `value` | float | scaled per `unit` |
| `growth_pct` | float | publisher's % change vs prior period; often null |
| `ytd` | int | 1 when `granularity == 'ytd'` |
| `derived` | int | 1 when computed or estimated by the pipeline rather than printed — 68 rows, mostly bar-chart readings. See [Derived figures](#derived-figures) |
| `source_block` | str | which block of the PDF the figure came from |

Metrics: `hotel_guests`, `occupancy_rate`, `alos`, `arr`, `revpar`,
`total_revenue`, `guest_nights`, `room_nights`, `room_revenue`, `fb_revenue`,
`other_revenue`, `occupied_rooms`, `international_guests`.

Coverage by grain:

| grain | rows | years |
|---|---|---|
| month | 786 | 2015-2024 |
| ytd | 390 | 2015-2023 |
| year | 25 | 2015-2024 |
| quarter | 10 | 2019 |

Monthly rows per year: 2015 has 121, 2016 131, 2017 51, 2018 66, 2019 72,
2020 72, 2021 69, 2022 72, 2023 60, 2024 72. The low 2017 and 2018 counts
reflect the `era2017` and `duplicate` routes, not missing data — see
`dct_coverage.csv`.

`source_block` values: `chart_baseline`, `guest_weighted_regions`,
`kpi_monthly_trend`, `kpi_monthly_trend_prior`, `main_indicator_summary`,
`monthly_summary`, `narrative`, `overall_performance_table`, `region_sum`.

### Which rows the analysis actually uses

Only `granularity == 'month'` (and `granularity == 'year'` for
`international_guests`). Every consumer filters explicitly:

- `notebooks/linkage_study.py` — month for KPI series, year for international share
- `pipeline/tests/test_aggregate_link.py`, `test_market_link.py`, `test_retarget_international.py` — month

**The `ytd` grain is not consumed anywhere** and is less reliable than `month`.
See [Known defects](#known-defects-and-limitations).

---

## DCT guest nationality — `dct_guests_nationality.csv`

664 rows, 31 nationalities. Producer: `dct_parse.py`. 2015-2024.

`source_file`, `year`, `period`, `granularity`, `ytd`, `segment`, `rank`,
`nationality`, `guests`, `guests_unit` (`count` or `thousands`), `share_pct`,
`share_basis` (`kpi_guests` or `report_total`), `alos`, `change_pct`,
`source_block`.

`share_basis` matters: `report_total` divides by the publisher's stated total,
`kpi_guests` divides by the KPI series. Mixing the two silently inflates or
deflates shares.

---

## DCT hotel supply — `dct_hotel_supply.csv`

115 rows, 2017-2024. Producer: `dct_parse.py`.

`source_file`, `year`, `period`, `as_of`, `segment`, `category`, `hotels`,
`rooms`, `source_block`.

`category` includes `total` alongside star bands: `1-3 STAR`, `3 STAR`,
`3-1 STAR`, `4 STAR`, `5 STAR`, `APTS`, `DELUXE APT`. **Do not sum categories
without excluding `total`**, which is the published aggregate and will double
count.

---

## DCT provenance tables

**`dct_coverage.csv`** — 99 rows, one per report. `source_file`, `route`,
`year`, `period`, `basis`, `kpi_rows`, `nationality_rows`, `supply_rows`.
`route` is one of `main`, `key`, `era2017`, `chart`, `quarter`, `annual2024`,
`duplicate`. This is the table to read first when a report appears to be
missing: `2015-02` and `2018-02` have `kpi_rows == 0` because they are
byte-identical duplicates of a 2017-02 report, not parse failures.

**`dct_cross_checks.csv`** — 32 rows of internal reconciliation. `source_file`,
`check`, `left`, `right`, `status`. All currently pass.

**`dct_notes_raw.csv`** — 94 rows of narrative text captured per report.
`source_file`, `stage`, `note`. Machine-extracted, unedited, and **not
reliable prose**; treat as a search index over the PDFs.

---

## Passenger movements — `pax_movements.csv`

377 rows, 2014-2025. Producer: `pipeline/hotel_supply.py`.

| Column | Notes |
|---|---|
| `period_start`, `period_end` | inclusive ISO dates |
| `granularity` | `month`, `quarter` (`H1`, `Q4`), `9M`, `year` |
| `scope` | `auh` (single airport) or `five_airports` |
| `metric` | `pax_arrivals`, `pax_departures`, `pax_transit`, `pax_total`, `movements_aircraft` |
| `value` | figure as printed |
| `unit` | `passengers` or `movements` |
| `source_name`, `source_url` | full provenance per row |
| `note` | qualification, e.g. partial-year caveats |

Mixes SCAD and Abu Dhabi Airports releases. `scope` must be respected: `auh` and
`five_airports` are different denominators.

---

## Etihad passengers — `etihad_passengers.csv`

103 rows, 2020-01 through 2026-02. Producer: `pipeline/hotel_supply.py`.

`period_start`, `period_end`, `granularity` (`month`, `YTD`, `year`), `metric`
(`pax_number`, `load_factor_pct`), `value`, `source_name`, `source_url`, `note`.

Airline-level, not emirate-level. Useful as an activity indicator, not as a
proxy for total hotel demand.

---

## Annual arrivals — `pax_arrivals_annual.csv`

9 rows. Producer: `pipeline/aviation/build_annual_arrivals.py`.

`year`, `arrivals`, `guests`, `imputed_months`, `guests_per_arrival`,
`arr_yoy`, `gst_yoy`, `source`.

Years present: 2015-2019, 2021-2024. **2020 is absent** — no source. 2018 has
`imputed_months == 2`, so its annual figure is not a direct publication. Check
`imputed_months` before treating any year as fully observed.

---

## ADS-B arrivals — `omaa_arrivals_2025.csv`

1,955 rows, one per arriving flight. Producer: `pipeline/aviation/build_adsb_2025.py`.

`month` (2025-01 to 2025-12), `snapshot_day`, `key`, `callsign`, `orig`,
`origin_iata`, `origin_city`, `origin_country`, `origin_known`,
`origin_patched`, `origin_kind`, `is_local_leg`, `distance_km`, `actype`, `hex`,
`status`, `std`, `arrival_utc`, `dep_date`, `equip`.

Quality, measured:

- **99.8%** of flights resolve to a known origin
- **132** rows are manually patched to an origin
- **203** rows are `is_local_leg` — and **all 203 fall in 2025-01 to 2025-03**

That local-leg cluster is a capture artifact, not real inbound demand.
`pipeline/tests/test_market_link.py` drops `is_local_leg` before computing
market shares, so published shares are unaffected. **Any new consumer of this
table must do the same** — without the filter, Q1 market shares are badly
distorted by domestic repositioning.

---

## ADS-B derived tables

| File | Rows | Producer | Notes |
|---|---|---|---|
| `monthly_summary.csv` | 12 | `build_adsb_2025.py` | `flights`, `local_legs`, `origins_known`, `unknown_origin`, `origins`, `countries`, `types` |
| `markets_by_month.csv` | 478 | `build_adsb_2025.py` | 62 origin countries across 12 months, observed combinations only (not the full 744-row cross product); `flights` excludes local legs |
| `aircraft_types.csv` | 41 | `build_adsb_2025.py` | `actype`, `flights` |
| `capture_rate.csv` | 12 | `calibrate_adsb_2025.py` | observed vs official movements |

**`capture_rate.csv` is the honesty table for everything ADS-B.** ADS-B sees
**48.9% to 61.5% of official arrivals, mean 56.4%**. Absolute ADS-B flight
counts are therefore *not* demand levels. Use it for shares, mix and direction;
never quote raw ADS-B volumes as arrivals.

---

## OpenSky — raw only, no processed table

`data/raw/opensky/` holds 1 CSV (6,535 rows, 15 columns) and 37 Parquet files
covering **36 distinct days**: 2026-06-20 to 2026-07-18, then 2026-09-07 to
2026-09-13.

`coverage_flag` is the field that matters:

| Flag | Rows | Meaning |
|---|---|---|
| `ok` | 5,963 | usable |
| `sparse` | 327 | partial day, treat counts as a floor |
| `low` | 245 | unreliable |

Nothing downstream consumes OpenSky yet. `notebooks/opensky.py` is exploratory.
The intended use is measuring the **hotel capture rate of the 52 air-only
origins**, which is currently unmeasured.

---

## Known defects and limitations

### Fixed 2026-10, recorded for traceability

Three parser defects in `pipeline/dct/eras/era_key.py` corrupted the `ytd`
grain. They were corrected; 30 rows changed, all in `ytd` plus one monthly
revenue row. No consumer reads those rows, so **no analytical finding moved** —
verified by regenerating all three test reports against pre-fix and post-fix
data and confirming byte-identical bodies.

1. **`BN` was truncated to `B`.** Neither suffix regex contained `BN`, so
   `2.7BN` was read as `2.7B` and then treated as millions.
2. **`B` was scaled as if it were `M`.** `SCALERS` defined the correct factors
   but was never used; `_value()` hardcoded a mapping that conflated the two,
   understating 2021 and 2022 YTD revenue a thousandfold. 18 rows affected.
3. **`MN` on a count metric was kept as `count`.** 2022 reports print guests as
   `4.1 MN`, which was recorded as 4.1 guests rather than 4,100 thousand. 7 rows
   affected.

Also fixed: some YTD tables print a bare comparison year in the first cell
(`2021-09` emits `2021, 2,328K, 68%, ...`). The parser read the first six cells
and shifted every metric one column left, filing the year itself as hotel
guests. `_transposed()` now skips a leading bare-year cell, but only when the
row carries a seventh cell.

### Open: source typo, deliberately not "fixed"

**`2021-12` monthly total revenue is wrong in the source PDF.** It prints
`676K` where every neighbouring month is in the hundreds of millions — November
is `562MN`, and December YTD `4.3B` minus November YTD `3.7B` implies roughly
600 million.

Scaled faithfully, `676K` becomes `0.676 aed_million`. That is what the parser
emits. It is left alone rather than inferred, and
`pipeline/dct/dct_validate.py` flags it on every run under
`implausible revenue magnitude`.

### Coverage gaps

- `2015-02` and `2018-02` are byte-identical duplicates of a 2017-02 report.
  Both parse to zero KPI rows by design.
- `2023-11` and `2023-12` reports are unavailable. 2023 has 10 monthly rows;
  2015 and 2018 have 11.
- The `ytd` grain is sparse and uneven: 2021-12 contributes no `ytd`
  `hotel_guests` row, and December reports sometimes contribute both a YTD
  table and an annual table. `dct_kpis.csv` carries 390 `ytd` rows against 786
  `month` rows, but the `ytd` rows are the ones carrying the historical unit
  defects. **Prefer `month`.**
- Monthly passenger data stops after 2018. 2020 annual arrivals are missing
  entirely; 2018 carries 2 imputed months.

### Unavailable sources

These do not exist in any public release and no substitute was invented:

- Monthly resident vs non-resident arrivals
- Transfer passenger share
- Accommodation share and VFR (visiting friends and relatives) share
- Event calendars

Because VFR is unmeasured, any passenger-to-visitor conversion is an explicit
assumption, not an estimate. See `docs/decisions/dct_notes.md`.

### Derived figures

`derived == 1` marks **68 rows** the pipeline computed or estimated rather than
read from a printed table cell. This is a much broader caveat than it looks:

**Chart-read estimates (64 rows).** For 2017-11 through 2021-03, and for all
four 2019 quarters, the publisher stopped printing monthly guest counts and
revenue in a table and moved them to a bar chart. The pipeline reads the bar
baseline and records the result as an estimate. Affected metrics are
`hotel_guests` (as `count`) and `total_revenue` (as `aed_million`), plus `alos`
for the 2019 quarters.

These are approximations off a rendered chart, accurate to roughly the nearest
half-percent at best. They are the reason `hotel_guests` carries `count` for
2020 and 2021-01 to 2021-03 and then switches to printed `thousands` from
2021-04. **Any trend analysis spanning 2019-2021 is partly built on chart
readings.**

**Narrative-derived (2 rows), both in `2024-annual-2024`:**

- **`international_guests == 3,000,000`** is derived from narrative prose ("about
  3 million"), not a table cell. It carries no more precision than the word
  "about", and it is the denominator of the 2024 true-international test in
  `docs/retarget_test.md`.
- `hotel_guests == 5,800,000` is the derived annual total.

`2024` `hotel_guests` also appears as `5,819` and `5,504` thousand, which are the
resident/ex-UAE and international splits. **These are not duplicates** of the
`5,800,000` count — they are its components.

Annual totals for 2015-2016 are sums of published months and are not flagged
`derived`, since they are arithmetic on printed values rather than estimates.

---

## Verification

`pipeline/dct/dct_validate.py` reconciles each year's monthly series against the
annual total printed in the same report, and checks an external anchor:

```
2017: months=12 sum=4,874,253 published=4,875,499 diff=1,246 (+0.026%) [OK]
```

The 4,875,499 figure is stated in the December 2017 DCT report. A parser that
quietly drops a month or transposes two produces a plausible-looking series, so
this is the cheapest test that catches both.

The four standalone diagnostics are run directly, **not via pytest** (pytest is
not installed in this environment):

```
.venv/bin/python pipeline/tests/test_market_link.py
.venv/bin/python pipeline/tests/test_aggregate_link.py
.venv/bin/python pipeline/tests/test_retarget_international.py
.venv/bin/python pipeline/tests/crosscheck_markets.py
```

They write `docs/market_link_test.md`, `docs/aggregate_link_test.md`,
`docs/retarget_test.md`. **Those three reports are gitignored**, so a clean
clone will not contain them — regenerate before reading. Each report contains a
`Generated by` line naming the script; that line changes when a script moves,
so report checksums are only comparable with that line excluded.

`data/processed/checksums.md5` records the current checksum of all 14 processed
tables and verifies with `cd data/processed && md5sum -c checksums.md5`.

---

## Provenance of this document

This dictionary was reconstructed from the data itself after a repository
reorganisation. Two hand-written notes files were deleted during that
reorganisation, before their content had been migrated, and the repository had
no commits or backups at the time:

| Lost file | Lines | md5 |
|---|---|---|
| `notebooks/data/ad_airports/notes.md` | 186 | `d7e417ae7d489b8bda1c9f9c436432b4` |
| `notebooks/data/adsb_2025/notes.md` | 127 | `cbff4429c8bb0b99654ceee31cf6bf29` |

Their original narrative interpretation and analyst commentary are
**unrecoverable**. Everything in this dictionary was re-derived by measuring
the current files and source PDFs, not recovered from those notes. Where a
figure is reconstructed rather than recorded, it is labelled as measured.

Two notes files did survive intact and hold the interpretive history:
`docs/decisions/source_availability.md` and `docs/decisions/dct_notes.md`.