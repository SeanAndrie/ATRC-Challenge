"""Reconcile the parsed DCT monthly series against the published annual totals.

A parser that quietly drops a month, or transposes two of them, produces a
plausible-looking series. Summing the months and comparing with the annual
total the publisher prints in the same document is the cheapest test that
catches both, and it uses only figures already extracted.

Known published totals used as external anchors:

    2017  4,875,499  Abu Dhabi emirate hotel guests, stated in the December
                    2017 report

Usage:  .venv/bin/python pipeline/dct/dct_validate.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import DCT_GUESTS_NATIONALITY, DCT_KPIS  # noqa: E402

PUBLISHED = {2017: 4875499}
TOLERANCE = 0.05

UNIT_FACTOR = {"count": 1.0, "thousands": 1_000.0, "millions": 1_000_000.0}


def to_absolute(rows):
    """Convert a set of hotel_guests rows to absolute guest counts.

    The reports change unit between eras: 2015-2018 and 2020 print raw counts,
    2019 and 2022-2023 print thousands, and 2024 prints both thousands by
    month and millions for the year. Summing without normalising produces
    differences of three orders of magnitude that look like data loss.
    """
    factor = rows.unit.map(UNIT_FACTOR)
    unknown = rows[factor.isna()]
    if not unknown.empty:
        raise ValueError(f"unrecognised unit(s): {sorted(unknown.unit.unique())}")
    return rows.value * factor


def load():
    kpi = pd.read_csv(DCT_KPIS)
    nat = pd.read_csv(DCT_GUESTS_NATIONALITY)
    return kpi, nat


def reconcile(kpi, year, metric="hotel_guests"):
    """Sum the monthly series for a year and compare with the annual row."""
    sel = kpi[(kpi.year == year) & (kpi.metric == metric)].copy()
    months = sel[sel.granularity == "month"].copy()
    annual = sel[sel.granularity == "year"].copy()
    if months.empty:
        return None
    months["abs"] = to_absolute(months)
    annual_value = None
    if not annual.empty:
        annual_value = to_absolute(annual).max()
    periods = sorted(months.period.astype(str))
    dupes = sorted({p for p in periods if periods.count(p) > 1})
    return {"year": year,
            "months": len(months),
            "distinct": months.period.nunique(),
            "dup_periods": ",".join(dupes),
            "units": ",".join(sorted(months.unit.unique())),
            "monthly_sum": months["abs"].sum(),
            "annual_row": annual_value}


def main():
    kpi, nat = load()
    rows = []
    for year in sorted(kpi.year.dropna().unique()):
        r = reconcile(kpi, int(year))
        if r:
            rows.append(r)
    rec = pd.DataFrame(rows)
    rec["diff"] = rec.annual_row - rec.monthly_sum
    rec["diff_pct"] = (100 * rec["diff"] / rec.annual_row).round(3)

    print("=== monthly sum vs annual row parsed from the same reports ===")
    print(rec.to_string(index=False))

    print("\n=== external anchor ===")
    for year, published in PUBLISHED.items():
        r = rec[rec.year == year]
        if r.empty:
            print(f"{year}: no monthly rows")
            continue
        got = float(r.iloc[0]["monthly_sum"])
        diff = published - got
        pct = 100 * diff / published
        flag = "OK" if abs(pct) <= TOLERANCE else "CHECK"
        print(f"{year}: months={int(r.iloc[0]['months'])} sum={got:,.0f} "
              f"published={published:,} diff={diff:,.0f} ({pct:+.3f}%) [{flag}]")

    incomplete = rec[rec.months < 12]
    if not incomplete.empty:
        print("\n=== years with fewer than 12 monthly rows ===")
        print(incomplete[["year", "months", "distinct", "monthly_sum"]].to_string(index=False))

    dupes = rec[rec.dup_periods != ""]
    if not dupes.empty:
        print("\n=== duplicate monthly periods ===")
        print(dupes[["year", "months", "distinct", "dup_periods"]].to_string(index=False))

    mixed = rec[rec.units.str.contains(",")]
    if not mixed.empty:
        print("\n=== years mixing units across months (check for a misread row) ===")
        print(mixed[["year", "units"]].to_string(index=False))

    nat_months = nat[nat.granularity == "month"].groupby("year").period.nunique()
    print("\n=== nationality month coverage ===")
    print(nat_months.to_string())

    # Revenue is printed in whatever unit each report happens to use, so a
    # single mistyped suffix is invisible in the unit column alone. The 2021-12
    # report prints "676K" for monthly revenue where every neighbouring month is
    # in the hundreds of millions; scaled faithfully that lands at 0.676 million.
    # Flagging the implausible magnitude keeps the error visible on every run
    # without inferring a number the publisher never printed.
    rev = kpi[kpi.metric == "total_revenue"]
    suspect = rev[(rev.unit == "thousands")
                  | ((rev.unit == "aed_million") & (rev.value < 10))]
    if not suspect.empty:
        print("\n=== implausible revenue magnitude (source typo, kept verbatim) ===")
        print(suspect[["source_file", "granularity", "period",
                       "value", "unit"]].to_string(index=False))
        print("  2021-12 month revenue is printed '676K' in the source PDF. The")
        print("  true figure is roughly 676 million AED. The parser does not")
        print("  guess a correction; correct this row by hand if it matters.")


if __name__ == "__main__":
    main()
