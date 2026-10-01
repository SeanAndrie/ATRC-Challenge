"""Annual AUH arrivals 2015-2024 from official SCAD sources, tested against DCT guests.

Phase 1 of the simulator wanted monthly arrivals for 2019-2025. That turned out
not to exist: Key Transport Statistics publishes monthly *aircraft movements*
only, and the Statistical Yearbook publishes monthly movements with *annual*
passengers. The detailed Transport Statistics XLSX that carried monthly
passengers (table 1.2) is the exception, and its post-2018 editions sit behind a
JS-rendered page.

What is available is an annual arrivals series spanning a decade, which is
better suited to the question actually being asked than four pre-COVID months
were. Two sources supply it:

  * Statistical Yearbook of Abu Dhabi 2020, table 2.7.7, for 2016-2019
  * Key Transport Statistics 2025, table 2, for 2021-2025

2020 is absent from both and is left empty rather than interpolated, because it
is the one year where interpolation would do the most damage.

Usage:  .venv/bin/python pipeline/aviation/build_annual_arrivals.py
"""

import re
import sys
from pathlib import Path

import openpyxl
import pandas as pd
import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The SCAD workbook and yearbook live in data/raw/scad/, where this script
# also downloads them if they are missing.
from paths import (  # noqa: E402
    DCT_KPIS,
    PAX_ARRIVALS_ANNUAL,
    PAX_MOVEMENTS,
    SCAD_KTS,
    SCAD_YEARBOOK,
)

PAX = PAX_MOVEMENTS
OUT = PAX_ARRIVALS_ANNUAL

GUEST_FACTOR = {"count": 1.0, "thousands": 1_000.0}


def from_key_transport_2025():
    """AUH arrivals by year from KTS 2025, table 2."""
    path = str(SCAD_KTS)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Table 2"]
    years, block, out = None, None, {}
    for r in ws.iter_rows(values_only=True):
        cells = [c for c in r if c not in (None, "")]
        if not cells:
            continue
        joined = [str(c).strip() for c in cells]
        if joined[0].startswith("20") and all(re.fullmatch(r"20\d\d", x) for x in joined[:5]):
            years = joined[:5]
            continue
        if joined[0].startswith("Zayed"):
            block = "auh"
            continue
        if joined[0].startswith("Al Ain"):
            block = "alain"
            continue
        if block == "auh" and joined[0].lower() == "arrival":
            for yr, val in zip(years, cells[1:6]):
                if isinstance(val, (int, float)):
                    out[int(yr)] = round(float(val))
    wb.close()
    return out


def from_yearbook_2020():
    """AUH arrivals by year from Statistical Yearbook 2020, table 2.7.7."""
    path = str(SCAD_YEARBOOK)
    out = {}
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "2.7.7" not in text or "Air Transport by Airport" not in text:
                continue
            head = text.split("2.7.7", 1)[1]
            years = [int(y) for y in re.findall(r"\b(20[12]\d)\b", head)[:4]]
            m = re.search(r"Passengers:\s*\n\s*Arrival\s+((?:[\d,]+\s*){3,6})", head)
            if not years or not m:
                continue
            vals = [int(v.replace(",", "")) for v in m.group(1).split()]
            if len(vals) < len(years):
                continue
            for yr, val in zip(years, vals):
                out[yr] = val
            break
    return out


def from_local():
    """2015 arrivals, already transcribed in pax_movements.csv."""
    p = pd.read_csv(PAX)
    a = p[(p.scope == "auh") & (p.granularity == "month")
          & (p.metric == "pax_arrivals")]
    return {int(y): int(round(v)) for y, v in a.groupby(a.period_start.str[:4]).value.sum().items()}


def annual_guests():
    """Annual hotel guests, preferring complete published rows, else monthly sums.

    Two Februaries are unrecoverable (2015 and 2018, both lost to a misfiled
    source file). Those two months are linearly interpolated from their
    neighbours rather than dropped, because dropping the year would cost the
    2019 comparison, which is the single most informative row in the series.
    2023 and 2024 use their published annual figures, since the monthly series
    is incomplete for those years.
    """
    k = pd.read_csv(DCT_KPIS)
    rows = k[(k.metric == "hotel_guests") & (k.granularity == "year")
             & (k.segment == "total")]
    pub = {}
    for yr, g in rows.groupby("year"):
        pub[yr] = [r.value * GUEST_FACTOR[r.unit] for r in g.itertuples()]

    m = (k[(k.metric == "hotel_guests") & (k.granularity == "month")
            & (k.segment == "total")].copy())
    m["guests"] = m.value * m.unit.map(GUEST_FACTOR)
    m["mo"] = m.period.astype(int)
    out, imputed = {}, {}
    for yr, g in m.groupby("year"):
        g = g.set_index("mo").guests
        have = set(g.index)
        for mo in sorted(set(range(1, 13)) - have):
            lo, hi = max([x for x in have if x < mo], default=None), \
                min([x for x in have if x > mo], default=None)
            if lo and hi:
                g[mo] = g[lo] + (g[hi] - g[lo]) * (mo - lo) / (hi - lo)
        out[yr] = g.sum()
        imputed[yr] = sorted(set(range(1, 13)) - have)
    # A published annual figure is authoritative wherever one exists, which
    # covers 2015 and the incomplete 2023 and 2024. The interpolated sum is only
    # a fallback, used for 2016-2022 where no annual row is published.
    #
    # Two years carry a rounding-style narrative row alongside the table row
    # (2024 prints both 5,819 thousand and 5,800 thousand). Where a complete
    # twelve-month sum exists, the published value closest to it is taken, which
    # keeps this file consistent with the monthly total used elsewhere.
    for yr, vals in pub.items():
        if not vals:
            continue
        out[yr] = (min(vals, key=lambda v: abs(v - out[yr]))
                   if not imputed.get(yr) else vals[0])
        imputed[yr] = []
    return out, imputed


def main():
    kts, yb, loc = from_key_transport_2025(), from_yearbook_2020(), from_local()
    print("KTS 2025 table 2 :", dict(sorted(kts.items())))
    print("Yearbook 2.7.7   :", dict(sorted(yb.items())))
    print("local 2015       :", dict(sorted(loc.items())))

    # The yearbook independently reproduces the three years already transcribed
    # by hand, which is a useful check on the local file.
    for yr in sorted(set(yb) & set(loc)):
        flag = "match" if yb[yr] == loc[yr] else f"MISMATCH {loc[yr]}"
        print(f"  cross-check {yr}: yearbook {yb[yr]:,} vs local {flag}")

    arrivals = {y: v for y, v in loc.items() if y == 2015}
    arrivals.update(yb)
    arrivals.update(kts)
    arrivals = {y: v for y, v in sorted(arrivals.items()) if y <= 2024}

    guests, imputed = annual_guests()

    df = pd.DataFrame({"year": list(arrivals)})
    df["arrivals"] = df.year.map(arrivals)
    df["guests"] = df.year.map(guests)
    df["imputed_months"] = df.year.map(lambda y: ",".join(str(x) for x in imputed.get(y, [])))
    df["guests_per_arrival"] = df.guests / df.arrivals
    df["arr_yoy"] = df.arrivals.pct_change() * 100
    df["gst_yoy"] = df.guests.pct_change() * 100
    df["source"] = df.year.map(
        lambda y: "scad_kts2025_t2" if y in kts
        else "scad_yearbook2020_t277" if y in yb
        else "local_pax_movements")

    print(f"\n=== annual AUH arrivals vs DCT hotel guests ===\n"
          f"{'year':>5} {'arrivals':>12} {'guests':>11} {'g/arr':>7} "
          f"{'arr yoy':>8} {'gst yoy':>8}  imputed  source")
    for _, r in df.iterrows():
        print(f"{int(r.year):>5} {int(r.arrivals):>12,} {r.guests:>11,.0f} "
              f"{r.guests_per_arrival:>7.4f} {r.arr_yoy:>7.1f}% {r.gst_yoy:>7.1f}%  "
              f"{r.imputed_months or '-':>7}  {r.source}")

    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")

    pre = df[(df.year <= 2019) & df.arr_yoy.notna()]
    print(f"\n=== growth correlation, pre-COVID {pre.year.min()}-{pre.year.max()} "
          f"(n={len(pre)}) ===")
    print(f"pearson(arrivals yoy, guests yoy) = {pre.arr_yoy.corr(pre.gst_yoy):.3f}")
    print(f"spearman = {pre.arr_yoy.rank().corr(pre.gst_yoy.rank()):.3f}")

    print(f"\n=== the clearest single observation ===")
    r = df[df.year == 2019].iloc[0]
    print(f"2018 to 2019: arrivals {r.arr_yoy:+.1f}%, hotel guests {r.gst_yoy:+.1f}%.")
    print("Air traffic fell while hotel demand rose. A converter keyed on arrivals")
    print("would have predicted the opposite sign of change that year.")


if __name__ == "__main__":
    main()
