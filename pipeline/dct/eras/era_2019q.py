"""Era B: 2019 quarterlies.

Quarter totals come from the three regional columns (exact counts); the monthly
rates come from the flat baseline row of each KPI chart; a guest-weighted ALOS
is derived from the regional table.
"""
from dct_common import read_text
from . import era_chart as CH

QMONTHS = {"Q1": (1, 2, 3), "Q2": (4, 5, 6), "Q3": (7, 8, 9), "Q4": (10, 11, 12)}


def parse(stem, year, period):
    txt = read_text(stem)
    rows, notes, checks = [], [], []
    reg, _ = CH.region_tables(txt, stem, year, period)
    have = set()
    for r in reg:
        r["granularity"] = "quarter"
        r["segment"] = "total"
        have.add(r["metric"])
        rows.append(r)
    # guest-weighted ALOS from the regional table
    _, ser = CH.chart_series(txt, stem, year, period)
    months = QMONTHS[period]
    for metric, d in sorted(ser.items()):
        vals = {i: v for i, v in enumerate(d.get("values") or [], 1)}
        for mi in months:
            v = vals.get(mi)
            if v is None or not CH.plausible(metric, v):
                continue
            rows.append(dict(source_file=stem, year=year, period=f"{mi:02d}",
                             granularity="month", segment="total",
                             metric=metric, unit=d["unit"], value=v,
                             growth_pct=None, ytd=0, derived=0,
                             source_block="chart_baseline"))
    # weighted ALOS for the quarter: sum(guests*alos)/sum(guests)
    g = next((r for r in reg if r["metric"] == "hotel_guests"), None)
    if g:
        parts = _regional_rows(txt, stem, year, period)
        tot_g = sum(p[0] for p in parts)
        if parts and tot_g:
            rows.append(dict(source_file=stem, year=year, period=period,
                             granularity="quarter", segment="total",
                             metric="alos", unit="days",
                             value=round(sum(p[0] * p[1] for p in parts) / tot_g, 2),
                             growth_pct=None, ytd=0, derived=1,
                             source_block="guest_weighted_regions"))
    q = [r for r in rows if r["metric"] == "hotel_guests" and r["granularity"] == "quarter"]
    s = next((r for r in rows if r["metric"] == "hotel_guests"
              and r["granularity"] == "month"), None)
    if q and s:
        months_sum = sum(r["value"] for r in rows
                         if r["metric"] == "hotel_guests"
                         and r["granularity"] == "month")
        checks.append(("quarter_guests_vs_chart_months", months_sum,
                       round(q[0]["value"] / 1000.0, 1),
                       abs(months_sum - q[0]["value"] / 1000.0) <= 3.0))
    return rows, notes, checks


def _regional_rows(txt, stem, year, period):
    """[(region guests, region alos)] for the first regional table."""
    for hm in CH.REGION_HEAD.finditer(txt):
        head = txt[hm.start(): hm.start() + 700]
        if not all(r in head.upper() for r in CH.REGIONS):
            continue
        win = txt[hm.end(): hm.end() + 1400]
        g, a = None, None
        for ln in win.split("\n")[:26]:
            m = CH.REGION_ROW.match(ln)
            if not m:
                continue
            lab = m.group(1).upper()
            if not lab.startswith("GUESTS"):
                continue
            vals = [CH._num(c)[0] for c in CH._cells(ln[m.end():])[:6]]
            vals = [v for v in vals if v is not None]
            g = [v[0] for v in (vals[0:2], vals[2:4], vals[4:6]) if v]
            if len(g) < 3:
                g = None
                continue
            for ln2 in win.split("\n")[:26]:
                m2 = CH.REGION_ROW.match(ln2)
                if m2 and m2.group(1).upper().startswith("ALOS"):
                    v2 = [CH._num(c)[0] for c in CH._cells(ln2[m2.end():])[:6]]
                    v2 = [v for v in v2 if v is not None]
                    a = [v[0] for v in (v2[0:2], v2[2:4], v2[4:6]) if v]
                    if len(a) < 3:
                        a = None
                    break
            if g and a:
                return list(zip(g, a))
    return []
