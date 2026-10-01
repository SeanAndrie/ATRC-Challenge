"""Chart-era parser (2017-11 .. 2018-12, 2020-01 .. 2021-03, and 2015-02 = 2018-02).

Two sources, in order of precision:

1. the 'KEY INDICATORS' table, when the report prints one for the reporting month
   (exact counts plus a YTD column) - used for 2020-01..2021-03;
2. otherwise the 12-month bar charts on 'OVERALL PERFORMANCE'.  The current-year
   series is the flat baseline row - the last row above the 'Jan Feb ... Dec' axis
   that still holds usable numbers (future months are printed as 0).  The
   current-month value is its last non-zero cell and the printed % change is the
   last percent at or above that row; the sum of its non-zero cells reproduces the
   printed YTD total.
"""

from __future__ import annotations

import re

from dct_common import num

CELL = r"[-+]?[\d,]+(?:\.\d+)?\s*(?:KK|MN|K|M|B)?%?"
CELL_RE = re.compile(CELL)

METRICS = [
    (r"HOTEL\s*GUESTS\s*\(?0*00\)?\*?", "hotel_guests", "thousands"),
    (r"OCCUPANCY\s*RATE", "occupancy_rate", "pct"),
    (r"AVERAGE\s*LENGTH\s*OF\s*STAY", "alos", "days"),
    (r"HOTEL\s*TOTAL\s*REVENUE", "total_revenue", "aed_million"),
    (r"AVERAGE\s*ROOM\s*RATE|\bARR\b", "arr", "aed"),
    (r"REVENUE\s*PER\s*AVAILABLE", "revpar", "aed"),
]
LABEL_RE = re.compile(
    r"(?:^|(?<=\s))(HOTEL\s*GUESTS\s*\(?0*00\)?\*?|OCCUPANCY\s*RATE"
    r"|AVERAGE\s*LENGTH\s*OF\s*STAY[^)\n]*|HOTEL\s*TOTAL\s*REVENUE[^)\n]*"
    r"|AVERAGE\s*ROOM\s*RATE[^)\n]*|REVENUE\s*PER\s*AVAILABLE[^)\n]*"
    r"|GUESTS|OCCUPANCY|ALOS|ARR|REVPAR)\s{2,}", re.I)
AXIS_RE = re.compile(r"\bJan\b\s+\bFeb\b\s+\bMar\b", re.I)
KEY_HDR = re.compile(r"Key\s+Indicators\s{2,}(\S.*)$", re.I | re.M)
PCT_RE = re.compile(r"[-+]?\s?[\d.]+\s?%")
ZERO_PCT = ("0%", "0.0%", "0.00%", "0 %", "0.0 %")


def _cells(ln: str, drop_years: bool = True):
    out = []
    for c in CELL_RE.findall(ln):
        c = c.strip()
        if not c:
            continue
        if drop_years and re.fullmatch(r"\d{4}", c) and 1980 <= int(c) <= 2100:
            continue
        out.append(c)
    return out


def _num(tok: str):
    t = re.sub(r"\s+", "", tok).replace(",", "").rstrip("%")
    m = re.match(r"^([-+]?[\d.]+)(KK|MN|K|M|B)?$", t, re.I)
    if not m:
        return None, ""
    return num(m.group(1)), (m.group(2) or "").upper()


def _pct(tok: str):
    t = re.sub(r"\s+", "", tok).replace(",", "")
    return num(t) if "%" in t else None


def _unit_for(metric: str, base_unit: str, suffix: str = ""):
    if suffix in ("K", "KK"):
        return "thousands"
    if suffix in ("M", "MN", "B") and base_unit == "aed_million":
        return "aed_million"
    return base_unit


# ------------------------------------------------------------- key indicators

def key_table(txt: str, src: str, year: int, period: str, month_names):
    for hm in KEY_HDR.finditer(txt):
        n_act = len(re.findall(r"Actuals?\b", hm.group(1), re.I))
        if n_act != 2:
            continue
        before = txt[max(0, hm.start() - 300): hm.start()]
        mons = [m for m in re.finditer(r"([A-Z][A-Za-z]{2,8})\s+(\d{4})", before)]
        if not mons or not any(m.group(1).lower() in month_names for m in mons):
            continue
        rows = _table_rows(txt[hm.end(): hm.end() + 1200])
        if len({r[0] for r in rows}) < 4:
            continue
        out = []
        for metric, unit, mv, mp, yv, yp in rows:
            if mv is None:
                continue
            out.append(dict(source_file=src, year=year, period=period,
                            granularity="month", segment="total", metric=metric,
                            unit=unit, value=mv, ytd=0, derived=0, prior_pct=mp,
                            source_block="key_indicators_table"))
            if yv is not None:
                out.append(dict(source_file=src, year=year, period=period,
                                granularity="ytd", segment="total", metric=metric,
                                unit=unit, value=yv, ytd=1, derived=0, prior_pct=yp,
                                source_block="key_indicators_table"))
        if out:
            return out
    return []


def _table_rows(win: str):
    rows = []
    for ln in win.split("\n")[:30]:
        m = LABEL_RE.search(ln)
        if not m:
            continue
        lab = re.sub(r"\s+", " ", m.group(1)).strip()
        metric = unit = None
        for pat, mk, u in METRICS:
            if re.match(r"^\s*(?:" + pat + r")\s*$", lab, re.I):
                metric, unit = mk, u
                break
        if metric is None:
            continue
        cells = _cells(ln[m.end():])
        if len(cells) < 2:
            continue
        mv, suf = _num(cells[0])
        unit = _unit_for(metric, unit, suf)
        yv, _ = _num(cells[2]) if len(cells) > 2 else (None, "")
        rows.append((metric, unit, mv, _pct(cells[1]), yv,
                     _pct(cells[3]) if len(cells) > 3 else None))
    dedup = {}
    for r in rows:
        dedup.setdefault(r[0], r)
    return list(dedup.values())


# ----------------------------------------------------------------- chart pages

def chart_series(txt: str, src: str, year: int, period: str):
    lines = txt.split("\n")
    out, series = [], {}
    for pat, metric, unit in METRICS:
        cands = [i for i, ln in enumerate(lines)
                 if re.match(r"^\s*" + pat, ln, re.I) and "PAGE" not in ln.upper()]
        head = None
        for c in cands:                       # skip table-of-contents hits
            if any(AXIS_RE.search(lines[k]) for k in
                   range(c + 1, min(len(lines), c + 30))):
                head = c
                break
        if head is None:
            continue
        end = head + 1
        while end < len(lines) and not AXIS_RE.search(lines[end]):
            end += 1
            if end - head > 45:
                break
        block = lines[head + 1: min(end, head + 45)]
        base_idx, nums = None, []
        for j in range(len(block) - 1, -1, -1):
            if not _cells(block[j]):
                continue
            cand = []
            for c in _cells(block[j]):
                if c.endswith("%") and (c[0] in "+-" or
                                         c.replace(" ", "") in ZERO_PCT):
                    continue
                v, _ = _num(c)
                if v is not None:
                    cand.append((v, c))
            if cand:
                base_idx, nums = j, cand
                break
        if base_idx is None:
            continue
        nonzero = [(v, c) for v, c in nums if v != 0]
        growth = None
        for j in range(base_idx - 1 if unit == "pct" else base_idx, -1, -1):
            p = PCT_RE.findall(block[j])
            p = [t for t in p if t.replace(" ", "") not in ZERO_PCT]
            if p:
                growth = _pct(p[-1])
                break
        cur = nonzero[-1][0] if nonzero else None
        series[metric] = dict(values=[v for v, _ in nums],
                              nonzero=[v for v, _ in nonzero], current=cur,
                              growth=growth, unit=unit,
                              ytd_printed=next((v for v, _ in nonzero
                                                if abs(v - sum(x for x, _ in nonzero)) < 1),
                                               None))
        if cur is not None:
            out.append(dict(source_file=src, year=year, period=period,
                            granularity="month", segment="total", metric=metric,
                            unit=unit, value=cur, ytd=0, derived=0,
                            prior_pct=growth, source_block="chart_baseline"))
    return out, series


# ------------------------------------------------------------ regional tables

REGION_HEAD = re.compile(r"PERFORMANCE\s+BY\s+REGION", re.I)
REGION_ROW = re.compile(
    r"^\s*(GUESTS|OCCUPANCY RATE|ALOS DAYS?|REVENUES \(M AED\)|ARR \(AED\)|"
    r"REVPAR \(AED\))\s{2,}", re.I)
SCOPE_WORD = re.compile(r"YTD", re.I)
REGIONS = ("ABU DHABI", "AL AIN", "AL DHAFRA")
ADDITIVE_METRICS = {"hotel_guests", "total_revenue"}


def region_tables(txt: str, src: str, year: int, period: str):
    """Sum the three regional columns of every 'PERFORMANCE BY REGION' table."""
    out, notes = [], []
    for hm in REGION_HEAD.finditer(txt):
        head = txt[hm.start(): hm.start() + 700]
        if not all(r in head.upper() for r in REGIONS):
            continue
        ytd = bool(SCOPE_WORD.search(head))
        win = txt[hm.end(): hm.end() + 1400]
        for ln in win.split("\n")[:26]:
            m = REGION_ROW.match(ln)
            if not m:
                continue
            metric, unit = _metric(m.group(1))
            if metric is None:
                continue
            cells = _cells(ln[m.end():])
            vals = [_num(c)[0] for c in cells[:6]]
            vals = [v for v in vals if v is not None]
            if len(vals) < 3:
                continue
            groups = [vals[0:2], vals[2:4], vals[4:6]]
            groups = [g for g in groups if g]
            if len(groups) < 3:
                continue
            if metric in ADDITIVE_METRICS:
                out.append(dict(source_file=src, year=year, period=period,
                                granularity="ytd" if ytd else "month",
                                segment="total", metric=metric, unit=unit,
                                value=round(sum(g[0] for g in groups), 1),
                                ytd=int(ytd), derived=1,
                                source_block="region_sum",
                                region_values=[g[0] for g in groups]))
            else:
                notes.append((metric, [g[0] for g in groups]))
    return out, notes


def _metric(label: str):
    lab = re.sub(r"\s+", " ", label).strip(" :()")
    for pat, metric, unit in METRICS:
        if re.match(r"^\s*(?:" + pat + r")\s*$", lab.upper().replace("  ", " "), re.I):
            return metric, unit
    low = lab.lower()
    if low.startswith("guests"):
        return "hotel_guests", "count"
    if low.startswith("occupancy"):
        return "occupancy_rate", "pct"
    if low.startswith("alos"):
        return "alos", "days"
    if low.startswith("revenues"):
        return "total_revenue", "aed_million"
    if low.startswith("arr"):
        return "arr", "aed"
    if low.startswith("revpar"):
        return "revpar", "aed"
    return None, None


PLAUSIBLE = {"hotel_guests": (10.0, 2500.0), "occupancy_rate": (5.0, 100.0),
             "alos": (0.5, 20.0), "total_revenue": (10.0, 6000.0),
             "arr": (20.0, 10000.0), "revpar": (5.0, 5000.0)}


def plausible(metric, value):
    lo, hi = PLAUSIBLE.get(metric, (float("-inf"), float("inf")))
    return value is not None and lo <= value <= hi


def parse(stem: str, year: int, period: str):
    """Chart-era KPIs: exact regional sums where possible, chart for rates."""
    from dct_common import read_text
    txt = read_text(stem)
    rows, checks = [], []
    reg, _ = region_tables(txt, stem, year, period)
    have = {r["metric"] for r in reg}
    rows += reg
    _, ser = chart_series(txt, stem, year, period)
    for metric, d in sorted(ser.items()):
        if metric in ADDITIVE_METRICS and metric in have:
            continue
        if d["current"] is None or not plausible(metric, d["current"]):
            checks.append((metric, "chart_current_rejected"))
            continue
        if d["growth"] is not None and abs(d["growth"]) > 100:
            checks.append((metric, "chart_growth_rejected"))
            d["growth"] = None
        rows.append(dict(source_file=stem, year=year, period=period,
                         granularity="month", segment="total", metric=metric,
                         unit=d["unit"], value=d["current"],
                         growth_pct=d["growth"], ytd=0, derived=0,
                         source_block="chart_baseline"))
    for r in rows:
        if r["source_block"] != "region_sum" or r["metric"] != "hotel_guests":
            continue
        g = ser.get("hotel_guests", {})
        if g.get("current"):
            diff = r["value"] / 1000.0 - g["current"]
            checks.append(("hotel_guests_region_vs_chart",
                           round(diff, 2), "ok" if abs(diff) <= 1.0 else "MISMATCH"))
    return rows, checks
