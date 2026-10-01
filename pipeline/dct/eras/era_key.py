"""Key-indicators era parsers (2019-Q1..Q4, 2020-01..2021-03, 2021-04..2023-10).

Only the emirate-wide 'OVERALL PERFORMANCE' sections are read; the regional and
star-rating tables that share the same layout are skipped by section anchoring.

  * row-major   'GUESTS  177,283  0.5%  -42.1%  489,276  -37.8%  -45.9%' - the number
                of 'Actual(s)' columns in the header gives the number of scope groups.
  * transposed  'Hotel Guests | Hotel Occupancy | Hotel ALOS | ARR | REVPAR | Revenues'
                header with one value row and one '% change' row beneath it.
"""

from __future__ import annotations

import re

from dct_common import num

# "BN" must precede "B" so a two-letter billion suffix is not truncated to "B"
# with the trailing "N" dropped, and "MN" must precede "M" for the same reason.
CELL = r"[-+]?[\d,]+(?:\.\d+)?\s*(?:KK|MN|K|M|BN|B)?%?"
CELL_RE = re.compile(CELL)
SCALERS = {"KK": 1e6, "MN": 1e6, "K": 1e3, "M": 1e6, "BN": 1e9, "B": 1e9}

METRIC_ALIASES = [
    (r"hotel guests|^guests|guests \(k\)|actual guest arrivals", "hotel_guests", "count"),
    (r"hotel occupancy|occupancy", "occupancy_rate", "pct"),
    (r"alos|length of stay", "alos", "days"),
    (r"revenues?|hotel revenue|total rev", "total_revenue", "aed_million"),
    (r"arr \(aed\)|^arr\b|average room rate", "arr", "aed"),
    (r"revpar|rev par|revenue per available", "revpar", "aed"),
]
TRANS_METRICS = ["hotel_guests", "occupancy_rate", "alos", "arr", "revpar",
                 "total_revenue"]
TRANS_UNITS = {"hotel_guests": "count", "occupancy_rate": "pct", "alos": "days",
               "arr": "aed", "revpar": "aed", "total_revenue": "aed_million"}
ROW_METRIC = re.compile(
    r"(?:^|(?<=\s))(hotel guests|guests \(k\)|guests|occupancy rate|occupancy"
    r"|alos days|alos nights|alos \(nights\)|alos|revenues \(m aed\)"
    r"|total rev\(m aed\)|arr \(aed\)|revpar \(aed\))\s{2,}", re.I)
HDR_RE = re.compile(r"Key\s+Indicators\s{2,}(?P<rest>\S.*)$", re.I | re.M)
TRANS_HDR_RE = re.compile(
    r"Hotel\s+Guests\s+Hotel\s+Occupancy\s+Hotel\s+ALOS\s+"
    r"ARR\s*\(AED\)\s+REVPAR\s*\(AED\)\s+Revenues", re.I)
SCOPE_RE = re.compile(r"YTD\s*\(?\s*([A-Za-z]{3,9})[\s\-,]*(\d{4})", re.I)
MONTH_SCOPE_RE = re.compile(r"(?<![A-Za-z])([A-Z]{3,9})\s+(\d{4})\b")
OVERALL_RE = re.compile(r"^\s*(?:\S.*?)?\bOVERALL PERFORMANCE\b", re.I | re.M)
SECTION_END = re.compile(
    r"PERFORMANCE\s+BY|TOP\s+HOTEL|TOP\s+NATIONALITIES|HOTEL GUEST NATIONALITIES"
    r"|KEY\s+HIGHLIGHTS|HOTEL\s+SUPPLY\s*$|Copyright|CONFIDENTIAL", re.I)
BAD_SCOPE = {"DCT", "SOURCE", "CONFIDENTIAL", "DEPARTMENT", "ACTUAL", "PAGE"}


def _metric(label: str):
    lab = re.sub(r"\s+", " ", label).strip(" :()")
    for pat, metric, unit in METRIC_ALIASES:
        if re.search(pat, lab, re.I):
            return metric, unit
    return None, None


def _value(tok: str, unit: str):
    """Parse a printed cell into (value, unit).

    Only cells that carry a magnitude suffix are rescaled. A bare token keeps
    the metric's declared unit untouched, which preserves the 2015-2016 rows
    where the publisher printed raw AED and the unit must stay "aed".

    The reports are not consistent about their suffix. 2021 prints guests as
    "969K" but revenue as "1.2B", 2022 prints guests as "4.1 MN" and revenue as
    "5.4 B", while 2023 prints revenue as "4,938M". Treating "B" as if it were
    "M" understated revenue a thousandfold, and a millions suffix on a count
    metric was reported as a literal count of 4.1 guests. Both are corrected by
    scaling through SCALERS and expressing the result in the metric's own unit.
    """
    t = re.sub(r"\s+", "", tok).replace(",", "").rstrip("%")
    m = re.match(r"^([-+]?[\d.]+)(KK|MN|K|M|BN|B)?$", t, re.I)
    if not m:
        return None
    v = num(m.group(1))
    if v is None:
        return None
    suf = (m.group(2) or "").upper()
    if not suf:
        return v, unit
    scale = SCALERS[suf]
    if unit == "aed_million":
        return round(v * scale / 1e6, 6), "aed_million"
    return round(v * scale / 1e3, 6), "thousands"


def _pct(tok: str):
    t = re.sub(r"\s+", "", tok).replace(",", "")
    return num(t) if "%" in t else None


def overall_sections(txt: str):
    """Yield the text window of every 'OVERALL PERFORMANCE' section."""
    for om in OVERALL_RE.finditer(txt):
        if re.search(r"PERFORMANCE\s+BY", om.group(0), re.I):
            continue
        rest = txt[om.end():]
        end, off = len(rest), 0
        for ln in rest.split("\n"):
            if ln.strip() and SECTION_END.search(ln):
                end = off
                break
            off += len(ln) + 1
        yield om.group(0), rest[:end]


def _scope_order(before: str, n_act: int):
    ytd = SCOPE_RE.search(before)
    mons = [m for m in MONTH_SCOPE_RE.finditer(before)
            if not before[:m.start()].rstrip().upper().endswith("YTD")
            and m.group(1).upper() not in BAD_SCOPE]
    mon = mons[-1].group(0) if mons else None
    if n_act <= 1:
        return [("ytd" if (ytd and not mon) else "month", (ytd.group(0) if ytd else mon))]
    if n_act == 2:
        return [("month", mon or "current"), ("ytd", ytd.group(0) if ytd else "ytd")]
    return []


def _row_groups(section: str, src: str, year: int, period: str):
    out = []
    for hm in HDR_RE.finditer(section):
        n_act = len(re.findall(r"Actuals?\b", hm.group("rest"), re.I))
        if n_act > 2:
            continue
        before = section[max(0, hm.start() - 300): hm.start()]
        scopes = _scope_order(before, n_act)
        if not scopes:
            continue
        win = section[hm.end(): hm.end() + 1400]
        for kind, label in scopes:
            for r in _rows(win, src, year, period, kind, label):
                out.append(r)
    return out


def _rows(win, src, year, period, kind, scope_label):
    rows, seen = [], set()
    for ln in win.split("\n")[:40]:
        m = ROW_METRIC.match(ln)
        if not m:
            continue
        cells = CELL_RE.findall(ln[m.end(1):])
        if len(cells) < 2:
            continue
        metric, unit = _metric(m.group(1))
        if not metric or (metric, kind) in seen:
            continue
        v = _value(cells[0], unit)
        if v is None:
            continue
        seen.add((metric, kind))
        rows.append(dict(source_file=src, year=year, period=period,
                         granularity=kind, segment="total", metric=metric,
                         unit=v[1], value=v[0], ytd=int(kind == "ytd"),
                         derived=0, prior_pct=_pct(cells[1]), printed_scope=scope_label))
    return rows


def _transposed_sections(txt):
    """Yield (scope, body, is_ytd) for every transposed key-indicator table.

    The heading above these tables is not consistent, which is what made a
    heading-anchored search miss rows. 2023-08 titles its monthly table
    'OVERALL PERFORMANCE - AUG 2023', but 2023-09 titles the monthly table
    'TOP HOTEL GUEST NATIONALITIES - Sep 2023' and keeps 'OVERALL PERFORMANCE'
    only on the YTD table, so a heading search finds the YTD figure alone and
    files a year-to-date total as if it were the month.

    Anchoring on the six-column header instead finds every table. The 'YTD'
    marker is then read from the lines immediately around the header, which is
    where it reliably sits: either the heading line above, or the 'YTD <month>
    Performance' sub-label directly below.
    """
    lines = txt.split("\n")
    for i, line in enumerate(lines):
        if not TRANS_HDR_RE.search(line):
            continue
        near = lines[max(0, i - 2): i + 5]
        scope = re.sub(r"\s+", " ", " ".join(near)).strip()
        ytd = bool(re.search(r"\bYTD\b", scope, re.I))
        yield scope, "\n".join(lines[i + 1:]), ytd


def _transposed(body, src, year, period, scope, ytd):
    cand = []
    for ln in body.split("\n")[:14]:
        cells = CELL_RE.findall(ln)
        if len(cells) >= 5:
            cand.append(cells)
    if not cand:
        return []
    vals = cand[0][:6]
    # Some year-to-date tables print a bare comparison year in the first cell
    # before the six metrics (2021-09 emits "2021, 2,328K, 68%, ..."). Taking
    # the first six cells then shifts every metric one column left and files the
    # year itself as hotel guests. Only skip it when the row actually carries a
    # seventh cell, so a genuinely six-column row is never shifted.
    if len(cand[0]) >= 7 and re.fullmatch(r"(19|20)\d{2}", vals[0].strip()):
        vals = cand[0][1:7]
    pct = next((c for c in cand[1:] if sum("%" in x for x in c) >= 4), None)
    out = []
    for i, metric in enumerate(TRANS_METRICS):
        if i >= len(vals):
            break
        v = _value(vals[i], TRANS_UNITS[metric])
        if v is None:
            continue
        out.append(dict(source_file=src, year=year, period=period,
                        granularity="ytd" if ytd else "month", segment="total",
                        metric=metric, unit=v[1], value=v[0], ytd=ytd, derived=0,
                        prior_pct=_pct(pct[i]) if pct and i < len(pct) else None,
                        printed_scope=scope))
    return out


def parse(txt: str, src: str, year: int, period: str):
    rows, seen = [], set()

    def add(got):
        for r in got:
            key = (r["metric"], r["granularity"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    for scope, body, ytd in _transposed_sections(txt):
        add(_transposed(body, src, year, period, scope, ytd))
    if rows:
        return rows

    for head_txt, section in overall_sections(txt):
        scope = re.sub(r"\s+", " ", head_txt.split("OVERALL PERFORMANCE")[-1])
        add(_row_groups(section, src, year, period))
    return rows
