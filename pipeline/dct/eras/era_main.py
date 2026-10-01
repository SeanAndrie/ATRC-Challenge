"""Era A parser: 2015-01, 2015-03 .. 2016-12 'Main Indicators Summary' tables.

Each report carries an emirate-wide block and per-region blocks.  The emirate block
has current-year actual, previous-year actual and a growth-rate column; region blocks
are rejected by a region keyword in the block's heading.
"""

from __future__ import annotations

import re

from dct_common import num

TITLE = "Hotel Establishments Main Indicators Summary"
SCOPE_YTD = re.compile(r"Year To Date \(January\s*-\s*(\w+)\)", re.I)
SCOPE_MONTH = re.compile(r"^\s*(January|February|March|April|May|June|July|August"
                         r"|September|October|November|December)\b", re.I)
HDR_RE = re.compile(r"Indicator\s{2,}(20\d\d)\s{2,}(20\d\d)", re.I)
NUMTOK = re.compile(r"-?[\d,]+(?:\.\d+)?%?")
STOP_RE = re.compile(r"Copyright|Market Intelligence Section|Hotels Opened", re.I)
REGION_RE = re.compile(r"Abu Dhabi City|Al Ain|Al Dhafra|Western Region", re.I)

MONTH_OF = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}

LABELS = [
    (r"actual guest arrivals|guest arrivals", "hotel_guests", "count"),
    (r"guest nights", "guest_nights", "count"),
    (r"room nights", "room_nights", "count"),
    (r"occupied rooms", "occupied_rooms", "count"),
    (r"average length of stay", "alos", "days"),
    (r"occupancy rate", "occupancy_rate", "pct"),
    (r"room revenue", "room_revenue", "aed"),
    (r"food ?& ?beverages", "fb_revenue", "aed"),
    (r"other revenue", "other_revenue", "aed"),
    (r"total revenue", "total_revenue", "aed"),
    (r"^arr$", "arr", "aed"),
    (r"rev ?par|revpar", "revpar", "aed"),
]
LABEL_RE = [(re.compile(p, re.I), m, u) for p, m, u in LABELS]


def _metric(label: str):
    lab = re.sub(r"\s+", " ", label).strip(" :")
    for rx, metric, unit in LABEL_RE:
        if rx.search(lab):
            return metric, unit
    return None, None


def _scope(win: str, hm, month: int):
    head = win[:hm.start()]
    sy = SCOPE_YTD.search(head)
    if sy:
        return MONTH_OF.get(sy.group(1).lower(), month), 1
    sm = SCOPE_MONTH.search(head.split("\n")[-1] if "\n" in head else head)
    if sm:
        return MONTH_OF[sm.group(1).lower()], 0
    sm = SCOPE_MONTH.search(head)
    return (MONTH_OF[sm.group(1).lower()], 0) if sm else (month, 0)


def parse(txt: str, src: str, year: int, month: int):
    out = []
    for tm in re.finditer(re.escape(TITLE), txt, re.I):
        win = txt[tm.end(): tm.end() + 2600]
        hm = HDR_RE.search(win)
        if not hm:
            continue
        if REGION_RE.search(win[:hm.start()]):
            continue
        cur_yr, prev_yr = int(hm.group(1)), int(hm.group(2))
        if cur_yr != year:
            continue
        mo, ytd = _scope(win, hm, month)
        label, emitted = "", set()
        for ln in win[hm.end():].split("\n")[:60]:
            if STOP_RE.search(ln):
                break
            toks = NUMTOK.findall(ln)
            head_txt = NUMTOK.split(ln)[0].strip()
            if toks:
                lab = (label + " " + head_txt).strip() if head_txt else label
                label = ""
                if len(toks) < 2 or not lab:
                    continue
                metric, unit = _metric(lab)
                if not metric or metric in emitted:
                    continue
                cv, pv = num(toks[0]), num(toks[1])
                if cv is None:
                    continue
                out.append(dict(source_file=src, year=year, period=f"{mo:02d}",
                                granularity="month", segment="total", metric=metric,
                                unit=unit, value=cv, ytd=ytd, derived=0,
                                prior_year=prev_yr, prior_value=pv,
                                prior_pct=round(cv / pv * 100 - 100, 2)
                                if pv not in (None, 0) else None,
                                printed_pct=num(toks[2]) if len(toks) > 2 else None))
                emitted.add(metric)
            elif head_txt and not toks:
                label = (label + " " + head_txt).strip()
            else:
                label = ""
    return out
