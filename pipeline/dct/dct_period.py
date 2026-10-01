"""Detect each DCT PDF's *internal* reporting period from the document itself."""

from __future__ import annotations

import re

from dct_common import MONTH_NUM, TEXTDIR

MONTH_ALT = "|".join(sorted(MONTH_NUM, key=len, reverse=True))
SEP = r"[\s\.,\-&/]{0,12}"
MONTH_TITLE = re.compile(rf"(?<![A-Za-z])({MONTH_ALT})(?![a-z]){SEP}((?:19|20)\d\d)", re.I)
QUARTER_TITLE = re.compile(r"\bQ([1-4])\b[\s\.,\-&/]{0,12}((?:19|20)\d\d)", re.I)
FULLWORD = re.compile(r"full[\s\-]*year", re.I)
# "Indicator   2016   2015   Growth Rate %"
IND_YEARS = re.compile(r"Indicator\s{2,}(20\d\d)\s{2,}(20\d\d)", re.I)
# "2017        2018" chart year labels
CHART_YEARS = re.compile(r"\b(20\d\d)\s{2,}(20\d\d)\b")


def _month_no(tok: str):
    t = tok.lower()
    return MONTH_NUM.get(t[:3]) or MONTH_NUM.get(t)


def detect(stem: str, head_chars: int = 700):
    txt = (TEXTDIR / f"{stem}.txt").read_text(encoding="utf-8", errors="replace")
    head = txt[:head_chars]
    cands = []
    q = QUARTER_TITLE.search(head)
    if q:
        cands.append((q.start(), "Q" + q.group(1), int(q.group(2)), q))
    for m in MONTH_TITLE.finditer(head):
        mo = _month_no(m.group(1))
        if mo:
            cands.append((m.start(), mo, int(m.group(2)), m))
    if cands:
        cands.sort(key=lambda c: c[0])
        pos, mon, yr, mm = cands[0]
        if isinstance(mon, str):
            period, kind = f"Q{mon[1]}", "quarter"
        else:
            period, kind = f"{yr:04d}-{int(mon):02d}", "month"
        ctx = head[max(0, pos - 60):mm.end() + 40]
        return dict(period=period, year=yr, basis=f"{kind} title",
                    full_year=bool(FULLWORD.search(ctx)), raw=mm.group(0),
                    title_pos=pos)
    # fallback 1: year columns of the Main Indicators table
    m = IND_YEARS.search(txt[:6000])
    if m and int(m.group(2)) == int(m.group(1)) - 1:
        yr = int(m.group(1))
        mo = _find_month_in(txt[:head_chars])
        period = f"{yr:04d}-{mo:02d}" if mo else None
        return dict(period=period, year=yr, basis="indicator-table-year",
                    full_year=False, raw=m.group(0), title_pos=0)
    # fallback 2: chart year labels
    m = CHART_YEARS.search(txt[:6000])
    if m:
        yr = max(int(m.group(1)), int(m.group(2)))
        mo = _find_month_in(txt[:head_chars])
        return dict(period=f"{yr:04d}-{mo:02d}" if mo else None, year=yr,
                    basis="chart-year-labels", full_year=False,
                    raw=m.group(0), title_pos=0)
    y = re.search(r"\b((?:19|20)\d\d)\b", head)
    if y:
        return dict(period=None, year=int(y.group(1)), basis="year only",
                    full_year=bool(FULLWORD.search(head[:y.end() + 40])),
                    raw=y.group(0), title_pos=y.start())
    return dict(period=None, year=None, basis="undetermined", full_year=False,
                raw=None, title_pos=-1)


def _find_month_in(head: str):
    m = MONTH_TITLE.search(head)
    if m:
        return _month_no(m.group(1))
    for mm in re.finditer(rf"(?<![A-Za-z])({MONTH_ALT})(?![a-z])", head, re.I):
        v = _month_no(mm.group(1))
        if v:
            return v
    return None
