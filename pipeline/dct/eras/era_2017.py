"""Era 2017-01..2017-10: 'Dear Hotel Partner' monthly reports.

The first page carries an executive 'MONTHLY SUMMARY': clean headline figures
for the current month (hotel guests, occupancy, ALOS, RevPAR, hotel revenue)
plus a 'Top 5 Nationalities' list.  The second page carries the full
'HOTEL GUESTS BY NATIONALITY (thousands)' table (parsed in dct_nationality).
Only the headline KPI figures are extracted here; the 12-month charts on later
pages are not parsed.
"""
import re

from dct_common import read_text, num

GUESTS = re.compile(r"Hotel\s+Guests\s+([\d,]{5,})")
GUEST_LABEL = re.compile(r"Hotel\s+Guests", re.I)
GUEST_TAIL = re.compile(r"Guests", re.I)
LABEL_START = re.compile(r"^\s*\(?\s*Hotel\s+Guests\b", re.I)
TOP5 = re.compile(r"Top\s*5\s*Nationalities", re.I)
BIG_NUM = re.compile(r"\b(\d{1,3}(?:,\d{3}){1,2}|\d{6,7})\b")
OCCUP = re.compile(r"Occupancy\s+rate\s+([\d.]+)\s*%")
ALOS = re.compile(r"Average\s+length\s+of\s+stay\s+([\d.]+)")
REVPAR = re.compile(r"Revenue\s+per\s+available\s+room\s+([\d,]+)")
REVENUE = re.compile(r"Hotel\s+revenues\s+([\d,]+)")

PATTERNS = {
    "occupancy_rate": (OCCUP, "pct"),
    "alos": (ALOS, "days"),
    "revpar": (REVPAR, "aed"),
    "total_revenue": (REVENUE, "aed_million"),
}

BOUNDS = {
    "occupancy_rate": (30, 100),
    "alos": (1.0, 6.0),
    "revpar": (100, 900),
    "total_revenue": (100, 1500),
}


def find_guests(txt):
    """Locate the headline guest count on the summary page.

    Three traps make this harder than a label-adjacent regex suggests. The
    headline letter opens by discussing 'hotel guests' in prose. The number is
    in some issues typeset in its own text block, which pdftotext emits *above*
    the label rather than beside it. And the same document also contains a
    table of contents, a YTD block and a regional breakdown, all of which carry
    a 'Hotel Guests' label with an unrelated figure attached.

    The 'Top 5 Nationalities' block is unique to the summary page, so it is
    used to identify that page. Within it, only a line *starting* with 'Hotel
    Guests' is treated as the label, which rejects 'YTD Abu Dhabi Hotel
    Guests', 'Abu Dhabi Hotel Guests' and '(Hotel Guests)'. The value is then
    read on the same line, failing that upward from the label.
    """
    for page in txt.split("\f"):
        if not TOP5.search(page):
            continue
        lines = page.split("\n")
        for i, line in enumerate(lines):
            if not LABEL_START.match(line):
                continue
            tail = line[len(LABEL_START.match(line).group(0)):]
            for m in BIG_NUM.finditer(tail):
                value = num(m.group(1))
                if value and 50_000 <= value <= 2_000_000:
                    return value
            for j in range(i - 1, max(-1, i - 7), -1):
                for m in BIG_NUM.finditer(lines[j]):
                    value = num(m.group(1))
                    if value and 50_000 <= value <= 2_000_000:
                        return value
    return None


def parse(stem: str, year: int, period: str):
    txt = read_text(stem)
    rows, notes = [], []

    guests = find_guests(txt)
    if guests is None:
        notes.append(("hotel_guests", "missing"))
    else:
        growth = None
        gm = re.search(r"Hotel\s+Guests\s+[\d,]+\s+([+-]?\d+(?:\.\d+)?)\s*%",
                       txt, re.I)
        if gm:
            growth = num(gm.group(1))
        rows.append(dict(source_file=stem, year=year, period=period,
                         granularity="month", segment="total",
                         metric="hotel_guests", unit="count", value=guests,
                         growth_pct=growth, ytd=0, derived=0,
                         source_block="monthly_summary"))

    for metric, (pat, unit) in PATTERNS.items():
        lo, hi = BOUNDS[metric]
        m = pat.search(txt)
        value = num(m.group(1)) if m else None
        if value is None:
            notes.append((metric, "missing" if not m else "not_numeric"))
            continue
        if not (lo <= value <= hi):
            notes.append((metric, "implausible"))
            continue
        rows.append(dict(source_file=stem, year=year, period=period,
                         granularity="month", segment="total", metric=metric,
                         unit=unit, value=value, growth_pct=None,
                         ytd=0, derived=0, source_block="monthly_summary"))
    if not rows:
        notes.append(("summary", "no_figures"))
    return rows, notes
