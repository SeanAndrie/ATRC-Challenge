"""Normalise the DCT nationality table into a consistent monthly panel.

Two defects are corrected here, both of which are invisible in the raw parse
and fatal to any time-series work:

1. Period labels are not comparable across eras. The 2015-2019 reports are
   parsed to a two-digit month, but the 2021-2023 reports carry the printed
   month-and-year string ('AUG 2021'). Joined as-is, a decade of data sorts
   into two unrelated groups.

2. Share is only printed directly in the recent era. The 2015-2019 reports give
   a guest count by market and a total, but no percentage, so the older years
   have no share at all. The share is derived here from the report's own total,
   which is preferable to leaving it null because it uses the publisher's
   denominator rather than a reconstructed one.
"""

import re

from dct_common import MONTH_NUM

TOTAL_LABELS = {"TOTAL", "Total", "total"}
OTHER_LABELS = {"Other", "Other Nationalities", "Others"}

MONTH_TOKEN = re.compile(r"^([A-Za-z]{3,9})\s+(\d{4})$")


def normalize_period(period, year):
    """Reduce a printed period label to a two-digit month, or '' if it is not
    a month. Quarterly, annual and unknown labels are returned unchanged."""
    if period is None:
        return ""
    s = str(period).strip()
    if not s:
        return ""
    m = MONTH_TOKEN.match(s)
    if m:
        month = MONTH_NUM.get(m.group(1).lower())
        if month:
            return "%02d" % month
    return s


def unit_factor(unit):
    """Multiplier converting a guest count into absolute guests."""
    u = (unit or "count").strip().lower()
    if u.startswith("thousand") or u in ("000", "k"):
        return 1000.0
    return 1.0


def normalize(nats, kpis):
    """Return (rows, stats) with periods unified and shares derived."""
    periods_fixed = 0
    for r in nats:
        new = normalize_period(r.get("period"), r.get("year"))
        if new != r.get("period"):
            periods_fixed += 1
            r["period"] = new

    kpi_total = {}
    for r in kpis:
        if r.get("metric") == "hotel_guests" and r.get("granularity") == "month":
            v = r.get("value")
            if v:
                kpi_total[(r.get("year"), r.get("period"))] = v

    reported = {}
    for r in nats:
        label = str(r.get("nationality") or "").strip()
        if label in TOTAL_LABELS and r.get("guests"):
            reported[(r.get("source_file"), r.get("granularity"),
                      r.get("period"))] = r["guests"] * unit_factor(r.get("guests_unit"))

    derived = 0
    for r in nats:
        if r.get("share_pct") not in (None, "", 0):
            continue
        label = str(r.get("nationality") or "").strip()
        if label in TOTAL_LABELS:
            continue
        key = (r.get("source_file"), r.get("granularity"), r.get("period"))
        total = reported.get(key)
        basis = "report_total"
        if not total:
            total = kpi_total.get((r.get("year"), r.get("period")))
            basis = "kpi_guests"
        if not total or not r.get("guests"):
            continue
        guests = r["guests"] * unit_factor(r.get("guests_unit"))
        r["share_pct"] = round(100.0 * guests / total, 1)
        r["share_basis"] = basis
        derived += 1

    stats = {"periods_normalized": periods_fixed, "shares_derived": derived}
    return nats, stats
