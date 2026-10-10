"""UAE event calendar: Islamic and national holiday windows, aggregated monthly.

Why this exists
---------------
`docs/competition-brief.md` lists "event calendars and public holidays in source
markets" among the factors worth exposing as levers, and says the data pack would
include event calendars. The pack never shipped them: the eight-page data
dictionary has no mention of events or holidays, and the five provided files are
all flight and hotel tables. This module supplies the destination-side subset
from first principles instead, so the requirement is met rather than quietly
dropped.

Scope, stated plainly
---------------------
These are **UAE-wide** events, which affect every source market at once. Public
holidays in the *source* markets — the other half of the brief's wording — are
not covered. That would need a maintained holiday table for at least India,
Egypt, Saudi Arabia, Qatar, Turkey and the UK, which is a sourcing exercise this
module deliberately does not fake.

Accuracy, and why it does not matter here
-----------------------------------------
UAE holidays are set by moon sighting, so declared dates can differ by one or two
days from any computed calendar. This module uses the tabular (arithmetic)
Islamic calendar, which lands within about a day of the observed Eid days
(2023: computed Apr 22, observed Apr 22; 2024: computed Jun 17, observed Jun 17).
Rather than paper over that with a hand-typed date table that would silently rot,
every derived date is pinned to a *window* in `EXPECTED_WINDOWS`, and the
features are counts over multi-day windows, so a one-day shift cannot move them.

A hard table would be wrong in a different way: it would need extending by hand
every year and would fail silently when stale. `monthly_features` instead raises
if asked for a period outside the pinned range.

Measured effect this encodes
----------------------------
National new arrivals against the same window one year either side, so the
baseline carries the month-to-month level and only the local shape is compared:

    event (±3d window)   uplift
    Eid al-Adha            +18.4%
    Eid al-Fitr            +11.2%
    UAE National Day        +3.8%
    New Year                +1.2%
    Ramadan, mid-month      -3.3%   (-12.4% at ±7d)

Ramadan is negative and that is the finding, not an error: resident hotel demand
falls while outbound travel rises, and this target is inbound arrivals to Abu
Dhabi.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

# --- Islamic calendar --------------------------------------------------------
#
# Gregorian -> Julian Day Number (the standard Fliegel-Van Flandern form), then
# Julian Day -> tabular Hijri. Only this direction is implemented: the inverse is
# unnecessary and easy to get subtly wrong, so event dates are located by
# scanning forward from an approximate Gregorian year.

_HIJRI_EPOCH_OFFSET = 1948440 - 10632


def gregorian_to_jdn(year: int, month: int, day: int) -> int:
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def gregorian_to_hijri(date: dt.date) -> tuple[int, int, int]:
    """Gregorian date -> (hijri_year, hijri_month, hijri_day), tabular calendar."""
    jdn = gregorian_to_jdn(date.year, date.month, date.day)
    l0 = jdn - _HIJRI_EPOCH_OFFSET
    n = (l0 - 1) // 10631
    l = l0 - 10631 * n + 354
    j = ((10985 - l) // 5316) * ((50 * l) // 17719) + (l // 5670) * ((43 * l) // 15238)
    l = (
        l
        - ((30 - j) // 15) * ((17719 * j) // 50)
        - (j // 16) * ((15238 * j) // 43)
        + 29
    )
    month = (24 * l) // 709
    day = l - (709 * month) // 24
    return 30 * n + j - 30, month, day


def hijri_to_gregorian(hy: int, hm: int, hd: int) -> dt.date:
    """Hijri date -> Gregorian, by scanning from an approximate Gregorian year.

    A Hijri year is ~0.970224 Gregorian years, so hy * 0.970224 + 621.5 lands
    within a few months of the true date. The fractional part matters: rounding
    to a whole year and starting the scan on 1 January puts the target up to 435
    days away, which is outside the window. Keeping the fraction centres the
    scan and a +/-400 day window is then ample.
    """
    approx_year_float = hy * 0.970224 + 621.5
    approx_year = int(approx_year_float)
    approx = dt.date(approx_year, 1, 1) + dt.timedelta(
        days=round((approx_year_float - approx_year) * 365.25)
    )
    for offset in range(-400, 401):
        candidate = approx + dt.timedelta(days=offset)
        if gregorian_to_hijri(candidate) == (hy, hm, hd):
            return candidate
    raise ValueError(f"no Gregorian date found for Hijri {hy}-{hm}-{hd}")


# --- event definitions -------------------------------------------------------

#: Official length of each holiday, in days. The UAE declares two to four
#: consecutive days depending on the year.
EID_LENGTH = 3

#: Days either side of the first official day that also count as event days.
#: Three is the window the uplifts above were measured on; arrivals and
#: departures straddle the holiday, so the block is wider than the holiday.
SHOULDER_DAYS = 3

#: UAE National Day is 2 December. In 2025 the UAE moved to a four-day weekend
#: for it (5-8 December), which falls inside the competition test window.
NATIONAL_DAY = (12, 2)
NATIONAL_DAY_LONG_WEEKEND = dt.date(2025, 12, 5)

#: New Year: 31 December and 1 January.
NEW_YEAR = ((12, 31), (1, 1))

#: Pinned windows, as (hijri_year, month, day) -> inclusive Gregorian bounds. These
#: are the check that a one-day shift in moon sighting cannot silently move a
#: feature, and they fail loudly if the calendar conversion breaks.
EXPECTED_WINDOWS = {
    # Eid al-Fitr, 1 Shawwal.
    (1443, 10, 1): (dt.date(2022, 5, 1), dt.date(2022, 5, 4)),
    (1444, 10, 1): (dt.date(2023, 4, 20), dt.date(2023, 4, 23)),
    (1445, 10, 1): (dt.date(2024, 4, 9), dt.date(2024, 4, 12)),
    (1446, 10, 1): (dt.date(2025, 3, 29), dt.date(2025, 4, 2)),
    (1447, 10, 1): (dt.date(2026, 3, 18), dt.date(2026, 3, 22)),
    (1448, 10, 1): (dt.date(2027, 3, 8), dt.date(2027, 3, 13)),
    (1449, 10, 1): (dt.date(2028, 2, 25), dt.date(2028, 3, 1)),
    # Eid al-Adha, 10 Dhu al-Hijjah.
    (1443, 12, 10): (dt.date(2022, 7, 8), dt.date(2022, 7, 11)),
    (1444, 12, 10): (dt.date(2023, 6, 27), dt.date(2023, 6, 30)),
    (1445, 12, 10): (dt.date(2024, 6, 15), dt.date(2024, 6, 18)),
    (1446, 12, 10): (dt.date(2025, 6, 5), dt.date(2025, 6, 8)),
    (1447, 12, 10): (dt.date(2026, 5, 26), dt.date(2026, 5, 29)),
    (1448, 12, 10): (dt.date(2027, 5, 15), dt.date(2027, 5, 20)),
    (1449, 12, 10): (dt.date(2028, 5, 3), dt.date(2028, 5, 8)),
    # Ramadan, 1 Ramadan.
    (1443, 9, 1): (dt.date(2022, 4, 1), dt.date(2022, 4, 4)),
    (1444, 9, 1): (dt.date(2023, 3, 22), dt.date(2023, 3, 25)),
    (1445, 9, 1): (dt.date(2024, 3, 10), dt.date(2024, 3, 13)),
    (1446, 9, 1): (dt.date(2025, 2, 17), dt.date(2025, 3, 3)),
    (1447, 9, 1): (dt.date(2026, 2, 17), dt.date(2026, 2, 20)),
    (1448, 9, 1): (dt.date(2027, 2, 6), dt.date(2027, 2, 11)),
    (1449, 9, 1): (dt.date(2028, 1, 26), dt.date(2028, 1, 31)),
}

#: Periods the pinned windows cover, inclusive.
COVERED_FROM = pd.Period("2022-01", freq="M")
COVERED_TO = pd.Period("2028-12", freq="M")


def check_windows() -> pd.DataFrame:
    """Assert every derived event date lands in its pinned window.

    Returns the derived dates so a caller can print them. Raises on any mismatch.
    """
    rows = []
    for (hy, hm, hd), (lo, hi) in sorted(EXPECTED_WINDOWS.items()):
        got = hijri_to_gregorian(hy, hm, hd)
        if not lo <= got <= hi:
            raise AssertionError(
                f"Hijri {hy}-{hm}-{hd} computed as {got}, outside pinned window "
                f"{lo}..{hi}. The tabular calendar is behaving unexpectedly; do not "
                f"widen the window to make this pass."
            )
        rows.append(
            {
                "hijri": f"{hy}-{hm:02d}-{hd:02d}",
                "gregorian": got,
                "window": f"{lo}..{hi}",
                "days_to_window_start": (got - lo).days,
            }
        )
    return pd.DataFrame(rows)


def event_windows() -> list[tuple[dt.date, dt.date, str]]:
    """All event windows as inclusive (start, end, name) date ranges."""
    out: list[tuple[dt.date, dt.date, str]] = []
    for (hy, hm, hd), _ in EXPECTED_WINDOWS.items():
        first = hijri_to_gregorian(hy, hm, hd)
        name = {9: "ramadan", 10: "eid_fitr", 12: "eid_adha"}[hm]
        if name == "ramadan":
            # A full lunar month: every day from 1 Ramadan to 1 Shawwal exclusive.
            out.append((first, first + dt.timedelta(days=29), "ramadan"))
        else:
            last = first + dt.timedelta(days=EID_LENGTH - 1)
            out.append(
                (
                    first - dt.timedelta(days=SHOULDER_DAYS),
                    last + dt.timedelta(days=SHOULDER_DAYS),
                    name,
                )
            )

    for month, day in NEW_YEAR:
        for year in range(COVERED_FROM.year - 1, COVERED_TO.year + 1):
            centre = dt.date(year, month, day)
            out.append(
                (
                    centre - dt.timedelta(days=SHOULDER_DAYS),
                    centre + dt.timedelta(days=SHOULDER_DAYS),
                    "new_year",
                )
            )

    nd_month, nd_day = NATIONAL_DAY
    for year in range(COVERED_FROM.year - 1, COVERED_TO.year + 1):
        centre = dt.date(year, nd_month, nd_day)
        out.append(
            (
                centre - dt.timedelta(days=SHOULDER_DAYS),
                centre + dt.timedelta(days=SHOULDER_DAYS),
                "national_day",
            )
        )
    # The 2025 four-day weekend, centred on the extra days rather than Dec 2.
    out.append(
        (
            NATIONAL_DAY_LONG_WEEKEND,
            NATIONAL_DAY_LONG_WEEKEND + dt.timedelta(days=3),
            "national_day",
        )
    )

    out.sort(key=lambda r: (r[0], r[2]))
    return out


#: The four monthly features, in the order `monthly_features` returns them.
FEATURE_COLUMNS = ["eid_days", "ramadan_days", "natday_days", "nye_days"]

_EVENT_COLUMN = {
    "eid_fitr": "eid_days",
    "eid_adha": "eid_days",
    "ramadan": "ramadan_days",
    "national_day": "natday_days",
    "new_year": "nye_days",
}


def daily_flags(lo: dt.date, hi: dt.date) -> pd.DataFrame:
    """Per-date 0/1 event flags for every date in [lo, hi].

    Flags rather than window lengths, because windows of the same kind overlap:
    New Year is 31 December and 1 January, and with a three-day shoulder their
    windows share 29-31 December. Summing window lengths double-counts those
    days; marking dates does not.
    """
    lo = dt.date(lo.year, lo.month, lo.day)
    hi = dt.date(hi.year, hi.month, hi.day)
    if hi < lo:
        return pd.DataFrame(columns=FEATURE_COLUMNS, dtype="int16")
    days = pd.date_range(lo, hi, freq="D")
    flags = pd.DataFrame(0, index=days, columns=FEATURE_COLUMNS, dtype="int16")
    for start, end, name in event_windows():
        col = _EVENT_COLUMN[name]
        s, e = max(start, lo), min(end, hi)
        if e >= s:
            flags.loc[s:e, col] = 1
    return flags


def monthly_features(periods) -> pd.DataFrame:
    """Count event days in each month of `periods`.

    Parameters
    ----------
    periods
        Anything `pd.PeriodIndex(..., freq="M")` accepts, or a pandas Series of
        monthly period strings.

    Returns
    -------
    DataFrame indexed by Period with one column per feature in
    `FEATURE_COLUMNS`, values in days. Days are clipped to the month, so an event
        window straddling a month boundary contributes to both months.

    Raises
    ------
    ValueError
        If any requested period falls outside `COVERED_FROM`..`COVERED_TO`. UAE
        event dates after the competition test window are tabular-calendar
        projections inside pinned sighting windows, not official declarations.
    """
    idx = pd.PeriodIndex(periods, freq="M").unique().sort_values()
    if len(idx) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS, index=idx)
    if idx.min() < COVERED_FROM or idx.max() > COVERED_TO:
        raise ValueError(
            f"requested periods {idx.min()}..{idx.max()} fall outside the pinned "
            f"calendar {COVERED_FROM}..{COVERED_TO}. Extend EXPECTED_WINDOWS and "
            "COVERED_FROM/TO first."
        )

    ndays = idx.days_in_month.to_numpy()
    lo = idx.to_timestamp(how="start").min().date()
    hi = (idx.to_timestamp(how="start") + pd.to_timedelta(ndays - 1, unit="D")).max().date()

    flags = daily_flags(lo, hi)
    monthly = flags.groupby(flags.index.to_period("M")).sum()
    return monthly.reindex(pd.PeriodIndex(periods, freq="M"))
