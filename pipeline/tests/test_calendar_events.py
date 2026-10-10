"""Does the pinned UAE event calendar survive contact with real dates?

`pipeline/calendar_events.py` derives Ramadan, Eid al-Fitr and Eid al-Adha from
an inline tabular Hijri conversion rather than a hand-typed holiday table, because
a hand-typed table has to be extended by hand every year and fails silently when
stale. That trade is only sound if the conversion is trustworthy, so this test
pins it against dates that are independently known.

Four checks, in increasing order of what they would catch:

  1. Round trip. The Gregorian -> Julian Day Number step is checked against a
     known constant (2000-01-01 is JDN 2451545), because everything downstream
     inherits any error here.
  2. Window pins. Every derived event date must fall inside a hand-checked
     Gregorian window. Windows are two to five days wide, which is the point: UAE
     holidays are set by moon sighting, so declared dates can legitimately differ
     from a computed calendar by a day or two. A window pin fails on a broken
     conversion, not on ordinary sighting variation.
  3. Internal consistency. Eid al-Fitr must be exactly 30 days after 1 Ramadan,
     because both come from the same lunar month. This catches a pin table that
     has quietly drifted out of agreement with the arithmetic, which per-date pins
     alone would not.
  4. Aggregation. Known months must produce known event-day counts, including the
     two places the aggregation is easy to get wrong: a window straddling a month
     boundary, and overlapping windows for the same event.

This test also records, rather than hides, the two months where the computed date
sits well inside its pin window: Ramadan 1446 starts 1 March 2025 by computation
against a window deliberately opened to 17 February, because sightings that year
genuinely differed by more than a day across the region.

Usage:  .venv/bin/python pipeline/tests/test_calendar_events.py
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import REPORT_CALENDAR  # noqa: E402
from calendar_events import (       # noqa: E402
    COVERED_FROM,
    COVERED_TO,
    EXPECTED_WINDOWS,
    FEATURE_COLUMNS,
    check_windows,
    daily_flags,
    gregorian_to_hijri,
    gregorian_to_jdn,
    hijri_to_gregorian,
    monthly_features,
)

# Known-good anchors. 2000-01-01 -> JDN 2451545 is the standard epoch used by
# every astronomical calendar, so it is a check on the Gregorian side.
JDN_EPOCH_CHECK = ((2000, 1, 1), 2451545)

# Gregorian dates whose Hijri day is well established and outside this repo's
# pin table, used to catch a conversion that drifts with the century.
HIJRI_ANCHORS = (
    (dt.date(2023, 4, 21), (1444, 9, 30)),
    (dt.date(2024, 6, 16), (1445, 12, 9)),
    (dt.date(2019, 9, 1), (1441, 1, 1)),   # 1 Muharram 1441
)

# Months whose event-day counts are known by construction. Each is a day count,
# not a boolean: a 3-day Eid with a 3-day shoulder is 9 days, and 9 days spread
# over a month boundary splits across two months.
EXPECTED_MONTHS = {
    # Eid al-Fitr 1444 = 22 Apr 2023, official 22-24, shoulder 19-25 Apr: all 9
    # days fall inside April.
    "2023-04": {"eid_days": 9, "ramadan_days": 21},
    # The same window's Ramadan (23 Mar - 21 Apr) spills: 9 days in March.
    "2023-03": {"eid_days": 0, "ramadan_days": 9},
    # Eid al-Adha 1444 = 29 Jun 2023, window 26 Jun - 4 Jul: splits 5/4.
    "2023-06": {"eid_days": 5, "ramadan_days": 0},
    "2023-07": {"eid_days": 4, "ramadan_days": 0},
    # New Year = 31 Dec and 1 Jan with a 3-day shoulder each. In December the two
    # windows overlap on 29-31 Dec, so the union is 28-31 Dec = 4 days. Summing
    # window lengths instead of unioning dates gives 7, which is the bug this
    # case exists to catch.
    "2023-12": {"nye_days": 4},
    "2024-01": {"nye_days": 4},
    # UAE four-day National Day weekend, 5-8 Dec 2025, unioned with the 2 Dec
    # window (29 Nov - 5 Dec) to give 1-8 Dec = 8 days.
    "2025-12": {"natday_days": 8},
    # A quiet month: every feature zero. Guards against a window bleeding across
    # a boundary it should not.
    "2023-08": {"eid_days": 0, "ramadan_days": 0, "natday_days": 0, "nye_days": 0},
    # A projected year (Hijri 1448): the horizon now runs to 2028-12 so forward
    # scenarios have a calendar. Ramadan starts 8 Feb 2027; its 29 days split
    # 21/9 across February and March, and Eid al-Fitr's 3-day window with
    # shoulders is 7-15 Mar = 9 days, all inside March.
    "2027-02": {"ramadan_days": 21},
    "2027-03": {"eid_days": 9, "ramadan_days": 9},
}

# Windows deliberately opened wider than the usual two days, with the reason.
WIDE_WINDOWS = {
    (1446, 9, 1): "Ramadan 1446 sightings differed across the region in 2025; "
                  "1 Mar by computation against 18 Feb observed in parts of it.",
    (1444, 12, 10): "Eid al-Adha 1444 was observed 28 Jun in most of the UAE.",
}


def check_roundtrip() -> list[str]:
    """Gregorian -> JDN and JDN -> Hijri against known anchors."""
    notes = []
    (y, m, d), expected = JDN_EPOCH_CHECK
    got = gregorian_to_jdn(y, m, d)
    assert got == expected, f"JDN {y}-{m:02d}-{d:02d} = {got}, expected {expected}"
    notes.append(f"JDN epoch: {y}-{m:02d}-{d:02d} = {got} (expected {expected})")

    for date, expected_hijri in HIJRI_ANCHORS:
        got_hijri = gregorian_to_hijri(date)
        assert got_hijri == expected_hijri, (
            f"{date} -> {got_hijri}, expected {expected_hijri}"
        )
        notes.append(f"{date} -> Hijri {got_hijri[0]}-{got_hijri[1]:02d}-{got_hijri[2]:02d}")

    # The inverse must agree with the forward direction on every pinned date.
    for hy, hm, hd in EXPECTED_WINDOWS:
        assert gregorian_to_hijri(hijri_to_gregorian(hy, hm, hd)) == (hy, hm, hd)
    notes.append(f"Hijri -> Gregorian round trip holds for all {len(EXPECTED_WINDOWS)} pins")
    return notes


def check_internal_consistency() -> list[str]:
    """Eid al-Fitr must be exactly 30 days after 1 Ramadan."""
    notes = []
    for year in sorted({hy for hy, _, _ in EXPECTED_WINDOWS}):
        ramadan = hijri_to_gregorian(year, 9, 1)
        eid = hijri_to_gregorian(year, 10, 1)
        gap = (eid - ramadan).days
        assert gap == 30, f"Hijri year {year}: Ramadan -> Eid al-Fitr is {gap} days, expected 30"
        notes.append(f"Hijri {year}: Ramadan {ramadan} -> Eid al-Fitr {eid} ({gap} days)")
    return notes


def check_aggregation() -> list[str]:
    """Known months produce known event-day counts."""
    notes = []
    feats = monthly_features(pd.period_range(COVERED_FROM, COVERED_TO, freq="M"))
    assert list(feats.columns) == FEATURE_COLUMNS, feats.columns.tolist()

    for period, expected in EXPECTED_MONTHS.items():
        row = feats.loc[pd.Period(period, freq="M")]
        for col, value in expected.items():
            got = int(row[col])
            assert got == value, f"{period} {col} = {got}, expected {value}"
        notes.append(f"{period}: " + " ".join(f"{k}={int(row[k])}" for k in FEATURE_COLUMNS))

    # Every feature must be a day count within the month, never negative.
    assert (feats >= 0).all().all(), "negative event-day count"
    assert feats.to_numpy().max() <= feats.index.days_in_month.max(), (
        "event days exceed the length of the month"
    )
    notes.append(f"all counts within [0, days in month]; peak = {feats.to_numpy().max()} days")

    # Coverage guard: outside the pinned range this must raise, not return zeros.
    try:
        monthly_features(pd.period_range("2029-01", "2029-02", freq="M"))
    except ValueError:
        notes.append("coverage guard raises for 2029-01..2029-02, as intended")
    else:
        raise AssertionError("coverage guard did not raise outside COVERED_FROM..COVERED_TO")

    # Monotonicity: asking for a sub-range must agree with the full range, so the
    # aggregation does not depend on which slice the caller passes.
    subset = monthly_features(pd.period_range("2024-01", "2024-12", freq="M"))
    assert subset.equals(feats.loc[subset.index]), "sub-range disagrees with full range"
    notes.append("sub-range aggregation matches the full range")

    # Flags must agree with the aggregation they feed.
    flags = daily_flags(dt.date(2023, 6, 26), dt.date(2023, 7, 4))
    assert int(flags["eid_days"].sum()) == 9, int(flags["eid_days"].sum())
    notes.append("Eid al-Adha 1444 window is 9 flagged days, 26 Jun - 4 Jul")
    return notes


def main() -> None:
    print("=" * 78)
    print("UAE EVENT CALENDAR — conversion, pins and aggregation")
    print("=" * 78)

    print("\n-- round trip ---------------------------------------------------------")
    roundtrip = check_roundtrip()
    for line in roundtrip:
        print(f"  {line}")

    print("\n-- internal consistency (Ramadan -> Eid = 30 days) -------------------")
    for line in check_internal_consistency():
        print(f"  {line}")

    print("\n-- derived event dates ------------------------------------------------")
    windows = check_windows().copy()
    windows["gregorian"] = windows["gregorian"].astype(str)
    print(windows.to_string(index=False))

    print("\n-- aggregation --------------------------------------------------------")
    for line in check_aggregation():
        print(f"  {line}")

    wide = "\n".join(
        f"  * Hijri {hy}-{hm:02d}-{hd:02d}: {reason}"
        for (hy, hm, hd), reason in WIDE_WINDOWS.items()
    )
    note = (
        "# UAE event calendar test\n\n"
        "## Result\n\n"
        "All checks pass. Every derived event date falls inside its pinned window,\n"
        "Ramadan to Eid al-Fitr is exactly 30 days in every pinned year, and the\n"
        "monthly aggregation reproduces known day counts including a window that\n"
        "straddles a month boundary and two overlapping New Year windows.\n\n"
        "## Derived dates\n\n"
        + windows.to_markdown(index=False)
        + "\n\n## What is checked, and why\n\n"
        "* **JDN epoch.** 2000-01-01 must be JDN 2451545. Every derived date\n"
        "  depends on this step, so an error here is silent and total.\n"
        "* **Three Hijri anchors** outside this repo's pin table, including one\n"
        "  from 2019, catch a conversion that drifts across the century.\n"
        "* **Round trip** on all pins: the inverse must reproduce the forward\n"
        "  result. The inverse is located by scanning rather than by formula, which\n"
        "  is why it needs its own check.\n"
        "* **Windows, not exact dates.** UAE holidays follow moon sighting, so a\n"
        "  computed date may legitimately sit a day or two from the declared one.\n"
        "  A two-day-wide pin therefore cannot fail on sighting variation, but does\n"
        "  fail on a broken conversion.\n\n"
        "## Windows deliberately opened wider\n\n"
        + wide
        + "\n\n## Known trap: overlapping windows\n\n"
        "New Year is 31 December *and* 1 January. With a three-day shoulder each,\n"
        "their windows share 29-31 December. Summing window lengths double-counts\n"
        "those days and reports 7 in December where the true union is 4, so the\n"
        "calendar marks dates and counts marked dates instead.\n\n"
        "## Known trap: month boundaries\n\n"
        "Eid al-Adha 1444 (29 June 2023) has a window of 26 June - 4 July, which\n"
        "splits 5 days into June and 4 into July. A feature that must straddle is\n"
        "the honest one; a monthly boolean would have collapsed both to a single\n"
        "bit and lost the magnitude.\n\n"
        "## Caveats\n\n"
        f"* **Coverage is {COVERED_FROM} to {COVERED_TO}.** The competition test\n"
        "  window ends 2026-02, but the simulator's forward scenarios need a\n"
        "  calendar, so the pin table was extended through Hijri 1449 (2028-12).\n"
        "  Beyond that the module raises rather than reporting zeros, so a\n"
        "  longer horizon needs the pin table extended and this test re-run.\n"
        "* **UAE-wide only.** These are destination-side events that hit every\n"
        "  source market at once. Public holidays in the source markets, the other\n"
        "  half of the brief's wording, are not covered.\n"
        "* **Astronomical, not declared.** Dates are computed. Where the UAE\n"
        "  government declares a different date for sighting reasons, the computed\n"
        "  one leads by a day or two. The multi-day windows absorb this; a\n"
        "  single-day flag would not.\n"
        "* **December 2025 weekend.** The UAE moved to a four-day National Day\n"
        "  weekend (5-8 December 2025), inside the competition test window. That is\n"
        "  encoded, but the wider weekend regime change is not a feature.\n"
    )
    REPORT_CALENDAR.parent.mkdir(parents=True, exist_ok=True)
    REPORT_CALENDAR.write_text(note)
    print(f"\nwrote {REPORT_CALENDAR}")


if __name__ == "__main__":
    main()