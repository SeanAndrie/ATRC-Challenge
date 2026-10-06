"""Arrivals at Zayed International Airport by region of embarkation, from SCAD.

Key Transport Statistics 2025, Table 6: "Arrivals at Zayed International Airport
by Region of Embarkation, 2021-2025". Supplied by Abu Dhabi Airports Company.
The note under the table says it "excludes transit passengers who continued
their journey on the same flight", which aligns it with the pack's P2P concept
(transfer + transit stripped per flight row) rather than with Total PAX.

This is the only official, independent break-down of where AUH's incoming
air passengers embarked, and it is the anchor for validating the pack's
departure-country supply mix. It is *not* a nationality table: it says where
passengers started a leg, not who they are.

It is annual, not monthly, and region-level (11 buckets), not country-level.
Two definitional caveats matter for the diagnostic that consumes it:

  * "Region of embarkation" is the boarding point of the flight arriving at
    AUH. Whether SCAD resolves a connecting itinerary back to its true origin
    is not stated; the persistent gap between the pack's GCC departure share
    and SCAD's GCC share (the pack runs ~5-7 pp higher every year) suggests it
    leans toward true embarkation rather than last departure. That reading is
    checked empirically in the diagnostic, not assumed here.
  * The year rows are trusted as published. 2021 and 2022 were revised by the
    source; the workbook carries both revisions.

Layout (verified against the workbook on 2026-10-06): the year row sits at row
5 with years in columns 3-7 (2021-2025); region names are in column 2; values
for each region are in columns 3-7 aligned with that year row.

Usage:  .venv/bin/python pipeline/aviation/scad_region_arrivals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import SCAD_KTS  # noqa: E402

SHEET = "Table 6"
# Workbook layout, checked against the raw file (see module docstring).
YEAR_ROW = 5
FIRST_DATA_ROW = 6
LAST_DATA_ROW = 17
REGION_COL = 1
FIRST_YEAR_COL = 2
YEARS = list(range(2021, 2026))

REGIONS = [
    "Africa",
    "Asia",
    "Australasia",
    "Eastern Europe",
    "GCC",
    "Indian Subcontinent",
    "Middle East",
    "North America",
    "South America",
    "Western Europe",
    "Other",
]

# Country -> SCAD region. Ambiguous cases were resolved against the SCAD
# table's own buckets, by matching the pack's country-level P2P mix to SCAD's
# region totals year by year (see outputs/market_origin_validation_test.md):
#   * EGYPT -> Africa, not Middle East. SCAD's Africa bucket (~4.6-6.2% of
#     arrivals) is essentially the Cairo flow: its level matches the pack's
#     Egypt departures whether or not the remaining African countries appear
#     in the pack. Keeping Egypt in the Middle East inflated the pack's
#     Middle East share to 3-5x SCAD's.
#   * TURKEY -> Western Europe, not Middle East. SCAD's Middle East bucket
#     matches the pack's Levant only (Israel + Jordan + Lebanon), so Turkije
#     is filed elsewhere by the source; Western Europe leaves the ME bucket
#     self-consistent and minimises the pack-vs-SCAD distance.
#   * AZERBAIJAN, KAZAKHSTAN, UZBEKISTAN -> Asia (Caspian/Central Asia, per
#     SCAD's Asia bucket which SCAD fills well above the pack's East/Southeast
#     Asia supply).
#   * ARMENIA, GEORGIA -> Eastern Europe.
#   * MEXICO -> North America (UN M49 convention).
# The one large, mapping-independent gap is the GCC: the pack runs +5-7 pp
# above SCAD's GCC share every year, driven by the pack filing intra-GCC
# connecting traffic that SCAD resolves to a non-GCC embarkation.
COUNTRY_REGION = {
    # GCC
    "QATAR": "GCC", "SAUDI ARABIA": "GCC", "OMAN": "GCC",
    "BAHRAIN": "GCC", "KUWAIT": "GCC",
    # Indian Subcontinent
    "INDIA": "Indian Subcontinent", "PAKISTAN": "Indian Subcontinent",
    "BANGLADESH": "Indian Subcontinent", "SRI LANKA": "Indian Subcontinent",
    "NEPAL": "Indian Subcontinent",
    # Middle East (Levant in SCAD's counts; Egypt and Turkije sit elsewhere)
    "UNITED ARAB EMIRATES": "Middle East", "ISRAEL": "Middle East",
    "JORDAN": "Middle East", "LEBANON": "Middle East", "IRAQ": "Middle East",
    "IRAN": "Middle East", "SYRIA": "Middle East", "YEMEN": "Middle East",
    "PALESTINE": "Middle East",
    # Africa
    "EGYPT": "Africa", "MOROCCO": "Africa", "SOUTH AFRICA": "Africa",
    "NIGERIA": "Africa", "KENYA": "Africa", "ETHIOPIA": "Africa",
    "GHANA": "Africa", "TANZANIA": "Africa", "ALGERIA": "Africa",
    "TUNISIA": "Africa", "SUDAN": "Africa",
    # Asia (East/Southeast Asia plus Caspian and Central Asia)
    "CHINA": "Asia", "JAPAN": "Asia", "SOUTH KOREA": "Asia",
    "PHILIPPINES": "Asia", "TAIWAN": "Asia", "THAILAND": "Asia",
    "MALAYSIA": "Asia", "VIETNAM": "Asia", "INDONESIA": "Asia",
    "SINGAPORE": "Asia", "HONG KONG": "Asia", "AZERBAIJAN": "Asia",
    "KAZAKHSTAN": "Asia", "UZBEKISTAN": "Asia",
    # Eastern Europe
    "RUSSIAN FEDERATION": "Eastern Europe", "POLAND": "Eastern Europe",
    "ROMANIA": "Eastern Europe", "CZECHIA": "Eastern Europe",
    "BULGARIA": "Eastern Europe", "UKRAINE": "Eastern Europe",
    "HUNGARY": "Eastern Europe", "ARMENIA": "Eastern Europe",
    "GEORGIA": "Eastern Europe", "SERBIA": "Eastern Europe",
    "CROATIA": "Eastern Europe", "SLOVAKIA": "Eastern Europe",
    # Western Europe (includes Turkije per the SCAD buckets)
    "TURKEY": "Western Europe", "UNITED KINGDOM": "Western Europe",
    "FRANCE": "Western Europe", "GERMANY": "Western Europe",
    "ITALY": "Western Europe", "SPAIN": "Western Europe",
    "NETHERLANDS": "Western Europe", "BELGIUM": "Western Europe",
    "SWITZERLAND": "Western Europe", "AUSTRIA": "Western Europe",
    "IRELAND": "Western Europe", "PORTUGAL": "Western Europe",
    "SWEDEN": "Western Europe", "DENMARK": "Western Europe",
    "NORWAY": "Western Europe", "FINLAND": "Western Europe",
    "GREECE": "Western Europe", "LUXEMBOURG": "Western Europe",
    # North America
    "UNITED STATES OF AMERICA": "North America", "CANADA": "North America",
    "MEXICO": "North America",
    # Australasia
    "AUSTRALIA": "Australasia", "NEW ZEALAND": "Australasia",
    # South America
    "BRAZIL": "South America", "ARGENTINA": "South America",
    "CHILE": "South America", "COLOMBIA": "South America",
    "PERU": "South America",
}

# Headline connecting hubs used by the hub-exposure bound in the diagnostic.
# These are the departure cities that carry the most connecting traffic filed
# under a hub market: Indian, Egyptian and CIS passengers connecting through
# Doha, Cairo, Baku and Istanbul respectively.
HUB_CITIES = ["Doha", "Cairo", "Baku", "Istanbul"]


def region_of(country: str) -> str:
    """SCAD region for an upper-cased, stripped country name."""
    return COUNTRY_REGION[country.upper().strip()]


def arrivals() -> pd.DataFrame:
    """Long-form Table 6: region, year, arrivals, including the 'Total' row."""
    wb = openpyxl.load_workbook(str(SCAD_KTS), read_only=True, data_only=True)
    ws = wb[SHEET]
    rows = [list(r) for r in ws.iter_rows(values_only=True)][YEAR_ROW:(LAST_DATA_ROW + 1)]
    wb.close()

    # Layout guard: this is an official workbook, pin it rather than drift.
    year_row = rows[0]
    got_years = [year_row[c] for c in range(FIRST_YEAR_COL, FIRST_YEAR_COL + 5)]
    assert got_years == YEARS, f"SCAD Table 6 year row changed: {got_years}"
    got_regions = [str(rows[i][REGION_COL]).strip()
                   for i in range(1, len(rows))]
    assert got_regions == ["Total"] + REGIONS, f"SCAD Table 6 regions changed: {got_regions}"

    out = []
    for i in range(1, len(rows)):
        region = str(rows[i][REGION_COL]).strip()
        for j, year in enumerate(YEARS):
            val = rows[i][FIRST_YEAR_COL + j]
            assert isinstance(val, (int, float)), (region, year, val)
            out.append({"region": region, "year": year, "arrivals": float(val)})
    return pd.DataFrame(out)


def total(year: int) -> float:
    """Total arrivals in a year, from the Table 6 total row."""
    return float(arrivals().query("region == 'Total' and year == @year")["arrivals"].iloc[0])


def main() -> None:
    df = arrivals()
    piv = df.pivot(index="region", columns="year", values="arrivals")
    print("SCAD KTS 2025, Table 6 — arrivals at Zayed by region of embarkation")
    print(piv.round(0).to_string(float_format=lambda v: f"{v:,.0f}"))
    print("\nRegions used by the mapping:", len(REGIONS))
    print("Ambiguous mappings on record: EGYPT->Middle East, CAUCASUS/CENTRAL")
    print("->Eastern Europe, MEXICO->North America. Total row excluded from REGIONS.")


if __name__ == "__main__":
    main()