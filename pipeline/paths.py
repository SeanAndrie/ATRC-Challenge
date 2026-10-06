"""Single source of truth for every filesystem path in the pipeline.

Everything else imports from here. The previous arrangement had each script
compute its own repo root with `Path(__file__).resolve().parents[N]`, where the
correct N depended on how deep the script happened to live. Moving a script up or
down a directory silently produced a wrong-but-plausible path, which is the worst
kind of failure: it does not raise, it just reads the wrong directory or finds
nothing.

Anchoring once, here, means a script's location stops mattering. Import this
rather than recomputing a root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# pipeline/paths.py -> pipeline/ -> repo root.
ROOT = Path(__file__).resolve().parents[1]

# Data tiers. raw is downloaded and never written by a script; interim is
# extracted but not yet validated; processed is validated and analysis-ready.
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROC = ROOT / "data" / "processed"

# --- raw ---------------------------------------------------------------------
# DCT Abu Dhabi monthly and annual hotel reports, as downloaded, plus the
# download manifest and the PDF size record written at extraction time.
DCT_PDFS = RAW / "dct_pdfs"
DCT_MANIFEST = DCT_PDFS / "manifest.json"
DCT_PDF_SIZES = DCT_PDFS / "pdf_sizes.json"

# SCAD Key Transport Statistics workbook and the 2020 statistical yearbook.
SCAD = RAW / "scad"
SCAD_KTS = SCAD / "KTS2025.xlsx"
SCAD_YEARBOOK = SCAD / "Yearbook2020.pdf"

# One day per month of ADS-B arrivals at AUH, as downloaded. One file per
# month of 2025, named ax_arrivals_YYYYMMDD.csv for the snapshot's first day.
ADS_SNAPSHOTS = RAW / "adsb_snapshots"
ADS_RAW_GLOB = "ax_arrivals_*.csv"

# OpenSky leg-level extracts, both the flat CSV and the day-partitioned parquet.
OPENSKY = RAW / "opensky"
OPENSKY_CSV = OPENSKY / "opensky_omaa_2026-06-20_2026-09-13.csv"
OPENSKY_PARQUET_GLOB = "day=*.parquet"

# The Trino-backed pull, which is the better source and currently unreachable:
# FlightsData4 returns PERMISSION_DENIED pending an approved grant. Kept as a
# distinct directory from OPENSKY so the REST parquet and the Trino parquet are
# never silently mixed. Empty until access is granted.
OPENSKY_TRINO = RAW / "opensky_trino"

# OurAirports, used as the ICAO-to-country bridge.
REFERENCE = RAW / "reference"
OURAIRPORTS = REFERENCE / "ourairports_airports.csv"

# Competition data pack (DCT Abu Dhabi challenge). Licensed for use within that
# competition only, so it stays under data/raw and is never committed; its
# expected files, sizes and checksums are recorded in docs/METHODS.md §2.
COMPETITION = RAW / "competition"
COMPETITION_FLIGHT = COMPETITION / "flight_data.xlsx"
COMPETITION_INTL_TRAIN = COMPETITION / "data international_train.xlsx"
COMPETITION_INTL_TEST = COMPETITION / "data international_test.xlsx"
COMPETITION_DOM_TRAIN = COMPETITION / "data domestic_train.xlsx"
COMPETITION_DOM_TEST = COMPETITION / "data domestic_test.xlsx"
COMPETITION_DICTIONARY = COMPETITION / "Data_Dictionary.pdf"

# --- interim -----------------------------------------------------------------
# pdftotext -layout output, one file per PDF, plus the fingerprint parse.py reads
# to decide which pages a report prints.
DCT_TEXT = INTERIM / "dct_text"
DCT_FINGERPRINT = DCT_TEXT / "text_fingerprint.json"

# --- processed ---------------------------------------------------------------
DCT_KPIS = PROC / "dct_kpis.csv"
DCT_GUESTS_NATIONALITY = PROC / "dct_guests_nationality.csv"
DCT_HOTEL_SUPPLY = PROC / "dct_hotel_supply.csv"
DCT_COVERAGE = PROC / "dct_coverage.csv"
DCT_CROSS_CHECKS = PROC / "dct_cross_checks.csv"
DCT_NOTES_RAW = PROC / "dct_notes_raw.csv"

PAX_MOVEMENTS = PROC / "pax_movements.csv"
PAX_ARRIVALS_ANNUAL = PROC / "pax_arrivals_annual.csv"
ETIHAD_PASSENGERS = PROC / "etihad_passengers.csv"

ADSB_LEGS = PROC / "omaa_arrivals_2025.csv"
ADSB_MONTHLY = PROC / "monthly_summary.csv"
ADSB_MARKETS = PROC / "markets_by_month.csv"
ADSB_TYPES = PROC / "aircraft_types.csv"
ADSB_CAPTURE = PROC / "capture_rate.csv"

# --- code and docs -----------------------------------------------------------
PIPELINE = ROOT / "pipeline"
DCT_PIPELINE = PIPELINE / "dct"
AVIATION = PIPELINE / "aviation"
COMPETITION_PIPELINE = PIPELINE / "competition"
TESTS = PIPELINE / "tests"
NOTEBOOKS = ROOT / "notebooks"
TOOLS = ROOT / "tools"
APP = ROOT / "app"

# The pipeline modules are run as standalone scripts (`.venv/bin/python
# pipeline/dct/dct_parse.py`), not installed as a package, so they import each
# other by bare module name. That needs their directory on sys.path:
#   DCT_PIPELINE for dct_common, dct_period, dct_charts, ...
#   TESTS      for test_retarget_international importing test_market_link
#   AVIATION   for aircraft_seats, shared by the ADS-B builder, its calibration,
#              the market test and the notebook
#   COMPETITION_PIPELINE for load and audit, shared by the P2P-vs-PAX test, the
#              chain model and the competition notebook
# Adding pipeline/dct to sys.path is also what makes `eras` importable as a
# package, which is how dct_parse reaches the era-specific readers.
#
# Doing this on import means every script that imports paths can import these
# too, which is why it is a side effect rather than a function the caller has to
# remember to call.
for _d in (DCT_PIPELINE, TESTS, AVIATION, COMPETITION_PIPELINE):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# --- outputs -----------------------------------------------------------------
# Generated test reports. These are written by the scripts in pipeline/tests and
# are gitignored: they are findings about the data, not source, and a clean
# clone regenerates them in a second. They live in outputs/ rather than docs/ so
# docs/ holds only hand-written documentation.
OUTPUTS = ROOT / "outputs"
REPORT_MARKET_LINK = OUTPUTS / "market_link_test.md"
REPORT_AGGREGATE_LINK = OUTPUTS / "aggregate_link_test.md"
REPORT_RETARGET = OUTPUTS / "retarget_test.md"
REPORT_P2P_VS_PAX = OUTPUTS / "p2p_vs_pax_test.md"
REPORT_CALENDAR = OUTPUTS / "calendar_events_test.md"

# OpenSky REST credentials, used by tools/fetch_opensky.py and the window-cost tool.
CREDENTIALS = ROOT / "credentials.json"
