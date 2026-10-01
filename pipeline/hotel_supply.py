"""Build the airport passenger-movement and Etihad traffic CSVs.

Both builders transcribe official published tables by hand; neither estimates or
interpolates a value. They are grouped in one module because they frame the same
market from two sources and cross-check each other: AUH airport movements give
the market frame, Etihad's own releases give airline-level detail and an
independent read on the same passengers.

    .venv/bin/python pipeline/hotel_supply.py            # both
    .venv/bin/python pipeline/hotel_supply.py airport    # movements only
    .venv/bin/python pipeline/hotel_supply.py etihad     # Etihad only

Despite the filename this does not touch hotel supply. It produces
data/processed/pax_movements.csv and data/processed/etihad_passengers.csv. The
name is kept from the agreed layout; an accurate one would be
pipeline/passenger_movements.py.
"""

import csv
import sys
from calendar import monthrange
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import ETIHAD_PASSENGERS, PAX_MOVEMENTS  # noqa: E402




AIR_SOURCES = {
    "SCAD_AIR_2013": (
        "SCAD - Statistics of Air, Water and Land Transport 2013 (PDF)",
        "https://www.scad.gov.ae/documents/20122/2311280/Statistics%20of%20Air%20Water%20Land%20transport%202013%20en%20v1.pdf/75f38ede-1ed8-8837-5aca-ffab153ece71",
        "Underlying source: Abu Dhabi Airports Company",
    ),
    "SYB2016": (
        "SCAD - Statistical Yearbook of Abu Dhabi 2016 (PDF), tables 2.7.6/2.7.7",
        "https://s3.amazonaws.com/rgi-documents/2fcca7618e691dabc9c25bc66eea2ee9d40efff8.pdf",
        "Underlying source: Abu Dhabi Airports Company",
    ),
    "SYB2020": (
        "SCAD - Statistical Yearbook of Abu Dhabi 2020 (PDF), tables 2.7.6/2.7.7",
        "https://scad.gov.ae/documents/20122/1043412/Statistical%20Yearbook%20of%20Abu%20Dhabi_2020_Annual_Yearly_en.pdf/68b923b8-e4f2-3fe5-f922-11f60f9ddff4",
        "Underlying source: Abu Dhabi Airports Company",
    ),
    "TS2018": (
        "SCAD - Transport Statistics 2015-2018 (XLSX), tables 1.1/1.2",
        "https://scad.gov.ae/documents/20122/2311280/Transport%2520Statistics_2018_Annual_Yearly_en.xlsx/e189ae42-2214-35a7-cfe7-badc54fc4d21",
        "Arabic-language workbook; underlying source: Abu Dhabi Airports Company",
    ),
    "KTS2025": (
        "SCAD - Key Transport Statistics 2025 (XLSX), tables 1/2",
        "https://www.scad.gov.ae/documents/20122/0/Publication_ar_PID_344_2025_Annual_Yearly_ar.xlsx/40063176-6ed9-e21c-8d81-a1804ac8a445",
        "Bilingual workbook; underlying source: Abu Dhabi Airports Company",
    ),
    "ADA_2024FY": (
        "Abu Dhabi Airports - welcome recording-breaking 29m+ passengers in 2024",
        "https://www.adairports.ae/en/PressRelease/2025/02/abu-dhabi-airports-welcomes-recording-breaking-29m-passengers-in-2024",
        "Five-airport group (AUH, AAN, AZI, ZDY, XSB)",
    ),
    "ADA_2025FY": (
        "Abu Dhabi Airports - closes 2025 with record traffic",
        "https://www.adairports.ae/en/pressrelease/2026/01/abu-dhabi-airports-closes-2025-with-record-traffic-as--zayed-international-emerges-as-emea",
        "Five-airport group (AUH, AAN, AZI, ZDY, XSB)",
    ),
    "ADA_H1_2025": (
        "Abu Dhabi Airports - H1 2025 results highlight 17 straight quarters",
        "https://www.adairports.ae/en/pressrelease/2025/07/abu-dhabi-airports-h1-2025-results-highlight-17-straight-quarters",
        "Five-airport group (AUH, AAN, AZI, ZDY, XSB)",
    ),
    "ADA_9M2024": (
        "Abu Dhabi Airports - reports remarkable passenger growth (9M 2024)",
        "https://www.adairports.ae/en/PressRelease/2024/10/Abu-Dhabi-Airports-Reports-Remarkable-Passenger-Growth",
        "Five-airport group (AUH, AAN, AZI, ZDY, XSB)",
    ),
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def mstart(y, m):
    return f"{y:04d}-{m:02d}-01"


def mend(y, m):
    return f"{y:04d}-{m:02d}-{monthrange(y, m)[1]:02d}"


# --- Zayed International (AUH) annual, per SCAD/ADA ------------------------
# year: (arrivals, departures, transit, movements, source_key)
AUH_ANNUAL = {
    2014: (9797879, 9782403, 284845, 186207, "SCAD_AIR_2013"),
    2015: (11564563, 11558557, 163512, 169989, "SYB2016"),
    2016: (12139014, 12213371, 129734, 168535, "SYB2020"),
    2017: (11678045, 11717679, 25869, 154543, "SYB2020"),
    2018: (10633563, 10685185, 498, 133999, "SYB2020"),
    2019: (10564367, 10592215, 876, 122667, "SYB2020"),
    2021: (2515116, 2741811, 0, None, "KTS2025"),
    2022: (7821770, 7720802, 3923, None, "KTS2025"),
    2023: (11117530, 11254395, 1011, None, "KTS2025"),
    2024: (14522288, 14318969, 2149, None, "KTS2025"),
    2025: (16413437, 16075287, 1211, None, "KTS2025"),
}

# --- Zayed International monthly aircraft movements ------------------------
# year: (12 monthly values, source_key)
AUH_MOV_MONTH = {
    2015: ([14117, 12807, 14414, 13982, 14447, 14117, 14689, 14964, 14130,
            14304, 13753, 14265], "SYB2016"),
    2016: ([14233, 13092, 14117, 13718, 14137, 13538, 14478, 14681, 14201,
            14402, 13853, 14085], "SYB2020"),
    2017: ([14190, 12754, 13959, 13249, 13238, 12272, 12871, 13091, 12364,
            12247, 12063, 12245], "SYB2020"),
    2018: ([12085, 10634, 11733, 11205, 11365, 11058, 11821, 11618, 10926,
            10522, 10224, 10808], "SYB2020"),
    2019: ([10572, 9282, 10148, 10119, 10050, 9890, 10854, 10987, 10279,
            10401, 9925, 10160], "SYB2020"),
    2021: ([4977, 4719, 5587, 5476, 5317, 5330, 6075, 6242, 6091, 6700,
            6832, 7771], "KTS2025"),
    2022: ([7427, 6831, 7867, 7940, 9493, 9470, 10323, 10202, 9467, 10134,
            10565, 10817], "KTS2025"),
    2023: ([10812, 9889, 10834, 10818, 11477, 11497, 12221, 12446, 12046,
            12659, 12822, 13704], "KTS2025"),
    2024: ([13901, 13040, 13441, 13588, 13969, 14066, 15003, 15440, 14712,
            15060, 14749, 15966], "KTS2025"),
    2025: ([15834, 14260, 15522, 15260, 15618, 15131, 16386, 16722, 15019,
            16038, 16050, 17530], "KTS2025"),
}

# --- Zayed International monthly passengers, from TS2018 table 1.2 ---------
# year -> list of 12 x (arrivals, departures, transit)
AUH_PAX_MONTH = {
    2015: [(946238, 922778, 18615), (852684, 846360, 16496),
           (936350, 963335, 18254), (941263, 909831, 13853),
           (928002, 936336, 13102), (904124, 930716, 13240),
           (973221, 1113108, 14600), (1173334, 1064073, 17402),
           (1006365, 987290, 12093), (974925, 936387, 8699),
           (876943, 896459, 8127), (1051114, 1051884, 9031)],
    2016: [(1086189, 1058445, 9607), (931105, 925728, 7109),
           (982712, 1034671, 8459), (985151, 963797, 11998),
           (987341, 982110, 12559), (884821, 964583, 11974),
           (1074022, 1180504, 13046), (1199616, 1093844, 13372),
           (1049871, 1044044, 12045), (963101, 928392, 12063),
           (921151, 931364, 7692), (1073934, 1105889, 9810)],
    2017: [(1116359, 1084536, 8620), (951013, 951971, 7642),
           (984623, 1040084, 7935), (1050520, 1019835, 409),
           (967820, 974139, 0), (827656, 935077, 370),
           (1031949, 1106219, 290), (1090266, 1071565, 0),
           (989632, 877587, 0), (870120, 834231, 0),
           (826337, 844913, 299), (971750, 977522, 304)],
    2018: [(969903, 961036, 0), (838622, 843873, 0),
           (920773, 957967, 0), (932558, 909224, 0),
           (844086, 847977, 276), (805064, 869243, 0),
           (943973, 1055514, 222), (1041471, 998429, 0),
           (895230, 809911, 0), (786870, 769587, 0),
           (770887, 786540, 0), (884126, 875884, 0)],
}

# --- Five-airport group aggregates, from Abu Dhabi Airports releases -------
# (period_start, period_end, granularity, metric, value, unit, scope, source, note)
FIVE_AIRPORTS = [
    ("2024-01-01", "2024-06-30", "H1", "movements_aircraft", 84286,
     "movements", "auh", "ADA_H1_2025", "AUH aircraft movements; H1 2024 "
     "comparator quoted in the H1 2025 release"),
    ("2024-01-01", "2024-09-30", "9M", "pax_total", 21700000, "passengers",
     "five_airports", "ADA_9M2024", "Group total reported as 21.7m passengers"),
    ("2024-01-01", "2024-09-30", "9M", "movements_aircraft", 185485,
     "movements", "five_airports", "ADA_9M2024", "Group total, year-to-date"),
    ("2023-01-01", "2023-12-31", "year", "movements_aircraft", 226362,
     "movements", "five_airports", "ADA_2024FY", "Prior-year five-airport-group "
     "comparator quoted in the 2024 full-year release"),
    ("2024-01-01", "2024-12-31", "year", "pax_total", 29400000, "passengers",
     "five_airports", "ADA_2024FY", "Group total reported as 29.4m passengers"),
    ("2024-01-01", "2024-12-31", "year", "movements_aircraft", 249747,
     "movements", "five_airports", "ADA_2024FY", "Group total"),
    ("2025-01-01", "2025-06-30", "H1", "pax_total", 15800000, "passengers",
     "five_airports", "ADA_H1_2025", "Group total reported as 15.8m passengers"),
    ("2025-01-01", "2025-06-30", "H1", "movements_aircraft", 133533,
     "movements", "five_airports", "ADA_H1_2025", "Group total, described as "
     "flights"),
    ("2025-10-01", "2025-12-31", "Q4", "pax_total", 8590000, "passengers",
     "auh", "ADA_2025FY", "AUH Q4 2025 pax_total"),
    ("2025-01-01", "2025-12-31", "year", "pax_total", 33000000, "passengers",
     "five_airports", "ADA_2025FY", "Group total reported as 'more than 33m' "
     "passengers"),
]

TRANSIT_NOTE = (
    "Terminal-1 transit only. SCAD/ADA-counted transit is not the same as "
    "connecting/transfer passengers, which are reported separately by Abu "
    "Dhabi Airports."
)
AUH_NOTE = (
    "Zayed International Airport; source states these figures include Al "
    "Bateen Executive Airport data."
)


def build_airport_movements():
    rows = []

    for year, (arr, dep, tr, mov, skey) in sorted(AUH_ANNUAL.items()):
        name, url, extra = AIR_SOURCES[skey]
        note = f"{AUH_NOTE} {extra}"
        base = dict(period_start=f"{year}-01-01", period_end=f"{year}-12-31",
                    granularity="year", scope="auh", source_name=name,
                    source_url=url)
        rows.append({**base, "metric": "pax_arrivals", "value": arr,
                     "unit": "passengers",
                     "note": f"Annual arrivals. {note}"})
        rows.append({**base, "metric": "pax_departures", "value": dep,
                     "unit": "passengers",
                     "note": f"Annual departures. {note}"})
        rows.append({**base, "metric": "pax_transit", "value": tr,
                     "unit": "passengers",
                     "note": f"Annual transit. {TRANSIT_NOTE} {note}"})
        rows.append({**base, "metric": "pax_total", "value": arr + dep + tr,
                     "unit": "passengers",
                     "note": f"Arrivals + departures + transit. {note}"})
        if mov is not None:
            rows.append({**base, "metric": "movements_aircraft", "value": mov,
                         "unit": "movements",
                         "note": f"Annual aircraft movements. {note}"})
        elif year in AUH_MOV_MONTH:
            total = sum(AUH_MOV_MONTH[year][0])
            rows.append({**base, "metric": "movements_aircraft", "value": total,
                         "unit": "movements",
                         "note": "Annual aircraft movements computed as the "
                                 "sum of the 12 monthly totals in the source. "
                                 f"{note}"})

    for year, (vals, skey) in sorted(AUH_MOV_MONTH.items()):
        name, url, extra = AIR_SOURCES[skey]
        for i, v in enumerate(vals, start=1):
            rows.append(dict(
                period_start=mstart(year, i), period_end=mend(year, i),
                granularity="month", scope="auh",
                metric="movements_aircraft", value=v, unit="movements",
                source_name=name, source_url=url,
                note=f"Aircraft movements, {MONTHS[i-1]} {year}. {AUH_NOTE} "
                     f"{extra}"))

    for year, monthly in sorted(AUH_PAX_MONTH.items()):
        name, url, extra = AIR_SOURCES["TS2018"]
        for i, (arr, dep, tr) in enumerate(monthly, start=1):
            base = dict(period_start=mstart(year, i), period_end=mend(year, i),
                        granularity="month", scope="auh", source_name=name,
                        source_url=url)
            tail = f"{MONTHS[i-1]} {year}. {AUH_NOTE} {extra}"
            rows.append({**base, "metric": "pax_arrivals", "value": arr,
                         "unit": "passengers",
                         "note": f"Monthly arrivals. {tail}"})
            rows.append({**base, "metric": "pax_departures", "value": dep,
                         "unit": "passengers",
                         "note": f"Monthly departures. {tail}"})
            rows.append({**base, "metric": "pax_transit", "value": tr,
                         "unit": "passengers",
                         "note": f"Monthly transit. {TRANSIT_NOTE} {tail}"})
            rows.append({**base, "metric": "pax_total",
                         "value": arr + dep + tr, "unit": "passengers",
                         "note": f"Arrivals + departures + transit. {tail}"})

    for ps, pe, gran, metric, val, unit, scope, skey, note in FIVE_AIRPORTS:
        name, url, extra = AIR_SOURCES[skey]
        rows.append(dict(
            period_start=ps, period_end=pe, granularity=gran, scope=scope,
            metric=metric, value=val, unit=unit, source_name=name,
            source_url=url, note=f"{note} {extra}"))

    fields = ["period_start", "period_end", "granularity", "scope", "metric",
              "value", "unit", "source_name", "source_url", "note"]
    rows.sort(key=lambda r: (r["period_start"], r["period_end"],
                             r["scope"], r["granularity"], r["metric"]))
    path = PAX_MOVEMENTS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")

    # --- cross-checks against totals printed in the sources -----------------
    print("\nAUH monthly movement sums vs published/derived annual totals:")
    for year in sorted(AUH_MOV_MONTH):
        sum12 = sum(AUH_MOV_MONTH[year][0])
        mov = AUH_ANNUAL[year][3]
        if mov is not None:
            flag = "OK " if sum12 == mov else "MISMATCH"
            print(f"  {year}: sum={sum12:,} published={mov:,} {flag}")
        else:
            print(f"  {year}: sum={sum12:,} annual row derived from monthly sum")

    print("\nAUH monthly pax sums vs published annual totals:")
    for year, monthly in sorted(AUH_PAX_MONTH.items()):
        a = sum(x[0] for x in monthly)
        d = sum(x[1] for x in monthly)
        t = sum(x[2] for x in monthly)
        pa, pd, pt, _, _ = AUH_ANNUAL[year]
        print(f"  {year}: arr {a:,}/{pa:,} {'OK' if a == pa else 'DIFF'}"
              f"  dep {d:,}/{pd:,} {'OK' if d == pd else 'DIFF'}"
              f"  transit {t:,}/{pt:,} {'OK' if t == pt else 'DIFF'}")

    print("\nAUH annual pax_total:")
    for year, (arr, dep, tr, _, _) in sorted(AUH_ANNUAL.items()):
        print(f"  {year}: {arr + dep + tr:,}")




ETIHAD_SOURCES = {
    "EY_DEC24": (
        "Etihad Airways - Etihad reports December 2024 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-december-2024-traffic-statistics",
    ),
    "EY_FY21": (
        "Etihad Airways - Transformation continues to deliver results "
        "(2021 results release)",
        "https://www.etihad.com/en-ch/news/etihad-airways-transformation-continues-to-deliver-results-with-41-improvement-on-precovid-performance",
    ),
    "EY_FY25": (
        "Etihad Airways - Highest-ever profit of AED 2.6 billion in 2025 "
        "results release",
        "https://www.etihad.com/en-ae/news/etihad-airways-delivers-record-aed-26-billion-us--698-million-profit-in-2025-marking-strongest-performance-in-the-airlines-history",
    ),
    "EY_2025_01": (
        "Etihad Airways - Etihad reports January 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-january-2025-traffic-statistics",
    ),
    "EY_2025_02": (
        "Etihad Airways - Etihad reports February 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-february-2025-traffic-statistics",
    ),
    "EY_2025_03": (
        "Etihad Airways - Etihad reports March 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-march-2025-traffic-statistics",
    ),
    "EY_2025_04": (
        "Etihad Airways - Etihad reports April 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-april-2025-traffic-statistics",
    ),
    "EY_2025_05": (
        "Etihad Airways - Etihad reports May 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-may-2025-traffic-statistics",
    ),
    "EY_2025_06": (
        "Etihad Airways - Etihad reports June 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-june-2025-traffic-statistics",
    ),
    "EY_2025_07": (
        "Etihad Airways - Etihad reports July 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-july-2025-traffic-statistics",
    ),
    "EY_2025_08": (
        "Etihad Airways - Etihad reports August 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-august-2025-traffic-statistics",
    ),
    "EY_2025_09": (
        "Etihad Airways - Etihad reports September 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-september-2025-traffic-statistics",
    ),
    "EY_2025_10": (
        "Etihad Airways - Etihad reports October 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-october-2025-traffic-statistics",
    ),
    "EY_2025_11": (
        "Etihad Airways - Etihad reports November 2025 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-november-2025-traffic-statistics",
    ),
    "EY_2026_01": (
        "Etihad Airways - Etihad reports January 2026 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-january-2026-traffic-statistics",
    ),
    "EY_2026_02": (
        "Etihad Airways - Etihad reports February 2026 traffic statistics",
        "https://www.etihad.com/en/news/etihad-reports-february-2026-traffic-statistics",
    ),
}

# Full-year: (year, pax, load_factor_pct, source, note)
FY = [
    (2020, 4150000, 52.9, "EY_FY21",
     "Comparator column of the 2021 results release; 4.15 million passengers, "
     "52.9% load factor"),
    (2021, 3460000, 39.6, "EY_FY21",
     "2021 results release; text states 3.5 million passengers, table value "
     "3.46 million, load factor 39.6%"),
    (2023, 14000000, 86, "EY_DEC24",
     "'2023 YTD' column of the December 2024 release; 14.0 million passengers, "
     "86% load factor"),
    (2024, 18500000, 87, "EY_DEC24",
     "'2024 YTD' column of the December 2024 release; 18.5 million passengers, "
     "87% load factor"),
    (2025, 22400000, 88.3, "EY_FY25",
     "Full-year results release; 22.4 million passengers, 88.3% load factor"),
]

# Monthly: (year, month, pax_m, load_factor_pct, source)
# pax_m is the rounded figure in millions printed by the source.
MONTHLY = [
    (2023, 12, 1.4, 84, "EY_DEC24"),
    (2024, 1, 1.5, 86, "EY_2025_01"),
    (2024, 2, 1.4, 89, "EY_2025_02"),
    (2024, 3, 1.4, 84, "EY_2025_03"),
    (2024, 4, 1.4, 84, "EY_2025_04"),
    (2024, 5, 1.5, 84, "EY_2025_05"),
    (2024, 6, 1.5, 86, "EY_2025_06"),
    (2024, 7, 1.7, 89, "EY_2025_07"),
    (2024, 8, 1.7, 89, "EY_2025_08"),
    (2024, 9, 1.6, 88, "EY_2025_09"),
    (2024, 10, 1.6, 87, "EY_2025_10"),
    (2024, 11, 1.6, 88, "EY_2025_11"),
    (2024, 12, 1.7, 87, "EY_DEC24"),
    (2025, 1, 1.7, 89, "EY_2025_01"),
    (2025, 2, 1.6, 89, "EY_2025_02"),
    (2025, 3, 1.6, 82, "EY_2025_03"),
    (2025, 4, 1.7, 89, "EY_2025_04"),
    (2025, 5, 1.7, 87, "EY_2025_05"),
    (2025, 6, 1.8, 88, "EY_2025_06"),
    (2025, 7, 2.0, 90, "EY_2025_07"),
    (2025, 8, 2.0, 91, "EY_2025_08"),
    (2025, 9, 1.9, 89, "EY_2025_09"),
    (2025, 10, 2.0, 88, "EY_2025_10"),
    (2025, 11, 2.1, 89, "EY_2025_11"),
    (2026, 1, 2.2, 89.9, "EY_2026_01"),
    (2026, 2, 1.9, 89.4, "EY_2026_02"),
]

# YTD: (year, last_month, pax_m, load_factor_pct, source)
YTD = [
    (2024, 1, 1.5, 86, "EY_2025_01"),
    (2024, 2, 2.9, 87, "EY_2025_02"),
    (2024, 3, 4.3, 86, "EY_2025_03"),
    (2024, 4, 5.7, 86, "EY_2025_04"),
    (2024, 5, 7.2, 85, "EY_2025_05"),
    (2024, 6, 8.7, 85, "EY_2025_06"),
    (2024, 7, 10.4, 86, "EY_2025_07"),
    (2024, 8, 12.0, 86, "EY_2025_08"),
    (2024, 9, 13.6, 87, "EY_2025_09"),
    (2024, 10, 15.2, 87, "EY_2025_10"),
    (2024, 11, 16.8, 87, "EY_2025_11"),
    (2025, 1, 1.7, 89, "EY_2025_01"),
    (2025, 2, 3.3, 89, "EY_2025_02"),
    (2025, 3, 5.0, 87, "EY_2025_03"),
    (2025, 4, 6.7, 87, "EY_2025_04"),
    (2025, 5, 8.4, 87, "EY_2025_05"),
    (2025, 6, 10.2, 87, "EY_2025_06"),
    (2025, 7, 12.2, 88, "EY_2025_07"),
    (2025, 8, 14.2, 88, "EY_2025_08"),
    (2025, 9, 16.1, 88, "EY_2025_09"),
    (2025, 10, 18.1, 88, "EY_2025_10"),
    (2025, 11, 20.2, 88, "EY_2025_11"),
    (2026, 2, 4.2, 89.7, "EY_2026_02"),
]

ROUNDED = (
    "Passenger figures as printed by Etihad are rounded (to 0.1 million); "
    "monthly rounded figures do not sum exactly to the reported annual total."
)


def build_etihad():
    rows = []

    for year, pax, lf, skey, note in FY:
        name, url = ETIHAD_SOURCES[skey]
        rows.append(dict(
            period_start=f"{year}-01-01", period_end=f"{year}-12-31",
            granularity="year", metric="pax_number", value=pax,
            source_name=name, source_url=url,
            note=f"Full-year passengers. {note}. {ROUNDED}"))

    for year, month, pax_m, lf, skey in MONTHLY:
        name, url = ETIHAD_SOURCES[skey]
        ps = f"{year:04d}-{month:02d}-01"
        pe = f"{year:04d}-{month:02d}-{monthrange(year, month)[1]:02d}"
        rows.append(dict(
            period_start=ps, period_end=pe, granularity="month",
            metric="pax_number", value=int(round(pax_m * 1e6)),
            source_name=name, source_url=url,
            note=f"Monthly passengers reported as {pax_m:g} million. {ROUNDED}"))
        rows.append(dict(
            period_start=ps, period_end=pe, granularity="month",
            metric="load_factor_pct", value=lf,
            source_name=name, source_url=url,
            note="Monthly passenger load factor as printed by source."))

    for year, last_month, pax_m, lf, skey in YTD:
        name, url = ETIHAD_SOURCES[skey]
        ps = f"{year:04d}-01-01"
        pe = f"{year:04d}-{last_month:02d}-"
        pe += f"{monthrange(year, last_month)[1]:02d}"
        rows.append(dict(
            period_start=ps, period_end=pe, granularity="YTD",
            metric="pax_number", value=int(round(pax_m * 1e6)),
            source_name=name, source_url=url,
            note=f"Year-to-date passengers reported as {pax_m:g} million "
                 f"through {last_month:02d}-{year}. {ROUNDED}"))
        rows.append(dict(
            period_start=ps, period_end=pe, granularity="YTD",
            metric="load_factor_pct", value=lf,
            source_name=name, source_url=url,
            note="Year-to-date passenger load factor as printed by source."))

    fields = ["period_start", "period_end", "granularity", "metric",
              "value", "source_name", "source_url", "note"]
    rows.sort(key=lambda r: (r["period_start"], r["period_end"],
                             r["granularity"], r["metric"]))
    path = ETIHAD_PASSENGERS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")

    byyear = {}
    for y, m, pax_m, lf, s in MONTHLY:
        byyear.setdefault(y, []).append(pax_m)
    fy_pax = {yy: p / 1e6 for (yy, p, l, s, n) in FY}
    print("\nSum of rounded monthly pax (millions) vs reported annual:")
    for y, vals in sorted(byyear.items()):
        print(f"  {y}: monthly sum {sum(vals):.1f} vs FY "
              f"{fy_pax.get(y, 0):.1f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "airport"):
        build_airport_movements()
    if which in ("all", "etihad"):
        build_etihad()
