"""Cross-check the 2025 snapshot market mix against two independent sources.

Three views of "where does Abu Dhabi's traffic come from" exist in this repo,
and they were built from unrelated pipelines:

  1. the 2025 adsb-exchange snapshot (airborne, one day per month)
  2. the 2026 OpenSky arrival extract (airborne, three months, coverage-gated)
  3. the DCT hotel reports (who actually checked in, not who flew)

Agreement between 1 and 2 supports the data. Disagreement with 3 is expected
and informative, because hotels see visitors rather than movements. The point
of this script is to make the disagreement measurable instead of assumed.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# paths first: importing it is what puts pipeline/aviation on sys.path, where
# build_adsb_2025 lives. Reversing these two lines fails to import.
from paths import (  # noqa: E402
    ADSB_LEGS,
    DCT_GUESTS_NATIONALITY,
    OURAIRPORTS,
    OPENSKY_CSV,
)
from build_adsb_2025 import MANUAL_ICAO_COUNTRY  # noqa: E402

ISO_TO_NAME = {
    "IN": "India", "QA": "Qatar", "AE": "UAE", "GB": "UK", "DE": "Germany",
    "US": "USA", "ES": "Spain", "KW": "Kuwait", "JO": "Jordan", "IT": "Italy",
    "OM": "Oman", "PK": "Pakistan", "TH": "Thailand", "IL": "Israel",
    "BH": "Bahrain", "CH": "Switzerland", "RU": "Russia", "AU": "Australia",
    "FR": "France", "NL": "Netherlands", "TR": "Turkey", "EG": "Egypt",
    "SA": "Saudi Arabia", "CN": "China", "NG": "Nigeria", "ZA": "South Africa",
    "IR": "Iran", "BD": "Bangladesh", "MY": "Malaysia", "PH": "Philippines",
    "KR": "South Korea", "JP": "Japan", "SG": "Singapore", "HK": "Hong Kong",
    "ID": "Indonesia", "VN": "Vietnam", "GR": "Greece", "SE": "Sweden",
    "NO": "Norway", "DK": "Denmark", "FI": "Finland", "PL": "Poland",
    "BE": "Belgium", "AT": "Austria", "IE": "Ireland", "PT": "Portugal",
    "CZ": "Czechia", "RO": "Romania", "UA": "Ukraine", "KZ": "Kazakhstan",
    "UZ": "Uzbekistan", "MA": "Morocco", "DZ": "Algeria", "TN": "Tunisia",
    "KE": "Kenya", "ET": "Ethiopia", "MV": "Maldives", "MV ": "Maldives",
    "LK": "Sri Lanka", "NP": "Nepal", "MM": "Myanmar", "KH": "Cambodia",
    "LA": "Laos", "MO": "Macau", "TW": "Taiwan", "US ": "USA",
    "GB ": "UK", "RU ": "Russia", "NZ": "New Zealand", "CL": "Chile",
    "BR": "Brazil", "MX": "Mexico", "AR": "Argentina", "CA": "Canada",
}


def name_for(iso):
    if not isinstance(iso, str):
        return None
    return ISO_TO_NAME.get(iso.strip())


def airport_country_map():
    ap = pd.read_csv(OURAIRPORTS, usecols=["icao_code", "iso_country"], dtype="string")
    ap = ap[ap.icao_code.notna() & ~ap.icao_code.duplicated()]
    return ap.set_index("icao_code").iso_country


def mix_2025():
    df = pd.read_csv(ADSB_LEGS, low_memory=False)
    df = df[~df.is_local_leg]
    n = len(df)
    df = df[df.origin_country != "AE"]
    return df.origin_country.value_counts(), n


def mix_2026():
    df = pd.read_csv(OPENSKY_CSV, low_memory=False)
    df = df[df.direction == "arrival"]
    if "coverage_flag" in df.columns:
        df = df[df.coverage_flag.astype(str).str.lower() == "ok"]
    if "origin_known" in df.columns:
        df = df[df.origin_known == True]
    ctry = airport_country_map()
    df = df.assign(country=df.origin.map(ctry))
    df["country"] = df.apply(
        lambda r: MANUAL_ICAO_COUNTRY.get(r["origin"], r["country"]), axis=1)
    df = df[df.country.notna() & (df.country != "AE")]
    return df.country.value_counts(), len(df)


def mix_dct(years=(2022, 2023)):
    df = pd.read_csv(DCT_GUESTS_NATIONALITY, low_memory=False)
    df = df[(df.year.isin(years)) & (df.nationality.notna())]
    df = df[~df.nationality.isin(["TOTAL", "Total", "Other"])]
    return df.nationality.value_counts(), len(df)


def show(title, counts, total, top=15):
    print(f"\n=== {title} ===  (n={total})")
    share = (counts / counts.sum() * 100).round(1)
    for k, v in counts.head(top).items():
        label = name_for(k) or k
        print(f"  {label:15s} {v:5d}  {share[k]:5.1f}%")


def main():
    c25, n25 = mix_2025()
    c26, n26 = mix_2026()
    cdct, ndct = mix_dct()

    show("2025 adsb-exchange snapshot, inbound international", c25, n25)
    show("2026 OpenSky extract, coverage-gated arrivals", c26, n26)
    show("DCT hotel nationality, 2022-2023", cdct, ndct)

    top = 12
    a = (c25 / c25.sum() * 100).head(top).rename("pct_2025_adsb")
    b = (c26 / c26.sum() * 100).head(top).rename("pct_2026_opensky")
    cmp = pd.concat([a, b], axis=1).fillna(0).round(1)
    cmp["diff"] = (cmp.pct_2025_adsb - cmp.pct_2026_opensky).round(1)
    cmp.index = [name_for(i) or i for i in cmp.index]
    print(f"\n=== market share comparison, top {top} ===")
    print(cmp.to_string())

    common = cmp.dropna()
    corr = common.pct_2025_adsb.corr(common.pct_2026_opensky)
    print(f"\ncorrelation of shares across the two airborne sources: {corr:.3f}")
    print(f"mean absolute share difference: {common['diff'].abs().mean():.1f} pp")


if __name__ == "__main__":
    main()
