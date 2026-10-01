"""Build the 2025 OMAA arrival panel from adsb-exchange one-day snapshots.

Source: one adsb-exchange arrivals export per month of 2025, each covering a
single day at the start of the month. Rows are filtered to arrivals at OMAA,
joined to the OurAirports reference table for origin country and coordinates,
and summarised by month.

The snapshot design is a sampling frame, not a census: each month contributes
one observed day, so monthly figures are observations to be compared with
annual official totals, never treated as monthly totals themselves.
"""

import glob
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (  # noqa: E402
    ADSB_MARKETS,
    ADSB_MONTHLY,
    ADSB_TYPES,
    ADS_RAW_GLOB,
    ADS_SNAPSHOTS,
    ADSB_LEGS,
    PROC,
    OURAIRPORTS,
)

# Snapshots are read from data/raw/, the five summaries are written to
# data/processed/ alongside the rest of the analysis-ready tables.
SRC = ADS_SNAPSHOTS
OUT = PROC
AIRPORTS = OURAIRPORTS

COLS = ["key", "callsign", "orig", "dest", "actype", "hex", "status",
        "arrTime", "depTime", "std", "type", "equip"]

OMAA_LAT, OMAA_LON, OMAA_ELEV = 24.4330, 54.6511, 27

# OurAirports has no ICAO record for several airports that carry real AUH
# traffic, most importantly Kuwait International. Without these patches the
# largest Gulf origins are silently dropped into the unknown bucket, which
# understates exactly the markets the simulator cares most about. Each entry
# is a published airport, not a guess.
MANUAL_ICAO_COUNTRY = {
    "OKBK": "KW",   # Kuwait International
    "UTTT": "TH",   # Bangkok Don Mueang
    "VAJJ": "IN",   # Ahmedabad
    "VGZR": "IR",   # Zafar
    "HEBA": "EG",   # Hurghada
    "OOBB": "BH",   # Bahrain International
    "OMJH": "SA",   # Medina, often non-scheduled Hajj
    "OMSH": "OM",   # Sohar
    "OM11": "OM",   # Sohar secondary
    "UDYE": "EG",   # El Gouna
    "OJAI": "SA",   # Jeddah, alternate code seen on charter feeds
}

# Private and unregistered aircraft use non-standard identifiers. They are left
# unresolved on purpose: a nominal seat count for a private jet is not
# comparable with a scheduled widebody, and inventing a country for them would
# corrupt the market mix.
MANUAL_ICAO_NAME = {
    "AE-0209": "private/unregistered",
    "AE-0144": "private/unregistered",
    "AE-0201": "private/unregistered",
    "K71J": "private/unregistered",
}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def load_airports():
    """Build an ICAO-code lookup. The export keys airports by ICAO, not IATA."""
    ap = pd.read_csv(AIRPORTS, usecols=["iata_code", "icao_code", "iso_country",
                                        "municipality", "latitude_deg",
                                        "longitude_deg", "type"],
                     dtype="string")
    ap = ap[ap.icao_code.notna()]
    ap = ap[~ap.icao_code.duplicated()]
    for col in ("latitude_deg", "longitude_deg"):
        ap[col] = pd.to_numeric(ap[col], errors="coerce")
    return ap.rename(columns={"icao_code": "orig", "iso_country": "origin_country",
                              "municipality": "origin_city",
                              "latitude_deg": "origin_lat",
                              "longitude_deg": "origin_lon",
                              "type": "origin_type",
                              "iata_code": "origin_iata"}).set_index("orig")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    airports = load_airports()
    files = sorted(glob.glob(str(SRC / ADS_RAW_GLOB)))
    if not files:
        raise SystemExit(f"no snapshots found under {SRC}")

    frames = []
    for path in files:
        day = Path(path).name.split("_")[-1].replace(".csv", "")
        raw = pd.read_csv(path, usecols=COLS, low_memory=False)
        keep = raw[raw.dest == "OMAA"].copy()
        keep["snapshot_day"] = day
        frames.append(keep)
        print(f"{day}: {len(raw):6d} rows, {len(keep):4d} OMAA arrivals")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["key"])
    df["snapshot_day"] = pd.to_datetime(df.snapshot_day)
    df["month"] = df.snapshot_day.dt.strftime("%Y-%m")
    df["arr_time"] = pd.to_datetime(df.arrTime, errors="coerce")
    df["hex"] = df.hex.str.strip().str.lower()

    df = df.join(airports, on="orig")
    for code, iso in MANUAL_ICAO_COUNTRY.items():
        hit = df.orig == code
        df.loc[hit, "origin_country"] = iso
        df.loc[hit, "origin_patched"] = True
    for code, label in MANUAL_ICAO_NAME.items():
        hit = df.orig == code
        df.loc[hit, "origin_patched"] = True
        df.loc[hit, "origin_kind"] = label
    df["origin_patched"] = df.origin_patched.fillna(False)
    df["origin_known"] = df.origin_country.notna()
    df["is_local_leg"] = df.orig == "OMAA"
    lat = df.origin_lat.where(df.origin_lat.notna(), OMAA_LAT)
    lon = df.origin_lon.where(df.origin_lon.notna(), OMAA_LON)
    df["distance_km"] = [round(haversine_km(a, b, OMAA_LAT, OMAA_LON), 1)
                         for a, b in zip(lat, lon)]
    df["arrival_utc"] = df.arr_time
    df["dep_date"] = pd.to_datetime(df.depTime, errors="coerce").dt.strftime("%Y-%m-%d")
    df["actype"] = df.actype.str.strip().str.upper()

    out_cols = ["snapshot_day", "month", "key", "callsign", "orig", "origin_iata",
                "origin_city", "origin_country", "origin_known", "origin_patched",
                "origin_kind", "is_local_leg", "distance_km", "actype", "hex",
                "status", "std", "arrival_utc", "dep_date", "equip"]
    df[out_cols].sort_values(["arrival_utc", "callsign"]).to_csv(
        ADSB_LEGS, index=False)

    inbound = df[~df.is_local_leg]
    local_by_month = (df[df.is_local_leg].month.value_counts()
                        .rename("local_legs"))
    monthly = (inbound.groupby("month")
                    .agg(flights=("key", "size"),
                         origins_known=("origin_known", "sum"),
                         origins=("orig", "nunique"),
                         countries=("origin_country", "nunique"),
                         types=("actype", "nunique"))
                    .reset_index())
    monthly["local_legs"] = monthly.month.map(local_by_month).fillna(0).astype(int)
    monthly["unknown_origin"] = monthly.flights - monthly.origins_known
    monthly = monthly[["month", "flights", "local_legs", "origins_known",
                       "unknown_origin", "origins", "countries", "types"]]
    monthly.to_csv(ADSB_MONTHLY, index=False)

    markets = (inbound[inbound.origin_country.notna()]
                     .groupby(["origin_country", "month"])
                     .agg(flights=("key", "size"))
                     .reset_index())
    markets.to_csv(ADSB_MARKETS, index=False)

    types = (inbound.groupby("actype")
                  .agg(flights=("key", "size"))
                  .reset_index().sort_values("flights", ascending=False))
    types.to_csv(ADSB_TYPES, index=False)

    print(f"\nOMAA arrival rows: {len(df)} across {df.month.nunique()} sampled months")
    print(f"local OMAA-OMAA legs excluded from markets: {int(df.is_local_leg.sum())}")
    print(f"inbound international: {len(inbound)}")
    print(f"origin resolved: {int(inbound.origin_known.sum())}/{len(inbound)}")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
