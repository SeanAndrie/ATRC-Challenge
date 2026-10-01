"""Calibrate the 2025 adsb-exchange snapshot against official AUH figures.

The snapshot is a one-day-per-month sample drawn from a feed that does not
cover every flight. This script establishes how much of AUH's arrivals the
snapshot actually sees, so that downstream work can use the snapshot for what
it is good at (market mix, aircraft mix, hourly shape) while treating its
levels as a sample rather than a census.

Capture rate is computed per month against the official monthly aircraft
movement count published by Abu Dhabi Airports Company and transcribed in
data/processed/pax_movements.csv. Movements count both arrivals and
departures, so daily arrivals are movements / 2 / days_in_month.
"""

import sys
from calendar import monthrange
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aircraft_seats as S                      # noqa: E402
from paths import ADSB_CAPTURE, ADSB_LEGS, PAX_MOVEMENTS  # noqa: E402

# The snapshot panel and the capture rate both read from and write to
# data/processed/; official movements come from the airport movements table.
PAX = PAX_MOVEMENTS


def official_movements():
    p = pd.read_csv(PAX)
    p = p[(p.scope == "auh") & (p.granularity == "month")
          & (p.metric == "movements_aircraft")]
    p = p.assign(month=p.period_start.str[:7])
    return p.groupby("month").value.sum().rename("official_movements")


def main():
    arr = pd.read_csv(ADSB_LEGS, low_memory=False)
    inbound = arr[~arr.is_local_leg].copy()
    inbound = S.add_seats(inbound)

    obs = inbound.groupby("month").agg(
        obs_flights=("key", "size"),
        obs_seats=("seats_typ", "sum"),
        obs_seats_low=("seats_low", "sum"),
        obs_seats_high=("seats_high", "sum"),
        obs_countries=("origin_country", "nunique"),
    )

    off = official_movements()
    cal = obs.join(off, how="left")
    cal.index = cal.index.astype(str)
    cal["days"] = [monthrange(int(m[:4]), int(m[5:7]))[1] for m in cal.index]
    cal["official_daily_arrivals"] = (cal.official_movements / 2 / cal.days).round(1)
    cal["capture_rate"] = (cal.obs_flights / cal.official_daily_arrivals).round(3)
    cal["seats_per_flight"] = (cal.obs_seats / cal.obs_flights).round(0)
    cal["scaled_daily_flights"] = (cal.obs_flights / cal.capture_rate).round(0)
    cal["scaled_daily_seats"] = (cal.obs_seats / cal.capture_rate).round(0)

    cal.to_csv(ADSB_CAPTURE)

    print("=== per-month calibration ===")
    print(cal[["obs_flights", "official_daily_arrivals", "capture_rate",
               "seats_per_flight", "scaled_daily_flights"]].to_string())

    mean_cap = cal.capture_rate.mean()
    pooled = cal.obs_flights.sum() / cal.official_daily_arrivals.sum()
    print(f"\nmean capture rate        : {mean_cap:.3f}")
    print(f"pooled capture rate      : {pooled:.3f}")
    print(f"capture rate range       : {cal.capture_rate.min():.3f} - {cal.capture_rate.max():.3f}")

    daily_flights = cal.scaled_daily_flights.mean()
    daily_seats = cal.scaled_daily_seats.mean()
    mean_seats = inbound.seats_typ.mean()
    official_arrivals = cal.official_movements.sum() / 2
    official_pax_arr = 16_413_437

    print(f"\nmean seats per flight    : {mean_seats:.0f}")
    print(f"scaled daily flights     : {daily_flights:,.0f}")
    print(f"official annual arrivals : {official_arrivals:,.0f}")
    print(f"scaled annual flights    : {daily_flights * 365:,.0f}")
    print(f"scaled daily seats       : {daily_seats:,.0f}")
    print(f"scaled annual seats      : {daily_seats * 365:,.0f}")
    print(f"official pax arrivals    : {official_pax_arr:,}")
    print(f"implied pax per arrival  : {official_pax_arr / official_arrivals:.0f}")
    print(f"implied load factor      : {official_pax_arr / (daily_seats * 365):.3f}")


if __name__ == "__main__":
    main()
