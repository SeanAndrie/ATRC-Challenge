"""Load competition data (DCT Abu Dhabi challenge) with explicit repairs."""
from __future__ import annotations

import pandas as pd
from pipeline import paths


def _daterange(s: pd.Series) -> str:
    d = pd.to_datetime(s, format="mixed", errors="coerce")
    return f"{d.min().date()} .. {d.max().date()} ({d.nunique()} distinct)"


def read_flights() -> pd.DataFrame:
    f = pd.read_excel(paths.COMPETITION_FLIGHT)
    f["dt"] = pd.to_datetime(f["Date"], format="mixed", errors="coerce")
    f["ym"] = f["dt"].dt.to_period("M")

    awf = pd.to_numeric(f["Average Weekly Frequency"], errors="coerce")
    f["awf_per_week"] = (awf * 31).round(6)
    f["flights_per_week"] = (awf * 31).round(2)

    num_cols = [
        "Business Class P2P Count",
        "Business Class Seat Capacity",
        "Economy Class P2P Count",
        "Economy Class Seat Capacity",
        "First Class P2P Count",
        "First Class Seat Capacity",
        "Total P2P",
        "Total PAX",
        "Total PAX Excluding Infant",
        "Total Seats",
        "Total Transfer",
        "Total Transit",
        "Transfer Business Class Count",
        "Transfer Economy Class Count",
        "Transfer First Class Count",
        "Transit Business Class Count",
        "Transit Economy Class Count",
        "Transit First Count",
    ]
    for c in num_cols:
        f[c + "_n"] = pd.to_numeric(f[c], errors="coerce")

    f["lf_frac"] = pd.to_numeric(f["Load Factor"], errors="coerce")
    return f


def flights_monthly() -> pd.DataFrame:
    f = read_flights()
    g = f.groupby(
        ["ym", "Departure Country Name", "Departure City", "Airline Name"],
        dropna=False,
    ).agg(
        rows=("Date", "size"),
        dates=("dt", "nunique"),
        dt_min=("dt", "min"),
        dt_max=("dt", "max"),
        Total_PAX=("Total PAX_n", "sum"),
        Total_P2P=("Total P2P_n", "sum"),
        Total_Seats=("Total Seats_n", "sum"),
        Total_Transfer=("Total Transfer_n", "sum"),
        Total_Transit=("Total Transit_n", "sum"),
        awf_per_week_median=("awf_per_week", "median"),
        flights_per_week_median=("flights_per_week", "median"),
    )
    g = g.reset_index()
    g["ym"] = g["ym"].astype(str)
    return g


def flights_monthly_agg() -> pd.DataFrame:
    fm = flights_monthly()
    a = fm.groupby(["ym", "Departure Country Name"], dropna=False).agg(
        routes=("Departure City", lambda x: x.nunique()),
        airlines=("Airline Name", lambda x: x.nunique()),
        rows=("rows", "sum"),
        dates=("dates", "sum"),
        Total_PAX=("Total_PAX", "sum"),
        Total_P2P=("Total_P2P", "sum"),
        Total_Seats=("Total_Seats", "sum"),
        Total_Transfer=("Total_Transfer", "sum"),
        Total_Transit=("Total_Transit", "sum"),
    ).reset_index()
    a["ym"] = pd.PeriodIndex(a["ym"], freq="M")
    return a


def read_international_train() -> pd.DataFrame:
    df = pd.read_excel(paths.COMPETITION_INTL_TRAIN)
    df["dt"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df["ym"] = df["dt"].dt.to_period("M")
    for c in ["Guests", "New Arrivals"]:
        df[c + "_n"] = pd.to_numeric(df[c], errors="coerce")
    df["Same-Day Guests_n"] = pd.to_numeric(df["Same-Day Guests"], errors="coerce")
    df = df.rename(columns={"Nationality": "Nationality", "Residence (groups)": "Residence_groups"})
    return df


def read_international_test() -> pd.DataFrame:
    df = pd.read_excel(paths.COMPETITION_INTL_TEST)
    df["dt"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df["ym"] = df["dt"].dt.to_period("M")
    df["Same-Day Guests_n"] = pd.to_numeric(df["Same-Day Guests"], errors="coerce")
    df = df.rename(columns={"Nationality": "Nationality", "Residence (groups)": "Residence_groups"})
    return df


def read_domestic_train() -> pd.DataFrame:
    df = pd.read_excel(paths.COMPETITION_DOM_TRAIN)
    df["dt"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df["ym"] = df["dt"].dt.to_period("M")
    for c in ["Guests", "New Arrivals"]:
        df[c + "_n"] = pd.to_numeric(df[c], errors="coerce")
    df["Same-Day Guests_n"] = pd.to_numeric(df["Same-Day Guests"], errors="coerce")
    df = df.rename(columns={"Residence (groups)": "Residence_groups"})
    return df


def read_domestic_test() -> pd.DataFrame:
    df = pd.read_excel(paths.COMPETITION_DOM_TEST)
    df["dt"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df["ym"] = df["dt"].dt.to_period("M")
    df["Same-Day Guests_n"] = pd.to_numeric(df["Same-Day Guests"], errors="coerce")
    df = df.rename(columns={"Residence (groups)": "Residence_groups"})
    return df
