"""Audit the competition data pack against known properties and defects."""
from __future__ import annotations

import pandas as pd
from pipeline.competition import load as cl


def audit():
    f = cl.read_flights()
    # identity
    p2p = f["Total P2P_n"]
    tr = f["Total Transfer_n"]
    tn = f["Total Transit_n"]
    tot = f["Total PAX_n"]
    resid = tot - (p2p + tr + tn)
    identity_ok = (resid.notna() & (resid.abs() < 1e-9)).all()

    seats = f["Total Seats_n"]
    lf = f["lf_frac"]
    lf_eq = (lf - tot/seats).abs() < 1e-9
    lf_ok = lf_eq.notna().all()

    # awf x31 integrity
    awf_rep = f["awf_per_week"]
    integr = (awf_rep % 1 < 1e-6) | ((1 - (awf_rep % 1)) < 1e-6)
    awf_ok = (integr | awf_rep.isna()).all() and (awf_rep.dropna() % 1 < 1e-6).mean() > 0.99

    # blanks in 2022 only
    blank = f["Average Weekly Frequency"].isna()
    blanks_2022_only = (f.loc[blank, "dt"].dt.year == 2022).all() if blank.any() else True

    print("IDENTITY PAX==P2P+Transfer+Transit:", identity_ok)
    print("LF == PAX/Seats (all rows):", lf_ok)
    print("AWF*31 near-integer:", awf_ok)
    print("Blanks confined to 2022:", blanks_2022_only)
    print("Total rows with blanks:", blank.sum(), "months affected:", f.loc[blank,"dt"].dt.to_period("M").nunique() if blank.any() else 0)

    # market match
    intl = cl.read_international_train()
    fc = set(f["Departure Country Name"].dropna().unique())
    hn = set(intl["Nationality"].dropna().unique())
    norm = lambda s: {x.upper().strip() for x in s}
    fcn, hnn = norm(fc), norm(hn)
    print("flight countries:", len(fcn), "hotel nationalities:", len(hnn))
    print("overlap:", len(fcn&hnn), "flight-only:", len(fcn-hnn), "hotel-only:", len(hnn-fcn))
    print("hotel-only list:", sorted(hnn-fcn))
    print("flight-only list:", sorted(fcn-hnn))


if __name__ == "__main__":
    audit()
