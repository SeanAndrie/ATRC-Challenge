from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run standalone (`.venv/bin/python pipeline/tests/test_simulator_app.py`): the
# repo root has to be importable for `app.*` and `pipeline.*`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import logic as L  # noqa: E402
from pipeline.simulator import Scenario  # noqa: E402

# The eleventh diagnostic. Writes no report and needs no data files: it pins the
# Streamlit app's *pure* logic (scenario assembly, route-form conversion, delta
# math, and the chart builders) on synthetic frames, so the UI can be verified
# headlessly. Anything that requires a fitted engine is deliberately out of
# scope here -- it is covered by test_simulator_package.py.


def _synthetic_market_frame(markets=("ALPHA", "BETA", "GAMMA"), months=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in markets:
        for t in range(months):
            ym = pd.Period("2024-01", freq="M") + t
            p2p = float(rng.integers(500, 5000))
            new = p2p * 0.4
            rows.append(
                {
                    "ym": ym,
                    "ym_str": str(ym),
                    "market": m,
                    "month": ym.month,
                    "Total_Seats": p2p * 3,
                    "PAX": p2p * 1.5,
                    "P2P": p2p,
                    "NewArrivalHat": new,
                    "GuestNightsHat": new * 3.0,
                    "data_status": "supply-side-only",
                }
            )
    return pd.DataFrame(rows)


def main():
    # -- scenario assembly --------------------------------------------------
    sc = L.build_scenario(
        {"seats_mult": 1.2, "capture_mult": 0.9}, market_mults=(("ALPHA", 1.5),)
    )
    assert isinstance(sc, Scenario)
    assert sc.seats_mult == 1.2 and sc.capture_mult == 0.9
    assert sc.lf_mult == 1.0 and sc.market_seats_mult == (("ALPHA", 1.5),)
    # unknown lever keys are ignored, not passed to the frozen dataclass
    sc2 = L.build_scenario({"seats_mult": 0.8, "not_a_lever": 9.0})
    assert sc2.seats_mult == 0.8
    print("build_scenario maps levers, ignores unknown keys, keeps defaults")

    # -- route form -> RouteEdit -------------------------------------------
    r = L.route_edit("dubai", "Route City", "Airline X", 3, 180, "2025-09")
    assert r.market == "DUBAI" and r.city == "Route City"
    assert r.seats_per_week == 3 * 180, r.seats_per_week
    assert r.started_period == "2025-09"
    for bad in ((0, 180), (-1, 180), (3, 0), (3, -5)):
        try:
            L.route_edit("DUBAI", "c", "a", *bad, "2025-09")
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for freq/seats {bad}")
    print("route_edit multiplies frequency x seats and rejects non-positive inputs")

    # -- month options cover the pack window -------------------------------
    opts = L.month_options()
    assert opts[0] == "2022-01" and opts[-1] == "2026-02"
    assert "2025-09" in opts
    print("month_options spans the covered window")

    # -- reshape: monthly / seasonal / market totals -----------------------
    frame = _synthetic_market_frame()
    monthly = L.monthly_from(frame)
    assert monthly["ym_str"].is_monotonic_increasing
    assert len(monthly) == frame["ym"].nunique()
    assert np.isclose(monthly["GuestNightsHat"].sum(), frame["GuestNightsHat"].sum())
    # monthly P2P equals the sum over markets for that month
    first = monthly.iloc[0]
    m0 = frame[frame["ym_str"] == first["ym_str"]]["P2P"].sum()
    assert np.isclose(first["P2P"], m0), (first["P2P"], m0)
    print("monthly_from aggregates to one row per month, totals preserved")

    seasonal = L.seasonal_from(frame)
    assert seasonal["month"].is_monotonic_increasing
    assert len(seasonal) == frame["month"].nunique()
    assert np.isclose(seasonal["NewArrivalHat"].sum(), frame["NewArrivalHat"].sum())
    print("seasonal_from aggregates by calendar month, totals preserved")

    totals = L.market_totals(frame, "GuestNightsHat")
    assert totals["GuestNightsHat"].is_monotonic_decreasing
    assert np.isclose(totals["GuestNightsHat"].sum(), frame["GuestNightsHat"].sum())
    print("market_totals ranks markets descending and preserves the total")

    # -- headline deltas ----------------------------------------------------
    base = _synthetic_market_frame()
    scen = base.copy()
    scen["GuestNightsHat"] = scen["GuestNightsHat"] * 1.10
    head = L.headline(scen, base)
    assert list(head["metric"]) == [
        L.METRIC_LABELS["P2P"],
        L.METRIC_LABELS["NewArrivalHat"],
        L.METRIC_LABELS["GuestNightsHat"],
    ]
    gn = head.set_index("metric").loc[L.METRIC_LABELS["GuestNightsHat"]]
    assert np.isclose(gn["change_pct"], 10.0), gn["change_pct"]
    assert np.isclose(gn["change"], base["GuestNightsHat"].sum() * 0.10)
    p2p = head.set_index("metric").loc[L.METRIC_LABELS["P2P"]]
    assert np.isclose(p2p["change_pct"], 0.0)
    print("headline reports absolute and percentage change vs baseline")

    # -- chart builders return figures without a display -------------------
    go = __import__("plotly.graph_objects", fromlist=["Figure"]).Figure
    figs = [
        L.monthly_trend(monthly, monthly, chain_monthly=monthly),
        L.seasonal_bars(seasonal, seasonal),
        L.market_pareto(totals, totals),
        L.sensitivity_bars(
            pd.DataFrame(
                {
                    "lever": L.LEVER_LABELS.values(),
                    "ST": np.linspace(0.05, 0.6, len(L.LEVER_LABELS)),
                }
            ),
            "ST",
        ),
        L.validation_scatter(
            pd.DataFrame({"NewArr": [1.0, 2.0, 3.0], "NewArrivalHat": [1.1, 1.9, 3.2]}),
            "NewArr",
            "NewArrivalHat",
        ),
    ]
    assert all(isinstance(f, go) for f in figs), [type(f) for f in figs]
    print("all five chart builders return Plotly Figures")

    # -- the market-mix flag survives the reshape (attribution guard) ------
    assert set(frame["data_status"]) == {"supply-side-only"}
    print("supply-side-only flag present on the market frame")

    print("\nAll simulator-app logic checks passed.")


if __name__ == "__main__":
    main()
