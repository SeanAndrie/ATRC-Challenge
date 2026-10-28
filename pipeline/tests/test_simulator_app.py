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

# The app-logic diagnostic. Writes no report and needs no data files: it pins the
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

    rc = L.route_change("dubai", 0.5, "2025-09", "2025-12", "DXB", "EK")
    assert rc.market == "DUBAI" and rc.seats_mult == 0.5
    assert rc.ended_period == "2025-12"
    for bad in ((0.5, "2025-12", "2025-09"), (-0.1, "2025-09", None)):
        try:
            L.route_change("DUBAI", bad[0], bad[1], bad[2])
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for route change {bad}")
    sc_rc = L.build_scenario({}, route_changes=(rc,))
    assert sc_rc.route_changes == (rc,)
    print("route_change validates periods and non-negative multipliers")

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

    # -- impact windows, seasonal shape, market x season -------------------
    last12 = L.filter_window(frame, "Last 12 months")
    assert len(last12["ym"].unique()) == 6  # synthetic frame only has 6 months
    first12 = L.filter_window(frame, L.WINDOW_OPTIONS[1], route_start="2024-03")
    assert first12["ym"].min() == pd.Period("2024-03", freq="M")
    assert first12["ym"].max() == pd.Period("2024-06", freq="M")
    winter = L.filter_window(frame, "Winter (Dec–Feb)")
    assert set(winter["month"].unique()) <= {12, 1, 2}
    profile = L.season_profile_from(0.8, 1.0, 1.2, 1.0)
    assert len(profile) == 12 and profile[0] == profile[11] == 0.8
    pairs = L.market_month_pairs(
        [{"market": "ALPHA", "Winter": 0.8, "Spring": 1.0, "Summer": 1.2, "Autumn": 1.0}]
    )
    assert ("ALPHA", 12, 0.8) in pairs and ("ALPHA", 7, 1.2) in pairs
    assert all(m == 1 for _, m, _ in pairs if _ == "ALPHA" and m in (3, 4, 5))
    matrix = L.market_season_matrix(frame, "GuestNightsHat")
    assert matrix.shape == (3, 12)
    assert np.isclose(matrix.to_numpy().sum(), frame["GuestNightsHat"].sum())
    print("impact window, seasonal profile and market x season matrix ok")

    # -- attribution range and efficiency ratio ---------------------------
    hotel_mix = frame.groupby("market")["GuestNightsHat"].sum()
    ranges = L.attribution_range(frame, hotel_mix)
    assert len(ranges) == 3
    assert np.all(ranges["low"] <= ranges["last_departure"] + 1e-9)
    assert np.all(ranges["high"] >= ranges["last_departure"] - 1e-9)
    assert np.isclose(ranges["last_departure"].sum(), frame["GuestNightsHat"].sum())
    eff = L.efficiency_table(frame)
    assert "guests_per_1k_seats" in eff.columns
    assert np.isclose(
        eff.set_index("market").loc["ALPHA", "guests_per_1k_seats"],
        1000.0
        * frame.loc[frame["market"] == "ALPHA", "NewArrivalHat"].sum()
        / frame.loc[frame["market"] == "ALPHA", "Total_Seats"].sum(),
    )
    assert np.isclose(
        L.wmape([100.0, 50.0], [110.0, 45.0]), 100.0 * 15.0 / 150.0
    )
    print("attribution range, efficiency ratio and WMAPE ok")

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
        L.market_season_heatmap(L.market_season_matrix(frame)),
        L.attribution_range_chart(
            L.attribution_range(frame, frame.groupby("market")["GuestNightsHat"].sum())
        ),
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
    print("all chart builders return Plotly Figures")

    # -- the market-mix flag survives the reshape (attribution guard) ------
    assert set(frame["data_status"]) == {"supply-side-only"}
    print("supply-side-only flag present on the market frame")

    print("\nAll simulator-app logic checks passed.")


if __name__ == "__main__":
    main()
