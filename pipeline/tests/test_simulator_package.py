from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Run standalone (`.venv/bin/python pipeline/tests/test_simulator_package.py`):
# the repo root has to be importable for `pipeline.*`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import hybrid as H  # noqa: E402
from pipeline.calendar_events import COVERED_TO  # noqa: E402
from pipeline.simulator import RouteEdit, Scenario, Simulator  # noqa: E402

# This is the ninth diagnostic. Unlike the other eight it writes NO report:
# it exists so the packaging of the simulator cannot silently drift from the
# notebook model without a test failing. Runtime is dominated by the two
# LightGBM ensembles it builds (~2 minutes); every assertion mirrors a pin the
# notebook Part C already commits to, restated here against the packaged code
# the Streamlit contributor will actually call.


def main():
    t0 = time.time()

    # Lazy fit: a fresh simulator holds no boosters, and fit() trains only the
    # requested window (the competition model is deferred until asked for).
    sim = Simulator(window="full")
    assert sim._boosters == {"full": None, "competition": None}
    sim.fit()
    assert sim._boosters["full"] is not None
    assert sim._boosters["competition"] is None, "fit() must train only its window"
    print(f"lazy per-window fit ok ({time.time()-t0:.1f}s for full only)")

    # featurize now has its categorical encodings (set by the build), and the
    # leak-free feature set is pinned: 13 with the calendar, 9 without.
    assert H.featurize(sim.panel.head(3), events=True).shape[1] == 13
    assert H.featurize(sim.panel.head(3), events=False).shape[1] == 9
    print("build_panels + featurize pins ok")

    # Determinism of the packaged ensemble on a small fit (cheap).
    sub = sim.panel.sample(n=300, random_state=0)
    p1 = H.predict_hybrid(H.fit_hybrid(sub, 60), sub)
    p2 = H.predict_hybrid(H.fit_hybrid(sub, 60), sub)
    assert np.allclose(p1, p2), "LightGBM fit must be deterministic"
    print("determinism (two small fits identical) ok")

    # The pinned T5 protocol: trains the competition window on demand, refit on
    # the full panel, scored on the competition test window. Notebook Part C
    # pins this at 20.8% +- 2.0 with bias near -0.1%.
    rep = sim.reproduce_competition_window()
    assert sim._boosters["competition"] is not None, "reproduce must train its window"
    wm = rep.attrs["wm_na"]
    bias = rep.attrs["bias_pct"]
    print(
        f"T5 competition window: WMAPE={wm:.2f}% bias={bias:.3f}% "
        f"best_iter={rep.attrs['best_iter']} ({time.time()-t0:.0f}s elapsed)"
    )
    assert abs(wm - 20.8) < 2.0, f"T5 WMAPE {wm:.2f}% outside 20.8 +- 2.0"
    assert abs(bias) < 2.0, f"T5 aggregate bias {bias:.3f}% outside +- 2.0"

    # Supply assembly: a unit scenario must reproduce the measured pack data
    # exactly (this is what licenses the lever framework at all).
    base = sim.chain(Scenario())
    cmp = sim.fa[["ym", "market", "Total_Seats", "Total_PAX", "Total_P2P"]]
    m = base.merge(cmp, on=["ym", "market"], suffixes=("_b", "_f"))
    for col in ("Total_Seats", "Total_PAX", "Total_P2P"):
        d = float((m[f"{col}_b"] - m[f"{col}_f"]).abs().max())
        assert d < 1e-6, f"Scenario() supply drift on {col}: {d}"
    print("Scenario() base supply == fa ok")

    # Chain path with unit levers == the notebook simulate(panel, params).
    s = H.simulate(sim.panel, sim.params)
    c = sim.chain(Scenario())
    cm = c.merge(
        s[["ym_str", "market", "NewArrivalHat", "GuestNightsHat"]],
        on=["ym_str", "market"],
        suffixes=("_c", "_n"),
    )
    d = float((cm["NewArrivalHat_c"] - cm["NewArrivalHat_n"]).abs().max())
    assert d < 1e-9, f"chain(Scenario()) drifted from simulate: {d}"
    print("chain(Scenario()) == simulate(panel, params) ok")

    # Supply levers are pre-model: more seats must mean more arrivals, and the
    # response is non-linear (exact multiplier scaling is impossible once
    # features and exposure move).
    g1 = sim.monthly(Scenario()).GuestNightsHat.sum()
    g2 = sim.monthly(Scenario(seats_mult=1.2)).GuestNightsHat.sum()
    assert g2 > g1, "seats_mult=1.2 must raise guest-nights"
    g3 = sim.monthly(Scenario(seats_mult=1.2, lf_mult=1.1)).GuestNightsHat.sum()
    assert g3 > g2
    print(f"lever direction ok (nights {g1:,.0f} -> {g2:,.0f})")

    # Post-hoc dials are exactly multiplicative on the aggregate output.
    assert sim.monthly(Scenario(capture_mult=0.0)).NewArrivalHat.sum() == 0.0
    assert sim.monthly(Scenario(alos_mult=0.0)).GuestNightsHat.sum() == 0.0
    print("capture/alos zero dials ok")

    # Market view is explicitly supply-side-only, as both an attr and a column
    # (the column survives groupby/serialization; the attr does not).
    mdf = sim.market(Scenario(seats_mult=1.1))
    assert mdf.attrs["data_status"] == "supply-side-only"
    assert (mdf["data_status"] == "supply-side-only").all()
    print("market() flagged supply-side-only (attr + column) ok")

    # A per-market multiplier is immutable (tuple) and scales exactly its
    # market, leaving the others untouched. (Direction on the *hybrid* is not
    # asserted: the model can learn a negative log_seats response for a single
    # market -- CHINA's capture is the join defect's worst case -- so scaling
    # one market need not move its predicted nights the expected way. The
    # supply frame itself is the contract.)
    assert isinstance(Scenario().market_seats_mult, tuple)
    assert hash(Scenario(market_seats_mult=(("CHINA", 1.2),))) is not None
    scaled0, _ = sim._scenario_supply(Scenario())
    scaled1, _ = sim._scenario_supply(Scenario(market_seats_mult=(("CHINA", 1.2),)))
    ch0 = scaled0.loc[scaled0.market == "CHINA", "Total_Seats"].sum()
    ch1 = scaled1.loc[scaled1.market == "CHINA", "Total_Seats"].sum()
    assert abs(ch1 / ch0 - 1.2) < 1e-9, "market multiplier must scale that market by 1.2"
    assert (
        abs(
            scaled0.loc[scaled0.market != "CHINA", "Total_Seats"].sum()
            - scaled1.loc[scaled1.market != "CHINA", "Total_Seats"].sum()
        )
        < 1e-6
    ), "market multiplier must not touch other markets"
    print("market_seats_mult immutable + scoped to its market ok")

    # Route edits add supply for an existing market only, and the added seats
    # convert through the chain (where P2P -> New Arrivals is monotonic).
    edit = Scenario(routes=(RouteEdit("CHINA", "Chengdu", "Etihad", 4000.0, "2025-09"),))
    _base_raw = sim._scenario_supply(Scenario())[1]
    _edit_raw = sim._scenario_supply(edit)[1]
    added = (
        _edit_raw.loc[_edit_raw.market == "CHINA", "Total_Seats"].sum()
        - _base_raw.loc[_base_raw.market == "CHINA", "Total_Seats"].sum()
    )
    assert added > 0, "RouteEdit must add CHINA seats"
    assert (
        sim.chain(edit).NewArrivalHat.sum() > sim.chain(Scenario()).NewArrivalHat.sum()
    ), "RouteEdit must raise chain New Arrivals"
    print("RouteEdit adds supply + arrivals ok")

    # The baseline now runs to the calendar horizon: observed months through
    # 2026-02, then a held-forward seasonal projection (same month, year - 1).
    supply = sim.route_supply()
    assert set(supply["basis"]) == {"observed", "projected"}
    assert supply["ym"].max() == COVERED_TO
    print(f"baseline projects through {supply['ym'].max()} ok")

    # Dated, signed edits: a winter-only trim touches only its window, and the
    # basis marker distinguishes observed / projected / scenario supply.
    base_i = sim._scenario_supply(Scenario())[1].set_index(["ym", "market"])
    win = Scenario(routes=(RouteEdit("INDIA", "Mumbai", "Etihad", 1000.0, "2026-11", "2027-02"),))
    win_i = sim._scenario_supply(win)[1].set_index(["ym", "market"])
    for ym, changed in (("2026-10", False), ("2026-11", True), ("2027-02", True), ("2027-03", False)):
        delta = win_i.loc[(ym, "INDIA"), "Total_Seats"] - base_i.loc[(ym, "INDIA"), "Total_Seats"]
        if changed:
            assert delta > 0, f"winter edit must raise {ym} seats"
        else:
            assert abs(delta) < 1e-9, f"winter edit must not touch {ym}"
    assert base_i.loc[("2025-10", "INDIA"), "basis"] == "observed"
    assert base_i.loc[("2026-10", "INDIA"), "basis"] == "projected"
    assert win_i.loc[("2026-11", "INDIA"), "basis"] == "scenario"
    print("dated winter edit + basis markers ok")

    # Reductions floor at zero: an oversized negative edit removes the market
    # from the frame instead of emitting negative seats.
    neg = sim._scenario_supply(
        Scenario(routes=(RouteEdit("INDIA", "X", "Y", -1e12, "2026-11"),))
    )[1]
    assert (neg["Total_Seats"] > 0).all(), "floored supply must stay positive"
    nov = set(neg.loc[neg["ym"].astype(str) == "2026-11", "market"])
    assert "INDIA" not in nov, "oversized reduction must drop INDIA from 2026-11"
    print("negative edit floors at zero ok")

    # A per-market load-factor multiplier scales exactly that market's PAX.
    s0 = sim._scenario_supply(Scenario())[0].set_index(["ym", "market"])
    s1 = sim._scenario_supply(Scenario(market_lf_mult=(("INDIA", 1.1),)))[0].set_index(["ym", "market"])
    idx = s0.index.get_level_values("market")
    a = s0.loc[idx == "INDIA", "Total_PAX"].sum()
    b = s1.loc[idx == "INDIA", "Total_PAX"].sum()
    assert abs(b / a - 1.1) < 1e-9, "market_lf_mult must scale INDIA PAX by 1.1"
    assert (
        abs(s0.loc[idx != "INDIA", "Total_PAX"].sum() - s1.loc[idx != "INDIA", "Total_PAX"].sum())
        < 1e-6
    ), "market_lf_mult must not touch other markets"
    print("market_lf_mult scoped to its market ok")

    # Guards: bad markets, bad values, degenerate supply, bad periods, unknown
    # view filters must all fail loudly.
    bad_scenarios = [
        ("unknown RouteEdit market", dict(routes=(RouteEdit("NARNIA", "X", "Y", 1.0, "2025-09"),))),
        ("post-horizon edit", dict(routes=(RouteEdit("CHINA", "X", "Y", 1.0, "2029-01"),))),
        ("malformed start month", dict(routes=(RouteEdit("CHINA", "X", "Y", 1.0, "next-winter"),))),
        ("end before start", dict(routes=(RouteEdit("CHINA", "X", "Y", 1.0, "2026-11", "2026-05"),))),
        ("zero route seats", dict(routes=(RouteEdit("CHINA", "X", "Y", 0.0, "2026-11"),))),
        ("unknown market_seats_mult key", dict(market_seats_mult=(("NARNIA", 1.2),))),
        ("zero market_seats_mult", dict(market_seats_mult=(("CHINA", 0.0),))),
        ("unknown market_lf_mult key", dict(market_lf_mult=(("NARNIA", 1.2),))),
        ("zero market_lf_mult", dict(market_lf_mult=(("CHINA", 0.0),))),
        ("zero seats_mult", dict(seats_mult=0.0)),
    ]
    for label, kw in bad_scenarios:
        try:
            sim.monthly(Scenario(**kw))
            raise SystemExit(f"FAIL: {label} did not raise")
        except ValueError:
            pass
    try:
        sim._predict(Scenario(), markets="NARNIA")
        raise SystemExit("FAIL: unknown markets= did not raise")
    except ValueError:
        pass
    print("guards ok")

    # Persistence round-trip: a saved + loaded simulator predicts identically,
    # including after resetting the module's categorical encodings -- the
    # fresh-process case, where load() must restore them from the payload
    # rather than assume build_panels() already ran.
    with tempfile.TemporaryDirectory() as td:
        art = Path(td) / "sim.pkl"
        sim.save(art)
        loaded = Simulator.load(art)
        a = sim.monthly(Scenario(seats_mult=1.2))
        b = loaded.monthly(Scenario(seats_mult=1.2))
        assert np.allclose(a.GuestNightsHat, b.GuestNightsHat)
        assert abs(
            loaded.reproduce_competition_window().attrs["wm_na"] - wm
        ) < 1e-9
        H._MKT_DTYPE = None
        H._MON_DTYPE = None
        fresh = Simulator.load(art)
        c = fresh.monthly(Scenario(seats_mult=1.2))
        assert np.allclose(a.GuestNightsHat, c.GuestNightsHat), (
            "load() must restore the categorical encodings by itself"
        )
    print("save/load round-trip ok (including fresh-process encodings)")

    print(f"\nALL SIMULATOR-PACKAGE PINS PASSED ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()