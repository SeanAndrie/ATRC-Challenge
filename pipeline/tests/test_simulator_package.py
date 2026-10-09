from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Run standalone (`.venv/bin/python pipeline/tests/test_simulator_package.py`):
# the repo root has to be importable for `pipeline.*`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import hybrid as H  # noqa: E402
from pipeline.simulator import RouteEdit, Scenario, Simulator  # noqa: E402

# This is the ninth diagnostic. Unlike the other eight it writes NO report:
# it exists so the packaging of the simulator cannot silently drift from the
# notebook model without a test failing. Runtime is dominated by the two
# LightGBM ensembles Simulator.fit() builds (~60-70s); every assertion below
# mirrors a pin the notebook Part C already commits to, restated here against
# the packaged code the Streamlit contributor will actually call.


def main():
    t0 = time.time()

    # The packaged canonical build must expose the leakage-free feature set,
    # 13 with the calendar, 9 without (the assert lives inside featurize).
    fa, _a, _b, panel, _c, test_panel, _d = H.build_panels()
    assert H.featurize(panel.head(3), events=True).shape[1] == 13
    assert H.featurize(panel.head(3), events=False).shape[1] == 9
    print(f"build_panels + featurize pins ok ({time.time()-t0:.1f}s)")

    # Determinism of the packaged ensemble on a small fit (cheap).
    sub = panel.sample(n=300, random_state=0)
    p1 = H.predict_hybrid(H.fit_hybrid(sub, 60), sub)
    p2 = H.predict_hybrid(H.fit_hybrid(sub, 60), sub)
    assert np.allclose(p1, p2), "LightGBM fit must be deterministic"
    print("determinism (two small fits identical) ok")

    # The pinned T5 protocol: window = competition, refit on the full panel,
    # scored on the competition test window. Notebook Part C pins this at
    # 20.8% +- 2.0 with bias near -0.1%.
    sim = Simulator().fit()
    rep = sim.reproduce_competition_window()
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
    s = H.simulate(panel, sim.params)
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
    # response is non-linear (scaling by exactly the multiplier is impossible
    # once features and exposure change).
    g1 = sim.monthly(Scenario()).GuestNightsHat.sum()
    g2 = sim.monthly(Scenario(seats_mult=1.2)).GuestNightsHat.sum()
    assert g2 > g1, "seats_mult=1.2 must raise guest-nights"
    g3 = sim.monthly(Scenario(seats_mult=1.2, lf_mult=1.1)).GuestNightsHat.sum()
    assert g3 > g2
    print(f"lever direction ok (nights {g1:,.0f} -> {g2:,.0f})")

    # Post-hoc dials are exactly multiplicative on the aggregate output.
    z = sim.monthly(Scenario(capture_mult=0.0))
    assert z.NewArrivalHat.sum() == 0.0, "capture_mult=0 must zero arrivals"
    zn = sim.monthly(Scenario(alos_mult=0.0))
    assert zn.GuestNightsHat.sum() == 0.0, "alos_mult=0 must zero nights"
    print("capture/alos zero dials ok")

    # Market view is explicitly supply-side-only: the join attributes by
    # departure country, which is not guest nationality (METHODS 4.5).
    mdf = sim.market(Scenario(seats_mult=1.1))
    assert mdf.attrs["data_status"] == "supply-side-only"
    print("market() flagged supply-side-only ok")

    # Route edits add supply for an existing market only.
    fe = sim.monthly(
        Scenario(routes=(RouteEdit("CHINA", "Chengdu", "Etihad", 4000.0, "2025-09"),))
    )
    be = sim.monthly(Scenario())
    assert fe.NewArrivalHat.sum() > be.NewArrivalHat.sum(), "route edit must add arrivals"
    print("RouteEdit adds arrivals ok")

    # Guards: unknown markets and out-of-horizon supply must fail loudly.
    for bad, kw in [
        ("unknown market", {"routes": (RouteEdit("NARNIA", "X", "Y", 1.0, "2025-09"),)}),
        ("post-horizon edit", {"routes": (RouteEdit("CHINA", "X", "Y", 1.0, "2027-01"),)}),
    ]:
        try:
            sim.predict(Scenario(**kw))
            raise SystemExit(f"FAIL: {bad} did not raise")
        except ValueError:
            pass
    try:
        sim.predict(Scenario(), markets="NARNIA")
        raise SystemExit("FAIL: unknown markets= did not raise")
    except ValueError:
        pass
    print("guards ok")

    print(f"\nALL SIMULATOR-PACKAGE PINS PASSED ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()