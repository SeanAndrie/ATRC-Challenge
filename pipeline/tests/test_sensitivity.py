from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run standalone (`.venv/bin/python pipeline/tests/test_sensitivity.py`): the
# repo root has to be importable for `pipeline.*`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import sensitivity as S  # noqa: E402

# The tenth diagnostic. Writes no report: it pins the empirical-sensitivity
# module on a small synthetic panel (no data files needed, so it is fast and
# hermetic), while the real-data numbers are produced by
# `notebooks/sensitivity_empirical.py`. Every assertion is about the contracts
# the notebook relies on: the lever series are finite, the three kernels return
# ordered positive windows, thin-month filtering actually filters, and the Sobol
# decomposition responds to the bounds it is given.


def _synthetic_panel(seed: int = 0, months: int = 24, markets: int = 6):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(markets):
        mkt = f"M{m}"
        for t in range(months):
            ym = pd.Period("2023-01", freq="M") + t
            seats = int(rng.integers(15_000, 90_000))
            pax = int(seats * rng.uniform(0.55, 0.9))
            # one thin market exercises the min_p2p floor
            p2p = int(pax * rng.uniform(0.5, 0.95)) if m else int(rng.integers(0, 40))
            new = max(1, int(p2p * rng.uniform(0.1, 0.9)))
            guests = max(1, int(new * rng.uniform(2.0, 5.0)))
            rows.append(
                {
                    "market": mkt,
                    "month": ym.month,
                    "ym": ym,
                    "ym_str": str(ym),
                    "Total_Seats": seats,
                    "Total_PAX": pax,
                    "Total_P2P": p2p,
                    "NewArr": new,
                    "Guests": guests,
                }
            )
    panel = pd.DataFrame(rows)

    g = panel.groupby("market").agg(
        New=("NewArr", "sum"), P2P=("Total_P2P", "sum"), G=("Guests", "sum")
    )
    season = pd.Series(
        1.0 + 0.05 * np.sin(np.arange(12)), index=range(1, 13), dtype=float
    )
    season = season / season.mean()
    params = {
        "capture": (g["New"] / g["P2P"]).clip(upper=5.0),
        "alos": g["G"] / g["New"],
        "season": season,
    }
    return panel, params


def main():
    panel, params = _synthetic_panel()

    # NAMES order is the x layout the Sobol problem and notebook T6 share.
    assert S.NAMES == [
        "seats / frequency",
        "load factor",
        "P2P share (transfer/transit)",
        "capture rate",
        "ALOS",
        "season",
    ], S.NAMES
    print("NAMES order/length ok")

    series = S.lever_series(panel, params, min_p2p=0)
    assert set(series) == set(S.NAMES)
    for name, (u, w) in series.items():
        assert np.all(np.isfinite(u)), f"{name}: non-finite u"
        assert np.all(np.isfinite(w)) and np.all(w > 0), f"{name}: bad weights"
        assert u.size == w.size and u.size > 0
    # season is exactly the 12 fitted values, uniformly weighted.
    u_seas, w_seas = series["season"]
    assert u_seas.size == 12 and np.allclose(w_seas, 1.0)
    print("lever series finite; season is 12 uniform values")


    # min_p2p floor actually drops the thin-market months from capture.
    n_open = S.lever_series(panel, params, min_p2p=0)["capture rate"][0].size
    n_floor = S.lever_series(panel, params, min_p2p=2000)["capture rate"][0].size
    assert n_floor < n_open, (n_floor, n_open)
    print(f"min_p2p floor filters capture months ({n_open} -> {n_floor})")

    # Bounds: all three kernels, all six levers, ordered and positive, and
    # inside the observed support of the lever.
    for kernel in ("uniform", "weighted", "winsorized"):
        b = S.empirical_lever_bounds(panel, params, kernel=kernel, min_p2p=0)
        assert set(b) == set(S.NAMES)
        for name in S.NAMES:
            lo, hi = b[name]
            u = series[name][0]
            assert 0.0 < lo < hi, (kernel, name, lo, hi)
            assert u.min() - 1e-9 <= lo and hi <= u.max() + 1e-9, (kernel, name)
    print("all three kernels: tight, ordered, positive windows")

    # Clean error when the noise floor removes every capture month.
    try:
        S.lever_series(panel, params, min_p2p=int(10**12))
    except ValueError:
        print("empty capture series raises a clear ValueError")
    else:
        raise AssertionError("expected ValueError for an impossible min_p2p")

    # Sobol problem is built strictly from NAMES order.
    apriori = dict(
        zip(S.NAMES, [[0.8, 1.2], [0.9, 1.1], [0.9, 1.1], [0.7, 1.3], [0.9, 1.1], [0.9, 1.1]])
    )
    prob = S.sobol_problem(apriori)
    assert prob["num_vars"] == 6 and prob["names"] == S.NAMES
    assert prob["bounds"][3] == [0.7, 1.3]
    print("sobol_problem ok")

    # Sobol on the synthetic panel: shape, unit elasticity, sane indices.
    sens = S.sobol_sensitivity(panel, params, apriori, n=64)
    assert list(sens.columns) == ["lever", "elasticity", "S1", "ST"]
    assert len(sens) == 6
    assert np.allclose(sens["elasticity"], 1.0), sens["elasticity"].tolist()
    assert sens["ST"].between(0.0, 1.0).all(), sens["ST"].tolist()
    assert sens["S1"].between(-1.0, 1.0).all(), sens["S1"].tolist()
    assert sens["ST"].is_monotonic_decreasing
    print("sobol_sensitivity shape/elasticity/index contracts ok")

    # Determinism: SALib sampling and analysis are seed-free here.
    again = S.sobol_sensitivity(panel, params, apriori, n=64)
    assert np.allclose(sens["ST"], again["ST"]) and np.allclose(sens["S1"], again["S1"])
    print("sobol_sensitivity is deterministic")

    # The decomposition must respond to the bounds: widening capture alone
    # raises capture's total-order share.
    narrow = dict(apriori)
    wide = dict(apriori)
    wide["capture rate"] = (0.2, 1.8)
    st_narrow = S.sobol_sensitivity(panel, params, narrow, n=256).set_index("lever")["ST"]
    st_wide = S.sobol_sensitivity(panel, params, wide, n=256).set_index("lever")["ST"]
    assert st_wide["capture rate"] > st_narrow["capture rate"], (
        st_narrow["capture rate"],
        st_wide["capture rate"],
    )
    print("widening capture raises its ST share (bounds drive the ranking)")

    print("\nAll sensitivity checks passed.")


if __name__ == "__main__":
    main()