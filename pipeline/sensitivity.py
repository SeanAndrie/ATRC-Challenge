"""Empirically measured lever spreads and their Sobol decomposition.

The chain in ``pipeline.hybrid`` exposes six multiplicative levers. The
notebook's T6 cell decomposes their variance over **a-priori** ranges (the room
the simulator lets a planner press). This module supplies the **empirical**
companion: for each lever it derives the month-level multiplier actually
observed in the panel (``u = z / baseline``), takes a volume-weighted 16th–84th
percentile as that lever's "Sobol range", and runs the same SALib Sobol
decomposition on those ranges.

The two views answer different questions and are kept labelled, never merged:

* a-priori bounds -> "given the room the UI allows, which lever can move the
  number most?" (deliberately wider than observed; feeds the simulator)
* empirical bounds -> "which lever actually moved the number over the fit
  window?" (descriptive; feeds interpretation, criterion 04)

The empirical ranges are informational: they never overwrite the simulator's
``LEVER_BOUNDS`` (new-route and step-change scenarios legitimately exceed the
historical wobble).

Baselines and row weights per lever:

| lever | month measure ``z`` | baseline | row weight |
| --- | --- | --- | --- |
| seats / frequency | ``Total_Seats`` month-over-month | 1.0 | ``Total_Seats`` (current month) |
| load factor | ``Total_PAX / Total_Seats`` | market seat-weighted mean | ``Total_Seats`` |
| P2P share | ``Total_P2P / Total_PAX`` | market PAX-weighted mean | ``Total_PAX`` |
| capture rate | ``NewArr / Total_P2P`` | ``capture[m]`` (capped, shipped) | ``Total_P2P``; months with ``Total_P2P < min_p2p`` dropped (noise-aware) |
| ALOS | ``Guests / NewArr`` | ``ALOS[m]`` (shipped) | ``NewArr``; months with 0 arrivals or guests dropped |
| season | ``season[month]`` (12 values) | 1.0 | uniform |

``sobol_sensitivity`` mirrors the notebook T6 cell's algebra exactly
(``seats0``, ``lf0``, ``p2p_share0``, ``cap_arr``, ``alos_arr``,
``season_arr``; one scalar multiplier each), so the two views are directly
comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lever names and order must match the notebook T6 cell / the SALib problem's
# x layout: x[0..5] = seats, load factor, P2P share, capture, ALOS, season.
NAMES = [
    "seats / frequency",
    "load factor",
    "P2P share (transfer/transit)",
    "capture rate",
    "ALOS",
    "season",
]


def _clean(u: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only finite ratios with positive weight."""
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(u) & np.isfinite(w) & (w > 0)
    return u[ok], w[ok]


def _weighted_quantile(u: np.ndarray, w: np.ndarray, q: float) -> float:
    """Empirical quantile of ``u`` weighted by ``w`` (weights>0)."""
    idx = np.argsort(u, kind="mergesort")
    us = u[idx]
    ws = w[idx]
    cw = np.cumsum(ws, dtype=float)
    cw = cw / cw[-1]
    return float(np.interp(q, cw, us))


def lever_series(panel, params, min_p2p: int = 50):
    """Return ``{lever: (u, w)}``, the observed monthly multiplier and weight.

    ``u = z / baseline`` per market-month (see module docstring); ``w`` is the
    volume weight (or uniform for ``season``). ``min_p2p`` only affects the
    capture lever, where thin months are too noisy to trust.
    """
    p = panel.reset_index(drop=True)

    # seats / frequency: month-over-month seat change; a market's first month
    # has no previous month and is dropped by the finite mask.
    srt = p.sort_values(["market", "ym_str"])
    prev = srt["Total_Seats"].groupby(srt["market"]).shift(1)
    u_seats, w_seats = _clean(srt["Total_Seats"] / prev, srt["Total_Seats"])

    # load factor around its market seat-weighted mean.
    lf = p["Total_PAX"] / p["Total_Seats"]
    num = (lf * p["Total_Seats"]).groupby(p["market"]).transform("sum")
    den = p["Total_Seats"].groupby(p["market"]).transform("sum")
    u_lf, w_lf = _clean(lf / (num / den), p["Total_Seats"])

    # P2P share around its market PAX-weighted mean.
    p2ps = p["Total_P2P"] / p["Total_PAX"]
    num2 = (p2ps * p["Total_PAX"]).groupby(p["market"]).transform("sum")
    den2 = p["Total_PAX"].groupby(p["market"]).transform("sum")
    u_p2p, w_p2p = _clean(p2ps / (num2 / den2), p["Total_PAX"])

    # capture rate around the shipped (capped) capture[m], with a noise-aware
    # floor on terminating passengers.
    cap = p["NewArr"] / p["Total_P2P"]
    u_cap = (cap / p["market"].map(params["capture"])).to_numpy(dtype=float)
    w_cap = p["Total_P2P"].to_numpy(dtype=float)
    ok_cap = (
        np.isfinite(u_cap) & (w_cap >= min_p2p)
    )  # min_p2p is the passenger floor
    u_cap, w_cap = u_cap[ok_cap], w_cap[ok_cap]
    if u_cap.size == 0:
        raise ValueError(
            f"capture lever has no usable months under min_p2p={min_p2p}"
        )

    # ALOS around the shipped ALOS[m]; drop months with no arrivals/guests
    # (NaN/inf, removed by the finite mask).
    alos = p["Guests"] / p["NewArr"]
    u_alos, w_alos = _clean(alos / p["market"].map(params["alos"]), p["NewArr"])

    # season: the 12 fitted monthly multipliers themselves (already ~1).
    seasons = params["season"]
    u_seas = seasons.to_numpy(dtype=float)
    w_seas = np.ones_like(u_seas)

    return {
        "seats / frequency": (u_seats, w_seats),
        "load factor": (u_lf, w_lf),
        "P2P share (transfer/transit)": (u_p2p, w_p2p),
        "capture rate": (u_cap, w_cap),
        "ALOS": (u_alos, w_alos),
        "season": (u_seas, w_seas),
    }


def _kernel_bounds(u: np.ndarray, w: np.ndarray, kernel: str) -> tuple[float, float]:
    """16th–84th percentile band of ``u`` under one of three kernels."""
    if kernel == "uniform":
        lo, hi = np.quantile(u, [0.16, 0.84])
    elif kernel == "weighted":
        lo, hi = _weighted_quantile(u, w, 0.16), _weighted_quantile(u, w, 0.84)
    elif kernel == "winsorized":
        lo5, hi95 = np.quantile(u, [0.05, 0.95])
        uu = np.clip(u, lo5, hi95)
        lo, hi = _weighted_quantile(uu, w, 0.16), _weighted_quantile(uu, w, 0.84)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    return (float(lo), float(hi))


def empirical_lever_bounds(panel, params, kernel: str = "weighted", min_p2p: int = 50):
    """``{lever: (lo, hi)}`` Sobol ranges from the observed month multipliers.

    ``kernel`` selects the aggregation of each lever's ``u`` series:

    * ``"uniform"`` — plain 16th–84th percentiles (every month counts equally);
    * ``"weighted"`` (default, headline) — volume-weighted, so big markets set
      the "typical" band;
    * ``"winsorized"`` — clip ``u`` to its 5th–95th percentiles first, then the
      volume-weighted band (guards a few extreme months).
    """
    series = lever_series(panel, params, min_p2p=min_p2p)
    return {
        name: _kernel_bounds(series[name][0], series[name][1], kernel)
        for name in NAMES
    }


def lever_spread_table(panel, params, min_p2p: int = 50) -> pd.DataFrame:
    """One row per lever: volume-weighted q16 / median / q84 and spread."""
    series = lever_series(panel, params, min_p2p=min_p2p)
    rows = []
    for name in NAMES:
        u, w = series[name]
        q16 = _weighted_quantile(u, w, 0.16)
        med = _weighted_quantile(u, w, 0.50)
        q84 = _weighted_quantile(u, w, 0.84)
        rows.append(
            {
                "lever": name,
                "n": int(u.size),
                "q16": round(q16, 4),
                "median": round(med, 4),
                "q84": round(q84, 4),
                "middle spread (q84-q16)": round(q84 - q16, 4),
            }
        )
    return pd.DataFrame(rows)


def sobol_problem(bounds, names=NAMES):
    """SALib problem dict built from ``{name: (lo, hi)}`` bounds."""
    if set(names) != set(bounds) or len(names) != len(set(names)):
        raise ValueError("bounds keys must match NAMES exactly")
    return {
        "num_vars": len(names),
        "names": list(names),
        "bounds": [[float(bounds[n][0]), float(bounds[n][1])] for n in names],
    }


def sobol_sensitivity(
    panel, params, bounds, n: int = 256, seed: int = 42
) -> pd.DataFrame:
    """SALib Sobol on the chain under a set of lever bounds.

    Algebra is byte-for-byte the notebook T6 cell so the empirical and a-priori
    views are directly comparable. ``seed`` fixes SALib's scrambled sequence and
    its bootstrap resamples, so two calls return identical indices. Returns
    ``lever, elasticity, S1, ST`` sorted by ``ST`` descending.
    """
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    cap_arr = panel["market"].map(params["capture"]).to_numpy(dtype=float)
    alos_arr = panel["market"].map(params["alos"]).to_numpy(dtype=float)
    season_arr = panel["month"].map(params["season"]).to_numpy(dtype=float)
    seats0 = panel["Total_Seats"].to_numpy(dtype=float)
    lf0 = (panel["Total_PAX"] / panel["Total_Seats"]).to_numpy(dtype=float)
    p2p_share0 = (panel["Total_P2P"] / panel["Total_PAX"]).to_numpy(dtype=float)

    def total_nights(x):
        seats = seats0 * x[0]
        pax = seats * lf0 * x[1]
        p2p = pax * p2p_share0 * x[2]
        new = p2p * cap_arr * season_arr * x[3]
        nights = new * alos_arr * x[4] * x[5]
        return nights.sum()

    nights_base = total_nights([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    elasticities = {}
    for i, name in enumerate(NAMES):
        x = [1.0] * 6
        x[i] = 1.10
        elasticities[name] = (total_nights(x) / nights_base - 1.0) / 0.10

    problem = sobol_problem(bounds, names=NAMES)
    sample = sobol_sample.sample(
        problem, n, calc_second_order=False, seed=seed
    )
    y = np.array([total_nights(row) for row in sample])
    si = sobol_analyze.analyze(
        problem, y, calc_second_order=False, print_to_console=False, seed=seed
    )

    return (
        pd.DataFrame(
            {
                "lever": list(NAMES),
                "elasticity": [elasticities[n] for n in NAMES],
                "S1": si["S1"],
                "ST": si["ST"],
            }
        )
        .sort_values("ST", ascending=False)
        .reset_index(drop=True)
    )