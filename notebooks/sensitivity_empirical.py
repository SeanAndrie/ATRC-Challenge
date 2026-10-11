import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="Empirical lever spreads")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Empirical lever spreads — the second sensitivity view

    `notebooks/hybrid_layer.py` T6 decomposes the chain's output variance over
    **a-priori** lever ranges: the room the simulator lets a planner press
    (capture 0.7–1.3, seats 0.8–1.2, the rest 0.9–1.1). Those ranges are
    deliberately wider than observed, so T6 answers *which lever is **allowed**
    to move the number most*.

    This notebook is the honest companion (criterion 04): it measures how far
    each lever **actually moved** over the fit window, and runs the identical
    Sobol decomposition on those measured spreads.

    | | a-priori view (T6) | empirical view (this notebook) |
    | --- | --- | --- |
    | input ranges | 0.7–1.3, 0.8–1.2, 0.9–1.1 | volume-weighted 16th–84th pct of observed monthly multipliers |
    | question | *can* move the number | *did* move the number |
    | feeds | the simulator's `LEVER_BOUNDS` | reading and interpretation |
    | changes the UI? | — | **no** (kept separate, both labelled) |

    Sections **E1** spreads, **E2** empirical Sobol vs a-priori, **E3** kernel
    robustness, **E4** noise diagnostics. E5 (per-market shock decomposition)
    was planned only if the ranking were ambiguous; it is not, so it is omitted.
    """)
    return


@app.cell
def _():
    import pathlib
    import sys

    import numpy as np
    import pandas as pd

    here = (
        pathlib.Path(__file__).resolve().parent
        if "__file__" in globals()
        else pathlib.Path.cwd()
    )
    root = next(
        (c for c in (here, *here.parents) if (c / "pipeline" / "paths.py").exists()),
        here.parent,
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from pipeline import hybrid as H
    from pipeline import sensitivity as S

    return H, S, np, pd


@app.cell
def _(H):
    _out = H.build_panels()
    _fa, _mkt, _mon, panel, _test_m, _test_panel, _train_m = _out
    params = H.fit_params(panel)
    return panel, params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E1 — What "typical" looks like for each lever

    Each lever's monthly multiplier is `u = z / baseline` (baselines per lever in
    `pipeline/sensitivity.py`). The table shows the volume-weighted 16th
    percentile, median and 84th percentile of `u`, i.e. the middle-band wobble a
    big market actually experienced; the second table shows the resulting Sobol
    range under three aggregation kernels.
    """)
    return


@app.cell
def _(S, panel, params, pd):
    spread = S.lever_spread_table(panel, params)

    _rows = []
    for _kernel in ("uniform", "weighted", "winsorized"):
        _b = S.empirical_lever_bounds(panel, params, kernel=_kernel)
        for _name in S.NAMES:
            _lo, _hi = _b[_name]
            _rows.append(
                {
                    "kernel": _kernel,
                    "lever": _name,
                    "lo": round(_lo, 3),
                    "hi": round(_hi, 3),
                    "width": round(_hi - _lo, 3),
                }
            )
    bounds = pd.DataFrame(_rows)
    return bounds, spread


@app.cell(hide_code=True)
def _(bounds, mo, spread):
    mo.vstack(
        [
            mo.md(
                "**E1 — measured spreads.** The `weighted` column is the headline: "
                "big markets set the typical band. Capture is the widest lever "
                "(~0.67–1.29); season is the narrowest (~0.93–1.05). The uniform "
                "and winsorized kernels bracket it."
            ),
            spread,
            bounds.pivot(index="lever", columns="kernel", values="width")
            .reindex(
                [
                    "seats / frequency",
                    "load factor",
                    "P2P share (transfer/transit)",
                    "capture rate",
                    "ALOS",
                    "season",
                ]
            )
            .round(3),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E2 — Empirical Sobol vs the a-priori T6 Sobol

    The same SALib decomposition, run twice: once on the **measured** bounds
    (E1, weighted) and once on the **a-priori** decision ranges. Both use an
    identical panel and the identical `total_nights` algebra as the notebook's
    T6 cell, so the two columns are directly comparable.

    The honest read: **capture dominates both views** — it is both the widest
    measured lever and the one whose baseline level already varies hugely across
    markets. The views disagree about #2. A-priori grants seats/frequency a wide
    0.8–1.2 and it ranks second; measured, seat supply only wobbles ~±9%
    month-over-month, so it falls and **P2P share** (the widest measured supply
    lever) takes its place. That difference is the whole point of measuring.
    """)
    return


@app.cell
def _(S, panel, params, pd):
    import plotly.graph_objects as go

    apriori_bounds = dict(
        zip(
            S.NAMES,
            [[0.8, 1.2], [0.9, 1.1], [0.9, 1.1], [0.7, 1.3], [0.9, 1.1], [0.9, 1.1]],
        )
    )
    empirical_bounds = S.empirical_lever_bounds(panel, params, kernel="weighted")

    sens_emp = S.sobol_sensitivity(panel, params, empirical_bounds)
    sens_apri = S.sobol_sensitivity(panel, params, apriori_bounds)

    compare = (
        sens_emp[["lever", "ST"]]
        .rename(columns={"ST": "ST empirical"})
        .merge(
            sens_apri[["lever", "ST"]].rename(columns={"ST": "ST a-priori"}),
            on="lever",
        )
        .assign(
            rank_empirical=lambda d: d["ST empirical"].rank(ascending=False).astype(int),
            rank_apriori=lambda d: d["ST a-priori"].rank(ascending=False).astype(int),
        )
        .sort_values("ST empirical", ascending=False)
        .reset_index(drop=True)
    )

    fig = go.Figure(
        go.Bar(
            x=sens_emp["ST"],
            y=sens_emp["lever"],
            orientation="h",
            marker_color="#4c78a8",
        )
    )
    fig.update_layout(
        title="Empirical Sobol total-order sensitivity of guest-nights",
        xaxis_title="ST (share of output variance)",
        yaxis_title="",
        height=360,
    )
    return compare, empirical_bounds, fig, sens_emp


@app.cell(hide_code=True)
def _(compare, empirical_bounds, fig, mo):
    mo.vstack(
        [
            mo.md("**E2 — measured bounds used for the empirical run.**"),
            {k: (round(v[0], 3), round(v[1], 3)) for k, v in empirical_bounds.items()},
            mo.md("**E2 — ranking, measured vs a-priori.**"),
            compare.round(3),
            fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E3 — Kernel robustness

    The headline uses volume weighting. Re-running the Sobol on the uniform and
    winsorized bounds checks that the ranking is not an artefact of that choice.
    """)
    return


@app.cell
def _(S, panel, params, pd):
    _frames = []
    for _kernel in ("uniform", "weighted", "winsorized"):
        _b = S.empirical_lever_bounds(panel, params, kernel=_kernel)
        _st = S.sobol_sensitivity(panel, params, _b).set_index("lever")["ST"]
        _frames.append(_st.rename(_kernel))
    robust = pd.concat(_frames, axis=1).reindex(S.NAMES)
    robust["rank spread"] = (
        robust.rank(ascending=False).max(axis=1)
        - robust.rank(ascending=False).min(axis=1)
    ).astype(int)
    return (robust,)


@app.cell(hide_code=True)
def _(mo, robust):
    mo.vstack(
        [
            mo.md(
                "**E3 — ST by kernel.** A `rank spread` of 0 means every kernel "
                "puts the lever in the same position. The top of the ranking is "
                "kernel-stable; the small middle-order differences do not change "
                "any conclusion."
            ),
            robust.round(3),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E4 — Noise diagnostics

    Two sources of fragility are measured rather than hidden. First, the capture
    lever's `min_p2p` floor: thin markets have noisy `NewArr / P2P`, so the
    default drops months with fewer than 50 terminating passengers. Second, the
    ALOS lever drops months with no arrivals or no guests; those counts are
    reported.
    """)
    return


@app.cell
def _(S, panel, params, pd):
    _floors = [0, 25, 50, 100, 250, 500]
    _rows = []
    for _f in _floors:
        _b = S.empirical_lever_bounds(panel, params, kernel="weighted", min_p2p=_f)
        _n = S.lever_series(panel, params, min_p2p=_f)["capture rate"][0].size
        _rows.append(
            {
                "min_p2p": _f,
                "capture months kept": int(_n),
                "capture lo": round(_b["capture rate"][0], 3),
                "capture hi": round(_b["capture rate"][1], 3),
                "capture width": round(_b["capture rate"][1] - _b["capture rate"][0], 3),
            }
        )
    floor_table = pd.DataFrame(_rows)

    _alos_u = S.lever_series(panel, params)["ALOS"][0]
    alos_counts = {
        "panel rows": int(len(panel)),
        "ALOS months kept": int(_alos_u.size),
        "ALOS months dropped (0 arrivals/guests)": int(len(panel) - _alos_u.size),
    }
    return alos_counts, floor_table


@app.cell(hide_code=True)
def _(alos_counts, floor_table, mo):
    mo.vstack(
        [
            mo.md(
                "**E4 — capture floor.** Raising `min_p2p` trims the noisiest "
                "months; the capture band barely moves, so the headline does not "
                "hinge on the floor."
            ),
            floor_table,
            mo.md("**E4 — ALOS drop diagnosis.**"),
            alos_counts,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this changes, and what it does not

    - **It changes the write-up, not the model.** METHODS §3.4 now reports two
      labelled sensitivity views; the notebook T6 cell and its a-priori ranges
      are untouched, and the simulator's `LEVER_BOUNDS` are untouched.
    - **It sharpens criterion 04.** "Which lever matters" now has an empirical
      answer (capture, then P2P share) alongside the decision-space answer
      (capture, then seats) — with the disagreement between them stated, not
      hidden.
    - **It stays honest about the join.** These are relative movements of
      already-contaminated per-market coefficients (§4.5). They describe how the
      *measured* chain behaves; they do not rescue market-level attribution.
    """)
    return


if __name__ == "__main__":
    app.run()
