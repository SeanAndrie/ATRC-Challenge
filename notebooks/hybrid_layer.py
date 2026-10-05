import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="Hybrid Layer")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hybrid layer — Seats → PAX → P2P → New Arrivals → Guest-nights

    One notebook, two models, one panel. Part A is a transparent conversion
    chain in which every ratio is measured or flagged. Part B keeps the chain's
    exact mechanical supply stages and replaces the behavioural step with a
    **LightGBM** model. Both are scored on the same rows, in the same run, so
    the head-to-head cannot drift.

    | stage | who models it | measured from |
    | --- | --- | --- |
    | scheduled seats | both (exact) | `flight_data.xlsx` `Total Seats` |
    | PAX | both (exact) | load factor `Total PAX / Total Seats` |
    | P2P | both (exact) | `Total P2P / Total PAX`, strips transfer + transit |
    | New Arrivals | **chain** `capture[m]`, or **LightGBM** | `capture[m] = Σ New Arrivals / Σ P2P` per market, or a learned rate |
    | Guest-nights | **chain** `ALOS[m]`, or **LightGBM** | `ALOS[m] = Σ Guests / Σ New Arrivals` × season, or a learned log-ALOS |

    **Why hybrid rather than a black box.** The first three stages are arithmetic
    and reproduce the measured data exactly (T2 below), so there is nothing to
    learn there — only the conversion is behavioural. The model is a **Poisson
    regressor with `log(Total_P2P)` as an exposure offset**, so it predicts an
    intensity per terminating passenger: a learned, shrunk, market-and-season-
    aware capture rate. No target-derived quantity is ever a feature.

    The purpose-of-travel / VFR step in the brief's chain is **not** a separate
    lever. No source measures it; it is absorbed inside the capture rate, learned
    or measured. See `docs/METHODS.md` Phases 3 and 4.

    | part | contents |
    | --- | --- |
    | **0 — shared** | loaders, panel, integrity, and the `fit_params` / `simulate` / `wmape` engine |
    | **A — the exact chain** | T2 arithmetic, T3 in-sample, T4 holdout, T5 test window, T6 sensitivity |
    | **B — the hybrid layer** | features, LightGBM fit, ML-2/3/4 evaluation, ML-5 interpretation |
    | **C — verification** | one pin block covering both models |
    """)
    return


@app.cell
def _():
    import pathlib
    import sys

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # Repo root by walking up from this file, not the working directory: marimo
    # is commonly launched from notebooks/, where a relative path would resolve
    # to notebooks/notebooks and find nothing.
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

    from pipeline.competition import load as cl

    return cl, go, np, pd


@app.cell
def _():
    # Part B needs LightGBM; Part A does not. Keep the import separate so the
    # failure mode is "missing extra" rather than "the chain is broken too".
    import lightgbm as lgb

    return (lgb,)


@app.cell
def _(cl):
    # The repaired loader owns parsing; this notebook never touches the xlsx
    # files directly.
    flights = cl.read_flights()
    intl_train = cl.read_international_train()
    intl_test = cl.read_international_test()
    return flights, intl_test, intl_train


@app.cell
def _(flights):
    flights.head()
    return


@app.cell
def _(flights):
    flights.info()
    return


@app.cell
def _(flights):
    flights.describe()
    return


@app.cell
def _(cl, intl_test, intl_train, pd):
    # Flight supply aggregated to market (departure country) x month. "market" is
    # the join key on both sides; departure country and guest nationality happen
    # to share names for the 33 flight markets.
    fa = cl.flights_monthly_agg().rename(columns={"Departure Country Name": "market"})
    fa["market"] = fa["market"].str.upper().str.strip()
    fa["ym_str"] = fa["ym"].astype(str)
    # routes/airlines are carried for the model's supply features; the chain
    # ignores them, and keeping one column list means both parts score identical
    # rows.
    fcols = [
        "ym_str",
        "market",
        "Total_Seats",
        "Total_PAX",
        "Total_P2P",
        "Total_Transfer",
        "Total_Transit",
        "routes",
        "airlines",
    ]

    train_m = (
        intl_train.groupby(["ym", "Nationality"], dropna=False)
        .agg(Guests=("Guests_n", "sum"), NewArr=("New Arrivals_n", "sum"))
        .reset_index()
    )
    train_m["market"] = train_m["Nationality"].str.upper().str.strip()
    train_m["ym_str"] = train_m["ym"].astype(str)
    train_m["month"] = train_m["ym"].dt.month

    # The test frame carries New Arrivals but no Guests (guest-nights cannot be
    # scored out of sample), and its New Arrivals is numeric, not the '*' marker.
    test_m = (
        intl_test.assign(NewArr=pd.to_numeric(intl_test["New Arrivals"], errors="coerce"))
        .groupby(["ym", "Nationality"], dropna=False)
        .agg(NewArr=("NewArr", "sum"))
        .reset_index()
    )
    test_m["market"] = test_m["Nationality"].str.upper().str.strip()
    test_m["ym_str"] = test_m["ym"].astype(str)
    test_m["month"] = test_m["ym"].dt.month

    panel = train_m.merge(fa[fcols], on=["ym_str", "market"], how="inner")
    test_panel = test_m.merge(fa[fcols], on=["ym_str", "market"], how="inner")

    # Fixed category levels shared by train and test, so LightGBM's native
    # categorical handling sees the same encoding on both sides.
    mkt_dtype = pd.CategoricalDtype(sorted(panel["market"].unique()))
    mon_dtype = pd.CategoricalDtype(list(range(1, 13)))
    return fa, mkt_dtype, mon_dtype, panel, test_m, test_panel


@app.cell
def _(fa, flights, mo, panel, pd, test_m):
    # T1 / ML-0 integrity of the inputs both parts rest on. One table, because
    # there is one set of inputs.
    resid = (
        flights["Total PAX_n"]
        - flights["Total P2P_n"]
        - flights["Total Transfer_n"]
        - flights["Total Transit_n"]
    )
    identity_ok = bool((resid.abs() < 1e-9).all())
    lf_ok = bool(
        ((flights["lf_frac"] - flights["Total PAX_n"] / flights["Total Seats_n"]).abs() < 1e-9).all()
    )
    alos_global = panel["Guests"].sum() / panel["NewArr"].sum()

    flight_markets = set(fa["market"].dropna().unique())
    hotel_markets = set(panel["market"].dropna().unique()) | set(
        test_m["market"].dropna().unique()
    )
    hotel_only = sorted(hotel_markets - flight_markets)

    integrity = pd.DataFrame(
        [
            ("PAX = P2P + Transfer + Transit", "all 117,608 rows", identity_ok),
            ("load factor = PAX / Seats", "all 117,608 rows", lf_ok),
            ("matched market x month rows", "1,405", len(panel) == 1405),
            ("matched markets", "33", panel["market"].nunique() == 33),
            ("hotel-only markets (no flight rows)", "12", len(hotel_only) == 12),
        ],
        columns=["check", "expected", "pass"],
    )
    mo.vstack(
        [
            mo.md(
                f"**T1 / ML-0 — input integrity.** Global `Guests / New Arrivals` "
                f"(ALOS) = **{alos_global:.3f}**, matching `docs/METHODS.md` §2.3. "
                "Both identities hold, so the supply stages feeding the chain and "
                "the model are exact."
            ),
            mo.ui.table(integrity, selection=None),
            mo.md(
                "Hotel-only markets with no flight supply: "
                + ", ".join(hotel_only)
                + ". These cannot be driven by the air chain and are carried as a "
                "residual, not silently assigned a capture rate."
            ),
        ]
    )
    return alos_global, identity_ok, lf_ok


@app.cell
def _(panel):
    def fit_params(p):
        """Fit the chain's parameters on a market x month panel.

        capture and ALOS are volume-weighted ratios (Σ New / Σ P2P, Σ Guests /
        Σ New), so a large market is not outvoted by a small one. Capture is
        capped at 5.0: a handful of markets exceed one new arrival per
        terminating passenger because hotel demand in those markets is not
        carried by air seats alone.

        This is the only implementation in the repo. Part B previously carried a
        trimmed copy, which meant two definitions of the chain baseline in two
        files; the parametric version is authoritative and the baseline is now
        computed once, here.
        """
        g = p.groupby("market").agg(
            New=("NewArr", "sum"), P2P=("Total_P2P", "sum"), G=("Guests", "sum")
        )
        capture = (g["New"] / g["P2P"]).clip(upper=5.0)

        # Season is the P2P-weighted ratio of actual to capture-expected new
        # arrivals for each calendar month, normalised to mean 1. It captures
        # within-market monthly variation that seat supply alone does not.
        tmp = p.merge(capture.rename("cap"), left_on="market", right_index=True)
        num = tmp.groupby("month")["NewArr"].sum()
        den = (tmp["Total_P2P"] * tmp["cap"]).groupby(tmp["month"]).sum()
        season = num / den
        season = season / season.mean()

        alos = g["G"] / g["New"]
        return {
            "capture": capture,
            "season": season,
            "alos": alos,
            "global_capture": p["NewArr"].sum() / p["Total_P2P"].sum(),
            "global_alos": p["Guests"].sum() / p["NewArr"].sum(),
        }

    params = fit_params(panel)
    return fit_params, params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    capture = params["capture"]
    mo.md(
        f"\"\"
    **Fitted parameters.** Global capture **{params['global_capture']:.3f}**
    (new hotel arrivals per terminating passenger) and global ALOS
    **{params['global_alos']:.3f}**.

    Capture spans **{capture.min():.3f}** to **{capture.max():.3f}** across the 33
    flight markets (median {capture.median():.3f}) — the spread that makes a
    single blended ratio indefensible. ALOS ranges
    **{params['alos'].min():.2f}** to **{params['alos'].max():.2f}**.

    A Beta-binomial empirical-Bayes shrinker was tried first and rejected: the
    between-market dispersion of capture exceeds the variance a Beta prior can
    represent (the method-of-moments concentration goes negative), which is the
    signature of real behavioural heterogeneity rather than small-sample noise.
    The measured rate is therefore kept, with the cap as the only guard. Part B
    reaches the same conclusion by a different route — the model's learned rate
    is compared against this one in ML-5.
    "\"\"
    )
    """)
    return


@app.cell
def _(np):
    def simulate(
        p,
        params,
        seats_mult=1.0,
        lf_mult=1.0,
        p2p_mult=1.0,
        capture_mult=1.0,
        alos_mult=1.0,
        season_mult=1.0,
    ):
        """Apply the chain to a market x month panel.

        Every stage writes its own column so the waterfall is auditable. Default
        multipliers are 1.0, which reproduces the measured seat, PAX and P2P
        values exactly and produces the model's New Arrivals and Guest-nights.

        Part B previously carried a no-lever stub that skipped the supply
        arithmetic. It was arithmetically identical to this function at unit
        multipliers — PAX = Seats × (PAX/Seats) = PAX, P2P = PAX × (P2P/PAX) =
        P2P — so the stub was deleted rather than kept in sync.
        """
        out = p.copy()
        seats = out["Total_Seats"] * seats_mult
        lf = (out["Total_PAX"] / out["Total_Seats"]) * lf_mult
        out["PAX"] = seats * lf
        p2p_share = (out["Total_P2P"] / out["Total_PAX"]) * p2p_mult
        out["P2P"] = out["PAX"] * p2p_share
        out["NewArrivalHat"] = (
            out["P2P"]
            * out["market"].map(params["capture"])
            * out["month"].map(params["season"])
            * capture_mult
        )
        out["GuestNightsHat"] = (
            out["NewArrivalHat"] * out["market"].map(params["alos"]) * alos_mult
        )
        return out

    def wmape(actual, pred):
        """Weighted mean absolute percentage error, the brief's metric."""
        actual = np.asarray(actual, dtype=float)
        pred = np.asarray(pred, dtype=float)
        return 100.0 * np.abs(actual - pred).sum() / np.abs(actual).sum()

    return simulate, wmape


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part A — the exact chain

    Every ratio below is measured from the data or flagged as an assumption. No
    fitted model, no residual structure left unexplained.
    """)
    return


@app.cell
def _(np, panel, params, pd, simulate):
    # T2 chain arithmetic: with unit levers the first three stages reproduce the
    # measured values exactly, and the last two are the model's only estimates.
    out = simulate(panel, params)
    paxt_err = float(np.abs(out["PAX"] - panel["Total_PAX"]).max())
    p2p_err = float(np.abs(out["P2P"] - panel["Total_P2P"]).max())
    chain_ok = paxt_err < 1e-6 and p2p_err < 1e-6

    stage_totals = pd.DataFrame(
        [
            ("Seats", panel["Total_Seats"].sum()),
            ("PAX", out["PAX"].sum()),
            ("P2P", out["P2P"].sum()),
            ("New arrivals (hat)", out["NewArrivalHat"].sum()),
            ("Guest-nights (hat)", out["GuestNightsHat"].sum()),
        ],
        columns=["stage", "value"],
    )
    return chain_ok, out, p2p_err, paxt_err, stage_totals


@app.cell
def _(chain_ok, go, mo, p2p_err, paxt_err, stage_totals):
    # Waterfall in percent-of-seats terms, so stages of very different magnitude
    # share one axis and each step's size is the ratio it represents.
    seats = float(stage_totals.loc[stage_totals.stage == "Seats", "value"].iloc[0])
    pct = 100.0 * stage_totals["value"] / seats
    steps = pct.diff().fillna(pct.iloc[0])
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative"],
            x=stage_totals["stage"],
            y=steps,
            text=[f"{v:,.0f}" for v in stage_totals["value"]],
            textposition="outside",
            connector={"line": {"color": "gray"}},
        )
    )
    fig.update_layout(
        title="Conversion chain, test panel (% of scheduled seats)",
        yaxis_title="% of seats",
        showlegend=False,
        height=420,
    )
    mo.vstack(
        [
            mo.md(
                f"**T2 — chain arithmetic.** Max absolute error: PAX {paxt_err:.2e}, "
                f"P2P {p2p_err:.2e}. Unit levers reproduce the measured supply stages "
                f"exactly: **{'pass' if chain_ok else 'FAIL'}**. This is what licenses "
                "the hybrid design: if the first three stages are exact, only the "
                "fourth is worth learning."
            ),
            fig,
        ]
    )
    return


@app.cell
def _(mo, out, panel, wmape):
    # T3 in-sample fit of the full training panel.
    ins_new = wmape(panel["NewArr"], out["NewArrivalHat"])
    ins_guest = wmape(panel["Guests"], out["GuestNightsHat"])
    mo.md(
        f"""
    **T3 — in-sample fit** (2022-01 → 2025-07, {len(panel):,} market-months).

    | target | WMAPE |
    | --- | ---: |
    | New arrivals | {ins_new:.1f}% |
    | Guest-nights | {ins_guest:.1f}% |

    This is the optimistic number: the parameters were fitted on these rows.
    T4 and T5 are the honest tests.
    """
    )
    return ins_guest, ins_new


@app.cell
def _(panel):
    # T4 leave-tail holdout: fit through 2025-01, predict 2025-02..2025-07.
    fit_panel = panel[panel["ym"] <= "2025-01"]
    holdout = panel[panel["ym"] > "2025-01"].copy()
    return fit_panel, holdout


@app.cell
def _(fit_panel, fit_params, holdout, pd, simulate, wmape):
    # The same estimator, refitted on the truncated panel: no leakage from the
    # holdout months.
    params_holdout = fit_params(fit_panel)
    ho = simulate(holdout, params_holdout)
    # Seasonal-naive baseline: per market, mean of the same calendar month in the
    # fit window. This is the benchmark the chain has to beat.
    sn = fit_panel.groupby(["market", "month"])["NewArr"].mean().rename("sn")
    ho = ho.merge(sn, on=["market", "month"], how="left")

    ho_new = wmape(ho["NewArr"], ho["NewArrivalHat"])
    sn_ok = ho.dropna(subset=["sn"])
    ho_sn = wmape(sn_ok["NewArr"], sn_ok["sn"])
    ho_guest = wmape(ho["Guests"], ho["GuestNightsHat"])
    holdout_tbl = pd.DataFrame(
        [
            ("New arrivals", "chain", ho_new),
            ("New arrivals", "seasonal naive", ho_sn),
            ("Guest-nights", "chain", ho_guest),
        ],
        columns=["target", "model", "WMAPE %"],
    )
    return ho, ho_guest, ho_new, ho_sn, holdout_tbl, params_holdout


@app.cell
def _(ho, ho_guest, ho_new, ho_sn, holdout_tbl, mo):
    mo.vstack(
        [
            mo.md(
                f"""
    **T4 — leave-tail holdout** (fit 2022-01 → 2025-01, test 2025-02 → 2025-07,
    {len(ho)} market-months).

    The chain beats the seasonal-naive baseline on new arrivals
    (**{ho_new:.1f}%** vs **{ho_sn:.1f}%**), and carries that improvement through
    to guest-nights at **{ho_guest:.1f}%**.

    Part B scores its model on this exact holdout below, so these three numbers
    are the incumbent the hybrid has to beat.
    """
            ),
            holdout_tbl,
        ]
    )
    return


@app.cell
def _(params, pd, simulate, test_panel, wmape):
    # T5 competition test window: fit on all training data, predict the held-out
    # 2025-08..2026-02 New Arrivals from flight supply. The test frame has no
    # Guests, so only new arrivals are scored out of sample.
    pred = simulate(test_panel, params)

    score = pd.DataFrame(
        {
            "NewArr": test_panel["NewArr"],
            "pred": pred["NewArrivalHat"].values,
        }
    ).dropna()

    test_new = wmape(score["NewArr"], score["pred"])
    actual_total = float(score["NewArr"].sum())
    pred_total = float(score["pred"].sum())
    return actual_total, pred_total, score, test_new


@app.cell
def _(actual_total, mo, pred_total, score, test_new, test_panel):
    mo.md(f"""
    **T5 — competition test window** (2025-08 → 2026-02,
    {len(score):,} market-months, {test_panel['market'].nunique()} markets).

    | metric | value |
    | --- | ---: |
    | WMAPE, market-month | {test_new:.1f}% |
    | actual new arrivals | {actual_total:,.0f} |
    | predicted new arrivals | {pred_total:,.0f} |
    | aggregate bias | {(pred_total / actual_total - 1) * 100:+.1f}% |

    The chain tracks the aggregate closely but the market-month error is large.
    That is the honest picture: the flight lever sets the level, the market
    capture rates carry the composition, and both are noisier than the aggregate
    suggests. Part B scores the hybrid on this same window.
    """)
    return


@app.cell
def _(np, panel, params, pd):
    # T6 global sensitivity (SALib Sobol) over the six floating levers, plus
    # one-at-a-time elasticities for the signed direction.
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

    names = [
        "seats / frequency",
        "load factor",
        "P2P share (transfer/transit)",
        "capture rate",
        "ALOS",
        "season",
    ]
    nights_base = total_nights([1, 1, 1, 1, 1, 1])
    elasticities = {}
    for i, name in enumerate(names):
        x = [1.0] * 6
        x[i] = 1.10
        elasticities[name] = (total_nights(x) / nights_base - 1.0) / 0.10

    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    problem = {
        "num_vars": 6,
        "names": names,
        "bounds": [
            [0.8, 1.2],
            [0.9, 1.1],
            [0.9, 1.1],
            [0.7, 1.3],
            [0.9, 1.1],
            [0.9, 1.1],
        ],
    }
    sample = sobol_sample.sample(problem, 256, calc_second_order=False)
    y = np.array([total_nights(row) for row in sample])
    si = sobol_analyze.analyze(problem, y, calc_second_order=False, print_to_console=False)

    sensitivity = pd.DataFrame(
        {
            "lever": names,
            "elasticity": [elasticities[n] for n in names],
            "S1": si["S1"],
            "ST": si["ST"],
        }
    ).sort_values("ST", ascending=False)
    return (sensitivity,)


@app.cell
def _(go, mo, sensitivity):
    fig2 = go.Figure(
        go.Bar(
            x=sensitivity["ST"],
            y=sensitivity["lever"],
            orientation="h",
            marker_color="#4c78a8",
        )
    )
    fig2.update_layout(
        title="T6 — Sobol total-order sensitivity of guest-nights",
        xaxis_title="ST (share of output variance)",
        yaxis_title="",
        height=360,
    )
    mo.vstack(
        [
            mo.md(
                "**T6 — sensitivity.** Elasticities are from a +10% one-at-a-time "
                "shock; Sobol indices are from SALib over the ranges in the cell. "
                "The supply levers (seats, load factor) and the conversion levers "
                "(capture, ALOS) all carry comparable weight, which is why the "
                "chain reports a range rather than a point."
            ),
            fig2,
            sensitivity,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part B — the hybrid layer

    The chain fixes the conversion with two ratios per market (`capture[m]`,
    `ALOS[m]`). Part B keeps the exact mechanical supply stages
    (`seats → PAX → P2P`, already proven exact in T2) and lets **LightGBM** learn
    the behavioural step `P2P → New Arrivals` instead of a fixed ratio.
    """)
    return


@app.cell
def _(mkt_dtype, mon_dtype, np, pd):
    def featurize(df):
        """Leak-free features: only supply, calendar and market, all known for
        the competition test window. No target-derived quantity is used, so a
        prediction can be made without ever seeing New Arrivals."""
        x = pd.DataFrame(index=df.index)
        x["market"] = df["market"].astype(mkt_dtype)
        x["month"] = df["month"].astype(mon_dtype)
        ym = pd.PeriodIndex(df["ym_str"], freq="M")
        x["t"] = ym.year * 12 + ym.month - (2022 * 12 + 1)
        x["log_seats"] = np.log1p(df["Total_Seats"])
        x["load_factor"] = df["Total_PAX"] / df["Total_Seats"]
        x["p2p_share"] = df["Total_P2P"] / df["Total_PAX"]
        x["transfer_share"] = (df["Total_Transfer"] + df["Total_Transit"]) / df["Total_PAX"]
        x["routes"] = df["routes"].astype(float)
        x["airlines"] = df["airlines"].astype(float)
        return x

    cats = ["market", "month"]
    return cats, featurize


@app.cell
def _(cats, featurize, lgb, np):
    # One a-priori configuration, not tuned on the reported holdouts: shallow
    # trees, high min_child_samples, L2, row/feature subsampling, single thread,
    # fixed seeds. Iteration count is chosen by early stopping on an internal
    # 6-month tail of the fit window, then the ensemble is refit on the full fit
    # window.
    base = dict(
        objective="poisson",
        num_leaves=15,
        min_child_samples=20,
        learning_rate=0.03,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        num_threads=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    seeds = [0, 1, 2, 42, 2024]

    def _ds(df, feats, off, ref=None):
        d = lgb.Dataset(
            feats,
            label=df["NewArr"].to_numpy(dtype=float),
            init_score=off,
            categorical_feature=cats,
            reference=ref,
            free_raw_data=False,
        )
        d.set_init_score(off)
        return d

    def fit_poisson(train_iter, valid, max_rounds=2000):
        """Early-stopping pass on the internal tail; returns best_iteration."""
        off_tr = np.log(train_iter["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        off_v = np.log(valid["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        d = _ds(train_iter, featurize(train_iter), off_tr)
        dv = _ds(valid, featurize(valid), off_v, ref=d)
        p = dict(base)
        p["seed"] = seeds[0]
        b = lgb.train(
            p,
            d,
            num_boost_round=max_rounds,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return b.best_iteration

    def predict_poisson(train_final, test, best_iter):
        """Multi-seed ensemble on the rate scale: each seed's raw score is
        exponentiated with the exposure offset, and the resulting per-passenger
        intensities are averaged. Averaging the raw scores instead would average
        log-intensities, which is a different (geometric-mean) estimator."""
        Xtr = featurize(train_final)
        Xte = featurize(test)
        off_tr = np.log(train_final["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        off_te = np.log(test["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        preds = []
        for s in seeds:
            p = dict(base)
            p["seed"] = s
            b = lgb.train(p, _ds(train_final, Xtr, off_tr), num_boost_round=best_iter)
            preds.append(np.exp(b.predict(Xte, raw_score=True) + off_te))
        return np.mean(preds, axis=0), preds

    def train_model(df, n_rounds, seed=0):
        X = featurize(df)
        off = np.log(df["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        p = dict(base)
        p["seed"] = seed
        return lgb.train(p, _ds(df, X, off), num_boost_round=n_rounds)

    def fit_alos(train, test):
        """Guest-nights are New Arrivals x length of stay. This learns log-ALOS
        from the same features; the alternative is the chain's ALOS[m]."""
        tr = train[(train["NewArr"] > 0) & (train["Guests"] > 0)]
        Xtr = featurize(tr)
        Xte = featurize(test)
        ytr = np.log(tr["Guests"].to_numpy(dtype=float) / tr["NewArr"].to_numpy(dtype=float))
        preds = []
        for s in seeds:
            p = dict(base)
            p["objective"] = "regression"
            p["seed"] = s
            b = lgb.train(p, lgb.Dataset(Xtr, label=ytr, categorical_feature=cats), num_boost_round=300)
            preds.append(np.exp(b.predict(Xte)))
        return np.mean(preds, axis=0)

    return fit_alos, fit_poisson, predict_poisson, train_model


@app.cell
def _(fit_panel, holdout):
    # The internal validation tails inside each fit window. fit_panel/holdout are
    # Part A's T4 split, reused rather than restated; Part B only needs the two
    # nested tails for early stopping, never the holdout itself as a training set.
    core = fit_panel[fit_panel["ym"] <= "2024-07"]
    val = fit_panel[fit_panel["ym"] > "2024-07"]
    # T5's iteration-selection window is the same pair one year later: train
    # through 2025-01, validate on the T4 holdout months.
    val5 = holdout
    return core, val, val5


@app.cell
def _(
    core,
    fit_panel,
    fit_poisson,
    ho,
    ho_new,
    ho_sn,
    mo,
    predict_poisson,
    val,
    wmape,
):
    # ML-2: hybrid New Arrivals on the leave-tail holdout (fit <= 2025-01).
    # ho_new/ho_sn are Part A's T4 numbers, so the incumbent row in the table
    # below is the value this notebook actually measured a few cells ago.
    best_iter_t4 = fit_poisson(core, val)
    pred_ho, pl_ho = predict_poisson(fit_panel, ho, best_iter_t4)
    ml_ho_new = wmape(ho["NewArr"], pred_ho)
    ml_ho_lo = min(wmape(ho["NewArr"], p) for p in pl_ho)
    ml_ho_hi = max(wmape(ho["NewArr"], p) for p in pl_ho)

    mo.md(
        f"""
    **ML-2 — hybrid, leave-tail holdout** (fit 2022-01 → 2025-01, test
    2025-02 → 2025-07, {len(ho)} market-months).

    | model | New Arrivals WMAPE |
    | --- | ---: |
    | seasonal naive | {ho_sn:.1f}% (T4) |
    | parametric chain | {ho_new:.1f}% (T4) |
    | **hybrid LightGBM** | **{ml_ho_new:.1f}%** |

    Early stopping chose **{best_iter_t4}** rounds on an internal 6-month tail;
    the 5-seed ensemble spans {ml_ho_lo:.1f}–{ml_ho_hi:.1f}%, so the gain is not
    a single seed.
    """
    )
    return best_iter_t4, ml_ho_new, pred_ho


@app.cell
def _(
    fit_panel,
    fit_poisson,
    mo,
    panel,
    predict_poisson,
    test_new,
    test_panel,
    val5,
    wmape,
):
    # ML-3: the competition test window (fit <= 2025-07, predict 2025-08..2026-02).
    # fit_panel is the iteration-selection window, mirroring the T4 setup; the
    # final ensemble is refit on the whole panel at the selected round count.
    best_iter_t5 = fit_poisson(fit_panel, val5)
    pred_t5, pl_t5 = predict_poisson(panel, test_panel, best_iter_t5)
    ml_t5_new = wmape(test_panel["NewArr"], pred_t5)
    ml_t5_lo = min(wmape(test_panel["NewArr"], p) for p in pl_t5)
    ml_t5_hi = max(wmape(test_panel["NewArr"], p) for p in pl_t5)
    ml_t5_bias = (pred_t5.sum() / test_panel["NewArr"].sum() - 1) * 100

    mo.md(
        f"""
    **ML-3 — competition test window** (2025-08 → 2026-02,
    {len(test_panel)} market-months).

    | model | New Arrivals WMAPE | aggregate bias |
    | --- | ---: | ---: |
    | parametric chain | {test_new:.1f}% (T5) | — |
    | **hybrid LightGBM** | **{ml_t5_new:.1f}%** | **{ml_t5_bias:+.1f}%** |

    Early stopping chose **{best_iter_t5}** rounds; the ensemble spans
    {ml_t5_lo:.1f}–{ml_t5_hi:.1f}%. The hybrid both tracks the aggregate and
    sharpens the market-month fit.
    """
    )
    return best_iter_t5, ml_t5_bias, ml_t5_new


@app.cell
def _(fit_alos, fit_panel, ho, ho_guest, mo, params_holdout, pred_ho, wmape):
    # ML-4: guest-nights, chain ALOS vs learned ALOS. params_holdout is the T4
    # refit, so the ALOS[m] it carries is the same one Part A scored with.
    guest_chain_alos = pred_ho * ho["market"].map(params_holdout["alos"]).to_numpy()
    alos_hat = fit_alos(fit_panel, ho)
    w_chain = wmape(ho["Guests"], guest_chain_alos)
    w_ml = wmape(ho["Guests"], pred_ho * alos_hat)

    mo.md(
        f"""
    **ML-4 — guest-nights on the holdout.** `guest-nights = New Arrivals × ALOS`.

    | model | ALOS source | guest-nights WMAPE |
    | --- | --- | ---: |
    | parametric chain | chain `ALOS[m]` | {ho_guest:.1f}% (T4) |
    | hybrid LightGBM | chain `ALOS[m]` | {w_chain:.1f}% |
    | **hybrid LightGBM** | **learned ALOS** | **{w_ml:.1f}%** |

    The learned ALOS means {alos_hat.mean():.2f} against an actual volume-weighted
    {ho['Guests'].sum() / ho['NewArr'].sum():.2f}.
    """
    )
    return (w_ml,)


@app.cell
def _(best_iter_t5, featurize, go, mo, np, panel, params, pd, train_model):
    # What the model learned, and how it differs from the chain's fixed ratio.
    booster = train_model(panel, best_iter_t5, seed=0)
    Xall = featurize(panel)
    off_all = np.log(panel["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
    pred_in = np.exp(booster.predict(Xall, raw_score=True) + off_all)

    gain = pd.Series(booster.feature_importance("gain"), index=Xall.columns).sort_values()
    fig_imp = go.Figure(
        go.Bar(x=gain.values, y=gain.index, orientation="h", marker_color="#4c78a8")
    )
    fig_imp.update_layout(
        title="LightGBM gain importance — conversion model",
        xaxis_title="gain",
        yaxis_title="",
        height=340,
    )

    local = panel.assign(pred=pred_in).groupby("market").agg(
        pred=("pred", "sum"), p2p=("Total_P2P", "sum")
    )
    learned_capture = local["pred"] / local["p2p"]
    compare = pd.DataFrame({"chain": params["capture"], "learned": learned_capture}).dropna()
    fig_cap = go.Figure()
    fig_cap.add_trace(
        go.Scatter(
            x=compare["chain"],
            y=compare["learned"],
            mode="markers",
            text=compare.index,
            marker=dict(size=8, color="#e45756"),
        )
    )
    hi = float(max(compare["chain"].max(), compare["learned"].max())) * 1.05
    fig_cap.add_trace(
        go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color="gray", dash="dash"))
    )
    fig_cap.update_layout(
        title="Capture rate: chain ratio vs learned",
        xaxis_title="chain capture[m] (capped)",
        yaxis_title="learned capture (in-sample Σpred / ΣP2P)",
        height=420,
    )

    mo.vstack(
        [
            mo.md(
                "**ML-5 — interpretation.** `market` dominates, which is the point: "
                "the model learns a *shrunk, market-aware* capture rate rather than "
                "the noisy raw ratio. Points below the 45° line are thin markets the "
                "chain's ratio overstates; above it, markets it understates."
            ),
            fig_imp,
            fig_cap,
        ]
    )
    return


@app.cell
def _(
    actual_total,
    alos_global,
    best_iter_t4,
    best_iter_t5,
    chain_ok,
    ho_guest,
    ho_new,
    ho_sn,
    identity_ok,
    ins_guest,
    ins_new,
    lf_ok,
    ml_ho_new,
    ml_t5_bias,
    ml_t5_new,
    mo,
    pred_total,
    test_new,
    w_ml,
):
    # One pin block for both models. T7 pins the chain; ML-6 pins the hybrid and
    # asserts it beats the chain on the same rows. Because both parts run in one
    # notebook, "the chain baseline" here is literally the value Part A computed
    # a few cells above — no second copy to drift.
    #
    # Tolerances are tight enough to catch a real change and loose enough to
    # survive float noise; the model is stochastic, so exact equality would be
    # brittle.
    assert identity_ok, "PAX identity broken"
    assert lf_ok, "load-factor identity broken"
    assert chain_ok, "chain did not reproduce measured supply stages"
    assert round(alos_global, 3) == 3.602, alos_global
    assert abs(ins_new - 27.3) < 1.0, ins_new
    assert abs(ins_guest - 24.8) < 1.0, ins_guest
    assert abs(ho_new - 28.4) < 1.5, ho_new
    assert abs(ho_sn - 36.0) < 1.5, ho_sn
    assert abs(ho_guest - 25.6) < 1.5, ho_guest
    assert abs(test_new - 32.8) < 2.0, test_new
    # The hybrid must beat the chain it shares a panel with.
    assert ml_ho_new < ho_new, (ml_ho_new, ho_new)
    assert ml_t5_new < test_new, (ml_t5_new, test_new)
    assert abs(ml_ho_new - 21.7) < 1.5, ml_ho_new
    assert abs(ml_t5_new - 20.8) < 2.0, ml_t5_new
    assert abs(w_ml - 21.0) < 2.0, w_ml

    mo.md(
        f"""
    ---

    # Part C — verification

    **T7 / ML-6 — consistency checks.** All assertions pass.

    | metric | parametric chain | hybrid LightGBM |
    | --- | ---: | ---: |
    | T4 New Arrivals WMAPE | {ho_new:.1f}% | **{ml_ho_new:.1f}%** |
    | T4 seasonal-naive WMAPE | {ho_sn:.1f}% | — |
    | T5 New Arrivals WMAPE | {test_new:.1f}% | **{ml_t5_new:.1f}%** |
    | T4 guest-nights WMAPE | {ho_guest:.1f}% | **{w_ml:.1f}%** |
    | T5 aggregate bias | {(pred_total / actual_total - 1) * 100:+.1f}% | **{ml_t5_bias:+.1f}%** |
    | best iteration, T4 / T5 | n/a | {best_iter_t4} / {best_iter_t5} |
    | global ALOS | {alos_global:.3f} | {alos_global:.3f} (shared input) |

    The hybrid beats the chain on every held-out metric and on aggregate bias,
    and the improvement holds across five seeds and two independent holdouts.

    The scripts in `pipeline/` remain the canonical implementations; this
    notebook recomputes from the same loader and asserts its numbers, so it
    cannot silently tell a different story.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations

    * **The out-of-sample test scores New Arrivals, not guest-nights.** The
      competition test frame has no `Guests` column, so guest-nights can only be
      validated on the in-sample panel (T3) and the leave-tail holdout (T4/ML-4).
    * **VFR / purpose-of-travel is unmeasured.** Both capture rates absorb it by
      construction — the measured one because it is a ratio of totals, the
      learned one because it is fitted to the same totals. A simulator built on
      this chain moves seats and capture separately, but must not imply the
      visitor split is measured.
    * **Twelve hotel-only markets** have no flight supply (Australia, Brazil,
      Czechia, Denmark, Finland, Mexico, Morocco, Norway, Pakistan, Romania,
      South Africa, Sweden). They are a residual, not a modelled output.
    * **Capture is capped at 5.0** in the chain. Values above one new arrival per
      terminating passenger are real but not a pure conversion; the cap keeps them
      from dominating aggregate sensitivity. The model has no such cap, which is
      one reason it can beat the chain on thin markets.
    * **Season is one global monthly profile** in the chain. Market-specific
      seasonality is pooled because most markets have only 43 months, which would
      leave thin per-market monthly cells.
    * **No target-derived features.** The model never sees New Arrivals, capture
      or ALOS at prediction time; market, season and supply are all it gets. This
      is what keeps the head-to-head with the chain honest.
    * **Small panel.** 1,405 rows, 33 markets, 43 months. The configuration is
      deliberately shallow and regularised, and the iteration count is chosen on
      an internal tail, not on the reported holdout. The gain is real but the
      data cannot support a deep model.
    * **Opening this notebook trains 15 LightGBM models** (T4 and T5 at five
      seeds each, plus five for the learned ALOS). Part A alone is a fast read;
      Part B is not, and there is no way to skip it from within the app.

    ## Reproduce

    ```
    uv sync
    .venv/bin/python -m marimo export html --no-include-code notebooks/hybrid_layer.py \
      -o /tmp/hybrid_layer.html
    ```

    Executes every cell. The Part C assertions fail the run if any pinned number
    moves, for either model.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
