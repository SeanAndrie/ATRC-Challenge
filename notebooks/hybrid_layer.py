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


@app.cell(hide_code=True)
def readme_howto(mo):
    mo.md(r"""
    ## How to read this notebook

    Every step in the analysis has been given a short explanation. Each one opens
    with an **"In plain words"** sentence that says what the step does and why it
    exists without any jargon, then moves into **"The detail"** for the mechanics.

    The naming below doubles as the reading plan:

    | step | what it answers |
    | --- | --- |
    | **0.1–0.5** | where the data comes from and the panel is built (shared by both models) |
    | **A.1–A.6** | the parametric chain: does the arithmetic reproduce reality, and which levers move the number? |
    | **B.1–B.8** | the hybrid layer: what LightGBM learns, how it is validated, and where it beats the chain |
    | **C** | what the final pins protect |
    | **Glossary** | plain-words definitions of the terms used throughout |

    Two honest notes you should read first (both stay with you the whole way):

    * **The `market` key is not trustworthy at market level.** The flight table
      files a route by its *last departure point*; the hotel table files a guest by
      *nationality*. The panel joins the two as if they were the same object, which
      is false. *Aggregate* results survive this (the error cancels in the national
      total); *market-level* attribution does not. Everything here is that line,
      kept honest.
    * **Pins, not vibes.** The final block asserts the headline numbers. If a real
      change moves one, the notebook fails loudly instead of drifting quietly.
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

    from pipeline import hybrid
    from pipeline.competition import load as cl

    return cl, go, hybrid, np, pd


@app.cell(hide_code=True)
def step0_1_sources(mo):
    mo.md(r"""
    #### Step 0.1 — The three source tables

    **In plain words:** the whole analysis starts from three spreadsheets — one
    about flights into Abu Dhabi, one about the hotel guests who actually stayed,
    and one with the period we have to *predict* (the competition test window).

    **The detail.** `cl.read_flights()` loads `flight_data.xlsx` (117,608
    flight rows — every scheduled flight into the region), with columns like
    `Total Seats`, `Total PAX`, `Total P2P` (passenger-to-passenger), `Total
    Transfer`, `Total Transit`, and a market name. `cl.read_international_train()`
    loads the hotel train table (per `Nationality` × month: `Guests`,
    `New Arrivals`). `cl.read_international_test()` loads the same shape for the
    held-out test window (2025-08 → 2026-02) — and note it has **no `Guests`
    column**; that single fact later decides what we can and cannot score out of
    sample.

    Parsing quirks live inside `pipeline/competition/load.py`, not here: this
    notebook never reads the `xlsx` files directly, so there is exactly one place
    to fix a read problem.
    """)
    return


@app.cell
def _(cl):
    # The repaired loader owns parsing; this notebook never touches the xlsx
    # files directly.
    flights = cl.read_flights()
    intl_train = cl.read_international_train()
    intl_test = cl.read_international_test()
    return (flights,)


@app.cell(hide_code=True)
def step0_2_firstlook(mo):
    mo.md(r"""
    #### Step 0.2 — A first look at the flights table

    **In plain words:** a quick sanity scan of the raw flight data before any
    modelling — rows, columns, types, missing values, and basic summary
    statistics.

    **The detail.** Three trivial cells do the scanning: `head()` (first five
    rows), `info()` (dtypes and null counts), and `describe()` (means, min/max,
    percentiles). Nothing is learned here; the point is the opposite — catch a
    malformed table *before* it silently feeds a model. If `Total_PAX` ever
    exceeded `Total_Seats`, or a `market` came back as `NaN`, this is where a
    reader would notice.
    """)
    return


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


@app.cell(hide_code=True)
def step0_3_panel(mo):
    mo.md(r"""
    #### Step 0.3 — Building the panel: the one place both models meet

    **In plain words:** we glue the flights and the hotel guests into one table of
    market × month, then flag the trickiest assumption in the whole project — and
    it is an assumption we cannot fix from this pack.

    **The detail.** `hybrid.build_panels()` returns the canonical panel plus the
    categorical encodings both models share. The join key is `market`, and the
    reason this notebook exists is that **the two tables mean different things by
    "market"**:

    * flight side — `Departure Country Name`, the *last* departure point before
      Abu Dhabi;
    * hotel side — `Nationality`, the guest's citizenship.

    So this panel asserts "departure country == nationality". It is the model's
    one structural weakness (Section 4.5 in `docs/METHODS.md`). The row count is
    **1,405 market-months, 33 markets**, and a further **12 hotel-only markets**
    that have guests but no flights — kept as a residual, never invented a capture
    rate.
    """)
    return


@app.cell
def _(hybrid):
    # The single canonical build now lives in pipeline/hybrid.build_panels():
    # same loaders, same join, and the structural assumption is stated in that
    # docstring and in docs/METHODS.md section 4.5. This cell keeps the exact
    # return contract the rest of the notebook unpacks.
    fa, mkt_dtype, mon_dtype, panel, test_m, test_panel, train_m = (
        hybrid.build_panels()
    )
    return fa, panel, test_m, test_panel, train_m


@app.cell
def _(train_m):
    train_m.head()
    return


@app.cell(hide_code=True)
def step0_4_integrity(mo):
    mo.md(r"""
    #### Step 0.4 — T1 / ML-0: are the bookkeeping identities actually true?

    **In plain words:** before trusting any model we check two *arithmetic facts*
    that have to be exactly true in every one of 117,608 rows — this is
    accounting, not an assumption. If either fails, nothing downstream can be
    trusted.

    **The detail.** Two identities are pinned:

    1. **PAX = P2P + Transfer + Transit.** Every passenger is either a transfer, a
       transit, or point-to-point. `P2P` (the "real" arriving passengers) is
       *defined* as total PAX minus transfer/transit, so this identity must hold
       to within floating-point noise everywhere.
    2. **Load factor = PAX / Seats.** The load factor column must equal the ratio
       of its two components everywhere it is used.

    Both are checked to `1e-9` absolute error. The cell also computes the global
    **ALOS** (`Guests / New Arrivals` = **3.602**) and lists the 12 hotel-only
    markets so nobody later treats their missing flight supply as a download
    failure. It is called **T1** for the chain and **ML-0** for the model because
    one table, one check, both parts.
    """)
    return


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


@app.cell(hide_code=True)
def step0_5_params(mo):
    mo.md(r"""
    #### Step 0.5 — Fitting the chain's parameters

    **In plain words:** the chain's only real "model" is three ratios fitted from
    the train panel: the capture rate (guests ÷ inbound passengers), the ALOS
    (nights ÷ guests), and a season profile. This cell computes them.

    **The detail.** `fit_params(panel)` returns four things:

    * `capture[m]` — per-market **volume-weighted** ratio `Σ New Arrivals / Σ
      P2P`. Volume-weighted because summing first lets a big market outvote a thin
      one.
    * `alos[m]` — `Σ Guests / Σ New Arrivals`, same idea.
    * `season` — a 12-value monthly profile: actual arrivals in month *m* divided
      by what `P2P × capture` would imply, normalised to mean 1. It cleans out the
      calendar shape the seat supply alone does not explain.
    * `global_capture` / `global_alos` — the same ratios across all markets.

    Two things worth knowing now: **capture is capped at 5.0** (there is a whole
    step about it later — A.5), and a fancy "empirical-Bayes shrinker" was tried
    to tidy thin markets and **rejected**, because the real spread between markets
    (0.085 → 5.854) is bigger than what a Beta-binomial prior can even represent.
    That spread is a signal, not noise.
    """)
    return


@app.cell
def _(hybrid, panel):
    # The canonical fit_params (and its cap semantics) now lives in
    # pipeline/hybrid.py; the notebook and the simulator share one engine.
    # In plain words: capture[m] = guests per inbound passenger per market,
    # volume-weighted; ALOS[m] = nights per guest; season is a 12-month profile.
    fit_params = hybrid.fit_params
    params = fit_params(panel)
    return fit_params, params


@app.cell
def _(mo, params):
    capture = params["capture"]

    mo.md(
        f"""
        **Fitted parameters**. Global capture {params["global_capture"]:.3f} (new hotel arrivals per terminating passenger) and global ALOS {params["global_alos"]:.3f}.

        Capture spans {capture.min():.3f} to {capture.max():.3f} across the 33 flight markets (median {capture.median():.3f}) - the spread that makes a single blended ratio indefensible. ALOS ranges {params["alos"].min():.2f} to {params["alos"].max():.2f}.

        A Beta-binomial empirical-Bayes shrinker was tried first and rejected: the between-market dispersion of capture exceeds the variance of Beta prior can represent (the method-of-moments concentraion goes negative), which is the signature of real behavioural heterogeneity rather than small-sample noise. The measured rate is therefore kept, with the cap as the only guard. Part B reaches the same conclusion by a different route - the model's learned rate is compared against this one in ML-5. 
        """
    )
    return


@app.cell
def _(hybrid):
    # Canonical simulate / wmape live in pipeline/hybrid.py (this cell used to
    # define them; the extraction is byte-for-byte identical).
    simulate = hybrid.simulate
    wmape = hybrid.wmape
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


@app.cell(hide_code=True)
def stepA_1_t2(mo):
    mo.md(r"""
    #### Step A.1 — T2: why the first three stages are *exact*

    **In plain words:** the simulator's front half (seats → passengers → P2P) is
    pure arithmetic with no fitted numbers. We turn it on with all levers at 1 and
    confirm it reproduces the measured data exactly. That is what licenses
    everything after it.

    **The detail.** `simulate(panel, params)` walks the chain
    `Seats → PAX (× load factor) → P2P (× P2P share) → New Arrivals (× capture ×
    season) → Guest-nights (× ALOS)`. With unit levers the first three stages must
    reproduce the input columns to `1e-6`; `chain_ok` records that. The waterfall
    chart then shows each stage as a percentage of scheduled seats, so a reader
    sees how much of the original seats actually reach each step. The punchline:

    > If the first three stages are exact, only the fourth (conversion) is worth
    > learning. That is the entire argument for Part B.

    It is also the honest scope note: the last two stages carry the model's only
    estimates, and the waterfall is about the *train* panel — the honest test of
    those estimates comes in T4/T5.
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


@app.cell(hide_code=True)
def stepA_2_t3(mo):
    mo.md(r"""
    #### Step A.2 — T3: the optimistic number (in-sample fit)

    **In plain words:** score the chain on the same rows it was fitted on. It looks
    good, but "good on your own homework" proves little — the next two steps rep-
    resent the real tests.

    **The detail.** `wmape(actual, pred)` is the weighted mean absolute percentage
    error: `100 × Σ|actual − pred| / Σ|actual|`. Because it divides by the *sum*
    of actuals (not per-row), a big market cannot hide a small one's error. T3
    scores 27.3% (New Arrivals) and 24.8% (guest-nights) on the 1,405
    market-months **that the parameters were fitted on** — the ceiling, not the
    truth. Everything that follows exists so this number is never quoted as proof.
    """)
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


@app.cell(hide_code=True)
def stepA_3_t4(mo):
    mo.md(r"""
    #### Step A.3 — T4: the leave-tail holdout, and why the split date matters

    **In plain words:** we fake "the future": fit on everything up to January 2025,
    predict February–July 2025, and compare against what actually happened. The
    prediction months were never seen by the fitted parameters, so the score is
    honest.

    **The detail.** The panel is split into `fit_panel` (ym ≤ 2025-01) and
    `holdout` (2025-02..2025-07). The parameters are *refit on the truncated
    panel* — reusing the full-panel parameters here would leak the future into the
    "prediction". The chain is then scored against a **seasonal-naive baseline**:
    per market, last year's average for the same calendar month. That naive guess
    is the bar the chain has to clear (28.4% vs 36.0% — it clears it). Guest-nights
    are also scored (25.6%), but remember the test window has no `Guests` column,
    so T4/ML-4 is the only honest out-of-sample home for guest-nights.

    Why 2025-01? A 6-month tail leaves 43 months to fit, and it keeps a year's
    worth of seasonal shape inside the fit window. Part B reuses these exact
    splits so its model can be scored on identical rows.
    """)
    return


@app.cell
def _(panel):
    # T4 leave-tail holdout: fit through 2025-01, predict 2025-02..2025-07.
    # The holdout months are never seen by the fitted parameters (params are
    # refit on fit_panel only, in the next step) - that is what makes the score
    # an honest out-of-sample estimate rather than an in-sample one.
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


@app.cell(hide_code=True)
def stepA_4_t5(mo):
    mo.md(r"""
    #### Step A.4 — T5: the real competition test window

    **In plain words:** run the fitted chain forward into the window the
    competition actually wants predicted (2025-08 → 2026-02), using only the
    flight supply. Score New Arrivals against the truth.

    **The detail.** Fit **all** training data (the competition permits it — this is
    the deploy-style run), predict the test panel's New Arrivals, and score with
    `wmape` (32.8%). Crucially the cell also reports the **aggregate bias**: the
    predicted national total vs the actual total. The chain lands within ~1% of
    the aggregate — *that* is the property that survives the market-misfiling
    defect (the error cancels nationally). The market-month error (32.8%) is large
    and is **not** over-claimed: it is measured against a composition that is
    misfiled between markets. Two numbers, two different kinds of honesty.
    """)
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


@app.cell(hide_code=True)
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

    These numbers are aggregate-valid and market-invalid. The capture rates
    behind them conflate departure country with nationality, so the market-month
    WMAPE is measuring error against a composition that is misfiled. Read the
    gap below before quoting 32.8% as a fair chain baseline.
    """)
    return


@app.cell(hide_code=True)
def stepA_5_cap(mo):
    mo.md(r"""
    #### Step A.5 — Is the capture cap hiding a defect?

    **In plain words:** a few markets (CHINA = 5.854) have more guests than inbound
    passengers — impossible *if* the market key were true. We cap the worst at 5.0,
    then check the obvious worry head-on: **does the cap flatter the baseline?**
    The answer is the opposite — removing the cap makes the chain *worse*.

    **The detail.** The cap exists because capture is a contaminated coefficient,
    not a probability; an overly-inflated number in the biggest market is partly
    fitting *connecting traffic filed under the wrong market*. The honest test is
    to re-fit the chain **with and without** the cap on the same holdout (T4) and
    test window (T5) and compare. Uncapped: 36.10%; capped: 32.77% on T5. The cap
    clips a coefficient that extrapolates badly, so it *improves* out-of-sample
    accuracy and the **capped chain is the stronger, fairer baseline**. The
    headline gap is therefore measured against the best version of the chain, not
    a deflated one — this cell exists so nobody can say the baseline was easy to
    beat.
    """)
    return


@app.cell
def _(
    fit_panel,
    fit_params,
    holdout,
    mo,
    panel,
    pd,
    simulate,
    test_panel,
    wmape,
):
    # Cap decomposition. The chain's 5.0 capture clip is load-bearing, and the
    # direction is the opposite of what the misattribution defect alone would
    # suggest: CHINA's raw capture of 5.854 is partly fitting connecting traffic
    # filed under China rather than Chinese guests, so extrapolating it forward
    # hurts. Clipping shrinks that contaminated coefficient and IMPROVES the
    # baseline out of sample. The capped chain is therefore the stronger, fairer
    # baseline, and this cell exists so the hybrid's headline gap is measured
    # against the best version of the chain rather than a deflated one.

    raw = panel.groupby("market").agg(New=("NewArr", "sum"), P2P=("Total_P2P", "sum"))
    raw_capture = raw["New"] / raw["P2P"]
    clipped = raw_capture[raw_capture > 5.0]

    cap_rows = []
    for label, kwargs in (("capped at 5.0", {}), ("uncapped", {"cap": None})):
        # T4: same fit window as the reported holdout, so only the cap differs.
        t4 = wmape(
            holdout["NewArr"],
            simulate(holdout, fit_params(fit_panel, **kwargs))["NewArrivalHat"],
        )
        # T5: fit on all training data, predict the competition test window.
        t5 = wmape(
            test_panel["NewArr"],
            simulate(test_panel, fit_params(panel, **kwargs))["NewArrivalHat"],
        )
        cap_rows.append({"variant": label, "T4 WMAPE": t4, "T5 WMAPE": t5})

    cap_tbl = pd.DataFrame(cap_rows).set_index("variant")
    capped_t5 = cap_tbl.loc["capped at 5.0", "T5 WMAPE"]
    uncapped_t5 = cap_tbl.loc["uncapped", "T5 WMAPE"]
    cap_delta = capped_t5 - uncapped_t5

    mo.md(f"""
    **Cap decomposition — the cap makes the baseline stronger, not weaker**

    | capture clip | markets clipped | T4 WMAPE | T5 WMAPE |
    | --- | ---: | ---: | ---: |
    | capped at 5.0 | {len(clipped)} | {cap_tbl.loc['capped at 5.0', 'T4 WMAPE']:.2f}% | {capped_t5:.2f}% |
    | uncapped | 0 | {cap_tbl.loc['uncapped', 'T4 WMAPE']:.2f}% | {uncapped_t5:.2f}% |

    Clipped markets: {', '.join(f'{m} ({v:.3f})' for m, v in clipped.items()) or 'none'}.

    Removing the cap **{'improves' if cap_delta > 0 else 'worsens'}** the chain by
    {abs(cap_delta):.2f} pp on T5 and {abs(cap_tbl.loc['capped at 5.0', 'T4 WMAPE'] - cap_tbl.loc['uncapped', 'T4 WMAPE']):.2f} pp on T4.

    Worth stating plainly because it runs against intuition: the clip is
    **load-bearing on the baseline's score and it favours the baseline**.
    China's raw 5.854 is inflated partly by connecting traffic filed under
    China rather than Chinese guests, so extrapolating it forward is harmful;
    clipping shrinks that contaminated coefficient and helps out of sample.

    So the capped chain is the **stronger, fairer** baseline, and it is the one
    the headline comparison uses. This was worth checking, because the obvious
    guess -- that the cap hides a defect and inflates the chain's margin of
    victory -- is the opposite of what the data shows. Part B carries the gap
    against both variants so it is measured against the best chain, not a
    deflated one.
    """)
    return cap_delta, cap_tbl


@app.cell(hide_code=True)
def stepA_6_t6(mo):
    mo.md(r"""
    #### Step A.6 — T6: sensitivity — which lever can move the number most?

    **In plain words:** briefly, if you could shake each of the six levers, which
    shake would shake guest-nights most? The answer reflects the *room each lever
    is allowed to move*, not what the data says actually moves.

    **The detail.** Global sensitivity uses **Sobol** variance decomposition
    (SALib). `total_nights(x)` is the six-multiplier chain summed over the panel;
    SALib samples ~2,048 random 6-number combinations inside **per-lever ranges**
    (seats 0.8–1.2, capture 0.7–1.3, everything else 0.9–1.1); the output
    variance is then decomposed into each lever's share (S1, and ST with
    interactions). One-at-a-time +10% elasticities give the signed direction
    (all exactly 1.0, by construction).

    Two honest caveats, because this is a frequently-read table:

    * The ranges are **a-priori choices**, not measured spreads. Because the
      chain is a pure product, the ST ranking mostly mirrors the widths *we*
      chose (capture widest, hence "top"). It answers "which lever, given the room
      we allow it", not "which lever really moved things in history".
    * An empirical version (ranges derived from the data's own wobble) is a
      planned follow-up experiment; it will be reported separately and honestly,
      not silently swapped into this cell.
    """)
    return


@app.cell
def _(np, panel, params, pd):
    # T6 global sensitivity (SALib Sobol) over the six floating levers, plus
    # one-at-a-time elasticities for the signed direction.
    #
    # Honest caveat, in plain words: the bounds below are a-priori choices, not
    # measured spreads. Because the chain is a pure product, the ST ranking mostly
    # mirrors the widths we chose. An empirical-spread version is a planned,
    # separately-reported follow-up experiment.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The event calendar

    The brief lists *"event calendars and public holidays in source markets"* as a
    factor worth exploiting, and says supporting event data would be supplied. It
    was not: the eight-page data dictionary never mentions events, and the five
    provided files are all flight and hotel tables. Rather than drop the
    requirement silently, `pipeline/calendar_events.py` reconstructs the
    destination-side half of it — the UAE-wide events that hit every source
    market at once.

    Islamic dates are derived from an inline tabular Hijri conversion, then pinned
    to hand-checked Gregorian windows by
    `pipeline/tests/test_calendar_events.py`. UAE holidays follow moon sighting, so
    declared dates can differ from a computed calendar by a day or two; the pins
    are two-to-five days wide for that reason, and every feature counts days
    across a multi-day window, so a one-day shift cannot move them.

    Measured against the same window one year either side, so the baseline carries
    the month-to-month level and only the local shape is compared:

    | event (±3d window) | uplift in national new arrivals |
    | --- | --- |
    | Eid al-Adha | **+18.4%** |
    | Eid al-Fitr | **+11.2%** |
    | UAE National Day | +3.8% |
    | New Year | +1.2% |
    | Ramadan, mid-month | **−3.3%** (−12.4% at ±7d) |

    Ramadan is negative, and that is the finding rather than an error: resident
    hotel demand falls while outbound travel rises, and this target is inbound
    arrivals to Abu Dhabi.

    **Source-market public holidays are not covered.** That is the other half of
    the brief's wording and it needs a maintained holiday table for at least India,
    Egypt, Saudi Arabia, Qatar, Turkey and the UK — a sourcing exercise this does
    not fake. ML-7 therefore measures what the UAE-wide subset is worth, and the
    gap stays on the record.
    """)
    return


@app.cell
def _(mo):
    from pipeline.calendar_events import (
        COVERED_FROM,
        COVERED_TO,
        FEATURE_COLUMNS,
        check_windows,
        monthly_features,
    )

    # Re-derive and print the calendar rather than trusting it silently: if the
    # conversion ever breaks, this cell raises before a single model is fitted.
    event_dates = check_windows().assign(
        gregorian=lambda d: d["gregorian"].astype(str)
    )
    event_dates["declared_window"] = (
        event_dates["hijri"] + "  →  " + event_dates["window"]
    )
    mo.md(
        "#### Derived event dates\n\n"
        + event_dates[
            ["hijri", "gregorian", "declared_window", "days_to_window_start"]
        ].to_markdown(index=False)
        + "\n\n`days_to_window_start` is how far the computed date sits inside its\n"
        "pinned window. It is 1–2 days for most events, which is ordinary moon-\n"
        "sighting variation; 2025's Ramadan is the one year that spread further\n"
        "across the region, and its window is deliberately wider for that reason."
    )
    return


@app.cell(hide_code=True)
def stepB_1_features(mo):
    mo.md(r"""
    #### Step B.1 — The features: what the model is allowed to see

    **In plain words:** the model gets only *supply-side* facts — the market, the
    month, seat volume, load factor, P2P share, and how many routes/airlines serve
    the market. It never gets told the answer, or any ratio derived from the
    answer.

    **The detail.** `hybrid.featurize(panel)` builds the design matrix, and
    `hybrid.cats` carries the fixed `market`/`month` categorical encodings. The
    rule "no target-derived features" is the guarantee that keeps the head-to-head
    with the chain honest: at prediction time the model has never seen New
    Arrivals, capture or ALOS. It must *predict* the conversion from supply alone.
    The in-function asserts pin the feature counts (13 and 9) so a future edit
    cannot silently drop or add a feature.
    """)
    return


@app.cell
def _(hybrid):
    # The canonical featurize (and its in-function 13/9 feature-count asserts)
    # lives in pipeline/hybrid.py; the categorical encodings it needs are set by
    # build_panels().
    cats = hybrid.cats
    featurize = hybrid.featurize
    return (featurize,)


@app.cell(hide_code=True)
def stepB_2_machinery(mo):
    mo.md(r"""
    #### Step B.2 — The model machinery (booster, Poisson, the 5-seed ensemble)

    **In plain words:** this is what "let LightGBM learn the conversion" means, in
    one cell: a purpose-built regression that learns *guests per terminating
    passenger* (a "capture rate"), trained with settings chosen up-front — not by
    shopping for test score — and repeated over five random starting points that
    are then averaged.

    **The detail.** Four functions come from `pipeline/hybrid.py` (the notebook
    delegates so it always tests the packaged code the simulator ships):

    * `fit_poisson(core, val, ...)` — runs a **Poisson regressor** on `log(P2P)`
      as an exposure offset, so the model learns an intensity per passenger: a
      learned capture rate. Early stopping watches `val` and returns the chosen
      round count. A **booster** is the trained LightGBM object (a set of boosted
      trees); the ensemble is five of them, one per seed.
    * `predict_poisson(fit_df, pred_df, iters, ...)` — predicts, per market-month,
      in **5 seeds**, and returns both the averaged prediction and the per-seed
      list.
    * `train_model(...)` — single-seed refit (used by the interpretation cell).
    * `fit_alos(...)` — the learned length-of-stay model (ML-4).

    The a-priori settings (depth ≤ 15 leaves, `min_child_samples=20`, lr 0.03,
    L2 = 1.0, feature/bagging fractions 0.8, `deterministic=True`) were fixed
    before any holdout was observed. Deterministic + single-threaded + fixed seeds
    is what makes the whole notebook reproducible bit-for-bit.
    """)
    return


@app.cell
def _(hybrid):
    # The canonical fitters (base config, seeds, _ds, fit_poisson,
    # predict_poisson, train_model, fit_alos) were extracted verbatim to
    # pipeline/hybrid.py, so the notebook's ML-2..ML-7 numbers and Part C pins
    # now assert against the exactly-scored packaged code the simulator uses.
    fit_alos = hybrid.fit_alos
    fit_poisson = hybrid.fit_poisson
    predict_poisson = hybrid.predict_poisson
    train_model = hybrid.train_model
    return fit_alos, fit_poisson, predict_poisson, train_model


@app.cell(hide_code=True)
def stepB_3_nested(mo):
    mo.md(r"""
    #### Step B.3 — Nested validation: the "why does it feel like two fits" cell

    **In plain words:** every model needs two numbers: (1) *how many rounds* to
    train — tuned on a small internal slice; (2) *how good it is* — measured on
    rows it never trained on. These must use different months, or the second
    number lies. This cell sets up the split that keeps them separate.

    **The detail.** Within the T4 fit window we carve two internal tails:

    * `core` (≤ 2024-07) trains, `val` (2024-08..2025-01) chooses the round count
      via early stopping — this is the **internal tail**.
    * `val5 = holdout` (2025-02..2025-07) is that role for T5: pick rounds on the
      T4 holdout months, then refit the final ensemble on the whole panel.

    The rule in one line: the *reported* holdouts (T4's `ho`, T5's `test_panel`)
    are used for scoring only — never to choose hyper-parameters. Choosing rounds
    on `ho` would make ML-2's score a tuned number, not an estimate. This is the
    same discipline that says the LightGBM config itself was set a-priori.
    """)
    return


@app.cell
def _(fit_panel, holdout):
    # The internal validation tails inside each fit window. fit_panel/holdout are
    # Part A's T4 split, reused rather than restated; Part B only needs the two
    # nested tails for early stopping, never the holdout itself as a training set.
    #
    # Rule in one line: rounds are chosen on this internal tail; the reported
    # holdout (ho / test_panel) is used for scoring ONLY. Different rows, or the
    # score stops being an estimate.
    core = fit_panel[fit_panel["ym"] <= "2024-07"]
    val = fit_panel[fit_panel["ym"] > "2024-07"]

    # T5's iteration-selection window is the same pair one year later: train
    # through 2025-01, validate on the T4 holdout months.
    val5 = holdout
    return core, val, val5


@app.cell(hide_code=True)
def stepB_4_ml2(mo):
    mo.md(r"""
    #### Step B.4 — ML-2: the hybrid on the leave-tail holdout

    **In plain words:** the model's first honest exam, on the exact same six-month
    window the chain just sat. It posts ~21.7% vs the chain's 28.4% and the naive
    baseline's 36.0% — and the gain survives averaging over five seeds.

    **The detail.** Protocol in three moves: (1) `fit_poisson(core, val)` picks
    rounds on the internal tail; (2) `predict_poisson(fit_panel, ho, ...)` refits
    on the full fit window and predicts `ho` in all 5 seeds; (3) score the
    averaged prediction (`ml_ho_new`) and the min/max across seeds
    (`ml_ho_lo..ml_ho_hi`). The spread matters because a single seed can be lucky;
    a gain that only one seed shows is not a gain.
    """)
    return


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
    return best_iter_t4, ml_ho_hi, ml_ho_lo, ml_ho_new, pred_ho


@app.cell(hide_code=True)
def stepB_5_ml3(mo):
    mo.md(r"""
    #### Step B.5 — ML-3: the hybrid on the competition window

    **In plain words:** the headline result — the model against the chain on the
    actual competition test months. The hybrid posts ~20.8% vs the chain's 32.8%,
    tracking the aggregate and sharpening the market-month fit at the same time.

    **The detail.** Same protocol as ML-2, one year later: `fit_poisson(fit_panel,
    val5)` selects rounds on the T4-holdout months, then the final ensemble refits
    on the **whole panel** and predicts the test window. `ml_t5_bias` is the
    aggregate bias of the averaged ensemble. The cell then settles the "is the
    baseline too weak?" question with Part A's cap numbers: the chain is 32.77%
    at its strongest (capped) and 36.10% without the cap — the 12+ pp gap is
    measured against the *better* chain. Honest scope note repeated at the bottom:
    this gap is aggregate-valid but says nothing about market-level attribution.
    """)
    return


@app.cell
def _(
    cap_delta,
    cap_tbl,
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

    # The cap delta from Part A's decomposition, held against the hybrid's T5
    # score, which only exists here. Doing the ratio here rather than in Part A
    # keeps the two numbers from drifting apart.
    gap = test_new - ml_t5_new
    # Private: cap_tbl already carries these, and marimo requires names to be
    # unique across cells unless underscore-prefixed.
    _capped_t5 = cap_tbl.loc["capped at 5.0", "T5 WMAPE"]
    _uncapped_t5 = cap_tbl.loc["uncapped", "T5 WMAPE"]

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

    **Is the baseline as strong as it can be?** Part A's cap decomposition shows
    the chain's 5.0 capture clip is load-bearing *in the baseline's favour*: it
    clips CHINA (raw 5.854, inflated by connecting traffic filed under China) and
    is worth {abs(cap_delta):.2f} pp to the chain. So:

    | comparison | T5 WMAPE | gap vs hybrid |
    | --- | ---: | ---: |
    | chain, uncapped | {_uncapped_t5:.2f}% | {_uncapped_t5 - ml_t5_new:+.2f} pp |
    | chain, capped at 5.0 | {_capped_t5:.2f}% | {_capped_t5 - ml_t5_new:+.2f} pp |
    | **hybrid LightGBM** | **{ml_t5_new:.2f}%** | — |

    The hybrid beats the chain by **{gap:.2f} pp against the capped chain, which
    is the stronger of the two**. The uncapped chain would concede
    {_uncapped_t5 - ml_t5_new:.2f} pp, so the headline gap is the conservative
    one. The obvious worry — that the baseline's own guard is doing the
    model's work — was checked and does not hold.

    **Scope of this result.** The {gap:.2f} pp gap is a genuine 7-month
    forward test on identical rows, both scores are aggregate-valid, and the
    baseline is at its strongest. It is *not* evidence that the market-level
    attribution is correct — the `market` join conflates departure country with
    nationality, so both models are scored against a composition that is
    misfiled between markets. See the panel join and section 4.5.
    """
    )
    return best_iter_t5, ml_t5_bias, ml_t5_hi, ml_t5_lo, ml_t5_new


@app.cell(hide_code=True)
def stepB_6_ml4(mo):
    mo.md(r"""
    #### Step B.6 — ML-4: guest-nights (New Arrivals × ALOS)

    **In plain words:** guest-nights = arrivals × average length of stay. The chain
    uses one fixed ALOS per market; the model *learns* ALOS from the data on the
    same holdout, and the learned version is a little better when the two are
    measured honestly.

    **The detail.** Two ALOS candidates are scored against `ho.Guests`:

    * chain: `pred_ho × ALOS[m]` from the T4 refit (`params_holdout`) —
      `ho_guest` is Part A's 25.6%;
    * learned: `pred_ho × alos_hat`, where `alos_hat = fit_alos(fit_panel, ho)`
      predicts per-market-month ALOS with a 5-seed LightGBM regression on
      `log(Guests / NewArr)`.

    WMAPE is ~25.6% (chain) vs ~20.8% (learned). The header formula
    `guest-nights = New Arrivals × ALOS` is itself an assumption worth stating:
    it treats length of stay as market behaviour, not as a per-month accounting
    artefact. (And it can only be scored on T4 — the test window has no `Guests`.)
    """)
    return


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


@app.cell(hide_code=True)
def stepB_7_ml5(mo):
    mo.md(r"""
    #### Step B.7 — ML-5: what the model learned

    **In plain words:** two pictures: which features mattered most, and how the
    model's per-market capture compares to the chain's fixed ratio. Then a warning
    about how *not* to read the first picture.

    **The detail.** A single-seed model is refit on the whole panel. Picture 1:
    LightGBM *gain* importance — `market` dominates, which is expected (markets
    differ hugely). **The warning:** part of `market`'s importance is shrinking a
    *noisy* ratio, and part is shrinking a *contaminated* one — `market` is the
    join key that mislabels departure country as nationality. So the importances
    are not a clean "market awareness" reading.

    Picture 2: a scatter of chain `capture[m]` vs the model's in-sample learned
    capture (`Σ predicted P2P`-weighted). Below the 45° line: thin markets the
    chain's ratio overstates. **Five points sit beyond capture = 1.0** — the model
    is forced by the data to assign more guests than inbound passengers to those
    markets. That is the same market defect as CHINA's 5.854, now visible from the
    model side. The learned rate is a calibration coefficient, never claimed as a
    conversion probability.
    """)
    return


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
                "**ML-5 — interpretation.** `market` dominates importance. "
                "Read that carefully: part of it is *shrinking the noise* in a "
                "noisy raw ratio, and part of it is **shrinking a contaminated "
                "signal**. `market` is the join key that conflates flight "
                "departure country with guest nationality, so its gain importance "
                "partly measures the size of the misattribution term, not only "
                "the model's market awareness. The learned rate is a calibration "
                "coefficient, not a conversion probability. Points below the 45° "
                "line are thin markets the chain's ratio overstates; above it, "
                "markets it understates. The five points beyond capture 1.0 are "
                "impossible for a probability and are the clearest single picture "
                "of the defect."
            ),
            fig_imp,
            fig_cap,
        ]
    )
    return


@app.cell(hide_code=True)
def stepB_8_ml7(mo):
    mo.md(r"""
    #### Step B.8 — ML-7: what the event calendar is actually worth

    **In plain words:** does adding UAE-wide holiday features help? We test it once
    — fit with and without — and report the difference. The row spent on this
    matters: it is an *ablation* (one honest measurement), not a *tuning loop*
    (which would quietly invalidate the holdout scores).

    **The detail.** Both arms reuse the exact ML-2/ML-3 protocol
    (`events=False` drops the four event features), so the two columns differ only
    in features — same seed-0 round selection, same 5-seed refits. That is why it
    costs 12 extra fits (hence "30 models" in Limitations). The result is small,
    and the reasons are stated plainly: the features are nearly collinear with the
    `month` categorical (Eid al-Adha sits in June or July every year), so the only
    genuine signal is *Islamic drift* (Ramadan slid April → March → February
    across the window) — which is why any effect concentrates in the test window.

    "Recorded, not pinned": the assert just checks both arms actually ran — a pin
    that *demanded* a gain would turn the ablation into the tuning search the
    a-priori rule forbids.
    """)
    return


@app.cell
def _(
    best_iter_t4,
    best_iter_t5,
    core,
    fit_panel,
    fit_poisson,
    ho,
    ml_ho_new,
    ml_t5_new,
    mo,
    panel,
    pd,
    predict_poisson,
    test_panel,
    val,
    val5,
    wmape,
):
    # ML-7: is the UAE-wide event calendar worth anything?
    #
    # This is an ablation, not a tuning loop, and the distinction is the whole
    # point. The four features were fixed a priori from the brief's wording and
    # from the measured uplifts quoted in the calendar section above. They are
    # evaluated once here and reported once. If they had been chosen by trying
    # variants against T4 and T5, the reported holdout scores would no longer be
    # an estimate of anything, which is the failure this notebook's a-priori
    # rule exists to prevent.
    #
    # Both arms use the identical protocol as ML-2 and ML-3: a seed-0
    # early-stopping pass picks the round count, then a 5-seed ensemble is refit on
    # the full fit window. The comparison is therefore like for like — the
    # "without events" column is a full ensemble, not a single seed, so the
    # difference below is attributable to the features and not to seed luck.
    # (It costs 12 extra fits, which is why the count in Limitations is 30.)
    iter_no = fit_poisson(core, val, events=False)
    pred_no, _ = predict_poisson(fit_panel, ho, iter_no, events=False)
    abl_ho_no = wmape(ho["NewArr"], pred_no)

    iter_5_no = fit_poisson(fit_panel, val5, events=False)
    pred_5_no, _ = predict_poisson(panel, test_panel, iter_5_no, events=False)
    abl_t5_no = wmape(test_panel["NewArr"], pred_5_no)

    d_ho = abl_ho_no - ml_ho_new
    d_t5 = abl_t5_no - ml_t5_new

    mo.md(
        f"""
    **ML-7 — ablation: what the event calendar is worth.**

    | New Arrivals WMAPE | without events | with events | difference |
    | --- | ---: | ---: | ---: |
    | leave-tail holdout (T4) | {abl_ho_no:.1f}% | {ml_ho_new:.1f}% | {d_ho:+.1f} pp |
    | competition test window (T5) | {abl_t5_no:.1f}% | {ml_t5_new:.1f}% | {d_t5:+.1f} pp |

    Seed 0 picks the round count for both arms; both ensembles then refit on the full
    fit window at five seeds, so the two columns differ only in their features.
    Early stopping chose {iter_no} and {iter_5_no} rounds without events against
    {best_iter_t4} and {best_iter_t5} with them.

    Read this as a single number, not two. The four features are nearly
    collinear with the `month` categorical: Eid al-Adha fell in June or July every
    year of the fit window, which `month` already encodes. The genuine signal the
    calendar adds is *Islamic drift* — Ramadan moved April → March → February
    across the window, so a month that contained it and one that did not are the
    same `month` value to the model. That is why the effect, if there is one,
    concentrates in the test window.
    """
    )

    # Recorded rather than asserted: an honest ablation may show no gain, and a
    # pin that demands a gain would turn this into the search the a-priori rule
    # forbids. What *is* pinned is that the ablation actually ran both arms.
    assert abl_ho_no > 0 and abl_t5_no > 0, (abl_ho_no, abl_t5_no)
    abl_summary = pd.DataFrame(
        [
            {"window": "T4 leave-tail holdout", "without_events": abl_ho_no,
             "with_events": ml_ho_new, "difference_pp": d_ho,
             "iter_without": iter_no, "iter_with": best_iter_t4},
            {"window": "T5 competition window", "without_events": abl_t5_no,
             "with_events": ml_t5_new, "difference_pp": d_t5,
             "iter_without": iter_5_no, "iter_with": best_iter_t5},
        ]
    ).round(2)
    return (abl_summary,)


@app.cell(hide_code=True)
def stepC_pins(mo):
    mo.md(r"""
    #### Step C — what the pins protect

    **In plain words:** one block that fails loudly if any headline number moves.
    This is the notebook's contract with itself: pin the numbers in the write-up,
    so a real change is caught and a silent drift is not.

    **The detail.** T7 pins the chain, ML-6 pins the hybrid, and both assert the
    hybrid beats the chain on identical rows. Because everything runs in one
    notebook, "the chain baseline" is literally the value computed a few cells
    up (no second copy to drift). Tolerances are tight enough to catch a real
    change, loose enough to survive float noise — the model is stochastic, so
    exact equality would be brittle. An export of this notebook fails the run if
    any pin moves.
    """)
    return


@app.cell
def _(
    abl_summary,
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
    ml_ho_hi,
    ml_ho_lo,
    ml_ho_new,
    ml_t5_bias,
    ml_t5_hi,
    ml_t5_lo,
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
    | per-seed spread, T4 | n/a | {ml_ho_lo:.1f}–{ml_ho_hi:.1f}% |
    | per-seed spread, T5 | n/a | {ml_t5_lo:.1f}–{ml_t5_hi:.1f}% |
    | global ALOS | {alos_global:.3f} | {alos_global:.3f} (shared input) |

    The hybrid beats the chain on every held-out metric and on aggregate bias,
    and the improvement holds across five seeds and two independent holdouts.

    The event calendar's contribution, from ML-7, is the small part of that:

    {abl_summary.to_markdown(index=False)}

    Positive `difference_pp` means dropping the calendar features made the score
    worse, so the features helped. The four ML-6 pins above are unchanged by
    adding them: adding the calendar moved T4 by
    {ml_ho_new - 21.7:+.1f} pp and T5 by {ml_t5_new - 20.8:+.1f} pp against their
    previous values, both inside the existing tolerances. That is the honest
    summary — the calendar is real and directionally right, and small.

    The scripts in `pipeline/` remain the canonical implementations; this
    notebook recomputes from the same loader and asserts its numbers, so it
    cannot silently tell a different story.
    """
    )
    return


@app.cell(hide_code=True)
def glossary_plain(mo):
    mo.md(r"""
    ## Plain-words glossary

    Short definitions of every term you have met, in the order you met them.

    * **Market.** The key both tables are joined on. Flights call it the flight's
      last departure country; the hotel table calls it the guest's nationality.
      The mismatch is the project's central defect (aggregate survives; market
      level does not).
    * **P2P (point-to-point).** Passengers whose journey ends in Abu Dhabi —
      i.e. total passengers minus those transferring/transiting through.
    * **Load factor.** Seats actually filled: `Total PAX / Total Seats`.
    * **WMAPE.** Weighted mean absolute percentage error —
      `100 × Σ|actual − pred| / Σ|actual|`. Lower is better; it divides by the
      *total* so small markets can't hide error.
    * **Capture rate.** Guests per terminating passenger — the behavioural link
      between flights and hotels. A calibration coefficient, not a probability
      (it exceeds 1.0 in five markets, which is the defect talking).
    * **ALOS.** Average length of stay in days: `Guests / New Arrivals`.
    * **Seasonal-naive baseline.** Last year's average for the same calendar
      month, per market. The chain has to beat it; it does (T4).
    * **Holdout / leave-tail.** Months held back from fitting and used only for
      scoring. "Leave-tail" because they are the *newest* months — a true
      forward test, unlike a random split.
    * **Internal tail.** A small slice of the *fit* window used only to pick how
      many rounds to train. It must differ from the holdout, or the reported
      score stops being an estimate.
    * **Booster.** One trained LightGBM model (a set of boosted trees). Five of
      them — one per seed — form the ensemble that is averaged into a prediction.
    * **Poisson with an offset.** The model's objective. It predicts how many
      events (new arrivals) happen per unit of exposure (passengers), instead of
      modelling counts directly — exactly the "learned capture rate" idea.
    * **5-seed ensemble.** Train five models from different random starts and
      average their predictions. Guards against one lucky (or unlucky) seed.
    * **A-priori config.** Model settings picked up-front from general good
      practice (shallow, regularised) and *not* tuned on the evaluated holdouts.
      Tuning on the holdouts would make the reported scores over-optimistic.
    * **Ablation vs tuning.** An ablation removes a feature once and reports what
      happened. Tuning would search many variants and report the best — which
      silently re-used the test data. This notebook only ablated, on purpose.
    * **Pins.** `assert` statements against the headline numbers. They leave no
      room for the notebook to quietly tell a different story.
    """)
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
    * **`market` conflates flight departure country with guest nationality.** This is
      the model's structural weakness. The pack supplies `Departure Country Name`
      and `Nationality` as independent columns and the panel joins on them, but an
      India → Doha → Abu Dhabi itinerary files supply under Qatar and demand under
      India. Falsified by capture exceeding 1.0 in five markets (CHINA 5.854,
      CANADA 2.288, USA 1.990, RUSSIA 1.527, FRANCE 1.220): ~903,000 guests cannot
      have arrived the way the pack files them. **Aggregate stays valid** (the
      error is zero-sum across hub and spoke markets and cancels in the national
      total); **market-level attribution does not**, which includes the simulator's
      market breakdown and the `seats_mult` lever. Not fixable from this pack —
      `Residence (groups)` is constant. See `pipeline/tests/test_market_attribution.py`
      and METHODS §4.5.
    * **The chain's 5.0 capture cap is load-bearing and favours the baseline.** It
      clips CHINA (raw 5.854, inflated by connecting traffic filed under China),
      and removing it *worsens* the chain from 32.77% to 36.10% on T5. The
      capped chain is therefore the stronger baseline and the one the headline
      comparison uses. Both are reported so the gap is not inflated.
    * **Twelve hotel-only markets** have no flight supply (Australia, Brazil,
      Czechia, Denmark, Finland, Mexico, Morocco, Norway, Pakistan, Romania,
      South Africa, Sweden). They are a residual, not a modelled output.
    * *** **Season is one global monthly profile** in the chain. Market-specific
      seasonality is pooled because most markets have only 43 months, which would
      leave thin per-market monthly cells.
    * **No target-derived features.** The model never sees New Arrivals, capture
      or ALOS at prediction time; market, season and supply are all it gets. This
      is what keeps the head-to-head with the chain honest.
    * **Small panel.** 1,405 rows, 33 markets, 43 months. The configuration is
      deliberately shallow and regularised, and the iteration count is chosen on
      an internal tail, not on the reported holdout. The gain is real but the
      data cannot support a deep model.
    * **Opening this notebook trains 30 LightGBM models**: T4 and T5 at a seed-0
      early-stopping pass plus a five-seed refit each (12), the same again for
      both ML-7 arms (12), five for the learned ALOS, and one for ML-5. Part A
      alone is a fast read; Part B is not, and there is no way to skip it from
      within the app.
    * **The event calendar is UAE-wide only.** The brief asks for "event calendars
      and public holidays in source markets"; the pack shipped neither, so
      `pipeline/calendar_events.py` reconstructs the destination-side events that
      hit every market at once. Public holidays in India, Egypt, Saudi Arabia,
      Qatar, Turkey and the UK are not covered — that needs a maintained holiday
      table per market, and ML-7 measures only what the UAE-wide subset is worth.
    * **Event dates are computed, not declared.** UAE holidays follow moon
      sighting, so a computed Islamic date can lead the official one by a day or
      two. The features count days across multi-day windows so a one-day shift
      cannot move them, and `pipeline/tests/test_calendar_events.py` pins every
      date to a two-to-five-day window. Outside 2022-01 to 2026-02 the module
      raises rather than reporting zeros, so a longer horizon needs the pin table
      extended and the test re-run.
    * **The UAE changed its weekend from Fri–Sat to Sat–Sun in January 2025**,
      and gave National Day a four-day weekend in December 2025 — inside the
      competition test window and inside the training span, so the regime change
      straddles the split. The December 2025 extra days are encoded; the wider
      regime change is not a feature, because at monthly resolution weekend length
      is nearly constant. A model that has to extrapolate past February 2026
      should treat this as a known structural break.

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
