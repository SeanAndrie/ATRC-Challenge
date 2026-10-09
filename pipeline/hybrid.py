"""Canonical implementation of the parametric chain and the hybrid ML layer.

This is the single source of truth for the model functions that previously lived
inside ``notebooks/hybrid_layer.py`` cells: ``fit_params``, ``simulate``,
``wmape`` (Part A) and the LightGBM Poisson fitters ``featurize``,
``fit_poisson``, ``predict_poisson``, ``train_model``, ``fit_alos`` (Part B).
The notebook and ``pipeline/simulator.py`` import from here so there is exactly
one model in the repo, and the notebook's pinned Part C numbers assert against
this implementation.

Data assembly
    ``build_panels()`` reads the competition pack through ``load.py`` and returns
    the analysable frames. Call it once before ``featurize`` / the fitters: the
    categorical encodings it derives are cached on this module.

Lever semantics
    ``simulate`` applies the six chain levers exactly as Part A does. The
    simulator's supply levers act pre-model on the hybrid path (they change real
    features and the exposure offset); ``capture_mult``/``season_mult``/
    ``alos_mult`` are post-hoc multipliers on the learned output. See
    ``pipeline/simulator.py``.

Attribution caveat (kept verbatim with the panel join)
    The join asserts departure country == guest nationality; it is false. Market
    *levels* are not attributable; the aggregate and the within-market timing
    signal are. See docs/METHODS.md §4.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo-root bootstrap (same pattern as the standalone diagnostics): pipeline is a
# namespace package, so the repo root must be on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import paths  # noqa: E402  (side effect: adds pipeline/* subdirs)
from pipeline.calendar_events import FEATURE_COLUMNS, monthly_features  # noqa: E402
from pipeline.competition import load as cl  # noqa: E402

# Native categorical features for LightGBM.
cats = ["market", "month"]

# Categorical encodings shared by train and test, set by build_panels().
_MKT_DTYPE: pd.CategoricalDtype | None = None
_MON_DTYPE: pd.CategoricalDtype | None = None


def _require_dtypes() -> tuple[pd.CategoricalDtype, pd.CategoricalDtype]:
    if _MKT_DTYPE is None or _MON_DTYPE is None:
        raise RuntimeError(
            "pipeline.hybrid: build_panels() must run before featurize/the fitters "
            "so the categorical encodings exist."
        )
    return _MKT_DTYPE, _MON_DTYPE


def build_panels():
    """Load the pack and build the analysable frames.

    Returns ``(fa, mkt_dtype, mon_dtype, panel, test_m, test_panel, train_m)`` in
    the same order and shape the notebook's Part A used, so notebook cells can
    unpack them unchanged.

    THE STRUCTURAL ASSUMPTION, kept with the join it is made in.

    The panel join asserts that flight departure country equals guest
    nationality. It does not. ``fa.market`` is Departure Country Name -- the last
    departure point before AUH -- while ``train_m.market`` is Nationality -- the
    guest's citizenship. An Indian national flying India -> Doha -> Abu Dhabi is
    Transfer/Transit on the first leg and P2P on the second, so their supply
    lands on QATAR and their demand lands on INDIA. P2P is computed per flight
    row, so stripping transfers does not remove this: it relocates connecting
    traffic onto the hub as if it originated there.

    This is falsifiable, and it is false. capture = NewArr / P2P exceeds 1.0 in
    five markets (CHINA 5.854, CANADA 2.288, USA 1.990, RUSSIA 1.527, FRANCE
    1.220) -- more guests than terminating passengers, which is impossible under
    a valid join. About 903,000 guests cannot have arrived the way the pack
    files them. See pipeline/tests/test_market_attribution.py and
    docs/METHODS.md section 4.5.

    Consequences, and they differ sharply:

    * Aggregate is defensible. The error is zero-sum across hub and spoke
      markets and stationary over the fit window, so it cancels in the national
      total. T5's -0.1% aggregate bias is real evidence.
    * Market-level attribution is NOT defensible. Per-market outputs, the
      simulator's market breakdown, and the seats_mult lever are all filed to
      the wrong market.
    * ``capture`` is a calibration coefficient between two differently composed
      populations -- not a conversion probability.

    The pack cannot fix this: ``Residence (groups)`` is constant
    ("International", 100% of rows) and no flight-side residence field exists.
    """
    global _MKT_DTYPE, _MON_DTYPE

    flights = cl.read_flights()
    intl_train = cl.read_international_train()
    intl_test = cl.read_international_test()

    fa = cl.flights_monthly_agg().rename(columns={"Departure Country Name": "market"})
    fa["market"] = fa["market"].str.upper().str.strip()
    fa["ym_str"] = fa["ym"].astype(str)

    # Routes/airlines are carried for the model's supply features; the chain
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
    _MKT_DTYPE = pd.CategoricalDtype(sorted(panel["market"].unique()))
    _MON_DTYPE = pd.CategoricalDtype(list(range(1, 13)))

    return fa, _MKT_DTYPE, _MON_DTYPE, panel, test_m, test_panel, train_m


def fit_params(p, cap=5.0):
    """Fit the chain's parameters on a market x month panel.

    capture and ALOS are volume-weighted ratios (Σ New / Σ P2P, Σ Guests /
    Σ New), so a large market is not outvoted by a small one.

    `cap` bounds capture, default 5.0. The bound turns out to be load-bearing
    and *helpful* to the baseline: CHINA's raw capture is 5.854, and that
    inflated value is partly fitting connecting traffic filed under China rather
    than Chinese guests. Clipping it shrinks the contaminated coefficient and
    improves out-of-sample accuracy -- the uncapped chain scores 36.10% on T5
    against 32.77% capped. So the capped version is the stronger baseline and
    the fair comparison. Pass `cap=None` to reproduce the decomposition.

    capture is NOT a conversion probability. Because `market` conflates departure
    country with nationality (see the panel join), it exceeds 1.0 in five
    markets, which a probability cannot. It is a calibration coefficient between
    two differently composed populations.
    """
    g = p.groupby("market").agg(
        New=("NewArr", "sum"), P2P=("Total_P2P", "sum"), G=("Guests", "sum")
    )
    capture = g["New"] / g["P2P"]
    if cap is not None:
        capture = capture.clip(upper=cap)

    # Season is the P2P-weighted ratio of actual to capture-expected new arrivals
    # for each calendar month, normalised to mean 1. It captures within-market
    # monthly variation that seat supply alone does not.
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
    multipliers are 1.0, which reproduces the measured seat, PAX and P2P values
    exactly and produces the model's New Arrivals and Guest-nights.
    """
    out = p.copy()
    seats = out["Total_Seats"] * seats_mult
    lf = (out["Total_PAX"] / out["Total_Seats"]) * lf_mult
    out["PAX"] = seats * lf
    p2p_share = (out["Total_P2P"] / out["Total_PAX"]) * p2p_mult
    out["P2P"] = out["PAX"] * p2p_share

    # NOTE: `season_mult` is accepted for signature symmetry with the brief's six
    # levers but intentionally not applied here -- the notebook's simulate()
    # leaves it a no-op, and the seasonal override is wired in
    # pipeline/simulator.py so the canonical chain is untouched.
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


def _safe_ratio(num, den):
    """Element-wise ``num / den`` with 0 where the denominator is 0.

    The training panel never has a zero-seat or zero-PAX market-month, so this is
    numerically identical on every reported number. It exists for scenario
    frames, where a supply lever can zero a denominator; without it featurize
    would emit NaN features (0/0) that LightGBM would then treat as missing.
    """
    return (num / den).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def featurize(df, events=True):
    """Leak-free features: only supply, calendar and market, all known for the
    competition test window. No target-derived quantity is used, so a prediction
    can be made without ever seeing New Arrivals.

    `events=False` drops the four calendar features and is used only by ML-7, to
    measure what the event calendar is worth. Everything else -- the chain in
    Part A and ML-2 to ML-6 -- uses the default `events=True`.
    """

    mkt_dtype, mon_dtype = _require_dtypes()
    x = pd.DataFrame(index=df.index)
    x["market"] = df["market"].astype(mkt_dtype)
    x["month"] = df["month"].astype(mon_dtype)
    ym = pd.PeriodIndex(df["ym_str"], freq="M")
    x["t"] = ym.year * 12 + ym.month - (2022 * 12 + 1)
    x["log_seats"] = np.log1p(df["Total_Seats"])
    x["load_factor"] = _safe_ratio(df["Total_PAX"], df["Total_Seats"])
    x["p2p_share"] = _safe_ratio(df["Total_P2P"], df["Total_PAX"])
    x["transfer_share"] = _safe_ratio(
        df["Total_Transfer"] + df["Total_Transit"], df["Total_PAX"]
    )
    x["routes"] = df["routes"].astype(float)
    x["airlines"] = df["airlines"].astype(float)

    if events:
        # Daily windows already aggregated to monthly day counts, clipped to the
        # month. A 9-day Eid reads as 9, and as 5 + 4 when it straddles a
        # boundary, rather than collapsing to a single bit.
        ev = monthly_features(ym)
        for col in FEATURE_COLUMNS:
            x[col] = ev[col].to_numpy().astype(np.float32)

    # Feature counts are pinned so a silently dropped or duplicated column fails
    # loudly: 9 without the calendar, 13 with it.
    assert x.shape[1] == (13 if events else 9), x.columns.tolist()
    return x


# One a-priori configuration, not tuned on the reported holdouts: shallow trees,
# high min_child_samples, L2, row/feature subsampling, single thread, fixed
# seeds. Iteration count is chosen by early stopping on an internal 6-month tail
# of the fit window, then the ensemble is refit on the full fit window.
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


def _get_lgb():
    import lightgbm as lgb

    return lgb


def _ds(df, feats, off, ref=None):
    lgb = _get_lgb()
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


def fit_poisson(train_iter, valid, max_rounds=2000, events=True):
    """Early-stopping pass on the internal tail; returns best_iteration."""
    lgb = _get_lgb()
    off_tr = np.log(train_iter["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
    off_v = np.log(valid["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
    d = _ds(train_iter, featurize(train_iter, events), off_tr)
    dv = _ds(valid, featurize(valid, events), off_v, ref=d)
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


def predict_poisson(train_final, test, best_iter, events=True):
    """Multi-seed ensemble on the rate scale: each seed's raw score is
    exponentiated with the exposure offset, and the resulting per-passenger
    intensities are averaged. Averaging the raw scores instead would average
    log-intensities, which is a different (geometric-mean) estimator."""
    lgb = _get_lgb()
    Xtr = featurize(train_final, events)
    Xte = featurize(test, events)
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
    lgb = _get_lgb()
    X = featurize(df)
    off = np.log(df["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
    p = dict(base)
    p["seed"] = seed
    return lgb.train(p, _ds(df, X, off), num_boost_round=n_rounds)


def fit_alos(train, test):
    """Guest-nights are New Arrivals x length of stay. This learns log-ALOS from
    the same features; the alternative is the chain's ALOS[m]."""

    lgb = _get_lgb()
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


def fit_hybrid(df, best_iter, events=True):
    """Fit the 5-seed Poisson ensemble on a full training frame (no holdout).

    Returns a list of ``num_boost_round=best_iter`` boosters, one per seed, ready
    for ``predict_hybrid``. Used by the simulator and by scenario prediction; the
    notebook's ``predict_poisson`` remains the faithful experiment-path copy.
    """
    X = featurize(df, events)
    off = np.log(df["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
    out = []
    for s in seeds:
        p = dict(base)
        p["seed"] = s
        out.append(_get_lgb().train(p, _ds(df, X, off), num_boost_round=best_iter))
    return out


def predict_hybrid(boosters, frame, events=True):
    """Ensemble prediction for a scenario frame (exposure + features only)."""
    X = featurize(frame, events)
    off = np.log(frame["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
    return np.mean(
        [np.exp(b.predict(X, raw_score=True) + off) for b in boosters], axis=0
    )
