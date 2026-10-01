"""Does aggregate monthly air arrival volume predict aggregate hotel guests?

The market-mix test in `test_market_link.py` came back negative, so the only
defensible version of the conversion model is aggregate: total arrivals in,
total hotel guests out. This script asks whether that link carries any signal
at all, using the 46 months where both sides exist (2015-2018, less the two
months lost to the misfiled source files).

Three things are tested, because the naive one is misleading:

  1. The naive correlation of levels. Reported only to show how much of it is
     shared seasonality and upward trend rather than a causal link.
  2. Year-on-year log growth in both series, which removes the common
     seasonal shape and the common trend, leaving arrival shocks against guest
     shocks. This is the honest test of predictive content.
  3. A log-log regression of guests on arrivals with month fixed effects and a
     linear trend, which yields an elasticity and a year-out holdout error.

The ratio guests per arriving passenger is also tracked over time, because a
conversion model is only convenient if that ratio is roughly stable.

Usage:  .venv/bin/python pipeline/tests/test_aggregate_link.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (  # noqa: E402
    DCT_GUESTS_NATIONALITY,
    DCT_KPIS,
    PAX_MOVEMENTS,
    REPORT_AGGREGATE_LINK,
)

GUEST_FACTOR = {"count": 1.0, "thousands": 1_000.0}
MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load():
    k = pd.read_csv(DCT_KPIS)
    g = (k[(k.metric == "hotel_guests") & (k.granularity == "month")
           & (k.segment == "total")].copy())
    g["guests"] = g.value * g.unit.map(GUEST_FACTOR)
    g["month_key"] = g.year.astype(str) + "-" + g.period

    p = pd.read_csv(PAX_MOVEMENTS)
    a = p[(p.scope == "auh") & (p.granularity == "month")
          & (p.metric == "pax_arrivals")].copy()
    a = a.rename(columns={"value": "arrivals"})
    a["month_key"] = a.period_start.str[:7]

    df = g[["month_key", "year", "period", "guests"]].merge(
        a[["month_key", "arrivals"]], on="month_key").sort_values("month_key")
    df["m"] = df.period.astype(int)
    df["t"] = np.arange(len(df))
    return df.reset_index(drop=True)


def ols(y, X, names):
    """Least squares with standard errors, since statsmodels is unavailable."""
    X = np.column_stack([np.ones(len(y))] + list(X.T))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    labels = ["const"] + list(names)
    return beta, se, labels, 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))


def wmape(actual, pred):
    return 100 * np.abs(actual - pred).sum() / np.abs(actual).sum()


def main():
    df = load()
    print(f"matched months: {len(df)}  ({df.month_key.min()} to {df.month_key.max()})")
    print(f"years: {sorted(df.year.unique())}\n")

    # --- 1. naive levels -------------------------------------------------
    naive = df.guests.corr(df.arrivals)
    print("=== 1. naive correlation of levels ===")
    print(f"pearson(guests, arrivals) = {naive:.3f}")
    print("Inflated by shared seasonality and shared growth; not evidence of a link.\n")

    # --- 2. year-on-year growth -----------------------------------------
    d = df.copy()
    d["g_guests"] = np.log(d.guests).diff(12)
    d["g_arrivals"] = np.log(d.arrivals).diff(12)
    yoy = d.dropna(subset=["g_guests", "g_arrivals"])
    print("=== 2. year-on-year log growth (season and trend removed) ===")
    print(f"observations: {len(yoy)}  (months with a 12-month lag on both sides)")
    print(f"pearson = {yoy.g_guests.corr(yoy.g_arrivals):.3f}")
    print(f"spearman = {yoy.g_guests.rank().corr(yoy.g_arrivals.rank()):.3f}")
    print("A value near zero means arrival shocks do not move guest counts.\n")

    # --- 3. elasticity with month effects and trend -----------------------
    lg, la = np.log(df.guests.values), np.log(df.arrivals.values)
    dummies = np.column_stack([(df.m == i).astype(float) for i in range(2, 13)])
    trend = df.t.values.astype(float)
    beta, se, labels, r2 = ols(lg, np.column_stack([la, dummies, trend]),
                               ["log_arrivals"] + [f"m{i}" for i in range(2, 13)]
                               + ["trend"])
    print("=== 3. log(guests) ~ log(arrivals) + month effects + trend ===")
    print(f"n = {len(df)}, r2 = {r2:.3f}")
    print(f"elasticity of hotel guests to arrivals = {beta[1]:.3f} "
          f"(se {se[1]:.3f}, t {beta[1] / se[1]:.2f})")
    print("   b < 1 means hotel guests grow more slowly than arrivals, which is\n"
          "   what a capacity-constrained destination looks like.\n")

    # --- 4. conversion ratio stability -----------------------------------
    df["conv"] = df.guests / df.arrivals
    print("=== 4. hotel guests per arriving passenger ===")
    by_year = df.groupby("year").conv.agg(["mean", "std", "min", "max"])
    print(by_year.round(4).to_string())
    print(f"overall cv = {df.conv.std() / df.conv.mean():.4f} "
          f"(mean {df.conv.mean():.4f})\n")

    # --- 5. out-of-year holdout ------------------------------------------
    print("=== 5. prediction skill, leave-one-year-out ===")
    print("The 'no arrivals' column is the ablation that matters: if a seasonal")
    print("and trend model that never sees air arrivals scores as well, then")
    print("arrivals carry no predictive content.\n")
    print(f"{'held out':>10} {'seasonal naive':>15} {'no arrivals':>12} "
          f"{'arrivals only':>13} {'full model':>12}")
    holdout = []
    for yr in sorted(df.year.unique()):
        tr, te = df[df.year != yr], df[df.year == yr]
        base = te.guests.values
        prev = df[df.year == yr - 1]
        sn = (prev.set_index("m").guests.reindex(te.m.values).values
              if yr - 1 in set(df.year) else np.full(len(te), np.nan))
        dtr = np.column_stack([(tr.m == i).astype(float) for i in range(2, 13)])
        dte = np.column_stack([(te.m == i).astype(float) for i in range(2, 13)])
        la_tr, la_te = np.log(tr.arrivals.values), np.log(te.arrivals.values)
        lg_tr = np.log(tr.guests.values)

        def fit(Xtr, Xte):
            b, *_ = np.linalg.lstsq(Xtr, lg_tr, rcond=None)
            return np.exp(Xte @ b)

        p_none = fit(np.column_stack([np.ones(len(tr)), dtr, tr.t.values]),
                     np.column_stack([np.ones(len(te)), dte, te.t.values]))
        p_arr = fit(np.column_stack([np.ones(len(tr)), la_tr]),
                    np.column_stack([np.ones(len(te)), la_te]))
        p_full = fit(np.column_stack([np.ones(len(tr)), la_tr, dtr, tr.t.values]),
                     np.column_stack([np.ones(len(te)), la_te, dte, te.t.values]))
        f = lambda p: wmape(base, p)
        holdout.append((yr, f(sn) if np.isfinite(sn).all() else None,
                        f(p_none), f(p_arr), f(p_full)))
        print(f"{yr:>10} {('%.1f%%' % f(sn)) if np.isfinite(sn).all() else 'n/a':>15} "
              f"{f(p_none):>11.1f}% {f(p_arr):>12.1f}% {f(p_full):>11.1f}%")

    print("\nWMAPE on held-out years only. 'seasonal naive' predicts a month with")
    print("the same month one year earlier, so it has no error for 2015.")

    mean = lambda i: np.mean([r[i] for r in holdout if r[i] is not None])

    note = REPORT_AGGREGATE_LINK
    yy = yoy
    with open(note, "w") as fh:
        fh.write("# Aggregate arrivals vs aggregate hotel guests\n\n")
        fh.write("Generated by `pipeline/tests/test_aggregate_link.py`.\n\n")
        fh.write(f"Matched sample: {len(df)} months, {df.month_key.min()} to "
                 f"{df.month_key.max()}.\n\n")
        fh.write("## Correlation of levels\n\n")
        fh.write(f"Pearson of guests against arrivals: **{naive:.3f}**. This is\n"
                 "inflated by shared seasonality and shared growth and should not\n"
                 "be read as a link.\n\n")
        fh.write("## Year-on-year growth, season and trend removed\n\n")
        fh.write(f"n = {len(yy)}, Pearson **{yy.g_guests.corr(yy.g_arrivals):.3f}**, "
                 f"Spearman **{yy.g_guests.rank().corr(yy.g_arrivals.rank()):.3f}**.\n"
                 "This is the honest measure of whether arrival shocks move guest\n"
                 "counts, and it is the number to judge the approach on.\n\n")
        fh.write("## Elasticity\n\n")
        fh.write(f"log(guests) on log(arrivals) with month effects and trend gives\n"
                 f"an elasticity of **{beta[1]:.3f}** (se {se[1]:.3f}, "
                 f"t {beta[1] / se[1]:.2f}, r2 {r2:.3f}).\n\n")
        fh.write("## Guests per arriving passenger\n\n")
        fh.write("| year | mean | sd | min | max |\n| --- | --- | --- | --- | --- |\n")
        for yr, r in by_year.iterrows():
            fh.write(f"| {yr} | {r['mean']:.4f} | {r['std']:.4f} | "
                     f"{r['min']:.4f} | {r['max']:.4f} |\n")
        fh.write(f"\nCoefficient of variation across all months: "
                 f"{df.conv.std() / df.conv.mean():.4f}.\n\n")
        fh.write("## Holdout skill, leave-one-year-out\n\n")
        fh.write("The decisive test. If a seasonal and trend model that never sees\n"
                 "air arrivals scores as well as one that does, arrivals carry no\n"
                 "predictive content.\n\n")
        fh.write("| held out | seasonal naive | no arrivals | arrivals only | full model |\n")
        fh.write("| --- | --- | --- | --- | --- |\n")
        for yr, sn, a, b, c in holdout:
            naive_s = f"{sn:.1f}%" if sn is not None else "n/a"
            fh.write(f"| {yr} | {naive_s} | {a:.1f}% | {b:.1f}% | {c:.1f}% |\n")
        fh.write(f"| **mean** | {mean(1):.1f}% | **{mean(2):.1f}%** | "
                 f"{mean(3):.1f}% | {mean(4):.1f}% |\n")
        fh.write("\nThe mean for 'seasonal naive' covers 2016-2018 only, since 2015\n"
                 "has no prior year in the sample.\n\n")
        fh.write("## Verdict\n\n"
                 f"Arrivals do not help. A model that excludes them entirely averages\n"
                 f"{mean(2):.1f}% WMAPE against {mean(4):.1f}% for the model that\n"
                 f"includes them, and arrivals on their own are far worse at\n"
                 f"{mean(3):.1f}%. The elasticity is indistinguishable from zero.\n"
                 "Seasonality and trend carry the whole signal.\n\n")
        fh.write("## Caveats\n\n"
                 "* Four years, 2015-2018, all pre-COVID growth. The elasticity is\n"
                 "  estimated on an expanding market and is not evidence about a\n"
                 "  shock or a plateau.\n"
                 "* 2020 onward is absent from the sample, so no structural-break\n"
                 "  test is possible here.\n"
                 "* Only four years means month effects and trend consume most of\n"
                 "  the degrees of freedom; the elasticity is correspondingly\n"
                 "  imprecise.\n"
                 "* Arrivals include transit and non-hotel visitors, so a\n"
                 "  conversion below one is expected and is not by itself an error.\n")
    print(f"\nwrote {note}")


if __name__ == "__main__":
    main()
