"""Does air seat share predict *international* hotel demand, as opposed to total?

The notebook asks whether the weak flights-to-hotel link is an artefact of
targeting the wrong population. UAE residents are roughly a quarter to a third
of Abu Dhabi hotel guests, and a returning resident is an arrival without being
demand, so restricting the target to international guests should in principle
sharpen the relationship. This script tests that.

The first result is structural and settles how the question can be asked at all.
UAE residents have no airborne counterpart: ADS-B's AE rows are domestic legs
from other UAE airports, which is not the population arriving from abroad. So
the "total guests" arm of the comparison cannot be built. Any market test on
this data is already forced onto the international proxy, which means retargeting
is not available as a clean A/B here. What can be tested is whether the residual
relationship in the international proxy is real, and what breaks it.

Three things are separated, because they are usually conflated:

  * Whether the observed correlation is larger than chance. With five to nine
    markets a point estimate of 0.9 is nearly worthless, so every correlation
    here carries a bootstrap confidence interval and a permutation p-value.
  * Whether it is a timing artefact. Airborne shares are 2025; hotel shares are
    2015-2023. If hotel market ranks move between those years, then part of any
    correlation is measuring drift rather than level agreement.
  * Whether visiting-friends-and-relatives travel is the mechanism. If it is,
    regional markets should convert airborne seats into hotel guests at a
    systematically lower rate than long-haul leisure markets, and restricting
    to long-haul markets should improve the fit.

Usage:  .venv/bin/python pipeline/tests/test_retarget_international.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import DCT_GUESTS_NATIONALITY, REPORT_RETARGET  # noqa: E402

from test_market_link import (           # noqa: E402
    airborne_shares, common_support, hotel_shares,
)


# Regional markets where visiting friends and relatives is a large share of
# trips. Long-haul is the complement among markets actually in common support.
REGIONAL = {"SA", "EG", "QA", "KW", "BH", "OM", "JO", "LB", "IQ"}

# Resamples for each bootstrap interval and permutation p-value. Five thousand is
# far more than the resolution these sample sizes support, and the interval
# endpoints move by less than a hundredth between runs at this count.
N_RESAMPLES = 5_000

# The 2024 annual report's KEY INTERNATIONAL MARKETS block: the only
# unambiguous non-resident hotel counts anywhere in the corpus, because it is
# printed as international guests rather than inferred by removing UAE.
INTL_2024 = {"IN": 358_000, "CN": 268_000, "GB": 219_000,
             "RU": 207_000, "SA": 177_000, "US": 116_000}


def spearman(a, b):
    """Rank correlation, computed without scipy, which is not in this venv."""
    return a.rank().corr(b.rank())


def _boot(x, y, stat, n=N_RESAMPLES, seed=11):
    """Percentile bootstrap over resampled market pairs.

    Resampling pairs rather than variables is the only honest option at this
    sample size, because the two shares are constrained to sum to one over the
    same markets and are not independent draws.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 3:
            continue
        out.append(stat(pd.Series(x[idx]), pd.Series(y[idx])))
    out = np.array([v for v in out if np.isfinite(v)])
    return np.percentile(out, [2.5, 97.5]) if len(out) else (np.nan, np.nan)


def _perm(x, y, stat, n=N_RESAMPLES, seed=11):
    """Two-sided permutation p-value by shuffling one side."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = abs(stat(pd.Series(x), pd.Series(y)))
    rng = np.random.default_rng(seed)
    cnt = sum(abs(stat(pd.Series(x), pd.Series(rng.permutation(y)))) >= obs
              for _ in range(n))
    return (cnt + 1) / (n + 1)


def report(name, x, y):
    """Correlation with an interval and a p-value, or an explicit refusal.

    Pairs where either side is missing are dropped first, so the reported n is
    the number of markets actually compared. Russia is the case that matters:
    it has 2024 international hotel guests but no 2015-2023 monthly nationality
    row, so it is absent from the common-support frame and present here.
    """
    x, y = pd.Series(x).astype(float), pd.Series(y).astype(float)
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 4 or x.nunique() < 3 or y.nunique() < 3:
        return {"name": name, "n": n, "pearson": np.nan, "spearman": np.nan,
                "p_lo": np.nan, "p_hi": np.nan, "p": np.nan,
                "s_lo": np.nan, "s_hi": np.nan, "sp": np.nan}
    pear = x.corr(y)
    spear = spearman(x, y)
    plo, phi = _boot(x, y, lambda a, b: a.corr(b))
    slo, shi = _boot(x, y, spearman)
    return {"name": name, "n": n, "pearson": pear, "spearman": spear,
            "p_lo": plo, "p_hi": phi, "p": _perm(x, y, lambda a, b: a.corr(b)),
            "s_lo": slo, "s_hi": shi, "sp": _perm(x, y, spearman)}


def show(r, rows):
    if r["n"] < 4:
        rows.append([r["name"], r["n"], "—", "—", "—", "—", "—", "—", "—"])
        return
    rows.append([r["name"], r["n"], f"{r['pearson']:.3f}",
                 f"[{r['p_lo']:.2f}, {r['p_hi']:.2f}]", f"{r['p']:.3f}",
                 f"{r['spearman']:.3f}", f"[{r['s_lo']:.2f}, {r['s_hi']:.2f}]",
                 f"{r['sp']:.3f}", ""])


def main():
    air = airborne_shares()
    hotel = hotel_shares()
    a, hh, common, _, _ = common_support(air, hotel)
    mix = hh.merge(a[["seats_share"]], left_on="iso", right_index=True)

    print("=== 1. why a clean A/B is not available on this data ===")
    uae = pd.read_csv(DCT_GUESTS_NATIONALITY)
    uae = uae[(uae.granularity == "month") &
              (uae.nationality.str.upper() == "UAE")]
    span = uae.groupby("year").share_pct.mean().round(1)
    print("UAE nationality as a share of all hotel guests, by year:")
    print("  " + "  ".join(f"{y}:{v}%" for y, v in span.items()))
    print("UAE has no airborne counterpart (ADS-B AE rows are domestic legs from")
    print("other UAE airports), so a 'total guests' market test cannot be built.")
    print("Every market test on this data is already on the international proxy.\n")

    # The proxy is "ex-UAE nationality", which is not the same population as
    # "non-resident". Abu Dhabi's resident population is mostly expatriate, so
    # residents of other nationalities sit inside the proxy. The 2024 report is
    # the only place the true resident share is printed, and it shows the size
    # of the error in the proxy the rest of the analysis relies on.
    INTL_TOTAL_2024, TOTAL_GUESTS_2024 = 3_000_000, 5_819_000
    resident_share = 100 * (TOTAL_GUESTS_2024 - INTL_TOTAL_2024) / TOTAL_GUESTS_2024
    intl_share = 100 - resident_share
    print("=== 1b. how good is the ex-UAE proxy for 'non-resident'? ===")
    print(f"2024 true resident share: {resident_share:.1f} per cent, so the")
    print(f"international segment is {intl_share:.1f} per cent of guests.")
    gap = []
    for year, uae_share in span.items():
        if year < 2021:
            continue
        ex_uae = 100 - uae_share
        gap.append((int(year), uae_share, ex_uae, ex_uae - intl_share))
        print(f"  {year}: UAE nationality {uae_share:.1f} per cent, so the ex-UAE "
              f"proxy is {ex_uae:.1f} per cent")
    if gap:
        worst = max(gap, key=lambda g: g[3])
        print(f"\n  The proxy overstates the international segment by up to "
              f"{worst[3]:.1f} points ({worst[0]}).")
    print("  Excluding UAE nationality therefore removes only part of the")
    print("  resident contamination. Residents of other nationalities remain in")
    print("  the proxy, and they are arrivals without being demand, which is the")
    print("  same defect retargeting was supposed to fix. The retargeting")
    print("  hypothesis is therefore half-tested at best, and the only")
    print("  uncontaminated international counts anywhere are 2024's six markets.\n")

    print("=== 2. baseline, international proxy vs 2025 airborne seat share ===")
    print("    95% bootstrap CI and permutation p, markets are the unit\n")
    rows = []
    for year, g in mix.groupby("year"):
        j = g.set_index("iso")
        if len(j) < 4:
            print(f"{year}  n={len(j):2d}  too few markets to report anything")
            continue
        show(report(f"DCT {year}", j.hotel_share, j.seats_share), rows)
    for r in rows:
        print(f"  {r[0]:<10} n={r[1]:2d}  pearson {r[2]:>6} {r[3]:>16} "
              f"p={r[4]:>6}   spearman {r[5]:>6} {r[6]:>16} p={r[7]:>6}")
    print("  CI columns are the 95% bootstrap interval on the point estimate.\n")

    print("=== 3. is it a timing artefact? hotel rank stability over time ===")
    p = mix.pivot_table(index="iso", columns="year", values="hotel_share")
    print(p.round(2).to_string())
    yrs = [c for c in p.columns if p[c].notna().sum() >= 4]
    print("\nSpearman between hotel-share ranks of the two adjacent usable years:")
    for i in range(len(yrs) - 1):
        s = spearman(p[yrs[i]].dropna(), p[yrs[i + 1]].dropna())
        print(f"  {yrs[i]} vs {yrs[i + 1]}: {s:.3f}")
    first, last = yrs[0], yrs[-1]
    both = p[[first, last]].dropna()
    if len(both) >= 4:
        r = report(f"ranks {first} vs {last}", both[first], both[last])
        show(r, rows)
        print(f"  {first} vs {last} on {r['n']} markets present in both: "
              f"spearman {r['spearman']:.3f} "
              f"[{r['s_lo']:.2f}, {r['s_hi']:.2f}] p={r['sp']:.3f}")
    print("  Hotel market mix is not stable across the decade, so pairing a 2025")
    print("  air cross-section with an earlier hotel year mixes level agreement")
    print("  with drift. No air-side change series exists to net that out.\n")

    print("=== 4. 2024, the only true international counts in the corpus ===")
    intl = pd.DataFrame({"hotel_guests": INTL_2024})
    tot = intl.hotel_guests.sum()
    intl["hotel_share"] = 100 * intl.hotel_guests / intl.hotel_guests.sum()
    # Full airborne frame, not the common-support subset: Russia carries 2024
    # international hotel guests but no 2015-2023 monthly nationality row, so it
    # is missing from the common support and would be dropped here in error.
    seats = air["seats"].reindex(intl.index)
    intl["legs"] = air["flights"].reindex(intl.index)
    intl["air_share"] = 100 * seats / seats.sum()
    intl["ratio"] = intl.hotel_share / intl.air_share
    intl["group"] = ["regional" if i in REGIONAL else "long-haul"
                     for i in intl.index]
    print(intl.sort_values("ratio").round(2)[
        ["hotel_guests", "hotel_share", "air_share", "ratio", "legs",
         "group"]].to_string())
    print(f"\n  six markets sum to {tot:,.0f} international guests")
    intl_results = []
    for label, sub in (("all six", intl),
                       ("excluding China", intl.drop(index="CN"))):
        r = report(label, sub.hotel_share, sub.air_share)
        intl_results.append(r)
        if r["n"] < 4:
            print(f"  {label:<17} n={r['n']}  too few markets to report")
            continue
        print(f"  {label:<17} n={r['n']}  pearson {r['pearson']:.3f} "
              f"[{r['p_lo']:.2f}, {r['p_hi']:.2f}] p={r['p']:.3f}   "
              f"spearman {r['spearman']:.3f} "
              f"[{r['s_lo']:.2f}, {r['s_hi']:.2f}] p={r['sp']:.3f}")
    print("  China is sampled on four ADS-B legs, so its air share is a sampling")
    print("  artefact and its ratio is not meaningful. Dropping it removes the")
    print("  only extreme point, which is why the correlation rises; with five")
    print("  markets that rise is one market's weight, not evidence.\n")

    print("=== 5. mechanism: does regional under-conversion explain the gap? ===")
    # Guests are summed over all usable years per market, then shared against
    # the common-support total. The denominator has to be the grand total, not
    # the per-market sum, which would make every share 100 by construction.
    market_guests = mix.groupby("iso").guests.sum()
    pooled = 100 * market_guests / market_guests.sum()
    cmp_ = pd.DataFrame({"hotel_share": pooled}).join(a[["seats_share"]]).dropna()
    cmp_["ratio"] = cmp_.hotel_share / cmp_.seats_share
    cmp_["group"] = ["regional" if i in REGIONAL else "long-haul" for i in cmp_.index]
    print(cmp_.sort_values("ratio").round(2).to_string())
    print()
    print(cmp_.groupby("group").ratio.agg(["count", "mean", "median"]).round(2).to_string())
    lh = cmp_[cmp_.group == "long-haul"]
    r_lh = report("long-haul only", lh.hotel_share, lh.seats_share)
    if r_lh["n"] < 4:
        print(f"\n  long-haul markets only: n={r_lh['n']}, too few to report")
    else:
        print(f"\n  long-haul markets only: n={r_lh['n']}  "
              f"pearson {r_lh['pearson']:.3f} [{r_lh['p_lo']:.2f}, "
              f"{r_lh['p_hi']:.2f}]  "
              f"spearman {r_lh['spearman']:.3f} [{r_lh['s_lo']:.2f}, "
              f"{r_lh['s_hi']:.2f}]")
    print("  Two of nine common-support markets are regional, so the group")
    print("  comparison is a direction, not a test.\n")

    note = REPORT_RETARGET
    with open(note, "w") as fh:
        fh.write("# Retargeting the model at international hotel demand\n\n"
                 "Generated by `pipeline/tests/test_retarget_international.py`.\n\n"
                 "## Question\n\n"
                 "The notebook's weak flights-to-hotel link is measured against\n"
                 "*total* hotel guests, of whom UAE residents are a large share and\n"
                 "who are arrivals but not demand. Does targeting international\n"
                 "guests instead recover a usable relationship?\n\n"
                 "## The comparison cannot be built as a clean A/B\n\n"
                 "UAE has no airborne counterpart in ADS-B, whose AE rows are\n"
                 "domestic legs from other UAE airports. A market test against total\n"
                 "guests would therefore have to invent an air share for the single\n"
                 "largest hotel market. Every market test already on this data is\n"
                 "implicitly on the international proxy, so there is no un-retargeted\n"
                 "arm to compare against.\n\n"
                 "UAE nationality as a share of all hotel guests:\n\n"
                 "| year | UAE share % |\n| --- | --- |\n")
        for y, v in span.items():
            fh.write(f"| {y} | {v} |\n")
        fh.write(f"\n## The proxy is not the international segment\n\n"
                 f"The 2024 report puts the true resident share at "
                 f"{resident_share:.1f} per cent, so international guests are\n"
                 f"{intl_share:.1f} per cent. Abu Dhabi's resident population is "
                 f"mostly expatriate, so residents of\nother nationalities sit "
                 f"inside the ex-UAE proxy. Excluding UAE nationality\nremoves "
                 f"only part of the contamination, and the residents left in are\n"
                 f"arrivals without being demand, which is the exact defect "
                 f"retargeting was meant to fix.\n\n"
                 "| year | ex-UAE proxy % | true international % | overstates by |\n"
                 "| --- | --- | --- | --- |\n")
        for year, uae_share, ex_uae, over in gap:
            fh.write(f"| {year} | {ex_uae:.1f} | {intl_share:.1f} | {over:.1f} |\n")
        fh.write("\nSo the retargeting hypothesis is half-tested at best, and the "
                 "only uncontaminated\ninternational counts in the corpus are "
                 "2024's six named markets.\n\n")
        fh.write("## Baseline with uncertainty\n\n"
                 "The existing international-proxy test, with a 95% bootstrap\n"
                 "interval over markets and a permutation p-value:\n\n"
                 "| year | markets | Pearson | 95% CI | p | Spearman | 95% CI | p |\n"
                 "| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            fh.write("| " + " | ".join(str(c) for c in r[:8]) + " |\n")
        fh.write("\nAt five to nine markets these intervals are wide by\n"
                 "construction. A point estimate near 0.9 is not evidence of much\n"
                 "when its interval spans most of the unit range. Note the split\n"
                 "between the two columns: Pearson is often high while Spearman is\n"
                 "not, which means a few large markets pull a straight line through\n"
                 "the levels while the ordering of markets carries almost no\n"
                 "information. Allocating guests to markets is an ordering problem,\n"
                 "so the ordering column is the one that matters, and it is null.\n\n"
                 "## Timing artefact\n\n"
                 "Airborne shares are 2025 and hotel shares are 2015-2023. Hotel\n"
                 "market mix is not stable across that span, so part of any\n"
                 "correlation is measuring drift rather than level agreement. There\n"
                 "is only one airborne cross-section, so the drift cannot be netted\n"
                 "out by differencing the two sides.\n\n"
                 "## 2024, true international counts\n\n"
                 "The only uncontaminated international figures in the corpus:\n\n"
                 "| market | guests | hotel share % | air seat share % | ratio | "
                 "legs | group |\n| --- | --- | --- | --- | --- | --- | --- |\n")
        for mkt, r in intl.sort_values("ratio").iterrows():
            fh.write(f"| {mkt} | {r.hotel_guests:,.0f} | {r.hotel_share:.2f} "
                     f"| {r.air_share:.2f} | {r.ratio:.2f} | {int(r.legs)} "
                     f"| {r.group} |\n")
        fh.write("\n| test | markets | Pearson | 95% CI | p | Spearman | 95% CI | p |\n"
                 "| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in intl_results:
            if r["n"] < 4:
                continue
            fh.write(f"| {r['name']} | {r['n']} | {r['pearson']:.3f} "
                     f"| [{r['p_lo']:.2f}, {r['p_hi']:.2f}] | {r['p']:.3f} "
                     f"| {r['spearman']:.3f} | [{r['s_lo']:.2f}, {r['s_hi']:.2f}] "
                     f"| {r['sp']:.3f} |\n")
        fh.write("\nChina is sampled on four ADS-B legs, so its air share is a\n"
                 "sampling artefact and its ratio of 21.92 is not meaningful.\n"
                 "Dropping it removes the only extreme point and the correlation\n"
                 "rises accordingly, which is one market carrying the result rather\n"
                 "than a relationship.\n\n"
                 "## Mechanism: is it visiting friends and relatives?\n\n"
                 "If VFR travel is the contaminant, regional markets should convert\n"
                 "airborne seats into hotel guests at a lower rate than long-haul\n"
                 "leisure markets. Pooled over all usable years, on common support:\n\n"
                 "| market | hotel share % | air seat share % | ratio | group |\n"
                 "| --- | --- | --- | --- | --- |\n")
        for mkt, r in cmp_.sort_values("ratio").iterrows():
            fh.write(f"| {mkt} | {r.hotel_share:.2f} | {r.seats_share:.2f} "
                     f"| {r.ratio:.2f} | {r.group} |\n")
        fh.write("\n| group | markets | mean ratio | median ratio |\n"
                 "| --- | --- | --- | --- |\n")
        for grp, r in cmp_.groupby("group").ratio.agg(["count", "mean", "median"]).iterrows():
            fh.write(f"| {grp} | {int(r['count'])} | {r['mean']:.2f} "
                     f"| {r['median']:.2f} |\n")
        if r_lh["n"] >= 4:
            fh.write(f"\nRestricting to the long-haul markets alone gives Pearson "
                     f"{r_lh['pearson']:.3f} "
                     f"[{r_lh['p_lo']:.2f}, {r_lh['p_hi']:.2f}] and Spearman "
                     f"{r_lh['spearman']:.3f} "
                     f"[{r_lh['s_lo']:.2f}, {r_lh['s_hi']:.2f}].\n")
        fh.write("\nThe direction is what the VFR story predicts, regional markets\n"
                 "converting less, but the test cannot carry it: two of the nine\n"
                 "common-support markets are regional, one of them is Egypt at a\n"
                 "ratio of 7.22, and the long-haul mean is itself dragged upward by\n"
                 "China's sampling artefact. This is a direction, not a finding.\n\n"
                 "## Conclusion\n\n"
                 "Retargeting at international guests does not recover a link that\n"
                 "can carry a forecast, and the reason is now specific rather than\n"
                 "general.\n\n"
                 "It is not primarily UAE residents. They are already absent from\n"
                 "every market test on this data, because they have no airborne\n"
                 "counterpart. Removing them was never the fix.\n\n"
                 "It is not mainly a timing artefact either, though that contributes:\n"
                 "hotel market ranks are unstable across the decade, so part of any\n"
                 "correlation is drift.\n\n"
                 "What remains is the visitor who arrives and does not need a room.\n"
                 "Retargeting removes residents, who are the *observable* half of the\n"
                 "non-demand population. The unobservable half is the regional\n"
                 "traveller staying with family, and that population is not measured\n"
                 "in any source available here, which is why the group comparison\n"
                 "above can only be a direction.\n\n"
                 "So the second link in the chain stays unmeasured, and the honest\n"
                 "position is unchanged: allocate hotel demand from DCT's own\n"
                 "history, and carry the flight lever as a bounded assumption rather\n"
                 "than a fitted coefficient.\n")
    print(f"wrote {note}")


if __name__ == "__main__":
    main()
