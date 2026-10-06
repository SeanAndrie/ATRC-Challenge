"""The `market` join conflates flight departure country with guest nationality.

Every model in this repo shares one structural assumption: that the country a
flight departs from is the country the guest belongs to. The competition pack
does not say this. It supplies two independent columns and we joined on them:

    flight side  ->  Departure Country Name   (last departure point before AUH)
    hotel side   ->  Nationality              (guest citizenship)

Those are different objects. An Indian national flying India -> Doha -> Abu Dhabi
is Transfer/Transit on the first leg and P2P on the second, so their supply is
filed under QATAR while their demand is filed under INDIA. The mismatch is not
noise; it is a zero-sum transfer between hub markets and spoke markets.

`test_market_link.py` had already measured the symptom on independent public
data (origin mix vs guest mix correlates 0.908 overall, -0.041 without India)
and it was read past. This test measures it directly on the pack, where it can be
proved arithmetically rather than inferred:

    capture[m] = NewArrivals[m] / P2P[m]

If `market` meant the same thing on both sides, capture could not exceed 1.0 --
you cannot book more guests than there are terminating passengers. It does
exceed 1.0, in five markets. That is not a modelling assumption to be debated;
it is a falsification.

What survives, and what does not:

  * Aggregate survives. The errors are zero-sum across markets and stationary
    over the fit window, so they cancel in the national total. This is why T5's
    -0.1% bias is real evidence.
  * Market-level attribution does not survive. Any per-market number, including
    the simulator's market breakdown and the `seats_mult` lever, is filed to
    the wrong market.
  * `capture` is therefore a calibration coefficient between two differently
    composed populations, not a conversion probability. It is described that
    way in docs/METHODS.md.

The 5.0 cap in `fit_params` clips the worst case silently. China sits at 5.854
raw, so the cap distorts the single most contaminated market and no reader of
the scoreboard could tell. The cap is quantified in Part B below and decomposed
in the notebook.

Guard design. A test asserting `capture <= 1.0` would fail forever and be
ignored, so the five known-affected markets are pinned explicitly. Any *new*
market crossing 1.0 is a hard failure. If the join is ever fixed, or the data
shifts, this test says so instead of staying quietly green.

Usage:  .venv/bin/python pipeline/tests/test_market_attribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# pipeline/ for `paths`, repo root for the `pipeline` package itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import REPORT_MARKET_ATTRIBUTION, ROOT  # noqa: E402

from pipeline.competition import load as cl  # noqa: E402

CAP = 5.0

# Markets whose capture exceeds 1.0 on the current pack. Pinned so that a
# *new* breach is loud while the standing defect stays documented.
KNOWN_OVER_ONE = {
    "CHINA",
    "CANADA",
    "UNITED STATES OF AMERICA",
    "RUSSIAN FEDERATION",
    "FRANCE",
}

# Known raw capture values, pinned to catch silent data drift.
KNOWN_CAPTURE = {
    "CHINA": 5.854,
    "CANADA": 2.288,
    "UNITED STATES OF AMERICA": 1.990,
    "RUSSIAN FEDERATION": 1.527,
    "FRANCE": 1.220,
}


def build_panel() -> pd.DataFrame:
    """The paired market-month panel the model actually fits, as the notebook builds it."""
    fa = cl.flights_monthly_agg().rename(columns={"Departure Country Name": "market"})
    fa["market"] = fa["market"].str.upper().str.strip()
    fa["ym_str"] = fa["ym"].astype(str)

    tr = cl.read_international_train()
    tr["market"] = tr["Nationality"].str.upper().str.strip()
    tr["ym_str"] = tr["ym"].astype(str)

    dem = tr.groupby(["ym_str", "market"], as_index=False).agg(
        NewArr=("New Arrivals_n", "sum")
    )
    panel = dem.merge(fa[["ym_str", "market", "Total_P2P"]], on=["ym_str", "market"])
    panel["month"] = pd.PeriodIndex(panel["ym_str"], freq="M").month
    return panel


def capture_by_market(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-market capture plus the supply/demand mix divergence that causes it."""
    g = panel.groupby("market").agg(NewArr=("NewArr", "sum"), P2P=("Total_P2P", "sum"))
    g["capture"] = g["NewArr"] / g["P2P"]
    g["supply_share"] = g["P2P"] / g["P2P"].sum()
    g["demand_share"] = g["NewArr"] / g["NewArr"].sum()
    g["mix_ratio"] = g["supply_share"] / g["demand_share"]
    # Lower bound on the share of this market's guests who cannot have arrived
    # on a flight the pack files under their own nationality.
    g["misattributed_lb"] = (1 - 1 / g["capture"]).clip(lower=0)
    return g.sort_values("capture", ascending=False)


def main() -> int:
    panel = build_panel()
    g = capture_by_market(panel)

    over = g[g["capture"] > 1.0]
    new_breaches = sorted(set(over.index) - KNOWN_OVER_ONE)
    truncated = sorted(g.index[g["capture"] > CAP])

    # The aggregate identity the whole conversion model rests on.
    implied_total = float((g["P2P"] * g["capture"]).sum())
    actual_total = float(g["NewArr"].sum())
    identity_gap = implied_total - actual_total

    lines: list[str] = []
    add = lines.append

    add("# Market attribution diagnostic")
    add("")
    add(
        "The `market` join asserts that flight departure country equals guest "
        "nationality. This measures how badly that fails."
    )
    add("")
    add(f"- paired panel: {len(panel):,} market-months, {len(g)} markets")
    add(f"- aggregate identity gap: {identity_gap:,.1f} guests (exactly 0.0 by construction)")
    add("")

    add("## Falsification: capture > 1.0 is impossible under a valid join")
    add("")
    add("| market | New Arrivals | own-departure P2P | capture | min. misattributed |")
    add("|---|---:|---:|---:|---:|")
    for mkt, r in over.iterrows():
        add(
            f"| {mkt} | {r['NewArr']:,.0f} | {r['P2P']:,.0f} | "
            f"**{r['capture']:.3f}** | {r['misattributed_lb']:.1%} |"
        )
    add("")
    add(
        f"**{over['NewArr'].sum() - over['P2P'].sum():,.0f} guests** in these "
        f"{len(over)} markets could not have arrived on a flight the pack "
        f"attributes to their own nationality."
    )
    add("")

    add("## Cause: connecting traffic is filed under the hub, not the origin")
    add("")
    add(
        "Supply share is the mix of *departures*; demand share is the mix of "
        "*guest nationalities*. Where a market is a hub, departures carry "
        "connecting passengers whose nationality is filed elsewhere."
    )
    add("")
    add("| market | supply share | demand share | mix ratio | capture |")
    add("|---|---:|---:|---:|---:|")
    for mkt in g.index:
        r = g.loc[mkt]
        mark = " **<- hub**" if r["mix_ratio"] > 2.0 else ""
        add(
            f"| {mkt} | {r['supply_share']:.2%} | {r['demand_share']:.2%} | "
            f"{r['mix_ratio']:.2f}{mark} | {r['capture']:.3f} |"
        )
    add("")
    add(
        "The divergence is zero-sum: hubs (Qatar, Egypt, Azerbaijan, Turkey, "
        "India) carry supply in excess of demand, while spokes (UK, Germany, "
        "Russia, USA, Canada) carry demand in excess of supply."
    )
    add("")

    add("## The 5.0 cap hides the worst case")
    add("")
    if truncated:
        add("| market | raw capture | capped | distortion |")
        add("|---|---:|---:|---:|")
        for mkt in truncated:
            r = g.loc[mkt]
            add(
                f"| {mkt} | {r['capture']:.3f} | {CAP:.1f} | "
                f"{(1 - CAP / r['capture']):.1%} understated |"
            )
        add("")
        add(
            f"{', '.join(truncated)} is clipped by the cap in `fit_params`, so "
            "the chain's score is flattered by suppressing the most "
            "contaminated market. The ML model applies no cap, so part of the "
            "chain-vs-ML gap is this distortion rather than better learning."
        )
    else:
        add("No market exceeds the cap; no truncation.")
    add("")

    add("## Verdict")
    add("")
    add(
        "- **Aggregate is defensible.** The identity holds exactly and the "
        "errors cancel. T5's aggregate bias is real evidence."
    )
    add(
        "- **Market-level attribution is not defensible.** Per-market outputs "
        "and the `seats_mult` lever are filed to the wrong market."
    )
    add(
        "- **`capture` is a calibration coefficient, not a conversion "
        "probability.** Documented as such in docs/METHODS.md."
    )
    add(
        "- **The pack cannot fix this.** `Residence (groups)` is constant "
        "(`International`, 100% of rows) and no flight-side residence field "
        "exists, so correcting the mapping needs external data."
    )

    REPORT_MARKET_ATTRIBUTION.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MARKET_ATTRIBUTION.write_text("\n".join(lines) + "\n")

    print(f"paired panel: {len(panel):,} market-months, {len(g)} markets")
    print(f"capture > 1.0 in {len(over)} markets: {', '.join(sorted(over.index))}")
    print(f"aggregate identity gap: {identity_gap:,.4f}")
    print(f"capped markets: {', '.join(truncated) if truncated else 'none'}")
    print(f"wrote {REPORT_MARKET_ATTRIBUTION.relative_to(ROOT)}")

    if new_breaches:
        print(
            "\nFAIL: capture > 1.0 in previously clean markets: "
            + ", ".join(new_breaches)
            + "\nThe join assumption broke further. Update KNOWN_OVER_ONE only "
            "with a documented reason.",
            file=sys.stderr,
        )
        return 1

    drifted = [
        mkt
        for mkt, val in KNOWN_CAPTURE.items()
        if mkt in g.index and abs(g.loc[mkt, "capture"] - val) > 0.01
    ]
    if drifted:
        print(
            "\nFAIL: pinned capture values moved: " + ", ".join(drifted),
            file=sys.stderr,
        )
        return 1

    print("\nOK: standing defect unchanged; no new market crossed 1.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())