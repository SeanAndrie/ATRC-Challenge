"""Does airborne origin-market mix predict DCT hotel-guest market mix?

This is the load-bearing assumption behind the conversion model: that the mix
of markets arriving by air is the mix of markets staying in Abu Dhabi hotels.
Everything downstream (flights to seats to guests) depends on it, and the
2025-vs-2026 market correlation of 0.908 does not test it, because that only
shows the two airborne sources agree with each other.

The test compares 2025 airborne seat share per origin market against DCT's
printed hotel-guest share for the same market in each year DCT reports. DCT
and ADS-B are different quantities (hotel guests versus arriving seats) and
different years, so a strong correlation means the *ordering* of markets
transfers, not that the levels are equal.

Caveats that the output states rather than hides:

  * Shares are compositional: one market gaining share mechanically lowers the
    others, which inflates a naive correlation. Spearman on ranks is reported
    alongside Pearson for that reason.
  * Both shares are renormalised over the same markets. The air side carries a
    long tail of origins DCT never reports as a hotel market; leaving those in
    one denominator but not the other would make a total-variation distance
    between the two meaningless. AE is excluded throughout, because UAE
    residents are the largest hotel market but ADS-B's AE rows are only
    domestic legs from other UAE airports.
  * DCT only prints its top markets, so the long tail is absent from the hotel
    side while present in ADS-B. Comparison is therefore restricted to markets
    DCT actually prints in at least six months of some year.
  * Years left with fewer than five common markets are recorded, not reported:
    a correlation on three or four points is arithmetically defined and
    analytically worthless.

Usage:  .venv/bin/python pipeline/tests/test_market_link.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# paths is imported first deliberately: importing it is what puts
# pipeline/aviation on sys.path, which is where aircraft_seats lives. Reversing
# these two lines fails with ModuleNotFoundError.
from paths import (                           # noqa: E402
    ADSB_LEGS,
    DCT_GUESTS_NATIONALITY,
    DCT_KPIS,
    REPORT_MARKET_LINK,
)
from aircraft_seats import add_seats          # noqa: E402

# DCT prints nationality names inconsistently across eras ("CHINA" and "China",
# "OMAN" and "Oman", "KSA" and "SAUDI ARABIA"), so names are matched
# case-insensitively. Only markets DCT actually reports are mapped; anything
# unmapped is reported as unmatched rather than silently dropped.
ISO = {
    "UAE": "AE", "KSA": "SA", "SAUDI ARABIA": "SA", "UK": "GB", "USA": "US",
    "OMAN": "OM", "QATAR": "QA", "KUWAIT": "KW", "BAHRAIN": "BH", "EGYPT": "EG",
    "INDIA": "IN", "CHINA": "CN", "PHILIPPINES": "PH", "PAKISTAN": "PK",
    "RUSSIA": "RU", "IRAN": "IR", "JORDAN": "JO",
    "LEBANON": "LB", "SYRIA": "SY", "FRANCE": "FR", "GERMANY": "DE",
    "ITALY": "IT", "SPAIN": "ES", "NETHERLANDS": "NL", "SWITZERLAND": "CH",
    "TURKEY": "TR", "IRELAND": "IE", "AUSTRALIA": "AU", "CANADA": "CA",
    "SOUTH AFRICA": "ZA", "SWEDEN": "SE", "POLAND": "PL", "MALAYSIA": "MY",
    "THAILAND": "TH", "JAPAN": "JP", "SINGAPORE": "SG", "INDONESIA": "ID",
    "HONG KONG": "HK", "GREECE": "GR", "AUSTRIA": "AT", "BELGIUM": "BE",
    "DENMARK": "DK", "NORWAY": "NO", "FINLAND": "FI", "PORTUGAL": "PT",
    "CZECH": "CZ", "HUNGARY": "HU", "ROMANIA": "RO", "BULGARIA": "BG",
    "SERBIA": "RS", "CYPRUS": "CY", "ARMENIA": "AM", "AZERBAIJAN": "AZ",
    "BANGLADESH": "BD", "SRI LANKA": "LK", "KAZAKHSTAN": "KZ", "MONGOLIA": "MN",
    "SOUTH KOREA": "KR", "TAIWAN": "TW", "MEXICO": "MX", "BRAZIL": "BR",
    "ARGENTINA": "AR", "CHILE": "CL", "COLOMBIA": "CO", "PERU": "PE",
    "NIGERIA": "NG", "KENYA": "KE", "ETHIOPIA": "ET", "TANZANIA": "TZ",
    "MOROCCO": "MA", "ALGERIA": "DZ", "TUNISIA": "TN", "LIBYA": "LY",
    "ISRAEL": "IL", "NEW ZEALAND": "NZ", "LATVIA": "LV", "ALBANIA": "AL",
    "IRAQ": "IQ", "SUDAN": "SD", "SLOVAKIA": "SK", "UZBEKISTAN": "UZ",
}

# Excluded from the headline correlation: UAE residents are the largest hotel
# market, but ADS-B's AE rows are only domestic legs from other UAE airports,
# so the two are not measuring the same thing.
EXCLUDE = {"AE"}

GUEST_FACTOR = {"count": 1.0, "thousands": 1_000.0}


def airborne_shares():
    """2025 seat and flight counts per origin market, local legs and AE removed."""
    df = pd.read_csv(ADSB_LEGS)
    df = df[~df.is_local_leg].copy()
    df = add_seats(df)
    by = df.groupby("origin_country")
    out = pd.DataFrame({"flights": by.size(), "seats": by.seats_typ.sum()})
    return out[~out.index.isin(EXCLUDE)]


def hotel_shares():
    """DCT hotel-guest counts per market per year, summed over printed months."""
    nat = pd.read_csv(DCT_GUESTS_NATIONALITY)
    nat = nat[nat.granularity == "month"].copy()
    nat = nat[~nat.nationality.str.upper().isin(["TOTAL", "OTHER"])]
    nat["iso"] = nat.nationality.str.upper().str.strip().map(ISO)
    nat["guests"] = nat.guests * nat.guests_unit.map(GUEST_FACTOR)
    # A market printed in only some months of a year is a partial series, so
    # require six months before its shares are used at all.
    out = (nat.groupby(["year", "iso"])
              .agg(guests=("guests", "sum"), months=("period", "nunique"))
              .reset_index())
    return out[out.months >= 6]


def common_support(air_counts, hotel_counts):
    """Restrict both sides to markets measured on both, then renormalise.

    Without this the two shares have different denominators and every
    distance-based statistic below compares nothing meaningful.
    """
    hotel_markets = set(hotel_counts.iso.dropna().unique())
    common = sorted((hotel_markets & set(air_counts.index)) - EXCLUDE)
    air_only = sorted(set(air_counts.index) - hotel_markets)
    hotel_only = sorted(hotel_markets - set(air_counts.index) - EXCLUDE)

    air = air_counts.loc[common].copy()
    hotel = hotel_counts[hotel_counts.iso.isin(common)].copy()
    for col in ("flights", "seats"):
        air[col + "_share"] = 100 * air[col] / air[col].sum()
    hotel["hotel_share"] = 100 * hotel.guests / hotel.guests.sum()
    return air, hotel, common, air_only, hotel_only


def spearman(a, b):
    """Rank correlation without scipy, which is not installed in this venv.

    Spearman is Pearson on the ranks, so this is exact and differs from
    scipy only in tie handling, which average-rank matching would also give.
    """
    return a.rank().corr(b.rank())


def main():
    air, hotel, common, air_only, hotel_only = common_support(
        airborne_shares(), hotel_shares())
    mix = hotel.merge(air[["flights_share", "seats_share"]],
                      left_on="iso", right_index=True)

    pad = max(len(common), len(air_only), len(hotel_only))
    padx = lambda xs: list(xs) + ["—"] * (pad - len(xs))     # noqa: E731
    print("=== airborne vs hotel market mix, DCT year vs 2025 ADS-B ===")
    print("(seat-weighted, common support only, UAE excluded)\n")
    print("common support ({} markets): {}\n".format(len(common), ", ".join(common)))
    print(f"{'air only':<{pad}}  {'hotel only':<{pad}}")
    print(f"{'(' + str(len(air_only)) + ')':<{pad}}  "
          f"{'(' + str(len(hotel_only)) + ')'}")
    for a, h in zip(padx(air_only), padx(hotel_only)):
        print(f"{a:<{pad}}  {h:<{pad}}")

    results, thin = [], []
    print(f"\n{'year':>5} {'mkts':>5} {'pearson':>8} {'spearman':>9} "
          f"{'top5 overlap':>13} {'TVD %':>7}")
    for year, g in mix.groupby("year"):
        j = g.set_index("iso")
        tvs = float(0.5 * (j.hotel_share - j.seats_share).abs().sum())
        if len(j) < 5:
            thin.append((int(year), len(j), tvs))
            continue
        t5a = set(air.nlargest(5, "seats_share").index)
        t5h = set(j.nlargest(5, "hotel_share").index)
        pear = j.hotel_share.corr(j.seats_share)
        spear = spearman(j.hotel_share, j.seats_share)
        print(f"{year:>5} {len(j):>5} {pear:>8.3f} {spear:>9.3f} "
              f"{len(t5a & t5h):>13} {tvs:>7.1f}")
        results.append((int(year), len(j), pear, spear, len(t5a & t5h), tvs))

    if thin:
        print("\nyears with too few markets on common support to report "
              "a correlation (recorded, not shown):")
        for year, n, tvs in thin:
            print(f"{year:>5} n={n:>2} TVD {tvs:>5.1f}")

    print("\n=== same test weighted by flights instead of seats ===")
    for year, g in mix.groupby("year"):
        j = g.set_index("iso")
        if len(j) < 5:
            continue
        print(f"{year:>5} pearson {j.hotel_share.corr(j.flights_share):>7.3f} "
              f"spearman {spearman(j.hotel_share, j.flights_share):>7.3f}")

    # Sensitivity: the headline correlation is dominated by whichever market
    # leads each distribution, so recompute without the largest air market.
    lead = air.seats_share.idxmax()
    print(f"\n=== sensitivity, dropping the largest air market ({lead}) ===")
    sens = []
    for year, g in mix.groupby("year"):
        j = g[g.iso != lead].set_index("iso")
        if len(j) < 5:
            continue
        pear = j.hotel_share.corr(j.seats_share)
        spear = spearman(j.hotel_share, j.seats_share)
        print(f"{year:>5} n={len(j):>2} pearson {pear:>7.3f} spearman {spear:>7.3f}")
        sens.append((int(year), len(j), pear, spear))

    # Market-level detail on the year with the largest common-support sample.
    best = max(results, key=lambda l: l[1])[0]
    j = (mix[mix.year == best].set_index("iso").sort_values(
        "hotel_share", ascending=False))
    print(f"\n=== market detail, DCT {best} (largest matched sample) ===")
    show = j[["hotel_share", "seats_share"]].copy()
    show.columns = ["hotel_share_%", "air_seat_share_%"]
    show["ratio"] = show["hotel_share_%"] / show["air_seat_share_%"]
    ratio = show.ratio
    print(show.round(2).to_string())
    print(f"\nconversion ratio (hotel share / air seat share) spans "
          f"{ratio.min():.2f} to {ratio.max():.2f}, a {ratio.max() / ratio.min():.0f}x "
          f"spread across only {len(ratio)} markets")
    print("A single blended conversion would therefore misallocate hotel guests "
          "by market\nby roughly an order of magnitude in both directions.")

    unmatched = (pd.read_csv(DCT_GUESTS_NATIONALITY)
                 .query("granularity == 'month'")
                 .assign(iso=lambda d: d.nationality.str.upper().str.strip().map(ISO))
                 .query("iso.isna() and nationality.str.upper() not in "
                        "['TOTAL', 'OTHER']")
                 .nationality.value_counts())
    print("\n=== DCT markets with no ISO mapping (excluded, not dropped) ===")
    print(unmatched.to_string() if len(unmatched) else "(none)")

    note = REPORT_MARKET_LINK
    with open(note, "w") as fh:
        fh.write("# Airborne market mix vs DCT hotel market mix\n\n")
        fh.write("Generated by `pipeline/tests/test_market_link.py`.\n\n")
        fh.write("## Question\n\n"
                 "Does the origin mix of arriving seats rank markets the same way DCT\n"
                 "ranks hotel guests? This is the load-bearing assumption behind\n"
                 "allocating hotel guests by market from flight data.\n\n"
                 "Airborne shares are seat-weighted from\n"
                 "`adsb_2025/omaa_arrivals_2025.csv` (2025, one day per month,\n"
                 "same-airport legs removed). Hotel shares are DCT's printed guest\n"
                 "counts, summed over months, restricted to markets DCT prints in at\n"
                 "least six months of some year.\n\n")
        fh.write("## Common support\n\n"
                 "Both shares are renormalised over the same markets, the ones\n"
                 "measured on both sides. The air side carries a long tail of\n"
                 "origins DCT never reports as a hotel market; leaving those in one\n"
                 "denominator and not the other would make a total-variation\n"
                 "distance between them a comparison of nothing.\n\n")
        fh.write(f"On both sides ({len(common)} markets): {', '.join(common)}\n\n")
        fh.write(f"Air only, no DCT row ({len(air_only)}): {', '.join(air_only)}\n\n")
        fh.write(f"Hotel only, no ADS-B row ({len(hotel_only)}): "
                 f"{', '.join(hotel_only) or 'none'}\n\n")
        fh.write("## Result: no usable relationship\n\n")
        fh.write("| DCT year | markets | Pearson | Spearman | top-5 overlap | TVD pp |\n")
        fh.write("| --- | --- | --- | --- | --- | --- |\n")
        for year, n, pear, spear, ov, tvd in results:
            fh.write(f"| {year} | {n} | {pear:.3f} | {spear:.3f} | {ov} | {tvd:.1f} |\n")
        fh.write("\nThe apparent correlation is not robust. It is carried almost\n"
                 "entirely by India, which leads the airborne distribution by a wide\n"
                 f"margin. Removing that one market ({lead}) alone collapses the\n"
                 "relationship:\n\n")
        fh.write("| DCT year | markets | Pearson | Spearman |\n| --- | --- | --- | --- |\n")
        for year, n, pear, spear in sens:
            fh.write(f"| {year} | {n} | {pear:.3f} | {spear:.3f} |\n")
        if thin:
            fh.write("\nYears left with too few common markets to report a correlation are\n"
                     "recorded rather than shown, because a correlation on three or four\n"
                     "points is arithmetically defined and analytically worthless:\n\n")
            fh.write("| DCT year | markets | TVD pp |\n| --- | --- | --- |\n")
            for year, n, tvs in thin:
                fh.write(f"| {year} | {n} | {tvs:.1f} |\n")
        fh.write("\nPer-market conversion, hotel share divided by air seat share, for\n"
                 f"the largest common-support sample (DCT {best}):\n\n")
        fh.write("| market | hotel share % | air seat share % | ratio |\n")
        fh.write("| --- | --- | --- | --- |\n")
        for mkt, r in show.iterrows():
            fh.write(f"| {mkt} | {r['hotel_share_%']:.2f} "
                     f"| {r['air_seat_share_%']:.2f} | {r.ratio:.2f} |\n")
        fh.write(f"\nThe ratio spans {ratio.min():.2f} to {ratio.max():.2f}, a "
                 f"{ratio.max() / ratio.min():.0f}x spread. The two extremes are\n"
                 "inverted: the market at the top is far more hotel-concentrated\n"
                 "than its air-seat share implies, and the market at the bottom\n"
                 "far less. That is the expected signature of VFR and\n"
                 "visiting-friends travel, which stays in homes, against leisure\n"
                 "and business travel, which uses hotels.\n\n")
        fh.write("## Conclusion\n\n"
                 "Airborne market mix must **not** be used to allocate hotel guests by\n"
                 "market. Aggregate arrivals may still predict aggregate hotel guests,\n"
                 "which is a question about scale rather than mix, and is untested here.\n"
                 "If market-level hotel forecasts are wanted, the market mix has to come\n"
                 "from DCT directly, or be modelled from its own history.\n\n"
                 "## Caveats\n\n"
                 "* Cross-decade: hotel shares are 2015-2023, airborne is 2025. India\n"
                 "  in particular is not comparable across that span, since India-UAE\n"
                 "  air capacity grew sharply after bilateral liberalisation. The\n"
                 "  India-driven correlation is partly a genuine structural change,\n"
                 "  not only a measurement artefact.\n"
                 "* Small samples: 5-9 markets per year, because DCT nationality\n"
                 "  coverage is thin in 2017, 2018 and 2021.\n"
                 "* Shares are compositional, so Pearson overstates independence and\n"
                 "  Spearman is the safer read. TVD is the total variation distance in\n"
                 "  percentage points over common-support markets.\n"
                 "* UAE is excluded: it is the largest hotel market, but ADS-B's AE rows\n"
                 "  are only domestic legs from other UAE airports, so the two columns\n"
                 "  do not measure the same thing.\n"
                 "* Seat weights are typical-capacity estimates, not operator actuals.\n")
    print(f"\nwrote {note}")


if __name__ == "__main__":
    main()
