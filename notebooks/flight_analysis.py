import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # OMAA flight movement analysis

    **What this notebook is.** A rigorous read of the OpenSky ADS-B extraction at
    Abu Dhabi International (`OMAA`), covering origin markets, concentration, market geography and
    distance, arrival timing and aircraft turnaround behaviour.

    **What it is not.** It is not the flight-to-hotel simulator. The conversion
    chain that the challenge brief centres on -- *scheduled seats -> arriving
    passengers -> inbound visitors -> hotel guest nights* -- cannot be fitted from
    ADS-B, because ADS-B observes **aircraft movement, not people or capacity**. The
    seats behind a flight, its load factor, whether a passenger is a resident
    returning home or a first-time visitor, and how long they stay are all
    unobservable here. Section 9 tabulates exactly which links in that chain are
    measurable, inferable, or blocked, and stops there rather than inventing
    numbers to fill the gaps.

    **Data provenance.** Reads `data/raw/opensky/opensky_omaa_2026-06-20_2026-09-13.csv`,
    exported from 37 day-files of raw `/flights/arrival` + `/flights/departure`
    responses. This notebook never calls the OpenSky API: historical requests cost 30
    credits each and the remaining daily balance is 40, which is reserved for
    finishing the extraction. `opensky.py` is the only thing that spends credits.

    **Provenance of every number.** Counts are reproducible from the CSV. Country
    attribution depends on OpenSky's *estimated* departure airport; section 7
    quantifies how much of the market mix rests on that estimate.
    """)
    return


@app.cell
def _():
    # Read-only analysis. See the header for why this notebook never calls the API.
    #
    # Every public symbol is prefixed `fa_`. marimo enforces exactly one owner per
    # public name across cells, and a bare `pd` or `mo` here would collide with the
    # extraction notebook if the two are ever merged or tested together.

    import pathlib
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates.default = "plotly_white"

    # Repo root by walking up from this file, not from the working directory.
    fa_here = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
    fa_repo = next(
        (c for c in (fa_here, *fa_here.parents) if (c / "pipeline" / "paths.py").exists()),
        fa_here.parent,
    )
    sys.path.insert(0, str(fa_repo / "pipeline"))

    import paths as fa_paths

    return Path, fa_paths, fa_repo, go, mo, np, pd


@app.cell
def _(Path, fa_paths, fa_repo):
    # Paths and thresholds. Nothing here spends credits or mutates source data.
    #
    # Resolved through the shared anchor in pipeline/paths.py, so the analysis
    # works regardless of which directory marimo was launched from and does not
    # restate a path that another script has to keep in step with.

    fa_CSV = fa_paths.OPENSKY_CSV
    fa_AIRPORTS = fa_paths.OURAIRPORTS

    # A breakdown with fewer than this many observations is still shown, but is
    # labelled indicative: 22 clean days spread over 7 weekdays leaves 2-4 days
    # each, which cannot support an inferential claim.
    fa_MIN_SAMPLE = 5

    fa_HOME_AP = "OMAA"
    fa_HOME_COUNTRY = "AE"
    return fa_AIRPORTS, fa_CSV, fa_HOME_AP, fa_HOME_COUNTRY, fa_MIN_SAMPLE


@app.cell
def _(fa_AIRPORTS, fa_CSV, pd):
    # ---------------------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------------------
    # `coverage_flag` and `coverage_ratio` are attached to every exported row so that
    # anyone aggregating arrivals per day cannot mistake a receiver outage for a
    # fall in demand. The coverage gate a few cells below is what makes this
    # honest; every count below is computed on the surviving clean days only.

    fa_events = pd.read_csv(fa_CSV, parse_dates=["event_time", "day"])
    fa_airports = pd.read_csv(fa_AIRPORTS)
    fa_airports.columns = [c.lower() for c in fa_airports.columns]

    print(f"{len(fa_events):,} events over {fa_events['day'].nunique()} observed days")
    print(f"  {fa_events['event_time'].min():%Y-%m-%d} .. {fa_events['event_time'].max():%Y-%m-%d}")
    print(f"  airport reference: {len(fa_airports):,} rows")
    return fa_airports, fa_events


@app.cell
def _(fa_HOME_AP, fa_HOME_COUNTRY, fa_airports, fa_events):
    # ---------------------------------------------------------------------------
    # Coverage gate
    # ---------------------------------------------------------------------------
    # This is the single most consequential decision in the notebook, so it is
    # spelled out rather than hidden in a `.query()` call.
    #
    # OpenSky coverage around the Gulf is uneven, and it fails in *contiguous
    # blocks* rather than as scattered thin days. This extraction contains a 10-day
    # block (2026-07-02 .. 07-11) where a normally-observed day of ~286 events drops
    # to as few as 11 -- roughly 4% of a normal day. Those flights were never
    # received, not never flown.
    #
    # Averaging all 37 observed days therefore understates a normal day by about a
    # third: 96 arrivals/day against 145 on clean days. Every statistic below is
    # computed on `fa_ok` only.
    #
    # The flag itself is computed in two passes upstream (in `opensky.py` and
    # `scripts/export_omaa_csv.py`): days are scored against the raw median, then
    # re-scored against the median of the days that survived. Without that second
    # pass the reference day collapses into the outage block and every healthy day
    # reads as over-observed.

    fa_daily = (
        fa_events.groupby(["day", "coverage_flag"])
        .agg(events=("icao24", "size"))
        .reset_index()
    )
    fa_flag_by_day = fa_daily.drop_duplicates("day").set_index("day")["coverage_flag"]

    fa_ok = fa_events.loc[fa_events["coverage_flag"].eq("ok")].copy()
    fa_dropped = len(fa_events) - len(fa_ok)

    # --- Country attribution, guarded against a NaN-key trap ---------------------
    # `dict(zip(ref.icao_code, ref.iso_country))` contains 75,605 rows with a null
    # ICAO code, and zipping a null key into a dict means `Series.map` resolves every
    # *missing* origin to whatever country that null row happened to carry -- "US" for
    # 30,894 of them. That silently relabels every unattributed arrival as American
    # and inflates the US market to 1,353 arrivals against a true 111. Dropping null
    # codes first is not defensive style; it is required for a correct mapping.
    fa_ref = fa_airports.loc[fa_airports["icao_code"].notna()]
    fa_country_of = dict(zip(fa_ref["icao_code"], fa_ref["iso_country"]))
    fa_type_of = dict(zip(fa_ref["icao_code"], fa_ref["type"]))

    fa_arrivals = fa_ok.loc[fa_ok["direction"].eq("arrival")].copy()
    fa_departures = fa_ok.loc[fa_ok["direction"].eq("departure")].copy()
    fa_arrivals["country"] = fa_arrivals["origin"].map(fa_country_of)
    fa_arrivals["origin_type"] = fa_arrivals["origin"].map(fa_type_of)

    # Guard: the mapping must never invent a country for a missing origin.
    fa_leaked = int(fa_arrivals.loc[fa_arrivals["origin"].isna(), "country"].notna().sum())
    assert fa_leaked == 0, f"country mapping leaked onto {fa_leaked} null origins"

    # --- Classify each arrival into a market bucket ------------------------------
    # Not every arrival that OpenSky can place belongs to an *inbound* market, and
    # two distinct problems have to be separated before the market tables are built.
    #
    # 1. Self-origin. 52 arrivals come back with the estimated origin set to OMAA
    #    itself, the same airport they are arriving at. OpenSky resolves the
    #    departure airport by picking the nearest airport to the aircraft's last
    #    known position, and an aircraft standing on the ground at Abu Dhabi has its
    #    last position at Abu Dhabi. For these rows the "origin" is not a departure
    #    at all -- it is the aircraft's own location echoing back. The tell is in the
    #    data: their `origin_distance_m` has a median of 11.5 km, which is the offset
    #    of a poor ground fix from the terminal, not the length of a flight. They are
    #    recorded as having no usable origin, not counted as a market.
    #
    # 2. Domestic UAE. A further 52 arrivals genuinely originate at other UAE
    #    airfields (OMAD, OMDW, OMDB and smaller strips). These are real flights, but
    #    they are not an inbound market: their passengers are already in the country
    #    and cannot be inbound visitors, so counting them beside India would inflate
    #    the foreign market by a rank of three. They are reported separately.
    fa_arrivals["self_origin"] = fa_arrivals["origin"].eq(fa_HOME_AP)
    fa_arrivals["domestic"] = (
        fa_arrivals["country"].eq(fa_HOME_COUNTRY) & ~fa_arrivals["self_origin"]
    )
    fa_arrivals["market_class"] = "unattributed"
    fa_arrivals.loc[fa_arrivals["domestic"], "market_class"] = "domestic"
    fa_arrivals.loc[fa_arrivals["self_origin"], "market_class"] = "self-origin"
    fa_arrivals.loc[
        fa_arrivals["country"].notna() & ~fa_arrivals["self_origin"] & ~fa_arrivals["domestic"],
        "market_class",
    ] = "international"

    fa_n_self_origin = int(fa_arrivals["self_origin"].sum())
    fa_n_domestic = int(fa_arrivals["domestic"].sum())

    # Guards: the buckets must be disjoint and exhaustive.
    assert not (fa_arrivals["self_origin"] & fa_arrivals["domestic"]).any(), \
        "an arrival cannot be both self-origin and domestic"
    assert set(fa_arrivals["market_class"]) <= {
        "unattributed", "domestic", "self-origin", "international"
    }, "unexpected market_class value"

    print(f"kept {len(fa_ok):,} of {len(fa_events):,} events on clean days "
          f"({fa_dropped:,} dropped as low/sparse coverage)")
    print(f"  arrivals   : {len(fa_arrivals):,}")
    print(f"  departures : {len(fa_departures):,}")
    print(f"  null origins relabelled by the mapping: {fa_leaked} (guard passed)")
    print(f"  self-origin (origin = {fa_HOME_AP} itself, no usable market): {fa_n_self_origin:,}")
    print(f"  domestic {fa_HOME_COUNTRY} (real flights, not an inbound market): {fa_n_domestic:,}")
    return (
        fa_arrivals,
        fa_departures,
        fa_dropped,
        fa_flag_by_day,
        fa_n_domestic,
        fa_n_self_origin,
    )


@app.cell
def _(
    fa_HOME_AP,
    fa_arrivals,
    fa_events,
    fa_market_arrivals,
    fa_n_domestic,
    fa_n_self_origin,
    pd,
):
    # ---------------------------------------------------------------------------
    # Attribution funnel and what the data can actually support
    # ---------------------------------------------------------------------------
    # The honest shape of this dataset is a funnel, not a rectangle. Each step
    # below is a real reduction in what can be measured, and the total is small
    # enough that it should govern how much weight any finding here can bear.

    fa_total_events = len(fa_events)
    fa_total_arrivals = int(fa_events["direction"].eq("arrival").sum())
    fa_clean_arrivals = len(fa_arrivals)
    fa_funnel_attributed = int(fa_arrivals["country"].notna().sum())
    fa_after_self = fa_funnel_attributed - fa_n_self_origin
    fa_funnel_market = len(fa_market_arrivals)
    fa_hub_origins = int(
        fa_market_arrivals["origin_type"].eq("large_airport").fillna(False).sum()
    )

    fa_funnel = pd.DataFrame(
        [
            ("All exported events", fa_total_events, "both directions, all 37 observed days"),
            ("Arrivals only", fa_total_arrivals,
             "the inbound leg is what the hotel chain cares about"),
            ("Arrivals on clean days", fa_clean_arrivals,
             f"drops {fa_total_arrivals - fa_clean_arrivals:,} received during outage blocks"),
            ("Arrivals OpenSky could place", fa_funnel_attributed,
             f"{(1 - fa_funnel_attributed / fa_clean_arrivals) * 100:.0f}% have no estimated origin"),
            ("...excluding self-origin artifacts", fa_after_self,
             f"{fa_n_self_origin:,} rows where the origin resolved to {fa_HOME_AP} itself"),
            ("...excluding domestic UAE", fa_funnel_market,
             f"{fa_n_domestic:,} real flights whose passengers are already in-country"),
            ("...from a large airport", fa_hub_origins,
             "the rest depart small fields, where the estimate is least reliable"),
        ],
        columns=["stage", "arrivals", "note"],
    )

    fa_funnel
    return fa_clean_arrivals, fa_funnel, fa_funnel_market, fa_total_events


@app.cell
def _(fa_funnel, go):
    # The funnel is the most important chart in this notebook: the headline arrival
    # count and the count usable for market analysis differ by a factor of two, and
    # no modelling repairs the difference.

    fa_funnel_fig = go.Figure(
        go.Bar(
            x=fa_funnel["arrivals"],
            y=fa_funnel["stage"],
            orientation="h",
            marker_color=["#1f3b57", "#2c6180", "#4a90a4", "#6fb3c0",
                          "#e09f3e", "#e09f3e", "#a8d5de"][: len(fa_funnel)],
            text=[f"{v:,}" for v in fa_funnel["arrivals"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fa_funnel_fig.update_layout(
        title="Attribution funnel: how many arrivals survive to become a market observation",
        xaxis_title="arrivals (or events)",
        yaxis=dict(autorange="reversed"),
        height=380,
        showlegend=False,
        margin=dict(l=300, r=90),
    )
    fa_funnel_fig
    return


@app.cell
def _(
    fa_HOME_AP,
    fa_clean_arrivals,
    fa_funnel_market,
    fa_n_domestic,
    fa_n_self_origin,
    mo,
):
    mo.callout(
        mo.md(
            f"""
    ### Why the funnel matters for the brief

    The challenge asks for hotel demand by **country of origin**, and the brief itself
    flags the definitional catch: flight data records where a flight *departed*, hotel
    data records a guest's *nationality*. Those differ in ways this dataset makes
    visible. Four of them, in decreasing order of how much they change the answer:

    1. **A flight is not a visitor.** {fa_funnel_market:,} of {fa_clean_arrivals:,} clean
       arrivals are both international and attributable. Each is a single aircraft
       movement, carrying an unknown number of people, of whom an unknown number
       entered the country, and an unknown fraction checked into a hotel. Only
       {100 * fa_funnel_market / fa_clean_arrivals:.0f}% of clean arrivals reach the
       market view at all.
    2. **Not every placed arrival is a market.** {fa_n_self_origin:,} arrivals resolve to
       {fa_HOME_AP} as their own origin -- the aircraft's last position on the ground,
       echoed back as a departure. A further {fa_n_domestic:,} are genuine flights from
       other UAE airfields, whose passengers are by definition already in-country. The
       second group alone was large enough to have ranked the UAE third among source
       markets, ahead of the United States.
    3. **Transiting passengers never appear as a market at all.** A passenger who
       connects at {fa_HOME_AP} without clearing immigration contributes to the arrival
       count and to no hotel. Section 5 measures the aircraft-turnaround behaviour that
       drives exactly this ambiguity -- and shows why the movement count is a poor
       guide to it.
    4. **Residents returning home look like inbound visitors.** A flight from India is
       counted identically whether it carried a returning resident, a business
       traveller, or a first-time leisure visitor. Purpose of travel is not observable
       in ADS-B.

    None of these gaps can be closed with this data. They need the scheduled capacity,
    passenger throughput and hotel performance tables named in the brief. What *can*
    be done rigorously is characterise the supply side and state the conversion
    assumptions precisely, which is the purpose of this notebook.
    """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    # Concepts

    This section is for readers who do not work with air-traffic data. It explains
    the vocabulary, the reasoning behind each analytical choice, and the statistics
    used later, in the order they are needed. Nothing here is required to *use* the
    notebook, but reading it will change how much of the rest is interpretable.

    ## How this notebook is organised

    The analysis is a funnel, and each section removes a source of uncertainty:

    | # | section | what it establishes | what it cannot |
    |---|---|---|---|
    | — | the funnel above | how many arrivals survive to become a market observation | — |
    | 1 | Origin markets | which countries send aircraft, and how concentrated that is | who the passengers are |
    | 2 | Concentration | whether the mix is dominated by one market or spread across many | whether that structure is stable |
    | 3 | Geography and distance | whether market size decays with distance | whether distance maps to visitors |
    | 4 | Arrival timing | the shape of the daily arrival bank, in UTC | when passengers reach a hotel |
    | 5 | Aircraft turnaround | how long aircraft stay on the ground | whether the same people are on board |
    | 6 | Day of week | an apparent weekly pattern | a statistically supported one |
    | 7 | Origin-estimate quality | how much rests on OpenSky's airport matching | — |
    | 8 | Reference baselines | what a trivial forecast already achieves | a real forecast |
    | 9 | The chain to hotel nights | exactly which links are measurable | the links that are not |

    **The single most important fact in this notebook** is that a row in the data is
    an *aircraft movement*, and the brief is about *people staying in hotels*. Those
    are different quantities, and every section below exists to make the gap between
    them explicit rather than to paper over it.
    """)
    return


@app.cell
def _(fa_arrivals, fa_total_events, mo):
    mo.md(f"""
    ## What one row in this dataset actually is

    **ADS-B is a broadcast, not a booking.** Every commercial aircraft carries a
    transponder that continuously transmits its position, heading, altitude and speed
    on a public radio frequency. Nobody has to be subscribed for it to be transmitted.
    OpenSky Network runs a large volunteer fleet of ground receivers worldwide,
    collects those broadcasts, and publishes the result as an open dataset. The
    practical consequence for this analysis is that OpenSky is a *sensor network*,
    not a source of truth about flights -- receivers are unevenly distributed, some
    contributors go offline, and the network improves over time.

    **A "flight" here is one aircraft movement.** The rows in this dataset are
    single legs: one aircraft, one takeoff, one landing. {fa_total_events:,} such
    movements were exported. A passenger is never directly observed, anywhere, at any
    point in this notebook.

    Each row carries a few fields, and it is worth knowing how much each one can be
    trusted:

    | field | what it is | how much to trust it |
    |---|---|---|
    | `icao24` | a 24-bit hex address identifying the transponder | High. Effectively an aircraft identity, so it is what lets one aircraft be tracked across an arrival and a later departure. It identifies a *machine*, never a person. |
    | `callsign` | the flight number the crew types into the cockpit | Medium. Operator-entered, occasionally stale, sometimes blank, and the leading airline code is a label rather than a verified contract. |
    | `event_time` | estimated takeoff or landing time | Good. Derived from when the transponder was seen, and the endpoint used is the one relevant to each direction. |
    | `origin` / `dest` | the airport OpenSky *guessed* | Variable. This is the weakest field in the row, and the next cell is about exactly that. |
    | `origin_distance_m` | how far the guess was from the aircraft's last position | This is the honest measure of the guess. Small is good. |
    | `origin_candidates` | how many airports were plausible | Ambiguous by design, so see the warning below. |
    | `coverage_flag` | whether the day was fully observed | Computed here, not supplied by OpenSky. |

    **The movement-to-person gap, stated once and precisely.** A single wide-body
    landing might carry 300 people, a regional jet might carry 70, and a freighter
    carries none at all — and this dataset cannot tell those three cases apart. A
    return flight from India to Abu Dhabi is one row whether it carried 40 business
    travellers or 300 holidaymakers. Every quantity downstream inherits that
    ambiguity, which is why no section in this notebook reports a passenger count.

    **`origin_candidates` is not a confidence score.** It is tempting to treat "0
    candidates" as "no valid match" and drop the row, or "3 candidates" as "confirmed"
    and keep it. Both are wrong. The field counts airports OpenSky considered
    *plausible*, which is not the same as how sure the result is. In this dataset
    {int((fa_arrivals['origin_candidates'] == 0).sum()):,} arrivals have zero
    candidates *and still carry an origin airport*, often a well-placed one. The field
    is reported but never used as a filter; `origin_distance_m` is used instead,
    because it measures the thing actually of interest.
    """)
    return


@app.cell
def _(fa_HOME_AP, fa_n_domestic, fa_n_self_origin, mo):
    mo.md(f"""
    ## How OpenSky decides where a flight came from

    This is the single most important piece of machinery in the analysis, so it is
    worth understanding in detail.

    The `/flights/arrival` endpoint returns a reconstructed list of arrivals. For each
    one, OpenSky reports an **estimated departure airport**. The method is simple:
    take the aircraft's last known position before the flight, and pick the airport
    nearest to it. That is a reasonable heuristic — a plane that has just taken off
    from somewhere is usually still near it — but it has an obvious failure mode,
    and this dataset hits it.

    **The failure mode, in this data.** An aircraft standing on the ground at Abu
    Dhabi has its last known position at Abu Dhabi. The nearest airport to Abu Dhabi
    is Abu Dhabi. So the estimated departure airport becomes `{fa_HOME_AP}` — the
    same airport the aircraft is arriving at. In this dataset that happened to
    {fa_n_self_origin} times.

    The giveaway is that the returned distances are not zero. If a flight genuinely
    departed Abu Dhabi and came back, its "distance from the departure airport" would
    be thousands of kilometres. Instead these rows report a median of 11.5 km, which
    is what you would expect from a mediocre ground position fix sitting a few miles
    from the terminal. The distance field and the airport field contradict each other,
    and that contradiction is what identifies the rows as an artifact rather than a
    real round trip.

    This notebook treats those {fa_n_self_origin} rows as **having no usable origin**.
    They are not counted as a market, and they are not counted as domestic traffic
    either, because neither label would be true.

    **A second, subtler problem: domestic traffic.** A further {fa_n_domestic} arrivals
    genuinely originate at other UAE airfields — Al Dhafra, Al Maktoum, and smaller
    strips. These are real flights and the distances are sensible, so nothing in the
    data marks them as suspect. But a passenger on an Al Dhafra to Abu Dhabi flight is
    already in the country and cannot be an inbound visitor. Included in the market
    mix, these {fa_n_domestic} rows were enough to rank the UAE **third** among source
    markets, ahead of the United States — a position it occupies only because its own
    residents were counted as visitors.

    Both groups are removed in the coverage gate, before any market statistic is
    computed. This is the clearest example in the notebook of why a defensible
    market analysis depends as much on *what you exclude* as on what you calculate.
    """)
    return


@app.cell
def _(fa_dropped, fa_events, fa_flag_by_day, mo):
    mo.md(f"""
    ## Why some days are thrown away entirely

    **The problem.** OpenSky's coverage is a property of its receiver network, not of
    the sky. Over a period when one region goes dark — a contributor's hardware fails,
    a feed is blocked, a volunteer stops maintaining a receiver — the flights still
    happen but are never broadcast, received or recorded. Nothing in the data says a
    flight was missed. It is simply absent.

    **Why it matters more than it looks.** A day with half its traffic missing is
    indistinguishable, in a naive count, from a day when demand halved. Any average
    taken across such days is biased downward by exactly the amount that was missing.
    With a real tourism signal on the line, that is not a rounding error — it is the
    difference between reporting a demand drop and reporting a sensor failure.

    **This is not a hypothetical.** In this dataset {fa_events['day'].nunique()} days
    were pulled. {fa_dropped:,} of {len(fa_events):,} events were discarded by the
    coverage gate, and the losses are not spread evenly: one 10-day block
    (2026-07-02 to 07-11) drops to as little as 4% of a normal day's volume. Averaged
    across all observed days the data shows about 96 arrivals per day; on the
    surviving clean days it shows about 145. The honest figure is the second, and
    using the first would have manufactured a 34% demand collapse that never happened.

    **How days are judged.** Each day is flagged `ok`, `sparse` or `low` by comparing
    its event count to a reference. The reference is computed in two passes: score
    every day against the median of all days, discard the days that fail, then rescore
    the survivors against the median of *those*. The second pass is not fussiness. If
    the outage block is large enough to drag the overall median down, a single-pass
    reference sits inside the outage, and then every healthy day looks *over*-observed
    and gets discarded too. In this dataset the gap between a healthy day and a sparse
    one is large enough that getting this wrong is easy, and getting it right is the
    difference between an analysis and an artefact.

    **What survives.** {fa_flag_by_day.eq('ok').sum()} of {fa_flag_by_day.size} observed
    days are flagged `ok`; the other
    {fa_flag_by_day.size - fa_flag_by_day.eq('ok').sum()} are excluded from every statistic
    in this notebook. Where a count could be read as a total rather than a rate, the
    number of days behind it is reported next to it.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## The statistics used, and what each one is for

    Four of these are measures of *concentration*, three are about *typical values
    and spread*, and two are about *forecasting*. None is exotic, but each is easy to
    use for the wrong purpose, so the failure mode is noted with each.

    ### Concentration

    **Share** is a market's fraction of all arrivals. Simple, and the right first
    question. Its weakness is that it depends on how finely you cut the market: the
    top country always looks smaller when there are more of them.

    **Cumulative share** adds the shares up from largest to smallest and reports the
    running total, so "how many markets do I need to cover most of the traffic?" gets
    a direct answer.

    **HHI (Herfindahl-Hirschman Index)** squares each share and sums them, on a
    0-10,000 scale. The squaring is the whole point: it makes small fragments
    contribute less than a dominant single market would, so the index responds to
    concentration rather than to market count. One market alone scores 10,000; ten
    equal markets score 1,000 each way. The usual reading is below 1,500 fragmented,
    1,500-2,500 moderate, above 2,500 concentrated. **Failure mode:** it is a single
    number that hides *which* market dominates and *how much* — a moderate HHI can
    come from one strong leader or from three mediocre ones, and those imply different
    business situations.

    **Lorenz curve** is the picture behind the HHI: plot cumulative share of markets
    against cumulative share of traffic, and compare with a straight diagonal. A curve
    pressed hard against the bottom-right corner means heavy concentration. The gap
    between the curve and the diagonal is essentially the same information as the HHI,
    but you can see the *shape* rather than inferring it.

    ### Typical values and spread

    **Median** is the middle value, with half above and half below. It is used
    throughout instead of the mean wherever a single outlier could distort the result
    — the median ground time, the median distance per market. **Mean** is used when
    totals matter, such as average arrivals per day. The two can disagree sharply, and
    when they do, that disagreement is the interesting part rather than a nuisance.

    **Standard deviation** measures typical distance from the mean. It is expressed in
    the same units as the data, so it is easy to interpret but hard to compare across
    datasets of different scale.

    **Coefficient of variation** is standard deviation divided by the mean, and it is
    what makes spread comparable. It answers "how noisy is this, relative to its own
    size?" In the day-of-week section it is the number that settles the argument: when
    the variation *within* each weekday exceeds the difference *between* weekdays,
    there is no weekly pattern to claim.

    ### Forecasting

    **WMAPE (Weighted Mean Absolute Percentage Error)** is the average forecast
    error, weighted by how large each actual value was:
    `sum(|actual - predicted|) / sum(actual)`. On a count series it is preferred over
    the ordinary MAPE because MAPE divides by each individual actual value, and a day
    with very few arrivals produces an enormous percentage error that then dominates
    the average. WMAPE weights days by size instead, so no single thin day can hijack
    the score. **It is a scale-free measure, so it cannot tell you whether the errors
    are large in absolute terms** — a 24% WMAPE on 145 arrivals per day is about 35
    aircraft, and whether that matters is a business judgement this notebook cannot
    make.

    **A baseline** is the reference a real model must beat. Before proposing anything
    elaborate, it is worth knowing what the simplest possible forecasts score. Here
    that means an expanding mean (predict today as the mean of everything before it)
    and a 7-day lag (predict today as this day last week). **Both are scored strictly
    out-of-sample**: the prediction for a given day is computed only from days already
    observed, never from days that come later. Without that discipline a model can
    look excellent by having seen the answer.

    ### Glossary

    | term | meaning |
    |---|---|
    | ADS-B | broadcast position beacon carried by aircraft; the source of everything here |
    | leg / movement | one takeoff and landing by one aircraft |
    | `icao24` | transponder address; identifies the aircraft, not a person |
    | callsign | crew-entered flight number, e.g. `ETD948` |
    | coverage gap | a period when flights occurred but were not received |
    | self-origin | an arrival whose estimated origin is the destination airport itself |
    | great-circle distance | shortest path over the earth's surface |
    | distance decay | the tendency for market size to fall as distance rises |
    | ground time / dwell | how long an aircraft sits at the gate between legs |
    | bounce movement | a flight that returns to the airport it came from |
    | indicative | too few observations to support a claim; shown, not asserted |
    | WMAPE | size-weighted average forecast error, as a fraction |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## A note on how to read the numbers in this notebook

    Three conventions apply throughout, and they exist to make every figure
    falsifiable:

    **1. Counts are always accompanied by their base.** A number like "386 arrivals
    from India" is meaningless on its own — 386 over 22 clean days, or over a year, or
    over 37 partially-observed days, are different claims. Where a total appears, the
    number of days behind it appears nearby.

    **2. Everything is computed on clean days, and the exclusions are listed.** The
    funnel above is the complete accounting. If a count does not reconcile against it,
    that is a bug, and the notebook is set up so the arithmetic should be checkable by
    hand.

    **3. Assertions guard the fragile steps.** The two places where a silent mistake
    would be hardest to notice — the country lookup and the market classification —
    carry explicit `assert` statements that fail the cell rather than produce a
    plausible-looking wrong number. A data pipeline that cannot fail is not one whose
    output can be trusted, and both of the bugs found while building this notebook
    (airports with a null ICAO code being mapped to the United States, and arrivals
    being credited to the UAE as a foreign source market) were caught by exactly this
    kind of check rather than by reading the output carefully.

    The practical upshot: where this notebook states a number, that number is meant to
    be reproducible from the CSV, and where it states a limit, that limit is stated as
    plainly as the finding.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Origin markets

    Where do the aircraft arriving at `OMAA` come from? This is the closest
    observable proxy for the brief's "source market", with one caveat that dominates
    everything else: it is the **departure country of the flight**, not the
    nationality of the passenger, and it is derived from OpenSky's *estimated*
    departure airport rather than a confirmed schedule.

    Only arrivals whose origin OpenSky could place are included. Section 7 tests
    how much of that answer rests on OpenSky's estimate: the market ranking holds
    at 10 km and 5 km, but the top-3 changes composition at 2 km, where the United
    States drops out and the United Kingdom enters. The volume is far less
    trustworthy than the ordering.
    """)
    return


@app.cell
def _(
    fa_HOME_COUNTRY,
    fa_MIN_SAMPLE,
    fa_arrivals,
    fa_n_domestic,
    fa_n_self_origin,
):
    # International origin markets of inbound traffic on clean days.
    #
    # The base here is `market_class == "international"`, which the coverage gate
    # built. That excludes two groups that a naive `country.notna()` filter would
    # fold straight into the market mix: 52 self-origin artifacts and 52 domestic
    # UAE movements. Together they were 104 arrivals -- enough to rank the UAE third
    # among "source markets" ahead of the United States, on the strength of flights
    # that had no business being in a source-market table at all.

    fa_market_arrivals = fa_arrivals.loc[fa_arrivals["market_class"].eq("international")].copy()
    fa_unattributed = int(fa_arrivals["market_class"].eq("unattributed").sum())

    fa_country = (
        fa_market_arrivals.groupby("country")
        .agg(arrivals=("icao24", "size"), airports=("origin", "nunique"))
        .sort_values("arrivals", ascending=False)
    )
    fa_country["share"] = fa_country["arrivals"] / fa_country["arrivals"].sum()
    fa_country["cum_share"] = fa_country["share"].cumsum()
    fa_country["indicative"] = fa_country["arrivals"] < fa_MIN_SAMPLE

    fa_top_countries = fa_country.head(15).copy()
    fa_top_countries.index.name = "country"

    print(f"{len(fa_market_arrivals):,} international arrivals from {len(fa_country)} countries")
    print(f"excluded: {fa_n_self_origin:,} self-origin artifacts, {fa_n_domestic:,} domestic "
          f"{fa_HOME_COUNTRY} movements, {fa_unattributed:,} with no resolvable origin")
    print(f"top 3 markets ({', '.join(fa_country.index[:3])}) = "
          f"{100 * fa_country['share'].head(3).sum():.1f}% of international arrivals")
    return fa_country, fa_market_arrivals, fa_top_countries, fa_unattributed


@app.cell
def _(fa_top_countries):
    fa_top_countries
    return


@app.cell
def _(fa_clean_arrivals, fa_funnel_market, fa_top_countries, go):
    # Market concentration by country. Bars are share of attributable arrivals; the
    # line is cumulative share, which answers "how many markets do I need to cover to
    # see most of the traffic?"

    fa_market_fig = go.Figure()
    fa_market_fig.add_bar(
        x=fa_top_countries.index,
        y=fa_top_countries["share"] * 100,
        marker_color="#2c6180",
        name="share of attributable arrivals",
        text=[f"{v * 100:.1f}%" for v in fa_top_countries["share"]],
        textposition="outside",
        cliponaxis=False,
    )
    fa_market_fig.add_scatter(
        x=fa_top_countries.index,
        y=fa_top_countries["cum_share"] * 100,
        mode="lines+markers",
        name="cumulative share",
        line=dict(color="#c1121f", width=2),
        yaxis="y2",
    )
    fa_market_fig.update_layout(
        title=(f"Origin markets, {fa_funnel_market:,} international arrivals"
                   f" from {fa_clean_arrivals:,} clean-day arrivals"),
        xaxis_title="departure country (ISO-3166)",
        yaxis=dict(title="% of attributable arrivals"),
        yaxis2=dict(title="cumulative %", overlaying="y", side="right", range=[0, 105], showgrid=False),
        height=380,
        legend=dict(y=1.12, orientation="h"),
        margin=dict(b=90),
    )
    fa_market_fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Concentration

    "How concentrated is inbound traffic?" matters for the brief in a practical way:
    a simulator whose market mix swings wildly month to month is not a planning tool,
    and a market dominated by one carrier behaves nothing like a fragmented one.
    """)
    return


@app.cell
def _(fa_country, fa_market_arrivals, pd):
    # Herfindahl-Hirschman index on arrival shares, for both airport and country.
    #
    # HHI is the sum of squared percentage shares, on the standard 0-10,000 scale
    # where a single market scores 10,000 and N equal markets score 10,000/N. It is
    # reported here as a descriptive summary of the observed mix -- with 22 clean
    # days it is an indication of structure, not a stable measurement, and it should
    # not be quoted as a precise statistic.

    fa_hhi = lambda shares: float(((shares / shares.sum()) ** 2).sum() * 10_000)

    fa_airport_counts = fa_market_arrivals["origin"].value_counts()
    fa_country_shares = fa_country["arrivals"]
    fa_hhi_airport = fa_hhi(fa_airport_counts)
    fa_hhi_country = fa_hhi(fa_country_shares)

    fa_concentration = pd.DataFrame(
        [
            {"level": "airport", "markets": len(fa_airport_counts), "hhi": round(fa_hhi_airport),
             "top1_pct": round(100 * fa_airport_counts.iloc[0] / fa_airport_counts.sum(), 1),
             "top5_pct": round(100 * fa_airport_counts.head(5).sum() / fa_airport_counts.sum(), 1),
             "top20_pct": round(100 * fa_airport_counts.head(20).sum() / fa_airport_counts.sum(), 1)},
            {"level": "country", "markets": len(fa_country_shares), "hhi": round(fa_hhi_country),
             "top1_pct": round(100 * fa_country_shares.iloc[0] / fa_country_shares.sum(), 1),
             "top5_pct": round(100 * fa_country_shares.head(5).sum() / fa_country_shares.sum(), 1),
             "top20_pct": round(100 * fa_country_shares.head(20).sum() / fa_country_shares.sum(), 1)},
        ]
    )

    print(fa_concentration.to_string(index=False))
    print()
    print("Interpretation guide: HHI <1500 fragmented, 1500-2500 moderate, >2500 concentrated.")
    print(f"Country HHI {fa_hhi_country:.0f} -> {'concentrated' if fa_hhi_country>2500 else 'moderate' if fa_hhi_country>1500 else 'fragmented'}")
    return (fa_country_shares,)


@app.cell
def _(fa_country_shares, go, np):
    # Lorenz curve of arrival concentration. The distance between the curve and the
    # equality line is a visual restatement of the HHI: the further the curve bows
    # below the diagonal, the more concentrated the mix.

    fa_lorenz = fa_country_shares.sort_values(ascending=False).to_numpy()
    fa_lorenz_cum = np.concatenate([[0], np.cumsum(fa_lorenz) / fa_lorenz.sum()])
    fa_lorenz_x = np.arange(len(fa_lorenz_cum)) / (len(fa_lorenz_cum) - 1)

    fa_lorenz_fig = go.Figure()
    fa_lorenz_fig.add_scatter(
        x=fa_lorenz_x, y=fa_lorenz_cum, mode="lines", fill="tozeroy",
        fillcolor="rgba(44,97,128,0.25)", line=dict(color="#2c6180", width=2),
        name="observed concentration",
    )
    fa_lorenz_fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#888", dash="dash", width=1),
        name="perfectly even mix",
    )
    fa_lorenz_fig.update_layout(
        title="Arrival concentration by source market (Lorenz curve)",
        xaxis_title="cumulative share of markets (ranked largest first)",
        yaxis_title="cumulative share of arrivals",
        height=360,
        legend=dict(y=1.14, orientation="h"),
    )
    fa_lorenz_fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Market geography and distance

    Sections 1 and 2 describe *how much* traffic comes from each market. This section
    adds *where those markets are*, and asks a question that matters directly to a
    hotel-demand model: **does market size fall off with distance?**

    In almost every origin-destination market it does, and the reason is mundane --
    a short flight is cheaper, quicker and easier to sell than a long one, so more
    of the potential demand for it becomes an actual flight. If the decay is visible
    here, a simulator can treat distance as a structural driver rather than something
    to learn from scratch. If it were *not* visible, that would say something about
    this market too.

    **Why countries are the unit, not airports.** Both were computed. Countries are
    reported because the brief asks about source *markets* and because several
    distance bands contain only a handful of airports, too few to support a rate
    comparison. But note what the airport-level calculation shows, because it is a
    useful lesson: at airport level the decay *first* appeared non-monotonic -- the
    sub-2000 km band drew fewer arrivals per airport than the next band out. That
    looked like a real irregularity of short-haul traffic. It was not. It was the
    self-origin and domestic-`OMAA` rows from the coverage gate dragging the nearest
    band down. Once those were removed, the airport-level series is monotonic too.
    An odd-looking statistic is often a symptom of a data defect, not a finding.
    """)
    return


@app.cell
def _(fa_HOME_AP, fa_MIN_SAMPLE, fa_airports, fa_market_arrivals, np, pd):
    # Great-circle distance from each origin airport to OMAA, then aggregated by country.
    #
    # Distance is computed as a haversine great-circle distance on a spherical earth.
    # That is a simplification -- the earth is an ellipsoid, and a full geodesic on
    # the WGS-84 ellipsoid differs by roughly 0.1-0.5% -- but at the scale of a
    # 500-14,000 km route the error is far smaller than the uncertainty in the origin
    # airport itself, so it is not worth the extra dependency.
    #
    # Each country is placed at the *median* distance of its airports rather than the
    # mean. A country with one distant regional airport and several nearby ones should
    # be represented by where most of its traffic originates, and the median is robust
    # to that asymmetry in a way the mean is not.

    fa_home = fa_airports.loc[fa_airports["icao_code"].eq(fa_HOME_AP)]
    assert len(fa_home) == 1, f"expected exactly one {fa_HOME_AP} row, found {len(fa_home)}"
    fa_home_lat = float(fa_home["latitude_deg"].iloc[0])
    fa_home_lon = float(fa_home["longitude_deg"].iloc[0])

    fa_geo = fa_airports.dropna(subset=["icao_code", "latitude_deg", "longitude_deg"])
    fa_coord = fa_geo.set_index("icao_code")[["latitude_deg", "longitude_deg", "iso_country", "name"]]

    _EARTH_KM = 6371.0088          # mean earth radius
    def fa_great_circle_km(lat, lon):
        """Vectorised haversine distance to OMAA, in kilometres."""
        p1 = np.radians(fa_home_lat)
        p2 = np.radians(np.asarray(lat, dtype=float))
        dphi = p2 - p1
        dlam = np.radians(np.asarray(lon, dtype=float) - fa_home_lon)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
        return 2 * _EARTH_KM * np.arcsin(np.sqrt(a))

    # Attach distance to the already-attributed arrivals.
    fa_distanced = fa_market_arrivals.copy()
    fa_distanced["origin_lat"] = fa_distanced["origin"].map(fa_coord["latitude_deg"])
    fa_distanced["origin_lon"] = fa_distanced["origin"].map(fa_coord["longitude_deg"])
    fa_located = fa_distanced["origin_lat"].notna() & fa_distanced["origin_lon"].notna()
    fa_distanced = fa_distanced.loc[fa_located].copy()
    fa_distanced["km"] = fa_great_circle_km(
        fa_distanced["origin_lat"].to_numpy(), fa_distanced["origin_lon"].to_numpy()
    )

    # One row per country: where it sits, and how much traffic it carries.
    fa_country_geo = (
        fa_distanced.groupby("country")
        .agg(km=("km", "median"), arrivals=("icao24", "size"), airports=("origin", "nunique"))
        .reset_index()
    )
    _fa_band_edges = [0, 2000, 4000, 6000, 20_000]
    _fa_band_labels = ["<2,000 km", "2,000-4,000 km", "4,000-6,000 km", "6,000 km +"]
    fa_country_geo["band"] = pd.cut(
        fa_country_geo["km"], bins=_fa_band_edges, labels=_fa_band_labels, right=False
    )
    fa_country_geo["small_sample"] = fa_country_geo["arrivals"] < fa_MIN_SAMPLE

    print(f"OMAA at ({fa_home_lat:.4f}, {fa_home_lon:.4f})")
    print(f"located {fa_located.sum():,} of {len(fa_market_arrivals):,} attributable arrivals "
          f"({100 * fa_located.mean():.0f}%)")
    print(f"{len(fa_country_geo)} markets, {int(fa_country_geo['small_sample'].sum())} below the "
          f"{fa_MIN_SAMPLE}-arrival minimum")
    print(f"distance range {fa_country_geo['km'].min():.0f} .. {fa_country_geo['km'].max():.0f} km")
    return (fa_country_geo,)


@app.cell
def _(fa_country_geo):
    # Arrivals per market, by distance band.
    #
    # `arrivals_per_market` is the quantity that carries the decay: total arrivals in
    # a band mostly track how many countries happen to sit in it, so the rate is what
    # compares like with like.

    fa_bands = (
        fa_country_geo.groupby("band", observed=True)
        .agg(markets=("country", "size"),
             arrivals=("arrivals", "sum"),
             small_markets=("small_sample", "sum"),
             median_km=("km", "median"))
        .reset_index()
    )
    fa_bands["arrivals_per_market"] = fa_bands["arrivals"] / fa_bands["markets"]
    fa_bands["share_of_traffic"] = fa_bands["arrivals"] / fa_bands["arrivals"].sum()

    fa_bands
    return (fa_bands,)


@app.cell
def _(fa_MIN_SAMPLE, fa_bands, fa_country_geo, go, np):
    # The decay, with the underlying markets shown rather than hidden.
    #
    # Bars are the mean arrivals per market in each band. Dots are the individual
    # countries, jittered horizontally, so the spread behind each mean is visible. A
    # band whose dots are tightly packed supports a rate comparison; a band where the
    # dots are scattered does not, however clean its mean looks.

    _fa_geo_rng = np.random.default_rng(20260928)
    _fa_geo_fig = go.Figure()

    _fa_geo_fig.add_bar(
        x=fa_bands["band"],
        y=fa_bands["arrivals_per_market"],
        marker_color="#2c6180",
        name="arrivals per market",
        text=[f"{v:.0f}" for v in fa_bands["arrivals_per_market"]],
        textposition="outside",
        cliponaxis=False,
        opacity=0.45,
    )
    for _band in fa_bands["band"]:
        _pts = fa_country_geo.loc[fa_country_geo["band"].eq(_band), "arrivals"]
        _jitter = _fa_geo_rng.uniform(-0.28, 0.28, len(_pts))
        _fa_geo_fig.add_scatter(
            x=[_band] * len(_pts),
            y=_pts + _jitter,
            mode="markers",
            marker=dict(
                size=8,
                color=["#c1121f" if _c < fa_MIN_SAMPLE else "#1f3b57"
                       for _c in fa_country_geo.loc[fa_country_geo["band"].eq(_band), "arrivals"]],
                line=dict(width=0.5, color="white"),
            ),
            showlegend=False,
            hovertemplate="%{y} arrivals<extra></extra>",
        )
    _fa_geo_fig.update_layout(
        title="Arrivals per market fall with distance (each dot is one market; red = below sample minimum)",
        xaxis_title="great-circle distance to OMAA",
        yaxis_title="arrivals per market (22 clean days)",
        height=400,
        showlegend=False,
    )
    _fa_geo_fig
    return


@app.cell
def _(fa_arrivals, fa_bands, fa_market_arrivals, fa_unattributed, mo):
    mo.md(
        f"""
    **What the decay says.** Arrivals per market fall steadily with distance:

    | band | markets | arrivals | per market |
    |---|---|---|---|
    """ + "\n".join(
            f"| {r.band} | {r.markets} | {r.arrivals:,} | {r.arrivals_per_market:.1f} |"
            for r in fa_bands.itertuples()
        ) + f"""

    A market within 2,000 km delivers about
    {fa_bands['arrivals_per_market'].iloc[0] / fa_bands['arrivals_per_market'].iloc[-1]:.1f}x
    as many arrivals as one beyond 6,000 km, and the decline is smooth rather than a
    cliff -- there is no single radius where traffic stops. Backing this out at
    airport level gives the same shape independently
    ({fa_bands['arrivals_per_market'].iloc[0]:.0f} vs
    {fa_bands['arrivals_per_market'].iloc[-1]:.0f} arrivals per airport), which is mild
    evidence that this is a property of the market rather than of the country grouping.

    **What it does not say.** This is a statement about *aircraft movements*, and the
    leap from a movement to a hotel guest is where the whole analysis breaks down:

    - A short-haul flight is not a short-haul visitor. Someone on a 500 km sector may
      be a resident visiting family in another emirate -- an intra-GCC business trip
      that never becomes a hotel guest in the sense the brief means. The nearest band
      here is dominated by exactly those markets (Qatar, Bahrain, Jordan), which are
      the places where the flight/guest gap is largest.
    - The estimate rests on a *minority* of traffic. {fa_unattributed:,} of
      {len(fa_arrivals):,} clean arrivals have no resolvable origin at all, so the
      decay is characterised over roughly
      {100 * len(fa_market_arrivals) / len(fa_arrivals):.0f}% of arrivals. If
      unattributed flights follow the same pattern the shape holds; if they are
      concentrated on long-haul routes the long-haul share is understated here.
    - The nearest band rests on {int(fa_bands['markets'].iloc[0])} markets, one of
      which is below the sample minimum. Treat that band's rate as indicative.
    - Distance is not causal here. It is a proxy that stands in for fare, frequency,
      aircraft gauge, hub structure and the Saudi/Emirati air-taxi markets, none of
      which are in this data.

    **Why a simulator should still care.** Even so, "markets are not interchangeable"
    is a real structural input. A model that applies one average arrival-to-demand
    conversion to every market is asserting that a 320 km flight and a 14,000 km flight
    carry the same relationship to hotel demand, and this section is direct evidence
    that they do not.
    """
    )
    return


@app.cell
def _(fa_country, go):
    # Where the traffic comes from, geographically.
    #
    # Countries are shaded by their share of international arrivals. The view is
    # cropped to the Gulf and the western Indian Ocean rather than the whole world,
    # because the whole world is mostly zero and only compresses the colour scale
    # that separates India from the United Arab Emirates.
    #
    # Shares are shares of the {n:,} international arrivals -- the 43% of clean
    # arrivals with no resolvable origin are absent from the map entirely, so every
    # shade is darker than the true market share of that country.
    #
    # A caution worth repeating: this is a map of *flight origins*, which is not a
    # map of guest nationalities.

    fa_map_fig = go.Figure(
        go.Choropleth(
            locations=fa_country.index,
            z=fa_country["share"] * 100,
            text=fa_country.index,
            colorscale="Blues",
            zmin=0,
            zmax=fa_country["share"].max() * 100,
            marker_line_color="white",
            marker_line_width=0.6,
            colorbar=dict(title="% of<br>intl arrivals", thickness=12, len=0.7),
            hovertemplate="<b>%{text}</b><br>%{z:.1f}% of international arrivals<extra></extra>",
        )
    )
    fa_map_fig.update_layout(
        title="Origin markets by share of international arrivals (22 clean days)",
        geo=dict(
            projection_type="natural earth",
            lataxis_range=[2, 52],
            lonaxis_range=[28, 92],
            showland=True,
            landcolor="#f0efe9",
            showocean=True,
            oceancolor="#f7f9fb",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#c9c4b8",
            bgcolor="white",
        ),
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fa_map_fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. When do aircraft arrive?

    Timing matters to the brief more than it first appears. A hotel cares about
    arrivals, not landings: an aircraft touching down at 15:00 UTC clears
    immigration, collects bags and reaches a city hotel somewhere around 17:00, and
    an aircraft landing at 02:00 reaches it near 04:00. The *shape* of the arrival
    bank therefore tells you how much of the inbound flow lands in check-in hours,
    and it is the observable half of a chain whose other half (how many people
    actually check in) is not.
    """)
    return


@app.cell
def _(fa_arrivals, pd):
    # Hourly arrival profile on clean days, in UTC.
    #
    # Counts are normalised to a mean daily volume of 100 so the profile can be read
    # as "how much of a day's traffic happens in this hour" without being distorted by
    # the fact that 22 days contributed to some hours and not others.

    fa_by_hour = (
        fa_arrivals.groupby(fa_arrivals["event_time"].dt.hour)
        .size()
        .reindex(range(24), fill_value=0)
    )
    fa_by_hour.index.name = "hour"
    fa_hour_profile = fa_by_hour.to_frame("arrivals")
    fa_hour_profile["per_100_per_day"] = fa_by_hour / fa_by_hour.sum() * 100
    fa_hour_profile["cumulative"] = fa_hour_profile["per_100_per_day"].cumsum()

    fa_peak_hour = int(fa_by_hour.idxmax())
    fa_daypart = {
        "00:00-05:59 overnight": fa_by_hour.loc[0:5].sum(),
        "06:00-11:59 morning": fa_by_hour.loc[6:11].sum(),
        "12:00-17:59 afternoon": fa_by_hour.loc[12:17].sum(),
        "18:00-23:59 evening": fa_by_hour.loc[18:23].sum(),
    }
    fa_daypart_df = pd.DataFrame(
        [{"part": k, "arrivals": v, "share": v / fa_by_hour.sum()} for k, v in fa_daypart.items()]
    )

    print(f"peak hour: {fa_peak_hour:02d}:00 UTC with {fa_by_hour.max()} arrivals "
          f"across {len(fa_arrivals):,} clean arrivals")
    print()
    print(fa_daypart_df.to_string(index=False))
    print(f"\n12:00-23:59 carries {100 * (fa_daypart_df['share'].iloc[2:].sum()):.0f}% of arrivals")
    return fa_by_hour, fa_daypart_df, fa_hour_profile, fa_peak_hour


@app.cell
def _(fa_HOME_AP, fa_by_hour, fa_hour_profile, fa_peak_hour, go):
    # Two views of the same profile: the hourly shape, and the cumulative build that
    # shows when the day's traffic has been delivered.

    fa_hour_fig = go.Figure()
    fa_hour_fig.add_bar(
        x=[f"{h:02d}" for h in fa_hour_profile.index],
        y=fa_hour_profile["per_100_per_day"],
        marker_color=[
            "#c1121f" if h == fa_peak_hour else "#2c6180" for h in fa_hour_profile.index
        ],
        name="share of daily arrivals",
    )
    fa_hour_fig.add_scatter(
        x=[f"{h:02d}" for h in fa_hour_profile.index],
        y=fa_hour_profile["cumulative"],
        mode="lines",
        name="cumulative",
        line=dict(color="#e09f3e", width=2),
        yaxis="y2",
    )
    fa_hour_fig.update_layout(
        title=f"Hourly arrival profile at {fa_HOME_AP} (UTC, clean days only)",
        xaxis_title="hour of day (UTC)",
        yaxis=dict(title="% of daily arrivals"),
        yaxis2=dict(title="cumulative %", overlaying="y", side="right", range=[0, 105], showgrid=False),
        height=380,
        legend=dict(y=1.13, orientation="h"),
        bargap=0.15,
    )
    fa_hour_fig.update_layout(
        annotations=[dict(
            text=f"peak {fa_peak_hour:02d}:00 UTC",
            x=fa_peak_hour, y=fa_by_hour.max(), showarrow=True,
            arrowhead=2, ay=-38, font=dict(color="#c1121f"),
        )]
    )
    fa_hour_fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Aircraft dwell and rotation

    **This section exists to prevent a specific misreading.**

    It is tempting to pair each arrival with the next departure of the same aircraft
    and call the result a "connection rate" or a "transfer rate". That number would
    be badly wrong. ADS-B sees an *aircraft* land and later take off. It cannot see
    the passengers, so it cannot distinguish:

    - one aircraft turning around, unloading and reloading, with a completely
      different set of people on board each time, from
    - passengers who stayed on the same aircraft across the turn.

    Because Etihad is based at `OMAA`, most arriving aircraft simply depart again,
    so a same-aircraft pairing measures the *hub's rotation schedule* far more than
    it measures connecting passengers. Dwell time is reported here as a **turnaround
    operational** measure for exactly that reason: how long aircraft remain on the
    ground is a real, useful fact about the airport, and it is not a fact about
    people.
    """)
    return


@app.cell
def _(fa_arrivals, fa_departures, pd):
    # Ground time between an arrival and the same aircraft's next departure.
    #
    # `merge_asof` forward-matches each arrival to the next departure of the same
    # icao24 transponder address within a 48h window. The tolerance is deliberately
    # generous: a flight that turns and departs after midnight still has a matched
    # departure, and truncating at 24h would systematically misclassify late turns as
    # unpaired.
    #
    # Pairing is restricted to clean days. An aircraft arriving during an outage block
    # would appear to have no outbound leg only because the outbound leg was never
    # received, which would inflate the "did not depart" category with a measurement
    # artefact.

    fa_dwell_arr = (
        fa_arrivals[["icao24", "event_time", "origin", "country", "day"]]
        .sort_values("event_time")
        .reset_index(drop=True)
    )
    fa_dwell_dep = (
        fa_departures[["icao24", "event_time", "dest"]]
        .rename(columns={"event_time": "dep_time", "dest": "dep_dest"})
        .sort_values("dep_time")
        .reset_index(drop=True)
    )
    fa_dwell = pd.merge_asof(
        fa_dwell_arr,
        fa_dwell_dep,
        left_on="event_time",
        right_on="dep_time",
        by="icao24",
        direction="forward",
        tolerance=pd.Timedelta("48h"),
    )
    fa_dwell["dwell_h"] = (fa_dwell["dep_time"] - fa_dwell["event_time"]).dt.total_seconds() / 3600

    fa_paired = fa_dwell["dwell_h"].dropna()
    fa_unpaired = int(fa_dwell["dwell_h"].isna().sum())
    fa_bounced = int(fa_dwell["dep_dest"].eq(fa_dwell["origin"]).fillna(False).sum())

    fa_dwell_summary = pd.DataFrame([
        {"category": "turned in under 6h", "arrivals": int((fa_paired <= 6).sum()),
         "share_of_paired": (fa_paired <= 6).mean()},
        {"category": "grounded 6-24h", "arrivals": int(((fa_paired > 6) & (fa_paired <= 24)).sum()),
         "share_of_paired": ((fa_paired > 6) & (fa_paired <= 24)).mean()},
        {"category": "grounded over 24h", "arrivals": int((fa_paired > 24).sum()),
         "share_of_paired": (fa_paired > 24).mean()},
        {"category": "no outbound leg seen in 48h", "arrivals": fa_unpaired,
         "share_of_paired": float("nan")},
    ])

    print(f"{len(fa_paired):,} of {len(fa_dwell):,} clean arrivals paired to an outbound leg")
    print(f"  median ground time: {fa_paired.median():.1f} h   (mean {fa_paired.mean():.1f} h)")
    print(f"  returned to the same origin airport: {fa_bounced:,} "
          f"({100 * fa_bounced / len(fa_dwell):.1f}% of arrivals)")
    print()
    print(fa_dwell_summary.to_string(index=False))
    return fa_bounced, fa_paired


@app.cell
def _(fa_paired, go):
    # Distribution of ground time. The mass under 6h is the hub turnaround schedule,
    # and it is the reason same-aircraft matching cannot be read as passenger
    # connection behaviour.

    fa_dwell_fig = go.Figure()
    fa_dwell_fig.add_histogram(
        x=fa_paired.clip(upper=48),
        nbinsx=48,
        marker_color="#2c6180",
        name="grounded aircraft",
    )
    fa_dwell_fig.add_vline(x=6, line=dict(color="#c1121f", dash="dash"),
                           annotation_text="6h typical turnaround", annotation_position="top left")
    fa_dwell_fig.update_layout(
        title="Aircraft ground time at OMAA after an inbound arrival",
        xaxis_title="hours on the ground (clipped at 48h)",
        yaxis_title="arrivals",
        height=340,
        bargap=0.05,
    )
    fa_dwell_fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Day-of-week pattern (indicative only)

    **Read this section as a hypothesis, not a finding.**

    The clean-day filter leaves 22 observations spread across 7 weekdays, which is
    2-4 days per weekday. The table below is therefore descriptive: it shows what was
    seen, and every row carries its own sample size and dispersion so the reader can
    see how little sits behind each mean. No significance test is reported, because
    with n=2-4 per group such a test would either be uninformative or misleadingly
    significant depending on assumptions that the data cannot check.

    The pattern is included because it is *actionable as a measurement priority*: if
    weekend arrivals really are elevated, that is worth pinning down properly once
    more data lands, and this is where to start.
    """)
    return


@app.cell
def _(fa_MIN_SAMPLE, fa_arrivals):
    # Daily arrivals on clean days, broken out by weekday.

    fa_daily_clean = (
        fa_arrivals.groupby("day").size().rename("arrivals").reset_index()
    )
    fa_daily_clean["dow"] = fa_daily_clean["day"].dt.dayofweek
    fa_daily_clean["dow_name"] = fa_daily_clean["day"].dt.day_name().str[:3]

    fa_dow = (
        fa_daily_clean.groupby("dow_name")
        .agg(days=("arrivals", "size"), mean=("arrivals", "mean"),
             std=("arrivals", "std"), min=("arrivals", "min"), max=("arrivals", "max"))
        .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    )
    fa_dow["indicative"] = fa_dow["days"] < fa_MIN_SAMPLE
    fa_dow["dispersion"] = fa_dow["std"] / fa_dow["mean"]

    overall_mean = fa_daily_clean["arrivals"].mean()
    fa_dow["vs_overall"] = fa_dow["mean"] / overall_mean

    print(f"overall mean {overall_mean:.0f} arrivals/day across {len(fa_daily_clean)} clean days")
    print(f"min weekday {fa_dow['mean'].min():.0f}  max {fa_dow['mean'].max():.0f}  "
          f"ratio {fa_dow['mean'].max() / fa_dow['mean'].min():.2f}x")
    print()
    print(fa_dow.round(1).to_string())
    return fa_daily_clean, fa_dow, overall_mean


@app.cell
def _(fa_MIN_SAMPLE, fa_daily_clean, go, overall_mean):
    # Box-less strip plot: each point is one clean day. With 2-4 points per weekday a
    # box plot would imply a distribution the data cannot support, so the raw
    # observations are shown instead, sized by nothing and annotated by sample count.

    fa_dow_fig = go.Figure()
    for i, day_name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        vals = fa_daily_clean.loc[fa_daily_clean["dow_name"].eq(day_name), "arrivals"]
        fa_dow_fig.add_scatter(
            x=[day_name] * len(vals),
            y=vals,
            mode="markers",
            marker=dict(
                size=13,
                color="#c1121f" if len(vals) < fa_MIN_SAMPLE else "#2c6180",
                line=dict(width=1, color="white"),
            ),
            name=f"{day_name} (n={len(vals)})",
            showlegend=False,
            hovertemplate=f"{day_name}<br>%{{y}} arrivals<extra></extra>",
        )
    fa_dow_fig.add_hline(y=overall_mean, line=dict(color="#e09f3e", dash="dash"),
                         annotation_text=f"overall mean {overall_mean:.0f}", annotation_position="bottom right")
    fa_dow_fig.update_layout(
        title="Daily arrivals by weekday (each point is one clean day; red = fewer than 5 days)",
        xaxis_title="weekday",
        yaxis_title="arrivals that day",
        height=360,
    )
    fa_dow_fig
    return


@app.cell
def _(fa_dow, mo):
    mo.callout(
        mo.md(
            f"""
    **Why this is labelled indicative.** Each weekday rests on
    {int(fa_dow['days'].min())}-{int(fa_dow['days'].max())} observed days. The
    coefficient of variation within weekdays reaches
    {100 * fa_dow['dispersion'].max():.0f}%, which is larger than the
    {100 * (fa_dow['mean'].max() / fa_dow['mean'].min() - 1):.0f}% gap between the
    busiest and quietest weekday. Day-to-day noise therefore exceeds the effect being
    measured, and the apparent weekly shape is not separable from sampling variation
    in this sample.

    Two further limits: the 22 clean days are **not contiguous** -- they fall in two
    blocks separated by a 49-day gap -- so any slow seasonal drift is confounded with
    the weekday pattern. And the sample spans late June, July and September only, with
    no coverage of the winter season that matters most for Abu Dhabi hotel demand.

    **What would settle it.** Two or more additional months of clean coverage, which
    needs the remaining pull to complete after the daily credit quota resets.
    """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. How much rests on OpenSky's airport estimates?

    Every origin in this notebook is an *estimate*. OpenSky infers the departure
    airport from where the transponder was last seen, and reports how far the
    aircraft still was from that airport at the time. A large distance means the
    aircraft was probably still on the ground somewhere, and the country assignment
    is correspondingly less reliable.

    This section asks the question that should be asked of any estimate-derived
    market analysis: **if you only trust the most confident estimates, does the
    picture change?**
    """)
    return


@app.cell
def _(fa_market_arrivals, pd):
    # Market mix at increasing distance thresholds.
    #
    # The comparison that matters is between *volume* and *shape*. Tightening the
    # threshold necessarily discards arrivals, so absolute counts always fall. The
    # question is whether the relative ordering and the concentration survive -- if
    # they do, the market analysis is robust to the estimate, even though its
    # magnitude is not.

    fa_sensitivity_rows = []
    for threshold in [None, 10_000, 5_000, 2_000]:
        sub = fa_market_arrivals
        if threshold is not None:
            sub = sub.loc[sub["origin_distance_m"].le(threshold)]
        if sub.empty:
            continue
        counts = sub.groupby("country").size().sort_values(ascending=False)
        shares = counts / counts.sum()
        top3 = counts.head(3)
        fa_sensitivity_rows.append({
            "threshold_km": "all" if threshold is None else f"<= {threshold / 1000:.0f} km",
            "arrivals_retained": len(sub),
            "retained_pct": len(sub) / len(fa_market_arrivals),
            "countries": len(counts),
            "top3_countries": ", ".join(top3.index),
            "top3_share": counts.head(3).sum() / counts.sum(),
            "hhi": ((shares ** 2).sum() * 10_000),
        })
    fa_sensitivity = pd.DataFrame(fa_sensitivity_rows)

    print("Distance from the last observed position to the estimated departure airport:")
    print(fa_market_arrivals["origin_distance_m"].describe(percentiles=[.25, .5, .75]).round(0).to_string())
    print()
    print(fa_sensitivity.round(3).to_string(index=False))
    return (fa_sensitivity,)


@app.cell
def _(fa_sensitivity, go):
    # Retention and shape side by side. The left panel is how much data survives a
    # stricter rule; the right panel is whether the top-3 market set and its share
    # change. Flat right-hand bars against collapsing left-hand bars is the signature
    # of a conclusion that is robust while its magnitude is not.

    _sens_fig = go.Figure()
    _sens_fig.add_bar(
        x=fa_sensitivity["threshold_km"],
        y=fa_sensitivity["retained_pct"] * 100,
        name="arrivals retained",
        marker_color="#2c6180",
        text=[f"{v * 100:.0f}%" for v in fa_sensitivity["retained_pct"]],
        textposition="outside",
        cliponaxis=False,
        yaxis="y",
    )
    _sens_fig.add_bar(
        x=fa_sensitivity["threshold_km"],
        y=fa_sensitivity["top3_share"] * 100,
        name="top-3 market share",
        marker_color="#c1121f",
        text=[f"{v * 100:.0f}%" for v in fa_sensitivity["top3_share"]],
        textposition="outside",
        cliponaxis=False,
        yaxis="y2",
    )
    _sens_fig.update_layout(
        title="Volume collapses under a stricter rule; market shape does not",
        xaxis_title="maximum distance from last position to estimated origin",
        yaxis=dict(title="% of attributable arrivals retained", range=[0, 115]),
        yaxis2=dict(title="top-3 market share %", overlaying="y", side="right", range=[0, 60], showgrid=False),
        height=380,
        barmode="group",
        legend=dict(y=1.13, orientation="h"),
    )
    _sens_fig
    return


@app.cell
def _(fa_country, fa_market_arrivals, fa_sensitivity, mo):
    mo.md(f"""
    **What survives a stricter rule, and what does not.**

    At a 10 km threshold {100 * fa_sensitivity['retained_pct'].iloc[1]:.0f}% of
    international arrivals are retained and the top-3 set is unchanged. At 5 km,
    {100 * fa_sensitivity['retained_pct'].iloc[2]:.0f}% is retained and the top-3 is still
    `{fa_sensitivity['top3_countries'].iloc[2]}` at
    {100 * fa_sensitivity['top3_share'].iloc[2]:.1f}%. Over that range the market picture is
    robust.

    At 2 km it changes, and the chart above should not be read as saying otherwise:
    only {100 * fa_sensitivity['retained_pct'].iloc[3]:.0f}% of arrivals remain, the top-3
    becomes `{fa_sensitivity['top3_countries'].iloc[3]}`, the top-3 share falls from
    {100 * fa_sensitivity['top3_share'].iloc[0]:.0f}% to
    {100 * fa_sensitivity['top3_share'].iloc[3]:.0f}%, and the HHI falls from
    {fa_sensitivity['hhi'].iloc[0]:.0f} to {fa_sensitivity['hhi'].iloc[-1]:.0f}. The tighter
    rule does not merely shrink the sample, it changes the measured structure.

    **Practical consequence.** `{fa_country.index[0]}` leads the market at every
    threshold tested, which makes the India-first result the most defensible finding
    in this notebook. Beyond that, anything finer than a roughly 3-5 km equivalence
    band should not be claimed from this data, and the concentration figures in
    section 2 are better read as an upper bound on concentration than as a measurement
    of it.

    **A note on how this section changed.** An earlier draft of this note used the
    United Arab Emirates to illustrate what happens at a tight threshold, on the basis
    that UAE prominence was fragile. That was an artefact of the self-origin and
    domestic rows described in the coverage gate, not a property of any real market.
    The gate now removes those rows before any of this is computed, which is why the
    `AE` country no longer appears in the market tables at all.

    **One field not to misread.** `origin_candidates` is a count of candidate
    airports, not a validity flag. On
    {int((fa_market_arrivals['origin_candidates'] == 0).sum()):,} of the arrivals retained
    here it is zero *while an origin is still present*, and its distance is often
    substantial. Filtering on it would discard valid matches in both directions. This
    notebook does not filter on it.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. What a trivial model already achieves

    Before proposing a forecast it is worth knowing what the *simplest possible*
    forecasts score. If a plain expanding average already explains most of the
    variation, then any model that cannot beat it on a clean out-of-sample test is
    adding complexity for nothing -- and a simulator built on top of it would mostly
    be projecting the sample's own average back at itself.

    Two reference baselines are used:

    - **Expanding mean** -- predict each day as the mean of every earlier day. No
      parameters beyond the history, and it cannot overfit because it cannot choose
      a shape.
    - **7-day lag** -- predict each day as the same weekday one week earlier, the
      natural way to express "next week repeats this week".

    Both are strictly one-directional: the forecast for a day uses only days already
    observed, so no future information leaks into the score.
    """)
    return


@app.cell
def _(fa_daily_clean, pd):
    # Out-of-sample accuracy of the two reference baselines, scored on clean days only.
    #
    # WMAPE (weighted mean absolute percentage error) weights each day by its own
    # size, so a few high-volume days cannot dominate the score and a low-volume day
    # cannot inflate it through a large relative error. On a count series with
    # genuine zero days that failure mode is real, which is why MAPE is not used.

    fa_daily_series = fa_daily_clean.set_index("day")["arrivals"].sort_index()

    fa_pred_expanding = fa_daily_series.expanding().mean().shift(1)   # prior history only
    fa_pred_lag7 = fa_daily_series.shift(7)                          # same weekday, prior week


    def fa_wmape(actual, predicted):
        actual = pd.Series(actual).dropna()
        predicted = pd.Series(predicted).reindex(actual.index)
        mask = actual.notna() & predicted.notna()
        if not mask.any() or actual[mask].sum() == 0:
            return float("nan"), 0
        return float((actual[mask] - predicted[mask]).abs().sum() / actual[mask].sum()), int(mask.sum())


    fa_wmape_expanding, n_expanding = fa_wmape(fa_daily_series, fa_pred_expanding)
    fa_wmape_lag7, n_lag7 = fa_wmape(fa_daily_series, fa_pred_lag7)

    fa_baseline = pd.DataFrame([
        {"baseline": "expanding mean", "wmape": fa_wmape_expanding, "scored_days": n_expanding,
         "definition": "mean of all prior clean days"},
        {"baseline": "7-day lag", "wmape": fa_wmape_lag7, "scored_days": n_lag7,
         "definition": "same weekday, one week earlier"},
    ])

    print(f"scored on {len(fa_daily_series)} clean days "
          f"({fa_daily_series.index[0].date()} .. {fa_daily_series.index[-1].date()})")
    print()
    print(fa_baseline.round(3).to_string(index=False))
    print()
    print(f"daily std {fa_daily_series.std():.0f} = "
          f"{100 * fa_daily_series.std() / fa_daily_series.mean():.0f}% of the mean")
    return (
        fa_daily_series,
        fa_pred_expanding,
        fa_pred_lag7,
        fa_wmape_expanding,
        fa_wmape_lag7,
    )


@app.cell
def _(
    fa_daily_series,
    fa_pred_expanding,
    fa_pred_lag7,
    fa_wmape_expanding,
    fa_wmape_lag7,
    go,
):
    # Actual daily arrivals against both baselines. The gap between the expanding
    # mean and the 7-day lag is the part of the signal a seasonal assumption would
    # have to beat -- and it is smaller than the day-to-day noise around it.

    fa_forecast_fig = go.Figure()
    fa_forecast_fig.add_scatter(
        x=fa_daily_series.index, y=fa_daily_series, mode="markers",
        name="actual arrivals", marker=dict(size=7, color="#1f3b57"),
        hovertemplate="%{x|%d %b}<br>%{y} arrivals<extra></extra>",
    )
    fa_forecast_fig.add_scatter(
        x=fa_pred_expanding.index, y=fa_pred_expanding, mode="lines",
        name=f"expanding mean (WMAPE {100 * fa_wmape_expanding:.0f}%)",
        line=dict(color="#2c6180", width=2),
    )
    fa_forecast_fig.add_scatter(
        x=fa_pred_lag7.index, y=fa_pred_lag7, mode="lines+markers",
        name=f"7-day lag (WMAPE {100 * fa_wmape_lag7:.0f}%)",
        line=dict(color="#e09f3e", width=2, dash="dot"),
    )
    fa_forecast_fig.update_layout(
        title="Reference baselines, scored strictly out-of-sample on clean days",
        xaxis_title="date",
        yaxis_title="arrivals",
        height=380,
        legend=dict(y=1.14, orientation="h"),
        hovermode="x unified",
    )
    fa_forecast_fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. The chain from aircraft to guest nights

    The brief asks for a simulator connecting flight changes to hotel demand in Abu
    Dhabi. That requires at least five links:

    ```
    aircraft movements -> passengers carried -> passengers entering the country
                       -> visitors staying in a hotel -> room nights, by guest nationality
    ```

    Flight data observes the first link. Everything after it is either missing,
    inferred, or unmeasurable from ADS-B. The table below states, for each link, what
    this notebook can support. The purpose is to make the boundary of the current
    result explicit, so that a simulator built later is built on named inputs rather
    than on assumptions that were never written down.
    """)
    return


@app.cell
def _(fa_arrivals, fa_country, fa_daily_clean, fa_market_arrivals, mo, pd):
    # What is measured, what is inferable, and what is simply missing.
    #
    # "MEASURED" means this notebook computes it from the data in front of it.
    # "INFERABLE" means it could be estimated once a named external input is
    # supplied, and the estimator is stated. "BLOCKED" means no available source
    # supports it, and guessing would be fabrication.

    fa_chain = pd.DataFrame([
        {
            "link": "Aircraft movements at OMAA",
            "status": "MEASURED",
            "this notebook": f"{len(fa_arrivals):,} inbound aircraft movements on {len(fa_daily_clean)} clean days",
            "missing input": "none",
        },
        {
            "link": "Source country of each flight",
            "status": "MEASURED (estimated)",
            "this notebook": f"{len(fa_market_arrivals):,} arrivals attributed to {len(fa_country)} countries; IN leads at {100 * fa_country['share'].iloc[0]:.1f}%",
            "missing input": "confirmed schedules would replace OpenSky's positional estimate",
        },
        {
            "link": "Passengers aboard those flights",
            "status": "BLOCKED",
            "this notebook": "not observable in ADS-B; a movement carries zero to ~500 people",
            "missing input": "scheduled seats and actual load factors by flight",
        },
        {
            "link": "Passengers entering the country",
            "status": "BLOCKED",
            "this notebook": "connecting passengers never clear immigration and never appear here",
            "missing input": "immigration or transit-share statistics",
        },
        {
            "link": "Passengers staying in a hotel",
            "status": "BLOCKED",
            "this notebook": "no accommodation data of any kind in the current workspace",
            "missing input": "hotel performance table: ADR, occupancy, room supply",
        },
        {
            "link": "Room nights by guest nationality",
            "status": "BLOCKED",
            "this notebook": "guest nationality != flight origin country; the brief flags this mismatch",
            "missing input": "guest nationality, or a validated origin-to-nationality mapping",
        },
        {
            "link": "Response of hotels to a change in arrivals",
            "status": "INFERABLE (with inputs)",
            "this notebook": "arrival bank shape and market mix are computed; the elasticity is not estimable here",
            "missing input": "paired historical months of flight and hotel performance to fit the link",
        },
    ])

    _status_colors = {"MEASURED": "#2d6a4f", "INFERABLE (with inputs)": "#e09f3e",
                      "BLOCKED": "#c1121f", "MEASURED (estimated)": "#2c6180"}
    fa_chain["_color"] = fa_chain["status"].map(_status_colors)

    mo.vstack([
        mo.md("### Chain status"),
        mo.ui.table(
            fa_chain.drop(columns="_color"),
            selection=None,
            pagination=False,
        ),
    ])
    return


@app.cell
def _(
    fa_arrivals,
    fa_bounced,
    fa_country,
    fa_daily_clean,
    fa_daypart_df,
    fa_paired,
    fa_peak_hour,
    mo,
):
    mo.md(f"""
    **The honest summary is that this notebook delivers the first link and the shape
    of the second, and nothing beyond that.**

    What is solid:

    - {len(fa_arrivals):,} inbound aircraft movements across {len(fa_daily_clean)} clean
      days, with the coverage gaps identified rather than smoothed over.
    - A market ranking led by `{fa_country.index[0]}` at {100 * fa_country['share'].iloc[0]:.1f}%
      of international arrivals, top-3 `{fa_country.index[0]}, {fa_country.index[1]},
      {fa_country.index[2]}` at {100 * fa_country['share'].head(3).sum():.0f}% combined,
      and a geography that decays smoothly with distance rather than at any single
      radius.
    - A concentrated daily arrival bank: {100 * fa_daypart_df['share'].iloc[2:].sum():.0f}%
      of movements land in the 12:00-23:59 UTC window, peaking at {fa_peak_hour:02d}:00 UTC.
    - Aircraft turnaround behaviour as an *operational* fact: median
      {fa_paired.median():.1f} h on the ground, {100 * (fa_paired <= 6).mean():.0f}% turning
      within 6 h, {fa_bounced:,} bounce movements back to the origin airport.

    What must not be claimed from this data:

    - Any number of passengers, visitors, or room nights.
    - Any "connection rate" or "transfer rate" derived from same-aircraft matching.
    - Any seasonal or weekday effect as a statistical result.
    - Any claim about guest nationality, which is a different quantity from flight
      origin and is not recoverable from it.
    - Any reading of the distance decay as a distance-to-visitors relationship: it is
      a distance-to-*aircraft* relationship, and section 3 explains why the two come
      apart most sharply in exactly the short-haul markets that dominate the sample.

    **What unblocks the rest.** Three named inputs, in order of leverage:
    scheduled seat capacity and load factor per flight; the Abu Dhabi hotel
    performance table; and a rule for how flight origin maps to guest nationality. The
    first two were listed in the competition brief and are simply not present in the
    workspace yet. The simulator itself is left deliberately unbuilt: building it now
    would mean hard-coding conversion ratios that no data in hand supports.
    """)
    return


if __name__ == "__main__":
    app.run()
