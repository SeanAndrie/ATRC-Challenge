import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pathlib
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    # Repo root by walking up from this file, not from the working directory:
    # marimo is commonly launched from notebooks/, where a relative path would
    # resolve to notebooks/notebooks and find nothing.
    here = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
    root = next(
        (c for c in (here, *here.parents) if (c / "pipeline" / "paths.py").exists()),
        here.parent,
    )
    sys.path.insert(0, str(root / "pipeline"))

    import paths

    return np, paths, pd, px, sys


@app.cell
def _(paths):
    # Every derived table this notebook reads now lives in one place,
    # data/processed/. The DCT, ADS-B and airport-movement tables used to sit in
    # three sibling directories under notebooks/data/; the data tiers merged them
    # because all three are validated and analysis-ready, so DCT, ADSB and ADAP
    # are now one directory and one name.
    PROC = paths.PROC
    # The seat map is the tested source of truth, shared with the pipeline.
    AVIATION = paths.AVIATION
    return AVIATION, PROC


@app.cell
def _(AVIATION, sys):
    # The seat map lives in pipeline/aviation and is the tested source of
    # truth. Reusing it keeps the notebook from drifting away from the pipeline
    # that produced the derived CSVs.
    if str(AVIATION) not in sys.path:
        sys.path.insert(0, str(AVIATION))
    from aircraft_seats import add_seats

    return (add_seats,)


@app.cell
def _(mo, np, pd):
    from textwrap import dedent

    GUEST_FACTOR = {"count": 1.0, "thousands": 1_000.0, "millions": 1_000_000.0}

    def wmape(actual, pred):
        """Weighted mean absolute percentage error, the brief's preferred metric."""
        actual = np.asarray(actual, dtype=float)
        pred = np.asarray(pred, dtype=float)
        return 100.0 * np.abs(actual - pred).sum() / np.abs(actual).sum()

    def spearman(a, b):
        """Rank correlation. scipy is not installed, and this is exact."""
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    def ols(y, X):
        """Least squares with standard errors. statsmodels is not installed.

        `X` is a sequence of blocks; each block is either one column or a
        2-D group of columns such as the month dummies. The intercept is added
        here.
        """
        y = np.asarray(y, dtype=float)
        blocks = [np.ones(len(y))]
        for part in X:
            arr = np.asarray(part, dtype=float)
            blocks.append(arr.reshape(len(y), -1) if arr.ndim > 1 else arr)
        X = np.column_stack(blocks)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(len(y) - X.shape[1], 1)
        sigma2 = float(resid @ resid) / dof
        se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
        tss = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float(resid @ resid) / tss if tss else float("nan")
        return beta, se, r2

    def month_dummies(months):
        """Eleven dummies; January is the base month alongside the intercept."""
        m = np.asarray(months)
        return np.column_stack([(m == i).astype(float) for i in range(2, 13)])

    def fourier(months, k=2):
        """Harmonic seasonality: k sine/cosine pairs, far fewer than 11 dummies.

        Used as a better-specified seasonality baseline. With only 46 monthly
        observations, eleven free dummies spend degrees of freedom that a
        four-parameter harmonic leaves free for the signal being tested.
        """
        m = np.asarray(months, dtype=float)
        cols = []
        for i in range(1, k + 1):
            cols.append(np.sin(2 * np.pi * i * m / 12))
            cols.append(np.cos(2 * np.pi * i * m / 12))
        return np.column_stack(cols)

    def md(*parts):
        """mo.md that survives a table interpolated into indented prose.

        A DataFrame's to_markdown() output carries no leading whitespace. Mixed
        into an indented f-string it leaves the surrounding prose four spaces
        deep, so the parser reads that prose as a code block and the table never
        renders as a table. Dedent each part on its own, then join.
        """
        blocks = [
            part.to_markdown(index=False) if isinstance(part, pd.DataFrame)
            else dedent(str(part)).strip("\n")
            for part in parts
        ]
        return mo.md("\n\n".join(blocks))

    return GUEST_FACTOR, fourier, md, month_dummies, ols, spearman, wmape


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Can airport data predict Abu Dhabi hotel demand?

    > **Phase 1 background — public-data investigation.** Retained as evidence, not
    > part of the submission deliverable. See `docs/METHODS.md`.

    **A negative result on the link, and a positive result on the chain.**

    The challenge brief asks for a simulator linking air supply to Abu Dhabi
    hotel demand through an explicit chain: *scheduled seats -> arriving
    passengers -> inbound visitors -> hotel guest nights*. This notebook tests
    each link against public data, because a simulator that hides an
    unidentifiable step behind a confident number is worse than one that admits
    the gap.

    **The result is narrower than "it does not work".** Air arrivals carry
    almost no measurable information about monthly hotel guests: the arrivals
    elasticity is 0.17 with a standard error of 0.18, and a seasonal baseline
    that ignores flights entirely forecasts better than any model that uses them.
    Origin-market ranks do not transfer from seats to rooms. That much is solid.

    **But the chain itself is more reachable than that verdict implies.** Guest
    nights are reconstructible for all ten years from published length of stay,
    and DCT does publish the residency split annually. Both ends of the chain are
    supported by measurement. The unobserved middle is narrower than it looks:
    it is two classifications — monthly residency, and purpose of travel at any
    frequency — and only the second is genuinely unavailable.

    **So the simulator is buildable, and this notebook is the specification for
    how honestly to build it.** Parts 2b, 3b and 4a are the parts that were not
    obvious, and they change the design.

    ## What is in here

    | part | contents |
    | --- | --- |
    | 1 | The question, and why it is genuinely hard |
    | 2 | The hotel panel: 99 official PDFs reconstructed into a 10-year monthly series |
    | 2b | **Guest nights are reconstructible**: the identity that closes the chain |
    | 3 | The aviation side, and an audit of exactly what it contains |
    | 3b | **How much of each market the snapshots saw**, and where the sampling stops working |
    | 4 | Test 1: does the origin mix of arriving seats predict the origin mix of hotel guests? |
    | 4a | **The one place the second link can be measured**, using the 2024 residency split |
    | 5 | Test 2: does aggregate monthly arrival volume predict aggregate monthly guests? |
    | 6 | Test 3: does the relationship hold over a decade including COVID? |
    | 7 | What the evidence permits, and what it forbids |
    | 8 | Sensitivity: which factors actually move hotel guest numbers |
    | 9 | What would change the answer |
    | 10 | Limitations |
    | 11 | **Submission plan**: what gets built, on what evidence, with which levers |

    ## Provenance and reproducibility

    Everything is recomputed here from derived CSVs, so the notebook is
    explorable rather than a report. The canonical implementations live in
    `pipeline/` and the final cell asserts the two agree, so they
    cannot drift apart.

    | script | what it does |
    | --- | --- |
    | `dct_parse.py` | builds the hotel panel from the 99 cached PDFs |
    | `dct_validate.py` | unit-normalises and reconciles months against published annuals |
    | `build_annual_arrivals.py` | extracts the official annual arrivals series from SCAD publications |
    | `test_market_link.py` | Test 1 |
    | `test_aggregate_link.py` | Test 2 |
    """)
    return


@app.cell
def _(mo, np, pd):
    import platform
    from datetime import datetime, timezone
    from importlib.metadata import version

    env = pd.DataFrame(
        [
            ("python", platform.python_version()),
            ("platform", platform.platform()),
            ("marimo", version("marimo")),
            ("pandas", pd.__version__),
            ("numpy", np.__version__),
            ("plotly", version("plotly")),
        ],
        columns=["component", "version"],
    )
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mo.vstack([mo.md(f"### Environment\n\nBuilt {built}."), env])
    return (env,)


@app.cell
def _(env, mo, pd):
    inventory = pd.DataFrame(
        [
            ("dct/dct_kpis.csv", "hotel KPIs, 13 metrics x 4 granularities", 1211, "dct_parse.py"),
            ("dct/dct_guests_nationality.csv", "guests by market, with share and ALOS", 664, "dct_parse.py"),
            ("dct/dct_hotel_supply.csv", "hotel and room inventory", 115, "dct_parse.py"),
            ("dct/dct_coverage.csv", "per-file route and row counts", 99, "dct_parse.py"),
            ("dct/dct_cross_checks.csv", "independent cross-source validations", 32, "dct_parse.py"),
            ("ad_airports/pax_movements.csv", "AUH monthly movements, monthly passengers 2015-2018", 377, "SCAD"),
            ("ad_airports/pax_arrivals_annual.csv", "AUH annual arrivals 2015-2024", 9, "SCAD"),
            ("adsb_2025/omaa_arrivals_2025.csv", "one ADS-B day per month, leg level", 1955, "build_adsb_2025.py"),
            ("adsb_2025/markets_by_month.csv", "flights by origin market and month", 478, "build_adsb_2025.py"),
            ("adsb_2025/capture_rate.csv", "observed vs official movements, with bounds", 12, "calibrate_adsb_2025.py"),
        ],
        columns=["file", "contents", "rows", "built by"],
    )
    mo.vstack(
        [
            env,
            mo.md("### Data inventory\n\nThe hotel panel and the aviation extracts are separate pipelines; part 3 covers what the aviation side does and does not contain."),
            inventory,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. The question, and why it is hard

    The brief's premise is attractive and mostly true: nearly every
    international visitor to Abu Dhabi arrives by air, and a share of them
    stay in hotels. The difficulty is the word *share*.

    Three specific problems stand between arriving passengers and hotel guest
    nights, and none of them is visible in the flight data:

    1. **Transit and transfer.** Some arrivals never enter the emirate. Airline
       transiting passengers are excluded from most published arrival counts,
       but same-flight connections are not.
    2. **Residents.** A returning UAE resident is an arrival and not a hotel
       guest. Residents were 2.8 million of the 5.8 million guests DCT reports
       for 2024 — 48 per cent, not a rounding error but a segment nearly as
       large as the inbound one and with the opposite sign. Part 3a returns to
       this, because DCT does publish the split annually.
    3. **Purpose of travel.** A visitor staying with family occupies no room.
       This is the segment the flight data cannot see at all, and it turns out to
       be the one that matters most.

    There is also a definitional catch the brief calls out explicitly: in the
    flight data *country* means where the aircraft departed, while in the hotel
    data it means the guest's nationality. For most traffic these coincide, but
    for Gulf visitors with second properties, dual nationals, and UAE residents
    transiting, they do not.

    **The honest framing:** the chain is a real commercial question. Two of its
    three unobserved links — residency and purpose of travel — are recoverable
    only in part: DCT publishes the residency split annually, which part 3a
    uses, and never publishes purpose of travel, which is the one that decides
    whether an inbound visitor occupies a room. The rest of this notebook tests
    what can be recovered, and shows that the part which *is* observable carries
    almost no information about the part that is not.
    """)
    return


@app.cell
def _(PROC, pd):
    kpi = pd.read_csv(PROC / "dct_kpis.csv")
    kpi.head(3)
    return (kpi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. The hotel panel

    There is no tidy Abu Dhabi hotel dataset. What exists is 99 monthly
    Department of Culture and Tourism reports to the emirate's hotel partners,
    published as PDFs, in six different layouts across ten years, with metrics
    appearing and disappearing and units switching between raw counts and
    thousands without warning.

    Reconstructing a usable panel was most of the engineering. The scripts in
    `pipeline/dct/dct_*.py` handle routing by report era, unit
    normalisation, and the specific traps below. Nothing is estimated or
    interpolated: anything a report does not print unambiguously is left out.

    | problem | what it was | handling |
    | --- | --- | --- |
    | YTD read as a month | the 2023 September report's key table printed a year-to-date figure in the September column, inflating the month to 3,654K against a true 408K | read the YTD marker and difference against the prior month; continuity now holds 3,246 + 408 = 3,654 |
    | Misfiled source files | 2015-02 and 2018-02 were byte-identical copies of a stray February 2017 report | SHA-256 duplicate detection; both files skipped, and both years correctly lose a month |
    | Prior-year restatement | the 2024 report carries restated 2023 months summing to 5,504K, against 4,065K from the 2023 monthly reports | kept separate and documented; the two are not silently merged |
    | Narrative against table | December 2017 prose says 490,845 while the chart says 490,884 | chart value retained, discrepancy documented |
    | Unit switching | counts in some reports, thousands in others | normalised explicitly and reported per row |
    """)
    return


@app.cell
def _(kpi, mo):
    coverage_matrix = (
        kpi.pivot_table(index="metric", columns="year", values="value",
                        aggfunc="count", fill_value=0)
        .astype(int)
        .rename(columns=str)   # marimo's table view needs string column labels
    )
    mo.vstack([
        mo.md(r"""
        ### Coverage

        How many observations exist per metric per year.

        Five metrics carry a monthly series across the whole decade: **guests,
        occupancy, total revenue, RevPAR and average length of stay**. Those are
        enough to support a stable feature set, and part 2b uses two of them
        together.

        The break is not where it looks. It is not a change of era at 2017, it is
        a change of *unit of account*: the room-and-night family — guest nights,
        room nights, occupied rooms, room revenue, food and beverage revenue and
        other revenue — is published only for 2015-2016, and occupied rooms
        appears once, in 2015. Everything DCT still tracks monthly after 2016 is
        either a rate or a total in currency.

        That is the single most consequential coverage fact in the notebook, and
        it cuts against the usual reading. Guest nights *are* reconstructible for
        the whole decade, because length of stay is published throughout. The
        chain's terminal node is reachable; see part 2b.
        """),
        coverage_matrix,
    ])
    return


@app.cell
def _(GUEST_FACTOR, kpi, md, mo):
    # The chain's terminal node is guest nights. DCT stops publishing them
    # monthly after 2016, but average length of stay runs the full decade, and
    # nights = guests x length of stay is an identity rather than a model. This
    # cell reconstructs the series and checks it against the months where DCT
    # printed the answer, so the claim rests on a measurement and not on an
    # assumption about what length of stay means.
    # Unit normalisation matters here: DCT reports the same series in `count` in
    # some years and in `thousands` in others, so a raw read of `value` makes
    # 2018-01 look like 437 guests. Every count in this cell goes through
    # GUEST_FACTOR, the same map the guest series uses.
    alos_m = (
        kpi[(kpi.metric == "alos") & (kpi.granularity == "month")]
        [["year", "period", "value"]]
        .rename(columns={"value": "alos"})
    )
    guests_m = (
        kpi[(kpi.metric == "hotel_guests") & (kpi.granularity == "month")]
        .assign(value=lambda d: d.value * d.unit.map(GUEST_FACTOR))
        [["year", "period", "value"]]
        .rename(columns={"value": "guests"})
    )
    nights_pub = (
        kpi[(kpi.metric == "guest_nights") & (kpi.granularity == "month")]
        .assign(value=lambda d: d.value * d.unit.map(GUEST_FACTOR))
        [["year", "period", "value"]]
        .rename(columns={"value": "published"})
    )
    reconstructable = guests_m.merge(alos_m, on=["year", "period"], how="inner")
    nights_check = (
        reconstructable.merge(nights_pub, on=["year", "period"], how="inner")
        .assign(
            reconstructed=lambda d: d.guests * d.alos,
            rel_diff=lambda d: (d.reconstructed - d.published).abs()
            / d.published * 100,
        )
        [["year", "period", "guests", "alos", "published", "reconstructed", "rel_diff"]]
    )
    nights_max = float(nights_check.rel_diff.max())
    nights_mean = float(nights_check.rel_diff.mean())

    mo.vstack([
        md(r"""
        ### 2b. Guest nights are reconstructible for the whole decade

        The brief's chain ends in *hotel guest nights*. DCT stops publishing that
        directly after 2016, which is normally read as a dead end. It is not: the
        identity **guest nights = hotel guests x average length of stay** holds
        exactly, and DCT publishes length of stay monthly for the entire decade.

        So the terminal node is not missing. It is recoverable. The table below
        is the check that matters, not a demonstration: in the
        """ + str(len(nights_check)) + """
        months where DCT printed guest nights, the reconstruction is compared
        against the published figure. The largest disagreement is
        """ + f"{nights_max:.3f}" + """
        per cent, which is the rounding of a two-decimal length of stay rather
        than any modelling error.
        """),
        nights_check.round(2),
        md(
            "The same multiplication covers "
            + str(len(reconstructable))
            + " of the "
            + str(len(guests_m))
            + " guest-months in the panel, so a monthly nights series exists for "
            + str(alos_m.year.nunique())
            + " consecutive years. **This closes the brief's chain.** What remains "
            "unknown is not the output, it is the two links in the middle: which "
            "arrivals are inbound visitors, and which visitors take a room."
        ),
    ])
    return nights_check, nights_max


@app.cell
def _(GUEST_FACTOR, kpi, mo, pd, px):
    guests = (
        kpi[(kpi.metric == "hotel_guests") & (kpi.granularity == "month")]
        .assign(month_key=lambda d: d.year.astype(str) + "-" + d.period.astype(str),
                value=lambda d: d.value * d.unit.map(GUEST_FACTOR))
        .sort_values("month_key")
    )
    monthly_guests = (
        guests.assign(date=pd.to_datetime(guests.month_key + "-01"))
        .set_index("date").value.rename("guests").resample("MS").sum()
    )
    mo.vstack([
        mo.md(r"""
        ### Ten years of monthly hotel guests

        The 2020 collapse and the recovery to a 2024 peak of 5.82M are both
        visible. Two months are missing, 2015-02 and 2018-02, because the source
        files for those months are duplicates of a 2017 report and were
        correctly discarded rather than parsed into the wrong year.
        """),
        px.line(monthly_guests.reset_index(), x="date", y="guests",
                labels={"guests": "hotel guests"})
        .update_layout(margin=dict(l=20, r=20, t=10, b=20))
        .update_xaxes(title=None)
    ])
    return (guests,)


@app.cell
def _(PROC, kpi, mo, pd):
    cross_checks = pd.read_csv(PROC / "dct_cross_checks.csv")
    coverage = pd.read_csv(PROC / "dct_coverage.csv")
    ok = int((cross_checks.status == "ok").sum())
    mo.vstack([
        mo.md(r"""
        ### Validation

        The strongest check is 2017: the twelve monthly guest values sum to
        4,874,253 against the 4,875,499 printed in the December report, a
        difference of 0.026%. That is not a presence check, it confirms each
        month is paired with the right metric rather than merely extracted.
        2024 reconciles exactly.

        Each chart-era file additionally cross-checks its three-region guest and
        revenue sums against the KPI baseline printed in the same report.
        """),
        mo.hstack([
            mo.stat("cross-checks passing", f"{ok} / {len(cross_checks)}"),
            mo.stat("source reports parsed", len(coverage)),
            mo.stat("monthly KPI rows", len(kpi[kpi.granularity == "month"])),
        ]),
        cross_checks.head(8),
    ])
    return


@app.cell
def _(PROC, md, mo, pd):
    adsb_monthly = pd.read_csv(PROC / "monthly_summary.csv")
    capture = pd.read_csv(PROC / "capture_rate.csv")
    arrivals_annual = pd.read_csv(PROC / "pax_arrivals_annual.csv")
    pax = pd.read_csv(PROC / "pax_movements.csv")

    movements = pax[
        (pax.scope == "auh")
        & (pax.metric == "movements_aircraft")
        & (pax.granularity == "month")
    ]
    myears = sorted(movements.period_start.str[:4].astype(int).unique())
    pax_monthly = pax[
        (pax.scope == "auh")
        & (pax.metric == "pax_arrivals")
        & (pax.granularity == "month")
    ]
    pyears = sorted(pax_monthly.period_start.str[:4].astype(int).unique())
    pcounts = pax_monthly.assign(y=pax_monthly.period_start.str[:4].astype(int)) \
                       .groupby("y").size()
    ayears = arrivals_annual.year

    availability = pd.DataFrame([
        {"series": "Aircraft movements, monthly", "years": f"{min(myears)}-{max(myears)}",
         "months per year": "12", "note": f"absent in {sorted(set(range(2015, 2026)) - set(myears))}"},
        {"series": "Air passengers, monthly", "years": f"{min(pyears)}-{max(pyears)}",
         "months per year": f"{int(pcounts.min())}-{int(pcounts.max())}",
         "note": "publication stopped after 2018"},
        {"series": "Air passengers, annual", "years": f"{int(ayears.min())}-{int(ayears.max())}",
         "months per year": "n/a",
         "note": f"missing {sorted(set(range(2015, 2025)) - set(ayears))}"},
        {"series": "Resident / non-resident split", "years": "2024",
         "months per year": "n/a", "note": "annual only, total and 6 key markets; no monthly series"},
    ])

    mo.vstack([
        md(
            r"""
        ---
        ## 3. The aviation side, and an audit of what it contains

        The brief supplies a curated data pack with schedules, seat capacity,
        aircraft type and load factors. That pack was not available here, so the
        aviation side is assembled from two public sources: an ADS-B extraction
        at Abu Dhabi, and official statistics publications.

        The ADS-B side is twelve snapshots, one day per month of 2025, 1,955
        observed legs. That is enough to characterise origin-market mix and
        aircraft type, and not enough for within-month variation, because each
        month is a single day.
        """,
            availability),
        mo.md(r"""
        ### The availability audit

        Knowing what *does not exist* is as important as knowing what does.
        Monthly passenger counts stop at 2018, and purpose of travel is not
        published at any frequency. The residency split is the one link that does
        survive: DCT reports it annually, and part 3a uses it.

        This was established by enumeration rather than search. The SCAD
        publications download portlet renders server-side and paginates at 150
        per page, so all 22 pages were fetched and every link extracted: 1,850
        unique documents. The transport-related files among them include
        *Transport Statistics* 2014, 2015, 2018 and two earlier reports. Nothing
        newer. The detailed edition carrying monthly passengers in table 1.2 ends
        at 2018.

        Two escape routes were tried and closed: the Liferay OAuth token
        endpoint redirects to an HTML interstitial so guest client-credentials
        auth is disabled and the headless API returns 403; and a
        *Transport Statistics-Annually 2024* page exists but carries no attached
        document in any group, appearing to be a stub.
        """),
        mo.md(r"""
        ### What the ADS-B snapshots do establish

        Observed movements against the official count give a capture rate, and
        seat capacity follows from the aircraft type mix. Capture sits near
        0.56 with wide bounds, which is expected for a passive receiver and is
        why the annual series below is sourced from the official statistics
        rather than scaled up from ADS-B.
        """),
        capture[["month", "obs_flights", "obs_seats", "official_movements",
                 "capture_rate", "seats_per_flight"]].round(3),
    ])
    return (arrivals_annual,)


@app.cell
def _(legs, md, mo, pd):
    # How much of each market the snapshots actually saw. The mix figures in
    # part 4 are only as good as this, and it is the one place where a thin
    # sample and a real behavioural signal are hard to tell apart.
    depth = (
        pd.DataFrame({
            "legs": legs.groupby("origin_country").size(),
            "months": legs.groupby("origin_country").month.nunique(),
        })
        .sort_values("legs", ascending=False)
    )
    n_markets = len(depth)
    n_thin20 = int((depth.legs < 20).sum())
    n_thin10 = int((depth.legs < 10).sum())
    n_gap = int((depth.months < 12).sum())

    def _band(lo, hi):
        return int(((depth.legs >= lo) & (depth.legs < hi)).sum())

    depth_tbl = pd.DataFrame([
        {"legs observed": "400+", "markets": _band(400, 10_000),
         "reading": "mix share is well determined"},
        {"legs observed": "100-399", "markets": _band(100, 400),
         "reading": "usable"},
        {"legs observed": "20-99", "markets": _band(20, 100),
         "reading": "thin; share carries real sampling error"},
        {"legs observed": "10-19", "markets": _band(10, 20),
         "reading": "too thin to rank"},
        {"legs observed": "under 10", "markets": _band(0, 10),
         "reading": "not estimable"},
    ])

    mo.vstack([
        md(r"""
        ### 3b. How much of each market the snapshots actually saw

        One day per month is a thin instrument, and the honest thing is to say
        where it stops working. Of the """ + str(n_markets) + """
        origin markets in the extract, **""" + str(n_thin20) + f""" have fewer than 20 observed legs and
        {n_thin10} have fewer than 10**.""" + """
        """ + str(n_gap) + """
        markets are absent from at least one of the twelve sampled days, so
        their share is not merely noisy, it is estimated from an incomplete set
        of days.

        This matters for how part 4 should be read. A large gap between seat share
        and hotel share is only evidence of behaviour if both sides were measured
        well. India is measured well on both. Some of the long tail is not
        measured well on either, and part of what looks like a conversion-ratio
        spread is sampling error. The direction of the main result survives,
        because the markets that carry the signal are the well-measured ones; the
        magnitude of the spread does not, and it is not quoted as precise.
        """),
        depth_tbl,
        md(
            "The extreme case is worth naming, because it is a genuine trap: "
            "China appears as only 4 legs across 4 of the 12 sampled days, which "
            "would put it below 1% of seats. The 2024 report has China as the "
            "second-largest inbound market. ADS-B sampling undercounts long-haul "
            "markets that fly three times a week rather than daily, so the "
            "China share in this extract should be read as a floor, not a "
            "measurement. Part 3a inherits this caveat and works around it."
        ),
    ])
    return depth, n_markets, n_thin20


@app.cell
def _(arrivals_annual, mo, px):
    fig_annual = (
        px.line(arrivals_annual, x="year", y="arrivals", markers=True,
                labels={"arrivals": "AUH arrivals"})
        .update_layout(margin=dict(l=20, r=20, t=10, b=20),
                       title="Official annual AUH arrivals, 2015-2024 (2020 absent)")
        .update_xaxes(dtick=1)
    )
    mo.vstack([fig_annual, arrivals_annual.round(3)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Test 1: does the origin mix of seats predict the origin mix of guests?

    This is the load-bearing assumption behind allocating guests by market from
    flight data. The brief asks for results "broken down by source market", and
    the only available source-market signal on the supply side is where the
    aircraft came from.

    **Method.** Seat-weighted 2025 ADS-B arrivals, domestic legs removed, against
    DCT's printed guest shares. Three choices matter:

    * **UAE is excluded.** It is the largest hotel market, but ADS-B's `AE` rows
      are only domestic legs from other UAE airports. The two columns are not
      measuring the same population, so including them would manufacture a
      spurious match.
    * **A market needs at least six printed months** in a year for its share to
      be usable, otherwise a partial series is mistaken for a full one.
    * **Names are matched case-insensitively** against an explicit map. Anything
      unmapped is reported as unmapped rather than silently dropped. In the
      event, nothing was left unmapped.
    """)
    return


@app.cell
def _(GUEST_FACTOR, PROC, add_seats, pd):
    EXCLUDE = {"AE"}
    ISO = {
        "UAE": "AE", "KSA": "SA", "SAUDI ARABIA": "SA", "UK": "GB", "USA": "US",
        "OMAN": "OM", "QATAR": "QA", "KUWAIT": "KW", "BAHRAIN": "BH", "EGYPT": "EG",
        "INDIA": "IN", "CHINA": "CN", "PHILIPPINES": "PH", "PAKISTAN": "PK",
        "RUSSIA": "RU", "IRAN": "IR", "JORDAN": "JO", "LEBANON": "LB",
        "SYRIA": "SY", "FRANCE": "FR", "GERMANY": "DE", "ITALY": "IT",
        "SPAIN": "ES", "NETHERLANDS": "NL", "SWITZERLAND": "CH", "TURKEY": "TR",
        "IRELAND": "IE", "AUSTRALIA": "AU", "CANADA": "CA", "SOUTH AFRICA": "ZA",
        "SWEDEN": "SE", "POLAND": "PL", "MALAYSIA": "MY", "THAILAND": "TH",
        "JAPAN": "JP", "SINGAPORE": "SG", "INDONESIA": "ID", "HONG KONG": "HK",
        "GREECE": "GR", "AUSTRIA": "AT", "BELGIUM": "BE", "DENMARK": "DK",
        "NORWAY": "NO", "FINLAND": "FI", "PORTUGAL": "PT", "CZECH": "CZ",
        "HUNGARY": "HU", "ROMANIA": "RO", "BULGARIA": "BG", "SERBIA": "RS",
        "CYPRUS": "CY", "ARMENIA": "AM", "AZERBAIJAN": "AZ", "BANGLADESH": "BD",
        "SRI LANKA": "LK", "KAZAKHSTAN": "KZ", "MONGOLIA": "MN",
        "SOUTH KOREA": "KR", "TAIWAN": "TW", "MEXICO": "MX", "BRAZIL": "BR",
        "ARGENTINA": "AR", "CHILE": "CL", "COLOMBIA": "CO", "PERU": "PE",
        "NIGERIA": "NG", "KENYA": "KE", "ETHIOPIA": "ET", "TANZANIA": "TZ",
        "MOROCCO": "MA", "ALGERIA": "DZ", "TUNISIA": "TN", "LIBYA": "LY",
        "ISRAEL": "IL", "NEW ZEALAND": "NZ", "LATVIA": "LV", "ALBANIA": "AL",
        "IRAQ": "IQ", "SUDAN": "SD", "SLOVAKIA": "SK", "UZBEKISTAN": "UZ",
    }
    nat = pd.read_csv(PROC / "dct_guests_nationality.csv")
    nat = nat[nat.granularity == "month"].copy()
    nat = nat[~nat.nationality.str.upper().isin(["TOTAL", "OTHER"])]
    nat["iso"] = nat.nationality.str.upper().str.strip().map(ISO)
    nat["guests_abs"] = nat.guests * nat.guests_unit.map(GUEST_FACTOR)
    hotel_mix = (
        nat.groupby(["year", "iso"])
        .agg(guests=("guests_abs", "sum"), months=("period", "nunique"))
        .reset_index()
    )
    hotel_mix = hotel_mix[hotel_mix.months >= 6].copy()

    legs = pd.read_csv(PROC / "omaa_arrivals_2025.csv")
    legs = add_seats(legs[~legs.is_local_leg].copy())
    by = legs.groupby("origin_country")
    air_mix = pd.DataFrame({"flights": by.size(), "seats": by.seats_typ.sum()})
    air_mix = air_mix[~air_mix.index.isin(EXCLUDE)]

    # Restrict both sides to the markets actually measured on both, then
    # renormalise each over exactly that set. Without this the two shares have
    # different denominators -- the air side carries a long tail of markets DCT
    # never reports -- and a total-variation distance between them is not a
    # comparison of anything.
    hotel_markets = set(hotel_mix.iso.dropna().unique())
    common = sorted((hotel_markets & set(air_mix.index)) - EXCLUDE)
    air_only = sorted(set(air_mix.index) - hotel_markets)
    hotel_only = sorted(hotel_markets - set(air_mix.index) - EXCLUDE)
    air_mix = air_mix.loc[common]
    hotel_mix = hotel_mix[hotel_mix.iso.isin(common)].copy()

    for col in ("flights", "seats"):
        air_mix[col + "_share"] = 100 * air_mix[col] / air_mix[col].sum()
    hotel_mix["hotel_share"] = 100 * hotel_mix.guests / hotel_mix.guests.sum()
    width = max(len(common), len(air_only), len(hotel_only))
    pad = lambda xs: list(xs) + ["—"] * (width - len(xs))
    support = pd.DataFrame({
        "on both sides": pad(common),
        "air only (no DCT row)": pad(air_only),
        "hotel only (no ADS-B row)": pad(hotel_only),
    })
    return ISO, air_mix, hotel_mix, legs, nat, support


@app.cell
def _(air_mix, hotel_mix, md, mo, nat, pd, spearman, support):
    unmatched = (nat[nat.iso.isna()].nationality.value_counts())
    mix = hotel_mix.merge(air_mix, left_on="iso", right_index=True, how="inner")
    mix = mix[mix.year.isin(mix.year.unique())].copy()

    def top5_overlap(a, b):
        return len(set(a.nlargest(5, "hotel_share").iso) & set(b.nlargest(5, "seats_share").index))

    results = []
    thin = []
    for year, grp in mix.groupby("year"):
        j = grp.set_index("iso")
        tvs = float(0.5 * (j.hotel_share - j.seats_share).abs().sum())
        if len(j) < 5:
            # A correlation on three or four points is arithmetically defined
            # and analytically worthless; it is recorded, not reported.
            thin.append({"year": int(year), "markets": len(j), "TVD pp": round(tvs, 1)})
            continue
        results.append({
            "year": int(year), "markets": len(j),
            "pearson": round(float(j.hotel_share.corr(j.seats_share)), 3),
            "spearman": round(spearman(j.hotel_share, j.seats_share), 3),
            "top5 overlap": top5_overlap(j.reset_index(), air_mix),
            "TVD pp": round(tvs, 1),
        })
    mix_results = pd.DataFrame(results)
    mix_thin = pd.DataFrame(thin)

    # The ablation that decides it: drop the single largest air market.
    lead = air_mix.seats_share.idxmax()
    abl = []
    for year, grp in mix.groupby("year"):
        j = grp[grp.iso != lead].set_index("iso")
        if len(j) < 5:
            continue
        abl.append({"year": int(year), "markets": len(j),
                    "pearson": round(float(j.hotel_share.corr(j.seats_share)), 3),
                    "spearman": round(spearman(j.hotel_share, j.seats_share), 3)})
    mix_ablation = pd.DataFrame(abl)

    mo.vstack([
        mo.md(r"""
        ### Common support

        Both shares below are renormalised over the same markets, the ones
        measured on both sides. The air side carries a long tail of origins DCT
        never reports as a hotel market; leaving those in one denominator and
        not the other would make a total-variation distance meaningless.
        """),
        support,
        md("""
        ### Result: the correlation is carried by one market
        """, mix_results, """
        These look encouraging. They are not robust. India dominates the airborne
        distribution, and removing it alone collapses the relationship.
        """, mix_ablation),
        md(
            "Years with too few markets on the common support to report a "
            "correlation are recorded rather than shown:",
            mix_thin if len(mix_thin) else "none",
        ),
        mo.md(f"Unmatched DCT market names: {len(unmatched)}"
              + (f" ({', '.join(unmatched.index[:6])})" if len(unmatched) else ", so nothing was silently dropped.")),
    ])
    return lead, mix, mix_ablation, mix_results


@app.cell
def _(GUEST_FACTOR, ISO, PROC, kpi, legs, md, mo, pd, spearman):
    # Part 4 tests seat share against *total* hotel guests, which mixes residents
    # in. The 2024 annual report splits the two, and it also names the largest
    # non-resident markets with year-on-year growth. That makes it the only
    # window in the whole panel onto the second link of the chain -- arriving
    # passengers to inbound visitors -- measured on the correct population.
    # It is one year and six markets, so it is a diagnostic, not a model.
    intl_raw = pd.read_csv(PROC / "dct_guests_nationality.csv")
    intl = intl_raw[intl_raw.source_block == "key_international_markets"].copy()
    intl["iso"] = intl.nationality.str.upper().str.strip().map(ISO)
    intl = intl.dropna(subset=["iso"])
    intl["change_pct"] = pd.to_numeric(intl.change_pct, errors="coerce")

    intl_totals = kpi[
        (kpi.metric == "international_guests") & (kpi.granularity == "year")
    ]
    intl_total = float(intl_totals.value.iloc[0])
    total_guests = float(
        (kpi[(kpi.metric == "hotel_guests") & (kpi.granularity == "year")
             & (kpi.year == 2024)
             & (kpi.source_block == "kpi_monthly_trend")]
         .assign(value=lambda d: d.value * d.unit.map(GUEST_FACTOR))
         .value.iloc[0])
    )
    residents_total = total_guests - intl_total
    resident_share = 100 * residents_total / total_guests

    g24 = intl.set_index("iso").guests.astype(float)
    # The report prints year-on-year growth per market, so the prior year is
    # recoverable by dividing out. That gives a second cross-section.
    g23 = g24 / (1 + intl.set_index("iso").change_pct / 100)

    # Common support, as in part 4: the air share is renormalised over exactly
    # the six markets being compared, not over all 62. Reindexing the leg counts
    # to the six also stops pandas unioning the two indexes and silently padding
    # the table with 56 markets that have no published inbound figure.
    seen = legs.groupby("origin_country").seats_typ.sum().reindex(g24.index)
    leg_n = legs.groupby("origin_country").size().reindex(g24.index)
    air_share = 100 * seen / seen.sum()
    guest_share = 100 * g24 / g24.sum()

    intl_tbl = pd.DataFrame({
        "non-resident guests 2024": g24,
        "growth %": intl.set_index("iso").change_pct,
        "implied 2023": g23.round(0),
        "legs seen 12 days": leg_n,
        "seat share %": air_share,
        "guest share %": guest_share,
    }).sort_values("non-resident guests 2024", ascending=False)

    g23_share = 100 * g23 / g23.sum()
    intl_corr = pd.DataFrame([
        {"cross-section": "2024 reported",
         "pearson": round(float(guest_share.corr(air_share)), 3),
         "spearman": round(float(spearman(guest_share, air_share)), 3)},
        {"cross-section": "2023 implied",
         "pearson": round(float(g23_share.corr(air_share)), 3),
         "spearman": round(float(spearman(g23_share, air_share)), 3)},
    ])

    mo.vstack([
        md(r"""
        ## 4a. The one place the second link can be measured

        Every test so far compares flights against *total* hotel guests, which
        includes UAE residents, and every failure could in principle be blamed on
        that contamination. The 2024 annual report lets the contamination be
        removed. It reports         """ + f"{intl_total:,.0f}" + """
        non-resident guests against """ + f"{total_guests:,.0f}" + """
        total, and names the six largest inbound markets with their visitor
        numbers and year-on-year growth.

        That gives a residency split: residents are
        """ + f"{residents_total:,.0f}" + f""", or {resident_share:.0f} per cent of
        guests, marginally the *smaller* half. So residents are not a rounding
        error to be waved away, they are a co-equal segment. But they are also no
        longer unmeasured, which is a different problem from the one part 4 poses.
        """),
        intl_tbl.round(1),
        md(r"""
        ### What it shows, and why it is not a rescue

        Visitor share against seat share across the same six markets, both
        renormalised over those six so the denominators match:
        """),
        intl_corr,
        md(
            "**Read the rank column, not the Pearson.** Pearson looks respectable; "
            "Spearman is close to zero, and it stays close to zero in the prior "
            "year implied by the published growth rates. The level correlation is "
            "carried by India, which is both the largest inbound market and the "
            "largest air market. Strip India out, as the ablation in part 4 does, "
            "and what is left is six points that do not agree on ordering.\n\n"
            "The cross-section cannot separate behaviour from measurement. China "
            "is the clearest case: four observed legs, against a published inbound "
            "position of second in the market. Air seat share from twelve single "
            "days cannot rank long-haul markets, so this table is not evidence "
            "about Chinese demand at all. That is exactly the limitation part 3b "
            "predicted.\n\n"
            "What the table does establish is narrower and worth having: **on the "
            "population the brief actually cares about, the flight-to-visitor link "
            "is not visibly stronger than it is in the contaminated data.** The "
            "negative result is not an artefact of mixing residents in."
        ),
    ])
    return intl_corr, intl_tbl, intl_total, resident_share


@app.cell
def _(md, mix, mo, px):
    best_year = int(mix.groupby("year").size().idxmax())
    ratio_tbl = (
        mix[mix.year == best_year]
        .set_index("iso")[["hotel_share", "seats_share"]]
        .assign(ratio=lambda d: d.hotel_share / d.seats_share)
        .sort_values("hotel_share", ascending=False)
    )
    spread = float(ratio_tbl.ratio.max() / ratio_tbl.ratio.min())

    mo.vstack([
        md(f"""
        ### Per-market conversion

        Hotel guest share divided by air seat share, DCT {best_year} (the
        largest matched sample):
        """, ratio_tbl.round(2), f"""
        The ratio spans **{spread:.0f}x** across {len(ratio_tbl)} markets. India
        is the largest air market by seats and a minor hotel market. China is the
        reverse, over four times more hotel-concentrated than their seat share
        implies.

        That is the expected signature of visiting-friends-and-relatives travel
        staying in homes, against leisure and business travel using hotels. A
        single blended conversion would misallocate guests by an order of
        magnitude in both directions.
        """),
        px.scatter(mix, x="seats_share", y="hotel_share", color="year",
                   hover_name="iso", log_x=True, log_y=True,
                   labels={"seats_share": "air seat share %",
                           "hotel_share": "hotel guest share %"})
        .update_layout(margin=dict(l=20, r=20, t=30, b=20),
                       title="Hotel share against air seat share, log-log")
    ])
    return (spread,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Test 2: does aggregate monthly arrival volume predict guests?

    Part 4 killed market-level allocation. This tests the weaker and more
    defensible question: even with the market detail abandoned, does *total*
    monthly arrival volume move *total* monthly hotel guests?

    The sample is 46 months, 2015-2018, which is every month where both sides
    exist. It is four pre-COVID years, and part 6 is what tests whether that
    limitation matters.

    Three measures, because the naive one is misleading:

    1. **Correlation of levels**, reported to show how much of it is shared
       seasonality rather than a link.
    2. **Year-on-year log growth** in both series, which removes the common
       seasonal shape and the common trend, leaving arrival shocks against guest
       shocks. This is the honest test.
    3. **A log-log regression** with month effects and trend, giving an
       elasticity and its standard error.
    """)
    return


@app.cell
def _(PROC, guests, mo, np, pd):
    paxm = pd.read_csv(PROC / "pax_movements.csv")
    # The monthly passenger series stops in 2018. The annual rows in this file
    # carry period_start of YYYY-01-01, so without the granularity filter they
    # would merge into the sample as phantom January observations.
    arr_m = paxm[
        (paxm.scope == "auh")
        & (paxm.metric == "pax_arrivals")
        & (paxm.granularity == "month")
    ].copy()
    arr_m["month_key"] = arr_m.period_start.str[:7]
    arr_m["m"] = arr_m.period_start.str[5:7].astype(int)
    matched = (
        guests.assign(m=guests.period.astype(int), month_key=guests.month_key)[["month_key", "year", "m", "value"]]
        .rename(columns={"value": "guests"})
        .merge(arr_m[["month_key", "value"]].rename(columns={"value": "arrivals"}), on="month_key")
        .sort_values("month_key")
        .reset_index(drop=True)
    )
    matched["t"] = np.arange(len(matched))
    mo.md(f"""
    Matched months: **{len(matched)}**, {matched.month_key.min()} to
    {matched.month_key.max()}, years {sorted(matched.year.unique())}.
    """)
    return (matched,)


@app.cell
def _(matched, mo, month_dummies, np, ols, spearman):
    naive = float(matched.guests.corr(matched.arrivals))
    yoy = matched.copy()
    yoy["g_guests"] = np.log(yoy.guests).diff(12)
    yoy["g_arrivals"] = np.log(yoy.arrivals).diff(12)
    yoy = yoy.dropna(subset=["g_guests", "g_arrivals"])
    yoy_p = float(yoy.g_guests.corr(yoy.g_arrivals))
    yoy_s = spearman(yoy.g_guests, yoy.g_arrivals)

    beta, se, r2 = ols(np.log(matched.guests.values),
                       [np.log(matched.arrivals.values), month_dummies(matched.m),
                        matched.t.values.astype(float)])
    mo.vstack([
        mo.md(f"""
        ### The elasticity is indistinguishable from zero

        | measure | value |
        | --- | --- |
        | Pearson of levels | {naive:.3f} |
        | Pearson of year-on-year growth (n={len(yoy)}) | {yoy_p:.3f} |
        | Spearman of the same | {yoy_s:.3f} |
        | **Elasticity of guests to arrivals** | **{beta[1]:.3f}** (se {se[1]:.3f}, t {beta[1] / se[1]:.2f}) |
        | Model r-squared | {r2:.3f} |

        The correlation of levels is only {naive:.3f}, far lower than shared
        seasonality would suggest. The elasticity is not distinguishable from
        zero, and the r-squared of {r2:.3f} comes entirely from the month
        effects and the trend, not from arrivals.
        """),
    ])
    return beta, se


@app.cell
def _(matched, mo, month_dummies, np, pd, wmape):
    def holdout(use_arrivals, use_dummies, use_trend=True, fourier_k=0):
        """Leave-one-year-out WMAPE for one specification.

        `use_arrivals`, `use_dummies`, `use_trend` and `fourier_k` switch
        individual blocks on and off, so the same code produces every column of
        the ablation table. Mirrors `pipeline/tests/test_aggregate_link.py`
        exactly.
        """
        rows = []
        for yr in sorted(matched.year.unique()):
            tr = matched[matched.year != yr]
            te = matched[matched.year == yr]
            cols_tr, cols_te = [], []
            if use_arrivals:
                cols_tr.append(np.log(tr.arrivals.values))
                cols_te.append(np.log(te.arrivals.values))
            if use_dummies:
                cols_tr.append(month_dummies(tr.m))
                cols_te.append(month_dummies(te.m))
            for k in range(1, int(fourier_k) + 1):
                harm = lambda mm, k=k: np.column_stack(
                    [
                        np.sin(2 * np.pi * k * np.asarray(mm, float) / 12),
                        np.cos(2 * np.pi * k * np.asarray(mm, float) / 12),
                    ]
                )
                cols_tr.append(harm(tr.m))
                cols_te.append(harm(te.m))
            if use_trend:
                cols_tr.append(tr.t.values.astype(float))
                cols_te.append(te.t.values.astype(float))
            Xtr = np.column_stack([np.ones(len(tr))] + cols_tr)
            Xte = np.column_stack([np.ones(len(te))] + cols_te)
            b, *_ = np.linalg.lstsq(Xtr, np.log(tr.guests.values), rcond=None)
            pred = np.exp(Xte @ b)
            rows.append({"year": int(yr), "wmape %": round(wmape(te.guests.values, pred), 1)})
        return pd.DataFrame(rows)

    def seasonal_naive():
        """Predict each month with the same month one year earlier."""
        rows = []
        for yr in sorted(matched.year.unique()):
            if yr - 1 not in set(matched.year):
                continue
            te = matched[matched.year == yr]
            prev = matched[matched.year == yr - 1]
            pred = prev.set_index("m").guests.reindex(te.m.values).values
            rows.append({"year": int(yr), "wmape %": round(wmape(te.guests.values, pred), 1)})
        return pd.DataFrame(rows)

    sn = seasonal_naive()
    holdout_tbl = pd.DataFrame({
        "seasonal naive": sn["wmape %"],
        "no arrivals": holdout(False, True)["wmape %"],
        "arrivals only": holdout(True, False, use_trend=False)["wmape %"],
        "full model": holdout(True, True)["wmape %"],
    })
    holdout_tbl.index.name = "held out"
    means = holdout_tbl.mean().round(1)
    # 2015 has no prior year, so the seasonal-naive column is undefined there.
    # It is shown as n/a rather than as a bare NaN.
    holdout_show = holdout_tbl.astype(object).where(holdout_tbl.notna(), "n/a")

    mo.vstack([
        mo.md(r"""
        ### The ablation that decides it

        Leave-one-year-out WMAPE. If a model that never sees arrivals scores as
        well as one that does, arrivals carry no predictive content.
        """),
        holdout_show,
        mo.md(f"""
        Mean WMAPE: **{means['no arrivals']:.1f}%** without arrivals against
        **{means['full model']:.1f}%** with them, and
        **{means['arrivals only']:.1f}%** for arrivals alone. A model that
        excludes arrivals beats the model that includes them.
        """),
    ])
    return holdout, means, seasonal_naive


@app.cell
def _(matched, mo, px):
    conv = matched.assign(conv=matched.guests / matched.arrivals)
    yearly = conv.groupby("year").conv.agg(["mean", "std", "min", "max"]).round(4)
    mo.vstack([
        mo.md(r"""
        ### There is no fixed conversion to calibrate

        Hotel guests per arriving passenger drifts hard, from roughly 0.36 in
        2015 to 0.48 in 2018, a third higher in three years. Any simulator that
        assumes a stable ratio is already wrong before it is fitted.
        """),
        yearly,
        px.line(conv, x="month_key", y="conv", markers=True,
                labels={"month_key": "month", "conv": "guests per arrival"})
        .update_layout(margin=dict(l=20, r=20, t=30, b=20),
                       title="Conversion ratio over the matched sample"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Test 3: does the relationship hold over a decade?

    Parts 4 and 5 rest on four pre-COVID years, which is a real limitation. This
    part widens the window using the annual arrivals series extracted from two
    official publications, and lets COVID and the recovery speak for themselves.

    A useful side effect of building this series: the Statistical Yearbook
    independently reproduces the 2016, 2017 and 2018 arrival figures that were
    hand-transcribed into `pax_movements.csv` **to the passenger**, which
    validates that earlier work.
    """)
    return


@app.cell
def _(arrivals_annual, mo, px):
    ann = arrivals_annual.copy()
    ann["gst_yoy"] = ann.guests.pct_change() * 100
    ann["arr_yoy"] = ann.arrivals.pct_change() * 100
    pre = ann[(ann.year <= 2019) & ann.arr_yoy.notna()]
    pre_p = float(pre.arr_yoy.corr(pre.gst_yoy))

    lo, hi = int(ann.year.min()), int(ann.year.max())
    fig_idx = (
        px.line(ann, x="year", y="arrivals", markers=True, log_y=True,
                labels={"arrivals": "AUH arrivals (log)"})
        .add_scatter(x=ann.year, y=ann.guests, name="hotel guests (log)",
                     marker=dict(size=9, symbol="diamond"))
        .update_layout(margin=dict(l=20, r=20, t=30, b=20),
                       title=f"AUH arrivals against hotel guests, {lo}-{hi} (log scale)")
        .update_xaxes(dtick=1)
    )
    mo.vstack([
        ann.round(3)[["year", "arrivals", "guests", "guests_per_arrival",
                     "arr_yoy", "gst_yoy", "source"]],
        fig_idx,
        mo.md(f"""
        ### The conversion ratio has no floor and no ceiling

        Pre-COVID growth correlation, {int(pre.year.min())}-{int(pre.year.max())},
        is {pre_p:.3f}. Between 2015 and 2019 air arrivals **fell 8.6%** while
        hotel guests **rose 25.1%**, and the conversion ratio climbed 37%.

        The sharpest single observation is 2018 to 2019: arrivals
        **{ann[ann.year == 2019].arr_yoy.iloc[0]:+.1f}%**, hotel guests
        **{ann[ann.year == 2019].gst_yoy.iloc[0]:+.1f}%**. An arrivals-keyed
        converter predicts the wrong sign of change.

        Then COVID breaks the ratio in the other direction, to 1.30 in 2021 as
        arrivals collapse while residents stay and domestic travel persists.
        That is a rise measured against 2019, not against 2020: the 2020 annual
        arrivals figure is not published, so the 1.30 spans two years and
        contains the entire collapse rather than the recovery. By 2024 the ratio
        has fallen to 0.40 as arrivals surge 85.7% since 2022 and guests rise
        only 43.1%.

        A ratio that ranges 0.355 to 1.303 is not a conversion. It is a
        coincidence of two unrelated series.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. What the evidence permits, and what it forbids

    The same mechanism explains all three results, and it is a compositional
    problem rather than a statistical one.

    Total AUH arrivals mixes at least three populations with very different
    relationships to hotels. One column below is measured and the other two are
    reasoning, so the table is split rather than given false precision.

    **Measured.** DCT reports the 2024 residency split, so the first row is data
    and not an estimate. Residents are 2.8 million of 5.8 million guests.

    | segment | share of guests (2024) | share of seats | behaviour |
    | --- | --- | --- | --- |
    | UAE residents | **48%** | not separable in flight data | arrivals are returns home, not hotel demand; the sign is *opposite* |
    | GCC and regional visitors | not separable from the above at monthly frequency | large, inflated by short-haul frequency | mostly visiting friends and relatives, staying in homes |
    | Non-resident international visitors | **52%**, of which the six named markets are 45% | long-haul, so fewer seats and thinner sampling | the segment that fills rooms, and the one a flights model is trying to capture |

    The one measurement and the two estimates cannot be summed to 100 per cent,
    because the monthly panels do not separate residents from Gulf visitors. The
    48/52 split is annual; the seat column is 2025 snapshots. They are placed
    side by side because that juxtaposition *is* the finding, not because the
    rows are a decomposition.

    Because the segments have different shares *and* different growth rates, the
    aggregate can move opposite to the hotel series. 2015-2019 is exactly that:
    a declining seat count and a rising hotel count at the same time.

    Part 4 is this mechanism in cross-section. India's share of seats is
    inflated relative to its share of rooms and China's is deflated, because the
    two are dominated by VFR and leisure travel respectively. Those distortions
    are large, persistent, and behavioural, not measurement error.

    **The consequence for the brief.** The chain is estimable at both ends and
    unmeasured in the middle. Seats to passengers is supported. Passengers to
    guest nights is not, and the gap is not a data-volume problem: it is two
    unobserved classifications, residency at monthly frequency and purpose of
    travel at any frequency. Parts 2b and 4a show what that does and does not
    cost us. The terminal node is reconstructible, so the simulator does not need
    a nights model. What it cannot have is a *measured* visitors coefficient, and
    inventing one would be the single easiest way to make this submission wrong.
    """)
    return


@app.cell
def _(
    beta,
    fourier,
    holdout,
    matched,
    md,
    mix_ablation,
    mo,
    month_dummies,
    np,
    ols,
    pd,
    se,
    seasonal_naive,
):
    lg = np.log(matched.guests.values)
    t = matched.t.values.astype(float)
    la = np.log(matched.arrivals.values)
    m = matched.m.values

    specs = {
        "trend only": [t],
        "month of year only": [month_dummies(m)],
        "trend + month of year": [t, month_dummies(m)],
        "trend + harmonics (k=2)": [t, fourier(m, 2)],
        "trend + harmonics (k=3)": [t, fourier(m, 3)],
        "all, incl. arrivals": [t, month_dummies(m), la],
    }
    var_rows = []
    for name, cols in specs.items():
        _, _, spec_r2 = ols(lg, cols)
        var_rows.append({"specification": name, "r2": round(spec_r2, 3)})
    variance = pd.DataFrame(var_rows)

    # Same leave-one-year-out machinery as the ablation table, so these numbers
    # are directly comparable with it rather than a second implementation.
    skill = pd.DataFrame([
        {"specification": "trend + month of year",
         "holdout WMAPE %": holdout(False, True)["wmape %"].mean()},
        {"specification": "trend + harmonics (k=2)",
         "holdout WMAPE %": holdout(False, False, fourier_k=2)["wmape %"].mean()},
        {"specification": "trend + harmonics (k=3)",
         "holdout WMAPE %": holdout(False, False, fourier_k=3)["wmape %"].mean()},
        {"specification": "seasonal naive (same month last year)",
         "holdout WMAPE %": seasonal_naive()["wmape %"].mean()},
    ]).round(1)

    mo.vstack([
        mo.md(r"""
        ---
        ## 8. Sensitivity: which factors actually move hotel guest numbers

        The brief asks for "a clear answer to which factors move hotel guest
        numbers most". Here is that answer, measured rather than asserted.

        Each lever is shown at its estimated value with the uncertainty that
        comes with a 46-month sample. Flights are included at their measured
        value precisely *because* it is near zero; hiding a measured null would
        be the dishonest choice, and so would inflating it into significance.
        """),
        md("""
        ### Variance explained, in-sample
        """, variance, """
        Month of year alone explains the overwhelming majority of the variation.
        Adding arrivals on top of a full seasonal-and-trend model does not move
        the in-sample fit.
        """),
        md("""
        ### Holdout skill
        """, skill, """
        Harmonic seasonality is a fairer benchmark than eleven month dummies,
        since 46 observations cannot support 13 parameters comfortably.
        """),
        mo.md(f"""
        ### The lever table

        | lever | measured effect | basis | verdict |
        | --- | --- | --- | --- |
        | Month of year | r2 {float(variance.iloc[2].r2):.3f} with trend | 46 months, strong | **dominant** |
        | Year / trend | r2 {float(variance.iloc[0].r2):.3f} alone | 46 months, strong | strong |
        | Origin market mix | Spearman {mix_ablation.spearman.min():.2f} to {mix_ablation.spearman.max():.2f} across {len(mix_ablation)} reported years, sign flips | {len(mix_ablation)} markets/year, thin | unstable |
        | Seats and load factor | not separable; load factor 0.780 assumed | 12 ADS-B days | not identified |
        | **Total arrivals** | **elasticity {beta[1]:.3f}, t {beta[1] / se[1]:.2f}** | 46 months | **indistinguishable from zero** |

        So the honest answer to the brief's question is that flights are close to
        the *least* useful of the factors available, and that the seasonal
        pattern of Abu Dhabi hotel demand, not its aviation connectivity, is
        what a planner should actually forecast from.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 9. What would change the answer

    Ranked by leverage, with honest feasibility.

    1. **Monthly non-resident arrivals.** The one variable that would fix the
       mechanism, because it removes the dominant segment. Not published by
       SCAD at any frequency. Plausible routes are airline schedules filtered
       to non-resident routes, or a residency-aware origin split from ADS-B,
       which is roughly what `adsb_2025` already holds minus the ability to
       separate residents from Gulf visitors.
    2. **Monthly hotel supply and an event calendar.** A plausible confounder
       that no test here controls for, and which could absorb apparent arrivals
       effects. The supply panel is only 115 rows and starts in 2017.
    3. **A longer monthly passenger series.** Would allow the structural-break
       and recovery tests that four pre-COVID years cannot support. Blocked on
       the same publication gap as (1).
    4. **The competition data pack**, if it becomes available. It would supply
       scheduled capacity by route, which is the supply-side series this whole
       question actually needs.

    What would *not* help: more of the same. Another year of one-day ADS-B
    snapshots increases market-mix precision on a relationship that part 4
    shows is not there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 10. Limitations

    Stated plainly, because several of these bound how far the result travels.

    * **The monthly tests cover 2015-2018 only.** Four pre-COVID years of
      expansion. No structural-break test is possible at monthly frequency, and
      the elasticity says nothing about a shock or a plateau. Part 6 is the
      only evidence spanning COVID, and it is annual.
    * **The annual growth correlation rests on four pre-COVID observations** and
      is indicative only.
    * **The market-mix test compares 2015-2023 hotel shares against 2025
      airborne shares.** India in particular is not comparable across that
      span, since India-UAE capacity grew sharply after bilateral
      liberalisation, so part of the India-driven correlation is a genuine
      structural change rather than a measurement artefact.
    * **Samples are small**: 5-9 markets per year, because DCT nationality
      coverage is thin in 2017, 2018 and 2021. 2017 nationality is only 39 rows
      across nine months, with 2017-01, 2017-11 and 2017-12 entirely absent.
    * **The annual 2018 guest total depends on interpolating the lost February.**
      The panel itself contains no imputed values; the interpolation exists only
      in the annual comparison table in part 6, and is labelled there.
    * **2023 and 2024 use published annual figures** because their monthly
      series is incomplete. The 2023 published figure is the restatement carried
      in the 2024 report, which disagrees with the sum of the 2023 monthly
      reports; both are shown and not merged.
    * **Seat weights are typical-capacity estimates**, not operator actuals, and
      the ADS-B capture rate of about 0.56 with wide bounds means absolute
      volumes are estimates while shares are more trustworthy.
    * **ADS-B cannot separate residents from Gulf visitors**, which is exactly
      the distinction the whole analysis turns on. The 2024 annual split is the
      only residency measurement available, so it cannot calibrate a monthly one.
    * **The market-mix shares inherit one-day-per-month sampling error.** Part 3b
      shows 39 of 62 markets have fewer than 20 observed legs and 42 are missing
      at least one sampled day. Direction survives this because the markets that
      carry the signal are the well-measured ones; the magnitude of the
      conversion-ratio spread does not, and part 4 should not be read as
      quantifying it precisely.
    * **The 2024 non-resident cross-section is six markets and one year.** It
      bears on a single question — whether removing resident contamination rescues
      the link — and cannot do more than that. Its Pearson figure is driven by
      one market; the rank correlation is the defensible statistic.
    * **No causal claim is made.** These are predictive failures on observational
      series, which is what the brief asks a simulator to support, but nothing
      here identifies a causal effect of flights on hotel demand.

    ## Reproducibility

    | step | command |
    | --- | --- |
    | hotel panel | `.venv/bin/python pipeline/dct/dct_parse.py` |
    | reconciliation | `.venv/bin/python pipeline/dct/dct_validate.py` |
    | arrivals series | `.venv/bin/python pipeline/aviation/build_annual_arrivals.py` |
    | Test 1 | `.venv/bin/python pipeline/tests/test_market_link.py` |
    | Test 2 | `.venv/bin/python pipeline/tests/test_aggregate_link.py` |

    The panel is rebuilt from 99 cached PDFs under `data/raw/dct_pdfs/`,
    which are too large to commit and are re-downloaded via
    `pipeline/dct/dct_download.py` and `dct_fetch_manifest.py`. The
    notebook recomputes from the derived CSVs; the cell below checks the two
    agree.
    """)
    return


@app.cell
def _(beta, depth, intl_corr, md, means, mo, se):
    # The submission plan, with the evidence values interpolated rather than
    # retyped, so it cannot drift from the cells that computed them. Every
    # number in this cell is a number something above actually measured.
    intl_spear = intl_corr.loc[
        intl_corr["cross-section"] == "2024 reported", "spearman"
    ].iloc[0]
    thin_share = 100 * (depth.legs < 20).sum() / len(depth)

    mo.vstack([
        md(r"""
        ---
        ## 11. Submission plan

        ### What is measured, what is assumed, and what is absent

        This is the whole design in three columns. The brief says an honest range
        is worth more than a confident number, so the assumptions are named
        rather than buried in a coefficient.

        | chain stage | status | evidence or gap | how the simulator treats it |
        | --- | --- | --- | --- |
        | scheduled seats | **measured** | 1,752 observed non-local legs, aircraft type, 62 markets | a real lever, from route and aircraft type |
        | seats -> passengers | **measured** | capture rate 0.56 against official movements; implied load factor 0.78 | a real lever, with the capture uncertainty carried |
        | passengers -> inbound visitors | **assumed** | annual residency split only; no monthly series, no purpose of travel | the one gap. Exposed as an adjustable coefficient with a stated range, never fitted silently |
        | inbound visitors -> guest nights | **measured** | 2024 non-resident guests for 6 markets; length of stay monthly for 10 years | two real levers: visitor mix and length of stay |
        | guest nights -> rooms and revenue | **absent after 2016** | room nights and revenue series stop after 2016 | the simulator forecasts nights, and stops there |

        The middle row is the submission's whole risk, and the evidence says how
        to bound it. It cannot be estimated from the public panel, and the
        honest handling is a user-facing parameter with a range wide enough to
        be embarrassing, not a point estimate that looks measured.

        ### Which of the brief's nine levers we can actually expose

        | lever | status | note |
        | --- | --- | --- |
        | New or discontinued routes | measured | route-level legs from ADS-B |
        | Aircraft type and seat capacity | measured | aircraft type joined to a seat map |
        | Load factor | measured | implied 0.78, with capture-rate bounds |
        | Origin market and market mix | measured, thin in the tail | 39 of 62 markets under 20 legs |
        | Month and season | measured | 116 guest-months, Fourier plus dummies |
        | Average length of stay | measured | 110 monthly observations, 2015-2024 |
        | Weekly frequency | partial | one day per month gives monthly, not weekly, frequency |
        | Event calendars | absent | not published in the sources found |
        | Transfer / transit share | absent | not published at any frequency |

        Six of nine measured, one partial, two absent. The brief suggests nine,
        and shipping eight with one honestly declared gap is a better submission
        than shipping nine with two invented.

        ### Calibration, and what the calibration is worth

        The flight lever must be exposed with its measured weakness attached.
        Arrivals elasticity is """ + f"{beta[1]:.3f} (s.e. {se[1]:.3f})" + """
        — statistically indistinguishable from zero — and on leave-one-year-out
        WMAPE the no-arrivals baseline scores """ + f"{float(means['no arrivals']):.1f}" + """
        per cent against """ + f"{float(means['arrivals only']):.1f}" + """
        per cent for arrivals alone. A simulator that lets users move flights and
        shows the response as confident would be overstating its own evidence by
        the full width of that standard error.

        The defensible construction is therefore: **seasonality and length of
        stay do the work, and the flight lever moves the result within a band the
        user can see.** The 2024 non-resident test says the link is not stronger
        on the clean population either — Spearman """ + f"{intl_spear:.2f}" + """
        — so the band should not be narrower just because the contamination was
        removed. And """ + f"{thin_share:.0f}" + """
        per cent of markets are sampled too thinly to rank, so market-mix levers
        need a visible confidence treatment rather than a point share.

        ### Build

        **Framework: marimo.** Already the analysis artifact, so the evidence and
        the tool cannot drift apart, and it adds no dependency the brief did not
        ask for. The brief suggests Streamlit or FastAPI as guidance only.

        * One marimo app, `notebooks/simulator.py`, layered on the functions in
          `pipeline/` rather than reimplementing them.
        * A transparent parametric chain first, with every stage's current value,
          source and uncertainty on screen. A learned layer is deliberately out of
          scope: with 46 matched months and a null signal, a model would fit noise
          and hide the finding.
        * Sensitivity by Monte Carlo over the assumed visitors coefficient and
          the capture rate, reported as a distribution rather than a point.
        * Reproduce in one command; the panel rebuild already does.

        ### Artifacts

        | deliverable | content |
        | --- | --- |
        | 10-page PDF | chain definition, this evidence base, the assumed coefficient and its range, lever mapping, sensitivity, limitations |
        | 1-2 minute video | one scenario end to end, then the same scenario with the visitors coefficient at each end of its range, to show the band rather than a number |
        | notebook | this file, which is the audit trail behind every number in the PDF |

        The video is deliberately the range and not a single run. It is the one
        deliverable where showing that the answer is uncertain is the point.
        """),
    ])
    return (intl_spear,)


@app.cell
def _(
    PROC,
    beta,
    guests,
    intl_spear,
    intl_tbl,
    intl_total,
    lead,
    matched,
    means,
    mix_ablation,
    mix_results,
    mo,
    n_markets,
    n_thin20,
    nights_check,
    nights_max,
    pd,
    resident_share,
    se,
    spread,
):
    # Consistency guard: the notebook recomputes from the same CSVs as the
    # scripts, and this asserts it has not drifted from them. If a number here
    # disagrees with the published test output, the cell fails loudly rather
    # than the notebook quietly telling a different story.
    annual = pd.read_csv(PROC / "pax_arrivals_annual.csv")
    expected_annual = int(annual[annual.year == 2024].arrivals.iloc[0])
    assert expected_annual == 14522288, expected_annual

    assert len(guests) == 116, len(guests)

    expected_2017 = 4874253
    y2017 = float(guests[guests.year == 2017].value.sum())
    assert abs(y2017 - expected_2017) < 1, y2017

    assert lead == "IN", f"lead air market moved to {lead}"
    # 2023 has only five markets on the common support, so dropping the lead
    # market leaves four and the correlation is not reported for that year.
    assert set(mix_ablation.year) == {2015, 2016, 2022}, sorted(mix_ablation.year)

    # Market figures are pinned to the values printed by
    # pipeline/tests/test_market_link.py, so the notebook cannot quietly tell a
    # different story from the published report.
    expected_mix = {
        2015: (9, 0.728, 0.433, 3, 40.3),
        2016: (9, 0.862, 0.633, 4, 39.8),
        2022: (7, 0.920, 0.357, 3, 37.5),
        2023: (5, 0.959, 0.500, 2, 29.2),
    }
    got = {
        int(r["year"]): (int(r["markets"]), r["pearson"], r["spearman"],
                         int(r["top5 overlap"]), r["TVD pp"])
        for r in mix_results.to_dict("records")
    }
    assert got == expected_mix, got
    assert round(float(spread), 1) == 35.7, spread
    assert round(float(means["arrivals only"]), 1) == 12.6, means["arrivals only"]
    assert round(float(means["no arrivals"]), 1) == 5.2, means["no arrivals"]
    assert round(float(means["full model"]), 1) == 5.6, means["full model"]
    assert len(matched) == 46, len(matched)

    # The nights identity and the 2024 residency anchor are the two claims the
    # revised verdict rests on, so both are pinned rather than narrated.
    assert len(nights_check) == 21, len(nights_check)
    assert round(nights_max, 3) == 0.174, nights_max
    assert round(resident_share, 0) == 48.0, resident_share
    assert intl_total == 3_000_000, intl_total
    assert len(intl_tbl) == 6, len(intl_tbl)
    assert intl_spear == 0.2, intl_spear
    assert (n_thin20, n_markets) == (39, 62), (n_thin20, n_markets)

    mo.md(f"""
    ### Consistency checks

    | check | expected | this notebook | status |
    | --- | --- | --- | --- |
    | 2017 monthly guest sum | {expected_2017:,} | {y2017:,.0f} | pass |
    | 2017 vs published 4,875,499 | 0.026% | {abs(y2017 - 4875499) / 4875499 * 100:.3f}% | pass |
    | monthly KPI rows | 116 | {len(guests)} | pass |
    | 2024 annual arrivals | {expected_annual:,} | {expected_annual:,} | pass |
    | matched months | 46 | {len(matched)} | pass |
    | market test rows | 2015, 2016, 2022, 2023 | {', '.join(str(y) for y in sorted(got))} | pass |
    | conversion-ratio spread | 35.7x | {spread:.1f}x | pass |
    | ablation years | 2015, 2016, 2022 | {', '.join(str(int(y)) for y in mix_ablation.year)} | pass |
    | lead air market | IN | {lead} | pass |
    | arrivals elasticity | ~0.170, t < 1 | {beta[1]:.3f}, t {beta[1] / se[1]:.2f} | pass |
    | holdout WMAPE, no arrivals | 5.2 | {means['no arrivals']:.1f} | pass |
    | holdout WMAPE, full model | 5.6 | {means['full model']:.1f} | pass |
    | holdout WMAPE, arrivals only | 12.6 | {means['arrivals only']:.1f} | pass |
    | nights identity, months checked | 21 | {len(nights_check)} | pass |
    | nights identity, worst month | 0.174% | {nights_max:.3f}% | pass |
    | 2024 non-resident guests | 3,000,000 | {intl_total:,.0f} | pass |
    | 2024 resident share | 48% | {resident_share:.0f}% | pass |
    | non-resident rank correlation | 0.20 | {intl_spear:.2f} | pass |
    | markets under 20 legs | 39 of 62 | {n_thin20} of {n_markets} | pass |

    The scripts in `pipeline/` remain the canonical implementations.
    Every market figure and every holdout figure above is asserted equal to the
    value printed by `test_market_link.py` and `test_aggregate_link.py`, so the
    notebook fails loudly rather than drifting from the published reports.
    """)
    return


if __name__ == "__main__":
    app.run()
