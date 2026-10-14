"""Model access and scenario logic shared by every page and by the chatbot.

Nothing here draws UI. Pages and the agent's tools both call the same scenario
builders and the same ``summarize``, so a number in the chat is the number on
the page. All model work goes through the public API of ``pipeline.simulator``.

Three rules from docs/METHODS.md shape every helper:

* **Aggregate first.** The flight market is the last departure country; the
  hotel market is guest nationality. Market-level rows are supply-side-only,
  so summaries lead with the national total.
* **New origins are analog estimates.** The model knows 33 markets; a country
  that never flew here is priced on a chosen analog and labelled.
* **The baseline after 2026-02 is held forward**, not forecast: each month
  repeats the same calendar month a year earlier.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from pipeline import paths  # noqa: E402
from pipeline.calendar_events import COVERED_FROM, COVERED_TO  # noqa: E402
from pipeline.simulator import LEVER_BOUNDS, RouteEdit, Scenario, Simulator  # noqa: E402

OBSERVED_TO = "2026-02"
FIRST_FORWARD = "2026-03"
HORIZON_END = str(COVERED_TO)
FORWARD_YEAR = ("2026-03", "2027-02")  # the evaluation window for global levers
WINTER = ("2026-11", "2027-02")
# The app's own artifacts. A distinct model file name keeps other front ends
# that save to the engine's default (simulator_full.pkl) from overwriting it.
MODEL_PATH = paths.MODELS / "app_simulator.pkl"
VALIDATION_PATH = paths.MODELS / "app_validation.parquet"

# Pinned held-out results (T5, 2025-08..2026-02); see docs/METHODS.md §4.2.
PINNED = {"t5_wmape": 20.55, "t5_bias": -0.14, "chain_wmape": 32.77, "t4_nights_wmape": 20.8}

# Typical single-class-equivalent seat counts, for sizing; not airline-specific.
AIRCRAFT = {
    "A320 (≈174 seats)": 174,
    "A321neo (≈200 seats)": 200,
    "787-9 (≈290 seats)": 290,
    "777-300ER (≈350 seats)": 350,
    "A350-1000 (≈371 seats)": 371,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ModelUnavailable(RuntimeError):
    """No saved model and no competition pack to fit one from."""


def compatible(sim: Simulator) -> bool:
    """True if a loaded model has what the app needs: the learned length of stay
    and route supply held forward to the horizon, with its observed/projected flag."""
    try:
        routes = sim.route_supply()
        return (
            sim._alos_boosters is not None
            and "basis" in routes.columns
            and routes["ym"].max() >= COVERED_TO
        )
    except Exception:
        return False


@st.cache_resource(show_spinner="Loading the model (the first run fits it, about 1–2 minutes)…")
def get_sim() -> Simulator:
    """The fitted simulator: the app's saved artifact if it is compatible,
    otherwise a one-off fit from the pack, saved for next time."""
    try:
        sim = Simulator.load(MODEL_PATH)
        if compatible(sim):
            return sim
    except Exception:
        pass
    try:
        sim = Simulator(window="full").fit()
    except Exception as exc:
        raise ModelUnavailable(
            f"No usable model at {MODEL_PATH.relative_to(paths.ROOT)} and the competition "
            "pack is not available to fit one. Run `python -m app.build_model` where the "
            "pack is present and ship the outputs/models/ files with the app."
        ) from exc
    sim.save(MODEL_PATH)
    return sim


def markets() -> list[str]:
    return sorted(get_sim().fa["market"].unique())


def months(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def add_months(ym: str, n: int) -> str:
    return str(pd.Period(ym, freq="M") + n)


def forward_months() -> list[str]:
    return months(FIRST_FORWARD, HORIZON_END)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=256)
def score(scenario: Scenario) -> dict[str, pd.DataFrame]:
    """Market-month hybrid predictions, the monthly total and the chain total."""
    sim = get_sim()
    market = sim.market(scenario).drop(columns=["ym"])
    monthly = (
        market.groupby("ym_str")[["Total_Seats", "PAX", "P2P", "NewArrivalHat", "GuestNightsHat"]]
        .sum()
        .reset_index()
    )
    chain = (
        sim.chain(scenario)
        .groupby("ym_str")[["Total_Seats", "PAX", "P2P", "NewArrivalHat", "GuestNightsHat"]]
        .sum()
        .reset_index()
    )
    return {"market": market, "monthly": monthly, "chain": chain}


def baseline() -> dict[str, pd.DataFrame]:
    return score(Scenario())


@dataclass
class Summary:
    """What a scenario changes, against the no-change baseline."""

    delta_nights: float
    delta_arrivals: float
    window: tuple[str, str] | None
    n_months: int
    window_base_nights: float
    pct_of_window: float
    chain_delta_nights: float
    stages: dict[str, float]  # Δ seats/PAX/P2P (chain supply) + hybrid arrivals/nights
    monthly: pd.DataFrame  # ym_str, base, scenario, delta over the window
    movers: pd.DataFrame  # market, delta_nights
    labels: dict[str, str] = field(default_factory=dict)  # analog market -> new origin

    @property
    def per_year(self) -> float:
        return self.delta_nights * 12.0 / self.n_months if self.n_months else 0.0

    def to_dict(self) -> dict:
        """Compact JSON-able form for the chatbot."""
        return {
            "delta_guest_nights": round(self.delta_nights),
            "delta_new_hotel_arrivals": round(self.delta_arrivals),
            "window": {"from": self.window[0], "to": self.window[1]} if self.window else None,
            "months": self.n_months,
            "delta_guest_nights_per_year": round(self.per_year),
            "pct_of_baseline_guest_nights_in_window": round(self.pct_of_window, 2),
            "parametric_chain_delta_guest_nights": round(self.chain_delta_nights),
            "supply_change": {k: round(v) for k, v in self.stages.items()},
            "top_market_movers_supply_side_only": [
                {
                    "market": self.labels.get(r.market, r.market),
                    "delta_guest_nights": round(r.delta_nights),
                }
                for r in self.movers.head(5).itertuples()
            ],
        }


def summarize(scenario: Scenario, window: tuple[str, str] | None = None) -> Summary:
    """Score ``scenario`` and compare it to the baseline.

    ``window`` limits the comparison to a month range; by default it is the
    months the scenario actually changes (a winter-only edit reports winter).
    """
    base, scen = baseline(), score(scenario)
    cmp = base["monthly"].merge(scen["monthly"], on="ym_str", suffixes=("_b", "_s"))
    cmp["delta"] = cmp["GuestNightsHat_s"] - cmp["GuestNightsHat_b"]
    if window is None:
        changed = cmp.loc[cmp["delta"].abs() > 0.5, "ym_str"]
        window = (changed.iloc[0], changed.iloc[-1]) if len(changed) else None
    in_win = cmp["ym_str"].between(*window) if window else cmp["ym_str"] == ""
    w = cmp[in_win]

    chain = base["chain"].merge(scen["chain"], on="ym_str", suffixes=("_b", "_s"))
    chain = chain[chain["ym_str"].between(*window)] if window else chain.iloc[0:0]
    stages = {
        "seats": chain["Total_Seats_s"].sum() - chain["Total_Seats_b"].sum(),
        "passengers": chain["PAX_s"].sum() - chain["PAX_b"].sum(),
        "terminating_passengers": chain["P2P_s"].sum() - chain["P2P_b"].sum(),
        "hotel_arrivals": (w["NewArrivalHat_s"] - w["NewArrivalHat_b"]).sum(),
        "guest_nights": w["delta"].sum(),
    }

    bm, sm = base["market"], scen["market"]
    if window:
        bm, sm = bm[bm["ym_str"].between(*window)], sm[sm["ym_str"].between(*window)]
    movers = (
        sm.groupby("market")["GuestNightsHat"].sum()
        .sub(bm.groupby("market")["GuestNightsHat"].sum(), fill_value=0.0)
        .rename("delta_nights")
        .reset_index()
    )
    movers = movers[movers["delta_nights"].abs() > 0.5].sort_values(
        "delta_nights", key=abs, ascending=False
    )
    base_nights = w["GuestNightsHat_b"].sum()
    return Summary(
        delta_nights=float(w["delta"].sum()),
        delta_arrivals=float((w["NewArrivalHat_s"] - w["NewArrivalHat_b"]).sum()),
        window=window,
        n_months=int(len(w)),
        window_base_nights=float(base_nights),
        pct_of_window=float(100.0 * w["delta"].sum() / base_nights) if base_nights else 0.0,
        chain_delta_nights=float(chain["GuestNightsHat_s"].sum() - chain["GuestNightsHat_b"].sum()),
        stages={k: float(v) for k, v in stages.items()},
        monthly=w.rename(
            columns={"GuestNightsHat_b": "base", "GuestNightsHat_s": "scenario"}
        )[["ym_str", "base", "scenario", "delta"]].reset_index(drop=True),
        movers=movers.reset_index(drop=True),
        labels={e.market: f"{e.label} (analog: {e.market})" for e in scenario.routes if e.label},
    )


# ---------------------------------------------------------------------------
# Supply tables
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def route_table() -> pd.DataFrame:
    """Current routes: mean weekly seats/flights over the last 3 observed months."""
    r = get_sim().route_supply()
    obs = r[r["basis"] == "observed"]
    recent = obs[obs["ym"].isin(sorted(obs["ym"].unique())[-3:])].copy()
    recent["weekly_seats"] = recent["Total_Seats"] * 7.0 / recent["ym"].dt.days_in_month
    out = (
        recent.groupby(["market", "city", "airline"])
        .agg(weekly_seats=("weekly_seats", "mean"), weekly_flights=("flights_per_week", "mean"))
        .reset_index()
    )
    out["seats_per_flight"] = out["weekly_seats"] / out["weekly_flights"].where(out["weekly_flights"] > 0)
    return out.sort_values("weekly_seats", ascending=False).reset_index(drop=True)


def find_route(market: str, city: str | None, airline: str | None) -> pd.Series | None:
    """The named route, or the market's busiest one when city/airline are omitted."""
    t = route_table()
    t = t[t["market"] == market.upper().strip()]
    if city:
        t = t[t["city"].str.lower() == city.lower().strip()]
    if airline:
        t = t[t["airline"].str.lower().str.contains(airline.lower().strip(), regex=False)]
    return None if t.empty else t.iloc[0]


@st.cache_data(show_spinner=False)
def market_weekly_seats() -> pd.DataFrame:
    """Baseline weekly seats per market and month (observed + held forward)."""
    r = get_sim().route_supply()
    g = r.groupby(["ym", "market"])["Total_Seats"].sum().reset_index()
    g["weekly_seats"] = g["Total_Seats"] * 7.0 / g["ym"].dt.days_in_month
    g["ym_str"] = g["ym"].astype(str)
    return g[["ym_str", "market", "weekly_seats"]]


@st.cache_data(show_spinner=False)
def market_yield() -> pd.DataFrame:
    """Measured conversion per market on the training panel (2022-01..2025-07)."""
    p = get_sim().panel
    g = (
        p.groupby("market")
        .agg(
            arrivals=("NewArr", "sum"),
            guest_nights=("Guests", "sum"),
            p2p=("Total_P2P", "sum"),
            seats=("Total_Seats", "sum"),
        )
        .reset_index()
    )
    g["arrivals_per_1000_seats"] = 1000.0 * g["arrivals"] / g["seats"]
    g["nights_per_1000_seats"] = 1000.0 * g["guest_nights"] / g["seats"]
    g["capture"] = g["arrivals"] / g["p2p"]
    g["alos"] = g["guest_nights"] / g["arrivals"]
    return g.sort_values("arrivals_per_1000_seats", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def hotel_rooms() -> tuple[int, str]:
    """Latest published Abu Dhabi room count from the public DCT reports."""
    try:
        d = pd.read_csv(paths.ROOT / "data/processed/dct_hotel_supply.csv")
        d = d[(d["segment"] == "total") & (d["category"] == "total")].dropna(subset=["rooms"])
        row = d.sort_values(["year", "period"]).iloc[-1]
        return int(row["rooms"]), f"DCT {int(row['year'])}"
    except Exception:
        return 34_000, "DCT 2024"


def occupancy_points(delta_nights: float, window: tuple[str, str], rooms: int,
                     guests_per_room: float) -> float:
    """Indicative occupancy change (percentage points) for a guest-night delta.

    The pack has no room data, so this rests on two stated assumptions: the
    room stock and the average guests per occupied room.
    """
    days = sum(pd.Period(m, freq="M").days_in_month for m in months(*window))
    return 100.0 * delta_nights / guests_per_room / (rooms * days)


# ---------------------------------------------------------------------------
# Scenario builders: one per question in the brief
# ---------------------------------------------------------------------------
def _end(start: str, n_months: int) -> str:
    return min(add_months(start, n_months - 1), HORIZON_END)


def new_route(market: str, flights_per_week: float, seats_per_flight: float, start: str,
              n_months: int = 12, label: str | None = None,
              count_as_new_route: bool = False) -> tuple[Scenario, list[str]]:
    """New weekly flights from ``market`` (or a new country priced on it as an analog).

    By default the seats join the market's busiest existing route, so the
    estimate is the *capacity* effect alone. The model also carries per-market
    route and airline counts, which mostly encode market size; incrementing
    them for one service inflates the estimate several-fold, so that variant
    (``count_as_new_route``) is opt-in.
    """
    notes = [f"{label} is not in the data; it is priced on {market}'s behaviour."] if label else []
    route = None if count_as_new_route else find_route(market, None, None)
    city, airline = (route["city"], route["airline"]) if route is not None else ("New route", "New carrier")
    edit = RouteEdit(
        market=market, city=city, airline=airline,
        seats_per_week=float(flights_per_week) * float(seats_per_flight),
        started_period=start, ended_period=_end(start, n_months), label=label or None,
    )
    return Scenario(routes=(edit,)), notes


def frequency_change(market: str, delta_flights: float, start: str, end: str,
                     city: str | None = None, airline: str | None = None,
                     seats_per_flight: float | None = None) -> tuple[Scenario, list[str]]:
    """More or fewer weekly flights on a route; cuts are clamped to what flies."""
    notes: list[str] = []
    route = find_route(market, city, airline)
    if route is not None:
        city, airline = route["city"], route["airline"]
        spf = seats_per_flight or route["seats_per_flight"]
        if pd.isna(spf):
            spf = 300.0
        current = route["weekly_flights"]
        if delta_flights < 0 and pd.notna(current) and -delta_flights > current:
            notes.append(f"Cut limited to the {current:.0f} weekly flights {city} / {airline} operates.")
            delta_flights = -float(current)
    else:
        if delta_flights < 0:
            raise ValueError(f"No current route matches {market} {city or ''} {airline or ''}; nothing to cut.")
        spf = seats_per_flight or 300.0
        notes.append("Route not in current schedule; added as a new service.")
    edit = RouteEdit(
        market=market, city=city or "New route", airline=airline or "New carrier",
        seats_per_week=float(delta_flights) * float(spf), started_period=start, ended_period=end,
    )
    return Scenario(routes=(edit,)), notes


def gauge_change(market: str, delta_seats_per_flight: float, start: str, end: str,
                 city: str | None = None, airline: str | None = None) -> tuple[Scenario, list[str]]:
    """Larger (or smaller) aircraft on an existing route, same frequency."""
    route = find_route(market, city, airline)
    if route is None or pd.isna(route["weekly_flights"]):
        raise ValueError(f"No current route matches {market} {city or ''} {airline or ''}.")
    edit = RouteEdit(
        market=market, city=route["city"], airline=route["airline"],
        seats_per_week=float(route["weekly_flights"]) * float(delta_seats_per_flight),
        started_period=start, ended_period=end,
    )
    return Scenario(routes=(edit,)), []


def seasonal_mix(shifts: dict[str, float], start: str, end: str) -> tuple[Scenario, list[str]]:
    """Scale chosen markets' seats over a season (e.g. ``{"RUSSIAN FEDERATION": 1.2}``).

    The model has no seasonal per-market lever, so each month becomes a
    one-month route edit on the market's busiest route, sized from that month's
    baseline seats.
    """
    ws = market_weekly_seats()
    edits = []
    for market, mult in shifts.items():
        route = find_route(market, None, None)
        if route is None:
            raise ValueError(f"{market} has no current routes.")
        for ym in months(start, end):
            base = ws.loc[(ws["market"] == market) & (ws["ym_str"] == ym), "weekly_seats"].sum()
            delta = (float(mult) - 1.0) * base
            if abs(delta) > 0:
                edits.append(RouteEdit(market=market, city=route["city"], airline=route["airline"],
                                       seats_per_week=delta, started_period=ym, ended_period=ym))
    return Scenario(routes=tuple(edits)), []


@st.cache_data(show_spinner=False)
def lever_sensitivity(step: float = 0.10) -> pd.DataFrame:
    """Each global lever moved ±``step`` alone, over the forward year."""
    rows = []
    for lever in LEVER_BOUNDS:
        lo = summarize(Scenario(**{lever: 1 - step}), FORWARD_YEAR).delta_nights
        hi = summarize(Scenario(**{lever: 1 + step}), FORWARD_YEAR).delta_nights
        rows.append({"lever": lever, "low": lo, "high": hi, "swing": hi - lo})
    return pd.DataFrame(rows).sort_values("swing").reset_index(drop=True)


# pipeline.sensitivity's lever names, in its order, mapped to Scenario fields.
SENSITIVITY_LEVERS = {
    "seats / frequency": "seats_mult",
    "load factor": "lf_mult",
    "P2P share (transfer/transit)": "p2p_mult",
    "capture rate": "capture_mult",
    "ALOS": "alos_mult",
    "season": "season_mult",
}


@st.cache_data(show_spinner=False)
def measured_sensitivity() -> pd.DataFrame:
    """Each lever moved to its measured 16th and 84th percentile, alone.

    The swings come from ``pipeline.sensitivity`` (how far each ratio actually
    moved month to month in 2022-2025); the scoring is the hybrid model over
    the forward year, so it is comparable with ``lever_sensitivity``.
    """
    from pipeline import sensitivity as S

    sim = get_sim()
    spread = S.lever_spread_table(sim.panel, sim.params).set_index("lever")
    rows = []
    for name, lever in SENSITIVITY_LEVERS.items():
        q16, q84 = float(spread.at[name, "q16"]), float(spread.at[name, "q84"])
        lo = summarize(Scenario(**{lever: q16}), FORWARD_YEAR).delta_nights
        hi = summarize(Scenario(**{lever: q84}), FORWARD_YEAR).delta_nights
        rows.append({"lever": lever, "q16": q16, "q84": q84, "low": lo, "high": hi, "swing": hi - lo})
    return pd.DataFrame(rows).sort_values("swing").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def sobol_views() -> pd.DataFrame | None:
    """Share of guest-night variance per lever (Sobol total order, parametric chain)
    under the a-priori ranges and the measured ranges; None without SALib."""
    try:
        from pipeline import sensitivity as S
        import SALib  # noqa: F401
    except ImportError:
        return None
    sim = get_sim()
    apriori = {name: LEVER_BOUNDS[lever][:2] for name, lever in SENSITIVITY_LEVERS.items()}
    measured = S.empirical_lever_bounds(sim.panel, sim.params)
    a = S.sobol_sensitivity(sim.panel, sim.params, apriori).set_index("lever")["ST"]
    m = S.sobol_sensitivity(sim.panel, sim.params, measured).set_index("lever")["ST"]
    out = pd.DataFrame({"apriori": a, "measured": m}).reset_index()
    out["lever"] = out["lever"].map(SENSITIVITY_LEVERS)
    return out.sort_values("measured").reset_index(drop=True)


LEVER_LABELS = {
    "seats_mult": "Seats (capacity)",
    "lf_mult": "Load factor",
    "p2p_mult": "Point-to-point share",
    "capture_mult": "Hotel capture rate",
    "alos_mult": "Length of stay",
    "season_mult": "Seasonal demand",
}


@st.cache_data(show_spinner=False)
def validation() -> pd.DataFrame | None:
    """The precomputed back-tests (see ``app.build_model``), or None."""
    if VALIDATION_PATH.exists():
        return pd.read_parquet(VALIDATION_PATH)
    return None


def scenario_json(scenario: Scenario) -> str:
    from dataclasses import asdict
    return json.dumps(asdict(scenario), indent=2)


__all__ = [
    "AIRCRAFT", "COVERED_FROM", "COVERED_TO", "LEVER_BOUNDS", "LEVER_LABELS", "PINNED",
    "RouteEdit", "Scenario", "Summary", "WINTER", "FORWARD_YEAR",
]
