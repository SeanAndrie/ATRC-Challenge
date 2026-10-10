"""Scenario simulator built on the canonical model in ``pipeline.hybrid``.

This is the integration point for the Streamlit front end. It exposes a fitted
parametric chain *and* the hybrid LightGBM layer behind one small class, so the
UI needs only::

    from pipeline.simulator import Simulator, Scenario
    sim = Simulator().fit()                     # trains the requested window
    sim.monthly(Scenario(seats_mult=1.2))       # safe aggregate view
    sim.market(Scenario(seats_mult=1.2))        # market view, supply-side-only

Fitting is lazy and per-window. ``Simulator(window=...)`` chooses which model
default scenarios are scored with, and ``fit()`` trains **only that window**:

* ``"full"`` -- the default scenario model. Iteration count is chosen on the
  internal last-6-month tail (the notebook's T4 protocol), then the 5-seed
  ensemble is refit on the whole training panel (2022-01..2025-07). Scenarios
  fly the model forward to ``COVERED_TO`` = 2028-12: observed supply ends at
  2026-02 and the remaining months use a held-forward seasonal baseline (see
  ``route_supply()``), with every supply row marked ``basis`` =
  observed / projected / scenario.
* ``"competition"`` -- the T5 protocol (iterations on the leave-tail holdout).
  ``reproduce_competition_window`` trains it on demand and replays the pinned
  numbers live.

A fitted simulator can be cached with ``save()`` / ``load()`` (pickled to
``outputs/models/``, gitignored). No artifact is required -- the fit-on-demand
path is deterministic and yields identical predictions -- so the cache is purely
to spare the ~2-minute fit on a Streamlit restart.

Lever semantics

* Supply levers (``seats_mult``, ``lf_mult``, ``p2p_mult``) act *pre-model*:
  they rewrite the supply frame, so the hybrid's features (``log_seats``,
  ``load_factor``, ``p2p_share``, ``routes``, ``airlines``) *and* its Poisson
  exposure offset respond non-linearly. That is the feature, not a bug. They must
  be positive; a zeroed supply lever is rejected rather than silently degenerated.
* ``capture_mult``, ``season_mult``, ``alos_mult`` are post-hoc dials on the
  model output: they scale New Arrivals (capture, season) and guest-nights
  (ALOS) without touching the learned features. In the chain path the same three
  apply, so both paths honour every dial.
* ``market_seats_mult`` is a tuple of ``(market, multiplier)`` pairs that scale a
  single market on top of the global ``seats_mult`` (a per-market multiplier,
  not a cap); unknown market names and non-positive values are rejected. A tuple
  keeps ``Scenario`` genuinely immutable.
* ``market_lf_mult`` is the load-factor twin of ``market_seats_mult``: a tuple
  of ``(market, multiplier)`` pairs scaling one market's fill rate on top of the
  global ``lf_mult``.
* ``RouteEdit`` edits terminating supply for a model market over an inclusive
  month range (``started_period``..``ended_period``, default through
  ``COVERED_TO``). ``seats_per_week`` is signed: positive opens or grows a
  route, negative trims one (aggregate supply is floored at zero; the caller
  should keep reductions within the route's own weekly seats). PAX/P2P are
  estimated at the market's average load factor and P2P share, and ``label``
  names a new origin whose demand is modelled on an analog market.

Attribution guard (docs/METHODS.md §4.5)

* Aggregate views (``monthly``, ``seasonal``) are the safe defaults.
* Market-level frames carry both ``frame.attrs["data_status"]`` and a constant
  ``data_status`` column (``"supply-side-only"``), so the flag survives
  ``groupby`` and serialization; the UI must not present market levels as
  measured.
* Scenarios cannot invent new markets (the model's ``market`` categorical is a
  fixed 33-level encoding): a genuinely new origin is modelled on a chosen
  analog market and its delta read from that market. Supply edits must lie
  within ``COVERED_FROM``..``COVERED_TO`` (the event calendar's horizon); bad
  months, unknown markets and zeroed levers all produce clear errors.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import hybrid as H  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.calendar_events import COVERED_FROM, COVERED_TO  # noqa: E402
from pipeline.competition import load as cl  # noqa: E402

# Recommended slider ranges, taken from METHODS §3.4's sensitivity bounds.
LEVER_BOUNDS = {
    "seats_mult": (0.8, 1.2, 0.01),
    "lf_mult": (0.9, 1.1, 0.01),
    "p2p_mult": (0.9, 1.1, 0.01),
    "capture_mult": (0.7, 1.3, 0.01),
    "alos_mult": (0.9, 1.1, 0.01),
    "season_mult": (0.9, 1.1, 0.01),
}

SUPPLY_LEVERS = ("seats_mult", "lf_mult", "p2p_mult")


@dataclass(frozen=True)
class RouteEdit:
    """Edit terminating supply for one model market over a month range.

    ``seats_per_week`` is signed: positive adds supply (a new route or extra
    seats on an existing one), negative trims it. PAX and P2P are estimated at
    the model market's average load factor and P2P share observed over the pack,
    so the added seats convert like the market's history; supply is applied
    monthly (``seats_per_week * days_in_month / 7``). ``ended_period`` is
    inclusive and defaults to the calendar horizon, which makes winter-only or
    seasonal edits possible. ``label`` is display-only: a new origin (say
    KENYA) is encoded on an analog market (say INDIA) and its delta read from
    that market, because the model's categorical cannot admit new levels.
    """

    market: str
    city: str
    airline: str
    seats_per_week: float
    started_period: str  # e.g. "2026-11", a month within COVERED_FROM..COVERED_TO
    ended_period: str | None = None  # inclusive; None = through COVERED_TO
    label: str | None = None  # display name for an analog-market new origin


@dataclass(frozen=True)
class Scenario:
    seats_mult: float = 1.0
    lf_mult: float = 1.0
    p2p_mult: float = 1.0
    capture_mult: float = 1.0
    alos_mult: float = 1.0
    season_mult: float = 1.0
    market_seats_mult: tuple = ()  # ((market, multiplier), ...)
    market_lf_mult: tuple = ()  # ((market, multiplier), ...)
    routes: tuple = ()


class Simulator:
    """Fitted chain + hybrid, scenario prediction, and pinned reproduction."""

    def __init__(self, window: str = "full"):
        if window not in ("full", "competition"):
            raise ValueError(f"window must be 'full' or 'competition', got {window!r}")
        self.window = window
        self._data_ready = False
        self.fa = None
        self.panel = None
        self.test_panel = None
        self.params = None
        self._base_routes = None
        self._boosters = {"full": None, "competition": None}
        self._best_iter = {"full": None, "competition": None}

    # -- fitting -------------------------------------------------------------
    def fit(self) -> "Simulator":
        """Load the pack and train the requested window. Idempotent."""
        self._ensure_data()
        self._ensure(self.window)
        return self

    def _ensure_data(self) -> "Simulator":
        """Load panels, fit the chain, and cache the base supply. Once."""
        if self._data_ready:
            return self
        fa, _mkt_dtype, _mon_dtype, panel, _test_m, test_panel, _train_m = H.build_panels()
        self.fa = fa
        self.panel = panel
        self.test_panel = test_panel
        self.params = H.fit_params(panel)
        self._base_routes = self._build_base_routes()
        self._data_ready = True
        return self

    def _ensure(self, window: str) -> None:
        """Train ``window`` once, on demand. Only that window is ever fitted."""
        if self._boosters[window] is not None:
            return
        self._ensure_data()
        if window == "full":
            # Iterations on the internal last-6-month tail (T4 protocol).
            core, val = self.panel[self.panel["ym"] <= "2024-07"], self.panel[
                self.panel["ym"] > "2024-07"
            ]
        else:
            # Iterations on the leave-tail holdout (T5 protocol).
            core, val = self.panel[self.panel["ym"] <= "2025-01"], self.panel[
                self.panel["ym"] > "2025-01"
            ]
        best = H.fit_poisson(core, val)
        self._best_iter[window] = best
        self._boosters[window] = H.fit_hybrid(self.panel, best)

    def _require_data(self) -> None:
        if not self._data_ready:
            raise RuntimeError(
                "Simulator.fit() (or Simulator.load()) must be called before use"
            )

    # -- scenario scoring ----------------------------------------------------
    def _predict(self, scenario: Scenario, markets: str | list[str] | None = "all"):
        """Market-month predictions under ``scenario`` (low level; prefer the
        ``monthly`` / ``seasonal`` / ``market`` views).

        Frames carry ``data_status="supply-side-only"`` as both a column and an
        attr, because the ``market`` join is structurally wrong (METHODS §4.5).
        """
        self._ensure(self.window)
        self._require_data()
        scaled, _raw = self._scenario_supply(scenario)
        scaled = self._slice_markets(scaled, markets)

        pred = H.predict_hybrid(self._boosters[self.window], scaled)
        alos = scaled["market"].map(self.params["alos"]).to_numpy(dtype=float)
        new_arr = pred * scenario.capture_mult * scenario.season_mult
        out = scaled[
            ["ym_str", "ym", "market", "month", "routes", "airlines",
             "Total_Seats", "PAX", "P2P", "basis"]
        ].copy()
        out["NewArrivalHat"] = new_arr
        out["GuestNightsHat"] = new_arr * alos * scenario.alos_mult
        out["data_status"] = "supply-side-only"
        out.attrs["data_status"] = "supply-side-only"
        return out

    def chain(self, scenario: Scenario, markets: str | list[str] | None = "all"):
        """The parametric chain under the same scenario (via ``hybrid.simulate``).

        Supply levers pass straight to ``hybrid.simulate``. ``season_mult`` is
        applied once, here, rather than passed in: ``simulate`` accepts but never
        applies it (a documented no-op kept for notebook fidelity), so both the
        hybrid and chain paths honour the dial identically.
        """
        self._require_data()
        _scaled, raw = self._scenario_supply(scenario)
        raw = self._slice_markets(raw, markets)

        out = H.simulate(
            raw,
            self.params,
            seats_mult=scenario.seats_mult,
            lf_mult=scenario.lf_mult,
            p2p_mult=scenario.p2p_mult,
            capture_mult=scenario.capture_mult,
            alos_mult=scenario.alos_mult,
        )
        out["NewArrivalHat"] = out["NewArrivalHat"] * scenario.season_mult
        out["GuestNightsHat"] = out["GuestNightsHat"] * scenario.season_mult
        out["data_status"] = "supply-side-only"
        out.attrs["data_status"] = "supply-side-only"
        return out

    def monthly(self, scenario: Scenario):
        """Aggregate guest-nights by month -- the safe default view."""
        f = self._predict(scenario)
        g = f.groupby("ym").agg(
            P2P=("P2P", "sum"),
            NewArrivalHat=("NewArrivalHat", "sum"),
            GuestNightsHat=("GuestNightsHat", "sum"),
        )
        g["ym_str"] = g.index.astype(str)
        return g.reset_index(drop=True)

    def seasonal(self, scenario: Scenario):
        """Aggregate by calendar month (seasonal shape)."""
        f = self._predict(scenario)
        return (
            f.groupby("month")
            .agg(
                P2P=("P2P", "sum"),
                NewArrivalHat=("NewArrivalHat", "sum"),
                GuestNightsHat=("GuestNightsHat", "sum"),
            )
            .reset_index()
        )

    def market(self, scenario: Scenario):
        """Market-level view. Rows are supply-side-only (see the module guard)."""
        return self._predict(scenario)

    def native_capture(self) -> pd.Series:
        """Per-market chain capture (calibration coefficient), the model's view."""
        self._require_data()
        return self.params["capture"].copy()

    def validate_scenario(self, scenario: Scenario) -> None:
        """Raise ``ValueError`` if ``scenario`` cannot be scored.

        A convenience for the UI: the same checks prediction runs, surfaced
        before any model work so bad periods, unknown markets or degenerate
        levers read as form errors rather than stack traces.
        """
        self._require_data()
        self._validate(scenario)

    def reproduce_competition_window(self):
        """Train the competition window if needed and replay the T5 protocol.

        Returns the test-panel predictions with ``wm_na`` (WMAPE of New Arrivals)
        and ``bias_pct`` on the attrs. The notebook Part C pins these near 20.8%
        and -0.1%; the test suite asserts them within tolerance.
        """
        self._ensure("competition")
        pred = H.predict_hybrid(self._boosters["competition"], self.test_panel)
        actual = self.test_panel["NewArr"].to_numpy(dtype=float)
        wm = H.wmape(actual, pred)
        bias = 100.0 * (pred.sum() / actual.sum() - 1.0)
        out = self.test_panel[
            ["ym_str", "market", "month", "Total_Seats", "Total_PAX", "Total_P2P"]
        ].copy()
        out["NewArr"] = actual
        out["NewArrivalHat"] = pred
        out.attrs["wm_na"] = wm
        out.attrs["bias_pct"] = bias
        out.attrs["best_iter"] = self._best_iter["competition"]
        return out

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        """Pickle the fitted state (params + supply cache + boosters + dtypes).

        Defaults to ``outputs/models/simulator_<window>.pkl`` (gitignored). The
        artifact is a cache; predictions from a re-fit are identical. The
        categorical encodings ride along so ``load()`` works in a fresh process
        without re-reading the competition pack.
        """
        self._require_data()
        p = Path(path) if path is not None else paths.MODELS / f"simulator_{self.window}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window": self.window,
            "params": self.params,
            "fa": self.fa,
            "panel": self.panel,
            "test_panel": self.test_panel,
            "base_routes": self._base_routes,
            "boosters": self._boosters,
            "best_iter": self._best_iter,
            "dtypes": H.categorical_dtypes(),
        }
        with p.open("wb") as fh:
            pickle.dump(payload, fh)
        return p

    @classmethod
    def load(cls, path: str | Path | None = None, window: str | None = None) -> "Simulator":
        """Load a simulator saved by ``save()``.

        Restores the hybrid module's categorical encodings from the payload. An
        older pickle without them falls back to ``build_panels()``, which needs
        the pack on disk.
        """
        w = window or "full"
        p = Path(path) if path is not None else paths.MODELS / f"simulator_{w}.pkl"
        with p.open("rb") as fh:
            payload = pickle.load(fh)
        mkt_dtype, mon_dtype = payload.get("dtypes", (None, None))
        if mkt_dtype is None or mon_dtype is None:
            H.build_panels()
        else:
            H.set_categorical_dtypes(mkt_dtype, mon_dtype)
        obj = cls(window=window or payload["window"])
        obj.params = payload["params"]
        obj.fa = payload["fa"]
        obj.panel = payload["panel"]
        obj.test_panel = payload["test_panel"]
        obj._base_routes = payload["base_routes"]
        obj._boosters = payload["boosters"]
        obj._best_iter = payload["best_iter"]
        obj._data_ready = True
        return obj

    def route_supply(self) -> pd.DataFrame:
        """Route-grain baseline supply (observed + projected). A copy, so the
        caller can filter or join without mutating the fitted state.

        Columns: ``ym, market, city, airline, Total_Seats, Total_PAX,
        Total_P2P, Total_Transfer, Total_Transit, flights_per_week, basis``.
        ``basis`` is ``"observed"`` for pack months (through 2026-02) and
        ``"projected"`` beyond it, where each month holds forward the same
        calendar month one year earlier.
        """
        self._require_data()
        return self._base_routes.copy()

    # -- supply assembly -----------------------------------------------------
    def _build_base_routes(self) -> pd.DataFrame:
        """Route-grain supply (market, city, airline, month), cached once.

        Route grain is kept -- not market grain -- so a ``RouteEdit`` can add a
        city/airline row and have ``routes``/``airlines`` recount correctly.
        ``flights_per_week`` supports the UI's frequency/aircraft calculators;
        transfer/transit ride along so scenario frames never re-join ``fa``.
        """
        rows = cl.flights_monthly()[
            ["ym", "Departure Country Name", "Departure City", "Airline Name",
             "Total_Seats", "Total_PAX", "Total_P2P", "Total_Transfer",
             "Total_Transit", "flights_per_week_median"]
        ].rename(
            columns={
                "Departure Country Name": "market",
                "Departure City": "city",
                "Airline Name": "airline",
                "flights_per_week_median": "flights_per_week",
            }
        ).copy()
        rows["market"] = rows["market"].str.upper().str.strip()
        rows["ym"] = pd.PeriodIndex(rows["ym"], freq="M")
        rows["basis"] = "observed"
        return self._project_routes(rows)

    @staticmethod
    def _project_routes(rows: pd.DataFrame) -> pd.DataFrame:
        """Hold the last observed same-month schedule forward to ``COVERED_TO``.

        The pack's supply ends at 2026-02; scenarios for the next winter would
        otherwise have no baseline. Each future month copies the same calendar
        month one year earlier -- chaining through months that are themselves
        projected -- so seasonality carries forward while no new airline
        schedule is invented. Copied rows are marked ``basis="projected"``.
        """
        last = rows["ym"].max()
        if last >= COVERED_TO:
            return rows
        by_month = {ym: group for ym, group in rows.groupby("ym")}
        parts = [rows]
        ym = last + 1
        while ym <= COVERED_TO:
            source = by_month.get(ym - 12)
            if source is not None and len(source):
                new = source.copy()
                new["ym"] = ym
                new["basis"] = "projected"
                by_month[ym] = new
                parts.append(new)
            ym += 1
        return pd.concat(parts, ignore_index=True)

    def _scenario_supply(self, scenario: Scenario):
        """Build raw (route edits, no multipliers) and scaled supply frames.

        Both are at market x month grain with the feature columns the model and
        chain need. Route edits are signed and dated, so a scenario can open a
        winter-only route or trim an airline's weekly seats. ``scaled`` applies
        the supply multipliers, which rewrite ``Total_Seats`` and re-derive
        PAX/P2P so features and the exposure offset respond; ``raw`` (used by
        ``chain``) carries the same seats/PAX/P2P, so ``Scenario()`` exactly
        reproduces ``fa`` on observed months. Market-months whose seats net to
        zero are dropped: no flights means no modelled demand.
        """
        self._require_data()
        self._validate(scenario)
        rows = self._base_routes.copy()

        edit_frames = []
        for e in scenario.routes:
            mkt = e.market.upper().strip()
            start = pd.Period(e.started_period, freq="M")
            end = COVERED_TO if e.ended_period is None else pd.Period(e.ended_period, freq="M")
            fa_m = self.fa[self.fa["market"] == mkt]
            lf = fa_m["Total_PAX"].sum() / max(fa_m["Total_Seats"].sum(), 1.0)
            share = fa_m["Total_P2P"].sum() / max(fa_m["Total_PAX"].sum(), 1.0)
            periods = pd.period_range(start, end, freq="M")
            seats = e.seats_per_week * periods.days_in_month / 7.0
            edit_frames.append(
                pd.DataFrame(
                    {
                        "ym": periods,
                        "market": mkt,
                        "city": e.city,
                        "airline": e.airline,
                        "Total_Seats": seats,
                        "Total_PAX": seats * lf,
                        "Total_P2P": seats * share,
                        "Total_Transfer": 0.0,
                        "Total_Transit": 0.0,
                        "flights_per_week": 0.0,
                        "basis": "scenario",
                    }
                )
            )
        if edit_frames:
            rows = pd.concat([rows, *edit_frames], ignore_index=True)

        # Counts only over routes that actually operate: a reduction that zeroes
        # a route must drop it from routes/airlines, and negative edit rows are
        # not routes.
        positive = rows[rows["Total_Seats"] > 0]
        counts = (
            positive.groupby(["ym", "market"], dropna=False)
            .agg(routes=("city", "nunique"), airlines=("airline", "nunique"))
            .reset_index()
        )
        priority = {"observed": 0, "projected": 1, "scenario": 2}
        rows["_pri"] = rows["basis"].map(priority).fillna(0).astype(int)
        agg = (
            rows.groupby(["ym", "market"], dropna=False)
            .agg(
                Total_Seats=("Total_Seats", "sum"),
                Total_PAX=("Total_PAX", "sum"),
                Total_P2P=("Total_P2P", "sum"),
                Total_Transfer=("Total_Transfer", "sum"),
                Total_Transit=("Total_Transit", "sum"),
                pri=("_pri", "max"),
            )
            .reset_index()
            .merge(counts, on=["ym", "market"], how="left")
        )
        agg[["routes", "airlines"]] = agg[["routes", "airlines"]].fillna(0).astype(int)
        agg["basis"] = agg["pri"].map({v: k for k, v in priority.items()})
        agg = agg.drop(columns="pri")

        # A reduction can overshoot a small market-month; floor supply at zero
        # rather than emit negative seats (the UI checks route-level overshoot).
        for col in ("Total_Seats", "Total_PAX", "Total_P2P", "Total_Transfer", "Total_Transit"):
            agg[col] = agg[col].clip(lower=0)
        agg = agg[agg["Total_Seats"] > 0].copy()
        agg["ym_str"] = agg["ym"].astype(str)
        agg["month"] = agg["ym"].dt.month

        raw = agg[
            ["ym_str", "ym", "market", "month", "routes", "airlines",
             "Total_Seats", "Total_PAX", "Total_P2P",
             "Total_Transfer", "Total_Transit", "basis"]
        ].copy()

        # Scale supply: pre-model levers that rebake into features + exposure.
        seats_had = raw["Total_Seats"].to_numpy(dtype=float) > 0
        base_seats = raw["Total_Seats"].clip(lower=1).to_numpy(dtype=float)
        lf = np.where(seats_had, raw["Total_PAX"].to_numpy(dtype=float) / base_seats, 0.0)
        share = np.where(
            seats_had,
            raw["Total_P2P"].to_numpy(dtype=float)
            / raw["Total_PAX"].clip(lower=1).to_numpy(dtype=float),
            0.0,
        )
        cmult = self._market_mult(scenario, raw["market"])
        lfmult = self._market_lf_mult(scenario, raw["market"])
        scaled = raw.copy()
        scaled["Total_Seats"] = raw["Total_Seats"] * cmult
        scaled["Total_PAX"] = scaled["Total_Seats"] * lf * scenario.lf_mult * lfmult
        scaled["Total_P2P"] = scaled["Total_PAX"] * share * scenario.p2p_mult
        scaled["PAX"] = scaled["Total_PAX"]
        scaled["P2P"] = scaled["Total_P2P"]
        return scaled, raw

    def _validate(self, scenario: Scenario) -> None:
        for name in SUPPLY_LEVERS:
            if getattr(scenario, name) <= 0:
                raise ValueError(
                    f"{name} must be > 0 (a zeroed supply lever is degenerate); "
                    f"got {getattr(scenario, name)!r}"
                )
        known = set(self.fa["market"])
        for name, pairs in (
            ("market_seats_mult", scenario.market_seats_mult),
            ("market_lf_mult", scenario.market_lf_mult),
        ):
            for mkt, mult in pairs:
                if mkt not in known:
                    raise ValueError(
                        f"{name} key {mkt!r} is not one of the 33 flight "
                        "markets; new markets cannot be encoded by the model."
                    )
                if mult <= 0:
                    raise ValueError(f"{name}[{mkt!r}] must be > 0; got {mult!r}")
        for e in scenario.routes:
            mkt = e.market.upper().strip()
            if mkt not in known:
                raise ValueError(
                    f"RouteEdit market {mkt!r} is not one of the 33 flight "
                    "markets; new markets cannot be encoded by the model. Use an "
                    "analog market and a label instead."
                )
            if e.seats_per_week == 0:
                raise ValueError("RouteEdit seats_per_week must be non-zero.")
            try:
                start = pd.Period(e.started_period, freq="M")
            except Exception as exc:
                raise ValueError(
                    f"RouteEdit started_period {e.started_period!r} is not a "
                    f"YYYY-MM month: {exc}"
                ) from exc
            if start < COVERED_FROM or start > COVERED_TO:
                raise ValueError(
                    f"RouteEdit started_period {start} must be within "
                    f"{COVERED_FROM}..{COVERED_TO}."
                )
            if e.ended_period is not None:
                try:
                    end = pd.Period(e.ended_period, freq="M")
                except Exception as exc:
                    raise ValueError(
                        f"RouteEdit ended_period {e.ended_period!r} is not a "
                        f"YYYY-MM month: {exc}"
                    ) from exc
                if end < start:
                    raise ValueError(
                        f"RouteEdit ended_period {end} is before started_period {start}."
                    )
                if end > COVERED_TO:
                    raise ValueError(
                        f"RouteEdit ended_period {end} must be at most {COVERED_TO}."
                    )

    @staticmethod
    def _market_mult(scenario: Scenario, markets) -> np.ndarray:
        mult = np.full(len(markets), scenario.seats_mult, dtype=float)
        keys = markets.to_numpy()
        for mkt, v in scenario.market_seats_mult:
            mult[keys == mkt] *= v
        return mult

    @staticmethod
    def _market_lf_mult(scenario: Scenario, markets) -> np.ndarray:
        mult = np.ones(len(markets), dtype=float)
        keys = markets.to_numpy()
        for mkt, v in scenario.market_lf_mult:
            mult[keys == mkt] *= v
        return mult

    @staticmethod
    def _slice_markets(frame, markets):
        if markets != "all":
            wanted = {markets} if isinstance(markets, str) else set(markets)
            missing = wanted - set(frame["market"].unique())
            if missing:
                raise ValueError(f"unknown market(s): {sorted(missing)}")
            return frame[frame["market"].isin(wanted)]
        return frame