"""Scenario simulator built on the canonical model in ``pipeline.hybrid``.

This is the integration point for the Streamlit front end. It exposes a fitted
parametric chain *and* the hybrid LightGBM layer behind one small class, so the
UI needs only::

    from pipeline.simulator import Simulator, Scenario
    sim = Simulator().fit()
    sim.monthly(Scenario(seats_mult=1.2))      # safe aggregate view
    sim.market(Scenario(seats_mult=1.2))       # market view, flagged supply-side

Two models are fitted lazily by ``fit()``:

* ``"full"`` -- the default scenario model. Iteration count is chosen on the
  internal last-6-month tail (the notebook's T4 protocol), then the 5-seed
  ensemble is refit on the whole training panel (2022-01..2025-07). Scenarios
  therefore fly the model forward over any month the pack covers
  (up to ``COVERED_TO`` = 2026-02).
* ``"competition"`` -- the T5 protocol (iterations on the leave-tail holdout).
  ``reproduce_competition_window`` replays it to surface the pinned numbers
  live.

``window`` selects which model ``predict`` uses on a scenario, default ``"full"``.

Lever semantics

* Supply levers (``seats_mult``, ``lf_mult``, ``p2p_mult``) act *pre-model*:
  they rewrite the supply frame, so the hybrid's features (``log_seats``,
  ``load_factor``, ``p2p_share``, ``routes``, ``airlines``) *and* its Poisson
  exposure offset respond non-linearly. That is the feature, not a bug.
* ``capture_mult``, ``season_mult``, ``alos_mult`` are post-hoc dials on the
  model output: they scale New Arrivals (capture, season) and guest-nights
  (ALOS) without touching the learned features. In the chain path the same
  three are passed to ``hybrid.simulate`` (whose ``season_mult`` is a documented
  no-op there) and ``season_mult`` is applied as a demand-level override
  afterwards, so both paths honour every dial.
* ``market_seats_mult`` scales a single market's seats on top of the global
  ``seats_mult`` (it is a per-market multiplier, not a cap).
* ``RouteEdit`` appends terminating supply for an existing market from a start
  month onward (a new route, or extra seats on an existing market), with PAX/P2P
  estimated at the market's average load factor and P2P share.

Attribution guard (docs/METHODS.md §4.5)

* Aggregate views (``monthly``, ``seasonal``) are the defaults and are valid.
* ``market`` carries ``frame.attrs["data_status"] == "supply-side-only"`` and a
  constant warning column; the UI must not present market levels as measured.
* Scenarios cannot invent new markets (the model's ``market`` categorical is a
  fixed 33-level encoding) and cannot start supply past ``COVERED_TO`` = 2026-02
  (the event calendar's horizon); both produce clear errors.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import hybrid as H  # noqa: E402
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


@dataclass(frozen=True)
class RouteEdit:
    """Add terminating supply for an existing market from ``started_period`` on.

    PAX and P2P are estimated at the market's average load factor and P2P share
    observed over the fit window, so the added seats convert like the market's
    history. Supply is applied monthly (``seats_per_week * days_in_month / 7``).
    """

    market: str
    city: str
    airline: str
    seats_per_week: float
    started_period: str  # e.g. "2025-09", within COVERED_FROM..COVERED_TO


@dataclass(frozen=True)
class Scenario:
    seats_mult: float = 1.0
    lf_mult: float = 1.0
    p2p_mult: float = 1.0
    capture_mult: float = 1.0
    alos_mult: float = 1.0
    season_mult: float = 1.0
    market_seats_mult: dict = field(default_factory=dict)
    routes: tuple = ()


class Simulator:
    """Fitted chain + hybrid, scenario prediction, and pinned reproduction."""

    def __init__(self, window: str = "full"):
        if window not in ("full", "competition"):
            raise ValueError(f"window must be 'full' or 'competition', got {window!r}")
        self.window = window
        self._fit_done = False
        self.fa = None
        self.panel = None
        self.test_panel = None
        self.params = None
        self._full_boosters = None
        self._comp_boosters = None
        self._best_iter_full = None
        self._best_iter_comp = None
        self._monthly = None

    # -- fitting -------------------------------------------------------------
    def fit(self) -> "Simulator":
        """Fit both models once. Idempotent; returns self for chaining."""
        if self._fit_done:
            return self

        fa, _mkt_dtype, _mon_dtype, panel, _test_m, test_panel, _train_m = H.build_panels()
        holdout = panel[panel["ym"] > "2025-01"].copy()

        # Full window: iterations on the internal last-6-month tail (T4 protocol).
        core_full = panel[panel["ym"] <= "2024-07"]
        val_full = panel[panel["ym"] > "2024-07"]
        self._best_iter_full = H.fit_poisson(core_full, val_full)
        self._full_boosters = H.fit_hybrid(panel, self._best_iter_full)

        # Competition window: iterations on the leave-tail holdout (T5 protocol),
        # ensemble refit on the full panel for maximum training signal.
        core_comp = panel[panel["ym"] <= "2025-01"]
        self._best_iter_comp = H.fit_poisson(core_comp, holdout)
        self._comp_boosters = H.fit_hybrid(panel, self._best_iter_comp)

        self.fa = fa
        self.panel = panel
        self.test_panel = test_panel
        self.params = H.fit_params(panel)
        self._monthly = cl.flights_monthly()
        self._fit_done = True
        return self

    def _require_fit(self):
        if not self._fit_done:
            raise RuntimeError("Simulator.fit() must be called before prediction")

    # -- scenario scoring ----------------------------------------------------
    def predict(self, scenario: Scenario, markets: str | list[str] | None = "all"):
        """Market-month predictions under ``scenario``.

        Returns a frame with ``ym_str, ym, market, month, routes, airlines,
        Total_Seats, PAX, P2P, NewArrivalHat, GuestNightsHat``. Market-level rows
        are attributed on the ``market`` join and are supply-side-only, so the
        frame carries ``attrs["data_status"] = "supply-side-only"``.
        """
        self._require_fit()
        boosters = self._full_boosters if self.window == "full" else self._comp_boosters
        scaled, _raw = self._scenario_supply(scenario)
        scaled = self._slice_markets(scaled, markets)

        pred = H.predict_hybrid(boosters, scaled)
        alos = scaled["market"].map(self.params["alos"]).to_numpy(dtype=float)
        new_arr = pred * scenario.capture_mult * scenario.season_mult
        out = scaled[
            ["ym_str", "ym", "market", "month", "routes", "airlines",
             "Total_Seats", "PAX", "P2P"]
        ].copy()
        out["NewArrivalHat"] = new_arr
        out["GuestNightsHat"] = new_arr * alos * scenario.alos_mult
        out.attrs["data_status"] = "supply-side-only"
        return out

    def chain(self, scenario: Scenario, markets: str | list[str] | None = "all"):
        """The parametric chain under the same scenario (South simulate exactly).

        Supply levers pass straight to ``hybrid.simulate``; ``season_mult`` is a
        demand-level override applied afterwards, because ``simulate``'s own
        ``season_mult`` is a documented no-op kept for notebook fidelity.
        """
        self._require_fit()
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
            season_mult=scenario.season_mult,
        )
        out["NewArrivalHat"] = out["NewArrivalHat"] * scenario.season_mult
        out["GuestNightsHat"] = out["GuestNightsHat"] * scenario.season_mult
        out.attrs["data_status"] = "supply-side-only"
        return out

    def monthly(self, scenario: Scenario):
        """Aggregate guest-nights by month -- the safe default view."""
        f = self.predict(scenario)
        g = f.groupby("ym").agg(
            P2P=("P2P", "sum"),
            NewArrivalHat=("NewArrivalHat", "sum"),
            GuestNightsHat=("GuestNightsHat", "sum"),
        )
        g["ym_str"] = g.index.astype(str)
        return g.reset_index(drop=True)

    def seasonal(self, scenario: Scenario):
        """Aggregate by calendar month (seasonal shape)."""
        f = self.predict(scenario)
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
        return self.predict(scenario)

    def native_capture(self) -> pd.Series:
        """Per-market chain capture (calibration coefficient), the model's view."""
        self._require_fit()
        return self.params["capture"].copy()

    def reproduce_competition_window(self):
        """Replay the T5 protocol and surface the pinned numbers live.

        Returns the test-panel predictions with ``wm_na`` (WMAPE of New Arrivals)
        and ``bias_pct`` on the attrs. The notebook Part C pins these near 20.8%
        and -0.1%; the test suite asserts them within tolerance.
        """
        self._require_fit()
        pred = H.predict_hybrid(self._comp_boosters, self.test_panel)
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
        out.attrs["best_iter"] = self._best_iter_comp
        return out

    # -- supply assembly -----------------------------------------------------
    def _scenario_supply(self, scenario: Scenario):
        """Build raw (route edits, no multipliers) and scaled supply frames.

        Both are at market x month grain with the feature columns the model and
        chain need. ``_scaled`` applies the supply multipliers, which rewrite
        ``Total_Seats`` and re-derive PAX/P2P so features and the exposure offset
        respond; ``_raw`` (used by ``chain``) carries the same seats/PAX/P2P, so
        ``Scenario()`` exactly reproduces ``fa``.
        """
        self._require_fit()
        rows = self._monthly[
            ["ym", "Departure Country Name", "Departure City", "Airline Name",
             "Total_Seats", "Total_PAX", "Total_P2P"]
        ].rename(
            columns={
                "Departure Country Name": "market",
                "Departure City": "city",
                "Airline Name": "airline",
            }
        ).copy()
        rows["market"] = rows["market"].str.upper().str.strip()
        rows["ym"] = pd.PeriodIndex(rows["ym"], freq="M")

        for e in scenario.routes:
            mkt = e.market.upper().strip()
            if mkt not in set(self.fa["market"]):
                raise ValueError(
                    f"RouteEdit market {e.market!r} is not one of the 33 "
                    "flight markets; new markets cannot be encoded by the model."
                )
            start = pd.Period(e.started_period, freq="M")
            if start < COVERED_FROM or start > COVERED_TO:
                raise ValueError(
                    f"RouteEdit started_period {start} must be within "
                    f"{COVERED_FROM}..{COVERED_TO}."
                )
            fa_m = self.fa[self.fa["market"] == mkt]
            lf = fa_m["Total_PAX"].sum() / max(fa_m["Total_Seats"].sum(), 1.0)
            share = fa_m["Total_P2P"].sum() / max(fa_m["Total_PAX"].sum(), 1.0)
            for offset in range(0, int((COVERED_TO - start).n) + 1):
                ym = start + offset
                seats = e.seats_per_week * ym.days_in_month / 7.0
                pax = seats * lf
                rows = pd.concat(
                    [
                        rows,
                        pd.DataFrame(
                            [{
                                "ym": ym, "market": mkt, "city": e.city,
                                "airline": e.airline, "Total_Seats": seats,
                                "Total_PAX": pax, "Total_P2P": pax * share,
                            }]
                        ),
                    ],
                    ignore_index=True,
                )

        agg = rows.groupby(["ym", "market"], dropna=False).agg(
            Total_Seats=("Total_Seats", "sum"),
            Total_PAX=("Total_PAX", "sum"),
            Total_P2P=("Total_P2P", "sum"),
            routes=("city", lambda s: s.nunique()),
            airlines=("airline", lambda s: s.nunique()),
        ).reset_index()
        agg = agg.merge(
            self.fa[["ym", "market", "Total_Transfer", "Total_Transit"]],
            on=["ym", "market"],
            how="left",
        ).fillna({"Total_Transfer": 0.0, "Total_Transit": 0.0})
        agg["ym_str"] = agg["ym"].astype(str)
        agg["month"] = agg["ym"].dt.month

        raw = agg[
            ["ym_str", "ym", "market", "month", "routes", "airlines",
             "Total_Seats", "Total_PAX", "Total_P2P",
             "Total_Transfer", "Total_Transit"]
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
        scaled = raw.copy()
        scaled["Total_Seats"] = raw["Total_Seats"] * cmult
        scaled["Total_PAX"] = scaled["Total_Seats"] * lf * scenario.lf_mult
        scaled["Total_P2P"] = scaled["Total_PAX"] * share * scenario.p2p_mult
        scaled["PAX"] = scaled["Total_PAX"]
        scaled["P2P"] = scaled["Total_P2P"]
        return scaled, raw

    @staticmethod
    def _market_mult(scenario: Scenario, markets) -> np.ndarray:
        mult = np.full(len(markets), scenario.seats_mult, dtype=float)
        keys = markets.to_numpy()
        for m, v in scenario.market_seats_mult.items():
            mult[keys == m] *= v
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