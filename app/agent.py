"""The chatbot: an OpenRouter tool-calling agent over the same scenario builders
the pages use, so every number it quotes comes from the model.

Configuration, first non-empty wins: the page's sidebar, ``st.secrets``,
environment variables, then a repo-root ``.env`` (template: ``.env.example``).
"""

from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

from app import core
from pipeline import paths

DEFAULT_MODEL = "openai/gpt-4o-mini"
URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TURNS = 6
OCC_GUESTS_PER_ROOM = 1.7


class AgentError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Configuration and transport
# ---------------------------------------------------------------------------
def _dotenv() -> dict[str, str]:
    path = paths.ROOT / ".env"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip().strip("\"'"):
                out[k.strip()] = v.strip().strip("\"'")
    return out


def config(key_override: str = "", model_override: str = "") -> dict[str, str]:
    def pick(name: str, override: str) -> str:
        try:
            secret = st.secrets.get(name, "")
        except Exception:
            secret = ""
        return override or secret or os.environ.get(name, "") or _dotenv().get(name, "")

    return {
        "api_key": pick("OPENROUTER_API_KEY", key_override),
        "model": pick("OPENROUTER_MODEL", model_override) or DEFAULT_MODEL,
    }


def _chat(cfg: dict, messages: list[dict]) -> dict:
    try:
        r = requests.post(
            URL,
            headers={"Authorization": f"Bearer {cfg['api_key']}", "X-Title": "ATRC Flight-to-Hotel Simulator"},
            json={"model": cfg["model"], "messages": messages, "tools": TOOLS,
                  "tool_choice": "auto", "temperature": 0.2},
            timeout=120,
        )
    except requests.RequestException as exc:
        raise AgentError(f"Could not reach OpenRouter: {exc}") from exc
    if r.status_code == 401:
        raise AgentError("OpenRouter rejected the API key.")
    if r.status_code != 200:
        raise AgentError(f"OpenRouter error {r.status_code}: {r.text[:300]}")
    choices = r.json().get("choices")
    if not choices:
        raise AgentError("OpenRouter returned no answer.")
    return choices[0]["message"]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def _fn(name: str, description: str, props: dict, required: list[str] = ()) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


S = {"type": "string"}
N = {"type": "number"}
MONTH = {"type": "string", "description": "Month YYYY-MM, between 2026-03 and 2028-12."}

TOOLS = [
    _fn("new_route", "A new route opens: extra weekly flights from a country. For a country with no "
        "flights in the data, set market to the closest analog market and new_country to its name.",
        {"market": S, "flights_per_week": N, "seats_per_flight": {**N, "description": "Default 290."},
         "start": MONTH, "months": {**N, "description": "Months operated, default 12."},
         "new_country": {**S, "description": "Name of a country not in the data, priced on market."}},
        ["market", "flights_per_week"]),
    _fn("change_frequency", "An airline runs more (positive) or fewer (negative) weekly flights on an "
        "existing route over a period. Seat size comes from the route. Omit city/airline for the "
        "market's busiest route.",
        {"market": S, "delta_flights_per_week": N, "city": S, "airline": S, "start": MONTH, "end": MONTH},
        ["market", "delta_flights_per_week", "start", "end"]),
    _fn("change_seats", "Airlines add (or remove) seats. Either larger aircraft on a route "
        "(delta_seats_per_flight, same frequency), or a seat multiplier for one market over a period, "
        "or for the whole network when market is omitted (then reported over the next 12 months).",
        {"market": S, "city": S, "airline": S, "delta_seats_per_flight": N, "seats_multiplier": N,
         "start": MONTH, "end": MONTH}),
    _fn("change_load_factor", "A higher or lower share of seats gets filled, network-wide or for one "
        "market. Load factor applies in every month; the result is reported over start..end "
        "(default the next 12 months).",
        {"multiplier": {**N, "description": "e.g. 1.05 for 5% fuller planes."}, "market": S,
         "start": MONTH, "end": MONTH}, ["multiplier"]),
    _fn("shift_market_mix", "The mix of countries shifts with the seasons: scale several markets' seats "
        "over a season, e.g. Russia 1.2 and India 0.9 in winter (2026-11..2027-02).",
        {"shifts": {"type": "array", "items": {"type": "object", "properties": {
            "market": S, "multiplier": N}, "required": ["market", "multiplier"]}},
         "start": MONTH, "end": MONTH}, ["shifts", "start", "end"]),
    _fn("rank_markets", "Measured hotel arrivals and guest-nights per 1,000 seats, capture and stay "
        "length by market (2022-01..2025-07).",
        {"sort_by": {"type": "string", "enum": ["arrivals_per_1000_seats", "nights_per_1000_seats",
                                                 "capture", "alos", "seats"]},
         "limit": N}),
    _fn("route_supply", "Current routes for a market: weekly flights, seats and seats per flight.",
        {"market": S}, ["market"]),
    _fn("model_facts", "Validation results, horizon and limitations of the model.", {}),
]


def _result(scenario: core.Scenario, window, notes: list[str]) -> tuple[str, pd.DataFrame]:
    s = core.summarize(scenario, window)
    out = s.to_dict()
    if s.window:
        rooms, src = core.hotel_rooms()
        out["indicative_occupancy_points"] = round(
            core.occupancy_points(s.delta_nights, s.window, rooms, OCC_GUESTS_PER_ROOM), 2)
        out["occupancy_assumptions"] = f"{rooms:,} rooms ({src}), {OCC_GUESTS_PER_ROOM} guests per room"
    if notes:
        out["notes"] = notes
    return json.dumps(out), s.monthly


def _market(name: str | None) -> str | None:
    return name.upper().strip() if name else None


def run_tool(name: str, a: dict) -> tuple[str, pd.DataFrame | None]:
    """Execute one tool call; errors return as JSON so the model can correct itself."""
    fwd = core.FORWARD_YEAR
    try:
        if name == "new_route":
            sc, notes = core.new_route(_market(a["market"]), a["flights_per_week"],
                                       a.get("seats_per_flight") or 290, a.get("start") or "2026-11",
                                       int(a.get("months") or 12), label=a.get("new_country"))
            return _result(sc, None, notes)
        if name == "change_frequency":
            sc, notes = core.frequency_change(_market(a["market"]), a["delta_flights_per_week"],
                                              a["start"], a["end"], a.get("city"), a.get("airline"))
            return _result(sc, (a["start"], a["end"]), notes)
        if name == "change_seats":
            start, end = a.get("start") or fwd[0], a.get("end") or fwd[1]
            if a.get("delta_seats_per_flight"):
                sc, notes = core.gauge_change(_market(a["market"]), a["delta_seats_per_flight"],
                                              start, end, a.get("city"), a.get("airline"))
                return _result(sc, (start, end), notes)
            mult = float(a.get("seats_multiplier") or 1.0)
            if a.get("market"):
                sc, notes = core.seasonal_mix({_market(a["market"]): mult}, start, end)
                return _result(sc, (start, end), notes)
            return _result(core.Scenario(seats_mult=mult), fwd, [])
        if name == "change_load_factor":
            mult = float(a["multiplier"])
            sc = (core.Scenario(market_lf_mult=((_market(a["market"]), mult),)) if a.get("market")
                  else core.Scenario(lf_mult=mult))
            return _result(sc, (a.get("start") or fwd[0], a.get("end") or fwd[1]), [])
        if name == "shift_market_mix":
            sc, notes = core.seasonal_mix({_market(x["market"]): x["multiplier"] for x in a["shifts"]},
                                          a["start"], a["end"])
            return _result(sc, (a["start"], a["end"]), notes)
        if name == "rank_markets":
            y = core.market_yield().sort_values(a.get("sort_by") or "arrivals_per_1000_seats",
                                                ascending=False).head(int(a.get("limit") or 10))
            return json.dumps({"rows": y.round(2).to_dict("records"), "note":
                               "capture > 1 means the flight-hotel link for that market is inconsistent "
                               "(connecting traffic of that nationality is filed under other countries)."}), None
        if name == "route_supply":
            t = core.route_table()
            t = t[t["market"] == _market(a["market"])]
            if t.empty:
                return json.dumps({"error": f"No current routes for {a['market']}. Markets: {core.markets()}"}), None
            return json.dumps({"routes": t.head(12).round(1).to_dict("records")}), None
        if name == "model_facts":
            return json.dumps({
                "validation": {"heldout_window": "2025-08..2026-02", "hybrid_wmape_pct": core.PINNED["t5_wmape"],
                               "national_bias_pct": core.PINNED["t5_bias"],
                               "fixed_ratio_chain_wmape_pct": core.PINNED["chain_wmape"],
                               "guest_nights_wmape_pct": core.PINNED["t4_nights_wmape"]},
                "horizon": f"{core.COVERED_FROM}..{core.HORIZON_END}; observed flights to {core.OBSERVED_TO}, "
                           "later months repeat the same month a year earlier",
                "markets": core.markets(),
                "limits": ["market splits are supply-side (departure country, not nationality)",
                           "countries never flown are analog estimates",
                           "occupancy needs assumed room stock and guests per room"],
            }), None
        return json.dumps({"error": f"unknown tool {name}"}), None
    except (KeyError, TypeError) as exc:
        return json.dumps({"error": f"missing or bad argument: {exc}"}), None
    except ValueError as exc:
        return json.dumps({"error": str(exc)}), None


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def system_prompt() -> str:
    return f"""You are the analyst inside the Abu Dhabi Flight → Hotel Demand Simulator (DCT Abu Dhabi, ATRC challenge).
You answer what-if questions about flights and hotel demand by calling tools that run a validated model.

Map questions to tools:
- a new route, including from a country that has never flown here → new_route (pick the closest analog market for a new country; say which and why)
- an airline running more or fewer weekly flights → change_frequency (check route_supply first if the route is unclear)
- more seats via larger aircraft or extra capacity → change_seats
- a higher or lower share of seats filled → change_load_factor
- the mix of countries shifting with the seasons → shift_market_mix
- which countries yield the most hotel guests per seat → rank_markets

Rules:
- Every number you state must come from a tool result. Never estimate numbers yourself.
- Markets are upper-case names from this list: {", ".join(core.markets())}.
- Months are YYYY-MM between {core.FIRST_FORWARD} and {core.HORIZON_END}. "Winter" means {core.WINTER[0]}..{core.WINTER[1]}; "next year" means {core.FORWARD_YEAR[0]}..{core.FORWARD_YEAR[1]}.
- After {core.OBSERVED_TO} the baseline repeats last year's schedule: describe results as the effect of the change, not as a forecast.
- Market-level movers are supply-side only (flight market = last departure country; hotel market = nationality). Lead with national totals.
- Occupancy is indicative and rests on the stated room and guests-per-room assumptions; say so when you quote it.
- If a tool returns an error, fix the arguments and retry.

Style: short markdown. Lead with the headline number (guest-nights, then hotel arrivals), then the window and assumptions in one line, then at most one caveat. Round sensibly (e.g. "≈ 35,000")."""


def run(cfg: dict, history: list[dict]) -> tuple[str, list[dict]]:
    """Answer the last user turn. Returns the answer and a trace of tool calls,
    each with its arguments, result and (for scenarios) the monthly delta frame."""
    messages = [{"role": "system", "content": system_prompt()}, *history]
    trace: list[dict] = []
    for _ in range(MAX_TURNS):
        msg = _chat(cfg, messages)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content") or "", trace
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
        for call in calls:
            fn = call.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            content, monthly = run_tool(fn.get("name", ""), args)
            trace.append({"tool": fn.get("name"), "args": args, "result": content, "monthly": monthly})
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": content})
    return "I couldn't finish within a few tool calls — try a narrower question.", trace
