"""OpenRouter-backed chatbot agent grounded in the packaged simulator.

The agent answers open-ended questions about flight-to-hotel demand by calling
tools that run the real model: ``run_scenario`` scores what-ifs end to end,
the other tools expose measured market rankings, route-level supply and the
pinned validation facts. Numbers are never taken from the language model's own
imagination -- every quantitative claim in an answer must come from a tool
result.

Configuration resolution order (first non-empty wins):

1. in-app sidebar overrides (session state, passed in explicitly)
2. ``st.secrets`` (Streamlit Cloud)
3. environment variables (``OPENROUTER_API_KEY``, ``OPENROUTER_MODEL``)
4. a repo-root ``.env`` file (gitignored; template in ``.env.example``)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from pipeline import paths  # noqa: E402
from pipeline.calendar_events import COVERED_FROM, COVERED_TO  # noqa: E402
from pipeline.simulator import RouteEdit, Scenario, Simulator  # noqa: E402
from app.shared import baseline_frames  # noqa: E402

DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_START = "2026-03"  # first month without observed supply
OBSERVED_TO = "2026-02"
MAX_TURNS = 6

LEVERS = (
    "seats_mult", "lf_mult", "p2p_mult",
    "capture_mult", "alos_mult", "season_mult",
)

# Pinned by pipeline/tests/test_simulator_package.py (T5 competition window).
PINNED_T5_WMAPE = 20.55
PINNED_T5_BIAS = -0.14
PINNED_CHAIN_WMAPE = 32.77


class OpenRouterError(RuntimeError):
    """Raised for any OpenRouter transport/API failure worth showing the user."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser: ``#`` comments, blank lines, stripped quotes."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:  # an empty value means "not set"
            out[key] = value
    return out


def _secret(key: str) -> str:
    try:
        value = st.secrets.get(key)
    except Exception:
        return ""
    return str(value) if value else ""


def resolve_config(overrides: dict[str, str] | None = None) -> tuple[dict, str]:
    """Return ``({"api_key": ..., "model": ...}, source_description)``.

    ``overrides`` comes from the sidebar widgets; empty strings mean "not set"
    and fall through to secrets, environment, then the .env file.
    """
    overrides = overrides or {}
    file_env = parse_env_file(paths.ROOT / ".env")

    def pick(key: str) -> tuple[str, str]:
        if overrides.get(key):
            return overrides[key], "sidebar"
        if _secret(key):
            return _secret(key), "streamlit secrets"
        if os.environ.get(key):
            return os.environ[key], "environment"
        if file_env.get(key):
            return file_env[key], ".env file"
        return "", "not set"

    api_key, key_src = pick("OPENROUTER_API_KEY")
    model, model_src = pick("OPENROUTER_MODEL")
    if not model:
        model, model_src = DEFAULT_MODEL, "built-in default"
    return (
        {"api_key": api_key, "model": model},
        f"key: {key_src} · model: {model_src}",
    )


def save_to_env_file(api_key: str, model: str) -> Path:
    """Write/update the repo-root ``.env``, preserving other lines."""
    target = paths.ROOT / ".env"
    wanted = {"OPENROUTER_API_KEY": api_key, "OPENROUTER_MODEL": model}
    out: list[str] = []
    seen: set[str] = set()
    if target.exists():
        for line in target.read_text().splitlines():
            stripped = line.strip()
            key = (
                stripped.split("=", 1)[0].strip()
                if "=" in stripped and not stripped.startswith("#")
                else None
            )
            if key in wanted:
                out.append(f"{key}={wanted[key]}")
                seen.add(key)
            else:
                out.append(line)
    for key, value in wanted.items():
        if key not in seen:
            out.append(f"{key}={value}")
    target.write_text("\n".join(out) + "\n")
    return target


# ---------------------------------------------------------------------------
# OpenRouter transport
# ---------------------------------------------------------------------------
def chat(cfg: dict, messages: list[dict], tools: list[dict] | None = None,
         temperature: float = 0.2, timeout: int = 120) -> dict:
    """One chat-completions call; returns the assistant message dict."""
    body: dict = {"model": cfg["model"], "messages": messages, "temperature": temperature}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/SeanAndrie/ATRC-Challenge",
                "X-Title": "ATRC Flight-to-Hotel Simulator",
            },
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise OpenRouterError(f"request failed: {exc}") from exc
    if resp.status_code == 401:
        raise OpenRouterError(
            "rejected the API key (401). Check the key in the sidebar or .env."
        )
    if resp.status_code != 200:
        raise OpenRouterError(f"error {resp.status_code}: {resp.text[:300]}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise OpenRouterError(f"non-JSON response: {resp.text[:200]}") from exc
    if not data.get("choices"):
        raise OpenRouterError(f"response missing choices: {json.dumps(data)[:300]}")
    return data["choices"][0]["message"]


# ---------------------------------------------------------------------------
# Tools: the agent's hands on the real model
# ---------------------------------------------------------------------------
def _tool(name: str, description: str, properties: dict, required: list[str] | None = None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


TOOLS = [
    _tool(
        "run_scenario",
        "Score a what-if scenario against the no-change baseline with the fitted "
        "hybrid model and return hotel-demand deltas. Use for questions like "
        "'how many extra hotel nights would X generate?'. Supply edits are "
        "seats/week deltas; reductions are negative numbers.",
        {
            "seats_mult": {"type": "number", "description": "Global seats multiplier, default 1.0."},
            "lf_mult": {"type": "number", "description": "Global load-factor multiplier, default 1.0."},
            "p2p_mult": {"type": "number", "description": "Global point-to-point share multiplier, default 1.0."},
            "capture_mult": {"type": "number", "description": "Post-hoc conversion dial, default 1.0."},
            "alos_mult": {"type": "number", "description": "Post-hoc length-of-stay dial, default 1.0."},
            "season_mult": {"type": "number", "description": "Post-hoc seasonality dial, default 1.0."},
            "market_seats_mult": {
                "type": "array",
                "description": "Per-market seat multipliers as [market, multiplier] pairs, e.g. [[\"INDIA\", 1.15]].",
                "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {}},
            },
            "market_lf_mult": {
                "type": "array",
                "description": "Per-market load-factor multipliers as [market, multiplier] pairs.",
                "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {}},
            },
            "routes": {
                "type": "array",
                "description": "Route-level supply edits over an inclusive month range.",
                "items": {
                    "type": "object",
                    "properties": {
                        "market": {"type": "string", "description": "One of the 33 model markets; for a brand-new origin use the closest analog market and set label to the new country."},
                        "city": {"type": "string", "description": "Departure city, e.g. Mumbai."},
                        "airline": {"type": "string", "description": "Operating airline, e.g. Etihad."},
                        "seats_per_week": {"type": "number", "description": "Signed weekly seat delta: positive adds supply, negative trims it."},
                        "started_period": {"type": "string", "description": "Start month YYYY-MM; default " + DEFAULT_START + "."},
                        "ended_period": {"type": "string", "description": "Inclusive end month YYYY-MM; omit to run to the horizon (2028-12)."},
                        "label": {"type": "string", "description": "Display name for a new origin modelled on this analog market, e.g. KENYA."},
                    },
                    "required": ["market", "seats_per_week"],
                },
            },
        },
    ),
    _tool(
        "get_market_ranking",
        "Rank source markets by measured hotel demand per available seat on the "
        "matched training panel. Use for 'which countries bring in the most "
        "hotel guests per available seat?'.",
        {
            "limit": {"type": "integer", "description": "How many markets to return, default 10."},
            "sort_by": {
                "type": "string",
                "enum": ["guests_per_1000_seats", "guests", "new_arrivals", "capture", "alos"],
                "description": "Ranking metric, default guests_per_1000_seats.",
            },
        },
    ),
    _tool(
        "get_route_supply",
        "Current route-level supply (mean weekly seats and flights over the last "
        "three observed months), for sizing scenario edits. Omit market for a "
        "per-market summary.",
        {"market": {"type": "string", "description": "Optional: restrict to one of the 33 model markets."}},
    ),
    _tool(
        "get_model_facts",
        "Model validation facts, horizon, projection rule, market list and "
        "caveats. Call before making claims about accuracy or limitations.",
        {},
    ),
]


def _lever(args: dict, key: str) -> float:
    value = args.get(key)
    return 1.0 if value is None else float(value)


def _build_scenario(args: dict) -> Scenario:
    routes = []
    for route in args.get("routes") or []:
        routes.append(
            RouteEdit(
                market=str(route["market"]),
                city=str(route.get("city") or "New route"),
                airline=str(route.get("airline") or "New carrier"),
                seats_per_week=float(route["seats_per_week"]),
                started_period=str(route.get("started_period") or DEFAULT_START),
                ended_period=str(route["ended_period"]) if route.get("ended_period") else None,
                label=str(route["label"]) if route.get("label") else None,
            )
        )
    return Scenario(
        **{key: _lever(args, key) for key in LEVERS},
        market_seats_mult=tuple(
            (str(market), float(mult)) for market, mult in args.get("market_seats_mult") or []
        ),
        market_lf_mult=tuple(
            (str(market), float(mult)) for market, mult in args.get("market_lf_mult") or []
        ),
        routes=tuple(routes),
    )


def _tool_run_scenario(sim: Simulator, args: dict) -> str:
    scenario = _build_scenario(args)
    sim.validate_scenario(scenario)
    baseline_monthly, baseline_market, _ = baseline_frames(sim)
    scenario_monthly = sim.monthly(scenario)
    scenario_market = sim.market(scenario)

    base_nights = float(baseline_monthly["GuestNightsHat"].sum())
    base_arrivals = float(baseline_monthly["NewArrivalHat"].sum())
    delta_nights = float(scenario_monthly["GuestNightsHat"].sum()) - base_nights
    delta_arrivals = float(scenario_monthly["NewArrivalHat"].sum()) - base_arrivals

    # The scenario is scored over the full horizon, but most scenarios only
    # move a window of months (a winter-only edit, say). Report the changed
    # window separately so per-month averages mean what they claim.
    monthly_cmp = baseline_monthly[["ym_str", "GuestNightsHat"]].merge(
        scenario_monthly[["ym_str", "GuestNightsHat"]],
        on="ym_str",
        suffixes=("_base", "_scen"),
    )
    monthly_cmp["delta"] = (
        monthly_cmp["GuestNightsHat_scen"] - monthly_cmp["GuestNightsHat_base"]
    )
    changed = monthly_cmp[monthly_cmp["delta"].abs() > 1e-6]
    changed_months = int(len(changed))
    changed_window = (
        {"from": str(changed["ym_str"].iloc[0]), "to": str(changed["ym_str"].iloc[-1])}
        if changed_months
        else None
    )

    movers = (
        baseline_market.groupby("market")["GuestNightsHat"].sum().rename("base")
        .to_frame()
        .join(scenario_market.groupby("market")["GuestNightsHat"].sum().rename("scen"))
        .fillna(0.0)
    )
    movers["delta"] = movers["scen"] - movers["base"]
    top = (
        movers.sort_values("delta", key=lambda s: s.abs(), ascending=False)
        .head(5)
        .reset_index()
    )
    new_origin_map = {e.market: e.label for e in scenario.routes if e.label}
    return json.dumps(
        {
            "delta_guest_nights": round(delta_nights, 1),
            "delta_new_arrivals": round(delta_arrivals, 1),
            "pct_guest_nights": round(100.0 * delta_nights / base_nights, 2) if base_nights else 0.0,
            "avg_monthly_delta_guest_nights": (
                round(delta_nights / changed_months, 1) if changed_months else 0.0
            ),
            "baseline_guest_nights": round(base_nights, 1),
            "scenario_guest_nights": round(base_nights + delta_nights, 1),
            "months_scored": int(len(monthly_cmp)),
            "changed_months": changed_months,
            "changed_window": changed_window,
            "top_market_movers": [
                {
                    "market": (
                        f"{row.market} <- analog for {new_origin_map[row.market]}"
                        if row.market in new_origin_map
                        else row.market
                    ),
                    "delta_guest_nights": round(float(row.delta), 1),
                }
                for row in top.itertuples()
            ],
            "note": (
                "Deltas are vs the no-change baseline over the same horizon "
                "(observed supply to 2026-02, held-forward projection after). "
                "Market attribution is supply-side-only: departure country != "
                "guest nationality. A labelled origin is an analog estimate."
            ),
        }
    )


def _tool_market_ranking(sim: Simulator, args: dict) -> str:
    panel = sim.panel
    by_market = (
        panel.groupby("market")
        .agg(
            guests=("Guests", "sum"),
            new_arrivals=("NewArr", "sum"),
            p2p=("Total_P2P", "sum"),
            seats=("Total_Seats", "sum"),
        )
        .reset_index()
    )
    by_market["guests_per_1000_seats"] = 1000.0 * by_market["guests"] / by_market["seats"]
    by_market["capture"] = by_market["new_arrivals"] / by_market["p2p"]
    by_market["alos"] = by_market["guests"] / by_market["new_arrivals"]
    sort_by = args.get("sort_by") or "guests_per_1000_seats"
    limit = int(args.get("limit") or 10)
    ranked = by_market.sort_values(sort_by, ascending=False).head(limit)
    return json.dumps(
        {
            "rows": ranked.round(3).to_dict(orient="records"),
            "note": (
                "Measured on the matched training panel (2022-01..2025-07). The "
                "market key is departure country, so this is supply-side "
                "attribution, not guest nationality. Capture > 1.0 in five "
                "markets is the known join defect."
            ),
        }
    )


def _tool_route_supply(sim: Simulator, args: dict) -> str:
    routes = sim.route_supply()
    observed = routes[routes["basis"] == "observed"]
    recent_months = sorted(observed["ym"].unique())[-3:]
    recent = observed[observed["ym"].isin(recent_months)].copy()
    recent["weekly_seats"] = recent["Total_Seats"] * 7.0 / recent["ym"].dt.days_in_month
    market = args.get("market")
    if market:
        market = str(market).upper().strip()
        recent = recent[recent["market"] == market]
        if recent.empty:
            return json.dumps(
                {"error": f"market {market!r} not found; call get_model_facts for the list."}
            )
        grouped = (
            recent.groupby(["market", "city", "airline"])
            .agg(weekly_seats=("weekly_seats", "mean"), weekly_flights=("flights_per_week", "mean"))
            .reset_index()
            .sort_values("weekly_seats", ascending=False)
            .head(15)
        )
    else:
        grouped = (
            recent.groupby("market")
            .agg(weekly_seats=("weekly_seats", "sum"), routes=("city", "nunique"))
            .reset_index()
            .sort_values("weekly_seats", ascending=False)
        )
    return json.dumps(
        {
            "rows": grouped.round(1).to_dict(orient="records"),
            "months_used": [str(m) for m in recent_months],
        }
    )


def _tool_model_facts(sim: Simulator) -> str:
    return json.dumps(
        {
            "model": "Poisson LightGBM hybrid (5 fixed seeds) over the parametric chain; exposure offset log(P2P).",
            "validation_pinned": {
                "t5_wmape_new_arrivals_pct": PINNED_T5_WMAPE,
                "t5_aggregate_bias_pct": PINNED_T5_BIAS,
                "parametric_chain_wmape_pct": PINNED_CHAIN_WMAPE,
                "test_window": "2025-08..2026-02",
            },
            "horizon": {
                "from": str(COVERED_FROM),
                "to": str(COVERED_TO),
                "observed_supply_to": OBSERVED_TO,
                "projection_rule": "months after 2026-02 repeat the same calendar month one year earlier",
            },
            "global_alos_nights_per_arrival": round(float(sim.params["global_alos"]), 3),
            "panel": {"market_months": int(len(sim.panel)), "markets": int(sim.panel["market"].nunique())},
            "flight_markets": sorted(sim.fa["market"].unique().tolist()),
            "caveats": [
                "Market rows attribute demand by departure country, not guest nationality (METHODS 4.5); aggregate is the valid read.",
                "A brand-new origin must be modelled on an analog market with a label; its delta is an analog estimate.",
                "Occupancy is not computable: the pack has no room-capacity data. Report impact in guest-nights.",
                "Capture exceeds 1.0 in five markets (CHINA worst) - the visible sign of the join defect.",
                "Purpose-of-travel / VFR split is unmeasured and absorbed in capture.",
            ],
        }
    )


def execute_tool(sim: Simulator, name: str, args: dict) -> str:
    """Dispatch a tool call; errors come back as JSON so the agent can recover."""
    try:
        if name == "run_scenario":
            return _tool_run_scenario(sim, args)
        if name == "get_market_ranking":
            return _tool_market_ranking(sim, args)
        if name == "get_route_supply":
            return _tool_route_supply(sim, args)
        if name == "get_model_facts":
            return _tool_model_facts(sim)
        return json.dumps({"error": f"unknown tool {name!r}"})
    except ValueError as exc:
        return json.dumps({"error": f"invalid scenario: {exc}"})
    except Exception as exc:  # keep the agent alive; surface the failure
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def system_prompt(sim: Simulator) -> str:
    markets = ", ".join(sorted(sim.fa["market"].unique()))
    return f"""You are the analyst inside the ATRC Flight-to-Hotel Demand Simulator for Abu Dhabi (a DCT Abu Dhabi competition prototype).

WHAT YOU CAN DO
- Score real what-if scenarios with the `run_scenario` tool: new routes, more/fewer weekly flights (seats/week deltas), larger aircraft, load-factor changes, seasonal market-mix shifts.
- Answer "which markets are most efficient" questions with `get_market_ranking`.
- Size edits correctly with `get_route_supply` (current weekly seats and flights per route).
- Ground accuracy and limitation claims with `get_model_facts`.

HARD RULES
- Every number in your answer must come from a tool result. Never invent or extrapolate figures.
- The model horizon is {COVERED_FROM}..{COVERED_TO}. Observed flight supply ends {OBSERVED_TO}; later months hold forward the same calendar month one year earlier. Present post-{OBSERVED_TO} results as scenario differences vs that held-forward baseline, not forecasts.
- The 33 model markets are: {markets}.
- A country that never flew here cannot be encoded: pick the closest analog market, set `label` to the new country's name, and tell the user the result is an analog estimate.
- Reductions are negative seats/week. Check the route's current weekly seats with `get_route_supply` first and never cut more than exists.
- Default scenario window: start {DEFAULT_START}, run to the horizon, unless the user names a season or window (e.g. winter = 2026-11..2027-02).
- Market-level rows are supply-side-only: flight market = last departure country before AUH, hotel market = guest nationality; the join is structurally wrong. Aggregate results are the valid read. Do not present market levels as measured origin demand.
- Occupancy is not computable (no room-capacity data): report impact in guest-nights, and say so if asked about occupancy.
- If a tool returns an error, fix the arguments and retry rather than guessing.

STYLE
- Concise markdown. Lead with the number the user asked for, then assumptions (analog market, window), then one-line caveats. Use round numbers; this is a planning tool, not an oracle.
"""


def run_agent(cfg: dict, sim: Simulator, messages: list[dict]) -> tuple[str, list[dict]]:
    """Run the tool loop until a final answer.

    ``messages`` must start with the system prompt; the caller appends the
    user turn. Tool exchanges stay inside this loop (they are not persisted),
    so each turn keeps the context small. Returns ``(answer, tool_trace)``.
    """
    protocol = list(messages)
    trace: list[dict] = []
    for _ in range(MAX_TURNS):
        msg = chat(cfg, protocol, tools=TOOLS)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return (msg.get("content") or "", trace)
        protocol.append(
            {"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls}
        )
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(sim, name, args)
            trace.append({"tool": name, "args": args, "result_head": result[:500]})
            protocol.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})
    return (
        "I stopped after several tool rounds without a final answer. "
        "Try narrowing the question.",
        trace,
    )
