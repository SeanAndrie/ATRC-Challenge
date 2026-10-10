# ATRC Flight → Hotel Demand Simulator

Run from the repository root after `uv sync`:

```bash
.venv/bin/streamlit run app/simulator.py
```

The app wraps the packaged model in `pipeline.simulator`. On startup it loads
the fitted cache from `outputs/models/simulator_full.pkl` when one exists
(instant) and otherwise fits the competition pack once (~1–2 minutes) and saves
the cache. The pickle carries the fitted boosters, chain parameters, the supply
cache and the model's categorical encodings, so a loaded app can score
scenarios without re-reading the pack.

## What it answers

The scenario builder maps the brief's questions onto levers:

| Question | How to express it |
| --- | --- |
| New route from a country that didn't fly here | Pick the closest **model market** as an analog, add the city/airline, set weekly seats, and name the origin in the label column — the estimate is read from the analog's behaviour and labelled as such |
| Airline runs more (or fewer) flights per week | Signed **Seats/week Δ** = ±Δ flights × seats per flight on an existing route, with optional start/end months |
| Airlines add seats via larger aircraft | Same row: **Seats/week Δ** = flights/week × Δ seats per flight |
| A higher/lower share of seats gets filled | Global **Load-factor multiplier**, or a per-market **Load factor ×** row |
| The mix of countries shifts with the seasons | Per-market **Seats ×** row, or a route edit scoped to a season (e.g. 2026-11 → 2027-02) |

The Results tab shows the horizon delta in guest-nights and arrivals, monthly
and seasonal baseline-vs-scenario charts, the parametric chain stages beside the
hybrid, one-at-a-time contribution bars (which factor moves the number most),
and a supply-side-only market breakdown. The Route insights tab ranks markets by
hotel guests per 1,000 available seats and explores current weekly route supply.
The Validation tab replays the T5 competition window live. The Scenario library
saves runs to `outputs/sessions/` (parquet + JSON) with downloads and a
two-scenario comparison.

## Read this before trusting a number

- **Market rows are supply-side-only.** The pack's flight market is the last
  departure country before AUH; the hotel market is guest nationality. The join
  is structurally wrong (docs/METHODS.md §4.5) and capture exceeds 1.0 in five
  markets, so market-level levels are not measured origin demand. Aggregate
  results are the safe read.
- **New origins are analog estimates.** The model's market categorical has 33
  fixed levels; a genuinely new country is modelled on a chosen analog market.
- **Occupancy is not computed.** The pack has no room-capacity or room-nights
  data, so winter impact is reported as guest-nights lost/gained. Adding hotel
  capacity data would turn that into occupancy points.
- **The baseline beyond 2026-02 is held forward, not forecast.** Each future
  month repeats the same calendar month one year earlier, so seasonality carries
  but no new airline schedule is invented. Read those months as scenario
  differences.
- **Market attribution caveats aside**, the hybrid is validated: T5 WMAPE
  20.55% against 32.77% for the parametric chain, with −0.14% aggregate bias.

## Chatbot endpoint (`/chatbot`)

`app/pages/chatbot.py` is a second page served at `/chatbot` (Streamlit
multipage; the simulator stays at `/`). It is a full agent over
[OpenRouter](https://openrouter.ai/) whose tools call the real fitted model:

- `run_scenario` — scores a what-if end to end (new routes, ± weekly flights,
  aircraft/seat changes, load-factor and seasonal-mix levers) and returns the
  guest-night delta, the changed window and the top market movers;
- `get_market_ranking` — hotel guests per 1,000 seats, capture, ALOS per market;
- `get_route_supply` — current weekly seats/flights per route, for sizing edits;
- `get_model_facts` — pinned validation numbers, horizon, projection rule,
  market list and caveats.

Every number in an answer comes from a tool result; the system prompt forbids
the model from inventing figures and carries the attribution caveats, so the
chat answers new-origin questions as labelled analog estimates and reports
winter impact in guest-nights.

**Configuration** (first non-empty wins): sidebar fields → `st.secrets` →
environment variables → repo-root `.env`. Copy `.env.example` to `.env` and
fill in `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` (default
`openai/gpt-4o-mini`, changeable in the sidebar, which can also save both to
`.env`). `.env` is gitignored.

## Deploy for a short demo

The app is prepared for [Streamlit Community Cloud](https://share.streamlit.io/),
which provides free Streamlit hosting. Vercel is designed around short-lived web
functions and is not a suitable host for Streamlit's persistent server/WebSocket
process.

1. Fit once locally so `outputs/models/simulator_full.pkl` exists, then commit
   that pickle plus the app source and `app/requirements.txt` to the repository
   branch Community Cloud can access. `outputs/` is gitignored by default; the
   pickle is the one artifact the hosted app needs, because the competition pack
   under `data/raw/competition/` is licensed for competition use only and must
   not be published.
2. In Community Cloud, create an app from that repository and set the entrypoint
   to `app/simulator.py`.
3. In **Advanced settings**, select Python 3.12 (or newer) and point
   dependencies at `app/requirements.txt`.
4. Deploy. Since the GitHub repository is private, the app will also be private
   by default; keep it private to stay within the competition data licence.
5. Share the generated `*.streamlit.app` URL with the judges.

The free service sleeps after 12 hours without traffic; opening the app wakes it
again. Saved scenario runs live on the app's local filesystem and may be lost if
the hosted instance is rebuilt or restarted; each saved run has a download
button.
