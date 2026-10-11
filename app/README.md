# ATRC Flight → Hotel Demand Simulator (Streamlit app)

An interactive front end on the packaged model (`pipeline/simulator.py`). Judges and
planners can test flight changes and see the effect on Abu Dhabi hotel guest-nights.

## Run it

From the repository root, after `uv sync`:

```bash
.venv/bin/python -m app.build_model          # once; needs data/raw/competition/
.venv/bin/streamlit run app/streamlit_app.py
```

`build_model` fits the model and replays the back-tests. It writes two files to
`outputs/models/` (gitignored):

| File | What it holds |
| --- | --- |
| `simulator_full.pkl` | the fitted simulator (scenario model, back-test model, learned length of stay, supply cache) |
| `validation.parquet` | held-out predictions vs actuals for the Validation page |

With these two files present, the app starts in about a second and never reads the
licensed pack. Without the pickle, the app fits from the pack on first start (about
1–2 minutes). If the pack is also missing, it shows an error explaining what to ship.

## Pages

| Page | Answers |
| --- | --- |
| **Overview** | What the model does and how accurate it is. A Sankey diagram follows a year of seats to hotel arrivals. Shows live default answers to the three questions. |
| **New route** | *How many extra hotel nights would a new twice-weekly flight from a given country generate?* Pick a country (or a new one priced on an analog market), flights per week, aircraft, start month and duration. Shows guest-nights, arrivals and yield per seat, a monthly chart, and a funnel of where the seats go. A toggle ranks all 33 origins for the same service. |
| **Winter flight cut** | *How much would winter hotel occupancy suffer if an airline reduced its flights?* Pick a route and the number of weekly flights cut. Shows guest-nights lost, the share of the season's demand, baseline vs scenario, and an **indicative** occupancy change. |
| **Guests per seat** | *Which countries bring in the most hotel guests per available seat?* Measured ranking, conversion vs length-of-stay bubble chart, month-by-market heatmap, and a table. |
| **Scenario lab** | Combines network levers, route changes and seasonal market-mix shifts. Shows each change's contribution, sensitivity three ways ("which factors move the number most"): an equal ±10% shock, each lever moved by its measured real-world swing, and Sobol variance shares from `pipeline/sensitivity.py`, and a session scenario library with compare and download. |
| **Ask the model** | Chatbot for the brief's five what-if questions, with suggested prompts, an inline chart per scenario, and every model call shown. |
| **Validation** | Held-out back-test (Aug 2025 – Feb 2026): hybrid vs fixed-ratio chain vs actual, by month and by market-month, plus what the model can and cannot say. |

## Code layout

| File | Role |
| --- | --- |
| `streamlit_app.py` | Entry point: page config, navigation, sidebar |
| `core.py` | Model loading, cached scoring, `summarize()`, one scenario builder per brief question (`new_route`, `frequency_change`, `gauge_change`, `seasonal_mix`), supply and yield tables, occupancy conversion |
| `charts.py` | Plotly builders on one palette: grey baseline, blue scenario, blue/red for gains and losses |
| `agent.py` | OpenRouter client and the chatbot's tools, all thin wrappers over `core.py` |
| `ui.py` | Formatting and shared page snippets |
| `views/` | One file per page |
| `build_model.py` | Builds the artifacts above |

Pages and chatbot share `core.py`, so the same question gives the same number
everywhere. Nothing in `pipeline/` is modified by the app.

## Chatbot setup

The chatbot needs an [OpenRouter](https://openrouter.ai/) API key. The first
non-empty source wins:
1. the page's sidebar;
2. Streamlit secrets;
3. environment variables;
4. a repo-root `.env`.

For local use, create a `.env` file at the **repository root** (next to
`README.md`, not inside `app/`; it is gitignored):

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini   # optional; any tool-capable OpenRouter model
```

The same template is at the repository root as `.env.example`, so
`cp .env.example .env` from the repository root also works. Without a key, the
chat page explains how to add one, and every other page works normally.

| Question in the brief | Tool |
| --- | --- |
| A new route opens from a country that didn't fly here | `new_route` (closest analog market + `new_country` label) |
| An airline runs more or fewer flights per week | `change_frequency` |
| Airlines add seats through larger aircraft or extra flights | `change_seats` |
| A higher or lower share of seats gets filled | `change_load_factor` |
| The mix of countries shifts with the seasons | `shift_market_mix` |

Supporting tools: `rank_markets`, `route_supply`, `model_facts`. The system prompt
requires every number in an answer to come from a tool result.

## Deploy (Streamlit Community Cloud)

1. Create an app from the repository with entrypoint `app/streamlit_app.py`,
   Python 3.12 or newer, and dependencies from `app/requirements.txt`. The theme is in
   `.streamlit/config.toml` at the repository root.
2. Make the two `outputs/models/` files available to the deployed app. They are
   gitignored, and they contain data derived from the licensed competition pack, so
   keep any deployment private to the judges. The pack itself never needs to be
   deployed.
3. Add `OPENROUTER_API_KEY` (and optionally `OPENROUTER_MODEL`) under the app's
   **Secrets**.

Saved scenarios live in the browser session; use the download button to keep them.

## Read this before trusting a number

- **National totals are the reliable read.** On held-out months the model scored
  20.55% WMAPE on new hotel arrivals (fixed-ratio chain 32.77%), and the national
  total was within 0.14%.
- **Market splits are supply-side.** In the flight data a market is the last
  departure country; in the hotel data it is guest nationality. Connecting traffic
  therefore lands on hubs, and five markets show more hotel arrivals than
  terminating passengers (marked ⚠).
- **New flights are priced on capacity.** Added seats join the market's busiest
  existing route. The model's route and airline counts mostly encode market size, and
  counting one service as a new route inflates estimates several-fold. That variant
  is an opt-in toggle on the New route page.
- **New countries are analog estimates.** The model knows 33 markets, so a country
  that never flew here is priced on a market you choose, and labelled.
- **After 2026-02 the baseline repeats last year's schedule.** Results are the
  effect of a change, not a forecast.
- **Occupancy is indicative.** The pack has no room data. The app converts
  guest-nights using the published room stock (34,000, DCT 2024) and an editable
  guests-per-room assumption (default 1.7).
