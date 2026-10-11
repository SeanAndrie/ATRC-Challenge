"""Q2: how much would winter hotel demand suffer if an airline reduced its flights?"""

import math

import streamlit as st

from app import charts, core, ui

ui.header(
    "Winter flight cut",
    "How much would winter hotel occupancy suffer if an airline reduced its flights?",
)
routes = core.route_table().dropna(subset=["weekly_flights"])
routes = routes[routes["weekly_flights"] >= 1]
markets = sorted(routes["market"].unique())

SEASONS = {
    "Winter 2026/27 (Nov – Feb)": core.WINTER,
    "Peak winter (Dec – Jan)": ("2026-12", "2027-01"),
    "Summer 2027 (Jun – Aug)": ("2027-06", "2027-08"),
    "Custom": None,
}

with st.container(border=True):
    c1, c2, c3 = st.columns([1, 1.6, 1.2])
    with c1:
        market = st.selectbox("Country", markets, index=markets.index("UNITED KINGDOM"))
    mr = routes[routes["market"] == market].reset_index(drop=True)
    with c2:
        i = st.selectbox(
            "Route",
            mr.index,
            format_func=lambda j: f"{mr.at[j, 'city']} · {mr.at[j, 'airline']} — "
                                  f"{mr.at[j, 'weekly_flights']:.0f} flights/wk, "
                                  f"{mr.at[j, 'seats_per_flight']:.0f} seats each",
        )
        route = mr.loc[i]
    with c3:
        current = max(1, math.floor(route["weekly_flights"]))
        cut = st.slider("Flights cut per week", 1, current, min(7, current))
    c4, c5 = st.columns([1.2, 2])
    with c4:
        season = st.radio("Season", list(SEASONS), index=0)
    with c5:
        if SEASONS[season] is None:
            opts = ui.month_options()
            a, b = st.columns(2)
            start = a.selectbox("From", opts, index=opts.index("2026-11"))
            ends = [m for m in opts if m >= start]
            end = b.selectbox("To", ends, index=min(3, len(ends) - 1))
            window = (start, end)
        else:
            window = SEASONS[season]
        rooms0, src = core.hotel_rooms()
        a, b = st.columns(2)
        rooms = a.number_input("Hotel rooms in Abu Dhabi", 5_000, 100_000, rooms0, step=1_000,
                               help=f"Default from the {src} report.")
        gpr = b.number_input("Guests per occupied room", 1.0, 3.0, 1.7, step=0.1,
                             help="Assumption: the pack has no room data. Adjust to your planning figure.")

sc, notes = core.frequency_change(market, -cut, *window, city=route["city"], airline=route["airline"])
s = core.summarize(sc, window)
for n in notes:
    st.info(n, icon=":material/info:")
occ = core.occupancy_points(s.delta_nights, window, rooms, gpr)

st.subheader(f"{route['airline']} flies {cut} fewer {route['city']} flights a week, "
             f"{ui.window_label(window)}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Guest-nights lost", ui.signed(s.delta_nights), border=True)
k2.metric("Share of the season's demand", f"{s.pct_of_window:+.2f}%", border=True,
          help="Against the national baseline guest-nights in the same months.")
k3.metric("Hotel arrivals lost", ui.signed(s.delta_arrivals), border=True)
k4.metric("Occupancy, indicative", f"{occ:+.2f} pts", border=True,
          help=f"Guest-nights ÷ {gpr} guests per room ÷ ({rooms:,} rooms × days in the season).")

left, right = st.columns([3, 2])
with left:
    st.markdown("**National guest-nights in the season**")
    st.plotly_chart(charts.compare_lines(s.monthly), width="stretch")
with right:
    st.markdown("**Guest-nights lost by month**")
    st.plotly_chart(charts.delta_bars(s.monthly, height=360), width="stretch")

seats_lost = -s.stages["seats"]
st.caption(
    f"{seats_lost:,.0f} seats removed → {-s.stages['terminating_passengers']:,.0f} fewer terminating "
    f"passengers → {-s.delta_arrivals:,.0f} fewer hotel arrivals → {-s.delta_nights:,.0f} fewer "
    f"guest-nights ({-s.delta_nights / seats_lost:.2f} per seat). Fixed ratios alone would say "
    f"{ui.signed(s.chain_delta_nights)}."
)
st.caption(
    "Occupancy is an estimate built on the two assumptions above; guest-nights are what the model "
    "predicts. The season's baseline repeats the previous year's schedule."
)
ui.market_caveat()
