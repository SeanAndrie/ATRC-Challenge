"""The Overview page's seats-to-hotel Sankey, rebuilt on the training data.

Reads the competition training pack through ``pipeline.hybrid.build_panels``
(flights joined to international hotel guests, 2022-01 .. 2025-07), runs the
parametric chain with unit levers, and draws ``charts.chain_sankey`` with the
same headline figures the Overview shows beside it. Seats, passengers and
terminating passengers are measured; hotel arrivals and guest-nights are the
model's estimates. Writes outputs/training_sankey.png (needs kaleido).

    .venv/bin/python tools/training_sankey.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import charts  # noqa: E402
from pipeline import hybrid, paths  # noqa: E402

DEST = paths.OUTPUTS / "training_sankey.png"
BG = "#e8e6d9"
PILL = "#d6d3c4"


def compact(v: float) -> str:
    return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"


def figure():
    _fa, _m, _mo, panel, *_ = hybrid.build_panels()
    out = hybrid.simulate(panel, hybrid.fit_params(panel))
    seats, pax, p2p, arr, nights = (
        float(out[c].sum()) for c in ("Total_Seats", "PAX", "P2P", "NewArrivalHat", "GuestNightsHat")
    )
    span = f"{panel['ym_str'].min()} – {panel['ym_str'].max()}"

    fig = charts.chain_sankey(seats, pax, p2p, arr)
    # Pin the nodes so each leak peels off the top, as on the Overview: the
    # right-hand column stacks top to bottom in the order passengers leave.
    gap = 0.04
    unit = (1 - 3 * gap) / seats  # paper height per person
    right = [seats - pax, pax - p2p, p2p - arr, arr]  # empty, connecting, no hotel, hotel
    tops = [sum(right[:i]) * unit + i * gap for i in range(4)]
    mid = [t + v * unit / 2 for t, v in zip(tops, right)]
    span_mid = lambda top, v: top + v * unit / 2  # noqa: E731
    fig.update_traces(
        arrangement="fixed", domain=dict(x=[0, 0.76], y=[0, 1]),
        # node order: Seats, Empty, Passengers, Connecting, Terminating, No hotel, Hotel
        node=dict(
            x=[0.001, 0.999, 0.34, 0.999, 0.67, 0.999, 0.999],
            y=[span_mid(0, seats), mid[0], span_mid(tops[1], pax), mid[1],
               span_mid(tops[2], p2p), mid[2], mid[3]],
        ),
    )

    # Headline column, laid out like the Overview's st.metric stack.
    tiles = [
        ("Seats flown in", seats, None),
        ("Hotel arrivals", arr, f"↑ {1000 * arr / seats:.0f} per 1,000 seats"),
        ("Guest-nights", nights, f"↑ {nights / arr:.1f} nights per arrival"),
    ]
    x, y = 0.795, 0.99
    for label, value, note in tiles:
        fig.add_annotation(x=x, y=y, text=label, font=dict(size=15, color=charts.INK))
        fig.add_annotation(x=x, y=y - 0.07, text=compact(value), font=dict(size=40, color=charts.INK))
        if note:
            fig.add_annotation(x=x, y=y - 0.215, text=note, bgcolor=PILL, borderpad=5,
                               font=dict(size=14, color=charts.INK_2))
        y -= 0.34 if note else 0.27
    fig.update_annotations(xref="paper", yref="paper", xanchor="left", yanchor="top",
                           showarrow=False, align="left")

    fig.update_layout(
        height=560, width=1320, paper_bgcolor=BG, plot_bgcolor=BG,
        title=dict(text="<b>Training data, followed to the hotel</b>", x=0.012, y=0.97,
                   font=dict(size=28, color=charts.INK)),
        margin=dict(l=36, r=24, t=84, b=76),
    )
    fig.add_annotation(
        x=0, y=-0.04, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
        text=(f"{span}, {panel['market'].nunique()} markets, {len(panel):,} market-months. "
              f"Seats to terminating passengers are measured;<br>hotel arrivals and guest-nights are "
              f"model estimates (measured: {compact(panel['NewArr'].sum())} arrivals, "
              f"{compact(panel['Guests'].sum())} guest-nights)."),
        font=dict(size=13, color=charts.MUTED),
    )
    return fig


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    figure().write_image(DEST, scale=2)
    print(f"wrote {DEST.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
