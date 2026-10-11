"""Small presentation helpers shared by the pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import core


def signed(v: float, digits: int = 0) -> str:
    return f"{v:+,.{digits}f}"


def compact(v: float) -> str:
    a = abs(v)
    sign = "−" if v < 0 else ""
    if a >= 1e6:
        return f"{sign}{a / 1e6:.2f}M"
    if a >= 1e4:
        return f"{sign}{a / 1e3:.1f}k"
    return f"{sign}{a:,.0f}"


def header(title: str, question: str | None = None, caption: str | None = None) -> None:
    st.title(title)
    if question:
        st.markdown(f"#### *“{question}”*")
    if caption:
        st.caption(caption)


def month_options(start: str = core.FIRST_FORWARD, end: str = core.HORIZON_END) -> list[str]:
    return core.months(start, end)


def window_label(window: tuple[str, str] | None) -> str:
    if not window:
        return "no months changed"
    fmt = lambda ym: pd.Period(ym, freq="M").strftime("%b %Y")  # noqa: E731
    return f"{fmt(window[0])} – {fmt(window[1])}"


def market_caveat() -> None:
    st.caption(
        "⚠ Market rows are supply-side-only: in the flight data a market is the "
        "last departure country, in the hotel data it is guest nationality, so "
        "connecting traffic lands on hubs. National totals are the reliable read."
    )


def sidebar_status() -> None:
    sim = core.get_sim()
    with st.sidebar:
        st.markdown("**Model**")
        st.caption(
            f"LightGBM hybrid over a parametric chain · {sim.panel['market'].nunique()} "
            f"markets · held-out error {core.PINNED['t5_wmape']:.1f}% WMAPE, "
            f"national bias {core.PINNED['t5_bias']:+.2f}%."
        )
        st.caption(
            f"Observed flights to {core.OBSERVED_TO}; later months repeat the same "
            f"month a year earlier, through {core.HORIZON_END}. Read results as "
            "differences from that baseline, not forecasts."
        )
