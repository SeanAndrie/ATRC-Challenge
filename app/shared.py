"""Shared app state: one fitted Simulator per Streamlit process.

Both the simulator page (``app/simulator.py``) and the chatbot page
(``app/pages/chatbot.py``) import from here, so ``st.cache_resource`` keeps a
single fitted model (or a single pickle load) for the whole app instead of one
per page. ``baseline_frames`` likewise caches the no-change baseline views once
and is shared by the Results tab and the chatbot's tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from pipeline.simulator import Scenario, Simulator  # noqa: E402


@st.cache_resource(show_spinner=False)
def load_simulator() -> tuple[Simulator, str]:
    """Load the pickled fitted simulator, or fit and cache it once."""
    try:
        return Simulator.load(), "loaded the saved model from outputs/models/"
    except Exception:
        sim = Simulator(window="full").fit()
        try:
            sim.save()
            return sim, "fitted from the competition pack and saved a cache"
        except Exception:
            return sim, "fitted from the competition pack"


@st.cache_data(show_spinner=False)
def baseline_frames(_sim: Simulator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Monthly / market-grain / chain views of the no-change baseline."""
    return (
        _sim.monthly(Scenario()),
        _sim.market(Scenario()),
        _sim.chain(Scenario()),
    )
