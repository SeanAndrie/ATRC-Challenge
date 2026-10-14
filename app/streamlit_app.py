"""ATRC Flight → Hotel Demand Simulator.

Run from the repository root::

    .venv/bin/streamlit run app/streamlit_app.py

The model artifacts live in ``outputs/models/`` (build them with
``python -m app.build_model``). See ``app/README.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import core, ui  # noqa: E402

st.set_page_config(
    page_title="Flight → Hotel Demand Simulator",
    page_icon="✈️",
    layout="wide",
)

try:
    core.get_sim()
except core.ModelUnavailable as exc:
    st.error(str(exc))
    st.stop()

VIEWS = Path(__file__).parent / "views"
pages = {
    "": [st.Page(VIEWS / "overview.py", title="Overview", icon=":material/flight_land:", default=True)],
    "Explore": [
        st.Page(VIEWS / "scenario_lab.py", title="Scenario lab", icon=":material/tune:"),
        st.Page(VIEWS / "assistant.py", title="Ask the model", icon=":material/forum:"),
    ],
    "Evidence": [
        st.Page(VIEWS / "validation.py", title="Validation", icon=":material/verified:"),
    ],
}
nav = st.navigation(pages)
ui.sidebar_status()
nav.run()
