"""Streamlit front end for the ATRC scenario simulator (``app/simulator.py``).

The UI is intentionally thin: all pure, testable logic lives in ``app.logic``
(scenario assembly, delta math, chart builders) and all static prose in
``app.copy``. ``app.simulator`` only wires widgets to those helpers.

Run it with::

    .venv/bin/streamlit run app/simulator.py
"""
