import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Flight -> Hotel Simulator", layout="wide")
st.title("Flight → Hotel Demand Simulator (Parametric Chain)")
st.info("Parametric chain: seats → PAX (LF) → inbound (P2P share) → new arrivals (capture rate) → guest nights (ALOS). Lean, transparent.")
