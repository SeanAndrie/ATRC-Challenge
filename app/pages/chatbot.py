"""Chatbot endpoint for the flight-to-hotel simulator.

Streamlit auto-discovers ``app/pages/*.py`` as multipage routes when the entry
script is ``app/simulator.py``, so this page is served at ``/chatbot`` while
the simulator stays at ``/``.

The chat is an agent over OpenRouter whose tools call the real fitted model
(``app/llm.py``): scenario scoring, market rankings, route supply and pinned
validation facts. Configuration comes from the sidebar, ``st.secrets``,
environment variables or a repo-root ``.env`` (template: ``.env.example``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.llm import (  # noqa: E402
    DEFAULT_MODEL,
    OpenRouterError,
    resolve_config,
    run_agent,
    save_to_env_file,
    system_prompt,
)
from app.shared import load_simulator  # noqa: E402

st.set_page_config(page_title="Flight → Hotel Chatbot", layout="wide")

sim, model_source = load_simulator()

# --- sidebar: OpenRouter settings -------------------------------------------
st.sidebar.header("OpenRouter settings")
st.sidebar.caption(
    "Key and model can also come from a repo-root `.env` (see `.env.example`), "
    "environment variables or Streamlit secrets — sidebar wins when filled."
)
st.sidebar.text_input(
    "OpenRouter API key",
    type="password",
    key="or_api_key",
    help="Create one at openrouter.ai/keys. Stored in session only unless saved below.",
)
st.sidebar.text_input(
    "Model",
    value=st.session_state.get("or_model", DEFAULT_MODEL),
    key="or_model",
    help="Any tool-capable model on openrouter.ai/models, e.g. openai/gpt-4o-mini, "
    "anthropic/claude-3.5-sonnet, google/gemini-2.0-flash-001.",
)
if st.sidebar.button("Save key + model to .env"):
    key_value = st.session_state.get("or_api_key", "")
    if not key_value:
        st.sidebar.error("Enter an API key first.")
    else:
        saved = save_to_env_file(key_value, st.session_state.get("or_model") or DEFAULT_MODEL)
        st.sidebar.success(f"Saved to `{saved.name}` (gitignored). Restart not needed.")
st.sidebar.caption("Saving writes the key in plain text at the repo root; `.env` is gitignored.")

overrides = {
    "OPENROUTER_API_KEY": st.session_state.get("or_api_key", ""),
    "OPENROUTER_MODEL": st.session_state.get("or_model", ""),
}
cfg, config_source = resolve_config(overrides)
st.sidebar.caption(f"Config — {config_source}.")

# --- header ------------------------------------------------------------------
st.title("Flight → Hotel Chatbot")
st.caption(
    "Open-ended questions answered by the fitted model, not by guesswork: the "
    "assistant calls tools that score real scenarios and read measured stats. "
    f"Model: {model_source}."
)
st.info(
    "Reminder — aggregate guest-nights are the valid read. Market-level rows "
    "are supply-side-only (departure country ≠ guest nationality), and a "
    "country that never flew here is an analog estimate."
)

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # [{"role", "content", "tools": [...]}]
if "chat_protocol" not in st.session_state:
    st.session_state.chat_protocol = [{"role": "system", "content": system_prompt(sim)}]

# --- history -----------------------------------------------------------------
for entry in st.session_state.chat_display:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("tools"):
            with st.expander(f"Tool calls ({len(entry['tools'])})"):
                for call in entry["tools"]:
                    st.markdown(f"**{call['tool']}**")
                    st.caption(json.dumps(call["args"])[:300])
                    st.text(call["result_head"])

# --- turn --------------------------------------------------------------------
prompt = st.chat_input(
    "e.g. How many extra hotel nights would a new twice-weekly flight from Nairobi generate?"
)
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_display.append({"role": "user", "content": prompt})

    if not cfg["api_key"]:
        answer = (
            "I need an OpenRouter API key to reason. Add it in the sidebar, or "
            "put `OPENROUTER_API_KEY` in a repo-root `.env` (see `.env.example`), "
            "then ask again."
        )
        trace: list[dict] = []
    else:
        st.session_state.chat_protocol.append({"role": "user", "content": prompt})
        try:
            with st.chat_message("assistant"), st.spinner("Scoring with the model…"):
                answer, trace = run_agent(cfg, sim, st.session_state.chat_protocol)
        except OpenRouterError as exc:
            answer, trace = f"**OpenRouter error** — {exc}", []

    st.session_state.chat_protocol.append({"role": "assistant", "content": answer})
    st.session_state.chat_display.append(
        {"role": "assistant", "content": answer, "tools": trace}
    )
    st.rerun()
