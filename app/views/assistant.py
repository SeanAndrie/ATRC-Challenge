"""Ask the model in plain language."""

import json

import streamlit as st

from app import agent, charts, ui

ui.header(
    "Ask the model",
    caption="Plain-language what-ifs. The assistant runs the same model as the other pages and "
    "shows every calculation it made.",
)

with st.sidebar:
    st.divider()
    st.markdown("**Assistant settings**")
    key = st.text_input("OpenRouter API key", type="password",
                        help="Optional if the key is already in Streamlit secrets, the environment or .env.")
    model = st.text_input("Model", placeholder=agent.DEFAULT_MODEL,
                          help="Any tool-capable model on openrouter.ai, e.g. anthropic/claude-sonnet-4.5.")
    cfg = agent.config(key, model)
    st.caption(f"Using `{cfg['model']}` · key {'found' if cfg['api_key'] else 'missing'}")
    if st.button("Clear conversation"):
        st.session_state.chat = []

SUGGESTIONS = [
    "A new twice-weekly route opens from Kenya next November. How many extra hotel nights?",
    "What if Etihad runs 4 fewer London flights a week this winter?",
    "Airlines put 40 more seats on every Mumbai flight next year — what happens?",
    "What if planes from India are 5% fuller next year?",
    "Russia grows 20% and India shrinks 10% in winter. Net effect?",
    "Which countries bring the most hotel guests per seat?",
]

chat = st.session_state.setdefault("chat", [])


def show(entry: dict) -> None:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        for t in entry.get("trace", []):
            m = t.get("monthly")
            if m is not None and len(m) and m["delta"].abs().sum() > 0:
                st.plotly_chart(charts.delta_bars(m, height=220), width="stretch",
                                key=f"chart_{id(t)}")
        if entry.get("trace"):
            with st.expander(f"{len(entry['trace'])} model call(s)"):
                for t in entry["trace"]:
                    st.markdown(f"**{t['tool']}** `{json.dumps(t['args'])}`")
                    st.code(json.dumps(json.loads(t["result"]), indent=2)[:2000], language="json")


if not chat:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 2].button(q, key=f"sugg_{i}", width="stretch"):
            st.session_state.pending = q
            st.rerun()

for entry in chat:
    show(entry)

prompt = st.chat_input("Ask about a route, frequency, aircraft, load factor or seasonal mix…")
prompt = prompt or st.session_state.pop("pending", None)
if prompt:
    chat.append({"role": "user", "content": prompt})
    show(chat[-1])
    if not cfg["api_key"]:
        answer, trace = ("I need an OpenRouter API key to answer. Add one in the sidebar (or set "
                         "`OPENROUTER_API_KEY` in Streamlit secrets or `.env`). Meanwhile, every "
                         "question here also has a page in the menu."), []
    else:
        history = [{"role": e["role"], "content": e["content"]} for e in chat]
        with st.spinner("Running the model…"):
            try:
                answer, trace = agent.run(cfg, history)
            except agent.AgentError as exc:
                answer, trace = f"**Something went wrong:** {exc}", []
    chat.append({"role": "assistant", "content": answer, "trace": trace})
    st.rerun()
