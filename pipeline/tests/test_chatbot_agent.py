"""Does the chatbot agent ground its answers in the real model?

This is the tenth diagnostic. It tests the OpenRouter wiring without the
network: config precedence (sidebar > secrets > env > .env), .env parsing and
saving, all four agent tools against the fitted simulator, and the agent loop
itself with a scripted (mocked) transport so a tool call and a final answer are
exercised end to end. Writes NO report.

Usage:  .venv/bin/python pipeline/tests/test_chatbot_agent.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import llm  # noqa: E402
from app.shared import load_simulator  # noqa: E402


def main():
    t0 = time.time()

    # -- .env parsing --------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        env_path = Path(td) / ".env"
        env_path.write_text(
            "# comment\n\n"
            "OPENROUTER_API_KEY = 'quoted-key'\n"
            "OPENROUTER_MODEL=\"quoted/model\"\n"
            "MALFORMED_LINE\n"
            "EMPTY=\n"
        )
        parsed = llm.parse_env_file(env_path)
        assert parsed["OPENROUTER_API_KEY"] == "quoted-key", parsed
        assert parsed["OPENROUTER_MODEL"] == "quoted/model", parsed
        assert "MALFORMED_LINE" not in parsed and "EMPTY" not in parsed
    print(".env parsing ok")

    # -- config precedence: overrides > env > file; default model ------------
    real_root = llm.paths.ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            llm.paths.ROOT = Path(td)
            (Path(td) / ".env").write_text(
                "OPENROUTER_API_KEY=filekey\nOPENROUTER_MODEL=file/model\n"
            )
            env = {"OPENROUTER_API_KEY": "", "OPENROUTER_MODEL": ""}  # blank out any real env
            with mock.patch.dict("os.environ", env):
                cfg, src = llm.resolve_config()
                assert cfg["api_key"] == "filekey" and cfg["model"] == "file/model", cfg
                assert ".env file" in src, src

                with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "envkey"}):
                    cfg, _ = llm.resolve_config()
                    assert cfg["api_key"] == "envkey" and cfg["model"] == "file/model"

                    cfg, src = llm.resolve_config(
                        {"OPENROUTER_API_KEY": "sidekey", "OPENROUTER_MODEL": "side/model"}
                    )
                    assert cfg["api_key"] == "sidekey" and cfg["model"] == "side/model"
                    assert "sidebar" in src

            (Path(td) / ".env").unlink()
            cfg, src = llm.resolve_config()
            assert cfg["api_key"] == "" and cfg["model"] == llm.DEFAULT_MODEL
            assert "built-in default" in src
    finally:
        llm.paths.ROOT = real_root
    print("config precedence ok (sidebar > secrets > env > .env > default)")

    # -- save_to_env_file preserves unrelated lines ---------------------------
    try:
        with tempfile.TemporaryDirectory() as td:
            llm.paths.ROOT = Path(td)
            (Path(td) / ".env").write_text(
                "OTHER=keep\n# a comment\nOPENROUTER_API_KEY=old\n"
            )
            saved = llm.save_to_env_file("newkey", "new/model")
            text = saved.read_text()
            assert "OTHER=keep" in text and "# a comment" in text
            assert "OPENROUTER_API_KEY=newkey" in text
            assert "OPENROUTER_MODEL=new/model" in text
            assert "old" not in text
    finally:
        llm.paths.ROOT = real_root
    print(".env save round-trip ok")

    # -- tools against the real fitted model ----------------------------------
    sim, model_source = load_simulator()
    print(f"model: {model_source}")

    result = json.loads(
        llm.execute_tool(
            sim,
            "run_scenario",
            {"routes": [{
                "market": "INDIA", "city": "Mumbai", "airline": "Etihad",
                "seats_per_week": 1260.0, "started_period": "2026-11",
                "ended_period": "2027-02", "label": "KENYA",
            }]},
        )
    )
    assert "error" not in result, result
    assert result["delta_guest_nights"] > 0, result
    assert result["months_scored"] == 84, result  # full horizon scored
    assert result["changed_months"] == 4, result  # only the edit window moved
    assert result["changed_window"] == {"from": "2026-11", "to": "2027-02"}, result
    assert abs(result["avg_monthly_delta_guest_nights"] * 4 - result["delta_guest_nights"]) < 1.0
    assert any("analog for KENYA" in m["market"] for m in result["top_market_movers"]), result
    print(f"run_scenario ok (delta {result['delta_guest_nights']:+,.0f} nights)")

    result = json.loads(llm.execute_tool(sim, "run_scenario", {"routes": [
        {"market": "NARNIA", "seats_per_week": 100.0},
    ]}))
    assert "error" in result and "NARNIA" in result["error"], result
    print("run_scenario rejects unknown markets ok")

    result = json.loads(llm.execute_tool(sim, "get_market_ranking", {"sort_by": "capture", "limit": 5}))
    assert len(result["rows"]) == 5 and result["rows"][0]["capture"] >= result["rows"][-1]["capture"]
    print("get_market_ranking ok")

    result = json.loads(llm.execute_tool(sim, "get_route_supply", {"market": "INDIA"}))
    assert result["rows"] and result["rows"][0]["weekly_seats"] > 0, result
    result = json.loads(llm.execute_tool(sim, "get_route_supply", {"market": "ATLANTIS"}))
    assert "error" in result, result
    print("get_route_supply ok")

    result = json.loads(llm.execute_tool(sim, "get_model_facts", {}))
    assert len(result["flight_markets"]) == 33, result
    assert result["validation_pinned"]["t5_wmape_new_arrivals_pct"] == 20.55
    assert result["horizon"]["observed_supply_to"] == "2026-02"
    print("get_model_facts ok")

    # -- agent loop with a scripted transport ---------------------------------
    calls = []

    def fake_chat(cfg, messages, tools=None, **kwargs):
        calls.append(list(messages))
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "get_model_facts", "arguments": "{}"},
                }],
            }
        return {"role": "assistant", "content": "Final answer.", "tool_calls": None}

    with mock.patch.object(llm, "chat", fake_chat):
        answer, trace = llm.run_agent(
            {"api_key": "k", "model": "m"},
            sim,
            [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        )
    assert answer == "Final answer.", answer
    assert len(trace) == 1 and trace[0]["tool"] == "get_model_facts"
    assert len(calls) == 2, f"expected two transport calls, got {len(calls)}"
    second = calls[1]
    assert second[2]["role"] == "assistant" and second[2]["tool_calls"][0]["id"] == "c1"
    tool_msg = second[3]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert "flight_markets" in tool_msg["content"], "the real tool must have run in-loop"
    print("agent loop ok (tool exchange assembled and executed)")

    print(f"\nALL CHATBOT-AGENT PINS PASSED ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
