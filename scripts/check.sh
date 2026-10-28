#!/usr/bin/env bash
# One-command verification for the submission.
#
# The app-logic and sensitivity checks are fast; the packaging gate re-fits the
# hybrid (~90s) and is the one that must pass before submission.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Python not found at $PY. Set PY=/path/to/python or run: uv sync" >&2
  exit 1
fi

echo "== app logic =="
"$PY" pipeline/tests/test_simulator_app.py

echo "== sensitivity =="
"$PY" pipeline/tests/test_sensitivity.py

echo "== packaging gate (pins T5/T4) =="
"$PY" pipeline/tests/test_simulator_package.py

echo "== market attribution guard =="
"$PY" pipeline/tests/test_market_attribution.py

echo "ALL CHECKS PASSED"
