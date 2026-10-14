"""Build the artifacts the app serves, from the competition pack.

Run once from the repository root (needs ``data/raw/competition/``)::

    .venv/bin/python -m app.build_model

Writes two files to ``outputs/models/`` (gitignored):

* ``app_simulator.pkl`` -- the fitted simulator, including the competition
  window, so scenarios and the back-test replay load instantly;
* ``app_validation.parquet`` -- the T5 back-test (hybrid and parametric chain vs
  actual New Arrivals) and the T4 guest-nights holdout, for the Validation page.

A deployed app needs only these two files; it never reads the licensed pack.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.core import MODEL_PATH, VALIDATION_PATH  # noqa: E402
from pipeline.simulator import Scenario, Simulator  # noqa: E402


def validation_frame(sim: Simulator) -> pd.DataFrame:
    """T5 (hybrid + chain vs actual arrivals) and T4 (guest-nights) in one frame."""
    t5 = sim.reproduce_competition_window()
    chain = sim.chain(Scenario())[["ym_str", "market", "NewArrivalHat"]]
    t5 = t5.merge(
        chain.rename(columns={"NewArrivalHat": "ChainHat"}), on=["ym_str", "market"], how="left"
    )
    t5["window"] = "T5"

    t4 = sim.reproduce_holdout()
    t4 = t4[["ym_str", "market", "NewArr", "NewArrivalHat", "Guests", "GuestNightsHat"]].copy()
    t4["window"] = "T4"
    return pd.concat([t5, t4], ignore_index=True)


def main() -> None:
    print("Fitting the scenario model (about a minute)…")
    sim = Simulator(window="full").fit()
    print("Replaying the back-tests…")
    frame = validation_frame(sim)
    t5 = frame[frame["window"] == "T5"]
    t4 = frame[frame["window"] == "T4"]
    wmape = lambda a, p: 100.0 * (a - p).abs().sum() / a.sum()  # noqa: E731
    print(f"  T5 hybrid WMAPE {wmape(t5.NewArr, t5.NewArrivalHat):.2f}% "
          f"(bias {100 * (t5.NewArrivalHat.sum() / t5.NewArr.sum() - 1):+.2f}%), "
          f"chain {wmape(t5.NewArr, t5.ChainHat):.2f}%")
    print(f"  T4 guest-nights WMAPE {wmape(t4.Guests, t4.GuestNightsHat):.2f}%")

    model_path = sim.save(MODEL_PATH)
    frame.to_parquet(VALIDATION_PATH, index=False)
    print(f"Saved {model_path.relative_to(ROOT)} and {VALIDATION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
