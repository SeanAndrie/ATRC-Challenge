"""Measure what a 2-day /flights/* window actually costs.

The published table is ambiguous between the "1-2 partitions" band (30 credits)
and a per-partition reading. Everything downstream -- whether the full range fits
in one quota cycle -- depends on the answer, so it is measured, not assumed.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyopensky.rest as rest_mod
from pyopensky.rest import REST

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from paths import CREDENTIALS  # noqa: E402

# credentials.json is gitignored and stays out of version control. It was
# previously hardcoded to an absolute home-directory path, which made the script
# fail on any other machine and on any other checkout location.
CREDS = CREDENTIALS


def remaining(client, begin, end, airport="OMAA", kind="arrival"):
    url = (
        f"https://opensky-network.org/api/flights/{kind}"
        f"?begin={begin.timestamp():.0f}&end={end.timestamp():.0f}&airport={airport}"
    )
    r = client.client.get(url, headers=client.headers)
    left = r.headers.get("X-Rate-Limit-Remaining")
    rows = len(r.json()) if r.status_code == 200 else 0
    return int(left) if left is not None else None, rows, r.status_code


def main():
    raw = json.loads(CREDS.read_text())
    rest_mod.client_id = raw["clientId"]
    rest_mod.client_secret = raw["clientSecret"]
    client = REST()
    assert client.token is not None, "no token"

    d = pd.Timestamp("2026-09-15", tz="utc")

    before, _, _ = remaining(client, d, d + pd.Timedelta("1D"))
    print(f"balance before      : {before}")
    after_1d, rows_1d, st1 = remaining(client, d, d + pd.Timedelta("1D"))
    cost_1d = before - after_1d
    print(f"1-day window        : {st1}, {rows_1d} rows, cost {cost_1d} credits")

    after_2d, rows_2d, st2 = remaining(client, d, d + pd.Timedelta("2D"))
    cost_2d = after_1d - after_2d
    print(f"2-day window        : {st2}, {rows_2d} rows, cost {cost_2d} credits")

    after_3d, rows_3d, st3 = remaining(client, d, d + pd.Timedelta("3D"))
    cost_3d = after_2d - after_3d
    print(f"3-day window        : {st3}, {rows_3d} rows, cost {cost_3d} credits")

    print()
    print("credits per day of data:")
    for span, cost, rows in [(1, cost_1d, rows_1d), (2, cost_2d, rows_2d), (3, cost_3d, rows_3d)]:
        if cost is None or cost <= 0:
            print(f"  {span}-day: cost unknown (header missing)")
            continue
        per_day = cost / span
        print(f"  {span}-day: {cost:3} credits / {span} day(s) = {per_day:5.1f} per day"
              f"  -> full 100-day range both directions = {per_day * 200:.0f} credits")

    print()
    best = min(
        [(s, c) for s, c in [(1, cost_1d), (2, cost_2d), (3, cost_3d)] if c],
        key=lambda t: t[1] / t[0],
    )
    print(f"optimal span: {best[0]}-day ({best[1]} credits/window)")
    print(f"balance after tests: {after_3d}")


if __name__ == "__main__":
    main()
