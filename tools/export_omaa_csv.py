"""Compile the pulled OpenSky OMAA event parquet files into a single CSV.

Read-only against the parquet files; writes one CSV. Kept as a script so the
export is reproducible rather than a one-off shell redirect.

Reads and writes inside data/raw/opensky/. The compiled CSV is treated as raw
alongside the parquet it comes from: it is a verbatim concatenation plus a
coverage label, with no modelling, and consumers read it directly. Note that
both are untracked, so a clean clone must re-pull OpenSky before anything that
consumes this can run.
"""

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from paths import OPENSKY, OPENSKY_PARQUET_GLOB  # noqa: E402

SRC_GLOB = str(OPENSKY / OPENSKY_PARQUET_GLOB)
DEST_DIR = OPENSKY
ISO = "%Y-%m-%dT%H:%M:%S%z"

# Mirrors daily_coverage() in the notebook: a day's score is relative to a
# well-observed reference day, and the reference is re-derived from the healthy
# days so a long outage cannot drag it down and make good days look over-observed.
COVERAGE_LOW = 0.50
COVERAGE_SPARSE = 0.25


def coverage_flags(frame, low=COVERAGE_LOW, sparse=COVERAGE_SPARSE, passes=2):
    """Per-day coverage label, matching the notebook's `daily_coverage`."""
    daily = frame.groupby("day").size().rename("events").reset_index()
    reference = float(daily["events"].median())
    for _ in range(max(1, passes)):
        ratio = daily["events"] / reference if reference else float("nan")
        healthy = daily.loc[ratio.ge(low), "events"]
        if healthy.empty or float(healthy.median()) == reference:
            break
        reference = float(healthy.median())
    daily["ratio"] = daily["events"] / reference if reference else float("nan")
    daily["coverage_flag"] = pd.cut(
        daily["ratio"],
        bins=[float("-inf"), sparse, low, float("inf")],
        labels=["sparse", "low", "ok"],
        right=False,
    ).astype(str)
    return daily, reference


def main():
    files = sorted(glob.glob(SRC_GLOB))
    if not files:
        raise SystemExit(f"no parquet files matched {SRC_GLOB}; run backfill_rest first")

    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    frame["day"] = pd.to_datetime(frame["day"], utc=True)
    source_rows = len(frame)
    source_cols = frame.shape[1] + 2  # + coverage_flag and coverage_ratio
    print(f"read {len(files)} day-files: {len(frame)} rows x {frame.shape[1]} cols")
    print(f"nulls preserved: {int(frame['origin'].isna().sum())} origin, "
          f"{int(frame['counterpart'].isna().sum())} counterpart")

    # Carry the coverage label with the data. Without it a consumer summing
    # arrivals per day would read an outage as a collapse in demand.
    daily, reference = coverage_flags(frame)
    flag_map = daily.set_index("day")["coverage_flag"]
    ratio_map = daily.set_index("day")["ratio"]
    frame["coverage_flag"] = frame["day"].map(flag_map)
    frame["coverage_ratio"] = (frame["day"].map(ratio_map)).round(3)

    counts = daily["coverage_flag"].value_counts().to_dict()
    print(f"\ncoverage reference day: {reference:.0f} events")
    print(f"day flags: {counts}")
    for flag in ("sparse", "low"):
        hit = sorted(daily.loc[daily["coverage_flag"].eq(flag), "day"].dt.strftime("%Y-%m-%d"))
        if hit:
            print(f"  {flag}: {len(hit)} day(s) -> {', '.join(hit)}")

    # Distances are whole metres, but the nulls force a float column and they
    # then serialise as "11775.0". Nullable Int64 writes them as plain integers
    # with an empty cell where OpenSky could not estimate the airport.
    for col in ("origin_distance_m", "home_distance_m"):
        frame[col] = frame[col].round().astype("Int64")

    frame = frame.sort_values(["event_time", "direction", "icao24"], kind="stable")

    # Name the file after the span it actually contains, so a widened pull does
    # not leave a stale date range in the filename.
    start, end = frame["day"].min(), frame["day"].max()
    dest = DEST_DIR / f"opensky_omaa_{start:%Y-%m-%d}_{end:%Y-%m-%d}.csv"
    frame.to_csv(dest, index=False, date_format=ISO, encoding="utf-8")

    # Round-trip check: the CSV must read back with the same shape, and its
    # timestamps must be recoverable as tz-aware UTC. pandas does not infer
    # datetimes on read by default, so parse explicitly -- the point is that the
    # ISO-8601 offsets make the timezone *recoverable*, not that it is guessed.
    back = pd.read_csv(dest)
    assert back.shape == (source_rows, source_cols), back.shape
    assert int(back["origin"].isna().sum()) == int(frame["origin"].isna().sum())

    parsed = pd.to_datetime(back["event_time"], utc=True, format="ISO8601")
    assert parsed.notna().all()
    assert str(parsed.dtype).endswith("UTC]"), parsed.dtype
    assert parsed.min() == frame["event_time"].min(), (parsed.min(), frame["event_time"].min())
    assert parsed.max() == frame["event_time"].max(), (parsed.max(), frame["event_time"].max())

    # The coverage label must survive the round trip intact, or the warning it
    # carries is lost exactly where someone aggregates the data. `counts` is per
    # day, so compare per-day labels, not per-row row counts.
    back_flags = back.groupby("day")["coverage_flag"].agg(lambda s: s.mode().iloc[0])
    assert len(back_flags) == len(daily), (len(back_flags), len(daily))
    assert back_flags.value_counts().to_dict() == counts, (
        back_flags.value_counts().to_dict(), counts
    )
    # `back` comes back from CSV with `day` as a string, so align before comparing.
    back_index = pd.to_datetime(back_flags.index, utc=True)
    assert (pd.Series(back_flags.to_numpy(), index=back_index)
            == daily.set_index("day")["coverage_flag"]).all()

    print(f"\nwrote {dest}")
    print(f"  {back.shape[0]} rows x {back.shape[1]} cols")
    print(f"  event_time reparsed as: {parsed.dtype}")
    print(f"  null origin preserved: {int(back['origin'].isna().sum())}")
    print(f"  span: {parsed.min():%Y-%m-%d} -> {parsed.max():%Y-%m-%d}")
    print(f"  by direction: {back['direction'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
