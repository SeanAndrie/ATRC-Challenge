import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import logging
    import pathlib
    import sys
    import urllib.request
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyopensky.rest as opensky_rest
    from pyopensky.rest import REST
    from pyopensky.schema import FlightsData4, FlightsData5
    from pyopensky.trino import Trino

    # pyopensky logs a warning whenever fewer than 100 calls remain. At 30 credits
    # per historical query a 7-day pull spends ~420, so that fires constantly and
    # buries real output. `view` surfaces credits from the response headers instead.
    logging.getLogger("pyopensky.rest").setLevel(logging.ERROR)

    # Repo root is found by walking up from this file, not assumed from the
    # working directory: marimo is often launched from notebooks/, and a
    # CWD-relative path then silently resolves to the wrong place.
    _here = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
    _root = next(
        (c for c in (_here, *_here.parents) if (c / "pipeline" / "paths.py").exists()),
        _here,
    )
    sys.path.insert(0, str(_root / "pipeline"))

    import paths

    return (
        FlightsData4,
        FlightsData5,
        Path,
        REST,
        json,
        mo,
        np,
        opensky_rest,
        paths,
        pathlib,
        pd,
        urllib,
    )


@app.cell
def _(FlightsData4, Path, paths, pd):
    # REST parquet, Trino parquet, and the OurAirports reference table.
    DATA_DIR = paths.OPENSKY_TRINO
    API_DIR = paths.OPENSKY


    def as_utc(value):
        """Coerce anything date-like to a UTC timestamp.

        pandas 3 refuses `pd.Timestamp(x, tz=...)` on an already-aware value, and
        this project's timestamps arrive aware from OpenSky, naive from CSVs, and
        as plain strings from config -- so normalise in one place.
        """
        ts = pd.Timestamp(value)
        return ts.tz_localize("utc") if ts.tzinfo is None else ts.tz_convert("utc")


    REF_DIR = paths.REFERENCE

    # --- Scope -------------------------------------------------------------------
    # OMAA = Abu Dhabi / Zayed International (AUH): the city gateway, ~250k movements
    # a year. OMAD = Al Bateen Executive (VIP, small). OMAL is Al Ain -- a different
    # city, so it is deliberately excluded from Abu Dhabi hotel demand.
    HOME_AP = "OMAA"

    # --- OpenSky source ----------------------------------------------------------
    # Two sources, different shapes and very different coverage:
    #
    # Trino (FlightsData4) is unreachable -- PERMISSION_DENIED, awaiting an
    # approved grant. It is the upgrade path for confirmed airports.
    #
    # REST /flights/* works with the repo's OAuth credentials but its historical
    # coverage is regional, not global. Measured on 2025-01-15: Heathrow returned
    # 624 arrivals while OMAA, OMDM, OTBH, OOMS and OTHH all returned 404. OpenSky
    # has almost no Middle East receiver density in 2025, so a full-year Gulf
    # backfill is not possible over REST at any window size. Coverage for OMAA
    # only becomes usable from roughly 2026-06-20. Treat that as a hard floor.
    FLIGHT_TABLE = FlightsData4
    ARRIVAL_COL = "estarrivalairport"
    DEPARTURE_COL = "estdepartureairport"
    ARRIVAL_TIME_COL = "lastseen"
    DEPARTURE_TIME_COL = "firstseen"

    # flights_data5 equivalent:
    #   FLIGHT_TABLE = FlightsData5
    #   ARRIVAL_COL, DEPARTURE_COL = "airportofdestination", "airportofdeparture"
    #   ARRIVAL_TIME_COL, DEPARTURE_TIME_COL = "landingtime", "takeofftime"

    # --- Window ------------------------------------------------------------------
    # Half-open [START, STOP) in UTC. 2026-06-20 is the earliest date with
    # usable OMAA coverage; everything before it returns 404 for the Gulf. The
    # range spans the coverage ramp, so early months are thinly observed -- see
    # `daily_coverage` in the aggregate cell, not raw monthly totals.
    # modelling window: it exists to prove the pipeline end to end. Widen it once
    # the organisers' hotel table is available -- but note the REST floor above,
    # and that a wider window costs 30 credits per request per direction.
    START = pd.Timestamp("2026-06-20", tz="utc")
    STOP = pd.Timestamp("2026-09-28", tz="utc")

    # --- Cost model --------------------------------------------------------------
    # Measured, not documented. Every historical /flights/* request costs 30 credits
    # regardless of window size (4h and 24h windows both cost 30); the advertised
    # 4-credit tier applies to live queries only. A 1-day window per direction is
    # therefore the cheapest way to cover a day, and 2-day windows halve the
    # request count at the same 30 credits.
    CREDITS_PER_REQUEST = 30
    DAILY_CREDIT_BUDGET = 4000

    # --- Transfer detection ------------------------------------------------------
    # A passenger is treated as connecting when the *same aircraft* (icao24) lands
    # at and then departs HOME_AP inside this window. Minimum connection time is
    # ~1h; 24h catches overnight layovers. This under-counts, because many
    # connections change equipment -- treat the result as a lower bound.
    MIN_CONNECT_H = 1.0
    MAX_CONNECT_H = 24.0

    # --- Seats -------------------------------------------------------------------
    # ADS-B carries no seat counts and no passenger numbers. Until the organisers'
    # capacity data lands, seats are an explicit assumption, not a measurement.
    # Wide-belly two-class seat counts, used only to scale assumptions; they are
    # deliberately NOT presented as observed data.
    SEAT_BAND_NARROWBODY = 174
    SEAT_BAND_WIDEBODY = 300

    return (
        API_DIR,
        ARRIVAL_COL,
        ARRIVAL_TIME_COL,
        CREDITS_PER_REQUEST,
        DATA_DIR,
        DEPARTURE_COL,
        DEPARTURE_TIME_COL,
        FLIGHT_TABLE,
        HOME_AP,
        MAX_CONNECT_H,
        MIN_CONNECT_H,
        REF_DIR,
        START,
        STOP,
        as_utc,
    )


@app.cell(hide_code=True)
def client(Path, REST, json, opensky_rest):
    # REST API client. The credentials live at the repo root and are gitignored; they
    # are read here rather than copied into ~/.config/pyopensky/settings.conf so
    # that no second copy of the secret exists on disk.
    #
    # `REST.__init__` copies the module-level `client_id`/`client_secret` at
    # construction time, so these globals must be set *before* the instance is
    # built. That is the one ordering trap in this cell.
    CREDENTIALS_PATH = Path(__file__).resolve().parent.parent / "credentials.json"
    _token_state = {"remaining": None}


    def _load_rest_client(credentials_path=CREDENTIALS_PATH):
        """Build an authenticated `REST` client from the repo's credentials file."""
        creds = json.loads(Path(credentials_path).read_text())
        missing = {"clientId", "clientSecret"} - set(creds)
        if missing:
            raise FileNotFoundError(
                f"{credentials_path} is missing {sorted(missing)}; "
                "REST pulls need an OpenSky OAuth client."
            )
        opensky_rest.client_id = creds["clientId"]
        opensky_rest.client_secret = creds["clientSecret"]
        client = REST()
        if client.token is None:
            raise RuntimeError("OpenSky rejected the credentials; no access token issued.")
        return client


    def rest_get(client, url):
        """`REST.get` but with the rate-limit header captured and 429 handled.

        `REST.get` discards `X-Rate-Limit-Remaining` after warning on it, and its
        429 branch recurses with no cap -- at 30 credits a request, a long pull
        needs to know where the budget stands and must not spin.
        """
        response = client.client.get(url, headers=client.headers)
        remaining = response.headers.get("X-Rate-Limit-Remaining")
        _token_state["remaining"] = int(remaining) if remaining is not None else None

        if response.status_code == 429:
            retry_after = int(response.headers.get("X-Rate-Limit-Retry-After-Seconds", 60))
            raise RuntimeError(
                f"Out of OpenSky credits ({_token_state['remaining']} left). "
                f"Quota refills daily; retry in {retry_after}s."
            )
        if response.status_code == 404:
            return []  # documented: "no flights found in this interval"
        response.raise_for_status()
        return response.json()


    def credits_remaining():
        """OpenSky credits left today, or None before the first request."""
        return _token_state["remaining"]


    rest = _load_rest_client()

    return credits_remaining, rest_get


@app.cell
def probe(
    ARRIVAL_COL,
    DEPARTURE_COL,
    FLIGHT_TABLE,
    FlightsData4,
    FlightsData5,
    HOME_AP,
    START,
    as_utc,
    or_,
    pd,
    select,
):
    def probe(trino, day=None, Table=FLIGHT_TABLE, arrival_col=ARRIVAL_COL, departure_col=DEPARTURE_COL, n_days=3):
        """Cheap, partition-pruned sanity checks. Run this before any bulk pull.

        Answers three things that are not documented and that silently ruin a
        backfill if guessed wrong:
          1. is the chosen table populated for this window at all?
          2. is the chosen arrival/departure column actually filled in?
          3. how many legs a day are we really talking about?
        """
        day = as_utc(day or START)
        days = pd.date_range(day, periods=n_days, freq="1D")
        lo, hi = days.min() - pd.Timedelta("1D"), days.max() + pd.Timedelta("1D")

        rows = []
        for label, tbl, arr, dep in [
            ("data4/est", FlightsData4, "estarrivalairport", "estdepartureairport"),
            ("data5/confirmed", FlightsData5, "airportofdestination", "airportofdeparture"),
        ]:
            for d in days:
                try:
                    q = select( tbl.day,
                        arr,
                        dep,
                    ).where(
                        tbl.day >= lo,
                        tbl.day < hi,
                        or_(arr == HOME_AP, dep == HOME_AP),
                    )
                    r = trino.query(q)
                except Exception as exc:  # table may not exist / no access yet
                    rows.append(dict(table=label, day=f"{d:%Y-%m-%d}", result=f"ERROR: {type(exc).__name__}"))
                    continue
                legs = r if r is not None else pd.DataFrame()
                if len(legs) == 0:
                    rows.append(dict(table=label, day=f"{d:%Y-%m-%d}", result="no rows"))
                    continue
                rows.append(
                    dict(
                        table=label,
                        day=f"{d:%Y-%m-%d}",
                        result=f"{len(legs)} legs",
                        arr_col=arr,
                        dep_col=dep,
                        arr_filled=f"{legs[arr].notna().mean():.0%}",
                        dep_filled=f"{legs[dep].notna().mean():.0%}",
                        codes=", ".join(sorted(set(legs[arr].dropna()) | set(legs[dep].dropna()))),
                    )
                )
        return pd.DataFrame(rows)


    return


@app.cell
def backfill(
    ARRIVAL_COL,
    ARRIVAL_TIME_COL,
    DATA_DIR,
    DEPARTURE_COL,
    DEPARTURE_TIME_COL,
    FLIGHT_TABLE,
    HOME_AP,
    Path,
    START,
    STOP,
    as_utc,
    or_,
    pd,
    select,
):
    def leg_columns(Table, arrival_col=ARRIVAL_COL, departure_col=DEPARTURE_COL):
        """Columns to pull, de-duplicated and filtered to what `Table` actually has."""
        wanted = ["icao24", "callsign", "day", "firstseen", "lastseen", arrival_col, departure_col]
        extra = ["airportofdeparture", "airportofdestination", "takeofftime", "landingtime"]
        available = set(Table.__table__.columns.keys())
        return [c for c in dict.fromkeys(wanted + extra) if c in available]


    def fetch_window(trino, day, Table=FLIGHT_TABLE, arrival_col=ARRIVAL_COL, departure_col=DEPARTURE_COL, cached=True):
        """One partition-pruned query for the legs touching HOME_AP on `day`.

        OpenSky partitions `flights_*` by the flight's own start, not by arrival, so
        a leg that lands just after midnight UTC usually lives in the *previous*
        day's partition. Querying `day` alone silently drops overnight long-haul
        arrivals -- exactly the inbound European legs that matter most here. So we
        read a three-partition window and let `backfill` bucket on event time.
        """
        day = as_utc(day)
        lo, hi = day - pd.Timedelta("1D"), day + pd.Timedelta("1D")
        names = leg_columns(Table, arrival_col, departure_col)
        stmt = (
            select(*[getattr(Table, n) for n in names])
            .where(
                Table.day >= lo,
                Table.day < hi,
                or_(getattr(Table, arrival_col) == HOME_AP, getattr(Table, departure_col) == HOME_AP),
            )
        )
        res = trino.query(stmt, cached=cached)
        if res is None or len(res) == 0:
            return pd.DataFrame({n: pd.Series(dtype="object") for n in names})
        return res.rename(columns={arrival_col: "arrival", departure_col: "departure"})


    def backfill(trino, start=START, stop=STOP, Table=FLIGHT_TABLE, arrival_col=ARRIVAL_COL, departure_col=DEPARTURE_COL, data_dir=DATA_DIR, cached=True, verbose=True):
        """Pull legs day by day into hive-partitioned parquet under `DATA_DIR`.

        Idempotent and resumable: a day whose file already exists is read from disk
        instead of re-queried, so an interrupted run costs nothing to restart and
        never re-spends OpenSky query quota. Each leg is written to exactly one
        file (bucketed on the timestamp of the event that makes it interesting), so
        concatenating the directory yields no duplicates.
        """
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        days = pd.date_range(as_utc(start), as_utc(stop), freq="1D", inclusive="left")
        frames = []

        for i, day in enumerate(days, start=1):
            target = data_dir / f"day={day:%Y-%m-%d}.parquet"
            if target.exists():
                frames.append(pd.read_parquet(target))
                if verbose:
                    print(f"[{i}/{len(days)}] {day:%Y-%m-%d}  cached")
                continue

            legs = fetch_window(trino, day, Table, arrival_col, departure_col, cached=cached)
            arr_t = pd.to_datetime(legs[ARRIVAL_TIME_COL], utc=True, errors="coerce")
            dep_t = pd.to_datetime(legs[DEPARTURE_TIME_COL], utc=True, errors="coerce")
            is_arr = legs["arrival"].eq(HOME_AP) & arr_t.ge(day) & arr_t.lt(day + pd.Timedelta("1D"))
            is_dep = legs["departure"].eq(HOME_AP) & dep_t.ge(day) & dep_t.lt(day + pd.Timedelta("1D"))

            keep = is_arr | is_dep
            legs = legs.loc[keep].assign(
                arrival_time=arr_t[keep],
                departure_time=dep_t[keep],
                is_arrival=is_arr[keep].to_numpy(),
                is_departure=is_dep[keep].to_numpy(),
            )
            legs.to_parquet(target, index=False, compression="zstd")
            frames.append(legs)
            if verbose:
                print(f"[{i}/{len(days)}] {day:%Y-%m-%d}  {len(legs)} legs "
                      f"({int(is_arr[keep].sum())} in / {int(is_dep[keep].sum())} out)")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


    return


@app.cell(hide_code=True)
def backfill_rest(
    API_DIR,
    CREDITS_PER_REQUEST,
    HOME_AP,
    Path,
    START,
    STOP,
    as_utc,
    credits_remaining,
    pd,
    rest_get,
):
    # REST returns two independent event streams -- arrivals and departures -- not
    # one row per leg. They are stored as one row per *event* with a `direction`
    # flag, because collapsing them into a single leg row would require guessing
    # which arrival pairs with which departure when an aircraft turns more than
    # once a day, fabricating legs and corrupting the transit signal.
    #
    # Kept beyond `REST.arrival()`'s hard-coded 6-column select: the airport-distance
    # and candidate-count fields quantify how trustworthy each estimated airport is,
    # which is the basis for the origin-ambiguity sensitivity check. `REST.arrival()`
    # also drops rows whose callsign is null, which would bias leg counts downward.
    REST_COLUMNS = [
        "icao24",
        "callsign",
        "firstSeen",
        "lastSeen",
        "estDepartureAirport",
        "estArrivalAirport",
        "estDepartureAirportHorizDistance",
        "estArrivalAirportHorizDistance",
        "departureAirportCandidatesCount",
        "arrivalAirportCandidatesCount",
    ]

    EVENT_COLUMNS = [
        "icao24", "callsign", "direction", "event_time", "day",
        "origin", "dest", "counterpart", "origin_distance_m",
        "origin_candidates", "home_distance_m", "home_candidates",
        "origin_known",
    ]


    def _events_from(payload, direction, home_ap):
        """Normalise one API payload into the event-row schema."""
        if not payload:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(payload)
        missing = {"icao24", "estDepartureAirport"} - set(frame.columns)
        if missing:
            raise KeyError(f"unexpected /flights/* schema, missing {sorted(missing)}")
        for column in REST_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[REST_COLUMNS]

        # firstSeen is estimated takeoff, lastSeen estimated landing. For an
        # arrival the landing time is what we want; for a departure, the takeoff.
        frame["event_time"] = pd.to_datetime(
            frame["lastSeen" if direction == "arrival" else "firstSeen"], utc=True, unit="s"
        )
        frame["counterpart"] = frame[
            "estDepartureAirport" if direction == "arrival" else "estArrivalAirport"
        ]
        frame = frame.rename(columns={"estDepartureAirport": "origin", "estArrivalAirport": "dest"})
        frame["direction"] = direction
        frame["day"] = frame["event_time"].dt.floor("1D")
        # Origin attribution quality, carried through to the sensitivity analysis.
        frame["origin_distance_m"] = pd.to_numeric(frame["estDepartureAirportHorizDistance"], errors="coerce")
        frame["origin_candidates"] = pd.to_numeric(frame["departureAirportCandidatesCount"], errors="coerce")
        frame["home_distance_m"] = pd.to_numeric(frame["estArrivalAirportHorizDistance"], errors="coerce")
        frame["home_candidates"] = pd.to_numeric(frame["arrivalAirportCandidatesCount"], errors="coerce")
        frame["origin_known"] = frame["origin"].notna()
        return frame[EVENT_COLUMNS]


    def fetch_window_rest(client, begin, end, airport=HOME_AP):
        """Arrivals and departures at `airport` over [begin, end). Two requests.

        The window is at most 2 days: the API rejects anything wider with HTTP 400
        ("You can only query across 2 partitions"). That limit happens to be free --
        measured, a 1-day and a 2-day window both bill 30 credits, so two days of
        data costs the same as one and this is the cheapest possible cover.
        """
        begin, end = as_utc(begin), as_utc(end)
        if (end - begin) > pd.Timedelta("2D"):
            raise ValueError(f"window {(end - begin)} exceeds the API's 2-day maximum")
        out = []
        for direction in ("arrival", "departure"):
            url = (
                f"https://opensky-network.org/api/flights/{direction}"
                f"?begin={begin.timestamp():.0f}&end={end.timestamp():.0f}&airport={airport}"
            )
            out.append(_events_from(rest_get(client, url), direction, airport))
        frames = [f for f in out if not f.empty]
        if not frames:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _read_no_data_marker(data_dir):
        marker = Path(data_dir) / "no_data_days.txt"
        if not marker.exists():
            return set()
        return {line.strip() for line in marker.read_text().splitlines() if line.strip()}


    def _write_no_data_marker(data_dir, days):
        marker = Path(data_dir) / "no_data_days.txt"
        existing = _read_no_data_marker(data_dir)
        updated = sorted(existing | {d for d in days})
        marker.write_text("\n".join(updated) + ("\n" if updated else ""))


    def backfill_rest(client, start=START, stop=STOP, airport=HOME_AP, data_dir=API_DIR, span_days=2, fetch=None, verbose=True):
        """Pull one parquet per day into `data_dir`, resumable and credit-aware.

        Fetches in `span_days`-wide windows (cheapest per day of data) but still
        writes one file per day, so an interrupted run resumes at day granularity
        rather than re-fetching a whole window.

        Three rules that the credit model forces:
          * reserve before starting -- if the balance cannot cover both requests of a
            window, stop cleanly rather than burn 30 credits on half a window;
          * never cache an empty day -- a day with no flights is recorded in
            `no_data_days.txt` rather than as a day-file, so a future run can
            re-test it if OpenSky backfills the gap (which it does: the 2025 Gulf
            data that returns 404 today may not stay missing);
          * credits are spent per request whether or not the run completes.

        `fetch` is a seam for testing the credit and empty-day logic without
        spending a credit; it defaults to the real fetcher.
        """
        fetch = fetch or fetch_window_rest
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        days = pd.date_range(as_utc(start), as_utc(stop), freq="1D", inclusive="left")
        no_data = _read_no_data_marker(data_dir)

        # Keyed by day, so a day fetched inside a 2-day window is not appended a
        # second time when the loop later reaches it as a cache hit. A plain list
        # double-counts every fetched day in the returned frame.
        collected = {}
        pending_no_data = set()
        stop_reason = None
        stop_at = None

        for i, day in enumerate(days, start=1):
            target = data_dir / f"day={day:%Y-%m-%d}.parquet"
            if target.exists():
                key = f"{day:%Y-%m-%d}"
                if key not in collected:
                    collected[key] = pd.read_parquet(target)
                if verbose:
                    print(f"[{i}/{len(days)}] {day:%Y-%m-%d}  cached")
                continue
            if f"{day:%Y-%m-%d}" in no_data:
                if verbose:
                    print(f"[{i}/{len(days)}] {day:%Y-%m-%d}  known-empty (skipped)")
                continue

            # Extend the window to span_days, stopping at the range end.
            window_end = min(day + pd.Timedelta(days=span_days), as_utc(stop))

            # Reserve before spending: a window needs one request per direction.
            needed = 2 * CREDITS_PER_REQUEST
            left = credits_remaining()
            if left is not None and left < needed:
                stop_reason = (
                    f"insufficient credits: {left} left, {needed} needed for a "
                    f"{span_days}-day window"
                )
                stop_at = day
                if verbose:
                    print(f"\nstopped at {day:%Y-%m-%d}: {stop_reason}")
                break

            events = fetch(client, day, window_end, airport)
            covered = [day + pd.Timedelta(days=d) for d in range((window_end - day).days)]
            written = []
            for covered_day in covered:
                part = events.loc[events["day"].eq(covered_day)] if len(events) else events
                if len(part) == 0:
                    pending_no_data.add(f"{covered_day:%Y-%m-%d}")
                    continue
                part.to_parquet(
                    data_dir / f"day={covered_day:%Y-%m-%d}.parquet", index=False, compression="zstd"
                )
                collected[f"{covered_day:%Y-%m-%d}"] = part
                written.append(covered_day)

            if verbose:
                n_in = int(events["direction"].eq("arrival").sum()) if len(events) else 0
                n_out = int(events["direction"].eq("departure").sum()) if len(events) else 0
                summary = ", ".join(
                    f"{d:%m-%d}={int((events['day'] == d).sum())}" for d in written
                ) or "none"
                print(
                    f"[{i}/{len(days)}] {(window_end - day).days}d window "
                    f"{day:%Y-%m-%d}..{window_end - pd.Timedelta('1D'):%Y-%m-%d}: "
                    f"{n_in} in / {n_out} out  [{summary}]  credits left: {credits_remaining()}"
                )

        if pending_no_data:
            _write_no_data_marker(data_dir, pending_no_data)
            if verbose:
                print(f"recorded {len(pending_no_data)} empty day(s) in no_data_days.txt")

        if verbose and stop_at is not None:
            print(
                f"\nPARTIAL: stopped at {stop_at:%Y-%m-%d} ({stop_reason}). "
                f"Re-run after the daily quota refills to continue."
            )

        if not collected:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        return pd.concat(
            [collected[k] for k in sorted(collected)], ignore_index=True
        )

    return


@app.cell
def transit(HOME_AP, MAX_CONNECT_H, MIN_CONNECT_H, np, pd):
    def tag_connections(events, home_ap=HOME_AP, min_h=MIN_CONNECT_H, max_h=MAX_CONNECT_H):
        """Tag arrivals whose onward flight is the *same aircraft* leaving `home_ap`.

        Operates on the event stream (one row per arrival or departure, `direction`
        flag) rather than one row per leg. That is deliberate: an aircraft can turn
        more than once a day, and collapsing events into a leg row would require
        guessing which arrival pairs with which departure. Matching events in time
        order handles repeated turns correctly.

        This is the only part of the brief's chain that OpenSky can actually measure
        rather than assume. It is a deliberate lower bound, not a transfer share:

          * a connection is only visible when the pax stays on the same airframe, and
            most hub connections at AUH change equipment, so real transit is higher;
          * the aircraft is also the thing that turns around, so crew-only and
            positioning legs are counted as connections too;
          * ADS-B sees flights, never people -- one aircraft does not tell us how many
            of its seats were transferred.

        Because of the first point, treat `transit_share` as a floor and carry the
        rest of the uncertainty as an explicit range in the scenario layer.
        """
        if events is None or len(events) == 0:
            return events

        # Idempotent: `load_events` already applies this transform, and callers may
        # wrap it again. Re-running must not double-apply, and must not trip over
        # columns this function itself added on a previous pass.
        if "transit" in events.columns or "connection_h" in events.columns:
            return events

        out = events.copy()
        tol = pd.Timedelta(hours=max_h)
        out["_row"] = np.arange(len(out))

        arr = out.loc[out["direction"].eq("arrival"), ["_row", "icao24", "event_time", "day"]].dropna(subset=["event_time"])
        dep = out.loc[out["direction"].eq("departure"), ["icao24", "event_time", "dest"]].dropna(subset=["event_time"])
        dep = dep.rename(columns={"event_time": "dep_time", "dest": "onward_to"})

        if len(arr) == 0 or len(dep) == 0:
            out["connection_h"] = np.nan
            out["bounce_h"] = np.nan
            out["onward_to"] = pd.NA
            out["transit"] = False
            return out.drop(columns="_row")

        matched = pd.merge_asof(
            arr.sort_values("event_time"),
            dep.sort_values("dep_time"),
            left_on="event_time",
            right_on="dep_time",
            by="icao24",
            direction="forward",
            tolerance=tol,
        )
        gap_h = (matched["dep_time"] - matched["event_time"]).dt.total_seconds() / 3600
        matched["connection_h"] = gap_h
        matched["transit"] = gap_h.ge(min_h)

        # A second signature: the aircraft leaves and comes straight back. Round trips
        # are not visitors arriving in Abu Dhabi, and this catches overnight layovers
        # that same-aircraft matching alone would miss.
        #
        # `bounce_h` is initialised up front: when no arrival clears `min_h` the match
        # set is empty, and building the column only inside that branch left the final
        # merge referencing a column that was never created.
        matched["bounce_h"] = np.nan
        back = matched.loc[matched["transit"], ["icao24", "dep_time", "_row"]]
        if len(back) > 0:
            bounced = pd.merge_asof(
                back.sort_values("dep_time"),
                arr[["icao24", "event_time"]].sort_values("event_time"),
                left_on="dep_time",
                right_on="event_time",
                by="icao24",
                direction="forward",
                tolerance=tol,
            ).dropna(subset=["_row"])
            if len(bounced) > 0:
                bounce_h = (bounced["event_time"] - bounced["dep_time"]).dt.total_seconds() / 3600
                matched = matched.drop(columns="bounce_h").merge(
                    pd.DataFrame({"_row": bounced["_row"], "bounce_h": bounce_h}),
                    on="_row",
                    how="left",
                )

        out = out.merge(
            matched[["_row", "connection_h", "onward_to", "transit", "bounce_h"]],
            on="_row",
            how="left",
        )
        out["transit"] = out["transit"].fillna(False).astype(bool)
        return out.drop(columns="_row")


    return (tag_connections,)


@app.cell
def reference(Path, REF_DIR, pd, urllib):
    AIRPORTS_CSV_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


    def load_airports(url=AIRPORTS_CSV_URL, dest=None):
        """Airport -> ISO country lookup, cached on disk.

        This is the join key the whole model turns on. The brief's data catch is that
        flight data records where a flight *departed from* while hotel data records
        the guest's *nationality*; nothing reconciles the two for you. OurAirports
        gives `ident` (the ICAO code wherever one exists) -> `iso_country`, which is
        the cheapest defensible bridge. It is a departure-country proxy, not a
        nationality, and the brief should say so.
        """
        dest = Path(dest) if dest is not None else REF_DIR / "ourairports_airports.csv"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as resp:
                dest.write_bytes(resp.read())
        return pd.read_csv(dest, dtype=str, keep_default_na=False)


    def airport_country_map(airports=None):
        """Merge the three code columns into one ICAO-keyed lookup.

        `icao_code` is the explicit column but is blank for small fields; `ident` and
        `gps_code` fill in the rest. They never disagree on country (checked), so the
        merge order only affects coverage, not correctness.
        """
        airports = load_airports() if airports is None else airports
        mapping = {}
        for col in ("icao_code", "ident", "gps_code"):
            codes = airports[col].str.upper().str.strip()
            keep = codes.ne("") & airports["iso_country"].ne("")
            mapping.update(dict(zip(codes[keep], airports["iso_country"][keep])))
        return mapping


    def add_country(events, mapping=None):
        """Attach an origin country to each event.

        `origin` is null for roughly 37% of OMAA arrivals -- OpenSky could not
        identify where the flight came from. Those become "UNKNOWN" rather than being
        dropped, so the coverage panel can report them; the market table filters them
        out explicitly and `sensitivity` reports the result both ways.
        """
        if events is None or len(events) == 0:
            return events
        mapping = airport_country_map() if mapping is None else mapping
        out = events.copy()
        out["origin_country"] = out["origin"].str.upper().map(mapping).fillna("UNKNOWN")
        return out


    return add_country, airport_country_map


@app.cell
def aggregate(
    API_DIR,
    HOME_AP,
    Path,
    add_country,
    airport_country_map,
    as_utc,
    credits_remaining,
    pd,
    tag_connections,
):
    def load_events(data_dir=None):
        """Read every day-file either backfill has written, country-tagged and matched.

        `add_country` and `tag_connections` are applied here rather than left to the
        caller: both add columns that `monthly_market` requires, so skipping them
        fails deep inside a groupby with a confusing KeyError.
        """
        frames = []
        for folder in (data_dir, API_DIR if data_dir is None else None):
            if folder is None:
                continue
            files = sorted(Path(folder).glob("day=*.parquet"))
            if files:
                frames.extend(pd.read_parquet(f) for f in files)
        if not frames:
            return pd.DataFrame()
        events = pd.concat(frames, ignore_index=True)
        if "transit" not in events.columns or "origin_country" not in events.columns:
            events = tag_connections(add_country(events))
        return events


    def monthly_market(events, max_origin_distance_m=None):
        """Monthly arrivals by origin market -- the grain success criterion 05 asks for.

        This is the top of the brief's chain, and deliberately the last observable
        step: `inbound_visitors` counts *flights*, not people. Turning flights into
        visitors, and visitors into guest nights, needs the seats/load-factor/
        transfer assumptions, which live in the scenario layer, not here.

        `max_origin_distance_m` is an origin-attribution quality filter. The REST API
        estimates the departure airport and reports how far the aircraft was still
        from it, so this bounds how much of the mix rests on OpenSky's airport
        matching. `sensitivity` reports the result at several thresholds.

        Rows with an unidentifiable origin are excluded here -- they have no country
        to attribute -- and counted separately in `coverage_report`.
        """
        if events is None or len(events) == 0:
            return pd.DataFrame()

        arr = events.loc[events["direction"].eq("arrival") & events["origin"].notna()].copy()
        if max_origin_distance_m is not None:
            arr = arr.loc[arr["origin_distance_m"].le(max_origin_distance_m)]
        if arr.empty:
            return pd.DataFrame()

        arr["month"] = arr["event_time"].dt.strftime("%Y-%m")
        agg = (
            arr.groupby(["month", "origin", "origin_country"], dropna=False)
            .agg(
                flights=("icao24", "size"),
                aircraft=("icao24", "nunique"),
                connections=("transit", "sum"),
            )
            .reset_index()
        )
        agg["inbound_visitors"] = agg["flights"] - agg["connections"]
        agg["connection_share"] = (agg["connections"] / agg["flights"]).round(3)
        return agg.sort_values(["month", "flights"], ascending=[True, False]).reset_index(drop=True)


    def outage_blocks(daily):
        """Contiguous runs of low/sparse days, which is how coverage actually fails.

        Receiver outages are not scattered thin days, they are blocks -- this pull
        contains a 10-day block in July. Grouping the flags makes that visible, and
        a block is a reason to distrust every daily total inside it rather than to
        quietly average it away.
        """
        empty = pd.DataFrame(
            columns=["start", "end", "days", "min_events", "max_events", "severity"]
        )
        if daily is None or daily.empty:
            return empty
        frame = daily.copy()
        frame["day"] = pd.to_datetime(frame["day"], utc=True)
        frame = frame.sort_values("day")
        bad = frame["coverage_flag"].isin(["low", "sparse"])
        if not bad.any():
            return empty
        # A new block starts wherever the previous day was healthy or absent.
        run_id = (bad != bad.shift(fill_value=False)).cumsum()
        frame["block"] = run_id.where(bad)
        rows = []
        for _, group in frame.dropna(subset=["block"]).groupby("block"):
            rows.append({
                "start": group["day"].min().strftime("%Y-%m-%d"),
                "end": group["day"].max().strftime("%Y-%m-%d"),
                "days": len(group),
                "min_events": int(group["events"].min()),
                "max_events": int(group["events"].max()),
                "severity": "sparse" if (group["coverage_flag"] == "sparse").all() else "mixed",
            })
        return pd.DataFrame(rows).sort_values("days", ascending=False).reset_index(drop=True)


    def monthly_coverage(daily):
        """How well each month was observed, for labelling the monthly market table.

        A month with few observed days or many sparse days cannot be compared
        like-for-like with a fully-observed one, and that is a property of the
        measurement, not of demand.
        """
        if daily is None or daily.empty:
            return pd.DataFrame(
                columns=["month", "days_observed", "days_in_month", "coverage_of_month",
                         "median_daily_events", "sparse_days", "low_days"]
            )
        frame = daily.copy()
        frame["month"] = pd.to_datetime(frame["day"], utc=True).dt.strftime("%Y-%m")
        grouped = frame.groupby("month").agg(
            days_observed=("day", "nunique"),
            median_daily_events=("events", "median"),
            sparse_days=("coverage_flag", lambda s: int((s == "sparse").sum())),
            low_days=("coverage_flag", lambda s: int((s == "low").sum())),
        ).reset_index()
        grouped["days_in_month"] = [
            pd.Period(m).days_in_month for m in grouped["month"]
        ]
        grouped["coverage_of_month"] = grouped["days_observed"] / grouped["days_in_month"]
        return grouped


    def sensitivity(events):
        """How much the market mix depends on trusting OpenSky's origin estimate."""
        rows = []
        for label, kwargs in [
            ("all identified origins", {}),
            ("origin within 10 km", {"max_origin_distance_m": 10000}),
            ("origin within 5 km", {"max_origin_distance_m": 5000}),
            ("origin within 2 km", {"max_origin_distance_m": 2000}),
        ]:
            table = monthly_market(events, **kwargs)
            if table.empty:
                continue
            by_country = table.groupby("origin_country")["flights"].sum()
            total = by_country.sum()
            rows.append(
                {
                    "filter": label,
                    "arrivals used": int(total),
                    "share of arrivals kept": f"{total / max(len(events[events['direction'].eq('arrival') & events['origin'].notna()]), 1):.1%}",
                    "markets": int(len(by_country)),
                    "top 3 share": round(float(by_country.sort_values(ascending=False).head(3).sum() / total), 3),
                }
            )
        return pd.DataFrame(rows)


    def rotation_diagnostics(events, home_ap=HOME_AP):
        """Why the raw connection rate overstates passenger connections.

        Same-aircraft matching cannot tell a passenger who stayed on board from a
        cabin that was simply turned around and reloaded. At OMAA -- where Etihad
        (ETD) is based -- that distinction dominates the number, so the raw
        `transit_share` is a measure of *aircraft rotations*, not of transferring
        passengers. These rows are the evidence a reviewer needs to see that.
        """
        if events is None or len(events) == 0:
            return pd.DataFrame()
        arr = events.loc[events["direction"].eq("arrival")].copy()
        arr["airline"] = arr["callsign"].fillna("").str.strip().str[:3]

        transiting = arr.loc[arr["transit"]]
        onward = transiting.dropna(subset=["onward_to"])
        # Onward to the airport it came from: the aircraft went back where it started,
        # so the round trip is one aircraft's schedule, not a passenger's connection.
        reversed_leg = onward["onward_to"].eq(onward["origin"])
        # Already recorded as an OMAA->OMAA leg: a domestic or repositioning movement,
        # which is not inbound demand at all.
        already_home = arr["origin"].eq(home_ap)

        return pd.DataFrame(
            [
                {"metric": "arrivals", "value": str(len(arr)), "note": "every arrival in window"},
                {
                    "metric": "same-aircraft departure within window",
                    "value": f"{arr['transit'].mean():.1%}",
                    "note": "CONFOUNDED -- see rows below",
                },
                {
                    "metric": "  onward airport unknown",
                    "value": f"{transiting['onward_to'].isna().mean():.1%}",
                    "note": "cannot classify; excluded from the ratios below",
                },
                {
                    "metric": "  onward == origin airport",
                    "value": f"{reversed_leg.mean():.1%}",
                    "note": "aircraft returned to where it came from: a rotation",
                },
                {
                    "metric": "  onward == a different airport",
                    "value": f"{(~reversed_leg).mean():.1%}",
                    "note": "plausible onward connection, still aircraft-based",
                },
                {
                    "metric": "arrivals already recorded at OMAA",
                    "value": f"{already_home.mean():.1%}",
                    "note": "domestic/repositioning legs, not inbound demand",
                },
                {
                    "metric": "largest airline share of arrivals",
                    "value": (
                        f"{arr['airline'].value_counts().index[0]} "
                        f"{arr['airline'].value_counts().iloc[0] / len(arr):.1%}"
                    ),
                    "note": "hub-carrier concentration inflates the rotation share",
                },
            ]
        )


    def coverage_report(events):
        """Data-quality panel. The numbers a reviewer should be told about, not buried."""
        if events is None or len(events) == 0:
            return pd.DataFrame()
        arr = events.loc[events["direction"].eq("arrival")]
        dep = events.loc[events["direction"].eq("departure")]
        unmapped = sorted(set(events["origin"].dropna()) - set(airport_country_map()))
        span = events["event_time"].dropna()
        identified = int(arr["origin"].notna().sum())
        return pd.DataFrame(
            [
                ("events in window", f"{len(events):,}"),
                ("arrivals / departures", f"{len(arr):,} / {len(dep):,}"),
                ("distinct aircraft", f"{events['icao24'].nunique():,}"),
                ("distinct origin airports", f"{events['origin'].nunique():,}"),
                ("arrivals with identified origin", f"{identified:,} ({identified / max(len(arr), 1):.1%})"),
                ("callsigns missing", f"{int(events['callsign'].isna().sum() + (events['callsign'] == '').sum()):,}"),
                ("airport->country unmapped", f"{len(unmapped):,}" + (f"  e.g. {unmapped[:5]}" if unmapped else "")),
                ("same-aircraft connection (NOT a transfer rate)", f"{arr['transit'].mean():.1%}"),
                ("coverage span (UTC)", f"{span.min():%Y-%m-%d} to {span.max():%Y-%m-%d}"),
                ("OpenSky credits left today", f"{credits_remaining():,}" if credits_remaining() is not None else "n/a"),
            ],
            columns=["check", "value"],
        )


    # ---------------------------------------------------------------------------
    # Daily observation quality
    # ---------------------------------------------------------------------------
    # OpenSky's coverage is not uniform in time: receiver density around the Gulf
    # improves sharply through 2026, so early days in a pull legitimately carry far
    # fewer flights than recent ones. A fixed events-per-day threshold would read
    # that as "these months were quiet" when it actually means "we were not looking
    # closely enough yet". Normalising each day against the median of the window
    # that day belongs to makes low days comparable across the ramp.

    COVERAGE_LOW = 0.50
    COVERAGE_SPARSE = 0.25


    def daily_coverage(events, low=COVERAGE_LOW, sparse=COVERAGE_SPARSE, passes=2):
        """Per-day event counts with a coverage flag scaled to a robust reference day.

        `ratio` is that day's events over a typical well-observed day, so it answers
        "how much of a normal day did we see" rather than "was traffic busy".

        The reference is refined over two passes. A single median is not enough:
        receiver outages arrive as *contiguous blocks* (this pull contains a
        10-day block in July), so a single median is dragged down into the outage
        and every healthy day then reads as over-observed. Pass 1 flags against the
        raw median; pass 2 re-references against the median of the days that
        survived pass 1, which is the level a well-covered day actually looks like.
        """
        if events is None or events.empty:
            return pd.DataFrame(
                columns=["day", "events", "arrivals", "departures", "median_events",
                         "ratio", "coverage_flag"]
            )
        frame = events.copy()
        frame["day"] = pd.to_datetime(frame["day"], utc=True)
        daily = (
            frame.groupby("day")
            .agg(
                events=("icao24", "size"),
                arrivals=("direction", lambda s: int((s == "arrival").sum())),
                departures=("direction", lambda s: int((s == "departure").sum())),
            )
            .reset_index()
        )

        reference = float(daily["events"].median()) if len(daily) else float("nan")
        for _ in range(max(1, passes)):
            ratio = daily["events"] / reference if reference else float("nan")
            healthy = daily.loc[ratio.ge(low), "events"]
            if healthy.empty or float(healthy.median()) == reference:
                break
            reference = float(healthy.median())

        daily["median_events"] = reference
        daily["ratio"] = daily["events"] / reference if reference else float("nan")
        daily["coverage_flag"] = pd.cut(
            daily["ratio"],
            bins=[float("-inf"), sparse, low, float("inf")],
            labels=["sparse", "low", "ok"],
            right=False,
        ).astype(str)
        return daily


    def coverage_gaps(daily, start, stop):
        """Days inside the requested range that produced no data at all."""
        if daily.empty:
            return pd.DataFrame(columns=["day", "status"])
        expected = pd.date_range(as_utc(start), as_utc(stop), freq="1D", inclusive="left")
        seen = set(pd.to_datetime(daily["day"], utc=True))
        missing = [d for d in expected if d not in seen]
        return pd.DataFrame({"day": missing, "status": "no data"})



    return (
        coverage_gaps,
        coverage_report,
        daily_coverage,
        load_events,
        monthly_coverage,
        monthly_market,
        outage_blocks,
        rotation_diagnostics,
        sensitivity,
    )


@app.cell
def view(
    API_DIR,
    DATA_DIR,
    HOME_AP,
    START,
    STOP,
    coverage_gaps,
    coverage_report,
    daily_coverage,
    load_events,
    mo,
    monthly_coverage,
    monthly_market,
    outage_blocks,
    rotation_diagnostics,
    sensitivity,
):
    events = load_events()

    if len(events) == 0:
        extraction = mo.callout(
            mo.md(
                f"""
    ### No flight data pulled yet

    Expected day files under `{API_DIR}/` (REST) or `{DATA_DIR}/` (Trino). Neither
    has been extracted.

    **REST path works now** -- the repo's OpenSky OAuth credentials are valid:

    ```python
    events = backfill_rest(rest)       # 30 credits per 2-day window, per direction
    ```

    **Trino remains blocked.** It returned `PERMISSION_DENIED` and needs an approved
    grant, which is the only route to *confirmed* airports via `FlightsData5`:

    1. *My OpenSky -> Request Data Access*, then wait for approval.
    2. Set `OPENSKY_TRINO_USERNAME` (**lowercase**) and `OPENSKY_TRINO_PASSWORD`.
    3. Then `probe(trino)` and `backfill(trino)` as normal.

    The backfill resumes from day files already on disk, so re-running it after a
    quota reset continues where it stopped instead of starting over.
    """
            ),
            kind="info",
        )
    else:
        daily = daily_coverage(events)
        blocks = outage_blocks(daily)
        monthly_cov = monthly_coverage(daily)
        gaps = coverage_gaps(daily, START, STOP)
        sparse_days = daily.loc[daily["coverage_flag"].eq("sparse"), "day"]
        low_days = daily.loc[daily["coverage_flag"].eq("low"), "day"]
        thin_months = monthly_cov.loc[monthly_cov["coverage_of_month"].lt(0.9), "month"]
        reference = daily["median_events"].iloc[0]
        table = monthly_market(events)

        first, last = events["day"].min(), events["day"].max()
        span = f"{first:%Y-%m-%d} to {last:%Y-%m-%d}"

        extraction = mo.vstack(
            [
                mo.md(f"### {HOME_AP} extraction -- {len(events):,} events over {events['day'].nunique()} days"),
                mo.md(f"Range: **{span}**"),
                mo.md("**Daily observation quality**"),
                mo.md(
                    f"Each day is scored against a well-observed reference day "
                    f"({reference:.0f} events). `ratio` is that day's events over that "
                    f"reference, so it answers *how much of a normal day we observed* "
                    f"rather than *was traffic busy*. The reference is re-derived from "
                    f"the healthy days so that a long outage cannot drag it down and "
                    f"make good days look over-observed."
                ),
                daily.assign(day=daily["day"].dt.strftime("%Y-%m-%d")).drop(columns=["median_events"]),
                (
                    mo.callout(
                        mo.md(
                            f"""
    **{len(sparse_days)} sparse and {len(low_days)} low-count day(s) detected.**

    Sparse days carry less than a quarter of a normal day's flights, low days less
    than half. These are *not* quiet trading days -- they are days where OpenSky's
    receiver coverage around the Gulf was thinner than usual, so the flights were
    never observed. Do not read them as a fall in demand.
    """
                        ),
                        kind="warn",
                    )
                    if (len(sparse_days) or len(low_days))
                    else mo.callout(mo.md("Every day in this pull observed a normal share of traffic."), kind="success")
                ),
                (
                    mo.callout(
                        mo.md(
                            f"""
    **Coverage fails in blocks, not as scattered thin days.** {len(blocks)} contiguous
    block(s) were found, the longest running {int(blocks['days'].max()) if len(blocks) else 0}
    day(s) and {int(blocks['days'].sum()) if len(blocks) else 0} day(s) affected in total.
    Every daily and monthly total inside a block understates real traffic by an
    unknown amount, so those periods are unusable for volume estimates.
    """
                        ),
                        kind="warn",
                    )
                    if len(blocks) else None
                ),
                mo.md("**Outage blocks**"),
                blocks,
                mo.md("**Coverage by month**"),
                monthly_cov,
                (
                    mo.vstack([
                        mo.md("**Days with no data at all**"),
                        mo.ui.table(gaps.assign(day=gaps["day"].dt.strftime("%Y-%m-%d"))),
                    ])
                    if len(gaps)
                    else mo.callout(mo.md("No day in the requested range is missing."), kind="success")
                ),
                mo.md("**Monthly totals are not comparable across this range**")
                if len(thin_months)
                else None,
                (
                    mo.callout(
                        mo.md(
                            f"""
    {", ".join(thin_months)} is covered for only part of the month. Coverage improved
    through 2026 as receiver density rose, so a partly-observed month understates
    arrivals without anything having happened in the real world. Read the monthly
    table against `coverage_of_month`, never on its own.
    """
                        ),
                        kind="warn",
                    )
                    if len(thin_months) else None
                ),
                mo.md("**Data quality**"),
                coverage_report(events),
                mo.callout(
                    mo.md(
                        """
    **The connection rate below is not a transfer rate.** Same-aircraft matching
    cannot distinguish a passenger who stayed aboard from a cabin that was turned
    around and reloaded. Because Etihad is based at OMAA, most arriving aircraft
    depart again, so this mostly measures *aircraft rotations*. See the diagnostics
    table. Treat it as a floor on connecting behaviour, and carry the rest as an
    explicit range in the scenario layer.
    """
                    ),
                    kind="warn",
                ),
                mo.md("**Connection diagnostics**"),
                rotation_diagnostics(events),
                mo.md("**Monthly arrivals by origin market**"),
                table.head(60),
                mo.md("**Origin-attribution sensitivity**"),
                mo.md(
                    "OpenSky estimates the departure airport. These filters keep only "
                    "origins it matched confidently, to show how much the market mix "
                    "depends on that estimate rather than on measurement."
                ),
                sensitivity(events),
            ]
        )

    extraction

    return


if __name__ == "__main__":
    app.run()
