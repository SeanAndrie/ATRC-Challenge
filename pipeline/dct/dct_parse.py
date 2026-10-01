"""Build the DCT Abu Dhabi deliverables from the extracted report text.

Usage:  .venv/bin/python pipeline/dct/dct_parse.py
"""
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dct_common import PDFDIR, TEXTDIR, read_text                # noqa: E402
from paths import (                                              # noqa: E402
    DCT_COVERAGE,
    DCT_CROSS_CHECKS,
    DCT_FINGERPRINT,
    DCT_GUESTS_NATIONALITY,
    DCT_HOTEL_SUPPLY,
    DCT_KPIS,
    DCT_NOTES_RAW,
    PROC,
)
import dct_period as P                                    # noqa: E402
from eras import era_2017 as E2017                         # noqa: E402
from eras import era_2019q as E2019Q                       # noqa: E402
from eras import era_2024 as E2024                         # noqa: E402
from eras import era_chart as CHART                        # noqa: E402
from eras import era_key as KEY                            # noqa: E402
from eras import era_main as MAIN                          # noqa: E402
import dct_nationality as NAT                             # noqa: E402
import dct_normalize as NORM                              # noqa: E402
import dct_supply as SUP                                  # noqa: E402

KPI_COLS = ["source_file", "year", "period", "granularity", "segment", "metric",
            "unit", "value", "growth_pct", "ytd", "derived", "source_block"]
NAT_COLS = ["source_file", "year", "period", "granularity", "segment", "rank",
            "nationality", "guests", "guests_unit", "share_pct", "share_basis",
            "alos", "change_pct", "ytd", "source_block"]
SUP_COLS = ["source_file", "year", "period", "as_of", "segment", "category",
            "hotels", "rooms", "source_block"]
CHK_COLS = ["source_file", "check", "left", "right", "status"]

CHART_FILES = (["2017-%02d" % m for m in (11, 12)]
               + ["2018-%02d" % m for m in range(1, 13)]
               + ["2020-%02d" % m for m in range(1, 13)]
               + ["2021-%02d" % m for m in range(1, 4)])


def route(stem):
    if stem == "2024-annual-2024":
        return "annual2024"
    if stem == "2015-02":
        return "chart"          # byte-identical duplicate of 2018-02
    if stem.startswith("2019-Q"):
        return "quarter"
    if stem.startswith(("2015-", "2016-")):
        return "main"
    if stem.startswith("2017-") and stem[5:7] <= "10":
        return "era2017"
    if stem in CHART_FILES:
        return "chart"
    if stem.startswith(("2021-", "2022-", "2023-")):
        return "key"
    return "unknown"


def norm(rows, year, period, block="unknown"):
    out = []
    for r in rows:
        d = {c: r.get(c) for c in KPI_COLS}
        d["year"] = r.get("year", year)
        d["period"] = r.get("period", period)
        if d.get("growth_pct") is None:
            d["growth_pct"] = r.get("prior_pct")
        if d.get("ytd") and d.get("granularity") == "month":
            d["granularity"] = "ytd"
        d.setdefault("segment", "total")
        d.setdefault("ytd", 0)
        d.setdefault("derived", 0)
        if not d.get("source_block"):
            d["source_block"] = block
        out.append(d)
    return out


def fix_december_annual(kpis):
    """December main-era reports print the current month and the full year in
    the same row group; the full-year column arrives as a second 'month' row.

    Which of the pair is the annual total is decided by document order, not by
    magnitude. The main-era parser walks the report top to bottom, so within a
    file the current-month block is emitted before the full-year block and the
    second row of the pair is the annual one. A magnitude test is wrong here:
    it works for additive metrics, where the year dwarfs the month, but it can
    never work for a rate or an average, where the year is by construction close
    to the month. That left alos, arr, occupancy_rate and revpar in the 2015-12
    report filing their full-year average as a December monthly reading.

    Only 'month' groups are considered. Reports that print a current and a
    prior year at 'year' granularity are two-row groups too, and they are
    already correct, so a broader rule would relabel the prior-year row."""
    by = defaultdict(list)
    for r in kpis:
        if r.get("granularity") == "month":
            by[(r["source_file"], r["metric"], r["granularity"], r["unit"])].append(r)
    n = 0
    for rows in by.values():
        if len(rows) != 2:
            continue
        _, annual = rows
        annual["granularity"] = "year"
        annual["ytd"] = 1
        n += 1
    return n


def content_periods():
    """Map each cached text file to the reporting period its content declares."""
    path = DCT_FINGERPRINT
    if not path.exists():
        return {}
    import json
    items = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for it in items:
        p = it.get("period_head")
        if p and re.fullmatch(r"\d{4}-\d{2}", p):
            out[it["file"]] = p
    return out


def misfiled_duplicates():
    """Stems whose cached text is a byte-identical copy of another report.

    One bad manifest link caused the February 2017 'Hotel Report' to be cached
    twice, as both 2015-02 and 2018-02. Parsing both injected a phantom
    2018-02 row (the regional sum of a 2017 document) and gave 2018 a second
    February, so the year summed to more than its published total.

    Only genuine byte-identical copies are treated as duplicates. The
    fingerprint's own period field is far too noisy to gate on: it reads the
    prior-year comparison header, so it reports the 2017 reports as 2016 and
    the 2019 quarters as '2019-??'. Where copies disagree about which stem is
    correct, the text decides: a copy is dropped when the report it contains is
    already covered under its own name.
    """
    groups = defaultdict(list)
    for p in sorted(TEXTDIR.glob("*.txt")):
        groups[hashlib.sha256(p.read_bytes()).hexdigest()].append(p.stem)
    fp = content_periods()
    drop = {}
    for stems in groups.values():
        if len(stems) < 2:
            continue
        inner = {fp.get(s) for s in stems}
        keeper = next((s for s in stems if s in inner), None)
        for s in stems:
            if s != keeper:
                drop[s] = (f"byte-identical to {keeper}" if keeper
                           else f"byte-identical copies of a {sorted(inner)[0]} "
                                f"report cached under {sorted(stems)}")
    return drop


def main():
    files = sorted(p.stem for p in PDFDIR.glob("*.pdf"))
    skip = misfiled_duplicates()
    kpis, nats, sups, checks, coverage, notes = [], [], [], [], [], []
    for stem in files:
        if stem in skip:
            notes.append((stem, "duplicate", f"skipped: {skip[stem]}"))
            coverage.append(dict(source_file=stem, route="duplicate",
                                 year=P.detect(stem).get("year"),
                                 period=P.detect(stem).get("period"),
                                 basis=skip[stem], kpi_rows=0,
                                 nationality_rows=0, supply_rows=0))
            continue
        kind = route(stem)
        det = P.detect(stem)
        year = det.get("year")
        period = det.get("period") or ""
        if "-" in period:
            year, period = int(period[:4]), period[5:7]
        if kind == "annual2024":
            year, period = 2024, "annual"
        nk = nn = ns = 0
        if kind == "main":
            rows = MAIN.parse(read_text(stem), stem, year, int(period))
            kpis += norm(rows, year, period, "main_indicator_summary")
            nk = len(rows)
        elif kind == "era2017":
            rows, nt = E2017.parse(stem, year, period)
            kpis += norm(rows, year, period, "hotel_type_panel")
            nk = len(rows)
            notes += [(stem, kind) + tuple(x) for x in nt]
        elif kind == "chart":
            rows, ck = CHART.parse(stem, year, period)
            kpis += norm(rows, year, period)
            nk = len(rows)
            checks += [(stem, c[0], c[1], c[2] if len(c) > 2 else "",
                        str(c[3]) if len(c) > 3 else "ok") for c in ck]
        elif kind == "quarter":
            rows, nt, ck = E2019Q.parse(stem, year, period)
            kpis += norm(rows, year, period)
            nk = len(rows)
            checks += [(stem,) + (c[0], c[1], c[2], "ok" if c[3] else "MISMATCH")
                       for c in ck]
            notes += [(stem, kind) + tuple(x) for x in nt]
        elif kind == "key":
            rows = KEY.parse(read_text(stem), stem, year, period)
            kpis += norm(rows, year, period, "overall_performance_table")
            nk = len(rows)
        elif kind == "annual2024":
            k, n, s, nt, ck = E2024.parse(stem, year, period)
            kpis += norm(k, year, period)
            nats += n
            sups += s
            nk, nn, ns = len(k), len(n), len(s)
            checks += [(stem,) + (c[0], c[2], c[3], "ok" if c[4] else "MISMATCH")
                       for c in ck]
            notes += [(stem, kind) + tuple(x) for x in nt]
        if kind == "main":
            nrows, nt = NAT.era_2015(stem, year, period)
        elif kind == "era2017":
            nrows, nt = NAT.era_2017(stem, year, period)
        elif kind == "quarter":
            nrows, nt = NAT.era_2019q(stem, year, period)
        elif kind == "key":
            nrows, nt = NAT.era_ranked(stem, year, period)
        elif kind == "chart":
            nrows, nt = NAT.era_chart(stem, year, period)
        else:
            nrows, nt = [], ["nationality_not_applicable"]
        for r in nrows:
            r.setdefault("segment", "total")
            r.setdefault("ytd", 0)
        uniq, seen = [], set()
        for r in nrows:
            key = (r.get("granularity"), r.get("nationality"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        nats += uniq
        nn += len(uniq)
        notes += [(stem, "nationality:" + kind) + tuple(x) for x in nt]
        if kind != "annual2024":
            srows, nt = SUP.parse(stem, year, period, asof=stem)
            uniq, seen = [], set()
            for r in srows:
                key = (r.get("segment"), r.get("category"))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(r)
            sups += uniq
            ns = len(uniq)
            notes += [(stem, "supply:" + kind) + tuple(x) for x in nt]
        coverage.append(dict(source_file=stem, route=kind, year=year,
                             period=period, basis=det.get("basis"),
                             kpi_rows=nk, nationality_rows=nn, supply_rows=ns))
    nfix = fix_december_annual(kpis)
    print("December annual rows relabelled:", nfix)
    nats, nstats = NORM.normalize(nats, kpis)
    print("nationality periods normalized: %d, shares derived: %d"
          % (nstats["periods_normalized"], nstats["shares_derived"]))
    write(DCT_KPIS, KPI_COLS, kpis)
    write(DCT_GUESTS_NATIONALITY, NAT_COLS, nats)
    write(DCT_HOTEL_SUPPLY, SUP_COLS, sups)
    write(DCT_CROSS_CHECKS, CHK_COLS, checks)
    write(DCT_COVERAGE, list(coverage[0]), coverage)
    write(DCT_NOTES_RAW, ["source_file", "stage", "note"], notes)
    bad = [c for c in checks if c[4] == "MISMATCH"]
    print("cross-check mismatches: %d / %d" % (len(bad), len(checks)))
    for c in bad[:20]:
        print("   ", c)
    print("files with no KPI rows:",
          [c["source_file"] for c in coverage if not c["kpi_rows"]])
    return kpis, nats, sups, checks, coverage, notes


def write(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if isinstance(r, tuple):
                r = dict(zip(cols, r))
            w.writerow({c: r.get(c) for c in cols})
    print("wrote %-42s %6d rows" % (path.name, len(rows)))


if __name__ == "__main__":
    main()
