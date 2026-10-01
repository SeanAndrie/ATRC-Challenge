"""Era A tables: Main Indicators / Top 10 / Top 25 / Regional Summary / Rooms Summary."""

from __future__ import annotations

import re

from dct_common import canon_nationality, canon_region, canon_star, num

IND_ROWS = {
    "actual guest arrivals": ("guests", "count"),
    "guest arrivals": ("guests", "count"),
    "guest nights": ("guest_nights", "count"),
    "room nights": ("room_nights", "count"),
    "occupied rooms": ("occupied_rooms", "count"),
    "average length of stay": ("alos_days", "days"),
    "occupancy rate %": ("occupancy_pct", "pct"),
    "occupancy %": ("occupancy_pct", "pct"),
    "occupancy rate": ("occupancy_pct", "pct"),
    "room revenue": ("room_revenue_aed", "aed"),
    "food & beverages": ("fnb_revenue_aed", "aed"),
    "other revenue": ("other_revenue_aed", "aed"),
    "total revenue": ("total_revenue_aed", "aed"),
    "arr": ("arr_aed", "aed"),
    "rev par": ("revpar_aed", "aed"),
}

SUPPLY_LABELS = {
    "hotel establishments": "Total",
    "hotels": "Hotels",
    "hotel apartments": "Hotel Apartments",
}

NUMTOK = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?|-?\.\d+")
CONT = re.compile(r"^\s*(?:p|q|u|pu|qu)\s*$", re.I)


def nk(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(" ", " ").strip().lower())


def toks(line: str):
    """[(raw_token, value, is_pct)] preserving order."""
    out = []
    for t in NUMTOK.findall(line):
        v = num(t)
        if v is not None:
            out.append((t, v, t.endswith("%")))
    return out


def row_key(line: str):
    low = nk(line)
    for k in IND_ROWS:
        if low.startswith(k):
            return k
    return None


def find_table_bounds(lines, anchor_re, stop_res):
    """Yield (start, end) line spans for each table whose header matches anchor_re."""
    starts = [i for i, ln in enumerate(lines) if re.search(anchor_re, ln, re.I)]
    out = []
    for s in starts:
        e = len(lines)
        for j in range(s + 1, len(lines)):
            if any(re.search(r, lines[j], re.I) for r in stop_res):
                e = j
                break
        out.append((s, e))
    return out


def period_kind(lines, idx, limit=16):
    for j in range(idx, min(len(lines), idx + limit)):
        s = nk(lines[j])
        if "year to date" in s or s.startswith("ytd"):
            return "ytd"
        if "full year" in s or "full-year" in s:
            return "full_year"
    for j in range(idx, min(len(lines), idx + limit)):
        s = nk(lines[j])
        if "month of" in s or re.match(
            r"^(january|february|march|april|may|june|july|august|september|october|november|december)\b", s
        ):
            return "month"
    return "month"


def parse_indicators(lines, idx, end, year, period, srcfile, out, kind, region=None,
                     granularity="Esimirate", ncols=2):
    """Parse indicator rows.  ncols = 2 (cur, prev) or 4 (month cur/prev, ytd cur/prev)."""
    i = idx
    seen = set()
    while i < end:
        key = row_key(lines[i])
        if key and key not in seen:
            vals = toks(lines[i])
            j = i
            if len(vals) < 2 and i + 1 < end and nk(lines[i + 1]) in (
                "revenues", "revenue", "arrivals", "of stay", "stay", "days"
            ):
                vals = vals + toks(lines[i + 1])
                j += 1
            if len(vals) >= 2:
                seen.add(key)
                metric, unit = IND_ROWS[key]
                plain = [v for _, v, p in vals if not p]
                pcts = [(v, t) for _, v, p in vals if p]
                if unit == "pct":
                    plain = [v / 100.0 if v > 1000 else v for v in plain]
                cur, prev = plain[0], plain[1]
                growth = pcts[0][0] if pcts else None
                if ncols == 2:
                    groups = [(cur, prev, growth, 0)]
                else:
                    ycur = plain[2] if len(plain) > 2 else None
                    yprev = plain[3] if len(plain) > 3 else None
                    ygrowth = pcts[1][0] if len(pcts) > 1 else None
                    groups = [(cur, prev, growth, 0)]
                    if ycur is not None:
                        groups.append((ycur, yprev, ygrowth, 1))
                for val, pval, gr, yflag in groups:
                    if val is None:
                        continue
                    out.append(
                        dict(year=year, period=period, source_file=srcfile,
                             granularity=granularity,
                             segment=region or "Total", metric=metric, unit=unit,
                             value=val, ytd=yflag, prev_year=year - 1,
                             prev_value=pval, growth_pct=gr, derived=0, era="A",
                             label=key)
                    )
        i = j + 1 if key and key not in seen else i + 1
    return len(seen)


# ---------------------------------------------------------------- nationalities
NAME_CHARS = r"[A-Za-z][A-Za-z \.\-\(\)&\'/]*?"


def parse_top10(lines, start, end, period, year, srcfile, sink, ytd):
    i = start
    header = False
    while i < end:
        s = lines[i].strip()
        low = nk(s)
        if "share" in low and "guest" in low:
            header = True
        m = re.match(r"^(\d{1,2})\s+(" + NAME_CHARS + r")\s*(.*)$", s)
        if header and m and 1 <= int(m.group(1)) <= 15:
            rank = int(m.group(1))
            name = m.group(2).strip()
            body = m.group(3)
            vals = toks(body)
            j = i
            plain_need = 2
            if len([1 for _, _, p in vals if not p]) < plain_need and j + 1 < end:
                vals = vals + toks(lines[j + 1])
                j += 1
            pcts = [v for _, v, p in vals if p]
            plain = [v for _, v, p in vals if not p]
            share = pcts[0] if len(pcts) >= 1 else None
            growth = pcts[1] if len(pcts) >= 2 else None
            guests = plain[0] if plain else None
            alos = plain[1] if len(plain) > 1 else None
            sink.append(
                dict(year=year, period=period, source_file=srcfile, rank=rank,
                     nationality=canon_nationality(name), guests=guests,
                     share_pct=share, alos=alos, unit="count", scope="Esimirate",
                     ytd=ytd, guests_prev=None, growth_pct=growth, table="top10")
            )
            i = j + 1
            continue
        if "total hotel establishment guests" in low:
            m2 = re.search(r"[\d,]{3,}", s)
            if m2:
                sink.append(
                    dict(year=year, period=period, source_file=srcfile, rank=None,
                         nationality="TOTAL (printed)", guests=num(m2.group(0)),
                         share_pct=None, alos=None, unit="count", scope="Esimirate",
                         ytd=ytd, guests_prev=None, growth_pct=None,
                         table="top10_total")
                )
        i += 1


TOP25_RE = re.compile(r"^\s*(\d{1,2})\s+(" + NAME_CHARS + r")\s{2,}(.*)$")
BARE_NAME_RE = re.compile(r"^\s*(" + NAME_CHARS + r")\s*$")


def parse_top25(lines, start, end, period, year, srcfile, sink, ytd):
    i = start
    header = False
    while i < end:
        s = lines[i].strip()
        low = nk(s)
        if low.startswith("rank") or ("nationality" in low and re.search(r"20\d\d", low)):
            header = True
        name = None
        nums_line = None
        j = i
        m = TOP25_RE.match(lines[i])
        if m and 1 <= int(m.group(1)) <= 30:
            rank = int(m.group(1))
            name = m.group(2).strip()
            nums_line = m.group(3)
        elif header and BARE_NAME_RE.match(lines[i]) and i + 1 < end:
            nxt = lines[i + 1]
            m2 = re.match(r"^\s*(\d{1,2})\s+(.*)$", nxt)
            if m2 and 1 <= int(m2.group(1)) <= 30 and toks(m2.group(2)):
                rank = int(m2.group(1))
                name = BARE_NAME_RE.match(lines[i]).group(1).strip()
                nums_line = m2.group(2)
                j = i + 1
        if name:
            vals = toks(nums_line)
            if len([1 for _, _, p in vals if not p]) < 4 and j + 1 < end:
                vals = vals + toks(lines[j + 1])
                j += 1
            plain = [v for _, v, p in vals if not p]
            if plain:
                sink.append(
                    dict(year=year, period=period, source_file=srcfile, rank=rank,
                         nationality=canon_nationality(name), guests=plain[0],
                         share_pct=None,
                         alos=plain[4] if len(plain) > 4 else None,
                         unit="count", scope="Esimirate", ytd=ytd,
                         guests_prev=plain[1] if len(plain) > 1 else None,
                         growth_pct=None, table="top25",
                         guest_nights=plain[2] if len(plain) > 2 else None)
                )
                i = j + 1
                continue
        if "total hotel establishment guests" in low:
            m2 = re.search(r"[\d,]{3,}", s)
            if m2:
                sink.append(
                    dict(year=year, period=period, source_file=srcfile, rank=None,
                         nationality="TOTAL (printed)", guests=num(m2.group(0)),
                         share_pct=None, alos=None, unit="count", scope="Esimirate",
                         ytd=ytd, guests_prev=None, growth_pct=None,
                         table="top25_total")
                )
        i += 1


# ------------------------------------------------------------------ supply
SUPPLY_ROW = re.compile(r"^([A-Za-z0-9\-\&\. ]{2,32}?)\s{2,}([\d,]+)\s+([\d,]+)")


def parse_rooms_summary(lines, start, end, year, period, srcfile, supply, region):
    for i in range(start, min(end, len(lines))):
        s = lines[i].strip()
        m = SUPPLY_ROW.match(s)
        if not m:
            continue
        label = re.sub(r"\s+", " ", m.group(1)).strip()
        low = nk(label)
        seg = canon_star(label)
        if seg is None:
            if low in SUPPLY_LABELS:
                seg = SUPPLY_LABELS[low]
            elif region:
                seg = f"{region} - {label}"
            else:
                seg = label
        supply.append(
            dict(year=year, period=period, source_file=srcfile, segment=seg,
                 region=region, hotels_count=num(m.group(2)),
                 rooms_count=num(m.group(3)), rooms_unit="count",
                 scope=region or "Esimirate")
        )
