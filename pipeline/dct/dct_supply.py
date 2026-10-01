"""Hotel supply (property and room inventory) extraction.

The supply pages are laid out as floating number/label pairs (a value sits on
one text line and its ``HOTELS`` / ``ROOMS`` / star-class / region label on the
next), so every block is read by matching each number to the nearest label
anchor by column position.  Multi-word labels ("ABU DHABI", "5 STAR") are
matched against the whole line.
"""
import re
from dct_common import read_text, num, canon_region

UNIT = re.compile(r"\b(HOTELS?|ROOMS?)\b", re.I)
REGION = re.compile(r"\b(ABU\s+DHABI|ABU\s+DHABI\s+CITY|AL\s+AIN(?:"
                    r"\s+REGION)?|AL\s+DHAFRA(?:REGION)?)\b", re.I)
CLASS = re.compile(r"\b((?:5|4|3|2|1)[\s\-–]*STARS?|1-3\s*STARS?|3-1\s*STARS?|"
                   r"APTS|APARTMENTS|DELUXE(?:\s*APTS?)?|STANDARD|"
                   r"UNCLASSIFIED)\b", re.I)
NUMBER = re.compile(r"^[\d,]+(?:\.\d+)?[KM]?$", re.I)
BADLINE = re.compile(r"AS OF|Copyright|Source:|PAGE\b|CLOSED|OPEN:", re.I)
SECTION = re.compile(
    r"(HOTEL SUPPLY AS OF [^ ].*|ABU DHABI ISLAND PERFORMANCE|"
    r"CURRENT HOTEL SUPPLY|KEY INDICATORS PER HOTEL TYPE|"
    r"OVERALL HOTEL SUPPLY)", re.I)
PANEL = re.compile(
    r"(Total\s+room\s+capacity\s+([\d,]+)|Occupancy\s+(\d+(?:\.\d+)?)\s*%"
    r"|ALOS\s+([\d.]+)\s*days?|RevPAR\s+([\d,]+)\s*AE\s*D?"
    r"|Number\s+of\s+Hotels:?\s*(\d+))", re.I)


def _anchors(line):
    """[(column, kind, name)] for every label in one line."""
    out = []
    for m in UNIT.finditer(line):
        out.append((m.start(), "hotels" if m.group(0).upper().startswith("H")
                    else "rooms", m.group(0).upper()))
    for m in REGION.finditer(line):
        out.append((m.start(), "region", re.sub(r"\s+", " ", m.group(1)).upper()))
    for m in CLASS.finditer(line):
        out.append((m.start(), "class", re.sub(r"\s+", " ", m.group(1)).upper()))
    out.sort()
    return out


def _toks(line):
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", line)]


def _val(tok):
    m = re.match(r"^([\d,]+(?:\.\d+)?)([KM]?)$", tok, re.I)
    if not m:
        return None
    v = num(m.group(1))
    if v is None:
        return None
    return v * 1000.0 if m.group(2).upper() == "K" else (
        v * 1e6 if m.group(2).upper() == "M" else v)


def _pair_value(lines, j, acol, kind):
    """Value sitting immediately left of the anchor at ``acol`` on line j."""
    best = None
    for c0, c1, txt in _toks(lines[j]):
        if not NUMBER.match(txt):
            continue
        d = acol - c0
        if 0 <= d <= 10 and (best is None or d < best[0]):
            best = (d, _val(txt))
    return best[1] if best else None


def _pairs(lines, i, span=(-2, 1)):
    """(value, kind, region, class) for every number near line i."""
    lo, hi = max(0, i + span[0]), min(len(lines), i + span[1] + 1)
    out = []
    for j in range(lo, hi):
        line = lines[j]
        if BADLINE.search(line):
            continue
        anc = _anchors(line)
        units = [a for a in anc if a[1] in ("hotels", "rooms")]
        for c0, c1, txt in _toks(line):
            if not NUMBER.match(txt):
                continue
            val = _val(txt)
            if val is None or val <= 0 or val > 250000:
                continue
            hit = None
            for ac0, kind, _ in units:            # "11 HOTELS"
                if 0 <= ac0 - c0 <= 10:
                    hit = (kind, ac0)
                    break
            if hit is None:                        # "1,207 ROOMS"
                for ac0, kind, name in units:
                    if 0 <= c0 - (ac0 + len(name)) <= 3:
                        hit = (kind, ac0)
                        break
            if hit is None:                        # value above its label
                for k in (j + 1, j + 2, j - 1, j - 2):
                    if not 0 <= k < len(lines) or BADLINE.search(lines[k]):
                        continue
                    for ac0, akind, aname in _anchors(lines[k]):
                        if akind not in ("hotels", "rooms"):
                            continue
                        d = ac0 - c0
                        if -3 <= d <= 30:
                            hit = (akind, ac0)
                            break
                    if hit:
                        break
            if hit is None:
                continue
            kind, acol = hit
            if kind == "hotels" and 1900 <= val <= 2035:
                continue
            if len(units) >= 2:                      # class / region grid
                ctxs = []
                for k in range(max(0, j - 2), min(len(lines), j + 2)):
                    for ac0, akind, aname in _anchors(lines[k]):
                        if akind in ("region", "class") and (
                                ac0, akind, aname) not in units:
                            ctxs.append((ac0, akind, aname))
                if len(ctxs) == len(units):
                    ordered = sorted(ctxs)
                    pos = dict(((c, kk), nm) for c, kk, nm in ordered)
                    seq = sorted(((ac0, kind), ac0) for ac0, kind, _ in units)
                    names = [pos.get((ac0, kind)) for ac0, kind in seq]
                    if all(names):
                        for (ac0, kind), nm in zip(seq, names):
                            out.append((_pair_value(lines, j, ac0, kind),
                                        kind,
                                        nm if REGION.match(nm) else None,
                                        nm if CLASS.match(nm) else None))
                        continue
            reg = cls = None
            for k in range(max(0, j - 2), min(len(lines), j + 2)):
                ka = [a for a in _anchors(lines[k]) if a[1] in ("region", "class")]
                for ac0, akind, aname in ka:
                    if akind not in ("region", "class"):
                        continue
                    if akind == "class" and sum(1 for a in ka
                                                if a[1] == "class") < 2:
                        continue
                    d = abs(ac0 - acol)
                    if d > 34:
                        continue
                    if akind == "region" and (reg is None or d < reg[1]):
                        reg = (aname, d)
                    if akind == "class" and (cls is None or d < cls[1]):
                        cls = (aname, d)
            out.append((val, kind, reg[0] if reg else None,
                        cls[0] if cls else None))
    return out


WIN = re.compile(r"HOTEL SUPPLY\s+AS\s+OF|CURRENT\s+HOTEL\s+SUPPLY|"
                 r"^\s*HOTEL\s+SUPPLY\s*$|^\s*HOTELS\s+ROOMS\s*$", re.I)


def _windows(lines):
    for i, ln in enumerate(lines):
        if WIN.search(ln):
            span = 14 if re.match(r"^\s*HOTELS\s+ROOMS\s*$", ln) else 40
            yield range(i, min(len(lines), i + span))
            continue
        if re.search(r"Hotels\s+with\s+[\d,]+\s*rooms?", ln, re.I):
            yield range(i, i + 1)


def parse(stem, year, period, asof=None):
    txt = read_text(stem)
    lines = txt.split("\n")
    found = {}
    for win in _windows(lines):
        for i in win:
            line = lines[i]
            if not line.strip() or BADLINE.search(line):
                continue
            island = bool(re.search(r"ABU\s+DHABI\s+ISLAND", "\n".join(
                lines[max(0, i - 6):i]), re.I))
            for val, kind, reg, cls in _pairs(lines, i):
                seg = canon_region(reg) if reg else (
                    "abu_dhabi_island" if island else "total")
                cat = re.sub(r"\s+", " ", cls).upper() if cls else "total"
                key = (seg, cat, kind)
                if key not in found or val > found[key]:
                    found[key] = val
    rows = []
    for (seg, cat, kind), v in sorted(found.items()):
        rec = dict(source_file=stem, year=year, period=period, as_of=asof,
                   segment=seg, category=cat, hotels=None, rooms=None,
                   source_block="supply_page")
        rec[kind] = v
        rows.append(rec)
    pm = re.search(r"KEY\s+INDICATORS\s+PER\s+HOTEL\s+TYPE", txt, re.I)
    if pm:
        vals = {}
        for g in PANEL.finditer(txt[pm.start(): pm.start() + 700]):
            label = g.group(0).lower()
            if "room capacity" in label:
                key, raw = "rooms", g.group(2)
            elif "hotels" in label:
                key, raw = "hotels", g.group(6)
            else:
                continue
            if key not in vals:
                vals[key] = num(raw)
        if vals:
            rows.append(dict(source_file=stem, year=year, period=period,
                             as_of=asof, segment="total", category="total",
                             hotels=vals.get("hotels"), rooms=vals.get("rooms"),
                             source_block="hotel_type_panel"))
    return rows, [] if rows else ["supply_not_reported"]
