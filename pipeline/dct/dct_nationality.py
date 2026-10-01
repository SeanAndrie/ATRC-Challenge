"""Guests-by-nationality extraction across every report era.

Five layouts are supported; each returns a list of row dicts with a fixed
schema.  Values that cannot be attributed with confidence are left as None.
"""
import re
from dct_common import read_text, num, canon_nationality

NAT = re.compile(
    r"\b(UAE|United Arab Emirates|United Arab|Saudi Arabia|KSA|Kuwait|Qatar|Bahrain|Oman"
    r"|Egypt|India|China|Japan|Korea|South Korea|Philippines|Thailand|Vietnam"
    r"|Indonesia|Malaysia|Singapore|Hong Kong|Taiwan|Russia|Russian Federation"
    r"|UK|United Kingdom|GB|Great Britain|USA|United States|US|"
    r"Germany|France|Italy|Spain|Netherlands|Switzerland|Belgium|Austria|"
    r"Sweden|Norway|Denmark|Poland|Turkey|Greece|Ireland|Portugal|"
    r"Jordan|Lebanon|Iran|Iraq|Pakistan|Bangladesh|Sri Lanka|Nepal|"
    r"South Africa|Nigeria|Kenya|Egypt Arab Republic|Brazil|Mexico|Canada|"
    r"Australia|New Zealand|Morocco|Algeria|Tunisia|Kazakhstan|Uzbekistan|"
    r"Armenia|Georgia|Azerbaijan|Moldova|Serbia|Croatia|Romania|Bulgaria|"
    r"Czech Republic|Hungary|Slovakia|Slovenia|Estonia|Latvia|Lithuania|"
    r"Cyprus|Malta|Iceland|Finland|Belarus|Ukraine|Mongolia|Cambodia|"
    r"Myanmar|Brunei|Macau|Timor|Afghanistan|Sri lanka|Maldives|Bahrain)\b",
    re.I)


# ------------------------------------------------------------ 2017-01..10
def era_2017(stem, year, period):
    txt = read_text(stem)
    m = re.search(r"HOTEL GUESTS\s+BY\s+NATIONALITY\s*\((?:thousand|000)", txt, re.I)
    if not m:
        return [], ["nationality_table_missing"]
    win = txt[m.end(): m.end() + 2400]
    # strip stray artifacts: month-column headers and chart watermarks
    win = re.sub(r"\b\d{1,2}-[A-Z]{3}\b", "", win)
    win = re.sub(r"\b20\d{2}\b", "", win)
    win = re.sub(r"\b(?:chart|title)\b", "", win, flags=re.I)
    ROW = re.compile(
        r"^\s*(\d{1,2})\s+([A-Z][A-Za-z .'&,-]{2,34}?)\s+(\d{1,4})"
        r"(?:\s+(\d{1,4}))?(?:\s+(\d{1,4}(?:\.\d+)?))?\s*$")
    lines = win.split("\n")
    out, seen = [], set()
    for i, ln in enumerate(lines):
        gm = ROW.match(ln)
        if not gm:
            continue
        rank = int(gm.group(1))
        name_raw = gm.group(2).strip().title()
        name = "TOTAL" if name_raw.lower() == "total" else (
            "Other" if "other" in name_raw.lower() else name_raw)
        if name not in {"TOTAL", "Other"} and not NAT.search(name_raw):
            continue
        nums = [float(x.replace(",", "")) for x in (gm.group(3), gm.group(4))
                if x is not None]
        if gm.group(5):
            alos = float(gm.group(5))
        else:
            follow = re.findall(r"\d+(?:\.\d+)?", lines[i + 1]) if i + 1 < len(lines) else []
            alos = next((float(x) for x in follow if "." in x), None)
        if not nums:
            continue
        cur, prior = nums[0], (nums[1] if len(nums) > 1 else None)
        if name == "TOTAL" and cur < 50:
            continue
        if name not in {"TOTAL", "Other"} and not (1 <= cur <= 1200):
            continue
        if name == "TOTAL":
            key = ("total", "")
        else:
            key = (name, rank)
        if key in seen:
            continue
        seen.add(key)
        canon = name
        if name not in {"TOTAL", "Other"}:
            canon = canon_nationality(name)
        out.append(dict(source_file=stem, year=year, period=period,
                        granularity="month", segment="total", rank=rank,
                        nationality=canon, guests=cur,
                        guests_unit="thousands", share_pct=None,
                        alos=alos, prior_guests=prior, ytd=0,
                        source_block="guests_by_nationality"))
    if not out:
        return [], ["nationality_table_missing"]
    # normalise duplicate spellings of the same market within a file
    dedup, seen_nat = [], set()
    for r in sorted(out, key=lambda r: (r["rank"] if r["rank"] is not None else 99)):
        if r["nationality"] in seen_nat:
            continue
        seen_nat.add(r["nationality"])
        dedup.append(r)
    return dedup, []


# ------------------------------------------- 2017-11..2018, 2020-01..2021-03
BLOCK = re.compile(
    r"(TOP\s+NATIONALITIES(?:\s+ACROSS\s+REGIONS)?|HOTEL\s+GUESTS\s+BY\s+NATIONALITY)",
    re.I)
REGION = re.compile(r"^\s*(ABU DHABI|AL AIN|AL DHAFRA)\s*$", re.I)


def era_blocks(stem, year, period):
    txt = read_text(stem)
    rows, notes = [], []
    heads = list(BLOCK.finditer(txt))
    if not heads:
        return [], ["nationality_table_missing"]
    for hi, hm in enumerate(heads):
        end = heads[hi + 1].start() if hi + 1 < len(heads) else len(txt)
        win = txt[hm.start(): min(end, hm.start() + 3000)]
        head = win[:400]
        ytd = bool(re.search(r"YTD", head, re.I))
        gran = "ytd" if ytd else "month"
        seg = "total"
        cur_nat, pending = None, []
        for ln in win.split("\n")[1:60]:
            rm = REGION.match(ln)
            if rm:
                seg = canon_nationality(rm.group(1)) or rm.group(1).title()
                cur_nat = None
                continue
            nm = NAT.search(ln)
            pct = re.search(r"(\d+(?:\.\d+)?)\s*%", ln)
            if nm and pct:
                rows.append(dict(source_file=stem, year=year, period=period,
                                 granularity=gran, segment=seg, rank=None,
                                 nationality=canon_nationality(nm.group(1)),
                                 guests=None, guests_unit="count",
                                 share_pct=num(pct.group(1)), alos=None,
                                 prior_guests=None, ytd=int(ytd),
                                 source_block="top_nationalities_block"))
                cur_nat = None
                continue
            if nm:
                cur_nat = canon_nationality(nm.group(1))
                continue
            nums = [num(c) for c in re.findall(r"\d+(?:\.\d+)?", ln)]
            nums = [n for n in nums if n is not None
                    and not (1980 <= n <= 2035)]
            if cur_nat and nums and max(nums) >= 5:
                rows.append(dict(source_file=stem, year=year, period=period,
                                 granularity=gran, segment=seg, rank=None,
                                 nationality=cur_nat, guests=nums[0],
                                 guests_unit="thousands", share_pct=None,
                                 alos=None, prior_guests=None, ytd=int(ytd),
                                 source_block="top_nationalities_block"))
                cur_nat = None
    if not rows:
        notes.append("nationality_table_missing")
    return rows, notes


# --------------------------------------------------------- 2019 quarterlies
def era_2019q(stem, year, period):
    txt = read_text(stem)
    rows, notes = [], []
    m = re.search(r"TOP\s+NON-UAE\s+NATIONALITIES\s*\(000\)\s*[–-]\s*"
                  r"(Q\d|YTD)", txt, re.I)
    if not m:
        return [], ["nationality_table_missing"]
    gran = "ytd" if m.group(1).upper() == "YTD" else "quarter"
    head_end = txt.find("NATIONALITIES", m.end())
    win = txt[txt.find("\n", head_end):]
    share_block = re.search(r"(\d{2})\s*%\s*\n?\s*OF\s+TOTAL\s+GUESTS", win, re.I)
    top10_share = num(share_block.group(1)) if share_block else None
    cur_nat, seen = None, {}
    for ln in win.split("\n")[:26]:
        nm = NAT.search(ln)
        if nm:
            cur_nat = canon_nationality(nm.group(1))
            continue
        nums = [num(c) for c in re.findall(r"[\d,]+(?:\.\d+)?", ln)]
        nums = [n for n in nums if n is not None]
        if not nums or cur_nat is None:
            continue
        if re.search(r"OF\s+TOTAL|NATIONALITIES|TOP 10", ln, re.I):
            continue
        if cur_nat in seen:
            continue
        seen[cur_nat] = nums[0]
        rows.append(dict(source_file=stem, year=year, period=period,
                         granularity=gran, segment="total", rank=None,
                         nationality=cur_nat, guests=nums[0],
                         guests_unit="thousands", share_pct=None, alos=None,
                         prior_guests=None, ytd=int(gran == "ytd"),
                         source_block="top_non_uae_2019q",
                         top10_non_uae_share_pct=top10_share))
        cur_nat = None
    rows.sort(key=lambda r: -(r["guests"] or 0))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows, notes


# --------------------------------------------- 2021-04 .. 2023-10 ranked
RANK = re.compile(r"#\s*(\d+)")
VALUES = re.compile(r"([\d,]+)\s{2,}([+-]?[\d,.]+%)\s+(\d+(?:\.\d+)?%)?")


def era_ranked(stem, year, period, total_guests=None):
    txt = read_text(stem)
    rows, notes = [], []
    for hm in re.finditer(r"HOTEL GUEST NATIONALITIES\s*[\u2013-]\s*([A-Z]{3}\s+\d{4}"
                          r"|YTD\s+[A-Z]{3}\s+\d{4})", txt):
        scope = re.sub(r"\s+", " ", hm.group(1)).upper()
        gran = "ytd" if scope.startswith("YTD") else "month"
        pl = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{4})",
                       scope)
        per = f"{pl.group(1)} {pl.group(2)}" if pl else period
        win = txt[hm.end(): hm.end() + 2600]
        cut = win.find("PERFORMANCE BY")
        if cut > 0:
            win = win[:cut]
        lines = win.split("\n")[:70]
        parsed, rank, name = [], None, None
        for i, ln in enumerate(lines):
            rm = RANK.search(ln)
            nm = NAT.search(ln)
            if rm:
                rank = int(rm.group(1))
            if nm:
                name = canon_nationality(nm.group(1))
                name = "UAE" if name == "UNITED ARAB" else name
            v = VALUES.search(ln)
            if v and rank and name and num(v.group(1)) is not None:
                parsed.append(dict(rank=rank, nationality=name,
                                   guests=num(v.group(1)),
                                   guests_unit="thousands",
                                   change_pct=v.group(2),
                                   share_pct=num(v.group(3)) if v.group(3) else None))
                rank, name = None, None
                continue
            if i + 1 < len(lines) and nm and rank and not v:
                v2 = VALUES.search(lines[i + 1])
                if v2 and num(v2.group(1)) is not None:
                    parsed.append(dict(rank=rank, nationality=name,
                                       guests=num(v2.group(1)),
                                       guests_unit="thousands",
                                       change_pct=v2.group(2),
                                       share_pct=num(v2.group(3)) if v2.group(3) else None))
                    rank, name = None, None
        if not parsed:
            continue
        shares = [r["share_pct"] for r in parsed]
        if None in shares:
            cand = sorted([v for v in shares if v is not None], reverse=True)
            have = [v for v in shares if v is not None]
            if len(cand) == len(shares) and cand == have:
                shares = cand
        for r, sh in zip(parsed, shares):
            r["share_pct"] = sh
            if total_guests and r["guests"]:
                r["share_check_pct"] = round(r["guests"] * 1000.0 / total_guests, 1)
        seen_rank, seen_nat = set(), set()
        dedup = []
        for r in sorted(parsed, key=lambda r: (r["rank"], -(r["guests"] or 0))):
            if r["rank"] in seen_rank or r["nationality"] in seen_nat:
                continue
            seen_rank.add(r["rank"])
            seen_nat.add(r["nationality"])
            dedup.append(r)
        parsed = dedup
        for r in sorted(parsed, key=lambda r: r["rank"]):
            rows.append(dict(source_file=stem, year=year, period=per,
                             granularity=gran, segment="total", rank=r["rank"],
                             nationality=r["nationality"], guests=r["guests"],
                             guests_unit=r["guests_unit"], share_pct=r["share_pct"],
                             alos=None, prior_guests=None, ytd=int(gran == "ytd"),
                             change_pct=r["change_pct"],
                             source_block="ranked_table"))
    if not rows:
        notes.append("nationality_table_missing")
    return rows, notes
# ------------------------------------- 2015-2016 ranked tables
R2016 = re.compile(
    r"^(\d{1,2})\s+([A-Za-z][A-Za-z .'-]{2,24}?)\s+([\d,]+)\s+([\d,]+)\s+"
    r"([+-]?\d+%\s*[pq])\s+([\d,]+)\s+([\d,]+)\s+([+-]?\d+%\s*[pq])\s+"
    r"([\d.]+)\s+([\d.]+)(?:\s+([+-]?[\d.]+%?\s*[pq]))?")
R2015 = re.compile(
    r"^(\d{1,2})\s+([A-Za-z][A-Za-z .'-]{2,24}?)\s+(\d+)%\s+"
    r"([+-]?\d+%\s*[pq])\s+([\d,]+)\s+([\d.]+)\s*$")
HEADNAT = re.compile(r"^\s*(?:Rank\s+)?Nationality\b", re.I)


def era_2015(stem, year, period):
    txt = read_text(stem)
    heads = [i for i, ln in enumerate(txt.split("\n")) if HEADNAT.match(ln)]
    if not heads:
        m = re.search(r"Guests'?\s+by\s+Nationality", txt, re.I)
        if not m:
            return [], ["nationality_table_missing"]
        head0 = txt[:m.start()].count("\n")
        heads = [head0]
    lines = txt.split("\n")
    rows, notes = [], []
    for h in heads:
        ytd = bool(re.search(r"YTD|Year\s+to\s+Date", "\n".join(lines[max(0, h - 4):h]), re.I))
        for ln in lines[h + 1: h + 24]:
            if HEADNAT.match(ln):
                break
            g = R2016.match(ln.strip())
            if g:
                rows.append(dict(source_file=stem, year=year, period=period,
                                 granularity="ytd" if ytd else "month",
                                 segment="total", rank=int(g.group(1)),
                                 nationality=canon_nationality(g.group(2).strip()),
                                 guests=num(g.group(3)), guests_unit="count",
                                 prior_guests=num(g.group(4)),
                                 share_pct=None, alos=num(g.group(9)),
                                 ytd=int(ytd),
                                 change_pct=num(g.group(5).rstrip(" pq").rstrip("%")),
                                 ytd_guests=num(g.group(6)),
                                 ytd_prior_guests=num(g.group(7)),
                                 ytd_change_pct=num(g.group(8).rstrip(" pq").rstrip("%")),
                                 source_block="top10_2016"))
                continue
            g = R2015.match(ln.strip())
            if g:
                rows.append(dict(source_file=stem, year=year, period=period,
                                 granularity="month", segment="total",
                                 rank=int(g.group(1)),
                                 nationality=canon_nationality(g.group(2).strip()),
                                 guests=num(g.group(5)), guests_unit="count",
                                 prior_guests=None,
                                 share_pct=num(g.group(3)), alos=num(g.group(6)),
                                 ytd=0,
                                 change_pct=num(g.group(4).rstrip(" pq").rstrip("%")),
                                 source_block="top10_2015"))
    if not rows:
        notes.append("nationality_table_missing")
    return rows, notes


# ------------------------- 2017-11 .. 2021-03 "TOP NON-UAE NATIONALITIES (000)"
CHARTNAT_HEAD = re.compile(r"TOP\s*(?:\d+\s*)?NON-?UAE\s+NATIONALITIES", re.I)
NUMTOK = re.compile(r"^\s*(\d{1,4}(?:\.\d+)?)\s*$")
_nat_src = NAT.pattern[3:] if NAT.pattern.startswith(r"\b(") else NAT.pattern[1:]
_nat_src = re.sub(r"\\b$", "", _nat_src)
_NAT_ALT = re.sub(r"[)\?]+$", "", _nat_src)
NAMEPCT = re.compile(r"\b(" + _NAT_ALT + r")\s+([+-]?\d+(?:\.\d+)?)%")


def _chart_block(lines):
    """Return rows for one left/right half of the interleaved chart table."""
    pairs = []
    for ln in lines:
        m = NAMEPCT.search(ln)
        if m:
            pairs.append([canon_nationality(m.group(1)),
                          num(m.group(2).rstrip("%"))])
    toks = [num(m.group(1)) for ln in lines if (m := NUMTOK.match(ln.strip()))]
    toks = [t for t in toks if t is not None]
    if not pairs or not toks:
        return []
    out, p = [], 0
    for name, chg in pairs:
        pick = None
        for j in range(p, min(p + 6, len(toks) - 1)):
            cur = toks[j]
            if cur is None or cur < 1 or cur > 300:
                continue
            prior = None
            for k in range(j + 1, min(j + 5, len(toks))):
                cand = toks[k]
                if cand and cand > 0 and abs((cand / cur - 1) * 100 - chg) <= 3.5:
                    prior = cand
                    pick = (j, k, cur, prior)
                    break
            if pick:
                break
        if not pick:
            continue
        j, k, cur, prior = pick
        out.append(dict(nationality=name, guests=cur, prior_guests=prior,
                        change_pct=chg))
        p = k + 1
    return out


def era_chart(stem, year, period, asof="current"):
    txt = read_text(stem)
    hm = CHARTNAT_HEAD.search(txt)
    if hm is None:
        return [], ["nationality_table_missing"]
    win = txt[hm.start(): hm.start() + 2400]
    cut = win.find("ALOS BY NATIONALITY")
    if cut > 0:
        win = win[:cut]
    left, right = [], []
    for ln in win.split("\n"):
        left.append(ln[:110])
        right.append(ln[110:])
    rows = []
    for blk, gran in ((left, "month"), (right, "ytd")):
        got = _chart_block(blk)
        if len(got) < 3:
            continue
        for r in got:
            rows.append(dict(source_file=stem, year=year, period=period,
                             granularity=gran, segment="total", rank=None,
                             nationality=r["nationality"],
                             guests=r["guests"], guests_unit="thousands",
                             share_pct=None, alos=None,
                             prior_guests=r["prior_guests"],
                             ytd=int(gran == "ytd"),
                             change_pct=r["change_pct"],
                             source_block="top_non_uae_%s" % gran))
    if not rows:
        return [], ["nationality_table_missing"]
    seen, dedup = set(), []
    for r in sorted(rows, key=lambda r: -(r["guests"] or 0)):
        key = (r["granularity"], r["nationality"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    tot = 0.0
    return dedup, []
