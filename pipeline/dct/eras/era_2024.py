"""2024 annual report (narrative + monthly KPI trend tables)."""
import re
from dct_common import read_text, num

TREND = re.compile(
    r"^\s*(TOTAL GUESTS|ALOS \(NIGHTS\)|OCCUPANCY|ADR|REVPAR|"
    r"TOTAL REVENUES)\s*\(?", re.I)
PCT = re.compile(r"^\s*([\d.]+)%")

METRIC = {
    "total guests": ("hotel_guests", "thousands"),
    "alos (nights)": ("alos", "days"),
    "occupancy": ("occupancy_rate", "pct"),
    "adr": ("arr", "aed"),
    "revpar": ("revpar", "aed"),
    "total revenues": ("total_revenue", "aed_million"),
}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
          "NOV", "DEC"]


def _kilo(tok):
    m = re.match(r"^([\d,.]+)([KkMm])?$", tok.strip())
    if not m:
        return None
    v = num(m.group(1))
    if v is None:
        return None
    sfx = (m.group(2) or "").lower()
    return v * 1000.0 if sfx == "k" else (v * 1e6 if sfx == "m" else v)


def _cells(ln):
    out = []
    for m in re.finditer(r"[\d,]+(?:\.\d+)?%?", ln):
        out.append((m.start(), m.end(), m.group(0)))
    return out


def _values(ln):
    vals = []
    for _, _, tok in _cells(ln):
        v = num(tok.rstrip("%"))
        if v is not None:
            vals.append(v)
    return vals


def parse(stem="2024-annual-2024", year=2024, period="annual"):
    txt = read_text(stem)
    rows, notes, checks = [], [], []
    lines = txt.split("\n")
    for i, ln in enumerate(lines):
        m = TREND.match(ln.strip())
        if not m:
            continue
        metric, unit = METRIC[m.group(1).lower()]
        block = [b for b in lines[i + 1:i + 10] if _values(b)]
        big = [b for b in block if len(_values(b)) >= 12]
        pair = next((b for b in block if len(_values(b)) == 2), None)
        growth = None
        for b in block:
            g = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)%\s*$", b.strip())
            if g:
                growth = num(g.group(1))
        if not big:
            notes.append((metric, "trend_rows_missing"))
            continue
        # a 14-value line carries the prior-year months plus both annual totals
        wide = next((b for b in big if len(_values(b)) >= 14), None)
        months12 = [b for b in big if len(_values(b)) == 12]
        cur12 = _values(months12[-1])[:12] if months12 else []
        ytd_prev = ytd_cur = None
        prior12 = []
        if wide:
            v = _values(wide)
            prior12 = v[:12]
            ytd_prev, ytd_cur = (v[12] if len(v) > 12 else None,
                                 v[13] if len(v) > 13 else None)
        elif pair:
            v = _values(pair)
            ytd_prev, ytd_cur = v[0], v[1]
        if not cur12 and wide:
            cur12 = _values(wide)[:12]
        if ytd_cur and sum(cur12) and metric in ("hotel_guests", "total_revenue"):
            checks.append((metric, "ytd_sum_check", round(sum(cur12), 1),
                           ytd_cur, abs(sum(cur12) - ytd_cur) <= 0.02 * ytd_cur))
        for idx, v in enumerate(cur12):
            rows.append(dict(source_file=stem, year=year, period=f"{idx + 1:02d}",
                             granularity="month", segment="total", metric=metric,
                             unit=unit, value=v, growth_pct=None, ytd=0,
                             derived=0, source_block="kpi_monthly_trend"))
        # The wide line's leading twelve numbers are a prior-year series, but
        # they are not the same series as that year's own monthly reports: for
        # 2023 they sum to the printed 5,504K total while the 2023 monthly
        # reports give 4,065K for January to October, a restatement of roughly
        # ten per cent. Merging them would produce a year containing two
        # irreconcilable values per month, so the months are recorded as a
        # note and only the prior-year total, which is explicitly labelled, is
        # kept.
        if ytd_prev and prior12:
            notes.append((metric, "prior_year_months_not_merged",
                          f"prior-year months sum {sum(prior12):,.0f} vs "
                          f"prior-year total {ytd_prev:,.0f}; left unmerged"))
        if ytd_cur is not None:
            rows.append(dict(source_file=stem, year=year, period="annual",
                             granularity="year", segment="total", metric=metric,
                             unit=unit, value=ytd_cur, growth_pct=growth, ytd=1,
                             derived=0, source_block="kpi_monthly_trend"))
        if ytd_prev is not None:
            rows.append(dict(source_file=stem, year=year - 1, period="annual",
                             granularity="year", segment="total", metric=metric,
                             unit=unit, value=ytd_prev, growth_pct=None, ytd=1,
                             derived=0, source_block="kpi_monthly_trend_prior"))
    # headline prose
    prose = re.search(r"welcoming\s+([\d.]+)\s*million\s+hotel\s+guests\s*"
                      r"\(including\s+([\d.]+)\s*million\s+international",
                      txt, re.I)
    if prose:
        rows.append(dict(source_file=stem, year=year, period="annual",
                         granularity="year", segment="total",
                         metric="hotel_guests", unit="count",
                         value=num(prose.group(1)) * 1e6, ytd=1, derived=1,
                         growth_pct=None, source_block="narrative"))
        rows.append(dict(source_file=stem, year=year, period="annual",
                         granularity="year", segment="total",
                         metric="international_guests", unit="count",
                         value=num(prose.group(2)) * 1e6, ytd=1, derived=1,
                         growth_pct=None, source_block="narrative"))
    rooms = re.search(r"inventory\s+reached\s+(\d+)\s+properties,?\s+offering\s+"
                      r"approximately\s+([\d,]+)\s+rooms", txt, re.I)
    supply = []
    if rooms:
        supply.append(dict(source_file=stem, year=year, period="annual",
                           as_of="2024-12", segment="total", category="total",
                           hotels=num(rooms.group(1)),
                           rooms=num(rooms.group(2)),
                           source_block="narrative"))
    if rooms:                      # 171 properties / ~34,000 rooms is printed
        pass
    else:
        mt = re.search(r"(\d+)\s+HOTELS\s+(\d+(?:\.\d+)?K)\s+ROOMS", txt, re.I)
        if mt:
            supply.append(dict(source_file=stem, year=year, period="annual",
                               as_of="2024-12", segment="total",
                               category="total", hotels=num(mt.group(1)),
                               rooms=num(mt.group(2)) * 1000,
                               source_block="supply_page"))
    nat = []
    marks = re.search(r"KEY\s+INTERNATIONAL\s+MARKETS", txt, re.I)
    if marks:
        win = txt[marks.end(): marks.end() + 700]
        name = None
        for ln in win.split("\n"):
            nm = re.match(r"^\s*([A-Z][A-Z ]{2,14}?)\s*(\+[\d.]+%.*)?$", ln)
            if nm:
                cand = re.sub(r"\s+", " ", nm.group(1)).strip().upper()
                if cand == "ARABIA" and name and "SAUDI" in name:
                    cand = "SAUDI ARABIA"
                if cand and cand != "VISITOR":
                    name = cand
            g = re.search(r"\+(\d+)%\s+([\d.]+K)\b", ln)
            if g and name:
                nat.append(dict(source_file=stem, year=year, period="annual",
                                granularity="year", segment="total", rank=None,
                                nationality=name, guests=_kilo(g.group(2)),
                                guests_unit="count", share_pct=None, alos=None,
                                ytd=1, change_pct=num(g.group(1)),
                                source_block="key_international_markets"))
                name = None
    nat.sort(key=lambda r: -(r["guests"] or 0))
    for i, r in enumerate(nat, 1):
        r["rank"] = i
    return rows, nat, supply, notes, checks
