"""Positional extraction of the 12-month bar-chart series (Era B/C/D + 2017 chart reports).

Charts in these reports render two bar series (previous year / current year) per metric.
The previous-year labels sit on top of their (varied-height) bars; the current-year labels
are drawn on a common baseline row.  Every extracted series is cross-validated against
either (a) the printed full-year / YTD total printed in the right-hand column, or
(b) the printed current-month % change.  Anything that fails validation is discarded.
"""

from __future__ import annotations

import re

import pdfplumber

from dct_common import num

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_IDX = {m.upper(): i + 1 for i, m in enumerate(MONTHS)}
NUM_RE = re.compile(r"^-?[\d,]*\.?\d+%?$")
ADDITIVE = {"guests", "revenue"}  # totals are sums


def _rows(words, tol=2.6):
    rows = []
    for w in sorted(words, key=lambda w: w["top"]):
        placed = False
        for r in rows:
            if abs(r["top"] - w["top"]) <= tol:
                r["words"].append(w)
                r["top"] = (r["top"] * (len(r["words"]) - 1) + w["top"]) / len(r["words"])
                placed = True
                break
        if not placed:
            rows.append({"top": w["top"], "words": [w]})
    for r in rows:
        r["words"].sort(key=lambda w: w["x0"])
        r["y0"] = min(w["top"] for w in r["words"])
        r["y1"] = max(w["bottom"] for w in r["words"])
        r["x0"] = min(w["x0"] for w in r["words"])
        r["x1"] = max(w["x1"] for w in r["words"])
    rows.sort(key=lambda r: r["top"])
    return rows


def _label_row(row):
    """Return dict month->x for a row that is a Jan..Dec axis label row."""
    toks = [w["text"] for w in row["words"]]
    hits = []
    for i, t in enumerate(toks):
        tt = t.strip().upper().rstrip(".")
        if tt in MONTH_IDX and tt not in [h[0] for h in hits]:
            hits.append((tt, row["words"][i]["x0"], row["words"][i]["x1"]))
    if len(hits) < 8:
        return None
    hits.sort(key=lambda h: MONTH_IDX[h[0]])
    seq = [h[0] for h in hits]
    if seq != sorted(seq, key=lambda s: MONTH_IDX[s]):
        return None
    return {MONTH_IDX[h[0]]: (h[1] + h[2]) / 2 for h in hits}


def _col_for(x, cols):
    best, bd = None, 1e9
    for m, cx in cols.items():
        d = abs(x - cx)
        if d < bd:
            best, bd = m, d
    return best, bd


def _match_metric(txt_upper, metric_map):
    tight = txt_upper.replace(" ", "")
    for k, pats in metric_map.items():
        for p in pats:
            pu = p.upper()
            if pu in tight or pu in txt_upper:
                return k
    return None


def extract_page_charts(page, metric_map):
    """Return a list of chart dicts found on one pdfplumber page."""
    words = page.extract_words()
    rows = _rows(words)
    charts = []
    for ri, row in enumerate(rows):
        cols = _label_row(row)
        if not cols:
            continue
        prev_label_rows = [k for k in range(ri) if _label_row(rows[k])]
        lower = max(prev_label_rows) if prev_label_rows else -1
        # year labels sitting to the right of the month axis
        year_cols = {}
        for w in row["words"]:
            if re.fullmatch(r"(19|20)\d\d", w["text"]):
                year_cols[(w["x0"] + w["x1"]) / 2] = int(w["text"])
        span = (min(cols.values()), max(cols.values()))
        # walk up to the metric title row (bounded by the previous chart)
        title_row = None
        for j in range(ri - 1, lower, -1):
            r = rows[j]
            if r["y1"] >= row["y0"]:
                continue
            txt = " ".join(w["text"] for w in r["words"]).upper()
            if _match_metric(txt, metric_map):
                title_row = j
                break
        if title_row is None:
            continue
        ttxt = " ".join(w["text"] for w in rows[title_row]["words"]).upper()
        key = _match_metric(ttxt, metric_map)
        if key is None:
            continue
        vals = []
        for j in range(title_row + 1, ri):
            for w in rows[j]["words"]:
                if not NUM_RE.match(w["text"]) or w["text"].endswith("%"):
                    continue
                cx = (w["x0"] + w["x1"]) / 2
                m, d = _col_for(cx, cols)
                if d <= 26:
                    v = num(w["text"])
                    if v is not None:
                        vals.append((cx, w["top"], v, m))
        # right-hand totals columns (full year / YTD) + their year labels
        right = []
        for j in range(title_row + 1, ri + 1):
            for w in rows[j]["words"]:
                if not NUM_RE.match(w["text"]) or w["text"].endswith("%"):
                    continue
                cx = (w["x0"] + w["x1"]) / 2
                if cx <= span[1] + 12:
                    continue
                v = num(w["text"])
                if v is None or 1980 <= v <= 2100 and "." not in w["text"]:
                    continue
                right.append((cx, v))
        yr_tokens = []
        for j in range(title_row + 1, ri + 1):
            for w in rows[j]["words"]:
                if re.fullmatch(r"(19|20)\d\d", w["text"]):
                    yr_tokens.append(((w["x0"] + w["x1"]) / 2, int(w["text"])))
        groups = {}
        for cx, v in right:
            hit = next((g for g in groups if abs(g - cx) < 30), None)
            if hit is None:
                groups[cx] = v
        cols_sorted = sorted(groups.items(), key=lambda kv: -kv[0])
        totals = []
        for cx, v in cols_sorted[:2]:
            yr = None
            for tx, tv in yr_tokens:
                if abs(tx - cx) < 45:
                    yr = tv
            totals.append((yr, v))
        legend = []
        for j in range(max(0, title_row - 3), title_row):
            for w in rows[j]["words"]:
                if re.fullmatch(r"(19|20)\d\d", w["text"]):
                    legend.append(int(w["text"]))
        charts.append(dict(metric=key, cols=cols, values=vals, totals=totals,
                           legend=sorted(set(legend)),
                           y_label=row["top"], y_title=rows[title_row]["top"]))
    return charts


def assign_series(chart, additive, tol_rel=0.012, tol_abs=2.0):
    """Split chart values into prev/current series and validate against printed totals."""
    cols = chart["cols"]
    per_col = {m: [] for m in cols}
    for x0, cx, top, v in chart["values"]:
        m, d = _col_for(cx, cols)
        if d <= 26:
            per_col[m].append((top, v))
    for m in per_col:
        per_col[m].sort()
    series = {}
    for orient in ("prev_upper", "prev_lower"):
        prev, cur = {}, {}
        ok = True
        for m, vs in per_col.items():
            if len(vs) == 2:
                a, b = vs
                if orient == "prev_upper":
                    prev[m], cur[m] = a[1], b[1]
                else:
                    prev[m], cur[m] = b[1], a[1]
            elif len(vs) == 1:
                prev[m] = vs[0][1]
            else:
                ok = False
        if not ok:
            continue
        if _validate(prev, cur, chart, additive, tol_rel, tol_abs):
            series = dict(prev=prev, cur=cur, orient=orient)
            break
    return series


def _validate(prev, cur, chart, additive, tol_rel, tol_abs):
    tot = dict(chart["totals"])
    leg = chart["legend"]
    if len(tot) < 1:
        return False
    def close(a, b):
        return abs(a - b) <= max(tol_abs, tol_rel * abs(b))
    if additive:
        pv = sum(v for v in prev.values() if v is not None)
        cv = sum(v for v in cur.values() if v is not None)
        cands = list(tot.values())
        if len(cands) >= 2:
            return close(pv, cands[0]) and close(cv, cands[1])
        return close(pv, cands[0]) or close(cv, cands[0])
    pv = [v for v in prev.values() if v is not None]
    cv = [v for v in cur.values() if v is not None]
    if not pv or not cv:
        return False
    cands = list(tot.values())
    mp, mc = sum(pv) / len(pv), sum(cv) / len(cv)
    if len(cands) >= 2:
        return (close(mp, cands[0]) and close(mc, cands[1])) or (
            close(mp, cands[1]) and close(mc, cands[0]))
    return close(mp, cands[0]) or close(mc, cands[0])
