"""Fingerprint every extracted DCT text file: internal period title + structural markers."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import DCT_FINGERPRINT, DCT_TEXT  # noqa: E402

TEXT = DCT_TEXT

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

MARKERS = {
    "main_indicators": r"Main Indicators Summary",
    "top10_nat": r"Top 10 Nationalities",
    "top25_nat": r"Top 25 Nationalities",
    "guest_nat_table": r"HOTEL GUEST NATIONALITIES|Guests' by Nationality|GUESTS BY NATIONALITY",
    "regional_summary": r"Regional Summary|PERFORMANCE BY REGION|Performance by Region",
    "star_rating": r"STAR RATING|Star Rating|star rating",
    "rooms_summary": r"Rooms Summary|HOTEL SUPPLY|Hotel Supply|HOTEL INVENTORY",
    "overall_perf": r"OVERALL PERFORMANCE|Overall Performance",
    "key_highlights": r"KEY HIGHLIGHTS|Key Highlights",
    "other_key_indicators": r"Other Key Indicators|OTHER KEY INDICATORS",
    "top_nat_thousands": r"TOP NATIONALITIES|Top Nationalities",
    "top_nation_nonuae": r"NON-UAE|Non-UAE|Non UAE",
    "dear_partner": r"Dear Hotel Partner",
    "section_keyword": r"HOTEL REPORT|HOTEL PERFORMANCE|ABU DHABI HOTEL",
    "contacts": r"Market Intelligence Team|Contacts",
    "definitions": r"Terms and Def|Terms and Def|Definitions",
    "zones": r"ABU DHABI ZONES|Abu Dhabi Zones",
    "alos_nat": r"ALOS ACROSS|ALOS ACROSS TOP",
    "vs_year": r"Vs 20\d\d|% Vs|vs 20\d\d",
}


def title_guess(txt):
    """Find explicit 'MONTH YEAR' / 'Q# YEAR' / 'MONTH & YTD YEAR' titles."""
    pats = [
        rf"({MONTHS})\s*(?:&\s*(?:YTD\s*)?(?:{MONTHS})\s*)?,?\s*&?\s*YTD\s+(20\d\d)",
        rf"({MONTHS})\s+(?:&\s*YTD\s*)?(20\d\d)",
        rf"(Q[1-4])\s*(?:&|&?\s*YTD)?\s*(20\d\d)",
        rf"(20\d\d)\s+({MONTHS})",
        rf"({MONTHS})\s*(?:&\s*({MONTHS})\s*)?((?:19|20)\d\d)",
    ]
    hits = []
    for p in pats:
        for m in re.finditer(p, txt):
            hits.append((m.start(), " ".join(g for g in m.groups() if g)))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def norm_period(s):
    if not s:
        return None
    m = re.match(r"^Q([1-4])$", s.strip(), re.I)
    if m:
        return f"Q{m.group(1)}"
    m = re.search(r"(\d{4})", s)
    yr = int(m.group(1)) if m else None
    mo = None
    for tok in re.findall(r"[A-Za-z]+", s):
        if tok.lower() in MONTH_MAP:
            mo = MONTH_MAP[tok.lower()]
            break
    if yr and mo:
        return f"{yr:04d}-{mo:02d}"
    if yr:
        return f"{yr:04d}-??"
    return None


def main():
    rows = []
    for f in sorted(TEXT.glob("*.txt")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        head = txt[:1500]
        first = next((l.strip() for l in txt.split("\n")[:6] if l.strip()), "")
        markers = {k: len(re.findall(v, txt, re.I)) for k, v in MARKERS.items()}
        # title in header region
        t_head = title_guess(head) or title_guess(txt[:6000])
        rows.append(
            {
                "file": f.stem,
                "chars": len(txt),
                "first_line": first[:80],
                "title_head": t_head,
                "period_head": norm_period(t_head),
                **{k: v for k, v in markers.items()},
            }
        )
    DCT_FINGERPRINT.write_text(json.dumps(rows, indent=1))
    hdr = ["file", "chars", "period_head", "title_head", "first_line"]
    keys = [k for k in MARKERS]
    print(" | ".join(hdr))
    for r in rows:
        mk = " ".join(k[0].upper() for k in keys if r[k])
        print(f"{r['file']:<18} {r['chars']:>6} {str(r['period_head']):<10} {str(r['title_head'])[:26]:<26} {mk}")
    print(f"\nfiles={len(rows)}")


if __name__ == "__main__":
    main()
