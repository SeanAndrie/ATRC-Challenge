"""Shared helpers for DCT Abu Dhabi hotel report parsing."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Paths come from the one anchor. These three used to be siblings under a
# single notebooks/data/dct directory; the data tiers now separate them, and
# the names below keep the split visible at every use site.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DCT_PDFS, DCT_TEXT  # noqa: E402

# Extracted text, read by every parser. Interim: derived from the PDFs, not yet
# validated.
TEXTDIR = DCT_TEXT
# The downloaded PDFs themselves. Raw: never written by a script.
PDFDIR = DCT_PDFS
# Retained as the historical name for the DCT working set. It no longer names a
# directory; it is a namespace for the constants below, and a reminder that
# these paths used to be one flat folder.
DCT = TEXTDIR

MONTH_NUM = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

NAT_CANON = {
    "u": "UAE", "uae": "UAE", "united arab emirates": "UAE",
    "u a e": "UAE", "uae.": "UAE", "u.a.e": "UAE", "u.a.e.": "UAE",
    "us": "USA", "u.s": "USA", "u.s.": "USA", "u.s.a": "USA", "u.s.a.": "USA",
    "usa": "USA", "united states": "USA", "united states of america": "USA",
    "uk": "UK", "u.k": "UK", "u.k.": "UK", "united kingdom": "UK",
    "ksa": "KSA", "k.s.a": "KSA", "saudi arabia": "KSA", "saudi": "KSA",
    "saudi arabia ": "KSA",
    "netherlands": "Netherlands", "netherlands (holland)": "Netherlands",
    "holland": "Netherlands", "(holland)": "Netherlands",
    "south korea": "South Korea", "korea": "South Korea",
    "russian federation": "Russia", "russia ": "Russia",
    "iran": "Iran", "syrian arab republic": "Syria",
    "cote d'ivoire": "Cote d'Ivoire",
    "cote d ivoire": "Cote d'Ivoire",
    "bosnia": "Bosnia", "czech republic": "Czech Republic",
    "republic of korea": "South Korea",
    "swaziland": "Eswatini", "macedonia": "North Macedonia",
    "burma": "Myanmar", "burmese": "Myanmar",
    "united kingdom of great britain and northern ireland": "UK",
    "state of palestine": "Palestine", "palestinian territory": "Palestine",
    "taiwan": "Taiwan", "chinese taipei": "Taiwan",
    "macau": "Macau", "macao": "Macau",
    "brunei darussalam": "Brunei", "brunei": "Brunei",
    "republic of moldova": "Moldova",
    "united republic of tanzania": "Tanzania",
    "sri lanka ": "Sri Lanka",
    "yemen": "Yemen",
    "syrian arab republic ": "Syria",
}

REGION_CANON = {
    "abu dhabi": "Abu Dhabi",
    "abu dhabi city": "Abu Dhabi",
    "abudhabi": "Abu Dhabi",
    "abudhabi city": "Abu Dhabi",
    "abu dhabi emirate": "Abu Dhabi",
    "al ain": "Al Ain",
    "al ain city": "Al Ain",
    "alain": "Al Ain",
    "al dhafra": "Al Dhafra",
    "al dhafra region": "Al Dhafra",
    "al dhafrah": "Al Dhafra",
    "aldhafra": "Al Dhafra",
    "western region": "Al Dhafra",
    "western": "Al Dhafra",
    "al ain region": "Al Ain",
    "abu dhabi city ": "Abu Dhabi",
}

STAR_CANON = {
    "5 star": "5 Star", "5-star": "5 Star", "5 stars": "5 Star",
    "4 star": "4 Star", "4-star": "4 Star",
    "3 star": "3 Star", "3-star": "3 Star",
    "2 star": "2 Star", "1 star": "1 Star",
    "1-3 star": "1-3 Star", "3-1 star": "1-3 Star", "1-3 stars": "1-3 Star",
    "3-1 stars": "1-3 Star", "3 & 1 star": "1-3 Star", "1 & 3 star": "1-3 Star",
    "3-1 star hotels": "1-3 Star", "1-3 star hotels": "1-3 Star",
    "5-star hotels": "5 Star", "4-star hotels": "4 Star",
    "deluxe": "Deluxe Apartments", "superior": "Superior Apartments",
    "standard": "Standard Apartments",
    "hotel apartments": "Hotel Apartments",
    "apts": "Hotel Apartments", "apts ": "Hotel Apartments",
    "apartments": "Hotel Apartments",
}


def canon_nationality(name: str) -> str:
    if name is None:
        return name
    n = re.sub(r"\s+", " ", str(name)).strip()
    key = n.lower().strip(" .,-")
    if key in NAT_CANON:
        return NAT_CANON[key]
    if n.lower() == "united arab":
        return "UAE"
    if n.upper() in {"UAE", "UK", "USA", "KSA"}:
        return n.upper()
    return n


def canon_region(name: str):
    n = re.sub(r"\s+", " ", str(name)).strip()
    return REGION_CANON.get(n.lower().strip(), None)


def canon_star(name: str):
    n = re.sub(r"\s+", " ", str(name)).strip()
    return STAR_CANON.get(n.lower().strip(), None)


def load_manifest() -> dict:
    rows = json.loads((DCT / "manifest.json").read_text())
    return {r["href"] for r in rows}, rows


def num(s):
    """Parse a printed number into a float, returning None when not a number."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s in {"-", "--", "###", "#REF!", "########", "N/A", "n/a"}:
        return None
    s = s.replace(",", "").replace("\u2013", "-").replace("−", "-")
    s = s.replace("%", "").replace("AED", "").replace("M", "").strip()
    m = re.match(r"^[-+]?\d+(\.\d+)?$", s)
    if m:
        v = float(s)
        return v
    return None


def clean_lines(txt: str):
    """Yield (lineno, stripped_line) skipping empties and form feeds."""
    for i, ln in enumerate(txt.split("\n")):
        s = ln.replace("\f", "").rstrip()
        if s.strip():
            yield i, s


def page_marks(txt: str):
    """Return the raw line index where each PDF page starts (form feed)."""
    idx = []
    pos = 0
    for i, ln in enumerate(txt.split("\n")):
        if "\f" in ln:
            idx.append(i)
        pos += len(ln) + 1
    return idx


def split_pages(txt: str):
    """Split cached pdftotext output into per-page line lists."""
    pages, cur = [], []
    for ln in txt.split("\n"):
        if "\f" in ln:
            parts = ln.split("\f")
            if parts[0].strip():
                cur.append(parts[0].rstrip())
            pages.append(cur)
            cur = []
            if len(parts) > 1 and parts[1].strip():
                cur.append(parts[1].rstrip())
        else:
            cur.append(ln.rstrip())
    if any(l.strip() for l in cur):
        pages.append(cur)
    return pages


def read_text(stem: str) -> str:
    return (TEXTDIR / f"{stem}.txt").read_text(encoding="utf-8", errors="replace")
