import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://dct.gov.ae"
PAGE = BASE + "/en/who.we.are/reports.statistics.aspx"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import DCT_MANIFEST  # noqa: E402

OUT = DCT_MANIFEST

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


def parse_label(label, section_year):
    label = re.sub(r"\s+", " ", label).strip().replace("ownload", "").strip()
    year = section_year
    quarter = None
    month = None
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", label, re.I):
            month = num
            break
    m = re.search(r"\b(20\d{2})\b", label)
    if m:
        year = int(m.group(1))
    q = re.search(r"\bQ(?:uarter)?\s?([1-4])\b", label, re.I)
    if q:
        quarter = int(q.group(1))
        month = None
    return year, month, quarter


def main():
    html = requests.get(PAGE, timeout=60).text
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    section_year = None
    seen = set()
    for tr in soup.find_all("tr"):
        th = tr.find("th")
        if th:
            m = re.match(r"^\s*(20\d{2})\s*$", th.get_text())
            if m:
                section_year = int(m.group(1))
            continue
        links = [a for a in tr.find_all("a", href=True)
                 if a["href"].lower().endswith(".pdf")
                 and "price index" not in a["href"].lower().replace(" proce ", " price ")]
        if not links:
            continue
        label = re.sub(r"\s+", " ", " ".join(tr.stripped_strings)).strip()
        label = re.sub(r"\bdownload\b", "", label, flags=re.I).strip()
        for a in links:
            href = a["href"].strip()
            if href in seen:
                continue
            seen.add(href)
            year, month, quarter = parse_label(label, section_year)
            rtype = "annual" if (month is None and quarter is None and year) else ("quarterly" if quarter else "monthly")
            rows.append({
                "label": label,
                "href": href,
                "url": href if href.startswith("http") else BASE + href,
                "year": year,
                "month": month,
                "quarter": quarter,
                "type": rtype,
            })

    rows.sort(key=lambda r: (r["year"] or 0, r["month"] or 0, r["quarter"] or 0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    for r in rows:
        key = f"{r['year']}-{r['month'] or ('Q'+str(r['quarter']) if r['quarter'] else '?')}"
        print(f"{key}  {r['type']:8s}  {r['label']}")
    print(f"\nTotal: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())