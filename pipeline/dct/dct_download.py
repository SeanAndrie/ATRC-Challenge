import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import DCT_MANIFEST, DCT_PDFS  # noqa: E402

MANIFEST = DCT_MANIFEST
OUTDIR = DCT_PDFS


def fname(r):
    if r["type"] == "annual":
        return f"{r['year']:d}-annual-{r['label'].split()[-1]}.pdf"
    if r["type"] == "quarterly":
        return f"{r['year']:d}-Q{r['quarter']}.pdf"
    return f"{r['year']:d}-{r['month']:02d}.pdf"


def main():
    rows = json.loads(MANIFEST.read_text())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research parsing of DCT Abu Dhabi hotel reports)"
    ok = fail = skip = 0
    for r in rows:
        dest = OUTDIR / fname(r)
        if dest.exists() and dest.stat().st_size > 1000:
            skip += 1
            continue
        for attempt in range(3):
            try:
                resp = session.get(r["url"], timeout=90)
                resp.raise_for_status()
                if not resp.content.startswith(b"%PDF"):
                    raise ValueError("not a PDF")
                dest.write_bytes(resp.content)
                ok += 1
                print(f"OK   {dest.name}  {len(resp.content)//1024} kB")
                break
            except Exception as e:
                print(f"FAIL {r['year']}-{r.get('month')} attempt {attempt}: {e}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        else:
            fail += 1
        time.sleep(0.3)
    print(f"\ndownloaded={ok} skipped={skip} failed={fail} total={len(rows)}")


if __name__ == "__main__":
    main()