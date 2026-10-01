"""Cache `pdftotext -layout` output for every DCT PDF in data/raw/dct_pdfs/.

Writes the derived text to data/interim/dct_text/. The PDFs are read-only.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import DCT_PDFS, DCT_PDF_SIZES, DCT_TEXT  # noqa: E402

PDFDIR = DCT_PDFS
TEXTDIR = DCT_TEXT


def main():
    TEXTDIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDFDIR.glob("*.pdf"))
    if len(pdfs) != 99:
        print(f"WARNING: expected 99 pdfs, found {len(pdfs)}", file=sys.stderr)
    ok = 0
    for p in pdfs:
        dest = TEXTDIR / (p.stem + ".txt")
        if dest.exists() and dest.stat().st_size > 200:
            ok += 1
            continue
        proc = subprocess.run(
            ["pdftotext", "-layout", str(p), "-"], capture_output=True, timeout=300
        )
        text = proc.stdout.decode("utf-8", errors="replace")
        dest.write_text(text, encoding="utf-8")
        if text.strip():
            ok += 1
        print(f"{'OK ' if text.strip() else 'EMPTY'} {p.name} chars={len(text)}")
    print(f"\ntext files ok={ok}/{len(pdfs)}")
    sizes = {p.stem: p.stat().st_size for p in pdfs}
    (DCT_PDF_SIZES).write_text(json.dumps(sizes, indent=1))
    print("wrote pdf_sizes.json")


if __name__ == "__main__":
    main()
