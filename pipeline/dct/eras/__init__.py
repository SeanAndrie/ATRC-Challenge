"""Report-era readers for the Abu Dhabi DCT hotel statistics series.

The published series is not one document reprinted. Between 2015 and 2025 DCT
changed layout, unit conventions and table structure several times, so each era
gets its own reader and dct_parse.py routes a report to the right one. See
docs/decisions/dct_notes.md for what distinguishes each era and why the split
happens where it does.

Each module here exposes parse functions over extracted PDF text. They read
regexes and bounds, not files or paths: text location is dct_common's job, and
keeping it out of here means an era reader can be tested against a text fixture
without a repository checkout.

Imported as a package by dct_parse.py, which adds pipeline/dct to sys.path
first. That is what lets these modules use relative imports of each other.
"""

from __future__ import annotations
