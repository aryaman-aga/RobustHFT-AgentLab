#!/usr/bin/env python
"""Build the anonymized code supplement for a double-blind submission (paper/aaai/aaai27_supplementary.zip).

Only tracked files under INCLUDE are copied (no git history, outputs, paper sources or BTP reports).
Lines of the README that name the authors or their institution are dropped, and the frontend's
"B.Tech project" tag is removed, in the copy only. Every copied text file is then scanned for the
author names, affiliation and repository owner; the zip is written only if nothing is found.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile

from _common import ROOT

INCLUDE = [
    "src",
    "scripts",
    "configs",
    "tests",
    "frontend",
    "docs",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "requirements-optional.txt",
    ".env.example",
    "README.md",
]
IDENTITY = re.compile(r"aryaman|agarwal|arusha|habib|ayana|kumar|netaji|subhas|nsut|neatmail|b\.?\s?tech|\bbtp\b", re.I)
OUT = ROOT / "paper" / "aaai" / "aaai27_supplementary.zip"


def redact(path: str, text: str) -> str:
    """Remove the identifying lines/phrases this repository is known to contain (see module docstring)."""
    if path == "README.md":
        text = "".join(line for line in text.splitlines(keepends=True) if not IDENTITY.search(line))
    return text.replace("B.Tech project · ", "").replace('name = "BTP Student"', 'name = "Anonymous"')


def main() -> None:
    files = subprocess.run(["git", "ls-files", "--", *INCLUDE], cwd=ROOT, capture_output=True, text=True, check=True)
    paths = sorted(p for p in files.stdout.splitlines() if (ROOT / p).is_file())
    hits: list[str] = []
    payload: dict[str, bytes] = {}
    for p in paths:
        raw = (ROOT / p).read_bytes()
        try:
            text = redact(p, raw.decode("utf-8"))
        except UnicodeDecodeError:  # binary file: keep as is, but still scan its bytes
            if IDENTITY.search(raw.decode("latin-1")):
                hits.append(f"{p} (binary)")
            payload[p] = raw
            continue
        hits += [
            f"{p}:{i}: {line.strip()[:80]}" for i, line in enumerate(text.splitlines(), 1) if IDENTITY.search(line)
        ]
        payload[p] = text.encode("utf-8")
    if hits:
        print("identifying text left in the supplement; nothing written:", *hits, sep="\n  ")
        sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    note = (
        "Anonymized code supplement for double-blind review. Quick start: `make install test`, then\n"
        "`make pipeline ablation report` (see README.md). The paper sources and project reports are omitted.\n"
    )
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("SUPPLEMENT.txt", note)
        for p, data in payload.items():
            z.writestr(f"code/{p}", data)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(payload)} files, no identifying text found")


if __name__ == "__main__":
    main()
