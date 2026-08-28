#!/usr/bin/env python3
"""Verify that every shipped manuscript derivative matches its Markdown source.

Individual file checksums cannot detect a workflow error in which the Markdown is
edited and all manifests are refreshed without rebuilding the DOCX/PDF files.  This
small provenance record closes that gap.

Run with ``--record`` only immediately after rebuilding every listed derivative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "paper" / "derivative_provenance.json"
DERIVATIVES = {
    "paper/Chinese_Rap_Evidence_Grounded_Manuscript.docx": "paper/manuscript.md",
    "paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf": "paper/manuscript.md",
    "paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx": "paper/manuscript.md",
    "paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf": "paper/manuscript.md",
    "paper/Chinese_Rap_Evidence_Grounded_Supplement.docx": "paper/supplementary_methods.md",
    "paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf": "paper/supplementary_methods.md",
    "submission/dsh/manuscript.docx": "paper/manuscript.md",
    "submission/dsh/manuscript_preview.pdf": "paper/manuscript.md",
    "submission/dsh/supplementary_methods.docx": "paper/supplementary_methods.md",
    "submission/dsh/supplementary_methods_preview.pdf": "paper/supplementary_methods.md",
}


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        action="store_true",
        help="record current hashes immediately after rebuilding all derivatives",
    )
    args = parser.parse_args()

    required = set(DERIVATIVES) | set(DERIVATIVES.values())
    missing = sorted(path for path in required if not (ROOT / path).is_file())
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 2

    sources = sorted(set(DERIVATIVES.values()))
    source_hashes = {source: sha256(source) for source in sources}

    if args.record:
        payload = {
            "note": (
                "Source and derivative hashes recorded immediately after rebuilding. "
                "Verified by tools/check_manuscript_derivatives.py."
            ),
            "sources": source_hashes,
            "derivatives": {
                name: {"built_from": source, "sha256": sha256(name)}
                for name, source in DERIVATIVES.items()
            },
        }
        PROVENANCE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"recorded {len(sources)} sources and {len(DERIVATIVES)} derivatives")
        return 0

    if not PROVENANCE.is_file():
        print(
            f"missing {PROVENANCE.relative_to(ROOT).as_posix()}; rebuild and run --record",
            file=sys.stderr,
        )
        return 2

    recorded = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    stale_sources = [
        source
        for source in sources
        if recorded.get("sources", {}).get(source) != source_hashes[source]
    ]
    recorded_derivatives = recorded.get("derivatives", {})
    changed_derivatives = [
        name
        for name in DERIVATIVES
        if recorded_derivatives.get(name, {}).get("built_from") != DERIVATIVES[name]
        or recorded_derivatives.get(name, {}).get("sha256") != sha256(name)
    ]

    if stale_sources:
        print("manuscript source changed after its derivatives were built:", file=sys.stderr)
        for source in stale_sources:
            print(f"  {source}", file=sys.stderr)
        return 1
    if changed_derivatives:
        print("manuscript derivatives differ from the provenance record:", file=sys.stderr)
        for name in changed_derivatives:
            print(f"  {name}", file=sys.stderr)
        return 1

    print(f"manuscript derivatives are current with {len(sources)} source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
