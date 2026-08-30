"""Detect a manuscript whose DOCX and PDF no longer match the Markdown they came from.

The release ships `paper/manuscript.md` alongside four binary derivatives, and every
one of them carries a SHA-256 in the published manifests. Those checksums describe
each file individually, so editing the Markdown and re-deriving the manifests leaves
a release where the source and the derivatives say different things and **every
checksum still validates**. The PDF step is not part of the tracked pipeline, which
makes it easy to update the Markdown and forget.

This closes that gap by recording, in `paper/derivative_provenance.json`, the source
hash each derivative was built from, and failing when the current source differs.

    python tools/check_manuscript_derivatives.py            # verify
    python tools/check_manuscript_derivatives.py --record   # after a rebuild

`--record` asserts the derivatives are current, so run it only immediately after
rebuilding them, never to silence a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = ROOT / "paper" / "derivative_provenance.json"

# derivative -> the Markdown it is built from
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true",
                        help="write the current source hashes; use only just after rebuilding")
    args = parser.parse_args()

    missing = [path for path in (*DERIVATIVES, *set(DERIVATIVES.values())) if not (ROOT / path).is_file()]
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 2

    sources = sorted(set(DERIVATIVES.values()))
    current = {source: sha256(source) for source in sources}

    if args.record:
        PROVENANCE.write_text(json.dumps({
            "note": "Source hashes the manuscript derivatives were last built from. "
                    "Checked by tools/check_manuscript_derivatives.py; update only by "
                    "rebuilding the derivatives and rerunning that script with --record.",
            "sources": current,
            "derivatives": {name: {"built_from": source, "sha256": sha256(name)}
                            for name, source in DERIVATIVES.items()},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"recorded {len(sources)} source hash(es) and {len(DERIVATIVES)} derivative(s)")
        return 0

    if not PROVENANCE.is_file():
        print(f"no provenance record at {PROVENANCE.relative_to(ROOT).as_posix()}; "
              "run with --record just after a rebuild", file=sys.stderr)
        return 2

    recorded = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    stale_sources = [s for s in sources if recorded["sources"].get(s) != current[s]]
    changed_derivatives = [name for name, entry in recorded["derivatives"].items()
                           if sha256(name) != entry["sha256"]]

    if stale_sources:
        print("the manuscript source has changed since the derivatives were built:",
              file=sys.stderr)
        for source in stale_sources:
            affected = sorted(n for n, s in DERIVATIVES.items() if s == source)
            print(f"  {source} changed; rebuild {len(affected)} derivative(s):", file=sys.stderr)
            for name in affected:
                print(f"    {name}", file=sys.stderr)
        print("\nEvery individual checksum still validates, which is exactly why this",
              file=sys.stderr)
        print("check exists. Rebuild, then rerun with --record.", file=sys.stderr)
        return 1

    if changed_derivatives:
        print("derivatives changed without their source changing "
              "(a rebuild that was not recorded):", file=sys.stderr)
        for name in changed_derivatives:
            print(f"  {name}", file=sys.stderr)
        print("\nRerun with --record if the rebuild was intended.", file=sys.stderr)
        return 1

    print(f"manuscript derivatives are current with {len(sources)} source file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
