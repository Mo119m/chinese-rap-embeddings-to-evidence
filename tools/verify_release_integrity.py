"""Verify that every published checksum in this release matches the committed bytes.

Runs on a plain checkout with no dependencies beyond the standard library:

    python tools/verify_release_integrity.py

Exits non-zero if any in-repository claim fails, so it can gate CI. Claims that
point at private build inputs under work/ or outputs/ are reported separately and
are expected to be unresolvable in the public release.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HEX = re.compile(r"^[0-9a-f]{64}$")

# sha256 fields that name their target in the key rather than in a "path" entry
KEYED_TARGETS = {
    "validation/release_validation.json": {
        "portable_site_sha256": "index.html",
        "application_data_sha256": "site/app/data/researchData.json",
        "manuscript_markdown_sha256": "paper/manuscript.md",
        "review_manuscript_docx_sha256": "paper/Chinese_Rap_Evidence_Grounded_Manuscript.docx",
        "review_manuscript_pdf_sha256": "paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf",
        "dsh_manuscript_docx_sha256": "paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx",
        "dsh_manuscript_pdf_sha256": "paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf",
        "supplement_docx_sha256": "paper/Chinese_Rap_Evidence_Grounded_Supplement.docx",
        "supplement_pdf_sha256": "paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf",
        "journal_figure_validation_sha256": "figures/journal_figure_validation.json",
    },
    "validation/portable_site_manifest.json": {
        "portableHtmlSha256": "index.html",
        "researchDataSourceSha256": "site/app/data/researchData.json",
    },
}

_digests: dict[str, str] = {}


def sha256(rel: str) -> str:
    if rel not in _digests:
        _digests[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    return _digests[rel]


def resolve(hint: object, manifest_dir: str) -> str | None:
    if not isinstance(hint, str) or not hint:
        return None
    hint = hint.replace("\\", "/")
    for candidate in (f"{manifest_dir}/{hint}" if manifest_dir else hint, hint):
        if (ROOT / candidate).is_file():
            return candidate
    return None


def claims_in(path: Path):
    rel = path.relative_to(ROOT).as_posix()
    manifest_dir = "/".join(rel.split("/")[:-1])
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return

    def walk(node, key_hint):
        if isinstance(node, dict):
            digest = node.get("sha256")
            if isinstance(digest, str) and HEX.match(digest):
                target = resolve(node.get("path") or node.get("file") or key_hint, manifest_dir)
                if target is None:
                    yield rel, str(node.get("path") or node.get("file") or key_hint), "EXTERNAL"
                elif sha256(target) != digest:
                    yield rel, target, "BAD_SHA"
                elif isinstance(node.get("bytes"), int) and node["bytes"] != (ROOT / target).stat().st_size:
                    yield rel, target, "BAD_BYTES"
                else:
                    yield rel, target, "OK"
            for key, value in node.items():
                yield from walk(value, key)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value, key_hint)

    yield from walk(document, None)

    if isinstance(document, dict):
        holders = [document] + [v for v in document.values() if isinstance(v, dict)]
        for key, target in KEYED_TARGETS.get(rel, {}).items():
            for holder in holders:
                if key in holder:
                    yield rel, f"{key} -> {target}", "OK" if holder[key] == sha256(target) else "BAD_SHA"


def main() -> int:
    results = set()
    for path in sorted(ROOT.rglob("*.json")):
        if ".git" in path.parts or "node_modules" in path.parts:
            continue
        results.update(claims_in(path))

    tally = Counter(kind for _, _, kind in results)
    print(f"checksum claims: {len(results)}")
    for kind in ("OK", "BAD_SHA", "BAD_BYTES", "EXTERNAL"):
        print(f"  {kind:<10} {tally[kind]}")

    failures = sorted(r for r in results if r[2].startswith("BAD"))
    if failures:
        print("\nFAILED:")
        for manifest, target, kind in failures:
            print(f"  [{kind}] {manifest} -> {target}")
        return 1

    if tally["OK"] == 0:
        # zero verified claims is not success: run from a repository checkout, where the
        # documents carrying the checksum claims exist
        print("\nno in-repository checksum claims were found; this tool verifies a "
              "repository checkout, not a packaged subset", file=sys.stderr)
        return 2
    print(f"\nall {tally['OK']} in-repository claims verified "
          f"({tally['EXTERNAL']} reference private build inputs and are not checked here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
