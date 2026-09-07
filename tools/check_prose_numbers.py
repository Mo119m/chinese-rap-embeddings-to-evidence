#!/usr/bin/env python3
"""Check that numbers stated in prose are still the numbers the artifacts hold.

This repository has already shipped one bug of this shape. README.md said the median was
0.82; the artifact said 0.7357. 0.82 was real -- it was the Shanghai row -- so the number
existed, it just was not the median. Nothing caught it, because nothing was comparing prose
against artifacts at all.

Two tiers, because the two halves of the problem need different machinery.

Tier 1 is automatic and needs no annotation. Every number appearing in prose is looked for
across every published JSON and CSV value. A number that appears in NO artifact is either
computed in the prose, rhetorical, or stale -- and after an artifact is regenerated, stale is
the common case. This tier catches "the artifact moved and the prose did not". It cannot
catch the 0.82 bug, because 0.82 did occur in an artifact.

Tier 2 catches that bug. A claim is registered with the exact artifact path it comes from,
and the check reads the artifact and compares. It costs one line per claim and it is the only
thing that can distinguish "this number exists somewhere" from "this number is the median".
The registry starts with the headline figures and grows; an unregistered claim is not an
error, so the file can be adopted gradually rather than all at once.

    python tools/check_prose_numbers.py                 # both tiers
    python tools/check_prose_numbers.py --tier2-only    # just the registered claims
    python tools/check_prose_numbers.py --list-unsourced

Exit status is non-zero if any registered claim is wrong, which is what CI should gate on.
Tier 1 reports but does not fail, because prose legitimately contains arithmetic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent

PROSE_GLOBS = [
    "README.md",
    "paper/*.md",
    "methods/*.md",
    "results/*/README.md",
    "results/*/METHOD.md",
    "results/*/*/*.md",
]
ARTIFACT_GLOBS = ["results/**/*.json", "results/**/*.csv", "validation/**/*.json"]

# A number token. Trailing % is kept so "26.2%" and "26.2" are not silently conflated.
NUMBER = re.compile(r"(?<![\w.\-])(\d+(?:,\d{3})*(?:\.\d+)?)(%?)(?![\w.])")

# Numbers that carry no artifact meaning: years, list markers, section numbers, small counts
# that appear in every prose file for structural reasons.
# A references section, and any single line that looks like a citation.
REFERENCE_HEADING = re.compile(r"#+\s*(references|bibliography|works cited|参考文献)\s*$", re.I)
CITATION_NOISE = re.compile(r"(doi\.org|10\.\d{4,5}/|arXiv:|ISBN|pp?\.\s*\d)", re.I)
# `340564...` is a truncated SHA-256 quoted as an identifier, not a quantity.
HASH_PREFIX = re.compile(r"`?[0-9a-f]{6,}(…|\.\.\.)")

IGNORE_EXACT = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "100", "1000",
    "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027",
}


# --------------------------------------------------------------------------- registry

def at(node, path: str):
    """Follow a '/a/b[2]/c' path into parsed JSON. Returns MISSING if it does not exist."""
    current = node
    for part in [p for p in path.split("/") if p]:
        while part.endswith("]") and "[" in part:
            part, _, index = part[:-1].partition("[")
            if part:
                if not isinstance(current, dict) or part not in current:
                    return MISSING
                current = current[part]
            if not isinstance(current, list) or int(index) >= len(current):
                return MISSING
            current = current[int(index)]
            part = ""
        if part:
            if not isinstance(current, dict) or part not in current:
                return MISSING
            current = current[part]
    return current


MISSING = object()

# Each entry: the prose file, the exact text that must appear in it, the artifact, the JSON
# path inside it, and how the artifact value is rendered in that sentence.
#
# `render` exists because prose does not print raw floats. A value of 0.7357 is written
# "0.74" in one place and "0.7357" in another, and both are correct; what would be wrong is
# writing the Shanghai row's value and calling it the median.
REGISTRY = [
    {
        "claim": "repertoire network eligible labels",
        "prose": "results/repertoire-network-v1/README.md",
        "text": "**204 source-credit labels**",
        "artifact": "results/repertoire-network-v1/graph/analysis_summary.json",
        "path": "/counts/graph_eligible_labels",
        "render": "{:d}",
    },
    {
        "claim": "repertoire network released edges",
        "prose": "results/repertoire-network-v1/README.md",
        "text": "**86 released edges**",
        "artifact": "results/repertoire-network-v1/graph/analysis_summary.json",
        "path": "/counts/stable_retained_edges",
        "render": "{:d}",
    },
    {
        "claim": "repertoire network connected labels",
        "prose": "results/repertoire-network-v1/README.md",
        "text": "**93 labels** have at least one released edge",
        "artifact": "results/repertoire-network-v1/graph/analysis_summary.json",
        "path": "/counts/connected_stable_graph_nodes",
        "render": "{:d}",
    },
    {
        "claim": "repertoire network PCA variance explained",
        "prose": "results/repertoire-network-v1/README.md",
        "text": "**26.2%**",
        "artifact": "results/repertoire-network-v1/graph/analysis_summary.json",
        "path": "/model/spatial_projection_variance_explained_2d",
        "render": "{:.1%}",
    },
    {
        "claim": "corpus v2 chunks",
        "prose": "methods/PROTOCOL_AMENDMENT_PD002_UPSTREAM_CHUNK_DEDUPLICATION.md",
        "text": "25,026 cleaned chunks",
        "artifact": "results/repaired-corpus-v2/analysis_summary.json",
        "path": "/corpus_geometry/chunk_rows",
        "render": "{:,d}",
    },
    {
        "claim": "corpus v2 text components",
        "prose": "methods/PROTOCOL_AMENDMENT_PD002_UPSTREAM_CHUNK_DEDUPLICATION.md",
        "text": "6,025 exact-text components",
        "artifact": "results/repaired-corpus-v2/analysis_summary.json",
        "path": "/corpus_geometry/text_components",
        "render": "{:,d}",
    },
    {
        "claim": "stray-title whole-chunk occurrences",
        "prose": "methods/METADATA_BLOCK_AUDIT_PROTOCOL.md",
        "text": "1,195 sit alone as a whole chunk",
        "artifact": "results/repaired-corpus-v2/analysis_summary.json",
        "path": "/stray_title_sensitivity/whole_chunk_occurrences",
        "render": "{:,d}",
    },
    {
        "claim": "stray-title line occurrences",
        "prose": "methods/METADATA_BLOCK_AUDIT_PROTOCOL.md",
        "text": "Of the 1,291 occurrences",
        "artifact": "results/repaired-corpus-v2/analysis_summary.json",
        "path": "/stray_title_sensitivity/line_occurrences_anywhere_in_a_chunk",
        "render": "{:,d}",
    },
    {
        "claim": "stray-title song records",
        "prose": "methods/METADATA_BLOCK_AUDIT_PROTOCOL.md",
        "text": "1,232 records carry at least one such line",
        "artifact": "results/repaired-corpus-v2/analysis_summary.json",
        "path": "/stray_title_sensitivity/song_records_carrying_at_least_one",
        "render": "{:,d}",
    },
    {
        "claim": "corpus v2 song records",
        "prose": "methods/METADATA_BLOCK_AUDIT_PROTOCOL.md",
        "text": "Over corpus v2 (7,391 song records)",
        "artifact": "results/repaired-corpus-v2/analysis_summary.json",
        "path": "/corpus_geometry/song_records",
        "render": "{:,d}",
    },
]


# --------------------------------------------------------------------------- tier 1

def artifact_values(root: Path) -> set[str]:
    """Every scalar an artifact publishes, as the strings a person would write."""
    seen: set[str] = set()

    def add(value):
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int):
            seen.add(str(value))
            seen.add(f"{value:,d}")
        elif isinstance(value, float):
            seen.add(repr(value))
            for digits in range(0, 7):
                seen.add(f"{value:.{digits}f}")
                seen.add(f"{value * 100:.{digits}f}")
        elif isinstance(value, str):
            text = value.strip().replace(",", "")
            if text.replace(".", "", 1).replace("-", "", 1).isdigit():
                add(float(text) if "." in text else int(text))

    def walk(node):
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        else:
            add(node)

    for pattern in ARTIFACT_GLOBS:
        for path in root.glob(pattern):
            try:
                if path.suffix == ".json":
                    walk(json.loads(path.read_text(encoding="utf-8")))
                else:
                    with path.open(encoding="utf-8", newline="") as handle:
                        for row in csv.reader(handle):
                            for cell in row:
                                add(cell)
            except Exception:
                continue
    return seen


def tier1(root: Path, published: set[str]) -> list[tuple[str, int, str]]:
    unsourced = []
    for pattern in PROSE_GLOBS:
        for path in sorted(root.glob(pattern)):
            in_references = False
            for line_number, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith(("|", "```", "    ")):
                    continue  # tables and code blocks quote artifacts directly
                if in_references or REFERENCE_HEADING.match(stripped):
                    in_references = in_references or bool(REFERENCE_HEADING.match(stripped))
                    continue
                # A bibliography entry is mostly DOI fragments, years and page ranges, none of
                # which are claims about this corpus. Left in, they bury the real signal: of
                # the 42 first found in the manuscript, 41 were citations and 1 was a result.
                if CITATION_NOISE.search(line) or HASH_PREFIX.search(line):
                    continue
                for digits, percent in NUMBER.findall(line):
                    bare = digits.replace(",", "")
                    if bare in IGNORE_EXACT:
                        continue
                    if bare in published or digits in published:
                        continue
                    unsourced.append(
                        (str(path.relative_to(root)), line_number, digits + percent))
    return unsourced


# --------------------------------------------------------------------------- tier 2

def tier2(root: Path) -> list[str]:
    failures = []
    for entry in REGISTRY:
        prose_path = root / entry["prose"]
        artifact_path = root / entry["artifact"]
        if not prose_path.is_file():
            failures.append(f"{entry['claim']}: prose file missing -- {entry['prose']}")
            continue
        if not artifact_path.is_file():
            failures.append(f"{entry['claim']}: artifact missing -- {entry['artifact']}")
            continue

        value = at(json.loads(artifact_path.read_text(encoding="utf-8")), entry["path"])
        if value is MISSING:
            failures.append(
                f"{entry['claim']}: {entry['path']} is not in {entry['artifact']}")
            continue

        expected = entry["render"].format(value)
        prose = prose_path.read_text(encoding="utf-8", errors="replace")

        if entry["text"]:
            if entry["text"] not in prose:
                failures.append(
                    f"{entry['claim']}: the registered sentence is no longer in "
                    f"{entry['prose']}\n      looked for: {entry['text']}\n"
                    f"      artifact now says: {expected}")
            elif expected not in entry["text"]:
                failures.append(
                    f"{entry['claim']}: prose and artifact disagree\n"
                    f"      prose says:    {entry['text']}\n"
                    f"      artifact says: {expected}  ({entry['path']})")
        elif expected not in prose:
            failures.append(
                f"{entry['claim']}: {entry['prose']} never states {expected} "
                f"({entry['path']} in {entry['artifact']})")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--tier2-only", action="store_true")
    parser.add_argument("--list-unsourced", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    failures = tier2(root)
    print(f"tier 2 -- {len(REGISTRY)} registered claims checked against their artifacts")
    for failure in failures:
        print(f"  FAIL  {failure}")
    if not failures:
        print("  every registered claim matches its artifact")

    if not args.tier2_only:
        published = artifact_values(root)
        unsourced = tier1(root, published)
        print(f"\ntier 1 -- {len(published):,} distinct published values; "
              f"{len(unsourced)} prose numbers match none of them")
        if args.list_unsourced:
            for path, line, token in unsourced:
                print(f"  {path}:{line}  {token}")
        else:
            by_file: dict[str, int] = {}
            for path, _, _ in unsourced:
                by_file[path] = by_file.get(path, 0) + 1
            for path, count in sorted(by_file.items(), key=lambda kv: -kv[1])[:12]:
                print(f"  {count:4d}  {path}")
            print("  (--list-unsourced for every one; prose arithmetic lands here too)")

    print()
    if failures:
        print(f"{len(failures)} registered claim(s) do not match their artifact")
        return 1
    print("no registered claim contradicts its artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
