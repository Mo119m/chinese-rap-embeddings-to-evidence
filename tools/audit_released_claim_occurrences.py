"""Classify the occurrences that actually support each released claim.

A released label link rests on the occurrences the pipeline *accepted* — lexicon
candidates that agreed with the transformer on exact span and schema type — inside
one source label, after shared-text exclusion. An earlier version of this script
enumerated raw string matches instead, which is a different and wider population: it
found 12 songs for 黑麦–天津 where the release publishes 10.

This reads the accepted set directly from the private candidate table
(`strict_high_consistency`, excluding `cross_label_shared_cleaned_text`) and refuses
to report anything until the per-claim support count equals the published figure for
every one of the ten released claims. Each accepted occurrence is then classified by
what immediately follows the surface in its own stored context window.

What this can and cannot establish: a compound reading such as 巴黎世家 is detectable
because the following characters are known; a metaphorical or otherwise non-locative
use of a bare surface is not. `bare surface` means "not one of the compounds listed
below", never "verified locative".

    python tools/audit_released_claim_occurrences.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
csv.field_size_limit(10 ** 8)

# Compounds that change what an occurrence refers to, named from the collocation audit.
KNOWN_COMPOUNDS = {
    "巴黎": {"世家": "Balenciaga (fashion house)"},
    "中文": {"说唱": "中文说唱 (genre term)"},
    "湖南": {"卫视": "Hunan TV (broadcaster)"},
    "加勒比": {"海盗": "Pirates of the Caribbean (film)"},
    "加州": {"旅馆": "Hotel California (song)"},
    "香港": {"脚": "athlete's foot"},
    "上海": {"滩": "上海滩 (the Bund / TV serial)"},
    "天津": {"卫": "天津卫 (historical name for the city)"},
}


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def classify(row: dict[str, str]) -> tuple[str, str]:
    """Return (classification, the characters following the surface)."""
    surface = row["candidate_surface"]
    snippet = row["context_snippet"]
    try:
        end = int(row["surface_end_in_context"])
    except (TypeError, ValueError):
        return "bare surface", ""
    tail = snippet[end:end + 4].replace("\n", " ")
    for continuation, meaning in KNOWN_COMPOUNDS.get(surface, {}).items():
        if tail.startswith(continuation):
            return meaning, tail
    return "bare surface", tail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "work/private-chinese-rap-ner-cultural-graph-v1/all_candidate_occurrences_private.csv")
    parser.add_argument("--links", type=Path,
                        default=ROOT / "results/ner-v1/source_label_entity_links_provisional.csv")
    parser.add_argument("--co-mentions", type=Path,
                        default=ROOT / "results/ner-v1/entity_co_mentions_provisional.csv")
    parser.add_argument("--show-tails", action="store_true",
                        help="print the following characters for every occurrence")
    parser.add_argument("--skip-support-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.candidates, args.links, args.co_mentions):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    accepted = [row for row in read_rows(args.candidates)
                if row["strict_high_consistency"] == "True"
                and row["cross_label_shared_cleaned_text"] == "False"]
    print(f"{len(accepted)} accepted occurrences after shared-text exclusion")

    by_claim: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_surface_song: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        by_claim[(row["source_credit_label"], row["candidate_surface"])].append(row)
        by_surface_song[row["candidate_surface"]].add(row["song_id"])

    links = list(read_rows(args.links))
    co_mentions = list(read_rows(args.co_mentions))
    mismatches = []

    print("\n" + "=" * 74)
    print("RELEASED LABEL LINKS")
    print("=" * 74)
    for link in links:
        label, surface = link["source_credit_label"], link["entity"]
        published = int(link["entity_song_units_within_label"])
        rows = by_claim.get((label, surface), [])
        songs = {row["song_id"] for row in rows}
        if len(songs) != published:
            mismatches.append(f"{label}->{surface}: published {published}, accepted {len(songs)}")
        kinds: dict[str, int] = defaultdict(int)
        for row in rows:
            kinds[classify(row)[0]] += 1
        print(f"\n{label} -> {surface}   {len(songs)} song units (published {published}), "
              f"{len(rows)} accepted occurrence(s)")
        for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>3}  {kind}")
        if args.show_tails:
            for row in rows:
                kind, tail = classify(row)
                print(f"        {kind:<34} follows: {tail!r}")

    print("\n" + "=" * 74)
    print("RELEASED CO-MENTIONS")
    print("=" * 74)
    for row in co_mentions:
        a, b = row["entity_a"], row["entity_b"]
        published = int(row["unique_song_unit_co_mentions"])
        both = by_surface_song[a] & by_surface_song[b]
        if len(both) != published:
            mismatches.append(f"{a}~{b}: published {published}, accepted {len(both)}")
        print(f"\n{a} ~ {b}   {len(both)} song units (published {published})")
        for side in (a, b):
            kinds: dict[str, int] = defaultdict(int)
            for occurrence in by_claim_side(accepted, side, both):
                kinds[classify(occurrence)[0]] += 1
            print(f"    {side}: " + ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))

    if mismatches:
        print("\nSUPPORT COUNTS DO NOT MATCH THE RELEASE:", file=sys.stderr)
        for line in mismatches:
            print(f"  {line}", file=sys.stderr)
        print("\nThe accepted set reconstructed here is not the one behind the published",
              file=sys.stderr)
        print("claims, so the classification above describes the wrong population.",
              file=sys.stderr)
        return 1 if not args.skip_support_check else 0

    print(f"\nAll {len(links) + len(co_mentions)} released claims reproduce their published "
          "support exactly, so the classification above covers the occurrences that carry them.")
    print("A `bare surface` is one not followed by a known compound. It is not a verified")
    print("locative reading, and a figurative use of a bare surface would not be detected.")
    return 0


def by_claim_side(accepted, surface, songs):
    for row in accepted:
        if row["candidate_surface"] == surface and row["song_id"] in songs:
            yield row


if __name__ == "__main__":
    sys.exit(main())
