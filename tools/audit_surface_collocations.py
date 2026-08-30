"""Turn occurrence-level review of the entity inventory into a table you can read.

`boundary_ok()` in the NER builder returns True unconditionally for surfaces with
no ASCII characters, i.e. every Chinese surface, so the lexicon stage alone cannot
tell 巴黎 from 巴黎世家 or 加勒比 from 加勒比海盗. Compound absorption is caught
downstream, by the exact-span agreement requirement against the transformer
baseline -- but nothing reports *what* the disagreements were.

This script reports it. For every released surface it tabulates the characters
that immediately follow and precede each match, so a compound that swallows the
surface shows up as a high-frequency continuation instead of having to be found by
reading occurrences one at a time. Twenty-two surfaces produce one table per
surface; reviewing them is minutes of work, not an 800-item annotation round.

It reads only the private lyric sidecar and the curated lexicon, and it prints
counts and short context windows -- never full lines -- so its output stays inside
the public release boundary.

    python tools/audit_surface_collocations.py \
        --chunks work/private-canonical-lyric-text-sidecar-v1/cleaned_analysis_chunks_v1.csv \
        --lexicon outputs/chinese-rap-curated-atlas-v3/safe_lexicon_catalog.csv \
        --surfaces results/ner-v1/entity_aggregate_provisional.csv \
        --window 3 --top 12
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=Path,
                        default=ROOT / "work/private-canonical-lyric-text-sidecar-v1/cleaned_analysis_chunks_v1.csv")
    parser.add_argument("--lexicon", type=Path,
                        default=ROOT / "outputs/chinese-rap-curated-atlas-v3/safe_lexicon_catalog.csv")
    parser.add_argument("--surfaces", type=Path,
                        default=ROOT / "results/ner-v1/entity_aggregate_provisional.csv")
    parser.add_argument("--window", type=int, default=3, help="context characters on each side")
    parser.add_argument("--top", type=int, default=12, help="continuations to print per surface")
    parser.add_argument("--out", type=Path, default=None, help="optional CSV of the full table")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    for path in (args.chunks, args.lexicon, args.surfaces):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    released = {row["entity"]: row for row in read_rows(args.surfaces)}
    # Longer lexicon surfaces are what the builder's longest-first resolution would
    # have used, so knowing which ones exist tells you which compounds are already
    # handled and which are invisible to the matcher.
    lexicon_surfaces = {str(row["entity"]).strip() for row in read_rows(args.lexicon) if str(row["entity"]).strip()}

    right: dict[str, Counter] = {s: Counter() for s in released}
    left: dict[str, Counter] = {s: Counter() for s in released}
    totals: Counter = Counter()

    with args.chunks.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = row.get("analysis_text") or ""
            if not text:
                continue
            for surface in released:
                start = text.find(surface)
                while start != -1:
                    end = start + len(surface)
                    totals[surface] += 1
                    right[surface][text[end:end + args.window]] += 1
                    left[surface][text[max(0, start - args.window):start]] += 1
                    start = text.find(surface, start + 1)

    records = []
    print(f"{'surface':<10}{'type':<32}{'matches':>8}  {'strict/lex':>11}  top right-context continuations")
    print("-" * 118)
    for surface, meta in sorted(released.items(), key=lambda kv: -totals[kv[0]]):
        rate = meta.get("strict_agreement_rate", "")
        ratio = f"{meta.get('strict_agreement_occurrences','?')}/{meta.get('lexicon_candidate_occurrences','?')}"
        shown = []
        for continuation, count in right[surface].most_common(args.top):
            if not continuation.strip():
                continue
            compound = surface + continuation
            # flag continuations the lexicon cannot resolve on its own
            covered = any(compound.startswith(other) and other != surface for other in lexicon_surfaces)
            shown.append(f"{continuation}({count}){'' if covered else '*'}")
        print(f"{surface:<10}{meta.get('entity_type',''):<32}{totals[surface]:>8}  {ratio:>11}  {' '.join(shown[:args.top])}")
        for side, counter in (("right", right[surface]), ("left", left[surface])):
            for continuation, count in counter.most_common():
                records.append({
                    "entity": surface,
                    "entity_type": meta.get("entity_type", ""),
                    "strict_agreement_rate": rate,
                    "side": side,
                    "context": continuation,
                    "occurrences": count,
                    "lexicon_has_longer_entry": any(
                        (surface + continuation if side == "right" else continuation + surface).startswith(other)
                        and other != surface for other in lexicon_surfaces),
                })

    print("\n* = no longer lexicon entry covers this compound, so the lexicon stage")
    print("  matched the bare surface and only the transformer span check could reject it.")
    print("  Read these first: they are where an occurrence may not mean the entity.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        print(f"\nfull table written to {args.out} ({len(records)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
