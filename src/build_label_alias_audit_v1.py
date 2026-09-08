#!/usr/bin/env python3
"""Does an author-supplied artist list show one person under two corpus labels?

The label identity audit (`build_label_identity_adjudication_v1.py`) looks for one artist
filed under two strings through shared lyric text, and finds only collaborations. That
method is blind to the case it cannot see: an artist whose early songs sit under one name
and later songs under another, sharing no text. An artist list that writes both names on
one line -- `刘聪 (Key.L)`, `Regi 陈彦希` -- can catch exactly that.

This file reads the author's list, takes only the name column, splits each entry into
its surface forms (the whole entry, a parenthesised alternative, and space-separated
parts), normalises them the way the identity audit normalises labels, and reports which
corpus labels the list covers and whether any list entry maps to more than one corpus
label. A pair that does is a candidate for the author to rule on; nothing is merged here.

The list also carries a region column. It is not read. The project's evidence boundary
forbids validating any label-to-place association with an artist's hometown, base, or
biography, and a builder that loaded the column would be one edit away from doing so.

    python src/build_label_alias_audit_v1.py --private-root <ni-k> --artist-list <csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "label-identity-v1"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)
PAREN = re.compile(r"[（(]([^()（）]+)[)）]")


def normalise(value: str) -> str:
    return re.sub(r"[\s\-_·．。,、/\\()（）\[\]]+", "", unicodedata.normalize("NFKC", value)).lower()


def surface_forms(entry: str) -> list[str]:
    """The whole entry, any parenthesised alternative, and the parts around it."""
    forms = [entry.strip()]
    for inner in PAREN.findall(entry):
        forms.append(inner.strip())
    outside = PAREN.sub(" ", entry)
    forms.extend(part for part in outside.split() if part)
    seen, out = set(), []
    for form in forms:
        key = normalise(form)
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            out.append(form)
    return out


def build(private_root: Path, artist_list: Path, out_dir: Path) -> int:
    corpus = (private_root / "work" / "private-repaired-corpus-v2"
              / "repaired_lyric_chunks_v2.csv")
    labels = sorted({row["source_credit_label"]
                     for row in csv.DictReader(corpus.open(encoding="utf-8"))})
    by_key: dict[str, str] = {normalise(label): label for label in labels}

    # only the name column; the region column is deliberately never read
    entries = [row["artist"].strip() for row in csv.DictReader(artist_list.open(encoding="utf-8-sig"))
               if row.get("artist", "").strip()]

    matched_labels: set[str] = set()
    entry_hits: dict[str, list[str]] = {}
    for entry in entries:
        hits = []
        for form in surface_forms(entry):
            label = by_key.get(normalise(form))
            if label and label not in hits:
                hits.append(label)
        if hits:
            entry_hits[entry] = hits
            matched_labels.update(hits)
    # An entry that reaches two labels only because one of its words is itself another
    # artist's whole name -- `Ice Paper` reaching `ICE` -- is the stage-name-particle
    # artefact the identity audit already documented, not an alias. If the other label is
    # matched by a separate list entry as a whole, the list itself says they are two people.
    whole_entry_labels = {by_key.get(normalise(entry)) for entry in entries} - {None}
    multi = {}
    for entry, hits in entry_hits.items():
        if len(hits) < 2:
            continue
        own = by_key.get(normalise(entry))
        others = [label for label in hits if label != own and label not in whole_entry_labels]
        if others:
            multi[entry] = hits
    dismissed = {entry: hits for entry, hits in entry_hits.items()
                 if len(hits) > 1 and entry not in multi}
    # the reverse: one corpus label reached from more than one list entry
    label_entries: dict[str, list[str]] = defaultdict(list)
    for entry, hits in entry_hits.items():
        for label in hits:
            label_entries[label].append(entry)
    shared = {label: ents for label, ents in label_entries.items() if len(ents) > 1}

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_id": "chinese-rap-label-alias-audit-v1",
        "question": ("whether an author-supplied artist list, which writes alternative names "
                     "on one line, maps any single entry to more than one corpus label -- "
                     "the split the shared-text audit cannot see"),
        "artist_list": {"file": artist_list.name, "entries": len(entries),
                        "columns_read": ["artist"],
                        "columns_not_read": ["region -- the evidence boundary forbids using an "
                                             "artist's place to validate any label-to-place edge"],
                        "completeness": "partial; the author will supply a complete list"},
        "corpus_labels": len(labels),
        "labels_matched_by_the_list": len(matched_labels),
        "list_entries_matching_a_label": len(entry_hits),
        "entries_mapping_to_more_than_one_label": [
            {"entry": entry, "labels": hits} for entry, hits in sorted(multi.items())],
        "entries_dismissed_as_particle_matches": [
            {"entry": entry, "labels": hits, "why": "every other label reached is itself a "
                                                    "separate whole entry of the list"}
            for entry, hits in sorted(dismissed.items())],
        "labels_reached_by_more_than_one_entry": [
            {"label": label, "entries": ents} for label, ents in sorted(shared.items())],
        "consequence": ("no label is merged here; every pair above is a candidate for the "
                        "author's ruling, recorded as a decision if and when it is made"),
        "privacy": "label strings and counts only; no lyric text, song identifiers, or regions",
    }
    (out_dir / "alias_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"{len(entries)} list entries; {len(matched_labels)} of {len(labels)} corpus labels matched")
    print(f"  entries mapping to more than one label: {len(multi)}")
    for entry, hits in sorted(multi.items()):
        print(f"    {entry}  ->  {hits}")
    print(f"  labels reached by more than one entry: {len(shared)}")
    for label, ents in sorted(shared.items()):
        print(f"    {label}  <-  {ents}")
    print(f"wrote {out_dir / 'alias_audit.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--artist-list", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.artist_list.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
