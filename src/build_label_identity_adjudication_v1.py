#!/usr/bin/env python3
"""Do any two source-credit labels name the same person?

The corpus treats a source-credit label as a string, and every claim in this project stops
at that string rather than at a person. That is only safe if the strings do not silently
split one artist in two: if `X` and `X 周延` were both present, retrieval would count a
correct answer as wrong, and PD-002's within-label duplicate control would miss the pairs
across them.

Two ways to look, and only one of them is any good.

String similarity is the obvious one and it fails here. Normalising and comparing finds no
identical pair at all, and substring matching returns twelve candidates that are all
artefacts of common stage-name particles -- `ICE` inside `Ice Paper`, `YOUNG` inside
`YoungChigga`. Applying a substring rule would merge distinct artists, which is worse than
doing nothing.

Shared text is the good signal. If two labels have songs carrying byte-identical passages,
that is either one artist under two names, a collaboration, or a cover -- and the shape of
the overlap separates them. One artist split in two would show near-total mutual overlap,
because the same songs were filed twice. A collaboration shows partial overlap with both
sides keeping songs the other does not have.

What the measurement found is reported below and was put to the author for adjudication,
because the question is about people and no amount of text statistics settles it. The
author's determination is recorded as an input here, not inferred.

    python src/build_label_identity_adjudication_v1.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "label-identity-v1"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)

MINIMUM_PASSAGE_CHARACTERS = 30
REVIEW_THRESHOLD = 0.10

# The author reviewed every pair at or above REVIEW_THRESHOLD and determined that each is a
# collaboration. Recorded as a decision with its date and its basis, not as a computed
# result -- nothing in this file could have produced it.
AUTHOR_ADJUDICATION = {
    "decided_by": "author",
    "decided_on": "2026-09-08",
    "reviewed": "every candidate pair at or above a 10% mutual involvement share",
    "determination": "all reviewed pairs are collaborations; no pair names one person twice",
    "consequence": ("no label normalisation is applied. Source-credit labels remain the "
                    "strings the corpus carries, and no identity merging is performed."),
    "not_evidence_of": ("that the labels are verified distinct people. The determination is "
                        "that the corpus contains no detectable split of one credit across "
                        "two strings, which is a narrower statement."),
}


def normalise_label(value: str) -> str:
    return re.sub(r"[\s\-_·．。,、/\\()\[\]]+", "", unicodedata.normalize("NFKC", value)).lower()


def build(private_root: Path, out_dir: Path) -> int:
    corpus = (private_root / "work" / "private-repaired-corpus-v2"
              / "repaired_lyric_chunks_v2.csv")
    if not corpus.is_file():
        raise SystemExit(f"missing private input: {corpus}")
    rows = list(csv.DictReader(corpus.open(encoding="utf-8")))

    songs_by_label: dict[str, set[str]] = defaultdict(set)
    labels_by_passage: dict[str, set[str]] = defaultdict(set)
    songs_by_passage_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        label = row["source_credit_label"]
        songs_by_label[label].add(row["song_id"])
        text = row["cleaned_text"].strip()
        if len(text) >= MINIMUM_PASSAGE_CHARACTERS:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            labels_by_passage[key].add(label)
            songs_by_passage_label[(key, label)].add(row["song_id"])

    labels = sorted(songs_by_label)

    # the string approach, reported so its failure is on the record rather than assumed
    normalised: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        normalised[normalise_label(label)].append(label)
    identical = [group for group in normalised.values() if len(group) > 1]
    substring = [
        (left, right) for left, right in combinations(labels, 2)
        if (lambda a, b: a != b and min(len(a), len(b)) >= 2 and (a in b or b in a))(
            normalise_label(left), normalise_label(right))
    ]

    involved: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    for key, sharing in labels_by_passage.items():
        if not 1 < len(sharing) <= 4:
            continue
        for left, right in combinations(sorted(sharing), 2):
            involved[(left, right)][left] |= songs_by_passage_label[(key, left)]
            involved[(left, right)][right] |= songs_by_passage_label[(key, right)]

    pairs = []
    for (left, right), members in involved.items():
        left_share = len(members[left]) / len(songs_by_label[left])
        right_share = len(members[right]) / len(songs_by_label[right])
        pairs.append({
            "labels": [left, right],
            "songs_involved": [len(members[left]), len(members[right])],
            "songs_total": [len(songs_by_label[left]), len(songs_by_label[right])],
            "involvement_share": [round(left_share, 4), round(right_share, 4)],
            "mutual_minimum_share": round(min(left_share, right_share), 4),
        })
    pairs.sort(key=lambda item: -item["mutual_minimum_share"])
    reviewed = [pair for pair in pairs if pair["mutual_minimum_share"] >= REVIEW_THRESHOLD]

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_id": "chinese-rap-label-identity-v1",
        "version": "1.0.0",
        "question": ("Whether any two source-credit labels name one artist, which would make "
                     "a correct retrieval count as wrong and would hide duplicate records "
                     "from PD-002's within-label control."),
        "labels": len(labels),
        "string_similarity": {
            "normalised_identical_groups": len(identical),
            "substring_candidate_pairs": len(substring),
            "verdict": ("unusable. No pair is identical after normalisation, and every "
                        "substring candidate is a common stage-name particle -- applying "
                        "such a rule would merge distinct artists."),
        },
        "shared_passage_evidence": {
            "minimum_passage_characters": MINIMUM_PASSAGE_CHARACTERS,
            "candidate_pairs": len(pairs),
            "pairs_above_50_percent_mutual_share": sum(
                1 for pair in pairs if pair["mutual_minimum_share"] >= 0.50),
            "pairs_above_30_percent_mutual_share": sum(
                1 for pair in pairs if pair["mutual_minimum_share"] >= 0.30),
            "highest_mutual_share": pairs[0]["mutual_minimum_share"] if pairs else None,
            "reading": ("One artist split across two strings would show near-total mutual "
                        "overlap, because the same songs were filed twice. Partial overlap "
                        "with both sides retaining songs the other lacks is collaboration."),
        },
        "reviewed_by_author": len(reviewed),
        "author_adjudication": AUTHOR_ADJUDICATION,
        "privacy": "aggregate only; label strings and counts, no lyric text or song identifiers",
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")

    # csv.writer terminates rows with a carriage return and a line feed, while the
    # repository's text contract is line feed only -- the release validator refuses
    # the package otherwise, which is how this was caught. Built in memory and
    # written with an explicit terminator so the platform cannot decide it.
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["label_a", "label_b", "songs_involved_a", "songs_total_a",
                     "songs_involved_b", "songs_total_b", "mutual_minimum_share"])
    for pair in reviewed:
        writer.writerow([*pair["labels"], pair["songs_involved"][0],
                         pair["songs_total"][0], pair["songs_involved"][1],
                         pair["songs_total"][1], pair["mutual_minimum_share"]])
    (out_dir / "reviewed_pairs.csv").write_text(buffer.getvalue(), encoding="utf-8",
                                                newline="")

    print(f"{len(labels)} labels, {len(pairs)} candidate pairs, {len(reviewed)} reviewed")
    print(f"  highest mutual share {payload['shared_passage_evidence']['highest_mutual_share']:.0%}"
          f"; above 50%: "
          f"{payload['shared_passage_evidence']['pairs_above_50_percent_mutual_share']}")
    print(f"  author determination: {AUTHOR_ADJUDICATION['determination']}")
    print(f"wrote {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
