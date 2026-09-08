#!/usr/bin/env python3
"""Written-ending continuation on the PD-002 repaired corpus.

The v1 artifact withheld its predictive metrics, in its own words "pending duplicate-aware
corpus reconstruction, splitting, fitting, and evaluation under PD-002". This is that rerun.
The v1 builder and its outputs are not edited; both are pinned by manifests, and the
amendment records v1 as internally valid on the population it used.

The change that matters is the split. v1 assigned songs to train, validation and test
individually, with no notion of shared text, so two songs carrying byte-identical chunks
could land on opposite sides of the boundary. Measured by running v1's own assign_song_splits
over corpus v2: 433 of 799 multi-song text components straddled a partition, touching 1,330
song records -- 18.0% of the population. PD-002 rule 3 forbids exactly that.

The fix is not a new algorithm. assign_song_splits is multilabel-aware, balanced and
deterministic, and all of that is worth keeping; what changes is the UNIT it assigns. Whole
leakage groups are assigned instead of songs, a group carrying the union of its members'
labels, and each song inherits its group's partition. Measured: component straddling falls
to zero, the split still lands at 70.0/14.9/15.0 by song, and all 226 labels still have
songs in every partition.

Everything downstream is imported from the v1 builder rather than restated -- the line
frame, the pinyin normalisation and rhyme families, the adjacency rule, the models and their
selection protocol, the abstention machinery, the seeds. The two results therefore cannot
drift apart in their definitions.

    python src/build_written_rhyme_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_written_rhyme_v1 as wr  # noqa: E402
from leakage_groups_v2 import build_groups, group_audit, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)

ARTIFACT_ID = "chinese-rap-written-rhyme-v2"
VERSION = "2.0.0"
OUT_DIR = ROOT / "results" / "written-rhyme-v2"

CORPUS_CONTENT_SHA256 = "8102da32084496fac0fb51c7b78485dec9e216696ff98e21b922fa1708b4313b"
EXPECTED_CHUNKS = 25026
EXPECTED_SONGS = 7391
MINIMUM_SONGS_PER_LABEL = 5


def corpus_content_sha256(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        canonical = json.dumps(
            [str(row["source_credit_label"]), str(row["song_id"]), str(row["song_title"]),
             int(row["chunk_id"]), int(row["source_order"]), str(row["cleaned_text"])],
            ensure_ascii=False, separators=(",", ":"))
        digest.update((canonical + "\n").encode("utf-8"))
    return digest.hexdigest()


def load_corpus_v2(private_root: Path):
    """The v1 loader's five-tuple, built from corpus v2.

    v1 joined a graph membership table to a clean-text sidecar on a four-part key and
    filtered on the graph's eligibility flags. None of that exists here: corpus v2 carries
    the text, the label and the component id on one row, and PD-002 retains every chunk, so
    there is nothing to filter on. The line frame itself -- what counts as a line, which
    endings are classifiable, how repetition is recorded -- is v1's, unchanged.
    """
    corpus_path = (private_root / "work" / "private-repaired-corpus-v2"
                   / "repaired_lyric_chunks_v2.csv")
    if not corpus_path.is_file():
        raise SystemExit(f"missing private input: {corpus_path}")
    rows = list(csv.DictReader(corpus_path.open(encoding="utf-8")))

    digest = corpus_content_sha256(rows)
    if digest != CORPUS_CONTENT_SHA256:
        raise SystemExit(f"corpus digest {digest} is not the published corpus v2")
    if len(rows) != EXPECTED_CHUNKS:
        raise SystemExit(f"expected {EXPECTED_CHUNKS} chunks, read {len(rows)}")

    songs_by_label: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        songs_by_label[row["source_credit_label"]].add(row["song_id"])
    eligible = {label for label, members in songs_by_label.items()
                if len(members) >= MINIMUM_SONGS_PER_LABEL}

    sequences: dict[tuple[str, str, str], list[dict]] = {}
    exclusions: Counter[str] = Counter()
    terminal_by_song: dict[str, Counter[str]] = defaultdict(Counter)
    song_to_labels: dict[str, set[str]] = defaultdict(set)
    components_by_song: dict[str, set[str]] = defaultdict(set)
    documents: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for row in rows:
        label = row["source_credit_label"]
        if label not in eligible:
            exclusions["label_below_minimum_songs"] += 1
            continue
        song_id, chunk_id = row["song_id"], row["chunk_id"]
        song_to_labels[song_id].add(label)
        components_by_song[song_id].add(row["text_component_id"])
        documents[song_id].append((int(row["source_order"]), row["cleaned_text"]))
        text = row["cleaned_text"]
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        sequence: list[dict] = []
        for line_index, raw_line in enumerate(text.split("\n")):
            line = wr.display_normalise(raw_line)
            if not line:
                exclusions["empty_line"] += 1
                continue
            if wr.is_header_line(line):
                exclusions["section_or_credit_header"] += 1
                continue
            if not wr.HAN_RE.search(line):
                exclusions["no_han_character"] += 1
                continue
            terminal_kind = wr.terminal_content_kind(line)
            if terminal_kind != "han":
                exclusions[f"code_switch_or_non_han_terminal::{terminal_kind}"] += 1
                terminal_by_song[song_id][terminal_kind] += 1
                continue
            duplicate_key = wr.duplicate_normalise(line)
            if not duplicate_key:
                exclusions["empty_after_normalisation"] += 1
                continue
            ending = wr.written_ending_features(line)
            if ending is None:
                exclusions["unclassified_written_ending"] += 1
                continue
            sequence.append({
                "artist_label_id": label,
                "song_id": song_id,
                "chunk_id": chunk_id,
                "chunk_line_index": line_index,
                "line_hash": wr.sha256_text(duplicate_key),
                "duplicate_key": duplicate_key,
                # under v2 this field carries the corpus chunk's own cleaned-text digest;
                # there is no separate analysis-text sidecar. Private provenance only -- the
                # public payload guard forbids this column.
                "analysis_text_sha256": text_hash,
                **ending,
            })
        if sequence:
            for index, item in enumerate(sequence):
                item["sequence_index"] = index
                item["sequence_length"] = len(sequence)
            sequences[(label, song_id, chunk_id)] = sequence
        else:
            exclusions["empty_label_song_chunk_after_filters"] += 1

    # v1's repetition bookkeeping, unchanged: repeats are kept so adjacency survives, and
    # their multiplicity is recorded for the duplicate-balanced sensitivity stratum.
    by_label_song: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (label, song_id, _), sequence in sequences.items():
        by_label_song[(label, song_id)].extend(sequence)
    for group in by_label_song.values():
        counts = Counter(item["duplicate_key"] for item in group)
        seen: set[str] = set()
        for item in sorted(group, key=lambda value: (wr.chunk_sort_key(value["chunk_id"]),
                                                     value["chunk_line_index"])):
            item["within_label_song_line_multiplicity"] = counts[item["duplicate_key"]]
            item["first_occurrence_within_label_song"] = item["duplicate_key"] not in seen
            item["within_label_song_repeat"] = item["duplicate_key"] in seen
            seen.add(item["duplicate_key"])
        exclusions["retained_repeat_line_occurrences"] += sum(c - 1 for c in counts.values())

    normalised = {
        song: normalise_document("".join(text for _, text in sorted(parts)))
        for song, parts in documents.items()
    }
    nodes = {label: {"artist_label_id": label} for label in sorted(eligible)}
    return (sequences, nodes, dict(song_to_labels), dict(exclusions),
            {song: dict(counter) for song, counter in terminal_by_song.items()},
            components_by_song, normalised)


def split_by_group(song_to_labels, components_by_song, normalised):
    """Assign whole leakage groups, then hand each song its group's partition.

    v1's assign_song_splits is reused verbatim. Only the unit changes: a group stands in for
    a song and carries the union of its members' labels, so the multilabel balancing, the
    deterministic ordering and the 70/15/15 targets all still apply -- to groups.
    """
    songs = sorted(song_to_labels)
    groups = build_groups(songs, components_by_song, normalised)
    labels_by_group: dict[str, set[str]] = defaultdict(set)
    for song in songs:
        labels_by_group[groups[song]] |= song_to_labels[song]
    group_split, group_audit_rows = wr.assign_song_splits(dict(labels_by_group))
    return {song: group_split[groups[song]] for song in songs}, groups, group_audit_rows


def build(private_root: Path, out_dir: Path) -> int:
    print("loading corpus v2", flush=True)
    (sequences, nodes, song_to_labels, exclusions, terminal_by_song,
     components_by_song, normalised) = load_corpus_v2(private_root)
    print(f"  {len(sequences):,} label/song/chunk sequences over "
          f"{len(song_to_labels):,} songs and {len(nodes)} labels", flush=True)

    print("splitting by leakage group, not by song", flush=True)
    song_split, groups, split_audit = split_by_group(
        song_to_labels, components_by_song, normalised)
    audit = group_audit(groups, {s: sorted(v)[0] for s, v in song_to_labels.items()},
                        components_by_song)

    straddling = defaultdict(set)
    for song, split in song_split.items():
        for component in components_by_song[song]:
            straddling[component].add(split)
    crossed = sum(1 for splits in straddling.values() if len(splits) > 1)
    counts = Counter(song_split.values())
    print(f"  {counts['train']}/{counts['validation']}/{counts['test']} songs, "
          f"{audit['groups']:,} groups", flush=True)
    print(f"  text components crossing a partition: {crossed}", flush=True)
    if crossed:
        raise SystemExit("a text component crosses a partition; PD-002 rule 3 is violated")

    uncovered = [
        label for label in nodes
        if len({song_split[song] for song, labels in song_to_labels.items()
                if label in labels}) < 3
    ]
    if uncovered:
        raise SystemExit(f"{len(uncovered)} label(s) miss a partition: {uncovered[:5]}")

    print("building events and applying the line-level leakage filter", flush=True)
    events = wr.build_events(sequences, song_split)
    leakage_audit = wr.apply_leakage_filter(events, sequences, song_split)

    print("training", flush=True)
    model_audit, probabilities, classes, validation_events, test_events, _ = wr.train_models(events)
    print("evaluating", flush=True)
    metric_rows, per_label_rows, stratified_rows = wr.evaluate_models(
        probabilities, classes, test_events, events, sorted(nodes))
    paired = wr.paired_model_deltas(probabilities, classes, test_events)
    abstention_rows, abstention = wr.evaluate_abstention(
        probabilities, classes, validation_events, test_events, events)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "corpus": {
            "content_sha256": CORPUS_CONTENT_SHA256,
            "chunks": EXPECTED_CHUNKS,
            "song_records": EXPECTED_SONGS,
            "songs_in_eligible_labels": len(song_to_labels),
            "labels": len(nodes),
            "minimum_songs_per_label": MINIMUM_SONGS_PER_LABEL,
        },
        "split": {
            "unit": "leakage group (exact-text component union near-duplicate)",
            "songs": dict(counts),
            "groups": audit["groups"],
            "text_components_crossing_a_partition": crossed,
            "labels_missing_a_partition": len(uncovered),
            "why_not_songs": (
                "v1 assigned songs individually. Running that same function over corpus v2 "
                "puts 433 of 799 multi-song text components on both sides of a boundary, "
                "touching 1,330 song records -- 18.0% of the population. The algorithm is "
                "unchanged; only the unit it assigns is."
            ),
            "audit": split_audit,
        },
        "leakage_filter": leakage_audit,
        "models": metric_rows,
        "paired_deltas": paired,
        "abstention": abstention,
        "line_frame_exclusions": exclusions,
        "reused_from_v1_builder": [
            "display_normalise", "is_header_line", "terminal_content_kind",
            "written_ending_features", "duplicate_normalise", "assign_song_splits",
            "build_events", "apply_leakage_filter", "train_models", "evaluate_models",
            "paired_model_deltas", "evaluate_abstention",
        ],
        "replaces": (
            "results/written-rhyme-v1, whose predictive metrics were withheld pending exactly "
            "this rerun. The numbers are not comparable point to point: the population "
            "differs and the split unit differs."
        ),
        "privacy": "aggregate only; no lyric text, song or chunk identifiers, or line hashes",
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
