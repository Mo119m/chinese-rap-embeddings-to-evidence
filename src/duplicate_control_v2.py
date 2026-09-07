"""Duplicate control for the PD-002 repaired corpus, as pure functions.

The rules that decide what counts as a duplicate record, what shares text with what, and
what a human still has to rule on are the part of corpus v2 that can be quietly wrong. They
live here, apart from the builder, so they can be exercised against hand-checked fixtures
with nothing installed: no pandas, no numpy, no test framework.

``src/build_repaired_corpus_v2.py`` is the only caller. It keeps pandas for reading CSVs and
assembling frames and delegates every decision to this module.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

# written as a code point so the control never has to survive an editor or a diff viewer
DEL = chr(0x7F)


def normalise_title(value: object) -> str:
    """NFKC, case-fold, keep alphanumerics.

    ``src/build_corpus_reconciliation_v1.py`` carries its own copy of this rule, and that
    file is fingerprinted in a published manifest, so it is not edited to import this one.
    The v2 builder gates the two against each other on every title in the corpus instead, so
    a drift shows up as a failed check rather than as a silently different grouping.
    """
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())


def strip_del(value: str) -> str:
    """Remove DEL controls while preserving the characters around them.

    The live Google Sheet import deletes both the DEL and the character before it. PD-002
    adjudicated that as a destructive import side effect of the sheet, not of the source, so
    corpus v2 drops only the control itself.
    """
    return value.replace(DEL, "")


def build_song_records(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fold cleaned chunk rows into one record per song, in source order.

    Each row needs ``song_id``, ``artist``, ``song_title``, ``text``, and ``source_row``.
    """
    ordered: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        ordered[str(row["song_id"])].append(row)

    records: dict[str, dict[str, Any]] = {}
    for song_id, group in ordered.items():
        group = sorted(group, key=lambda row: int(row["source_row"]))
        sequence = tuple(str(row["text"]) for row in group)
        records[song_id] = {
            "label": str(group[0]["artist"]),
            "title": str(group[0]["song_title"]),
            "sequence": sequence,
            "counter": Counter(sequence),
            "chunk_rows": len(sequence),
            "first_source_row": int(group[0]["source_row"]),
        }
    return records


def assign_duplicate_groups(
    records: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, bool], dict[str, list[str]]]:
    """Apply the amendment's primary automatic stratum.

    A group requires the same source-credit label, the same normalized title, and the exact
    complete cleaned chunk sequence. The representative is the earliest record in source
    order; every other member stays in the corpus, flagged rather than deleted.
    """
    buckets: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
    for song_id, record in records.items():
        key = (record["label"], normalise_title(record["title"]), record["sequence"])
        buckets[key].append(song_id)

    ordered_groups = sorted(
        (members for members in buckets.values() if len(members) > 1),
        key=lambda members: min(records[song_id]["first_source_row"] for song_id in members),
    )
    group_of: dict[str, str] = {}
    is_representative: dict[str, bool] = {}
    membership: dict[str, list[str]] = {}
    for index, members in enumerate(ordered_groups, start=1):
        group_id = f"SDG-{index:05d}"
        by_order = sorted(members, key=lambda song_id: (records[song_id]["first_source_row"], song_id))
        membership[group_id] = by_order
        for position, song_id in enumerate(by_order):
            group_of[song_id] = group_id
            is_representative[song_id] = position == 0
    return group_of, is_representative, membership


def assign_text_components(
    pairs: Iterable[tuple[str, str]], records: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, list[str]], int]:
    """Union songs that share any exact cleaned chunk text.

    ``pairs`` is ``(song_id, cleaned_text)`` for every retained chunk row. Components are the
    leakage unit: rule 3 of the amendment requires component-linked records to stay together
    across train/validation/test splits, or the whole component to be removed from every
    candidate profile. Repetition *inside* one song is not a link, so a hook repeated in one
    song does not merge that song with itself.
    """
    parent = {song_id: song_id for song_id in records}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    songs_by_text: dict[str, set[str]] = defaultdict(set)
    for song_id, text in pairs:
        songs_by_text[text].add(str(song_id))

    shared_texts = 0
    for song_ids in songs_by_text.values():
        if len(song_ids) < 2:
            continue
        shared_texts += 1
        iterator = iter(sorted(song_ids))
        anchor = next(iterator)
        for other in iterator:
            union(anchor, other)

    grouped: dict[str, list[str]] = defaultdict(list)
    for song_id in records:
        grouped[find(song_id)].append(song_id)

    ordered = sorted(
        grouped.values(),
        key=lambda members: min(records[song_id]["first_source_row"] for song_id in members),
    )
    component_of: dict[str, str] = {}
    membership: dict[str, list[str]] = {}
    for index, members in enumerate(ordered, start=1):
        component_id = f"CMP-{index:06d}"
        by_order = sorted(members, key=lambda song_id: (records[song_id]["first_source_row"], song_id))
        membership[component_id] = by_order
        for song_id in by_order:
            component_of[song_id] = component_id
    return component_of, membership, shared_texts


def stray_title_chunks(rows, records):
    """Chunks that are one line, and that line is another same-label song's title.

    A scrape that appends a neighbouring title to a record produces these. They are not text
    reuse, but ``assign_text_components`` cannot tell the difference: two songs sharing such
    a line are unioned exactly as if they shared a verse, and the weighting rule then divides
    weight between songs that have nothing in common but an export artefact.

    This identifies the shape so it can be *reported*. It does not remove anything. The rule
    is a heuristic, it has not been adjudicated, and a corpus is not quietly altered on the
    strength of one.
    """
    titles = defaultdict(set)
    for song_id, record in records.items():
        titles[record["label"]].add(normalise_title(record["title"]))
    flagged = []
    for index, row in enumerate(rows):
        song_id = str(row["song_id"])
        record = records.get(song_id)
        if record is None:
            continue
        lines = [line for line in str(row["text"]).split(chr(10)) if line.strip()]
        if len(lines) != 1:
            continue
        normalised = normalise_title(lines[0])
        others = titles[record["label"]] - {normalise_title(record["title"])}
        if len(normalised) >= 6 and normalised in others:
            flagged.append(index)
    return flagged


def stray_title_lines(rows, records):
    """Every line that is another same-label song's title, wherever it sits.

    ``stray_title_chunks`` counts only the case where such a line is the whole chunk, because
    that is the case that unions two songs into one text component. The shape is wider than
    that: the same bled title can sit inside a multi-line chunk, where it does not affect the
    component structure but is still not this song's text.

    Reported separately so the two counts cannot be confused. Returns (row index, line index)
    pairs, so a caller can count occurrences and the song records carrying them.
    """
    titles = defaultdict(set)
    for song_id, record in records.items():
        titles[record["label"]].add(normalise_title(record["title"]))
    flagged = []
    for index, row in enumerate(rows):
        record = records.get(str(row["song_id"]))
        if record is None:
            continue
        others = titles[record["label"]] - {normalise_title(record["title"])}
        for line_index, line in enumerate(str(row["text"]).split(chr(10))):
            if not line.strip():
                continue
            normalised = normalise_title(line)
            if len(normalised) >= 6 and normalised in others:
                flagged.append((index, line_index))
    return flagged


def component_weights(
    records: Mapping[str, Mapping[str, Any]], component_of: Mapping[str, str]
) -> dict[str, float]:
    """Rule 4: each component carries total weight one inside a source-label aggregate.

    A component spanning several labels contributes weight one to each label it appears in;
    the rule bounds what repeated imports can do to a single label's centroid, it does not
    redistribute a component across labels.
    """
    sizes: Counter[tuple[str, str]] = Counter(
        (records[song_id]["label"], component_id) for song_id, component_id in component_of.items()
    )
    return {
        song_id: 1.0 / sizes[(records[song_id]["label"], component_id)]
        for song_id, component_id in component_of.items()
    }


def classify_review_queue(
    records: Mapping[str, Mapping[str, Any]],
    legacy_retained: set[str],
    group_of: Mapping[str, str],
    is_representative: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild PD-002's review queue over the songs the legacy rule erased.

    The queue is defined exactly as PD-002 published it: erased song records that are not
    exact-sequence duplicates of a *retained* song under the same normalized title. Some of
    those records are nevertheless grouped by the v2 primary rule, because their
    exact-sequence twin was itself erased and so survives only in v2. Those are reported as
    their own stratum instead of being dropped from the queue -- a published safeguard is not
    narrowed by a later rebuild.
    """
    lost = sorted(set(records) - legacy_retained)
    retained_by_label: dict[str, list[str]] = defaultdict(list)
    for song_id in sorted(legacy_retained):
        retained_by_label[records[song_id]["label"]].append(song_id)

    queue: list[dict[str, Any]] = []
    high_confidence = 0
    for song_id in lost:
        record = records[song_id]
        candidates = retained_by_label[record["label"]]
        exact = [other for other in candidates if records[other]["sequence"] == record["sequence"]]
        same_multiset = [
            other for other in candidates
            if Counter(records[other]["sequence"]) == record["counter"]
        ]
        containers = [
            other for other in candidates
            if not (record["counter"] - records[other]["counter"])
        ]
        owners = sorted(
            {
                other for other in candidates
                for chunk in record["counter"]
                if chunk in records[other]["counter"]
            }
        )
        related = exact or same_multiset or containers or owners
        title_match = any(
            normalise_title(records[other]["title"]) == normalise_title(record["title"])
            for other in related
        )
        if exact and title_match:
            high_confidence += 1
            continue
        if exact:
            reason = "exact_cleaned_sequence_but_different_normalized_title"
        elif same_multiset:
            reason = "same_chunk_multiset_different_order"
        elif containers:
            reason = "chunk_multiset_subset_of_one_retained_song"
        elif owners:
            reason = "chunks_distributed_across_multiple_retained_songs"
        else:
            reason = "unreconciled_other"
        queue.append(
            {
                "song_id": song_id,
                "reason": reason,
                "chunk_rows": record["chunk_rows"],
                "related_retained_song_count": len(set(related)),
                "related_song_ids": sorted(set(related)),
                "auto_grouped_by_v2_primary_rule": (
                    song_id in group_of and not is_representative[song_id]
                ),
            }
        )

    tally = Counter(item["reason"] for item in queue)
    diagnostics = {
        "legacy_erased_song_records": len(lost),
        "high_confidence_duplicate_records": high_confidence,
        "manual_review_queue_records": len(queue),
        "queue_records_also_grouped_by_v2_primary_rule": sum(
            1 for item in queue if item["auto_grouped_by_v2_primary_rule"]
        ),
        "queue_by_reason": dict(sorted(tally.items())),
    }
    return queue, diagnostics


CONTENT_FIELDS: Sequence[str] = (
    "source_credit_label",
    "song_id",
    "song_title",
    "chunk_id",
    "source_order",
    "cleaned_text",
)


def corpus_content_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Content digest over the ordered repaired corpus, independent of CSV formatting."""
    digest = hashlib.sha256()
    for row in rows:
        canonical = json.dumps(
            [
                str(row["source_credit_label"]),
                str(row["song_id"]),
                str(row["song_title"]),
                int(row["chunk_id"]),
                int(row["source_order"]),
                str(row["cleaned_text"]),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update((canonical + "\n").encode("utf-8"))
    return digest.hexdigest()
