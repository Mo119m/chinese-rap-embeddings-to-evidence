#!/usr/bin/env python3
"""Build the PD-002 duplicate-aware repaired corpus (corpus v2).

PD-002 established that the legacy cleaner's fourth stage -- keep the first exact
``(source-credit label, cleaned chunk text)`` occurrence -- controlled copied text by
deleting source rows, erasing 2,894 chunks and 177 song records outright. This builder
implements the amendment's replacement rule: every non-empty cleaned source chunk is
preserved together with its ``(song ID, chunk ID, source order)`` relation, and duplicate
control is *represented* as group membership and weights rather than *enacted* by deletion.

Two output trees are written and they are not interchangeable:

* ``--private-out`` receives the repaired corpus itself. It carries lyric text, titles,
  source-credit labels, and identifiers, so it must live outside the public repository.
* ``--output-dir`` receives the public artifact: aggregate counts, input hashes, method
  text, and validation gates. It never carries text, titles, labels, identifiers, or
  row-level hashes.

The builder refuses to run unless the legacy stages still reconstruct the frozen snapshot
exactly, so corpus v2 is always bound to the lineage PD-002 verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_corpus_reconciliation_v1 import (
    reconstruct,
    sha256_file,
    sha256_text,
    title_normalise,
    write_csv,
    write_json,
)
from duplicate_control_v2 import (
    DEL,
    assign_duplicate_groups,
    assign_text_components,
    build_song_records,
    classify_review_queue,
    component_weights,
    corpus_content_sha256,
    normalise_title,
    stray_title_chunks,
    strip_del,
)

ARTIFACT_ID = "chinese-rap-repaired-corpus-v2"
VERSION = "2.0.0"
CORPUS_CONTRACT_VERSION = "chinese-rap-repaired-corpus-v2/2.0.0"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "repaired-corpus-v2"

# Stage counts PD-002 reconciled against the frozen snapshot. They are asserted here rather
# than read from any artifact, so a changed input cannot quietly redefine what "reproduced"
# means.
EXPECTED_RAW_ROWS = 26833
EXPECTED_RAW_SONGS = 7721
EXPECTED_AFTER_TITLE_ROWS = 25279
EXPECTED_AFTER_TITLE_SONGS = 7420
EXPECTED_AFTER_TEXT_ROWS = 25026
EXPECTED_AFTER_TEXT_SONGS = 7391
EXPECTED_LEGACY_ROWS = 22132
EXPECTED_LEGACY_SONGS = 7214
EXPECTED_LEGACY_LOST_SONGS = 177

# PD-002 recorded the high-confidence duplicate-record stratum and the residual review queue
# against the songs the legacy rule erased. Those published figures are the safeguard; this
# builder reproduces them and is not permitted to shrink the queue.
EXPECTED_HIGH_CONFIDENCE_ERASED = 131
EXPECTED_REVIEW_QUEUE = 46

METHOD = """# Duplicate-aware corpus repair method

Corpus v2 replaces one stage of the historical cleaner and nothing else. Title exclusions, configured line cleaning, and empty-chunk removal are replayed unchanged from `src/build_corpus_reconciliation_v1.py`, and the builder refuses to run unless those stages still reconstruct the frozen snapshot exactly. Only the fourth stage — keep the first exact `(source-credit label, cleaned chunk text)` occurrence — is withdrawn.

Every non-empty cleaned chunk is retained together with `song_id`, `chunk_id`, a global `source_order`, and a per-song `within_song_order`. Repetition inside a song is preserved, so hooks and refrains remain observable in sequence, and no song record disappears.

Duplicate control is represented rather than enacted. A song-record duplicate group requires the same source-credit label, the same normalized title, and the exact complete cleaned chunk sequence; the earliest record in source order is the representative and the others are flagged, not deleted. This rule is deliberately conservative and establishes only that two ingestion records carry identical cleaned content under one label and title. It does not establish real-world work identity, reissue status, authorship, or performer identity.

Cross-song text reuse is carried separately. Songs that share any exact cleaned chunk text are unioned into a text component; repetition inside one song is not a link. Components are the leakage unit: a component must stay together across train, validation, and test partitions, or be removed from every candidate profile. Each component carries total weight one inside a source-label aggregate, so repeated imports cannot dominate a label centroid, vocabulary probe, or cultural-reference rate.

The DEL control characters adjudicated in PD-002 are removed while their preceding characters are kept, which is the source semantics; the live sheet's destructive import behaviour is not copied.

Song records the legacy rule erased that no automatic rule resolves stay in a review queue for author adjudication. The queue is reproduced at the size PD-002 published. Where the v2 primary rule happens to group a queued record — which occurs when that record's exact-sequence twin was itself erased and so is retained only in v2 — the record remains in the queue and is reported as its own stratum, because a published safeguard is not narrowed by a later rebuild.

Two output trees are written. The private tree carries the repaired corpus itself, including lyric text, titles, source-credit labels, and identifiers, and is never published. The public artifact carries aggregate counts, input and software hashes, this method text, and validation results, and carries no text, titles, labels, identifiers, embeddings, or row-level hashes.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build(args: argparse.Namespace) -> None:
    raw_path = args.raw_chunks.resolve()
    frozen_path = args.frozen_snapshot.resolve()
    private_out = args.private_out.resolve()
    out = args.output_dir.resolve()
    for path in (raw_path, frozen_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if private_out == ROOT or private_out.is_relative_to(ROOT):
        raise SystemExit(
            f"--private-out must not resolve inside the public repository: {private_out}"
        )

    if args.generated_at_utc:
        generated_at_utc = args.generated_at_utc
    elif args.reuse_generated_at:
        existing = out / "analysis_summary.json"
        if not existing.is_file():
            raise FileNotFoundError(f"Cannot reuse a missing artifact timestamp: {existing}")
        generated_at_utc = json.loads(existing.read_text(encoding="utf-8"))["generated_at_utc"]
    else:
        generated_at_utc = utc_now()

    raw = pd.read_csv(raw_path)
    frozen = pd.read_csv(frozen_path)
    raw_ordered, after_title, after_text, after_dedup = reconstruct(raw)

    columns = ["artist", "song_id", "song_title", "chunk_id", "text"]
    frozen_exact = (
        after_dedup[columns].fillna("").astype(str).reset_index(drop=True).equals(
            frozen[columns].fillna("").astype(str).reset_index(drop=True)
        )
    )
    if not frozen_exact:
        raise SystemExit(
            "the legacy stages no longer reconstruct the frozen snapshot; corpus v2 must not "
            "be built from an unverified lineage"
        )

    ordered = after_text.sort_values("_source_row").reset_index(drop=True)
    # zip rather than itertuples: itertuples renames "_source_row" to a positional field
    chunk_rows = [
        {"song_id": song, "artist": artist, "song_title": title, "text": text, "source_row": source}
        for song, artist, title, text, source in zip(
            ordered["song_id"].astype(str),
            ordered["artist"].astype(str),
            ordered["song_title"].astype(str),
            ordered["text"].astype(str),
            ordered["_source_row"].astype(int),
        )
    ]

    records = build_song_records(chunk_rows)
    legacy_retained = set(after_dedup["song_id"].astype(str))
    group_of, is_representative, group_membership = assign_duplicate_groups(records)
    component_of, component_membership, shared_texts = assign_text_components(
        ((row["song_id"], row["text"]) for row in chunk_rows), records
    )
    weights = component_weights(records, component_of)
    queue, queue_diagnostics = classify_review_queue(
        records, legacy_retained, group_of, is_representative
    )

    within_song_order = ordered.groupby("song_id", sort=False).cumcount()
    del_rows = int(ordered["text"].astype(str).str.contains(DEL, regex=False).sum())
    del_characters = int(sum(str(value).count(DEL) for value in ordered["text"]))

    corpus = pd.DataFrame(
        {
            "repaired_corpus_id": [
                sha256_text(f"{CORPUS_CONTRACT_VERSION}\t{song}\t{chunk}")
                for song, chunk in zip(ordered["song_id"].astype(str), ordered["chunk_id"].astype(int))
            ],
            "repaired_corpus_contract_version": CORPUS_CONTRACT_VERSION,
            "song_id": ordered["song_id"].astype(str),
            "chunk_id": ordered["chunk_id"].astype(int),
            "source_order": ordered["_source_row"].astype(int),
            "within_song_order": within_song_order.astype(int),
            "source_credit_label": ordered["artist"].astype(str),
            "song_title": ordered["song_title"].astype(str),
            "cleaned_text": ordered["text"].astype(str).map(strip_del),
        }
    )
    corpus["song_duplicate_group_id"] = corpus["song_id"].map(lambda s: group_of.get(s, ""))
    corpus["is_group_representative"] = corpus["song_id"].map(
        lambda s: True if s not in group_of else is_representative[s]
    )
    corpus["text_component_id"] = corpus["song_id"].map(component_of)
    corpus["song_component_weight"] = corpus["song_id"].map(weights)
    corpus["retained_by_legacy_cleaner"] = corpus["song_id"].map(lambda s: s in legacy_retained)

    song_table = (
        corpus.groupby("song_id", sort=False)
        .agg(
            source_credit_label=("source_credit_label", "first"),
            song_title=("song_title", "first"),
            chunk_rows=("chunk_id", "size"),
            first_source_order=("source_order", "min"),
            song_duplicate_group_id=("song_duplicate_group_id", "first"),
            is_group_representative=("is_group_representative", "first"),
            text_component_id=("text_component_id", "first"),
            song_component_weight=("song_component_weight", "first"),
            retained_by_legacy_cleaner=("retained_by_legacy_cleaner", "first"),
        )
        .reset_index()
        .sort_values("first_source_order")
    )

    queue_table = pd.DataFrame(
        [
            {
                "song_id": item["song_id"],
                "reason": item["reason"],
                "chunk_rows": item["chunk_rows"],
                "related_retained_song_count": item["related_retained_song_count"],
                "related_song_ids": ";".join(item["related_song_ids"]),
                "auto_grouped_by_v2_primary_rule": item["auto_grouped_by_v2_primary_rule"],
            }
            for item in queue
        ]
    )

    # What the component structure looks like if the scrape artefact is not treated as text
    # reuse. Reported, not applied: a chunk that is only a neighbouring title unions two songs
    # exactly as a shared verse does, and the weighting rule then divides weight between songs
    # that have nothing in common but an export error.
    stray = set(stray_title_chunks(chunk_rows, records))
    without_stray = [row for index, row in enumerate(chunk_rows) if index not in stray]
    stray_records = build_song_records(without_stray)
    stray_component_of, stray_membership, _ = assign_text_components(
        ((row["song_id"], row["text"]) for row in without_stray), stray_records)
    stray_weights = component_weights(stray_records, stray_component_of)
    stray_multi = {k: v for k, v in stray_membership.items() if len(v) > 1}
    penalised = sorted(song for song, weight in weights.items()
                       if weight < 1 and stray_weights.get(song, 1.0) == 1.0)
    penalised_weights = sorted(weights[song] for song in penalised)

    multi_song_components = {k: v for k, v in component_membership.items() if len(v) > 1}
    component_sizes = Counter(len(v) for v in component_membership.values())
    group_sizes = Counter(len(v) for v in group_membership.values())
    non_representative = sum(len(v) - 1 for v in group_membership.values())

    geometry = {
        "chunk_rows": len(corpus),
        "song_records": int(corpus["song_id"].nunique()),
        "chunk_rows_restored_versus_legacy": len(after_text) - len(after_dedup),
        "song_records_restored_versus_legacy": len(records) - len(legacy_retained),
        "duplicate_groups": len(group_membership),
        "song_records_in_duplicate_groups": sum(len(v) for v in group_membership.values()),
        "non_representative_duplicate_records": non_representative,
        "duplicate_group_size_histogram": {str(k): v for k, v in sorted(group_sizes.items())},
        "cross_song_shared_exact_texts": shared_texts,
        "text_components": len(component_membership),
        "single_song_components": component_sizes[1],
        "multi_song_components": len(multi_song_components),
        "song_records_in_multi_song_components": sum(len(v) for v in multi_song_components.values()),
        "largest_text_component_songs": max(len(v) for v in component_membership.values()),
        "del_control_rows_corrected": del_rows,
        "del_control_characters_removed": del_characters,
    }

    weight_by_label: dict[str, float] = defaultdict(float)
    for song_id, weight in weights.items():
        weight_by_label[records[song_id]["label"]] += weight
    components_by_label: dict[str, set[str]] = defaultdict(set)
    for song_id, component_id in component_of.items():
        components_by_label[records[song_id]["label"]].add(component_id)
    weights_sum_to_component_counts = all(
        abs(weight_by_label[label] - len(components_by_label[label])) < 1e-9
        for label in weight_by_label
    )

    title_rule_agrees = all(
        normalise_title(record["title"]) == title_normalise(record["title"])
        for record in records.values()
    )

    checks = [
        {
            "name": "duplicate_control_title_rule_agrees_with_reconciliation_v1",
            "passed": title_rule_agrees,
        },
        {
            "name": "legacy_stages_reproduce_pd002_counts",
            "passed": (
                len(raw_ordered) == EXPECTED_RAW_ROWS
                and raw_ordered["song_id"].nunique() == EXPECTED_RAW_SONGS
                and len(after_title) == EXPECTED_AFTER_TITLE_ROWS
                and after_title["song_id"].nunique() == EXPECTED_AFTER_TITLE_SONGS
                and len(after_text) == EXPECTED_AFTER_TEXT_ROWS
                and after_text["song_id"].nunique() == EXPECTED_AFTER_TEXT_SONGS
                and len(after_dedup) == EXPECTED_LEGACY_ROWS
                and after_dedup["song_id"].nunique() == EXPECTED_LEGACY_SONGS
            ),
        },
        {"name": "frozen_snapshot_content_exactly_reconstructed", "passed": frozen_exact},
        {
            "name": "no_chunk_row_deleted_by_duplicate_control",
            "passed": len(corpus) == EXPECTED_AFTER_TEXT_ROWS
            and int(corpus["song_id"].nunique()) == EXPECTED_AFTER_TEXT_SONGS,
        },
        {
            "name": "source_order_and_song_chunk_relation_preserved",
            "passed": bool(
                corpus["source_order"].is_monotonic_increasing
                and not corpus.duplicated(["song_id", "chunk_id"]).any()
                and corpus.groupby("song_id", sort=False)["within_song_order"]
                .apply(lambda s: list(s) == list(range(len(s))))
                .all()
            ),
        },
        {
            "name": "within_song_repetition_preserved",
            "passed": bool(
                sum(
                    len(record["sequence"]) - len(set(record["sequence"]))
                    for record in records.values()
                )
                == int(corpus.duplicated(["song_id", "cleaned_text"]).sum())
            ),
        },
        {
            "name": "pd002_published_erasure_and_queue_reproduced",
            "passed": (
                queue_diagnostics["legacy_erased_song_records"] == EXPECTED_LEGACY_LOST_SONGS
                and queue_diagnostics["high_confidence_duplicate_records"]
                == EXPECTED_HIGH_CONFIDENCE_ERASED
                and queue_diagnostics["manual_review_queue_records"] == EXPECTED_REVIEW_QUEUE
            ),
        },
        {
            "name": "review_queue_not_narrowed_by_automatic_grouping",
            "passed": len(queue_table) == EXPECTED_REVIEW_QUEUE
            and "unreconciled_other" not in set(queue_table["reason"]),
        },
        {
            "name": "every_song_has_exactly_one_component_and_positive_weight",
            "passed": bool(
                song_table["text_component_id"].ne("").all()
                and (song_table["song_component_weight"] > 0).all()
                and (song_table["song_component_weight"] <= 1).all()
            ),
        },
        {
            "name": "component_weights_total_one_per_label_component",
            "passed": weights_sum_to_component_counts,
        },
        {
            "name": "duplicate_groups_have_exactly_one_representative",
            "passed": all(
                sum(1 for song_id in members if is_representative[song_id]) == 1
                for members in group_membership.values()
            ),
        },
        {
            "name": "stray_title_sensitivity_is_published_both_ways",
            "passed": len(stray) > 0 and len(stray_membership) != len(component_membership),
        },
        {
            "name": "duplicate_group_members_share_one_text_component",
            "passed": all(
                len({component_of[song_id] for song_id in members}) == 1
                for members in group_membership.values()
            ),
        },
    ]
    status = "pass_pending_author_review" if all(item["passed"] for item in checks) else "fail"

    software_paths = [
        Path(__file__).resolve(),
        ROOT / "src" / "build_corpus_reconciliation_v1.py",
        ROOT / "src" / "build_canonical_lyric_text_sidecar_v1.py",
        ROOT / "src" / "build_chinese_rap_written_rhyme_v1.py",
    ]

    private_out.mkdir(parents=True, exist_ok=True)
    write_csv(private_out / "repaired_lyric_chunks_v2.csv", corpus)
    write_csv(private_out / "repaired_song_records_v2.csv", song_table)
    write_csv(private_out / "duplicate_review_queue_v2.csv", queue_table)
    content_sha = corpus_content_sha256(corpus.to_dict("records"))
    write_json(
        private_out / "private_manifest.json",
        {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": generated_at_utc,
            "warning": "contains lyric text, titles, source-credit labels, and identifiers; never publish",
            "repaired_corpus_content_sha256": content_sha,
            "files": {
                name: {
                    "sha256": sha256_file(private_out / name),
                    "bytes": (private_out / name).stat().st_size,
                }
                for name in (
                    "repaired_lyric_chunks_v2.csv",
                    "repaired_song_records_v2.csv",
                    "duplicate_review_queue_v2.csv",
                )
            },
        },
    )

    summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": generated_at_utc,
        "status": status,
        "privacy": (
            "aggregate only; no lyric text, titles, source-credit labels, song/chunk "
            "identifiers, embeddings, or row-level hashes"
        ),
        "input_fingerprints": {
            "raw_chunks": {"file": raw_path.name, "sha256": sha256_file(raw_path)},
            "frozen_snapshot": {"file": frozen_path.name, "sha256": sha256_file(frozen_path)},
        },
        "software_fingerprints": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
            for path in software_paths
        ],
        "repaired_corpus_content_sha256": content_sha,
        "replacement_rule": {
            "preservation": "every non-empty cleaned source chunk retained with its (song ID, chunk ID, source order) relation",
            "primary_duplicate_stratum": "same source-credit label + same normalized title + exact complete cleaned chunk sequence",
            "duplicate_control": "represented as group membership and component weights; no chunk is deleted",
            "component_rule": "songs sharing any exact cleaned chunk text form one component; components must not cross evaluation partitions",
            "weighting_rule": "each component carries total weight one inside a source-label aggregate",
            "del_control_correction": "DEL controls removed while preserving the preceding character, unlike the destructive live-sheet import",
        },
        "corpus_geometry": geometry,
        "stray_title_sensitivity": {
            "what_it_is": (
                "a chunk whose whole content is one line, and that line is a different song's "
                "title under the same source-credit label -- a scrape bleeding a neighbouring "
                "title into a record, not text the two songs share"
            ),
            "chunks_flagged": len(stray),
            "as_published": {
                "text_components": len(component_membership),
                "multi_song_components": len(multi_song_components),
                "song_records_in_multi_song_components": sum(
                    len(v) for v in multi_song_components.values()),
                "largest_text_component_songs": max(
                    len(v) for v in component_membership.values()),
            },
            "if_stray_titles_were_not_treated_as_shared_text": {
                "text_components": len(stray_membership),
                "multi_song_components": len(stray_multi),
                "song_records_in_multi_song_components": sum(
                    len(v) for v in stray_multi.values()),
                "largest_text_component_songs": max(
                    len(v) for v in stray_membership.values()),
            },
            "songs_down_weighted_only_by_a_stray_title": len(penalised),
            "their_weights": ({"min": penalised_weights[0],
                               "median": penalised_weights[len(penalised_weights) // 2],
                               "max": penalised_weights[-1]} if penalised else None),
            "status": "reported, not applied",
            "why_not_applied": (
                "the detection rule is a heuristic written after inspecting the review queue, "
                "it has not been adjudicated, and the corpus is not altered on the strength of "
                "one. Whether these chunks should be excluded from component linking is a "
                "cleaning-policy decision, and it belongs with the author alongside the "
                "duplicate review queue"
            ),
            "consequence_if_ignored": (
                "the leakage unit is coarser than the text warrants, and the component "
                "weighting penalises genuinely distinct songs for carrying an export artefact"
            ),
        },
        "review_queue": queue_diagnostics,
        "independent_human_review_status": "pending",
        "permitted_claim": (
            "corpus v2 preserves song identity, within-song repetition, and source order, and "
            "represents duplicate structure without deleting source rows"
        ),
        "withheld_claim": (
            "that the duplicate groups establish real-world work identity, reissue status, "
            "authorship, or performer identity; that any downstream predictive metric is "
            "stable under the repair, since no model has been retrained on this population; "
            "and that the published component structure is entirely text reuse -- see "
            "stray_title_sensitivity, part of it is a scrape artefact"
        ),
        "checks": checks,
    }

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "analysis_summary.json", summary)
    write_csv(
        out / "corpus_stage_counts.csv",
        pd.DataFrame(
            [
                {"stage": "raw lyric chunks", "rows": len(raw_ordered), "songs": raw_ordered["song_id"].nunique()},
                {"stage": "after title exclusions", "rows": len(after_title), "songs": after_title["song_id"].nunique()},
                {"stage": "after line cleaning and empty removal", "rows": len(after_text), "songs": after_text["song_id"].nunique()},
                {"stage": "legacy keep-first (superseded)", "rows": len(after_dedup), "songs": after_dedup["song_id"].nunique()},
                {"stage": "repaired corpus v2", "rows": len(corpus), "songs": int(corpus["song_id"].nunique())},
            ]
        ),
    )
    write_csv(
        out / "duplicate_group_size_distribution.csv",
        pd.DataFrame(
            [{"member_song_records": size, "groups": count} for size, count in sorted(group_sizes.items())]
        ),
    )
    write_csv(
        out / "text_component_size_distribution.csv",
        pd.DataFrame(
            [{"member_song_records": size, "components": count} for size, count in sorted(component_sizes.items())]
        ),
    )
    write_csv(
        out / "review_queue_by_reason.csv",
        pd.DataFrame(
            [
                {"reason": reason, "song_records": count}
                for reason, count in sorted(queue_diagnostics["queue_by_reason"].items())
            ]
        ),
    )
    write_json(
        out / "validation.json",
        {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": generated_at_utc,
            "status": status,
            "checks": checks,
            "privacy": summary["privacy"],
        },
    )
    (out / "METHOD.md").write_text(METHOD, encoding="utf-8", newline="")
    readme = f"""# Repaired corpus v2

Corpus v2 implements the PD-002 replacement rule. All **{geometry['chunk_rows']:,} cleaned chunk rows** across **{geometry['song_records']:,} song records** are retained with their original `(song ID, chunk ID, source order)` relation, restoring the **{geometry['chunk_rows_restored_versus_legacy']:,} chunks** and **{geometry['song_records_restored_versus_legacy']:,} song records** the legacy keep-first rule deleted. No chunk is removed for being a duplicate.

Duplicate structure is represented instead. The primary automatic stratum — same source-credit label, same normalized title, exact complete cleaned chunk sequence — forms **{geometry['duplicate_groups']:,} groups** over **{geometry['song_records_in_duplicate_groups']:,} song records**, of which **{geometry['non_representative_duplicate_records']:,}** are flagged as non-representative rather than deleted. Songs sharing any exact cleaned chunk text form **{geometry['text_components']:,} components**; **{geometry['multi_song_components']:,}** span more than one song and cover **{geometry['song_records_in_multi_song_components']:,} song records**, the largest reaching **{geometry['largest_text_component_songs']:,}**. Each component carries total weight one inside a source-label aggregate.

The PD-002 review queue is reproduced and not narrowed: of the **{queue_diagnostics['legacy_erased_song_records']:,}** song records the legacy rule erased, **{queue_diagnostics['high_confidence_duplicate_records']:,}** meet the high-confidence duplicate-record rule and **{queue_diagnostics['manual_review_queue_records']:,}** remain queued for author adjudication. **{queue_diagnostics['queue_records_also_grouped_by_v2_primary_rule']:,}** of those queued records are additionally grouped by the v2 primary rule, because their exact-sequence twin was itself erased; they stay in the queue and are reported as their own stratum.

Status: **{status}**. Structure, preservation, and duplicate representation pass their gates. The duplicate groups do not establish work identity, reissue status, authorship, or performer identity, and no downstream predictive metric has been rerun on this population.
"""
    (out / "README.md").write_text(readme, encoding="utf-8", newline="")

    software_fingerprints = summary["software_fingerprints"]
    output_files = sorted(path for path in out.iterdir() if path.name != "manifest.json")
    write_json(
        out / "manifest.json",
        {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": generated_at_utc,
            "status": status,
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in output_files
            ],
            "software": software_fingerprints,
        },
    )

    if not all(item["passed"] for item in checks):
        failed = [item["name"] for item in checks if not item["passed"]]
        raise RuntimeError(f"Repaired corpus v2 failed: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-chunks", type=Path, required=True)
    parser.add_argument("--frozen-snapshot", type=Path, required=True)
    parser.add_argument(
        "--private-out",
        type=Path,
        required=True,
        help="Directory outside this repository for the repaired corpus itself",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    timestamp = parser.add_mutually_exclusive_group()
    timestamp.add_argument("--generated-at-utc")
    timestamp.add_argument("--reuse-generated-at", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
