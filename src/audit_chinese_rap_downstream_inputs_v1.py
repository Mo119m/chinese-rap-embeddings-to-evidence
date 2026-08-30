#!/usr/bin/env python3
"""Audit the frozen inputs used by the three Chinese-rap downstream tasks.

The public artifact is aggregate only. It never exports lyric text, titles,
source-credit labels, song/chunk identifiers, embeddings, or membership rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


VERSION = "1.0.0"
ARTIFACT_ID = "chinese-rap-downstream-input-audit-v1"
ROOT = Path(__file__).resolve().parents[1]
SONGS = ROOT / "work" / "private-canonical-analysis-input-v1" / "canonical_analysis_songs_v1.csv"
CHUNKS = ROOT / "work" / "private-canonical-analysis-input-v1" / "canonical_analysis_chunks_v1.csv"
CLEAN = ROOT / "work" / "private-canonical-lyric-text-sidecar-v1" / "cleaned_analysis_chunks_v1.csv"
METADATA = ROOT / "work" / "private-canonical-song-metadata-quality-v1" / "canonical_song_metadata_quality_v1.csv"
OUT = ROOT / "outputs" / ARTIFACT_ID


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def check(name: str, passed: bool, value, expected, risk: str) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "expected": expected,
        "downstream_risk_if_failed": risk,
    }


def build() -> None:
    for path in (SONGS, CHUNKS, CLEAN, METADATA):
        if not path.is_file():
            raise FileNotFoundError(path)

    song_ids: set[str] = set()
    song_labels: Counter[str] = Counter()
    song_content_hashes: Counter[str] = Counter()
    identity_status: Counter[str] = Counter()
    artist_title_eligible: Counter[str] = Counter()
    empty_artist = 0
    empty_title = 0
    duplicate_song_ids = 0
    song_rows = 0
    for row in read_rows(SONGS):
        song_rows += 1
        song_id = row["song_id"]
        duplicate_song_ids += int(song_id in song_ids)
        song_ids.add(song_id)
        artist = row["canonical_artist"].strip()
        title = row["canonical_song_title"].strip()
        empty_artist += int(not artist)
        empty_title += int(not title)
        song_labels[artist] += 1
        identity_status[row["identity_validation_status"]] += 1
        artist_title_eligible[row["artist_title_comparison_eligible"].lower()] += 1
        content_hash = row["song_lyric_content_sha256"].strip()
        if content_hash:
            song_content_hashes[content_hash] += 1

    chunk_keys: set[tuple[str, str]] = set()
    chunk_song_ids: set[str] = set()
    duplicate_chunk_keys = 0
    orphan_chunk_rows = 0
    chunk_rows = 0
    for row in read_rows(CHUNKS):
        chunk_rows += 1
        key = (row["song_id"], row["chunk_id"])
        duplicate_chunk_keys += int(key in chunk_keys)
        chunk_keys.add(key)
        chunk_song_ids.add(row["song_id"])
        orphan_chunk_rows += int(row["song_id"] not in song_ids)

    clean_keys: set[tuple[str, str]] = set()
    clean_status: Counter[str] = Counter()
    duplicate_clean_keys = 0
    blank_eligible_text = 0
    invalid_clean_hash = 0
    clean_rows = 0
    for row in read_rows(CLEAN):
        clean_rows += 1
        key = (row["song_id"], row["chunk_id"])
        duplicate_clean_keys += int(key in clean_keys)
        clean_keys.add(key)
        status = row["analysis_text_status"]
        clean_status[status] += 1
        if status == "eligible_clean_text":
            blank_eligible_text += int(not row["analysis_text"].strip())
            clean_hash = row["analysis_text_sha256"].strip().lower()
            invalid_clean_hash += int(len(clean_hash) != 64 or any(c not in "0123456789abcdef" for c in clean_hash))

    metadata_rows = 0
    metadata_song_ids: set[str] = set()
    title_semantic_ineligible = 0
    for row in read_rows(METADATA):
        metadata_rows += 1
        metadata_song_ids.add(row["song_id"])
        title_semantic_ineligible += int(row["title_semantic_eligible"].lower() != "true")

    labels_with_five_songs = sum(count >= 5 for count in song_labels.values())
    cross_song_exact_content_groups = sum(count > 1 for count in song_content_hashes.values())
    songs_in_cross_song_exact_content_groups = sum(count for count in song_content_hashes.values() if count > 1)
    checks = [
        check("song_primary_key_unique", duplicate_song_ids == 0, duplicate_song_ids, 0, "Mixed song grain or train/test leakage."),
        check("chunk_composite_key_unique", duplicate_chunk_keys == 0, duplicate_chunk_keys, 0, "Duplicated lyric evidence and inflated metrics."),
        check("clean_sidecar_composite_key_unique", duplicate_clean_keys == 0, duplicate_clean_keys, 0, "Ambiguous cleaned text joins."),
        check("all_chunks_join_to_song", orphan_chunk_rows == 0, orphan_chunk_rows, 0, "Orphan text rows and incorrect source-label attribution."),
        check("clean_sidecar_keyset_matches_chunks", clean_keys == chunk_keys, {"missing": len(chunk_keys-clean_keys), "extra": len(clean_keys-chunk_keys)}, {"missing": 0, "extra": 0}, "Training/evaluation populations would silently diverge."),
        check("eligible_clean_text_nonempty", blank_eligible_text == 0, blank_eligible_text, 0, "Empty model inputs."),
        check("eligible_clean_text_hash_valid", invalid_clean_hash == 0, invalid_clean_hash, 0, "Broken duplicate grouping and leakage controls."),
        check("canonical_artist_nonempty", empty_artist == 0, empty_artist, 0, "Undefined retrieval/NER/rhyme profile labels."),
        check("canonical_title_nonempty", empty_title == 0, empty_title, 0, "Unusable title-level review records."),
        check("metadata_sidecar_covers_songs", metadata_song_ids == song_ids, {"missing": len(song_ids-metadata_song_ids), "extra": len(metadata_song_ids-song_ids)}, {"missing": 0, "extra": 0}, "Missing eligibility flags."),
        check("expected_frozen_song_count", song_rows == 7211, song_rows, 7211, "Input snapshot drift."),
        check("expected_frozen_chunk_count", chunk_rows == 22128 and clean_rows == 22128, {"canonical": chunk_rows, "clean": clean_rows}, {"canonical": 22128, "clean": 22128}, "Input snapshot drift."),
    ]

    summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "intended_use": [
            "explainable lyrical-repertoire retrieval",
            "provisional Chinese-rap NER and grounded cultural-network research",
            "dictionary-estimated written-rhyme modelling and recommendation",
        ],
        "grain": {"songs": song_rows, "chunks": chunk_rows, "clean_sidecar_rows": clean_rows, "metadata_rows": metadata_rows},
        "coverage": {
            "source_credit_labels": len(song_labels),
            "labels_with_at_least_five_songs": labels_with_five_songs,
            "eligible_clean_text_chunks": clean_status["eligible_clean_text"],
            "withheld_metadata_only_chunks": clean_status["withheld_metadata_only"],
            "artist_title_comparison_eligible_songs": artist_title_eligible["true"],
            "artist_title_comparison_ineligible_songs": artist_title_eligible["false"],
            "title_semantic_ineligible_songs": title_semantic_ineligible,
        },
        "identity_status": dict(sorted(identity_status.items())),
        "duplicate_risk": {
            "exact_song_content_groups_spanning_multiple_songs": cross_song_exact_content_groups,
            "songs_in_those_groups": songs_in_cross_song_exact_content_groups,
            "required_split_rule": "All rows sharing exact song-content or cleaned-text hashes stay in one split; split by song before line/chunk expansion.",
        },
        "claim_boundary": {
            "labels": "Corpus source-credit labels, not independently verified performer identities.",
            "titles": "Use artist/title comparisons only where artist_title_comparison_eligible is true; no guessed correction.",
            "rhyme": "Dictionary-estimated written endings, not performed rhyme, flow, voice, or beat.",
            "ner": "Model candidates and lyric co-mentions are not verified identities or real-world relations.",
        },
        "checks": checks,
    }

    stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}-", dir=OUT.parent))
    try:
        (stage / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        with (stage / "quality_checks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "passed", "value", "expected", "downstream_risk_if_failed"], lineterminator="\n")
            writer.writeheader()
            for item in checks:
                writer.writerow({**item, "value": json.dumps(item["value"], ensure_ascii=False), "expected": json.dumps(item["expected"], ensure_ascii=False)})
        method = """# Downstream input audit method

The audit freezes the exact song, chunk, cleaned-text, and metadata sidecars used by the three downstream tasks. It verifies grain, key uniqueness, join coverage, non-empty eligible text, hash validity, label/title completeness, metadata-flag coverage, and snapshot counts. Exact-content groups are reported so every task can keep duplicate-linked songs and texts in one split.

Passing this audit means the files satisfy the declared structural contract. It does not mean every source-credit label, title, lyric line, entity, or pronunciation has been manually or externally verified. The task builders must preserve the claim boundaries in `analysis_summary.json` and publish aggregate-only outputs.
"""
        (stage / "METHOD.md").write_text(method, encoding="utf-8", newline="\n")
        manifest_files = {}
        for path in (SONGS, CHUNKS, CLEAN, METADATA):
            manifest_files[str(path.relative_to(ROOT)).replace("\\", "/")] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        (stage / "input_manifest.json").write_text(json.dumps({"artifact_id": ARTIFACT_ID, "inputs": manifest_files}, indent=2) + "\n", encoding="utf-8", newline="\n")
        validation = {
            "artifact_id": ARTIFACT_ID,
            "generated_at_utc": utc_now(),
            "status": summary["status"],
            "checks": [{"name": item["name"], "passed": item["passed"]} for item in checks],
            "privacy": "aggregate only; no lyric text, labels, titles, identifiers, embeddings, or private rows",
        }
        (stage / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8", newline="\n")
        if OUT.exists():
            shutil.rmtree(OUT)
        stage.replace(OUT)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    print(json.dumps({"artifact": ARTIFACT_ID, "status": summary["status"], "checks": len(checks)}, indent=2))


if __name__ == "__main__":
    build()
