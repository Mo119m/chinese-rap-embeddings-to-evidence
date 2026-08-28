#!/usr/bin/env python3
"""Validate the byte integrity and basic usability of the public release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from normalize_public_text_v1 import is_text_path, normalized_bytes, tracked_paths


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS_DIR = ROOT / "results" / "repertoire-network-v1" / "robustness"
RETRIEVAL_SENSITIVITY_DIR = ROOT / "results" / "retrieval-inductive-sensitivity-v1"
CORPUS_RECONCILIATION_DIR = ROOT / "results" / "corpus-reconciliation-v1"
CORPUS_RECONCILIATION_FILES = {
    "analysis_summary.json",
    "lost_song_classification.csv",
    "manifest.json",
    "METHOD.md",
    "README.md",
    "stage_counts.csv",
    "task_aligned_family_distribution_sensitivity.csv",
    "task_aligned_written_rhyme_sensitivity.csv",
    "title_exclusion_counts.csv",
    "validation.json",
    "written_rhyme_sensitivity.csv",
}
CORPUS_RECONCILIATION_SOFTWARE = {
    "src/build_canonical_lyric_text_sidecar_v1.py",
    "src/build_chinese_rap_written_rhyme_v1.py",
    "src/build_corpus_reconciliation_v1.py",
}
MANUSCRIPT_DERIVATIVES = {
    "paper/Chinese_Rap_Evidence_Grounded_Manuscript.docx": "paper/manuscript.md",
    "paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf": "paper/manuscript.md",
    "paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx": "paper/manuscript.md",
    "paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf": "paper/manuscript.md",
    "paper/Chinese_Rap_Evidence_Grounded_Supplement.docx": "paper/supplementary_methods.md",
    "paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf": "paper/supplementary_methods.md",
    "submission/dsh/manuscript.docx": "paper/manuscript.md",
    "submission/dsh/manuscript_preview.pdf": "paper/manuscript.md",
    "submission/dsh/supplementary_methods.docx": "paper/supplementary_methods.md",
    "submission/dsh/supplementary_methods_preview.pdf": "paper/supplementary_methods.md",
}
CORE_REQUIRED_PATHS = {
    ".gitattributes",
    ".github/workflows/release-integrity.yml",
    ".python-version",
    "LICENSE",
    "LICENSE-CODE",
    "README.md",
    "requirements.txt",
    "paper/derivative_provenance.json",
    "src/build_chinese_rap_paper_docx_v1.py",
    "src/build_chinese_rap_release_v4.py",
    "src/build_ner_released_claim_audit_v1.py",
    "src/build_repertoire_robustness_inference_v1.py",
    "src/build_retrieval_inductive_sensitivity_v1.py",
    "src/build_canonical_lyric_text_sidecar_v1.py",
    "src/build_chinese_rap_written_rhyme_v1.py",
    "src/build_corpus_reconciliation_v1.py",
    "src/normalize_public_text_v1.py",
    "src/restore_committed_bytes_v1.py",
    "src/update_public_result_manifests_v1.py",
    "src/validate_public_release_integrity_v1.py",
    "tools/check_manuscript_derivatives.py",
    "results/repertoire-network-v1/robustness/README.md",
    "results/repertoire-network-v1/robustness/METHOD.md",
    "results/repertoire-network-v1/robustness/analysis_summary.json",
    "results/repertoire-network-v1/robustness/diagnostic_summary.csv",
    "results/repertoire-network-v1/robustness/manifest.json",
    "results/repertoire-network-v1/robustness/null_overlap_distribution.csv",
    "results/repertoire-network-v1/robustness/projection_fidelity.csv",
    "results/repertoire-network-v1/robustness/validation.json",
    "results/retrieval-inductive-sensitivity-v1/README.md",
    "results/retrieval-inductive-sensitivity-v1/METHOD.md",
    "results/retrieval-inductive-sensitivity-v1/analysis_summary.json",
    "results/retrieval-inductive-sensitivity-v1/fold_summary.csv",
    "results/retrieval-inductive-sensitivity-v1/manifest.json",
    "results/retrieval-inductive-sensitivity-v1/metrics.csv",
    "results/retrieval-inductive-sensitivity-v1/paired_deltas.csv",
    "results/retrieval-inductive-sensitivity-v1/validation.json",
    "methods/NER_RELEASED_CLAIM_AUDIT_PROTOCOL.md",
    "methods/PROTOCOL_AMENDMENT_PD002_UPSTREAM_CHUNK_DEDUPLICATION.md",
    "results/ner-v1/released_claim_audit_status.json",
} | {f"results/corpus-reconciliation-v1/{name}" for name in CORPUS_RECONCILIATION_FILES}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_within(path: Path, root: Path) -> None:
    resolved = path.resolve()
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        raise AssertionError(f"Manifest path escapes its release root: {path}")


def verify_record(path: Path, record: dict[str, Any], release_root: Path) -> None:
    assert_within(path, release_root)
    if not path.is_file():
        raise AssertionError(f"Manifest target is missing: {path}")
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise AssertionError(
            f"Byte-count mismatch for {path}: {path.stat().st_size} != {expected_bytes}"
        )
    expected_hash = record.get("sha256")
    if expected_hash and sha256(path) != expected_hash:
        raise AssertionError(f"SHA-256 mismatch for {path}")


def iter_records(section: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(section, list):
        for record in section:
            yield record["path"], record
    elif isinstance(section, dict):
        for relative, record in section.items():
            if isinstance(record, dict) and "sha256" in record:
                yield relative, record
    else:
        raise TypeError(f"Unsupported manifest record collection: {type(section).__name__}")


def verify_manifest(manifest_path: Path, key: str, base: Path, release_root: Path) -> int:
    manifest = read_json(manifest_path)
    count = 0
    for relative, record in iter_records(manifest[key]):
        verify_record(base / relative, record, release_root)
        count += 1
    return count


def verify_repository_manifests() -> int:
    specifications = [
        (ROOT / "validation" / "MANIFEST.json", "files", ROOT),
        (ROOT / "submission" / "dsh" / "MANIFEST.json", "files", ROOT / "submission" / "dsh"),
        (ROOT / "figures" / "manifest.json", "files", ROOT),
        (ROOT / "results" / "retrieval-v1" / "manifest.json", "files", ROOT / "results" / "retrieval-v1"),
        (ROOT / "results" / "retrieval-inductive-sensitivity-v1" / "manifest.json", "files", ROOT / "results" / "retrieval-inductive-sensitivity-v1"),
        (ROOT / "results" / "corpus-reconciliation-v1" / "manifest.json", "files", ROOT / "results" / "corpus-reconciliation-v1"),
        (ROOT / "results" / "ner-v1" / "manifest.json", "files", ROOT / "results" / "ner-v1"),
        (ROOT / "results" / "written-rhyme-v1" / "manifest.json", "output_files", ROOT / "results" / "written-rhyme-v1"),
        (ROOT / "results" / "repertoire-network-v1" / "graph" / "manifest.json", "output_files", ROOT / "results" / "repertoire-network-v1" / "graph"),
        (ROOT / "results" / "repertoire-network-v1" / "profiles" / "manifest.json", "files", ROOT / "results" / "repertoire-network-v1" / "profiles"),
        (ROOT / "results" / "repertoire-network-v1" / "bootstrap" / "manifest.json", "files", ROOT / "results" / "repertoire-network-v1" / "bootstrap"),
        (ROOT / "results" / "repertoire-network-v1" / "robustness" / "manifest.json", "output_files", ROOT / "results" / "repertoire-network-v1" / "robustness"),
    ]
    checked = sum(verify_manifest(path, key, base, ROOT) for path, key, base in specifications)

    parent = read_json(ROOT / "results" / "repertoire-network-v1" / "manifest.json")
    for name, component in parent["components"].items():
        directory = ROOT / "results" / "repertoire-network-v1" / name
        verify_record(directory / "manifest.json", {"sha256": component["manifest_sha256"]}, ROOT)
        verify_record(directory / "validation.json", {"sha256": component["validation_sha256"]}, ROOT)
        checked += 2
    verify_record(
        ROOT / "results" / "repertoire-network-v1" / parent["validation"]["file"],
        parent["validation"],
        ROOT,
    )
    checked += 1

    figure_manifest = read_json(ROOT / "figures" / "manifest.json")
    public_lineage = figure_manifest.get("lineage", {}).get("public_files")
    if not public_lineage:
        raise AssertionError("figures/manifest.json lacks verifiable lineage.public_files")
    for record in public_lineage:
        verify_record(ROOT / record["path"], record, ROOT)
        checked += 1
    return checked


def verify_core_manifest_contract() -> int:
    manifest_path = ROOT / "validation" / "MANIFEST.json"
    records = list(iter_records(read_json(manifest_path)["files"]))
    paths = [relative for relative, _ in records]
    if len(paths) != len(set(paths)):
        raise AssertionError("validation/MANIFEST.json contains duplicate paths")
    missing = sorted(CORE_REQUIRED_PATHS - set(paths))
    if missing:
        raise AssertionError(f"Core release manifest omits required release files: {missing}")
    return len(CORE_REQUIRED_PATHS)


def verify_manuscript_derivative_provenance() -> int:
    provenance = read_json(ROOT / "paper" / "derivative_provenance.json")
    sources = set(MANUSCRIPT_DERIVATIVES.values())
    if set(provenance.get("sources", {})) != sources:
        raise AssertionError("Manuscript derivative provenance has the wrong source set")
    if set(provenance.get("derivatives", {})) != set(MANUSCRIPT_DERIVATIVES):
        raise AssertionError("Manuscript derivative provenance has the wrong derivative set")

    checked = 0
    for source in sources:
        if provenance["sources"][source] != sha256(ROOT / source):
            raise AssertionError(f"Manuscript source is newer than its derivatives: {source}")
        checked += 1
    for derivative, source in MANUSCRIPT_DERIVATIVES.items():
        record = provenance["derivatives"][derivative]
        if record.get("built_from") != source or record.get("sha256") != sha256(ROOT / derivative):
            raise AssertionError(f"Manuscript derivative provenance mismatch: {derivative}")
        checked += 1
    return checked


def verify_corpus_reconciliation_claims() -> int:
    artifact_id = "chinese-rap-corpus-reconciliation-v1"
    expected_status = "pass_with_release_action"
    manifest = read_json(CORPUS_RECONCILIATION_DIR / "manifest.json")
    validation = read_json(CORPUS_RECONCILIATION_DIR / "validation.json")
    summary = read_json(CORPUS_RECONCILIATION_DIR / "analysis_summary.json")

    for name, payload in (("manifest", manifest), ("validation", validation), ("summary", summary)):
        if payload.get("artifact_id") != artifact_id:
            raise AssertionError(f"Corpus reconciliation {name} has the wrong artifact identifier")
        if payload.get("status") != expected_status:
            raise AssertionError(f"Corpus reconciliation {name} has the wrong release status")
    generated_at_values = {
        manifest.get("generated_at_utc"),
        validation.get("generated_at_utc"),
        summary.get("generated_at_utc"),
    }
    if None in generated_at_values or len(generated_at_values) != 1:
        raise AssertionError("Corpus reconciliation generated_at_utc values disagree")
    if manifest.get("version") != "1.0.0" or summary.get("version") != "1.0.0":
        raise AssertionError("Corpus reconciliation version is not 1.0.0")

    declared_records = list(iter_records(manifest.get("files")))
    declared_paths = [relative for relative, _ in declared_records]
    if len(declared_paths) != len(set(declared_paths)):
        raise AssertionError("Corpus reconciliation manifest contains duplicate file paths")
    expected_output_files = CORPUS_RECONCILIATION_FILES - {"manifest.json"}
    actual_output_files = {
        path.name
        for path in CORPUS_RECONCILIATION_DIR.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    if set(declared_paths) != expected_output_files or actual_output_files != expected_output_files:
        raise AssertionError(
            "Corpus reconciliation directory disagrees with its release contract; "
            f"manifest_missing={sorted(expected_output_files - set(declared_paths))}, "
            f"manifest_extra={sorted(set(declared_paths) - expected_output_files)}, "
            f"directory_missing={sorted(expected_output_files - actual_output_files)}, "
            f"directory_extra={sorted(actual_output_files - expected_output_files)}"
        )

    def software_index(records: Any, label: str) -> dict[str, dict[str, Any]]:
        if not isinstance(records, list):
            raise AssertionError(f"Corpus reconciliation {label} must be a list")
        indexed: dict[str, dict[str, Any]] = {}
        for position, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise AssertionError(f"Corpus reconciliation {label}[{position}] lacks a string path")
            relative = record["path"]
            if relative in indexed:
                raise AssertionError(f"Corpus reconciliation {label} repeats software path: {relative}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
                raise AssertionError(f"Corpus reconciliation {label} has an invalid SHA-256: {relative}")
            indexed[relative] = record
        return indexed

    manifest_software = software_index(manifest.get("software"), "manifest software")
    summary_software = software_index(summary.get("software_fingerprints"), "summary software_fingerprints")
    if set(manifest_software) != CORPUS_RECONCILIATION_SOFTWARE:
        raise AssertionError(
            "Corpus reconciliation software manifest has the wrong file set: "
            f"{sorted(set(manifest_software) ^ CORPUS_RECONCILIATION_SOFTWARE)}"
        )
    if {
        relative: record["sha256"] for relative, record in manifest_software.items()
    } != {
        relative: record["sha256"] for relative, record in summary_software.items()
    }:
        raise AssertionError("Corpus reconciliation manifest and summary software hashes disagree")
    for relative, record in manifest_software.items():
        verify_record(ROOT / relative, record, ROOT)

    validation_checks = validation.get("checks")
    summary_checks = summary.get("checks")
    if not isinstance(validation_checks, list) or not isinstance(summary_checks, list):
        raise AssertionError("Corpus reconciliation checks must be lists")
    validation_check_index = {record.get("name"): record for record in validation_checks}
    summary_check_index = {record.get("name"): record for record in summary_checks}
    if len(validation_check_index) != len(validation_checks) or len(summary_check_index) != len(summary_checks):
        raise AssertionError("Corpus reconciliation contains duplicate validation check names")
    required_checks = {
        "drive_live_sheet_keys_and_row_count_match_local_export",
        "drive_live_substantive_mismatches_all_adjudicated",
        "frozen_snapshot_content_exactly_reconstructed",
        "source_only_metadata_count_reconciles",
        "frozen_ids_all_join_source_metadata",
        "all_dedup_lost_songs_reconciled_to_retained_exact_chunks",
        "expected_reconstructed_counts",
        "task_aligned_released_population_reconstructed",
        "task_aligned_counterfactual_expected_counts",
        "task_aligned_family_totals_and_headline_deltas_reconstructed",
        "legacy_full_source_written_ending_sensitivity_reconstructed",
    }
    if set(validation_check_index) != required_checks or set(summary_check_index) != required_checks:
        raise AssertionError("Corpus reconciliation validation check set is incomplete or unexpected")
    if validation_check_index != summary_check_index or not all(
        record.get("passed") is True for record in validation_checks
    ):
        raise AssertionError("Corpus reconciliation validation checks disagree or do not all pass")

    stage_rows = read_csv_rows(CORPUS_RECONCILIATION_DIR / "stage_counts.csv")
    expected_stages = {
        "raw lyric chunks": (26833, 7721, 0, 0),
        "after title exclusions": (25279, 7420, 1554, 301),
        "after line cleaning and empty removal": (25026, 7391, 253, 29),
        "after artist + exact-text deduplication": (22132, 7214, 2894, 177),
    }
    if len(stage_rows) != len(expected_stages) or {row["stage"] for row in stage_rows} != set(expected_stages):
        raise AssertionError("Corpus reconciliation stage table has the wrong stages")
    for row in stage_rows:
        observed = tuple(
            int(row[key])
            for key in ("rows", "songs", "rows_removed_at_stage", "songs_removed_at_stage")
        )
        if observed != expected_stages[row["stage"]]:
            raise AssertionError(f"Corpus reconciliation stage counts changed: {row['stage']}")

    lost_rows = read_csv_rows(CORPUS_RECONCILIATION_DIR / "lost_song_classification.csv")
    expected_lost = {
        "exact_sequence_duplicate_of_retained_song": (136, 289, 131, 1.0),
        "same_chunk_multiset_different_order": (0, 0, 0, 0.0),
        "chunk_multiset_subset_of_one_retained_song": (33, 44, 13, 1.0),
        "chunks_distributed_across_multiple_retained_songs": (8, 20, 8, 2.0),
        "unreconciled_other": (0, 0, 0, 0.0),
    }
    if len({row["category"] for row in lost_rows}) != len(lost_rows):
        raise AssertionError("Corpus reconciliation lost-song table repeats a category")
    if unknown_categories := sorted({row["category"] for row in lost_rows} - set(expected_lost)):
        raise AssertionError(f"Corpus reconciliation has unexpected lost-song categories: {unknown_categories}")
    observed_lost = {
        category: (0, 0, 0, 0.0)
        for category in expected_lost
    }
    for row in lost_rows:
        observed_lost[row["category"]] = (
            int(row["songs"]),
            int(row["cleaned_chunk_rows"]),
            int(row["songs_with_exact_title_match"]),
            float(row["median_related_retained_songs"]),
        )
    if observed_lost != expected_lost or sum(record[0] for record in observed_lost.values()) != 177:
        raise AssertionError(f"Corpus reconciliation lost-song headline changed: {observed_lost}")

    geometry = summary["duplicate_geometry"]
    diagnostics = summary["lost_song_diagnostics"]
    interpretation = summary["interpretation"]
    expected_geometry = {
        "rows_removed_by_keep_first": 2894,
        "songs_removed_entirely_by_deduplication": 177,
    }
    if any(int(geometry.get(key, -1)) != value for key, value in expected_geometry.items()):
        raise AssertionError("Corpus reconciliation duplicate-removal geometry changed")
    expected_diagnostics = {
        "lost_songs": 177,
        "exact_cleaned_sequence_lost_songs": 136,
        "exact_cleaned_sequence_and_title_lost_songs": 131,
        "exact_raw_sequence_lost_songs": 134,
        "exact_raw_sequence_and_title_lost_songs": 130,
        "conservative_manual_review_queue": 46,
    }
    if any(int(diagnostics.get(key, -1)) != value for key, value in expected_diagnostics.items()):
        raise AssertionError("Corpus reconciliation lost-song diagnostics changed")
    if (
        int(interpretation.get("high_confidence_ingestion_duplicate_records", -1)) != 131
        or int(interpretation.get("manual_review_queue_records", -1)) != 46
    ):
        raise AssertionError("Corpus reconciliation release action counts changed")

    drive = summary["drive_lineage"]
    expected_drive_values = {
        "live_sheet_rows": 26833,
        "local_export_rows": 26833,
        "song_id_exact_mismatches": 0,
        "chunk_id_exact_mismatches": 0,
        "artist_exact_mismatches": 7,
        "artist_normalisation_equivalent_mismatches": 7,
        "title_exact_mismatches": 161,
        "title_normalisation_equivalent_mismatches": 132,
        "drive_native_sheet_title_type_coercions": 29,
        "text_exact_mismatches": 23836,
        "text_newline_only_mismatches": 23832,
        "drive_leading_apostrophe_escape_semantics": 2,
        "drive_control_character_import_side_effects": 2,
        "unresolved_substantive_mismatches": 0,
    }
    if any(int(drive.get(key, -1)) != value for key, value in expected_drive_values.items()):
        raise AssertionError("Corpus reconciliation Drive-lineage headline changed")
    if drive.get("row_count_match") is not True or drive.get("local_canonical_content_bound_to_current_raw_export") is not True:
        raise AssertionError("Corpus reconciliation Drive comparison is not bound to the current raw export")
    if drive.get("remote_drive_object_byte_identity_verified") is not False:
        raise AssertionError("Corpus reconciliation overstates remote Drive byte identity")

    task_rows = read_csv_rows(CORPUS_RECONCILIATION_DIR / "task_aligned_written_rhyme_sensitivity.csv")
    task_index = {row["population"]: row for row in task_rows}
    expected_task_rows = {
        "released_frozen_task": (15760, 5619, 283806, 238881, 52152),
        "pre_snapshot_duplicate_chunk_counterfactual": (17715, 5621, 290839, 243819, 58624),
    }
    if set(task_index) != set(expected_task_rows) or len(task_index) != len(task_rows):
        raise AssertionError("Corpus reconciliation task-aligned sensitivity populations changed")
    for population, expected in expected_task_rows.items():
        row = task_index[population]
        observed = tuple(
            int(row[key])
            for key in (
                "input_chunks",
                "input_songs",
                "strict_han_ending_line_occurrences",
                "adjacent_transitions_before_leakage_filter",
                "repeat_excess_after_first",
            )
        )
        if observed != expected:
            raise AssertionError(f"Corpus reconciliation task-aligned headline changed: {population}")

    family_rows = read_csv_rows(
        CORPUS_RECONCILIATION_DIR / "task_aligned_family_distribution_sensitivity.csv"
    )
    if len(family_rows) != 17:
        raise AssertionError("Corpus reconciliation must report all 17 written-ending families")
    released_family_total = sum(int(row["released_count"]) for row in family_rows)
    counterfactual_family_total = sum(int(row["counterfactual_count"]) for row in family_rows)
    family_total_variation = 0.5 * sum(
        abs(float(row["counterfactual_minus_released_share"])) for row in family_rows
    )
    if released_family_total != 283806 or counterfactual_family_total != 290839:
        raise AssertionError("Corpus reconciliation family counts do not reconcile to the task populations")
    if not math.isclose(family_total_variation, 0.0017890317967706243, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Corpus reconciliation task-aligned family total variation changed")

    task_summary = summary["task_aligned_written_rhyme_sensitivity"]
    expected_task_summary = {
        "strict_han_lines_restored": 7033,
        "adjacent_transitions_restored": 4938,
        "repeat_excess_restored": 6472,
    }
    if any(int(task_summary.get(key, -1)) != value for key, value in expected_task_summary.items()):
        raise AssertionError("Corpus reconciliation task-aligned restored counts changed")
    if task_summary.get("predictive_metrics_retrained") is not False:
        raise AssertionError("Corpus reconciliation incorrectly claims predictive retraining")
    if not math.isclose(
        float(task_summary["family_distribution_total_variation"]),
        0.0017890317967706243,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(task_summary["switch_rate_counterfactual_minus_released"]),
        -0.0013899941300251628,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Corpus reconciliation task-aligned sensitivity deltas changed")

    broad_summary = summary["written_rhyme_sensitivity"]
    expected_broad_counts = {
        "strict_han_lines_removed_by_chunk_dedup": 16623,
        "adjacent_transitions_removed_by_chunk_dedup": 13112,
    }
    if any(int(broad_summary.get(key, -1)) != value for key, value in expected_broad_counts.items()):
        raise AssertionError("Corpus reconciliation broad written-ending counts changed")
    if not math.isclose(
        float(broad_summary["global_family_distribution_total_variation"]),
        0.0017671763323308868,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(broad_summary["switch_rate_absolute_difference"]),
        0.0017874646921119952,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Corpus reconciliation broad written-ending deltas changed")

    expected_privacy = "aggregate only; no lyric text, labels, titles, song/chunk identifiers, or row-level hashes"
    if summary.get("privacy") != expected_privacy or validation.get("privacy") != expected_privacy:
        raise AssertionError("Corpus reconciliation privacy declaration changed")
    private_path_pattern = re.compile(r"(?:[A-Za-z]:[\\/]|/Users/|/home/|/mnt/|\\\\Users\\\\)", re.IGNORECASE)
    for path in sorted(CORPUS_RECONCILIATION_DIR.iterdir()):
        if path.is_file() and is_text_path(path):
            text = path.read_text(encoding="utf-8")
            if private_path_pattern.search(text):
                raise AssertionError(f"Corpus reconciliation publishes an absolute private path: {path.name}")

    prohibited_csv_columns = {
        "artist",
        "chunk_id",
        "content_hash",
        "label_id",
        "line",
        "line_id",
        "line_text",
        "lyric_hash",
        "lyrics",
        "song_id",
        "song_title",
        "source_credit_label",
        "text",
    }
    for path in CORPUS_RECONCILIATION_DIR.glob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if prohibited := sorted(set(header) & prohibited_csv_columns):
            raise AssertionError(f"Corpus reconciliation CSV exposes private columns in {path.name}: {prohibited}")

    input_fingerprints = summary.get("input_fingerprints")
    if not isinstance(input_fingerprints, dict) or "drive_live_comparison_summary" not in input_fingerprints:
        raise AssertionError("Corpus reconciliation input fingerprints are incomplete")
    for name, record in input_fingerprints.items():
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise AssertionError(f"Corpus reconciliation input fingerprint is malformed: {name}")
        file_name = record["file"]
        if Path(file_name).name != file_name or "/" in file_name or "\\" in file_name:
            raise AssertionError(f"Corpus reconciliation publishes a private input path: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
            raise AssertionError(f"Corpus reconciliation input fingerprint SHA-256 is malformed: {name}")

    return (
        len(declared_paths)
        + len(manifest_software)
        + len(required_checks)
        + len(expected_stages)
        + len(expected_lost)
        + len(expected_drive_values)
        + len(expected_task_rows)
        + len(family_rows)
        + len(input_fingerprints)
        + 18
    )


def verify_robustness_claims() -> int:
    summary = read_json(ROBUSTNESS_DIR / "analysis_summary.json")
    validation = read_json(ROBUSTNESS_DIR / "validation.json")
    manifest = read_json(ROBUSTNESS_DIR / "manifest.json")
    declared_files = set(manifest.get("output_files", {})) | {"manifest.json", "validation.json"}
    actual_files = {path.name for path in ROBUSTNESS_DIR.iterdir() if path.is_file()}
    if actual_files != declared_files:
        raise AssertionError(
            "Repertoire robustness directory disagrees with its manifest; "
            f"missing={sorted(declared_files - actual_files)}, unmanifested={sorted(actual_files - declared_files)}"
        )
    if summary.get("status") != "pass" or validation.get("status") != "pass":
        raise AssertionError("Repertoire robustness component is not passing")
    if summary.get("artifact_id") != manifest.get("artifact_id") or summary.get("artifact_id") != validation.get("artifact_id"):
        raise AssertionError("Repertoire robustness artifact identifiers disagree")

    checks = {record["name"]: record for record in validation.get("checks", [])}
    required_checks = {
        "private_source_manifest_hashes_match",
        "released_edges_equal_two_layer_intersection",
        "public_pca_reproduces_from_consensus_vectors",
        "primary_null_is_seeded_and_degree_sequence_preserving",
        "all_outputs_are_aggregate_only",
    }
    failed_or_missing = sorted(name for name in required_checks if not checks.get(name, {}).get("passed"))
    if failed_or_missing or not all(record.get("passed") for record in validation.get("checks", [])):
        raise AssertionError(f"Repertoire robustness checks failed or are missing: {failed_or_missing}")

    graph_manifest_path = ROOT / "results" / "repertoire-network-v1" / "graph" / "manifest.json"
    if manifest["source_hashes"]["public_graph_manifest_sha256"] != sha256(graph_manifest_path):
        raise AssertionError("Robustness source hash does not match the public graph manifest")

    parent = read_json(ROOT / "results" / "repertoire-network-v1" / "manifest.json")
    expected = parent["expected_counts"]
    null = summary["graph_alignment_null"]
    projection = summary["projection_fidelity"]
    if null.get("null_model") != "degree-preserving double-edge swaps of the sensitivity layer":
        raise AssertionError("Primary graph-alignment null is not degree-preserving")
    if int(null.get("null_replicates", 0)) < 10_000:
        raise AssertionError("Primary degree-preserving null has fewer than 10,000 replicates")
    if int(null.get("successful_swaps_per_replicate", 0)) < 10 * int(null["sensitivity_edges"]):
        raise AssertionError("Degree-preserving null did not perform the declared mixing work")
    if int(null["observed_intersection_edges"]) != int(expected["released_edges"]):
        raise AssertionError("Robustness observed edge intersection disagrees with the released graph")
    if int(null["observed_intersection_edges"]) <= int(null["null_maximum"]):
        raise AssertionError("Observed graph alignment is not outside the released null distribution")
    expected_p = (int(null["exceedances"]) + 1) / (int(null["null_replicates"]) + 1)
    if not math.isclose(float(null["monte_carlo_p_add_one"]), expected_p, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Primary add-one Monte Carlo p-value is inconsistent")

    auxiliary = null.get("auxiliary_label_permutation", {})
    if int(auxiliary.get("null_replicates", 0)) < 100_000:
        raise AssertionError("Auxiliary source-label permutation null has fewer than 100,000 replicates")
    auxiliary_p = (int(auxiliary["exceedances"]) + 1) / (int(auxiliary["null_replicates"]) + 1)
    if not math.isclose(float(auxiliary["monte_carlo_p_add_one"]), auxiliary_p, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Auxiliary add-one Monte Carlo p-value is inconsistent")

    if int(projection["population"]) != int(expected["eligible_labels"]):
        raise AssertionError("Projection-fidelity population disagrees with the released graph")
    if int(projection["released_edges"]) != int(expected["released_edges"]):
        raise AssertionError("Projection-fidelity edge count disagrees with the released graph")
    neighbourhoods = projection.get("neighbourhood_fidelity", [])
    if [int(row["k"]) for row in neighbourhoods] != [5, 10, 15]:
        raise AssertionError("Projection fidelity must report k=5, 10, and 15")
    for row in neighbourhoods:
        for key in ("trustworthiness", "mean_exact_neighbour_overlap", "random_overlap_expectation"):
            if not 0.0 <= float(row[key]) <= 1.0:
                raise AssertionError(f"Projection-fidelity value is outside [0, 1]: {key}")
    if not 0 <= int(projection["released_edges_mutual_top5_in_2d"]) <= int(projection["released_edges"]):
        raise AssertionError("Invalid mutual-top-five edge count in projection diagnostics")
    if not 0 <= int(projection["released_edges_at_least_one_way_top5_in_2d"]) <= int(projection["released_edges"]):
        raise AssertionError("Invalid one-way-top-five edge count in projection diagnostics")

    distribution_rows = read_csv_rows(ROBUSTNESS_DIR / "null_overlap_distribution.csv")
    expected_replicates = {
        "degree_preserving_edge_swap": int(null["null_replicates"]),
        "source_label_permutation": int(auxiliary["null_replicates"]),
    }
    for model, replicates in expected_replicates.items():
        rows = [row for row in distribution_rows if row["null_model"] == model]
        if sum(int(row["replicates"]) for row in rows) != replicates:
            raise AssertionError(f"Null-distribution replicate count is inconsistent: {model}")
        if not math.isclose(sum(float(row["probability"]) for row in rows), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise AssertionError(f"Null-distribution probabilities do not sum to one: {model}")

    projection_rows = read_csv_rows(ROBUSTNESS_DIR / "projection_fidelity.csv")
    by_k = {int(row["k"]): row for row in projection_rows}
    for record in neighbourhoods:
        row = by_k.get(int(record["k"]))
        if row is None:
            raise AssertionError(f"Projection-fidelity CSV omits k={record['k']}")
        for key in ("trustworthiness", "mean_exact_neighbour_overlap", "random_overlap_expectation"):
            if not math.isclose(float(row[key]), float(record[key]), rel_tol=0.0, abs_tol=5e-9):
                raise AssertionError(f"Projection-fidelity CSV disagrees with JSON at k={record['k']}: {key}")

    diagnostics = {row["metric"]: row for row in read_csv_rows(ROBUSTNESS_DIR / "diagnostic_summary.csv")}
    required_metrics = {
        "observed_cross_treatment_edge_intersection",
        "degree_preserving_null_expected_intersection",
        "degree_preserving_null_monte_carlo_p_add_one",
        "label_permutation_null_expected_intersection",
        "pca_trustworthiness_at_5",
        "pca_exact_top5_overlap",
        "pca_pairwise_rank_spearman",
    }
    if missing_metrics := sorted(required_metrics - diagnostics.keys()):
        raise AssertionError(f"Robustness diagnostic summary omits metrics: {missing_metrics}")
    diagnostic_expectations = {
        "observed_cross_treatment_edge_intersection": float(null["observed_intersection_edges"]),
        "degree_preserving_null_expected_intersection": float(null["null_mean"]),
        "degree_preserving_null_monte_carlo_p_add_one": float(null["monte_carlo_p_add_one"]),
        "label_permutation_null_expected_intersection": float(auxiliary["null_mean"]),
        "pca_trustworthiness_at_5": float(neighbourhoods[0]["trustworthiness"]),
        "pca_exact_top5_overlap": float(neighbourhoods[0]["mean_exact_neighbour_overlap"]),
        "pca_pairwise_rank_spearman": float(projection["pairwise_rank_spearman"]),
    }
    for metric, expected_value in diagnostic_expectations.items():
        if not math.isclose(float(diagnostics[metric]["value"]), expected_value, rel_tol=0.0, abs_tol=5e-5):
            raise AssertionError(f"Robustness diagnostic CSV disagrees with JSON: {metric}")

    site_graph = read_json(ROOT / "site" / "app" / "data" / "researchData.json")["repertoireGraph"]
    if site_graph.get("alignmentNull") != null or site_graph.get("projectionFidelity") != projection:
        raise AssertionError("Site data does not reproduce the released robustness summary")

    parent_validation = read_json(ROOT / "results" / "repertoire-network-v1" / "validation.json")
    parent_checks = {record["name"]: record for record in parent_validation.get("checks", [])}
    parent_robustness = parent_checks.get("graph_null_and_projection_fidelity_component_pass", {})
    expected_observed = {
        "null_replicates": int(null["null_replicates"]),
        "released_edges": int(projection["released_edges"]),
        "pca_population": int(projection["population"]),
    }
    if not parent_robustness.get("passed") or parent_robustness.get("observed") != expected_observed:
        raise AssertionError("Parent repertoire validation does not match the robustness component")
    return 28


def verify_retrieval_sensitivity_claims() -> int:
    manifest = read_json(RETRIEVAL_SENSITIVITY_DIR / "manifest.json")
    validation = read_json(RETRIEVAL_SENSITIVITY_DIR / "validation.json")
    summary = read_json(RETRIEVAL_SENSITIVITY_DIR / "analysis_summary.json")
    declared_files = set(manifest.get("files", {})) | {"manifest.json"}
    actual_files = {path.name for path in RETRIEVAL_SENSITIVITY_DIR.iterdir() if path.is_file()}
    if actual_files != declared_files:
        raise AssertionError(
            "Retrieval sensitivity directory disagrees with its manifest; "
            f"missing={sorted(declared_files - actual_files)}, unmanifested={sorted(actual_files - declared_files)}"
        )
    artifact_id = "retrieval-inductive-sensitivity-v1"
    if any(payload.get("artifact_id") != artifact_id for payload in (manifest, validation, summary)):
        raise AssertionError("Retrieval sensitivity artifact identifiers disagree")
    if validation.get("status") != "pass" or not all(record.get("passed") for record in validation.get("checks", [])):
        raise AssertionError("Retrieval sensitivity validation is not passing")

    checks = {record["name"]: record for record in validation["checks"]}
    required_checks = {
        "global_duplicate_components_never_cross_folds",
        "inductive_and_matched_systems_use_identical_fold_profiles",
        "every_query_has_exactly_one_valid_out_of_fold_rank",
        "paired_two_stage_bootstrap_is_complete",
        "point_estimates_reproduce_independent_read_only_run",
        "public_payloads_are_utf8_without_bom_and_lf_only",
        "claim_boundary_present_in_public_documents",
    }
    if missing_checks := sorted(required_checks - checks.keys()):
        raise AssertionError(f"Retrieval sensitivity validation omits checks: {missing_checks}")
    regression = checks["point_estimates_reproduce_independent_read_only_run"].get("detail", {})
    if float(regression.get("maximum_absolute_difference", math.inf)) > float(regression.get("tolerance", -math.inf)):
        raise AssertionError("Retrieval sensitivity independent point-estimate regression exceeds tolerance")

    if manifest["lineage"]["builder_code_sha256"] != sha256(ROOT / "src" / "build_retrieval_inductive_sensitivity_v1.py"):
        raise AssertionError("Retrieval sensitivity builder hash is stale")
    if manifest["lineage"]["source_retrieval_validation_sha256"] != sha256(ROOT / "results" / "retrieval-v1" / "validation.json"):
        raise AssertionError("Retrieval sensitivity public validation lineage is stale")
    if manifest["lineage"]["graph_validation_sha256"] != sha256(ROOT / "results" / "repertoire-network-v1" / "graph" / "validation.json"):
        raise AssertionError("Retrieval sensitivity graph-validation lineage is stale")

    population = summary["population"]
    if int(population["folds"]) != 6 or int(population["queries"]) != 5_455 or int(population["source_credit_labels"]) != 204:
        raise AssertionError("Retrieval sensitivity population contract changed")
    if sum(int(value) for value in population["fold_query_counts"]) != int(population["queries"]):
        raise AssertionError("Retrieval sensitivity fold counts do not cover every query exactly once")

    metric_rows = read_csv_rows(RETRIEVAL_SENSITIVITY_DIR / "metrics.csv")
    macro_metrics = summary["macro_metrics"]
    expected_metrics = {"mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10"}
    if set(macro_metrics) != {row["system"] for row in metric_rows}:
        raise AssertionError("Retrieval sensitivity systems disagree between JSON and CSV")
    if len(metric_rows) != len(macro_metrics) * len(expected_metrics):
        raise AssertionError("Retrieval sensitivity metric table is incomplete")
    metric_index = {(row["system"], row["metric"]): row for row in metric_rows}
    for system, metrics in macro_metrics.items():
        if set(metrics) != expected_metrics:
            raise AssertionError(f"Retrieval sensitivity JSON metric set is incomplete: {system}")
        for metric, values in metrics.items():
            row = metric_index.get((system, metric))
            if row is None:
                raise AssertionError(f"Retrieval sensitivity CSV omits {system} / {metric}")
            released = (float(row["ci95_lower"]), float(row["estimate"]), float(row["ci95_upper"]))
            expected_values = (float(values["ci95_lower"]), float(values["estimate"]), float(values["ci95_upper"]))
            if any(not math.isclose(left, right, rel_tol=0.0, abs_tol=5e-7) for left, right in zip(released, expected_values)):
                raise AssertionError(f"Retrieval sensitivity metric CSV disagrees with JSON: {system} / {metric}")
            if not 0.0 <= released[0] <= released[1] <= released[2] <= 1.0:
                raise AssertionError(f"Retrieval sensitivity interval is invalid: {system} / {metric}")

    delta_rows = read_csv_rows(RETRIEVAL_SENSITIVITY_DIR / "paired_deltas.csv")
    summary_deltas = summary["paired_deltas"]
    if len(delta_rows) != len(summary_deltas) * len(expected_metrics):
        raise AssertionError("Retrieval sensitivity paired-delta table is incomplete")
    delta_index = {(row["comparison"], row["metric"]): row for row in delta_rows}
    for comparison, metrics in summary_deltas.items():
        if set(metrics) != expected_metrics:
            raise AssertionError(f"Retrieval sensitivity paired metric set is incomplete: {comparison}")
        for metric, values in metrics.items():
            row = delta_index.get((comparison, metric))
            if row is None:
                raise AssertionError(f"Retrieval sensitivity paired CSV omits {comparison} / {metric}")
            released = (float(row["ci95_lower"]), float(row["estimate_delta"]), float(row["ci95_upper"]))
            expected_values = (float(values["ci95_lower"]), float(values["estimate_delta"]), float(values["ci95_upper"]))
            if any(not math.isclose(left, right, rel_tol=0.0, abs_tol=5e-7) for left, right in zip(released, expected_values)):
                raise AssertionError(f"Retrieval sensitivity paired CSV disagrees with JSON: {comparison} / {metric}")
            if not released[0] <= released[1] <= released[2]:
                raise AssertionError(f"Retrieval sensitivity paired interval is invalid: {comparison} / {metric}")
            expected_direction = "left_higher" if released[0] > 0 else ("right_higher" if released[2] < 0 else "interval_includes_zero")
            if row["interval_direction"] != expected_direction or values["interval_direction"] != expected_direction:
                raise AssertionError(f"Retrieval sensitivity interval direction is inconsistent: {comparison} / {metric}")

    exposure = summary_deltas["TF-IDF matched transductive minus inductive train-only"]
    inductive_fusion = summary_deltas["inductive fusion minus inductive TF-IDF"]
    for metric in expected_metrics:
        if float(inductive_fusion[metric]["ci95_lower"]) <= 0:
            raise AssertionError(f"Inductive fusion advantage is not supported for {metric}")
        if abs(float(exposure[metric]["estimate_delta"])) >= float(inductive_fusion[metric]["estimate_delta"]):
            raise AssertionError(f"TF-IDF exposure effect is not smaller than the inductive fusion advantage for {metric}")

    fold_rows = read_csv_rows(RETRIEVAL_SENSITIVITY_DIR / "fold_summary.csv")
    if len(fold_rows) != int(population["folds"]) or sum(int(row["test_queries"]) for row in fold_rows) != int(population["queries"]):
        raise AssertionError("Retrieval sensitivity fold CSV does not cover the released population")
    if any(int(row["test_source_credit_labels"]) != 204 or int(row["minimum_training_components_per_label"]) < 5 for row in fold_rows):
        raise AssertionError("Retrieval sensitivity fold coverage contract failed")
    return 36


def verify_ner_released_claim_audit() -> int:
    status_path = ROOT / "results" / "ner-v1" / "released_claim_audit_status.json"
    protocol_path = ROOT / "methods" / "NER_RELEASED_CLAIM_AUDIT_PROTOCOL.md"
    builder_path = ROOT / "src" / "build_ner_released_claim_audit_v1.py"
    status = read_json(status_path)
    if status.get("artifact_id") != "chinese-rap-ner-released-claim-audit-v1":
        raise AssertionError("Unexpected released-claim NER audit artifact identifier")
    if status.get("status") != "PENDING_DUAL_HUMAN_REVIEW_AND_ADJUDICATION":
        raise AssertionError("Released-claim NER audit must remain pending until human adjudication")
    if status.get("protocol") != "methods/NER_RELEASED_CLAIM_AUDIT_PROTOCOL.md":
        raise AssertionError("Released-claim NER audit protocol pointer is invalid")

    lineage = status["input_lineage"]
    public_lineage = {
        "builder_code_sha256": builder_path,
        "protocol_sha256": protocol_path,
        "released_entity_co_mentions_sha256": ROOT / "results" / "ner-v1" / "entity_co_mentions_provisional.csv",
        "released_source_label_entity_links_sha256": ROOT / "results" / "ner-v1" / "source_label_entity_links_provisional.csv",
    }
    for key, path in public_lineage.items():
        if lineage.get(key) != sha256(path):
            raise AssertionError(f"Released-claim NER audit public lineage is stale: {key}")

    benchmark = status["global_ner_benchmark"]
    if benchmark != {
        "completed_gold_items": 0,
        "planned_dual_review_items": 800,
        "precision_recall_f1": "WITHHELD",
        "status": "NOT_COMPLETED",
    }:
        raise AssertionError("Released-claim audit overclaims the unfinished global NER benchmark")
    if any(int(value) != 0 for value in status["review_progress"].values()):
        raise AssertionError("Released-claim NER audit reports human progress before decisions are completed")
    if any(not str(value).startswith("WITHHELD") for value in status["targeted_audit_metrics"].values()):
        raise AssertionError("Released-claim NER audit exposes metrics before adjudication")

    privacy = status["privacy"]
    if privacy["private_package_location_published"] or int(privacy["public_lyric_or_context_rows"]) != 0 or int(privacy["public_song_or_chunk_locators"]) != 0:
        raise AssertionError("Released-claim NER audit violates the public/private boundary")
    serialized = json.dumps(status, ensure_ascii=False)
    if re.search(r"[A-Za-z]:[\\/]", serialized) or "/Users/" in serialized or "\\Users\\" in serialized:
        raise AssertionError("Released-claim NER audit status contains an absolute local path")

    scope = status["scope"]
    if int(scope["released_claims_total"]) != int(scope["released_link_claims"]) + int(scope["released_co_mention_claims"]):
        raise AssertionError("Released-claim NER audit claim counts do not reconcile")
    expected_unique = (
        int(scope["link_contributing_occurrence_rows"])
        + int(scope["co_mention_contributing_occurrence_rows"])
        - int(scope["overlapping_link_and_co_mention_occurrence_rows"])
    )
    if int(scope["unique_contributing_occurrence_rows"]) != expected_unique:
        raise AssertionError("Released-claim NER audit occurrence counts do not reconcile")
    expected_occurrence_tasks = expected_unique + int(scope["real_comparator_controls"]) + int(scope["synthetic_boundary_controls"])
    if int(scope["occurrence_review_tasks_total"]) != expected_occurrence_tasks:
        raise AssertionError("Released-claim NER audit occurrence task count does not reconcile")
    expected_pair_tasks = int(scope["released_co_mention_pair_tasks"]) + int(scope["real_co_mention_comparator_tasks"])
    if int(scope["co_mention_pair_review_tasks_total"]) != expected_pair_tasks:
        raise AssertionError("Released-claim NER audit pair task count does not reconcile")

    package_validation = status["validation"]
    if package_validation["package_generation"] != "pass" or float(package_validation["released_claim_occurrence_coverage"]) != 1.0:
        raise AssertionError("Released-claim NER audit package or occurrence coverage is not passing")
    if package_validation["released_co_mention_support_reconciliation"] != "pass" or package_validation["released_link_support_reconciliation"] != "pass":
        raise AssertionError("Released-claim NER audit support reconciliation is not passing")
    if not package_validation["review_templates_blank"] or not package_validation["reviewer_claim_and_control_status_blinded"]:
        raise AssertionError("Released-claim NER review templates are not blank and blinded")

    protocol = protocol_path.read_text(encoding="utf-8")
    required_statements = (
        "cannot estimate corpus-wide NER precision, recall, or F1",
        "independent dual review",
        "Private local research material",
        "all 157 released-claim occurrences",
    )
    if not all(statement.lower() in protocol.lower() for statement in required_statements):
        raise AssertionError("Released-claim NER audit protocol omits a required scope or privacy statement")
    return 24


def verify_internal_hashes() -> int:
    checked = 0
    standalone = read_json(ROOT / "validation" / "standalone_site_validation.json")
    verify_record(ROOT / "index.html", standalone, ROOT)
    checked += 1

    portable = read_json(ROOT / "validation" / "portable_site_manifest.json")
    if portable["portableHtmlSha256"] != sha256(ROOT / "index.html"):
        raise AssertionError("portable_site_manifest portableHtmlSha256 is stale")
    if portable["researchDataSourceSha256"] != sha256(ROOT / "site" / "app" / "data" / "researchData.json"):
        raise AssertionError("portable_site_manifest researchDataSourceSha256 is stale")
    checked += 2

    site_manifest = read_json(ROOT / "site" / "public" / "data" / "manifest.json")
    site_data = ROOT / "site" / "public" / "data" / "researchData.json"
    if site_manifest["researchDataSha256"] != sha256(site_data):
        raise AssertionError("site public-data manifest hash is stale")
    if site_data.read_bytes() != (ROOT / "site" / "app" / "data" / "researchData.json").read_bytes():
        raise AssertionError("React and public site data are not byte-identical")
    checked += 2

    release = read_json(ROOT / "validation" / "release_validation.json")
    artifact_paths = {
        "portable_site_sha256": ROOT / "index.html",
        "application_data_sha256": ROOT / "site" / "app" / "data" / "researchData.json",
        "manuscript_markdown_sha256": ROOT / "paper" / "manuscript.md",
        "review_manuscript_docx_sha256": ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript.docx",
        "review_manuscript_pdf_sha256": ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript.pdf",
        "dsh_manuscript_docx_sha256": ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx",
        "dsh_manuscript_pdf_sha256": ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf",
        "supplement_docx_sha256": ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Supplement.docx",
        "supplement_pdf_sha256": ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Supplement.pdf",
        "journal_figure_validation_sha256": ROOT / "figures" / "journal_figure_validation.json",
    }
    for key, path in artifact_paths.items():
        if release["artifact_hashes"][key] != sha256(path):
            raise AssertionError(f"release_validation artifact hash is stale: {key}")
        checked += 1
    return checked


def verify_text_and_syntax() -> tuple[int, int, int]:
    text_count = 0
    json_count = 0
    csv_count = 0
    for path in tracked_paths():
        if is_text_path(path):
            payload = path.read_bytes()
            if payload != normalized_bytes(payload):
                raise AssertionError(f"Text encoding/line-ending contract failed: {path.relative_to(ROOT)}")
            text_count += 1
        if path.suffix.lower() == ".json":
            read_json(path)
            json_count += 1
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
            if not header or any(not cell for cell in header) or len(header) != len(set(header)):
                raise AssertionError(f"Invalid or duplicate CSV headers: {path.relative_to(ROOT)}")
            csv_count += 1
    for validation_name in ("validation.json", "journal_figure_validation.json"):
        payload = read_json(ROOT / "figures" / validation_name)
        names = [record["name"] for record in payload.get("checks", [])]
        if len(names) != len(set(names)):
            raise AssertionError(f"Duplicate figure check names in figures/{validation_name}")
    return text_count, json_count, csv_count


def verify_local_links(document: Path, target: Path, pattern: str) -> int:
    checked = 0
    for relative in re.findall(pattern, document.read_text(encoding="utf-8")):
        relative = relative.split("#", 1)[0].split("?", 1)[0]
        if not relative or relative.startswith(("http://", "https://", "mailto:")):
            continue
        linked = target / Path(relative)
        assert_within(linked, target)
        if not linked.is_file():
            raise AssertionError(f"Broken local link in {document}: {relative}")
        checked += 1
    return checked


def verify_desktop_package(target: Path) -> int:
    target = target.resolve()
    if target.name != "Chinese_Rap_Research_Release_V4" or not target.is_dir():
        raise AssertionError(f"Unexpected desktop release path: {target}")
    manifest_path = target / "Validation" / "RELEASE_PACKAGE_MANIFEST.json"
    checked = verify_manifest(manifest_path, "files", target, target)
    package_manifest = read_json(manifest_path)
    declared_paths = [relative for relative, _ in iter_records(package_manifest["files"])]
    if len(declared_paths) != len(set(declared_paths)):
        raise AssertionError("Desktop release manifest contains duplicate paths")
    actual_paths = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(declared_paths) != actual_paths:
        missing = sorted(set(declared_paths) - actual_paths)
        unmanifested = sorted(actual_paths - set(declared_paths))
        raise AssertionError(f"Desktop file set disagrees with its manifest; missing={missing}, unmanifested={unmanifested}")

    required_paths = {
        "LICENSE",
        "LICENSE-CODE",
        "PROJECT_README.md",
        "Methods/NER_RELEASED_CLAIM_AUDIT_PROTOCOL.md",
        "Methods/PROTOCOL_AMENDMENT_PD002_UPSTREAM_CHUNK_DEDUPLICATION.md",
        "Paper/derivative_provenance.json",
        "Reproducibility/.python-version",
        "Reproducibility/requirements.txt",
        "Reproducibility/src/build_canonical_lyric_text_sidecar_v1.py",
        "Reproducibility/src/build_chinese_rap_written_rhyme_v1.py",
        "Reproducibility/src/build_chinese_rap_paper_docx_v1.py",
        "Reproducibility/src/build_corpus_reconciliation_v1.py",
        "Reproducibility/src/build_repertoire_robustness_inference_v1.py",
        "Reproducibility/src/build_retrieval_inductive_sensitivity_v1.py",
        "Reproducibility/src/build_ner_released_claim_audit_v1.py",
        "Reproducibility/src/normalize_public_text_v1.py",
        "Reproducibility/src/restore_committed_bytes_v1.py",
        "Reproducibility/src/update_public_result_manifests_v1.py",
        "Reproducibility/src/validate_public_release_integrity_v1.py",
        "Reproducibility/tools/check_manuscript_derivatives.py",
        "Results/repertoire-network-v1/robustness/analysis_summary.json",
        "Results/repertoire-network-v1/robustness/diagnostic_summary.csv",
        "Results/repertoire-network-v1/robustness/METHOD.md",
        "Results/repertoire-network-v1/robustness/validation.json",
        "Results/retrieval-inductive-sensitivity-v1/analysis_summary.json",
        "Results/retrieval-inductive-sensitivity-v1/METHOD.md",
        "Results/retrieval-inductive-sensitivity-v1/validation.json",
        "Results/ner-v1/released_claim_audit_status.json",
    } | {f"Results/corpus-reconciliation-v1/{name}" for name in CORPUS_RECONCILIATION_FILES}
    if missing_required := sorted(required_paths - actual_paths):
        raise AssertionError(f"Desktop release omits required integrity or robustness files: {missing_required}")
    for name in ("LICENSE", "LICENSE-CODE"):
        if (target / name).read_bytes() != (ROOT / name).read_bytes():
            raise AssertionError(f"Desktop release {name} is not byte-identical to the repository licence")

    checked += verify_local_links(target / "START_HERE.html", target, r'href="([^"]+)"')
    checked += verify_local_links(target / "PROJECT_README.md", target, r"\]\(([^)]+)\)")
    submission_readme = (target / "Submission_DSH" / "README_BEFORE_SUBMISSION.md").read_text(encoding="utf-8")
    stale_licence_action = "exact ai-tool/model disclosure, repository licence" in submission_readme.lower()
    if "../../figures" in submission_readme or stale_licence_action:
        raise AssertionError("Desktop submission README contains a stale repository-relative path or licence action")

    tiffs = sorted(path.relative_to(target).as_posix() for path in target.rglob("fig[1-4].tif"))
    expected = [f"Figures/fig{number}.tif" for number in range(1, 5)]
    if tiffs != expected:
        raise AssertionError(f"Desktop release TIFF set is not canonical and unique: {tiffs}")
    archive = target.with_suffix(".zip")
    if not archive.is_file():
        raise AssertionError(f"Desktop ZIP is missing: {archive}")
    with zipfile.ZipFile(archive) as package:
        corrupt = package.testzip()
        if corrupt:
            raise AssertionError(f"Desktop ZIP contains a corrupt member: {corrupt}")
        archive_files = [name for name in package.namelist() if not name.endswith("/")]
        expected_archive_files = {
            f"{target.name}/{relative}"
            for relative in actual_paths | {manifest_path.relative_to(target).as_posix()}
        }
        if len(archive_files) != len(set(archive_files)) or set(archive_files) != expected_archive_files:
            raise AssertionError("Desktop ZIP members do not exactly match the validated desktop directory")
    return checked + len(required_paths) + 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desktop", type=Path, help="Also validate a generated desktop package and its ZIP")
    args = parser.parse_args()

    declarations = verify_repository_manifests() + verify_internal_hashes()
    core_contract_checks = verify_core_manifest_contract()
    manuscript_derivative_checks = verify_manuscript_derivative_provenance()
    corpus_reconciliation_checks = verify_corpus_reconciliation_claims()
    robustness_checks = verify_robustness_claims()
    retrieval_sensitivity_checks = verify_retrieval_sensitivity_claims()
    ner_released_claim_audit_checks = verify_ner_released_claim_audit()
    text_files, json_files, csv_files = verify_text_and_syntax()
    desktop_checks = verify_desktop_package(args.desktop) if args.desktop else 0
    print(
        json.dumps(
            {
                "status": "pass",
                "manifest_and_internal_hash_declarations": declarations,
                "core_manifest_contract_checks": core_contract_checks,
                "manuscript_derivative_checks": manuscript_derivative_checks,
                "corpus_reconciliation_checks": corpus_reconciliation_checks,
                "robustness_checks": robustness_checks,
                "retrieval_sensitivity_checks": retrieval_sensitivity_checks,
                "ner_released_claim_audit_checks": ner_released_claim_audit_checks,
                "normalized_text_files": text_files,
                "parsed_json_files": json_files,
                "parsed_csv_files": csv_files,
                "desktop_checks": desktop_checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
