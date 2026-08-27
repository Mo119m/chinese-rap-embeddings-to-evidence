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
CORE_REQUIRED_PATHS = {
    ".gitattributes",
    ".github/workflows/release-integrity.yml",
    ".python-version",
    "LICENSE",
    "LICENSE-CODE",
    "README.md",
    "requirements.txt",
    "src/build_chinese_rap_release_v4.py",
    "src/build_ner_released_claim_audit_v1.py",
    "src/build_repertoire_robustness_inference_v1.py",
    "src/build_retrieval_inductive_sensitivity_v1.py",
    "src/normalize_public_text_v1.py",
    "src/restore_committed_bytes_v1.py",
    "src/update_public_result_manifests_v1.py",
    "src/validate_public_release_integrity_v1.py",
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
    "results/ner-v1/released_claim_audit_status.json",
}


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
        "Reproducibility/.python-version",
        "Reproducibility/requirements.txt",
        "Reproducibility/src/build_repertoire_robustness_inference_v1.py",
        "Reproducibility/src/build_retrieval_inductive_sensitivity_v1.py",
        "Reproducibility/src/build_ner_released_claim_audit_v1.py",
        "Reproducibility/src/normalize_public_text_v1.py",
        "Reproducibility/src/restore_committed_bytes_v1.py",
        "Reproducibility/src/update_public_result_manifests_v1.py",
        "Reproducibility/src/validate_public_release_integrity_v1.py",
        "Results/repertoire-network-v1/robustness/analysis_summary.json",
        "Results/repertoire-network-v1/robustness/diagnostic_summary.csv",
        "Results/repertoire-network-v1/robustness/METHOD.md",
        "Results/repertoire-network-v1/robustness/validation.json",
        "Results/retrieval-inductive-sensitivity-v1/analysis_summary.json",
        "Results/retrieval-inductive-sensitivity-v1/METHOD.md",
        "Results/retrieval-inductive-sensitivity-v1/validation.json",
        "Results/ner-v1/released_claim_audit_status.json",
    }
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
