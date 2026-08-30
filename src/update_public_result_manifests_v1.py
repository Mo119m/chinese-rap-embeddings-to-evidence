#!/usr/bin/env python3
"""Refresh whitelisted public result manifests after byte-only maintenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PUBLIC_LINEAGE_PATHS = [
    "src/build_chinese_rap_downstream_figures_v1.py",
    "src/build_chinese_rap_figure_3_v1.py",
    "src/build_chinese_rap_journal_figures_v1.py",
    "results/input-audit-v1/analysis_summary.json",
    "results/retrieval-v1/analysis_summary.json",
    "results/retrieval-v1/metrics.csv",
    "results/retrieval-v1/uncertainty.csv",
    "results/ner-v1/entity_co_mentions_provisional.csv",
    "results/ner-v1/reconciliation_validation.json",
    "results/ner-v1/release_sensitivity_summary.csv",
    "results/ner-v1/source_label_entity_links_provisional.csv",
    "results/ner-v1/summary.json",
    "results/ner-v1/validation.json",
    "results/written-rhyme-v1/analysis_summary.json",
    "results/written-rhyme-v1/model_metrics.csv",
    "results/written-rhyme-v1/paired_model_deltas.csv",
    "results/written-rhyme-v1/stratified_metrics.csv",
    "methods/RESEARCH_CONTRACT.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def file_record(path: Path, relative: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if relative is not None:
        record = {"path": relative, **record}
    return record


def refresh_mapping(manifest_path: Path, key: str) -> int:
    manifest = read_json(manifest_path)
    directory = manifest_path.parent
    records = manifest[key]
    if isinstance(records, dict):
        for relative, existing in list(records.items()):
            path = directory / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            if not isinstance(existing, dict):
                raise TypeError(f"Manifest record is not an object: {manifest_path}:{relative}")
            refreshed = dict(existing)
            refreshed.update(file_record(path))
            records[relative] = refreshed
    elif isinstance(records, list):
        seen: set[str] = set()
        for index, existing in enumerate(records):
            if not isinstance(existing, dict) or not isinstance(existing.get("path"), str):
                raise TypeError(f"Manifest list record lacks a string path: {manifest_path}:{index}")
            relative = existing["path"]
            if relative in seen:
                raise ValueError(f"Manifest list contains a duplicate path: {manifest_path}:{relative}")
            seen.add(relative)
            path = directory / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            refreshed = dict(existing)
            refreshed.update(file_record(path))
            records[index] = refreshed
    else:
        raise TypeError(
            f"Unsupported manifest record collection for {manifest_path}:{key}: "
            f"{type(records).__name__}"
        )
    write_json(manifest_path, manifest)
    return len(records)


def refresh_repertoire_parent() -> int:
    path = ROOT / "results" / "repertoire-network-v1" / "manifest.json"
    manifest = read_json(path)
    robustness_dir = path.parent / "robustness"
    if robustness_dir.is_dir():
        robustness_manifest = read_json(robustness_dir / "manifest.json")
        robustness_validation = read_json(robustness_dir / "validation.json")
        robustness_summary = read_json(robustness_dir / "analysis_summary.json")
        if robustness_validation.get("status") != "pass":
            raise RuntimeError("Repertoire robustness validation is not passing")
        manifest["components"]["robustness"] = {
            "artifact_id": robustness_manifest["artifact_id"],
            "manifest_sha256": sha256(robustness_dir / "manifest.json"),
            "validation_sha256": sha256(robustness_dir / "validation.json"),
        }

        root_validation_path = path.parent / manifest["validation"]["file"]
        root_validation = read_json(root_validation_path)
        check = {
            "name": "graph_null_and_projection_fidelity_component_pass",
            "passed": True,
            "observed": {
                "null_replicates": robustness_summary["graph_alignment_null"]["null_replicates"],
                "released_edges": robustness_summary["projection_fidelity"]["released_edges"],
                "pca_population": robustness_summary["projection_fidelity"]["population"],
            },
        }
        checks = [item for item in root_validation["checks"] if item.get("name") != check["name"]]
        checks.append(check)
        root_validation["checks"] = checks
        root_validation["status"] = "pass" if all(item.get("passed") for item in checks) else "fail"
        root_validation["validated_with"] = [
            "src/build_chinese_rap_release_site_data_v1.py",
            "src/build_repertoire_robustness_inference_v1.py",
        ]
        write_json(root_validation_path, root_validation)
    checked = 0
    for name, component in manifest["components"].items():
        directory = path.parent / name
        component["manifest_sha256"] = sha256(directory / "manifest.json")
        component["validation_sha256"] = sha256(directory / "validation.json")
        checked += 2
    validation = path.parent / manifest["validation"]["file"]
    manifest["validation"].update(file_record(validation))
    write_json(path, manifest)
    return checked + 1


def refresh_figure_manifest() -> int:
    path = ROOT / "figures" / "manifest.json"
    manifest = read_json(path)
    if "historical_render_lineage" not in manifest:
        manifest["historical_render_lineage"] = {
            "status": "historical_build_workspace_not_fully_public",
            "note": "Preserved for provenance only. These original paths are not presented as public, checkout-verifiable lineage.",
            "records": manifest.get("lineage", {}),
        }
    manifest["lineage"] = {
        "status": "public_checkout_verifiable",
        "note": "Published sources are value-equivalent promotions of the aggregate inputs used for the historical render; paths, bytes, and hashes below resolve in this repository.",
        "public_files": [file_record(ROOT / relative, relative) for relative in FIGURE_PUBLIC_LINEAGE_PATHS],
    }
    refreshed_files = []
    for record in manifest["files"]:
        relative = record["path"]
        refreshed_files.append(file_record(ROOT / relative, relative))
    manifest["files"] = refreshed_files
    write_json(path, manifest)
    return len(refreshed_files) + len(FIGURE_PUBLIC_LINEAGE_PATHS)


def main() -> None:
    mappings = [
        (ROOT / "results" / "retrieval-v1" / "manifest.json", "files"),
        (ROOT / "results" / "ner-v1" / "manifest.json", "files"),
        (ROOT / "results" / "written-rhyme-v1" / "manifest.json", "output_files"),
        (ROOT / "results" / "corpus-reconciliation-v1" / "manifest.json", "files"),
        (ROOT / "results" / "repertoire-network-v1" / "graph" / "manifest.json", "output_files"),
        (ROOT / "results" / "repertoire-network-v1" / "profiles" / "manifest.json", "files"),
        (ROOT / "results" / "repertoire-network-v1" / "bootstrap" / "manifest.json", "files"),
        (ROOT / "results" / "repertoire-network-v1" / "robustness" / "manifest.json", "output_files"),
    ]
    refreshed = sum(refresh_mapping(path, key) for path, key in mappings)
    refreshed += refresh_repertoire_parent()
    refreshed += refresh_figure_manifest()
    print(json.dumps({"status": "pass", "refreshed_records": refreshed}, indent=2))


if __name__ == "__main__":
    main()
