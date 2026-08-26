#!/usr/bin/env python3
"""Independently validate the private clean-text BGE-M3 embedding artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from canonical_semantic_embeddings_v1 import (
    EXPECTED_DIMENSIONS,
    L2_TOLERANCE,
    CanonicalSemanticArtifactError,
    load_verified_embedding_artifact,
    sha256_file,
    sha256_text,
)
from run_private_rap_topic_search_canonical_v2 import CanonicalTopicSearchError, load_canonical_corpus


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ID = "private-canonical-clean-text-embeddings-v1"
TEXT_SIDECAR_ID = "canonical-lyric-text-sidecar-v1"
PRIVATE_DIR = ROOT / "work" / "private-canonical-clean-text-embeddings-v1"
PUBLIC_DIR = ROOT / "outputs" / "canonical-clean-text-embeddings-v1"
CONTRACT_FILE = PRIVATE_DIR / "canonical_clean_text_embedding_contract_v1.json"
VECTOR_FILE = PRIVATE_DIR / "canonical_clean_text_bge_m3_embeddings_v1.npy"
ROW_MAP_FILE = PRIVATE_DIR / "canonical_clean_text_embedding_row_map_v1.csv"
PRIVATE_VALIDATION = PRIVATE_DIR / "validation.json"
PUBLIC_MANIFEST = PUBLIC_DIR / "manifest.json"
PUBLIC_VALIDATION = PUBLIC_DIR / "validation.json"
TEXT_SIDECAR_POINTER = ROOT / "outputs" / TEXT_SIDECAR_ID / "private_sidecar_pointer.json"
TEXT_SIDECAR_VALIDATION = ROOT / "outputs" / TEXT_SIDECAR_ID / "validation.json"

EXPECTED_CLEAN_CHUNKS = 21553
EXPECTED_CLEAN_SONGS = 7206
PRIVATE_ALLOWLIST = {VECTOR_FILE.name, ROW_MAP_FILE.name, CONTRACT_FILE.name, PRIVATE_VALIDATION.name}
PUBLIC_ALLOWLIST = {PUBLIC_MANIFEST.name, PUBLIC_VALIDATION.name}
ROW_MAP_COLUMNS = (
    "clean_row_index", "song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256",
    "analysis_text_weight", "embedding_provenance", "base_canonical_vector_index",
    "base_canonical_vector_sha256", "reembed_reason",
)
REUSED = "reused_verified_canonical_vector_exact_clean_text_sha"
REEMBEDDED = "reembedded_local_bge_m3_clean_text"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(payload_text)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def validate() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for path, allowlist, label in ((PRIVATE_DIR, PRIVATE_ALLOWLIST, "private"), (PUBLIC_DIR, PUBLIC_ALLOWLIST, "public")):
        actual = {item.name for item in path.iterdir()} if path.is_dir() else set()
        checks.append(
            {
                "name": f"{label}_allowlist",
                "passed": path.is_dir() and actual == allowlist and not any(item.is_dir() for item in path.iterdir()),
                "detail": {"missing": sorted(allowlist - actual), "unexpected": sorted(actual - allowlist)},
            }
        )
    if not all(check["passed"] for check in checks):
        return {"artifact_id": ARTIFACT_ID, "passed": False, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks}

    contract = read_json(CONTRACT_FILE)
    public_manifest = read_json(PUBLIC_MANIFEST)
    pointer = read_json(TEXT_SIDECAR_POINTER)
    text_validation = read_json(TEXT_SIDECAR_VALIDATION)
    checks.append(
        {
            "name": "artifact_and_sidecar_identity",
            "passed": contract.get("artifact_id") == ARTIFACT_ID
            and public_manifest.get("artifact_id") == ARTIFACT_ID
            and contract.get("clean_text_sidecar", {}).get("artifact_id") == TEXT_SIDECAR_ID
            and pointer.get("artifact_id") == TEXT_SIDECAR_ID
            and text_validation.get("status") == "pass",
        }
    )
    try:
        corpus = load_canonical_corpus()
        base = load_verified_embedding_artifact(corpus)
        base_loaded = True
    except (CanonicalTopicSearchError, CanonicalSemanticArtifactError):
        base_loaded = False
        corpus = None
        base = None
    checks.append({"name": "canonical_corpus_and_base_embedding_validate", "passed": base_loaded})
    if not base_loaded:
        return {"artifact_id": ARTIFACT_ID, "passed": False, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks}
    assert corpus is not None and base is not None

    clean_dir = (ROOT / str(pointer["private_directory"])).resolve()
    clean_path = clean_dir / str(pointer["cleaned_chunk_file"])
    clean_hash = sha256_file(clean_path)
    checks.append(
        {
            "name": "clean_text_input_hash_current",
            "passed": clean_hash == pointer.get("cleaned_chunk_file_sha256")
            and contract.get("clean_text_sidecar", {}).get("manifest_sha256") == sha256_file(ROOT / "outputs" / TEXT_SIDECAR_ID / "manifest.json"),
        }
    )
    clean_fields, clean_records = read_csv(clean_path)
    required_clean = {"song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text", "analysis_text_sha256", "analysis_text_status", "analysis_text_weight"}
    checks.append({"name": "clean_text_sidecar_schema", "passed": required_clean.issubset(clean_fields)})
    clean_by_key = {(row["song_id"], row["chunk_id"]): row for row in clean_records}
    active_clean_records = [row for row in clean_records if row.get("analysis_text_status") == "eligible_clean_text"]
    checks.append(
        {
            "name": "clean_text_population_reconciles",
            "passed": len(clean_by_key) == len(corpus.eligible_rows)
            and len(active_clean_records) == EXPECTED_CLEAN_CHUNKS
            and len({row["song_id"] for row in active_clean_records}) == EXPECTED_CLEAN_SONGS,
        }
    )

    matrix = np.load(VECTOR_FILE, mmap_mode="r", allow_pickle=False)
    map_fields, map_rows = read_csv(ROW_MAP_FILE)
    checks.append(
        {
            "name": "private_contract_file_hashes",
            "passed": contract.get("vector_file", {}).get("sha256") == sha256_file(VECTOR_FILE)
            and contract.get("row_map_file", {}).get("sha256") == sha256_file(ROW_MAP_FILE),
        }
    )
    checks.append(
        {
            "name": "matrix_shape_dtype_finite_l2",
            "passed": tuple(matrix.shape) == (EXPECTED_CLEAN_CHUNKS, EXPECTED_DIMENSIONS)
            and str(matrix.dtype) == "float32"
            and bool(np.isfinite(matrix).all())
            and bool(np.all(np.abs(np.linalg.norm(matrix, axis=1) - 1.0) <= L2_TOLERANCE)),
        }
    )
    checks.append(
        {
            "name": "row_map_schema_and_cardinality",
            "passed": map_fields == ROW_MAP_COLUMNS and len(map_rows) == EXPECTED_CLEAN_CHUNKS,
        }
    )

    alignment_ok = True
    reuse_exact_ok = True
    reembed_ok = True
    reused = 0
    reembedded = 0
    for expected_index, row in enumerate(map_rows):
        try:
            clean_index = int(row["clean_row_index"])
            base_index = int(row["base_canonical_vector_index"])
            weight = float(row["analysis_text_weight"])
        except (TypeError, ValueError):
            alignment_ok = False
            continue
        if clean_index != expected_index or base_index < 0 or base_index >= len(corpus.eligible_rows):
            alignment_ok = False
            continue
        canonical = corpus.eligible_rows[base_index]
        clean = clean_by_key.get((row["song_id"], row["chunk_id"]))
        if clean is None:
            alignment_ok = False
            continue
        alignment_ok = alignment_ok and (
            canonical.song_id == row["song_id"]
            and canonical.chunk_id == row["chunk_id"]
            and canonical.canonical_text_sha256 == row["canonical_lyric_text_sha256"]
            and clean.get("canonical_lyric_text_sha256") == row["canonical_lyric_text_sha256"]
            and clean.get("analysis_text_sha256") == row["analysis_text_sha256"]
            and clean.get("analysis_text_status") == "eligible_clean_text"
            and math.isclose(weight, canonical.analysis_weight, rel_tol=0.0, abs_tol=1e-12)
            and sha256_text(clean.get("analysis_text", "")) == row["analysis_text_sha256"]
            and row["base_canonical_vector_sha256"] == base.vector_sha256
        )
        provenance = row["embedding_provenance"]
        if provenance == REUSED:
            reused += 1
            reuse_exact_ok = reuse_exact_ok and (
                row["analysis_text_sha256"] == row["canonical_lyric_text_sha256"]
                and row["reembed_reason"] == ""
                and bool(np.array_equal(matrix[clean_index], base.vector_matrix[base_index]))
            )
        elif provenance == REEMBEDDED:
            reembedded += 1
            reembed_ok = reembed_ok and (
                row["analysis_text_sha256"] != row["canonical_lyric_text_sha256"]
                and row["reembed_reason"] == "clean_text_sha_mismatch"
            )
        else:
            alignment_ok = False
    checks.append({"name": "row_map_exactly_aligns_clean_text_and_canonical_keys", "passed": alignment_ok})
    checks.append({"name": "reused_rows_are_bit_identical_to_verified_base_vectors", "passed": reuse_exact_ok})
    checks.append({"name": "reembedded_rows_have_clean_text_mismatch_provenance", "passed": reembed_ok})
    declared = contract.get("incremental_provenance", {})
    checks.append(
        {
            "name": "provenance_counts_complete",
            "passed": reused + reembedded == EXPECTED_CLEAN_CHUNKS
            and declared.get("reused_exact_clean_text_rows") == reused
            and declared.get("reembedded_changed_clean_text_rows") == reembedded
            and declared.get("metadata_only_rows_excluded") == len(corpus.eligible_rows) - EXPECTED_CLEAN_CHUNKS,
            "detail": {"reused": reused, "reembedded": reembedded},
        }
    )
    public_keys_ok = all(
        key not in json.dumps(public_manifest, ensure_ascii=False)
        for key in ("song_id", "chunk_id", "canonical_artist", "canonical_song_title", "analysis_text")
    )
    checks.append({"name": "public_manifest_excludes_private_identifiers_and_lyric_text", "passed": public_keys_ok})
    passed = all(bool(check.get("passed")) for check in checks)
    return {
        "artifact_id": ARTIFACT_ID,
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


def main() -> int:
    try:
        result = validate()
    except Exception as exc:
        result = {
            "artifact_id": ARTIFACT_ID,
            "passed": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": [{"name": "validator_runtime", "passed": False, "detail": str(exc)}],
        }
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(PRIVATE_VALIDATION, result)
    public_result = {
        "artifact_id": ARTIFACT_ID,
        "passed": result["passed"],
        "checks": [{"name": check["name"], "passed": check["passed"]} for check in result["checks"]],
    }
    atomic_write_json(PUBLIC_VALIDATION, public_result)
    try:
        manifest = read_json(PUBLIC_MANIFEST)
        manifest["validation"] = {"passed": result["passed"], "check_count": len(result["checks"])}
        atomic_write_json(PUBLIC_MANIFEST, manifest)
    except Exception:
        pass
    print(f"Validation {'pass' if result['passed'] else 'fail'}: {sum(bool(check.get('passed')) for check in result['checks'])}/{len(result['checks'])} checks passed.")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
