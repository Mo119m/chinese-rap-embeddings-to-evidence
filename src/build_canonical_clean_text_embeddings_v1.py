#!/usr/bin/env python3
"""Build a BGE-M3 embedding artifact for the cleaned lyric-text sidecar.

The immutable canonical BGE-M3 matrix remains untouched.  This derivative uses
that verified matrix only when a cleaned-text row is byte-for-byte unchanged.
Every changed but still eligible clean-text row is encoded again by the same
local, offline BGE-M3 model.  Header-only chunks receive no vector and are
explicitly excluded from the clean analysis population.

Run from the repository root:
    work\\semantic-ml-venv\\Scripts\\python.exe work\\build_canonical_clean_text_embeddings_v1.py
    work\\semantic-ml-venv\\Scripts\\python.exe work\\validate_canonical_clean_text_embeddings_v1.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from canonical_semantic_embeddings_v1 import (
    EXPECTED_DIMENSIONS,
    L2_TOLERANCE,
    MODEL_BINARY,
    MODEL_CONFIG,
    MODEL_DIR,
    CanonicalSemanticArtifactError,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npy,
    encode_dense_with_local_model,
    load_local_bge_m3_model,
    load_verified_embedding_artifact,
    sha256_file,
    sha256_text,
)
from run_private_rap_topic_search_canonical_v2 import CanonicalTopicSearchError, load_canonical_corpus


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ID = "private-canonical-clean-text-embeddings-v1"
ARTIFACT_VERSION = "1.0.0"
TEXT_SIDECAR_ID = "canonical-lyric-text-sidecar-v1"

TEXT_SIDECAR_DIR = ROOT / "outputs" / TEXT_SIDECAR_ID
TEXT_SIDECAR_POINTER = TEXT_SIDECAR_DIR / "private_sidecar_pointer.json"
TEXT_SIDECAR_MANIFEST = TEXT_SIDECAR_DIR / "manifest.json"
TEXT_SIDECAR_VALIDATION = TEXT_SIDECAR_DIR / "validation.json"

PRIVATE_DIR = ROOT / "work" / "private-canonical-clean-text-embeddings-v1"
VECTOR_FILE = PRIVATE_DIR / "canonical_clean_text_bge_m3_embeddings_v1.npy"
ROW_MAP_FILE = PRIVATE_DIR / "canonical_clean_text_embedding_row_map_v1.csv"
CONTRACT_FILE = PRIVATE_DIR / "canonical_clean_text_embedding_contract_v1.json"
PRIVATE_VALIDATION_FILE = PRIVATE_DIR / "validation.json"
PUBLIC_DIR = ROOT / "outputs" / "canonical-clean-text-embeddings-v1"
PUBLIC_MANIFEST_FILE = PUBLIC_DIR / "manifest.json"
PUBLIC_VALIDATION_FILE = PUBLIC_DIR / "validation.json"

EXPECTED_CLEAN_CHUNKS = 21553
EXPECTED_CLEAN_SONGS = 7206

PRIVATE_ALLOWLIST = {
    VECTOR_FILE.name,
    ROW_MAP_FILE.name,
    CONTRACT_FILE.name,
    PRIVATE_VALIDATION_FILE.name,
}
PUBLIC_ALLOWLIST = {PUBLIC_MANIFEST_FILE.name, PUBLIC_VALIDATION_FILE.name}

ROW_MAP_COLUMNS = (
    "clean_row_index",
    "song_id",
    "chunk_id",
    "canonical_lyric_text_sha256",
    "analysis_text_sha256",
    "analysis_text_weight",
    "embedding_provenance",
    "base_canonical_vector_index",
    "base_canonical_vector_sha256",
    "reembed_reason",
)
REUSED_PROVENANCE = "reused_verified_canonical_vector_exact_clean_text_sha"
REEMBEDDED_PROVENANCE = "reembedded_local_bge_m3_clean_text"


class CleanEmbeddingBuildError(RuntimeError):
    """Fail closed whenever clean text or vector lineage is incomplete."""


@dataclass(frozen=True)
class CleanTextRow:
    clean_index: int
    song_id: str
    chunk_id: str
    canonical_text_sha256: str
    analysis_text: str
    analysis_text_sha256: str
    analysis_weight: float
    base_index: int


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CleanEmbeddingBuildError(f"Missing {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanEmbeddingBuildError(f"Could not read {label}.") from exc
    if not isinstance(data, dict):
        raise CleanEmbeddingBuildError(f"{label} must be a JSON object.")
    return data


def require_allowlist(directory: Path, allowlist: set[str], label: str) -> None:
    if not directory.exists():
        return
    if not directory.is_dir():
        raise CleanEmbeddingBuildError(f"{label} is not a directory.")
    actual = {path.name for path in directory.iterdir()}
    extras = sorted(actual - allowlist)
    nested = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if extras or nested:
        raise CleanEmbeddingBuildError(f"{label} has undeclared content: {', '.join(extras + nested)}")


def resolve_private_sidecar() -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    manifest = read_json(TEXT_SIDECAR_MANIFEST, "clean text sidecar manifest")
    pointer = read_json(TEXT_SIDECAR_POINTER, "clean text sidecar pointer")
    validation = read_json(TEXT_SIDECAR_VALIDATION, "clean text sidecar validation")
    if manifest.get("artifact_id") != TEXT_SIDECAR_ID or pointer.get("artifact_id") != TEXT_SIDECAR_ID:
        raise CleanEmbeddingBuildError("Clean text sidecar identity is inconsistent.")
    if validation.get("status") != "pass":
        raise CleanEmbeddingBuildError("Clean text sidecar has not passed independent validation.")
    if pointer.get("classification") != "private_local_only_cleaned_full_lyric_text":
        raise CleanEmbeddingBuildError("Clean text sidecar is not marked private-local-only.")
    directory_raw = pointer.get("private_directory")
    chunk_name = pointer.get("cleaned_chunk_file")
    audit_name = pointer.get("audit_file")
    if not all(isinstance(value, str) and value for value in (directory_raw, chunk_name, audit_name)):
        raise CleanEmbeddingBuildError("Clean text sidecar pointer lacks controlled filenames.")
    private_dir = (ROOT / str(directory_raw)).resolve()
    chunks_path = (private_dir / str(chunk_name)).resolve()
    audit_path = (private_dir / str(audit_name)).resolve()
    try:
        chunks_path.relative_to(private_dir)
        audit_path.relative_to(private_dir)
        private_dir.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise CleanEmbeddingBuildError("Clean text sidecar pointer escapes a controlled directory.") from exc
    if sha256_file(chunks_path) != pointer.get("cleaned_chunk_file_sha256"):
        raise CleanEmbeddingBuildError("Cleaned lyric sidecar CSV hash mismatch.")
    if sha256_file(audit_path) != pointer.get("audit_file_sha256"):
        raise CleanEmbeddingBuildError("Cleaned lyric audit CSV hash mismatch.")
    return chunks_path, audit_path, manifest, pointer


def load_clean_rows(corpus: Any) -> tuple[tuple[CleanTextRow, ...], int, dict[str, Any]]:
    cleaned_path, _, sidecar_manifest, _ = resolve_private_sidecar()
    required_columns = {
        "analysis_text_sidecar_id", "cleaning_version", "song_id", "chunk_id",
        "canonical_lyric_text_sha256", "analysis_text", "analysis_text_sha256",
        "analysis_text_status", "analysis_text_weight",
    }
    with cleaned_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not required_columns.issubset(fields):
            raise CleanEmbeddingBuildError(f"Cleaned lyric sidecar lacks columns: {sorted(required_columns - fields)}")
        raw_rows = list(reader)
    if len(raw_rows) != len(corpus.eligible_rows):
        raise CleanEmbeddingBuildError("Cleaned lyric sidecar does not retain the full canonical eligible row population.")
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for raw in raw_rows:
        key = (str(raw.get("song_id", "")), str(raw.get("chunk_id", "")))
        if not all(key) or key in by_key:
            raise CleanEmbeddingBuildError("Cleaned lyric sidecar contains an empty or duplicate key.")
        if raw.get("analysis_text_sidecar_id") != TEXT_SIDECAR_ID:
            raise CleanEmbeddingBuildError("Cleaned lyric sidecar row carries an unexpected artifact ID.")
        by_key[key] = raw

    active: list[CleanTextRow] = []
    withheld_metadata_only = 0
    seen_clean_song_ids: set[str] = set()
    duplicate_status_by_group: dict[str, set[str]] = {}
    for base_index, canonical_row in enumerate(corpus.eligible_rows):
        if canonical_row.index != base_index:
            raise CleanEmbeddingBuildError("Canonical corpus eligible order is not contiguous.")
        raw = by_key.get((canonical_row.song_id, canonical_row.chunk_id))
        if raw is None:
            raise CleanEmbeddingBuildError("A canonical eligible chunk has no clean-text sidecar row.")
        if raw.get("canonical_lyric_text_sha256") != canonical_row.canonical_text_sha256:
            raise CleanEmbeddingBuildError("Clean-text sidecar does not match canonical lyric text hash.")
        try:
            weight = float(raw.get("analysis_text_weight", ""))
        except ValueError as exc:
            raise CleanEmbeddingBuildError("Clean-text sidecar has a nonnumeric duplicate weight.") from exc
        if not math.isclose(weight, float(canonical_row.analysis_weight), rel_tol=0.0, abs_tol=1e-12):
            raise CleanEmbeddingBuildError("Clean-text sidecar duplicate weight differs from canonical corpus.")
        status = raw.get("analysis_text_status")
        group = canonical_row.duplicate_group_id
        if group:
            duplicate_status_by_group.setdefault(group, set()).add(str(status))
        if status == "withheld_metadata_only":
            if raw.get("analysis_text") or raw.get("analysis_text_sha256"):
                raise CleanEmbeddingBuildError("Metadata-only sidecar row must not retain analysis text or a clean hash.")
            withheld_metadata_only += 1
            continue
        if status != "eligible_clean_text":
            raise CleanEmbeddingBuildError(f"Unknown clean-text status: {status!r}")
        text = str(raw.get("analysis_text", ""))
        clean_sha = str(raw.get("analysis_text_sha256", ""))
        if not text.strip() or sha256_text(text) != clean_sha:
            raise CleanEmbeddingBuildError("Eligible clean text is empty or does not match its SHA-256.")
        active.append(
            CleanTextRow(
                clean_index=len(active),
                song_id=canonical_row.song_id,
                chunk_id=canonical_row.chunk_id,
                canonical_text_sha256=canonical_row.canonical_text_sha256,
                analysis_text=text,
                analysis_text_sha256=clean_sha,
                analysis_weight=weight,
                base_index=base_index,
            )
        )
        seen_clean_song_ids.add(canonical_row.song_id)
    if set(by_key) != {(row.song_id, row.chunk_id) for row in corpus.eligible_rows}:
        raise CleanEmbeddingBuildError("Clean-text sidecar key population differs from the canonical eligible population.")
    # A cross-song duplicate group must either remain intact or be withheld as
    # a whole group.  This preserves the canonical duplicate-weight algebra.
    partial_groups = sorted(group for group, statuses in duplicate_status_by_group.items() if len(statuses) != 1)
    if partial_groups:
        raise CleanEmbeddingBuildError("A canonical duplicate group is only partially kept after header cleaning.")
    if len(active) != EXPECTED_CLEAN_CHUNKS or len(seen_clean_song_ids) != EXPECTED_CLEAN_SONGS:
        raise CleanEmbeddingBuildError(
            f"Clean analysis population changed: {len(active)} chunks / {len(seen_clean_song_ids)} songs; "
            "change the declared sidecar protocol deliberately before rebuilding vectors."
        )
    return tuple(active), withheld_metadata_only, sidecar_manifest


def write_private_contract(
    corpus: Any,
    base: Any,
    sidecar_manifest: dict[str, Any],
    clean_rows: tuple[CleanTextRow, ...],
    withheld_metadata_only: int,
    reused: int,
    reembedded: int,
) -> dict[str, Any]:
    vector_sha = sha256_file(VECTOR_FILE)
    row_map_sha = sha256_file(ROW_MAP_FILE)
    clean_order_digest = hashlib.sha256()
    for row in clean_rows:
        clean_order_digest.update(
            f"{row.clean_index}\t{row.song_id}\t{row.chunk_id}\t{row.canonical_text_sha256}\t"
            f"{row.analysis_text_sha256}\t{row.analysis_weight:.17g}\n".encode("utf-8")
        )
    return {
        "artifact_id": ARTIFACT_ID,
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "complete_contract": True,
        "privacy": "private_local_only_cleaned_text_vectors_and_row_mapping",
        "canonical_input": {
            "canonical_corpus_id": corpus.corpus_id,
            "canonical_contract_version": corpus.contract_version,
            "canonical_contract_sha256": corpus.contract_sha256,
            "private_content_sha256": corpus.private_content_sha256,
            "canonical_eligible_chunk_rows": len(corpus.eligible_rows),
            "canonical_withheld_rows": corpus.withheld_rows,
        },
        "clean_text_sidecar": {
            "artifact_id": sidecar_manifest["artifact_id"],
            "cleaning_version": sidecar_manifest["cleaning_version"],
            "manifest_sha256": sha256_file(TEXT_SIDECAR_MANIFEST),
            "cleaned_eligible_rows": len(clean_rows),
            "metadata_only_rows_excluded": withheld_metadata_only,
            "clean_song_count": EXPECTED_CLEAN_SONGS,
            "clean_row_order_sha256": clean_order_digest.hexdigest(),
        },
        "base_canonical_embedding": {
            "artifact_id": base.artifact_id,
            "artifact_version": base.artifact_version,
            "private_contract_sha256": base.contract_sha256,
            "vector_sha256": base.vector_sha256,
            "row_map_sha256": base.row_map_sha256,
        },
        "vector_file": {
            "filename": VECTOR_FILE.name,
            "sha256": vector_sha,
            "rows": len(clean_rows),
            "dimensions": EXPECTED_DIMENSIONS,
            "dtype": "float32",
            "l2_normalized": True,
        },
        "row_map_file": {"filename": ROW_MAP_FILE.name, "sha256": row_map_sha, "rows": len(clean_rows)},
        "incremental_provenance": {
            "reused_exact_clean_text_rows": reused,
            "reembedded_changed_clean_text_rows": reembedded,
            "reuse_rule": "exact canonical song_id + chunk_id + original SHA and clean_text_sha256 equals canonical text SHA",
            "metadata_only_rows_excluded": withheld_metadata_only,
        },
        "model": {
            "implementation": "FlagEmbedding.BGEM3FlagModel dense vectors",
            "local_model_relative_path": str(MODEL_DIR.relative_to(ROOT)).replace("\\", "/"),
            "offline_only": True,
            "max_length": 2048,
            "batch_size": 2,
            "pytorch_model_bin_sha256": sha256_file(MODEL_BINARY),
            "config_sha256": sha256_file(MODEL_CONFIG),
        },
        "builder_code_sha256": sha256_file(Path(__file__)),
    }


def build() -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise CleanEmbeddingBuildError("numpy is unavailable.") from exc
    require_allowlist(PRIVATE_DIR, PRIVATE_ALLOWLIST, "clean-text embedding private directory")
    require_allowlist(PUBLIC_DIR, PUBLIC_ALLOWLIST, "clean-text embedding public directory")
    try:
        corpus = load_canonical_corpus()
        base = load_verified_embedding_artifact(corpus)
    except (CanonicalTopicSearchError, CanonicalSemanticArtifactError) as exc:
        raise CleanEmbeddingBuildError("The canonical corpus or base embedding artifact did not validate.") from exc
    clean_rows, metadata_only_rows, sidecar_manifest = load_clean_rows(corpus)

    vectors = np.empty((len(clean_rows), EXPECTED_DIMENSIONS), dtype=np.float32)
    row_map_payload: list[dict[str, Any]] = []
    changed_positions: list[int] = []
    changed_texts: list[str] = []
    reused = 0
    for clean_row in clean_rows:
        base_row = base.row_map[clean_row.base_index]
        if (
            base_row.canonical_row_index != clean_row.base_index
            or base_row.song_id != clean_row.song_id
            or base_row.chunk_id != clean_row.chunk_id
            or base_row.canonical_text_sha256 != clean_row.canonical_text_sha256
            or not math.isclose(base_row.analysis_text_weight, clean_row.analysis_weight, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise CleanEmbeddingBuildError("Verified base embedding row no longer aligns with the canonical clean-text input.")
        if clean_row.analysis_text_sha256 == clean_row.canonical_text_sha256:
            vectors[clean_row.clean_index] = base.vector_matrix[clean_row.base_index]
            provenance = REUSED_PROVENANCE
            reembed_reason = ""
            reused += 1
        else:
            changed_positions.append(clean_row.clean_index)
            changed_texts.append(clean_row.analysis_text)
            provenance = REEMBEDDED_PROVENANCE
            reembed_reason = "clean_text_sha_mismatch"
        row_map_payload.append(
            {
                "clean_row_index": clean_row.clean_index,
                "song_id": clean_row.song_id,
                "chunk_id": clean_row.chunk_id,
                "canonical_lyric_text_sha256": clean_row.canonical_text_sha256,
                "analysis_text_sha256": clean_row.analysis_text_sha256,
                "analysis_text_weight": f"{clean_row.analysis_weight:.17g}",
                "embedding_provenance": provenance,
                "base_canonical_vector_index": clean_row.base_index,
                "base_canonical_vector_sha256": base.vector_sha256,
                "reembed_reason": reembed_reason,
            }
        )
    if changed_texts:
        model = load_local_bge_m3_model()
        changed_vectors = encode_dense_with_local_model(model, changed_texts, batch_size=2)
        for position, vector in zip(changed_positions, changed_vectors, strict=True):
            vectors[position] = vector
    reembedded = len(changed_positions)
    if reused + reembedded != len(clean_rows):
        raise CleanEmbeddingBuildError("Clean-text vector provenance does not cover every eligible clean row.")
    if tuple(vectors.shape) != (len(clean_rows), EXPECTED_DIMENSIONS) or vectors.dtype != np.float32:
        raise CleanEmbeddingBuildError("Clean-text BGE matrix has an unexpected shape or dtype.")
    if not bool(np.isfinite(vectors).all()) or not bool(np.all(np.abs(np.linalg.norm(vectors, axis=1) - 1.0) <= L2_TOLERANCE)):
        raise CleanEmbeddingBuildError("Clean-text BGE matrix has nonfinite or non-normalized rows.")

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_npy(VECTOR_FILE, vectors)
    atomic_write_csv(ROW_MAP_FILE, ROW_MAP_COLUMNS, row_map_payload)
    contract = write_private_contract(
        corpus, base, sidecar_manifest, clean_rows, metadata_only_rows, reused, reembedded
    )
    atomic_write_json(CONTRACT_FILE, contract)
    private_validation = {
        "artifact_id": ARTIFACT_ID,
        "status": "pending_independent_validation",
        "required_validator": "work/validate_canonical_clean_text_embeddings_v1.py",
    }
    atomic_write_json(PRIVATE_VALIDATION_FILE, private_validation)
    public_manifest = {
        "artifact_id": ARTIFACT_ID,
        "artifact_version": ARTIFACT_VERSION,
        "privacy": "generic_output_has_no_lyrics_vectors_or_row_mapping",
        "canonical_input": contract["canonical_input"],
        "clean_text_sidecar": {
            "artifact_id": TEXT_SIDECAR_ID,
            "cleaned_eligible_rows": len(clean_rows),
            "metadata_only_rows_excluded": metadata_only_rows,
            "clean_song_count": EXPECTED_CLEAN_SONGS,
        },
        "counts": {
            "dimensions": EXPECTED_DIMENSIONS,
            "reused_exact_clean_text_rows": reused,
            "reembedded_changed_clean_text_rows": reembedded,
        },
        "private_artifact_hashes": {
            "contract_sha256": sha256_file(CONTRACT_FILE),
            "vector_sha256": sha256_file(VECTOR_FILE),
            "row_map_sha256": sha256_file(ROW_MAP_FILE),
        },
        "validation": {"passed": False, "status": "pending_independent_validation"},
    }
    atomic_write_json(PUBLIC_MANIFEST_FILE, public_manifest)
    atomic_write_json(PUBLIC_VALIDATION_FILE, private_validation)
    return {
        "clean_rows": len(clean_rows),
        "metadata_only_rows": metadata_only_rows,
        "reused": reused,
        "reembedded": reembedded,
    }


def main() -> int:
    try:
        result = build()
    except CleanEmbeddingBuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 2
    print(
        f"Built {ARTIFACT_ID}: {result['clean_rows']} clean chunks; "
        f"{result['reused']} exact reuse, {result['reembedded']} local re-encodes, "
        f"{result['metadata_only_rows']} metadata-only exclusions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
