#!/usr/bin/env python3
"""Build the first reproducible Chinese Rap Cultural Coordinates analysis.

This is deliberately a *text-only* cultural-reference study.  It combines:

* the verified canonical lyric input and its duplicate-text weights;
* a reviewed literal entity lexicon, re-matched against that canonical input;
* verified local BGE-M3 chunk embeddings aggregated to the track level; and
* unsupervised clustering plus song-level stability and enrichment checks.

The output does **not** infer an artist's real location, identity, friendship,
collaboration, preferred Flow, or performance style.  A link in this study
means only that two high-precision reviewed references were mentioned in the
same cleaned lyric chunk.  BGE-M3 clusters are labels for textual similarity,
not musical genres or social scenes.

Full lyrics, titles, source labels, song identifiers, and song-level map
coordinates remain in ``work/private-chinese-rap-cultural-coordinates-v1``.
The shareable output contains aggregate statistics, safe reviewed entities,
method documentation, and provenance only.

Run from the repository root:
    work\\semantic-ml-venv\\Scripts\\python.exe work\\build_chinese_rap_cultural_coordinates_v1.py
    work\\semantic-ml-venv\\Scripts\\python.exe work\\validate_chinese_rap_cultural_coordinates_v1.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import jieba
import numpy as np
import pandas as pd
import umap
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ID = "chinese-rap-cultural-coordinates-v1"
METHOD_VERSION = "1.1.0"
RANDOM_SEED = 20260824

ANALYSIS_MANIFEST = ROOT / "outputs" / "chinese-rap-analysis-input-v1" / "analysis_input_manifest.json"
ANALYSIS_POINTER = ROOT / "outputs" / "chinese-rap-analysis-input-v1" / "private_analysis_input_pointer.json"
ANALYSIS_VALIDATION = ROOT / "outputs" / "chinese-rap-analysis-input-v1" / "independent_validation.json"
TEXT_SIDECAR_DIR = ROOT / "outputs" / "canonical-lyric-text-sidecar-v1"
TEXT_SIDECAR_MANIFEST = TEXT_SIDECAR_DIR / "manifest.json"
TEXT_SIDECAR_POINTER = TEXT_SIDECAR_DIR / "private_sidecar_pointer.json"
TEXT_SIDECAR_VALIDATION = TEXT_SIDECAR_DIR / "validation.json"
CLEAN_EMBEDDING_MANIFEST = ROOT / "outputs" / "canonical-clean-text-embeddings-v1" / "manifest.json"
CLEAN_EMBEDDING_VALIDATION = ROOT / "outputs" / "canonical-clean-text-embeddings-v1" / "validation.json"
CLEAN_EMBEDDING_PRIVATE_DIR = ROOT / "work" / "private-canonical-clean-text-embeddings-v1"
NER_LEDGER_DIR = ROOT / "outputs" / "chinese-rap-ner-reference-ledger-v1"
NER_LEDGER_MANIFEST = NER_LEDGER_DIR / "manifest.json"
NER_LEDGER_VALIDATION = NER_LEDGER_DIR / "validation.json"
LEXICON_CATALOG = NER_LEDGER_DIR / "core_reference_ledger_v1.csv"

PRIVATE_DIR = ROOT / "work" / "private-chinese-rap-cultural-coordinates-v1"
OUTPUT_DIR = ROOT / "outputs" / ARTIFACT_ID

EXPECTED_CANONICAL_SONGS = 7211
EXPECTED_CANONICAL_CHUNKS = 22128
EXPECTED_SONGS = 7206
EXPECTED_CHUNKS = 21553
EXPECTED_LEXICON_ENTRIES = 128
EMBEDDING_DIMENSIONS = 1024

# The ledger is intentionally smaller than the legacy candidate atlas.  All
# seven categories below are high-precision literal-reference families, not
# personality or performance attributes.  No generic English-token list enters
# inference, so this study does not confuse character frequency with
# code-switching.
PRIMARY_INFERENCE_CATEGORIES = {
    "SPATIAL_NAMED", "PERSON_NAMED", "ORG_NAMED", "RAP_CRAFT", "LANGUAGE_CUE",
    "MATERIAL_REFERENCE", "CULTURAL_REFERENCE",
}

# The study works at the song level.  Subsampling is deliberately by song,
# never by chunks, so repeated chunks cannot leak from a train-like split into
# a test-like split.
K_VALUES = tuple(range(4, 11))
BOOTSTRAP_REPLICATES = 12
BOOTSTRAP_SHARE = 0.80
MIN_CLUSTER_SHARE = 0.03
SILHOUETTE_SAMPLE = 5000

PUBLIC_ALLOWLIST = {
    "README.md",
    "research_protocol.md",
    "data_dictionary.md",
    "manifest.json",
    "validation.json",
    "analysis_summary.json",
    "category_summary.csv",
    "entity_summary.csv",
    "cluster_model_selection.csv",
    "cluster_summary.csv",
    "cluster_entity_enrichment.csv",
    "cluster_keyword_summary.csv",
    "chart_contracts.json",
}

PRIVATE_ALLOWLIST = {
    "private_manifest.json",
    "entity_mentions_v1.csv",
    "category_co_mentions_v1.csv",
    "song_feature_map_v1.csv",
    "song_cluster_assignments_v1.csv",
    "private_validation.json",
}


class BuildError(RuntimeError):
    """Raised when an input contract or safety guard fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
        help="Song-level subsamples per candidate k (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip the private two-dimensional map for a fast diagnostic run.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Return the UTF-8 content hash used by the clean-text sidecar."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    atomic_write_text(path, output.getvalue())


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BuildError(f"Missing {label}: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"Could not read {label}: {path}") from exc
    if not isinstance(result, dict):
        raise BuildError(f"{label} must be a JSON object.")
    return result


def resolve_repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BuildError(f"{label} is missing a controlled relative path.")
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise BuildError(f"{label} escapes the repository.") from exc
    return candidate


def require_directory_allowlist(path: Path, allowed: set[str], label: str) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise BuildError(f"{label} exists but is not a directory: {path}")
    actual = {item.name for item in path.iterdir()}
    unexpected = sorted(actual - allowed)
    if unexpected:
        raise BuildError(
            f"{label} contains undeclared files; refusing to overwrite: {', '.join(unexpected)}"
        )
    nested = [item.name for item in path.iterdir() if item.is_dir()]
    if nested:
        raise BuildError(f"{label} may not contain nested directories: {', '.join(nested)}")


def bool_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise BuildError(f"Expected a Boolean-like value, received {value!r}.")


def ascii_lower(value: str) -> str:
    """Case-fold ASCII only, preserving exact Chinese character offsets."""
    return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in value)


def needs_ascii_boundary(surface: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", surface))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def build_literal_matcher(entities: list[str]) -> re.Pattern[str]:
    """Create a deterministic longest-first literal matcher.

    Literal entities with an ASCII letter or digit must not match inside a
    larger token.  Longest-first order prevents a shorter safe entity from
    winning at the same offset.
    """
    alternatives: list[str] = []
    for entity in sorted(set(entities), key=lambda item: (-len(item), item)):
        literal = re.escape(entity)
        if needs_ascii_boundary(entity):
            literal = rf"(?<![A-Za-z0-9]){literal}(?![A-Za-z0-9])"
        alternatives.append(literal)
    if not alternatives:
        raise BuildError("The reviewed entity lexicon is empty.")
    return re.compile("|".join(alternatives), flags=re.IGNORECASE)


def load_canonical_input() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], dict[str, Path]]:
    """Load the frozen canonical registry, then its validated clean-text view.

    The canonical input remains the immutable identity and duplicate-weight
    registry.  All linguistic analysis must instead use the strict clean-text
    sidecar: it removes only audited leading metadata/header lines and withholds
    rows that become metadata-only.  This makes a title or producer credit
    incapable of becoming a lyric token, entity, or embedding feature.
    """
    manifest = read_json_object(ANALYSIS_MANIFEST, "canonical analysis manifest")
    pointer = read_json_object(ANALYSIS_POINTER, "canonical analysis private pointer")
    validation = read_json_object(ANALYSIS_VALIDATION, "canonical analysis validation")

    if manifest.get("analysis_input_id") != "chinese-rap-canonical-analysis-input-v1":
        raise BuildError("Unexpected canonical analysis input ID.")
    if pointer.get("analysis_input_id") != manifest.get("analysis_input_id"):
        raise BuildError("Analysis pointer does not belong to the selected manifest.")
    if validation.get("status") != "pass":
        raise BuildError("The canonical analysis input has not passed independent validation.")
    if pointer.get("classification") != "private_local_only_full_lyrics":
        raise BuildError("The input pointer is not marked private-local-only lyrics.")

    private_dir = resolve_repo_path(pointer.get("private_directory"), "analysis private directory")
    chunk_name = pointer.get("chunk_file")
    song_name = pointer.get("song_file")
    if not isinstance(chunk_name, str) or not isinstance(song_name, str):
        raise BuildError("The analysis pointer lacks a chunk or song filename.")
    chunk_path = (private_dir / chunk_name).resolve()
    song_path = (private_dir / song_name).resolve()
    try:
        chunk_path.relative_to(private_dir)
        song_path.relative_to(private_dir)
    except ValueError as exc:
        raise BuildError("Analysis CSV path escapes its controlled private directory.") from exc

    expected_chunk_hash = pointer.get("chunk_file_sha256")
    expected_song_hash = pointer.get("song_file_sha256")
    if not isinstance(expected_chunk_hash, str) or not isinstance(expected_song_hash, str):
        raise BuildError("The analysis pointer lacks expected hashes.")
    if sha256_file(chunk_path) != expected_chunk_hash or sha256_file(song_path) != expected_song_hash:
        raise BuildError("Canonical analysis input hash mismatch; refusing to analyse stale data.")

    private_artifacts = manifest.get("output_private_artifacts")
    if not isinstance(private_artifacts, dict):
        raise BuildError("Canonical analysis manifest has no private artifact declaration.")
    if private_artifacts.get(chunk_name, {}).get("sha256") != expected_chunk_hash:
        raise BuildError("Chunk hash differs between canonical pointer and manifest.")
    if private_artifacts.get(song_name, {}).get("sha256") != expected_song_hash:
        raise BuildError("Song hash differs between canonical pointer and manifest.")

    raw_chunks = pd.read_csv(chunk_path, dtype=str, keep_default_na=False)
    raw_songs = pd.read_csv(song_path, dtype=str, keep_default_na=False)
    required_chunks = {
        "song_id", "chunk_id", "text", "canonical_lyric_text_sha256",
        "analysis_text_weight", "downstream_eligibility",
        "cross_song_duplicate_text_group_id",
    }
    required_songs = {"song_id", "canonical_artist", "canonical_song_title", "downstream_eligibility"}
    if missing := required_chunks - set(raw_chunks.columns):
        raise BuildError(f"Canonical chunks omit required columns: {sorted(missing)}")
    if missing := required_songs - set(raw_songs.columns):
        raise BuildError(f"Canonical songs omit required columns: {sorted(missing)}")
    if len(raw_chunks) != EXPECTED_CANONICAL_CHUNKS or len(raw_songs) != EXPECTED_CANONICAL_SONGS:
        raise BuildError(
            f"Unexpected frozen canonical population: {len(raw_songs)} songs / {len(raw_chunks)} chunks; "
            "update the protocol deliberately before proceeding."
        )
    if not (raw_chunks["downstream_eligibility"] == "eligible").all():
        raise BuildError("A non-eligible chunk entered the canonical analysis input.")
    if not (raw_songs["downstream_eligibility"] == "eligible").all():
        raise BuildError("A non-eligible song entered the canonical analysis input.")
    if raw_songs["song_id"].duplicated().any() or raw_chunks[["song_id", "chunk_id"]].duplicated().any():
        raise BuildError("Canonical song or song/chunk primary keys are not unique.")
    if raw_chunks["text"].map(lambda value: bool(str(value).strip())).eq(False).any():
        raise BuildError("Canonical input contains an empty eligible lyric chunk.")

    raw_chunks["analysis_text_weight"] = pd.to_numeric(raw_chunks["analysis_text_weight"], errors="raise")
    if not raw_chunks["analysis_text_weight"].gt(0).all() or not raw_chunks["analysis_text_weight"].le(1).all():
        raise BuildError("Canonical duplicate-text weights must fall in (0, 1].")
    duplicate_groups = raw_chunks.loc[
        raw_chunks["cross_song_duplicate_text_group_id"].ne(""),
        ["cross_song_duplicate_text_group_id", "analysis_text_weight"],
    ]
    if not duplicate_groups.empty:
        grouped_weight = duplicate_groups.groupby("cross_song_duplicate_text_group_id")["analysis_text_weight"].sum()
        if not np.allclose(grouped_weight.to_numpy(dtype=float), 1.0, atol=1e-9):
            raise BuildError("At least one cross-song duplicate group does not sum to weight one.")

    song_ids_from_chunks = set(raw_chunks["song_id"])
    if song_ids_from_chunks != set(raw_songs["song_id"]):
        raise BuildError("Canonical song table and chunk table disagree on song IDs.")

    sidecar_manifest = read_json_object(TEXT_SIDECAR_MANIFEST, "clean-text sidecar manifest")
    sidecar_pointer = read_json_object(TEXT_SIDECAR_POINTER, "clean-text sidecar pointer")
    sidecar_validation = read_json_object(TEXT_SIDECAR_VALIDATION, "clean-text sidecar validation")
    if sidecar_manifest.get("artifact_id") != "canonical-lyric-text-sidecar-v1":
        raise BuildError("Unexpected clean-text sidecar ID.")
    if sidecar_pointer.get("artifact_id") != sidecar_manifest.get("artifact_id"):
        raise BuildError("Clean-text sidecar pointer does not match its manifest.")
    if sidecar_validation.get("status") != "pass":
        raise BuildError("The clean-text sidecar has not passed independent validation.")
    if sidecar_pointer.get("classification") != "private_local_only_cleaned_full_lyric_text":
        raise BuildError("Clean-text sidecar is not classified as local-only cleaned lyrics.")
    sidecar_canonical = sidecar_manifest.get("canonical_input", {})
    if (
        sidecar_canonical.get("analysis_input_id") != manifest.get("analysis_input_id")
        or sidecar_canonical.get("private_chunk_input_sha256") != expected_chunk_hash
        or sidecar_canonical.get("private_song_input_sha256") != expected_song_hash
    ):
        raise BuildError("Clean-text sidecar does not derive from the current frozen canonical input.")
    sidecar_counts = sidecar_manifest.get("counts", {})
    if (
        sidecar_counts.get("input_songs") != EXPECTED_CANONICAL_SONGS
        or sidecar_counts.get("input_chunks") != EXPECTED_CANONICAL_CHUNKS
        or sidecar_counts.get("eligible_clean_text_chunks") != EXPECTED_CHUNKS
        or sidecar_counts.get("withheld_metadata_only_chunks") != EXPECTED_CANONICAL_CHUNKS - EXPECTED_CHUNKS
    ):
        raise BuildError("Clean-text sidecar counts do not match the declared research population.")

    sidecar_private_dir = resolve_repo_path(sidecar_pointer.get("private_directory"), "clean-text private directory")
    clean_name = sidecar_pointer.get("cleaned_chunk_file")
    if not isinstance(clean_name, str):
        raise BuildError("Clean-text sidecar pointer lacks a cleaned chunk filename.")
    clean_path = (sidecar_private_dir / clean_name).resolve()
    try:
        clean_path.relative_to(sidecar_private_dir)
    except ValueError as exc:
        raise BuildError("Clean-text file path escapes its controlled private directory.") from exc
    if sha256_file(clean_path) != sidecar_pointer.get("cleaned_chunk_file_sha256"):
        raise BuildError("Clean-text sidecar hash mismatch; refusing to analyse stale text.")

    clean_rows = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
    required_clean = {
        "analysis_text_sidecar_id", "cleaning_version", "song_id", "chunk_id",
        "canonical_lyric_text_sha256", "analysis_text", "analysis_text_sha256",
        "analysis_text_status", "analysis_text_weight",
    }
    if missing := required_clean - set(clean_rows.columns):
        raise BuildError(f"Clean-text sidecar omits required columns: {sorted(missing)}")
    if len(clean_rows) != EXPECTED_CANONICAL_CHUNKS:
        raise BuildError("Clean-text sidecar does not preserve every frozen canonical chunk for audit.")
    if clean_rows[["song_id", "chunk_id"]].duplicated().any():
        raise BuildError("Clean-text sidecar has duplicate song/chunk keys.")
    if not clean_rows["analysis_text_sidecar_id"].eq(sidecar_manifest["artifact_id"]).all():
        raise BuildError("A clean-text row has the wrong sidecar identity.")
    if not clean_rows["cleaning_version"].eq(sidecar_manifest.get("cleaning_version")).all():
        raise BuildError("A clean-text row has an unexpected cleaning version.")
    permitted_statuses = {"eligible_clean_text", "withheld_metadata_only"}
    if not set(clean_rows["analysis_text_status"]).issubset(permitted_statuses):
        raise BuildError("Clean-text sidecar contains an undeclared analysis-text status.")
    clean_rows["analysis_text_weight"] = pd.to_numeric(clean_rows["analysis_text_weight"], errors="raise")

    join_columns = ["song_id", "chunk_id", "canonical_lyric_text_sha256"]
    joined = raw_chunks.merge(
        clean_rows[
            join_columns + ["analysis_text", "analysis_text_sha256", "analysis_text_status", "analysis_text_weight"]
        ],
        on=join_columns,
        how="left",
        suffixes=("_canonical", "_clean"),
        validate="one_to_one",
        sort=False,
    )
    if len(joined) != len(raw_chunks) or joined["analysis_text_status"].eq("").any() or joined["analysis_text_status"].isna().any():
        raise BuildError("Clean-text sidecar does not exactly join to every frozen canonical chunk.")
    if not np.allclose(
        joined["analysis_text_weight_canonical"].to_numpy(dtype=float),
        joined["analysis_text_weight_clean"].to_numpy(dtype=float),
        atol=1e-12,
    ):
        raise BuildError("Clean-text sidecar altered a frozen duplicate-text weight.")
    if (joined.loc[joined["analysis_text_status"] == "eligible_clean_text", "analysis_text"].map(lambda value: bool(str(value).strip())).eq(False)).any():
        raise BuildError("An analysis-eligible clean-text row is empty.")
    if not joined.loc[joined["analysis_text_status"] == "eligible_clean_text"].apply(
        lambda row: sha256_text(str(row["analysis_text"])) == str(row["analysis_text_sha256"]), axis=1
    ).all():
        raise BuildError("An analysis-eligible clean-text content hash is stale or invalid.")
    withheld = joined.loc[joined["analysis_text_status"] == "withheld_metadata_only"]
    if len(withheld) != EXPECTED_CANONICAL_CHUNKS - EXPECTED_CHUNKS:
        raise BuildError("Unexpected count of metadata-only chunks withheld from linguistic analysis.")
    if withheld["analysis_text"].map(lambda value: bool(str(value).strip())).any():
        raise BuildError("A metadata-only row retains analysis text.")

    # A duplicate group must remain whole or be excluded whole.  Otherwise a
    # frozen group-level duplicate weight could silently become invalid.
    original_group_members = raw_chunks.loc[
        raw_chunks["cross_song_duplicate_text_group_id"].ne(""),
        ["cross_song_duplicate_text_group_id", "song_id", "chunk_id"],
    ]
    clean_group_members = joined.loc[
        joined["analysis_text_status"].eq("eligible_clean_text")
        & joined["cross_song_duplicate_text_group_id"].ne(""),
        ["cross_song_duplicate_text_group_id", "song_id", "chunk_id"],
    ]
    original_group_sizes = original_group_members.groupby("cross_song_duplicate_text_group_id").size()
    clean_group_sizes = clean_group_members.groupby("cross_song_duplicate_text_group_id").size()
    for group_id, full_size in original_group_sizes.items():
        retained_size = int(clean_group_sizes.get(group_id, 0))
        if retained_size not in {0, int(full_size)}:
            raise BuildError("A cross-song duplicate group was only partly retained after header cleaning.")

    chunks = joined.loc[joined["analysis_text_status"] == "eligible_clean_text"].copy()
    chunks["text"] = chunks["analysis_text"]
    chunks["analysis_text_weight"] = chunks["analysis_text_weight_clean"]
    chunks = chunks.drop(columns=["analysis_text_weight_canonical", "analysis_text_weight_clean", "analysis_text"])
    if len(chunks) != EXPECTED_CHUNKS:
        raise BuildError("Clean-text analytic chunk count is not the declared population.")
    clean_duplicate_groups = chunks.loc[
        chunks["cross_song_duplicate_text_group_id"].ne(""),
        ["cross_song_duplicate_text_group_id", "analysis_text_weight"],
    ]
    if not clean_duplicate_groups.empty:
        clean_grouped_weight = clean_duplicate_groups.groupby("cross_song_duplicate_text_group_id")["analysis_text_weight"].sum()
        if not np.allclose(clean_grouped_weight.to_numpy(dtype=float), 1.0, atol=1e-9):
            raise BuildError("A retained duplicate group no longer sums to one after header cleaning.")

    active_song_ids = set(chunks["song_id"])
    songs = raw_songs.loc[raw_songs["song_id"].isin(active_song_ids)].copy()
    if len(songs) != EXPECTED_SONGS or set(songs["song_id"]) != active_song_ids:
        raise BuildError("Clean-text song population does not match active lyric chunks.")
    songs["clean_analysis_chunk_count"] = songs["song_id"].map(chunks.groupby("song_id").size()).astype(int)
    if int(songs["clean_analysis_chunk_count"].sum()) != EXPECTED_CHUNKS:
        raise BuildError("Clean-text per-song chunk counts do not reconcile to active chunks.")

    return chunks, songs, {"canonical": manifest, "clean_text_sidecar": sidecar_manifest}, {
        "canonical_chunks": chunk_path,
        "canonical_songs": song_path,
        "cleaned_chunks": clean_path,
    }


def load_embedding_matrix(chunks: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Load BGE-M3 rows aligned to the validated clean-text sidecar exactly."""
    manifest = read_json_object(CLEAN_EMBEDDING_MANIFEST, "clean-text embedding manifest")
    validation = read_json_object(CLEAN_EMBEDDING_VALIDATION, "clean-text embedding validation")
    if manifest.get("artifact_id") != "private-canonical-clean-text-embeddings-v1":
        raise BuildError("Unexpected clean-text embedding artifact ID.")
    if manifest.get("validation", {}).get("passed") is not True or validation.get("passed") is not True:
        raise BuildError("The clean-text embedding artifact has not passed independent validation.")
    clean_population = manifest.get("clean_text_sidecar", {})
    counts = manifest.get("counts", {})
    if (
        clean_population.get("cleaned_eligible_rows") != EXPECTED_CHUNKS
        or clean_population.get("clean_song_count") != EXPECTED_SONGS
        or clean_population.get("metadata_only_rows_excluded") != EXPECTED_CANONICAL_CHUNKS - EXPECTED_CHUNKS
        or counts.get("dimensions") != EMBEDDING_DIMENSIONS
    ):
        raise BuildError("Clean-text embedding manifest does not match the declared analytic population.")

    contract_path = CLEAN_EMBEDDING_PRIVATE_DIR / "canonical_clean_text_embedding_contract_v1.json"
    contract = read_json_object(contract_path, "private embedding contract")
    if contract.get("artifact_id") != manifest.get("artifact_id"):
        raise BuildError("Private clean-text embedding contract does not match its public manifest.")
    published_hashes = manifest.get("private_artifact_hashes", {})
    if (
        published_hashes.get("contract_sha256") != sha256_file(contract_path)
        or contract.get("clean_text_sidecar", {}).get("artifact_id") != "canonical-lyric-text-sidecar-v1"
        or contract.get("clean_text_sidecar", {}).get("manifest_sha256") != sha256_file(TEXT_SIDECAR_MANIFEST)
    ):
        raise BuildError("Clean-text embedding contract does not prove lineage to the current validated sidecar.")
    vector_file = contract.get("vector_file", {})
    row_map_file = contract.get("row_map_file", {})
    vector_name = vector_file.get("filename")
    row_map_name = row_map_file.get("filename")
    if not isinstance(vector_name, str) or not isinstance(row_map_name, str):
        raise BuildError("Private embedding contract lacks controlled filenames.")
    vectors_path = (CLEAN_EMBEDDING_PRIVATE_DIR / vector_name).resolve()
    row_map_path = (CLEAN_EMBEDDING_PRIVATE_DIR / row_map_name).resolve()
    try:
        vectors_path.relative_to(CLEAN_EMBEDDING_PRIVATE_DIR)
        row_map_path.relative_to(CLEAN_EMBEDDING_PRIVATE_DIR)
    except ValueError as exc:
        raise BuildError("Embedding file path escapes its private directory.") from exc
    if sha256_file(vectors_path) != vector_file.get("sha256"):
        raise BuildError("Embedding matrix hash mismatch.")
    if sha256_file(row_map_path) != row_map_file.get("sha256"):
        raise BuildError("Embedding row-map hash mismatch.")
    if (
        published_hashes.get("vector_sha256") != vector_file.get("sha256")
        or published_hashes.get("row_map_sha256") != row_map_file.get("sha256")
    ):
        raise BuildError("Public clean-text embedding manifest and private contract disagree on file hashes.")

    row_map = pd.read_csv(row_map_path, dtype=str, keep_default_na=False)
    required_map_columns = {
        "clean_row_index", "song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256",
        "analysis_text_weight", "embedding_provenance", "base_canonical_vector_index",
        "base_canonical_vector_sha256", "reembed_reason",
    }
    if missing := required_map_columns - set(row_map.columns):
        raise BuildError(f"Embedding row map omits required columns: {sorted(missing)}")
    row_map["clean_row_index"] = pd.to_numeric(row_map["clean_row_index"], errors="raise").astype(int)
    row_map["analysis_text_weight"] = pd.to_numeric(row_map["analysis_text_weight"], errors="raise")
    if len(row_map) != EXPECTED_CHUNKS or row_map["clean_row_index"].tolist() != list(range(EXPECTED_CHUNKS)):
        raise BuildError("Embedding row map is not an exact contiguous clean-text row order.")

    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.shape != (EXPECTED_CHUNKS, EMBEDDING_DIMENSIONS) or vectors.dtype != np.float32:
        raise BuildError(f"Embedding matrix has unexpected shape or dtype: {vectors.shape} / {vectors.dtype}")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise BuildError("Embedding rows are not L2-normalized.")

    clean_keys = chunks[["song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256", "analysis_text_weight"]].copy()
    clean_keys["chunk_id"] = clean_keys["chunk_id"].astype(str)
    clean_keys = clean_keys.sort_values(["song_id", "chunk_id"], key=lambda series: series.astype(str), kind="stable")
    map_keys = row_map[["song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256", "analysis_text_weight"]].copy()
    map_keys["chunk_id"] = map_keys["chunk_id"].astype(str)
    map_keys = map_keys.sort_values(["song_id", "chunk_id"], key=lambda series: series.astype(str), kind="stable")
    if not clean_keys[["song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256"]].reset_index(drop=True).equals(
        map_keys[["song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256"]].reset_index(drop=True)
    ):
        raise BuildError("Embedding row-map keys do not exactly match clean chunk keys and text hashes.")
    if not np.allclose(
        clean_keys["analysis_text_weight"].to_numpy(dtype=float),
        map_keys["analysis_text_weight"].to_numpy(dtype=float),
        atol=1e-12,
    ):
        raise BuildError("Embedding row-map duplicate weights do not match clean chunk weights.")

    reused = row_map["embedding_provenance"].eq("reused_verified_canonical_vector_exact_clean_text_sha")
    reembedded = row_map["embedding_provenance"].eq("reembedded_local_bge_m3_clean_text")
    if not (reused | reembedded).all():
        raise BuildError("Clean-text embedding row map contains an undeclared vector provenance.")
    if int(reused.sum()) != counts.get("reused_exact_clean_text_rows") or int(reembedded.sum()) != counts.get("reembedded_changed_clean_text_rows"):
        raise BuildError("Clean-text embedding provenance counts do not reconcile to the manifest.")

    return np.asarray(vectors), row_map, contract, {"vectors": vectors_path, "row_map": row_map_path}


def load_reviewed_lexicon() -> pd.DataFrame:
    if not LEXICON_CATALOG.is_file():
        raise BuildError(f"Missing high-precision NER reference ledger: {LEXICON_CATALOG}")
    ledger_manifest = read_json_object(NER_LEDGER_MANIFEST, "NER reference-ledger manifest")
    ledger_validation = read_json_object(NER_LEDGER_VALIDATION, "NER reference-ledger validation")
    if ledger_manifest.get("artifact_id") != "chinese-rap-ner-reference-ledger-v1" or ledger_validation.get("status") != "pass":
        raise BuildError("The NER reference ledger has not passed independent validation.")
    if ledger_manifest.get("counts", {}).get("core_reference_surfaces") != EXPECTED_LEXICON_ENTRIES:
        raise BuildError("The NER reference ledger does not contain the declared high-precision core size.")
    ledger_file = ledger_manifest.get("output_files", {}).get(LEXICON_CATALOG.name, {})
    if ledger_file.get("sha256") != sha256_file(LEXICON_CATALOG):
        raise BuildError("The NER reference ledger hash does not match its manifest.")
    catalog = pd.read_csv(LEXICON_CATALOG, dtype=str, keep_default_na=False)
    required = {
        "entity", "canonical_category", "category_name", "latin_or_digit_boundary",
        "presentation_group", "evidence_tier", "primary_for_inference", "selection_status",
        "claim_scope",
    }
    if missing := required - set(catalog.columns):
        raise BuildError(f"NER reference ledger omits required fields: {sorted(missing)}")
    if len(catalog) != EXPECTED_LEXICON_ENTRIES:
        raise BuildError(f"Expected {EXPECTED_LEXICON_ENTRIES} core references, found {len(catalog)}.")
    if catalog["entity"].eq("").any() or catalog["entity"].duplicated().any():
        raise BuildError("NER reference ledger contains empty or duplicate literal surfaces.")
    if not set(catalog["canonical_category"]).issubset(PRIMARY_INFERENCE_CATEGORIES):
        raise BuildError("NER reference ledger contains a non-core reference family.")
    catalog["primary_for_inference"] = catalog["primary_for_inference"].map(bool_text)
    if not catalog["primary_for_inference"].all() or not catalog["selection_status"].eq("core_reference").all():
        raise BuildError("NER reference ledger contains a non-core or non-inferential row.")
    if not catalog["claim_scope"].str.contains("literal lyric reference only", regex=False).all():
        raise BuildError("NER reference ledger omits the literal-reference claim boundary.")
    catalog["entity_key"] = catalog["entity"].map(ascii_lower)
    # The reviewed catalog intentionally preserves source casing.  Its four
    # ASCII-only case variants (for example, ``GAI``/``gai``) collapse to one
    # matcher pattern because matching is case-insensitive.  Refuse a collapse
    # only if it would conflate different reviewed categories.
    raw_surface_count = int(len(catalog))
    collapsed_variant_count = int(catalog.duplicated("entity_key").sum())
    if collapsed_variant_count:
        for _, group in catalog.groupby("entity_key", sort=False):
            if len(group) > 1 and group["canonical_category"].nunique() != 1:
                surfaces = ", ".join(group["entity"].tolist())
                raise BuildError(
                    "ASCII-case-equivalent reviewed surfaces disagree on category: " + surfaces
                )
        catalog = (
            catalog.sort_values(["entity_key", "entity"], kind="stable")
            .drop_duplicates("entity_key", keep="first")
            .copy()
        )
    catalog = catalog.sort_values(["canonical_category", "entity"], kind="stable").reset_index(drop=True)
    catalog.attrs["raw_surface_count"] = raw_surface_count
    catalog.attrs["casefold_collapsed_variant_count"] = collapsed_variant_count
    return catalog


def presentation_group(category: str) -> str:
    """Map reviewed fine-grained labels to reader-facing cultural dimensions."""
    groups = {
        "SPATIAL_NAMED": "Spatial references & local imaginaries",
        "PERSON_NAMED": "Named people / crews / organizations",
        "ORG_NAMED": "Named people / crews / organizations",
        "RAP_CRAFT": "Rap craft & scene vocabulary",
        "LANGUAGE_CUE": "Language & dialect cues",
        "MATERIAL_REFERENCE": "Material life & consumption",
        "CULTURAL_REFERENCE": "Cultural / historical / media references",
    }
    return groups.get(category, "Audit-only or excluded imagery")


def evidence_tier(category: str) -> str:
    """State the strongest claim allowed for a reviewed literal surface."""
    tiers = {
        "SPATIAL_NAMED": "reviewed + precision-whitelisted literal surface; not geocoded",
        "PERSON_NAMED": "reviewed + precision-whitelisted literal surface; not a verified relationship",
        "ORG_NAMED": "reviewed + precision-whitelisted literal surface; not a verified relationship",
        "RAP_CRAFT": "reviewed + precision-whitelisted literal surface; not a performed-style measure",
        "LANGUAGE_CUE": "reviewed + precision-whitelisted literal surface; not demonstrated language use",
        "MATERIAL_REFERENCE": "reviewed + precision-whitelisted literal surface; not a lifestyle attribute",
        "CULTURAL_REFERENCE": "reviewed + precision-whitelisted literal surface",
    }
    return tiers.get(category, "audit-only literal surface")


def re_match_entities(
    chunks: pd.DataFrame, catalog: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """Re-match reviewed surfaces against cleaned lyric text.

    The emitted private sidecar never includes lyric text.  It keeps a text
    hash and character offsets, which permits local audit without distributing
    complete copyrighted lyric chunks.
    """
    matcher = build_literal_matcher(catalog["entity"].tolist())
    lookup = catalog.set_index("entity_key").to_dict("index")
    mention_rows: list[dict[str, Any]] = []
    entity_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "chunks": set(), "songs": set(), "weighted_chunk_presence": 0.0,
            "span_occurrences": 0, "text_hashes": set(),
        }
    )
    category_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "chunks": set(), "songs": set(), "weighted_chunk_presence": 0.0,
            "entities": set(), "text_hashes": set(),
        }
    )
    pair_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"chunks": set(), "songs": set(), "weighted_chunk_presence": 0.0}
    )
    song_categories: dict[str, set[str]] = defaultdict(set)

    for row in chunks.itertuples(index=False):
        text = str(getattr(row, "text"))
        song_id = str(getattr(row, "song_id"))
        chunk_id = str(getattr(row, "chunk_id"))
        canonical_text_hash = str(getattr(row, "canonical_lyric_text_sha256"))
        text_hash = str(getattr(row, "analysis_text_sha256"))
        weight = float(getattr(row, "analysis_text_weight"))
        chunk_key = (song_id, chunk_id)
        observed_entities: dict[str, dict[str, Any]] = {}
        for match in matcher.finditer(text):
            matched_surface = match.group(0)
            item = lookup.get(ascii_lower(matched_surface))
            if item is None:
                raise BuildError(f"No lexicon lookup exists for matched entity {matched_surface!r}.")
            entity = str(item["entity"])
            category = str(item["canonical_category"])
            group = str(item["presentation_group"])
            mention_rows.append(
                {
                    "song_id": song_id,
                    "chunk_id": chunk_id,
                    "canonical_lyric_text_sha256": canonical_text_hash,
                    "analysis_text_sha256": text_hash,
                    "entity": entity,
                    "canonical_category": category,
                    "presentation_group": group,
                    "evidence_tier": str(item["evidence_tier"]),
                    "primary_for_inference": bool(item["primary_for_inference"]),
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "analysis_text_weight": f"{weight:.12g}",
                }
            )
            observed_entities.setdefault(entity, item)
            entity_stats[entity]["span_occurrences"] += 1

        if not observed_entities:
            continue
        observed_categories: set[str] = set()
        for entity, item in observed_entities.items():
            category = str(item["canonical_category"])
            record = entity_stats[entity]
            record["chunks"].add(chunk_key)
            record["songs"].add(song_id)
            record["weighted_chunk_presence"] += weight
            record["text_hashes"].add(text_hash)
            category_record = category_stats[category]
            category_record["chunks"].add(chunk_key)
            category_record["songs"].add(song_id)
            category_record["entities"].add(entity)
            category_record["text_hashes"].add(text_hash)
            observed_categories.add(category)
            if bool(item["primary_for_inference"]):
                song_categories[song_id].add(category)
        for category in observed_categories:
            category_stats[category]["weighted_chunk_presence"] += weight
        for left, right in combinations_sorted(observed_categories):
            pair_record = pair_stats[(left, right)]
            pair_record["chunks"].add(chunk_key)
            pair_record["songs"].add(song_id)
            pair_record["weighted_chunk_presence"] += weight

    mention_columns = [
        "song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256", "entity",
        "canonical_category", "presentation_group", "evidence_tier", "primary_for_inference",
        "start_char", "end_char", "analysis_text_weight",
    ]
    mentions = pd.DataFrame(mention_rows, columns=mention_columns)
    if mentions.empty:
        raise BuildError("No reviewed entities matched the clean lyric corpus.")

    entity_rows: list[dict[str, Any]] = []
    lookup_by_entity = catalog.set_index("entity").to_dict("index")
    for entity in sorted(entity_stats, key=lambda value: (-entity_stats[value]["weighted_chunk_presence"], value)):
        stats = entity_stats[entity]
        item = lookup_by_entity[entity]
        entity_rows.append(
            {
                "entity": entity,
                "canonical_category": item["canonical_category"],
                "presentation_group": item["presentation_group"],
                "evidence_tier": item["evidence_tier"],
                "primary_for_inference": bool(item["primary_for_inference"]),
                "song_count": len(stats["songs"]),
                "chunk_count": len(stats["chunks"]),
                "weighted_chunk_presence": round(float(stats["weighted_chunk_presence"]), 6),
                "span_occurrences": int(stats["span_occurrences"]),
                "unique_text_count": len(stats["text_hashes"]),
            }
        )
    entity_summary = pd.DataFrame(entity_rows)

    category_lookup = catalog.drop_duplicates("canonical_category", keep="first").set_index("canonical_category").to_dict("index")
    category_rows: list[dict[str, Any]] = []
    for category in sorted(category_stats):
        stats = category_stats[category]
        item = category_lookup[category]
        category_rows.append(
            {
                "canonical_category": category,
                "presentation_group": item["presentation_group"],
                "evidence_tier": item["evidence_tier"],
                "primary_for_inference": bool(item["primary_for_inference"]),
                "song_count": len(stats["songs"]),
                "chunk_count": len(stats["chunks"]),
                "weighted_chunk_presence": round(float(stats["weighted_chunk_presence"]), 6),
                "entity_surface_count": len(stats["entities"]),
                "unique_text_count": len(stats["text_hashes"]),
            }
        )
    category_summary = pd.DataFrame(category_rows).sort_values(
        ["weighted_chunk_presence", "canonical_category"], ascending=[False, True], kind="stable"
    )

    pair_rows: list[dict[str, Any]] = []
    for (left, right), stats in pair_stats.items():
        pair_rows.append(
            {
                "left_category": left,
                "right_category": right,
                "left_presentation_group": presentation_group(left),
                "right_presentation_group": presentation_group(right),
                "left_primary_for_inference": bool(category_lookup[left]["primary_for_inference"]),
                "right_primary_for_inference": bool(category_lookup[right]["primary_for_inference"]),
                "song_count": len(stats["songs"]),
                "chunk_count": len(stats["chunks"]),
                "weighted_chunk_presence": round(float(stats["weighted_chunk_presence"]), 6),
                "relation_definition": "co-mentioned in one cleaned lyric chunk",
            }
        )
    pair_summary = pd.DataFrame(pair_rows).sort_values(
        ["weighted_chunk_presence", "left_category", "right_category"],
        ascending=[False, True, True], kind="stable"
    )

    return mentions, entity_summary, category_summary, pair_summary, song_categories


def combinations_sorted(items: set[str]) -> Iterable[tuple[str, str]]:
    ordered = sorted(items)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            yield left, right


def aggregate_song_embeddings(
    chunks: pd.DataFrame, songs: pd.DataFrame, vectors: np.ndarray, row_map: pd.DataFrame
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Average clean chunk embeddings using frozen duplicate-text weights."""
    key_to_index: dict[tuple[str, str, str, str], int] = {}
    for row in row_map.itertuples(index=False):
        key = (
            str(row.song_id), str(row.chunk_id), str(row.canonical_lyric_text_sha256),
            str(row.analysis_text_sha256),
        )
        if key in key_to_index:
            raise BuildError("Embedding row map has a duplicate exact clean chunk key.")
        key_to_index[key] = int(row.clean_row_index)

    per_song_indices: dict[str, list[int]] = defaultdict(list)
    per_song_weights: dict[str, list[float]] = defaultdict(list)
    for row in chunks.itertuples(index=False):
        key = (
            str(row.song_id), str(row.chunk_id), str(row.canonical_lyric_text_sha256),
            str(row.analysis_text_sha256),
        )
        index = key_to_index.get(key)
        if index is None:
            raise BuildError("A clean lyric chunk has no exact semantic embedding row.")
        per_song_indices[str(row.song_id)].append(index)
        per_song_weights[str(row.song_id)].append(float(row.analysis_text_weight))

    song_ids = songs["song_id"].astype(str).tolist()
    aggregate_rows: list[np.ndarray] = []
    for song_id in song_ids:
        indices = per_song_indices.get(song_id)
        weights = per_song_weights.get(song_id)
        if not indices or not weights:
            raise BuildError(f"An active clean-text song has no semantic embedding rows: {song_id}")
        song_vector = np.average(vectors[np.asarray(indices)], axis=0, weights=np.asarray(weights))
        norm = float(np.linalg.norm(song_vector))
        if not math.isfinite(norm) or norm <= 0:
            raise BuildError(f"An eligible song received a degenerate aggregate embedding: {song_id}")
        aggregate_rows.append((song_vector / norm).astype(np.float32))
    song_vectors = np.vstack(aggregate_rows)
    if song_vectors.shape != (EXPECTED_SONGS, EMBEDDING_DIMENSIONS):
        raise BuildError("Unexpected aggregated song embedding shape.")
    if not np.allclose(np.linalg.norm(song_vectors, axis=1), 1.0, atol=1e-5):
        raise BuildError("Aggregated song embeddings are not L2-normalized.")

    private_song_metadata = songs[["song_id", "canonical_artist", "canonical_song_title", "clean_analysis_chunk_count"]].copy()
    return song_ids, song_vectors, private_song_metadata


def select_clusters(
    song_vectors: np.ndarray, bootstrap_replicates: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    """Choose a stable, interpretable KMeans partition without using the UMAP map.

    UMAP is strictly a visual projection.  Candidate clusters are fitted in a
    deterministic PCA representation.  A candidate must retain at least 3% of
    tracks in its smallest group and stay within 90% of the best silhouette
    before stability becomes the selection criterion.
    """
    n_songs = song_vectors.shape[0]
    pca_components = min(64, song_vectors.shape[1], n_songs - 1)
    pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=RANDOM_SEED)
    pca_vectors = normalize(pca.fit_transform(song_vectors), norm="l2")
    silhouette_indices = np.random.default_rng(RANDOM_SEED).choice(
        n_songs, size=min(SILHOUETTE_SAMPLE, n_songs), replace=False
    )
    candidates: list[dict[str, Any]] = []
    reference_labels_by_k: dict[int, np.ndarray] = {}

    for k in K_VALUES:
        reference = KMeans(
            n_clusters=k, random_state=RANDOM_SEED, n_init=25, max_iter=500, algorithm="lloyd"
        ).fit_predict(pca_vectors)
        reference_labels_by_k[k] = reference
        cluster_sizes = np.bincount(reference, minlength=k)
        silhouette = float(silhouette_score(pca_vectors[silhouette_indices], reference[silhouette_indices], metric="cosine"))
        stability_values: list[float] = []
        rng = np.random.default_rng(RANDOM_SEED + k)
        for replicate in range(bootstrap_replicates):
            sample_indices = np.sort(
                rng.choice(n_songs, size=round(n_songs * BOOTSTRAP_SHARE), replace=False)
            )
            sampled_labels = KMeans(
                n_clusters=k,
                random_state=RANDOM_SEED + k * 100 + replicate,
                n_init=8,
                max_iter=350,
                algorithm="lloyd",
            ).fit_predict(pca_vectors[sample_indices])
            stability_values.append(float(adjusted_rand_score(reference[sample_indices], sampled_labels)))
        candidates.append(
            {
                "k": k,
                "silhouette_cosine": silhouette,
                "bootstrap_ari_mean": float(np.mean(stability_values)),
                "bootstrap_ari_sd": float(np.std(stability_values, ddof=1)) if len(stability_values) > 1 else 0.0,
                "bootstrap_replicates": bootstrap_replicates,
                "min_cluster_song_count": int(cluster_sizes.min()),
                "max_cluster_song_count": int(cluster_sizes.max()),
                "min_cluster_share": float(cluster_sizes.min() / n_songs),
                "pca_components": pca_components,
            }
        )

    selection_table = pd.DataFrame(candidates).sort_values("k", kind="stable")
    best_silhouette = float(selection_table["silhouette_cosine"].max())
    eligible = selection_table.loc[
        (selection_table["min_cluster_share"] >= MIN_CLUSTER_SHARE)
        & (selection_table["silhouette_cosine"] >= best_silhouette * 0.90)
    ].copy()
    if eligible.empty:
        raise BuildError("No cluster candidate satisfies the predeclared minimum-size and silhouette rules.")
    eligible = eligible.sort_values(
        ["bootstrap_ari_mean", "silhouette_cosine", "k"],
        ascending=[False, False, True],
        kind="stable",
    )
    selected_k = int(eligible.iloc[0]["k"])
    selection_table["selected"] = selection_table["k"].eq(selected_k)
    selection_table["selection_rule"] = (
        "smallest-cluster share >= 3%; silhouette >= 90% of candidate maximum; "
        "then highest mean song-level subsample ARI, tie-break higher silhouette and lower k"
    )
    metadata = {
        "algorithm": "KMeans on L2-normalized PCA-reduced BGE-M3 track embeddings",
        "candidate_k_values": list(K_VALUES),
        "selected_k": selected_k,
        "pca_components": pca_components,
        "silhouette_metric": "cosine",
        "silhouette_sample_size": int(len(silhouette_indices)),
        "stability_unit": "song",
        "bootstrap_share": BOOTSTRAP_SHARE,
        "bootstrap_replicates": bootstrap_replicates,
        "selection_rule": str(selection_table.loc[selection_table["selected"], "selection_rule"].iloc[0]),
    }
    return pca_vectors, reference_labels_by_k[selected_k], metadata, selection_table


def build_umap_map(pca_vectors: np.ndarray) -> np.ndarray:
    """Create a visual-only two-dimensional map after clustering is fixed."""
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.12,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_SEED,
        transform_seed=RANDOM_SEED,
    )
    coordinates = reducer.fit_transform(pca_vectors)
    if coordinates.shape != (pca_vectors.shape[0], 2) or not np.isfinite(coordinates).all():
        raise BuildError("UMAP did not produce a finite two-dimensional coordinate for every song.")
    return coordinates


STOP_WORDS = {
    "我们", "你们", "他们", "她们", "自己", "这个", "那个", "这些", "那些", "一个", "一种", "一样",
    "不是", "没有", "就是", "因为", "所以", "如果", "但是", "然后", "可以", "已经", "还是", "不会",
    "怎么", "什么", "哪里", "时候", "现在", "今天", "明天", "一直", "真的", "可能", "好像", "其实",
    "我的", "你的", "他的", "她的", "它的", "他们的", "我们", "你们", "还有", "让我", "给我",
    "不要", "不要", "不会", "不能", "一起", "还有", "他们", "她们", "所有", "开始", "结束",
    "因为", "为了", "这样", "那样", "一下", "一些", "一点", "然后", "如果", "虽然", "但是",
    "这首", "这段", "歌词", "说唱", "rap", "yeah", "oh", "uh", "ah", "na", "la", "the", "and",
    "you", "me", "my", "your", "i", "im", "we", "they", "he", "she", "it", "this", "that",
    "these", "those", "baby", "love", "like", "get", "got", "up", "down", "to", "of", "in",
    "on", "for", "with", "is", "are", "was", "be", "do", "does", "dont", "can", "will", "just",
    "prod", "feat", "ft",
}


def tokenize_lyrics(text: str) -> list[str]:
    """Tokenize for descriptive keyness, not for a linguistic gold standard."""
    tokens: list[str] = []
    for raw_token in jieba.lcut(text, HMM=False):
        token = raw_token.strip()
        if not token:
            continue
        lowered = ascii_lower(token)
        if lowered in STOP_WORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            tokens.append(token)
        elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_+\-]{1,}", token):
            tokens.append(lowered)
        elif re.fullmatch(r"\d{2,}", token):
            tokens.append(token)
    return tokens


def song_token_counters(chunks: pd.DataFrame, song_ids: list[str]) -> list[Counter[str]]:
    """Build duplicate-weighted token counters at the same song grain as clustering."""
    by_song: dict[str, Counter[str]] = {song_id: Counter() for song_id in song_ids}
    for row in chunks.itertuples(index=False):
        song_id = str(row.song_id)
        weight = float(row.analysis_text_weight)
        token_counts = Counter(tokenize_lyrics(str(row.text)))
        for token, count in token_counts.items():
            by_song[song_id][token] += float(count) * weight
    return [by_song[song_id] for song_id in song_ids]


def log_odds_z(
    in_count: float,
    out_count: float,
    in_total: float,
    out_total: float,
    corpus_count: float,
    corpus_total: float,
    prior_strength: float = 1000.0,
) -> float:
    """Weighted log-odds z score with a corpus-informed Dirichlet prior."""
    if corpus_total <= 0:
        return float("nan")
    alpha = max(1e-8, prior_strength * corpus_count / corpus_total)
    alpha_other = max(1e-8, prior_strength - alpha)
    left_denominator = max(1e-8, in_total - in_count + alpha_other)
    right_denominator = max(1e-8, out_total - out_count + alpha_other)
    delta = math.log((in_count + alpha) / left_denominator) - math.log((out_count + alpha) / right_denominator)
    variance = 1.0 / (in_count + alpha) + 1.0 / (out_count + alpha)
    return delta / math.sqrt(variance)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return monotone FDR-adjusted p values in original order."""
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    total = len(indexed)
    adjusted_ranked = [0.0] * total
    running = 1.0
    for reverse_rank in range(total - 1, -1, -1):
        _, value = indexed[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, value * total / rank)
        adjusted_ranked[reverse_rank] = min(1.0, running)
    adjusted = [0.0] * total
    for (original_index, _), value in zip(indexed, adjusted_ranked):
        adjusted[original_index] = value
    return adjusted


def build_cluster_descriptions(
    chunks: pd.DataFrame,
    song_ids: list[str],
    labels: np.ndarray,
    song_categories: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explain latent clusters with high-dispersion keywords and FDR-controlled entities."""
    counters = song_token_counters(chunks, song_ids)
    global_counts: Counter[str] = Counter()
    global_song_presence: Counter[str] = Counter()
    for counter in counters:
        global_counts.update(counter)
        for token in counter:
            global_song_presence[token] += 1
    global_total = float(sum(global_counts.values()))
    if global_total <= 0:
        raise BuildError("No usable descriptive tokens were extracted from clean lyric text.")

    all_categories = sorted({category for values in song_categories.values() for category in values})
    keyword_rows: list[dict[str, Any]] = []
    enrichment_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    label_values = sorted(set(int(value) for value in labels))
    song_categories_by_index = [song_categories.get(song_id, set()) for song_id in song_ids]

    for cluster_label in label_values:
        indices = np.flatnonzero(labels == cluster_label)
        rest_indices = np.flatnonzero(labels != cluster_label)
        cluster_counter: Counter[str] = Counter()
        cluster_song_presence: Counter[str] = Counter()
        for index in indices:
            cluster_counter.update(counters[int(index)])
            for token in counters[int(index)]:
                cluster_song_presence[token] += 1
        cluster_total = float(sum(cluster_counter.values()))
        rest_total = max(0.0, global_total - cluster_total)
        scored_keywords: list[tuple[str, float, int, float]] = []
        for token, in_count in cluster_counter.items():
            in_song_count = int(cluster_song_presence[token])
            if in_song_count < 5 or global_song_presence[token] < 10:
                continue
            out_count = max(0.0, float(global_counts[token]) - float(in_count))
            z_score = log_odds_z(
                float(in_count), out_count, cluster_total, rest_total,
                float(global_counts[token]), global_total,
            )
            if math.isfinite(z_score):
                scored_keywords.append((token, z_score, in_song_count, float(in_count)))
        scored_keywords.sort(key=lambda item: (-item[1], -item[2], item[0]))
        for rank, (token, z_score, song_count, weighted_count) in enumerate(scored_keywords[:20], start=1):
            keyword_rows.append(
                {
                    "cluster": f"C{cluster_label + 1}",
                    "keyword_rank": rank,
                    "keyword": token,
                    "log_odds_z": round(z_score, 5),
                    "cluster_song_count": song_count,
                    "weighted_token_count": round(weighted_count, 6),
                    "method_note": "duplicate-weighted song-level token counts; minimum five cluster songs and ten corpus songs",
                }
            )

        cluster_entity_rows: list[dict[str, Any]] = []
        for category in all_categories:
            a = sum(category in song_categories_by_index[int(index)] for index in indices)
            b = len(indices) - a
            c = sum(category in song_categories_by_index[int(index)] for index in rest_indices)
            d = len(rest_indices) - c
            if a + c == 0:
                continue
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            # Haldane-Anscombe correction makes an interval available even for zero cells.
            corrected = [a + 0.5, b + 0.5, c + 0.5, d + 0.5]
            corrected_log_or = math.log((corrected[0] * corrected[3]) / (corrected[1] * corrected[2]))
            corrected_se = math.sqrt(sum(1.0 / value for value in corrected))
            cluster_entity_rows.append(
                {
                    "cluster": f"C{cluster_label + 1}",
                    "canonical_category": category,
                    "presentation_group": presentation_group(category),
                    "evidence_tier": evidence_tier(category),
                    "primary_for_inference": True,
                    "cluster_songs_with_category": int(a),
                    "cluster_song_count": int(len(indices)),
                    "rest_songs_with_category": int(c),
                    "rest_song_count": int(len(rest_indices)),
                    "odds_ratio": float(odds_ratio) if math.isfinite(float(odds_ratio)) else float("inf"),
                    "odds_ratio_ci_low": math.exp(corrected_log_or - 1.96 * corrected_se),
                    "odds_ratio_ci_high": math.exp(corrected_log_or + 1.96 * corrected_se),
                    "p_value": float(p_value),
                }
            )
        adjusted = benjamini_hochberg([row["p_value"] for row in cluster_entity_rows])
        for row, fdr in zip(cluster_entity_rows, adjusted):
            row["fdr_bh"] = fdr
            row["enriched"] = bool(
                row["cluster_songs_with_category"] >= 5
                and row["odds_ratio"] > 1.0
                and fdr < 0.05
            )
            row["test_note"] = "two-sided Fisher exact test on song presence; Benjamini-Hochberg adjustment within cluster"
            enrichment_rows.append(row)

        top_keywords = [item[0] for item in scored_keywords[:3]]
        enriched_groups = []
        for row in sorted(
            (item for item in cluster_entity_rows if item.get("enriched")),
            key=lambda item: (-item["odds_ratio"], item["canonical_category"]),
        ):
            if row["presentation_group"] not in enriched_groups:
                enriched_groups.append(str(row["presentation_group"]))
        cluster_rows.append(
            {
                "cluster": f"C{cluster_label + 1}",
                "song_count": int(len(indices)),
                "song_share": round(float(len(indices) / len(song_ids)), 6),
                "top_keywords": " · ".join(top_keywords) if top_keywords else "No stable high-dispersion keyword met the predeclared threshold",
                "enriched_cultural_dimensions": " · ".join(enriched_groups[:3]) if enriched_groups else "No entity category passed FDR-controlled enrichment",
                "interpretation_boundary": "latent text cluster; not a genre, real-world scene, artist identity, or performance-style label",
            }
        )

    keyword_table = pd.DataFrame(keyword_rows).sort_values(["cluster", "keyword_rank"], kind="stable")
    enrichment_table = pd.DataFrame(enrichment_rows).sort_values(
        ["cluster", "fdr_bh", "p_value", "canonical_category"], kind="stable"
    )
    cluster_table = pd.DataFrame(cluster_rows).sort_values("cluster", kind="stable")
    return cluster_table, keyword_table, enrichment_table


def write_documents(summary: dict[str, Any]) -> None:
    """Write the public English method record before reader-facing results."""
    readme = f"""# Chinese Rap Cultural Coordinates v1

This is the first reproducible core analysis for the Chinese rap lyric project.
It asks a narrow question: **how do reviewed references to place, people,
collectives, cultural works, goods, and rap-language practice organize the
semantic landscape of the verified lyric corpus?**

The frozen canonical registry contains {summary['counts']['canonical_eligible_songs']:,}
eligible tracks and {summary['counts']['canonical_eligible_chunks']:,} lyric chunks.
The analytic population contains {summary['counts']['clean_analysis_songs']:,} tracks
and {summary['counts']['clean_analysis_chunks']:,} cleaned lyric chunks after a
strict, audited leading-header pass withheld
{summary['counts']['metadata_only_chunks_excluded']:,} metadata-only chunks.  Cross-song
repeated text is discounted through the frozen `analysis_text_weight` rather
than deleting records.

## What the outputs mean

- An entity mention is a reviewed literal surface re-matched in a cleaned
  lyric chunk.  It is not a claimed real-world identity or location.
- A co-mention is two entity categories in one cleaned lyric chunk.  It is not a
  friendship, collaboration, influence, or social relation.
- A cluster is an unsupervised group of semantically similar track texts.  It
  is not a music genre, artist persona, regional scene, or Flow label.
- UMAP is used only to lay out a private visual map.  It does not choose or
  validate clusters.

## Reproduce

```powershell
work\\semantic-ml-venv\\Scripts\\python.exe work\\build_chinese_rap_cultural_coordinates_v1.py
work\\semantic-ml-venv\\Scripts\\python.exe work\\validate_chinese_rap_cultural_coordinates_v1.py
```

Full lyrics, titles, source labels, and song-level coordinates are private
local artifacts and must not be bundled for external sharing.
"""
    protocol = f"""# Research protocol: Chinese Rap Cultural Coordinates v1

## Research question

How do **reviewed cultural references in Chinese rap lyrics**—especially place,
people and collectives, cultural works, material references, and rap-language
practice—help describe the semantic organization of a verified Chinese rap
lyric corpus?

The question is intentionally about **what the text writes and how textual
reference patterns cohere**.  It does not infer artists' biographical origin,
real relationships, personal preference, genre identity, Flow, voice, beat,
or performed rhyme.

## Corpus and unit of analysis

- Frozen canonical registry: {summary['counts']['canonical_eligible_songs']:,} eligible
  tracks and {summary['counts']['canonical_eligible_chunks']:,} lyric chunks.
- Clean analytic population: {summary['counts']['clean_analysis_songs']:,} tracks and
  {summary['counts']['clean_analysis_chunks']:,} lyric chunks.  The strict header-clean
  sidecar withheld {summary['counts']['metadata_only_chunks_excluded']:,} rows that
  contained only leading title/credit/production metadata after cleaning.
- Analysis unit: track for embeddings, clustering, stability, and enrichment;
  lyric chunk for literal reviewed-entity evidence.
- Repeated text: the frozen canonical duplicate-text design supplies a weight
  for every chunk.  Cross-song duplicate groups sum to one, so copied/repeated
  text cannot contribute once per source label.
- Names and titles: canonical matching has already validated record identity;
  this analysis uses exact IDs only and does not fuzzy-match labels.

## Entity evidence layer

The layer begins with {summary['counts']['reviewed_entity_catalog_surfaces']:,} reviewed
literal surfaces ({summary['counts']['casefold_distinct_match_patterns']:,} distinct
case-insensitive matcher patterns after collapsing equivalent ASCII casing).
Every item is re-matched against cleaned lyric text with a
longest-first, literal matcher.  ASCII/digit-bearing surfaces use token
boundaries.  The sidecar records a text hash and character span, not full lyric
text.  Co-mention means **same cleaned lyric chunk only**.

Fine-grained reviewed categories are grouped only for presentation:

1. Spatial references & local imaginaries
2. Rap craft & scene vocabulary
3. Language & dialect cues
4. Named people, crews & organizations
5. Cultural, historical & media references
6. Material life & consumption

Place surfaces are **not geocoded** in this version; language surfaces are not
proof that a song is performed in that language; people and organization
surfaces are not verified social relationships.  Exploratory identity labels,
hand-picked English tokens, and generic imagery remain audit-only and are
excluded from cluster-enrichment inference.

## Semantic model and clustering

Each track is represented by the duplicate-weighted mean of its verified local
BGE-M3 chunk vectors, then L2-normalized.  A deterministic PCA reduction feeds
KMeans.  Candidate `k` values are 4–10.  We predeclare a selection rule:

1. reject candidates with a smallest cluster below 3% of tracks;
2. retain candidates within 90% of the highest cosine silhouette; and
3. select the highest mean adjusted Rand index over song-level 80% subsamples;
   resolve a tie with silhouette and then lower `k`.

The two-dimensional UMAP map is calculated only after this selection.  It is a
reader-facing projection, not evidence that two points are related or that a
cluster is correct.

## Statistical explanation

Cluster words use duplicate-weighted, song-level token counts and a
corpus-informed log-odds z score.  A word must appear in at least five tracks
inside the cluster and ten in the corpus, avoiding a label driven by one
repeated lyric.

Reviewed entity categories are tested at the track level with two-sided Fisher
exact tests.  Benjamini–Hochberg correction is applied within a cluster.  A
category is described as enriched only if it appears in at least five cluster
tracks, has odds ratio above one, and passes FDR < 0.05.

## Evidence boundary and planned extensions

Text can support claims about literal reference, same-chunk co-mention,
duplicate-controlled frequency, and semantic similarity.  It cannot alone
prove collaboration, biography, locality, dialect pronunciation, Flow, beat,
or performed rhyme.  A future language/line annotation sidecar can add
audited structure, code-switching function, dialect candidates, and text-rhyme
candidates; audio-dependent claims require lawful audio and alignment data.

## Method context

- [Keyness in song lyrics: challenges of highly clumpy data](https://aclanthology.org/2023.jlcl-1.3/)
  motivates duplicate and dispersion controls.
- [Rappers as knights errant: classic allusions in the mainstreaming of Chinese rap](https://nottingham-repository.worktribe.com/output/3787336/rappers-as-knights-errant-classic-allusions-in-the-mainstreaming-of-chinese-rap)
  motivates a culture- and allusion-sensitive lens.
- [BGE-M3](https://arxiv.org/abs/2402.03216) documents the multilingual embedding
  family used here; this corpus-specific application is still validated locally.
- [The code of the streets in Beijing](https://www.cambridge.org/core/journals/language-in-society/article/abs/code-of-the-streets-in-beijing-styleshifting-and-changing-personae-in-the-performance-of-beijing-male-rappers/7787BC9FD8E7C55CAEACD59195F933DA)
  illustrates why performed persona and style cannot be inferred from lyric
  text alone.
"""
    dictionary = """# Data dictionary

## Public aggregate outputs

- `category_summary.csv` — reviewed entity categories observed in cleaned
  lyric text. `primary_for_inference` says whether a category enters the
  predeclared enrichment hypothesis family. `weighted_chunk_presence` uses the frozen duplicate-text
  weights; a category is counted at most once per chunk.
- `entity_summary.csv` — reviewed literal surfaces with duplicate-weighted
  occurrence summaries and an explicit evidence tier. It contains no lyric excerpts, songs, titles, or
  source labels.
- `cluster_model_selection.csv` — unsupervised cluster candidate diagnostics.
  `bootstrap_ari_mean` is the mean agreement between the reference partition
  and a partition fitted on an 80% song-level subsample.
- `cluster_summary.csv` — cautious, aggregate descriptions of selected latent
  text clusters.
- `cluster_keyword_summary.csv` — high-dispersion descriptive words only; not
  a topic model or a quote collection.
- `cluster_entity_enrichment.csv` — FDR-controlled entity-category enrichment
  tests at the track level.

## Private local artifacts

- `entity_mentions_v1.csv` — song/chunk identifiers, original and clean-text
  hashes, reviewed surface, category, and
  character span; no lyric text.
- `category_co_mentions_v1.csv` — categories co-mentioned within a chunk; no
  claim of a real-world relationship.
- `song_feature_map_v1.csv` and `song_cluster_assignments_v1.csv` — titles,
  source labels, track-level coordinates, and assignments for local-only
  inspection. These must not be added to a generic sharing bundle.
"""
    atomic_write_text(OUTPUT_DIR / "README.md", readme)
    atomic_write_text(OUTPUT_DIR / "research_protocol.md", protocol)
    atomic_write_text(OUTPUT_DIR / "data_dictionary.md", dictionary)


def chart_contracts(summary: dict[str, Any]) -> dict[str, Any]:
    """Document visuals before a later English explorer is built."""
    return {
        "artifact_id": ARTIFACT_ID,
        "visuals": [
            {
                "id": "cluster_map",
                "question": "How does the duplicate-weighted semantic representation distribute across selected latent text clusters?",
                "chart_family": "relationship",
                "chart_type": "scatter",
                "observation_grain": "eligible track",
                "x": "UMAP dimension 1 (visual projection only)",
                "y": "UMAP dimension 2 (visual projection only)",
                "color": "selected latent text cluster",
                "takeaway_boundary": "Point proximity is visual only; cluster meaning is explained through independent token and entity evidence.",
                "data_scope": f"{summary['counts']['clean_analysis_songs']} clean-text analytic tracks",
            },
            {
                "id": "entity_enrichment",
                "question": "Which reviewed cultural-reference categories are over-represented inside each text cluster?",
                "chart_family": "comparison",
                "chart_type": "heatmap",
                "observation_grain": "track-by-reviewed-category presence",
                "value": "log odds ratio; show only FDR-controlled enriched cells",
                "takeaway_boundary": "An enriched category is a text-reference pattern, not a statement about a performer or scene.",
            },
            {
                "id": "model_stability",
                "question": "Does the selected number of latent text clusters remain stable under song-level subsampling?",
                "chart_family": "comparison",
                "chart_type": "dot-and-interval",
                "observation_grain": "candidate k",
                "value": "mean adjusted Rand index across song-level subsamples",
                "takeaway_boundary": "Stability supports reproducibility of this partition, not a claim of causal or social structure.",
            },
        ],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_replicates < 3:
        raise BuildError("Use at least three song-level subsamples for a stability estimate.")
    require_directory_allowlist(OUTPUT_DIR, PUBLIC_ALLOWLIST, "Public cultural-coordinate output")
    require_directory_allowlist(PRIVATE_DIR, PRIVATE_ALLOWLIST, "Private cultural-coordinate output")

    chunks, songs, input_metadata, input_paths = load_canonical_input()
    vectors, row_map, embedding_contract, embedding_paths = load_embedding_matrix(chunks)
    catalog = load_reviewed_lexicon()
    mentions, entity_summary, category_summary, pair_summary, song_categories = re_match_entities(chunks, catalog)
    song_ids, song_vectors, song_metadata = aggregate_song_embeddings(chunks, songs, vectors, row_map)
    pca_vectors, labels, clustering_metadata, selection_table = select_clusters(song_vectors, args.bootstrap_replicates)
    coordinates = None if args.skip_umap else build_umap_map(pca_vectors)
    cluster_summary, keyword_summary, enrichment_summary = build_cluster_descriptions(
        chunks, song_ids, labels, song_categories
    )

    if len(set(labels)) != clustering_metadata["selected_k"]:
        raise BuildError("The selected KMeans partition does not contain the declared number of clusters.")
    if len(cluster_summary) != clustering_metadata["selected_k"]:
        raise BuildError("Cluster description table does not cover every selected cluster.")

    summary = {
        "artifact_id": ARTIFACT_ID,
        "method_version": METHOD_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "canonical_eligible_songs": EXPECTED_CANONICAL_SONGS,
            "canonical_eligible_chunks": EXPECTED_CANONICAL_CHUNKS,
            "clean_analysis_songs": int(len(songs)),
            "clean_analysis_chunks": int(len(chunks)),
            "metadata_only_chunks_excluded": int(
                input_metadata["clean_text_sidecar"]["counts"]["withheld_metadata_only_chunks"]
            ),
            "leading_header_lines_removed": int(
                input_metadata["clean_text_sidecar"]["counts"]["leading_header_lines_removed"]
            ),
            "reviewed_entity_catalog_surfaces": int(catalog.attrs["raw_surface_count"]),
            "casefold_distinct_match_patterns": int(len(catalog)),
            "casefold_collapsed_variant_count": int(catalog.attrs["casefold_collapsed_variant_count"]),
            "literal_entity_spans": int(len(mentions)),
            "entities_matched": int(len(entity_summary)),
            "entity_categories_matched": int(len(category_summary)),
            "tracks_with_any_reviewed_entity": int(sum(bool(song_categories.get(song_id)) for song_id in song_ids)),
            "selected_clusters": int(clustering_metadata["selected_k"]),
        },
        "clustering": clustering_metadata,
        "evidence_boundary": {
            "entity_relation": "same cleaned lyric chunk co-mention only",
            "not_inferred": [
                "real-world relationship", "collaboration", "artist biography", "artist locality",
                "artist preference", "music genre", "Flow", "beat", "voice", "performed rhyme",
            ],
            "map_use": "UMAP is a private visual projection only; it does not fit or validate clusters.",
        },
    }

    # Public results are entirely aggregate.  Song-level and title/label data
    # are written only below the private directory.
    public_entity_rows = entity_summary.to_dict("records")
    public_category_rows = category_summary.to_dict("records")
    public_selection_rows = selection_table.to_dict("records")
    public_cluster_rows = cluster_summary.to_dict("records")
    public_keyword_rows = keyword_summary.to_dict("records")
    public_enrichment_rows = enrichment_summary.to_dict("records")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    write_documents(summary)
    atomic_write_json(OUTPUT_DIR / "analysis_summary.json", summary)
    atomic_write_json(OUTPUT_DIR / "chart_contracts.json", chart_contracts(summary))
    atomic_write_csv(
        OUTPUT_DIR / "category_summary.csv",
        ["canonical_category", "presentation_group", "evidence_tier", "primary_for_inference", "song_count", "chunk_count", "weighted_chunk_presence", "entity_surface_count", "unique_text_count"],
        public_category_rows,
    )
    atomic_write_csv(
        OUTPUT_DIR / "entity_summary.csv",
        ["entity", "canonical_category", "presentation_group", "evidence_tier", "primary_for_inference", "song_count", "chunk_count", "weighted_chunk_presence", "span_occurrences", "unique_text_count"],
        public_entity_rows,
    )
    atomic_write_csv(
        OUTPUT_DIR / "cluster_model_selection.csv",
        ["k", "silhouette_cosine", "bootstrap_ari_mean", "bootstrap_ari_sd", "bootstrap_replicates", "min_cluster_song_count", "max_cluster_song_count", "min_cluster_share", "pca_components", "selected", "selection_rule"],
        public_selection_rows,
    )
    atomic_write_csv(
        OUTPUT_DIR / "cluster_summary.csv",
        ["cluster", "song_count", "song_share", "top_keywords", "enriched_cultural_dimensions", "interpretation_boundary"],
        public_cluster_rows,
    )
    atomic_write_csv(
        OUTPUT_DIR / "cluster_keyword_summary.csv",
        ["cluster", "keyword_rank", "keyword", "log_odds_z", "cluster_song_count", "weighted_token_count", "method_note"],
        public_keyword_rows,
    )
    atomic_write_csv(
        OUTPUT_DIR / "cluster_entity_enrichment.csv",
        ["cluster", "canonical_category", "presentation_group", "evidence_tier", "primary_for_inference", "cluster_songs_with_category", "cluster_song_count", "rest_songs_with_category", "rest_song_count", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high", "p_value", "fdr_bh", "enriched", "test_note"],
        public_enrichment_rows,
    )

    # Private evidence sidecars.  Mentions intentionally retain no lyric text;
    # only the local feature map includes canonical source labels/titles.
    atomic_write_csv(
        PRIVATE_DIR / "entity_mentions_v1.csv",
        ["song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256", "entity", "canonical_category", "presentation_group", "evidence_tier", "primary_for_inference", "start_char", "end_char", "analysis_text_weight"],
        mentions.to_dict("records"),
    )
    atomic_write_csv(
        PRIVATE_DIR / "category_co_mentions_v1.csv",
        ["left_category", "right_category", "left_presentation_group", "right_presentation_group", "left_primary_for_inference", "right_primary_for_inference", "song_count", "chunk_count", "weighted_chunk_presence", "relation_definition"],
        pair_summary.to_dict("records"),
    )
    private_map = song_metadata.copy()
    private_map["cluster"] = [f"C{int(label) + 1}" for label in labels]
    private_map["semantic_vector_l2_norm"] = np.linalg.norm(song_vectors, axis=1).round(7)
    if coordinates is not None:
        private_map["umap_x"] = coordinates[:, 0]
        private_map["umap_y"] = coordinates[:, 1]
    else:
        private_map["umap_x"] = ""
        private_map["umap_y"] = ""
    atomic_write_csv(
        PRIVATE_DIR / "song_feature_map_v1.csv",
        ["song_id", "canonical_artist", "canonical_song_title", "clean_analysis_chunk_count", "cluster", "semantic_vector_l2_norm", "umap_x", "umap_y"],
        private_map.to_dict("records"),
    )
    atomic_write_csv(
        PRIVATE_DIR / "song_cluster_assignments_v1.csv",
        ["song_id", "cluster"],
        private_map[["song_id", "cluster"]].to_dict("records"),
    )

    # A manifest cannot reasonably checksum itself.  Its file inventory covers
    # every other public payload; the independent validator checks the manifest
    # and validation JSON separately.
    public_files = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "validation.json"}
    )
    private_manifest = {
        "artifact_id": ARTIFACT_ID,
        "method_version": METHOD_VERSION,
        "classification": "private_local_only_song_level_and_evidence_sidecars",
        "files": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(PRIVATE_DIR.iterdir()) if path.is_file() and path.name != "private_manifest.json" and path.name != "private_validation.json"},
    }
    atomic_write_json(PRIVATE_DIR / "private_manifest.json", private_manifest)
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "method_version": METHOD_VERSION,
        "canonical_input": {
            "analysis_input_id": input_metadata["canonical"]["analysis_input_id"],
            "canonical_corpus_id": input_metadata["canonical"]["canonical_source_contract"]["canonical_corpus_id"],
            "canonical_contract_sha256": input_metadata["canonical"]["canonical_source_contract"]["sha256"],
            "private_chunk_input_sha256": sha256_file(input_paths["canonical_chunks"]),
            "private_song_input_sha256": sha256_file(input_paths["canonical_songs"]),
        },
        "clean_text_input": {
            "artifact_id": input_metadata["clean_text_sidecar"]["artifact_id"],
            "cleaning_version": input_metadata["clean_text_sidecar"]["cleaning_version"],
            "private_cleaned_chunk_input_sha256": sha256_file(input_paths["cleaned_chunks"]),
            "clean_analysis_songs": summary["counts"]["clean_analysis_songs"],
            "clean_analysis_chunks": summary["counts"]["clean_analysis_chunks"],
            "metadata_only_chunks_excluded": summary["counts"]["metadata_only_chunks_excluded"],
        },
        "embedding_input": {
            "artifact_id": embedding_contract["artifact_id"],
            "vector_sha256": sha256_file(embedding_paths["vectors"]),
            "row_map_sha256": sha256_file(embedding_paths["row_map"]),
            "model": embedding_contract["model"]["implementation"],
        },
        "reviewed_entity_lexicon": {
            "catalog_path": str(LEXICON_CATALOG.relative_to(ROOT)),
            "catalog_sha256": sha256_file(LEXICON_CATALOG),
            "catalog_surfaces": int(catalog.attrs["raw_surface_count"]),
            "casefold_distinct_match_patterns": int(len(catalog)),
            "casefold_collapsed_variant_count": int(catalog.attrs["casefold_collapsed_variant_count"]),
            "matching": "longest-first literal re-match; ASCII/digit token boundaries",
        },
        "privacy": {
            "generic_output_contains_full_lyrics": False,
            "generic_output_contains_song_ids": False,
            "generic_output_contains_titles_or_source_labels": False,
            "private_directory": str(PRIVATE_DIR.relative_to(ROOT)),
        },
        "counts": summary["counts"],
        "method": summary["clustering"],
        "output_files": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in public_files},
    }
    atomic_write_json(OUTPUT_DIR / "manifest.json", manifest)
    atomic_write_json(
        PRIVATE_DIR / "private_validation.json",
        {
            "artifact_id": ARTIFACT_ID,
            "status": "pending_independent_validation",
            "required_validator": "work/validate_chinese_rap_cultural_coordinates_v1.py",
        },
    )
    atomic_write_json(
        OUTPUT_DIR / "validation.json",
        {
            "artifact_id": ARTIFACT_ID,
            "status": "pending_independent_validation",
            "required_validator": "work/validate_chinese_rap_cultural_coordinates_v1.py",
        },
    )
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = build(args)
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 2
    print(
        "Built "
        f"{ARTIFACT_ID}: {summary['counts']['clean_analysis_songs']} clean-text tracks, "
        f"{summary['counts']['clean_analysis_chunks']} clean-text chunks, "
        f"{summary['counts']['selected_clusters']} selected clusters."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
