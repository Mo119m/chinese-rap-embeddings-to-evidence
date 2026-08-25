#!/usr/bin/env python3
"""Build a duplicate-controlled Chinese rap lyrical-repertoire graph.

Research question: which *artist-labelled corpus slices* have nearby clean
lyric repertoires in BGE-M3 semantic space, and which neighbourhoods remain
after exact cleaned text shared across labels is removed?

This is a text-only analysis.  A retained edge means mutual, stable lyrical
repertoire proximity under two duplicate-control representations.  It is never
evidence of collaboration, friendship, influence, crew membership, hometown,
genre, Flow, voice, beat, or any artist preference.  Source artist labels are
shown as corpus labels until an external identity registry is supplied.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

import build_chinese_rap_cultural_coordinates_v1 as clean_core
import build_canonical_song_metadata_quality_v1 as metadata_quality


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ID = "chinese-rap-lyrical-repertoire-graph-v2"
VERSION = "2.3.0"
TOP_K = 5
MIN_CLEAN_SONGS = 5
MIN_EFFECTIVE_TEXT_MASS = 20.0
LABEL_STATUS = "source_artist_label_not_externally_identity_verified"
EDGE_RULE = "mutual_top_5_neighbours_in_primary_duplicate_weighted_chunk_centroids_and_shared_text_exclusion_sensitivity"
COMPARISON_POPULATION_POLICY = "canonical_artist_title_comparison_eligible=true"
COMPARISON_DUPLICATE_WEIGHT_POLICY = "within_metadata_eligible_population_exact_clean_text_hash_total_mass_one"
PRIVATE_DIR = ROOT / "work" / "private-chinese-rap-lyrical-repertoire-graph-v2"
OUTPUT_DIR = ROOT / "outputs" / ARTIFACT_ID
PUBLIC_ALLOWLIST = {
    "README.md",
    "research_protocol.md",
    "data_dictionary.md",
    "analysis_summary.json",
    "artist_label_registry.csv",
    "artist_repertoire_nodes.csv",
    "artist_repertoire_edges.csv",
    "artist_repertoire_layout.csv",
    "robustness_summary.csv",
    "manifest.json",
    "validation.json",
}
PRIVATE_ALLOWLIST = {
    "artist_chunk_membership_v2.csv",
    "artist_repertoire_vectors_v2.npy",
    "artist_repertoire_vector_rowmap_v2.csv",
    "neighbor_rank_audit_v2.csv",
    "private_manifest.json",
    "private_validation.json",
}
REGISTRY_COLUMNS = (
    "artist_label_id",
    "source_artist_label",
    "label_attribution_status",
    "external_identity_verified",
    "graph_display_status",
)
NODE_COLUMNS = (
    "artist_label_id",
    "source_artist_label",
    "label_attribution_status",
    "clean_song_count",
    "independent_clean_song_count",
    "primary_raw_chunk_count",
    "primary_effective_text_mass",
    "primary_unique_clean_text_count",
    "sensitivity_raw_chunk_count",
    "sensitivity_effective_text_mass",
    "sensitivity_unique_clean_text_count",
    "shared_text_dropped_effective_mass",
    "shared_text_dropped_mass_share",
    "eligible_primary_support",
    "eligible_sensitivity_support",
    "graph_node_eligible",
    "stable_graph_degree",
)
EDGE_COLUMNS = (
    "artist_label_id_a",
    "source_artist_label_a",
    "artist_label_id_b",
    "source_artist_label_b",
    "primary_rank_a_to_b",
    "primary_rank_b_to_a",
    "primary_cosine_similarity",
    "primary_pair_percentile",
    "sensitivity_rank_a_to_b",
    "sensitivity_rank_b_to_a",
    "sensitivity_cosine_similarity",
    "sensitivity_pair_percentile",
    "stable_across_shared_text_exclusion",
    "edge_definition",
)
LAYOUT_COLUMNS = (
    "artist_label_id",
    "x",
    "y",
    "component_id",
    "stable_graph_degree",
    "projection_population",
    "projection_variance_explained_2d",
    "layout_note",
)
ROBUSTNESS_COLUMNS = ("metric", "value", "interpretation")
MEMBERSHIP_COLUMNS = (
    "artist_label_id",
    "source_artist_label",
    "song_id",
    "chunk_id",
    "canonical_lyric_text_sha256",
    "analysis_text_sha256",
    "frozen_analysis_text_weight",
    "comparison_text_weight",
    "clean_row_index",
    "shared_across_artist_labels",
    "included_in_primary_centroid",
    "included_in_shared_text_exclusion_sensitivity",
)
VECTOR_ROWMAP_COLUMNS = (
    "vector_row_index",
    "artist_label_id",
    "source_artist_label",
    "graph_node_eligible",
    "primary_effective_text_mass",
    "sensitivity_effective_text_mass",
)
NEIGHBOR_AUDIT_COLUMNS = (
    "artist_label_id",
    "source_artist_label",
    "representation",
    "neighbor_rank",
    "neighbor_artist_label_id",
    "neighbor_source_artist_label",
    "cosine_similarity",
    "pair_percentile",
    "mutual_top_k",
)


class RepertoireError(RuntimeError):
    """Raised when a graph input or persisted output violates the protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def artist_label_id(label: str) -> str:
    return "ALBL-" + sha256_text(f"chinese-rap-source-artist-label-v2\t{label}")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    atomic_write_text(path, buffer.getvalue())


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RepertoireError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RepertoireError(f"Invalid JSON in {label}.") from exc
    if not isinstance(payload, dict):
        raise RepertoireError(f"{label} must be a JSON object.")
    return payload


def read_csv_exact(path: Path, expected: tuple[str, ...], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise RepertoireError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != expected:
            raise RepertoireError(f"{label} schema mismatch; expected={list(expected)}, actual={list(actual)}")
        return list(reader)


def require_exact_allowlist(directory: Path, allowed: set[str], label: str, permit_missing_validation: bool = False) -> None:
    if not directory.exists():
        return
    if not directory.is_dir():
        raise RepertoireError(f"{label} is not a directory.")
    actual = {item.name for item in directory.iterdir()}
    acceptable = {frozenset(allowed)}
    if permit_missing_validation:
        acceptable.add(frozenset(name for name in allowed if not name.endswith("validation.json")))
    if frozenset(actual) not in acceptable or any(item.is_dir() for item in directory.iterdir()):
        raise RepertoireError(f"{label} contains undeclared files or nested directories.")


def decimal(value: float, places: int = 8) -> str:
    if not math.isfinite(float(value)):
        raise RepertoireError("A graph statistic is non-finite.")
    return f"{float(value):.{places}f}"


def input_hashes() -> dict[str, str]:
    return {
        "canonical_analysis_manifest_sha256": sha256_file(clean_core.ANALYSIS_MANIFEST),
        "clean_text_sidecar_manifest_sha256": sha256_file(clean_core.TEXT_SIDECAR_MANIFEST),
        "clean_bge_manifest_sha256": sha256_file(clean_core.CLEAN_EMBEDDING_MANIFEST),
        "metadata_quality_manifest_sha256": sha256_file(metadata_quality.OUTPUT_DIR / "manifest.json"),
        "metadata_quality_private_manifest_sha256": sha256_file(metadata_quality.PRIVATE_DIR / "private_manifest.json"),
        "metadata_quality_validation_sha256": sha256_file(metadata_quality.OUTPUT_DIR / "validation.json"),
    }


def load_aligned_input() -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Reuse the exact clean text/BGE loaders but no NER code or lexicon."""
    metadata_validation = metadata_quality.validate_persisted()
    if metadata_validation.get("status") != "pass":
        raise RepertoireError("The song metadata-quality sidecar is not currently validated.")
    chunks, songs, _lineage, _paths = clean_core.load_canonical_input()
    vectors, row_map, _embedding_manifest, _embedding_paths = clean_core.load_embedding_matrix(chunks)
    key_columns = ["song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256"]
    if vectors.shape != (len(chunks), clean_core.EMBEDDING_DIMENSIONS):
        raise RepertoireError("The clean BGE matrix shape does not match the clean lyric population.")
    if row_map["clean_row_index"].duplicated().any() or set(row_map["clean_row_index"].astype(int)) != set(range(len(vectors))):
        raise RepertoireError("The clean BGE row map does not contain exactly one index per vector.")
    joined = chunks.merge(
        row_map[key_columns + ["clean_row_index"]],
        on=key_columns,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(joined) != len(chunks) or joined["clean_row_index"].isna().any():
        raise RepertoireError("A clean lyric chunk cannot rejoin to exactly one BGE vector.")
    joined["clean_row_index"] = joined["clean_row_index"].astype(int)
    song_columns = [
        "song_id",
        "canonical_artist",
        "analysis_deduplication_required",
        "artist_title_comparison_eligible",
    ]
    song_registry = songs[song_columns].copy()
    if song_registry["song_id"].duplicated().any() or song_registry["canonical_artist"].astype(str).str.strip().eq("").any():
        raise RepertoireError("The canonical song registry has an invalid artist-label join.")
    if not song_registry["artist_title_comparison_eligible"].isin(["true", "false"]).all():
        raise RepertoireError("The canonical song comparison-eligibility field must be literal true/false.")
    metadata_sidecar = pd.read_csv(metadata_quality.PRIVATE_DIR / "canonical_song_metadata_quality_v1.csv", dtype=str, keep_default_na=False)
    required_metadata_columns = {"song_id", "source_artist_title_comparison_eligible"}
    if not required_metadata_columns.issubset(metadata_sidecar.columns) or metadata_sidecar["song_id"].duplicated().any():
        raise RepertoireError("The metadata-quality sidecar cannot validate the comparison-eligibility join.")
    eligibility_check = song_registry.merge(
        metadata_sidecar[["song_id", "source_artist_title_comparison_eligible"]],
        on="song_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if eligibility_check["source_artist_title_comparison_eligible"].isna().any() or not eligibility_check["artist_title_comparison_eligible"].eq(eligibility_check["source_artist_title_comparison_eligible"]).all():
        raise RepertoireError("The canonical and metadata-quality comparison-eligibility fields disagree.")
    pre_filter = joined.drop(columns=["canonical_artist", "analysis_deduplication_required", "artist_title_comparison_eligible"], errors="ignore").merge(
        song_registry,
        on="song_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if pre_filter["canonical_artist"].isna().any() or pre_filter["analysis_deduplication_required"].isna().any() or pre_filter["artist_title_comparison_eligible"].isna().any():
        raise RepertoireError("A clean lyric chunk cannot rejoin to its canonical source artist label.")
    joined = pre_filter.loc[pre_filter["artist_title_comparison_eligible"].eq("true")].copy()
    if joined.empty:
        raise RepertoireError("The comparison-eligible song filter removed every clean lyric chunk.")
    if not joined["analysis_text_weight"].between(0.0, 1.0, inclusive="right").all():
        raise RepertoireError("Duplicate-control text weights must be in (0, 1].")
    if joined["analysis_text_sha256"].str.fullmatch(r"[0-9a-f]{64}").eq(False).any():
        raise RepertoireError("Clean text hash format is invalid.")
    # The canonical weights were frozen before this graph's conservative song
    # filter. Recompute exact-clean-text weights inside the retained comparison
    # population so a surviving copy is never penalized for a copy we excluded.
    comparison_counts = joined.groupby("analysis_text_sha256", sort=False)["analysis_text_sha256"].transform("size")
    joined["comparison_text_weight"] = 1.0 / comparison_counts.astype(float)
    comparison_sums = joined.groupby("analysis_text_sha256", sort=False)["comparison_text_weight"].sum()
    if not np.allclose(comparison_sums.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-12):
        raise RepertoireError("Comparison-population duplicate weights must sum to one per exact clean-text hash.")
    return joined, vectors, {
        "hashes": input_hashes(),
        "pre_filter_clean_chunks": int(len(pre_filter)),
        "pre_filter_clean_songs": int(pre_filter["song_id"].nunique()),
        "pre_filter_source_labels": int(pre_filter["canonical_artist"].nunique()),
        "comparison_ineligible_clean_chunks_excluded": int(len(pre_filter) - len(joined)),
        "comparison_ineligible_songs_excluded": int(pre_filter.loc[pre_filter["artist_title_comparison_eligible"].eq("false"), "song_id"].nunique()),
        "comparison_unique_clean_text_groups": int(joined["analysis_text_sha256"].nunique()),
        "comparison_effective_text_mass": float(joined["comparison_text_weight"].sum()),
        "clean_chunks": int(len(joined)),
        "clean_songs": int(joined["song_id"].nunique()),
        "source_labels": int(joined["canonical_artist"].nunique()),
    }


def shared_text_hashes(frame: pd.DataFrame) -> set[str]:
    label_count = frame.groupby("analysis_text_sha256", sort=False)["canonical_artist"].nunique()
    return set(label_count.loc[label_count.gt(1)].index.astype(str))


def aggregate_artist_vectors(
    frame: pd.DataFrame, vectors: np.ndarray, exclude_shared: bool
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], pd.DataFrame]:
    work = frame.copy()
    shared = shared_text_hashes(work)
    work["shared_across_artist_labels"] = work["analysis_text_sha256"].isin(shared)
    if exclude_shared:
        work = work.loc[~work["shared_across_artist_labels"]].copy()
    sums: dict[str, np.ndarray] = {}
    support: dict[str, dict[str, Any]] = {}
    for label, group in work.groupby("canonical_artist", sort=True):
        indices = group["clean_row_index"].to_numpy(dtype=int)
        weights = group["comparison_text_weight"].to_numpy(dtype=float)
        effective_mass = float(weights.sum())
        if effective_mass <= 0:
            continue
        weighted_vector = np.einsum("i,ij->j", weights, vectors[indices], optimize=True) / effective_mass
        norm = float(np.linalg.norm(weighted_vector))
        if not math.isfinite(norm) or norm <= 0:
            raise RepertoireError("An artist-label centroid is degenerate.")
        string_label = str(label)
        sums[string_label] = (weighted_vector / norm).astype(np.float32)
        support[string_label] = {
            "raw_chunk_count": int(len(group)),
            "effective_text_mass": effective_mass,
            "unique_clean_text_count": int(group["analysis_text_sha256"].nunique()),
            "clean_song_count": int(group["song_id"].nunique()),
            "independent_clean_song_count": int(group.loc[group["analysis_deduplication_required"].eq("false"), "song_id"].nunique()),
        }
    return sums, support, work


def all_label_support(frame: pd.DataFrame, primary: dict[str, dict[str, Any]], sensitivity: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    labels = sorted(frame["canonical_artist"].astype(str).unique())
    result: dict[str, dict[str, Any]] = {}
    for label in labels:
        primary_support = primary.get(label, {"raw_chunk_count": 0, "effective_text_mass": 0.0, "unique_clean_text_count": 0, "clean_song_count": 0, "independent_clean_song_count": 0})
        sensitivity_support = sensitivity.get(label, {"raw_chunk_count": 0, "effective_text_mass": 0.0, "unique_clean_text_count": 0, "clean_song_count": 0, "independent_clean_song_count": 0})
        primary_eligible = (
            int(primary_support["clean_song_count"]) >= MIN_CLEAN_SONGS
            and float(primary_support["effective_text_mass"]) >= MIN_EFFECTIVE_TEXT_MASS
        )
        sensitivity_eligible = (
            int(sensitivity_support["clean_song_count"]) >= MIN_CLEAN_SONGS
            and float(sensitivity_support["effective_text_mass"]) >= MIN_EFFECTIVE_TEXT_MASS
        )
        result[label] = {
            "primary": primary_support,
            "sensitivity": sensitivity_support,
            "primary_eligible": primary_eligible,
            "sensitivity_eligible": sensitivity_eligible,
            "graph_eligible": primary_eligible and sensitivity_eligible,
        }
    return result


def cosine_matrix(label_order: list[str], vectors: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.vstack([vectors[label] for label in label_order]).astype(np.float64)
    values = matrix @ matrix.T
    np.fill_diagonal(values, -np.inf)
    if not np.isfinite(values[np.triu_indices(len(label_order), 1)]).all():
        raise RepertoireError("A repertoire cosine matrix contains invalid off-diagonal values.")
    return values


def neighbor_tables(label_order: list[str], similarities: np.ndarray, representation: str) -> tuple[dict[str, dict[str, int]], dict[tuple[str, str], float], list[dict[str, str]]]:
    ranks: dict[str, dict[str, int]] = {}
    pair_percentile: dict[tuple[str, str], float] = {}
    upper = similarities[np.triu_indices(len(label_order), 1)]
    audit_rows: list[dict[str, str]] = []
    for index, label in enumerate(label_order):
        ordered_indices = sorted(
            (candidate for candidate in range(len(label_order)) if candidate != index),
            key=lambda candidate: (-float(similarities[index, candidate]), label_order[candidate]),
        )
        ranks[label] = {}
        for rank, candidate in enumerate(ordered_indices, start=1):
            neighbor = label_order[candidate]
            ranks[label][neighbor] = rank
            score = float(similarities[index, candidate])
            pair_key = tuple(sorted((label, neighbor)))
            pair_percentile[pair_key] = float(np.mean(upper <= score) * 100.0)
            if rank <= TOP_K:
                audit_rows.append(
                    {
                        "artist_label_id": artist_label_id(label),
                        "source_artist_label": label,
                        "representation": representation,
                        "neighbor_rank": str(rank),
                        "neighbor_artist_label_id": artist_label_id(neighbor),
                        "neighbor_source_artist_label": neighbor,
                        "cosine_similarity": decimal(score),
                        "pair_percentile": decimal(pair_percentile[pair_key]),
                        "mutual_top_k": "",  # filled after both directions are known
                    }
                )
    for row in audit_rows:
        label = row["source_artist_label"]
        neighbor = row["neighbor_source_artist_label"]
        row["mutual_top_k"] = "true" if ranks.get(neighbor, {}).get(label, TOP_K + 1) <= TOP_K else "false"
    return ranks, pair_percentile, audit_rows


def stable_edges(
    label_order: list[str],
    primary_scores: np.ndarray,
    sensitivity_scores: np.ndarray,
    primary_ranks: dict[str, dict[str, int]],
    sensitivity_ranks: dict[str, dict[str, int]],
    primary_percentile: dict[tuple[str, str], float],
    sensitivity_percentile: dict[tuple[str, str], float],
) -> tuple[list[dict[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    primary_mutual: set[tuple[str, str]] = set()
    sensitivity_mutual: set[tuple[str, str]] = set()
    index = {label: position for position, label in enumerate(label_order)}
    for left_position, left in enumerate(label_order):
        for right in label_order[left_position + 1:]:
            pair = (left, right)
            if primary_ranks[left][right] <= TOP_K and primary_ranks[right][left] <= TOP_K:
                primary_mutual.add(pair)
            if sensitivity_ranks[left][right] <= TOP_K and sensitivity_ranks[right][left] <= TOP_K:
                sensitivity_mutual.add(pair)
    stable = sorted(primary_mutual & sensitivity_mutual)
    rows: list[dict[str, str]] = []
    for left, right in stable:
        left_index, right_index = index[left], index[right]
        pair = (left, right)
        rows.append(
            {
                "artist_label_id_a": artist_label_id(left),
                "source_artist_label_a": left,
                "artist_label_id_b": artist_label_id(right),
                "source_artist_label_b": right,
                "primary_rank_a_to_b": str(primary_ranks[left][right]),
                "primary_rank_b_to_a": str(primary_ranks[right][left]),
                "primary_cosine_similarity": decimal(float(primary_scores[left_index, right_index])),
                "primary_pair_percentile": decimal(primary_percentile[pair]),
                "sensitivity_rank_a_to_b": str(sensitivity_ranks[left][right]),
                "sensitivity_rank_b_to_a": str(sensitivity_ranks[right][left]),
                "sensitivity_cosine_similarity": decimal(float(sensitivity_scores[left_index, right_index])),
                "sensitivity_pair_percentile": decimal(sensitivity_percentile[pair]),
                "stable_across_shared_text_exclusion": "true",
                "edge_definition": EDGE_RULE,
            }
        )
    return rows, primary_mutual, sensitivity_mutual


def graph_layout(
    node_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
    vector_rowmap: list[dict[str, str]],
    vector_matrix: np.ndarray,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return a deterministic semantic PCA projection plus stable-edge topology.

    The original component-ring layout was visually tidy but analytically
    meaningless.  This projection uses the average of each label's primary and
    shared-text-exclusion unit centroids, calculated over all graph-eligible
    labels.  It makes spatial proximity an approximate, visible summary of the
    same textual representation used for the graph.  Stable edges remain the
    stricter, discrete result; 2D geometry is never a substitute for an edge.
    """
    eligible_ids = {row["artist_label_id"] for row in node_rows if row["graph_node_eligible"] == "true"}
    if not eligible_ids:
        return [], {"method": "consensus_semantic_pca", "projection_population": 0, "variance_explained_2d": 0.0}
    rowmap_ids = [row["artist_label_id"] for row in vector_rowmap]
    if len(rowmap_ids) != len(set(rowmap_ids)) or set(rowmap_ids) != eligible_ids:
        raise RepertoireError("Vector row map does not exactly cover graph-eligible labels for semantic projection.")
    if vector_matrix.ndim != 2 or vector_matrix.shape[0] != len(rowmap_ids) or vector_matrix.shape[1] < 4 or vector_matrix.shape[1] % 2:
        raise RepertoireError("Invalid paired primary/sensitivity vector matrix for semantic projection.")

    half = vector_matrix.shape[1] // 2
    consensus = vector_matrix[:, :half].astype(np.float64) + vector_matrix[:, half:].astype(np.float64)
    norms = np.linalg.norm(consensus, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(consensus).all():
        raise RepertoireError("Semantic consensus vectors are invalid.")
    consensus /= norms
    centered = consensus - consensus.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * singular_values[:2]
    variance = singular_values * singular_values
    total_variance = float(variance.sum())
    if total_variance <= 0 or not np.isfinite(scores).all():
        raise RepertoireError("Semantic PCA projection is degenerate.")
    # SVD component signs are arbitrary.  Fix them by a deterministic anchor so
    # reruns produce byte-stable public coordinates.
    for column in range(2):
        anchor = int(np.argmax(np.abs(scores[:, column])))
        if scores[anchor, column] < 0:
            scores[:, column] *= -1.0
    scale = float(np.max(np.abs(scores)))
    if scale <= 0:
        raise RepertoireError("Semantic PCA scale is invalid.")
    scores /= scale
    coordinate_by_id = {label_id: (float(scores[index, 0]), float(scores[index, 1])) for index, label_id in enumerate(rowmap_ids)}

    graph = nx.Graph()
    graph.add_nodes_from(sorted(eligible_ids))
    graph.add_edges_from((row["artist_label_id_a"], row["artist_label_id_b"]) for row in edge_rows)
    connected_nodes = sorted(node for node, degree in graph.degree() if degree > 0)
    subgraph = graph.subgraph(connected_nodes).copy()
    component_id: dict[str, int] = {node: 0 for node in eligible_ids}
    if connected_nodes:
        components = sorted(
            (tuple(sorted(members)) for members in nx.connected_components(subgraph)),
            key=lambda members: (-len(members), members[0]),
        )
        for index, members in enumerate(components, start=1):
            component_id.update({node: index for node in members})
    explained = float((variance[0] + variance[1]) / total_variance)
    rows = [
        {
            "artist_label_id": node,
            "x": decimal(coordinate_by_id[node][0]),
            "y": decimal(coordinate_by_id[node][1]),
            "component_id": str(component_id[node]),
            "stable_graph_degree": str(int(graph.degree(node))),
            "projection_population": str(len(eligible_ids)),
            "projection_variance_explained_2d": decimal(explained),
            "layout_note": "deterministic 2D PCA of consensus primary-plus-sensitivity lyric-repertoire centroids across all eligible labels; near positions approximate semantic proximity, while a retained edge is the stricter stable-neighbour result",
        }
        for node in sorted(eligible_ids)
    ]
    return rows, {
        "method": "consensus_semantic_pca",
        "projection_population": len(eligible_ids),
        "variance_explained_2d": explained,
        "connected_node_count": len(connected_nodes),
    }


def public_documents() -> dict[str, str]:
    """Return the complete, deterministic public text payload for validation."""
    readme = """# Chinese Rap Lyrical Repertoire Graph v2

## The single question

Which **artist-labelled corpus slices** have similar duplicate-controlled lyric
repertoires, and which neighbours remain after removing exact cleaned text
shared across source labels?

An edge is deliberately narrow: the two labels are mutual top-five semantic
neighbours in both the primary duplicate-weighted chunk representation and the
shared-text-exclusion sensitivity representation. It is a relationship between
two corpus slices, not a real-world relationship between rappers.

The current artist strings are source labels. They remain labelled as not
externally identity-verified until an evidence-backed identity registry exists.

Only songs whose canonical `artist_title_comparison_eligible` field is `true`
enter this source-label comparison. This is a conservative metadata-quality
filter, not a correction or renaming of source labels.
"""
    protocol = """# Research protocol

## Representation

Each active clean lyric chunk has a validated BGE-M3 embedding. For every
source artist label, the primary centroid is the L2-normalized weighted mean of
chunk embeddings. Within the metadata-eligible comparison population, every
exact clean-text hash is assigned total mass one across its retained copies.
The weights are recomputed after the conservative song filter, so a surviving
copy is not penalized for a duplicate copy that was excluded from comparison.

## Mandatory sensitivity

An exact clean-text hash used under more than one source artist label is removed
entirely for the sensitivity centroid. A graph edge is displayed only when it
is mutual top-five in both representations. This removes apparent closeness
that can be driven solely by shared text across labels.

## Support, graph, and spatial-projection rule

A source label needs at least 5 clean songs and 20 effective clean-text mass
in **both** representations. Pairwise cosine is ranked within the eligible
corpus labels. No absolute cosine threshold is used. Edge membership is defined
only by the mutual-neighbour rule above.

For the public map, every graph-eligible label receives a deterministic 2D PCA
coordinate from its normalized consensus (primary plus sensitivity) centroid.
Near positions are an approximate visual summary of textual semantic proximity;
the two-dimensional map necessarily discards information. A line remains the
stricter result: it is drawn only for a retained mutual-top-five edge.

## Non-claims

Do not read an edge as collaboration, friendship, influence, affiliation,
location, genre, Flow, vocal technique, beat choice, or personal preference.
The data describe only this frozen, unevenly collected corpus slice.
"""
    dictionary = """# Data dictionary

- `primary_effective_text_mass`: sum of comparison-population exact-clean-text
  duplicate-control weights under the source label. The weights are recomputed
  after the metadata-eligibility filter so each retained exact-text group has
  total mass one.
- `shared_text_dropped_mass_share`: primary mass removed when exact cleaned text
  appearing under more than one source label is excluded.
- `primary_rank_*` / `sensitivity_rank_*`: reciprocal nearest-neighbour ranks
  among the eligible labels, not a social rank or popularity rank.
- `*_pair_percentile`: empirical percentile among all eligible label-pair
  cosine values for that representation.
- `stable_across_shared_text_exclusion`: retained only when both mutual-top-five
  tests pass.
- `x` / `y`: deterministic PCA coordinates for all graph-eligible labels;
  distance is an approximate semantic-proximity display, not an edge test.
- `projection_variance_explained_2d`: share of consensus-vector variation
  represented by the two displayed PCA axes.
- `component_id`: stable-edge connected component only. `0` means that an
  otherwise eligible source label has no retained edge; it does not mean that
  the label has no semantic neighbours.

Public files contain only aggregate/source-label graph data. Song IDs, chunk
IDs, lyric text, embeddings, and membership records stay private local-only.
"""
    return {
        "README.md": readme,
        "research_protocol.md": protocol,
        "data_dictionary.md": dictionary,
    }


def payload_hashes(directory: Path, exclude: set[str]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in exclude
    }


def compute() -> dict[str, Any]:
    frame, vectors, input_info = load_aligned_input()
    primary_vectors, primary_support, primary_frame = aggregate_artist_vectors(frame, vectors, exclude_shared=False)
    sensitivity_vectors, sensitivity_support, sensitivity_frame = aggregate_artist_vectors(frame, vectors, exclude_shared=True)
    support = all_label_support(frame, primary_support, sensitivity_support)
    label_order = sorted(label for label, values in support.items() if values["graph_eligible"])
    if len(label_order) < TOP_K + 1:
        raise RepertoireError("Too few supported source labels exist for a mutual-neighbour graph.")
    if any(label not in primary_vectors or label not in sensitivity_vectors for label in label_order):
        raise RepertoireError("A graph-eligible label is missing a primary or sensitivity centroid.")
    primary_scores = cosine_matrix(label_order, primary_vectors)
    sensitivity_scores = cosine_matrix(label_order, sensitivity_vectors)
    primary_ranks, primary_percentile, primary_audit = neighbor_tables(label_order, primary_scores, "primary_duplicate_weighted_chunk_centroid")
    sensitivity_ranks, sensitivity_percentile, sensitivity_audit = neighbor_tables(label_order, sensitivity_scores, "shared_text_exclusion_sensitivity_centroid")
    edges, primary_mutual, sensitivity_mutual = stable_edges(
        label_order, primary_scores, sensitivity_scores, primary_ranks, sensitivity_ranks, primary_percentile, sensitivity_percentile
    )
    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge["artist_label_id_a"]] += 1
        degree[edge["artist_label_id_b"]] += 1
    registry_rows: list[dict[str, str]] = []
    node_rows: list[dict[str, str]] = []
    vector_rowmap: list[dict[str, str]] = []
    vector_matrix: list[np.ndarray] = []
    for label in sorted(support):
        label_id = artist_label_id(label)
        values = support[label]
        primary = values["primary"]
        sensitivity = values["sensitivity"]
        registry_rows.append(
            {
                "artist_label_id": label_id,
                "source_artist_label": label,
                "label_attribution_status": LABEL_STATUS,
                "external_identity_verified": "false",
                "graph_display_status": "eligible" if values["graph_eligible"] else "support_below_protocol_threshold",
            }
        )
        primary_mass = float(primary["effective_text_mass"])
        sensitivity_mass = float(sensitivity["effective_text_mass"])
        node_rows.append(
            {
                "artist_label_id": label_id,
                "source_artist_label": label,
                "label_attribution_status": LABEL_STATUS,
                "clean_song_count": str(primary["clean_song_count"]),
                "independent_clean_song_count": str(primary["independent_clean_song_count"]),
                "primary_raw_chunk_count": str(primary["raw_chunk_count"]),
                "primary_effective_text_mass": decimal(primary_mass),
                "primary_unique_clean_text_count": str(primary["unique_clean_text_count"]),
                "sensitivity_raw_chunk_count": str(sensitivity["raw_chunk_count"]),
                "sensitivity_effective_text_mass": decimal(sensitivity_mass),
                "sensitivity_unique_clean_text_count": str(sensitivity["unique_clean_text_count"]),
                "shared_text_dropped_effective_mass": decimal(max(0.0, primary_mass - sensitivity_mass)),
                "shared_text_dropped_mass_share": decimal((primary_mass - sensitivity_mass) / primary_mass if primary_mass else 0.0),
                "eligible_primary_support": "true" if values["primary_eligible"] else "false",
                "eligible_sensitivity_support": "true" if values["sensitivity_eligible"] else "false",
                "graph_node_eligible": "true" if values["graph_eligible"] else "false",
                "stable_graph_degree": str(degree.get(label_id, 0)),
            }
        )
        if values["graph_eligible"]:
            vector_rowmap.append(
                {
                    "vector_row_index": str(len(vector_matrix)),
                    "artist_label_id": label_id,
                    "source_artist_label": label,
                    "graph_node_eligible": "true",
                    "primary_effective_text_mass": decimal(primary_mass),
                    "sensitivity_effective_text_mass": decimal(sensitivity_mass),
                }
            )
            vector_matrix.append(np.concatenate([primary_vectors[label], sensitivity_vectors[label]]))
    memberships = frame[[
        "canonical_artist", "song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256", "analysis_text_weight", "comparison_text_weight", "clean_row_index"
    ]].copy()
    shared = shared_text_hashes(frame)
    memberships["shared_across_artist_labels"] = memberships["analysis_text_sha256"].isin(shared)
    membership_rows = [
        {
            "artist_label_id": artist_label_id(str(row.canonical_artist)),
            "source_artist_label": str(row.canonical_artist),
            "song_id": str(row.song_id),
            "chunk_id": str(row.chunk_id),
            "canonical_lyric_text_sha256": str(row.canonical_lyric_text_sha256),
            "analysis_text_sha256": str(row.analysis_text_sha256),
            "frozen_analysis_text_weight": decimal(float(row.analysis_text_weight)),
            "comparison_text_weight": decimal(float(row.comparison_text_weight)),
            "clean_row_index": str(int(row.clean_row_index)),
            "shared_across_artist_labels": "true" if bool(row.shared_across_artist_labels) else "false",
            "included_in_primary_centroid": "true",
            "included_in_shared_text_exclusion_sensitivity": "false" if bool(row.shared_across_artist_labels) else "true",
        }
        for row in memberships.itertuples(index=False)
    ]
    vector_array = np.vstack(vector_matrix).astype(np.float32)
    layouts, layout_info = graph_layout(node_rows, edges, vector_rowmap, vector_array)
    shared_rows = frame.loc[frame["analysis_text_sha256"].isin(shared)]
    union = primary_mutual | sensitivity_mutual
    robustness_rows = [
        {"metric": "comparison_population_policy", "value": COMPARISON_POPULATION_POLICY, "interpretation": "source-label centroids exclude songs marked ineligible for artist/title comparison by the canonical metadata contract"},
        {"metric": "comparison_duplicate_weight_policy", "value": COMPARISON_DUPLICATE_WEIGHT_POLICY, "interpretation": "within the filtered comparison population, each exact clean-text hash has total duplicate-control mass one"},
        {"metric": "pre_filter_clean_lyric_chunks", "value": str(input_info["pre_filter_clean_chunks"]), "interpretation": "active clean-text chunks before the conservative comparison-eligibility filter"},
        {"metric": "comparison_ineligible_clean_chunks_excluded", "value": str(input_info["comparison_ineligible_clean_chunks_excluded"]), "interpretation": "active clean-text chunks excluded because their songs are not eligible for artist/title comparison"},
        {"metric": "pre_filter_clean_songs", "value": str(input_info["pre_filter_clean_songs"]), "interpretation": "songs with active clean text before the conservative comparison-eligibility filter"},
        {"metric": "pre_filter_source_artist_labels", "value": str(input_info["pre_filter_source_labels"]), "interpretation": "source artist labels before the conservative comparison-eligibility filter"},
        {"metric": "comparison_ineligible_songs_excluded", "value": str(input_info["comparison_ineligible_songs_excluded"]), "interpretation": "songs excluded because their canonical artist/title comparison eligibility is false"},
        {"metric": "comparison_unique_clean_text_groups", "value": str(input_info["comparison_unique_clean_text_groups"]), "interpretation": "exact clean-text groups after the comparison-eligibility filter"},
        {"metric": "comparison_effective_text_mass", "value": decimal(input_info["comparison_effective_text_mass"]), "interpretation": "sum of recomputed comparison-population duplicate-control weights; exactly one unit per retained exact clean-text group"},
        {"metric": "clean_lyric_chunks", "value": str(len(frame)), "interpretation": "active clean-text analytic population"},
        {"metric": "clean_songs", "value": str(input_info["clean_songs"]), "interpretation": "songs with at least one active clean lyric chunk"},
        {"metric": "source_artist_labels", "value": str(input_info["source_labels"]), "interpretation": "source labels; not externally verified identities"},
        {"metric": "graph_eligible_labels", "value": str(len(label_order)), "interpretation": f"at least {MIN_CLEAN_SONGS} clean songs and {MIN_EFFECTIVE_TEXT_MASS:g} effective-text mass in both primary and shared-text-exclusion representations"},
        {"metric": "shared_cross_label_clean_text_hashes", "value": str(len(shared)), "interpretation": "exact clean text used under more than one source artist label"},
        {"metric": "shared_cross_label_chunk_rows", "value": str(len(shared_rows)), "interpretation": "active clean chunks removed in shared-text-exclusion sensitivity"},
        {"metric": "shared_cross_label_effective_text_mass", "value": decimal(float(shared_rows["comparison_text_weight"].sum())), "interpretation": "comparison-population duplicate-controlled mass removed in sensitivity"},
        {"metric": "primary_mutual_top_k_edges", "value": str(len(primary_mutual)), "interpretation": f"mutual top-{TOP_K} primary neighbours before sensitivity gate"},
        {"metric": "sensitivity_mutual_top_k_edges", "value": str(len(sensitivity_mutual)), "interpretation": f"mutual top-{TOP_K} neighbours after shared-text exclusion"},
        {"metric": "stable_retained_edges", "value": str(len(edges)), "interpretation": "edges retained by both mutual-neighbour tests"},
        {"metric": "mutual_edge_jaccard_overlap", "value": decimal(len(primary_mutual & sensitivity_mutual) / len(union) if union else 0.0), "interpretation": "sensitivity agreement; not a confidence probability"},
        {"metric": "semantic_projection_population", "value": str(layout_info["projection_population"]), "interpretation": "all graph-eligible labels represented in the 2D semantic PCA display"},
        {"metric": "semantic_projection_variance_explained_2d", "value": decimal(layout_info["variance_explained_2d"]), "interpretation": "consensus-vector variation represented by the two displayed semantic PCA axes"},
        {"metric": "connected_stable_graph_nodes", "value": str(layout_info["connected_node_count"]), "interpretation": "eligible labels with at least one retained lyrical-repertoire edge"},
    ]
    return {
        "input_info": input_info,
        "registry_rows": registry_rows,
        "node_rows": node_rows,
        "edge_rows": edges,
        "layout_rows": layouts,
        "robustness_rows": robustness_rows,
        "membership_rows": membership_rows,
        "vector_matrix": vector_array,
        "vector_rowmap": vector_rowmap,
        "layout_info": layout_info,
        "neighbor_audit": sorted(primary_audit + sensitivity_audit, key=lambda row: (row["representation"], row["artist_label_id"], int(row["neighbor_rank"]))),
        "counts": {
            "pre_filter_clean_lyric_chunks": input_info["pre_filter_clean_chunks"],
            "comparison_ineligible_clean_chunks_excluded": input_info["comparison_ineligible_clean_chunks_excluded"],
            "pre_filter_clean_songs": input_info["pre_filter_clean_songs"],
            "comparison_ineligible_songs_excluded": input_info["comparison_ineligible_songs_excluded"],
            "pre_filter_source_artist_labels": input_info["pre_filter_source_labels"],
            "clean_lyric_chunks": len(frame),
            "clean_songs": input_info["clean_songs"],
            "source_artist_labels": input_info["source_labels"],
            "graph_eligible_labels": len(label_order),
            "stable_retained_edges": len(edges),
            "connected_stable_graph_nodes": layout_info["connected_node_count"],
        },
    }


def build_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "model": {
            "embedding": "BGE-M3 clean-text embeddings; no audio, beat, voice, or Flow input",
            "representation": "comparison-population exact-clean-text duplicate-controlled lyric chunk centroids by source artist label",
            "sensitivity": "remove exact clean text that occurs under more than one source artist label",
            "comparison_population_policy": COMPARISON_POPULATION_POLICY,
            "comparison_duplicate_weight_policy": COMPARISON_DUPLICATE_WEIGHT_POLICY,
            "edge_rule": EDGE_RULE,
            "spatial_projection": "deterministic 2D PCA of normalized consensus primary-plus-sensitivity lyric-repertoire centroids across all graph-eligible labels; distance is an approximate semantic display and does not define edges",
            "spatial_projection_method": result["layout_info"]["method"],
            "spatial_projection_population": result["layout_info"]["projection_population"],
            "spatial_projection_variance_explained_2d": result["layout_info"]["variance_explained_2d"],
            "top_k": TOP_K,
            "minimum_clean_songs": MIN_CLEAN_SONGS,
            "minimum_effective_text_mass": MIN_EFFECTIVE_TEXT_MASS,
        },
        "counts": result["counts"],
        "claim_boundary": "lyrical-repertoire proximity between source-labelled corpus slices only; not a real-world rapper relationship, identity, affiliation, style, genre, or audio-performance claim",
    }


def validate_persisted(permit_missing_validation: bool = False, require_prior_validation: bool = True) -> dict[str, Any]:
    require_exact_allowlist(OUTPUT_DIR, PUBLIC_ALLOWLIST, "public repertoire-graph output", permit_missing_validation)
    require_exact_allowlist(PRIVATE_DIR, PRIVATE_ALLOWLIST, "private repertoire-graph output", permit_missing_validation)
    result = compute()
    expected_summary = build_summary(result)
    actual_registry = read_csv_exact(OUTPUT_DIR / "artist_label_registry.csv", REGISTRY_COLUMNS, "artist label registry")
    actual_summary = read_json(OUTPUT_DIR / "analysis_summary.json", "analysis summary")
    actual_nodes = read_csv_exact(OUTPUT_DIR / "artist_repertoire_nodes.csv", NODE_COLUMNS, "artist repertoire nodes")
    actual_edges = read_csv_exact(OUTPUT_DIR / "artist_repertoire_edges.csv", EDGE_COLUMNS, "artist repertoire edges")
    actual_layout = read_csv_exact(OUTPUT_DIR / "artist_repertoire_layout.csv", LAYOUT_COLUMNS, "artist repertoire layout")
    actual_robustness = read_csv_exact(OUTPUT_DIR / "robustness_summary.csv", ROBUSTNESS_COLUMNS, "robustness summary")
    actual_documents = {
        name: (OUTPUT_DIR / name).read_text(encoding="utf-8")
        for name in ("README.md", "research_protocol.md", "data_dictionary.md")
    }
    actual_membership = read_csv_exact(PRIVATE_DIR / "artist_chunk_membership_v2.csv", MEMBERSHIP_COLUMNS, "private artist chunk membership")
    actual_vector_rowmap = read_csv_exact(PRIVATE_DIR / "artist_repertoire_vector_rowmap_v2.csv", VECTOR_ROWMAP_COLUMNS, "private artist vector row map")
    actual_audit = read_csv_exact(PRIVATE_DIR / "neighbor_rank_audit_v2.csv", NEIGHBOR_AUDIT_COLUMNS, "private neighbour audit")
    vector_path = PRIVATE_DIR / "artist_repertoire_vectors_v2.npy"
    actual_vectors = np.load(vector_path, allow_pickle=False)
    manifest = read_json(OUTPUT_DIR / "manifest.json", "repertoire graph manifest")
    private_manifest = read_json(PRIVATE_DIR / "private_manifest.json", "private repertoire graph manifest")
    expected_hashes = result["input_info"]["hashes"]
    expected_config = {
        "top_k": TOP_K,
        "minimum_clean_songs": MIN_CLEAN_SONGS,
        "minimum_effective_text_mass": MIN_EFFECTIVE_TEXT_MASS,
        "edge_rule": EDGE_RULE,
        "spatial_projection_method": "consensus_semantic_pca",
        "spatial_projection_population": result["layout_info"]["projection_population"],
        "spatial_projection_variance_explained_2d": result["layout_info"]["variance_explained_2d"],
        "label_attribution_status": LABEL_STATUS,
        "comparison_population_policy": COMPARISON_POPULATION_POLICY,
        "comparison_duplicate_weight_policy": COMPARISON_DUPLICATE_WEIGHT_POLICY,
    }
    manifest_ok = (
        manifest.get("artifact_id") == ARTIFACT_ID
        and manifest.get("version") == VERSION
        and manifest.get("input_hashes") == expected_hashes
        and manifest.get("configuration") == expected_config
        and manifest.get("counts") == result["counts"]
        and manifest.get("claim_boundary") == expected_summary["claim_boundary"]
        and private_manifest.get("artifact_id") == ARTIFACT_ID
        and private_manifest.get("version") == VERSION
        and private_manifest.get("classification") == "private_local_only_membership_and_embedding_audit_without_lyric_text"
    )
    prior_ok = True
    if require_prior_validation:
        previous_public = read_json(OUTPUT_DIR / "validation.json", "prior public repertoire validation")
        previous_private = read_json(PRIVATE_DIR / "private_validation.json", "prior private repertoire validation")
        prior_ok = previous_public.get("artifact_id") == ARTIFACT_ID and previous_public.get("version") == VERSION and previous_public.get("status") == "pass" and previous_private == previous_public
    checks = [
        {"name": "public_inventory_exact", "passed": True},
        {"name": "private_inventory_exact", "passed": True},
        {"name": "manifest_lineage_configuration_and_counts_current", "passed": manifest_ok},
        {"name": "analysis_summary_recomputes_exactly", "passed": actual_summary == expected_summary},
        {"name": "public_documents_recompute_exactly", "passed": actual_documents == public_documents()},
        {"name": "registry_recomputes_exactly", "passed": actual_registry == result["registry_rows"]},
        {"name": "nodes_recompute_exactly", "passed": actual_nodes == result["node_rows"]},
        {"name": "stable_edges_recompute_exactly", "passed": actual_edges == result["edge_rows"]},
        {"name": "display_layout_recomputes_exactly", "passed": actual_layout == result["layout_rows"]},
        {"name": "robustness_summary_recomputes_exactly", "passed": actual_robustness == result["robustness_rows"]},
        {"name": "private_membership_recomputes_exactly", "passed": actual_membership == result["membership_rows"]},
        {"name": "private_neighbour_audit_recomputes_exactly", "passed": actual_audit == result["neighbor_audit"]},
        {"name": "private_vectors_recompute_exactly", "passed": actual_vectors.shape == result["vector_matrix"].shape and np.allclose(actual_vectors, result["vector_matrix"], rtol=0.0, atol=1e-7)},
        {"name": "private_vector_rowmap_recomputes_exactly", "passed": actual_vector_rowmap == result["vector_rowmap"]},
        {"name": "public_payload_hashes_match", "passed": manifest.get("output_files") == payload_hashes(OUTPUT_DIR, {"manifest.json", "validation.json"})},
        {"name": "private_payload_hashes_match", "passed": private_manifest.get("files") == payload_hashes(PRIVATE_DIR, {"private_manifest.json", "private_validation.json"})},
        {"name": "public_outputs_contain_no_song_chunk_or_lyric_columns", "passed": all("song_id" not in row and "chunk_id" not in row and "text" not in row for row in actual_nodes + actual_edges + actual_layout + actual_robustness)},
        {"name": "comparison_population_filter_applied", "passed": result["input_info"]["clean_songs"] + result["input_info"]["comparison_ineligible_songs_excluded"] == result["input_info"]["pre_filter_clean_songs"] and result["input_info"]["clean_chunks"] + result["input_info"]["comparison_ineligible_clean_chunks_excluded"] == result["input_info"]["pre_filter_clean_chunks"]},
        {"name": "comparison_duplicate_weights_sum_to_one_per_retained_exact_clean_text", "passed": math.isclose(float(result["input_info"]["comparison_effective_text_mass"]), float(result["input_info"]["comparison_unique_clean_text_groups"]), rel_tol=0.0, abs_tol=1e-8)},
        {"name": "all_retained_edges_pass_the_two_representation_rule", "passed": all(row["stable_across_shared_text_exclusion"] == "true" and row["edge_definition"] == EDGE_RULE and int(row["primary_rank_a_to_b"]) <= TOP_K and int(row["primary_rank_b_to_a"]) <= TOP_K and int(row["sensitivity_rank_a_to_b"]) <= TOP_K and int(row["sensitivity_rank_b_to_a"]) <= TOP_K for row in actual_edges)},
        {"name": "prior_validation_current_and_passing", "passed": prior_ok},
    ]
    passed = all(bool(check["passed"]) for check in checks)
    return {"artifact_id": ARTIFACT_ID, "version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pass" if passed else "fail", "checks": checks}


def build() -> dict[str, Any]:
    require_exact_allowlist(OUTPUT_DIR, PUBLIC_ALLOWLIST, "public repertoire-graph output", permit_missing_validation=True)
    require_exact_allowlist(PRIVATE_DIR, PRIVATE_ALLOWLIST, "private repertoire-graph output", permit_missing_validation=True)
    result = compute()
    summary = build_summary(result)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in public_documents().items():
        atomic_write_text(OUTPUT_DIR / name, content)
    atomic_write_json(OUTPUT_DIR / "analysis_summary.json", summary)
    atomic_write_csv(OUTPUT_DIR / "artist_label_registry.csv", REGISTRY_COLUMNS, result["registry_rows"])
    atomic_write_csv(OUTPUT_DIR / "artist_repertoire_nodes.csv", NODE_COLUMNS, result["node_rows"])
    atomic_write_csv(OUTPUT_DIR / "artist_repertoire_edges.csv", EDGE_COLUMNS, result["edge_rows"])
    atomic_write_csv(OUTPUT_DIR / "artist_repertoire_layout.csv", LAYOUT_COLUMNS, result["layout_rows"])
    atomic_write_csv(OUTPUT_DIR / "robustness_summary.csv", ROBUSTNESS_COLUMNS, result["robustness_rows"])
    atomic_write_csv(PRIVATE_DIR / "artist_chunk_membership_v2.csv", MEMBERSHIP_COLUMNS, result["membership_rows"])
    atomic_write_csv(PRIVATE_DIR / "artist_repertoire_vector_rowmap_v2.csv", VECTOR_ROWMAP_COLUMNS, result["vector_rowmap"])
    atomic_write_csv(PRIVATE_DIR / "neighbor_rank_audit_v2.csv", NEIGHBOR_AUDIT_COLUMNS, result["neighbor_audit"])
    vectors_path = PRIVATE_DIR / "artist_repertoire_vectors_v2.npy"
    with tempfile.NamedTemporaryFile("wb", dir=PRIVATE_DIR, delete=False) as handle:
        np.save(handle, result["vector_matrix"], allow_pickle=False)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, vectors_path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "input_hashes": result["input_info"]["hashes"],
        "configuration": {
            "top_k": TOP_K,
            "minimum_clean_songs": MIN_CLEAN_SONGS,
            "minimum_effective_text_mass": MIN_EFFECTIVE_TEXT_MASS,
            "edge_rule": EDGE_RULE,
            "spatial_projection_method": result["layout_info"]["method"],
            "spatial_projection_population": result["layout_info"]["projection_population"],
            "spatial_projection_variance_explained_2d": result["layout_info"]["variance_explained_2d"],
            "label_attribution_status": LABEL_STATUS,
            "comparison_population_policy": COMPARISON_POPULATION_POLICY,
            "comparison_duplicate_weight_policy": COMPARISON_DUPLICATE_WEIGHT_POLICY,
        },
        "counts": result["counts"],
        "claim_boundary": summary["claim_boundary"],
        "privacy": "public files contain source-label graph summaries only; song/chunk identifiers, membership, and embeddings remain private local-only",
        "output_files": payload_hashes(OUTPUT_DIR, {"manifest.json", "validation.json"}),
    }
    private_manifest = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "classification": "private_local_only_membership_and_embedding_audit_without_lyric_text",
        "files": payload_hashes(PRIVATE_DIR, {"private_manifest.json", "private_validation.json"}),
    }
    atomic_write_json(OUTPUT_DIR / "manifest.json", manifest)
    atomic_write_json(PRIVATE_DIR / "private_manifest.json", private_manifest)
    validation = validate_persisted(permit_missing_validation=True, require_prior_validation=False)
    atomic_write_json(OUTPUT_DIR / "validation.json", validation)
    atomic_write_json(PRIVATE_DIR / "private_validation.json", validation)
    return validation


def main() -> int:
    try:
        validation = build()
    except RepertoireError as exc:
        print(f"BUILD FAILED: {exc}")
        return 2
    print(f"Built {ARTIFACT_ID}: {validation['status']} ({sum(bool(check['passed']) for check in validation['checks'])}/{len(validation['checks'])} checks).")
    return 0 if validation["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
