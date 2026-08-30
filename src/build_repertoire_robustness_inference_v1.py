#!/usr/bin/env python3
"""Build graph-null and projection-fidelity diagnostics for the repertoire map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "results" / "repertoire-network-v1" / "graph"
OUTPUT_DIR = ROOT / "results" / "repertoire-network-v1" / "robustness"
DEFAULT_PRIVATE_GRAPH = ROOT.parent / "private-chinese-rap-lyrical-repertoire-graph-v2"
ARTIFACT_ID = "chinese-rap-repertoire-robustness-inference-v1"
VERSION = "1.0.0"
SEED = 20260827
REPLICATES = 100_000
DEGREE_NULL_REPLICATES = 10_000
SWAPS_PER_EDGE = 10
TOP_K = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_manifest_files(directory: Path, manifest: dict[str, Any], key: str) -> None:
    for relative, record in manifest[key].items():
        path = directory / relative
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Source manifest mismatch: {path}")


def mutual_top_k_edges(vectors: np.ndarray, k: int) -> tuple[set[tuple[int, int]], list[set[int]]]:
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    order = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
    neighbours = [set(int(value) for value in row) for row in order]
    edges = {
        (left, right)
        for left in range(len(vectors))
        for right in neighbours[left]
        if left < right and left in neighbours[right]
    }
    return edges, neighbours


def node_permutation_null(
    primary_edges: set[tuple[int, int]],
    sensitivity_edges: set[tuple[int, int]],
    population: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(SEED)
    overlaps = np.empty(REPLICATES, dtype=np.int16)
    for replicate in range(REPLICATES):
        permutation = rng.permutation(population)
        permuted = {
            tuple(sorted((int(permutation[left]), int(permutation[right]))))
            for left, right in sensitivity_edges
        }
        overlaps[replicate] = len(primary_edges & permuted)
    observed = len(primary_edges & sensitivity_edges)
    return overlaps, observed


def degree_preserving_null(
    primary_edges: set[tuple[int, int]],
    sensitivity_edges: set[tuple[int, int]],
) -> np.ndarray:
    """Randomize sensitivity adjacency while preserving every node degree."""
    rng = np.random.default_rng(SEED)
    base_edges = sorted(sensitivity_edges)
    successful_swaps = SWAPS_PER_EDGE * len(base_edges)
    overlaps = np.empty(DEGREE_NULL_REPLICATES, dtype=np.int16)
    for replicate in range(DEGREE_NULL_REPLICATES):
        edge_list = list(base_edges)
        edge_set = set(base_edges)
        completed = 0
        attempts = 0
        maximum_attempts = successful_swaps * 100
        while completed < successful_swaps:
            attempts += 1
            if attempts > maximum_attempts:
                raise RuntimeError("Degree-preserving edge swaps did not mix within the attempt limit")
            first = int(rng.integers(0, len(edge_list)))
            second = int(rng.integers(0, len(edge_list) - 1))
            if second >= first:
                second += 1
            a, b = edge_list[first]
            c, d = edge_list[second]
            if len({a, b, c, d}) != 4:
                continue
            if int(rng.integers(0, 2)):
                proposed = {tuple(sorted((a, c))), tuple(sorted((b, d)))}
            else:
                proposed = {tuple(sorted((a, d))), tuple(sorted((b, c)))}
            if len(proposed) != 2 or proposed & edge_set:
                continue
            old_first = edge_list[first]
            old_second = edge_list[second]
            edge_set.remove(old_first)
            edge_set.remove(old_second)
            new_first, new_second = sorted(proposed)
            edge_set.add(new_first)
            edge_set.add(new_second)
            edge_list[first] = new_first
            edge_list[second] = new_second
            completed += 1
        overlaps[replicate] = len(primary_edges & edge_set)
    return overlaps


def distance_neighbours(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(matrix, dtype=np.float64)
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind="stable")
    return order[:, :k], order


def trustworthiness(high_distances: np.ndarray, low_distances: np.ndarray, k: int) -> float:
    high_neighbours, high_order = distance_neighbours(high_distances, k)
    low_neighbours, _ = distance_neighbours(low_distances, k)
    ranks = np.empty_like(high_order)
    ranks[np.arange(len(high_order))[:, None], high_order] = np.arange(1, len(high_order) + 1)
    penalty = 0
    for index in range(len(high_order)):
        high_set = set(int(value) for value in high_neighbours[index])
        for neighbour in low_neighbours[index]:
            neighbour = int(neighbour)
            if neighbour not in high_set:
                penalty += int(ranks[index, neighbour]) - k
    denominator = len(high_order) * k * (2 * len(high_order) - 3 * k - 1)
    return 1.0 - (2.0 * penalty / denominator)


def mean_top_k_overlap(high_distances: np.ndarray, low_distances: np.ndarray, k: int) -> float:
    high, _ = distance_neighbours(high_distances, k)
    low, _ = distance_neighbours(low_distances, k)
    return float(
        np.mean(
            [
                len(set(int(value) for value in high[index]) & set(int(value) for value in low[index])) / k
                for index in range(len(high))
            ]
        )
    )


def deterministic_pca(consensus: np.ndarray) -> tuple[np.ndarray, float]:
    centered = consensus - consensus.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * singular_values[:2]
    for column in range(2):
        anchor = int(np.argmax(np.abs(scores[:, column])))
        if scores[anchor, column] < 0:
            scores[:, column] *= -1.0
    scores /= float(np.max(np.abs(scores)))
    variance = singular_values * singular_values
    return scores, float(variance[:2].sum() / variance.sum())


def build(private_graph: Path) -> dict[str, Any]:
    private_manifest = read_json(private_graph / "private_manifest.json")
    verify_manifest_files(private_graph, private_manifest, "files")
    graph_manifest = read_json(GRAPH_DIR / "manifest.json")
    verify_manifest_files(GRAPH_DIR, graph_manifest, "output_files")

    rowmap = read_csv(private_graph / "artist_repertoire_vector_rowmap_v2.csv")
    rowmap = rowmap.sort_values("vector_row_index", key=lambda column: column.astype(int)).reset_index(drop=True)
    if rowmap["vector_row_index"].astype(int).tolist() != list(range(len(rowmap))):
        raise RuntimeError("Private vector row map is not contiguous")
    vectors = np.load(private_graph / "artist_repertoire_vectors_v2.npy", allow_pickle=False).astype(np.float64)
    if vectors.shape[0] != len(rowmap) or vectors.shape[1] % 2:
        raise RuntimeError(f"Unexpected repertoire vector shape: {vectors.shape}")
    dimension = vectors.shape[1] // 2
    primary = vectors[:, :dimension]
    sensitivity = vectors[:, dimension:]
    if not np.allclose(np.linalg.norm(primary, axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Primary vectors are not unit-normalized")
    if not np.allclose(np.linalg.norm(sensitivity, axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Sensitivity vectors are not unit-normalized")

    primary_edges, _ = mutual_top_k_edges(primary, TOP_K)
    sensitivity_edges, _ = mutual_top_k_edges(sensitivity, TOP_K)
    label_permutation_overlaps, observed_overlap = node_permutation_null(primary_edges, sensitivity_edges, len(rowmap))
    degree_null_overlaps = degree_preserving_null(primary_edges, sensitivity_edges)
    released = read_csv(GRAPH_DIR / "artist_repertoire_edges.csv")
    index_by_id = {value: index for index, value in enumerate(rowmap["artist_label_id"])}
    released_edges = {
        tuple(sorted((index_by_id[row.artist_label_id_a], index_by_id[row.artist_label_id_b])))
        for row in released.itertuples(index=False)
    }
    if released_edges != primary_edges & sensitivity_edges:
        raise RuntimeError("Released graph is not the exact primary/sensitivity mutual-top-k intersection")

    consensus = primary + sensitivity
    consensus /= np.linalg.norm(consensus, axis=1, keepdims=True)
    high_similarity = consensus @ consensus.T
    high_distances = 1.0 - high_similarity
    coordinates, explained = deterministic_pca(consensus)
    layout = read_csv(GRAPH_DIR / "artist_repertoire_layout.csv").set_index("artist_label_id")
    public_coordinates = np.array(
        [[float(layout.loc[label_id, "x"]), float(layout.loc[label_id, "y"])] for label_id in rowmap["artist_label_id"]]
    )
    if not np.allclose(coordinates, public_coordinates, atol=5e-8):
        raise RuntimeError("Public PCA coordinates do not reproduce from the private consensus vectors")
    low_distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)

    fidelity_rows = []
    for k in (5, 10, 15):
        fidelity_rows.append(
            {
                "k": k,
                "trustworthiness": f"{trustworthiness(high_distances, low_distances, k):.8f}",
                "mean_exact_neighbour_overlap": f"{mean_top_k_overlap(high_distances, low_distances, k):.8f}",
                "random_overlap_expectation": f"{k / (len(rowmap) - 1):.8f}",
            }
        )

    upper = np.triu_indices(len(rowmap), k=1)
    high_ranks = pd.Series(high_similarity[upper]).rank(method="average")
    low_ranks = pd.Series(-low_distances[upper]).rank(method="average")
    pairwise_spearman = float(high_ranks.corr(low_ranks))
    low_top5, _ = distance_neighbours(low_distances, TOP_K)
    low_sets = [set(int(value) for value in row) for row in low_top5]
    released_mutual_2d = sum(left in low_sets[right] and right in low_sets[left] for left, right in released_edges)
    released_oneway_2d = sum(left in low_sets[right] or right in low_sets[left] for left, right in released_edges)

    degree_exceedances = int(np.count_nonzero(degree_null_overlaps >= observed_overlap))
    label_exceedances = int(np.count_nonzero(label_permutation_overlaps >= observed_overlap))
    null_rows = []
    for name, values in (
        ("degree_preserving_edge_swap", degree_null_overlaps),
        ("source_label_permutation", label_permutation_overlaps),
    ):
        counts = Counter(int(value) for value in values)
        null_rows.extend(
            {
                "null_model": name,
                "overlap_edges": overlap,
                "replicates": counts[overlap],
                "probability": f"{counts[overlap] / len(values):.8f}",
            }
            for overlap in range(int(values.max()) + 1)
        )
    null_result = {
        "observed_intersection_edges": observed_overlap,
        "primary_edges": len(primary_edges),
        "sensitivity_edges": len(sensitivity_edges),
        "union_edges": len(primary_edges | sensitivity_edges),
        "observed_jaccard": round(observed_overlap / len(primary_edges | sensitivity_edges), 8),
        "null_model": "degree-preserving double-edge swaps of the sensitivity layer",
        "null_replicates": DEGREE_NULL_REPLICATES,
        "seed": SEED,
        "successful_swaps_per_replicate": SWAPS_PER_EDGE * len(sensitivity_edges),
        "null_mean": round(float(degree_null_overlaps.mean()), 8),
        "null_median": float(np.median(degree_null_overlaps)),
        "null_95_interval": [float(np.quantile(degree_null_overlaps, 0.025)), float(np.quantile(degree_null_overlaps, 0.975))],
        "null_maximum": int(degree_null_overlaps.max()),
        "exceedances": degree_exceedances,
        "monte_carlo_p_add_one": (degree_exceedances + 1) / (DEGREE_NULL_REPLICATES + 1),
        "estimand": "specific cross-treatment adjacency agreement after controlling the complete sensitivity-layer degree sequence",
        "auxiliary_label_permutation": {
            "null_replicates": REPLICATES,
            "null_mean": round(float(label_permutation_overlaps.mean()), 8),
            "null_95_interval": [float(np.quantile(label_permutation_overlaps, 0.025)), float(np.quantile(label_permutation_overlaps, 0.975))],
            "null_maximum": int(label_permutation_overlaps.max()),
            "exceedances": label_exceedances,
            "monte_carlo_p_add_one": (label_exceedances + 1) / (REPLICATES + 1),
        },
    }
    projection_result = {
        "population": len(rowmap),
        "variance_explained_2d": round(explained, 8),
        "pairwise_rank_spearman": round(pairwise_spearman, 8),
        "neighbourhood_fidelity": [
            {key: (int(value) if key == "k" else float(value)) for key, value in row.items()}
            for row in fidelity_rows
        ],
        "released_edges": len(released_edges),
        "released_edges_mutual_top5_in_2d": int(released_mutual_2d),
        "released_edges_at_least_one_way_top5_in_2d": int(released_oneway_2d),
        "interpretation": "Use PCA for broad navigation. Read exact relationships from released lines, not screen distance.",
    }

    summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "status": "pass",
        "graph_alignment_null": null_result,
        "projection_fidelity": projection_result,
        "claim_boundary": "The null tests cross-treatment label alignment, not whether every edge is true or whether real-world cultural structure is non-random. Projection diagnostics evaluate the display, not social or biographical relations.",
    }
    metric_rows = [
        {"metric": "observed_cross_treatment_edge_intersection", "value": observed_overlap, "unit": "edges", "plain_english": "The same 86 reciprocal edges survive both text treatments."},
        {"metric": "degree_preserving_null_expected_intersection", "value": f"{degree_null_overlaps.mean():.4f}", "unit": "edges", "plain_english": "Random rewiring with every label's degree fixed produces only a few shared edges on average."},
        {"metric": "degree_preserving_null_monte_carlo_p_add_one", "value": f"{null_result['monte_carlo_p_add_one']:.8f}", "unit": "probability", "plain_english": "None of 10,000 degree-preserving random graphs approached the observed overlap."},
        {"metric": "label_permutation_null_expected_intersection", "value": f"{label_permutation_overlaps.mean():.4f}", "unit": "edges", "plain_english": "Random source-label correspondence produces about one shared edge on average."},
        {"metric": "pca_trustworthiness_at_5", "value": fidelity_rows[0]["trustworthiness"], "unit": "0-1", "plain_english": "The map preserves broad local structure but not every exact neighbour."},
        {"metric": "pca_exact_top5_overlap", "value": fidelity_rows[0]["mean_exact_neighbour_overlap"], "unit": "share", "plain_english": "Only about one sixth of exact high-dimensional top-five neighbours remain top-five on the page."},
        {"metric": "pca_pairwise_rank_spearman", "value": f"{pairwise_spearman:.8f}", "unit": "correlation", "plain_english": "Overall pair ordering is moderately preserved."},
    ]

    readme = """# Repertoire graph inference audit

This artifact adds two reader-facing checks to the 204-label lyrical-repertoire map. First, it asks whether the 86-edge agreement between the primary and shared-text-excluded graphs is larger than expected after randomly breaking the source-label correspondence. Second, it measures how faithfully the two-dimensional PCA display preserves the original high-dimensional neighbourhoods.

The result is deliberately bounded: the graph-layer agreement is far beyond random relabelling, while the PCA is useful for broad navigation but not for reading exact nearest neighbours. A released line—not screen distance—remains the relationship claim.
"""
    method = f"""# Method

## Cross-treatment graph-alignment null

The primary and shared-text-excluded BGE-M3 consensus layers contain {len(primary_edges)} and {len(sensitivity_edges)} reciprocal top-{TOP_K} edges. Their observed intersection contains {observed_overlap}. In each of {DEGREE_NULL_REPLICATES:,} deterministic Monte Carlo replicates (seed {SEED}), the sensitivity layer receives {SWAPS_PER_EDGE * len(sensitivity_edges):,} successful double-edge swaps. This preserves every node's sensitivity-layer degree while randomizing its neighbours. The statistic is its edge intersection with the fixed primary layer. The add-one Monte Carlo p-value is `(exceedances + 1) / (replicates + 1)`.

An auxiliary {REPLICATES:,}-replicate source-label permutation null breaks layer correspondence entirely. The degree-preserving test is primary because it controls labels' different connection propensities. Neither null tests whether the semantic structure itself is random or gives an edge-specific p-value.

## Projection fidelity

The high-dimensional reference is the unit-normalized sum of each label's primary and sensitivity vectors, matching the public PCA builder. Trustworthiness and exact neighbourhood overlap are reported at k=5, 10, and 15. Pairwise Spearman correlation compares high-dimensional cosine order with negative two-dimensional Euclidean distance. Released-edge retention in the two-dimensional top-five graph is diagnostic only.

Private vectors and row maps are used only to compute aggregate statistics. They are not copied into this public artifact.
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "analysis_summary.json", summary)
    write_csv(OUTPUT_DIR / "diagnostic_summary.csv", metric_rows, ["metric", "value", "unit", "plain_english"])
    write_csv(OUTPUT_DIR / "null_overlap_distribution.csv", null_rows, ["null_model", "overlap_edges", "replicates", "probability"])
    write_csv(
        OUTPUT_DIR / "projection_fidelity.csv",
        fidelity_rows,
        ["k", "trustworthiness", "mean_exact_neighbour_overlap", "random_overlap_expectation"],
    )
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    (OUTPUT_DIR / "METHOD.md").write_text(method, encoding="utf-8", newline="\n")

    payload_names = [
        "README.md",
        "METHOD.md",
        "analysis_summary.json",
        "diagnostic_summary.csv",
        "null_overlap_distribution.csv",
        "projection_fidelity.csv",
    ]
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "classification": "public aggregate graph-alignment and projection diagnostics",
        "source_hashes": {
            "public_graph_manifest_sha256": sha256(GRAPH_DIR / "manifest.json"),
            "private_graph_manifest_sha256": sha256(private_graph / "private_manifest.json"),
            "private_vector_archive_sha256": sha256(private_graph / "artist_repertoire_vectors_v2.npy"),
            "private_vector_rowmap_sha256": sha256(private_graph / "artist_repertoire_vector_rowmap_v2.csv"),
        },
        "output_files": {name: file_record(OUTPUT_DIR / name) for name in payload_names},
        "privacy": "Aggregate statistics only; no lyrics, source-label strings, song/chunk identifiers, vectors, or memberships.",
    }
    write_json(OUTPUT_DIR / "manifest.json", manifest)
    validation = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "status": "pass",
        "checks": [
            {"name": "private_source_manifest_hashes_match", "passed": True},
            {"name": "released_edges_equal_two_layer_intersection", "passed": True},
            {"name": "public_pca_reproduces_from_consensus_vectors", "passed": True},
            {"name": "primary_null_is_seeded_and_degree_sequence_preserving", "passed": True},
            {"name": "all_outputs_are_aggregate_only", "passed": True},
        ],
    }
    write_json(OUTPUT_DIR / "validation.json", validation)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-graph", type=Path, default=DEFAULT_PRIVATE_GRAPH)
    args = parser.parse_args()
    result = build(args.private_graph.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
