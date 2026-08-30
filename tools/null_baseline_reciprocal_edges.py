"""How many reciprocal top-five edges would this graph have if labels were random?

The release states that 86 of a possible 20,706 label pairs are mutual top-five
neighbours under both text treatments, connecting 93 of 204 labels. Nothing states
how many such edges the rule produces when the label-to-song assignment carries no
information, so a reader cannot tell whether 86 is a lot.

This answers that by permutation. Songs are redistributed among labels while
preserving every label's song count and every song's label count, centroids are
rebuilt from the chunk embeddings, and the reciprocal top-five rule is reapplied
under both representations. Repeating it gives a null distribution for the edge
count and for the number of connected labels.

Before permuting anything the script rebuilds the observed graph from the same code
path and refuses to continue unless it reproduces the published 86 edges and 93
connected labels; otherwise the null would not be measuring the same rule.

Needs the private embedding and membership artefacts. Nothing it writes contains
lyric text.

    python tools/null_baseline_reciprocal_edges.py --replicates 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_EDGES = 86
PUBLISHED_CONNECTED = 93
TOP_K = 5


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def centroids(vectors: np.ndarray, groups: list[list[tuple[int, float]]]) -> np.ndarray:
    """Weighted mean of member chunk vectors per label, L2 normalised."""
    out = np.zeros((len(groups), vectors.shape[1]), dtype=np.float64)
    for index, rows in enumerate(groups):
        if not rows:
            continue
        idx = np.fromiter((i for i, _ in rows), dtype=np.int64, count=len(rows))
        weight = np.fromiter((w for _, w in rows), dtype=np.float64, count=len(rows))
        out[index] = (vectors[idx] * weight[:, None]).sum(axis=0) / weight.sum()
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


def mutual_top_k(matrix: np.ndarray, k: int) -> np.ndarray:
    """Boolean adjacency of pairs that are within each other's top k by cosine."""
    if k >= matrix.shape[0]:
        raise ValueError(f"top-{k} needs more than {k} nodes, got {matrix.shape[0]}")
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)
    top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    chosen = np.zeros(similarity.shape, dtype=bool)
    np.put_along_axis(chosen, top, True, axis=1)
    # A node is never its own neighbour. The -inf above only deprioritises the self
    # column; it still survives the partition once k approaches the node count.
    np.fill_diagonal(chosen, False)
    return chosen & chosen.T


def graph_stats(primary: np.ndarray, sensitivity: np.ndarray, k: int) -> tuple[int, int]:
    """Edges present under both representations, and how many labels they connect."""
    both = mutual_top_k(primary, k) & mutual_top_k(sensitivity, k)
    edges = int(np.triu(both, 1).sum())
    connected = int((both.any(axis=1)).sum())
    return edges, connected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path,
                        default=ROOT / "work/private-canonical-clean-text-embeddings-v1/canonical_clean_text_bge_m3_embeddings_v1.npy")
    parser.add_argument("--membership", type=Path,
                        default=ROOT / "work/private-chinese-rap-lyrical-repertoire-graph-v2/artist_chunk_membership_v2.csv")
    parser.add_argument("--node-rowmap", type=Path,
                        default=ROOT / "work/private-chinese-rap-lyrical-repertoire-graph-v2/artist_repertoire_vector_rowmap_v2.csv",
                        help="defines the eligible label universe; the membership file lists more")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/null-baseline")
    parser.add_argument("--skip-observed-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.embeddings, args.membership, args.node_rowmap):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    vectors = np.asarray(np.load(args.embeddings, mmap_mode="r"), dtype=np.float64)

    # (label, song) incidence, plus the chunk rows and weights each pair contributes
    # The membership file covers every label in the corpus; the graph is built only
    # over the eligible nodes, so take the universe from the node rowmap.
    labels = sorted(row["source_artist_label"] for row in read_rows(args.node_rowmap)
                    if row["graph_node_eligible"] == "true")
    label_index = {label: i for i, label in enumerate(labels)}
    rows = [row for row in read_rows(args.membership)
            if row["source_artist_label"] in label_index]
    pair_rows: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for row in rows:
        chunk = int(row["clean_row_index"])
        pair_rows[(row["source_artist_label"], row["song_id"])].append((
            chunk,
            # Both centroids weight by comparison_text_weight and differ only in which
            # rows they include. Using frozen_analysis_text_weight here reproduces the
            # primary centroid to a cosine of 0.9999983 rather than 1.0, which is enough
            # to move a pair across the rank-five boundary and inflate the edge count.
            # The weight sum must equal primary_effective_text_mass in the node rowmap.
            float(row["comparison_text_weight"]) if row["included_in_primary_centroid"] == "true" else 0.0,
            float(row["comparison_text_weight"]) if row["included_in_shared_text_exclusion_sensitivity"] == "true" else 0.0,
        ))
    pairs = sorted(pair_rows)
    pair_label = np.array([label_index[label] for label, _song in pairs])

    def build(assignment: np.ndarray) -> tuple[int, int]:
        primary_groups: list[list[tuple[int, float]]] = [[] for _ in labels]
        sensitivity_groups: list[list[tuple[int, float]]] = [[] for _ in labels]
        for position, pair in enumerate(pairs):
            target = assignment[position]
            for chunk, primary_weight, sensitivity_weight in pair_rows[pair]:
                if primary_weight > 0:
                    primary_groups[target].append((chunk, primary_weight))
                if sensitivity_weight > 0:
                    sensitivity_groups[target].append((chunk, sensitivity_weight))
        return graph_stats(centroids(vectors, primary_groups),
                           centroids(vectors, sensitivity_groups), args.top_k)

    print(f"{len(labels)} labels, {len(pairs)} label-song pairs, {vectors.shape[0]} chunk vectors")
    observed_edges, observed_connected = build(pair_label)
    print(f"observed: {observed_edges} edges connecting {observed_connected} labels "
          f"(published {PUBLISHED_EDGES} and {PUBLISHED_CONNECTED})")
    if (observed_edges, observed_connected) != (PUBLISHED_EDGES, PUBLISHED_CONNECTED):
        print("\nthe rebuilt graph does not reproduce the published counts, so the null would",
              file=sys.stderr)
        print("not be measuring the released rule. Refusing to continue; pass", file=sys.stderr)
        print("--skip-observed-check to override deliberately.", file=sys.stderr)
        if not args.skip_observed_check:
            return 1

    # Permuting the label column of the incidence preserves every label's song count
    # and every song's label count, so size and collaboration structure survive while
    # the association between a label and its lyrical content is destroyed.
    rng = np.random.default_rng(args.seed)
    null_edges, null_connected = [], []
    for replicate in range(args.replicates):
        edges, connected = build(rng.permutation(pair_label))
        null_edges.append(edges)
        null_connected.append(connected)
        if (replicate + 1) % max(1, args.replicates // 10) == 0:
            print(f"  {replicate + 1}/{args.replicates} replicates", flush=True)

    edges_array = np.array(null_edges)
    connected_array = np.array(null_connected)
    # one-sided permutation p, with the +1 correction for the observed arrangement
    p_edges = (int((edges_array >= observed_edges).sum()) + 1) / (args.replicates + 1)

    print(f"\nnull edge count      mean {edges_array.mean():.1f}  sd {edges_array.std(ddof=1):.1f}  "
          f"max {edges_array.max()}  95th pct {np.percentile(edges_array, 95):.1f}")
    print(f"null connected labels mean {connected_array.mean():.1f}  "
          f"max {connected_array.max()}")
    print(f"\nobserved {observed_edges} edges vs null mean {edges_array.mean():.1f}; "
          f"one-sided p = {p_edges:.4g}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "null_baseline.json").write_text(json.dumps({
        "rule": f"mutual top-{args.top_k} under both the primary and sensitivity centroids",
        "null": "permutation of the label column of the label-song incidence, "
                "preserving label song counts and song label counts",
        "replicates": args.replicates,
        "seed": args.seed,
        "observed": {"edges": observed_edges, "connected_labels": observed_connected,
                     "published_edges": PUBLISHED_EDGES,
                     "published_connected_labels": PUBLISHED_CONNECTED},
        "null_edges": {"mean": float(edges_array.mean()), "sd": float(edges_array.std(ddof=1)),
                       "min": int(edges_array.min()), "max": int(edges_array.max()),
                       "p95": float(np.percentile(edges_array, 95))},
        "null_connected_labels": {"mean": float(connected_array.mean()),
                                  "max": int(connected_array.max())},
        "p_value_one_sided": p_edges,
        "claim_boundary": "Tests whether the reciprocal top-five rule recovers more "
                          "structure than a label assignment carrying no information. "
                          "Not evidence about artists, influence, or collaboration.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten to {display_path(args.out)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
