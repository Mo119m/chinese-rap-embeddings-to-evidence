#!/usr/bin/env python3
"""Song-level bootstrap stability for the 86 validated repertoire edges."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "1.0.0"
ARTIFACT_ID = "chinese-rap-edge-bootstrap-v1"
ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "outputs" / "chinese-rap-lyrical-repertoire-graph-v2"
PRIVATE_GRAPH_DIR = ROOT / "work" / "private-chinese-rap-lyrical-repertoire-graph-v2"
EMBED_DIR = ROOT / "work" / "private-canonical-clean-text-embeddings-v1"
OUT_DIR = ROOT / "outputs" / ARTIFACT_ID
PRIVATE_OUT_DIR = ROOT / "work" / f"private-{ARTIFACT_ID}"
REPLICATES = 250
TOP_K = 5
RANDOM_SEED = 20260825


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def pair_key(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


def validate_inputs() -> dict[str, str]:
    for root, name in (
        (GRAPH_DIR, "validation.json"),
        (PRIVATE_GRAPH_DIR, "private_validation.json"),
        (EMBED_DIR, "validation.json"),
    ):
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        passed = payload.get("status") == "pass" or payload.get("passed") is True
        if not passed or not all(item.get("passed") for item in payload.get("checks", [])):
            raise RuntimeError(f"Input validation is not passing: {root / name}")
    contract = json.loads((EMBED_DIR / "canonical_clean_text_embedding_contract_v1.json").read_text(encoding="utf-8"))
    vector_path = EMBED_DIR / contract["vector_file"]["filename"]
    rowmap_path = EMBED_DIR / contract["row_map_file"]["filename"]
    if sha256_file(vector_path) != contract["vector_file"]["sha256"] or sha256_file(rowmap_path) != contract["row_map_file"]["sha256"]:
        raise RuntimeError("Clean embedding contract hashes do not match.")
    return {
        "graph_manifest_sha256": sha256_file(GRAPH_DIR / "manifest.json"),
        "graph_validation_sha256": sha256_file(GRAPH_DIR / "validation.json"),
        "private_graph_manifest_sha256": sha256_file(PRIVATE_GRAPH_DIR / "private_manifest.json"),
        "embedding_contract_sha256": sha256_file(EMBED_DIR / "canonical_clean_text_embedding_contract_v1.json"),
    }


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("A bootstrap centroid has zero norm.")
    return matrix / norms


def mutual_top_k(similarity: np.ndarray) -> np.ndarray:
    work = np.asarray(similarity, dtype=np.float32).copy()
    np.fill_diagonal(work, -np.inf)
    top = np.argpartition(-work, TOP_K - 1, axis=1)[:, :TOP_K]
    directed = np.zeros_like(work, dtype=bool)
    directed[np.arange(len(work))[:, None], top] = True
    return directed & directed.T


def build_song_arrays() -> tuple[
    list[str], dict[str, str], list[dict[str, np.ndarray]], np.ndarray, list[dict[str, str]], dict[str, str]
]:
    lineage = validate_inputs()
    vector_rowmap = read_csv(PRIVATE_GRAPH_DIR / "artist_repertoire_vector_rowmap_v2.csv")
    vector_rowmap.sort(key=lambda row: int(row["vector_row_index"]))
    identifiers = [row["artist_label_id"] for row in vector_rowmap]
    labels = {row["artist_label_id"]: row["source_artist_label"] for row in vector_rowmap}
    if len(identifiers) != 204 or len(set(identifiers)) != 204:
        raise RuntimeError("Unexpected eligible-label vector population.")
    index_by_id = {identifier: index for index, identifier in enumerate(identifiers)}
    clean_contract = json.loads((EMBED_DIR / "canonical_clean_text_embedding_contract_v1.json").read_text(encoding="utf-8"))
    chunk_vectors = np.load(EMBED_DIR / clean_contract["vector_file"]["filename"], mmap_mode="r")
    song_accumulator: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {
        "primary_num": np.zeros(1024, dtype=np.float64),
        "primary_mass": 0.0,
        "sensitivity_num": np.zeros(1024, dtype=np.float64),
        "sensitivity_mass": 0.0,
    }))
    for row in read_csv(PRIVATE_GRAPH_DIR / "artist_chunk_membership_v2.csv"):
        identifier = row["artist_label_id"]
        if identifier not in index_by_id or row["included_in_primary_centroid"] != "true":
            continue
        vector = np.asarray(chunk_vectors[int(row["clean_row_index"])], dtype=np.float64)
        weight = float(row["comparison_text_weight"])
        acc = song_accumulator[identifier][row["song_id"]]
        acc["primary_num"] += vector * weight
        acc["primary_mass"] += weight
        if row["included_in_shared_text_exclusion_sensitivity"] == "true":
            acc["sensitivity_num"] += vector * weight
            acc["sensitivity_mass"] += weight

    song_arrays: list[dict[str, np.ndarray]] = []
    recomputed_primary = np.zeros((len(identifiers), 1024), dtype=np.float64)
    recomputed_sensitivity = np.zeros((len(identifiers), 1024), dtype=np.float64)
    for label_index, identifier in enumerate(identifiers):
        songs = song_accumulator[identifier]
        primary_num = np.stack([item["primary_num"] for item in songs.values()])
        primary_mass = np.asarray([item["primary_mass"] for item in songs.values()], dtype=np.float64)
        sensitivity_num = np.stack([item["sensitivity_num"] for item in songs.values()])
        sensitivity_mass = np.asarray([item["sensitivity_mass"] for item in songs.values()], dtype=np.float64)
        if primary_mass.sum() <= 0 or sensitivity_mass.sum() <= 0:
            raise RuntimeError("An eligible label has no centroid mass.")
        recomputed_primary[label_index] = primary_num.sum(axis=0) / primary_mass.sum()
        recomputed_sensitivity[label_index] = sensitivity_num.sum(axis=0) / sensitivity_mass.sum()
        song_arrays.append(
            {
                "primary_num": primary_num,
                "primary_mass": primary_mass,
                "sensitivity_num": sensitivity_num,
                "sensitivity_mass": sensitivity_mass,
            }
        )
    recomputed = np.concatenate([l2_normalize(recomputed_primary), l2_normalize(recomputed_sensitivity)], axis=1).astype(np.float32)
    persisted = np.load(PRIVATE_GRAPH_DIR / "artist_repertoire_vectors_v2.npy")
    if persisted.shape != recomputed.shape or not np.allclose(persisted, recomputed, atol=3e-6):
        raise RuntimeError("Song aggregates do not reproduce the persisted graph centroids.")
    edges = read_csv(GRAPH_DIR / "artist_repertoire_edges.csv")
    return identifiers, labels, song_arrays, persisted, edges, lineage


def run_bootstrap() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    identifiers, labels, song_arrays, persisted, edges, lineage = build_song_arrays()
    index_by_id = {identifier: index for index, identifier in enumerate(identifiers)}
    original_edges = [(index_by_id[row["artist_label_id_a"]], index_by_id[row["artist_label_id_b"]]) for row in edges]
    stable_counts = np.zeros(len(edges), dtype=np.int64)
    primary_counts = np.zeros(len(edges), dtype=np.int64)
    sensitivity_counts = np.zeros(len(edges), dtype=np.int64)
    selected_edge_counts: list[int] = []
    rng = np.random.default_rng(RANDOM_SEED)
    resample_retries = 0
    for replicate in range(REPLICATES):
        primary = np.zeros((len(identifiers), 1024), dtype=np.float64)
        sensitivity = np.zeros((len(identifiers), 1024), dtype=np.float64)
        for label_index, arrays in enumerate(song_arrays):
            song_count = len(arrays["primary_mass"])
            for attempt in range(100):
                sampled = rng.integers(0, song_count, size=song_count)
                sensitivity_mass = float(arrays["sensitivity_mass"][sampled].sum())
                if sensitivity_mass > 0:
                    break
                resample_retries += 1
            else:
                raise RuntimeError("Could not obtain a positive-mass sensitivity bootstrap sample.")
            primary[label_index] = arrays["primary_num"][sampled].sum(axis=0) / float(arrays["primary_mass"][sampled].sum())
            sensitivity[label_index] = arrays["sensitivity_num"][sampled].sum(axis=0) / sensitivity_mass
        primary = l2_normalize(primary).astype(np.float32)
        sensitivity = l2_normalize(sensitivity).astype(np.float32)
        primary_mutual = mutual_top_k(primary @ primary.T)
        sensitivity_mutual = mutual_top_k(sensitivity @ sensitivity.T)
        stable = primary_mutual & sensitivity_mutual
        selected_edge_counts.append(int(np.triu(stable, 1).sum()))
        for edge_index, (left, right) in enumerate(original_edges):
            primary_counts[edge_index] += int(primary_mutual[left, right])
            sensitivity_counts[edge_index] += int(sensitivity_mutual[left, right])
            stable_counts[edge_index] += int(stable[left, right])
        if (replicate + 1) % 25 == 0:
            print(f"completed {replicate + 1}/{REPLICATES} song bootstraps", flush=True)

    rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for edge_index, row in enumerate(edges):
        stable_probability = stable_counts[edge_index] / REPLICATES
        public = {
            "edge_key": pair_key(row["artist_label_id_a"], row["artist_label_id_b"]),
            "artist_label_id_a": row["artist_label_id_a"],
            "source_artist_label_a": row["source_artist_label_a"],
            "artist_label_id_b": row["artist_label_id_b"],
            "source_artist_label_b": row["source_artist_label_b"],
            "primary_mutual_probability": f"{primary_counts[edge_index] / REPLICATES:.6f}",
            "sensitivity_mutual_probability": f"{sensitivity_counts[edge_index] / REPLICATES:.6f}",
            "two_representation_edge_probability": f"{stable_probability:.6f}",
            "bootstrap_band": "high" if stable_probability >= 0.80 else "moderate" if stable_probability >= 0.50 else "low",
        }
        rows.append(public)
        private_rows.append({**public, "selected_replicates": int(stable_counts[edge_index]), "replicates": REPLICATES})
    rows.sort(key=lambda item: (-float(item["two_representation_edge_probability"]), item["edge_key"]))
    probabilities = np.asarray([float(row["two_representation_edge_probability"]) for row in rows], dtype=np.float64)
    summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "method": "within-source-label song bootstrap; reciprocal top-5 recomputed in primary and exact-cross-label-shared-text-exclusion representations",
        "counts": {
            "eligible_labels": len(identifiers),
            "original_stable_edges": len(edges),
            "replicates": REPLICATES,
            "edges_probability_at_least_0_50": int(np.sum(probabilities >= 0.50)),
            "edges_probability_at_least_0_80": int(np.sum(probabilities >= 0.80)),
            "edges_probability_at_least_0_90": int(np.sum(probabilities >= 0.90)),
        },
        "edge_probability_distribution": {
            "minimum": round(float(np.min(probabilities)), 6),
            "q25": round(float(np.quantile(probabilities, 0.25)), 6),
            "median": round(float(np.median(probabilities)), 6),
            "q75": round(float(np.quantile(probabilities, 0.75)), 6),
            "maximum": round(float(np.max(probabilities)), 6),
        },
        "bootstrap_graph_edge_count": {
            "minimum": int(min(selected_edge_counts)),
            "median": float(np.median(selected_edge_counts)),
            "maximum": int(max(selected_edge_counts)),
        },
        "resample_retries_for_zero_sensitivity_mass": resample_retries,
        "claim_boundary": "Selection probability under within-label song resampling for the existing representation rule; not a confidence probability for a social or cultural relationship.",
        "lineage": lineage,
    }
    return summary, rows, private_rows, lineage


def method_text() -> str:
    return f"""# Song-level edge bootstrap

The bootstrap tests whether each of the 86 existing stable semantic edges is
selected again when the observed songs inside every eligible source-credit
label are resampled with replacement.

For each of {REPLICATES} deterministic replicates:

1. sample the label's observed songs with replacement;
2. reconstruct duplicate-weighted BGE-M3 centroids for the primary and
   exact-cross-label-shared-text-exclusion representations;
3. L2-normalize the centroids and recompute every cosine similarity;
4. recompute reciprocal top-{TOP_K} edges separately in both representations;
5. record whether an original edge survives their intersection.

The reported probability is a selection frequency under this empirical
resampling scheme. It is not a Bayesian posterior, a p-value, or evidence of a
social, collaborative, geographic, or influence relationship.
"""


def build() -> None:
    summary, rows, private_rows, lineage = run_bootstrap()
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    public_stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}-", dir=OUT_DIR.parent))
    private_stage = Path(tempfile.mkdtemp(prefix=f".private-{ARTIFACT_ID}-", dir=PRIVATE_OUT_DIR.parent))
    try:
        atomic_write_json(public_stage / "analysis_summary.json", summary)
        write_csv(
            public_stage / "stable_edge_bootstrap.csv", rows,
            [
                "edge_key", "artist_label_id_a", "source_artist_label_a", "artist_label_id_b",
                "source_artist_label_b", "primary_mutual_probability", "sensitivity_mutual_probability",
                "two_representation_edge_probability", "bootstrap_band",
            ],
        )
        atomic_write_text(public_stage / "method.md", method_text())
        validation = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": utc_now(),
            "status": "pass",
            "checks": [
                {"name": "edge_count_exact", "passed": len(rows) == 86},
                {"name": "edge_keys_unique", "passed": len({row["edge_key"] for row in rows}) == len(rows)},
                {"name": "probability_ranges", "passed": all(0 <= float(row[key]) <= 1 for row in rows for key in ("primary_mutual_probability", "sensitivity_mutual_probability", "two_representation_edge_probability"))},
                {"name": "intersection_probability_bounded", "passed": all(float(row["two_representation_edge_probability"]) <= min(float(row["primary_mutual_probability"]), float(row["sensitivity_mutual_probability"])) + 1e-9 for row in rows)},
                {"name": "replicate_count_exact", "passed": summary["counts"]["replicates"] == REPLICATES},
            ],
        }
        if not all(item["passed"] for item in validation["checks"]):
            validation["status"] = "fail"
            raise RuntimeError(f"Bootstrap validation failed: {validation}")
        atomic_write_json(public_stage / "validation.json", validation)
        manifest = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": utc_now(),
            "classification": "public_aggregate_song_bootstrap_edge_stability",
            "lineage": lineage,
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(public_stage.iterdir()) if path.is_file()},
        }
        atomic_write_json(public_stage / "manifest.json", manifest)
        write_csv(
            private_stage / "stable_edge_bootstrap_audit.csv", private_rows,
            [
                "edge_key", "artist_label_id_a", "source_artist_label_a", "artist_label_id_b",
                "source_artist_label_b", "primary_mutual_probability", "sensitivity_mutual_probability",
                "two_representation_edge_probability", "bootstrap_band", "selected_replicates", "replicates",
            ],
        )
        private_manifest = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "classification": "private_local_only_bootstrap_edge_audit_without_lyrics_or_embeddings",
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in private_stage.iterdir() if path.is_file()},
        }
        atomic_write_json(private_stage / "private_manifest.json", private_manifest)
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        os.replace(public_stage, OUT_DIR)
        if PRIVATE_OUT_DIR.exists():
            shutil.rmtree(PRIVATE_OUT_DIR)
        os.replace(private_stage, PRIVATE_OUT_DIR)
    except Exception:
        shutil.rmtree(public_stage, ignore_errors=True)
        shutil.rmtree(private_stage, ignore_errors=True)
        raise


def main() -> int:
    build()
    print(json.dumps({"artifact": ARTIFACT_ID, "status": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
