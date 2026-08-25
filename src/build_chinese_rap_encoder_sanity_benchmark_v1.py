#!/usr/bin/env python3
"""Benchmark BGE-M3 against a lexical baseline on low-overlap same-song retrieval.

This is a corpus-internal sanity check, not a substitute for human semantic
judgements.  Public output is aggregate only; query rows and relevance sets are
kept in a private audit directory without lyric text.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


VERSION = "1.0.0"
ARTIFACT_ID = "chinese-rap-encoder-sanity-benchmark-v1"
ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "outputs" / "chinese-rap-lyrical-repertoire-graph-v2"
PRIVATE_GRAPH_DIR = ROOT / "work" / "private-chinese-rap-lyrical-repertoire-graph-v2"
CLEAN_DIR = ROOT / "work" / "private-canonical-lyric-text-sidecar-v1"
EMBED_DIR = ROOT / "work" / "private-canonical-clean-text-embeddings-v1"
OUT_DIR = ROOT / "outputs" / ARTIFACT_ID
PRIVATE_OUT_DIR = ROOT / "work" / f"private-{ARTIFACT_ID}"
QUERY_COUNT = 1000
LOW_OVERLAP_MAX_JACCARD = 0.15
MIN_EFFECTIVE_CHARACTERS = 50
TFIDF_MAX_FEATURES = 150_000
BLOCK_SIZE = 100


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


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def char_ngrams(value: str, n: int = 3) -> set[str]:
    compact = normalized(value)
    return {compact[index : index + n] for index in range(max(0, len(compact) - n + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def validate_source_contracts() -> dict[str, str]:
    graph_validation = json.loads((GRAPH_DIR / "validation.json").read_text(encoding="utf-8"))
    private_validation = json.loads((PRIVATE_GRAPH_DIR / "private_validation.json").read_text(encoding="utf-8"))
    clean_validation = json.loads((CLEAN_DIR / "private_validation.json").read_text(encoding="utf-8"))
    embedding_validation = json.loads((EMBED_DIR / "validation.json").read_text(encoding="utf-8"))
    for name, payload in (
        ("graph", graph_validation), ("private graph", private_validation),
        ("clean sidecar", clean_validation), ("clean embeddings", embedding_validation),
    ):
        passed = payload.get("status") == "pass" or payload.get("passed") is True
        if not passed or not all(item.get("passed") for item in payload.get("checks", [])):
            raise RuntimeError(f"The {name} validation is not passing.")
    contract = json.loads((EMBED_DIR / "canonical_clean_text_embedding_contract_v1.json").read_text(encoding="utf-8"))
    vector_path = EMBED_DIR / contract["vector_file"]["filename"]
    rowmap_path = EMBED_DIR / contract["row_map_file"]["filename"]
    if sha256_file(vector_path) != contract["vector_file"]["sha256"] or sha256_file(rowmap_path) != contract["row_map_file"]["sha256"]:
        raise RuntimeError("The clean embedding contract hashes do not match.")
    return {
        "graph_validation_sha256": sha256_file(GRAPH_DIR / "validation.json"),
        "private_graph_validation_sha256": sha256_file(PRIVATE_GRAPH_DIR / "private_validation.json"),
        "clean_manifest_sha256": sha256_file(CLEAN_DIR / "private_manifest.json"),
        "embedding_contract_sha256": sha256_file(EMBED_DIR / "canonical_clean_text_embedding_contract_v1.json"),
    }


def load_population() -> tuple[list[dict[str, Any]], np.ndarray, dict[str, str]]:
    lineage = validate_source_contracts()
    eligible = {
        row["artist_label_id"]
        for row in read_csv(GRAPH_DIR / "artist_repertoire_nodes.csv")
        if row["graph_node_eligible"] == "true"
    }
    clean_lookup: dict[tuple[str, str, str, str], str] = {}
    for row in read_csv(CLEAN_DIR / "cleaned_analysis_chunks_v1.csv"):
        clean_lookup[(row["song_id"], row["chunk_id"], row["canonical_lyric_text_sha256"], row["analysis_text_sha256"])] = row["analysis_text"]
    rowmap = read_csv(EMBED_DIR / "canonical_clean_text_embedding_row_map_v1.csv")
    rowmap_by_index = {int(row["clean_row_index"]): row for row in rowmap}
    records: list[dict[str, Any]] = []
    for row in read_csv(PRIVATE_GRAPH_DIR / "artist_chunk_membership_v2.csv"):
        if row["artist_label_id"] not in eligible or row["included_in_shared_text_exclusion_sensitivity"] != "true":
            continue
        clean_index = int(row["clean_row_index"])
        mapped = rowmap_by_index.get(clean_index)
        if mapped is None or mapped["analysis_text_sha256"] != row["analysis_text_sha256"]:
            raise RuntimeError("Graph membership does not align with the clean embedding row map.")
        lookup_key = (row["song_id"], row["chunk_id"], row["canonical_lyric_text_sha256"], row["analysis_text_sha256"])
        text = clean_lookup.get(lookup_key)
        if text is None:
            raise RuntimeError("Benchmark record does not rejoin the clean text sidecar.")
        records.append(
            {
                "clean_index": clean_index,
                "song": row["song_id"],
                "chunk": row["chunk_id"],
                "label_id": row["artist_label_id"],
                "text": text,
            }
        )
    records.sort(key=lambda item: item["clean_index"])
    if len({item["clean_index"] for item in records}) != len(records):
        raise RuntimeError("Benchmark candidate vector rows are not unique.")
    contract = json.loads((EMBED_DIR / "canonical_clean_text_embedding_contract_v1.json").read_text(encoding="utf-8"))
    all_vectors = np.load(EMBED_DIR / contract["vector_file"]["filename"], mmap_mode="r")
    indices = np.asarray([item["clean_index"] for item in records], dtype=np.int64)
    vectors = np.asarray(all_vectors[indices], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if vectors.shape != (len(records), 1024) or not np.allclose(norms, 1.0, atol=2e-4):
        raise RuntimeError("Candidate BGE-M3 vectors do not satisfy the clean contract.")
    return records, vectors, lineage


def metric_row(scores: np.ndarray, relevant: set[int], excluded: set[int]) -> dict[str, float]:
    masked = np.asarray(scores, dtype=np.float64).copy()
    if excluded:
        masked[np.fromiter(excluded, dtype=np.int64)] = -np.inf
    top10 = np.argpartition(masked, -10)[-10:]
    top10 = top10[np.argsort(masked[top10])[::-1]]
    hits = [rank for rank, candidate in enumerate(top10, start=1) if int(candidate) in relevant]
    best_relevant_score = max(masked[index] for index in relevant)
    reciprocal_rank = 1.0 / (1 + int(np.sum(masked > best_relevant_score + 1e-12)))
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    return {
        "reciprocal_rank": reciprocal_rank,
        "hit_1": float(any(rank <= 1 for rank in hits)),
        "hit_5": float(any(rank <= 5 for rank in hits)),
        "hit_10": float(bool(hits)),
        "ndcg_10": dcg / ideal if ideal else 0.0,
    }


def z_normalize_available(scores: np.ndarray, excluded: set[int]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).copy()
    available = np.ones(len(values), dtype=bool)
    if excluded:
        available[np.fromiter(excluded, dtype=np.int64)] = False
    mean = float(np.mean(values[available]))
    standard_deviation = float(np.std(values[available])) or 1.0
    values = (values - mean) / standard_deviation
    values[~available] = -1e9
    return values


def evaluate() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    records, bge_vectors, lineage = load_population()
    local_by_clean = {item["clean_index"]: index for index, item in enumerate(records)}
    by_song: dict[str, list[int]] = defaultdict(list)
    ngrams: list[set[str]] = []
    for index, item in enumerate(records):
        by_song[item["song"]].append(index)
        ngrams.append(char_ngrams(item["text"]))
    query_candidates: list[tuple[str, int, set[int], set[int]]] = []
    for song, members in by_song.items():
        if len(members) < 2:
            continue
        for query_index in members:
            if len(normalized(records[query_index]["text"])) < MIN_EFFECTIVE_CHARACTERS:
                continue
            relevant = {
                candidate
                for candidate in members
                if candidate != query_index
                and len(normalized(records[candidate]["text"])) >= MIN_EFFECTIVE_CHARACTERS
                and jaccard(ngrams[query_index], ngrams[candidate]) <= LOW_OVERLAP_MAX_JACCARD
            }
            if not relevant:
                continue
            # Same-song siblings with obvious lexical overlap are removed from
            # this low-overlap task rather than counted as negatives.
            excluded = {query_index} | (set(members) - relevant - {query_index})
            digest = hashlib.sha256(f"{song}\x1f{records[query_index]['clean_index']}".encode("utf-8")).hexdigest()
            query_candidates.append((digest, query_index, relevant, excluded))
    query_candidates.sort(key=lambda item: item[0])
    selected = query_candidates[: min(QUERY_COUNT, len(query_candidates))]
    if len(selected) < 500:
        raise RuntimeError("Too few low-overlap same-song retrieval queries.")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=3,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    tfidf = vectorizer.fit_transform([item["text"] for item in records])
    audit_rows: list[dict[str, Any]] = []
    metric_values: dict[str, dict[str, list[float]]] = {
        "BGE-M3 dense": defaultdict(list),
        "character 2-5 gram TF-IDF": defaultdict(list),
        "equal-weight z-score fusion": defaultdict(list),
    }
    for start in range(0, len(selected), BLOCK_SIZE):
        block = selected[start : start + BLOCK_SIZE]
        query_indices = np.asarray([item[1] for item in block], dtype=np.int64)
        bge_scores = np.asarray(bge_vectors @ bge_vectors[query_indices].T, dtype=np.float32)
        lexical_scores = np.asarray((tfidf @ tfidf[query_indices].T).toarray(), dtype=np.float32)
        for offset, (_digest, query_index, relevant, excluded) in enumerate(block):
            bge_metric = metric_row(bge_scores[:, offset], relevant, excluded)
            tfidf_metric = metric_row(lexical_scores[:, offset], relevant, excluded)
            fused_scores = 0.5 * z_normalize_available(bge_scores[:, offset], excluded) + 0.5 * z_normalize_available(lexical_scores[:, offset], excluded)
            fused_metric = metric_row(fused_scores, relevant, excluded)
            for metric, value in bge_metric.items():
                metric_values["BGE-M3 dense"][metric].append(value)
            for metric, value in tfidf_metric.items():
                metric_values["character 2-5 gram TF-IDF"][metric].append(value)
            for metric, value in fused_metric.items():
                metric_values["equal-weight z-score fusion"][metric].append(value)
            audit_rows.append(
                {
                    "query_clean_row_index": records[query_index]["clean_index"],
                    "query_song_id": records[query_index]["song"],
                    "query_chunk_id": records[query_index]["chunk"],
                    "relevant_low_overlap_siblings": len(relevant),
                    "excluded_same_song_high_overlap_siblings": len(excluded) - 1,
                    "bge_reciprocal_rank": f"{bge_metric['reciprocal_rank']:.8f}",
                    "tfidf_reciprocal_rank": f"{tfidf_metric['reciprocal_rank']:.8f}",
                    "fusion_reciprocal_rank": f"{fused_metric['reciprocal_rank']:.8f}",
                    "bge_hit_10": int(bge_metric["hit_10"]),
                    "tfidf_hit_10": int(tfidf_metric["hit_10"]),
                    "fusion_hit_10": int(fused_metric["hit_10"]),
                }
            )
        print(f"evaluated {min(start + BLOCK_SIZE, len(selected)):,}/{len(selected):,} queries", flush=True)

    rows = []
    for model in ("BGE-M3 dense", "character 2-5 gram TF-IDF", "equal-weight z-score fusion"):
        values = metric_values[model]
        rows.append(
            {
                "model": model,
                "queries": len(selected),
                "mrr": round(float(np.mean(values["reciprocal_rank"])), 6),
                "recall_at_1": round(float(np.mean(values["hit_1"])), 6),
                "recall_at_5": round(float(np.mean(values["hit_5"])), 6),
                "recall_at_10": round(float(np.mean(values["hit_10"])), 6),
                "ndcg_at_10": round(float(np.mean(values["ndcg_10"])), 6),
            }
        )
    result = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "task": "low-character-trigram-overlap same-song chunk retrieval",
        "claim_boundary": "Corpus-internal continuation/cohesion sanity check; not a human semantic-similarity benchmark and not proof of cultural interpretation.",
        "population": {
            "candidate_chunks": len(records),
            "candidate_source_labels": len({item["label_id"] for item in records}),
            "eligible_query_pool": len(query_candidates),
            "evaluated_queries": len(selected),
            "maximum_query_relevant_char_trigram_jaccard": LOW_OVERLAP_MAX_JACCARD,
            "minimum_effective_characters": MIN_EFFECTIVE_CHARACTERS,
        },
        "models": rows,
        "lineage": lineage,
    }
    return result, audit_rows, lineage


def method_text() -> str:
    return f"""# Encoder sanity benchmark

## Question

Can a representation retrieve another, lexically non-overlapping chunk from the
same song? This is a weakly supervised corpus-cohesion test used to supplement,
not replace, the BGE-M3 model rationale.

## Population

The candidate pool is the same 204-label comparison population used by the
stable repertoire graph, restricted to clean chunks retained after exact text
shared across source-credit labels is removed. Query–relevant pairs must belong
to the same canonical song, contain at least {MIN_EFFECTIVE_CHARACTERS} effective
characters, and have character-trigram Jaccard similarity at most
{LOW_OVERLAP_MAX_JACCARD:.2f}. Higher-overlap siblings are excluded from that
query rather than treated as false negatives. Query sampling is deterministic.

## Compared representations

- Validated local BGE-M3 1,024-dimensional dense vectors.
- Character 2–5 gram TF-IDF (`min_df=3`, at most {TFIDF_MAX_FEATURES:,} features).
- An untuned equal-weight fusion after per-query z-normalization of the dense
  and lexical scores.

The candidate pool and relevance sets are identical for both systems. Reported
metrics are MRR, Recall@1/5/10, and nDCG@10.

## Interpretation limit

Same-song membership is not a gold-standard semantic judgement. Shared context,
speaker, transcription practices, or song structure can make sibling chunks
cohere. A publishable encoder-selection claim still requires blinded human
Chinese-rap similarity and retrieval annotations plus additional embedding and
BM25 baselines.
"""


def build() -> None:
    result, audit_rows, lineage = evaluate()
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    public_stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}-", dir=OUT_DIR.parent))
    private_stage = Path(tempfile.mkdtemp(prefix=f".private-{ARTIFACT_ID}-", dir=PRIVATE_OUT_DIR.parent))
    try:
        atomic_write_json(public_stage / "analysis_summary.json", result)
        write_csv(public_stage / "benchmark_results.csv", result["models"], ["model", "queries", "mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10"])
        atomic_write_text(public_stage / "method.md", method_text())
        validation = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": utc_now(),
            "status": "pass",
            "checks": [
                {"name": "query_count_exact", "passed": result["population"]["evaluated_queries"] == min(QUERY_COUNT, result["population"]["eligible_query_pool"])},
                {"name": "three_model_rows", "passed": len(result["models"]) == 3},
                {"name": "metric_ranges", "passed": all(0 <= row[key] <= 1 for row in result["models"] for key in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10"))},
                {"name": "recall_monotonic", "passed": all(row["recall_at_1"] <= row["recall_at_5"] <= row["recall_at_10"] for row in result["models"])},
                {"name": "public_aggregate_only", "passed": not any(term in json.dumps(result).casefold() for term in ("song_id", "chunk_id", "analysis_text"))},
            ],
        }
        if not all(item["passed"] for item in validation["checks"]):
            validation["status"] = "fail"
            raise RuntimeError(f"Benchmark validation failed: {validation}")
        atomic_write_json(public_stage / "validation.json", validation)
        manifest = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": utc_now(),
            "classification": "public_aggregate_encoder_sanity_benchmark",
            "lineage": lineage,
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(public_stage.iterdir()) if path.is_file()},
        }
        atomic_write_json(public_stage / "manifest.json", manifest)
        write_csv(
            private_stage / "query_relevance_audit.csv",
            audit_rows,
            [
                "query_clean_row_index", "query_song_id", "query_chunk_id", "relevant_low_overlap_siblings",
                "excluded_same_song_high_overlap_siblings", "bge_reciprocal_rank", "tfidf_reciprocal_rank",
                "fusion_reciprocal_rank", "bge_hit_10", "tfidf_hit_10", "fusion_hit_10",
            ],
        )
        private_manifest = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "classification": "private_local_only_query_and_relevance_audit_without_lyric_text",
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
