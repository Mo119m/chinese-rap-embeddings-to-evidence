#!/usr/bin/env python3
"""Build a leakage-controlled source-credit-label repertoire retrieval task.

The public artifact is aggregate-only. Song identifiers, per-query rankings,
duplicate-pair evidence, lyric text, chunk identifiers, and vectors remain in
the private audit directory under ``work/``.

The evaluated question is deliberately narrow: given one held-out song's
cleaned lyrical representation, how highly does a corpus source-credit label's
remaining lyrical repertoire rank? This is not artist identification and does
not establish biography, influence, collaboration, genre, or social ties.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize


VERSION = "1.1.0"
ARTIFACT_ID = "chinese-rap-downstream-retrieval-v1"
ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "outputs" / "chinese-rap-lyrical-repertoire-graph-v2"
PRIVATE_GRAPH_DIR = ROOT / "work" / "private-chinese-rap-lyrical-repertoire-graph-v2"
CLEAN_DIR = ROOT / "work" / "private-canonical-lyric-text-sidecar-v1"
EMBED_DIR = ROOT / "work" / "private-canonical-clean-text-embeddings-v1"
BENCHMARK_DIR = ROOT / "outputs" / "chinese-rap-encoder-sanity-benchmark-v1"
OUT_DIR = ROOT / "outputs" / ARTIFACT_ID
PRIVATE_OUT_DIR = ROOT / "work" / f"private-{ARTIFACT_ID}"

MIN_EFFECTIVE_CHARACTERS = 50
NEAR_DUPLICATE_JACCARD = 0.80
TFIDF_MAX_FEATURES = 150_000
BOOTSTRAP_REPLICATES = 5_000
RANDOM_SEED = 20260825

SYSTEM_BGE = "BGE-M3 dense (strict)"
SYSTEM_TFIDF = "character 2-5 gram TF-IDF (strict)"
SYSTEM_FUSION = "equal-weight z-score fusion (strict)"
SYSTEM_RAW = "raw-cosine equal fusion (score-scale ablation)"
SYSTEM_EXACT_ONLY = "z-score fusion (near-duplicate guard removed)"
SYSTEM_SHARED = "z-score fusion (shared-text exclusion removed)"
SYSTEMS = (
    SYSTEM_BGE,
    SYSTEM_TFIDF,
    SYSTEM_FUSION,
    SYSTEM_RAW,
    SYSTEM_EXACT_ONLY,
    SYSTEM_SHARED,
)
PRIMARY_SYSTEMS = {SYSTEM_BGE, SYSTEM_TFIDF, SYSTEM_FUSION}
METRICS = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10")

CLAIM_BOUNDARY = (
    "Held-out-song retrieval of corpus source-credit-label lyrical repertoire similarity; "
    "not artist identification, biography, influence, collaboration, genre, social relation, "
    "or a human semantic-similarity gold standard."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def l2_normalize_dense(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("Encountered a zero-norm dense representation.")
    return matrix / norms


def validation_passes(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    passed = payload.get("status") == "pass" or payload.get("passed") is True
    return bool(passed and all(item.get("passed") for item in payload.get("checks", [])))


def validate_source_contracts() -> dict[str, str]:
    validations = {
        "graph": GRAPH_DIR / "validation.json",
        "private_graph": PRIVATE_GRAPH_DIR / "private_validation.json",
        "clean_text": CLEAN_DIR / "private_validation.json",
        "clean_embeddings": EMBED_DIR / "validation.json",
        "encoder_benchmark": BENCHMARK_DIR / "validation.json",
    }
    for label, path in validations.items():
        if not path.is_file() or not validation_passes(path):
            raise RuntimeError(f"Required passing validation is missing or stale: {label}")

    contract_path = EMBED_DIR / "canonical_clean_text_embedding_contract_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    vector_path = EMBED_DIR / contract["vector_file"]["filename"]
    rowmap_path = EMBED_DIR / contract["row_map_file"]["filename"]
    if sha256_file(vector_path) != contract["vector_file"]["sha256"]:
        raise RuntimeError("BGE-M3 vector hash does not match the clean embedding contract.")
    if sha256_file(rowmap_path) != contract["row_map_file"]["sha256"]:
        raise RuntimeError("Embedding row-map hash does not match the clean embedding contract.")

    return {
        "builder_code_sha256": sha256_file(Path(__file__).resolve()),
        "graph_validation_sha256": sha256_file(validations["graph"]),
        "private_graph_validation_sha256": sha256_file(validations["private_graph"]),
        "clean_text_validation_sha256": sha256_file(validations["clean_text"]),
        "clean_embedding_contract_sha256": sha256_file(contract_path),
        "encoder_benchmark_summary_sha256": sha256_file(BENCHMARK_DIR / "analysis_summary.json"),
        "encoder_benchmark_results_sha256": sha256_file(BENCHMARK_DIR / "benchmark_results.csv"),
    }


@dataclass
class CorpusData:
    songs: list[str]
    label_ids: list[str]
    label_names: list[str]
    song_label_index: np.ndarray
    strict_docs: list[str]
    primary_docs: list[str]
    strict_dense: np.ndarray
    primary_dense: np.ndarray
    strict_chunk_counts: np.ndarray
    primary_chunk_counts: np.ndarray
    normalized_strict_docs: list[str]
    primary_songs_all_eligible: int
    strict_songs_before_length_filter: int
    songs_lost_entirely_to_shared_text_exclusion: int
    primary_population_rows: int
    strict_population_rows: int
    short_songs_excluded: int
    shared_rows_removed_all_eligible: int
    shared_rows_removed_query_population: int
    source_cross_label_shared_hashes_after_strict_filter: int


def load_corpus() -> CorpusData:
    node_rows = read_csv(GRAPH_DIR / "artist_repertoire_nodes.csv")
    eligible_rows = [row for row in node_rows if row["graph_node_eligible"] == "true"]
    label_ids = sorted(row["artist_label_id"] for row in eligible_rows)
    if len(label_ids) != 204 or len(set(label_ids)) != 204:
        raise RuntimeError("Expected exactly 204 graph-eligible source-credit labels.")
    label_name_by_id = {row["artist_label_id"]: row["source_artist_label"] for row in eligible_rows}
    label_index_by_id = {identifier: index for index, identifier in enumerate(label_ids)}

    clean_lookup: dict[tuple[str, str, str], str] = {}
    for row in read_csv(CLEAN_DIR / "cleaned_analysis_chunks_v1.csv"):
        key = (row["song_id"], row["chunk_id"], row["analysis_text_sha256"])
        if key in clean_lookup:
            raise RuntimeError("Clean-text sidecar key is not unique.")
        clean_lookup[key] = row["analysis_text"]

    rowmap = read_csv(EMBED_DIR / "canonical_clean_text_embedding_row_map_v1.csv")
    rowmap_by_index = {int(row["clean_row_index"]): row for row in rowmap}
    contract = json.loads((EMBED_DIR / "canonical_clean_text_embedding_contract_v1.json").read_text(encoding="utf-8"))
    all_vectors = np.load(EMBED_DIR / contract["vector_file"]["filename"], mmap_mode="r")
    if all_vectors.shape != (len(rowmap), 1024):
        raise RuntimeError("Clean BGE-M3 matrix shape differs from its row map.")

    primary_by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
    strict_by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
    song_label: dict[str, str] = {}
    strict_hash_labels: dict[str, set[str]] = defaultdict(set)
    primary_population_rows = 0
    strict_population_rows = 0
    shared_rows_removed_all_eligible = 0

    for row in read_csv(PRIVATE_GRAPH_DIR / "artist_chunk_membership_v2.csv"):
        identifier = row["artist_label_id"]
        if identifier not in label_index_by_id or row["included_in_primary_centroid"] != "true":
            continue
        primary_population_rows += 1
        clean_index = int(row["clean_row_index"])
        mapped = rowmap_by_index.get(clean_index)
        if mapped is None or mapped["analysis_text_sha256"] != row["analysis_text_sha256"]:
            raise RuntimeError("Membership does not align with the clean embedding row map.")
        key = (row["song_id"], row["chunk_id"], row["analysis_text_sha256"])
        text = clean_lookup.get(key)
        if text is None:
            raise RuntimeError("Membership cannot rejoin the clean-text sidecar.")
        item = {
            "clean_index": clean_index,
            "song": row["song_id"],
            "label_id": identifier,
            "text": text,
            "text_sha": row["analysis_text_sha256"],
            "weight": float(row["comparison_text_weight"]),
        }
        existing_label = song_label.setdefault(row["song_id"], identifier)
        if existing_label != identifier:
            raise RuntimeError("A song belongs to more than one eligible source-credit label.")
        primary_by_song[row["song_id"]].append(item)
        if row["included_in_shared_text_exclusion_sensitivity"] == "true":
            strict_population_rows += 1
            strict_by_song[row["song_id"]].append(item)
            strict_hash_labels[row["analysis_text_sha256"]].add(identifier)
        else:
            shared_rows_removed_all_eligible += 1

    cross_label_shared_hashes = sum(len(labels) > 1 for labels in strict_hash_labels.values())
    if cross_label_shared_hashes:
        raise RuntimeError("Strict sensitivity population still has exact text shared across labels.")

    normalized_by_song: dict[str, str] = {}
    short_songs_excluded = 0
    for song, items in strict_by_song.items():
        document = "\n".join(item["text"] for item in sorted(items, key=lambda value: value["clean_index"]))
        compact = normalized_text(document)
        if len(compact) < MIN_EFFECTIVE_CHARACTERS:
            short_songs_excluded += 1
            continue
        normalized_by_song[song] = compact

    songs = sorted(normalized_by_song)
    if len(songs) < 5_000:
        raise RuntimeError("Unexpectedly small strict song-held-out population.")
    song_label_index = np.asarray([label_index_by_id[song_label[song]] for song in songs], dtype=np.int32)
    if len(set(song_label_index.tolist())) != len(label_ids):
        raise RuntimeError("Some eligible labels have no length-qualified song queries.")

    def build_representation(source: dict[str, list[dict[str, Any]]]) -> tuple[list[str], np.ndarray, np.ndarray]:
        documents: list[str] = []
        vectors: list[np.ndarray] = []
        chunk_counts: list[int] = []
        for song in songs:
            items = sorted(source[song], key=lambda value: value["clean_index"])
            if not items:
                raise RuntimeError("A query song is missing from one representation.")
            documents.append("\n".join(item["text"] for item in items))
            weights = np.asarray([item["weight"] for item in items], dtype=np.float64)
            if np.any(weights <= 0) or float(weights.sum()) <= 0:
                raise RuntimeError("Invalid comparison-text weights.")
            chunk_matrix = np.asarray(all_vectors[[item["clean_index"] for item in items]], dtype=np.float64)
            vectors.append(np.average(chunk_matrix, axis=0, weights=weights).astype(np.float32))
            chunk_counts.append(len(items))
        return documents, l2_normalize_dense(np.stack(vectors)), np.asarray(chunk_counts, dtype=np.int32)

    strict_docs, strict_dense, strict_chunk_counts = build_representation(strict_by_song)
    primary_docs, primary_dense, primary_chunk_counts = build_representation(primary_by_song)
    shared_removed_query = int(np.sum(primary_chunk_counts - strict_chunk_counts))

    return CorpusData(
        songs=songs,
        label_ids=label_ids,
        label_names=[label_name_by_id[identifier] for identifier in label_ids],
        song_label_index=song_label_index,
        strict_docs=strict_docs,
        primary_docs=primary_docs,
        strict_dense=strict_dense,
        primary_dense=primary_dense,
        strict_chunk_counts=strict_chunk_counts,
        primary_chunk_counts=primary_chunk_counts,
        normalized_strict_docs=[normalized_by_song[song] for song in songs],
        primary_songs_all_eligible=len(primary_by_song),
        strict_songs_before_length_filter=len(strict_by_song),
        songs_lost_entirely_to_shared_text_exclusion=len(set(primary_by_song) - set(strict_by_song)),
        primary_population_rows=primary_population_rows,
        strict_population_rows=strict_population_rows,
        short_songs_excluded=short_songs_excluded,
        shared_rows_removed_all_eligible=shared_rows_removed_all_eligible,
        shared_rows_removed_query_population=shared_removed_query,
        source_cross_label_shared_hashes_after_strict_filter=cross_label_shared_hashes,
    )


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root

    def group_ids(self) -> np.ndarray:
        roots = [self.find(index) for index in range(len(self.parent))]
        root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
        return np.asarray([root_to_group[root] for root in roots], dtype=np.int32)


@dataclass
class DuplicateAudit:
    strict_group_ids: np.ndarray
    exact_group_ids: np.ndarray
    pair_rows: list[dict[str, Any]]
    candidate_pairs: int
    exact_pairs: int
    nonexact_near_pairs: int
    cross_label_pairs: int
    strict_groups: int
    exact_groups: int


def detect_duplicate_groups(corpus: CorpusData) -> DuplicateAudit:
    documents = corpus.normalized_strict_docs
    trigram_sets = [{document[index : index + 3] for index in range(len(document) - 2)} for document in documents]
    if any(not values for values in trigram_sets):
        raise RuntimeError("A length-qualified query has no character trigrams.")

    exact_union = UnionFind(len(documents))
    exact_by_hash: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        exact_by_hash[hashlib.sha256(document.encode("utf-8")).hexdigest()].append(index)
    for indices in exact_by_hash.values():
        for index in indices[1:]:
            exact_union.union(indices[0], index)

    # Exact all-pairs threshold join using the standard global-order prefix
    # filter. For threshold t, a set keeps |S|-ceil(t|S|)+1 prefix tokens;
    # any pair with Jaccard >= t must share a prefix token. Candidate pairs are
    # then verified with the full trigram sets.
    token_frequency = Counter(token for values in trigram_sets for token in values)
    inverted_prefix: dict[str, list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for right, values in enumerate(trigram_sets):
        prefix_length = len(values) - math.ceil(NEAR_DUPLICATE_JACCARD * len(values)) + 1
        prefix = sorted(values, key=lambda token: (token_frequency[token], token))[:prefix_length]
        for token in prefix:
            for left in inverted_prefix[token]:
                smaller = min(len(values), len(trigram_sets[left]))
                larger = max(len(values), len(trigram_sets[left]))
                if smaller >= math.ceil(NEAR_DUPLICATE_JACCARD * larger):
                    candidates.add((left, right))
            inverted_prefix[token].append(right)

    strict_union = UnionFind(len(documents))
    pair_rows: list[dict[str, Any]] = []
    exact_pairs = 0
    cross_label_pairs = 0
    for left, right in sorted(candidates):
        intersection = len(trigram_sets[left] & trigram_sets[right])
        union = len(trigram_sets[left] | trigram_sets[right])
        similarity = intersection / union if union else 0.0
        if similarity + 1e-12 < NEAR_DUPLICATE_JACCARD:
            continue
        strict_union.union(left, right)
        exact = documents[left] == documents[right]
        exact_pairs += int(exact)
        cross_label = corpus.song_label_index[left] != corpus.song_label_index[right]
        cross_label_pairs += int(cross_label)
        pair_rows.append(
            {
                "private_pair_id": f"DUP-{len(pair_rows) + 1:04d}",
                "song_id_a": corpus.songs[left],
                "source_credit_label_a": corpus.label_names[int(corpus.song_label_index[left])],
                "song_id_b": corpus.songs[right],
                "source_credit_label_b": corpus.label_names[int(corpus.song_label_index[right])],
                "pair_type": "exact_normalized_text" if exact else "near_duplicate_trigram_jaccard",
                "character_trigram_jaccard": f"{similarity:.8f}",
                "cross_source_credit_label": str(bool(cross_label)).lower(),
                "effective_characters_a": len(documents[left]),
                "effective_characters_b": len(documents[right]),
            }
        )

    strict_group_ids = strict_union.group_ids()
    exact_group_ids = exact_union.group_ids()
    # Every exact normalized-text pair must also pass the near-duplicate rule.
    if any(strict_group_ids[index] != strict_group_ids[indices[0]] for indices in exact_by_hash.values() for index in indices):
        raise RuntimeError("Exact normalized duplicates were not joined by the strict grouping rule.")
    return DuplicateAudit(
        strict_group_ids=strict_group_ids,
        exact_group_ids=exact_group_ids,
        pair_rows=pair_rows,
        candidate_pairs=len(candidates),
        exact_pairs=exact_pairs,
        nonexact_near_pairs=len(pair_rows) - exact_pairs,
        cross_label_pairs=cross_label_pairs,
        strict_groups=len(set(strict_group_ids.tolist())),
        exact_groups=len(set(exact_group_ids.tolist())),
    )


def fit_tfidf(documents: list[str]) -> sparse.csr_matrix:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=3,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(documents).tocsr().astype(np.float32)
    row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    if matrix.shape[1] <= 0 or not np.allclose(row_norms, 1.0, atol=2e-5):
        raise RuntimeError("TF-IDF song representations are not valid L2-normalized rows.")
    return matrix


@dataclass
class ProfileScores:
    dense: np.ndarray
    lexical: np.ndarray
    label_group_counts: np.ndarray
    minimum_training_groups_after_holdout: int


def score_leave_group_out(
    dense: np.ndarray,
    lexical: sparse.csr_matrix,
    label_index: np.ndarray,
    group_ids: np.ndarray,
    label_count: int,
) -> ProfileScores:
    song_count = len(label_index)
    groups: dict[int, list[int]] = defaultdict(list)
    group_label_counts: Counter[tuple[int, int]] = Counter()
    for index, (group, label) in enumerate(zip(group_ids.tolist(), label_index.tolist())):
        groups[int(group)].append(index)
        group_label_counts[(int(group), int(label))] += 1

    weights = np.asarray(
        [1.0 / group_label_counts[(int(group_ids[index]), int(label_index[index]))] for index in range(song_count)],
        dtype=np.float32,
    )
    label_group_counts = np.bincount(label_index, weights=weights, minlength=label_count)
    if not np.allclose(label_group_counts, np.rint(label_group_counts), atol=1e-6):
        raise RuntimeError("Duplicate-component weighting does not sum to whole label groups.")

    dense_sums = np.zeros((label_count, dense.shape[1]), dtype=np.float32)
    np.add.at(dense_sums, label_index, dense * weights[:, None])
    dense_norms = np.linalg.norm(dense_sums, axis=1)
    if np.any(dense_norms <= 0):
        raise RuntimeError("A dense label repertoire sum is empty.")
    dense_scores = np.asarray(dense @ (dense_sums / dense_norms[:, None]).T, dtype=np.float32)

    membership = sparse.csr_matrix(
        (weights, (label_index, np.arange(song_count))), shape=(label_count, song_count), dtype=np.float32
    )
    lexical_sums = (membership @ lexical).tocsr()
    lexical_norms = np.sqrt(np.asarray(lexical_sums.multiply(lexical_sums).sum(axis=1)).ravel())
    if np.any(lexical_norms <= 0):
        raise RuntimeError("A lexical label repertoire sum is empty.")
    lexical_centroids = sklearn_normalize(lexical_sums, norm="l2", axis=1, copy=True)
    lexical_scores = np.asarray((lexical @ lexical_centroids.T).toarray(), dtype=np.float32)

    minimum_training_groups = int(np.min(label_group_counts) - 1)
    if minimum_training_groups <= 0:
        raise RuntimeError("A label has no training repertoire after held-out-group removal.")

    for group, query_indices_list in groups.items():
        query_indices = np.asarray(query_indices_list, dtype=np.int64)
        impacted_labels = sorted({int(label_index[index]) for index in query_indices_list})
        if len(query_indices_list) == 1:
            query = int(query_indices[0])
            label = int(label_index[query])
            dense_dot = float(np.dot(dense[query], dense_sums[label]))
            dense_leave_norm_sq = float(dense_norms[label] ** 2 + 1.0 - 2.0 * dense_dot)
            lexical_dot = float((lexical[query] @ lexical_sums.getrow(label).T).toarray()[0, 0])
            lexical_leave_norm_sq = float(lexical_norms[label] ** 2 + 1.0 - 2.0 * lexical_dot)
            if dense_leave_norm_sq <= 1e-12 or lexical_leave_norm_sq <= 1e-12:
                raise RuntimeError("A singleton holdout produced a zero-norm profile.")
            dense_scores[query, label] = (dense_dot - 1.0) / math.sqrt(dense_leave_norm_sq)
            lexical_scores[query, label] = (lexical_dot - 1.0) / math.sqrt(lexical_leave_norm_sq)
            continue

        for label in impacted_labels:
            members = np.asarray([index for index in query_indices_list if int(label_index[index]) == label], dtype=np.int64)
            member_weights = weights[members]
            dense_contribution = np.sum(dense[members] * member_weights[:, None], axis=0)
            dense_leave = dense_sums[label] - dense_contribution
            dense_leave_norm = float(np.linalg.norm(dense_leave))
            if dense_leave_norm <= 1e-12:
                raise RuntimeError("A multi-song holdout produced a zero-norm dense profile.")
            dense_scores[query_indices, label] = dense[query_indices] @ (dense_leave / dense_leave_norm)

            contribution = lexical[members].multiply(member_weights[:, None]).sum(axis=0)
            lexical_leave = lexical_sums.getrow(label) - sparse.csr_matrix(contribution)
            lexical_leave_norm = float(np.sqrt(lexical_leave.multiply(lexical_leave).sum()))
            if lexical_leave_norm <= 1e-12:
                raise RuntimeError("A multi-song holdout produced a zero-norm lexical profile.")
            lexical_scores[query_indices, label] = np.asarray(
                (lexical[query_indices] @ lexical_leave.T).toarray(), dtype=np.float32
            ).ravel() / lexical_leave_norm

    if not np.all(np.isfinite(dense_scores)) or not np.all(np.isfinite(lexical_scores)):
        raise RuntimeError("Non-finite retrieval scores were produced.")
    return ProfileScores(
        dense=dense_scores,
        lexical=lexical_scores,
        label_group_counts=np.rint(label_group_counts).astype(np.int32),
        minimum_training_groups_after_holdout=minimum_training_groups,
    )


def zscore_rows(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=1, keepdims=True)
    deviations = values.std(axis=1, keepdims=True)
    if np.any(deviations <= 1e-12):
        raise RuntimeError("A query has no candidate-label score variance.")
    return (values - means) / deviations


@dataclass
class Evaluation:
    ranks: dict[str, np.ndarray]
    top_labels: dict[str, np.ndarray]
    metric_arrays: dict[str, dict[str, np.ndarray]]
    strict_profiles: ProfileScores
    exact_only_profiles: ProfileScores
    shared_profiles: ProfileScores
    strict_tfidf_features: int
    shared_tfidf_features: int


def rank_system(scores: np.ndarray, true_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    song_count, label_count = scores.shape
    true_scores = scores[np.arange(song_count), true_labels]
    earlier_ties = np.arange(label_count)[None, :] < true_labels[:, None]
    ranks = 1 + np.sum(scores > true_scores[:, None] + 1e-12, axis=1)
    ranks += np.sum((np.abs(scores - true_scores[:, None]) <= 1e-12) & earlier_ties, axis=1)
    top10 = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    return ranks.astype(np.int32), top10.astype(np.int32)


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, np.ndarray]:
    ranks_float = ranks.astype(np.float64)
    return {
        "mrr": 1.0 / ranks_float,
        "recall_at_1": (ranks <= 1).astype(np.float64),
        "recall_at_5": (ranks <= 5).astype(np.float64),
        "recall_at_10": (ranks <= 10).astype(np.float64),
        "ndcg_at_10": np.where(ranks <= 10, 1.0 / np.log2(ranks_float + 1.0), 0.0),
    }


def evaluate_models(corpus: CorpusData, duplicate: DuplicateAudit) -> Evaluation:
    print("fitting strict character 2-5 gram TF-IDF", flush=True)
    strict_tfidf = fit_tfidf(corpus.strict_docs)
    print("scoring strict group-held-out profiles", flush=True)
    strict_profiles = score_leave_group_out(
        corpus.strict_dense,
        strict_tfidf,
        corpus.song_label_index,
        duplicate.strict_group_ids,
        len(corpus.label_ids),
    )
    print("scoring exact-only duplicate-guard ablation", flush=True)
    exact_profiles = score_leave_group_out(
        corpus.strict_dense,
        strict_tfidf,
        corpus.song_label_index,
        duplicate.exact_group_ids,
        len(corpus.label_ids),
    )
    print("fitting shared-text-included TF-IDF ablation", flush=True)
    shared_tfidf = fit_tfidf(corpus.primary_docs)
    print("scoring shared-text-included ablation", flush=True)
    shared_profiles = score_leave_group_out(
        corpus.primary_dense,
        shared_tfidf,
        corpus.song_label_index,
        duplicate.strict_group_ids,
        len(corpus.label_ids),
    )

    strict_dense_z = zscore_rows(strict_profiles.dense)
    strict_lexical_z = zscore_rows(strict_profiles.lexical)
    exact_dense_z = zscore_rows(exact_profiles.dense)
    exact_lexical_z = zscore_rows(exact_profiles.lexical)
    shared_dense_z = zscore_rows(shared_profiles.dense)
    shared_lexical_z = zscore_rows(shared_profiles.lexical)

    system_scores = {
        SYSTEM_BGE: strict_profiles.dense,
        SYSTEM_TFIDF: strict_profiles.lexical,
        SYSTEM_FUSION: 0.5 * strict_dense_z + 0.5 * strict_lexical_z,
        SYSTEM_RAW: 0.5 * strict_profiles.dense + 0.5 * strict_profiles.lexical,
        SYSTEM_EXACT_ONLY: 0.5 * exact_dense_z + 0.5 * exact_lexical_z,
        SYSTEM_SHARED: 0.5 * shared_dense_z + 0.5 * shared_lexical_z,
    }
    ranks: dict[str, np.ndarray] = {}
    top_labels: dict[str, np.ndarray] = {}
    metric_arrays: dict[str, dict[str, np.ndarray]] = {}
    for system, scores in system_scores.items():
        system_ranks, system_top = rank_system(scores, corpus.song_label_index)
        ranks[system] = system_ranks
        top_labels[system] = system_top
        metric_arrays[system] = metrics_from_ranks(system_ranks)
    return Evaluation(
        ranks=ranks,
        top_labels=top_labels,
        metric_arrays=metric_arrays,
        strict_profiles=strict_profiles,
        exact_only_profiles=exact_profiles,
        shared_profiles=shared_profiles,
        strict_tfidf_features=strict_tfidf.shape[1],
        shared_tfidf_features=shared_tfidf.shape[1],
    )


def component_metric_values(
    evaluation: Evaluation,
    labels: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    system_count = len(SYSTEMS)
    metric_count = len(METRICS)
    label_count = int(labels.max()) + 1
    components: list[np.ndarray] = []
    label_group_counts = np.zeros(label_count, dtype=np.int32)
    for label in range(label_count):
        label_indices = np.flatnonzero(labels == label)
        groups = sorted(set(int(group_ids[index]) for index in label_indices))
        label_group_counts[label] = len(groups)
        group_values = np.zeros((len(groups), system_count, metric_count), dtype=np.float64)
        for group_offset, group in enumerate(groups):
            members = np.asarray([index for index in label_indices if int(group_ids[index]) == group], dtype=np.int64)
            for system_offset, system in enumerate(SYSTEMS):
                for metric_offset, metric in enumerate(METRICS):
                    group_values[group_offset, system_offset, metric_offset] = float(
                        np.mean(evaluation.metric_arrays[system][metric][members])
                    )
        components.append(group_values)
    return components, label_group_counts


@dataclass
class BootstrapResult:
    point_macro: np.ndarray
    replicates: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    diagnostics: dict[str, Any]


def run_bootstrap(components: list[np.ndarray]) -> BootstrapResult:
    """Literal paired two-stage cluster bootstrap.

    Each replicate first draws ``label_count`` outer label occurrences with
    replacement. Every occurrence then receives its own independent within-
    label component resample, including when the same label appears more than
    once in the outer draw. The selected component indices are applied to the
    full system-by-metric tensor, preserving all paired comparisons.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    label_count = len(components)
    point_label = np.zeros((label_count, len(SYSTEMS), len(METRICS)), dtype=np.float64)
    for label, values in enumerate(components):
        point_label[label] = values.mean(axis=0)
    replicates = np.zeros((BOOTSTRAP_REPLICATES, len(SYSTEMS), len(METRICS)), dtype=np.float64)
    outer_occurrences = 0
    repeated_outer_occurrences = 0
    inner_resample_occurrences = 0
    inner_component_draws = 0
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_labels = rng.integers(0, label_count, size=label_count)
        label_occurrence_counts = Counter(int(label) for label in sampled_labels.tolist())
        outer_occurrences += label_count
        repeated_outer_occurrences += label_count - len(label_occurrence_counts)
        replicate_total = np.zeros((len(SYSTEMS), len(METRICS)), dtype=np.float64)
        for label, occurrence_count in label_occurrence_counts.items():
            values = components[label]
            # Each row is a distinct outer-label occurrence. Repeated outer
            # selections therefore do not reuse one precomputed inner mean.
            draws = rng.integers(0, len(values), size=(occurrence_count, len(values)))
            occurrence_means = values[draws].mean(axis=1)
            replicate_total += occurrence_means.sum(axis=0)
            inner_resample_occurrences += occurrence_count
            inner_component_draws += occurrence_count * len(values)
        replicates[replicate] = replicate_total / label_count
        if (replicate + 1) % 500 == 0:
            print(f"completed {replicate + 1}/{BOOTSTRAP_REPLICATES} occurrence-wise bootstraps", flush=True)
    point_macro = point_label.mean(axis=0)
    lower = np.quantile(replicates, 0.025, axis=0)
    upper = np.quantile(replicates, 0.975, axis=0)
    diagnostics = {
        "algorithm": "literal_occurrence_wise_two_stage_paired_bootstrap_v1",
        "replicates": BOOTSTRAP_REPLICATES,
        "outer_label_occurrences_per_replicate": label_count,
        "total_outer_label_occurrences": outer_occurrences,
        "repeated_outer_label_occurrences": repeated_outer_occurrences,
        "independent_inner_resample_occurrences": inner_resample_occurrences,
        "total_inner_component_draws": inner_component_draws,
        "same_component_draw_applied_to_all_systems_and_metrics": True,
        "repeated_outer_occurrences_reuse_inner_draw": False,
    }
    return BootstrapResult(
        point_macro=point_macro,
        replicates=replicates,
        lower=lower,
        upper=upper,
        diagnostics=diagnostics,
    )


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")])


def metric(value: float) -> str:
    return f"{float(value):.6f}"


def build_public_rows(
    corpus: CorpusData,
    duplicate: DuplicateAudit,
    evaluation: Evaluation,
    components: list[np.ndarray],
    label_group_counts: np.ndarray,
    bootstrap: BootstrapResult,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    metrics_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    system_index = {system: index for index, system in enumerate(SYSTEMS)}
    metric_index = {name: index for index, name in enumerate(METRICS)}
    label_stratum_component_units = int(label_group_counts.sum())

    for system in SYSTEMS:
        si = system_index[system]
        role = "primary_comparison" if system in PRIMARY_SYSTEMS else "diagnostic_ablation"
        for name in METRICS:
            mi = metric_index[name]
            metrics_rows.append(
                {
                    "system": system,
                    "task_role": role,
                    "aggregation": "source_credit_label_macro_duplicate_group_adjusted",
                    "metric": name,
                    "estimate": metric(bootstrap.point_macro[si, mi]),
                    "ci95_lower": metric(bootstrap.lower[si, mi]),
                    "ci95_upper": metric(bootstrap.upper[si, mi]),
                    "queries": len(corpus.songs),
                    "source_credit_labels": len(corpus.label_ids),
                    "global_duplicate_components": duplicate.strict_groups,
                    "label_stratum_component_units": label_stratum_component_units,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
            micro = float(np.mean(evaluation.metric_arrays[system][name]))
            metrics_rows.append(
                {
                    "system": system,
                    "task_role": role,
                    "aggregation": "query_micro_descriptive",
                    "metric": name,
                    "estimate": metric(micro),
                    "ci95_lower": "",
                    "ci95_upper": "",
                    "queries": len(corpus.songs),
                    "source_credit_labels": len(corpus.label_ids),
                    "global_duplicate_components": duplicate.strict_groups,
                    "label_stratum_component_units": label_stratum_component_units,
                    "bootstrap_replicates": "",
                }
            )

        ranks = evaluation.ranks[system]
        top = evaluation.top_labels[system]
        coverage_rows.append(
            {
                "system": system,
                "task_role": role,
                "labels_with_any_correct_at_1": sum(
                    np.any(ranks[corpus.song_label_index == label] <= 1) for label in range(len(corpus.label_ids))
                ),
                "labels_with_any_correct_at_5": sum(
                    np.any(ranks[corpus.song_label_index == label] <= 5) for label in range(len(corpus.label_ids))
                ),
                "labels_with_any_correct_at_10": sum(
                    np.any(ranks[corpus.song_label_index == label] <= 10) for label in range(len(corpus.label_ids))
                ),
                "prediction_labels_covered_at_1": len(set(top[:, 0].tolist())),
                "prediction_labels_covered_at_5": len(set(top[:, :5].ravel().tolist())),
                "prediction_labels_covered_at_10": len(set(top[:, :10].ravel().tolist())),
                "eligible_source_credit_labels": len(corpus.label_ids),
            }
        )
        coverage_rows[-1].update(
            {
                "fraction_labels_with_any_correct_at_1": metric(
                    coverage_rows[-1]["labels_with_any_correct_at_1"] / len(corpus.label_ids)
                ),
                "fraction_labels_with_any_correct_at_5": metric(
                    coverage_rows[-1]["labels_with_any_correct_at_5"] / len(corpus.label_ids)
                ),
                "fraction_labels_with_any_correct_at_10": metric(
                    coverage_rows[-1]["labels_with_any_correct_at_10"] / len(corpus.label_ids)
                ),
                "fraction_prediction_labels_covered_at_1": metric(
                    coverage_rows[-1]["prediction_labels_covered_at_1"] / len(corpus.label_ids)
                ),
                "fraction_prediction_labels_covered_at_5": metric(
                    coverage_rows[-1]["prediction_labels_covered_at_5"] / len(corpus.label_ids)
                ),
                "fraction_prediction_labels_covered_at_10": metric(
                    coverage_rows[-1]["prediction_labels_covered_at_10"] / len(corpus.label_ids)
                ),
            }
        )

        for label in range(len(corpus.label_ids)):
            indices = np.flatnonzero(corpus.song_label_index == label)
            counts = Counter((int(duplicate.strict_group_ids[index]), label) for index in indices)
            weights = np.asarray(
                [1.0 / counts[(int(duplicate.strict_group_ids[index]), label)] for index in indices], dtype=np.float64
            )
            values = evaluation.metric_arrays[system]
            top1 = evaluation.top_labels[system][indices, 0]
            wrong = top1 != label
            wrong_weights: Counter[int] = Counter()
            for predicted, weight, is_wrong in zip(top1.tolist(), weights.tolist(), wrong.tolist()):
                if is_wrong:
                    wrong_weights[int(predicted)] += float(weight)
            if wrong_weights:
                most_common_wrong, wrong_mass = sorted(wrong_weights.items(), key=lambda item: (-item[1], item[0]))[0]
                total_wrong = sum(wrong_weights.values())
                common_wrong_name = corpus.label_names[most_common_wrong]
                common_wrong_share = wrong_mass / total_wrong
            else:
                common_wrong_name = ""
                common_wrong_share = 0.0
            per_label_rows.append(
                {
                    "system": system,
                    "task_role": role,
                    "source_credit_label": corpus.label_names[label],
                    "queries": len(indices),
                    "within_label_component_units": int(label_group_counts[label]),
                    "mrr": metric(np.average(values["mrr"][indices], weights=weights)),
                    "recall_at_1": metric(np.average(values["recall_at_1"][indices], weights=weights)),
                    "recall_at_5": metric(np.average(values["recall_at_5"][indices], weights=weights)),
                    "recall_at_10": metric(np.average(values["recall_at_10"][indices], weights=weights)),
                    "ndcg_at_10": metric(np.average(values["ndcg_at_10"][indices], weights=weights)),
                    "median_true_label_rank": metric(weighted_median(evaluation.ranks[system][indices], weights)),
                    "most_common_incorrect_top1_source_credit_label": common_wrong_name,
                    "share_of_incorrect_group_weight_for_common_top1": metric(common_wrong_share),
                }
            )

    comparisons = (
        (SYSTEM_FUSION, SYSTEM_BGE, "strict fusion minus strict BGE-M3"),
        (SYSTEM_FUSION, SYSTEM_TFIDF, "strict fusion minus strict TF-IDF"),
        (SYSTEM_FUSION, SYSTEM_RAW, "standardized fusion minus raw-score fusion"),
        (SYSTEM_FUSION, SYSTEM_EXACT_ONLY, "strict near-duplicate guard minus exact-only guard"),
        (SYSTEM_FUSION, SYSTEM_SHARED, "strict shared-text exclusion minus shared-text-included ablation"),
    )
    uncertainty_rows: list[dict[str, Any]] = []
    for left, right, comparison in comparisons:
        li, ri = system_index[left], system_index[right]
        for name in METRICS:
            mi = metric_index[name]
            replicate_delta = bootstrap.replicates[:, li, mi] - bootstrap.replicates[:, ri, mi]
            delta_lower = float(np.quantile(replicate_delta, 0.025))
            delta_upper = float(np.quantile(replicate_delta, 0.975))
            uncertainty_rows.append(
                {
                    "comparison": comparison,
                    "metric": name,
                    "estimate_delta": metric(bootstrap.point_macro[li, mi] - bootstrap.point_macro[ri, mi]),
                    "ci95_lower": metric(delta_lower),
                    "ci95_upper": metric(delta_upper),
                    "interval_direction": (
                        "left_higher" if delta_lower > 0 else "left_lower" if delta_upper < 0 else "interval_includes_zero"
                    ),
                    "paired_two_stage_bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )

    fusion_rows = [row for row in per_label_rows if row["system"] == SYSTEM_FUSION]
    for row in fusion_rows:
        row["mrr_numeric"] = float(row["mrr"])
        matching_tfidf = next(
            item for item in per_label_rows if item["system"] == SYSTEM_TFIDF and item["source_credit_label"] == row["source_credit_label"]
        )
        matching_bge = next(
            item for item in per_label_rows if item["system"] == SYSTEM_BGE and item["source_credit_label"] == row["source_credit_label"]
        )
        row["lift_numeric"] = float(row["mrr"]) - max(float(matching_tfidf["mrr"]), float(matching_bge["mrr"]))
    selections = (
        ("highest_strict_fusion_mrr", sorted(fusion_rows, key=lambda row: (-row["mrr_numeric"], row["source_credit_label"]))[:10]),
        ("lowest_strict_fusion_mrr", sorted(fusion_rows, key=lambda row: (row["mrr_numeric"], row["source_credit_label"]))[:10]),
        ("largest_fusion_lift_over_best_single_system", sorted(fusion_rows, key=lambda row: (-row["lift_numeric"], row["source_credit_label"]))[:10]),
    )
    example_rows: list[dict[str, Any]] = []
    for category, selected in selections:
        for rank, row in enumerate(selected, start=1):
            example_rows.append(
                {
                    "aggregate_example_category": category,
                    "category_rank": rank,
                    "source_credit_label": row["source_credit_label"],
                    "queries": row["queries"],
                    "within_label_component_units": row["within_label_component_units"],
                    "strict_fusion_mrr": row["mrr"],
                    "strict_fusion_recall_at_1": row["recall_at_1"],
                    "strict_fusion_recall_at_5": row["recall_at_5"],
                    "strict_fusion_recall_at_10": row["recall_at_10"],
                    "median_true_label_rank": row["median_true_label_rank"],
                    "fusion_mrr_lift_over_better_single_system": metric(row["lift_numeric"]),
                    "most_common_incorrect_top1_source_credit_label": row[
                        "most_common_incorrect_top1_source_credit_label"
                    ],
                    "share_of_incorrect_group_weight_for_common_top1": row[
                        "share_of_incorrect_group_weight_for_common_top1"
                    ],
                }
            )
    for row in fusion_rows:
        row.pop("mrr_numeric", None)
        row.pop("lift_numeric", None)
    return metrics_rows, coverage_rows, uncertainty_rows, per_label_rows, example_rows


def build_private_rows(
    corpus: CorpusData,
    duplicate: DuplicateAudit,
    evaluation: Evaluation,
    label_group_counts: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strict_group_sizes = Counter(duplicate.strict_group_ids.tolist())
    exact_group_sizes = Counter(duplicate.exact_group_ids.tolist())
    query_rows: list[dict[str, Any]] = []
    for index, song in enumerate(corpus.songs):
        row: dict[str, Any] = {
            "private_query_id": f"RETQ-{index + 1:05d}",
            "song_id": song,
            "true_label_id": corpus.label_ids[int(corpus.song_label_index[index])],
            "source_credit_label": corpus.label_names[int(corpus.song_label_index[index])],
            "strict_duplicate_group_private_id": f"SDG-{int(duplicate.strict_group_ids[index]) + 1:05d}",
            "strict_duplicate_group_size": strict_group_sizes[int(duplicate.strict_group_ids[index])],
            "exact_duplicate_group_size": exact_group_sizes[int(duplicate.exact_group_ids[index])],
            "effective_characters": len(corpus.normalized_strict_docs[index]),
            "strict_chunk_count": int(corpus.strict_chunk_counts[index]),
            "shared_text_chunks_removed": int(corpus.primary_chunk_counts[index] - corpus.strict_chunk_counts[index]),
        }
        for system in SYSTEMS:
            short = {
                SYSTEM_BGE: "bge_strict",
                SYSTEM_TFIDF: "tfidf_strict",
                SYSTEM_FUSION: "fusion_strict",
                SYSTEM_RAW: "raw_fusion_ablation",
                SYSTEM_EXACT_ONLY: "exact_only_ablation",
                SYSTEM_SHARED: "shared_text_ablation",
            }[system]
            rank = int(evaluation.ranks[system][index])
            predicted = int(evaluation.top_labels[system][index, 0])
            row[f"{short}_true_label_rank"] = rank
            row[f"{short}_top1_label_id"] = corpus.label_ids[predicted]
        query_rows.append(row)

    profile_rows: list[dict[str, Any]] = []
    for label, (identifier, name) in enumerate(zip(corpus.label_ids, corpus.label_names)):
        indices = np.flatnonzero(corpus.song_label_index == label)
        profile_rows.append(
            {
                "label_id": identifier,
                "source_credit_label": name,
                "query_songs": len(indices),
                "strict_within_label_component_units": int(label_group_counts[label]),
                "minimum_training_groups_after_strict_holdout": int(label_group_counts[label] - 1),
                "strict_chunks": int(corpus.strict_chunk_counts[indices].sum()),
                "shared_text_chunks_removed": int(
                    (corpus.primary_chunk_counts[indices] - corpus.strict_chunk_counts[indices]).sum()
                ),
            }
        )
    return query_rows, profile_rows


def build_documents(
    corpus: CorpusData,
    duplicate: DuplicateAudit,
    evaluation: Evaluation,
    benchmark: dict[str, Any],
    headline: dict[str, Any],
) -> tuple[str, str]:
    readme = f"""# Explainable lyrical-repertoire retrieval

This release evaluates whether a held-out song's cleaned lyrical representation retrieves its corpus source-credit label's remaining lyrical repertoire. It compares BGE-M3 dense similarity, character 2–5 gram TF-IDF, and an equal-weight per-query z-score fusion.

## Read first

- `metrics.csv` contains macro and micro MRR, Recall@1/5/10, and nDCG@10.
- `uncertainty.csv` contains paired literal occurrence-wise two-stage bootstrap intervals for system differences.
- `coverage.csv` reports ground-truth-label and prediction-label coverage.
- `per_label_metrics.csv` contains aggregate source-credit-label results.
- `label_level_examples.csv` contains only aggregate label-level examples; it contains no lyrics or song/chunk identifiers.
- `METHOD.md` defines the leakage controls and claim boundary.

Headline strict-fusion macro MRR: **{headline['fusion_macro_mrr']:.3f}** (95% bootstrap CI {headline['fusion_macro_mrr_ci'][0]:.3f}–{headline['fusion_macro_mrr_ci'][1]:.3f}).

Population: {len(corpus.songs):,} length-qualified held-out song queries and {len(corpus.label_ids)} eligible source-credit labels. Exact/near-duplicate grouping yields {duplicate.strict_groups:,} global components, which map to {int(evaluation.strict_profiles.label_group_counts.sum()):,} within-label component units for macro estimation and bootstrap resampling.

## Claim boundary

{CLAIM_BOUNDARY}
"""

    benchmark_models = {row["model"]: row for row in benchmark["models"]}
    method = f"""# Method and limitations

## Research question

For each song, can its cleaned lyric-based representation retrieve the corpus source-credit label attached to that song when the entire song—and every detected exact or near-duplicate variant—is removed from every candidate label repertoire?

The outcome is **corpus source-credit-label lyrical similarity**. It is not an identity classifier and does not imply artist style, biography, influence, collaboration, genre, social relation, or cultural causation.

## Why these three systems

BGE-M3 supplies a multilingual dense representation, but it was not assumed to be superior. The preceding low-overlap same-song continuation benchmark evaluated 1,000 queries and found MRR {float(benchmark_models['BGE-M3 dense']['mrr']):.3f} for BGE-M3, {float(benchmark_models['character 2-5 gram TF-IDF']['mrr']):.3f} for character TF-IDF, and {float(benchmark_models['equal-weight z-score fusion']['mrr']):.3f} for fusion. That benchmark motivated testing complementary dense and character-form evidence here; it is not reused as downstream gold data.

## Evaluation population

- Candidate labels: the 204 graph-eligible corpus source-credit labels.
- Query unit: one whole song assembled from its eligible cleaned chunks after exact cross-label shared-text exclusion.
- Minimum query length: {MIN_EFFECTIVE_CHARACTERS} NFKC-normalized alphanumeric characters.
- Queries: {len(corpus.songs):,} songs.
- Global exact/near-duplicate components: {duplicate.strict_groups:,}.
- Within-label component units: {int(evaluation.strict_profiles.label_group_counts.sum()):,}. A cross-label global component contributes one unit to each label stratum it intersects, so the within-label total can exceed the global total.
- Each query has one corpus source-credit label in this fixed analysis population.

## Representations

For BGE-M3, eligible cleaned chunk vectors are averaged within a song using the frozen comparison-text weights and then L2-normalized. No new model fitting occurs. For the lexical system, whole-song cleaned text is represented by character 2–5 gram TF-IDF (`min_df=3`, sublinear term frequency, L2 normalization, maximum {TFIDF_MAX_FEATURES:,} features). The strict matrix retained {evaluation.strict_tfidf_features:,} features.

Each candidate label profile is an equal-component mean of its training-song representations. Songs in the same exact/near-duplicate component receive weights summing to one within a label, preventing repeated variants from dominating the profile.

## Strict held-out and leakage controls

1. **Whole-song holdout:** for every query, its complete song representation is removed from its true label profile.
2. **Exact shared-text exclusion:** chunks whose exact cleaned text occurs across source-credit labels are removed before strict representations are built. This removed {corpus.shared_rows_removed_all_eligible:,} eligible membership rows overall and {corpus.shared_rows_removed_query_population:,} rows in the evaluated song population; zero exact cleaned-text hashes remain shared across strict labels. Of {corpus.primary_songs_all_eligible:,} otherwise eligible songs, {corpus.songs_lost_entirely_to_shared_text_exclusion:,} had no non-shared chunk remaining and therefore cannot enter the strict query population.
3. **Exact and near-duplicate holdout:** normalized whole-song text is converted to character-trigram sets. A complete prefix-filtered all-pairs join finds every pair with Jaccard similarity at least {NEAR_DUPLICATE_JACCARD:.2f}; full sets verify candidates. All connected variants are held out together from every affected label profile. The audit found {duplicate.exact_pairs} exact pair and {duplicate.nonexact_near_pairs} additional near-duplicate pairs, including {duplicate.cross_label_pairs} cross-label pairs.
4. **Duplicate-adjusted aggregation:** every duplicate component contributes total weight one within a label in both profiles and macro evaluation.
5. **No query rows in public outputs:** song/chunk identifiers, pair evidence, and per-query ranks remain under the private audit directory. No lyric text or vectors are copied into either output directory.

TF-IDF vocabulary and IDF are estimated on the fixed unlabeled evaluation corpus. Candidate source-credit labels are not used during TF-IDF fitting; the label profiles themselves are strictly group-held-out. This is a corpus-internal transductive retrieval design, not an estimate for unseen future corpora.

## Fusion and ablations

Candidate-label scores are standardized separately within each query for BGE-M3 and TF-IDF, then averaged with equal weight. No fusion weight is tuned on the evaluation outcomes.

Three diagnostics are reported: averaging unstandardized cosine scores, removing the non-exact near-duplicate guard while retaining exact duplicate holdout, and rebuilding representations with cross-label shared text included. The shared-text diagnostic holds the strict {len(corpus.songs):,}-song query population fixed; it does not reintroduce the {corpus.songs_lost_entirely_to_shared_text_exclusion:,} songs that consist only of cross-label shared text. Only the systems explicitly marked `(strict)` are primary results.

## Metrics and uncertainty

With one relevant label per query, MRR is reciprocal true-label rank; Recall@k is whether the true label occurs in the first k candidates; nDCG@10 is `1/log2(rank+1)` for ranks at most 10 and zero otherwise. Query-micro values are descriptive. Primary inference uses source-credit-label macro means after duplicate-component adjustment.

Uncertainty uses {BOOTSTRAP_REPLICATES:,} fixed-seed, literal occurrence-wise paired two-stage replicates. For each replicate, exactly {len(corpus.label_ids)} outer source-credit-label occurrences are sampled with replacement. For **every outer occurrence**, its within-label component units are independently sampled with replacement, using the original stratum size—even when the same label appears multiple times in the outer draw. One inner component-index draw is applied jointly to the complete system-by-metric tensor, preserving query/component pairing for system differences. The {len(corpus.label_ids)} occurrence means are averaged; percentile 2.5% and 97.5% endpoints form the reported 95% intervals. These intervals quantify resampling variability within the fixed corpus strata, not uncertainty for an external population of artists or songs.

## Interpretation limits

- Source-credit strings are corpus labels, not externally verified identity records.
- Correct retrieval shows consistency with a label's remaining lyric corpus, not authorship verification.
- The target label comes from corpus provenance rather than an independently annotated semantic gold standard.
- Strict leakage control changes the estimand: {corpus.songs_lost_entirely_to_shared_text_exclusion:,} otherwise eligible songs with no label-specific text are excluded, followed by {corpus.short_songs_excluded:,} songs below the minimum effective length.
- Dense similarity remains difficult to explain at the individual token level; TF-IDF and the aggregate error summaries supply partial, not complete, interpretability.
- Small repertoires have wider effective uncertainty; the smallest strict profile has {evaluation.strict_profiles.minimum_training_groups_after_holdout} independent training components after holdout.
- Chinese rap performance, delivery, beat, timing, and audio rhyme are outside this text-only task.

## Claim boundary

{CLAIM_BOUNDARY}
"""
    return readme, method


def build() -> None:
    generated_at = utc_now()
    lineage = validate_source_contracts()
    corpus = load_corpus()
    print(f"loaded {len(corpus.songs)} query songs across {len(corpus.label_ids)} labels", flush=True)
    duplicate = detect_duplicate_groups(corpus)
    print(
        f"verified {len(duplicate.pair_rows)} exact/near pairs; {duplicate.strict_groups} global components",
        flush=True,
    )
    evaluation = evaluate_models(corpus, duplicate)
    components, label_group_counts = component_metric_values(
        evaluation, corpus.song_label_index, duplicate.strict_group_ids
    )
    bootstrap = run_bootstrap(components)
    metrics_rows, coverage_rows, uncertainty_rows, per_label_rows, example_rows = build_public_rows(
        corpus, duplicate, evaluation, components, label_group_counts, bootstrap
    )
    query_rows, profile_rows = build_private_rows(corpus, duplicate, evaluation, label_group_counts)

    system_index = {system: index for index, system in enumerate(SYSTEMS)}
    metric_index = {name: index for index, name in enumerate(METRICS)}
    fi = system_index[SYSTEM_FUSION]
    mi = metric_index["mrr"]
    headline = {
        "fusion_macro_mrr": float(bootstrap.point_macro[fi, mi]),
        "fusion_macro_mrr_ci": [float(bootstrap.lower[fi, mi]), float(bootstrap.upper[fi, mi])],
    }
    uncertainty_by_comparison: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in uncertainty_rows:
        uncertainty_by_comparison[row["comparison"]].append(row)

    def directional_metrics(comparison: str, direction: str) -> list[str]:
        return [row["metric"] for row in uncertainty_by_comparison[comparison] if row["interval_direction"] == direction]

    benchmark = json.loads((BENCHMARK_DIR / "analysis_summary.json").read_text(encoding="utf-8"))
    readme, method_document = build_documents(corpus, duplicate, evaluation, benchmark, headline)

    def reset_build_directory(path: Path, expected_name: str) -> None:
        resolved_root = ROOT.resolve()
        resolved_path = path.resolve()
        if resolved_path.name != expected_name or not resolved_path.is_relative_to(resolved_root):
            raise RuntimeError(f"Refusing to reset unexpected build directory: {resolved_path}")
        if resolved_path.exists():
            shutil.rmtree(resolved_path)

    reset_build_directory(OUT_DIR, ARTIFACT_ID)
    reset_build_directory(PRIVATE_OUT_DIR, f"private-{ARTIFACT_ID}")
    OUT_DIR.mkdir(parents=True)
    PRIVATE_OUT_DIR.mkdir(parents=True)

    atomic_write_text(OUT_DIR / "README.md", readme)
    atomic_write_text(OUT_DIR / "METHOD.md", method_document)
    atomic_write_csv(
        OUT_DIR / "metrics.csv",
        metrics_rows,
        [
            "system", "task_role", "aggregation", "metric", "estimate", "ci95_lower", "ci95_upper",
            "queries", "source_credit_labels", "global_duplicate_components",
            "label_stratum_component_units", "bootstrap_replicates",
        ],
    )
    atomic_write_csv(
        OUT_DIR / "coverage.csv",
        coverage_rows,
        [
            "system", "task_role", "labels_with_any_correct_at_1", "labels_with_any_correct_at_5",
            "labels_with_any_correct_at_10", "fraction_labels_with_any_correct_at_1",
            "fraction_labels_with_any_correct_at_5", "fraction_labels_with_any_correct_at_10",
            "prediction_labels_covered_at_1", "prediction_labels_covered_at_5",
            "prediction_labels_covered_at_10", "fraction_prediction_labels_covered_at_1",
            "fraction_prediction_labels_covered_at_5", "fraction_prediction_labels_covered_at_10",
            "eligible_source_credit_labels",
        ],
    )
    atomic_write_csv(
        OUT_DIR / "uncertainty.csv",
        uncertainty_rows,
        [
            "comparison", "metric", "estimate_delta", "ci95_lower", "ci95_upper", "interval_direction",
            "paired_two_stage_bootstrap_replicates",
        ],
    )
    atomic_write_csv(
        OUT_DIR / "per_label_metrics.csv",
        per_label_rows,
        [
            "system", "task_role", "source_credit_label", "queries", "within_label_component_units",
            "mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10",
            "median_true_label_rank", "most_common_incorrect_top1_source_credit_label",
            "share_of_incorrect_group_weight_for_common_top1",
        ],
    )
    atomic_write_csv(
        OUT_DIR / "label_level_examples.csv",
        example_rows,
        [
            "aggregate_example_category", "category_rank", "source_credit_label", "queries",
            "within_label_component_units", "strict_fusion_mrr", "strict_fusion_recall_at_1",
            "strict_fusion_recall_at_5", "strict_fusion_recall_at_10", "median_true_label_rank",
            "fusion_mrr_lift_over_better_single_system", "most_common_incorrect_top1_source_credit_label",
            "share_of_incorrect_group_weight_for_common_top1",
        ],
    )

    analysis_summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": generated_at,
        "task": "strict held-out-song corpus source-credit-label lyrical-repertoire retrieval",
        "claim_boundary": CLAIM_BOUNDARY,
        "population": {
            "eligible_source_credit_labels": len(corpus.label_ids),
            "primary_songs_before_shared_text_exclusion": corpus.primary_songs_all_eligible,
            "strict_songs_before_length_filter": corpus.strict_songs_before_length_filter,
            "songs_lost_entirely_to_shared_text_exclusion": corpus.songs_lost_entirely_to_shared_text_exclusion,
            "length_qualified_song_queries": len(corpus.songs),
            "minimum_effective_characters": MIN_EFFECTIVE_CHARACTERS,
            "short_songs_excluded": corpus.short_songs_excluded,
            "global_strict_duplicate_components": duplicate.strict_groups,
            "strict_label_stratum_component_units": int(label_group_counts.sum()),
            "minimum_label_query_groups": int(label_group_counts.min()),
            "median_label_query_groups": float(np.median(label_group_counts)),
            "maximum_label_query_groups": int(label_group_counts.max()),
            "minimum_training_groups_after_holdout": evaluation.strict_profiles.minimum_training_groups_after_holdout,
        },
        "leakage_controls": {
            "strict_membership_rows": corpus.strict_population_rows,
            "primary_membership_rows": corpus.primary_population_rows,
            "shared_membership_rows_removed_all_eligible": corpus.shared_rows_removed_all_eligible,
            "shared_membership_rows_removed_evaluation_population": corpus.shared_rows_removed_query_population,
            "cross_label_exact_text_hashes_after_strict_filter": corpus.source_cross_label_shared_hashes_after_strict_filter,
            "near_duplicate_threshold_character_trigram_jaccard": NEAR_DUPLICATE_JACCARD,
            "duplicate_candidate_pairs_verified": duplicate.candidate_pairs,
            "exact_duplicate_pairs": duplicate.exact_pairs,
            "nonexact_near_duplicate_pairs": duplicate.nonexact_near_pairs,
            "cross_label_exact_or_near_duplicate_pairs": duplicate.cross_label_pairs,
            "exact_only_groups": duplicate.exact_groups,
            "strict_exact_and_near_groups": duplicate.strict_groups,
        },
        "representations": {
            "bge_m3_dimensions": 1024,
            "strict_tfidf_features": evaluation.strict_tfidf_features,
            "shared_text_ablation_tfidf_features": evaluation.shared_tfidf_features,
            "fusion": "equal-weight per-query candidate-label z-score average; no tuned weight",
        },
        "headline": headline,
        "evidence_interpretation": {
            "primary_fusion_comparison": {
                "finding": "Fusion is higher than each strict single representation for every requested metric under paired intervals.",
                "fusion_higher_than_bge_metrics": directional_metrics(
                    "strict fusion minus strict BGE-M3", "left_higher"
                ),
                "fusion_higher_than_tfidf_metrics": directional_metrics(
                    "strict fusion minus strict TF-IDF", "left_higher"
                ),
            },
            "score_standardization_ablation": {
                "finding": "Standardization has a small positive advantage for some metrics; unsupported metrics remain explicitly inconclusive.",
                "standardized_higher_metrics": directional_metrics(
                    "standardized fusion minus raw-score fusion", "left_higher"
                ),
                "interval_includes_zero_metrics": directional_metrics(
                    "standardized fusion minus raw-score fusion", "interval_includes_zero"
                ),
            },
            "near_duplicate_guard_ablation": {
                "finding": "Removing the near-duplicate guard creates a small optimistic advantage where strict-minus-ablation intervals are below zero; this supports retaining the guard rather than treating the ablation as a preferred model.",
                "strict_lower_metrics": directional_metrics(
                    "strict near-duplicate guard minus exact-only guard", "left_lower"
                ),
                "interval_includes_zero_metrics": directional_metrics(
                    "strict near-duplicate guard minus exact-only guard", "interval_includes_zero"
                ),
            },
            "shared_text_guard_ablation": {
                "finding": "Including cross-label shared text creates a small optimistic advantage where strict-minus-ablation intervals are below zero; the strict leakage-controlled result remains primary.",
                "strict_lower_metrics": directional_metrics(
                    "strict shared-text exclusion minus shared-text-included ablation", "left_lower"
                ),
                "interval_includes_zero_metrics": directional_metrics(
                    "strict shared-text exclusion minus shared-text-included ablation", "interval_includes_zero"
                ),
            },
            "claim_rule": "A diagnostic performance increase after relaxing a leakage guard is reported as removed optimism, not as evidence that the guard should be discarded.",
        },
        "primary_macro_metrics": {
            system: {
                name: {
                    "estimate": round(float(bootstrap.point_macro[system_index[system], metric_index[name]]), 6),
                    "ci95_lower": round(float(bootstrap.lower[system_index[system], metric_index[name]]), 6),
                    "ci95_upper": round(float(bootstrap.upper[system_index[system], metric_index[name]]), 6),
                }
                for name in METRICS
            }
            for system in (SYSTEM_BGE, SYSTEM_TFIDF, SYSTEM_FUSION)
        },
        "uncertainty": {
            "method": "literal occurrence-wise paired two-stage source-credit-label then within-label duplicate-component bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": RANDOM_SEED,
            "definition": (
                "Each replicate samples 204 outer label occurrences with replacement; each occurrence independently "
                "resamples its label-stratum components with replacement at the original stratum size, including "
                "repeated occurrences of the same label. One component draw is shared across every system and metric."
            ),
            "scope": "resampling variability within the fixed corpus strata; not external-population uncertainty",
            "diagnostics": bootstrap.diagnostics,
        },
        "prior_encoder_sanity_benchmark": {
            "task": benchmark["task"],
            "queries": benchmark["population"]["evaluated_queries"],
            "claim_boundary": benchmark["claim_boundary"],
        },
        "lineage": lineage,
    }
    atomic_write_json(OUT_DIR / "analysis_summary.json", analysis_summary)

    atomic_write_csv(
        PRIVATE_OUT_DIR / "query_level_audit.csv",
        query_rows,
        list(query_rows[0].keys()),
    )
    atomic_write_csv(
        PRIVATE_OUT_DIR / "duplicate_pair_audit.csv",
        duplicate.pair_rows,
        [
            "private_pair_id", "song_id_a", "source_credit_label_a", "song_id_b",
            "source_credit_label_b", "pair_type", "character_trigram_jaccard",
            "cross_source_credit_label", "effective_characters_a", "effective_characters_b",
        ],
    )
    atomic_write_csv(
        PRIVATE_OUT_DIR / "label_profile_audit.csv",
        profile_rows,
        list(profile_rows[0].keys()),
    )

    # Independent saved-artifact checks used by validation.json.
    forbidden_public_headers = {"song_id", "chunk_id", "analysis_text", "text", "embedding", "vector", "clean_row_index"}
    public_csvs = sorted(OUT_DIR.glob("*.csv"))
    bad_headers: dict[str, list[str]] = {}
    for path in public_csvs:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            headers = set(next(csv.reader(handle)))
        overlap = sorted(headers & forbidden_public_headers)
        if overlap:
            bad_headers[path.name] = overlap
    expected_per_label_rows = len(SYSTEMS) * len(corpus.label_ids)
    checks = [
        {"name": "all_input_contracts_current_and_passing", "passed": True},
        {
            "name": "strict_song_held_out_population_complete",
            "passed": len(query_rows) == len(corpus.songs)
            and len(corpus.songs) == len(set(corpus.songs))
            and corpus.primary_songs_all_eligible
            - corpus.songs_lost_entirely_to_shared_text_exclusion
            - corpus.short_songs_excluded
            == len(corpus.songs),
            "detail": {
                "primary_songs": corpus.primary_songs_all_eligible,
                "songs_lost_entirely_to_shared_text_exclusion": corpus.songs_lost_entirely_to_shared_text_exclusion,
                "short_songs_excluded": corpus.short_songs_excluded,
                "queries": len(corpus.songs),
                "source_credit_labels": len(corpus.label_ids),
            },
        },
        {
            "name": "exact_cross_label_shared_text_removed",
            "passed": corpus.source_cross_label_shared_hashes_after_strict_filter == 0,
            "detail": {"shared_rows_removed": corpus.shared_rows_removed_all_eligible},
        },
        {
            "name": "exact_and_near_duplicate_groups_held_out_and_weighted_once",
            "passed": duplicate.strict_groups <= len(corpus.songs)
            and np.array_equal(label_group_counts, evaluation.strict_profiles.label_group_counts)
            and evaluation.strict_profiles.minimum_training_groups_after_holdout >= 1,
            "detail": {
                "pairs": len(duplicate.pair_rows),
                "global_duplicate_components": duplicate.strict_groups,
                "label_stratum_component_units": int(label_group_counts.sum()),
                "minimum_training_groups": evaluation.strict_profiles.minimum_training_groups_after_holdout,
            },
        },
        {
            "name": "bootstrap_is_literal_occurrence_wise_two_stage_and_paired",
            "passed": BOOTSTRAP_REPLICATES >= 2_000
            and bootstrap.diagnostics["total_outer_label_occurrences"]
            == BOOTSTRAP_REPLICATES * len(corpus.label_ids)
            and bootstrap.diagnostics["independent_inner_resample_occurrences"]
            == bootstrap.diagnostics["total_outer_label_occurrences"]
            and bootstrap.diagnostics["repeated_outer_label_occurrences"] > 0
            and bootstrap.diagnostics["same_component_draw_applied_to_all_systems_and_metrics"] is True
            and bootstrap.diagnostics["repeated_outer_occurrences_reuse_inner_draw"] is False,
            "detail": bootstrap.diagnostics,
        },
        {
            "name": "all_requested_systems_and_metrics_present",
            "passed": len(metrics_rows) == len(SYSTEMS) * len(METRICS) * 2,
        },
        {
            "name": "macro_metrics_recompute_from_component_values",
            "passed": all(
                abs(float(row["estimate"]) - bootstrap.point_macro[system_index[row["system"]], metric_index[row["metric"]]])
                <= 5.1e-7
                for row in metrics_rows
                if row["aggregation"] == "source_credit_label_macro_duplicate_group_adjusted"
            ),
        },
        {
            "name": "bootstrap_intervals_are_ordered_and_bounded",
            "passed": all(
                0.0 <= float(row["ci95_lower"]) <= float(row["estimate"]) <= float(row["ci95_upper"]) <= 1.0
                for row in metrics_rows
                if row["aggregation"] == "source_credit_label_macro_duplicate_group_adjusted"
            ),
        },
        {
            "name": "per_label_metrics_complete",
            "passed": len(per_label_rows) == expected_per_label_rows,
            "detail": {"rows": len(per_label_rows), "expected": expected_per_label_rows},
        },
        {
            "name": "public_examples_are_aggregate_label_level_only",
            "passed": len(example_rows) == 30
            and all("source_credit_label" in row and not any(key in row for key in ("song_id", "chunk_id", "text")) for row in example_rows),
        },
        {
            "name": "public_csv_headers_exclude_song_chunk_text_and_vector_fields",
            "passed": not bad_headers,
            "detail": bad_headers,
        },
        {
            "name": "no_vector_artifact_copied_to_public_or_private_output",
            "passed": not list(OUT_DIR.glob("*.npy"))
            and not list(OUT_DIR.glob("*.npz"))
            and not list(PRIVATE_OUT_DIR.glob("*.npy"))
            and not list(PRIVATE_OUT_DIR.glob("*.npz")),
        },
        {
            "name": "claim_boundary_present_in_public_documents",
            "passed": CLAIM_BOUNDARY in readme and CLAIM_BOUNDARY in method_document,
        },
    ]
    validation = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": generated_at,
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "checks": checks,
    }
    atomic_write_json(OUT_DIR / "validation.json", validation)
    if validation["status"] != "pass":
        raise RuntimeError("Public artifact validation failed.")

    private_manifest = {
        "artifact_id": f"private-{ARTIFACT_ID}",
        "version": VERSION,
        "generated_at_utc": generated_at,
        "privacy": "private local audit; contains song identifiers and per-query rows but no lyric text or vectors",
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(PRIVATE_OUT_DIR.glob("*.csv"))
        },
        "lineage": lineage,
    }
    atomic_write_json(PRIVATE_OUT_DIR / "private_manifest.json", private_manifest)

    public_files = [
        "README.md", "METHOD.md", "metrics.csv", "coverage.csv", "uncertainty.csv",
        "per_label_metrics.csv", "label_level_examples.csv", "analysis_summary.json", "validation.json",
    ]
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": generated_at,
        "classification": "public aggregate-only held-out-song retrieval evaluation",
        "files": {
            name: {"sha256": sha256_file(OUT_DIR / name), "bytes": (OUT_DIR / name).stat().st_size}
            for name in public_files
        },
        "lineage": lineage,
        "private_audit_relative_path": f"work/private-{ARTIFACT_ID}",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    atomic_write_json(OUT_DIR / "manifest.json", manifest)
    print(f"wrote {OUT_DIR}", flush=True)
    print(f"wrote {PRIVATE_OUT_DIR}", flush=True)


if __name__ == "__main__":
    build()
