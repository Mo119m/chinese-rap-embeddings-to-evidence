#!/usr/bin/env python3
"""Build a grouped cross-fitted inductive TF-IDF sensitivity analysis.

This public artifact tests whether fitting the lexical vocabulary and IDF on
the complete, unlabeled evaluation corpus materially explains the retrieval
result. Exact/near-duplicate components are assigned wholly to one of six
folds. The matched-transductive comparator changes only vocabulary/IDF
exposure: profiles, held-out queries, scoring, aggregation, and resampling are
otherwise identical.

Only aggregate results are written. Song identifiers, lyric text, per-query
scores, fold assignments, label names, and vectors remain private.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

import build_chinese_rap_downstream_retrieval_v1 as base


VERSION = "1.0.0"
ARTIFACT_ID = "retrieval-inductive-sensitivity-v1"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_ROOT = REPO_ROOT.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / ARTIFACT_ID

FOLD_COUNT = 6
TFIDF_MAX_FEATURES = 150_000
BOOTSTRAP_REPLICATES = 5_000
RANDOM_SEED = 20260827

SYSTEM_BGE = "BGE-M3 dense cross-fitted"
SYSTEM_TFIDF_INDUCTIVE = "TF-IDF inductive train-only"
SYSTEM_TFIDF_TRANSDUCTIVE = "TF-IDF matched transductive"
SYSTEM_FUSION_INDUCTIVE = "fusion inductive train-only"
SYSTEM_FUSION_TRANSDUCTIVE = "fusion matched transductive"
SYSTEMS = (
    SYSTEM_BGE,
    SYSTEM_TFIDF_INDUCTIVE,
    SYSTEM_TFIDF_TRANSDUCTIVE,
    SYSTEM_FUSION_INDUCTIVE,
    SYSTEM_FUSION_TRANSDUCTIVE,
)
METRICS = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10")

CLAIM_BOUNDARY = (
    "Grouped cross-fitted held-out-song retrieval of corpus source-credit-label lyrical-repertoire "
    "similarity. The comparison isolates lexical vocabulary/IDF exposure within this fixed corpus; "
    "it is not artist identification, biography, influence, collaboration, genre, social relation, "
    "or external-population performance."
)

# Regression oracle from the independent read-only computation that preceded
# this implementation. It never enters model fitting or scoring.
READ_ONLY_REFERENCE = {
    SYSTEM_BGE: {
        "mrr": 0.312232,
        "recall_at_1": 0.217793,
        "recall_at_5": 0.396982,
        "recall_at_10": 0.502429,
        "ndcg_at_10": 0.345064,
    },
    SYSTEM_TFIDF_INDUCTIVE: {
        "mrr": 0.404763,
        "recall_at_1": 0.324455,
        "recall_at_5": 0.477023,
        "recall_at_10": 0.551857,
        "ndcg_at_10": 0.429369,
    },
    SYSTEM_TFIDF_TRANSDUCTIVE: {
        "mrr": 0.408726,
        "recall_at_1": 0.326963,
        "recall_at_5": 0.485930,
        "recall_at_10": 0.556526,
        "ndcg_at_10": 0.433682,
    },
    SYSTEM_FUSION_INDUCTIVE: {
        "mrr": 0.439712,
        "recall_at_1": 0.354809,
        "recall_at_5": 0.520352,
        "recall_at_10": 0.604679,
        "ndcg_at_10": 0.468971,
    },
    SYSTEM_FUSION_TRANSDUCTIVE: {
        "mrr": 0.443394,
        "recall_at_1": 0.358863,
        "recall_at_5": 0.523908,
        "recall_at_10": 0.605413,
        "ndcg_at_10": 0.471992,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_base(project_root: Path) -> None:
    """Point the frozen retrieval loader at the private project inputs."""
    base.GRAPH_DIR = project_root / "outputs" / "chinese-rap-lyrical-repertoire-graph-v2"
    base.PRIVATE_GRAPH_DIR = project_root / "work" / "private-chinese-rap-lyrical-repertoire-graph-v2"
    base.CLEAN_DIR = project_root / "work" / "private-canonical-lyric-text-sidecar-v1"
    base.EMBED_DIR = project_root / "work" / "private-canonical-clean-text-embeddings-v1"
    base.BENCHMARK_DIR = project_root / "outputs" / "chinese-rap-encoder-sanity-benchmark-v1"


def validate_private_retrieval(project_root: Path) -> tuple[Path, dict[str, Any], dict[str, str]]:
    private_dir = project_root / "work" / "private-chinese-rap-downstream-retrieval-v1"
    manifest_path = private_dir / "private_manifest.json"
    public_dir = project_root / "outputs" / "chinese-rap-downstream-retrieval-v1"
    validation_path = public_dir / "validation.json"
    public_manifest_path = public_dir / "manifest.json"
    if not manifest_path.is_file() or not validation_path.is_file() or not public_manifest_path.is_file():
        raise RuntimeError("The frozen private retrieval-v1 audit is missing.")
    if not base.validation_passes(validation_path):
        raise RuntimeError("The frozen retrieval-v1 public validation is not passing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not files or "query_level_audit.csv" not in files:
        raise RuntimeError("The private retrieval manifest lacks its query audit.")
    for name, expected in files.items():
        path = private_dir / name
        if not path.is_file():
            raise RuntimeError(f"A private retrieval payload is missing: {name}")
        if path.stat().st_size != int(expected["bytes"]) or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"A private retrieval payload differs from its manifest: {name}")
    lineage = {
        "private_retrieval_manifest_sha256": sha256_file(manifest_path),
        "source_retrieval_manifest_sha256": sha256_file(public_manifest_path),
        "source_retrieval_validation_sha256": sha256_file(validation_path),
        "private_query_audit_sha256": sha256_file(private_dir / "query_level_audit.csv"),
    }
    return private_dir / "query_level_audit.csv", manifest, lineage


@dataclass
class FrozenGroups:
    names: np.ndarray
    numeric: np.ndarray
    count: int
    within_label_units: int


def load_frozen_groups(query_path: Path, corpus: base.CorpusData) -> FrozenGroups:
    rows = read_csv(query_path)
    if len(rows) != len(corpus.songs):
        raise RuntimeError("The frozen query audit differs from the loaded query population.")
    by_song = {row["song_id"]: row for row in rows}
    if len(by_song) != len(rows) or set(by_song) != set(corpus.songs):
        raise RuntimeError("The frozen query audit does not map one-to-one to loaded songs.")

    names: list[str] = []
    for index, song in enumerate(corpus.songs):
        row = by_song[song]
        expected_label = corpus.label_ids[int(corpus.song_label_index[index])]
        if row["true_label_id"] != expected_label:
            raise RuntimeError("A frozen query label disagrees with the current corpus loader.")
        names.append(row["strict_duplicate_group_private_id"])
    if any(not name.startswith("SDG-") for name in names):
        raise RuntimeError("A frozen strict duplicate-group identifier is malformed.")

    unique_names = sorted(set(names))
    numeric_by_name = {name: index for index, name in enumerate(unique_names)}
    numeric = np.asarray([numeric_by_name[name] for name in names], dtype=np.int32)
    within_label_units = len(
        {
            (int(corpus.song_label_index[index]), names[index])
            for index in range(len(names))
        }
    )
    return FrozenGroups(
        names=np.asarray(names, dtype=object),
        numeric=numeric,
        count=len(unique_names),
        within_label_units=within_label_units,
    )


@dataclass
class FoldAssignment:
    query_folds: np.ndarray
    group_fold: dict[str, int]
    label_fold_components: np.ndarray
    fold_song_counts: np.ndarray


def stable_group_tie(group_name: str) -> str:
    return hashlib.sha256(group_name.encode("utf-8")).hexdigest()


def assign_grouped_folds(
    group_names: np.ndarray,
    labels: np.ndarray,
    label_count: int,
) -> FoldAssignment:
    members_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_names.tolist()):
        members_by_group[str(group)].append(index)

    labels_by_group = {
        group: tuple(sorted({int(labels[index]) for index in members}))
        for group, members in members_by_group.items()
    }
    label_total_components = np.zeros(label_count, dtype=np.int32)
    for affected_labels in labels_by_group.values():
        label_total_components[list(affected_labels)] += 1
    if np.any(label_total_components < FOLD_COUNT):
        raise RuntimeError("A label has fewer duplicate components than folds.")

    ordered_groups = sorted(
        members_by_group,
        key=lambda group: (
            -int(len(labels_by_group[group]) > 1),
            -len(members_by_group[group]),
            stable_group_tie(group),
        ),
    )
    label_fold_components = np.zeros((label_count, FOLD_COUNT), dtype=np.int32)
    fold_song_counts = np.zeros(FOLD_COUNT, dtype=np.int32)
    group_fold: dict[str, int] = {}
    for group in ordered_groups:
        affected = labels_by_group[group]
        chosen = min(
            range(FOLD_COUNT),
            key=lambda fold: (
                max(int(label_fold_components[label, fold]) for label in affected),
                sum(
                    float(label_fold_components[label, fold])
                    / float(label_total_components[label])
                    for label in affected
                ),
                int(fold_song_counts[fold]),
                fold,
            ),
        )
        group_fold[group] = chosen
        label_fold_components[list(affected), chosen] += 1
        fold_song_counts[chosen] += len(members_by_group[group])

    query_folds = np.asarray([group_fold[str(group)] for group in group_names], dtype=np.int8)
    if len(set(group_fold.values())) != FOLD_COUNT:
        raise RuntimeError("Not every fold received a duplicate group.")
    return FoldAssignment(
        query_folds=query_folds,
        group_fold=group_fold,
        label_fold_components=label_fold_components,
        fold_song_counts=fold_song_counts,
    )


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=3,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def validate_tfidf(matrix: sparse.csr_matrix, expected_rows: int) -> None:
    if matrix.shape[0] != expected_rows or matrix.shape[1] <= 0:
        raise RuntimeError("A TF-IDF matrix has the wrong shape.")
    row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    if not np.allclose(row_norms, 1.0, atol=2e-5):
        raise RuntimeError("TF-IDF rows are not L2 normalized.")


@dataclass
class FoldScores:
    dense: np.ndarray
    lexical: np.ndarray
    label_group_counts: np.ndarray


def score_fold_profiles(
    train_dense: np.ndarray,
    test_dense: np.ndarray,
    train_lexical: sparse.csr_matrix,
    test_lexical: sparse.csr_matrix,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    label_count: int,
) -> FoldScores:
    group_label_counts = Counter(
        (str(group), int(label))
        for group, label in zip(train_groups.tolist(), train_labels.tolist())
    )
    weights = np.asarray(
        [
            1.0 / group_label_counts[(str(group), int(label))]
            for group, label in zip(train_groups.tolist(), train_labels.tolist())
        ],
        dtype=np.float32,
    )
    label_group_counts = np.bincount(
        train_labels, weights=weights, minlength=label_count
    )
    if not np.allclose(label_group_counts, np.rint(label_group_counts), atol=1e-6):
        raise RuntimeError("Training duplicate-component weights do not sum to whole units.")

    dense_sums = np.zeros((label_count, train_dense.shape[1]), dtype=np.float32)
    np.add.at(dense_sums, train_labels, train_dense * weights[:, None])
    dense_norms = np.linalg.norm(dense_sums, axis=1, keepdims=True)
    if np.any(dense_norms <= 0):
        raise RuntimeError("A cross-fitted dense label profile is empty.")
    dense_scores = np.asarray(test_dense @ (dense_sums / dense_norms).T, dtype=np.float32)

    membership = sparse.csr_matrix(
        (
            weights,
            (train_labels, np.arange(len(train_labels), dtype=np.int32)),
        ),
        shape=(label_count, len(train_labels)),
        dtype=np.float32,
    )
    lexical_sums = (membership @ train_lexical).tocsr()
    lexical_norms = np.sqrt(np.asarray(lexical_sums.multiply(lexical_sums).sum(axis=1)).ravel())
    if np.any(lexical_norms <= 0):
        raise RuntimeError("A cross-fitted lexical label profile is empty.")
    lexical_centroids = sklearn_normalize(lexical_sums, norm="l2", axis=1, copy=True)
    lexical_scores = np.asarray((test_lexical @ lexical_centroids.T).toarray(), dtype=np.float32)

    if not np.isfinite(dense_scores).all() or not np.isfinite(lexical_scores).all():
        raise RuntimeError("A cross-fitted score matrix contains non-finite values.")
    return FoldScores(
        dense=dense_scores,
        lexical=lexical_scores,
        label_group_counts=np.rint(label_group_counts).astype(np.int32),
    )


@dataclass
class CrossFitResult:
    ranks: dict[str, np.ndarray]
    metric_arrays: dict[str, dict[str, np.ndarray]]
    fold_rows: list[dict[str, Any]]
    transductive_features: int


def run_crossfit(
    corpus: base.CorpusData,
    groups: FrozenGroups,
    assignment: FoldAssignment,
) -> CrossFitResult:
    query_count = len(corpus.songs)
    label_count = len(corpus.label_ids)
    ranks = {
        system: np.full(query_count, -1, dtype=np.int32)
        for system in SYSTEMS
    }

    print("fitting the matched transductive TF-IDF vocabulary on all unlabeled queries", flush=True)
    transductive_vectorizer = make_vectorizer()
    transductive = transductive_vectorizer.fit_transform(corpus.strict_docs).tocsr().astype(np.float32)
    validate_tfidf(transductive, query_count)
    transductive_features = int(transductive.shape[1])

    fold_rows: list[dict[str, Any]] = []
    all_indices = np.arange(query_count, dtype=np.int32)
    for fold in range(FOLD_COUNT):
        fold_started = time.perf_counter()
        test_indices = all_indices[assignment.query_folds == fold]
        train_indices = all_indices[assignment.query_folds != fold]
        print(
            f"fold {fold + 1}/{FOLD_COUNT}: fitting train-only TF-IDF on {len(train_indices)} queries",
            flush=True,
        )
        inductive_vectorizer = make_vectorizer()
        train_docs = [corpus.strict_docs[int(index)] for index in train_indices]
        test_docs = [corpus.strict_docs[int(index)] for index in test_indices]
        inductive_train = inductive_vectorizer.fit_transform(train_docs).tocsr().astype(np.float32)
        inductive_test = inductive_vectorizer.transform(test_docs).tocsr().astype(np.float32)
        validate_tfidf(inductive_train, len(train_indices))
        validate_tfidf(inductive_test, len(test_indices))

        train_labels = corpus.song_label_index[train_indices]
        test_labels = corpus.song_label_index[test_indices]
        train_groups = groups.names[train_indices]
        inductive_scores = score_fold_profiles(
            corpus.strict_dense[train_indices],
            corpus.strict_dense[test_indices],
            inductive_train,
            inductive_test,
            train_labels,
            train_groups,
            label_count,
        )
        transductive_scores = score_fold_profiles(
            corpus.strict_dense[train_indices],
            corpus.strict_dense[test_indices],
            transductive[train_indices],
            transductive[test_indices],
            train_labels,
            train_groups,
            label_count,
        )
        if not np.array_equal(
            inductive_scores.label_group_counts,
            transductive_scores.label_group_counts,
        ):
            raise RuntimeError("Matched systems used different training profile populations.")

        dense_z = base.zscore_rows(inductive_scores.dense)
        inductive_lexical_z = base.zscore_rows(inductive_scores.lexical)
        transductive_lexical_z = base.zscore_rows(transductive_scores.lexical)
        score_by_system = {
            SYSTEM_BGE: inductive_scores.dense,
            SYSTEM_TFIDF_INDUCTIVE: inductive_scores.lexical,
            SYSTEM_TFIDF_TRANSDUCTIVE: transductive_scores.lexical,
            SYSTEM_FUSION_INDUCTIVE: 0.5 * dense_z + 0.5 * inductive_lexical_z,
            SYSTEM_FUSION_TRANSDUCTIVE: 0.5 * dense_z + 0.5 * transductive_lexical_z,
        }
        for system, scores in score_by_system.items():
            fold_ranks, _ = base.rank_system(scores, test_labels)
            ranks[system][test_indices] = fold_ranks

        test_component_counts = assignment.label_fold_components[:, fold]
        train_component_counts = assignment.label_fold_components.sum(axis=1) - test_component_counts
        test_groups = {
            str(groups.names[int(index)])
            for index in test_indices
        }
        fold_rows.append(
            {
                "fold": fold + 1,
                "train_queries": len(train_indices),
                "test_queries": len(test_indices),
                "train_global_duplicate_components": groups.count - len(test_groups),
                "test_global_duplicate_components": len(test_groups),
                "test_source_credit_labels": int(np.sum(test_component_counts > 0)),
                "minimum_test_components_per_label": int(test_component_counts.min()),
                "median_test_components_per_label": f"{float(np.median(test_component_counts)):.6f}",
                "maximum_test_components_per_label": int(test_component_counts.max()),
                "minimum_training_components_per_label": int(train_component_counts.min()),
                "median_training_components_per_label": f"{float(np.median(train_component_counts)):.6f}",
                "maximum_training_components_per_label": int(train_component_counts.max()),
                "inductive_tfidf_features": int(inductive_train.shape[1]),
                "matched_transductive_tfidf_features": transductive_features,
                "fold_runtime_seconds": f"{time.perf_counter() - fold_started:.3f}",
            }
        )

        del inductive_train, inductive_test, inductive_scores, transductive_scores

    if any(np.any(values < 1) for values in ranks.values()):
        raise RuntimeError("At least one out-of-fold query lacks a valid rank.")
    metric_arrays = {
        system: base.metrics_from_ranks(system_ranks)
        for system, system_ranks in ranks.items()
    }
    return CrossFitResult(
        ranks=ranks,
        metric_arrays=metric_arrays,
        fold_rows=fold_rows,
        transductive_features=transductive_features,
    )


def component_metric_values(
    evaluation: CrossFitResult,
    labels: np.ndarray,
    group_names: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    label_count = int(labels.max()) + 1
    components: list[np.ndarray] = []
    label_group_counts = np.zeros(label_count, dtype=np.int32)
    for label in range(label_count):
        label_indices = np.flatnonzero(labels == label)
        names = sorted({str(group_names[index]) for index in label_indices})
        label_group_counts[label] = len(names)
        group_values = np.zeros((len(names), len(SYSTEMS), len(METRICS)), dtype=np.float64)
        for group_offset, group in enumerate(names):
            members = label_indices[group_names[label_indices] == group]
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
    rng = np.random.default_rng(RANDOM_SEED)
    label_count = len(components)
    point_label = np.stack([values.mean(axis=0) for values in components])
    replicates = np.zeros(
        (BOOTSTRAP_REPLICATES, len(SYSTEMS), len(METRICS)), dtype=np.float64
    )
    outer_occurrences = 0
    repeated_outer_occurrences = 0
    inner_resample_occurrences = 0
    inner_component_draws = 0
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_labels = rng.integers(0, label_count, size=label_count)
        occurrence_counts = Counter(int(label) for label in sampled_labels.tolist())
        outer_occurrences += label_count
        repeated_outer_occurrences += label_count - len(occurrence_counts)
        replicate_total = np.zeros((len(SYSTEMS), len(METRICS)), dtype=np.float64)
        for label, occurrence_count in occurrence_counts.items():
            values = components[label]
            draws = rng.integers(
                0, len(values), size=(occurrence_count, len(values))
            )
            replicate_total += values[draws].mean(axis=1).sum(axis=0)
            inner_resample_occurrences += occurrence_count
            inner_component_draws += occurrence_count * len(values)
        replicates[replicate] = replicate_total / label_count
        if (replicate + 1) % 500 == 0:
            print(f"completed {replicate + 1}/{BOOTSTRAP_REPLICATES} paired bootstraps", flush=True)
    return BootstrapResult(
        point_macro=point_label.mean(axis=0),
        replicates=replicates,
        lower=np.quantile(replicates, 0.025, axis=0),
        upper=np.quantile(replicates, 0.975, axis=0),
        diagnostics={
            "algorithm": "literal_occurrence_wise_two_stage_paired_bootstrap_v1",
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": RANDOM_SEED,
            "outer_label_occurrences_per_replicate": label_count,
            "total_outer_label_occurrences": outer_occurrences,
            "repeated_outer_label_occurrences": repeated_outer_occurrences,
            "independent_inner_resample_occurrences": inner_resample_occurrences,
            "total_inner_component_draws": inner_component_draws,
            "same_component_draw_applied_to_all_systems_and_metrics": True,
            "repeated_outer_occurrences_reuse_inner_draw": False,
        },
    )


def decimal(value: float) -> str:
    return f"{float(value):.6f}"


def interval_direction(lower: float, upper: float) -> str:
    if lower > 0:
        return "left_higher"
    if upper < 0:
        return "left_lower"
    return "interval_includes_zero"


COMPARISONS = (
    (
        "TF-IDF matched transductive minus inductive train-only",
        SYSTEM_TFIDF_TRANSDUCTIVE,
        SYSTEM_TFIDF_INDUCTIVE,
    ),
    (
        "fusion matched transductive minus inductive train-only",
        SYSTEM_FUSION_TRANSDUCTIVE,
        SYSTEM_FUSION_INDUCTIVE,
    ),
    (
        "inductive fusion minus inductive TF-IDF",
        SYSTEM_FUSION_INDUCTIVE,
        SYSTEM_TFIDF_INDUCTIVE,
    ),
    (
        "matched-transductive fusion minus matched-transductive TF-IDF",
        SYSTEM_FUSION_TRANSDUCTIVE,
        SYSTEM_TFIDF_TRANSDUCTIVE,
    ),
)


def build_metric_rows(
    bootstrap: BootstrapResult,
    query_count: int,
    label_count: int,
    global_groups: int,
    within_label_units: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system_index, system in enumerate(SYSTEMS):
        for metric_index, metric in enumerate(METRICS):
            rows.append(
                {
                    "system": system,
                    "aggregation": "source_credit_label_macro_duplicate_component_adjusted",
                    "metric": metric,
                    "estimate": decimal(bootstrap.point_macro[system_index, metric_index]),
                    "ci95_lower": decimal(bootstrap.lower[system_index, metric_index]),
                    "ci95_upper": decimal(bootstrap.upper[system_index, metric_index]),
                    "queries": query_count,
                    "source_credit_labels": label_count,
                    "global_duplicate_components": global_groups,
                    "within_label_component_units": within_label_units,
                    "crossfit_folds": FOLD_COUNT,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    return rows


def build_delta_rows(bootstrap: BootstrapResult) -> list[dict[str, Any]]:
    system_index = {system: index for index, system in enumerate(SYSTEMS)}
    rows: list[dict[str, Any]] = []
    for comparison, left, right in COMPARISONS:
        for metric_offset, metric in enumerate(METRICS):
            left_index = system_index[left]
            right_index = system_index[right]
            replicate_delta = (
                bootstrap.replicates[:, left_index, metric_offset]
                - bootstrap.replicates[:, right_index, metric_offset]
            )
            lower, upper = np.quantile(replicate_delta, [0.025, 0.975])
            point = (
                bootstrap.point_macro[left_index, metric_offset]
                - bootstrap.point_macro[right_index, metric_offset]
            )
            rows.append(
                {
                    "comparison": comparison,
                    "left_system": left,
                    "right_system": right,
                    "metric": metric,
                    "estimate_delta": decimal(point),
                    "ci95_lower": decimal(lower),
                    "ci95_upper": decimal(upper),
                    "interval_direction": interval_direction(float(lower), float(upper)),
                    "paired_two_stage_bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    return rows


def build_documents(
    bootstrap: BootstrapResult,
    delta_rows: list[dict[str, Any]],
    query_count: int,
    global_groups: int,
    within_label_units: int,
) -> tuple[str, str]:
    system_index = {system: index for index, system in enumerate(SYSTEMS)}
    metric_index = {metric: index for index, metric in enumerate(METRICS)}

    def value(system: str, metric: str) -> float:
        return float(bootstrap.point_macro[system_index[system], metric_index[metric]])

    transductive_mrr_delta = value(SYSTEM_TFIDF_TRANSDUCTIVE, "mrr") - value(
        SYSTEM_TFIDF_INDUCTIVE, "mrr"
    )
    inductive_fusion_mrr_delta = value(SYSTEM_FUSION_INDUCTIVE, "mrr") - value(
        SYSTEM_TFIDF_INDUCTIVE, "mrr"
    )
    readme = f"""# Retrieval inductive sensitivity v1

## Result

Fitting character TF-IDF on the complete unlabeled evaluation corpus gives a small matched advantage over fitting vocabulary and IDF on training folds only: macro MRR changes from **{value(SYSTEM_TFIDF_INDUCTIVE, 'mrr'):.3f}** to **{value(SYSTEM_TFIDF_TRANSDUCTIVE, 'mrr'):.3f}** (delta {transductive_mrr_delta:+.3f}). The central fusion result remains under the fully inductive design: inductive fusion reaches macro MRR **{value(SYSTEM_FUSION_INDUCTIVE, 'mrr'):.3f}**, {inductive_fusion_mrr_delta:+.3f} above inductive TF-IDF.

This supports a bounded conclusion: transductive vocabulary/IDF exposure contributes slight optimism but does not explain the semantic-lexical fusion gain.

## Files

- `metrics.csv`: duplicate-component-adjusted source-credit-label macro estimates and paired bootstrap intervals.
- `paired_deltas.csv`: paired system differences with 95% intervals.
- `fold_summary.csv`: aggregate six-fold coverage and feature counts.
- `analysis_summary.json`: machine-readable result and lineage.
- `METHOD.md`: complete estimand, split, scoring, aggregation, and uncertainty definitions.
- `validation.json` and `manifest.json`: release checks and byte-level inventory.

## Population and privacy

The analysis contains {query_count:,} length-qualified song queries, {global_groups:,} global exact/near-duplicate components, {within_label_units:,} within-label component units, and 204 eligible source-credit labels. Public outputs contain no song identifiers, label names, lyric text, per-query scores, fold assignments, or vectors.

## Claim boundary

{CLAIM_BOUNDARY}
"""

    method = f"""# Method: grouped cross-fitted inductive TF-IDF sensitivity

## Question and estimand

The analysis asks whether the retrieval-v1 lexical result and its equal-weight semantic-lexical fusion depend materially on fitting TF-IDF vocabulary and inverse-document-frequency weights on the complete unlabeled evaluation corpus. The estimand is the paired difference between two otherwise matched systems within the fixed, leakage-controlled corpus.

This is a sensitivity analysis, not a replacement for retrieval-v1. Absolute cross-fitted estimates use five-sixths of duplicate components to build each fold's label profiles and therefore should not be interpreted as a direct estimate of the original leave-one-component-out profile-size effect.

## Frozen population

The population is inherited from retrieval-v1: {query_count:,} length-qualified held-out songs from 204 eligible source-credit labels after shared-text exclusion. Exact and character-trigram-Jaccard >= 0.80 near duplicates use the frozen private retrieval-v1 component assignments. There are {global_groups:,} global components and {within_label_units:,} within-label component units; a cross-label component contributes one unit to every label stratum that it intersects.

## Grouped six-fold split

Every global duplicate component is assigned wholly to one fold, so no exact/near-duplicate component crosses train and test. Components are ordered with multi-label components first, then larger components, then an unseeded SHA-256 tie-break of the frozen private component identifier. For a component touching labels L, the selected fold lexicographically minimizes:

1. the maximum current component count in that fold over labels in L;
2. the sum of each affected label's current fold count divided by its total component count;
3. the fold's current song count; and
4. the fold index.

All six test folds contain every source-credit label, and every corresponding training partition retains at least five components per label.

## Duplicate-component-weighted profiles

Within a training partition, song i in component g and label l receives weight

`w_i = 1 / n_train(g,l)`.

Weights therefore sum to one for every observed component-label unit. For normalized dense song vector d_i, the label profile is

`c_dense(l) = normalize(sum_i I[y_i=l] * w_i * d_i)`.

The lexical profile uses the same weighted sum and L2 normalization in sparse TF-IDF space. Test-to-label scores are cosine similarities.

## Lexical representations

Both lexical systems use character 2-5 gram TF-IDF with `min_df=3`, sublinear term frequency, L2 normalization, float32 values, and at most {TFIDF_MAX_FEATURES:,} features.

- **Inductive:** fit vocabulary and IDF only on the current fold's training documents, then transform training and test documents.
- **Matched transductive:** fit vocabulary and IDF once on all {query_count:,} unlabeled documents, while still constructing each fold's label profiles only from that fold's training rows and scoring only its test rows.

The paired comparison therefore changes lexical vocabulary/IDF exposure while holding query population, fold membership, label-profile membership, duplicate weights, candidate labels, score fusion, ranking, metrics, and resampling fixed.

## Fusion and ranking

Dense and lexical candidate-label score vectors are standardized separately within every query over the same 204 candidate labels. Fusion is the untrained average

`s_fusion(q,l) = 0.5 * z_l(s_dense(q,l)) + 0.5 * z_l(s_lexical(q,l))`.

Candidate ties use the frozen ascending label-ID order. Each query receives exactly one out-of-fold prediction.

## Metrics and aggregation

With one relevant source-credit label per query, the metrics are reciprocal rank, Recall@1, Recall@5, Recall@10, and nDCG@10. Query metrics are first averaged within every duplicate-component-by-label unit. Units are then averaged within labels, and the headline estimate is the unweighted macro mean over 204 labels.

## Uncertainty

Intervals use {BOOTSTRAP_REPLICATES:,} fixed-seed paired two-stage bootstrap replicates. Each replicate samples 204 outer label occurrences with replacement. For every outer occurrence, that label's component units are independently sampled with replacement at the original stratum size. The same sampled units are applied to the complete system-by-metric tensor, preserving paired differences. Percentile 2.5% and 97.5% endpoints form the 95% intervals.

These intervals describe resampling variability within the fixed corpus strata. They do not quantify uncertainty for an external population of artists or songs.

## Interpretation rule

The matched-transductive-minus-inductive delta estimates the contribution of unlabeled evaluation-corpus vocabulary/IDF exposure under matched folds and profiles. It does not estimate causal generalization to a new corpus. A small positive delta is described as limited transductive optimism, not leakage-free evidence. The inductive-fusion-minus-inductive-TF-IDF comparison tests whether the fusion gain remains when lexical fitting is training-only.

## Public-release boundary

Only aggregate metrics, aggregate fold diagnostics, methods, validation, and hashes are released. Song identifiers, source-credit-label names, lyric text, per-query ranks or scores, fold assignments, and vector arrays remain private.

{CLAIM_BOUNDARY}
"""
    return readme, method


def build_analysis_summary(
    generated_at: str,
    runtime_seconds: float,
    corpus: base.CorpusData,
    groups: FrozenGroups,
    assignment: FoldAssignment,
    evaluation: CrossFitResult,
    bootstrap: BootstrapResult,
    delta_rows: list[dict[str, Any]],
    lineage: dict[str, str],
) -> dict[str, Any]:
    system_index = {system: index for index, system in enumerate(SYSTEMS)}
    metric_index = {metric: index for index, metric in enumerate(METRICS)}

    def metric_payload(system: str, metric: str) -> dict[str, float]:
        si, mi = system_index[system], metric_index[metric]
        return {
            "estimate": round(float(bootstrap.point_macro[si, mi]), 6),
            "ci95_lower": round(float(bootstrap.lower[si, mi]), 6),
            "ci95_upper": round(float(bootstrap.upper[si, mi]), 6),
        }

    delta_payload: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in delta_rows:
        delta_payload[row["comparison"]][row["metric"]] = {
            "estimate_delta": float(row["estimate_delta"]),
            "ci95_lower": float(row["ci95_lower"]),
            "ci95_upper": float(row["ci95_upper"]),
            "interval_direction": row["interval_direction"],
        }

    return {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": generated_at,
        "runtime_seconds": round(runtime_seconds, 3),
        "task": "grouped cross-fitted inductive-versus-matched-transductive lexical retrieval sensitivity",
        "claim_boundary": CLAIM_BOUNDARY,
        "population": {
            "queries": len(corpus.songs),
            "source_credit_labels": len(corpus.label_ids),
            "global_exact_near_duplicate_components": groups.count,
            "within_label_component_units": groups.within_label_units,
            "folds": FOLD_COUNT,
            "fold_query_counts": assignment.fold_song_counts.tolist(),
            "minimum_test_components_per_label": int(assignment.label_fold_components.min()),
            "minimum_training_components_per_label": int(
                (
                    assignment.label_fold_components.sum(axis=1, keepdims=True)
                    - assignment.label_fold_components
                ).min()
            ),
        },
        "split": {
            "unit": "global exact/near-duplicate component",
            "component_crosses_folds": False,
            "stratification_labels_used_only_for_fold_balance": True,
            "randomized": False,
            "assignment": "deterministic greedy label-component balancing with SHA-256 tie-break",
        },
        "representations": {
            "dense": "frozen 1024-dimensional normalized BGE-M3 song vectors",
            "tfidf": {
                "analyzer": "character",
                "ngram_range": [2, 5],
                "min_df": 3,
                "sublinear_tf": True,
                "norm": "l2",
                "maximum_features": TFIDF_MAX_FEATURES,
                "matched_transductive_features": evaluation.transductive_features,
                "inductive_features_by_fold": [
                    int(row["inductive_tfidf_features"]) for row in evaluation.fold_rows
                ],
            },
            "fusion": "equal-weight per-query candidate-label z-score average; no tuned weight",
        },
        "macro_metrics": {
            system: {
                metric: metric_payload(system, metric)
                for metric in METRICS
            }
            for system in SYSTEMS
        },
        "paired_deltas": dict(delta_payload),
        "interpretation": {
            "tfidf_exposure": "Matched transductive vocabulary/IDF exposure has a small positive advantage on several metrics.",
            "fusion_robustness": "The inductive fusion advantage over inductive TF-IDF remains materially larger than the exposure effect.",
            "scope": "Sensitivity within the fixed corpus; not external-population or causal evidence.",
        },
        "uncertainty": bootstrap.diagnostics,
        "lineage": lineage,
    }


def check_regression_reference(bootstrap: BootstrapResult) -> dict[str, Any]:
    system_index = {system: index for index, system in enumerate(SYSTEMS)}
    metric_index = {metric: index for index, metric in enumerate(METRICS)}
    differences: dict[str, float] = {}
    for system, expected_metrics in READ_ONLY_REFERENCE.items():
        for metric, expected in expected_metrics.items():
            actual = float(bootstrap.point_macro[system_index[system], metric_index[metric]])
            differences[f"{system}::{metric}"] = abs(actual - expected)
    return {
        "passed": max(differences.values()) <= 5.1e-7,
        "maximum_absolute_difference": max(differences.values()),
        "tolerance": 5.1e-7,
    }


def text_contract(path: Path) -> bool:
    payload = path.read_bytes()
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return not payload.startswith(b"\xef\xbb\xbf") and b"\r" not in payload and payload.endswith(b"\n")


def build_manifest(directory: Path, generated_at: str, lineage: dict[str, str]) -> dict[str, Any]:
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    return {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": generated_at,
        "classification": "public aggregate-only grouped cross-fitted retrieval sensitivity",
        "claim_boundary": CLAIM_BOUNDARY,
        "files": files,
        "lineage": lineage,
    }


def publish_staging(staging: Path, output_dir: Path) -> None:
    expected_parent = (REPO_ROOT / "results").resolve()
    resolved_output = output_dir.resolve()
    if resolved_output.name != ARTIFACT_ID or resolved_output.parent != expected_parent:
        raise RuntimeError(f"Refusing to replace unexpected output directory: {resolved_output}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(staging, output_dir)


def build(project_root: Path, output_dir: Path) -> None:
    started = time.perf_counter()
    generated_at = utc_now()
    project_root = project_root.resolve()
    configure_base(project_root)
    source_lineage = base.validate_source_contracts()
    query_path, _private_manifest, private_lineage = validate_private_retrieval(project_root)
    corpus = base.load_corpus()
    print(f"loaded {len(corpus.songs)} queries across {len(corpus.label_ids)} labels", flush=True)
    groups = load_frozen_groups(query_path, corpus)
    assignment = assign_grouped_folds(groups.names, corpus.song_label_index, len(corpus.label_ids))
    print(f"fold query counts: {assignment.fold_song_counts.tolist()}", flush=True)
    evaluation = run_crossfit(corpus, groups, assignment)
    components, label_group_counts = component_metric_values(
        evaluation, corpus.song_label_index, groups.names
    )
    if int(label_group_counts.sum()) != groups.within_label_units:
        raise RuntimeError("Component aggregation does not reproduce the frozen within-label unit count.")
    bootstrap = run_bootstrap(components)
    regression = check_regression_reference(bootstrap)
    if not regression["passed"]:
        raise RuntimeError(
            "Cross-fitted point estimates do not reproduce the independent read-only reference: "
            f"{regression}"
        )

    metric_rows = build_metric_rows(
        bootstrap,
        len(corpus.songs),
        len(corpus.label_ids),
        groups.count,
        groups.within_label_units,
    )
    delta_rows = build_delta_rows(bootstrap)
    readme, method = build_documents(
        bootstrap,
        delta_rows,
        len(corpus.songs),
        groups.count,
        groups.within_label_units,
    )
    runtime_seconds = time.perf_counter() - started
    lineage = {
        **source_lineage,
        **private_lineage,
        "builder_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    analysis_summary = build_analysis_summary(
        generated_at,
        runtime_seconds,
        corpus,
        groups,
        assignment,
        evaluation,
        bootstrap,
        delta_rows,
        lineage,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}-", dir=output_dir.parent))
    try:
        atomic_write_text(staging / "README.md", readme)
        atomic_write_text(staging / "METHOD.md", method)
        atomic_write_csv(
            staging / "metrics.csv",
            metric_rows,
            [
                "system",
                "aggregation",
                "metric",
                "estimate",
                "ci95_lower",
                "ci95_upper",
                "queries",
                "source_credit_labels",
                "global_duplicate_components",
                "within_label_component_units",
                "crossfit_folds",
                "bootstrap_replicates",
            ],
        )
        atomic_write_csv(
            staging / "paired_deltas.csv",
            delta_rows,
            [
                "comparison",
                "left_system",
                "right_system",
                "metric",
                "estimate_delta",
                "ci95_lower",
                "ci95_upper",
                "interval_direction",
                "paired_two_stage_bootstrap_replicates",
            ],
        )
        atomic_write_csv(
            staging / "fold_summary.csv",
            evaluation.fold_rows,
            [
                "fold",
                "train_queries",
                "test_queries",
                "train_global_duplicate_components",
                "test_global_duplicate_components",
                "test_source_credit_labels",
                "minimum_test_components_per_label",
                "median_test_components_per_label",
                "maximum_test_components_per_label",
                "minimum_training_components_per_label",
                "median_training_components_per_label",
                "maximum_training_components_per_label",
                "inductive_tfidf_features",
                "matched_transductive_tfidf_features",
                "fold_runtime_seconds",
            ],
        )
        atomic_write_json(staging / "analysis_summary.json", analysis_summary)

        expected_files_before_validation = {
            "README.md",
            "METHOD.md",
            "metrics.csv",
            "paired_deltas.csv",
            "fold_summary.csv",
            "analysis_summary.json",
        }
        forbidden_headers = {
            "song_id",
            "chunk_id",
            "analysis_text",
            "text",
            "embedding",
            "vector",
            "clean_row_index",
            "source_credit_label",
        }
        bad_headers: dict[str, list[str]] = {}
        for csv_path in sorted(staging.glob("*.csv")):
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                headers = set(next(csv.reader(handle)))
            overlap = sorted(headers & forbidden_headers)
            if overlap:
                bad_headers[csv_path.name] = overlap

        train_counts = (
            assignment.label_fold_components.sum(axis=1, keepdims=True)
            - assignment.label_fold_components
        )
        checks = [
            {
                "name": "all_frozen_input_contracts_and_hashes_pass",
                "passed": True,
            },
            {
                "name": "frozen_query_population_and_duplicate_units_reconciled",
                "passed": len(corpus.songs) == 5455
                and len(corpus.label_ids) == 204
                and groups.count == 5430
                and groups.within_label_units == 5432,
                "detail": {
                    "queries": len(corpus.songs),
                    "source_credit_labels": len(corpus.label_ids),
                    "global_components": groups.count,
                    "within_label_component_units": groups.within_label_units,
                },
            },
            {
                "name": "global_duplicate_components_never_cross_folds",
                "passed": len(assignment.group_fold) == groups.count
                and all(
                    len(set(assignment.query_folds[groups.names == group].tolist())) == 1
                    for group in assignment.group_fold
                ),
            },
            {
                "name": "six_folds_cover_all_labels_and_retain_five_training_components",
                "passed": assignment.fold_song_counts.tolist() == [910, 909, 909, 909, 909, 909]
                and int(assignment.label_fold_components.min()) >= 1
                and int(train_counts.min()) >= 5,
                "detail": {
                    "fold_query_counts": assignment.fold_song_counts.tolist(),
                    "minimum_test_components_per_label": int(
                        assignment.label_fold_components.min()
                    ),
                    "minimum_training_components_per_label": int(train_counts.min()),
                },
            },
            {
                "name": "inductive_and_matched_systems_use_identical_fold_profiles",
                "passed": True,
            },
            {
                "name": "all_tfidf_fits_reach_prespecified_feature_cap",
                "passed": evaluation.transductive_features == TFIDF_MAX_FEATURES
                and all(
                    int(row["inductive_tfidf_features"]) == TFIDF_MAX_FEATURES
                    for row in evaluation.fold_rows
                ),
            },
            {
                "name": "every_query_has_exactly_one_valid_out_of_fold_rank",
                "passed": all(
                    len(values) == len(corpus.songs)
                    and np.all((values >= 1) & (values <= len(corpus.label_ids)))
                    for values in evaluation.ranks.values()
                ),
            },
            {
                "name": "paired_two_stage_bootstrap_is_complete",
                "passed": bootstrap.diagnostics["total_outer_label_occurrences"]
                == BOOTSTRAP_REPLICATES * len(corpus.label_ids)
                and bootstrap.diagnostics["independent_inner_resample_occurrences"]
                == bootstrap.diagnostics["total_outer_label_occurrences"]
                and bootstrap.diagnostics["same_component_draw_applied_to_all_systems_and_metrics"]
                is True,
                "detail": bootstrap.diagnostics,
            },
            {
                "name": "point_estimates_reproduce_independent_read_only_run",
                "passed": regression["passed"],
                "detail": regression,
            },
            {
                "name": "all_system_metric_and_paired_comparison_rows_present",
                "passed": len(metric_rows) == len(SYSTEMS) * len(METRICS)
                and len(delta_rows) == len(COMPARISONS) * len(METRICS),
            },
            {
                "name": "macro_intervals_are_ordered_and_bounded",
                "passed": all(
                    0.0
                    <= float(row["ci95_lower"])
                    <= float(row["estimate"])
                    <= float(row["ci95_upper"])
                    <= 1.0
                    for row in metric_rows
                ),
            },
            {
                "name": "public_csv_headers_exclude_private_query_label_text_and_vector_fields",
                "passed": not bad_headers,
                "detail": bad_headers,
            },
            {
                "name": "public_payloads_are_utf8_without_bom_and_lf_only",
                "passed": set(path.name for path in staging.iterdir())
                == expected_files_before_validation
                and all(text_contract(path) for path in staging.iterdir() if path.is_file()),
            },
            {
                "name": "claim_boundary_present_in_public_documents",
                "passed": CLAIM_BOUNDARY in readme and CLAIM_BOUNDARY in method,
            },
        ]
        validation = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": generated_at,
            "status": "pass" if all(check["passed"] for check in checks) else "fail",
            "checks": checks,
        }
        atomic_write_json(staging / "validation.json", validation)
        if validation["status"] != "pass":
            raise RuntimeError("Sensitivity artifact validation failed before publication.")

        atomic_write_json(
            staging / "manifest.json",
            build_manifest(staging, generated_at, lineage),
        )
        expected_final_files = expected_files_before_validation | {"validation.json", "manifest.json"}
        if set(path.name for path in staging.iterdir()) != expected_final_files:
            raise RuntimeError("The staged public allowlist is not exact.")
        if not all(text_contract(path) for path in staging.iterdir() if path.is_file()):
            raise RuntimeError("A final public text file violates UTF-8/no-BOM/LF-only policy.")
        persisted_manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in persisted_manifest["files"].items():
            path = staging / name
            if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                raise RuntimeError(f"A staged payload differs from manifest: {name}")

        publish_staging(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(
        f"built {output_dir} in {time.perf_counter() - started:.3f} seconds",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Private project root containing outputs/ and work/ (default: auto-detected shared root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Must be this repository's results/retrieval-inductive-sensitivity-v1 directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build(args.project_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
