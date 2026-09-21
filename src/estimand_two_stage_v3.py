#!/usr/bin/env python3
"""Do the headline contrasts survive resampling the artists? The retrieval builder's
two-stage bootstrap and label-macro estimand, ported to corpus v3.

Every interval published on corpus v3 resamples leakage groups with replacement and weights
each (group, label) component one (identity_spaces_v2.paired_group_bootstrap). Its estimand
is the component-weighted MRR over queries, so a label with a hundred songs moves it twenty
times as far as a label with five, and its interval conditions on the 226 labels being the labels.
The retrieval builder (build_chinese_rap_downstream_retrieval_v1, carried to corpus v2 by
build_downstream_retrieval_v2) reported a different estimand under a different resampling:
the label-macro MRR -- the component-weighted mean within each label, then the plain mean
over labels -- with a paired two-stage bootstrap that draws labels with replacement and then,
for every drawn label occurrence, that label's components with replacement. That design
treats the artists as a sample from a population of artists, so its interval also carries the
between-artist variation the group bootstrap holds fixed. The label-size study
(label_size_calibration_v3) found the raw word lead absent among labels with fewer than
twenty songs and asked for a label-macro estimand beside the query-weighted one. Nothing of
the builder's design has been run on corpus v3. This file scores the published systems once
more, unchanged, and reports every headline contrast under both estimands and both
resamplings.

QUESTION. Which of the published contrasts hold when the artists themselves are resampled,
read on the label-macro MRR and R@1, and how far apart are the query-weighted,
component-weighted and label-macro readings of each system?

SYSTEMS, all scored on the same 7,220 queries against leave-group-out profiles of the same
226 labels (exemplar_vs_prototype_v3.setup; the folds of identity_probe_v2):
  semantic_raw                       BGE-M3 song centroids, the builder's dense arm
  characters_raw                     character 2-5-gram TF-IDF, the builder's lexical arm
  words_raw                          jieba word 1-2-gram TF-IDF (word_identity_anatomy_v2)
  semantic_total_whitened            total-covariance whitening of the centroids, fitted on
                                     the training folds and applied to the held-out fold
  semantic_within_whitened           within-author whitening, the same folds
  semantic_within_whitened_permuted  within-author whitening fitted on permuted training
                                     labels, the same folds and generator seeds (the null,
                                     recomputed exactly as identity_probe_v2 does)
  words_svd_within_whitened          word TF-IDF reduced by SVD-1024 fitted on the training
                                     folds, then within-author whitening (word_space_probe_v2)
  semantic_whitened_chunks           within-author whitening fitted on chunk vectors, the
                                     whitened chunks averaged into songs (representation_unit_v2)
  fusion_whitened_words_semantic     equal-weight z-score fusion of words_svd_within_whitened
                                     and semantic_whitened_chunks, the published best system

ESTIMANDS, from one set of ranks per system (ranks_of: the builder's tie rule), each for
MRR, R@1 and R@10:
  query_weighted       mean over queries -- the published numbers
  component_weighted   each (leakage group, label) component weight one -- the estimand of
                       every published interval (identity_spaces_v2.weighted_mean)
  label_macro          component-weighted mean within each label, then the plain mean over
                       labels -- the retrieval builder's estimand

RESAMPLING. Two designs, both paired (one draw indexes every system and metric at once):
  group bootstrap      paired_group_bootstrap: 2,000 replicates, seed 20260825, leakage
                       groups drawn with replacement; intervals of the component-weighted
                       MRR and R@1 differences
  two-stage bootstrap  the builder's own functions, v1.component_metric_values and
                       v1.run_bootstrap, called through the constant redirect that
                       build_downstream_retrieval_v2 uses (v1.SYSTEMS and v1.METRICS pointed
                       at this file's systems and metrics for the duration of the call), with
                       the builder's seed 20260825 and 5,000 replicates: 226 label occurrences
                       drawn with replacement, then for every occurrence an independent draw,
                       with replacement, of that label's components (a component's value is
                       the mean over its songs, so a song filed five times counts once);
                       intervals of the label-macro MRR, R@1 and R@10 differences

CONTRASTS, confirmatory and fixed before the run: words - characters; characters - semantic;
words - semantic; within-whitened - total-whitened semantic; within-whitened - permuted-label
within-whitened semantic; whitened word SVD - within-whitened semantic (the centroid-whitened
space of identity_probe_v2); fusion - words. Supplementary: whitened word SVD - chunk-whitened
semantic; fusion - whitened word SVD.

CHECKS before any result is used (each raises SystemExit on failure):
  1. the query-weighted MRR of every system reproduces its published corpus-v3 number the
     moment the system is scored, before the next system is built: gap <= 5e-4 for the three
     raw spaces, the repo's own tolerance for them, and <= 0.002 for the fold-wise systems;
     where the repo published a component-weighted MRR it must reproduce within 0.002 too;
  2. dense_leave_group_out on the float64 centroids reproduces the builder's dense scores
     (max gap <= 1e-4), the builder's dense arm is identical between its two calls, and the
     number of (group, label) components equals the builder's label_group_counts total;
  3. the second load of the corpus, made for the chunk vectors, contains every query song
     and gives the same song centroids as setup() (max gap <= 1e-6);
  4. this file's vectorised component means equal the builder's component tensors label by
     label (max gap <= 1e-12), the builder's component count equals the count from check 2,
     and the label-macro MRR computed directly from the ranks equals the builder's two-stage
     point estimate (gap <= 1e-9);
  5. the three contrasts the repo already published from paired_group_bootstrap on these
     ranks (within - total, within - permuted, fusion - whitened word SVD) reproduce their
     published point differences within 0.002;
  6. test_estimand_two_stage.py checks, on random data without the corpus, the component
     means against v1.component_metric_values, the two-stage wrapper replicate for replicate
     against a literal transcription of v1.run_bootstrap under the same seed, the fold-wise
     scorer against a literal transcription of identity_probe_v2's fold loop, and the three
     estimands and the contrast intervals against brute-force definitions.

READING RULE, fixed before the run. A contrast is "established under label resampling" if
its two-stage 95% interval for the label-macro MRR difference excludes zero, and "not
established under label resampling" otherwise. Every contrast is reported either way, with
its label-macro R@1 and R@10 intervals and its group-bootstrap component-weighted intervals
beside it; when the two resamplings disagree the disagreement is named, not resolved. The
reconciliation table (query-weighted, component-weighted, label-macro) is description.

    CHINESE_RAP_CORPUS=v3 python src/estimand_two_stage_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from build_downstream_retrieval_v2 import build_songs, corpus_version  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from exemplar_vs_prototype_v3 import setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402
from word_identity_anatomy_v2 import EXPECTED_WORD_MRR, fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "estimand_two_stage.json"
METRICS = ("mrr", "recall_at_1", "recall_at_10")
SYSTEMS = ("semantic_raw", "characters_raw", "words_raw",
           "semantic_total_whitened", "semantic_within_whitened", "semantic_within_whitened_permuted",
           "words_svd_within_whitened", "semantic_whitened_chunks", "fusion_whitened_words_semantic")
RAW_SPACES = ("semantic_raw", "characters_raw", "words_raw")
CONFIRMATORY = (("words_raw", "characters_raw"),
                ("characters_raw", "semantic_raw"),
                ("words_raw", "semantic_raw"),
                ("semantic_within_whitened", "semantic_total_whitened"),
                ("semantic_within_whitened", "semantic_within_whitened_permuted"),
                ("words_svd_within_whitened", "semantic_within_whitened"),
                ("fusion_whitened_words_semantic", "words_raw"))
SUPPLEMENTARY = (("words_svd_within_whitened", "semantic_whitened_chunks"),
                 ("fusion_whitened_words_semantic", "words_svd_within_whitened"))
TWO_STAGE_REPLICATES = v1.BOOTSTRAP_REPLICATES      # 5,000, the builder's
TWO_STAGE_SEED = v1.RANDOM_SEED                     # 20260825, the builder's
RAW_GAP, FOLD_GAP = 5e-4, 0.002
# published corpus-v3 numbers: (query-weighted MRR, tolerance, component-weighted MRR or None, source)
EXPECTED = {
    "semantic_raw": (0.2997, RAW_GAP, 0.2979, "identity_probe.json systems.none; representation_unit.json song_centroid"),
    "characters_raw": (0.4266, RAW_GAP, 0.4310, "identity_probe.json systems.lexical_character_ngrams"),
    "words_raw": (EXPECTED_WORD_MRR, RAW_GAP, None, "three_spaces.json systems.lexical_words (read by expected())"),
    "semantic_total_whitened": (0.3996, FOLD_GAP, 0.3968, "identity_probe.json systems.total_whitening"),
    "semantic_within_whitened": (0.4164, FOLD_GAP, 0.4136, "identity_probe.json systems.within_author_whitening"),
    "semantic_within_whitened_permuted": (0.3983, FOLD_GAP, 0.3958,
                                          "identity_probe.json systems.within_author_whitening_permuted_labels"),
    "words_svd_within_whitened": (0.5426, FOLD_GAP, None, "word_space_probe.json systems.words_svd_within_author_whitening"),
    "semantic_whitened_chunks": (0.4271, FOLD_GAP, None,
                                 "word_space_probe.json systems.semantic_whitened_chunks; representation_unit.json whitened_chunks_then_mean"),
    "fusion_whitened_words_semantic": (0.5854, FOLD_GAP, None,
                                       "word_space_probe.json systems.fusion_whitened_words_whitened_semantic"),
}
# group-bootstrap contrasts the repo already published on these ranks: (left, right, published point, source)
PUBLISHED_GROUP_CONTRASTS = (
    ("semantic_within_whitened", "semantic_total_whitened", 0.0168, "identity_probe.json"),
    ("semantic_within_whitened", "semantic_within_whitened_permuted", 0.0179, "identity_probe.json"),
    ("fusion_whitened_words_semantic", "words_svd_within_whitened", 0.0394, "word_space_probe.json"),
)


# ------------------------------------------------------------------ estimands
def component_index(label_index: np.ndarray, group_ids: np.ndarray):
    """Component id per song, components ordered by (label, group): every label's components
    are contiguous and in increasing group order, the row order of v1.component_metric_values."""
    keys = sorted(set(zip(label_index.tolist(), group_ids.tolist())))
    position = {key: i for i, key in enumerate(keys)}
    comp_of = np.asarray([position[(int(l), int(g))] for l, g in zip(label_index.tolist(), group_ids.tolist())], dtype=np.int64)
    comp_label = np.asarray([key[0] for key in keys], dtype=np.int64)
    return comp_of, comp_label


def component_means(values: np.ndarray, comp_of: np.ndarray, component_count: int) -> np.ndarray:
    """Mean of each column of values (n_queries, K) over the songs of each component."""
    sums = np.zeros((component_count, values.shape[1]))
    np.add.at(sums, comp_of, values)
    counts = np.bincount(comp_of, minlength=component_count)
    if np.any(counts == 0):
        raise RuntimeError("an empty component")
    return sums / counts[:, None]


def per_label_components(comp_values: np.ndarray, comp_label: np.ndarray, label_count: int) -> list[np.ndarray]:
    out = [comp_values[comp_label == l] for l in range(label_count)]
    if any(len(v) == 0 for v in out):
        raise RuntimeError("a label without components")
    return out


def label_macro(values: np.ndarray, weights: np.ndarray, label_index: np.ndarray, label_count: int) -> np.ndarray:
    """Component-weighted mean within each label, then the plain mean over labels; values (n, K)."""
    num = np.zeros((label_count, values.shape[1]))
    np.add.at(num, label_index, values * weights[:, None])
    den = np.bincount(label_index, weights=weights, minlength=label_count)
    if np.any(den <= 0):
        raise RuntimeError("a label without weight")
    return (num / den[:, None]).mean(axis=0)


def estimand_table(values: np.ndarray, weights: np.ndarray, label_index: np.ndarray, label_count: int) -> dict:
    """query-weighted (plain), component-weighted and label-macro mean of each column."""
    everything = np.ones(len(values), dtype=bool)
    macro = label_macro(values, weights, label_index, label_count)
    return {"plain_all": [float(values[:, k].mean()) for k in range(values.shape[1])],
            "component_weighted": [weighted_mean(values[:, k], weights, everything) for k in range(values.shape[1])],
            "label_macro": [float(m) for m in macro]}


def two_stage(metric_arrays: dict, label_index: np.ndarray, group_ids: np.ndarray, systems, metrics,
              replicates: int = v1.BOOTSTRAP_REPLICATES, seed: int = v1.RANDOM_SEED):
    """The builder's two-stage paired bootstrap by the builder's own functions.

    v1.component_metric_values and v1.run_bootstrap read v1.SYSTEMS, v1.METRICS,
    v1.BOOTSTRAP_REPLICATES and v1.RANDOM_SEED at call time; they are pointed at this file's
    systems and metrics for the duration of the call and restored afterwards, the redirect
    build_downstream_retrieval_v2 uses. metric_arrays is {system: {metric: (n,) array}}.
    Returns the builder's per-label component tensors, its label_group_counts, the macro point
    flattened system-major/metric-minor (S*M,), the replicate matrix (replicates, S*M) in the
    same column order, and the builder's diagnostics.
    """
    systems, metrics = tuple(systems), tuple(metrics)
    saved = (v1.SYSTEMS, v1.METRICS, v1.BOOTSTRAP_REPLICATES, v1.RANDOM_SEED)
    v1.SYSTEMS, v1.METRICS, v1.BOOTSTRAP_REPLICATES, v1.RANDOM_SEED = systems, metrics, int(replicates), int(seed)
    try:
        if (v1.SYSTEMS, v1.METRICS, v1.BOOTSTRAP_REPLICATES, v1.RANDOM_SEED) != (systems, metrics, int(replicates), int(seed)):
            raise SystemExit("the builder's constants were not redirected")
        evaluation = types.SimpleNamespace(metric_arrays=metric_arrays)   # component_metric_values reads only metric_arrays
        components, label_group_counts = v1.component_metric_values(evaluation, label_index, group_ids)
        outcome = v1.run_bootstrap(components)
    finally:
        v1.SYSTEMS, v1.METRICS, v1.BOOTSTRAP_REPLICATES, v1.RANDOM_SEED = saved
    S, M = len(systems), len(metrics)
    if outcome.replicates.shape != (int(replicates), S, M) or outcome.point_macro.shape != (S, M):
        raise RuntimeError("the builder's bootstrap tensors have an unexpected shape")
    return (components, label_group_counts, outcome.point_macro.reshape(S * M),
            outcome.replicates.reshape(int(replicates), S * M), outcome.diagnostics)


def interval(draws: np.ndarray, point: float) -> dict:
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"difference": round(float(point), 4), "ci95": [round(float(low), 4), round(float(high), 4)],
            "excludes_zero": bool(low > 0 or high < 0),
            "direction": "left_higher" if low > 0 else ("right_higher" if high < 0 else "interval_includes_zero")}


def two_stage_contrast(point: np.ndarray, reps: np.ndarray, left: int, right: int) -> dict:
    return interval(reps[:, left] - reps[:, right], point[left] - point[right])


# ------------------------------------------------------------------ scoring
def fold_wise_centroid_scores(dense, li, gi, w, L, fold, transform: str) -> np.ndarray:
    """identity_probe_v2: a fold's transform is fitted on the other folds' songs and only that
    fold's queries are scored in it, against profiles from all other songs."""
    scores = np.zeros((len(li), L))
    for k in range(FOLDS):
        train = fold != k
        queries = np.flatnonzero(~train)
        mean, matrix, _ = fit_transform(transform, dense[train], li[train], w[train], np.random.default_rng(SEED + k))
        scores[queries] = dense_leave_group_out(unit_rows((dense - mean) @ matrix.T), li, gi, w, L, queries)
    return scores


def check_published(name: str, rr: np.ndarray, w: np.ndarray) -> dict:
    """The reproduction check, run the moment a system is scored: SystemExit if the published
    corpus-v3 number is not recovered under the unchanged configuration."""
    expected_qw, tolerance, expected_cw, source = EXPECTED[name]
    all_mask = np.ones(len(rr), dtype=bool)
    qw = float(np.mean(rr))
    cw = weighted_mean(rr, w, all_mask)
    entry = {"published_query_weighted_mrr": expected_qw, "recomputed_query_weighted_mrr": round(qw, 4),
             "gap": round(abs(qw - expected_qw), 4), "tolerance": tolerance, "source": source}
    ok = abs(qw - expected_qw) <= tolerance
    line = f"check {name:34s} MRR {qw:.4f} published {expected_qw:.4f} (gap {abs(qw - expected_qw):.4f}, tol {tolerance})"
    if expected_cw is not None:
        entry.update({"published_component_weighted_mrr": expected_cw,
                      "recomputed_component_weighted_mrr": round(cw, 4),
                      "component_weighted_gap": round(abs(cw - expected_cw), 4)})
        ok = ok and abs(cw - expected_cw) <= FOLD_GAP
        line += f"  component-weighted {cw:.4f} published {expected_cw:.4f}"
    entry["passes"] = bool(ok)
    print(line + ("" if ok else "  FAILED"), flush=True)
    if not ok:
        raise SystemExit(f"the published number does not reproduce for {name}: {entry}")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()
    if corpus_version() != "v3":
        raise SystemExit("set CHINESE_RAP_CORPUS=v3; this file reads corpus v3 only")
    if EXPECTED_WORD_MRR is None:
        raise SystemExit("three_spaces.json is missing or stale, so the word-space check has no target")
    import jieba
    jieba.setLogLevel(60)
    private_root = args.private_root.resolve()

    d = setup(private_root)
    li, gi, w, L, fold, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"], d["dense"]
    n = len(li)
    everything = np.arange(n)
    group_count = int(len(np.unique(gi)))
    comp_of, comp_label = component_index(li, gi)
    component_count = int(len(comp_label))
    print(f"{n:,} queries, {L} labels, {group_count:,} groups, {component_count:,} components; "
          f"fold sizes {np.bincount(fold, minlength=FOLDS).tolist()}", flush=True)

    scores: dict[str, np.ndarray] = {}
    ranks: dict[str, np.ndarray] = {}
    rr: dict[str, np.ndarray] = {}
    checks: dict[str, dict] = {}

    def score_done(name: str) -> None:
        ranks[name] = ranks_of(scores[name], li)
        rr[name] = 1.0 / ranks[name]
        checks[name] = check_published(name, rr[name], w)

    # ---- the three raw spaces, by the builder's own scorer
    print("raw spaces", flush=True)
    documents = [d["documents"][s] for s in d["songs"]]
    chars = v1.fit_tfidf(documents)
    profiles_c = v1.score_leave_group_out(dense.astype(np.float32), chars, li, gi, L)
    builder_components = int(profiles_c.label_group_counts.sum())
    if builder_components != component_count:
        raise SystemExit(f"{component_count} components here against the builder's {builder_components}")
    words, _ = fit_words([" ".join(segment(doc)) for doc in documents])
    profiles_w = v1.score_leave_group_out(dense.astype(np.float32), words, li, gi, L)
    dense_gap_calls = float(np.abs(profiles_c.dense - profiles_w.dense).max())
    dense_scores = dense_leave_group_out(dense, li, gi, w, L, everything)
    dense_gap = float(np.abs(dense_scores - profiles_c.dense.astype(np.float64)).max())
    print(f"  dense scorer against the builder: max gap {dense_gap:.3e}; builder's dense arm across calls {dense_gap_calls:.3e}", flush=True)
    if dense_gap > 1e-4 or dense_gap_calls > 1e-6:
        raise SystemExit("the dense scorer does not reproduce the builder")
    scores["semantic_raw"] = dense_scores
    scores["characters_raw"] = profiles_c.lexical.astype(np.float64)
    scores["words_raw"] = profiles_w.lexical.astype(np.float64)
    for name in RAW_SPACES:
        score_done(name)
    del profiles_c, profiles_w, chars

    # ---- fold-wise transforms of the centroids (identity_probe_v2)
    print("fold-wise transforms of the centroids", flush=True)
    for name, transform in (("semantic_total_whitened", "total_whitening"),
                            ("semantic_within_whitened", "within_author_whitening"),
                            ("semantic_within_whitened_permuted", "within_author_whitening_permuted_labels")):
        scores[name] = fold_wise_centroid_scores(dense, li, gi, w, L, fold, transform)
        score_done(name)

    # ---- the word SVD, whitened within author (word_space_probe_v2)
    print("word SVD-1024, whitened within author", flush=True)
    scores["words_svd_within_whitened"] = np.zeros((n, L))
    svd_variance = []
    for k in range(FOLDS):
        train = fold != k
        queries = np.flatnonzero(~train)
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(words[train])
        svd_variance.append(round(float(svd.explained_variance_ratio_.sum()), 4))
        reduced = unit_rows(svd.transform(words))
        mean, matrix, _ = fit_transform("within_author_whitening", reduced[train], li[train], w[train], np.random.default_rng(SEED + k))
        scores["words_svd_within_whitened"][queries] = dense_leave_group_out(
            unit_rows((reduced - mean) @ matrix.T), li, gi, w, L, queries)
        print(f"  fold {k}: {svd_variance[-1]:.3f} of variance kept", flush=True)
    score_done("words_svd_within_whitened")
    del words

    # ---- chunk vectors whitened before averaging (representation_unit_v2 / word_space_probe_v2)
    print("semantic space, chunk vectors whitened before averaging", flush=True)
    rows, vectors, _ = load_v3(private_root)
    chunks_by_song, _, _, _, centroids_by_song = build_songs(rows, vectors)
    if any(s not in centroids_by_song for s in d["songs"]):
        raise SystemExit("the second load does not contain every query song")
    centroid_gap = float(np.abs(v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in d["songs"]])).astype(np.float64) - dense).max())
    print(f"  second load against setup(): centroid max gap {centroid_gap:.3e}", flush=True)
    if centroid_gap > 1e-6:
        raise SystemExit("the second load of the corpus does not reproduce setup()")
    song_pos = {s: i for i, s in enumerate(d["songs"])}
    chunk_rows = [i for s in d["songs"] for i in sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"]))]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    chunk_vec = unit_rows(vectors[chunk_rows].astype(np.float64))
    chunk_label = li[chunk_song]
    chunk_weight = w[chunk_song] / np.bincount(chunk_song)[chunk_song]
    chunk_fold = fold[chunk_song]
    chunk_count = len(chunk_rows)
    del rows, vectors, chunks_by_song, centroids_by_song
    semantic = np.zeros((n, L))
    for k in range(FOLDS):
        train = chunk_fold != k
        mean, matrix, _ = fit_transform("within_author_whitening", chunk_vec[train], chunk_label[train],
                                        chunk_weight[train], np.random.default_rng(SEED + k))
        projected = unit_rows((chunk_vec - mean) @ matrix.T)
        song_vec = np.zeros((n, projected.shape[1]))
        np.add.at(song_vec, chunk_song, projected)
        song_vec = unit_rows(song_vec / np.bincount(chunk_song)[:, None])
        queries = np.flatnonzero(fold == k)
        semantic[queries] = dense_leave_group_out(song_vec, li, gi, w, L, queries)
        print(f"  fold {k}", flush=True)
    scores["semantic_whitened_chunks"] = semantic
    score_done("semantic_whitened_chunks")
    del chunk_vec

    # ---- the published best system
    scores["fusion_whitened_words_semantic"] = (v1.zscore_rows(scores["words_svd_within_whitened"])
                                                + v1.zscore_rows(scores["semantic_whitened_chunks"])) / 2.0
    score_done("fusion_whitened_words_semantic")
    if set(scores) != set(SYSTEMS) or set(checks) != set(SYSTEMS):
        raise SystemExit("the systems scored are not the systems declared")
    print("reproduction check passed for every system", flush=True)

    # ---- the published group-bootstrap contrasts must come back from these ranks
    all_mask = np.ones(n, dtype=bool)
    pairs = list(CONFIRMATORY) + list(SUPPLEMENTARY)
    group_mrr = {(c["system"], c["minus"]): c for c in paired_group_bootstrap(rr, w, gi, all_mask, pairs)}
    hit1 = {name: (ranks[name] <= 1).astype(np.float64) for name in SYSTEMS}
    hit10 = {name: (ranks[name] <= 10).astype(np.float64) for name in SYSTEMS}
    group_r1 = {(c["system"], c["minus"]): c for c in paired_group_bootstrap(hit1, w, gi, all_mask, pairs)}
    published_contrasts = {}
    for left, right, published, source in PUBLISHED_GROUP_CONTRASTS:
        got = group_mrr[(left, right)]
        gap = abs(got["mrr_difference"] - published)
        published_contrasts[f"{left} - {right}"] = {"published_mrr_difference": published, "recomputed": got["mrr_difference"],
                                                     "recomputed_ci95": got["ci95"], "gap": round(gap, 4), "source": source,
                                                     "passes": bool(gap <= FOLD_GAP)}
        print(f"check contrast {left} - {right}: {got['mrr_difference']:+.4f} [{got['ci95'][0]:+.4f}, {got['ci95'][1]:+.4f}] "
              f"published {published:+.4f} (gap {gap:.4f})", flush=True)
        if gap > FOLD_GAP:
            raise SystemExit(f"the published group-bootstrap contrast {left} - {right} does not reproduce")

    # ---- estimands and the builder's two-stage bootstrap
    metric_values = {name: {"mrr": rr[name], "recall_at_1": hit1[name], "recall_at_10": hit10[name]} for name in SYSTEMS}
    column = {(name, metric): s * len(METRICS) + m
              for s, name in enumerate(SYSTEMS) for m, metric in enumerate(METRICS)}   # system-major, metric-minor
    values = np.column_stack([metric_values[name][metric] for name in SYSTEMS for metric in METRICS])
    table = estimand_table(values, w, li, L)
    print(f"two-stage bootstrap by the builder's functions: {TWO_STAGE_REPLICATES} replicates, seed {TWO_STAGE_SEED}, "
          f"{component_count:,} components", flush=True)
    components, label_group_counts, point, reps, diagnostics = two_stage(metric_values, li, gi, SYSTEMS, METRICS)
    own = per_label_components(component_means(values, comp_of, component_count), comp_label, L)
    if int(label_group_counts.sum()) != component_count or len(components) != L or \
            any(len(a) != len(b) for a, b in zip(components, own)):
        raise SystemExit("the builder's component tensors do not match this file's components")
    component_gap = max(float(np.abs(a.reshape(len(a), -1) - b).max()) for a, b in zip(components, own))
    macro_gap = float(np.abs(point - np.asarray(table["label_macro"])).max())
    print(f"  builder's component tensors against this file's component means: max gap {component_gap:.2e}", flush=True)
    print(f"  label-macro from the ranks against the builder's point_macro: max gap {macro_gap:.2e}", flush=True)
    if component_gap > 1e-12 or macro_gap > 1e-9:
        raise SystemExit("the two-stage bootstrap's point is not the label-macro estimand of these ranks")
    del components, own

    systems = {}
    for name in SYSTEMS:
        entry = {}
        for metric in METRICS:
            c = column[(name, metric)]
            low, high = np.quantile(reps[:, c], [0.025, 0.975])
            entry[metric] = {"query_weighted": round(table["plain_all"][c], 4),
                             "component_weighted": round(table["component_weighted"][c], 4),
                             "label_macro": round(table["label_macro"][c], 4),
                             "label_macro_ci95_two_stage": [round(float(low), 4), round(float(high), 4)]}
        entry["published"] = {"query_weighted_mrr": EXPECTED[name][0], "source": EXPECTED[name][3]}
        systems[name] = entry
        m = entry["mrr"]
        print(f"  {name:34s} MRR query {m['query_weighted']:.4f}  component {m['component_weighted']:.4f}  "
              f"label-macro {m['label_macro']:.4f} [{m['label_macro_ci95_two_stage'][0]:.4f}, {m['label_macro_ci95_two_stage'][1]:.4f}]  "
              f"R@1 query {entry['recall_at_1']['query_weighted']:.4f} label-macro {entry['recall_at_1']['label_macro']:.4f}", flush=True)

    # ---- contrasts under both resamplings
    print("contrasts", flush=True)
    contrasts = []
    readings = {}
    for left, right in pairs:
        role = "confirmatory" if (left, right) in CONFIRMATORY else "supplementary"
        g = group_mrr[(left, right)]
        g1 = group_r1[(left, right)]
        macro = {metric: two_stage_contrast(point, reps, column[(left, metric)], column[(right, metric)]) for metric in METRICS}
        established = macro["mrr"]["excludes_zero"]
        reading = "established under label resampling" if established else "not established under label resampling"
        agree = bool(established == g["excludes_zero"] and
                     (not established or np.sign(macro["mrr"]["difference"]) == np.sign(g["mrr_difference"])))
        entry = {"system": left, "minus": right, "role": role,
                 "query_weighted_mrr_difference": round(float(np.mean(rr[left]) - np.mean(rr[right])), 4),
                 "component_weighted_group_bootstrap": {
                     "mrr": {"difference": g["mrr_difference"], "ci95": g["ci95"], "excludes_zero": g["excludes_zero"]},
                     "recall_at_1": {"difference": g1["mrr_difference"], "ci95": g1["ci95"], "excludes_zero": g1["excludes_zero"]}},
                 "label_macro_two_stage": macro,
                 "reading": reading,
                 "resamplings_agree_on_mrr": agree}
        contrasts.append(entry)
        readings[f"{left} - {right}"] = reading + ("" if agree else " (the group bootstrap reads it differently)")
        print(f"  {left} - {right} [{role}]", flush=True)
        print(f"     component-weighted MRR, group bootstrap:   {g['mrr_difference']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]", flush=True)
        print(f"     component-weighted R@1, group bootstrap:   {g1['mrr_difference']:+.4f} [{g1['ci95'][0]:+.4f}, {g1['ci95'][1]:+.4f}]", flush=True)
        for metric in METRICS:
            mm = macro[metric]
            print(f"     label-macro {metric:12s}, two-stage:     {mm['difference']:+.4f} [{mm['ci95'][0]:+.4f}, {mm['ci95'][1]:+.4f}]", flush=True)
        print(f"     {readings[f'{left} - {right}']}", flush=True)

    # ---- the reconciliation table for the three raw spaces
    identity_estimands = {}
    for name in RAW_SPACES:
        identity_estimands[name] = {metric: {"plain_all": round(table["plain_all"][column[(name, metric)]], 4),
                                             "component_weighted": round(table["component_weighted"][column[(name, metric)]], 4),
                                             "label_macro": round(table["label_macro"][column[(name, metric)]], 4)}
                                    for metric in METRICS}
        m = identity_estimands[name]["mrr"]
        print(f"  estimands {name:16s} plain {m['plain_all']:.4f}  component {m['component_weighted']:.4f}  label-macro {m['label_macro']:.4f}", flush=True)

    payload = {
        "analysis": "the headline contrasts under the retrieval builder's two-stage bootstrap and label-macro estimand, corpus v3",
        "question": "which published contrasts hold when labels (artists) are resampled, read on the label-macro MRR and R@1, "
                    "and how far apart the query-weighted, component-weighted and label-macro readings of each system are",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L, "groups": group_count,
                   "components": component_count, "chunks_in_queries": chunk_count},
        "design": {
            "systems": {
                "semantic_raw": "BGE-M3 song centroids, leave-group-out cosine (the builder's dense arm)",
                "characters_raw": "character 2-5-gram TF-IDF, leave-group-out cosine (the builder's lexical arm)",
                "words_raw": "jieba word 1-2-gram TF-IDF, the same scorer",
                "semantic_total_whitened": "total-covariance whitening of the centroids, fitted per fold on the training folds (identity_probe_v2)",
                "semantic_within_whitened": "within-author whitening of the centroids, the same folds",
                "semantic_within_whitened_permuted": "within-author whitening fitted on permuted training labels, the same folds and seeds (identity_probe_v2's null)",
                "words_svd_within_whitened": f"word TF-IDF reduced by SVD-{SVD_COMPONENTS} fitted on the training folds, then within-author whitening (word_space_probe_v2)",
                "semantic_whitened_chunks": "within-author whitening fitted on chunk vectors, whitened chunks averaged into songs (representation_unit_v2)",
                "fusion_whitened_words_semantic": "equal-weight z-score fusion of words_svd_within_whitened and semantic_whitened_chunks (the published best system)",
            },
            "folds": {"count": FOLDS, "seed": SEED, "unit": "leakage group", "sizes": np.bincount(fold, minlength=FOLDS).tolist(),
                      "svd_variance_kept_by_fold": svd_variance},
            "estimands": {
                "query_weighted": "mean over queries (the published numbers)",
                "component_weighted": "each (leakage group, label) component weight one (the estimand of every published interval)",
                "label_macro": "component-weighted mean within each label, then the plain mean over labels (the retrieval builder's estimand)",
            },
            "resampling": {
                "component_weighted_group_bootstrap": "identity_spaces_v2.paired_group_bootstrap: 2000 replicates, seed 20260825, leakage groups drawn "
                                                      "with replacement, paired across systems; MRR and R@1",
                "label_macro_two_stage": f"the builder's own v1.component_metric_values and v1.run_bootstrap, called through the constant redirect "
                                         f"build_downstream_retrieval_v2 uses: {TWO_STAGE_REPLICATES} replicates, seed {TWO_STAGE_SEED}; {L} label "
                                         f"occurrences drawn with replacement, then for every occurrence an independent draw with replacement of that "
                                         f"label's components, each component valued by the mean over its songs; MRR, R@1 and R@10",
                "two_stage_diagnostics": diagnostics,
            },
            "contrasts": {"confirmatory": [f"{a} - {b}" for a, b in CONFIRMATORY], "supplementary": [f"{a} - {b}" for a, b in SUPPLEMENTARY]},
            "reading_rule": "a contrast is 'established under label resampling' if its two-stage 95% interval for the label-macro MRR difference excludes zero, "
                            "and 'not established under label resampling' otherwise; every contrast is reported either way, with its label-macro R@1 and R@10 "
                            "intervals and its group-bootstrap component-weighted intervals beside it; a disagreement between the two resamplings is named, not resolved",
            "checks": "query-weighted MRR of every system reproduces its published corpus-v3 number as soon as it is scored (gap <= 5e-4 for the raw spaces, "
                      "<= 0.002 fold-wise; published component-weighted MRR within 0.002 where one exists); dense scorer reproduces the builder (<= 1e-4), "
                      "the builder's dense arm is identical across calls, and the component count equals the builder's; the second corpus load reproduces "
                      "setup()'s centroids (<= 1e-6); the builder's component tensors equal this file's component means (<= 1e-12) and its point_macro equals "
                      "the label-macro estimand of the ranks (<= 1e-9); the three published group-bootstrap contrasts on these ranks reproduce within 0.002; "
                      "the synthetic unit test checks the component means against v1's own function, the two-stage wrapper against a transcription of "
                      "v1.run_bootstrap, the fold-wise scorer against identity_probe_v2's loop, and the estimands and intervals against brute force",
        },
        "checks": {"against_published": checks, "published_group_bootstrap_contrasts": published_contrasts,
                   "dense_scorer_max_gap": dense_gap, "dense_arm_across_calls_max_gap": dense_gap_calls,
                   "components_equal_builder": True, "second_load_centroid_max_gap": centroid_gap,
                   "builder_component_tensors_max_gap": component_gap,
                   "label_macro_against_two_stage_point_max_gap": macro_gap},
        "systems": systems,
        "contrasts": contrasts,
        "readings": readings,
        "identity_estimands": {"definitions": {"plain_all": "each query weight one", "component_weighted": "each (group, label) component weight one",
                                               "label_macro": "component-weighted mean within each label, then the mean over labels"},
                               "queries": n, "labels": L, "systems": identity_estimands},
        "privacy": "aggregate only",
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / OUT_NAME).write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nwrote {args.out_dir / OUT_NAME}  ({payload['minutes']} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
