#!/usr/bin/env python3
"""Does classical stylometry match the word space, and is words-over-characters an artefact
of one untuned TF-IDF configuration?

The published lexical numbers come from two TF-IDF spaces whose settings were never tuned:
jieba words with word bigrams 0.4963 MRR, character 2-5-grams 0.4266, both min_df 3,
max_features 150,000, sublinear tf, fitted on all songs. Two things a stylometrist would ask.

QUESTION A. Authorship attribution has its own standard scorer, Burrows's Delta (Burrows
2002) and its best-performing variant Cosine Delta (Smith and Aldridge 2011; Evert et al.
2017): z-scored relative frequencies of the most frequent words (MFW). How does it compare
with the published word TF-IDF space under the same leave-group-out protocol?

  tokens    the word space's own unigrams: jieba default cut (word_identity_anatomy_v2.
            segment), lower-cased, every non-space token kept (punctuation included, as in
            the word TF-IDF space), so the comparison changes the representation and not
            the tokeniser
  folds     identity_probe_v2's: leakage groups dealt into five folds, seed 20260825
  MFW       per fold, the N unigrams with the highest document frequency over the
            training-fold songs (ties: the higher total count in the training folds, then
            vocabulary order), N pre-fixed in 100, 300, 500, 1000, 2000, 5000; the lists
            are nested, so a smaller N is a prefix of a larger one
  features  relative frequency = count / the song's total unigram tokens; z = (rf - mean) /
            sd, mean and sd fitted on the training-fold songs with the protocol's
            per-(group, label) weights; the transform is applied to every song and only the
            fold's own queries are scored in it
  profiles  leave-group-out over ALL songs as in the protocol: label l's profile for query q
            is built from l's songs outside q's whole leakage group, protocol-weighted
  scorers
    cosine_delta            EXACT definition (the primary scorer): cos(z_q, m_l), m_l the
                            weighted MEAN z-vector of the label's remaining songs. z is
                            linear in relative frequency, so m_l is the z-vector of the
                            label's mean relative-frequency profile, the Delta 'author
                            profile'. Scored as the leave-group-out cosine with the weighted
                            sum of the RAW z-rows (the query's norm does not change a rank).
    cosine_delta_unit_rows  the protocol's prototype in z-space: z-rows unit-normalised
                            first, then cosine with the weighted sum (what lgo_parts of
                            label_size_calibration_v3 computes on unit rows). Secondary.
    burrows_delta           minus the mean absolute z difference against m_l:
                            -(1/N) sum_j |z_qj - m_lj|

QUESTION B. Is the ordering words > characters sensitive to the TF-IDF configuration?
  grid      max_features in {150000, 500000, None} x sublinear_tf in {True, False} x n-gram
            range: characters (1,3), (2,4), (2,5); words unigrams, (1,2). min_df 3, smoothed
            IDF and l2 rows stay fixed. 18 character and 12 word configurations; where the
            cap does not bind, configurations that differ only in the cap are one matrix and
            are scored once (recorded as cap_active false).
  arithmetic  sklearn's TfidfVectorizer reproduced on fixed count matrices (one character
            1-5-gram and one word 1-2-gram count matrix, column subsets per range), so that a
            fit on a subset of songs costs seconds: document frequency, min_df, the
            max_features cap by total term count with sklearn's own expression, smoothed IDF,
            sublinear tf, l2 rows.
  systems   (every one scored leave-group-out over all songs by the protocol's prototype)
    <space>_current                   the published configuration, fitted on all songs
    <space>_inner_selected            per fold k, the configuration with the highest INNER
                                      MRR (below), fitted on all songs as the protocol does,
                                      scored on fold k's queries; the five folds pooled
    <space>_current_inductive         the published configuration with vocabulary and IDF
                                      fitted on the training folds only (the precedent of
                                      build_retrieval_inductive_sensitivity_v1), applied to
                                      every song, fold k's queries scored
    <space>_inner_selected_inductive  the selected configuration fitted the same way: nothing
                                      of fold k enters the fit or the selection
    words_grid_worst, chars_grid_best the lowest word cell and the highest character cell of
                                      the all-songs grid, chosen on the evaluated queries
                                      AGAINST the ordering (description only)
    <space>_best_sublinear_tf, <space>_best_raw_tf
                                      the highest all-songs cell of a space within one tf
                                      setting, chosen on the evaluated queries (description
                                      only). NOT pre-registered: added after a first complete
                                      run showed that raw tf costs both spaces about a third
                                      of their MRR, so that a worst-against-best contrast
                                      compares tf settings rather than spaces; the ordering is
                                      therefore also shown within each tf setting.
  inner selection  for fold k only training-fold songs exist: vocabulary and IDF are fitted
            on them, the queries are the training-fold songs, the profiles are leave-group-
            out over the training-fold songs, the candidates are the labels that keep at
            least two (group, label) components in the training folds (a label with one
            would be emptied by its own query), and the criterion is the plain mean
            reciprocal rank. Ties: the published configuration, then grid order. A
            description beside it: the configuration that the all-songs grid would select
            from the training-fold QUERIES alone (profiles then include fold k's songs).

Estimands. MRR is the plain mean of reciprocal ranks over the 7,220 queries (the published
'mrr'), ranks by the builder's rule (analyse_identity_encoder_v3.ranks_of); a query whose row
is empty under a fit gets the worst rank. Every contrast carries a paired group bootstrap
interval (identity_spaces_v2.paired_group_bootstrap: leakage groups resampled with
replacement, per-(group, label) weights, 2000 replicates, seed 20260825); as in every
published contrast its point and interval are of the per-(group, label)-weighted MRR, which
is reported beside the plain mean as mrr_component_weighted. The selection is not repeated
inside the bootstrap. Reported contrasts: every Delta scorer and size minus the published
word space; the same minus the word grid's unigram TF-IDF cell (description: same tokens, so
it separates the Delta representation from the word bigrams); the systems of question B
against each other; every cell of the all-songs grid minus its space's published cell.

Checks, before any new number is used (SystemExit on failure)
  0. CHINESE_RAP_CORPUS=v3; the literals 0.4266 and 0.4963 are the ones three_spaces.json
     publishes for the current corpus build; the population is identity_probe.json's
     (7,220 queries, 226 labels, 5,875 groups, its fold sizes).
  1. REPRODUCTION, unchanged configuration, the protocol's own code: v1.fit_tfidf and
     word_identity_anatomy_v2.fit_words scored by v1.score_leave_group_out must give
     characters 0.4266 and words 0.4963, gap <= 5e-4.
  2. the count-matrix TF-IDF equals the protocol's matrices under the published
     configuration, both spaces: same shape, elementwise gap <= 1e-5.
  3. the vectorised leave-group-out scorer equals the protocol's sparse scorer on every
     query in both spaces (gap <= 1e-4, the reference is float32), equals lgo_parts on 300
     queries of the word space and dense_leave_group_out on a z-space fold (gap <= 1e-9).
  4. the vectorised Burrows scorer equals its brute-force definition on six queries of the
     first fold at every MFW size (gap <= 1e-9).
  5. the grid's own cells for the published configuration give 0.4266 and 0.4963
     (gap <= 5e-4). Every fit's rows are the training folds by construction (recorded as such);
     the evidence that selection and features are blind to the test fold is the synthetic test.
  The synthetic test (test_cosine_delta_and_tfidf_grid.py) checks on random data, without
  the corpus: both scorers against brute-force definitions and the protocol's scorers, the
  count-matrix TF-IDF against sklearn's TfidfVectorizer for every grid cell with binding and
  non-binding caps (transductive and inductive), the MFW/z features against a literal loop,
  that fold k's selection and features are blind to fold k, the tie rule, the reading rule
  on constructed intervals, and an end-to-end run whose JSON holds no token.

READING RULE, fixed before the run
  A. For each Delta scorer the best size is the one of the six pre-fixed MFW sizes with the
     highest MRR. Delta 'matches' the word space if the best size of the primary scorer
     (cosine_delta) reaches 0.4963 - 0.02 = 0.4763 or more; otherwise it 'does not match',
     and the paired interval of best-size minus words says by how much. The other two
     scorers are read by the same threshold and named if either reaches it when the primary
     does not. Choosing the best of six sizes on the evaluated queries favours Delta: a
     'does not match' is conservative, a 'matches' carries that selection.
  B. The ordering words > characters is 'configuration-robust' if words_inner_selected minus
     chars_inner_selected is positive with the 95% interval clear of zero; otherwise
     'configuration-sensitive'. The same contrast under inductive fitting
     (words_inner_selected_inductive - chars_inner_selected_inductive) is reported beside it
     and named if it disagrees. words_grid_worst - chars_grid_best and the two within-tf
     contrasts are description and do not enter the verdict.
  Scope. Delta was designed for texts of thousands of words; a song here has a few hundred
  tokens. Question A compares scorers under this protocol and says nothing about Delta on
  long texts.

    set CHINESE_RAP_CORPUS=v3
    python src/cosine_delta_and_tfidf_grid_v3.py --private-root <ni-k>
    python src/cosine_delta_and_tfidf_grid_v3.py --private-root <ni-k> --check-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

if os.environ.get("CHINESE_RAP_CORPUS") != "v3":
    raise SystemExit("run with CHINESE_RAP_CORPUS=v3: the corpus loader and the published v3 literals depend on it")


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from exemplar_vs_prototype_v3 import setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out  # noqa: E402
from identity_spaces_v2 import REPLICATES, paired_group_bootstrap  # noqa: E402
from label_size_calibration_v3 import lgo_parts  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = REPO / "results" / "retrieval-v3"
OUT_DIR = RESULTS_DIR
OUT_NAME = "cosine_delta_and_tfidf_grid.json"

EXPECTED = {"chars": 0.4266, "words": 0.4963}
EXPECTED_SOURCE = {"chars": "lexical_char_2_5", "words": "lexical_words"}       # three_spaces.json systems
EXPECTED_POPULATION = {"queries": 7220, "labels": 226, "groups": 5875}
CHECK_GAP = 5e-4
MATRIX_GAP = 1e-5
SCORER_GAP_FLOAT32 = 1e-4
SCORER_GAP = 1e-9

MFW_SIZES = (100, 300, 500, 1000, 2000, 5000)
DELTA_SCORERS = ("cosine_delta", "cosine_delta_unit_rows", "burrows_delta")
PRIMARY_DELTA = "cosine_delta"
MATCH_BAND = 0.02

MIN_DF = 3
RANGES = {"chars": (("1-3", 1, 3), ("2-4", 2, 4), ("2-5", 2, 5)), "words": (("1", 1, 1), ("1-2", 1, 2))}
MAX_FEATURES = (150_000, 500_000, None)
SUBLINEAR = (True, False)
CURRENT = {"chars": ("2-5", 150_000, True), "words": ("1-2", 150_000, True)}
UNIGRAM_COMPARATOR = ("1", 150_000, True)                          # the word grid's unigram cell with the published cap and tf
SPACES = ("chars", "words")


# ------------------------------------------------------------------ small helpers
def log(message: str) -> None:
    print(message, flush=True)


def config_name(config) -> str:
    ngram, cap, sublinear = config
    return f"ngram {ngram} | max_features {'none' if cap is None else cap} | sublinear_tf {str(bool(sublinear)).lower()}"


def grid_of(space: str, max_features=MAX_FEATURES):
    """Grid order: range, then cap, then sublinear; the tie rule falls back on this order."""
    return [(name, cap, sub) for name, _, _ in RANGES[space] for cap in max_features for sub in SUBLINEAR]


def row_sumsq(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.multiply(x).sum(axis=1)).ravel()
    return np.einsum("ij,ij->i", x, x)


def components(label_index: np.ndarray, group_ids: np.ndarray, label_count: int):
    """The protocol's (group, label) components, numbered by group and then label."""
    key = group_ids.astype(np.int64) * int(label_count) + label_index.astype(np.int64)
    unique, comp_of = np.unique(key, return_inverse=True)
    return comp_of.astype(np.int64), (unique % label_count).astype(np.int64), (unique // label_count).astype(np.int64)


def _holdout_pairs(label_index, group_ids, weights, label_count, queries):
    """Every (query, component) pair whose component lies in the query's leakage group: the
    profiles a held-out group changes. Queries must be whole leakage groups."""
    n = len(label_index)
    queries = np.asarray(queries, dtype=np.int64)
    in_queries = np.zeros(n, dtype=bool)
    in_queries[queries] = True
    touched = np.zeros(int(group_ids.max()) + 1, dtype=bool)
    touched[group_ids[queries]] = True
    if not np.array_equal(touched[group_ids], in_queries):
        raise RuntimeError("the queries must be whole leakage groups")
    comp_of, comp_label, comp_group = components(label_index, group_ids, label_count)
    needed = np.unique(comp_of[queries])                               # ascending: by group, then label
    needed_group = comp_group[needed]
    start = np.searchsorted(needed_group, group_ids[queries], side="left")
    stop = np.searchsorted(needed_group, group_ids[queries], side="right")
    reps = stop - start
    if np.any(reps < 1):
        raise RuntimeError("a query without its own component")
    pair_query = np.repeat(np.arange(len(queries)), reps)
    pair_comp = np.repeat(start, reps) + (np.arange(int(reps.sum())) - np.repeat(np.cumsum(reps) - reps, reps))
    position = np.full(len(comp_label), -1, dtype=np.int64)
    position[needed] = np.arange(len(needed))
    pick = sparse.csr_matrix((weights[queries], (position[comp_of[queries]], np.arange(len(queries)))),
                             shape=(len(needed), len(queries)))       # component sums over the queries' rows
    own_weight = np.asarray(pick.sum(axis=1)).ravel()
    return queries, pair_query, pair_comp, comp_label[needed], position[comp_of[queries]], pick, own_weight


def fast_lgo(x, label_index, group_ids, weights, label_count, queries) -> np.ndarray:
    """The protocol's prototype score for every (query, label), vectorised: the dot of the
    query's row with the label's weighted sum of rows outside the query's leakage group,
    divided by that sum's norm. x is a dense array or a CSR matrix; rows need not be unit
    length. The held-out sum is expanded: for a component c of label l in the query's group,
    ||S_l - c||^2 = ||S_l||^2 - 2 S_l.c + ||c||^2 with S_l.c = sum_m w_m (x_m . S_l)."""
    n = len(label_index)
    queries, pair_query, pair_comp, needed_label, comp_pos_of_query, pick, own_weight = _holdout_pairs(
        label_index, group_ids, weights, label_count, queries)
    member = sparse.csr_matrix((weights, (label_index, np.arange(n))), shape=(label_count, n))
    sums = member @ x
    xq = x[queries]
    own = pick @ xq
    if sparse.issparse(x):
        sums = sums.tocsr()
        dots = np.asarray((xq @ sums.T).todense(), dtype=np.float64)
        cross = np.asarray(xq[pair_query].multiply(own.tocsr()[pair_comp]).sum(axis=1)).ravel()
    else:
        sums = np.asarray(sums, dtype=np.float64)
        own = np.asarray(own, dtype=np.float64)
        dots = np.asarray(xq @ sums.T, dtype=np.float64)
        cross = np.einsum("ij,ij->i", xq[pair_query], own[pair_comp])
    norm2 = row_sumsq(sums)
    own_norm2 = row_sumsq(own)
    counts = np.bincount(label_index, weights=weights, minlength=label_count)
    if np.any(norm2 <= 1e-24):
        raise RuntimeError("an empty label profile")
    sum_dot_own = np.bincount(comp_pos_of_query,
                              weights=weights[queries] * dots[np.arange(len(queries)), label_index[queries]],
                              minlength=len(own_weight))
    label = needed_label[pair_comp]
    left_count = counts[label] - own_weight[pair_comp]
    left_norm2 = norm2[label] - 2.0 * sum_dot_own[pair_comp] + own_norm2[pair_comp]
    if np.any(left_count <= 1e-9) or np.any(left_norm2 <= 1e-12):
        raise RuntimeError("a held-out group empties a label profile")
    scores = dots / np.sqrt(norm2)[None, :]
    scores[pair_query, label] = (dots[pair_query, label] - cross) / np.sqrt(left_norm2)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("non-finite scores")
    return scores


def burrows_lgo(z: np.ndarray, sizes, label_index, group_ids, weights, label_count, queries) -> dict[int, np.ndarray]:
    """Burrows's Delta against leave-group-out label means, as a score (minus the distance),
    for every MFW size at once: the columns of z are ordered so that size N is the first N."""
    sizes = tuple(sizes)
    if list(sizes) != sorted(set(sizes)) or sizes[-1] > z.shape[1]:
        raise ValueError("sizes must be increasing and no larger than the z-matrix")
    n = len(label_index)
    queries, pair_query, pair_comp, needed_label, _, pick, own_weight = _holdout_pairs(
        label_index, group_ids, weights, label_count, queries)
    member = sparse.csr_matrix((weights, (label_index, np.arange(n))), shape=(label_count, n))
    width = sizes[-1]
    z = z[:, :width]
    sums = np.asarray(member @ z, dtype=np.float64)
    counts = np.bincount(label_index, weights=weights, minlength=label_count)
    if np.any(counts <= 0):
        raise RuntimeError("an empty label")
    zq = z[queries]
    own = np.asarray(pick @ zq, dtype=np.float64)
    out = {size: np.empty((len(queries), label_count)) for size in sizes}

    def prefix_means(absolute):
        previous, running, result = 0, np.zeros(absolute.shape[0]), {}
        for size in sizes:
            running = running + absolute[:, previous:size].sum(axis=1)
            result[size] = running / size
            previous = size
        return result

    for label in range(label_count):
        for size, value in prefix_means(np.abs(zq - (sums[label] / counts[label])[None, :])).items():
            out[size][:, label] = -value
    label = needed_label[pair_comp]
    left_count = counts[label] - own_weight[pair_comp]
    if np.any(left_count <= 1e-9):
        raise RuntimeError("a held-out group empties a label profile")
    for begin in range(0, len(pair_query), 512):
        rows = slice(begin, begin + 512)
        means = (sums[label[rows]] - own[pair_comp[rows]]) / left_count[rows, None]
        for size, value in prefix_means(np.abs(zq[pair_query[rows]] - means)).items():
            out[size][pair_query[rows], label[rows]] = -value
    return out


def burrows_brute(z, size, label_index, group_ids, weights, label_count, query) -> np.ndarray:
    """The definition, one query: minus the mean absolute z difference against the weighted
    mean of the label's songs outside the query's leakage group."""
    scores = np.empty(label_count)
    for label in range(label_count):
        keep = (label_index == label) & (group_ids != group_ids[query])
        mean = (z[keep, :size] * weights[keep, None]).sum(axis=0) / weights[keep].sum()
        scores[label] = -float(np.mean(np.abs(z[query, :size] - mean)))
    return scores


def reciprocal_ranks(scores, truth, label_count, undefined=None):
    ranks = ranks_of(scores, truth)
    if undefined is not None and np.any(undefined):
        ranks = ranks.copy()
        ranks[undefined] = label_count                               # the worst rank, not a tie broken by label order
    return 1.0 / ranks


def summary(rr: np.ndarray, weights=None) -> dict:
    out = {"mrr": round(float(np.mean(rr)), 4), "recall_at_1": round(float(np.mean(rr >= 1.0)), 4),
           "recall_at_10": round(float(np.mean(rr >= 0.1)), 4)}
    if weights is not None:
        out["mrr_component_weighted"] = round(float(np.sum(rr * weights) / np.sum(weights)), 4)
    return out


# ------------------------------------------------------------------ count matrices and TF-IDF
def build_counts(documents: list[str], segmented: list[str]) -> dict:
    """One character 1-5-gram and one word 1-2-gram count matrix (sklearn's analysers and
    lower-casing, float32 as TfidfVectorizer holds them), and the column subset of every
    n-gram range of the grid. Subsets keep sklearn's alphabetical feature order."""
    out = {"chars": {}, "words": {}}
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 5), dtype=np.float32)
    counts = vectorizer.fit_transform(documents).tocsr()
    length = np.zeros(counts.shape[1], dtype=np.int8)
    for term, index in vectorizer.vocabulary_.items():
        length[index] = len(term)
    del vectorizer
    for name, low, high in RANGES["chars"]:
        out["chars"][name] = counts[:, np.flatnonzero((length >= low) & (length <= high))].tocsr()
    del counts
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\S+", dtype=np.float32)
    counts = vectorizer.fit_transform(segmented).tocsr()
    length = np.zeros(counts.shape[1], dtype=np.int8)
    for term, index in vectorizer.vocabulary_.items():
        length[index] = 1 + term.count(" ")
    del vectorizer
    for name, low, high in RANGES["words"]:
        out["words"][name] = counts[:, np.flatnonzero((length >= low) & (length <= high))].tocsr()
    return out


def vocabulary(counts_fit, max_features, min_df=MIN_DF):
    """sklearn's CountVectorizer._limit_features and TfidfTransformer.fit on a count matrix:
    the kept-column mask, the smoothed IDF (zero on dropped columns) and what happened."""
    width = counts_fit.shape[1]
    df = np.bincount(counts_fit.indices, minlength=width)
    mask = df >= min_df
    passing = int(mask.sum())
    if passing == 0:
        raise RuntimeError("no feature passes min_df")
    cap_active = max_features is not None and passing > max_features
    if cap_active:
        tfs = np.asarray(counts_fit.sum(axis=0)).ravel()
        mask_inds = (-tfs[mask]).argsort()[:max_features]            # sklearn's expression, its tie order
        capped = np.zeros(width, dtype=bool)
        capped[np.where(mask)[0][mask_inds]] = True
        mask = capped
    idf = np.zeros(width)
    idf[mask] = np.log((counts_fit.shape[0] + 1.0) / (df[mask] + 1.0)) + 1.0
    return mask, idf, {"features": int(mask.sum()), "features_passing_min_df": passing, "cap_active": bool(cap_active)}


def weigh(counts_apply, idf, sublinear_tf):
    """TfidfTransformer.transform: (sublinear) tf x idf, l2 rows. Returns the CSR matrix
    (dropped columns left empty) and the mask of rows that came out empty."""
    x = counts_apply.astype(np.float64)
    if x is counts_apply:
        x = x.copy()
    if sublinear_tf:
        np.log(x.data, out=x.data)
        x.data += 1.0
    x.data *= idf[x.indices]
    x.eliminate_zeros()
    norms = np.sqrt(row_sumsq(x))
    empty = norms <= 0
    x.data /= np.repeat(np.where(empty, 1.0, norms), np.diff(x.indptr))
    return x.tocsr(), empty


def inner_problem(train_rows, label_index, group_ids, weights, label_count):
    """The training folds as a closed retrieval problem: the songs whose label keeps at least
    two (group, label) components there, labels renumbered in their original order."""
    mass = np.bincount(label_index[train_rows], weights=weights[train_rows], minlength=label_count)
    kept_labels = np.flatnonzero(mass >= 2.0 - 1e-9)
    renumber = np.full(label_count, -1, dtype=np.int64)
    renumber[kept_labels] = np.arange(len(kept_labels))
    rows = train_rows[renumber[label_index[train_rows]] >= 0]
    return rows, renumber[label_index[rows]], len(kept_labels)


def select_config(inner_mrr: dict, grid, current):
    """The highest inner MRR; ties go to the published configuration, then to grid order."""
    best = max(inner_mrr[c] for c in grid)
    tied = [c for c in grid if inner_mrr[c] >= best - 1e-12]
    return current if current in tied else tied[0]


def grid_cells(counts_by_range, fit_rows, apply_rows, grid, scorer):
    """Score every cell of a grid for one fit. scorer(x, empty_rows) -> reciprocal ranks.
    Cells that differ only in a cap that does not bind are one matrix, scored once."""
    cells, cache = {}, {}
    for ngram in dict.fromkeys(c[0] for c in grid):
        counts = counts_by_range[ngram]
        counts_fit = counts if fit_rows is None else counts[fit_rows]
        counts_apply = counts if apply_rows is None else counts[apply_rows]
        for cap in dict.fromkeys(c[1] for c in grid if c[0] == ngram):
            mask, idf, info = vocabulary(counts_fit, cap)
            for sublinear in dict.fromkeys(c[2] for c in grid if c[0] == ngram and c[1] == cap):
                key = (ngram, cap if info["cap_active"] else "all", sublinear)
                if key not in cache:
                    x, empty = weigh(counts_apply, idf, sublinear)
                    cache[key] = (scorer(x, empty), int(empty.sum()))
                rr, empty_rows = cache[key]
                cells[(ngram, cap, sublinear)] = {"rr": rr, "info": {**info, "empty_rows": empty_rows}}
    return cells


# ------------------------------------------------------------------ question A
def delta_features(unigrams, train_mask, weights, sizes):
    """MFW by document frequency on the training folds, relative frequencies, z-scores with
    the training folds' weighted mean and sd. Returns the z-matrix of every song (columns in
    MFW order, as wide as the largest size) and aggregate facts per size."""
    width = sizes[-1]
    totals = np.asarray(unigrams.sum(axis=1)).ravel().astype(np.float64)
    if np.any(totals <= 0):
        raise RuntimeError("a song without tokens")
    fit = unigrams[np.flatnonzero(train_mask)]
    df = np.bincount(fit.indices, minlength=unigrams.shape[1])
    tf = np.asarray(fit.sum(axis=0)).ravel().astype(np.float64)
    if int((df > 0).sum()) < width:
        raise RuntimeError(f"only {int((df > 0).sum())} unigrams occur in the training folds, fewer than {width}")
    order = np.lexsort((np.arange(len(df)), -tf, -df))[:width]
    relative = np.asarray(unigrams[:, order].todense(), dtype=np.float64) / totals[:, None]
    share = weights[train_mask] / weights[train_mask].sum()
    mean = share @ relative[train_mask]
    sd = np.sqrt(share @ (relative[train_mask] - mean) ** 2)
    if np.any(sd <= 1e-15):
        raise RuntimeError("a most-frequent word with no variance in the training folds")
    z = (relative - mean) / sd
    facts = {"unigram_types_in_training_folds": int((df > 0).sum()),
             "document_frequency_of_the_last_word": {str(s): int(df[order[s - 1]]) for s in sizes},
             "mean_share_of_a_songs_tokens_covered": {str(s): round(float(relative[:, :s].sum(axis=1).mean()), 4) for s in sizes}}
    return z, facts


def question_a(unigrams, label_index, group_ids, weights, label_count, fold, sizes=MFW_SIZES, folds=FOLDS):
    n = len(label_index)
    rr = {f"{scorer}/{size}": np.zeros(n) for scorer in DELTA_SCORERS for size in sizes}
    fold_info, checks = {}, {"burrows_brute_force_gap": None, "dense_scorer_gap": None, "fit_rows_are_the_training_folds_by_construction": []}
    for k in range(folds):
        started = time.time()
        train_mask = fold != k
        test = np.flatnonzero(fold == k)
        checks["fit_rows_are_the_training_folds_by_construction"].append(bool(not np.any(fold[train_mask] == k)))
        z, facts = delta_features(unigrams, train_mask, weights, tuple(sizes))
        burrows = burrows_lgo(z, sizes, label_index, group_ids, weights, label_count, test)
        if k == 0:
            gap = 0.0
            for row in range(min(6, len(test))):
                for size in sizes:
                    brute = burrows_brute(z, size, label_index, group_ids, weights, label_count, int(test[row]))
                    gap = max(gap, float(np.abs(brute - burrows[size][row]).max()))
            checks["burrows_brute_force_gap"] = gap
            if gap > SCORER_GAP:
                raise SystemExit(f"the vectorised Burrows scorer differs from its definition: {gap:.3e}")
        for size in sizes:
            zn = z[:, :size]
            unit = zn / np.maximum(np.linalg.norm(zn, axis=1, keepdims=True), 1e-12)
            exact = fast_lgo(zn, label_index, group_ids, weights, label_count, test)
            prototype = fast_lgo(unit, label_index, group_ids, weights, label_count, test)
            if k == 0 and size == sizes[0]:
                reference = dense_leave_group_out(unit, label_index, group_ids, weights, label_count, test)
                checks["dense_scorer_gap"] = float(np.abs(reference - prototype).max())
                if checks["dense_scorer_gap"] > SCORER_GAP:
                    raise SystemExit(f"the vectorised scorer differs from dense_leave_group_out: {checks['dense_scorer_gap']:.3e}")
            rr[f"cosine_delta/{size}"][test] = reciprocal_ranks(exact, label_index[test], label_count)
            rr[f"cosine_delta_unit_rows/{size}"][test] = reciprocal_ranks(prototype, label_index[test], label_count)
            rr[f"burrows_delta/{size}"][test] = reciprocal_ranks(burrows[size], label_index[test], label_count)
        fold_info[str(k)] = {"training_songs": int(train_mask.sum()), "queries": int(len(test)), **facts}
        log(f"  delta fold {k}: {time.time() - started:.0f}s")
    if not all(checks["fit_rows_are_the_training_folds_by_construction"]):
        raise SystemExit("a Delta fit saw its test fold")
    return rr, fold_info, checks


def reading_a(systems: dict, contrasts: list, sizes=MFW_SIZES, words=EXPECTED["words"]) -> dict:
    threshold = round(words - MATCH_BAND, 4)
    out = {"threshold": threshold, "by_scorer": {}}
    for scorer in DELTA_SCORERS:
        best = max(sizes, key=lambda s: (systems[f"{scorer}/{s}"]["mrr"], -s))
        contrast = next(c for c in contrasts if c["system"] == f"{scorer}/{best}")
        out["by_scorer"][scorer] = {"best_size": int(best), "mrr": systems[f"{scorer}/{best}"]["mrr"],
                                    "matches": bool(systems[f"{scorer}/{best}"]["mrr"] >= threshold),
                                    "best_size_minus_words": contrast["mrr_difference"], "ci95": contrast["ci95"]}
    primary = out["by_scorer"][PRIMARY_DELTA]
    others = [s for s in DELTA_SCORERS if s != PRIMARY_DELTA and out["by_scorer"][s]["matches"]]
    if primary["matches"]:
        verdict = f"Delta matches the word space: {PRIMARY_DELTA} at {primary['best_size']} MFW gives {primary['mrr']}"
    else:
        verdict = (f"Delta does not match the word space: {PRIMARY_DELTA} is best at {primary['best_size']} MFW with "
                   f"{primary['mrr']}, below {threshold}")
        if others:
            verdict += "; but reaching the threshold: " + ", ".join(others)
    out["verdict"] = verdict
    return out


# ------------------------------------------------------------------ question B
def question_b(counts, label_index, group_ids, weights, label_count, fold, max_features=MAX_FEATURES,
               current=CURRENT, folds=FOLDS):
    n = len(label_index)
    everything = np.arange(n)
    rr, report, cell_rr = {}, {}, {}
    fit_checks = []
    for space in SPACES:
        started = time.time()
        grid = grid_of(space, max_features)
        if current[space] not in grid:
            raise RuntimeError("the published configuration is not a cell of the grid")

        def all_songs_scorer(x, empty):
            return reciprocal_ranks(fast_lgo(x, label_index, group_ids, weights, label_count, everything),
                                    label_index, label_count, empty)

        cells = grid_cells(counts[space], None, None, grid, all_songs_scorer)
        log(f"  {space}: all-songs grid, {len(grid)} cells, {time.time() - started:.0f}s")
        against_published = {c["system"]: c for c in paired_group_bootstrap(
            {config_name(c): cells[c]["rr"] for c in grid}, weights, group_ids, np.ones(n, dtype=bool),
            [(config_name(c), config_name(current[space])) for c in grid if c != current[space]])}
        table = [{"configuration": config_name(c), **cells[c]["info"], **summary(cells[c]["rr"], weights),
                  "published_configuration": c == current[space],
                  "minus_published": None if c == current[space] else
                  {key: against_published[config_name(c)][key] for key in ("mrr_difference", "ci95", "excludes_zero")}}
                 for c in grid]
        cell_rr[space] = {c: cells[c]["rr"] for c in grid}
        rr[f"{space}_current"] = cells[current[space]]["rr"]
        for system in ("inner_selected", "current_inductive", "inner_selected_inductive"):
            rr[f"{space}_{system}"] = np.zeros(n)
        by_fold = {}
        for k in range(folds):
            started = time.time()
            empty_inductive = {}
            train_rows = np.flatnonzero(fold != k)
            test = np.flatnonzero(fold == k)
            fit_checks.append(bool(not np.any(fold[train_rows] == k)))
            rows, inner_labels, inner_label_count = inner_problem(train_rows, label_index, group_ids, weights, label_count)
            inner_queries = np.arange(len(rows))

            def inner_scorer(x, empty):
                return reciprocal_ranks(fast_lgo(x, inner_labels, group_ids[rows], weights[rows], inner_label_count,
                                                 inner_queries), inner_labels, inner_label_count, empty)

            inner = grid_cells(counts[space], train_rows, rows, grid, inner_scorer)
            inner_mrr = {c: float(np.mean(inner[c]["rr"])) for c in grid}
            chosen = select_config(inner_mrr, grid, current[space])
            on_training_queries = {c: float(np.mean(cells[c]["rr"][train_rows])) for c in grid}
            chosen_all_songs_grid = select_config(on_training_queries, grid, current[space])
            rr[f"{space}_inner_selected"][test] = cells[chosen]["rr"][test]
            for system, config in (("current_inductive", current[space]), ("inner_selected_inductive", chosen)):
                if system == "inner_selected_inductive" and chosen == current[space]:
                    rr[f"{space}_{system}"][test] = rr[f"{space}_current_inductive"][test]
                    empty_inductive[system] = empty_inductive["current_inductive"]
                    continue
                matrix = counts[space][config[0]]
                _, idf, _ = vocabulary(matrix[train_rows], config[1])
                x, empty = weigh(matrix, idf, config[2])
                scores = fast_lgo(x, label_index, group_ids, weights, label_count, test)
                rr[f"{space}_{system}"][test] = reciprocal_ranks(scores, label_index[test], label_count, empty[test])
                empty_inductive[system] = int(np.asarray(empty[test]).sum())
            ordered = sorted(grid, key=lambda c: -inner_mrr[c])
            by_fold[str(k)] = {
                "inner_queries": int(len(rows)), "inner_labels": int(inner_label_count),
                "training_songs": int(len(train_rows)), "test_queries": int(len(test)),
                "selected": config_name(chosen), "selected_is_published": bool(chosen == current[space]),
                "selected_features": inner[chosen]["info"]["features"],
                "selected_cap_active": inner[chosen]["info"]["cap_active"],
                "inner_mrr_selected": round(inner_mrr[chosen], 4),
                "inner_mrr_published": round(inner_mrr[current[space]], 4),
                "inner_rank_of_published": int(1 + ordered.index(current[space])),
                "inner_mrr_range": [round(min(inner_mrr.values()), 4), round(max(inner_mrr.values()), 4)],
                "inner_mrr_by_cell": {config_name(c): round(inner_mrr[c], 4) for c in grid},
                "selected_from_training_queries_of_the_all_songs_grid": config_name(chosen_all_songs_grid),
                "test_fold_mrr_selected": round(float(np.mean(cells[chosen]["rr"][test])), 4),
                "test_fold_mrr_published": round(float(np.mean(cells[current[space]]["rr"][test])), 4),
                "empty_test_rows_inductive": empty_inductive,
            }
            log(f"  {space} fold {k}: selected [{config_name(chosen)}] inner {inner_mrr[chosen]:.4f} "
                f"(published {inner_mrr[current[space]]:.4f}), {time.time() - started:.0f}s")
        extreme = (min if space == "words" else max)(grid, key=lambda c: float(np.mean(cells[c]["rr"])))
        rr["words_grid_worst" if space == "words" else "chars_grid_best"] = cells[extreme]["rr"]
        best_within_tf = {}
        for sublinear, name in ((True, "best_sublinear_tf"), (False, "best_raw_tf")):
            best = max((c for c in grid if c[2] == sublinear), key=lambda c: float(np.mean(cells[c]["rr"])))
            rr[f"{space}_{name}"] = cells[best]["rr"]
            best_within_tf[name] = config_name(best)
        report[space] = {"all_songs_grid": table, "inner_selection_by_fold": by_fold,
                         "grid_extreme": {"which": "lowest cell" if space == "words" else "highest cell",
                                          "configuration": config_name(extreme)},
                         "best_cell_within_tf_setting": best_within_tf}
    if not all(fit_checks):
        raise SystemExit("an inner or inductive fit saw its test fold")
    return rr, report, {"fit_rows_are_the_training_folds_by_construction": fit_checks}, cell_rr


B_PAIRS = [
    ("words_current", "chars_current"),
    ("words_inner_selected", "chars_inner_selected"),
    ("words_current_inductive", "chars_current_inductive"),
    ("words_inner_selected_inductive", "chars_inner_selected_inductive"),
    ("words_inner_selected", "words_current"),
    ("chars_inner_selected", "chars_current"),
    ("words_current_inductive", "words_current"),
    ("chars_current_inductive", "chars_current"),
    ("words_inner_selected_inductive", "words_inner_selected"),
    ("chars_inner_selected_inductive", "chars_inner_selected"),
    ("words_grid_worst", "chars_grid_best"),
    ("words_best_sublinear_tf", "chars_best_sublinear_tf"),
    ("words_best_raw_tf", "chars_best_raw_tf"),
]
DESCRIPTION_ONLY = {"words_grid_worst", "chars_grid_best", "words_best_sublinear_tf", "chars_best_sublinear_tf",
                    "words_best_raw_tf", "chars_best_raw_tf"}


def reading_b(contrasts: list) -> dict:
    def find(left, right):
        return next(c for c in contrasts if (c["system"], c["minus"]) == (left, right))

    def holds(c):
        return bool(c["mrr_difference"] > 0 and c["ci95"][0] > 0)

    primary = find("words_inner_selected", "chars_inner_selected")
    inductive = find("words_inner_selected_inductive", "chars_inner_selected_inductive")
    verdict = "configuration-robust" if holds(primary) else "configuration-sensitive"
    text = (f"words > characters is {verdict}: inner-selected words minus inner-selected characters "
            f"{primary['mrr_difference']:+.4f} [{primary['ci95'][0]:+.4f}, {primary['ci95'][1]:+.4f}]")
    if holds(inductive) != holds(primary):
        text += (f"; the inductive contrast disagrees: {inductive['mrr_difference']:+.4f} "
                 f"[{inductive['ci95'][0]:+.4f}, {inductive['ci95'][1]:+.4f}]")
    return {"verdict": verdict, "inductive_agrees": holds(inductive) == holds(primary), "text": text}


# ------------------------------------------------------------------ checks against the published record
def published_literals() -> dict:
    three = json.loads((RESULTS_DIR / "three_spaces.json").read_text(encoding="utf-8"))
    if three.get("corpus", {}).get("content_sha256") != V3_CONTENT_SHA256:
        raise SystemExit("three_spaces.json was not computed on the current v3 build")
    for space, system in EXPECTED_SOURCE.items():
        if three["systems"][system]["mrr"] != EXPECTED[space]:
            raise SystemExit(f"the literal for {space} is not the published {three['systems'][system]['mrr']}")
    probe = json.loads((RESULTS_DIR / "identity_probe.json").read_text(encoding="utf-8"))
    for key, value in EXPECTED_POPULATION.items():
        if probe[key] != value:
            raise SystemExit(f"identity_probe.json records {key} {probe[key]}, not {value}")
    return {"fold_sizes": [int(v) for v in probe["design"]["fold_sizes"]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--check-only", action="store_true",
                        help="run checks 0-3 (the reproduction of the published numbers and the equalities) and stop")
    args = parser.parse_args()
    started = time.time()
    import jieba
    jieba.setLogLevel(60)

    record = published_literals()
    d = setup(args.private_root.resolve())
    li, gi, w, L, fold = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"]
    n = len(li)
    population = {"queries": n, "labels": int(L), "groups": int(len(np.unique(gi)))}
    fold_sizes = np.bincount(fold, minlength=FOLDS).tolist()
    log(f"{n:,} queries, {L} labels, {population['groups']:,} groups, fold sizes {fold_sizes}")
    if population != EXPECTED_POPULATION or fold_sizes != record["fold_sizes"]:
        raise SystemExit(f"the population is not the published one: {population}, folds {fold_sizes}")

    documents = [d["documents"][s] for s in d["songs"]]
    segmented = [" ".join(segment(text)) for text in documents]
    log(f"segmented ({time.time() - started:.0f}s)")

    # check 1: the published numbers, by the protocol's own code, before anything else
    checks = {"against_published": {}, "count_matrix_tfidf_gap": {}, "scorer_gap_against_protocol": {}}
    protocol = {}
    for space, fit in (("chars", lambda: v1.fit_tfidf(documents)), ("words", lambda: fit_words(segmented)[0])):
        matrix = fit()
        scores = v1.score_leave_group_out(d["dense"].astype(np.float32), matrix, li, gi, L).lexical
        value = float(np.mean(1.0 / ranks_of(scores.astype(np.float64), li)))
        gap = abs(value - EXPECTED[space])
        checks["against_published"][space] = {"expected": EXPECTED[space], "recomputed": round(value, 4), "gap": round(gap, 4)}
        log(f"check {space}: protocol code gives {value:.4f}, published {EXPECTED[space]}, gap {gap:.4f}")
        if gap > CHECK_GAP:
            raise SystemExit(f"the {space} space does not reproduce the published number: {value:.4f} vs {EXPECTED[space]}")
        protocol[space] = (matrix, scores)
    log(f"published numbers reproduced ({time.time() - started:.0f}s)")

    counts = build_counts(documents, segmented)
    del documents, segmented
    log("count matrices: " + ", ".join(f"{space} {name} {m.shape[1]:,} columns" for space in SPACES
                                       for name, m in counts[space].items()) + f" ({time.time() - started:.0f}s)")

    # checks 2 and 3: the count-matrix TF-IDF and the vectorised scorer against the protocol's
    everything = np.arange(n)
    for space in SPACES:
        ngram, cap, sublinear = CURRENT[space]
        mask, idf, info = vocabulary(counts[space][ngram], cap)
        x, empty = weigh(counts[space][ngram], idf, sublinear)
        matrix, reference = protocol[space]
        if info["features"] != matrix.shape[1] or np.any(empty):
            raise SystemExit(f"{space}: the count-matrix TF-IDF keeps {info['features']} features, the protocol {matrix.shape[1]}")
        difference = (x[:, np.flatnonzero(mask)] - matrix.astype(np.float64)).tocsr()
        gap_matrix = float(np.abs(difference.data).max()) if difference.nnz else 0.0
        mine = fast_lgo(x, li, gi, w, L, everything)
        gap_scores = float(np.abs(mine - reference.astype(np.float64)).max())
        checks["count_matrix_tfidf_gap"][space] = gap_matrix
        checks["scorer_gap_against_protocol"][space] = gap_scores
        log(f"check {space}: count-matrix TF-IDF gap {gap_matrix:.2e}, vectorised scorer gap {gap_scores:.2e}")
        if gap_matrix > MATRIX_GAP or gap_scores > SCORER_GAP_FLOAT32:
            raise SystemExit(f"{space}: the count-matrix TF-IDF or the vectorised scorer differs from the protocol's")
        if space == "words":
            dots, norm2, _ = lgo_parts(x, li, gi, w, L, everything[:300])
            checks["scorer_gap_against_lgo_parts"] = float(np.abs(dots / np.sqrt(norm2) - mine[:300]).max())
            log(f"check words: vectorised scorer against lgo_parts, gap {checks['scorer_gap_against_lgo_parts']:.2e}")
            if checks["scorer_gap_against_lgo_parts"] > SCORER_GAP:
                raise SystemExit("the vectorised scorer differs from lgo_parts")
        del x, mine, difference
    del protocol
    log(f"checks 0-3 passed ({time.time() - started:.0f}s)")
    if args.check_only:
        log("check-only: stopping before any new number; nothing written")
        return 0

    all_mask = np.ones(n, dtype=bool)

    # question B first: its published-configuration cells are the reference of both questions
    log("question B: the TF-IDF grid")
    rr_b, report_b, checks_b, cell_rr = question_b(counts, li, gi, w, L, fold)
    checks["grid_cells_for_the_published_configuration"] = {}
    for space in SPACES:
        value = float(np.mean(rr_b[f"{space}_current"]))
        checks["grid_cells_for_the_published_configuration"][space] = round(value, 4)
        log(f"check {space}: the grid's published cell gives {value:.4f}")
        if abs(value - EXPECTED[space]) > CHECK_GAP:
            raise SystemExit(f"the grid's published {space} cell gives {value:.4f}, not {EXPECTED[space]}")
    systems_b = {name: summary(values, w) for name, values in rr_b.items()}
    contrasts_b = paired_group_bootstrap(rr_b, w, gi, all_mask, B_PAIRS)
    verdict_b = reading_b(contrasts_b)
    for name, values in systems_b.items():
        log(f"  {name:36s} MRR {values['mrr']:.4f}")
    for c in contrasts_b:
        c["role"] = "description" if c["system"] in DESCRIPTION_ONLY else "contrast"
        log(f"    {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
            + ("  (description)" if c["role"] == "description" else ""))
    log(f"  reading B: {verdict_b['text']}")

    log("question A: Delta")
    rr_a, fold_info_a, checks_a = question_a(counts["words"]["1"], li, gi, w, L, fold)
    delta_systems = list(rr_a)
    rr_a["words_current"] = rr_b["words_current"]
    rr_a["words_unigram_tfidf"] = cell_rr["words"][UNIGRAM_COMPARATOR]
    systems_a = {name: summary(values, w) for name, values in rr_a.items()}
    contrasts_a = paired_group_bootstrap(rr_a, w, gi, all_mask, [(name, "words_current") for name in delta_systems])
    contrasts_unigram = paired_group_bootstrap(rr_a, w, gi, all_mask, [(name, "words_unigram_tfidf") for name in delta_systems]
                                               + [("words_unigram_tfidf", "words_current")])
    verdict_a = reading_a(systems_a, contrasts_a)
    for scorer in DELTA_SCORERS:
        log(f"  {scorer:24s} " + "  ".join(f"{size}: {systems_a[f'{scorer}/{size}']['mrr']:.4f}" for size in MFW_SIZES))
    for scorer, best in verdict_a["by_scorer"].items():
        log(f"    best {scorer} at {best['best_size']}: {best['mrr']:.4f}, minus words {best['best_size_minus_words']:+.4f} "
            f"[{best['ci95'][0]:+.4f}, {best['ci95'][1]:+.4f}]")
    best_names = {f"{scorer}/{best['best_size']}" for scorer, best in verdict_a["by_scorer"].items()}
    for c in contrasts_unigram:
        if c["system"] == "words_unigram_tfidf" or c["system"] in best_names:
            log(f"    (description) {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    log(f"  reading A: {verdict_a['verdict']}")

    payload = {
        "analysis": "Burrows's Delta and Cosine Delta against the word TF-IDF space; the word-over-character ordering over a TF-IDF grid with inner selection",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, **population, "fold_sizes": fold_sizes},
        "design": {
            "protocol": "leave-group-out prototype scoring over all songs; folds of identity_probe_v2 (leakage groups, seed 20260825); ranks by the builder's rule; MRR the plain mean of reciprocal ranks",
            "question_a": {
                "tokens": "jieba default cut, lower-cased, every non-space token (punctuation included), the word TF-IDF space's own unigrams",
                "mfw": "per fold, the N unigrams with the highest document frequency over the training-fold songs; ties by total count in the training folds, then vocabulary order; nested lists",
                "sizes": list(MFW_SIZES),
                "features": "relative frequency over the song's unigram tokens, z-scored with the protocol-weighted mean and sd of the training-fold songs",
                "scorers": {"cosine_delta": "cosine of the query's z-vector with the leave-group-out weighted mean z-vector of the label (exact definition; primary)",
                            "cosine_delta_unit_rows": "z-rows unit-normalised, then the protocol's prototype (cosine with the weighted sum)",
                            "burrows_delta": "minus the mean absolute z difference against the leave-group-out weighted mean z-vector"}},
            "question_b": {
                "grid": {"max_features": ["none" if c is None else c for c in MAX_FEATURES], "sublinear_tf": list(SUBLINEAR),
                         "character_ngram_ranges": [r[0] for r in RANGES["chars"]], "word_ngram_ranges": [r[0] for r in RANGES["words"]],
                         "fixed": f"min_df {MIN_DF}, smoothed idf, l2 rows"},
                "published_configuration": {space: config_name(CURRENT[space]) for space in SPACES},
                "inner_selection": "per fold, vocabulary and IDF fitted on the training-fold songs; queries, profiles and candidate labels from the training folds only (labels with at least two components there); criterion plain MRR; ties to the published configuration, then grid order",
                "inductive": "vocabulary and IDF fitted on the training folds only, applied to every song; profiles leave-group-out over all songs; the fold's queries scored",
                "cap": "cells that differ only in a max_features cap that does not bind are one matrix (cap_active false)"},
            "bootstrap": f"paired_group_bootstrap: {REPLICATES} replicates, seed {SEED}, leakage groups resampled with replacement, per-(group, label) weights; the selection is not repeated inside the bootstrap",
            "reading_rule": {
                "A": f"Delta matches the word space if the best of the six pre-fixed MFW sizes of {PRIMARY_DELTA} reaches {EXPECTED['words']} - {MATCH_BAND} or more; the other scorers are read by the same threshold and named",
                "B": "configuration-robust if words_inner_selected - chars_inner_selected is positive with the interval clear of zero; the inductive contrast is reported beside it and named if it disagrees; words_grid_worst - chars_grid_best and the within-tf contrasts are description",
                "not_pre_registered": "the within-tf description (<space>_best_sublinear_tf, <space>_best_raw_tf) was added after a first complete run showed that raw tf costs both spaces about a third of their MRR",
                "scope": "Delta was designed for texts of thousands of words; a song has a few hundred tokens; question A compares scorers under this protocol only"},
        },
        "checks": {**checks, "question_a": checks_a, "question_b": {"fit_rows_are_the_training_folds_by_construction": bool(all(checks_b["fit_rows_are_the_training_folds_by_construction"]))},
                   "tolerances": {"published": CHECK_GAP, "matrix": MATRIX_GAP, "scorer_float32_reference": SCORER_GAP_FLOAT32, "scorer": SCORER_GAP}},
        "question_a": {"systems": systems_a, "contrasts_against_words": contrasts_a,
                       "contrasts_against_the_word_unigram_tfidf_cell": {
                           "comparator": config_name(UNIGRAM_COMPARATOR),
                           "role": "description: the same tokens under TF-IDF, so the contrast separates the Delta representation from the word bigrams",
                           "contrasts": contrasts_unigram},
                       "fold_info": fold_info_a, "reading": verdict_a},
        "question_b": {"systems": systems_b, "paired_contrasts": contrasts_b, **report_b, "reading": verdict_b},
        "privacy": "aggregate only; no token, text, title, song id or vector",
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / OUT_NAME).write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    log(f"\nwrote {args.out_dir / OUT_NAME}  ({payload['minutes']} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
