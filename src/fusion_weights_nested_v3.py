#!/usr/bin/env python3
"""Were the published fusions selected on the evaluation queries, and does the best fused
system survive weights fitted inside the training folds?

Every published fusion in this project is an equal-weight z-score fusion: for a query, each
component's 226 label scores are standardised (build_chinese_rap_downstream_retrieval_v1.
zscore_rows) and the two rows are averaged. The weight 0.5 was fixed in the code and never
tuned, but it was never tested either: a fixed weight can happen to sit at the optimum of the
evaluation queries, and a fusion's margin over its better component has only ever been
measured with that fixed weight. This file fits the weight inside the training folds and
scores the test fold with it, for four pairs of spaces, all under the unchanged leave-group-
out protocol (7,220 song queries, 226 labels, 5,875 leakage groups, per-(group, label)
weight one; corpus v3 1.3.0).

Pairs (A with B; A is the word space in every pair)
  whitened_words_whitened_semantic  the word SVD-1024 within-author whitened per fold
                                    (0.5426) with the chunk-whitened-then-mean semantic
                                    space (0.4271): the published best system, 0.5854 at
                                    equal weights (word_space_probe.json)
  raw_words_raw_semantic            jieba word TF-IDF (0.4963) with the BGE-M3 song mean
                                    (0.2997); published equal-weight fusion 0.4894
  raw_words_raw_chars               jieba word TF-IDF with character 2-5-gram TF-IDF
                                    (0.4266); no published fusion
  raw_words_rhyme_strict            jieba word TF-IDF with the strict rhyme-form space
                                    (0.0954 over all queries with uncovered songs at the
                                    worst rank; 0.0967 on the 7,117 songs with at least four
                                    classifiable line endings). This pair is evaluated on
                                    that covered subset, where every row of both spaces is
                                    defined, so the surface space is words alone on the same
                                    queries.

Fusion and weight
  fused(w) = w * zscore(A) + (1 - w) * zscore(B), row-wise over a query's label scores;
  w on the grid 0.00, 0.05, ..., 1.00 (21 values). w = 0.5 is the published fusion,
  bitwise ((zA + zB) / 2; checked).

Nested design (identity_probe_v2 folds: leakage groups dealt into five folds, seed 20260825)
  outer fold k   the test fold. Its queries are scored in the published fold-k spaces: word
                 SVD-1024 and within-author whitening fitted on the four other folds (seed
                 SEED + k), chunk whitening fitted on the four other folds' chunks and the
                 whitened chunks averaged per song; the raw spaces need no fitting. Label
                 profiles are leave-group-out over all other songs, as in the protocol.
  inner folds    for the weight only. For outer fold k and each training fold j != k, the
                 fold-j queries are scored in a space fitted on the three folds outside
                 {k, j} (SVD and whitening seed SEED + 10 + 5a + b for the pair a < b; the
                 same fitted space serves outer k / inner j and outer j / inner k, so ten
                 inner fits cover the twenty (k, j) cases). For a raw space the inner scores
                 are its protocol scores. The weight for outer fold k is the grid value with
                 the highest per-(group, label)-weighted MRR over the training-fold queries'
                 inner scores (ties: the value nearest 0.5, then the smaller w). No fold-k
                 query's own score or rank enters the choice of its weight, and no fold-k
                 song enters the SVD or whitening behind the selection scores; fold-k songs
                 remain profile members for the other folds' queries, as every song is for
                 every other query under the protocol.
  applied        the chosen weight fuses the fold-k queries' published fold-k scores; the
                 test-fold rows of the five folds are pooled into one system, fusion_fitted.

Systems per pair, over the pair's evaluated queries
  A, B                          the components (their protocol scores)
  fusion_equal                  w = 0.5, the published fusion
  fusion_fitted                 the nested fusion above (the result)
  fusion_w_chosen_on_test_fold  w chosen on the test fold's own queries and applied to them:
                                what selecting on the evaluation queries would give
                                (description)
  fusion_w_chosen_on_all        one w chosen on all evaluated queries (description)
  fusion_w_from_outer_training  pairs with a fitted component only: w chosen on the training
                                folds' published (outer) scores, whose transforms saw the
                                fold-k songs; shows whether the inner fits change the choice
                                (description)
Estimands: MRR as the plain mean of reciprocal ranks (the published 'mrr'), the
per-(group, label)-weighted MRR (the bootstrap's estimand and the selection criterion),
recall@1 and recall@10. Every contrast carries a paired group-bootstrap interval
(identity_spaces_v2.paired_group_bootstrap: leakage groups resampled with replacement, 2000
replicates, seed 20260825) over the pair's evaluated queries:
  fusion_fitted - fusion_equal            fusion_equal - better component
  fusion_fitted - better component        fusion_fitted - other component
  fusion_w_chosen_on_test_fold - fusion_fitted, fusion_w_chosen_on_all - fusion_fitted and
  fusion_w_from_outer_training - fusion_fitted (description)

Checks, before any new number is used (SystemExit on failure)
  0. CHINESE_RAP_CORPUS=v3 is set; the literals below are matched, one by one, against the
     result file that publishes each (results/retrieval-v3/*.json).
  1. the population is the probe's: 7,220 queries, 226 labels, 5,875 groups, fold sizes as
     identity_probe.json records; 7,117 covered songs as identity_spaces.json records.
  2. the raw spaces reproduce the published numbers, gap <= 5e-4: semantic 0.2997,
     characters 0.4266, words 0.4963, rhyme strict 0.0954 under three_spaces_v3's rank
     policy (empty rows and songs under four endings at the worst rank) and 0.0967 on the
     covered subset under identity_spaces_v2's; the equal-weight raw words + raw semantic
     fusion 0.4894. The dense protocol scores are identical across the three sparse-scorer
     calls.
  3. the fold-wise spaces reproduce the published numbers, gap <= 0.002: whitened word SVD
     0.5426, chunk-whitened semantic 0.4271 and their equal-weight fusion 0.5854.
  4. fused(0.5) is bitwise the published (zA + zB) / 2; every outer fit's training rows
     exclude its fold and every inner fit's exclude both folds of its pair (recorded); the
     nested stage's fusion_equal reproduces the check's own number where all rows are
     evaluated.
  The synthetic test (test_fusion_weights_nested.py) checks on random data, without the
  corpus: the fold-wise scorers against a hand-fitted transform and a brute-force leave-
  group-out cosine; the grid curve and the weight selection against a brute-force grid
  search with the protocol's rank rule; that the chosen weight and the fitted test scores
  are blind to the test fold; the mask for undefined rows; the reading rule on constructed
  intervals; and an end-to-end run.

READING RULE, fixed before the run
  Contrasts are over each pair's evaluated queries: the test-fold queries of the five folds
  pooled, each fused with the weight fitted on its own training folds; the rhyme pair over
  the covered songs.
  1. A fusion 'adds' to its better component if fusion_fitted minus that component is
     positive with the 95% interval clear of zero; otherwise it does not add.
  2. Rhyme 'adds nothing beyond the surface' if words+rhyme (fusion_fitted) minus words is
     not positive with the interval clear of zero; otherwise rhyme adds beyond the surface.
  3. The published equal weight: if fusion_fitted minus fusion_equal is negative with the
     interval clear of zero, the published equal-weight number exceeds what a weight fitted
     without the evaluation queries achieves, and the fitted-weight number is the honest one
     to report; if positive with the interval clear of zero, the published number is
     conservative and honest weighting improves on it; if the interval covers zero, the
     published number stands as what honest weighting achieves.
  Description, not results: the chosen weights per fold, the training and test curves over
  the grid, whether the equal weight is the grid optimum of the evaluation queries, and the
  selection-inflation contrasts (w chosen on the test fold's own queries, on all queries, or
  on the training folds' outer scores, against fusion_fitted).

    set CHINESE_RAP_CORPUS=v3
    python src/fusion_weights_nested_v3.py --private-root <ni-k>

Output: results/retrieval-v3/fusion_weights_nested.json, aggregate only (no lyric text,
song identifiers, label names or per-song vectors).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if os.environ.get("CHINESE_RAP_CORPUS") != "v3":
    raise SystemExit("run with CHINESE_RAP_CORPUS=v3: the corpus loader and the published v3 literals depend on it")


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of, song_level  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import (  # noqa: E402
    MINIMUM_ENDINGS,
    REPLICATES,
    ending_sequence,
    fit_phonological,
    paired_group_bootstrap,
    phonological_tokens,
    weighted_mean,
)
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from word_identity_anatomy_v2 import EXPECTED_WORD_MRR, fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = REPO / "results" / "retrieval-v3"
OUT_DIR = RESULTS_DIR
OUT_NAME = "fusion_weights_nested.json"

GRID = tuple(round(float(x), 2) for x in np.linspace(0.0, 1.0, 21))    # 0.00, 0.05, ..., 1.00
EQUAL_W = 0.5
CHECK_GAP_RAW = 5e-4
CHECK_GAP_FITTED = 0.002
DENSE_GAP = 1e-6
VARIANCE_FLOOR = 1e-12                    # zscore_rows refuses a row at or below this

EXPECTED_POPULATION = {"queries": 7220, "labels": 226, "groups": 5875}
EXPECTED_COVERED = 7117
EXPECTED_RAW = {"semantic_raw": 0.2997, "chars_raw": 0.4266, "words_raw": 0.4963, "rhyme_strict": 0.0954,
                "rhyme_strict_covered": 0.0967, "fusion_raw_words_raw_semantic": 0.4894}
EXPECTED_FITTED = {"words_svd_whitened": 0.5426, "semantic_whitened_chunks": 0.4271,
                   "fusion_whitened_words_whitened_semantic": 0.5854}
# where each literal is published, so a stale literal cannot pass
PUBLISHED_SOURCES = {
    "semantic_raw": ("three_spaces.json", ("systems", "semantic", "mrr")),
    "chars_raw": ("three_spaces.json", ("systems", "lexical_char_2_5", "mrr")),
    "words_raw": ("three_spaces.json", ("systems", "lexical_words", "mrr")),
    "rhyme_strict": ("three_spaces.json", ("systems", "rhyme_form_strict", "mrr")),
    "rhyme_strict_covered": ("identity_spaces.json", ("systems", "phonological_strict", "covered_subset", "mrr")),
    "fusion_raw_words_raw_semantic": ("word_space_probe.json", ("systems", "fusion_raw_words_raw_semantic", "mrr")),
    "words_svd_whitened": ("word_space_probe.json", ("systems", "words_svd_within_author_whitening", "mrr")),
    "semantic_whitened_chunks": ("word_space_probe.json", ("systems", "semantic_whitened_chunks", "mrr")),
    "fusion_whitened_words_whitened_semantic": ("word_space_probe.json", ("systems", "fusion_whitened_words_whitened_semantic", "mrr")),
}
PUBLISHED_COUNTS = {"covered_queries": ("identity_spaces.json", ("phonological_space", "covered_queries")),
                    "fold_sizes": ("identity_probe.json", ("design", "fold_sizes"))}

# (pair name, A, B); A is the word space
PAIRS = (("whitened_words_whitened_semantic", "words_svd_whitened", "semantic_whitened_chunks"),
         ("raw_words_raw_semantic", "words_raw", "semantic_raw"),
         ("raw_words_raw_chars", "words_raw", "chars_raw"),
         ("raw_words_rhyme_strict", "words_raw", "rhyme_strict"))
RHYME_PAIR = "raw_words_rhyme_strict"
DESCRIPTION_SYSTEMS = ("fusion_w_chosen_on_test_fold", "fusion_w_chosen_on_all", "fusion_w_from_outer_training")


# ------------------------------------------------------------------ spaces
@dataclass
class Space:
    """A space's protocol scores. outer: (n, L), every query scored in its own fold's space
    (for a raw space, its plain protocol scores). inner: outer fold k -> (n, L) whose rows of
    folds j != k hold the fold-j queries' scores in the space fitted outside {k, j}; the
    fold-k rows are NaN and never read (for a raw space, inner[k] is outer)."""
    name: str
    outer: np.ndarray
    inner: dict
    defined: np.ndarray
    fits: dict = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return bool(self.fits.get("fitted", False))


def defined_rows(scores: np.ndarray) -> np.ndarray:
    return np.std(scores, axis=1) > VARIANCE_FLOOR


def raw_space(name: str, scores: np.ndarray) -> Space:
    scores = np.asarray(scores, dtype=np.float64)
    return Space(name, scores, {k: scores for k in range(FOLDS)}, defined_rows(scores), {"fitted": False})


def inner_seed(a: int, b: int) -> int:
    if not 0 <= a < b < FOLDS:
        raise ValueError((a, b))
    return SEED + 10 + 5 * a + b


def outer_scores(fit, label_index, group_ids, weights, label_count, fold, log=print):
    """Fold k's queries scored in the space fitted on the other folds (seed SEED + k): the
    published fold-wise scores. fit(train_mask, seed) -> (unit song matrix, info)."""
    n = len(label_index)
    outer = np.zeros((n, label_count))
    fits = {}
    for k in range(FOLDS):
        started = time.time()
        train = fold != k
        x, info = fit(train, SEED + k)
        seen = sorted(set(fold[train].tolist()))
        if k in seen:
            raise RuntimeError("an outer fit saw its own fold")
        queries = np.flatnonzero(~train)
        outer[queries] = dense_leave_group_out(x, label_index, group_ids, weights, label_count, queries)
        fits[str(k)] = {"training_rows": int(train.sum()), "folds_in_training": seen, "seed": SEED + k, **info}
        log(f"    outer fold {k}: {time.time() - started:.0f}s")
    return outer, fits


def inner_scores(fit, label_index, group_ids, weights, label_count, fold, log=print):
    """For every pair a < b: the space fitted on the folds outside {a, b} (seed inner_seed),
    folds a and b scored in it. inner[b] receives the fold-a rows (outer b, inner a) and
    inner[a] the fold-b rows."""
    n = len(label_index)
    inner = {k: np.full((n, label_count), np.nan) for k in range(FOLDS)}
    fits = {}
    for a in range(FOLDS):
        for b in range(a + 1, FOLDS):
            started = time.time()
            train = (fold != a) & (fold != b)
            x, info = fit(train, inner_seed(a, b))
            seen = sorted(set(fold[train].tolist()))
            if a in seen or b in seen:
                raise RuntimeError("an inner fit saw a fold of its pair")
            queries = np.flatnonzero(~train)
            scores = dense_leave_group_out(x, label_index, group_ids, weights, label_count, queries)
            in_a = fold[queries] == a
            inner[b][queries[in_a]] = scores[in_a]
            inner[a][queries[~in_a]] = scores[~in_a]
            fits[f"{a},{b}"] = {"training_rows": int(train.sum()), "folds_in_training": seen,
                                "seed": inner_seed(a, b), **info}
            log(f"    inner pair {a},{b}: {time.time() - started:.0f}s")
    for k in range(FOLDS):
        rows = fold != k
        if not np.all(np.isfinite(inner[k][rows])):
            raise RuntimeError("an inner score matrix has a missing training row")
        if np.any(np.isfinite(inner[k][~rows])):
            raise RuntimeError("an inner score matrix carries its own fold's rows")
    return inner, fits


def word_svd_fit(lexical, label_index, weights, components=SVD_COMPONENTS):
    """The published word SVD path: TruncatedSVD on the training rows, unit rows, within-
    author whitening fitted on the training rows, unit rows (word_space_probe_v2)."""
    from sklearn.decomposition import TruncatedSVD

    def fit(train, seed):
        svd = TruncatedSVD(n_components=components, random_state=seed).fit(lexical[train])
        reduced = unit_rows(svd.transform(lexical))
        mean, matrix, info = fit_transform("within_author_whitening", reduced[train], label_index[train],
                                           weights[train], np.random.default_rng(seed))
        return unit_rows((reduced - mean) @ matrix.T), {"svd_variance_kept": round(float(svd.explained_variance_ratio_.sum()), 4), **info}
    return fit


def chunk_whitening_fit(chunk_vec, chunk_song, chunk_label, chunk_weight, song_count):
    """The published chunk path: within-author whitening fitted on the training songs'
    chunks, applied to every chunk, the whitened chunks averaged per song, unit rows
    (word_space_probe_v2 / representation_unit_v2)."""
    def fit(train_songs, seed):
        train = train_songs[chunk_song]
        mean, matrix, info = fit_transform("within_author_whitening", chunk_vec[train], chunk_label[train],
                                           chunk_weight[train], np.random.default_rng(seed))
        projected = unit_rows((chunk_vec - mean) @ matrix.T)
        return song_level(projected, chunk_song, song_count), info
    return fit


# ------------------------------------------------------------------ fusion and selection
def fuse(za: np.ndarray, zb: np.ndarray, w: float) -> np.ndarray:
    return w * za + (1.0 - w) * zb


def published_equal_fusion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The published expression, verbatim (word_space_probe_v2, identity_probe_v2)."""
    return (v1.zscore_rows(np.asarray(a, dtype=np.float64)) + v1.zscore_rows(np.asarray(b, dtype=np.float64))) / 2.0


def weighted_mrr(rr: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(rr * weights) / np.sum(weights))


def grid_curve(za, zb, truth, weights, grid=GRID) -> list:
    """Per-(group, label)-weighted MRR of fused(w) for every w on the grid, protocol ranks."""
    return [weighted_mrr(1.0 / ranks_of(fuse(za, zb, w), truth), weights) for w in grid]


def choose_weight(curve, grid=GRID) -> float:
    """The grid value with the highest curve value; exact ties go to the value nearest 0.5,
    then to the smaller w."""
    best = max(curve)
    candidates = [w for w, value in zip(grid, curve) if value == best]
    return float(min(candidates, key=lambda w: (abs(w - EQUAL_W), w)))


def pair_mask(a: Space, b: Space, covered=None) -> np.ndarray:
    mask = a.defined & b.defined
    if covered is not None:
        mask = mask & covered
    return mask


def nested_fusion(a: Space, b: Space, label_index, group_ids, weights, fold, mask, grid=GRID):
    """The nested design. Returns ranks per system (int arrays over all queries; rows outside
    the mask carry label_count + 1 and are never reported) and per-fold description."""
    n, label_count = a.outer.shape
    if b.outer.shape != (n, label_count):
        raise ValueError("the two spaces score different populations")
    at = {w: i for i, w in enumerate(grid)}
    za = np.full((n, label_count), np.nan)
    zb = np.full((n, label_count), np.nan)
    za[mask] = v1.zscore_rows(a.outer[mask])
    zb[mask] = v1.zscore_rows(b.outer[mask])
    fitted = np.full((n, label_count), np.nan)
    on_test = np.full((n, label_count), np.nan)
    from_outer = np.full((n, label_count), np.nan)
    any_fitted = a.fitted or b.fitted
    folds = []
    for k in range(FOLDS):
        train = mask & (fold != k)
        test = mask & (fold == k)
        if train.sum() == 0 or test.sum() == 0:
            raise RuntimeError(f"fold {k} has no training or no test query under the mask")
        za_train = v1.zscore_rows(a.inner[k][train])
        zb_train = v1.zscore_rows(b.inner[k][train])
        if not (np.all(np.isfinite(za_train)) and np.all(np.isfinite(zb_train))):
            raise RuntimeError("an inner score of a training query is missing")
        curve_train = grid_curve(za_train, zb_train, label_index[train], weights[train], grid)
        w_hat = choose_weight(curve_train, grid)
        curve_test = grid_curve(za[test], zb[test], label_index[test], weights[test], grid)
        w_test = choose_weight(curve_test, grid)
        fitted[test] = fuse(za[test], zb[test], w_hat)
        on_test[test] = fuse(za[test], zb[test], w_test)
        entry = {"fold": k, "training_queries": int(train.sum()), "test_queries": int(test.sum()),
                 "chosen_w": w_hat,
                 "training_curve_weighted_mrr": [round(v, 4) for v in curve_train],
                 "test_curve_weighted_mrr": [round(v, 4) for v in curve_test],
                 "test_weighted_mrr_at_chosen_w": round(curve_test[at[w_hat]], 4),
                 "test_weighted_mrr_at_equal_w": round(curve_test[at[EQUAL_W]], 4),
                 "w_chosen_on_test_fold": w_test,
                 "test_weighted_mrr_at_w_chosen_on_test_fold": round(curve_test[at[w_test]], 4)}
        if any_fitted:
            curve_outer = grid_curve(za[train], zb[train], label_index[train], weights[train], grid)
            w_outer = choose_weight(curve_outer, grid)
            from_outer[test] = fuse(za[test], zb[test], w_outer)
            entry["w_chosen_on_outer_training_scores"] = w_outer
            entry["outer_training_curve_weighted_mrr"] = [round(v, 4) for v in curve_outer]
        folds.append(entry)
    equal = fuse(za[mask], zb[mask], EQUAL_W)
    if not np.array_equal(equal, published_equal_fusion(a.outer[mask], b.outer[mask])):
        raise RuntimeError("fused(0.5) is not bitwise the published (zA + zB) / 2")
    curve_all = grid_curve(za[mask], zb[mask], label_index[mask], weights[mask], grid)
    w_all = choose_weight(curve_all, grid)
    scores = {"fusion_equal": equal, "fusion_fitted": fitted[mask], "fusion_w_chosen_on_test_fold": on_test[mask],
              "fusion_w_chosen_on_all": fuse(za[mask], zb[mask], w_all),
              a.name: a.outer[mask], b.name: b.outer[mask]}
    if any_fitted:
        scores["fusion_w_from_outer_training"] = from_outer[mask]
    ranks = {}
    for name, s in scores.items():
        if not np.all(np.isfinite(s)):
            raise RuntimeError(f"{name}: a non-finite score under the mask")
        r = np.full(n, label_count + 1, dtype=np.int64)
        r[mask] = ranks_of(s, label_index[mask])
        ranks[name] = r
    return {"ranks": ranks, "folds": folds, "chosen_w_by_fold": [f["chosen_w"] for f in folds],
            "w_chosen_on_all": w_all, "all_queries_curve_weighted_mrr": [round(v, 4) for v in curve_all],
            "equal_weight_is_grid_optimum_of_all_queries": bool(w_all == EQUAL_W)}


# ------------------------------------------------------------------ summaries and reading
def system_summary(ranks: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> dict:
    r = ranks[mask]
    rr = 1.0 / r
    return {"queries": int(mask.sum()), "mrr": round(float(np.mean(rr)), 4),
            "mrr_component_weighted": round(weighted_mrr(rr, weights[mask]), 4),
            "recall_at_1": round(float(np.mean(r <= 1)), 4), "recall_at_10": round(float(np.mean(r <= 10)), 4)}


def positive_clear(c: dict) -> bool:
    return c["mrr_difference"] > 0 and c["ci95"][0] > 0


def negative_clear(c: dict) -> bool:
    return c["mrr_difference"] < 0 and c["ci95"][1] < 0


def reading_of(pair_name: str, contrasts: list, better: str, a_name: str) -> dict:
    """The reading rule of the docstring applied to a pair's contrasts."""
    def get(system, minus):
        return next(c for c in contrasts if (c["system"], c["minus"]) == (system, minus))

    def text(c):
        return f"{c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"

    fit_minus_better = get("fusion_fitted", better)
    adds = positive_clear(fit_minus_better)
    out = {"fusion_adds_to_better_component": adds,
           "fusion_versus_better_component": (f"the fusion adds to {better} ({text(fit_minus_better)})" if adds
                                              else f"the fusion does not add to {better} ({text(fit_minus_better)})")}
    if pair_name == RHYME_PAIR:
        fit_minus_words = get("fusion_fitted", a_name)
        rhyme_adds = positive_clear(fit_minus_words)
        out["rhyme_adds_beyond_the_surface"] = rhyme_adds
        out["rhyme"] = (f"rhyme adds beyond the surface (words+rhyme minus words {text(fit_minus_words)})" if rhyme_adds
                        else f"rhyme adds nothing beyond the surface (words+rhyme minus words {text(fit_minus_words)})")
    fit_minus_equal = get("fusion_fitted", "fusion_equal")
    if negative_clear(fit_minus_equal):
        out["published_equal_weight_verdict"] = "exceeds_honest_weighting"
        out["published_equal_weight"] = (f"the published equal-weight fusion exceeds what a weight fitted without the "
                                         f"evaluation queries achieves (fitted minus equal {text(fit_minus_equal)}); "
                                         f"the fitted-weight number is the honest one to report")
    elif positive_clear(fit_minus_equal):
        out["published_equal_weight_verdict"] = "conservative"
        out["published_equal_weight"] = (f"the published equal-weight fusion is conservative; honest weighting improves "
                                         f"on it (fitted minus equal {text(fit_minus_equal)})")
    else:
        out["published_equal_weight_verdict"] = "stands"
        out["published_equal_weight"] = (f"the published equal-weight fusion stands as what honest weighting achieves "
                                         f"(fitted minus equal {text(fit_minus_equal)})")
    return out


def analyse_pair(pair_name, a: Space, b: Space, label_index, group_ids, weights, fold, mask, log=print) -> dict:
    out = nested_fusion(a, b, label_index, group_ids, weights, fold, mask)
    ranks = out["ranks"]
    rr = {name: np.where(mask, 1.0 / r, 0.0) for name, r in ranks.items()}
    systems = {name: system_summary(r, weights, mask) for name, r in ranks.items()}
    better, other = ((a.name, b.name) if weighted_mean(rr[a.name], weights, mask) >= weighted_mean(rr[b.name], weights, mask)
                     else (b.name, a.name))
    pairs = [("fusion_fitted", "fusion_equal"), ("fusion_fitted", better), ("fusion_equal", better),
             ("fusion_fitted", other), ("fusion_w_chosen_on_test_fold", "fusion_fitted"),
             ("fusion_w_chosen_on_all", "fusion_fitted")]
    if "fusion_w_from_outer_training" in ranks:
        pairs.append(("fusion_w_from_outer_training", "fusion_fitted"))
    contrasts = paired_group_bootstrap(rr, weights, group_ids, mask, pairs)
    for c in contrasts:
        c["description_only"] = c["system"] in DESCRIPTION_SYSTEMS
    reading = reading_of(pair_name, contrasts, better, a.name)
    log(f"== {pair_name}: queries {int(mask.sum()):,}; chosen w by fold {out['chosen_w_by_fold']}; "
        f"w on all queries {out['w_chosen_on_all']}"
        + (f"; w on outer training scores {[f['w_chosen_on_outer_training_scores'] for f in out['folds']]}"
           if "fusion_w_from_outer_training" in ranks else ""))
    for name, s in systems.items():
        log(f"    {name:30s} MRR {s['mrr']:.4f}  weighted {s['mrr_component_weighted']:.4f}  R@1 {s['recall_at_1']:.4f}  R@10 {s['recall_at_10']:.4f}")
    for c in contrasts:
        log(f"    {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
            + ("  (description)" if c["description_only"] else ""))
    for key in ("fusion_versus_better_component", "rhyme", "published_equal_weight"):
        if key in reading:
            log(f"    reading: {reading[key]}")
    return {"components": {"a": a.name, "b": b.name}, "better_component": better, "queries": int(mask.sum()),
            "systems": systems, "paired_contrasts": contrasts, "folds": out["folds"],
            "chosen_w_by_fold": out["chosen_w_by_fold"], "w_chosen_on_all_queries": out["w_chosen_on_all"],
            "equal_weight_is_grid_optimum_of_all_queries": out["equal_weight_is_grid_optimum_of_all_queries"],
            "all_queries_curve_weighted_mrr": out["all_queries_curve_weighted_mrr"], "reading": reading}


def analyse(d: dict, spaces: dict, masks: dict, pairs=PAIRS, log=print) -> dict:
    li, gi, w, fold = d["label_index"], d["group_ids"], d["weights"], d["fold"]
    return {name: analyse_pair(name, spaces[a], spaces[b], li, gi, w, fold, masks[name], log) for name, a, b in pairs}


# ------------------------------------------------------------------ corpus
def setup_with_chunks(private_root: Path) -> dict:
    """exemplar_vs_prototype_v3.setup, line for line, plus the chunk arrays of
    word_space_probe_v2 and the rhyme transcription of identity_spaces_v2 / three_spaces_v3."""
    rows, vectors, state = load_v3(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, centroids_by_song = build_songs(rows, vectors)
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {s for s in chunks_by_song
                   if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(l for l, m in songs_by_label.items()
                      if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
    label_index = np.asarray([eligible.index(label_by_song[s]) for s in songs], dtype=np.int64)
    groups = build_groups(songs, components_by_song, {s: normalise_document(documents[s]) for s in songs})
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    label_count = len(eligible)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs])).astype(np.float64)
    song_pos = {s: i for i, s in enumerate(songs)}
    ordered_chunks = {s: sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"])) for s in songs}
    chunk_rows = [i for s in songs for i in ordered_chunks[s]]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    chunk_vec = unit_rows(vectors[chunk_rows].astype(np.float64))
    chunk_weight = weights[chunk_song] / np.bincount(chunk_song)[chunk_song]
    sequences = {s: ending_sequence([rows[i]["cleaned_text"] for i in ordered_chunks[s]]) for s in songs}
    endings = np.asarray([len(sequences[s]) for s in songs])
    rhyme_docs = [phonological_tokens(sequences[s], with_finals=False) for s in songs]
    return dict(songs=songs, label_index=label_index, group_ids=group_ids, label_count=label_count, groups=len(order),
                weights=weights, fold=fold, dense=dense, documents=documents, chunk_song=chunk_song,
                chunk_vec=chunk_vec, chunk_weight=chunk_weight, endings=endings, rhyme_docs=rhyme_docs,
                embeddings_sha256=state["sha256"])


def published_value(file_name: str, keys: tuple):
    path = RESULTS_DIR / file_name
    if not path.is_file():
        raise SystemExit(f"published result missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        value = value[key]
    return value


def match_literals() -> dict:
    """The literals above must be what results/retrieval-v3 publishes."""
    matched = {}
    for name, (file_name, keys) in PUBLISHED_SOURCES.items():
        expected = EXPECTED_RAW.get(name, EXPECTED_FITTED.get(name))
        found = published_value(file_name, keys)
        if abs(float(found) - expected) > 1e-9:
            raise SystemExit(f"literal {name} = {expected} but {file_name} publishes {found}")
        matched[name] = {"file": file_name, "value": expected}
    covered = published_value(*PUBLISHED_COUNTS["covered_queries"])
    if int(covered) != EXPECTED_COVERED:
        raise SystemExit(f"covered count literal {EXPECTED_COVERED} but identity_spaces.json publishes {covered}")
    fold_sizes = [int(v) for v in published_value(*PUBLISHED_COUNTS["fold_sizes"])]
    return {"literals": matched, "fold_sizes": fold_sizes}


def plain_mrr(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / ranks))


def rhyme_ranks_published(scores: np.ndarray, matrix, truth: np.ndarray, covered: np.ndarray, label_count: int):
    """The two published rank policies for the strict rhyme space. three_spaces_v3 (via
    lexical_identity_anatomy_v2.score_arm): rows with no feature at the worst rank, then
    songs under MINIMUM_ENDINGS at the worst rank. identity_spaces_v2.ranks_for: only the
    uncovered songs at the worst rank, read on the covered subset."""
    base = ranks_of(scores, truth)
    empty = np.asarray(matrix.getnnz(axis=1) == 0)
    three_spaces = base.copy()
    three_spaces[empty] = label_count
    three_spaces[~covered] = label_count
    identity_spaces = base.copy()
    identity_spaces[~covered] = label_count
    return three_spaces, identity_spaces, int((empty & covered).sum())


def check_against(name: str, value: float, expected: float, gap: float, checks: dict, log=print) -> None:
    checks[name] = {"expected": expected, "recomputed": round(value, 4), "gap": round(abs(value - expected), 5), "allowed": gap}
    log(f"check {name}: recomputed {value:.4f} expected {expected} (gap {abs(value - expected):.5f}, allowed {gap})")
    if abs(value - expected) > gap:
        raise SystemExit(f"{name} does not reproduce the published number: {value:.4f} vs {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()

    def minutes() -> str:
        return f"{(time.time() - started) / 60:.1f} min"

    import jieba
    jieba.setLogLevel(60)

    published = match_literals()
    if EXPECTED_WORD_MRR is None or abs(float(EXPECTED_WORD_MRR) - EXPECTED_RAW["words_raw"]) > 1e-9:
        raise SystemExit(f"word_identity_anatomy_v2 expects {EXPECTED_WORD_MRR}, this file {EXPECTED_RAW['words_raw']}; "
                         f"is CHINESE_RAP_CORPUS=v3 and results/retrieval-v3/three_spaces.json current?")
    print("literals matched against results/retrieval-v3; loading corpus v3", flush=True)
    d = setup_with_chunks(args.private_root.resolve())
    li, gi, w, L, fold = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"]
    n = len(li)
    fold_sizes = np.bincount(fold, minlength=FOLDS).tolist()
    population = {"queries": n, "labels": L, "groups": d["groups"]}
    print(f"  {n:,} queries, {L} labels, {d['groups']:,} groups, {len(d['chunk_song']):,} chunks; fold sizes {fold_sizes}  ({minutes()})", flush=True)
    if population != EXPECTED_POPULATION or fold_sizes != published["fold_sizes"]:
        raise SystemExit(f"the population is not the probe's: {population}, folds {fold_sizes}")
    covered = d["endings"] >= MINIMUM_ENDINGS
    print(f"  {int(covered.sum()):,} songs with at least {MINIMUM_ENDINGS} classifiable endings", flush=True)
    if int(covered.sum()) != EXPECTED_COVERED:
        raise SystemExit(f"{int(covered.sum())} covered songs, published {EXPECTED_COVERED}")
    checks = {}

    # ------------------------------------------------------------ raw spaces under the protocol's own scorer
    print("raw spaces", flush=True)
    docs = [d["documents"][s] for s in d["songs"]]
    dense32 = d["dense"].astype(np.float32)
    chars = v1.fit_tfidf(docs)
    words, _ = fit_words([" ".join(segment(doc)) for doc in docs])
    rhyme, _ = fit_phonological(d["rhyme_docs"])
    print(f"  spaces fitted: characters {chars.shape[1]:,} features, words {words.shape[1]:,}, rhyme {rhyme.shape[1]:,}  ({minutes()})", flush=True)
    p_chars = v1.score_leave_group_out(dense32, chars, li, gi, L)
    p_words = v1.score_leave_group_out(dense32, words, li, gi, L)
    p_rhyme = v1.score_leave_group_out(dense32, rhyme, li, gi, L)
    dense_gap = max(float(np.abs(p_chars.dense - p_words.dense).max()), float(np.abs(p_chars.dense - p_rhyme.dense).max()))
    print(f"  dense protocol scores across the three calls: max gap {dense_gap:.1e}  ({minutes()})", flush=True)
    if dense_gap > DENSE_GAP:
        raise SystemExit("the dense scores differ between calls; the protocol is not shared")
    raw = {"semantic_raw": p_chars.dense.astype(np.float64), "chars_raw": p_chars.lexical.astype(np.float64),
           "words_raw": p_words.lexical.astype(np.float64), "rhyme_strict": p_rhyme.lexical.astype(np.float64)}
    for name in ("semantic_raw", "chars_raw", "words_raw"):
        check_against(name, plain_mrr(ranks_of(raw[name], li)), EXPECTED_RAW[name], CHECK_GAP_RAW, checks)
    rhyme_three, rhyme_identity, covered_but_empty = rhyme_ranks_published(raw["rhyme_strict"], rhyme, li, covered, L)
    print(f"  rhyme rows with no feature among the covered songs: {covered_but_empty}", flush=True)
    check_against("rhyme_strict", plain_mrr(rhyme_three), EXPECTED_RAW["rhyme_strict"], CHECK_GAP_RAW, checks)
    check_against("rhyme_strict_covered", plain_mrr(rhyme_identity[covered]), EXPECTED_RAW["rhyme_strict_covered"], CHECK_GAP_RAW, checks)
    check_against("fusion_raw_words_raw_semantic", plain_mrr(ranks_of(published_equal_fusion(raw["words_raw"], raw["semantic_raw"]), li)),
                  EXPECTED_RAW["fusion_raw_words_raw_semantic"], CHECK_GAP_RAW, checks)
    spaces = {name: raw_space(name, scores) for name, scores in raw.items()}
    del p_chars, p_words, p_rhyme, chars, rhyme

    # ------------------------------------------------------------ fold-wise spaces: the published outer fits first
    print(f"fold-wise spaces, outer (published) fits  ({minutes()})", flush=True)
    fit_words_svd = word_svd_fit(words, li, w)
    fit_chunks = chunk_whitening_fit(d["chunk_vec"], d["chunk_song"], li[d["chunk_song"]], d["chunk_weight"], n)
    print("  words svd whitened", flush=True)
    outer_w, fits_w = outer_scores(fit_words_svd, li, gi, w, L, fold)
    print("  semantic whitened chunks", flush=True)
    outer_s, fits_s = outer_scores(fit_chunks, li, gi, w, L, fold)
    check_against("words_svd_whitened", plain_mrr(ranks_of(outer_w, li)), EXPECTED_FITTED["words_svd_whitened"], CHECK_GAP_FITTED, checks)
    check_against("semantic_whitened_chunks", plain_mrr(ranks_of(outer_s, li)), EXPECTED_FITTED["semantic_whitened_chunks"], CHECK_GAP_FITTED, checks)
    check_against("fusion_whitened_words_whitened_semantic", plain_mrr(ranks_of(published_equal_fusion(outer_w, outer_s), li)),
                  EXPECTED_FITTED["fusion_whitened_words_whitened_semantic"], CHECK_GAP_FITTED, checks)
    print(f"reproduction checks passed  ({minutes()})", flush=True)

    print("fold-wise spaces, inner fits (three folds each)", flush=True)
    print("  words svd whitened", flush=True)
    inner_w, fits_w_inner = inner_scores(fit_words_svd, li, gi, w, L, fold)
    print("  semantic whitened chunks", flush=True)
    inner_s, fits_s_inner = inner_scores(fit_chunks, li, gi, w, L, fold)
    spaces["words_svd_whitened"] = Space("words_svd_whitened", outer_w, inner_w, defined_rows(outer_w),
                                         {"fitted": True, "outer": fits_w, "inner": fits_w_inner})
    spaces["semantic_whitened_chunks"] = Space("semantic_whitened_chunks", outer_s, inner_s, defined_rows(outer_s),
                                               {"fitted": True, "outer": fits_s, "inner": fits_s_inner})
    del words
    print(f"inner fits done  ({minutes()})", flush=True)

    # ------------------------------------------------------------ the nested fusions
    masks = {name: pair_mask(spaces[a], spaces[b], covered if name == RHYME_PAIR else None) for name, a, b in PAIRS}
    undefined = {name: int((~spaces[name].defined).sum()) for name in spaces}
    for name, a, b in PAIRS:
        print(f"  {name}: {int(masks[name].sum()):,} evaluated queries ({undefined[a]} undefined rows in {a}, {undefined[b]} in {b})", flush=True)
    print("nested fusions", flush=True)
    results = analyse(d, spaces, masks)
    # the nested stage's equal-weight fusion must be the check's own number where every row is evaluated
    for pair_name, check_name in (("whitened_words_whitened_semantic", "fusion_whitened_words_whitened_semantic"),
                                  ("raw_words_raw_semantic", "fusion_raw_words_raw_semantic")):
        if results[pair_name]["queries"] == n:
            got = results[pair_name]["systems"]["fusion_equal"]["mrr"]
            if abs(got - checks[check_name]["recomputed"]) > 1e-9:
                raise SystemExit(f"{pair_name}: fusion_equal {got} differs from the check's {checks[check_name]['recomputed']}")
    print(f"nested fusions done  ({minutes()})", flush=True)

    payload = {
        "analysis": "fusion weights fitted inside the training folds, against the published equal weight, four pairs of spaces",
        "question": "were the published fusions selected on the evaluation queries, and does the best fused system survive weights fitted inside the training folds",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "embeddings_sha256": d["embeddings_sha256"], "queries": n, "labels": L,
                   "groups": d["groups"], "chunks": int(len(d["chunk_song"])), "fold_sizes": fold_sizes,
                   "rhyme_covered_queries": int(covered.sum()), "minimum_endings_for_coverage": MINIMUM_ENDINGS,
                   "rhyme_rows_with_no_feature_among_covered": covered_but_empty},
        "design": {
            "fusion": "fused(w) = w * zscore(A) + (1 - w) * zscore(B), row-wise z-scores of a query's label scores (v1.zscore_rows); w = 0.5 is the published fusion",
            "grid": list(GRID),
            "folds": f"identity_probe_v2: leakage groups dealt into {FOLDS} folds, seed {SEED}",
            "outer": "the test fold's queries scored in the published fold-wise spaces (word SVD-1024 and within-author whitening fitted on the other four folds, seed SEED + k; chunk whitening fitted on the other four folds' chunks, whitened chunks averaged per song); raw spaces need no fitting; profiles leave-group-out over all other songs",
            "inner": "for outer fold k and training fold j, the fold-j queries scored in a space fitted on the three folds outside {k, j} (seed SEED + 10 + 5a + b for the pair a < b; one fit serves both orderings); raw spaces use their protocol scores; profiles leave-group-out over all other songs as under the protocol",
            "selection": "the grid value with the highest per-(group, label)-weighted MRR over the training-fold queries' inner scores; ties to the value nearest 0.5, then the smaller w; applied to the test fold's published scores; test folds pooled",
            "rhyme_pair": f"evaluated on the songs with at least {MINIMUM_ENDINGS} classifiable endings, where every row of both spaces is defined; words alone is scored on the same queries",
            "estimands": "MRR (plain mean of reciprocal ranks), per-(group, label)-weighted MRR (selection criterion and bootstrap estimand), recall@1, recall@10",
            "contrasts": f"paired group bootstrap, {REPLICATES} replicates, seed {SEED}, leakage groups resampled with replacement, each (group, label) component weighted one, over the pair's evaluated queries",
            "reading_rule": {
                "fusion_adds": "a fusion adds to its better component if fusion_fitted minus that component is positive with the 95% interval clear of zero; otherwise it does not add",
                "rhyme": "rhyme adds nothing beyond the surface if words+rhyme (fusion_fitted) minus words is not positive with the interval clear of zero; otherwise rhyme adds beyond the surface",
                "published_equal_weight": "if fusion_fitted minus fusion_equal is negative with the interval clear of zero, the published number exceeds what a weight fitted without the evaluation queries achieves and the fitted-weight number is the honest one; if positive with the interval clear of zero, the published number is conservative; if the interval covers zero, the published number stands as what honest weighting achieves",
                "description_only": "chosen weights per fold, grid curves, whether the equal weight is the grid optimum of all queries, and the contrasts of fusion_w_chosen_on_test_fold, fusion_w_chosen_on_all and fusion_w_from_outer_training against fusion_fitted (selection inflation)"},
            "checks": "CHINESE_RAP_CORPUS=v3; literals matched against the result files; population and fold sizes as in identity_probe.json; raw spaces reproduce the published numbers (gap <= 5e-4) under the published rank policies; fold-wise spaces reproduce 0.5426 / 0.4271 / 0.5854 (gap <= 0.002); dense scores identical across sparse-scorer calls; fused(0.5) bitwise the published mean; every fit's training folds recorded; the nested stage's fusion_equal equals the check's number"},
        "checks": {"against_published": checks, "literals_matched": published["literals"],
                   "dense_scores_identical_across_sparse_calls_max_gap": dense_gap,
                   "fused_half_bitwise_equals_published_mean": True,
                   "undefined_rows_by_space": undefined,
                   "fits": {name: spaces[name].fits for name in ("words_svd_whitened", "semantic_whitened_chunks")}},
        "pairs": results,
        "privacy": "aggregate only",
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / OUT_NAME
    out_path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nwrote {out_path}  ({payload['minutes']} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
