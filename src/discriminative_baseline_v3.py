#!/usr/bin/env python3
"""Does an instance-based discriminative classifier identify labels better than the
nearest-centroid prototype?

QUESTION. Every published number scores a held-out song by cosine with one profile per
label, the weighted sum of the label's other songs (a prototype: a one-centre model of the
label that never looks at any other label). A discriminative classifier fitted on the same
representation sees the instances and their labels together and learns, per label, a
boundary against every other label. If the prototype is leaving identity on the table, a
linear one-vs-rest classifier should identify better in the same space; if a label is one
centre with topic noise around it, the prototype should hold, and the ordering of the
spaces should not move.

DESIGN
  spaces      characters   character 2-5-gram TF-IDF (v1.fit_tfidf, the protocol's fit over
                           all songs: the idf sees every song's text and never a label)
              words        jieba word 1-2-gram TF-IDF (word_identity_anatomy_v2.fit_words)
              semantic_raw BGE-M3 song centroids, unit rows
              semantic_whitened   the centroids after within-author whitening fitted on the
                           training folds of the fold being scored (identity_probe_v2
                           fit_transform, Ledoit-Wolf shrinkage, the protocol's weights),
                           unit rows
  folds       the protocol's five folds of leakage groups (exemplar_vs_prototype_v3.setup:
              rng(20260825) over groups, so a group never straddles folds); every song is
              the test query of exactly one fold
  systems     protocol_prototype  the published scorer: cosine with the leave-group-out
                           weighted-sum profile built from ALL other songs
              fold_prototype      the same profile built from the training-fold songs only
                           (leave-group-out within the training folds, which removes nothing
                           for a test query because its group lies wholly in the test fold);
                           the fair comparison for a classifier trained on the same songs
              linear_svc          one-vs-rest LinearSVC: L2 penalty, squared hinge loss,
                           liblinear's primal trust-region Newton solver (dual=False)
              logistic_regression one-vs-rest L2 logistic regression, liblinear's primal
                           trust-region Newton solver
              Both classifiers are fitted on the training-fold songs with the protocol's
              per-(group, label) weights as sample weights, one binary problem per label
              (the label against every other label), tol 1e-4, max_iter 1000, the intercept
              a constant feature of value one and so regularised with the weights (sklearn's
              liblinear default). Objective per label: 0.5 (|w|^2 + b^2) + C sum_i s_i
              loss(y_i (w.x_i + b)). A test song is scored by each label's decision value
              w.x + b; ranking by decision value is ranking by one-vs-rest probability. The
              primal solvers draw no random numbers, so a fit does not depend on the thread
              count or on the order the labels are fitted in (liblinear's dual solvers share
              one global generator across threads and would not be reproducible).
  C grid      {0.1, 1, 10}, selected per outer fold and classifier on inner folds of the
              training folds: the four training folds are paired into two halves by fold
              index, each half is validated by a classifier fitted on the other half (for
              semantic_whitened the whitening is refitted on the inner fitting half), and
              the C with the highest plain-mean MRR over all validated training-fold songs
              wins, ties to the smaller C. A fit on a given half is the same whichever outer
              fold asks for it, so it is made once and reused (6 distinct halves, not 10);
              only reciprocal ranks of rows inside the asking fold's training folds are read.
              A label absent from an inner fitting half scores below every present label
              there, identically under every C. The test fold plays no part in selection.
              The grid was fixed beforehand; every fold whose selected C sits on an edge of
              it is counted and reported, because a reading reached with C on the edge in
              every fold is a reading of the classifier on this grid, not at its optimum.
  sample weights  1 / (songs in the query's (group, label) component), the protocol's
              weights: a song filed five times counts once, in the fits and in the bootstrap.
              A component lies inside one fold, so the weights are the same whether counted
              over all songs or over the training folds.

SCORERS AND ESTIMANDS, over all 7,220 queries, each scored in the fold that holds it out
  mrr         plain mean of reciprocal ranks, the rank by v1.rank_system on float32 scores
              (ties broken by label index, as everywhere in the protocol); the published "mrr"
  recall_at_1 share of queries whose true label ranks first
  label_macro_f1  top-1 prediction (the true label when it ranks first, otherwise the best
              other label); F1 per label over all queries, averaged over the labels present
              in the truth or the predictions (every label is present in the truth); each
              query counts once
  contrasts   each carries a paired group-bootstrap 95% interval (identity_spaces_v2
              paired_group_bootstrap: leakage groups resampled with replacement, 2000
              replicates, seed 20260825, one resample driving every system, each
              (group, label) component weighted one, so the point and interval of a contrast
              are the component-weighted difference, as in every contrast of the repository;
              recall_at_1 passes the top-1 indicator through the same function;
              label_macro_f1 uses the identical group draws on per-group weighted
              true/predicted/hit counts, macro F1 recomputed per replicate over the labels
              present in that replicate):
                linear_svc - fold_prototype, logistic_regression - fold_prototype   (the rule)
                linear_svc - logistic_regression
                fold_prototype - protocol_prototype   what restricting the profile to the
                                                      training folds costs the prototype
                linear_svc - protocol_prototype, logistic_regression - protocol_prototype
              and, for the space orderings, adjacent spaces in the ordering under the best
              classifier, under fold_prototype and under protocol_prototype.

CHECKS, before any new number is used (SystemExit on failure)
  1. reproduction first: protocol_prototype reproduces the published v3 numbers on unrounded
     values, against the result files (three_spaces.json, identity_probe.json, which must
     carry the current corpus digest and query/label counts) and against the literals pinned
     here: semantic_raw 0.2997, characters 0.4266, words 0.4963 (gap <= 5e-4: no fitted
     transform), semantic_whitened 0.4164 (gap <= 0.002); recall_at_1 to the same tolerances;
  2. by index, before any fit: every fold's test rows are disjoint from its training rows and
     share no leakage group with them; every inner validation half is disjoint from its
     fitting half, by row and by group; every label is present in every outer training set;
     the training weights of every label sum to a whole number of components;
  3. the restricted prototype equals the protocol's leave-group-out scorer when the training
     rows are all songs outside the query's group: max gap <= 1e-9 against
     label_size_calibration_v3.lgo_parts on 40 queries, dense and sparse, and lgo_parts
     <= 1e-9 against identity_probe_v2.dense_leave_group_out on the dense space;
  4. the one-vs-rest loop equals sklearn's native one-vs-rest LinearSVC (one liblinear call,
     one-vs-rest inside, the same primal solver) on a 30-label subset of the dense space:
     decision values within 1e-6;
  5. every whitening is fitted on rows disjoint, by row and by group, from the rows it
     scores; every test-fold score is finite; every selected C was read from finite inner
     reciprocal ranks of training-fold rows only; the top-1 prediction hits exactly when the
     rank is one; binary fits that reach max_iter are counted and reported.
  The synthetic test (test_discriminative_baseline.py) checks the fold prototype, the
  classifiers' objective and sample weights, the cached inner selection, macro F1 and its
  bootstrap against brute-force definitions.

READING RULE, fixed before the run. A classifier "beats the prototype" in a space if its
MRR minus the fold-restricted prototype's MRR is positive with the 95% paired
group-bootstrap interval clear of zero; it "loses to the prototype" if the difference is
negative with the interval clear of zero; otherwise "no clear difference". A space's reading
is that of its better classifier (the higher MRR), named. The best classifier overall is the
one with the higher mean MRR across the four spaces; the ordering of the spaces by MRR under
it is reported beside the ordering under the fold-restricted prototype and under the
published prototype, with the adjacent-space contrasts and whether the orderings agree.
Recall@1, macro F1, the prototype's fold-restriction cost, the comparison with the published
all-songs prototype and the selected C values are description, not the rule.

Added 2026-09-21, after the author's timing run had printed the whitened-semantic test-fold
numbers and after a reviewer's probe on one inner split of fold 0's training folds (no test
fold scored), before any classifier had scored a test fold of the TF-IDF spaces. On that inner
split logistic regression's MRR was still rising at C = 10, the top of the grid (words: 0.365
at C 10, 0.412 at 100, 0.427 at 1000). Two changes follow, neither entering the rule:
  - logistic_regression_wide, the same classifier selected on C in (10, 100, 1000) by the same
    inner halves, is reported beside the others with its contrasts; it is description only,
    excluded from every reading and from the choice of the best classifier;
  - a classifier whose selected C sits on an edge of its grid in all five folds carries a
    qualifier in its reading ("a reading of this grid").
Component-weighted MRR, R@1 and macro F1 are reported beside the plain means
(systems_component_weighted), since every contrast is component-weighted.

    set CHINESE_RAP_CORPUS=v3
    python src/discriminative_baseline_v3.py --private-root <ni-k> --low-priority    (about an hour at 12 threads)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from exemplar_vs_prototype_v3 import setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import REPLICATES, paired_group_bootstrap, weighted_mean  # noqa: E402
from identity_spaces_v2 import SEED as BOOTSTRAP_SEED  # noqa: E402
from label_size_calibration_v3 import lgo_parts  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# a fit that reaches max_iter is counted from n_iter_ (warning capture is not thread-safe)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "discriminative_baseline.json"
PUBLISHED_THREE_SPACES = OUT_DIR / "three_spaces.json"
PUBLISHED_IDENTITY_PROBE = OUT_DIR / "identity_probe.json"

SPACES = ("characters", "words", "semantic_raw", "semantic_whitened")
CLASSIFIERS = ("linear_svc", "logistic_regression")
PROTOTYPES = ("protocol_prototype", "fold_prototype")
METRICS = ("mrr", "recall_at_1", "label_macro_f1")
C_GRID = (0.1, 1.0, 10.0)
# declared sensitivity, description only: name -> (estimator kind, its own C grid)
SENSITIVITY = {"logistic_regression_wide": ("logistic_regression", (10.0, 100.0, 1000.0))}
SENSITIVITY_PAIRS = (("logistic_regression_wide", "fold_prototype"), ("logistic_regression_wide", "linear_svc"),
                     ("logistic_regression_wide", "logistic_regression"))
MAX_ITER = 1000
TOL = 1e-4
ABSENT_SCORE = -1e30        # a label with no fitting row: below every real score, finite in float32
PAIRS = (("linear_svc", "fold_prototype"), ("logistic_regression", "fold_prototype"),
         ("linear_svc", "logistic_regression"), ("fold_prototype", "protocol_prototype"),
         ("linear_svc", "protocol_prototype"), ("logistic_regression", "protocol_prototype"))

# pinned literals of the published v3 results, (mrr, recall_at_1): three_spaces.json systems
# semantic, lexical_char_2_5, lexical_words; identity_probe.json system within_author_whitening
PINNED = {"semantic_raw": (0.2997, 0.2019), "characters": (0.4266, 0.3342),
          "words": (0.4963, 0.4029), "semantic_whitened": (0.4164, 0.3179)}
TOLERANCE = {"semantic_raw": 5e-4, "characters": 5e-4, "words": 5e-4, "semantic_whitened": 0.002}
PUBLISHED_KEYS = {"semantic_raw": ("three_spaces", "semantic"), "characters": ("three_spaces", "lexical_char_2_5"),
                  "words": ("three_spaces", "lexical_words"), "semantic_whitened": ("identity_probe", "within_author_whitening")}
SCORER_CHECK_QUERIES = 40
SCORER_CHECK_GAP = 1e-9
NATIVE_CHECK_LABELS = 30
NATIVE_CHECK_GAP = 1e-6


def say(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------------------------ representations
def as_float64(x):
    """A float64 CSR with sorted indices, or a contiguous float64 array: what liblinear
    takes without a conversion per binary problem."""
    if sparse.issparse(x):
        out = sparse.csr_matrix(x, dtype=np.float64)
        out.sum_duplicates()
        out.sort_indices()
        return out
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def whiten(dense: np.ndarray, fit_rows: np.ndarray, label_index: np.ndarray, weights: np.ndarray):
    """Within-author whitening fitted on fit_rows (identity_probe_v2), applied to every song,
    unit rows. The generator is unused by this transform; the fit is deterministic."""
    mean, matrix, info = fit_transform("within_author_whitening", dense[fit_rows], label_index[fit_rows],
                                       weights[fit_rows], np.random.default_rng(SEED))
    return unit_rows((dense - mean) @ matrix.T), {k: float(v) for k, v in info.items()}


# ------------------------------------------------------------------ fold structure
def inner_halves(training_folds) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """The training folds paired into two halves by fold index: (fitting half, validation
    half) both ways, so every training fold is validated exactly once."""
    folds = sorted(int(f) for f in training_folds)
    a, b = tuple(folds[: len(folds) // 2]), tuple(folds[len(folds) // 2:])
    if not a or not b:
        raise SystemExit("inner selection needs at least two training folds")
    return [(a, b), (b, a)]


def assert_disjoint(fit_rows: np.ndarray, query_rows: np.ndarray, group_ids: np.ndarray, what: str) -> None:
    if np.intersect1d(fit_rows, query_rows).size:
        raise SystemExit(f"{what}: a scored row is among the fitting rows")
    if np.intersect1d(group_ids[fit_rows], group_ids[query_rows]).size:
        raise SystemExit(f"{what}: a scored row's leakage group has a member among the fitting rows")


def check_folds(d: dict) -> dict:
    """Check 2, by index and before any fit. Returns aggregate counts."""
    li, gi, w, L, fold = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"]
    if fold.min() < 0 or fold.max() >= FOLDS or len(np.unique(fold)) != FOLDS:
        raise SystemExit("the fold assignment does not have the protocol's five folds")
    seen = np.zeros(len(li), dtype=np.int64)
    absent_inner = {}
    for k in range(FOLDS):
        train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        assert_disjoint(train, test, gi, f"fold {k}")
        seen[test] += 1
        if not np.all(np.bincount(li[train], minlength=L) > 0):
            raise SystemExit(f"fold {k}: a label is absent from the training folds")
        mass = np.bincount(li[train], weights=w[train], minlength=L)
        if not np.allclose(mass, np.rint(mass), atol=1e-6):
            raise SystemExit(f"fold {k}: a (group, label) component straddles the fold boundary")
        for fit_folds, val_folds in inner_halves([f for f in range(FOLDS) if f != k]):
            if k in fit_folds or k in val_folds or set(fit_folds) & set(val_folds):
                raise SystemExit(f"fold {k}: the inner halves touch the test fold or each other")
            fit_rows, val_rows = np.flatnonzero(np.isin(fold, fit_folds)), np.flatnonzero(np.isin(fold, val_folds))
            assert_disjoint(fit_rows, val_rows, gi, f"fold {k} inner {fit_folds}->{val_folds}")
            assert_disjoint(fit_rows, test, gi, f"fold {k} inner {fit_folds} against the test fold")
            absent_inner["".join(map(str, fit_folds))] = int(np.sum(np.bincount(li[fit_rows], minlength=L) == 0))
    if not np.all(seen == 1):
        raise SystemExit("a song is not the test query of exactly one fold")
    return {"folds_disjoint_by_row_and_group": True, "inner_halves_disjoint_by_row_and_group": True,
            "every_label_in_every_outer_training_set": True, "components_whole_within_training_folds": True,
            "distinct_inner_fitting_halves": len(absent_inner),
            "labels_absent_from_inner_fitting_half": dict(sorted(absent_inner.items()))}


# ------------------------------------------------------------------ the prototypes
def restricted_prototype(x_train, y_train: np.ndarray, w_train: np.ndarray, label_count: int, x_query):
    """Cosine of each (unit) query row with every label's weighted sum of the training rows;
    a label without training rows scores ABSENT_SCORE. x is dense or CSR, float64."""
    n = x_train.shape[0]
    member = sparse.csr_matrix((w_train, (y_train, np.arange(n))), shape=(label_count, n))
    sums = member @ x_train
    if sparse.issparse(sums):
        sums = sums.tocsr()
        norm2 = np.asarray(sums.multiply(sums).sum(axis=1)).ravel()
        dots = np.asarray((x_query @ sums.T).todense(), dtype=np.float64)
    else:
        sums = np.asarray(sums)
        norm2 = np.einsum("ij,ij->i", sums, sums)
        dots = np.asarray(x_query @ sums.T, dtype=np.float64)
    present = norm2 > 1e-24
    scores = np.full((x_query.shape[0], label_count), ABSENT_SCORE)
    scores[:, present] = dots[:, present] / np.sqrt(norm2[present])
    return scores, present


def check_restricted_scorer(x, label_index, group_ids, weights, label_count, queries, dense_reference: bool = False,
                            gap: float = SCORER_CHECK_GAP) -> float:
    """Check 3: with the training rows = every song outside the query's group, the restricted
    prototype is the protocol's leave-group-out scorer. Returns the max gap; SystemExit past it."""
    dots, norm2, _ = lgo_parts(x, label_index, group_ids, weights, label_count, queries)
    protocol = dots / np.sqrt(norm2)
    if dense_reference:
        reference = dense_leave_group_out(x, label_index, group_ids, weights, label_count, queries)
        protocol_gap = float(np.abs(protocol - reference).max())
        if protocol_gap > gap:
            raise SystemExit(f"lgo_parts and dense_leave_group_out disagree ({protocol_gap:.2e})")
    worst = 0.0
    for row, q in enumerate(queries.tolist()):
        train = np.flatnonzero(group_ids != group_ids[q])
        scores, present = restricted_prototype(x[train], label_index[train], weights[train], label_count, x[[q]])
        if not np.all(present):
            raise SystemExit("a label has no song outside a query's group")
        worst = max(worst, float(np.abs(scores[0] - protocol[row]).max()))
    if worst > gap:
        raise SystemExit(f"the restricted prototype differs from the protocol's scorer (gap {worst:.2e})")
    return worst


# ------------------------------------------------------------------ the classifiers
def make_estimator(kind: str, C: float, tol: float = TOL):
    """Both are liblinear primal trust-region Newton solvers: deterministic, no random draws."""
    if kind == "linear_svc":
        return LinearSVC(penalty="l2", loss="squared_hinge", dual=False, tol=tol, C=C, max_iter=MAX_ITER)
    if kind == "logistic_regression":
        return LogisticRegression(solver="liblinear", dual=False, C=C, tol=tol, max_iter=MAX_ITER)
    raise ValueError(kind)


def fit_ovr(kind: str, C: float, x, y: np.ndarray, w: np.ndarray, label_count: int, n_jobs: int = 1, tol: float = TOL) -> dict:
    """One binary problem per label present in y (the label against every other label),
    sample weights w, threads over labels (liblinear releases the GIL while it trains).
    Returns the stacked weights (label_count, dim), the intercepts (ABSENT_SCORE for a label
    with no row, whose weights stay zero) and diagnostics."""
    present = np.bincount(y, minlength=label_count) > 0
    if present.sum() < 2:
        raise SystemExit("fewer than two labels in a fitting set")

    def one(label: int):
        model = make_estimator(kind, C, tol).fit(x, (y == label).astype(np.int64), sample_weight=w)
        if model.coef_.shape[0] != 1 or list(model.classes_) != [0, 1]:
            raise SystemExit("a binary fit did not return one weight vector for the classes (0, 1)")
        return label, model.coef_.ravel().astype(np.float64), float(model.intercept_[0]), int(np.max(model.n_iter_))

    fitted = Parallel(n_jobs=n_jobs, backend="threading")(delayed(one)(int(l)) for l in np.flatnonzero(present))
    weight = np.zeros((label_count, x.shape[1]))
    bias = np.full(label_count, ABSENT_SCORE)
    most, unconverged = 0, 0
    for label, coef, intercept, iterations in fitted:
        weight[label], bias[label] = coef, intercept
        most, unconverged = max(most, iterations), unconverged + int(iterations >= MAX_ITER)
    return {"kind": kind, "C": C, "weight": weight, "bias": bias, "present": present,
            "max_n_iter": most, "unconverged": unconverged, "rows": int(x.shape[0])}


def decision(model: dict, x) -> np.ndarray:
    """Decision value w.x + b of every label for every row; ABSENT_SCORE for a label absent from the fit."""
    return np.asarray(x @ model["weight"].T, dtype=np.float64) + model["bias"][None, :]


def check_native_ovr(x, y: np.ndarray, w: np.ndarray, label_count: int, x_test, gap: float = NATIVE_CHECK_GAP) -> float:
    """Check 4: the one-vs-rest loop is sklearn's own one-vs-rest LinearSVC (one liblinear
    call, the same solver). Returns the max decision gap on x_test; SystemExit past it."""
    native = make_estimator("linear_svc", 1.0).fit(x, y, sample_weight=w)
    if len(native.classes_) < 3:
        raise SystemExit("the native check needs at least three labels")
    loop = fit_ovr("linear_svc", 1.0, x, y, w, label_count, n_jobs=1)
    mine = decision(loop, x_test)[:, native.classes_]
    worst = float(np.abs(native.decision_function(x_test) - mine).max())
    if worst > gap:
        raise SystemExit(f"the one-vs-rest loop differs from sklearn's native LinearSVC (gap {worst:.2e})")
    return worst


def select_c(inner_mrr: dict, grid=C_GRID) -> float:
    """The C with the highest inner MRR; ties to the smaller C."""
    return min(grid, key=lambda C: (-inner_mrr[C], C))


# ------------------------------------------------------------------ scoring a space
def score_space(space: str, d: dict, fixed, n_jobs: int, c_grid=C_GRID, classifiers=CLASSIFIERS, log=say,
                sensitivity: dict | None = None) -> dict:
    """fold_prototype and every classifier, each test fold scored in its own fold's
    representation; C selected on inner halves of the training folds. `fixed` is the matrix
    of the space, or None for semantic_whitened (whitened per fitting set)."""
    li, gi, w, L, fold, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"], d["dense"]
    n = len(li)
    sensitivity = dict(sensitivity or {})
    names = ("fold_prototype",) + tuple(classifiers) + tuple(sensitivity)
    # every (estimator kind, C) the inner stage must fit, the main grid first
    wanted = [(kind, C) for kind in classifiers for C in c_grid]
    for kind, grid in sensitivity.values():
        wanted += [(kind, C) for C in grid if (kind, C) not in wanted]
    scores = {name: np.full((n, L), np.nan) for name in names}
    selection = {kind: {} for kind in tuple(classifiers) + tuple(sensitivity)}
    fits = {kind: {} for kind in tuple(classifiers) + tuple(sensitivity)}
    whitening = {}

    def represent(fit_rows: np.ndarray, scored_rows: np.ndarray, what: str):
        if space != "semantic_whitened":
            return fixed, None
        assert_disjoint(fit_rows, scored_rows, gi, f"whitening for {what}")
        return whiten(dense, fit_rows, li, w)

    # stage 1: one fit per distinct inner fitting half, classifier and C; the reciprocal
    # ranks of every row outside the half are kept, NaN inside it
    plan = {k: inner_halves([f for f in range(FOLDS) if f != k]) for k in range(FOLDS)}
    distinct = sorted({fit_folds for halves in plan.values() for fit_folds, _ in halves})
    inner_rr = {}
    for half in distinct:
        started = time.time()
        inside = np.isin(fold, half)
        fit_rows, other = np.flatnonzero(inside), np.flatnonzero(~inside)
        assert_disjoint(fit_rows, other, gi, f"{space} inner half {half}")
        x_all, _ = represent(fit_rows, other, f"{space} inner half {half}")
        x_fit, x_other = as_float64(x_all[fit_rows]), as_float64(x_all[other])
        for kind, C in wanted:
            model = fit_ovr(kind, C, x_fit, li[fit_rows], w[fit_rows], L, n_jobs)
            rr = np.full(n, np.nan)
            rr[other] = 1.0 / ranks_of(decision(model, x_other), li[other])
            inner_rr[(kind, C, half)] = rr
            del model
        log(f"  {space} inner half {half}: fitted on {len(fit_rows):,} rows, {len(wanted)} fits  [{time.time() - started:.0f}s]")

    # stage 2: per outer fold, select C from its own training folds, fit, score the test fold
    for k in range(FOLDS):
        started = time.time()
        train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        assert_disjoint(train, test, gi, f"{space} fold {k}")
        x_all, info = represent(train, test, f"{space} fold {k}")
        if info is not None:
            whitening[str(k)] = info
        x_train, x_test = as_float64(x_all[train]), as_float64(x_all[test])
        proto, present = restricted_prototype(x_train, li[train], w[train], L, x_test)
        if not np.all(present):
            raise SystemExit(f"{space} fold {k}: the fold prototype lost a label")
        scores["fold_prototype"][test] = proto
        line = []
        for system in tuple(classifiers) + tuple(sensitivity):
            kind, grid = (system, c_grid) if system in classifiers else sensitivity[system]
            inner_mrr = {}
            for C in grid:
                parts = []
                for fit_folds, val_folds in plan[k]:
                    val_rows = np.flatnonzero(np.isin(fold, val_folds))
                    if np.any(fold[val_rows] == k):
                        raise SystemExit(f"{space} fold {k}: a test row entered the selection of C")
                    parts.append(inner_rr[(kind, C, fit_folds)][val_rows])
                values = np.concatenate(parts)
                if len(values) != len(train) or not np.all(np.isfinite(values)):
                    raise SystemExit(f"{space} fold {k}: the inner reciprocal ranks do not cover the training folds once")
                inner_mrr[C] = float(np.mean(values))
            best = select_c(inner_mrr, grid)
            model = fit_ovr(kind, best, x_train, li[train], w[train], L, n_jobs)
            if not np.all(model["present"]):
                raise SystemExit(f"{space} fold {k}: {system} lost a label")
            scores[system][test] = decision(model, x_test)
            selection[system][str(k)] = {"selected_C": best, "at_grid_edge": bool(best in (min(grid), max(grid))),
                                         "inner_mrr_by_C": {f"{C:g}": round(v, 4) for C, v in inner_mrr.items()}}
            fits[system][str(k)] = {"max_n_iter": model["max_n_iter"], "unconverged_binary_fits": model["unconverged"],
                                    "training_rows": model["rows"]}
            line.append(f"{system} C={best:g} (inner " + " ".join(f"{C:g}:{v:.4f}" for C, v in inner_mrr.items())
                        + f"; n_iter<={model['max_n_iter']}, unconverged {model['unconverged']})")
            del model
        log(f"  {space} fold {k}: train {len(train):,} test {len(test):,}; " + "; ".join(line) + f"  [{time.time() - started:.0f}s]")
    for name, array in scores.items():
        if not np.all(np.isfinite(array)):
            raise SystemExit(f"{space}: non-finite {name} scores")
    return {"scores": scores, "selection": selection, "fits": fits, "whitening": whitening,
            "inner_fits": len(inner_rr)}


# ------------------------------------------------------------------ metrics and bootstrap
def predictions(scores: np.ndarray, truth: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Top-1 on the float32 values the rank uses: the true label when it ranks first,
    otherwise the first label of highest score (the best other label if that is the truth,
    which can only happen inside rank_system's 1e-12 tie band)."""
    s = scores.astype(np.float32)
    top = np.argmax(s, axis=1)
    s[np.arange(len(truth)), truth] = -np.inf
    runner_up = np.argmax(s, axis=1)
    return np.where(ranks == 1, truth, np.where(top == truth, runner_up, top))


def macro_f1(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray, label_count: int) -> float:
    """Weighted label-macro F1 over the labels present in the truth or the predictions."""
    hit = pred == truth
    tw = np.bincount(truth, weights=weights, minlength=label_count)
    pw = np.bincount(pred, weights=weights, minlength=label_count)
    tp = np.bincount(truth[hit], weights=weights[hit], minlength=label_count)
    denominator = tw + pw
    live = denominator > 0
    return float(np.mean(2.0 * tp[live] / denominator[live]))


def summarise(scores: np.ndarray, truth: np.ndarray) -> dict:
    ranks = ranks_of(scores, truth)
    pred = predictions(scores, truth, ranks)
    if not np.array_equal(pred == truth, ranks == 1):
        raise SystemExit("the top-1 prediction does not hit exactly when the rank is one")
    return {"ranks": ranks, "rr": 1.0 / ranks, "top1": (ranks == 1).astype(np.float64), "pred": pred,
            "mrr": float(np.mean(1.0 / ranks)), "recall_at_1": float(np.mean(ranks == 1)),
            "label_macro_f1": macro_f1(pred, truth, np.ones(len(truth)), scores.shape[1])}


def group_draws(group_ids: np.ndarray, mask: np.ndarray, replicates: int = REPLICATES):
    """The draws of identity_spaces_v2.paired_group_bootstrap, call for call: the per-row
    group position, the (replicates, groups) draws and the group count."""
    groups = np.unique(group_ids[mask])
    position_of = {int(g): i for i, g in enumerate(groups.tolist())}
    position = np.asarray([position_of[int(g)] for g in group_ids[mask]])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(groups), size=(replicates, len(groups)))
    return position, draws, len(groups)


def multiplicity(draws: np.ndarray, group_count: int) -> np.ndarray:
    """How many times each group is drawn in each replicate."""
    counts = np.zeros((draws.shape[0], group_count))
    for r in range(draws.shape[0]):
        counts[r] = np.bincount(draws[r], minlength=group_count)
    return counts


def macro_f1_replicates(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray, position: np.ndarray,
                        counts: np.ndarray, label_count: int) -> np.ndarray:
    """Weighted label-macro F1 of every resample given as group multiplicities (rows of
    counts): per-group weighted true, predicted and hit counts per label are summed with the
    multiplicities and F1 recomputed over the labels present in that resample."""
    group_count = counts.shape[1]
    hit = pred == truth
    true_weight = np.zeros((group_count, label_count))
    pred_weight = np.zeros((group_count, label_count))
    tp = np.zeros((group_count, label_count))
    np.add.at(true_weight, (position, truth), weights)
    np.add.at(pred_weight, (position, pred), weights)
    np.add.at(tp, (position[hit], truth[hit]), weights[hit])
    denominator = counts @ true_weight + counts @ pred_weight
    numerator = 2.0 * (counts @ tp)
    live = denominator > 0
    f1 = np.where(live, numerator / np.where(live, denominator, 1.0), 0.0)
    return f1.sum(axis=1) / live.sum(axis=1)


def paired_group_bootstrap_macro_f1(pred_by_system: dict, truth: np.ndarray, weights: np.ndarray, group_ids: np.ndarray,
                                    label_count: int, pairs, replicates: int = REPLICATES) -> list[dict]:
    """Percentile intervals for weighted label-macro F1 differences under the group draws of
    paired_group_bootstrap; the point is the weighted macro F1 difference on the full sample."""
    mask = np.ones(len(truth), dtype=bool)
    position, draws, group_count = group_draws(group_ids, mask, replicates)
    counts = multiplicity(draws, group_count)
    whole = np.ones((1, group_count))
    replicate = {name: macro_f1_replicates(pred, truth, weights, position, counts, label_count) for name, pred in pred_by_system.items()}
    point = {name: float(macro_f1_replicates(pred, truth, weights, position, whole, label_count)[0]) for name, pred in pred_by_system.items()}
    out = []
    for left, right in pairs:
        low, high = np.percentile(replicate[left] - replicate[right], [2.5, 97.5])
        out.append({"system": left, "minus": right, "label_macro_f1_difference": round(point[left] - point[right], 4),
                    "ci95": [round(float(low), 4), round(float(high), 4)], "excludes_zero": bool(low > 0 or high < 0)})
    return out


def rename(contrasts: list[dict], key: str) -> list[dict]:
    return [{(key if k == "mrr_difference" else k): v for k, v in c.items()} for c in contrasts]


def contrasts_for(summaries: dict, truth: np.ndarray, weights: np.ndarray, group_ids: np.ndarray, label_count: int, pairs) -> dict:
    """The three metrics' paired group-bootstrap contrasts for the given (system, minus) pairs."""
    mask = np.ones(len(truth), dtype=bool)
    rr = {name: s["rr"] for name, s in summaries.items()}
    top1 = {name: s["top1"] for name, s in summaries.items()}
    pred = {name: s["pred"] for name, s in summaries.items()}
    return {"mrr": paired_group_bootstrap(rr, weights, group_ids, mask, list(pairs)),
            "recall_at_1": rename(paired_group_bootstrap(top1, weights, group_ids, mask, list(pairs)), "recall_at_1_difference"),
            "label_macro_f1": paired_group_bootstrap_macro_f1(pred, truth, weights, group_ids, label_count, list(pairs))}


# ------------------------------------------------------------------ readings
def verdict(contrast: dict) -> str:
    """The reading rule on one MRR contrast (classifier minus fold_prototype)."""
    low, high = contrast["ci95"]
    if contrast["excludes_zero"] and low >= 0 and high > 0:
        return "beats the prototype"
    if contrast["excludes_zero"] and high <= 0 and low < 0:
        return "loses to the prototype"
    return "no clear difference"


def read_space(mrr_by_system: dict, mrr_contrasts: list[dict], classifiers=CLASSIFIERS) -> dict:
    """The rule per classifier, and the space's reading from its better classifier
    (unrounded MRR; a tie goes to the first named)."""
    by_classifier = {}
    for kind in classifiers:
        c = next(c for c in mrr_contrasts if (c["system"], c["minus"]) == (kind, "fold_prototype"))
        by_classifier[kind] = {"verdict": verdict(c), "mrr_difference": c["mrr_difference"], "ci95": c["ci95"]}
    better = max(classifiers, key=lambda kind: (mrr_by_system[kind], -classifiers.index(kind)))
    return {"by_classifier": by_classifier, "better_classifier": better, "reading": f"{better} {by_classifier[better]['verdict']}"}


def ordering(mrr_by_space: dict) -> list[str]:
    return sorted(mrr_by_space, key=lambda s: (-mrr_by_space[s], s))


# ------------------------------------------------------------------ the published numbers
def check_against_published(recomputed: dict, published: dict, pinned: dict | None = PINNED, tolerance: dict = TOLERANCE, log=say) -> dict:
    """Check 1: every space's (mrr, recall_at_1) within tolerance of the published file's
    values and of the pinned literals; SystemExit otherwise."""
    record = {}
    failed = []
    for space, (mrr, r1) in recomputed.items():
        gaps = [abs(mrr - published[space][0]), abs(r1 - published[space][1])]
        if pinned:
            gaps += [abs(mrr - pinned[space][0]), abs(r1 - pinned[space][1])]
        worst = max(gaps)
        record[space] = {"recomputed_mrr": round(mrr, 4), "recomputed_recall_at_1": round(r1, 4),
                         "published_mrr": published[space][0], "published_recall_at_1": published[space][1],
                         "gap": round(worst, 5), "tolerance": tolerance[space], "passed": bool(worst <= tolerance[space])}
        log(f"  check {space:18s} MRR {mrr:.4f} published {published[space][0]:.4f}  R@1 {r1:.4f} published "
            f"{published[space][1]:.4f}  (gap {worst:.5f}, tolerance {tolerance[space]})")
        if worst > tolerance[space]:
            failed.append(space)
    if failed:
        raise SystemExit(f"the protocol prototype does not reproduce the published numbers in {failed}")
    return record


def load_published(three_spaces: Path = PUBLISHED_THREE_SPACES, identity_probe: Path = PUBLISHED_IDENTITY_PROBE,
                   queries: int | None = None, labels: int | None = None) -> dict:
    """(mrr, recall_at_1) per space from the two published files, which must be v3 results
    of the current corpus build."""
    files = {"three_spaces": json.loads(three_spaces.read_bytes().decode("utf-8")),
             "identity_probe": json.loads(identity_probe.read_bytes().decode("utf-8"))}
    if files["three_spaces"]["corpus"]["content_sha256"] != V3_CONTENT_SHA256:
        raise SystemExit("three_spaces.json was computed on a different corpus build")
    if queries is not None and (files["three_spaces"]["corpus"]["queries"] != queries or files["identity_probe"]["queries"] != queries):
        raise SystemExit("the published files do not have this build's query count")
    if labels is not None and (files["three_spaces"]["corpus"]["labels"] != labels or files["identity_probe"]["labels"] != labels):
        raise SystemExit("the published files do not have this build's label count")
    out = {}
    for space, (which, key) in PUBLISHED_KEYS.items():
        entry = files[which]["systems"][key]
        out[space] = (float(entry["mrr"]), float(entry["recall_at_1"]))
    return out


# ------------------------------------------------------------------ the protocol prototype
def protocol_prototype_scores(d: dict, chars, words, log=say) -> dict:
    """The published scorer in every space, the unchanged configuration: v1's sparse scorer
    for the TF-IDF spaces, identity_probe_v2's dense scorer and fold-wise whitening."""
    li, gi, w, L, fold, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"], d["dense"]
    n = len(li)
    scores = {"semantic_raw": dense_leave_group_out(dense, li, gi, w, L, np.arange(n))}
    dense32 = dense.astype(np.float32)
    scores["characters"] = v1.score_leave_group_out(dense32, chars, li, gi, L).lexical.astype(np.float64)
    scores["words"] = v1.score_leave_group_out(dense32, words, li, gi, L).lexical.astype(np.float64)
    whitened = np.full((n, L), np.nan)
    for k in range(FOLDS):
        train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        assert_disjoint(train, test, gi, f"whitening fold {k}")
        matrix, info = whiten(dense, train, li, w)
        whitened[test] = dense_leave_group_out(matrix, li, gi, w, L, test)
        log(f"  whitening fold {k}: shrinkage {info['shrinkage']:.4f}, within-author share {info['within_author_share_of_variance']:.4f}")
    scores["semantic_whitened"] = whitened
    return scores


# ------------------------------------------------------------------ the analysis
def analyse(d: dict, chars, words, published: dict, *, n_jobs: int = 1, c_grid=C_GRID, pinned=PINNED, tolerance=TOLERANCE,
            checks_only: bool = False, spaces=SPACES, classifiers=CLASSIFIERS, debug: dict | None = None, log=say,
            sensitivity: dict | None = None) -> dict:
    """Everything after setup, on any corpus of the protocol's shape; returns the aggregate payload."""
    started = time.time()
    li, gi, w, L, fold, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"], d["dense"]
    n = len(li)
    fold_sizes = np.bincount(fold, minlength=FOLDS).tolist()
    log(f"{n:,} queries, {L} labels, {len(np.unique(gi)):,} groups; fold sizes {fold_sizes}")

    log("the published prototype in every space (unchanged configuration)")
    protocol = protocol_prototype_scores(d, chars, words, log=log)
    summaries = {space: {"protocol_prototype": summarise(protocol[space], li)} for space in SPACES}
    log("check 1: reproducing the published numbers")
    reproduction = check_against_published({space: (summaries[space]["protocol_prototype"]["mrr"],
                                                    summaries[space]["protocol_prototype"]["recall_at_1"]) for space in SPACES},
                                           published, pinned, tolerance, log=log)

    log("check 2: folds, inner halves, labels and components, by index")
    fold_checks = check_folds(d)
    log(f"  labels absent from an inner fitting half: {fold_checks['labels_absent_from_inner_fitting_half']}")

    log("check 3: the restricted prototype against the protocol's scorer")
    probe = np.random.default_rng(SEED).choice(n, size=min(SCORER_CHECK_QUERIES, n), replace=False)
    scorer_gaps = {"dense": check_restricted_scorer(dense, li, gi, w, L, probe, dense_reference=True),
                   "sparse": check_restricted_scorer(as_float64(words), li, gi, w, L, probe)}
    log(f"  max gap dense {scorer_gaps['dense']:.2e}, sparse {scorer_gaps['sparse']:.2e} (limit {SCORER_CHECK_GAP})")

    log("check 4: the one-vs-rest loop against sklearn's native one-vs-rest LinearSVC")
    subset_labels = np.arange(min(NATIVE_CHECK_LABELS, L))
    sub_train = np.flatnonzero((fold != 0) & np.isin(li, subset_labels))
    sub_test = np.flatnonzero((fold == 0) & np.isin(li, subset_labels))
    native_gap = check_native_ovr(as_float64(dense[sub_train]), li[sub_train], w[sub_train], L, as_float64(dense[sub_test]))
    log(f"  max decision gap {native_gap:.2e} on {len(subset_labels)} labels, {len(sub_train):,} rows (limit {NATIVE_CHECK_GAP})")
    checks = {"against_published": reproduction, "folds": fold_checks, "restricted_scorer_max_gap": scorer_gaps,
              "native_ovr_max_gap": native_gap}
    if checks_only:
        return {"checks": checks, "minutes": round((time.time() - started) / 60, 1)}

    fixed = {"characters": chars, "words": words, "semantic_raw": dense, "semantic_whitened": None}
    results = {}
    for space in spaces:
        log(f"== {space}")
        detail = score_space(space, d, fixed[space], n_jobs, c_grid, classifiers, log=log, sensitivity=sensitivity)
        for name, array in detail["scores"].items():
            summaries[space][name] = summarise(array, li)
        if debug is not None:
            debug.setdefault("scores", {})[space] = {"protocol_prototype": protocol[space], **detail["scores"]}
        del detail["scores"]
        side = {name: summaries[space].pop(name) for name in (sensitivity or {})}
        systems = {name: {m: round(s[m], 4) for m in METRICS} for name, s in summaries[space].items()}
        everything = np.ones(n, dtype=bool)
        weighted = {name: {"mrr": round(weighted_mean(s["rr"], w, everything), 4),
                           "recall_at_1": round(weighted_mean(s["top1"], w, everything), 4),
                           "label_macro_f1": round(macro_f1(s["pred"], li, w, L), 4)}
                    for name, s in {**summaries[space], **side}.items()}
        live_pairs = [(a, b) for a, b in PAIRS if a in summaries[space] and b in summaries[space]]
        contrasts = contrasts_for(summaries[space], li, w, gi, L, live_pairs)
        reading = read_space({name: s["mrr"] for name, s in summaries[space].items()}, contrasts["mrr"], classifiers)
        for kind in classifiers:
            every_fold_on_edge = all(e["at_grid_edge"] for e in detail["selection"][kind].values())
            reading["by_classifier"][kind]["selected_C_on_grid_edge_in_every_fold"] = bool(every_fold_on_edge)
            if every_fold_on_edge:
                reading["by_classifier"][kind]["verdict_qualified"] = (
                    reading["by_classifier"][kind]["verdict"] + " (selected C on the grid edge in every fold: a reading of this grid)")
        better = reading["better_classifier"]
        if reading["by_classifier"][better]["selected_C_on_grid_edge_in_every_fold"]:
            reading["reading"] += " (selected C on the grid edge in every fold: a reading of this grid)"
        results[space] = {"systems": systems, "systems_component_weighted": weighted, "contrasts": contrasts,
                          "selected_C": detail["selection"], "fits": detail["fits"], "whitening_by_fold": detail["whitening"],
                          "distinct_inner_fits": detail["inner_fits"], "reading": reading}
        if side:
            both = {**summaries[space], **side}
            side_pairs = [(a, b) for a, b in SENSITIVITY_PAIRS if a in both and b in both]
            results[space]["sensitivity"] = {
                "role": "description only; not in the reading rule or the choice of the best classifier",
                "grids": {name: list(grid) for name, (_, grid) in sensitivity.items()},
                "systems": {name: {m: round(s[m], 4) for m in METRICS} for name, s in side.items()},
                "contrasts": contrasts_for(both, li, w, gi, L, side_pairs)}
            for c in results[space]["sensitivity"]["contrasts"]["mrr"]:
                log(f"    sensitivity {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
        log("  " + "  ".join(f"{name}: MRR {e['mrr']:.4f} R@1 {e['recall_at_1']:.4f} F1 {e['label_macro_f1']:.4f}" for name, e in systems.items()))
        for metric, key in (("mrr", "mrr_difference"), ("recall_at_1", "recall_at_1_difference"), ("label_macro_f1", "label_macro_f1_difference")):
            for c in contrasts[metric]:
                log(f"    {metric:14s} {c['system']} - {c['minus']}: {c[key]:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
        log(f"  reading: {reading['reading']}")

    # the space orderings, on unrounded MRR
    mean_mrr = {kind: float(np.mean([summaries[s][kind]["mrr"] for s in spaces])) for kind in classifiers}
    best = max(classifiers, key=lambda kind: (mean_mrr[kind], -classifiers.index(kind)))
    orderings, adjacent = {}, {}
    rr_all = {f"{space}/{name}": summaries[space][name]["rr"] for space in spaces for name in summaries[space]}
    mask = np.ones(n, dtype=bool)
    for name in (best, "fold_prototype", "protocol_prototype"):
        order = ordering({space: summaries[space][name]["mrr"] for space in spaces})
        orderings[name] = order
        pairs = [(f"{a}/{name}", f"{b}/{name}") for a, b in zip(order[:-1], order[1:])]
        adjacent[name] = paired_group_bootstrap(rr_all, w, gi, mask, pairs) if pairs else []
    agree = {"best_classifier_vs_fold_prototype": orderings[best] == orderings["fold_prototype"],
             "best_classifier_vs_protocol_prototype": orderings[best] == orderings["protocol_prototype"]}
    log(f"best classifier overall: {best} (mean MRR " + ", ".join(f"{k} {v:.4f}" for k, v in mean_mrr.items()) + ")")
    for name, order in orderings.items():
        log(f"  ordering under {name}: " + " > ".join(f"{s} ({results[s]['systems'][name]['mrr']:.4f})" for s in order))
        for c in adjacent[name]:
            log(f"    {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    beaten = [space for space in spaces if results[space]["reading"]["by_classifier"][best]["verdict"] == "beats the prototype"]
    unconverged = int(sum(f["unconverged_binary_fits"] for space in spaces for kind in classifiers for f in results[space]["fits"][kind].values()))
    at_edge = {space: {kind: int(sum(e["at_grid_edge"] for e in results[space]["selected_C"][kind].values()))
                       for kind in results[space]["selected_C"] if kind in classifiers}
               for space in spaces}
    summary = {"best_classifier": best, "mean_mrr_over_spaces": {k: round(v, 4) for k, v in mean_mrr.items()},
               "folds_with_selected_C_at_grid_edge": at_edge,
               "spaces_where_best_classifier_beats_fold_prototype": beaten,
               "readings": {space: results[space]["reading"]["reading"] for space in spaces},
               "unconverged_binary_fits_in_scored_models": unconverged}
    log("readings: " + "; ".join(f"{space}: {r}" for space, r in summary["readings"].items()))
    log(f"binary fits that reached max_iter in the scored models: {unconverged}")
    log("folds (of five) whose selected C sits on an edge of the grid: "
        + "; ".join(f"{space} " + ", ".join(f"{kind} {v}" for kind, v in by_kind.items()) for space, by_kind in at_edge.items()))

    return {
        "analysis": "one-vs-rest linear classifiers against the nearest-centroid prototype, four spaces, five folds",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L, "groups": int(len(np.unique(gi))),
                   "fold_sizes": fold_sizes},
        "design": {
            "spaces": {"characters": "character 2-5-gram TF-IDF, the protocol's fit over all songs (the idf sees every song's text, labels never)",
                       "words": "jieba word 1-2-gram TF-IDF, the protocol's fit over all songs",
                       "semantic_raw": "BGE-M3 song centroids, unit rows",
                       "semantic_whitened": "within-author whitening fitted on the training folds of the fold scored (identity_probe_v2); "
                                            "for the selection of C refitted on the inner fitting half"},
            "folds": f"{FOLDS} folds of leakage groups, seed {SEED}, as in identity_probe_v2; every song is a test query once",
            "systems": {"protocol_prototype": "cosine with the leave-group-out weighted-sum profile of all other songs (the published scorer)",
                        "fold_prototype": "the same profile built from the training-fold songs only",
                        "linear_svc": f"one-vs-rest LinearSVC, L2 penalty, squared hinge, liblinear primal trust-region Newton (dual=False), tol {TOL}, "
                                      f"max_iter {MAX_ITER}, regularised intercept, sample weights = the protocol's component weights",
                        "logistic_regression": f"one-vs-rest L2 logistic regression, liblinear primal trust-region Newton, tol {TOL}, max_iter {MAX_ITER}, "
                                               "regularised intercept, the same sample weights"},
            "c_grid": list(c_grid),
            "sensitivity": ({name: {"estimator": kind, "c_grid": list(grid)} for name, (kind, grid) in sensitivity.items()}
                            if sensitivity else None),
            "c_selection": "per outer fold and classifier: the four training folds paired into two halves by fold index, each half validated by a "
                           "classifier fitted on the other half; the C with the highest plain-mean MRR over all validated training-fold "
                           "songs, ties to the smaller C; a fit on a half is made once and reused by every outer fold whose training folds "
                           "contain it; a label absent from an inner fitting half scores below every present label there",
            "scoring": "a test song is scored by each label's decision value w.x + b (ranking by one-vs-rest probability); "
                       "ranks by v1.rank_system, ties by label index",
            "metrics": {"mrr": "plain mean of reciprocal ranks", "recall_at_1": "share of queries ranked first",
                        "label_macro_f1": "top-1 prediction, F1 per label, mean over labels present in truth or prediction; each query once"},
            "bootstrap": f"paired group bootstrap, {REPLICATES} replicates, seed {BOOTSTRAP_SEED}, leakage groups resampled with replacement, each "
                         "(group, label) component weighted one (point and interval of a contrast are component-weighted); recall_at_1 "
                         "through the same function on the top-1 indicator; label_macro_f1 with the identical draws on per-group "
                         "weighted counts, F1 recomputed per replicate",
            "checks": "protocol prototype reproduces the published MRR and R@1 in every space (files and pinned literals; gap <= 5e-4 raw, "
                      "<= 0.002 whitened); folds and inner halves disjoint by row and leakage group, every label in every outer training "
                      "set, components whole within training folds; restricted prototype equals the protocol's leave-group-out scorer "
                      f"when the training rows are all songs outside the query's group (<= {SCORER_CHECK_GAP}, {SCORER_CHECK_QUERIES} queries, "
                      f"dense and sparse); one-vs-rest loop equals sklearn's native LinearSVC on {NATIVE_CHECK_LABELS} labels "
                      f"(<= {NATIVE_CHECK_GAP}); whitening rows disjoint from scored rows; all scores finite; top-1 hits exactly at rank one",
            "reading_rule": "a classifier beats the prototype in a space if its MRR minus the fold-restricted prototype's is positive with "
                            "the 95% interval clear of zero (loses if negative and clear of zero; otherwise no clear difference); a space "
                            "reads by its better classifier; the space ordering by MRR under the best classifier (higher mean MRR over "
                            "the spaces) is reported beside the orderings under the fold-restricted and the published prototype",
        },
        "checks": checks,
        "spaces": results,
        "space_orderings": {"best_classifier": best, "orderings": orderings, "adjacent_contrasts_mrr": adjacent, "agree": agree},
        "summary": summary,
        "privacy": "aggregate only; no lyric text, song title, song identifier, word or per-song vector",
        "minutes": round((time.time() - started) / 60, 1),
    }


# ------------------------------------------------------------------ main
def lower_priority() -> str:
    """Below-normal process priority, so a shared machine stays responsive. Best effort."""
    try:
        import psutil
        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
        return "below normal"
    except Exception as error:                                    # noqa: BLE001
        return f"unchanged ({type(error).__name__})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=min(12, max(1, (os.cpu_count() or 1))),
                        help="threads over the binary problems of one fit; the results do not depend on it")
    parser.add_argument("--low-priority", action="store_true", help="run at below-normal process priority")
    parser.add_argument("--checks-only", action="store_true", help="stop after the checks; write nothing")
    parser.add_argument("--spaces", nargs="+", choices=SPACES, default=None,
                        help="a timing run on a subset of spaces; prints, writes nothing")
    args = parser.parse_args()
    if os.environ.get("CHINESE_RAP_CORPUS") != "v3":
        raise SystemExit("set CHINESE_RAP_CORPUS=v3; this analysis is a corpus v3 result")
    for path in (PUBLISHED_THREE_SPACES, PUBLISHED_IDENTITY_PROBE):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    if args.low_priority:
        say(f"process priority: {lower_priority()}")
    started = time.time()
    import jieba
    jieba.setLogLevel(60)
    say("loading corpus v3 and building the protocol's queries, groups, weights and folds")
    d = setup(args.private_root.resolve())
    published = load_published(queries=len(d["label_index"]), labels=d["label_count"])
    probe_folds = json.loads(PUBLISHED_IDENTITY_PROBE.read_bytes().decode("utf-8"))["design"]["fold_sizes"]
    if [int(v) for v in probe_folds] != np.bincount(d["fold"], minlength=FOLDS).tolist():
        raise SystemExit("the folds are not identity_probe_v2's")
    say("fitting the character and word TF-IDF spaces")
    docs = [d["documents"][s] for s in d["songs"]]
    chars = v1.fit_tfidf(docs)
    words, _ = fit_words([" ".join(segment(doc)) for doc in docs])
    del docs
    say(f"  characters {chars.shape[1]:,} features, words {words.shape[1]:,} features  [{time.time() - started:.0f}s]")
    partial = args.spaces is not None
    payload = analyse(d, chars, words, published, n_jobs=args.n_jobs, checks_only=args.checks_only,
                      spaces=tuple(args.spaces) if partial else SPACES, sensitivity=SENSITIVITY)
    payload["minutes"] = round((time.time() - started) / 60, 1)
    if args.checks_only or partial:
        say(f"\nchecks passed; nothing written ({payload['minutes']} min)")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / OUT_NAME).write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    say(f"\nwrote {args.out_dir / OUT_NAME}  ({payload['minutes']} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
