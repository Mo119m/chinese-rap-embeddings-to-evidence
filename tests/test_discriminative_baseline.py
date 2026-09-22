#!/usr/bin/env python3
"""Synthetic unit test for discriminative_baseline_v3: random data, no corpus, no private files.

Checks, each against an independent definition:
  1. restricted_prototype equals a brute-force loop (weighted label sums, cosine) on dense and
     sparse data, gives ABSENT_SCORE to a label without training rows, passes
     check_restricted_scorer against the protocol's scorers, and that check FAILS when the
     scorer is corrupted (weights ignored);
  2. the classifiers minimise the stated objective, 0.5 (|w|^2 + b^2) + C sum_i s_i loss_i,
     with the sample weights as per-row multipliers: decision values equal a scipy
     minimisation of that objective written here, for the squared hinge and the logistic
     loss; a row of weight two equals the row entered twice; the fitted label is the
     positive class;
  3. the one-vs-rest loop equals sklearn's native one-vs-rest LinearSVC (check_native_ovr),
     the comparison is sensitive (a loop at another C is far from it), a label absent from
     the fit scores ABSENT_SCORE, and a fit is identical for one thread and for four;
  4. macro_f1 equals sklearn's f1_score(average="macro"), unweighted and weighted, with labels
     absent from truth and predictions; predictions() hits exactly when ranks_of gives rank
     one, under exact ties too;
  5. group_draws and multiplicity reproduce paired_group_bootstrap's own intervals (so the
     macro-F1 bootstrap shares its draws), macro_f1_replicates equals sklearn's weighted macro
     F1 on explicitly resampled data, and paired_group_bootstrap_macro_f1 returns the
     percentile interval of those brute-force replicates;
  6. select_c, inner_halves (every training fold validated once, never the test fold, six
     distinct fitting halves), verdict, ordering, check_against_published (passes inside the
     tolerance, SystemExit outside), check_folds (SystemExit when a leakage group straddles
     folds or a label is missing from a training set);
  7. end to end on a synthetic corpus of the protocol's shape (leakage groups, duplicate
     components, five group-level folds, a dense space and two sparse spaces): the
     reproduction check passes against brute-force published numbers; fold_prototype equals
     a brute-force per-fold loop in all four spaces (whitening refitted here); the selected C,
     the inner MRRs and the test-fold decision values of both classifiers equal a direct
     recomputation without the cache (fit per outer fold, per inner half, one sklearn binary
     fit per label), in a fixed space and in the whitened space; the test fold's labels do
     not influence its own selection or scores' inputs (scrambling them leaves the selected C
     and the test-fold scores unchanged); the payload is JSON-serialisable, aggregate only,
     and its readings follow the rule.

    python test_discriminative_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("CHINESE_RAP_CORPUS", "v3")
HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src",):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import discriminative_baseline_v3 as db  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from identity_probe_v2 import FOLDS, SEED, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402

N, L, DIM, VOCAB = 420, 10, 24, 160
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("ok   " if condition else "FAIL ") + message, flush=True)
    if not condition:
        FAILURES.append(message)


def raises_system_exit(function) -> bool:
    try:
        function()
    except SystemExit:
        return True
    return False


# ------------------------------------------------------------------ synthetic corpus
def synthetic(seed: int) -> tuple[dict, sparse.csr_matrix, sparse.csr_matrix]:
    rng = np.random.default_rng(seed)
    label_index = rng.permutation(np.repeat(np.arange(L), N // L))
    group_ids = np.arange(N)
    for i in range(0, 60, 2):                          # thirty pairs; labels random, so some span labels
        group_ids[i + 1] = group_ids[i]
    group_ids[60:64] = group_ids[60]                   # one group of four
    same = np.flatnonzero(label_index == label_index[100])[:3]
    group_ids[same] = group_ids[same[0]]               # a within-label component of three: weights 1/3
    _, group_ids = np.unique(group_ids, return_inverse=True)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=int(group_ids.max()) + 1)[group_ids]
    centres = rng.standard_normal((L, DIM))
    dense = 0.9 * centres[label_index] + rng.standard_normal((N, DIM))
    topic = rng.gamma(0.3, 1.0, size=(L, VOCAB)) + 0.02

    def bag(scale: float) -> sparse.csr_matrix:
        counts = rng.poisson(scale * topic[label_index] / topic[label_index].sum(axis=1, keepdims=True))
        counts[np.arange(N), rng.integers(0, VOCAB, size=N)] += 1          # no empty row
        values = np.log1p(counts.astype(np.float64))
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return sparse.csr_matrix(values.astype(np.float32))

    chars, words = bag(30.0), bag(18.0)
    for g in np.flatnonzero(np.bincount(group_ids) > 1):                   # a leakage group is near-duplicate text
        members = np.flatnonzero(group_ids == g)
        dense[members[1:]] = dense[members[0]] + 0.01 * rng.standard_normal((len(members) - 1, DIM))
    dense = dense / np.linalg.norm(dense, axis=1, keepdims=True)
    d = dict(songs=[f"song_{i:05d}" for i in range(N)], label_index=label_index.astype(np.int64),
             group_ids=group_ids.astype(np.int64), label_count=L, weights=weights, fold=fold, dense=dense, documents={})
    return d, chars, words


# ------------------------------------------------------------------ brute-force definitions
def brute_rank(scores: np.ndarray, truth: int) -> int:
    s = scores.astype(np.float32)
    better = sum(1 for l in range(len(s)) if s[l] > s[truth] + 1e-12)
    tied_before = sum(1 for l in range(truth) if abs(float(s[l]) - float(s[truth])) <= 1e-12)
    return 1 + better + tied_before


def brute_prototype(x: np.ndarray, li, w, label_count, train_rows, query: int) -> np.ndarray:
    out = np.zeros(label_count)
    for l in range(label_count):
        profile = np.zeros(x.shape[1])
        for i in train_rows:
            if li[i] == l:
                profile += w[i] * x[i]
        out[l] = float(x[query] @ profile) / float(np.linalg.norm(profile))
    return out


def brute_leave_group_out(x: np.ndarray, d: dict, queries) -> np.ndarray:
    li, gi, w = d["label_index"], d["group_ids"], d["weights"]
    return np.stack([brute_prototype(x, li, w, d["label_count"], [i for i in range(len(li)) if gi[i] != gi[q]], q) for q in queries])


def brute_objective_fit(kind: str, C: float, x: np.ndarray, y01: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimise 0.5 (|w|^2 + b^2) + C sum_i s_i loss(y_i (w.x_i + b)) with scipy."""
    y = 2.0 * y01 - 1.0
    xb = np.hstack([x, np.ones((len(x), 1))])

    def squared_hinge(theta):
        margin = np.maximum(0.0, 1.0 - y * (xb @ theta))
        return 0.5 * theta @ theta + C * np.sum(s * margin ** 2), theta - 2.0 * C * xb.T @ (s * margin * y)

    def logistic(theta):
        z = y * (xb @ theta)
        return 0.5 * theta @ theta + C * np.sum(s * np.logaddexp(0.0, -z)), theta - C * xb.T @ (s * y / (1.0 + np.exp(z)))

    result = minimize(squared_hinge if kind == "linear_svc" else logistic, np.zeros(xb.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10})
    return result.x[:-1], float(result.x[-1])


# ------------------------------------------------------------------ 1. the restricted prototype
def test_restricted_prototype(d, words):
    li, gi, w = d["label_index"], d["group_ids"], d["weights"]
    rng = np.random.default_rng(1)
    train = np.sort(rng.choice(N, size=300, replace=False))
    queries = np.setdiff1d(np.arange(N), train)[:25]
    for name, x in (("dense", d["dense"]), ("sparse", db.as_float64(words))):
        full = np.asarray(x.todense()) if sparse.issparse(x) else x
        got, present = db.restricted_prototype(x[train], li[train], w[train], L, x[queries])
        expect = np.stack([brute_prototype(full, li, w, L, train.tolist(), q) for q in queries])
        check(bool(np.all(present)) and float(np.abs(got - expect).max()) <= 1e-12,
              f"restricted_prototype equals the brute-force weighted-sum cosine, {name} (gap {float(np.abs(got - expect).max()):.1e})")
    keep = train[li[train] != 3]
    got, present = db.restricted_prototype(d["dense"][keep], li[keep], w[keep], L, d["dense"][queries])
    check(not present[3] and bool(np.all(got[:, 3] == db.ABSENT_SCORE)) and bool(np.all(got[:, present] > -1.5)),
          "a label without training rows scores ABSENT_SCORE, every other label a cosine")
    probe = np.arange(0, N, 11)
    gap_dense = db.check_restricted_scorer(d["dense"], li, gi, w, L, probe, dense_reference=True)
    gap_sparse = db.check_restricted_scorer(db.as_float64(words), li, gi, w, L, probe)
    check(gap_dense <= 1e-9 and gap_sparse <= 1e-9, f"check_restricted_scorer passes on dense and sparse (gaps {gap_dense:.1e}, {gap_sparse:.1e})")
    expect = brute_leave_group_out(d["dense"], d, probe[:8])
    got = np.stack([db.restricted_prototype(d["dense"][gi != gi[q]], li[gi != gi[q]], w[gi != gi[q]], L, d["dense"][[q]])[0][0] for q in probe[:8]])
    check(float(np.abs(got - expect).max()) <= 1e-12, "all songs outside the query's group as training rows = brute-force leave-group-out")
    original = db.restricted_prototype
    db.restricted_prototype = lambda xt, yt, wt, lc, xq: original(xt, yt, np.ones(len(yt)), lc, xq)
    try:
        failed = raises_system_exit(lambda: db.check_restricted_scorer(d["dense"], li, gi, w, L, np.arange(N)))
    finally:
        db.restricted_prototype = original
    check(failed, "check_restricted_scorer raises SystemExit when the scorer ignores the weights")


# ------------------------------------------------------------------ 2. objective and sample weights
def test_objective(d):
    rng = np.random.default_rng(2)
    rows = np.sort(rng.choice(N, size=140, replace=False))
    x, y = d["dense"][rows][:, :6] * 3.0, d["label_index"][rows] % 4
    s = rng.choice([1.0, 0.5, 1.0 / 3.0], size=len(rows))
    x_new = rng.standard_normal((30, 6))
    for kind in db.CLASSIFIERS:
        for C in (0.1, 10.0):
            model = db.fit_ovr(kind, C, x, y, s, 4, n_jobs=1, tol=1e-10)
            worst = 0.0
            for label in range(4):
                coef, intercept = brute_objective_fit(kind, C, x, (y == label).astype(np.float64), s)
                worst = max(worst, float(np.abs(db.decision(model, x_new)[:, label] - (x_new @ coef + intercept)).max()))
            check(worst <= 2e-4, f"{kind} C={C:g}: decision values equal the scipy minimiser of the weighted objective (gap {worst:.1e})")
        doubled = np.r_[np.arange(len(rows)), np.arange(0, len(rows), 3)]
        weight_two = np.ones(len(rows))
        weight_two[::3] = 2.0
        a = db.fit_ovr(kind, 1.0, x, y, weight_two, 4, tol=1e-10)
        b = db.fit_ovr(kind, 1.0, x[doubled], y[doubled], np.ones(len(doubled)), 4, tol=1e-10)
        gap = float(np.abs(db.decision(a, x_new) - db.decision(b, x_new)).max())
        check(gap <= 1e-5, f"{kind}: a row of sample weight two equals the row entered twice (gap {gap:.1e})")
        model = db.fit_ovr(kind, 10.0, x, y, s, 4)
        own = np.mean([db.decision(model, x[y == l])[:, l].mean() for l in range(4)])
        other = np.mean([db.decision(model, x[y != l])[:, l].mean() for l in range(4)])
        check(own > other, f"{kind}: a label's decision value is higher on its own rows (the fitted label is the positive class)")
    svc, lr = db.make_estimator("linear_svc", 1.0), db.make_estimator("logistic_regression", 1.0)
    check(isinstance(svc, LinearSVC) and svc.dual is False and svc.loss == "squared_hinge" and svc.penalty == "l2",
          "linear_svc is a primal L2 squared-hinge LinearSVC")
    check(isinstance(lr, LogisticRegression) and lr.solver == "liblinear" and lr.dual is False, "logistic_regression is primal liblinear")


# ------------------------------------------------------------------ 3. the one-vs-rest loop
def test_ovr_loop(d, words):
    li, w, fold = d["label_index"], d["weights"], d["fold"]
    train, test = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    x_train, x_test = db.as_float64(d["dense"][train]), db.as_float64(d["dense"][test])
    gap = db.check_native_ovr(x_train, li[train], w[train], L, x_test)
    check(gap <= 1e-6, f"check_native_ovr: the loop equals sklearn's native one-vs-rest LinearSVC (gap {gap:.1e})")
    native = db.make_estimator("linear_svc", 1.0).fit(x_train, li[train], sample_weight=w[train])
    other = db.fit_ovr("linear_svc", 2.0, x_train, li[train], w[train], L)
    far = float(np.abs(native.decision_function(x_test) - db.decision(other, x_test)).max())
    check(far > 1e-3, f"the comparison is sensitive: a loop at another C is {far:.1e} away")
    unweighted = db.fit_ovr("linear_svc", 1.0, x_train, li[train], np.ones(len(train)), L)
    far = float(np.abs(native.decision_function(x_test) - db.decision(unweighted, x_test)).max())
    check(far > 1e-4, f"and a loop without the sample weights is {far:.1e} away")
    keep = train[li[train] != 7]
    model = db.fit_ovr("logistic_regression", 1.0, db.as_float64(d["dense"][keep]), li[keep], w[keep], L)
    scores = db.decision(model, x_test)
    check(not model["present"][7] and bool(np.all(scores[:, 7] == db.ABSENT_SCORE)) and bool(np.all(np.abs(scores[:, model["present"]]) < 1e6)),
          "a label absent from the fit scores ABSENT_SCORE; present labels score finite decision values")
    truth = np.full(len(test), 7)
    check(bool(np.all(db.ranks_of(scores, truth) == L)), "an absent true label ranks last, without NaN")
    xs_train, xs_test = db.as_float64(words[train]), db.as_float64(words[test])
    for kind in db.CLASSIFIERS:
        one = db.fit_ovr(kind, 1.0, xs_train, li[train], w[train], L, n_jobs=1)
        four = db.fit_ovr(kind, 1.0, xs_train, li[train], w[train], L, n_jobs=4)
        check(np.array_equal(one["weight"], four["weight"]) and np.array_equal(one["bias"], four["bias"]),
              f"{kind}: one thread and four threads give bit-identical fits (sparse space)")
        check(one["unconverged"] == 0 and 0 < one["max_n_iter"] < db.MAX_ITER, f"{kind}: converged, n_iter {one['max_n_iter']}")
    check(raises_system_exit(lambda: db.fit_ovr("linear_svc", 1.0, x_train, np.zeros(len(train), dtype=np.int64), w[train], L)),
          "fit_ovr refuses a fitting set with one label")


# ------------------------------------------------------------------ 4. macro F1 and predictions
def test_metrics():
    rng = np.random.default_rng(4)
    truth = rng.integers(0, 9, size=500)
    truth[truth == 5] = 4                               # label 5 never true
    pred = np.where(rng.random(500) < 0.6, truth, rng.integers(0, 12, size=500))
    pred[pred == 10] = 0                                # label 10 in neither; label 11 only predicted
    weights = rng.choice([1.0, 0.5, 0.25], size=500)
    got = db.macro_f1(pred, truth, np.ones(500), 12)
    expect = f1_score(truth, pred, average="macro")
    check(abs(got - expect) <= 1e-12, f"macro_f1 equals sklearn's macro F1 ({got:.6f})")
    got = db.macro_f1(pred, truth, weights, 12)
    expect = f1_score(truth, pred, average="macro", sample_weight=weights)
    check(abs(got - expect) <= 1e-12, f"weighted macro_f1 equals sklearn's with sample_weight ({got:.6f})")
    scores = np.round(rng.random((400, 7)), 1)          # one decimal: many exact ties
    truth = rng.integers(0, 7, size=400)
    ranks = db.ranks_of(scores, truth)
    expect = np.asarray([brute_rank(scores[i], int(truth[i])) for i in range(400)])
    check(np.array_equal(ranks, expect), "ranks_of equals the brute-force rank with ties by label index")
    pred = db.predictions(scores, truth, ranks)
    check(np.array_equal(pred == truth, ranks == 1), "predictions() hits exactly when the rank is one, under exact ties")
    top = np.argmax(scores.astype(np.float32), axis=1)
    check(np.array_equal(pred[ranks > 1], top[ranks > 1]) and bool(np.all(pred[ranks > 1] != truth[ranks > 1])),
          "a missed query's prediction is the first label of highest score")
    summary = db.summarise(scores, truth)
    check(abs(summary["mrr"] - float(np.mean(1.0 / expect))) <= 1e-12 and abs(summary["recall_at_1"] - float(np.mean(expect == 1))) <= 1e-12
          and abs(summary["label_macro_f1"] - f1_score(truth, pred, average="macro")) <= 1e-12,
          "summarise: MRR, R@1 and macro F1 equal their brute-force definitions")


# ------------------------------------------------------------------ 5. the bootstrap
def test_bootstrap(d):
    li, gi, w = d["label_index"], d["group_ids"], d["weights"]
    rng = np.random.default_rng(5)
    mask = np.ones(N, dtype=bool)
    rr = {"a": 1.0 / rng.integers(1, 6, size=N), "b": 1.0 / rng.integers(1, 9, size=N)}
    published = paired_group_bootstrap(rr, w, gi, mask, [("a", "b")])[0]
    position, draws, group_count = db.group_draws(gi, mask)
    counts = db.multiplicity(draws, group_count)
    check(counts.shape == (db.REPLICATES, group_count) and bool(np.all(counts.sum(axis=1) == group_count)),
          "multiplicity: every replicate draws as many groups as there are")
    weight_sum = np.bincount(position, weights=w, minlength=group_count)
    replicate = {s: (counts @ np.bincount(position, weights=v * w, minlength=group_count)) / (counts @ weight_sum) for s, v in rr.items()}
    low, high = np.percentile(replicate["a"] - replicate["b"], [2.5, 97.5])
    check([round(float(low), 4), round(float(high), 4)] == published["ci95"],
          f"group_draws + multiplicity reproduce paired_group_bootstrap's interval {published['ci95']}")
    pred = {"a": np.where(rng.random(N) < 0.7, li, rng.integers(0, L, size=N)), "b": np.where(rng.random(N) < 0.4, li, rng.integers(0, L, size=N))}
    replicates = 120
    position, draws, group_count = db.group_draws(gi, mask, replicates)
    counts = db.multiplicity(draws, group_count)
    members = [np.flatnonzero(position == g) for g in range(group_count)]
    brute = {}
    for s in pred:
        got = db.macro_f1_replicates(pred[s], li, w, position, counts, L)
        values = []
        for r in range(replicates):
            rows = np.concatenate([members[g] for g in draws[r]])
            values.append(f1_score(li[rows], pred[s][rows], average="macro", sample_weight=w[rows]))
        brute[s] = np.asarray(values)
        check(float(np.abs(got - brute[s]).max()) <= 1e-12, f"macro_f1_replicates equals sklearn's weighted macro F1 on resampled rows, system {s}")
    out = db.paired_group_bootstrap_macro_f1(pred, li, w, gi, L, [("a", "b")], replicates)[0]
    low, high = np.percentile(brute["a"] - brute["b"], [2.5, 97.5])
    point = f1_score(li, pred["a"], average="macro", sample_weight=w) - f1_score(li, pred["b"], average="macro", sample_weight=w)
    check(out["ci95"] == [round(float(low), 4), round(float(high), 4)] and out["label_macro_f1_difference"] == round(point, 4)
          and out["excludes_zero"] == bool(low > 0 or high < 0) and (out["system"], out["minus"]) == ("a", "b"),
          f"paired_group_bootstrap_macro_f1: brute-force point {point:+.4f} and interval [{low:+.4f}, {high:+.4f}]")
    small = li.copy()
    small[:] = np.where(np.arange(N) < 3, 9, li % 9)    # label 9 lives in few groups: absent from some resamples
    got = db.macro_f1_replicates(small, small, w, position, counts, L)
    check(bool(np.all(np.abs(got - 1.0) <= 1e-12)), "a label missing from a resample is left out of that replicate's macro mean")
    summaries = {s: {"rr": rr[s], "top1": (rr[s] == 1.0).astype(np.float64), "pred": pred[s]} for s in rr}
    table = db.contrasts_for(summaries, li, w, gi, L, [("a", "b")])
    r1 = paired_group_bootstrap({s: v["top1"] for s, v in summaries.items()}, w, gi, mask, [("a", "b")])[0]
    check(table["mrr"][0] == published and table["recall_at_1"][0]["recall_at_1_difference"] == r1["mrr_difference"]
          and table["recall_at_1"][0]["ci95"] == r1["ci95"] and "mrr_difference" not in table["recall_at_1"][0]
          and set(table) == set(db.METRICS), "contrasts_for: MRR and R@1 through paired_group_bootstrap itself, F1 beside them")


# ------------------------------------------------------------------ 6. small pieces
def test_small_pieces(d):
    check(db.select_c({0.1: 0.30, 1.0: 0.35, 10.0: 0.35}) == 1.0 and db.select_c({0.1: 0.2, 1.0: 0.2, 10.0: 0.2}) == 0.1
          and db.select_c({0.1: 0.1, 1.0: 0.2, 10.0: 0.3}) == 10.0, "select_c: the highest inner MRR, ties to the smaller C")
    distinct = set()
    fine = True
    for k in range(FOLDS):
        halves = db.inner_halves([f for f in range(FOLDS) if f != k])
        fine &= len(halves) == 2 and halves[0] == halves[1][::-1]
        for fit_folds, val_folds in halves:
            fine &= k not in fit_folds + val_folds and not set(fit_folds) & set(val_folds) and len(fit_folds) == len(val_folds) == 2
            distinct.add(fit_folds)
        fine &= sorted(halves[0][0] + halves[0][1]) == [f for f in range(FOLDS) if f != k]
    check(bool(fine) and len(distinct) == 6, "inner_halves: two disjoint halves of the training folds, never the test fold, six distinct fitting halves")
    good = {"system": "x", "minus": "fold_prototype", "mrr_difference": 0.02, "ci95": [0.01, 0.03], "excludes_zero": True}
    check(db.verdict(good) == "beats the prototype" and db.verdict({**good, "mrr_difference": -0.02, "ci95": [-0.03, -0.01]}) == "loses to the prototype"
          and db.verdict({**good, "ci95": [-0.001, 0.03], "excludes_zero": False}) == "no clear difference"
          and db.verdict({**good, "mrr_difference": -0.0, "ci95": [-0.0, 0.0], "excludes_zero": False}) == "no clear difference",
          "verdict: beats / loses only with the interval clear of zero")
    contrasts = [{**good, "system": "linear_svc"}, {**good, "system": "logistic_regression", "mrr_difference": -0.01, "ci95": [-0.02, 0.0], "excludes_zero": False}]
    reading = db.read_space({"linear_svc": 0.5, "logistic_regression": 0.6, "fold_prototype": 0.4}, contrasts)
    check(reading["better_classifier"] == "logistic_regression" and reading["reading"] == "logistic_regression no clear difference"
          and reading["by_classifier"]["linear_svc"]["verdict"] == "beats the prototype", "read_space: a space reads by its better classifier")
    check(db.ordering({"b": 0.3, "a": 0.3, "c": 0.5}) == ["c", "a", "b"], "ordering: by MRR, ties by name")
    published = {s: (0.4, 0.3) for s in db.SPACES}
    quiet = lambda message: None                        # noqa: E731
    record = db.check_against_published({s: (0.4004, 0.2997) for s in db.SPACES}, published, None, db.TOLERANCE, log=quiet)
    check(all(r["passed"] for r in record.values()), "check_against_published passes inside the tolerance")
    check(raises_system_exit(lambda: db.check_against_published({s: (0.4007, 0.3) for s in db.SPACES}, published, None, db.TOLERANCE, log=quiet)),
          "check_against_published: SystemExit when an unfitted space is off by 7e-4")
    check(raises_system_exit(lambda: db.check_against_published({s: (0.4, 0.3) for s in db.SPACES}, published,
                                                                {s: (0.41, 0.3) for s in db.SPACES}, db.TOLERANCE, log=quiet)),
          "check_against_published: SystemExit when the pinned literals disagree")
    check(db.PINNED == {"semantic_raw": (0.2997, 0.2019), "characters": (0.4266, 0.3342), "words": (0.4963, 0.4029),
                        "semantic_whitened": (0.4164, 0.3179)} and max(db.TOLERANCE.values()) <= 0.002,
          "the pinned published numbers and tolerances are the specified ones")
    record = db.check_folds(d)
    check(record["distinct_inner_fitting_halves"] == 6 and record["folds_disjoint_by_row_and_group"], "check_folds passes on the synthetic corpus")
    leaky = dict(d)
    leaky["fold"] = d["fold"].copy()
    pair = np.flatnonzero(d["group_ids"] == d["group_ids"][0])
    leaky["fold"][pair[0]] = (d["fold"][pair[0]] + 1) % FOLDS
    check(len(pair) > 1 and raises_system_exit(lambda: db.check_folds(leaky)), "check_folds: SystemExit when a leakage group straddles two folds")
    lonely = dict(d)
    lonely["label_index"] = np.where(d["fold"] == 2, L - 1, d["label_index"] % (L - 1))
    lonely["weights"] = np.ones(N)
    check(raises_system_exit(lambda: db.check_folds(lonely)), "check_folds: SystemExit when a label is missing from a training set")


# ------------------------------------------------------------------ 7. end to end
def direct_classifier(kind, x_by_fit, d, k, c_grid):
    """Without the cache: per inner half fit and validate, select, fit on the training folds,
    one sklearn binary fit per label. x_by_fit(rows) returns the representation fitted on rows."""
    li, w, fold = d["label_index"], d["weights"], d["fold"]
    training_folds = [f for f in range(FOLDS) if f != k]
    a, b = training_folds[:2], training_folds[2:]

    def fit_and_score(C, fit_rows, score_rows):
        x = x_by_fit(fit_rows)
        out = np.full((len(score_rows), L), db.ABSENT_SCORE)
        for label in sorted(set(li[fit_rows].tolist())):
            model = db.make_estimator(kind, C).fit(x[fit_rows], (li[fit_rows] == label).astype(int), sample_weight=w[fit_rows])
            out[:, label] = model.decision_function(x[score_rows])
        return out

    inner = {}
    for C in c_grid:
        rr = []
        for fit_folds, val_folds in ((a, b), (b, a)):
            fit_rows, val_rows = np.flatnonzero(np.isin(fold, fit_folds)), np.flatnonzero(np.isin(fold, val_folds))
            scores = fit_and_score(C, fit_rows, val_rows)
            rr += [1.0 / brute_rank(scores[i], int(li[q])) for i, q in enumerate(val_rows)]
        inner[C] = float(np.mean(rr))
    best = sorted(c_grid, key=lambda C: (-inner[C], C))[0]
    train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    return best, inner, fit_and_score(best, train, test)


def test_end_to_end(d, chars, words):
    li, gi, w, fold, dense = d["label_index"], d["group_ids"], d["weights"], d["fold"], d["dense"]
    dense_words, dense_chars = np.asarray(words.todense(), dtype=np.float64), np.asarray(chars.todense(), dtype=np.float64)

    def whitened_on(rows):
        mean, matrix, _ = fit_transform("within_author_whitening", dense[rows], li[rows], w[rows], np.random.default_rng(0))
        return unit_rows((dense - mean) @ matrix.T)

    # "published" numbers from brute-force leave-group-out over all songs
    everything = list(range(N))
    brute = {"semantic_raw": brute_leave_group_out(dense, d, everything), "characters": brute_leave_group_out(dense_chars, d, everything),
             "words": brute_leave_group_out(dense_words, d, everything), "semantic_whitened": np.zeros((N, L))}
    for k in range(FOLDS):
        test = np.flatnonzero(fold == k)
        brute["semantic_whitened"][test] = brute_leave_group_out(whitened_on(np.flatnonzero(fold != k)), d, test.tolist())
    published = {}
    for space, scores in brute.items():
        ranks = np.asarray([brute_rank(scores[i], int(li[i])) for i in range(N)])
        published[space] = (round(float(np.mean(1.0 / ranks)), 4), round(float(np.mean(ranks == 1)), 4))

    debug, lines = {}, []
    payload = db.analyse(d, chars, words, published, n_jobs=2, pinned=None, debug=debug, log=lines.append)
    check(all(r["passed"] and r["gap"] <= 1e-4 for r in payload["checks"]["against_published"].values()),
          "the reproduction check passes against brute-force leave-group-out numbers in all four spaces")
    wrong = {**published, "words": (published["words"][0] + 0.01, published["words"][1])}
    check(raises_system_exit(lambda: db.analyse(d, chars, words, wrong, pinned=None, checks_only=True, log=lambda m: None)),
          "analyse stops with SystemExit before any new number when a published number is not reproduced")
    only = db.analyse(d, chars, words, published, pinned=None, checks_only=True, log=lambda m: None)
    check(set(only) == {"checks", "minutes"}, "checks_only returns the checks and nothing else")

    # fold_prototype against a brute-force per-fold loop
    matrices = {"semantic_raw": lambda rows: dense, "characters": lambda rows: dense_chars, "words": lambda rows: dense_words,
                "semantic_whitened": whitened_on}
    for space in db.SPACES:
        worst = 0.0
        for k in range(FOLDS):
            train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            x = matrices[space](train)
            for q in test[:12]:
                worst = max(worst, float(np.abs(debug["scores"][space]["fold_prototype"][q] - brute_prototype(x, li, w, L, train.tolist(), int(q))).max()))
        check(worst <= 1e-9, f"{space}: fold_prototype equals the brute-force training-fold profile cosine (gap {worst:.1e})")
        gap = float(np.abs(debug["scores"][space]["protocol_prototype"] - brute[space]).max())
        check(gap <= 1e-5, f"{space}: protocol_prototype equals brute-force leave-group-out over all songs (gap {gap:.1e})")

    # the classifiers against a direct recomputation without the cache
    for space in ("words", "semantic_whitened"):
        for kind in db.CLASSIFIERS:
            same_c, same_inner, worst = True, True, 0.0
            for k in range(FOLDS):
                best, inner, scores = direct_classifier(kind, matrices[space], d, k, db.C_GRID)
                entry = payload["spaces"][space]["selected_C"][kind][str(k)]
                same_c &= entry["selected_C"] == best
                same_inner &= all(abs(entry["inner_mrr_by_C"][f"{C:g}"] - round(inner[C], 4)) <= 1e-12 for C in db.C_GRID)
                worst = max(worst, float(np.abs(debug["scores"][space][kind][fold == k] - scores).max()))
            check(bool(same_c) and bool(same_inner), f"{space} {kind}: selected C and inner MRRs equal the direct nested computation in every fold")
            check(worst <= 1e-8, f"{space} {kind}: test-fold decision values equal direct per-label sklearn fits (gap {worst:.1e})")

    # the test fold's labels play no part in what is fitted or selected for it
    k = 1
    scrambled = dict(d)
    rng = np.random.default_rng(7)
    new_labels = li.copy()
    test = np.flatnonzero(fold == k)
    new_labels[test] = rng.integers(0, L, size=len(test))
    scrambled["label_index"] = new_labels
    component_size = Counter(zip(gi.tolist(), new_labels.tolist()))
    scrambled["weights"] = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(gi, new_labels)])
    for space, fixed in (("semantic_raw", dense), ("semantic_whitened", None)):
        base = db.score_space(space, d, fixed, 1, log=lambda m: None)
        moved = db.score_space(space, scrambled, fixed, 1, log=lambda m: None)
        same = all(base["selection"][kind][str(k)] == moved["selection"][kind][str(k)] for kind in db.CLASSIFIERS)
        gap = max(float(np.abs(base["scores"][name][test] - moved["scores"][name][test]).max()) for name in base["scores"])
        check(same and gap == 0.0, f"{space}: scrambling the test fold's labels leaves its selected C and its scores unchanged")
        check(base["inner_fits"] == 6 * len(db.CLASSIFIERS) * len(db.C_GRID), f"{space}: one inner fit per distinct half, classifier and C")

    # the payload
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    check("song_" not in text and "NaN" not in text and "Infinity" not in text, "the payload is JSON, aggregate only, without NaN")
    systems_ok = all(set(payload["spaces"][s]["systems"]) == set(db.PROTOTYPES + db.CLASSIFIERS)
                     and all(set(v) == set(db.METRICS) and all(0.0 <= x <= 1.0 for x in v.values()) for v in payload["spaces"][s]["systems"].values())
                     for s in db.SPACES)
    check(systems_ok, "every space reports the four systems with MRR, R@1 and macro F1 in [0, 1]")
    pairs_ok = all([(c["system"], c["minus"]) for c in payload["spaces"][s]["contrasts"][m]] == list(db.PAIRS) and
                   all(len(c["ci95"]) == 2 and c["ci95"][0] <= c["ci95"][1] for c in payload["spaces"][s]["contrasts"][m])
                   for s in db.SPACES for m in db.METRICS)
    check(pairs_ok, "every space carries the six contrasts with an interval for each of the three metrics")
    for space in db.SPACES:
        entry = payload["spaces"][space]
        scores = debug["scores"][space]
        for name in entry["systems"]:
            ranks = np.asarray([brute_rank(scores[name][i], int(li[i])) for i in range(N)])
            if abs(entry["systems"][name]["mrr"] - round(float(np.mean(1.0 / ranks)), 4)) > 1e-12:
                check(False, f"{space} {name}: reported MRR differs from the brute-force ranks")
        rr = {name: 1.0 / db.ranks_of(scores[name], li) for name in scores}
        expect = paired_group_bootstrap(rr, w, gi, np.ones(N, dtype=bool), list(db.PAIRS))
        check(entry["contrasts"]["mrr"] == expect, f"{space}: the MRR contrasts are paired_group_bootstrap's on the reported systems' reciprocal ranks")
        for kind in db.CLASSIFIERS:
            c = next(c for c in expect if (c["system"], c["minus"]) == (kind, "fold_prototype"))
            want = ("beats the prototype" if c["ci95"][0] > 0 else "loses to the prototype" if c["ci95"][1] < 0 else "no clear difference")
            if c["excludes_zero"] or want == "no clear difference":
                check(entry["reading"]["by_classifier"][kind]["verdict"] == want, f"{space} {kind}: the reading follows the rule ({want})")
        mrr = {kind: float(np.mean(rr[kind])) for kind in db.CLASSIFIERS}
        check(entry["reading"]["better_classifier"] == max(db.CLASSIFIERS, key=lambda kind: (mrr[kind], -db.CLASSIFIERS.index(kind))),
              f"{space}: the better classifier is the one with the higher MRR")
        check(min(entry["systems"][name]["mrr"] for name in entry["systems"]) > 0.4, f"{space}: every system finds the planted labels (MRR > 0.4)")
    orderings = payload["space_orderings"]
    best = orderings["best_classifier"]
    mean_mrr = {kind: np.mean([np.mean(1.0 / db.ranks_of(debug["scores"][s][kind], li)) for s in db.SPACES]) for kind in db.CLASSIFIERS}
    check(best == max(db.CLASSIFIERS, key=lambda kind: mean_mrr[kind]) and set(orderings["orderings"]) == {best, "fold_prototype", "protocol_prototype"}
          and all(sorted(order) == sorted(db.SPACES) for order in orderings["orderings"].values())
          and all(len(v) == len(db.SPACES) - 1 for v in orderings["adjacent_contrasts_mrr"].values()),
          "space orderings: under the best classifier and both prototypes, with the adjacent contrasts")
    for name, order in orderings["orderings"].items():
        values = [float(np.mean(1.0 / db.ranks_of(debug["scores"][s][name], li))) for s in order]
        check(values == sorted(values, reverse=True), f"ordering under {name} is by descending MRR")
    check(payload["summary"]["unconverged_binary_fits_in_scored_models"] == 0, "no scored model reached max_iter")
    edge = payload["summary"]["folds_with_selected_C_at_grid_edge"]
    recount = {s: {kind: sum(1 for e in payload["spaces"][s]["selected_C"][kind].values() if e["selected_C"] in (min(db.C_GRID), max(db.C_GRID)))
                   for kind in db.CLASSIFIERS} for s in db.SPACES}
    check(edge == recount and all(e["at_grid_edge"] == (e["selected_C"] in (0.1, 10.0)) for s in db.SPACES for kind in db.CLASSIFIERS
                                  for e in payload["spaces"][s]["selected_C"][kind].values()),
          "the folds whose selected C sits on a grid edge are counted as reported")
    check(any("check 1" in line for line in lines) and lines.index(next(l for l in lines if "check 1" in l)) < lines.index(next(l for l in lines if l.startswith("=="))),
          "the reproduction check is logged before any classifier is fitted")


def test_sensitivity(d, words):
    """The declared sensitivity system: selected on its own grid by the same inner halves,
    reusing the cached fits of shared (kind, C), reported apart from the rule."""
    fixed = words
    base = db.score_space("words", d, fixed, 1, log=lambda m: None)
    wide = db.score_space("words", d, fixed, 1, log=lambda m: None, sensitivity=db.SENSITIVITY)
    same = all(np.array_equal(base["scores"][k], wide["scores"][k]) for k in base["scores"])
    check(same, "adding the sensitivity system leaves every rule system's scores unchanged")
    grid = db.SENSITIVITY["logistic_regression_wide"][1]
    extra = len([C for C in grid if C not in db.C_GRID])
    check(wide["inner_fits"] == base["inner_fits"] + 6 * extra, "the sensitivity grid reuses the cached fits of shared (kind, C)")
    ok = all(wide["selection"]["logistic_regression_wide"][str(k)]["selected_C"] in grid for k in range(FOLDS))
    check(ok, "the sensitivity system selects its C from its own grid")
    shared = [C for C in grid if C in db.C_GRID]
    for k in range(FOLDS):
        a = wide["selection"]["logistic_regression_wide"][str(k)]["inner_mrr_by_C"]
        b = wide["selection"]["logistic_regression"][str(k)]["inner_mrr_by_C"]
        if not all(a[f"{C:g}"] == b[f"{C:g}"] for C in shared):
            check(False, "a shared C must give the same inner MRR under both grids")
    check(True, "a shared C gives the same inner MRR under both grids")
    good = {"mrr_difference": 0.0, "ci95": [0.0001, 0.0002], "excludes_zero": True}
    check(db.verdict(good) == "beats the prototype", "the verdict is decided by the interval, not the rounded point")


def main() -> int:
    d, chars, words = synthetic(20260921)
    test_restricted_prototype(d, words)
    test_objective(d)
    test_ovr_loop(d, words)
    test_metrics()
    test_bootstrap(d)
    test_small_pieces(d)
    test_end_to_end(d, chars, words)
    test_sensitivity(d, words)
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED")
        for message in FAILURES:
            print("  " + message)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
