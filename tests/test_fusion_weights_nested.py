#!/usr/bin/env python3
"""Synthetic unit test for fusion_weights_nested_v3: random data, no corpus, no private files.

Checks, each against an independent definition:
  1. fuse(zA, zB, 0.5) is bitwise the published (zscore(A) + zscore(B)) / 2;
  2. grid_curve equals a brute-force grid search: for every w, ranks by a literal loop of
     the builder's tie rule (v1.rank_system on float32 scores) and the per-(group, label)-
     weighted MRR by an explicit sum; choose_weight is the brute-force argmax, with the tie
     rule (nearest 0.5, then the smaller w) on constructed curves;
  3. outer_scores equals a literal transcription of the published fold loop
     (identity_probe_v2 / word_space_probe_v2) and a brute-force leave-group-out cosine;
     every outer fit's recorded training folds exclude its fold;
  4. inner_scores: the fold-a rows of inner[b] and the fold-b rows of inner[a] equal the
     scores of one hand-fitted space on the three folds outside {a, b}; a fold's own rows
     are NaN, every training row finite; every inner fit's recorded folds exclude both;
  5. word_svd_fit and chunk_whitening_fit equal hand-fitted transcriptions of the published
     word-SVD and chunk-whitening paths;
  6. blindness: perturbing the test fold's outer rows of a raw pair leaves that fold's chosen
     weight unchanged (while its test curve changes); for a pair with a fitted component,
     perturbing every outer row leaves every chosen weight unchanged, and perturbing
     inner[k] leaves the other folds' weights unchanged; the fitted test rows equal
     fuse(zA, zB, chosen w) recomputed by hand;
  7. the mask: zero-variance rows and uncovered rows are excluded, their ranks never
     reported, and the reported query count is the mask size;
  8. rhyme_ranks_published applies the two published rank policies;
  9. reading_of on constructed intervals gives every verdict of the reading rule;
 10. an end-to-end analyse on two raw and two fitted spaces: JSON-serialisable, systems'
     MRR equal the plain mean of reciprocal ranks, contrast points equal the weighted-mean
     differences, fusion_equal equals the published expression's ranks, the selection-
     inflation curves are never below the fitted weight's on their own fold, and the
     description-only system appears only where a component is fitted.

    set CHINESE_RAP_CORPUS=v3
    python test_fusion_weights_nested.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

os.environ.setdefault("CHINESE_RAP_CORPUS", "v3")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import fusion_weights_nested_v3 as fw  # noqa: E402
import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402

N, L, DIM = 300, 8, 16
LEX_FEATURES, SVD_K = 60, 8


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {message}")
    print(f"ok: {message}")


def synthetic(seed: int):
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(N // 2), 2)
    group_ids[-6:] = group_ids[-6]                                 # one group of six songs
    label_index = rng.integers(0, L, size=N)
    for l in range(L):
        if len(set(group_ids[label_index == l].tolist())) < 6:
            raise RuntimeError("regenerate: a label with too few groups")
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    n_groups = int(group_ids.max()) + 1
    fold = rng.integers(0, FOLDS, size=n_groups)[group_ids]
    dense = rng.standard_normal((N, DIM)) + 0.8 * rng.standard_normal((L, DIM))[label_index]
    dense = unit_rows(dense)
    # a sparse non-negative "word" matrix with unit rows, some label structure
    lexical = rng.random((N, LEX_FEATURES)) * (rng.random((N, LEX_FEATURES)) < 0.25)
    lexical += 0.6 * (rng.random((L, LEX_FEATURES)) < 0.15)[label_index]
    lexical = sparse.csr_matrix(unit_rows(lexical).astype(np.float32))
    # chunks: one to three per song
    counts = rng.integers(1, 4, size=N)
    chunk_song = np.repeat(np.arange(N), counts)
    chunk_vec = unit_rows(rng.standard_normal((len(chunk_song), DIM)) + 0.8 * rng.standard_normal((L, DIM))[label_index[chunk_song]])
    chunk_weight = weights[chunk_song] / np.bincount(chunk_song)[chunk_song]
    return dict(group_ids=group_ids, label_index=label_index, weights=weights, fold=fold, dense=dense,
                lexical=lexical, chunk_song=chunk_song, chunk_vec=chunk_vec, chunk_weight=chunk_weight)


def random_scores(rng, label_index, signal):
    scores = rng.standard_normal((N, L))
    scores[np.arange(N), label_index] += signal
    return scores


def brute_ranks(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The builder's tie rule, one query at a time, on float32 scores."""
    s = scores.astype(np.float32)
    out = np.zeros(len(truth), dtype=np.int64)
    for q in range(len(truth)):
        t = s[q, truth[q]]
        rank = 1
        for l in range(s.shape[1]):
            if s[q, l] > t + 1e-12:
                rank += 1
            elif abs(s[q, l] - t) <= 1e-12 and l < truth[q]:
                rank += 1
        out[q] = rank
    return out


def brute_weighted_mrr(ranks, weights) -> float:
    num = sum(float(weights[i]) / float(ranks[i]) for i in range(len(ranks)))
    return num / sum(float(x) for x in weights)


def brute_lgo(x, label_index, group_ids, weights, queries):
    out = np.zeros((len(queries), L))
    for row, q in enumerate(queries):
        for l in range(L):
            members = [i for i in range(len(label_index)) if label_index[i] == l and group_ids[i] != group_ids[q]]
            profile = sum(weights[i] * x[i] for i in members)
            out[row, l] = float(x[q] @ profile) / float(np.linalg.norm(profile))
    return out


def whitening_fit(dense, label_index, weights):
    def fit(train, seed):
        mean, matrix, info = fit_transform("within_author_whitening", dense[train], label_index[train], weights[train],
                                           np.random.default_rng(seed))
        return unit_rows((dense - mean) @ matrix.T), info
    return fit


def main() -> int:
    d = synthetic(11)
    li, gi, w, fold = d["label_index"], d["group_ids"], d["weights"], d["fold"]
    rng = np.random.default_rng(5)

    # 1. the equal weight is the published expression, bitwise
    a_scores, b_scores = random_scores(rng, li, 1.0), random_scores(rng, li, 0.5)
    za, zb = v1.zscore_rows(a_scores), v1.zscore_rows(b_scores)
    check(np.array_equal(fw.fuse(za, zb, 0.5), fw.published_equal_fusion(a_scores, b_scores)),
          "fuse(zA, zB, 0.5) is bitwise the published (zA + zB) / 2")
    check(np.array_equal(fw.fuse(za, zb, 0.0), zb) and np.array_equal(fw.fuse(za, zb, 1.0), za),
          "fuse at w = 0 is B and at w = 1 is A")

    # 2. the grid curve and the selection against brute force
    curve = fw.grid_curve(za, zb, li, w)
    for i, wt in enumerate(fw.GRID):
        fused = wt * za + (1.0 - wt) * zb
        ranks = brute_ranks(fused, li)
        if not np.array_equal(ranks, fw.ranks_of(fused, li)):
            raise SystemExit(f"FAILED: ranks_of differs from the literal tie-rule loop at w = {wt}")
        if abs(curve[i] - brute_weighted_mrr(ranks, w)) > 1e-12:
            raise SystemExit(f"FAILED: grid_curve differs from the brute-force weighted MRR at w = {wt}")
    print("ok: grid_curve equals the brute-force grid search under the builder's tie rule, all 21 weights")
    check(fw.choose_weight(curve) == fw.GRID[int(np.argmax(curve))] or curve.count(max(curve)) > 1,
          "choose_weight is the brute-force argmax")
    grid5 = (0.0, 0.25, 0.5, 0.75, 1.0)
    check(fw.choose_weight([0.3, 0.5, 0.5, 0.2, 0.1], grid5) == 0.5, "tie between 0.25 and 0.5 goes to 0.5")
    check(fw.choose_weight([0.3, 0.5, 0.4, 0.5, 0.1], grid5) == 0.25, "tie between 0.25 and 0.75 goes to the smaller w")
    check(fw.choose_weight([0.5, 0.3, 0.4, 0.3, 0.5], grid5) == 0.0, "tie between 0.0 and 1.0 goes to 0.0")
    check(abs(fw.weighted_mrr(1.0 / brute_ranks(za, li), w) - brute_weighted_mrr(brute_ranks(za, li), w)) <= 1e-12,
          "weighted_mrr equals the explicit weighted sum")

    # 3. outer_scores against the published fold loop and a brute-force leave-group-out cosine
    logs = []
    fit = whitening_fit(d["dense"], li, w)
    outer, fits = fw.outer_scores(fit, li, gi, w, L, fold, log=logs.append)
    expect = np.zeros((N, L))
    for k in range(FOLDS):
        train = fold != k
        queries = np.flatnonzero(~train)
        mean, matrix, _ = fit_transform("within_author_whitening", d["dense"][train], li[train], w[train], np.random.default_rng(SEED + k))
        x = unit_rows((d["dense"] - mean) @ matrix.T)
        expect[queries] = dense_leave_group_out(x, li, gi, w, L, queries)
        if k == 1:
            brute = brute_lgo(x, li, gi, w, queries[:8])
            check(float(np.abs(brute - outer[queries[:8]]).max()) <= 1e-9, "outer scores equal a brute-force leave-group-out cosine (fold 1, eight queries)")
    check(float(np.abs(outer - expect).max()) <= 1e-12, "outer_scores equals the published fold loop, transcribed")
    check(all(k not in fits[str(k)]["folds_in_training"] and len(fits[str(k)]["folds_in_training"]) == FOLDS - 1 for k in range(FOLDS)),
          "every outer fit records four training folds, its own excluded")
    check(all(fits[str(k)]["seed"] == SEED + k for k in range(FOLDS)), "outer fit seeds are SEED + k")

    # 4. inner_scores against one hand-fitted space per pair
    inner, ifits = fw.inner_scores(fit, li, gi, w, L, fold, log=logs.append)
    for a in range(FOLDS):
        for b in range(a + 1, FOLDS):
            train = (fold != a) & (fold != b)
            mean, matrix, _ = fit_transform("within_author_whitening", d["dense"][train], li[train], w[train],
                                            np.random.default_rng(fw.inner_seed(a, b)))
            x = unit_rows((d["dense"] - mean) @ matrix.T)
            rows_a, rows_b = np.flatnonzero(fold == a), np.flatnonzero(fold == b)
            got_a = inner[b][rows_a]
            got_b = inner[a][rows_b]
            exp_a = dense_leave_group_out(x, li, gi, w, L, rows_a)
            exp_b = dense_leave_group_out(x, li, gi, w, L, rows_b)
            if float(np.abs(got_a - exp_a).max()) > 1e-12 or float(np.abs(got_b - exp_b).max()) > 1e-12:
                raise SystemExit(f"FAILED: inner scores for the pair {a},{b} differ from the hand-fitted space")
            seen = ifits[f"{a},{b}"]["folds_in_training"]
            if a in seen or b in seen or len(seen) != FOLDS - 2 or ifits[f"{a},{b}"]["seed"] != fw.inner_seed(a, b):
                raise SystemExit(f"FAILED: inner fit {a},{b} records the wrong training folds or seed")
    print("ok: inner_scores equals one hand-fitted three-fold space per pair, both orderings; training folds and seeds recorded")
    for k in range(FOLDS):
        if not (np.all(np.isnan(inner[k][fold == k])) and np.all(np.isfinite(inner[k][fold != k]))):
            raise SystemExit(f"FAILED: inner[{k}] must be NaN on its own fold and finite elsewhere")
    print("ok: every inner matrix is NaN on its own fold and finite on the training folds")
    check(len({fw.inner_seed(a, b) for a in range(FOLDS) for b in range(a + 1, FOLDS)} | {SEED + k for k in range(FOLDS)}) == 15,
          "the fifteen fit seeds are distinct")

    # 5. the two published fit paths against hand-fitted transcriptions
    from sklearn.decomposition import TruncatedSVD
    word_fit = fw.word_svd_fit(d["lexical"], li, w, components=SVD_K)
    train = fold != 2
    got, info = word_fit(train, SEED + 2)
    svd = TruncatedSVD(n_components=SVD_K, random_state=SEED + 2).fit(d["lexical"][train])
    reduced = unit_rows(svd.transform(d["lexical"]))
    mean, matrix, _ = fit_transform("within_author_whitening", reduced[train], li[train], w[train], np.random.default_rng(SEED + 2))
    check(float(np.abs(got - unit_rows((reduced - mean) @ matrix.T)).max()) <= 1e-12 and "svd_variance_kept" in info and "shrinkage" in info,
          "word_svd_fit equals the published word-SVD path (word_space_probe_v2), transcribed")
    chunk_fit = fw.chunk_whitening_fit(d["chunk_vec"], d["chunk_song"], li[d["chunk_song"]], d["chunk_weight"], N)
    got, info = chunk_fit(train, SEED + 2)
    chunk_fold = fold[d["chunk_song"]]
    ctrain = chunk_fold != 2
    mean, matrix, _ = fit_transform("within_author_whitening", d["chunk_vec"][ctrain], li[d["chunk_song"]][ctrain],
                                    d["chunk_weight"][ctrain], np.random.default_rng(SEED + 2))
    projected = unit_rows((d["chunk_vec"] - mean) @ matrix.T)
    song_vec = np.zeros((N, DIM))
    np.add.at(song_vec, d["chunk_song"], projected)
    song_vec = unit_rows(song_vec / np.bincount(d["chunk_song"])[:, None])
    check(float(np.abs(got - song_vec).max()) <= 1e-12, "chunk_whitening_fit equals the published chunk path (word_space_probe_v2), transcribed")

    # 6. blindness of the chosen weight to the test fold
    A = fw.raw_space("A", a_scores)
    B = fw.raw_space("B", b_scores)
    mask = fw.pair_mask(A, B)
    base = fw.nested_fusion(A, B, li, gi, w, fold, mask)
    for k in range(FOLDS):
        test = fold == k
        A2 = fw.raw_space("A", np.where(test[:, None], a_scores + rng.standard_normal((N, L)), a_scores))
        B2 = fw.raw_space("B", np.where(test[:, None], b_scores + rng.standard_normal((N, L)), b_scores))
        other = fw.nested_fusion(A2, B2, li, gi, w, fold, mask)
        if other["chosen_w_by_fold"][k] != base["chosen_w_by_fold"][k]:
            raise SystemExit(f"FAILED: perturbing fold {k}'s test rows changed its chosen weight")
        if other["folds"][k]["test_curve_weighted_mrr"] == base["folds"][k]["test_curve_weighted_mrr"]:
            raise SystemExit(f"FAILED: the perturbation of fold {k} did not change its test curve, so the check is empty")
    print("ok: raw pair: perturbing a test fold's rows leaves its chosen weight unchanged while its test curve changes")
    # a fitted component: inner matrices independent of the outer
    inner_c = {}
    c_scores = random_scores(rng, li, 0.8)
    for k in range(FOLDS):
        m = np.full((N, L), np.nan)
        m[fold != k] = random_scores(rng, li, 0.8)[fold != k]
        inner_c[k] = m
    C = fw.Space("C", c_scores, inner_c, fw.defined_rows(c_scores), {"fitted": True})
    maskc = fw.pair_mask(A, C)
    base_c = fw.nested_fusion(A, C, li, gi, w, fold, maskc)
    C2 = fw.Space("C", c_scores + rng.standard_normal((N, L)), inner_c, C.defined, {"fitted": True})
    other = fw.nested_fusion(A, C2, li, gi, w, fold, maskc)
    check(other["chosen_w_by_fold"] == base_c["chosen_w_by_fold"] and other["all_queries_curve_weighted_mrr"] != base_c["all_queries_curve_weighted_mrr"],
          "fitted pair: perturbing every outer row leaves every chosen weight unchanged (only the inner scores choose)")
    for k in range(FOLDS):
        inner_k = {j: (m if j != k else np.where(np.isnan(m), m, m + rng.standard_normal((N, L)))) for j, m in inner_c.items()}
        other = fw.nested_fusion(A, fw.Space("C", c_scores, inner_k, C.defined, {"fitted": True}), li, gi, w, fold, maskc)
        if any(other["chosen_w_by_fold"][j] != base_c["chosen_w_by_fold"][j] for j in range(FOLDS) if j != k):
            raise SystemExit(f"FAILED: perturbing inner[{k}] changed another fold's chosen weight")
    print("ok: fitted pair: perturbing inner[k] leaves the other folds' chosen weights unchanged")
    # the fitted test rows are fuse(zA, zB, chosen w), by hand
    zA, zC = v1.zscore_rows(a_scores), v1.zscore_rows(c_scores)
    by_hand = np.zeros((N, L))
    for k in range(FOLDS):
        test = fold == k
        by_hand[test] = base_c["chosen_w_by_fold"][k] * zA[test] + (1 - base_c["chosen_w_by_fold"][k]) * zC[test]
    check(np.array_equal(base_c["ranks"]["fusion_fitted"], fw.ranks_of(by_hand, li)), "fusion_fitted ranks equal fuse(zA, zB, chosen w) recomputed by hand")
    check("fusion_w_from_outer_training" in base_c["ranks"] and "fusion_w_from_outer_training" not in base["ranks"],
          "the outer-training description system appears only when a component is fitted")
    # for a raw pair, inner scores are the outer, so the chosen weights equal a selection on the training folds' outer scores
    for k in range(FOLDS):
        train = fold != k
        c = fw.grid_curve(v1.zscore_rows(a_scores[train]), v1.zscore_rows(b_scores[train]), li[train], w[train])
        if fw.choose_weight(c) != base["chosen_w_by_fold"][k]:
            raise SystemExit(f"FAILED: raw pair fold {k}: chosen weight is not the training-fold grid optimum")
    print("ok: raw pair: the chosen weight is the training folds' grid optimum, recomputed by hand")

    # 7. the mask
    a_holes = a_scores.copy()
    a_holes[[3, 7]] = 0.0                                          # zero-variance rows
    A3 = fw.raw_space("A", a_holes)
    check(not A3.defined[3] and not A3.defined[7] and A3.defined.sum() == N - 2, "zero-variance rows are undefined")
    covered = np.ones(N, dtype=bool)
    covered[[7, 20, 21]] = False
    m3 = fw.pair_mask(A3, B, covered)
    check(m3.sum() == N - 4 and not m3[3] and not m3[7] and not m3[20] and not m3[21], "pair mask = defined(A) & defined(B) & covered")
    out3 = fw.nested_fusion(A3, B, li, gi, w, fold, m3)
    check(all(np.all(r[~m3] == L + 1) and np.all(r[m3] >= 1) and np.all(r[m3] <= L) for r in out3["ranks"].values()),
          "ranks outside the mask are the sentinel L + 1 and inside the mask are valid")
    s3 = fw.system_summary(out3["ranks"]["fusion_fitted"], w, m3)
    check(s3["queries"] == N - 4 and s3["mrr"] == round(float(np.mean(1.0 / out3["ranks"]["fusion_fitted"][m3])), 4)
          and s3["mrr_component_weighted"] == round(brute_weighted_mrr(out3["ranks"]["fusion_fitted"][m3], w[m3]), 4),
          "system_summary reports the mask size, the plain MRR and the weighted MRR over the mask")

    # 8. the two published rhyme rank policies
    sc = random_scores(rng, li, 0.3)
    mat = sparse.csr_matrix(np.where(np.arange(N)[:, None] == 5, 0.0, 1.0))   # row 5 has no feature
    cov = np.ones(N, dtype=bool)
    cov[[2, 5]] = False
    three, ident, empty_covered = fw.rhyme_ranks_published(sc, mat, li, cov, L)
    base_r = fw.ranks_of(sc, li)
    check(three[5] == L and three[2] == L and ident[2] == L and ident[5] == L and np.array_equal(three[cov], base_r[cov])
          and np.array_equal(ident[cov], base_r[cov]) and empty_covered == 0, "rhyme policies: empty and uncovered rows at the worst rank; covered rows untouched")
    cov[5] = True
    three, ident, empty_covered = fw.rhyme_ranks_published(sc, mat, li, cov, L)
    check(three[5] == L and ident[5] == base_r[5] and empty_covered == 1,
          "a covered row with no feature: worst rank under three_spaces_v3's policy, the tie-broken rank under identity_spaces_v2's, and counted")

    # 9. the reading rule on constructed intervals
    def c(system, minus, lo, hi):
        return {"system": system, "minus": minus, "mrr_difference": round((lo + hi) / 2, 4), "ci95": [lo, hi], "excludes_zero": lo > 0 or hi < 0}
    r = fw.reading_of("p", [c("fusion_fitted", "fusion_equal", -0.02, -0.01), c("fusion_fitted", "A", 0.01, 0.02)], "A", "A")
    check(r["published_equal_weight_verdict"] == "exceeds_honest_weighting" and r["fusion_adds_to_better_component"] and "rhyme" not in r,
          "reading: fitted below equal, clear of zero -> published exceeds honest weighting; fusion adds")
    r = fw.reading_of("p", [c("fusion_fitted", "fusion_equal", 0.01, 0.02), c("fusion_fitted", "A", -0.01, 0.01)], "A", "A")
    check(r["published_equal_weight_verdict"] == "conservative" and not r["fusion_adds_to_better_component"],
          "reading: fitted above equal, clear of zero -> published conservative; interval covering zero -> does not add")
    r = fw.reading_of("p", [c("fusion_fitted", "fusion_equal", -0.01, 0.01), c("fusion_fitted", "A", -0.02, -0.01)], "A", "A")
    check(r["published_equal_weight_verdict"] == "stands" and not r["fusion_adds_to_better_component"],
          "reading: interval covering zero -> published stands; fusion below its component -> does not add")
    r = fw.reading_of(fw.RHYME_PAIR, [c("fusion_fitted", "fusion_equal", -0.01, 0.01), c("fusion_fitted", "words", -0.003, 0.002)], "words", "words")
    check(r["rhyme_adds_beyond_the_surface"] is False and "adds nothing" in r["rhyme"], "reading: rhyme pair, words+rhyme minus words not clear of zero -> adds nothing beyond the surface")
    r = fw.reading_of(fw.RHYME_PAIR, [c("fusion_fitted", "fusion_equal", -0.01, 0.01), c("fusion_fitted", "words", 0.004, 0.009)], "words", "words")
    check(r["rhyme_adds_beyond_the_surface"] is True, "reading: rhyme pair, positive and clear of zero -> rhyme adds beyond the surface")

    # 10. end to end on two raw and two fitted spaces
    fitted_w = fw.Space("W", outer, inner, fw.defined_rows(outer), {"fitted": True, "outer": fits, "inner": ifits})
    spaces = {"A": A, "B": B, "W": fitted_w, "C": C}
    pairs = (("w_with_c", "W", "C"), ("a_with_b", "A", "B"), (fw.RHYME_PAIR, "A", "C"))
    masks = {"w_with_c": fw.pair_mask(fitted_w, C), "a_with_b": fw.pair_mask(A, B), fw.RHYME_PAIR: fw.pair_mask(A, C, covered)}
    results = fw.analyse(dict(label_index=li, group_ids=gi, weights=w, fold=fold), spaces, masks, pairs, log=logs.append)
    json.dumps(results, sort_keys=True)
    print("ok: analyse runs end to end on three pairs and its result is JSON-serialisable")
    for name, res in results.items():
        m = masks[name]
        check(res["queries"] == int(m.sum()), f"{name}: reported queries equal the mask size")
        a_name, b_name = res["components"]["a"], res["components"]["b"]
        ranks_eq = fw.ranks_of(fw.published_equal_fusion(spaces[a_name].outer[m], spaces[b_name].outer[m]), li[m])
        check(abs(res["systems"]["fusion_equal"]["mrr"] - round(float(np.mean(1.0 / ranks_eq)), 4)) <= 1e-12,
              f"{name}: fusion_equal MRR equals the published expression's ranks on the mask")
        for f in res["folds"]:
            if f["test_weighted_mrr_at_w_chosen_on_test_fold"] + 1e-12 < max(f["test_weighted_mrr_at_chosen_w"], f["test_weighted_mrr_at_equal_w"]):
                raise SystemExit(f"FAILED: {name}: the test-fold optimum is below the fitted or equal weight on fold {f['fold']}")
            if f["test_curve_weighted_mrr"][fw.GRID.index(f["w_chosen_on_test_fold"])] != f["test_weighted_mrr_at_w_chosen_on_test_fold"]:
                raise SystemExit(f"FAILED: {name}: w_chosen_on_test_fold does not index its own curve value")
        check(("fusion_w_from_outer_training" in res["systems"]) == (spaces[a_name].fitted or spaces[b_name].fitted),
              f"{name}: the outer-training description system is present exactly when a component is fitted")
        for ctr in res["paired_contrasts"]:
            check(ctr["description_only"] == (ctr["system"] in fw.DESCRIPTION_SYSTEMS), f"{name}: description flag on {ctr['system']} - {ctr['minus']}")
        better = res["better_component"]
        sys_w = {s: res["systems"][s]["mrr_component_weighted"] for s in (a_name, b_name)}
        check(sys_w[better] == max(sys_w.values()), f"{name}: the better component has the higher weighted MRR")
        expected_pairs = {("fusion_fitted", "fusion_equal"), ("fusion_fitted", better), ("fusion_equal", better),
                          ("fusion_w_chosen_on_test_fold", "fusion_fitted"), ("fusion_w_chosen_on_all", "fusion_fitted")}
        got_pairs = {(ctr["system"], ctr["minus"]) for ctr in res["paired_contrasts"]}
        check(expected_pairs <= got_pairs, f"{name}: every required contrast is present")
        check(("rhyme" in res["reading"]) == (name == fw.RHYME_PAIR), f"{name}: the rhyme reading appears only for the rhyme pair")
    # contrast points equal the weighted-mean differences, recomputed from the returned ranks
    res = results["a_with_b"]
    m = masks["a_with_b"]
    out = fw.nested_fusion(A, B, li, gi, w, fold, m)
    for ctr in res["paired_contrasts"]:
        left = np.sum(w[m] / out["ranks"][ctr["system"]][m]) / np.sum(w[m])
        right = np.sum(w[m] / out["ranks"][ctr["minus"]][m]) / np.sum(w[m])
        if abs(ctr["mrr_difference"] - round(float(left - right), 4)) > 1e-12:
            raise SystemExit(f"FAILED: contrast point {ctr['system']} - {ctr['minus']} is not the weighted-mean difference")
        got = paired_group_bootstrap({"x": np.where(m, 1.0 / out["ranks"][ctr["system"]], 0.0), "y": np.where(m, 1.0 / out["ranks"][ctr["minus"]], 0.0)},
                                     w, gi, m, [("x", "y")])[0]
        if got["ci95"] != ctr["ci95"]:
            raise SystemExit(f"FAILED: contrast interval {ctr['system']} - {ctr['minus']} is not paired_group_bootstrap's")
    print("ok: contrast points are the weighted-mean differences and intervals are paired_group_bootstrap's, recomputed from the ranks")
    check(all("chosen w by fold" in line for line in logs if line.startswith("==")), "the pair log line reports the chosen weights")

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
