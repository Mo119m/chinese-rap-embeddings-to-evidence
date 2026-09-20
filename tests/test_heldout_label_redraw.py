#!/usr/bin/env python3
"""Synthetic test of heldout_label_redraw_v3: random data, no private corpus.

Checks, against brute-force definitions written independently here:
  * the transfer scoring path (fit on seen labels, score held-out queries) equals a
    brute-force leave-group-out cosine in the transformed space, for all three transforms;
  * a draw's transform is blind to the held-out labels' songs (replacing them with noise
    leaves mean and matrix bitwise unchanged), and the training mask has no held-out song;
  * the fold-wise seen-label space equals brute force per fold and is blind to the fold's songs;
  * reciprocal ranks and the MRR summaries equal a brute-force rank with the protocol's tie rule;
  * score_draw's unrounded MRRs and its bootstrap point estimates equal brute force;
  * the paired group bootstrap's point estimate is the component-weighted MRR difference;
  * the draw construction: sizes, the probe's formula for draw 0, pairwise disjointness,
    determinism, and the top-up fallback when disjoint blocks are impossible;
  * the gap check passes within tolerance and raises SystemExit beyond it;
  * the reading rule and the spread helper on hand-made inputs;
  * an end-to-end run of analyse() on the synthetic corpus: after_draw0 runs exactly once,
    on draw 0, before any redraw; six draws, finite numbers, draw-0 MRRs equal to brute force.

    set CHINESE_RAP_CORPUS=v3
    python test_heldout_label_redraw.py          (or: python -m pytest test_heldout_label_redraw.py)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("CHINESE_RAP_CORPUS", "v3")
HERE = Path(__file__).resolve().parent.parent / "src"
_spec = importlib.util.spec_from_file_location("heldout_label_redraw_v3", HERE / "heldout_label_redraw_v3.py")
hlr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hlr)

from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402


# ------------------------------------------------------------------ synthetic corpus
def make_synthetic(seed: int = 7, label_count: int = 20, dim: int = 24):
    """Random songs around label centres, leakage groups of 1-3 songs (some across labels),
    the protocol's per-(group, label) weights and fold assignment."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(6, 30, size=label_count)
    label_index = np.repeat(np.arange(label_count), sizes)
    n = len(label_index)
    perm = rng.permutation(n)
    label_index = label_index[perm]
    centres = rng.normal(size=(label_count, dim))
    dense = centres[label_index] + 1.5 * rng.normal(size=(n, dim))
    # groups: singletons, then merge some pairs and triples, a few across labels
    group_ids = np.arange(n)
    for _ in range(n // 8):
        a, b = rng.integers(0, n, size=2)
        group_ids[group_ids == group_ids[b]] = group_ids[a]
    order = {g: i for i, g in enumerate(sorted(set(group_ids.tolist())))}
    group_ids = np.asarray([order[g] for g in group_ids.tolist()], dtype=np.int64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    dense = unit_rows(dense).astype(np.float64)
    return dict(label_index=label_index, group_ids=group_ids, weights=weights, label_count=label_count,
                fold=fold, dense=dense)


def brute_force_lgo(x, label_index, group_ids, weights, label_count, queries):
    """For each query and label: cosine with the weighted sum of the label's songs outside
    the query's leakage group. Written from the definition, no vectorisation."""
    out = np.zeros((len(queries), label_count))
    for row, q in enumerate(queries.tolist()):
        for l in range(label_count):
            members = [s for s in range(len(label_index)) if label_index[s] == l and group_ids[s] != group_ids[q]]
            profile = sum(weights[s] * x[s] for s in members)
            out[row, l] = float(x[q] @ profile) / float(np.linalg.norm(profile))
    return out


def brute_force_rank(scores, truth):
    """The protocol's rank: 1 + labels scoring above the true one by more than 1e-12, plus
    ties (within 1e-12) with a smaller label index."""
    ranks = []
    for row in range(scores.shape[0]):
        t = scores[row, truth[row]]
        above = sum(1 for l in range(scores.shape[1]) if scores[row, l] > t + 1e-12)
        ties = sum(1 for l in range(truth[row]) if abs(scores[row, l] - t) <= 1e-12)
        ranks.append(1 + above + ties)
    return np.asarray(ranks)


def brute_force_transfer_rr(d, held):
    """Reciprocal ranks of the held-out queries under each transform, from the definitions."""
    li, gi, w, L, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["dense"]
    unseen = held[li]
    queries = np.flatnonzero(unseen)
    out = {}
    for name in hlr.TRANSFORMS:
        mean, matrix, _ = fit_transform(name, dense[~unseen], li[~unseen], w[~unseen], np.random.default_rng(SEED))
        x = unit_rows((dense - mean) @ matrix.T)
        out[name] = 1.0 / brute_force_rank(brute_force_lgo(x, li, gi, w, L, queries), li[queries])
    return queries, out


# ------------------------------------------------------------------ tests
def test_draws_disjoint_and_probe_formula():
    L = 20
    draws, seeds, disjoint = hlr.held_out_draws(L)
    size = int(round(hlr.HELD_OUT_LABEL_SHARE * L))
    assert len(draws) == 6 and seeds == [SEED, *hlr.REDRAW_SEEDS]
    assert disjoint
    for held in draws:
        assert held.dtype == bool and int(held.sum()) == size
    for i in range(6):
        for j in range(i + 1, 6):
            assert not np.any(draws[i] & draws[j]), "draws overlap"
    # draw 0 is the probe's formula
    expect = np.zeros(L, dtype=bool)
    expect[np.random.default_rng(SEED).permutation(L)[:size]] = True
    assert np.array_equal(draws[0], expect)
    # deterministic
    again, _, _ = hlr.held_out_draws(L)
    assert all(np.array_equal(a, b) for a, b in zip(draws, again))
    # 226 labels: six blocks of 34, 22 never held out
    draws226, _, disjoint226 = hlr.held_out_draws(226)
    assert disjoint226 and all(int(h.sum()) == 34 for h in draws226)
    assert int(np.any(np.stack(draws226), axis=0).sum()) == 204
    # the seeds are the ones the docstring states
    assert hlr.REDRAW_SEEDS == (20260918, 20260919, 20260920, 20260921, 20260922)


def test_draws_fallback_when_blocks_impossible():
    L, share = 10, 0.3          # size 3, six draws need 18 > 10 labels
    draws, _, disjoint = hlr.held_out_draws(L, share=share)
    assert not disjoint
    assert all(int(h.sum()) == 3 for h in draws)
    for i in range(1, len(draws)):     # consecutive draws never share a label
        assert not np.any(draws[i] & draws[i - 1])


def test_transfer_scores_match_brute_force_and_are_blind_to_held_out_songs():
    d = make_synthetic()
    li, gi, w, L, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["dense"]
    held = hlr.probe_draw(L)
    queries, scores, info = hlr.transfer_scores(dense, li, gi, w, L, held)
    assert np.array_equal(queries, np.flatnonzero(held[li]))
    unseen = held[li]
    rng = np.random.default_rng(1)
    for name in hlr.TRANSFORMS:
        mean, matrix, _ = fit_transform(name, dense[~unseen], li[~unseen], w[~unseen], np.random.default_rng(SEED))
        x = unit_rows((dense - mean) @ matrix.T)
        expect = brute_force_lgo(x, li, gi, w, L, queries)
        gap = float(np.abs(scores[name] - expect).max())
        assert gap < 1e-9, f"{name}: transfer scores differ from brute force by {gap}"
        # blind to the held-out labels' songs: replace them and refit
        noisy = dense.copy()
        noisy[unseen] = unit_rows(rng.normal(size=(int(unseen.sum()), dense.shape[1])))
        mean2, matrix2, _ = fit_transform(name, noisy[~unseen], li[~unseen], w[~unseen], np.random.default_rng(SEED))
        assert np.array_equal(mean, mean2) and np.array_equal(matrix, matrix2), f"{name}: the fit saw held-out songs"
        assert info[name]["labels_seen"] == L - int(held.sum())
    # the untransformed transfer equals the raw protocol scorer
    raw = dense_leave_group_out(dense, li, gi, w, L, queries)
    assert float(np.abs(scores["none"] - raw).max()) < 1e-9
    # a training mask with a held-out song is refused
    try:
        hlr.transfer_scores(dense, li, gi, w, L, np.zeros(L, dtype=bool))
        raise AssertionError("a draw holding out no label was accepted")
    except RuntimeError:
        pass


def test_foldwise_seen_space_matches_brute_force_and_is_blind_to_the_fold():
    d = make_synthetic(seed=11)
    li, gi, w, L, fold, dense = (d["label_index"], d["group_ids"], d["weights"], d["label_count"],
                                 d["fold"], d["dense"])
    scores = hlr.foldwise_within_author(dense, li, gi, w, L, fold)
    rng = np.random.default_rng(2)
    for k in range(FOLDS):
        train = fold != k
        queries = np.flatnonzero(~train)
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train], li[train], w[train],
                                        np.random.default_rng(SEED + k))
        x = unit_rows((dense - mean) @ matrix.T)
        expect = brute_force_lgo(x, li, gi, w, L, queries)
        assert float(np.abs(scores[queries] - expect).max()) < 1e-9
        noisy = dense.copy()
        noisy[~train] = unit_rows(rng.normal(size=(int((~train).sum()), dense.shape[1])))
        mean2, matrix2, _ = fit_transform("within_author_whitening", noisy[train], li[train], w[train],
                                          np.random.default_rng(SEED + k))
        assert np.array_equal(mean, mean2) and np.array_equal(matrix, matrix2)


def test_reciprocal_ranks_and_summary_match_brute_force():
    rng = np.random.default_rng(3)
    n, L = 60, 9
    scores = rng.normal(size=(n, L))
    truth = rng.integers(0, L, size=n)
    scores[:10, :] = 0.0                                  # full ties: rank by label index
    scores[10:20, 2] = scores[np.arange(10, 20), truth[10:20]]   # a tie with label 2
    rr = hlr.reciprocal_ranks(scores, truth)
    assert np.allclose(rr, 1.0 / brute_force_rank(scores, truth))
    li = truth
    gi = np.arange(n) // 3
    component = Counter(zip(gi.tolist(), li.tolist()))
    w = np.asarray([1.0 / component[(int(g), int(l))] for g, l in zip(gi, li)])
    mask = np.zeros(n, dtype=bool)
    mask[rng.permutation(n)[:31]] = True
    s = hlr.system_summary(rr, w, li, mask)
    assert abs(s["mrr"] - float(np.mean(rr[mask]))) < 1e-4
    assert abs(s["mrr_component_weighted"] - float(np.sum(rr[mask] * w[mask]) / np.sum(w[mask]))) < 1e-4
    assert abs(s["recall_at_10"] - 1.0) < 1e-9                # 9 labels: every rank <= 10
    macro = []
    for l in np.unique(li[mask]):
        m = mask & (li == l)
        macro.append(float(np.sum(rr[m] * w[m]) / np.sum(w[m])))
    assert abs(s["mrr_label_macro"] - float(np.mean(macro))) < 1e-4


def test_score_draw_matches_brute_force():
    d = make_synthetic(seed=17)
    li, gi, w, L, fold, dense = (d["label_index"], d["group_ids"], d["weights"], d["label_count"],
                                 d["fold"], d["dense"])
    held = hlr.probe_draw(L)
    mask = held[li]
    seen_scores = hlr.foldwise_within_author(dense, li, gi, w, L, fold)
    rr_seen = hlr.reciprocal_ranks(seen_scores, li)
    entry, exact = hlr.score_draw(d, held, rr_seen)
    queries, bf = brute_force_transfer_rr(d, held)
    full = {}
    for name in hlr.TRANSFORMS:
        assert abs(exact[f"unseen_{name}"] - float(np.mean(bf[name]))) < 1e-12
        assert abs(entry["systems"][f"unseen_{name}"]["mrr"] - float(np.mean(bf[name]))) < 1e-4
        arr = np.ones(len(li))
        arr[queries] = bf[name]
        full[f"unseen_{name}"] = arr
    full["seen_within_author_whitening"] = rr_seen
    assert abs(exact["seen_within_author_whitening"] - float(np.mean(rr_seen[mask]))) < 1e-12
    # the contrasts' point estimates are the component-weighted differences over the draw
    for c in entry["contrasts"]:
        point = weighted_mean(full[c["system"]], w, mask) - weighted_mean(full[c["minus"]], w, mask)
        assert abs(c["mrr_difference"] - point) < 1e-4
        assert c["ci95"][0] - 1e-4 <= c["mrr_difference"] <= c["ci95"][1] + 1e-4
    assert entry["held_out_labels"] == int(held.sum()) and entry["queries"] == int(mask.sum())
    assert entry["groups"] == len(np.unique(gi[mask]))
    assert set(entry["fit"]) == {"total_whitening", "within_author_whitening"}


def test_bootstrap_point_estimate_is_the_weighted_difference():
    d = make_synthetic(seed=5)
    li, gi, w = d["label_index"], d["group_ids"], d["weights"]
    rng = np.random.default_rng(9)
    n = len(li)
    rr = {"a": 1.0 / rng.integers(1, 20, size=n), "b": 1.0 / rng.integers(1, 20, size=n)}
    mask = hlr.probe_draw(d["label_count"])[li]
    out = paired_group_bootstrap(rr, w, gi, mask, [("a", "b")])
    point = weighted_mean(rr["a"], w, mask) - weighted_mean(rr["b"], w, mask)
    assert abs(out[0]["mrr_difference"] - point) < 1e-4
    assert out[0]["ci95"][0] <= out[0]["mrr_difference"] <= out[0]["ci95"][1]


def test_gap_checks():
    expected = {"a": 0.2658, "b": 0.4072}
    tolerance = {"a": 5e-4, "b": 0.002}
    logged = []
    checks = hlr.gap_checks({"a": 0.26584, "b": 0.4090}, expected, tolerance, log=logged.append)
    assert checks["a"]["passed"] and checks["b"]["passed"] and len(logged) == 2
    assert abs(checks["b"]["gap"] - 0.0018) < 1e-9
    for bad in ({"a": 0.2664, "b": 0.4072}, {"a": 0.2658, "b": 0.4093}):
        try:
            hlr.gap_checks(bad, expected, tolerance, log=lambda *_: None)
            raise AssertionError(f"{bad} passed the gap check")
        except SystemExit:
            pass


def _contrast(system, minus, diff, low, high):
    return {"system": system, "minus": minus, "mrr_difference": diff, "ci95": [low, high],
            "excludes_zero": bool(low > 0 or high < 0)}


def _draw(free, specific, seen):
    return {"contrasts": [_contrast(*hlr.LABEL_FREE, *free),
                          _contrast("unseen_within_author_whitening", "unseen_none", 0.1, 0.05, 0.15),
                          _contrast(*hlr.LABEL_SPECIFIC, *specific),
                          _contrast(*hlr.SEEN_MINUS_UNSEEN, *seen)]}


def test_reading_rule():
    positive = (0.12, 0.10, 0.14)
    negative_clear = (-0.03, -0.04, -0.02)
    not_significant = (0.005, -0.003, 0.013)
    positive_clear = (0.02, 0.01, 0.03)
    seen_ns = (0.002, -0.005, 0.009)
    # every draw: label-free clear; specific negative and clear -> both readings
    r = hlr.reading_of([_draw(positive, negative_clear, seen_ns) for _ in range(6)])
    assert r["label_free_part"] == "the label-free part transfers"
    assert r["label_specific_part"] == "the label-specific part does not transfer"
    assert r["draws_within_minus_total_negative_clear_of_zero"] == 6
    assert r["draws_within_minus_total_point_estimate_at_or_below_zero"] == 6
    assert r["draws_seen_minus_unseen_clear_of_zero"] == 0
    # one draw n.s. for label-free -> count stated; specific 4 negative + 1 n.s. + 1 positive-clear -> 5 not positive
    draws = [_draw(positive, negative_clear, seen_ns) for _ in range(4)]
    draws.append(_draw(not_significant, not_significant, seen_ns))
    draws.append(_draw(positive, positive_clear, (0.02, 0.01, 0.03)))
    r = hlr.reading_of(draws)
    assert r["label_free_part"].endswith("in 5 of 6 draws")
    assert r["label_specific_part"] == "the label-specific part does not transfer"
    assert r["draws_within_minus_total_not_positive"] == 5
    assert r["draws_within_minus_total_negative_clear_of_zero"] == 4
    assert r["draws_within_minus_total_point_estimate_at_or_below_zero"] == 4
    assert r["draws_seen_minus_unseen_clear_of_zero"] == 1
    # two draws positive-clear for the label-specific part -> count stated
    draws = [_draw(positive, negative_clear, seen_ns) for _ in range(4)] + [_draw(positive, positive_clear, seen_ns) for _ in range(2)]
    r = hlr.reading_of(draws)
    assert r["label_specific_part"].endswith("is not positive in 4 of 6 draws")
    # a positive point estimate with the interval touching zero counts as not positive
    r = hlr.reading_of([_draw(positive, (0.01, 0.0, 0.02), seen_ns) for _ in range(6)])
    assert r["draws_within_minus_total_not_positive"] == 6
    assert r["draws_within_minus_total_point_estimate_at_or_below_zero"] == 0
    # a negative point estimate with an interval clear of zero on the negative side is not positive
    assert not hlr.positive_clear(_contrast("x", "y", -0.02, -0.03, -0.01))
    assert hlr.positive_clear(_contrast("x", "y", 0.02, 0.01, 0.03))
    assert not hlr.positive_clear(_contrast("x", "y", 0.02, -0.01, 0.05))


def test_spread():
    s = hlr.spread([0.1, 0.2, 0.4])
    assert s["draws"] == 3 and abs(s["min"] - 0.1) < 1e-9 and abs(s["max"] - 0.4) < 1e-9
    assert abs(s["mean"] - 0.2333) < 1e-4 and abs(s["sd"] - round(float(np.std([0.1, 0.2, 0.4], ddof=1)), 4)) < 1e-9
    assert hlr.spread([0.3])["sd"] == 0.0


def test_end_to_end_on_synthetic_corpus():
    d = make_synthetic(seed=13, label_count=24)
    li, gi, w, L, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["dense"]
    calls = []
    logged = []

    def after_draw0(entry, exact):
        # runs once, on draw 0, before any redraw has been logged
        calls.append((entry["draw"], dict(exact)))
        assert sum(1 for line in logged if line.startswith("draw ")) == 1

    out = hlr.analyse(d, after_draw0=after_draw0, log=logged.append)
    assert len(calls) == 1 and calls[0][0] == 0
    assert sum(1 for line in logged if line.startswith("draw ")) == 6
    assert len(out["draws"]) == 6 and out["held_out_sets_pairwise_disjoint"]
    assert out["seeds"] == [SEED, *hlr.REDRAW_SEEDS]
    assert out["labels_never_held_out"] == L - 6 * int(round(hlr.HELD_OUT_LABEL_SHARE * L))
    for entry in out["draws"]:
        assert set(entry["systems"]) == set(hlr.SYSTEMS)
        assert [(c["system"], c["minus"]) for c in entry["contrasts"]] == list(hlr.PAIRS)
        for s in entry["systems"].values():
            assert all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in s.values())
        assert entry["held_out_labels"] == int(round(hlr.HELD_OUT_LABEL_SHARE * L))
        assert entry["queries"] > 0 and entry["groups"] <= entry["queries"]
        assert "shrinkage" in entry["fit"]["within_author_whitening"] and "none" not in entry["fit"]
    assert set(out["reading"]) >= {"label_free_part", "label_specific_part", "draws"}
    assert set(out["spread_across_draws"]["contrasts"]) == {f"{a} - {b}" for a, b in hlr.PAIRS}
    # draw 0's MRRs against brute force from scratch, and the unrounded values handed to the check
    held = hlr.probe_draw(L)
    queries, bf = brute_force_transfer_rr(d, held)
    for name in hlr.TRANSFORMS:
        mrr = float(np.mean(bf[name]))
        assert abs(out["draws"][0]["systems"][f"unseen_{name}"]["mrr"] - mrr) < 1e-4
        assert abs(calls[0][1][f"unseen_{name}"] - mrr) < 1e-12
    # no per-song array leaves analyse()
    assert not any(isinstance(v, np.ndarray) for v in out.values())


TESTS = [test_draws_disjoint_and_probe_formula, test_draws_fallback_when_blocks_impossible,
         test_transfer_scores_match_brute_force_and_are_blind_to_held_out_songs,
         test_foldwise_seen_space_matches_brute_force_and_is_blind_to_the_fold,
         test_reciprocal_ranks_and_summary_match_brute_force,
         test_score_draw_matches_brute_force,
         test_bootstrap_point_estimate_is_the_weighted_difference,
         test_gap_checks, test_reading_rule, test_spread, test_end_to_end_on_synthetic_corpus]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(TESTS) - failed} of {len(TESTS)} tests passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
