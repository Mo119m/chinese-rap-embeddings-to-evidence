#!/usr/bin/env python3
"""Synthetic unit test for estimand_two_stage_v3: random data, no corpus, no private files.

Checks, each against an independent definition:
  1. component_index / component_means / per_label_components give the same per-label
     component tensors, in the same row order, as v1.component_metric_values, and every
     component mean equals the brute-force mean over its songs;
  2. the two_stage wrapper (the builder's v1.component_metric_values and v1.run_bootstrap
     through the constant redirect) returns the builder's tensors, restores the builder's
     constants, and its replicates equal, replicate for replicate, a literal transcription
     of v1.run_bootstrap written here with the same generator calls under the same seed;
     its point is the mean over labels of the mean over that label's components; a
     different seed gives different replicates;
  3. label_macro and estimand_table equal brute-force loops (per query, per component, per
     label) for every column, and the two-stage point equals the label-macro estimand;
  4. interval / two_stage_contrast equal brute-force quantiles of the replicate differences,
     with the excludes_zero and direction fields right on constructed draws;
  5. fold_wise_centroid_scores equals a literal transcription of identity_probe_v2.build's
     fold loop for the three transforms, including the permuted-label null;
  6. paired_group_bootstrap's point on 0/1 hit arrays is the component-weighted recall (so
     the R@1 group intervals in the script are what they claim), and ranks_of breaks a tie
     against the truth by label order, the builder's rule.

    python test_estimand_two_stage.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import estimand_two_stage_v3 as ets  # noqa: E402
import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402

N, L, DIM = 240, 8, 12
SYSTEMS = ("s0", "s1", "s2")
METRICS = ("mrr", "recall_at_1", "recall_at_10")
REPLICATES = 60


def synthetic(seed: int):
    rng = np.random.default_rng(seed)
    # leakage groups: mostly pairs, one group of six; labels random, so some groups span labels
    group_ids = np.repeat(np.arange(N // 2), 2)
    group_ids[-6:] = group_ids[-6]
    label_index = rng.integers(0, L, size=N)
    for l in range(L):                                 # every label needs several groups
        if len(set(group_ids[label_index == l].tolist())) < 4:
            raise RuntimeError("regenerate: a label with too few groups")
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    n_groups = int(group_ids.max()) + 1
    fold = rng.integers(0, FOLDS, size=n_groups)[group_ids]
    dense = rng.standard_normal((N, DIM))
    dense += 0.8 * rng.standard_normal((L, DIM))[label_index]       # some label structure
    dense = dense / np.linalg.norm(dense, axis=1, keepdims=True)
    ranks = {s: rng.integers(1, L + 1, size=N) for s in SYSTEMS}
    return group_ids, label_index, weights, fold, dense, ranks


def metric_arrays(ranks):
    return {s: {"mrr": 1.0 / r.astype(np.float64), "recall_at_1": (r <= 1).astype(np.float64),
                "recall_at_10": (r <= 10).astype(np.float64)} for s, r in ranks.items()}


def columns_of(arrays):
    return np.column_stack([arrays[s][m] for s in SYSTEMS for m in METRICS])


def reference_two_stage(per_label: list[np.ndarray], replicates: int, seed: int):
    """A literal transcription of v1.run_bootstrap on per-label arrays (components, K):
    the same generator, the same calls in the same order."""
    rng = np.random.default_rng(seed)
    label_count = len(per_label)
    width = per_label[0].shape[1]
    out = np.zeros((replicates, width))
    for replicate in range(replicates):
        sampled = rng.integers(0, label_count, size=label_count)
        occurrences = Counter(int(label) for label in sampled.tolist())
        total = np.zeros(width)
        for label, count in occurrences.items():
            values = per_label[label]
            draws = rng.integers(0, len(values), size=(count, len(values)))
            total += values[draws].mean(axis=1).sum(axis=0)
        out[replicate] = total / label_count
    point = np.mean([values.mean(axis=0) for values in per_label], axis=0)
    return point, out


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {message}")
    print(f"ok: {message}")


def main() -> int:
    group_ids, label_index, weights, fold, dense, ranks = synthetic(7)
    arrays = metric_arrays(ranks)
    values = columns_of(arrays)
    K = values.shape[1]

    # 1. components against v1.component_metric_values and brute force
    comp_of, comp_label = ets.component_index(label_index, group_ids)
    comp_values = ets.component_means(values, comp_of, len(comp_label))
    per_label = ets.per_label_components(comp_values, comp_label, L)
    saved = (v1.SYSTEMS, v1.METRICS)
    v1.SYSTEMS, v1.METRICS = SYSTEMS, METRICS
    try:
        v1_components, v1_counts = v1.component_metric_values(
            type("E", (), {"metric_arrays": arrays})(), label_index, group_ids)
    finally:
        v1.SYSTEMS, v1.METRICS = saved
    check(len(v1_components) == L and all(len(a) == len(b) for a, b in zip(v1_components, per_label)),
          "one tensor per label with the builder's component counts")
    check(int(v1_counts.sum()) == len(comp_label), "component count equals the builder's label_group_counts total")
    gap = max(float(np.abs(a.reshape(len(a), -1) - b).max()) for a, b in zip(v1_components, per_label))
    check(gap <= 1e-12, f"component means equal v1.component_metric_values in the builder's row order (gap {gap:.1e})")
    for c in range(len(comp_label)):
        members = np.flatnonzero(comp_of == c)
        if np.abs(values[members].mean(axis=0) - comp_values[c]).max() > 1e-12:
            raise SystemExit("FAILED: a component mean differs from its brute-force definition")
        if len({int(label_index[m]) for m in members}) != 1 or int(label_index[members[0]]) != int(comp_label[c]):
            raise SystemExit("FAILED: a component spans labels or carries the wrong label")
    print("ok: every component mean equals the brute-force mean over its songs, and each component sits in one label")

    # 2. the wrapper around the builder's functions against a transcription of v1.run_bootstrap
    before = (v1.SYSTEMS, v1.METRICS, v1.BOOTSTRAP_REPLICATES, v1.RANDOM_SEED)
    components, counts, point, reps, diagnostics = ets.two_stage(arrays, label_index, group_ids, SYSTEMS, METRICS,
                                                                 replicates=REPLICATES, seed=v1.RANDOM_SEED)
    check((v1.SYSTEMS, v1.METRICS, v1.BOOTSTRAP_REPLICATES, v1.RANDOM_SEED) == before, "the builder's constants are restored after the call")
    check(reps.shape == (REPLICATES, K) and point.shape == (K,), "replicate matrix and point shapes")
    check(diagnostics["replicates"] == REPLICATES and diagnostics["outer_label_occurrences_per_replicate"] == L,
          "the builder's diagnostics describe this call")
    gap = max(float(np.abs(a - b).max()) for a, b in zip(components, v1_components))
    check(gap <= 1e-12 and int(counts.sum()) == int(v1_counts.sum()), "the wrapper returns the builder's own component tensors")
    ref_point, ref_reps = reference_two_stage(per_label, REPLICATES, v1.RANDOM_SEED)
    gap = float(np.abs(ref_reps - reps).max())
    check(gap <= 1e-12, f"wrapper replicates equal the transcription of v1.run_bootstrap under the same seed (gap {gap:.1e})")
    gap = float(np.abs(ref_point - point).max())
    check(gap <= 1e-12, f"wrapper point equals the mean over labels of the mean over components (gap {gap:.1e})")
    # column order: system-major, metric-minor, checked by brute force per (system, metric)
    macro_cols = []
    for s, system in enumerate(SYSTEMS):
        for m, metric in enumerate(METRICS):
            per = [np.mean([arrays[system][metric][comp_of == c].mean() for c in np.flatnonzero(comp_label == l)]) for l in range(L)]
            macro_cols.append(float(np.mean(per)))
    check(float(np.abs(np.asarray(macro_cols) - point).max()) <= 1e-12, "columns are system-major, metric-minor")
    _, _, _, reps_other, _ = ets.two_stage(arrays, label_index, group_ids, SYSTEMS, METRICS,
                                           replicates=REPLICATES, seed=v1.RANDOM_SEED + 1)
    check(float(np.abs(reps_other - reps).max()) > 1e-6, "a different seed gives different replicates")

    # 3. the three estimands against brute force
    table = ets.estimand_table(values, weights, label_index, L)
    for k in range(K):
        plain = float(np.mean(values[:, k]))
        component = float(np.sum(values[:, k] * weights) / np.sum(weights))
        per = []
        for l in range(L):
            rows = np.flatnonzero(label_index == l)
            per.append(float(np.sum(values[rows, k] * weights[rows]) / np.sum(weights[rows])))
        macro = float(np.mean(per))
        macro_b = float(np.mean([per_label[l][:, k].mean() for l in range(L)]))
        if abs(table["plain_all"][k] - plain) > 1e-12 or abs(table["component_weighted"][k] - component) > 1e-12:
            raise SystemExit(f"FAILED: plain or component-weighted estimand differs at column {k}")
        if abs(table["label_macro"][k] - macro) > 1e-12 or abs(macro - macro_b) > 1e-12:
            raise SystemExit(f"FAILED: label-macro estimand differs at column {k}")
    print("ok: plain, component-weighted and label-macro estimands equal brute force for every column")
    check(float(np.abs(np.asarray(table["label_macro"]) - point).max()) <= 1e-12,
          "the two-stage point equals the label-macro estimand from the ranks")
    # the three estimands differ on this data, so the table is not degenerate
    check(any(abs(table["plain_all"][k] - table["label_macro"][k]) > 1e-6 for k in range(K)), "plain and label-macro differ somewhere")

    # 4. intervals against brute force
    a, b = 0, 3                                        # s0/mrr against s1/mrr
    got = ets.two_stage_contrast(point, reps, a, b)
    diff = reps[:, a] - reps[:, b]
    low, high = np.quantile(diff, [0.025, 0.975])
    check(got["ci95"] == [round(float(low), 4), round(float(high), 4)] and got["difference"] == round(float(point[a] - point[b]), 4),
          "two-stage contrast interval and point equal brute-force quantiles")
    check(got["excludes_zero"] == bool(low > 0 or high < 0), "excludes_zero follows the interval")
    pos = ets.interval(np.linspace(0.01, 0.05, 100), 0.03)
    neg = ets.interval(np.linspace(-0.05, -0.01, 100), -0.03)
    mixed = ets.interval(np.linspace(-0.02, 0.05, 100), 0.015)
    check(pos["excludes_zero"] and pos["direction"] == "left_higher" and neg["excludes_zero"] and neg["direction"] == "right_higher"
          and not mixed["excludes_zero"] and mixed["direction"] == "interval_includes_zero", "direction fields on constructed draws")

    # 5. the fold-wise scorer against identity_probe_v2's loop, transcribed
    for transform in ("total_whitening", "within_author_whitening", "within_author_whitening_permuted_labels"):
        got = ets.fold_wise_centroid_scores(dense, label_index, group_ids, weights, L, fold, transform)
        expect = np.zeros((N, L))
        for k in range(FOLDS):
            train_mask = fold != k
            queries = np.flatnonzero(~train_mask)
            mean, matrix, _ = fit_transform(transform, dense[train_mask], label_index[train_mask],
                                            weights[train_mask], np.random.default_rng(SEED + k))
            projected = (dense - mean) @ matrix.T
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            projected = projected / np.maximum(norms, 1e-12)
            expect[queries] = dense_leave_group_out(projected, label_index, group_ids, weights, L, queries)
        gap = float(np.abs(got - expect).max())
        check(gap <= 1e-12, f"fold_wise_centroid_scores equals identity_probe_v2's fold loop for {transform} (gap {gap:.1e})")
    base = ets.fold_wise_centroid_scores(dense, label_index, group_ids, weights, L, fold, "within_author_whitening")
    relabelled = ets.fold_wise_centroid_scores(dense, label_index, group_ids, weights, L, (fold + 1) % FOLDS, "within_author_whitening")
    check(float(np.abs(relabelled - base).max()) <= 1e-12, "relabelling the folds leaves the fold-wise scores unchanged")
    other_fold = np.random.default_rng(SEED + 99).integers(0, FOLDS, size=int(group_ids.max()) + 1)[group_ids]
    check(bool(np.any(other_fold != fold)), "the second partition differs from the first")
    other = ets.fold_wise_centroid_scores(dense, label_index, group_ids, weights, L, other_fold, "within_author_whitening")
    check(float(np.abs(other - base).max()) > 1e-9, "a different partition changes the fold-wise scores")
    permuted = ets.fold_wise_centroid_scores(dense, label_index, group_ids, weights, L, fold, "within_author_whitening_permuted_labels")
    check(float(np.abs(permuted - base).max()) > 1e-9, "the permuted-label null differs from the within-author fit")

    # 6. paired_group_bootstrap on hit arrays, and the tie rule
    hits = {s: arrays[s]["recall_at_1"] for s in SYSTEMS}
    out = paired_group_bootstrap(hits, weights, group_ids, np.ones(N, dtype=bool), [("s0", "s1")])[0]
    expect = float(np.sum(hits["s0"] * weights) / np.sum(weights) - np.sum(hits["s1"] * weights) / np.sum(weights))
    check(abs(out["mrr_difference"] - round(expect, 4)) <= 1e-12, "group-bootstrap point on hit arrays is the component-weighted R@1 difference")
    scores = np.zeros((2, 4))
    scores[0] = [0.5, 0.5, 0.1, 0.0]                   # truth label 1 ties with label 0 -> rank 2
    scores[1] = [0.5, 0.5, 0.1, 0.0]                   # truth label 0 ties with label 1 -> rank 1
    r = ets.ranks_of(scores, np.asarray([1, 0]))
    check(r.tolist() == [2, 1], "ranks_of breaks a tie by label order, the builder's rule")

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
