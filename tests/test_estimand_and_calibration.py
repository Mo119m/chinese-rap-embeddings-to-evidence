#!/usr/bin/env python3
"""Synthetic checks for src/estimand_and_calibration_v3.py, without the corpus:

    python tests/test_estimand_and_calibration.py

Every estimator the script adds is checked against an independent definition, on data small
enough to run in seconds: the weighted AUC against a double loop over all target/non-target
pairs, C_llr against a per-trial loop that forms the log-likelihood ratio explicitly, min-C_llr
against an exhaustive search over monotone consecutive partitions (on a bounded integer score
alphabet, so the search can never blow up) and against sklearn's isotonic regression on larger
cases, the block pooling against the unpooled trial list with deliberate ties, the estimand
weights against the repo's own estimand functions, the rebuilt group bootstrap against
identity_spaces_v2.paired_group_bootstrap, and the two-stage columns -- both the label-macro one
and the size-weighted ratio that carries the component-weighted estimand -- against a literal
transcription of v1.run_bootstrap's draws written here. The known limiting cases are included: a
system that always outputs 0, a perfectly separating one, its reverse, and one whose scores are
the true log-likelihood ratios of a two-Gaussian model.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import estimand_and_calibration_v3 as ec  # noqa: E402
import estimand_two_stage_v3 as ets  # noqa: E402
import identity_spaces_v2 as isv2  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402

LOG2 = math.log(2.0)


# ------------------------------------------------------------------ independent definitions
def softplus(z: float) -> float:
    if z > 30.0:
        return z
    if z < -30.0:
        return math.exp(z)
    return math.log1p(math.exp(z))


def brute_auc(score, target, weight):
    """Every target/non-target pair, one at a time."""
    tar = [i for i in range(len(score)) if target[i]]
    non = [i for i in range(len(score)) if not target[i]]
    num = 0.0
    for i in tar:
        for j in non:
            if score[i] > score[j]:
                num += weight[i] * weight[j]
            elif score[i] == score[j]:
                num += 0.5 * weight[i] * weight[j]
    return num / (sum(weight[i] for i in tar) * sum(weight[j] for j in non))


def brute_cllr(score, target, weight):
    """C_llr trial by trial, reading each score as a natural-log likelihood ratio."""
    total_t = sum(weight[i] for i in range(len(score)) if target[i])
    total_f = sum(weight[i] for i in range(len(score)) if not target[i])
    tar = sum(weight[i] * softplus(-float(score[i])) for i in range(len(score)) if target[i])
    non = sum(weight[i] * softplus(float(score[i])) for i in range(len(score)) if not target[i])
    return (tar / total_t + non / total_f) / (2.0 * LOG2)


def cost_through_llr(blocks, total_t, total_f):
    """The cost of a block-constant log-likelihood ratio, formed explicitly from the posterior."""
    odds = total_t / total_f
    tar = non = 0.0
    for t, f in blocks:
        p = t / (t + f)
        if p <= 0.0:
            llr = -math.inf
        elif p >= 1.0:
            llr = math.inf
        else:
            llr = math.log(p / (1.0 - p)) - math.log(odds)
        if t > 0.0:
            tar += t * (0.0 if llr == math.inf else softplus(-llr))
        if f > 0.0:
            non += f * (0.0 if llr == -math.inf else softplus(llr))
    return (tar / total_t + non / total_f) / (2.0 * LOG2)


def brute_min_cllr(score, target, weight):
    """Every monotone partition of the score-ordered trials into consecutive blocks. The search is
    2**(m-1) in the number m of DISTINCT scores, so every caller must bound m."""
    groups = []
    for i in sorted(range(len(score)), key=lambda j: float(score[j])):
        s = float(score[i])
        if groups and groups[-1][0] == s:
            groups[-1][1] += weight[i] if target[i] else 0.0
            groups[-1][2] += 0.0 if target[i] else weight[i]
        else:
            groups.append([s, weight[i] if target[i] else 0.0, 0.0 if target[i] else weight[i]])
    m = len(groups)
    assert m <= 8, f"the exhaustive search refuses {m} distinct scores"
    total_t = sum(g[1] for g in groups)
    total_f = sum(g[2] for g in groups)
    best = None
    for bits in range(1 << (m - 1)):
        blocks = []
        cur_t = cur_f = 0.0
        for i in range(m):
            cur_t += groups[i][1]
            cur_f += groups[i][2]
            if i == m - 1 or (bits >> i) & 1:
                blocks.append((cur_t, cur_f))
                cur_t = cur_f = 0.0
        shares = [t / (t + f) for t, f in blocks]
        if any(shares[i] > shares[i + 1] + 1e-15 for i in range(len(shares) - 1)):
            continue
        cost = cost_through_llr(blocks, total_t, total_f)
        best = cost if best is None else min(best, cost)
    assert best is not None
    return best


def sklearn_min_cllr(score, target, weight):
    """min-C_llr from sklearn's isotonic regression of the label on the score."""
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    weight = np.asarray(weight, dtype=np.float64)
    fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
        score, target.astype(np.float64), sample_weight=weight)
    blocks = []
    for i in np.argsort(score, kind="stable"):
        p = float(fitted[i])
        if blocks and abs(blocks[-1][0] - p) <= 1e-12:
            blocks[-1][1] += weight[i] if target[i] else 0.0
            blocks[-1][2] += 0.0 if target[i] else weight[i]
        else:
            blocks.append([p, weight[i] if target[i] else 0.0, 0.0 if target[i] else weight[i]])
    return cost_through_llr([(b[1], b[2]) for b in blocks], float(weight[target].sum()),
                            float(weight[~target].sum()))


def blocks_of_scores(score, target, weight):
    """One block per distinct score, in score order, for the array estimators."""
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="stable")
    s = np.asarray(score, dtype=np.float64)[order]
    tg = np.asarray(target, dtype=bool)[order]
    wt = np.asarray(weight, dtype=np.float64)[order]
    fresh = np.empty(len(s), dtype=bool)
    fresh[0] = True
    fresh[1:] = s[1:] != s[:-1]
    starts = np.flatnonzero(fresh)
    total = np.add.reduceat(wt, starts)
    hit = np.add.reduceat(wt * tg, starts)
    return hit, total - hit


def flatten(score_matrix, label_index, u):
    """The trial list as flat python lists, for the brute-force definitions."""
    n, L = score_matrix.shape
    score, target, weight = [], [], []
    for q in range(n):
        for l in range(L):
            score.append(float(score_matrix[q, l]))
            target.append(bool(l == label_index[q]))
            weight.append(float(u[q]))
    return score, target, weight


def component_weights(li, gi):
    size = Counter(zip(gi.tolist(), li.tolist()))
    return np.asarray([1.0 / size[(g, l)] for g, l in zip(gi, li)])


def reference_two_stage(per_label, replicates, seed):
    """A literal transcription of v1.run_bootstrap's generator calls: label_count outer label
    occurrences, then an independent within-label resample per occurrence. Returns the macro
    replicates and, separately, the plain mean over every drawn component."""
    rng = np.random.default_rng(int(seed))
    label_count = len(per_label)
    columns = int(per_label[0].shape[1])
    macro = np.zeros((replicates, columns))
    pooled = np.zeros((replicates, columns))
    for r in range(replicates):
        sampled = rng.integers(0, label_count, size=label_count)
        total = np.zeros(columns)
        num = np.zeros(columns)
        den = 0.0
        for label, count in Counter(int(x) for x in sampled.tolist()).items():
            block = per_label[label]
            draws = rng.integers(0, len(block), size=(count, len(block)))
            total += block[draws].mean(axis=1).sum(axis=0)
            num += block[draws].sum(axis=1).sum(axis=0)
            den += count * len(block)
        macro[r] = total / label_count
        pooled[r] = num / den
    return macro, pooled


# ------------------------------------------------------------------ checks
def check_constants() -> None:
    assert ec.MARGIN == 0.005
    assert ec.direction(0.006) == 1 and ec.direction(-0.006) == -1
    assert ec.direction(0.004) == 0 and ec.direction(-0.004) == 0
    assert ec.direction(ec.MARGIN) == 0 and ec.direction(-ec.MARGIN) == 0
    assert ec.direction(0.005000000000000004) == 0        # a rounded difference on the band
    assert ec.OUTCOME[(1, -1)].startswith("improvement")
    assert ec.OUTCOME[(1, 1)].startswith("a ranking gain bought with a calibration loss")
    assert ec.OUTCOME[(-1, -1)].startswith("a calibration gain bought with a ranking loss")
    assert ec.OUTCOME[(-1, 1)].startswith("no gain")
    assert len(ec.OUTCOME) == 9
    assert set(ec.ESTIMANDS) == {"query_weighted", "component_weighted", "label_macro"}
    assert ec.EXPECTED_PROTOTYPE == {"semantic_raw": 0.2997, "words_raw": 0.4963, "chars_raw": 0.4266,
                                     "semantic_whitened": 0.4164, "words_svd_whitened": 0.5426}
    assert ec.EXPECTED_SHAPE == {"queries": 7220, "labels": 226, "groups": 5875, "components": 6889}
    assert sum(v[0] for v in ec.EXPECTED_BANDS.values()) == 226
    assert sum(v[1] for v in ec.EXPECTED_BANDS.values()) == 7220
    assert ec.SPACES.index("words_svd_whitened") > ec.SPACES.index("words_raw")
    assert ec.GROUP_SEED == isv2.SEED and ec.GROUP_REPLICATES == isv2.REPLICATES
    assert ec.LLR_REPLICATES <= ec.GROUP_REPLICATES
    assert ec.UNIT_ROW_TOLERANCE["words_raw"] == 4e-5 and ec.UNIT_ROW_TOLERANCE["chars_raw"] == 4e-5
    assert ec.UNIT_ROW_TOLERANCE["semantic_raw"] == 4e-5
    assert ec.UNIT_ROW_TOLERANCE["semantic_whitened"] == 1e-12
    assert ec.GROUP_REBUILD_TOLERANCE > abs(0.0643 - 0.0644)   # one unit in the published decimal
    # the pinned/imported cross-check: None means the other file's own check is off, not a clash
    assert ec.pinned_mismatch({"a": 0.5}, {"a": None}, 1e-9) == []
    assert ec.pinned_mismatch({"a": 0.5}, {"a": 0.5}, 1e-9) == []
    assert len(ec.pinned_mismatch({"a": 0.5}, {"a": 0.5188}, 1e-9)) == 1
    print("ok: the declared band, the four named outcomes, the pinned publications and the "
          "provenance tolerances are as fixed")


def check_quantile_conventions(rng) -> None:
    a = rng.normal(size=777)
    assert list(np.percentile(a, [2.5, 97.5])) == list(np.quantile(a, [0.025, 0.975]))
    print("ok: paired_group_bootstrap's percentile convention and estimand_two_stage's quantile "
          "convention are the same call")


def check_trial_blocks(rng) -> None:
    n, L = 7, 5
    li = rng.integers(0, L, size=n)
    scores = np.round(rng.normal(size=(n, L)), 1)
    bundle = ec.trial_blocks(scores, li)
    assert bundle["trials"] == n * L
    assert bundle["target_trials"] == n and bundle["nontarget_trials"] == n * (L - 1)
    assert int(bundle["target"].sum()) == n
    flat = scores.reshape(-1)
    order = np.argsort(flat, kind="stable")
    assert np.all(np.diff(flat[order]) >= 0.0)
    for position in range(len(order)):
        q, l = divmod(int(order[position]), L)
        assert int(bundle["query"][position]) == q
        assert bool(bundle["target"][position]) == bool(l == li[q])
    print("ok: the trial list decodes to the right query and class in score order")


def check_auc_and_cllr(rng) -> None:
    worst = [0.0, 0.0]
    for _ in range(40):
        n, L = int(rng.integers(4, 9)), int(rng.integers(3, 6))
        li = rng.integers(0, L, size=n)
        scores = np.round(rng.normal(size=(n, L)), 1)          # deliberate exact ties
        u = rng.random(n) + 0.05
        bundle = ec.trial_blocks(scores, li)
        target_term, nontarget_term = ec.cllr_parts(scores, li)
        t, f = ec.block_weight(bundle, u)
        mine_cllr = ec.cllr_from_parts(target_term, nontarget_term, u, L)
        score, target, weight = flatten(scores, li, u)
        worst[0] = max(worst[0], abs(ec.auc_from_blocks(t, f) - brute_auc(score, target, weight)))
        worst[1] = max(worst[1], abs(mine_cllr - brute_cllr(score, target, weight)))
        assert ec.min_cllr(t, f) <= mine_cllr + 1e-12
    assert max(worst) < 1e-12, worst
    print(f"ok: AUC and C_llr equal their brute-force definitions (max gaps {worst[0]:.1e}, "
          f"{worst[1]:.1e}); min-C_llr never exceeds C_llr")


def check_min_cllr_exhaustive(rng) -> None:
    """min-C_llr against the exhaustive monotone-partition search. The search is 2**(m-1) in the
    number m of distinct scores, so the scores come from a four-value integer alphabet: m <= 4 and
    at most eight partitions, whatever the seed does."""
    worst, cases = 0.0, 0
    for _ in range(30):
        n, L = 4, 3
        li = rng.integers(0, L, size=n)
        scores = rng.integers(0, 4, size=(n, L)).astype(np.float64)
        u = rng.random(n) + 0.05
        t, f = ec.block_weight(ec.trial_blocks(scores, li), u)
        score, target, weight = flatten(scores, li, u)
        assert len(set(score)) <= 4
        worst = max(worst, abs(ec.min_cllr(t, f) - brute_min_cllr(score, target, weight)))
        cases += 1
    assert cases == 30 and worst < 1e-12, (cases, worst)
    print(f"ok: min-C_llr equals the exhaustive monotone-partition search on {cases} heavily tied "
          f"cases (max gap {worst:.1e})")


def check_against_sklearn(rng) -> None:
    worst = 0.0
    for _ in range(30):
        m = int(rng.integers(6, 40))
        score = np.round(rng.normal(size=m), 1)
        target = rng.random(m) < 0.4
        if target.sum() == 0 or (~target).sum() == 0:
            continue
        weight = rng.random(m) + 0.05
        hit, miss = blocks_of_scores(score, target, weight)
        worst = max(worst, abs(ec.min_cllr(hit, miss) - sklearn_min_cllr(score, target, weight)))
    assert worst < 1e-10, worst
    print(f"ok: the PAV calibration agrees with sklearn's isotonic regression on up to 40 distinct "
          f"scores (max gap {worst:.1e})")


def check_pooling(rng) -> None:
    worst = [0.0, 0.0]
    for _ in range(40):
        n, L = int(rng.integers(5, 14)), int(rng.integers(3, 8))
        li = rng.integers(0, L, size=n)
        scores = np.round(rng.normal(size=(n, L)), 1)
        u = rng.random(n) + 0.05
        pooled = ec.trial_blocks(scores, li, pool=True)
        plain = ec.trial_blocks(scores, li, pool=False)
        assert pooled["blocks"] <= plain["blocks"]
        a, b = ec.block_weight(pooled, u), ec.block_weight(plain, u)
        worst[0] = max(worst[0], abs(ec.auc_from_blocks(*a) - ec.auc_from_blocks(*b)))
        worst[1] = max(worst[1], abs(ec.min_cllr(*a) - ec.min_cllr(*b)))
    assert max(worst) < 1e-12, worst
    print(f"ok: pooling neighbouring same-class trials changes no reading (max gaps {worst[0]:.1e}, "
          f"{worst[1]:.1e}), ties included")


def check_limiting_cases() -> None:
    assert abs(ec.cost_of_blocks(np.array([3.0]), np.array([7.0])) - 1.0) < 1e-12
    zeros = np.zeros((4, 3))
    li = np.array([0, 1, 2, 0])
    u = np.ones(4)
    t, f = ec.block_weight(ec.trial_blocks(zeros, li), u)
    target_term, nontarget_term = ec.cllr_parts(zeros, li)
    assert abs(ec.cllr_from_parts(target_term, nontarget_term, u, 3) - 1.0) < 1e-12
    assert abs(ec.min_cllr(t, f) - 1.0) < 1e-12
    assert abs(ec.auc_from_blocks(t, f) - 0.5) < 1e-12
    print("ok: a system that always outputs 0 gives C_llr = min-C_llr = 1 and AUC = 0.5")

    perfect = np.array([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    li3 = np.array([0, 1, 2])
    t, f = ec.block_weight(ec.trial_blocks(perfect, li3), np.ones(3))
    assert abs(ec.min_cllr(t, f)) < 1e-12 and abs(ec.auc_from_blocks(t, f) - 1.0) < 1e-12
    t, f = ec.block_weight(ec.trial_blocks(-perfect, li3), np.ones(3))
    assert abs(ec.min_cllr(t, f) - 1.0) < 1e-12 and abs(ec.auc_from_blocks(t, f)) < 1e-12
    print("ok: perfect separation gives min-C_llr 0 and AUC 1; reversing it gives 1 and 0")


def check_true_llr(rng) -> None:
    """Scores that ARE the log-likelihood ratios of a two-Gaussian model are calibrated, and
    min-C_llr is unchanged by any strictly increasing map of them."""
    m, sd, size = 1.3, 1.0, 40000
    score = np.concatenate([2.0 * m * rng.normal(m, sd, size=size) / sd ** 2,
                            2.0 * m * rng.normal(-m, sd, size=size) / sd ** 2])
    target = np.concatenate([np.ones(size, dtype=bool), np.zeros(size, dtype=bool)])
    weight = np.ones(2 * size)
    mine_min = ec.min_cllr(*blocks_of_scores(score, target, weight))
    mine_cllr = brute_cllr(score.tolist(), target.tolist(), weight.tolist())
    assert mine_min <= mine_cllr + 1e-12
    assert mine_cllr - mine_min < 0.01, (mine_cllr, mine_min)
    assert 0.1 < mine_min < 0.9, mine_min
    for transform in (lambda z: 3.0 * z + 1.0, np.tanh, lambda z: np.exp(z / 8.0), lambda z: z ** 3):
        moved = np.asarray(transform(score), dtype=np.float64)
        assert abs(ec.min_cllr(*blocks_of_scores(moved, target, weight)) - mine_min) < 1e-9, transform
    print(f"ok: the true log-likelihood ratios of a two-Gaussian model are calibrated (C_llr "
          f"{mine_cllr:.4f}, min-C_llr {mine_min:.4f}) and min-C_llr is monotone-invariant")


def check_estimands(rng):
    n, L = 240, 11
    li = rng.integers(0, L, size=n)
    li[:L] = np.arange(L)
    gi = rng.integers(0, 90, size=n)
    w = component_weights(li, gi)
    rr = rng.random(n)
    u = ec.estimand_weights(li, w, L)
    mask = np.ones(n, dtype=bool)
    assert abs(weighted_mean(rr, u["query_weighted"], mask) - float(rr.mean())) < 1e-12
    assert abs(weighted_mean(rr, u["component_weighted"], mask) - weighted_mean(rr, w, mask)) < 1e-12
    macro = float(ets.label_macro(rr[:, None], w, li, L)[0])
    assert abs(weighted_mean(rr, u["label_macro"], mask) - macro) < 1e-12
    assert abs(float(np.sum(u["label_macro"])) - 1.0) < 1e-12
    comp_of, comp_label = ets.component_index(li, gi)
    comp_values = ets.component_means(rr[:, None], comp_of, len(comp_label))
    assert abs(float(comp_values.mean()) - weighted_mean(rr, w, mask)) < 1e-12
    starts = ec.label_blocks(comp_label, L)
    assert starts[0] == 0 and len(starts) == L
    assert list(starts) == sorted(starts)
    assert list(np.add.reduceat(np.ones(len(comp_label)), starts)) == \
           list(np.bincount(comp_label, minlength=L).astype(float))
    print("ok: the three estimand weights reproduce the plain, component-weighted and label-macro "
          "means, and the component table is grouped by label")
    return li, gi, w, rr, comp_of, comp_label


def check_group_bootstrap(li, gi, w, rr, comp_of, comp_label) -> None:
    L = int(li.max()) + 1
    n = len(li)
    other = rr * 0.5 + 0.1
    values = np.column_stack([rr, other])
    comp_values = ets.component_means(values, comp_of, len(comp_label))
    index, draws, group_count = ec.group_positions_and_draws(gi, isv2.REPLICATES, isv2.SEED)
    comp_group = np.zeros(len(comp_label), dtype=np.int64)
    comp_group[comp_of] = index
    starts = ec.label_blocks(comp_label, L)
    component, macro, empty = ec.group_replicates(comp_values, comp_group, starts, draws, group_count)
    mask = np.ones(n, dtype=bool)
    point = weighted_mean(rr, w, mask) - weighted_mean(other, w, mask)
    mine = ec.cell(component[:, 0] - component[:, 1], point)
    theirs = paired_group_bootstrap({"a": rr, "b": other}, w, gi, mask, [("a", "b")])[0]
    assert mine["difference"] == theirs["mrr_difference"], (mine, theirs)
    assert mine["ci95"] == theirs["ci95"], (mine, theirs)
    assert "excludes_zero" not in mine                      # reading rule 1: no verdict in the 2x2
    assert "excludes_zero" in ec.cell(component[:, 0] - component[:, 1], point, verdict=True)
    # one replicate, by hand: the label-macro of the drawn components
    m = np.bincount(draws[0], minlength=group_count).astype(np.float64)[comp_group]
    by_hand = [float(np.sum(comp_values[comp_label == l, 0] * m[comp_label == l])
                     / np.sum(m[comp_label == l])) for l in range(L) if np.sum(m[comp_label == l]) > 0]
    assert abs(float(macro[0, 0]) - float(np.mean(by_hand))) < 1e-12
    assert int(empty[0]) == L - len(by_hand)
    assert empty.shape == (isv2.REPLICATES,)
    macro_point = ets.label_macro(values, w, li, L)
    assert abs(float(macro.mean(axis=0)[0]) - float(macro_point[0])) < 0.05
    print("ok: the rebuilt group bootstrap reproduces paired_group_bootstrap exactly, point and "
          "interval, and its label-macro replicates equal the definition replicate by replicate")


def check_two_stage_columns(rng) -> None:
    import build_chinese_rap_downstream_retrieval_v1 as v1
    n, L, replicates, seed = 200, 9, 60, 4321
    li = rng.integers(0, L, size=n)
    li[:L] = np.arange(L)
    gi = rng.integers(0, 70, size=n)
    rr = rng.random(n)
    other = rng.random(n)
    comp_of, comp_label = ets.component_index(li, gi)
    size = np.bincount(comp_label, minlength=L).astype(np.float64)[li]
    systems = ("a", "b")
    metric_arrays = {name: {"mrr": v, "size_weighted_mrr": v * size, "component_count": size}
                     for name, v in (("a", rr), ("b", other))}
    _components, counts, point, reps, _diag = ets.two_stage(
        metric_arrays, li, gi, systems, ec.TWO_STAGE_METRICS, replicates=replicates, seed=seed)
    macro_point, ratio_point, macro_reps, component_reps, spread = ec.two_stage_columns(point, reps)
    assert spread == 0.0, spread
    assert int(counts.sum()) == len(comp_label)
    assert v1.BOOTSTRAP_REPLICATES == 5000 and v1.RANDOM_SEED == 20260825

    values = np.column_stack([rr, other])
    comp_values = ets.component_means(values, comp_of, len(comp_label))
    mask = np.ones(n, dtype=bool)
    # the points: the macro column is the label-macro, the ratio is the component-weighted mean
    assert np.abs(macro_point - ets.label_macro(values, _w(li, gi), li, L)).max() < 1e-12
    assert abs(float(ratio_point[0]) - weighted_mean(rr, _w(li, gi), mask)) < 1e-12
    assert abs(float(ratio_point[1]) - weighted_mean(other, _w(li, gi), mask)) < 1e-12
    # the replicates: against a literal transcription of v1.run_bootstrap's draws
    per_label = [np.column_stack([comp_values[comp_label == l, 0], comp_values[comp_label == l, 1]])
                 for l in range(L)]
    per_label = [np.column_stack([block, block * 0.0 + 0.0]) for block in per_label]  # placeholder
    per_label = []
    for l in range(L):
        rows = comp_label == l
        n_l = float(rows.sum())
        per_label.append(np.column_stack([comp_values[rows, 0], comp_values[rows, 0] * n_l,
                                          np.full(int(n_l), n_l), comp_values[rows, 1],
                                          comp_values[rows, 1] * n_l, np.full(int(n_l), n_l)]))
    macro_reference, pooled_reference = reference_two_stage(per_label, replicates, seed)
    assert float(np.abs(macro_reps[:, 0] - macro_reference[:, 0]).max()) == 0.0
    assert float(np.abs(macro_reps[:, 1] - macro_reference[:, 3]).max()) == 0.0
    ratio_reference = macro_reference[:, 1] / macro_reference[:, 2]
    assert float(np.abs(component_reps[:, 0] - ratio_reference).max()) < 1e-12
    # and the ratio IS the plain mean over every drawn component, the component-weighted estimand
    assert float(np.abs(component_reps[:, 0] - pooled_reference[:, 0]).max()) < 1e-12
    assert float(np.abs(component_reps[:, 1] - pooled_reference[:, 3]).max()) < 1e-12
    print("ok: the builder's macro column is the label-macro estimand and the size-weighted ratio "
          "is the plain mean over every drawn component, replicate for replicate")


def _w(li, gi):
    return component_weights(li, gi)


def check_min_cllr_series(rng) -> None:
    n, L = 60, 5
    li = rng.integers(0, L, size=n)
    gi = rng.integers(0, 20, size=n)
    scores = rng.normal(size=(n, L))
    u = component_weights(li, gi)
    index, draws, group_count = ec.group_positions_and_draws(gi, 40, 99)
    bundle = ec.trial_blocks(scores, li)
    series = ec.min_cllr_group_series(bundle, u, index, draws, group_count)
    assert series.shape == (40,)
    for r in range(40):
        m = np.bincount(draws[r], minlength=group_count).astype(np.float64)
        t, f = ec.block_weight(bundle, u * m[index])
        assert abs(series[r] - ec.min_cllr(t, f)) < 1e-12, r
    print("ok: the min-C_llr bootstrap series equals the definition applied replicate by replicate")


def check_decomposition() -> None:
    widths = {("component_weighted", "group_bootstrap"): 0.0142,
              ("label_macro", "group_bootstrap"): 0.0201,
              ("component_weighted", "two_stage"): 0.0199,
              ("label_macro", "two_stage"): 0.0283}
    out = ec.decompose(widths)
    assert abs(out["total_factor"] - 0.0283 / 0.0142) < 5e-4, out
    path_one = (0.0201 / 0.0142) * (0.0283 / 0.0201)
    path_two = (0.0283 / 0.0199) * (0.0199 / 0.0142)
    assert abs(path_one - path_two) < 1e-12
    assert abs(path_one - 0.0283 / 0.0142) < 1e-12
    assert abs(out["share_of_log_total_estimand"] + out["share_of_log_total_resampling"] - 1.0) < 2e-3
    flat = ec.decompose({k: 0.02 for k in widths})
    assert flat["total_factor"] == 1.0 and flat["share_of_log_total_estimand"] is None
    print("ok: the width decomposition is exact along either path, its shares sum to one, and a "
          "total factor of one reports no share")


def main() -> int:
    rng = np.random.default_rng(20260923)
    check_constants()
    check_quantile_conventions(rng)
    check_trial_blocks(rng)
    check_auc_and_cllr(rng)
    check_min_cllr_exhaustive(rng)
    check_against_sklearn(rng)
    check_pooling(rng)
    check_limiting_cases()
    check_true_llr(rng)
    li, gi, w, rr, comp_of, comp_label = check_estimands(rng)
    check_group_bootstrap(li, gi, w, rr, comp_of, comp_label)
    check_two_stage_columns(rng)
    check_min_cllr_series(rng)
    check_decomposition()
    print("all estimand-and-calibration checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
