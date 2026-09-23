"""Synthetic checks for src/calibrated_profile_scorer_v3.py, without the corpus:

    python tests/test_calibrated_profile_scorer.py

Every estimator the file adds is checked against a definition written out independently here, on
small random data, and the resultant-length estimator is checked for bias against synthetic labels
whose true r^2 is known exactly, at this corpus's n and d. Runs in a few seconds.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import calibrated_profile_scorer_v3 as cps  # noqa: E402
from identity_probe_v2 import dense_leave_group_out  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from label_size_calibration_v3 import lgo_parts, znorm  # noqa: E402


def synthetic(seed=7, n=300, labels=11, dimension=24, duplicate_groups=14):
    """A small population with the protocol's shape: leakage groups, some of them holding several
    songs of one label (a near-duplicate component), unit rows, component weights, folds assigned
    per group so that a component lies wholly in one fold."""
    rng = np.random.default_rng(seed)
    label_index = rng.integers(0, labels, size=n)
    group_ids = rng.integers(0, 90, size=n)
    for k in range(duplicate_groups):          # force some multi-song components
        members = rng.choice(n, size=3, replace=False)
        group_ids[members] = 200 + k
        label_index[members] = k % labels
    size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    x = rng.normal(size=(n, dimension)) + 0.9 * rng.normal(size=(labels, dimension))[label_index]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    fold_of_group = {int(g): int(rng.integers(0, 5)) for g in np.unique(group_ids)}
    fold = np.asarray([fold_of_group[int(g)] for g in group_ids], dtype=np.int64)
    return x, label_index, group_ids, weights, labels, fold


def literal(x, query, group_ids, label_index, weights, label, prior) -> dict:
    """The definitions, written out here and not imported: the label's remaining components, the
    U-statistic as a plain average over ORDERED PAIRS of distinct components, q.m_l, the prototype
    and S_C."""
    rows = [s for s in np.flatnonzero(label_index == label) if group_ids[s] != group_ids[query]]
    by_group = defaultdict(list)
    for s in rows:
        by_group[int(group_ids[s])].append(s)
    units = [np.sum([weights[s] * x[s] for s in members], axis=0) for members in by_group.values()]
    pairs = [float(a @ b) for i, a in enumerate(units) for j, b in enumerate(units) if i != j]
    r2 = float(np.mean(pairs))
    total = np.sum(units, axis=0)
    exemplar = float(x[query] @ total) / float(len(units))
    variance = 4.0 * float(prior["zeta1"][label]) / float(len(units))
    tau2 = prior["between_variance"]
    b = tau2 / (tau2 + variance)
    r_tilde2 = prior["prior_mean"] + b * (r2 - prior["prior_mean"])
    v = b * b * variance
    r_star2 = 0.5 * (r_tilde2 + np.sqrt(r_tilde2 * r_tilde2 + 4.0 * v))
    kappa = 1.0 / (1.0 + cps.DELTA * v / (r_star2 * r_star2))
    return {"components": len(units), "q_sum": float(np.sum([u @ u for u in units])), "r2": r2,
            "exemplar_mean": exemplar, "prototype": float(x[query] @ total) / float(np.linalg.norm(total)),
            "calibrated_S_C": exemplar * kappa / float(np.sqrt(r_star2))}


def unit_population(alpha, n, replicates, dimension, rng):
    """Unit vectors whose mean vector is EXACTLY alpha * e0, so r^2 = alpha^2 is known.

    x_i = c_i e0 + sqrt(1 - c_i^2) v_i with v_i uniform on the sphere of the complement and
    c_i = alpha +/- h equiprobable, so E[c] = alpha and E[v] = 0 give E[x] = alpha e0 exactly,
    while Var(x.mu) = alpha^2 h^2 > 0: the label varies along its own mean direction too.
    """
    h = 0.4
    c = alpha + h * (2.0 * rng.integers(0, 2, size=(replicates, n)) - 1.0)
    v = rng.normal(size=(replicates, n, dimension - 1))
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    return np.concatenate([c[:, :, None], np.sqrt(1.0 - c ** 2)[:, :, None] * v], axis=2)


def main() -> int:
    x, li, gi, w, L, fold = synthetic()
    n = len(li)
    everything = np.arange(n)
    prior = cps.resultant_prior(x, li, gi, w, L, fold != 0)

    # ---- component_parts, and the weight count the U-statistic's denominator assumes
    ncomp, q_sum = cps.component_parts(x, li, gi, w, L, everything)
    dots, norm2, counts = lgo_parts(x, li, gi, w, L, everything)
    assert float(np.abs(counts - ncomp).max()) <= cps.COUNT_GAP
    for query in range(0, n, 37):
        for label in range(L):
            reference = literal(x, query, gi, li, w, label, prior)
            assert int(ncomp[query, label]) == reference["components"], (query, label)
            assert abs(q_sum[query, label] - reference["q_sum"]) < 1e-12
    print("ok: component_parts matches its definition; the weight count is the component count")

    # ---- the U-statistic, q.m_l, the prototype and S_C against the definitions above
    r2 = cps.resultant_squared(norm2, q_sum, ncomp)
    calibrated, parts = cps.calibrated_scores(dots, counts, r2, ncomp, prior)
    worst = {"r2": 0.0, "exemplar_mean": 0.0, "prototype": 0.0, "calibrated_S_C": 0.0}
    for query in range(0, n, 29):
        for label in range(L):
            reference = literal(x, query, gi, li, w, label, prior)
            got = {"r2": float(r2[query, label]),
                   "exemplar_mean": float(dots[query, label] / counts[query, label]),
                   "prototype": float(dots[query, label] / np.sqrt(norm2[query, label])),
                   "calibrated_S_C": float(calibrated[query, label])}
            brute = cps.brute_scores(x, query, gi, li, w, label, prior)
            for what in worst:
                worst[what] = max(worst[what], abs(got[what] - reference[what]))
                assert abs(brute[what] - reference[what]) < 1e-10, (what, query, label)
    for what, gap in worst.items():
        assert gap <= 1e-9, (what, gap)
    print(f"ok: r_hat^2, q.m_l, the prototype and S_C equal their definitions "
          f"({max(worst.values()):.1e}), and so does the in-run brute-force probe")
    gap = float(np.abs(dots / np.sqrt(norm2) - dense_leave_group_out(x, li, gi, w, L, everything)).max())
    assert gap <= 1e-9, gap
    print(f"ok: the prototype is the protocol's own dense scorer (gap {gap:.1e})")

    # ---- the prior: per-label U-statistic and delete-one-component jackknife, literally
    rows = np.flatnonzero(fold != 0)
    for label in range(L):
        members = [s for s in rows if li[s] == label]
        by_group = defaultdict(list)
        for s in members:
            by_group[int(gi[s])].append(s)
        units = [np.sum([w[s] * x[s] for s in group], axis=0) for group in by_group.values()]
        count = len(units)
        assert count == int(prior["components_train"][label])
        if count < 2:
            continue
        pairs = [float(a @ b) for i, a in enumerate(units) for j, b in enumerate(units) if i != j]
        assert abs(prior["u_full"][label] - float(np.mean(pairs))) < 1e-10, label
        if count < 3:
            continue
        deleted = []
        for j in range(count):
            rest = [u for i, u in enumerate(units) if i != j]
            deleted.append(float(np.mean([float(a @ b) for p, a in enumerate(rest)
                                          for q, b in enumerate(rest) if p != q])))
        deleted = np.asarray(deleted)
        v_reference = (count - 1) / count * float(np.sum((deleted - deleted.mean()) ** 2))
        zeta_reference = v_reference * count / 4.0
        assert abs(prior["zeta1"][label] - zeta_reference) < 1e-9 * max(1.0, zeta_reference), label
    assert prior["prior_mean"] > 0 and prior["between_variance"] >= 0
    print("ok: the prior's U-statistic and its delete-one-component jackknife match literal loops")

    # ---- the shape of the divisor, on a grid, with no tolerance that is not exact arithmetic
    assert np.all(parts["r_star2"] > 0) and np.all(parts["kappa"] > 0) and np.all(parts["kappa"] <= 1.0)
    non_negative = parts["r_tilde2"] >= 0.0
    assert np.all(parts["kappa"][non_negative] >= 8.0 / 11.0 - 1e-12)
    sd = 0.02
    grid = np.linspace(-1.0, 1.0, 2001)
    variance = np.full_like(grid, sd ** 2)
    star = 0.5 * (grid + np.sqrt(grid * grid + 4.0 * variance))
    assert np.all(star > 0)
    assert np.all(np.diff(star) > 0)
    # r_star^2 is the positive root of r_star^2 (r_star^2 - m) = v: an identity, not an expansion
    assert float(np.abs(star * star - grid * star - variance).max()) < 1e-12
    assert abs(star[np.argmin(np.abs(grid))] - sd) < 1e-15        # equals the sd at zero
    # above zero the offset is 2v/(sqrt(m^2+4v)+m), which is positive and below v/m exactly
    up = grid > 0
    assert np.all(star[up] - grid[up] > 0)
    assert np.all(star[up] - grid[up] <= variance[up] / grid[up])
    kappa = 1.0 / (1.0 + cps.DELTA * variance / (star * star))
    assert np.all((kappa > 0) & (kappa <= 1.0))
    assert np.all(kappa[grid >= 0] >= 8.0 / 11.0 - 1e-12)
    # 1 - kappa = z/(1+z) with z = DELTA v / r_star^4 <= DELTA v / m^2 for m > 0
    assert np.all(1.0 - kappa[up] <= cps.DELTA * variance[up] / (grid[up] * grid[up]))
    print("ok: the soft positive part is positive, monotone and solves its own identity; kappa "
          "stays in (0, 1] and above 8/11 where r_tilde^2 >= 0")

    # ---- unbiasedness at KNOWN r^2, at the corpus's n and d, and the r_hat^2 <= 0 rate
    rng = np.random.default_rng(20260923)
    alpha = float(np.sqrt(0.03))
    dimension, replicates, block_size = 1024, 1500, 150
    print(f"  known r^2 = {alpha ** 2:.4f}, d = {dimension}, {replicates} replicates per n")
    by_n = {}
    for size in (4, 5, 8, 16, 32):
        values = []
        for start in range(0, replicates, block_size):
            block = unit_population(alpha, size, min(block_size, replicates - start), dimension, rng)
            total = block.sum(axis=1)
            norms = np.einsum("ij,ij->i", total, total)
            values.append(cps.resultant_squared(norms, np.full(len(total), float(size)),
                                                np.full(len(total), size, dtype=np.int64)))
        values = np.concatenate(values)
        bias = float(values.mean() - alpha ** 2)
        sem = float(values.std(ddof=1) / np.sqrt(len(values)))
        by_n[size] = (bias, sem, float(np.mean(values <= 0.0)), float(values.std(ddof=1)))
        print(f"  n = {size:2d}: bias {bias:+.5f} (4 sem {4 * sem:.5f}), sd {by_n[size][3]:.4f}, "
              f"share r_hat^2 <= 0 {by_n[size][2]:.3f}")
        assert abs(bias) <= 4 * sem, (size, bias, sem)   # a seeded 4-sigma check, not a proof
    assert by_n[5][2] > 0.15, by_n[5]
    assert by_n[5][3] > alpha ** 2, "the sd should rival r^2 itself at n = 5"
    assert by_n[32][2] < by_n[5][2]
    print("ok: the component estimator is unbiased at every n, and r_hat^2 <= 0 is common at n = 5")

    # ---- the estimand helpers
    rr = 1.0 / (1.0 + np.abs(np.random.default_rng(3).normal(size=n)))
    table = cps.three_estimands(rr, w, li)
    plain = float(np.mean(rr))
    component = float(np.sum(rr * w) / np.sum(w))
    macro = float(np.mean([np.sum(rr[li == l] * w[li == l]) / np.sum(w[li == l]) for l in range(L)]))
    assert abs(table["query_weighted"] - round(plain, 4)) < 1e-9
    assert abs(table["component_weighted"] - round(component, 4)) < 1e-9
    assert abs(table["label_macro"] - round(macro, 4)) < 1e-9
    table = cps.three_estimands(rr, w, li, li < 4)
    macro = float(np.mean([np.sum(rr[li == l] * w[li == l]) / np.sum(w[li == l]) for l in range(4)]))
    assert abs(table["label_macro"] - round(macro, 4)) < 1e-9
    print("ok: the three estimands match literal loops, whole population and inside a band")

    # ---- the shared-draw bootstrap reproduces paired_group_bootstrap on one universe
    other = 1.0 / (1.0 + np.abs(np.random.default_rng(4).normal(size=n)))
    systems = {"left": rr, "right": other}
    mask = li < 6
    drawn, _ = cps.shared_draw_bootstrap(systems, w, gi, li, mask, {"band": mask})
    mine = cps.contrast("left", "right", drawn["band"]["component_weighted"]["left"],
                        drawn["band"]["component_weighted"]["right"], "component_weighted")
    theirs = paired_group_bootstrap(systems, w, gi, mask, [("left", "right")])[0]
    assert mine["difference"] == theirs["mrr_difference"], (mine, theirs)
    assert mine["ci95"] == theirs["ci95"], (mine, theirs)
    print("ok: the shared-draw bootstrap reproduces paired_group_bootstrap exactly on one band")

    # one draw shared by two masks lets a gradient be a difference of differences
    low, high, everyone = li < 4, li >= 4, np.ones(n, dtype=bool)
    drawn, lost = cps.shared_draw_bootstrap(systems, w, gi, li, everyone,
                                            {"low": low, "high": high, "all": everyone},
                                            full_estimands=("all",))
    gradient = (drawn["high"]["component_weighted"]["left"]["replicates"]
                - drawn["low"]["component_weighted"]["left"]["replicates"])
    assert gradient.shape == drawn["all"]["component_weighted"]["left"]["replicates"].shape
    point = (drawn["high"]["component_weighted"]["left"]["point"]
             - drawn["low"]["component_weighted"]["left"]["point"])
    assert np.percentile(gradient, 2.5) <= point + 1e-6
    assert np.percentile(gradient, 97.5) >= point - 1e-6
    assert "label_macro" not in drawn["low"] and "query_weighted" not in drawn["low"]
    reference = float(np.mean([np.sum(rr[li == l] * w[li == l]) / np.sum(w[li == l]) for l in range(L)]))
    assert abs(drawn["all"]["label_macro"]["left"]["point"] - reference) < 1e-9
    assert 0.0 <= lost < 1.0
    print(f"ok: one shared draw gives the gradient an interval, band masks carry only the "
          f"component-weighted estimand, labels lost at most {lost:.3f} of a replicate")

    # ---- znorm is invariant to a per-label rescale, so a correction-then-cohort stack is a no-op
    scores = np.random.default_rng(5).normal(size=(n, L))
    scale = 0.5 + np.random.default_rng(6).random(L)
    test = np.flatnonzero(fold == 0)
    a = znorm(scores, li, fold != 0, test)
    b = znorm(scores * scale[None, :], li, fold != 0, test)
    assert float(np.abs(a - b).max()) < 1e-9, float(np.abs(a - b).max())
    print("ok: znorm is invariant to per-label positive rescaling; the stack would be a no-op")

    print("all calibrated-profile-scorer checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
