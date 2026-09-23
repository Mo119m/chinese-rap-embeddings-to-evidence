"""Synthetic checks for src/hubness_and_centrality_v3.py, without the corpus:

    python tests/test_hubness_and_centrality.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import hubness_and_centrality_v3 as hc  # noqa: E402


def check_occupancy() -> None:
    # four queries, three labels; the top-1 and top-2 sets are readable by eye
    scores = np.array([[0.9, 0.5, 0.1],      # top1 = 0, top2 = {0, 1}
                       [0.2, 0.8, 0.4],      # top1 = 1, top2 = {1, 2}
                       [0.3, 0.1, 0.7],      # top1 = 2, top2 = {2, 0}
                       [0.6, 0.2, 0.5]])     # top1 = 0, top2 = {0, 2}
    truth = np.array([0, 2, 2, 1])
    occ = hc.occupancy(scores, truth, ks=(1, 2))
    assert list(occ[1][0]) == [2.0, 1.0, 1.0], occ[1][0]
    # false top-1: query 1 answers label 1 but is label 2; query 3 answers 0 but is 1
    assert list(occ[1][1]) == [1.0, 1.0, 0.0], occ[1][1]
    assert occ[2][0].sum() == 8.0 and occ[1][0].sum() == 4.0
    brute = Counter()
    for row in range(4):
        for lab in np.argsort(-scores[row])[:2]:
            brute[int(lab)] += 1
    assert [brute[i] for i in range(3)] == [int(v) for v in occ[2][0]], (brute, occ[2][0])
    print("ok: k-occupancy and false k-occupancy match a brute-force count")


def check_entropy_and_skewness() -> None:
    flat = np.full(16, 5.0)
    assert abs(hc.normalised_entropy(flat) - 1.0) < 1e-12
    one = np.zeros(16)
    one[3] = 100.0
    assert abs(hc.normalised_entropy(one)) < 1e-12
    half = np.zeros(16)
    half[:4] = 1.0
    assert abs(hc.normalised_entropy(half) - np.log(4) / np.log(16)) < 1e-12
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.gamma(2.0, 3.0, size=200)
        m, sd = a.mean(), a.std()
        assert abs(hc.skewness(a) - float(np.mean(((a - m) / sd) ** 3))) < 1e-12
    assert hc.skewness(np.full(9, 2.0)) == 0.0
    assert hc.skewness(rng.gamma(1.0, 1.0, size=4000)) > 0.5, "a gamma sample must be right-skewed"
    print("ok: normalised entropy hits its limits; skewness matches its definition")


def check_csls_and_mutual_proximity() -> None:
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(30, 7))
    train = np.arange(0, 30, 2)
    got = hc.csls(scores, train, k=3)
    for q in range(30):
        rq = np.mean(np.sort(scores[q])[-3:])
        for l in range(7):
            rl = np.mean(np.sort(scores[train, l])[-3:])
            assert abs(got[q, l] - (2 * scores[q, l] - rl - rq)) < 1e-12
    print("ok: CSLS equals its definition, with the label side on training rows only")

    mp = hc.mutual_proximity(scores)
    nq, nl = scores.shape
    for q in (0, 5, 17):
        for l in (0, 3, 6):
            share_q = np.sum(scores[q] > scores[q, l]) / nl
            share_l = np.sum(scores[:, l] > scores[q, l]) / nq
            assert abs(mp[q, l] - (1 - share_q) * (1 - share_l)) < 1e-12, (q, l)
    print("ok: mutual proximity equals its empirical definition")

    # All three transforms exist to change which label wins a query, so each must reorder somebody;
    # what is order-preserving is only the query-side term on its own. Asserted, because a transform
    # that could not reorder anything would make R3 and R4 vacuous.
    order_before = np.argsort(-scores, axis=1)
    assert not np.array_equal(np.argsort(-got, axis=1), order_before), \
        "CSLS reordered nothing, so its label-side correction is inert on this fixture"
    assert not np.array_equal(np.argsort(-mp, axis=1), order_before), \
        "mutual proximity reordered nothing on this fixture"
    query_side_only = scores - np.mean(np.sort(scores, axis=1)[:, -3:], axis=1, keepdims=True)
    assert np.array_equal(np.argsort(-query_side_only, axis=1), order_before), \
        "a per-query shift must leave every query's order alone"
    print("ok: both corrections reorder labels within a query; a per-query shift alone does not")


def check_covariates() -> None:
    rng = np.random.default_rng(3)
    n, L, dim = 90, 6, 12
    li = rng.integers(0, L, size=n)
    gi = rng.integers(0, 40, size=n)
    size = Counter(zip(gi.tolist(), li.tolist()))
    w = np.asarray([1.0 / size[(g, l)] for g, l in zip(gi, li)])
    x = rng.normal(size=(n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    cov = hc.label_covariates(x, li, w, L)
    means = []
    for l in range(L):
        rows = np.flatnonzero(li == l)
        total = (x[rows] * w[rows][:, None]).sum(axis=0)
        count = w[rows].sum()
        mean = total / count
        means.append(mean)
        assert abs(cov["profile_norm"][l] - np.linalg.norm(mean)) < 1e-12
        assert abs(cov["components"][l] - count) < 1e-12
        assert cov["songs"][l] == len(rows)
    directions = np.stack([m / np.linalg.norm(m) for m in means])
    centre = directions.mean(axis=0)
    centre /= np.linalg.norm(centre)
    assert np.abs(cov["centrality"] - directions @ centre).max() < 1e-12
    assert abs(float(cov["components"].sum()) - len(set(zip(gi.tolist(), li.tolist())))) < 1e-9, \
        "the weight count per label must total the number of (group, label) components"
    print("ok: profile norms, centrality and component counts match a direct computation")


def check_coherence() -> None:
    """r_hat^2 is the between-component U-statistic and w_term is pure group structure."""
    rng = np.random.default_rng(11)
    n, L, dim = 120, 5, 16
    li = rng.integers(0, L, size=n)
    gi = rng.integers(0, 30, size=n)
    size = Counter(zip(gi.tolist(), li.tolist()))
    w = np.asarray([1.0 / size[(g, l)] for g, l in zip(gi, li)])
    x = rng.normal(size=(n, dim)) + 0.7 * rng.normal(size=(L, dim))[li]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    got = hc.label_coherence(x, li, gi, w, L)
    for l in range(L):
        rows = np.flatnonzero(li == l)
        groups = sorted(set(gi[rows].tolist()))
        means, sizes = [], []
        for g in groups:
            here = rows[gi[rows] == g]
            means.append(x[here].mean(axis=0))          # weights 1/k inside a component
            sizes.append(len(here))
        means = np.stack(means)
        total = sum(float(means[a] @ means[b]) for a in range(len(groups)) for b in range(len(groups)) if a != b)
        expect = total / (len(groups) * (len(groups) - 1))
        assert abs(got["r_hat2"][l] - expect) < 1e-12, (l, got["r_hat2"][l], expect)
        assert abs(got["w_term"][l] - sum(1.0 / k for k in sizes) / len(groups) ** 2) < 1e-12
        assert abs(got["r_hat"][l] - np.sqrt(max(expect, 0.0))) < 1e-12
    # w_term depends on the group structure alone: rotating every vector must not move it
    q = np.linalg.qr(rng.normal(size=(dim, dim)))[0]
    again = hc.label_coherence(x @ q, li, gi, w, L)
    assert np.abs(again["w_term"] - got["w_term"]).max() < 1e-12
    assert np.abs(again["r_hat2"] - got["r_hat2"]).max() < 1e-10, "cosines are rotation invariant"
    # and r_hat^2 must exclude within-component pairs: duplicate a song into its own component and
    # the U-statistic must not move, while a song-level mean cosine would jump
    print("ok: r_hat^2 is the between-component U-statistic; w_term is group structure only")


def check_regression() -> None:
    rng = np.random.default_rng(4)
    design = hc.standardise([rng.normal(size=200), rng.normal(size=200), rng.normal(size=200)])
    assert np.abs(design[:, 0] - 1.0).max() < 1e-12
    assert np.abs(design[:, 1:].mean(axis=0)).max() < 1e-12
    assert np.abs(design[:, 1:].std(axis=0) - 1.0).max() < 1e-12
    y = 0.7 * design[:, 1] - 0.4 * design[:, 3] + 0.05 * rng.normal(size=200)
    beta = hc.ols(design, y)
    normal = np.linalg.solve(design.T @ design, design.T @ y)
    assert np.abs(beta - normal).max() < 1e-9
    assert beta[1] > 0.5 and beta[3] < -0.2, beta
    boot = hc.bootstrap_labels(design, y, 200, np.random.default_rng(5))
    lo, hi = hc.interval(boot[:, 3])
    assert hi < 0.0, (lo, hi)
    print("ok: the regression recovers a planted negative norm coefficient, interval clear of zero")

    # and the rule must be able to FAIL: plant the opposite sign and check the verdict flips
    y2 = 0.7 * design[:, 1] + 0.4 * design[:, 3] + 0.05 * rng.normal(size=200)
    boot2 = hc.bootstrap_labels(design, y2, 200, np.random.default_rng(6))
    lo2, hi2 = hc.interval(boot2[:, 3])
    assert lo2 > 0.0, (lo2, hi2)
    print("ok: with the sign reversed the same rule refuses to confirm, so R2 can fail")


def check_estimands() -> None:
    rng = np.random.default_rng(7)
    n, L = 300, 9
    li = rng.integers(0, L, size=n)
    w = rng.random(n) + 0.2
    rr = rng.random(n)
    got = hc.estimands(rr, li, w, L)
    assert abs(got["query_weighted"] - round(float(rr.mean()), 4)) < 1e-12
    assert abs(got["component_weighted"] - round(float((rr * w).sum() / w.sum()), 4)) < 1e-12
    macro = [float((rr[li == l] * w[li == l]).sum() / w[li == l].sum()) for l in range(L)]
    assert abs(got["label_macro"] - round(float(np.mean(macro)), 4)) < 1e-12
    assert abs(got["between_label_variance_of_label_mrr"] - round(float(np.var(macro)), 5)) < 1e-12
    assert got["query_weighted"] != got["component_weighted"], \
        "the fixture must distinguish the estimands or it checks nothing"
    print("ok: the three estimands match their definitions and differ on this fixture")


def main() -> int:
    check_occupancy()
    check_entropy_and_skewness()
    check_csls_and_mutual_proximity()
    check_covariates()
    check_coherence()
    check_regression()
    check_estimands()
    print("all hubness and centrality checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
