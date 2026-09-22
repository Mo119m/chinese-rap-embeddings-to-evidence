"""Synthetic checks for src/temperature_scale_v3.py, without the corpus:

    python tests/test_temperature_scale.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import attention_exemplar_v3 as ae  # noqa: E402
import temperature_scale_v3 as ts  # noqa: E402
from identity_probe_v2 import dense_leave_group_out  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(11)
    n, L = 280, 12
    li = rng.integers(0, L, size=n)
    gi = rng.integers(0, 120, size=n)
    size = Counter(zip(gi.tolist(), li.tolist()))
    w = np.asarray([1.0 / size[(g, l)] for g, l in zip(gi, li)])
    x = rng.normal(size=(n, 26)) + 0.8 * rng.normal(size=(L, 26))[li]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sets = ae.LabelSets(li, gi, w, L)
    sets.label_of_query = li
    q = np.arange(n)
    grams = ae.gram_matrices(x, sets)
    fixed, pair, perq, own, pool, sample = ts.sweep(x, q, gi, sets, grams, beta_fold=5.0, block=31)

    gap0 = float(np.abs(fixed[0.0] - dense_leave_group_out(x, li, gi, w, L, q)).max())
    assert gap0 <= 1e-9, gap0
    print(f"ok: beta 0 equals the protocol's dense scorer (gap {gap0:.1e})")

    gapc = float(np.abs(pair[0.0] - fixed[0.0]).max())
    assert gapc <= 1e-12, gapc
    gapq = float(np.abs(perq[0.0] - fixed[0.0]).max())
    assert gapq <= 1e-12, gapq
    print(f"ok: c = 0 is the prototype under both rules (gap {max(gapc, gapq):.1e})")

    worst = worst_sigma = 0.0
    for const in (0.5, 1.0, 2.0):
        for query in range(0, n, 29):
            for label in range(L):
                ref, sd = ts.brute_pair(x, query, gi, li, w, label, const)
                worst = max(worst, abs(pair[const][query, label] - ref))
                if label == li[query]:
                    worst_sigma = max(worst_sigma, abs(own["sigma"][query] - sd))
    assert worst <= 1e-9, worst
    assert worst_sigma <= 1e-12, worst_sigma
    print(f"ok: the per-pair scorer equals its definition (gap {worst:.1e}); sigma matches ({worst_sigma:.1e})")

    # a per-pair temperature with a constant scale must equal the fixed-beta scorer at that beta
    flat = np.full(n, 0.3)
    one = np.zeros((n, L))
    for start in range(0, n, 31):
        blk = q[start:start + 31]
        cos = x[blk] @ x.T
        for l in range(L):
            m, ww, g, G = sets.members[l], sets.w[l], sets.g[l], grams[l]
            c = cos[:, m]
            keep = gi[blk][:, None] != g[None, :]
            e = ts.attended(c, ww, keep, 1.0 / flat[start:start + len(blk)])
            one[start:start + len(blk), l] = (e * c).sum(axis=1) / np.sqrt(np.maximum(((e @ G) * e).sum(axis=1), 1e-300))
    ref = ae.attention_scores(x, q, gi, sets, grams, betas=(1.0 / 0.3,), block=31)[1.0 / 0.3]
    gapf = float(np.abs(one - ref).max())
    assert gapf <= 1e-9, gapf
    print(f"ok: a constant scale reduces to the fixed-temperature scorer (gap {gapf:.1e})")

    # the window statistics: beta = 0 uses every song, a large beta uses one
    e0 = ts.attended(np.zeros((4, 7)), np.ones(7), np.ones((4, 7), dtype=bool), np.zeros(4))
    ess0, top0 = ts.window_stats(e0)
    assert np.allclose(ess0, 7.0) and np.allclose(top0, 1 / 7), (ess0, top0)
    c = np.tile(np.array([0.9, 0.1, 0.1, 0.1]), (3, 1))
    e1 = ts.attended(c, np.ones(4), np.ones((3, 4), dtype=bool), np.full(3, 500.0))
    ess1, top1 = ts.window_stats(e1)
    assert np.allclose(ess1, 1.0, atol=1e-6) and np.allclose(top1, 1.0, atol=1e-6), (ess1, top1)
    print("ok: the window is the whole set at beta 0 and one song at a large beta")

    # a weighted sd against an independent computation
    cc = rng.normal(size=(5, 9))
    ww = rng.random(9) + 0.1
    keep = rng.random((5, 9)) < 0.8
    keep[:, 0] = True
    mu, sd = ts.weighted_moments(cc, ww, keep)
    for r in range(5):
        k = keep[r]
        m_ref = np.average(cc[r, k], weights=ww[k])
        s_ref = np.sqrt(np.average((cc[r, k] - m_ref) ** 2, weights=ww[k]))
        assert abs(mu[r] - m_ref) < 1e-12 and abs(sd[r] - s_ref) < 1e-12
    print("ok: the weighted moments match numpy on masked rows")

    assert own["n"].min() >= 1 and pool.min() > 0 and len(sample) >= 0
    print("all temperature-scale checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
