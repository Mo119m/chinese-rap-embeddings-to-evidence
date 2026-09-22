"""Synthetic checks for src/label_specific_covariance_v3.py, without the corpus:

    python tests/test_label_specific_covariance.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import label_specific_covariance_v3 as lc  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(5)
    n, L, d = 240, 9, 30
    li = rng.integers(0, L, size=n)
    gi = rng.integers(0, 100, size=n)
    size = Counter(zip(gi.tolist(), li.tolist()))
    w = np.asarray([1.0 / size[(g, l)] for g, l in zip(gi, li)])
    # each label scatters along its own directions
    centres = rng.normal(size=(L, d))
    axes = rng.normal(size=(L, 3, d))
    z = centres[li] * 0.6 + np.einsum("nk,nkd->nd", rng.normal(size=(n, 3)) * 1.5, axes[li]) + 0.4 * rng.normal(size=(n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    train = rng.random(n) < 0.8
    s2 = lc.pooled_sigma2(z, li, w, train)
    queries = np.flatnonzero(~train)
    out = lc.score_fold(z, queries, li, gi, w, L, s2)
    worst = 0.0
    for row, qi in enumerate(queries.tolist()):
        for label in range(L):
            m = np.asarray([s for s in np.flatnonzero(li == label) if gi[s] != gi[qi]])
            for i, lam in enumerate(lc.LAMBDAS):
                ref = lc.explicit_score(z[qi], z[m], w[m], s2, lam)
                worst = max(worst, abs(out[i, row, label] - ref) / max(1.0, abs(ref)))
    assert worst <= 1e-8, worst
    print(f"ok: every (query, label, lambda) equals the explicit d x d computation with the query's group removed (relative gap {worst:.1e})")
    # lambda = 1 ranks labels by Euclidean distance to the weighted label mean
    for row, qi in enumerate(queries[:10].tolist()):
        dist = []
        for label in range(L):
            m = np.asarray([s for s in np.flatnonzero(li == label) if gi[s] != gi[qi]])
            mu = (z[m] * w[m][:, None]).sum(axis=0) / w[m].sum()
            dist.append(float(((z[qi] - mu) ** 2).sum()))
        assert np.argsort(-out[0, row]).tolist() == np.argsort(dist).tolist()
    print("ok: lambda = 1 ranks labels by Euclidean distance to the weighted label mean")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
