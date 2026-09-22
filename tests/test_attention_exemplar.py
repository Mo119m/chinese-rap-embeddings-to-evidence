"""Synthetic checks for src/attention_exemplar_v3.py, without the corpus:

    python tests/test_attention_exemplar.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import attention_exemplar_v3 as ae  # noqa: E402
import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from identity_probe_v2 import dense_leave_group_out  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(3)
    n, L = 260, 11
    li = rng.integers(0, L, size=n)
    gi = rng.integers(0, 110, size=n)
    size = Counter(zip(gi.tolist(), li.tolist()))
    w = np.asarray([1.0 / size[(g, l)] for g, l in zip(gi, li)])
    x = rng.normal(size=(n, 24)) + 0.8 * rng.normal(size=(L, 24))[li]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sets = ae.LabelSets(li, gi, w, L)
    q = np.arange(n)
    out = ae.attention_scores(x, q, gi, sets, ae.gram_matrices(x, sets), block=37)
    gap0 = float(np.abs(out[0.0] - dense_leave_group_out(x, li, gi, w, L, q)).max())
    assert gap0 <= 1e-9, gap0
    print(f"ok: beta 0 equals the protocol's dense scorer (gap {gap0:.1e})")
    worst = 0.0
    for b in ae.BETAS:
        for query in range(0, n, 23):
            for label in range(L):
                worst = max(worst, abs(out[b][query, label] - ae.brute_attention(x, query, gi, li, w, label, b)))
    assert worst <= 1e-9, worst
    print(f"ok: every beta equals the brute-force definition (gap {worst:.1e})")
    rows = sparse.random(n, 400, density=0.05, random_state=2, format="csr")
    rows.data = np.abs(rows.data)
    sx = normalize(rows, norm="l2", axis=1).tocsr().astype(np.float64)
    outs = ae.attention_scores(sx, q, gi, sets, ae.gram_matrices(sx, sets), betas=(0.0, 5.0))
    ref = v1.score_leave_group_out(x.astype(np.float32), sx.astype(np.float32), li, gi, L).lexical.astype(np.float64)
    gaps = float(np.abs(outs[0.0] - ref).max())
    assert gaps <= 1e-4, gaps
    worst = max(abs(outs[5.0][qq, l] - ae.brute_attention(sx, qq, gi, li, w, l, 5.0)) for qq in range(0, n, 29) for l in range(L))
    assert worst <= 1e-9, worst
    print(f"ok: sparse rows: beta 0 equals the protocol's sparse scorer (gap {gaps:.1e}); beta 5 equals the definition")
    # large beta approaches the nearest remaining song's cosine
    big = ae.attention_scores(x, q[:20], gi, sets, ae.gram_matrices(x, sets), betas=(2000.0,))[2000.0]
    far, cases = 0.0, 0
    for query in range(20):
        for label in range(L):
            rows_l = [s for s in np.flatnonzero(li == label) if gi[s] != gi[query]]
            cs = sorted((float(x[query] @ x[s]) for s in rows_l), reverse=True)
            if len(cs) > 1 and cs[0] - cs[1] > 0.01:        # no near-tie: the profile collapses onto one song
                far = max(far, abs(big[query, label] - cs[0]))
                cases += 1
    assert cases > 50 and far <= 1e-6, (cases, far)
    print(f"ok: without a near-tie, a large beta gives the nearest remaining song's cosine ({cases} cases, gap {far:.1e})")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
