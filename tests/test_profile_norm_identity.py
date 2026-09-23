"""Synthetic checks for src/profile_norm_identity_v3.py, without the corpus:

    python tests/test_profile_norm_identity.py

Every estimator the script adds is checked against a brute-force or independent definition on
small random data: the component structure, the leave-group-out column sums (component counts,
K, and the sum of squared component norms), the per-label residual variance, the corrected
scorer and its exact factor-of-two identity at the floor, the correction multiplier, the
component-level U-statistic, the band tables with their suppression rule, and the three
estimands. The last two blocks verify the derivation the module docstring rests on: the gap of
the decomposition equals its closed form, and it is identically zero on unit rows whose
components are all singletons -- which is why the script tests the four empirical questions
instead of the decomposition.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import profile_norm_identity_v3 as pn  # noqa: E402
from label_size_calibration_v3 import FLOOR_SHARE, lgo_parts, residual_variance, scorers_from_parts  # noqa: E402


def draw(rng, n, labels, groups_per_label, dim, scale=0.8, nu=1.0):
    """Songs whose leakage groups sit inside a label, so components hold several songs.

    nu is the squared row norm: nu = 1 gives unit rows, anything else breaks the premise the
    per-song residual 1 - 2d + m2 assumes.
    """
    li = rng.integers(0, labels, size=n)
    gi = np.asarray([int(l) * groups_per_label + int(rng.integers(0, groups_per_label)) for l in li])
    size = Counter(zip(gi.tolist(), li.tolist()))
    w = np.asarray([1.0 / size[(int(g), int(l))] for g, l in zip(gi, li)])
    x = rng.normal(size=(n, dim)) + scale * rng.normal(size=(labels, dim))[li]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x *= np.sqrt(nu)
    return li, gi, w, x


def brute_label(x, li, gi, w, label):
    """One label's component geometry, written out: G, K, C_bar, S, the within-component mean
    cosines, and the component-weighted residual against the leave-one-component-out mean."""
    members = np.flatnonzero(li == label)
    groups = sorted({int(g) for g in gi[members].tolist()})
    count = len(groups)
    sizes = np.asarray([sum(1 for s in members if int(gi[s]) == g) for g in groups], dtype=float)
    xbar = np.stack([x[[s for s in members if int(gi[s]) == g]].mean(axis=0) for g in groups])
    total = xbar.sum(axis=0)
    sumsq = float(np.sum(xbar * xbar))
    c_bar = (float(total @ total) - sumsq) / (count * (count - 1))
    chat = []
    for j, g in enumerate(groups):
        own = [s for s in members if int(gi[s]) == g]
        if len(own) < 2:
            chat.append(0.0)
        else:
            chat.append(float(np.mean([x[s] @ x[t] for s in own for t in own if s != t])))
    resid = mass = 0.0
    for s in members:
        j = groups.index(int(gi[s]))
        mhat = (total - xbar[j]) / (count - 1)
        resid += w[s] * float(1.0 - 2.0 * (x[s] @ mhat) + mhat @ mhat)
        mass += w[s]
    return {"count": count, "kinv": float(np.sum(1.0 / sizes)), "c_bar": c_bar, "sumsq": sumsq,
            "sizes": sizes, "chat": np.asarray(chat), "s2": resid / mass,
            "nu": float(np.mean([np.sum(x[[s for s in members if int(gi[s]) == g]] ** 2)
                                 / sum(1 for s in members if int(gi[s]) == g) for g in groups]))}


def closed_form_gap(ref):
    """gap_l = [K(nu - 1) + sum_c (1 - 1/k_c)(chat_c - C_bar)] / (G(G - 1)), the docstring's line."""
    count, kinv = ref["count"], ref["kinv"]
    mass = 1.0 - 1.0 / ref["sizes"]
    return (kinv * (ref["nu"] - 1.0) + float(np.sum(mass * (ref["chat"] - ref["c_bar"])))) / (count * (count - 1))


def predicted(ref):
    """(1 - C_bar)(1 + K/(G(G-1))), the decomposition's right-hand side at the leave-one-out mean."""
    count, kinv = ref["count"], ref["kinv"]
    return (1.0 - ref["c_bar"]) * (1.0 + kinv / (count * (count - 1)))


def main() -> int:
    rng = np.random.default_rng(17)
    labels, groups_per_label, dim, n = 8, 9, 24, 200
    li, gi, w, x = draw(rng, n, labels, groups_per_label, dim)
    L = labels
    comp_of, comp_label, comp_group, comp_size = pn.components(li, gi)
    queries = np.arange(n)

    # the component structure, against a dictionary partition
    keys: dict[tuple[int, int], list[int]] = {}
    for i, (g, l) in enumerate(zip(gi.tolist(), li.tolist())):
        keys.setdefault((int(g), int(l)), []).append(i)
    assert len(comp_size) == len(keys)
    for (g, l), members in keys.items():
        ids = {int(comp_of[i]) for i in members}
        assert len(ids) == 1, (g, l, ids)
        c = ids.pop()
        assert int(comp_label[c]) == l and int(comp_group[c]) == g
        assert float(comp_size[c]) == len(members)
        for i in members:
            assert abs(w[i] - 1.0 / len(members)) < 1e-12
    assert int(np.sum(comp_size >= 2)) > 0, "the fixture must hold multi-song components"
    print(f"ok: {len(comp_size)} components, {int(np.sum(comp_size >= 2))} of them with several songs")

    # row squared norms and component squared norms, dense and sparse
    assert float(np.abs(pn.row_squared_norms(x) - 1.0).max()) < 1e-12
    sq = pn.component_sumsq(x, comp_of, w, len(comp_size))
    sq_sparse = pn.component_sumsq(sparse.csr_matrix(x), comp_of, w, len(comp_size))
    assert float(np.abs(sq - sq_sparse).max()) < 1e-12
    for c in range(len(comp_size)):
        members = np.flatnonzero(comp_of == c)
        mean = x[members].mean(axis=0)
        assert abs(sq[c] - float(mean @ mean)) < 1e-12, c
    print(f"ok: component squared norms match their means (dense and sparse agree to "
          f"{float(np.abs(sq - sq_sparse).max()):.1e})")

    # the leave-group-out column sums, against a brute-force loop and the protocol's own counts
    counts = pn.lgo_column_sums(np.ones(len(comp_size)), comp_label, comp_group, gi, L, queries)
    kinv = pn.lgo_column_sums(1.0 / comp_size, comp_label, comp_group, gi, L, queries)
    sumsq = pn.lgo_column_sums(sq, comp_label, comp_group, gi, L, queries)
    worst = [0.0, 0.0, 0.0]
    for q in range(0, n, 7):
        qg = int(gi[q])
        for l in range(L):
            kept = sorted({int(gi[s]) for s in np.flatnonzero(li == l)} - {qg})
            sizes = [sum(1 for s in np.flatnonzero(li == l) if int(gi[s]) == g) for g in kept]
            want_sumsq = 0.0
            for g in kept:
                own = [s for s in np.flatnonzero(li == l) if int(gi[s]) == g]
                mean = x[own].mean(axis=0)
                want_sumsq += float(mean @ mean)
            worst[0] = max(worst[0], abs(counts[q, l] - len(kept)))
            worst[1] = max(worst[1], abs(kinv[q, l] - sum(1.0 / v for v in sizes)))
            worst[2] = max(worst[2], abs(sumsq[q, l] - want_sumsq))
    assert max(worst) < 1e-12, worst
    dots, norm2, parts_counts = lgo_parts(x, li, gi, w, L, queries)
    gap_counts = float(np.abs(parts_counts - counts).max())
    assert gap_counts < 1e-12, gap_counts
    assert float((counts / kinv).min()) >= 1.0 - 1e-12
    assert float((counts / kinv).mean()) > 1.05, "the fixture must make the two W terms differ"
    print(f"ok: the column sums equal their definitions (worst {max(worst):.1e}) and the counts equal the "
          f"protocol's (gap {gap_counts:.1e}); published/weight-aware ratio mean {(counts / kinv).mean():.3f}")

    # the component-level U-statistic, against the weighted sum over distinct-component song pairs
    r2 = (norm2 - sumsq) / (parts_counts ** 2 - parts_counts)
    worst_r2 = 0.0
    for q in range(0, n, 11):
        qg = int(gi[q])
        for l in range(L):
            members = [s for s in np.flatnonzero(li == l) if int(gi[s]) != qg]
            g_count = len({int(gi[s]) for s in members})
            want = sum(w[s] * w[t] * float(x[s] @ x[t]) for s in members for t in members
                       if int(gi[s]) != int(gi[t])) / (g_count * g_count - g_count)
            worst_r2 = max(worst_r2, abs(r2[q, l] - want))
    assert worst_r2 < 1e-10, worst_r2
    print(f"ok: r_hat^2 equals the weight-aware U-statistic over distinct-component song pairs "
          f"(worst {worst_r2:.1e})")

    # the per-label residual variance, against the brute-force version and the pooled helper
    s2_label = pn.per_label_s2(dots, norm2, parts_counts, li, w, L)
    worst_s2 = 0.0
    for l in range(L):
        worst_s2 = max(worst_s2, abs(s2_label[l] - brute_label(x, li, gi, w, l)["s2"]))
    assert worst_s2 < 1e-10, worst_s2
    pooled_direct = float(np.sum(s2_label * np.bincount(li, weights=w, minlength=L)) / np.sum(w))
    pooled_ref = residual_variance(x, dots, norm2, parts_counts, li, w, np.ones(n, dtype=bool))
    assert abs(pooled_direct - pooled_ref) < 1e-12, (pooled_direct, pooled_ref)
    print(f"ok: the per-label s2 equals its definition (worst {worst_s2:.1e}) and pools to residual_variance")

    # the corrected scorer, against the published helper, its definition, and the multiplier form
    s2 = residual_variance(x, dots, norm2, parts_counts, li, w, rng.random(n) < 0.8)
    proto_ref, corrected_ref, floor_hits = scorers_from_parts(dots, norm2, parts_counts, s2)
    published, floored = pn.corrected_scores(dots, norm2, parts_counts, s2 / parts_counts)
    assert float(np.abs(published - corrected_ref).max()) == 0.0, "not bit-for-bit scorers_from_parts"
    assert int(floored.sum()) == int(floor_hits), (int(floored.sum()), int(floor_hits))
    m2 = norm2 / parts_counts ** 2
    worst_scorer = 0.0
    for q in range(0, n, 13):
        for l in range(L):
            want = (dots[q, l] / parts_counts[q, l]) / np.sqrt(max(m2[q, l] - s2 / parts_counts[q, l],
                                                                  FLOOR_SHARE * m2[q, l]))
            worst_scorer = max(worst_scorer, abs(published[q, l] - want))
            assert abs(proto_ref[q, l] - dots[q, l] / np.sqrt(norm2[q, l])) < 1e-12
    assert worst_scorer < 1e-12, worst_scorer
    factor = pn.multiplier((s2 / parts_counts) / m2)
    assert float(np.abs(published - proto_ref * factor).max()) < 1e-12
    assert float(factor.max()) <= 1.0 / np.sqrt(FLOOR_SHARE) + 1e-12
    # finding 0b: where the floor binds the corrected score is exactly twice the prototype.
    # The main fixture is too COHERENT for the floor to bind -- with scale 0.8 in 24 dimensions a
    # label's resultant length is about 0.6, while the corpus's lexical spaces sit near r^2 = 0.035,
    # i.e. r about 0.19. The floor binds when s^2/counts > 0.75 m2, so it needs a weak label and few
    # components, exactly the corner the published scorer caps. Draw that corner on purpose.
    weak_li, weak_gi, weak_w, weak_x = draw(rng, 120, 6, 4, 24, scale=0.12)
    weak_parts = lgo_parts(weak_x, weak_li, weak_gi, weak_w, 6, np.arange(120))
    weak_s2 = residual_variance(weak_x, *weak_parts, weak_li, weak_w, np.ones(120, dtype=bool))
    weak_published, weak_floored = pn.corrected_scores(*weak_parts, weak_s2 / weak_parts[2])
    weak_proto, weak_ref, weak_hits = scorers_from_parts(*weak_parts, weak_s2)
    assert int(weak_floored.sum()) == int(weak_hits) > 0, \
        f"the weak fixture must make the floor bind, got {int(weak_floored.sum())} hits"
    assert float(np.abs(weak_published - weak_ref).max()) == 0.0
    assert float(np.abs(weak_published[weak_floored] - 2.0 * weak_proto[weak_floored]).max()) < 1e-12
    print(f"ok: the floor binds on {int(weak_floored.sum()):,} of {weak_floored.size:,} weak-label "
          f"pairs, and there the corrected score is exactly twice the prototype")
    assert int(floored.sum()) == 0, \
        "the coherent fixture is expected not to floor; if it does, the two arms no longer contrast"
    weak_m2 = weak_parts[1] / weak_parts[2] ** 2
    weak_factor = pn.multiplier((weak_s2 / weak_parts[2]) / weak_m2)
    assert float(np.abs(weak_factor[weak_floored] - 2.0).max()) < 1e-12
    assert float(np.abs(weak_published - weak_proto * weak_factor).max()) < 1e-12
    print(f"ok: the corrected scorer is bit-for-bit scorers_from_parts on both fixtures, equals "
          f"prototype times the multiplier, and the multiplier is exactly 2 wherever the floor binds")

    # the weight-aware subtraction is never the larger one, and sometimes smaller
    aware, floored_aware = pn.corrected_scores(dots, norm2, parts_counts, s2 * kinv / parts_counts ** 2)
    assert float((s2 / parts_counts - s2 * kinv / parts_counts ** 2).min()) >= -1e-15
    assert float((s2 / parts_counts - s2 * kinv / parts_counts ** 2).max()) > 0.0
    assert int(floored_aware.sum()) <= int(floored.sum())
    print("ok: the published correction is never smaller than the weight-aware one and is sometimes larger")

    # the band tables, the banding guard and the suppression rule
    sizes = np.asarray([4, 5, 9, 10, 19, 20, 49, 50])
    assert list(pn.band_of_label(sizes, pn.COMPONENT_BANDS)) == ["4-9", "4-9", "4-9", "10-19", "10-19",
                                                                "20-49", "20-49", "50-up"]
    try:
        pn.band_of_label(np.asarray([3]), pn.COMPONENT_BANDS)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a size outside every band must raise")
    values = rng.normal(size=L)
    songs_per_label = np.bincount(li, minlength=L)
    fake = np.asarray(["5-9"] * (L - 3) + ["50-up"] * 3, dtype=object)
    table = pn.label_band_table(values, fake, songs_per_label)
    assert table["5-9"]["quantiles"] is not None and table["5-9"]["median"] is not None
    assert table["50-up"]["labels"] == 3 and table["50-up"]["quantiles"] is None
    assert table["50-up"]["median"] is None, "the median of three labels is one label's own value"
    assert abs(table["50-up"]["mean"] - round(float(np.mean(values[-3:])), 4)) < 1e-9
    assert table["10-19"]["labels"] == 0 and table["10-19"]["mean"] is None
    pair = pn.pair_band_table(counts / kinv, fake)
    assert pair["5-9"]["pairs"] == n * (L - 3) and pair["50-up"]["quantiles"] is None
    assert pair["50-up"]["median"] is None and pair["50-up"]["mean"] is not None
    assert abs(pair["5-9"]["mean"] - round(float((counts / kinv)[:, :L - 3].mean()), 4)) < 1e-9
    print("ok: the band tables match their definitions and suppress the median of a three-label band")

    # the three estimands, against explicit definitions
    rr = 1.0 / np.maximum(rng.integers(1, L + 1, size=n).astype(float), 1.0)
    mask = rng.random(n) < 0.6
    got = pn.estimands(rr, w, li, mask)
    assert abs(got["query_weighted"] - round(float(rr[mask].mean()), 4)) < 1e-9
    assert abs(got["component_weighted"]
               - round(float(np.sum(rr[mask] * w[mask]) / np.sum(w[mask])), 4)) < 1e-9
    present = sorted({int(v) for v in li[mask].tolist()})
    macro = float(np.mean([np.sum(rr[mask & (li == l)] * w[mask & (li == l)]) / np.sum(w[mask & (li == l)])
                           for l in present]))
    assert abs(got["label_macro"] - round(macro, 4)) < 1e-9, (got["label_macro"], macro)
    assert got["labels"] == len(present) and got["queries"] == int(mask.sum())
    assert pn.estimands(rr, w, li, np.zeros(n, dtype=bool)) is None
    print("ok: the three estimands equal their definitions and are labelled")

    # the gain/cost wording is gated on the interval, not on the point estimate
    gain = {"mrr_difference": 0.03, "ci95": [0.02, 0.04], "above_margin": True, "below_negative_margin": False}
    cost = {"mrr_difference": -0.03, "ci95": [-0.04, -0.02], "above_margin": False, "below_negative_margin": True}
    flat = {"mrr_difference": 0.004, "ci95": [-0.001, 0.009], "above_margin": False, "below_negative_margin": False}
    # the substring "gain" appears in the flat wording too ("neither a gain nor a cost is claimed"),
    # so the assertions test the CLAIMING phrases, not the bare words
    assert "is a gain of" in pn.change_sentence(gain, "x") and "is a cost of" not in pn.change_sentence(gain, "x")
    assert "is a cost of" in pn.change_sentence(cost, "x") and "is a gain of" not in pn.change_sentence(cost, "x")
    flat_sentence = pn.change_sentence(flat, "x")
    assert "is a gain of" not in flat_sentence and "is a cost of" not in flat_sentence
    assert "neither a gain nor a cost is claimed" in flat_sentence, flat_sentence
    print("ok: a gain is claimed only above the margin and a cost only below it")

    # the docstring's closed form for the gap, on unit rows and on rows of the wrong length
    for tag, nu in (("unit rows", 1.0), ("quarter-length rows", 0.25)):
        li2, gi2, w2, x2 = draw(np.random.default_rng(23), 900, 12, 18, 48, scale=0.9, nu=nu)
        d2, n2, c2 = lgo_parts(x2, li2, gi2, w2, 12, np.arange(900))
        measured = pn.per_label_s2(d2, n2, c2, li2, w2, 12)
        gap = np.asarray([measured[l] - predicted(brute_label(x2, li2, gi2, w2, l)) for l in range(12)])
        closed = np.asarray([closed_form_gap(brute_label(x2, li2, gi2, w2, l)) for l in range(12)])
        assert float(np.abs(gap - closed).max()) < 1e-10, (tag, float(np.abs(gap - closed).max()))
        print(f"ok: the gap equals its closed form on {tag} (worst {float(np.abs(gap - closed).max()):.1e}, "
              f"median gap {float(np.median(gap)):+.4f})")
        if nu != 1.0:
            # the reason the 0.02 margin was dropped: quartering every row -- the grossest possible
            # breach of the premise a unit-row check protects -- leaves the gap inside that margin
            assert float(np.median(gap)) < 0.0
            assert abs(float(np.median(gap))) < 0.02, float(np.median(gap))

    # and the tautology: on unit rows with singleton components the gap is identically zero
    for nu, expect_zero in ((1.0, True), (0.25, False)):
        li3 = np.repeat(np.arange(10), 24)
        gi3 = np.arange(240)                          # every song its own leakage group
        w3 = np.ones(240)
        x3 = np.random.default_rng(29).normal(size=(240, 32))
        x3 /= np.linalg.norm(x3, axis=1, keepdims=True)
        x3 *= np.sqrt(nu)
        d3, n3, c3 = lgo_parts(x3, li3, gi3, w3, 10, np.arange(240))
        measured = pn.per_label_s2(d3, n3, c3, li3, w3, 10)
        gap = np.asarray([measured[l] - predicted(brute_label(x3, li3, gi3, w3, l)) for l in range(10)])
        _, _, _, size3 = pn.components(li3, gi3)
        assert float(size3.max()) == 1.0
        bound = 2.0 * (24.0 - 24.0) / (24.0 * 23.0)
        assert bound == 0.0
        if expect_zero:
            assert float(np.abs(gap).max()) < 1e-12, float(np.abs(gap).max())
        else:
            # with singleton components the only surviving term is K(nu-1)/(G(G-1)) = (nu-1)/(G-1)
            assert float(np.abs(gap - (nu - 1.0) / 23.0).max()) < 1e-12, float(np.abs(gap).max())
            assert abs(float(np.median(gap))) < 0.04
    print("ok: with singleton components the gap is exactly zero on unit rows and exactly (nu-1)/(G-1) "
          "otherwise, so a margin test of the decomposition could not have failed")

    print("all profile-norm identity checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
