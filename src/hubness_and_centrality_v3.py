#!/usr/bin/env python3
"""Which labels are answered too often, and what makes a label a hub?

QUESTION. In a fixed-corpus retrieval task some candidates are returned for far more queries than
their share, a geometry effect rather than an accuracy one (Radovanovic et al. 2010). Nothing in
this repository has measured it, yet the whole size argument turns on it: if a few labels absorb the
top of every ranking, then MRR is partly a statement about which labels are central and how many
songs they carry, not only about which representation holds identity.

THE PREDICTION THIS ARM TESTS, and it is a correction of an earlier draft. For the prototype the
score is q.u_l, a cosine with a direction, so a first reading says the profile norm cancels and
||m_l|| can have no effect. That reading is wrong. For any fixed direction v,

    E cos(u_l, v) ~ (mu_l . v) / E||m_l||,      E||m_l||^2 = r_l^2 + s^2 W_l,

so a single scalar kappa_l = r_l / E||m_l|| = (1 + s^2 W_l / r_l^2)^(-1/2) multiplies every one of
label l's scores, its own-label score and its scores against other labels alike. A label with few
songs has an inflated ||m_l||, hence a smaller kappa_l, hence a systematically deflated column --
which deflates its own-label score AND its chance of being a false hub. So the deflation of small
labels and the centrality bonus of large ones are the SAME scalar and run the SAME way in n, not
two biases of opposite sign, and ||m_l|| has a DIRECT negative association with hub occurrence
rather than no effect at all. That is the falsifiable content here.

WHAT IS MEASURED.
  * k-occurrence N_k(l): the number of queries for which label l appears in the top k of the
    ranking, at k = 1 and 10, and the FALSE k-occurrence over queries whose true label is not l.
    Its skewness over the 226 labels is the standard hubness statistic.
  * A query-side concentration, because a skewness over 226 points is a thin estimate: the
    normalised entropy of the top-1 mass over labels, bootstrapped over the 7,220 queries.
  * Per-label covariates: components and songs, the profile mean's norm ||m_l||, and centrality
    cos(m_l, g) with g the mean of the label mean directions. These three are collinear by
    construction -- averaging more unit vectors raises centrality and lowers the norm -- so their
    pairwise correlations are reported beside every coefficient, and no coefficient is read as an
    isolated effect.
  * Four de-hubbing read-outs on the same score matrix: Z-normalisation (the published one), CSLS
    and mutual proximity (the standard corrections), against the untransformed prototype. CSLS's
    label-side mean is computed on training-fold queries only, as Z-normalisation's cohort is.

READING RULES, fixed before the run.
 R1 (hubness is present). The skewness of N_10 over labels is positive with its label-bootstrap
    interval clear of zero in every space, and is larger in the raw spaces than in the whitened
    ones. A space whose interval includes zero has no measurable hubness and is reported that way.
 R2 (the corrected single-scalar prediction, confirmatory). In the regression of log(1 + N_10) on
    standardised (centrality, log components, ||m_l||) over the 226 labels, the coefficient on
    centrality is positive and the coefficient on ||m_l|| is NEGATIVE, both with label-bootstrap
    intervals clear of zero, in at least four of the five spaces. A non-negative ||m_l||
    coefficient refutes the single-scalar account and the paper must then say that the profile norm
    acts only through the own-label score.

    RUN 2026-09-23, AND REFUTED. The ||m_l|| coefficient is POSITIVE in all five spaces (+0.302
    semantic_raw, +0.075 semantic_whitened, +1.856 words_raw, +0.957 chars_raw, +0.213
    words_svd_whitened), every interval clear of zero. The diagnosis, and it is a mis-specification
    rather than a surprise about the corpus: ||m_l||^2 = r_l^2 + s^2 W_l, so at fixed component
    count the profile norm IS the resultant length. It measures how COHERENT a repertoire is, not
    how noisily its mean is estimated, and a coherent label sits near more queries. The noise
    channel runs through the component count, and the two cannot be separated by putting the norm
    and the count in one linear model: they correlate -0.92 to -0.93 in three of the five spaces.

 R2b (the identified re-specification, declared 2026-09-23 AFTER seeing R2 refuted, so it is a
    re-specification and not a confirmatory test). Replace the norm and the count with the two
    quantities they confound, which are algebraically distinct: r_hat_l, the component-level
    U-statistic (coherence), and W_l = sum_c(1/k_c)/G_l^2, a function of the group structure alone
    (estimation noise). Prediction, fixed before this regression runs: the coefficient on r_hat is
    POSITIVE and the coefficient on W_l is NEGATIVE, both intervals clear of zero, in at least four
    of five spaces. The first regression is kept and reported as the pre-registered one that failed.
 R3 (fairness against accuracy). At least one transform lowers query-weighted MRR while lowering
    BOTH the between-label variance of per-label MRR and the false-hub mass. If no transform
    produces that split, the estimand warning in the paper loses its evidence and becomes a
    theoretical remark; that is a real negative and is reported as one.
 R4 (the standard corrections against ours, equivalence). CSLS and mutual proximity reproduce
    Z-normalisation's query-weighted MRR to within 0.01 in the two raw lexical spaces. The margin
    is declared here; a difference whose whole interval lies inside +/-0.01 counts as reproduction,
    an interval reaching outside it does not, and an interval including zero but wider than the
    margin is undecided rather than confirmation.

ESTIMANDS. k-occurrence is a count over queries and has no weighted analogue, so it is
query-defined throughout and said to be. Every MRR is reported query-weighted, component-weighted
and label-macro; every interval on an MRR difference comes from paired_group_bootstrap and is
therefore component-weighted, and is never printed beside a query-weighted point estimate. Band
claims use the BANDS of label_size_calibration_v3, which band by songs per label.

CHECKS. The prototype reproduces the published MRR in every space before anything else is read; the
top-k occupancy counts sum to k times the number of queries; the regression is reproduced by an
independent normal-equations solve. On the transforms: all three change which label wins a query --
that is what they are for, since a per-query monotone transform cannot move a rank at all -- so the
test asserts that each one reorders and that the query-side term alone does not, and the regression
test plants both signs of the norm coefficient to show that R2 can fail. Synthetic test:
tests/test_hubness_and_centrality.py.

    CHINESE_RAP_CORPUS=v3 python src/hubness_and_centrality_v3.py --private-root <root>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from exemplar_vs_prototype_v3 import setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from label_size_calibration_v3 import BANDS, lgo_parts, znorm  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "hubness_and_centrality.json"
SPACES = ("semantic_raw", "semantic_whitened", "words_raw", "chars_raw", "words_svd_whitened")
FOLDWISE = ("semantic_whitened", "words_svd_whitened")
EXPECTED = {"semantic_raw": 0.2997, "semantic_whitened": 0.4164, "words_raw": 0.4963,
            "chars_raw": 0.4266, "words_svd_whitened": 0.5426}
TOLERANCE = {s: (0.002 if s in FOLDWISE else 5e-4) for s in SPACES}
KS = (1, 10)
CSLS_K = 10
REPLICATES = 2000
MARGIN = 0.01
RAW_LEXICAL = ("words_raw", "chars_raw")


def skewness(a):
    a = np.asarray(a, dtype=np.float64)
    m = a.mean()
    sd = a.std()
    return float(np.mean((a - m) ** 3) / sd ** 3) if sd > 0 else 0.0


def occupancy(scores, label_index, ks=KS):
    """N_k(l) and the false N_k(l), by partial selection of each query's top k labels."""
    out = {}
    for k in ks:
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        counts = np.bincount(top.ravel(), minlength=scores.shape[1]).astype(np.float64)
        wrong = top != label_index[:, None]
        false = np.bincount(top[wrong].ravel(), minlength=scores.shape[1]).astype(np.float64)
        if int(counts.sum()) != k * scores.shape[0]:
            raise RuntimeError("the top-k occupancy does not account for every query")
        out[k] = (counts, false)
    return out


def normalised_entropy(counts):
    """Entropy of the top-1 mass over labels, divided by log L: 1 is flat, 0 is one label."""
    p = counts / counts.sum()
    live = p > 0
    return float(-(p[live] * np.log(p[live])).sum() / np.log(len(counts)))


def label_coherence(x, label_index, group_ids, weights, label_count):
    """Per label, the two quantities the profile norm confounds, separated.

    ||m_l||^2 = r_l^2 + s^2 W_l, so at fixed component count the profile norm IS the resultant
    length: it measures how coherent a repertoire is, not how noisily its mean is estimated. The
    two channels are separable, because they are algebraically distinct objects:

      r_hat_l^2  the component-level U-statistic, the weighted mean cosine between DISTINCT
                 components -- within-component pairs are near-duplicates by construction and are
                 what the leakage protocol quarantines, so they are excluded;
      W_l        sum_c (1/k_c) / G_l^2, a function of the group structure alone and of nothing else.
    """
    r_hat2 = np.zeros(label_count)
    w_term = np.zeros(label_count)
    for l in range(label_count):
        members = np.flatnonzero(label_index == l)
        comp = group_ids[members]
        order = np.unique(comp)
        g = len(order)
        if g < 2:
            raise RuntimeError("a label with one component")
        rows = x[members]
        gram = rows @ rows.T
        gram = np.asarray(gram.todense() if sparse.issparse(gram) else gram, dtype=np.float64)
        spread = np.zeros((g, len(members)))
        for j, c in enumerate(comp.tolist()):
            spread[int(np.searchsorted(order, c)), j] = weights[members][j]
        if np.abs(spread.sum(axis=1) - 1.0).max() > 1e-9:
            raise RuntimeError("a component's weights do not sum to one")
        means = spread @ gram @ spread.T
        r_hat2[l] = (float(means.sum()) - float(np.trace(means))) / (g * (g - 1))
        sizes = np.asarray([int(np.sum(comp == c)) for c in order], dtype=np.float64)
        w_term[l] = float(np.sum(1.0 / sizes)) / g ** 2
    return {"r_hat": np.sqrt(np.maximum(r_hat2, 0.0)), "r_hat2": r_hat2, "w_term": w_term}


def label_covariates(x, label_index, weights, label_count):
    """Per label: components, songs, the profile mean's norm, and centrality against the mean
    of the label mean directions. Whole-label, not leave-group-out: these describe the label."""
    n = len(label_index)
    member = sparse.csr_matrix((weights, (label_index, np.arange(n))), shape=(label_count, n))
    sums = member @ x
    counts = np.bincount(label_index, weights=weights, minlength=label_count)
    if sparse.issparse(sums):
        sums = np.asarray(sums.todense())
    means = sums / counts[:, None]
    norms = np.linalg.norm(means, axis=1)
    directions = means / norms[:, None]
    centre = directions.mean(axis=0)
    centre = centre / np.linalg.norm(centre)
    return {"profile_norm": norms, "centrality": directions @ centre,
            "components": counts, "songs": np.bincount(label_index, minlength=label_count).astype(float)}


def csls(scores, train_rows, k=CSLS_K):
    """2 S - r_label - r_query, the label side averaged over training-fold queries only."""
    part = scores[train_rows]
    kk = min(k, part.shape[0])
    r_label = np.sort(part, axis=0)[-kk:].mean(axis=0)
    kq = min(k, scores.shape[1])
    r_query = np.sort(scores, axis=1)[:, -kq:].mean(axis=1)
    return 2.0 * scores - r_label[None, :] - r_query[:, None]


def mutual_proximity(scores):
    """The empirical form: (1 - within-query rank share) x (1 - within-label rank share)."""
    nq, nl = scores.shape
    by_query = np.argsort(np.argsort(-scores, axis=1), axis=1) / nl
    by_label = np.argsort(np.argsort(-scores, axis=0), axis=0) / nq
    return (1.0 - by_query) * (1.0 - by_label)


def standardise(columns):
    m = np.stack(columns, axis=1).astype(np.float64)
    m = (m - m.mean(axis=0)) / m.std(axis=0)
    return np.hstack([np.ones((len(m), 1)), m])


def ols(design, y):
    return np.linalg.lstsq(design, y, rcond=None)[0]


def bootstrap_labels(design, y, replicates, rng):
    out = np.zeros((replicates, design.shape[1]))
    for r in range(replicates):
        pick = rng.integers(0, len(y), size=len(y))
        out[r] = ols(design[pick], y[pick])
    return out


def interval(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def per_label_mrr(rr, label_index, weights, label_count):
    num = np.bincount(label_index, weights=rr * weights, minlength=label_count)
    den = np.bincount(label_index, weights=weights, minlength=label_count)
    return num / np.maximum(den, 1e-300)


def estimands(rr, label_index, weights, label_count):
    macro = per_label_mrr(rr, label_index, weights, label_count)
    return {"query_weighted": round(float(np.mean(rr)), 4),
            "component_weighted": round(float(np.sum(rr * weights) / np.sum(weights)), 4),
            "label_macro": round(float(np.mean(macro)), 4),
            "between_label_variance_of_label_mrr": round(float(np.var(macro)), 5)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--spaces", nargs="*", default=list(SPACES))
    args = parser.parse_args()
    started = time.time()
    import jieba
    jieba.setLogLevel(60)
    d = setup(args.private_root.resolve())
    li, gi, w, L, fold, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"], d["dense"]
    n = len(li)
    everything = np.arange(n)
    rng = np.random.default_rng(SEED)
    print(f"{n:,} queries, {L} labels", flush=True)

    docs = [d["documents"][s] for s in d["songs"]]
    words, _ = fit_words([" ".join(segment(doc)) for doc in docs])
    words = words.astype(np.float64).tocsr()
    chars = v1.fit_tfidf(docs).astype(np.float64).tocsr()
    del docs

    spaces = {"semantic_raw": {k: dense for k in range(FOLDS)}, "words_raw": {k: words for k in range(FOLDS)},
              "chars_raw": {k: chars for k in range(FOLDS)}, "semantic_whitened": {}, "words_svd_whitened": {}}
    for k in range(FOLDS):
        train = fold != k
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train], li[train], w[train],
                                        np.random.default_rng(SEED + k))
        spaces["semantic_whitened"][k] = unit_rows((dense - mean) @ matrix.T)
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(words[train])
        reduced = unit_rows(svd.transform(words))
        mean, matrix, _ = fit_transform("within_author_whitening", reduced[train], li[train], w[train],
                                        np.random.default_rng(SEED + k))
        spaces["words_svd_whitened"][k] = unit_rows((reduced - mean) @ matrix.T)
        print(f"  fold {k} transforms fitted", flush=True)

    results, rr_all, checks = {}, {}, {}
    songs_per_label = np.bincount(li, minlength=L)

    for sp in args.spaces:
        t0 = time.time()
        proto = np.zeros((n, L))
        zed = np.zeros((n, L))
        cs = np.zeros((n, L))
        cov = None
        for k in range(FOLDS):
            x = spaces[sp][k]
            rows = everything[fold == k] if sp in FOLDWISE else everything
            dots, norm2, counts = lgo_parts(x, li, gi, w, L, rows)
            here = dots / np.sqrt(norm2)
            proto[rows] = here
            train = fold != k
            # Z-normalisation needs the whole matrix on the training rows, as the published arm does
            full_dots, full_norm2, _ = lgo_parts(x, li, gi, w, L, everything) if sp in FOLDWISE else (dots, norm2, counts)
            full_proto = full_dots / np.sqrt(full_norm2)
            zed[rows] = znorm(full_proto, li, train, rows)
            cs[rows] = csls(full_proto, np.flatnonzero(train))[rows]
            here_cov = {**label_covariates(x, li, w, L), **label_coherence(x, li, gi, w, L)}
            if cov is None:
                cov = {key: value / (FOLDS if sp in FOLDWISE else 1) for key, value in here_cov.items()}
            elif sp in FOLDWISE:
                for key in cov:
                    cov[key] = cov[key] + here_cov[key] / FOLDS
            if sp not in FOLDWISE:
                break
        mp = mutual_proximity(proto)

        got = float(np.mean(1.0 / ranks_of(proto, li)))
        checks[sp] = {"expected": EXPECTED[sp], "recomputed": round(got, 4), "gap": round(abs(got - EXPECTED[sp]), 5)}
        if abs(got - EXPECTED[sp]) > TOLERANCE[sp]:
            raise SystemExit(f"{sp}: the prototype does not reproduce the published MRR ({got:.4f})")

        systems = {"prototype": proto, "znorm": zed, "csls": cs, "mutual_proximity": mp}
        block = {"systems": {}, "hubness": {}}
        for label, scores in systems.items():
            rr = 1.0 / ranks_of(scores, li)
            rr_all[f"{sp}/{label}"] = rr
            occ = occupancy(scores, li)
            ent = normalised_entropy(occ[1][0])
            top1 = np.argmax(scores, axis=1)
            draws = np.zeros(REPLICATES)
            for r in range(REPLICATES):
                pick = rng.integers(0, n, size=n)
                draws[r] = normalised_entropy(np.bincount(top1[pick], minlength=L).astype(np.float64))
            sk = {}
            for k in KS:
                counts, false = occ[k]
                boot = np.asarray([skewness(counts[rng.integers(0, L, size=L)]) for _ in range(REPLICATES)])
                sk[f"n{k}"] = {"skewness": round(skewness(counts), 4), "ci95_label_bootstrap": interval(boot),
                              "false_skewness": round(skewness(false), 4),
                              "max_over_mean": round(float(counts.max() / counts.mean()), 3),
                              "false_mass": round(float(false.sum() / max(counts.sum(), 1)), 4)}
            block["systems"][label] = estimands(rr, li, w, L)
            block["hubness"][label] = {**sk,
                                       "top1_entropy_over_labels": round(ent, 4),
                                       "top1_entropy_ci95_query_bootstrap": interval(draws)}
        if cov is None:
            raise RuntimeError("covariates were never computed")
        y = np.log1p(occupancy(proto, li)[10][0])
        design = standardise([cov["centrality"], np.log(cov["components"]), cov["profile_norm"]])
        beta = ols(design, y)
        boot = bootstrap_labels(design, y, REPLICATES, rng)
        check = np.linalg.solve(design.T @ design, design.T @ y)
        if float(np.abs(beta - check).max()) > 1e-8:
            raise SystemExit(f"{sp}: the regression does not reproduce a normal-equations solve")
        no_centrality = standardise([np.log(cov["components"]), cov["profile_norm"]])
        beta_nc = ols(no_centrality, y)
        raw = np.stack([cov["centrality"], np.log(cov["components"]), cov["profile_norm"]], axis=1)
        # the identified re-specification: coherence and estimation noise are separate covariates
        split_design = standardise([cov["centrality"], cov["r_hat"], cov["w_term"]])
        split_beta = ols(split_design, y)
        split_boot = bootstrap_labels(split_design, y, REPLICATES, rng)
        split_raw = np.stack([cov["centrality"], cov["r_hat"], cov["w_term"]], axis=1)
        block["regression_identified"] = {
            "outcome": "log(1 + N_10) over the 226 labels, N_10 query-defined",
            "terms": ["intercept", "centrality", "r_hat", "w_term"],
            "coefficients": [round(float(b), 4) for b in split_beta],
            "ci95_label_bootstrap": [interval(split_boot[:, j]) for j in range(split_design.shape[1])],
            "covariate_correlations": {
                "centrality_vs_r_hat": round(float(np.corrcoef(split_raw[:, 0], split_raw[:, 1])[0, 1]), 3),
                "centrality_vs_w_term": round(float(np.corrcoef(split_raw[:, 0], split_raw[:, 2])[0, 1]), 3),
                "r_hat_vs_w_term": round(float(np.corrcoef(split_raw[:, 1], split_raw[:, 2])[0, 1]), 3)},
            "median_r_hat2": round(float(np.median(cov["r_hat2"])), 4),
            "non_positive_r_hat2_labels": int(np.sum(cov["r_hat2"] <= 0)),
            "note": ("r_hat is the component-level U-statistic (coherence) and w_term = sum_c(1/k_c)/G^2 "
                     "is a function of the group structure alone (estimation noise); unlike the profile "
                     "norm and the component count, these two are not the same variable twice")}
        block["regression"] = {
            "outcome": "log(1 + N_10) over the 226 labels, N_10 query-defined",
            "terms": ["intercept", "centrality", "log_components", "profile_norm"],
            "coefficients": [round(float(b), 4) for b in beta],
            "ci95_label_bootstrap": [interval(boot[:, j]) for j in range(design.shape[1])],
            "log_components_without_centrality": round(float(beta_nc[1]), 4),
            "covariate_correlations": {
                "centrality_vs_log_components": round(float(np.corrcoef(raw[:, 0], raw[:, 1])[0, 1]), 3),
                "centrality_vs_profile_norm": round(float(np.corrcoef(raw[:, 0], raw[:, 2])[0, 1]), 3),
                "log_components_vs_profile_norm": round(float(np.corrcoef(raw[:, 1], raw[:, 2])[0, 1]), 3)},
            "note": ("the three covariates are collinear by construction: averaging more unit vectors "
                     "raises centrality and lowers the norm, so no coefficient is an isolated effect")}
        block["profile_norm_by_band"] = {
            name: {"labels": int(np.sum((songs_per_label >= lo) & (songs_per_label <= hi))),
                   "median_profile_norm": round(float(np.median(cov["profile_norm"][
                       (songs_per_label >= lo) & (songs_per_label <= hi)])), 4),
                   "median_centrality": round(float(np.median(cov["centrality"][
                       (songs_per_label >= lo) & (songs_per_label <= hi)])), 4)}
            for name, lo, hi in BANDS if np.sum((songs_per_label >= lo) & (songs_per_label <= hi)) >= 10}
        block["minutes"] = round((time.time() - t0) / 60.0, 1)
        results[sp] = block
        print(f"== {sp}: N_10 skewness {block['hubness']['prototype']['n10']['skewness']}, "
              f"top-1 entropy {block['hubness']['prototype']['top1_entropy_over_labels']}, "
              f"norm coefficient {block['regression']['coefficients'][3]}, "
              f"{block['minutes']} min", flush=True)

    done = list(results)
    mask = np.ones(n, dtype=bool)
    pairs = []
    for sp in done:
        pairs += [(f"{sp}/{t}", f"{sp}/prototype") for t in ("znorm", "csls", "mutual_proximity")]
        pairs += [(f"{sp}/csls", f"{sp}/znorm"), (f"{sp}/mutual_proximity", f"{sp}/znorm")]
    contrasts = paired_group_bootstrap(rr_all, w, gi, mask, pairs)
    for c in contrasts:
        results[c["system"].split("/")[0]].setdefault("paired_contrasts", []).append(c)

    readings = {}
    for sp in done:
        h = results[sp]["hubness"]["prototype"]["n10"]
        readings[f"{sp}: hubness present (R1)"] = (
            "present" if h["ci95_label_bootstrap"][0] > 0 else "not measurable")
        co = results[sp]["regression"]["coefficients"]
        ci = results[sp]["regression"]["ci95_label_bootstrap"]
        readings[f"{sp}: the single-scalar prediction (R2)"] = (
            "confirmed" if ci[1][0] > 0 and ci[3][1] < 0 else
            "centrality only" if ci[1][0] > 0 else
            "refuted: the profile norm has no negative direct association")
        ri = results[sp]["regression_identified"]["ci95_label_bootstrap"]
        readings[f"{sp}: coherence against estimation noise (R2b, a re-specification)"] = (
            "confirmed: coherence raises hub occurrence and estimation noise lowers it"
            if ri[2][0] > 0 and ri[3][1] < 0 else
            "coherence only: the noise term does not carry a negative association" if ri[2][0] > 0 else
            "noise only: the coherence term does not carry a positive association" if ri[3][1] < 0 else
            "refuted: neither term carries the predicted sign")
        splits = []
        base = results[sp]["systems"]["prototype"]
        for t in ("znorm", "csls", "mutual_proximity"):
            row = results[sp]["systems"][t]
            fair = (row["between_label_variance_of_label_mrr"] < base["between_label_variance_of_label_mrr"]
                    and results[sp]["hubness"][t]["n10"]["false_mass"] <
                    results[sp]["hubness"]["prototype"]["n10"]["false_mass"])
            if row["query_weighted"] < base["query_weighted"] and fair:
                splits.append(t)
        readings[f"{sp}: fairness against accuracy (R3)"] = (
            f"split under {', '.join(splits)}" if splits else "no transform splits fairness from accuracy")
        if sp in RAW_LEXICAL:
            verdicts = []
            for t in ("csls", "mutual_proximity"):
                c = next(x for x in results[sp]["paired_contrasts"]
                         if x["system"].endswith(t) and x["minus"].endswith("znorm"))
                lo, hi = c["ci95"]
                verdicts.append(f"{t}: " + ("reproduces" if lo > -MARGIN and hi < MARGIN
                                            else "differs" if lo > MARGIN or hi < -MARGIN else "undecided"))
            readings[f"{sp}: the standard corrections against Z-norm (R4)"] = "; ".join(verdicts)

    out = {"analysis": "hubness_and_centrality_v3",
           "question": "which labels are answered too often, and what makes a label a hub?",
           "corpus": {"content_sha256": V3_CONTENT_SHA256, "songs": n, "labels": L},
           "design": {"spaces": done, "ks": list(KS), "csls_k": CSLS_K, "replicates": REPLICATES,
                      "margin": MARGIN,
                      "k_occurrence": "a count over queries; it has no component-weighted analogue",
                      "bands": "songs per label, as label_size_calibration_v3",
                      "covariates": ("whole-label profile mean: its norm, and its cosine with the mean of "
                                     "the label mean directions; in a fold-wise space they are averaged "
                                     "over the five folds' spaces, while the outcome scores each query in "
                                     "its own fold's space"),
                      "csls_cohort": "the label side averaged over training-fold queries only",
                      "reading_rules": {
                          "R1": "N_10 skewness positive with the label-bootstrap interval clear of zero",
                          "R2": ("in log(1 + N_10) ~ centrality + log components + profile norm, centrality "
                                 "positive and the profile norm NEGATIVE, both intervals clear of zero, in "
                                 "at least four of five spaces; a non-negative norm coefficient refutes the "
                                 "single-scalar account. RUN 2026-09-23 AND REFUTED: the norm coefficient is "
                                 "positive in all five spaces, because at fixed component count the profile "
                                 "norm is the resultant length and measures coherence, not estimation noise"),
                          "R2b": ("declared after R2 was refuted, so a re-specification and not a "
                                  "confirmatory test: in log(1 + N_10) ~ centrality + r_hat + w_term, the "
                                  "coefficient on r_hat (coherence) is POSITIVE and on w_term = "
                                  "sum_c(1/k_c)/G^2 (estimation noise) is NEGATIVE, both intervals clear of "
                                  "zero, in at least four of five spaces"),
                          "R3": ("at least one transform lowers query-weighted MRR while lowering both the "
                                 "between-label variance of per-label MRR and the false-hub mass"),
                          "R4": ("CSLS and mutual proximity reproduce Z-normalisation to within 0.01 in the "
                                 "two raw lexical spaces; an interval reaching outside the margin does not "
                                 "count, and one including zero but wider than the margin is undecided")}},
           "spaces": results, "readings": readings, "checks": checks,
           "minutes": round((time.time() - started) / 60.0, 1),
           "privacy": ("aggregate only: skewness, entropies, regression coefficients, band medians and "
                       "quantiles over labels; no per-label array and no song, label or artist is named")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / OUT_NAME
    path.write_bytes((json.dumps(out, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    for key, value in readings.items():
        print(f"  {key}: {value}", flush=True)
    print(f"wrote {path} in {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
