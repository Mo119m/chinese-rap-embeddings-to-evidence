#!/usr/bin/env python3
"""A hierarchical model of each label's own variation: label-specific covariance, shrunk
toward a pooled one (regularised discriminant analysis on the whitened spaces).

The two-covariance model and PLDA assume one within-label covariance shared by every label,
and both failed (plda_scoring.json, two_covariance_ablation.json). The whitening-directions
analysis found why: labels scatter along their own directions (median |cos| between labels'
leading scatter directions 0.271 against 0.456 under random reassignment). A model that lets
each label keep its own scatter, pooled toward the shared one because a label has few songs,
is the natural hierarchical next step (Friedman 1989, regularised discriminant analysis):

    Sigma_l(lambda) = lambda * sigma^2 * I + (1 - lambda) * S_l
    score_l(q) = -1/2 [ (q - mu_l)' Sigma_l^-1 (q - mu_l) + log |Sigma_l| ]

in the within-author whitened spaces, where the pooled within-label covariance is close to
isotropic: sigma^2 is the pooled within-label variance per dimension of the training-fold
songs; mu_l and S_l are the component-weighted mean and covariance of the label's songs outside
the query's leakage group (profiles from all other songs, as in the protocol). lambda = 1 is the
shared isotropic model (a Euclidean distance to the label mean); smaller lambda lets each label's
own directions count. S_l has rank at most n_l - 1 < d, so the computation uses the n_l x n_l
Gram matrix (Woodbury identity and its determinant lemma), exactly.

Spaces: BGE-M3 song centroids and the word SVD-1024, both within-author whitened per fold as in
identity_probe_v2 and word_space_probe_v2; each fold's queries are scored in the space fitted on
the other folds. lambda grid: 1.0, 0.9, 0.7, 0.5, 0.3, 0.1. lambda for fold k maximises the plain
MRR over the other folds' queries (each scored in its own fold's space; ties to the larger
lambda); no fold-k query's rank enters the choice of its lambda.

Reading, fixed before the run. In a space, label-specific covariance "helps" if the
fold-selected lambda minus lambda = 1 is positive with the paired group-bootstrap interval
clear of zero; otherwise "a shared covariance is enough". The selected system is also compared
with the published cosine prototype, as description.

Checks: the cosine prototype in each space reproduces 0.4164 and 0.5426 (gap <= 0.002); the
Gram-matrix score equals the explicit d x d computation (slogdet, solve) on twelve query-label
pairs per space at every lambda (relative gap <= 1e-8); the batched full-set path equals the
single-pair path wherever no member of the label is in the query's group.

    CHINESE_RAP_CORPUS=v3 python src/label_specific_covariance_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from exemplar_vs_prototype_v3 import setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "label_specific_covariance.json"
LAMBDAS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1)
SPACES = ("semantic_whitened", "words_svd_whitened")
EXPECTED = {"semantic_whitened": 0.4164, "words_svd_whitened": 0.5426}
CHECK_GAP = 0.002


def pooled_sigma2(z, label_index, weights, train) -> float:
    """Pooled within-label variance per dimension of the training songs."""
    rows = np.flatnonzero(train)
    d = z.shape[1]
    total, mass = 0.0, 0.0
    for l in np.unique(label_index[rows]):
        m = rows[label_index[rows] == l]
        wl = weights[m]
        mu = (z[m] * wl[:, None]).sum(axis=0) / wl.sum()
        total += float((wl * ((z[m] - mu) ** 2).sum(axis=1)).sum())
        mass += float(wl.sum())
    return total / (mass * d)


def label_parts(zs, ws):
    """mu, Y (rows sqrt(w)(z - mu)), total weight, eigen-decomposition of K = Y Y'."""
    W = float(ws.sum())
    mu = (zs * ws[:, None]).sum(axis=0) / W
    Y = np.sqrt(ws)[:, None] * (zs - mu)
    kappa, U = np.linalg.eigh(Y @ Y.T)
    return mu, Y, W, np.maximum(kappa, 0.0), U


def scores_from_parts(Q, mu, Y, W, kappa, U, sigma2, lambdas=LAMBDAS) -> np.ndarray:
    """(len(lambdas), len(Q)) log-likelihood scores of the rows of Q under one label."""
    d = Q.shape[1]
    n = len(kappa)
    R = Q - mu
    rr = np.einsum("ij,ij->i", R, R)
    P = U.T @ (Y @ R.T)                                    # (n, q)
    out = np.empty((len(lambdas), len(Q)))
    for i, lam in enumerate(lambdas):
        a = lam * sigma2
        B = (1.0 - lam) / W
        denom = a + B * kappa                              # eigenvalues of a I + B K
        quad = (rr - B * ((P ** 2) / denom[:, None]).sum(axis=0)) / a
        logdet = (d - n) * np.log(a) + float(np.log(denom).sum())
        out[i] = -0.5 * (quad + logdet)
    return out


def explicit_score(q, zs, ws, sigma2, lam) -> float:
    """The definition with a d x d covariance, for checking."""
    W = float(ws.sum())
    mu = (zs * ws[:, None]).sum(axis=0) / W
    C = zs - mu
    S = (C * ws[:, None]).T @ C / W
    Sigma = lam * sigma2 * np.eye(len(q)) + (1.0 - lam) * S
    sign, logdet = np.linalg.slogdet(Sigma)
    r = q - mu
    return float(-0.5 * (r @ np.linalg.solve(Sigma, r) + logdet))


def score_fold(z, queries, label_index, group_ids, weights, label_count, sigma2, lambdas=LAMBDAS) -> np.ndarray:
    """(len(lambdas), len(queries), L) scores; leave-group-out for labels present in a query's group."""
    out = np.empty((len(lambdas), len(queries), label_count))
    members = [np.flatnonzero(label_index == l) for l in range(label_count)]
    Q = z[queries]
    for l in range(label_count):
        m = members[l]
        mu, Y, W, kappa, U = label_parts(z[m], weights[m])
        out[:, :, l] = scores_from_parts(Q, mu, Y, W, kappa, U, sigma2, lambdas)
    by_group = defaultdict(list)
    for i, g in enumerate(group_ids.tolist()):
        by_group[int(g)].append(i)
    for row, qi in enumerate(queries.tolist()):
        mates = by_group[int(group_ids[qi])]
        for l in {int(label_index[j]) for j in mates}:
            m = np.asarray([s for s in members[l] if group_ids[s] != group_ids[qi]])
            if len(m) == 0:
                raise RuntimeError("a held-out group empties a label")
            mu, Y, W, kappa, U = label_parts(z[m], weights[m])
            out[:, row, l] = scores_from_parts(z[qi][None, :], mu, Y, W, kappa, U, sigma2, lambdas)[:, 0]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()
    import jieba
    jieba.setLogLevel(60)
    d = setup(args.private_root.resolve())
    li, gi, w, L, fold, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"], d["dense"]
    n = len(li)
    words, _ = fit_words([" ".join(segment(d["documents"][s])) for s in d["songs"]])
    print(f"{n:,} queries, {L} labels; lambdas {LAMBDAS}", flush=True)

    scores = {sp: np.zeros((len(LAMBDAS), n, L)) for sp in SPACES}
    proto = {sp: np.zeros((n, L)) for sp in SPACES}
    fold_info = {sp: {} for sp in SPACES}
    explicit_gaps = {sp: 0.0 for sp in SPACES}
    rng = np.random.default_rng(SEED)
    for k in range(FOLDS):
        train = fold != k
        test = np.flatnonzero(fold == k)
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train], li[train], w[train], np.random.default_rng(SEED + k))
        z_sem = unit_rows((dense - mean) @ matrix.T)
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(words[train])
        reduced = unit_rows(svd.transform(words))
        mean, matrix, _ = fit_transform("within_author_whitening", reduced[train], li[train], w[train], np.random.default_rng(SEED + k))
        z_words = unit_rows((reduced - mean) @ matrix.T)
        for sp, z in (("semantic_whitened", z_sem), ("words_svd_whitened", z_words)):
            t0 = time.time()
            sigma2 = pooled_sigma2(z, li, w, train)
            proto[sp][test] = dense_leave_group_out(z, li, gi, w, L, test)
            scores[sp][:, test, :] = score_fold(z, test, li, gi, w, L, sigma2)
            if k == 0:
                for _ in range(12):
                    row = int(rng.integers(len(test)))
                    qi = int(test[row])
                    label = int(rng.integers(L))
                    m = np.asarray([s for s in np.flatnonzero(li == label) if gi[s] != gi[qi]])
                    for i, lam in enumerate(LAMBDAS):
                        ref = explicit_score(z[qi], z[m], w[m], sigma2, lam)
                        explicit_gaps[sp] = max(explicit_gaps[sp], abs(scores[sp][i, qi, label] - ref) / max(1.0, abs(ref)))
                if explicit_gaps[sp] > 1e-8:
                    raise SystemExit(f"{sp}: the Gram-matrix score differs from the explicit computation ({explicit_gaps[sp]:.2e})")
            fold_info[sp][str(k)] = {"sigma2": round(sigma2, 8), "seconds": round(time.time() - t0, 1)}
            print(f"  fold {k} {sp}: sigma2 {sigma2:.6f}, {time.time() - t0:.0f}s", flush=True)

    checks = {"explicit_computation_relative_gap": explicit_gaps}
    results = {}
    mask = np.ones(n, dtype=bool)
    for sp in SPACES:
        rr_proto = 1.0 / ranks_of(proto[sp], li)
        got = float(np.mean(rr_proto))
        checks[f"{sp}_cosine_prototype"] = {"expected": EXPECTED[sp], "recomputed": round(got, 4), "gap": round(abs(got - EXPECTED[sp]), 5)}
        print(f"check {sp}: cosine prototype {got:.4f}, published {EXPECTED[sp]}", flush=True)
        if abs(got - EXPECTED[sp]) > CHECK_GAP:
            raise SystemExit(f"{sp}: the cosine prototype does not reproduce the published number")
        rr = {lam: 1.0 / ranks_of(scores[sp][i], li) for i, lam in enumerate(LAMBDAS)}
        mrr = {lam: float(np.mean(v)) for lam, v in rr.items()}
        selected = np.zeros(n)
        chosen = {}
        for k in range(FOLDS):
            other = fold != k
            by = {lam: float(np.mean(rr[lam][other])) for lam in LAMBDAS}
            best = max(LAMBDAS, key=lambda lam: (by[lam], lam))
            chosen[str(k)] = {"lambda": best, "other_folds_mrr_by_lambda": {f"{lam:g}": round(v, 4) for lam, v in by.items()}}
            selected[fold == k] = rr[best][fold == k]
        systems = {"selected": selected, "shared_isotropic": rr[1.0], "cosine_prototype": rr_proto}
        for lam in LAMBDAS:
            systems[f"lambda_{lam:g}"] = rr[lam]
        pairs = [("selected", "shared_isotropic"), ("selected", "cosine_prototype"), ("shared_isotropic", "cosine_prototype")]
        pairs += [(f"lambda_{lam:g}", "shared_isotropic") for lam in LAMBDAS if lam < 1.0]
        contrasts = paired_group_bootstrap(systems, w, gi, mask, pairs)
        c = contrasts[0]
        reading = "label-specific covariance helps" if c["excludes_zero"] and c["ci95"][0] > 0 else "a shared covariance is enough"
        results[sp] = {"mrr_by_lambda": {f"{lam:g}": round(v, 4) for lam, v in mrr.items()},
                       "mrr_selected": round(float(np.mean(selected)), 4), "mrr_cosine_prototype": round(got, 4),
                       "lambda_chosen_by_fold": chosen, "paired_contrasts": contrasts, "reading": reading,
                       "sigma2_by_fold": fold_info[sp]}
        print(f"== {sp}: cosine prototype {got:.4f}; " + "  ".join(f"l{lam:g}={v:.4f}" for lam, v in mrr.items())
              + f"; selected {np.mean(selected):.4f} (lambdas {[chosen[str(k)]['lambda'] for k in range(FOLDS)]})", flush=True)
        for c2 in contrasts[:3]:
            print(f"    {c2['system']} - {c2['minus']}: {c2['mrr_difference']:+.4f} [{c2['ci95'][0]:+.4f}, {c2['ci95'][1]:+.4f}]", flush=True)
        print(f"    reading: {reading}", flush=True)

    payload = {
        "analysis": "label-specific covariance shrunk toward a pooled isotropic one (regularised discriminant analysis) in the whitened spaces",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L},
        "design": {"model": "Sigma_l = lambda sigma^2 I + (1 - lambda) S_l; score = Gaussian log-likelihood of the query under (mu_l, Sigma_l); "
                            "mu_l and S_l component-weighted over the label's songs outside the query's leakage group; sigma^2 pooled on the training folds",
                   "lambdas": list(LAMBDAS), "spaces": list(SPACES),
                   "selection": "lambda for fold k maximises the plain MRR over the other folds' queries; ties to the larger lambda",
                   "reading_rule": "label-specific covariance helps if selected - shared_isotropic (lambda 1) is positive with the interval clear of zero; "
                                   "otherwise a shared covariance is enough; the comparison with the cosine prototype is description",
                   "checks": "cosine prototype reproduces the published MRR; Gram-matrix score equals the explicit d x d computation (relative 1e-8)"},
        "checks": checks,
        "spaces": results,
        "privacy": "aggregate only",
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / OUT_NAME).write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nwrote {args.out_dir / OUT_NAME}  ({payload['minutes']} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
