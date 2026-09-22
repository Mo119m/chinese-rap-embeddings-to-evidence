#!/usr/bin/env python3
"""Is the attention temperature a reciprocal cosine scale?

attention_exemplar_v3 swept a fixed grid of temperatures beta in

    a_s(beta) = w_s * exp(beta * cos(q, x_s)),   score_l(q) = cos(q, sum_s a_s x_s),

and the temperature that won differs by a factor of twenty across the five spaces (words
SVD 1-2, characters 10, raw words 20, raw semantic 20). A temperature is not a free number:
beta multiplies a cosine, so beta carries the units of one over a cosine. If that is all the
variation is, then the dimensionless product

    c = beta * sigma(q, l),    sigma(q, l) = the weighted sd of the query's cosines to label l's songs,

should be far more nearly constant across spaces than beta itself, and a scorer that sets
beta(q, l) = c / sigma(q, l) with a single global c should do what the per-space tuned grid
does with no per-space tuning at all.

This script measures the scale and tests that scorer. It also records, for the query's own
label at the space's fold-selected beta, how many songs the attention weights actually use
(Shannon effective sample size) and the largest single weight -- the "recognition window",
which was asserted in an earlier draft and is here measured instead.

Spaces, folds, transforms, leave-group-out and component weights exactly as in
attention_exemplar_v3; beta = 0 must reproduce the published prototype before anything is read.

Reading rules, fixed before the run.

 1. Scale. The dimensionless reading is supported if the spread of beta* * sigma_tilde across
    the five spaces (max over min, sigma_tilde the median within-label cosine sd of the space)
    is below 2, against a spread of 20 for beta* alone. Between 2 and 5 is "reduced but not
    constant"; above 5 the reading fails.
 2. One constant. The one-constant scorer uses a single c for every space, chosen on the other
    folds' queries pooled over all five spaces. In a space it is non-inferior to the per-space
    tuned fixed beta if the whole 95% interval of (one-constant - tuned) lies above -0.005,
    the margin declared here. Beating the tuned beta is a separate, stronger outcome and is
    reported as such. The per-pair form beta(q, l) = c / sigma(q, l) is primary; the per-query
    form beta(q) = c / sigma_pool(q), which uses only the candidate pool and not the label, is
    the secondary arm.
 3. The window. Effective sample size and top weight are descriptive. They are reported as
    distributions over queries and by band, never as a per-label list.

Checks: c = 0 equals the protocol's prototype in every space (the same tolerances as
attention_exemplar_v3); the vectorised per-pair scorer equals a brute-force loop on six
queries in every space (1e-9); sigma(q, l) computed in the block loop equals a direct
computation on the same six.

    CHINESE_RAP_CORPUS=v3 python src/temperature_scale_v3.py --private-root <ni-k>
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
from attention_exemplar_v3 import LabelSets, gram_matrices  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from exemplar_vs_prototype_v3 import setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "temperature_scale.json"
SPACES = ("semantic_raw", "semantic_whitened", "words_raw", "chars_raw", "words_svd_whitened")
FOLDWISE = ("semantic_whitened", "words_svd_whitened")
EXPECTED = {"semantic_raw": 0.2997, "semantic_whitened": 0.4164, "words_raw": 0.4963, "chars_raw": 0.4266,
            "words_svd_whitened": 0.5426}
TOLERANCE = {s: (0.002 if s in FOLDWISE else 5e-4) for s in SPACES}
# the temperature attention_exemplar_v3 selected per fold, read from its published output
SELECTED_BETA = {"semantic_raw": (20.0,) * 5, "semantic_whitened": (5.0, 5.0, 5.0, 5.0, 10.0),
                 "words_raw": (20.0,) * 5, "chars_raw": (10.0,) * 5,
                 "words_svd_whitened": (2.0, 2.0, 1.0, 1.0, 1.0)}
# the fixed grid attention_exemplar_v3 swept, so the tuned baseline is the same object
BETAS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
CONSTANTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
MARGIN = 0.005
BLOCK = 400
EPS = 1e-9


def weighted_moments(c, w, keep):
    """Weighted mean and sd of each row of c over the kept columns."""
    wk = np.where(keep, w[None, :], 0.0)
    tot = wk.sum(axis=1)
    mu = (wk * c).sum(axis=1) / tot
    var = (wk * (c - mu[:, None]) ** 2).sum(axis=1) / tot
    return mu, np.sqrt(np.maximum(var, 0.0))


def attended(c, w, keep, b):
    """Score and weights for one label's block, with b a per-row temperature."""
    masked = np.where(keep, c, -np.inf)
    top = masked.max(axis=1, keepdims=True)
    e = np.where(keep, np.exp(b[:, None] * np.minimum(c - top, 0.0)), 0.0) * w[None, :]
    return e


def window_stats(e):
    """Shannon effective sample size and top share of each row of the weights."""
    tot = e.sum(axis=1, keepdims=True)
    p = e / np.maximum(tot, 1e-300)
    logp = np.log(np.maximum(p, 1e-300))
    return np.exp(-(p * logp).sum(axis=1)), p.max(axis=1)


def brute_pair(x, query, group_ids, label_index, weights, label, c_const):
    """The per-pair scorer, literally, for one query and label."""
    rows = [s for s in np.flatnonzero(label_index == label) if group_ids[s] != group_ids[query]]
    xq = np.asarray(x[query].todense()).ravel() if sparse.issparse(x) else x[query]
    vecs = [np.asarray(x[s].todense()).ravel() if sparse.issparse(x) else x[s] for s in rows]
    cs = np.asarray([float(xq @ v) for v in vecs])
    ws = weights[rows]
    mu = float(np.sum(ws * cs) / np.sum(ws))
    sd = float(np.sqrt(np.sum(ws * (cs - mu) ** 2) / np.sum(ws)))
    beta = c_const / max(sd, EPS)
    a = ws * np.exp(beta * (cs - cs.max()))
    v = np.sum([ai * vi for ai, vi in zip(a, vecs)], axis=0)
    return float(xq @ v / np.linalg.norm(v)), sd


def sweep(x, queries, group_ids, sets: LabelSets, grams, beta_fold, block=BLOCK):
    """One pass over the space.

    Returns per-query own-label scale and window statistics, the pooled candidate scale, and
    the score matrices for every fixed beta, every constant under the per-pair rule, and every
    constant under the per-query rule.
    """
    nq, L = len(queries), sets.L
    fixed = {b: np.zeros((nq, L)) for b in BETAS}
    per_pair = {c: np.zeros((nq, L)) for c in CONSTANTS}
    per_query = {c: np.zeros((nq, L)) for c in CONSTANTS}
    own = {k: np.zeros(nq) for k in ("sigma", "mu", "n", "ess", "top")}
    sigma_pool = np.zeros(nq)
    sigma_all = []  # the (query, label) scales, kept only as a pooled sample for quantiles
    rng = np.random.default_rng(SEED)
    for start in range(0, nq, block):
        q = queries[start:start + block]
        sl = slice(start, start + len(q))
        cos = x[q] @ x.T
        cos = np.asarray(cos.todense() if sparse.issparse(cos) else cos, dtype=np.float64)
        qg = group_ids[q]
        pool_keep = qg[:, None] != group_ids[None, :]
        pool = np.where(pool_keep, cos, np.nan)
        sigma_pool[sl] = np.nanstd(pool, axis=1)
        del pool
        for l in range(L):
            m, w, g, G = sets.members[l], sets.w[l], sets.g[l], grams[l]
            c = cos[:, m]
            keep = qg[:, None] != g[None, :]
            if not np.all(keep.any(axis=1)):
                raise RuntimeError("a held-out group empties a label")
            mu, sd = weighted_moments(c, w, keep)
            own_rows = sets.label_of_query[q] == l
            if own_rows.any():
                idx = np.flatnonzero(own_rows)
                own["sigma"][start + idx] = sd[idx]
                own["mu"][start + idx] = mu[idx]
                own["n"][start + idx] = keep[idx].sum(axis=1)
            take = rng.random(len(q)) < 0.02
            if take.any():
                sigma_all.append(sd[take])
            for b in BETAS:
                e = attended(c, w, keep, np.full(len(q), b))
                fixed[b][sl, l] = (e * c).sum(axis=1) / np.sqrt(np.maximum(((e @ G) * e).sum(axis=1), 1e-300))
            for const in CONSTANTS:
                b_pair = const / np.maximum(sd, EPS)
                e = attended(c, w, keep, b_pair)
                per_pair[const][sl, l] = (e * c).sum(axis=1) / np.sqrt(np.maximum(((e @ G) * e).sum(axis=1), 1e-300))
                b_q = const / np.maximum(sigma_pool[sl], EPS)
                e2 = attended(c, w, keep, b_q)
                per_query[const][sl, l] = (e2 * c).sum(axis=1) / np.sqrt(np.maximum(((e2 @ G) * e2).sum(axis=1), 1e-300))
            if own_rows.any():
                idx = np.flatnonzero(own_rows)
                e = attended(c[idx], w, keep[idx], np.full(len(idx), beta_fold))
                ess, top = window_stats(e)
                own["ess"][start + idx] = ess
                own["top"][start + idx] = top
    return fixed, per_pair, per_query, own, sigma_pool, np.concatenate(sigma_all) if sigma_all else np.zeros(0)


def quantiles(a, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    return {f"q{int(p * 100):02d}": round(float(np.quantile(a, p)), 4) for p in qs}


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
    sets = LabelSets(li, gi, w, L)
    sets.label_of_query = li
    print(f"{n:,} queries, {L} labels; constants {CONSTANTS}", flush=True)
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

    results, rr, checks = {}, {}, {}
    scale = {}
    for sp in args.spaces:
        t0 = time.time()
        fixed = {b: np.zeros((n, L)) for b in BETAS}
        pair = {c: np.zeros((n, L)) for c in CONSTANTS}
        perq = {c: np.zeros((n, L)) for c in CONSTANTS}
        own = {k: np.zeros(n) for k in ("sigma", "mu", "n", "ess", "top")}
        pool = np.zeros(n)
        sample = []
        for k in range(FOLDS):
            x = spaces[sp][k]
            queries = everything[fold == k] if sp in FOLDWISE else everything
            grams = gram_matrices(x, sets)
            f, p, pq, o, sg, sa = sweep(x, queries, gi, sets, grams, SELECTED_BETA[sp][k])
            for b in BETAS:
                fixed[b][queries] = f[b]
            for c in CONSTANTS:
                pair[c][queries] = p[c]
                perq[c][queries] = pq[c]
            for key in own:
                own[key][queries] = o[key]
            pool[queries] = sg
            sample.append(sa)
            if sp not in FOLDWISE:
                break
        sample = np.concatenate(sample)

        got = float(np.mean(1.0 / ranks_of(fixed[0.0], li)))
        checks[sp] = {"expected": EXPECTED[sp], "recomputed": round(got, 4), "gap": round(abs(got - EXPECTED[sp]), 5)}
        if abs(got - EXPECTED[sp]) > TOLERANCE[sp]:
            raise SystemExit(f"{sp}: beta = 0 does not reproduce the published prototype ({got:.4f})")
        zero_gap = float(np.abs(pair[0.0] - fixed[0.0]).max())
        if zero_gap > 1e-12:
            raise SystemExit(f"{sp}: c = 0 is not the prototype ({zero_gap:.2e})")
        x0 = spaces[sp][0]
        brute_gap = brute_sigma_gap = 0.0
        rng = np.random.default_rng(SEED)
        for _ in range(6):
            qi = int(rng.integers(n))
            if sp in FOLDWISE:
                x0 = spaces[sp][int(fold[qi])]
            lab = int(rng.integers(L))
            ref, sd_ref = brute_pair(x0, qi, gi, li, w, lab, 1.0)
            brute_gap = max(brute_gap, abs(pair[1.0][qi, lab] - ref))
            if lab == li[qi]:
                brute_sigma_gap = max(brute_sigma_gap, abs(own["sigma"][qi] - sd_ref))
        if brute_gap > 1e-9:
            raise SystemExit(f"{sp}: the per-pair scorer differs from its definition ({brute_gap:.2e})")
        checks[f"{sp}_brute_force"] = {"scorer": brute_gap, "sigma": brute_sigma_gap, "c_zero": zero_gap}

        for b in BETAS:
            rr[f"{sp}/beta_{b:g}"] = 1.0 / ranks_of(fixed[b], li)
        for c in CONSTANTS:
            rr[f"{sp}/pair_c{c:g}"] = 1.0 / ranks_of(pair[c], li)
            rr[f"{sp}/query_c{c:g}"] = 1.0 / ranks_of(perq[c], li)
        scale[sp] = {"own_label_sigma": quantiles(own["sigma"]), "own_label_mean_cosine": quantiles(own["mu"]),
                     "candidate_pool_sigma": quantiles(pool),
                     "all_pairs_sigma_2pc_sample": quantiles(sample),
                     "window_at_selected_beta": {"effective_sample_size": quantiles(own["ess"]),
                                                 "top_weight_share": quantiles(own["top"]),
                                                 "songs_in_profile": quantiles(own["n"]),
                                                 "beta_by_fold": list(SELECTED_BETA[sp])},
                     "minutes": round((time.time() - t0) / 60.0, 1)}
        print(f"== {sp}: own sigma median {scale[sp]['own_label_sigma']['q50']}, "
              f"pool sigma median {scale[sp]['candidate_pool_sigma']['q50']}, "
              f"ESS median {scale[sp]['window_at_selected_beta']['effective_sample_size']['q50']} "
              f"of {scale[sp]['window_at_selected_beta']['songs_in_profile']['q50']} songs, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        results[sp] = {"mrr_by_beta": {f"{b:g}": round(float(np.mean(rr[f'{sp}/beta_{b:g}'])), 4) for b in BETAS},
                       "mrr_by_constant_per_pair": {f"{c:g}": round(float(np.mean(rr[f'{sp}/pair_c{c:g}'])), 4) for c in CONSTANTS},
                       "mrr_by_constant_per_query": {f"{c:g}": round(float(np.mean(rr[f'{sp}/query_c{c:g}'])), 4) for c in CONSTANTS}}

    done = [sp for sp in args.spaces]
    # one constant for every space, chosen on the other folds' queries pooled over the spaces
    chosen = {}
    for k in range(FOLDS):
        other = fold != k
        by = {c: float(np.mean([np.mean(rr[f"{sp}/pair_c{c:g}"][other]) for sp in done])) for c in CONSTANTS if c > 0}
        best = max(by, key=lambda c: (by[c], -c))
        chosen[str(k)] = {"c": best, "other_folds_mean_mrr_over_spaces": {f"{c:g}": round(v, 4) for c, v in by.items()}}
    for sp in done:
        one = np.zeros(n)
        oneq = np.zeros(n)
        tuned = np.zeros(n)
        for k in range(FOLDS):
            here = fold == k
            one[here] = rr[f"{sp}/pair_c{chosen[str(k)]['c']:g}"][here]
            oneq[here] = rr[f"{sp}/query_c{chosen[str(k)]['c']:g}"][here]
            other = fold != k
            best_b = max([b for b in BETAS], key=lambda b: (float(np.mean(rr[f"{sp}/beta_{b:g}"][other])), -b))
            tuned[here] = rr[f"{sp}/beta_{best_b:g}"][here]
        rr[f"{sp}/one_constant_pair"] = one
        rr[f"{sp}/one_constant_query"] = oneq
        rr[f"{sp}/tuned_beta"] = tuned
        results[sp]["mrr_one_constant_per_pair"] = round(float(np.mean(one)), 4)
        results[sp]["mrr_one_constant_per_query"] = round(float(np.mean(oneq)), 4)
        results[sp]["mrr_tuned_beta"] = round(float(np.mean(tuned)), 4)

    mask = np.ones(n, dtype=bool)
    pairs = []
    for sp in done:
        pairs += [(f"{sp}/one_constant_pair", f"{sp}/tuned_beta"),
                  (f"{sp}/one_constant_pair", f"{sp}/beta_0"),
                  (f"{sp}/one_constant_query", f"{sp}/one_constant_pair")]
    contrasts = paired_group_bootstrap(rr, w, gi, mask, pairs)
    for c in contrasts:
        sp = c["system"].split("/")[0]
        results[sp].setdefault("paired_contrasts", []).append(c)
    for sp in done:
        c = next(x for x in results[sp]["paired_contrasts"] if x["minus"].endswith("tuned_beta"))
        results[sp]["non_inferior_to_tuned_beta"] = bool(c["ci95"][0] > -MARGIN)
        results[sp]["beats_tuned_beta"] = bool(c["ci95"][0] > 0.0)

    betastar = {sp: float(np.median(SELECTED_BETA[sp])) for sp in done}
    sig = {sp: scale[sp]["own_label_sigma"]["q50"] for sp in done}
    prod = {sp: round(betastar[sp] * sig[sp], 3) for sp in done}
    spread_beta = round(max(betastar.values()) / min(betastar.values()), 2)
    spread_prod = round(max(prod.values()) / min(prod.values()), 2)
    reading = ("dimensionless" if spread_prod < 2 else "reduced but not constant" if spread_prod < 5 else "fails")
    out = {"analysis": "temperature_scale_v3",
           "question": "is the attention temperature a reciprocal cosine scale?",
           "corpus": {"content_sha256": V3_CONTENT_SHA256, "songs": n, "labels": L},
           "design": {"spaces": done, "betas": list(BETAS), "constants": list(CONSTANTS), "margin": MARGIN,
                      "selected_beta_by_fold": {sp: list(SELECTED_BETA[sp]) for sp in done},
                      "sigma": "weighted sd of the query's cosines to the label's songs outside its leakage group",
                      "one_constant": "c chosen per fold on the other folds' queries, averaged over the spaces",
                      "reading_rule": ("the dimensionless reading holds if the spread of beta* * median sigma across "
                                       "the spaces is below 2 (reduced if below 5); the one-constant scorer is "
                                       "non-inferior in a space if the whole interval of (one constant - tuned beta) "
                                       "is above -0.005")},
           "scale": scale, "spaces": results, "constant_chosen_by_fold": chosen,
           "dimensionless": {"median_selected_beta": betastar, "median_own_label_sigma": sig,
                             "product": prod, "spread_of_beta": spread_beta, "spread_of_product": spread_prod,
                             "reading": reading},
           "checks": checks, "minutes": round((time.time() - started) / 60.0, 1),
           "privacy": "aggregate only: distributions over queries, no song, label or artist is identified"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / OUT_NAME
    path.write_bytes((json.dumps(out, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"\nbeta* spread {spread_beta}x, beta* * sigma spread {spread_prod}x -> {reading}", flush=True)
    print(f"wrote {path} in {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
