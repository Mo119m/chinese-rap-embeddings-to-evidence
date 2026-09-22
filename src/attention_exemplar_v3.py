#!/usr/bin/env python3
"""Between a centre and a cloud: query-dependent attention over a label's songs.

exemplar_vs_prototype_v3 found the prototype (the normalised weighted sum of a label's other
songs) far ahead of every fixed exemplar rule (nearest song, top three, mean cosine). Those
rules are the ends of a family; the question left is whether anything between them beats the
centre. A query attends to each remaining song s of label l with weight

    a_s(beta) = w_s * exp(beta * cos(q, x_s))

(w_s the protocol's per-(group, label) component weight) and the label is scored by cosine with
the attended profile v = sum_s a_s x_s:

    score_l(q) = sum_s a_s cos(q, x_s) / || sum_s a_s x_s ||.

beta = 0 gives exactly the protocol's prototype (checked before anything else); as beta grows
the profile collapses onto the query's nearest song of the label. This is the one-parameter,
unlearned form of attention pooling over a set (matching networks; late interaction), the
"exemplar with a learned temperature" end of what an attention aggregator can do.

Spaces: raw BGE-M3 song centroids; the same after within-author whitening fitted per fold;
raw jieba word TF-IDF; raw character 2-5-gram TF-IDF; the word SVD-1024 whitened per fold.
Leave-group-out throughout: the query's whole leakage group is removed from every label.
Folds and transforms as in identity_probe_v2; fold-wise spaces score each fold's queries in the
space fitted on the other folds.

beta grid: 0, 1, 2, 5, 10, 20, 50, 100. Selection: beta for fold k is the grid value with the
highest plain-mean MRR over the other four folds' queries, each scored as in the protocol (in
its own fold's space); no fold-k query's rank enters the choice of its beta (ties: the
smaller beta).

Reading, fixed before the run. In a space, attention "improves on the prototype" if the
fold-selected system minus the prototype (beta = 0) is positive with the paired group-bootstrap
interval clear of zero; otherwise "the prototype stands". The MRR at every beta and the beta
chosen per fold are description.

Checks: beta = 0 reproduces the published prototype in every space (gap <= 5e-4 raw, 0.002
fold-wise) and equals the protocol's scorer on 300 queries (dense 1e-9, sparse 1e-4 against
the float32 reference); the vectorised attention scorer equals a brute-force loop on six
queries at every beta in every space (1e-9).

    CHINESE_RAP_CORPUS=v3 python src/attention_exemplar_v3.py --private-root <ni-k>
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
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "attention_exemplar.json"
BETAS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
SPACES = ("semantic_raw", "semantic_whitened", "words_raw", "chars_raw", "words_svd_whitened")
FOLDWISE = ("semantic_whitened", "words_svd_whitened")
EXPECTED = {"semantic_raw": 0.2997, "semantic_whitened": 0.4164, "words_raw": 0.4963, "chars_raw": 0.4266,
            "words_svd_whitened": 0.5426}
TOLERANCE = {s: (0.002 if s in FOLDWISE else 5e-4) for s in SPACES}
BLOCK = 400


class LabelSets:
    """Per label: member song indices, their weights and group ids."""

    def __init__(self, label_index, group_ids, weights, label_count):
        self.members = [np.flatnonzero(label_index == l) for l in range(label_count)]
        self.w = [weights[m] for m in self.members]
        self.g = [group_ids[m] for m in self.members]
        self.L = label_count


def gram_matrices(x, sets: LabelSets):
    """Per label, the Gram matrix of its member rows (dense)."""
    out = []
    for m in sets.members:
        rows = x[m]
        g = rows @ rows.T
        out.append(np.asarray(g.todense() if sparse.issparse(g) else g, dtype=np.float64))
    return out


def attention_scores(x, queries, group_ids, sets: LabelSets, grams, betas=BETAS, block=BLOCK) -> dict:
    """scores[beta] (len(queries), L) for every beta, leave-group-out."""
    out = {b: np.zeros((len(queries), sets.L)) for b in betas}
    for start in range(0, len(queries), block):
        q = queries[start:start + block]
        cos = x[q] @ x.T
        cos = np.asarray(cos.todense() if sparse.issparse(cos) else cos, dtype=np.float64)
        qg = group_ids[q]
        for l in range(sets.L):
            m, w, g, G = sets.members[l], sets.w[l], sets.g[l], grams[l]
            c = cos[:, m]
            keep = qg[:, None] != g[None, :]
            if not np.all(keep.any(axis=1)):
                raise RuntimeError("a held-out group empties a label")
            masked = np.where(keep, c, -np.inf)
            top = masked.max(axis=1, keepdims=True)
            for b in betas:
                e = np.where(keep, np.exp(b * np.minimum(c - top, 0.0)), 0.0) * w[None, :]
                num = (e * c).sum(axis=1)
                den = np.sqrt(np.maximum(((e @ G) * e).sum(axis=1), 1e-300))
                out[b][start:start + len(q), l] = num / den
    return out


def brute_attention(x, query, group_ids, label_index, weights, label, beta) -> float:
    """The definition, literally, for one query and label."""
    rows = [s for s in np.flatnonzero(label_index == label) if group_ids[s] != group_ids[query]]
    xq = np.asarray(x[query].todense()).ravel() if sparse.issparse(x) else x[query]
    vecs = [np.asarray(x[s].todense()).ravel() if sparse.issparse(x) else x[s] for s in rows]
    cs = np.asarray([float(xq @ v) for v in vecs])
    a = weights[rows] * np.exp(beta * (cs - cs.max()))
    v = np.sum([ai * vi for ai, vi in zip(a, vecs)], axis=0)
    return float(xq @ v / np.linalg.norm(v))


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
    everything = np.arange(n)
    sets = LabelSets(li, gi, w, L)
    print(f"{n:,} queries, {L} labels; betas {BETAS}", flush=True)
    docs = [d["documents"][s] for s in d["songs"]]
    words, _ = fit_words([" ".join(segment(doc)) for doc in docs])
    words = words.astype(np.float64).tocsr()
    chars = v1.fit_tfidf(docs).astype(np.float64).tocsr()
    del docs

    # the spaces, as {fold: matrix}; raw spaces share one matrix
    spaces = {"semantic_raw": {k: dense for k in range(FOLDS)}, "words_raw": {k: words for k in range(FOLDS)},
              "chars_raw": {k: chars for k in range(FOLDS)}, "semantic_whitened": {}, "words_svd_whitened": {}}
    for k in range(FOLDS):
        train = fold != k
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train], li[train], w[train], np.random.default_rng(SEED + k))
        spaces["semantic_whitened"][k] = unit_rows((dense - mean) @ matrix.T)
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(words[train])
        reduced = unit_rows(svd.transform(words))
        mean, matrix, _ = fit_transform("within_author_whitening", reduced[train], li[train], w[train], np.random.default_rng(SEED + k))
        spaces["words_svd_whitened"][k] = unit_rows((reduced - mean) @ matrix.T)
        print(f"  fold {k} transforms fitted", flush=True)

    # check: beta = 0 equals the protocol's scorer
    probe = everything[:300]
    g_dense = gram_matrices(dense, sets)
    a0 = attention_scores(dense, probe, gi, sets, g_dense, betas=(0.0,))[0.0]
    gap_dense = float(np.abs(a0 - dense_leave_group_out(dense, li, gi, w, L, probe)).max())
    g_words = gram_matrices(words, sets)
    a0w = attention_scores(words, probe, gi, sets, g_words, betas=(0.0,))[0.0]
    ref = v1.score_leave_group_out(dense.astype(np.float32), words.astype(np.float32), li, gi, L).lexical[:300].astype(np.float64)
    gap_sparse = float(np.abs(a0w - ref).max())
    print(f"check: beta 0 against the protocol's scorer: dense gap {gap_dense:.2e}, sparse gap {gap_sparse:.2e}", flush=True)
    if gap_dense > 1e-9 or gap_sparse > 1e-4:
        raise SystemExit("beta = 0 is not the protocol's prototype")

    scores = {sp: {b: np.zeros((n, L)) for b in BETAS} for sp in SPACES}
    brute_gaps = {}
    for sp in SPACES:
        t0 = time.time()
        for k in range(FOLDS):
            x = spaces[sp][k]
            grams = g_dense if x is dense else g_words if x is words else gram_matrices(x, sets)
            if sp in FOLDWISE:
                test = np.flatnonzero(fold == k)
                part = attention_scores(x, test, gi, sets, grams)
                for b in BETAS:
                    scores[sp][b][test] = part[b]
                check_rows = test[:6]
                check_scores = {b: part[b][:6] for b in BETAS}
            else:
                full = attention_scores(x, everything, gi, sets, grams)
                for b in BETAS:
                    scores[sp][b][:] = full[b]
                check_rows = everything[:6]
                check_scores = {b: full[b][:6] for b in BETAS}
            if sp not in brute_gaps:
                gap = 0.0
                for b in BETAS:
                    for r, q in enumerate(check_rows):
                        for label in range(0, L, 25):
                            gap = max(gap, abs(check_scores[b][r, label] - brute_attention(x, int(q), gi, li, w, label, b)))
                brute_gaps[sp] = gap
                if gap > 1e-9:
                    raise SystemExit(f"{sp}: the vectorised attention scorer differs from its definition ({gap:.2e})")
            if sp not in FOLDWISE:
                break
        print(f"  {sp}: scored in {time.time() - t0:.0f}s; brute-force gap {brute_gaps[sp]:.1e}", flush=True)

    checks = {"beta_zero_against_protocol_scorer": {"dense": gap_dense, "sparse": gap_sparse}, "brute_force_gap": brute_gaps}
    results = {}
    all_mask = np.ones(n, dtype=bool)
    for sp in SPACES:
        rr = {b: 1.0 / ranks_of(scores[sp][b], li) for b in BETAS}
        mrr = {b: float(np.mean(rr[b])) for b in BETAS}
        gap = abs(mrr[0.0] - EXPECTED[sp])
        checks[f"{sp}_prototype"] = {"expected": EXPECTED[sp], "recomputed": round(mrr[0.0], 4), "gap": round(gap, 5)}
        print(f"check {sp}: beta 0 gives {mrr[0.0]:.4f}, published {EXPECTED[sp]}", flush=True)
        if gap > TOLERANCE[sp]:
            raise SystemExit(f"{sp}: beta = 0 does not reproduce the published prototype")
        selected = np.zeros(n)
        chosen = {}
        for k in range(FOLDS):
            other = fold != k
            by_beta = {b: float(np.mean(rr[b][other])) for b in BETAS}
            best = min(BETAS, key=lambda b: (-by_beta[b], b))
            chosen[str(k)] = {"beta": best, "other_folds_mrr_by_beta": {f"{b:g}": round(v, 4) for b, v in by_beta.items()}}
            test = fold == k
            selected[test] = rr[best][test]
        systems = {"prototype": rr[0.0], "attention_selected": selected}
        for b in BETAS:
            systems[f"beta_{b:g}"] = rr[b]
        pairs = [("attention_selected", "prototype")] + [(f"beta_{b:g}", "prototype") for b in BETAS if b > 0]
        contrasts = paired_group_bootstrap(systems, w, gi, all_mask, pairs)
        c = contrasts[0]
        reading = ("attention improves on the prototype" if c["excludes_zero"] and c["ci95"][0] > 0 else "the prototype stands")
        results[sp] = {"mrr_by_beta": {f"{b:g}": round(v, 4) for b, v in mrr.items()},
                       "mrr_attention_selected": round(float(np.mean(selected)), 4),
                       "beta_chosen_by_fold": chosen, "paired_contrasts": contrasts, "reading": reading}
        print(f"== {sp}: " + "  ".join(f"b{b:g}={v:.4f}" for b, v in mrr.items()) + f"  selected={np.mean(selected):.4f} "
              f"(betas {[chosen[str(k)]['beta'] for k in range(FOLDS)]})", flush=True)
        print(f"    attention_selected - prototype: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] -> {reading}", flush=True)

    payload = {
        "analysis": "query-dependent softmax attention over a label's songs, from the prototype (beta 0) toward the nearest song",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L},
        "design": {"score": "cosine of the query with sum_s a_s x_s, a_s = w_s exp(beta cos(q, x_s)), over the label's songs outside the query's leakage group",
                   "betas": list(BETAS), "spaces": list(SPACES),
                   "selection": "beta for fold k maximises the plain MRR over the other folds' queries, each scored as in the protocol; ties to the smaller beta",
                   "reading_rule": "attention improves on the prototype in a space if attention_selected - prototype is positive with the interval clear of zero; otherwise the prototype stands",
                   "checks": "beta 0 equals the protocol scorer (dense 1e-9, sparse 1e-4) and reproduces the published MRR; vectorised scorer equals a brute-force loop (1e-9)"},
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
