#!/usr/bin/env python3
"""Does the headline ordering hold for labels with few songs? Prototype scoring, its
calibration, and a hierarchical correction, read by label size.

exemplar_vs_prototype_v3 found that under the protocol's prototype scorer the raw semantic
space outranks the raw word space for queries whose label has fewer than twenty songs, while
under a plain mean cosine the word space leads in every size band. The prototype scores a
query by cosine with the label's normalised weighted SUM of songs, so a label's profile norm
divides its scores. A profile built from few songs keeps more sampling noise in its norm, and
a sparse TF-IDF profile far more than a dense one: a small label's scores are deflated, and
deflated most in the word space. Two scorers from the speaker-recognition back-end remove
that dependence on label size without adding a tuned parameter:

  prototype        the protocol: cosine with the normalised leave-group-out profile
  znorm            Z-normalisation (Auckenthaler et al. 2000): per fold and label, the
                   prototype scores of the training-fold songs of OTHER labels give an
                   impostor mean and standard deviation; a test query's score for the label
                   is standardised by them
  noise_corrected  the hierarchical (two-covariance) correction of the profile norm: the
                   mean of n unit songs has squared norm ||mu||^2 + s^2/n, where s^2 is the
                   within-label residual variance of one song, estimated on the training
                   folds; the prototype divides by sqrt(max(||m||^2 - s^2/n, ||m||^2 / 4))
                   instead of ||m||, the floor capping the inflation at two

Spaces: raw BGE-M3 song centroids; the same whitened within-author per fold; raw jieba
word TF-IDF; the word SVD-1024 whitened within-author per fold. Folds and transforms as in
identity_probe_v2; profiles are leave-group-out over all songs as in the protocol; the
cohort and s^2 use training-fold songs only.

Bands by songs per label: 5-9, 10-19, 20-49, 50 and more. The headline contrasts read by
band: words_raw - semantic_raw and words_svd_whitened - semantic_whitened, paired group
bootstrap within the band.

Reading, fixed before the run. Under a scorer, the word-over-semantic ordering is
"size-robust" if the contrast is positive with the interval clear of zero in every band
with at least ten labels and one hundred queries; otherwise "size-dependent", and the
bands where it fails are named. Each scorer's overall MRR is compared with the prototype's.

Checks: the prototype must reproduce the published numbers in every space (gap <= 0.002);
the generic leave-group-out scorer must equal the protocol's dense and sparse scorers.

    python src/label_size_calibration_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from exemplar_vs_prototype_v3 import EXPECTED, setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
SCORERS = ("prototype", "znorm", "noise_corrected")
BANDS = (("5-9", 5, 9), ("10-19", 10, 19), ("20-49", 20, 49), ("50-up", 50, 10**9))
MIN_LABELS, MIN_QUERIES = 10, 100
CHECK_GAP = 0.002
FLOOR_SHARE = 0.25
HEADLINE = (("words_raw", "semantic_raw"), ("words_svd_whitened", "semantic_whitened"))


def lgo_parts(x, label_index, group_ids, weights, label_count, queries):
    """Leave-group-out ingredients for every (query, label): the dot product with the label's
    weighted sum, that sum's squared norm and its weight count, after removing the query's
    leakage group. x is a dense unit-row array or a CSR matrix with unit rows."""
    n = len(label_index)
    member = sparse.csr_matrix((weights, (label_index, np.arange(n))), shape=(label_count, n))
    sums = member @ x                                            # (L, dim) dense or sparse
    if sparse.issparse(sums):
        sums = sums.tocsr()
        norm2 = np.asarray(sums.multiply(sums).sum(axis=1)).ravel()
        dots = np.asarray((x[queries] @ sums.T).todense(), dtype=np.float64)
    else:
        norm2 = np.einsum("ij,ij->i", sums, sums)
        dots = x[queries] @ sums.T
    counts = np.bincount(label_index, weights=weights, minlength=label_count)
    dots = np.array(dots, dtype=np.float64)
    norm2 = np.repeat(norm2[None, :], len(queries), axis=0)
    counts = np.repeat(counts[None, :], len(queries), axis=0)
    members_by_group: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids.tolist()):
        members_by_group[int(group)].append(index)
    for row, query in enumerate(queries.tolist()):
        members = members_by_group[int(group_ids[query])]
        for label in {int(label_index[m]) for m in members}:
            own = [m for m in members if int(label_index[m]) == label]
            own_w = weights[own]
            if sparse.issparse(x):
                own_vec = np.asarray((sparse.diags(own_w) @ x[own]).sum(axis=0)).ravel()
                s_row = np.asarray(sums[label].todense()).ravel()
                q_row = np.asarray(x[query].todense()).ravel()
            else:
                own_vec = (x[own] * own_w[:, None]).sum(axis=0)
                s_row = sums[label]
                q_row = x[query]
            leave = s_row - own_vec
            dots[row, label] = float(q_row @ leave)
            norm2[row, label] = float(leave @ leave)
            counts[row, label] = counts[row, label] - float(own_w.sum())
    if np.any(norm2 <= 1e-24) or np.any(counts <= 0):
        raise RuntimeError("a held-out group empties a label profile")
    return dots, norm2, counts


def residual_variance(x, dots_all, norm2_all, counts_all, label_index, weights, train):
    """Pooled within-label residual variance of one unit-norm song on the training folds,
    against its label's leave-group-out profile (the song and its group excluded)."""
    l = label_index[train]
    rows = np.flatnonzero(train)
    d = dots_all[rows, l] / counts_all[rows, l]
    m2 = norm2_all[rows, l] / counts_all[rows, l] ** 2
    resid = 1.0 - 2.0 * d + m2
    return float(np.sum(resid * weights[train]) / np.sum(weights[train]))


def scorers_from_parts(dots, norm2, counts, s2):
    prototype = dots / np.sqrt(norm2)
    m2 = norm2 / counts ** 2
    corrected = np.maximum(m2 - s2 / counts, FLOOR_SHARE * m2)
    floor_hits = int(np.sum(m2 - s2 / counts < FLOOR_SHARE * m2))
    noise_corrected = (dots / counts) / np.sqrt(corrected)
    return prototype, noise_corrected, floor_hits


def znorm(prototype_all, label_index, train, test):
    """Standardise each label's column by its impostor scores on the training-fold songs."""
    out = np.zeros((len(test), prototype_all.shape[1]))
    for label in range(prototype_all.shape[1]):
        cohort = train & (label_index != label)
        col = prototype_all[cohort, label]
        mu, sd = float(col.mean()), float(col.std())
        if sd <= 1e-12:
            raise RuntimeError("a degenerate impostor distribution")
        out[:, label] = (prototype_all[test, label] - mu) / sd
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()
    import jieba
    jieba.setLogLevel(60)
    d = setup(args.private_root.resolve())
    li, gi, w, L, fold = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"]
    n = len(li)
    everything = np.arange(n)
    songs_per_label = np.bincount(li, minlength=L)
    band_of_song = np.full(n, "", dtype=object)
    band_labels = {}
    for name, lo, hi in BANDS:
        labels_in = np.flatnonzero((songs_per_label >= lo) & (songs_per_label <= hi))
        band_labels[name] = int(len(labels_in))
        band_of_song[np.isin(li, labels_in)] = name
    print(f"{n:,} queries, {L} labels; labels per band {band_labels}", flush=True)

    # the spaces, per fold: (x, is_sparse)
    print("building spaces", flush=True)
    dense = d["dense"]
    lexical, _ = fit_words([" ".join(segment(d["documents"][s])) for s in d["songs"]])
    lexical = lexical.astype(np.float64).tocsr()
    spaces = {"semantic_raw": {k: dense for k in range(FOLDS)}, "words_raw": {k: lexical for k in range(FOLDS)},
              "semantic_whitened": {}, "words_svd_whitened": {}}
    for k in range(FOLDS):
        train = fold != k
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train], li[train], w[train], np.random.default_rng(SEED + k))
        spaces["semantic_whitened"][k] = unit_rows((dense - mean) @ matrix.T)
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(lexical[train])
        reduced = unit_rows(svd.transform(lexical))
        mean, matrix, _ = fit_transform("within_author_whitening", reduced[train], li[train], w[train], np.random.default_rng(SEED + k))
        spaces["words_svd_whitened"][k] = unit_rows((reduced - mean) @ matrix.T)
        print(f"  fold {k} transforms fitted", flush=True)

    # checks of the generic scorer against the protocol's two scorers
    dots, norm2, counts = lgo_parts(dense, li, gi, w, L, everything[:300])
    reference = dense_leave_group_out(dense, li, gi, w, L, everything[:300])
    gap_dense = float(np.abs(dots / np.sqrt(norm2) - reference).max())
    profiles = v1.score_leave_group_out(dense.astype(np.float32), lexical.astype(np.float32), li, gi, L)
    dots, norm2, counts = lgo_parts(lexical, li, gi, w, L, everything[:300])
    gap_sparse = float(np.abs(dots / np.sqrt(norm2) - profiles.lexical[:300].astype(np.float64)).max())
    print(f"generic scorer against the protocol: dense gap {gap_dense:.2e}, sparse gap {gap_sparse:.2e}", flush=True)
    if gap_dense > 1e-9 or gap_sparse > 1e-4:
        raise SystemExit("the generic leave-group-out scorer differs from the protocol's")

    scores = {sp: {s: np.zeros((n, L)) for s in SCORERS} for sp in spaces}
    fold_info = {sp: {} for sp in spaces}
    parts_cache = {}
    for sp, by_fold in spaces.items():
        print(f"scoring {sp}", flush=True)
        for k in range(FOLDS):
            x = by_fold[k]
            train, test = fold != k, np.flatnonzero(fold == k)
            if id(x) not in parts_cache:                        # raw spaces are the same in every fold
                parts_cache[id(x)] = lgo_parts(x, li, gi, w, L, everything)
            dots, norm2, counts = parts_cache[id(x)]
            s2 = residual_variance(x, dots, norm2, counts, li, w, train)
            prototype_all, corrected_all, floor_hits = scorers_from_parts(dots, norm2, counts, s2)
            scores[sp]["prototype"][test] = prototype_all[test]
            scores[sp]["noise_corrected"][test] = corrected_all[test]
            scores[sp]["znorm"][test] = znorm(prototype_all, li, train, test)
            fold_info[sp][str(k)] = {"s2": round(s2, 4), "floor_hits_of_query_label_pairs": floor_hits,
                                     "query_label_pairs": int(dots.size)}
            print(f"  fold {k}: s2 {s2:.4f}, floor hits {floor_hits:,} of {dots.size:,}", flush=True)

    checks = {}
    rr = {}
    systems = {}
    for sp in spaces:
        systems[sp] = {}
        for s in SCORERS:
            ranks = ranks_of(scores[sp][s], li)
            rr[f"{sp}/{s}"] = 1.0 / ranks
            systems[sp][s] = round(float(np.mean(1.0 / ranks)), 4)
        checks[sp] = {"expected": EXPECTED[sp], "recomputed": systems[sp]["prototype"],
                      "gap": round(abs(systems[sp]["prototype"] - EXPECTED[sp]), 4)}
        print(f"check {sp}: prototype {systems[sp]['prototype']} expected {EXPECTED[sp]}  | " +
              "  ".join(f"{s}={v}" for s, v in systems[sp].items()), flush=True)
    if any(c["gap"] > CHECK_GAP for c in checks.values()):
        raise SystemExit(f"the prototype does not reproduce the published numbers: {checks}")

    all_mask = np.ones(n, dtype=bool)
    overall = {sp: paired_group_bootstrap(rr, w, gi, all_mask, [(f"{sp}/{s}", f"{sp}/prototype") for s in SCORERS if s != "prototype"])
               for sp in spaces}
    bands = {}
    for name, lo, hi in BANDS:
        mask = band_of_song == name
        entry = {"labels": band_labels[name], "queries": int(mask.sum()),
                 "counts_for_the_rule": band_labels[name] >= MIN_LABELS and int(mask.sum()) >= MIN_QUERIES,
                 "mrr": {}, "headline_contrasts": {}}
        for sp in spaces:
            entry["mrr"][sp] = {s: round(float(np.sum(rr[f"{sp}/{s}"][mask] * w[mask]) / np.sum(w[mask])), 4) for s in SCORERS}
        for s in SCORERS:
            pairs = [(f"{a}/{s}", f"{b}/{s}") for a, b in HEADLINE]
            entry["headline_contrasts"][s] = paired_group_bootstrap(rr, w, gi, mask, pairs) if mask.sum() > 0 else []
        bands[name] = entry
    readings = {}
    for s in SCORERS:
        for a, b in HEADLINE:
            failing = []
            for name, entry in bands.items():
                if not entry["counts_for_the_rule"]:
                    continue
                c = next(c for c in entry["headline_contrasts"][s] if (c["system"], c["minus"]) == (f"{a}/{s}", f"{b}/{s}"))
                if not (c["mrr_difference"] > 0 and c["ci95"][0] > 0):
                    failing.append(name)
            readings[f"{a} - {b} under {s}"] = ("size-robust" if not failing else f"size-dependent; fails in bands {failing}")
    for key, value in readings.items():
        print(f"reading: {key}: {value}", flush=True)
    for name, entry in bands.items():
        print(f"band {name}: labels {entry['labels']} queries {entry['queries']}", flush=True)
        for sp in spaces:
            print(f"   {sp:20s} " + "  ".join(f"{s}={v:.4f}" for s, v in entry["mrr"][sp].items()), flush=True)

    payload = {
        "analysis": "prototype scoring against Z-normalised and noise-corrected profiles, read by label size",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L, "songs_per_label_bands": band_labels},
        "design": {"scorers": {"prototype": "cosine with the normalised leave-group-out weighted sum (the protocol)",
                               "znorm": "prototype score standardised per label by the impostor scores of training-fold songs of other labels",
                               "noise_corrected": f"dot with the profile mean divided by sqrt(max(||m||^2 - s^2/n, {FLOOR_SHARE}*||m||^2)); s^2 the pooled within-label residual variance on the training folds"},
                   "folds": "identity_probe_v2 folds and transforms; profiles leave-group-out over all songs; cohort and s^2 from training folds",
                   "bands": "songs per label of the query's label",
                   "reading_rule": f"size-robust if the headline contrast is positive with the interval clear of zero in every band with at least {MIN_LABELS} labels and {MIN_QUERIES} queries",
                   "checks": "generic scorer equals the protocol's dense and sparse scorers on 300 queries; prototype MRR reproduces the published numbers (gap <= 0.002)"},
        "checks": {"generic_scorer_gap_dense": gap_dense, "generic_scorer_gap_sparse": gap_sparse, "against_published": checks},
        "fold_info": fold_info,
        "systems": systems,
        "overall_contrasts_against_prototype": overall,
        "by_band": bands,
        "readings": readings,
        "privacy": "aggregate only",
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "label_size_calibration.json").write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nwrote {args.out_dir / 'label_size_calibration.json'}  ({payload['minutes']} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
