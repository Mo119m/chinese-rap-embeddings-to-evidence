#!/usr/bin/env python3
"""PLDA scoring of the semantic song vectors: is the Bayes-optimal scorer of the
two-covariance model any better than whitened cosine?

The identity probe showed that whitening BGE-M3's song vectors by the within-label
covariance lifts the leave-group-out MRR from 0.30 to 0.42. Whitened cosine is a heuristic
reading of the two-covariance model (a label mean drawn from a between-label Gaussian, songs
scattered around it by a shared within-label Gaussian); the model's own scorer is
probabilistic linear discriminant analysis (Prince and Elder 2007; Ioffe 2006), and cosine
is the special case of it with both covariances fixed to the identity (Peng et al.,
Interspeech 2022). The two-covariance ablation found that the model over-predicts the MRR
twofold however its marginals are chosen, and that what every version shares -- a song's
scatter is exchangeable across labels -- is the assumption at fault. This file scores the
model's way and adds the smallest relaxation of that assumption:

  cosine                  the untransformed space, the builder's scorer (reference)
  wccn_cosine             within-label whitening then cosine (the probe's best transform)
  plda                    two-covariance PLDA, simultaneously diagonalised: within = I,
                          between = diag(psi); each label's enrolment songs give a Gaussian
                          posterior over its mean and the query is scored by the posterior
                          predictive log-density (the LLR's denominator is the same for
                          every label, so it does not move the ranking)
  plda_diagonal           the same with both covariances diagonal in the centred
                          coordinates (the cosine-like independence assumption)
  plda_label_scale        the same as plda with a label-specific within-label scale s_l^2,
                          estimated from the label's training songs and shrunk towards one

Everything is fold-wise as in the probe: leakage groups dealt into five folds, a fold's
model fitted on the other four, only that fold's queries scored, enrolment leave-group-out
as in the original protocol, per-(group, label) weight one.

    python src/plda_scoring_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.covariance import ledoit_wolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
SCALE_SHRINKAGE = 5.0   # pseudo-songs pulling a label's within scale towards one
PSI_FLOOR = 1e-6


def weighted_covariance(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scaled = rows * np.sqrt(weights * len(weights) / weights.sum())[:, None]
    covariance, _ = ledoit_wolf(scaled, assume_centered=True)
    return covariance


def fit_plda(train: np.ndarray, labels: np.ndarray, weights: np.ndarray, diagonal: bool):
    """Return (mean, A, psi): x' = A (x - mean) has within = I and between = diag(psi)."""
    mean = np.average(train, axis=0, weights=weights)
    centred = train - mean
    label_count = int(labels.max()) + 1
    sums = np.zeros((label_count, train.shape[1]))
    np.add.at(sums, labels, centred * weights[:, None])
    mass = np.bincount(labels, weights=weights, minlength=label_count)
    present = mass > 0
    centroids = np.zeros_like(sums)
    centroids[present] = sums[present] / mass[present, None]
    residuals = centred - centroids[labels]
    within = weighted_covariance(residuals, weights)
    total = weighted_covariance(centred, weights)
    between = total - within
    if diagonal:
        w = np.maximum(np.diag(within), 1e-8)
        b = np.maximum(np.diag(between), PSI_FLOOR * w)
        return mean, np.diag(1.0 / np.sqrt(w)), b / w
    values, vectors = np.linalg.eigh(within)
    if values.min() <= 0:
        raise RuntimeError("the within covariance is not positive definite")
    whitener = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    between_w = whitener @ between @ whitener
    between_w = (between_w + between_w.T) / 2.0
    psi, rotation = np.linalg.eigh(between_w)
    psi = np.maximum(psi, PSI_FLOOR)
    return mean, rotation.T @ whitener, psi


def label_scales(projected: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                 label_count: int) -> np.ndarray:
    """Per-label within scale s_l^2 in the diagonalised space, shrunk towards one."""
    dimension = projected.shape[1]
    scales = np.ones(label_count)
    for label in range(label_count):
        members = np.flatnonzero(labels == label)
        mass = float(weights[members].sum()) if len(members) else 0.0
        if mass <= 1.0:
            continue
        centroid = np.average(projected[members], axis=0, weights=weights[members])
        spread = float(np.sum(weights[members] * np.sum((projected[members] - centroid) ** 2, axis=1)))
        estimate = spread / ((mass - 1.0) * dimension)
        scales[label] = ((mass - 1.0) * estimate + SCALE_SHRINKAGE * 1.0) / ((mass - 1.0) + SCALE_SHRINKAGE)
    # Ledoit-Wolf shrinkage of W inflates its small eigenvalues, so in-sample residuals in the
    # whitened space sit below unit variance for every label (median 0.83 on the first run);
    # only the RELATIVE scale is the hypothesis, so the scales are normalised to a
    # mass-weighted mean of one and the absolute level stays the pooled model's.
    mass_by_label = np.bincount(labels, weights=weights, minlength=label_count)
    scales = scales / float(np.average(scales, weights=np.maximum(mass_by_label, 1e-12)))
    return scales


def plda_scores(projected: np.ndarray, psi: np.ndarray, label_index: np.ndarray,
                group_ids: np.ndarray, weights: np.ndarray, label_count: int,
                queries: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
    """Posterior-predictive log-density of each query under each label's enrolment,
    with the query's own leakage group removed from the enrolment as in the protocol."""
    if scales is None:
        scales = np.ones(label_count)
    sums = np.zeros((label_count, projected.shape[1]))
    np.add.at(sums, label_index, projected * weights[:, None])
    counts = np.bincount(label_index, weights=weights, minlength=label_count)

    def stats(sum_vector, count, scale):
        # posterior over the label mean per dimension, then the predictive of a new song
        mean = sum_vector / max(count, 1e-12)
        gain = count * psi / (scale + count * psi) if count > 0 else np.zeros_like(psi)
        post_mean = gain * mean
        post_var = scale + psi * scale / (scale + count * psi) if count > 0 else scale + psi
        return post_mean, post_var

    post_mean = np.zeros((label_count, projected.shape[1]))
    post_var = np.zeros((label_count, projected.shape[1]))
    for label in range(label_count):
        post_mean[label], post_var[label] = stats(sums[label], counts[label], scales[label])
    x = projected[queries]
    inv = 1.0 / post_var
    scores = -0.5 * ((x ** 2) @ inv.T - 2.0 * x @ (post_mean * inv).T
                     + np.sum(post_mean ** 2 * inv, axis=1)[None, :]
                     + np.sum(np.log(post_var), axis=1)[None, :])
    members_by_group: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids.tolist()):
        members_by_group[int(group)].append(index)
    for row, query in enumerate(queries.tolist()):
        members = members_by_group[int(group_ids[query])]
        for label in {int(label_index[m]) for m in members}:
            own = [m for m in members if int(label_index[m]) == label]
            leave_sum = sums[label] - np.sum(projected[own] * weights[own, None], axis=0)
            leave_count = counts[label] - float(weights[own].sum())
            if leave_count <= 1e-9:
                raise RuntimeError("a held-out group empties a label's enrolment")
            m, v = stats(leave_sum, leave_count, scales[label])
            scores[row, label] = -0.5 * float(np.sum((x[row] - m) ** 2 / v + np.log(v)))
    return scores


def build(private_root: Path, out_dir: Path) -> int:
    print("loading corpus v3", flush=True)
    rows, vectors, vector_state = load_v3(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, centroids_by_song = build_songs(rows, vectors)
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {s for s in chunks_by_song
                   if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(l for l, m in songs_by_label.items()
                      if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
    label_index = np.asarray([eligible.index(label_by_song[s]) for s in songs], dtype=np.int64)
    groups = build_groups(songs, components_by_song, {s: normalise_document(documents[s]) for s in songs})
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    label_count = len(eligible)
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs])).astype(np.float64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    rng = np.random.default_rng(SEED)
    fold_of_group = rng.integers(0, FOLDS, size=len(order))
    fold = fold_of_group[group_ids]
    systems = ("cosine", "wccn_cosine", "plda", "plda_diagonal", "plda_label_scale")
    scores = {name: np.zeros((len(songs), label_count)) for name in systems}
    diagnostics = []
    for k in range(FOLDS):
        train_mask = fold != k
        queries = np.flatnonzero(~train_mask)
        train_labels = label_index[train_mask]
        train_weights = weights[train_mask]
        print(f"fold {k}: fitting on {int(train_mask.sum()):,} songs, scoring {len(queries):,}", flush=True)
        # the probe's two cosine systems, recomputed here so every system shares the folds
        for name, transform in (("cosine", "none"), ("wccn_cosine", "within_author_whitening")):
            mean, matrix, _ = fit_transform(transform, dense[train_mask], train_labels, train_weights,
                                            np.random.default_rng(SEED + k))
            scores[name][queries] = dense_leave_group_out(unit_rows((dense - mean) @ matrix.T), label_index,
                                                          group_ids, weights, label_count, queries)
        for name, diagonal in (("plda", False), ("plda_diagonal", True)):
            mean, projection, psi = fit_plda(dense[train_mask], train_labels, train_weights, diagonal)
            projected = (dense - mean) @ projection.T
            scores[name][queries] = plda_scores(projected, psi, label_index, group_ids, weights,
                                                label_count, queries)
            if name == "plda":
                scales = label_scales(projected[train_mask], train_labels, train_weights, label_count)
                scores["plda_label_scale"][queries] = plda_scores(projected, psi, label_index, group_ids,
                                                                  weights, label_count, queries, scales)
                diagnostics.append({
                    "fold": k,
                    "between_to_within_ratio_top": [round(float(v), 3) for v in np.sort(psi)[::-1][:5]],
                    "dimensions_with_psi_over_0.1": int((psi > 0.1).sum()),
                    "dimensions_with_psi_over_1": int((psi > 1.0).sum()),
                    "label_scale_quartiles": [round(float(v), 3) for v in np.percentile(scales, [5, 25, 50, 75, 95])],
                })
                print(f"  psi > 0.1 in {diagnostics[-1]['dimensions_with_psi_over_0.1']} dimensions; "
                      f"label scale 5/50/95%: {diagnostics[-1]['label_scale_quartiles'][0]:.2f} / "
                      f"{diagnostics[-1]['label_scale_quartiles'][2]:.2f} / "
                      f"{diagnostics[-1]['label_scale_quartiles'][4]:.2f}", flush=True)

    ranks = {name: v1.rank_system(s.astype(np.float32), label_index)[0].astype(np.int64)
             for name, s in scores.items()}
    rr = {name: 1.0 / r for name, r in ranks.items()}
    all_mask = np.ones(len(songs), dtype=bool)
    report = {}
    for name, r in ranks.items():
        report[name] = {"mrr": round(float(np.mean(1.0 / r)), 4),
                        "mrr_component_weighted": round(weighted_mean(rr[name], weights, all_mask), 4),
                        "recall_at_1": round(float(np.mean(r <= 1)), 4),
                        "recall_at_10": round(float(np.mean(r <= 10)), 4)}
        print(f"  {name:20s} MRR {report[name]['mrr']:.4f}  R@1 {report[name]['recall_at_1']:.4f}  "
              f"R@10 {report[name]['recall_at_10']:.4f}", flush=True)
    pairs = [("plda", "cosine"), ("plda", "wccn_cosine"), ("plda_diagonal", "plda"),
             ("plda_label_scale", "plda"), ("plda_label_scale", "wccn_cosine")]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, all_mask, pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "PLDA scoring of the semantic song vectors against whitened cosine",
        "corpus": {"v3_content_sha256": V3_CONTENT_SHA256, "vectors_sha256": vector_state["sha256"],
                   "queries": len(songs), "labels": label_count, "groups": len(order)},
        "design": {"folds": FOLDS, "seed": SEED, "unit_dealt_into_folds": "leakage group",
                   "enrolment": "each label's other songs, the query's leakage group removed, "
                                "per-(group, label) weight one",
                   "scoring": "posterior-predictive log-density of the query under the label's "
                              "Gaussian posterior mean; the different-label denominator is "
                              "label-independent and omitted",
                   "covariances": "within: Ledoit-Wolf of weighted within-label residuals; "
                                  "between: total minus within, eigenvalues floored at 1e-6 after "
                                  "simultaneous diagonalisation",
                   "label_scale": f"s_l^2 = weighted mean squared residual per dimension of the label's "
                                  f"training songs, shrunk towards one with {SCALE_SHRINKAGE:g} pseudo-songs"},
        "systems": report,
        "fold_diagnostics": diagnostics,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with "
                                       "replacement, each (group, label) component weighted one",
                             "contrasts": contrasts},
        "privacy": "aggregate only",
    }
    (out_dir / "plda_scoring.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'plda_scoring.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
