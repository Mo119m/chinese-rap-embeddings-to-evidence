#!/usr/bin/env python3
"""A generative model of why whitening recovers identity, and a test of its predictions.

The probe measured that within-author whitening lifts BGE-M3 from 0.318 to 0.449, that a
label-free total whitening gets 0.432 of that, and that centring alone does nothing. Those
are measurements. This file asks whether the simplest model of the situation predicts them.

The model is the two-covariance model of speaker recognition (Prince and Elder 2007; Ioffe
2006), stated for songs:

    x = m + mu_c + e,     mu_c ~ N(0, S_b),     e ~ N(0, S_w)

A song vector is a corpus mean m, plus an author offset mu_c drawn from a between-author
covariance S_b, plus a within-author residual e drawn from S_w -- what the song is about.
Nothing about identity is in the model except that authors have a mean.

Under a symmetric positive transform A of the centred vectors, the retrieval margin of the
true author t over a rival r for a query x_q = mu_t + e_q, scored by dot product against
the authors' means, is D = (mu_t + e_q)' A (mu_t - mu_r). Gaussian moment identities give

    E[D]   = tr(A S_b)
    Var[D] = 2 tr(A S_w A S_b) + 3 tr((A S_b)^2)

so the separability index of a transform is

    S(A) = tr(A S_b) / sqrt( 2 tr(A S_w A S_b) + 3 tr((A S_b)^2) ).

S is maximised over A by the within-author whitening A = S_w^{-1} (Fisher's argument); the
raw dot product A = I weights each direction by its total variance whether or not authors
differ along it, and total whitening A = (S_b + S_w)^{-1} approaches S_w^{-1} exactly when
S_b is small next to S_w, which is what the 89.6% within-author share says. The index is
computed here with the very transforms the probe used, fitted on the same folds.

The index orders transforms; it does not give an MRR. For that the model is simulated:
authors with the corpus's own label sizes, mu_c and e drawn from the fitted S_b and S_w,
the corpus mean m added back, and the leave-one-out cosine protocol run on the synthetic
songs with transforms fitted on the synthetic data exactly as they were fitted on the real
data. The model has no free parameter beyond m, S_b and S_w, all estimated from the
training folds, so the synthetic MRRs are predictions. Where they match the observed
0.318 / 0.314 / 0.432 / 0.449, the geometry effect is explained; where they do not, the
gap says what the model leaves out.

Also reported: the between-author share of variance along each principal direction of the
total covariance, which is the picture of where identity sits -- in the large directions
that dominate a raw cosine, or in the small ones it ignores.

    python src/two_covariance_model_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import (  # noqa: E402
    MINIMUM_SONGS_PER_LABEL,
    build_songs,
    load,
)
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
TRANSFORMS = ("none", "centred", "total_whitening", "within_author_whitening")
OBSERVED = {"none": 0.3181, "centred": 0.3136, "total_whitening": 0.4316,
            "within_author_whitening": 0.4494}
REPLICATES = 3


def covariances(train: np.ndarray, labels: np.ndarray, weights: np.ndarray, label_count: int):
    """m, S_w, S_b from weighted training songs; S_b corrected for the noise in label means."""
    m = np.average(train, axis=0, weights=weights)
    centred = train - m
    sums = np.zeros((label_count, train.shape[1]))
    np.add.at(sums, labels, centred * weights[:, None])
    mass = np.bincount(labels, weights=weights, minlength=label_count)
    present = mass > 0
    means = np.zeros_like(sums)
    means[present] = sums[present] / mass[present, None]
    residuals = centred - means[labels]
    s_w = (residuals * weights[:, None]).T @ residuals / weights.sum()
    # between: covariance of the label means, minus the within noise each mean carries
    counts = np.bincount(labels, minlength=label_count).astype(float)
    s_b_raw = (means[present].T @ means[present]) / present.sum()
    s_b = s_b_raw - s_w * float(np.mean(1.0 / counts[present]))
    values, vectors = np.linalg.eigh((s_b + s_b.T) / 2)
    s_b = (vectors * np.clip(values, 0, None)) @ vectors.T
    return m, s_w, s_b


def separability(a: np.ndarray, s_b: np.ndarray, s_w: np.ndarray) -> float:
    signal = float(np.trace(a @ s_b))
    noise = 2 * float(np.trace(a @ s_w @ a @ s_b)) + 3 * float(np.trace((a @ s_b) @ (a @ s_b)))
    return signal / np.sqrt(max(noise, 1e-18))


def protocol_mrr(vectors: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                 label_count: int, queries: np.ndarray) -> float:
    groups = np.arange(len(labels))  # singleton groups: leave-one-out
    scores = dense_leave_group_out(unit_rows(vectors), labels, groups, weights, label_count, queries)
    ranks = v1.rank_system(scores.astype(np.float32), labels[queries])[0]
    return float(np.mean(1.0 / ranks))


def build(private_root: Path, out_dir: Path) -> int:
    print("loading", flush=True)
    rows, vectors, _ = load(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, centroids_by_song = build_songs(
        rows, vectors)
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {s for s in chunks_by_song
                   if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(l for l, m in songs_by_label.items()
                      if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
    label_index = np.asarray([eligible.index(label_by_song[s]) for s in songs], dtype=np.int64)
    groups = build_groups(songs, components_by_song,
                          {s: normalise_document(documents[s]) for s in songs})
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    label_count = len(eligible)
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs])).astype(np.float64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))]
                          for g, l in zip(group_ids, label_index)])
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    label_sizes = np.bincount(label_index, minlength=label_count)
    print(f"  {len(songs):,} songs, {label_count} labels", flush=True)

    per_fold = []
    for k in range(FOLDS):
        train = fold != k
        m, s_w, s_b = covariances(dense[train], label_index[train], weights[train], label_count)
        total_var, within_var, between_var = np.trace(s_w + s_b), np.trace(s_w), np.trace(s_b)

        # the probe's own transforms, refitted on this fold's training songs
        matrices = {}
        for name in TRANSFORMS:
            _, matrix, _ = fit_transform(name, dense[train], label_index[train], weights[train],
                                         np.random.default_rng(SEED + k))
            matrices[name] = matrix
        index = {name: separability(matrix.T @ matrix, s_b, s_w) for name, matrix in matrices.items()}

        # where identity sits: between-author share along the principal directions of S_t
        values, directions = np.linalg.eigh(s_w + s_b)
        order_desc = np.argsort(-values)
        directions, values = directions[:, order_desc], values[order_desc]
        b_along = np.einsum("ij,ji->i", directions.T @ s_b, directions)
        share_along = b_along / np.maximum(values, 1e-12)
        cumulative = np.cumsum(values) / values.sum()
        bands = {}
        for lo, hi, name in ((0, 8, "top_8"), (8, 64, "9_to_64"), (64, 256, "65_to_256"), (256, 1024, "257_to_1024")):
            bands[name] = {"total_variance_share": round(float(values[lo:hi].sum() / values.sum()), 4),
                           "between_author_share_within_band": round(float(b_along[lo:hi].sum() / values[lo:hi].sum()), 4)}
        print(f"fold {k}: within {within_var / total_var:.3f} of variance; separability "
              + " ".join(f"{n} {index[n]:.3f}" for n in TRANSFORMS), flush=True)
        print("  between-author share by variance band: "
              + ", ".join(f"{n} {b['between_author_share_within_band']:.3f}" for n, b in bands.items()), flush=True)

        # simulate the model with the corpus's label sizes and score the exact protocol
        predicted = {name: [] for name in TRANSFORMS}
        rng = np.random.default_rng(SEED + 100 * k)
        chol_b = np.linalg.cholesky(s_b + 1e-9 * np.eye(s_b.shape[0]))
        chol_w = np.linalg.cholesky(s_w + 1e-9 * np.eye(s_w.shape[0]))
        for r in range(REPLICATES):
            mu = rng.standard_normal((label_count, dense.shape[1])) @ chol_b.T
            synthetic_labels = np.repeat(np.arange(label_count), label_sizes)
            e = rng.standard_normal((len(synthetic_labels), dense.shape[1])) @ chol_w.T
            x = m + mu[synthetic_labels] + e
            x = unit_rows(x)  # the real vectors are unit length before any transform
            synthetic_weights = np.ones(len(synthetic_labels))
            synthetic_fold = rng.integers(0, FOLDS, size=len(synthetic_labels))
            for name in TRANSFORMS:
                scores = np.zeros((len(synthetic_labels), label_count))
                for kk in range(FOLDS):
                    tr = synthetic_fold != kk
                    mean_s, matrix_s, _ = fit_transform(name, x[tr], synthetic_labels[tr],
                                                        synthetic_weights[tr], np.random.default_rng(SEED + kk))
                    projected = unit_rows((x - mean_s) @ matrix_s.T)
                    queries = np.flatnonzero(~tr)
                    scores[queries] = dense_leave_group_out(projected, synthetic_labels,
                                                            np.arange(len(synthetic_labels)),
                                                            synthetic_weights, label_count, queries)
                ranks = v1.rank_system(scores.astype(np.float32), synthetic_labels)[0]
                predicted[name].append(float(np.mean(1.0 / ranks)))
            print(f"  replicate {r}: " + " ".join(f"{n} {predicted[n][-1]:.3f}" for n in TRANSFORMS), flush=True)
        per_fold.append({
            "fold": k,
            "variance": {"total": round(float(total_var), 5), "within_share": round(float(within_var / total_var), 4),
                         "between_share": round(float(between_var / total_var), 4)},
            "separability_index": {n: round(v, 4) for n, v in index.items()},
            "between_author_share_by_variance_band": bands,
            "top_direction_cumulative_variance": {str(i): round(float(cumulative[i - 1]), 4) for i in (1, 8, 64, 256)},
            "predicted_mrr": {n: round(float(np.mean(v)), 4) for n, v in predicted.items()},
            "predicted_mrr_replicates": {n: [round(x, 4) for x in v] for n, v in predicted.items()},
        })

    summary = {}
    for name in TRANSFORMS:
        pred = float(np.mean([f["predicted_mrr"][name] for f in per_fold]))
        summary[name] = {"observed_mrr": OBSERVED[name], "predicted_mrr": round(pred, 4),
                         "prediction_error": round(pred - OBSERVED[name], 4),
                         "separability_index_mean": round(float(np.mean([f["separability_index"][name] for f in per_fold])), 4)}
    gain_obs = OBSERVED["within_author_whitening"] - OBSERVED["none"]
    gain_pred = summary["within_author_whitening"]["predicted_mrr"] - summary["none"]["predicted_mrr"]
    print("\npredicted vs observed MRR:", flush=True)
    for name, s in summary.items():
        print(f"  {name:26s} predicted {s['predicted_mrr']:.4f}  observed {s['observed_mrr']:.4f}  "
              f"error {s['prediction_error']:+.4f}  S {s['separability_index_mean']:.3f}", flush=True)
    print(f"  whitening gain: predicted {gain_pred:+.4f}, observed {gain_obs:+.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "two-covariance model of the song vectors: separability index, simulated protocol, and the observed probe",
        "model": "x = m + mu_c + e, mu_c ~ N(0, S_b), e ~ N(0, S_w); m, S_b, S_w estimated on each fold's training songs with the probe's shrinkage; no free parameter",
        "separability_index": "S(A) = tr(A S_b) / sqrt(2 tr(A S_w A S_b) + 3 tr((A S_b)^2)) for the transform A = T'T actually used by the probe",
        "simulation": (f"{REPLICATES} synthetic corpora per fold with the corpus's 226 label sizes, unit-normalised as the "
                       "real vectors are, transforms refitted on synthetic folds, leave-one-out cosine protocol"),
        "songs": len(songs), "labels": label_count,
        "summary": summary,
        "whitening_gain": {"observed": round(gain_obs, 4), "predicted": round(gain_pred, 4)},
        "folds": per_fold,
        "reading": ("if the predicted MRRs track the observed ones across the four transforms, the "
                    "probe's effect is the geometry the model describes -- identity in directions "
                    "of small total variance that a raw cosine cannot weight -- and nothing more; a "
                    "systematic gap would point to structure the Gaussian model lacks"),
        "privacy": "aggregate only",
    }
    (out_dir / "two_covariance_model.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'two_covariance_model.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
