#!/usr/bin/env python3
"""Which assumption of the two-covariance model is wrong? One assumption at a time, on v3.

The Gaussian two-covariance model predicted the ordering of the probe's transforms and
twice their size: raw cosine at 0.66 against 0.32 observed, whitened at 1.00 against 0.45.
The model makes exactly two distributional assumptions -- author means are independent
Gaussian draws from S_b, and residuals are Gaussian draws from S_w -- and the corpus can
replace either with the real thing while keeping the other. Four synthetic corpora per
fold, scored under the exact protocol with transforms refitted on the synthetic data:

    gaussian_means + gaussian_residuals    the model as published
    real_means     + gaussian_residuals    authors as they are, songs about them Gaussian
    gaussian_means + real_residuals        authors Gaussian, songs' departures as they are
    real_means     + real_residuals        both real; residuals reassigned across authors,
                                           which is the real corpus with author and content
                                           decoupled

"Real residuals" are the training songs' actual departures from their label mean, drawn
with replacement and attached to a random author, so that within-author structure
survives but its pairing with a particular author does not. The arm whose prediction
falls to the observed value names the assumption that was carrying the error.

    python src/two_covariance_ablation_v3.py --private-root <ni-k>
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
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from two_covariance_model_v2 import covariances  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
TRANSFORMS = ("none", "total_whitening", "within_author_whitening")
ARMS = ("gaussian_means+gaussian_residuals", "real_means+gaussian_residuals",
        "gaussian_means+real_residuals", "real_means+real_residuals")


def build(private_root: Path, out_dir: Path) -> int:
    print("loading corpus v3 with its vectors", flush=True)
    rows, vectors, state = load_v3(private_root)
    if state["vectors"] != "v3":
        raise SystemExit("the v3 embedding run is required")
    chunks_by_song, label_by_song, components_by_song, documents, centroids = build_songs(rows, vectors)
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
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    dense = v1.l2_normalize_dense(np.stack([centroids[s] for s in songs])).astype(np.float64)
    label_sizes = np.bincount(label_index, minlength=label_count)
    print(f"  {len(songs):,} songs, {label_count} labels", flush=True)

    # the observed protocol on the real v3 vectors, the same three transforms
    print("observed", flush=True)
    observed = {}
    real_scores = {name: np.zeros((len(songs), label_count)) for name in TRANSFORMS}
    for k in range(FOLDS):
        train = fold != k
        queries = np.flatnonzero(~train)
        for name in TRANSFORMS:
            mean, matrix, _ = fit_transform(name, dense[train], label_index[train], weights[train],
                                            np.random.default_rng(SEED + k))
            real_scores[name][queries] = dense_leave_group_out(unit_rows((dense - mean) @ matrix.T), label_index,
                                                               group_ids, weights, label_count, queries)
    for name in TRANSFORMS:
        observed[name] = float(np.mean(1.0 / v1.rank_system(real_scores[name].astype(np.float32), label_index)[0]))
    print("  " + "  ".join(f"{n} {v:.4f}" for n, v in observed.items()), flush=True)

    def score_synthetic(x, labels, rng):
        synthetic_fold = rng.integers(0, FOLDS, size=len(labels))
        w = np.ones(len(labels))
        out = {}
        for name in TRANSFORMS:
            scores = np.zeros((len(labels), label_count))
            for kk in range(FOLDS):
                tr = synthetic_fold != kk
                mean_s, matrix_s, _ = fit_transform(name, x[tr], labels[tr], w[tr], np.random.default_rng(SEED + kk))
                queries = np.flatnonzero(~tr)
                scores[queries] = dense_leave_group_out(unit_rows((x - mean_s) @ matrix_s.T), labels,
                                                        np.arange(len(labels)), w, label_count, queries)
            out[name] = float(np.mean(1.0 / v1.rank_system(scores.astype(np.float32), labels)[0]))
        return out

    per_fold = []
    for k in range(FOLDS):
        train = fold != k
        m, s_w, s_b = covariances(dense[train], label_index[train], weights[train], label_count)
        centred = dense[train] - m
        sums = np.zeros((label_count, dense.shape[1]))
        np.add.at(sums, label_index[train], centred * weights[train][:, None])
        mass = np.bincount(label_index[train], weights=weights[train], minlength=label_count)
        real_means = sums / np.maximum(mass, 1e-12)[:, None]
        real_residuals = centred - real_means[label_index[train]]
        chol_b = np.linalg.cholesky(s_b + 1e-9 * np.eye(s_b.shape[0]))
        chol_w = np.linalg.cholesky(s_w + 1e-9 * np.eye(s_w.shape[0]))
        rng = np.random.default_rng(SEED + 100 * k)
        synthetic_labels = np.repeat(np.arange(label_count), label_sizes)
        results = {}
        for arm in ARMS:
            means = real_means if arm.startswith("real_means") else rng.standard_normal((label_count, dense.shape[1])) @ chol_b.T
            if arm.endswith("gaussian_residuals"):
                e = rng.standard_normal((len(synthetic_labels), dense.shape[1])) @ chol_w.T
            else:
                e = real_residuals[rng.integers(0, len(real_residuals), size=len(synthetic_labels))]
            x = unit_rows(m + means[synthetic_labels] + e)
            results[arm] = score_synthetic(x, synthetic_labels, rng)
            print(f"fold {k} {arm:36s} " + "  ".join(f"{n} {v:.3f}" for n, v in results[arm].items()), flush=True)
        per_fold.append({"fold": k, "predicted_mrr": results})

    summary = {}
    for arm in ARMS:
        summary[arm] = {name: {"predicted": round(float(np.mean([f["predicted_mrr"][arm][name] for f in per_fold])), 4),
                               "observed": round(observed[name], 4)} for name in TRANSFORMS}
    print("\nmean over folds (predicted / observed):", flush=True)
    for arm, block in summary.items():
        print(f"  {arm:36s} " + "  ".join(f"{n} {b['predicted']:.3f}/{b['observed']:.3f}" for n, b in block.items()), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "which assumption of the two-covariance model carries its over-prediction",
        "corpus": {"v3_content_sha256": V3_CONTENT_SHA256, "vectors_sha256": state["sha256"],
                   "songs": len(songs), "labels": label_count},
        "arms": {a: ("independent Gaussian author means from S_b" if a.startswith("gaussian") else "the labels' real means")
                    + " + " + ("Gaussian residuals from S_w" if a.endswith("gaussian_residuals")
                               else "the training songs' real residuals, resampled with replacement and reassigned to random authors")
                 for a in ARMS},
        "observed_mrr": {n: round(v, 4) for n, v in observed.items()},
        "summary": summary,
        "folds": per_fold,
        "reading": ("the arm whose prediction meets the observed value names the assumption that was carrying the "
                    "error; if only the both-real arm meets it, author means and residuals are entangled -- the "
                    "content of a song depends on who wrote it -- and no two-covariance model separates them"),
        "privacy": "aggregate only",
    }
    (out_dir / "two_covariance_ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'two_covariance_ablation.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
