#!/usr/bin/env python3
"""Two claims made about the semantic space, now looked at directly rather than inferred.

1. 'Whitening removes topic.' Within-label whitening shrinks the directions along which a
   label's own songs scatter most. Which directions are those? For the top eigenvectors of
   the pooled within-label covariance, this file reports how much between-label variance
   they carry (little, if they are topic) and which jieba words correlate most with a
   song's position along them (readable, if they are topic: love, street, money, ...).
   The words are common vocabulary and are published; no lyric text is.

2. 'Within-label scatter is label-specific in direction.' The two-covariance ablation
   inferred it from the failure of every arm that reassigns residuals across labels. Here
   it is measured: for every label with enough songs, the leading direction of its own
   residual scatter; the distribution of |cos| between labels' leading directions; and the
   share of each label's residual variance that the pooled covariance's top directions
   explain against what the label's own top directions explain -- all compared with a
   null in which real residuals are reassigned to random labels, the ablation's
   assumption, so the excess is the label-specific part.

    python src/whitening_directions_v3.py --private-root <ni-k>
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
from identity_probe_v2 import SEED  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
TOP_DIRECTIONS = 8
WORDS_PER_DIRECTION = 12
MIN_SONGS_FOR_DIRECTION = 15
TOP_K_SHARE = 10
NULL_DRAWS = 20


def weighted_centroids(x, labels, weights, label_count):
    sums = np.zeros((label_count, x.shape[1]))
    np.add.at(sums, labels, x * weights[:, None])
    mass = np.bincount(labels, weights=weights, minlength=label_count)
    centroids = np.zeros_like(sums)
    present = mass > 0
    centroids[present] = sums[present] / mass[present, None]
    return centroids, mass


def build(private_root: Path, out_dir: Path) -> int:
    import jieba
    jieba.setLogLevel(60)
    print("loading corpus v3", flush=True)
    rows, vectors, _ = load_v3(private_root)
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
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    x = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs])).astype(np.float64)
    print(f"  {len(songs):,} songs, {label_count} labels", flush=True)

    mean = np.average(x, axis=0, weights=weights)
    centred = x - mean
    centroids, mass = weighted_centroids(centred, label_index, weights, label_count)
    residuals = centred - centroids[label_index]
    scale = np.sqrt(weights * len(weights) / weights.sum())[:, None]
    within, _ = ledoit_wolf(residuals * scale, assume_centered=True)
    total, _ = ledoit_wolf(centred * scale, assume_centered=True)
    between = total - within
    w_values, w_vectors = np.linalg.eigh(within)
    order_desc = np.argsort(-w_values)
    w_values, w_vectors = w_values[order_desc], w_vectors[:, order_desc]

    # 1. the directions whitening presses hardest: between-label share and word correlates
    print("word space for correlates", flush=True)
    word_matrix, features = fit_words([" ".join(segment(documents[s])) for s in songs])
    word_df = np.asarray((word_matrix > 0).sum(axis=0)).ravel()
    common = word_df >= 50   # words in at least 50 songs, so a correlate is a real tendency
    directions = []
    total_var_along = np.einsum("ij,jk,ki->i", w_vectors.T, total, w_vectors)
    between_var_along = np.einsum("ij,jk,ki->i", w_vectors.T, between, w_vectors)
    for d in range(TOP_DIRECTIONS):
        v = w_vectors[:, d]
        projection = centred @ v
        z = (projection - projection.mean()) / projection.std()
        # correlation of each word's tf-idf weight with the projection, over songs
        col_mean = np.asarray(word_matrix.mean(axis=0)).ravel()
        cov = np.asarray(word_matrix.T @ z).ravel() / len(z) - col_mean * z.mean()
        col_sq = np.asarray(word_matrix.multiply(word_matrix).mean(axis=0)).ravel()
        col_std = np.sqrt(np.maximum(col_sq - col_mean ** 2, 1e-12))
        corr = np.where(common, cov / col_std, 0.0)
        top_pos = [features[i] for i in np.argsort(-corr)[:WORDS_PER_DIRECTION]]
        top_neg = [features[i] for i in np.argsort(corr)[:WORDS_PER_DIRECTION]]
        directions.append({
            "rank": d + 1, "within_variance_share": round(float(w_values[d] / w_values.sum()), 4),
            "between_share_of_total_along_direction": round(float(between_var_along[d] / max(total_var_along[d], 1e-12)), 4),
            "words_at_positive_end": top_pos, "words_at_negative_end": top_neg})
        print(f"  within-direction {d + 1}: between share {directions[-1]['between_share_of_total_along_direction']:.3f}  "
              f"+ {' '.join(top_pos[:6])}  |  - {' '.join(top_neg[:6])}", flush=True)
    overall_between_share = float(np.trace(between) / np.trace(total))

    # 2. label-specific directions of scatter
    print("label scatter directions", flush=True)
    rng = np.random.default_rng(SEED)
    rich = [l for l in range(label_count) if int(np.sum(label_index == l)) >= MIN_SONGS_FOR_DIRECTION]
    pooled_top = w_vectors[:, :TOP_K_SHARE]

    def analyse(res: np.ndarray) -> dict:
        leading, own_share, pooled_share = [], [], []
        for l in rich:
            r = res[label_index == l]
            u, s, vt = np.linalg.svd(r - r.mean(axis=0), full_matrices=False)
            leading.append(vt[0])
            var = np.sum(s ** 2)
            own_share.append(float(np.sum(s[:TOP_K_SHARE] ** 2) / var))
            proj = (r - r.mean(axis=0)) @ pooled_top
            pooled_share.append(float(np.sum(proj ** 2) / var))
        L = np.stack(leading)
        cos = np.abs(L @ L.T)[np.triu_indices(len(rich), k=1)]
        return {"median_abs_cos_between_leading_directions": round(float(np.median(cos)), 4),
                "mean_abs_cos_between_leading_directions": round(float(np.mean(cos)), 4),
                "share_of_label_variance_in_pooled_top_k": round(float(np.mean(pooled_share)), 4),
                "share_of_label_variance_in_own_top_k": round(float(np.mean(own_share)), 4)}

    observed = analyse(residuals)
    null_runs = []
    for _ in range(NULL_DRAWS):
        shuffled = residuals[rng.permutation(len(residuals))]
        null_runs.append(analyse(shuffled))
    null = {k: [round(float(np.mean([r[k] for r in null_runs])), 4), round(float(np.std([r[k] for r in null_runs])), 4)]
            for k in observed}
    for k in observed:
        print(f"  {k}: observed {observed[k]:.4f}  null {null[k][0]:.4f} ± {null[k][1]:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "what within-label whitening presses, and whether label scatter has label-specific directions",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "songs": len(songs), "labels": label_count,
                   "labels_with_at_least_min_songs": len(rich), "min_songs_for_direction": MIN_SONGS_FOR_DIRECTION},
        "overall_between_share_of_total_variance": round(overall_between_share, 4),
        "pressed_directions": {"design": "top eigenvectors of the Ledoit-Wolf within-label covariance of BGE-M3 song "
                                         "vectors; per direction, the between-label share of total variance along it, "
                                         "and the jieba words (document frequency >= 50 songs) whose TF-IDF weight "
                                         "correlates most with a song's coordinate",
                               "directions": directions},
        "label_scatter": {"design": f"per label with >= {MIN_SONGS_FOR_DIRECTION} songs: leading SVD direction of its "
                                    f"residuals; |cos| between labels' leading directions; share of a label's residual "
                                    f"variance in the pooled covariance's top {TOP_K_SHARE} directions vs its own top "
                                    f"{TOP_K_SHARE}; null = real residuals reassigned to random labels, {NULL_DRAWS} draws",
                          "observed": observed, "null_mean_sd": null},
        "privacy": "aggregate only; the correlated words are common vocabulary",
    }
    (out_dir / "whitening_directions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'whitening_directions.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
