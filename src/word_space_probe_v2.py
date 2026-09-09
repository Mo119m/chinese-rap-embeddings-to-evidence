#!/usr/bin/env python3
"""The word space given the probe's treatment, and the fair best system.

The probe whitened a 1,024-dimensional SVD of the character space to 0.518 and fused it
with the whitened semantic space to 0.587. The representation study then found that jieba
words identify a label better than characters (0.519 raw). This file closes the loop: the
word space is reduced fold-wise to 1,024 dimensions, given the same three transforms, and
fused with the better of the two whitened semantic spaces -- chunk vectors whitened before
averaging (0.462) -- so that every component of the best system has had the same
label-free and label-fitted corrections applied under the same folds.

    python src/word_space_probe_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import (  # noqa: E402
    MINIMUM_SONGS_PER_LABEL,
    build_songs,
    load,
)
from identity_probe_v2 import (  # noqa: E402
    FOLDS,
    SEED,
    SVD_COMPONENTS,
    dense_leave_group_out,
    fit_transform,
    unit_rows,
)
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from word_identity_anatomy_v2 import EXPECTED_WORD_MRR, fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
TRANSFORMS = ("none", "total_whitening", "within_author_whitening")


def build(private_root: Path, out_dir: Path) -> int:
    import jieba
    jieba.setLogLevel(60)
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
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))]
                          for g, l in zip(group_ids, label_index)])
    song_pos = {s: i for i, s in enumerate(songs)}
    everything = np.arange(len(songs))
    print(f"  {len(songs):,} queries, {label_count} labels", flush=True)

    # folds shared with the probe and the representation study
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]

    # ------------------------------------------------------------ the word space
    print("word space", flush=True)
    lexical, _ = fit_words([" ".join(segment(documents[s])) for s in songs])
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    profiles = v1.score_leave_group_out(dense, lexical, label_index, group_ids, label_count)
    word_ranks = v1.rank_system(profiles.lexical, label_index)[0].astype(np.int64)
    word_mrr = float(np.mean(1.0 / word_ranks))
    print(f"  raw MRR {word_mrr:.4f}", flush=True)
    if abs(word_mrr - EXPECTED_WORD_MRR) > 5e-4:
        raise SystemExit(f"the word space gives {word_mrr:.4f}, not {EXPECTED_WORD_MRR}")

    scores = {f"words_svd_{name}": np.zeros((len(songs), label_count)) for name in TRANSFORMS}
    variance = []
    for k in range(FOLDS):
        train = fold != k
        queries = np.flatnonzero(~train)
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(lexical[train])
        variance.append(round(float(svd.explained_variance_ratio_.sum()), 4))
        reduced = unit_rows(svd.transform(lexical))
        for name in TRANSFORMS:
            mean, matrix, _ = fit_transform(name, reduced[train], label_index[train], weights[train],
                                            np.random.default_rng(SEED + k))
            scores[f"words_svd_{name}"][queries] = dense_leave_group_out(
                unit_rows((reduced - mean) @ matrix.T), label_index, group_ids, weights,
                label_count, queries)
        print(f"  fold {k}: {variance[-1]:.3f} of variance kept", flush=True)

    # ------------------------------------------------------------ the semantic space, chunks whitened first
    print("semantic space, chunk vectors whitened before averaging", flush=True)
    chunk_rows = [i for s in songs for i in sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"]))]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    chunk_vec = unit_rows(vectors[chunk_rows].astype(np.float64))
    chunk_label, chunk_group = label_index[chunk_song], group_ids[chunk_song]
    chunk_weight = weights[chunk_song] / np.bincount(chunk_song)[chunk_song]
    chunk_fold = fold[chunk_song]
    semantic = np.zeros((len(songs), label_count))
    for k in range(FOLDS):
        train = chunk_fold != k
        mean, matrix, _ = fit_transform("within_author_whitening", chunk_vec[train], chunk_label[train],
                                        chunk_weight[train], np.random.default_rng(SEED + k))
        projected = unit_rows((chunk_vec - mean) @ matrix.T)
        song_vec = np.zeros((len(songs), projected.shape[1]))
        np.add.at(song_vec, chunk_song, projected)
        song_vec = unit_rows(song_vec / np.bincount(chunk_song)[:, None])
        queries = np.flatnonzero(fold == k)
        semantic[queries] = dense_leave_group_out(song_vec, label_index, group_ids, weights,
                                                  label_count, queries)
        print(f"  fold {k}", flush=True)
    scores["semantic_whitened_chunks"] = semantic

    # ------------------------------------------------------------ fusions
    fused = (v1.zscore_rows(scores["words_svd_within_author_whitening"]) + v1.zscore_rows(semantic)) / 2.0
    scores["fusion_whitened_words_whitened_semantic"] = fused
    scores["fusion_raw_words_raw_semantic"] = (
        v1.zscore_rows(profiles.lexical.astype(np.float64)) + v1.zscore_rows(profiles.dense.astype(np.float64))) / 2.0

    ranks = {"words_raw": word_ranks}
    for name, s in scores.items():
        ranks[name] = v1.rank_system(s.astype(np.float32), label_index)[0].astype(np.int64)
    rr = {name: 1.0 / r for name, r in ranks.items()}
    report = {}
    for name, r in ranks.items():
        report[name] = {"mrr": round(float(np.mean(1.0 / r)), 4),
                        "recall_at_1": round(float(np.mean(r <= 1)), 4),
                        "recall_at_10": round(float(np.mean(r <= 10)), 4)}
        print(f"  {name:42s} MRR {report[name]['mrr']:.4f}  R@1 {report[name]['recall_at_1']:.3f}  "
              f"R@10 {report[name]['recall_at_10']:.3f}", flush=True)

    pairs = [("words_svd_none", "words_raw"),
             ("words_svd_total_whitening", "words_svd_none"),
             ("words_svd_within_author_whitening", "words_svd_total_whitening"),
             ("fusion_whitened_words_whitened_semantic", "words_svd_within_author_whitening"),
             ("fusion_whitened_words_whitened_semantic", "fusion_raw_words_raw_semantic")]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "the word space under the probe's transforms, and the fair best system",
        "queries": len(songs), "labels": label_count,
        "design": {"folds": FOLDS, "seed": SEED, "svd_components": SVD_COMPONENTS,
                   "svd_variance_kept_by_fold": variance,
                   "rule": "every transform fitted on four folds of leakage groups and applied "
                           "to the fifth; profiles from all other songs as in the protocol"},
        "verification": {"words_raw_reproduces_representation_study": round(word_mrr, 4)},
        "systems": report,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled "
                                       "with replacement, each (group, label) component weighted one",
                             "contrasts": contrasts},
        "privacy": "aggregate only",
    }
    (out_dir / "word_space_probe.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'word_space_probe.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
