#!/usr/bin/env python3
"""Style over content: can each space pick the right label among content rivals?

Wegmann et al. (2022) test whether a representation prefers style to content by giving it
a choice between a text of the same style and a text of the same content. The retrieval
protocol has a natural version of that choice. For every query song, the frozen semantic
space names the labels whose repertoires are closest to it in content -- its content
rivals. Restricting the candidates to the true label plus its K nearest content rivals
removes the easy wins (labels about something else altogether) and leaves exactly the
question the paper cares about: with subject held roughly constant, does the space still
know who wrote it?

Every space is scored on the same queries, first over all 226 labels (the usual MRR) and
then within the rival set; the rival set is defined once, from the frozen semantic space,
so it is the same for every space. Chance within a set of K+1 candidates is the mean
reciprocal rank of a random permutation, H(K+1)/(K+1).

    python src/content_rival_test_v3.py --private-root <ni-k>
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
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from lexical_identity_anatomy_v2 import score_arm  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
RIVALS = (5, 10, 25)


def rank_within(scores: np.ndarray, truth: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Rank of the true label among the candidate columns (true label included), ties
    broken pessimistically, as v1.rank_system does over the full label set."""
    rows = np.arange(len(truth))
    own = scores[rows, truth]
    picked = scores[rows[:, None], candidates]
    return 1 + np.sum(picked > own[:, None], axis=1) + np.sum((picked == own[:, None]) & (candidates != truth[:, None]), axis=1)


def build(private_root: Path, out_dir: Path) -> int:
    import jieba
    jieba.setLogLevel(60)
    print("loading corpus v3", flush=True)
    rows, vectors, state = load_v3(private_root)
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
    corpus_docs = [documents[s] for s in songs]
    everything = np.arange(len(songs))
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    print("scoring the spaces", flush=True)
    scores = {}
    char_matrix = v1.fit_tfidf(corpus_docs)
    profiles = v1.score_leave_group_out(dense.astype(np.float32), char_matrix, label_index, group_ids, label_count)
    scores["semantic"] = profiles.dense.astype(np.float64)
    scores["lexical_char_2_5"] = profiles.lexical.astype(np.float64)
    word_matrix, _ = fit_words([" ".join(segment(d)) for d in corpus_docs])
    scores["lexical_words"] = v1.score_leave_group_out(dense.astype(np.float32), word_matrix, label_index, group_ids,
                                                       label_count).lexical.astype(np.float64)
    # fold-wise within-label whitening of the semantic space, as in the probe
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    whitened = np.zeros((len(songs), label_count))
    for k in range(FOLDS):
        train_mask = fold != k
        queries = np.flatnonzero(~train_mask)
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train_mask], label_index[train_mask],
                                        weights[train_mask], np.random.default_rng(SEED + k))
        whitened[queries] = dense_leave_group_out(unit_rows((dense - mean) @ matrix.T), label_index, group_ids,
                                                  weights, label_count, queries)
    scores["semantic_within_author_whitening"] = whitened
    scores["fusion_whitened_semantic_words"] = (v1.zscore_rows(whitened) + v1.zscore_rows(scores["lexical_words"])) / 2.0
    scores["fusion_semantic_words"] = (v1.zscore_rows(scores["semantic"]) + v1.zscore_rows(scores["lexical_words"])) / 2.0

    # content rivals: the labels nearest in the frozen semantic space, the true label excluded
    semantic_scores = scores["semantic"].copy()
    semantic_scores[everything, label_index] = -np.inf
    rival_order = np.argsort(-semantic_scores, axis=1)

    full_ranks = {name: v1.rank_system(s.astype(np.float32), label_index)[0].astype(np.int64) for name, s in scores.items()}
    report = {"all_labels": {name: round(float(np.mean(1.0 / r)), 4) for name, r in full_ranks.items()}}
    rr_by_k = {}
    for k in RIVALS:
        candidates = np.concatenate([label_index[:, None], rival_order[:, :k]], axis=1)
        chance = float(np.sum(1.0 / np.arange(1, k + 2)) / (k + 1))
        rival_ranks = {name: rank_within(s, label_index, candidates) for name, s in scores.items()}
        rr_by_k[k] = {name: 1.0 / r for name, r in rival_ranks.items()}
        report[f"among_{k}_content_rivals"] = {
            "chance_mrr": round(chance, 4),
            **{name: {"mrr": round(float(np.mean(1.0 / r)), 4), "top1": round(float(np.mean(r == 1)), 4)}
               for name, r in rival_ranks.items()}}
        print(f"  among {k} content rivals (chance {chance:.3f}):", flush=True)
        for name, r in rival_ranks.items():
            print(f"    {name:36s} MRR {np.mean(1.0 / r):.4f}  top-1 {np.mean(r == 1):.4f}   (all labels {report['all_labels'][name]:.4f})",
                  flush=True)

    pairs = [("lexical_words", "semantic"), ("lexical_words", "lexical_char_2_5"),
             ("semantic_within_author_whitening", "semantic"), ("lexical_words", "semantic_within_author_whitening"),
             ("fusion_whitened_semantic_words", "lexical_words")]
    contrasts = {}
    for k in RIVALS:
        contrasts[f"among_{k}_content_rivals"] = paired_group_bootstrap(rr_by_k[k], weights, group_ids,
                                                                        np.ones(len(songs), dtype=bool), pairs)
        for c in contrasts[f"among_{k}_content_rivals"]:
            print(f"  among {k}: {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
                  f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "identity among content rivals: the retrieval protocol's style-or-content test",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "vectors_sha256": state["sha256"],
                   "queries": len(songs), "labels": label_count, "groups": len(order)},
        "design": {"rivals": "for each query, the K labels whose leave-group-out profiles score highest in the "
                             "frozen BGE-M3 space, the true label excluded; the same set for every space",
                   "candidate_set": "the true label plus its K rivals; rank of the true label within the set, "
                                    "ties counted against the query",
                   "spaces": {"semantic": "frozen BGE-M3, cosine", "semantic_within_author_whitening":
                              "fold-wise within-label whitening as in identity_probe",
                              "lexical_char_2_5": "character 2-5-gram TF-IDF", "lexical_words": "jieba word TF-IDF",
                              "fusion_semantic_words": "z-score mean", "fusion_whitened_semantic_words": "z-score mean"},
                   "reference": "Wegmann, Schraagen and Nguyen 2022, the STEL-or-content choice"},
        "results": report,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with replacement, "
                                       "each (group, label) component weighted one", "by_rival_count": contrasts},
        "privacy": "aggregate only",
    }
    (out_dir / "content_rival_test.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'content_rival_test.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
