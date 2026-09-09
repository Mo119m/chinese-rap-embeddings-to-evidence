#!/usr/bin/env python3
"""How a song is represented is a choice, and each choice is tested here.

Every result so far represents a song as one vector: the mean of its chunk vectors in the
semantic space, one TF-IDF row over character 2-5-grams in the lexical space. The first
query-unit check found that scoring a song by its best-matching chunk against centroid
profiles is worse than the mean. That left the label side untested, and the lexical
tokenisation untested. This file changes one thing at a time and rescores under the
unchanged leave-group-out protocol.

Semantic arms
  song_centroid        the published representation; must reproduce 0.318
  chunk_maxsim         no averaging on either side: a label is the set of its songs'
                       chunk vectors, a query song scores a label by the mean over its
                       chunks of the best cosine to any of that label's chunks outside
                       the query's leakage group -- the late-interaction reading of
                       "does this song sound like anything that label wrote"
  chunk_query          every chunk is its own query against the centroid profiles; a
                       chunk is a passage, so this is the passage-level task, and its
                       ranks are aggregated to songs by the mean reciprocal rank of the
                       song's chunks
  whitened_chunks      within-author whitening fitted on chunk vectors (five folds by
                       leakage group), the whitened chunks averaged into songs -- does it
                       matter whether the geometry is corrected before or after averaging

Lexical arms
  char_2_5             the published representation; must reproduce 0.450
  char_1               single characters only: a vocabulary of characters, no order
  char_2_3 / char_4_5  short and long n-grams alone
  word_jieba           jieba word segmentation, unigram and bigram words, the same TF-IDF
                       settings -- if words score below characters, the identity signal
                       is sub-word

All arms score the same 7,236 songs against the same 226 labels over the same 5,888
leakage groups. Contrasts carry paired leakage-group bootstrap intervals.

    python src/representation_unit_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import (  # noqa: E402
    MINIMUM_SONGS_PER_LABEL,
    build_songs,
    load,
)
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
from build_downstream_retrieval_v2 import expected  # noqa: E402
EXPECTED_DENSE_MRR = expected(0.3181, 0.2995)
EXPECTED_LEXICAL_MRR = expected(0.4503, 0.4267)


def tfidf(documents, **kwargs):
    vectorizer = TfidfVectorizer(min_df=3, max_features=v1.TFIDF_MAX_FEATURES, sublinear_tf=True,
                                 norm="l2", dtype=np.float32, **kwargs)
    matrix = vectorizer.fit_transform(documents).tocsr().astype(np.float32)
    return matrix, int(matrix.shape[1])


def lexical_ranks(dense, lexical, label_index, group_ids, label_count):
    profiles = v1.score_leave_group_out(dense, lexical, label_index, group_ids, label_count)
    ranks, _ = v1.rank_system(profiles.lexical, label_index)
    ranks = ranks.astype(np.int64)
    ranks[np.asarray(lexical.getnnz(axis=1) == 0)] = label_count
    return ranks


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
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))]
                          for g, l in zip(group_ids, label_index)])
    song_pos = {s: i for i, s in enumerate(songs)}
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    # chunk-level arrays, in song order
    chunk_rows = [i for s in songs for i in sorted(chunks_by_song[s],
                                                    key=lambda i: int(rows[i]["source_order"]))]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    chunk_vec = unit_rows(vectors[chunk_rows].astype(np.float64))
    chunk_label = label_index[chunk_song]
    chunk_group = group_ids[chunk_song]
    chunk_weight = weights[chunk_song] / np.bincount(chunk_song)[chunk_song]  # a song's weight spread over its chunks
    print(f"  {len(chunk_rows):,} chunks", flush=True)

    ranks: dict[str, np.ndarray] = {}
    notes: dict[str, dict] = {}
    everything = np.arange(len(songs))

    # ------------------------------------------------------------ semantic: centroid
    print("song_centroid", flush=True)
    scores = dense_leave_group_out(dense.astype(np.float64), label_index, group_ids, weights,
                                   label_count, everything)
    ranks["song_centroid"] = v1.rank_system(scores.astype(np.float32), label_index)[0].astype(np.int64)
    mrr = float(np.mean(1.0 / ranks["song_centroid"]))
    print(f"  MRR {mrr:.4f}", flush=True)
    if abs(mrr - EXPECTED_DENSE_MRR) > 5e-4:
        raise SystemExit(f"song centroid gives {mrr:.4f}, not {EXPECTED_DENSE_MRR}")

    # ------------------------------------------------------------ semantic: chunk MaxSim
    print("chunk_maxsim", flush=True)
    label_order = np.argsort(chunk_label, kind="stable")
    sorted_label = chunk_label[label_order]
    starts = np.searchsorted(sorted_label, np.arange(label_count))
    ends = np.searchsorted(sorted_label, np.arange(label_count), side="right")
    chunk_sorted = chunk_vec[label_order]
    group_sorted = chunk_group[label_order]
    maxsim = np.zeros((len(songs), label_count))
    members_by_song: dict[int, list[int]] = defaultdict(list)
    for c, s in enumerate(chunk_song.tolist()):
        members_by_song[s].append(c)
    for q in range(len(songs)):
        block = chunk_vec[members_by_song[q]]                      # query chunks
        sims = block @ chunk_sorted.T                              # chunks x all chunks
        sims[:, group_sorted == group_ids[q]] = -np.inf            # leave the group out
        best = np.full((block.shape[0], label_count), -np.inf)
        for l in range(label_count):
            if ends[l] > starts[l]:
                best[:, l] = sims[:, starts[l]:ends[l]].max(axis=1)
        best[~np.isfinite(best)] = -1.0                            # a label emptied by the holdout
        maxsim[q] = best.mean(axis=0)
        if (q + 1) % 1000 == 0:
            print(f"  {q + 1:,}/{len(songs):,}", flush=True)
    ranks["chunk_maxsim"] = v1.rank_system(maxsim.astype(np.float32), label_index)[0].astype(np.int64)
    print(f"  MRR {float(np.mean(1.0 / ranks['chunk_maxsim'])):.4f}", flush=True)

    # ------------------------------------------------------------ semantic: chunk queries
    print("chunk_query", flush=True)
    chunk_scores = dense_leave_group_out(chunk_vec, chunk_label, chunk_group, chunk_weight,
                                         label_count, np.arange(len(chunk_rows)))
    chunk_ranks = v1.rank_system(chunk_scores.astype(np.float32), chunk_label)[0].astype(np.int64)
    per_song_rr = np.zeros(len(songs))
    np.add.at(per_song_rr, chunk_song, 1.0 / chunk_ranks)
    per_song_rr /= np.bincount(chunk_song)
    notes["chunk_query"] = {"chunks": int(len(chunk_rows)),
                            "chunk_level_mrr": round(float(np.mean(1.0 / chunk_ranks)), 4),
                            "chunk_level_recall_at_10": round(float(np.mean(chunk_ranks <= 10)), 4),
                            "song_level_mean_of_chunk_reciprocal_ranks": round(float(per_song_rr.mean()), 4)}
    print(f"  chunk-level MRR {notes['chunk_query']['chunk_level_mrr']:.4f}; "
          f"song mean of chunk RR {notes['chunk_query']['song_level_mean_of_chunk_reciprocal_ranks']:.4f}", flush=True)

    # ------------------------------------------------------------ semantic: whiten chunks first
    print("whitened_chunks (within-author, five folds by leakage group)", flush=True)
    rng = np.random.default_rng(SEED)
    fold_of_group = rng.integers(0, FOLDS, size=len(order))
    fold = fold_of_group[group_ids]
    chunk_fold = fold[chunk_song]
    whitened_scores = np.zeros((len(songs), label_count))
    centroid_scores_after = np.zeros((len(songs), label_count))
    for k in range(FOLDS):
        train = chunk_fold != k
        mean, matrix, _ = fit_transform("within_author_whitening", chunk_vec[train],
                                        chunk_label[train], chunk_weight[train],
                                        np.random.default_rng(SEED + k))
        projected = unit_rows((chunk_vec - mean) @ matrix.T)
        song_vec = np.zeros((len(songs), projected.shape[1]))
        np.add.at(song_vec, chunk_song, projected)
        song_vec = unit_rows(song_vec / np.bincount(chunk_song)[:, None])
        queries = np.flatnonzero(fold == k)
        whitened_scores[queries] = dense_leave_group_out(song_vec, label_index, group_ids, weights,
                                                         label_count, queries)
        # the published probe for comparison: whiten the centroids instead
        mean_c, matrix_c, _ = fit_transform("within_author_whitening", dense[fold != k].astype(np.float64),
                                            label_index[fold != k], weights[fold != k],
                                            np.random.default_rng(SEED + k))
        centroid_scores_after[queries] = dense_leave_group_out(
            unit_rows((dense.astype(np.float64) - mean_c) @ matrix_c.T), label_index, group_ids,
            weights, label_count, queries)
        print(f"  fold {k}", flush=True)
    ranks["whitened_chunks_then_mean"] = v1.rank_system(whitened_scores.astype(np.float32), label_index)[0].astype(np.int64)
    ranks["whitened_centroids"] = v1.rank_system(centroid_scores_after.astype(np.float32), label_index)[0].astype(np.int64)
    for name in ("whitened_chunks_then_mean", "whitened_centroids"):
        print(f"  {name}: MRR {float(np.mean(1.0 / ranks[name])):.4f}", flush=True)

    # ------------------------------------------------------------ lexical tokenisations
    corpus_docs = [documents[s] for s in songs]
    print("char_2_5", flush=True)
    lexical = v1.fit_tfidf(corpus_docs)
    ranks["char_2_5"] = lexical_ranks(dense, lexical, label_index, group_ids, label_count)
    mrr = float(np.mean(1.0 / ranks["char_2_5"]))
    print(f"  MRR {mrr:.4f}", flush=True)
    if abs(mrr - EXPECTED_LEXICAL_MRR) > 5e-4:
        raise SystemExit(f"char 2-5 gives {mrr:.4f}, not {EXPECTED_LEXICAL_MRR}")
    features = {"char_2_5": int(lexical.shape[1])}
    for name, kwargs in (("char_1", {"analyzer": "char", "ngram_range": (1, 1)}),
                         ("char_2_3", {"analyzer": "char", "ngram_range": (2, 3)}),
                         ("char_4_5", {"analyzer": "char", "ngram_range": (4, 5)})):
        print(name, flush=True)
        matrix, features[name] = tfidf(corpus_docs, **kwargs)
        ranks[name] = lexical_ranks(dense, matrix, label_index, group_ids, label_count)
        print(f"  {features[name]:,} features, MRR {float(np.mean(1.0 / ranks[name])):.4f}", flush=True)

    print("word_jieba", flush=True)
    import jieba  # noqa: E402  (imported here so the semantic arms do not need it)
    jieba.setLogLevel(60)
    segmented = [" ".join(t for t in jieba.cut(doc) if t.strip()) for doc in corpus_docs]
    matrix, features["word_jieba"] = tfidf(segmented, analyzer="word", ngram_range=(1, 2),
                                           token_pattern=r"(?u)\S+")
    ranks["word_jieba"] = lexical_ranks(dense, matrix, label_index, group_ids, label_count)
    print(f"  {features['word_jieba']:,} features, MRR {float(np.mean(1.0 / ranks['word_jieba'])):.4f}", flush=True)

    # ------------------------------------------------------------ report
    rr = {name: 1.0 / r for name, r in ranks.items()}
    report = {name: {"mrr": round(float(np.mean(v)), 4),
                     "recall_at_1": round(float(np.mean(ranks[name] <= 1)), 4),
                     "recall_at_10": round(float(np.mean(ranks[name] <= 10)), 4)}
              for name, v in rr.items()}
    for name in features:
        report[name]["features"] = features[name]
    pairs = [("chunk_maxsim", "song_centroid"),
             ("whitened_chunks_then_mean", "whitened_centroids"),
             ("whitened_centroids", "song_centroid"),
             ("char_1", "char_2_5"), ("char_2_3", "char_2_5"), ("char_4_5", "char_2_5"),
             ("word_jieba", "char_2_5")]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "the representation of a song, one choice at a time",
        "queries": len(songs), "labels": label_count, "groups": len(order),
        "chunks": int(len(chunk_rows)),
        "verification": {"song_centroid_reproduces_builder": round(report["song_centroid"]["mrr"], 4),
                         "char_2_5_reproduces_builder": round(report["char_2_5"]["mrr"], 4)},
        "systems": report,
        "chunk_query": notes["chunk_query"],
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled "
                                       "with replacement, each (group, label) component weighted one",
                             "contrasts": contrasts},
        "definitions": {
            "chunk_maxsim": "label = the set of its chunk vectors outside the query's leakage "
                            "group; score = mean over the query's chunks of the best cosine to "
                            "any of the label's chunks",
            "chunk_query": "each chunk scored alone against centroid profiles built from all "
                           "songs outside its leakage group; reported at chunk level and as the "
                           "mean reciprocal rank of a song's chunks",
            "whitened_chunks_then_mean": "within-author whitening fitted on chunk vectors of four "
                                         "folds, applied to every chunk, then averaged per song",
            "whitened_centroids": "the published probe: whitening fitted on and applied to song "
                                  "centroids, same folds",
            "word_jieba": "jieba default segmentation, word unigrams and bigrams, TF-IDF with the "
                          "character system's settings",
        },
        "privacy": "aggregate only",
    }
    (out_dir / "representation_unit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'representation_unit.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
