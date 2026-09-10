#!/usr/bin/env python3
"""What did the fine-tuned encoder learn, and is it new?

train_identity_encoder_v3 leaves two private chunk-vector sets for a test fold: the frozen
BGE-M3 on the label-masked text and the fine-tuned adapter on the same text, plus the
protocol MRR of each on the test fold and on the held-out labels. This file reads those
vectors back and asks the questions the rest of the project asks of every space, scored on
the test fold only (the model saw the other four folds) and on the held-out labels:

  whitening       does within-label whitening still help the tuned space, or did training
                  already do what whitening does? (whitening fitted on the training folds,
                  which the tuned model has seen -- stated, and the reason the gain is
                  read as a bound rather than a result)
  fusion          does the tuned space add to the word space, which no semantic space had
                  beaten alone? z-score fusion of the dense and word scores, tuned against
                  frozen
  chunk level     the training unit was the stanza: chunk queries against song-mean
                  profiles, tuned against frozen
  who gained      the test-fold gain split by songs the training could and could not use
                  as anchors (collaboration titles were excluded) and by song length in
                  chunks

    python src/analyse_identity_encoder_v3.py --private-root <ni-k> --test-fold 0
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
from training_data_audit_v3 import FEAT  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"


def song_level(chunk_vectors: np.ndarray, chunk_song: np.ndarray, song_count: int) -> np.ndarray:
    acc = np.zeros((song_count, chunk_vectors.shape[1]))
    np.add.at(acc, chunk_song, chunk_vectors)
    return unit_rows(acc / np.bincount(chunk_song, minlength=song_count)[:, None])


def chunk_leave_group_out(song_vec: np.ndarray, chunk_vec: np.ndarray, chunk_song: np.ndarray,
                          label_index: np.ndarray, group_ids: np.ndarray, weights: np.ndarray,
                          label_count: int, chunk_queries: np.ndarray) -> np.ndarray:
    """Chunk queries against label profiles built from song vectors, the chunk's own
    song's leakage group removed from every profile it touches."""
    sums = np.zeros((label_count, song_vec.shape[1]))
    np.add.at(sums, label_index, song_vec * weights[:, None])
    norms = np.linalg.norm(sums, axis=1)
    scores = chunk_vec[chunk_queries] @ (sums / np.maximum(norms, 1e-12)[:, None]).T
    members_by_group: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids.tolist()):
        members_by_group[int(group)].append(index)
    for row, chunk in enumerate(chunk_queries.tolist()):
        song = int(chunk_song[chunk])
        members = members_by_group[int(group_ids[song])]
        for label in {int(label_index[m]) for m in members}:
            own = [m for m in members if int(label_index[m]) == label]
            leave = sums[label] - np.sum(song_vec[own] * weights[own, None], axis=0)
            norm = float(np.linalg.norm(leave))
            if norm <= 1e-12:
                raise RuntimeError("a held-out group empties a label profile")
            scores[row, label] = float(chunk_vec[chunk] @ leave) / norm
    return scores


def ranks_of(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return v1.rank_system(scores.astype(np.float32), truth)[0].astype(np.int64)


def summary(ranks: np.ndarray) -> dict:
    return {"queries": int(len(ranks)), "mrr": round(float(np.mean(1.0 / ranks)), 4),
            "recall_at_1": round(float(np.mean(ranks <= 1)), 4), "recall_at_10": round(float(np.mean(ranks <= 10)), 4)}


def build(private_root: Path, out_dir: Path, test_fold: int, held_out_share: float) -> int:
    import jieba
    jieba.setLogLevel(60)
    print("loading corpus v3", flush=True)
    rows, vectors, state = load_v3(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, _ = build_songs(rows, vectors)
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
    song_pos = {s: i for i, s in enumerate(songs)}
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    held = np.zeros(label_count, dtype=bool)
    held[np.random.default_rng(SEED).permutation(label_count)[:int(round(held_out_share * label_count))]] = True
    titles = {r["song_id"]: r["song_title"] for r in rows}
    chunk_rows = [i for s in songs for i in sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"]))]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    collaboration = np.asarray([bool(FEAT.search(titles[s])) for s in songs])
    print(f"  {len(songs):,} query songs, {label_count} labels, {len(order):,} groups, {len(chunk_rows):,} chunks; "
          f"test fold {test_fold}, {int(held.sum())} labels held out", flush=True)

    private = private_root / "work" / "private-identity-encoder-v3"
    tuned_chunks = np.load(private / f"fine_tuned_chunk_vectors_fold{test_fold}.npy").astype(np.float64)
    frozen_chunks = np.load(private / f"frozen_masked_chunk_vectors_fold{test_fold}.npy").astype(np.float64)
    if tuned_chunks.shape[0] != len(chunk_rows) or frozen_chunks.shape[0] != len(chunk_rows):
        raise SystemExit(f"the saved chunk vectors ({tuned_chunks.shape[0]}) do not match the {len(chunk_rows):,} "
                         f"chunks of the current corpus build; retrain on this build")
    spaces = {"frozen_masked": song_level(frozen_chunks, chunk_song, len(songs)),
              "fine_tuned": song_level(tuned_chunks, chunk_song, len(songs))}

    test = np.flatnonzero((fold == test_fold) & ~held[label_index])
    unseen = np.flatnonzero(held[label_index])
    train_mask = (fold != test_fold) & ~held[label_index]
    scopes = {"test_fold": test, "unseen_labels": unseen}
    print(f"  scoring {len(test):,} test-fold and {len(unseen):,} unseen-label queries", flush=True)

    # 1. cosine, then whitening fitted on the training folds (seen by the tuned model)
    scores: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for name, song_vec in spaces.items():
        for scope, queries in scopes.items():
            scores[name][scope] = dense_leave_group_out(song_vec, label_index, group_ids, weights, label_count, queries)
        for transform in ("total_whitening", "within_author_whitening"):
            mean, matrix, _ = fit_transform(transform, song_vec[train_mask], label_index[train_mask],
                                            weights[train_mask], np.random.default_rng(SEED + test_fold))
            projected = unit_rows((song_vec - mean) @ matrix.T)
            for scope, queries in scopes.items():
                scores[f"{name}_{transform}"][scope] = dense_leave_group_out(projected, label_index, group_ids,
                                                                             weights, label_count, queries)

    # 2. the word space and fusions
    print("fitting the word space", flush=True)
    word_matrix, _ = fit_words([" ".join(segment(documents[s])) for s in songs])
    lexical_full = v1.score_leave_group_out(spaces["frozen_masked"].astype(np.float32), word_matrix,
                                            label_index, group_ids, label_count).lexical.astype(np.float64)
    for scope, queries in scopes.items():
        scores["words"][scope] = lexical_full[queries]
        for name in ("frozen_masked", "fine_tuned"):
            scores[f"fusion_{name}_words"][scope] = (v1.zscore_rows(scores[name][scope])
                                                     + v1.zscore_rows(lexical_full[queries])) / 2.0

    # 3. chunk queries
    print("chunk-level queries", flush=True)
    chunk_scope = {scope: np.flatnonzero(np.isin(chunk_song, queries)) for scope, queries in scopes.items()}
    chunk_ranks = {}
    for name, chunk_vec in (("frozen_masked", frozen_chunks), ("fine_tuned", tuned_chunks)):
        for scope, chunk_queries in chunk_scope.items():
            s = chunk_leave_group_out(spaces[name], chunk_vec, chunk_song, label_index, group_ids, weights,
                                      label_count, chunk_queries)
            chunk_ranks[(name, scope)] = ranks_of(s, label_index[chunk_song[chunk_queries]])

    # ranks, reports, bootstrap
    ranks = {name: {scope: ranks_of(s, label_index[scopes[scope]]) for scope, s in by_scope.items()}
             for name, by_scope in scores.items()}
    report = {scope: {name: summary(ranks[name][scope]) for name in ranks} for scope in scopes}
    for scope in scopes:
        print(f"{scope}", flush=True)
        for name in ranks:
            r = report[scope][name]
            print(f"  {name:44s} MRR {r['mrr']:.4f}  R@1 {r['recall_at_1']:.4f}  R@10 {r['recall_at_10']:.4f}", flush=True)
    chunk_report = {scope: {name: summary(chunk_ranks[(name, scope)]) for name in ("frozen_masked", "fine_tuned")}
                    for scope in scopes}
    for scope in scopes:
        for name in ("frozen_masked", "fine_tuned"):
            r = chunk_report[scope][name]
            print(f"  chunk queries, {scope}, {name:14s} MRR {r['mrr']:.4f}  R@10 {r['recall_at_10']:.4f}", flush=True)

    pairs = [("fine_tuned", "frozen_masked"),
             ("fine_tuned_within_author_whitening", "fine_tuned"),
             ("fine_tuned_within_author_whitening", "frozen_masked_within_author_whitening"),
             ("fine_tuned", "frozen_masked_within_author_whitening"),
             ("fusion_fine_tuned_words", "fusion_frozen_masked_words"),
             ("fusion_fine_tuned_words", "words"),
             ("fine_tuned", "words")]
    contrasts = {}
    for scope, queries in scopes.items():
        mask = np.zeros(len(songs), dtype=bool)
        mask[queries] = True
        rr = {}
        for name in ranks:
            full = np.ones(len(songs))
            full[queries] = 1.0 / ranks[name][scope]
            rr[name] = full
        contrasts[scope] = paired_group_bootstrap(rr, weights, group_ids, mask, pairs)
        # chunk-level contrast, chunks aggregated to their song before the group bootstrap
        chunk_rr = {}
        for name in ("frozen_masked", "fine_tuned"):
            per_song = np.zeros(len(songs))
            counts = np.zeros(len(songs))
            np.add.at(per_song, chunk_song[chunk_scope[scope]], 1.0 / chunk_ranks[(name, scope)])
            np.add.at(counts, chunk_song[chunk_scope[scope]], 1.0)
            full = np.ones(len(songs))
            full[queries] = per_song[queries] / np.maximum(counts[queries], 1.0)
            chunk_rr[f"chunk_{name}"] = full
        contrasts[scope] += paired_group_bootstrap(chunk_rr, weights, group_ids, mask, [("chunk_fine_tuned", "chunk_frozen_masked")])
        for c in contrasts[scope]:
            print(f"  {scope}: {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
                  f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    # 4. who gained, on the test fold
    gain = 1.0 / ranks["fine_tuned"]["test_fold"] - 1.0 / ranks["frozen_masked"]["test_fold"]
    chunk_count = np.bincount(chunk_song, minlength=len(songs))[test]
    subgroups = {
        "collaboration_title": collaboration[test],
        "single_artist_title": ~collaboration[test],
        "one_or_two_chunks": chunk_count <= 2,
        "three_to_five_chunks": (chunk_count >= 3) & (chunk_count <= 5),
        "six_or_more_chunks": chunk_count >= 6,
    }
    who_gained = {}
    for name, mask in subgroups.items():
        if mask.sum() == 0:
            continue
        who_gained[name] = {"queries": int(mask.sum()),
                            "frozen_mrr": round(float(np.mean(1.0 / ranks["frozen_masked"]["test_fold"][mask])), 4),
                            "fine_tuned_mrr": round(float(np.mean(1.0 / ranks["fine_tuned"]["test_fold"][mask])), 4),
                            "mean_gain": round(float(gain[mask].mean()), 4)}
        print(f"  {name:22s} n {who_gained[name]['queries']:5d}  frozen {who_gained[name]['frozen_mrr']:.4f}  "
              f"tuned {who_gained[name]['fine_tuned_mrr']:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "what the fine-tuned identity encoder learned, read with the project's own tools",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count,
                   "groups": len(order), "chunks": len(chunk_rows)},
        "design": {"test_fold": test_fold, "held_out_labels": int(held.sum()),
                   "scopes": {k: int(len(v)) for k, v in scopes.items()},
                   "whitening": "fitted on the training folds' non-held songs, which the tuned model was trained "
                                "on; a bound on what whitening adds, not an unseen-data result",
                   "words": "jieba word TF-IDF fitted over all query songs, the protocol's lexical scorer",
                   "fusion": "z-score mean of the dense and word score rows",
                   "chunk_queries": "each chunk scored on its own against song-mean label profiles, the chunk's "
                                    "song's leakage group left out; bootstrap over songs after averaging chunks"},
        "systems": report,
        "chunk_queries": chunk_report,
        "who_gained_test_fold": who_gained,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with replacement, "
                                       "each (group, label) component weighted one", "by_scope": contrasts},
        "privacy": "aggregate only",
    }
    (out_dir / f"identity_encoder_analysis_fold{test_fold}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / f'identity_encoder_analysis_fold{test_fold}.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--held-out-label-share", type=float, default=0.15)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir, args.test_fold, args.held_out_label_share)


if __name__ == "__main__":
    sys.exit(main())
