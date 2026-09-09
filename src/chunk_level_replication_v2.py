#!/usr/bin/env python3
"""Does the ordering of the spaces hold at the level of a passage?

Every comparison so far is between whole songs. A chunk is the corpus's passage unit --
a verse, a hook, a section as the source delivered it -- and it is a harder query: less
text, one topic, no averaging over the song. The representation study scored chunk-level
queries in the semantic space alone (0.190). This file scores the same chunk queries in
the lexical spaces too, character n-grams and jieba words, against label profiles built
from chunks outside the query's leakage group, so the three spaces can be ordered at the
passage level as they were at the song level.

Profiles are chunk-level: a label's profile is the weighted sum of its chunk vectors, each
song's weight spread evenly over its chunks so a song still counts once inside its
leakage group. Only chunks of the 7,236 query songs take part, so the query set is the
song-level one seen at passage grain (24,635 chunks).

    python src/chunk_level_replication_v2.py --private-root <ni-k>
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
from identity_probe_v2 import unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from representation_unit_v2 import tfidf  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
from build_downstream_retrieval_v2 import expected  # noqa: E402
EXPECTED_CHUNK_DENSE_MRR = expected(0.1903, None)  # no v3 chunk-level dense number has been published yet
MIN_CHUNK_CHARACTERS = 20


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

    chunk_rows = [i for s in songs for i in sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"]))]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    chunk_label, chunk_group = label_index[chunk_song], group_ids[chunk_song]
    chunk_weight = (weights[chunk_song] / np.bincount(chunk_song)[chunk_song]).astype(np.float32)
    chunk_text = [rows[i]["cleaned_text"] for i in chunk_rows]
    chunk_len = np.asarray([len(v1.normalized_text(t)) for t in chunk_text])
    print(f"  {len(chunk_rows):,} chunks from {len(songs):,} songs; "
          f"{int((chunk_len < MIN_CHUNK_CHARACTERS).sum()):,} shorter than {MIN_CHUNK_CHARACTERS} characters", flush=True)

    dense = unit_rows(vectors[chunk_rows].astype(np.float64)).astype(np.float32)
    spaces = {}
    print("character 2-5-grams over chunks", flush=True)
    # not v1.fit_tfidf: it refuses a zero row, and a short chunk whose every n-gram falls
    # under min_df has one; such chunks get the worst rank below
    spaces["lexical_char_2_5"], _ = tfidf(chunk_text, analyzer="char", ngram_range=(2, 5))
    print("jieba words over chunks", flush=True)
    spaces["lexical_words"], _ = fit_words([" ".join(segment(t)) for t in chunk_text])

    # score_leave_group_out weights each (group, label) component one; at chunk grain the
    # weights must be the chunk weights above, so the function's own weighting is bypassed by
    # scaling: it recomputes weights from counts, which would give every chunk of a song weight
    # one. Rebuild the dense branch here with the chunk weights and use the sparse branch
    # through the same routine on a per-space basis with explicit profiles.
    def leave_group_out(matrix, dense_rows: bool):
        from scipy import sparse
        sums_dense = None
        if dense_rows:
            sums = np.zeros((label_count, matrix.shape[1]), dtype=np.float64)
            np.add.at(sums, chunk_label, matrix.astype(np.float64) * chunk_weight[:, None])
        else:
            membership = sparse.csr_matrix((chunk_weight, (chunk_label, np.arange(len(chunk_rows)))),
                                           shape=(label_count, len(chunk_rows)), dtype=np.float32)
            sums = (membership @ matrix).tocsr()
        norms = (np.linalg.norm(sums, axis=1) if dense_rows
                 else np.sqrt(np.asarray(sums.multiply(sums).sum(axis=1)).ravel()))
        if dense_rows:
            scores = matrix.astype(np.float64) @ (sums / np.maximum(norms, 1e-12)[:, None]).T
        else:
            profiles = sparse.diags(1.0 / np.maximum(norms, 1e-12)) @ sums
            scores = np.asarray((matrix @ profiles.T).toarray(), dtype=np.float64)
        members_by_group: dict[int, list[int]] = defaultdict(list)
        for c, g in enumerate(chunk_group.tolist()):
            members_by_group[g].append(c)
        for g, members in members_by_group.items():
            members = np.asarray(members)
            for label in {int(chunk_label[m]) for m in members}:
                own = members[chunk_label[members] == label]
                if dense_rows:
                    leave = sums[label] - np.sum(matrix[own].astype(np.float64) * chunk_weight[own, None], axis=0)
                    norm = float(np.linalg.norm(leave))
                    scores[members, label] = (matrix[members].astype(np.float64) @ leave) / max(norm, 1e-12)
                else:
                    contribution = matrix[own].multiply(chunk_weight[own, None]).sum(axis=0)
                    leave = sums.getrow(label) - sparse.csr_matrix(contribution)
                    norm = float(np.sqrt(leave.multiply(leave).sum()))
                    scores[members, label] = np.asarray((matrix[members] @ leave.T).toarray(),
                                                        dtype=np.float64).ravel() / max(norm, 1e-12)
        return scores

    ranks = {}
    scores_by_space = {}
    print("scoring the semantic space", flush=True)
    scores_by_space["semantic"] = leave_group_out(dense, dense_rows=True)
    ranks["semantic"] = v1.rank_system(scores_by_space["semantic"].astype(np.float32), chunk_label)[0].astype(np.int64)
    mrr = float(np.mean(1.0 / ranks["semantic"]))
    print(f"  MRR {mrr:.4f}", flush=True)
    if EXPECTED_CHUNK_DENSE_MRR is not None and abs(mrr - EXPECTED_CHUNK_DENSE_MRR) > 5e-4:
        raise SystemExit(f"chunk-level dense gives {mrr:.4f}, not the representation study's {EXPECTED_CHUNK_DENSE_MRR}")
    empty = {}
    for name, matrix in spaces.items():
        print(f"scoring {name}", flush=True)
        scores_by_space[name] = leave_group_out(matrix, dense_rows=False)
        empty[name] = np.asarray(matrix.getnnz(axis=1) == 0)
        r = v1.rank_system(scores_by_space[name].astype(np.float32), chunk_label)[0].astype(np.int64)
        r[empty[name]] = label_count
        ranks[name] = r
        print(f"  MRR {float(np.mean(1.0 / r)):.4f}  ({int(empty[name].sum()):,} empty chunks ranked worst)", flush=True)

    # z-score fusion that tolerates a chunk with no lexical features: its lexical row has no
    # variance, so v1.zscore_rows would refuse; here it contributes zero and, as in the
    # lexical space alone, the chunk takes the worst rank
    def zscore(values):
        deviations = values.std(axis=1, keepdims=True)
        return np.where(deviations > 1e-12, (values - values.mean(axis=1, keepdims=True)) / np.maximum(deviations, 1e-12), 0.0)

    fused = (zscore(scores_by_space["semantic"]) + zscore(scores_by_space["lexical_words"])) / 2.0
    r = v1.rank_system(fused.astype(np.float32), chunk_label)[0].astype(np.int64)
    r[empty["lexical_words"]] = label_count
    ranks["fusion_semantic_words"] = r

    rr = {name: 1.0 / r for name, r in ranks.items()}
    long_chunks = chunk_len >= MIN_CHUNK_CHARACTERS
    report = {}
    for name, r in ranks.items():
        report[name] = {
            "all_chunks": {"mrr": round(float(np.mean(1.0 / r)), 4),
                           "recall_at_10": round(float(np.mean(r <= 10)), 4)},
            f"chunks_with_at_least_{MIN_CHUNK_CHARACTERS}_characters": {
                "mrr": round(float(np.mean(1.0 / r[long_chunks])), 4),
                "recall_at_10": round(float(np.mean(r[long_chunks] <= 10)), 4)},
            "song_mean_of_chunk_reciprocal_ranks": round(float(
                (np.bincount(chunk_song, weights=1.0 / r) / np.bincount(chunk_song)).mean()), 4),
        }
        print(f"  {name:24s} chunk MRR {report[name]['all_chunks']['mrr']:.4f}  "
              f"song-mean {report[name]['song_mean_of_chunk_reciprocal_ranks']:.4f}", flush=True)
    pairs = [("lexical_char_2_5", "semantic"), ("lexical_words", "lexical_char_2_5"),
             ("fusion_semantic_words", "lexical_words")]
    contrasts = paired_group_bootstrap(rr, chunk_weight.astype(np.float64), chunk_group,
                                       np.ones(len(chunk_rows), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "the three spaces at passage level",
        "chunks": int(len(chunk_rows)), "songs": len(songs), "labels": label_count,
        "design": ("every chunk of the 7,236 query songs is a query; label profiles are "
                   "weighted sums of chunk vectors with each song's component weight spread "
                   "over its chunks; the query's whole leakage group is left out; TF-IDF "
                   "spaces are fitted on the chunk texts"),
        "verification": {"semantic_reproduces_representation_study": round(mrr, 4)},
        "systems": report,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled "
                                       "with replacement, chunk weights summing to one per "
                                       "(group, label) component",
                             "contrasts": contrasts},
        "privacy": "aggregate only",
    }
    (out_dir / "chunk_level_replication.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'chunk_level_replication.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
