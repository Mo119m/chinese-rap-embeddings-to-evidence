#!/usr/bin/env python3
"""Is a song's average vector the right thing to retrieve with?

The retrieval task summarises each song as the mean of its chunk vectors. That is the
conventional move and it was never tested here. It is also lossy in a way that can be
measured: over corpus v2 the median pair of chunks inside one song has cosine 0.572, and
58.8% of songs average below 0.6 internally. A song is not one thing, and averaging it makes
it a thing it is not.

This changes one variable. Label profiles stay exactly as the main result builds them --
weighted song centroids, leave-group-out, normalised -- and only the QUERY side moves, from
the song's centroid to its best-matching chunk. That is the standard passage-retrieval move,
and stating it as one variable is what makes the comparison mean anything.

The profile arithmetic is recomputed here rather than imported, because the builder returns
scores and not profile vectors. Recomputation is a risk: a subtly different profile would
make the comparison meaningless while still producing plausible numbers. So the centroid arm
is checked against v1.score_leave_group_out's own dense output before either arm is
reported, and the run stops if they disagree.

    python src/retrieval_query_unit_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"


def leave_group_profiles(centroids, label_index, group_ids, label_count):
    """The same profiles score_leave_group_out builds, returned as vectors.

    Each group contributes total weight one inside a label, so repeated variants of one song
    cannot dominate a profile -- PD-002 rule 4. For a query, its own label's profile is
    rebuilt without its whole group; every other label's profile is the full one.
    """
    song_count = len(label_index)
    group_label_counts = defaultdict(int)
    for group, label in zip(group_ids.tolist(), label_index.tolist()):
        group_label_counts[(int(group), int(label))] += 1
    weights = np.asarray(
        [1.0 / group_label_counts[(int(group_ids[i]), int(label_index[i]))]
         for i in range(song_count)], dtype=np.float64)

    sums = np.zeros((label_count, centroids.shape[1]), dtype=np.float64)
    np.add.at(sums, label_index, centroids * weights[:, None])

    members_by_group: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids.tolist()):
        members_by_group[int(group)].append(index)

    def profile_for(query: int, label: int) -> np.ndarray:
        # Every label represented in the query's group is corrected, not just the query's
        # own. A group spans labels whenever two songs credited differently share text --
        # which is what collaboration looks like in this corpus, and there are 511 such
        # label pairs. Subtracting only the query's own label leaves the collaborator's
        # profile still containing the held-out group, and the first version of this
        # function did exactly that. The self-check below is what caught it.
        vector = sums[label]
        members = [i for i in members_by_group[int(group_ids[query])]
                   if int(label_index[i]) == label]
        if members:
            vector = vector - np.sum(centroids[members] * weights[members, None], axis=0)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise SystemExit("a leave-group-out profile is empty")
        return vector / norm

    return profile_for, weights


def build(private_root: Path, out_dir: Path) -> int:
    print("loading", flush=True)
    rows, vectors, _contract = load(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, centroids_by_song = build_songs(
        rows, vectors)

    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {song for song in chunks_by_song
                   if len(v1.normalized_text(documents[song])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(label for label, members in songs_by_label.items()
                      if sum(1 for s in members if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    eligible_set = set(eligible)
    songs = sorted(s for s in long_enough if label_by_song[s] in eligible_set)
    print(f"  {len(songs):,} queries over {len(eligible)} labels", flush=True)

    normalised = {s: normalise_document(documents[s]) for s in songs}
    groups = build_groups(songs, components_by_song, normalised)
    label_index = np.asarray([eligible.index(label_by_song[s]) for s in songs], dtype=np.int64)
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)

    centroids = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    chunk_vectors = {s: vectors[sorted(chunks_by_song[s],
                                       key=lambda i: int(rows[i]["source_order"]))]
                     for s in songs}
    for song, block in chunk_vectors.items():
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        chunk_vectors[song] = block / np.maximum(norms, 1e-12)

    print("rebuilding the leave-group-out profiles", flush=True)
    profile_for, _weights = leave_group_profiles(
        centroids.astype(np.float64), label_index, group_ids, len(eligible))

    print("scoring both query units against identical profiles", flush=True)
    centroid_scores = np.zeros((len(songs), len(eligible)), dtype=np.float64)
    chunkmax_scores = np.zeros_like(centroid_scores)
    for query in range(len(songs)):
        block = chunk_vectors[songs[query]]
        for label in range(len(eligible)):
            profile = profile_for(query, label)
            centroid_scores[query, label] = float(centroids[query] @ profile)
            chunkmax_scores[query, label] = float((block @ profile).max())
        if (query + 1) % 1000 == 0:
            print(f"  {query + 1:,}/{len(songs):,}", flush=True)

    # Prove the recomputation before trusting the comparison: the centroid arm must be the
    # builder's own dense scores. A profile that differs subtly would still produce numbers.
    print("checking the centroid arm against the builder", flush=True)
    lexical = v1.fit_tfidf([documents[s] for s in songs])
    reference = v1.score_leave_group_out(centroids, lexical, label_index, group_ids,
                                         len(eligible))
    gap = float(np.abs(reference.dense.astype(np.float64) - centroid_scores).max())
    print(f"  max |recomputed - builder| = {gap:.3e}", flush=True)
    if gap > 1e-4:
        raise SystemExit("the recomputed profiles do not reproduce the builder's scores; "
                         "the comparison would be meaningless")

    report = {}
    for name, scores in (("song centroid", centroid_scores),
                         ("best-matching chunk", chunkmax_scores)):
        ranks, _ = v1.rank_system(scores.astype(np.float32), label_index)
        metrics = v1.metrics_from_ranks(ranks)
        report[name] = {metric: round(float(values.mean()), 6)
                        for metric, values in metrics.items()}
        print(f"  {name:22s} MRR {report[name]['mrr']:.4f}  "
              f"R@1 {report[name]['recall_at_1']:.4f}  "
              f"R@10 {report[name]['recall_at_10']:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "query representation unit",
        "question": ("The task summarises a song as the mean of its chunk vectors. Over "
                     "corpus v2 the median pair of chunks inside one song has cosine 0.572 "
                     "and 58.8% of songs average below 0.6 internally, so the mean is a "
                     "summary of things that are not alike."),
        "design": ("One variable. Label profiles are the ones the main result builds -- "
                   "weighted song centroids, leave-group-out, normalised -- and only the "
                   "query side changes."),
        "verification": {
            "centroid_arm_matches_builder": True,
            "max_absolute_difference": gap,
            "why": ("The profile arithmetic is recomputed here because the builder returns "
                    "scores rather than profile vectors. A subtly different profile would "
                    "produce plausible numbers and a meaningless comparison, so the centroid "
                    "arm is required to reproduce score_leave_group_out's dense output."),
        },
        "queries": len(songs),
        "labels": len(eligible),
        "metrics": report,
        "privacy": "aggregate only",
    }
    (out_dir / "query_unit_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'query_unit_comparison.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
