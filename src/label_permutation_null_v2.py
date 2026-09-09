#!/usr/bin/env python3
"""If the labels are shuffled, does anything survive?

Every retrieval number in this project is a rank of the true label among 226. The one
test that can show those numbers are not leaking from somewhere -- shared text the groups
missed, a profile that still contains the query, a bootstrap that resamples the wrong unit
-- is to break the only thing the numbers are supposed to depend on. Here the labels are
permuted and the full protocol is run again, unchanged, in the three raw spaces and the
word space. Under a correct protocol every space must fall to chance.

The permutation is at the level of the leakage group, because that is the unit the protocol
holds out: every song in a group receives the same permuted label, so a group of five
duplicate songs still moves as one and the holdout is exercised exactly as in the real run.
Label sizes are approximately preserved by permuting labels across groups of equal size
class. Ten permutations, each scored in every space, so the null has a spread and not just
a point.

    python src/label_permutation_null_v2.py --private-root <ni-k>
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
from identity_probe_v2 import dense_leave_group_out  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from lexical_identity_anatomy_v2 import score_arm  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
PERMUTATIONS = 10
SEED = 20260825


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
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    corpus_docs = [documents[s] for s in songs]
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    print("fitting the spaces once (labels never enter a fit)", flush=True)
    spaces = {"lexical_char_2_5": v1.fit_tfidf(corpus_docs)}
    spaces["lexical_words"], _ = fit_words([" ".join(segment(d)) for d in corpus_docs])
    chance = float(np.mean(1.0 / np.arange(1, label_count + 1)))

    def score_all(labels: np.ndarray) -> dict[str, float]:
        component_size = Counter(zip(group_ids.tolist(), labels.tolist()))
        weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, labels)])
        out = {}
        scores = dense_leave_group_out(dense.astype(np.float64), labels, group_ids, weights,
                                       label_count, np.arange(len(songs)))
        out["semantic"] = float(np.mean(1.0 / v1.rank_system(scores.astype(np.float32), labels)[0]))
        for name, matrix in spaces.items():
            ranks, error = score_arm(dense, matrix, labels, group_ids, label_count)
            out[name] = None if error else float(np.mean(1.0 / ranks))
        return out

    print("observed", flush=True)
    observed = score_all(label_index)
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in observed.items()), flush=True)

    # group-level permutation within size classes: a group's label is swapped with another
    # group's whose label has a similar number of songs, so label sizes stay close
    group_label = {}
    for g, l in zip(group_ids.tolist(), label_index.tolist()):
        group_label.setdefault(g, l)
    label_songs = np.bincount(label_index, minlength=label_count)
    size_class = np.digitize(label_songs, np.quantile(label_songs, [0.25, 0.5, 0.75]))
    nulls = []
    rng = np.random.default_rng(SEED)
    for p in range(PERMUTATIONS):
        permuted_group_label = dict(group_label)
        for klass in range(4):
            members = [g for g, l in group_label.items() if size_class[l] == klass]
            labels_here = [group_label[g] for g in members]
            rng.shuffle(labels_here)
            for g, l in zip(members, labels_here):
                permuted_group_label[g] = l
        permuted = np.asarray([permuted_group_label[int(g)] for g in group_ids], dtype=np.int64)
        # every label must keep at least two groups or a leave-group-out profile empties
        counts = Counter(zip(group_ids.tolist(), permuted.tolist()))
        per_label_groups = Counter(l for (_, l) in counts)
        if min(per_label_groups.get(l, 0) for l in range(label_count)) < 2:
            print(f"  permutation {p}: a label lost its groups; redrawing", flush=True)
            continue
        result = score_all(permuted)
        result["agreement_with_true_labels"] = float(np.mean(permuted == label_index))
        nulls.append(result)
        print(f"  permutation {p}: " + "  ".join(
            f"{k} {v:.4f}" for k, v in result.items() if v is not None), flush=True)

    summary = {}
    for name in observed:
        values = [n[name] for n in nulls if n.get(name) is not None]
        summary[name] = {"observed": round(observed[name], 4),
                         "null_mean": round(float(np.mean(values)), 4),
                         "null_max": round(float(np.max(values)), 4),
                         "null_min": round(float(np.min(values)), 4),
                         "chance": round(chance, 4),
                         "observed_over_null_max": round(observed[name] / max(np.max(values), 1e-9), 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "label-permutation null for the retrieval protocol",
        "question": ("whether any retrieval number survives when the labels are shuffled; a "
                     "correct protocol must fall to chance in every space"),
        "design": (f"{PERMUTATIONS} group-level permutations with seed {SEED}: every song in a "
                   "leakage group takes the same permuted label, labels are swapped among "
                   "groups whose true labels fall in the same size quartile, and the full "
                   "leave-group-out protocol is rerun unchanged; spaces are fitted once, "
                   "without labels"),
        "queries": len(songs), "labels": label_count, "permutations_scored": len(nulls),
        "summary": summary,
        "permutations": nulls,
        "privacy": "aggregate only",
    }
    (out_dir / "label_permutation_null.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'label_permutation_null.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
