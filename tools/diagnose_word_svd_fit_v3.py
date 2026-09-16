#!/usr/bin/env python3
"""Where a word SVD is fitted changes what whitened word vectors identify (fold 0).

The within-author-whitened word SVD gave 0.40 on fold 0's test songs but 0.54 on the unseen
labels in a first version of finetune_learned_identity_v3. One factor per arm: which songs the
SVD is fitted on, and which songs the whitening is fitted on. Arm C fits the SVD on all songs
(no labels are used by an SVD); arm D fits the whitening with the held-out labels included, which
leaks their labels and is shown only to size that leak. Prints MRRs; writes nothing.

    python tools/diagnose_word_svd_fit_v3.py --private-root <ni-k>
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

import jieba
jieba.setLogLevel(60)
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--private-root", type=Path, required=True)
args = parser.parse_args()
rows, vectors, _ = load_v3(args.private_root.resolve())
chunks_by_song, label_by_song, components_by_song, documents, _ = build_songs(rows, vectors)
songs_by_label = defaultdict(list)
for s, l in label_by_song.items():
    songs_by_label[l].append(s)
long_enough = {s for s in chunks_by_song if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
eligible = sorted(l for l, m in songs_by_label.items() if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
label_index = np.asarray([eligible.index(label_by_song[s]) for s in songs], dtype=np.int64)
groups = build_groups(songs, components_by_song, {s: normalise_document(documents[s]) for s in songs})
order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
L = len(eligible)
comp = Counter(zip(group_ids.tolist(), label_index.tolist()))
weights = np.asarray([1.0 / comp[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
held = np.zeros(L, dtype=bool)
held[np.random.default_rng(SEED).permutation(L)[:int(round(0.15 * L))]] = True
k = 0
test = np.flatnonzero((fold == k) & ~held[label_index])
unseen = np.flatnonzero(held[label_index])
train_seen = (fold != k) & ~held[label_index]
train_all = fold != k

word_matrix, _ = fit_words([" ".join(segment(documents[s])) for s in songs])


def mrr(space, queries):
    return float(np.mean(1.0 / ranks_of(dense_leave_group_out(space, label_index, group_ids, weights, L, queries), label_index[queries])))


def arm(name, svd_rows, whiten_rows):
    svd = TruncatedSVD(n_components=1024, random_state=SEED + k).fit(word_matrix[svd_rows])
    reduced = unit_rows(svd.transform(word_matrix))
    raw = (mrr(reduced, test), mrr(reduced, unseen))
    mean, matrix, _ = fit_transform("within_author_whitening", reduced[whiten_rows], label_index[whiten_rows],
                                    weights[whiten_rows], np.random.default_rng(SEED + k))
    wh = unit_rows((reduced - mean) @ matrix.T)
    white = (mrr(wh, test), mrr(wh, unseen))
    print(f"{name:52s} SVD raw test {raw[0]:.4f} unseen {raw[1]:.4f} | whitened test {white[0]:.4f} unseen {white[1]:.4f}", flush=True)


print(f"test {len(test)}, unseen {len(unseen)}, train seen-only {int(train_seen.sum())}, train all labels {int(train_all.sum())}")
lex = v1.score_leave_group_out(unit_rows(np.random.default_rng(0).standard_normal((len(songs), 8))).astype(np.float32),
                               word_matrix, label_index, group_ids, L).lexical
r = v1.rank_system(lex, label_index)[0]
print(f"raw word TF-IDF: test {np.mean(1.0 / r[test]):.4f}  unseen {np.mean(1.0 / r[unseen]):.4f}  (analysis file: 0.5145 / 0.4233)")
arm("A: SVD on seen-label training songs, whiten on the same", train_seen, train_seen)
arm("B: SVD on all training-fold songs, whiten on seen-label", train_all, train_seen)
arm("C: SVD on all songs (unsupervised), whiten on seen-label", np.ones(len(songs), bool), train_seen)
arm("D: SVD on seen-label training, whiten on all training-fold", train_seen, train_all)
