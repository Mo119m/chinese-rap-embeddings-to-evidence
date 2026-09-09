#!/usr/bin/env python3
"""How many of the most common words does it take to identify a label?

The word anatomy found the identity signal in the commonest quarter of the vocabulary:
that quarter alone reaches 0.515 of the full model's 0.519, and the rarest quarter can be
removed without loss. Stylometry has known a version of this since Burrows (2002): the
relative frequencies of the most common words separate authors. This file draws the
curve. For K in a fixed ladder, only the K words with the highest document frequency are
kept (unigrams only, since bigrams add nothing), rows re-normalised, and the space
rescored under the unchanged protocol. Each K also has a companion arm with the
605-surface catalogue stripped before segmentation, so names and slang cannot be what the
common words are carrying.

    python src/common_word_curve_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
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
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from lexical_identity_anatomy_v2 import masked, score_arm  # noqa: E402
from lexical_identity_decomposition_v2 import neutralise  # noqa: E402
from word_identity_anatomy_v2 import segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
LADDER = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
EXPECTED_UNIGRAM_MRR = 0.5332


def fit_unigrams(documents):
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), token_pattern=r"(?u)\S+",
                                 min_df=3, max_features=v1.TFIDF_MAX_FEATURES, sublinear_tf=True,
                                 norm="l2", dtype=np.float32)
    matrix = vectorizer.fit_transform(documents).tocsr().astype(np.float32)
    return matrix, list(vectorizer.get_feature_names_out())


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
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))]
                          for g, l in zip(group_ids, label_index)])
    print(f"  {len(songs):,} queries, {label_count} labels", flush=True)

    lexicon = private_root / "work" / "lexicon_arm_everything.csv"
    surfaces = sorted({r["entity"].strip() for r in csv.DictReader(lexicon.open(encoding="utf-8-sig"))
                       if r.get("entity", "").strip()}, key=len, reverse=True)

    results = {}
    rr = {}
    for variant, docs in (("original", corpus_docs),
                          ("neutralised", [neutralise(d, surfaces) for d in corpus_docs])):
        print(f"{variant}: segmenting and fitting unigrams", flush=True)
        matrix, features = fit_unigrams([" ".join(segment(d)) for d in docs])
        document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
        rank = np.argsort(-document_frequency, kind="stable")
        ranks, error = score_arm(dense, matrix, label_index, group_ids, label_count)
        if error:
            raise SystemExit(error)
        full = float(np.mean(1.0 / ranks))
        print(f"  all {matrix.shape[1]:,} unigrams: MRR {full:.4f}", flush=True)
        # The word anatomy's unigram arm (0.5332) is the unigram COLUMNS of a joint
        # unigram-plus-bigram fit, where 150,000 features were shared and the IDF came from
        # that fit; a unigram-only fit keeps every unigram (49,866) and lands close but not
        # on it. The check is a sanity bound on that gap, not a byte-identity claim.
        if variant == "original" and abs(full - EXPECTED_UNIGRAM_MRR) > 5e-3:
            raise SystemExit(f"unigrams give {full:.4f}, more than 0.005 from the anatomy's "
                             f"masked unigram arm {EXPECTED_UNIGRAM_MRR}")
        rr[f"{variant}_all"] = 1.0 / ranks
        results[variant] = {"features": int(matrix.shape[1]), "all_unigrams_mrr": round(full, 4),
                            "note": "a unigram-only fit; the word anatomy's masked unigram arm of "
                                    "the joint unigram-bigram fit gives 0.5332",
                            "ladder": []}
        for k in LADDER:
            if k >= matrix.shape[1]:
                continue
            keep = np.zeros(matrix.shape[1], dtype=bool)
            keep[rank[:k]] = True
            arm = masked(matrix, keep)
            ranks_k, error = score_arm(dense, arm, label_index, group_ids, label_count)
            entry = {"top_k_by_document_frequency": k,
                     "minimum_document_frequency_in_set": int(document_frequency[rank[k - 1]])}
            if error:
                entry.update({"defined": False, "why": error})
                print(f"  top {k:>6,}: undefined ({error[:40]})", flush=True)
            else:
                rr[f"{variant}_top_{k}"] = 1.0 / ranks_k
                entry.update({"defined": True, "mrr": round(float(np.mean(1.0 / ranks_k)), 4),
                              "recall_at_10": round(float(np.mean(ranks_k <= 10)), 4),
                              "empty_queries": int((np.asarray(arm.getnnz(axis=1)) == 0).sum())})
                print(f"  top {k:>6,} (df >= {entry['minimum_document_frequency_in_set']:,}): "
                      f"MRR {entry['mrr']:.4f}", flush=True)
            results[variant]["ladder"].append(entry)

    print("paired bootstrap over leakage groups", flush=True)
    pairs = [(name, "original_all") for name in rr if name != "original_all"]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    by_name = {c["system"]: c for c in contrasts}
    for variant in results:
        for entry in results[variant]["ladder"]:
            name = f"{variant}_top_{entry['top_k_by_document_frequency']}"
            if name in by_name:
                entry["difference_from_all_original_unigrams"] = by_name[name]["mrr_difference"]
                entry["ci95"] = by_name[name]["ci95"]
        if f"{variant}_all" in by_name:
            results[variant]["difference_from_all_original_unigrams"] = by_name[f"{variant}_all"]["mrr_difference"]
            results[variant]["ci95"] = by_name[f"{variant}_all"]["ci95"]

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "how many of the most common words identify a label",
        "queries": len(songs), "labels": label_count,
        "design": ("jieba unigrams, TF-IDF with the character system's settings; for each K only "
                   "the K unigrams with the highest document frequency are kept and rows "
                   "re-normalised; the neutralised variant strips the 605-surface catalogue "
                   "before segmentation"),
        "results": results,
        "bootstrap": {"design": "2000 replicates, seed 20260825, leakage groups resampled with "
                                "replacement, each (group, label) component weighted one; "
                                "difference from all original unigrams"},
        "privacy": "aggregate only; no word is published",
    }
    (out_dir / "common_word_curve.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'common_word_curve.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
