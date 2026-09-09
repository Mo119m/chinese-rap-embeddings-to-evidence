#!/usr/bin/env python3
"""Which words identify a label?

The representation study found that jieba-segmented words with word bigrams identify a
source-credit label better than any character n-gram space (0.519 against 0.450), and the
character anatomy found the signal in short Han strings that contain a less common
character. Both point at vocabulary choice. This file asks which vocabulary, in the only
terms that can be published: part of speech, frequency band, script, and whether the word
is one of the 605 named references and slang terms the neutralisation arms strip.

Every word in the vocabulary is tagged with jieba's part-of-speech guess, taken as a
description of the vocabulary rather than a truth about each use, and grouped: content
words (nouns, verbs, adjectives, other proper nouns), person and place names, function
words (pronouns, prepositions, particles, conjunctions, modal particles, auxiliaries),
adverbs, English tokens, numerals, and punctuation or symbols. A word bigram takes the
class of its rarer member and is also tracked as its own class. Frequency bands split the
vocabulary by document frequency into quartiles.

Three kinds of arm, all under the unchanged protocol: remove a class and re-normalise;
keep only the class; and the neutralised arm, which strips the 605-surface catalogue from
the text before segmentation, to check that the word advantage does not come from names.
The full model is required to reproduce the representation study's 0.5188.

    python src/word_identity_anatomy_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
from lexical_identity_decomposition_v2 import HAN, LATIN, lexical_profiles, neutralise  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
from build_downstream_retrieval_v2 import expected  # noqa: E402
EXPECTED_WORD_MRR = expected(0.5188, "lexical_words")

POS_GROUPS = {
    "content": {"n", "nz", "nt", "ng", "nl", "v", "vd", "vn", "vg", "vi", "vl", "a", "ad",
                "an", "ag", "al", "i", "l", "j", "b", "z", "s", "t", "tg", "f"},
    "person_or_place_name": {"nr", "nrfg", "nrt", "ns", "nsf"},
    "function": {"r", "rr", "rz", "rg", "p", "u", "ud", "ug", "uj", "ul", "uv", "uz", "c",
                 "y", "e", "o", "h", "k", "q"},
    "adverb": {"d", "dg", "df"},
    "english": {"eng"},
    "numeral": {"m", "mq"},
    "punctuation_or_symbol": {"x", "w"},
}


def segment(document: str) -> list[str]:
    import jieba
    return [t for t in jieba.cut(document) if t.strip()]


def fit_words(documents):
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\S+",
                                 min_df=3, max_features=v1.TFIDF_MAX_FEATURES, sublinear_tf=True,
                                 norm="l2", dtype=np.float32)
    matrix = vectorizer.fit_transform(documents).tocsr().astype(np.float32)
    return matrix, list(vectorizer.get_feature_names_out())


def pos_group(tag: str, word: str) -> str:
    for group, tags in POS_GROUPS.items():
        if tag in tags:
            return group
    if LATIN.fullmatch(word.replace(" ", "")) or re.fullmatch(r"[A-Za-z0-9'\-]+", word):
        return "english"
    if not HAN.search(word):
        return "punctuation_or_symbol"
    return "content"


def build(private_root: Path, out_dir: Path) -> int:
    import jieba
    import jieba.posseg as posseg
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

    print("segmenting and fitting the word space", flush=True)
    segmented = [" ".join(segment(d)) for d in corpus_docs]
    lexical, features = fit_words(segmented)
    full_ranks, error = score_arm(dense, lexical, label_index, group_ids, label_count)
    if error:
        raise SystemExit(error)
    full_mrr = float(np.mean(1.0 / full_ranks))
    print(f"  {lexical.shape[1]:,} features, MRR {full_mrr:.4f}", flush=True)
    if EXPECTED_WORD_MRR is not None and abs(full_mrr - EXPECTED_WORD_MRR) > 5e-4:
        raise SystemExit(f"the word space gives {full_mrr:.4f}, not {EXPECTED_WORD_MRR}")

    # ------------------------------------------------------------ tag the vocabulary
    print("tagging the vocabulary", flush=True)
    unigram_tag: dict[str, str] = {}
    for feature in features:
        for word in feature.split(" "):
            if word not in unigram_tag:
                pairs = list(posseg.cut(word))
                tag = pairs[0].flag if len(pairs) == 1 else "x" if not pairs else Counter(
                    p.flag for p in pairs).most_common(1)[0][0]
                unigram_tag[word] = pos_group(tag, word)
    document_frequency = np.asarray((lexical > 0).sum(axis=0)).ravel()
    unigram_df = {f: int(document_frequency[i]) for i, f in enumerate(features) if " " not in f}
    is_bigram = np.asarray([" " in f for f in features])

    def feature_group(feature: str) -> str:
        words = feature.split(" ")
        if len(words) == 1:
            return unigram_tag[feature]
        # a bigram takes the class of its rarer member (the one carrying more information)
        rarer = min(words, key=lambda w: unigram_df.get(w, 0))
        return unigram_tag[rarer]

    group_of = np.asarray([feature_group(f) for f in features])
    classes: dict[str, np.ndarray] = {f"pos_{g}": group_of == g for g in POS_GROUPS}
    classes["word_bigram"] = is_bigram
    classes["word_unigram"] = ~is_bigram
    quartiles = np.quantile(document_frequency, [0.25, 0.5, 0.75])
    band = np.digitize(document_frequency, quartiles)
    for b, name in enumerate(("rarest_quartile", "second_quartile", "third_quartile", "commonest_quartile")):
        classes[f"df_{name}"] = band == b
    inventory = {name: int(mask.sum()) for name, mask in classes.items()}
    print("  " + ", ".join(f"{k} {v:,}" for k, v in inventory.items()), flush=True)

    # ------------------------------------------------------------ shares
    print("share of the correct label's score by class", flush=True)
    profile_for = lexical_profiles(lexical, label_index, group_ids, label_count)
    class_names = list(classes)
    class_matrix = np.stack([classes[n] for n in class_names]).astype(np.float64)
    totals = np.zeros(len(class_names))
    grand = 0.0
    for q in range(len(songs)):
        profile = profile_for(q, int(label_index[q]))
        contribution = lexical.getrow(q).astype(np.float64).multiply(profile)
        total = float(contribution.sum())
        if total <= 0:
            continue
        totals += (class_matrix[:, contribution.indices] @ contribution.data) / total
        grand += 1.0
        if (q + 1) % 2000 == 0:
            print(f"  {q + 1:,}/{len(songs):,}", flush=True)
    shares = {name: round(float(totals[i] / grand), 4) for i, name in enumerate(class_names)}

    # ------------------------------------------------------------ arms
    arms = {}
    rr = {"full": 1.0 / full_ranks}
    for kind, name in [("remove", n) for n in class_names] + [("only", n) for n in class_names]:
        keep = ~classes[name] if kind == "remove" else classes[name]
        arm = f"{kind}_{name}"
        print(f"{arm}: {int(keep.sum()):,} features kept", flush=True)
        ranks, error = score_arm(dense, masked(lexical, keep), label_index, group_ids, label_count)
        if error:
            arms[arm] = {"kind": kind, "class": name, "features_kept": int(keep.sum()),
                         "defined": False, "why": error}
            print(f"  undefined: {error}", flush=True)
            continue
        rr[arm] = 1.0 / ranks
        arms[arm] = {"kind": kind, "class": name, "features_kept": int(keep.sum()), "defined": True,
                     "mrr": round(float(np.mean(rr[arm])), 4),
                     "recall_at_10": round(float(np.mean(ranks <= 10)), 4)}
        print(f"  MRR {arms[arm]['mrr']:.4f}  R@10 {arms[arm]['recall_at_10']:.4f}", flush=True)

    # the neutralised arm: strip the catalogue, segment again, refit, rescore
    lexicon = private_root / "work" / "lexicon_arm_everything.csv"
    surfaces = sorted({r["entity"].strip() for r in csv.DictReader(lexicon.open(encoding="utf-8-sig"))
                       if r.get("entity", "").strip()}, key=len, reverse=True)
    print(f"neutralised_words: {len(surfaces)} surfaces stripped before segmentation", flush=True)
    neutral_docs = [" ".join(segment(neutralise(d, surfaces))) for d in corpus_docs]
    neutral, neutral_features = fit_words(neutral_docs)
    ranks, error = score_arm(dense, neutral, label_index, group_ids, label_count)
    if error:
        raise SystemExit(error)
    rr["neutralised_words"] = 1.0 / ranks
    arms["neutralised_words"] = {"kind": "neutralise", "class": "the 605-surface catalogue",
                                 "features_kept": int(neutral.shape[1]), "defined": True,
                                 "mrr": round(float(np.mean(rr["neutralised_words"])), 4),
                                 "recall_at_10": round(float(np.mean(ranks <= 10)), 4)}
    print(f"  MRR {arms['neutralised_words']['mrr']:.4f}", flush=True)

    print("paired bootstrap over leakage groups", flush=True)
    pairs = [(arm, "full") for arm in arms if arms[arm]["defined"]]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        arms[c["system"]]["difference_from_full"] = c["mrr_difference"]
        arms[c["system"]]["ci95"] = c["ci95"]

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "anatomy of the word-level identity signal",
        "question": "which vocabulary carries the identity that jieba words capture",
        "queries": len(songs), "labels": label_count,
        "full_model": {"mrr": round(full_mrr, 4), "features": int(lexical.shape[1]),
                       "reproduces_representation_study": True,
                       "tokenisation": "jieba default cut, word unigrams and bigrams, TF-IDF "
                                       "with the character system's settings"},
        "classes": {
            "definitions": {
                "pos_*": "jieba part-of-speech guess for the word in isolation, grouped; a "
                         "description of the vocabulary, not a truth about each use; a bigram "
                         "takes the class of its rarer member",
                "pos_groups": {k: sorted(v) for k, v in POS_GROUPS.items()},
                "df_*": "document-frequency quartile of the feature over the query corpus",
            },
            "inventory": inventory,
            "correct_label_score_share": shares,
        },
        "arms": arms,
        "bootstrap": {"design": "2000 replicates, seed 20260825, leakage groups resampled with "
                                "replacement, each (group, label) component weighted one; "
                                "difference of each arm from the full model"},
        "privacy": "aggregate only; no word is published",
    }
    (out_dir / "word_identity_anatomy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'word_identity_anatomy.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
