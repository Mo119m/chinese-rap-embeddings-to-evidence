#!/usr/bin/env python3
"""The three spaces on corpus v3, and how much the cleaning moved them.

Corpus v3 removed 1.22% of lines -- credits, stray titles, HTML, digits -- from corpus v2.
Every headline number was measured on v2, so every one has to be measured again on v3
before anything else is built on it. This file runs the song-level protocol on v3 for the
lexical spaces (character 2-5-grams, jieba words, jieba words with the 605-surface
catalogue stripped) and the rhyme-form space, and for the semantic space in one of two
states: the v3 BGE-M3 vectors if the recorded Colab run over the v3 text exists, or the
v2 vectors of the same chunks as a clearly marked interim, which were computed on the
uncleaned text and are reported here only so the lexical numbers have a companion column.

Queries, labels and leakage groups are rebuilt from v3 by the same rules as v2 (at least
50 normalised characters; labels with at least five such songs; exact-text components
union near-duplicates), so the population is what v3 supports, not v2's population
filtered.

    python src/three_spaces_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
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
from identity_spaces_v2 import ending_sequence, fit_phonological, paired_group_bootstrap, phonological_tokens  # noqa: E402
from leakage_groups_v2 import build_groups, group_audit, normalise_document  # noqa: E402
from lexical_identity_anatomy_v2 import score_arm  # noqa: E402
from lexical_identity_decomposition_v2 import neutralise  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
V2_REFERENCE = {"semantic": 0.3181, "lexical_char_2_5": 0.4503, "lexical_words": 0.5188,
                "lexical_words_neutralised": 0.5113, "rhyme_form_strict": 0.0967}


def build(private_root: Path, out_dir: Path) -> int:
    import jieba
    jieba.setLogLevel(60)
    print("loading corpus v3", flush=True)
    rows, vectors, vector_state = load_v3(private_root, allow_interim_v2_vectors=True)
    print(f"  vectors: {vector_state['vectors']}", flush=True)
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
    normalised = {s: normalise_document(documents[s]) for s in songs}
    groups = build_groups(songs, components_by_song, normalised)
    audit = group_audit(groups, {s: components_by_song[s] for s in songs}, label_by_song) \
        if "group_audit" in dir() else {}
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    label_count = len(eligible)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    corpus_docs = [documents[s] for s in songs]
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    lexicon = private_root / "work" / "lexicon_arm_everything.csv"
    surfaces = sorted({r["entity"].strip() for r in csv.DictReader(lexicon.open(encoding="utf-8-sig"))
                       if r.get("entity", "").strip()}, key=len, reverse=True)

    print("fitting the spaces", flush=True)
    spaces = {"lexical_char_2_5": v1.fit_tfidf(corpus_docs)}
    spaces["lexical_words"], _ = fit_words([" ".join(segment(d)) for d in corpus_docs])
    spaces["lexical_words_neutralised"], _ = fit_words([" ".join(segment(neutralise(d, surfaces))) for d in corpus_docs])
    sequences = {}
    for song in songs:
        ordered = sorted(chunks_by_song[song], key=lambda i: int(rows[i]["source_order"]))
        sequences[song] = ending_sequence([rows[i]["cleaned_text"] for i in ordered])
    spaces["rhyme_form_strict"], _ = fit_phonological([phonological_tokens(sequences[s], with_finals=False) for s in songs])
    covered = np.asarray([len(sequences[s]) >= 4 for s in songs])

    ranks = {}
    profiles = v1.score_leave_group_out(dense, spaces["lexical_char_2_5"], label_index, group_ids, label_count)
    ranks["semantic"] = v1.rank_system(profiles.dense, label_index)[0].astype(np.int64)
    ranks["lexical_char_2_5"] = v1.rank_system(profiles.lexical, label_index)[0].astype(np.int64)
    for name in ("lexical_words", "lexical_words_neutralised", "rhyme_form_strict"):
        r, error = score_arm(dense, spaces[name], label_index, group_ids, label_count)
        if error:
            raise SystemExit(f"{name}: {error}")
        if name == "rhyme_form_strict":
            r[~covered] = label_count
        ranks[name] = r
    fused = (v1.zscore_rows(profiles.dense.astype(np.float64)) + v1.zscore_rows(profiles.lexical.astype(np.float64))) / 2.0
    ranks["fusion_semantic_char"] = v1.rank_system(fused.astype(np.float32), label_index)[0].astype(np.int64)

    rr = {name: 1.0 / r for name, r in ranks.items()}
    report = {}
    for name, r in ranks.items():
        report[name] = {"mrr": round(float(np.mean(1.0 / r)), 4),
                        "recall_at_1": round(float(np.mean(r <= 1)), 4),
                        "recall_at_10": round(float(np.mean(r <= 10)), 4),
                        "v2_mrr_for_reference": V2_REFERENCE.get(name)}
        ref = V2_REFERENCE.get(name)
        print(f"  {name:28s} MRR {report[name]['mrr']:.4f}" + (f"  (v2 {ref:.4f})" if ref else ""), flush=True)
    pairs = [("lexical_char_2_5", "semantic"), ("lexical_words", "lexical_char_2_5"),
             ("lexical_words_neutralised", "lexical_words"), ("fusion_semantic_char", "lexical_char_2_5")]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "the three spaces on corpus v3",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "chunks": len(rows),
                   "song_records": len(chunks_by_song), "queries": len(songs), "labels": label_count,
                   "groups": len(order)},
        "semantic_vectors": vector_state,
        "semantic_caveat": (None if vector_state["vectors"] == "v3" else
                            "the semantic column uses the v2 vectors of the same chunks, computed on the "
                            "uncleaned text; it is an interim companion, not a v3 semantic result"),
        "systems": report,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with "
                                       "replacement, each (group, label) component weighted one",
                             "contrasts": contrasts},
        "not_comparable_to": "results/retrieval-v2 point to point: the population differs (24,343 chunks, "
                             "7,381 songs); the v2 numbers are quoted only so the reader can see the direction",
        "privacy": "aggregate only",
    }
    (out_dir / "three_spaces.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'three_spaces.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
