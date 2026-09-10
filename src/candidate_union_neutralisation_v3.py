#!/usr/bin/env python3
"""Does the 'not names' finding depend on how complete the NER was?

The neutralisation test strips the 605 reviewed surfaces and finds the word space loses
0.006. That catalogue is what two NER arms agreed on and a reviewer screened; what either
arm proposed and the gate rejected is much larger, and what neither arm ever saw is
unknown. Recall of the NER is unmeasured (the released-claim audit measures precision
only). This file bounds what NER recall could change: it strips, from the word space,
the UNION of every surface either arm ever proposed for a named-entity type -- lexicon
hits, transformer hits, conflicts, everything, whether or not the gate accepted it --
and then, in a second arm, also every proposed rap-culture term (flow, beat, hook, ...:
the rapper's trade vocabulary). If the word space still holds, nothing the NER could have
found is what identifies a label; what would remain is only what neither arm ever saw.

Surfaces come from the private candidate table as strings only; no context, no counts
per song, no song identifiers leave the private root.

    python src/candidate_union_neutralisation_v3.py --private-root <ni-k>
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from lexical_identity_anatomy_v2 import score_arm  # noqa: E402
from lexical_identity_decomposition_v2 import LATIN, neutralise  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
NAMED_TYPES = ("PLACE", "PERSON_REFERENCE", "GROUP_CREW_OR_ORGANIZATION", "LANGUAGE_OR_DIALECT_REFERENCE",
               "ETHNOCULTURAL_GROUP_REFERENCE", "EVENT", "WORK_OR_MEDIA", "BRAND_OR_PRODUCT",
               "OTHER_CULTURAL_REFERENCE")
CULTURE_TYPES = ("RAP_CULTURE_CONCEPT",)
MIN_SURFACE_LENGTH = 2
SONG_SHARE_CUTOFF = 0.02


def candidate_surfaces(private_root: Path):
    path = private_root / "work" / "private-chinese-rap-ner-cultural-graph-v1" / "all_candidate_occurrences_private.csv"
    by_type: dict[str, set[str]] = defaultdict(set)
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        for surface in (row.get("candidate_surface", ""), row.get("transformer_surface", "")):
            surface = surface.strip()
            if len(surface) >= MIN_SURFACE_LENGTH:
                by_type[row["candidate_schema_type"]].add(surface)
    return by_type


def build(private_root: Path, out_dir: Path) -> int:
    import jieba
    jieba.setLogLevel(60)
    print("loading corpus v3", flush=True)
    rows, vectors, _ = load_v3(private_root)
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
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    corpus_docs = [documents[s] for s in songs]
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    by_type = candidate_surfaces(private_root)
    lexicon = private_root / "work" / "lexicon_arm_everything.csv"
    reviewed = {r["entity"].strip() for r in csv.DictReader(lexicon.open(encoding="utf-8-sig")) if r.get("entity", "").strip()}
    named = set().union(*(by_type[t] for t in NAMED_TYPES if t in by_type))
    culture = set().union(*(by_type[t] for t in CULTURE_TYPES if t in by_type))
    # The transformer arm proposed ordinary words as PERSON on a large scale: on the first
    # run the forty surfaces that removed the most text were two-character "names" each
    # covering 0.3-0.9% of all characters, which no name does. A surface found in more than
    # SONG_SHARE_CUTOFF of the songs is a common word, not a name; the union is split there
    # so the bound is not paid for with the corpus's common vocabulary.
    print("  measuring each candidate surface's song frequency", flush=True)
    song_share = {}
    for surface in named:
        if LATIN.fullmatch(surface.replace(" ", "")):
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(surface) + r"(?![A-Za-z0-9])", re.I)
            hits = sum(1 for d in corpus_docs if pattern.search(d))
        else:
            hits = sum(1 for d in corpus_docs if surface in d)
        song_share[surface] = hits / len(corpus_docs)
    named_rare = {s for s in named if song_share[s] <= SONG_SHARE_CUTOFF}
    named_common = named - named_rare
    arms = {
        "reviewed_605": sorted(reviewed, key=len, reverse=True),
        "every_named_candidate": sorted(named | reviewed, key=len, reverse=True),
        "named_candidates_in_at_most_2pct_of_songs": sorted(named_rare | reviewed, key=len, reverse=True),
        "named_candidates_in_over_2pct_of_songs_only": sorted(named_common, key=len, reverse=True),
        "every_named_and_culture_candidate": sorted(named | culture | reviewed, key=len, reverse=True),
    }
    inventory = {name: len(s) for name, s in arms.items()}
    inventory["by_type_unique_surfaces"] = {t: len(by_type[t]) for t in sorted(by_type)}
    inventory["song_share_cutoff"] = SONG_SHARE_CUTOFF
    print("  surfaces: " + ", ".join(f"{k} {v:,}" for k, v in inventory.items() if isinstance(v, int)), flush=True)

    rr = {}
    report = {}
    print("scoring", flush=True)
    matrix, _ = fit_words([" ".join(segment(d)) for d in corpus_docs])
    ranks, error = score_arm(dense, matrix, label_index, group_ids, label_count)
    if error:
        raise SystemExit(error)
    rr["words"] = 1.0 / ranks
    report["words"] = {"mrr": round(float(np.mean(rr["words"])), 4)}
    print(f"  words                                   MRR {report['words']['mrr']:.4f}", flush=True)
    for name, surfaces in arms.items():
        stripped = [neutralise(d, surfaces) for d in corpus_docs]
        removed = sum(len(a) - len(b) for a, b in zip(corpus_docs, stripped))
        matrix, _ = fit_words([" ".join(segment(d)) for d in stripped])
        ranks, error = score_arm(dense, matrix, label_index, group_ids, label_count)
        if error:
            raise SystemExit(f"{name}: {error}")
        rr[name] = 1.0 / ranks
        report[name] = {"mrr": round(float(np.mean(rr[name])), 4), "surfaces": len(surfaces),
                        "characters_removed": int(removed),
                        "share_of_text_removed": round(removed / sum(len(d) for d in corpus_docs), 4)}
        print(f"  {name:38s} MRR {report[name]['mrr']:.4f}  ({len(surfaces):,} surfaces, "
              f"{report[name]['share_of_text_removed']:.2%} of text)", flush=True)
    pairs = [(name, "words") for name in arms] + [("every_named_candidate", "reviewed_605"),
                                                   ("named_candidates_in_at_most_2pct_of_songs", "reviewed_605"),
                                                   ("every_named_and_culture_candidate", "every_named_candidate")]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "word-space neutralisation with the union of every NER candidate surface",
        "question": "whether the 'not names' finding could depend on NER recall",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count, "groups": len(order)},
        "arms": {"reviewed_605": "the screened catalogue the published neutralisation uses",
                 "every_named_candidate": "plus every surface either NER arm ever proposed for a named-entity type, "
                                          "whatever the agreement state or gate outcome",
                 "named_candidates_in_at_most_2pct_of_songs": "the catalogue plus the proposed named surfaces found in "
                                                              f"at most {SONG_SHARE_CUTOFF:.0%} of songs -- the ones that "
                                                              "could be names",
                 "named_candidates_in_over_2pct_of_songs_only": "only the proposed named surfaces found in more than "
                                                                f"{SONG_SHARE_CUTOFF:.0%} of songs -- ordinary words the "
                                                                "transformer tagged as names",
                 "every_named_and_culture_candidate": "plus every proposed rap-culture term (flow, beat, hook, ...)"},
        "named_types": NAMED_TYPES, "culture_types": CULTURE_TYPES, "minimum_surface_length": MIN_SURFACE_LENGTH,
        "surface_inventory": inventory,
        "systems": report,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with replacement, "
                                       "each (group, label) component weighted one", "contrasts": contrasts},
        "reading": "if the word space holds under the widest strip, nothing the NER could have found is what "
                   "identifies a label; only what neither arm ever saw is left open",
        "privacy": "aggregate only; no surface is published",
    }
    (out_dir / "candidate_union_neutralisation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'candidate_union_neutralisation.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
