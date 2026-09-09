#!/usr/bin/env python3
"""What the data says about how an identity encoder should be trained on it.

Before any model is fitted to corpus v3, five facts about the corpus decide the design,
and each is measured here rather than assumed:

  1. How long a chunk is in BGE-M3's own tokens. The recorded embedding run used
     max_length 2048; if the median chunk is a few dozen tokens, the unit of encoding is a
     stanza and the budget is irrelevant, and a song-level input would be a different
     experiment.
  2. How many songs and how many leakage groups each label has. A contrastive positive
     must be two texts by one label from DIFFERENT leakage groups, or the model learns to
     recognise a repeated hook; labels with one group cannot supply positives.
  3. How many songs are collaborations. A title carrying feat./ft./& puts another artist's
     verses under this label; those songs are label noise for training positives.
  4. How often a chunk names its own label, so the training text can mask it -- otherwise
     the first thing learned is to read the name.
  5. How close the nearest same-content rival is: for each chunk, the cosine to its nearest
     BGE-M3 neighbour from another label, which is what a content-controlled negative
     looks like (Wegmann et al. 2022) and how hard the negatives will be.

Aggregate only; the per-song tables stay private.

    python src/training_data_audit_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
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
from corpus_v3 import load_v3  # noqa: E402
from identity_probe_v2 import unit_rows  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
FEAT = re.compile(r"(?:\bfeat\.?\b|\bft\.?\b|featuring|&|/|\bx\b|×|，|,|、)", re.I)
LATIN = re.compile(r"^[A-Za-z0-9 .'&$\-]+$")


def quantiles(values, points=(10, 25, 50, 75, 90, 99)):
    values = np.asarray(values)
    return {str(p): int(np.percentile(values, p)) for p in points}


def build(private_root: Path, out_dir: Path) -> int:
    rows, vectors, state = load_v3(private_root, allow_interim_v2_vectors=True)
    chunks_by_song, label_by_song, components_by_song, documents, _ = build_songs(rows, vectors)
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {s for s in chunks_by_song
                   if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(l for l, m in songs_by_label.items()
                      if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
    groups = build_groups(songs, components_by_song, {s: normalise_document(documents[s]) for s in songs})
    print(f"{len(rows):,} chunks; {len(songs):,} query songs; {len(eligible)} labels", flush=True)

    # 1. tokens per chunk, in BGE-M3's tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    texts = [r["cleaned_text"] for r in rows]
    lengths = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    chars = [len(t) for t in texts]
    lines = [t.count("\n") + 1 for t in texts]
    print(f"  tokens per chunk: median {int(np.median(lengths))}, p90 {int(np.percentile(lengths, 90))}, "
          f"max {max(lengths)}; over 512: {sum(l > 512 for l in lengths):,}", flush=True)

    # 2. songs and groups per label
    groups_per_label = Counter()
    seen = set()
    for s in songs:
        key = (label_by_song[s], groups[s])
        if key not in seen:
            seen.add(key)
            groups_per_label[label_by_song[s]] += 1
    songs_per_label = Counter(label_by_song[s] for s in songs)
    labels_with_positives = sum(1 for l in eligible if groups_per_label[l] >= 2)
    print(f"  labels with >= 2 leakage groups (can supply positives): {labels_with_positives} of {len(eligible)}", flush=True)

    # 3. collaborations by title
    titles = {r["song_id"]: r["song_title"] for r in rows}
    feat_songs = {s for s in songs if FEAT.search(titles[s])}
    feat_by_label = Counter(label_by_song[s] for s in feat_songs)
    print(f"  songs whose title marks a collaboration: {len(feat_songs):,} ({len(feat_songs) / len(songs):.1%})", flush=True)

    # 4. self-mentions
    def mentions(text, label):
        if LATIN.match(label):
            return re.search(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])", text, re.I) is not None
        return label in text
    self_chunks = sum(1 for r in rows if len(r["source_credit_label"].strip()) >= 2
                      and mentions(r["cleaned_text"], r["source_credit_label"].strip()))
    print(f"  chunks naming their own label: {self_chunks:,} ({self_chunks / len(rows):.1%})", flush=True)

    # 5. nearest rival by content
    song_pos = {s: i for i, s in enumerate(songs)}
    chunk_rows = [i for s in songs for i in chunks_by_song[s]]
    chunk_label = np.asarray([eligible.index(label_by_song[rows[i]["song_id"]]) for i in chunk_rows])
    chunk_group = np.asarray([groups[rows[i]["song_id"]] for i in chunk_rows])
    chunk_vec = unit_rows(vectors[chunk_rows].astype(np.float64))
    nearest_rival = np.zeros(len(chunk_rows))
    nearest_same = np.zeros(len(chunk_rows))
    for start in range(0, len(chunk_rows), 1000):
        block = chunk_vec[start:start + 1000] @ chunk_vec.T
        same_label = chunk_label[start:start + 1000][:, None] == chunk_label[None, :]
        same_group = chunk_group[start:start + 1000][:, None] == chunk_group[None, :]
        rival = np.where(~same_label, block, -np.inf)
        own = np.where(same_label & ~same_group, block, -np.inf)
        nearest_rival[start:start + 1000] = rival.max(axis=1)
        nearest_same[start:start + 1000] = own.max(axis=1)
    finite_same = np.isfinite(nearest_same)
    harder = float(np.mean(nearest_rival[finite_same] > nearest_same[finite_same]))
    print(f"  nearest content rival cosine: median {np.median(nearest_rival):.3f}; nearest same-label "
          f"(other group) median {np.median(nearest_same[finite_same]):.3f}; rival closer than own in "
          f"{harder:.1%} of chunks", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "what corpus v3 says about training an identity encoder on it",
        "vectors_used_for_neighbours": state["vectors"],
        "chunk_length": {"tokens_bge_m3": quantiles(lengths), "characters": quantiles(chars),
                         "lines": quantiles(lines), "chunks_over_512_tokens": int(sum(l > 512 for l in lengths)),
                         "reading": "the recorded run's max_length of 2,048 is never reached; the natural "
                                    "unit of encoding is the stanza"},
        "labels": {"eligible": len(eligible), "with_at_least_two_leakage_groups": labels_with_positives,
                   "songs_per_label": quantiles(list(songs_per_label.values()), (10, 25, 50, 75, 90, 100)),
                   "groups_per_label": quantiles(list(groups_per_label.values()), (10, 25, 50, 75, 90, 100))},
        "collaborations": {"songs_with_a_collaboration_mark_in_the_title": len(feat_songs),
                           "share": round(len(feat_songs) / len(songs), 4),
                           "labels_with_any": len(feat_by_label),
                           "rule": "title contains feat., ft., featuring, &, /, x, × or a list separator; "
                                   "a candidate list for exclusion from training positives, not a truth"},
        "self_mentions": {"chunks": self_chunks, "share": round(self_chunks / len(rows), 4),
                          "consequence": "mask the label string in training text"},
        "content_controlled_negatives": {
            "nearest_rival_cosine": {str(p): round(float(np.percentile(nearest_rival, p)), 3) for p in (10, 50, 90)},
            "nearest_same_label_other_group_cosine": {str(p): round(float(np.percentile(nearest_same[finite_same], p)), 3) for p in (10, 50, 90)},
            "share_of_chunks_whose_nearest_rival_is_closer_than_nearest_own": round(harder, 4),
            "reading": "in the raw semantic space most chunks sit closer to some other label's chunk than to "
                       "any of their own label's; a negative drawn from those neighbours controls for content"},
        "privacy": "aggregate only",
    }
    (out_dir / "training_data_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'training_data_audit.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
