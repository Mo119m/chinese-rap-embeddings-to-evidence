#!/usr/bin/env python3
"""The leakage unit for corpus v2: exact-text components unioned with near-duplicate songs.

PD-002 makes exact-cleaned-text components the leakage unit -- "keep component-linked
records together across train, validation, and test". The downstream builders already had
their own duplicate detection, at the whole-song-document level: identical normalised
documents, plus trigram Jaccard for near-duplicates.

Neither rule contains the other, which is the reason this module exists rather than a
one-line substitution. Measured on corpus v2, over its 7,391 song records:

* 799 text components span more than one song. 404 of them -- 50.6% -- would be split apart
  by the document rule, because the songs share one exact chunk and differ elsewhere.
* 1,434 song pairs reach trigram Jaccard 0.80. 44 of those pairs share no text component at
  all: near-identical songs in which every chunk was edited just enough that none is
  byte-identical.

So each rule catches leakage the other cannot see, and the correct unit is their union: the
connected components of a graph whose edges are "these two songs share a text component" or
"these two songs are near-duplicates".

The union does not run away. On the 204 eligible labels it yields 5,799 groups from 7,021
songs with a largest group of 27 -- the same largest group as components alone, so the added
near-duplicate edges merge locally rather than cascading. Every label keeps at least 7
groups (median 36), which matters because leave-group-out holds an entire group out and
needs training groups left over.

    from leakage_groups_v2 import build_groups
    groups = build_groups(song_ids, component_ids_by_song, normalised_documents)
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

# The threshold the retrieval builder has always used for near-duplicate songs. Kept
# identical so the v2 grouping is a superset of the v1 near-duplicate behaviour rather than
# a different rule wearing the same name.
NEAR_DUPLICATE_JACCARD = 0.80
TRIGRAM = 3


class UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self._parent[right] = left


def normalise_document(text: str) -> str:
    """NFKC, whitespace removed, lowercased -- the retrieval builder's own normalisation."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def near_duplicate_pairs(documents: Mapping[str, str],
                         threshold: float = NEAR_DUPLICATE_JACCARD) -> list[tuple[str, str]]:
    """Every pair at or above the Jaccard threshold, by exact all-pairs join.

    Uses the standard global-order prefix filter: for threshold t a set keeps
    |S| - ceil(t|S|) + 1 tokens, and any pair reaching t must share one of them. Candidates
    are then verified against the full trigram sets, so this is exact, not a sample.
    """
    trigrams = {
        key: {value[index:index + TRIGRAM] for index in range(len(value) - TRIGRAM + 1)}
        for key, value in documents.items() if len(value) >= TRIGRAM
    }
    frequency = Counter(token for tokens in trigrams.values() for token in tokens)
    inverted: dict[str, list[str]] = defaultdict(list)
    candidates: set[tuple[str, str]] = set()
    for key in sorted(trigrams):
        tokens = trigrams[key]
        keep = len(tokens) - math.ceil(threshold * len(tokens)) + 1
        for token in sorted(tokens, key=lambda t: (frequency[t], t))[:keep]:
            for other in inverted[token]:
                candidates.add((other, key) if other < key else (key, other))
            inverted[token].append(key)

    pairs = []
    for left, right in sorted(candidates):
        shared = len(trigrams[left] & trigrams[right])
        union = len(trigrams[left]) + len(trigrams[right]) - shared
        if union and shared / union >= threshold:
            pairs.append((left, right))
    return pairs


def build_groups(
    song_ids: Sequence[str],
    components_by_song: Mapping[str, Iterable[str]],
    documents: Mapping[str, str],
    threshold: float = NEAR_DUPLICATE_JACCARD,
) -> dict[str, str]:
    """Map each song to its leakage group id.

    `components_by_song` carries corpus v2's text_component_id values; `documents` carries
    the already-normalised whole-song text. Group ids are deterministic: a group is named
    for its lowest-sorting member, so the mapping does not depend on input order.
    """
    songs = sorted(song_ids)
    index = {song: position for position, song in enumerate(songs)}
    union_find = UnionFind(len(songs))

    by_component: dict[str, list[str]] = defaultdict(list)
    for song in songs:
        for component in components_by_song.get(song, ()):
            by_component[component].append(song)
    for members in by_component.values():
        for other in members[1:]:
            union_find.union(index[members[0]], index[other])

    for left, right in near_duplicate_pairs(
            {song: documents[song] for song in songs if song in documents}, threshold):
        union_find.union(index[left], index[right])

    members_by_root: dict[int, list[str]] = defaultdict(list)
    for song in songs:
        members_by_root[union_find.find(index[song])].append(song)
    return {song: min(members) for members in members_by_root.values() for song in members}


def group_audit(groups: Mapping[str, str],
                labels_by_song: Mapping[str, str],
                components_by_song: Mapping[str, Iterable[str]]) -> dict:
    """Numbers a reader needs to judge the grouping, including the ones that could go wrong."""
    members: dict[str, list[str]] = defaultdict(list)
    for song, group in groups.items():
        members[group].append(song)
    sizes = sorted((len(value) for value in members.values()), reverse=True)

    per_label: dict[str, set[str]] = defaultdict(set)
    for song, group in groups.items():
        per_label[labels_by_song[song]].add(group)
    label_group_counts = sorted(len(value) for value in per_label.values())

    component_groups: dict[str, set[str]] = defaultdict(set)
    for song, group in groups.items():
        for component in components_by_song.get(song, ()):
            component_groups[component].add(group)
    return {
        "songs": len(groups),
        "groups": len(members),
        "multi_song_groups": sum(1 for size in sizes if size > 1),
        "largest_group_songs": sizes[0] if sizes else 0,
        "labels": len(per_label),
        "minimum_groups_in_a_label": label_group_counts[0] if label_group_counts else 0,
        "median_groups_in_a_label": (label_group_counts[len(label_group_counts) // 2]
                                     if label_group_counts else 0),
        # must be zero: a component split across groups would be exactly the leakage
        # PD-002 rule 3 forbids
        "text_components_split_across_groups": sum(
            1 for value in component_groups.values() if len(value) > 1),
    }
