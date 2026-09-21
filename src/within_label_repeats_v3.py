#!/usr/bin/env python3
"""Are the headline contrasts carried by hooks and formula lines repeated across songs of one label?

QUESTION. On corpus v3 the word space identifies a source-credit label better than the character
space, which beats the semantic space: 0.4963 > 0.4266 > 0.2997 MRR. A label that reuses a hook, an
ad-lib or a formula line across its songs hands every surface space an easy match: leave-group-out
removes the query's leakage group (exact-text components union near-duplicate songs) from the
profiles, not the label's other songs that merely share a passage with the query. If the ordering
rests on such self-repetition it is about repeated text, not about vocabulary as identity. Two arms
remove the repeats in two ways and rerun the unchanged protocol.

DETECTOR. The label-identity audit (build_label_identity_adjudication_v1, whose 511 label pairs
sharing a passage of at least thirty characters the manuscript reports) calls two songs
passage-sharing when a whole chunk of at least MINIMUM_PASSAGE_CHARACTERS = 30 characters (stripped
of surrounding whitespace) recurs verbatim. It is reimplemented here at finer
grain and restricted to same-label pairs: two songs share a passage iff their normalised texts
(leakage_groups_v2.normalise_document: NFKC, whitespace removed, lower-cased) share a run of
PASSAGE_CHARACTERS = 30 characters wherever it falls, so a repeated hook inside a longer stanza
counts. Every whole-chunk match of the audit's rule is a match here, and that is checked. No text is
printed or written; only counts.

ARM (a), merged groups. Same-label songs that share a passage are joined into one leakage group (the
union of their published groups), so no profile a query is scored against contains a song sharing a
hook with it. Per-(group, label) weights are rebuilt by the protocol's rule; a merged group keeps the
published fold of its lowest-numbered member group, so the fold-wise whitening differs from the
published one only through the merge. The query population is unchanged unless a label is left with
a single group (it cannot be scored and is dropped; counted); the surface spaces are refitted only if
the population changed. Text is the corpus's own.

ARM (b), stripped lines. A line (a chunk's text split on newlines, normalised as above) is recurrent
for a label when songs of at least two published leakage groups of that label contain it. Every
recurrent line is removed from every song of that label before any space is built; within-song
repetition and lines recurring inside one group only stay. Songs left below MIN_EFFECTIVE_CHARACTERS
(50) normalised characters, labels left with fewer than MINIMUM_SONGS_PER_LABEL (5) such songs and
labels left with one group leave the population by the protocol's own rules (counted); the published
groups and folds are kept over the survivors, so stripping cannot weaken the leakage control. The
recorded BGE-M3 vectors are per chunk and re-embedding stripped text is a GPU stage this file does
not contain, so the semantic space is bracketed at the chunk grain: 'emptied' drops only chunks
whose every line was stripped (hooks inside kept chunks survive in the vector: the bracket that
favours the semantic space), 'touched' drops every chunk containing a stripped line (removes more
than the hook: the bracket that favours the surface spaces); a song with no untouched chunk falls
back to its emptied set. The share of stripped characters surviving inside kept chunks is reported
per bracket.

SPACES AND SCORERS, the protocol's own. semantic_raw: cosine of the unit chunk-mean song vector with
the leave-group-out weighted-sum label profile (identity_probe_v2.dense_leave_group_out). chars_raw:
character 2-5-gram TF-IDF (build_chinese_rap_downstream_retrieval_v1.fit_tfidf); words_raw: jieba
1-2-gram TF-IDF (word_identity_anatomy_v2.fit_words); both scored by v1.score_leave_group_out, an
empty row taking the worst rank as in lexical_identity_anatomy_v2.score_arm. semantic_whitened:
within-author whitening fitted per fold on the training folds (identity_probe_v2.fit_transform,
Ledoit-Wolf shrinkage), each fold's queries scored in its own space. Ranks by v1.rank_system.
ESTIMANDS per space: MRR over queries (plain mean, the published figure) and the per-(group,
label)-weighted MRR. Every contrast is a difference of weighted MRR with a paired group-bootstrap
interval (identity_spaces_v2.paired_group_bootstrap: 2000 replicates, seed 20260825, leakage groups
resampled with replacement) over the arm's own groups and population.

CHECKS, before any result is used. (1) The published configuration must reproduce semantic_raw
0.2997, chars_raw 0.4266 and words_raw 0.4963 within 0.0005 (the repo's tolerance where an unchanged
space is re-scored: identity_probe_v2, word_identity_anatomy_v2) and semantic_whitened 0.4164 within
0.002 (the repo's tolerance for fold-wise transforms: exemplar_vs_prototype_v3), or the run stops.
(2) The population rebuilt here for the published configuration equals the protocol's
(labels, groups, weights, folds). (3) The documents and chunk-mean song vectors rebuilt here with
nothing stripped equal the protocol's documents and centroids. (4) Every same-label whole-chunk match
of the audit's rule among the queries whose normalised chunk has at least 30 characters is found by
the 30-character detector.

READING RULE, fixed before the run. The ordering is 'not driven by repeated passages' if all three
contrasts (words - characters, characters - semantic, words - semantic) keep their published sign
with intervals clear of zero in both arms, the two semantic contrasts of arm (b) under both
chunk-grain brackets. Otherwise the ordering is 'at least partly carried by repeated passages' and
the failing contrasts are named. Contrasts involving the whitened semantic space, the counts (songs
affected, groups merged, lines stripped) and the movement of each space's MRR are description, not
part of the rule.

    set CHINESE_RAP_CORPUS=v3
    python src/within_label_repeats_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs, corpus_version  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402
from leakage_groups_v2 import UnionFind, build_groups, normalise_document  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "within_label_repeats.json"
PASSAGE_CHARACTERS = 30            # the label-identity audit's MINIMUM_PASSAGE_CHARACTERS
CHECK_GAP_RESCORED = 5e-4          # an unchanged space re-scored (identity_probe_v2, word_identity_anatomy_v2)
CHECK_GAP_FOLDWISE = 0.002         # a fold-wise transform refitted (exemplar_vs_prototype_v3)
EXPECTED = {"semantic_raw": 0.2997, "chars_raw": 0.4266, "words_raw": 0.4963, "semantic_whitened": 0.4164}
TOLERANCE = {"semantic_raw": CHECK_GAP_RESCORED, "chars_raw": CHECK_GAP_RESCORED, "words_raw": CHECK_GAP_RESCORED,
             "semantic_whitened": CHECK_GAP_FOLDWISE}
PRIMARY = (("words_raw", "chars_raw"), ("chars_raw", "semantic_raw"), ("words_raw", "semantic_raw"))
SECONDARY = (("words_raw", "semantic_whitened"), ("chars_raw", "semantic_whitened"), ("semantic_whitened", "semantic_raw"))
TOUCHED = "_touched"
LINE_BANDS = (("1-3", 1, 3), ("4-9", 4, 9), ("10-19", 10, 19), ("20-29", 20, 29), ("30-up", 30, 10**9))


# ------------------------------------------------------------------ the corpus, as the protocol builds it
def setup_full(private_root: Path) -> dict:
    """exemplar_vs_prototype_v3.setup, plus the rows, chunk vectors, each song's chunk order and the
    published fold of every group."""
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
    normalised = {s: normalise_document(documents[s]) for s in songs}
    groups = build_groups(songs, components_by_song, normalised)
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold_of_group = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))
    fold = fold_of_group[group_ids]
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs])).astype(np.float64)
    chunk_rows = {s: sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"])) for s in songs}
    return dict(rows=rows, vectors=vectors, songs=songs, label_of={s: label_by_song[s] for s in songs},
                label_index=label_index, group_ids=group_ids, weights=weights, fold=fold, fold_of_group=fold_of_group,
                label_count=len(eligible), dense=dense, documents={s: documents[s] for s in songs},
                normalised=normalised, chunk_rows=chunk_rows)


def chunk_lines(rows, chunk_rows: dict[str, list[int]]) -> dict[str, list[list[str]]]:
    """Each song's chunks in source order, each chunk split on its newlines."""
    return {s: [rows[i]["cleaned_text"].split("\n") for i in indices] for s, indices in chunk_rows.items()}


# ------------------------------------------------------------------ the detector and the merge (arm a)
def shared_passage_pairs(documents: dict[str, str], label_of: dict[str, str],
                         k: int = PASSAGE_CHARACTERS) -> list[tuple[str, str]]:
    """Every pair of same-label songs whose (already normalised) documents share a substring of
    at least k characters, which is the same as sharing a k-gram. Sorted, each pair once."""
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song in sorted(documents):
        songs_by_label[label_of[song]].append(song)
    pairs: set[tuple[str, str]] = set()
    for label in sorted(songs_by_label):
        owners: dict[str, set[str]] = defaultdict(set)
        for song in songs_by_label[label]:
            text = documents[song]
            for start in range(len(text) - k + 1):
                owners[text[start:start + k]].add(song)
        for members in owners.values():
            if len(members) > 1:
                ordered = sorted(members)
                for i in range(len(ordered)):
                    for j in range(i + 1, len(ordered)):
                        pairs.add((ordered[i], ordered[j]))
    return sorted(pairs)


def whole_chunk_pairs(rows, chunk_rows: dict[str, list[int]], label_of: dict[str, str],
                      minimum: int = PASSAGE_CHARACTERS) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The label-identity audit's rule restricted to same-label pairs: a whole chunk, stripped, of at
    least `minimum` characters recurring verbatim. Returns (all such pairs, the pairs whose shared
    chunk also has at least `minimum` normalised characters, which the k-gram detector must find)."""
    songs_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    long_normalised: dict[tuple[str, str], bool] = {}
    for song, indices in chunk_rows.items():
        for i in indices:
            text = rows[i]["cleaned_text"].strip()
            if len(text) >= minimum:
                key = (label_of[song], hashlib.sha256(text.encode("utf-8")).hexdigest())
                songs_by_key[key].add(song)
                long_normalised[key] = len(normalise_document(text)) >= minimum
    every: set[tuple[str, str]] = set()
    must: set[tuple[str, str]] = set()
    for key, members in songs_by_key.items():
        ordered = sorted(members)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                every.add((ordered[i], ordered[j]))
                if long_normalised[key]:
                    must.add((ordered[i], ordered[j]))
    return sorted(every), sorted(must)


def merge_groups(group_ids: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Union the published groups of every pair of song positions. Returns, per song, the name of its
    merged group: the smallest published group id the merged group contains (an unmerged group keeps
    its own id, so it also keeps its published fold)."""
    n_groups = int(group_ids.max()) + 1 if len(group_ids) else 0
    union_find = UnionFind(n_groups)
    for a, b in pairs:
        union_find.union(int(group_ids[a]), int(group_ids[b]))
    smallest: dict[int, int] = {}
    for g in range(n_groups):
        root = union_find.find(g)
        smallest[root] = min(smallest.get(root, g), g)
    return np.asarray([smallest[union_find.find(int(g))] for g in group_ids.tolist()], dtype=np.int64)


# ------------------------------------------------------------------ recurrent lines and the strip (arm b)
def recurrent_lines(lines_by_song: dict[str, list[list[str]]], label_of: dict[str, str],
                    group_of: dict[str, int]) -> dict[tuple[str, str], int]:
    """(label, normalised line) -> number of leakage groups of that label whose songs contain the
    line, for every line found in at least two groups. Empty normalised lines never count."""
    groups: dict[tuple[str, str], set[int]] = defaultdict(set)
    for song, chunks in lines_by_song.items():
        for lines in chunks:
            for line in lines:
                key = normalise_document(line)
                if key:
                    groups[(label_of[song], key)].add(group_of[song])
    return {key: len(g) for key, g in groups.items() if len(g) >= 2}


def strip_songs(lines_by_song: dict[str, list[list[str]]], label_of: dict[str, str],
                recurrent) -> tuple[dict[str, tuple[list[str], np.ndarray, np.ndarray, np.ndarray]], Counter]:
    """Remove every recurrent line of the song's label. Returns per song (stripped chunk texts,
    emptied flags, touched flags, normalised characters stripped per chunk) and aggregate counts;
    with nothing recurrent the chunk texts are the corpus's own."""
    out = {}
    counts: Counter = Counter()
    for song, chunks in lines_by_song.items():
        label = label_of[song]
        texts, emptied, touched, stripped_chars = [], [], [], []
        song_hit = False
        for lines in chunks:
            kept, chunk_chars, chunk_lines = [], 0, 0
            for line in lines:
                key = normalise_document(line)
                counts["characters_total"] += len(key)
                counts["lines_total"] += int(bool(key))
                if key and (label, key) in recurrent:
                    chunk_chars += len(key)
                    chunk_lines += 1
                    for band, lo, hi in LINE_BANDS:
                        if lo <= len(key) <= hi:
                            counts[f"lines_stripped_of_length_{band}"] += 1
                else:
                    kept.append(line)
            text = "\n".join(kept)
            hit = chunk_lines > 0
            empty = hit and normalise_document(text) == ""
            texts.append(text)
            emptied.append(empty)
            touched.append(hit)
            stripped_chars.append(chunk_chars)
            counts["lines_stripped"] += chunk_lines
            counts["characters_stripped"] += chunk_chars
            counts["chunks_touched"] += int(hit)
            counts["chunks_emptied"] += int(empty)
            counts["chunks_total"] += 1
            song_hit = song_hit or hit
        counts["songs_affected"] += int(song_hit)
        out[song] = (texts, np.asarray(emptied, dtype=bool), np.asarray(touched, dtype=bool),
                     np.asarray(stripped_chars, dtype=np.int64))
    return out, counts


def song_vectors(vectors: np.ndarray, chunk_rows: dict[str, list[int]], songs: list[str],
                 keep: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """The protocol's song centroid (plain mean of the recorded chunk vectors, unit-normalised),
    optionally over a kept subset of each song's chunks."""
    stacked = []
    for song in songs:
        indices = np.asarray(chunk_rows[song], dtype=np.int64)
        if keep is not None:
            indices = indices[keep[song]]
        if len(indices) == 0:
            raise RuntimeError("a song has no chunk left")
        stacked.append(vectors[indices].mean(axis=0))
    return v1.l2_normalize_dense(np.stack(stacked)).astype(np.float64)


# ------------------------------------------------------------------ populations and scoring
def make_population(songs: list[str], label_of: dict[str, str], keep: np.ndarray, group_name: np.ndarray,
                    fold_of_name: np.ndarray) -> dict:
    """The protocol's population rules on a song subset with a grouping: a label needs at least
    MINIMUM_SONGS_PER_LABEL kept songs and at least two groups among them; labels, group ids and
    per-(group, label) weights are rebuilt exactly as the protocol builds them; a group's fold is
    fold_of_name[its name], the published assignment."""
    kept = np.flatnonzero(keep)
    songs_of_label: dict[str, list[int]] = defaultdict(list)
    for i in kept.tolist():
        songs_of_label[label_of[songs[i]]].append(i)
    eligible, few, single = [], 0, 0
    for label in sorted(songs_of_label):
        members = songs_of_label[label]
        if len(members) < MINIMUM_SONGS_PER_LABEL:
            few += 1
        elif len({int(group_name[i]) for i in members}) < 2:
            single += 1
        else:
            eligible.append(label)
    position = {label: i for i, label in enumerate(eligible)}
    index = np.asarray([i for i in kept.tolist() if label_of[songs[i]] in position], dtype=np.int64)
    label_index = np.asarray([position[label_of[songs[i]]] for i in index.tolist()], dtype=np.int64)
    names = group_name[index]
    order = {g: i for i, g in enumerate(sorted(set(names.tolist())))}
    group_ids = np.asarray([order[int(g)] for g in names.tolist()], dtype=np.int64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold = np.asarray(fold_of_name)[names]
    return dict(index=index, label_index=label_index, group_ids=group_ids, weights=weights, fold=fold,
                label_count=len(eligible), groups=len(order), queries=int(len(index)),
                labels_dropped_few_songs=few, labels_dropped_single_group=single,
                songs_dropped_by_length=int(len(songs) - len(kept)),
                songs_dropped_with_their_label=int(len(kept) - len(index)))


def score_population(pop: dict, dense_variants: dict[str, np.ndarray], chars, words) -> tuple[dict, dict]:
    """Ranks per space under the protocol. dense_variants maps a suffix ('' for the main space) to a
    unit-row song matrix over the population's rows; chars and words are the population's TF-IDF
    rows. Returns (ranks by system, empty-row counts of the sparse spaces)."""
    li, gi, w, L, fold = pop["label_index"], pop["group_ids"], pop["weights"], pop["label_count"], pop["fold"]
    n = len(li)
    everything = np.arange(n)
    scores: dict[str, np.ndarray] = {}
    for suffix, dense in dense_variants.items():
        scores[f"semantic_raw{suffix}"] = dense_leave_group_out(dense, li, gi, w, L, everything)
        whitened = np.zeros((n, L))
        for k in range(FOLDS):
            train = fold != k
            queries = np.flatnonzero(fold == k)
            if len(queries) == 0:
                continue
            mean, matrix, _ = fit_transform("within_author_whitening", dense[train], li[train], w[train],
                                            np.random.default_rng(SEED + k))
            projected = unit_rows((dense - mean) @ matrix.T)
            whitened[queries] = dense_leave_group_out(projected, li, gi, w, L, queries)
        scores[f"semantic_whitened{suffix}"] = whitened
    dense32 = dense_variants[""].astype(np.float32)
    scores["chars_raw"] = v1.score_leave_group_out(dense32, chars, li, gi, L).lexical.astype(np.float64)
    scores["words_raw"] = v1.score_leave_group_out(dense32, words, li, gi, L).lexical.astype(np.float64)
    ranks = {name: ranks_of(s, li) for name, s in scores.items()}
    empty_rows = {}
    for name, matrix in (("chars_raw", chars), ("words_raw", words)):
        empty = np.asarray(matrix.getnnz(axis=1) == 0)
        ranks[name][empty] = L
        empty_rows[name] = int(empty.sum())
    return ranks, empty_rows


def lexical_matrices(documents: list[str]):
    """The protocol's two surface spaces over the given documents."""
    chars = v1.fit_tfidf(documents)
    words, _ = fit_words([" ".join(segment(d)) for d in documents])
    return chars, words.tocsr()


def report_systems(ranks: dict[str, np.ndarray], pop: dict) -> dict:
    w = pop["weights"]
    all_mask = np.ones(len(w), dtype=bool)
    out = {}
    for name, r in ranks.items():
        rr = 1.0 / r
        out[name] = {"mrr": round(float(np.mean(rr)), 4),
                     "mrr_component_weighted": round(weighted_mean(rr, w, all_mask), 4),
                     "recall_at_1": round(float(np.mean(r <= 1)), 4),
                     "recall_at_10": round(float(np.mean(r <= 10)), 4)}
    return out


def contrasts_for(ranks: dict[str, np.ndarray], pop: dict, pairs) -> list[dict]:
    rr = {name: 1.0 / r for name, r in ranks.items()}
    return paired_group_bootstrap(rr, pop["weights"], pop["group_ids"], np.ones(len(pop["weights"]), dtype=bool), pairs)


def lookup(contrasts: list[dict], system: str, minus: str) -> dict:
    return next(c for c in contrasts if (c["system"], c["minus"]) == (system, minus))


def verdict(contrasts_pub: list[dict], contrasts_by_arm: dict[str, list[dict]]) -> tuple[str, dict, list[str]]:
    """The reading rule. A primary contrast is kept in an arm if its interval excludes zero and its
    sign is the published sign; in an arm that carries a '_touched' semantic variant that variant
    must be kept too."""
    failures: list[str] = []
    verdicts: dict[str, dict] = {}
    for arm_name, contrasts in contrasts_by_arm.items():
        present = {(c["system"], c["minus"]) for c in contrasts}
        for system, minus in PRIMARY:
            variants = [(system, minus)]
            if (system, minus + TOUCHED) in present:
                variants.append((system, minus + TOUCHED))
            for s, m in variants:
                c = lookup(contrasts, s, m)
                p = lookup(contrasts_pub, system, minus)
                kept = bool(c["excludes_zero"] and np.sign(c["mrr_difference"]) == np.sign(p["mrr_difference"]))
                verdicts[f"{arm_name}: {s} - {m}"] = {"published_difference": p["mrr_difference"],
                                                      "arm_difference": c["mrr_difference"], "ci95": c["ci95"],
                                                      "sign_kept_and_clear_of_zero": kept}
                if not kept:
                    failures.append(f"{arm_name}: {s} - {m}")
    reading = ("not driven by repeated passages" if not failures
               else "at least partly carried by repeated passages; failing contrasts: " + "; ".join(failures))
    return reading, verdicts, failures


def print_arm(name: str, systems: dict, contrasts: list[dict]) -> None:
    print(f"== {name}: " + "  ".join(f"{s}={v['mrr']:.4f}" for s, v in systems.items()), flush=True)
    for c in contrasts:
        print(f"    {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              + ("" if c["excludes_zero"] else "  (includes zero)"), flush=True)


def movement(systems_arm: dict, systems_pub: dict) -> dict:
    return {name: round(systems_arm[name]["mrr"] - systems_pub[name.replace(TOUCHED, "")]["mrr"], 4) for name in systems_arm}


# ------------------------------------------------------------------ the analysis
def analyse(base: dict, expected: dict[str, float], tolerance: dict[str, float] | None = None) -> dict:
    """Everything after loading: checks, the published configuration, the detector, both arms, the
    reading. `base` is setup_full's dictionary; `expected` the published numbers the unchanged
    configuration must reproduce, each within `tolerance` (TOLERANCE by default)."""
    tolerance = dict(TOLERANCE) if tolerance is None else tolerance
    started = time.time()
    songs, label_of, rows = base["songs"], base["label_of"], base["rows"]
    n = len(songs)
    gi, fold_of_group = base["group_ids"], base["fold_of_group"]
    print(f"  {n:,} queries, {base['label_count']} labels, {int(gi.max()) + 1:,} groups", flush=True)

    # ---- checks of the rebuilt inputs against the protocol's
    lines_by_song = chunk_lines(rows, base["chunk_rows"])
    untouched, zero_counts = strip_songs(lines_by_song, label_of, {})
    if zero_counts["lines_stripped"] != 0 or any("".join(untouched[s][0]) != base["documents"][s] for s in songs):
        raise SystemExit("the documents rebuilt with nothing stripped differ from the protocol's documents")
    rebuilt = song_vectors(base["vectors"], base["chunk_rows"], songs)
    centroid_gap = float(np.abs(rebuilt - base["dense"]).max())
    print(f"  rebuilt centroids against the protocol's: max gap {centroid_gap:.2e}", flush=True)
    if centroid_gap > 1e-6:
        raise SystemExit("the chunk-mean song vectors rebuilt here differ from the protocol's centroids")
    published = make_population(songs, label_of, np.ones(n, dtype=bool), gi, fold_of_group)
    if (published["queries"] != n or not np.array_equal(published["label_index"], base["label_index"])
            or not np.array_equal(published["group_ids"], gi)
            or not np.allclose(published["weights"], base["weights"]) or not np.array_equal(published["fold"], base["fold"])):
        raise SystemExit("the population built here differs from the protocol's")

    # ---- the published configuration, which must reproduce the published numbers before anything else
    print("fitting the surface spaces on the published documents", flush=True)
    chars_pub, words_pub = lexical_matrices([base["documents"][s] for s in songs])
    print("scoring the published configuration", flush=True)
    ranks_pub, empty_pub = score_population(published, {"": base["dense"]}, chars_pub, words_pub)
    systems_pub = report_systems(ranks_pub, published)
    checks = {}
    for name, value_expected in expected.items():
        value = float(np.mean(1.0 / ranks_pub[name]))                     # unrounded, the plain mean as published
        gap = abs(value - value_expected)
        checks[name] = {"expected": value_expected, "recomputed": round(value, 4), "gap": round(gap, 5),
                        "tolerance": tolerance[name], "passed": bool(gap <= tolerance[name])}
        print(f"check {name}: recomputed {value:.4f} expected {value_expected} (gap {gap:.5f}, tolerance {tolerance[name]})",
              flush=True)
    if not all(c["passed"] for c in checks.values()):
        raise SystemExit(f"the published numbers do not reproduce: {checks}")
    contrasts_pub = contrasts_for(ranks_pub, published, list(PRIMARY) + list(SECONDARY))
    print_arm("published", systems_pub, contrasts_pub)

    # ---- the detector
    print("detecting shared passages within labels", flush=True)
    position = {s: i for i, s in enumerate(songs)}
    pairs = shared_passage_pairs(base["normalised"], label_of)
    pair_index = [(position[a], position[b]) for a, b in pairs]
    already = sum(1 for a, b in pair_index if gi[a] == gi[b])
    chunk_every, chunk_must = whole_chunk_pairs(rows, base["chunk_rows"], label_of)
    pair_set = set(pairs)
    missed = sum(1 for p in chunk_must if p not in pair_set)
    chunk_in_one_group = sum(1 for a, b in chunk_every if gi[position[a]] == gi[position[b]])
    print(f"  {len(pairs):,} same-label song pairs share a {PASSAGE_CHARACTERS}-character passage; "
          f"{already:,} already in one leakage group; {len(pairs) - already:,} new edges", flush=True)
    print(f"  audit rule (whole chunk >= {PASSAGE_CHARACTERS} characters): {len(chunk_every):,} same-label pairs, "
          f"{chunk_in_one_group:,} in one group, {missed} of {len(chunk_must):,} missed by the detector", flush=True)
    if missed:
        raise SystemExit("the k-gram detector misses whole-chunk matches of the audit's rule")
    songs_in_pairs = {s for p in pairs for s in p}
    songs_in_new_edges = {s for (a, b), p in zip(pair_index, pairs) if gi[a] != gi[b] for s in p}
    partners = Counter(s for p in pairs for s in p)
    detector = {
        "same_label_song_pairs_sharing_passage": len(pairs),
        "pairs_already_in_one_published_group": already,
        "pairs_merging_distinct_groups": len(pairs) - already,
        "songs_in_at_least_one_pair": len(songs_in_pairs),
        "songs_in_a_pair_across_published_groups": len(songs_in_new_edges),
        "labels_with_at_least_one_pair": len({label_of[s] for s in songs_in_pairs}),
        "partners_per_involved_song_quantiles": {str(p): int(np.percentile(list(partners.values()), p))
                                                 for p in (50, 75, 90, 99)} if partners else {},
        "audit_rule_same_label_pairs_whole_chunk": len(chunk_every),
        "audit_rule_pairs_in_one_published_group": chunk_in_one_group,
        "audit_rule_pairs_the_detector_must_find": len(chunk_must),
        "audit_rule_pairs_missed": missed,
    }

    # ---- arm (a): merged groups
    print("arm (a): merging groups", flush=True)
    merged_name = merge_groups(gi, pair_index)
    pop_a = make_population(songs, label_of, np.ones(n, dtype=bool), merged_name, fold_of_group)
    sizes_before = np.bincount(gi)
    sizes_after = np.asarray(list(Counter(merged_name.tolist()).values()))
    li_base = base["label_index"]
    groups_per_label_before = {l: len(set(gi[li_base == l].tolist())) for l in range(base["label_count"])}
    groups_per_label_after = {l: len(set(merged_name[li_base == l].tolist())) for l in range(base["label_count"])}
    published_groups_in: dict[int, set[int]] = defaultdict(set)
    for name, g in zip(merged_name.tolist(), gi.tolist()):
        published_groups_in[int(name)].add(int(g))
    merge_info = {
        "groups_before": int(len(sizes_before)), "groups_after": int(len(sizes_after)),
        "groups_merged_away": int(len(sizes_before) - len(sizes_after)),
        "largest_group_before": int(sizes_before.max()), "largest_group_after": int(sizes_after.max()),
        "multi_song_groups_before": int((sizes_before > 1).sum()), "multi_song_groups_after": int((sizes_after > 1).sum()),
        "songs_in_a_merged_group": int(sum(1 for name in merged_name.tolist() if len(published_groups_in[int(name)]) > 1)),
        "labels_losing_groups": int(sum(1 for l in groups_per_label_before if groups_per_label_after[l] < groups_per_label_before[l])),
        "minimum_groups_in_a_label_after": int(min(groups_per_label_after.values())),
        "median_groups_in_a_label_before": float(np.median(list(groups_per_label_before.values()))),
        "median_groups_in_a_label_after": float(np.median(list(groups_per_label_after.values()))),
        "labels_dropped_single_group": pop_a["labels_dropped_single_group"],
        "songs_dropped_with_their_label": pop_a["songs_dropped_with_their_label"],
        "queries": pop_a["queries"], "labels": pop_a["label_count"], "groups": pop_a["groups"],
    }
    print(f"  {merge_info['groups_before']:,} -> {merge_info['groups_after']:,} groups, largest "
          f"{merge_info['largest_group_before']} -> {merge_info['largest_group_after']}, "
          f"{merge_info['labels_losing_groups']} labels lose groups, {pop_a['labels_dropped_single_group']} dropped; "
          f"{pop_a['queries']:,} queries over {pop_a['label_count']} labels", flush=True)
    idx_a = pop_a["index"]
    if len(idx_a) == n:
        chars_a, words_a = chars_pub, words_pub
    else:
        print("  refitting the surface spaces on the reduced population", flush=True)
        chars_a, words_a = lexical_matrices([base["documents"][songs[i]] for i in idx_a.tolist()])
    print("  scoring", flush=True)
    ranks_a, empty_a = score_population(pop_a, {"": base["dense"][idx_a]}, chars_a, words_a)
    systems_a = report_systems(ranks_a, pop_a)
    contrasts_a = contrasts_for(ranks_a, pop_a, list(PRIMARY) + list(SECONDARY))
    print_arm("arm (a) merged groups", systems_a, contrasts_a)

    # ---- arm (b): stripped lines
    print("arm (b): stripping lines recurrent across leakage groups of a label", flush=True)
    group_of = {s: int(gi[i]) for i, s in enumerate(songs)}
    recurrent = recurrent_lines(lines_by_song, label_of, group_of)
    stripped, counts = strip_songs(lines_by_song, label_of, recurrent)
    docs_b = {s: "".join(stripped[s][0]) for s in songs}
    keep_b = np.asarray([len(v1.normalized_text(docs_b[s])) >= v1.MIN_EFFECTIVE_CHARACTERS for s in songs])
    pop_b = make_population(songs, label_of, keep_b, gi, fold_of_group)
    idx_b = pop_b["index"]
    songs_b = [songs[i] for i in idx_b.tolist()]
    keep_emptied = {s: ~stripped[s][1] for s in songs_b}
    keep_touched, fallbacks = {}, 0
    for s in songs_b:
        untouched_chunks = ~stripped[s][2]
        if untouched_chunks.any():
            keep_touched[s] = untouched_chunks
        else:
            keep_touched[s] = keep_emptied[s]
            fallbacks += 1
    stripped_total_b = int(sum(int(stripped[s][3].sum()) for s in songs_b))
    surviving_emptied = int(sum(int(stripped[s][3][keep_emptied[s]].sum()) for s in songs_b))
    surviving_touched = int(sum(int(stripped[s][3][keep_touched[s]].sum()) for s in songs_b))
    recurrence_bands = Counter("2" if g == 2 else "3-4" if g <= 4 else "5-9" if g <= 9 else "10-up" for g in recurrent.values())
    strip_info = {
        "recurrent_label_line_types": len(recurrent),
        "recurrent_types_by_groups_containing_them": dict(sorted(recurrence_bands.items())),
        "labels_with_a_recurrent_line": len({label for label, _ in recurrent}),
        "line_occurrences_stripped": int(counts["lines_stripped"]),
        "line_occurrences_total": int(counts["lines_total"]),
        "stripped_occurrences_by_normalised_length": {band: int(counts[f"lines_stripped_of_length_{band}"]) for band, _, _ in LINE_BANDS},
        "normalised_characters_stripped": int(counts["characters_stripped"]),
        "normalised_characters_total": int(counts["characters_total"]),
        "share_of_characters_stripped": round(counts["characters_stripped"] / max(counts["characters_total"], 1), 4),
        "songs_with_a_stripped_line": int(counts["songs_affected"]),
        "chunks_touched": int(counts["chunks_touched"]), "chunks_emptied": int(counts["chunks_emptied"]),
        "chunks_total": int(counts["chunks_total"]),
        "songs_dropped_below_minimum_length": pop_b["songs_dropped_by_length"],
        "labels_dropped_few_songs": pop_b["labels_dropped_few_songs"],
        "labels_dropped_single_group": pop_b["labels_dropped_single_group"],
        "songs_dropped_with_their_label": pop_b["songs_dropped_with_their_label"],
        "queries": pop_b["queries"], "labels": pop_b["label_count"], "groups": pop_b["groups"],
        "semantic_brackets": {
            "emptied": {"rule": "drop only chunks whose every line was stripped",
                        "share_of_stripped_characters_surviving_in_kept_chunks": round(surviving_emptied / max(stripped_total_b, 1), 4)},
            "touched": {"rule": "drop every chunk containing a stripped line; a song with no untouched chunk keeps its emptied set",
                        "songs_falling_back_to_emptied_set": fallbacks,
                        "share_of_stripped_characters_surviving_in_kept_chunks": round(surviving_touched / max(stripped_total_b, 1), 4)},
        },
    }
    print(f"  {len(recurrent):,} recurrent (label, line) types; {counts['lines_stripped']:,} line occurrences stripped "
          f"({strip_info['share_of_characters_stripped']:.1%} of characters) from {counts['songs_affected']:,} songs; "
          f"{counts['chunks_touched']:,} chunks touched, {counts['chunks_emptied']:,} emptied", flush=True)
    print(f"  population: {pop_b['queries']:,} queries over {pop_b['label_count']} labels "
          f"({pop_b['songs_dropped_by_length']} songs below length, {pop_b['labels_dropped_few_songs']} labels with too few songs, "
          f"{pop_b['labels_dropped_single_group']} with one group); touched bracket falls back for {fallbacks:,} songs", flush=True)
    print("  fitting the surface spaces on the stripped documents", flush=True)
    chars_b, words_b = lexical_matrices([docs_b[s] for s in songs_b])
    dense_b = {"": song_vectors(base["vectors"], base["chunk_rows"], songs_b, keep_emptied),
               TOUCHED: song_vectors(base["vectors"], base["chunk_rows"], songs_b, keep_touched)}
    print("  scoring", flush=True)
    ranks_b, empty_b = score_population(pop_b, dense_b, chars_b, words_b)
    systems_b = report_systems(ranks_b, pop_b)
    pairs_b = list(PRIMARY) + list(SECONDARY)
    pairs_b += [(a if a.startswith(("words", "chars")) else a + TOUCHED, b + TOUCHED)
                for a, b in PRIMARY + SECONDARY if b.startswith("semantic")]
    contrasts_b = contrasts_for(ranks_b, pop_b, pairs_b)
    print_arm("arm (b) stripped lines", systems_b, contrasts_b)

    # ---- the reading
    reading, verdicts, failures = verdict(contrasts_pub, {"merged_groups": contrasts_a, "stripped_lines": contrasts_b})
    print(f"reading: {reading}", flush=True)
    secondary = {}
    for arm_name, contrasts in (("published", contrasts_pub), ("merged_groups", contrasts_a), ("stripped_lines", contrasts_b)):
        secondary[arm_name] = {f"{c['system']} - {c['minus']}": {"difference": c["mrr_difference"], "ci95": c["ci95"],
                                                                "excludes_zero": c["excludes_zero"]}
                               for c in contrasts if "semantic_whitened" in c["system"] or "semantic_whitened" in c["minus"]}

    return {
        "analysis": "the headline contrasts with within-label repeated passages removed: merged leakage groups (a) and stripped recurrent lines (b)",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": base["label_count"], "groups": int(gi.max()) + 1},
        "design": {
            "detector": f"two same-label songs share a passage iff their normalised texts (NFKC, whitespace removed, lower-cased) "
                        f"share a {PASSAGE_CHARACTERS}-character substring; generalises the label-identity audit's whole-chunk rule, "
                        f"which it must subsume",
            "arm_a_merged_groups": "songs sharing a passage are joined into one leakage group (union of published groups); weights rebuilt; "
                                   "a merged group keeps the published fold of its lowest-numbered member; the corpus's text; the surface "
                                   "spaces refitted only if a label was dropped",
            "arm_b_stripped_lines": "a line recurring in songs of two or more published leakage groups of a label is removed from every song of "
                                    "that label before the spaces are built; population re-filtered by the protocol's length and label rules; "
                                    "published groups and folds kept over the survivors; the semantic space bracketed at the chunk grain "
                                    "(emptied, touched)",
            "spaces": {"semantic_raw": "unit chunk-mean song vector of the recorded BGE-M3 run, dense leave-group-out cosine",
                       "chars_raw": "character 2-5-gram TF-IDF, the protocol's sparse leave-group-out scorer",
                       "words_raw": "jieba 1-2-gram TF-IDF, the same scorer",
                       "semantic_whitened": "within-author whitening fitted per fold on the training folds, scored on the held-out fold",
                       "*_touched": "arm (b) only: the semantic space under the touched bracket"},
            "estimands": "MRR (plain mean, as published) and per-(group, label)-weighted MRR; contrasts are differences of weighted MRR with "
                         "paired group-bootstrap intervals (2000 replicates, seed 20260825) over the arm's own leakage groups",
            "checks": "published configuration reproduces 0.2997 / 0.4266 / 0.4963 within 0.0005 and 0.4164 within 0.002; rebuilt "
                      "population, documents and centroids equal the protocol's; every whole-chunk match of the audit's rule is found "
                      "by the detector",
            "reading_rule": "not driven by repeated passages if words - characters, characters - semantic and words - semantic keep their "
                            "published sign with intervals clear of zero in both arms, the semantic contrasts of arm (b) under both brackets; "
                            "otherwise at least partly carried by repeated passages, failing contrasts named",
            "not_in_the_rule": "contrasts involving the whitened semantic space; all counts; the movement of each space's MRR",
            "semantic_arm_b_caveat": "a line-level semantic arm needs re-embedding of the stripped text (GPU); the two chunk-grain brackets bound it",
        },
        "checks": {"against_published": checks, "rebuilt_centroids_max_gap": centroid_gap,
                   "population_equals_protocol": True, "documents_equal_protocol": True,
                   "audit_rule_pairs_missed_by_detector": missed},
        "empty_sparse_rows": {"published": empty_pub, "merged_groups": empty_a, "stripped_lines": empty_b},
        "detector": detector,
        "arms": {
            "published": {"population": {"queries": n, "labels": base["label_count"], "groups": int(gi.max()) + 1},
                          "systems": systems_pub, "paired_contrasts": contrasts_pub},
            "merged_groups": {"merge": merge_info, "systems": systems_a, "mrr_minus_published": movement(systems_a, systems_pub),
                              "paired_contrasts": contrasts_a},
            "stripped_lines": {"strip": strip_info, "systems": systems_b, "mrr_minus_published": movement(systems_b, systems_pub),
                               "paired_contrasts": contrasts_b},
        },
        "reading": {"verdict": reading, "failing_contrasts": failures, "primary_contrasts": verdicts,
                    "whitened_semantic_contrasts_description": secondary},
        "privacy": "aggregate only; no lyric text, line, song identifier or vector",
        "minutes": round((time.time() - started) / 60, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if corpus_version() != "v3":
        raise SystemExit("set CHINESE_RAP_CORPUS=v3; this analysis is defined on corpus v3")
    started = time.time()
    import jieba
    jieba.setLogLevel(60)
    print("loading corpus v3 and building the protocol's population", flush=True)
    base = setup_full(args.private_root.resolve())
    payload = analyse(base, EXPECTED)
    payload["minutes"] = round((time.time() - started) / 60, 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / OUT_NAME).write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nwrote {args.out_dir / OUT_NAME}  ({payload['minutes']} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
