#!/usr/bin/env python3
"""Synthetic test of within_label_repeats_v3: random data, no private corpus.

A small random corpus of twelve labels with planted repeats: a 32-character line shared by two
same-label songs in different groups, a 29-character one that must not count, a 32-character line
shared across two labels that must not count, a whole duplicated chunk (the audit's rule, which the
detector must subsume), a duplicated chunk whose normalised form is under 30 characters (the audit's
rule finds it, the detector need not), ad-lib lines recurring across groups under different casing
and spacing, a line recurring inside one group only, a five-song label whose songs all share a
passage (merged into one group, dropped in arm a), a song that falls under 50 characters once its
recurrent lines go, and a song whose every chunk carries a recurrent line (the touched bracket's
fallback). Each function is checked against a brute-force definition written independently here:

  * shared_passage_pairs against "some 30-character window of one song occurs in the other";
  * whole_chunk_pairs against "some stripped chunk of at least 30 characters is common", and the
    must-find subset lies inside the detector's pairs;
  * merge_groups against connected components over published groups, names and fold inheritance;
  * recurrent_lines and strip_songs against the definition line by line, counts included, and the
    no-op strip returns the corpus's own chunk texts;
  * song_vectors against a plain mean over kept chunks;
  * make_population against the protocol's population, weights and folds, on the full corpus (must
    equal the base exactly) and on a reduced one;
  * score_population against a brute-force leave-group-out cosine, dense, per-fold whitened and
    sparse with empty rows;
  * the reading rule on hand-made contrasts;
  * analyse() end to end with brute-force expected numbers: the reproduction check passes, the
    planted counts come out, a wrong expected number stops the run.

    set CHINESE_RAP_CORPUS=v3
    python test_within_label_repeats.py          (or: python -m pytest test_within_label_repeats.py)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

os.environ.setdefault("CHINESE_RAP_CORPUS", "v3")
HERE = Path(__file__).resolve().parent.parent / "src"
_spec = importlib.util.spec_from_file_location("within_label_repeats_v3", HERE / "within_label_repeats_v3.py")
wlr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wlr)

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, fit_transform, unit_rows  # noqa: E402
from leakage_groups_v2 import normalise_document  # noqa: E402

ALPHABET = list("天地人山水火风雷雨雪云月日星光影声色味形动静春夏秋冬江河湖海")
DIM = 16


# ------------------------------------------------------------------ synthetic corpus
def random_line(rng, lo=6, hi=14) -> str:
    return "".join(rng.choice(ALPHABET, size=int(rng.integers(lo, hi + 1))))


def make_corpus(seed: int = 11) -> dict:
    """Songs as lists of chunks, each a list of lines; labels; published group ids; the planted
    repeats recorded so the tests can look for them."""
    rng = np.random.default_rng(seed)
    labels = [f"L{i:02d}" for i in range(12)]
    sizes = {label: int(rng.integers(7, 13)) for label in labels}
    sizes["L11"] = 5                                     # the label that will merge into one group
    songs, label_of, chunks_of = [], {}, {}
    for label in labels:
        for _ in range(sizes[label]):
            song = f"S{len(songs) + 1:04d}"
            songs.append(song)
            label_of[song] = label
            chunks_of[song] = [[random_line(rng) for _ in range(int(rng.integers(4, 9)))]
                               for _ in range(int(rng.integers(2, 5)))]
    by_label = defaultdict(list)
    for song in songs:
        by_label[label_of[song]].append(song)

    # published groups: singletons, then a same-label pair and a cross-label pair
    group_of = {song: i for i, song in enumerate(songs)}
    group_of[by_label["L00"][1]] = group_of[by_label["L00"][0]]           # same label, one group
    group_of[by_label["L04"][0]] = group_of[by_label["L03"][0]]           # collaboration-like group
    group_of[by_label["L01"][2]] = group_of[by_label["L01"][3]]

    planted = {}
    # a 32-character passage in two L00 songs of different groups
    passage = "".join(rng.choice(ALPHABET, size=32))
    a, b = by_label["L00"][0], by_label["L00"][2]
    chunks_of[a][0].append(passage)
    chunks_of[b][1].insert(0, passage)
    planted["passage_pair"] = tuple(sorted((a, b)))
    # a chain of three L01 songs sharing a passage (two disjoint 31-character lines)
    p1 = "".join(rng.choice(ALPHABET, size=31))
    p2 = "".join(rng.choice(ALPHABET, size=31))
    c1, c2, c3 = by_label["L01"][0], by_label["L01"][1], by_label["L01"][4]
    chunks_of[c1][0].append(p1)
    chunks_of[c2][0].append(p1)
    chunks_of[c2][1].append(p2)
    chunks_of[c3][0].append(p2)
    planted["chain"] = (c1, c2, c3)
    # a 29-character line shared inside L02: not a passage
    short = "".join(rng.choice(ALPHABET, size=29))
    d1, d2 = by_label["L02"][0], by_label["L02"][1]
    chunks_of[d1][0].append(short)
    chunks_of[d2][0].append(short)
    planted["short_pair"] = tuple(sorted((d1, d2)))
    # a 32-character line shared across labels: not a same-label passage
    cross = "".join(rng.choice(ALPHABET, size=32))
    chunks_of[by_label["L03"][1]][0].append(cross)
    chunks_of[by_label["L04"][1]][0].append(cross)
    planted["cross_pair"] = tuple(sorted((by_label["L03"][1], by_label["L04"][1])))
    # a whole duplicated chunk in L05 (the audit's rule); padded so it has at least 30 raw and
    # normalised characters whatever the random lines came out as
    e1, e2 = by_label["L05"][0], by_label["L05"][3]
    chunks_of[e1][0].append("".join(rng.choice(ALPHABET, size=31)))
    chunks_of[e2].append(list(chunks_of[e1][0]))
    planted["whole_chunk_pair"] = tuple(sorted((e1, e2)))
    # a duplicated chunk of 39 raw but 20 normalised characters in L06
    spaced = ["天 地 人 山 水", "火 风 雷 雨 雪", "云 月 日 星 光", "影 声 色 味 形"]
    f1, f2 = by_label["L06"][0], by_label["L06"][2]
    chunks_of[f1].append(list(spaced))
    chunks_of[f2].append(list(spaced))
    planted["spaced_chunk_pair"] = tuple(sorted((f1, f2)))
    # every L11 song shares a passage: one merged group, dropped in arm (a)
    hook = "".join(rng.choice(ALPHABET, size=33))
    for song in by_label["L11"]:
        chunks_of[song][0].append(hook)
    # recurrent ad-lib lines in L00, different casing and spacing, different groups
    chunks_of[by_label["L00"][3]][0].append("YEAH yeah")
    chunks_of[by_label["L00"][4]][1].append("yeah YEAH")
    chunks_of[by_label["L00"][5]][0].append("yeahyeah")
    planted["recurrent_key"] = ("L00", "yeahyeah")
    # a line recurring inside one group only (L01 songs 2 and 3 share a group)
    chunks_of[by_label["L01"][2]][0].append("嘿嘿嘿")
    chunks_of[by_label["L01"][3]][0].append("嘿嘿嘿")
    planted["one_group_key"] = ("L01", "嘿嘿嘿")
    # a line in two labels, one group each: not recurrent anywhere
    chunks_of[by_label["L02"][2]][0].append("哦哦哦哦")
    chunks_of[by_label["L03"][2]][0].append("哦哦哦哦")
    # L07: a recurrent line R in three groups; one song made almost entirely of R: 58 normalised
    # characters before the strip (the protocol's length rule is 50), 10 after it
    r_line = "".join(rng.choice(ALPHABET, size=12))
    for song in by_label["L07"][:3]:
        chunks_of[song][0].append(r_line)
    z = by_label["L07"][4]
    chunks_of[z] = [[r_line, r_line, r_line], [r_line, "".join(rng.choice(ALPHABET, size=10))]]
    planted["song_dropped_after_strip"] = z
    # L08: a song whose every chunk carries a recurrent line but keeps enough text
    r2 = "".join(rng.choice(ALPHABET, size=11))
    chunks_of[by_label["L08"][0]][0].append(r2)
    y = by_label["L08"][1]
    for lines in chunks_of[y]:
        lines.append(r2)
    planted["song_all_chunks_touched"] = y

    return dict(songs=songs, label_of=label_of, chunks_of=chunks_of, group_of=group_of, planted=planted)


def build_base(corpus: dict, seed: int = 5) -> dict:
    """The dictionary setup_full would return, from the synthetic corpus: shuffled rows with a
    source order, random unit chunk vectors, the protocol's documents, groups, weights and folds."""
    rng = np.random.default_rng(seed)
    songs, label_of, chunks_of, group_of = corpus["songs"], corpus["label_of"], corpus["chunks_of"], corpus["group_of"]
    rows = []
    for song in songs:
        for c, lines in enumerate(chunks_of[song]):
            rows.append({"song_id": song, "chunk_id": str(c), "source_order": str(c), "cleaned_text": "\n".join(lines),
                         "source_credit_label": label_of[song]})
    perm = rng.permutation(len(rows))
    rows = [rows[i] for i in perm.tolist()]
    vectors = rng.normal(size=(len(rows), DIM)).astype(np.float32)
    chunk_rows = {s: sorted([i for i, r in enumerate(rows) if r["song_id"] == s], key=lambda i: int(rows[i]["source_order"]))
                  for s in songs}
    documents = {s: "".join(rows[i]["cleaned_text"] for i in chunk_rows[s]) for s in songs}
    normalised = {s: normalise_document(documents[s]) for s in songs}
    for s in songs:
        if len(v1.normalized_text(documents[s])) < v1.MIN_EFFECTIVE_CHARACTERS:
            raise RuntimeError("a synthetic song is too short for the protocol")
    eligible = sorted(set(label_of.values()))
    label_index = np.asarray([eligible.index(label_of[s]) for s in songs], dtype=np.int64)
    order = {g: i for i, g in enumerate(sorted(set(group_of.values())))}
    group_ids = np.asarray([order[group_of[s]] for s in songs], dtype=np.int64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold_of_group = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))
    fold = fold_of_group[group_ids]
    dense = v1.l2_normalize_dense(np.stack([vectors[chunk_rows[s]].mean(axis=0) for s in songs])).astype(np.float64)
    return dict(rows=rows, vectors=vectors, songs=songs, label_of=dict(label_of), label_index=label_index,
                group_ids=group_ids, weights=weights, fold=fold, fold_of_group=fold_of_group, label_count=len(eligible),
                dense=dense, documents=documents, normalised=normalised, chunk_rows=chunk_rows)


# ------------------------------------------------------------------ brute-force definitions
def brute_passage_pairs(normalised, label_of, k=30):
    songs = sorted(normalised)
    out = set()
    for i, a in enumerate(songs):
        for b in songs[i + 1:]:
            if label_of[a] != label_of[b]:
                continue
            ta, tb = normalised[a], normalised[b]
            if any(ta[s:s + k] in tb for s in range(len(ta) - k + 1)):
                out.add((a, b))
    return sorted(out)


def brute_lgo(x, label_index, group_ids, weights, label_count, queries):
    """Cosine of each query with the weighted sum of the label's songs outside the query's group."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((len(queries), label_count))
    for row, q in enumerate(queries.tolist()):
        for l in range(label_count):
            members = [s for s in range(len(label_index)) if label_index[s] == l and group_ids[s] != group_ids[q]]
            profile = sum(weights[s] * x[s] for s in members)
            out[row, l] = float(x[q] @ profile) / float(np.linalg.norm(profile))
    return out


def brute_ranks(scores, label_index):
    return v1.rank_system(scores.astype(np.float32), label_index)[0].astype(np.int64)


def components_over_groups(n_groups, edges):
    adjacency = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, component_of = set(), {}
    for start in range(n_groups):
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            g = stack.pop()
            members.append(g)
            for h in adjacency[g]:
                if h not in seen:
                    seen.add(h)
                    stack.append(h)
        for g in members:
            component_of[g] = min(members)
    return component_of


# ------------------------------------------------------------------ tests
CORPUS = make_corpus()
BASE = build_base(CORPUS)


def test_detector_against_brute_force():
    pairs = wlr.shared_passage_pairs(BASE["normalised"], BASE["label_of"])
    brute = brute_passage_pairs(BASE["normalised"], BASE["label_of"])
    assert pairs == brute, "the k-gram detector differs from the substring definition"
    planted = CORPUS["planted"]
    assert planted["passage_pair"] in pairs
    c1, c2, c3 = planted["chain"]
    assert tuple(sorted((c1, c2))) in pairs and tuple(sorted((c2, c3))) in pairs
    assert tuple(sorted((c1, c3))) not in pairs, "a chain is not a pair"
    assert planted["short_pair"] not in pairs, "29 characters must not count"
    assert planted["cross_pair"] not in pairs, "a cross-label share must not count"
    assert planted["whole_chunk_pair"] in pairs
    by_label = defaultdict(list)
    for s, l in BASE["label_of"].items():
        by_label[l].append(s)
    for i, a in enumerate(sorted(by_label["L11"])):
        for b in sorted(by_label["L11"])[i + 1:]:
            assert (a, b) in pairs


def test_whole_chunk_pairs_against_brute_force():
    every, must = wlr.whole_chunk_pairs(BASE["rows"], BASE["chunk_rows"], BASE["label_of"])
    texts = {s: [BASE["rows"][i]["cleaned_text"].strip() for i in BASE["chunk_rows"][s]] for s in BASE["songs"]}
    brute_every, brute_must = set(), set()
    songs = BASE["songs"]
    for i, a in enumerate(songs):
        for b in songs[i + 1:]:
            if BASE["label_of"][a] != BASE["label_of"][b]:
                continue
            common = {t for t in texts[a] if len(t) >= 30} & set(texts[b])
            if common:
                brute_every.add((a, b))
                if any(len(normalise_document(t)) >= 30 for t in common):
                    brute_must.add((a, b))
    assert set(every) == brute_every and set(must) == brute_must
    planted = CORPUS["planted"]
    assert planted["whole_chunk_pair"] in must
    assert planted["spaced_chunk_pair"] in every and planted["spaced_chunk_pair"] not in must
    detected = set(wlr.shared_passage_pairs(BASE["normalised"], BASE["label_of"]))
    assert set(must) <= detected, "the detector must subsume the audit's rule"


def test_merge_groups_against_components():
    pairs = wlr.shared_passage_pairs(BASE["normalised"], BASE["label_of"])
    position = {s: i for i, s in enumerate(BASE["songs"])}
    pair_index = [(position[a], position[b]) for a, b in pairs]
    gi = BASE["group_ids"]
    merged = wlr.merge_groups(gi, pair_index)
    edges = [(int(gi[a]), int(gi[b])) for a, b in pair_index]
    component_of = components_over_groups(int(gi.max()) + 1, edges)
    expect = np.asarray([component_of[int(g)] for g in gi.tolist()])
    assert np.array_equal(merged, expect), "merged names differ from the connected components"
    # a published group never splits; an unmerged group keeps its id (and so its fold)
    for g in set(gi.tolist()):
        assert len(set(merged[gi == g].tolist())) == 1
    touched = {int(gi[a]) for a, b in pair_index if gi[a] != gi[b]} | {int(gi[b]) for a, b in pair_index if gi[a] != gi[b]}
    for g in set(gi.tolist()) - touched:
        assert np.all(merged[gi == g] == g)
    by_label = defaultdict(list)
    for i, s in enumerate(BASE["songs"]):
        by_label[BASE["label_of"][s]].append(i)
    assert len(set(merged[by_label["L11"]].tolist())) == 1, "L11 must collapse into one group"
    assert len(set(merged[by_label["L09"]].tolist())) == len(by_label["L09"]), "an untouched label keeps its groups"


def test_recurrent_lines_and_strip_against_definition():
    lines_by_song = wlr.chunk_lines(BASE["rows"], BASE["chunk_rows"])
    for s in BASE["songs"]:
        assert lines_by_song[s] == CORPUS["chunks_of"][s], "chunk lines must come back in source order"
    group_of = {s: int(BASE["group_ids"][i]) for i, s in enumerate(BASE["songs"])}
    recurrent = wlr.recurrent_lines(lines_by_song, BASE["label_of"], group_of)
    brute = defaultdict(set)
    for s, chunks in lines_by_song.items():
        for lines in chunks:
            for line in lines:
                key = normalise_document(line)
                if key:
                    brute[(BASE["label_of"][s], key)].add(group_of[s])
    brute = {k: len(v) for k, v in brute.items() if len(v) >= 2}
    assert recurrent == brute
    planted = CORPUS["planted"]
    assert recurrent[planted["recurrent_key"]] == 3
    assert planted["one_group_key"] not in recurrent
    assert ("L02", "哦哦哦哦") not in recurrent and ("L03", "哦哦哦哦") not in recurrent
    # the strip, line by line
    stripped, counts = wlr.strip_songs(lines_by_song, BASE["label_of"], recurrent)
    exp_lines, exp_chars, exp_songs, exp_touched, exp_emptied, exp_total_lines, exp_total_chars = 0, 0, 0, 0, 0, 0, 0
    for s, chunks in lines_by_song.items():
        label = BASE["label_of"][s]
        texts, emptied, touched, chars = stripped[s]
        hit_song = False
        for c, lines in enumerate(chunks):
            kept = [l for l in lines if not (normalise_document(l) and (label, normalise_document(l)) in recurrent)]
            removed = [l for l in lines if normalise_document(l) and (label, normalise_document(l)) in recurrent]
            assert texts[c] == "\n".join(kept)
            assert bool(touched[c]) == bool(removed)
            assert bool(emptied[c]) == (bool(removed) and normalise_document("\n".join(kept)) == "")
            assert int(chars[c]) == sum(len(normalise_document(l)) for l in removed)
            exp_lines += len(removed)
            exp_chars += sum(len(normalise_document(l)) for l in removed)
            exp_touched += int(bool(removed))
            exp_emptied += int(bool(removed) and normalise_document("\n".join(kept)) == "")
            exp_total_lines += sum(1 for l in lines if normalise_document(l))
            exp_total_chars += sum(len(normalise_document(l)) for l in lines)
            hit_song = hit_song or bool(removed)
        exp_songs += int(hit_song)
    assert counts["lines_stripped"] == exp_lines and counts["characters_stripped"] == exp_chars
    assert counts["songs_affected"] == exp_songs and counts["chunks_touched"] == exp_touched
    assert counts["chunks_emptied"] == exp_emptied and counts["lines_total"] == exp_total_lines
    assert counts["characters_total"] == exp_total_chars
    assert counts["chunks_total"] == sum(len(c) for c in lines_by_song.values())
    assert sum(counts[f"lines_stripped_of_length_{b}"] for b, _, _ in wlr.LINE_BANDS) == exp_lines
    z = planted["song_dropped_after_strip"]
    assert len(v1.normalized_text("".join(stripped[z][0]))) < v1.MIN_EFFECTIVE_CHARACTERS
    y = planted["song_all_chunks_touched"]
    assert stripped[y][2].all() and not stripped[y][1].any()
    # the no-op strip returns the corpus's own chunk texts
    untouched, zero = wlr.strip_songs(lines_by_song, BASE["label_of"], {})
    assert zero["lines_stripped"] == 0 and zero["songs_affected"] == 0
    for s in BASE["songs"]:
        assert "".join(untouched[s][0]) == BASE["documents"][s]
        assert not untouched[s][1].any() and not untouched[s][2].any()


def test_song_vectors_against_mean():
    full = wlr.song_vectors(BASE["vectors"], BASE["chunk_rows"], BASE["songs"])
    assert np.abs(full - BASE["dense"]).max() < 1e-7
    rng = np.random.default_rng(3)
    keep = {}
    for s in BASE["songs"]:
        m = rng.random(len(BASE["chunk_rows"][s])) < 0.6
        if not m.any():
            m[0] = True
        keep[s] = m
    kept = wlr.song_vectors(BASE["vectors"], BASE["chunk_rows"], BASE["songs"], keep)
    for i, s in enumerate(BASE["songs"]):
        rows = [r for r, k in zip(BASE["chunk_rows"][s], keep[s]) if k]
        mean = np.mean([BASE["vectors"][r].astype(np.float64) for r in rows], axis=0)
        assert np.abs(kept[i] - mean / np.linalg.norm(mean)).max() < 1e-6
    try:
        wlr.song_vectors(BASE["vectors"], BASE["chunk_rows"], BASE["songs"][:1], {BASE["songs"][0]: np.zeros(len(BASE["chunk_rows"][BASE["songs"][0]]), dtype=bool)})
        raise AssertionError("a song with no chunk must raise")
    except RuntimeError:
        pass


def test_make_population():
    n = len(BASE["songs"])
    pop = wlr.make_population(BASE["songs"], BASE["label_of"], np.ones(n, dtype=bool), BASE["group_ids"], BASE["fold_of_group"])
    assert pop["queries"] == n and pop["label_count"] == BASE["label_count"]
    assert np.array_equal(pop["index"], np.arange(n))
    assert np.array_equal(pop["label_index"], BASE["label_index"]) and np.array_equal(pop["group_ids"], BASE["group_ids"])
    assert np.allclose(pop["weights"], BASE["weights"]) and np.array_equal(pop["fold"], BASE["fold"])
    assert pop["labels_dropped_few_songs"] == 0 and pop["labels_dropped_single_group"] == 0
    # a reduced corpus: drop songs so L09 has four left, and merge L11 into one group
    by_label = defaultdict(list)
    for i, s in enumerate(BASE["songs"]):
        by_label[BASE["label_of"][s]].append(i)
    keep = np.ones(n, dtype=bool)
    keep[by_label["L09"][4:]] = False
    name = BASE["group_ids"].copy()
    name[by_label["L11"]] = name[by_label["L11"]].min()
    pop = wlr.make_population(BASE["songs"], BASE["label_of"], keep, name, BASE["fold_of_group"])
    assert pop["labels_dropped_few_songs"] == 1 and pop["labels_dropped_single_group"] == 1
    expect_labels = sorted(l for l in by_label if l not in ("L09", "L11"))
    assert pop["label_count"] == len(expect_labels)
    expect_index = [i for i in range(n) if keep[i] and BASE["label_of"][BASE["songs"][i]] in expect_labels]
    assert pop["index"].tolist() == expect_index
    assert pop["label_index"].tolist() == [expect_labels.index(BASE["label_of"][BASE["songs"][i]]) for i in expect_index]
    names = name[pop["index"]]
    ranks = {g: r for r, g in enumerate(sorted(set(names.tolist())))}
    assert pop["group_ids"].tolist() == [ranks[int(g)] for g in names.tolist()]
    size = Counter(zip(pop["group_ids"].tolist(), pop["label_index"].tolist()))
    assert np.allclose(pop["weights"], [1.0 / size[(int(g), int(l))] for g, l in zip(pop["group_ids"], pop["label_index"])])
    assert np.array_equal(pop["fold"], BASE["fold_of_group"][names]), "a group keeps its published fold"
    assert pop["songs_dropped_by_length"] == int((~keep).sum())
    # the kept songs of both dropped labels leave: L09's four survivors and all of L11
    assert pop["songs_dropped_with_their_label"] == 4 + len(by_label["L11"])


def random_sparse_unit(rng, n, m, empty_rows=()):
    x = sparse.random(n, m, density=0.15, random_state=rng.integers(1 << 30), dtype=np.float32, data_rvs=lambda k: rng.random(k) + 0.1).tocsr()
    x = x.tolil()
    for r in empty_rows:
        x[r, :] = 0
    for r in range(n):
        if r not in empty_rows and x[r].count_nonzero() == 0:
            x[r, 0] = 1.0
    x = x.tocsr()
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    scale = np.where(norms > 0, 1.0 / np.maximum(norms, 1e-12), 0.0).astype(np.float32)
    return (sparse.diags(scale) @ x).tocsr().astype(np.float32)


def test_score_population_against_brute_force():
    n = len(BASE["songs"])
    pop = wlr.make_population(BASE["songs"], BASE["label_of"], np.ones(n, dtype=bool), BASE["group_ids"], BASE["fold_of_group"])
    li, gi, w, L, fold = pop["label_index"], pop["group_ids"], pop["weights"], pop["label_count"], pop["fold"]
    rng = np.random.default_rng(9)
    chars = random_sparse_unit(rng, n, 60)
    words = random_sparse_unit(rng, n, 45, empty_rows=(3, 17))
    other = unit_rows(rng.normal(size=(n, DIM)))
    ranks, empty = wlr.score_population(pop, {"": BASE["dense"], "_touched": other}, chars, words)
    everything = np.arange(n)
    assert np.array_equal(ranks["semantic_raw"], brute_ranks(brute_lgo(BASE["dense"], li, gi, w, L, everything), li))
    assert np.array_equal(ranks["semantic_raw_touched"], brute_ranks(brute_lgo(other, li, gi, w, L, everything), li))
    for suffix, x in (("", BASE["dense"]), ("_touched", other)):
        expect = np.zeros(n, dtype=np.int64)
        for k in range(FOLDS):
            train, queries = fold != k, np.flatnonzero(fold == k)
            if len(queries) == 0:
                continue
            mean, matrix, _ = fit_transform("within_author_whitening", x[train], li[train], w[train], np.random.default_rng(SEED + k))
            projected = unit_rows((x - mean) @ matrix.T)
            expect[queries] = brute_ranks(brute_lgo(projected, li, gi, w, L, queries), li[queries])
        assert np.array_equal(ranks[f"semantic_whitened{suffix}"], expect), f"whitened{suffix} differs from brute force"
    for name, x in (("chars_raw", chars), ("words_raw", words)):
        expect = brute_ranks(brute_lgo(x.toarray(), li, gi, w, L, everything), li)
        empties = np.flatnonzero(np.asarray(x.getnnz(axis=1) == 0))
        expect[empties] = L
        assert np.array_equal(ranks[name], expect), f"{name} differs from brute force"
    assert empty == {"chars_raw": 0, "words_raw": 2}
    report = wlr.report_systems(ranks, pop)
    assert abs(report["semantic_raw"]["mrr"] - float(np.mean(1.0 / ranks["semantic_raw"]))) < 1e-4
    weighted = float(np.sum(w / ranks["semantic_raw"]) / np.sum(w))
    assert abs(report["semantic_raw"]["mrr_component_weighted"] - weighted) < 1e-4
    contrasts = wlr.contrasts_for(ranks, pop, [("chars_raw", "semantic_raw")])
    point = float(np.sum(w / ranks["chars_raw"]) / np.sum(w) - weighted)
    assert abs(contrasts[0]["mrr_difference"] - point) < 1e-4, "the bootstrap point is the weighted MRR difference"


def contrast(system, minus, diff, lo, hi):
    return {"system": system, "minus": minus, "mrr_difference": diff, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}


def test_verdict_rule():
    pub = [contrast("words_raw", "chars_raw", 0.07, 0.05, 0.09), contrast("chars_raw", "semantic_raw", 0.13, 0.10, 0.16),
           contrast("words_raw", "semantic_raw", 0.20, 0.17, 0.23)]
    good = [contrast("words_raw", "chars_raw", 0.05, 0.02, 0.08), contrast("chars_raw", "semantic_raw", 0.10, 0.07, 0.13),
            contrast("words_raw", "semantic_raw", 0.15, 0.12, 0.18)]
    good_b = good + [contrast("chars_raw", "semantic_raw_touched", 0.12, 0.09, 0.15), contrast("words_raw", "semantic_raw_touched", 0.17, 0.14, 0.20)]
    reading, verdicts, failures = wlr.verdict(pub, {"merged_groups": good, "stripped_lines": good_b})
    assert reading == "not driven by repeated passages" and failures == []
    assert len(verdicts) == 3 + 5 and all(v["sign_kept_and_clear_of_zero"] for v in verdicts.values())
    # an interval touching zero fails; a flipped sign with a clear interval fails; a touched variant counts
    bad_a = [contrast("words_raw", "chars_raw", 0.01, -0.02, 0.04), good[1], good[2]]
    bad_b = good + [contrast("chars_raw", "semantic_raw_touched", -0.05, -0.08, -0.02), contrast("words_raw", "semantic_raw_touched", 0.17, 0.14, 0.20)]
    reading, verdicts, failures = wlr.verdict(pub, {"merged_groups": bad_a, "stripped_lines": bad_b})
    assert reading.startswith("at least partly carried by repeated passages")
    assert failures == ["merged_groups: words_raw - chars_raw", "stripped_lines: chars_raw - semantic_raw_touched"]


def brute_expected(base):
    """MRRs of the four published spaces on the synthetic corpus, from the protocol's TF-IDF fits
    and the brute-force leave-group-out cosine."""
    n = len(base["songs"])
    pop = wlr.make_population(base["songs"], base["label_of"], np.ones(n, dtype=bool), base["group_ids"], base["fold_of_group"])
    li, gi, w, L, fold = pop["label_index"], pop["group_ids"], pop["weights"], pop["label_count"], pop["fold"]
    chars, words = wlr.lexical_matrices([base["documents"][s] for s in base["songs"]])
    everything = np.arange(n)
    out = {}
    out["semantic_raw"] = brute_ranks(brute_lgo(base["dense"], li, gi, w, L, everything), li)
    whitened = np.zeros(n, dtype=np.int64)
    for k in range(FOLDS):
        train, queries = fold != k, np.flatnonzero(fold == k)
        if len(queries) == 0:
            continue
        mean, matrix, _ = fit_transform("within_author_whitening", base["dense"][train], li[train], w[train], np.random.default_rng(SEED + k))
        projected = unit_rows((base["dense"] - mean) @ matrix.T)
        whitened[queries] = brute_ranks(brute_lgo(projected, li, gi, w, L, queries), li[queries])
    out["semantic_whitened"] = whitened
    for name, x in (("chars_raw", chars), ("words_raw", words)):
        r = brute_ranks(brute_lgo(x.toarray(), li, gi, w, L, everything), li)
        r[np.asarray(x.getnnz(axis=1) == 0)] = L
        out[name] = r
    return {name: round(float(np.mean(1.0 / r)), 4) for name, r in out.items()}


def test_analyse_end_to_end():
    import jieba
    jieba.setLogLevel(60)
    expected = brute_expected(BASE)
    payload = wlr.analyse(BASE, expected)
    json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for name, check in payload["checks"]["against_published"].items():
        assert check["recomputed"] == expected[name] and check["passed"] and check["gap"] < 5e-5, f"{name}: {check}"
        assert check["tolerance"] == wlr.TOLERANCE[name]
    assert payload["checks"]["audit_rule_pairs_missed_by_detector"] == 0
    planted = CORPUS["planted"]
    detector = payload["detector"]
    brute = brute_passage_pairs(BASE["normalised"], BASE["label_of"])
    assert detector["same_label_song_pairs_sharing_passage"] == len(brute)
    merge = payload["arms"]["merged_groups"]["merge"]
    assert merge["labels_dropped_single_group"] == 1 and merge["songs_dropped_with_their_label"] == 5
    assert merge["labels"] == BASE["label_count"] - 1 and merge["queries"] == len(BASE["songs"]) - 5
    assert merge["groups_after"] < merge["groups_before"]
    strip = payload["arms"]["stripped_lines"]["strip"]
    lines_by_song = wlr.chunk_lines(BASE["rows"], BASE["chunk_rows"])
    group_of = {s: int(BASE["group_ids"][i]) for i, s in enumerate(BASE["songs"])}
    recurrent = wlr.recurrent_lines(lines_by_song, BASE["label_of"], group_of)
    _, counts = wlr.strip_songs(lines_by_song, BASE["label_of"], recurrent)
    assert strip["recurrent_label_line_types"] == len(recurrent)
    assert strip["line_occurrences_stripped"] == counts["lines_stripped"]
    assert strip["songs_with_a_stripped_line"] == counts["songs_affected"]
    assert strip["songs_dropped_below_minimum_length"] >= 1
    assert strip["semantic_brackets"]["touched"]["songs_falling_back_to_emptied_set"] >= 1
    assert 0.0 <= strip["semantic_brackets"]["touched"]["share_of_stripped_characters_surviving_in_kept_chunks"] \
        <= strip["semantic_brackets"]["emptied"]["share_of_stripped_characters_surviving_in_kept_chunks"] <= 1.0
    for arm in ("published", "merged_groups", "stripped_lines"):
        systems = payload["arms"][arm]["systems"]
        for name, entry in systems.items():
            assert 0.0 < entry["mrr"] <= 1.0 and np.isfinite(entry["mrr_component_weighted"])
        for c in payload["arms"][arm]["paired_contrasts"]:
            assert c["ci95"][0] <= c["ci95"][1] and np.isfinite(c["mrr_difference"])
    b_names = set(payload["arms"]["stripped_lines"]["systems"])
    assert {"semantic_raw_touched", "semantic_whitened_touched", "chars_raw", "words_raw", "semantic_raw", "semantic_whitened"} == b_names
    assert payload["reading"]["verdict"] == "not driven by repeated passages" or \
        payload["reading"]["verdict"].startswith("at least partly carried by repeated passages")
    assert len(payload["reading"]["primary_contrasts"]) == 3 + 5
    # a wrong published number stops the run before any arm: one past the fold-wise tolerance,
    # one past the re-scored tolerance but inside the fold-wise one
    for name, offset in (("words_raw", 0.01), ("semantic_raw", 0.001)):
        wrong = dict(expected)
        wrong[name] = round(expected[name] + offset, 4)
        try:
            wlr.analyse(BASE, wrong)
            raise AssertionError(f"a reproduction failure of {name} must raise SystemExit")
        except SystemExit:
            pass


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(tests) - failed} of {len(tests)} passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
