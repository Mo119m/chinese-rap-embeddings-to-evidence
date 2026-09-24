#!/usr/bin/env python3
"""Synthetic checks for src/twin_control_v3.py: random data, no private corpus, seconds to run.

A small random corpus of ten labels with planted same-label repeated passages across leakage groups,
so arm (a)'s merge has clusters to make and the control has something to match. Every estimator the
file adds is checked against a brute-force or independent definition:

  * label_clusters against connected components of the merge, per label, and against the realised
    per-label component loss (the identity the control's matching target rests on);
  * merged_groups_of and union_grouping against connected components over published groups;
  * merge_report against counts computed here from scratch;
  * draw_size_matched: it never takes a group arm (a) merged, never merges a cross-group twin pair,
    loses exactly as many components per label as arm (a), never costs a THIRD label a component
    (on a purpose-built base where the unconstrained draw would), picks the closest available
    component size at every step, and is reproducible from its seed;
  * score_space at beta 0 against a brute-force leave-group-out cosine, over the whole beta grid
    against attention_exemplar_v3.brute_attention, and on a sparse space at beta 0 alone;
  * top_attention_weight against an argmax share written out by hand, including a block that would
    straddle folds in a fold-wise space;
  * quintile_of against np.quantile, selected_rr against a per-fold pick;
  * query_weighted, component_weighted and label_macro against independent definitions;
  * the difference-in-differences pseudo-system against the four-term difference it stands for;
  * gain_reading, clause_one_verdict (an undecided space must dominate a bounded one),
    concentration_reading and residual_reading on hand-made contrasts, including the underpowered
    cases that must NOT read as a confirmation;
  * stratum withholds an interval below the labels/queries threshold;
  * curve_shape on monotone and interior-maximum curves;
  * the cell cache round trip, and betas_for's published/other-arm split;
  * the bands are the songs-per-label bands, not the components-per-label ones.

    set CHINESE_RAP_CORPUS=v3
    python tests/test_twin_control.py        (or: python -m pytest tests/test_twin_control.py)
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

os.environ.setdefault("CHINESE_RAP_CORPUS", "v3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import attention_exemplar_v3 as ae  # noqa: E402
import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
import twin_control_v3 as tc  # noqa: E402
import within_label_repeats_v3 as wlr  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import normalise_document  # noqa: E402

ALPHABET = list("天地人山水火风雷雨雪云月日星光影声色味形动静春夏秋冬江河湖海")
DIM = 14


# ------------------------------------------------------------------ a synthetic corpus
def make_base(seed: int = 7) -> dict:
    """The fields of within_label_repeats_v3.setup_full this file uses, on random data."""
    rng = np.random.default_rng(seed)
    labels = [f"L{i:02d}" for i in range(10)]
    songs, label_of, text_of = [], {}, {}
    for label in labels:
        # 10 labels x at least 11 songs, so the fixture ALWAYS clears MIN_CELL_QUERIES = 100. With
        # rng.integers(8, 13) the total averaged exactly 100 and the "this cell counts" assertion
        # below was a coin flip on the seed.
        for _ in range(int(rng.integers(11, 15))):
            song = f"S{len(songs) + 1:04d}"
            songs.append(song)
            label_of[song] = label
            text_of[song] = "\n".join("".join(rng.choice(ALPHABET, size=int(rng.integers(8, 15))))
                                      for _ in range(6))
    by_label: dict[str, list[str]] = defaultdict(list)
    for song in songs:
        by_label[label_of[song]].append(song)

    group_of = {song: i for i, song in enumerate(songs)}
    group_of[by_label["L00"][1]] = group_of[by_label["L00"][0]]          # two songs in one group
    group_of[by_label["L05"][0]] = group_of[by_label["L04"][0]]          # a cross-label group

    planted = {}
    for label, (i, j) in (("L01", (0, 2)), ("L02", (1, 4)), ("L03", (0, 3)), ("L06", (2, 5))):
        passage = "".join(rng.choice(ALPHABET, size=32))
        text_of[by_label[label][i]] += "\n" + passage
        text_of[by_label[label][j]] += "\n" + passage
        planted[label] = (by_label[label][i], by_label[label][j])
    first = "".join(rng.choice(ALPHABET, size=31))
    second = "".join(rng.choice(ALPHABET, size=31))
    text_of[by_label["L07"][0]] += "\n" + first
    text_of[by_label["L07"][1]] += "\n" + first + "\n" + second
    text_of[by_label["L07"][4]] += "\n" + second
    planted["chain"] = tuple(by_label["L07"][k] for k in (0, 1, 4))
    inside = "".join(rng.choice(ALPHABET, size=33))                      # a pair inside one group
    text_of[by_label["L00"][0]] += "\n" + inside
    text_of[by_label["L00"][1]] += "\n" + inside
    text_of[by_label["L08"][0]] += "\n" + "".join(rng.choice(ALPHABET, size=29))
    text_of[by_label["L08"][1]] += text_of[by_label["L08"][0]][-29:]     # 29 characters must not count

    rows, chunk_rows = [], {}
    for song in songs:
        chunk_rows[song] = [len(rows)]
        rows.append({"song_id": song, "chunk_id": "0", "source_order": "0",
                     "cleaned_text": text_of[song], "source_credit_label": label_of[song]})
    documents = {s: text_of[s] for s in songs}
    for song in songs:
        if len(v1.normalized_text(documents[song])) < v1.MIN_EFFECTIVE_CHARACTERS:
            raise RuntimeError("a synthetic song is too short for the protocol")
    eligible = sorted(set(label_of.values()))
    label_index = np.asarray([eligible.index(label_of[s]) for s in songs], dtype=np.int64)
    order = {g: i for i, g in enumerate(sorted(set(group_of.values())))}
    group_ids = np.asarray([order[group_of[s]] for s in songs], dtype=np.int64)
    size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold_of_group = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))
    vectors = np.random.default_rng(seed + 1).normal(size=(len(rows), DIM))
    dense = v1.l2_normalize_dense(np.stack([vectors[chunk_rows[s]].mean(axis=0)
                                            for s in songs])).astype(np.float64)
    return dict(rows=rows, vectors=vectors, songs=songs, label_of=dict(label_of), label_index=label_index,
                group_ids=group_ids, weights=weights, fold=fold_of_group[group_ids],
                fold_of_group=fold_of_group, label_count=len(eligible), dense=dense, documents=documents,
                normalised={s: normalise_document(documents[s]) for s in songs}, chunk_rows=chunk_rows,
                planted=planted)


BASE = make_base()
N = len(BASE["songs"])
POSITION = {s: i for i, s in enumerate(BASE["songs"])}
PAIRS = wlr.shared_passage_pairs(BASE["normalised"], BASE["label_of"])
PAIR_INDEX = [(POSITION[a], POSITION[b]) for a, b in PAIRS]
NAME_A = wlr.merge_groups(BASE["group_ids"], PAIR_INDEX)
POP_A = wlr.make_population(BASE["songs"], BASE["label_of"], np.ones(N, dtype=bool), NAME_A,
                            BASE["fold_of_group"])
POP_PUB = wlr.make_population(BASE["songs"], BASE["label_of"], np.ones(N, dtype=bool), BASE["group_ids"],
                              BASE["fold_of_group"])
CLUSTERS = tc.label_clusters(BASE, NAME_A)
FORBIDDEN = tc.merged_groups_of(BASE, NAME_A)
SIZES = tc.component_sizes(BASE)
POOL = tc.groups_by_label(BASE)


def brute_components(n_groups, edges):
    adjacency = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, out = set(), {}
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
            out[g] = min(members)
    return out


def partition(naming, group_ids):
    return {frozenset(np.flatnonzero(naming == value).tolist()) for value in set(naming.tolist())}


def brute_lgo(x, li, gi, w, L, queries):
    x = np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float64)
    out = np.zeros((len(queries), L))
    for row, q in enumerate(queries.tolist()):
        for l in range(L):
            members = [s for s in range(len(li)) if li[s] == l and gi[s] != gi[q]]
            profile = sum(w[s] * x[s] for s in members)
            out[row, l] = float(x[q] @ profile) / float(np.linalg.norm(profile))
    return out


# ------------------------------------------------------------------ arm (a) as an object
def test_planted_pairs_are_detected():
    # the skip has to happen before the unpacking: "chain" holds three songs, not two
    for label, planted in BASE["planted"].items():
        if label == "chain":
            continue
        a, b = planted
        assert tuple(sorted((a, b))) in PAIRS, label
    first, middle, last = BASE["planted"]["chain"]
    assert tuple(sorted((first, middle))) in PAIRS and tuple(sorted((middle, last))) in PAIRS
    assert tuple(sorted((first, last))) not in PAIRS, "a chain is not a pair"
    assert any(BASE["group_ids"][a] != BASE["group_ids"][b] for a, b in PAIR_INDEX), "nothing merges"


def test_label_clusters_are_the_realised_component_loss():
    for label in range(BASE["label_count"]):
        rows = np.flatnonzero(BASE["label_index"] == label)
        by_name = defaultdict(set)
        for i in rows.tolist():
            by_name[int(NAME_A[i])].add(int(BASE["group_ids"][i]))
        expect = sorted(sorted(groups) for groups in by_name.values() if len(groups) > 1)
        assert CLUSTERS.get(label, []) == expect, label
    assert any(len(c) >= 3 for clusters in CLUSTERS.values() for c in clusters), "the chain must give a triple"
    # the identity the control's matching target rests on: cluster arithmetic == realised loss, per label
    lost = tc.components_lost_per_label(BASE, NAME_A)
    for label in range(BASE["label_count"]):
        assert lost[label] == sum(len(c) - 1 for c in CLUSTERS.get(label, [])), label
    before = len(set(zip(BASE["group_ids"].tolist(), BASE["label_index"].tolist())))
    after = len(set(zip(NAME_A.tolist(), BASE["label_index"].tolist())))
    assert sum(lost.values()) == before - after
    assert tc.components_lost_per_label(BASE, BASE["group_ids"]) == {l: 0 for l in range(BASE["label_count"])}


def test_merged_groups_of_and_union_grouping():
    gi = BASE["group_ids"]
    edges = [(int(gi[a]), int(gi[b])) for a, b in PAIR_INDEX]
    component_of = brute_components(int(gi.max()) + 1, edges)
    assert np.array_equal(NAME_A, np.asarray([component_of[int(g)] for g in gi.tolist()]))
    expect = {g for g in range(int(gi.max()) + 1)
              if sum(1 for h in range(int(gi.max()) + 1) if component_of[h] == component_of[g]) > 1}
    assert tc.merged_groups_of(BASE, NAME_A) == expect
    # the union of two namings is the components of the union of their edges, as a partition of songs
    other = gi.copy()
    swap = sorted(set(gi.tolist()))[:4]
    other[np.isin(gi, swap)] = swap[0]
    union = tc.union_grouping(BASE, [NAME_A, other])
    both = brute_components(int(gi.max()) + 1, edges + [(swap[0], g) for g in swap[1:]])
    assert partition(union, gi) == partition(np.asarray([both[int(g)] for g in gi.tolist()]), gi)
    assert np.array_equal(tc.union_grouping(BASE, [gi]), gi), "the published naming is a fixed point"
    assert len(np.unique(union)) <= len(np.unique(NAME_A)), "a union is never finer"


def test_merge_report_against_brute_force():
    report = tc.merge_report(BASE, NAME_A, POP_A)
    gi, li = BASE["group_ids"], BASE["label_index"]
    assert report["groups_before"] == len(set(gi.tolist()))
    assert report["groups_after"] == len(set(NAME_A.tolist()))
    assert report["groups_merged_away"] == report["groups_before"] - report["groups_after"]
    assert report["largest_group_after"] == max(Counter(NAME_A.tolist()).values())
    holds = defaultdict(set)
    for name, g in zip(NAME_A.tolist(), gi.tolist()):
        holds[int(name)].add(int(g))
    assert report["songs_in_a_merged_group"] == sum(1 for name in NAME_A.tolist() if len(holds[int(name)]) > 1)
    before = {l: len(set(gi[li == l].tolist())) for l in range(BASE["label_count"])}
    after = {l: len(set(NAME_A[li == l].tolist())) for l in range(BASE["label_count"])}
    assert report["labels_losing_groups"] == sum(1 for l in before if after[l] < before[l]) == len(CLUSTERS)
    assert report["components_before"] - report["components_after"] == \
        sum(len(c) - 1 for v in CLUSTERS.values() for c in v)
    assert tc.merge_report(BASE, gi, POP_PUB)["groups_merged_away"] == 0


# ------------------------------------------------------------------ the size-matched control
def test_draw_is_size_matched_and_twin_free():
    gi = BASE["group_ids"]
    lost_a = tc.components_lost_per_label(BASE, NAME_A)
    seen = []
    for draw in range(6):
        rng = np.random.default_rng(SEED + tc.DRAW_SEED_OFFSET + draw)
        pairs, achieved = tc.draw_size_matched(BASE, CLUSTERS, FORBIDDEN, SIZES, POOL, rng)
        name = wlr.merge_groups(gi, pairs)
        seen.append(name.copy())
        assert achieved["components_merged_away"] == sum(lost_a.values()), achieved
        assert achieved["clusters_short_of_available_groups"] == 0, achieved
        assert tc.components_lost_per_label(BASE, name) == lost_a, "components lost per label must match"
        for group in tc.merged_groups_of(BASE, name):
            assert group not in FORBIDDEN, "the control took a group arm (a) merged"
        for a, b in PAIR_INDEX:
            if gi[a] != gi[b]:
                assert name[a] != name[b], "the control merged a cross-group twin pair"
        assert 0.0 <= achieved["component_size_match"] <= 1.0
    again, _ = tc.draw_size_matched(BASE, CLUSTERS, FORBIDDEN, SIZES, POOL,
                                    np.random.default_rng(SEED + tc.DRAW_SEED_OFFSET))
    assert np.array_equal(wlr.merge_groups(gi, again), seen[0]), "a draw must be reproducible"
    assert any(not np.array_equal(seen[0], other) for other in seen[1:]), "every draw is the same merge"


def test_the_pool_cannot_reach_a_cross_group_twin_pair():
    gi = BASE["group_ids"]
    reachable = [(a, b) for a, b in PAIR_INDEX
                 if gi[a] != gi[b] and int(gi[a]) not in FORBIDDEN and int(gi[b]) not in FORBIDDEN]
    assert not reachable, "merged_groups_of must cover both groups of every cross-group pair"


def test_draw_never_costs_a_third_label_a_component():
    """A base where the unconstrained draw would: label 0's pool holds two groups that also carry
    label 1, so merging that pair would shrink label 1's repertoire as well."""
    label_index = np.asarray([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.int64)
    group_ids = np.asarray([0, 0, 1, 1, 2, 3, 4, 5], dtype=np.int64)
    base = {"label_index": label_index, "group_ids": group_ids, "label_count": 2}
    clusters = {0: [[4, 5]]}                            # arm (a) merged groups 4 and 5
    forbidden = {4, 5}
    sizes = tc.component_sizes(base)
    pool = tc.groups_by_label(base)
    assert pool[0] == [0, 1, 2, 3, 4, 5] and pool[1] == [0, 1]
    lost_a = tc.components_lost_per_label(base, wlr.merge_groups(group_ids, [(6, 7)]))
    assert lost_a == {0: 1, 1: 0}
    saw_pair = False
    for seed in range(40):
        pairs, achieved = tc.draw_size_matched(base, clusters, forbidden, sizes, pool,
                                               np.random.default_rng(seed))
        name = wlr.merge_groups(group_ids, pairs)
        assert tc.components_lost_per_label(base, name) == lost_a, (seed, pairs)
        assert achieved["clusters_short_of_available_groups"] == 0, (seed, achieved)
        drawn = tc.merged_groups_of(base, name)
        assert not ({0, 1} <= drawn), "the draw merged the two groups that share label 1"
        saw_pair = saw_pair or bool(drawn & {0, 1})
    assert saw_pair, "the constraint was never exercised: no draw ever touched a shared group"


def test_greedy_picks_the_closest_available_size():
    """On a hand-made pool the greedy matcher must take a closest-size group at every step."""
    label = next(iter(CLUSTERS))
    pool = {label: [90, 91, 92, 93]}
    clusters = {label: [[80, 81]]}                      # the cluster's groups are NOT in the pool
    # the pool holds 1, 5, 2, 9 and the target is (5, 1): both sizes are present exactly
    sizes = Counter({(90, label): 1, (91, label): 5, (92, label): 2, (93, label): 9,
                     (80, label): 5, (81, label): 1})
    for seed in range(12):
        _, achieved = _draw_on_fake_pool(clusters, sizes, pool, np.random.default_rng(seed))
        assert achieved["component_size_l1_distance"] == 0, achieved
        assert achieved["component_size_target_mass"] == 6, achieved
        assert achieved["component_size_match"] == 1.0, achieved
    # the exact sizes absent: a target of (5, 5) against a pool of 1, 2, 9, 4 must take 4 then 2
    sizes2 = Counter({(90, label): 1, (91, label): 2, (92, label): 9, (93, label): 4,
                      (80, label): 5, (81, label): 5})
    _, achieved = _draw_on_fake_pool(clusters, sizes2, pool, np.random.default_rng(3))
    assert achieved["component_size_l1_distance"] == abs(4 - 5) + abs(2 - 5), achieved
    assert achieved["largest_component_drawn"] == 4, achieved


def _draw_on_fake_pool(clusters, sizes, pool, rng):
    """draw_size_matched on a base whose group ids are the pool's plus the cluster's, all one label,
    so only the achieved match is under test."""
    label = next(iter(clusters))
    members = sorted(pool[label]) + sorted({g for c in clusters[label] for g in c})
    fake = {"group_ids": np.asarray(members, dtype=np.int64),
            "label_index": np.full(len(members), label, dtype=np.int64),
            "label_count": label + 1}
    return tc.draw_size_matched(fake, clusters, set(), sizes, pool, rng)


# ------------------------------------------------------------------ the scorers
def test_score_space_beta_zero_and_grid():
    li, gi, w, L = POP_PUB["label_index"], POP_PUB["group_ids"], POP_PUB["weights"], POP_PUB["label_count"]
    x = BASE["dense"]
    rr = tc.score_space("semantic_raw", {0: x}, POP_PUB, (0.0,))
    expect = 1.0 / v1.rank_system(brute_lgo(x, li, gi, w, L, np.arange(N)).astype(np.float32), li)[0]
    assert np.abs(rr[0.0] - expect).max() < 1e-12, "beta 0 is not the protocol's prototype"
    assert np.abs(brute_lgo(x, li, gi, w, L, np.arange(N))
                  - dense_leave_group_out(x, li, gi, w, L, np.arange(N))).max() < 1e-9
    rr_grid = tc.score_space("words_raw", {0: x}, POP_PUB, tuple(ae.BETAS))
    assert np.abs(rr_grid[0.0] - rr[0.0]).max() < 1e-12, "the grid's beta 0 differs from the prototype"
    assert any(np.abs(rr_grid[b] - rr_grid[0.0]).max() > 0 for b in ae.BETAS if b > 0), "the grid is flat"
    # a sparse space at beta 0 alone must not go near dense_leave_group_out's weight column
    rows = sparse.random(N, 120, density=0.2, random_state=5, format="csr")
    rows.data = np.abs(rows.data) + 0.1
    sx = normalize(rows, norm="l2", axis=1).tocsr().astype(np.float64)
    sparse_rr = tc.score_space("chars_raw", {0: sx}, POP_PUB, (0.0,))
    reference = v1.score_leave_group_out(x.astype(np.float32), sx.astype(np.float32), li, gi, L)
    assert np.abs(sparse_rr[0.0] - 1.0 / v1.rank_system(reference.lexical, li)[0]).max() < 1e-12


def test_top_attention_weight_against_brute_force():
    li, gi, w, L = POP_PUB["label_index"], POP_PUB["group_ids"], POP_PUB["weights"], POP_PUB["label_count"]
    x = BASE["dense"]
    beta_of_query = np.full(N, 7.0)
    got = tc.top_attention_weight({0: x}, POP_PUB, beta_of_query, "words_raw", block=13)
    for query in range(0, N, 3):
        label = int(li[query])
        members = [s for s in np.flatnonzero(li == label).tolist() if gi[s] != gi[query]]
        cosines = np.asarray([float(x[query] @ x[s]) for s in members])
        attention = np.asarray([w[s] for s in members]) * np.exp(7.0 * (cosines - cosines.max()))
        assert abs(got[query] - attention.max() / attention.sum()) < 1e-12, query
    # at beta 0 the share is the largest weight's share of the weight mass
    zero = tc.top_attention_weight({0: x}, POP_PUB, np.zeros(N), "words_raw", block=13)
    for query in range(0, N, 7):
        label = int(li[query])
        members = [s for s in np.flatnonzero(li == label).tolist() if gi[s] != gi[query]]
        weights = np.asarray([w[s] for s in members])
        assert abs(zero[query] - weights.max() / weights.sum()) < 1e-12, query
    # a fold-wise space with a block wider than any single fold: rows are grouped by fold, not sliced
    matrices = {k: np.roll(x, k, axis=1) for k in range(FOLDS)}
    fold_wise = tc.top_attention_weight(matrices, POP_PUB, beta_of_query, "semantic_whitened", block=N)
    assert np.all(np.isfinite(fold_wise)) and np.all(fold_wise > 0)
    for query in range(0, N, 5):
        label, k = int(li[query]), int(POP_PUB["fold"][query])
        y = matrices[k]
        members = [s for s in np.flatnonzero(li == label).tolist() if gi[s] != gi[query]]
        cosines = np.asarray([float(y[query] @ y[s]) for s in members])
        attention = np.asarray([w[s] for s in members]) * np.exp(7.0 * (cosines - cosines.max()))
        assert abs(fold_wise[query] - attention.max() / attention.sum()) < 1e-12, query


def test_quintile_of_and_selected_rr():
    values = np.random.default_rng(11).random(N)
    bins, edges = tc.quintile_of(values)
    assert edges == [round(float(e), 5) for e in np.quantile(values, [0.2, 0.4, 0.6, 0.8])]
    assert set(bins.tolist()) <= set(range(tc.QUINTILES))
    counts = np.bincount(bins, minlength=tc.QUINTILES)
    assert counts.sum() == N and max(counts) - min(counts) <= 2, counts
    for i in range(N):
        assert bins[i] == int(np.searchsorted(np.quantile(values, [0.2, 0.4, 0.6, 0.8]), values[i],
                                              side="right"))
    by_beta = {b: np.arange(N, dtype=np.float64) + 100.0 * b for b in ae.BETAS}
    selected = (1.0, 2.0, 5.0, 10.0, 20.0)
    out = tc.selected_rr(by_beta, BASE["fold"], selected)
    for k in range(FOLDS):
        here = BASE["fold"] == k
        assert np.array_equal(out[here], by_beta[selected[k]][here])


# ------------------------------------------------------------------ estimands and the reading
def test_estimands_against_independent_definitions():
    rng = np.random.default_rng(4)
    rr = rng.random(N)
    w, li = BASE["weights"], BASE["label_index"]
    mask = rng.random(N) < 0.7
    assert abs(tc.query_weighted(rr, mask) - float(np.mean(rr[mask]))) < 1e-12
    assert abs(tc.component_weighted(rr, w, mask)
               - float(sum(rr[i] * w[i] for i in range(N) if mask[i])
                       / sum(w[i] for i in range(N) if mask[i]))) < 1e-12
    per_label = []
    for label in sorted({int(l) for l, m in zip(li.tolist(), mask.tolist()) if m}):
        rows = [i for i in range(N) if mask[i] and li[i] == label]
        per_label.append(sum(rr[i] * w[i] for i in rows) / sum(w[i] for i in rows))
    assert abs(tc.label_macro(rr, w, li, mask) - float(np.mean(per_label))) < 1e-12
    assert abs(tc.label_macro(rr, np.ones(N), li, np.ones(N, dtype=bool))
               - float(np.mean([np.mean(rr[li == l]) for l in range(BASE["label_count"])]))) < 1e-12


def test_difference_in_differences_pseudo_system():
    rng = np.random.default_rng(6)
    a_selected, a_zero, c_selected, c_zero = (rng.random(N) for _ in range(4))
    mask, unit = np.ones(N, dtype=bool), np.ones(N)
    groups = tc.union_grouping(BASE, [NAME_A, BASE["group_ids"]])
    contrast = paired_group_bootstrap({"lhs": a_selected - a_zero + c_zero, "rhs": c_selected},
                                      unit, groups, mask, [("lhs", "rhs")])[0]
    expect = (np.mean(a_selected) - np.mean(a_zero)) - (np.mean(c_selected) - np.mean(c_zero))
    assert abs(contrast["mrr_difference"] - round(float(expect), 4)) < 1e-4, (contrast, expect)
    plain = paired_group_bootstrap({"a": a_selected, "b": a_zero}, unit, groups, mask, [("a", "b")])[0]
    assert abs(plain["mrr_difference"] - round(float(np.mean(a_selected) - np.mean(a_zero)), 4)) < 1e-4
    weighted = paired_group_bootstrap({"a": a_selected, "b": a_zero}, BASE["weights"], groups, mask,
                                      [("a", "b")])[0]
    point = np.sum((a_selected - a_zero) * BASE["weights"]) / np.sum(BASE["weights"])
    assert abs(weighted["mrr_difference"] - round(float(point), 4)) < 1e-4
    half = np.zeros(N, dtype=bool)
    half[: N // 2] = True
    masked = paired_group_bootstrap({"a": a_selected, "b": a_zero}, unit, groups, half, [("a", "b")])[0]
    assert abs(masked["mrr_difference"]
               - round(float(np.mean(a_selected[half]) - np.mean(a_zero[half])), 4)) < 1e-4


def contrast(point, low, high):
    return {"system": "x", "minus": "y", "mrr_difference": point, "ci95": [low, high],
            "excludes_zero": bool(low > 0 or high < 0)}


def test_gain_reading():
    survives = tc.gain_reading(contrast(0.017, 0.013, 0.021))
    assert survives["positive_clear_of_zero"] and not survives["whole_interval_below_margin"]
    assert survives["verdict"].startswith("survives")
    small = tc.gain_reading(contrast(0.003, 0.001, 0.004))
    assert small["positive_clear_of_zero"] and small["whole_interval_below_margin"]
    assert "below the margin" in small["verdict"]
    bounded = tc.gain_reading(contrast(0.001, -0.002, 0.004))
    assert not bounded["positive_clear_of_zero"] and bounded["whole_interval_below_margin"]
    assert bounded["verdict"].startswith("bounded below the margin")
    assert tc.gain_reading(contrast(0.004, -0.030, 0.038))["verdict"].startswith("undecided")
    assert tc.gain_reading(contrast(-0.020, -0.030, -0.010))["verdict"].startswith("bounded below")


def test_clause_one_undecided_dominates_a_bound():
    lexical = ["words_raw", "chars_raw"]
    clear = tc.gain_reading(contrast(0.017, 0.013, 0.021))
    bound = tc.gain_reading(contrast(0.001, -0.002, 0.004))
    undecided = tc.gain_reading(contrast(0.004, -0.030, 0.038))
    both = tc.clause_one_verdict({"words_raw": clear, "chars_raw": clear}, lexical)
    assert both["verdict"].startswith("the attention gain survives")
    # the blocking case: one space undecided, the other bounded, must NOT publish the bound
    mixed = tc.clause_one_verdict({"words_raw": undecided, "chars_raw": bound}, lexical)
    assert mixed["verdict"].startswith("undecided on the complement set in words_raw"), mixed
    assert mixed["undecided_lexical_spaces"] == ["words_raw"]
    assert "bounded by the margin" not in mixed["verdict"]
    bounded_both = tc.clause_one_verdict({"words_raw": bound, "chars_raw": bound}, lexical)
    assert "bounded by the margin" in bounded_both["verdict"]
    # positive but below the margin is a bound, not a survival
    small = tc.gain_reading(contrast(0.003, 0.001, 0.004))
    partial = tc.clause_one_verdict({"words_raw": clear, "chars_raw": small}, lexical)
    assert "bounded by the margin in chars_raw" in partial["verdict"], partial


def test_concentration_reading():
    clear = tc.gain_reading(contrast(0.017, 0.013, 0.021))
    bound = tc.gain_reading(contrast(0.001, -0.002, 0.004))
    undecided = tc.gain_reading(contrast(0.004, -0.030, 0.038))
    assert tc.concentration_reading(clear, clear)["verdict"].startswith("contradicts")
    assert tc.concentration_reading(bound, clear)["verdict"].startswith("consistent with a residual")
    assert tc.concentration_reading(bound, bound)["verdict"].startswith("undecided")
    assert tc.concentration_reading(undecided, clear)["verdict"].startswith("undecided")
    assert tc.concentration_reading(None, clear)["verdict"].startswith("not read")


def test_residual_reading():
    words = ("inside", "below", "above")
    tight = [contrast(0.000, -0.002, 0.002) for _ in range(20)]
    assert tc.residual_reading(tight, *words)["verdict"] == "inside"
    assert tc.residual_reading(tight, *words)["draws_whole_interval_inside_margin"] == 20
    almost = tight[:18] + [contrast(0.000, -0.020, 0.020)] * 2
    assert tc.residual_reading(almost, *words)["verdict"] == "inside"
    loose = tight[:17] + [contrast(0.000, -0.020, 0.020)] * 3
    assert tc.residual_reading(loose, *words)["verdict"] == "undecided"
    # a null that only looks confirmed because nothing has power must read undecided
    powerless = [contrast(0.000, -0.050, 0.050) for _ in range(20)]
    assert tc.residual_reading(powerless, *words)["verdict"] == "undecided"
    assert tc.residual_reading(powerless, *words)["draws_whole_interval_inside_margin"] == 0
    twin = [contrast(-0.014, -0.020, -0.008) for _ in range(20)]
    assert tc.residual_reading(twin, *words)["verdict"] == "below"
    assert tc.residual_reading(twin, *words)["mean_point"] == -0.014
    assert tc.residual_reading([contrast(0.014, 0.008, 0.020)] * 20, *words)["verdict"] == "above"
    # a point past the margin with an interval that touches zero decides nothing
    assert tc.residual_reading([contrast(-0.014, -0.030, 0.002)] * 20, *words)["verdict"] == "undecided"
    # the gain decomposition gets its own vocabulary, never the prototype drop's sentence
    gain = tc.residual_reading(tight, "the attention gain is the same", "smaller", "larger")
    assert gain["verdict"] == "the attention gain is the same"


def test_stratum_withholds_a_small_cell():
    li, gi, w = BASE["label_index"], BASE["group_ids"], BASE["weights"]
    rng = np.random.default_rng(12)
    systems = {"prototype": rng.random(N), "attention_selected": rng.random(N)}
    small = np.zeros(N, dtype=bool)
    small[np.flatnonzero(li == int(li[0]))[:3]] = True
    entry = tc.stratum(systems, w, gi, li, small)
    assert entry["counts_for_the_rule"] is False and "gain_component_weighted" not in entry
    assert "withheld" in entry and entry["labels"] == 1 and entry["queries"] == 3
    empty = tc.stratum(systems, w, gi, li, np.zeros(N, dtype=bool))
    assert empty["queries"] == 0 and empty["labels"] == 0 and "gain_component_weighted" not in empty
    # a cell above both thresholds does get an interval
    assert tc.MIN_CELL_LABELS == 10 and tc.MIN_CELL_QUERIES == 100
    big = tc.stratum(systems, w, gi, li, np.ones(N, dtype=bool))
    assert big["counts_for_the_rule"] and "gain_component_weighted" in big and "reading" in big


def test_curve_shape():
    interior = {"0": 0.4266, "1": 0.4302, "2": 0.4333, "5": 0.4404, "10": 0.4436, "20": 0.4391,
                "50": 0.3997, "100": 0.3061}
    assert tc.curve_shape(interior) == "interior maximum at beta 10"
    falling = {"0": 0.50, "1": 0.49, "2": 0.48, "5": 0.47, "10": 0.46, "20": 0.45, "50": 0.44, "100": 0.43}
    assert tc.curve_shape(falling).startswith("maximum at beta 0")
    rising = {k: 0.40 + i * 0.01 for i, k in enumerate(["0", "1", "2", "5", "10", "20", "50", "100"])}
    assert tc.curve_shape(rising).startswith("maximum at the largest beta")


# ------------------------------------------------------------------ the cache, the betas and the bands
def test_cell_cache_round_trip():
    betas = (0.0, 1.0)
    rr = {0.0: np.random.default_rng(1).random(N), 1.0: np.random.default_rng(2).random(N)}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "published__words_raw.npz"
        fingerprint = tc.cell_fingerprint("published", "words_raw", betas, POP_PUB)
        assert tc.load_cell(path, fingerprint, betas) is None, "an absent cell must not load"
        tc.save_cell(path, fingerprint, betas, rr)
        back = tc.load_cell(path, fingerprint, betas)
        assert back is not None and all(np.array_equal(back[b], rr[b]) for b in betas)
        assert tc.load_cell(path, fingerprint, (0.0,)) is None, "a different beta set must not load"
        assert tc.load_cell(path, tc.cell_fingerprint("merged_groups", "words_raw", betas, POP_PUB),
                            betas) is None
        assert tc.load_cell(path, tc.cell_fingerprint("published", "words_raw", betas, POP_A),
                            betas) is None, "a changed population must invalidate the cell"
        assert not list(Path(directory).glob("*.partial.npz")), "a temporary cell was left behind"


def test_betas_for_cuts_the_grid_off_the_other_arms():
    grid = tuple(tc.GRID_SPACES)
    for space in grid:
        assert tc.betas_for(space, grid, True) == tuple(tc.BETAS)
        other = tc.betas_for(space, grid, False)
        assert other == tuple(sorted({0.0, *(float(b) for b in tc.SELECTED_BETA[space])})), space
        assert 0.0 in other and len(other) <= 3 and set(other) <= set(tc.BETAS)
    assert tc.betas_for("semantic_raw", grid, True) == (0.0,)
    assert tc.betas_for("words_raw", ("chars_raw",), True) == (0.0,)


def test_bands_are_songs_per_label():
    songs_per_label = np.bincount(BASE["label_index"], minlength=BASE["label_count"])
    total = 0
    for name, low, high in tc.BANDS:
        labels_in = np.flatnonzero((songs_per_label >= low) & (songs_per_label <= high))
        queries = int(np.isin(BASE["label_index"], labels_in).sum())
        total += queries
        assert queries == sum(int(songs_per_label[l]) for l in labels_in.tolist()), name
    assert total == N, "the songs-per-label bands must cover every query exactly once"
    components_per_label = np.zeros(BASE["label_count"], dtype=np.int64)
    for _, label in set(zip(BASE["group_ids"].tolist(), BASE["label_index"].tolist())):
        components_per_label[label] += 1
    assert not np.array_equal(components_per_label, songs_per_label), \
        "the synthetic corpus must have a group with two songs of one label, or the warning is untestable"
    assert set(tc.PUBLISHED_BANDS) == {name for name, _, _ in tc.BANDS}


def test_constants_match_the_published_record():
    assert tc.EXPECTED == {"semantic_raw": 0.2997, "chars_raw": 0.4266, "words_raw": 0.4963,
                           "semantic_whitened": 0.4164}
    assert tc.TOLERANCE["words_raw"] == 5e-4 and tc.TOLERANCE["semantic_whitened"] == 0.002
    assert tc.GAIN_TOLERANCE == {s: 2 * tc.TOLERANCE[s] for s in tc.TOLERANCE}, \
        "a contrast is a difference of two levels, so its gate is twice the level tolerance"
    assert tuple(tc.BETAS) == (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    for space in tc.GRID_SPACES:
        assert set(tc.PUBLISHED_CURVE[space]) == set(tc.BETAS)
        assert tc.PUBLISHED_CURVE[space][0.0] == tc.EXPECTED[space], space
        assert len(tc.SELECTED_BETA[space]) == FOLDS
        assert all(b in tc.BETAS for b in tc.SELECTED_BETA[space]), space
        assert set(tc.PUBLISHED_TOP_WEIGHT[space]) == {"q50", "q95"}
        assert 0.0 < tc.PUBLISHED_GAIN[space] < 0.05, space
    assert tc.MARGIN == 0.005 and tc.DRAWS >= 20 and tc.QUORUM <= tc.DRAWS
    assert sum(q for _, q in tc.PUBLISHED_BANDS.values()) == tc.PUBLISHED_POPULATION["queries"]
    assert tc.PUBLISHED_MERGE["groups_before"] == tc.PUBLISHED_POPULATION["groups"]
    assert tc.PUBLISHED_BANDS["50-up"][0] < tc.MIN_CELL_LABELS, \
        "the three-label band must fall below the reporting threshold"
    assert tc.UNIT_ROW_ATOL_FLOAT32 >= 4e-5 > tc.UNIT_ROW_ATOL_FLOAT64, \
        "a float32-normalised space cannot be held to a float64 unit-row tolerance"


def main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
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
