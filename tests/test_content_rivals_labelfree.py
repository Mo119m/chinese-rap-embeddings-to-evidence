#!/usr/bin/env python3
"""Synthetic test of content_rivals_labelfree_v3: random data, no private corpus, no jieba.

Checks, against brute-force definitions written independently here or against the
protocol's own scorers:
  * rank_within (the protocol's) and among_rivals agree with a brute-force rank within the
    candidate set, ties counted against the query, on scores with deliberate ties;
  * rivals_from_scores gives the K highest-scoring labels without the true one, for both
    argsort calls, and refuses more rivals than there are other labels; check_rival_sets
    passes them and rejects a set holding the true label or a repeat;
  * chance(K) and the mean-Jaccard overlap equal their plain definitions;
  * distort keeps exactly the top-k words by training-fold document frequency, is blind to
    the test fold's words, replaces everything else by the placeholder, and reports the
    kept share; distortion_arm refuses a corpus containing the placeholder;
  * topic_scores_fold equals a brute-force leave-group-out cosine on its own topic matrix,
    refuses to fit on its queries, refuses a song with a zero topic vector, and its
    training rows' topic vectors are bitwise blind to the query fold's documents;
  * levels_and_contrasts: every level is the component-weighted mean and every contrast
    the difference of two such means; the interval contains the point;
  * rival_tables: MRR and top-1 per space equal brute force; an excluded space is neither
    judged nor in any contrast;
  * the reading rule on hand-made cells;
  * reproduce_published passes on a file built by brute force from the published-design
    scores of a synthetic corpus and raises SystemExit when one number is moved by 0.003;
  * an end-to-end analyse() on the synthetic corpus: keys, finite numbers in [0, 1],
    all-labels and among-rivals numbers equal to brute force, no per-song array.

    set CHINESE_RAP_CORPUS=v3
    python test_content_rivals_labelfree.py     (or: python -m pytest test_content_rivals_labelfree.py)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("CHINESE_RAP_CORPUS", "v3")
HERE = Path(__file__).resolve().parent
_candidates = (HERE / "content_rivals_labelfree_v3.py", HERE.parent / "src" / "content_rivals_labelfree_v3.py")
_module_path = next(p for p in _candidates if p.is_file())
_spec = importlib.util.spec_from_file_location("content_rivals_labelfree_v3", _module_path)
crl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crl)

from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402

SILENT = lambda *_: None  # noqa: E731


# ------------------------------------------------------------------ synthetic corpus
def make_synthetic(seed: int = 7, label_count: int = 14, dim: int = 24, vocabulary: int = 200):
    """Random songs around label centres; leakage groups of 1-3 songs (some across labels)
    sharing their text as duplicate groups do; the protocol's per-(group, label) weights and
    fold assignment; documents of ASCII pseudo-words so str.split is the tokenizer."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(7, 24, size=label_count)
    label_index = np.repeat(np.arange(label_count), sizes)
    n = len(label_index)
    label_index = label_index[rng.permutation(n)]
    centres = rng.normal(size=(label_count, dim))
    dense = unit_rows(centres[label_index] + 1.5 * rng.normal(size=(n, dim))).astype(np.float64)
    group_ids = np.arange(n)
    for _ in range(n // 8):
        a, b = rng.integers(0, n, size=2)
        group_ids[group_ids == group_ids[b]] = group_ids[a]
    order = {g: i for i, g in enumerate(sorted(set(group_ids.tolist())))}
    group_ids = np.asarray([order[g] for g in group_ids.tolist()], dtype=np.int64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    background = rng.dirichlet(np.full(vocabulary, 0.3))
    preference = rng.dirichlet(np.full(vocabulary, 0.15), size=label_count)
    words = [f"w{i}" for i in range(vocabulary)]
    songs = [f"song{i:04d}" for i in range(n)]
    documents = {}
    for i, song in enumerate(songs):
        p = 0.5 * background + 0.5 * preference[label_index[i]]
        length = int(rng.integers(40, 90))
        documents[song] = " ".join(words[j] for j in rng.choice(vocabulary, size=length, p=p))
    for g in range(len(order)):
        members = np.flatnonzero(group_ids == g)
        for m in members[1:]:
            documents[songs[m]] = documents[songs[members[0]]]
    return dict(songs=songs, label_index=label_index, group_ids=group_ids, label_count=label_count,
                weights=weights, fold=fold, dense=dense, documents=documents)


# ------------------------------------------------------------------ brute-force definitions
def brute_force_lgo(x, label_index, group_ids, weights, label_count, queries):
    """Cosine of each query with the weighted sum of every label's songs outside the query's
    leakage group; written from the definition."""
    out = np.zeros((len(queries), label_count))
    for row, q in enumerate(queries.tolist()):
        for l in range(label_count):
            members = [s for s in range(len(label_index)) if label_index[s] == l and group_ids[s] != group_ids[q]]
            profile = sum(weights[s] * x[s] for s in members)
            norm = float(np.linalg.norm(profile))
            out[row, l] = float(x[q] @ profile) / norm if norm > 0 else 0.0
    return out


def brute_force_full_rank(scores, truth):
    """The protocol's rank over all labels (v1.rank_system): labels above by more than
    1e-12, plus ties within 1e-12 with a smaller label index; scores seen in float32."""
    s = scores.astype(np.float32)
    ranks = []
    for row in range(s.shape[0]):
        t = s[row, truth[row]]
        above = sum(1 for l in range(s.shape[1]) if s[row, l] > t + 1e-12)
        ties = sum(1 for l in range(truth[row]) if abs(s[row, l] - t) <= 1e-12)
        ranks.append(1 + above + ties)
    return np.asarray(ranks)


def brute_force_rivals(scores, truth, k):
    """The k best-scoring other labels, by sorting a Python list."""
    out = []
    for q in range(scores.shape[0]):
        others = [l for l in range(scores.shape[1]) if l != truth[q]]
        others.sort(key=lambda l: -scores[q, l])
        out.append(others[:k])
    return np.asarray(out)


def brute_force_rank_in_set(scores, truth, rivals, k):
    ranks = []
    for q in range(scores.shape[0]):
        own = scores[q, truth[q]]
        ranks.append(1 + sum(1 for c in rivals[q][:k].tolist() if scores[q, c] >= own))
    return np.asarray(ranks)


def expect_exit(call, message):
    try:
        call()
    except SystemExit:
        return
    raise AssertionError(message)


# ------------------------------------------------------------------ tests
def test_rank_within_and_among_rivals_match_brute_force():
    rng = np.random.default_rng(3)
    n, L = 80, 12
    scores = {"a": rng.normal(size=(n, L)), "b": rng.integers(0, 3, size=(n, L)).astype(np.float64)}
    truth = rng.integers(0, L, size=n)
    scores["a"][:10, :] = 0.0                                               # full ties: rank K+1 within any set
    scores["a"][10:20, 3] = scores["a"][np.arange(10, 20), truth[10:20]]    # a tie with label 3
    rivals = crl.rivals_from_scores(scores["a"], truth, 6)
    for k in (1, 3, 6):
        ranks = crl.among_rivals(scores, truth, rivals, k, ("a", "b"), check=n)   # brute-force check on every query
        for name in ("a", "b"):
            assert np.array_equal(ranks[name], brute_force_rank_in_set(scores[name], truth, rivals, k))
        assert np.all(ranks["a"][:10] == k + 1)
    tied = [q for q in range(10, 20) if truth[q] != 3 and 3 in rivals[q][:6].tolist()]
    assert tied, "the tie rows should meet label 3 among their rivals"
    for q in tied:
        assert ranks["a"][q] >= 2
    # the internal check itself catches a wrong rank
    original = crl.rank_within
    crl.rank_within = lambda s, t, c: original(s, t, c) + 1
    try:
        expect_exit(lambda: crl.among_rivals(dict(scores), truth, rivals, 3, ("a",), check=5), "a wrong rank passed the brute-force check")
    finally:
        crl.rank_within = original


def test_rivals_from_scores_and_check_rival_sets():
    rng = np.random.default_rng(4)
    n, L, k = 50, 9, 4
    scores = rng.normal(size=(n, L))
    truth = rng.integers(0, L, size=n)
    expect = brute_force_rivals(scores, truth, k)
    for stable in (True, False):
        got = crl.rivals_from_scores(scores, truth, k, stable=stable)
        assert np.array_equal(got, expect)
        crl.check_rival_sets(got, truth, k)
        crl.check_rival_sets(got, truth, 2)
    full = crl.rivals_from_scores(scores, truth, L - 1)
    assert full.shape == (n, L - 1) and np.all(full != truth[:, None])
    expect_exit(lambda: crl.rivals_from_scores(scores, truth, L), "more rivals than other labels passed")
    bad = expect.copy()
    bad[7, 1] = truth[7]
    expect_exit(lambda: crl.check_rival_sets(bad, truth, k), "a rival set with the true label passed")
    bad = expect.copy()
    bad[5, 2] = bad[5, 0]
    expect_exit(lambda: crl.check_rival_sets(bad, truth, k), "a rival set with a repeat passed")
    expect_exit(lambda: crl.check_rival_sets(expect, truth, k + 1), "too few rivals passed")
    assert np.array_equal(crl.candidate_sets(truth, expect, 2)[:, 0], truth)
    assert crl.candidate_sets(truth, expect, 2).shape == (n, 3)


def test_chance_and_overlap():
    for k in (1, 5, 10, 25):
        c = crl.chance(k)
        assert abs(c["mrr"] - round(float(np.mean([1.0 / r for r in range(1, k + 2)])), 4)) < 1e-9
        assert abs(c["top1"] - round(1.0 / (k + 1), 4)) < 1e-9
    a = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7], [1, 2, 3, 4]])
    b = np.asarray([[0, 1, 5, 6], [4, 5, 6, 7], [9, 8, 7, 6]])
    # K=4: 2/6, 4/4, 0/8 ; K=2: 2/2, 2/2, 0/4
    assert abs(crl.overlap(a, b, 4) - float(np.mean([2 / 6, 1.0, 0.0]))) < 1e-12
    assert abs(crl.overlap(a, b, 2) - float(np.mean([1.0, 1.0, 0.0]))) < 1e-12


def test_distort_keeps_top_k_by_training_document_frequency_and_is_blind_to_the_test_fold():
    tokens = [["A", "b", "c", "a"], ["a", "B", "d"], ["a", "c", "e", "e"], ["b", "d", "zz", "zz"], ["zz", "zz", "zz"]]
    train = np.asarray([True, True, True, False, False])
    docs, keep, share = crl.distort(tokens, train, 2, placeholder="<oov>")
    # training document frequency (lowercased): a 3, b 2, c 2, d 1, e 1 -> top 2 by (-df, string): a, b
    assert keep == {"a", "b"}
    assert docs[0] == "a b <oov> a" and docs[1] == "a b <oov>" and docs[3] == "b <oov> <oov> <oov>" and docs[4] == "<oov> <oov> <oov>"
    kept = 3 + 2 + 1 + 1 + 0
    assert abs(share - kept / 18) < 1e-12
    # the test fold's words never enter the kept list, however frequent
    tokens_b = [t[:] for t in tokens]
    tokens_b[3] = ["zz", "qq", "zz"]
    tokens_b[4] = ["qq", "zz"]
    _, keep_b, _ = crl.distort(tokens_b, train, 2, placeholder="<oov>")
    assert keep_b == keep
    # brute force on random tokens
    rng = np.random.default_rng(5)
    words = [f"w{i}" for i in range(40)]
    random_tokens = [[words[j] for j in rng.integers(0, 40, size=int(rng.integers(5, 30)))] for _ in range(60)]
    mask = rng.random(60) < 0.7
    docs, keep, share = crl.distort(random_tokens, mask, 12)
    df = Counter()
    for i in np.flatnonzero(mask).tolist():
        df.update(set(random_tokens[i]))
    ranked = sorted(df, key=lambda t: (-df[t], t))
    assert keep == set(ranked[:12]) and len(keep) == 12
    kept = sum(1 for t in random_tokens for x in t if x in keep)
    total = sum(len(t) for t in random_tokens)
    assert abs(share - kept / total) < 1e-12
    for doc, toks in zip(docs, random_tokens):
        assert doc.split(" ") == [t if t in keep else crl.PLACEHOLDER for t in toks]
    # distortion_arm refuses a corpus containing the placeholder
    d = make_synthetic(seed=21, label_count=8)
    toks = [d["documents"][s].split() for s in d["songs"]]
    toks[3] = toks[3] + [crl.PLACEHOLDER.upper()]
    expect_exit(lambda: crl.distortion_arm(toks, d["fold"], d["dense"].astype(np.float32), d["label_index"], d["group_ids"],
                                           d["label_count"], 20, log=SILENT), "a corpus containing the placeholder passed")


def test_topic_scores_fold_matches_brute_force_and_is_blind_to_the_query_fold():
    d = make_synthetic(seed=11, label_count=30)
    li, gi, w, L, fold = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"]
    joined = [d["documents"][s] for s in d["songs"]]
    train = fold != 0
    queries = np.flatnonzero(~train)
    scores, topic_matrix, info = crl.topic_scores_fold(joined, train, queries, li, gi, w, L, SEED, topics=6, max_iter=200)
    assert topic_matrix.shape == (len(li), 6) and np.all(topic_matrix >= 0)
    expect = brute_force_lgo(unit_rows(topic_matrix), li, gi, w, L, queries)
    assert float(np.abs(scores - expect).max()) < 1e-9
    assert info["training_songs"] == int(train.sum()) and info["queries"] == len(queries)
    assert info["vocabulary"] <= 200 and info["zero_topic_rows"] == 0
    # blind to the query fold: replace its documents by other random pseudo-words and refit
    rng = np.random.default_rng(8)
    other = list(joined)
    for q in queries.tolist():
        other[q] = " ".join(f"w{j}" for j in rng.integers(0, 200, size=50))
    _, topic_b, _ = crl.topic_scores_fold(other, train, queries, li, gi, w, L, SEED, topics=6, max_iter=200)
    assert np.array_equal(topic_matrix[train], topic_b[train]), "the topic model saw the query fold"
    # refuses to fit on its own queries
    expect_exit(lambda: crl.topic_scores_fold(joined, train, np.flatnonzero(train)[:3], li, gi, w, L, SEED, topics=6),
                "a topic model fitted on its queries passed")
    # refuses a song whose words are all outside the training vocabulary (a zero topic vector)
    orphan = list(joined)
    orphan[int(queries[0])] = "qqq qqq"
    expect_exit(lambda: crl.topic_scores_fold(orphan, train, queries, li, gi, w, L, SEED, topics=6, max_iter=200),
                "a zero topic row passed")
    # topic_rivals: the same per fold, and rival sets that pass the checks
    rivals, by_fold = crl.topic_rivals(joined, fold, li, gi, w, L, topics=6, max_iter=200, log=SILENT)
    assert rivals.shape == (len(li), max(crl.RIVALS)) and set(by_fold) == {str(f) for f in range(FOLDS)}
    for k in crl.RIVALS:
        crl.check_rival_sets(rivals, li, k)
    # fold 0's rows of the rival table come from the fold-0 model checked above
    assert np.array_equal(rivals[queries], brute_force_rivals(scores, li[queries], max(crl.RIVALS)))


def test_levels_and_contrasts_are_weighted_means_and_differences():
    d = make_synthetic(seed=5)
    li, gi, w = d["label_index"], d["group_ids"], d["weights"]
    rng = np.random.default_rng(9)
    n = len(li)
    values = {"a": 1.0 / rng.integers(1, 20, size=n), "b": 1.0 / rng.integers(1, 20, size=n),
              "c": (rng.random(n) < 0.4).astype(np.float64)}
    levels, contrasts = crl.levels_and_contrasts(values, w, gi, [("a", "b"), ("c", "a")])
    mask = np.ones(n, dtype=bool)
    for name, v in values.items():
        assert abs(levels[name]["component_weighted"] - weighted_mean(v, w, mask)) < 1e-4
        assert levels[name]["ci95"][0] <= levels[name]["component_weighted"] <= levels[name]["ci95"][1]
    assert [(c["system"], c["minus"]) for c in contrasts] == [("a", "b"), ("c", "a")]
    for c in contrasts:
        point = weighted_mean(values[c["system"]], w, mask) - weighted_mean(values[c["minus"]], w, mask)
        assert abs(c["mrr_difference"] - point) < 1e-4
        assert c["ci95"][0] <= c["mrr_difference"] <= c["ci95"][1]
    # the same replicates as the protocol's own call
    direct = paired_group_bootstrap(values, w, gi, mask, [("a", "b")])[0]
    assert direct == contrasts[0]


def test_rival_tables_match_brute_force_and_exclusion():
    d = make_synthetic(seed=6)
    li, gi, w, L = d["label_index"], d["group_ids"], d["weights"], d["label_count"]
    rng = np.random.default_rng(10)
    n = len(li)
    scores = {name: rng.normal(size=(n, L)) for name in ("a", "b", "c")}
    scores["b"][:15, :] = 1.0                       # ties against the query
    rivals = crl.rivals_from_scores(scores["c"], li, 8)
    pairs = (("a", "b"), ("a", "c"), ("b", "c"))
    out = crl.rival_tables(scores, li, gi, w, rivals, excluded=("c",), defines=("c",), spaces=("a", "b", "c"),
                           pairs=pairs, ks=(2, 5, 8), check=n)
    assert out["defines_the_rivals"] == ["c"] and out["excluded_from_contrasts"] == ["c"]
    m = np.ones(n, dtype=bool)
    for k in (2, 5, 8):
        entry = out["by_k"][str(k)]
        assert entry["chance"] == crl.chance(k)
        assert [(c["system"], c["minus"]) for c in entry["mrr_contrasts"]] == [("a", "b")]
        assert [(c["system"], c["minus"]) for c in entry["top1_contrasts"]] == [("a", "b")]
        assert "top1_difference" in entry["top1_contrasts"][0] and "mrr_difference" not in entry["top1_contrasts"][0]
        for name in ("a", "b", "c"):
            r = brute_force_rank_in_set(scores[name], li, rivals, k)
            cell = entry["spaces"][name]
            assert abs(cell["mrr"] - round(float(np.mean(1.0 / r)), 4)) < 1e-9
            assert abs(cell["top1"] - round(float(np.mean(r == 1)), 4)) < 1e-9
            assert abs(cell["mrr_component_weighted"] - weighted_mean(1.0 / r, w, m)) < 1e-4
            assert abs(cell["top1_component_weighted"] - weighted_mean((r == 1).astype(np.float64), w, m)) < 1e-4
            assert cell["judged"] == (name != "c") and cell["defines_the_rivals"] == (name == "c")
        top1_a = (brute_force_rank_in_set(scores["a"], li, rivals, k) == 1).astype(np.float64)
        top1_b = (brute_force_rank_in_set(scores["b"], li, rivals, k) == 1).astype(np.float64)
        assert abs(entry["top1_contrasts"][0]["top1_difference"] - (weighted_mean(top1_a, w, m) - weighted_mean(top1_b, w, m))) < 1e-4
        rr_a, rr_b = 1.0 / brute_force_rank_in_set(scores["a"], li, rivals, k), 1.0 / brute_force_rank_in_set(scores["b"], li, rivals, k)
        assert abs(entry["mrr_contrasts"][0]["mrr_difference"] - (weighted_mean(rr_a, w, m) - weighted_mean(rr_b, w, m))) < 1e-4
    assert out["by_k"]["8"]["spaces"]["b"]["top1"] <= out["by_k"]["2"]["spaces"]["b"]["top1"]   # more rivals, never more top-1


def _cell(diff, low, high):
    return {"system": crl.WORDS, "minus": crl.WHITENED, "mrr_difference": diff, "ci95": [low, high],
            "excludes_zero": bool(low > 0 or high < 0)}


def _definition(cells_by_k):
    return {"by_k": {str(k): {"mrr_contrasts": [{"system": "x", "minus": "y", "mrr_difference": 0.0, "ci95": [-1, 1],
                                                 "excludes_zero": False}, _cell(*cells_by_k[k])]} for k in crl.RIVALS}}


def test_reading_rule():
    clear = {5: (0.09, 0.08, 0.10), 10: (0.08, 0.07, 0.09), 25: (0.07, 0.06, 0.08)}
    touching = dict(clear)
    touching[10] = (0.01, 0.0, 0.02)
    negative = dict(clear)
    negative[25] = (-0.02, -0.03, -0.01)
    r = crl.reading({"nmf_topics": _definition(clear), "character_space": _definition(clear)})
    assert r["verdict"] == "not an artefact of the rival definition" and r["failing_cells"] == []
    assert r["contrast"] == f"{crl.WORDS} - {crl.WHITENED}" and len(r["cells"]) == 6
    r = crl.reading({"nmf_topics": _definition(clear), "character_space": _definition(touching)})
    assert r["verdict"] == "definition-dependent" and r["failing_cells"] == ["character_space@K=10"]
    r = crl.reading({"nmf_topics": _definition(negative), "character_space": _definition(touching)})
    assert r["verdict"] == "definition-dependent" and r["failing_cells"] == ["nmf_topics@K=25", "character_space@K=10"]


def synthetic_published(scores, d):
    """A content_rival_test.json for the synthetic corpus, its levels by brute force and its
    contrasts by the protocol's own bootstrap."""
    li, gi, w = d["label_index"], d["group_ids"], d["weights"]
    results = {"all_labels": {name: round(float(np.mean(1.0 / brute_force_full_rank(scores[name], li))), 4)
                              for name in crl.PUBLISHED_SPACES}}
    contrasts = {}
    rivals = brute_force_rivals(scores[crl.SEMANTIC], li, max(crl.RIVALS))
    for k in crl.RIVALS:
        ranks = {name: brute_force_rank_in_set(scores[name], li, rivals, k) for name in crl.PUBLISHED_SPACES}
        results[f"among_{k}_content_rivals"] = {"chance_mrr": crl.chance(k)["mrr"],
                                                **{name: {"mrr": round(float(np.mean(1.0 / r)), 4), "top1": round(float(np.mean(r == 1)), 4)}
                                                   for name, r in ranks.items()}}
        contrasts[f"among_{k}_content_rivals"] = paired_group_bootstrap({n: 1.0 / r for n, r in ranks.items()}, w, gi,
                                                                        np.ones(len(li), dtype=bool), list(crl.PUBLISHED_PAIRS))
    return {"corpus": {"content_sha256": V3_CONTENT_SHA256}, "results": results,
            "paired_contrasts": {"by_rival_count": contrasts}}


def test_reproduction_check_passes_and_fails():
    import copy
    d = make_synthetic(seed=13, label_count=30)
    li, gi, w = d["label_index"], d["group_ids"], d["weights"]
    docs = [d["documents"][s] for s in d["songs"]]
    scores = crl.published_design_scores(d, docs, docs, log=SILENT)
    assert set(scores) == set(crl.PUBLISHED_SPACES)
    published = synthetic_published(scores, d)
    record, rivals = crl.reproduce_published(scores, li, gi, w, published, pinned=None, check=len(li), log=SILENT)
    assert record["worst_gap"] <= 1e-4
    assert np.array_equal(rivals, brute_force_rivals(scores[crl.SEMANTIC], li, max(crl.RIVALS)))
    assert set(record["among_rivals"]) == {"5", "10", "25"} and len(record["paired_contrasts"]["5"]) == len(crl.PUBLISHED_PAIRS)
    # a moved level, a moved top-1, a moved interval bound: each is refused
    for path in (("results", "all_labels", crl.WORDS), ("results", "among_10_content_rivals", crl.CHARS, "top1")):
        broken = copy.deepcopy(published)
        node = broken
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = node[path[-1]] + 0.003
        expect_exit(lambda: crl.reproduce_published(scores, li, gi, w, broken, pinned=None, check=0, log=SILENT),
                    f"a moved number at {path} passed")
    broken = copy.deepcopy(published)
    broken["paired_contrasts"]["by_rival_count"]["among_25_content_rivals"][3]["ci95"][1] += 0.003
    expect_exit(lambda: crl.reproduce_published(scores, li, gi, w, broken, pinned=None, check=0, log=SILENT),
                "a moved interval bound passed")
    broken = copy.deepcopy(published)
    broken["corpus"]["content_sha256"] = "0" * 64
    expect_exit(lambda: crl.reproduce_published(scores, li, gi, w, broken, pinned=None, check=0, log=SILENT),
                "a foreign corpus digest passed")
    # a pinned literal that disagrees with the file is refused too
    pinned = {"all_labels": {n: published["results"]["all_labels"][n] for n in crl.PUBLISHED_SPACES},
              "among": {k: {n: (published["results"][f"among_{k}_content_rivals"][n]["mrr"],
                                published["results"][f"among_{k}_content_rivals"][n]["top1"]) for n in crl.PUBLISHED_SPACES}
                        for k in crl.RIVALS},
              "headline": {k: (c["mrr_difference"], c["ci95"][0], c["ci95"][1])
                           for k in crl.RIVALS for c in published["paired_contrasts"]["by_rival_count"][f"among_{k}_content_rivals"]
                           if (c["system"], c["minus"]) == crl.HEADLINE}}
    record, _ = crl.reproduce_published(scores, li, gi, w, published, pinned=pinned, check=0, log=SILENT)
    assert record["worst_gap"] <= 1e-4
    pinned["headline"][5] = (pinned["headline"][5][0] + 0.01, pinned["headline"][5][1], pinned["headline"][5][2])
    expect_exit(lambda: crl.reproduce_published(scores, li, gi, w, published, pinned=pinned, check=0, log=SILENT),
                "a wrong pinned literal passed")


def test_end_to_end_on_synthetic_corpus():
    import json
    d = make_synthetic(seed=17, label_count=30)
    li, gi, w, L = d["label_index"], d["group_ids"], d["weights"], d["label_count"]
    docs = [d["documents"][s] for s in d["songs"]]
    published = synthetic_published(crl.published_design_scores(d, docs, docs, log=SILENT), d)
    out = crl.analyse(d, docs, published, tokenize=str.split, topics=6, nmf_max_iter=200, distortion_words=40,
                      check=40, pinned=None, log=SILENT)
    assert set(out["among_rivals"]) == set(crl.DEFINITIONS)
    assert set(out["all_labels"]["spaces"]) == set(crl.SPACES)
    assert out["checks"]["reproduction_of_content_rival_test"]["worst_gap"] <= 1e-4
    assert out["reading"]["verdict"] in ("not an artefact of the rival definition", "definition-dependent")
    assert set(out["reading"]["cells"]) == {f"{dfn}@K={k}" for dfn in crl.LABEL_FREE for k in crl.RIVALS}
    for f in range(FOLDS):
        e = out["distortion_by_fold"][str(f)]
        assert 0 < e["words_kept"] <= 40 and 0.0 < e["kept_token_share"] <= 1.0
        t = out["topic_model_by_fold"][str(f)]
        assert t["training_songs"] + t["queries"] == len(li) and t["queries"] == int((d["fold"] == f).sum())
        assert t["zero_topic_rows"] == 0
    for definition in crl.DEFINITIONS:
        table = out["among_rivals"][definition]
        assert table["defines_the_rivals"] == list(crl.DEFINES[definition])
        for k in crl.RIVALS:
            entry = table["by_k"][str(k)]
            for name, cell in entry["spaces"].items():
                for key in ("mrr", "top1", "mrr_component_weighted", "top1_component_weighted"):
                    assert np.isfinite(cell[key]) and 0.0 <= cell[key] <= 1.0
                assert cell["mrr_ci95"][0] <= cell["mrr_component_weighted"] <= cell["mrr_ci95"][1]
            judged = {n for n, c in entry["spaces"].items() if c["judged"]}
            for c in entry["mrr_contrasts"] + entry["top1_contrasts"]:
                assert c["system"] in judged and c["minus"] in judged
            if definition == "character_space":
                assert crl.CHARS not in judged and not entry["spaces"][crl.CHARS]["judged"]
                assert all(crl.CHARS not in (c["system"], c["minus"]) for c in entry["mrr_contrasts"])
            else:
                assert [(c["system"], c["minus"]) for c in entry["mrr_contrasts"]] == list(crl.PAIRS)
    # all-labels and among-rivals numbers against brute force from the published-design scores
    scores = crl.published_design_scores(d, docs, docs, log=SILENT)
    for name in crl.PUBLISHED_SPACES:
        mrr = float(np.mean(1.0 / brute_force_full_rank(scores[name], li)))
        assert abs(out["all_labels"]["spaces"][name]["mrr"] - mrr) < 1e-4
    char_rivals = brute_force_rivals(scores[crl.CHARS], li, 25)
    semantic_rivals = brute_force_rivals(scores[crl.SEMANTIC], li, 25)
    for k in crl.RIVALS:
        for name in crl.PUBLISHED_SPACES:
            r = brute_force_rank_in_set(scores[name], li, char_rivals, k)
            cell = out["among_rivals"]["character_space"]["by_k"][str(k)]["spaces"][name]
            assert abs(cell["mrr"] - float(np.mean(1.0 / r))) < 1e-4 and abs(cell["top1"] - float(np.mean(r == 1))) < 1e-4
            r = brute_force_rank_in_set(scores[name], li, semantic_rivals, k)
            cell = out["among_rivals"]["semantic_space"]["by_k"][str(k)]["spaces"][name]
            assert abs(cell["mrr"] - float(np.mean(1.0 / r))) < 1e-4
        head = next(c for c in out["among_rivals"]["nmf_topics"]["by_k"][str(k)]["mrr_contrasts"]
                    if (c["system"], c["minus"]) == crl.HEADLINE)
        assert out["reading"]["cells"][f"nmf_topics@K={k}"] == head
    # the semantic space's top-1 among its own rivals is its full top-1, at every K
    full_top1 = out["all_labels"]["spaces"][crl.SEMANTIC]["top1"]
    for k in crl.RIVALS:
        assert abs(out["among_rivals"]["semantic_space"]["by_k"][str(k)]["spaces"][crl.SEMANTIC]["top1"] - full_top1) < 1e-4
    # aggregate only: no array anywhere in the payload, and it serialises
    def walk(node):
        if isinstance(node, dict):
            return all(walk(v) for v in node.values())
        if isinstance(node, (list, tuple)):
            return all(walk(v) for v in node)
        return not isinstance(node, np.ndarray)
    assert walk(out)
    json.dumps(out, ensure_ascii=False, sort_keys=True)


TESTS = [test_rank_within_and_among_rivals_match_brute_force, test_rivals_from_scores_and_check_rival_sets,
         test_chance_and_overlap, test_distort_keeps_top_k_by_training_document_frequency_and_is_blind_to_the_test_fold,
         test_topic_scores_fold_matches_brute_force_and_is_blind_to_the_query_fold,
         test_levels_and_contrasts_are_weighted_means_and_differences, test_rival_tables_match_brute_force_and_exclusion,
         test_reading_rule, test_reproduction_check_passes_and_fails, test_end_to_end_on_synthetic_corpus]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(TESTS) - failed} of {len(TESTS)} tests passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
