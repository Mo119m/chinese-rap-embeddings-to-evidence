#!/usr/bin/env python3
"""Synthetic unit test for cosine_delta_and_tfidf_grid_v3: random data, no corpus, no private files.

Checks, each against an independent definition:
  1. fast_lgo (dense rows of any length, CSR unit rows; all queries and one fold's) equals a
     literal leave-group-out loop, the protocol's dense_leave_group_out and sparse
     v1.score_leave_group_out, and lgo_parts; it refuses queries that split a leakage group
     and a label that its own query's group empties;
  2. burrows_lgo equals a literal loop of the definition at every size, and the script's own
     burrows_brute (used in the run's check) equals that loop too;
  3. delta_features equals a literal loop (MFW by document frequency on the training rows,
     ties by total count then column order; relative frequencies; weighted mean and sd of
     the training rows), and is blind to the test rows;
  4. the count-matrix TF-IDF (build_counts, vocabulary, weigh) equals sklearn's
     TfidfVectorizer for every cell of both grids, with binding and non-binding caps,
     fitted on all documents and fitted on a subset and applied to all;
  5. inner_problem keeps exactly the labels with two or more components in the training
     rows, in their order; select_config's tie rule; reciprocal_ranks' worst rank for an
     empty row;
  6. question_b end to end on planted data: the published cell equals TfidfVectorizer +
     the protocol's scorer; one fold's inner selection equals an independent selection
     (TfidfVectorizer fitted on the training documents, dense_leave_group_out among the
     inner songs); the inductive system equals TfidfVectorizer.fit(train).transform(all) +
     dense_leave_group_out; fold k's selection is blind to fold k's rows; pooled systems
     take each fold's rows from the selected cell; every inner cell's MRR is reported and
     equals the independent one; every all-songs cell carries its contrast against the
     published cell on the bootstrap's weighted estimand; the grid extremes and the best
     cells within a tf setting are what their names say;
  7. question_a end to end: cosine_delta, cosine_delta_unit_rows and burrows_delta of one
     fold equal literal definitions built from the literal features;
  8. reading_a and reading_b on constructed inputs give every verdict of the reading rule;
  9. a payload built from the outputs is JSON-serialisable and holds no token of the data.

    set CHINESE_RAP_CORPUS=v3
    python test_cosine_delta_and_tfidf_grid.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ.setdefault("CHINESE_RAP_CORPUS", "v3")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import cosine_delta_and_tfidf_grid_v3 as cd  # noqa: E402
import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from identity_probe_v2 import FOLDS, dense_leave_group_out  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from label_size_calibration_v3 import lgo_parts  # noqa: E402

N, L = 240, 8
CAPS = (60, 300, None)
CURRENT = {"chars": ("2-5", 60, True), "words": ("1-2", 60, True)}
SIZES = (10, 20, 40)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {message}")
    print(f"ok: {message}", flush=True)


def expect_error(function, kind, message: str) -> None:
    try:
        function()
    except kind:
        print(f"ok: {message}", flush=True)
        return
    raise SystemExit(f"FAILED (no error raised): {message}")


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def structure(seed: int):
    """Labels, leakage groups (singletons, pairs of one or two labels, one group of five),
    the protocol's component weights and folds that keep groups whole."""
    rng = np.random.default_rng(seed)
    label_index = rng.integers(0, L, size=N)
    label_index[:L] = np.arange(L)
    group = np.arange(N)
    for i in range(20, 60, 2):
        group[i + 1] = group[i]
    group[60:65] = group[60]
    _, group_ids = np.unique(group, return_inverse=True)
    size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold = rng.integers(0, FOLDS, size=group_ids.max() + 1)[group_ids]
    return label_index, group_ids.astype(np.int64), weights, fold


def literal_lgo(x, label_index, group_ids, weights, queries):
    """The definition: cosine-style prototype score, one (query, label) at a time."""
    scores = np.zeros((len(queries), L))
    for row, q in enumerate(queries):
        for label in range(L):
            profile = np.zeros(x.shape[1])
            for m in range(len(label_index)):
                if label_index[m] == label and group_ids[m] != group_ids[q]:
                    profile += weights[m] * x[m]
            scores[row, label] = float(x[q] @ profile) / float(np.sqrt(profile @ profile))
    return scores


def literal_burrows(z, size, label_index, group_ids, weights, q):
    scores = np.zeros(L)
    for label in range(L):
        total, mass = np.zeros(size), 0.0
        for m in range(len(label_index)):
            if label_index[m] == label and group_ids[m] != group_ids[q]:
                total += weights[m] * z[m, :size]
                mass += weights[m]
        scores[label] = -sum(abs(z[q, j] - total[j] / mass) for j in range(size)) / size
    return scores


def planted_documents(seed: int, label_index):
    """Token sequences with a label-specific distribution. Tokens: one or two Han characters,
    two case variants of a Latin token (lower-casing must merge them) and a punctuation mark."""
    rng = np.random.default_rng(seed)
    vocabulary = [chr(0x4E00 + i) for i in range(30)] + [chr(0x5000 + i) + chr(0x5100 + i % 7) for i in range(40)]
    vocabulary += ["Yo", "yo", "，", "OK"]
    base = 1.0 / np.arange(1, len(vocabulary) + 1)
    base /= base.sum()
    per_label = [rng.dirichlet(base * 1000) for _ in range(L)]        # a weak signal: the grid's cells must differ
    documents, segmented, tokens_seen = [], [], set(vocabulary)
    for label in label_index.tolist():
        tokens = [vocabulary[i] for i in rng.choice(len(vocabulary), size=int(rng.integers(40, 90)), p=per_label[label])]
        segmented.append(" ".join(tokens))
        pieces = []
        for token in tokens:
            pieces.append(token)
            draw = rng.random()
            if draw < 0.10:
                pieces.append("\n")
            elif draw < 0.15:
                pieces.append("  ")
        documents.append("".join(pieces))
    return documents, segmented, tokens_seen


def vectorizer_for(space, ngram, cap, sublinear):
    low, high = next((lo, hi) for name, lo, hi in cd.RANGES[space] if name == ngram)
    if space == "chars":
        return TfidfVectorizer(analyzer="char", ngram_range=(low, high), min_df=3, max_features=cap,
                               sublinear_tf=sublinear, norm="l2", dtype=np.float32)
    return TfidfVectorizer(analyzer="word", ngram_range=(low, high), token_pattern=r"(?u)\S+", min_df=3,
                           max_features=cap, sublinear_tf=sublinear, norm="l2", dtype=np.float32)


def main() -> int:
    label_index, group_ids, weights, fold = structure(11)
    rng = np.random.default_rng(5)
    everything = np.arange(N)
    fold_queries = np.flatnonzero(fold == 2)

    # ---------------------------------------------------------------- 1. fast_lgo
    dense = rng.standard_normal((N, 12)) * rng.uniform(0.2, 3.0, size=(N, 1))          # rows of any length
    literal = literal_lgo(dense, label_index, group_ids, weights, everything)
    mine = cd.fast_lgo(dense, label_index, group_ids, weights, L, everything)
    check(np.abs(mine - literal).max() < 1e-10, "fast_lgo (dense, rows of any length) equals the literal leave-group-out loop")
    mine_fold = cd.fast_lgo(dense, label_index, group_ids, weights, L, fold_queries)
    check(np.abs(mine_fold - literal[fold_queries]).max() < 1e-10, "fast_lgo on one fold's queries equals the literal loop's rows")
    check(np.abs(mine_fold - dense_leave_group_out(dense, label_index, group_ids, weights, L, fold_queries)).max() < 1e-10,
          "fast_lgo equals the protocol's dense_leave_group_out")
    lexical = sparse.random(N, 70, density=0.25, random_state=3, format="csr", dtype=np.float64)
    lexical.data = np.abs(lexical.data) + 0.1
    lexical = sparse.csr_matrix(unit(np.asarray(lexical.todense())))
    literal_sparse = literal_lgo(np.asarray(lexical.todense()), label_index, group_ids, weights, everything)
    mine_sparse = cd.fast_lgo(lexical, label_index, group_ids, weights, L, everything)
    check(np.abs(mine_sparse - literal_sparse).max() < 1e-10, "fast_lgo (CSR) equals the literal loop")
    protocol = v1.score_leave_group_out(unit(dense).astype(np.float32), lexical.astype(np.float32), label_index, group_ids, L)
    check(np.abs(mine_sparse - protocol.lexical.astype(np.float64)).max() < 2e-5,
          "fast_lgo (CSR) equals the protocol's sparse scorer v1.score_leave_group_out (float32 reference)")
    check(np.abs(cd.fast_lgo(unit(dense), label_index, group_ids, weights, L, everything)
                 - protocol.dense.astype(np.float64)).max() < 2e-5, "fast_lgo (dense unit rows) equals the protocol's dense scorer")
    dots, norm2, _ = lgo_parts(lexical, label_index, group_ids, weights, L, everything[:80])
    check(np.abs(dots / np.sqrt(norm2) - mine_sparse[:80]).max() < 1e-10, "fast_lgo equals lgo_parts of label_size_calibration_v3")
    expect_error(lambda: cd.fast_lgo(dense, label_index, group_ids, weights, L, np.asarray([20])), RuntimeError,
                 "fast_lgo refuses queries that split a leakage group")
    lonely = label_index.copy()
    lonely[lonely == L - 1] = 0
    lonely[60:65] = L - 1                                          # the last label lives in one group only
    lonely_size = Counter(zip(group_ids.tolist(), lonely.tolist()))
    lonely_weights = np.asarray([1.0 / lonely_size[(int(g), int(l))] for g, l in zip(group_ids, lonely)])
    expect_error(lambda: cd.fast_lgo(dense, lonely, group_ids, lonely_weights, L, everything), RuntimeError,
                 "fast_lgo refuses a label emptied by its own query's group")

    # ---------------------------------------------------------------- 2. Burrows
    z = rng.standard_normal((N, 40))
    burrows = cd.burrows_lgo(z, SIZES, label_index, group_ids, weights, L, fold_queries)
    gap = gap_brute = 0.0
    for row, q in enumerate(fold_queries.tolist()):
        for size in SIZES:
            reference = literal_burrows(z, size, label_index, group_ids, weights, q)
            gap = max(gap, float(np.abs(burrows[size][row] - reference).max()))
            gap_brute = max(gap_brute, float(np.abs(cd.burrows_brute(z, size, label_index, group_ids, weights, L, q) - reference).max()))
    check(gap < 1e-10, "burrows_lgo equals the literal definition at every size, every query of a fold")
    check(gap_brute < 1e-10, "the script's burrows_brute (its in-run check) equals the literal definition")
    in_group = [row for row, q in enumerate(fold_queries.tolist()) if np.sum(group_ids == group_ids[q]) > 1]
    check(len(in_group) > 0, "the fold holds multi-song groups, so the held-out correction was exercised")
    expect_error(lambda: cd.burrows_lgo(z, (20, 10), label_index, group_ids, weights, L, fold_queries), ValueError,
                 "burrows_lgo refuses sizes that are not increasing")

    # ---------------------------------------------------------------- 3. Delta features
    rates = rng.gamma(0.6, 1.0, size=90)
    rates[:6] = 0.01                                               # rare columns: document-frequency ties
    drawn = rng.poisson(rates[None, :] * rng.uniform(0.5, 2.0, size=(N, 1))).astype(np.float32)
    drawn[:, 88:] = 1.0                                            # two tokens in every song: df ties at the top
    unigrams = sparse.csr_matrix(drawn)
    train_mask = fold != 2
    zed, facts = cd.delta_features(unigrams, train_mask, weights, SIZES)
    counts_dense = np.asarray(unigrams.todense(), dtype=np.float64)
    train_rows = [i for i in range(N) if train_mask[i]]
    df = [sum(1 for i in train_rows if counts_dense[i, j] > 0) for j in range(90)]
    tf = [sum(counts_dense[i, j] for i in train_rows) for j in range(90)]
    order = sorted(range(90), key=lambda j: (-df[j], -tf[j], j))[:SIZES[-1]]
    mass = sum(weights[i] for i in train_rows)
    z_literal = np.zeros((N, SIZES[-1]))
    for column, j in enumerate(order):
        relative = [counts_dense[i, j] / counts_dense[i].sum() for i in range(N)]
        mean = sum(weights[i] * relative[i] for i in train_rows) / mass
        sd = np.sqrt(sum(weights[i] * (relative[i] - mean) ** 2 for i in train_rows) / mass)
        z_literal[:, column] = [(r - mean) / sd for r in relative]
    check(np.abs(zed - z_literal).max() < 1e-9, "delta_features equals the literal MFW / relative-frequency / z-score loop")
    check(facts["document_frequency_of_the_last_word"] == {str(s): int(df[order[s - 1]]) for s in SIZES},
          "delta_features reports the document frequency of the last word of every size")
    check(len({df[j] for j in order}) < len(order), "the MFW order contained document-frequency ties, so the tie rule was exercised")
    disturbed = unigrams.tolil()
    test_rows = np.flatnonzero(~train_mask)
    disturbed[test_rows] = unigrams[rng.permutation(test_rows)]
    zed_disturbed, _ = cd.delta_features(sparse.csr_matrix(disturbed), train_mask, weights, SIZES)
    check(np.array_equal(zed_disturbed[train_mask], zed[train_mask]), "delta_features is blind to the test fold's rows (training z bitwise unchanged)")

    # ---------------------------------------------------------------- 4. count-matrix TF-IDF against sklearn
    documents, segmented, tokens_seen = planted_documents(7, label_index)
    counts = cd.build_counts(documents, segmented)
    sources = {"chars": documents, "words": segmented}
    subset = np.flatnonzero(fold != 1)
    worst = {"all": 0.0, "subset": 0.0}
    binding = {"all": 0, "subset": 0}
    cells = 0
    for space in cd.SPACES:
        for ngram, cap, sublinear in cd.grid_of(space, CAPS):
            matrix = counts[space][ngram]
            for kind, rows in (("all", None), ("subset", subset)):
                vectorizer = vectorizer_for(space, ngram, cap, sublinear)
                fit_documents = sources[space] if rows is None else [sources[space][i] for i in rows]
                vectorizer.fit(fit_documents)
                reference = vectorizer.transform(sources[space]).astype(np.float64)
                mask, idf, info = cd.vocabulary(matrix if rows is None else matrix[rows], cap)
                x, _ = cd.weigh(matrix, idf, sublinear)
                if x.shape[1] != matrix.shape[1] or int(mask.sum()) != reference.shape[1] or info["features"] != reference.shape[1]:
                    raise SystemExit(f"FAILED: feature count differs for {space} {ngram} {cap} {sublinear} {kind}")
                difference = (x[:, np.flatnonzero(mask)] - reference).tocsr()
                worst[kind] = max(worst[kind], float(np.abs(difference.data).max()) if difference.nnz else 0.0)
                binding[kind] += int(info["cap_active"])
                cells += 1
    check(worst["all"] < 1e-6, f"count-matrix TF-IDF equals TfidfVectorizer.fit_transform in every grid cell (gap {worst['all']:.1e})")
    check(worst["subset"] < 1e-6, f"count-matrix TF-IDF fitted on a subset equals TfidfVectorizer.fit(subset).transform(all) (gap {worst['subset']:.1e})")
    check(0 < binding["all"] < cells // 2 and 0 < binding["subset"] < cells // 2, "the grid held both binding and non-binding caps")
    check(counts["words"]["1"].shape[1] < counts["words"]["1-2"].shape[1] and counts["chars"]["2-4"].shape[1] < counts["chars"]["2-5"].shape[1],
          "the n-gram ranges are nested column subsets")

    # ---------------------------------------------------------------- 5. inner problem, tie rule, worst rank
    train = np.flatnonzero(fold != 0)
    scarce = label_index.copy()
    victims = np.flatnonzero((scarce == 3) & (fold != 0))
    scarce[victims[1:]] = 4                                        # label 3 keeps one training song
    scarce_size = Counter(zip(group_ids.tolist(), scarce.tolist()))
    scarce_weights = np.asarray([1.0 / scarce_size[(int(g), int(l))] for g, l in zip(group_ids, scarce)])
    rows, inner_labels, inner_count = cd.inner_problem(train, scarce, group_ids, scarce_weights, L)
    expected_labels = [l for l in range(L) if len({int(group_ids[i]) for i in train if scarce[i] == l}) >= 2]
    check(3 not in expected_labels and inner_count == len(expected_labels) == L - 1, "inner_problem drops the label with one training component")
    check(set(rows.tolist()) == {int(i) for i in train if scarce[i] in expected_labels} and
          np.array_equal(inner_labels, np.asarray([expected_labels.index(int(scarce[i])) for i in rows])),
          "inner_problem keeps the other labels' training songs and renumbers labels in order")
    inner_scores = cd.fast_lgo(dense[rows], inner_labels, group_ids[rows], scarce_weights[rows], inner_count, np.arange(len(rows)))
    check(np.all(np.isfinite(inner_scores)), "the inner problem scores without an emptied label")
    grid = cd.grid_of("chars", CAPS)
    flat = {c: 0.5 for c in grid}
    check(cd.select_config(flat, grid, CURRENT["chars"]) == CURRENT["chars"], "select_config: a tie goes to the published configuration")
    lifted = dict(flat)
    lifted[grid[5]] = lifted[grid[9]] = 0.6
    check(cd.select_config(lifted, grid, CURRENT["chars"]) == grid[5], "select_config: otherwise a tie goes to grid order")
    lifted[grid[11]] = 0.7
    check(cd.select_config(lifted, grid, CURRENT["chars"]) == grid[11], "select_config: the highest inner MRR wins")
    toy_scores = np.asarray([[0.9, 0.1, 0.0], [0.0, 0.0, 0.0]])
    toy = cd.reciprocal_ranks(toy_scores, np.asarray([0, 0]), 3, undefined=np.asarray([False, True]))
    check(np.allclose(toy, [1.0, 1.0 / 3.0]), "reciprocal_ranks gives an empty row the worst rank")

    # ---------------------------------------------------------------- 6. question B end to end
    rr, report, checks_b, cell_rr = cd.question_b(counts, label_index, group_ids, weights, L, fold, max_features=CAPS, current=CURRENT)
    check(all(checks_b["fit_rows_are_the_training_folds_by_construction"]), "every inner and inductive fit excludes its test fold")
    random_dense = unit(rng.standard_normal((N, 6))).astype(np.float32)
    for space in cd.SPACES:
        ngram, cap, sublinear = CURRENT[space]
        reference = vectorizer_for(space, ngram, cap, sublinear).fit_transform(sources[space]).tocsr().astype(np.float32)
        scores = v1.score_leave_group_out(random_dense, reference, label_index, group_ids, L).lexical
        check(np.array_equal(1.0 / ranks_of(scores, label_index), rr[f"{space}_current"]),
              f"{space}_current equals TfidfVectorizer + the protocol's scorer, query by query")
        k = 1
        train = np.flatnonzero(fold != k)
        test = np.flatnonzero(fold == k)
        rows, inner_labels, inner_count = cd.inner_problem(train, label_index, group_ids, weights, L)
        independent = {}
        for config in cd.grid_of(space, CAPS):
            vectorizer = vectorizer_for(space, *config)
            vectorizer.fit([sources[space][i] for i in train])
            x = np.asarray(vectorizer.transform([sources[space][i] for i in rows]).todense(), dtype=np.float64)
            s = dense_leave_group_out(x, inner_labels, group_ids[rows], weights[rows], inner_count, np.arange(len(rows)))
            independent[config] = float(np.mean(1.0 / ranks_of(s, inner_labels)))
        best = max(independent.values())
        tied = [c for c in cd.grid_of(space, CAPS) if independent[c] >= best - 1e-12]
        choice = CURRENT[space] if CURRENT[space] in tied else tied[0]
        entry = report[space]["inner_selection_by_fold"][str(k)]
        check(entry["selected"] == cd.config_name(choice) and abs(entry["inner_mrr_selected"] - round(best, 4)) < 1e-9
              and abs(entry["inner_mrr_published"] - round(independent[CURRENT[space]], 4)) < 1e-9,
              f"{space}: fold {k}'s inner selection equals an independent TfidfVectorizer + dense_leave_group_out selection")
        check(entry["inner_queries"] == len(rows) and entry["inner_labels"] == inner_count, f"{space}: the inner problem's size is reported")
        check(len({round(v, 6) for v in independent.values()}) > 3 and choice != CURRENT[space],
              f"{space}: the inner grid's cells differ and the selection left the published cell, so the selection path was exercised")
        vectorizer = vectorizer_for(space, ngram, cap, sublinear)
        vectorizer.fit([sources[space][i] for i in train])
        x = np.asarray(vectorizer.transform(sources[space]).todense(), dtype=np.float64)
        s = dense_leave_group_out(x, label_index, group_ids, weights, L, test)
        check(np.array_equal(1.0 / ranks_of(s, label_index[test]), rr[f"{space}_current_inductive"][test]),
              f"{space}_current_inductive equals TfidfVectorizer.fit(train).transform(all) + dense_leave_group_out on fold {k}")
        vectorizer = vectorizer_for(space, *choice)
        vectorizer.fit([sources[space][i] for i in train])
        x = np.asarray(vectorizer.transform(sources[space]).todense(), dtype=np.float64)
        s = dense_leave_group_out(x, label_index, group_ids, weights, L, test)
        check(np.array_equal(1.0 / ranks_of(s, label_index[test]), rr[f"{space}_inner_selected_inductive"][test]),
              f"{space}_inner_selected_inductive equals the selected configuration fitted on the training folds, fold {k}")
        reference = vectorizer_for(space, *choice).fit_transform(sources[space]).tocsr().astype(np.float32)
        scores = v1.score_leave_group_out(random_dense, reference, label_index, group_ids, L).lexical
        check(np.array_equal((1.0 / ranks_of(scores, label_index))[test], rr[f"{space}_inner_selected"][test]),
              f"{space}_inner_selected takes fold {k}'s rows from the selected configuration fitted on all songs")
        table = report[space]["all_songs_grid"]
        check(len(table) == len(cd.grid_of(space, CAPS)) and sum(t["published_configuration"] for t in table) == 1,
              f"{space}: the all-songs grid reports every cell and marks the published one")
        published_row = next(t for t in table if t["published_configuration"])
        other = next(t for t in table if not t["published_configuration"])
        other_config = next(c for c in cd.grid_of(space, CAPS) if cd.config_name(c) == other["configuration"])
        point = float(np.sum((cell_rr[space][other_config] - rr[f"{space}_current"]) * weights) / weights.sum())
        check(published_row["minus_published"] is None and abs(other["minus_published"]["mrr_difference"] - round(point, 4)) < 1e-9
              and abs(other["mrr_component_weighted"] - round(float(np.sum(cell_rr[space][other_config] * weights) / weights.sum()), 4)) < 1e-9,
              f"{space}: every other cell carries its contrast against the published cell, on the weighted estimand")
        for sublinear, system in (("true", "best_sublinear_tf"), ("false", "best_raw_tf")):
            within = max(t["mrr"] for t in table if t["configuration"].endswith(f"sublinear_tf {sublinear}"))
            check(abs(cd.summary(rr[f"{space}_{system}"])["mrr"] - within) < 1e-9
                  and report[space]["best_cell_within_tf_setting"][system].endswith(f"sublinear_tf {sublinear}"),
                  f"{space}_{system} is the highest all-songs cell of that tf setting")
        check(set(report[space]["inner_selection_by_fold"]["1"]["inner_mrr_by_cell"]) == {cd.config_name(c) for c in cd.grid_of(space, CAPS)}
              and all(abs(report[space]["inner_selection_by_fold"]["1"]["inner_mrr_by_cell"][cd.config_name(c)] - round(independent[c], 4)) < 1e-9
                      for c in cd.grid_of(space, CAPS)),
              f"{space}: every inner cell's MRR is reported and equals the independent one")
        extreme = (min if space == "words" else max)(t["mrr"] for t in table)
        name = "words_grid_worst" if space == "words" else "chars_grid_best"
        check(abs(cd.summary(rr[name])["mrr"] - extreme) < 1e-9, f"{name} is the grid's extreme cell")

    # blindness: shuffle fold 0's rows among themselves; fold 0's selection must not move
    test0 = np.flatnonzero(fold == 0)
    shuffled = rng.permutation(test0)
    disturbed_counts = {space: {} for space in cd.SPACES}
    for space in cd.SPACES:
        for ngram, matrix in counts[space].items():
            lil = matrix.tolil()
            lil[test0] = matrix[shuffled]
            disturbed_counts[space][ngram] = sparse.csr_matrix(lil)
    rr2, report2, _, _ = cd.question_b(disturbed_counts, label_index, group_ids, weights, L, fold, max_features=CAPS, current=CURRENT)
    inner_keys = ("selected", "selected_cap_active", "inner_mrr_by_cell", "inner_mrr_selected", "inner_mrr_published", "inner_rank_of_published", "inner_mrr_range", "selected_features")
    for space in cd.SPACES:
        a, b = report[space]["inner_selection_by_fold"]["0"], report2[space]["inner_selection_by_fold"]["0"]
        check(all(a[key] == b[key] for key in inner_keys), f"{space}: fold 0's inner selection is blind to fold 0's rows")
        check(not np.array_equal(rr[f"{space}_current"], rr2[f"{space}_current"]), f"{space}: the disturbance did change the scored rows")

    # ---------------------------------------------------------------- 7. question A end to end
    word_unigrams = counts["words"]["1"]
    rr_a, fold_info, checks_a = cd.question_a(word_unigrams, label_index, group_ids, weights, L, fold, sizes=SIZES)
    check(checks_a["burrows_brute_force_gap"] < 1e-9 and checks_a["dense_scorer_gap"] < 1e-9 and all(checks_a["fit_rows_are_the_training_folds_by_construction"]),
          "question_a's in-run checks pass")
    k = 3
    train_mask = fold != k
    test = np.flatnonzero(fold == k)
    dense_counts = np.asarray(word_unigrams.todense(), dtype=np.float64)
    train_rows = [i for i in range(N) if train_mask[i]]
    width = dense_counts.shape[1]
    df = [sum(1 for i in train_rows if dense_counts[i, j] > 0) for j in range(width)]
    tf = [sum(dense_counts[i, j] for i in train_rows) for j in range(width)]
    order = sorted(range(width), key=lambda j: (-df[j], -tf[j], j))[:SIZES[-1]]
    relative = dense_counts[:, order] / dense_counts.sum(axis=1, keepdims=True)
    share = weights[train_mask] / weights[train_mask].sum()
    mean = share @ relative[train_mask]
    sd = np.sqrt(share @ (relative[train_mask] - mean) ** 2)
    z_fold = (relative - mean) / sd
    for size in SIZES:
        zn = z_fold[:, :size]
        cosine = np.zeros((len(test), L))
        for row, q in enumerate(test.tolist()):
            for label in range(L):
                keep = (label_index == label) & (group_ids != group_ids[q])
                mean_z = (zn[keep] * weights[keep, None]).sum(axis=0) / weights[keep].sum()
                cosine[row, label] = float(zn[q] @ mean_z) / (np.linalg.norm(zn[q]) * np.linalg.norm(mean_z))
        check(np.array_equal(1.0 / ranks_of(cosine, label_index[test]), rr_a[f"cosine_delta/{size}"][test]),
              f"cosine_delta at {size} MFW equals the cosine with the leave-group-out mean z-vector, fold {k}")
        prototype = literal_lgo(unit(zn), label_index, group_ids, weights, test.tolist())
        check(np.array_equal(1.0 / ranks_of(prototype, label_index[test]), rr_a[f"cosine_delta_unit_rows/{size}"][test]),
              f"cosine_delta_unit_rows at {size} MFW equals the prototype on unit z-rows, fold {k}")
        distance = np.stack([literal_burrows(z_fold, size, label_index, group_ids, weights, q) for q in test.tolist()])
        check(np.array_equal(1.0 / ranks_of(distance, label_index[test]), rr_a[f"burrows_delta/{size}"][test]),
              f"burrows_delta at {size} MFW equals the literal definition, fold {k}")
    check(all(np.all((v > 0) & (v <= 1)) for v in rr_a.values()) and len(rr_a) == len(cd.DELTA_SCORERS) * len(SIZES),
          "question_a fills every query of every scorer and size")
    chance = float(np.mean(1.0 / np.arange(1, L + 1)))
    check(cd.summary(rr_a[f"cosine_delta/{SIZES[-1]}"])["mrr"] > chance + 0.1 and cd.summary(rr["words_current"])["mrr"] > chance + 0.1,
          "the planted signal is found by Delta and by the word space")

    # ---------------------------------------------------------------- 8. reading rules
    def contrast(system, minus, difference, low, high):
        return {"system": system, "minus": minus, "mrr_difference": difference, "ci95": [low, high], "excludes_zero": bool(low > 0 or high < 0)}

    def delta_systems(values):
        return {f"{scorer}/{size}": {"mrr": values[scorer][i]} for scorer in cd.DELTA_SCORERS for i, size in enumerate(SIZES)}

    def delta_contrasts(systems):
        return [contrast(name, "words_current", round(v["mrr"] - 0.4963, 4), -0.1, -0.01) for name, v in systems.items()]

    systems = delta_systems({"cosine_delta": [0.30, 0.48, 0.47], "cosine_delta_unit_rows": [0.2, 0.3, 0.3], "burrows_delta": [0.1, 0.2, 0.2]})
    verdict = cd.reading_a(systems, delta_contrasts(systems), sizes=SIZES)
    check(verdict["by_scorer"]["cosine_delta"]["best_size"] == 20 and verdict["by_scorer"]["cosine_delta"]["matches"]
          and verdict["verdict"].startswith("Delta matches") and verdict["threshold"] == 0.4763, "reading A: the primary scorer within 0.02 matches")
    systems = delta_systems({"cosine_delta": [0.30, 0.4762, 0.47], "cosine_delta_unit_rows": [0.2, 0.3, 0.49], "burrows_delta": [0.1, 0.2, 0.2]})
    verdict = cd.reading_a(systems, delta_contrasts(systems), sizes=SIZES)
    check(verdict["verdict"].startswith("Delta does not match") and "cosine_delta_unit_rows" in verdict["verdict"]
          and not verdict["by_scorer"]["burrows_delta"]["matches"], "reading A: below the threshold does not match; another scorer reaching it is named")
    systems = delta_systems({"cosine_delta": [0.30, 0.30, 0.20], "cosine_delta_unit_rows": [0.2, 0.3, 0.3], "burrows_delta": [0.1, 0.2, 0.2]})
    verdict = cd.reading_a(systems, delta_contrasts(systems), sizes=SIZES)
    check(verdict["by_scorer"]["cosine_delta"]["best_size"] == 10 and "but reaching" not in verdict["verdict"],
          "reading A: a tie between sizes goes to the smaller; nothing is named when nothing reaches the threshold")

    def b_contrasts(primary, inductive):
        out = [contrast(left, right, 0.0, -0.01, 0.01) for left, right in cd.B_PAIRS]
        for c in out:
            if (c["system"], c["minus"]) == ("words_inner_selected", "chars_inner_selected"):
                c.update(mrr_difference=primary[0], ci95=[primary[1], primary[2]])
            if (c["system"], c["minus"]) == ("words_inner_selected_inductive", "chars_inner_selected_inductive"):
                c.update(mrr_difference=inductive[0], ci95=[inductive[1], inductive[2]])
        return out

    check(cd.reading_b(b_contrasts((0.06, 0.04, 0.08), (0.05, 0.03, 0.07)))["verdict"] == "configuration-robust", "reading B: positive and clear of zero is robust")
    check(cd.reading_b(b_contrasts((0.01, -0.01, 0.03), (0.01, -0.01, 0.03)))["verdict"] == "configuration-sensitive", "reading B: an interval over zero is sensitive")
    check(cd.reading_b(b_contrasts((-0.05, -0.07, -0.03), (-0.05, -0.07, -0.03)))["verdict"] == "configuration-sensitive", "reading B: a reversed ordering is sensitive")
    disagreeing = cd.reading_b(b_contrasts((0.06, 0.04, 0.08), (0.01, -0.01, 0.03)))
    check(disagreeing["verdict"] == "configuration-robust" and not disagreeing["inductive_agrees"] and "disagrees" in disagreeing["text"],
          "reading B: a disagreeing inductive contrast is named")

    # ---------------------------------------------------------------- 9. the payload holds no token
    all_mask = np.ones(N, dtype=bool)
    contrasts_b = paired_group_bootstrap(rr, weights, group_ids, all_mask, cd.B_PAIRS)
    point = (np.sum(rr["words_current"] * weights) - np.sum(rr["chars_current"] * weights)) / weights.sum()
    check(abs(contrasts_b[0]["mrr_difference"] - round(float(point), 4)) < 1e-9, "a contrast's point is the weighted-mean difference")
    rr_a["words_current"] = rr["words_current"]
    contrasts_a = paired_group_bootstrap(rr_a, weights, group_ids, all_mask, [(name, "words_current") for name in rr_a if name != "words_current"])
    systems_a = {name: cd.summary(values) for name, values in rr_a.items()}
    payload = {"question_a": {"systems": systems_a, "contrasts": contrasts_a, "fold_info": fold_info, "checks": checks_a,
                              "reading": cd.reading_a(systems_a, contrasts_a, sizes=SIZES)},
               "question_b": {"systems": {name: cd.summary(values) for name, values in rr.items()}, "contrasts": contrasts_b, **report,
                              "reading": cd.reading_b(contrasts_b)}}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    leaked = [token for token in tokens_seen if token.strip() and token.lower() in text.lower() and token not in ("OK", "Yo", "yo")]
    han = [ch for ch in text if "一" <= ch <= "鿿"]
    check(not leaked and not han, "the payload is JSON-serialisable and holds no token or Han character of the data")

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
