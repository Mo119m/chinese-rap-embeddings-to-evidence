#!/usr/bin/env python3
"""Synthetic test of label_mask_sensitivity_v3: random data, no private corpus.

Checks, against brute-force definitions written independently here:
  * the population rule (five songs and two leakage groups per label), its group ids and
    the per-(group, label) weights;
  * score_spaces (the protocol scorer wrapper) against a brute-force leave-group-out cosine
    for the dense space and for two sparse spaces, and its empty-row report;
  * evaluate: ranks with the protocol's tie rule, plain and component-weighted MRR, recall,
    the worst rank for a sparse row without features, the query mask, and the bootstrap's
    point estimate as the weighted MRR difference;
  * mask_all_labels: hand-computed counts, own/other split, whole-word Latin rule,
    case-insensitivity, longest-first order, skipped one-character labels, and agreement of
    the masked text with sequential mask_label; names_label on the audit's definition;
  * nearest_rival and song_maximum against double loops;
  * reading_of on hand-made contrasts;
  * an end-to-end masked run on a synthetic text corpus: masked documents carry no label
    string, and every space's MRR equals a brute-force MRR from the fitted matrices.

    set CHINESE_RAP_CORPUS=v3
    python test_label_mask_sensitivity.py          (or: python -m pytest test_label_mask_sensitivity.py)
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

os.environ.setdefault("CHINESE_RAP_CORPUS", "v3")
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("label_mask_sensitivity_v3", HERE.parent / "src" / "label_mask_sensitivity_v3.py")
lms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lms)

from train_identity_encoder_v3 import mask_label  # noqa: E402


# ------------------------------------------------------------------ synthetic data
def make_population(seed: int = 3, label_count: int = 14, dim: int = 16):
    """Songs with labels, leakage groups of 1-3 songs (some across labels), dense unit rows
    and two sparse unit-row matrices, plus the protocol's population dictionary."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(5, 14, size=label_count)
    labels = [f"L{l:02d}" for l in range(label_count)]
    songs = [f"s{k:04d}" for k in range(int(sizes.sum()))]
    label_by_song = {}
    k = 0
    for l, size in enumerate(sizes):
        for _ in range(size):
            label_by_song[songs[k]] = labels[l]
            k += 1
    n = len(songs)
    group_of_index = np.arange(n)
    for _ in range(n // 6):
        a, b = rng.integers(0, n, size=2)
        group_of_index[group_of_index == group_of_index[b]] = group_of_index[a]
    group_of = {s: f"g{int(g):04d}" for s, g in zip(songs, group_of_index)}
    pop = lms.population(songs, label_by_song, group_of)
    dense = rng.normal(size=(n, dim))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)

    def sparse_unit(features: int, density: float):
        rows = []
        for _ in range(n):
            nnz = max(1, int(rng.binomial(features, density)))
            cols = rng.choice(features, size=nnz, replace=False)
            vals = rng.random(nnz)
            rows.append((cols, vals / np.linalg.norm(vals)))
        indptr = np.cumsum([0] + [len(c) for c, _ in rows])
        indices = np.concatenate([c for c, _ in rows])
        data = np.concatenate([v for _, v in rows]).astype(np.float32)
        return sparse.csr_matrix((data, indices, indptr), shape=(n, features))

    return dict(songs=songs, labels=labels, label_by_song=label_by_song, group_of=group_of, pop=pop,
                dense=dense, chars=sparse_unit(300, 0.05), words=sparse_unit(200, 0.06))


def brute_force_lgo(x, label_index, group_ids, weights, label_count):
    """For each query and label: cosine with the weighted sum of the label's songs outside the
    query's leakage group. Written from the definition, no vectorisation."""
    x = np.asarray(x.todense() if sparse.issparse(x) else x, dtype=np.float64)
    n = len(label_index)
    out = np.zeros((n, label_count))
    for q in range(n):
        for l in range(label_count):
            members = [s for s in range(n) if label_index[s] == l and group_ids[s] != group_ids[q]]
            profile = sum(weights[s] * x[s] for s in members)
            out[q, l] = float(x[q] @ profile) / float(np.linalg.norm(profile))
    return out


def brute_force_ranks(scores, truth):
    """The protocol's tie rule: strictly higher scores count, ties count only for earlier labels."""
    ranks = np.zeros(len(truth), dtype=np.int64)
    for q in range(len(truth)):
        t = float(scores[q, truth[q]])
        rank = 1
        for l in range(scores.shape[1]):
            if scores[q, l] > t + 1e-12 or (abs(scores[q, l] - t) <= 1e-12 and l < truth[q]):
                rank += 1
        ranks[q] = rank
    return ranks


# ------------------------------------------------------------------ tests
def test_population_rule():
    d = make_population()
    pop, label_by_song, group_of = d["pop"], d["label_by_song"], d["group_of"]
    by_label = defaultdict(list)
    for s in d["songs"]:
        by_label[label_by_song[s]].append(s)
    expected_labels = sorted(l for l, m in by_label.items()
                             if len(m) >= 5 and len({group_of[s] for s in m}) >= 2)
    assert pop["labels"] == expected_labels
    assert pop["songs"] == [s for s in d["songs"] if label_by_song[s] in set(expected_labels)]
    assert pop["label_count"] == len(expected_labels)
    for k, s in enumerate(pop["songs"]):
        assert expected_labels[pop["label_index"][k]] == label_by_song[s]
    # group ids: same group <=> same id; ids dense from zero
    for a in range(len(pop["songs"])):
        for b in range(a, len(pop["songs"])):
            same = group_of[pop["songs"][a]] == group_of[pop["songs"][b]]
            assert same == (pop["group_ids"][a] == pop["group_ids"][b])
    assert sorted(set(pop["group_ids"].tolist())) == list(range(pop["groups"]))
    size = Counter(zip(pop["group_ids"].tolist(), pop["label_index"].tolist()))
    for k in range(len(pop["songs"])):
        assert abs(pop["weights"][k] - 1.0 / size[(int(pop["group_ids"][k]), int(pop["label_index"][k]))]) < 1e-12
    assert pop["largest_group"] == max(Counter(pop["group_ids"].tolist()).values())
    # a label with one group only is dropped, a label with four songs is dropped
    songs = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(4)] + [f"c{i}" for i in range(5)]
    lbs = {s: s[0] for s in songs}
    grp = {s: ("A" if s[0] == "a" else s) for s in songs}
    p = lms.population(songs, lbs, grp)
    assert p["labels"] == ["c"] and p["songs"] == [f"c{i}" for i in range(5)]
    assert p["songs_dropped_by_label_rule"] == 10 and p["labels_dropped_by_label_rule"] == 2
    # keep: a filtered population
    p = lms.population(songs, lbs, grp, keep={s: s != "c0" for s in songs})
    assert p["songs_offered"] == 14 and p["labels"] == [] and p["songs"] == []


def test_score_spaces_against_brute_force():
    d = make_population()
    pop = d["pop"]
    li, gi, w, L = pop["label_index"], pop["group_ids"], pop["weights"], pop["label_count"]
    scores, empty = lms.score_spaces(pop, d["dense"], d["chars"], d["words"])
    assert set(scores) == {"semantic", "characters", "words"}
    for name, x, tol in (("semantic", d["dense"], 1e-5), ("characters", d["chars"], 1e-5), ("words", d["words"], 1e-5)):
        expect = brute_force_lgo(x, li, gi, w, L)
        gap = float(np.abs(scores[name] - expect).max())
        assert gap < tol, f"{name}: gap {gap}"
    assert not empty["characters"].any() and not empty["words"].any()
    scores2, _ = lms.score_spaces(pop, d["dense"], d["chars"], d["words"], with_semantic=False)
    assert "semantic" not in scores2 and np.array_equal(scores2["words"], scores["words"])
    # an empty sparse row is reported
    chars = d["chars"].tolil()
    chars[3, :] = 0
    _, empty = lms.score_spaces(pop, d["dense"], chars.tocsr(), d["words"])
    assert empty["characters"][3] and empty["characters"].sum() == 1


def test_evaluate_against_brute_force():
    d = make_population()
    pop = d["pop"]
    li, gi, w, L = pop["label_index"], pop["group_ids"], pop["weights"], pop["label_count"]
    scores, empty = lms.score_spaces(pop, d["dense"], d["chars"], d["words"])
    systems, contrasts = lms.evaluate(pop, scores, empty)
    rr = {}
    for name in ("semantic", "characters", "words"):
        ranks = brute_force_ranks(scores[name], li)
        rr[name] = 1.0 / ranks
        assert systems[name]["mrr"] == round(float(np.mean(rr[name])), 4)
        assert systems[name]["mrr_component_weighted"] == round(float(np.sum(rr[name] * w) / np.sum(w)), 4)
        assert systems[name]["recall_at_1"] == round(float(np.mean(ranks <= 1)), 4)
        assert systems[name]["recall_at_10"] == round(float(np.mean(ranks <= 10)), 4)
        assert systems[name]["queries"] == len(li) and systems[name]["rows_without_features"] == 0
    assert [(c["system"], c["minus"]) for c in contrasts] == list(lms.PAIRS)
    for c in contrasts:
        point = float(np.sum((rr[c["system"]] - rr[c["minus"]]) * w) / np.sum(w))
        assert c["mrr_difference"] == round(point, 4)
        assert c["ci95"][0] <= c["mrr_difference"] <= c["ci95"][1]
    # a query mask restricts the estimands to the kept queries
    mask = np.zeros(len(li), dtype=bool)
    mask[::3] = True
    sub, sub_contrasts = lms.evaluate(pop, scores, empty, mask)
    for name in rr:
        assert sub[name]["mrr"] == round(float(np.mean(rr[name][mask])), 4)
        assert sub[name]["queries"] == int(mask.sum())
    for c in sub_contrasts:
        point = float(np.sum((rr[c["system"]][mask] - rr[c["minus"]][mask]) * w[mask]) / np.sum(w[mask]))
        assert c["mrr_difference"] == round(point, 4)
    # a sparse row without features takes the worst rank
    empty2 = {"characters": np.zeros(len(li), dtype=bool), "words": np.zeros(len(li), dtype=bool)}
    empty2["words"][[0, 5]] = True
    sys2, _ = lms.evaluate(pop, scores, empty2)
    ranks = brute_force_ranks(scores["words"], li)
    ranks[[0, 5]] = L
    assert sys2["words"]["mrr"] == round(float(np.mean(1.0 / ranks)), 4)
    assert sys2["words"]["rows_without_features"] == 2
    # the semantic column pending: only words - characters is contrasted
    partial = {k: v for k, v in scores.items() if k != "semantic"}
    sys3, con3 = lms.evaluate(pop, partial, empty)
    assert set(sys3) == {"characters", "words"} and [(c["system"], c["minus"]) for c in con3] == [("words", "characters")]


def test_mask_all_labels():
    labels = ["MC Alpha", "Beta", "小龙", "龙人", "X"]
    texts = ["MC Alpha and Beta with 小龙人 and betamax and mc alpha", "nothing here", "Beta Beta 龙人"]
    own = [0, 3, 1]
    masked, counts, tally = lms.mask_all_labels(texts, labels, own)
    assert counts.tolist() == [2, 3, 1, 1, 0]
    assert tally == {"own_label_occurrences": 4, "other_label_occurrences": 3, "chunks_naming_own_label": 2,
                     "chunks_naming_another_label": 2, "chunks_touched": 2}
    assert masked[1] == "nothing here"
    for text in masked:
        for label in labels:
            if len(label) < 2:
                continue
            if lms.LATIN.match(label):
                assert re.search(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])", text, re.I) is None
            else:
                assert label not in text
    assert "betamax" in masked[0]            # a whole-word rule for Latin labels
    assert "人" in masked[0]                  # 小龙 removed first leaves 人; 龙人 no longer present
    # the masked text is what sequential mask_label gives, longest label first
    for text, out in zip(texts, masked):
        expect = text
        for label in sorted((l for l in labels if len(l) >= 2), key=lambda l: (-len(l), l)):
            expect = mask_label(expect, label)
        assert expect == out
    # without `own`, everything is an other-label mention; a label of one character is skipped
    _, counts2, tally2 = lms.mask_all_labels(texts, labels)
    assert counts2.tolist() == counts.tolist() and tally2["own_label_occurrences"] == 0 and tally2["other_label_occurrences"] == 7
    _, counts3, _ = lms.mask_all_labels(["XXX X x"], ["X"])
    assert counts3.tolist() == [0]
    # names_label: the audit's definition
    assert lms.names_label("hello mc alpha!", "MC Alpha") and not lms.names_label("mcalpha", "MC Alpha")
    assert lms.names_label("小龙人", "小龙") and not lms.names_label("小人", "小龙") and not lms.names_label("X", "X")
    summary = lms.masking_summary(counts, tally, len(texts), [len(l) for l in labels])
    assert summary["occurrences_total"] == 7 and summary["labels_with_any_occurrence"] == 4
    assert summary["labels_masked"] == 4 and summary["labels_under_two_characters_skipped"] == 1
    assert summary["occurrences_by_label_length_in_characters"]["2"] == {"labels": 2, "occurrences": 2}
    assert summary["share_of_chunks_touched"] == round(2 / 3, 4)


def test_nearest_rival_against_brute_force():
    rng = np.random.default_rng(11)
    n = 70
    vec = rng.normal(size=(n, 8))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    label = rng.integers(0, 5, size=n)
    group = rng.integers(0, 12, size=n)
    anywhere, outside = lms.nearest_rival(vec, label, group, block=16)
    for i in range(n):
        a = max(float(vec[i] @ vec[j]) for j in range(n) if label[j] != label[i])
        o = [float(vec[i] @ vec[j]) for j in range(n) if label[j] != label[i] and group[j] != group[i]]
        assert abs(anywhere[i] - a) < 1e-12
        assert (abs(outside[i] - max(o)) < 1e-12) if o else outside[i] == -np.inf
    song = rng.integers(0, 9, size=n)
    m = lms.song_maximum(anywhere, song, 9)
    for s in range(9):
        members = [i for i in range(n) if song[i] == s]
        assert (abs(m[s] - max(anywhere[i] for i in members)) < 1e-12) if members else m[s] == -np.inf


def test_reading_rule():
    def c(system, minus, diff, lo, hi):
        return {"system": system, "minus": minus, "mrr_difference": diff, "ci95": [lo, hi], "excludes_zero": lo > 0 or hi < 0}
    good = [c("words", "characters", 0.07, 0.06, 0.08), c("characters", "semantic", 0.13, 0.12, 0.14),
            c("words", "semantic", 0.2, 0.19, 0.21)]
    assert lms.reading_of(good, ("semantic", "characters", "words")) == "robust to the arm"
    bad = [c("words", "characters", 0.01, -0.01, 0.03)] + good[1:]
    assert lms.reading_of(bad, ("semantic", "characters", "words")).startswith("not robust to the arm: ['words - characters']")
    flipped = good[:1] + [c("characters", "semantic", -0.02, -0.03, -0.01)] + good[2:]
    assert "characters - semantic" in lms.reading_of(flipped, ("semantic", "characters", "words"))
    assert lms.reading_of(good[:1], ("characters", "words")) == "partial: words > characters holds; semantic pending"
    assert lms.reading_of(bad[:1], ("characters", "words")).startswith("partial: not robust")


def make_text_corpus(seed: int = 5, label_count: int = 10, songs_per_label: int = 7, chars: int = 30):
    """Random Han text over a small alphabet with label names dropped in, so that masking
    has something to remove and the n-gram spaces have shared features."""
    rng = np.random.default_rng(seed)
    alphabet = [chr(0x4E00 + i) for i in range(chars)]
    latin_names = [f"MC {chr(65 + l)}ra" for l in range(label_count // 2)]
    han_names = ["".join(chr(0x9F00 + 7 * l + j) for j in range(2)) for l in range(label_count - label_count // 2)]
    labels = latin_names + han_names
    label_by_song, documents, songs = {}, {}, []
    for l, name in enumerate(labels):
        for k in range(songs_per_label):
            song = f"song{l:02d}_{k}"
            lines = []
            for _ in range(6):
                line = "".join(rng.choice(alphabet, size=int(rng.integers(8, 16))))
                if rng.random() < 0.5:
                    line = line + " " + name + " "
                if rng.random() < 0.2:
                    line = labels[int(rng.integers(0, label_count))] + line
                lines.append(line)
            songs.append(song)
            label_by_song[song] = name
            documents[song] = "\n".join(lines)
    group_of = {s: s for s in songs}
    # two songs share a group inside a label, one pair across labels
    group_of[songs[1]] = songs[0]
    group_of[songs[songs_per_label + 2]] = songs[songs_per_label]
    group_of[songs[-1]] = songs[songs_per_label * 3]
    return dict(songs=songs, labels=labels, label_by_song=label_by_song, documents=documents, group_of=group_of)


def test_masked_pipeline_end_to_end():
    import jieba
    jieba.setLogLevel(60)
    d = make_text_corpus()
    songs, labels = d["songs"], d["labels"]
    own = [labels.index(d["label_by_song"][s]) for s in songs]
    masked, counts, tally = lms.mask_all_labels([d["documents"][s] for s in songs], labels, own)
    assert counts.sum() > 0 and tally["own_label_occurrences"] > 0 and tally["other_label_occurrences"] > 0
    for text in masked:
        for label in labels:
            if lms.LATIN.match(label):
                assert re.search(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])", text, re.I) is None
            else:
                assert label not in text
    keep = {s: len(lms.v1.normalized_text(m)) >= lms.v1.MIN_EFFECTIVE_CHARACTERS for s, m in zip(songs, masked)}
    pop = lms.population(songs, d["label_by_song"], d["group_of"], keep)
    assert pop["label_count"] == len(labels) and len(pop["songs"]) >= len(songs) - 3
    masked_of = dict(zip(songs, masked))
    char_matrix, word_matrix = lms.fit_spaces([masked_of[s] for s in pop["songs"]])
    for m in (char_matrix, word_matrix):
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        assert np.allclose(norms[norms > 0], 1.0, atol=2e-5)
    rng = np.random.default_rng(1)
    dense = rng.normal(size=(len(pop["songs"]), 12))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    scores, empty = lms.score_spaces(pop, dense, char_matrix, word_matrix, with_semantic=False)
    systems, contrasts = lms.evaluate(pop, scores, empty)
    li, gi, w, L = pop["label_index"], pop["group_ids"], pop["weights"], pop["label_count"]
    for name, x in (("characters", char_matrix), ("words", word_matrix)):
        expect = brute_force_lgo(x, li, gi, w, L)
        assert float(np.abs(scores[name] - expect).max()) < 1e-5
        ranks = brute_force_ranks(expect, li)
        ranks[empty[name]] = L
        assert systems[name]["mrr"] == round(float(np.mean(1.0 / ranks)), 4)
    assert [(c["system"], c["minus"]) for c in contrasts] == [("words", "characters")]
    assert lms.reading_of(contrasts, scores).startswith("partial")


def test_masked_table_contract_and_readiness(tmp_path=None):
    """The private contract carries the per-label table and the masked-text digest; the public
    summary carries no label string; the readiness check refuses a missing, stale or failed
    embedding run and accepts a matching one."""
    import json
    import tempfile
    store = Path(tempfile.mkdtemp(prefix="lms_test_"))
    try:
        labels = ["MC Alpha", "Beta", "小龙", "龙人", "X"]
        rows = [{"song_id": f"s{i}", "chunk_id": i, "source_order": i, "cleaned_text": t, "source_credit_label": labels[i % 3]}
                for i, t in enumerate(["MC Alpha and Beta with 小龙人", "nothing here", "Beta Beta 龙人"])]
        chunk_rows = [0, 1, 2]
        masked, counts, tally = lms.mask_all_labels([r["cleaned_text"] for r in rows], labels, [0, 1, 1])
        summary = lms.masking_summary(counts, tally, 3, [len(l) for l in labels])
        contract = lms.write_masked_table(store, rows, chunk_rows, masked, [1], labels, summary, counts_by_label=counts)
        on_disk = json.loads((store / "masked_table_contract.json").read_text(encoding="utf-8"))
        assert on_disk["masked_text_sha256"] == lms.text_digest(masked) == contract["masked_text_sha256"]
        assert on_disk["masked_occurrences_by_label_PRIVATE"] == {l: int(c) for l, c in zip(labels, counts)}
        assert on_disk["chunks"] == 3 and on_disk["check_corpus_rows"] == [1]
        # the public summary names no label
        text = json.dumps(summary, ensure_ascii=False)
        assert all(l not in text for l in labels if len(l) >= 2)
        # readiness: nothing yet
        vectors, why = lms.masked_vectors_if_ready(store, contract["masked_text_sha256"], 3)
        assert vectors is None and why.startswith("pending")
        # a run on another masked text is stale
        good = {"corpus_content_sha256": lms.V3_CONTENT_SHA256, "masked_text_sha256": contract["masked_text_sha256"],
                "check_unmasked_chunks": {"median_cosine": 0.9995}}
        np.save(store / "masked_chunk_bge_m3.npy", np.zeros((3, 1024), dtype=np.float32))
        (store / "masked_embedding_contract.json").write_text(json.dumps({**good, "masked_text_sha256": "0" * 64}), encoding="utf-8")
        vectors, why = lms.masked_vectors_if_ready(store, contract["masked_text_sha256"], 3)
        assert vectors is None and why.startswith("stale")
        # a failed reproduction check is refused
        (store / "masked_embedding_contract.json").write_text(
            json.dumps({**good, "check_unmasked_chunks": {"median_cosine": 0.99}}), encoding="utf-8")
        vectors, why = lms.masked_vectors_if_ready(store, contract["masked_text_sha256"], 3)
        assert vectors is None and why.startswith("failed")
        # a matching run is accepted, a wrong shape is not
        (store / "masked_embedding_contract.json").write_text(json.dumps(good), encoding="utf-8")
        vectors, info = lms.masked_vectors_if_ready(store, contract["masked_text_sha256"], 3)
        assert vectors is not None and vectors.shape == (3, 1024) and info["masked_text_sha256"] == contract["masked_text_sha256"]
        vectors, why = lms.masked_vectors_if_ready(store, contract["masked_text_sha256"], 4)
        assert vectors is None and why.startswith("stale")
    finally:
        import shutil
        shutil.rmtree(store, ignore_errors=True)


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"pass  {test.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL  {test.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(tests) - failed} of {len(tests)} tests passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
