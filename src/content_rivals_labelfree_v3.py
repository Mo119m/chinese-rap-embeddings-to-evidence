#!/usr/bin/env python3
"""Does the content-rival result survive a rival definition that is label-free and
independent of the compared spaces?

content_rival_test_v3 restricts each query's candidates to its true label plus the K labels
closest to it in content and reads which space still names the label. Its rivals are the K
labels whose leave-group-out profiles score highest for the query in the frozen semantic
space. Two objections. The definition is not label-free: the rivals are label profiles, so
the frozen space is judged on the labels it finds hardest by construction and its own row
is a floor, not a test. And it is not independent of the headline contrast: the whitened
semantic space is a linear map of the same vectors, so "words minus whitened semantic among
rivals" is measured on a candidate set that one side of the contrast helped to choose.

Three rival definitions, the published one reproduced as the check and two that meet the
objections. Under every definition the candidate set is the true label plus its K rivals,
K in (5, 10, 25), the same set for every space:

  semantic_space   the published definition: the K labels whose leave-group-out profiles
                   score highest in the frozen BGE-M3 space, true label excluded
  nmf_topics       label-free. Per fold a 50-topic NMF (Frobenius loss, nndsvda start,
                   coordinate descent) is fitted on the training-fold songs' jieba word
                   counts, sublinear tf-idf weighted with l2 rows (the standard input of a
                   Frobenius NMF); vocabulary (min_df 3, at most 50,000 words), idf and
                   topics are fitted on the training folds only, and no label enters the
                   model. Every song is given a unit topic vector; a query's rivals are the
                   K labels whose leave-group-out weighted mean topic vector is nearest by
                   cosine (the protocol's dense scorer on unit topic rows), true label
                   excluded. Labels enter only after the topic space is fixed.
  character_space  the K labels whose leave-group-out character 2-5-gram profiles score
                   highest for the query, true label excluded. Not label-free, but
                   independent of every space it judges: the character space is excluded
                   from the comparison under this definition.

Spaces, all leave-group-out under the unchanged protocol: frozen semantic (BGE-M3 song
mean, cosine); the same after fold-wise within-author whitening (identity_probe_v2);
character 2-5-gram TF-IDF; jieba word 1-2-gram TF-IDF; the two z-score fusions of
content_rival_test_v3; and a text-distortion arm (Stamatatos 2018, DV-SA): per fold the
500 words of highest document frequency over the training-fold songs are kept, every other
word becomes one placeholder token, the word TF-IDF space is rebuilt on the distorted
documents of all songs (the protocol's transductive fit), and the fold's queries are scored
in that rebuild under the protocol. The arm asks how much of the word space's rival result
the frequent words carry.

Estimands. Per definition, K and space: MRR and top-1 accuracy of the true label within
the candidate set (rank by content_rival_test_v3.rank_within; ties count against the
query); over all 226 labels: the protocol MRR and top-1 of every space, the distortion arm
included. Every level and every contrast between spaces carries a paired group bootstrap
(identity_spaces_v2.paired_group_bootstrap: 2000 replicates, seed 20260825, leakage groups
resampled with replacement, each (group, label) component weighted one), for MRR and for
top-1; a level's interval is its contrast against an all-zero system, so the same
replicates serve levels and contrasts. Mean Jaccard overlap between the definitions' rival
sets is reported as description.

Checks before any new number is used:
  1. the published content_rival_test.json is reproduced under its own design: the
     all-labels MRR of every published space, the MRR and top-1 among 5, 10 and 25
     semantic rivals of every published space, and the published paired contrasts (point
     and both interval bounds), each within 0.002 of the file and of the literals pinned
     in PINNED (SystemExit otherwise);
  2. rank_within agrees with a brute-force rank on the first 200 queries of every space
     under every definition and K (SystemExit otherwise);
  3. every rival set excludes the true label and holds K distinct labels, and no rival
     definition is asked for more rivals than there are other labels;
  4. the topic model, vocabulary and idf of fold k are fitted on rows disjoint from that
     fold's queries (asserted by index), and every song receives a nonzero topic vector
     (a zero vector would tie every label and make the rivals arbitrary; SystemExit);
  5. the kept-word list of the distortion arm of fold k is counted over the training-fold
     songs only, keeps at most 500 distinct words, and the placeholder is not a word of the
     corpus; the kept share of word tokens is reported.

READING RULE, fixed before the run. The word space's advantage among content rivals is
"not an artefact of the rival definition" if lexical_words minus
semantic_within_author_whitening among rivals is positive with the 95% interval clear of
zero under BOTH label-free definitions (nmf_topics and character_space) at every K in
(5, 10, 25); otherwise it is "definition-dependent" and the failing (definition, K) cells
are named. Top-1 contrasts are reported alongside as description, not as the rule. The
distortion arm is described, not read: its numbers say how much of the word space's rival
result survives with 500 words, under each definition.

Note added 2026-09-21, after the run and outside the rule. The nmf_topics definition is
label-free but it is built from word counts, and the character definition from the character
surface: both choose rivals by lexical similarity, as the published definition chooses them by
semantic similarity. A rival test handicaps the family of evidence that defines its rivals, so
no definition available here is neutral between the lexical and the semantic family; the
results under the three definitions are to be read together.

    set CHINESE_RAP_CORPUS=v3
    python src/content_rivals_labelfree_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from content_rival_test_v3 import RIVALS, rank_within  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from exemplar_vs_prototype_v3 import setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "content_rivals_labelfree.json"
PUBLISHED = OUT_DIR / "content_rival_test.json"

TOPICS = 50
TOPIC_MIN_DF = 3
TOPIC_MAX_FEATURES = 50_000
NMF_MAX_ITER = 300
DISTORTION_WORDS = 500
PLACEHOLDER = "<oov>"
CHECK_GAP = 0.002
BRUTE_FORCE_QUERIES = 200
ZERO = "__zero__"

SEMANTIC, WHITENED = "semantic", "semantic_within_author_whitening"
CHARS, WORDS = "lexical_char_2_5", "lexical_words"
FUSION_RAW, FUSION_WHITENED = "fusion_semantic_words", "fusion_whitened_semantic_words"
DISTORTED = "lexical_words_distorted_500"
PUBLISHED_SPACES = (SEMANTIC, WHITENED, CHARS, WORDS, FUSION_RAW, FUSION_WHITENED)
SPACES = PUBLISHED_SPACES + (DISTORTED,)
DEFINITIONS = ("semantic_space", "nmf_topics", "character_space")
LABEL_FREE = ("nmf_topics", "character_space")
# the space a definition is built from: its row is the floor, not a test
DEFINES = {"semantic_space": (SEMANTIC,), "nmf_topics": (), "character_space": (CHARS,)}
# the published design keeps the frozen space in its contrasts; the character definition
# excludes the character space from every comparison it judges
EXCLUDED_FROM_CONTRASTS = {"semantic_space": (), "nmf_topics": (), "character_space": (CHARS,)}
HEADLINE = (WORDS, WHITENED)
PUBLISHED_PAIRS = ((WORDS, SEMANTIC), (WORDS, CHARS), (WHITENED, SEMANTIC), (WORDS, WHITENED), (FUSION_WHITENED, WORDS))
PAIRS = PUBLISHED_PAIRS + ((DISTORTED, WORDS), (DISTORTED, WHITENED), (DISTORTED, CHARS), (DISTORTED, SEMANTIC))
ALL_LABEL_PAIRS = ((WORDS, WHITENED), (DISTORTED, WORDS), (DISTORTED, WHITENED), (DISTORTED, CHARS), (DISTORTED, SEMANTIC))

# pinned literals of the published v3 result (content_rival_test.json; the all-labels row is
# also three_spaces.json / identity_probe.json). The file is read as well; both must reproduce.
PINNED = {
    "all_labels": {SEMANTIC: 0.2997, WHITENED: 0.4164, CHARS: 0.4266, WORDS: 0.4963,
                   FUSION_RAW: 0.4894, FUSION_WHITENED: 0.5371},
    "among": {5: {WORDS: (0.6765, 0.5298), WHITENED: (0.5897, 0.414), CHARS: (0.6308, 0.4657),
                  SEMANTIC: (0.3716, 0.2019), FUSION_RAW: (0.5789, 0.4227), FUSION_WHITENED: (0.6571, 0.5032)},
              10: {WORDS: (0.6142, 0.4834), WHITENED: (0.5305, 0.3795), CHARS: (0.5593, 0.4168),
                   SEMANTIC: (0.3296, 0.2019), FUSION_RAW: (0.5332, 0.4065), FUSION_WHITENED: (0.61, 0.4763)},
              25: {WORDS: (0.553, 0.442), WHITENED: (0.4755, 0.3485), CHARS: (0.4898, 0.3716),
                   SEMANTIC: (0.3069, 0.2019), FUSION_RAW: (0.502, 0.3964), FUSION_WHITENED: (0.5677, 0.4508)}},
    # lexical_words - semantic_within_author_whitening among rivals: point, low, high
    "headline": {5: (0.0948, 0.0856, 0.1043), 10: (0.092, 0.0822, 0.102), 25: (0.086, 0.076, 0.0962)},
}


def say(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------------------------ rival sets
def rivals_from_scores(scores: np.ndarray, truth: np.ndarray, k: int, stable: bool = True) -> np.ndarray:
    """The k labels with the highest score for each query, the true label excluded.
    stable=False is the published definition's own argsort call, kept for reproduction.
    Refuses k larger than the number of other labels: the true label, sorted last, would
    otherwise be handed back as a rival."""
    if k > scores.shape[1] - 1:
        raise SystemExit(f"{k} rivals asked of {scores.shape[1]} labels; at most {scores.shape[1] - 1} exist")
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[np.arange(len(truth)), truth] = -np.inf
    order = np.argsort(-masked, axis=1, kind="stable") if stable else np.argsort(-masked, axis=1)
    return order[:, :k]


def candidate_sets(truth: np.ndarray, rivals: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([truth[:, None], rivals[:, :k]], axis=1)


def check_rival_sets(rivals: np.ndarray, truth: np.ndarray, k: int) -> None:
    if rivals.shape[1] < k:
        raise SystemExit(f"rival sets hold {rivals.shape[1]} labels, fewer than K={k}")
    block = rivals[:, :k]
    if np.any(block == truth[:, None]):
        raise SystemExit("a rival set contains the true label")
    ordered = np.sort(block, axis=1)
    if k > 1 and np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise SystemExit("a rival set repeats a label")


def brute_force_rank_within(scores: np.ndarray, truth: np.ndarray, candidates: np.ndarray, q: int) -> int:
    """Rank of the true label among the candidates by the plain definition: one plus the
    number of other candidates scoring at least as high (ties count against the query)."""
    own = scores[q, truth[q]]
    rank = 1
    for c in candidates[q].tolist():
        if c != truth[q] and scores[q, c] >= own:
            rank += 1
    return rank


def among_rivals(scores_by_space: dict, truth: np.ndarray, rivals: np.ndarray, k: int, spaces,
                 check: int = 0) -> dict:
    """Rank of the true label within (true label + k rivals), per space; the vectorised
    protocol scorer is checked against the brute-force definition on the first `check` queries."""
    candidates = candidate_sets(truth, rivals, k)
    out = {}
    for name in spaces:
        ranks = rank_within(scores_by_space[name], truth, candidates)
        for q in range(min(check, len(truth))):
            slow = brute_force_rank_within(scores_by_space[name], truth, candidates, q)
            if int(ranks[q]) != slow:
                raise SystemExit(f"rank_within differs from its definition in {name} at query {q}: {ranks[q]} vs {slow}")
        out[name] = ranks.astype(np.int64)
    return out


def chance(k: int) -> dict:
    """A random permutation of K+1 candidates: MRR H(K+1)/(K+1), top-1 1/(K+1)."""
    return {"mrr": round(float(np.sum(1.0 / np.arange(1, k + 2)) / (k + 1)), 4), "top1": round(1.0 / (k + 1), 4)}


def overlap(rivals_a: np.ndarray, rivals_b: np.ndarray, k: int) -> float:
    """Mean Jaccard overlap of two definitions' k-rival sets over the queries."""
    a, b = rivals_a[:, :k], rivals_b[:, :k]
    shared = np.asarray([len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)], dtype=np.float64)
    return float(np.mean(shared / (2 * k - shared)))


# ------------------------------------------------------------------ text distortion
def distort(tokens_by_song, train_mask: np.ndarray, k: int, placeholder: str = PLACEHOLDER):
    """Stamatatos (2018) DV-SA on word tokens: keep the k words of highest document
    frequency over the training-fold songs (ties by string), replace every other word by
    one placeholder token. Tokens are lowercased as the protocol's vectorizer lowercases.
    Returns the distorted documents of ALL songs, the kept set, and the kept token share."""
    frequency: Counter = Counter()
    for i in np.flatnonzero(train_mask).tolist():
        frequency.update({t.lower() for t in tokens_by_song[i]})
    keep = {t for t, _ in sorted(frequency.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}
    documents, kept, total = [], 0, 0
    for tokens in tokens_by_song:
        low = [t.lower() for t in tokens]
        kept += sum(1 for t in low if t in keep)
        total += len(low)
        documents.append(" ".join(t if t in keep else placeholder for t in low))
    return documents, keep, kept / max(total, 1)


def distortion_arm(tokens_by_song, fold: np.ndarray, dense32: np.ndarray, label_index, group_ids, label_count: int,
                   k: int = DISTORTION_WORDS, placeholder: str = PLACEHOLDER, log=say):
    """Fold-wise DV-SA word space scored under the protocol: a fold's kept words are counted
    on its training songs; the TF-IDF space is rebuilt on the distorted documents of all
    songs and the fold's queries take their scores from that rebuild."""
    if any(t.lower() == placeholder for song in tokens_by_song for t in song):
        raise SystemExit("the placeholder token occurs in the corpus")
    scores = np.zeros((len(label_index), label_count))
    info = {}
    for f in range(FOLDS):
        train = fold != f
        test = np.flatnonzero(~train)
        docs_f, keep, kept_share = distort(tokens_by_song, train, k, placeholder)
        if len(keep) > k:
            raise SystemExit("the distortion kept more words than allowed")
        matrix_f, _ = fit_words(docs_f)
        scores[test] = v1.score_leave_group_out(dense32, matrix_f, label_index, group_ids, label_count).lexical.astype(np.float64)[test]
        info[str(f)] = {"words_kept": int(len(keep)), "kept_token_share": round(float(kept_share), 4),
                        "tfidf_features": int(matrix_f.shape[1]), "queries": int(len(test))}
        log(f"  fold {f}: {len(keep)} words kept, {kept_share:.3f} of word tokens kept, {matrix_f.shape[1]:,} tf-idf features")
    return scores, info


# ------------------------------------------------------------------ topic rivals
def topic_scores_fold(joined_docs, train_mask: np.ndarray, queries: np.ndarray, label_index, group_ids, weights,
                      label_count: int, seed: int, topics: int = TOPICS, max_iter: int = NMF_MAX_ITER):
    """Fold-wise, label-free topic model. Vocabulary, idf and the NMF are fitted on the
    training-fold rows only; every song is then given a topic vector, and the queries are
    scored by cosine with every label's leave-group-out weighted mean topic vector (the
    protocol's dense scorer on unit topic rows). A song with a zero topic vector would tie
    every label, so it stops the run. Returns (scores, topic matrix, diagnostics)."""
    train_rows = np.flatnonzero(train_mask)
    if np.intersect1d(train_rows, queries).size:
        raise SystemExit("a topic model would be fitted on its own queries")
    counter = CountVectorizer(analyzer="word", token_pattern=r"(?u)\S+", min_df=TOPIC_MIN_DF,
                              max_features=TOPIC_MAX_FEATURES, dtype=np.float64)
    counter.fit([joined_docs[i] for i in train_rows.tolist()])
    counts = counter.transform(joined_docs).tocsr()
    weighting = TfidfTransformer(sublinear_tf=True, norm="l2").fit(counts[train_rows])
    x = weighting.transform(counts).tocsr()
    model = NMF(n_components=topics, init="nndsvda", random_state=seed, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x[train_rows])
        topic_matrix = model.transform(x)
    zero_rows = int(np.sum(topic_matrix.sum(axis=1) <= 0))
    if zero_rows:
        raise SystemExit(f"{zero_rows} songs have a zero topic vector; their rivals would be arbitrary")
    scores = dense_leave_group_out(unit_rows(topic_matrix), label_index, group_ids, weights, label_count, queries)
    info = {"vocabulary": int(x.shape[1]), "iterations": int(model.n_iter_), "converged": bool(model.n_iter_ < max_iter),
            "reconstruction_error": round(float(model.reconstruction_err_), 4), "zero_topic_rows": zero_rows,
            "training_songs": int(len(train_rows)), "queries": int(len(queries))}
    return scores, topic_matrix, info


def topic_rivals(joined_docs, fold: np.ndarray, label_index, group_ids, weights, label_count: int,
                 topics: int = TOPICS, max_iter: int = NMF_MAX_ITER, k: int = max(RIVALS), log=say):
    """Topic scores of every query from its own fold's model; the k rivals by stable argsort."""
    scores = np.zeros((len(label_index), label_count))
    info = {}
    for f in range(FOLDS):
        train = fold != f
        test = np.flatnonzero(~train)
        scores[test], _, info[str(f)] = topic_scores_fold(joined_docs, train, test, label_index, group_ids, weights,
                                                          label_count, SEED + f, topics, max_iter)
        e = info[str(f)]
        log(f"  fold {f}: vocabulary {e['vocabulary']:,}, {e['iterations']} iterations, error {e['reconstruction_error']:.3f}, "
            f"zero topic rows {e['zero_topic_rows']}")
    return rivals_from_scores(scores, label_index, k), info


# ------------------------------------------------------------------ the published spaces
def published_design_scores(d: dict, docs, joined, log=say) -> dict:
    """The six spaces of content_rival_test_v3, built exactly as that file builds them."""
    li, gi, w, L, fold, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"], d["dense"]
    n = len(li)
    dense32 = dense.astype(np.float32)
    scores: dict[str, np.ndarray] = {}
    char_matrix = v1.fit_tfidf(docs)
    profiles = v1.score_leave_group_out(dense32, char_matrix, li, gi, L)
    scores[SEMANTIC] = profiles.dense.astype(np.float64)
    scores[CHARS] = profiles.lexical.astype(np.float64)
    log(f"  character space: {char_matrix.shape[1]:,} features")
    word_matrix, _ = fit_words(joined)
    scores[WORDS] = v1.score_leave_group_out(dense32, word_matrix, li, gi, L).lexical.astype(np.float64)
    log(f"  word space: {word_matrix.shape[1]:,} features")
    whitened = np.zeros((n, L))
    for k in range(FOLDS):
        train = fold != k
        queries = np.flatnonzero(~train)
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train], li[train], w[train],
                                        np.random.default_rng(SEED + k))
        whitened[queries] = dense_leave_group_out(unit_rows((dense - mean) @ matrix.T), li, gi, w, L, queries)
    scores[WHITENED] = whitened
    scores[FUSION_WHITENED] = (v1.zscore_rows(whitened) + v1.zscore_rows(scores[WORDS])) / 2.0
    scores[FUSION_RAW] = (v1.zscore_rows(scores[SEMANTIC]) + v1.zscore_rows(scores[WORDS])) / 2.0
    return scores


# ------------------------------------------------------------------ reproduction check
def reproduce_published(scores: dict, label_index, group_ids, weights, published: dict, pinned=None,
                        gap: float = CHECK_GAP, check: int = BRUTE_FORCE_QUERIES, log=say):
    """Check 1: the published file's numbers under its own design, all within `gap`.
    Returns the record and the published rival order (the file's own argsort call)."""
    if published["corpus"]["content_sha256"] != V3_CONTENT_SHA256:
        raise SystemExit("content_rival_test.json was computed on a different corpus build")
    rivals = rivals_from_scores(scores[SEMANTIC], label_index, max(RIVALS), stable=False)
    all_mask = np.ones(len(label_index), dtype=bool)
    record = {"all_labels": {}, "among_rivals": {}, "paired_contrasts": {}}
    worst = 0.0
    for name in PUBLISHED_SPACES:
        got = float(np.mean(1.0 / ranks_of(scores[name], label_index)))
        expected_file = float(published["results"]["all_labels"][name])
        g = abs(got - expected_file)
        if pinned:
            g = max(g, abs(got - pinned["all_labels"][name]))
        worst = max(worst, g)
        record["all_labels"][name] = {"expected": expected_file, "recomputed": round(got, 4), "gap": round(g, 4)}
        log(f"  check all_labels {name:32s} recomputed {got:.4f} expected {expected_file:.4f} gap {g:.4f}")
    for k in RIVALS:
        check_rival_sets(rivals, label_index, k)
        ranks = among_rivals(scores, label_index, rivals, k, PUBLISHED_SPACES, check=check)
        rr = {name: 1.0 / r for name, r in ranks.items()}
        entry = {}
        for name, r in ranks.items():
            got_mrr, got_top1 = float(np.mean(1.0 / r)), float(np.mean(r == 1))
            expected_file = published["results"][f"among_{k}_content_rivals"][name]
            g = max(abs(got_mrr - expected_file["mrr"]), abs(got_top1 - expected_file["top1"]))
            if pinned:
                g = max(g, abs(got_mrr - pinned["among"][k][name][0]), abs(got_top1 - pinned["among"][k][name][1]))
            worst = max(worst, g)
            entry[name] = {"expected_mrr": expected_file["mrr"], "expected_top1": expected_file["top1"],
                           "recomputed_mrr": round(got_mrr, 4), "recomputed_top1": round(got_top1, 4), "gap": round(g, 4)}
            log(f"  check among_{k:<2d} {name:32s} mrr {got_mrr:.4f}/{expected_file['mrr']:.4f} "
                f"top1 {got_top1:.4f}/{expected_file['top1']:.4f} gap {g:.4f}")
        record["among_rivals"][str(k)] = entry
        contrasts = paired_group_bootstrap(rr, weights, group_ids, all_mask, list(PUBLISHED_PAIRS))
        on_file = {(c["system"], c["minus"]): c for c in published["paired_contrasts"]["by_rival_count"][f"among_{k}_content_rivals"]}
        rows = []
        for c in contrasts:
            f = on_file[(c["system"], c["minus"])]
            g = max(abs(c["mrr_difference"] - f["mrr_difference"]), abs(c["ci95"][0] - f["ci95"][0]), abs(c["ci95"][1] - f["ci95"][1]))
            if pinned and (c["system"], c["minus"]) == HEADLINE:
                p = pinned["headline"][k]
                g = max(g, abs(c["mrr_difference"] - p[0]), abs(c["ci95"][0] - p[1]), abs(c["ci95"][1] - p[2]))
            worst = max(worst, g)
            rows.append({"system": c["system"], "minus": c["minus"], "expected": f["mrr_difference"], "expected_ci95": f["ci95"],
                         "recomputed": c["mrr_difference"], "recomputed_ci95": c["ci95"], "gap": round(g, 4)})
            log(f"  check among_{k:<2d} {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] "
                f"expected {f['mrr_difference']:+.4f} [{f['ci95'][0]:+.4f}, {f['ci95'][1]:+.4f}] gap {g:.4f}")
        record["paired_contrasts"][str(k)] = rows
    record["worst_gap"] = round(worst, 4)
    record["limit"] = gap
    log(f"  worst gap {worst:.4f} (limit {gap})")
    if worst > gap:
        raise SystemExit(f"the published content-rival numbers do not reproduce (worst gap {worst:.4f})")
    return record, rivals


# ------------------------------------------------------------------ tables and reading
def levels_and_contrasts(values: dict, weights: np.ndarray, group_ids: np.ndarray, pairs):
    """Component-weighted level of every system with its interval (the contrast against an
    all-zero system, so the same replicates serve levels and contrasts), and the paired
    contrasts between systems."""
    systems = dict(values)
    systems[ZERO] = np.zeros(len(weights))
    mask = np.ones(len(weights), dtype=bool)
    names = list(values)
    out = paired_group_bootstrap(systems, weights, group_ids, mask, [(name, ZERO) for name in names] + list(pairs))
    levels = {c["system"]: {"component_weighted": c["mrr_difference"], "ci95": c["ci95"]} for c in out[:len(names)]}
    return levels, out[len(names):]


def rename_top1(contrasts):
    return [{("top1_difference" if key == "mrr_difference" else key): value for key, value in c.items()} for c in contrasts]


def rival_tables(scores: dict, truth: np.ndarray, group_ids: np.ndarray, weights: np.ndarray, rivals: np.ndarray,
                 excluded=(), defines=(), spaces=SPACES, pairs=PAIRS, ks=RIVALS, check: int = BRUTE_FORCE_QUERIES) -> dict:
    """Per K: MRR and top-1 among rivals for every space, each with its interval, and paired
    group-bootstrap contrasts (MRR and top-1) between the judged spaces."""
    judged = [s for s in spaces if s not in excluded]
    live = [(a, b) for a, b in pairs if a in judged and b in judged]
    by_k = {}
    for k in ks:
        check_rival_sets(rivals, truth, k)
        ranks = among_rivals(scores, truth, rivals, k, spaces, check=check)
        rr = {name: 1.0 / r for name, r in ranks.items()}
        top1 = {name: (r == 1).astype(np.float64) for name, r in ranks.items()}
        mrr_levels, mrr_contrasts = levels_and_contrasts(rr, weights, group_ids, live)
        top1_levels, top1_contrasts = levels_and_contrasts(top1, weights, group_ids, live)
        table = {name: {"mrr": round(float(np.mean(rr[name])), 4), "top1": round(float(np.mean(top1[name])), 4),
                        "mrr_component_weighted": mrr_levels[name]["component_weighted"], "mrr_ci95": mrr_levels[name]["ci95"],
                        "top1_component_weighted": top1_levels[name]["component_weighted"], "top1_ci95": top1_levels[name]["ci95"],
                        "judged": name in judged, "defines_the_rivals": name in defines} for name in spaces}
        by_k[str(k)] = {"chance": chance(k), "spaces": table, "mrr_contrasts": mrr_contrasts,
                        "top1_contrasts": rename_top1(top1_contrasts)}
    return {"defines_the_rivals": list(defines), "excluded_from_contrasts": list(excluded), "by_k": by_k}


def reading(by_definition: dict, definitions=LABEL_FREE, pair=HEADLINE, ks=RIVALS) -> dict:
    """The rule of the docstring: positive with the interval clear of zero under every
    label-free definition at every K, else definition-dependent with the failing cells named."""
    failing, cells = [], {}
    for definition in definitions:
        for k in ks:
            contrast = next(c for c in by_definition[definition]["by_k"][str(k)]["mrr_contrasts"]
                            if (c["system"], c["minus"]) == pair)
            cells[f"{definition}@K={k}"] = contrast
            if not (contrast["mrr_difference"] > 0 and contrast["ci95"][0] > 0):
                failing.append(f"{definition}@K={k}")
    verdict = "not an artefact of the rival definition" if not failing else "definition-dependent"
    return {"contrast": f"{pair[0]} - {pair[1]}", "verdict": verdict, "failing_cells": failing, "cells": cells}


# ------------------------------------------------------------------ the analysis
def analyse(d: dict, docs, published: dict, *, tokenize=segment, topics: int = TOPICS, nmf_max_iter: int = NMF_MAX_ITER,
            distortion_words: int = DISTORTION_WORDS, check: int = BRUTE_FORCE_QUERIES, pinned=None, log=say) -> dict:
    """Everything after setup, on any corpus of the protocol's shape; returns the aggregate payload."""
    started = time.time()
    li, gi, w, L, fold, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"], d["dense"]
    n = len(li)
    dense32 = dense.astype(np.float32)
    log(f"{n:,} queries, {L} labels, {len(np.unique(gi)):,} groups; fold sizes {np.bincount(fold, minlength=FOLDS).tolist()}")

    log("scoring the published spaces")
    tokens = [tokenize(doc) for doc in docs]
    joined = [" ".join(t) for t in tokens]
    scores = published_design_scores(d, docs, joined, log=log)

    log("check: reproducing content_rival_test.json under its own design")
    reproduction, semantic_rivals = reproduce_published(scores, li, gi, w, published, pinned=pinned, check=check, log=log)
    rivals = {"semantic_space": semantic_rivals}

    log(f"distortion arm: {distortion_words} words of highest training-fold document frequency")
    scores[DISTORTED], distortion_info = distortion_arm(tokens, fold, dense32, li, gi, L, distortion_words, log=log)

    log(f"rival definition nmf_topics: {topics} topics per fold")
    rivals["nmf_topics"], topic_info = topic_rivals(joined, fold, li, gi, w, L, topics, nmf_max_iter, log=log)
    rivals["character_space"] = rivals_from_scores(scores[CHARS], li, max(RIVALS))
    overlaps = {f"{a}_vs_{b}": {str(k): round(overlap(rivals[a], rivals[b], k), 4) for k in RIVALS}
                for a, b in combinations(DEFINITIONS, 2)}
    for name, values in overlaps.items():
        log(f"  rival-set overlap {name}: " + ", ".join(f"K={k} {v:.3f}" for k, v in values.items()))

    log("all labels, every space")
    all_rr = {name: 1.0 / ranks_of(scores[name], li) for name in SPACES}
    all_top1 = {name: (rr == 1.0).astype(np.float64) for name, rr in all_rr.items()}
    mrr_levels, mrr_contrasts = levels_and_contrasts(all_rr, w, gi, ALL_LABEL_PAIRS)
    top1_levels, top1_contrasts = levels_and_contrasts(all_top1, w, gi, ALL_LABEL_PAIRS)
    all_labels = {name: {"mrr": round(float(np.mean(all_rr[name])), 4), "top1": round(float(np.mean(all_top1[name])), 4),
                         "mrr_component_weighted": mrr_levels[name]["component_weighted"], "mrr_ci95": mrr_levels[name]["ci95"],
                         "top1_component_weighted": top1_levels[name]["component_weighted"], "top1_ci95": top1_levels[name]["ci95"]}
                  for name in SPACES}
    log("  " + "  ".join(f"{name}={entry['mrr']:.4f}" for name, entry in all_labels.items()))
    for c in mrr_contrasts:
        log(f"  {c['system']} - {c['minus']}: MRR {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")

    by_definition = {}
    for definition in DEFINITIONS:
        log(f"== rivals by {definition}")
        by_definition[definition] = rival_tables(scores, li, gi, w, rivals[definition],
                                                 excluded=EXCLUDED_FROM_CONTRASTS[definition], defines=DEFINES[definition], check=check)
        for k, entry in by_definition[definition]["by_k"].items():
            log(f"  K={k} (chance mrr {entry['chance']['mrr']:.3f}, top-1 {entry['chance']['top1']:.3f})")
            for name, cell in entry["spaces"].items():
                flag = "   (defines the rivals; floor, not a test)" if cell["defines_the_rivals"] else ""
                log(f"    {name:32s} MRR {cell['mrr']:.4f} [{cell['mrr_ci95'][0]:.4f}, {cell['mrr_ci95'][1]:.4f}]  "
                    f"top-1 {cell['top1']:.4f} [{cell['top1_ci95'][0]:.4f}, {cell['top1_ci95'][1]:.4f}]{flag}")
            for c in entry["mrr_contrasts"]:
                log(f"    {c['system']} - {c['minus']}: MRR {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
            for c in entry["top1_contrasts"]:
                log(f"    {c['system']} - {c['minus']}: top-1 {c['top1_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    verdict = reading(by_definition)
    log(f"reading: {verdict['contrast']} among rivals is {verdict['verdict']}"
        + (f"; failing {verdict['failing_cells']}" if verdict["failing_cells"] else ""))

    return {
        "analysis": "the content-rival test under label-free rival definitions, with a text-distortion arm",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L, "groups": int(len(np.unique(gi)))},
        "design": {
            "candidate_set": "the true label plus its K rivals; rank of the true label within the set, ties counted "
                             "against the query (content_rival_test_v3.rank_within)",
            "rival_definitions": {
                "semantic_space": "the published definition: K labels whose leave-group-out profiles score highest in "
                                  "the frozen BGE-M3 space, true label excluded",
                "nmf_topics": f"label-free: per fold a {topics}-topic NMF (Frobenius, nndsvda, coordinate descent, at most "
                              f"{nmf_max_iter} iterations) on jieba word counts, sublinear tf-idf weighted with l2 rows; "
                              f"vocabulary (min_df {TOPIC_MIN_DF}, at most {TOPIC_MAX_FEATURES:,} words), idf and topics "
                              "fitted on the training-fold songs only; rivals are the K labels whose leave-group-out "
                              "weighted mean topic vector is nearest by cosine on unit topic rows, true label excluded",
                "character_space": "K labels whose leave-group-out character 2-5-gram profiles score highest; the "
                                   "character space is excluded from every contrast under this definition",
            },
            "spaces": {SEMANTIC: "frozen BGE-M3 song mean, cosine",
                       WHITENED: "fold-wise within-author whitening as in identity_probe_v2",
                       CHARS: "character 2-5-gram TF-IDF", WORDS: "jieba word 1-2-gram TF-IDF",
                       FUSION_RAW: "z-score mean of semantic and words", FUSION_WHITENED: "z-score mean of whitened semantic and words",
                       DISTORTED: f"Stamatatos 2018 DV-SA: per fold the {distortion_words} words of highest "
                                  "training-fold document frequency kept, every other word one placeholder token; "
                                  "word 1-2-gram TF-IDF rebuilt on the distorted documents of all songs, the fold's "
                                  "queries scored from that rebuild under the protocol"},
            "folds": f"{FOLDS} folds of leakage groups, seed {SEED}, as in identity_probe_v2",
            "metrics": "MRR and top-1 accuracy among rivals; protocol MRR and top-1 over all labels; levels are plain "
                       "means over queries, component-weighted levels and all contrasts carry the bootstrap",
            "bootstrap": "paired group bootstrap, 2000 replicates, seed 20260825, leakage groups resampled with "
                         "replacement, each (group, label) component weighted one; a level's interval is its contrast "
                         "against an all-zero system",
            "checks": "published content_rival_test.json reproduced under its own design (levels, top-1 and paired "
                      f"contrasts with intervals, gap <= {CHECK_GAP}, file and pinned literals); rank_within equals its "
                      f"brute-force definition on {check} queries per space, definition and K; rival sets exclude the "
                      "true label and hold K distinct labels; fold-k topic model, vocabulary and idf fitted on rows "
                      "disjoint from fold-k queries and no song has a zero topic vector; the distortion keeps at most "
                      f"{distortion_words} words counted on training-fold songs and its placeholder is not a corpus word",
            "reading_rule": "the word space's advantage among content rivals is 'not an artefact of the rival "
                            "definition' if lexical_words - semantic_within_author_whitening is positive with the 95% "
                            "interval clear of zero under both nmf_topics and character_space at every K in "
                            f"{list(RIVALS)}; otherwise 'definition-dependent' with the failing cells named; top-1 and "
                            "the distortion arm are description",
            "reference": "Wegmann, Schraagen and Nguyen 2022 (style-or-content choice); Stamatatos 2018 (text distortion)",
        },
        "checks": {"reproduction_of_content_rival_test": reproduction, "brute_force_queries_per_table": check,
                   "topic_model_fitted_on_rows_disjoint_from_queries": True, "no_zero_topic_rows": True,
                   "placeholder_absent_from_corpus": True},
        "distortion_by_fold": distortion_info,
        "topic_model_by_fold": topic_info,
        "rival_set_overlap_mean_jaccard": overlaps,
        "all_labels": {"spaces": all_labels, "mrr_contrasts": mrr_contrasts, "top1_contrasts": rename_top1(top1_contrasts)},
        "among_rivals": by_definition,
        "reading": verdict,
        "privacy": "aggregate only; no lyric text, song identifier, word or per-song vector",
        "minutes": round((time.time() - started) / 60, 1),
    }


# ------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if os.environ.get("CHINESE_RAP_CORPUS") != "v3":
        raise SystemExit("set CHINESE_RAP_CORPUS=v3; this analysis is a corpus v3 result")
    if not PUBLISHED.is_file():
        raise SystemExit(f"missing {PUBLISHED}")
    published = json.loads(PUBLISHED.read_bytes().decode("utf-8"))
    started = time.time()
    import jieba
    jieba.setLogLevel(60)
    d = setup(args.private_root.resolve())
    docs = [d["documents"][s] for s in d["songs"]]
    payload = analyse(d, docs, published, pinned=PINNED)
    payload["minutes"] = round((time.time() - started) / 60, 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / OUT_NAME).write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    say(f"\nwrote {args.out_dir / OUT_NAME}  ({payload['minutes']} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
