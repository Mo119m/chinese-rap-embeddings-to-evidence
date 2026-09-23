#!/usr/bin/env python3
"""What the published width factor is made of, and whether the calibrations that raise MRR
also improve the evidence. Corpus v3, the 7,220 queries and 226 labels of every other arm here.

QUESTION 1: THE WIDTH FACTOR IS NOT IDENTIFIED. estimand_two_stage.json reports, for words
minus characters, +0.0711 [+0.0643, +0.0785] (width 0.0142) and +0.0551 [+0.0409, +0.0692]
(width 0.0283), and the README turns the ratio into a reusable "1.6 to 3.0 times". Those two
cells differ in BOTH the estimand -- component-weighted against label-macro -- AND the
resampling -- leakage groups against the builder's two-stage label-then-group -- so the ratio
is the product of two factors nobody separated. This file fills the 2x2

                              group bootstrap          two-stage label-then-group
      component-weighted      PUBLISHED intervals      new
      label-macro             new                      PUBLISHED intervals

for every contrast below, out of ONE component table, so the four cells are the same numbers
resampled two ways rather than four different analyses. The new diagonal is the point: under
the group bootstrap the 226 labels are held FIXED and only leakage groups are drawn, so a
label's mean is the mean of ITS drawn components and the macro is the plain mean of those label
means (a label left with no drawn component is dropped from that replicate, and the number
dropped is recorded). An estimand has one point estimate and a resampling only has an interval,
so the four cells carry TWO points and FOUR intervals, and the width factors are identified:

    estimand factor at a fixed resampling   W(label-macro) / W(component-weighted)
    resampling factor at a fixed estimand   W(two-stage)   / W(group)
    total                                   W(label-macro, two-stage) / W(component-weighted, group)

The total is the product along either path exactly; the log of the total is split between the
two factors by the symmetric (Shapley) average of the two paths, and how far the estimand
factor itself depends on which resampling it is measured under is reported as the interaction.

THE CONVENTION THE PUBLISHED CELLS FOLLOW. README.md's note of 2026-09-23 states it: a system's
`mrr` field is QUERY-WEIGHTED, while every interval from identity_spaces_v2.paired_group_bootstrap
-- and that function's own point estimate -- is COMPONENT-WEIGHTED, so two printed levels must
never be subtracted to reconstruct a published difference. The same note bounds the discrepancy
from the artifacts that already record both: at most 0.0057 on a level and 0.0076 on a difference
across the nine headline systems. Those bounds are CITED here, not rediscovered. In the 2x2 every
published paired_group_bootstrap interval is the (component-weighted, group bootstrap) cell and
every published estimand_two_stage interval is the (label-macro, two-stage) cell; the other two
cells have no published counterpart. The published LEVELS this file reproduces are query-weighted,
which is why no point estimate of one estimand is ever printed beside an interval of another.

QUESTION 2: EVERY METRIC HERE IS A RANK STATISTIC. MRR is computed within a query, so it is
invariant to any per-query monotone map of the scores; the paper's thesis, though, is that a
scorer is an ESTIMATOR of same-label evidence. A calibration that raised MRR while destroying
the likelihood ratio would be invisible to this directory, because nothing in it reads the
scores as a pooled detector. So each system is also read over all 1,631,720 (query, label)
trials, scored leave-group-out exactly as the retrieval protocol scores them (7,220 target,
1,624,500 non-target):

    auc               weighted probability that a target trial outranks a non-target one, ties
                      at one half; invariant to any GLOBAL monotone map
    cllr              the cost of log-likelihood-ratio, reading each score AS a natural-log
                      likelihood ratio: (1/2ln2)[ E_tar log(1+e^-s) + E_non log(1+e^s) ]
    min_cllr          the same cost after the optimal monotone (PAV) calibration
    calibration_loss  cllr - min_cllr

min_cllr is the DISCRIMINATION term (what no monotone recalibration can remove) and
calibration_loss is the calibration term. These scores are cosines and z-scores, never built to
be log-likelihood ratios, so a raw cllr near 1 reports a wrong scale, not an uninformative
system; systems are compared on min_cllr. Both readings are computed from the same float32 cast
of the scores that analyse_identity_encoder_v3.ranks_of ranks, so the ranking and the detection
statistics read identical numbers, ties included.

THREE ESTIMANDS, and every number carries its own. A weighted mean of a per-query quantity with
weight u IS the estimand, for the rank statistic and the detection statistics alike: u = 1 is
query-weighted, u = the protocol's per-(group, label) component weight is component-weighted
(what paired_group_bootstrap returns), and u = w / (W_label * 226) is label-macro (the
component-weighted mean within a label, then the plain mean over labels). A trial (q, l) carries
its query's u on both sides, which also makes the impostor side uniform over labels, since every
query is scored against every other label exactly once.

SYSTEMS, fifteen: the five published prototype spaces (raw BGE-M3 centroids; the same whitened
within author per fold; raw jieba word TF-IDF; raw character 2-5-gram TF-IDF; the word SVD-1024
whitened per fold, all from exemplar_vs_prototype_v3.setup and the folds of identity_probe_v2),
each under the protocol's prototype scorer and under label_size_calibration_v3's two published
calibrations, znorm and noise_corrected, by that file's own functions.

READING RULES, fixed before the run.

 1. THE 2x2 IS A MEASUREMENT, NOT A TEST. It carries no verdict beyond the decomposition: no
    interval in it is turned into a claim about a contrast (the cells carry no excludes_zero
    field), and no width factor is offered as reusable. The factor is reported for every
    contrast filled, with its spread, and with the plain statement that another study must fill
    its own 2x2 rather than borrow this corpus's "1.6 to 3.0".

 2. THE CALIBRATION READING is a four-way outcome, named here in advance, per space, per
    calibration and per estimand, from the change in MRR and in min_cllr against the prototype
    of the SAME space under the SAME estimand:
        MRR up,   min_cllr down  ->  "improvement"
        MRR up,   min_cllr up    ->  "a ranking gain bought with a calibration loss"
        MRR down, min_cllr down  ->  "a calibration gain bought with a ranking loss"
        MRR down, min_cllr up    ->  "no gain"
    A change of at most 0.005 in either quantity is reported as "unchanged within the declared
    band" rather than as a direction. 0.005 is the non-inferiority margin this repository
    already declares for MRR, carried over to min_cllr for want of an established one and named
    as a declared band, not an established margin; the raw differences are printed so another
    band can be applied. A calibration is an IMPROVEMENT only under the first outcome, and the
    outcome is ESTABLISHED only when, under the component-weighted estimand, the whole 95%
    interval of both differences clears the band in the required direction -- the MRR interval
    from paired_group_bootstrap, the min_cllr interval from a group bootstrap on the first 400
    rows of the identical draw matrix. Otherwise it is reported as a point comparison and said
    to be one, so a calibration cannot be credited by a run that merely lacks power. Where the
    three estimands disagree, the disagreement is named, not resolved.

 3. AUC, cllr and calibration_loss are DESCRIPTION and decide nothing.

CHECKS, and where they can abort. Up front, before the corpus is read: this file's pinned
prototype MRRs against label_size_calibration_v3.EXPECTED_ALL. Then the published corpus shape
(7,220 queries, 226 labels, 5,875 groups, 6,889 components) and the published songs-per-label
band counts; the estimand weights; the component table's grouping. Then, per space, unit rows
(4e-5 on the squared norm where the provenance is float32 -- v1.l2_normalize_dense normalises
in float32, v1.fit_tfidf's own contract is atol 2e-5, fit_words builds float32 -- and 1e-12 for
the two fold-wise spaces, whose matrices are identity_probe_v2.unit_rows output in float64), and
then every system's published query-weighted MRR the MOMENT it is scored and before anything new
is read from it (5e-4 raw and 0.002 fold-wise for the prototype, label_size_calibration_v3's own
CHECK_GAP for the calibrations). Those gates abort, by design, at the earliest moment the number
they check exists: the first fires about three minutes in. Every check that cannot fire until the
expensive work is finished -- the component table against the component-weighted mean, the
rebuilt group bootstrap against paired_group_bootstrap, the builder's component tensors and both
two-stage points, and the eight published cells of estimand_two_stage.json -- is recorded in the
payload, the payload is WRITTEN, and only then does a failure exit non-zero, so a late gate can
never throw away a finished run. tests/test_estimand_and_calibration.py checks every estimator
added here against an independent definition on data small enough to run in seconds.

    CHINESE_RAP_CORPUS=v3 python src/estimand_and_calibration_v3.py --private-root <private corpus root>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from estimand_two_stage_v3 import (  # noqa: E402
    FOLD_GAP,
    RAW_GAP,
    component_index,
    component_means,
    label_macro,
    per_label_components,
    two_stage,
)
from exemplar_vs_prototype_v3 import setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import REPLICATES as GROUP_REPLICATES  # noqa: E402
from identity_spaces_v2 import SEED as GROUP_SEED  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402
from label_size_calibration_v3 import (  # noqa: E402
    BANDS,
    CHECK_GAP,
    EXPECTED_ALL,
    lgo_parts,
    residual_variance,
    scorers_from_parts,
    znorm,
)
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "estimand_and_calibration.json"

# words_svd_whitened reads the word matrix, so it must stay after words_raw
SPACES = ("semantic_raw", "words_raw", "chars_raw", "semantic_whitened", "words_svd_whitened")
FOLDWISE = ("semantic_whitened", "words_svd_whitened")
SCORERS = ("prototype", "znorm", "noise_corrected")
CALIBRATIONS = ("znorm", "noise_corrected")
ESTIMANDS = ("query_weighted", "component_weighted", "label_macro")

# published corpus-v3 query-weighted prototype MRRs, cross-checked at run time against
# label_size_calibration_v3.EXPECTED_ALL (which reads them through expected())
EXPECTED_PROTOTYPE = {"semantic_raw": 0.2997, "words_raw": 0.4963, "chars_raw": 0.4266,
                      "semantic_whitened": 0.4164, "words_svd_whitened": 0.5426}
# label_size_calibration.json, systems.<space>.<scorer>: query-weighted MRR
EXPECTED_CALIBRATED = {
    "semantic_raw": {"znorm": 0.2100, "noise_corrected": 0.2635},
    "words_raw": {"znorm": 0.4868, "noise_corrected": 0.5394},
    "chars_raw": {"znorm": 0.3266, "noise_corrected": 0.4591},
    "semantic_whitened": {"znorm": 0.4144, "noise_corrected": 0.3902},
    "words_svd_whitened": {"znorm": 0.5577, "noise_corrected": 0.5291},
}
# label_size_calibration.json, fold_info.<space>."0".floor_hits_of_query_label_pairs: recorded
# beside this run's own count, NOT gated -- the count is a float comparison at a boundary and
# swings by a fifth between folds, so the producing code guarantees it to no tolerance
PUBLISHED_FLOOR_HITS_FOLD0 = {"semantic_raw": 0, "words_raw": 144428, "chars_raw": 130045,
                              "semantic_whitened": 115634, "words_svd_whitened": 43379}
EXPECTED_SHAPE = {"queries": 7220, "labels": 226, "groups": 5875, "components": 6889}
# songs-per-label bands (label_size_calibration_v3.BANDS): (labels, queries)
EXPECTED_BANDS = {"5-9": (19, 149), "10-19": (33, 477), "20-49": (171, 6356), "50-up": (3, 238)}

# estimand_two_stage.json, contrasts: (left, right, its name there, the component-weighted
# group-bootstrap point and interval, the label-macro two-stage point and interval)
PUBLISHED_CELLS = (
    ("words_raw", "chars_raw", "words_raw - characters_raw",
     0.0711, (0.0643, 0.0785), 0.0551, (0.0409, 0.0692)),
    ("chars_raw", "semantic_raw", "characters_raw - semantic_raw",
     0.1331, (0.1229, 0.1435), 0.0935, (0.0698, 0.1182)),
    ("words_raw", "semantic_raw", "words_raw - semantic_raw",
     0.2042, (0.1944, 0.2142), 0.1486, (0.1201, 0.1767)),
    ("words_svd_whitened", "semantic_whitened", "words_svd_within_whitened - semantic_within_whitened",
     0.1316, (0.1217, 0.1418), 0.1059, (0.0835, 0.1278)),
)
README_NOTE = {
    "source": "results/retrieval-v3/README.md, 'A note on estimands' (2026-09-23)",
    "convention": ("a system's mrr field is query-weighted; every paired_group_bootstrap interval "
                   "and its own point estimate is component-weighted, so two printed levels must "
                   "not be subtracted to reconstruct a published difference"),
    "query_weighted_minus_component_weighted_bound_on_a_level": 0.0057,
    "query_weighted_minus_component_weighted_bound_on_a_difference": 0.0076,
    "bounds_are": "cited from that note, measured there over the nine headline systems, not recomputed here",
}

TWO_STAGE_SEED = v1.RANDOM_SEED                 # 20260825, the builder's
TWO_STAGE_REPLICATES = v1.BOOTSTRAP_REPLICATES  # 5,000, the builder's
TWO_STAGE_METRICS = ("mrr", "size_weighted_mrr", "component_count")
LLR_REPLICATES = 400            # the first 400 rows of the group draw matrix, so the min_cllr
#                                 interval is paired with, and nested in, the MRR one
MARGIN = 0.005                  # the declared band, this repository's MRR margin
LOG2 = float(np.log(2.0))

PROTOTYPE_TOLERANCE = {s: (FOLD_GAP if s in FOLDWISE else RAW_GAP) for s in SPACES}
# float32 provenance: v1.l2_normalize_dense normalises in float32, v1.fit_tfidf's own contract is
# np.allclose(row_norms, 1.0, atol=2e-5) and fit_words builds float32, so |n^2 - 1| <= ~4e-5
UNIT_ROW_TOLERANCE = {s: (1e-12 if s in FOLDWISE else 4e-5) for s in SPACES}
IDENTITY_TOLERANCE = 1e-12      # float64 identities between two summation orders of one mean
POINT_TOLERANCE = 1e-9          # the same, where a rounded published point is on one side
# paired_group_bootstrap returns values already rounded to four decimals and its summation order
# differs from this file's matvec, so one unit in the fourth decimal is expected; 1e-4 itself is
# unusable as the bound because abs(0.0643 - 0.0644) == 1.0000000000000029e-04 in float64
GROUP_REBUILD_TOLERANCE = 1.5e-4
PUBLISHED_CELL_TOLERANCE = FOLD_GAP   # 0.002: the fold-wise spaces reach the published endpoints
#                                       through an SVD and a whitening, as their MRRs do

OUTCOME = {
    (1, -1): "improvement: MRR up and min_cllr down",
    (1, 1): "a ranking gain bought with a calibration loss: MRR up and min_cllr up",
    (1, 0): "a ranking gain with min_cllr unchanged within the declared band",
    (-1, -1): "a calibration gain bought with a ranking loss: MRR down and min_cllr down",
    (-1, 1): "no gain: MRR down and min_cllr up",
    (-1, 0): "a ranking loss with min_cllr unchanged within the declared band",
    (0, -1): "min_cllr down with MRR unchanged within the declared band",
    (0, 1): "min_cllr up with MRR unchanged within the declared band",
    (0, 0): "both unchanged within the declared band",
}


# ------------------------------------------------------------------ estimands
def pinned_mismatch(pinned: dict, imported: dict, tolerance: float) -> list[str]:
    """Names whose imported expectation contradicts this file's pinned literal.

    An imported None means the other file's own check is switched off -- expected() returns None
    when the published JSON is absent or stale -- and is not a contradiction.
    """
    return [f"{name}: pinned {value}, imported {imported.get(name)!r}" for name, value in pinned.items()
            if imported.get(name) is not None and abs(float(imported[name]) - value) > tolerance]


def estimand_weights(label_index: np.ndarray, weights: np.ndarray, label_count: int) -> dict:
    """The per-query weight u whose weighted mean IS each estimand.

    query_weighted     u = 1
    component_weighted u = the protocol's per-(group, label) component weight
    label_macro        u = w / (the label's total weight * label count), so each label
                       contributes total weight 1 / label_count and u sums to one
    """
    per_label = np.bincount(label_index, weights=weights, minlength=label_count)
    if np.any(per_label <= 0):
        raise SystemExit("a label without weight")
    w = np.asarray(weights, dtype=np.float64)
    return {"query_weighted": np.ones(len(label_index), dtype=np.float64),
            "component_weighted": w,
            "label_macro": w / (per_label[label_index] * label_count)}


def label_blocks(comp_label: np.ndarray, label_count: int) -> np.ndarray:
    """Row where each label's components start. component_index orders components by (label,
    group), so every label's components are contiguous and np.add.reduceat is the label sum."""
    counts = np.bincount(comp_label, minlength=label_count)
    if len(counts) != label_count or np.any(counts == 0) or np.any(np.diff(comp_label) < 0):
        raise SystemExit("the component table is not grouped by label")
    return np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)


# ------------------------------------------------------------------ the trial list
def trial_blocks(scores: np.ndarray, label_index: np.ndarray, pool: bool = True) -> dict:
    """Every (query, label) trial in score order, pooled into blocks no statistic here can tell
    apart from the trials themselves.

    A block never splits trials that share a score, and it pools neighbouring score-groups only
    when both are pure and of the same class. That pooling changes nothing: AUC only ever
    compares a target with a non-target, so the order inside a run of one class is invisible to
    it, and the PAV solution never puts a boundary between two neighbouring trials of the same
    class either -- the fitted values are the slopes of the greatest convex minorant of the
    cumulative (weight, weight x label) diagram, every slope of which lies in [0, 1], and a kink
    just after a non-target would force the slope to its right to be <= 0 while a kink just
    after a target would force the slope to its left to be >= 1. pool=False keeps one block per
    distinct score and exists so the test file can check the two against each other.
    """
    n, label_count = scores.shape
    if label_count < int(np.max(label_index)) + 1:
        raise SystemExit("the score matrix has fewer columns than there are labels")
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_score = flat[order]
    query = (order // label_count).astype(np.int32)
    target = label_index[query] == (order % label_count)
    fresh = np.empty(len(order), dtype=bool)
    fresh[0] = True
    fresh[1:] = sorted_score[1:] != sorted_score[:-1]
    group_of = np.cumsum(fresh) - 1
    group_count = int(group_of[-1]) + 1
    group_start = np.flatnonzero(fresh)
    hits = np.bincount(group_of, weights=target.astype(np.float64), minlength=group_count)
    misses = np.bincount(group_of, weights=(~target).astype(np.float64), minlength=group_count)
    kind = np.where((hits > 0) & (misses > 0), 2, np.where(hits > 0, 1, 0))
    if pool:
        opens = np.ones(group_count, dtype=bool)
        if group_count > 1:
            opens[1:] = (kind[1:] != kind[:-1]) | (kind[1:] == 2) | (kind[:-1] == 2)
        block_start = group_start[opens]
    else:
        block_start = group_start
    return {"query": query, "target": target, "block_start": np.ascontiguousarray(block_start),
            "blocks": int(len(block_start)), "score_groups": group_count,
            "mixed_score_groups": int(np.count_nonzero(kind == 2)), "trials": int(n * label_count),
            "target_trials": int(n), "nontarget_trials": int(n * (label_count - 1))}


def block_weight(bundle: dict, u: np.ndarray):
    """Target and non-target weight of every block, under the per-query weighting u."""
    wt = u[bundle["query"]]
    total = np.add.reduceat(wt, bundle["block_start"])
    hit = np.add.reduceat(wt * bundle["target"], bundle["block_start"])
    return hit, total - hit


# ------------------------------------------------------------------ detection statistics
def auc_from_blocks(t: np.ndarray, f: np.ndarray) -> float:
    """Weighted probability that a target trial outranks a non-target one, ties at one half."""
    keep = (t + f) > 0
    t, f = t[keep], f[keep]
    total_t, total_f = float(t.sum()), float(f.sum())
    if total_t <= 0.0 or total_f <= 0.0:
        raise SystemExit("one side of the trial list carries no weight")
    below = np.concatenate(([0.0], np.cumsum(f)))[:-1]
    return float((float(np.sum(t * below)) + 0.5 * float(np.sum(t * f))) / (total_t * total_f))


def pav_blocks(t: np.ndarray, f: np.ndarray):
    """Pool adjacent violators: the coarsest blocks on which the target share is non-decreasing.

    The comparison is cross-multiplied, so no division and no posterior is formed: block a
    violates block b when t_a (t_b + f_b) > t_b (t_a + f_a).
    """
    stack_t: list[float] = []
    stack_f: list[float] = []
    for ti, fi in zip(t.tolist(), f.tolist()):
        stack_t.append(ti)
        stack_f.append(fi)
        while len(stack_t) >= 2 and (stack_t[-2] * (stack_t[-1] + stack_f[-1])
                                     > stack_t[-1] * (stack_t[-2] + stack_f[-2])):
            merged_t = stack_t.pop()
            merged_f = stack_f.pop()
            stack_t[-1] += merged_t
            stack_f[-1] += merged_f
    return np.asarray(stack_t, dtype=np.float64), np.asarray(stack_f, dtype=np.float64)


def cost_of_blocks(bt: np.ndarray, bf: np.ndarray) -> float:
    """C_llr of a system whose log-likelihood ratio is constant on each block.

    On a block the cost-minimising constant is logit(t / (t + f)) - logit(prior), the prior being
    the trial list's own target proportion, so exp(-llr) = (T/F)(f/t) and exp(llr) = (F/T)(t/f)
    and the two terms are written directly with log1p. A block with no target weight costs the
    target side nothing however extreme its ratio, which is why perfect separation gives 0.
    """
    total_t, total_f = float(bt.sum()), float(bf.sum())
    if total_t <= 0.0 or total_f <= 0.0:
        raise SystemExit("one side of the trial list carries no weight")
    odds = total_t / total_f
    safe_t = np.where(bt > 0, bt, 1.0)
    safe_f = np.where(bf > 0, bf, 1.0)
    target_cost = np.where(bt > 0, bt * np.log1p(odds * bf / safe_t), 0.0)
    nontarget_cost = np.where(bf > 0, bf * np.log1p(bt / (odds * safe_f)), 0.0)
    return float((target_cost.sum() / total_t + nontarget_cost.sum() / total_f) / (2.0 * LOG2))


def min_cllr(t: np.ndarray, f: np.ndarray) -> float:
    keep = (t + f) > 0
    return cost_of_blocks(*pav_blocks(t[keep], f[keep]))


def cllr_parts(scores: np.ndarray, label_index: np.ndarray):
    """Per query, the two softplus sums C_llr needs, so that C_llr under any per-query weighting
    is a ratio of per-query sums and its group bootstrap is a bincount, not a rescoring."""
    rows = np.arange(len(label_index))
    positive = np.logaddexp(0.0, scores)
    target_term = np.logaddexp(0.0, -scores[rows, label_index])
    nontarget_term = positive.sum(axis=1) - positive[rows, label_index]
    return np.asarray(target_term, dtype=np.float64), np.asarray(nontarget_term, dtype=np.float64)


def cllr_from_parts(target_term, nontarget_term, u, label_count: int) -> float:
    total = float(np.sum(u))
    if total <= 0.0:
        raise SystemExit("an empty weighting")
    tar = float(np.sum(u * target_term)) / total
    non = float(np.sum(u * nontarget_term)) / (total * (label_count - 1))
    return (tar + non) / (2.0 * LOG2)


def detection_readings(bundle, target_term, nontarget_term, u, label_count) -> dict:
    t, f = block_weight(bundle, u)
    cl = cllr_from_parts(target_term, nontarget_term, u, label_count)
    mc = min_cllr(t, f)
    return {"auc": auc_from_blocks(t, f), "cllr": cl, "min_cllr": mc, "calibration_loss": cl - mc}


# ------------------------------------------------------------------ resampling
def group_positions_and_draws(group_ids: np.ndarray, replicates: int, seed: int):
    """The group index and draw matrix identity_spaces_v2.paired_group_bootstrap builds for an
    all-true mask, rebuilt here so every group cell rests on the identical draws."""
    groups = np.unique(group_ids)
    position = {int(g): i for i, g in enumerate(groups.tolist())}
    index = np.asarray([position[int(g)] for g in group_ids.tolist()], dtype=np.int64)
    rng = np.random.default_rng(seed)
    return index, rng.integers(0, len(groups), size=(replicates, len(groups))), int(len(groups))


def group_replicates(comp_values, comp_group, label_starts, draws, group_count):
    """Per replicate of the group bootstrap, the component-weighted and the label-macro statistic
    of every column of the component table.

    A component's total protocol weight is exactly one, so a group draw with multiplicities m
    gives sum_k m_k v_k / sum_k m_k for the component-weighted estimand and, within a label, the
    mean of that label's drawn components; the macro is the plain mean over the labels the draw
    leaves non-empty, and the number it leaves empty is recorded.
    """
    replicates, columns = int(draws.shape[0]), int(comp_values.shape[1])
    label_count = int(len(label_starts))
    comp_weighted = np.zeros((replicates, columns))
    macro = np.zeros((replicates, columns))
    empty = np.zeros(replicates, dtype=np.int64)
    for r in range(replicates):
        mult = np.bincount(draws[r], minlength=group_count).astype(np.float64)[comp_group]
        comp_weighted[r] = (mult @ comp_values) / mult.sum()
        num = np.add.reduceat(comp_values * mult[:, None], label_starts, axis=0)
        den = np.add.reduceat(mult, label_starts)
        ok = den > 0
        empty[r] = int(label_count - int(np.count_nonzero(ok)))
        macro[r] = (num[ok] / den[ok, None]).mean(axis=0)
    return comp_weighted, macro, empty


def two_stage_columns(point: np.ndarray, reps: np.ndarray, metrics=TWO_STAGE_METRICS):
    """Both estimands off the builder's own two-stage draws, with no transcription of them.

    v1.run_bootstrap consumes its generator only through label_count and each drawn label's
    component count, never through the systems or metrics passed, so extra metric columns ride
    the identical draws. Two of them turn the macro into the component-weighted statistic: with
    n_l the number of components of a label, the macro of n_l * v is (1/L) sum_occ n_l mean_occ
    and the macro of n_l is (1/L) sum_occ n_l, so their ratio is the plain mean over every drawn
    component -- the component-weighted estimand -- replicate by replicate.
    """
    m = len(metrics)
    macro = reps[:, 0::m]
    component = reps[:, 1::m] / reps[:, 2::m]
    return (point[0::m], point[1::m] / point[2::m], macro, component,
            float(np.abs(reps[:, 2::m] - reps[:, 2::m][:, :1]).max()))


def min_cllr_group_series(bundle, u, group_of_query, draws, group_count) -> np.ndarray:
    """min-C_llr under every replicate of the group bootstrap.

    The block structure of the trial list depends only on which class each trial belongs to, not
    on the weights, so it is built once; a replicate only re-aggregates the weights into it and
    pools adjacent violators again.
    """
    out = np.zeros(int(draws.shape[0]))
    for r in range(int(draws.shape[0])):
        m = np.bincount(draws[r], minlength=group_count).astype(np.float64)
        out[r] = min_cllr(*block_weight(bundle, u * m[group_of_query]))
    return out


def cell(draws: np.ndarray, point: float, verdict: bool = False) -> dict:
    """One cell: the estimand's point and the resampling's interval. np.quantile at 0.025/0.975
    is the same call as paired_group_bootstrap's np.percentile at 2.5/97.5, so both published
    conventions coincide; the test asserts it."""
    low, high = np.quantile(draws, [0.025, 0.975])
    out = {"difference": round(float(point), 4), "ci95": [round(float(low), 4), round(float(high), 4)],
           "width": round(float(high) - float(low), 6)}
    if verdict:
        out["excludes_zero"] = bool(low > 0 or high < 0)
    return out


def decompose(widths: dict) -> dict:
    """Split the total width factor into an estimand factor and a resampling factor."""
    g_cw = widths[("component_weighted", "group_bootstrap")]
    g_lm = widths[("label_macro", "group_bootstrap")]
    t_cw = widths[("component_weighted", "two_stage")]
    t_lm = widths[("label_macro", "two_stage")]
    if min(g_cw, g_lm, t_cw, t_lm) <= 0.0:
        raise SystemExit("a cell of the 2x2 has a non-positive width")
    at_group, at_two_stage = g_lm / g_cw, t_lm / t_cw
    total = t_lm / g_cw
    log_total = float(np.log(total))
    share = None if abs(log_total) < 1e-6 else float(
        0.5 * (np.log(at_group) + np.log(at_two_stage)) / log_total)
    return {
        "estimand_factor_at_group_bootstrap": round(at_group, 3),
        "estimand_factor_at_two_stage": round(at_two_stage, 3),
        "resampling_factor_at_component_weighted": round(t_cw / g_cw, 3),
        "resampling_factor_at_label_macro": round(t_lm / g_lm, 3),
        "total_factor": round(total, 3),
        "interaction_estimand_factor_two_stage_over_group": round(at_two_stage / at_group, 3),
        "share_of_log_total_estimand": None if share is None else round(share, 3),
        "share_of_log_total_resampling": None if share is None else round(1.0 - share, 3),
        "note": ("the total is W(label-macro, two-stage) / W(component-weighted, group) and is the "
                 "product of the estimand and resampling factors along either path exactly; the "
                 "shares are the symmetric (Shapley) split of its logarithm"),
    }


def direction(delta: float, band: float = MARGIN) -> int:
    """-1, 0 or 1. The band is compared with a tolerance so that a difference landing on it
    exactly cannot fall either side by binary representation."""
    return 1 if delta > band + 1e-12 else (-1 if delta < -band - 1e-12 else 0)


# ------------------------------------------------------------------ spaces
def unit_row_gap(x):
    """max |row squared norm - 1| over the non-empty rows, and the number of empty rows."""
    if sparse.issparse(x):
        norm2 = np.asarray(x.multiply(x).sum(axis=1)).ravel()
    else:
        norm2 = np.einsum("ij,ij->i", x, x)
    empty = int(np.count_nonzero(norm2 <= 1e-12))
    live = norm2[norm2 > 1e-12]
    return float(np.abs(live - 1.0).max()) if len(live) else 0.0, empty


def fold_matrices(space: str, dense, sparse_space: dict, label_index, weights, fold) -> dict:
    """{fold: matrix} for one space. A raw space is ONE matrix shared by every fold, so its
    leave-group-out parts are computed once."""
    if space == "semantic_raw":
        return {k: dense for k in range(FOLDS)}
    if space in ("words_raw", "chars_raw"):
        return {k: sparse_space[space] for k in range(FOLDS)}
    out = {}
    for k in range(FOLDS):
        train = fold != k
        if space == "semantic_whitened":
            mean, matrix, _ = fit_transform("within_author_whitening", dense[train], label_index[train],
                                            weights[train], np.random.default_rng(SEED + k))
            out[k] = unit_rows((dense - mean) @ matrix.T)
        elif space == "words_svd_whitened":
            words = sparse_space["words_raw"]
            svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(words[train])
            reduced = unit_rows(svd.transform(words))
            mean, matrix, _ = fit_transform("within_author_whitening", reduced[train], label_index[train],
                                            weights[train], np.random.default_rng(SEED + k))
            out[k] = unit_rows((reduced - mean) @ matrix.T)
        else:
            raise SystemExit(f"unknown space {space}")
        print(f"    fold {k} transform fitted", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True,
                        help="the private corpus root; never hard-coded in this file")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()

    # ---- up front, before the corpus is read
    mismatch = pinned_mismatch(EXPECTED_PROTOTYPE, EXPECTED_ALL, RAW_GAP)
    if mismatch:
        raise SystemExit("this file's pinned prototype MRRs contradict label_size_calibration_v3."
                         "EXPECTED_ALL; CHINESE_RAP_CORPUS must be v3 with a current "
                         f"three_spaces.json: {mismatch}")
    import jieba
    jieba.setLogLevel(60)

    d = setup(args.private_root.resolve())
    li, gi, w, L, fold, dense = (d["label_index"], d["group_ids"], d["weights"], d["label_count"],
                                 d["fold"], d["dense"])
    n = len(li)
    all_queries = np.arange(n)                      # lgo_parts wants indices
    all_mask = np.ones(n, dtype=bool)               # weighted_mean wants a mask
    comp_of, comp_label = component_index(li, gi)
    component_count = int(len(comp_label))
    label_starts = label_blocks(comp_label, L)
    group_of_query, draws, group_count = group_positions_and_draws(gi, GROUP_REPLICATES, GROUP_SEED)
    comp_group = np.zeros(component_count, dtype=np.int64)
    comp_group[comp_of] = group_of_query
    shape = {"queries": n, "labels": L, "groups": group_count, "components": component_count}
    print(f"{n:,} queries, {L} labels, {group_count:,} groups, {component_count:,} components", flush=True)
    if shape != EXPECTED_SHAPE:
        raise SystemExit(f"the corpus shape is not the published one: {shape} against {EXPECTED_SHAPE}")

    songs_per_label = np.bincount(li, minlength=L)
    band_counts = {}
    for name, lo, hi in BANDS:
        labels_in = np.flatnonzero((songs_per_label >= lo) & (songs_per_label <= hi))
        band_counts[name] = (int(len(labels_in)), int(np.isin(li, labels_in).sum()))
    print(f"songs-per-label bands (labels, queries): {band_counts}", flush=True)
    if band_counts != {k: tuple(v) for k, v in EXPECTED_BANDS.items()}:
        raise SystemExit(f"the songs-per-label bands are not the published ones: {band_counts}")

    u_by_estimand = estimand_weights(li, w, L)
    if abs(float(np.sum(u_by_estimand["label_macro"])) - 1.0) > IDENTITY_TOLERANCE:
        raise SystemExit("the label-macro weighting does not sum to one")

    print("segmenting and fitting the two sparse spaces", flush=True)
    documents = [d["documents"][s] for s in d["songs"]]
    words, _ = fit_words([" ".join(segment(doc)) for doc in documents])
    sparse_space = {"words_raw": words.astype(np.float64).tocsr(),
                    "chars_raw": v1.fit_tfidf(documents).astype(np.float64).tocsr()}
    del documents, words
    d["documents"] = None

    rr: dict[str, np.ndarray] = {}
    mrr_by: dict[str, dict] = {}
    det_by: dict[str, dict] = {}
    trials_by: dict[str, dict] = {}
    min_series: dict[str, np.ndarray] = {}
    published_mrr: dict[str, dict] = {}
    space_checks: dict[str, dict] = {}

    for space in SPACES:
        print(f"space {space}", flush=True)
        t_space = time.time()
        matrices = fold_matrices(space, dense, sparse_space, li, w, fold)
        gap, empty_rows = unit_row_gap(matrices[0])
        print(f"  unit rows: max |norm^2 - 1| {gap:.2e} (tolerance {UNIT_ROW_TOLERANCE[space]}), "
              f"{empty_rows} empty rows", flush=True)
        if gap > UNIT_ROW_TOLERANCE[space]:
            raise SystemExit(f"{space}: the rows are not unit length within the tolerance its "
                             f"provenance guarantees ({gap:.2e} > {UNIT_ROW_TOLERANCE[space]})")
        space_checks[space] = {"unit_row_squared_norm_gap": gap, "tolerance": UNIT_ROW_TOLERANCE[space],
                               "empty_rows": empty_rows, "folds": {}}

        scores = {s: np.zeros((n, L)) for s in SCORERS}
        parts_by_matrix: dict[int, tuple] = {}
        for k in range(FOLDS):
            x = matrices[k]
            train, test = fold != k, np.flatnonzero(fold == k)
            if id(x) not in parts_by_matrix:
                parts_by_matrix = {id(x): lgo_parts(x, li, gi, w, L, all_queries)}
            dots, norm2, counts = parts_by_matrix[id(x)]
            s2 = residual_variance(x, dots, norm2, counts, li, w, train)
            prototype_all, corrected_all, floor_hits = scorers_from_parts(dots, norm2, counts, s2)
            scores["prototype"][test] = prototype_all[test]
            scores["noise_corrected"][test] = corrected_all[test]
            scores["znorm"][test] = znorm(prototype_all, li, train, test)
            space_checks[space]["folds"][str(k)] = {"s2": round(float(s2), 4),
                                                   "noise_corrected_floor_hits": int(floor_hits)}
            print(f"    fold {k}: s2 {s2:.4f}, floor hits {floor_hits:,} of {n * L:,}", flush=True)
            del prototype_all, corrected_all
        del parts_by_matrix, matrices
        space_checks[space]["published_floor_hits_fold0"] = PUBLISHED_FLOOR_HITS_FOLD0[space]
        space_checks[space]["floor_hits_note"] = ("recorded beside label_size_calibration.json's own "
                                                 "fold-0 count, not gated: the count is a float "
                                                 "comparison at a boundary and swings by a fifth "
                                                 "between folds")

        for scorer in SCORERS:
            name = f"{space}/{scorer}"
            # the float32 cast ranks_of ranks, so the ranking and the detection statistics read
            # identical numbers, ties included
            matrix = scores.pop(scorer).astype(np.float32).astype(np.float64)
            rr[name] = 1.0 / ranks_of(matrix, li)
            got = float(np.mean(rr[name]))
            if scorer == "prototype":
                expected, tolerance = EXPECTED_PROTOTYPE[space], PROTOTYPE_TOLERANCE[space]
            else:
                expected, tolerance = EXPECTED_CALIBRATED[space][scorer], CHECK_GAP
            published_mrr[name] = {"published_query_weighted_mrr": expected,
                                   "recomputed_query_weighted_mrr": round(got, 4),
                                   "gap": round(abs(got - expected), 5), "tolerance": tolerance,
                                   "source": ("label_size_calibration.json systems." + space + "." + scorer)}
            print(f"  check {name:34s} query-weighted MRR {got:.4f} published {expected:.4f} "
                  f"(gap {abs(got - expected):.4f}, tol {tolerance})", flush=True)
            if abs(got - expected) > tolerance:
                raise SystemExit(f"{name}: the published MRR does not reproduce")

            # nothing new is read from a system until its published MRR has passed
            bundle = trial_blocks(matrix, li)
            target_term, nontarget_term = cllr_parts(matrix, li)
            del matrix
            mrr_by[name] = {e: weighted_mean(rr[name], u_by_estimand[e], all_mask) for e in ESTIMANDS}
            det_by[name] = {e: detection_readings(bundle, target_term, nontarget_term,
                                                  u_by_estimand[e], L) for e in ESTIMANDS}
            trials_by[name] = {"total": bundle["trials"], "target": bundle["target_trials"],
                               "nontarget": bundle["nontarget_trials"],
                               "distinct_scores": bundle["score_groups"],
                               "scores_shared_across_the_two_classes": bundle["mixed_score_groups"],
                               "blocks_before_pooling_violators": bundle["blocks"]}
            min_series[name] = min_cllr_group_series(bundle, u_by_estimand["component_weighted"],
                                                     group_of_query, draws[:LLR_REPLICATES], group_count)
            det = det_by[name]["component_weighted"]
            print(f"    {name:34s} MRR {mrr_by[name]['component_weighted']:.4f}  AUC {det['auc']:.4f}  "
                  f"cllr {det['cllr']:.4f}  min_cllr {det['min_cllr']:.4f}  "
                  f"calibration loss {det['calibration_loss']:.4f}  (component-weighted)", flush=True)
            del bundle, target_term, nontarget_term
        # dropping a dict entry frees the matrix without unbinding any name
        if space == "chars_raw":
            sparse_space.pop("chars_raw")
        if space == "words_svd_whitened":
            sparse_space.pop("words_raw")
        print(f"  {space} done in {(time.time() - t_space) / 60:.1f} min", flush=True)

    systems = [f"{space}/{scorer}" for space in SPACES for scorer in SCORERS]
    column = {name: j for j, name in enumerate(systems)}
    contrasts = tuple([(f"{a}/prototype", f"{b}/prototype") for a, b, *_rest in PUBLISHED_CELLS]
                      + [(f"{space}/{scorer}", f"{space}/prototype")
                         for space in SPACES for scorer in CALIBRATIONS])

    # ---- one component table, for both resamplings
    print("every published MRR reproduced; building the component table", flush=True)
    values = np.column_stack([rr[name] for name in systems])
    comp_values = component_means(values, comp_of, component_count)
    point_component = np.asarray([weighted_mean(values[:, j], w, all_mask) for j in range(len(systems))])
    point_macro = label_macro(values, w, li, L)
    late = {}
    late["component_table_is_the_component_weighted_estimand"] = {
        "gap": float(np.abs(comp_values.mean(axis=0) - point_component).max()),
        "tolerance": POINT_TOLERANCE}

    # ---- the group bootstrap, on paired_group_bootstrap's own draws
    print(f"group bootstrap: {GROUP_REPLICATES} replicates, seed {GROUP_SEED}", flush=True)
    group_component, group_macro, empty_labels = group_replicates(
        comp_values, comp_group, label_starts, draws, group_count)
    print(f"  labels left empty by a group draw: mean {empty_labels.mean():.2f}, "
          f"max {int(empty_labels.max())} of {L}", flush=True)
    reference = {(c["system"], c["minus"]): c
                 for c in paired_group_bootstrap(rr, w, gi, all_mask, list(contrasts))}
    rebuild_gap = 0.0
    for left, right in contrasts:
        mine = cell(group_component[:, column[left]] - group_component[:, column[right]],
                    point_component[column[left]] - point_component[column[right]])
        theirs = reference[(left, right)]
        rebuild_gap = max(rebuild_gap, abs(mine["difference"] - theirs["mrr_difference"]),
                          abs(mine["ci95"][0] - theirs["ci95"][0]),
                          abs(mine["ci95"][1] - theirs["ci95"][1]))
    print(f"check: the rebuilt group bootstrap against paired_group_bootstrap: {rebuild_gap:.2e}", flush=True)
    late["rebuilt_group_bootstrap_against_paired_group_bootstrap"] = {
        "gap": rebuild_gap, "tolerance": GROUP_REBUILD_TOLERANCE}

    # ---- the builder's two-stage bootstrap, one call, both estimands off its own draws
    print(f"two-stage bootstrap by the builder's own functions: {TWO_STAGE_REPLICATES} replicates, "
          f"seed {TWO_STAGE_SEED}", flush=True)
    label_component_count = np.bincount(comp_label, minlength=L).astype(np.float64)[li]
    metric_arrays = {name: {"mrr": rr[name],
                            "size_weighted_mrr": rr[name] * label_component_count,
                            "component_count": label_component_count} for name in systems}
    components, label_group_counts, builder_point, builder_reps, diagnostics = two_stage(
        metric_arrays, li, gi, systems, TWO_STAGE_METRICS)
    macro_point, ratio_point, macro_reps, component_reps, size_spread = two_stage_columns(
        builder_point, builder_reps)
    own = per_label_components(comp_values, comp_label, L)
    if int(label_group_counts.sum()) != component_count or len(components) != L:
        raise SystemExit("the builder's component tensors do not match this file's components")
    late["builder_component_tensors_against_this_table"] = {
        "gap": max(float(np.abs(np.asarray(a)[:, :, 0] - b).max()) for a, b in zip(components, own)),
        "tolerance": IDENTITY_TOLERANCE}
    late["builder_two_stage_point_is_the_label_macro"] = {
        "gap": float(np.abs(macro_point - point_macro).max()), "tolerance": POINT_TOLERANCE}
    late["builder_size_weighted_ratio_is_the_component_weighted_point"] = {
        "gap": float(np.abs(ratio_point - point_component).max()), "tolerance": POINT_TOLERANCE}
    late["component_count_column_is_system_independent"] = {
        "gap": size_spread, "tolerance": IDENTITY_TOLERANCE}
    del components, own, values

    # ---- the 2x2
    cells, widths = {}, {}
    for left, right in contrasts:
        i, j = column[left], column[right]
        d_component = float(point_component[i] - point_component[j])
        d_macro = float(point_macro[i] - point_macro[j])
        here = {("component_weighted", "group_bootstrap"):
                    cell(group_component[:, i] - group_component[:, j], d_component),
                ("label_macro", "group_bootstrap"):
                    cell(group_macro[:, i] - group_macro[:, j], d_macro),
                ("component_weighted", "two_stage"):
                    cell(component_reps[:, i] - component_reps[:, j], d_component),
                ("label_macro", "two_stage"):
                    cell(macro_reps[:, i] - macro_reps[:, j], d_macro)}
        cells[(left, right)] = here
        widths[(left, right)] = {key: value["width"] for key, value in here.items()}

    published_checks = {}
    worst_published = 0.0
    for left_space, right_space, published_name, cw_point, cw_ci, lm_point, lm_ci in PUBLISHED_CELLS:
        key = (f"{left_space}/prototype", f"{right_space}/prototype")
        cw = cells[key][("component_weighted", "group_bootstrap")]
        lm = cells[key][("label_macro", "two_stage")]
        gaps = [abs(cw["difference"] - cw_point), abs(cw["ci95"][0] - cw_ci[0]),
                abs(cw["ci95"][1] - cw_ci[1]), abs(lm["difference"] - lm_point),
                abs(lm["ci95"][0] - lm_ci[0]), abs(lm["ci95"][1] - lm_ci[1])]
        worst_published = max(worst_published, float(max(gaps)))
        published_checks[published_name] = {
            "published_component_weighted_group_bootstrap": [cw_point, list(cw_ci)],
            "recomputed_component_weighted_group_bootstrap": [cw["difference"], cw["ci95"]],
            "published_label_macro_two_stage": [lm_point, list(lm_ci)],
            "recomputed_label_macro_two_stage": [lm["difference"], lm["ci95"]],
            "max_gap": round(float(max(gaps)), 5), "tolerance": PUBLISHED_CELL_TOLERANCE,
            "source": "results/retrieval-v3/estimand_two_stage.json"}
        print(f"check published {published_name}: component-weighted group {cw['difference']:+.4f} "
              f"{cw['ci95']} against {cw_point:+.4f} {list(cw_ci)}; label-macro two-stage "
              f"{lm['difference']:+.4f} {lm['ci95']} against {lm_point:+.4f} {list(lm_ci)} "
              f"(max gap {max(gaps):.4f})", flush=True)
    late["published_two_by_two_cells"] = {"gap": worst_published, "tolerance": PUBLISHED_CELL_TOLERANCE}

    two_by_two, totals = {}, {}
    for left, right in contrasts:
        factors = decompose(widths[(left, right)])
        totals[f"{left} - {right}"] = factors["total_factor"]
        two_by_two[f"{left} - {right}"] = {
            "cells": {f"{estimand}/{resampling}": value
                      for (estimand, resampling), value in cells[(left, right)].items()},
            "width_factors": factors}
        print(f"  {left} - {right}: estimand factor {factors['estimand_factor_at_group_bootstrap']} "
              f"(group) / {factors['estimand_factor_at_two_stage']} (two-stage); resampling factor "
              f"{factors['resampling_factor_at_component_weighted']} (component-weighted) / "
              f"{factors['resampling_factor_at_label_macro']} (label-macro); total "
              f"{factors['total_factor']}", flush=True)

    estimand_factors = [v["width_factors"]["estimand_factor_at_group_bootstrap"] for v in two_by_two.values()]
    resampling_factors = [v["width_factors"]["resampling_factor_at_component_weighted"] for v in two_by_two.values()]
    shares = [v["width_factors"]["share_of_log_total_estimand"] for v in two_by_two.values()
              if v["width_factors"]["share_of_log_total_estimand"] is not None]
    width_summary = {
        "contrasts_filled": len(two_by_two),
        "total_factor": {"min": min(totals.values()), "max": max(totals.values()),
                         "median": round(float(np.median(list(totals.values()))), 3)},
        "estimand_factor_at_group_bootstrap": {"min": min(estimand_factors), "max": max(estimand_factors)},
        "resampling_factor_at_component_weighted": {"min": min(resampling_factors),
                                                    "max": max(resampling_factors)},
        "share_of_log_total_attributable_to_the_estimand": (
            {"min": round(min(shares), 3), "max": round(max(shares), 3),
             "median": round(float(np.median(shares)), 3)} if shares else None),
        "labels_left_empty_by_a_group_draw": {"mean": round(float(empty_labels.mean()), 3),
                                              "max": int(empty_labels.max()), "of": L,
                                              "rule": "a label with no drawn component is dropped "
                                                      "from that replicate's label-macro"},
        "reading": ("the published 1.6-to-3.0 width factor is a product of an estimand factor and a "
                    "resampling factor and neither is constant across contrasts; the number is not "
                    "reusable and another study must fill its own 2x2 rather than borrow this one"),
    }
    print(f"width factor: total {width_summary['total_factor']}", flush=True)

    # ---- the pre-registered calibration reading
    calibration = {}
    for space in SPACES:
        base = f"{space}/prototype"
        calibration[space] = {}
        for scorer in CALIBRATIONS:
            name = f"{space}/{scorer}"
            per_estimand = {}
            for estimand in ESTIMANDS:
                d_mrr = mrr_by[name][estimand] - mrr_by[base][estimand]
                d_min = det_by[name][estimand]["min_cllr"] - det_by[base][estimand]["min_cllr"]
                per_estimand[estimand] = {
                    "mrr_difference": round(float(d_mrr), 5),
                    "min_cllr_difference": round(float(d_min), 5),
                    "cllr_difference": round(float(det_by[name][estimand]["cllr"]
                                                   - det_by[base][estimand]["cllr"]), 5),
                    "auc_difference": round(float(det_by[name][estimand]["auc"]
                                                  - det_by[base][estimand]["auc"]), 5),
                    "outcome": OUTCOME[(direction(d_mrr), direction(d_min))]}
            mrr_cell = reference[(name, base)]
            min_cell = cell(min_series[name] - min_series[base],
                            det_by[name]["component_weighted"]["min_cllr"]
                            - det_by[base]["component_weighted"]["min_cllr"], verdict=True)
            clears = bool(mrr_cell["ci95"][0] > MARGIN and min_cell["ci95"][1] < -MARGIN)
            outcome = per_estimand["component_weighted"]["outcome"]
            distinct = sorted({per_estimand[e]["outcome"] for e in ESTIMANDS})
            calibration[space][scorer] = {
                "by_estimand": per_estimand,
                "component_weighted_intervals": {
                    "estimand": "component_weighted, both of them",
                    "mrr_difference_group_bootstrap": {"difference": mrr_cell["mrr_difference"],
                                                       "ci95": mrr_cell["ci95"],
                                                       "excludes_zero": mrr_cell["excludes_zero"],
                                                       "replicates": GROUP_REPLICATES},
                    "min_cllr_difference_group_bootstrap": {"difference": min_cell["difference"],
                                                            "ci95": min_cell["ci95"],
                                                            "excludes_zero": min_cell["excludes_zero"],
                                                            "replicates": LLR_REPLICATES}},
                "outcome_component_weighted": outcome,
                "both_intervals_clear_the_band": clears,
                "established_improvement": bool(outcome.startswith("improvement") and clears),
                "estimands_agree": bool(len(distinct) == 1),
                "estimand_outcomes": distinct}
            print(f"  {name:34s} dMRR {per_estimand['component_weighted']['mrr_difference']:+.4f} "
                  f"{mrr_cell['ci95']}  dmin_cllr "
                  f"{per_estimand['component_weighted']['min_cllr_difference']:+.4f} "
                  f"{min_cell['ci95']} -> {outcome}"
                  f"{'' if clears else ' (point comparison)'}", flush=True)

    improvements = [f"{s}/{c}" for s in SPACES for c in CALIBRATIONS
                    if calibration[s][c]["established_improvement"]]
    ranking_gains = [f"{s}/{c}" for s in SPACES for c in CALIBRATIONS
                     if calibration[s][c]["outcome_component_weighted"].startswith("a ranking gain bought")]
    failed = sorted(k for k, v in late.items() if v["gap"] > v["tolerance"])
    for key, value in late.items():
        value["passes"] = bool(value["gap"] <= value["tolerance"])

    payload = {
        "analysis": "estimand_and_calibration_v3",
        "question": ("how much of the published width factor is the estimand and how much is the "
                     "resampling, and whether the calibrations that raise MRR also improve the "
                     "likelihood ratio"),
        "status": "every gate passed" if not failed else f"FAILED late gates: {failed}",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L,
                   "groups": group_count, "components": component_count,
                   "query_label_trials": int(n * L),
                   "songs_per_label_bands": {k: {"labels": v[0], "queries": v[1]}
                                             for k, v in band_counts.items()},
                   "bands_are": ("by SONGS per label (label_size_calibration_v3.BANDS), used here only "
                                 "as a corpus-shape gate; the components-per-label banding used "
                                 "elsewhere in this repo is a different banding and no reading in this "
                                 "file is by band")},
        "conventions": README_NOTE,
        "design": {
            "systems": {"spaces": list(SPACES), "scorers": list(SCORERS),
                        "prototype": "cosine with the normalised leave-group-out weighted sum (the protocol)",
                        "znorm": "the prototype score standardised per label by the impostor scores of "
                                 "training-fold songs of other labels (label_size_calibration_v3.znorm)",
                        "noise_corrected": "the hierarchical correction of the profile norm "
                                           "(label_size_calibration_v3.scorers_from_parts)"},
            "estimands": {"query_weighted": "each query weight one (the published mrr levels)",
                          "component_weighted": "each (leakage group, label) component weight one "
                                                "(every published paired_group_bootstrap interval)",
                          "label_macro": "component-weighted mean within a label, then the mean over "
                                         "labels (every published estimand_two_stage interval)",
                          "how": "a per-query weight u whose weighted mean is the estimand, applied to "
                                 "the rank statistic and to both sides of the trial list alike"},
            "resamplings": {
                "group_bootstrap": f"paired_group_bootstrap's own draws: {GROUP_REPLICATES} replicates, "
                                   f"seed {GROUP_SEED}, leakage groups with replacement, the {L} labels "
                                   f"held fixed",
                "two_stage": f"the builder's v1.component_metric_values and v1.run_bootstrap through "
                             f"estimand_two_stage_v3's constant redirect: {TWO_STAGE_REPLICATES} "
                             f"replicates, seed {TWO_STAGE_SEED}, {L} label occurrences with replacement "
                             f"then each occurrence's components with replacement",
                "both_estimands_off_the_same_draws": ("the component-weighted statistic under the "
                                                      "two-stage draws is the ratio of the macro of "
                                                      "n_l * v to the macro of n_l, two extra metric "
                                                      "columns of the builder's own run, so no "
                                                      "transcription of its generator is involved"),
                "two_stage_diagnostics": diagnostics,
                "min_cllr_interval": f"the first {LLR_REPLICATES} rows of the same group draw matrix, so "
                                     f"the min_cllr interval is paired with the MRR one and nested in "
                                     f"it; 400 rather than {GROUP_REPLICATES} because each replicate "
                                     f"re-pools the whole trial list",
                "note": "an estimand has one point estimate and a resampling only has an interval, so "
                        "the four cells of the 2x2 carry two points and four intervals",
                "published_cells": "every published paired_group_bootstrap interval is the "
                                   "(component_weighted, group_bootstrap) cell; every published "
                                   "estimand_two_stage interval is the (label_macro, two_stage) cell; "
                                   "the other two cells have no published counterpart"},
            "detection_statistics": {
                "trials": "every (query, label) pair, scored leave-group-out exactly as the retrieval "
                          "protocol scores it; target when the label is the query's own",
                "auc": "weighted probability that a target trial outranks a non-target one, ties at one half",
                "cllr": "(1/2ln2)[E_tar log(1+e^-s) + E_non log(1+e^s)], reading the score as a "
                        "natural-log likelihood ratio; these scores were never built to be likelihood "
                        "ratios, so a raw cllr near 1 reports a wrong scale, not an uninformative system",
                "min_cllr": "the same cost after the optimal monotone (PAV) calibration: the "
                            "DISCRIMINATION term, invariant to any global monotone map of the scores",
                "calibration_loss": "cllr - min_cllr, the calibration term",
                "why": "MRR is invariant to any per-query monotone map and min_cllr is not, so the two "
                       "can move in opposite directions; that is the reading this arm exists to expose",
                "precision": "both readings are computed from the float32 cast of the scores that "
                             "ranks_of ranks, so no float32/float64 asymmetry remains; the number of "
                             "scores shared by trials of both classes is recorded per system"},
            "margin": {"value": MARGIN,
                       "role": "the non-inferiority margin this repository declares for MRR, carried "
                               "over to min_cllr as a DECLARED band and not an established margin; a "
                               "change within the band is unchanged, and a direction is established "
                               "only when the whole 95% interval clears the band"},
            "reading_rule": (
                "1. the 2x2 is a measurement, not a test: no verdict beyond the decomposition, no "
                "excludes_zero on its cells, no width factor offered as reusable. "
                "2. against the prototype of the same space under the same estimand, MRR up with "
                "min_cllr down is an improvement, MRR up with min_cllr up is a ranking gain bought with "
                "a calibration loss, MRR down with min_cllr down is a calibration gain bought with a "
                "ranking loss, MRR down with min_cllr up is no gain; a change of at most 0.005 in "
                "either quantity is unchanged within the declared band; an improvement is established "
                "only when the whole 95% interval of both differences clears the band, and otherwise is "
                "reported as a point comparison; a disagreement between estimands is named, not "
                "resolved. 3. AUC, cllr and calibration_loss are description."),
            "gate_placement": ("the gates that abort fire the moment the number they check exists (the "
                               "pinned expectations before the corpus is read, the shape and bands after "
                               "it, unit rows when a space is built, a published MRR when a system is "
                               "scored); every check that cannot fire until the expensive work is done is "
                               "recorded in this payload, which is written before a failure exits "
                               "non-zero")},
        "checks": {"against_published_mrr": published_mrr, "by_space": space_checks,
                   "published_two_by_two_cells": published_checks, "late_gates": late},
        "gap_1_width_factor": {"by_contrast": two_by_two, "summary": width_summary},
        "gap_2_calibration": {
            "systems": {name: {"mrr_by_estimand": {e: round(float(mrr_by[name][e]), 5) for e in ESTIMANDS},
                               "detection_by_estimand": {e: {k: round(float(v), 5)
                                                             for k, v in det_by[name][e].items()}
                                                         for e in ESTIMANDS},
                               "trials": trials_by[name]} for name in systems},
            "against_the_prototype": calibration,
            "summary": {"established_improvements": improvements,
                        "ranking_gains_bought_with_a_calibration_loss": ranking_gains,
                        "disagreeing_estimands": [f"{s}/{c}" for s in SPACES for c in CALIBRATIONS
                                                  if not calibration[s][c]["estimands_agree"]],
                        "reading": ("a calibration counts as an improvement only where it lowers "
                                    "min_cllr as well as raising MRR, with both component-weighted "
                                    "intervals clear of the declared 0.005 band; a calibration that "
                                    "raises MRR while raising min_cllr is a ranking gain bought with a "
                                    "calibration loss, which was named as an outcome before the run")}},
        "privacy": ("aggregate only: per-system and per-fold scalars over the five spaces and three "
                    "scorers, and label and query COUNTS per songs-per-label band; no per-label array, "
                    "no band-level distribution, no song, label, artist, title, lyric text or vector"),
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / OUT_NAME
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nestablished improvements: {improvements or 'none'}", flush=True)
    print(f"ranking gains bought with a calibration loss: {ranking_gains or 'none'}", flush=True)
    print(f"wrote {path}  ({payload['minutes']} min)", flush=True)
    if failed:
        raise SystemExit(f"the payload was written, and these late gates failed: "
                         f"{ {k: late[k] for k in failed} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
