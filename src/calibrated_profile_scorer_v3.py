#!/usr/bin/env python3
"""Divide by the label's resultant length, not by a norm that carries sampling noise.

QUESTION. The protocol scores a query q against label l by the prototype q.m_l / ||m_l||, m_l
the protocol-weighted mean of the label's songs outside q's leakage group. Every song vector is
a unit vector, so the numerator is EXACTLY exemplar_mean, the protocol-weighted mean cosine of
exemplar_vs_prototype_v3, and prototype and exemplar_mean differ by exactly one scalar per
(query, label): 1 / ||m_l||. That scalar is where label size enters the scorer. The numerator
carries no sample size -- E[q.m_l] = r_l (q.u_l), with mu_l = E x the label's mean vector,
r_l = ||mu_l|| its resultant length and u_l = mu_l / r_l -- while the divisor does:
E||m_l||^2 = r_l^2 + s^2 W_l, with W_l = sum_i w_i^2 / (sum_i w_i)^2 the weight-aware inverse
sample size and s^2 the within-label variance of one song. A label with few songs is divided by
an inflated number and its scores are deflated. Under raw words the prototype reads 0.0958,
0.2738, 0.5290 and 0.4396 across the 5-9, 10-19, 20-49 and 50-up bands of the SONGS-PER-LABEL
banding (the top band is 3 labels and 238 queries, and the curve is not monotone there). The
principled fix divides by an estimate of the label's TRUE resultant length instead:

    S_C(q, l) = q.m_l / sqrt(r_tilde_l^2).

WHAT r_hat^2 IS, AND WHAT IS THEREFORE NOT AN EMPIRICAL QUESTION. r_hat^2 is the component-level
U-statistic, BY DEFINITION, not a hypothesis about the profile norm. Once it is defined that way
the decomposition above is an ALGEBRAIC IDENTITY, derived independently and verified numerically
to 1.8e-16:

    s2_gram      = 1 - C_bar * G/(G-1) + S/(G(G-1))
    s2_predicted = (1 - C_bar) * (1 + K/(G(G-1)))
    S            = nu*K + sum_c (1 - 1/k_c) chat_c
    gap          = [ K(nu-1) + sum_c (1 - 1/k_c)(chat_c - C_bar) ] / (G(G-1))

with G the label's component count, k_c a component's song count, K = sum_c 1/k_c, nu the mean
squared row norm, chat_c the within-component mean cosine and C_bar the between-component mean
cosine. Singleton components contribute exactly nothing, because 1 - 1/k_c = 0, and this corpus is
almost all singletons: 6,688 of its 6,889 components hold one song (97.1%), the mean component
holds 1.048 songs, and 142 of the 226 labels have NO multi-song component at all, so for a clear
majority of labels the gap is EXACTLY ZERO at unit rows. Over the 226 labels the loose bound
2(G_l - K_l)/(G_l(G_l - 1)) on |gap_l| has median 0.00000 and 90th percentile 0.00350, and only 6
labels have a loose bound that could even in principle exceed 0.02. Those five figures are the
corpus's own structure, computed outside this file from the label index, the group ids and the
protocol weights alone (scratchpad/exact_identity_bound.py, 2026-09-23); nothing here recomputes
them. So r_hat^2 is NOT framed here as validated by any such check, and no such check is run: an
identity does not need confirming on data. The empirical questions are the four this file does
answer: the rate of r_hat^2 <= 0, the size of the empirical-Bayes shrinkage weight b, the bias of
the variance-stabilised form at these n and d (the synthetic test), and whether S_C flattens the
band gradient.

DESIGN. Five spaces, folds, transforms, leave-group-out and component weights exactly as in
label_size_calibration_v3 and attention_exemplar_v3: raw BGE-M3 song centroids; the same after
within-author whitening fitted per fold; raw jieba word TF-IDF; raw character 2-5-gram TF-IDF;
the word SVD-1024 whitened per fold. Profiles are leave-group-out over all songs, as in the
protocol; every training-fold quantity (s^2, the Z-norm cohort, the empirical-Bayes prior) is
fitted on the training folds only. The gate comes first in every space: the prototype must
reproduce 0.2997, 0.4164, 0.4963, 0.4266 and 0.5426 (5e-4 for the three raw spaces, 0.002 for the
two fold-wise ones), znorm and noise_corrected their ten published MRRs, exemplar_mean its four,
and the prototype the twenty published band MRRs of label_size_calibration.json, before any new
number is read.

THE ESTIMATOR, in three declared steps.

 1. r_hat_l^2 is the weight-aware U-statistic for r_l^2 = ||E x||^2 over DISTINCT COMPONENTS, not
    over songs. A component is a (group, label) set of near-duplicate songs and the protocol gives
    it total weight one, so its member weights sum to one and y_c = sum_{i in c} w_i x_i is the
    component's plain mean. With C components left after the query's leakage group is removed,

        r_hat^2 = ( || sum_c y_c ||^2 - sum_c ||y_c||^2 ) / (C^2 - C),

    whose first term is the profile's own squared norm, again because a component's weights sum to
    one. Taken over distinct SONGS instead it would also average within-component pairs, whose
    cosines are near 1 by construction, and so would be biased upward; that variant is not scored
    here (see NOT HERE).

 2. The scorer needs 1 / sqrt(r_hat^2), and unbiasedness does not survive that map. At n_eff
    around 5 with r^2 of order 0.03 the sampling sd of r_hat^2 is of the same order as r_hat^2
    itself, so r_hat^2 <= 0 occurs; its rate is reported per band and space. Empirical Bayes over
    the labels comes first: with a delete-one-component jackknife variance v of r_hat^2 (rescaled
    by 1/n_eff, the leading linear term of a U-statistic's variance) and a prior fitted on the
    training folds -- mean p and between-label variance tau^2 = max(0, Var_l(r_hat^2) - mean_l(v))
    --

        b = tau^2 / (tau^2 + v),   r_tilde^2 = p + b (r_hat^2 - p),   v_tilde = b^2 v.

    b is reported per band beside every S_C number, because a residual size dependence of S_C is a
    function of b and not only of the bias correction.

 3. The division is not a raw plug-in. (a) A soft positive part at the scale of the estimate's own
    sd,

        r_star^2 = ( r_tilde^2 + sqrt( (r_tilde^2)^2 + 4 v_tilde ) ) / 2,

    the positive root of r_star^2 (r_star^2 - r_tilde^2) = v_tilde. It is strictly positive for
    every input -- v_tilde > 0 wherever b > 0, and where b = 0 it returns p > 0, so no clamp and no
    positivity gate is needed -- it passes r_tilde^2 through with an offset smaller than
    v_tilde / r_tilde^2 when the estimate stands several sd above zero, and it degrades smoothly to
    sqrt(v_tilde) at zero. It is a function of the MEASURED uncertainty, unlike the published
    noise_corrected floor max(m2 - s2/n, 0.25 m2), which caps a small label's column at a constant
    ratio of the very quantity it is correcting. (b) A second-order correction of the convexity
    (Jensen) bias of x -> x^(-1/2), which would otherwise inflate exactly the small labels the
    scorer is meant to fix: E[X^(-1/2)] ~ mu^(-1/2) (1 + 3 Var(X) / (8 mu^2)), so

        kappa = 1 / (1 + 0.375 v_tilde / (r_star^2)^2),   S_C = (q.m_l) * kappa / sqrt(r_star^2).

    Written this way kappa lies in (0, 1] for every input and in [8/11, 1] wherever r_tilde^2 is
    non-negative, because the soft positive part gives r_star^2 >= sqrt(v_tilde) there; so the form
    needs no tuned constant and cannot change sign. The share of r_tilde^2 <= 0 -- the one region
    where that bound lapses -- is reported per band. The estimator's bias at this corpus's n and d
    is measured against synthetic labels with KNOWN r^2 in tests/test_calibrated_profile_scorer.py.

WHY S_C IS NOT ALSO Z-NORMALISED. znorm standardises each label's column by its impostor scores
and is therefore invariant to any per-label positive rescaling of the column; S_C differs from
exemplar_mean by a factor that is constant down a label's column except for the queries whose
leakage group touches that label, so a "bias correction then cohort normalisation" stack is a
no-op. It is not proposed, and the top-1 agreement between znorm(S_C) and znorm(prototype) is
reported as the measurement of that claim.

ESTIMANDS. Three are in play and they are never mixed. query_weighted is np.mean(1/ranks);
component_weighted weights each (leakage group, label) component one and is what
paired_group_bootstrap returns; label_macro is the component-weighted mean within a label then the
plain mean over labels. Every number carries its estimand, and no point estimate of one estimand
is printed beside an interval of another. The shared-draw bootstrap written here exists only
because a difference of two bands' MRRs needs one draw shared by both bands; with its universe set
to a single band it reproduces paired_group_bootstrap exactly, and that is checked. A label whose
groups a replicate misses entirely is dropped from that replicate's label-macro mean; the draw is
paired, so a dropped label is dropped for every scorer at once and a PAIRED label-macro difference
is unaffected, but the LEVEL of a label-macro MRR under label resampling is
estimand_two_stage_v3's two-stage bootstrap and not this one. The worst dropped share is reported.

BANDINGS. Two, never mixed, and every band claim names its banding. songs_per_label: 5-9 (19
labels, 149 queries), 10-19 (33, 477), 20-49 (171, 6,356), 50-up (3, 238). components_per_label:
4-9 (159 queries), 10-19 (648), 20-49 (6,413), 50-up (0). The reading rule is declared on
songs_per_label; components_per_label is its sensitivity.

READING RULES, fixed before the run, with margins declared.

 R1 (the headline, a superiority test with a margin). Let G(scorer, space) be the
    component-weighted MRR of the 20-49 band minus that of the 5-9 band, both of the
    SONGS-PER-LABEL banding, and P(scorer, space) the label-macro penalty, the query-weighted MRR
    minus the label-macro MRR. S_C is "a better-calibrated scorer than the published pair" in a
    space only if, for BOTH znorm and noise_corrected, the whole 95% interval of
    |G(other)| - |G(S_C)| lies above +0.005 MRR AND the whole 95% interval of P(other) - P(S_C)
    lies above +0.005 MRR. The verdict holds overall only if it holds in all five spaces;
    otherwise the failing spaces and the failing half are named. 0.005 MRR is the margin used
    elsewhere in this repo. Both intervals come from one shared leakage-group draw, so the
    gradient is a difference of differences and not two independent intervals.

 R2 (non-inferiority). S_C is non-inferior to the prototype in a space if the whole interval of
    the component-weighted (S_C - prototype) lies above -0.005, and superior if above 0. The same
    contrasts are reported against exemplar_mean, znorm and noise_corrected.

 R4 (the pre-registered null, WHICH OVERRIDES R1). If the share of own-label (query, label) pairs
    with r_hat^2 <= 0 in the 5-9 band of the songs-per-label banding exceeds 0.10, or the mean
    empirical-Bayes shrinkage weight b in that band falls below 0.25 so that shrinkage collapses
    S_C onto the pooled scorer (which is exemplar_mean up to one global constant per space and
    fold), then the honest reading is that small repertoires are UNDER-DETERMINED rather than
    mis-scored, which makes label-macro averaging a fairness choice rather than a bias fix - and
    THAT is then the section's finding. R1's gradient is built on that same 5-9 band, and R1's
    overall verdict is a conjunction over all five spaces, so wherever R4 fires the payload's
    headline IS R4's reading and the R1 verdict is emitted inside it as description only. The
    precedence is implemented in readings(); it is not left to whoever writes the section up.

CHECKS, in the order they can fire. Up front, before any scoring and so before any expensive
work: the population is 7,220 queries (one per eligible song of the 7,379 in the cleaned corpus),
226 labels, 5,875 leakage groups, 6,889 components and 1,631,720 (query, label) pairs; every label
keeps at least 3 components, which is what makes C >= 2 after the holdout and the C^2 - C
denominator exact; both bandings reproduce their published label and query counts; the generic
leave-group-out parts equal the protocol's own dense and sparse scorers (1e-9 / 1e-4). The payload
skeleton is written there, so nothing below can abort the run without leaving a file. Per space:
song vectors are unit rows (1e-9 for the two float64 fold-wise spaces, 4e-5 on the SQUARED norm
for the three float32-derived raw spaces, which is the squared-norm form of v1.fit_tfidf's own
atol=2e-5 contract - no matrix in this repo has ever been verified tighter, and the realised gap
is reported, not only gated on); the weight count equals the component count (1e-9); q.m_l, the
prototype, r_hat^2 and S_C equal brute-force loops over their definitions (1e-9); the published
MRRs and the twenty published band MRRs reproduce; the shared-draw bootstrap reproduces
paired_group_bootstrap on a single-band universe (1e-4). The payload is rewritten after every
space, so a late failure leaves the spaces already scored on disk.

NOT HERE, on purpose, to keep this file the size of every other analysis script in the repo.
(a) The gamma family S^gamma = q.m_l / ||m_l||^gamma: no reading rule turned on it, its two anchors
are exactly the exemplar_mean and prototype scorers kept here, and its own caution said an
interior maximum was not to be read as repertoire coherence anyway. (b) The U-statistic over
distinct SONGS as a second scorer: the near-duplicate channel it diagnoses is measured on the
SCORER by within_label_repeats_v3, and no rule read the difference. (c) The permuted-label arm and
its rule R3: it quadrupled the run by rescoring every space under three group-level permutations,
and label_permutation_null_v2 already answers that question for the published scorers. If it is
added back, the admissibility of the group-level shuffle must be checked BEFORE any scoring: the
shuffle preserves each label's count of leakage groups whose FIRST label it is, so a ">= 3
components per permuted label" rule is decided identically on every attempt, and a redraw loop
around it either always passes or always fails - in the second case aborting at the end of a long
run with nothing written.

    CHINESE_RAP_CORPUS=v3 python src/calibrated_profile_scorer_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from estimand_two_stage_v3 import label_macro  # noqa: E402
from exemplar_vs_prototype_v3 import setup  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import REPLICATES as BOOTSTRAP_REPLICATES  # noqa: E402
from identity_spaces_v2 import SEED as BOOTSTRAP_SEED  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402
from label_size_calibration_v3 import lgo_parts, residual_variance, scorers_from_parts, znorm  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "calibrated_profile_scorer.json"

SPACES = ("semantic_raw", "semantic_whitened", "words_raw", "chars_raw", "words_svd_whitened")
FOLDWISE = ("semantic_whitened", "words_svd_whitened")
SCORERS = ("exemplar_mean", "prototype", "znorm", "noise_corrected", "calibrated_S_C")
DIAGNOSTICS = ("r2", "r_tilde2", "shrinkage", "kappa", "n_eff")

# published corpus-v3 numbers, query-weighted MRR
EXPECTED_PROTOTYPE = {"semantic_raw": 0.2997, "semantic_whitened": 0.4164, "words_raw": 0.4963,
                      "chars_raw": 0.4266, "words_svd_whitened": 0.5426}
EXPECTED_ZNORM = {"semantic_raw": 0.2100, "semantic_whitened": 0.4144, "words_raw": 0.4868,
                  "chars_raw": 0.3266, "words_svd_whitened": 0.5577}
EXPECTED_NOISE = {"semantic_raw": 0.2635, "semantic_whitened": 0.3902, "words_raw": 0.5394,
                  "chars_raw": 0.4591, "words_svd_whitened": 0.5291}
# exemplar_vs_prototype.json covers four spaces: it has no chars_raw column
EXPECTED_EXEMPLAR_MEAN = {"semantic_raw": 0.1046, "semantic_whitened": 0.3224, "words_raw": 0.3450,
                          "words_svd_whitened": 0.4218}
# label_size_calibration.json by_band, component-weighted, SONGS-per-label banding
EXPECTED_PROTOTYPE_BANDS = {
    "semantic_raw": {"5-9": 0.1597, "10-19": 0.2806, "20-49": 0.3010, "50-up": 0.4016},
    "semantic_whitened": {"5-9": 0.3404, "10-19": 0.4152, "20-49": 0.4140, "50-up": 0.5068},
    "words_raw": {"5-9": 0.0958, "10-19": 0.2738, "20-49": 0.5290, "50-up": 0.4396},
    "chars_raw": {"5-9": 0.1123, "10-19": 0.2610, "20-49": 0.4517, "50-up": 0.3685},
    "words_svd_whitened": {"5-9": 0.3250, "10-19": 0.4530, "20-49": 0.5570, "50-up": 0.5561},
}
RAW_GAP, FOLD_GAP, BAND_GAP = 5e-4, 0.002, 0.002
TOLERANCE = {sp: (FOLD_GAP if sp in FOLDWISE else RAW_GAP) for sp in SPACES}

SONG_BANDS = (("5-9", 5, 9), ("10-19", 10, 19), ("20-49", 20, 49), ("50-up", 50, 10 ** 9))
COMPONENT_BANDS = (("4-9", 4, 9), ("10-19", 10, 19), ("20-49", 20, 49), ("50-up", 50, 10 ** 9))
EXPECTED_SONG_BAND_LABELS = {"5-9": 19, "10-19": 33, "20-49": 171, "50-up": 3}
EXPECTED_SONG_BAND_QUERIES = {"5-9": 149, "10-19": 477, "20-49": 6356, "50-up": 238}
EXPECTED_COMPONENT_BAND_QUERIES = {"4-9": 159, "10-19": 648, "20-49": 6413, "50-up": 0}
EXPECTED_POPULATION = {"queries": 7220, "labels": 226, "groups": 5875, "components": 6889,
                       "query_label_pairs": 1631720}

# A unit-row check can be no tighter than the guarantee of the code that built the matrix.
# v1.l2_normalize_dense normalises in float32, v1.fit_tfidf's own contract is
# np.allclose(row_norms, 1.0, atol=2e-5) and fit_words builds float32 with no norm check at all,
# so on the SQUARED norm the three raw spaces get 4e-5; only the two fold-wise spaces are float64
# unit_rows output and can be held to 1e-9.
UNIT_ROW_GAP = {sp: (1e-9 if sp in FOLDWISE else 4e-5) for sp in SPACES}
COUNT_GAP = 1e-9               # weights sum to one per component exactly, up to float64 rounding
BRUTE_GAP = 1e-9
BOOTSTRAP_GAP = 1e-4           # both sides are rounded to four decimals before comparison
BRUTE_QUERIES = 4
BRUTE_LABEL_STRIDE = 25
BOOTSTRAP_CHUNK = 500
DELTA = 0.375                  # 3/8, the second-order coefficient of x -> x^(-1/2)
MARGIN = 0.005                 # the repo's MRR margin
UNDER_DETERMINED_RATE = 0.10   # R4: share of own-label pairs with r_hat^2 <= 0 in the 5-9 band
COLLAPSE_WEIGHT = 0.25         # R4: mean empirical-Bayes shrinkage weight in that band
MIN_LABELS_FOR_QUANTILES = 10  # below this a band's quantiles would list its labels
GRADIENT_BANDS = ("20-49", "5-9")      # G = MRR(first) - MRR(second), songs_per_label banding
PUBLISHED_PAIR = ("znorm", "noise_corrected")

READING_RULE = (
    "R1: S_C is better calibrated than the published pair in a space only if, for BOTH znorm and "
    "noise_corrected, the whole 95% interval of |their 20-49 minus 5-9 band gradient| minus |S_C's| "
    f"lies above +{MARGIN} MRR AND the whole 95% interval of (their label-macro penalty minus S_C's) "
    f"lies above +{MARGIN}; bands are of the songs_per_label banding, gradients are component-weighted, "
    "the penalty is query-weighted minus label-macro, and both intervals come from one shared "
    "leakage-group draw. The verdict holds overall only if it holds in all five spaces. "
    f"R2: S_C is non-inferior to a comparison scorer in a space if the whole interval of the "
    f"component-weighted difference lies above -{MARGIN}, superior if above 0. "
    "R4 OVERRIDES R1: wherever the small-repertoire columns come out under-determined by the "
    "thresholds below, the section's finding is R4's reading and the R1 verdict is description only."
)
NULL_READING = (
    "Pre-registered before the run: if the share of own-label (query, label) pairs with r_hat^2 <= 0 "
    f"in the 5-9 band of the songs_per_label banding exceeds {UNDER_DETERMINED_RATE}, or the mean "
    f"empirical-Bayes shrinkage weight in that band falls below {COLLAPSE_WEIGHT} so that shrinkage "
    "collapses S_C onto the pooled scorer (exemplar_mean up to one global constant per space and "
    "fold), then the honest reading is that small repertoires are UNDER-DETERMINED rather than "
    "mis-scored, which makes label-macro averaging a fairness choice rather than a bias fix - and "
    "that is then the section's finding, in preference to R1's."
)
ZNORM_NOOP = (
    "znorm standardises each label's column by its impostor scores, so it is invariant to any "
    "per-label positive rescaling of that column; S_C differs from exemplar_mean by a factor that is "
    "constant down a label's column except for the queries whose leakage group touches the label, so "
    "a 'bias correction then cohort normalisation' stack is a no-op. It is not proposed; the top-1 "
    "agreement between znorm(S_C) and znorm(prototype) is reported as the measurement of that claim."
)
BOOTSTRAP_UNIVERSE = ("all leakage groups, one draw shared by every band, so that a band gradient is "
                      "a difference of differences; this is wider than paired_group_bootstrap's "
                      "band-only universe, which label_size_calibration.json's band intervals use")
NOT_INCLUDED = ("the gamma family S^gamma = q.m_l/||m_l||^gamma (no rule read it; its anchors are the "
                "exemplar_mean and prototype scorers here), the U-statistic over distinct songs as a "
                "second scorer (within_label_repeats_v3 measures the near-duplicate channel on the "
                "scorer), and the permuted-label arm with its rule R3 (label_permutation_null_v2 "
                "answers that for the published scorers)")


# ------------------------------------------------------------------ small helpers
def quantiles(a, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size == 0:
        return {f"q{int(p * 100):02d}": None for p in qs}
    return {f"q{int(p * 100):02d}": round(float(np.quantile(a, p)), 5) for p in qs}


def component_ids(label_index: np.ndarray, group_ids: np.ndarray):
    """Component id per song, and the label and group of each component. A component is a
    (group, label) set of near-duplicate songs, the unit the protocol gives total weight one."""
    key: dict[tuple[int, int], int] = {}
    comp_of = np.zeros(len(label_index), dtype=np.int64)
    for i, (g, l) in enumerate(zip(group_ids.tolist(), label_index.tolist())):
        comp_of[i] = key.setdefault((int(g), int(l)), len(key))
    comp_label = np.zeros(len(key), dtype=np.int64)
    comp_group = np.zeros(len(key), dtype=np.int64)
    for (g, l), c in key.items():
        comp_label[c], comp_group[c] = l, g
    return comp_of, comp_label, comp_group


def row_squared_norms(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    return np.einsum("ij,ij->i", matrix, matrix)


def component_means(x, comp_of, weights, component_count):
    """y_c = sum_{i in c} w_i x_i, the component's plain mean because its weights sum to one."""
    mixer = sparse.csr_matrix((weights, (comp_of, np.arange(len(comp_of)))),
                              shape=(component_count, len(comp_of)))
    mass = np.asarray(mixer.sum(axis=1)).ravel()
    if float(np.abs(mass - 1.0).max()) > COUNT_GAP:
        raise RuntimeError("a component does not carry total weight one")
    y = mixer @ x
    return y.tocsr() if sparse.issparse(y) else y


def dense_row(x, index) -> np.ndarray:
    return np.asarray(x[index].todense()).ravel() if sparse.issparse(x) else np.asarray(x[index])


# ------------------------------------------------------------------ leave-group-out ingredients
def component_parts(x, label_index, group_ids, weights, label_count, queries):
    """Per (query, label), leave-group-out: how many components the label keeps, and the sum over
    them of their squared component-mean norms.

    The query's leakage group holds at most one component of any label, so removing the group
    removes at most one component per label and the corrections below are exact.
    """
    comp_of, comp_label, comp_group = component_ids(label_index, group_ids)
    count = len(comp_label)
    y = component_means(x, comp_of, weights, count)
    y_norm2 = row_squared_norms(y)
    ncomp = np.repeat(np.bincount(comp_label, minlength=label_count)[None, :], len(queries), axis=0)
    q_sum = np.repeat(np.bincount(comp_label, weights=y_norm2, minlength=label_count)[None, :],
                      len(queries), axis=0)
    comps_by_group: dict[int, list[int]] = defaultdict(list)
    for c in range(count):
        comps_by_group[int(comp_group[c])].append(c)
    for row, query in enumerate(queries.tolist()):
        for c in comps_by_group[int(group_ids[query])]:
            label = int(comp_label[c])
            ncomp[row, label] -= 1
            q_sum[row, label] -= y_norm2[c]
    return ncomp, q_sum


def resultant_squared(norm2, q_sum, ncomp):
    """The weight-aware U-statistic for r^2 = ||E x||^2 over DISTINCT COMPONENTS:
    (||sum_c y_c||^2 - sum_c ||y_c||^2) / (C^2 - C). The first term is the profile's own squared
    norm, because a component's weights sum to one. C >= 2 is checked once, up front, from the
    label geometry (every label keeps at least three components, and a leakage group holds at most
    one component of any label)."""
    c = np.asarray(ncomp, dtype=np.float64)
    return (norm2 - q_sum) / (c * c - c)


def resultant_prior(x, label_index, group_ids, weights, label_count, train):
    """The empirical-Bayes prior for r^2, fitted on the TRAINING folds only.

    Per label: the component U-statistic on its training-fold components, a delete-one-component
    jackknife variance of it in closed form, and the implied linear variance component
    zeta1 = C Var / 4, which gives the per-(query, label) variance 4 zeta1 / n_eff. The prior mean
    and the between-label variance are a PLAIN mean and variance over the labels with at least
    three training components -- the prior is a prior over labels, not over queries, which is why
    it is unweighted where every other mean in this file is component-weighted. A label with fewer
    components takes the pooled zeta1 and does not enter the prior.
    """
    rows = np.flatnonzero(train)
    xt = x[rows]
    lt, gt, wt = label_index[rows], group_ids[rows], weights[rows]
    comp_of, comp_label, _ = component_ids(lt, gt)
    count = len(comp_label)
    y = component_means(xt, comp_of, wt, count)      # raises if a leakage group spanned folds
    y_norm2 = row_squared_norms(y)
    member = sparse.csr_matrix((np.ones(count), (comp_label, np.arange(count))),
                               shape=(label_count, count))
    profile = member @ y
    if sparse.issparse(profile):
        profile = profile.tocsr()
    s_dot_y = np.zeros(count)
    n2 = np.zeros(label_count)
    for label in range(label_count):
        idx = np.flatnonzero(comp_label == label)
        row = profile[label]
        if sparse.issparse(profile):
            n2[label] = float(row.multiply(row).sum())
            if len(idx):
                s_dot_y[idx] = np.asarray((y[idx] @ row.T).todense(), dtype=np.float64).ravel()
        else:
            n2[label] = float(row @ row)
            if len(idx):
                s_dot_y[idx] = y[idx] @ row
    q_sum = np.bincount(comp_label, weights=y_norm2, minlength=label_count)
    cnt = np.bincount(comp_label, minlength=label_count)
    c = cnt.astype(np.float64)
    two, three = cnt >= 2, cnt >= 3
    if int(np.sum(three)) < 2:
        raise RuntimeError("fewer than two labels have three training-fold components; no prior")
    u_full = np.zeros(label_count)
    u_full[two] = ((n2 - q_sum) / np.where(two, c * c - c, 1.0))[two]
    # delete-one-component jackknife, in closed form:
    # ||S - y_j||^2 - (Q - ||y_j||^2) = (||S||^2 - Q) - 2 S.y_j + 2 ||y_j||^2, over (C-1)(C-2)
    cl = c[comp_label]
    numerator = n2[comp_label] - q_sum[comp_label] - 2.0 * s_dot_y + 2.0 * y_norm2
    den = (cl - 1.0) * (cl - 2.0)
    keep = den > 0
    u_minus = np.zeros(count)
    u_minus[keep] = numerator[keep] / den[keep]
    kept = np.bincount(comp_label, weights=keep.astype(np.float64), minlength=label_count)
    total = np.bincount(comp_label, weights=np.where(keep, u_minus, 0.0), minlength=label_count)
    bar = np.zeros(label_count)
    bar[three] = total[three] / kept[three]
    spread = np.bincount(comp_label, weights=np.where(keep, (u_minus - bar[comp_label]) ** 2, 0.0),
                         minlength=label_count)
    v_jack = np.zeros(label_count)
    v_jack[three] = (c[three] - 1.0) / c[three] * spread[three]
    zeta1 = np.zeros(label_count)
    zeta1[three] = v_jack[three] * c[three] / 4.0
    positive = zeta1[zeta1 > 0]
    if positive.size == 0:
        raise RuntimeError("every jackknife variance is degenerate; there is no variance scale")
    zeta1 = np.where(zeta1 > 0, zeta1, float(np.median(positive)))
    prior_mean = float(np.mean(u_full[three]))
    if prior_mean <= 0.0:
        raise RuntimeError("the pooled resultant length is not positive; nothing to shrink toward")
    between = max(0.0, float(np.var(u_full[three], ddof=1)) - float(np.mean(v_jack[three])))
    return {"prior_mean": prior_mean, "between_variance": between, "zeta1": zeta1, "u_full": u_full,
            "components_train": cnt, "labels_off_prior": int(np.sum(~three)),
            "training_median_r2": round(float(np.median(u_full[three])), 6),
            "training_negative_r2_share": round(float(np.mean(u_full[three] <= 0.0)), 4)}


# ------------------------------------------------------------------ the calibrated scorer
def calibrated_scores(dots, counts, r2, ncomp, prior):
    """S_C = q.m_l * kappa / sqrt(r_star^2): shrink toward the training-fold prior, take a soft
    positive part at the scale of the estimate's own sd, then correct the convexity bias of
    x -> x^(-1/2) to second order. r_star^2 > 0 for every input, so there is no clamp: where the
    shrinkage weight is positive v_tilde is positive too, and where it is zero r_tilde^2 is the
    prior mean, which resultant_prior has already refused to return non-positive."""
    n_eff = np.asarray(ncomp, dtype=np.float64)
    variance = 4.0 * prior["zeta1"][None, :] / n_eff
    tau2 = prior["between_variance"]
    shrinkage = tau2 / (tau2 + variance)
    r_tilde2 = prior["prior_mean"] + shrinkage * (r2 - prior["prior_mean"])
    v_tilde = shrinkage ** 2 * variance
    r_star2 = 0.5 * (r_tilde2 + np.sqrt(r_tilde2 * r_tilde2 + 4.0 * v_tilde))
    kappa = 1.0 / (1.0 + DELTA * v_tilde / (r_star2 * r_star2))
    scores = (dots / counts) * kappa / np.sqrt(r_star2)
    return scores, {"shrinkage": shrinkage, "r_tilde2": r_tilde2, "r_star2": r_star2,
                    "kappa": kappa, "variance": v_tilde}


def brute_scores(x, query, group_ids, label_index, weights, label, prior) -> dict:
    """q.m_l, the prototype, r_hat^2 and S_C, literally, for one (query, label). One densification
    of the label's rows serves all four."""
    rows = [s for s in np.flatnonzero(label_index == label) if group_ids[s] != group_ids[query]]
    by_group: dict[int, list[int]] = defaultdict(list)
    for s in rows:
        by_group[int(group_ids[s])].append(s)
    units = [sum(weights[s] * dense_row(x, s) for s in members) for members in by_group.values()]
    count = float(len(units))
    total = sum(units)
    squares = float(sum(float(u @ u) for u in units))
    r2 = (float(total @ total) - squares) / (count * count - count)
    q = dense_row(x, query)
    exemplar = float(q @ total) / count
    variance = 4.0 * float(prior["zeta1"][label]) / count
    tau2 = prior["between_variance"]
    b = tau2 / (tau2 + variance)
    r_tilde2 = prior["prior_mean"] + b * (r2 - prior["prior_mean"])
    v_tilde = b * b * variance
    r_star2 = 0.5 * (r_tilde2 + np.sqrt(r_tilde2 * r_tilde2 + 4.0 * v_tilde))
    kappa = 1.0 / (1.0 + DELTA * v_tilde / (r_star2 * r_star2))
    return {"exemplar_mean": exemplar,
            "prototype": float(q @ total) / float(np.linalg.norm(total)),
            "r2": r2,
            "calibrated_S_C": exemplar * kappa / float(np.sqrt(r_star2))}


# ------------------------------------------------------------------ estimands and resampling
def three_estimands(rr, weights, label_index, mask=None) -> dict:
    """query-weighted, component-weighted and label-macro MRR, each labelled.

    Restricted to a mask, the label-macro mean is taken over the labels present in the mask; the
    bands of this file select whole labels, so that is the band's label-macro.
    """
    if mask is None:
        mask = np.ones(len(rr), dtype=bool)
    labels = np.unique(label_index[mask])
    remapped = np.searchsorted(labels, label_index[mask])
    macro = label_macro(rr[mask][:, None], weights[mask], remapped, len(labels))[0]
    return {"query_weighted": round(float(np.mean(rr[mask])), 4),
            "component_weighted": round(weighted_mean(rr, weights, mask), 4),
            "label_macro": round(float(macro), 4)}


def shared_draw_bootstrap(rr_by_system, weights, group_ids, label_index, universe, masks,
                          full_estimands=(), replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                          chunk=BOOTSTRAP_CHUNK):
    """identity_spaces_v2.paired_group_bootstrap's design with ONE draw shared by several masks.

    The leakage groups of `universe` are drawn with replacement, `replicates` times, with the same
    seed, so with universe == mask and a single mask this reproduces paired_group_bootstrap exactly
    (checked in main). Sharing the draw is what gives a difference of two bands' MRRs -- a band
    gradient -- an interval. Every returned quantity names its estimand; the component-weighted one
    is paired_group_bootstrap's own and is returned for every mask, while query_weighted and
    label_macro are returned only for the masks named in `full_estimands`, because no rule here
    reads a band's label-macro interval.

    label_macro under this resampling: a replicate that draws none of a label's groups cannot score
    that label, and the label is dropped from that replicate's mean (the worst share is returned).
    The draw is paired, so a dropped label is dropped for every system at once and a PAIRED
    label-macro difference is unaffected; the LEVEL of a label-macro MRR under label resampling is
    estimand_two_stage_v3's two-stage bootstrap, not this one.
    """
    group_list = np.unique(group_ids[universe])
    position = {int(g): i for i, g in enumerate(group_list.tolist())}
    ngroups = len(group_list)
    draws = np.random.default_rng(seed).integers(0, ngroups, size=(replicates, ngroups))
    names = list(rr_by_system)
    out: dict[str, dict] = {}
    lost = 0.0
    for mask_name, mask in masks.items():
        if not np.any(mask):
            out[mask_name] = {}
            continue
        if not np.all(universe[mask]):
            raise RuntimeError(f"mask {mask_name} is not inside the bootstrap universe")
        index = np.asarray([position[int(g)] for g in group_ids[mask].tolist()], dtype=np.int64)
        mass = weights[mask]
        weight_sum = np.bincount(index, weights=mass, minlength=ngroups)
        denominators = weight_sum[draws].sum(axis=1)
        if np.any(denominators <= 0):
            raise RuntimeError(f"mask {mask_name}: a replicate drew no weight")
        entry: dict[str, dict] = {"component_weighted": {}}
        for name in names:
            numerator = np.bincount(index, weights=rr_by_system[name][mask] * mass, minlength=ngroups)
            entry["component_weighted"][name] = {
                "point": float(numerator.sum() / weight_sum.sum()),
                "replicates": numerator[draws].sum(axis=1) / denominators}
        if mask_name in full_estimands:
            plain = np.bincount(index, weights=np.ones(int(mask.sum())), minlength=ngroups)
            plain_den = plain[draws].sum(axis=1)
            entry["query_weighted"] = {}
            for name in names:
                numerator = np.bincount(index, weights=rr_by_system[name][mask], minlength=ngroups)
                entry["query_weighted"][name] = {
                    "point": float(numerator.sum() / plain.sum()),
                    "replicates": numerator[draws].sum(axis=1) / plain_den}
            labels = np.unique(label_index[mask])
            remapped = np.searchsorted(labels, label_index[mask])
            width = len(labels)
            flat = index * width + remapped
            weight_gl = np.bincount(flat, weights=mass,
                                    minlength=ngroups * width).reshape(ngroups, width)
            numerator_gl = {name: np.bincount(flat, weights=rr_by_system[name][mask] * mass,
                                              minlength=ngroups * width).reshape(ngroups, width)
                            for name in names}
            macro = {name: np.zeros(replicates) for name in names}
            dropped = np.zeros(replicates)
            for start in range(0, replicates, chunk):
                block = draws[start:start + chunk]
                offsets = np.arange(len(block))[:, None] * ngroups
                multiplicity = np.bincount(
                    (block + offsets).ravel(),
                    minlength=len(block) * ngroups).reshape(len(block), ngroups).astype(np.float64)
                drawn_weight = multiplicity @ weight_gl
                alive = drawn_weight > 0
                dropped[start:start + len(block)] = 1.0 - alive.sum(axis=1) / float(width)
                for name in names:
                    got = multiplicity @ numerator_gl[name]
                    ratio = np.where(alive, got / np.where(alive, drawn_weight, 1.0), 0.0)
                    macro[name][start:start + len(block)] = ratio.sum(axis=1) / alive.sum(axis=1)
            point = label_macro(np.stack([rr_by_system[name][mask] for name in names], axis=1),
                                mass, remapped, width)
            entry["label_macro"] = {name: {"point": float(point[i]), "replicates": macro[name]}
                                    for i, name in enumerate(names)}
            lost = max(lost, float(dropped.max()))
        out[mask_name] = entry
    return out, lost


def contrast(left_name, right_name, left, right, estimand, margin=MARGIN) -> dict:
    """A paired difference, its interval and the margin verdicts, all in one estimand."""
    difference = left["replicates"] - right["replicates"]
    point = left["point"] - right["point"]
    low, high = np.percentile(difference, [2.5, 97.5])
    return {"system": left_name, "minus": right_name, "estimand": estimand,
            "difference": round(float(point), 4),
            "ci95": [round(float(low), 4), round(float(high), 4)],
            "excludes_zero": bool(low > 0 or high < 0),
            "superior": bool(low > 0.0),
            "non_inferior_at_margin": bool(low > -margin),
            "clears_margin": bool(low > margin),
            "bootstrap_universe": BOOTSTRAP_UNIVERSE}


# ------------------------------------------------------------------ spaces
def space_matrix(name, fold_index, dense, words, chars, reduced, label_index, weights, fold):
    """The space a fold's queries are scored in. Label-dependent whitening is refitted on the
    training folds with identity_probe_v2's own generator seeds; the SVD is label-free and is
    passed in already fitted per fold."""
    if name == "semantic_raw":
        return dense
    if name == "words_raw":
        return words
    if name == "chars_raw":
        return chars
    train = fold != fold_index
    base = dense if name == "semantic_whitened" else reduced[fold_index]
    mean, matrix, _ = fit_transform("within_author_whitening", base[train], label_index[train],
                                    weights[train], np.random.default_rng(SEED + fold_index))
    return unit_rows((base - mean) @ matrix.T)


def score_space(name, li, gi, w, L, fold, dense, words, chars, reduced):
    """Every scorer in one space, each fold's queries scored in that fold's space.

    Returns the score matrices, the per-(query, label) estimator diagnostics, the per-fold
    information, and the realised gaps of this space's checks.
    """
    n = len(li)
    everything = np.arange(n)
    scores = {key: np.zeros((n, L)) for key in SCORERS}
    diagnostics = {key: np.zeros((n, L)) for key in DIAGNOSTICS}
    agreement = np.zeros(n)
    fold_info: dict[str, dict] = {}
    audit: dict[str, dict] = {"unit_row_gap": {}, "weight_count_minus_component_count": {},
                             "brute_force": {}}
    cache: dict[object, tuple] = {}
    for k in range(FOLDS):
        x = space_matrix(name, k, dense, words, chars, reduced, li, w, fold)
        train, test = fold != k, np.flatnonzero(fold == k)
        key = k if name in FOLDWISE else "raw"       # a raw space is the same in every fold
        if key not in cache:
            unit_gap = float(np.abs(row_squared_norms(x) - 1.0).max())
            audit["unit_row_gap"][str(key)] = unit_gap
            if unit_gap > UNIT_ROW_GAP[name]:
                raise SystemExit(f"{name} fold {k}: song rows are not unit rows to "
                                 f"{UNIT_ROW_GAP[name]:.0e} on the squared norm ({unit_gap:.2e})")
            dots, norm2, counts = lgo_parts(x, li, gi, w, L, everything)
            ncomp, q_sum = component_parts(x, li, gi, w, L, everything)
            count_gap = float(np.abs(counts - ncomp).max())
            audit["weight_count_minus_component_count"][str(key)] = count_gap
            if count_gap > COUNT_GAP:
                raise SystemExit(f"{name} fold {k}: the weight count is not the component count "
                                 f"({count_gap:.2e}), so the C^2 - C denominator would be wrong")
            cache[key] = (dots, norm2, counts, ncomp, q_sum)
        dots, norm2, counts, ncomp, q_sum = cache[key]
        r2 = resultant_squared(norm2, q_sum, ncomp)
        prior = resultant_prior(x, li, gi, w, L, train)
        calibrated, parts = calibrated_scores(dots, counts, r2, ncomp, prior)
        s2 = residual_variance(x, dots, norm2, counts, li, w, train)
        prototype, noise_corrected, floor_hits = scorers_from_parts(dots, norm2, counts, s2)
        cohort = znorm(prototype, li, train, test)
        agreement[test] = (np.argmax(cohort, axis=1)
                           == np.argmax(znorm(calibrated, li, train, test), axis=1)).astype(np.float64)
        scores["exemplar_mean"][test] = (dots / counts)[test]
        scores["prototype"][test] = prototype[test]
        scores["noise_corrected"][test] = noise_corrected[test]
        scores["znorm"][test] = cohort
        scores["calibrated_S_C"][test] = calibrated[test]
        for slot, array in (("r2", r2), ("r_tilde2", parts["r_tilde2"]),
                            ("shrinkage", parts["shrinkage"]), ("kappa", parts["kappa"]),
                            ("n_eff", np.asarray(ncomp, dtype=np.float64))):
            diagnostics[slot][test] = array[test]
        fold_info[str(k)] = {
            "within_label_residual_variance_s2": round(s2, 4),
            "noise_corrected_floor_share": round(floor_hits / float(dots.size), 4),
            "prior_mean_r2": round(prior["prior_mean"], 6),
            "between_label_variance_of_r2": round(prior["between_variance"], 8),
            "training_median_r2": prior["training_median_r2"],
            "training_negative_r2_share": prior["training_negative_r2_share"],
            "labels_off_the_prior": prior["labels_off_prior"],
        }
        print(f"  fold {k}: s2 {s2:.4f}, prior r^2 {prior['prior_mean']:.5f}, "
              f"tau^2 {prior['between_variance']:.3e}", flush=True)
        if k == 0:
            gaps = {"exemplar_mean": 0.0, "prototype": 0.0, "r2": 0.0, "calibrated_S_C": 0.0}
            for query in range(BRUTE_QUERIES):
                for label in range(0, L, BRUTE_LABEL_STRIDE):
                    reference = brute_scores(x, query, gi, li, w, label, prior)
                    got = {"exemplar_mean": float(dots[query, label] / counts[query, label]),
                           "prototype": float(dots[query, label] / np.sqrt(norm2[query, label])),
                           "r2": float(r2[query, label]),
                           "calibrated_S_C": float(calibrated[query, label])}
                    for what in gaps:
                        gaps[what] = max(gaps[what], abs(got[what] - reference[what]))
            audit["brute_force"] = {what: float(gap) for what, gap in gaps.items()}
            for what, gap in gaps.items():
                if gap > BRUTE_GAP:
                    raise SystemExit(f"{name}: {what} differs from its definition ({gap:.2e})")
            print(f"  brute force on {BRUTE_QUERIES} queries: worst gap "
                  f"{max(gaps.values()):.2e}", flush=True)
        del x
    diagnostics["znorm_top1_agreement"] = agreement
    return scores, diagnostics, fold_info, audit


# ------------------------------------------------------------------ reporting
def estimator_block(diagnostics, mask, label_index, labels_in_band) -> dict:
    """The estimator's own diagnostics inside a band, aggregate only.

    n_eff is the label's leave-group-out component count, constant down its column but for the
    query's own group, and r^2, b and kappa are nearly so; quantiles of them over a band are
    therefore quantiles of a PER-LABEL quantity, and a band holding a handful of labels would
    publish a sorted list of those labels. The songs_per_label 50-up band holds three labels and
    the artist list is public as names, so below MIN_LABELS_FOR_QUANTILES only the means R4 reads
    are emitted.
    """
    own = np.arange(len(label_index))[mask], label_index[mask]
    block = {
        "non_positive_r2_share_own_label": round(float(np.mean(diagnostics["r2"][own] <= 0.0)), 4),
        "non_positive_r_tilde2_share_own_label": round(
            float(np.mean(diagnostics["r_tilde2"][own] <= 0.0)), 4),
        "mean_shrinkage_weight_own_label": round(float(np.mean(diagnostics["shrinkage"][own])), 4),
        "mean_kappa_own_label": round(float(np.mean(diagnostics["kappa"][own])), 4),
        "mean_n_eff_own_label": round(float(np.mean(diagnostics["n_eff"][own])), 2),
    }
    if labels_in_band >= MIN_LABELS_FOR_QUANTILES:
        block["r2_own_label"] = quantiles(diagnostics["r2"][own])
        block["shrinkage_weight_own_label"] = quantiles(diagnostics["shrinkage"][own])
        block["n_eff_own_label"] = quantiles(diagnostics["n_eff"][own])
    else:
        block["quantiles_suppressed"] = (
            f"fewer than {MIN_LABELS_FOR_QUANTILES} labels in this band: quantiles of a per-label "
            "quantity would amount to a per-label listing, so only means are reported here")
    return block


def readings(results, scored) -> tuple:
    """R1's verdict, then R4's declared precedence over it."""
    complete = sorted(scored) == sorted(SPACES)
    better = [sp for sp in scored if results[sp]["R1_better_calibrated_here"]]
    under = [sp for sp in scored if results[sp]["R4_small_band_under_determined"]]
    if complete and len(better) == len(SPACES):
        r1 = ("R1 met: S_C is a better-calibrated scorer than the published pair in every space - a "
              f"flatter 20-49 minus 5-9 band gradient than both znorm and noise_corrected by more "
              f"than the {MARGIN} margin, and a smaller label-macro penalty than both by more than "
              "the margin")
    else:
        r1 = {"reading": "R1 not met: S_C is not a better-calibrated scorer than the published pair",
              "met_in": better,
              "failing": {sp: {"flatter": {o: results[sp]["R1_flatter_than"][o]["clears_margin"]
                                           for o in PUBLISHED_PAIR},
                               "penalty_shrinks": {o: results[sp]["R1_penalty_shrinks_against"][o]
                                                   ["clears_margin"] for o in PUBLISHED_PAIR}}
                          for sp in scored if sp not in better},
              "all_five_spaces_scored": bool(complete)}
    if under:
        headline = {"reading": (
            "R4 pre-empts R1, as pre-registered: the small-repertoire columns are UNDER-DETERMINED "
            f"in {under}, so there label-macro averaging is a fairness choice and not a bias fix, "
            "and THAT is this section's finding. R1's gradient is built on the same 5-9 band and "
            "R1's overall verdict is a conjunction over all five spaces, so the R1 verdict carried "
            "below is description only"),
            "R4_under_determined_spaces": under,
            "R1_verdict_as_description_only": r1,
            "all_five_spaces_scored": bool(complete)}
    else:
        headline = r1
    r4 = (f"R4: the small-repertoire columns are UNDER-DETERMINED rather than mis-scored in {under}, "
          "so for those spaces label-macro averaging is a fairness choice and not a bias fix, and "
          "that is this section's finding, in preference to R1's"
          if under else
          "R4: in every space scored, the small-repertoire columns are determined well enough to be "
          "read as mis-scored rather than under-determined, so R1's verdict stands as the headline")
    return headline, r4


# ------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--spaces", nargs="*", default=list(SPACES))
    args = parser.parse_args()
    unknown = [sp for sp in args.spaces if sp not in SPACES]
    if unknown:
        raise SystemExit(f"unknown spaces {unknown}; choose from {list(SPACES)}")
    started = time.time()
    import jieba
    jieba.setLogLevel(60)
    d = setup(args.private_root.resolve())
    li, gi, w, L, fold, dense = (d["label_index"], d["group_ids"], d["weights"], d["label_count"],
                                 d["fold"], d["dense"])
    n = len(li)
    everything = np.arange(n)

    # ---- invariants, all of them before any scoring
    _, comp_label, _ = component_ids(li, gi)
    population = {"queries": n, "labels": int(L), "groups": int(len(np.unique(gi))),
                  "components": int(len(comp_label)), "query_label_pairs": int(n * L)}
    print(f"{population['queries']:,} queries, {population['labels']} labels, "
          f"{population['groups']:,} groups, {population['components']:,} components", flush=True)
    for field, value in EXPECTED_POPULATION.items():
        if population[field] != value:
            raise SystemExit(f"the population changed: {field} is {population[field]}, not {value}")
    songs_per_label = np.bincount(li, minlength=L)
    components_per_label = np.bincount(comp_label, minlength=L)
    smallest = int(components_per_label.min())
    if smallest < 3:
        raise SystemExit(f"a label holds only {smallest} components, so removing a query's leakage "
                         "group can leave fewer than two and the C^2 - C denominator of the "
                         "component U-statistic is not defined")
    bandings: dict[str, dict] = {}
    band_masks: dict[str, np.ndarray] = {}
    for banding, bands, by_label in (("songs_per_label", SONG_BANDS, songs_per_label),
                                     ("components_per_label", COMPONENT_BANDS, components_per_label)):
        bandings[banding] = {}
        for band, low, high in bands:
            in_band = (by_label >= low) & (by_label <= high)
            mask = in_band[li]
            bandings[banding][band] = {"banding": banding, "labels": int(in_band.sum()),
                                       "queries": int(mask.sum())}
            band_masks[f"{banding}/{band}"] = mask
    for band, count in EXPECTED_SONG_BAND_LABELS.items():
        got = bandings["songs_per_label"][band]["labels"]
        if got != count:
            raise SystemExit(f"songs_per_label band {band} holds {got} labels, not {count}")
    for band, count in EXPECTED_SONG_BAND_QUERIES.items():
        got = bandings["songs_per_label"][band]["queries"]
        if got != count:
            raise SystemExit(f"songs_per_label band {band} holds {got} queries, not {count}")
    for band, count in EXPECTED_COMPONENT_BAND_QUERIES.items():
        got = bandings["components_per_label"][band]["queries"]
        if got != count:
            raise SystemExit(f"components_per_label band {band} holds {got} queries, not {count}")
    print(f"bands check out in both bandings; smallest label {smallest} components", flush=True)

    print("building the surface spaces", flush=True)
    documents = [d["documents"][s] for s in d["songs"]]
    words, _ = fit_words([" ".join(segment(doc)) for doc in documents])
    words = words.astype(np.float64).tocsr()
    chars = v1.fit_tfidf(documents).astype(np.float64).tocsr() if "chars_raw" in args.spaces else None
    del documents
    reduced: dict[int, np.ndarray] = {}
    if "words_svd_whitened" in args.spaces:
        for k in range(FOLDS):
            svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(words[fold != k])
            reduced[k] = unit_rows(svd.transform(words))
            print(f"  fold {k} word SVD fitted (label-free)", flush=True)

    # ---- the generic parts against the protocol's own dense and sparse scorers
    probe = everything[:300]
    dots, norm2, _ = lgo_parts(dense, li, gi, w, L, probe)
    gap_dense = float(np.abs(dots / np.sqrt(norm2)
                             - dense_leave_group_out(dense, li, gi, w, L, probe)).max())
    reference = v1.score_leave_group_out(dense.astype(np.float32), words.astype(np.float32), li, gi, L)
    dots, norm2, _ = lgo_parts(words, li, gi, w, L, probe)
    gap_sparse = float(np.abs(dots / np.sqrt(norm2) - reference.lexical[:300].astype(np.float64)).max())
    del reference, dots, norm2
    print(f"generic parts against the protocol: dense {gap_dense:.2e}, sparse {gap_sparse:.2e}",
          flush=True)
    if gap_dense > 1e-9 or gap_sparse > 1e-4:
        raise SystemExit("the generic leave-group-out parts differ from the protocol's scorers")

    checks = {"population": population, "smallest_label_components": smallest,
              "generic_parts_gap_dense": gap_dense, "generic_parts_gap_sparse": gap_sparse,
              "against_published": {}, "band_reproduction": {}, "by_space": {}}
    results: dict[str, dict] = {}
    all_mask = np.ones(n, dtype=bool)

    def write_payload() -> Path:
        scored = list(results)
        headline, r4 = readings(results, scored) if scored else ("no space scored yet", "not read yet")
        payload = {
            "analysis": "calibrated_profile_scorer_v3",
            "question": ("whether dividing a query's dot with the label profile by an estimate of "
                         "the label's true resultant length, instead of by the realised profile "
                         "norm, removes the protocol prototype's dependence on label size"),
            "corpus": {"content_sha256": V3_CONTENT_SHA256, **population},
            "design": {
                "spaces_requested": list(args.spaces),
                "spaces_scored": scored,
                "scorers": {
                    "exemplar_mean": "q.m_l, the protocol-weighted mean cosine",
                    "prototype": "q.m_l / ||m_l||, the protocol",
                    "znorm": "the prototype standardised per label by the impostor scores of "
                             "training-fold songs of other labels (label_size_calibration_v3)",
                    "noise_corrected": "(dots/counts) / sqrt(max(m2 - s2/counts, 0.25 m2)); its "
                                       "floor caps a small label's column at a constant ratio of "
                                       "the quantity it is correcting rather than correcting it",
                    "calibrated_S_C": "q.m_l * kappa / sqrt(r_star^2): the component U-statistic "
                                      "for r^2, empirical-Bayes shrunk toward a training-fold "
                                      "prior, passed through a soft positive part at the scale of "
                                      "its own sd, and divided with a second-order correction of "
                                      "the convexity bias of x -> x^(-1/2)",
                },
                "estimator_form": {
                    "u_statistic_components": "(||sum_c y_c||^2 - sum_c ||y_c||^2) / (C^2 - C), "
                                              "y_c the component mean; this IS the definition of "
                                              "r_hat^2, not a hypothesis about the profile norm",
                    "identity_not_a_check": "with r_hat^2 so defined, E||m_l||^2 = r_l^2 + s^2 W_l "
                                            "is an algebraic identity, so no check here validates "
                                            "r_hat^2 against the profile norm; the derivation and "
                                            "its bound on this corpus's component structure are "
                                            "in the module docstring and were computed outside "
                                            "this file, not by this run",
                    "empirical_questions": "the r_hat^2 <= 0 rate, the shrinkage weight b, the "
                                           "bias of the variance-stabilised form at these n and d "
                                           "(the synthetic test), and whether S_C flattens the "
                                           "band gradient",
                    "empirical_bayes": "b = tau^2/(tau^2+v), v = 4 zeta1 / n_eff with zeta1 from a "
                                       "delete-one-component jackknife on the training folds; "
                                       "tau^2 = max(0, Var_l(r_hat^2) - mean_l(v)); prior mean and "
                                       "tau^2 are plain (unweighted) over labels, from the "
                                       "training folds only",
                    "soft_positive_part": "r_star^2 = (r_tilde^2 + sqrt((r_tilde^2)^2 + "
                                          "4 v_tilde))/2, the positive root of r_star^2 (r_star^2 "
                                          "- r_tilde^2) = v_tilde: strictly positive for every "
                                          "input, passed through up to an offset below "
                                          "v_tilde/r_tilde^2 when the estimate stands several sd "
                                          "above zero, sqrt(v_tilde) at zero",
                    "ratio_correction": f"kappa = 1/(1 + {DELTA} v_tilde / (r_star^2)^2), the "
                                        "second-order correction of E[X^(-1/2)] ~ mu^(-1/2)"
                                        "(1 + 3 Var/(8 mu^2)); in (0, 1] for every input and in "
                                        "[8/11, 1] wherever r_tilde^2 >= 0, so no clamp and no "
                                        "tuned constant",
                },
                "estimands": {"query_weighted": "np.mean(1/ranks)",
                              "component_weighted": "each (leakage group, label) component weight "
                                                    "one; the estimand of paired_group_bootstrap "
                                                    "and of every interval here",
                              "label_macro": "component-weighted mean within a label, then the "
                                             "plain mean over labels"},
                "bandings": {"songs_per_label": [b[0] for b in SONG_BANDS],
                             "components_per_label": [b[0] for b in COMPONENT_BANDS],
                             "note": "both have a band called 20-49 and they are different bands; "
                                     "every band entry here names its banding, the reading rule is "
                                     "declared on songs_per_label, and components_per_label is its "
                                     "sensitivity (its 50-up band is empty and is a stub)"},
                "margin": MARGIN,
                "under_determined_thresholds": {"non_positive_r2_share": UNDER_DETERMINED_RATE,
                                                "mean_shrinkage_weight": COLLAPSE_WEIGHT},
                "reading_rule": READING_RULE,
                "null_reading_fixed_before_the_run": NULL_READING,
                "znorm_no_op": ZNORM_NOOP,
                "bootstrap_universe": BOOTSTRAP_UNIVERSE,
                "not_included": NOT_INCLUDED,
                "checks": ("up front: the population, at least three components per label, both "
                           "bandings' published label and query counts, and the generic parts "
                           "against the protocol's dense and sparse scorers; per space: unit rows "
                           "at the producing code's own guarantee (1e-9 fold-wise, 4e-5 squared "
                           "for the float32-derived raw spaces), the weight count equal to the "
                           "component count, q.m_l / prototype / r_hat^2 / S_C against brute-force "
                           "loops, the published MRRs and the twenty published band MRRs, and the "
                           "shared-draw bootstrap against paired_group_bootstrap on one band"),
            },
            "checks": checks,
            "spaces": results,
            "readings": {"headline": headline,
                         "R2_against_the_prototype": {sp: results[sp]["R2_reading"] for sp in scored},
                         "R4_null": r4},
            "privacy": ("aggregate only: MRRs, intervals, shares, means and counts. No song id, "
                        "label name, artist, lyric text or vector appears here, and no per-label "
                        "array is emitted. Quantiles of the estimator's per-label quantities are "
                        f"suppressed in any band holding fewer than {MIN_LABELS_FOR_QUANTILES} "
                        "labels, because over so few labels they would amount to a per-label "
                        "listing; the songs_per_label 50-up band (three labels) is suppressed by "
                        "that rule and keeps only means."),
            "minutes": round((time.time() - started) / 60.0, 1),
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / OUT_NAME
        path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                          + "\n").encode("utf-8"))
        return path

    out_path = write_payload()       # the skeleton, so no later gate can abort with nothing written
    print(f"wrote the skeleton to {out_path}", flush=True)

    for sp in args.spaces:
        t0 = time.time()
        print(f"scoring {sp}", flush=True)
        scores, diagnostics, fold_info, audit = score_space(sp, li, gi, w, L, fold, dense, words,
                                                            chars, reduced)
        checks["by_space"][sp] = audit
        rr = {key: 1.0 / ranks_of(matrix, li) for key, matrix in scores.items()}
        del scores

        # ---- the gate: the published numbers, before any new number is read
        entry = {}
        for scorer, expected in (("prototype", EXPECTED_PROTOTYPE[sp]), ("znorm", EXPECTED_ZNORM[sp]),
                                 ("noise_corrected", EXPECTED_NOISE[sp]),
                                 ("exemplar_mean", EXPECTED_EXEMPLAR_MEAN.get(sp))):
            got = float(np.mean(rr[scorer]))
            cell = {"expected": expected, "recomputed": round(got, 4), "estimand": "query_weighted"}
            if expected is None:
                cell["note"] = ("exemplar_vs_prototype.json covers four spaces and has no chars_raw "
                                "column, so this anchor has no published value and is checked only "
                                "against its brute-force definition")
            else:
                cell["tolerance"] = TOLERANCE[sp]
                cell["gap"] = round(abs(got - expected), 5)
                if abs(got - expected) > TOLERANCE[sp]:
                    raise SystemExit(f"{sp}: {scorer} gives {got:.4f}, not the published {expected}")
            entry[scorer] = cell
            print(f"  check {sp} {scorer}: {got:.4f} against the published {expected}", flush=True)
        checks["against_published"][sp] = entry
        band_check = {}
        for band, expected in EXPECTED_PROTOTYPE_BANDS[sp].items():
            got = weighted_mean(rr["prototype"], w, band_masks[f"songs_per_label/{band}"])
            band_check[band] = {"expected": expected, "recomputed": round(got, 4),
                                "estimand": "component_weighted", "banding": "songs_per_label",
                                "tolerance": BAND_GAP}
            if abs(got - expected) > BAND_GAP:
                raise SystemExit(f"{sp}: the prototype's {band} band gives {got:.4f}, not the "
                                 f"published {expected}")
        checks["band_reproduction"][sp] = band_check
        print(f"  {sp}: the published MRRs and all four band MRRs reproduce", flush=True)

        # ---- estimands, whole population and both bandings
        estimands = {scorer: three_estimands(rr[scorer], w, li) for scorer in SCORERS}
        penalty = {scorer: round(estimands[scorer]["query_weighted"]
                                 - estimands[scorer]["label_macro"], 4) for scorer in SCORERS}
        band_tables: dict[str, dict] = {}
        for banding, table in bandings.items():
            band_tables[banding] = {}
            for band, info in table.items():
                cell = dict(info)
                mask = band_masks[f"{banding}/{band}"]
                if info["queries"] > 0:
                    cell["mrr"] = {scorer: three_estimands(rr[scorer], w, li, mask)
                                   for scorer in SCORERS}
                    cell["estimator"] = estimator_block(diagnostics, mask, li, info["labels"])
                band_tables[banding][band] = cell

        # ---- paired intervals, one shared draw over every band
        boot_masks = {"all": all_mask}
        for name, mask in band_masks.items():
            if mask.any():
                boot_masks[name] = mask
        drawn, labels_lost = shared_draw_bootstrap({s: rr[s] for s in SCORERS}, w, gi, li,
                                                   all_mask, boot_masks, full_estimands=("all",))
        pairs = [("calibrated_S_C", "prototype"), ("calibrated_S_C", "exemplar_mean"),
                 ("calibrated_S_C", "znorm"), ("calibrated_S_C", "noise_corrected"),
                 ("znorm", "prototype"), ("noise_corrected", "prototype")]
        overall = [contrast(a, b, drawn["all"]["component_weighted"][a],
                            drawn["all"]["component_weighted"][b], "component_weighted")
                   for a, b in pairs]
        band_contrasts: dict[str, dict] = {}
        for banding, table in bandings.items():
            band_contrasts[banding] = {}
            for band, info in table.items():
                if info["queries"] == 0:
                    continue
                cell = drawn[f"{banding}/{band}"]["component_weighted"]
                band_contrasts[banding][band] = [
                    contrast(a, b, cell[a], cell[b], "component_weighted")
                    for a, b in (("calibrated_S_C", "prototype"), ("calibrated_S_C", "znorm"),
                                 ("calibrated_S_C", "noise_corrected"))]

        # ---- a single-band universe must reproduce paired_group_bootstrap exactly (once per run)
        if "shared_draw_against_paired_group_bootstrap" not in checks:
            solo_mask = band_masks["songs_per_label/20-49"]
            two = {"calibrated_S_C": rr["calibrated_S_C"], "prototype": rr["prototype"]}
            solo, _ = shared_draw_bootstrap(two, w, gi, li, solo_mask, {"band": solo_mask})
            mine = contrast("calibrated_S_C", "prototype",
                            solo["band"]["component_weighted"]["calibrated_S_C"],
                            solo["band"]["component_weighted"]["prototype"], "component_weighted")
            theirs = paired_group_bootstrap(two, w, gi, solo_mask,
                                            [("calibrated_S_C", "prototype")])[0]
            gap = max(abs(mine["difference"] - theirs["mrr_difference"]),
                      abs(mine["ci95"][0] - theirs["ci95"][0]),
                      abs(mine["ci95"][1] - theirs["ci95"][1]))
            if gap > BOOTSTRAP_GAP:
                raise SystemExit(f"{sp}: the shared-draw bootstrap does not reproduce "
                                 f"paired_group_bootstrap ({gap:.2e})")
            checks["shared_draw_against_paired_group_bootstrap"] = {"space": sp, "band": "20-49",
                                                                    "gap": round(gap, 6)}

        # ---- R1: the band gradient and the label-macro penalty, from the one shared draw
        high, low = GRADIENT_BANDS
        gradient, penalty_draws = {}, {}
        for scorer in SCORERS:
            top = drawn[f"songs_per_label/{high}"]["component_weighted"][scorer]
            bottom = drawn[f"songs_per_label/{low}"]["component_weighted"][scorer]
            gradient[scorer] = {"point": top["point"] - bottom["point"],
                                "replicates": top["replicates"] - bottom["replicates"]}
            q_draw = drawn["all"]["query_weighted"][scorer]
            m_draw = drawn["all"]["label_macro"][scorer]
            penalty_draws[scorer] = {"point": q_draw["point"] - m_draw["point"],
                                     "replicates": q_draw["replicates"] - m_draw["replicates"]}
        flatter, shrinks = {}, {}
        for other in PUBLISHED_PAIR:
            difference = (np.abs(gradient[other]["replicates"])
                          - np.abs(gradient["calibrated_S_C"]["replicates"]))
            bound = np.percentile(difference, [2.5, 97.5])
            flatter[other] = {
                "comparison": other, "estimand": "component_weighted",
                "banding": "songs_per_label", "bands": list(GRADIENT_BANDS),
                "gradient_other": round(float(gradient[other]["point"]), 4),
                "gradient_calibrated_S_C": round(float(gradient["calibrated_S_C"]["point"]), 4),
                "absolute_difference": round(float(abs(gradient[other]["point"])
                                                   - abs(gradient["calibrated_S_C"]["point"])), 4),
                "ci95": [round(float(bound[0]), 4), round(float(bound[1]), 4)],
                "clears_margin": bool(bound[0] > MARGIN),
                "bootstrap_universe": BOOTSTRAP_UNIVERSE}
            difference = (penalty_draws[other]["replicates"]
                          - penalty_draws["calibrated_S_C"]["replicates"])
            bound = np.percentile(difference, [2.5, 97.5])
            shrinks[other] = {
                "comparison": other, "estimand": "query_weighted minus label_macro",
                "penalty_other": round(float(penalty_draws[other]["point"]), 4),
                "penalty_calibrated_S_C": round(float(penalty_draws["calibrated_S_C"]["point"]), 4),
                "difference": round(float(penalty_draws[other]["point"]
                                          - penalty_draws["calibrated_S_C"]["point"]), 4),
                "replicate_mean_difference": round(float(np.mean(difference)), 4),
                "ci95": [round(float(bound[0]), 4), round(float(bound[1]), 4)],
                "clears_margin": bool(bound[0] > MARGIN),
                "bootstrap_universe": BOOTSTRAP_UNIVERSE,
                "note": "a replicate that misses all of a label's groups drops it, so the replicate "
                        "mean sits slightly off the point; the difference is paired, so the same "
                        "labels are alive for both scorers"}
        better_here = (all(flatter[o]["clears_margin"] for o in PUBLISHED_PAIR)
                       and all(shrinks[o]["clears_margin"] for o in PUBLISHED_PAIR))

        # ---- R4: is the small band under-determined?
        small = band_tables["songs_per_label"]["5-9"]["estimator"]
        under_determined = (small["non_positive_r2_share_own_label"] > UNDER_DETERMINED_RATE
                            or small["mean_shrinkage_weight_own_label"] < COLLAPSE_WEIGHT)
        non_inferior = next(c for c in overall
                            if (c["system"], c["minus"]) == ("calibrated_S_C", "prototype"))

        results[sp] = {
            "estimands": estimands,
            "label_macro_penalty": penalty,
            "fold_info": fold_info,
            "paired_contrasts": overall,
            "bandings": band_tables,
            "band_contrasts": band_contrasts,
            "band_gradient": {"banding": "songs_per_label", "bands": list(GRADIENT_BANDS),
                              "estimand": "component_weighted",
                              "value": {scorer: round(float(gradient[scorer]["point"]), 4)
                                        for scorer in SCORERS}},
            "R1_flatter_than": flatter,
            "R1_penalty_shrinks_against": shrinks,
            "R1_better_calibrated_here": bool(better_here),
            "R2_reading": ("S_C is superior to the prototype" if non_inferior["superior"] else
                           f"S_C is non-inferior to the prototype at the {MARGIN} margin"
                           if non_inferior["non_inferior_at_margin"] else
                           f"S_C is not non-inferior to the prototype at the {MARGIN} margin"),
            "R4_small_band_under_determined": bool(under_determined),
            "znorm_of_S_C_top1_agreement_with_znorm_prototype": round(
                float(np.mean(diagnostics["znorm_top1_agreement"])), 4),
            "label_macro_labels_lost_max_share": round(labels_lost, 4),
            "minutes": round((time.time() - t0) / 60.0, 1),
        }
        print(f"== {sp}: prototype {estimands['prototype']['component_weighted']:.4f}, "
              f"S_C {estimands['calibrated_S_C']['component_weighted']:.4f} (component-weighted); "
              f"gradient prototype {gradient['prototype']['point']:+.4f} -> "
              f"S_C {gradient['calibrated_S_C']['point']:+.4f}; 5-9 band: r_hat^2<=0 "
              f"{small['non_positive_r2_share_own_label']:.3f}, mean b "
              f"{small['mean_shrinkage_weight_own_label']:.3f}; "
              f"R1 {'met' if better_here else 'not met'}; "
              f"R4 {'under-determined' if under_determined else 'determined'}; "
              f"{results[sp]['minutes']} min", flush=True)
        for c in overall[:4]:
            print(f"    {c['system']} - {c['minus']}: {c['difference']:+.4f} "
                  f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] (component-weighted)", flush=True)
        del rr, diagnostics
        out_path = write_payload()

    headline, r4 = readings(results, list(results))
    print(f"\n{headline if isinstance(headline, str) else headline['reading']}", flush=True)
    print(r4, flush=True)
    print(f"wrote {out_path} in {round((time.time() - started) / 60.0, 1)} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
