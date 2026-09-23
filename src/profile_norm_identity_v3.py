#!/usr/bin/env python3
"""What rests on the profile-norm identity, given that the identity itself cannot fail.

QUESTION. label_size_calibration_v3's noise-corrected scorer divides a query's dot product with
a label's leave-group-out profile MEAN m_l by sqrt(max(||m_l||^2 - s^2 W_l, 0.25 ||m_l||^2))
instead of by ||m_l||, taking ONE pooled s^2 per (space, fold), fitted on the training folds, and
W_l = 1 / G_l. The published noise_corrected MRRs, the band tables under them, and the
label-size Reading paragraph of results/retrieval-v3/README.md all rest on that scorer. This
file asks what in it is actually uncertain.

FINDING 0, AND WHY THIS FILE DOES NOT TEST THE DECOMPOSITION. The obvious validation -- measure
each label's residual variance and compare it with (1 - C_bar_l)(1 + W_l) -- is an ALGEBRAIC
IDENTITY given the definitions, not a hypothesis, so it cannot fail and must not be reported as
a check that passed. Write a label's components c = 1..G (a component is a (leakage group, label)
pair; the protocol's weight w_s = 1 / k_{c(s)} gives each component total weight one, so
xbar_c is the plain mean of its k_c songs), K = sum_c 1/k_c, T = sum_c xbar_c,
S = sum_c ||xbar_c||^2, C_bar = (||T||^2 - S) / (G(G-1)), chat_c the mean cosine over distinct
song pairs inside component c, nu the component-weighted mean squared row norm, and mhat(c) the
mean over the components other than c. Then, exactly:

    s2_measured  = nu - C_bar * G/(G-1) + S/(G(G-1))            [the estimator, with nu -> 1
                                                                 wherever unit rows are assumed]
    s2_predicted = (1 - C_bar) * (1 + K/(G(G-1)))               [W_l(c) averaged over c]
    S            = nu*K + sum_c (1 - 1/k_c) chat_c
    gap          = [ K(nu - 1) + sum_c (1 - 1/k_c)(chat_c - C_bar) ] / (G(G-1))

the last line being the gap of the estimator as coded, which substitutes 1 for nu (as
label_size_calibration_v3.residual_variance does: its resid is 1 - 2d + m2). Three consequences.
(i) A singleton component contributes EXACTLY ZERO, because 1 - 1/k_c = 0; a label whose every
leakage group holds one song has gap == 0.0 identically, for any vectors, in any space, whether
or not the decomposition is true. (ii) With unit rows the gap is bounded in magnitude by
2(G - K)/(G(G-1)), and this corpus supplies almost no multi-song mass to it: 7,220 songs sit in
6,889 components, so sum over all components of (k_c - 1) is 331 and sum_c (1 - 1/k_c) over ALL
226 labels is at most 165.5, about 0.73 per label against G of 20 to 49. The bound is computed
per label below and reported. (iii) The gap therefore cannot reach the 0.02 margin anyone would
declare, so a "the identity holds" verdict is fixed before the run and protects nothing. The
validation effort belongs in the four questions below, and the bound is reported so a reader can
see why.

FINDING 0b, also algebraic and also used below. Where the floor binds, the noise-corrected score
is EXACTLY twice the prototype score: the divisor becomes sqrt(0.25 ||m||^2) = ||m||/2. Away from
the floor the scorer is the prototype times (1 - s^2 W_l / ||m||^2)^(-1/2). So the whole
correction is a per-(query, label) MULTIPLIER in [1, 2] applied to the prototype column, and
ranks move only through its variation across labels within a query's row. Every number below is
reported on that multiplier scale, which is the scale that changes ranks. The run checks the
factor-of-two identity numerically on fold 0's floored pairs.

WHAT IS MEASURED, per space (raw BGE-M3 song centroids; the same within-author whitened per fold;
raw jieba word TF-IDF; raw character 2-5-gram TF-IDF; the word SVD-1024 whitened per fold; folds,
transforms, leave-group-out and component weights exactly as in identity_probe_v2 and
label_size_calibration_v3).

 1. IS POOLING VALID? Each label's own s^2_l against the pooled s^2 the scorer subtracts:
    quantiles of the ratio, the ratio by band, the resulting change in the correction multiplier,
    and the number of pairs whose floor status flips if the label's own s^2 is used.
 2. THE WEIGHT TERM. The published W = 1/G_l against the weight-aware W = K_l/G_l^2, which by the
    closed form above differ exactly by the multi-song-component term: the ratio G_l/K_l per band
    (structural, hence identical in every space), the change in the multiplier, and the MRR of the
    same scorer with the weight-aware subtraction.
 3. THE FLOOR. The recorded floor-hit counts confirmed or corrected in all five folds; whether the
    floored pairs are whole label columns and whether those columns are the 5-9-song labels; and
    the published (noise_corrected - prototype) change decomposed by band and by whether the floor
    bound on the query's own label.
 4. THE DEGENERACY RATE of the component-level U-statistic
    r_hat^2 = (||sum_c y_c||^2 - sum_c ||y_c||^2) / (G^2 - G), recorded per band and space before
    calibrated_profile_scorer_v3 is read. No scorer is built on it here.

ESTIMANDS. query_weighted = mean of 1/rank over queries; component_weighted = sum(rr w)/sum(w),
which is what paired_group_bootstrap returns and what label_size_calibration.json's by_band.*.mrr
reports; label_macro = the component-weighted mean within a label, then the plain mean over the
labels present in the mask. Every number carries its estimand and no point estimate of one
estimand is printed beside an interval of another. Note the published mix this file avoids:
README.md's "raises the raw word space by +0.034 [+0.027, +0.041] to 0.539" puts a
component-weighted difference beside a query-weighted level (query-weighted the difference is
+0.043).

BANDING. Every band table here is by SONGS per label: 5-9 (19 labels, 149 queries), 10-19
(33, 477), 20-49 (171, 6,356), 50-up (3, 238). The components-per-label banding of
exemplar_vs_prototype_v3 (4-9: 159 queries, 10-19: 648, 20-49: 6,413, 50-up: 0) is gated so the
two can never be confused, and is not reported. The repo calls two different bands "20-49".

READING RULES, fixed before the run.

 R1 POOLING. In a space, pooling one s^2 across labels is DEFENSIBLE if both quartiles of
    s^2_l / s^2_pooled over the 226 labels lie inside [0.90, 1.10], and INDEFENSIBLE otherwise.
    The interval is declared from the producing code: label_size_calibration.json's own pooled
    s^2 varies by less than 0.3% across the five folds in every space (0.3547-0.3567, 0.9965-
    0.9969, 0.9901-0.9910, 1.0012-1.0020, 0.9830-0.9848), so a per-label interquartile spread ten
    times that is the point at which one number stops describing the middle half of the labels;
    +/-10% of the subtracted term is also at most 7.5 points of ||m||^2, inside the 25 the floor
    already concedes. The rule reads the IQR, not the range, so a handful of extreme small labels
    cannot decide it. IF INDEFENSIBLE the consequence is fixed now: the space's published
    noise_corrected column is not "the size effect removed" but one particular per-label
    reweighting, mis-specified by the factor reported here; the band whose median ratio departs
    furthest from 1 is the band whose published noise_corrected MRR is least trustworthy; and
    README.md's "s^2 is 0.36 in the raw semantic space and 1.00 in the word space" must be read as
    a corpus average, not as a per-label constant. Nothing else is void -- the prototype and znorm
    columns use no s^2, and label_specific_covariance_v3's sigma^2 is a different estimator (a
    pooled within-label variance per dimension for lambda sigma^2 I) that this file does not touch.
 R2 THE WEIGHT TERM. The published W = 1/G_l is ADEQUATE in a space if the third quartile of
    G_l/K_l over the 226 labels is below 1.01 AND the whole 95% paired group-bootstrap interval of
    (weight-aware - published) MRR lies inside +/-0.005, the repo's MRR margin; INADEQUATE
    otherwise.
 R3 THE FLOOR. The floor is the MECHANISM of the published change in a space if (a) at least 95%
    of fold 0's floored pairs lie in label columns floored for every query, and (b) the
    component-weighted (noise_corrected - prototype) interval among queries whose own label was
    floored clears +/-0.005 in the same direction as the overall change. A sentence naming a GAIN
    is written only where the overall interval lies entirely above +0.005, and one naming a COST
    only where it lies entirely below -0.005; otherwise neither word is used. The published signs
    are known and are not uniform: noise_corrected beats the prototype in words_raw (+0.034) and
    chars_raw (+0.022) and loses in semantic_raw (-0.037), semantic_whitened (-0.027) and
    words_svd_whitened (-0.017), all component-weighted.
 R4 THE DEGENERACY RATE is a number recorded for a later run, not a test.

CHECKS THAT ABORT, and where they sit. Up front, before any space is built or scored: the two
band tables reproduce the published label and query counts; n*L equals the published 1,631,720
pairs; every row of the three raw spaces is a unit vector; the generic leave-group-out parts equal
the protocol's dense and sparse scorers on 300 queries; this file's component counts equal the
protocol's; every (query, label) keeps at least two components, which the U-statistic needs. Per
space, after that space's parts exist: the per-label s^2 pools to residual_variance; the
reimplemented scorer equals scorers_from_parts bit for bit and its floor mask equals that helper's
count; floored pairs are exactly twice the prototype; the pooled s^2 of all five folds, the
prototype and noise_corrected MRRs, and the twenty published band MRRs reproduce. NOTHING new is
read from a space whose gate fails, and the payload is written before any of these raises, so no
gate can abort the run after the expensive work without leaving the run's output behind. The
floor-hit counts are REPORTED, never gated: nothing in the producing code guarantees them to any
tolerance (the two fold-wise spaces depend on a randomized TruncatedSVD basis and a randomized
whitening fit, and the published fold-to-fold spread is itself around a percentage point), and
question 3 is asked to confirm or CORRECT them, which a gate cannot do.

TOLERANCES, none tighter than the guarantee of the code that produced the number. 5e-4 for the
three raw spaces and 0.002 for the two fold-wise ones, on every reproduced MRR and pooled s^2:
four-decimal published values round at 5e-5, the raw spaces are deterministic given the corpus,
and 0.002 is the repo's own concession (label_size_calibration_v3.CHECK_GAP) for the two spaces
whose basis is randomized. 4e-5 on |row squared norm - 1| in the three raw spaces, because
v1.fit_tfidf's own contract is np.allclose(row_norms, 1.0, atol=2e-5) and v1.l2_normalize_dense
normalises in float32; 1e-9 in the two fold-wise spaces, which are identity_probe_v2.unit_rows
output in float64. 1e-9 on the dense parts and 1e-4 on the sparse ones, as
label_size_calibration_v3 itself uses, the protocol's sparse scorer running in float32. Exact
equality only where the operations are literally identical on identical inputs.

PRIVACY. Aggregate only: quantiles, band tables and counts over labels and query-label pairs. No
song id, label name, artist, lyric text or vector, and no per-label array. A band holding fewer
than 5 labels (the 50-up songs band holds 3) gets a mean and counts but no median and no
quantiles, because the median of three labels is one label's own value.

    CHINESE_RAP_CORPUS=v3 python src/profile_norm_identity_v3.py --private-root <the private root>
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
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402
from label_size_calibration_v3 import BANDS as SONG_BANDS  # noqa: E402
from label_size_calibration_v3 import FLOOR_SHARE, lgo_parts, residual_variance, scorers_from_parts  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "profile_norm_identity.json"
SPACES = ("semantic_raw", "semantic_whitened", "words_raw", "chars_raw", "words_svd_whitened")
FOLDWISE = ("semantic_whitened", "words_svd_whitened")
SCORERS = ("prototype", "noise_corrected", "noise_corrected_weight_aware")
COMPONENT_BANDS = (("4-9", 4, 9), ("10-19", 10, 19), ("20-49", 20, 49), ("50-up", 50, 10**9))
QUERY_LABEL_PAIRS = 1631720
EXPECTED_SONG_BAND_LABELS = {"5-9": 19, "10-19": 33, "20-49": 171, "50-up": 3}
EXPECTED_SONG_BAND_QUERIES = {"5-9": 149, "10-19": 477, "20-49": 6356, "50-up": 238}
EXPECTED_COMPONENT_BAND_QUERIES = {"4-9": 159, "10-19": 648, "20-49": 6413, "50-up": 0}
# the task's prototype MRRs; results/retrieval-v3/label_size_calibration.json, systems.*, which
# are QUERY-WEIGHTED
EXPECTED_PROTOTYPE = {"semantic_raw": 0.2997, "semantic_whitened": 0.4164, "words_raw": 0.4963,
                      "chars_raw": 0.4266, "words_svd_whitened": 0.5426}
EXPECTED_NOISE_CORRECTED = {"semantic_raw": 0.2635, "semantic_whitened": 0.3902, "words_raw": 0.5394,
                            "chars_raw": 0.4591, "words_svd_whitened": 0.5291}
# the same file, by_band.<band>.mrr.<space>.<scorer>, which are COMPONENT-WEIGHTED
EXPECTED_BAND_PROTOTYPE = {
    "semantic_raw": {"5-9": 0.1597, "10-19": 0.2806, "20-49": 0.3010, "50-up": 0.4016},
    "semantic_whitened": {"5-9": 0.3404, "10-19": 0.4152, "20-49": 0.4140, "50-up": 0.5068},
    "words_raw": {"5-9": 0.0958, "10-19": 0.2738, "20-49": 0.5290, "50-up": 0.4396},
    "chars_raw": {"5-9": 0.1123, "10-19": 0.2610, "20-49": 0.4517, "50-up": 0.3685},
    "words_svd_whitened": {"5-9": 0.3250, "10-19": 0.4530, "20-49": 0.5570, "50-up": 0.5561}}
EXPECTED_BAND_NOISE_CORRECTED = {
    "semantic_raw": {"5-9": 0.3987, "10-19": 0.3780, "20-49": 0.2481, "50-up": 0.3545},
    "semantic_whitened": {"5-9": 0.4051, "10-19": 0.4593, "20-49": 0.3795, "50-up": 0.4722},
    "words_raw": {"5-9": 0.4278, "10-19": 0.5806, "20-49": 0.5331, "50-up": 0.7097},
    "chars_raw": {"5-9": 0.3868, "10-19": 0.5472, "20-49": 0.4459, "50-up": 0.6145},
    "words_svd_whitened": {"5-9": 0.5520, "10-19": 0.5201, "20-49": 0.5272, "50-up": 0.6299}}
# the same file, fold_info.<space>.<fold>.s2 and .floor_hits_of_query_label_pairs, folds 0..4
EXPECTED_S2 = {
    "semantic_raw": [0.3561, 0.3555, 0.3567, 0.3547, 0.3559],
    "semantic_whitened": [1.0014, 1.0015, 1.0012, 1.0013, 1.0020],
    "words_raw": [0.9967, 0.9965, 0.9966, 0.9965, 0.9969],
    "chars_raw": [0.9904, 0.9905, 0.9907, 0.9901, 0.9910],
    "words_svd_whitened": [0.9830, 0.9834, 0.9843, 0.9842, 0.9848]}
PUBLISHED_FLOOR_HITS = {
    "semantic_raw": [0, 0, 0, 0, 0],
    "semantic_whitened": [115634, 144535, 101264, 144479, 115704],
    "words_raw": [144428, 144428, 144428, 144428, 144429],
    "chars_raw": [130045, 130051, 137262, 130043, 137264],
    "words_svd_whitened": [43379, 36175, 50592, 36173, 28937]}
TOLERANCE = {s: (0.002 if s in FOLDWISE else 5e-4) for s in SPACES}
UNIT_ROW_TOLERANCE = {s: (1e-9 if s in FOLDWISE else 4e-5) for s in SPACES}
PARTS_GAP_DENSE = 1e-9         # float64 both sides, as label_size_calibration_v3 gates it
PARTS_GAP_SPARSE = 1e-4        # the protocol's sparse scorer runs in float32
COUNT_GAP = 1e-9               # sums of 1/k in float64
POOLED_GAP = 1e-9              # the same terms, summed in a different order
TWICE_GAP = 1e-12              # a few ulps of numbers of order one
MRR_MARGIN = 0.005             # the repo's MRR margin
POOLING_INTERVAL = (0.90, 1.10)   # R1, declared above
W_ADEQUATE_Q75 = 1.01             # R2, declared above
FLOOR_COLUMN_SHARE = 0.95         # R3, declared above
MIN_BAND_LABELS = 5            # below this a band keeps a mean and counts, never a median


# ------------------------------------------------------------------ structure
def components(label_index: np.ndarray, group_ids: np.ndarray):
    """The (group, label) components: an id per song, and each component's label, group and size."""
    keys = sorted(set(zip(group_ids.tolist(), label_index.tolist())))
    position = {key: i for i, key in enumerate(keys)}
    comp_of = np.asarray([position[(int(g), int(l))] for g, l in zip(group_ids.tolist(), label_index.tolist())],
                         dtype=np.int64)
    comp_group = np.asarray([key[0] for key in keys], dtype=np.int64)
    comp_label = np.asarray([key[1] for key in keys], dtype=np.int64)
    comp_size = np.bincount(comp_of, minlength=len(keys)).astype(np.float64)
    return comp_of, comp_label, comp_group, comp_size


def band_of_label(sizes: np.ndarray, bands) -> np.ndarray:
    out = np.full(len(sizes), "", dtype=object)
    for name, low, high in bands:
        out[(sizes >= low) & (sizes <= high)] = name
    if any(b == "" for b in out.tolist()):
        raise RuntimeError("a label falls outside every band")
    return out


def lgo_column_sums(base: np.ndarray, comp_label, comp_group, group_ids, label_count, queries):
    """Per (query, label): the sum of a per-component quantity over the label's components that
    survive removing the query's whole leakage group. base = 1 gives the protocol's component
    count G(q, l); base = 1/k_c gives K(q, l); base = ||xbar_c||^2 gives the sum of squared
    component norms the U-statistic subtracts."""
    total = np.bincount(comp_label, weights=base, minlength=label_count)
    out = np.repeat(total[None, :], len(queries), axis=0)
    rows_by_group: dict[int, list[int]] = defaultdict(list)
    for row, query in enumerate(queries.tolist()):
        rows_by_group[int(group_ids[query])].append(row)
    for c in range(len(comp_label)):
        rows = rows_by_group.get(int(comp_group[c]))
        if rows:
            out[rows, int(comp_label[c])] -= float(base[c])
    return out


def row_squared_norms(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.multiply(x).sum(axis=1)).ravel()
    return np.einsum("ij,ij->i", x, x)


def component_sumsq(x, comp_of: np.ndarray, weights: np.ndarray, component_count: int) -> np.ndarray:
    """||xbar_c||^2 per component; xbar_c is the protocol-weighted, hence plain, mean of its songs."""
    picker = sparse.csr_matrix((weights, (comp_of, np.arange(len(comp_of)))),
                               shape=(component_count, len(comp_of)))
    means = picker @ x
    return row_squared_norms(means.tocsr() if sparse.issparse(means) else means)


def per_label_s2(dots, norm2, counts, label_index, weights, label_count):
    """Per label, the component-weighted mean over its OWN songs of 1 - 2d + m2, which is
    ||x_s - mhat_l(c(s))||^2 when the rows are unit vectors: exactly the per-song residual that
    label_size_calibration_v3.residual_variance pools, pooled inside a label instead."""
    rows = np.arange(len(label_index))
    own = label_index
    d = dots[rows, own] / counts[rows, own]
    m2 = norm2[rows, own] / counts[rows, own] ** 2
    resid = 1.0 - 2.0 * d + m2
    num = np.bincount(own, weights=resid * weights, minlength=label_count)
    den = np.bincount(own, weights=weights, minlength=label_count)
    if np.any(den <= 0.0):
        raise RuntimeError("a label without weight")
    return num / den


# ------------------------------------------------------------------ the scorer, as a multiplier
def corrected_scores(dots, norm2, counts, subtract):
    """The published noise-corrected scorer and its floor mask. With subtract = s2 / counts these
    are scorers_from_parts bit for bit -- the same operations in the same order on the same
    arrays -- which the run checks by exact equality."""
    m2 = norm2 / counts ** 2
    floored = (m2 - subtract) < FLOOR_SHARE * m2
    return (dots / counts) / np.sqrt(np.maximum(m2 - subtract, FLOOR_SHARE * m2)), floored


def multiplier(u: np.ndarray) -> np.ndarray:
    """The factor the correction applies to a pair's prototype score, u = subtract / ||m||^2:
    corrected = prototype / sqrt(max(1 - u, FLOOR_SHARE)). It lies in [1, 2] and is exactly 2
    where the floor binds."""
    return 1.0 / np.sqrt(np.maximum(1.0 - u, FLOOR_SHARE))


def foldwise_matrix(name, k, dense, lexical, label_index, weights, fold):
    """semantic_whitened and words_svd_whitened for one fold, exactly as label_size_calibration_v3
    builds them: transforms fitted on the training folds, rows renormalised."""
    train = fold != k
    if name == "semantic_whitened":
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train], label_index[train],
                                        weights[train], np.random.default_rng(SEED + k))
        return unit_rows((dense - mean) @ matrix.T)
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(lexical[train])
    reduced = unit_rows(svd.transform(lexical))
    mean, matrix, _ = fit_transform("within_author_whitening", reduced[train], label_index[train],
                                    weights[train], np.random.default_rng(SEED + k))
    return unit_rows((reduced - mean) @ matrix.T)


# ------------------------------------------------------------------ reporting
def quantiles(a) -> dict | None:
    if len(a) == 0:
        return None
    return {f"q{int(p * 100):02d}": round(float(np.quantile(a, p)), 4) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}


def label_band_table(values, band, songs_per_label) -> dict:
    """A per-label array by band. Below MIN_BAND_LABELS labels the median and the quantiles are
    suppressed: the median of three labels is one label's own value."""
    out = {}
    for name, _, _ in SONG_BANDS:
        sel = band == name
        labels = int(sel.sum())
        entry = {"labels": labels, "queries": int(songs_per_label[sel].sum()),
                 "mean": None, "median": None, "quantiles": None}
        if labels:
            entry["mean"] = round(float(np.mean(values[sel])), 4)
            if labels >= MIN_BAND_LABELS:
                entry["median"] = round(float(np.median(values[sel])), 4)
                entry["quantiles"] = quantiles(values[sel])
            else:
                entry["note"] = f"fewer than {MIN_BAND_LABELS} labels: median and quantiles suppressed"
        out[name] = entry
    return out


def pair_band_table(values, band) -> dict:
    """A (query, label) array by the band of the LABEL, with the same suppression rule."""
    out = {}
    for name, _, _ in SONG_BANDS:
        cols = np.flatnonzero(band == name)
        entry = {"labels": int(len(cols)), "pairs": int(values.shape[0] * len(cols)),
                 "mean": None, "median": None, "quantiles": None}
        if len(cols):
            v = values[:, cols].ravel()
            entry["mean"] = round(float(np.mean(v)), 4)
            if len(cols) >= MIN_BAND_LABELS:
                entry["median"] = round(float(np.median(v)), 4)
                entry["quantiles"] = quantiles(v)
            else:
                entry["note"] = f"fewer than {MIN_BAND_LABELS} labels: median and quantiles suppressed"
        out[name] = entry
    return out


def estimands(rr, weights, label_index, mask) -> dict | None:
    """The three estimands of one system on one mask, each labelled. label_macro is taken over the
    labels present in the mask, so a band's macro is over that band's labels."""
    if not mask.any():
        return None
    present = np.unique(label_index[mask])
    remap = np.full(int(label_index.max()) + 1, -1, dtype=np.int64)
    remap[present] = np.arange(len(present))
    macro = label_macro(rr[mask][:, None], weights[mask], remap[label_index[mask]], len(present))[0]
    return {"queries": int(mask.sum()), "labels": int(len(present)),
            "query_weighted": round(float(np.mean(rr[mask])), 4),
            "component_weighted": round(weighted_mean(rr, weights, mask), 4),
            "label_macro": round(float(macro), 4)}


def contrasts(rr, weights, group_ids, mask, pairs) -> list:
    """paired_group_bootstrap, with its estimand and this file's margin on every row."""
    if not mask.any():
        return []
    rows = paired_group_bootstrap(rr, weights, group_ids, mask, pairs)
    for row in rows:
        row["estimand"] = "component_weighted"
        row["margin"] = MRR_MARGIN
        row["above_margin"] = bool(row["ci95"][0] > MRR_MARGIN)
        row["below_negative_margin"] = bool(row["ci95"][1] < -MRR_MARGIN)
        row["inside_margin"] = bool(row["ci95"][0] > -MRR_MARGIN and row["ci95"][1] < MRR_MARGIN)
    return rows


def change_sentence(row, what) -> str:
    """A gain is named only where the whole interval clears +margin, a cost only where it clears
    -margin; otherwise neither word is used."""
    tail = f"{row['mrr_difference']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}], component-weighted"
    if row["above_margin"]:
        return f"{what} is a gain of {tail}"
    if row["below_negative_margin"]:
        return f"{what} is a cost of {tail}"
    return f"{what} is inside the {MRR_MARGIN} margin ({tail}), so neither a gain nor a cost is claimed"


# ------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--spaces", nargs="*", default=list(SPACES))
    args = parser.parse_args()
    started = time.time()
    run = [s for s in SPACES if s in args.spaces]
    if not run:
        raise SystemExit(f"--spaces names none of {SPACES}")
    import jieba
    jieba.setLogLevel(60)
    d = setup(args.private_root.resolve())
    li, gi, w, L, fold, dense = (d["label_index"], d["group_ids"], d["weights"], d["label_count"],
                                 d["fold"], d["dense"])
    n = len(li)
    everything = np.arange(n)
    comp_of, comp_label, comp_group, comp_size = components(li, gi)
    ncomp = len(comp_size)
    songs_per_label = np.bincount(li, minlength=L)
    comps_per_label = np.bincount(comp_label, minlength=L).astype(np.float64)
    band = band_of_label(songs_per_label, SONG_BANDS)
    band_by_components = band_of_label(comps_per_label.astype(np.int64), COMPONENT_BANDS)

    # up-front structural gates: nothing is built or written yet, so these may abort outright
    got = {"songs_per_label_labels": {b: int((band == b).sum()) for b, _, _ in SONG_BANDS},
           "songs_per_label_queries": {b: int(songs_per_label[band == b].sum()) for b, _, _ in SONG_BANDS},
           "components_per_label_queries": {b: int(songs_per_label[band_by_components == b].sum())
                                            for b, _, _ in COMPONENT_BANDS}}
    print(f"{n:,} queries, {L} labels, {ncomp:,} components; bands {got['songs_per_label_labels']}", flush=True)
    if (got["songs_per_label_labels"] != EXPECTED_SONG_BAND_LABELS
            or got["songs_per_label_queries"] != EXPECTED_SONG_BAND_QUERIES
            or got["components_per_label_queries"] != EXPECTED_COMPONENT_BAND_QUERIES):
        raise SystemExit(f"the band tables do not reproduce the published counts: {got}")
    if n * L != QUERY_LABEL_PAIRS:
        raise SystemExit(f"{n * L} query-label pairs, published {QUERY_LABEL_PAIRS}")

    # finding 0: the gap's own bound, from the component structure alone, in any space
    k_label = np.bincount(comp_label, weights=1.0 / comp_size, minlength=L)
    bound = 2.0 * (comps_per_label - k_label) / (comps_per_label * (comps_per_label - 1.0))
    w_ratio_label = comps_per_label / k_label
    identity = {
        "statement": "the decomposition E||m_l||^2 = r_l^2 + s^2 W_l is an algebraic identity given the "
                     "definitions, not a hypothesis, so this file does not test it and reports no verdict on it",
        "gap_closed_form": "gap_l = [K_l(nu - 1) + sum_c (1 - 1/k_c)(chat_c - C_bar_l)] / (G_l(G_l - 1)), "
                           "with nu the component-weighted mean squared row norm; a singleton component "
                           "contributes exactly zero because 1 - 1/k_c = 0",
        "upper_bound_definition": "with unit rows, |gap_l| <= 2 (G_l - K_l) / (G_l (G_l - 1)), since the "
                                  "cosines of unit vectors differ by at most 2",
        "labels_whose_gap_is_zero_by_construction": int(np.sum(comps_per_label == k_label)),
        "upper_bound_on_abs_gap": {"mean": round(float(np.mean(bound)), 5),
                                   "max": round(float(np.max(bound)), 5), "quantiles": quantiles(bound)},
        "upper_bound_by_band": label_band_table(bound, band, songs_per_label),
        "margin_a_test_would_have_declared": 0.02,
        "multi_song_components": {"components_with_several_songs": int(np.sum(comp_size >= 2)),
                                  "songs_in_them": int(np.sum(comp_size[comp_size >= 2])),
                                  "excess_songs_over_components": int(n - ncomp),
                                  "total_within_component_pair_mass_sum_c_one_minus_one_over_k":
                                      round(float(np.sum(comps_per_label - k_label)), 3)},
        "reading": "the bound is an order of magnitude inside any margin a reader would declare, and it is "
                   "identically zero for every label whose leakage groups each hold one song, so a test of "
                   "the decomposition could not have failed; the four questions below are the empirical ones"}
    print(f"gap bound over labels: max {bound.max():.5f}, median {np.median(bound):.5f}; "
          f"{int(np.sum(comps_per_label == k_label))} labels where it is zero by construction", flush=True)

    # the three raw spaces, and the unit-row invariant, up front
    print("building the three raw spaces", flush=True)
    docs = [d["documents"][s] for s in d["songs"]]
    lexical, _ = fit_words([" ".join(segment(doc)) for doc in docs])
    lexical = lexical.astype(np.float64).tocsr()
    chars = v1.fit_tfidf(docs).astype(np.float64).tocsr()
    del docs
    raw = {"semantic_raw": dense, "words_raw": lexical, "chars_raw": chars}
    unit_gap = {}
    for sp, x in raw.items():
        unit_gap[sp] = float(np.abs(row_squared_norms(x) - 1.0).max())
        print(f"  {sp}: |row squared norm - 1| at most {unit_gap[sp]:.2e} "
              f"(tolerance {UNIT_ROW_TOLERANCE[sp]})", flush=True)
        if unit_gap[sp] > UNIT_ROW_TOLERANCE[sp]:
            raise SystemExit(f"{sp}: the rows are not unit vectors ({unit_gap[sp]:.2e} > "
                             f"{UNIT_ROW_TOLERANCE[sp]}); the per-song residual 1 - 2d + m2 does not apply")

    # the parts against the protocol's two scorers, and this file's counts against the protocol's
    probe = everything[:300]
    dots, norm2, counts = lgo_parts(dense, li, gi, w, L, probe)
    gap_dense = float(np.abs(dots / np.sqrt(norm2) - dense_leave_group_out(dense, li, gi, w, L, probe)).max())
    profiles = v1.score_leave_group_out(dense.astype(np.float32), lexical.astype(np.float32), li, gi, L)
    dots_s, norm2_s, _ = lgo_parts(lexical, li, gi, w, L, probe)
    gap_sparse = float(np.abs(dots_s / np.sqrt(norm2_s) - profiles.lexical[:300].astype(np.float64)).max())
    counts_all = lgo_column_sums(np.ones(ncomp), comp_label, comp_group, gi, L, everything)
    kinv_all = lgo_column_sums(1.0 / comp_size, comp_label, comp_group, gi, L, everything)
    gap_counts = float(np.abs(counts - counts_all[probe]).max())
    del dots, norm2, counts, dots_s, norm2_s, profiles
    print(f"parts against the protocol: dense {gap_dense:.2e}, sparse {gap_sparse:.2e}, "
          f"counts {gap_counts:.2e}", flush=True)
    if gap_dense > PARTS_GAP_DENSE or gap_sparse > PARTS_GAP_SPARSE:
        raise SystemExit("the generic leave-group-out parts differ from the protocol's scorers")
    if gap_counts > COUNT_GAP:
        raise SystemExit(f"this file's component counts differ from the protocol's ({gap_counts:.2e})")
    if float(counts_all.min()) < 2.0:
        raise SystemExit("a (query, label) pair keeps fewer than two components; the U-statistic needs two")
    w_ratio_pair = counts_all / kinv_all          # published W over weight-aware W, that is G/K

    checks = {"parts_against_protocol_dense": gap_dense, "parts_against_protocol_sparse": gap_sparse,
              "component_counts_against_protocol": gap_counts, "unit_row_gap": unit_gap, "per_space": {}}
    results: dict[str, dict] = {}
    payload = {
        "analysis": "profile_norm_identity_v3",
        "question": "the profile-norm decomposition is an algebraic identity, so what is uncertain in the "
                    "published noise-corrected scorer is elsewhere: is one pooled s^2 per (space, fold) "
                    "defensible, is W = 1/G_l right, what does the floor do, and how often is the "
                    "component-level U-statistic degenerate",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L, "components": ncomp,
                   "groups": int(len(np.unique(gi))), "query_label_pairs": int(n * L),
                   "songs_per_label_bands": got["songs_per_label_labels"]},
        "design": {
            "spaces_run": run,
            "partial_run": bool(run != list(SPACES)),
            "folds": "identity_probe_v2 folds and transforms; profiles leave-group-out over all songs; the "
                     "pooled s^2 from the training folds, as published",
            "banding": "every band table in this file is by SONGS per label; the components-per-label "
                       "banding is gated and never reported, because the repo calls two different bands "
                       "'20-49'",
            "estimands": {"query_weighted": "mean of 1/rank over queries; the estimand of the published "
                                            "systems.* MRRs and of this file's gate",
                          "component_weighted": "sum(rr w) / sum(w); the estimand of every "
                                                "paired_group_bootstrap interval here and of the published "
                                                "by_band.*.mrr numbers",
                          "label_macro": "the component-weighted mean within a label, then the plain mean "
                                         "over the labels present in the mask"},
            "scorers": {"prototype": "the protocol: cosine with the normalised leave-group-out profile",
                        "noise_corrected": f"the published scorer: (dots/counts) / sqrt(max(m2 - s2/counts, "
                                           f"{FLOOR_SHARE} m2))",
                        "noise_corrected_weight_aware": "the same with s2 K(q,l)/counts^2 subtracted instead "
                                                        "of s2/counts"},
            "reading_rules": {
                "R1_pooling": f"defensible in a space if both quartiles of s2_l / s2_pooled lie inside "
                              f"{list(POOLING_INTERVAL)}; if not, the space's published noise_corrected "
                              "column is one particular per-label reweighting, not the size effect removed",
                "R2_weight_term": f"the published 1/G_l is adequate if the third quartile of G_l/K_l is "
                                  f"below {W_ADEQUATE_Q75} and the whole interval of (weight-aware - "
                                  f"published) MRR lies inside +/-{MRR_MARGIN}",
                "R3_floor": f"the floor is the mechanism of the published change if at least "
                            f"{FLOOR_COLUMN_SHARE:.0%} of floored pairs lie in columns floored for every "
                            "query and the floored stratum's interval clears the margin in the same "
                            "direction as the overall change; a gain is named only where the whole interval "
                            "lies above +0.005 and a cost only where it lies below -0.005",
                "R4_degeneracy": "a number recorded for calibrated_profile_scorer_v3, not a test"},
            "tolerances": {"mrr_and_pooled_s2": TOLERANCE, "unit_rows": UNIT_ROW_TOLERANCE,
                           "parts_dense": PARTS_GAP_DENSE, "parts_sparse": PARTS_GAP_SPARSE,
                           "note": "no tolerance is tighter than the guarantee of the code that produced "
                                   "the number; the floor-hit counts are reported, never gated, because "
                                   "nothing guarantees them and question 3 is asked to correct them"},
            "not_built": "no scorer is built on r_hat^2 here, only its degeneracy rate recorded; no znorm "
                         "arm, no floor-removed arm (the floor's effect is read from the exact "
                         "factor-of-two identity and the floored stratum instead of from a scorer whose "
                         "divisor would be clamped near zero), and no verdict on the decomposition"},
        "structure": {"components_with_several_songs": int(np.sum(comp_size >= 2)),
                      "largest_component": int(comp_size.max()),
                      "bands_songs_per_label": got["songs_per_label_labels"],
                      "queries_songs_per_label": got["songs_per_label_queries"],
                      "queries_components_per_label_gated_only": got["components_per_label_queries"]},
        "identity_is_algebraic": identity,
        "weight_term_structural": {
            "definition": "G_l / K_l per label and per (query, label): the published W = 1/G_l divided by "
                          "the weight-aware W = K_l/G_l^2; structural, so identical in every space",
            "per_label": {"quantiles": quantiles(w_ratio_label),
                          "q75": round(float(np.quantile(w_ratio_label, 0.75)), 4),
                          "max": round(float(w_ratio_label.max()), 4),
                          "by_band": label_band_table(w_ratio_label, band, songs_per_label)},
            "per_pair": {"quantiles": quantiles(w_ratio_pair.ravel()),
                         "mean": round(float(w_ratio_pair.mean()), 4),
                         "max": round(float(w_ratio_pair.max()), 4),
                         "share_above_1.001": round(float(np.mean(w_ratio_pair > 1.001)), 4),
                         "by_band": pair_band_table(w_ratio_pair, band)}},
        "checks": checks,
        "spaces": results,
        "readings": {},
        "privacy": "aggregate only: quantiles, band tables and counts over labels and query-label pairs; no "
                   "song id, label name, artist, lyric text or vector, no per-label array, and no median or "
                   f"quantiles for a band holding fewer than {MIN_BAND_LABELS} labels",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / OUT_NAME

    def flush(status: str) -> None:
        payload["status"] = status
        payload["minutes"] = round((time.time() - started) / 60.0, 1)
        path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        print(f"wrote {path}  ({payload['minutes']} min, {status})", flush=True)

    def stop(message: str) -> None:
        flush(f"stopped: {message}")
        raise SystemExit(message)

    for sp in run:
        t0 = time.time()
        shared = raw.get(sp)
        if shared is not None:
            dots, norm2, counts = lgo_parts(shared, li, gi, w, L, everything)
            sumsq = lgo_column_sums(component_sumsq(shared, comp_of, w, ncomp),
                                    comp_label, comp_group, gi, L, everything)
            shared_parts = (dots, norm2, counts, sumsq)
        else:
            shared_parts = None
        score = {name: np.zeros((n, L)) for name in SCORERS}
        own_floor = np.zeros(n, dtype=bool)
        own_degenerate = np.zeros(n, dtype=bool)
        degenerate_pairs = 0
        s2_folds, pooled_all_folds, floor_folds, ratio_folds = [], [], [], []
        first: dict = {}
        exact: dict = {}
        floor_columns = np.zeros(L)
        for k in range(FOLDS):
            if shared is not None:
                x = shared
                dots, norm2, counts, sumsq = shared_parts
            else:
                x = foldwise_matrix(sp, k, dense, lexical, li, w, fold)
                gap_unit = float(np.abs(row_squared_norms(x) - 1.0).max())
                unit_gap[f"{sp}/fold{k}"] = gap_unit
                if gap_unit > UNIT_ROW_TOLERANCE[sp]:
                    stop(f"{sp} fold {k}: the rows are not unit vectors ({gap_unit:.2e} > "
                         f"{UNIT_ROW_TOLERANCE[sp]})")
                dots, norm2, counts = lgo_parts(x, li, gi, w, L, everything)
                sumsq = lgo_column_sums(component_sumsq(x, comp_of, w, ncomp),
                                        comp_label, comp_group, gi, L, everything)
            if k == 0:
                gap_counts_space = float(np.abs(counts - counts_all).max())
                checks["per_space"].setdefault(sp, {})["counts_against_protocol"] = gap_counts_space
                if gap_counts_space > COUNT_GAP:
                    stop(f"{sp}: this file's component counts differ from the protocol's "
                         f"({gap_counts_space:.2e})")
            train, test = fold != k, np.flatnonzero(fold == k)
            s2 = residual_variance(x, dots, norm2, counts, li, w, train)
            s2_folds.append(round(s2, 4))
            if abs(s2 - EXPECTED_S2[sp][k]) > TOLERANCE[sp]:
                checks["per_space"][sp]["pooled_s2"] = {"expected": EXPECTED_S2[sp], "recomputed": s2_folds,
                                                        "tolerance": TOLERANCE[sp]}
                stop(f"{sp} fold {k}: the pooled s^2 {s2:.4f} does not reproduce the published "
                     f"{EXPECTED_S2[sp][k]}")
            pooled_all = residual_variance(x, dots, norm2, counts, li, w, np.ones(n, dtype=bool))
            pooled_all_folds.append(round(pooled_all, 4))
            s2_label = per_label_s2(dots, norm2, counts, li, w, L)
            ratio_folds.append(s2_label / pooled_all)
            proto, corrected, floor_hits = scorers_from_parts(dots, norm2, counts, s2)
            subtract_pooled = s2 / counts
            subtract_aware = s2 * kinv_all / counts ** 2
            published, floored = corrected_scores(dots, norm2, counts, subtract_pooled)
            aware, _ = corrected_scores(dots, norm2, counts, subtract_aware)
            m2 = norm2 / counts ** 2
            r2 = (norm2 - sumsq) / (counts ** 2 - counts)
            if k == 0:
                pooled_direct = float(np.sum(s2_label * np.bincount(li, weights=w, minlength=L)) / np.sum(w))
                exact = {"per_label_s2_pools_to_residual_variance": abs(pooled_direct - pooled_all),
                         "corrected_against_scorers_from_parts": float(np.abs(published - corrected).max()),
                         "floor_mask_minus_helper_count": int(floored.sum()) - int(floor_hits),
                         "floored_pairs_minus_twice_the_prototype":
                             float(np.abs(published[floored] - 2.0 * proto[floored]).max()) if floored.any() else 0.0}
                checks["per_space"][sp]["exactness"] = {key: (value if isinstance(value, int) else round(value, 15))
                                                       for key, value in exact.items()}
                if exact["per_label_s2_pools_to_residual_variance"] > POOLED_GAP:
                    stop(f"{sp}: the per-label s^2 does not pool to residual_variance "
                         f"({pooled_direct} against {pooled_all})")
                if exact["corrected_against_scorers_from_parts"] > 0.0 or exact["floor_mask_minus_helper_count"] != 0:
                    stop(f"{sp}: the reimplemented noise_corrected scorer is not the published one")
                if exact["floored_pairs_minus_twice_the_prototype"] > TWICE_GAP:
                    stop(f"{sp}: a floored pair is not exactly twice its prototype score")
                first = {"s2": s2, "u_pooled": subtract_pooled / m2, "u_aware": subtract_aware / m2,
                         "u_own": (s2_label[None, :] / counts) / m2, "floored": floored}
                floor_columns = floored.mean(axis=0)
            score["prototype"][test] = proto[test]
            score["noise_corrected"][test] = published[test]
            score["noise_corrected_weight_aware"][test] = aware[test]
            own_floor[test] = floored[test, li[test]]
            own_degenerate[test] = r2[test, li[test]] <= 0.0
            degenerate_pairs += int(np.sum(r2[test] <= 0.0))
            floor_folds.append(int(floor_hits))
            print(f"  {sp} fold {k}: s2 {s2:.4f} (published {EXPECTED_S2[sp][k]}), floor hits "
                  f"{int(floor_hits):,} of {n * L:,} (published {PUBLISHED_FLOOR_HITS[sp][k]:,})", flush=True)
            if shared is None:
                del x, dots, norm2, counts, sumsq
            del proto, corrected, published, aware, floored, m2, r2, subtract_pooled, subtract_aware
        if shared is not None:
            del shared_parts, dots, norm2, counts, sumsq

        # the gate: nothing new is read from a space that does not reproduce
        rr = {name: 1.0 / ranks_of(score[name], li) for name in SCORERS}
        del score
        band_masks = {b: band[li] == b for b, _, _ in SONG_BANDS}
        query_weighted = {name: round(float(np.mean(rr[name])), 4) for name in SCORERS}
        band_component = {name: {b: round(weighted_mean(rr[name], w, band_masks[b]), 4)
                                 for b, _, _ in SONG_BANDS} for name in SCORERS}
        worst_band = {name: round(max(abs(band_component[name][b] - expected[sp][b]) for b, _, _ in SONG_BANDS), 5)
                      for name, expected in (("prototype", EXPECTED_BAND_PROTOTYPE),
                                             ("noise_corrected", EXPECTED_BAND_NOISE_CORRECTED))}
        gate = {"prototype": {"estimand": "query_weighted", "expected": EXPECTED_PROTOTYPE[sp],
                              "recomputed": query_weighted["prototype"],
                              "gap": round(abs(query_weighted["prototype"] - EXPECTED_PROTOTYPE[sp]), 5),
                              "tolerance": TOLERANCE[sp]},
                "noise_corrected": {"estimand": "query_weighted", "expected": EXPECTED_NOISE_CORRECTED[sp],
                                    "recomputed": query_weighted["noise_corrected"],
                                    "gap": round(abs(query_weighted["noise_corrected"]
                                                     - EXPECTED_NOISE_CORRECTED[sp]), 5),
                                    "tolerance": TOLERANCE[sp]},
                "bands": {"estimand": "component_weighted", "banding": "songs_per_label",
                          "expected_prototype": EXPECTED_BAND_PROTOTYPE[sp],
                          "recomputed_prototype": band_component["prototype"],
                          "expected_noise_corrected": EXPECTED_BAND_NOISE_CORRECTED[sp],
                          "recomputed_noise_corrected": band_component["noise_corrected"],
                          "worst_gap": worst_band, "tolerance": TOLERANCE[sp]},
                "pooled_s2": {"expected": EXPECTED_S2[sp], "recomputed": s2_folds, "tolerance": TOLERANCE[sp]}}
        checks["per_space"][sp].update(gate)
        print(f"gate {sp}: prototype {query_weighted['prototype']:.4f} (published {EXPECTED_PROTOTYPE[sp]}), "
              f"noise_corrected {query_weighted['noise_corrected']:.4f} (published "
              f"{EXPECTED_NOISE_CORRECTED[sp]}), worst band gap {max(worst_band.values()):.5f} "
              f"(all query-weighted except the bands, which are component-weighted)", flush=True)
        if gate["prototype"]["gap"] > TOLERANCE[sp]:
            stop(f"{sp}: the prototype does not reproduce the published MRR")
        if gate["noise_corrected"]["gap"] > TOLERANCE[sp]:
            stop(f"{sp}: noise_corrected does not reproduce the published MRR")
        if max(worst_band.values()) > TOLERANCE[sp]:
            stop(f"{sp}: the band MRRs do not reproduce the published ones")

        # 1: is pooling valid?
        ratio = np.mean(ratio_folds, axis=0)
        s2_label_mean = np.mean([r * p for r, p in zip(ratio_folds, pooled_all_folds)], axis=0)
        q25, q75 = float(np.quantile(ratio, 0.25)), float(np.quantile(ratio, 0.75))
        defensible = bool(POOLING_INTERVAL[0] <= q25 and q75 <= POOLING_INTERVAL[1])
        f_pooled = multiplier(first["u_pooled"])
        f_own = multiplier(first["u_own"])
        f_aware = multiplier(first["u_aware"])
        own_floored = first["u_own"] > 1.0 - FLOOR_SHARE
        flips = int(np.sum(own_floored != first["floored"]))
        by_band_ratio = label_band_table(ratio, band, songs_per_label)
        worst_band_name = max((b for b, _, _ in SONG_BANDS if by_band_ratio[b]["median"] is not None),
                             key=lambda b: abs(by_band_ratio[b]["median"] - 1.0))
        pooling = {
            "question": "the published scorer subtracts ONE pooled s^2 per (space, fold); how far is each "
                        "label's own s^2 from it",
            "estimand": "per label, the component-weighted mean over the label's own songs of "
                        "||x_s - mhat_l(c(s))||^2, the same per-song residual residual_variance pools",
            "rows": "all songs, for the per-label s^2 and for the pooled value it is divided by, so the two "
                    "are the same estimator on the same rows; the published pooled s^2 is fitted on the "
                    "training folds and differs from this one by less than 0.3% in every fold, and the ratio "
                    "is scale-free in any case",
            "pooled_over_all_songs_by_fold": pooled_all_folds,
            "published_pooled_s2_by_fold": EXPECTED_S2[sp],
            "per_label_s2": {"quantiles": quantiles(s2_label_mean),
                             "mean": round(float(np.mean(s2_label_mean)), 4),
                             "sd_over_labels": round(float(np.std(s2_label_mean)), 4)},
            "ratio_to_pooled": {"quantiles": quantiles(ratio), "quartiles": [round(q25, 4), round(q75, 4)],
                                "declared_interval": list(POOLING_INTERVAL),
                                "iqr_inside_the_declared_interval": defensible,
                                "by_band": by_band_ratio,
                                "band_whose_median_departs_furthest_from_one": worst_band_name},
            "correction_multiplier": {
                "definition": "corrected = prototype / sqrt(max(1 - subtract/||m||^2, 0.25)), so the "
                              "correction is a per-pair factor in [1, 2]; fold 0's space, fold 0's pooled s^2",
                "published": {"quantiles": quantiles(f_pooled.ravel()),
                              "mean": round(float(f_pooled.mean()), 4),
                              "share_at_the_floor_factor_two": round(float(np.mean(first["floored"])), 4),
                              "by_band": pair_band_table(f_pooled, band)},
                "own_s2_over_published": {"quantiles": quantiles((f_own / f_pooled).ravel()),
                                          "mean": round(float((f_own / f_pooled).mean()), 4),
                                          "by_band": pair_band_table(f_own / f_pooled, band)}},
            "floor_status_flips_if_the_label_own_s2_is_used": {
                "pairs": flips, "share": round(flips / float(n * L), 5),
                "meaning": "the pooling error alone moves this many pairs across the 0.25 floor, where the "
                           "multiplier jumps to exactly 2"},
            "reading": ""}
        pooling["reading"] = (
            f"pooling one s^2 across labels is DEFENSIBLE in {sp}: the interquartile ratio of a label's own "
            f"s^2 to the pooled one is [{q25:.3f}, {q75:.3f}], inside the declared "
            f"[{POOLING_INTERVAL[0]}, {POOLING_INTERVAL[1]}]"
            if defensible else
            f"pooling one s^2 across labels is INDEFENSIBLE in {sp}: the interquartile ratio is "
            f"[{q25:.3f}, {q75:.3f}], outside the declared [{POOLING_INTERVAL[0]}, {POOLING_INTERVAL[1]}]. "
            f"This space's published noise_corrected column is then not the size effect removed but one "
            f"particular per-label reweighting, mis-specified by the factor reported here; the {worst_band_name} "
            f"band's published noise_corrected MRR is the least trustworthy of the four; and README.md's "
            f"pooled s^2 sentence must be read as a corpus average, not as a per-label constant. The "
            f"prototype and znorm columns and label_specific_covariance_v3's sigma^2 are untouched.")

        # 2: the weight-aware W against the published 1/G
        aware_ratio = f_aware / f_pooled
        weight_term = {
            "question": "the published correction subtracts s^2 (1/G_l); the weight-aware term is "
                        "s^2 K_l/G_l^2, smaller exactly by the multi-song-component mass",
            "ratio_published_over_weight_aware": {"note": "structural; reported once in "
                                                          "weight_term_structural above, identical in every space",
                                                  "per_label_q75": round(float(np.quantile(w_ratio_label, 0.75)), 4)},
            "multiplier_weight_aware_over_published": {"quantiles": quantiles(aware_ratio.ravel()),
                                                       "mean": round(float(aware_ratio.mean()), 4),
                                                       "min": round(float(aware_ratio.min()), 4),
                                                       "by_band": pair_band_table(aware_ratio, band)},
            "change_in_the_correction_subtracted": {
                "definition": "1 - K_l(q)/G_l(q), the share of the subtracted term the published W adds in "
                              "excess of the weight-aware one",
                "quantiles": quantiles((1.0 - kinv_all / counts_all).ravel()),
                "mean": round(float(np.mean(1.0 - kinv_all / counts_all)), 5),
                "by_band": pair_band_table(1.0 - kinv_all / counts_all, band)},
            "reading": ""}

        # 3: the floor
        deviation = max(abs(100.0 * (a - b) / float(n * L))
                        for a, b in zip(floor_folds, PUBLISHED_FLOOR_HITS[sp]))
        fully = floor_columns >= 1.0
        small = SONG_BANDS[0][0]
        floored_total = float(first["floored"].sum())
        floor_block = {
            "published_floor_hits_by_fold": PUBLISHED_FLOOR_HITS[sp],
            "recomputed_floor_hits_by_fold": floor_folds,
            "query_label_pairs": int(n * L),
            "published_share_percent": [round(100.0 * v / QUERY_LABEL_PAIRS, 3)
                                        for v in PUBLISHED_FLOOR_HITS[sp]],
            "recomputed_share_percent": [round(100.0 * v / float(n * L), 3) for v in floor_folds],
            "largest_deviation_points": round(deviation, 3),
            "reported_not_gated": "nothing in the producing code guarantees this count to a tolerance, and "
                                  "question 3 is asked to confirm or CORRECT it; the MRR gate above is what "
                                  "protects the reading",
            "floored_pairs_are_exactly_twice_the_prototype": exact["floored_pairs_minus_twice_the_prototype"],
            "columns": {"basis": "fold 0's floor mask, by label column",
                        "columns_floored_for_every_query": int(fully.sum()),
                        "columns_floored_for_at_least_99_percent": int(np.sum(floor_columns >= 0.99)),
                        "columns_never_floored": int(np.sum(floor_columns <= 0.0)),
                        "share_of_floored_pairs_inside_fully_floored_columns":
                            round(float(n * fully.sum()) / floored_total, 4) if floored_total > 0 else None,
                        "smallest_band": small,
                        "labels_in_the_smallest_band": int((band == small).sum()),
                        "of_them_floored_for_every_query": int(np.sum(fully & (band == small))),
                        "fully_floored_columns_outside_the_smallest_band": int(np.sum(fully & (band != small)))},
            "share_of_pairs_floored_by_band": pair_band_table(first["floored"].astype(np.float64), band),
            "queries_floored_on_their_own_label": {
                "queries": int(own_floor.sum()), "share": round(float(own_floor.mean()), 4),
                "by_band": {b: {"queries": int(band_masks[b].sum()),
                                "share": round(float(own_floor[band_masks[b]].mean()), 4)
                                if band_masks[b].any() else None} for b, _, _ in SONG_BANDS}},
            "reading": ""}

        # the decomposition of the published change, by band and by floor stratum
        pairs = [("noise_corrected", "prototype"), ("noise_corrected_weight_aware", "noise_corrected")]
        all_mask = np.ones(n, dtype=bool)
        strata = {"floor_bound_on_the_query_own_label": own_floor,
                  "floor_did_not_bind_on_the_query_own_label": ~own_floor}
        decomposition = {
            "note": "every level is given under all three estimands and every interval is "
                    "component-weighted; no point estimate of one estimand stands beside an interval of "
                    "another. The published README sentence mixes them: +0.034 is component-weighted while "
                    "0.539 is query-weighted",
            "overall": {"mrr": {name: estimands(rr[name], w, li, all_mask) for name in SCORERS},
                        "contrasts": contrasts(rr, w, gi, all_mask, pairs)},
            "by_band": {b: {"banding": "songs_per_label", "labels": int((band == b).sum()),
                            "mrr": {name: estimands(rr[name], w, li, band_masks[b]) for name in SCORERS},
                            "contrasts": contrasts(rr, w, gi, band_masks[b], pairs)}
                        for b, _, _ in SONG_BANDS},
            "by_floor_stratum": {key: {"mrr": {name: estimands(rr[name], w, li, mask) for name in SCORERS},
                                       "contrasts": contrasts(rr, w, gi, mask, pairs)}
                                 for key, mask in strata.items()}}
        overall_rows = decomposition["overall"]["contrasts"]
        published_row = next(r for r in overall_rows if (r["system"], r["minus"]) == ("noise_corrected", "prototype"))
        aware_row = next(r for r in overall_rows
                         if (r["system"], r["minus"]) == ("noise_corrected_weight_aware", "noise_corrected"))
        floored_rows = decomposition["by_floor_stratum"]["floor_bound_on_the_query_own_label"]["contrasts"]
        floored_row = next((r for r in floored_rows
                            if (r["system"], r["minus"]) == ("noise_corrected", "prototype")), None)
        adequate = bool(float(np.quantile(w_ratio_label, 0.75)) < W_ADEQUATE_Q75 and aware_row["inside_margin"])
        weight_term["reading"] = (
            f"the published W = 1/G_l is {'ADEQUATE' if adequate else 'INADEQUATE'} in {sp}: the third "
            f"quartile of G_l/K_l is {float(np.quantile(w_ratio_label, 0.75)):.4f} against the declared "
            f"{W_ADEQUATE_Q75}, and " + change_sentence(aware_row, "replacing it with the weight-aware term"))
        column_share = floor_block["columns"]["share_of_floored_pairs_inside_fully_floored_columns"]
        same_direction = bool(floored_row is not None and (
            (published_row["above_margin"] and floored_row["above_margin"])
            or (published_row["below_negative_margin"] and floored_row["below_negative_margin"])))
        mechanism = bool(column_share is not None and column_share >= FLOOR_COLUMN_SHARE and same_direction)
        floor_block["reading"] = (
            (f"the floor {'IS' if mechanism else 'is NOT'} established as the mechanism of the published "
             f"change in {sp}: ")
            + (f"{column_share:.4f} of fold 0's floored pairs lie in the {int(fully.sum())} columns floored "
               f"for every query, of which {int(np.sum(fully & (band == small)))} of the "
               f"{int((band == small).sum())} labels in the {small} band"
               if column_share is not None else "the floor binds on no pair in this space")
            + "; " + change_sentence(published_row, "noise_corrected minus prototype overall")
            + ("; " + change_sentence(floored_row, "and among queries whose own label was floored")
               if floored_row is not None else "; no query was floored on its own label"))

        # 4: the degeneracy rate, recorded
        degeneracy = {
            "statement": "the rate at which the component-level U-statistic r_hat^2 = (||sum_c y_c||^2 - "
                         "sum_c ||y_c||^2) / (G^2 - G) is not positive, recorded before "
                         "calibrated_profile_scorer_v3 is read; no scorer here is built on it",
            "rows": "each query in the space and fold in which it is scored, leave-group-out",
            "all_pairs": {"pairs": int(n * L), "not_positive": degenerate_pairs,
                          "rate": round(degenerate_pairs / float(n * L), 5)},
            "own_label_pairs": {"queries": n, "not_positive": int(own_degenerate.sum()),
                                "rate": round(float(own_degenerate.mean()), 5),
                                "by_band": {b: {"queries": int(band_masks[b].sum()),
                                                "rate": round(float(own_degenerate[band_masks[b]].mean()), 5)
                                                if band_masks[b].any() else None} for b, _, _ in SONG_BANDS}}}

        results[sp] = {"gate": gate, "pooling": pooling, "weight_term": weight_term, "floor": floor_block,
                       "published_change_decomposed": decomposition, "degeneracy_rate": degeneracy,
                       "minutes": round((time.time() - t0) / 60.0, 1)}
        payload["readings"][sp] = {"pooling": pooling["reading"], "weight_term": weight_term["reading"],
                                   "floor": floor_block["reading"]}
        for line in (pooling["reading"], weight_term["reading"], floor_block["reading"]):
            print(f"   {line}", flush=True)
        print(f"   {sp} done in {results[sp]['minutes']} min", flush=True)
        del rr, f_pooled, f_own, f_aware, first
        flush("complete" if sp == run[-1] else f"partial: through {sp}")

    payload["readings"]["identity_is_algebraic"] = identity["reading"]
    flush("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
