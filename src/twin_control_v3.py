#!/usr/bin/env python3
"""Is the repeated-passage penalty a duplicate effect or a repertoire-size effect, and does the
attention gain survive on queries that have no twin at all?

QUESTION. Leave-group-out removes the query's own leakage group from every label profile, not the
label's OTHER songs that share a hook with it. within_label_repeats_v3 built the arm that fixes
that: its detector joins two same-label songs whose normalised texts share a 30-character run (601
pairs, 351 already inside one leakage group, 250 merging distinct groups, 696 songs, 116 labels) and
its arm (a) merges the groups of every cross-group pair (5,875 -> 5,691 groups, 184 merged away, 482
songs in a merged group, 91 labels losing groups, largest group 27 -> 42). Merging costs the
prototype -0.0183 (words), -0.0200 (characters), -0.0079 (raw semantic) and -0.0134 (whitened
semantic), and the arm was read as "not driven by repeated passages".

WHAT THIS ARM IS FOR, AND WHAT IT IS NOT FOR. The duplicate account is already weakened before this
file runs. temperature_scale_v3 (2026-09-22) measured the attention weights at the selected
temperature: their Shannon effective sample size is 22.7 to 39.4 songs at the median against a
median profile of 40 songs, and the largest single weight is 0.033 to 0.141 at the median, 0.447 at
the 95th percentile in raw words. At the selected temperature attention is therefore NOT collapsing
onto one near-duplicate song for the typical query -- the concentration lives in the upper tail. So
this arm's job is to BOUND the residual duplicate contribution, not to decide whether the attention
effect exists. A null here is a bound, and the existence of the effect is not on the table.

THE CONFOUND ARM (a) DOES NOT CONTROL. Merging does two things at once: it removes twins, and it
shrinks every affected label's repertoire by one component per merge. This repo's own size theory
says shrinking a repertoire lowers MRR by itself -- label_size_calibration_v3 traced the prototype's
dependence on label size to the sampling inflation of a profile norm built from few songs, s^2/n,
worst in the sparse spaces. So the recorded -0.018 to -0.020 is a compound of "twins removed" and
"repertoire shrunk". This file separates them.

THE CONTROL. A size-matched random merge. Arm (a)'s merge is described per label as a set of
clusters: two or more of that label's published groups that end up as one group, so the label loses
one component per group beyond the first in each cluster. That description is exact -- a label's
realised component loss equals the cluster arithmetic, collateral loss included, because a label
whose groups are merged only as a side effect of another label's twin still shows up here as a
cluster of its own. In the SAME labels each cluster is replicated with groups drawn at random,
greedily matched on component size, from the groups arm (a) did NOT merge; two groups that share any
OTHER label are never drawn into one replica, because merging them would cost that other label a
component the control was not asked to spend. Every draw's realised component loss is then compared
per label against arm (a)'s and the run stops if any label differs, so "matched exactly" is checked
and not asserted. DRAWS = 20, seeded SEED + 1000 + d (the offset keeps every draw clear of the
streams SEED + 0 .. SEED + 4 the per-fold transforms use), because one draw is not a control. The
control replicates each label's loss independently, so it does not reproduce the few arm (a) clusters
that merge groups of more than one label at once; that count is reported.

THE COMPLEMENT-SET ESTIMAND, PRE-REGISTERED AS PRIMARY. The primary estimand is the attention gain
on the COMPLEMENT set: the queries whose own label contains no passage-sharing partner of theirs at
all (7,220 - 696 expected). No merge is involved, so nothing about the population moves and the set
is defined on the PUBLISHED population -- never intersected with the arms' common set, which would
make the primary estimand a function of the random draws.

WHERE THE GAIN SITS IN THE ATTENTION WEIGHT. Because the recognition window is now measured, the
theory makes a sharper prediction than the twin/no-twin split: a residual duplicate contribution has
to be concentrated where one song holds an unusual share of the attention mass. So the complement-set
gain and the twin-set gain are each reported by quintile of the query's own TOP ATTENTION WEIGHT at
its pre-specified beta, the same quantity temperature_scale_v3 published (its median and 95th
percentile are reproduced here as a check). The stratifier is a function of the query's cosine
profile in a fixed space at a fixed beta -- a property of the data, not of either system's ranks --
so the stratification is legitimate; it is not randomised, so a difference between strata is a
falsifier of the duplicate account and not an estimate of a duplicate contribution.

SCOPE, and what it costs. The prototype is scored under every arm in the four spaces
within_label_repeats_v3 scored (semantic_raw, semantic_whitened, chars_raw, words_raw); the full beta
grid (0, 1, 2, 5, 10, 20, 50, 100) runs on the PUBLISHED arm only, in words_raw, chars_raw and
semantic_whitened. Every other arm is scored at beta 0 and at its space's selected betas alone,
which is all clause 2 reads and which cuts the grid cost by about three quarters. The word SVD-1024
space is left out on purpose: within_label_repeats_v3 never scored it, its published attention gain
is +0.0022, and refitting a 1024-component SVD for every arm would cost more than everything else
here together -- so this file does not reproduce its 0.5426. Every (arm, space) cell is written to
--cache-dir as one .npz keyed by a fingerprint of the corpus digest, the arm's population and the
betas, so a suspend loses at most one cell. Those files hold per-query reciprocal ranks, which is why
they are not in results/ (and *.npz is already git-ignored); nothing per query reaches the JSON.

ORDER OF WORK, so that no gate can abort the run after the expensive part. Every invariant and every
comparison against a published number is checked before the 20 control arms are scored: the
population, the bands, the unit rows, beta = 0 against the protocol's own scorer, the attention
scorer against its definition, the detector's counts, arm (a)'s eighteen merge counts, the four
published prototype MRRs, the whole published beta curve and its selected-beta MRRs, the published
COMPONENT-WEIGHTED attention gain, the published top-weight quantiles, arm (a)'s four recorded MRRs,
and every draw's realised per-label component loss. After that point nothing raises.

ESTIMANDS, labelled everywhere because this repo mixes them. Query-weighted MRR is np.mean(1/rank).
Component-weighted is sum(rr*w)/sum(w) over the (group, label) components, which is what
paired_group_bootstrap returns. Label-macro is the component-weighted mean within a label, then the
plain mean over labels. Here:
  * the complement-set gain is COMPONENT-WEIGHTED, point and interval alike, which is the estimand
    attention_exemplar_v3's published +0.0170 / +0.0167 / +0.0338 already uses, and those three
    numbers are gated on the all-queries cell of the same computation;
  * every cross-arm difference is QUERY-WEIGHTED, point and interval alike, obtained by calling
    paired_group_bootstrap with unit weights, because the recorded drops -0.0183 / -0.0200 / -0.0079
    / -0.0134 are query-weighted `mrr` levels and a reading rule that quotes them must be on their
    estimand. The arm's own weights are NOT used for a cross-arm difference, since the merge changes
    them and the difference would then mix a reweighting with a rank change;
  * label-macro is reported for the headline complement-set levels, over the published numbering.
Band claims use the BANDS of label_size_calibration_v3, which band by SONGS per label (5-9:
19 labels / 149 queries, 10-19: 33 / 477, 20-49: 171 / 6,356, 50-up: 3 / 238) and are checked against
those counts. That is NOT the components-per-label banding of exemplar_vs_prototype_v3, whose "20-49"
is a different set of queries; every band field here says which banding it uses. A band or quintile
with fewer than 10 labels or fewer than 100 queries is reported as counts only, with no interval:
that is label_size_calibration_v3's own counts_for_the_rule threshold, and it is what keeps the
three-label 50-up band from becoming a per-label listing in disguise.

READING RULE, fixed before the run: clause 1 (primary), clause 1b (secondary, the falsifier) and
clause 2 (equivalence), with the repo's non-inferiority margin of 0.005 MRR wherever the claim is
that something is small.

 1. THE GAIN ON TWIN-FREE QUERIES. In each lexical space, on the complement set, the gain of
    attention at the beta attention_exemplar_v3 selected for that space (words 20, characters 10,
    whitened semantic 5/5/5/5/10 by fold -- fixed constants imported from temperature_scale_v3, not
    re-selected here, so this arm carries no selection) minus the prototype, component-weighted with
    a paired group-bootstrap interval over the published leakage groups:
      * "survives" if the point estimate is positive and the WHOLE interval is above zero;
      * "positive but below the margin" if it is also entirely below +0.005;
      * "bounded below the margin" if the interval includes zero and lies entirely below +0.005 -- a
        real negative result, not an absence of evidence;
      * "undecided" if the interval includes zero and reaches above +0.005; then this run cannot
        separate the two and the paper must say so instead of claiming either.
    Both lexical spaces must survive for the clause to pass, and UNDECIDED IS TESTED FIRST so that
    one space's bound can never be published while another space is undecided. Whitened semantic is
    read by the same rule as a secondary space and does not decide the clause.
 1b. WHERE THE GAIN SITS. Quintiles of the query's own top attention weight, over the complement set
    and over the twin set. The residual-duplicate account is CONTRADICTED if the complement-set gain
    survives in the bottom quintile, SUPPORTED if it survives in the top quintile and is bounded
    below the margin in the bottom one, and undecided otherwise.
 2. THE DROP DECOMPOSITION (equivalence). residual = MRR(arm a) - mean over draws MRR(control),
    query-weighted on the published query set, with a paired group-bootstrap interval per draw over
    the union of arm (a)'s and that draw's groupings (a resampling unit must be a unit of both arms).
    Per space:
      * "mostly repertoire shrinkage" if the point lies inside +/-0.005 AND the WHOLE interval lies
        inside +/-0.005 in at least QUORUM = 18 of the 20 draws. This is an equivalence test at the
        repo's margin precisely because a run with no power could otherwise "confirm" a null;
      * "twin-specific beyond the size control" if the point is at most -0.005 with the whole
        interval below zero in at least 18 draws;
      * "the control loses more than arm (a)" in the mirror case;
      * "undecided" otherwise.
    The same decomposition is applied to the attention GAIN as a difference of differences, with its
    own verdict vocabulary.

WHAT EACH OUTCOME MEANS FOR THE PAPER.
  * Clause 1 survives and clause 2 reads "mostly repertoire shrinkage": the attention result stands
    as published, and the repeated-passage arm's -0.018 to -0.020 must be reported as a
    repertoire-size effect with the size control named, not as the price of duplicate text. The
    sentence that repeats are "worth 0.02 to 0.04 of MRR in the surface spaces" becomes an upper
    bound that the size control mostly explains.
  * Clause 1 gives "bounded below the margin" or "positive but below the margin" in the lexical
    spaces: the published +0.017 is then partly carried by within-label repeated text, and the
    attention section must say that on queries with no repeated text in their own label the mixture
    buys at most 0.005 MRR. If the beta curve nonetheless keeps an interior maximum on the complement
    set, what the interior optimum finds is NOT recurring text and the claim survives in weakened
    form; if the curve flattens or falls monotonically, the interior optimum WAS recurring text, and
    "a credited label is identified best as a mixture weighted toward the songs that resemble the
    query" must be narrowed to "toward the songs that repeat text with the query". Clause 1b says
    which of those is available: a bound that holds only in the top-weight quintile is a duplicate
    bound; a bound that holds in the bottom quintile too cannot be, since there the attention mass is
    spread over tens of songs, and that is the reading which would have to be squared with the
    measured recognition window rather than reported beside it.
  * Clause 2 reads "twin-specific": the repeated-passage arm measured what it meant to measure, the
    size control is a null control, and nothing in that section changes except that it gains a
    control.
  * Any clause "undecided": nothing is claimed. The counts, curves and achieved match are still
    reported, and the paper says the separation needs more queries or a bigger merge.

    set CHINESE_RAP_CORPUS=v3
    python src/twin_control_v3.py --private-root <private-root>
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
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import within_label_repeats_v3 as wlr  # noqa: E402
import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from attention_exemplar_v3 import BETAS, LabelSets, attention_scores, brute_attention, gram_matrices  # noqa: E402
from build_downstream_retrieval_v2 import corpus_version  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402
from label_size_calibration_v3 import BANDS  # noqa: E402
from leakage_groups_v2 import UnionFind  # noqa: E402
from temperature_scale_v3 import SELECTED_BETA  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "twin_control.json"
CACHE_DIR = ROOT / "cache" / "twin-control"

PROTO_SPACES = ("semantic_raw", "semantic_whitened", "chars_raw", "words_raw")
GRID_SPACES = ("words_raw", "chars_raw", "semantic_whitened")
LEXICAL = ("words_raw", "chars_raw")
FOLDWISE = ("semantic_whitened",)

DRAWS = 20
DRAW_SEED_OFFSET = 1000     # SEED + 0 .. SEED + 4 belong to the per-fold transforms
MARGIN = 0.005              # the non-inferiority / equivalence margin used elsewhere in this repo
QUORUM = 18                 # draws out of DRAWS that must agree for clause 2 to read
BLOCK = 400
PROBE = 300                 # queries in the protocol-equality check
BRUTE_QUERIES = 3
QUINTILES = 5
# label_size_calibration_v3's own counts_for_the_rule threshold. Below it a stratum gets counts and
# no interval: the 50-up band holds three labels and the artist list is public as names, so an
# interval over so few labels is a per-label listing in disguise.
MIN_CELL_LABELS, MIN_CELL_QUERIES = 10, 100
# l2_normalize_dense normalises in float32, fit_tfidf's own contract is
# np.allclose(row_norms, 1.0, atol=2e-5) and fit_words builds float32, so the squared row norm of the
# three float32-provenance spaces is only guaranteed to about |n-1|*|n+1| = 4e-5. unit_rows works in
# float64, so the whitened spaces are guaranteed far tighter.
UNIT_ROW_ATOL_FLOAT32 = 4e-5
UNIT_ROW_ATOL_FLOAT64 = 1e-9

EXPECTED = dict(wlr.EXPECTED)                      # 0.2997 / 0.4266 / 0.4963 / 0.4164
TOLERANCE = dict(wlr.TOLERANCE)                    # 5e-4 raw, 0.002 fold-wise
# A published CONTRAST is a difference of two levels, each guaranteed only to TOLERANCE, so its gate
# cannot be tighter than twice TOLERANCE. (The lexical prototype here is the float64 attention scorer
# at beta 0, equal to the protocol's float32 scorer to 1e-4 in the scores, not bit for bit.)
GAIN_TOLERANCE = {s: 2 * TOLERANCE[s] for s in TOLERANCE}
# attention_exemplar.json, mrr_by_beta (query-weighted plain means)
PUBLISHED_CURVE = {
    "words_raw": {0.0: 0.4963, 1.0: 0.4992, 2.0: 0.5011, 5.0: 0.5047, 10.0: 0.5093, 20.0: 0.5139,
                  50.0: 0.4952, 100.0: 0.3611},
    "chars_raw": {0.0: 0.4266, 1.0: 0.4302, 2.0: 0.4333, 5.0: 0.4404, 10.0: 0.4436, 20.0: 0.4391,
                  50.0: 0.3997, 100.0: 0.3061},
    "semantic_whitened": {0.0: 0.4164, 1.0: 0.4289, 2.0: 0.4355, 5.0: 0.4496, 10.0: 0.4484,
                          20.0: 0.4357, 50.0: 0.3940, 100.0: 0.3709},
}
PUBLISHED_SELECTED = {"words_raw": 0.5139, "chars_raw": 0.4436, "semantic_whitened": 0.4481}
# attention_exemplar.json, paired_contrasts: attention_selected minus prototype, COMPONENT-weighted
PUBLISHED_GAIN = {"words_raw": 0.0170, "chars_raw": 0.0167, "semantic_whitened": 0.0338}
# temperature_scale.json, scale[space].window_at_selected_beta.top_weight_share
PUBLISHED_TOP_WEIGHT = {"words_raw": {"q50": 0.0521, "q95": 0.4471},
                        "chars_raw": {"q50": 0.0438, "q95": 0.2319},
                        "semantic_whitened": {"q50": 0.0498, "q95": 0.2075}}
# the same formula on the same matrices, so only 4-dp rounding and float reassociation separate them
TOP_WEIGHT_TOLERANCE = 1e-3
# within_label_repeats.json, detector and arms.merged_groups.merge
PUBLISHED_DETECTOR = {"same_label_song_pairs_sharing_passage": 601, "pairs_already_in_one_published_group": 351,
                      "pairs_merging_distinct_groups": 250, "songs_in_at_least_one_pair": 696,
                      "songs_in_a_pair_across_published_groups": 340, "labels_with_at_least_one_pair": 116,
                      "audit_rule_pairs_the_detector_must_find": 319, "audit_rule_pairs_missed": 0}
PUBLISHED_MERGE = {"groups_before": 5875, "groups_after": 5691, "groups_merged_away": 184,
                   "largest_group_before": 27, "largest_group_after": 42, "multi_song_groups_before": 796,
                   "multi_song_groups_after": 852, "songs_in_a_merged_group": 482, "labels_losing_groups": 91,
                   "minimum_groups_in_a_label_after": 5, "median_groups_in_a_label_before": 33.0,
                   "median_groups_in_a_label_after": 32.0, "labels_dropped_single_group": 0,
                   "songs_dropped_with_their_label": 0, "queries": 7220, "labels": 226, "groups": 5691}
PUBLISHED_ARM_A = {"words_raw": 0.4780, "chars_raw": 0.4066, "semantic_raw": 0.2918, "semantic_whitened": 0.4030}
PUBLISHED_RAW_DROP = {"words_raw": -0.0183, "chars_raw": -0.0200, "semantic_raw": -0.0079,
                      "semantic_whitened": -0.0134}
PUBLISHED_POPULATION = {"queries": 7220, "labels": 226, "groups": 5875, "components": 6889}
PUBLISHED_BANDS = {"5-9": (19, 149), "10-19": (33, 477), "20-49": (171, 6356), "50-up": (3, 238)}
CONTROL_MERGE_KEYS = ("groups_after", "groups_merged_away", "largest_group_after", "multi_song_groups_after",
                      "labels_losing_groups", "components_after")


# ------------------------------------------------------------------ the spaces of one arm
def lexical_float64(documents: list[str]):
    """within_label_repeats_v3's own two surface spaces, in float64 for the attention scorer."""
    chars, words = wlr.lexical_matrices(documents)
    return chars.astype(np.float64).tocsr(), words.astype(np.float64).tocsr()


class ArmSpaces:
    """An arm's matrices per space, keyed by fold, with the whitening built on first use.

    Lazy because a fully cached arm must cost nothing: the whitening is what a resumed run would
    otherwise refit for no reason. Every arm shares the published TF-IDF matrices, which is sound
    because every arm is gated to keep the published population.
    """

    def __init__(self, base: dict, pop: dict, chars, words) -> None:
        self.base, self.pop, self.chars, self.words = base, pop, chars, words
        self.dense = base["dense"]
        self._whitened: dict[int, np.ndarray] | None = None

    def whitened(self) -> dict[int, np.ndarray]:
        if self._whitened is None:
            li, w, fold = self.pop["label_index"], self.pop["weights"], self.pop["fold"]
            out = {}
            for k in range(FOLDS):
                train = fold != k
                mean, matrix, _ = fit_transform("within_author_whitening", self.dense[train], li[train],
                                                w[train], np.random.default_rng(SEED + k))
                rows = unit_rows((self.dense - mean) @ matrix.T)
                worst = float(np.abs(np.einsum("ij,ij->i", rows, rows) - 1.0).max())
                if worst > UNIT_ROW_ATOL_FLOAT64:
                    raise SystemExit(f"the whitened space of fold {k} has non-unit rows ({worst:.2e}); "
                                     f"unit_rows works in float64 and cannot be this far off")
                out[k] = rows
            self._whitened = out
        return self._whitened

    def __getitem__(self, space: str) -> dict[int, object]:
        if space == "semantic_raw":
            return {k: self.dense for k in range(FOLDS)}
        if space == "chars_raw":
            return {k: self.chars for k in range(FOLDS)}
        if space == "words_raw":
            return {k: self.words for k in range(FOLDS)}
        if space == "semantic_whitened":
            return self.whitened()
        raise KeyError(space)


# ------------------------------------------------------------------ arm (a) described per label
def component_sizes(base: dict) -> Counter:
    """(group, label) -> songs of that label in that group. Its reciprocal is the protocol's weight."""
    return Counter(zip(base["group_ids"].tolist(), base["label_index"].tolist()))


def groups_by_label(base: dict) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for group, label in component_sizes(base):
        out[int(label)].append(int(group))
    return {label: sorted(groups) for label, groups in out.items()}


def labels_by_group(base: dict) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for group, label in zip(base["group_ids"].tolist(), base["label_index"].tolist()):
        out[int(group)].add(int(label))
    return out


def label_clusters(base: dict, group_name: np.ndarray) -> dict[int, list[list[int]]]:
    """Per label, the published groups of that label a naming merged together.

    A cluster is two or more published group ids whose songs of that label end up in one group under
    `group_name`; the label loses one component for every group beyond the first in each cluster, and
    that arithmetic is the label's whole realised loss (collateral merges show up here too). This is
    the object the size-matched control replicates.
    """
    li, gi = base["label_index"], base["group_ids"]
    out: dict[int, list[list[int]]] = {}
    for label in range(base["label_count"]):
        by_name: dict[int, set[int]] = defaultdict(set)
        for i in np.flatnonzero(li == label).tolist():
            by_name[int(group_name[i])].add(int(gi[i]))
        clusters = sorted(sorted(groups) for groups in by_name.values() if len(groups) > 1)
        if clusters:
            out[int(label)] = clusters
    return out


def merged_groups_of(base: dict, group_name: np.ndarray) -> set[int]:
    """Every published group a naming put together with another one."""
    holds: dict[int, set[int]] = defaultdict(set)
    for name, group in zip(group_name.tolist(), base["group_ids"].tolist()):
        holds[int(name)].add(int(group))
    return {g for groups in holds.values() if len(groups) > 1 for g in groups}


def components_lost_per_label(base: dict, group_name: np.ndarray) -> dict[int, int]:
    """Per label, the realised number of (group, label) components a naming cost it."""
    li, gi = base["label_index"], base["group_ids"]
    return {label: len(set(gi[li == label].tolist())) - len(set(group_name[li == label].tolist()))
            for label in range(base["label_count"])}


def merge_report(base: dict, group_name: np.ndarray, pop: dict) -> dict:
    """The counts within_label_repeats_v3 reports for its merge, for any naming, so arm (a)'s can be
    compared key by key with the recorded ones and a control's with arm (a)'s."""
    li, gi = base["label_index"], base["group_ids"]
    sizes_before = np.bincount(gi)
    sizes_after = np.asarray(sorted(Counter(group_name.tolist()).values()))
    before = {l: len(set(gi[li == l].tolist())) for l in range(base["label_count"])}
    after = {l: len(set(group_name[li == l].tolist())) for l in range(base["label_count"])}
    holds: dict[int, set[int]] = defaultdict(set)
    for name, group in zip(group_name.tolist(), gi.tolist()):
        holds[int(name)].add(int(group))
    return {
        "groups_before": int(len(sizes_before)), "groups_after": int(len(sizes_after)),
        "groups_merged_away": int(len(sizes_before) - len(sizes_after)),
        "largest_group_before": int(sizes_before.max()), "largest_group_after": int(sizes_after.max()),
        "multi_song_groups_before": int((sizes_before > 1).sum()),
        "multi_song_groups_after": int((sizes_after > 1).sum()),
        "songs_in_a_merged_group": int(sum(1 for name in group_name.tolist() if len(holds[int(name)]) > 1)),
        "labels_losing_groups": int(sum(1 for l in before if after[l] < before[l])),
        "components_before": int(len(set(zip(gi.tolist(), li.tolist())))),
        "components_after": int(len(set(zip(group_name.tolist(), li.tolist())))),
        "minimum_groups_in_a_label_after": int(min(after.values())),
        "median_groups_in_a_label_before": float(np.median(list(before.values()))),
        "median_groups_in_a_label_after": float(np.median(list(after.values()))),
        "labels_dropped_single_group": int(pop["labels_dropped_single_group"]),
        "songs_dropped_with_their_label": int(pop["songs_dropped_with_their_label"]),
        "queries": int(pop["queries"]), "labels": int(pop["label_count"]), "groups": int(pop["groups"]),
    }


# ------------------------------------------------------------------ the size-matched random merge
def draw_size_matched(base: dict, clusters: dict[int, list[list[int]]], forbidden: set[int],
                      sizes: Counter, pool_by_label: dict[int, list[int]],
                      rng: np.random.Generator) -> tuple[list[tuple[int, int]], dict]:
    """One size-matched random merge, as song-position pairs for within_label_repeats_v3.merge_groups.

    For every label and every cluster arm (a) made in it, draw as many of that label's own groups,
    greedily matched on component size, from the groups arm (a) did not merge and that no earlier
    cluster of this draw has taken. Two groups that share any OTHER label are never drawn into one
    replica: merging them would cost that other label a component too, and the control has to shrink
    the repertoires arm (a) shrank and no others.
    """
    li, gi = base["label_index"], base["group_ids"]
    others = labels_by_group(base)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    target: list[int] = []
    achieved: list[int] = []
    lost = 0
    short = 0
    for label in sorted(clusters):
        pool = [g for g in pool_by_label[label] if g not in forbidden]
        for cluster in clusters[label]:
            want = sorted((sizes[(g, label)] for g in cluster), reverse=True)
            chosen: list[int] = []
            collateral: set[int] = set()
            for size in want:
                free = [g for g in pool if g not in used and not (others[g] - {label}) & collateral]
                if not free:
                    break
                gaps = np.abs(np.asarray([sizes[(g, label)] for g in free], dtype=np.int64) - size)
                closest = np.flatnonzero(gaps == gaps.min())
                pick = int(free[int(closest[int(rng.integers(len(closest)))])])
                used.add(pick)
                chosen.append(pick)
                collateral |= others[pick] - {label}
            if len(chosen) < len(want):
                short += 1
            if len(chosen) < 2:
                used.difference_update(chosen)
                continue
            target.extend(want[:len(chosen)])
            achieved.extend(sorted((sizes[(g, label)] for g in chosen), reverse=True))
            lost += len(chosen) - 1
            first = int(np.flatnonzero((gi == chosen[0]) & (li == label))[0])
            for group in chosen[1:]:
                pairs.append((first, int(np.flatnonzero((gi == group) & (li == label))[0])))
    target, achieved = sorted(target, reverse=True), sorted(achieved, reverse=True)
    mass = int(sum(target))
    distance = int(sum(abs(a - b) for a, b in zip(achieved, target)))
    return pairs, {"components_merged_away": int(lost), "clusters_short_of_available_groups": int(short),
                   "groups_drawn": int(len(used)), "component_size_target_mass": mass,
                   "component_size_l1_distance": distance,
                   "component_size_match": round(1.0 - distance / max(mass, 1), 4),
                   "largest_component_drawn": int(achieved[0]) if achieved else 0}


def union_grouping(base: dict, namings) -> np.ndarray:
    """Per song, the coarsest grouping every naming refines: published groups are unioned whenever any
    naming puts them in one group, and the union is named by its smallest published group id. A
    resampling unit for a contrast between two arms has to be a unit of both, or the bootstrap treats
    a merged pair as two independent draws."""
    gi = base["group_ids"]
    n_groups = int(gi.max()) + 1
    union_find = UnionFind(n_groups)
    for naming in namings:
        for group, name in zip(gi.tolist(), naming.tolist()):
            union_find.union(int(name), int(group))
    smallest: dict[int, int] = {}
    for g in range(n_groups):
        root = union_find.find(g)
        smallest[root] = min(smallest.get(root, g), g)
    return np.asarray([smallest[union_find.find(int(g))] for g in gi.tolist()], dtype=np.int64)


# ------------------------------------------------------------------ scoring one (arm, space) cell
def score_space(space: str, matrices: dict[int, object], pop: dict,
                betas: tuple[float, ...]) -> dict[float, np.ndarray]:
    """Reciprocal ranks per beta for one cell.

    A dense space at beta 0 alone goes through the protocol's own scorer; everything else goes through
    the attention scorer, which preflight proves equal to the protocol at beta 0 and equal to its own
    definition at every beta. A sparse space must take that route: dense_leave_group_out multiplies by
    a weight column, which is matrix multiplication for a csr matrix.
    """
    li, gi, w, L, fold = pop["label_index"], pop["group_ids"], pop["weights"], pop["label_count"], pop["fold"]
    n = len(li)
    scores = {b: np.zeros((n, L)) for b in betas}
    sets = LabelSets(li, gi, w, L)
    for k in (range(FOLDS) if space in FOLDWISE else (0,)):
        x = matrices[int(k)]
        queries = np.flatnonzero(fold == k) if space in FOLDWISE else np.arange(n)
        if not len(queries):
            continue
        if tuple(betas) == (0.0,) and not sparse.issparse(x):
            scores[0.0][queries] = dense_leave_group_out(x, li, gi, w, L, queries)
            continue
        part = attention_scores(x, queries, gi, sets, gram_matrices(x, sets), betas=tuple(betas), block=BLOCK)
        for b in betas:
            scores[b][queries] = part[b]
    return {b: 1.0 / ranks_of(scores[b], li) for b in betas}


def cell_fingerprint(arm: str, space: str, betas: tuple[float, ...], pop: dict) -> str:
    digest = hashlib.sha256()
    for part in (arm.encode("utf-8"), space.encode("utf-8"), repr(tuple(betas)).encode("utf-8"),
                 V3_CONTENT_SHA256.encode("utf-8"), pop["index"].tobytes(), pop["label_index"].tobytes(),
                 pop["group_ids"].tobytes(), np.round(pop["weights"], 12).tobytes(), pop["fold"].tobytes()):
        digest.update(part)
    return digest.hexdigest()


def load_cell(path: Path, fingerprint: str, betas: tuple[float, ...]):
    if not path.exists():
        return None
    try:
        blob = np.load(path, allow_pickle=False)
        if str(blob["fingerprint"].item()) != fingerprint or blob["rr"].shape[0] != len(betas):
            return None
        return {b: np.asarray(blob["rr"][i], dtype=np.float64) for i, b in enumerate(betas)}
    except Exception:                                   # a half-written cell is simply recomputed
        return None


def save_cell(path: Path, fingerprint: str, betas: tuple[float, ...], rr: dict[float, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, fingerprint=np.asarray(fingerprint), rr=np.stack([rr[b] for b in betas]))
    temporary.replace(path)


def betas_for(space: str, grid_spaces: tuple[str, ...], published: bool) -> tuple[float, ...]:
    """The published arm carries the whole curve, because clause 1 reads its shape. Every other arm
    needs beta 0 and its space's pre-specified betas and nothing else."""
    if space not in grid_spaces:
        return (0.0,)
    return tuple(BETAS) if published else tuple(sorted({0.0, *(float(b) for b in SELECTED_BETA[space])}))


def score_arm(arm: str, pop: dict, spaces: ArmSpaces, grid_spaces: tuple[str, ...], published: bool,
              cache_dir: Path) -> tuple[dict, dict]:
    """Every space of one arm, each cell cached on its own so a suspend loses at most one."""
    out, minutes = {}, {}
    for space in PROTO_SPACES:
        betas = betas_for(space, grid_spaces, published)
        path = cache_dir / f"{arm}__{space}.npz"
        fingerprint = cell_fingerprint(arm, space, betas, pop)
        cached = load_cell(path, fingerprint, betas)
        if cached is not None:
            out[space] = cached
            print(f"  {arm} / {space}: from the cache", flush=True)
            continue
        started = time.time()
        rr = score_space(space, spaces[space], pop, betas)
        save_cell(path, fingerprint, betas, rr)
        out[space] = rr
        minutes[space] = round((time.time() - started) / 60.0, 2)
        print(f"  {arm} / {space}: {len(betas)} beta(s) in {minutes[space]} min", flush=True)
    return out, minutes


# ------------------------------------------------------------------ checks that run before any result
def preflight_checks(base: dict, pop: dict, chars, words) -> dict:
    """beta = 0 is the protocol's own scorer and the vectorised attention scorer is its definition,
    both in the raw spaces, which need no fitting, so this is seconds and always runs first."""
    li, gi, w, L = pop["label_index"], pop["group_ids"], pop["weights"], pop["label_count"]
    probe = np.arange(min(PROBE, len(li)))
    sets = LabelSets(li, gi, w, L)
    dense = base["dense"]
    unit = {}
    for name, matrix in (("semantic_raw", dense), ("chars_raw", chars), ("words_raw", words)):
        if sparse.issparse(matrix):
            norm2 = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
        else:
            norm2 = np.einsum("ij,ij->i", matrix, matrix)
        unit[name] = float(np.abs(norm2 - 1.0).max())
        if unit[name] > UNIT_ROW_ATOL_FLOAT32:
            raise SystemExit(f"{name}: squared row norms are {unit[name]:.2e} from one, beyond the "
                             f"{UNIT_ROW_ATOL_FLOAT32} the float32 producers guarantee")
    grams_dense = gram_matrices(dense, sets)
    zero_dense = attention_scores(dense, probe, gi, sets, grams_dense, betas=(0.0,))[0.0]
    gap_dense = float(np.abs(zero_dense - dense_leave_group_out(dense, li, gi, w, L, probe)).max())
    grams_words = gram_matrices(words, sets)
    zero_words = attention_scores(words, probe, gi, sets, grams_words, betas=(0.0,))[0.0]
    reference = v1.score_leave_group_out(dense.astype(np.float32), words.astype(np.float32),
                                         li, gi, L).lexical[probe].astype(np.float64)
    gap_sparse = float(np.abs(zero_words - reference).max())
    print(f"check: beta 0 against the protocol's scorer: dense {gap_dense:.2e}, sparse {gap_sparse:.2e}", flush=True)
    if gap_dense > 1e-9 or gap_sparse > 1e-4:
        raise SystemExit("beta = 0 is not the protocol's prototype")
    brute = {}
    for name, x, grams in (("semantic_raw", dense, grams_dense), ("words_raw", words, grams_words),
                           ("chars_raw", chars, gram_matrices(chars, sets))):
        part = attention_scores(x, probe[:BRUTE_QUERIES], gi, sets, grams, betas=BETAS, block=BLOCK)
        worst = 0.0
        for b in BETAS:
            for row in range(min(BRUTE_QUERIES, len(probe))):
                for label in range(0, L, 25):
                    worst = max(worst, abs(part[b][row, label]
                                           - brute_attention(x, int(probe[row]), gi, li, w, label, b)))
        brute[name] = worst
        print(f"check: {name} attention against its definition: {worst:.1e}", flush=True)
        if worst > 1e-9:
            raise SystemExit(f"{name}: the vectorised attention scorer differs from its definition")
    return {"beta_zero_against_protocol_scorer": {"dense": gap_dense, "sparse": gap_sparse},
            "brute_force_attention_gap": brute, "worst_squared_row_norm_error": unit,
            "probe_queries": int(len(probe))}


def require_published_population(pop: dict, n: int, label_count: int, who: str) -> None:
    """Every arm has to keep the published queries in the published order, or the arms are not
    comparable row by row and the cross-arm contrasts are not paired. Checked before any scoring."""
    if (pop["queries"] != n or pop["label_count"] != label_count
            or not np.array_equal(pop["index"], np.arange(n))):
        raise SystemExit(f"{who}: the merge changed the population ({pop['queries']} queries, "
                         f"{pop['label_count']} labels against {n} and {label_count}); every arm here "
                         f"must keep the published population")


# ------------------------------------------------------------------ estimands, each named where used
def query_weighted(rr: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(rr[mask]))


def component_weighted(rr: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    return weighted_mean(rr, weights, mask)


def label_macro(rr: np.ndarray, weights: np.ndarray, label_index: np.ndarray, mask: np.ndarray) -> float:
    """Component-weighted mean within a label, then the plain mean over the labels present."""
    values = []
    for label in np.unique(label_index[mask]).tolist():
        here = mask & (label_index == label)
        values.append(np.sum(rr[here] * weights[here]) / np.sum(weights[here]))
    return float(np.mean(values))


def selected_rr(by_beta: dict[float, np.ndarray], fold: np.ndarray, selected) -> np.ndarray:
    """The fold-selected system at fixed, pre-specified betas (no selection happens here)."""
    out = np.zeros_like(by_beta[0.0])
    for k in range(FOLDS):
        here = fold == k
        out[here] = by_beta[float(selected[k])][here]
    return out


def top_attention_weight(matrices: dict[int, object], pop: dict, beta_of_query: np.ndarray,
                         space: str, block: int = BLOCK) -> np.ndarray:
    """Per query, the largest share of the attention mass its own label's profile puts on one song,
    leave-group-out, at that query's pre-specified beta. This is temperature_scale_v3's
    top_weight_share, recomputed here as the stratifier of clause 1b.

    Rows are grouped by fold BEFORE blocking, so a block never straddles two transforms.
    """
    li, gi, w, L, fold = pop["label_index"], pop["group_ids"], pop["weights"], pop["label_count"], pop["fold"]
    n = len(li)
    sets = LabelSets(li, gi, w, L)
    out = np.full(n, np.nan)
    for k in (range(FOLDS) if space in FOLDWISE else (0,)):
        rows_in_fold = np.flatnonzero(fold == k) if space in FOLDWISE else np.arange(n)
        x = matrices[int(k)]
        for start in range(0, len(rows_in_fold), block):
            rows = rows_in_fold[start:start + block]
            cosine = x[rows] @ x.T
            cosine = np.asarray(cosine.todense() if sparse.issparse(cosine) else cosine, dtype=np.float64)
            for label in np.unique(li[rows]).tolist():
                inside = np.flatnonzero(li[rows] == label)
                members, weights, groups = sets.members[label], sets.w[label], sets.g[label]
                c = cosine[np.ix_(inside, members)]
                keep = gi[rows[inside]][:, None] != groups[None, :]
                if not np.all(keep.any(axis=1)):
                    raise RuntimeError("a held-out group empties a label")
                top = np.where(keep, c, -np.inf).max(axis=1, keepdims=True)
                beta = beta_of_query[rows[inside]][:, None]
                e = np.where(keep, np.exp(beta * np.minimum(c - top, 0.0)), 0.0) * weights[None, :]
                out[rows[inside]] = e.max(axis=1) / e.sum(axis=1)
    if np.isnan(out).any():
        raise RuntimeError("a query was left without an own-label attention weight")
    return out


def quintile_of(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Quintile index 0..4 of every value, with the four edges, taken over all the values given."""
    edges = np.quantile(values, [k / QUINTILES for k in range(1, QUINTILES)])
    return np.searchsorted(edges, values, side="right"), [round(float(e), 5) for e in edges]


# ------------------------------------------------------------------ the reading rule, as functions
def gain_reading(contrast: dict, margin: float = MARGIN) -> dict:
    """Clause 1 on one stratum. A positive claim needs the whole interval above zero; the negative
    outcome is stated as a bound at the margin, so a run without power cannot read as either."""
    low, high = float(contrast["ci95"][0]), float(contrast["ci95"][1])
    point = float(contrast["mrr_difference"])
    clear = point > 0 and low > 0
    below = high < margin
    if clear and not below:
        verdict = "survives: positive with the interval clear of zero"
    elif clear and below:
        verdict = f"positive but the whole interval is below the margin of {margin}"
    elif below:
        verdict = f"bounded below the margin of {margin}: no gain worth reading here"
    else:
        verdict = "undecided: the interval includes zero and reaches above the margin"
    return {"point": point, "ci95": [low, high], "positive_clear_of_zero": bool(clear),
            "whole_interval_below_margin": bool(below), "verdict": verdict}


def clause_one_verdict(readings: dict[str, dict], lexical) -> dict:
    """Clause 1 over the lexical spaces. UNDECIDED IS TESTED FIRST: a space whose interval includes
    zero and reaches above the margin means the run cannot separate the two, and the declared rule
    says nothing is claimed then. It must not be masked by another space's bound."""
    undecided = [s for s in lexical if readings[s]["verdict"].startswith("undecided")]
    survives = [s for s in lexical if readings[s]["positive_clear_of_zero"]]
    bounded = [s for s in lexical if readings[s]["whole_interval_below_margin"]]
    if undecided:
        verdict = ("undecided on the complement set in " + ", ".join(undecided) + ": the interval includes "
                   "zero and reaches above the margin, so nothing is claimed either way; the counts and "
                   "curves are reported as they stand")
    elif len(survives) == len(lexical) and not bounded:
        verdict = "the attention gain survives the duplicate confound in both lexical spaces"
    elif bounded:
        verdict = ("the gain on twin-free queries is bounded by the margin in " + ", ".join(bounded)
                   + ": the published gain is partly carried by within-label repeated text")
    else:
        verdict = "undecided on the complement set in at least one lexical space"
    return {"undecided_lexical_spaces": undecided, "surviving_lexical_spaces": survives,
            "bounded_lexical_spaces": bounded, "verdict": verdict}


def concentration_reading(bottom: dict | None, top: dict | None) -> dict:
    """Clause 1b. The residual-duplicate account needs the gain to sit in the top-weight tail."""
    if bottom is None or top is None:
        return {"verdict": "not read: a quintile was too small to report an interval"}
    # A quintile holds about a fifth of the complement set, so its interval is roughly twice as wide
    # as the whole set's -- wider than the margin. "Bounded below the margin" would then be
    # unreachable for the bottom quintile, and the SUPPORT branch with it, leaving only the branch
    # that contradicts this arm's own thesis. Precision is therefore reported and a quintile too
    # imprecise to decide is said to be, instead of defaulting to "undecided" and reading as a
    # contradiction.
    widths = {"bottom_quintile_ci95_half_width": bottom.get("ci95_half_width"),
              "top_quintile_ci95_half_width": top.get("ci95_half_width")}
    if bottom.get("ci95_half_width") is not None and bottom["ci95_half_width"] > MARGIN:
        return {"bottom_quintile": bottom["verdict"], "top_quintile": top["verdict"], **widths,
                "verdict": (f"not read: the bottom quintile's interval half-width "
                            f"{bottom['ci95_half_width']:.4f} exceeds the {MARGIN} margin, so this "
                            "split cannot bound a residual duplicate contribution either way")}
    if bottom["positive_clear_of_zero"] and not bottom["whole_interval_below_margin"]:
        verdict = ("contradicts the residual-duplicate account: the gain survives in the quintile whose "
                   "attention is most spread, where no single song can carry it")
    elif (top["positive_clear_of_zero"] and not top["whole_interval_below_margin"]
          and bottom["whole_interval_below_margin"]):
        verdict = ("consistent with a residual duplicate contribution: the gain survives only in the "
                   "top-weight quintile and is bounded below the margin where attention is spread")
    else:
        verdict = "undecided: the quintiles do not separate"
    return {"bottom_quintile": bottom["verdict"], "top_quintile": top["verdict"], **widths,
            "verdict": verdict}


def residual_reading(per_draw: list[dict], inside_text: str, below_text: str, above_text: str,
                     margin: float = MARGIN, quorum: int = QUORUM) -> dict:
    """Clause 2, an equivalence test: a claim that something is small needs the WHOLE interval inside
    the margin, in a quorum of draws. The verdict wording is passed in, so a sentence about the
    prototype drop can never be printed about the attention gain."""
    inside = sum(1 for c in per_draw
                 if abs(c["mrr_difference"]) < margin and c["ci95"][0] > -margin and c["ci95"][1] < margin)
    below = sum(1 for c in per_draw if c["mrr_difference"] <= -margin and c["ci95"][1] < 0.0)
    above = sum(1 for c in per_draw if c["mrr_difference"] >= margin and c["ci95"][0] > 0.0)
    if inside >= quorum:
        verdict = inside_text
    elif below >= quorum:
        verdict = below_text
    elif above >= quorum:
        verdict = above_text
    else:
        verdict = "undecided"
    points = [float(c["mrr_difference"]) for c in per_draw]
    return {"draws": len(per_draw), "mean_point": round(float(np.mean(points)), 5) if points else None,
            "min_point": round(float(np.min(points)), 5) if points else None,
            "max_point": round(float(np.max(points)), 5) if points else None,
            "draws_whole_interval_inside_margin": int(inside), "draws_whole_interval_below_zero": int(below),
            "draws_whole_interval_above_zero": int(above),
            "draws_excluding_zero": int(sum(1 for c in per_draw if c["excludes_zero"])),
            "quorum": quorum, "margin": margin, "verdict": verdict}


def curve_shape(curve: dict[str, float]) -> str:
    """Whether a beta curve still has an interior maximum, which is what the attention claim rests on."""
    betas = sorted(float(b) for b in curve)
    values = [curve[f"{b:g}"] for b in betas]
    best = int(np.argmax(values))
    if best == 0:
        return "maximum at beta 0: the prototype"
    if best == len(betas) - 1:
        return "maximum at the largest beta: the nearest-song end"
    return f"interior maximum at beta {betas[best]:g}"


def stratum(systems: dict, weights: np.ndarray, groups: np.ndarray, label_index: np.ndarray,
            mask: np.ndarray) -> dict:
    """One query stratum's component-weighted gain, or its counts alone when it is too small.

    Below MIN_CELL_LABELS labels or MIN_CELL_QUERIES queries -- label_size_calibration_v3's own
    counts_for_the_rule threshold -- no interval is emitted, because an interval over three labels
    beside a public artist list is a per-label listing in disguise.
    """
    queries = int(mask.sum())
    labels = int(len(np.unique(label_index[mask]))) if queries else 0
    entry = {"queries": queries, "labels": labels,
             "counts_for_the_rule": bool(labels >= MIN_CELL_LABELS and queries >= MIN_CELL_QUERIES)}
    if not entry["counts_for_the_rule"]:
        entry["withheld"] = (f"fewer than {MIN_CELL_LABELS} labels or {MIN_CELL_QUERIES} queries: counts "
                             f"only, because an interval over so few labels would identify them")
        return entry
    contrast = paired_group_bootstrap(systems, weights, groups, mask,
                                      [("attention_selected", "prototype")])[0]
    entry["gain_component_weighted"] = contrast
    # the cell's own precision, so a reading rule can refuse a cell too imprecise to decide rather
    # than defaulting to "undecided" and being read as evidence
    entry["ci95_half_width"] = round((contrast["ci95"][1] - contrast["ci95"][0]) / 2.0, 5)
    entry["reading"] = gain_reading(contrast)
    return entry


def lookup(contrasts: list[dict], system: str, minus: str) -> dict:
    return next(c for c in contrasts if (c["system"], c["minus"]) == (system, minus))


# ------------------------------------------------------------------ the analysis
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="per-cell reciprocal ranks (.npz, git-ignored); delete to force a rerun")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--grid-spaces", nargs="*", default=list(GRID_SPACES))
    args = parser.parse_args()
    if corpus_version() != "v3":
        raise SystemExit("set CHINESE_RAP_CORPUS=v3; this analysis is defined on corpus v3")
    if args.draws < 2:
        raise SystemExit("one draw is not a control")
    grid_spaces = tuple(s for s in PROTO_SPACES if s in set(args.grid_spaces))
    if not set(LEXICAL) <= set(grid_spaces):
        raise SystemExit("clause 1 is declared on both lexical spaces; keep words_raw and chars_raw "
                         "in --grid-spaces")
    unknown = [s for s in grid_spaces if s not in PUBLISHED_CURVE or s not in PUBLISHED_TOP_WEIGHT]
    if unknown:
        raise SystemExit(f"no published beta curve or top-weight quantiles are recorded for {unknown}; "
                         f"--grid-spaces is limited to {list(GRID_SPACES)}")
    started = time.time()
    import jieba
    jieba.setLogLevel(60)

    # ---- the corpus, the population and every invariant, before anything is scored
    print("loading corpus v3 and building the protocol's population", flush=True)
    base = wlr.setup_full(args.private_root.resolve())
    li, gi, w = base["label_index"], base["group_ids"], base["weights"]
    n = len(base["songs"])
    components = int(len(set(zip(gi.tolist(), li.tolist()))))
    population = {"queries": n, "labels": int(base["label_count"]), "groups": int(gi.max()) + 1,
                  "components": components}
    print(f"  {n:,} queries, {population['labels']} labels, {population['groups']:,} groups, "
          f"{components:,} components", flush=True)
    if population != PUBLISHED_POPULATION:
        raise SystemExit(f"the population is not the published one: {population} against {PUBLISHED_POPULATION}")
    pop_pub = wlr.make_population(base["songs"], base["label_of"], np.ones(n, dtype=bool), gi,
                                  base["fold_of_group"])
    require_published_population(pop_pub, n, base["label_count"], "the rebuilt published population")
    if (not np.array_equal(pop_pub["label_index"], li) or not np.array_equal(pop_pub["group_ids"], gi)
            or not np.allclose(pop_pub["weights"], w) or not np.array_equal(pop_pub["fold"], base["fold"])):
        raise SystemExit("the population rebuilt here differs from the protocol's")

    songs_per_label = np.bincount(li, minlength=base["label_count"])
    band_of_song = np.full(n, "", dtype=object)
    band_counts = {}
    for name, low, high in BANDS:
        labels_in = np.flatnonzero((songs_per_label >= low) & (songs_per_label <= high))
        band_of_song[np.isin(li, labels_in)] = name
        band_counts[name] = (int(len(labels_in)), int(np.isin(li, labels_in).sum()))
    if band_counts != PUBLISHED_BANDS:
        raise SystemExit(f"the songs-per-label bands are not the published ones: {band_counts}")

    print("fitting the surface spaces on the published documents", flush=True)
    chars_pub, words_pub = lexical_float64([base["documents"][s] for s in base["songs"]])
    for name, matrix in (("chars_raw", chars_pub), ("words_raw", words_pub)):
        empty = int(np.sum(np.asarray(matrix.getnnz(axis=1) == 0)))
        if empty:
            raise SystemExit(f"{name}: {empty} empty rows; this file's scorer has no worst-rank rule")
    checks = {"songs_per_label_bands": {k: list(v) for k, v in band_counts.items()}}
    checks.update(preflight_checks(base, pop_pub, chars_pub, words_pub))

    # ---- the detector, imported from within_label_repeats_v3 and gated against its published counts
    print("detecting shared passages within labels", flush=True)
    position = {s: i for i, s in enumerate(base["songs"])}
    pairs = wlr.shared_passage_pairs(base["normalised"], base["label_of"])
    pair_index = [(position[a], position[b]) for a, b in pairs]
    already = sum(1 for a, b in pair_index if gi[a] == gi[b])
    _, must_find = wlr.whole_chunk_pairs(base["rows"], base["chunk_rows"], base["label_of"])
    missed = sum(1 for p in must_find if p not in set(pairs))
    involved = np.zeros(n, dtype=bool)
    across = np.zeros(n, dtype=bool)
    for a, b in pair_index:
        involved[a] = involved[b] = True
        if gi[a] != gi[b]:
            across[a] = across[b] = True
    detector = {"same_label_song_pairs_sharing_passage": len(pairs),
                "pairs_already_in_one_published_group": int(already),
                "pairs_merging_distinct_groups": int(len(pairs) - already),
                "songs_in_at_least_one_pair": int(involved.sum()),
                "songs_in_a_pair_across_published_groups": int(across.sum()),
                "labels_with_at_least_one_pair": int(len(set(li[involved].tolist()))),
                "audit_rule_pairs_the_detector_must_find": len(must_find),
                "audit_rule_pairs_missed": int(missed)}
    print("  " + ", ".join(f"{k} {v}" for k, v in detector.items()), flush=True)
    for key, value in PUBLISHED_DETECTOR.items():
        if detector[key] != value:
            raise SystemExit(f"the detector differs from within_label_repeats.json: {key} "
                             f"{detector[key]} against {value}")
    complement = ~involved
    print(f"  complement set (own label holds no passage-sharing partner): {int(complement.sum()):,}", flush=True)

    # ---- arm (a), the same merge within_label_repeats_v3 performs
    name_a = wlr.merge_groups(gi, pair_index)
    pop_a = wlr.make_population(base["songs"], base["label_of"], np.ones(n, dtype=bool), name_a,
                                base["fold_of_group"])
    require_published_population(pop_a, n, base["label_count"], "arm (a)")
    report_a = merge_report(base, name_a, pop_a)
    for key, value in PUBLISHED_MERGE.items():
        if report_a[key] != value:
            raise SystemExit(f"arm (a) differs from within_label_repeats.json: {key} {report_a[key]} "
                             f"against {value}")
    clusters = label_clusters(base, name_a)
    lost_a = components_lost_per_label(base, name_a)
    if sum(len(c) - 1 for v in clusters.values() for c in v) != sum(lost_a.values()):
        raise SystemExit("the cluster arithmetic is not arm (a)'s realised component loss")
    forbidden = merged_groups_of(base, name_a)
    reachable = sum(1 for a, b in pair_index if gi[a] != gi[b]
                    and int(gi[a]) not in forbidden and int(gi[b]) not in forbidden)
    if reachable:
        raise SystemExit(f"{reachable} cross-group twin pairs are reachable by the draw pool; "
                         f"the control could remove a twin and is not a control")
    holds: dict[int, set[int]] = defaultdict(set)
    for merged, group in zip(name_a.tolist(), gi.tolist()):
        holds[int(merged)].add(int(group))
    multi_label = sum(1 for merged, groups in holds.items()
                      if len(groups) > 1 and len(set(li[name_a == merged].tolist())) > 1)
    sizes = component_sizes(base)
    pool_by_label = groups_by_label(base)
    arm_a_clusters = {"labels_with_a_cluster": len(clusters),
                      "clusters": int(sum(len(v) for v in clusters.values())),
                      "cluster_cardinalities": {str(k): int(v) for k, v in
                                                sorted(Counter(len(c) for v in clusters.values()
                                                               for c in v).items())},
                      "components_merged_away": int(sum(lost_a.values())),
                      "merged_groups_forbidden_to_the_control": len(forbidden),
                      "merged_groups_touching_more_than_one_label": int(multi_label),
                      "cross_group_twin_pairs_reachable_by_the_draw_pool": 0}
    print(f"arm (a): {arm_a_clusters['clusters']} clusters in {arm_a_clusters['labels_with_a_cluster']} "
          f"labels, {arm_a_clusters['components_merged_away']} components merged away, "
          f"{report_a['groups_before']:,} -> {report_a['groups_after']:,} groups", flush=True)

    # ---- the control draws, every one gated before a single arm is scored
    print(f"building {args.draws} size-matched random merges", flush=True)
    controls = []
    for draw in range(args.draws):
        rng = np.random.default_rng(SEED + DRAW_SEED_OFFSET + draw)
        control_pairs, achieved = draw_size_matched(base, clusters, forbidden, sizes, pool_by_label, rng)
        if achieved["clusters_short_of_available_groups"]:
            raise SystemExit(f"draw {draw}: {achieved['clusters_short_of_available_groups']} clusters ran "
                             f"out of drawable groups, so the control merge is smaller than arm (a)'s")
        name_c = wlr.merge_groups(gi, control_pairs)
        lost_c = components_lost_per_label(base, name_c)
        off = [l for l in lost_a if lost_a[l] != lost_c[l]]
        achieved["labels_whose_component_loss_differs_from_arm_a"] = int(len(off))
        achieved["extra_components_lost_beyond_arm_a"] = int(sum(lost_c[l] - lost_a[l] for l in off))
        achieved["components_lost_per_label_matches_arm_a"] = bool(not off)
        if off:
            raise SystemExit(f"draw {draw}: {len(off)} labels lose a different number of components than "
                             f"arm (a) ({achieved['extra_components_lost_beyond_arm_a']:+d} components); "
                             f"the control is not size-matched per label")
        pop_c = wlr.make_population(base["songs"], base["label_of"], np.ones(n, dtype=bool), name_c,
                                    base["fold_of_group"])
        require_published_population(pop_c, n, base["label_count"], f"control draw {draw}")
        report_c = merge_report(base, name_c, pop_c)
        controls.append({"draw": draw, "seed": int(SEED + DRAW_SEED_OFFSET + draw), "name": name_c,
                         "pop": pop_c, "achieved": achieved,
                         "merge": {k: report_c[k] for k in CONTROL_MERGE_KEYS}})
        print(f"  draw {draw}: {achieved['components_merged_away']} components, "
              f"{report_c['groups_merged_away']} groups merged away, size match "
              f"{achieved['component_size_match']}, largest group {report_c['largest_group_after']}", flush=True)

    # ---- the published arm, gated on every published number clause 1 reads
    cache_dir = args.cache_dir.resolve()
    spaces_pub = ArmSpaces(base, pop_pub, chars_pub, words_pub)
    print("scoring the published configuration", flush=True)
    rr = {}
    cell_minutes = {}
    rr["published"], cell_minutes["published"] = score_arm("published", pop_pub, spaces_pub, grid_spaces,
                                                           True, cache_dir)
    against_published = {}
    for space in PROTO_SPACES:
        value = float(np.mean(rr["published"][space][0.0]))
        gap = abs(value - EXPECTED[space])
        against_published[space] = {"estimand": "query-weighted", "expected": EXPECTED[space],
                                    "recomputed": round(value, 4), "gap": round(gap, 5),
                                    "tolerance": TOLERANCE[space], "passed": bool(gap <= TOLERANCE[space])}
        print(f"check {space}: prototype {value:.4f} expected {EXPECTED[space]}", flush=True)
    curve_check = {}
    for space in grid_spaces:
        curve_check[space] = {}
        for beta in BETAS:
            value = float(np.mean(rr["published"][space][beta]))
            gap = abs(value - PUBLISHED_CURVE[space][beta])
            curve_check[space][f"{beta:g}"] = {"expected": PUBLISHED_CURVE[space][beta],
                                               "recomputed": round(value, 4), "gap": round(gap, 5),
                                               "passed": bool(gap <= TOLERANCE[space])}
        rr["published"][space]["selected"] = selected_rr(rr["published"][space], pop_pub["fold"],
                                                         SELECTED_BETA[space])
        value = float(np.mean(rr["published"][space]["selected"]))
        gap = abs(value - PUBLISHED_SELECTED[space])
        curve_check[space]["selected"] = {"expected": PUBLISHED_SELECTED[space], "recomputed": round(value, 4),
                                          "gap": round(gap, 5), "passed": bool(gap <= TOLERANCE[space]),
                                          "beta_by_fold": [float(b) for b in SELECTED_BETA[space]]}
        print(f"check {space}: the published curve and its selected {value:.4f} against "
              f"{PUBLISHED_SELECTED[space]}", flush=True)
    checks["against_published_prototype"] = against_published
    checks["against_published_beta_curve"] = curve_check
    failed = [s for s, c in against_published.items() if not c["passed"]]
    failed += [f"{s}/beta {b}" for s in curve_check for b, c in curve_check[s].items() if not c["passed"]]
    if failed:
        raise SystemExit(f"the published numbers do not reproduce: {failed}")

    # ---- the stratifier of clause 1b, on the published arm, gated against temperature_scale.json
    print("measuring each query's own top attention weight", flush=True)
    top_weight, quintile, quintile_edges, top_weight_check = {}, {}, {}, {}
    for space in grid_spaces:
        beta_of_query = np.asarray([float(SELECTED_BETA[space][int(k)]) for k in pop_pub["fold"].tolist()])
        values = top_attention_weight(spaces_pub[space], pop_pub, beta_of_query, space)
        top_weight[space] = values
        quintile[space], quintile_edges[space] = quintile_of(values)
        got = {"q50": round(float(np.quantile(values, 0.5)), 4), "q95": round(float(np.quantile(values, 0.95)), 4)}
        top_weight_check[space] = {
            "expected": PUBLISHED_TOP_WEIGHT[space], "recomputed": got, "tolerance": TOP_WEIGHT_TOLERANCE,
            "passed": bool(all(abs(got[q] - PUBLISHED_TOP_WEIGHT[space][q]) <= TOP_WEIGHT_TOLERANCE
                               for q in got)),
            "quintile_edges": quintile_edges[space]}
        print(f"  {space}: top weight median {got['q50']} (published {PUBLISHED_TOP_WEIGHT[space]['q50']}), "
              f"q95 {got['q95']} (published {PUBLISHED_TOP_WEIGHT[space]['q95']})", flush=True)
    checks["against_published_top_weight_share"] = top_weight_check
    bad = [s for s in grid_spaces if not top_weight_check[s]["passed"]]
    if bad:
        raise SystemExit(f"the published top attention weight does not reproduce in {bad}; the clause 1b "
                         f"stratifier is not temperature_scale_v3's quantity")

    # ---- clause 1 and 1b, on the published population, no merge involved
    print("clause 1: the attention gain on twin-free queries", flush=True)
    masks = {"complement": complement, "has_a_partner": involved, "all": np.ones(n, dtype=bool)}
    clause_one = {}
    for space in grid_spaces:
        systems = {"prototype": rr["published"][space][0.0],
                   "attention_selected": rr["published"][space]["selected"]}
        entry = {"beta_by_fold": [float(b) for b in SELECTED_BETA[space]], "sets": {}}
        for set_name, mask in masks.items():
            cell = stratum(systems, w, gi, li, mask)
            cell["estimand"] = ("component-weighted, point and interval (paired group bootstrap over the "
                                "published leakage groups)")
            cell["prototype_mrr_component_weighted"] = round(component_weighted(systems["prototype"], w, mask), 4)
            cell["selected_mrr_component_weighted"] = round(component_weighted(systems["attention_selected"],
                                                                               w, mask), 4)
            cell["prototype_mrr_query_weighted"] = round(query_weighted(systems["prototype"], mask), 4)
            cell["selected_mrr_query_weighted"] = round(query_weighted(systems["attention_selected"], mask), 4)
            if set_name == "complement":
                cell["prototype_mrr_label_macro"] = round(label_macro(systems["prototype"], w, li, mask), 4)
                cell["selected_mrr_label_macro"] = round(label_macro(systems["attention_selected"],
                                                                     w, li, mask), 4)
                curve = {f"{b:g}": round(component_weighted(rr["published"][space][b], w, mask), 4)
                         for b in BETAS}
                cell["mrr_by_beta_component_weighted"] = curve
                cell["curve_shape"] = curve_shape(curve)
            entry["sets"][set_name] = cell
        point = entry["sets"]["all"]["gain_component_weighted"]["mrr_difference"]
        gap = abs(point - PUBLISHED_GAIN[space])
        entry["gain_on_all_queries_against_published"] = {
            "estimand": "component-weighted (the published estimand)", "expected": PUBLISHED_GAIN[space],
            "recomputed": point, "gap": round(gap, 5), "tolerance": GAIN_TOLERANCE[space],
            "tolerance_note": "twice the level tolerance: a contrast is a difference of two levels",
            "passed": bool(gap <= GAIN_TOLERANCE[space])}
        bands = {}
        for band, _, _ in BANDS:
            bands[band] = stratum(systems, w, gi, li, (band_of_song == band) & complement)
        entry["complement_by_songs_per_label_band"] = {
            "banding": "songs per label (label_size_calibration_v3's BANDS), NOT components per label",
            "bands": bands}
        by_quintile = {}
        for set_name in ("complement", "has_a_partner"):
            by_quintile[set_name] = {str(q): stratum(systems, w, gi, li, masks[set_name] & (quintile[space] == q))
                                     for q in range(QUINTILES)}
        entry["by_top_attention_weight_quintile"] = {
            "stratifier": "the query's own top attention weight at its pre-specified beta, the quantity "
                          "temperature_scale_v3 publishes as top_weight_share; a function of the query's "
                          "cosine profile in a fixed space, not of either system's ranks",
            "edges": quintile_edges[space], "sets": by_quintile}
        bottom, top = by_quintile["complement"]["0"], by_quintile["complement"][str(QUINTILES - 1)]
        entry["concentration_reading"] = concentration_reading(bottom.get("reading"), top.get("reading"))
        clause_one[space] = entry
        cell = entry["sets"]["complement"]
        print(f"  {space}: complement {cell['queries']:,} queries, gain "
              f"{cell['gain_component_weighted']['mrr_difference']:+.4f} "
              f"{cell['gain_component_weighted']['ci95']} -> {cell['reading']['verdict']}", flush=True)
        print(f"    by top weight: bottom quintile {bottom.get('reading', {}).get('verdict', 'withheld')}; "
              f"top quintile {top.get('reading', {}).get('verdict', 'withheld')} -> "
              f"{entry['concentration_reading']['verdict']}", flush=True)
    bad = [s for s in grid_spaces if not clause_one[s]["gain_on_all_queries_against_published"]["passed"]]
    if bad:
        raise SystemExit(f"the published component-weighted attention gain does not reproduce: {bad}")

    # ---- arm (a) and the draws. Nothing below this line raises.
    print("scoring arm (a)", flush=True)
    rr["merged_groups"], cell_minutes["merged_groups"] = score_arm(
        "merged_groups", pop_a, ArmSpaces(base, pop_a, chars_pub, words_pub), grid_spaces, False, cache_dir)
    against_arm_a = {}
    for space in PROTO_SPACES:
        value = float(np.mean(rr["merged_groups"][space][0.0]))
        gap = abs(value - PUBLISHED_ARM_A[space])
        against_arm_a[space] = {"estimand": "query-weighted", "expected": PUBLISHED_ARM_A[space],
                               "recomputed": round(value, 4), "gap": round(gap, 5),
                               "tolerance": TOLERANCE[space], "passed": bool(gap <= TOLERANCE[space])}
        print(f"check arm (a) {space}: {value:.4f} expected {PUBLISHED_ARM_A[space]}", flush=True)
    checks["against_published_arm_a"] = against_arm_a
    if not all(c["passed"] for c in against_arm_a.values()):
        raise SystemExit(f"arm (a) does not reproduce within_label_repeats.json: {against_arm_a}")
    for space in grid_spaces:
        rr["merged_groups"][space]["selected"] = selected_rr(rr["merged_groups"][space], pop_a["fold"],
                                                             SELECTED_BETA[space])
    for control in controls:
        key = f"control_{control['draw']:02d}"
        print(f"scoring {key}", flush=True)
        rr[key], cell_minutes[key] = score_arm(key, control["pop"],
                                               ArmSpaces(base, control["pop"], chars_pub, words_pub),
                                               grid_spaces, False, cache_dir)
        for space in grid_spaces:
            rr[key][space]["selected"] = selected_rr(rr[key][space], control["pop"]["fold"],
                                                     SELECTED_BETA[space])

    # ---- clause 2, per space, per draw
    print("clause 2: arm (a) against the size-matched control", flush=True)
    everything = np.ones(n, dtype=bool)
    unit = np.ones(n)
    keys = [f"control_{c['draw']:02d}" for c in controls]
    levels = {}
    for space in PROTO_SPACES:
        published = query_weighted(rr["published"][space][0.0], everything)
        merged = query_weighted(rr["merged_groups"][space][0.0], everything)
        draws = [query_weighted(rr[k][space][0.0], everything) for k in keys]
        raw_drop = merged - published
        levels[space] = {
            "estimand": "query-weighted on the published query set",
            "published": round(published, 4), "merged_groups": round(merged, 4),
            "control_mean": round(float(np.mean(draws)), 4), "control_min": round(float(np.min(draws)), 4),
            "control_max": round(float(np.max(draws)), 4), "raw_drop": round(raw_drop, 4),
            "control_drop": round(float(np.mean(draws)) - published, 4),
            "residual": round(merged - float(np.mean(draws)), 4),
            "published_raw_drop": PUBLISHED_RAW_DROP[space],
            # not a gate: both levels are already gated against their published values above, and a
            # gate here would abort after the expensive work.
            "raw_drop_reproduces_published": bool(abs(raw_drop - PUBLISHED_RAW_DROP[space])
                                                  <= 2 * TOLERANCE[space]),
            "share_of_the_raw_drop_the_size_control_explains":
                round((float(np.mean(draws)) - published) / raw_drop, 3) if raw_drop else None}

    per_draw = {space: [] for space in PROTO_SPACES}
    per_draw_gain = {space: [] for space in grid_spaces}
    for control, key in zip(controls, keys):
        groups = union_grouping(base, [name_a, control["name"]])
        systems, pair_list = {}, []
        for space in PROTO_SPACES:
            systems[f"{space}/a"] = rr["merged_groups"][space][0.0]
            systems[f"{space}/c"] = rr[key][space][0.0]
            pair_list.append((f"{space}/a", f"{space}/c"))
        for space in grid_spaces:
            # a difference of differences, exactly, because the estimand is linear in the reciprocal ranks
            systems[f"{space}/gain_a_plus_control_prototype"] = (rr["merged_groups"][space]["selected"]
                                                                 - rr["merged_groups"][space][0.0]
                                                                 + rr[key][space][0.0])
            systems[f"{space}/control_selected"] = rr[key][space]["selected"]
            pair_list.append((f"{space}/gain_a_plus_control_prototype", f"{space}/control_selected"))
        contrasts = paired_group_bootstrap(systems, unit, groups, everything, pair_list)
        units = int(len(np.unique(groups)))
        for space in PROTO_SPACES:
            contrast = dict(lookup(contrasts, f"{space}/a", f"{space}/c"))
            contrast.update(draw=control["draw"], resampling_units=units)
            per_draw[space].append(contrast)
        for space in grid_spaces:
            contrast = dict(lookup(contrasts, f"{space}/gain_a_plus_control_prototype",
                                   f"{space}/control_selected"))
            contrast.update(draw=control["draw"], resampling_units=units)
            per_draw_gain[space].append(contrast)
        print(f"  draw {control['draw']}: {units:,} resampling units  " + "  ".join(
            f"{space}={per_draw[space][-1]['mrr_difference']:+.4f}" for space in PROTO_SPACES), flush=True)

    decomposition = {}
    for space in PROTO_SPACES:
        decomposition[space] = {
            "estimand": "query-weighted, point and interval (paired group bootstrap over the union of "
                        "arm (a)'s and the draw's groupings, unit weights)",
            "levels": levels[space], "residual_per_draw": per_draw[space],
            "reading": residual_reading(
                per_draw[space],
                "mostly repertoire shrinkage: merging twins costs no more than the margin beyond a "
                "size-matched random merge",
                "twin-specific beyond the size control",
                "the size-matched control loses more than arm (a)")}
        print(f"  {space}: raw drop {levels[space]['raw_drop']:+.4f}, size control "
              f"{levels[space]['control_drop']:+.4f}, residual {levels[space]['residual']:+.4f} -> "
              f"{decomposition[space]['reading']['verdict']}", flush=True)
    gain_decomposition = {}
    for space in grid_spaces:
        gains = {arm: round(query_weighted(rr[arm][space]["selected"], everything)
                            - query_weighted(rr[arm][space][0.0], everything), 4)
                 for arm in ("published", "merged_groups")}
        draws_gain = [query_weighted(rr[k][space]["selected"], everything)
                      - query_weighted(rr[k][space][0.0], everything) for k in keys]
        gains.update(control_mean=round(float(np.mean(draws_gain)), 4),
                     control_min=round(float(np.min(draws_gain)), 4),
                     control_max=round(float(np.max(draws_gain)), 4))
        gain_decomposition[space] = {
            "estimand": "query-weighted difference of differences, point and interval",
            "gain_query_weighted": gains, "difference_in_differences_per_draw": per_draw_gain[space],
            "reading": residual_reading(
                per_draw_gain[space],
                "the attention gain is the same within the margin under arm (a) and under the "
                "size-matched control",
                "the gain is smaller under arm (a) than under the size-matched control",
                "the gain is larger under arm (a) than under the size-matched control")}
        print(f"  gain {space}: published {gains['published']:+.4f}, arm (a) {gains['merged_groups']:+.4f}, "
              f"control {gains['control_mean']:+.4f} -> {gain_decomposition[space]['reading']['verdict']}",
              flush=True)

    # ---- the reading
    lexical = [s for s in grid_spaces if s in LEXICAL]
    readings = {s: clause_one[s]["sets"]["complement"]["reading"] for s in grid_spaces}
    one = clause_one_verdict(readings, lexical)
    clause_two = {space: decomposition[space]["reading"]["verdict"] for space in PROTO_SPACES}
    reading = {
        "clause_1_complement_set_gain": {
            "by_space": {s: readings[s]["verdict"] for s in grid_spaces}, "lexical_spaces": lexical,
            "curve_shape_on_the_complement_set": {s: clause_one[s]["sets"]["complement"]["curve_shape"]
                                                  for s in grid_spaces}, **one},
        "clause_1b_where_the_gain_sits": {s: clause_one[s]["concentration_reading"]["verdict"]
                                          for s in grid_spaces},
        "clause_2_drop_decomposition": {
            "by_space": clause_two,
            "spaces_where_the_raw_drop_is_mostly_repertoire_shrinkage":
                [s for s, v in clause_two.items() if v.startswith("mostly repertoire shrinkage")],
            "gain_by_space": {s: gain_decomposition[s]["reading"]["verdict"] for s in grid_spaces}},
        "margin": MARGIN, "quorum": QUORUM,
    }
    print(f"reading, clause 1: {one['verdict']}", flush=True)
    print(f"reading, clause 1b: {reading['clause_1b_where_the_gain_sits']}", flush=True)
    print(f"reading, clause 2: {clause_two}", flush=True)

    payload = {
        "analysis": "twin_control_v3: a size-matched random-merge control for the within-label "
                    "repeated-passage arm, and the attention gain on queries with no passage-sharing partner",
        "question": "is the repeated-passage penalty a duplicate effect or a repertoire-size effect, and "
                    "does the attention gain survive on twin-free queries?",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, **population},
        "design": {
            "what_this_arm_bounds": "temperature_scale_v3 (2026-09-22) measured the attention weights at "
                                    "the selected temperature: effective sample size 22.7 to 39.4 songs at "
                                    "the median against a median 40-song profile, largest single weight "
                                    "0.033 to 0.141 at the median and 0.447 at the 95th percentile in raw "
                                    "words. The duplicate account is therefore already weakened, and this "
                                    "arm bounds the residual duplicate contribution rather than deciding "
                                    "whether the effect exists",
            "detector": f"within_label_repeats_v3.shared_passage_pairs: two same-label songs share a "
                        f"passage iff their normalised texts share a {wlr.PASSAGE_CHARACTERS}-character "
                        f"substring; imported, not re-implemented, and gated against the counts in "
                        f"within_label_repeats.json",
            "arm_a": "within_label_repeats_v3.merge_groups over the cross-group pairs, population by its "
                     "own make_population; gated against its recorded merge counts and MRRs",
            "control": "in the labels arm (a) merges, each of its clusters is replicated with groups drawn "
                       "at random from that label's groups arm (a) did not merge, greedily matched on "
                       "component size, never two groups sharing another label; every draw's realised "
                       "component loss is checked per label against arm (a)'s and the run stops if any "
                       "label differs, so the control shrinks the same repertoires and no others and can "
                       "remove no cross-group twin",
            "draws": args.draws, "draw_seeds": f"SEED + {DRAW_SEED_OFFSET} + draw index",
            "spaces_prototype": list(PROTO_SPACES), "spaces_beta_grid": list(grid_spaces),
            "betas": list(BETAS),
            "beta_grid_scope": "the whole curve on the published arm; beta 0 and the pre-specified betas "
                               "on every other arm, which is all clause 2 reads",
            "selected_beta": {s: [float(b) for b in SELECTED_BETA[s]] for s in grid_spaces},
            "selection": "none happens here: the betas are the fixed constants attention_exemplar_v3 "
                         "selected, imported from temperature_scale_v3",
            "primary_estimand": "the attention gain on the complement set (queries whose own label holds "
                                "no passage-sharing partner of theirs), component-weighted with a paired "
                                "group-bootstrap interval, on the published population",
            "estimands": {
                "query_weighted": "np.mean(1/rank); the estimand of every `mrr` level in this directory "
                                  "and of every cross-arm difference here",
                "component_weighted": "sum(rr*w)/sum(w) over (group, label) components; what "
                                      "paired_group_bootstrap returns, and the estimand of the "
                                      "complement-set gain",
                "label_macro": "component-weighted mean within a label, then the plain mean over labels; "
                               "reported for the complement-set levels over the published numbering",
                "cross_arm_weighting": "unit weights, so a cross-arm difference is query-weighted and does "
                                       "not mix the merge's reweighting with the rank change",
            },
            "banding": "songs per label (label_size_calibration_v3's BANDS: 5-9, 10-19, 20-49, 50-up), "
                       "checked against 19/149, 33/477, 171/6356, 3/238. NOT the components-per-label "
                       "banding, whose '20-49' is a different query set",
            "small_strata": f"a band or quintile with fewer than {MIN_CELL_LABELS} labels or "
                            f"{MIN_CELL_QUERIES} queries is reported as counts only, with no interval",
            "margin": MARGIN, "quorum": QUORUM,
            "reading_rule": (
                "Clause 1, primary. On the complement set, the gain of attention at the pre-specified beta "
                "minus the prototype, component-weighted: 'survives' if the point is positive and the whole "
                "interval is above zero in both lexical spaces; 'bounded below the margin' if the interval "
                "includes zero and lies wholly below +0.005, a negative result and not an absence of "
                "evidence; 'undecided' if the interval includes zero and reaches above the margin, and "
                "undecided is tested first so one space's bound cannot mask another's undecided. "
                "Clause 1b, secondary. Quintiles of the query's own top attention weight: the "
                "residual-duplicate account is contradicted if the gain survives in the bottom quintile, "
                "supported if it survives only in the top one, undecided otherwise. Clause 2, equivalence. "
                "residual = arm (a) minus the mean over draws of the size-matched control, query-weighted, "
                "with a paired group-bootstrap interval per draw over the union grouping: 'mostly "
                "repertoire shrinkage' only if the whole interval lies inside +/-0.005 in at least 18 of "
                "20 draws; 'twin-specific beyond the size control' if the point is at most -0.005 with the "
                "whole interval below zero in 18 draws; otherwise undecided. The same decomposition is "
                "applied to the attention gain as a difference of differences."),
            "order_of_work": "every invariant and every comparison against a published number is checked "
                             "before the control arms are scored; nothing after that point raises",
            "not_in_the_rule": "every count; the merge and match reports; the beta curves; the band and "
                               "quintile tables beyond the two quintiles clause 1b names",
            "scope_note": "the word SVD-1024 space is not scored and its published 0.5426 is not "
                          "reproduced here: within_label_repeats_v3 never scored it, its published "
                          "attention gain is +0.0022, and refitting its SVD for every arm would cost more "
                          "than the rest together",
            "cache": "one .npz per (arm, space) cell under --cache-dir, keyed by a fingerprint of the "
                     "corpus digest, the arm's population and the betas; holds per-query reciprocal ranks, "
                     "so it lives outside results/ and nothing per query reaches this file",
        },
        "checks": checks,
        "detector": detector,
        "complement_set": {"queries": int(complement.sum()), "queries_with_a_partner": int(involved.sum()),
                           "definition": f"the query's own label holds no song sharing a "
                                         f"{wlr.PASSAGE_CHARACTERS}-character run with it"},
        "arms": {
            "published": {"population": population},
            "merged_groups": {"merge": report_a, "clusters": arm_a_clusters},
            "control": {"draws": [{"draw": c["draw"], "seed": c["seed"], "achieved_match": c["achieved"],
                                   "merge": c["merge"]} for c in controls],
                        "over_draws": {
                            "components_merged_away": sorted({c["achieved"]["components_merged_away"]
                                                              for c in controls}),
                            "components_lost_per_label_matches_arm_a_in_every_draw":
                                bool(all(c["achieved"]["components_lost_per_label_matches_arm_a"]
                                         for c in controls)),
                            "component_size_match_min": round(float(np.min([c["achieved"]["component_size_match"]
                                                                            for c in controls])), 4),
                            "component_size_match_median": round(float(np.median(
                                [c["achieved"]["component_size_match"] for c in controls])), 4),
                            "groups_merged_away_min": int(np.min([c["merge"]["groups_merged_away"]
                                                                  for c in controls])),
                            "groups_merged_away_max": int(np.max([c["merge"]["groups_merged_away"]
                                                                  for c in controls])),
                            "largest_group_after_max": int(np.max([c["merge"]["largest_group_after"]
                                                                   for c in controls])),
                            "arm_a_groups_merged_away": report_a["groups_merged_away"],
                            "note": "a control merge is confined to one label at a time, so it can merge "
                                    "away more GROUPS than arm (a) for the same number of COMPONENTS; "
                                    "components are the quantity the size theory is about (the protocol's "
                                    "weights are 1/component, so a label's effective profile count is its "
                                    "components) and they are matched per label and checked"}}},
        "clause_1_complement_gain": clause_one,
        "clause_2_prototype_decomposition": decomposition,
        "clause_2_attention_gain_decomposition": gain_decomposition,
        "reading": reading,
        "cell_minutes": cell_minutes,
        "privacy": "aggregate only: counts, medians, quantiles, bands, MRRs and intervals. No song "
                   "identifier, label name, artist, lyric text or per-label listing appears here or in any "
                   "file this script writes to results/; a stratum too small to aggregate is reported as "
                   "counts alone",
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / OUT_NAME
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nwrote {path}  ({payload['minutes']} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
