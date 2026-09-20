#!/usr/bin/env python3
"""Does the transfer of whitening to unseen labels depend on the one held-out draw?

QUESTION. identity_probe_v2 asks whether the identity metric fitted on some labels' songs
still helps on labels it never saw. It holds out 15% of the labels by ONE fixed-seed
permutation (rng(20260825).permutation(labels), independent of the folds), fits each
transform on the seen labels' songs only, and scores the held-out labels' queries against
the usual leave-group-out profiles of all labels. On corpus v3 (1.3.0) that is 34 of 226
labels and 1,041 queries: none 0.2658, total whitening 0.4072, within-author whitening
0.3813, and the same queries in the fold-wise within-author space where their labels were
seen 0.3814. The published reading -- the label-free part of the gain transfers to unseen
labels, the label-specific part does not -- rests on that single draw of 34 labels. This
file asks whether five further draws read the same way.

DESIGN
  draw 0      the probe's draw, formula for formula:
              rng(20260825).permutation(226)[:round(0.15 x 226)]
  draws 1-5   seeds 20260918, 20260919, 20260920, 20260921, 20260922. Draw i permutes, with
              its own seed, the labels that no earlier draw held out and takes the first
              round(0.15 x 226) = 34, so the six held-out sets are pairwise disjoint blocks
              of the label set (6 x 34 = 204 of 226 labels; 22 labels are never held out).
              If a block could not be completed from never-held-out labels it would be
              topped up from labels outside the previous draw and the run would stop,
              because the reading below is defined for disjoint blocks.
  per draw    identity_probe_v2.fit_transform none / total_whitening /
              within_author_whitening, fitted on the seen labels' songs only (the
              protocol's per-(group, label) weights, Ledoit-Wolf shrinkage, rng(SEED) as
              the probe passes it), applied to every song; the held-out labels' queries
              scored by identity_probe_v2.dense_leave_group_out against profiles of all 226
              labels with the query's leakage group removed, exactly the probe's transfer
              block. The seen-label comparison is the probe's fold-wise within-author space
              (fitted on the other four folds, every label seen), restricted to the same
              queries.

SCORERS AND ESTIMANDS, over a draw's held-out queries
  systems     unseen_none, unseen_total_whitening, unseen_within_author_whitening,
              seen_within_author_whitening
  mrr         the plain mean of reciprocal ranks (v1.rank_system's rank, ties broken by
              label index as everywhere in the protocol) -- what identity_probe.json
              reports as "mrr" and what the checks compare
  mrr_component_weighted   the per-(group, label) component-weighted MRR, which is the
              bootstrap's estimand
  recall_at_10, and mrr_label_macro (mean over the held-out labels of the label's
              component-weighted MRR) as description
  contrasts   each with a paired group-bootstrap 95% interval from
              identity_spaces_v2.paired_group_bootstrap (the draw's leakage groups
              resampled with replacement, 2000 replicates, seed 20260825, one resample
              driving every system):
                unseen_total_whitening - unseen_none               the label-free part
                unseen_within_author_whitening - unseen_none
                unseen_within_author_whitening - unseen_total_whitening
                                                                   the label-specific part
                seen_within_author_whitening - unseen_within_author_whitening
  spread      over the six draws, of every system's MRRs and every contrast: min, max,
              mean, sample sd (ddof 1)

CHECKS, before any redraw is scored (SystemExit on failure)
  1. draw 0 reproduces identity_probe.json's held-out numbers on unrounded values: none
     0.2658 (gap <= 5e-4, no fitting involved), total whitening 0.4072, within-author
     whitening 0.3813 and seen-label within-author 0.3814 (gap <= 0.002); the literals
     pinned here are matched against the file, as are its 34 labels and 1,041 queries.
  2. over all 7,220 queries the untransformed space gives 0.2997 (gap <= 5e-4, the
     tolerance identity_probe_v2 uses for it) and the fold-wise within-author space
     0.4164 (gap <= 0.002), so queries, groups, weights, folds and scorer are the probe's.
  3. every draw holds out exactly round(0.15 x 226) labels; the six sets are pairwise
     disjoint; every draw's fit uses only songs of seen labels (the training mask holds
     no song of a held-out label).
  The synthetic test (test_heldout_label_redraw.py, random data, no corpus) checks the
  transfer scoring path and the fold-wise space against a brute-force leave-group-out
  cosine, that a draw's fit is blind to the held-out labels' songs, the rank and MRR
  helpers against a brute-force rank, the bootstrap's point estimate, the draw
  construction, the gap check, the reading rule and the spread helper.

READING RULE, fixed before the run
  'the label-free part transfers' if unseen_total_whitening - unseen_none is positive with
  the 95% interval clear of zero in EVERY one of the six draws; otherwise the reading
  states in how many draws it was.
  'the label-specific part does not transfer' only if unseen_within_author_whitening -
  unseen_total_whitening is NOT positive in at least five of the six draws, where
  "positive" means positive with the 95% interval clear of zero (so "not positive" is a
  point estimate at or below zero, or an interval reaching zero); otherwise the reading
  states the count.
  Reported beside the rule as description, not entering it: the number of draws in which
  the label-specific contrast is negative with the interval clear of zero, the number with
  a point estimate at or below zero, the seen - unseen contrast and its count of intervals
  clear of zero, the label-macro MRRs and the spreads.

    set CHINESE_RAP_CORPUS=v3
    set PYTHONIOENCODING=utf-8
    python src/heldout_label_redraw_v3.py --private-root <ni-k>

Output: results/retrieval-v3/heldout_label_redraw.json, aggregate only: no lyric text,
song identifiers, label names or per-song vectors, printed or written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


REPO = ROOT
sys.path.insert(0, str(REPO / "src"))

from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from identity_probe_v2 import (  # noqa: E402
    FOLDS,
    HELD_OUT_LABEL_SHARE,
    SEED,
    dense_leave_group_out,
    fit_transform,
    unit_rows,
)
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = REPO / "results" / "retrieval-v3"
OUT_NAME = "heldout_label_redraw.json"
PROBE_JSON = OUT_DIR / "identity_probe.json"
REDRAW_SEEDS = (20260918, 20260919, 20260920, 20260921, 20260922)
TRANSFORMS = ("none", "total_whitening", "within_author_whitening")
SYSTEMS = ("unseen_none", "unseen_total_whitening", "unseen_within_author_whitening",
           "seen_within_author_whitening")
PAIRS = (("unseen_total_whitening", "unseen_none"),
         ("unseen_within_author_whitening", "unseen_none"),
         ("unseen_within_author_whitening", "unseen_total_whitening"),
         ("seen_within_author_whitening", "unseen_within_author_whitening"))
LABEL_FREE = ("unseen_total_whitening", "unseen_none")
LABEL_SPECIFIC = ("unseen_within_author_whitening", "unseen_total_whitening")
SEEN_MINUS_UNSEEN = ("seen_within_author_whitening", "unseen_within_author_whitening")
RULE_MIN_NOT_POSITIVE = 5          # of the six draws
CHECK_GAP_FITTED = 0.002
CHECK_GAP_UNFITTED = 5e-4
# identity_probe.json (corpus v3 1.3.0): unseen_author_transfer.systems.*.mrr and systems.*.mrr
EXPECTED = {"unseen_none": 0.2658, "unseen_total_whitening": 0.4072,
            "unseen_within_author_whitening": 0.3813, "seen_within_author_whitening": 0.3814,
            "all_queries_none": 0.2997, "all_queries_within_author_whitening": 0.4164}
TOLERANCE = {name: (CHECK_GAP_UNFITTED if name in ("unseen_none", "all_queries_none") else CHECK_GAP_FITTED)
             for name in EXPECTED}
PROBE_KEYS = {"unseen_none": "unseen_authors_none",
              "unseen_total_whitening": "unseen_authors_total_whitening",
              "unseen_within_author_whitening": "unseen_authors_within_author_whitening",
              "seen_within_author_whitening": "seen_authors_within_author_whitening"}
PROBE_ALL_KEYS = {"all_queries_none": "none",
                  "all_queries_within_author_whitening": "within_author_whitening"}


# ------------------------------------------------------------------ draws
def probe_draw(label_count: int, share: float = HELD_OUT_LABEL_SHARE, seed: int = SEED) -> np.ndarray:
    """identity_probe_v2's held-out set, formula for formula."""
    held = np.zeros(label_count, dtype=bool)
    held[np.random.default_rng(seed).permutation(label_count)[:int(round(share * label_count))]] = True
    return held


def held_out_draws(label_count: int, seeds=REDRAW_SEEDS, share: float = HELD_OUT_LABEL_SHARE):
    """Draw 0 is the probe's; draw i takes, in the order of rng(seed_i), the first `size`
    labels no earlier draw held out. Returns (list of boolean masks, list of seeds, disjoint)."""
    size = int(round(share * label_count))
    draws = [probe_draw(label_count, share)]
    taken = draws[0].copy()
    disjoint = True
    for seed in seeds:
        rng = np.random.default_rng(seed)
        remaining = np.flatnonzero(~taken)
        pick = remaining[rng.permutation(len(remaining))[:size]]
        if len(pick) < size:
            # the block cannot be completed from labels never held out: top it up from
            # labels outside the previous draw, in the same seeded order; not disjoint
            disjoint = False
            outside = np.flatnonzero(taken & ~draws[-1])
            pick = np.concatenate([pick, outside[rng.permutation(len(outside))[:size - len(pick)]]])
        held = np.zeros(label_count, dtype=bool)
        held[pick] = True
        if int(held.sum()) != size:
            raise RuntimeError("a held-out draw does not have the required size")
        draws.append(held)
        taken |= held
    return draws, [SEED, *seeds], disjoint


# ------------------------------------------------------------------ scoring
def transfer_scores(dense, label_index, group_ids, weights, label_count, held):
    """identity_probe_v2's transfer block for one held-out set: each transform fitted on the
    seen labels' songs only, applied to every song, the held-out labels' queries scored
    leave-group-out against all labels. Returns (queries, {transform: scores}, {transform: fit info})."""
    unseen = held[label_index]
    queries = np.flatnonzero(unseen)
    if queries.size == 0:
        raise RuntimeError("a draw holds out no query")
    if np.any(held[label_index[~unseen]]):
        raise RuntimeError("the training mask contains a song of a held-out label")
    scores, info = {}, {}
    for name in TRANSFORMS:
        mean, matrix, diagnostics = fit_transform(name, dense[~unseen], label_index[~unseen],
                                                  weights[~unseen], np.random.default_rng(SEED))
        scores[name] = dense_leave_group_out(unit_rows((dense - mean) @ matrix.T), label_index,
                                             group_ids, weights, label_count, queries)
        info[name] = {"labels_seen": int(len(set(label_index[~unseen].tolist()))), **diagnostics}
    return queries, scores, info


def foldwise_within_author(dense, label_index, group_ids, weights, label_count, fold):
    """The probe's within_author_whitening system: per fold, fitted on the other folds with
    every label seen, that fold's queries scored in it. Scores for every query."""
    scores = np.zeros((len(label_index), label_count))
    for k in range(FOLDS):
        train = fold != k
        queries = np.flatnonzero(~train)
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train], label_index[train],
                                        weights[train], np.random.default_rng(SEED + k))
        scores[queries] = dense_leave_group_out(unit_rows((dense - mean) @ matrix.T), label_index,
                                                group_ids, weights, label_count, queries)
    return scores


def reciprocal_ranks(scores, truth) -> np.ndarray:
    return 1.0 / ranks_of(scores, truth)


def system_summary(rr: np.ndarray, weights, label_index, mask) -> dict:
    """MRR as the probe reports it (plain mean over the masked queries), the component-
    weighted MRR, recall@10 and the label-macro MRR over the labels present in the mask."""
    labels = np.unique(label_index[mask])
    macro = [weighted_mean(rr, weights, mask & (label_index == l)) for l in labels]
    return {"mrr": round(float(np.mean(rr[mask])), 4),
            "mrr_component_weighted": round(weighted_mean(rr, weights, mask), 4),
            "recall_at_10": round(float(np.mean(rr[mask] >= 0.1)), 4),
            "mrr_label_macro": round(float(np.mean(macro)), 4)}


def score_draw(d: dict, held: np.ndarray, rr_seen: np.ndarray):
    """One held-out set scored and summarised. Returns (entry, unrounded plain MRRs)."""
    li, gi, w, L, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["dense"]
    started = time.time()
    queries, scores, info = transfer_scores(dense, li, gi, w, L, held)
    mask = held[li]
    rr = {}
    for name in TRANSFORMS:
        full = np.ones(len(li))
        full[queries] = reciprocal_ranks(scores[name], li[queries])
        rr[f"unseen_{name}"] = full
    rr["seen_within_author_whitening"] = rr_seen
    systems = {name: system_summary(rr[name], w, li, mask) for name in SYSTEMS}
    exact = {name: float(np.mean(rr[name][mask])) for name in SYSTEMS}
    contrasts = paired_group_bootstrap(rr, w, gi, mask, list(PAIRS))
    sizes = np.bincount(li, minlength=L)[held]
    entry = {"held_out_labels": int(held.sum()), "queries": int(mask.sum()),
             "groups": int(len(np.unique(gi[mask]))),
             "songs_per_held_out_label": {"min": int(sizes.min()), "median": float(np.median(sizes)),
                                          "max": int(sizes.max())},
             "fit": {name: info[name] for name in TRANSFORMS if name != "none"},
             "systems": systems, "contrasts": contrasts,
             "seconds": round(time.time() - started, 1)}
    return entry, exact


# ------------------------------------------------------------------ summaries and the rule
def spread(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"draws": int(arr.size), "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "sd": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0}


def contrast_of(contrasts, pair):
    return next(c for c in contrasts if (c["system"], c["minus"]) == pair)


def positive_clear(contrast) -> bool:
    """Positive with the 95% interval clear of zero (the bootstrap's own unrounded flag)."""
    return bool(contrast["excludes_zero"] and contrast["mrr_difference"] > 0)


def reading_of(draw_results) -> dict:
    """The rule from the docstring, applied to the per-draw contrasts."""
    n = len(draw_results)
    free = [contrast_of(r["contrasts"], LABEL_FREE) for r in draw_results]
    specific = [contrast_of(r["contrasts"], LABEL_SPECIFIC) for r in draw_results]
    seen = [contrast_of(r["contrasts"], SEEN_MINUS_UNSEEN) for r in draw_results]
    n_free = sum(1 for c in free if positive_clear(c))
    n_not_positive = sum(1 for c in specific if not positive_clear(c))
    n_negative_clear = sum(1 for c in specific if c["excludes_zero"] and c["mrr_difference"] < 0)
    n_point_not_positive = sum(1 for c in specific if c["mrr_difference"] <= 0)
    n_seen_clear = sum(1 for c in seen if c["excludes_zero"])
    label_free = ("the label-free part transfers" if n_free == n
                  else f"unseen_total_whitening - unseen_none is positive with the interval clear of zero in {n_free} of {n} draws")
    label_specific = ("the label-specific part does not transfer" if n_not_positive >= RULE_MIN_NOT_POSITIVE
                      else f"unseen_within_author_whitening - unseen_total_whitening is not positive in {n_not_positive} of {n} draws")
    return {"label_free_part": label_free, "label_specific_part": label_specific,
            "draws_total_minus_none_positive_clear_of_zero": n_free,
            "draws_within_minus_total_not_positive": n_not_positive,
            "draws_within_minus_total_negative_clear_of_zero": n_negative_clear,
            "draws_within_minus_total_point_estimate_at_or_below_zero": n_point_not_positive,
            "draws_seen_minus_unseen_clear_of_zero": n_seen_clear,
            "draws": n, "rule_threshold_not_positive": RULE_MIN_NOT_POSITIVE}


def gap_checks(recomputed: dict, expected: dict, tolerance: dict, log=print) -> dict:
    """Compare recomputed numbers with the published ones; SystemExit on any gap above its
    tolerance. Pure, so the synthetic test can exercise it."""
    checks, failed = {}, []
    for name, value in expected.items():
        gap = abs(float(recomputed[name]) - value)
        checks[name] = {"expected": value, "recomputed": round(float(recomputed[name]), 4),
                        "gap": round(gap, 5), "tolerance": tolerance[name], "passed": bool(gap <= tolerance[name])}
        log(f"check {name}: recomputed {float(recomputed[name]):.4f} expected {value} "
            f"(gap {gap:.5f}, tolerance {tolerance[name]})")
        if gap > tolerance[name]:
            failed.append(name)
    if failed:
        raise SystemExit(f"the published numbers do not reproduce: {failed}; {checks}")
    return checks


# ------------------------------------------------------------------ analysis
def analyse(d: dict, seeds=REDRAW_SEEDS, seen_scores: np.ndarray | None = None,
            after_draw0=None, log=print) -> dict:
    """Every draw scored and summarised, draw 0 first; `after_draw0(entry, exact)` runs
    once draw 0 is scored and before any redraw is. d holds label_index, group_ids, weights,
    label_count, fold and dense (unit rows, float64) as exemplar_vs_prototype_v3.setup
    returns them."""
    li, gi, w, L, fold, dense = (d["label_index"], d["group_ids"], d["weights"], d["label_count"],
                                 d["fold"], d["dense"])
    if seen_scores is None:
        seen_scores = foldwise_within_author(dense, li, gi, w, L, fold)
    rr_seen = reciprocal_ranks(seen_scores, li)
    draws, draw_seeds, disjoint = held_out_draws(L, seeds)
    results = []
    for i, held in enumerate(draws):
        entry, exact = score_draw(d, held, rr_seen)
        entry = {"draw": i, "seed": int(draw_seeds[i]),
                 "source": "identity_probe_v2 (the published draw)" if i == 0 else "redraw", **entry}
        results.append(entry)
        systems = entry["systems"]
        log(f"draw {i} (seed {entry['seed']}): {entry['held_out_labels']} labels, {entry['queries']:,} queries, "
            f"{entry['groups']:,} groups; " + "  ".join(f"{n}={systems[n]['mrr']:.4f}" for n in SYSTEMS))
        for c in entry["contrasts"]:
            log(f"    {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
        if i == 0 and after_draw0 is not None:
            after_draw0(entry, exact)
    spreads = {"systems": {name: {stat: spread([r["systems"][name][stat] for r in results])
                                  for stat in ("mrr", "mrr_component_weighted", "mrr_label_macro")}
                           for name in SYSTEMS},
               "contrasts": {f"{a} - {b}": spread([contrast_of(r["contrasts"], (a, b))["mrr_difference"] for r in results])
                             for a, b in PAIRS}}
    return {"draws": results, "spread_across_draws": spreads, "reading": reading_of(results),
            "held_out_sets_pairwise_disjoint": bool(disjoint), "seeds": [int(s) for s in draw_seeds],
            "labels_never_held_out": int(L - int(np.any(np.stack(draws), axis=0).sum()))}


def reproduction_checks(d: dict, seen_scores: np.ndarray, draw0: dict, draw0_exact: dict, log=print) -> dict:
    """Checks 1 and 2 of the docstring on unrounded values; SystemExit on any failure."""
    li, gi, w, L, dense = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["dense"]
    checks = {}
    if PROBE_JSON.is_file():
        probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
        transfer = probe["unseen_author_transfer"]
        for name, key in PROBE_KEYS.items():
            if abs(transfer["systems"][key]["mrr"] - EXPECTED[name]) > 1e-9:
                raise SystemExit(f"the literal for {name} ({EXPECTED[name]}) is not identity_probe.json's "
                                 f"{transfer['systems'][key]['mrr']}")
        for name, key in PROBE_ALL_KEYS.items():
            if abs(probe["systems"][key]["mrr"] - EXPECTED[name]) > 1e-9:
                raise SystemExit(f"the literal for {name} ({EXPECTED[name]}) is not identity_probe.json's "
                                 f"{probe['systems'][key]['mrr']}")
        if transfer["held_out_labels"] != draw0["held_out_labels"] or transfer["queries"] != draw0["queries"]:
            raise SystemExit(f"draw 0 holds out {draw0['held_out_labels']} labels / {draw0['queries']} queries, "
                             f"the probe {transfer['held_out_labels']} / {transfer['queries']}")
        checks["literals_match_identity_probe_json"] = True
        log(f"literals match identity_probe.json; draw 0 holds out {draw0['held_out_labels']} labels, "
            f"{draw0['queries']:,} queries as the probe did")
    else:
        checks["literals_match_identity_probe_json"] = "identity_probe.json not found; literals used as pinned"
    everything = np.arange(len(li))
    recomputed = dict(draw0_exact)
    recomputed["all_queries_none"] = float(np.mean(reciprocal_ranks(
        dense_leave_group_out(dense, li, gi, w, L, everything), li)))
    recomputed["all_queries_within_author_whitening"] = float(np.mean(reciprocal_ranks(seen_scores, li)))
    checks.update(gap_checks(recomputed, EXPECTED, TOLERANCE, log=log))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()
    from build_downstream_retrieval_v2 import corpus_version
    if corpus_version() != "v3":
        raise SystemExit("this file's checks are corpus v3 numbers; set CHINESE_RAP_CORPUS=v3")
    from corpus_v3 import V3_CONTENT_SHA256
    from exemplar_vs_prototype_v3 import setup
    print("loading corpus v3 and building the protocol's queries, groups, weights and folds", flush=True)
    d = setup(args.private_root.resolve())
    li, gi, w, L, fold, dense = (d["label_index"], d["group_ids"], d["weights"], d["label_count"],
                                 d["fold"], d["dense"])
    n = len(li)
    print(f"  {n:,} queries, {L} labels, {len(np.unique(gi)):,} groups, fold sizes "
          f"{np.bincount(fold, minlength=FOLDS).tolist()}", flush=True)

    print("fold-wise within-author whitening (every label seen)", flush=True)
    seen_scores = foldwise_within_author(dense, li, gi, w, L, fold)

    # draw 0 is scored first and checked against the published numbers before any redraw
    checks = {}

    def after_draw0(entry, exact):
        print("draw 0 is the probe's held-out set; checking it against identity_probe.json", flush=True)
        checks.update(reproduction_checks(d, seen_scores, entry, exact))
        print("checks passed; redrawing the held-out set", flush=True)

    out = analyse(d, seeds=REDRAW_SEEDS, seen_scores=seen_scores, after_draw0=after_draw0)
    if not checks:
        raise SystemExit("the reproduction check did not run")
    if not out["held_out_sets_pairwise_disjoint"]:
        raise SystemExit("the six held-out sets are not pairwise disjoint; the reading requires disjoint blocks")
    if len(out["draws"]) != 1 + len(REDRAW_SEEDS):
        raise SystemExit("not every draw was scored")
    for key, value in out["reading"].items():
        print(f"reading: {key}: {value}", flush=True)
    for name, stats in out["spread_across_draws"]["systems"].items():
        s = stats["mrr"]
        print(f"spread {name}: mrr min {s['min']:.4f} max {s['max']:.4f} mean {s['mean']:.4f} sd {s['sd']:.4f}", flush=True)
    for name, s in out["spread_across_draws"]["contrasts"].items():
        print(f"spread {name}: min {s['min']:+.4f} max {s['max']:+.4f} mean {s['mean']:+.4f} sd {s['sd']:.4f}", flush=True)

    payload = {
        "analysis": "whitening transfer to unseen labels under six disjoint held-out draws (the probe's draw and five redraws)",
        "question": "does the label-free / label-specific transfer reading of identity_probe_v2 depend on its single held-out draw?",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L,
                   "groups": int(len(np.unique(gi))), "held_out_labels_per_draw": int(round(HELD_OUT_LABEL_SHARE * L))},
        "design": {
            "draw_0": f"identity_probe_v2's held-out set: rng({SEED}).permutation(labels)[:round({HELD_OUT_LABEL_SHARE} x labels)]",
            "redraws": "draw i permutes, with seeds[i], the labels no earlier draw held out and takes the first block; the six sets are pairwise disjoint",
            "seeds": out["seeds"],
            "held_out_sets_pairwise_disjoint": out["held_out_sets_pairwise_disjoint"],
            "labels_never_held_out": out["labels_never_held_out"],
            "transforms": "identity_probe_v2.fit_transform none / total_whitening / within_author_whitening, fitted on the seen labels' songs only, applied to every song",
            "scoring": "identity_probe_v2.dense_leave_group_out: the held-out labels' queries against leave-group-out profiles of all labels",
            "seen_label_comparison": "the probe's fold-wise within-author space (fitted on the other four folds, every label seen), the same queries",
            "estimands": {"mrr": "plain mean of reciprocal ranks over the draw's queries (identity_probe.json's mrr)",
                          "mrr_component_weighted": "per-(group, label) component-weighted MRR, the bootstrap's estimand",
                          "mrr_label_macro": "mean over the held-out labels of the label's component-weighted MRR (description only)",
                          "recall_at_10": "share of queries with rank <= 10"},
            "contrasts": [f"{a} - {b}" for a, b in PAIRS],
            "bootstrap": "identity_spaces_v2.paired_group_bootstrap: the draw's leakage groups resampled with replacement, 2000 replicates, seed 20260825",
            "spread": "min, max, mean and sample sd (ddof 1) over the six draws",
            "reading_rule": {
                "label_free_part": "'the label-free part transfers' if unseen_total_whitening - unseen_none is positive with the 95% interval clear of zero in every draw; otherwise the count is stated",
                "label_specific_part": f"'the label-specific part does not transfer' only if unseen_within_author_whitening - unseen_total_whitening is not positive (positive = point estimate above zero with the 95% interval clear of zero) in at least {RULE_MIN_NOT_POSITIVE} of the six draws; otherwise the count is stated",
                "description_only": "the counts of draws with that contrast negative and clear of zero or with a point estimate at or below zero, the seen - unseen contrast, the label-macro MRR and the spreads"},
            "checks": {"draw_0": f"reproduces identity_probe.json on unrounded values: unseen none (gap <= {CHECK_GAP_UNFITTED}), total, within-author and seen-label within-author (gap <= {CHECK_GAP_FITTED}); literals, 34 labels and 1,041 queries matched against the file",
                       "all_queries": f"untransformed space {EXPECTED['all_queries_none']} (gap <= {CHECK_GAP_UNFITTED}), fold-wise within-author {EXPECTED['all_queries_within_author_whitening']} (gap <= {CHECK_GAP_FITTED})",
                       "draws": "each of the required size, pairwise disjoint, training mask free of held-out labels' songs",
                       "order": "draw 0 is scored and checked before any redraw is scored"},
        },
        "checks_against_published": checks,
        "draws": out["draws"],
        "spread_across_draws": out["spread_across_draws"],
        "reading": out["reading"],
        "privacy": "aggregate only",
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / OUT_NAME
    target.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nwrote {target}  ({payload['minutes']} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
