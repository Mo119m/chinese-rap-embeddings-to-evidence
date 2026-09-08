#!/usr/bin/env python3
"""Is identity absent from the semantic embedding, or only hidden in it?

The character n-gram system identifies a song's author far better than BGE-M3 does, 0.450
to 0.318 MRR under one protocol. Two readings are possible. The dense model may have
discarded the information a semantic objective has no use for. Or the information may be
there, sitting in directions of the embedding that cosine similarity cannot pick out
because they are swamped by what varies within an author -- what each song is about.

Speaker recognition faced this exact problem and solved it with a linear step: estimate
the covariance of the embedding *within* a speaker, then whiten by it, so directions that
vary from one utterance to the next of the same person are shrunk and directions that stay
put are kept (within-class covariance normalisation, Hatch, Kajarekar and Stolcke 2006;
the same move sits inside every LDA/PLDA back-end since Dehak et al. 2011). Applied here:
if whitening by within-author covariance lifts the dense system, the identity information
was present and misaligned; if it does not, the objective erased it, and only retraining
could put it back.

The estimate must not see the songs it is scored on. Leakage groups are dealt into five
folds; the whitening for a fold is fitted on the other four and applied to every song, and
only that fold's queries are scored in that space -- against label profiles built from all
other songs, exactly as in the original protocol. Two controls fix what the gain means:
whitening by the *total* covariance (no labels used) separates identity-specific from
generic conditioning, and whitening fitted on permuted labels (author structure destroyed,
class sizes kept) is the null.

    python src/identity_probe_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.covariance import ledoit_wolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import (  # noqa: E402
    MINIMUM_SONGS_PER_LABEL,
    build_songs,
    load,
)
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
SEED = 20260825
FOLDS = 5
EXPECTED_DENSE_MRR = 0.3181


# ------------------------------------------------------------------ scoring
def dense_leave_group_out(vectors: np.ndarray, label_index: np.ndarray, group_ids: np.ndarray,
                          weights: np.ndarray, label_count: int, queries: np.ndarray) -> np.ndarray:
    """The builder's dense branch, for a subset of queries, in float64.

    Profiles are weighted label sums; for a query, every label present in its group has the
    group's members removed before normalising. Verified against v1.score_leave_group_out
    on the untransformed vectors before any transformed space is scored.
    """
    sums = np.zeros((label_count, vectors.shape[1]))
    np.add.at(sums, label_index, vectors * weights[:, None])
    norms = np.linalg.norm(sums, axis=1)
    if np.any(norms <= 0):
        raise RuntimeError("an empty label profile")
    scores = vectors[queries] @ (sums / norms[:, None]).T
    members_by_group: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids.tolist()):
        members_by_group[int(group)].append(index)
    for row, query in enumerate(queries.tolist()):
        members = members_by_group[int(group_ids[query])]
        for label in {int(label_index[m]) for m in members}:
            own = [m for m in members if int(label_index[m]) == label]
            leave = sums[label] - np.sum(vectors[own] * weights[own, None], axis=0)
            norm = float(np.linalg.norm(leave))
            if norm <= 1e-12:
                raise RuntimeError("a held-out group empties a label profile")
            scores[row, label] = float(vectors[query] @ leave) / norm
    return scores


# ------------------------------------------------------------------ transforms
def shrunk_inverse_sqrt(residuals: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """ZCA whitening matrix of a weighted covariance with Ledoit-Wolf shrinkage.

    Residuals are already centred on whatever mean the caller chose. Weights are the
    builder's per-(group, label) weights, so a song filed five times counts once; they
    enter as row scalings so that the shrinkage estimate sees the same weighted moments.
    """
    scaled = residuals * np.sqrt(weights * len(weights) / weights.sum())[:, None]
    covariance, shrinkage = ledoit_wolf(scaled, assume_centered=True)
    values, vectors = np.linalg.eigh(covariance)
    if values.min() <= 0:
        raise RuntimeError("the shrunk covariance is not positive definite")
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T, float(shrinkage)


def fit_transform(name: str, train: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                  rng: np.random.Generator):
    """Return (mean, whitening matrix, diagnostics) fitted on the training rows."""
    mean = np.average(train, axis=0, weights=weights)
    if name == "none":
        # no centring either: subtracting a mean before re-normalising changes every cosine,
        # and the first run of this file learned that the hard way (0.3136 against the
        # builder's 0.3181). Centring is its own control below.
        return np.zeros(train.shape[1]), np.eye(train.shape[1]), {}
    if name == "centred":
        return mean, np.eye(train.shape[1]), {}
    if name == "total_whitening":
        matrix, shrinkage = shrunk_inverse_sqrt(train - mean, weights)
        return mean, matrix, {"shrinkage": round(shrinkage, 4)}
    if name in ("within_author_whitening", "within_author_whitening_permuted_labels"):
        if name.endswith("permuted_labels"):
            labels = rng.permutation(labels)
        label_count = int(labels.max()) + 1
        sums = np.zeros((label_count, train.shape[1]))
        np.add.at(sums, labels, train * weights[:, None])
        mass = np.bincount(labels, weights=weights, minlength=label_count)
        present = mass > 0
        centroids = np.zeros_like(sums)
        centroids[present] = sums[present] / mass[present, None]
        residuals = train - centroids[labels]
        matrix, shrinkage = shrunk_inverse_sqrt(residuals, weights)
        total = np.average(np.sum((train - mean) ** 2, axis=1), weights=weights)
        within = np.average(np.sum(residuals ** 2, axis=1), weights=weights)
        return mean, matrix, {"shrinkage": round(shrinkage, 4),
                              "within_author_share_of_variance": round(within / total, 4)}
    raise ValueError(name)


TRANSFORMS = ("none", "centred", "total_whitening", "within_author_whitening",
              "within_author_whitening_permuted_labels")


# ------------------------------------------------------------------ build
def build(private_root: Path, out_dir: Path) -> int:
    print("loading", flush=True)
    rows, vectors, _ = load(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, centroids_by_song = build_songs(
        rows, vectors)
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {s for s in chunks_by_song
                   if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(l for l, m in songs_by_label.items()
                      if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
    label_index = np.asarray([eligible.index(label_by_song[s]) for s in songs], dtype=np.int64)
    groups = build_groups(songs, components_by_song,
                          {s: normalise_document(documents[s]) for s in songs})
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    label_count = len(eligible)
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs])).astype(np.float64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))]
                          for g, l in zip(group_ids, label_index)])
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    # the scorer must be the builder's before any transformed space is trusted
    print("verifying the dense scorer against the builder", flush=True)
    lexical = v1.fit_tfidf([documents[s] for s in songs])
    reference = v1.score_leave_group_out(dense.astype(np.float32), lexical, label_index,
                                         group_ids, label_count)
    everything = np.arange(len(songs))
    check = dense_leave_group_out(dense, label_index, group_ids, weights, label_count, everything)
    gap = float(np.abs(check - reference.dense.astype(np.float64)).max())
    print(f"  max |recomputed - builder| = {gap:.3e}", flush=True)
    if gap > 1e-4:
        raise SystemExit("the dense scorer does not reproduce the builder")

    # groups dealt into folds; a fold's whitening never sees that fold's songs
    rng = np.random.default_rng(SEED)
    fold_of_group = rng.integers(0, FOLDS, size=len(order))
    fold = fold_of_group[group_ids]
    fold_sizes = np.bincount(fold, minlength=FOLDS).tolist()
    print(f"  fold sizes {fold_sizes}", flush=True)

    scores = {name: np.zeros((len(songs), label_count)) for name in TRANSFORMS}
    diagnostics = {name: [] for name in TRANSFORMS}
    for k in range(FOLDS):
        train_mask = fold != k
        queries = np.flatnonzero(~train_mask)
        train_labels = label_index[train_mask]
        print(f"fold {k}: fitting on {int(train_mask.sum()):,} songs, "
              f"scoring {len(queries):,}", flush=True)
        for name in TRANSFORMS:
            mean, matrix, info = fit_transform(name, dense[train_mask], train_labels,
                                               weights[train_mask], np.random.default_rng(SEED + k))
            projected = (dense - mean) @ matrix.T
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            projected = projected / np.maximum(norms, 1e-12)
            scores[name][queries] = dense_leave_group_out(
                projected, label_index, group_ids, weights, label_count, queries)
            if info:
                diagnostics[name].append({"fold": k, "labels_seen": int(len(set(train_labels.tolist()))),
                                          **info})

    ranks = {name: v1.rank_system(s.astype(np.float32), label_index)[0].astype(np.int64)
             for name, s in scores.items()}
    none_mrr = float(np.mean(1.0 / ranks["none"]))
    if abs(none_mrr - EXPECTED_DENSE_MRR) > 5e-4:
        raise SystemExit(f"the untransformed space gives {none_mrr:.4f}, not {EXPECTED_DENSE_MRR}")

    # fusion with the lexical system, to see whether the recovered signal is new
    lexical_ranks = v1.rank_system(reference.lexical, label_index)[0].astype(np.int64)
    fused_scores = {}
    for name in ("none", "within_author_whitening"):
        fused = (v1.zscore_rows(scores[name]) + v1.zscore_rows(reference.lexical.astype(np.float64))) / 2.0
        fused_scores[f"fusion_lexical_{name}"] = fused
        ranks[f"fusion_lexical_{name}"] = v1.rank_system(fused.astype(np.float32), label_index)[0].astype(np.int64)
    ranks["lexical_character_ngrams"] = lexical_ranks

    rr = {name: 1.0 / r for name, r in ranks.items()}
    all_mask = np.ones(len(songs), dtype=bool)
    report = {}
    for name, r in ranks.items():
        report[name] = {"mrr": round(float(np.mean(1.0 / r)), 4),
                        "mrr_component_weighted": round(weighted_mean(rr[name], weights, all_mask), 4),
                        "recall_at_1": round(float(np.mean(r <= 1)), 4),
                        "recall_at_10": round(float(np.mean(r <= 10)), 4)}
        print(f"  {name:42s} MRR {report[name]['mrr']:.4f}  R@1 {report[name]['recall_at_1']:.4f}  "
              f"R@10 {report[name]['recall_at_10']:.4f}", flush=True)

    print("paired bootstrap over leakage groups", flush=True)
    pairs = [
        ("within_author_whitening", "none"),
        ("within_author_whitening", "centred"),
        ("within_author_whitening", "total_whitening"),
        ("within_author_whitening", "within_author_whitening_permuted_labels"),
        ("centred", "none"),
        ("total_whitening", "centred"),
        ("lexical_character_ngrams", "within_author_whitening"),
        ("fusion_lexical_within_author_whitening", "fusion_lexical_none"),
        ("fusion_lexical_within_author_whitening", "lexical_character_ngrams"),
    ]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, all_mask, pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    head = {}
    for left, right in (("within_author_whitening", "none"),
                        ("within_author_whitening", "lexical_character_ngrams")):
        a, b = ranks[left], ranks[right]
        head[f"{left}_vs_{right}"] = {"first_better": int((a < b).sum()),
                                      "second_better": int((b < a).sum()),
                                      "tied": int((a == b).sum())}

    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir = private_root / "work" / "private-identity-probe-v2"
    private_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    fields = ["song_id", "label", "group", "fold"] + [f"rank_{n}" for n in ranks]
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for q, song in enumerate(songs):
        writer.writerow({"song_id": song, "label": eligible[label_index[q]],
                         "group": int(group_ids[q]), "fold": int(fold[q]),
                         **{f"rank_{n}": int(r[q]) for n, r in ranks.items()}})
    (private_dir / "per_query.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")

    payload = {
        "analysis": "linear identity probe on the frozen semantic embedding",
        "question": ("whether the identity information the character n-gram system uses is "
                     "absent from BGE-M3's song vectors or present but misaligned with cosine"),
        "method": ("within-class covariance normalisation from speaker recognition: whiten the "
                   "song vectors by the covariance of songs within an author, fitted with "
                   "Ledoit-Wolf shrinkage on the builder's per-(group, label) weights, then "
                   "score with the unchanged leave-group-out cosine protocol"),
        "design": {
            "folds": FOLDS, "seed": SEED, "fold_sizes": fold_sizes,
            "unit_dealt_into_folds": "leakage group",
            "rule": ("a fold's transform is fitted on the other folds and applied to every "
                     "song; only that fold's queries are scored in it, against profiles built "
                     "from all other songs as in the original protocol"),
            "controls": {
                "centred": "mean subtraction and re-normalisation only, which every whitening "
                           "also does, so its effect can be separated from the whitening's",
                "total_whitening": "the same whitening from the total covariance, no labels used",
                "within_author_whitening_permuted_labels": "labels permuted across the training "
                                                           "songs before fitting; class sizes kept",
            },
        },
        "queries": len(songs), "labels": label_count, "groups": len(order),
        "verification": {
            "dense_scorer_reproduces_builder_max_gap": gap,
            "untransformed_space_reproduces_builder_mrr": round(none_mrr, 4),
        },
        "systems": report,
        "head_to_head": head,
        "paired_contrasts": {
            "design": ("2000 replicates, seed 20260825, leakage groups resampled with "
                       "replacement, each (group, label) component weighted one"),
            "contrasts": contrasts,
        },
        "fold_diagnostics": diagnostics,
        "privacy": "aggregate only; per-query ranks are private",
    }
    (out_dir / "identity_probe.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'identity_probe.json'}")
    print(f"wrote private per-query table to {private_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
