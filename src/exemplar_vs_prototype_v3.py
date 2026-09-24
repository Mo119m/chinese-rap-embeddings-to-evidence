#!/usr/bin/env python3
"""Is a label a centre or a cloud? Exemplar scoring against the protocol's prototype.

Every published number scores a held-out song against one profile per label: the weighted
mean of the label's other songs (a prototype). Exemplar theory (Nosofsky 1986; Pierrehumbert
2001) keeps every instance instead: a query belongs with the label holding the song nearest
to it. If a label is several styles rather than one voice, the prototype blurs them and the
nearest exemplar should identify better; if a label is one centre with topic noise around it,
the prototype should win, and more so after within-author whitening removes that noise.

Scorers, all leave-group-out (the query's whole leakage group is removed from every label):
  prototype       cosine with the weighted-mean profile (the protocol; checked against the
                  published numbers before anything else is scored)
  exemplar_max    the highest cosine with any remaining song of the label
  exemplar_top3   the mean of the three highest component scores of the label, a component
                  being a (group, label) set of near-duplicate songs scored by its best member
  exemplar_mean   the protocol-weighted mean cosine over the label's remaining songs (the
                  prototype without its profile-norm factor)
  exemplar_max_4  exemplar_max over at most four components per label, chosen by a fixed
                  seeded order: removes the extra chances a large label gets under a maximum

Spaces: raw BGE-M3 song centroids; the same whitened within-author per fold; raw jieba word
TF-IDF; the word SVD-1024 whitened within-author per fold (fitted on the training folds).

Reading, fixed before the run. In a space, exemplar scoring changes the answer if
exemplar_max - prototype has a paired group-bootstrap interval clear of zero AND
exemplar_max_4 - prototype does too, in the same direction. Otherwise the prototype stands.
Label-size bands (components per label) are reported as description.

    python src/exemplar_vs_prototype_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, SVD_COMPONENTS, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from word_identity_anatomy_v2 import EXPECTED_WORD_MRR, fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
SCORERS = ("prototype", "exemplar_max", "exemplar_top3", "exemplar_mean", "exemplar_max_4")
TOP_K = 3
MAX_COMPONENTS = 4
BANDS = ((4, 9), (10, 19), (20, 49), (50, 10**9))
CHECK_GAP = 0.002
EXPECTED = {"semantic_raw": 0.2997, "semantic_whitened": 0.4164, "words_raw": EXPECTED_WORD_MRR,
            "words_svd_whitened": 0.5426}


def setup(private_root: Path):
    rows, vectors, _ = load_v3(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, centroids_by_song = build_songs(rows, vectors)
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {s for s in chunks_by_song
                   if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(l for l, m in songs_by_label.items()
                      if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
    label_index = np.asarray([eligible.index(label_by_song[s]) for s in songs], dtype=np.int64)
    groups = build_groups(songs, components_by_song, {s: normalise_document(documents[s]) for s in songs})
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    label_count = len(eligible)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs])).astype(np.float64)
    return dict(songs=songs, label_index=label_index, group_ids=group_ids, label_count=label_count,
                weights=weights, fold=fold, dense=dense, documents=documents, labels=eligible)


class ExemplarScorer:
    """Leave-group-out exemplar scores from a full cosine matrix, one label at a time."""

    def __init__(self, label_index, group_ids, weights, label_count, seed):
        self.li, self.gi, self.w, self.L = label_index, group_ids, weights, label_count
        # components: (group, label) pairs; columns of every label sorted by component
        comp_key = {}
        self.comp_of = np.zeros(len(label_index), dtype=np.int64)
        for i, (g, l) in enumerate(zip(group_ids.tolist(), label_index.tolist())):
            self.comp_of[i] = comp_key.setdefault((g, l), len(comp_key))
        self.comp_label = np.zeros(len(comp_key), dtype=np.int64)
        self.comp_group = np.zeros(len(comp_key), dtype=np.int64)
        for (g, l), c in comp_key.items():
            self.comp_label[c], self.comp_group[c] = l, g
        self.rng = np.random.default_rng(seed)
        self.comp_order = {l: self.rng.permutation(np.flatnonzero(self.comp_label == l))
                           for l in range(label_count)}
        self.components_per_label = np.bincount(self.comp_label, minlength=label_count)
        # songs sorted by component, so a component's best member is a segment maximum
        self.song_order = np.argsort(self.comp_of, kind="stable")
        sorted_comp = self.comp_of[self.song_order]
        self.comp_starts = np.flatnonzero(np.r_[True, sorted_comp[1:] != sorted_comp[:-1]])
        if not np.array_equal(sorted_comp[self.comp_starts], np.arange(len(comp_key))):
            raise RuntimeError("component segments are not in component order")
        self.member = np.zeros((len(label_index), label_count))
        self.member[np.arange(len(label_index)), label_index] = weights

    def score(self, cosine: np.ndarray, queries: np.ndarray) -> dict[str, np.ndarray]:
        """cosine: (n_queries, n_songs) cosines of the queries with every song."""
        n = len(queries)
        q_group = self.gi[queries]
        excluded = q_group[:, None] == self.gi[None, :]                      # same leakage group
        masked = np.where(excluded, -np.inf, cosine)
        # component-level best member
        comp_best = np.maximum.reduceat(masked[:, self.song_order], self.comp_starts, axis=1)
        out = {name: np.full((n, self.L), -np.inf) for name in ("exemplar_max", "exemplar_top3", "exemplar_max_4")}
        for l in range(self.L):
            cols = self.comp_order[l]                                        # fixed seeded order
            block = comp_best[:, cols]
            out["exemplar_max"][:, l] = block.max(axis=1)
            k = min(TOP_K, block.shape[1])
            top = -np.partition(-block, k - 1, axis=1)[:, :k]
            top = np.where(np.isfinite(top), top, np.nan)
            out["exemplar_top3"][:, l] = np.nanmean(top, axis=1)
            valid = np.isfinite(block)
            keep = valid & (np.cumsum(valid, axis=1) <= MAX_COMPONENTS)
            out["exemplar_max_4"][:, l] = np.where(keep, block, -np.inf).max(axis=1)
        # protocol-weighted mean cosine over the remaining songs
        kept = np.where(excluded, 0.0, 1.0)
        numer = (np.where(excluded, 0.0, cosine)) @ self.member
        denom = kept @ self.member
        if np.any(denom <= 0):
            raise RuntimeError("a held-out group empties a label")
        out["exemplar_mean"] = numer / denom
        if not np.all(np.isfinite(out["exemplar_max"])):
            raise RuntimeError("a label lost every exemplar for some query")
        return out


def brute_force_check(scorer: ExemplarScorer, cosine, queries, out):
    """Slow definition of every exemplar scorer on a few queries; must agree with the vectorised one."""
    for row, q in enumerate(queries.tolist()[:6]):
        for l in range(scorer.L):
            songs = [s for s in np.flatnonzero(scorer.li == l) if scorer.gi[s] != scorer.gi[q]]
            comps = defaultdict(list)
            for s in songs:
                comps[int(scorer.comp_of[s])].append(float(cosine[row, s]))
            best = {c: max(v) for c, v in comps.items()}
            ordered = [best[c] for c in scorer.comp_order[l].tolist() if c in best]
            expect = {"exemplar_max": max(best.values()),
                      "exemplar_top3": float(np.mean(sorted(best.values(), reverse=True)[:TOP_K])),
                      "exemplar_max_4": max(ordered[:MAX_COMPONENTS]),
                      "exemplar_mean": float(np.sum([cosine[row, s] * scorer.w[s] for s in songs]) / np.sum([scorer.w[s] for s in songs]))}
            for name, value in expect.items():
                if abs(out[name][row, l] - value) > 1e-9:
                    raise SystemExit(f"{name} differs from its definition at query {q}, label {l}: {out[name][row, l]} vs {value}")


BLOCK = 1000


def score_space(name, vectors, d, scorer, queries_by_fold, sparse_matrix=None):
    """Prototype scores by the protocol, exemplar scores from cosines; per fold when given,
    queries in blocks to bound memory."""
    li, gi, w, L = d["label_index"], d["group_ids"], d["weights"], d["label_count"]
    n = len(li)
    scores = {s: np.zeros((n, L)) for s in SCORERS}
    checked = False
    for k, (queries, x) in queries_by_fold.items():
        if sparse_matrix is None:
            scores["prototype"][queries] = dense_leave_group_out(x, li, gi, w, L, queries)
        for start in range(0, len(queries), BLOCK):
            block = queries[start:start + BLOCK]
            if sparse_matrix is not None:
                cosine = np.asarray((sparse_matrix[block] @ sparse_matrix.T).todense(), dtype=np.float64)
            else:
                cosine = x[block] @ x.T
            ex = scorer.score(cosine, block)
            if not checked:
                brute_force_check(scorer, cosine, block, ex)
                checked = True
            for s, arr in ex.items():
                scores[s][block] = arr
    return scores


def words_prototype(lexical, d):
    """The protocol's sparse scorer for the word space."""
    profiles = v1.score_leave_group_out(d["dense"].astype(np.float32), lexical, d["label_index"], d["group_ids"], d["label_count"])
    return profiles.lexical.astype(np.float64)


def mrr(scores, d, mask=None):
    li, w = d["label_index"], d["weights"]
    ranks = ranks_of(scores, li)
    rr = 1.0 / ranks
    if mask is None:
        return float(np.mean(rr)), rr
    return float(np.sum(rr[mask] * w[mask]) / np.sum(w[mask])), rr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()
    import jieba
    jieba.setLogLevel(60)
    d = setup(args.private_root.resolve())
    li, gi, w, L, fold = d["label_index"], d["group_ids"], d["weights"], d["label_count"], d["fold"]
    n = len(li)
    everything = np.arange(n)
    scorer = ExemplarScorer(li, gi, w, L, SEED)
    print(f"{n:,} queries, {L} labels, {len(scorer.comp_label):,} components", flush=True)

    spaces = {}
    checks = {}
    # raw semantic: no fitting, all queries at once
    print("semantic raw", flush=True)
    spaces["semantic_raw"] = score_space("semantic_raw", d["dense"], d, scorer, {0: (everything, d["dense"])})

    # whitened semantic: per fold, fitted on training folds
    print("semantic whitened", flush=True)
    per_fold = {}
    for k in range(FOLDS):
        train = fold != k
        mean, matrix, _ = fit_transform("within_author_whitening", d["dense"][train], li[train], w[train],
                                        np.random.default_rng(SEED + k))
        per_fold[k] = (np.flatnonzero(fold == k), unit_rows((d["dense"] - mean) @ matrix.T))
    spaces["semantic_whitened"] = score_space("semantic_whitened", None, d, scorer, per_fold)

    # raw words: sparse TF-IDF, prototype from the protocol's own scorer
    print("words raw", flush=True)
    lexical, _ = fit_words([" ".join(segment(d["documents"][s])) for s in d["songs"]])
    spaces["words_raw"] = score_space("words_raw", None, d, scorer, {0: (everything, None)}, sparse_matrix=lexical)
    spaces["words_raw"]["prototype"] = words_prototype(lexical, d)

    # whitened word SVD: per fold
    print("words svd whitened", flush=True)
    per_fold = {}
    for k in range(FOLDS):
        train = fold != k
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED + k).fit(lexical[train])
        reduced = unit_rows(svd.transform(lexical))
        mean, matrix, _ = fit_transform("within_author_whitening", reduced[train], li[train], w[train],
                                        np.random.default_rng(SEED + k))
        per_fold[k] = (np.flatnonzero(fold == k), unit_rows((reduced - mean) @ matrix.T))
        print(f"  fold {k}", flush=True)
    spaces["words_svd_whitened"] = score_space("words_svd_whitened", None, d, scorer, per_fold)

    # checks against the published prototype numbers
    for name, sc in spaces.items():
        value, _ = mrr(sc["prototype"], d)
        checks[name] = {"expected": EXPECTED[name], "recomputed": round(value, 4), "gap": round(abs(value - EXPECTED[name]), 4)}
        print(f"check {name}: prototype {value:.4f} expected {EXPECTED[name]}", flush=True)
    if any(c["gap"] > CHECK_GAP for c in checks.values()):
        raise SystemExit(f"the prototype does not reproduce the published numbers: {checks}")

    all_mask = np.ones(n, dtype=bool)
    comp_per_label_of_song = scorer.components_per_label[li]
    results = {}
    for name, sc in spaces.items():
        rr = {}
        systems = {}
        for s in SCORERS:
            systems[s], rr[s] = mrr(sc[s], d)
            systems[s] = round(systems[s], 4)
        pairs = [(s, "prototype") for s in SCORERS if s != "prototype"] + [("exemplar_max", "exemplar_max_4")]
        contrasts = paired_group_bootstrap(rr, w, gi, all_mask, pairs)
        bands = {}
        for lo, hi in BANDS:
            m = (comp_per_label_of_song >= lo) & (comp_per_label_of_song <= hi)
            bands[f"{lo}-{hi if hi < 10**9 else 'up'}"] = {"queries": int(m.sum()),
                                                            **{s: round(mrr(sc[s], d, m)[0], 4) for s in SCORERS}}
        c_max = next(c for c in contrasts if (c["system"], c["minus"]) == ("exemplar_max", "prototype"))
        c_four = next(c for c in contrasts if (c["system"], c["minus"]) == ("exemplar_max_4", "prototype"))
        same_sign = np.sign(c_max["mrr_difference"]) == np.sign(c_four["mrr_difference"])
        if c_max["excludes_zero"] and c_four["excludes_zero"] and same_sign:
            reading = ("exemplar scoring identifies better than the prototype" if c_max["mrr_difference"] > 0
                       else "the prototype identifies better than exemplar scoring")
        else:
            reading = "the prototype stands (exemplar_max and exemplar_max_4 do not both clear zero in one direction)"
        results[name] = {"systems": systems, "paired_contrasts": contrasts, "by_components_per_label": bands, "reading": reading}
        print(f"== {name}: " + "  ".join(f"{s}={v:.4f}" for s, v in systems.items()), flush=True)
        for c in contrasts:
            print(f"    {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)
        print(f"    {reading}", flush=True)

    payload = {
        "analysis": "exemplar (nearest-instance) scoring against the protocol's prototype profile, four spaces",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L, "components": int(len(scorer.comp_label))},
        "design": {"scorers": {"prototype": "cosine with the weighted-mean leave-group-out profile (the protocol)",
                               "exemplar_max": "highest cosine with any remaining song of the label",
                               "exemplar_top3": f"mean of the {TOP_K} highest component scores (component = (group, label), scored by its best member)",
                               "exemplar_mean": "protocol-weighted mean cosine over the remaining songs",
                               "exemplar_max_4": f"exemplar_max over at most {MAX_COMPONENTS} components per label in a fixed seeded order"},
                   "spaces": {"semantic_raw": "BGE-M3 song centroids", "semantic_whitened": "the same, within-author whitening fitted per fold",
                              "words_raw": "jieba word TF-IDF (1-2-grams)", "words_svd_whitened": "word SVD-1024 fitted on the training folds, within-author whitening per fold"},
                   "leave_group_out": "the query's whole leakage group is removed from every label before any scorer",
                   "reading_rule": "exemplar scoring changes the answer in a space only if exemplar_max - prototype and exemplar_max_4 - prototype both have intervals clear of zero in the same direction",
                   "checks": "prototype MRR must reproduce the published number in every space (gap <= 0.002); the vectorised exemplar scorers must agree with their brute-force definition on six queries in every space",
                   "bands": "components per label of the query's own label"},
        "checks_against_published": checks,
        "spaces": results,
        "privacy": "aggregate only",
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "exemplar_vs_prototype.json").write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nwrote {args.out_dir / 'exemplar_vs_prototype.json'}  ({payload['minutes']} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
