#!/usr/bin/env python3
"""What did the 64-anchor fine-tune learn beyond whitening? Part A: descriptive, CPU.

Every fold of the GradCache fine-tune (tag _gradcache64) beats the frozen encoder after
within-author whitening, on its test fold and on the 34 labels never trained on. This file
describes the extra identity; the causal question, whether the tuned model's advantage
depends on particular kinds of words, is finetune_cue_masking_v3.py.

A first design removed from each space its ridge prediction from the word space and compared
what was left. It was dropped after a one-fold run, before any result was used. A regression
from words to vectors fitted on songs of the same labels learns the association "this
label's words, this label's direction", so subtracting it strips seen-label identity whether
or not the two spaces share information (fold 0: frozen whitened 0.401 -> 0.143 on the test
fold, 0.336 -> 0.337 on unseen labels, at an out-of-sample R² of 0.01). What remains here
uses no regression from words.

For each fold k, on the fold's own split (whitening and the one regression fitted on the
training songs: other folds, labels not held out):

  Fw, Tw      frozen and tuned song vectors after within-author whitening, as in
              analyse_identity_encoder_v3 (checked: their MRRs must reproduce that file's)
  N           the new part of the tuned space: Tw minus its ridge prediction from Fw. The
              regression is from one encoder to the other; seen-label identity it carries over
              is removed from N, so N's identity on the test fold is a conservative figure
  null        random Gaussian song vectors under the same protocol, three seeds
  CKA         linear centred kernel alignment (Kornblith et al. 2019) of each space with the
              raw word TF-IDF space, on the scope's songs, label-free
  correlates  the jieba words (in at least 50 songs) whose TF-IDF weight correlates most with
              the leading principal directions of N and of Fw, and their part-of-speech
              categories

Reading. N identifies labels above the null: fine-tuning added identity that no linear map of
the frozen whitened space provides. CKA(words, Tw) above CKA(words, Fw) on the scope's songs:
the tuned geometry moved toward the word space. Both are descriptive; neither says the extra
identity is word usage.

    python src/finetune_learned_identity_v3.py --private-root <ni-k> [--folds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of, song_level  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from word_identity_anatomy_v2 import fit_words, pos_group, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
TAG = "_gradcache64"
HELD_OUT_SHARE = 0.15
ALPHAS = np.logspace(-1, 3, 9)
TOP_DIRECTIONS = 6
WORDS_PER_END = 15
CATEGORY_WORDS = 50
MIN_DF = 50
NULL_SEEDS = (1, 2, 3)
CHECK_GAP = 0.002


def ridge_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray):
    from sklearn.linear_model import RidgeCV
    model = RidgeCV(alphas=ALPHAS).fit(x[train], y[train])
    return model.predict(x), float(model.alpha_)


def r_squared(y: np.ndarray, prediction: np.ndarray, rows: np.ndarray, train: np.ndarray) -> float:
    centre = y[train].mean(axis=0)
    return round(1.0 - float(np.sum((y[rows] - prediction[rows]) ** 2)) / float(np.sum((y[rows] - centre) ** 2)), 4)


def linear_cka(gram_x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA from a precomputed Gram matrix of X and the rows of Y (Kornblith et al. 2019)."""
    n = gram_x.shape[0]
    h = np.eye(n) - 1.0 / n
    kx = h @ gram_x @ h
    ky = h @ (y @ y.T) @ h
    return round(float(np.sum(kx * ky) / np.sqrt(np.sum(kx * kx) * np.sum(ky * ky))), 4)


def word_correlations(space: np.ndarray, train: np.ndarray, word_matrix, common: np.ndarray):
    centre = space[train].mean(axis=0)
    _, _, vt = np.linalg.svd(space[train] - centre, full_matrices=False)
    col_mean = np.asarray(word_matrix.mean(axis=0)).ravel()
    col_sq = np.asarray(word_matrix.multiply(word_matrix).mean(axis=0)).ravel()
    col_std = np.sqrt(np.maximum(col_sq - col_mean ** 2, 1e-12))
    out = []
    for d in range(TOP_DIRECTIONS):
        z = (space - centre) @ vt[d]
        z = (z - z.mean()) / z.std()
        cov = np.asarray(word_matrix.T @ z).ravel() / len(z) - col_mean * z.mean()
        out.append(np.where(common, cov / col_std, 0.0))
    return out


def word_category(word: str, cache: dict) -> str:
    if word not in cache:
        import jieba.posseg as posseg
        tagged = posseg.lcut(word)
        cache[word] = pos_group(tagged[0].flag, word) if len(tagged) == 1 else "multi_token"
    return cache[word]


def build(private_root: Path, out_dir: Path, folds: list[int]) -> int:
    import jieba
    jieba.setLogLevel(60)
    started = time.time()
    print("loading corpus v3", flush=True)
    rows, vectors, _ = load_v3(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, _ = build_songs(rows, vectors)
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
    song_pos = {s: i for i, s in enumerate(songs)}
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    held = np.zeros(label_count, dtype=bool)
    held[np.random.default_rng(SEED).permutation(label_count)[:int(round(HELD_OUT_SHARE * label_count))]] = True
    chunk_rows = [i for s in songs for i in sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"]))]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    print(f"  {len(songs):,} songs, {label_count} labels, {len(order):,} groups, {len(chunk_rows):,} chunks", flush=True)

    print("fitting the word space", flush=True)
    word_matrix, features = fit_words([" ".join(segment(documents[s])) for s in songs])
    word_df = np.asarray((word_matrix > 0).sum(axis=0)).ravel()
    common = (word_df >= MIN_DF) & np.asarray([" " not in f for f in features])
    word_gram = (word_matrix @ word_matrix.T).toarray()
    print(f"  {word_matrix.shape[1]:,} word features, {int(common.sum()):,} unigrams in at least {MIN_DF} songs", flush=True)

    private = private_root / "work" / "private-identity-encoder-v3"
    category_cache: dict[str, str] = {}
    per_fold = {}
    for k in folds:
        print(f"== fold {k}", flush=True)
        recorded = json.loads((out_dir / f"identity_encoder_analysis_fold{k}{TAG}.json").read_text(encoding="utf-8"))
        tuned_chunks = np.load(private / f"fine_tuned_chunk_vectors_fold{k}{TAG}.npy").astype(np.float64)
        frozen_chunks = np.load(private / f"frozen_masked_chunk_vectors_fold{k}.npy").astype(np.float64)
        if tuned_chunks.shape[0] != len(chunk_rows) or frozen_chunks.shape[0] != len(chunk_rows):
            raise SystemExit(f"fold {k}: saved chunk vectors do not match the {len(chunk_rows):,} chunks of this build")
        train = (fold != k) & ~held[label_index]
        scopes = {"test_fold": np.flatnonzero((fold == k) & ~held[label_index]), "unseen_labels": np.flatnonzero(held[label_index])}
        rng_whiten = np.random.default_rng(SEED + k)

        def whiten(x):
            mean, matrix, _ = fit_transform("within_author_whitening", x[train], label_index[train], weights[train], rng_whiten)
            return unit_rows((x - mean) @ matrix.T)

        f_raw = song_level(frozen_chunks, chunk_song, len(songs))
        t_raw = song_level(tuned_chunks, chunk_song, len(songs))
        fw, tw = whiten(f_raw), whiten(t_raw)
        pred_t_from_f, alpha_ft = ridge_predict(fw, tw, train)
        n_raw = tw - pred_t_from_f

        spaces = {"frozen_whitened": fw, "tuned_whitened": tw, "tuned_new_part": unit_rows(n_raw)}
        for seed in NULL_SEEDS:
            spaces[f"null_random_{seed}"] = unit_rows(np.random.default_rng(SEED + 100 * seed + k).standard_normal(fw.shape))

        systems, rr = {}, {}
        for scope, queries in scopes.items():
            systems[scope], rr[scope] = {}, {}
            for name, space in spaces.items():
                ranks = ranks_of(dense_leave_group_out(space, label_index, group_ids, weights, label_count, queries),
                                 label_index[queries])
                systems[scope][name] = round(float(np.mean(1.0 / ranks)), 4)
                full = np.ones(len(songs))
                full[queries] = 1.0 / ranks
                rr[scope][name] = full
            systems[scope]["null_random_mean"] = round(float(np.mean([systems[scope][f"null_random_{s}"] for s in NULL_SEEDS])), 4)
            print(f"  {scope:13s} " + "  ".join(f"{n}={systems[scope][n]:.4f}" for n in
                                                ("frozen_whitened", "tuned_whitened", "tuned_new_part", "null_random_mean")), flush=True)

        checks = {}
        for scope in scopes:
            for mine, theirs in (("frozen_whitened", "frozen_masked_within_author_whitening"),
                                 ("tuned_whitened", "fine_tuned_within_author_whitening")):
                value = recorded["systems"][scope][theirs]["mrr"]
                checks[f"{scope}/{mine}"] = {"recorded": value, "recomputed": systems[scope][mine],
                                             "gap": round(abs(systems[scope][mine] - value), 4)}
        if any(c["gap"] > CHECK_GAP for c in checks.values()):
            raise SystemExit(f"fold {k}: the whitened spaces do not reproduce the recorded analysis: {checks}")
        print(f"  check against identity_encoder_analysis_fold{k}{TAG}.json passed", flush=True)

        contrasts = {}
        for scope, queries in scopes.items():
            mask = np.zeros(len(songs), dtype=bool)
            mask[queries] = True
            contrasts[scope] = paired_group_bootstrap(rr[scope], weights, group_ids, mask,
                                                      [("tuned_whitened", "frozen_whitened"), ("tuned_new_part", "null_random_1")])
            for c in contrasts[scope]:
                print(f"  {scope}: {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
                      f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

        geometry = {}
        for scope, queries in scopes.items():
            g = word_gram[np.ix_(queries, queries)]
            geometry[scope] = {"cka_words_frozen_raw": linear_cka(g, f_raw[queries]),
                               "cka_words_tuned_raw": linear_cka(g, t_raw[queries]),
                               "cka_words_frozen_whitened": linear_cka(g, fw[queries]),
                               "cka_words_tuned_whitened": linear_cka(g, tw[queries]),
                               "cka_words_tuned_new_part": linear_cka(g, n_raw[queries]),
                               "cka_frozen_whitened_tuned_whitened": linear_cka(fw[queries] @ fw[queries].T, tw[queries]),
                               "r2_frozen_whitened_to_tuned_whitened": r_squared(tw, pred_t_from_f, queries, train)}
            print(f"  {scope} geometry {geometry[scope]}", flush=True)

        directions = {}
        for name, space in (("tuned_new_part", n_raw), ("frozen_whitened", fw)):
            corrs = word_correlations(space, train, word_matrix, common)
            shares, listed = Counter(), []
            for d, corr in enumerate(corrs):
                strongest = np.argsort(-np.abs(corr))[:CATEGORY_WORDS]
                shares.update(word_category(features[i], category_cache) for i in strongest)
                listed.append({"direction": d + 1,
                               "positive_end": [features[i] for i in np.argsort(-corr)[:WORDS_PER_END]],
                               "negative_end": [features[i] for i in np.argsort(corr)[:WORDS_PER_END]],
                               "largest_abs_correlation": round(float(np.abs(corr).max()), 3)})
            total = sum(shares.values())
            directions[name] = {"category_share_of_most_correlated_words": {c: round(v / total, 4) for c, v in shares.most_common()},
                                "directions": listed}
            print(f"  {name}: categories {directions[name]['category_share_of_most_correlated_words']}", flush=True)

        per_fold[str(k)] = {"train_songs": int(train.sum()), "queries": {s: int(len(q)) for s, q in scopes.items()},
                            "ridge_alpha_frozen_to_tuned": alpha_ft, "systems": systems, "paired_contrasts": contrasts,
                            "geometry": geometry, "checks_against_recorded_analysis": checks, "word_correlates": directions}
        print(f"  fold {k} done, {(time.time() - started) / 60:.1f} min", flush=True)

    payload = {
        "analysis": "what the 64-anchor GradCache fine-tune learned beyond within-author whitening: descriptive part",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count, "groups": len(order)},
        "design": {"runs": f"fine-tune tag {TAG}, one launch per fold", "folds": folds,
                   "whitening": "within-author, fitted on each fold's training songs (other folds, labels not held out)",
                   "new_part": "tuned whitened vectors minus their multi-output ridge prediction from the frozen whitened "
                               "vectors, fitted on the training songs, alpha by generalised cross-validation over "
                               + ", ".join(f"{a:g}" for a in ALPHAS),
                   "null": f"random Gaussian song vectors, seeds {list(NULL_SEEDS)}",
                   "cka": "linear centred kernel alignment with the raw word TF-IDF space over the scope's songs, label-free",
                   "word_correlates": f"top {TOP_DIRECTIONS} principal directions on the training songs; correlation of each "
                                      f"unigram's TF-IDF weight (words in at least {MIN_DF} songs) with all songs' coordinates; "
                                      f"categories by jieba part of speech for the {CATEGORY_WORDS} strongest per direction",
                   "dropped_design": "removing each space's ridge prediction from the word space was dropped after a one-fold "
                                     "run: fitted on songs of the same labels it strips seen-label identity through the label "
                                     "association alone",
                   "reading": "descriptive only; the causal test of which words the tuned advantage depends on is "
                              "finetune_cue_masking_v3.py"},
        "folds": per_fold,
        "privacy": "aggregate only; the listed words are common vocabulary in at least 50 songs, no lyric line is published",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finetune_learned_identity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'finetune_learned_identity.json'}  ({(time.time() - started) / 60:.1f} min)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(FOLDS)))
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir, args.folds)


if __name__ == "__main__":
    sys.exit(main())
