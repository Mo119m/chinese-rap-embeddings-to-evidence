#!/usr/bin/env python3
"""One character-level language model per label, and the query goes to the label under
which it is least surprising: the generative counterpart of the TF-IDF systems.

Every system so far is discriminative in the retrieval sense: a song and a label profile
are two vectors and cosine says how alike they are. Authorship attribution has an older
model class that scores a text by how well a model OF THE AUTHOR predicts it -- character
n-gram language models with interpolated smoothing (Peng et al. 2003), and lately a GPT-2
fine-tuned per author and read by perplexity (authorial language models, PLOS One 2025),
whose token-level view found content words carrying more authorship than function words,
as the word anatomy here did. This file builds the n-gram version under the project's
protocol so the two model classes can be compared on the same queries:

  model      per label, an interpolated character n-gram model of orders 1..N, counts
             weighted by the builder's per-(group, label) weights, conditional
             probabilities P_n(c | h) = C_n(hc) / C_{n-1}(h), interpolated with fixed
             weights and an add-delta unigram floor
  scoring    the mean log-probability per character of the query song under each label's
             model; the query's own leakage group is subtracted from every label it
             touches before scoring, as the protocol's profiles are
  systems    orders 1..3 and 1..5; the character TF-IDF and the word space on the same
             queries for reference

    python src/ngram_language_model_attribution_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from lexical_identity_anatomy_v2 import score_arm  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
ORDERS = {"ngram_lm_1_to_3": 3, "ngram_lm_1_to_5": 5}
LAMBDA = {3: (0.15, 0.35, 0.50), 5: (0.08, 0.12, 0.20, 0.25, 0.35)}   # weights for orders 1..N, sum 1
DELTA = 0.5                # add-delta floor at the unigram order
MIN_COUNT = 2              # an n-gram seen fewer times in the whole corpus is not modelled
WHITESPACE = re.compile(r"\s+")


def prepare(text: str) -> str:
    return WHITESPACE.sub(" ", text.lower()).strip()


def ngrams(text: str, n: int):
    return (text[i:i + n] for i in range(len(text) - n + 1))


class LabelModels:
    """Sparse count matrices per order (feature x label) and the corpus vocabulary."""

    def __init__(self, texts: list[str], labels: np.ndarray, weights: np.ndarray, label_count: int, max_order: int):
        self.max_order = max_order
        self.index: list[dict[str, int]] = [dict() for _ in range(max_order + 1)]
        self.counts: list[sparse.csr_matrix | None] = [None] * (max_order + 1)
        self.totals = np.zeros(label_count)   # weighted character mass per label (order 1)
        self.vocabulary = 0
        for n in range(1, max_order + 1):
            corpus = Counter()
            for t in texts:
                corpus.update(ngrams(t, n))
            keep = sorted(g for g, c in corpus.items() if c >= MIN_COUNT)
            self.index[n] = {g: i for i, g in enumerate(keep)}
            rows, cols, vals = [], [], []
            for t, label, w in zip(texts, labels, weights):
                local = Counter(ngrams(t, n))
                for g, c in local.items():
                    i = self.index[n].get(g)
                    if i is not None:
                        rows.append(i)
                        cols.append(int(label))
                        vals.append(w * c)
                if n == 1:
                    self.totals[int(label)] += w * len(t)
            self.counts[n] = sparse.csr_matrix((vals, (rows, cols)), shape=(len(keep), label_count))
            if n == 1:
                self.vocabulary = len(keep)
            print(f"    order {n}: {len(keep):,} n-grams modelled", flush=True)

    def gather(self, n: int, grams: list[str]) -> np.ndarray:
        """Dense (positions x labels) counts for a list of n-grams; unmodelled -> 0."""
        idx = np.asarray([self.index[n].get(g, -1) for g in grams])
        out = np.zeros((len(grams), self.counts[n].shape[1]))
        known = idx >= 0
        if known.any():
            out[known] = self.counts[n][idx[known]].toarray()
        return out


def log_likelihood(models: LabelModels, text: str, lambdas, adjust: dict[int, str] | None = None,
                   adjust_weight: dict[int, float] | None = None) -> np.ndarray:
    """Mean log P(text | label) per character under the interpolated model, for every label.

    adjust maps a label to the concatenated text of that label's songs in the query's leakage
    group; their (weighted) counts are subtracted from that label's column before scoring."""
    N = models.max_order
    positions = len(text)
    probs = np.zeros((positions, models.counts[1].shape[1]))
    for n in range(1, N + 1):
        grams = [text[max(0, i - n + 1):i + 1] for i in range(positions)]
        valid = np.asarray([len(g) == n for g in grams])
        numer = models.gather(n, grams)
        if n == 1:
            denom = np.broadcast_to(models.totals, numer.shape).copy()
        else:
            hist = [g[:-1] for g in grams]
            denom = models.gather(n - 1, hist)
        if adjust:
            for label, held_text in adjust.items():
                w = adjust_weight[label]
                held_n = Counter(ngrams(held_text, n))
                numer[:, label] -= w * np.asarray([held_n.get(g, 0) for g in grams])
                if n == 1:
                    denom[:, label] -= w * len(held_text)
                else:
                    held_h = Counter(ngrams(held_text, n - 1))
                    denom[:, label] -= w * np.asarray([held_h.get(h, 0) for h in hist])
        numer = np.clip(numer, 0.0, None)
        denom = np.clip(denom, 0.0, None)
        if n == 1:
            p = (numer + DELTA) / (denom + DELTA * (models.vocabulary + 1))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(denom > 0, numer / np.maximum(denom, 1e-12), 0.0)
            p[~valid] = 0.0
        probs += lambdas[n - 1] * p
    return np.log(np.maximum(probs, 1e-300)).mean(axis=0)


def build(private_root: Path, out_dir: Path) -> int:
    import jieba
    jieba.setLogLevel(60)
    print("loading corpus v3", flush=True)
    rows, vectors, state = load_v3(private_root)
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
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    corpus_docs = [documents[s] for s in songs]
    texts = [prepare(d) for d in corpus_docs]
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups; "
          f"{sum(len(t) for t in texts):,} characters", flush=True)
    members_by_group: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(group_ids.tolist()):
        members_by_group[int(g)].append(i)

    ranks = {}
    report = {}
    for name, N in ORDERS.items():
        print(f"{name}: counting", flush=True)
        models = LabelModels(texts, label_index, weights, label_count, N)
        print("  scoring every query, own leakage group removed", flush=True)
        scores = np.zeros((len(songs), label_count))
        for q in range(len(songs)):
            members = members_by_group[int(group_ids[q])]
            adjust, adjust_weight = {}, {}
            for label in {int(label_index[m]) for m in members}:
                own = [m for m in members if int(label_index[m]) == label]
                adjust[label] = " ".join(texts[m] for m in own)
                adjust_weight[label] = float(weights[own[0]])
            scores[q] = log_likelihood(models, texts[q], LAMBDA[N], adjust, adjust_weight)
            if (q + 1) % 1000 == 0:
                print(f"    {q + 1:,} queries", flush=True)
        ranks[name] = v1.rank_system(scores.astype(np.float32), label_index)[0].astype(np.int64)
        report[name] = {"mrr": round(float(np.mean(1.0 / ranks[name])), 4),
                        "recall_at_1": round(float(np.mean(ranks[name] <= 1)), 4),
                        "recall_at_10": round(float(np.mean(ranks[name] <= 10)), 4),
                        "orders": N, "lambdas": LAMBDA[N],
                        "modelled_ngrams_by_order": {n: len(models.index[n]) for n in range(1, N + 1)}}
        print(f"  {name}: MRR {report[name]['mrr']:.4f}  R@1 {report[name]['recall_at_1']:.4f}  "
              f"R@10 {report[name]['recall_at_10']:.4f}", flush=True)

    print("reference systems", flush=True)
    char_matrix = v1.fit_tfidf(corpus_docs)
    profiles = v1.score_leave_group_out(dense, char_matrix, label_index, group_ids, label_count)
    ranks["lexical_char_2_5"] = v1.rank_system(profiles.lexical, label_index)[0].astype(np.int64)
    word_matrix, _ = fit_words([" ".join(segment(d)) for d in corpus_docs])
    r, error = score_arm(dense, word_matrix, label_index, group_ids, label_count)
    if error:
        raise SystemExit(error)
    ranks["lexical_words"] = r
    for name in ("lexical_char_2_5", "lexical_words"):
        report[name] = {"mrr": round(float(np.mean(1.0 / ranks[name])), 4)}
        print(f"  {name}: MRR {report[name]['mrr']:.4f}", flush=True)

    rr = {name: 1.0 / r for name, r in ranks.items()}
    pairs = [("ngram_lm_1_to_5", "lexical_char_2_5"), ("ngram_lm_1_to_5", "lexical_words"),
             ("ngram_lm_1_to_5", "ngram_lm_1_to_3"), ("ngram_lm_1_to_3", "lexical_char_2_5")]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "per-label character n-gram language models scored by log-likelihood, against the TF-IDF spaces",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count, "groups": len(order)},
        "design": {"text": "the song's cleaned text, lower-cased, whitespace runs folded to one space",
                   "model": "interpolated character n-gram model per label, orders 1..N, fixed interpolation weights, "
                            f"add-{DELTA} unigram floor, n-grams seen fewer than {MIN_COUNT} times in the corpus not modelled",
                   "weights": "each song's counts enter its label's model with the builder's per-(group, label) weight",
                   "leave_group_out": "the query's leakage group is subtracted from every label it touches before scoring",
                   "score": "mean log-probability per character",
                   "reference": "Peng et al. 2003 (character n-gram LMs for attribution); authorial language models, "
                                "PLOS One 2025 (per-author GPT-2 read by perplexity)"},
        "systems": report,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with replacement, "
                                       "each (group, label) component weighted one", "contrasts": contrasts},
        "privacy": "aggregate only",
    }
    (out_dir / "ngram_language_model_attribution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'ngram_language_model_attribution.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
