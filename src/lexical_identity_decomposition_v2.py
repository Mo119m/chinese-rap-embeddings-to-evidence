#!/usr/bin/env python3
"""What does the character-n-gram system win on -- and does it survive removing the topic?

On corpus v2, character 2-5 gram TF-IDF retrieves a song's source-credit label better than
BGE-M3 dense embeddings do (MRR 0.398 against 0.301). That is the observation behind the
claim that author identity in Chinese rap lives in the lexical surface rather than in
meaning. An observation is not a mechanism, and two things could make it false.

First, it could be winning on nothing. If the discriminative n-grams are punctuation and
particles, "lexical surface" is an accident of tokenisation, not a property of the writing.
So the correct label's lexical score is decomposed feature by feature and bucketed by script:
Han, Latin, mixed, other. A third of the corpus's characters are Latin letters and 74.5% of
chunks contain some, so if identity rides on when and how a rapper switches into English,
the Latin bucket should carry a disproportionate share.

Second, it could be topic. Rappers name their cities. A system that recognises 重庆 is
recognising a subject, not a style, and the authorship literature treats this confound as
the central one (the topic-confusion task; topic leakage in cross-topic verification). The
control is neutralisation: strip every surface in the project's own reference lexicon --
place names and named references -- refit, rerun, and see whether the lexical system still
beats the original dense one. If the gap closes, identity here is partly geography, which is
a finding too, just a narrower one.

Both arms reuse the pipeline's own scoring. The TF-IDF refit is asserted byte-identical to
the builder's, and the leave-group-out lexical profiles recomputed for the decomposition are
asserted to reproduce the builder's lexical scores before anything is reported.

    python src/lexical_identity_decomposition_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import (  # noqa: E402
    MINIMUM_SONGS_PER_LABEL,
    build_songs,
    load,
)
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)

OUT_DIR = ROOT / "results" / "retrieval-v2"
HAN = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def script_class(feature: str) -> str:
    text = feature.replace(" ", "")
    han, latin = bool(HAN.search(text)), bool(LATIN.search(text))
    if han and latin:
        return "mixed"
    if han:
        return "han"
    if latin:
        return "latin"
    return "other"


def latin_share(text: str) -> float:
    han, latin = len(HAN.findall(text)), len(LATIN.findall(text))
    return latin / (han + latin) if han + latin else 0.0


def fit_with_vocabulary(documents):
    """The pipeline's TF-IDF, refit here because fit_tfidf returns no vocabulary."""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3,
                                 max_features=v1.TFIDF_MAX_FEATURES, sublinear_tf=True,
                                 norm="l2", dtype=np.float32)
    matrix = vectorizer.fit_transform(documents).tocsr().astype(np.float32)
    reference = v1.fit_tfidf(documents)
    if matrix.shape != reference.shape or abs(matrix - reference).max() != 0.0:
        raise SystemExit("the TF-IDF refit is not byte-identical to the builder's")
    return matrix, vectorizer.get_feature_names_out()


def lexical_profiles(lexical, label_index, group_ids, label_count):
    """Leave-group-out lexical profiles as vectors, the builder's arithmetic re-expressed."""
    song_count = len(label_index)
    counts = Counter((int(g), int(l)) for g, l in zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / counts[(int(group_ids[i]), int(label_index[i]))]
                          for i in range(song_count)], dtype=np.float64)
    membership = sparse.csr_matrix((weights, (label_index, np.arange(song_count))),
                                   shape=(label_count, song_count), dtype=np.float64)
    sums = (membership @ lexical.astype(np.float64)).tocsr()
    members_by_group: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(group_ids.tolist()):
        members_by_group[int(g)].append(i)

    def profile_for(query: int, label: int):
        vector = sums.getrow(label)
        members = [i for i in members_by_group[int(group_ids[query])]
                   if int(label_index[i]) == label]
        if members:
            held = lexical[members].astype(np.float64).multiply(weights[members, None]).sum(axis=0)
            vector = vector - sparse.csr_matrix(held)
        norm = float(np.sqrt(vector.multiply(vector).sum()))
        if norm <= 1e-12:
            raise SystemExit("an empty leave-group-out lexical profile")
        return vector / norm

    return profile_for


def neutralise(text: str, surfaces: list[str]) -> str:
    for surface in surfaces:
        if LATIN.fullmatch(surface.replace(" ", "")):
            text = re.sub(re.escape(surface), " ", text, flags=re.IGNORECASE)
        else:
            text = text.replace(surface, " ")
    return text


def evaluate(dense, lexical, label_index, group_ids, label_count):
    profiles = v1.score_leave_group_out(dense, lexical, label_index, group_ids, label_count)
    fused = (v1.zscore_rows(profiles.dense) + v1.zscore_rows(profiles.lexical)) / 2.0
    out = {}
    for name, scores in (("dense", profiles.dense), ("lexical", profiles.lexical),
                         ("fusion", fused)):
        ranks, _ = v1.rank_system(scores, label_index)
        out[name] = {"ranks": ranks, "mrr": float(np.mean(1.0 / ranks)),
                     "recall_at_10": float(np.mean(ranks <= 10))}
    return profiles, out


def build(private_root: Path, out_dir: Path, lexicon_path: Path) -> int:
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
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    corpus_docs = [documents[s] for s in songs]
    print(f"  {len(songs):,} queries, {len(eligible)} labels", flush=True)

    # ---------------------------------------------------------------- original arm
    print("original arm", flush=True)
    lexical, features = fit_with_vocabulary(corpus_docs)
    classes = np.asarray([script_class(f) for f in features])
    profiles, original = evaluate(dense, lexical, label_index, group_ids, len(eligible))
    for name in ("dense", "lexical", "fusion"):
        print(f"  {name:8s} MRR {original[name]['mrr']:.4f}", flush=True)

    # ---------------------------------------------------------------- decomposition
    print("decomposing the correct label's lexical score by script class", flush=True)
    profile_for = lexical_profiles(lexical, label_index, group_ids, len(eligible))
    lex_scores = profiles.lexical
    per_query = []
    check_gap = 0.0
    for q in range(len(songs)):
        truth = int(label_index[q])
        profile = profile_for(q, truth)
        row = lexical.getrow(q).astype(np.float64)
        contribution = row.multiply(profile)
        total = float(contribution.sum())
        check_gap = max(check_gap, abs(total - float(lex_scores[q, truth])))
        by_class = Counter()
        for idx, value in zip(contribution.indices, contribution.data):
            by_class[str(classes[idx])] += float(value)
        wrong = np.argsort(-lex_scores[q])
        rival = int(wrong[0] if wrong[0] != truth else wrong[1])
        per_query.append({
            "song_id": songs[q], "label": eligible[truth],
            "latin_share": round(latin_share(documents[songs[q]]), 4),
            "rank_dense": int(original["dense"]["ranks"][q]),
            "rank_lexical": int(original["lexical"]["ranks"][q]),
            "rank_fusion": int(original["fusion"]["ranks"][q]),
            "lexical_score_correct": round(total, 6),
            "lexical_score_rival": round(float(lex_scores[q, rival]), 6),
            **{f"share_{k}": round(by_class[k] / total, 4) if total else 0.0
               for k in ("han", "latin", "mixed", "other")},
        })
        if (q + 1) % 2000 == 0:
            print(f"  {q + 1:,}/{len(songs):,}", flush=True)
    print(f"  recomputed profiles vs builder: max gap {check_gap:.3e}", flush=True)
    if check_gap > 1e-4:
        raise SystemExit("recomputed lexical profiles do not reproduce the builder's scores")

    # ---------------------------------------------------------------- aggregates
    latin_s = np.asarray([p["latin_share"] for p in per_query])
    rd = np.asarray([p["rank_dense"] for p in per_query], dtype=float)
    rl = np.asarray([p["rank_lexical"] for p in per_query], dtype=float)
    lex_wins = rl < rd
    dense_wins = rd < rl

    def mean_shares(mask):
        return {k: round(float(np.mean([p[f"share_{k}"] for p, m in zip(per_query, mask) if m])), 4)
                for k in ("han", "latin", "mixed", "other")}

    # Spearman via rank correlation, no scipy.stats dependency assumptions
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    quartiles = np.quantile(latin_s, [0.25, 0.5, 0.75])
    bucket = np.digitize(latin_s, quartiles)
    by_quartile = []
    for b in range(4):
        m = bucket == b
        by_quartile.append({
            "latin_share_quartile": b + 1, "queries": int(m.sum()),
            "latin_share_range": [round(float(latin_s[m].min()), 3), round(float(latin_s[m].max()), 3)],
            "mrr_dense": round(float(np.mean(1 / rd[m])), 4),
            "mrr_lexical": round(float(np.mean(1 / rl[m])), 4),
            "mrr_fusion": round(float(np.mean(1 / np.asarray([p["rank_fusion"] for p, mm in zip(per_query, m) if mm]))), 4),
        })

    # ---------------------------------------------------------------- topic control
    print("topic-neutralised arm", flush=True)
    surfaces = sorted({r["entity"].strip() for r in csv.DictReader(lexicon_path.open(encoding="utf-8"))
                       if r.get("entity", "").strip()}, key=len, reverse=True)
    neutral_docs = [neutralise(d, surfaces) for d in corpus_docs]
    removed = sum(len(a) - len(b) for a, b in zip(corpus_docs, neutral_docs))
    touched = sum(1 for a, b in zip(corpus_docs, neutral_docs) if a != b)
    lexical_n = v1.fit_tfidf(neutral_docs)
    _, neutral = evaluate(dense, lexical_n, label_index, group_ids, len(eligible))
    for name in ("dense", "lexical", "fusion"):
        print(f"  {name:8s} MRR {neutral[name]['mrr']:.4f}  (dense unchanged by construction)",
              flush=True)

    # ---------------------------------------------------------------- write
    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir = private_root / "work" / "private-lexical-identity-decomposition-v2"
    private_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(per_query[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(per_query)
    (private_dir / "per_query.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")

    payload = {
        "analysis": "lexical identity decomposition and topic control",
        "queries": len(songs), "labels": len(eligible),
        "verification": {
            "tfidf_refit_byte_identical_to_builder": True,
            "lexical_profile_recomputation_max_gap": check_gap,
        },
        "feature_inventory_by_script": {k: int(v) for k, v in Counter(classes.tolist()).items()},
        "original": {name: {"mrr": round(original[name]["mrr"], 4),
                            "recall_at_10": round(original[name]["recall_at_10"], 4)}
                     for name in ("dense", "lexical", "fusion")},
        "head_to_head": {
            "lexical_ranks_better": int(lex_wins.sum()),
            "dense_ranks_better": int(dense_wins.sum()),
            "tied": int((rl == rd).sum()),
        },
        "correct_label_score_share_by_script": {
            "all_queries": mean_shares(np.ones(len(per_query), bool)),
            "queries_where_lexical_beats_dense": mean_shares(lex_wins),
            "queries_where_dense_beats_lexical": mean_shares(dense_wins),
            "reading": ("share of the correct label's lexical score contributed by Han, "
                        "Latin, mixed-script and other n-grams; compare against the "
                        "corpus-wide TF-IDF mass split, han 56.3% / latin 41.1%"),
        },
        "latin_share_vs_advantage": {
            "spearman_latin_share_vs_rank_dense_minus_rank_lexical": round(spearman(latin_s, rd - rl), 4),
            "reading": ("positive means songs with more Latin script are the ones where the "
                        "lexical system pulls ahead of the dense one"),
            "by_latin_share_quartile": by_quartile,
        },
        "topic_neutralisation": {
            "lexicon": str(lexicon_path.name), "surfaces_removed": len(surfaces),
            "characters_removed": int(removed), "documents_touched": int(touched),
            "lexical_mrr_original": round(original["lexical"]["mrr"], 4),
            "lexical_mrr_neutralised": round(neutral["lexical"]["mrr"], 4),
            "dense_mrr_original": round(original["dense"]["mrr"], 4),
            "fusion_mrr_neutralised_lexical_plus_original_dense": round(neutral["fusion"]["mrr"], 4),
            "reading": ("if the neutralised lexical system still beats the original dense "
                        "system, the lexical advantage is not carried by named places and "
                        "references; the dense arm is unchanged here because neutralising "
                        "it needs re-embedding, which is a separate GPU run"),
        },
        "privacy": "aggregate only; per-query rows are private",
    }
    (out_dir / "lexical_identity_decomposition.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'lexical_identity_decomposition.json'}")
    print(f"wrote private per-query table to {private_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--lexicon", type=Path,
                        help="entity surfaces to strip; defaults to the private reference core")
    args = parser.parse_args()
    root = args.private_root.resolve()
    lexicon = args.lexicon or (root / "work" / "chinese_rap_ner_reference_core_v1.csv")
    return build(root, args.out_dir, lexicon)


if __name__ == "__main__":
    sys.exit(main())
