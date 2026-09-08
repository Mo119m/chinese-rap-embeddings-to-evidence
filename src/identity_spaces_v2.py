#!/usr/bin/env python3
"""Where does a rapper's identity live? One protocol, three spaces.

The retrieval task asks one question -- given a song, which of the source-credit labels
wrote it -- and answers it from one kind of representation at a time. The dense system
represents a song by what it means: BGE-M3, trained to bring paraphrases together. The
lexical system represents it by what it is made of: character 2-5-grams, TF-IDF weighted.
The decomposition experiment found the second beats the first by a wide margin, 0.450 to
0.318 MRR, and that the advantage is carried neither by Latin script nor by named entities.

This adds a third representation that carries no lexical content at all: how the rapper
rhymes. Every classifiable line ending is reduced to its rhyme family (the seventeen
written-final classes the written-rhyme task uses), the tone of its last syllable, the
family pair across its last two syllables, the transition from the previous line's family,
and the length of each run of same-family endings. A song becomes a bag of those tokens,
TF-IDF weighted the way the lexical system weights its n-grams, and scored under the
identical leave-group-out protocol. Which words the rapper used is gone. What survives is
the rhyme scheme.

Two tiers, because the line between phonology and vocabulary is not sharp. The strict tier
uses families, tones, pairs, transitions and runs only. The second tier adds the raw pinyin
final of the last syllable, which is closer to the syllable itself and so leaks a little of
which words end lines. Both are reported; the gap between them is part of the result.

Every system is scored on the same queries with the same leakage groups, and the dense arm
is required to reproduce the builder's scores before anything is compared. Songs with too
few classifiable endings get the worst rank in the phonological systems rather than a
tie-broken one, and every comparison is repeated on the subset where all systems are
defined. Differences between systems carry a paired bootstrap over leakage groups, with
the builder's per-(group, label) weighting, so a song filed five times counts once.

    python src/identity_spaces_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
import build_chinese_rap_written_rhyme_v1 as wr  # noqa: E402
from build_downstream_retrieval_v2 import (  # noqa: E402
    MINIMUM_SONGS_PER_LABEL,
    build_songs,
    load,
)
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
SEED = 20260825
REPLICATES = 2000
MINIMUM_ENDINGS = 4
EXPECTED_DENSE_LEXICAL_FUSION_MRR = 0.4756  # the decomposition experiment's plain mean


# ------------------------------------------------------------------ phonological documents
def ending_sequence(chunk_texts: list[str]) -> list[dict]:
    """The written-rhyme task's line filter, applied to a song's chunks in order.

    Mirrors build_written_rhyme_v2.load_corpus_v2 line for line: a line counts only if it
    is not a section header, contains a Han character, ends in one, survives duplicate
    normalisation, and has a classifiable written final.
    """
    sequence = []
    for text in chunk_texts:
        for raw_line in text.split("\n"):
            line = wr.display_normalise(raw_line)
            if not line or wr.is_header_line(line) or not wr.HAN_RE.search(line):
                continue
            if wr.terminal_content_kind(line) != "han" or not wr.duplicate_normalise(line):
                continue
            ending = wr.written_ending_features(line)
            if ending is not None:
                sequence.append(ending)
    return sequence


def phonological_tokens(sequence: list[dict], with_finals: bool) -> str:
    tokens = []
    previous = None
    run = 0
    for ending in sequence:
        family = ending["final_family"]
        tokens.append("FAM_" + family)
        tokens.append("TONE_" + ending["tone"])
        if ending["two_family_pattern"]:
            tokens.append("PAIR_" + ending["two_family_pattern"])
        if previous is not None:
            tokens.append("TR_" + previous + ">" + family)
            tokens.append("SAME" if previous == family else "SHIFT")
        if with_finals:
            tokens.append("FIN_" + ending["raw_final"])
            tokens.append("FINT_" + ending["raw_final"] + ending["tone"])
            if ending["two_final_pattern"]:
                tokens.append("PFIN_" + ending["two_final_pattern"])
        # one token per maximal run of same-family endings, emitted when the run closes
        if previous is not None and family == previous:
            run += 1
        else:
            if previous is not None:
                tokens.append("RUN_" + wr.run_bucket(run))
            run = 1
        previous = family
    if previous is not None:
        tokens.append("RUN_" + wr.run_bucket(run))
    return " ".join(tokens)


def fit_phonological(documents: list[str]) -> tuple[sparse.csr_matrix, list[str]]:
    # the lexical system's weighting -- min_df 3, sublinear tf, l2 rows -- over rhyme
    # tokens instead of character n-grams; empty songs stay zero rows
    vectorizer = TfidfVectorizer(analyzer=str.split, min_df=3, sublinear_tf=True,
                                 norm="l2", dtype=np.float32)
    matrix = vectorizer.fit_transform(documents).tocsr().astype(np.float32)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    nonzero = norms > 0
    if not np.allclose(norms[nonzero], 1.0, atol=2e-5):
        raise RuntimeError("phonological rows are not unit length")
    return matrix, list(vectorizer.get_feature_names_out())


# ------------------------------------------------------------------ scoring
def ranks_for(scores: np.ndarray, label_index: np.ndarray, covered: np.ndarray,
              label_count: int) -> np.ndarray:
    ranks, _ = v1.rank_system(scores.astype(np.float32), label_index)
    ranks = ranks.astype(np.int64)
    ranks[~covered] = label_count  # the worst rank, not a tie broken by label order
    return ranks


def fuse(matrices: list[np.ndarray], covered: np.ndarray) -> np.ndarray:
    # z-score fusion, as the builder does it, on the rows where every arm is defined;
    # undefined rows are left at zero and ranked worst by ranks_for
    fused = np.zeros_like(matrices[0], dtype=np.float64)
    fused[covered] = np.mean([v1.zscore_rows(m[covered].astype(np.float64))
                              for m in matrices], axis=0)
    return fused


def weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def paired_group_bootstrap(rr_by_system: dict[str, np.ndarray], weights: np.ndarray,
                           group_ids: np.ndarray, mask: np.ndarray, pairs):
    """Percentile intervals for MRR differences, resampling leakage groups.

    Per-group sums of weight x reciprocal rank are precomputed once per system; a
    replicate then draws groups with replacement and takes the ratio of sums, so the
    estimand is the builder's per-(group, label)-weighted MRR restricted to `mask`.
    """
    groups = np.unique(group_ids[mask])
    position = {int(g): i for i, g in enumerate(groups.tolist())}
    index = np.asarray([position[int(g)] for g in group_ids[mask]])
    weight_sum = np.bincount(index, weights=weights[mask], minlength=len(groups))
    numerators = {name: np.bincount(index, weights=(rr[mask] * weights[mask]),
                                    minlength=len(groups))
                  for name, rr in rr_by_system.items()}
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(groups), size=(REPLICATES, len(groups)))
    denominators = weight_sum[draws].sum(axis=1)
    replicate = {name: num[draws].sum(axis=1) / denominators
                 for name, num in numerators.items()}
    out = []
    for left, right in pairs:
        diff = replicate[left] - replicate[right]
        point = (numerators[left].sum() - numerators[right].sum()) / weight_sum.sum()
        low, high = np.percentile(diff, [2.5, 97.5])
        out.append({"system": left, "minus": right, "mrr_difference": round(float(point), 4),
                    "ci95": [round(float(low), 4), round(float(high), 4)],
                    "excludes_zero": bool(low > 0 or high < 0)})
    return out


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
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    # the builder's weighting: each (group, label) component has total weight one
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))]
                          for g, l in zip(group_ids, label_index)])

    print("phonological transcription", flush=True)
    sequences = {}
    for song in songs:
        ordered = sorted(chunks_by_song[song], key=lambda i: int(rows[i]["source_order"]))
        sequences[song] = ending_sequence([rows[i]["cleaned_text"] for i in ordered])
    ending_counts = np.asarray([len(sequences[s]) for s in songs])
    covered = ending_counts >= MINIMUM_ENDINGS
    print(f"  endings per song: median {int(np.median(ending_counts))}, "
          f"{int((ending_counts == 0).sum()):,} songs with none, "
          f"{int((~covered).sum()):,} below {MINIMUM_ENDINGS}", flush=True)
    phon_strict, strict_features = fit_phonological(
        [phonological_tokens(sequences[s], with_finals=False) for s in songs])
    phon_finals, finals_features = fit_phonological(
        [phonological_tokens(sequences[s], with_finals=True) for s in songs])
    print(f"  strict tier {len(strict_features)} features, with finals "
          f"{len(finals_features)}", flush=True)

    print("scoring every space under the one protocol", flush=True)
    lexical = v1.fit_tfidf([documents[s] for s in songs])
    reference = v1.score_leave_group_out(dense, lexical, label_index, group_ids, label_count)
    strict = v1.score_leave_group_out(dense, phon_strict, label_index, group_ids, label_count)
    finals = v1.score_leave_group_out(dense, phon_finals, label_index, group_ids, label_count)
    gap = max(float(np.abs(reference.dense - strict.dense).max()),
              float(np.abs(reference.dense - finals.dense).max()))
    print(f"  dense arm reproduced across calls: max gap {gap:.3e}", flush=True)
    if gap > 1e-6:
        raise SystemExit("the dense scores differ between calls; the protocol is not shared")

    # a frequency prior: rank labels by how many groups they own outside the query's group
    groups_per_label = np.rint(reference.label_group_counts).astype(np.float64)
    prior = np.tile(groups_per_label, (len(songs), 1))
    labels_in_group: dict[int, set[int]] = defaultdict(set)
    for g, l in zip(group_ids.tolist(), label_index.tolist()):
        labels_in_group[g].add(l)
    for q in range(len(songs)):
        for l in labels_in_group[int(group_ids[q])]:
            prior[q, l] -= 1.0

    everything = np.ones(len(songs), dtype=bool)
    systems = {
        "label_frequency_prior": (prior, everything),
        "dense_semantic": (reference.dense, everything),
        "lexical_character_ngrams": (reference.lexical, everything),
        "phonological_strict": (strict.lexical, covered),
        "phonological_with_finals": (finals.lexical, covered),
        "fusion_dense_lexical": (fuse([reference.dense, reference.lexical], everything), everything),
        "fusion_dense_phonological": (fuse([reference.dense, strict.lexical], covered), covered),
        "fusion_lexical_phonological": (fuse([reference.lexical, strict.lexical], covered), covered),
        "fusion_all_three": (fuse([reference.dense, reference.lexical, strict.lexical], covered),
                             covered),
    }
    ranks = {name: ranks_for(scores, label_index, mask, label_count)
             for name, (scores, mask) in systems.items()}

    # the shared protocol must reproduce the decomposition experiment before anything else
    dl = float(np.mean(1.0 / ranks["fusion_dense_lexical"]))
    if abs(dl - EXPECTED_DENSE_LEXICAL_FUSION_MRR) > 5e-4:
        raise SystemExit(f"dense+lexical fusion MRR {dl:.4f} does not reproduce "
                         f"{EXPECTED_DENSE_LEXICAL_FUSION_MRR}; the query set differs")

    rr = {name: 1.0 / r for name, r in ranks.items()}
    chance = float(np.mean(1.0 / np.arange(1, label_count + 1)))
    report = {}
    for name, r in ranks.items():
        report[name] = {
            "all_queries": {
                "mrr": round(float(np.mean(1.0 / r)), 4),
                "mrr_component_weighted": round(weighted_mean(rr[name], weights, everything), 4),
                "recall_at_1": round(float(np.mean(r <= 1)), 4),
                "recall_at_10": round(float(np.mean(r <= 10)), 4),
            },
            "covered_subset": {
                "mrr": round(float(np.mean(1.0 / r[covered])), 4),
                "mrr_component_weighted": round(weighted_mean(rr[name], weights, covered), 4),
                "recall_at_1": round(float(np.mean(r[covered] <= 1)), 4),
                "recall_at_10": round(float(np.mean(r[covered] <= 10)), 4),
            },
        }
        print(f"  {name:28s} MRR all {report[name]['all_queries']['mrr']:.4f}  "
              f"covered {report[name]['covered_subset']['mrr']:.4f}  "
              f"R@10 {report[name]['covered_subset']['recall_at_10']:.4f}", flush=True)

    print("paired bootstrap over leakage groups", flush=True)
    pairs = [
        ("lexical_character_ngrams", "dense_semantic"),
        ("phonological_strict", "label_frequency_prior"),
        ("phonological_strict", "dense_semantic"),
        ("phonological_with_finals", "phonological_strict"),
        ("lexical_character_ngrams", "phonological_strict"),
        ("fusion_dense_phonological", "dense_semantic"),
        ("fusion_lexical_phonological", "lexical_character_ngrams"),
        ("fusion_all_three", "fusion_dense_lexical"),
    ]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, covered, pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    # head-to-head on the covered subset
    head = {}
    for left, right in combinations(("dense_semantic", "lexical_character_ngrams",
                                     "phonological_strict"), 2):
        a, b = ranks[left][covered], ranks[right][covered]
        head[f"{left}_vs_{right}"] = {"first_better": int((a < b).sum()),
                                      "second_better": int((b < a).sum()),
                                      "tied": int((a == b).sum())}

    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir = private_root / "work" / "private-identity-spaces-v2"
    private_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    fields = ["song_id", "label", "group", "endings", "covered"] + [f"rank_{n}" for n in ranks]
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for q, song in enumerate(songs):
        writer.writerow({"song_id": song, "label": eligible[label_index[q]],
                         "group": int(group_ids[q]), "endings": int(ending_counts[q]),
                         "covered": int(covered[q]),
                         **{f"rank_{n}": int(r[q]) for n, r in ranks.items()}})
    (private_dir / "per_query.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")

    payload = {
        "analysis": "author identity across semantic, lexical and phonological spaces",
        "protocol": ("one song, one vector per space; label profiles are leave-group-out "
                     "weighted centroids; cosine ranking over every eligible label. "
                     "Identical queries, labels and leakage groups in every space."),
        "queries": len(songs), "labels": label_count, "groups": len(order),
        "chance_mrr": round(chance, 4),
        "verification": {
            "dense_scores_identical_across_calls": True,
            "dense_max_gap": gap,
            "dense_lexical_fusion_reproduces_decomposition": round(dl, 4),
        },
        "phonological_space": {
            "tokens_strict": ("rhyme family of the line-final syllable (17 classes), its tone, "
                              "the family pair across the last two syllables, the transition "
                              "from the previous line's family, same/shift, and one token per "
                              "maximal run of same-family endings"),
            "tokens_with_finals": "the strict set plus the raw pinyin final, toned final and "
                                  "final pair, which are closer to the syllable itself",
            "features_strict": len(strict_features),
            "features_with_finals": len(finals_features),
            "weighting": "TF-IDF with the lexical system's settings (min_df 3, sublinear tf, "
                         "l2 rows) over rhyme tokens instead of character n-grams",
            "endings_per_song_quantiles": {str(p): int(np.percentile(ending_counts, p))
                                           for p in (10, 25, 50, 75, 90)},
            "songs_with_no_ending": int((ending_counts == 0).sum()),
            "minimum_endings_for_coverage": MINIMUM_ENDINGS,
            "covered_queries": int(covered.sum()),
            "uncovered_rank_policy": "worst rank in every system that uses the phonological "
                                     "space; excluded from the covered subset",
        },
        "systems": report,
        "head_to_head_covered": head,
        "paired_contrasts_covered": {
            "design": (f"{REPLICATES} replicates, seed {SEED}, leakage groups resampled with "
                       "replacement, each (group, label) component weighted one, restricted "
                       "to the covered subset so every system is defined on every query"),
            "contrasts": contrasts,
        },
        "privacy": "aggregate only; per-query ranks are private",
    }
    (out_dir / "identity_spaces.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'identity_spaces.json'}")
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
