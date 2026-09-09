#!/usr/bin/env python3
"""What in the character surface identifies a label?

The decomposition experiment established what the lexical advantage is not: not Latin
script, not any of 605 named references and slang terms. This file asks what it is, in the
only way a bag of character n-grams allows -- by class of feature. Every one of the 150,000
n-grams is tagged along five axes: its script (Han, Latin, mixed, other), its length (2 to
5 characters), whether it touches a line or word boundary (contains a space -- the analyser
folds every newline and run of whitespace to one space, so a boundary n-gram is one that
reaches across a line break or between Latin words), whether it is made entirely of the
hundred most frequent Han characters in the corpus (function-character strings such as
的我你不是), and whether it carries a digit or punctuation.

Two measurements per class, because mass and importance are different things. The *share*
is how much of the correct label's lexical score the class contributes in the full model.
The *ablation* removes the class's columns, re-normalises every song row, and rescores under
the unchanged protocol; the *only* arm keeps nothing but the class. Columns are masked rather
than the model refit, so the inverse document frequencies stay the full model's and a
removed class cannot be silently replaced by the next 150,000 features.

The full model is required to reproduce the builder's lexical MRR before any arm is trusted.

    python src/lexical_identity_anatomy_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import (  # noqa: E402
    MINIMUM_SONGS_PER_LABEL,
    build_songs,
    load,
)
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from lexical_identity_decomposition_v2 import (  # noqa: E402
    HAN,
    fit_with_vocabulary,
    lexical_profiles,
    script_class,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
FUNCTION_CHARACTERS = 100
from build_downstream_retrieval_v2 import expected  # noqa: E402
EXPECTED_LEXICAL_MRR = expected(0.4503, 0.4267)


def has_digit_or_punct(feature: str) -> bool:
    for char in feature:
        if char == " ":
            continue
        category = unicodedata.category(char)
        if char.isdigit() or category.startswith("P") or category.startswith("S"):
            return True
    return False


def feature_classes(features, common_han: set[str]) -> dict[str, np.ndarray]:
    """Boolean masks over the vocabulary, one per class."""
    scripts = np.asarray([script_class(f) for f in features])
    lengths = np.asarray([len(f) for f in features])
    boundary = np.asarray([" " in f for f in features])
    digit_punct = np.asarray([has_digit_or_punct(f) for f in features])
    function_only = np.asarray([
        scripts[i] == "han" and not digit_punct[i]
        and all(c in common_han for c in f.replace(" ", ""))
        for i, f in enumerate(features)])
    han_with_rare = (scripts == "han") & ~function_only
    return {
        "script_han": scripts == "han",
        "script_latin": scripts == "latin",
        "script_mixed_or_other": (scripts == "mixed") | (scripts == "other"),
        "length_2": lengths == 2,
        "length_3": lengths == 3,
        "length_4": lengths == 4,
        "length_5": lengths == 5,
        "boundary": boundary,
        "function_only": function_only,
        "han_with_a_less_common_character": han_with_rare,
        "digit_or_punctuation": digit_punct,
    }


def masked(matrix: sparse.csr_matrix, keep: np.ndarray) -> sparse.csr_matrix:
    """Keep the columns in `keep`, drop the rest, re-normalise each row to unit length."""
    kept = matrix @ sparse.diags(keep.astype(np.float32))
    kept = kept.tocsr()
    kept.eliminate_zeros()
    norms = np.sqrt(np.asarray(kept.multiply(kept).sum(axis=1)).ravel())
    scale = np.where(norms > 0, 1.0 / np.maximum(norms, 1e-12), 0.0).astype(np.float32)
    return (sparse.diags(scale) @ kept).tocsr()


def score_arm(dense, lexical, label_index, group_ids, label_count):
    """Lexical ranks under the protocol; zero rows get the worst rank."""
    try:
        profiles = v1.score_leave_group_out(dense, lexical, label_index, group_ids, label_count)
    except RuntimeError as error:
        return None, str(error)
    ranks, _ = v1.rank_system(profiles.lexical, label_index)
    ranks = ranks.astype(np.int64)
    empty = np.asarray(lexical.getnnz(axis=1) == 0)
    ranks[empty] = label_count
    return ranks, None


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
    corpus_docs = [documents[s] for s in songs]
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))]
                          for g, l in zip(group_ids, label_index)])
    print(f"  {len(songs):,} queries, {label_count} labels", flush=True)

    print("fitting and tagging the vocabulary", flush=True)
    lexical, features = fit_with_vocabulary(corpus_docs)
    character_counts = Counter(c for doc in corpus_docs for c in HAN.findall(doc))
    common_han = {c for c, _ in character_counts.most_common(FUNCTION_CHARACTERS)}
    classes = feature_classes(features, common_han)
    inventory = {name: int(mask.sum()) for name, mask in classes.items()}
    print("  " + ", ".join(f"{k} {v:,}" for k, v in inventory.items()), flush=True)

    print("full model", flush=True)
    full_ranks, error = score_arm(dense, lexical, label_index, group_ids, label_count)
    if error:
        raise SystemExit(error)
    full_mrr = float(np.mean(1.0 / full_ranks))
    print(f"  lexical MRR {full_mrr:.4f}", flush=True)
    if abs(full_mrr - EXPECTED_LEXICAL_MRR) > 5e-4:
        raise SystemExit(f"the full model gives {full_mrr:.4f}, not {EXPECTED_LEXICAL_MRR}")

    # ---------------------------------------------------------------- shares
    print("share of the correct label's score by class", flush=True)
    profile_for = lexical_profiles(lexical, label_index, group_ids, label_count)
    class_names = list(classes)
    class_matrix = np.stack([classes[n] for n in class_names]).astype(np.float64)  # C x F
    totals = np.zeros(len(class_names))
    grand = 0.0
    for q in range(len(songs)):
        truth = int(label_index[q])
        profile = profile_for(q, truth)
        row = lexical.getrow(q).astype(np.float64)
        contribution = row.multiply(profile)
        total = float(contribution.sum())
        if total <= 0:
            continue
        by_class = class_matrix[:, contribution.indices] @ contribution.data
        totals += by_class / total
        grand += 1.0
        if (q + 1) % 2000 == 0:
            print(f"  {q + 1:,}/{len(songs):,}", flush=True)
    shares = {name: round(float(totals[i] / grand), 4) for i, name in enumerate(class_names)}
    mass = {name: round(float(np.asarray(lexical[:, classes[name]].multiply(
        lexical[:, classes[name]]).sum()) / float(lexical.multiply(lexical).sum())), 4)
        for name in class_names}

    # ---------------------------------------------------------------- arms
    arms = {}
    rr = {"full": 1.0 / full_ranks}
    plan = [("remove", name) for name in class_names] + [("only", name) for name in class_names]
    for kind, name in plan:
        keep = ~classes[name] if kind == "remove" else classes[name]
        arm = f"{kind}_{name}"
        print(f"{arm}: {int(keep.sum()):,} features kept", flush=True)
        ranks, error = score_arm(dense, masked(lexical, keep), label_index, group_ids, label_count)
        if error:
            arms[arm] = {"kind": kind, "class": name, "features_kept": int(keep.sum()),
                         "defined": False, "why": error}
            print(f"  undefined: {error}", flush=True)
            continue
        rr[arm] = 1.0 / ranks
        arms[arm] = {"kind": kind, "class": name, "features_kept": int(keep.sum()),
                     "defined": True, "mrr": round(float(np.mean(rr[arm])), 4),
                     "recall_at_10": round(float(np.mean(ranks <= 10)), 4),
                     "empty_queries": int((np.asarray(masked(lexical, keep).getnnz(axis=1)) == 0).sum())}
        print(f"  MRR {arms[arm]['mrr']:.4f}  R@10 {arms[arm]['recall_at_10']:.4f}", flush=True)

    print("paired bootstrap over leakage groups", flush=True)
    everything = np.ones(len(songs), dtype=bool)
    pairs = [(arm, "full") for arm in arms if arms[arm]["defined"]]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, everything, pairs)
    for c in contrasts:
        arms[c["system"]]["difference_from_full"] = c["mrr_difference"]
        arms[c["system"]]["ci95"] = c["ci95"]

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "anatomy of the lexical identity signal by feature class",
        "question": ("which classes of character n-gram carry the identity the lexical "
                     "space has and the semantic space loses"),
        "queries": len(songs), "labels": label_count,
        "full_model": {"mrr": round(full_mrr, 4), "features": int(lexical.shape[1]),
                       "reproduces_builder": True},
        "classes": {
            "definitions": {
                "script_*": "script of the n-gram's characters, whitespace ignored",
                "length_n": "characters in the n-gram including any space",
                "boundary": "contains a space: the analyser folds newlines and whitespace "
                            "runs to one space, so the n-gram reaches across a line break "
                            "or between Latin words",
                "function_only": f"Han n-gram made entirely of the {FUNCTION_CHARACTERS} most "
                                 "frequent Han characters in the query corpus",
                "han_with_a_less_common_character": "Han n-gram with at least one character "
                                                    "outside that set",
                "digit_or_punctuation": "contains a digit, punctuation or symbol",
            },
            "most_frequent_han_characters": "".join(
                c for c, _ in character_counts.most_common(FUNCTION_CHARACTERS)),
            "inventory": inventory,
            "tfidf_mass_share": mass,
            "correct_label_score_share": shares,
        },
        "arms": arms,
        "bootstrap": {"design": "2000 replicates, seed 20260825, leakage groups resampled with "
                                "replacement, each (group, label) component weighted one; "
                                "difference of each arm from the full model"},
        "reading": ("share says where the score mass sits; remove says how much the model "
                    "loses without a class; only says how far a class alone gets. A class "
                    "whose removal costs little but which alone gets far is redundant with "
                    "the rest; a class whose removal costs much is where the signal lives"),
        "privacy": "aggregate only; no n-gram is published except the hundred most common "
                   "single characters",
    }
    (out_dir / "lexical_identity_anatomy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'lexical_identity_anatomy.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
