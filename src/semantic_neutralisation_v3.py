#!/usr/bin/env python3
"""Is the identity BGE-M3 keeps carried by names? The semantic arm of the neutralisation test.

The lexical neutralisation arms stripped the 605-surface catalogue -- places, people,
brands, slang, English words -- and the character space barely moved. The semantic space
could not be tested the same way without re-embedding the stripped text. Corpus v3 and its
neutralised variant have now both been embedded under the recorded configuration, so the
question can be asked of BGE-M3 directly: on the same songs, same labels, same leakage
groups, how much does the semantic system lose when the names are gone -- raw, and after
the label-free and within-author whitening that recover most of its identity signal?

Two vector sets, one protocol, one variable.

    python src/semantic_neutralisation_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs, corpus_content_sha256  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3, sha256_file  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
NEUTRALISED_DIGEST = "f71b30b70bb1b1b53fdf1f2cf0d3ddf244b60b2c065439573f49e25d2438975f"
TRANSFORMS = ("none", "total_whitening", "within_author_whitening")


def load_neutralised(private_root: Path, rows_v3):
    table = private_root / "work" / "private-cleaned-corpus-v3-neutralised" / "cleaned_lyric_chunks_v3_neutralised.csv"
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    if corpus_content_sha256(rows) != NEUTRALISED_DIGEST:
        raise SystemExit("the neutralised table is not the recorded one")
    if [(r["song_id"], r["chunk_id"]) for r in rows] != [(r["song_id"], r["chunk_id"]) for r in rows_v3]:
        raise SystemExit("the neutralised table does not align chunk for chunk with v3")
    embed_dir = private_root / "work" / "private-cleaned-corpus-v3-neutralised-embeddings"
    vectors = np.load(embed_dir / "cleaned_lyric_chunks_v3_neutralised_bge_m3_embeddings.npy")
    contract = json.loads((embed_dir / "cleaned_lyric_chunks_v3_neutralised_embedding_contract.json").read_text(encoding="utf-8"))
    if contract["corpus_content_sha256"] != NEUTRALISED_DIGEST or vectors.shape[0] != len(rows):
        raise SystemExit("the neutralised embedding run does not match its table")
    return rows, vectors, contract


def score_space(dense, label_index, group_ids, weights, label_count, fold):
    scores = {name: np.zeros((len(label_index), label_count)) for name in TRANSFORMS}
    for k in range(FOLDS):
        train = fold != k
        queries = np.flatnonzero(~train)
        for name in TRANSFORMS:
            mean, matrix, _ = fit_transform(name, dense[train], label_index[train], weights[train],
                                            np.random.default_rng(SEED + k))
            scores[name][queries] = dense_leave_group_out(unit_rows((dense - mean) @ matrix.T), label_index,
                                                          group_ids, weights, label_count, queries)
    return {name: v1.rank_system(s.astype(np.float32), label_index)[0].astype(np.int64) for name, s in scores.items()}


def build(private_root: Path, out_dir: Path) -> int:
    print("loading corpus v3 and both vector sets", flush=True)
    rows, vectors, state = load_v3(private_root)
    if state["vectors"] != "v3":
        raise SystemExit("the v3 embedding run is required")
    rows_n, vectors_n, contract_n = load_neutralised(private_root, rows)
    chunks_by_song, label_by_song, components_by_song, documents, centroids = build_songs(rows, vectors)
    _, _, _, _, centroids_n = build_songs(rows_n, vectors_n)
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
    dense = v1.l2_normalize_dense(np.stack([centroids[s] for s in songs])).astype(np.float64)
    dense_n = v1.l2_normalize_dense(np.stack([centroids_n[s] for s in songs])).astype(np.float64)
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    ranks = {}
    for variant, matrix in (("original", dense), ("neutralised", dense_n)):
        print(f"scoring the {variant} vectors under the three transforms", flush=True)
        for name, r in score_space(matrix, label_index, group_ids, weights, label_count, fold).items():
            ranks[f"{variant}_{name}"] = r
            print(f"  {variant:12s} {name:26s} MRR {float(np.mean(1.0 / r)):.4f}", flush=True)
    rr = {name: 1.0 / r for name, r in ranks.items()}
    report = {name: {"mrr": round(float(np.mean(v)), 4), "recall_at_1": round(float(np.mean(ranks[name] <= 1)), 4),
                     "recall_at_10": round(float(np.mean(ranks[name] <= 10)), 4)} for name, v in rr.items()}
    pairs = [(f"neutralised_{t}", f"original_{t}") for t in TRANSFORMS]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)
    cosine_shift = float(np.mean(np.sum(dense * dense_n, axis=1)))

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "the semantic arm of the named-reference neutralisation, on corpus v3",
        "corpus": {"v3_content_sha256": V3_CONTENT_SHA256, "neutralised_content_sha256": NEUTRALISED_DIGEST,
                   "queries": len(songs), "labels": label_count, "groups": len(order)},
        "vectors": {"original": state["sha256"], "neutralised": contract_n["embeddings_sha256"],
                    "configuration": contract_n["configuration"]},
        "mean_cosine_between_original_and_neutralised_song_vectors": round(cosine_shift, 4),
        "systems": report,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with "
                                       "replacement, each (group, label) component weighted one; "
                                       "neutralised minus original under each transform",
                             "contrasts": contrasts},
        "privacy": "aggregate only",
    }
    (out_dir / "semantic_neutralisation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'semantic_neutralisation.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
