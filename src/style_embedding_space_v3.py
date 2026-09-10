#!/usr/bin/env python3
"""A purpose-built style embedding as a fourth space: does a model trained to ignore
content find the label better than the word space or the whitened semantic space?

mStyleDistance (Patel et al. 2025; StyleDistance, NAACL 2025) is an xlm-roberta-base
encoder trained by contrastive learning on LLM-written parallel sentences that keep the
content and change one of about forty style features, in nine languages besides English.
It is the cleanest public instance of the thing the fine-tuning here tried to build --
a representation of how a text is written rather than what it says -- and it never saw
this corpus. This file embeds every chunk with it, takes the song mean, and scores the
space under the unchanged protocol, alone, whitened, and fused with the word space.

    python src/style_embedding_space_v3.py --private-root <ni-k> [--device cuda|cpu]
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
MODEL_ID = "StyleDistance/mstyledistance"
MODEL_REVISION = "d66ed25e48225a503b21a65bc804caf06c886f96"   # pinned after the first local load, 2026-09-10


def embed_chunks(rows, private_dir: Path, device: str) -> np.ndarray:
    cache = private_dir / "mstyledistance_chunk_vectors.npy"
    contract = private_dir / "mstyledistance_contract.json"
    if cache.is_file() and contract.is_file():
        c = json.loads(contract.read_text(encoding="utf-8"))
        if c.get("corpus_content_sha256") == V3_CONTENT_SHA256:
            print("  cached style vectors found", flush=True)
            return np.load(cache)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID, device=device, revision=MODEL_REVISION)
    revision = MODEL_REVISION
    texts = [r["cleaned_text"] for r in rows]
    started = time.time()
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
                           convert_to_numpy=True).astype(np.float32)
    private_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    contract.write_text(json.dumps({"model": MODEL_ID, "revision": revision, "device": device,
                                    "max_seq_length": model.max_seq_length, "chunks": len(rows),
                                    "dimension": int(vectors.shape[1]), "corpus_content_sha256": V3_CONTENT_SHA256,
                                    "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
                                    "minutes": round((time.time() - started) / 60, 1),
                                    "warning": "private; vectors of copyrighted text. Never commit."}, indent=2), encoding="utf-8")
    print(f"  embedded {len(rows):,} chunks in {(time.time() - started) / 60:.1f} min on {device}", flush=True)
    return vectors


def build(private_root: Path, out_dir: Path, device: str) -> int:
    import jieba
    jieba.setLogLevel(60)
    print("loading corpus v3", flush=True)
    rows, semantic_vectors, _ = load_v3(private_root)
    private_dir = private_root / "work" / "private-style-embedding-v3"
    style_chunks = embed_chunks(rows, private_dir, device)
    contract = json.loads((private_dir / "mstyledistance_contract.json").read_text(encoding="utf-8"))

    chunks_by_song, label_by_song, components_by_song, documents, semantic_centroids = build_songs(rows, semantic_vectors)
    _, _, _, _, style_centroids = build_songs(rows, style_chunks)
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
    semantic = v1.l2_normalize_dense(np.stack([semantic_centroids[s] for s in songs])).astype(np.float64)
    style = v1.l2_normalize_dense(np.stack([style_centroids[s] for s in songs])).astype(np.float64)
    everything = np.arange(len(songs))
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    scores = {}
    scores["style_cosine"] = dense_leave_group_out(style, label_index, group_ids, weights, label_count, everything)
    scores["semantic_cosine"] = dense_leave_group_out(semantic, label_index, group_ids, weights, label_count, everything)
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    for space_name, space in (("style", style), ("semantic", semantic)):
        for transform in ("total_whitening", "within_author_whitening"):
            out = np.zeros((len(songs), label_count))
            for k in range(FOLDS):
                train_mask = fold != k
                queries = np.flatnonzero(~train_mask)
                mean, matrix, _ = fit_transform(transform, space[train_mask], label_index[train_mask], weights[train_mask],
                                                np.random.default_rng(SEED + k))
                out[queries] = dense_leave_group_out(unit_rows((space - mean) @ matrix.T), label_index, group_ids, weights,
                                                     label_count, queries)
            scores[f"{space_name}_{transform}"] = out
    print("  word space", flush=True)
    word_matrix, _ = fit_words([" ".join(segment(documents[s])) for s in songs])
    scores["words"] = v1.score_leave_group_out(semantic.astype(np.float32), word_matrix, label_index, group_ids,
                                               label_count).lexical.astype(np.float64)
    z = v1.zscore_rows
    scores["fusion_style_words"] = (z(scores["style_within_author_whitening"]) + z(scores["words"])) / 2.0
    scores["fusion_semantic_words"] = (z(scores["semantic_within_author_whitening"]) + z(scores["words"])) / 2.0
    scores["fusion_style_semantic_words"] = (z(scores["style_within_author_whitening"])
                                             + z(scores["semantic_within_author_whitening"]) + z(scores["words"])) / 3.0

    ranks = {name: v1.rank_system(s.astype(np.float32), label_index)[0].astype(np.int64) for name, s in scores.items()}
    rr = {name: 1.0 / r for name, r in ranks.items()}
    report = {}
    for name, r in ranks.items():
        report[name] = {"mrr": round(float(np.mean(1.0 / r)), 4), "recall_at_1": round(float(np.mean(r <= 1)), 4),
                        "recall_at_10": round(float(np.mean(r <= 10)), 4)}
        print(f"  {name:36s} MRR {report[name]['mrr']:.4f}  R@1 {report[name]['recall_at_1']:.4f}  R@10 {report[name]['recall_at_10']:.4f}", flush=True)
    pairs = [("style_cosine", "semantic_cosine"), ("style_within_author_whitening", "style_cosine"),
             ("style_within_author_whitening", "semantic_within_author_whitening"), ("words", "style_within_author_whitening"),
             ("fusion_style_words", "words"), ("fusion_style_semantic_words", "fusion_semantic_words")]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "a content-independent style embedding (mStyleDistance) as a fourth space under the protocol",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count, "groups": len(order)},
        "style_model": {k: contract[k] for k in ("model", "revision", "max_seq_length", "dimension", "vectors_sha256")},
        "design": {"song_vector": "mean of the chunk embeddings, unit length, as for BGE-M3",
                   "whitening": "fold-wise as in identity_probe (five leakage-group folds, Ledoit-Wolf, per-(group, label) weights)",
                   "fusion": "z-score mean of score rows",
                   "reference": "Patel et al. 2025, StyleDistance / mStyleDistance: contrastive training on LLM-written "
                                "parallel sentences that keep content and change one style feature"},
        "systems": report,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with replacement, "
                                       "each (group, label) component weighted one", "contrasts": contrasts},
        "privacy": "aggregate only",
    }
    (out_dir / "style_embedding_space.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'style_embedding_space.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir, args.device)


if __name__ == "__main__":
    sys.exit(main())
