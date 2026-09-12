#!/usr/bin/env python3
"""Where inside BGE-M3 does identity sit? The same protocol at every layer.

Every semantic number so far read one vector per chunk: the final layer's [CLS] state,
which BGE-M3 was trained to make useful for retrieval by meaning. Probing studies of
BERT-style encoders find surface and lexical information strongest in the lower layers and
more abstract, semantic information higher up (Jawahar, Sagot and Seddah 2019; Tenney, Das
and Pavlick 2019), and contextual representations grow more anisotropic with depth
(Ethayarajh 2019). If the identity signal of this corpus is a matter of word use, it should
be most accessible low in the stack and fade toward the retrieval head -- and the gap
between the raw and the whitened final layer should be the part of that fading a linear
map can undo. This file measures the curve instead of assuming it.

Stage 1 (GPU) runs the pinned checkpoint once over every chunk with all hidden states and
keeps, per layer, the [CLS] state and the attention-masked mean of the token states
(float16, private). Its first check: the final layer's [CLS], normalised, must reproduce
the recorded v3 embedding run chunk by chunk.

Stage 2 (CPU) scores each layer and pooling under the unchanged protocol -- song = mean of
its normalised chunk vectors, leave-group-out label profiles, cosine rank among the 226 --
raw and after within-author whitening cross-fitted over the five leakage-group folds, as in
identity_probe_v2. Its second check: the final layer's [CLS] must reproduce the recorded
cosine and whitened MRRs. It also reports, per layer, the mean cosine between random chunk
pairs (anisotropy) and the within-author share of song-level variance.

    python src/layerwise_identity_probe_v3.py --private-root <ni-k> [--stage 1|2|both]
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
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs, corpus_content_sha256  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MAX_LENGTH = 2048          # the recorded embedding run's max_length
TOKEN_BUDGET = 16384       # tokens per batch; batches are length-sorted
LAYERS = 25                # embedding output + 24 transformer layers
POOLINGS = ("cls", "mean")
RECORDED = {"cosine": ("three_spaces.json", ("systems", "semantic", "mrr")),
            "within_author_whitening": ("identity_probe.json", ("systems", "within_author_whitening", "mrr"))}


def private_dir_of(private_root: Path) -> Path:
    return private_root / "work" / "private-layerwise-v3"


# ------------------------------------------------------------------ stage 1
def stage_one(private_root: Path, rows, vectors) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("stage 1 needs the CUDA device the recorded embedding run used (fp16)")
    out = private_dir_of(private_root)
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out / "contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous.get("corpus_content_sha256") != V3_CONTENT_SHA256:
            raise SystemExit("the layer store was written for another corpus build; delete it and rerun stage 1")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to("cuda").half().eval()
    texts = [r["cleaned_text"] for r in rows]
    lengths = [len(tokenizer(t, truncation=True, max_length=MAX_LENGTH)["input_ids"]) for t in texts]
    order = np.argsort(lengths, kind="stable")
    stores = {p: np.lib.format.open_memmap(out / f"{p}.npy", mode="w+", dtype=np.float16,
                                          shape=(len(rows), LAYERS, 1024)) for p in POOLINGS}
    started, done, start = time.time(), 0, 0
    with torch.no_grad():
        while start < len(order):
            longest = lengths[order[min(start + 64, len(order)) - 1]]
            size = int(max(1, min(64, TOKEN_BUDGET // max(longest, 1))))
            batch = order[start:start + size]
            enc = tokenizer([texts[i] for i in batch], padding=True, truncation=True, max_length=MAX_LENGTH,
                            return_tensors="pt").to("cuda")
            hidden = model(**enc, output_hidden_states=True).hidden_states
            if len(hidden) != LAYERS:
                raise SystemExit(f"expected {LAYERS} hidden states, got {len(hidden)}")
            # one layer at a time in float32: a float16 sum over 2,048 tokens overflows on
            # XLM-R's outlier dimensions, and a stacked float32 copy of every layer would not fit
            mask = enc["attention_mask"][:, :, None].float()
            count = mask.sum(dim=1)
            cls = torch.stack([h[:, 0, :].float() for h in hidden], dim=1)
            mean = torch.stack([(h.float() * mask).sum(dim=1) / count for h in hidden], dim=1)
            if not (torch.isfinite(cls).all() and torch.isfinite(mean).all()):
                raise SystemExit("a hidden state is not finite under fp16; the layer store would be corrupt")
            stores["cls"][batch] = cls.cpu().numpy().astype(np.float16)
            stores["mean"][batch] = mean.cpu().numpy().astype(np.float16)
            del hidden, mask, count, cls, mean
            done += len(batch)
            start += size
            if done % 2048 < size:
                print(f"    {done:,} / {len(order):,}  {(time.time() - started) / 60:.1f} min", flush=True)
    for store in stores.values():
        store.flush()

    # check 1: the final layer's [CLS], normalised, is the recorded embedding run
    final = stores["cls"][:, LAYERS - 1, :].astype(np.float32)
    final /= np.linalg.norm(final, axis=1, keepdims=True)
    recorded = vectors.astype(np.float32)
    recorded /= np.linalg.norm(recorded, axis=1, keepdims=True)
    agreement = np.sum(final * recorded, axis=1)
    check = {"min_cosine": round(float(agreement.min()), 6), "median_cosine": round(float(np.median(agreement)), 6),
             "chunks_below_0.999": int((agreement < 0.999).sum())}
    print(f"  final-layer [CLS] against the recorded run: {check}", flush=True)
    contract = {"model": MODEL_ID, "revision": MODEL_REVISION, "precision": "fp16", "max_length": MAX_LENGTH,
                "layers": LAYERS, "poolings": list(POOLINGS), "chunks": len(rows),
                "corpus_content_sha256": corpus_content_sha256(rows),
                "check_final_cls_against_recorded_run": check,
                "minutes": round((time.time() - started) / 60, 1),
                "warning": "private; hidden states of copyrighted lyric text. Never commit."}
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if check["median_cosine"] < 0.999:
        raise SystemExit("the final layer does not reproduce the recorded embedding run; stage 2 would score "
                         "a different space")


# ------------------------------------------------------------------ stage 2
def stage_two(private_root: Path, out_dir: Path, rows, vectors) -> int:
    store = private_dir_of(private_root)
    contract = json.loads((store / "contract.json").read_text(encoding="utf-8"))
    if contract["corpus_content_sha256"] != V3_CONTENT_SHA256:
        raise SystemExit("the layer store was written for another corpus build")
    if contract["check_final_cls_against_recorded_run"]["median_cosine"] < 0.999:
        raise SystemExit("the layer store failed its stage 1 check")

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
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)
    ordered_chunks = [sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"])) for s in songs]
    song_of_chunk = np.concatenate([[k] * len(c) for k, c in enumerate(ordered_chunks)])
    chunk_rows = np.concatenate(ordered_chunks)
    everything = np.arange(len(songs))
    pair_rng = np.random.default_rng(SEED)
    pair_a = pair_rng.integers(0, len(rows), size=20000)
    pair_b = pair_rng.integers(0, len(rows), size=20000)

    stores = {p: np.load(store / f"{p}.npy", mmap_mode="r") for p in POOLINGS}
    rr, systems = {}, {}
    for pooling in POOLINGS:
        for layer in range(LAYERS):
            chunk = stores[pooling][:, layer, :].astype(np.float32)
            chunk /= np.maximum(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12)
            anisotropy = float(np.mean(np.sum(chunk[pair_a] * chunk[pair_b], axis=1)))
            acc = np.zeros((len(songs), chunk.shape[1]))
            np.add.at(acc, song_of_chunk, chunk[chunk_rows])
            song_vec = unit_rows(acc / np.bincount(song_of_chunk)[:, None])
            scores = {"cosine": dense_leave_group_out(song_vec, label_index, group_ids, weights, label_count,
                                                       everything),
                      "within_author_whitening": np.zeros((len(songs), label_count))}
            within_share = []
            for k in range(FOLDS):
                train = fold != k
                queries = np.flatnonzero(~train)
                mean, matrix, info = fit_transform("within_author_whitening", song_vec[train], label_index[train],
                                                   weights[train], np.random.default_rng(SEED + k))
                projected = unit_rows((song_vec - mean) @ matrix.T)
                scores["within_author_whitening"][queries] = dense_leave_group_out(
                    projected, label_index, group_ids, weights, label_count, queries)
                within_share.append(info["within_author_share_of_variance"])
            name = f"{pooling}_layer_{layer:02d}"
            entry = {"pooling": pooling, "layer": layer, "anisotropy_mean_pair_cosine": round(anisotropy, 4),
                     "within_author_share_of_variance": round(float(np.mean(within_share)), 4)}
            for transform, s in scores.items():
                ranks = v1.rank_system(s.astype(np.float32), label_index)[0].astype(np.int64)
                rr[f"{name}/{transform}"] = 1.0 / ranks
                entry[transform] = {"mrr": round(float(np.mean(1.0 / ranks)), 4),
                                    "recall_at_10": round(float(np.mean(ranks <= 10)), 4)}
            systems[name] = entry
            print(f"  {name}  cosine {entry['cosine']['mrr']:.4f}  whitened {entry['within_author_whitening']['mrr']:.4f}  "
                  f"anisotropy {entry['anisotropy_mean_pair_cosine']:.3f}  within share "
                  f"{entry['within_author_share_of_variance']:.3f}", flush=True)
            del chunk

    # check 2: the final layer's [CLS] reproduces the recorded results
    final = systems[f"cls_layer_{LAYERS - 1:02d}"]
    checks = {}
    for transform, (file_name, path) in RECORDED.items():
        recorded = json.loads((out_dir / file_name).read_text(encoding="utf-8"))
        if recorded.get("corpus", {}).get("content_sha256") not in (None, V3_CONTENT_SHA256):
            raise SystemExit(f"{file_name} is from another corpus build")
        value = recorded
        for key in path:
            value = value[key]
        checks[transform] = {"recorded": value, "recomputed": final[transform]["mrr"],
                             "gap": round(abs(final[transform]["mrr"] - value), 4)}
    print(f"  final-layer [CLS] against the recorded results: {checks}", flush=True)
    if any(c["gap"] > 0.002 for c in checks.values()):
        raise SystemExit("the final layer does not reproduce the recorded MRRs")

    reference = f"cls_layer_{LAYERS - 1:02d}"
    pairs = []
    for pooling in POOLINGS:
        for layer in range(LAYERS):
            name = f"{pooling}_layer_{layer:02d}"
            for transform in ("cosine", "within_author_whitening"):
                if name != reference:
                    pairs.append((f"{name}/{transform}", f"{reference}/{transform}"))
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "identity under the unchanged protocol at every layer of BGE-M3",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count,
                   "groups": len(order)},
        "design": {"model": f"{MODEL_ID}@{MODEL_REVISION}", "precision": "fp16", "max_length": MAX_LENGTH,
                   "layers": "0 is the embedding output, 24 the final layer",
                   "poolings": {"cls": "the first token's state", "mean": "attention-masked mean of token states"},
                   "song_vector": "mean of the song's normalised chunk vectors, normalised",
                   "whitening": "within-author, Ledoit-Wolf shrunk, cross-fitted over the five leakage-group folds",
                   "anisotropy": "mean cosine of 20,000 random chunk pairs (seeded)",
                   "checks": {"final_cls_against_recorded_run": contract["check_final_cls_against_recorded_run"],
                              "final_cls_against_recorded_results": checks},
                   "caution": "the layer with the highest MRR is chosen after seeing the results; read the curve, "
                              "not the maximum",
                   "references": ["Jawahar, Sagot and Seddah 2019", "Tenney, Das and Pavlick 2019",
                                  "Ethayarajh 2019"]},
        "systems": systems,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with replacement, "
                                       "each (group, label) component weighted one; every layer against the final "
                                       "[CLS] under the same transform", "contrasts": contrasts},
        "privacy": "aggregate only; hidden states are private",
    }
    (out_dir / "layerwise_identity_probe.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'layerwise_identity_probe.json'}")
    return 0


def build(private_root: Path, out_dir: Path, stage: str) -> int:
    print("loading corpus v3", flush=True)
    rows, vectors, state = load_v3(private_root)
    if state["vectors"] != "v3":
        raise SystemExit("the layer probe needs the v3 embedding run")
    if stage in ("1", "both"):
        print("stage 1: hidden states of every chunk", flush=True)
        stage_one(private_root, rows, vectors)
    if stage in ("2", "both"):
        print("stage 2: identity by layer", flush=True)
        stage_two(private_root, out_dir, rows, vectors)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir, args.stage)


if __name__ == "__main__":
    sys.exit(main())
