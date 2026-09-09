#!/usr/bin/env python3
"""Embed a chunk table with BGE-M3 on the local GPU, under the recorded configuration.

The corpus v2 vectors were produced on Colab under one recorded configuration -- pinned
checkpoint, CUDA, fp16, batch 8, max_length 2,048, corpus source order, 512-chunk blocks
cast to float32 -- and the contract refuses any resume under another. This does the same
thing on the local card for corpus v3 and its neutralised variant, so the semantic column
of every v3 result comes from a run whose configuration is written down before it starts.

The table's content digest must match the one recorded in its contract; a CUDA device is
required, for the reason the notebook gives -- fp32 blocks on a CPU runtime would be
indistinguishable on disk from fp16 blocks and would put two precisions in one space.

    python tools/embed_corpus_local_v3.py --corpus <chunks.csv> --expected-digest <sha256> \
        --out <dir> --stem cleaned_lyric_chunks_v3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from build_downstream_retrieval_v2 import corpus_content_sha256  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)
MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
PINNED_WEIGHTS_SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
BLOCK, BATCH, MAX_LENGTH = 512, 8, 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--contract-version", required=True)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this run must not fall back to CPU")
    rows = list(csv.DictReader(args.corpus.open(encoding="utf-8")))
    digest = corpus_content_sha256(rows)
    if digest != args.expected_digest:
        raise SystemExit(f"corpus digest {digest} is not the expected {args.expected_digest}")
    print(f"{len(rows):,} chunks, digest verified", flush=True)

    from huggingface_hub import snapshot_download
    model_path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
    weights = next((model_path / n for n in ("pytorch_model.bin", "model.safetensors") if (model_path / n).is_file()), None)
    weights_sha = sha256_file(weights) if weights else None
    print(f"weights {weights.name if weights else '?'} sha256 {weights_sha}  pin matched: {weights_sha == PINNED_WEIGHTS_SHA256}", flush=True)

    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(str(model_path), use_fp16=True, devices="cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    blocks_dir = args.out / "blocks"
    blocks_dir.mkdir(exist_ok=True)
    n_blocks = (len(rows) + BLOCK - 1) // BLOCK
    started = time.time()
    for b in range(n_blocks):
        target = blocks_dir / f"{b}.npy"
        if target.is_file():
            try:
                if np.load(target).shape[0] == min(BLOCK, len(rows) - b * BLOCK):
                    continue
            except Exception:
                pass
        texts = [r["cleaned_text"] for r in rows[b * BLOCK:(b + 1) * BLOCK]]
        vectors = model.encode(texts, batch_size=BATCH, max_length=MAX_LENGTH)["dense_vecs"]
        np.save(target, np.asarray(vectors, dtype=np.float32))
        if (b + 1) % 5 == 0 or b + 1 == n_blocks:
            print(f"  block {b + 1}/{n_blocks}  {(time.time() - started) / 60:.1f} min", flush=True)
    matrix = np.concatenate([np.load(blocks_dir / f"{b}.npy") for b in range(n_blocks)], axis=0)
    assert matrix.shape[0] == len(rows)
    np.save(args.out / f"{args.stem}_bge_m3_embeddings.npy", matrix)
    with (args.out / f"{args.stem}_embedding_row_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_index", "song_id", "chunk_id", "source_order", "text_component_id",
                         "song_duplicate_group_id", "song_component_weight"])
        for i, r in enumerate(rows):
            writer.writerow([i, r["song_id"], r["chunk_id"], r["source_order"], r["text_component_id"],
                             r["song_duplicate_group_id"], r["song_component_weight"]])
    contract = {
        "contract_version": args.contract_version,
        "warning": "private; derived from copyrighted lyric text. Never commit or publish.",
        "chunks": len(rows), "dimension": int(matrix.shape[1]),
        "corpus_content_sha256": digest,
        "embeddings_sha256": sha256_file(args.out / f"{args.stem}_bge_m3_embeddings.npy"),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": weights_sha,
                  "weights_pin_matched": weights_sha == PINNED_WEIGHTS_SHA256,
                  "implementation": "FlagEmbedding.BGEM3FlagModel dense vectors"},
        "configuration": {"device": "cuda", "gpu": torch.cuda.get_device_name(0), "use_fp16": True,
                          "batch_size": BATCH, "max_length": MAX_LENGTH, "block_size": BLOCK,
                          "order": "corpus source order, not length-sorted",
                          "torch": torch.__version__, "python": platform.python_version()},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out / f"{args.stem}_embedding_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(json.dumps({k: v for k, v in contract.items() if k != "warning"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
