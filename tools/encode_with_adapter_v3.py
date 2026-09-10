#!/usr/bin/env python3
"""Encode a chunk table with a saved LoRA adapter of train_identity_encoder_v3.

Used to put the catalogue-stripped (neutralised) v3 text through the fine-tuned encoder,
so the tuned space can take the same neutralisation test as the frozen one. Text is
label-masked exactly as in training; vectors are CLS, unit length, in the table's row
order, with a row map so the analysis can align them by (song_id, chunk_id). Private
output: vectors of copyrighted text, never committed.

    python tools/encode_with_adapter_v3.py --corpus <chunks.csv> --expected-digest <sha256> \
        --adapter <private-identity-encoder-v3/lora_fold0> --out <dir> --stem <name>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from build_downstream_retrieval_v2 import corpus_content_sha256  # noqa: E402
from train_identity_encoder_v3 import MODEL_ID, MODEL_REVISION, encode_all, mask_label  # noqa: E402

csv.field_size_limit(10 ** 9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-mask", action="store_true", help="do not mask the label string (training masked it)")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.corpus.open(encoding="utf-8")))
    digest = corpus_content_sha256(rows)
    if digest != args.expected_digest:
        raise SystemExit(f"corpus digest {digest} is not the expected {args.expected_digest}")
    texts = [r["cleaned_text"] if args.no_mask else mask_label(r["cleaned_text"], r["source_credit_label"]) for r in rows]
    print(f"{len(rows):,} chunks; label masking {'off' if args.no_mask else 'on'}", flush=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    base = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device)
    model = PeftModel.from_pretrained(base, str(args.adapter)).to(device)
    adapter_files = sorted(p for p in args.adapter.iterdir() if p.is_file())
    adapter_sha = hashlib.sha256(b"".join(p.read_bytes() for p in adapter_files)).hexdigest()
    print(f"adapter {args.adapter.name} sha256 {adapter_sha[:12]} on {device}", flush=True)

    vectors = encode_all(model, tokenizer, texts, device, args.max_length)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / f"{args.stem}_vectors.npy", vectors.astype(np.float32))
    with (args.out / f"{args.stem}_row_map.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_index", "song_id", "chunk_id"])
        for i, r in enumerate(rows):
            writer.writerow([i, r["song_id"], r["chunk_id"]])
    contract = {"corpus_content_sha256": digest, "chunks": len(rows), "adapter": str(args.adapter.name),
                "adapter_sha256": adapter_sha, "model": f"{MODEL_ID}@{MODEL_REVISION}", "max_length": args.max_length,
                "label_masked": not args.no_mask, "pooling": "CLS, unit length",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "warning": "private; derived from copyrighted lyric text. Never commit or publish."}
    (args.out / f"{args.stem}_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / (args.stem + '_vectors.npy')}  {vectors.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
