#!/usr/bin/env python3
"""Re-embed the PD-002 repaired corpus under one recorded BGE-M3 configuration.

Written to run on Colab against a Drive mount, and unchanged on a local GPU.

Why all 25,026 chunks and not just the 2,898 restored ones. The stored vectors cover the
frozen 22,128-chunk snapshot. `requirements.txt` records that "the historical BGE
device/CUDA/use_fp16 state was not recorded", and BGE-M3 in fp16 and fp32 does not return
the same vectors. Embedding only the restored chunks would put two precisions in one
retrieval space, which is a confound that no downstream metric could separate afterwards.

Step 1 therefore tries to *recover* the historical configuration before replacing anything:
it re-embeds a sample of already-embedded chunks under each candidate configuration and
compares against the stored vectors. If one matches, the unrecorded device state stops being
unrecorded, and the new vectors are commensurable with the published results. If none
matches, that is itself the finding, and the run continues under a configuration that is
recorded from the start.

The model weights are pinned by SHA-256 in the existing contract. A different BGE-M3
snapshot produces different vectors, so the pin is checked before anything is embedded.

    python tools/colab_embed_corpus_v2.py --corpus <repaired_lyric_chunks_v2.csv> \
        --existing-embeddings <canonical_bge_m3_embeddings_v1.npy> \
        --existing-row-map <canonical_embedding_row_map_v1.csv> \
        --existing-contract <canonical_semantic_embedding_contract_v1.json> \
        --out <output directory>

Outputs are private: vectors, a row map keyed by song and chunk, and a contract recording
the configuration actually used. None of it may be committed or published.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

CONTRACT_VERSION = "chinese-rap-repaired-corpus-embeddings-v2/1.0.0"
RECOVERY_SAMPLE = 64
# cosine this close counts as the same configuration; fp16 vs fp32 differs far more
MATCH_TOLERANCE = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_configurations(has_cuda: bool) -> list[dict[str, object]]:
    """The configurations the historical run plausibly used, most likely first.

    `batch_size: 2` in the stored contract is the tell: that is a CPU or a very small card.
    Batch size does not change the vectors for a correct implementation, so it is not varied
    here; device and precision do.
    """
    options = [{"device": "cpu", "use_fp16": False}]
    if has_cuda:
        options = [{"device": "cuda", "use_fp16": True},
                   {"device": "cuda", "use_fp16": False}] + options
    return options


def recover_configuration(model_path: Path, texts: list[str], stored, configurations):
    """Find which configuration reproduces the stored vectors, if any."""
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

    results = []
    for configuration in configurations:
        try:
            model = BGEM3FlagModel(str(model_path), use_fp16=bool(configuration["use_fp16"]),
                                   devices=configuration["device"])
            fresh = model.encode(texts, batch_size=2, max_length=2048)["dense_vecs"]
        except Exception as error:                      # noqa: BLE001
            results.append({**configuration, "error": str(error)[:200]})
            continue
        fresh = np.asarray(fresh, dtype=np.float64)
        reference = np.asarray(stored, dtype=np.float64)
        cosine = (fresh * reference).sum(axis=1) / (
            np.linalg.norm(fresh, axis=1) * np.linalg.norm(reference, axis=1))
        worst = float(cosine.min())
        results.append({**configuration, "min_cosine_to_stored": worst,
                        "max_abs_difference": float(np.abs(fresh - reference).max()),
                        "matches": worst >= 1.0 - MATCH_TOLERANCE})
        del model
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--existing-embeddings", type=Path, required=True)
    parser.add_argument("--existing-row-map", type=Path, required=True)
    parser.add_argument("--existing-contract", type=Path, required=True)
    parser.add_argument("--model-path", type=Path,
                        help="local BGE-M3 snapshot; downloaded from the Hub if omitted")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--recover-only", action="store_true",
                        help="run the configuration recovery and stop")
    args = parser.parse_args()

    import numpy as np

    contract = json.loads(args.existing_contract.read_text(encoding="utf-8"))
    pinned = contract.get("model", {}).get("pytorch_model_bin_sha256")
    print(f"existing contract pins model weights sha256: {pinned}")

    model_path = args.model_path
    if model_path is None:
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download("BAAI/bge-m3"))
        print(f"downloaded BGE-M3 to {model_path}")

    weights = next((p for name in ("pytorch_model.bin", "model.safetensors")
                    for p in [model_path / name] if p.is_file()), None)
    if weights is None:
        print("could not find model weights to verify against the pin", file=sys.stderr)
        return 2
    actual = sha256_file(weights)
    print(f"  local {weights.name} sha256: {actual}")
    if pinned and actual != pinned:
        print("  WEIGHTS DO NOT MATCH THE PIN. A different BGE-M3 snapshot returns different\n"
              "  vectors, so anything embedded here would not be comparable with the published\n"
              "  results. Resolve this before continuing.", file=sys.stderr)
        if not args.recover_only:
            return 2
    elif pinned:
        print("  matches the pinned weights")

    corpus = load_corpus(args.corpus)
    print(f"corpus v2: {len(corpus):,} chunks")

    stored = np.load(args.existing_embeddings)
    row_map = list(csv.DictReader(args.existing_row_map.open(encoding="utf-8-sig")))
    index_of = {(r["song_id"], r["chunk_id"]): int(r["canonical_row_index"]) for r in row_map}
    print(f"stored vectors: {stored.shape[0]:,} x {stored.shape[1]}")

    # ---------------------------------------------------------------- recover the config
    overlap = [r for r in corpus if (r["song_id"], r["chunk_id"]) in index_of][:RECOVERY_SAMPLE]
    print(f"\nrecovering the historical configuration on {len(overlap)} already-embedded chunks")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if has_cuda else platform.processor()
    except Exception:                                    # noqa: BLE001
        has_cuda, device_name = False, platform.processor()
    print(f"  device available: {'cuda: ' + device_name if has_cuda else 'cpu'}")

    reference = np.stack([stored[index_of[(r["song_id"], r["chunk_id"])]] for r in overlap])
    trials = recover_configuration(model_path, [r["cleaned_text"] for r in overlap],
                                   reference, candidate_configurations(has_cuda))
    for trial in trials:
        if "error" in trial:
            print(f"  {trial['device']:>4} fp16={trial['use_fp16']}: failed -- {trial['error']}")
        else:
            print(f"  {trial['device']:>4} fp16={trial['use_fp16']}: "
                  f"min cosine {trial['min_cosine_to_stored']:.8f}  "
                  f"max abs diff {trial['max_abs_difference']:.2e}  "
                  f"{'MATCH' if trial['matches'] else 'no match'}")

    recovered = next((t for t in trials if t.get("matches")), None)
    if recovered:
        print(f"{chr(10)}  historical configuration identified: device={recovered['device']} "
              f"use_fp16={recovered['use_fp16']} -- the only one inside tolerance")
    elif passing:
        spread = ranked[-1]["max_abs_difference"] - ranked[0]["max_abs_difference"]
        print(f"{chr(10)}  {len(passing)} configurations are ALL inside tolerance, so this "
              "test does not identify which one was used.")
        print(f"  closest: device={best['device']} use_fp16={best['use_fp16']} "
              f"(max abs difference {best['max_abs_difference']:.3e}), "
              f"spread across configurations {spread:.3e}")
        print("  The useful finding is the bound, not the identity:")
        print("  every candidate reproduces the stored vectors this closely, so the")
        print("  unrecorded device state is not a material threat to comparability.")
        print("  Re-embedding the whole corpus in one run removes the question entirely.")
    else:
        best = max((t for t in trials if "min_cosine_to_stored" in t),
                   key=lambda t: t["min_cosine_to_stored"], default=None)
        print("\n  no candidate reproduced the stored vectors."
              f"{f' closest was device={best['device']} fp16={best['use_fp16']} at cosine {best['min_cosine_to_stored']:.6f}.' if best else ''}"
              "\n  That is a finding, not a failure: it means the published vectors cannot be\n"
              "  regenerated from the tracked pipeline, and the whole corpus must be re-embedded\n"
              "  under one recorded configuration for any downstream comparison to be valid.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "configuration_recovery.json").write_text(
        json.dumps({"contract_version": CONTRACT_VERSION,
                    "sample_size": len(overlap),
                    "match_tolerance": MATCH_TOLERANCE,
                    "model_weights_sha256": actual,
                    "model_weights_pin_matched": bool(pinned) and actual == pinned,
                    "device_available": device_name if has_cuda else "cpu",
                    "trials": trials,
                    "recovered": recovered},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {args.out / 'configuration_recovery.json'}")

    if args.recover_only:
        return 0

    # ------------------------------------------------------------------- embed everything
    configuration = recovered or {"device": "cuda" if has_cuda else "cpu",
                                  "use_fp16": bool(has_cuda)}
    print(f"\nembedding all {len(corpus):,} chunks: device={configuration['device']} "
          f"use_fp16={configuration['use_fp16']} batch_size={args.batch_size}")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(str(model_path), use_fp16=bool(configuration["use_fp16"]),
                           devices=configuration["device"])
    vectors = model.encode([r["cleaned_text"] for r in corpus],
                           batch_size=args.batch_size, max_length=2048)["dense_vecs"]
    vectors = np.asarray(vectors)
    print(f"  produced {vectors.shape[0]:,} x {vectors.shape[1]}")

    np.save(args.out / "repaired_corpus_bge_m3_embeddings_v2.npy", vectors)
    with (args.out / "repaired_corpus_embedding_row_map_v2.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_index", "song_id", "chunk_id", "cleaned_text_sha256"])
        for index, record in enumerate(corpus):
            writer.writerow([index, record["song_id"], record["chunk_id"],
                             hashlib.sha256(record["cleaned_text"].encode("utf-8")).hexdigest()])

    (args.out / "repaired_corpus_embedding_contract_v2.json").write_text(
        json.dumps({
            "contract_version": CONTRACT_VERSION,
            "warning": "private; derived from copyrighted lyric text. Never commit or publish.",
            "chunks": len(corpus),
            "dimension": int(vectors.shape[1]),
            "model": {"implementation": "FlagEmbedding.BGEM3FlagModel dense vectors",
                      "weights_sha256": actual,
                      "matched_existing_pin": bool(pinned) and actual == pinned},
            # the field whose absence caused this whole exercise
            "device": configuration["device"],
            "use_fp16": bool(configuration["use_fp16"]),
            "batch_size": args.batch_size,
            "max_length": 2048,
            "historical_configuration_recovered": bool(recovered),
            "commensurable_with_published_vectors": bool(recovered),
            "note": ("Every chunk here was embedded in one run under the configuration above. "
                     "If historical_configuration_recovered is false, these vectors are NOT "
                     "comparable with the published 22,128-vector snapshot and any downstream "
                     "metric must be recomputed end to end rather than compared across the two."),
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
