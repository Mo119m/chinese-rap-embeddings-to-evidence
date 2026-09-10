#!/usr/bin/env python3
"""Loading corpus v3, the cleaned table, for the downstream builders.

Corpus v3 (PD-003) is corpus v2 with metadata lines removed from the lyric text and 733
empty chunks dropped. Its rows carry the same columns and identifiers, so leakage groups
and components resolve exactly as before. This module loads the table, refuses any table
whose content digest differs from the recorded one, and attaches a vector set:

  * the v3 embeddings from the recorded local run over the current v3 text (the loader
    refuses a run computed on an earlier v3 build); or
  * as an interim for lexical-only work, the v2 vectors of the same (song, chunk) --
    computed on the uncleaned text, so they must never be reported as a v3 semantic
    result. The loader marks which it returned.

    rows, vectors, contract = load_v3(private_root)
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from build_downstream_retrieval_v2 import corpus_content_sha256

csv.field_size_limit(10 ** 9)

V3_CONTENT_SHA256 = "cd51bf699ddbb42b313a334ca4a13534f166feb60ed488c6e3ea81fd7b164ca4"  # v3/1.3.0
EXPECTED_CHUNKS = 24237
EXPECTED_SONGS = 7379


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v3(private_root: Path, allow_interim_v2_vectors: bool = False):
    table = private_root / "work" / "private-cleaned-corpus-v3" / "cleaned_lyric_chunks_v3.csv"
    if not table.is_file():
        raise SystemExit(f"missing private input: {table}")
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    digest = corpus_content_sha256(rows)
    if digest != V3_CONTENT_SHA256:
        raise SystemExit(f"corpus v3 digest {digest} is not the recorded {V3_CONTENT_SHA256}")
    if len(rows) != EXPECTED_CHUNKS or len({r["song_id"] for r in rows}) != EXPECTED_SONGS:
        raise SystemExit("corpus v3 has an unexpected number of chunks or songs")

    embed_dir = private_root / "work" / "private-cleaned-corpus-v3-embeddings"
    vectors_path = embed_dir / "cleaned_lyric_chunks_v3_bge_m3_embeddings.npy"
    if vectors_path.is_file():
        vectors = np.load(vectors_path)
        row_map = list(csv.DictReader((embed_dir / "cleaned_lyric_chunks_v3_embedding_row_map.csv").open(encoding="utf-8")))
        if len(row_map) != len(rows) or vectors.shape[0] != len(rows):
            raise SystemExit("the v3 embedding run does not match the v3 table")
        for index, (mapped, row) in enumerate(zip(row_map, rows)):
            if mapped["song_id"] != row["song_id"] or int(mapped["chunk_id"]) != int(row["chunk_id"]):
                raise SystemExit(f"v3 row map diverges from the table at row {index}")
        contract = json.loads((embed_dir / "cleaned_lyric_chunks_v3_embedding_contract.json").read_text(encoding="utf-8"))
        if contract.get("corpus_content_sha256") != V3_CONTENT_SHA256:
            raise SystemExit("the v3 embedding run was computed on a different v3 build "
                             f"({contract.get('corpus_content_sha256')}); re-embed the current table")
        return rows, vectors, {"vectors": "v3", "sha256": sha256_file(vectors_path), "contract": contract}

    if not allow_interim_v2_vectors:
        raise SystemExit("no v3 embedding run yet; pass allow_interim_v2_vectors=True for lexical-only work")
    # interim: the v2 vectors of the same (song, chunk); text differs, so semantic numbers
    # from these are not v3 results
    v2_dir = private_root / "work" / "private-repaired-corpus-v2-embeddings"
    v2_vectors = np.load(v2_dir / "repaired_corpus_v2_bge_m3_embeddings.npy")
    v2_map = list(csv.DictReader((v2_dir / "repaired_corpus_v2_embedding_row_map.csv").open(encoding="utf-8")))
    position = {(m["song_id"], int(m["chunk_id"])): int(m["row_index"]) for m in v2_map}
    index = np.asarray([position[(r["song_id"], int(r["chunk_id"]))] for r in rows])
    return rows, v2_vectors[index], {"vectors": "v2-interim", "warning": "computed on the uncleaned text"}
