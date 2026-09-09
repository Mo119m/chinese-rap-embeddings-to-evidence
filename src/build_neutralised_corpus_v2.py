#!/usr/bin/env python3
"""Corpus v2 with every named reference and slang term stripped, for re-embedding.

The neutralisation arms removed a lexicon from every song and refit the character space,
and the lexical advantage barely moved. The semantic arm could not be neutralised the same
way, because BGE-M3 vectors are computed once on a GPU: to ask whether the identity that
whitening recovers from BGE-M3 is carried by names either, the stripped text has to be
embedded again. This file writes that text -- corpus v2 with the whole 605-surface catalogue
removed, the same `neutralise` rule the lexical arm used -- as a private chunk table with
the columns the embedding tool reads, plus a content digest so the run can be tied to it.

The output is lyric text and stays private. It goes to the author's Drive for the Colab
run and nowhere else.

    python src/build_neutralised_corpus_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from build_downstream_retrieval_v2 import CORPUS_CONTENT_SHA256, corpus_content_sha256  # noqa: E402
from lexical_identity_decomposition_v2 import neutralise  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)


def build(private_root: Path, lexicon: Path, out_dir: Path, corpus: Path | None = None,
          expected_digest: str = CORPUS_CONTENT_SHA256, stem: str = "repaired_lyric_chunks_v2") -> int:
    corpus = corpus or private_root / "work" / "private-repaired-corpus-v2" / "repaired_lyric_chunks_v2.csv"
    rows = list(csv.DictReader(corpus.open(encoding="utf-8")))
    digest = corpus_content_sha256(rows)
    if digest != expected_digest:
        raise SystemExit(f"corpus digest {digest} is not the expected {expected_digest}")
    surfaces = sorted({r["entity"].strip() for r in csv.DictReader(lexicon.open(encoding="utf-8-sig"))
                       if r.get("entity", "").strip()}, key=len, reverse=True)
    removed = touched = 0
    for row in rows:
        before = row["cleaned_text"]
        after = neutralise(before, surfaces)
        removed += len(before) - len(after)
        touched += after != before
        row["cleaned_text"] = after
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    table = out_dir / f"{stem}_neutralised.csv"
    table.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    neutral_digest = corpus_content_sha256(rows)
    contract = {
        "derived_from_corpus_content_sha256": expected_digest,
        "neutralised_content_sha256": neutral_digest,
        "file_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "lexicon": lexicon.name, "surfaces": len(surfaces),
        "characters_removed": removed, "chunks_touched": touched, "chunks": len(rows),
        "rule": "Latin surfaces as whole words (a Han character is a boundary), Han surfaces "
                "as substrings; the rule the lexical neutralisation arms used",
        "warning": "private; lyric text. Never commit or publish.",
    }
    (out_dir / "neutralised_corpus_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"{len(rows):,} chunks; {removed:,} characters removed from {touched:,} chunks")
    print(f"neutralised content sha256 {neutral_digest}")
    print(f"wrote {table}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, help="defaults to work/lexicon_arm_everything.csv")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--corpus", type=Path, help="a chunk table other than corpus v2, e.g. the cleaned v3")
    parser.add_argument("--expected-digest", default=CORPUS_CONTENT_SHA256,
                        help="content digest the --corpus table must carry")
    parser.add_argument("--stem", default="repaired_lyric_chunks_v2", help="output file stem")
    args = parser.parse_args()
    root = args.private_root.resolve()
    lexicon = args.lexicon or root / "work" / "lexicon_arm_everything.csv"
    out_dir = args.out_dir or root / "work" / "private-repaired-corpus-v2-neutralised"
    return build(root, lexicon, out_dir, corpus=args.corpus, expected_digest=args.expected_digest,
                 stem=args.stem)


if __name__ == "__main__":
    sys.exit(main())
