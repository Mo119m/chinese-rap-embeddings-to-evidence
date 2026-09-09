#!/usr/bin/env python3
"""Corpus v3: corpus v2 with metadata lines removed from the lyric text (PD-003).

An audit of corpus v2 on 2026-09-09 found that the lyric text still carries what the
source pages carried around the lyrics: production credits (作词 / 作曲 / mix / master /
prod. by ...) in 5.8% of chunks, section markers ([Verse], 副歌) in 3.0%, HTML remnants
(`<RAP>`, `&apos;`) in 0.5%, a handful of URLs, and 295 lines of nothing but digits or
dashes. None of that is a lyric, and the identity experiments were run over it. Protocol
MB-001 already holds a frozen, self-tested detector for credit blocks, written after the
author's instruction that such blocks be removed from the input rather than labelled; it
was never applied to the corpus because it was sequenced behind PD-002. PD-002 has landed.
This file applies it.

What is removed, line by line, and counted:

  metadata_block     a line the MB-001 detector labels (credit roles, copyright marks,
                     organisations, contacts, handles, annotation lines, and the name-only
                     runs its rule 5 joins to them), classified over the whole song so a
                     block that straddles two chunks is seen whole
  section_header     [Verse], Hook, 副歌, 作词 ... as the written-rhyme task defines them
  role_line          the English-form credit the detector was not written for: `mix by
                     yoken`, `Master. 荨麻疹`, `Recorded By 小老虎`; and 作词：X shapes
  credit_tag_line    a line carrying a bracketed production tag -- `胸弟(Prod.4Harry)`,
                     `深渊（Lil Andy，Prod.Vessels）` -- which is a copied title line and
                     where featured artists' names enter the text
  stray_title        a line that is the song's own title, character for character after
                     punctuation and case are dropped (finding MB-001-F1)
  html               tags stripped and entities unescaped; a line left empty is dropped
  url                any line carrying http(s):// or www.
  digits_only        a line of nothing but digits, dashes, dots, colons and spaces
  whitespace         runs of spaces collapsed, lines stripped; empty lines dropped

What is deliberately kept: bracketed short spans -- (yeah), （出人头地） -- because they are
backing vocals and ad-libs, which are lyrics; and a song's mention of its own artist,
because a rapper naming themself is a lyric, however cheap an identity signal. The latter
gets its own control elsewhere.

Chunks left with no text are dropped and counted. Every other column of corpus v2 is
carried unchanged, so leakage groups, components and weights still resolve. A content
digest over the v3 rows is recorded so downstream builders can refuse any other table.

Outputs: the private v3 table under the private root, and a public aggregate audit
(counts by rule, nothing per song) under results/cleaned-corpus-v3/.

    python src/build_cleaned_corpus_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import build_chinese_rap_written_rhyme_v1 as wr  # noqa: E402
from build_downstream_retrieval_v2 import CORPUS_CONTENT_SHA256, corpus_content_sha256  # noqa: E402
from detect_metadata_blocks import DETECTOR_VERSION, classify  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)
OUT_DIR = ROOT / "results" / "cleaned-corpus-v3"
VERSION = "chinese-rap-cleaned-corpus-v3/1.0.0"
TAG = re.compile(r"<[^>]{1,40}>")
URL = re.compile(r"https?://|www\.", re.I)
DIGITS_ONLY = re.compile(r"^[\d\s.:\-—–_=*]+$")
SPACES = re.compile(r"[ \t　]{2,}")
# Two shapes the MB-001 detector was not written for, found in its residue on 2026-09-09:
# the English-form role line -- `mix by yoken`, `Master. 荨麻疹`, `Recorded By 小老虎` --
# and the bracketed production tag that a copied title line carries -- `胸弟(Prod.4Harry)`,
# `深渊（Lil Andy，Prod.Vessels）`. The second is also where featured artists' names enter
# the text. A stray copy of the song's own title is the third.
ROLE_LINE = re.compile(
    r"^\s*(?:(?:mix|master|mixing|mastering|mixed|mastered|prod\.?|produced|producer|beat|beats|"
    r"arranged|arrangement|composed|composer|written|lyrics|lyric|vocal|vocals|recorded|recording|"
    r"engineer|engineered|artwork|cover|design|designed|photo|video|director|directed)\b"
    r"[^\n]{0,40}?(?:by(?![A-Za-z])|:|：|\.|/)|"
    r"(?:作词|作曲|编曲|混音|母带|制作人|制作|监制|录音|和声|封面|出品|发行|词曲|词|曲)\s*[:：/／.．]\s*\S)",
    re.I)
CREDIT_TAG = re.compile(r"[\(（\[【][^\)）\]】]*\b(?:prod|beat|mix|master|remix|feat|ft)\b\.?[^\)）\]】]*[\)）\]】]", re.I)


def normalise_title(value: str) -> str:
    return re.sub(r"[\s\-_·．。,，、/\\()（）\[\]【】!！?？'\"“”‘’:：.]+", "", value).lower()


def clean_line(line: str, counts: Counter, title_key: str) -> str | None:
    if TAG.search(line) or "&" in line and html.unescape(line) != line:
        line = html.unescape(TAG.sub("", line))
        counts["html"] += 1
    line = SPACES.sub(" ", line).strip()
    if not line:
        counts["empty_after_cleaning"] += 1
        return None
    if URL.search(line):
        counts["url"] += 1
        return None
    if DIGITS_ONLY.match(line):
        counts["digits_only"] += 1
        return None
    if wr.is_header_line(line):
        counts["section_header"] += 1
        return None
    if ROLE_LINE.match(line):
        counts["role_line"] += 1
        return None
    if CREDIT_TAG.search(line):
        counts["credit_tag_line"] += 1
        return None
    if title_key and len(title_key) >= 2 and normalise_title(line) == title_key:
        counts["stray_title"] += 1
        return None
    return line


def build(private_root: Path, out_dir: Path) -> int:
    corpus = private_root / "work" / "private-repaired-corpus-v2" / "repaired_lyric_chunks_v2.csv"
    rows = list(csv.DictReader(corpus.open(encoding="utf-8")))
    digest = corpus_content_sha256(rows)
    if digest != CORPUS_CONTENT_SHA256:
        raise SystemExit(f"corpus digest {digest} is not the published corpus v2")
    print(f"{len(rows):,} chunks over {len({r['song_id'] for r in rows}):,} songs", flush=True)

    # classify metadata blocks over the whole song, then map lines back to chunks
    by_song: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_song[row["song_id"]].append(index)
    counts: Counter = Counter()
    rule_counts: Counter = Counter()
    chunks_touched = 0
    songs_touched = set()
    kept_rows = []
    lines_before = lines_after = 0
    for song, indices in by_song.items():
        indices.sort(key=lambda i: int(rows[i]["source_order"]))
        chunk_lines = [rows[i]["cleaned_text"].split("\n") for i in indices]
        labels = classify("\n".join("\n".join(lines) for lines in chunk_lines))
        cursor = 0
        song_changed = False
        title_key = normalise_title(rows[indices[0]]["song_title"])
        for i, lines in zip(indices, chunk_lines):
            new_lines = []
            changed = False
            for line in lines:
                lines_before += 1
                verdict = labels[cursor]["rule"]
                cursor += 1
                if verdict is not None:
                    counts["metadata_block"] += 1
                    rule_counts[verdict] += 1
                    changed = True
                    continue
                cleaned = clean_line(line, counts, title_key)
                if cleaned is None:
                    changed = True
                    continue
                if cleaned != line:
                    changed = True
                new_lines.append(cleaned)
            lines_after += len(new_lines)
            if changed:
                chunks_touched += 1
                song_changed = True
            if not new_lines:
                counts["chunks_dropped_empty"] += 1
                continue
            row = dict(rows[i])
            row["cleaned_text"] = "\n".join(new_lines)
            kept_rows.append(row)
        if song_changed:
            songs_touched.add(song)
    assert cursor == len(labels) if by_song else True

    out_private = private_root / "work" / "private-cleaned-corpus-v3"
    out_private.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(kept_rows)
    table = out_private / "cleaned_lyric_chunks_v3.csv"
    table.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    v3_digest = corpus_content_sha256(kept_rows)
    songs_after = len({r["song_id"] for r in kept_rows})

    audit = {
        "artifact_id": "chinese-rap-cleaned-corpus-v3",
        "version": VERSION,
        "derived_from": {"corpus_v2_content_sha256": CORPUS_CONTENT_SHA256,
                         "chunks": len(rows), "songs": len(by_song)},
        "content_sha256": v3_digest,
        "file_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "detector": {"protocol": "MB-001", "version": DETECTOR_VERSION,
                     "scope": "classified over the whole song in source order, so a block "
                              "straddling two chunks is seen whole"},
        "lines": {"before": lines_before, "after": lines_after,
                  "removed": lines_before - lines_after,
                  "removed_share": round((lines_before - lines_after) / lines_before, 4)},
        "removed_by_rule": dict(counts),
        "metadata_block_by_detector_rule": dict(rule_counts),
        "chunks": {"before": len(rows), "after": len(kept_rows), "touched": chunks_touched,
                   "touched_share": round(chunks_touched / len(rows), 4)},
        "songs": {"before": len(by_song), "after": songs_after, "touched": len(songs_touched),
                  "touched_share": round(len(songs_touched) / len(by_song), 4)},
        "kept_on_purpose": ["bracketed short spans such as (yeah) -- backing vocals and ad-libs",
                            "a song's mention of its own artist -- a lyric, controlled elsewhere"],
        "privacy": "aggregate counts only; the v3 table is private",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    (out_private / "cleaned_corpus_v3_contract.json").write_text(
        json.dumps({**audit, "warning": "private; lyric text. Never commit or publish."},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"lines {lines_before:,} -> {lines_after:,} ({audit['lines']['removed_share']:.2%} removed)")
    print("  by rule: " + ", ".join(f"{k} {v:,}" for k, v in counts.most_common()))
    print("  detector rules: " + ", ".join(f"{k} {v:,}" for k, v in rule_counts.most_common()))
    print(f"chunks {len(rows):,} -> {len(kept_rows):,} ({chunks_touched:,} touched); songs {len(by_song):,} -> {songs_after:,}")
    print(f"v3 content sha256 {v3_digest}")
    print(f"wrote {table} and {out_dir / 'analysis_summary.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
