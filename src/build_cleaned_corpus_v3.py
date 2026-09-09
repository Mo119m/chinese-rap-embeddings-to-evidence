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
                     block that straddles two chunks is seen whole. Amendment 1.1.0: the
                     detector's four fuzzy rules (organisation, name_list, name_extension,
                     bracketed_annotation) are honoured only inside an anchored block or a
                     long header/footer, or when the line alone is unmistakably a credit;
                     see FUZZY_RULES below. The rest are lyrics and are kept, counted per
                     rule under detector_hits_kept_as_lyric
  section_header     [Verse], Hook, 副歌, 作词 ... as the written-rhyme task defines them
  role_line          the English-form credit the detector was not written for: `mix by
                     yoken`, `Master. 荨麻疹`, `Recorded By 小老虎`; and 作词：X shapes
  credit_tag_line    a line carrying a bracketed production tag -- `胸弟(Prod.4Harry)`,
                     `深渊（Lil Andy，Prod.Vessels）` -- which is a copied title line and
                     where featured artists' names enter the text
  stray_title        a line that is the song's own title, character for character after
                     punctuation and case are dropped (finding MB-001-F1), AND that sits
                     next to a credit or header line -- i.e. inside the copied page header.
                     Amendment 1.1.0 (2026-09-09): build 1.0.0 removed every title-equal
                     line, 3,023 of them; sampling the deleted lines showed that about 2,800
                     were hook lines repeated inside the verses (a chorus is very often the
                     title), so a title-equal line with lyric lines on both sides is now
                     kept and counted as title_line_kept_as_lyric
  track_list_line    a numbered album-page entry -- 3.海河摇摆客(Swing Boyz), 1、小青龙 -- with
                     the number followed by a non-digit (amendment 1.2.0, from the author's
                     reading of the sample)
  show_transcript    a song that is a whole television episode (three or more X战队“Y”
                     team lines, fourteen performers) is dropped whole; see SHOW_TEAM_LINE.
                     Lines in scripts the corpus does not model
                     (Uyghur, Tibetan, Korean, Mongolian: about 1,300 lines over some fifty
                     labels) are lyrics and are KEPT; like names, they are a cheap identity
                     signal, and they get a control arm in the experiments instead
  html               real HTML tags (br, p, span, ...) stripped and entities unescaped
  bracket_marker     a whole line inside angle brackets -- `<RAP>`, `< Chorus >`, `<谢帝>`
                     (a cypher's rapper marker) -- is a marker, not a lyric, and is dropped;
                     a bracketed whole line carrying sentence punctuation is a lyric quoted
                     in brackets, unwrapped and kept (bracket_line_kept_as_lyric). Amendment
                     1.1.0: build 1.0.0 stripped every `<...>` span, which also cut quoted
                     song titles out of the middle of lyric lines; an inline non-HTML span
                     is now left alone
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
import detect_metadata_blocks as dm  # noqa: E402
from detect_metadata_blocks import DETECTOR_VERSION, classify  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)
OUT_DIR = ROOT / "results" / "cleaned-corpus-v3"
VERSION = "chinese-rap-cleaned-corpus-v3/1.2.0"
# rules whose removed line marks a credit / header block; a title-equal line touching one
# of these is a copied title, a title-equal line touching lyrics is a hook
BLOCK_RULES = frozenset({"metadata_block", "section_header", "role_line", "credit_tag_line", "url",
                         "bracket_marker", "track_list_line"})
# MB-001's four fuzzy rules hit lyric lines when applied line by line across a whole song:
# a verse about 大学 or 唱片 (organisation), a bar written as 词、词、词 (name_list), a short
# refrain beside a credit block (name_extension), a bracketed lyric that happens to contain
# 说唱 (bracketed_annotation). Sampled on 2026-09-09, about half of the organisation hits
# and a fifth of the name-list hits were lyrics. Amendment 1.1.0: a fuzzy hit is removed
# only when it is anchored -- its block holds a line no lyric could produce (a role prefix,
# a contact, a copyright mark, a sample attribution, or one of this file's own credit
# rules), or the block is a header or footer of three or more hits -- or when the line by
# itself is unmistakably a credit under the strict shapes in unmistakable_credit().
# Otherwise it goes back through the ordinary line cleaning and is counted as kept.
FUZZY_RULES = frozenset({"organisation", "name_list", "name_extension", "bracketed_annotation"})
ANCHOR_RULES = frozenset({"role_prefix", "contact", "copyright", "sample_attribution",
                          "role_line", "credit_tag_line", "url", "section_header", "bracket_marker"})
STRICT_ORG = re.compile(r"有限公司|文化传播|传媒|工作室|影业|合唱团|乐团|基金会|协会|研究中心|艺术中心|卫生中心|出品|发行")
SENTENCE_MARK = re.compile(r"[，。？！?!]")
HAN = re.compile(r"[一-鿿]")
HAN_NAME = re.compile(r"[一-鿿]{2,4}")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’.\-]*")
ANNOTATION_ASCII = re.compile(r"hook|verse|chorus|bridge|intro|outro|skit|refrain|interlude|spoken|ad-?lib|repeat|\d", re.I)


def short_name(token: str) -> bool:
    """A personal-name-sized token: at most four Han characters or two Latin words."""
    return bool(token) and len(HAN.findall(token)) <= 4 and len(LATIN_WORD.findall(token)) <= 2 \
        and bool(dm.NAMEISH.fullmatch(token))


def unmistakable_credit(rule: str, line: str) -> bool:
    """Would this fuzzy hit be a credit even standing alone between lyric lines?"""
    text = line.strip()
    if rule == "organisation":
        return bool(STRICT_ORG.search(text)) and not SENTENCE_MARK.search(text)
    if rule == "name_list":
        if SENTENCE_MARK.search(text):
            return False
        if dm.HANDLE.search(text) or text.endswith((":", "：")):
            return True  # a performer marker -- 法老/小精灵/万妮达： -- names a block, it is not sung
        if "、" in text:
            return False  # 忠、义、孝 is a bar; a thanks list written with 、 is left with it
        segments = [s.strip() for s in re.split(r"[/／]", text) if s.strip()]
        if len(segments) < 3 or not all(2 <= len(s) <= 20 and dm.NAMEISH.fullmatch(s) for s in segments):
            return False  # 上 /拳 下/脚 左/打 右/踢 is a bar of single characters
        if " / " not in text:
            return True
        # bars are also written as 词 / 词 / 词; only short name-shaped segments are a list
        return all(short_name(s) for s in segments)
    if rule == "name_extension":
        han = len(HAN.findall(text))
        words = len(LATIN_WORD.findall(text))
        if "/" in text or "@" in text or STRICT_ORG.search(text) or (han <= 4 and words <= 2):
            return True
        tokens = text.split()  # 孙冕峰 高端端 -- a row of Han names; To the top is not one
        return len(tokens) >= 2 and all(HAN_NAME.fullmatch(t) for t in tokens)
    if rule == "bracketed_annotation":
        inner = text.strip("()（）【】[] ")
        return len(inner) <= 6 or bool(ANNOTATION_ASCII.search(inner))
    return True


def gate_fuzzy_hits(entries: list[dict], counts: Counter, rule_counts: Counter,
                    kept_as_lyric: Counter) -> None:
    """Second pass over one song: send un-anchored fuzzy detector hits back to the lyrics."""
    substantive = [j for j, e in enumerate(entries) if e["rule"] != "empty_after_cleaning"]
    if not substantive:
        return
    blocks: list[list[int]] = []
    current: list[int] = []
    for j in substantive:
        if entries[j]["rule"] is not None:
            current.append(j)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    for block in blocks:
        anchored = any(entries[j]["rule"] in ANCHOR_RULES or entries[j]["detector"] in ANCHOR_RULES
                       for j in block)
        header = (block[0] == substantive[0] or block[-1] == substantive[-1]) and len(block) >= 3
        for j in block:
            entry = entries[j]
            if entry["detector"] not in FUZZY_RULES:
                continue
            if anchored or header or unmistakable_credit(entry["detector"], entry["original"]):
                continue
            counts["metadata_block"] -= 1
            rule_counts[entry["detector"]] -= 1
            kept_as_lyric[entry["detector"]] += 1
            rule, cleaned = clean_line(entry["original"], counts)
            entry.update({"rule": rule, "text": cleaned, "detector": None})
HTML_TAG = re.compile(r"</?(?:br|p|div|span|i|b|u|em|strong|font|a|img|hr)\b[^<>]{0,40}>", re.I)
# a track-list entry copied from an album page -- 3.海河摇摆客(Swing Boyz), 08.扎得紧(breakd&kane),
# 1、小青龙 -- found by the author in the sample (amendment 1.2.0). The number must be
# followed by something other than a digit, so 37.2熟悉的温度 and 21、22到我的23 stay lyrics;
# the rest is a short title without sentence punctuation.
TRACK_LIST = re.compile(r"^\s*\d{1,2}\s*[.、．]\s*(?!\d)[^，。？！?!]{1,30}$")
# a whole television episode filed under one rapper -- 02 福克斯 / 吴亦凡张震岳热狗战队“梦想” /
# ... fourteen performers' verses in one "song" (amendment 1.2.0, found by the author). The
# labelled rapper's own verse is a fraction of it and cannot be cut out reliably, so a song
# carrying three or more team-assignment lines is dropped whole and counted.
SHOW_TEAM_LINE = re.compile(r"^\s*\S{2,20}战队[“\"][^”\"]{1,6}[”\"]\s*$")
SHOW_TEAM_LINES_TO_DROP_SONG = 3
BRACKET_LINE = re.compile(r"^\s*<\s*([^<>]{1,60}?)\s*>\s*$")
SENTENCE_PUNCT = re.compile(r"[，。？！?!；;、…]")
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
    r"engineer|engineered|artwork|cover|design|designed|photo|video|director|directed|programming)\b"
    r"[^\n]{0,40}?(?:by(?![A-Za-z])|:|：|\.|/)|"
    r"(?:作词|作曲|编曲|混音|母带|制作人|制作|监制|录音|和声|封面|出品|发行|词曲|词|曲)\s*[:：/／.．]\s*\S)",
    re.I)
CREDIT_TAG = re.compile(r"[\(（\[【][^\)）\]】]*\b(?:prod|beat|mix|master|remix|feat|ft)\b\.?[^\)）\]】]*[\)）\]】]", re.I)


def normalise_title(value: str) -> str:
    return re.sub(r"[\s\-_·．。,，、/\\()（）\[\]【】!！?？'\"“”‘’:：.]+", "", value).lower()


def clean_line(line: str, counts: Counter) -> tuple[str | None, str | None]:
    """(rule that removed the line, None) or (None, the cleaned line)."""
    if HTML_TAG.search(line) or "&" in line and html.unescape(line) != line:
        line = html.unescape(HTML_TAG.sub("", line))
        counts["html"] += 1
    line = SPACES.sub(" ", line).strip()
    bracketed = BRACKET_LINE.match(line)
    if bracketed and SENTENCE_PUNCT.search(bracketed.group(1)):
        line = bracketed.group(1)
        counts["bracket_line_kept_as_lyric"] += 1
    elif bracketed:
        counts["bracket_marker"] += 1
        return "bracket_marker", None
    if not line:
        rule = "empty_after_cleaning"
    elif URL.search(line):
        rule = "url"
    elif DIGITS_ONLY.match(line):
        rule = "digits_only"
    elif wr.is_header_line(line):
        rule = "section_header"
    elif ROLE_LINE.match(line):
        rule = "role_line"
    elif CREDIT_TAG.search(line):
        rule = "credit_tag_line"
    elif TRACK_LIST.match(line):
        rule = "track_list_line"
    else:
        return None, line
    counts[rule] += 1
    return rule, None


def mark_stray_titles(entries: list[dict], title_key: str, counts: Counter) -> None:
    """Second pass over one song: a surviving title-equal line is a copied title only when
    the nearest surviving-or-removed neighbour on either side is a credit / header line."""
    if not title_key or len(title_key) < 2:
        return
    substantive = [j for j, e in enumerate(entries) if e["rule"] != "empty_after_cleaning"]
    for position, j in enumerate(substantive):
        entry = entries[j]
        if entry["rule"] is not None or normalise_title(entry["text"]) != title_key:
            continue
        before = entries[substantive[position - 1]]["rule"] if position > 0 else None
        after = (entries[substantive[position + 1]]["rule"]
                 if position + 1 < len(substantive) else None)
        if before in BLOCK_RULES or after in BLOCK_RULES:
            entry["rule"], entry["text"] = "stray_title", None
            counts["stray_title"] += 1
        else:
            counts["title_line_kept_as_lyric"] += 1


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
    kept_as_lyric: Counter = Counter()
    chunks_touched = 0
    songs_touched = set()
    kept_rows = []
    lines_before = lines_after = 0
    songs_dropped_show_transcript = 0
    for song, indices in by_song.items():
        indices.sort(key=lambda i: int(rows[i]["source_order"]))
        chunk_lines = [rows[i]["cleaned_text"].split("\n") for i in indices]
        team_lines = sum(1 for lines in chunk_lines for line in lines if SHOW_TEAM_LINE.match(line))
        if team_lines >= SHOW_TEAM_LINES_TO_DROP_SONG:
            songs_dropped_show_transcript += 1
            songs_touched.add(song)
            lines_before += sum(len(lines) for lines in chunk_lines)
            counts["show_transcript_line"] += sum(len(lines) for lines in chunk_lines)
            counts["chunks_dropped_show_transcript"] += len(indices)
            continue
        labels = classify("\n".join("\n".join(lines) for lines in chunk_lines))
        cursor = 0
        song_changed = False
        title_key = normalise_title(rows[indices[0]]["song_title"])
        # first pass: every line of the song in order, with the rule that removed it
        entries: list[dict] = []
        for i, lines in zip(indices, chunk_lines):
            for line in lines:
                lines_before += 1
                verdict = labels[cursor]["rule"]
                cursor += 1
                if verdict is not None:
                    counts["metadata_block"] += 1
                    rule_counts[verdict] += 1
                    entries.append({"chunk": i, "original": line, "rule": "metadata_block",
                                    "text": None, "detector": verdict})
                    continue
                rule, cleaned = clean_line(line, counts)
                entries.append({"chunk": i, "original": line, "rule": rule, "text": cleaned,
                                "detector": None})
        # second pass: fuzzy detector hits stay only when anchored; title-equal lines are
        # copied titles only inside a credit block
        gate_fuzzy_hits(entries, counts, rule_counts, kept_as_lyric)
        mark_stray_titles(entries, title_key, counts)
        for i in indices:
            own = [e for e in entries if e["chunk"] == i]
            new_lines = [e["text"] for e in own if e["rule"] is None]
            changed = any(e["rule"] is not None or e["text"] != e["original"] for e in own)
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
        "detector_hits_kept_as_lyric": dict(kept_as_lyric),
        "chunks": {"before": len(rows), "after": len(kept_rows), "touched": chunks_touched,
                   "touched_share": round(chunks_touched / len(rows), 4)},
        "songs": {"before": len(by_song), "after": songs_after, "touched": len(songs_touched),
                  "touched_share": round(len(songs_touched) / len(by_song), 4),
                  "dropped_as_show_transcript": songs_dropped_show_transcript},
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
    print("  detector hits kept as lyrics: " + ", ".join(f"{k} {v:,}" for k, v in kept_as_lyric.most_common()))
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
