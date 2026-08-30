"""Pilot-informed metadata-block CANDIDATE detector for protocol MB-001.

This proposes candidates; it does not establish what is metadata. Its rules were written
from a pilot look at real credit blocks and then measured against them, so agreement with
those examples is not accuracy, and this tool reports **no recall, precision or
F-measure**. An independently constructed gold set is required before any such figure
exists. Read `methods/METADATA_BLOCK_AUDIT_PROTOCOL.md` first.

What it does guarantee: the rules live here as code with self-tests, no model and no
lookup against the entity inventory is involved, and two runs over the same bytes classify
the same lines.

    python tools/detect_metadata_blocks.py --self-test
    python tools/detect_metadata_blocks.py --corpus DIR --dedup-state applied --out summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
DETECTOR_VERSION = "MB-001.1"

ROLES = (
    "作词", "作曲", "编曲", "制作人", "制作", "监制", "出品人", "出品方", "出品", "发行方",
    "发行", "录音师", "录音", "混音师", "混音", "母带工程师", "母带", "和声", "和音", "吉他",
    "贝斯", "鼓", "键盘", "弦乐", "合唱团", "合唱", "伴唱", "封面设计", "封面", "视觉",
    "企划", "统筹", "宣推", "营销", "版权", "词", "曲",
    "OP", "SP", "Producer", "Prod", "Beat", "Mixing", "Mix", "Mastering", "Master",
    "Arranged", "Arranger", "Composer", "Lyricist", "Lyrics", "Vocal", "Backing",
    "Artwork", "Design", "Label", "Publishing",
)
SEPARATOR = r"(?:：|:|／|/|[ \t]{2,})"
ROLE_ALTERNATION = "|".join(re.escape(role) for role in ROLES)
# A credit label may chain several role terms and a bilingual gloss before its separator
# ("企划统筹A&R COORDINATOR：..."), so the role run continues over further role terms and
# Latin label text. Han that is not itself a role term ends the run, which keeps lyric
# lines that merely begin with a role word (曲终人散..., 词穷了...) out.
ROLE_LINE = re.compile(
    r"^\s*(?:[\(（\[【][^）\)\]】]{1,12}[\)）\]】]\s*)?"
    r"(?:" + ROLE_ALTERNATION + r")"
    r"(?:" + ROLE_ALTERNATION + r"|[A-Za-z0-9&.\- ]|/(?=[A-Za-z]))*"
    r"\s*" + SEPARATOR)
COPYRIGHT_MARKS = ("©", "℗", "版权所有", "保留所有权利", "All Rights Reserved")
# Sample attribution names the borrowed work and its performer. The blinded adjudication
# labelled one such line as lyric text, which author review caught: it is a credit.
SAMPLE_LINE = re.compile(
    r"^\s*(?:sample[ds]?\s+from|samples?\s*[:：]|interpolat(?:es|ion of)|"
    r"采样自|采样于|采样[:：]|取样自)", re.IGNORECASE)
FORBID_AFTER = re.compile(r"未经许可.{0,20}(使用|复制|翻录)")
ORG_TERMS = ("有限公司", "传媒", "文化传播", "唱片", "工作室", "娱乐", "影业", "集团",
             "合唱团", "乐团", "学院", "大学", "剧院", "协会", "基金会", "厂牌",
             "卫生中心", "研究中心", "艺术中心")
# A contact address or an @handle marks a credit line: no lyric line in this corpus
# carries one, and several personnel lines carry nothing else a rule could key on.
CONTACT = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+|https?://|www\.")
HANDLE = re.compile(r"@[A-Za-z0-9一-鿿._\-]+")
SENTENCE_FINAL = ("。", "？", "！")
BRACKETED = re.compile(r"^\s*(?:（([^）]*)）|\(([^)]*)\)|【([^】]*)】|\[([^\]]*)\])\s*$")
# A fully bracketed line is a structure marker ONLY if its content is one, and only when
# short. Rap lyrics in this corpus carry bracketed ad-libs and glosses on their own lines
# constantly, and an earlier version classified one such gloss, sitting between two
# English lyric lines, as an annotation.
ANNOTATION_WORDS = (
    "副歌", "主歌", "间奏", "前奏", "尾奏", "过门", "桥段", "重复", "反复", "念白", "独白",
    "女声", "男声", "合声", "和声", "齐唱", "轮唱", "伴唱", "说唱", "口白", "旁白",
    "hook", "verse", "chorus", "bridge", "intro", "outro", "pre-chorus", "refrain",
    "interlude", "skit", "spoken", "ad-lib", "adlib", "repeat", "x2", "x3", "x4",
)
FINAL_PARTICLE = re.compile(r"[啊呀哦呢吧了嘛哈]\s*[）\)\]】]\s*$")
VERB_HINT = re.compile(r"[是在有要会去来把被让使得]")
NAMEISH = re.compile(r"[一-鿿A-Za-z0-9 .·\-_/&+]+")


def rule_hit(line: str) -> str | None:
    """The rule (1-4) that classifies this line as metadata, or None for lyric text."""
    stripped = line.strip()
    if not stripped:
        return None
    if ROLE_LINE.match(stripped):
        return "role_prefix"
    if SAMPLE_LINE.match(stripped):
        return "sample_attribution"
    if any(mark in stripped for mark in COPYRIGHT_MARKS) or FORBID_AFTER.search(stripped):
        return "copyright"
    sentence_like = any(mark in stripped for mark in SENTENCE_FINAL)
    if any(term in stripped for term in ORG_TERMS) and not sentence_like:
        # a contact address lifts the length cap: an organisation line that carries one is
        # a credit however long its registered name runs
        if re.search(SEPARATOR, stripped) or CONTACT.search(stripped) or len(stripped) <= 30:
            return "organisation"
    if CONTACT.search(stripped) and not sentence_like and not VERB_HINT.search(stripped):
        return "contact"
    # a slash-separated list of short name-like segments, optionally with @handles, is a
    # personnel or ensemble list; two or more separators and no verb keep lyric lines out
    if not sentence_like and not VERB_HINT.search(stripped):
        segments = [part.strip() for part in re.split(r"[/／、]", stripped) if part.strip()]
        if len(segments) >= 3 and all(len(part) <= 20 for part in segments):
            return "name_list"
        if len(segments) >= 2 and all(len(part) <= 20 for part in segments) \
                and HANDLE.search(stripped):
            return "name_list"
    bracket = BRACKETED.match(stripped)
    if bracket and not FINAL_PARTICLE.search(stripped):
        inner = next((group for group in bracket.groups() if group is not None), "").strip()
        lowered = inner.lower()
        if len(inner) <= 12 and any(word in lowered for word in ANNOTATION_WORDS):
            return "bracketed_annotation"
    return None


def name_only(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 20 or VERB_HINT.search(stripped):
        return False
    match = NAMEISH.fullmatch(stripped)
    return bool(match)


def classify(text: str) -> list[dict]:
    """Every line, labelled; then rule-5 block extension over the labels."""
    lines = text.split("\n")
    labels: list[dict] = [{"line": index, "rule": rule_hit(line), "text_length": len(line)}
                          for index, line in enumerate(lines)]
    # Rule 5a, bridge: a run of name-only lines flanked on BOTH sides by rule hits is a
    # personnel list between two credit labels. A threshold-only rule missed this shape —
    # a coordinator line, two bare names, then a publisher line gave each name a single
    # neighbouring hit and neither joined.
    index = 0
    while index < len(labels):
        if labels[index]["rule"] is not None or not name_only(lines[index]):
            index += 1
            continue
        end = index
        while end + 1 < len(labels) and labels[end + 1]["rule"] is None and name_only(lines[end + 1]):
            end += 1
        if index - 1 >= 0 and labels[index - 1]["rule"] is not None \
                and end + 1 < len(labels) and labels[end + 1]["rule"] is not None:
            for cursor in range(index, end + 1):
                labels[cursor]["rule"] = "name_extension"
        index = end + 1

    # Rule 5b, tail: a name-only run adjacent to a block carrying >= 2 rule hits, but ONLY
    # where the run reaches the start or the end of the text. Extending into the middle
    # swallowed the first lyric line after a credits header: a short Latin refrain looks
    # exactly like a name, and only the text boundary separates a trailing personnel list
    # from lyrics that merely follow a credits block.
    index = 0
    while index < len(labels):
        if labels[index]["rule"] is not None or not name_only(lines[index]):
            index += 1
            continue
        end = index
        while end + 1 < len(labels) and labels[end + 1]["rule"] is None and name_only(lines[end + 1]):
            end += 1
        block_rules = 0
        for start, step in ((index - 1, -1), (end + 1, 1)):
            cursor = start
            while 0 <= cursor < len(labels) and labels[cursor]["rule"] is not None:
                if labels[cursor]["rule"] != "name_extension":
                    block_rules += 1
                cursor += step
        if (index == 0 or end == len(labels) - 1) and block_rules >= 2:
            for cursor in range(index, end + 1):
                labels[cursor]["rule"] = "name_extension"
        index = end + 1
    return labels


def blocks_of(labels: list[dict]) -> list[tuple[int, int]]:
    spans, start = [], None
    for label in labels:
        if label["rule"] is not None and start is None:
            start = label["line"]
        elif label["rule"] is None and start is not None:
            spans.append((start, label["line"] - 1))
            start = None
    if start is not None:
        spans.append((start, labels[-1]["line"]))
    return spans


def self_test() -> int:
    failures = []

    def check(label: str, condition: bool) -> None:
        print(("  ok   " if condition else "  FAIL ") + label)
        if not condition:
            failures.append(label)

    check("a role-prefix credit line is metadata", rule_hit("作曲：某某") == "role_prefix")
    check("a bracketed role tag still matches", rule_hit("（编曲）编曲: 某某") == "role_prefix")
    check("an English producer credit is metadata", rule_hit("Producer: Somebody") == "role_prefix")
    check("a copyright line is metadata", rule_hit("© 2020 保留所有权利") == "copyright")
    check("a company registration line is metadata", rule_hit("某某文化传播有限公司") == "organisation")
    check("an org term inside a long sentence is NOT metadata",
          rule_hit("我把整个集团的故事写进这一段很长很长的歌词里面然后继续唱下去。") is None)
    check("a bracketed section marker is metadata", rule_hit("【副歌】") == "bracketed_annotation")
    check("a bracketed exclamation with a final particle is lyric",
          rule_hit("（就是这样啊）") is None)
    check("a plain lyric line is lyric", rule_hit("今天的天气很好我们出去走走") is None)
    check("the word 词 must be a prefixed role, not a substring",
          rule_hit("歌词里的故事没有人听") is None)

    check("a chained credit label reaches its separator past a bilingual gloss",
          rule_hit("企划统筹A&R COORDINATOR：甲乙丙") == "role_prefix")
    check("a sample attribution line is metadata",
          rule_hit("sample from 甲乙 《丙丁戊》") == "sample_attribution")
    check("a Chinese sample attribution is metadata",
          rule_hit("采样自 甲乙丙") == "sample_attribution")

    # the real shape author review surfaced: a coordinator label, bare personnel names,
    # a publisher label, then the song
    block = classify("企划统筹A&R COORDINATOR：甲乙丙\n丁戊\n己庚/辛壬\n"
                     "OP/SP：癸子文化传媒有限公司\nBaby Baby\n我想要问你")
    check("every line of a real credits header is caught",
          [entry["rule"] is not None for entry in block[:4]] == [True] * 4)
    check("and the first lyric line after it is NOT swallowed",
          block[4]["rule"] is None and block[5]["rule"] is None)

    labels = classify("作曲：甲\n作词：乙\n丙丙\n今天我们出去走走")
    check("a lone name-only line before lyrics stays lyric, the conservative reading",
          labels[2]["rule"] is None)
    check("the lyric line after the block stays lyric", labels[3]["rule"] is None)
    tail = classify("今天我们出去走走\n作曲：甲\n作词：乙\n丙丙\n丁丁")
    check("but a name run that ENDS the text joins its credits block",
          [entry["rule"] for entry in tail[3:]] == ["name_extension"] * 2)
    labels = classify("作曲：甲\n丙丙\n今天我们出去走走")
    check("one rule hit does not pull a name-only neighbour in", labels[1]["rule"] is None)
    check("blocks are maximal runs, split by a genuine lyric line",
          blocks_of(classify("作曲：甲\n作词：乙\n今天我们出去走走\n©x")) == [(0, 1), (3, 3)])
    check("a name-like line BETWEEN two blocks merges them, as rule 5 specifies",
          blocks_of(classify("作曲：甲\n作词：乙\n丙丙\n©x")) == [(0, 3)])

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nall detector self-tests passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--corpus", type=Path, help="directory of *.txt song-unit files (private)")
    parser.add_argument("--dedup-state", choices=("applied", "not-applied"),
                        help="whether PD-002 deduplication has been applied to this input")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not (args.corpus and args.dedup_state and args.out):
        parser.error("--corpus, --dedup-state and --out are required unless --self-test")
    if args.out.resolve().is_relative_to(Path(__file__).resolve().parent.parent) \
            and args.dedup_state == "not-applied":
        parser.error("a pre-dedup summary is provisional and must not be written into the repository")

    totals = {"song_units": 0, "lines": 0, "characters": 0,
              "metadata_lines": 0, "metadata_characters": 0, "metadata_blocks": 0,
              "rule_hits": {}, "block_position": {"leading": 0, "trailing": 0, "interior": 0}}
    corpus_hash = hashlib.sha256()
    for path in sorted(args.corpus.glob("*.txt")):
        payload = path.read_bytes()
        corpus_hash.update(hashlib.sha256(payload).digest())
        text = payload.decode("utf-8", "replace")
        labels = classify(text)
        totals["song_units"] += 1
        totals["lines"] += len(labels)
        totals["characters"] += sum(l["text_length"] for l in labels)
        for label in labels:
            if label["rule"]:
                totals["metadata_lines"] += 1
                totals["metadata_characters"] += label["text_length"]
                totals["rule_hits"][label["rule"]] = totals["rule_hits"].get(label["rule"], 0) + 1
        spans = blocks_of(labels)
        totals["metadata_blocks"] += len(spans)
        for start, end in spans:
            position = ("leading" if start == 0 else
                        "trailing" if end == len(labels) - 1 else "interior")
            totals["block_position"][position] += 1

    summary = {
        "protocol": "MB-001",
        "detector_version": DETECTOR_VERSION,
        "detector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "corpus_content_sha256": corpus_hash.hexdigest(),
        "dedup_state": args.dedup_state,
        "status": "final" if args.dedup_state == "applied" else "provisional_pre_dedup",
        "totals": totals,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=1))
    print(f"summary written to {args.out} ({summary['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
