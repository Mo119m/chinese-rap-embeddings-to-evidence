#!/usr/bin/env python3
"""Generate the PD-002 duplicate-record review sheet for author adjudication.

PD-002 leaves 46 erased song records that no automatic rule resolves. This tool renders
them as a self-contained review sheet so the author can rule on each one, and writes the
frozen instruction text alongside it so a later ruling can be tied to the exact question
that was asked.

The sheet is blinded in the two ways that matter for this question:

* the rater is not told which record the legacy cleaner erased and which it retained, and
  the two sides of each comparison are presented in a deterministically shuffled order; and
* the rater is not shown the automatic reason code, so the ruling does not simply ratify
  the classifier that put the record in the queue.

The sheet carries lyric text, titles, and source-credit labels. It is a private working
file and must never be written into the public repository or published.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_ID = "PD-002-DUPREV-001"

# Fixed so two runs of this tool put the same pair in the same order. It is a presentation
# seed only; nothing about the ruling depends on its value.
SHUFFLE_SEED = "pd002-duplicate-review-presentation-v1"

INSTRUCTIONS = """背景：这批语料以前用过一条清洗规则——同一个署名标签下，一段歌词文本如果之前出现过，
就只保留第一次那条、后面的删掉。结果有 177 首歌被整首删光了。其中 131 首能自动确认
是同一首歌被导入了两次，剩下这 46 条自动判不了，所以来问你。

你要判断的问题只有一个：下面这两条（有时三四条）记录，是不是同一首录音作品？

注意问的不是"文字一不一样"——机器已经比对过了，每张卡片上面那行灰字就是比对结果。
问的是：文字这样重合，意味着它们是同一首歌吗？

三个选项：

  same        = 同一首录音被重复收录了。比如同一首歌换了个标题又存了一遍、
                重发版、抓取时抓重了、其中一条是另一条的残缺版本。

  different   = 不是同一首录音。歌词重合不一定就是同一首：
                翻唱、remix、Live 版、feat. 版本、
                同一段 hook 被这位歌手用在两首不同的歌里、
                串烧或合辑里包含了另一首歌的一段、
                或者抓取时把别的歌的词错挂到了这首上。

  cannot_tell = 看不出来。这是一个正当答案。
                这 46 条之所以在这里，就是因为机器判不了。
                你判不了就选这个，不要为了填满而猜。

判断依据只能是卡片上有的东西：署名标签、标题、歌词内容、以及那行结构比对。

不要用来判断的：
  - 歌手的籍贯、生平、居住地、社会关系；
  - 你对这位歌手的既有印象；
  - 任何本页面之外的资料。

你的答案会怎么用：
  选 same 的，两条记录会合成一个"作品单位"来计数，
  免得同一首歌在统计里被算了两三遍。记录本身不会被删掉。
  选 different 的，就继续当成两首不同的歌。
  选 cannot_tell 的，会在论文里作为敏感性区间两头都报一遍。

卡片故意没告诉你哪一条是被旧规则删掉的、机器把它归成了哪一类，左右顺序也是打乱的。
这样你的判断才独立于机器的判断。

如果你发现某一条记录里混进了不属于它的内容（比如不重合的那部分其实是另一首歌），
请勾上"有记录混进了别的歌"。这个勾选和上面三选一是分开的两件事：
三选一回答"这两条是不是同一首"，勾选框记录"有记录被别的歌污染了"。
两者可以同时成立，也可以只成立一个。

段落下面如果写着"这段还出现在卡片外的 N 首歌里"，说明它并不是这两条记录独有的。

理由不是"歌词一样"的时候，请在备注里写一句。
"""


def title_normalise(value: str) -> str:
    """NFKC, case-fold, alphanumerics only -- the same shape the duplicate rule uses."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_card(index: int, item: dict[str, Any]) -> str:
    sides = "".join(
        f"""
      <div class="side">
        <div class="side-head">记录 {chr(65 + position)}</div>
        <dl>
          <dt>署名标签</dt><dd>{html.escape(side["label"])}</dd>
          <dt>标题</dt><dd>{html.escape(side["title"])}</dd>
          <dt>段落数</dt><dd>{side["chunk_rows"]}</dd>
        </dl>
        {"".join(
            f'<pre class="para{" shared" if para in item["shared"] else ""}'
            f'{" elsewhere" if item["off_card"].get(para) else ""}"'
            f' data-elsewhere="{item["off_card"].get(para, 0)}">'
            f'{html.escape(para)}</pre>'
            for para in side["paragraphs"])}
      </div>"""
        for position, side in enumerate(item["sides"])
    )
    name = f"ruling-{item['review_id']}"
    options = "".join(
        f"""<label class="opt"><input type="radio" name="{name}" value="{value}"> {label}</label>"""
        for value, label in (
            ("same", "same — 同一首录音"),
            ("different", "different — 不是同一首"),
            ("cannot_tell", "cannot_tell — 看不出来"),
        )
    )
    return f"""
  <section class="card" data-review-id="{item['review_id']}">
    <header>
      <span class="num">{index} / {item['total']}</span>
      <span class="rid">{item['review_id']}</span>
    </header>
    <p class="structure">{html.escape(item["structure"])}</p>
    <div class="sides">{sides}</div>
    <div class="ruling">
      {options}
      <label class="mix"><input type="checkbox" class="mixed" data-review-id="{item['review_id']}">
        有记录混进了别的歌</label>
      <input class="note" type="text" placeholder="备注（可选）" data-review-id="{item['review_id']}">
    </div>
  </section>"""


def render_sheet(items: list[dict[str, Any]], instructions_sha: str, generated_at: str) -> str:
    cards = "".join(render_card(i, item) for i, item in enumerate(items, start=1))
    return f"""<!doctype html>
<html lang="zh-Hans">
<head>
<meta charset="utf-8">
<title>PD-002 重复记录人工裁决 ({len(items)} 条)</title>
<style>
 :root {{ color-scheme: light dark; --line:#c8c8c8; --bg:#fff; --fg:#1a1a1a; --muted:#666;
          --panel:#f6f6f6; --shared:#b4541f; --sharedbg:#fdf3ec; --elsewhere:#6a4ba8; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --line:#3a3a3a; --bg:#151515; --fg:#e8e8e8; --muted:#999; --panel:#1e1e1e;
            --shared:#e0863f; --sharedbg:#251d16; --elsewhere:#b39ae0; }}
 }}
 body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:2rem 1.25rem 6rem; max-width:70rem;
        font:15px/1.65 "Segoe UI", system-ui, sans-serif; }}
 h1 {{ font-size:1.35rem; margin:0 0 .35rem; }}
 .lede {{ color:var(--muted); margin:0 0 1.5rem; }}
 pre.instructions {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
        padding:1rem; white-space:pre-wrap; font:13.5px/1.7 "Segoe UI", system-ui, sans-serif; }}
 .card {{ border:1px solid var(--line); border-radius:10px; margin:1.5rem 0; overflow:hidden; }}
 .card.done {{ border-color:#3a8f5a; }}
 .card > header {{ display:flex; justify-content:space-between; align-items:baseline;
        background:var(--panel); padding:.5rem .9rem; border-bottom:1px solid var(--line); }}
 .num {{ font-weight:600; }}
 .rid {{ color:var(--muted); font-family:ui-monospace, Consolas, monospace; font-size:.8rem; }}
 .structure {{ margin:0; padding:.5rem .9rem; background:var(--panel); color:var(--muted);
        font-size:.87rem; border-bottom:1px solid var(--line); }}
 .sides {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(19rem,1fr)); gap:1px; background:var(--line); }}
 .side {{ background:var(--bg); padding:.9rem; }}
 .side-head {{ font-weight:600; margin-bottom:.5rem; }}
 dl {{ display:grid; grid-template-columns:auto 1fr; gap:.15rem .7rem; margin:0 0 .7rem; font-size:.88rem; }}
 dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
 .para.shared {{ border-left:4px solid var(--shared); background:var(--sharedbg); }}
 .para.shared::before {{ content:"两边共有"; display:block; font-size:.72rem; color:var(--shared);
        font-family:"Segoe UI", system-ui, sans-serif; margin-bottom:.3rem; letter-spacing:.04em; }}
 pre {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:.7rem;
        max-height:18rem; overflow:auto; margin-bottom:.4rem; white-space:pre-wrap; overflow-wrap:anywhere;
        font:13px/1.6 ui-monospace, Consolas, monospace; margin:0; }}
 .ruling {{ display:flex; flex-wrap:wrap; gap:.6rem 1.1rem; align-items:center;
        padding:.75rem .9rem; border-top:1px solid var(--line); background:var(--panel); }}
 .para.elsewhere::after {{ content:"这段还出现在卡片外的 " attr(data-elsewhere) " 首歌里";
        display:block; margin-top:.35rem; font-size:.72rem; color:var(--elsewhere);
        font-family:"Segoe UI", system-ui, sans-serif; }}
 .mix {{ cursor:pointer; user-select:none; color:var(--elsewhere); }}
 .opt {{ cursor:pointer; user-select:none; }}
 .note {{ flex:1 1 16rem; min-width:12rem; padding:.35rem .5rem; border:1px solid var(--line);
        border-radius:5px; background:var(--bg); color:var(--fg); font:inherit; font-size:.9rem; }}
 #bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--panel);
        border-top:1px solid var(--line); padding:.7rem 1.25rem; display:flex; gap:1rem;
        align-items:center; justify-content:center; flex-wrap:wrap; }}
 button {{ font:inherit; padding:.45rem 1rem; border:1px solid var(--line); border-radius:6px;
        background:var(--bg); color:var(--fg); cursor:pointer; }}
 button:disabled {{ opacity:.5; cursor:default; }}
 #count {{ font-variant-numeric:tabular-nums; }}
</style>
</head>
<body>
<h1>PD-002 重复记录人工裁决</h1>
<p class="lede">{len(items)} 条 · 生成于 {html.escape(generated_at)} · 协议 {PROTOCOL_ID}</p>
<pre class="instructions">{html.escape(INSTRUCTIONS)}</pre>
<p class="lede">这份文件含有歌词原文，属于私有工作文件，不要放进公开仓库、也不要发布。</p>
{cards}
<div id="bar">
  <span id="count">0 / {len(items)}</span>
  <button id="save" disabled>下载裁决结果 (.json)</button>
  <button id="copy" disabled>复制到剪贴板</button>
</div>
<script>
const PROTOCOL = {json.dumps(PROTOCOL_ID)};
const INSTRUCTIONS_SHA256 = {json.dumps(instructions_sha)};
const TOTAL = {len(items)};
const KEY = "pd002-duplicate-review-v1";

function collect() {{
  const out = [];
  document.querySelectorAll(".card").forEach(card => {{
    const id = card.dataset.reviewId;
    const picked = card.querySelector("input[type=radio]:checked");
    const note = card.querySelector(".note").value.trim();
    const mixed = card.querySelector(".mixed").checked;
    // a note alone is still the rater's work: an earlier version pushed a card only when a
    // radio or the checkbox was set, so a note typed while thinking about a card was thrown
    // away on save without a word
    if (picked || mixed || note) out.push({{review_id: id, ruling: picked ? picked.value : "",
                                            mixed_in_content: mixed, note: note}});
  }});
  return out;
}}

function payload() {{
  return {{
    protocol_id: PROTOCOL,
    instructions_sha256: INSTRUCTIONS_SHA256,
    expected_records: TOTAL,
    rulings: collect()
  }};
}}

function refresh() {{
  const done = collect().filter(r => r.ruling).length;
  const partial = collect().filter(r => !r.ruling).length;
  document.getElementById("count").textContent =
    done + " / " + TOTAL + (partial ? "   (" + partial + " 条只写了备注/勾选，还没选选项)" : "");
  document.getElementById("save").disabled = done === 0;
  document.getElementById("copy").disabled = done === 0;
  document.querySelectorAll(".card").forEach(card => {{
    card.classList.toggle("done", !!card.querySelector("input[type=radio]:checked"));
  }});
  try {{ localStorage.setItem(KEY, JSON.stringify(payload())); }} catch (e) {{}}
}}

try {{
  const saved = JSON.parse(localStorage.getItem(KEY) || "null");
  if (saved && saved.rulings) {{
    saved.rulings.forEach(r => {{
      const card = document.querySelector('.card[data-review-id="' + r.review_id + '"]');
      if (!card) return;
      const radio = card.querySelector('input[value="' + r.ruling + '"]');
      if (radio) radio.checked = true;
      if (r.mixed_in_content) card.querySelector(".mixed").checked = true;
      if (r.note) card.querySelector(".note").value = r.note;
    }});
  }}
}} catch (e) {{}}

document.addEventListener("input", refresh);
document.getElementById("save").addEventListener("click", () => {{
  const blob = new Blob([JSON.stringify(payload(), null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "pd002_duplicate_rulings.json";
  a.click();
  URL.revokeObjectURL(a.href);
}});
document.getElementById("copy").addEventListener("click", async () => {{
  try {{ await navigator.clipboard.writeText(JSON.stringify(payload(), null, 2)); }}
  catch (e) {{ alert("复制失败，请用下载按钮。"); }}
}});
refresh();
</script>
</body>
</html>
"""


def build(args: argparse.Namespace) -> None:
    private_dir = args.private_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.is_relative_to(REPO_ROOT):
        raise SystemExit(
            f"--out-dir must not resolve inside the public repository: {out_dir}"
        )

    queue = read_csv(private_dir / "duplicate_review_queue_v2.csv")
    chunks = read_csv(private_dir / "repaired_lyric_chunks_v2.csv")
    manifest = json.loads((private_dir / "private_manifest.json").read_text(encoding="utf-8"))

    by_song: dict[str, list[dict[str, str]]] = defaultdict(list)
    owners: dict[str, set[str]] = defaultdict(set)
    for row in chunks:
        by_song[row["song_id"]].append(row)
        owners[row["cleaned_text"]].add(row["song_id"])
    for rows in by_song.values():
        rows.sort(key=lambda row: int(row["within_song_order"]))

    def side_for(song_id: str) -> dict[str, Any]:
        rows = by_song[song_id]
        return {
            "song_id": song_id,
            "label": rows[0]["source_credit_label"],
            "title": rows[0]["song_title"],
            "chunk_rows": len(rows),
            "paragraphs": [row["cleaned_text"] for row in rows],
            "text": "\n\n———\n\n".join(row["cleaned_text"] for row in rows),
        }

    def structure_note(sides: list[dict[str, Any]]) -> str:
        """Describe how the paragraph sets line up, without saying what it means.

        The rater could work this out by diffing the panels by hand; stating it saves that
        labour without supplying a verdict. The automatic classification stays withheld.
        """
        letters = [chr(65 + i) for i in range(len(sides))]
        sets = [set(side["paragraphs"]) for side in sides]
        seqs = [side["paragraphs"] for side in sides]
        counts = " / ".join(f"{letter} {len(seq)} 段" for letter, seq in zip(letters, seqs))
        if len(sides) == 2:
            if seqs[0] == seqs[1]:
                return f"{counts} —— 两边逐段完全相同，顺序也一样。"
            if sets[0] == sets[1]:
                return f"{counts} —— 段落内容相同，但顺序不同。"
            for a, b in ((0, 1), (1, 0)):
                if sets[a] < sets[b]:
                    shared = len(sets[a])
                    return (f"{counts} —— {letters[a]} 的 {shared} 段全部出现在 "
                            f"{letters[b]} 里面，{letters[b]} 还多出 {len(sets[b]) - shared} 段。")
            shared = len(sets[0] & sets[1])
            return (f"{counts} —— 共有 {shared} 段相同，"
                    f"{letters[0]} 另有 {len(sets[0] - sets[1])} 段，"
                    f"{letters[1]} 另有 {len(sets[1] - sets[0])} 段。")
        union = set().union(*sets)
        pairs = []
        for i, letter in enumerate(letters):
            others = set().union(*(sets[j] for j in range(len(sets)) if j != i))
            only = len(sets[i] - others)
            pairs.append(f"{letter} 独有 {only} 段" if only else f"{letter} 没有独有的段")
        return f"{counts} —— 合计 {len(union)} 个不同的段；" + "，".join(pairs) + "。"

    def shared_shape_note(shared: set[str], sides: list[dict[str, Any]]) -> str:
        """Say what the shared text physically is, when that is not a verse.

        A lone line that restates a title is a header artefact of the source export, not a
        shared verse, and a reader skimming two long panels will not notice that the marked
        paragraph is only one line long. Stating the shape is a description of the panels;
        it is still the rater who decides what the overlap means.
        """
        if not shared:
            return ""
        titles = {title_normalise(side["title"]) for side in sides}
        singles = [t for t in shared
                   if len([l for l in t.split(chr(10)) if l.strip()]) == 1]
        if len(singles) != len(shared):
            return ""
        restates = sum(
            1 for t in singles
            for line in [title_normalise(t)]
            if any(ti and (line.startswith(ti[:8]) or ti.startswith(line[:8]))
                   for ti in titles if len(ti) >= 4))
        if restates == len(singles):
            return "  标出来的这段只有一行，内容和标题重复。"
        return "  标出来的这段只有一行。"

    def off_card_note(off_card: dict[str, int], shared: set[str]) -> str:
        """Say when paragraphs on this card also belong to songs that are not shown.

        The card renders only the records the automatic rule related. Ownership is a property
        of the whole corpus, so a paragraph can look unique here and not be. Asserting
        nothing about what that means: it is a count, and it is the rater who decides whether
        a record carrying someone else's paragraph is still the same recording.
        """
        if not off_card:
            return ""
        outside = sum(1 for para in off_card if para not in shared)
        if not outside:
            return ""
        return (f"  注意：其中 {outside} 段在本卡之外的歌里也有，"
                "所以它们并不是这两条记录独有的。")

    shuffler = random.Random(SHUFFLE_SEED)
    items: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for position, row in enumerate(sorted(queue, key=lambda r: r["song_id"]), start=1):
        related = [s for s in row["related_song_ids"].split(";") if s]
        sides = [side_for(row["song_id"])] + [side_for(s) for s in related]
        shuffler.shuffle(sides)
        review_id = f"DR-{position:03d}"
        paragraph_sets = [set(side["paragraphs"]) for side in sides]
        shared = {para for i, first in enumerate(paragraph_sets)
                  for j, second in enumerate(paragraph_sets) if i < j
                  for para in first & second}
        on_card = {side["song_id"] for side in sides}
        off_card = {para: len(owners[para] - on_card)
                    for side in sides for para in side["paragraphs"]
                    if owners[para] - on_card}
        items.append({"review_id": review_id, "sides": sides, "total": len(queue),
                      "shared": shared, "off_card": off_card,
                      "structure": structure_note(sides) + shared_shape_note(shared, sides)
                      + off_card_note(off_card, shared)})
        key.append(
            {
                "review_id": review_id,
                "queued_song_id": row["song_id"],
                "related_song_ids": related,
                "presented_order": [side["song_id"] for side in sides],
                # withheld from the rater, recorded here so the ruling can be analysed against it
                "automatic_reason": row["reason"],
                "auto_grouped_by_v2_primary_rule": row["auto_grouped_by_v2_primary_rule"],
            }
        )

    instructions_sha = sha256_text(INSTRUCTIONS)
    generated_at = args.generated_at
    sheet = render_sheet(items, instructions_sha, generated_at)

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "PD002_重复记录人工裁决.html"
    sheet_path.write_text(sheet, encoding="utf-8", newline="")
    (out_dir / "pd002_duplicate_review_key.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "generated_at": generated_at,
                "instructions_sha256": instructions_sha,
                "instructions": INSTRUCTIONS,
                "presentation_seed": SHUFFLE_SEED,
                "repaired_corpus_content_sha256": manifest["repaired_corpus_content_sha256"],
                "review_queue_sha256": manifest["files"]["duplicate_review_queue_v2.csv"]["sha256"],
                "records": key,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    print(f"{len(items)} records -> {sheet_path}")
    print(f"blinding key      -> {out_dir / 'pd002_duplicate_review_key.json'}")
    print(f"instructions sha  -> {instructions_sha}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
