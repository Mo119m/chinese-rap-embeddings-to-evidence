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

INSTRUCTIONS = """\
你要判断的问题只有一个：下面两条(有时更多)歌曲记录，是不是同一首录音作品(the same recorded work)？

判断依据只能是你在卡片上看到的东西：署名标签、标题、以及歌词内容本身。

请不要用来判断的东西：
  - 歌手的籍贯、生平、居住地、社会关系；
  - 你对这位歌手的既有印象；
  - 任何本页面之外的资料。

三个选项的含义：
  same          = 同一首录音的重复导入记录(重发、换标题、抓取重复，都算同一首)。
  different     = 不是同一首录音。歌词相同或部分相同也可能是不同作品
                  (例如翻唱、remix、同一段 hook 被用在两首不同的歌里)。
  cannot_tell   = 看不出来。这是一个正当答案，不要为了填满而猜。

说明：卡片没有告诉你哪一条被旧的清洗规则删掉了，也没有告诉你自动规则怎么分类的。
这是故意的——你的判断需要独立于那个自动分类。

每条都可以写备注。如果你的理由不是"歌词一样"，请写下来。
"""


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
        <pre>{html.escape(side["text"])}</pre>
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
    <div class="sides">{sides}</div>
    <div class="ruling">
      {options}
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
 :root {{ color-scheme: light dark; --line:#c8c8c8; --bg:#fff; --fg:#1a1a1a; --muted:#666; --panel:#f6f6f6; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --line:#3a3a3a; --bg:#151515; --fg:#e8e8e8; --muted:#999; --panel:#1e1e1e; }}
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
 .sides {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(19rem,1fr)); gap:1px; background:var(--line); }}
 .side {{ background:var(--bg); padding:.9rem; }}
 .side-head {{ font-weight:600; margin-bottom:.5rem; }}
 dl {{ display:grid; grid-template-columns:auto 1fr; gap:.15rem .7rem; margin:0 0 .7rem; font-size:.88rem; }}
 dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
 pre {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:.7rem;
        max-height:22rem; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere;
        font:13px/1.6 ui-monospace, Consolas, monospace; margin:0; }}
 .ruling {{ display:flex; flex-wrap:wrap; gap:.6rem 1.1rem; align-items:center;
        padding:.75rem .9rem; border-top:1px solid var(--line); background:var(--panel); }}
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
    if (picked) out.push({{review_id: id, ruling: picked.value, note: note}});
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
  const done = collect().length;
  document.getElementById("count").textContent = done + " / " + TOTAL;
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
    for row in chunks:
        by_song[row["song_id"]].append(row)
    for rows in by_song.values():
        rows.sort(key=lambda row: int(row["within_song_order"]))

    def side_for(song_id: str) -> dict[str, Any]:
        rows = by_song[song_id]
        return {
            "song_id": song_id,
            "label": rows[0]["source_credit_label"],
            "title": rows[0]["song_title"],
            "chunk_rows": len(rows),
            "text": "\n\n———\n\n".join(row["cleaned_text"] for row in rows),
        }

    shuffler = random.Random(SHUFFLE_SEED)
    items: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for position, row in enumerate(sorted(queue, key=lambda r: r["song_id"]), start=1):
        related = [s for s in row["related_song_ids"].split(";") if s]
        sides = [side_for(row["song_id"])] + [side_for(s) for s in related]
        shuffler.shuffle(sides)
        review_id = f"DR-{position:03d}"
        items.append({"review_id": review_id, "sides": sides, "total": len(queue)})
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
