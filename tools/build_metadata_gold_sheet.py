#!/usr/bin/env python3
"""Generate the MB-001 gold-set labelling sheet for the metadata-block detector.

The detector in ``tools/detect_metadata_blocks.py`` has no precision or recall, because
nothing independent has ever labelled what a metadata line actually is. This renders a
stratified sample of the repaired corpus so a human can label it, blind to what the detector
said.

The sample is stratified by detector output -- half from chunks where at least one line
fires, half from chunks where none does -- because a uniform sample of a corpus this size
would contain almost no positives and could not estimate precision at all. Stratification
makes the sample non-representative by construction, so the per-stratum sampling
probabilities are written into the key file and every later estimate must be weighted by
them. The sheet is not a random sample of the corpus and must never be reported as one.

Blinding: the rater is not shown which lines fired, which stratum a chunk came from, or any
rule name. Chunks from both strata are interleaved under a fixed seed.

The sheet carries lyric text. It is a private working file and must never be written into
the public repository or published.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_ID = "MB-001-GOLD-001"
SAMPLE_SEED = "mb001-gold-sample-v1"

INSTRUCTIONS = """\
每一段下面是一首歌里的一段文本。请把其中**不是歌词**的行标出来。

默认所有行都算歌词，你只需要点掉不是歌词的那些。点一下变成"非歌词"，再点一下变回来。

什么算"非歌词"：
  制作署名（作词/作曲/编曲/混音/母带/OP/SP/厂牌/公司名）、
  采样出处、版权声明、联系方式、
  只有人名或人名列表的行、
  【副歌】【Verse 1】这类结构标记。

什么仍然算歌词：
  重复的 hook、语气词、ad-lib、英文行、
  歌词里提到的公司或地名（"我在某某公司楼下等你" 是歌词）、
  你不确定但读起来像唱出来的东西。

拿不准的行，请点"不确定"（第三种状态），不要硬猜。不确定是有用的信息。

说明：这些段落是抽样抽出来的，不是按顺序的全部语料；也没有告诉你机器判断的结果，
这是故意的——需要你的判断独立于机器。
"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_detector():
    spec = importlib.util.spec_from_file_location(
        "detect_metadata_blocks", REPO_ROOT / "tools" / "detect_metadata_blocks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_chunk(index: int, item: dict[str, Any], total: int) -> str:
    rows = "".join(
        f"""
      <li class="ln" data-line="{position}">
        <button type="button" class="state" data-state="lyric">歌词</button>
        <span class="txt">{html.escape(line) or "&nbsp;"}</span>
      </li>"""
        for position, line in enumerate(item["lines"])
    )
    return f"""
  <section class="chunk" data-gold-id="{item['gold_id']}">
    <header><span class="num">{index} / {total}</span>
            <span class="gid">{item['gold_id']}</span></header>
    <ol class="lines">{rows}</ol>
    <div class="confirm"><button type="button" class="allclear">本段全是歌词</button></div>
  </section>"""


def render_sheet(items: list[dict[str, Any]], instructions_sha: str, generated_at: str) -> str:
    chunks = "".join(render_chunk(i, item, len(items)) for i, item in enumerate(items, start=1))
    return f"""<!doctype html>
<html lang="zh-Hans">
<head>
<meta charset="utf-8">
<title>MB-001 元数据行标注 ({len(items)} 段)</title>
<style>
 :root {{ color-scheme: light dark; --line:#c8c8c8; --bg:#fff; --fg:#1a1a1a; --muted:#666;
          --panel:#f6f6f6; --meta:#b4441f; --unsure:#8a6d1f; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --line:#3a3a3a; --bg:#151515; --fg:#e8e8e8; --muted:#999; --panel:#1e1e1e;
            --meta:#e0763f; --unsure:#d3ad3c; }}
 }}
 body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:2rem 1.25rem 6rem;
        max-width:56rem; font:15px/1.65 "Segoe UI", system-ui, sans-serif; }}
 h1 {{ font-size:1.35rem; margin:0 0 .35rem; }}
 .lede {{ color:var(--muted); margin:0 0 1.5rem; }}
 pre.instructions {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
        padding:1rem; white-space:pre-wrap; font:13.5px/1.7 "Segoe UI", system-ui, sans-serif; }}
 .chunk {{ border:1px solid var(--line); border-radius:10px; margin:1.25rem 0; overflow:hidden; }}
 .chunk.done {{ border-color:#3a8f5a; }}
 .chunk > header {{ display:flex; justify-content:space-between; align-items:baseline;
        background:var(--panel); padding:.5rem .9rem; border-bottom:1px solid var(--line); }}
 .num {{ font-weight:600; }}
 .gid {{ color:var(--muted); font-family:ui-monospace, Consolas, monospace; font-size:.8rem; }}
 ol.lines {{ list-style:none; margin:0; padding:.4rem 0; }}
 .ln {{ display:flex; gap:.7rem; align-items:flex-start; padding:.15rem .9rem; }}
 .txt {{ white-space:pre-wrap; overflow-wrap:anywhere;
        font:14px/1.6 ui-monospace, Consolas, monospace; }}
 .state {{ flex:0 0 4.2rem; font:inherit; font-size:.78rem; padding:.15rem 0;
        border:1px solid var(--line); border-radius:5px; background:var(--bg);
        color:var(--muted); cursor:pointer; }}
 .state[data-state="metadata"] {{ color:#fff; background:var(--meta); border-color:var(--meta); }}
 .state[data-state="unsure"] {{ color:#fff; background:var(--unsure); border-color:var(--unsure); }}
 .ln:has(.state[data-state="metadata"]) .txt {{ opacity:.55; text-decoration:line-through; }}
 .confirm {{ padding:.5rem .9rem; border-top:1px solid var(--line); background:var(--panel); }}
 .allclear {{ font:inherit; font-size:.85rem; padding:.3rem .8rem; border:1px solid var(--line);
        border-radius:5px; background:var(--bg); color:var(--fg); cursor:pointer; }}
 .chunk.done .allclear {{ visibility:hidden; }}
 #bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--panel);
        border-top:1px solid var(--line); padding:.7rem 1.25rem; display:flex; gap:1rem;
        align-items:center; justify-content:center; flex-wrap:wrap; }}
 button.act {{ font:inherit; padding:.45rem 1rem; border:1px solid var(--line);
        border-radius:6px; background:var(--bg); color:var(--fg); cursor:pointer; }}
 button.act:disabled {{ opacity:.5; cursor:default; }}
 #count {{ font-variant-numeric:tabular-nums; }}
</style>
</head>
<body>
<h1>MB-001 元数据行标注</h1>
<p class="lede">{len(items)} 段 · 生成于 {html.escape(generated_at)} · 协议 {PROTOCOL_ID}</p>
<pre class="instructions">{html.escape(INSTRUCTIONS)}</pre>
<p class="lede">这份文件含有歌词原文，属于私有工作文件，不要放进公开仓库、也不要发布。
每一段都需要一个答案：改掉不是歌词的行，或者按"本段全是歌词"。两种都算完成。
整段跳过不算——"这一段没有署名"本身就是要收集的证据。中途可以关掉，进度存在浏览器里。</p>
{chunks}
<div id="bar">
  <span id="count">0 / {len(items)}</span>
  <button class="act" id="save" disabled>下载标注 (.json)</button>
  <button class="act" id="copy" disabled>复制到剪贴板</button>
</div>
<script>
const PROTOCOL = {json.dumps(PROTOCOL_ID)};
const INSTRUCTIONS_SHA256 = {json.dumps(instructions_sha)};
const TOTAL = {len(items)};
const KEY = "mb001-gold-v1";
const CYCLE = {{lyric: "metadata", metadata: "unsure", unsure: "lyric"}};
const TEXT = {{lyric: "歌词", metadata: "非歌词", unsure: "不确定"}};

document.querySelectorAll(".allclear").forEach(btn => {{
  btn.addEventListener("click", () => {{
    btn.closest(".chunk").dataset.touched = "1";
    refresh();
  }});
}});

document.querySelectorAll(".state").forEach(btn => {{
  btn.addEventListener("click", () => {{
    const next = CYCLE[btn.dataset.state];
    btn.dataset.state = next;
    btn.textContent = TEXT[next];
    btn.closest(".chunk").dataset.touched = "1";
    refresh();
  }});
}});

function collect() {{
  const out = [];
  document.querySelectorAll(".chunk").forEach(chunk => {{
    if (!chunk.dataset.touched) return;
    const labels = [];
    chunk.querySelectorAll(".ln").forEach(ln => {{
      labels.push({{line: Number(ln.dataset.line),
                   label: ln.querySelector(".state").dataset.state}});
    }});
    out.push({{gold_id: chunk.dataset.goldId, labels: labels}});
  }});
  return out;
}}

function payload() {{
  return {{protocol_id: PROTOCOL, instructions_sha256: INSTRUCTIONS_SHA256,
          expected_chunks: TOTAL, labelled: collect()}};
}}

function refresh() {{
  const done = collect().length;
  document.getElementById("count").textContent = done + " / " + TOTAL;
  document.getElementById("save").disabled = done === 0;
  document.getElementById("copy").disabled = done === 0;
  document.querySelectorAll(".chunk").forEach(c => c.classList.toggle("done", !!c.dataset.touched));
  try {{ localStorage.setItem(KEY, JSON.stringify(payload())); }} catch (e) {{}}
}}

try {{
  const saved = JSON.parse(localStorage.getItem(KEY) || "null");
  if (saved && saved.labelled) {{
    saved.labelled.forEach(rec => {{
      const chunk = document.querySelector('.chunk[data-gold-id="' + rec.gold_id + '"]');
      if (!chunk) return;
      chunk.dataset.touched = "1";
      rec.labels.forEach(l => {{
        const btn = chunk.querySelector('.ln[data-line="' + l.line + '"] .state');
        if (btn) {{ btn.dataset.state = l.label; btn.textContent = TEXT[l.label]; }}
      }});
    }});
  }}
}} catch (e) {{}}

document.getElementById("save").addEventListener("click", () => {{
  const blob = new Blob([JSON.stringify(payload(), null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "mb001_gold_labels.json";
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
        raise SystemExit(f"--out-dir must not resolve inside the public repository: {out_dir}")

    detector = load_detector()
    csv.field_size_limit(10**9)
    with (private_dir / "repaired_lyric_chunks_v2.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    manifest = json.loads((private_dir / "private_manifest.json").read_text(encoding="utf-8"))

    # stratify by detector output, and record how many chunks each stratum drew from so the
    # sampling probabilities are recoverable
    positive: list[dict[str, str]] = []
    negative: list[dict[str, str]] = []
    for row in rows:
        lines = row["cleaned_text"].split("\n")
        if len(lines) < 2 or len(lines) > args.max_lines:
            continue
        labels = detector.classify(row["cleaned_text"])
        (positive if any(entry["rule"] for entry in labels) else negative).append(row)

    rng = random.Random(SAMPLE_SEED)
    half = args.size // 2
    drawn = (
        [("detector_positive", row, len(positive)) for row in rng.sample(positive, min(half, len(positive)))]
        + [("detector_negative", row, len(negative)) for row in rng.sample(negative, min(args.size - half, len(negative)))]
    )
    rng.shuffle(drawn)

    items: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for position, (stratum, row, population) in enumerate(drawn, start=1):
        gold_id = f"MB-{position:03d}"
        lines = row["cleaned_text"].split("\n")
        labels = detector.classify(row["cleaned_text"])
        items.append({"gold_id": gold_id, "lines": lines})
        key.append(
            {
                "gold_id": gold_id,
                "song_id": row["song_id"],
                "chunk_id": int(row["chunk_id"]),
                "line_count": len(lines),
                # withheld from the rater; recorded so the labels can score the detector
                "stratum": stratum,
                "stratum_population_chunks": population,
                "stratum_sampled_chunks": sum(1 for s, _, _ in drawn if s == stratum),
                "detector_rules": [entry["rule"] for entry in labels],
            }
        )

    instructions_sha = sha256_text(INSTRUCTIONS)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "MB001_元数据行标注.html"
    sheet_path.write_text(
        render_sheet(items, instructions_sha, args.generated_at), encoding="utf-8", newline=""
    )
    (out_dir / "mb001_gold_key.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "generated_at": args.generated_at,
                "instructions_sha256": instructions_sha,
                "instructions": INSTRUCTIONS,
                "sample_seed": SAMPLE_SEED,
                "design": (
                    "stratified by detector output, not a uniform corpus sample; every "
                    "estimate must be weighted by the per-stratum sampling probability "
                    "recorded on each record"
                ),
                "eligible_chunk_filter": f"2 to {args.max_lines} lines",
                "detector_sha256": sha256_text(
                    (REPO_ROOT / "tools" / "detect_metadata_blocks.py").read_text(encoding="utf-8")
                ),
                "repaired_corpus_content_sha256": manifest["repaired_corpus_content_sha256"],
                "raters": 1,
                "reliability_estimable": False,
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
    lines_total = sum(len(item["lines"]) for item in items)
    print(f"{len(items)} chunks ({lines_total} lines) -> {sheet_path}")
    print(f"  detector_positive stratum: {sum(1 for r in key if r['stratum'] == 'detector_positive')}"
          f" of {len(positive)} eligible chunks")
    print(f"  detector_negative stratum: {sum(1 for r in key if r['stratum'] == 'detector_negative')}"
          f" of {len(negative)} eligible chunks")
    print(f"blinding key -> {out_dir / 'mb001_gold_key.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--max-lines", type=int, default=14)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
