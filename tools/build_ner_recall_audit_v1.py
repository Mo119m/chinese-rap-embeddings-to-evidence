#!/usr/bin/env python3
"""Build the NER recall audit package: a random sample of lyric chunks for two reviewers
to mark exhaustively, independent of anything the NER proposed.

The released-claim audit (NER-RELEASED-CLAIM-AUDIT) can only say whether the occurrences
behind the released claims are valid -- precision on a claim-conditioned set. It cannot say
what the NER missed, because every task in it was proposed by the NER. Recall needs the
opposite design: draw chunks at random from the NER's own input frame, have two people
mark every place name and every language/dialect name they can see without seeing the
NER's candidates, adjudicate, and count how many of those mentions the NER (each arm, and
the strict agreement gate) had found.

Sample: --sample chunks (default 100) drawn uniformly without replacement from the
eligible clean-text chunks of the canonical sidecar (the 21,553-chunk frame the NER ran
on), seeded; no length filter, so long chunks are represented as they are in the frame.
Task order is shuffled by the same seed. Task ids are blind hashes.

What leaves the private root: nothing but this file's public status JSON (design and
counts). The reviewer sheets carry lyric text and are written to the private output and
the desktop folder only.

    python tools/build_ner_recall_audit_v1.py --private-root <ni-k> --desktop-dir <folder>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ID = "chinese-rap-ner-recall-audit-v1"
VERSION = "1.0.0"
SEED = "chinese-rap-ner-recall-audit-v1-20260910"
TARGET_TYPES = ("PLACE", "LANGUAGE_OR_DIALECT_REFERENCE")
csv.field_size_limit(10 ** 9)

GUIDE = """\
这份和之前那份相反：之前是"系统圈了一个词，你判对不对"；这份是**系统什么都不告诉你**，
你把这段歌词里**所有**的地名和语言/方言名自己找出来。目的是量系统漏了多少。

每一条是一段歌词，行前有行号。请在下面的框里，每行写一个你找到的词，格式：

    行号 词 类型

类型只有两种：**地名** 或 **语言**。例子（编的）：

    3 上海 地名
    7 粤语 语言
    12 香港 地名

要标的：城市、省、国家、区、街区、山河湖海、具体的场馆或地标——只要它在这句里是在指那个地方。
语言、方言、口音的名字（粤语、四川话、东北话、English 这种）。
同一个词在不同行出现几次就写几次；同一行出现两次写两次。
比喻用法（"你是我的巴黎"）也写，词后面加个问号：`5 巴黎? 地名`。

不要标的：人名、艺名、团体名、品牌、歌名、作品名。
只看这段文字，不用歌手是谁、哪里人来判断。

这段里一个都没有就勾"没有"。confidence 1–5 是你对整条的把握。
两位评审人独立作业，中途不要互相对答案。做完点底部「下载 CSV」。
文件含受版权保护的歌词，只能留在本机：不要提交、不要发布、不要邮件、不要上传。"""


def blind_id(*parts: str) -> str:
    return "RECALL-" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16].upper()


def render_sheet(reviewer: str, tasks: list[dict]) -> str:
    cards = []
    for index, task in enumerate(tasks, 1):
        lines = task["text"].split("\n")
        numbered = "\n".join(f"{n:>3}  {html.escape(line)}" for n, line in enumerate(lines, 1))
        cards.append(f"""
<section class="task" data-task="{task['task_id']}">
  <div class="head"><span class="num">{index} / {len(tasks)}</span> <span class="tid">{task['task_id']}</span></div>
  <pre class="ctx">{numbered}</pre>
  <div class="field"><label>没有</label><label class="chk"><input type="checkbox" class="none" data-field="no_place_or_language_mentions"> 这段里没有任何地名或语言名</label></div>
  <div class="field"><label>mentions</label><textarea class="t mentions" data-field="mentions" rows="4" placeholder="行号 词 类型（每行一个）"></textarea></div>
  <div class="field"><label>confidence_1_to_5</label><div class="opts">{''.join(f'<button type="button" class="v" data-field="confidence_1_to_5" data-value="{v}">{v}</button>' for v in "12345")}</div></div>
  <div class="field"><label>notes</label><input type="text" class="t notes" data-field="notes" placeholder="备注（可选）"></div>
</section>""")
    guide = html.escape(GUIDE)
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>NER 召回复核 · {reviewer}</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#d9d9d9; --panel:#f6f6f6; --pick:#2457c5; --warn:#b00020; }}
body {{ margin:0; padding:1.2rem 1.2rem 6rem; font:15px/1.6 system-ui, "Microsoft YaHei", sans-serif; color:var(--fg); background:var(--bg); max-width:64rem; }}
pre.guide {{ white-space:pre-wrap; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:1rem; font:14px/1.7 inherit; }}
section.task {{ border:1px solid var(--line); border-radius:10px; margin:1.2rem 0; }}
section.task.done {{ border-color:#2e7d32; }}
.head {{ background:var(--panel); padding:.5rem .9rem; border-radius:10px 10px 0 0; }}
.num {{ font-weight:600; }} .tid {{ color:var(--muted); font-size:.78rem; font-family:ui-monospace, Consolas, monospace; margin-left:.8rem; }}
pre.ctx {{ margin:.7rem .9rem; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:.7rem; white-space:pre-wrap; overflow-wrap:anywhere; font:13.5px/1.7 ui-monospace, Consolas, monospace; }}
.field {{ display:flex; gap:.7rem; align-items:baseline; padding:.25rem .9rem; flex-wrap:wrap; }}
.field > label:first-child {{ flex:0 0 10rem; color:var(--muted); font-size:.82rem; font-family:ui-monospace, Consolas, monospace; }}
.opts {{ display:flex; gap:.35rem; }} button.v {{ font:inherit; font-size:.8rem; padding:.2rem .6rem; border:1px solid var(--line); border-radius:5px; background:var(--bg); cursor:pointer; }}
button.v.on {{ background:var(--pick); border-color:var(--pick); color:#fff; }}
textarea.t, input.t {{ flex:1 1 20rem; padding:.3rem .5rem; border:1px solid var(--line); border-radius:5px; font:inherit; font-size:.9rem; }}
#bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--panel); border-top:1px solid var(--line); padding:.7rem 1.25rem; display:flex; gap:1rem; align-items:center; }}
button.act {{ font:inherit; padding:.45rem 1rem; border:1px solid var(--line); border-radius:6px; background:var(--bg); cursor:pointer; }}
</style></head><body>
<h1>NER 召回复核 · {reviewer} · {len(tasks)} 段</h1>
<p>生成于 {datetime.now(timezone.utc).date().isoformat()} · 协议 {ARTIFACT_ID} · 评审人 {reviewer}</p>
<pre class="guide">{guide}</pre>
{''.join(cards)}
<div id="bar"><span id="count"></span><button class="act" id="save">下载 CSV</button><span style="color:var(--muted);font-size:.85rem">进度自动存在这个浏览器里，可以关掉再回来。</span></div>
<script>
const REVIEWER = {json.dumps(reviewer)};
const KEY = "ner-recall-audit-" + REVIEWER;
function readTask(el) {{
  const v = {{ task_id: el.dataset.task }};
  v.no_place_or_language_mentions = el.querySelector("input.none").checked ? "TRUE" : "FALSE";
  v.mentions = el.querySelector("textarea.mentions").value;
  const on = el.querySelector("button.v.on"); v.confidence_1_to_5 = on ? on.dataset.value : "";
  v.notes = el.querySelector("input.notes").value;
  return v;
}}
function complete(v) {{ return v.confidence_1_to_5 !== "" && (v.no_place_or_language_mentions === "TRUE" || v.mentions.trim() !== ""); }}
function collect() {{ return Array.from(document.querySelectorAll("section.task")).map(readTask); }}
function refresh() {{
  let done = 0;
  document.querySelectorAll("section.task").forEach(el => {{ const ok = complete(readTask(el)); el.classList.toggle("done", ok); if (ok) done++; }});
  document.getElementById("count").textContent = done + " / " + document.querySelectorAll("section.task").length + " 已完成";
  try {{ localStorage.setItem(KEY, JSON.stringify(collect())); }} catch (e) {{}}
}}
document.querySelectorAll("button.v").forEach(b => b.addEventListener("click", () => {{
  b.parentElement.querySelectorAll("button.v").forEach(o => o.classList.remove("on")); b.classList.add("on"); refresh();
}}));
document.addEventListener("input", refresh);
document.addEventListener("change", refresh);
(function restore() {{
  try {{
    const saved = JSON.parse(localStorage.getItem(KEY) || "null"); if (!saved) return;
    const by = Object.fromEntries(saved.map(v => [v.task_id, v]));
    document.querySelectorAll("section.task").forEach(el => {{
      const v = by[el.dataset.task]; if (!v) return;
      el.querySelector("input.none").checked = v.no_place_or_language_mentions === "TRUE";
      el.querySelector("textarea.mentions").value = v.mentions || "";
      el.querySelector("input.notes").value = v.notes || "";
      el.querySelectorAll("button.v").forEach(b => b.classList.toggle("on", b.dataset.value === v.confidence_1_to_5));
    }});
  }} catch (e) {{}}
  refresh();
}})();
function csvCell(s) {{ return '"' + String(s).replace(/"/g, '""') + '"'; }}
document.getElementById("save").addEventListener("click", () => {{
  const rows = [["task_id","reviewer","no_place_or_language_mentions","mentions","confidence_1_to_5","notes"]];
  collect().forEach(v => rows.push([v.task_id, REVIEWER, v.no_place_or_language_mentions, v.mentions.replace(/\\r?\\n/g, " | "), v.confidence_1_to_5, v.notes]));
  const text = rows.map(r => r.map(csvCell).join(",")).join("\\r\\n") + "\\r\\n";
  const blob = new Blob([new Uint8Array([0xEF,0xBB,0xBF]), text], {{type:"text/csv"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "ner_recall_" + REVIEWER + "_filled.csv"; a.click();
}});
</script></body></html>"""


def build(private_root: Path, desktop_dir: Path, sample_size: int, out_status: Path) -> int:
    sidecar = private_root / "work" / "private-canonical-lyric-text-sidecar-v1" / "cleaned_analysis_chunks_v1.csv"
    candidates = private_root / "work" / "private-chinese-rap-ner-cultural-graph-v1" / "all_candidate_occurrences_private.csv"
    rows = [r for r in csv.DictReader(sidecar.open(encoding="utf-8")) if r["analysis_text_status"] == "eligible_clean_text"]
    frame_digest = hashlib.sha256("\n".join(r["analysis_text_sha256"] for r in rows).encode()).hexdigest()
    rng = random.Random(SEED)
    chosen = rng.sample(rows, sample_size)
    rng.shuffle(chosen)
    tasks = [{"task_id": blind_id(SEED, r["song_id"], r["chunk_id"], r["analysis_text_sha256"]),
              "song_id": r["song_id"], "chunk_id": r["chunk_id"], "analysis_text_sha256": r["analysis_text_sha256"],
              "text": r["analysis_text"]} for r in chosen]
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise SystemExit("blind ids collide")
    sampled_keys = {(t["song_id"], t["chunk_id"]) for t in tasks}
    system = [c for c in csv.DictReader(candidates.open(encoding="utf-8-sig"))
              if (c["song_id"], c["chunk_id"]) in sampled_keys and c["candidate_schema_type"] in TARGET_TYPES]

    private_out = private_root / "work" / "private-ner-recall-audit-v1"
    private_out.mkdir(parents=True, exist_ok=True)
    with (private_out / "tasks_private.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["task_id", "song_id", "chunk_id", "analysis_text_sha256", "text"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(tasks)
    with (private_out / "system_candidates_in_sample_private.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = ["task_id", "candidate_id", "candidate_source", "candidate_surface", "transformer_surface", "candidate_schema_type",
                  "agreement_state", "strict_high_consistency", "transformer_confidence", "candidate_start_char", "candidate_end_char"]
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        by_key = {(t["song_id"], t["chunk_id"]): t["task_id"] for t in tasks}
        for c in system:
            writer.writerow({**c, "task_id": by_key[(c["song_id"], c["chunk_id"])]})
    desktop_dir.mkdir(parents=True, exist_ok=True)
    for reviewer in ("R1", "R2"):
        page = render_sheet(reviewer, tasks)
        (private_out / f"NER召回复核_{reviewer}.html").write_text(page, encoding="utf-8", newline="")
        (desktop_dir / f"NER召回复核_{reviewer}.html").write_text(page, encoding="utf-8", newline="")
    readme = f"""# NER 召回复核（随机 {sample_size} 段）

目的：量系统**漏了多少**地名/语言名。之前那份（224 条）只能说系统标出来的对不对。

- `NER召回复核_R1.html`：你做。浏览器打开，按页面顶部说明做，最后「下载 CSV」，把 CSV 放回这个文件夹。
- `NER召回复核_R2.html`：发给第二位评审人（和之前 224 条的同一位最好）。两人独立，不对答案。
- 每人约 1–1.5 小时。

样本是从 NER 跑过的 21,553 段里随机抽的（种子固定），没有按长度筛，长段就是长段。
系统在这些段里找到了什么，你们看不到；做完由程序去比。
含歌词原文，私有，不上传。
"""
    (desktop_dir / "README.md").write_text(readme, encoding="utf-8", newline="")
    status = {
        "artifact_id": ARTIFACT_ID, "version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "recall of the NER for PLACE and LANGUAGE_OR_DIALECT_REFERENCE mentions on the NER's own input frame",
        "design": {"frame": "eligible clean-text chunks of the canonical lyric sidecar v1 (the frame the NER ran on)",
                   "frame_chunks": len(rows), "frame_digest_sha256": frame_digest, "sample_chunks": sample_size,
                   "sampling": "uniform without replacement, seeded; task order shuffled by the same seed; no length filter",
                   "seed": SEED, "reviewers": ["R1", "R2"], "independence": "each reviewer marks every mention without "
                   "seeing the NER's candidates or the other reviewer",
                   "target_types": list(TARGET_TYPES),
                   "scoring": "after adjudication, each human mention is matched to the NER candidates of its chunk by "
                              "surface and line; recall reported for the lexicon arm, the transformer arm, and the "
                              "strict agreement gate; precision on the same chunks as a by-product"},
        "system_candidates_in_sample": {"total": len(system),
                                        "by_agreement_state": dict(sorted(
                                            __import__("collections").Counter(c["agreement_state"] for c in system).items()))},
        "sample_text_characters": sum(len(t["text"]) for t in tasks),
        "status": "awaiting two independent reviewers",
        "privacy": "tasks, text and candidates stay under the private root and the author's desktop; this file carries counts only",
    }
    out_status.parent.mkdir(parents=True, exist_ok=True)
    out_status.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"{len(rows):,} eligible chunks; sampled {sample_size}; {len(system)} system candidates of the target types in the sample")
    print(f"wrote {desktop_dir} and {private_out}; status {out_status}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--desktop-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--status", type=Path, default=REPO_ROOT / "results" / "ner-v1" / "recall_audit_status.json")
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.desktop_dir, args.sample, args.status)


if __name__ == "__main__":
    sys.exit(main())
