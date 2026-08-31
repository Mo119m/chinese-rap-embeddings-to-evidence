#!/usr/bin/env python3
"""Render the released-claim NER review CSVs as a reviewer sheet.

The blinded package already exists and already fixes the science: task ordering, control
placement, and the allowed categorical values are all decided in
``methods/NER_RELEASED_CLAIM_AUDIT_PROTOCOL.md`` and baked into the reviewer CSVs. This tool
changes none of that. It only renders one reviewer's CSV as a page they can work through,
because 193 occurrence rows and 31 co-mention rows with nine free columns each is a form, and
a spreadsheet is a bad form: it invites out-of-vocabulary values, silent row reordering, and
skipped required fields, all of which corrupt a blinded review that cannot be redone.

What the page enforces, taken straight from the protocol:

* only the allowed categorical values, offered as buttons rather than typed;
* ``exclusion_reason`` required as soon as ``mention_valid=NO`` or ``boundary_valid=NO``;
* row order, task ids, contexts and target markers preserved exactly as the CSV has them.

The output is a CSV with the same columns and the same row order as the input, so it drops
straight back into the package's own adjudication step.

The sheet carries copyrighted lyric contexts and source locators. It is a private working
file. It must never be written into the repository, published, emailed, or uploaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = REPO_ROOT / "methods" / "NER_RELEASED_CLAIM_AUDIT_PROTOCOL.md"

OCCURRENCE_FIELDS = {
    "mention_valid": ["YES", "NO", "UNCERTAIN"],
    "boundary_valid": ["YES", "NO", "UNCERTAIN"],
    "referential_status": ["NAMED_REFERENCE", "LANGUAGE_REFERENCE", "GENERIC_OR_COMMON",
                           "FIGURATIVE_OR_METONYMIC", "CREDIT_OR_METADATA", "OTHER", "UNCERTAIN"],
    "entity_type_decision": ["PLACE", "LANGUAGE_OR_DIALECT_REFERENCE", "NOT_ENTITY", "UNCERTAIN"],
    "confidence_1_to_5": ["1", "2", "3", "4", "5"],
}
PAIR_FIELDS = {
    "entity_a_has_valid_reference": ["YES", "NO", "UNCERTAIN"],
    "entity_b_has_valid_reference": ["YES", "NO", "UNCERTAIN"],
    "pair_semantically_supported": ["YES", "NO", "UNCERTAIN"],
    "confidence_1_to_5": ["1", "2", "3", "4", "5"],
}

GUIDE = """\
每一条给你一段歌词上下文，里面有一个被标出来的词。你要判断这个词在**这段上下文里**
是不是真的在指它被标称的那个东西。

四个必答项：

  mention_valid        这个被标出来的词，在这句里是不是一个指称表达？
                       （是在指一个地方/一种语言，而不是恰好这几个字连在一起）

  boundary_valid       标出来的范围对不对？多一个字少一个字都算 NO。

  referential_status   这是哪一类用法：
                         NAMED_REFERENCE          指名道姓地指一个地方
                         LANGUAGE_REFERENCE       指一种语言/方言
                         GENERIC_OR_COMMON        泛指、常用词，不是专名
                         FIGURATIVE_OR_METONYMIC  比喻或借代
                         CREDIT_OR_METADATA       这根本是署名/元数据，不是歌词
                         OTHER / UNCERTAIN

  entity_type_decision 这个用法表达的是什么类型：PLACE、LANGUAGE_OR_DIALECT_REFERENCE、
                       NOT_ENTITY、或 UNCERTAIN。
                       如果上下文明确支持另一个类型，就填那个，别迁就系统给的提议类型。

比喻和借代**不自动算无效**。要看它在这句里还算不算那个专名的指称，然后在备注里说明。

mention_valid 或 boundary_valid 选了 NO，就必须写 exclusion_reason。
边界错了的话，在 normalized_surface 里写出正确的最短完整形式。

confidence 1–5：1 = 很不确定，5 = 很确定。UNCERTAIN 是正当答案，不要硬猜。

不要用歌手的籍贯、生平、居住地、社会关系来判断。只看这段上下文。
不要在备注里写上下文之外的事实。
"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_tasks(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def buttons(task_id: str, field: str, values: list[str]) -> str:
    options = "".join(
        f'<button type="button" class="v" data-field="{field}" data-value="{value}">'
        f'{html.escape(value)}</button>' for value in values)
    return (f'<div class="field" data-field="{field}"><label>{field}</label>'
            f'<div class="opts">{options}</div></div>')


def render_occurrence(index: int, total: int, row: dict[str, str]) -> str:
    context = row["annotated_context"]
    target = row["target_surface"]
    # the CSV already marks the target; show the raw text so no re-marking can shift a span
    return f"""
  <section class="task occ" data-task-id="{html.escape(row['blind_task_id'])}">
    <header><span class="num">{index} / {total}</span>
      <span class="tid">{html.escape(row['blind_task_id'])}</span>
      <span class="ptype">提议类型 {html.escape(row['proposed_entity_type'])}</span></header>
    <p class="target">标出来的词：<b>{html.escape(target)}</b></p>
    <pre class="ctx">{html.escape(context)}</pre>
    {''.join(buttons(row['blind_task_id'], f, v) for f, v in OCCURRENCE_FIELDS.items())}
    <div class="field free"><label>normalized_surface</label>
      <input class="t" data-field="normalized_surface" placeholder="边界错时填正确的最短形式"></div>
    <div class="field free req-hint"><label>exclusion_reason</label>
      <input class="t" data-field="exclusion_reason" placeholder="mention_valid 或 boundary_valid 选 NO 时必填"></div>
    <div class="field free"><label>notes</label>
      <input class="t" data-field="notes" placeholder="备注（可选）"></div>
  </section>"""


def render_pair(index: int, total: int, row: dict[str, str]) -> str:
    return f"""
  <section class="task pair" data-task-id="{html.escape(row['blind_pair_task_id'])}">
    <header><span class="num">{index} / {total}</span>
      <span class="tid">{html.escape(row['blind_pair_task_id'])}</span></header>
    <div class="sides">
      <div class="side"><div class="side-head">实体 A · {html.escape(row['entity_a_surface'])}
        <span class="ptype">{html.escape(row['entity_a_type'])}</span></div>
        <pre class="ctx">{html.escape(row['entity_a_context_bundle'])}</pre></div>
      <div class="side"><div class="side-head">实体 B · {html.escape(row['entity_b_surface'])}
        <span class="ptype">{html.escape(row['entity_b_type'])}</span></div>
        <pre class="ctx">{html.escape(row['entity_b_context_bundle'])}</pre></div>
    </div>
    {''.join(buttons(row['blind_pair_task_id'], f, v) for f, v in PAIR_FIELDS.items())}
    <div class="field free"><label>notes</label>
      <input class="t" data-field="notes" placeholder="备注（可选）"></div>
  </section>"""


def render(kind: str, reviewer: str, rows: list[dict[str, str]], fieldnames: list[str],
           guide_sha: str, source_sha: str, generated_at: str) -> str:
    if kind == "occurrences":
        body = "".join(render_occurrence(i, len(rows), r) for i, r in enumerate(rows, 1))
        required = list(OCCURRENCE_FIELDS)
        id_field = "blind_task_id"
        title = f"NER 释义复核 · {reviewer} · 单条出现 ({len(rows)} 条)"
    else:
        body = "".join(render_pair(i, len(rows), r) for i, r in enumerate(rows, 1))
        required = list(PAIR_FIELDS)
        id_field = "blind_pair_task_id"
        title = f"NER 释义复核 · {reviewer} · 共现配对 ({len(rows)} 条)"

    return f"""<!doctype html>
<html lang="zh-Hans"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 :root {{ color-scheme: light dark; --line:#c8c8c8; --bg:#fff; --fg:#1a1a1a; --muted:#666;
          --panel:#f6f6f6; --pick:#1f5fb4; --warn:#b4441f; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --line:#3a3a3a; --bg:#151515; --fg:#e8e8e8; --muted:#999; --panel:#1e1e1e;
            --pick:#5c9ae8; --warn:#e0763f; }} }}
 body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:2rem 1.25rem 6rem;
        max-width:60rem; font:15px/1.65 "Segoe UI", system-ui, sans-serif; }}
 h1 {{ font-size:1.3rem; margin:0 0 .3rem; }}
 .lede {{ color:var(--muted); margin:0 0 1.2rem; }}
 pre.guide {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
        padding:1rem; white-space:pre-wrap; font:13.5px/1.7 "Segoe UI", system-ui, sans-serif; }}
 .task {{ border:1px solid var(--line); border-radius:10px; margin:1.4rem 0; padding:0 0 .8rem; }}
 .task.done {{ border-color:#3a8f5a; }}
 .task.incomplete {{ border-color:var(--warn); }}
 .task > header {{ display:flex; gap:1rem; align-items:baseline; flex-wrap:wrap;
        background:var(--panel); padding:.5rem .9rem; border-bottom:1px solid var(--line);
        border-radius:10px 10px 0 0; }}
 .num {{ font-weight:600; }}
 .tid, .ptype {{ color:var(--muted); font-size:.78rem;
        font-family:ui-monospace, Consolas, monospace; }}
 .target {{ margin:.7rem .9rem .3rem; }}
 pre.ctx {{ margin:0 .9rem .8rem; background:var(--panel); border:1px solid var(--line);
        border-radius:6px; padding:.7rem; white-space:pre-wrap; overflow-wrap:anywhere;
        font:13.5px/1.7 ui-monospace, Consolas, monospace; }}
 .sides {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(18rem,1fr)); gap:.5rem;
        margin:.7rem 0; }}
 .side-head {{ margin:0 .9rem .4rem; font-weight:600; }}
 .field {{ display:flex; gap:.7rem; align-items:baseline; padding:.2rem .9rem; flex-wrap:wrap; }}
 .field > label {{ flex:0 0 11rem; color:var(--muted); font-size:.82rem;
        font-family:ui-monospace, Consolas, monospace; }}
 .opts {{ display:flex; gap:.35rem; flex-wrap:wrap; }}
 button.v {{ font:inherit; font-size:.8rem; padding:.2rem .6rem; border:1px solid var(--line);
        border-radius:5px; background:var(--bg); color:var(--fg); cursor:pointer; }}
 button.v.on {{ background:var(--pick); border-color:var(--pick); color:#fff; }}
 input.t {{ flex:1 1 18rem; padding:.3rem .5rem; border:1px solid var(--line); border-radius:5px;
        background:var(--bg); color:var(--fg); font:inherit; font-size:.88rem; }}
 .field.needed > label {{ color:var(--warn); font-weight:600; }}
 .field.needed input.t {{ border-color:var(--warn); }}
 #bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--panel);
        border-top:1px solid var(--line); padding:.7rem 1.25rem; display:flex; gap:1rem;
        align-items:center; justify-content:center; flex-wrap:wrap; }}
 button.act {{ font:inherit; padding:.45rem 1rem; border:1px solid var(--line); border-radius:6px;
        background:var(--bg); color:var(--fg); cursor:pointer; }}
 button.act:disabled {{ opacity:.5; cursor:default; }}
 #count {{ font-variant-numeric:tabular-nums; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="lede">生成于 {html.escape(generated_at)} · 协议 NER-RELEASED-CLAIM-AUDIT ·
   评审人 {html.escape(reviewer)}</p>
<pre class="guide">{html.escape(GUIDE)}</pre>
<p class="lede"><b>这份文件含有受版权保护的歌词上下文和来源定位信息。</b>
   只能留在本机：不要提交、不要发布、不要邮件发送、不要上传。
   两位评审人独立作业，中途不要互相对答案。</p>
{body}
<div id="bar"><span id="count">0 / {len(rows)}</span>
  <button class="act" id="save" disabled>下载 CSV</button>
  <button class="act" id="copy" disabled>复制 CSV</button></div>
<script>
const REVIEWER = {json.dumps(reviewer)};
const KIND = {json.dumps(kind)};
const ID_FIELD = {json.dumps(id_field)};
const FIELDNAMES = {json.dumps(fieldnames)};
const ROWS = {json.dumps(rows, ensure_ascii=False)};
const REQUIRED = {json.dumps(required)};
const GUIDE_SHA256 = {json.dumps(guide_sha)};
const SOURCE_SHA256 = {json.dumps(source_sha)};
const TOTAL = {len(rows)};
const KEY = "ner-review-" + REVIEWER + "-" + KIND;

document.querySelectorAll("button.v").forEach(b => b.addEventListener("click", () => {{
  const wrap = b.closest(".field");
  const on = b.classList.contains("on");
  wrap.querySelectorAll("button.v").forEach(x => x.classList.remove("on"));
  if (!on) b.classList.add("on");
  refresh();
}}));
document.addEventListener("input", e => {{ if (e.target.classList.contains("t")) refresh(); }});

function readTask(el) {{
  const out = {{}};
  el.querySelectorAll(".field").forEach(f => {{
    const picked = f.querySelector("button.v.on");
    const text = f.querySelector("input.t");
    if (picked) out[picked.dataset.field] = picked.dataset.value;
    else if (text) out[text.dataset.field] = text.value.trim();
  }});
  return out;
}}

// the protocol makes a reason mandatory once a NO is recorded, so the page asks for it
// rather than letting an incomplete row reach the adjudicator
function needsReason(v) {{ return v.mention_valid === "NO" || v.boundary_valid === "NO"; }}

function complete(v) {{
  if (!REQUIRED.every(f => v[f])) return false;
  if (KIND === "occurrences" && needsReason(v) && !v.exclusion_reason) return false;
  return true;
}}

function refresh() {{
  let done = 0;
  document.querySelectorAll(".task").forEach(el => {{
    const v = readTask(el);
    const ok = complete(v);
    const touched = Object.values(v).some(x => x);
    el.classList.toggle("done", ok);
    el.classList.toggle("incomplete", touched && !ok);
    const reasonField = [...el.querySelectorAll(".field.free")]
      .find(f => f.querySelector('[data-field="exclusion_reason"]'));
    if (reasonField) reasonField.classList.toggle("needed", needsReason(v) && !v.exclusion_reason);
    if (ok) done++;
  }});
  document.getElementById("count").textContent = done + " / " + TOTAL;
  document.getElementById("save").disabled = done === 0;
  document.getElementById("copy").disabled = done === 0;
  try {{ localStorage.setItem(KEY, JSON.stringify(collect())); }} catch (e) {{}}
}}

function collect() {{
  const byId = {{}};
  document.querySelectorAll(".task").forEach(el => byId[el.dataset.taskId] = readTask(el));
  return byId;
}}

// same columns, same row order as the source CSV, so it drops back into the package
function toCsv() {{
  const answers = collect();
  const stamp = new Date().toISOString().replace(/\\.\\d+Z$/, "Z");
  const esc = s => /[",\\r\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  const lines = [FIELDNAMES.join(",")];
  ROWS.forEach(row => {{
    const a = answers[row[ID_FIELD]] || {{}};
    const merged = Object.assign({{}}, row, a);
    merged.reviewer_id = REVIEWER;
    if (Object.values(a).some(x => x)) merged.reviewed_at_utc = stamp;
    lines.push(FIELDNAMES.map(f => esc(String(merged[f] === undefined ? "" : merged[f]))).join(","));
  }});
  return lines.join("\\r\\n") + "\\r\\n";
}}

try {{
  const saved = JSON.parse(localStorage.getItem(KEY) || "null");
  if (saved) document.querySelectorAll(".task").forEach(el => {{
    const v = saved[el.dataset.taskId]; if (!v) return;
    el.querySelectorAll(".field").forEach(f => {{
      const name = f.dataset.field;
      if (name && v[name]) {{
        const b = f.querySelector('button.v[data-value="' + v[name] + '"]');
        if (b) b.classList.add("on");
      }}
      const t = f.querySelector("input.t");
      if (t && v[t.dataset.field]) t.value = v[t.dataset.field];
    }});
  }});
}} catch (e) {{}}

document.getElementById("save").addEventListener("click", () => {{
  const blob = new Blob([new Uint8Array([0xEF,0xBB,0xBF]), toCsv()], {{type:"text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "reviewer_" + REVIEWER + "_" + KIND + "_filled.csv";
  a.click(); URL.revokeObjectURL(a.href);
}});
document.getElementById("copy").addEventListener("click", async () => {{
  try {{ await navigator.clipboard.writeText(toCsv()); }}
  catch (e) {{ alert("复制失败，请用下载按钮。"); }}
}});
refresh();
</script></body></html>
"""


def validate(source: Path, filled: Path, kind: str) -> int:
    """Check a returned sheet against the blinded source it came from.

    The page writes this CSV in the browser, and a browser is not a place to take on trust
    that 193 rows of irreplaceable review survived a round trip. This checks the properties
    the adjudication step depends on: same columns, same rows in the same order, same task
    ids, unedited contexts, only allowed values, and no half-filled row. Run it on the
    download before the sheet is treated as locked.
    """
    fields = OCCURRENCE_FIELDS if kind == "occurrences" else PAIR_FIELDS
    id_field = "blind_task_id" if kind == "occurrences" else "blind_pair_task_id"
    frozen = (["annotated_context", "target_surface", "proposed_entity_type"]
              if kind == "occurrences"
              else ["entity_a_surface", "entity_b_surface",
                    "entity_a_context_bundle", "entity_b_context_bundle"])

    original, original_fields = read_tasks(source)
    returned, returned_fields = read_tasks(filled)
    problems: list[str] = []

    if returned_fields != original_fields:
        problems.append(f"columns changed: {set(original_fields) ^ set(returned_fields)}")
    if len(returned) != len(original):
        problems.append(f"row count changed: {len(original)} -> {len(returned)}")
    else:
        for position, (before, after) in enumerate(zip(original, returned), start=1):
            if before[id_field] != after[id_field]:
                problems.append(f"row {position}: task id changed, or rows were reordered")
                continue
            for column in frozen:
                if before.get(column, "") != after.get(column, ""):
                    problems.append(f"row {position} ({before[id_field]}): {column} was edited")

    complete = 0
    for position, row in enumerate(returned, start=1):
        answered = {name: row.get(name, "").strip() for name in fields}
        for name, allowed in fields.items():
            if answered[name] and answered[name] not in allowed:
                problems.append(
                    f"row {position}: {name}={answered[name]!r} is not an allowed value")
        filled_count = sum(1 for value in answered.values() if value)
        if filled_count == 0:
            continue
        if filled_count < len(fields):
            missing = [name for name, value in answered.items() if not value]
            problems.append(f"row {position}: partly filled, missing {missing}")
            continue
        if kind == "occurrences" and "NO" in (answered["mention_valid"],
                                              answered["boundary_valid"]):
            if not row.get("exclusion_reason", "").strip():
                problems.append(f"row {position}: a NO decision needs an exclusion_reason")
                continue
        complete += 1

    print(f"{filled.name}: {complete} of {len(returned)} tasks complete")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems[:40]:
            print(f"  {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    print("row order, task ids, contexts and allowed values all check out")
    return 0


def self_test() -> int:
    import tempfile

    failures: list[str] = []

    def check(label: str, condition: bool) -> None:
        print(f"  {'ok  ' if condition else 'FAIL'} {label}")
        if not condition:
            failures.append(label)

    header = ["review_order", "blind_task_id", "annotated_context", "target_surface",
              "proposed_entity_type", *OCCURRENCE_FIELDS, "normalized_surface",
              "exclusion_reason", "notes", "reviewer_id", "reviewed_at_utc"]

    def write(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\r\n")
            writer.writeheader()
            writer.writerows(rows)

    base = [{**{name: "" for name in header}, "review_order": str(index),
             "blind_task_id": f"T{index}", "annotated_context": f"ctx {index}",
             "target_surface": "x", "proposed_entity_type": "PLACE"} for index in (1, 2)]
    good = [{**row, "mention_valid": "YES", "boundary_valid": "YES",
             "referential_status": "NAMED_REFERENCE", "entity_type_decision": "PLACE",
             "confidence_1_to_5": "5", "reviewer_id": "R1"} for row in base]

    print("validate")
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        write(tmp / "src.csv", base)

        write(tmp / "ok.csv", good)
        check("a complete, faithful sheet passes",
              validate(tmp / "src.csv", tmp / "ok.csv", "occurrences") == 0)

        write(tmp / "reordered.csv", list(reversed(good)))
        check("reordered rows are caught",
              validate(tmp / "src.csv", tmp / "reordered.csv", "occurrences") == 1)

        write(tmp / "edited.csv", [{**good[0], "annotated_context": "tampered"}, good[1]])
        check("an edited context is caught",
              validate(tmp / "src.csv", tmp / "edited.csv", "occurrences") == 1)

        write(tmp / "bad_value.csv", [{**good[0], "mention_valid": "MAYBE"}, good[1]])
        check("an out-of-vocabulary value is caught",
              validate(tmp / "src.csv", tmp / "bad_value.csv", "occurrences") == 1)

        write(tmp / "partial.csv", [{**good[0], "confidence_1_to_5": ""}, good[1]])
        check("a half-filled row is caught",
              validate(tmp / "src.csv", tmp / "partial.csv", "occurrences") == 1)

        write(tmp / "no_reason.csv", [{**good[0], "mention_valid": "NO"}, good[1]])
        check("a NO with no exclusion_reason is caught",
              validate(tmp / "src.csv", tmp / "no_reason.csv", "occurrences") == 1)

        write(tmp / "with_reason.csv",
              [{**good[0], "mention_valid": "NO", "exclusion_reason": "generic use"}, good[1]])
        check("a NO with a reason passes",
              validate(tmp / "src.csv", tmp / "with_reason.csv", "occurrences") == 0)

        write(tmp / "blank.csv", base)
        check("an untouched sheet passes with zero complete",
              validate(tmp / "src.csv", tmp / "blank.csv", "occurrences") == 0)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all reviewer-sheet self-tests passed")
    return 0


def build(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    if out_dir.is_relative_to(REPO_ROOT):
        raise SystemExit(f"--out-dir must not resolve inside the repository: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    guide_sha = sha256_text(GUIDE)
    written: list[dict[str, Any]] = []
    for reviewer in args.reviewers:
        for kind, stem in (("occurrences", "occurrences"), ("co_mentions", "co_mentions")):
            source = args.package / f"reviewer_{reviewer}_{stem}_private.csv"
            if not source.is_file():
                raise SystemExit(f"missing reviewer sheet: {source}")
            rows, fieldnames = read_tasks(source)
            source_sha = sha256_text(source.read_text(encoding="utf-8-sig"))
            page = render(kind, reviewer, rows, fieldnames, guide_sha, source_sha,
                          args.generated_at)
            target = out_dir / f"NER复核_{reviewer}_{'单条出现' if kind == 'occurrences' else '共现配对'}.html"
            target.write_text(page, encoding="utf-8", newline="")
            written.append({"reviewer": reviewer, "kind": kind, "tasks": len(rows),
                            "source": source.name, "source_sha256": source_sha,
                            "page": target.name})
            print(f"  {reviewer} {kind}: {len(rows)} tasks -> {target.name}")

    (out_dir / "ner_review_sheet_manifest.json").write_text(
        json.dumps({"generated_at": args.generated_at, "guide_sha256": guide_sha,
                    "protocol_sha256": sha256_text(PROTOCOL.read_text(encoding="utf-8")),
                    "warning": "private; contains copyrighted lyric contexts",
                    "sheets": written}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", nargs=3, metavar=("SOURCE_CSV", "FILLED_CSV", "KIND"),
                        help="check a returned sheet against its blinded source")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--reviewers", nargs="+", default=["R1", "R2"])
    parser.add_argument("--generated-at")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        sys.exit(self_test())
    if arguments.validate:
        source_csv, filled_csv, kind = arguments.validate
        if kind not in ("occurrences", "co_mentions"):
            raise SystemExit("KIND must be 'occurrences' or 'co_mentions'")
        sys.exit(validate(Path(source_csv), Path(filled_csv), kind))
    for required in ("package", "out_dir", "generated_at"):
        if getattr(arguments, required) is None:
            raise SystemExit(f"--{required.replace(chr(95), chr(45))} is required unless "
                             "--self-test or --validate is given")
    build(arguments)
