#!/usr/bin/env python3
"""Score the NER recall audit once both reviewers' CSVs are back.

Reads the two filled sheets, checks them against the task table (row set, task ids), parses
each reviewer's mentions ('行号 词 类型' per entry), measures agreement between reviewers,
forms the adjudicated set (mentions both reviewers marked, plus a private list of the
disagreements for a third decision), and matches every adjudicated mention to the NER
candidates of its chunk to report recall for each arm and for the strict agreement gate,
with precision on the same chunks as a by-product. Public output: counts and rates only.

    python tools/score_ner_recall_audit_v1.py --private-root <ni-k> --r1 <csv> --r2 <csv> [--adjudicated <csv>]
    python tools/score_ner_recall_audit_v1.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
MENTION = re.compile(r"^\s*(\d+)\s+(\S+?)\??\s+(地名|语言)\s*$")
TYPE_OF = {"地名": "PLACE", "语言": "LANGUAGE_OR_DIALECT_REFERENCE"}
ARMS = {
    "lexicon_arm": {"LEXICON_ONLY", "EXACT_SPAN_TYPE_AGREE", "OVERLAP_TYPE_AGREE", "TYPE_OR_BOUNDARY_CONFLICT"},
    "transformer_arm": {"TRANSFORMER_ONLY", "EXACT_SPAN_TYPE_AGREE", "OVERLAP_TYPE_AGREE", "TYPE_OR_BOUNDARY_CONFLICT"},
    "strict_agreement_gate": {"EXACT_SPAN_TYPE_AGREE"},
}
csv.field_size_limit(10 ** 9)


def parse_mentions(cell: str) -> list[tuple[int, str, str]]:
    out = []
    for entry in re.split(r"\s*\|\s*|\r?\n", cell or ""):
        m = MENTION.match(entry)
        if m:
            out.append((int(m.group(1)), m.group(2), TYPE_OF[m.group(3)]))
    return out


def load_sheet(path: Path, tasks: set[str]) -> dict[str, dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    ids = [r["task_id"] for r in rows]
    if set(ids) != tasks or len(ids) != len(tasks):
        raise SystemExit(f"{path.name}: task ids do not match the package")
    return {r["task_id"]: {"none": r["no_place_or_language_mentions"].strip().upper() == "TRUE",
                           "mentions": parse_mentions(r["mentions"]), "confidence": r.get("confidence_1_to_5", "")}
            for r in rows}


def key_of(m: tuple[int, str, str]) -> tuple[int, str, str]:
    return (m[0], m[1].lower(), m[2])


def score(tasks: dict[str, dict], r1: dict, r2: dict, adjudicated: dict[str, set] | None,
          candidates: dict[str, list[dict]]) -> dict:
    # agreement between reviewers, per mention key
    both, only1, only2 = 0, 0, 0
    disagreements = []
    union_by_task: dict[str, set] = {}
    for t in tasks:
        a = {key_of(m) for m in r1[t]["mentions"]}
        b = {key_of(m) for m in r2[t]["mentions"]}
        both += len(a & b)
        only1 += len(a - b)
        only2 += len(b - a)
        for k in sorted((a ^ b)):
            disagreements.append({"task_id": t, "line": k[0], "surface": k[1], "type": k[2],
                                  "marked_by": "R1" if k in a else "R2"})
        union_by_task[t] = (a & b) if adjudicated is None else ((a & b) | adjudicated.get(t, set()))
    total_marked = both + only1 + only2
    agreement = {"mentions_marked_by_both": both, "only_R1": only1, "only_R2": only2,
                 "jaccard": round(both / max(both + only1 + only2, 1), 4),
                 "adjudication": "both-reviewer mentions only" if adjudicated is None else "both plus adjudicated disagreements"}

    # match adjudicated mentions to system candidates: same chunk, same type, surface equal
    # (case-insensitive) or one containing the other (boundary differences), on the same line
    recall = {arm: {"found": 0, "total": 0} for arm in ARMS}
    precision = {arm: {"true": 0, "proposed": 0} for arm in ARMS}
    for t, mentions in union_by_task.items():
        lines = tasks[t]["text"].split("\n")
        cands = candidates.get(t, [])
        for line_no, surface, etype in mentions:
            line = lines[line_no - 1].lower() if 0 < line_no <= len(lines) else ""
            for arm, states in ARMS.items():
                recall[arm]["total"] += 1
                hit = any(c["candidate_schema_type"] == etype and c["agreement_state"] in states
                          and (c["candidate_surface"].lower() in surface or surface in c["candidate_surface"].lower())
                          and c["candidate_surface"].lower() in line for c in cands)
                recall[arm]["found"] += int(hit)
        for c in cands:
            for arm, states in ARMS.items():
                if c["agreement_state"] not in states:
                    continue
                precision[arm]["proposed"] += 1
                s = c["candidate_surface"].lower()
                precision[arm]["true"] += int(any(c["candidate_schema_type"] == etype and (s in surface or surface in s)
                                                  for _, surface, etype in mentions))
    for arm in ARMS:
        recall[arm]["recall"] = round(recall[arm]["found"] / max(recall[arm]["total"], 1), 4)
        precision[arm]["precision"] = round(precision[arm]["true"] / max(precision[arm]["proposed"], 1), 4)
    return {"agreement": agreement, "adjudicated_mentions": sum(len(v) for v in union_by_task.values()),
            "recall_by_arm": recall, "precision_on_sample_by_arm": precision, "disagreements": disagreements}


def self_test() -> int:
    tasks = {"T1": {"text": "我从上海到西安\n讲粤语\n没有地名的一行"}}
    r1 = {"T1": {"none": False, "mentions": parse_mentions("1 上海 地名 | 1 西安 地名 | 2 粤语 语言"), "confidence": "5"}}
    r2 = {"T1": {"none": False, "mentions": parse_mentions("1 上海 地名 | 2 粤语 语言 | 3 地名 地名"), "confidence": "4"}}
    cands = {"T1": [{"candidate_surface": "上海", "candidate_schema_type": "PLACE", "agreement_state": "EXACT_SPAN_TYPE_AGREE"},
                    {"candidate_surface": "粤语", "candidate_schema_type": "LANGUAGE_OR_DIALECT_REFERENCE", "agreement_state": "LEXICON_ONLY"}]}
    out = score(tasks, r1, r2, None, cands)
    assert out["agreement"]["mentions_marked_by_both"] == 2 and out["agreement"]["only_R1"] == 1 and out["agreement"]["only_R2"] == 1
    assert out["recall_by_arm"]["strict_agreement_gate"] == {"found": 1, "total": 2, "recall": 0.5}, out["recall_by_arm"]
    assert out["recall_by_arm"]["lexicon_arm"]["recall"] == 1.0
    assert out["precision_on_sample_by_arm"]["lexicon_arm"]["precision"] == 1.0
    out2 = score(tasks, r1, r2, {"T1": {(1, "西安", "PLACE")}}, cands)
    assert out2["adjudicated_mentions"] == 3 and out2["recall_by_arm"]["lexicon_arm"]["recall"] == round(2 / 3, 4)
    print("self-test ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--r1", type=Path)
    parser.add_argument("--r2", type=Path)
    parser.add_argument("--adjudicated", type=Path, help="CSV of disagreements with a 'keep' column (TRUE/FALSE)")
    parser.add_argument("--status", type=Path, default=REPO_ROOT / "results" / "ner-v1" / "recall_audit_status.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not (args.private_root and args.r1 and args.r2):
        raise SystemExit("--private-root, --r1 and --r2 are required")
    private = args.private_root / "work" / "private-ner-recall-audit-v1"
    tasks = {r["task_id"]: r for r in csv.DictReader((private / "tasks_private.csv").open(encoding="utf-8"))}
    candidates: dict[str, list[dict]] = defaultdict(list)
    for c in csv.DictReader((private / "system_candidates_in_sample_private.csv").open(encoding="utf-8")):
        candidates[c["task_id"]].append(c)
    r1 = load_sheet(args.r1, set(tasks))
    r2 = load_sheet(args.r2, set(tasks))
    adjudicated = None
    if args.adjudicated:
        adjudicated = defaultdict(set)
        for r in csv.DictReader(args.adjudicated.open(encoding="utf-8-sig")):
            if r.get("keep", "").strip().upper() == "TRUE":
                adjudicated[r["task_id"]].add((int(r["line"]), r["surface"].lower(), r["type"]))
    out = score(tasks, r1, r2, adjudicated, candidates)
    (private / "disagreements_for_adjudication.csv").write_text(
        "task_id,line,surface,type,marked_by,keep\n" + "".join(
            f"{d['task_id']},{d['line']},{d['surface']},{d['type']},{d['marked_by']},\n" for d in out["disagreements"]),
        encoding="utf-8-sig", newline="")
    status = json.loads(args.status.read_text(encoding="utf-8")) if args.status.is_file() else {}
    status.update({"status": "scored" if adjudicated is not None else "scored on both-reviewer mentions; adjudication pending",
                   "scored_at_utc": datetime.now(timezone.utc).isoformat(),
                   "reviewer_agreement": out["agreement"], "adjudicated_mentions": out["adjudicated_mentions"],
                   "recall_by_arm": out["recall_by_arm"], "precision_on_sample_by_arm": out["precision_on_sample_by_arm"]})
    args.status.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(json.dumps({k: out[k] for k in ("agreement", "adjudicated_mentions", "recall_by_arm", "precision_on_sample_by_arm")},
                     ensure_ascii=False, indent=1))
    print(f"disagreements for adjudication: {len(out['disagreements'])} -> {private / 'disagreements_for_adjudication.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
