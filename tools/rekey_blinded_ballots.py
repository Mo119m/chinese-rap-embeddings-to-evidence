"""Fold raw blinded ballots into the canonical adjudication file (protocol NER-CR-001).

Adjudicators see only random blind identifiers. This script — not a judgement call — maps
them back to canonical occurrence uids, applies the frozen escalation rule (a record whose
first ruling is not high-confidence, or is DEFER, receives two further rulings; the
recorded verdict is the majority of the three), and emits the deterministic adjudication
file whose hash the freeze pins. Every input is validated; nothing here decides a verdict.

    python tools/rekey_blinded_ballots.py --ballots DIR --map blind_id_map.json --out FILE
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
VERDICTS = ("RETAIN_SHORT", "DROP_SHORT", "DEFER")
RELATIONS = ("TAGGER_ARTEFACT", "SAME_REFERENT", "DISTINCT_ENTITY", "UNKNOWN")
REGIONS = ("LYRIC", "PRODUCTION_CREDIT", "UNKNOWN")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ballots", type=Path, required=True,
                        help="directory of round1_batch_*.json and escalation_*.json ballots")
    parser.add_argument("--map", dest="mapping", type=Path, required=True,
                        help="frozen blind_id -> canonical 64-hex occurrence uid map")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="frozen ballot manifest: filename -> sha256")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--blinded-input", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    problems: list[str] = []

    # The chain is verified before a single ballot is read: every file the manifest names
    # must exist with the recorded bytes, and no unlisted ballot may sit in the directory.
    # Verifying only the manifest against itself proves nothing about the ballots.
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    present = {path.name for path in args.ballots.glob("*.json")}
    for name, digest in sorted(manifest.items()):
        path = args.ballots / name
        if not path.is_file():
            problems.append(f"ballot named in the manifest is missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"ballot {name} does not match its manifest hash")
    for extra in sorted(present - set(manifest)):
        problems.append(f"ballot directory holds a file the manifest does not name: {extra}")

    # the rest of the chain: instructions, blinded input, batch assignment
    batches = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    blinded = json.loads(args.blinded_input.read_text(encoding="utf-8"))
    if not args.prompt.is_file():
        problems.append(f"missing frozen instructions: {args.prompt}")
    blinded_ids = {record["blind_id"] for record in blinded}
    batched_ids = {identifier for ids in batches.values() for identifier in ids}
    if blinded_ids != set(mapping):
        problems.append("the blinded input and the blind-id map cover different ids")
    if batched_ids != blinded_ids:
        problems.append("the batch manifest and the blinded input cover different ids")
    if sum(len(ids) for ids in batches.values()) != len(batched_ids):
        problems.append("an id appears in more than one batch")
    if len(set(mapping.values())) != len(mapping):
        problems.append("the blind-id map is not injective")
    for uid in mapping.values():
        if len(uid) != 64 or any(c not in "0123456789abcdef" for c in uid):
            problems.append(f"mapped uid is not a full sha256 hex digest: {uid!r}")
            break

    first: dict[str, dict] = {}
    extras: dict[str, list[dict]] = collections.defaultdict(list)
    sources: dict[str, list[str]] = collections.defaultdict(list)
    round_one = sorted(args.ballots.glob("round1_batch_*.json"))
    escalation = sorted(args.ballots.glob("escalation_*.json"))
    if not round_one:
        print(f"no round-one ballots under {args.ballots}", file=sys.stderr)
        return 2
    for path in round_one:
        for ruling in json.loads(path.read_text(encoding="utf-8")):
            blind = ruling["blind_id"]
            if blind in first:
                problems.append(f"{blind}: ruled twice in round one")
            if blind not in mapping:
                problems.append(f"round-one ballot carries an unknown blind id: {blind}")
                continue
            first[blind] = ruling
            sources[blind].append(path.name)
    for path in escalation:
        for ruling in json.loads(path.read_text(encoding="utf-8")):
            if ruling["blind_id"] not in mapping:
                problems.append(f"escalation ballot carries an unknown blind id: {ruling['blind_id']}")
                continue
            extras[ruling["blind_id"]].append(ruling)
            sources[ruling["blind_id"]].append(path.name)

    if set(first) != set(mapping):
        problems.append(f"ballots cover {len(first)} blind ids, the map has {len(mapping)}; "
                        f"difference {sorted(set(first) ^ set(mapping))[:5]}")
    for blind, ruling in first.items():
        escalated = ruling["confidence"] != "high" or ruling["verdict"] == "DEFER"
        got = len(extras.get(blind, []))
        if escalated and got != 2:
            problems.append(f"{blind}: escalation required but {got} further rulings found")
        if not escalated and got:
            problems.append(f"{blind}: {got} unexpected extra rulings on a high-confidence record")
        for candidate in [ruling, *extras.get(blind, [])]:
            if candidate["verdict"] not in VERDICTS or candidate["span_relation"] not in RELATIONS \
                    or candidate["text_region"] not in REGIONS:
                problems.append(f"{blind}: a ruling carries a value outside the protocol")
    if problems:
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        print("refusing to fold: ballots do not satisfy the frozen procedure", file=sys.stderr)
        return 1

    final = {}
    for blind, ruling in first.items():
        ballots = [ruling, *extras.get(blind, [])]
        votes = [b["verdict"] for b in ballots]
        # Each field is aggregated under its own declared rule. Inheriting the span
        # relation and the text region from whichever ballot happened to carry the
        # majority verdict let one field decide two others.
        def majority(values: list[str]) -> str:
            counted = collections.Counter(values)
            top = max(counted.values())
            winners = sorted(name for name, count in counted.items() if count == top)
            return winners[0] if len(winners) == 1 else "UNRESOLVED"

        winner = majority(votes)
        relation = majority([b["span_relation"] for b in ballots])
        region = majority([b["text_region"] for b in ballots])
        uid = mapping[blind]
        final[uid] = {
            "occurrence_uid": uid,
            "verdict": winner,
            "span_relation": relation,
            "text_region": region,
            "confidence": majority([b["confidence"] for b in ballots]),
            "votes": votes,
            "span_relation_votes": [b["span_relation"] for b in ballots],
            "text_region_votes": [b["text_region"] for b in ballots],
            "rationale": next(b["rationale"] for b in ballots if b["verdict"] == winner)
                         if winner in votes else ballots[0]["rationale"],
            "ballots": sorted(sources[blind]),
        }
    payload = json.dumps(final, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    args.out.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    spread = collections.Counter(v["verdict"] for v in final.values())
    print(f"{len(final)} adjudications folded from {len(round_one)} round-one and "
          f"{len(escalation)} escalation ballots")
    print(f"spread {dict(spread)}; three-vote records "
          f"{sum(1 for v in final.values() if len(v['votes']) == 3)}")
    print(f"adjudication sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
