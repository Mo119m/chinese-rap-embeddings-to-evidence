"""Rebuild the compound resolution flag stage and occurrence ledger (protocol NER-CR-001).

The release's lexicon stage cannot separate a surface from a longer compound that contains
it: `boundary_ok()` returns True unconditionally for surfaces with no ASCII, and the
exact-span agreement with the release's own transformer does not catch these either -- it
agreed at iou 1.0 on every compound occurrence found here.

Two stages, kept apart on purpose:

  flag stage      automatic, high recall, decides nothing. Every lexicon candidate whose
                  span is strictly contained by an entity span proposed by either pinned
                  reference tagger is flagged, with the proposing model recorded.

  ledger          one adjudicated verdict per flagged occurrence, supplied from outside as
                  `--verdicts`. This script never derives a verdict, and never copies a
                  combination-level decision down onto its occurrences: an earlier version
                  did exactly that, which made the "occurrences of a combination agree"
                  check true by construction and unable to detect the case it existed to
                  detect. A combination is folded into one public row only when its
                  occurrences were independently adjudicated the same way.

`--check` rebuilds the whole private flag table and the whole ledger and compares their
SHA-256 against values hardcoded below, not read from `freeze.json`. It needs the private
candidate table, the verdicts, the frozen id map, and both models.

    python tools/build_compound_resolution_table.py --check --verdicts ... --ids ...
    python tools/build_compound_resolution_table.py --write-private /path/outside/the/repo ...
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import sys
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "analysis" / "compound-resolution"
csv.field_size_limit(10 ** 8)

PROTOCOL = "NER-CR-001"
FRAME_SIZE = 1098
TOTALS = {"flagged_occurrences": 91, "flagged_strict_occurrences": 39, "resolution_rows": 63}
MODELS = {
    "shibing624/bert4ner-base-chinese": "5d660ed2aa9da482bf2d99c6bc8cf2ce66758f6a",
    "uer/roberta-base-finetuned-cluener2020-chinese": "cddd8fc233e373855a8c0a7f4b7eb83acb686a2b",
}
CANDIDATE_TABLE_SHA = "3ac7a41cc971e34b2314c87d3c50c11c95af039e278b851cc144a8c553daa5e4"
GRAPH_UNIVERSE_SHA = "7a09b319c2bd1449c671c1d85c2bc1ef54fa4b18ac03130e2d3883332a4342b3"
INVENTORY_SHA = "6ab77221ca632d3117c6e4287305354fb8fa1266ffdde71e24d1f6e4b6535cb4"
PRIVATE_FLAG_SHA = "e711face34cb4e2ce1ecc96712255a51d73be133ad301bfb297f7ac83c090a8b"
PRIVATE_LEDGER_SHA = "1a1cde4820915b1bfadef6a4f90cb78aeb1f0e4e54ee9a856ce21ca2e7146d0d"
PRIVATE_FOLD_SHA = "83b0bc299dd0caa2abbc0635046af81d39bcbf13d365b54c5f91d4a8508132aa"
VERDICTS_SHA = "f4c3da5d6e053c7220f6b3566c56c5d05d7c75c828fac225e77ac556a6980919"

UNIT = chr(31)
VERDICTS = ("RETAIN_SHORT", "DROP_SHORT", "DEFER")
FLAG_COLUMNS = ["short_surface", "containing_span", "model_entity_group", "proposing_models",
                "flagging_models", "flagged_occurrences", "flagged_strict_occurrences",
                "distinct_source_labels"]
LEDGER_COLUMNS = ["occurrence_uid", "combination_id", "short_surface", "containing_span",
                  "model_entity_group", "proposing_models", "flagging_models",
                  "source_credit_label", "song_id",
                  "strict_high_consistency", "duplicate_text_key", "verdict", "span_relation",
                  "text_region", "confidence", "votes", "rationale"]
VOTES = ("RETAIN_SHORT", "DROP_SHORT", "DEFER")
RELATIONS = ("TAGGER_ARTEFACT", "SAME_REFERENT", "DISTINCT_ENTITY", "UNKNOWN")
REGIONS = ("LYRIC", "PRODUCTION_CREDIT", "UNKNOWN")


def compare_pinned(name: str, actual: str, pinned: str, failures: list[str]) -> None:
    """Compare a computed hash against its pin; PENDING means not yet frozen."""
    if pinned != "PENDING" and actual != pinned:
        failures.append(f"{name} hash {actual} != pinned {pinned}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def private_key(surface: str, span: str, group: str) -> str:
    return hashlib.sha256((surface + UNIT + span + UNIT + group).encode("utf-8")).hexdigest()


def render(columns: list[str], rows: list[list[object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "work/private-chinese-rap-ner-cultural-graph-v1/all_candidate_occurrences_private.csv")
    parser.add_argument("--verdicts", type=Path, required=True,
                        help="independent per-occurrence adjudication, keyed by occurrence_uid")
    parser.add_argument("--ids", type=Path, required=True,
                        help="frozen combination id map: sha256(private key) -> opaque id")
    parser.add_argument("--check", action="store_true",
                        help="rebuild the private flag table and ledger and compare whole-table hashes")
    parser.add_argument("--provenance-dir", type=Path, default=None,
                        help="directory holding the frozen blinded-adjudication inputs: prompt, "
                             "input, batch manifest, blind-id map and ballot manifest")
    parser.add_argument("--write-private", type=Path, default=None,
                        help="write the unredacted flag table and ledger here; must be outside the repository")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    freeze = json.loads((DIR / "freeze.json").read_text(encoding="utf-8"))
    if freeze.get("protocol") != PROTOCOL:
        print(f"freeze record is not {PROTOCOL}", file=sys.stderr)
        return 2

    if args.write_private is not None:
        resolved = args.write_private.resolve()
        if ROOT in resolved.parents or resolved == ROOT:
            print("refusing to write unredacted output inside the repository", file=sys.stderr)
            return 2

    for path, expected, what in ((args.candidates, CANDIDATE_TABLE_SHA, "private candidate table"),
                                 (ROOT / "results/ner-v1/graph_label_universe.csv", GRAPH_UNIVERSE_SHA, "graph label universe"),
                                 (ROOT / "results/ner-v1/entity_aggregate_provisional.csv", INVENTORY_SHA, "released inventory")):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 2
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            print(f"{what} has changed since the freeze:\n  now      {digest}\n  frozen   {expected}", file=sys.stderr)
            print("Re-freezing is a protocol amendment, not a re-run.", file=sys.stderr)
            return 1

    verdict_sha = hashlib.sha256(args.verdicts.read_bytes()).hexdigest()
    if VERDICTS_SHA != "PENDING" and verdict_sha != VERDICTS_SHA:
        print(f"the adjudication file has changed since the freeze:\n  now      {verdict_sha}\n"
              f"  frozen   {VERDICTS_SHA}", file=sys.stderr)
        return 1
    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8"))
    identifiers = json.loads(args.ids.read_text(encoding="utf-8"))

    # the adjudication file is evidence, so its structure is checked rather than assumed
    problems = []
    for key, record in verdicts.items():
        votes = record.get("votes") or [record.get("verdict")]
        if record.get("occurrence_uid", key) != key:
            problems.append(f"{key}: the embedded uid does not equal its key")
        if record.get("verdict") not in VOTES:
            problems.append(f"{key}: verdict outside the protocol")
        if any(vote not in VOTES for vote in votes):
            problems.append(f"{key}: a vote is outside the protocol")
        if len(votes) not in (1, 3):
            problems.append(f"{key}: {len(votes)} votes; the protocol takes one or three")
        winner = collections.Counter(votes).most_common(1)[0][0]
        if record.get("verdict") != winner:
            problems.append(f"{key}: the recorded verdict is not the majority of its votes")
        if record.get("span_relation") not in RELATIONS:
            problems.append(f"{key}: span_relation outside the protocol")
        if record.get("text_region") not in REGIONS:
            problems.append(f"{key}: text_region outside the protocol")
        if not record.get("rationale"):
            problems.append(f"{key}: no rationale")
    if problems:
        print("the adjudication file does not satisfy its declared structure:", file=sys.stderr)
        for line in problems[:20]:
            print(f"  {line}", file=sys.stderr)
        return 2

    universe = {r["source_credit_label"] for r in read_rows(ROOT / "results/ner-v1/graph_label_universe.csv")}
    inventory = {r["entity"]: r["entity_type"]
                 for r in read_rows(ROOT / "results/ner-v1/entity_aggregate_provisional.csv")}
    frame = [r for r in read_rows(args.candidates)
             if r["candidate_source"] == "LEXICON_WITH_TRANSFORMER_CHECK"
             and r["cross_label_shared_cleaned_text"] == "False"
             and r["source_credit_label"] in universe
             and r["candidate_surface"] in inventory]
    print(f"candidate frame: {len(frame)} (frozen: {FRAME_SIZE})")
    if len(frame) != FRAME_SIZE:
        print("frame size does not match the protocol", file=sys.stderr)
        return 1

    from transformers import pipeline  # noqa: PLC0415  -- only needed for the flag stage
    taggers = {name: pipeline("token-classification", model=name, revision=revision,
                              aggregation_strategy="simple")
               for name, revision in MODELS.items()}

    cache: dict[str, list[tuple[int, int, str, str]]] = {}

    def spans(text: str) -> list[tuple[int, int, str, str]]:
        """Every proposal, carrying the model that made it. Dropping the model name here
        made `proposing_models` unreproducible and left ties to iteration order."""
        if text not in cache:
            cache[text] = sorted({(int(x["start"]), int(x["end"]), x["entity_group"], name)
                                  for name, tag in taggers.items() for x in tag(text)})
        return cache[text]

    combos: dict[tuple[str, str, str], dict] = collections.defaultdict(
        lambda: {"occurrences": 0, "strict": 0, "labels": set(), "models": set(), "flaggers": set()})
    occurrences: list[dict[str, object]] = []
    for row in frame:
        try:
            start, end = int(row["surface_start_in_context"]), int(row["surface_end_in_context"])
        except (TypeError, ValueError):
            continue
        containing = [(a, b, g, m) for a, b, g, m in spans(row["context_snippet"])
                      if a <= start and b >= end and (a < start or b > end)]
        if not containing:
            continue
        # deterministic: longest span, then leftmost, then shortest, then label, then model
        a, b, group, _ = min(containing, key=lambda s: (-(s[1] - s[0]), s[0], s[1], s[2], s[3]))
        proposers = sorted({m for x, y, g, m in containing if (x, y, g) == (a, b, group)})
        # every model that proposed ANY strictly containing span, not only the winning one
        flaggers = sorted({m for *_, m in containing})
        surface, span = row["candidate_surface"], row["context_snippet"][a:b]
        entry = combos[(surface, span, group)]
        entry["occurrences"] += 1
        entry["strict"] += row["strict_high_consistency"] == "True"
        entry["labels"].add(row["source_credit_label"])
        entry["models"].update(proposers)
        entry["flaggers"].update(flaggers)

        # full digest, not a 16-character prefix: the freeze record described this as
        # "sha256 of (...)" while emitting a truncation, so the record was inaccurate
        uid = hashlib.sha256(UNIT.join([row["song_lyric_content_sha256"], row["chunk_id"],
                                        row["candidate_start_char"], row["candidate_end_char"]]
                                       ).encode("utf-8")).hexdigest()
        adjudication = verdicts.get(uid, {})
        occurrences.append({
            # Identity of the textual occurrence: lyric content, chunk, character span.
            # `surface_start_in_context` is an offset inside the context snippet, not
            # inside the song; keying on it collapsed 91 occurrences onto 81 ids.
            "occurrence_uid": uid,
            # same lyric content and character span in a different chunk: the upstream
            # duplication this protocol is sequenced behind, reported not repaired
            "duplicate_text_key": hashlib.sha256(
                UNIT.join([row["song_lyric_content_sha256"], row["candidate_start_char"],
                           row["candidate_end_char"]]).encode("utf-8")).hexdigest()[:16],
            "combination_id": identifiers.get(private_key(surface, span, group), "UNASSIGNED"),
            "short_surface": surface,
            "release_schema_type": inventory[surface],
            "containing_span": span,
            "model_entity_group": group,
            "proposing_models": "|".join(proposers),
            "flagging_models": "|".join(flaggers),
            "source_credit_label": row["source_credit_label"],
            "song_id": row["song_id"],
            "strict_high_consistency": row["strict_high_consistency"],
            "verdict": adjudication.get("verdict", "UNADJUDICATED"),
            # two independent judgements, never one variable: an earlier version folded
            # the span-to-surface relation and the lyric/credit region into one field, so
            # 8 credit-block occurrences carried a non-credit label and one lyric
            # occurrence carried a credit label
            "span_relation": adjudication.get("span_relation", ""),
            "text_region": adjudication.get("text_region", ""),
            "confidence": adjudication.get("confidence", ""),
            # votes and rationale enter the ledger, and therefore the frozen hash: an
            # earlier ledger omitted them, so the vote structure could be deleted from the
            # verdict file without changing anything the freeze committed to
            "votes": "|".join(adjudication.get("votes", [adjudication.get("verdict", "")])),
            "rationale": adjudication.get("rationale", ""),
        })

    print(f"flagged occurrences: {len(occurrences)} (frozen: {TOTALS['flagged_occurrences']})")
    print(f"unique combinations: {len(combos)} (frozen: {TOTALS['resolution_rows']})")

    flag_bytes = render(FLAG_COLUMNS, [
        [surface, span, group, "|".join(sorted(entry["models"])), "|".join(sorted(entry["flaggers"])),
         entry["occurrences"], entry["strict"], len(entry["labels"])]
        for (surface, span, group), entry in sorted(combos.items())])
    ledger_bytes = render(LEDGER_COLUMNS, [
        [record[column] for column in LEDGER_COLUMNS]
        for record in sorted(occurrences, key=lambda r: (r["short_surface"], r["containing_span"],
                                                         r["model_entity_group"], r["occurrence_uid"]))])
    flag_sha = hashlib.sha256(flag_bytes).hexdigest()
    ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    provenance_hashes = {}
    if args.provenance_dir:
        for name in ("blinded_prompt_v2.txt", "blinded_input_v2.json", "batch_manifest.json",
                     "blind_id_map.json", "ballots_v2_manifest.json"):
            path = args.provenance_dir / name
            if not path.is_file():
                print(f"missing frozen blinded-adjudication input: {path}", file=sys.stderr)
                return 2
            provenance_hashes[name.rsplit(".", 1)[0] + "_sha256"] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    print(f"private flag table sha256      {flag_sha}")
    print(f"private occurrence ledger sha  {ledger_sha}")
    strict_now = sum(int(e["strict"]) for e in combos.values())
    repeated = collections.Counter(r["duplicate_text_key"] for r in occurrences)
    duplicated = sum(n for n in repeated.values() if n > 1)

    # fold legitimacy, computed from the independent verdicts alone
    folded: dict[str, dict] = {}
    for record in occurrences:
        seen = folded.setdefault(private_key(record["short_surface"], record["containing_span"],
                                             record["model_entity_group"]), {
            "combination_id": record["combination_id"],
            "surface": record["short_surface"], "span": record["containing_span"],
            "group": record["model_entity_group"], "models": set(), "flaggers": set(),
            "release_schema_type": record["release_schema_type"], "labels": set(),
            "occurrences": 0, "strict": 0, "credit_region_occurrences": 0,
            "verdicts": collections.Counter(), "relations": collections.Counter(),
            "regions": collections.Counter()})
        seen["occurrences"] += 1
        seen["strict"] += record["strict_high_consistency"] == "True"
        seen["credit_region_occurrences"] += record["text_region"] == "PRODUCTION_CREDIT"
        seen["verdicts"][record["verdict"]] += 1
        seen["relations"][record["span_relation"]] += 1
        seen["regions"][record["text_region"]] += 1
        seen["models"].update(record["proposing_models"].split("|"))
        seen["flaggers"].update(record["flagging_models"].split("|"))
        seen["labels"].add(record["source_credit_label"])
    # Each of the three is folded on its own and only when its occurrences agree. Taking a
    # modal value and overwriting the rest hid genuine disagreement, and folding the span
    # relation and the text region together made one field answer two questions.
    for seen in folded.values():
        seen["folded_verdict"] = next(iter(seen["verdicts"])) if len(seen["verdicts"]) == 1 else "SPLIT_NOT_FOLDED"
        seen["folded_relation"] = next(iter(seen["relations"])) if len(seen["relations"]) == 1 else "MIXED"
        seen["folded_region"] = next(iter(seen["regions"])) if len(seen["regions"]) == 1 else "MIXED"
    for name, key in (("adjudicated differently", "verdicts"),
                      ("assigned different span relations", "relations"),
                      ("assigned different text regions", "regions")):
        split = {cid: seen for cid, seen in folded.items() if len(seen[key]) > 1}
        print(f"combinations whose occurrences were {name}: {len(split)} "
              f"(published as a mixed state, not folded)")
        for _, seen in sorted(split.items()):
            print(f"  {seen['combination_id']}  {seen['surface']} -> {dict(seen[key])}")
    contested = {cid: seen for cid, seen in folded.items() if len(seen["verdicts"]) > 1}

    # rendered in memory so its hash can be compared on every run, not only when the
    # private directory happens to be written
    fold_bytes = (json.dumps(
        {key: {"combination_id": s["combination_id"],
               "surface": s["surface"], "span": s["span"], "group": s["group"],
               "proposing_models": "|".join(sorted(s["models"])),
               "flagging_models": "|".join(sorted(s["flaggers"])),
               "release_schema_type": s["release_schema_type"],
               "distinct_source_labels": len(s["labels"]),
               "occurrences": s["occurrences"], "strict": s["strict"],
               "credit_region_occurrences": s["credit_region_occurrences"],
               "verdict": s["folded_verdict"], "span_relation": s["folded_relation"],
               "text_region": s["folded_region"],
               "verdicts": dict(s["verdicts"]), "relations": dict(s["relations"]),
               "regions": dict(s["regions"]),
               "folded": len(s["verdicts"]) == 1}
         for key, s in sorted(folded.items())}, ensure_ascii=False, indent=1) + "\n"
    ).encode("utf-8")
    fold_sha = hashlib.sha256(fold_bytes).hexdigest()
    print(f"private fold state sha256      {fold_sha}")

    if args.write_private:
        args.write_private.mkdir(parents=True, exist_ok=True)
        (args.write_private / "flag_table.csv").write_bytes(flag_bytes)
        (args.write_private / "occurrence_ledger.csv").write_bytes(ledger_bytes)
        (args.write_private / "fold_state.json").write_bytes(fold_bytes)
        (args.write_private / "freeze_inputs.json").write_text(json.dumps({
            "private_flag_table_sha256": flag_sha,
            "private_occurrence_ledger_sha256": ledger_sha,
            "private_fold_state_sha256": fold_sha,
            "private_adjudication_sha256": verdict_sha,
            "private_candidate_table_sha256": CANDIDATE_TABLE_SHA,
            "graph_label_universe_sha256": GRAPH_UNIVERSE_SHA,
            "released_inventory_sha256": INVENTORY_SHA,
            "candidate_frame_size": len(frame),
            "flagged_occurrences": len(occurrences),
            "flagged_strict_occurrences": strict_now,
            "resolution_rows": len(combos),
            "occurrence_ledger": {
                "rows": len(occurrences),
                "identity": "the full 64-character sha256 hex digest of "
                            "(song_lyric_content_sha256, chunk_id, candidate_start_char, "
                            "candidate_end_char), not truncated",
                "every_occurrence_appears_exactly_once": len({r["occurrence_uid"] for r in occurrences}) == len(occurrences),
                "adjudicated_independently_per_occurrence": True,
                "blinded_from": ["strict_high_consistency", "source_credit_label", "song_id",
                                 "occurrence and label counts", "any prior decision"],
                "batch_constraint": "no two occurrences of one combination share a batch; "
                                    "records carry random blind identifiers only",
                "combinations_not_folded": len(contested),
                "combinations_with_mixed_span_relation": sum(
                    1 for s in folded.values() if len(s["relations"]) > 1),
                "combinations_with_mixed_text_region": sum(
                    1 for s in folded.values() if len(s["regions"]) > 1),
                "occurrences_sharing_content_and_char_span_across_chunks": duplicated,
                "occurrences_in_production_credit_regions": sum(
                    1 for r in occurrences if r["text_region"] == "PRODUCTION_CREDIT"),
                "three_vote_occurrences": sum(
                    1 for r in occurrences if len(r["votes"].split("|")) == 3),
            },
            "reference_models": MODELS,
            "blinded_provenance": provenance_hashes,
            "combinations_flagged_by_both_models": sum(
                1 for entry in combos.values() if len(entry["flaggers"]) > 1),
            "combinations_with_the_same_winning_span_from_both_models": sum(
                1 for entry in combos.values() if len(entry["models"]) > 1),
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"unredacted flag table, ledger and fold state written to {args.write_private}")

    if not args.check:
        return 0

    failures = []
    if len(occurrences) != TOTALS["flagged_occurrences"]:
        failures.append(f"flagged occurrences {len(occurrences)} != {TOTALS['flagged_occurrences']}")
    if len(combos) != TOTALS["resolution_rows"]:
        failures.append(f"combinations {len(combos)} != {TOTALS['resolution_rows']}")
    strict_total = sum(int(e["strict"]) for e in combos.values())
    if strict_total != TOTALS["flagged_strict_occurrences"]:
        failures.append(f"strict {strict_total} != {TOTALS['flagged_strict_occurrences']}")
    compare_pinned("private flag table", flag_sha, PRIVATE_FLAG_SHA, failures)
    compare_pinned("occurrence ledger", ledger_sha, PRIVATE_LEDGER_SHA, failures)
    compare_pinned("fold state", fold_sha, PRIVATE_FOLD_SHA, failures)
    for name, expected in (("private_flag_table_sha256", PRIVATE_FLAG_SHA),
                           ("private_occurrence_ledger_sha256", PRIVATE_LEDGER_SHA)):
        if expected != "PENDING" and freeze.get(name) != expected:
            failures.append(f"freeze record {name} does not match the frozen value")

    uids = [r["occurrence_uid"] for r in occurrences]
    if len(set(uids)) != len(uids):
        failures.append(f"occurrence_uid is not unique: {len(set(uids))} of {len(uids)}")
    unadjudicated = [r for r in occurrences if r["verdict"] == "UNADJUDICATED"]
    if unadjudicated:
        failures.append(f"{len(unadjudicated)} occurrences have no independent verdict")
    unused = sorted(set(verdicts) - set(uids))
    if unused:
        failures.append(f"{len(unused)} verdicts do not correspond to any flagged occurrence: {unused[:5]}")
    unassigned = [r for r in occurrences if r["combination_id"] == "UNASSIGNED"]
    if unassigned:
        failures.append(f"{len(unassigned)} occurrences have no frozen combination id")
    if len(set(identifiers.values())) != len(identifiers):
        failures.append("the frozen combination id map is not injective")

    public = {r["combination_id"]: r for r in read_rows(DIR / "resolution_table.csv")}
    for _, seen in sorted(folded.items()):
        cid = seen["combination_id"]
        row = public.get(cid)
        if row is None:
            failures.append(f"{cid}: in the ledger but not in the public table")
            continue
        if seen["occurrences"] != int(row["flagged_occurrences"]):
            failures.append(f"{cid}: ledger has {seen['occurrences']} occurrences, table says {row['flagged_occurrences']}")
        if seen["strict"] != int(row["flagged_strict_occurrences"]):
            failures.append(f"{cid}: ledger has {seen['strict']} strict, table says {row['flagged_strict_occurrences']}")
        if "|".join(sorted(seen["models"])) != row["proposing_models"]:
            failures.append(f"{cid}: proposing_models does not reproduce")
        single = len(seen["verdicts"]) == 1
        published = row["resolution_action"]
        if single and published != next(iter(seen["verdicts"])):
            failures.append(f"{cid}: published action {published} is not the adjudicated verdict")
        if not single and published != "SPLIT_NOT_FOLDED":
            failures.append(f"{cid}: occurrences disagree, so the row must publish SPLIT_NOT_FOLDED, not {published}")
    for cid in sorted(set(public) - {s["combination_id"] for s in folded.values()}):
        failures.append(f"{cid}: in the public table but no occurrence in the ledger")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"compared against pinned hashes: flag table, {len(occurrences)}-occurrence ledger, "
          "fold state, adjudication file; per-combination fold cross-checked against the "
          "public table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
