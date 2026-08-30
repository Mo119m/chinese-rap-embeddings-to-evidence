"""Gate the frozen compound resolution table (protocol NER-CR-001).

Every expectation is hardcoded here and duplicated deliberately, including the rationale
phrase table the publisher also holds. Nothing is read from the artefacts under test.

The whole canonical `freeze.json` is pinned by hash, so a field the gate does not
individually understand cannot be edited either: an earlier version checked a handful of
declarations and let the rest be set to anything. Totals and action counts are recomputed
from the table and compared against the pinned values and the record, so an edit to one

# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
side alone fails, and an edit to both together fails against the pins.

Runs offline, in CI, and from inside the desktop package, where the artefacts sit under
`Results/compound-resolution-ner-cr-001/`.

    python tools/verify_compound_resolution.py
    python tools/verify_compound_resolution.py --private-dir /path/outside/the/repo
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

PROTOCOL = "NER-CR-001"
FRAME_SIZE = 1098
COMBINATIONS = 63
TOTALS = {"flagged_occurrences": 91, "flagged_strict_occurrences": 39, "resolution_rows": 63}
ACTIONS = {
    "RETAIN_SHORT": {"rows": 33, "occurrences": 42, "strict_occurrences": 21, "retain": "yes"},
    "DROP_SHORT": {"rows": 30, "occurrences": 49, "strict_occurrences": 18, "retain": "no"},
}
RETAIN_BY_ACTION = {"RETAIN_SHORT": "yes", "DROP_SHORT": "no", "DEFER": "withheld",
                    "SPLIT_NOT_FOLDED": "split"}
MODELS = {
    "shibing624/bert4ner-base-chinese": "5d660ed2aa9da482bf2d99c6bc8cf2ce66758f6a",
    "uer/roberta-base-finetuned-cluener2020-chinese": "cddd8fc233e373855a8c0a7f4b7eb83acb686a2b",
}
CANDIDATE_TABLE_SHA = "3ac7a41cc971e34b2314c87d3c50c11c95af039e278b851cc144a8c553daa5e4"
GRAPH_UNIVERSE_SHA = "7a09b319c2bd1449c671c1d85c2bc1ef54fa4b18ac03130e2d3883332a4342b3"
INVENTORY_SHA = "6ab77221ca632d3117c6e4287305354fb8fa1266ffdde71e24d1f6e4b6535cb4"
PUBLIC_TABLE_SHA = "d7878ca17c11a97a0ea8207f009a33e8a90a53a812902e80cb57ba8d76e67646"
CANONICAL_FREEZE_SHA = "f3745ebc69f09a42da98379e9e45076fae060b4d4307291dbfc27210bd6a1627"
ALLOWLIST_SHA = "537068f6380f9c58e9a11326e0093876f3c02ab3308a52a9b9d79d6d42a165e4"
PRIVATE_FLAG_SHA = "e711face34cb4e2ce1ecc96712255a51d73be133ad301bfb297f7ac83c090a8b"
PRIVATE_LEDGER_SHA = "1a1cde4820915b1bfadef6a4f90cb78aeb1f0e4e54ee9a856ce21ca2e7146d0d"
PRIVATE_FOLD_SHA = "83b0bc299dd0caa2abbc0635046af81d39bcbf13d365b54c5f91d4a8508132aa"
PRIVATE_ADJUDICATION_SHA = "f4c3da5d6e053c7220f6b3566c56c5d05d7c75c828fac225e77ac556a6980919"
BLINDED_PROVENANCE = {
    "blinded_prompt_v2_sha256": "bcb893375ea121e1c8464b78b82a3f5ac575582521549a6b0c0bf1d72fddaad6",
    "blinded_input_v2_sha256": "9b8cb99d4501536eabf7164a700db15d4739a25ad9379fd53fc7ae094cbaa4ad",
    "batch_manifest_sha256": "9d2f7472197127c9a3e8fbcef18b7142f1df9ec9b838a1ab3ccf6ab90a662b1d",
    "blind_id_map_sha256": "033526905e44549b409712d0101e06b95b51c9836f7f913690dd44993a0c572e",
    "ballots_v2_manifest_sha256": "eeee21561d0c0eb616b37882996aea237aedf61769536aa5c2e7dc104833bea1",
}

COLUMNS = ["combination_id", "short_surface", "release_schema_type", "containing_span", "span_kind",
           "containing_span_length_bucket", "model_entity_group", "proposing_models",
           "flagged_occurrences", "flagged_strict_occurrences", "distinct_source_labels",
           "credit_region_occurrences", "resolution_action", "retain_short", "span_relation",
           "text_region", "longer_span_disposition", "resolution_rationale"]
SPAN_KINDS = {"public_entity", "merged_span_structural", "production_credit_generalised"}
BUCKETS = {"2-3", "4-6", "7-12", "13+"}
PLACEHOLDERS = {"<PRODUCTION_CREDIT_COMPANY>", "<PRODUCTION_CREDIT_CHOIR>",
                "<PERSONNEL_AFFILIATION_ORG>", "<PUBLISHING_CREDIT_ORG>",
                "<SAMPLE_SOURCE_WORK_TITLE>"}
STRUCTURAL = {"<OTHER_PLACE>", "<HAN>", "<LATIN_RUN>", "<BREAK>"}
RELATIONS = {"TAGGER_ARTEFACT", "SAME_REFERENT", "DISTINCT_ENTITY", "UNKNOWN", "MIXED"}
REGIONS = {"LYRIC", "PRODUCTION_CREDIT", "UNKNOWN", "MIXED"}
DISPOSITION = {
    "TAGGER_ARTEFACT": "NOT_A_CANDIDATE_TAGGER_ARTEFACT",
    "SAME_REFERENT": "NOT_A_CANDIDATE_SAME_REFERENT",
    "DISTINCT_ENTITY": "NEW_CANDIDATE_REQUIRES_STANDARD_GATE",
    "UNKNOWN": "NOT_ASSESSED_RELATION_UNKNOWN",
    "MIXED": "NOT_ASSESSED_RELATION_MIXED",
}
# duplicated on purpose: a rationale must be a fixed phrase, never free text written while
# reading an occurrence
RATIONALE = {
    ("RETAIN_SHORT", "TAGGER_ARTEFACT"):
        "the reference span merged across a line, language or list boundary; the short entity stands",
    ("RETAIN_SHORT", "SAME_REFERENT"):
        "the reference span is a fuller co-referential form of the same referent; the short entity stands",
    ("RETAIN_SHORT", "DISTINCT_ENTITY"):
        "the reference span names something else, but the surface is an external locative modifier "
        "and refers to its schema type here",
    ("RETAIN_SHORT", "UNKNOWN"):
        "the relation of the reference span to the surface could not be settled, but the surface "
        "refers to its schema type here",
    ("DROP_SHORT", "DISTINCT_ENTITY"):
        "the surface occurs inside a larger proper name with a different referent",
    ("DROP_SHORT", "TAGGER_ARTEFACT"):
        "the surface does not refer to its schema type at these occurrences",
    ("DROP_SHORT", "SAME_REFERENT"):
        "the surface does not refer to its schema type at these occurrences",
    ("DROP_SHORT", "UNKNOWN"):
        "the surface does not refer to its schema type at these occurrences",
    ("RETAIN_SHORT", "MIXED"):
        "the occurrences agree the short entity stands, but were assigned different span "
        "relations; the relation is published as mixed, not folded",
    ("DROP_SHORT", "MIXED"):
        "the occurrences agree the short mention does not stand, but were assigned different "
        "span relations; the relation is published as mixed, not folded",
    ("DEFER", None):
        "automatic and contextual evidence did not decide; excluded from both the retained and "
        "the dropped set and counted separately",
    ("SPLIT_NOT_FOLDED", None):
        "occurrences of this combination were adjudicated differently and the row is not folded "
        "into a single action",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
HAN = re.compile(r"[一-鿿]")


def locate() -> tuple[Path | None, Path | None]:
    """Find the artefacts in the repository or inside the desktop package."""
    here = Path(__file__).resolve().parent
    for base, artefacts, results in ((here.parent, "analysis/compound-resolution", "results/ner-v1"),
                                     (here.parent.parent, "Results/compound-resolution-ner-cr-001",
                                      "Results/ner-v1")):
        directory = base / artefacts
        if (directory / "resolution_table.csv").is_file():
            return directory, base / results
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-dir", type=Path, default=None,
                        help="directory holding the private flag table, ledger, fold state and "
                             "adjudication, to check their bytes rather than only recorded hashes")
    args = parser.parse_args()

    directory, results = locate()
    if directory is None:
        print("could not locate the compound resolution artefacts", file=sys.stderr)
        return 2
    table_path = directory / "resolution_table.csv"
    freeze_path = directory / "freeze.json"
    allowlist_path = directory / "public_name_allowlist.json"
    for path in (freeze_path, allowlist_path):
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            return 2
    rows = list(csv.DictReader(table_path.open(encoding="utf-8", newline="")))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    allowlist_doc = json.loads(allowlist_path.read_text(encoding="utf-8"))
    # approved names only; a withheld ruling must never print the string it withholds
    allowlist = {entry["span"]: entry for entry in allowlist_doc.get("approved", {}).values()}
    fail: list[str] = []
    print(f"  artefacts: {directory}")

    if not rows or list(rows[0]) != COLUMNS:
        print("\nFAILED:\n  columns are not the declared whitelist: "
              f"{list(rows[0]) if rows else 'empty table'}", file=sys.stderr)
        return 1

    # whole-file pins: any edit to any field, declared or not, fails here
    for path, expected, what in ((table_path, PUBLIC_TABLE_SHA, "public table"),
                                 (freeze_path, CANONICAL_FREEZE_SHA, "canonical freeze record"),
                                 (allowlist_path, ALLOWLIST_SHA, "public name allowlist")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != "PENDING" and digest != expected:
            fail.append(f"{what} hash {digest} != pinned {expected}")

    if freeze.get("protocol") != PROTOCOL:
        fail.append(f"protocol {freeze.get('protocol')!r} != {PROTOCOL!r}")
    if freeze.get("candidate_frame", {}).get("size") != FRAME_SIZE:
        fail.append(f"candidate frame size != {FRAME_SIZE}")
    if freeze.get("downstream_consulted") is not False:
        fail.append("freeze does not assert downstream_consulted is false")
    adjudication = freeze.get("adjudication", {})
    if adjudication.get("is_human_gold") is not False:
        fail.append("adjudication must be declared not human gold")
    if adjudication.get("independent_human_review_status") != "pending":
        fail.append("independent human review status must be recorded as pending")
    if adjudication.get("blinded") is not True:
        fail.append("freeze does not declare the adjudication blinded")
    if not str(adjudication.get("unit", "")).startswith("one verdict per flagged occurrence"):
        fail.append("freeze does not declare per-occurrence adjudication")
    if freeze.get("reference_models") != MODELS:
        fail.append(f"reference_models does not match the pinned map: {freeze.get('reference_models')}")

    inputs = freeze.get("inputs", {})
    if inputs.get("private_candidate_table_sha256") != CANDIDATE_TABLE_SHA:
        fail.append("private candidate table hash does not match the pinned value")
    for name, relative, expected in (("graph_label_universe_sha256", "graph_label_universe.csv", GRAPH_UNIVERSE_SHA),
                                     ("released_inventory_sha256", "entity_aggregate_provisional.csv", INVENTORY_SHA)):
        if inputs.get(name) != expected:
            fail.append(f"{name} in the freeze record does not match the pinned value")
        path = results / relative
        if not path.is_file():
            fail.append(f"missing {path}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            fail.append(f"{relative} on disk does not match its pinned hash")
    for name, expected in (("public_table_sha256", PUBLIC_TABLE_SHA),
                           ("public_name_allowlist_sha256", ALLOWLIST_SHA),
                           ("private_flag_table_sha256", PRIVATE_FLAG_SHA),
                           ("private_occurrence_ledger_sha256", PRIVATE_LEDGER_SHA),
                           ("private_fold_state_sha256", PRIVATE_FOLD_SHA),
                           ("private_adjudication_sha256", PRIVATE_ADJUDICATION_SHA)):
        if expected != "PENDING" and freeze.get(name) != expected:
            fail.append(f"{name} in the freeze record does not match the pinned value")

    ledger = freeze.get("occurrence_ledger", {})
    if ledger.get("rows") != TOTALS["flagged_occurrences"]:
        fail.append(f"occurrence ledger rows != {TOTALS['flagged_occurrences']}")
    if ledger.get("every_occurrence_appears_exactly_once") is not True:
        fail.append("freeze does not assert every occurrence appears exactly once")
    if ledger.get("adjudicated_independently_per_occurrence") is not True:
        fail.append("freeze does not assert independent per-occurrence adjudication")
    if "strict_high_consistency" not in (ledger.get("blinded_from") or []):
        fail.append("freeze does not record that the adjudication was blinded to the strict flag")
    if "no two occurrences of one combination share a batch" not in str(ledger.get("batch_constraint", "")):
        fail.append("freeze does not state the batch isolation constraint")
    if freeze.get("blinded_provenance") != BLINDED_PROVENANCE:
        fail.append("blinded provenance hashes in the freeze do not match the pinned set")
    if "rekey_blinded_ballots" not in str(adjudication.get("method", "")):
        fail.append("freeze does not declare the scripted ballot fold")
    if "not demonstrated" not in str(adjudication.get("rulings", "")):
        fail.append("freeze overstates ruling independence")
    if "disclosure-safety gate" not in str(allowlist_doc.get("role", "")):
        fail.append("the allowlist does not confine itself to a disclosure-safety role")
    # the file must not print any string it withholds: withheld rulings are opaque ids only
    withheld = allowlist_doc.get("withheld", {})
    if not isinstance(withheld.get("count"), int) or "ids" not in withheld:
        fail.append("the allowlist does not record withheld rulings as a count and opaque ids")
    else:
        for identifier in withheld["ids"]:
            if not UUID4.fullmatch(str(identifier)):
                fail.append("a withheld ruling is not recorded as an opaque id")
                break
    if any(key not in ("protocol", "role", "rule", "adjudication", "disclosure",
                       "approved", "withheld") for key in allowlist_doc):
        fail.append("the allowlist carries an undeclared section")
    if not SHA256.fullmatch(str(allowlist_doc.get("adjudication", {}).get("private_rulings_sha256", ""))):
        fail.append("the allowlist does not bind the private rulings file by hash")

    identifiers = [r["combination_id"] for r in rows]
    if len(set(identifiers)) != len(identifiers):
        fail.append(f"combination_id is not unique: {len(set(identifiers))} of {len(identifiers)}")
    for value in identifiers:
        if not UUID4.fullmatch(value):
            fail.append(f"combination_id {value!r} is not an opaque UUID4; a content-derived id can "
                        "be recomputed from a guessed private key")
            break
    if len(rows) != COMBINATIONS:
        fail.append(f"rows {len(rows)} != {COMBINATIONS}")

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        seen = counts.setdefault(row["resolution_action"],
                                 {"rows": 0, "occurrences": 0, "strict_occurrences": 0})
        seen["rows"] += 1
        seen["occurrences"] += int(row["flagged_occurrences"])
        seen["strict_occurrences"] += int(row["flagged_strict_occurrences"])
    if set(counts) - set(RETAIN_BY_ACTION):
        fail.append(f"actions outside the protocol: {sorted(set(counts) - set(RETAIN_BY_ACTION))}")
    if counts != freeze.get("action_counts"):
        fail.append("action_counts in the freeze record do not match the table")
    for action, expected in ACTIONS.items():
        got = counts.get(action, {"rows": 0, "occurrences": 0, "strict_occurrences": 0})
        for field in ("rows", "occurrences", "strict_occurrences"):
            if expected[field] and got[field] != expected[field]:
                fail.append(f"{action} {field}: {got[field]} != {expected[field]}")
        print(f"  {action:<20}{got['rows']:>4}{got['occurrences']:>5}{got['strict_occurrences']:>5}   "
              f"retain={expected['retain']}")
    totals = {"flagged_occurrences": sum(c["occurrences"] for c in counts.values()),
              "flagged_strict_occurrences": sum(c["strict_occurrences"] for c in counts.values()),
              "resolution_rows": len(rows)}
    for field, want in TOTALS.items():
        if totals[field] != want:
            fail.append(f"total {field}: {totals[field]} != {want}")
    if freeze.get("totals") != totals:
        fail.append(f"totals in the freeze record do not match the table: {freeze.get('totals')}")
    print(f"  {'TOTAL':<20}{totals['resolution_rows']:>4}{totals['flagged_occurrences']:>5}"
          f"{totals['flagged_strict_occurrences']:>5}")

    # recomputed rather than trusted: these are the declarations an earlier gate accepted at
    # any value at all
    credit_occurrences = sum(int(r["credit_region_occurrences"]) for r in rows)
    if ledger.get("occurrences_in_production_credit_regions") != credit_occurrences:
        fail.append(f"the record's production-credit occurrence count does not match the table "
                    f"({ledger.get('occurrences_in_production_credit_regions')} vs {credit_occurrences})")
    not_folded = sum(1 for r in rows if r["resolution_action"] == "SPLIT_NOT_FOLDED")
    if ledger.get("combinations_not_folded") != not_folded:
        fail.append("the record's not-folded count does not match the table")
    mixed_relation = sum(1 for r in rows if r["span_relation"] == "MIXED")
    if ledger.get("combinations_with_mixed_span_relation") != mixed_relation:
        fail.append("the record's mixed-relation count does not match the table")
    mixed_region = sum(1 for r in rows if r["text_region"] == "MIXED")
    if ledger.get("combinations_with_mixed_text_region") != mixed_region:
        fail.append("the record's mixed-region count does not match the table")

    for row in rows:
        where = row["combination_id"]
        occurrences = int(row["flagged_occurrences"])
        strict = int(row["flagged_strict_occurrences"])
        credit = int(row["credit_region_occurrences"])
        labels = int(row["distinct_source_labels"])
        if occurrences < 1 or strict < 0 or credit < 0 or labels < 1:
            fail.append(f"{where}: counts must be non-negative and occurrences positive")
        if strict > occurrences:
            fail.append(f"{where}: strict {strict} > occurrences {occurrences}")
        if credit > occurrences:
            fail.append(f"{where}: credit-region occurrences {credit} > occurrences {occurrences}")
        if labels > occurrences:
            fail.append(f"{where}: more distinct labels than occurrences")
        if row["span_kind"] not in SPAN_KINDS:
            fail.append(f"{where}: span_kind {row['span_kind']!r}")
        if row["containing_span_length_bucket"] not in BUCKETS:
            fail.append(f"{where}: length is not bucketed")
        if row["span_relation"] not in RELATIONS:
            fail.append(f"{where}: span_relation {row['span_relation']!r}")
        if row["text_region"] not in REGIONS:
            fail.append(f"{where}: text_region {row['text_region']!r}")
        # the two are independent judgements: a credit-region count must not imply a relation
        if row["text_region"] == "PRODUCTION_CREDIT" and credit != occurrences:
            fail.append(f"{where}: every occurrence must be in a credit region for that region label")
        if row["text_region"] == "LYRIC" and credit != 0:
            fail.append(f"{where}: a lyric-region row cannot carry credit-region occurrences")
        if row["longer_span_disposition"] != DISPOSITION.get(row["span_relation"]):
            fail.append(f"{where}: the disposition does not follow the span relation")
        if not row["proposing_models"]:
            fail.append(f"{where}: proposing_models was dropped")
        for model in row["proposing_models"].split("|"):
            if model not in MODELS:
                fail.append(f"{where}: proposing model {model!r} is not a pinned reference model")
        if row["retain_short"] != RETAIN_BY_ACTION.get(row["resolution_action"]):
            fail.append(f"{where}: retain_short does not follow the action")

        expected_text = RATIONALE.get((row["resolution_action"], row["span_relation"])) \
            or RATIONALE.get((row["resolution_action"], None))
        if row["resolution_rationale"] != expected_text:
            fail.append(f"{where}: rationale is not the canonical phrase for its action and relation")

        span = row["containing_span"]
        if row["span_kind"] == "production_credit_generalised" and span not in PLACEHOLDERS:
            fail.append(f"{where}: a production credit is not generalised")
        if row["span_kind"] == "public_entity":
            # contiguous Han is not evidence of a complete name; only the frozen allowlist is
            if span not in allowlist:
                fail.append(f"{where}: a span is published verbatim without an approved ruling")
            if row["span_relation"] != "DISTINCT_ENTITY":
                fail.append(f"{where}: a verbatim span must be a distinct entity")
        if row["span_kind"] == "merged_span_structural":
            tokens = span.split("+")
            unknown = [t for t in tokens if t not in STRUCTURAL and t != row["short_surface"]]
            if unknown:
                fail.append(f"{where}: a structural span carries text beyond the target surface: {unknown}")
            if row["short_surface"] not in tokens:
                fail.append(f"{where}: a structural span does not carry its own surface")
        if HAN.search(re.sub(r"<[A-Z_]+>", "", row["resolution_rationale"]
                             .replace(row["short_surface"], "").replace(span, ""))):
            fail.append(f"{where}: rationale carries Han text beyond the entity")

    if args.private_dir:
        for name, expected in (("flag_table.csv", PRIVATE_FLAG_SHA),
                               ("occurrence_ledger.csv", PRIVATE_LEDGER_SHA),
                               ("fold_state.json", PRIVATE_FOLD_SHA),
                               ("occurrence_adjudication_v2.json", PRIVATE_ADJUDICATION_SHA),
                               ("blinded_prompt_v2.txt", BLINDED_PROVENANCE["blinded_prompt_v2_sha256"]),
                               ("blinded_input_v2.json", BLINDED_PROVENANCE["blinded_input_v2_sha256"]),
                               ("batch_manifest.json", BLINDED_PROVENANCE["batch_manifest_sha256"]),
                               ("blind_id_map.json", BLINDED_PROVENANCE["blind_id_map_sha256"]),
                               ("ballots_v2_manifest.json", BLINDED_PROVENANCE["ballots_v2_manifest_sha256"])):
            path = args.private_dir / name
            if not path.is_file():
                fail.append(f"missing private artefact: {path}")
            elif expected != "PENDING" and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                fail.append(f"private {name} bytes do not match the pinned hash")
        print(f"  private artefacts checked byte for byte in {args.private_dir}")

    if fail:
        print("\nFAILED:", file=sys.stderr)
        for line in fail:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"\n  frozen table verified against hardcoded expectations ({PROTOCOL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
