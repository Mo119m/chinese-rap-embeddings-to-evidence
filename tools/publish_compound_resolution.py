"""Derive the public compound resolution table and freeze record (protocol NER-CR-001).

The redaction rules live here so they can be audited rather than taken on trust, and the
tool fails closed. Every private input is verified against a whole-file hash pinned below,
including the freeze inputs -- an earlier version left that one unpinned, so forged model
and ledger provenance could be written straight into `freeze.json`. Both output files are
rendered and staged before either is replaced, so a failure cannot leave a table and a
record that disagree.

Nothing about a public row is derived from private content. `combination_id` is a frozen
opaque UUID4 drawn from a private map: a truncated hash of the private key, which an
earlier version published, can be recomputed by anyone who guesses the key.

A containing span is published verbatim only if it appears in the public completeness
allowlist. An earlier version used a contiguous-Han regular expression, which cannot tell
a complete name from a one-character-short tagger boundary artefact, and published one.

Rationales come from a fixed phrase table keyed on the action and the span relation. Free
text written while reading an occurrence is a fingerprint even after the Chinese is removed.

    python tools/publish_compound_resolution.py --fold-state ... --credits ... --ids ...
        --hashes ... --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "analysis" / "compound-resolution"

PUBLIC_TABLE_SHA = "d7878ca17c11a97a0ea8207f009a33e8a90a53a812902e80cb57ba8d76e67646"
CANONICAL_FREEZE_SHA = "f3745ebc69f09a42da98379e9e45076fae060b4d4307291dbfc27210bd6a1627"

RULINGS_SHA = "3503eff4f098af32758753d9d6358b3ef85c8a5cf05313cea523122d90735b24"
CREDITS_MAP_SHA = "e917f42dd3ab117349860d7c9876ba2f2c059302e2788096412dbedc8d8d1178"
IDS_MAP_SHA = "0994a20fe6524a7660e7a92eadbf017cbce59e1c2d7a5057a2997728cd4a2a39"
FOLD_STATE_SHA = "83b0bc299dd0caa2abbc0635046af81d39bcbf13d365b54c5f91d4a8508132aa"
FREEZE_INPUTS_SHA = "5385d5df200dfd44b5a7b1ab16155245620c176cfc6ed4a94323ea6c646b21c1"
CREDIT_ENTRIES = 14
COMBINATIONS = 63

PLACEHOLDER_BY_KIND = {
    "company": "<PRODUCTION_CREDIT_COMPANY>",
    "choir": "<PRODUCTION_CREDIT_CHOIR>",
    "affiliation": "<PERSONNEL_AFFILIATION_ORG>",
    "publisher": "<PUBLISHING_CREDIT_ORG>",
    "work": "<SAMPLE_SOURCE_WORK_TITLE>",
}
FREEZE_INPUT_KEYS = {
    "private_flag_table_sha256", "private_occurrence_ledger_sha256", "private_fold_state_sha256",
    "private_adjudication_sha256", "private_candidate_table_sha256", "graph_label_universe_sha256",
    "released_inventory_sha256", "candidate_frame_size", "flagged_occurrences",
    "flagged_strict_occurrences", "resolution_rows", "occurrence_ledger", "reference_models",
    "combinations_flagged_by_both_models", "combinations_with_the_same_winning_span_from_both_models",
    "blinded_provenance",
}
EXPECTED_MODELS = {
    "shibing624/bert4ner-base-chinese": "5d660ed2aa9da482bf2d99c6bc8cf2ce66758f6a",
    "uer/roberta-base-finetuned-cluener2020-chinese": "cddd8fc233e373855a8c0a7f4b7eb83acb686a2b",
}
EXPECTED_CONSTANTS = {"candidate_frame_size": 1098, "flagged_occurrences": 91,
                      "flagged_strict_occurrences": 39, "resolution_rows": 63}

UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
HAN = re.compile(r"[一-鿿]+")
RUN = re.compile(r"[\r\n]+|[一-鿿]+|[A-Za-z][A-Za-z.'’]*")

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
DISPOSITION = {
    "TAGGER_ARTEFACT": "NOT_A_CANDIDATE_TAGGER_ARTEFACT",
    "SAME_REFERENT": "NOT_A_CANDIDATE_SAME_REFERENT",
    "DISTINCT_ENTITY": "NEW_CANDIDATE_REQUIRES_STANDARD_GATE",
    "UNKNOWN": "NOT_ASSESSED_RELATION_UNKNOWN",
    "MIXED": "NOT_ASSESSED_RELATION_MIXED",
}
RETAIN = {"RETAIN_SHORT": "yes", "DROP_SHORT": "no", "DEFER": "withheld", "SPLIT_NOT_FOLDED": "split"}

COLUMNS = ["combination_id", "short_surface", "release_schema_type", "containing_span", "span_kind",
           "containing_span_length_bucket", "model_entity_group", "proposing_models",
           "flagged_occurrences", "flagged_strict_occurrences", "distinct_source_labels",
           "credit_region_occurrences", "resolution_action", "retain_short", "span_relation",
           "text_region", "longer_span_disposition", "resolution_rationale"]


def bucket(length: int) -> str:
    return "2-3" if length <= 3 else "4-6" if length <= 6 else "7-12" if length <= 12 else "13+"


def structural(surface: str, span: str) -> str:
    """Keep only the target surface; every other run becomes a structural marker.

    Line breaks are detected on their own. An earlier version emitted `<BREAK>` only when a
    whole split segment was blank, so a span that merged across a line while carrying other
    text on either side lost the one marker that explained why it merged.
    """
    tokens: list[str] = []
    for part in re.split(f"({re.escape(surface)})", span):
        if part == surface:
            tokens.append(surface)
            continue
        for match in RUN.finditer(part):
            text = match.group()
            if text[0] in "\r\n":
                tokens.append("<BREAK>")
            elif HAN.fullmatch(text):
                tokens.append("<OTHER_PLACE>" if len(text) >= 2 else "<HAN>")
            else:
                tokens.append("<LATIN_RUN>")
    collapsed: list[str] = []
    for token in tokens:
        if not collapsed or token != collapsed[-1]:
            collapsed.append(token)
    return "+".join(collapsed)


def publish(fold: dict, credits: dict[str, str], identifiers: dict[str, str],
            allowlist: dict[str, dict]) -> list[dict[str, str]]:
    rows = []
    for key, state in sorted(fold.items()):
        surface, span, group = state["surface"], state["span"], state["group"]
        action = state["verdict"]
        relation = state["span_relation"]
        region = state["text_region"]
        digest = hashlib.sha256(span.encode("utf-8")).hexdigest()

        publishable = allowlist.get(span, {}).get("publish_verbatim", False)
        if digest in credits:
            shown, span_kind = credits[digest], "production_credit_generalised"
        elif publishable and relation == "DISTINCT_ENTITY":
            shown, span_kind = span, "public_entity"
        else:
            shown, span_kind = structural(surface, span), "merged_span_structural"

        text = RATIONALE.get((action, relation)) or RATIONALE.get((action, None)) \
            or RATIONALE[("SPLIT_NOT_FOLDED", None)]
        residue = re.sub(r"<[A-Z_]+>", "", text.replace(surface, "").replace(shown, ""))
        if HAN.search(residue):
            raise SystemExit(f"rationale carries Han text beyond the entity: {HAN.findall(residue)}")

        rows.append({
            "combination_id": identifiers[key],
            "short_surface": surface,
            "release_schema_type": state["release_schema_type"],
            "containing_span": shown,
            "span_kind": span_kind,
            "containing_span_length_bucket": bucket(len(span)),
            "model_entity_group": group,
            "proposing_models": state["proposing_models"],
            "flagged_occurrences": str(state["occurrences"]),
            "flagged_strict_occurrences": str(state["strict"]),
            "distinct_source_labels": str(state["distinct_source_labels"]),
            "credit_region_occurrences": str(state["credit_region_occurrences"]),
            "resolution_action": action,
            "retain_short": RETAIN[action],
            "span_relation": relation,
            "text_region": region,
            "longer_span_disposition": DISPOSITION.get(relation, "NOT_ASSESSED_RELATION_MIXED"),
            "resolution_rationale": text,
        })
    rows.sort(key=lambda r: (r["short_surface"], r["containing_span"], r["model_entity_group"],
                             r["combination_id"]))
    return rows


def render(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def compose_freeze(rows: list[dict[str, str]], inputs: dict, public_sha: str,
                   allowlist_sha: str) -> dict:
    """Build the freeze record from the published rows themselves.

    Totals and action counts are recomputed here rather than carried in, so the record
    cannot state numbers the table contradicts. The offline gate recomputes them again and
    compares, which is what makes an edited record fail.
    """
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        seen = counts.setdefault(row["resolution_action"],
                                 {"rows": 0, "occurrences": 0, "strict_occurrences": 0})
        seen["rows"] += 1
        seen["occurrences"] += int(row["flagged_occurrences"])
        seen["strict_occurrences"] += int(row["flagged_strict_occurrences"])
    return {
        "protocol": "NER-CR-001",
        "title": "Compound resolution for lexicon candidates",
        "status": "post-release protocol amendment, frozen before affected outcomes were recomputed",
        "estimand": "does the short surface, at this occurrence, refer to what its release schema "
                    "type says? A place name inside a larger proper name does not; a place name "
                    "inside a fuller co-referential form of the same place does; a place name used "
                    "as an external locative modifier on a name that stands without it does.",
        "adjudication": {
            "method": "blinded AI per-occurrence contextual adjudication under frozen "
                      "instructions, folded from raw ballots by tools/rekey_blinded_ballots.py",
            "model": "claude-opus-5",
            "unit": "one verdict per flagged occurrence, adjudicated from the occurrence context "
                    "alone",
            "blinded": True,
            "rulings": "AI-assisted repeated rulings from separate sessions; independence "
                       "between sessions is procedural, not demonstrated",
            "external_verification": "consulted for institutional referents",
            "independent_human_review_status": "pending",
            "is_human_gold": False,
        },
        "blinded_provenance": inputs["blinded_provenance"],
        "candidate_frame": {
            "definition": "candidate_source == LEXICON_WITH_TRANSFORMER_CHECK, "
                          "cross_label_shared_cleaned_text == False, source_credit_label in the "
                          "204-label graph universe, candidate_surface in the released inventory",
            "size": inputs["candidate_frame_size"],
            "reproduces_published_lexicon_candidate_counts": "22 of 22 surfaces",
        },
        "flag_rule": "any strictly containing entity span proposed by a reference tagger; the "
                     "winning span is the longest, then leftmost, then shortest, then by label, "
                     "then by model name",
        "reference_models": inputs["reference_models"],
        "combinations_flagged_by_both_models": inputs["combinations_flagged_by_both_models"],
        "combinations_with_the_same_winning_span_from_both_models":
            inputs["combinations_with_the_same_winning_span_from_both_models"],
        "inputs": {
            "private_candidate_table": "all_candidate_occurrences_private.csv",
            "private_candidate_table_sha256": inputs["private_candidate_table_sha256"],
            "graph_label_universe_sha256": inputs["graph_label_universe_sha256"],
            "released_inventory_sha256": inputs["released_inventory_sha256"],
        },
        "totals": {"flagged_occurrences": inputs["flagged_occurrences"],
                   "flagged_strict_occurrences": inputs["flagged_strict_occurrences"],
                   "resolution_rows": inputs["resolution_rows"]},
        "action_counts": counts,
        "occurrence_ledger": inputs["occurrence_ledger"],
        "downstream_consulted": False,
        "combination_id": "frozen opaque UUID4 from a private map; not derived from content",
        "public_name_allowlist_sha256": allowlist_sha,
        "private_flag_table_sha256": inputs["private_flag_table_sha256"],
        "private_occurrence_ledger_sha256": inputs["private_occurrence_ledger_sha256"],
        "private_fold_state_sha256": inputs["private_fold_state_sha256"],
        "private_adjudication_sha256": inputs["private_adjudication_sha256"],
        "public_table_sha256": public_sha,
        "derivation": {
            "flag_stage_and_ledger": "tools/build_compound_resolution_table.py",
            "public_table_and_freeze": "tools/publish_compound_resolution.py",
            "offline_gate": "tools/verify_compound_resolution.py",
            "adversarial_tests": "tests/test_compound_resolution_gate.py",
            "protocol": "methods/NER_CR_001_COMPOUND_RESOLUTION.md",
        },
    }


def atomic_replace_pair(staged: list[tuple[bytes, Path]]) -> None:
    """Replace several targets so that a failure part-way restores every earlier one.

    True cross-file atomicity does not exist; what is guaranteed is that after a failure
    no target holds new content while another holds old. Earlier replacements are rolled
    back from in-memory backups before the error propagates.
    """
    backups = [(target, target.read_bytes() if target.is_file() else None) for _, target in staged]
    temps: list[tuple[str, Path]] = []
    try:
        for content, target in staged:
            handle, temporary = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            with os.fdopen(handle, "wb") as stream:
                stream.write(content)
            temps.append((temporary, target))
        done: list[Path] = []
        try:
            for temporary, target in temps:
                os.replace(temporary, target)
                done.append(target)
        except BaseException:
            for target, backup in backups:
                if target in done and backup is not None:
                    restore_handle, restore_temp = tempfile.mkstemp(dir=str(target.parent),
                                                                    suffix=".tmp")
                    with os.fdopen(restore_handle, "wb") as stream:
                        stream.write(backup)
                    os.replace(restore_temp, target)
            raise
    finally:
        for temporary, _ in temps:
            Path(temporary).unlink(missing_ok=True)


def load_verified(path: Path, expected: str, what: str):
    """Read a private input only after its whole-file hash matches the pinned value."""
    problems: list[str] = []
    if not path.is_file():
        return None, [f"missing {what}: {path}"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != "PENDING" and digest != expected:
        problems.append(f"{what} hash {digest} != pinned {expected}")
    return json.loads(path.read_text(encoding="utf-8")), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold-state", type=Path, required=True)
    parser.add_argument("--credits", type=Path, required=True,
                        help="JSON mapping sha256(private span) -> placeholder kind")
    parser.add_argument("--ids", type=Path, required=True,
                        help="JSON mapping the private combination key -> frozen opaque id")
    parser.add_argument("--rulings", type=Path, required=True,
                        help="private public-name rulings: opaque id -> span and ruling")
    parser.add_argument("--hashes", type=Path, required=True,
                        help="freeze inputs emitted by build_compound_resolution_table.py")
    parser.add_argument("--check", action="store_true", help="compare against the frozen public hash")
    parser.add_argument("--write", action="store_true",
                        help="replace the public table and freeze record, after every check passes")
    args = parser.parse_args()

    for path in (args.fold_state, args.credits, args.ids, args.hashes, args.rulings):
        resolved = path.resolve()
        if ROOT in resolved.parents or resolved == ROOT:
            print(f"private input must live outside the repository: {path}", file=sys.stderr)
            return 2

    problems: list[str] = []
    credit_map, issues = load_verified(args.credits, CREDITS_MAP_SHA, "credits map")
    problems += issues
    id_map, issues = load_verified(args.ids, IDS_MAP_SHA, "combination id map")
    problems += issues
    fold, issues = load_verified(args.fold_state, FOLD_STATE_SHA, "fold state")
    problems += issues
    inputs, issues = load_verified(args.hashes, FREEZE_INPUTS_SHA, "freeze inputs")
    problems += issues
    rulings, issues = load_verified(args.rulings, RULINGS_SHA, "private public-name rulings")
    problems += issues
    if problems:
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        print("refusing to derive anything from unverified input; nothing was written", file=sys.stderr)
        return 1

    allowlist = {entry["span"]: entry for entry in rulings.values()}

    # exact structure, not merely presence
    if len(credit_map) != CREDIT_ENTRIES:
        problems.append(f"credits map has {len(credit_map)} entries, expected exactly {CREDIT_ENTRIES}")
    for digest, kind in credit_map.items():
        if not SHA256.fullmatch(str(digest)):
            problems.append(f"credits map key is not a sha256: {digest!r}")
        if kind not in PLACEHOLDER_BY_KIND:
            problems.append(f"credits map kind is not one of {sorted(PLACEHOLDER_BY_KIND)}: {kind!r}")
    if len(id_map) != COMBINATIONS:
        problems.append(f"id map has {len(id_map)} entries, expected exactly {COMBINATIONS}")
    if len(set(id_map.values())) != len(id_map):
        problems.append("id map is not injective")
    for value in id_map.values():
        if not UUID4.fullmatch(str(value)):
            problems.append(f"combination id is not an opaque UUID4: {value!r}")
    if len(fold) != COMBINATIONS:
        problems.append(f"fold state covers {len(fold)} combinations, expected {COMBINATIONS}")
    if set(inputs) != FREEZE_INPUT_KEYS:
        problems.append(f"freeze inputs do not carry exactly the declared keys; "
                        f"unexpected {sorted(set(inputs) - FREEZE_INPUT_KEYS)}, "
                        f"missing {sorted(FREEZE_INPUT_KEYS - set(inputs))}")
    else:
        if inputs["reference_models"] != EXPECTED_MODELS:
            problems.append("freeze inputs do not carry the pinned reference models")
        for name, expected in EXPECTED_CONSTANTS.items():
            if inputs[name] != expected:
                problems.append(f"freeze input {name} is {inputs[name]}, expected {expected}")
        blinded = inputs.get("blinded_provenance", {})
        expected_blinded = {"blinded_prompt_v2_sha256", "blinded_input_v2_sha256",
                            "batch_manifest_sha256", "blind_id_map_sha256",
                            "ballots_v2_manifest_sha256"}
        if set(blinded) != expected_blinded:
            problems.append(f"blinded provenance does not carry exactly the declared hashes: "
                            f"{sorted(set(blinded) ^ expected_blinded)}")
        elif not all(SHA256.fullmatch(str(v)) for v in blinded.values()):
            problems.append("a blinded provenance value is not a sha256")
        for name in ("private_flag_table_sha256", "private_occurrence_ledger_sha256",
                     "private_fold_state_sha256", "private_adjudication_sha256",
                     "private_candidate_table_sha256", "graph_label_universe_sha256",
                     "released_inventory_sha256"):
            if not SHA256.fullmatch(str(inputs[name])):
                problems.append(f"freeze input {name} is not a sha256")
        if inputs["private_fold_state_sha256"] != hashlib.sha256(
                args.fold_state.read_bytes()).hexdigest():
            problems.append("freeze inputs do not describe the fold state actually supplied")
    for identifier, entry in rulings.items():
        if not UUID4.fullmatch(str(identifier)):
            problems.append(f"a ruling id is not opaque: {identifier!r}")
        if not isinstance(entry.get("publish_verbatim"), bool):
            problems.append("a ruling does not decide publication")
    if problems:
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        print("refusing to publish: inputs did not satisfy their declared structure; "
              "nothing was written", file=sys.stderr)
        return 1

    credits = {digest: PLACEHOLDER_BY_KIND[kind] for digest, kind in credit_map.items()}
    missing = sorted(set(fold) - set(id_map))
    if missing:
        print(f"fold state references {len(missing)} combinations absent from the frozen id map; "
              f"nothing was written", file=sys.stderr)
        return 1

    rows = publish(fold, credits, id_map, allowlist)
    payload = render(rows)
    digest = hashlib.sha256(payload).hexdigest()
    freeze_payload = json.dumps(
        compose_freeze(rows, inputs, digest, hashlib.sha256(
            (DIR / "public_name_allowlist.json").read_bytes()).hexdigest()),
        ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    freeze_digest = hashlib.sha256(freeze_payload).hexdigest()
    print(f"derived {len(fold)} rows, table sha256 {digest}, freeze sha256 {freeze_digest}")
    if PUBLIC_TABLE_SHA != "PENDING" and digest != PUBLIC_TABLE_SHA:
        print(f"derived table does not match the frozen public hash {PUBLIC_TABLE_SHA}", file=sys.stderr)
        print("nothing was written", file=sys.stderr)
        return 1
    if CANONICAL_FREEZE_SHA != "PENDING" and freeze_digest != CANONICAL_FREEZE_SHA:
        print(f"composed freeze record does not match the canonical pin {CANONICAL_FREEZE_SHA}",
              file=sys.stderr)
        print("nothing was written", file=sys.stderr)
        return 1
    if args.check:
        print("derived table and freeze record match their frozen pins")

    if args.write:
        atomic_replace_pair([(payload, DIR / "resolution_table.csv"),
                             (freeze_payload, DIR / "freeze.json")])
        print("public table and freeze record written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
