"""Adversarial checks on the NER-CR-001 gate, publisher and release packaging.

Every scenario here is a way the previous versions could be made to pass while publishing
something they should have withheld, or to publish without the inputs that make the output
meaningful. Each one is executed against the real tools on a throwaway copy of the
artefacts, so a regression fails the build rather than being argued about.

No private data and no third-party dependencies:

    python tests/test_compound_resolution_gate.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def locate() -> tuple[Path, str, str, str]:
    """Work in a repository checkout and inside the desktop package alike.

    In the package the artefacts sit under `Results/` and the tools under
    `Reproducibility/tools/`. A packaged test that cannot run is worse than none.
    """
    here = Path(__file__).resolve().parent
    layouts = (
        (here.parent, "analysis/compound-resolution", "tools", "results/ner-v1"),
        (here.parent.parent, "Results/compound-resolution-ner-cr-001",
         "Reproducibility/tools", "Results/ner-v1"),
    )
    for base, artefacts, tools, results in layouts:
        if (base / artefacts / "resolution_table.csv").is_file() and (base / tools).is_dir():
            return base, artefacts, tools, results
    raise SystemExit("could not locate the compound resolution artefacts")


ROOT, ARTEFACTS_DIR, TOOLS_DIR, RESULTS_DIR = locate()
ARTEFACTS = ROOT / ARTEFACTS_DIR
PACKAGED = ARTEFACTS_DIR.startswith("Results/")
TOOLS = ROOT / TOOLS_DIR
SRC = ROOT / ("Reproducibility/src" if PACKAGED else "src")
UUID4_RE = __import__("re").compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{': ' + detail if detail else ''}")
        FAILURES.append(label)


def sandbox() -> Path:
    """A copy of everything the gate reads, so a scenario can corrupt it freely.

    The sandbox always uses the repository layout, whichever layout the test itself was
    started from, so the gate's own path resolution is exercised the same way in both.
    """
    target = Path(tempfile.mkdtemp(prefix="ner-cr-001-"))
    for source_relative, sandbox_relative in ((TOOLS_DIR, "tools"),
                                              (ARTEFACTS_DIR, "analysis/compound-resolution"),
                                              (RESULTS_DIR, "results/ner-v1")):
        destination = target / sandbox_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / source_relative, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return target


def run(target: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(target / "tools" / arguments[0]), *arguments[1:]],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def verify(target: Path, *arguments: str) -> int:
    return run(target, "verify_compound_resolution.py", *arguments).returncode


def read_freeze(target: Path) -> dict:
    return json.loads((target / "analysis/compound-resolution/freeze.json").read_text(encoding="utf-8"))


def write_freeze(target: Path, freeze: dict) -> None:
    (target / "analysis/compound-resolution/freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_table_at(directory: Path) -> list[dict[str, str]]:
    path = directory / "resolution_table.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_table(target: Path) -> list[dict[str, str]]:
    return read_table_at(target / "analysis/compound-resolution")


def write_table(target: Path, rows: list[dict[str, str]]) -> None:
    path = target / "analysis/compound-resolution/resolution_table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def resync(target: Path) -> None:
    """Update the record's public hash, the way a coordinated edit would."""
    freeze = read_freeze(target)
    table = target / "analysis/compound-resolution/resolution_table.csv"
    freeze["public_table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    write_freeze(target, freeze)


def test_gate_rejects_tampering() -> None:
    print("\ngate")
    clean = sandbox()
    check("an untouched copy passes", verify(clean) == 0)
    shutil.rmtree(clean, ignore_errors=True)

    scenarios: list[tuple[str, object]] = [
        ("an empty reference model map is rejected",
         lambda t: write_freeze(t, {**read_freeze(t), "reference_models": {}})),
        ("a model pinned to a different revision is rejected",
         lambda t: write_freeze(t, {**read_freeze(t), "reference_models": {
             name: "0" * 40 for name in read_freeze(t)["reference_models"]}})),
        ("a forged private candidate hash is rejected",
         lambda t: write_freeze(t, {**read_freeze(t), "inputs": {
             **read_freeze(t)["inputs"], "private_candidate_table_sha256": "not-a-hash"}})),
        ("a forged graph universe hash is rejected",
         lambda t: write_freeze(t, {**read_freeze(t), "inputs": {
             **read_freeze(t)["inputs"], "graph_label_universe_sha256": "0" * 64}})),
        ("edited totals are rejected",
         lambda t: write_freeze(t, {**read_freeze(t), "totals": {
             **read_freeze(t)["totals"], "flagged_occurrences": 900}})),
        ("edited action counts are rejected",
         lambda t: write_freeze(t, {**read_freeze(t), "action_counts": {
             "RETAIN_SHORT": {"rows": 1, "occurrences": 1, "strict_occurrences": 1}}})),
        ("a dropped human-review declaration is rejected",
         lambda t: write_freeze(t, {**read_freeze(t), "adjudication": {
             **read_freeze(t)["adjudication"], "independent_human_review_status": "complete"}})),
        ("claiming human gold is rejected",
         lambda t: write_freeze(t, {**read_freeze(t), "adjudication": {
             **read_freeze(t)["adjudication"], "is_human_gold": True}})),
        ("dropping the per-occurrence ledger declaration is rejected",
         lambda t: write_freeze(t, {**read_freeze(t), "occurrence_ledger": {
             **read_freeze(t)["occurrence_ledger"], "every_occurrence_appears_exactly_once": False}})),
    ]
    for label, mutate in scenarios:
        target = sandbox()
        mutate(target)
        check(label, verify(target) == 1)
        shutil.rmtree(target, ignore_errors=True)


def test_gate_rejects_coordinated_edits() -> None:
    print("\ncoordinated edits")
    cases = [
        ("a content-derived combination id is rejected",
         lambda rows: [{**r, "combination_id": hashlib.sha256(
             (r["short_surface"] + r["containing_span"]).encode("utf-8")).hexdigest()[:12]} for r in rows]),
        ("a duplicated combination id is rejected",
         lambda rows: [{**r, "combination_id": rows[0]["combination_id"]} for r in rows]),
        ("strict occurrences above total occurrences are rejected",
         lambda rows: [{**r, "flagged_strict_occurrences": str(int(r["flagged_occurrences"]) + 1)}
                       if r is rows[0] else r for r in rows]),
        ("a negative count is rejected",
         lambda rows: [{**r, "flagged_occurrences": "-1"} if r is rows[0] else r for r in rows]),
        ("dropping proposing_models is rejected",
         lambda rows: [{**r, "proposing_models": ""} if r is rows[0] else r for r in rows]),
        ("an unpinned proposing model is rejected",
         lambda rows: [{**r, "proposing_models": "some/other-model"} if r is rows[0] else r for r in rows]),
        ("retain_short disagreeing with the action is rejected",
         lambda rows: [{**r, "retain_short": "yes"} if r["resolution_action"] == "DROP_SHORT" else r
                       for r in rows]),
        ("free text in a rationale is rejected",
         lambda rows: [{**r, "resolution_rationale": "because the line says so"} if r is rows[0] else r
                       for r in rows]),
        ("an ungeneralised production credit is rejected",
         lambda rows: [{**r, "containing_span": "some company"}
                       if r["span_kind"] == "production_credit_generalised" else r for r in rows]),
        ("a structural span carrying other text is rejected",
         lambda rows: [{**r, "containing_span": r["short_surface"] + "+someplace"}
                       if r["span_kind"] == "merged_span_structural" else r for r in rows]),
        ("an unbucketed span length is rejected",
         lambda rows: [{**r, "containing_span_length_bucket": "7"} if r is rows[0] else r for r in rows]),
    ]
    for label, mutate in cases:
        target = sandbox()
        write_table(target, mutate(read_table(target)))
        resync(target)          # the record is edited to match, as a real tamper would
        check(label, verify(target) == 1)
        shutil.rmtree(target, ignore_errors=True)

    target = sandbox()
    rows = read_table(target)
    write_table(target, rows[:-1])
    resync(target)
    check("dropping a row is rejected even with the record resynced", verify(target) == 1)
    shutil.rmtree(target, ignore_errors=True)

    target = sandbox()
    rows = read_table(target)
    write_table(target, [{k: v for k, v in r.items() if k != "credit_region_occurrences"} for r in rows])
    resync(target)
    check("removing a column is rejected", verify(target) == 1)
    shutil.rmtree(target, ignore_errors=True)

    # the whole record is pinned, so a field the gate does not individually understand
    # cannot be edited either
    for field, path in (("three-vote count", ("adjudication", "contested_occurrences_taken_to_a_three-vote_majority")),
                        ("both-model count", ("combinations_flagged_by_both_models",)),
                        ("not-folded count", ("occurrence_ledger", "combinations_not_folded")),
                        ("duplicate count", ("occurrence_ledger", "occurrences_sharing_content_and_char_span_across_chunks")),
                        ("credit-region count", ("occurrence_ledger", "occurrences_in_production_credit_regions")),
                        ("adjudication round count", ("adjudication", "rounds"))):
        target = sandbox()
        freeze = read_freeze(target)
        cursor = freeze
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = 999
        write_freeze(target, freeze)
        check(f"an edited {field} in the record is rejected", verify(target) == 1)
        shutil.rmtree(target, ignore_errors=True)


def test_publisher_fails_closed() -> None:
    print("\npublisher")
    module = TOOLS / "publish_compound_resolution.py"
    source = module.read_text(encoding="utf-8")
    check("the publisher requires its credits map", '"--credits", type=Path, required=True' in source)
    check("the publisher requires the frozen id map", '"--ids", type=Path, required=True' in source)
    check("the publisher pins the credits map hash",
          'CREDITS_MAP_SHA = "' in source and 'CREDITS_MAP_SHA = "PENDING"' not in source)
    check("the publisher pins the id map hash",
          'IDS_MAP_SHA = "' in source and 'IDS_MAP_SHA = "PENDING"' not in source)
    check("the publisher writes through a temporary file", "os.replace(temporary" in source)
    check("the publisher declares its exact credit count", "CREDIT_ENTRIES = 14" in source)

    target = sandbox()
    private = target / "private"
    private.mkdir()
    before = (target / "analysis/compound-resolution/resolution_table.csv").read_bytes()
    for label, payload in (("missing", None),
                           ("partial", {"0" * 64: "company"}),
                           ("wrong-kind", {"0" * 64: "not-a-kind"}),
                           ("not-a-hash-keyed", {"company": "company"})):
        credits = private / f"credits_{label}.json"
        if payload is not None:
            credits.write_text(json.dumps(payload), encoding="utf-8")
        fold = private / "fold.json"
        fold.write_text("{}", encoding="utf-8")
        ids = private / "ids.json"
        ids.write_text("{}", encoding="utf-8")
        hashes = private / "hashes.json"
        hashes.write_text("{}", encoding="utf-8")
        result = run(target, "publish_compound_resolution.py", "--fold-state", str(fold),
                     "--credits", str(credits), "--ids", str(ids), "--hashes", str(hashes), "--write")
        check(f"a {label} credits map fails closed", result.returncode != 0)
    check("nothing was written by any failed publish",
          (target / "analysis/compound-resolution/resolution_table.csv").read_bytes() == before)
    shutil.rmtree(target, ignore_errors=True)


def test_redaction_rules() -> None:
    print("\nredaction")
    sys.path.insert(0, str(TOOLS))
    import importlib.util
    spec = importlib.util.spec_from_file_location("publisher", TOOLS / "publish_compound_resolution.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)

    # a line break must be detected on its own, not only when a whole segment is blank
    check("a break between two runs is marked",
          publisher.structural("甲乙", "甲乙\n丙丁") == "甲乙+<BREAK>+<OTHER_PLACE>")
    check("a break before the surface is marked",
          publisher.structural("甲乙", "丙丁\n甲乙") == "<OTHER_PLACE>+<BREAK>+甲乙")
    check("a bare break is marked", publisher.structural("甲乙", "甲乙\n") == "甲乙+<BREAK>")
    check("a latin run is generalised",
          publisher.structural("甲乙", "甲乙 Room") == "甲乙+<LATIN_RUN>")
    check("a single foreign character is marked separately",
          publisher.structural("甲乙", "甲乙丙") == "甲乙+<HAN>")
    check("a bilingual gloss keeps no occurrence-specific writing",
          publisher.structural("甲乙", "Cali (甲乙）") == "<LATIN_RUN>+甲乙")
    check("length buckets never expose an exact length",
          {publisher.bucket(n) for n in range(2, 40)} == {"2-3", "4-6", "7-12", "13+"})

    rows = read_table_at(ARTEFACTS)
    verbatim = [r for r in rows if r["span_kind"] == "public_entity"]
    doc = json.loads((ARTEFACTS / "public_name_allowlist.json").read_text(encoding="utf-8"))
    approved = {entry["span"] for entry in doc["approved"].values()}
    check("every verbatim span carries an approved ruling",
          all(r["containing_span"] in approved for r in verbatim))
    check("the allowlist still withholds some rulings", doc["withheld"]["count"] > 0)
    # the disclosure defect this file previously carried: a ruling that withholds a string
    # must not print that string
    check("withheld rulings appear only as opaque identifiers",
          all(UUID4_RE.fullmatch(str(i)) for i in doc["withheld"]["ids"])
          and "span" not in json.dumps(doc["withheld"])),
    check("the withheld section carries no free text at all",
          set(doc["withheld"]) == {"count", "ids"})
    check("the allowlist binds the private rulings file by hash",
          len(doc["adjudication"].get("private_rulings_sha256", "")) == 64)
    check("no undeclared section carries content",
          set(doc) == {"protocol", "role", "rule", "adjudication", "disclosure",
                       "approved", "withheld"})
    check("every verbatim span is a distinct entity",
          all(r["span_relation"] == "DISTINCT_ENTITY" for r in verbatim))
    check("no row publishes a project schema type for a reference model span",
          all(r["longer_span_disposition"] in {
              "NEW_CANDIDATE_REQUIRES_STANDARD_GATE", "NOT_A_CANDIDATE_TAGGER_ARTEFACT",
              "NOT_A_CANDIDATE_SAME_REFERENT", "NOT_ASSESSED_RELATION_UNKNOWN",
              "NOT_ASSESSED_RELATION_MIXED"} for r in rows))
    check("the span relation and the text region are separate columns",
          {"span_relation", "text_region"} <= set(rows[0]))
    check("a credit-region row accounts for every one of its occurrences",
          all(int(r["credit_region_occurrences"]) == int(r["flagged_occurrences"])
              for r in rows if r["text_region"] == "PRODUCTION_CREDIT"))
    check("a lyric-region row carries no credit-region occurrences",
          all(int(r["credit_region_occurrences"]) == 0
              for r in rows if r["text_region"] == "LYRIC"))


def test_rollback_and_fold_comparison() -> None:
    print("\nrollback and fold comparison units")
    import importlib.util
    spec = importlib.util.spec_from_file_location("publisher2", TOOLS / "publish_compound_resolution.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    workspace = Path(tempfile.mkdtemp(prefix="rollback-"))
    try:
        first = workspace / "table.csv"
        second = workspace / "freeze.json"
        first.write_bytes(b"old table")
        second.mkdir()          # a directory target makes the SECOND os.replace fail
        raised = False
        try:
            publisher.atomic_replace_pair([(b"new table", first), (b"new freeze", second)])
        except OSError:
            raised = True
        check("a failing second replace raises", raised)
        check("and the first target is rolled back to its old bytes",
              first.read_bytes() == b"old table")
        check("and no temporary files are left behind",
              not list(workspace.glob("*.tmp")))
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    spec = importlib.util.spec_from_file_location("builder2", TOOLS / "build_compound_resolution_table.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    failures: list[str] = []
    builder.compare_pinned("fold state", "not-the-pin", builder.PRIVATE_FOLD_SHA, failures)
    check("the builder's fold comparison actually rejects a mismatch",
          bool(failures) and "fold state" in failures[0])
    failures = []
    builder.compare_pinned("fold state", builder.PRIVATE_FOLD_SHA, builder.PRIVATE_FOLD_SHA, failures)
    check("and accepts the pinned value", not failures)
    source = (TOOLS / "build_compound_resolution_table.py").read_text(encoding="utf-8")
    check("the --check block routes the fold hash through that comparison",
          'compare_pinned("fold state", fold_sha, PRIVATE_FOLD_SHA' in source)


def test_blinded_provenance_is_pinned() -> None:
    print("\nblinded provenance")
    target = sandbox()
    freeze = read_freeze(target)
    check("the freeze carries the blinded provenance block",
          isinstance(freeze.get("blinded_provenance"), dict)
          and len(freeze["blinded_provenance"]) == 5)
    del freeze["blinded_provenance"]
    write_freeze(target, freeze)
    check("removing it is rejected", verify(target) == 1)
    shutil.rmtree(target, ignore_errors=True)

    target = sandbox()
    freeze = read_freeze(target)
    freeze["blinded_provenance"]["ballots_v2_manifest_sha256"] = "0" * 64
    write_freeze(target, freeze)
    check("a forged ballot manifest hash is rejected", verify(target) == 1)
    shutil.rmtree(target, ignore_errors=True)

    target = sandbox()
    freeze = read_freeze(target)
    freeze["adjudication"]["rulings"] = "independent expert opinions"
    write_freeze(target, freeze)
    check("overstating ruling independence is rejected", verify(target) == 1)
    shutil.rmtree(target, ignore_errors=True)

    allow = json.loads((ARTEFACTS / "public_name_allowlist.json").read_text(encoding="utf-8"))
    check("the allowlist declares itself a disclosure-safety gate only",
          "disclosure-safety gate" in allow.get("role", "")
          and "does not validate" in allow.get("role", ""))
    check("its rulings are recorded as AI-assisted repeated rulings",
          "AI-assisted repeated rulings" in allow["adjudication"]["method"])


def test_release_packaging() -> None:
    print("\npackaging")
    builder = (SRC / "build_chinese_rap_release_v4.py").read_text(encoding="utf-8")
    check("the release builder uses an explicit publishable allowlist",
          "PUBLISHABLE_COMPOUND_RESOLUTION_FILES" in builder and "PUBLISHABLE_TOOLS" in builder)
    check("the builder refuses to package an unlisted artefact file",
          "outside the publishable allowlist" in builder)
    check("the desktop archive is written deterministically",
          "write_deterministic_archive(target, archive)" in builder
          and "shutil.make_archive(" not in builder)
    check("archive members carry a fixed timestamp", "date_time=(1980, 1, 1, 0, 0, 0)" in builder)

    validator = (SRC / "validate_public_release_integrity_v1.py").read_text(encoding="utf-8")
    check("the validator requires the packaged tools",
          "Reproducibility/tools/verify_compound_resolution.py" in validator)
    check("the validator compares the packaged table byte for byte",
          "differs from the frozen repository copy" in validator)

    gate = (TOOLS / "verify_compound_resolution.py").read_text(encoding="utf-8")
    check("the gate can locate artefacts inside the desktop package",
          "Results/compound-resolution-ner-cr-001" in gate)


def main() -> int:
    for suite in (test_gate_rejects_tampering, test_gate_rejects_coordinated_edits,
                  test_publisher_fails_closed, test_redaction_rules,
                  test_rollback_and_fold_comparison, test_blinded_provenance_is_pinned,
                  test_release_packaging):
        suite()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  {name}")
        return 1
    print("all adversarial checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
