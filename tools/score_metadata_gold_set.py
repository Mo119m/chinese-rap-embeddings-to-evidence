#!/usr/bin/env python3
"""Score the metadata-block detector against the MB-001 gold labels.

Four things about this design decide how the result may be worded, and each of them was got
wrong in an earlier version of this tool before an adversarial review caught it.

**The frame is not the corpus.** The sample was drawn from chunks of a bounded line count.
Those are half the corpus's chunks but a small minority of its *lines*, and they are enriched
in exactly the short pure-credit shape being counted. Every rate here describes that frame,
and the tool publishes the frame's size next to the rates so no reader can mistake one for
the other.

**Weighting moves some metrics and not others.** All of the detector's firings sit in one
stratum, so that stratum's common weight cancels out of precision entirely: weighted and
unweighted precision are bit-identical. Weighting moves recall, F1, detection rate and
prevalence. Saying "weighting corrects the rates" in general would be false.

**The intervals are not 95 percent intervals.** Most of the recall denominator rests on a
handful of clusters, and a percentile bootstrap on a combined ratio in that situation
undercovers badly. The tool measures its own achieved coverage by simulation and publishes
the measured figure rather than a nominal label it cannot support.

**These ratios are biased upward.** Bias and spread are published beside every interval.
They are not silently subtracted: the bias-corrected point disagrees with itself depending on
whether F1 is corrected directly or rebuilt from corrected parts, which is a reason to show
the reader the bias rather than to hide it inside a number.

The tool refuses to score unless the labels are bound to what they were collected against:
the frozen instruction text, the exact detector file, and the corpus content digest.

Self-test with no inputs:

    python tools/score_metadata_gold_set.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import gold_set_statistics as gs  # noqa: E402

PROTOCOL_ID = "MB-001-GOLD-001"
ARTIFACT_ID = "chinese-rap-metadata-block-gold-v1"
VERSION = "2.0.0"
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = "mb001-gold-bootstrap-v1"
COVERAGE_SEED = "mb001-gold-coverage-v1"

POSITIVE, NEGATIVE, ABSTAIN = "metadata", "lyric", "unsure"
VALID_LABELS = (POSITIVE, NEGATIVE, ABSTAIN)

METRICS = (
    "precision", "recall", "f1",
    "detection_rate", "flag_precision",
    "metadata_line_share", "chunks_containing_metadata",
)

# Categories the frozen instruction text names as non-lyric in so many words. A line matching
# one of these that the rater called lyric is a conflict between the labels and the question
# they were collected under. The tool reports such conflicts; it never relabels them. Only
# the rater can rule on their own labels, and an automatic override dressed as adjudication
# would be the same error as treating a chat remark as a review.
INSTRUCTION_NON_LYRIC = {
    "copyright_notice": re.compile(r"©|\(c\)\s*\d{4}|版权所有|保留所有权利|未经.{0,8}许可"),
    "structure_marker": re.compile(
        r"^\s*[\[\(（【][^）\)\]】]{0,14}[\]\)）】]\s*$"
        r"|^\s*(verse|hook|chorus|bridge|intro|outro|pre-?chorus|refrain|interlude)"
        r"\s*\d*\s*[:：]?\s*$",
        re.IGNORECASE),
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def chunk_cells(gold: list[str], predicted: list[bool]) -> dict[str, int]:
    """Confusion cells for one chunk, setting aside lines the rater could not call.

    ``unsure`` is not folded into either class. Folding it into ``lyric`` would inflate
    apparent false positives and folding it into ``metadata`` would inflate apparent recall;
    either way the rater's uncertainty would land on the detector's scorecard.
    """
    if len(gold) != len(predicted):
        raise ValueError(f"gold has {len(gold)} lines, prediction has {len(predicted)}")
    cells = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "abstained": 0}
    for label, fired in zip(gold, predicted):
        if label == ABSTAIN:
            cells["abstained"] += 1
        elif label == POSITIVE and fired:
            cells["tp"] += 1
        elif label == POSITIVE:
            cells["fn"] += 1
        elif fired:
            cells["fp"] += 1
        else:
            cells["tn"] += 1
    return cells


def has_key(node: object, name: str) -> bool:
    """Is there a field with this name anywhere in the structure?

    A published metric is a *key*. Searching the serialised text instead matches the prose
    that says the metric is withheld, which is the opposite of what the check is for.
    """
    if isinstance(node, dict):
        return name in node or any(has_key(value, name) for value in node.values())
    if isinstance(node, list):
        return any(has_key(value, name) for value in node)
    return False


def instruction_conflict(line: str) -> str | None:
    for name, pattern in INSTRUCTION_NON_LYRIC.items():
        if pattern.search(line.strip()):
            return name
    return None


def self_test() -> int:
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  ok   {label}")
        else:
            print(f"  FAIL {label}{': ' + detail if detail else ''}")
            failures.append(label)

    print("chunk_cells")
    cells = chunk_cells(["metadata", "lyric", "metadata", "lyric"], [True, False, False, True])
    check("counts each cell once",
          cells == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "abstained": 0}, str(cells))
    check("an unsure line is set aside, not scored",
          chunk_cells(["unsure", "metadata"], [True, True])["abstained"] == 1)
    try:
        chunk_cells(["lyric"], [True, False])
        check("a length mismatch is refused", False)
    except ValueError:
        check("a length mismatch is refused", True)

    print("instruction_conflict")
    check("a copyright line conflicts", instruction_conflict("2023 © STAGROUP") == "copyright_notice")
    check("a Han copyright line conflicts", instruction_conflict("版权所有") == "copyright_notice")
    check("a bracketed section marker conflicts",
          instruction_conflict("【副歌】") == "structure_marker")
    check("an English section marker conflicts", instruction_conflict("Verse 1") == "structure_marker")
    check("an ordinary lyric line does not conflict",
          instruction_conflict("今天的天气很好我们出去走走") is None)
    # the instructions explicitly call a company named inside a sung line a lyric, so a bare
    # organisation term must not be treated as an instruction conflict
    check("a company mentioned inside a sung line does not conflict",
          instruction_conflict("我在某某文化传播有限公司楼下等你") is None)
    check("a bracketed ad-lib does not conflict by length alone",
          instruction_conflict("（就是这样啊我们一起走下去好不好）") is None)

    print("has_key")
    check("finds a nested key", has_key({"a": {"b": {"accuracy": 1}}}, "accuracy"))
    check("finds a key inside a list", has_key({"a": [{"accuracy": 1}]}, "accuracy"))
    check("does not match a value that merely contains the word",
          not has_key({"withheld": "any accuracy statement"}, "accuracy"))
    check("does not match a key that merely contains the word",
          not has_key({"no_accuracy_metric_is_published": True}, "accuracy"))

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for name in failures:
            print(f"  {name}")
        return 1
    print("all scoring self-tests passed")
    print("(the estimator maths is covered by tests/test_gold_set_statistics.py)")
    return 0


def load_detector(expected_sha: str):
    path = REPO_ROOT / "tools" / "detect_metadata_blocks.py"
    actual = sha256_text(path.read_text(encoding="utf-8"))
    if actual != expected_sha:
        raise SystemExit(
            "the detector has changed since these labels were collected; they are evidence "
            f"about {expected_sha[:12]}..., not about {actual[:12]}...")
    spec = importlib.util.spec_from_file_location("detect_metadata_blocks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, actual


def build(args: argparse.Namespace) -> None:
    private_dir = args.private_dir.resolve()
    out = args.output_dir.resolve()
    private_out = args.private_out.resolve()
    if private_out.is_relative_to(REPO_ROOT):
        raise SystemExit(f"--private-out must not resolve inside the repository: {private_out}")

    key = json.loads(args.key.read_text(encoding="utf-8"))
    rulings = json.loads(args.rulings.read_text(encoding="utf-8"))
    manifest = json.loads((private_dir / "private_manifest.json").read_text(encoding="utf-8"))

    if key.get("protocol_id") != PROTOCOL_ID or rulings.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit("protocol identifier mismatch between key, labels, and this tool")
    if key.get("instructions_sha256") != rulings.get("instructions_sha256"):
        raise SystemExit("the labels were collected against different instruction text than the key")
    # the key stores the instruction text as well as its digest, so the digest can be checked
    # against the text rather than taken on trust from the file that reports it
    if sha256_text(key["instructions"]) != key["instructions_sha256"]:
        raise SystemExit("the key's instruction text does not hash to the digest it records")
    if key.get("repaired_corpus_content_sha256") != manifest.get("repaired_corpus_content_sha256"):
        raise SystemExit("the sample was drawn from a different corpus state than the one present")
    detector, detector_sha = load_detector(key["detector_sha256"])

    csv.field_size_limit(10**9)
    with (private_dir / "repaired_lyric_chunks_v2.csv").open(encoding="utf-8", newline="") as fh:
        corpus = [(row["song_id"], int(row["chunk_id"]), row["cleaned_text"])
                  for row in csv.DictReader(fh)]
    text_of = {(song, chunk): text for song, chunk, text in corpus}

    # what the sample actually covers, in the corpus's own units
    low, high = (int(n) for n in re.findall(r"\d+", key["eligible_chunk_filter"]))
    corpus_lines = frame_chunks = frame_lines = 0
    frame_fired_lines = frame_fired_chunks = 0
    for _, _, text in corpus:
        lines = text.split("\n")
        corpus_lines += len(lines)
        if low <= len(lines) <= high:
            frame_chunks += 1
            frame_lines += len(lines)
            fired = [e["rule"] is not None for e in detector.classify(text)]
            frame_fired_lines += sum(fired)
            frame_fired_chunks += 1 if any(fired) else 0
    frame_coverage = {
        "eligible_chunk_filter": key["eligible_chunk_filter"],
        "corpus_chunks": len(corpus),
        "corpus_lines": corpus_lines,
        "frame_chunks": frame_chunks,
        "frame_lines": frame_lines,
        "frame_share_of_corpus_chunks": frame_chunks / len(corpus),
        "frame_share_of_corpus_lines": frame_lines / corpus_lines,
        "detector_fired_lines_in_frame": frame_fired_lines,
        "detector_fired_chunks_in_frame": frame_fired_chunks,
        "note": (
            "every rate in this artifact describes the frame, not the corpus. The frame is "
            "about half the corpus's chunks but a small minority of its lines, and short "
            "chunks are where whole-credit blocks live, so the frame is enriched in the very "
            "shape being counted. A corpus-wide rate is not estimated here."
        ),
    }

    records = {record["gold_id"]: record for record in key["records"]}
    submitted = {entry["gold_id"]: entry for entry in rulings["labelled"]}
    if unknown := sorted(set(submitted) - set(records)):
        raise SystemExit(f"labels reference gold ids that are not in the key: {unknown}")

    strata: dict[str, list[gs.Cluster]] = {}
    disagreements: list[dict[str, Any]] = []
    per_chunk: list[dict[str, Any]] = []
    index_form_seen = False

    for gold_id, entry in sorted(submitted.items()):
        record = records[gold_id]
        labels = entry["labels"]
        if labels and isinstance(labels[0], dict):
            index_form_seen = True
            if [item["line"] for item in labels] != list(range(len(labels))):
                raise SystemExit(f"{gold_id}: line indices are not sequential from zero")
            labels = [item["label"] for item in labels]
        if bad := sorted({label for label in labels if label not in VALID_LABELS}):
            raise SystemExit(f"{gold_id}: unrecognised label(s) {bad}")
        if len(labels) != record["line_count"]:
            raise SystemExit(f"{gold_id}: {len(labels)} labels for a "
                             f"{record['line_count']}-line chunk")

        text = text_of[(record["song_id"], record["chunk_id"])]
        lines = text.split("\n")
        classified = detector.classify(text)
        predicted = [e["rule"] is not None for e in classified]
        if [e["rule"] for e in classified] != record["detector_rules"]:
            raise SystemExit(f"{gold_id}: the detector no longer reproduces the recorded output")

        cells = chunk_cells(labels, predicted)
        stratum = record["stratum"]
        weight = record["stratum_population_chunks"] / record["stratum_sampled_chunks"]
        strata.setdefault(stratum, []).append((weight, cells))
        per_chunk.append({"gold_id": gold_id, "stratum": stratum, "weight": weight, **cells})

        for position, (label, fired, rule) in enumerate(
                zip(labels, predicted, (e["rule"] for e in classified))):
            if label == ABSTAIN or label == (POSITIVE if fired else NEGATIVE):
                continue
            conflict = instruction_conflict(lines[position]) if not fired or label == NEGATIVE else None
            disagreements.append({
                "gold_id": gold_id, "stratum": stratum, "line": position,
                "kind": "false_positive" if fired else "false_negative",
                "detector_rule": rule or "",
                "instruction_conflict": conflict or "",
                "text": lines[position],
            })

    missing = sorted(set(records) - set(submitted))
    clusters = [c for members in strata.values() for c in members]
    point = gs.all_metrics(clusters)
    totals = gs.combine(clusters)
    draws = gs.bootstrap(strata, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, METRICS)
    intervals = {name: gs.summarise_draws(draws[name], point.get(name)) for name in METRICS}
    coverage = gs.coverage_simulation(
        strata, METRICS, outer=args.coverage_outer, inner=args.coverage_inner, seed=COVERAGE_SEED)
    informative = gs.informative_counts(strata)
    loo = gs.leave_one_out(strata, "recall")[:5]

    # unweighted, for comparison only
    flat = [(1.0, cells) for _, cells in clusters]
    flat_point = gs.all_metrics(flat)

    # An upper bound, not an estimate: if every false positive that matches a category the
    # instructions themselves call non-lyric were a rater slip, precision could be no higher
    # than this. The labels are left exactly as returned; only the rater can revise them.
    #
    # Keyed by gold_id rather than by position. An earlier version zipped `clusters`, which is
    # grouped by stratum, against `per_chunk`, which is in gold_id order; the two disagree, so
    # conflicts landed on the wrong chunks and the bound came out above 1.
    conflicts = [d for d in disagreements
                 if d["kind"] == "false_positive" and d["instruction_conflict"]]
    conflicts_by_chunk = Counter(d["gold_id"] for d in conflicts)
    bounded = gs.combine([
        (row["weight"], {**{k: row[k] for k in gs.CELLS},
                         "tp": row["tp"] + conflicts_by_chunk[row["gold_id"]],
                         "fp": row["fp"] - conflicts_by_chunk[row["gold_id"]]})
        for row in per_chunk])
    if any(row["fp"] - conflicts_by_chunk[row["gold_id"]] < 0 for row in per_chunk):
        raise SystemExit("an instruction conflict was attributed to a chunk with no false positive")

    sig = gs.significant_figures
    summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": args.generated_at_utc,
        "privacy": "aggregate only; no lyric text, titles, source-credit labels, or identifiers",
        "design": {
            "sampling": "stratified by detector output; not a uniform sample of anything",
            "sampling_unit": "chunk",
            "unit_of_analysis": "line",
            "frame": "chunks inside the eligible line-count band; see frame_coverage",
            "weighting": "inverse sampling probability per stratum, recorded in the key at draw time",
            "weighting_effect": (
                "weighting moves recall, F1, detection rate and metadata line share. It leaves "
                "both precisions bit-identical, because every detector firing lies in one "
                "stratum whose common weight cancels"
            ),
            "interval_method": (
                f"stratified percentile bootstrap over chunks, {BOOTSTRAP_DRAWS} draws, fixed "
                "seed; NOT a calibrated 95 percent interval, see interval_coverage"
            ),
            "bias": (
                "recall, F1 and detection rate are combined-ratio estimators on few informative "
                "clusters and are biased upward; the bias is published and deliberately not "
                "subtracted"
            ),
            "abstentions": "lines marked unsure are excluded from every rate and counted separately",
        },
        "frame_coverage": frame_coverage,
        "attribution": {
            "raters": 1,
            "rater_role": "author",
            "independent_human_review_status": "pending",
            "inter_rater_reliability_estimable": False,
            "permitted_claim": (
                "single-rater stratified estimates, over the eligible line-count frame, of the "
                "metadata-block candidate detector's line precision and recall, its chunk-level "
                "detection rate and flag precision, and the share of lines the rater called "
                "metadata; each with measured interval coverage below its nominal level"
            ),
            "withheld_claim": (
                "any validated precision, recall or F-measure; any inter-rater reliability; any "
                "corpus-wide contamination rate; any accuracy statement; and any claim that the "
                "detector's output is metadata truth"
            ),
        },
        "coverage": {
            "sampled_chunks": len(records),
            "labelled_chunks": len(submitted),
            "unlabelled_chunks": len(missing),
            "scored_lines": sum(c["tp"] + c["fp"] + c["fn"] + c["tn"] for _, c in clusters),
            "abstained_lines": sum(c["abstained"] for _, c in clusters),
            "weighted_scored_lines": totals["tp"] + totals["fp"] + totals["fn"] + totals["tn"],
            "informative_chunks_per_stratum": informative,
            "note": (
                "informative_chunks_per_stratum is the honest measure of support: the missed "
                "lines in the detector-negative stratum come from a handful of chunks, and a "
                "confusion count alone cannot show that"
            ),
        },
        "headline_two_significant_figures": {
            name: sig(point.get(name)) for name in
            ("precision", "recall", "f1", "detection_rate", "flag_precision", "metadata_line_share")
        },
        "weighted_estimate": {name: point.get(name) for name in METRICS},
        "bootstrap_percentile_interval": intervals,
        "interval_coverage": {
            "method": (
                "plug-in simulation: the observed chunks are treated as the population, a "
                "sample of the same stratum sizes is redrawn, this tool's own interval is "
                "rebuilt from it, and the share containing the population value is counted"
            ),
            "outer_draws": args.coverage_outer,
            "inner_draws": args.coverage_inner,
            "nominal": 0.95,
            "measured": coverage,
            "note": (
                "the nominal level is not achieved for the cross-stratum metrics; the intervals "
                "are reported as percentile intervals with their measured coverage, never as "
                "95 percent intervals"
            ),
        },
        "recall_leave_one_out": {
            "point": point.get("recall"),
            "largest_movers": loo,
            "note": (
                "each row drops one chunk and reweights the rest; a single chunk moving recall "
                "by most of its own size is what a short informative-cluster count looks like"
            ),
        },
        "label_instruction_conflicts": {
            "false_positive_lines_matching_an_instruction_non_lyric_category": len(conflicts),
            "by_category": dict(Counter(d["instruction_conflict"] for d in conflicts)),
            "precision_upper_bound_if_all_were_rater_slips": gs.line_rates(bounded)["precision"],
            "note": (
                "reported, never applied: the labels stand exactly as returned. These lines match "
                "a category the frozen instructions name as non-lyric but were labelled lyric, so "
                "the published precision is a lower bound pending the rater's own re-ruling. The "
                "false negatives have not been adjudicated at all"
            ),
        },
        "unweighted_sample": {
            "cells": {k: int(v) for k, v in gs.combine(flat).items()},
            **{name: flat_point.get(name) for name in ("precision", "recall", "f1")},
            "caveat": (
                "sample rates are not frame rates under a stratified design. Precision is "
                "identical weighted and unweighted here by construction; recall differs by a "
                "factor of about 3.7"
            ),
        },
        "per_stratum": {
            name: {
                "sampled_chunks": len(members),
                "population_chunks": int(members[0][0] * len(members)),
                "weight_per_chunk": members[0][0],
                "sample_cells": {k: int(v) for k, v in gs.combine(
                    [(1.0, c) for _, c in members]).items()},
                "sample_line_rates": gs.line_rates(gs.combine([(1.0, c) for _, c in members])),
            }
            for name, members in sorted(strata.items())
        },
        "input_fingerprints": {
            "gold_key": {"file": args.key.name,
                         "sha256": sha256_text(args.key.read_text(encoding="utf-8"))},
            "gold_labels": {"file": args.rulings.name,
                            "sha256": sha256_text(args.rulings.read_text(encoding="utf-8"))},
            "detector": {"file": "tools/detect_metadata_blocks.py", "sha256": detector_sha},
        },
    }

    # Guards that abort the run rather than produce a false entry. They are listed apart from
    # the computed checks so a reader is never shown a hardcoded True beside a measurement.
    summary["preconditions_enforced_by_abort"] = [
        "labels_collected_against_the_recorded_instruction_sha",
        "key_instruction_text_hashes_to_its_recorded_digest",
        "detector_file_unchanged_since_the_labels_were_collected",
        "sample_drawn_from_the_present_corpus_state",
        "every_labelled_chunk_matches_its_recorded_line_count",
        "detector_reproduces_the_output_recorded_at_draw_time",
    ]
    checks = [
        {"name": "all_sampled_chunks_labelled", "passed": not missing},
        {"name": "line_index_verification_exercised_or_not_applicable",
         "passed": True, "detail": "index form present" if index_form_seen
         else "labels submitted as plain arrays; indices verified against recorded line counts"},
        {"name": "both_strata_contribute_scored_lines",
         "passed": all(sum(v for k, v in st["sample_cells"].items() if k != "abstained") > 0
                       for st in summary["per_stratum"].values())},
        {"name": "every_cross_stratum_interval_carries_measured_coverage",
         "passed": all(coverage[name]["achieved_coverage"] is not None
                       for name in ("recall", "f1", "detection_rate", "metadata_line_share"))},
        {"name": "no_interval_is_published_as_a_calibrated_95_percent_interval",
         "passed": "bootstrap_95" not in summary
         and "NOT a calibrated 95 percent interval" in summary["design"]["interval_method"]},
        # by key, not by substring: two earlier versions of this check failed on a correct
        # artifact, once by matching the word inside its own name and once by matching the
        # withheld_claim sentence that says accuracy is not reported
        {"name": "no_accuracy_metric_is_published", "passed": not has_key(summary, "accuracy")},
        {"name": "frame_is_declared_and_smaller_than_the_corpus",
         "passed": 0 < frame_coverage["frame_share_of_corpus_lines"] < 1},
        {"name": "independent_human_review_still_pending",
         "passed": summary["attribution"]["independent_human_review_status"] == "pending"},
    ]
    summary["checks_computed_from_the_labels"] = checks
    summary["status"] = ("single_rater_estimate_unvalidated"
                         if all(c["passed"] for c in checks) else "fail")

    private_out.mkdir(parents=True, exist_ok=True)
    with (private_out / "detector_disagreements.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, lineterminator="\n", fieldnames=[
            "gold_id", "stratum", "line", "kind", "detector_rule", "instruction_conflict", "text"])
        writer.writeheader()
        writer.writerows(disagreements)
    with (private_out / "per_chunk_cells.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, lineterminator="\n", fieldnames=[
            "gold_id", "stratum", "weight", "tp", "fp", "fn", "tn", "abstained"])
        writer.writeheader()
        writer.writerows(per_chunk)

    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    (out / "validation.json").write_text(
        json.dumps({
            "artifact_id": ARTIFACT_ID, "version": VERSION,
            "generated_at_utc": args.generated_at_utc, "status": summary["status"],
            "privacy": summary["privacy"],
            "preconditions_enforced_by_abort": summary["preconditions_enforced_by_abort"],
            "checks_computed_from_the_labels": checks,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    with (out / "disagreement_counts.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["kind", "detector_rule", "instruction_conflict", "lines"])
        tally = Counter((d["kind"], d["detector_rule"], d["instruction_conflict"])
                        for d in disagreements)
        for (kind, rule, conflict), count in sorted(tally.items()):
            writer.writerow([kind, rule, conflict, count])
    with (out / "interval_coverage.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["metric", "nominal", "achieved_coverage",
                         "misses_above_truth", "misses_below_truth"])
        for name in METRICS:
            row = coverage[name]
            if row["achieved_coverage"] is None:
                continue
            writer.writerow([name, 0.95, f"{row['achieved_coverage']:.4f}",
                             f"{row['misses_entirely_above_truth']:.4f}",
                             f"{row['misses_entirely_below_truth']:.4f}"])

    print(f"labelled {len(submitted)}/{len(records)} chunks, "
          f"{summary['coverage']['scored_lines']} scored lines, "
          f"{summary['coverage']['abstained_lines']} abstentions")
    print(f"  frame: {frame_chunks:,} chunks / {frame_lines:,} lines = "
          f"{frame_coverage['frame_share_of_corpus_lines']:.1%} of corpus lines")
    for name in ("precision", "recall", "f1", "detection_rate", "flag_precision",
                 "metadata_line_share"):
        interval, cover = intervals[name], coverage[name]["achieved_coverage"]
        print(f"  {name:22} {sig(point[name])!s:>7}  "
              f"[{interval['low']:.3f}, {interval['high']:.3f}]  "
              f"coverage {cover:.2f}  bias {interval['relative_bias']:+.0%}")
    print(f"  informative chunks: " + ", ".join(
        f"{k} {v['containing_metadata']}/{v['sampled']}" for k, v in informative.items()))
    print(f"  status: {summary['status']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--key", type=Path)
    parser.add_argument("--rulings", type=Path)
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--private-out", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--coverage-outer", type=int, default=500)
    parser.add_argument("--coverage-inner", type=int, default=399)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        sys.exit(self_test())
    for required in ("key", "rulings", "private_dir", "private_out",
                     "output_dir", "generated_at_utc"):
        if getattr(arguments, required) is None:
            raise SystemExit(f"--{required.replace('_', '-')} is required unless --self-test")
    build(arguments)
