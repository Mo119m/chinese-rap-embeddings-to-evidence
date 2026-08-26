#!/usr/bin/env python3
"""Validate the conservative clean-text sidecar for canonical lyrics."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ID = "canonical-lyric-text-sidecar-v1"
OUTPUT_DIR = ROOT / "outputs" / ARTIFACT_ID
PRIVATE_DIR = ROOT / "work" / "private-canonical-lyric-text-sidecar-v1"
ANALYSIS_POINTER = ROOT / "outputs" / "chinese-rap-analysis-input-v1" / "private_analysis_input_pointer.json"

EXPECTED_SONGS = 7211
EXPECTED_CHUNKS = 22128
PUBLIC_ALLOWLIST = {
    "README.md", "method_and_limits.md", "data_dictionary.md", "manifest.json", "validation.json",
    "cleaning_summary.json", "header_rule_summary.csv", "private_sidecar_pointer.json",
}
PRIVATE_ALLOWLIST = {
    "cleaned_analysis_chunks_v1.csv", "header_cleaning_audit_v1.csv", "private_manifest.json", "private_validation.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def validate() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for directory, allowlist, label in (
        (OUTPUT_DIR, PUBLIC_ALLOWLIST, "public"),
        (PRIVATE_DIR, PRIVATE_ALLOWLIST, "private"),
    ):
        actual = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
        checks.append(
            {
                "name": f"{label}_allowlist",
                "passed": directory.is_dir() and actual == allowlist and not any(path.is_dir() for path in directory.iterdir()),
                "detail": {"missing": sorted(allowlist - actual), "unexpected": sorted(actual - allowlist)},
            }
        )
    if not all(check["passed"] for check in checks):
        return {"artifact_id": ARTIFACT_ID, "status": "fail", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks}

    manifest = read_json(OUTPUT_DIR / "manifest.json")
    summary = read_json(OUTPUT_DIR / "cleaning_summary.json")
    private_pointer = read_json(OUTPUT_DIR / "private_sidecar_pointer.json")
    private_manifest = read_json(PRIVATE_DIR / "private_manifest.json")
    checks.append(
        {
            "name": "artifact_identity_consistent",
            "passed": all(item.get("artifact_id") == ARTIFACT_ID for item in (manifest, summary, private_pointer, private_manifest)),
        }
    )
    source_pointer = read_json(ANALYSIS_POINTER)
    source_dir = (ROOT / str(source_pointer["private_directory"])).resolve()
    checks.append(
        {
            "name": "canonical_input_hashes_current",
            "passed": manifest.get("canonical_input", {}).get("private_chunk_input_sha256") == sha256_file(source_dir / str(source_pointer["chunk_file"]))
            and manifest.get("canonical_input", {}).get("private_song_input_sha256") == sha256_file(source_dir / str(source_pointer["song_file"])),
        }
    )

    public_hashes = manifest.get("output_files", {})
    public_hash_ok = isinstance(public_hashes, dict)
    for name, metadata in public_hashes.items() if isinstance(public_hashes, dict) else []:
        public_hash_ok = public_hash_ok and (OUTPUT_DIR / name).is_file() and metadata.get("sha256") == sha256_file(OUTPUT_DIR / name)
    checks.append({"name": "public_payload_hashes_match", "passed": bool(public_hash_ok)})
    private_hashes = private_manifest.get("files", {})
    private_hash_ok = isinstance(private_hashes, dict)
    for name, metadata in private_hashes.items() if isinstance(private_hashes, dict) else []:
        private_hash_ok = private_hash_ok and (PRIVATE_DIR / name).is_file() and metadata.get("sha256") == sha256_file(PRIVATE_DIR / name)
    checks.append({"name": "private_payload_hashes_match", "passed": bool(private_hash_ok)})

    clean_rows = read_csv(PRIVATE_DIR / "cleaned_analysis_chunks_v1.csv")
    audit_rows = read_csv(PRIVATE_DIR / "header_cleaning_audit_v1.csv")
    keys = [(row.get("song_id"), row.get("chunk_id")) for row in clean_rows]
    counts = summary.get("counts", {})
    checks.append(
        {
            "name": "canonical_population_preserved",
            "passed": len(clean_rows) == EXPECTED_CHUNKS
            and len(set(keys)) == EXPECTED_CHUNKS
            and counts.get("input_chunks") == EXPECTED_CHUNKS
            and counts.get("input_songs") == EXPECTED_SONGS,
        }
    )
    allowed_statuses = {"eligible_clean_text", "withheld_metadata_only"}
    clean_statuses = {row.get("analysis_text_status") for row in clean_rows}
    checks.append(
        {
            "name": "clean_text_statuses_and_hashes_valid",
            "passed": clean_statuses.issubset(allowed_statuses)
            and all(
                (
                    row.get("analysis_text_status") == "withheld_metadata_only"
                    and row.get("analysis_text", "") == ""
                    and row.get("analysis_text_sha256", "") == ""
                )
                or (
                    row.get("analysis_text_status") == "eligible_clean_text"
                    and bool(row.get("analysis_text", ""))
                    and hashlib.sha256(row.get("analysis_text", "").encode("utf-8")).hexdigest() == row.get("analysis_text_sha256")
                )
                for row in clean_rows
            ),
        }
    )
    changed_rows = [row for row in clean_rows if int(row.get("leading_header_lines_removed", "0")) > 0]
    checks.append(
        {
            "name": "audit_rows_exactly_match_changed_chunks",
            "passed": len(changed_rows) == len(audit_rows) == counts.get("changed_chunks")
            and {(row["song_id"], row["chunk_id"]) for row in changed_rows} == {(row["song_id"], row["chunk_id"]) for row in audit_rows},
        }
    )
    allowed_rules = {
        "leading_credit_or_production_cue", "leading_feature_credit_cue",
        "title_like_line_adjacent_to_header_cue", "leading_name_list_after_header_cue",
        "leading_structure_marker", "leading_copyright_notice",
    }
    rules_ok = all(
        set(filter(None, row.get("header_rule_ids", "").split(";"))).issubset(allowed_rules)
        for row in changed_rows
    )
    checks.append({"name": "only_declared_cleaning_rules_used", "passed": rules_ok})
    checks.append(
        {
            "name": "summary_counts_reconcile",
            "passed": counts.get("changed_chunks") + counts.get("unchanged_chunks") == EXPECTED_CHUNKS
            and counts.get("eligible_clean_text_chunks") + counts.get("withheld_metadata_only_chunks") == EXPECTED_CHUNKS
            and counts.get("review_queue_chunks") == sum(row.get("review_priority") == "review" for row in audit_rows),
        }
    )
    header_rules = read_csv(OUTPUT_DIR / "header_rule_summary.csv")
    summary_rule_count = sum(int(row.get("removed_line_count", "0")) for row in header_rules)
    # A removed line may have multiple legitimate rule IDs, so this aggregate
    # is at least the number of removed lines rather than exactly equal.
    checks.append(
        {
            "name": "rule_summary_covers_every_removed_line",
            "passed": summary_rule_count >= counts.get("leading_header_lines_removed", 0),
            "detail": {"rule_hits": summary_rule_count, "removed_lines": counts.get("leading_header_lines_removed")},
        }
    )
    public_headers_ok = True
    prohibited = {"song_id", "chunk_id", "text", "analysis_text", "canonical_song_title", "canonical_artist"}
    details: dict[str, list[str]] = {}
    for path in OUTPUT_DIR.glob("*.csv"):
        rows = read_csv(path)
        headers = set(rows[0]) if rows else set()
        found = sorted(headers & prohibited)
        if found:
            public_headers_ok = False
            details[path.name] = found
    checks.append({"name": "public_outputs_keep_lyrics_and_identifiers_private", "passed": public_headers_ok, "detail": details})

    passed = all(bool(check.get("passed")) for check in checks)
    return {
        "artifact_id": ARTIFACT_ID,
        "status": "pass" if passed else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


def main() -> int:
    try:
        result = validate()
    except Exception as exc:
        result = {
            "artifact_id": ARTIFACT_ID,
            "status": "fail",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": [{"name": "validator_runtime", "passed": False, "detail": str(exc)}],
        }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUTPUT_DIR / "validation.json", result)
    atomic_write_json(PRIVATE_DIR / "private_validation.json", result)
    print(f"Validation {result['status']}: {sum(bool(row.get('passed')) for row in result['checks'])}/{len(result['checks'])} checks passed.")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
