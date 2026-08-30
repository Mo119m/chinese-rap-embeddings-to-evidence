#!/usr/bin/env python3
"""Create an auditable, non-destructive metadata-quality sidecar for songs.

The canonical artist and title strings remain immutable.  This sidecar supplies
derived display/search forms and conservative title-quality flags so downstream
tools do not mistake credit headers, generic placeholders, or embedded credits
for meaningful song-title evidence.  Lyric analysis is never excluded here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ID = "canonical-song-metadata-quality-v1"
VERSION = "1.0.0"
SOURCE_DIR = ROOT / "outputs" / "chinese-rap-analysis-input-v1"
SOURCE_MANIFEST = SOURCE_DIR / "analysis_input_manifest.json"
SOURCE_POINTER = SOURCE_DIR / "private_analysis_input_pointer.json"
SOURCE_VALIDATION = SOURCE_DIR / "validation.json"
PRIVATE_DIR = ROOT / "work" / "private-canonical-song-metadata-quality-v1"
OUTPUT_DIR = ROOT / "outputs" / ARTIFACT_ID
PUBLIC_ALLOWLIST = {"README.md", "method_and_limits.md", "flag_summary.csv", "manifest.json", "validation.json"}
PRIVATE_ALLOWLIST = {"canonical_song_metadata_quality_v1.csv", "private_manifest.json", "private_validation.json"}
SOURCE_COLUMNS = (
    "analysis_input_id",
    "analysis_input_schema_version",
    "canonical_corpus_id",
    "canonical_corpus_contract_version",
    "song_id",
    "canonical_artist",
    "canonical_song_title",
    "identity_validation_status",
    "downstream_eligibility",
    "downstream_usage_status",
    "downstream_reason_codes",
    "analysis_deduplication_required",
    "artist_title_comparison_eligible",
    "normalized_artist_title_collision_group",
    "identical_song_lyric_content_group",
    "song_lyric_content_sha256",
    "canonical_chunk_count",
    "sum_analysis_text_weight",
)
SIDECAR_COLUMNS = (
    "song_id",
    "canonical_artist",
    "canonical_song_title",
    "artist_display",
    "title_display",
    "artist_search_key",
    "title_search_key",
    "source_artist_title_comparison_eligible",
    "title_quality_flags",
    "title_metadata_only_candidate",
    "generic_title_placeholder",
    "embedded_credit_or_feature_cue",
    "title_contains_nbsp",
    "nonstandard_title_whitespace",
    "title_semantic_eligible",
    "metadata_review_required",
    "normalization_rule_ids",
    "source_metadata_hash",
)
SUMMARY_COLUMNS = ("metric", "count", "denominator", "interpretation")

# These rules flag, never rewrite.  They were deliberately limited to forms
# that are structurally a credit field rather than a usable title label.
TITLE_METADATA_HEADER_RE = re.compile(r"^\s*(?:演唱|演唱者|歌手|合作歌手|特邀演唱|作曲)\s*[：:]")
EMBEDDED_CREDIT_RE = re.compile(r"(?i)(?:\bprod(?:ucer)?\.?\b|\bfeat(?:uring)?\.?\b|\bft\.?\b|作曲)")
TITLE_FLAG_ORDER = (
    "title_metadata_only_candidate",
    "generic_title_placeholder",
    "embedded_credit_or_feature_cue",
    "title_contains_nbsp",
    "nonstandard_title_whitespace",
)
NORMALIZATION_RULE_IDS = "nfc;unicode_space_to_ascii;collapse_internal_whitespace;trim"


class MetadataError(RuntimeError):
    """Raised when an immutable source or derived sidecar violates its contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    atomic_write_text(path, buffer.getvalue())


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise MetadataError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataError(f"Invalid JSON in {label}.") from exc
    if not isinstance(payload, dict):
        raise MetadataError(f"{label} must be a JSON object.")
    return payload


def read_csv_exact(path: Path, columns: tuple[str, ...], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise MetadataError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != columns:
            raise MetadataError(f"{label} schema mismatch; expected={list(columns)}, actual={list(actual)}")
        return list(reader)


def resolve_repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MetadataError(f"{label} lacks a controlled repository path.")
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise MetadataError(f"{label} escapes the workspace.") from exc
    return candidate


def require_exact_allowlist(directory: Path, allowed: set[str], label: str, permit_missing_validation: bool = False) -> None:
    if not directory.exists():
        return
    if not directory.is_dir():
        raise MetadataError(f"{label} is not a directory.")
    actual = {item.name for item in directory.iterdir()}
    accepted = {frozenset(allowed)}
    if permit_missing_validation:
        accepted.add(frozenset(name for name in allowed if not name.endswith("validation.json")))
    if frozenset(actual) not in accepted or any(item.is_dir() for item in directory.iterdir()):
        raise MetadataError(f"{label} contains undeclared files or directories.")


def normalise_display(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def source_metadata_hash(artist: str, title: str) -> str:
    return sha256_text(f"canonical-song-metadata-v1\t{artist}\t{title}")


def as_bool(value: str, label: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise MetadataError(f"{label} must be the literal string true or false.")


def derive_row(source: dict[str, str]) -> dict[str, str]:
    artist, title = source["canonical_artist"], source["canonical_song_title"]
    artist_display, title_display = normalise_display(artist), normalise_display(title)
    title_metadata_only = bool(TITLE_METADATA_HEADER_RE.search(title))
    generic_placeholder = title.casefold() == "untitled"
    embedded_credit = bool(EMBEDDED_CREDIT_RE.search(title))
    contains_nbsp = "\u00a0" in title
    nonstandard_whitespace = title != " ".join(title.split())
    flags = {
        "title_metadata_only_candidate": title_metadata_only,
        "generic_title_placeholder": generic_placeholder,
        "embedded_credit_or_feature_cue": embedded_credit,
        "title_contains_nbsp": contains_nbsp,
        "nonstandard_title_whitespace": nonstandard_whitespace,
    }
    return {
        "song_id": source["song_id"],
        "canonical_artist": artist,
        "canonical_song_title": title,
        "artist_display": artist_display,
        "title_display": title_display,
        "artist_search_key": artist_display.casefold(),
        "title_search_key": title_display.casefold(),
        "source_artist_title_comparison_eligible": source["artist_title_comparison_eligible"],
        "title_quality_flags": ";".join(name for name in TITLE_FLAG_ORDER if flags[name]),
        **{name: "true" if value else "false" for name, value in flags.items()},
        "title_semantic_eligible": "false" if title_metadata_only or generic_placeholder else "true",
        "metadata_review_required": "true" if title_metadata_only or generic_placeholder else "false",
        "normalization_rule_ids": NORMALIZATION_RULE_IDS,
        "source_metadata_hash": source_metadata_hash(artist, title),
    }


def load_source() -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest = read_json(SOURCE_MANIFEST, "canonical analysis-input manifest")
    pointer = read_json(SOURCE_POINTER, "canonical analysis-input pointer")
    validation = read_json(SOURCE_VALIDATION, "canonical analysis-input validation")
    if manifest.get("analysis_input_id") != "chinese-rap-canonical-analysis-input-v1" or validation.get("status") != "pass":
        raise MetadataError("The canonical analysis input is not the expected passing artifact.")
    private_dir = resolve_repo_path(pointer.get("private_directory"), "canonical analysis-input private directory")
    filename = pointer.get("song_file")
    if not isinstance(filename, str):
        raise MetadataError("The canonical analysis-input pointer lacks its song filename.")
    song_path = (private_dir / filename).resolve()
    try:
        song_path.relative_to(private_dir)
    except ValueError as exc:
        raise MetadataError("The canonical song filename escapes its controlled private directory.") from exc
    expected_hash = manifest.get("output_private_artifacts", {}).get(filename, {}).get("sha256")
    if not song_path.is_file() or sha256_file(song_path) != pointer.get("song_file_sha256") or sha256_file(song_path) != expected_hash:
        raise MetadataError("The canonical song metadata file does not match its source contract.")
    rows = read_csv_exact(song_path, SOURCE_COLUMNS, "canonical analysis songs")
    expected_count = manifest.get("counts", {}).get("included_song_rows")
    if expected_count != 7211 or len(rows) != expected_count:
        raise MetadataError("The canonical song population does not reconcile to the source manifest.")
    identifiers: set[str] = set()
    for row in rows:
        if row["song_id"] in identifiers or not row["canonical_artist"].strip() or not row["canonical_song_title"].strip():
            raise MetadataError("The canonical songs contain a duplicate ID or blank artist/title.")
        if row["downstream_eligibility"] != "eligible":
            raise MetadataError("The canonical analysis song sidecar may only contain eligible records.")
        as_bool(row["artist_title_comparison_eligible"], "source artist/title comparison eligibility")
        identifiers.add(row["song_id"])
    return rows, {
        "manifest": manifest,
        "manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "pointer_sha256": sha256_file(SOURCE_POINTER),
        "song_file_sha256": sha256_file(song_path),
    }


def summary_rows(sidecar: list[dict[str, str]]) -> list[dict[str, Any]]:
    denominator = len(sidecar)
    metrics = [
        ("songs", denominator, "all canonical analysis songs"),
        ("title_metadata_only_candidate", sum(row["title_metadata_only_candidate"] == "true" for row in sidecar), "flagged for metadata-like title header; lyric analysis retained"),
        ("generic_title_placeholder", sum(row["generic_title_placeholder"] == "true" for row in sidecar), "flagged generic title; lyric analysis retained"),
        ("embedded_credit_or_feature_cue", sum(row["embedded_credit_or_feature_cue"] == "true" for row in sidecar), "flagged cue; canonical title remains unchanged"),
        ("title_contains_nbsp", sum(row["title_contains_nbsp"] == "true" for row in sidecar), "derived display/search form normalizes spacing only"),
        ("nonstandard_title_whitespace", sum(row["nonstandard_title_whitespace"] == "true" for row in sidecar), "derived display/search form normalizes spacing only"),
        ("title_semantic_ineligible", sum(row["title_semantic_eligible"] == "false" for row in sidecar), "excluded only from title-semantic/UI uses until reviewed"),
    ]
    return [{"metric": metric, "count": count, "denominator": denominator, "interpretation": interpretation} for metric, count, interpretation in metrics]


def documents() -> None:
    readme = """# Canonical Song Metadata Quality v1

This sidecar protects downstream analyses from treating a malformed or
credit-heavy title field as song meaning. It does **not** edit the canonical
artist/title labels or remove any lyric from corpus analysis.

Use `title_display` and `title_search_key` for presentation/search only. Use
`title_semantic_eligible` to exclude credit-header and `UNTITLED` placeholders
from title-based comparison, recommendation, or topic interpretation until an
evidence-backed metadata review is available.
"""
    limits = """# Rules and limits

The sidecar applies deterministic Unicode display normalization (NFC, Unicode
spaces to an ordinary space, internal whitespace collapsed, trim) without
changing immutable canonical labels.

- A `title_metadata_only_candidate` starts with one of the documented credit
  headers: 演唱, 演唱者, 歌手, 合作歌手, 特邀演唱, or 作曲, followed by a colon.
- A `generic_title_placeholder` is a case-insensitive `UNTITLED` label.
- An `embedded_credit_or_feature_cue` detects `prod`, `producer`, `feat`, `ft`,
  or 作曲 anywhere in the title. It is a caution flag, not proof that the title
  is wrong.

No rule guesses a corrected title, parses collaborators, or turns a title into
artist/relationship evidence. All affected songs remain in lyric-level and
corpus-level analysis.
"""
    atomic_write_text(OUTPUT_DIR / "README.md", readme)
    atomic_write_text(OUTPUT_DIR / "method_and_limits.md", limits)


def payload_hashes(directory: Path, exclude: set[str]) -> dict[str, dict[str, Any]]:
    return {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(directory.iterdir()) if path.is_file() and path.name not in exclude}


def validate_persisted(permit_missing_validation: bool = False, require_prior_validation: bool = True) -> dict[str, Any]:
    require_exact_allowlist(OUTPUT_DIR, PUBLIC_ALLOWLIST, "public metadata-quality output", permit_missing_validation)
    require_exact_allowlist(PRIVATE_DIR, PRIVATE_ALLOWLIST, "private metadata-quality output", permit_missing_validation)
    source_rows, source_info = load_source()
    expected = [derive_row(row) for row in source_rows]
    actual = read_csv_exact(PRIVATE_DIR / "canonical_song_metadata_quality_v1.csv", SIDECAR_COLUMNS, "metadata-quality sidecar")
    summaries = summary_rows(expected)
    actual_summary = read_csv_exact(OUTPUT_DIR / "flag_summary.csv", SUMMARY_COLUMNS, "metadata flag summary")
    manifest = read_json(OUTPUT_DIR / "manifest.json", "metadata-quality manifest")
    private_manifest = read_json(PRIVATE_DIR / "private_manifest.json", "private metadata-quality manifest")
    expected_counts = {
        "songs": len(expected),
        "title_metadata_only_candidate": 97,
        "generic_title_placeholder": 5,
        "embedded_credit_or_feature_cue": 607,
        "title_contains_nbsp": 194,
        "nonstandard_title_whitespace": 195,
        "title_semantic_ineligible": 102,
    }
    config_ok = (
        manifest.get("artifact_id") == ARTIFACT_ID
        and manifest.get("version") == VERSION
        and manifest.get("source") == {
            "analysis_input_id": source_info["manifest"]["analysis_input_id"],
            "analysis_input_manifest_sha256": source_info["manifest_sha256"],
            "analysis_input_pointer_sha256": source_info["pointer_sha256"],
            "canonical_song_file_sha256": source_info["song_file_sha256"],
        }
        and manifest.get("counts") == expected_counts
        and manifest.get("claim_boundary") == "derived presentation/search quality sidecar only; no canonical label rewrite, external correction, title-theme inference, or lyric exclusion"
        and private_manifest.get("artifact_id") == ARTIFACT_ID
        and private_manifest.get("version") == VERSION
        and private_manifest.get("classification") == "private_local_only_song_metadata_sidecar_without_lyrics"
    )
    prior_ok = True
    if require_prior_validation:
        previous_public = read_json(OUTPUT_DIR / "validation.json", "prior metadata-quality validation")
        previous_private = read_json(PRIVATE_DIR / "private_validation.json", "prior private metadata-quality validation")
        prior_ok = previous_public.get("artifact_id") == ARTIFACT_ID and previous_public.get("version") == VERSION and previous_public.get("status") == "pass" and previous_private == previous_public
    checks = [
        {"name": "public_inventory_exact", "passed": True},
        {"name": "private_inventory_exact", "passed": True},
        {"name": "source_provenance_current", "passed": source_info["manifest_sha256"] == manifest.get("source", {}).get("analysis_input_manifest_sha256") and source_info["song_file_sha256"] == manifest.get("source", {}).get("canonical_song_file_sha256")},
        {"name": "sidecar_rejoins_and_recomputes_exactly", "passed": actual == expected},
        {"name": "summary_recomputes_exactly", "passed": actual_summary == [{key: str(value) for key, value in row.items()} for row in summaries]},
        {"name": "expected_conservative_title_flag_counts", "passed": all(int(next(row["count"] for row in summaries if row["metric"] == metric)) == count for metric, count in expected_counts.items())},
        {"name": "manifest_identity_configuration_and_counts_current", "passed": config_ok},
        {"name": "public_payload_hashes_match", "passed": manifest.get("output_files") == payload_hashes(OUTPUT_DIR, {"manifest.json", "validation.json"})},
        {"name": "private_payload_hashes_match", "passed": private_manifest.get("files") == payload_hashes(PRIVATE_DIR, {"private_manifest.json", "private_validation.json"})},
        {"name": "prior_validation_current_and_passing", "passed": prior_ok},
    ]
    passed = all(bool(check["passed"]) for check in checks)
    return {"artifact_id": ARTIFACT_ID, "version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pass" if passed else "fail", "checks": checks}


def build() -> dict[str, Any]:
    require_exact_allowlist(OUTPUT_DIR, PUBLIC_ALLOWLIST, "public metadata-quality output", permit_missing_validation=True)
    require_exact_allowlist(PRIVATE_DIR, PRIVATE_ALLOWLIST, "private metadata-quality output", permit_missing_validation=True)
    source_rows, source_info = load_source()
    sidecar = [derive_row(row) for row in source_rows]
    summaries = summary_rows(sidecar)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    documents()
    atomic_write_csv(PRIVATE_DIR / "canonical_song_metadata_quality_v1.csv", SIDECAR_COLUMNS, sidecar)
    atomic_write_csv(OUTPUT_DIR / "flag_summary.csv", SUMMARY_COLUMNS, summaries)
    counts = {
        "songs": len(sidecar),
        "title_metadata_only_candidate": sum(row["title_metadata_only_candidate"] == "true" for row in sidecar),
        "generic_title_placeholder": sum(row["generic_title_placeholder"] == "true" for row in sidecar),
        "embedded_credit_or_feature_cue": sum(row["embedded_credit_or_feature_cue"] == "true" for row in sidecar),
        "title_contains_nbsp": sum(row["title_contains_nbsp"] == "true" for row in sidecar),
        "nonstandard_title_whitespace": sum(row["nonstandard_title_whitespace"] == "true" for row in sidecar),
        "title_semantic_ineligible": sum(row["title_semantic_eligible"] == "false" for row in sidecar),
    }
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "source": {
            "analysis_input_id": source_info["manifest"]["analysis_input_id"],
            "analysis_input_manifest_sha256": source_info["manifest_sha256"],
            "analysis_input_pointer_sha256": source_info["pointer_sha256"],
            "canonical_song_file_sha256": source_info["song_file_sha256"],
        },
        "counts": counts,
        "normalization": NORMALIZATION_RULE_IDS,
        "claim_boundary": "derived presentation/search quality sidecar only; no canonical label rewrite, external correction, title-theme inference, or lyric exclusion",
        "privacy": "public files contain aggregate metadata-quality counts and method only; song-level labels remain private local-only",
        "output_files": payload_hashes(OUTPUT_DIR, {"manifest.json", "validation.json"}),
    }
    private_manifest = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "classification": "private_local_only_song_metadata_sidecar_without_lyrics",
        "files": payload_hashes(PRIVATE_DIR, {"private_manifest.json", "private_validation.json"}),
    }
    atomic_write_json(OUTPUT_DIR / "manifest.json", manifest)
    atomic_write_json(PRIVATE_DIR / "private_manifest.json", private_manifest)
    validation = validate_persisted(permit_missing_validation=True, require_prior_validation=False)
    atomic_write_json(OUTPUT_DIR / "validation.json", validation)
    atomic_write_json(PRIVATE_DIR / "private_validation.json", validation)
    return validation


def main() -> int:
    try:
        validation = build()
    except MetadataError as exc:
        print(f"BUILD FAILED: {exc}")
        return 2
    print(f"Built {ARTIFACT_ID}: {validation['status']} ({sum(bool(item['passed']) for item in validation['checks'])}/{len(validation['checks'])} checks).")
    return 0 if validation["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
