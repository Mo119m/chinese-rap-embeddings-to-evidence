#!/usr/bin/env python3
"""Build a conservative, auditable clean-text sidecar for canonical lyrics.

The canonical corpus is immutable.  This script does not modify it.  Instead,
it creates a private derivative that removes only **leading metadata/header
lines** from each canonical lyric chunk under explicit deterministic rules.

Why this exists: a small number of otherwise canonical chunks retain a title,
featured-artist list, production line, or writer-credit block before the first
lyric line.  Those headers are not lyrics and can distort a lyric-level NER or
semantic model.  The sidecar preserves original hashes, cleaning rules,
character counts, and an auditable review queue.  It never removes a matching
string once lyric content has begun.

Run from the repository root:
    work\\semantic-ml-venv\\Scripts\\python.exe work\\build_canonical_lyric_text_sidecar_v1.py
    work\\semantic-ml-venv\\Scripts\\python.exe work\\validate_canonical_lyric_text_sidecar_v1.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ID = "canonical-lyric-text-sidecar-v1"
CLEANING_VERSION = "1.0.0"

ANALYSIS_MANIFEST = ROOT / "outputs" / "chinese-rap-analysis-input-v1" / "analysis_input_manifest.json"
ANALYSIS_POINTER = ROOT / "outputs" / "chinese-rap-analysis-input-v1" / "private_analysis_input_pointer.json"
ANALYSIS_VALIDATION = ROOT / "outputs" / "chinese-rap-analysis-input-v1" / "independent_validation.json"

PRIVATE_DIR = ROOT / "work" / "private-canonical-lyric-text-sidecar-v1"
OUTPUT_DIR = ROOT / "outputs" / "canonical-lyric-text-sidecar-v1"

EXPECTED_SONGS = 7211
EXPECTED_CHUNKS = 22128

PUBLIC_ALLOWLIST = {
    "README.md",
    "method_and_limits.md",
    "data_dictionary.md",
    "manifest.json",
    "validation.json",
    "cleaning_summary.json",
    "header_rule_summary.csv",
    "private_sidecar_pointer.json",
}
PRIVATE_ALLOWLIST = {
    "cleaned_analysis_chunks_v1.csv",
    "header_cleaning_audit_v1.csv",
    "private_manifest.json",
    "private_validation.json",
}


class BuildError(RuntimeError):
    """Stop instead of guessing when a corpus contract cannot be proved."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def atomic_write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    atomic_write_text(path, output.getvalue())


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BuildError(f"Missing {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"Could not read {label}: {path}") from exc
    if not isinstance(data, dict):
        raise BuildError(f"{label} must be a JSON object.")
    return data


def resolve_repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BuildError(f"{label} lacks a controlled relative path.")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise BuildError(f"{label} escapes the repository.") from exc
    return path


def require_directory_allowlist(path: Path, allowed: set[str], label: str) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise BuildError(f"{label} is not a directory: {path}")
    unexpected = sorted(item.name for item in path.iterdir() if item.name not in allowed)
    nested = sorted(item.name for item in path.iterdir() if item.is_dir())
    if unexpected or nested:
        message = ", ".join(unexpected + nested)
        raise BuildError(f"{label} contains undeclared content; refusing to overwrite: {message}")


def ascii_lower(value: str) -> str:
    return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in value)


def normalized_compare(value: str) -> str:
    """Normalize only for title/header comparison; never write this as lyrics."""
    return re.sub(r"[\s\W_]+", "", ascii_lower(value), flags=re.UNICODE)


# Strong cues are intentionally applied only in the initial header region.  A
# lyric line that mentions a producer or says "feat" later in a verse remains
# untouched.
# A content-like word such as ``制作人`` or ``企划`` can occur inside a lyric.
# Therefore production cues must be either an explicit line prefix or appear
# inside a parenthetical title/credit clause; a bare occurrence anywhere in a
# leading line is intentionally insufficient.
_LEADING_PRODUCTION_CUE_RE = re.compile(
    r"^\s*(?:"
    r"prod(?:uced|uction)?\b|producer\b|beat(?:maker)?\s+by\b|"
    r"mix(?:ed|ing)?\s+by\b|master(?:ed|ing)?\s+by\b|record(?:ed|ing)?(?:\s+at|\s+by)?\b|"
    r"arrang(?:ed|ement)?\s+by\b|compos(?:ed|er)\s+by\b|lyricist\b|lyrics?\s+by\b|"
    r"(?:制作人|作词|作曲|词曲|编曲|混音|母带|录音|出品(?:公司)?|发行|监制|企划)\s*[:：\s]|"
    r"词\s*/\s*曲\s*[:：]"
    r")",
    re.IGNORECASE,
)
_PARENTHETICAL_PRODUCTION_CUE_RE = re.compile(
    r"[（(]\s*(?:prod(?:uced|uction)?\b|producer\b|beat(?:maker)?\s+by\b|"
    r"mix(?:ed|ing)?\s+by\b|master(?:ed|ing)?\s+by\b|record(?:ed|ing)?\b|"
    r"(?:制作人|作词|作曲|词曲|编曲|混音|母带|录音|出品(?:公司)?|发行|监制|企划)\s*[:：])",
    re.IGNORECASE,
)
_FEATURE_TITLE_CUE_RE = re.compile(
    r"^(?=.{1,180}$)(?!.*[。！？!?；;]).*(?:[（(]\s*|\s+)(?:feat(?:\.|uring)?|ft\.)\s*",
    re.IGNORECASE,
)
_STRUCTURE_CUE_RE = re.compile(
    r"^\s*[\[\(（【]*\s*(?:verse|hook|chorus|bridge|intro|outro|pre-?chorus|refrain|interlude)"
    r"[\s\d]*[\]\)）】]*\s*[:：]?\s*$",
    re.IGNORECASE,
)
_COPYRIGHT_CUE_RE = re.compile(r"未经.*(?:书面许可|权利人)|版权所有|copyright", re.IGNORECASE)
_NAME_LIST_SEPARATORS_RE = re.compile(r"[/／、&＆|]")
_SENTENCE_PUNCTUATION_RE = re.compile(r"[。！？!?；;]")


def title_core(title: str) -> str:
    """Remove obvious credit trailers before comparison, not from any lyrics."""
    value = str(title).strip()
    # Some imports prepend a generic collective credit such as ``群星-``.
    value = re.sub(r"^[^\-—–]{1,24}[\-—–]", "", value)
    value = re.sub(r"[（(].*?[）)]", "", value)
    value = re.split(r"(?i)\b(?:feat(?:\.|uring)?|ft\.)\b", value, maxsplit=1)[0]
    return normalized_compare(value)


def looks_title_like(line: str, title: str) -> bool:
    line_key = normalized_compare(line)
    title_key = title_core(title)
    if len(line_key) < 4 or len(title_key) < 4:
        return False
    # A narrow containment criterion is used because we only act when a
    # nearby strong header cue independently supports a header interpretation.
    return line_key in title_key or title_key in line_key


def strong_header_rules(line: str) -> list[str]:
    rules: list[str] = []
    if _LEADING_PRODUCTION_CUE_RE.search(line) or _PARENTHETICAL_PRODUCTION_CUE_RE.search(line):
        rules.append("leading_credit_or_production_cue")
    if _FEATURE_TITLE_CUE_RE.search(line):
        rules.append("leading_feature_credit_cue")
    if _STRUCTURE_CUE_RE.match(line):
        rules.append("leading_structure_marker")
    if _COPYRIGHT_CUE_RE.search(line):
        rules.append("leading_copyright_notice")
    return rules


def looks_name_list(line: str) -> bool:
    if len(line) > 180 or _SENTENCE_PUNCTUATION_RE.search(line):
        return False
    separators = len(_NAME_LIST_SEPARATORS_RE.findall(line))
    if separators < 2:
        return False
    # Require name-list-like segments rather than a slash-delimited sentence.
    segments = [segment.strip() for segment in _NAME_LIST_SEPARATORS_RE.split(line) if segment.strip()]
    return len(segments) >= 3 and all(len(segment) <= 40 for segment in segments)


def clean_initial_headers(text: str, song_title: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove only a supported leading header block and return its audit trail."""
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    nonempty_positions = [index for index, line in enumerate(lines) if line.strip()]
    lookahead_positions = nonempty_positions[:4]
    has_nearby_strong_cue = any(bool(strong_header_rules(lines[index].strip())) for index in lookahead_positions)
    removed: list[dict[str, Any]] = []
    saw_header = False
    retained_start = 0

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            if not removed:
                retained_start = index + 1
            continue
        rules = strong_header_rules(line)
        if looks_title_like(line, song_title) and has_nearby_strong_cue:
            rules.append("title_like_line_adjacent_to_header_cue")
        if saw_header and looks_name_list(line):
            rules.append("leading_name_list_after_header_cue")
        if not rules:
            retained_start = index
            break
        removed.append(
            {
                "line_index": index,
                "rule_ids": ";".join(sorted(set(rules))),
                "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                "line_length": len(line),
            }
        )
        saw_header = True
        retained_start = index + 1
    else:
        retained_start = len(lines)

    cleaned = "\n".join(lines[retained_start:]).strip()
    return cleaned, removed


def load_canonical_chunks() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    manifest = read_json_object(ANALYSIS_MANIFEST, "canonical analysis manifest")
    pointer = read_json_object(ANALYSIS_POINTER, "canonical analysis pointer")
    validation = read_json_object(ANALYSIS_VALIDATION, "canonical analysis validation")
    if manifest.get("analysis_input_id") != "chinese-rap-canonical-analysis-input-v1":
        raise BuildError("Unexpected canonical analysis input ID.")
    if validation.get("status") != "pass":
        raise BuildError("Canonical analysis input has not passed independent validation.")
    if pointer.get("analysis_input_id") != manifest.get("analysis_input_id"):
        raise BuildError("Canonical analysis pointer does not match its manifest.")
    if pointer.get("classification") != "private_local_only_full_lyrics":
        raise BuildError("Canonical lyrics are not marked private-local-only.")
    private_dir = resolve_repo_path(pointer.get("private_directory"), "canonical private directory")
    chunk_name = pointer.get("chunk_file")
    song_name = pointer.get("song_file")
    if not isinstance(chunk_name, str) or not isinstance(song_name, str):
        raise BuildError("Canonical pointer lacks input filenames.")
    chunk_path = (private_dir / chunk_name).resolve()
    song_path = (private_dir / song_name).resolve()
    try:
        chunk_path.relative_to(private_dir)
        song_path.relative_to(private_dir)
    except ValueError as exc:
        raise BuildError("Canonical input path escapes private directory.") from exc
    if sha256_file(chunk_path) != pointer.get("chunk_file_sha256") or sha256_file(song_path) != pointer.get("song_file_sha256"):
        raise BuildError("Canonical input hashes do not match pointer values.")
    chunks = pd.read_csv(chunk_path, dtype=str, keep_default_na=False)
    if len(chunks) != EXPECTED_CHUNKS or chunks["song_id"].nunique() != EXPECTED_SONGS:
        raise BuildError("Canonical input population differs from the declared protocol.")
    required = {
        "song_id", "chunk_id", "canonical_song_title", "text", "canonical_lyric_text_sha256",
        "analysis_text_weight", "downstream_eligibility",
    }
    if missing := required - set(chunks.columns):
        raise BuildError(f"Canonical chunks omit required fields: {sorted(missing)}")
    if not (chunks["downstream_eligibility"] == "eligible").all():
        raise BuildError("Non-eligible content entered the canonical analysis input.")
    if chunks[["song_id", "chunk_id"]].duplicated().any():
        raise BuildError("Canonical chunk keys are not unique.")
    return chunks, manifest, {"chunks": chunk_path, "songs": song_path}


def write_docs(summary: dict[str, Any]) -> None:
    readme = f"""# Canonical lyric text sidecar v1

This private derivative improves the **analysis text**, not the canonical
corpus.  It removes only header/metadata lines that occur before the first
retained lyric line.  The immutable canonical registry, song titles, source
labels, full input text, IDs, and hashes remain the provenance authority.

The current run preserved all {summary['counts']['input_chunks']:,} canonical
chunks and created a clean analysis text for {summary['counts']['eligible_clean_text_chunks']:,} chunks.
{summary['counts']['changed_chunks']:,} chunks had one or more leading header
lines removed under explicit rules.

## Reproduce

```powershell
work\\semantic-ml-venv\\Scripts\\python.exe work\\build_canonical_lyric_text_sidecar_v1.py
work\\semantic-ml-venv\\Scripts\\python.exe work\\validate_canonical_lyric_text_sidecar_v1.py
```

The cleaned full text is private local data.  Generic outputs include only
counts, rule summaries, hashes, and pointers.
"""
    methods = """# Method and limits

## What is removed

The cleaner operates only at the **beginning of a lyric chunk**.  It may remove:

1. a leading title/featured-credit line when it is adjacent to a separate
   production, feature, or writer-credit cue;
2. leading production / writer / mixing / recording / release credit lines;
3. a leading standalone structure marker or copyright notice; and
4. a slash-separated name list only after a strong header line has already
   been identified.

Every removed line receives a SHA-256 digest, line index, and rule ID in the
private audit sidecar.  A generic reader sees no lyric text.

## What is never removed automatically

- Metadata-looking strings that appear after the first retained lyric line.
- A title-like first line without an independent nearby header cue.
- A single short artist tag or a line that merely contains a slash without a
  preceding header signal.
- Any whole canonical record.  A chunk that becomes empty is held out with an
  explicit status rather than silently deleted.

## Claim boundary

This is a content-hygiene step.  It does not verify real-world artist identity,
biography, performance language, Flow, or any musical property.  Its purpose
is to keep non-lyric headers from contaminating text-level NER, semantic
embedding, and keyword analysis.
"""
    dictionary = """# Data dictionary

## Private local files

- `cleaned_analysis_chunks_v1.csv`: one row per canonical chunk with its
  original text hash, cleaned analysis text, cleaned-text hash, weighted
  duplicate control, and deterministic transformation metadata.  This file
  contains full lyrics and must not be shared externally.
- `header_cleaning_audit_v1.csv`: one row per changed chunk, with removed-line
  hashes, rule IDs, character/line counts, and a review priority.  It contains
  no full lyric text.

## Generic files

- `cleaning_summary.json`: population and transformation totals.
- `header_rule_summary.csv`: how often each rule ran.  It contains no song,
  title, source label, lyric text, or line excerpt.
- `private_sidecar_pointer.json`: controlled local file names and hashes for
  reproducibility.  The listed files are private local-only.
"""
    atomic_write_text(OUTPUT_DIR / "README.md", readme)
    atomic_write_text(OUTPUT_DIR / "method_and_limits.md", methods)
    atomic_write_text(OUTPUT_DIR / "data_dictionary.md", dictionary)


def build() -> dict[str, Any]:
    require_directory_allowlist(OUTPUT_DIR, PUBLIC_ALLOWLIST, "public sidecar output")
    require_directory_allowlist(PRIVATE_DIR, PRIVATE_ALLOWLIST, "private sidecar output")
    chunks, input_manifest, input_paths = load_canonical_chunks()

    cleaned_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    rule_counts: Counter[str] = Counter()
    total_removed_lines = 0
    metadata_only_chunks = 0

    for row in chunks.itertuples(index=False):
        cleaned_text, removed = clean_initial_headers(str(row.text), str(row.canonical_song_title))
        status = "eligible_clean_text" if cleaned_text else "withheld_metadata_only"
        if not cleaned_text:
            metadata_only_chunks += 1
        analysis_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest() if cleaned_text else ""
        rule_ids = sorted({rule for item in removed for rule in item["rule_ids"].split(";") if rule})
        for rule_id in rule_ids:
            rule_counts[rule_id] += sum(rule_id in item["rule_ids"].split(";") for item in removed)
        total_removed_lines += len(removed)
        cleaned_rows.append(
            {
                "analysis_text_sidecar_id": ARTIFACT_ID,
                "cleaning_version": CLEANING_VERSION,
                "song_id": str(row.song_id),
                "chunk_id": str(row.chunk_id),
                "canonical_lyric_text_sha256": str(row.canonical_lyric_text_sha256),
                "analysis_text": cleaned_text,
                "analysis_text_sha256": analysis_hash,
                "original_character_count": len(str(row.text)),
                "analysis_character_count": len(cleaned_text),
                "leading_header_lines_removed": len(removed),
                "header_rule_ids": ";".join(rule_ids),
                "analysis_text_status": status,
                "analysis_text_weight": str(row.analysis_text_weight),
            }
        )
        if removed:
            review_priority = "review" if "leading_name_list_after_header_cue" in rule_ids else "spot_check"
            audit_rows.append(
                {
                    "song_id": str(row.song_id),
                    "chunk_id": str(row.chunk_id),
                    "canonical_lyric_text_sha256": str(row.canonical_lyric_text_sha256),
                    "analysis_text_sha256": analysis_hash,
                    "leading_header_lines_removed": len(removed),
                    "header_rule_ids": ";".join(rule_ids),
                    "removed_line_indices": ";".join(str(item["line_index"]) for item in removed),
                    "removed_line_sha256s": ";".join(str(item["line_sha256"]) for item in removed),
                    "removed_line_lengths": ";".join(str(item["line_length"]) for item in removed),
                    "review_priority": review_priority,
                }
            )

    cleaned_frame = pd.DataFrame(cleaned_rows)
    if len(cleaned_frame) != EXPECTED_CHUNKS or cleaned_frame[["song_id", "chunk_id"]].duplicated().any():
        raise BuildError("The text sidecar did not preserve every canonical chunk exactly once.")
    eligible_clean_count = int((cleaned_frame["analysis_text_status"] == "eligible_clean_text").sum())
    if eligible_clean_count == 0:
        raise BuildError("All canonical chunks were withheld; the header rules are invalid.")

    summary = {
        "artifact_id": ARTIFACT_ID,
        "cleaning_version": CLEANING_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_input": {
            "analysis_input_id": input_manifest["analysis_input_id"],
            "canonical_corpus_id": input_manifest["canonical_source_contract"]["canonical_corpus_id"],
            "canonical_contract_sha256": input_manifest["canonical_source_contract"]["sha256"],
            "private_chunk_input_sha256": sha256_file(input_paths["chunks"]),
            "private_song_input_sha256": sha256_file(input_paths["songs"]),
        },
        "counts": {
            "input_songs": EXPECTED_SONGS,
            "input_chunks": EXPECTED_CHUNKS,
            "changed_chunks": int(len(audit_rows)),
            "unchanged_chunks": int(EXPECTED_CHUNKS - len(audit_rows)),
            "eligible_clean_text_chunks": eligible_clean_count,
            "withheld_metadata_only_chunks": metadata_only_chunks,
            "leading_header_lines_removed": int(total_removed_lines),
            "review_queue_chunks": int(sum(row["review_priority"] == "review" for row in audit_rows)),
        },
        "rules": {
            "scope": "leading header block only; stop at first retained lyric line",
            "auto_removed": [
                "title-like line adjacent to a strong header cue",
                "production, writer, mixing, recording, release, or feature credit cue",
                "standalone structure marker or copyright notice",
                "name list after a strong header cue",
            ],
            "never_auto_removed": [
                "metadata-looking strings after lyric content begins",
                "title-like line without independent header evidence",
                "single short tag",
                "whole canonical record",
            ],
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    write_docs(summary)
    atomic_write_csv(
        PRIVATE_DIR / "cleaned_analysis_chunks_v1.csv",
        [
            "analysis_text_sidecar_id", "cleaning_version", "song_id", "chunk_id", "canonical_lyric_text_sha256",
            "analysis_text", "analysis_text_sha256", "original_character_count", "analysis_character_count",
            "leading_header_lines_removed", "header_rule_ids", "analysis_text_status", "analysis_text_weight",
        ],
        cleaned_rows,
    )
    atomic_write_csv(
        PRIVATE_DIR / "header_cleaning_audit_v1.csv",
        [
            "song_id", "chunk_id", "canonical_lyric_text_sha256", "analysis_text_sha256", "leading_header_lines_removed",
            "header_rule_ids", "removed_line_indices", "removed_line_sha256s", "removed_line_lengths", "review_priority",
        ],
        audit_rows,
    )
    rule_rows = [
        {"rule_id": rule_id, "removed_line_count": count, "scope": "leading header only"}
        for rule_id, count in sorted(rule_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    atomic_write_csv(OUTPUT_DIR / "header_rule_summary.csv", ["rule_id", "removed_line_count", "scope"], rule_rows)
    atomic_write_json(OUTPUT_DIR / "cleaning_summary.json", summary)
    private_files = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in PRIVATE_DIR.iterdir()
        if path.is_file() and path.name not in {"private_manifest.json", "private_validation.json"}
    }
    atomic_write_json(
        PRIVATE_DIR / "private_manifest.json",
        {
            "artifact_id": ARTIFACT_ID,
            "cleaning_version": CLEANING_VERSION,
            "classification": "private_local_only_cleaned_full_lyric_text",
            "files": private_files,
        },
    )
    pointer = {
        "artifact_id": ARTIFACT_ID,
        "cleaning_version": CLEANING_VERSION,
        "classification": "private_local_only_cleaned_full_lyric_text",
        "handling": "Do not package or share externally. Use only through exact song_id/chunk_id joins.",
        "private_directory": str(PRIVATE_DIR.relative_to(ROOT)),
        "cleaned_chunk_file": "cleaned_analysis_chunks_v1.csv",
        "cleaned_chunk_file_sha256": sha256_file(PRIVATE_DIR / "cleaned_analysis_chunks_v1.csv"),
        "audit_file": "header_cleaning_audit_v1.csv",
        "audit_file_sha256": sha256_file(PRIVATE_DIR / "header_cleaning_audit_v1.csv"),
    }
    atomic_write_json(OUTPUT_DIR / "private_sidecar_pointer.json", pointer)
    public_files = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "validation.json"}
    }
    atomic_write_json(
        OUTPUT_DIR / "manifest.json",
        {
            "artifact_id": ARTIFACT_ID,
            "cleaning_version": CLEANING_VERSION,
            "canonical_input": summary["canonical_input"],
            "counts": summary["counts"],
            "privacy": {
                "generic_output_contains_full_lyrics": False,
                "generic_output_contains_song_ids": False,
                "generic_output_contains_titles_or_source_labels": False,
            },
            "output_files": public_files,
        },
    )
    pending_validation = {
        "artifact_id": ARTIFACT_ID,
        "status": "pending_independent_validation",
        "required_validator": "work/validate_canonical_lyric_text_sidecar_v1.py",
    }
    atomic_write_json(OUTPUT_DIR / "validation.json", pending_validation)
    atomic_write_json(PRIVATE_DIR / "private_validation.json", pending_validation)
    return summary


def main() -> int:
    try:
        summary = build()
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 2
    print(
        f"Built {ARTIFACT_ID}: {summary['counts']['changed_chunks']} changed chunks; "
        f"{summary['counts']['leading_header_lines_removed']} leading header lines removed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
