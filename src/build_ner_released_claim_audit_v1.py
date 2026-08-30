#!/usr/bin/env python3
"""Build a private, claim-complete human audit package for released NER claims.

The public NER artifact remains provisional.  This builder selects every
strict, shared-text-excluded occurrence that contributes to a currently
released source-label/entity edge or entity co-mention.  It writes lyric
contexts and source locators only to a validated private ``work/`` directory.
The repository receives one aggregate status JSON with no lyric context.

The package is deliberately claim-conditioned.  It can verify the occurrences
behind the released claims, but it cannot estimate corpus-wide NER recall,
precision, or F1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent.parent if REPO_ROOT.parent.name.lower() == "work" else REPO_ROOT
WORK_ROOT = WORKSPACE_ROOT / "work"

SOURCE_PRIVATE_DIR = WORK_ROOT / "private-chinese-rap-ner-cultural-graph-v1"
DEFAULT_PRIVATE_CANDIDATES = SOURCE_PRIVATE_DIR / "all_candidate_occurrences_private.csv"
DEFAULT_PRIVATE_SOURCE_MANIFEST = SOURCE_PRIVATE_DIR / "private_manifest.json"
DEFAULT_PUBLIC_LINKS = REPO_ROOT / "results/ner-v1/source_label_entity_links_provisional.csv"
DEFAULT_PUBLIC_CO_MENTIONS = REPO_ROOT / "results/ner-v1/entity_co_mentions_provisional.csv"
DEFAULT_PUBLIC_NER_MANIFEST = REPO_ROOT / "results/ner-v1/manifest.json"
DEFAULT_PRIVATE_OUTPUT = WORK_ROOT / "private-chinese-rap-ner-released-claim-audit-v1"
DEFAULT_PUBLIC_STATUS = REPO_ROOT / "results/ner-v1/released_claim_audit_status.json"
DEFAULT_PROTOCOL = REPO_ROOT / "methods/NER_RELEASED_CLAIM_AUDIT_PROTOCOL.md"

ARTIFACT_ID = "chinese-rap-ner-released-claim-audit-v1"
PRIVATE_ARTIFACT_ID = "private-chinese-rap-ner-released-claim-audit-v1"
VERSION = "1.0.0"
SEED = "chinese-rap-ner-released-claim-audit-v1-20260827"

DEFAULT_REAL_COMPARATOR_CONTROLS = 20
DEFAULT_SYNTHETIC_BOUNDARY_CONTROLS = 16
DEFAULT_REAL_PAIR_COMPARATOR_CONTROLS = 8

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
REVIEW_DECISION_FIELDS = (
    "mention_valid",
    "boundary_valid",
    "referential_status",
    "entity_type_decision",
    "normalized_surface",
    "confidence_1_to_5",
    "exclusion_reason",
    "notes",
    "reviewed_at_utc",
)
PAIR_REVIEW_DECISION_FIELDS = (
    "entity_a_has_valid_reference",
    "entity_b_has_valid_reference",
    "pair_semantically_supported",
    "confidence_1_to_5",
    "notes",
    "reviewed_at_utc",
)

ADMIN_ONLY_FIELDS = {
    "admin_task_id",
    "task_class",
    "control_subtype",
    "expected_control_outcome",
    "source_candidate_id",
    "candidate_source",
    "claim_ids",
    "claim_roles",
    "source_credit_label",
    "song_id",
    "chunk_id",
    "song_lyric_content_sha256",
    "analysis_text_sha256",
    "machine_agreement_state",
    "machine_transformer_confidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-candidates", type=Path, default=DEFAULT_PRIVATE_CANDIDATES)
    parser.add_argument("--private-source-manifest", type=Path, default=DEFAULT_PRIVATE_SOURCE_MANIFEST)
    parser.add_argument("--public-links", type=Path, default=DEFAULT_PUBLIC_LINKS)
    parser.add_argument("--public-co-mentions", type=Path, default=DEFAULT_PUBLIC_CO_MENTIONS)
    parser.add_argument("--public-ner-manifest", type=Path, default=DEFAULT_PUBLIC_NER_MANIFEST)
    parser.add_argument("--private-output", type=Path, default=DEFAULT_PRIVATE_OUTPUT)
    parser.add_argument("--public-status", type=Path, default=DEFAULT_PUBLIC_STATUS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--real-comparator-controls", type=int, default=DEFAULT_REAL_COMPARATOR_CONTROLS)
    parser.add_argument("--synthetic-boundary-controls", type=int, default=DEFAULT_SYNTHETIC_BOUNDARY_CONTROLS)
    parser.add_argument(
        "--real-pair-comparator-controls", type=int, default=DEFAULT_REAL_PAIR_COMPARATOR_CONTROLS
    )
    parser.add_argument(
        "--replace-empty",
        action="store_true",
        help="Replace an existing package only when every review/adjudication decision is still blank.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(*parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    return f"{prefix}-{stable_digest(*parts)[:length].upper()}"


def deterministic_rank(*parts: Any) -> str:
    return stable_digest(SEED, *parts)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = []
        for row in reader:
            rows.append({str(key): "" if value is None else str(value) for key, value in row.items()})
    return rows


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def require_columns(rows: Sequence[dict[str, str]], columns: Iterable[str], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    missing = sorted(set(columns) - set(rows[0]))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def int_value(row: dict[str, str], field: str) -> int:
    try:
        return int(float(row[field]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer in {field}: {row.get(field)!r}") from exc


def assert_manifest_entry(path: Path, manifest: dict[str, Any], entry_name: str) -> None:
    files = manifest.get("files", {})
    if entry_name not in files:
        raise ValueError(f"Manifest does not inventory {entry_name}: {path}")
    expected = files[entry_name]
    actual_bytes = path.stat().st_size
    actual_hash = sha256_file(path)
    if int(expected["bytes"]) != actual_bytes or str(expected["sha256"]) != actual_hash:
        raise ValueError(
            f"Manifest mismatch for {entry_name}: expected bytes/hash "
            f"{expected['bytes']}/{expected['sha256']}, got {actual_bytes}/{actual_hash}"
        )


def review_files_have_decisions(directory: Path) -> list[str]:
    completed: list[str] = []
    if not directory.exists():
        return completed
    for path in sorted(directory.glob("*.csv")):
        if "reviewer_" not in path.name and "adjudication" not in path.name:
            continue
        rows = read_csv_rows(path)
        decision_fields = set(REVIEW_DECISION_FIELDS) | set(PAIR_REVIEW_DECISION_FIELDS) | {
            "adjudicated_mention_valid",
            "adjudicated_boundary_valid",
            "adjudicated_referential_status",
            "adjudicated_entity_type",
            "adjudicated_pair_supported",
            "adjudicator_id",
            "adjudicated_at_utc",
        }
        if rows:
            decision_fields.update(
                field
                for field in rows[0]
                if field.startswith(("r1_", "r2_", "adjudicated_"))
                or field in {"agreement_state", "adjudicator_id", "adjudication_notes"}
            )
        if any(str(row.get(field, "")).strip() for row in rows for field in decision_fields):
            completed.append(path.name)
    return completed


def prepare_private_output(path: Path, replace_empty: bool) -> None:
    resolved = path.resolve()
    allowed_parent = WORK_ROOT.resolve()
    if resolved.parent != allowed_parent or resolved.name != PRIVATE_ARTIFACT_ID:
        raise RuntimeError(f"Refusing to write private audit outside the exact local work target: {resolved}")
    if REPO_ROOT.resolve() == resolved or REPO_ROOT.resolve() in resolved.parents:
        raise RuntimeError("Private lyric context output must not be placed inside the repository")
    if resolved.exists():
        if not replace_empty:
            raise FileExistsError(
                f"Private package already exists: {resolved}. Use --replace-empty only before review begins."
            )
        completed = review_files_have_decisions(resolved)
        if completed:
            raise RuntimeError(f"Refusing to overwrite human decisions in: {completed}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def context_resolves(row: dict[str, str]) -> bool:
    try:
        start = int_value(row, "surface_start_in_context")
        end = int_value(row, "surface_end_in_context")
    except ValueError:
        return False
    context = row.get("context_snippet", "")
    return 0 <= start < end <= len(context) and context[start:end] == row.get("candidate_surface", "")


def annotated_context(context: str, start: int, end: int) -> str:
    if not 0 <= start < end <= len(context):
        raise ValueError("Target offsets do not resolve inside the private context")
    return f"{context[:start]}⟦TARGET⟧{context[start:end]}⟦/TARGET⟧{context[end:]}"


def claim_id_for_link(row: dict[str, str]) -> str:
    return stable_id("RCA-LINK", row["source_credit_label"], row["entity"], row["entity_type"], length=14)


def claim_id_for_co_mention(row: dict[str, str]) -> str:
    a = (row["entity_a"], row["entity_a_type"])
    b = (row["entity_b"], row["entity_b_type"])
    low, high = sorted((a, b))
    return stable_id("RCA-CO", low[0], low[1], high[0], high[1], length=14)


def membership_value(records: Sequence[tuple[str, str]]) -> tuple[str, str]:
    ordered = sorted(set(records))
    return " | ".join(item[0] for item in ordered), " | ".join(f"{item[0]}:{item[1]}" for item in ordered)


def proportional_quotas(counts: Counter[str], total: int, availability: Counter[str]) -> dict[str, int]:
    if total < 0:
        raise ValueError("Control count must be non-negative")
    if total == 0:
        return {key: 0 for key in counts}
    population = sum(counts.values())
    if population <= 0:
        raise ValueError("Cannot allocate controls without contributing occurrence types")
    exact = {key: total * value / population for key, value in counts.items()}
    quotas = {key: min(int(math.floor(value)), availability.get(key, 0)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    ordering = sorted(counts, key=lambda key: (-(exact[key] - math.floor(exact[key])), key))
    while remaining:
        progressed = False
        for key in ordering:
            if quotas[key] < availability.get(key, 0):
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError(f"Only {total - remaining} eligible real comparator controls are available")
    return quotas


def select_diverse_rows(rows: Sequence[dict[str, str]], quota: int, rank_namespace: str) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda row: deterministic_rank(rank_namespace, row["candidate_id"]))
    selected: list[dict[str, str]] = []
    used_surfaces: set[str] = set()
    used_labels: set[str] = set()
    for row in ranked:
        if len(selected) >= quota:
            break
        surface = row["candidate_surface"]
        label = row["source_credit_label"]
        if surface in used_surfaces or label in used_labels:
            continue
        selected.append(row)
        used_surfaces.add(surface)
        used_labels.add(label)
    if len(selected) < quota:
        selected_ids = {row["candidate_id"] for row in selected}
        for row in ranked:
            if row["candidate_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["candidate_id"])
            if len(selected) >= quota:
                break
    if len(selected) != quota:
        raise ValueError(f"Could select only {len(selected)} of {quota} requested controls")
    return selected


def synthetic_boundary_variant(
    row: dict[str, str], all_candidate_surfaces: set[str], released_surfaces: set[str]
) -> tuple[int, int, str, str] | None:
    context = row["context_snippet"]
    start = int_value(row, "surface_start_in_context")
    end = int_value(row, "surface_end_in_context")
    options: list[tuple[int, int, str]] = []
    if start > 0:
        options.append((start - 1, end, "EXPAND_LEFT_ONE_CHARACTER"))
    if end < len(context):
        options.append((start, end + 1, "EXPAND_RIGHT_ONE_CHARACTER"))
    options = sorted(options, key=lambda item: deterministic_rank("synthetic-boundary-option", row["candidate_id"], item[2]))
    for shifted_start, shifted_end, subtype in options:
        surface = context[shifted_start:shifted_end]
        if not surface.strip() or surface == row["candidate_surface"]:
            continue
        if surface in all_candidate_surfaces or surface in released_surfaces:
            continue
        return shifted_start, shifted_end, surface, subtype
    return None


def base_task_row(
    source_row: dict[str, str],
    *,
    task_class: str,
    control_subtype: str = "",
    expected_control_outcome: str = "",
    memberships: Sequence[tuple[str, str]] = (),
    target_start: int | None = None,
    target_end: int | None = None,
    target_surface: str | None = None,
    source_candidate_id: str | None = None,
) -> dict[str, Any]:
    start = int_value(source_row, "surface_start_in_context") if target_start is None else target_start
    end = int_value(source_row, "surface_end_in_context") if target_end is None else target_end
    surface = source_row["candidate_surface"] if target_surface is None else target_surface
    source_id = source_row["candidate_id"] if source_candidate_id is None else source_candidate_id
    admin_task_id = stable_id("RCA-OCC", task_class, source_id, start, end, surface)
    blind_task_id = stable_id("BLIND-OCC", SEED, admin_task_id)
    claim_ids, claim_roles = membership_value(memberships)
    return {
        "admin_task_id": admin_task_id,
        "blind_task_id": blind_task_id,
        "task_class": task_class,
        "control_subtype": control_subtype,
        "expected_control_outcome": expected_control_outcome,
        "source_candidate_id": source_id,
        "candidate_source": source_row["candidate_source"],
        "claim_ids": claim_ids,
        "claim_roles": claim_roles,
        "source_credit_label": source_row["source_credit_label"],
        "song_id": source_row["song_id"],
        "chunk_id": source_row["chunk_id"],
        "song_lyric_content_sha256": source_row["song_lyric_content_sha256"],
        "analysis_text_sha256": source_row["analysis_text_sha256"],
        "context_snippet": source_row["context_snippet"],
        "target_start_in_context": start,
        "target_end_in_context": end,
        "target_surface": surface,
        "proposed_entity_type": source_row["candidate_schema_type"],
        "machine_agreement_state": source_row["agreement_state"],
        "machine_transformer_confidence": source_row["transformer_confidence"],
        "annotated_context": annotated_context(source_row["context_snippet"], start, end),
    }


def reviewer_occurrence_rows(tasks: Sequence[dict[str, Any]], reviewer: str) -> list[dict[str, Any]]:
    ordered = sorted(tasks, key=lambda row: deterministic_rank("review-order", reviewer, row["blind_task_id"]))
    output: list[dict[str, Any]] = []
    for index, task in enumerate(ordered, start=1):
        row = {
            "review_order": index,
            "blind_task_id": task["blind_task_id"],
            "annotated_context": task["annotated_context"],
            "target_surface": task["target_surface"],
            "proposed_entity_type": task["proposed_entity_type"],
            "mention_valid": "",
            "boundary_valid": "",
            "referential_status": "",
            "entity_type_decision": "",
            "normalized_surface": "",
            "confidence_1_to_5": "",
            "exclusion_reason": "",
            "notes": "",
            "reviewer_id": reviewer,
            "reviewed_at_utc": "",
        }
        output.append(row)
    return output


def occurrence_adjudication_rows(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(tasks, key=lambda row: row["blind_task_id"])
    rows: list[dict[str, Any]] = []
    for task in ordered:
        rows.append(
            {
                "blind_task_id": task["blind_task_id"],
                "annotated_context": task["annotated_context"],
                "target_surface": task["target_surface"],
                "proposed_entity_type": task["proposed_entity_type"],
                "r1_mention_valid": "",
                "r1_boundary_valid": "",
                "r1_referential_status": "",
                "r1_entity_type_decision": "",
                "r2_mention_valid": "",
                "r2_boundary_valid": "",
                "r2_referential_status": "",
                "r2_entity_type_decision": "",
                "agreement_state": "",
                "adjudicated_mention_valid": "",
                "adjudicated_boundary_valid": "",
                "adjudicated_referential_status": "",
                "adjudicated_entity_type": "",
                "adjudicator_id": "",
                "adjudicated_at_utc": "",
                "adjudication_notes": "",
            }
        )
    return rows


def bundle_contexts(tasks: Sequence[dict[str, Any]]) -> str:
    ordered = sorted(tasks, key=lambda row: row["blind_task_id"])
    return "\n\n".join(f"[{row['blind_task_id']}] {row['annotated_context']}" for row in ordered)


def select_pair_comparator_tasks(
    strict_nonshared: Sequence[dict[str, str]],
    released_entity_types: set[str],
    released_pair_keys: set[tuple[tuple[str, str], tuple[str, str]]],
    requested: int,
) -> list[dict[str, Any]]:
    if requested == 0:
        return []
    entity_song_support: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_by_song_entity: dict[str, dict[tuple[str, str], list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in strict_nonshared:
        key = (row["candidate_surface"], row["candidate_schema_type"])
        if key[1] not in released_entity_types:
            continue
        song_hash = row["song_lyric_content_sha256"]
        entity_song_support[key].add(song_hash)
        rows_by_song_entity[song_hash][key].append(row)

    candidates: list[dict[str, Any]] = []
    for song_hash, entity_rows in rows_by_song_entity.items():
        supported_keys = sorted(key for key in entity_rows if len(entity_song_support[key]) >= 2)
        for a, b in combinations(supported_keys, 2):
            canonical_pair = tuple(sorted((a, b)))
            if canonical_pair in released_pair_keys:
                continue
            a_rows = entity_rows[a]
            b_rows = entity_rows[b]
            pair_task_id = stable_id("RCA-PAIR-CONTROL", a[0], a[1], b[0], b[1], song_hash)

            def private_pair_occurrence_task(row: dict[str, str], side: str) -> dict[str, Any]:
                start = int_value(row, "surface_start_in_context")
                end = int_value(row, "surface_end_in_context")
                return {
                    "blind_task_id": stable_id("BLIND-PAIR-OCC", SEED, pair_task_id, side, row["candidate_id"]),
                    "annotated_context": annotated_context(row["context_snippet"], start, end),
                }

            a_context_tasks = [private_pair_occurrence_task(row, "A") for row in a_rows]
            b_context_tasks = [private_pair_occurrence_task(row, "B") for row in b_rows]
            candidates.append(
                {
                    "claim_id": "",
                    "pair_task_id": pair_task_id,
                    "blind_pair_task_id": stable_id("BLIND-PAIR", SEED, pair_task_id),
                    "task_class": "REAL_NONCONTRIBUTING_CO_MENTION_COMPARATOR",
                    "control_subtype": "STRICT_SAME_SONG_PAIR_NOT_IN_RELEASED_CO_MENTION_SET",
                    "expected_control_outcome": "NO_ASSUMED_GOLD_LABEL_DESCRIPTIVE_ONLY",
                    "support_song_lyric_content_sha256": song_hash,
                    "source_credit_labels": " | ".join(
                        sorted({row["source_credit_label"] for row in a_rows + b_rows})
                    ),
                    "entity_a_surface": a[0],
                    "entity_a_type": a[1],
                    "entity_a_candidate_ids": " | ".join(sorted(row["candidate_id"] for row in a_rows)),
                    "entity_a_occurrence_blind_ids": " | ".join(
                        sorted(row["blind_task_id"] for row in a_context_tasks)
                    ),
                    "entity_a_occurrence_count": len(a_rows),
                    "entity_a_context_bundle": bundle_contexts(a_context_tasks),
                    "entity_b_surface": b[0],
                    "entity_b_type": b[1],
                    "entity_b_candidate_ids": " | ".join(sorted(row["candidate_id"] for row in b_rows)),
                    "entity_b_occurrence_blind_ids": " | ".join(
                        sorted(row["blind_task_id"] for row in b_context_tasks)
                    ),
                    "entity_b_occurrence_count": len(b_rows),
                    "entity_b_context_bundle": bundle_contexts(b_context_tasks),
                    "lineage_status": "REAL_SAME_SONG_COMPARATOR_NOT_A_RELEASED_CLAIM",
                    "pair_key": canonical_pair,
                }
            )

    ranked = sorted(
        candidates,
        key=lambda row: deterministic_rank(
            "real-pair-comparator",
            row["entity_a_surface"],
            row["entity_a_type"],
            row["entity_b_surface"],
            row["entity_b_type"],
            row["support_song_lyric_content_sha256"],
        ),
    )
    selected: list[dict[str, Any]] = []
    used_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    used_songs: set[str] = set()
    for row in ranked:
        if len(selected) >= requested:
            break
        if row["pair_key"] in used_pairs or row["support_song_lyric_content_sha256"] in used_songs:
            continue
        selected.append(row)
        used_pairs.add(row["pair_key"])
        used_songs.add(row["support_song_lyric_content_sha256"])
    if len(selected) < requested:
        selected_ids = {row["pair_task_id"] for row in selected}
        for row in ranked:
            if row["pair_task_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["pair_task_id"])
            if len(selected) >= requested:
                break
    if len(selected) != requested:
        raise ValueError(f"Could select only {len(selected)} of {requested} requested pair comparators")
    for row in selected:
        row.pop("pair_key", None)
    return selected


def reviewer_pair_rows(pair_tasks: Sequence[dict[str, Any]], reviewer: str) -> list[dict[str, Any]]:
    ordered = sorted(pair_tasks, key=lambda row: deterministic_rank("pair-review-order", reviewer, row["blind_pair_task_id"]))
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(ordered, start=1):
        rows.append(
            {
                "review_order": index,
                "blind_pair_task_id": task["blind_pair_task_id"],
                "entity_a_surface": task["entity_a_surface"],
                "entity_a_type": task["entity_a_type"],
                "entity_a_occurrence_blind_ids": task["entity_a_occurrence_blind_ids"],
                "entity_a_context_bundle": task["entity_a_context_bundle"],
                "entity_b_surface": task["entity_b_surface"],
                "entity_b_type": task["entity_b_type"],
                "entity_b_occurrence_blind_ids": task["entity_b_occurrence_blind_ids"],
                "entity_b_context_bundle": task["entity_b_context_bundle"],
                "entity_a_has_valid_reference": "",
                "entity_b_has_valid_reference": "",
                "pair_semantically_supported": "",
                "confidence_1_to_5": "",
                "notes": "",
                "reviewer_id": reviewer,
                "reviewed_at_utc": "",
            }
        )
    return rows


def pair_adjudication_rows(pair_tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in sorted(pair_tasks, key=lambda row: row["blind_pair_task_id"]):
        rows.append(
            {
                "blind_pair_task_id": task["blind_pair_task_id"],
                "entity_a_surface": task["entity_a_surface"],
                "entity_a_type": task["entity_a_type"],
                "entity_a_context_bundle": task["entity_a_context_bundle"],
                "entity_b_surface": task["entity_b_surface"],
                "entity_b_type": task["entity_b_type"],
                "entity_b_context_bundle": task["entity_b_context_bundle"],
                "r1_entity_a_has_valid_reference": "",
                "r1_entity_b_has_valid_reference": "",
                "r1_pair_semantically_supported": "",
                "r2_entity_a_has_valid_reference": "",
                "r2_entity_b_has_valid_reference": "",
                "r2_pair_semantically_supported": "",
                "agreement_state": "",
                "adjudicated_pair_supported": "",
                "adjudicator_id": "",
                "adjudicated_at_utc": "",
                "adjudication_notes": "",
            }
        )
    return rows


def every_decision_blank(rows: Sequence[dict[str, Any]], fields: Iterable[str]) -> bool:
    return all(not str(row.get(field, "")).strip() for row in rows for field in fields)


def no_admin_fields(rows: Sequence[dict[str, Any]]) -> bool:
    return not rows or not (set(rows[0]) & ADMIN_ONLY_FIELDS)


def relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def build_readme(counts: dict[str, int]) -> str:
    return f"""# PRIVATE — released-claim NER audit v1

This directory contains copyrighted lyric contexts and source locators. Keep it local. Do not commit, publish, email, or upload it to an unapproved service.

## What is covered

- {counts['released_link_claims']} released source-label/entity claims;
- {counts['released_co_mention_claims']} released entity co-mention claims;
- all {counts['unique_contributing_occurrence_rows']} unique occurrence rows that contribute to those claims;
- {counts['released_co_mention_pair_tasks']} released co-mention claim × supporting-song checks;
- {counts['real_co_mention_comparator_tasks']} real non-contributing co-mention comparator checks;
- {counts['real_comparator_controls']} real non-contributing comparator candidates (no assumed gold label);
- {counts['synthetic_boundary_controls']} blinded synthetic boundary-corruption attention controls.

This is a targeted released-claim audit, not the planned 800-item corpus-wide NER benchmark. It cannot yield corpus-wide precision, recall, or F1.

## Coordinator-only files

- `occurrence_task_manifest_private.csv` — claim membership, control status, and source lineage.
- `co_mention_support_manifest_private.csv` — released co-mention claim/song lineage.
- `claim_coverage_private.csv` — exact coverage reconciliation.
- `negative_control_manifest_private.csv` — control identities and expected attention outcomes.
- `private_validation.json` and `private_manifest.json` — checksums and package validation.

Do not give coordinator-only files to reviewers before both review sheets are locked.

## Reviewer files

Give Reviewer 1 only `reviewer_R1_occurrences_private.csv`, `reviewer_R1_co_mentions_private.csv`, and the public protocol. Give Reviewer 2 the corresponding R2 files. Reviewers work independently and must not compare decisions.

Allowed categorical values are defined in `methods/NER_RELEASED_CLAIM_AUDIT_PROTOCOL.md`. Do not change task IDs, context text, target markers, or row order. Save as UTF-8 CSV.

## Adjudication

After both reviewers lock their sheets, transfer the two decisions into the adjudication templates. The adjudicator resolves disagreements while still blinded to `negative_control_manifest_private.csv`. Reveal control status only after adjudication is locked.

The scientific release must remain provisional until the completed sheets are scored and the original claim gates are rerun with adjudicated-valid occurrences.
"""


def main() -> None:
    args = parse_args()
    if (
        args.real_comparator_controls < 0
        or args.synthetic_boundary_controls < 0
        or args.real_pair_comparator_controls < 0
    ):
        raise ValueError("Control counts must be non-negative")

    for path in (
        args.private_candidates,
        args.private_source_manifest,
        args.public_links,
        args.public_co_mentions,
        args.public_ner_manifest,
        args.protocol,
    ):
        require_file(path)

    private_source_manifest = read_json(args.private_source_manifest)
    public_ner_manifest = read_json(args.public_ner_manifest)
    assert_manifest_entry(args.private_candidates, private_source_manifest, args.private_candidates.name)
    assert_manifest_entry(args.public_links, public_ner_manifest, args.public_links.name)
    assert_manifest_entry(args.public_co_mentions, public_ner_manifest, args.public_co_mentions.name)

    candidates = read_csv_rows(args.private_candidates)
    links = read_csv_rows(args.public_links)
    co_mentions = read_csv_rows(args.public_co_mentions)
    require_columns(
        candidates,
        {
            "candidate_id",
            "candidate_source",
            "candidate_surface",
            "candidate_schema_type",
            "agreement_state",
            "source_credit_label",
            "song_id",
            "chunk_id",
            "song_lyric_content_sha256",
            "analysis_text_sha256",
            "cross_label_shared_cleaned_text",
            "surface_start_in_context",
            "surface_end_in_context",
            "context_snippet",
            "transformer_confidence",
            "strict_high_consistency",
        },
        "private candidate occurrences",
    )
    require_columns(
        links,
        {
            "source_credit_label",
            "entity",
            "entity_type",
            "entity_song_units_within_label",
            "release_gate_pass",
            "status",
        },
        "released source-label/entity links",
    )
    require_columns(
        co_mentions,
        {
            "entity_a",
            "entity_a_type",
            "entity_b",
            "entity_b_type",
            "unique_song_unit_co_mentions",
            "release_gate_pass",
            "status",
        },
        "released entity co-mentions",
    )

    candidate_ids = [row["candidate_id"] for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Private candidate IDs are not unique")
    bad_contexts = [row["candidate_id"] for row in candidates if not context_resolves(row)]
    if bad_contexts:
        raise ValueError(f"{len(bad_contexts)} private candidate contexts do not resolve to their target surface")
    if not all(as_bool(row["release_gate_pass"]) for row in links + co_mentions):
        raise ValueError("Public claim input includes a row that did not pass its release gate")

    strict_nonshared = [
        row
        for row in candidates
        if as_bool(row["strict_high_consistency"]) and not as_bool(row["cross_label_shared_cleaned_text"])
    ]
    by_entity: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_candidate_id = {row["candidate_id"]: row for row in candidates}
    for row in strict_nonshared:
        by_entity[(row["candidate_surface"], row["candidate_schema_type"])].append(row)

    memberships: dict[str, list[tuple[str, str]]] = defaultdict(list)
    link_coverage: list[dict[str, Any]] = []
    link_occurrence_ids: set[str] = set()
    link_claim_song_units: set[tuple[str, str]] = set()
    sorted_links = sorted(links, key=lambda row: (row["source_credit_label"], row["entity"], row["entity_type"]))
    for link in sorted_links:
        claim_id = claim_id_for_link(link)
        matched = [
            row
            for row in by_entity[(link["entity"], link["entity_type"])]
            if row["source_credit_label"] == link["source_credit_label"]
        ]
        unique_songs = {row["song_lyric_content_sha256"] for row in matched}
        public_support = int_value(link, "entity_song_units_within_label")
        if len(unique_songs) != public_support:
            raise ValueError(
                f"Link support mismatch for {claim_id}: recomputed {len(unique_songs)}, public {public_support}"
            )
        for row in matched:
            memberships[row["candidate_id"]].append((claim_id, "LINK_OCCURRENCE"))
            link_occurrence_ids.add(row["candidate_id"])
            link_claim_song_units.add((claim_id, row["song_lyric_content_sha256"]))
        link_coverage.append(
            {
                "claim_id": claim_id,
                "claim_type": "SOURCE_LABEL_ENTITY_LINK",
                "source_credit_label": link["source_credit_label"],
                "entity_a": link["entity"],
                "entity_a_type": link["entity_type"],
                "entity_b": "",
                "entity_b_type": "",
                "public_support_song_units": public_support,
                "recomputed_support_song_units": len(unique_songs),
                "contributing_occurrence_rows": len(matched),
                "review_task_rows": len({row["candidate_id"] for row in matched}),
                "support_pair_tasks": 0,
                "coverage_status": "COMPLETE_PENDING_HUMAN_DECISIONS",
            }
        )

    co_support_records: list[dict[str, Any]] = []
    co_coverage: list[dict[str, Any]] = []
    co_occurrence_ids: set[str] = set()
    sorted_co_mentions = sorted(
        co_mentions,
        key=lambda row: (row["entity_a"], row["entity_a_type"], row["entity_b"], row["entity_b_type"]),
    )
    for co in sorted_co_mentions:
        claim_id = claim_id_for_co_mention(co)
        rows_a = by_entity[(co["entity_a"], co["entity_a_type"])]
        rows_b = by_entity[(co["entity_b"], co["entity_b_type"])]
        a_by_song: dict[str, list[dict[str, str]]] = defaultdict(list)
        b_by_song: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows_a:
            a_by_song[row["song_lyric_content_sha256"]].append(row)
        for row in rows_b:
            b_by_song[row["song_lyric_content_sha256"]].append(row)
        supporting_songs = sorted(set(a_by_song) & set(b_by_song))
        public_support = int_value(co, "unique_song_unit_co_mentions")
        if len(supporting_songs) != public_support:
            raise ValueError(
                f"Co-mention support mismatch for {claim_id}: recomputed {len(supporting_songs)}, public {public_support}"
            )
        claim_occurrence_ids: set[str] = set()
        for song_hash in supporting_songs:
            a_rows = a_by_song[song_hash]
            b_rows = b_by_song[song_hash]
            for row in a_rows:
                memberships[row["candidate_id"]].append((claim_id, "CO_ENTITY_A"))
                co_occurrence_ids.add(row["candidate_id"])
                claim_occurrence_ids.add(row["candidate_id"])
            for row in b_rows:
                memberships[row["candidate_id"]].append((claim_id, "CO_ENTITY_B"))
                co_occurrence_ids.add(row["candidate_id"])
                claim_occurrence_ids.add(row["candidate_id"])
            source_labels = sorted({row["source_credit_label"] for row in a_rows + b_rows})
            pair_task_id = stable_id("RCA-PAIR", claim_id, song_hash)
            co_support_records.append(
                {
                    "claim_id": claim_id,
                    "pair_task_id": pair_task_id,
                    "blind_pair_task_id": stable_id("BLIND-PAIR", SEED, pair_task_id),
                    "task_class": "RELEASED_CO_MENTION_SUPPORT_SONG",
                    "control_subtype": "",
                    "expected_control_outcome": "",
                    "support_song_lyric_content_sha256": song_hash,
                    "source_credit_labels": " | ".join(source_labels),
                    "entity_a_surface": co["entity_a"],
                    "entity_a_type": co["entity_a_type"],
                    "entity_a_candidate_ids": " | ".join(sorted(row["candidate_id"] for row in a_rows)),
                    "entity_b_surface": co["entity_b"],
                    "entity_b_type": co["entity_b_type"],
                    "entity_b_candidate_ids": " | ".join(sorted(row["candidate_id"] for row in b_rows)),
                    "entity_a_occurrence_count": len(a_rows),
                    "entity_b_occurrence_count": len(b_rows),
                    "lineage_status": "RECOMPUTED_SUPPORT_MATCHES_PUBLIC_RELEASE",
                }
            )
        co_coverage.append(
            {
                "claim_id": claim_id,
                "claim_type": "SAME_SONG_ENTITY_CO_MENTION",
                "source_credit_label": "",
                "entity_a": co["entity_a"],
                "entity_a_type": co["entity_a_type"],
                "entity_b": co["entity_b"],
                "entity_b_type": co["entity_b_type"],
                "public_support_song_units": public_support,
                "recomputed_support_song_units": len(supporting_songs),
                "contributing_occurrence_rows": len(claim_occurrence_ids),
                "review_task_rows": len(claim_occurrence_ids),
                "support_pair_tasks": len(supporting_songs),
                "coverage_status": "COMPLETE_PENDING_HUMAN_DECISIONS",
            }
        )

    positive_ids = link_occurrence_ids | co_occurrence_ids
    if not positive_ids:
        raise ValueError("No contributing released-claim occurrences were selected")
    positive_tasks = [
        base_task_row(
            by_candidate_id[candidate_id],
            task_class="RELEASED_CLAIM_OCCURRENCE",
            memberships=memberships[candidate_id],
        )
        for candidate_id in sorted(positive_ids)
    ]

    released_surfaces = {
        row["entity"] for row in links
    } | {row["entity_a"] for row in co_mentions} | {row["entity_b"] for row in co_mentions}
    released_types = Counter(task["proposed_entity_type"] for task in positive_tasks)
    comparator_pool = [
        row
        for row in candidates
        if row["candidate_id"] not in positive_ids
        and not as_bool(row["cross_label_shared_cleaned_text"])
        and not as_bool(row["strict_high_consistency"])
        and row["candidate_schema_type"] in released_types
        and row["candidate_surface"] not in released_surfaces
        and row["agreement_state"] in {"LEXICON_ONLY", "TYPE_OR_BOUNDARY_CONFLICT"}
    ]
    comparator_availability = Counter(row["candidate_schema_type"] for row in comparator_pool)
    comparator_quotas = proportional_quotas(
        released_types, args.real_comparator_controls, comparator_availability
    )
    comparator_rows: list[dict[str, str]] = []
    for entity_type, quota in sorted(comparator_quotas.items()):
        eligible = [row for row in comparator_pool if row["candidate_schema_type"] == entity_type]
        comparator_rows.extend(select_diverse_rows(eligible, quota, f"real-comparator-{entity_type}"))
    comparator_tasks = [
        base_task_row(
            row,
            task_class="REAL_NONCONTRIBUTING_COMPARATOR",
            control_subtype="AUTOMATED_DISAGREEMENT_OR_SINGLE_METHOD_CANDIDATE",
            expected_control_outcome="NO_ASSUMED_GOLD_LABEL_DESCRIPTIVE_ONLY",
        )
        for row in comparator_rows
    ]

    all_candidate_surfaces = {row["candidate_surface"] for row in candidates}
    synthetic_tasks: list[dict[str, Any]] = []
    for task in sorted(positive_tasks, key=lambda row: deterministic_rank("synthetic-source", row["admin_task_id"])):
        if len(synthetic_tasks) >= args.synthetic_boundary_controls:
            break
        source_row = by_candidate_id[task["source_candidate_id"]]
        variant = synthetic_boundary_variant(source_row, all_candidate_surfaces, released_surfaces)
        if variant is None:
            continue
        start, end, surface, subtype = variant
        synthetic_tasks.append(
            base_task_row(
                source_row,
                task_class="SYNTHETIC_BOUNDARY_ATTENTION_CONTROL",
                control_subtype=subtype,
                expected_control_outcome="INVALID_BOUNDARY",
                target_start=start,
                target_end=end,
                target_surface=surface,
                source_candidate_id=source_row["candidate_id"],
            )
        )
    if len(synthetic_tasks) != args.synthetic_boundary_controls:
        raise ValueError(
            f"Could create only {len(synthetic_tasks)} of {args.synthetic_boundary_controls} synthetic boundary controls"
        )

    all_tasks = positive_tasks + comparator_tasks + synthetic_tasks
    if len({task["admin_task_id"] for task in all_tasks}) != len(all_tasks):
        raise ValueError("Audit admin task IDs are not unique")
    if len({task["blind_task_id"] for task in all_tasks}) != len(all_tasks):
        raise ValueError("Audit blind task IDs are not unique")

    task_by_source_candidate = {
        task["source_candidate_id"]: task
        for task in positive_tasks
    }
    released_pair_tasks: list[dict[str, Any]] = []
    for support in co_support_records:
        a_source_ids = support["entity_a_candidate_ids"].split(" | ")
        b_source_ids = support["entity_b_candidate_ids"].split(" | ")
        a_tasks = [task_by_source_candidate[candidate_id] for candidate_id in a_source_ids]
        b_tasks = [task_by_source_candidate[candidate_id] for candidate_id in b_source_ids]
        released_pair_tasks.append(
            {
                **support,
                "entity_a_occurrence_blind_ids": " | ".join(sorted(task["blind_task_id"] for task in a_tasks)),
                "entity_a_context_bundle": bundle_contexts(a_tasks),
                "entity_b_occurrence_blind_ids": " | ".join(sorted(task["blind_task_id"] for task in b_tasks)),
                "entity_b_context_bundle": bundle_contexts(b_tasks),
            }
        )

    released_pair_keys = {
        tuple(sorted(((row["entity_a"], row["entity_a_type"]), (row["entity_b"], row["entity_b_type"]))))
        for row in co_mentions
    }
    pair_comparator_tasks = select_pair_comparator_tasks(
        strict_nonshared,
        set(released_types),
        released_pair_keys,
        args.real_pair_comparator_controls,
    )
    pair_tasks = released_pair_tasks + pair_comparator_tasks

    reviewer_r1 = reviewer_occurrence_rows(all_tasks, "R1")
    reviewer_r2 = reviewer_occurrence_rows(all_tasks, "R2")
    occurrence_adjudication = occurrence_adjudication_rows(all_tasks)
    pair_reviewer_r1 = reviewer_pair_rows(pair_tasks, "R1")
    pair_reviewer_r2 = reviewer_pair_rows(pair_tasks, "R2")
    pair_adjudication = pair_adjudication_rows(pair_tasks)

    counts = {
        "released_link_claims": len(links),
        "released_co_mention_claims": len(co_mentions),
        "released_claims_total": len(links) + len(co_mentions),
        "link_contributing_occurrence_rows": len(link_occurrence_ids),
        "co_mention_contributing_occurrence_rows": len(co_occurrence_ids),
        "overlapping_link_and_co_mention_occurrence_rows": len(link_occurrence_ids & co_occurrence_ids),
        "unique_contributing_occurrence_rows": len(positive_ids),
        "link_claim_song_unit_memberships": len(link_claim_song_units),
        "unique_link_support_song_units": len({song for _, song in link_claim_song_units}),
        "released_co_mention_pair_tasks": len(released_pair_tasks),
        "real_co_mention_comparator_tasks": len(pair_comparator_tasks),
        "co_mention_pair_review_tasks_total": len(pair_tasks),
        "unique_co_mention_support_song_units": len(
            {record["support_song_lyric_content_sha256"] for record in co_support_records}
        ),
        "real_comparator_controls": len(comparator_tasks),
        "synthetic_boundary_controls": len(synthetic_tasks),
        "occurrence_review_tasks_total": len(all_tasks),
    }

    prepare_private_output(args.private_output, args.replace_empty)

    task_manifest_fields = [
        "admin_task_id",
        "blind_task_id",
        "task_class",
        "control_subtype",
        "expected_control_outcome",
        "source_candidate_id",
        "candidate_source",
        "claim_ids",
        "claim_roles",
        "source_credit_label",
        "song_id",
        "chunk_id",
        "song_lyric_content_sha256",
        "analysis_text_sha256",
        "context_snippet",
        "target_start_in_context",
        "target_end_in_context",
        "target_surface",
        "proposed_entity_type",
        "machine_agreement_state",
        "machine_transformer_confidence",
        "annotated_context",
    ]
    reviewer_fields = list(reviewer_r1[0])
    occurrence_adjudication_fields = list(occurrence_adjudication[0])
    pair_manifest_fields = [
        "claim_id",
        "pair_task_id",
        "blind_pair_task_id",
        "task_class",
        "control_subtype",
        "expected_control_outcome",
        "support_song_lyric_content_sha256",
        "source_credit_labels",
        "entity_a_surface",
        "entity_a_type",
        "entity_a_candidate_ids",
        "entity_a_occurrence_blind_ids",
        "entity_a_occurrence_count",
        "entity_a_context_bundle",
        "entity_b_surface",
        "entity_b_type",
        "entity_b_candidate_ids",
        "entity_b_occurrence_blind_ids",
        "entity_b_occurrence_count",
        "entity_b_context_bundle",
        "lineage_status",
    ]
    pair_reviewer_fields = list(pair_reviewer_r1[0])
    pair_adjudication_fields = list(pair_adjudication[0])
    coverage_rows = sorted(link_coverage + co_coverage, key=lambda row: row["claim_id"])
    coverage_fields = list(coverage_rows[0])
    control_rows = [
        {
            "admin_task_id": task["admin_task_id"],
            "blind_task_id": task["blind_task_id"],
            "task_class": task["task_class"],
            "control_subtype": task["control_subtype"],
            "expected_control_outcome": task["expected_control_outcome"],
            "source_candidate_id": task["source_candidate_id"],
            "target_surface": task["target_surface"],
            "proposed_entity_type": task["proposed_entity_type"],
        }
        for task in comparator_tasks + synthetic_tasks
    ]
    control_rows.extend(
        {
            "admin_task_id": task["pair_task_id"],
            "blind_task_id": task["blind_pair_task_id"],
            "task_class": task["task_class"],
            "control_subtype": task["control_subtype"],
            "expected_control_outcome": task["expected_control_outcome"],
            "source_candidate_id": (
                f"A:{task['entity_a_candidate_ids']} | B:{task['entity_b_candidate_ids']}"
            ),
            "target_surface": f"{task['entity_a_surface']} | {task['entity_b_surface']}",
            "proposed_entity_type": f"{task['entity_a_type']} | {task['entity_b_type']}",
        }
        for task in pair_comparator_tasks
    )
    control_fields = list(control_rows[0]) if control_rows else [
        "admin_task_id", "blind_task_id", "task_class", "control_subtype",
        "expected_control_outcome", "source_candidate_id", "target_surface", "proposed_entity_type",
    ]

    write_csv_rows(args.private_output / "occurrence_task_manifest_private.csv", all_tasks, task_manifest_fields)
    write_csv_rows(args.private_output / "reviewer_R1_occurrences_private.csv", reviewer_r1, reviewer_fields)
    write_csv_rows(args.private_output / "reviewer_R2_occurrences_private.csv", reviewer_r2, reviewer_fields)
    write_csv_rows(
        args.private_output / "occurrence_adjudication_private.csv",
        occurrence_adjudication,
        occurrence_adjudication_fields,
    )
    write_csv_rows(args.private_output / "co_mention_support_manifest_private.csv", pair_tasks, pair_manifest_fields)
    write_csv_rows(args.private_output / "reviewer_R1_co_mentions_private.csv", pair_reviewer_r1, pair_reviewer_fields)
    write_csv_rows(args.private_output / "reviewer_R2_co_mentions_private.csv", pair_reviewer_r2, pair_reviewer_fields)
    write_csv_rows(
        args.private_output / "co_mention_adjudication_private.csv", pair_adjudication, pair_adjudication_fields
    )
    write_csv_rows(args.private_output / "claim_coverage_private.csv", coverage_rows, coverage_fields)
    write_csv_rows(args.private_output / "negative_control_manifest_private.csv", control_rows, control_fields)
    (args.private_output / "README_PRIVATE.md").write_text(
        build_readme(counts).rstrip() + "\n", encoding="utf-8", newline="\n"
    )

    checks = [
        {
            "name": "private_source_candidate_manifest_hash_matches",
            "passed": sha256_file(args.private_candidates)
            == private_source_manifest["files"][args.private_candidates.name]["sha256"],
        },
        {
            "name": "public_released_claim_inputs_match_ner_manifest",
            "passed": all(
                sha256_file(path) == public_ner_manifest["files"][path.name]["sha256"]
                for path in (args.public_links, args.public_co_mentions)
            ),
        },
        {"name": "source_candidate_ids_are_unique", "passed": len(candidate_ids) == len(set(candidate_ids))},
        {"name": "all_source_context_offsets_resolve", "passed": not bad_contexts},
        {"name": "all_public_claim_rows_pass_release_gate", "passed": all(as_bool(row["release_gate_pass"]) for row in links + co_mentions)},
        {"name": "all_released_link_support_reconciles", "passed": all(row["public_support_song_units"] == row["recomputed_support_song_units"] for row in link_coverage)},
        {"name": "all_released_co_mention_support_reconciles", "passed": all(row["public_support_song_units"] == row["recomputed_support_song_units"] for row in co_coverage)},
        {"name": "every_selected_positive_occurrence_has_claim_membership", "passed": all(memberships[candidate_id] for candidate_id in positive_ids)},
        {"name": "all_released_claim_occurrences_have_one_review_task", "passed": len(positive_tasks) == len(positive_ids)},
        {"name": "all_co_mention_support_song_units_have_pair_task", "passed": len(released_pair_tasks) == sum(int_value(row, "unique_song_unit_co_mentions") for row in co_mentions)},
        {"name": "real_comparator_control_count_matches_request", "passed": len(comparator_tasks) == args.real_comparator_controls},
        {"name": "real_comparators_are_not_assumed_gold", "passed": all(task["expected_control_outcome"] == "NO_ASSUMED_GOLD_LABEL_DESCRIPTIVE_ONLY" for task in comparator_tasks)},
        {"name": "synthetic_boundary_control_count_matches_request", "passed": len(synthetic_tasks) == args.synthetic_boundary_controls},
        {"name": "synthetic_controls_have_invalid_boundary_expectation", "passed": all(task["expected_control_outcome"] == "INVALID_BOUNDARY" for task in synthetic_tasks)},
        {"name": "real_pair_comparator_count_matches_request", "passed": len(pair_comparator_tasks) == args.real_pair_comparator_controls},
        {"name": "real_pair_comparators_are_not_assumed_gold", "passed": all(task["expected_control_outcome"] == "NO_ASSUMED_GOLD_LABEL_DESCRIPTIVE_ONLY" for task in pair_comparator_tasks)},
        {"name": "reviewer_occurrence_rows_match_all_tasks", "passed": len(reviewer_r1) == len(reviewer_r2) == len(all_tasks)},
        {"name": "reviewer_pair_rows_match_support_tasks", "passed": len(pair_reviewer_r1) == len(pair_reviewer_r2) == len(pair_tasks)},
        {"name": "reviewer_occurrence_orders_are_independently_shuffled", "passed": [row["blind_task_id"] for row in reviewer_r1] != [row["blind_task_id"] for row in reviewer_r2]},
        {"name": "reviewer_pair_orders_are_independently_shuffled", "passed": [row["blind_pair_task_id"] for row in pair_reviewer_r1] != [row["blind_pair_task_id"] for row in pair_reviewer_r2]},
        {"name": "reviewer_occurrence_templates_exclude_admin_fields", "passed": no_admin_fields(reviewer_r1) and no_admin_fields(reviewer_r2)},
        {"name": "reviewer_pair_templates_exclude_admin_fields", "passed": no_admin_fields(pair_reviewer_r1) and no_admin_fields(pair_reviewer_r2)},
        {"name": "reviewer_occurrence_decisions_are_blank", "passed": every_decision_blank(reviewer_r1 + reviewer_r2, REVIEW_DECISION_FIELDS)},
        {"name": "reviewer_pair_decisions_are_blank", "passed": every_decision_blank(pair_reviewer_r1 + pair_reviewer_r2, PAIR_REVIEW_DECISION_FIELDS)},
        {"name": "occurrence_adjudication_decisions_are_blank", "passed": every_decision_blank(occurrence_adjudication, [field for field in occurrence_adjudication_fields if field.startswith("r1_") or field.startswith("r2_") or field.startswith("adjudicated_") or field in {"agreement_state", "adjudicator_id", "adjudication_notes"}])},
        {"name": "pair_adjudication_decisions_are_blank", "passed": every_decision_blank(pair_adjudication, [field for field in pair_adjudication_fields if field.startswith("r1_") or field.startswith("r2_") or field.startswith("adjudicated_") or field in {"agreement_state", "adjudicator_id", "adjudication_notes"}])},
        {"name": "private_output_is_outside_repository", "passed": REPO_ROOT.resolve() not in args.private_output.resolve().parents},
        {"name": "private_output_is_exact_workspace_work_target", "passed": args.private_output.resolve().parent == WORK_ROOT.resolve() and args.private_output.resolve().name == PRIVATE_ARTIFACT_ID},
    ]
    validation = {
        "artifact_id": PRIVATE_ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "human_review_status": "PENDING_DUAL_REVIEW_AND_ADJUDICATION",
        "classification": "PRIVATE_LOCAL_ONLY_CONTAINS_COPYRIGHTED_LYRIC_CONTEXTS_AND_SOURCE_LOCATORS",
        "counts": counts,
        "checks": checks,
    }
    write_json(args.private_output / "private_validation.json", validation)
    if validation["status"] != "pass":
        raise AssertionError(json.dumps(validation, ensure_ascii=False, indent=2))

    generated_files = sorted(
        path for path in args.private_output.iterdir() if path.is_file() and path.name != "private_manifest.json"
    )
    private_manifest = {
        "artifact_id": PRIVATE_ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": validation["generated_at_utc"],
        "classification": validation["classification"],
        "status": "PENDING_HUMAN_REVIEW",
        "claim_boundary": "Claim-conditioned occurrence audit only; not corpus-wide NER gold and not a source of global precision, recall, or F1.",
        "counts": counts,
        "inputs": {
            "private_candidate_occurrences_sha256": sha256_file(args.private_candidates),
            "private_source_manifest_sha256": sha256_file(args.private_source_manifest),
            "released_source_label_entity_links_sha256": sha256_file(args.public_links),
            "released_entity_co_mentions_sha256": sha256_file(args.public_co_mentions),
            "public_ner_manifest_sha256": sha256_file(args.public_ner_manifest),
            "builder_code_sha256": sha256_file(Path(__file__)),
            "protocol_sha256": sha256_file(args.protocol),
            "sampling_seed": SEED,
            "real_comparator_controls_requested": args.real_comparator_controls,
            "synthetic_boundary_controls_requested": args.synthetic_boundary_controls,
            "real_pair_comparator_controls_requested": args.real_pair_comparator_controls,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in generated_files
        },
    }
    write_json(args.private_output / "private_manifest.json", private_manifest)
    private_manifest_hash = sha256_file(args.private_output / "private_manifest.json")

    status = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": validation["generated_at_utc"],
        "status": "PENDING_DUAL_HUMAN_REVIEW_AND_ADJUDICATION",
        "scope": counts,
        "review_progress": {
            "reviewer_1_occurrence_decisions_completed": 0,
            "reviewer_2_occurrence_decisions_completed": 0,
            "occurrence_tasks_adjudicated": 0,
            "reviewer_1_co_mention_decisions_completed": 0,
            "reviewer_2_co_mention_decisions_completed": 0,
            "co_mention_tasks_adjudicated": 0,
        },
        "global_ner_benchmark": {
            "status": "NOT_COMPLETED",
            "planned_dual_review_items": 800,
            "completed_gold_items": 0,
            "precision_recall_f1": "WITHHELD",
        },
        "targeted_audit_metrics": {
            "released_claim_occurrence_confirmation_rate": "WITHHELD_PENDING_ADJUDICATION",
            "inter_annotator_agreement": "WITHHELD_PENDING_BOTH_REVIEWERS",
            "synthetic_boundary_attention_control_pass_rate": "WITHHELD_PENDING_REVIEW",
            "co_mention_support_confirmation": "WITHHELD_PENDING_ADJUDICATION",
        },
        "evidence_boundary": (
            "The package covers every occurrence contributing to the currently released NER edges and co-mentions. "
            "It is claim-conditioned, includes no corpus-wide negative frame, and therefore cannot establish "
            "corpus-wide NER precision, recall, or F1. Public claims remain provisional until dual review, "
            "adjudication, and gate reruns are complete."
        ),
        "privacy": {
            "public_lyric_or_context_rows": 0,
            "public_song_or_chunk_locators": 0,
            "private_package_classification": validation["classification"],
            "private_package_location_published": False,
        },
        "validation": {
            "package_generation": "pass",
            "released_claim_occurrence_coverage": 1.0,
            "released_link_support_reconciliation": "pass",
            "released_co_mention_support_reconciliation": "pass",
            "review_templates_blank": True,
            "reviewer_claim_and_control_status_blinded": True,
            "private_manifest_sha256": private_manifest_hash,
            "private_validation_sha256": sha256_file(args.private_output / "private_validation.json"),
        },
        "input_lineage": {
            "released_source_label_entity_links_sha256": sha256_file(args.public_links),
            "released_entity_co_mentions_sha256": sha256_file(args.public_co_mentions),
            "private_candidate_occurrences_sha256": sha256_file(args.private_candidates),
            "private_source_manifest_sha256": sha256_file(args.private_source_manifest),
            "builder_code_sha256": sha256_file(Path(__file__)),
            "protocol_sha256": sha256_file(args.protocol),
        },
        "protocol": "methods/NER_RELEASED_CLAIM_AUDIT_PROTOCOL.md",
        "next_action": (
            "Two independent reviewers complete their blinded occurrence and co-mention sheets; an adjudicator "
            "then resolves disagreements before control identities are revealed and the original release gates are rerun."
        ),
    }
    write_json(args.public_status, status)

    status_text = args.public_status.read_text(encoding="utf-8")
    forbidden_public_tokens = (
        "context_snippet",
        "song_id",
        "chunk_id",
        "source_credit_label",
        str(args.private_output.resolve()),
        "⟦TARGET⟧",
    )
    hits = [token for token in forbidden_public_tokens if token in status_text]
    if hits:
        raise AssertionError(f"Public status contains private schema/path tokens: {hits}")

    print(
        json.dumps(
            {
                "status": "pass",
                "human_review_status": status["status"],
                "counts": counts,
                "public_status": relative_repo_path(args.public_status),
                "public_status_sha256": sha256_file(args.public_status),
                "private_manifest_sha256": private_manifest_hash,
                "private_validation_sha256": sha256_file(args.private_output / "private_validation.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
