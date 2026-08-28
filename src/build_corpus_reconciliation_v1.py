#!/usr/bin/env python3
"""Build an aggregate-only raw-to-frozen corpus reconciliation artifact.

The builder requires lawful local access to the private lyric-chunk export and
source metadata. Public outputs contain only aggregate counts, input hashes,
method text, and validation results; they never contain lyrics, source-credit
labels, titles, or song/chunk identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_canonical_lyric_text_sidecar_v1 import clean_initial_headers
from build_chinese_rap_written_rhyme_v1 import (
    HAN_RE,
    display_normalise,
    duplicate_normalise,
    is_header_line,
    terminal_content_kind,
    written_ending_features,
)


ARTIFACT_ID = "chinese-rap-corpus-reconciliation-v1"
VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "corpus-reconciliation-v1"

TITLE_EXCLUDE_PATTERNS = (
    r"[\(（]Live[\)）]",
    r"[\(（]live[\)）]",
    r"[\(（]伴奏[\)）]",
    r"伴奏$",
    r"[\(（]Instrumental[\)）]",
    r"[\(（]instrumental[\)）]",
)
TITLE_EXCLUDE_RE = re.compile("|".join(TITLE_EXCLUDE_PATTERNS), re.IGNORECASE)

CREDIT_LINE_PATTERNS = (
    r"^出品",
    r"^Prod[\.\s]",
    r"^prod[\.\s]",
    r"^Beat\s*by",
    r"^Mixed\s*by",
    r"^Mastered\s*by",
    r"^录音",
    r"^混音",
    r"^母带",
    r"^编曲",
    r"^作词",
    r"^作曲",
    r"^演唱",
    r"^制作人",
    r"^Recording",
    r"^Mixing",
    r"^Mastering",
    r"^OP\s*[:：]",
    r"^SP\s*[:：]",
)
CREDIT_LINE_RE = re.compile("|".join(CREDIT_LINE_PATTERNS), re.IGNORECASE)
COPYRIGHT_RE = re.compile(r"未经.*书面许可|未经.*权利人", re.UNICODE)
STRUCTURE_LINE_RE = re.compile(
    r"^\s*[\[\(（【]*\s*"
    r"(Verse|Hook|Chorus|Bridge|Intro|Outro|Pre-?Chorus|Refrain|Interlude)"
    r"[\s\d]*[\]\)）】]*\s*[:：]?\s*$",
    re.IGNORECASE,
)
SPEAKER_LABEL_RE = re.compile(r"^[\w\s\.\-]{1,20}[：:]\s*", re.UNICODE)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bool_field(value: object) -> bool:
    return str(value).strip().lower() == "true"


def stable_label_id(value: str) -> str:
    return "ALBL-" + sha256_text("chinese-rap-source-artist-label-v2\t" + value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def canonical_chunk_rows_sha256(raw: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in raw.itertuples(index=False):
        canonical = json.dumps(
            [str(row.artist), str(row.song_id), str(row.song_title), int(row.chunk_id), str(row.text)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update((canonical + "\n").encode("utf-8"))
    return digest.hexdigest()


def summarise_drive_lineage(
    payload: dict[str, Any], raw: pd.DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mismatch = payload["mismatch_columns"]
    text_diagnostics = payload["text_mismatch_diagnostics"]
    title_diagnostics = payload["title_mismatch_diagnostics"]
    artist_diagnostics = payload["artist_mismatch_diagnostics"]
    adjudications = payload["adjudication_counts"]

    artist_normalisation_equivalent = int(artist_diagnostics.get("whitespace_collapsed_equal", 0))
    title_normalisation_equivalent = int(title_diagnostics.get("whitespace_collapsed_equal", 0))
    text_newline_only = int(text_diagnostics.get("newline_normalised_equal", 0))
    title_type_coercions = int(adjudications.get("drive_native_sheet_type_coercion", 0))
    leading_apostrophe_escapes = int(adjudications.get("drive_leading_apostrophe_escape_semantics", 0))
    control_import_side_effects = int(adjudications.get("drive_control_character_import_side_effect", 0))
    unresolved = int(payload.get("unresolved_review_rows", -1))

    local_canonical_sha256 = canonical_chunk_rows_sha256(raw)
    lineage = {
        "comparison_scope": "live native Google Sheet versus the local raw chunk export, row aligned",
        "live_sheet_rows": int(payload["remote_rows"]),
        "local_export_rows": int(payload["local_rows"]),
        "row_count_match": int(payload["remote_rows"]) == int(payload["local_rows"]) == len(raw),
        "local_canonical_content_bound_to_current_raw_export": (
            str(payload["local_canonical_sha256"]) == local_canonical_sha256
        ),
        "song_id_exact_mismatches": int(mismatch["song_id"]),
        "chunk_id_exact_mismatches": int(mismatch["chunk_id"]),
        "artist_exact_mismatches": int(mismatch["artist"]),
        "artist_normalisation_equivalent_mismatches": artist_normalisation_equivalent,
        "title_exact_mismatches": int(mismatch["song_title"]),
        "title_normalisation_equivalent_mismatches": title_normalisation_equivalent,
        "drive_native_sheet_title_type_coercions": title_type_coercions,
        "text_exact_mismatches": int(mismatch["text"]),
        "text_newline_only_mismatches": text_newline_only,
        "drive_leading_apostrophe_escape_semantics": leading_apostrophe_escapes,
        "drive_control_character_import_side_effects": control_import_side_effects,
        "unresolved_substantive_mismatches": unresolved,
        "adjudication": {
            "title_type_coercions": "retain the local string title",
            "leading_apostrophe_escapes": "retain the local leading apostrophe",
            "control_character_import_side_effects": "remove DEL only in corpus v2 and retain the preceding character from the source CSV",
        },
        "remote_drive_object_byte_identity_verified": False,
    }
    checks = [
        {
            "name": "drive_live_sheet_keys_and_row_count_match_local_export",
            "passed": (
                lineage["row_count_match"]
                and lineage["local_canonical_content_bound_to_current_raw_export"]
                and lineage["song_id_exact_mismatches"] == 0
                and lineage["chunk_id_exact_mismatches"] == 0
            ),
        },
        {
            "name": "drive_live_substantive_mismatches_all_adjudicated",
            "passed": (
                lineage["artist_exact_mismatches"] == artist_normalisation_equivalent
                and lineage["title_exact_mismatches"]
                == title_normalisation_equivalent + title_type_coercions
                and lineage["text_exact_mismatches"]
                == text_newline_only + leading_apostrophe_escapes + control_import_side_effects
                and int(payload["substantive_review_rows"])
                == title_type_coercions + leading_apostrophe_escapes + control_import_side_effects
                and unresolved == 0
            ),
        },
    ]
    return lineage, checks


def clean_text_value(value: object) -> str:
    if pd.isna(value):
        return ""
    cleaned: list[str] = []
    for raw_line in str(value).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if CREDIT_LINE_RE.match(line):
            continue
        if STRUCTURE_LINE_RE.match(line):
            continue
        if COPYRIGHT_RE.search(line):
            continue
        line = SPEAKER_LABEL_RE.sub("", line)
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def title_normalise(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())


def title_exclusion_reason(value: object) -> str:
    title = "" if pd.isna(value) else str(value)
    flags: list[str] = []
    if re.search(r"live", title, flags=re.IGNORECASE):
        flags.append("live")
    if "伴奏" in title:
        flags.append("instrumental_zh")
    if re.search(r"instrumental", title, flags=re.IGNORECASE):
        flags.append("instrumental_en")
    return "+".join(flags) if flags else "other_matched_pattern"


def reconstruct(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"artist", "song_id", "song_title", "chunk_id", "text"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw chunk export is missing columns: {sorted(missing)}")
    raw_ordered = raw.dropna(subset=["artist", "text"]).copy()
    raw_ordered["_source_row"] = np.arange(len(raw_ordered))
    excluded_title = raw_ordered["song_title"].map(
        lambda value: False if pd.isna(value) else bool(TITLE_EXCLUDE_RE.search(str(value)))
    )
    after_title = raw_ordered.loc[~excluded_title].copy()
    after_text = after_title.copy()
    after_text["text"] = after_text["text"].map(clean_text_value)
    after_text = after_text.loc[after_text["text"].str.strip() != ""].copy()
    after_dedup = after_text.drop_duplicates(subset=["artist", "text"], keep="first").copy()
    return raw_ordered, after_title, after_text, after_dedup


def stage_counts(
    raw: pd.DataFrame,
    after_title: pd.DataFrame,
    after_text: pd.DataFrame,
    after_dedup: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    previous_rows: int | None = None
    previous_songs: int | None = None
    for stage, frame in (
        ("raw lyric chunks", raw),
        ("after title exclusions", after_title),
        ("after line cleaning and empty removal", after_text),
        ("after artist + exact-text deduplication", after_dedup),
    ):
        rows = len(frame)
        songs = frame["song_id"].nunique()
        records.append(
            {
                "stage": stage,
                "rows": rows,
                "songs": songs,
                "rows_removed_at_stage": 0 if previous_rows is None else previous_rows - rows,
                "songs_removed_at_stage": 0 if previous_songs is None else previous_songs - songs,
            }
        )
        previous_rows = rows
        previous_songs = songs
    return pd.DataFrame(records)


def duplicate_geometry(after_text: pd.DataFrame, after_dedup: pd.DataFrame) -> dict[str, int]:
    duplicate_rows = after_text[after_text.duplicated(["artist", "text"], keep=False)]
    groups = (
        duplicate_rows.groupby(["artist", "text"], sort=False)
        .agg(rows=("song_id", "size"), songs=("song_id", "nunique"))
        .reset_index(drop=True)
    )
    return {
        "duplicate_groups": len(groups),
        "rows_in_duplicate_groups": int(groups["rows"].sum()),
        "rows_removed_by_keep_first": len(after_text) - len(after_dedup),
        "groups_within_one_song_only": int((groups["songs"] == 1).sum()),
        "groups_crossing_songs": int((groups["songs"] > 1).sum()),
        "cross_song_groups_also_repeated_within_a_song": int(
            ((groups["songs"] > 1) & (groups["rows"] > groups["songs"])).sum()
        ),
        "songs_removed_entirely_by_deduplication": (
            after_text["song_id"].nunique() - after_dedup["song_id"].nunique()
        ),
    }


def classify_lost_songs(
    after_title: pd.DataFrame,
    after_text: pd.DataFrame,
    after_dedup: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    after_text = after_text.sort_values("_source_row")
    after_title = after_title.sort_values("_source_row")
    retained_ids = set(after_dedup["song_id"].astype(str))
    lost_ids = set(after_text["song_id"].astype(str)) - retained_ids

    records: dict[str, dict[str, Any]] = {}
    for song_id, group in after_text.groupby("song_id", sort=False):
        ordered = group.sort_values("_source_row")
        sequence = tuple(ordered["text"].astype(str))
        records[str(song_id)] = {
            "artist": str(ordered["artist"].iloc[0]),
            "title": str(ordered["song_title"].iloc[0]),
            "sequence": sequence,
            "counter": Counter(sequence),
        }
    raw_sequences = {
        str(song_id): tuple(group.sort_values("_source_row")["text"].fillna("").astype(str))
        for song_id, group in after_title.groupby("song_id", sort=False)
    }
    retained_by_artist: dict[str, list[str]] = defaultdict(list)
    for song_id in retained_ids:
        retained_by_artist[records[song_id]["artist"]].append(song_id)

    rows: list[dict[str, Any]] = []
    exact_raw_sequence = 0
    exact_title = 0
    exact_raw_and_title = 0
    for lost_id in sorted(lost_ids):
        lost = records[lost_id]
        candidates = retained_by_artist[lost["artist"]]
        exact = [candidate for candidate in candidates if records[candidate]["sequence"] == lost["sequence"]]
        same_multiset = [
            candidate
            for candidate in candidates
            if Counter(records[candidate]["sequence"]) == Counter(lost["sequence"])
        ]
        containers = [
            candidate
            for candidate in candidates
            if not (lost["counter"] - records[candidate]["counter"])
        ]
        owners: set[str] = set()
        all_chunks_covered = True
        for chunk in lost["counter"]:
            chunk_owners = {candidate for candidate in candidates if chunk in records[candidate]["counter"]}
            if not chunk_owners:
                all_chunks_covered = False
            owners.update(chunk_owners)

        if exact:
            category = "exact_sequence_duplicate_of_retained_song"
        elif same_multiset:
            category = "same_chunk_multiset_different_order"
        elif containers:
            category = "chunk_multiset_subset_of_one_retained_song"
        elif all_chunks_covered:
            category = "chunks_distributed_across_multiple_retained_songs"
        else:
            category = "unreconciled_other"

        related = set(exact or same_multiset or containers or list(owners))
        same_title = any(
            title_normalise(records[candidate]["title"]) == title_normalise(lost["title"])
            for candidate in related
        )
        same_raw = any(raw_sequences.get(candidate) == raw_sequences.get(lost_id) for candidate in exact)
        same_raw_title = any(
            raw_sequences.get(candidate) == raw_sequences.get(lost_id)
            and title_normalise(records[candidate]["title"]) == title_normalise(lost["title"])
            for candidate in exact
        )
        if exact:
            exact_title += int(same_title)
            exact_raw_sequence += int(same_raw)
            exact_raw_and_title += int(same_raw_title)
        rows.append(
            {
                "category": category,
                "cleaned_chunk_rows": len(lost["sequence"]),
                "candidate_retained_song_count": len(related),
                "title_exact_match_to_related_retained_song": same_title,
            }
        )

    detail = pd.DataFrame(rows)
    aggregate = (
        detail.groupby("category", as_index=False)
        .agg(
            songs=("category", "size"),
            cleaned_chunk_rows=("cleaned_chunk_rows", "sum"),
            songs_with_exact_title_match=("title_exact_match_to_related_retained_song", "sum"),
            median_related_retained_songs=("candidate_retained_song_count", "median"),
        )
        .sort_values(["songs", "category"], ascending=[False, True])
    )
    diagnostics = {
        "lost_songs": len(lost_ids),
        "exact_cleaned_sequence_lost_songs": int((detail["category"] == "exact_sequence_duplicate_of_retained_song").sum()),
        "exact_cleaned_sequence_and_title_lost_songs": exact_title,
        "exact_raw_sequence_lost_songs": exact_raw_sequence,
        "exact_raw_sequence_and_title_lost_songs": exact_raw_and_title,
        "conservative_manual_review_queue": len(lost_ids) - exact_title,
    }
    return aggregate, diagnostics


def written_rhyme_aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    transition_count = 0
    continuation_count = 0
    line_keys_by_song: dict[str, list[str]] = defaultdict(list)
    songs_with_lines: set[str] = set()
    for row in frame.sort_values("_source_row").itertuples(index=False):
        song_id = str(row.song_id)
        valid_rows: list[tuple[int, str]] = []
        for line_index, raw_line in enumerate(str(row.text).splitlines(), start=1):
            line = display_normalise(raw_line)
            if not line or is_header_line(line) or not HAN_RE.search(line):
                continue
            if terminal_content_kind(line) != "han":
                continue
            duplicate_key = duplicate_normalise(line)
            ending = written_ending_features(line)
            if not duplicate_key or ending is None:
                continue
            family = ending["final_family"]
            valid_rows.append((line_index, family))
            family_counts[family] += 1
            line_keys_by_song[song_id].append(duplicate_key)
            songs_with_lines.add(song_id)
        for previous, current in zip(valid_rows, valid_rows[1:]):
            if current[0] != previous[0] + 1:
                continue
            transition_count += 1
            continuation_count += int(current[1] == previous[1])
    repeat_occurrences = sum(len(values) - len(set(values)) for values in line_keys_by_song.values())
    return {
        "chunk_rows": len(frame),
        "songs_with_strict_han_ending_lines": len(songs_with_lines),
        "strict_han_ending_line_occurrences": sum(family_counts.values()),
        "within_song_repeat_line_occurrences": repeat_occurrences,
        "adjacent_transition_events": transition_count,
        "adjacent_same_family_events": continuation_count,
        "written_rhyme_switch_rate": 1.0 - continuation_count / transition_count,
        "family_counts": dict(family_counts),
    }


def written_rhyme_sensitivity(after_text: pd.DataFrame, after_dedup: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    full = written_rhyme_aggregate(after_text)
    dedup = written_rhyme_aggregate(after_dedup)
    families = sorted(set(full["family_counts"]) | set(dedup["family_counts"]))
    full_total = int(full["strict_han_ending_line_occurrences"])
    dedup_total = int(dedup["strict_han_ending_line_occurrences"])
    total_variation = 0.5 * sum(
        abs(
            full["family_counts"].get(family, 0) / full_total
            - dedup["family_counts"].get(family, 0) / dedup_total
        )
        for family in families
    )
    public_full = {key: value for key, value in full.items() if key != "family_counts"}
    public_dedup = {key: value for key, value in dedup.items() if key != "family_counts"}
    table = pd.DataFrame(
        [
            {"corpus": "all cleaned chunks before artist-text deduplication", **public_full},
            {"corpus": "frozen chunks after artist-text deduplication", **public_dedup},
        ]
    )
    delta = {
        "strict_han_lines_removed_by_chunk_dedup": full_total - dedup_total,
        "adjacent_transitions_removed_by_chunk_dedup": (
            int(full["adjacent_transition_events"]) - int(dedup["adjacent_transition_events"])
        ),
        "global_family_distribution_total_variation": total_variation,
        "switch_rate_absolute_difference": abs(
            float(full["written_rhyme_switch_rate"]) - float(dedup["written_rhyme_switch_rate"])
        ),
    }
    return table, delta


def build_task_frames(
    after_text: pd.DataFrame,
    registry: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    membership: pd.DataFrame,
    clean_sidecar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    four_hash_keys = [
        "song_id",
        "chunk_id",
        "canonical_lyric_text_sha256",
        "analysis_text_sha256",
    ]
    current_membership = membership.loc[membership["included_in_primary_centroid"].map(bool_field)].copy()
    current = current_membership.merge(
        clean_sidecar,
        on=four_hash_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_sidecar"),
    )
    if len(current) != len(current_membership):
        raise RuntimeError("Current membership did not join the clean sidecar one-to-one")
    if not current["analysis_text_status"].eq("eligible_clean_text").all():
        raise RuntimeError("Current comparison frame joined non-eligible analysis text")
    current_min = current[
        [
            "artist_label_id",
            "source_artist_label",
            "song_id",
            "chunk_id",
            "canonical_lyric_text_sha256",
            "analysis_text_sha256",
            "analysis_text",
        ]
    ].copy()

    removed = after_text.loc[after_text.duplicated(["artist", "text"], keep="first")].copy()
    registry_columns = [
        "song_id",
        "canonical_artist",
        "canonical_song_title",
        "analysis_eligible",
        "artist_title_comparison_eligible",
    ]
    added = removed.merge(
        registry[registry_columns],
        on="song_id",
        how="inner",
        validate="many_to_one",
    )
    joined_removed_rows = len(added)
    added = added.loc[
        added["analysis_eligible"].map(bool_field)
        & added["artist_title_comparison_eligible"].map(bool_field)
    ].copy()
    comparison_eligible_rows = len(added)
    cleaned_values = [
        clean_initial_headers(text, title)[0]
        for text, title in zip(added["text"].astype(str), added["canonical_song_title"].astype(str))
    ]
    added["analysis_text"] = cleaned_values
    added = added.loc[added["analysis_text"].str.strip() != ""].copy()
    nonempty_added_rows = len(added)
    added["canonical_lyric_text_sha256"] = added["text"].astype(str).map(sha256_text)
    added["analysis_text_sha256"] = added["analysis_text"].astype(str).map(sha256_text)
    added["source_artist_label"] = added["canonical_artist"].astype(str)
    added["artist_label_id"] = added["source_artist_label"].map(stable_label_id)
    added_min = added[
        [
            "artist_label_id",
            "source_artist_label",
            "song_id",
            "chunk_id",
            "canonical_lyric_text_sha256",
            "analysis_text_sha256",
            "analysis_text",
        ]
    ].copy()

    combined = pd.concat([current_min, added_min], ignore_index=True)
    duplicate_keys = combined.duplicated(["artist_label_id", "song_id", "chunk_id"]).sum()
    if duplicate_keys:
        raise RuntimeError(f"Counterfactual task frame has {duplicate_keys} duplicate label/song/chunk keys")

    def shared_hashes(frame: pd.DataFrame) -> set[str]:
        label_counts = frame.groupby("analysis_text_sha256")["source_artist_label"].nunique()
        return set(label_counts[label_counts > 1].index.astype(str))

    current_shared = shared_hashes(current_min)
    counterfactual_shared = shared_hashes(combined)
    eligible_ids = set(
        graph_nodes.loc[graph_nodes["graph_node_eligible"].map(bool_field), "artist_label_id"].astype(str)
    )
    baseline = current_min.loc[
        current_min["artist_label_id"].isin(eligible_ids)
        & ~current_min["analysis_text_sha256"].isin(current_shared)
    ].copy()
    counterfactual = combined.loc[
        combined["artist_label_id"].isin(eligible_ids)
        & ~combined["analysis_text_sha256"].isin(counterfactual_shared)
    ].copy()

    added_in_eligible_labels = int(added_min["artist_label_id"].isin(eligible_ids).sum())
    added_in_task = len(counterfactual) - len(baseline)
    diagnostics = {
        "removed_artist_text_rows": len(removed),
        "removed_rows_joining_canonical_registry": joined_removed_rows,
        "removed_rows_comparison_eligible": comparison_eligible_rows,
        "removed_rows_nonempty_after_current_header_cleaning": nonempty_added_rows,
        "added_rows_in_fixed_204_label_universe_before_shared_text_exclusion": added_in_eligible_labels,
        "added_rows_shared_text_excluded": added_in_eligible_labels - added_in_task,
        "added_rows_entering_task_counterfactual": added_in_task,
        "current_all_label_rows": len(current_min),
        "counterfactual_all_label_rows": len(combined),
        "current_shared_text_hashes": len(current_shared),
        "counterfactual_shared_text_hashes": len(counterfactual_shared),
        "fixed_graph_eligible_labels": len(eligible_ids),
    }
    return baseline, counterfactual, diagnostics


def task_rhyme_statistics(frame: pd.DataFrame) -> tuple[dict[str, Any], Counter[str]]:
    family_counts: Counter[str] = Counter()
    line_keys_by_label_song: dict[tuple[str, str], list[str]] = defaultdict(list)
    input_songs = set(frame["song_id"].astype(str))
    songs_with_lines: set[str] = set()
    songs_with_events: set[str] = set()
    nonempty_chunks = 0
    transitions = 0
    switches = 0

    for row in frame.itertuples(index=False):
        valid_rows: list[tuple[int, str]] = []
        for line_index, raw_line in enumerate(str(row.analysis_text).splitlines(), start=1):
            line = display_normalise(raw_line)
            if not line or is_header_line(line) or not HAN_RE.search(line):
                continue
            if terminal_content_kind(line) != "han":
                continue
            duplicate_key = duplicate_normalise(line)
            ending = written_ending_features(line)
            if not duplicate_key or ending is None:
                continue
            family = ending["final_family"]
            valid_rows.append((line_index, family))
            family_counts[family] += 1
            line_keys_by_label_song[(str(row.artist_label_id), str(row.song_id))].append(duplicate_key)
            songs_with_lines.add(str(row.song_id))
        if valid_rows:
            nonempty_chunks += 1
        for previous, current in zip(valid_rows, valid_rows[1:]):
            if current[0] != previous[0] + 1:
                continue
            transitions += 1
            switches += int(current[1] != previous[1])
            songs_with_events.add(str(row.song_id))

    repeat_groups = 0
    repeat_group_occurrences = 0
    repeat_excess = 0
    for values in line_keys_by_label_song.values():
        counts = Counter(values)
        for count in counts.values():
            if count > 1:
                repeat_groups += 1
                repeat_group_occurrences += count
                repeat_excess += count - 1

    chunk_groups = frame.groupby(["source_artist_label", "analysis_text_sha256"]).size()
    duplicate_chunk_groups = chunk_groups[chunk_groups > 1]
    statistics = {
        "input_chunks": len(frame),
        "nonempty_line_sequences": nonempty_chunks,
        "input_songs": len(input_songs),
        "songs_with_at_least_one_strict_han_line": len(songs_with_lines),
        "songs_with_at_least_one_adjacent_transition": len(songs_with_events),
        "strict_han_ending_line_occurrences": sum(family_counts.values()),
        "adjacent_transitions_before_leakage_filter": transitions,
        "switch_events": switches,
        "switch_rate": switches / transitions,
        "repeat_groups": repeat_groups,
        "repeat_group_occurrences_including_first": repeat_group_occurrences,
        "repeat_excess_after_first": repeat_excess,
        "exact_cleaned_duplicate_chunk_groups": len(duplicate_chunk_groups),
        "exact_cleaned_duplicate_chunk_occurrences": int(duplicate_chunk_groups.sum()),
        "exact_cleaned_duplicate_chunk_excess": int((duplicate_chunk_groups - 1).sum()),
    }
    return statistics, family_counts


def task_aligned_sensitivity(
    baseline: pd.DataFrame,
    counterfactual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline_stats, baseline_families = task_rhyme_statistics(baseline)
    counter_stats, counter_families = task_rhyme_statistics(counterfactual)
    table = pd.DataFrame(
        [
            {"population": "released_frozen_task", **baseline_stats},
            {"population": "pre_snapshot_duplicate_chunk_counterfactual", **counter_stats},
        ]
    )
    family_rows: list[dict[str, Any]] = []
    baseline_total = sum(baseline_families.values())
    counter_total = sum(counter_families.values())
    for family in sorted(set(baseline_families) | set(counter_families)):
        baseline_share = baseline_families[family] / baseline_total
        counter_share = counter_families[family] / counter_total
        family_rows.append(
            {
                "written_rhyme_family": family,
                "released_count": baseline_families[family],
                "released_share": baseline_share,
                "counterfactual_count": counter_families[family],
                "counterfactual_share": counter_share,
                "counterfactual_minus_released_share": counter_share - baseline_share,
            }
        )
    family_table = pd.DataFrame(family_rows)
    total_variation = 0.5 * float(family_table["counterfactual_minus_released_share"].abs().sum())
    largest = family_table.iloc[family_table["counterfactual_minus_released_share"].abs().argmax()]
    delta = {
        "strict_han_lines_restored": (
            counter_stats["strict_han_ending_line_occurrences"]
            - baseline_stats["strict_han_ending_line_occurrences"]
        ),
        "adjacent_transitions_restored": (
            counter_stats["adjacent_transitions_before_leakage_filter"]
            - baseline_stats["adjacent_transitions_before_leakage_filter"]
        ),
        "repeat_excess_restored": counter_stats["repeat_excess_after_first"] - baseline_stats["repeat_excess_after_first"],
        "switch_rate_counterfactual_minus_released": counter_stats["switch_rate"] - baseline_stats["switch_rate"],
        "family_distribution_total_variation": total_variation,
        "largest_absolute_family_share_change_family": str(largest["written_rhyme_family"]),
        "largest_absolute_family_share_change": float(largest["counterfactual_minus_released_share"]),
        "predictive_metrics_retrained": False,
        "permitted_claim": "aggregate family shares and switch rate are insensitive to the restored-chunk counterfactual",
        "withheld_claim": "top-k, MRR, calibration, abstention, and paired model differences are unchanged",
    }
    return table, family_table, delta


def build(args: argparse.Namespace) -> None:
    raw_path = args.raw_chunks.resolve()
    frozen_path = args.frozen_snapshot.resolve()
    metadata_path = args.metadata_ndjson.resolve()
    corpus_manifest_path = args.corpus_manifest.resolve()
    registry_path = args.registry.resolve()
    graph_nodes_path = args.graph_nodes.resolve()
    membership_path = args.membership.resolve()
    clean_sidecar_path = args.clean_sidecar.resolve()
    drive_comparison_path = args.drive_comparison_summary.resolve()
    out = args.output_dir.resolve()
    if args.generated_at_utc:
        generated_at_utc = args.generated_at_utc
    elif args.reuse_generated_at:
        existing_summary = out / "analysis_summary.json"
        if not existing_summary.is_file():
            raise FileNotFoundError(f"Cannot reuse a missing artifact timestamp: {existing_summary}")
        generated_at_utc = json.loads(existing_summary.read_text(encoding="utf-8"))["generated_at_utc"]
    else:
        generated_at_utc = utc_now()
    for path in (
        raw_path,
        frozen_path,
        metadata_path,
        corpus_manifest_path,
        registry_path,
        graph_nodes_path,
        membership_path,
        clean_sidecar_path,
        drive_comparison_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = pd.read_csv(raw_path)
    frozen = pd.read_csv(frozen_path)
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    graph_nodes = pd.read_csv(graph_nodes_path, dtype=str, keep_default_na=False)
    membership = pd.read_csv(membership_path, dtype=str, keep_default_na=False)
    clean_sidecar = pd.read_csv(clean_sidecar_path, dtype=str, keep_default_na=False)
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    drive_comparison = json.loads(drive_comparison_path.read_text(encoding="utf-8"))
    metadata_rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    metadata_ids = {
        str(row.get("song_id", "")).strip()
        for row in metadata_rows
        if str(row.get("song_id", "")).strip()
    }

    raw_ordered, after_title, after_text, after_dedup = reconstruct(raw)
    drive_lineage, drive_checks = summarise_drive_lineage(drive_comparison, raw)
    columns = ["artist", "song_id", "song_title", "chunk_id", "text"]
    frozen_exact = (
        after_dedup[columns].fillna("").astype(str).reset_index(drop=True).equals(
            frozen[columns].fillna("").astype(str).reset_index(drop=True)
        )
    )
    stages = stage_counts(raw_ordered, after_title, after_text, after_dedup)
    geometry = duplicate_geometry(after_text, after_dedup)
    lost_classification, lost_diagnostics = classify_lost_songs(after_title, after_text, after_dedup)
    rhyme_table, rhyme_delta = written_rhyme_sensitivity(after_text, after_dedup)
    baseline_task, counterfactual_task, task_frame_diagnostics = build_task_frames(
        after_text,
        registry,
        graph_nodes,
        membership,
        clean_sidecar,
    )
    task_rhyme_table, task_family_table, task_rhyme_delta = task_aligned_sensitivity(
        baseline_task,
        counterfactual_task,
    )

    raw_ids = set(raw_ordered["song_id"].astype(str))
    final_ids = set(after_dedup["song_id"].astype(str))
    source_reconciliation = {
        "nonblank_source_metadata_song_ids": len(metadata_ids),
        "raw_chunk_export_song_ids": len(raw_ids),
        "metadata_ids_absent_from_raw_chunk_export": len(metadata_ids - raw_ids),
        "raw_export_ids_removed_by_cleaning": len(raw_ids - final_ids),
        "frozen_clean_song_ids": len(final_ids),
        "frozen_ids_absent_from_metadata": len(final_ids - metadata_ids),
        "reconciled_source_only_metadata_rows": len(metadata_ids - raw_ids) + len(raw_ids - final_ids),
        "manifest_source_only_metadata_rows": int(corpus_manifest["counts"]["source_only_metadata_rows"]),
    }
    excluded_title_rows = raw_ordered.loc[~raw_ordered.index.isin(after_title.index)].copy()
    excluded_title_rows["reason"] = excluded_title_rows["song_title"].map(title_exclusion_reason)
    title_exclusions = (
        excluded_title_rows.groupby("reason", as_index=False)
        .agg(rows=("song_id", "size"), songs=("song_id", "nunique"))
        .sort_values(["songs", "reason"], ascending=[False, True])
    )

    released_task_row = task_rhyme_table.iloc[0]
    counterfactual_task_row = task_rhyme_table.iloc[1]
    legacy_pre_dedup_row = rhyme_table.iloc[0]
    legacy_frozen_row = rhyme_table.iloc[1]
    checks = drive_checks + [
        {"name": "frozen_snapshot_content_exactly_reconstructed", "passed": frozen_exact},
        {
            "name": "source_only_metadata_count_reconciles",
            "passed": source_reconciliation["reconciled_source_only_metadata_rows"]
            == source_reconciliation["manifest_source_only_metadata_rows"],
        },
        {
            "name": "frozen_ids_all_join_source_metadata",
            "passed": source_reconciliation["frozen_ids_absent_from_metadata"] == 0,
        },
        {
            "name": "all_dedup_lost_songs_reconciled_to_retained_exact_chunks",
            "passed": "unreconciled_other" not in set(lost_classification["category"]),
        },
        {
            "name": "expected_reconstructed_counts",
            "passed": len(after_dedup) == 22132 and after_dedup["song_id"].nunique() == 7214,
        },
        {
            "name": "task_aligned_released_population_reconstructed",
            "passed": (
                len(baseline_task) == 15760
                and baseline_task["song_id"].nunique() == 5619
                and int(released_task_row["songs_with_at_least_one_strict_han_line"]) == 5452
                and int(released_task_row["songs_with_at_least_one_adjacent_transition"]) == 5347
                and int(released_task_row["strict_han_ending_line_occurrences"]) == 283806
                and int(released_task_row["adjacent_transitions_before_leakage_filter"]) == 238881
                and int(released_task_row["switch_events"]) == 119436
                and int(released_task_row["repeat_groups"]) == 27642
                and int(released_task_row["repeat_group_occurrences_including_first"]) == 79794
                and int(released_task_row["repeat_excess_after_first"]) == 52152
                and int(released_task_row["repeat_group_occurrences_including_first"])
                - int(released_task_row["repeat_groups"])
                == int(released_task_row["repeat_excess_after_first"])
                and bool(np.isclose(
                    float(released_task_row["switch_rate"]),
                    int(released_task_row["switch_events"])
                    / int(released_task_row["adjacent_transitions_before_leakage_filter"]),
                ))
            ),
        },
        {
            "name": "task_aligned_counterfactual_expected_counts",
            "passed": (
                len(counterfactual_task) == 17715
                and counterfactual_task["song_id"].nunique() == 5621
                and int(counterfactual_task_row["songs_with_at_least_one_strict_han_line"]) == 5459
                and int(counterfactual_task_row["songs_with_at_least_one_adjacent_transition"]) == 5353
                and int(counterfactual_task_row["strict_han_ending_line_occurrences"]) == 290839
                and int(counterfactual_task_row["adjacent_transitions_before_leakage_filter"]) == 243819
                and int(counterfactual_task_row["switch_events"]) == 121566
                and int(counterfactual_task_row["repeat_groups"]) == 30300
                and int(counterfactual_task_row["repeat_group_occurrences_including_first"]) == 88924
                and int(counterfactual_task_row["repeat_excess_after_first"]) == 58624
                and int(counterfactual_task_row["repeat_group_occurrences_including_first"])
                - int(counterfactual_task_row["repeat_groups"])
                == int(counterfactual_task_row["repeat_excess_after_first"])
                and bool(np.isclose(
                    float(counterfactual_task_row["switch_rate"]),
                    int(counterfactual_task_row["switch_events"])
                    / int(counterfactual_task_row["adjacent_transitions_before_leakage_filter"]),
                ))
            ),
        },
        {
            "name": "task_aligned_family_totals_and_headline_deltas_reconstructed",
            "passed": (
                int(task_family_table["released_count"].sum()) == 283806
                and int(task_family_table["counterfactual_count"].sum()) == 290839
                and task_rhyme_delta["strict_han_lines_restored"] == 7033
                and task_rhyme_delta["adjacent_transitions_restored"] == 4938
                and task_rhyme_delta["repeat_excess_restored"] == 6472
                and np.isclose(task_rhyme_delta["family_distribution_total_variation"], 0.001789031820619097)
                and np.isclose(
                    abs(task_rhyme_delta["switch_rate_counterfactual_minus_released"]),
                    0.001389994130025182,
                )
                and task_rhyme_delta["predictive_metrics_retrained"] is False
            ),
        },
        {
            "name": "legacy_full_source_written_ending_sensitivity_reconstructed",
            "passed": (
                int(legacy_pre_dedup_row["strict_han_ending_line_occurrences"]) == 422656
                and int(legacy_frozen_row["strict_han_ending_line_occurrences"]) == 406033
                and int(legacy_pre_dedup_row["adjacent_transition_events"]) == 355530
                and int(legacy_frozen_row["adjacent_transition_events"]) == 342418
                and rhyme_delta["strict_han_lines_removed_by_chunk_dedup"] == 16623
                and rhyme_delta["adjacent_transitions_removed_by_chunk_dedup"] == 13112
                and np.isclose(rhyme_delta["global_family_distribution_total_variation"], 0.0017671763323308868)
                and bool(np.isclose(rhyme_delta["switch_rate_absolute_difference"], 0.0017874646921119952))
            ),
        },
    ]
    status = "pass_with_release_action" if all(item["passed"] for item in checks) else "fail"
    software_paths = [
        Path(__file__).resolve(),
        ROOT / "src" / "build_canonical_lyric_text_sidecar_v1.py",
        ROOT / "src" / "build_chinese_rap_written_rhyme_v1.py",
    ]
    software_fingerprints = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
        for path in software_paths
    ]

    summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": generated_at_utc,
        "status": status,
        "privacy": "aggregate only; no lyric text, labels, titles, song/chunk identifiers, or row-level hashes",
        "input_fingerprints": {
            "raw_chunks": {"file": raw_path.name, "sha256": sha256_file(raw_path)},
            "frozen_snapshot": {"file": frozen_path.name, "sha256": sha256_file(frozen_path)},
            "source_metadata": {"file": metadata_path.name, "sha256": sha256_file(metadata_path)},
            "corpus_manifest": {"file": corpus_manifest_path.name, "sha256": sha256_file(corpus_manifest_path)},
            "canonical_registry": {"file": registry_path.name, "sha256": sha256_file(registry_path)},
            "graph_nodes": {"file": graph_nodes_path.name, "sha256": sha256_file(graph_nodes_path)},
            "private_membership": {"file": membership_path.name, "sha256": sha256_file(membership_path)},
            "private_clean_sidecar": {"file": clean_sidecar_path.name, "sha256": sha256_file(clean_sidecar_path)},
            "drive_live_comparison_summary": {
                "file": drive_comparison_path.name,
                "sha256": sha256_file(drive_comparison_path),
            },
        },
        "software_fingerprints": software_fingerprints,
        "drive_lineage": drive_lineage,
        "frozen_snapshot_content_exactly_reconstructed": frozen_exact,
        "source_metadata_reconciliation": source_reconciliation,
        "duplicate_geometry": geometry,
        "lost_song_diagnostics": lost_diagnostics,
        "written_rhyme_sensitivity": rhyme_delta,
        "task_aligned_frame_diagnostics": task_frame_diagnostics,
        "task_aligned_written_rhyme_sensitivity": task_rhyme_delta,
        "interpretation": {
            "high_confidence_ingestion_duplicate_rule": "same source-credit label + same normalized title + exact cleaned chunk sequence",
            "high_confidence_ingestion_duplicate_records": lost_diagnostics[
                "exact_cleaned_sequence_and_title_lost_songs"
            ],
            "manual_review_queue_records": lost_diagnostics["conservative_manual_review_queue"],
            "primary_action": "Preserve song identity and within-song sequence; control exact-text duplicate components by grouped splitting or weighting instead of deleting source rows.",
            "rhyme_claim_action": "Qualify repeat-retention wording and run duplicate-aware sequence sensitivity before submission.",
        },
        "checks": checks,
    }

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage_counts.csv", stages)
    write_csv(out / "title_exclusion_counts.csv", title_exclusions)
    write_csv(out / "lost_song_classification.csv", lost_classification)
    write_csv(out / "written_rhyme_sensitivity.csv", rhyme_table)
    write_csv(out / "task_aligned_written_rhyme_sensitivity.csv", task_rhyme_table)
    write_csv(out / "task_aligned_family_distribution_sensitivity.csv", task_family_table)
    write_json(out / "analysis_summary.json", summary)
    write_json(
        out / "validation.json",
        {
            "artifact_id": ARTIFACT_ID,
            "generated_at_utc": summary["generated_at_utc"],
            "status": status,
            "checks": checks,
            "privacy": summary["privacy"],
        },
    )
    method = """# Corpus reconciliation and duplicate-control method

This aggregate-only audit first compares the live native Google Sheet with the local raw chunk export at row level. Song and chunk keys match exactly. The remaining substantive cell differences are fully adjudicated as native-sheet title type coercion, leading-apostrophe escape semantics, or control-character import side effects. The local string values are retained; corpus v2 removes the two DEL controls without deleting their preceding characters.

The audit then replays the historical raw-to-frozen cleaner in its original order: title exclusions, configured line cleaning, empty-chunk removal, and first-occurrence retention for exact `(source-credit label, cleaned chunk text)` matches. It verifies the reconstructed rows against the frozen snapshot and reconciles source metadata IDs with raw and frozen song IDs.

The duplicate audit then asks why songs disappeared. A lost song is classified as an exact cleaned chunk-sequence duplicate, a same-multiset reorder, a subset of one retained song, or a set of chunks distributed across retained songs. The public files contain counts only. They contain no lyrics, labels, titles, identifiers, embeddings, or row-level hashes.

The broad written-rhyme sensitivity applies strict terminal-Han, pypinyin-final-family, and original-adjacency rules to the exact legacy post-line-cleaning text before and after artist-level chunk deduplication. It intentionally precedes the later canonical leading-header pass so the contrast isolates the historical deduplication stage. A second, task-aligned counterfactual reconnects only restored chunks that join the canonical registry, pass current eligibility and leading-header rules, belong to the fixed 204-label universe, and survive a recomputed all-label shared-text exclusion. It exactly reconstructs the released 283,806 lines, 238,881 pre-leakage transitions, and 52,152 repeat-excess count before comparing the restored population. This diagnoses corpus composition; it is not a replacement predictive benchmark because splitting, leakage filtering, fitting, and evaluation are not rerun.

The repair rule is to preserve song identity and original chunk order, collapse only high-confidence duplicate source records under a declared song-level rule, and represent remaining exact-text sharing with duplicate-component IDs. Retrieval splits must keep components together, artist profiles must weight components, and rhyme models must preserve within-song repetition while reporting cross-song duplicate sensitivity.
"""
    (out / "METHOD.md").write_text(method, encoding="utf-8", newline="")
    readme = f"""# Corpus reconciliation v1

The live Drive sheet and the local raw export contain the same **{drive_lineage['live_sheet_rows']:,} row-aligned records** with exact song and chunk keys. All **{drive_comparison['substantive_review_rows']:,} substantive cell differences are adjudicated**: **{drive_lineage['drive_native_sheet_title_type_coercions']:,}** title type coercions, **{drive_lineage['drive_leading_apostrophe_escape_semantics']:,}** leading-apostrophe escapes, and **{drive_lineage['drive_control_character_import_side_effects']:,}** control-character import side effects. No substantive mismatch remains unresolved.

The historical cleaner is exactly reproducible, but its artist-level exact-text deduplication removed **{geometry['rows_removed_by_keep_first']:,} chunks** and erased **{geometry['songs_removed_entirely_by_deduplication']:,} songs** before downstream modeling. Of those lost songs, **{lost_diagnostics['exact_cleaned_sequence_and_title_lost_songs']:,}** satisfy a conservative high-confidence duplicate-record rule; **{lost_diagnostics['conservative_manual_review_queue']:,}** require review rather than automatic deletion.

At full cleaned-source scope, the global written-ending distribution changes little, but the frozen input omits **{rhyme_delta['strict_han_lines_removed_by_chunk_dedup']:,} strict-Han-ending line occurrences** and **{rhyme_delta['adjacent_transitions_removed_by_chunk_dedup']:,} adjacent transitions** present before chunk deduplication. In the exact released 204-label task frame, restoring eligible chunks adds **{task_rhyme_delta['strict_han_lines_restored']:,} lines**, **{task_rhyme_delta['adjacent_transitions_restored']:,} transitions**, and **{task_rhyme_delta['repeat_excess_restored']:,} additional repeat occurrences beyond first occurrences**; family-distribution total variation is **{task_rhyme_delta['family_distribution_total_variation']:.6f}**. Therefore the released repeat-retention wording is true only inside surviving chunks. Aggregate sensitivity is complete, while predictive metrics remain untested on the repaired population.

Status: **{status}**. Structural reconstruction and aggregate written-ending sensitivity pass; duplicate-aware corpus repair and predictive reruns remain release actions.
"""
    (out / "README.md").write_text(readme, encoding="utf-8", newline="")

    output_files = sorted(path for path in out.iterdir() if path.name != "manifest.json")
    write_json(
        out / "manifest.json",
        {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": summary["generated_at_utc"],
            "status": status,
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in output_files
            ],
            "software": software_fingerprints,
        },
    )

    if not all(item["passed"] for item in checks):
        failed = [item["name"] for item in checks if not item["passed"]]
        raise RuntimeError(f"Corpus reconciliation failed: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-chunks", type=Path, required=True)
    parser.add_argument("--frozen-snapshot", type=Path, required=True)
    parser.add_argument("--metadata-ndjson", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--graph-nodes", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--clean-sidecar", type=Path, required=True)
    parser.add_argument("--drive-comparison-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    timestamp = parser.add_mutually_exclusive_group()
    timestamp.add_argument("--generated-at-utc", help="Use this ISO-8601 timestamp in generated manifests")
    timestamp.add_argument(
        "--reuse-generated-at",
        action="store_true",
        help="Reuse generated_at_utc from the existing analysis_summary.json for an idempotent rebuild",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
