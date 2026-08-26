#!/usr/bin/env python3
"""Build Chinese written-rhyme fingerprints and a held-out recommender.

The unit of analysis is the final written Han syllable of a lyric line.  The
artifact must not be described as performed rhyme, flow, cadence, beat
alignment, or a verified artist identity analysis.  Source-credit labels are
kept under the attribution boundary inherited from the repertoire graph.

Run from the workspace root with the project semantic environment::

    & "work/semantic-ml-venv/Scripts/python.exe" work/build_chinese_rap_written_rhyme_v1.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pypinyin
import sklearn
from pypinyin import Style, lazy_pinyin
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier


ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "outputs" / "chinese-rap-written-rhyme-v1"
PRIVATE_DIR = ROOT / "work" / "private-chinese-rap-written-rhyme-v1"

NODE_FILE = ROOT / "outputs" / "chinese-rap-lyrical-repertoire-graph-v2" / "artist_repertoire_nodes.csv"
MEMBERSHIP_FILE = ROOT / "work" / "private-chinese-rap-lyrical-repertoire-graph-v2" / "artist_chunk_membership_v2.csv"
CLEAN_TEXT_FILE = ROOT / "work" / "private-canonical-lyric-text-sidecar-v1" / "cleaned_analysis_chunks_v1.csv"
MUCHIN_ROOT = ROOT / "work" / "external_sources" / "MuChin-V2-6066"
MUCHIN1000_DIR = MUCHIN_ROOT / "Datasets-for-MuChin-V2" / "MuChin-1000"
MUCHIN_LICENSE = MUCHIN_ROOT / "LICENSE"
MUCHIN_UI_FILE = MUCHIN_ROOT / "Code-for-MuChin-AP(AnnotationPlatform)" / "frontend" / "src" / "Component" / "Annotation" / "Step4" / "index.js"
MUCHIN_BACKEND_FILE = MUCHIN_ROOT / "Code-for-MuChin-AP(AnnotationPlatform)" / "backend_api" / "mapy" / "util" / "character_util.py"
MUCHIN_OVERLAP_FILE = MUCHIN_ROOT / "Datasets-for-MuChin-V2" / "muchin_5790_1000_overlap.jsonl"

ARTIFACT_ID = "chinese-rap-written-rhyme-v1"
VERSION = "1.1.0"
SEED = 20260825
NEAR_DUPLICATE_JACCARD = 0.80
BOOTSTRAP_REPLICATES = 2000
MIN_PER_LABEL_EVAL_EVENTS = 20
MIN_PUBLIC_CONTEXT_SUPPORT = 20
MAX_PUBLIC_CONTEXTS = 5000

HAN_RE = re.compile(r"[\u3400-\u9fff]")
KEEP_DUPLICATE_RE = re.compile(r"[\u3400-\u9fffA-Za-z0-9]+")
SECTION_RE = re.compile(r"^\([^\n]+\)$")
HEADER_RE = re.compile(
    r"^[\[\(（【<《]?(?:verse|hook|chorus|bridge|intro|outro|pre[- ]?chorus|"
    r"副歌|主歌|前奏|间奏|尾奏|歌名|歌词|作词|作曲|编曲|演唱|制作人)\s*\d*"
    r"[\]\)）】>》]?$",
    flags=re.IGNORECASE,
)


# Transparent, tone-free normalisation of strict pypinyin finals.  These are
# analytical written-ending classes, not an official phonological standard.
RHYME_FAMILIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("A", ("a", "ia", "ua"), "a / ia / ua"),
    ("O", ("o", "uo"), "o / uo"),
    ("E", ("e",), "e"),
    ("IE_VE", ("ie", "ve"), "ie / üe"),
    ("AI", ("ai", "uai"), "ai / uai"),
    ("EI", ("ei", "ui"), "ei / ui"),
    ("AO", ("ao", "iao"), "ao / iao"),
    ("OU", ("ou", "iu"), "ou / iu"),
    ("AN", ("an", "ian", "uan", "van"), "an / ian / uan / üan"),
    ("EN", ("en", "in", "un", "vn"), "en / in / un / ün"),
    ("ANG", ("ang", "iang", "uang"), "ang / iang / uang"),
    ("ENG", ("eng", "ing", "ueng"), "eng / ing / ueng"),
    ("ONG", ("ong", "iong"), "ong / iong"),
    ("I", ("i",), "i (pypinyin final; apical subtypes are not separated)"),
    ("U", ("u",), "u"),
    ("V", ("v",), "ü"),
    ("ER", ("er",), "er"),
)
FAMILY_ORDER = tuple(item[0] for item in RHYME_FAMILIES)
FINAL_TO_FAMILY = {final: family for family, finals, _ in RHYME_FAMILIES for final in finals}
FAMILY_DESCRIPTION = {family: description for family, _, description in RHYME_FAMILIES}


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


def stable_unit(value: str) -> float:
    integer = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
    return integer / float(16**16 - 1)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def clear_build_directory(path: Path) -> None:
    resolved = path.resolve()
    allowed = {PUBLIC_DIR.resolve(), PRIVATE_DIR.resolve()}
    if resolved not in allowed or ROOT.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing to clear unexpected build directory: {resolved}")
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def dependency_fingerprints() -> dict[str, Any]:
    modules = {"numpy": np, "scikit_learn": sklearn, "pypinyin": pypinyin}
    result: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable_sha256": sha256_file(Path(sys.executable)) if Path(sys.executable).is_file() else None,
    }
    for name, module in modules.items():
        module_path = Path(module.__file__).resolve()
        result[name] = {
            "version": getattr(module, "__version__", "unknown"),
            "module_file_name": module_path.name,
            "module_file_sha256": sha256_file(module_path),
        }
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_field(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def round_float(value: float | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def chunk_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def duplicate_normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(KEEP_DUPLICATE_RE.findall(value))


def display_normalise(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def is_header_line(value: str) -> bool:
    return bool(HEADER_RE.fullmatch(display_normalise(value)))


def terminal_content_kind(value: str) -> str:
    """Classify the final non-space/non-punctuation written character."""
    for character in reversed(unicodedata.normalize("NFKC", value)):
        category = unicodedata.category(character)
        if category.startswith("Z") or category.startswith("P"):
            continue
        if HAN_RE.fullmatch(character):
            return "han"
        if character.isascii() and character.isalpha():
            return "latin"
        if character.isdigit():
            return "digit"
        return "other_non_han"
    return "empty"


def canonical_final(value: str) -> str:
    value = re.sub(r"[1-5]$", "", value.lower().replace("ü", "v"))
    return {"uei": "ui", "iou": "iu", "uen": "un"}.get(value, value)


def written_ending_features(value: str) -> dict[str, Any] | None:
    if terminal_content_kind(value) != "han":
        return None
    han = "".join(HAN_RE.findall(value))
    if not han:
        return None
    finals_tone = lazy_pinyin(
        han,
        style=Style.FINALS_TONE3,
        strict=True,
        neutral_tone_with_five=True,
        errors=lambda item: [""] * len(item),
    )
    parsed: list[tuple[str, str]] = []
    for token in finals_tone:
        tone_match = re.search(r"([1-5])$", token)
        tone = tone_match.group(1) if tone_match else "5"
        final = canonical_final(token)
        family = FINAL_TO_FAMILY.get(final)
        if family:
            parsed.append((family, final + tone))
        else:
            parsed.append(("", final + tone))
    if not parsed or not parsed[-1][0]:
        return None
    last_family, last_toned = parsed[-1]
    last_final = re.sub(r"[1-5]$", "", last_toned)
    last_tone = re.search(r"([1-5])$", last_toned).group(1)  # type: ignore[union-attr]
    if len(parsed) >= 2 and parsed[-2][0]:
        penult_family, penult_toned = parsed[-2]
        penult_final = re.sub(r"[1-5]$", "", penult_toned)
        two_family = f"{penult_family}>{last_family}"
        two_final = f"{penult_final}>{last_final}"
    else:
        penult_family = ""
        penult_final = ""
        two_family = ""
        two_final = ""
    return {
        "han_character_count": len(han),
        "final_family": last_family,
        "raw_final": last_final,
        "tone": last_tone,
        "penultimate_family": penult_family,
        "penultimate_final": penult_final,
        "two_family_pattern": two_family,
        "two_final_pattern": two_final,
    }


def load_corpus() -> tuple[
    dict[tuple[str, str, str], list[dict[str, Any]]],
    dict[str, dict[str, str]],
    dict[str, set[str]],
    dict[str, int],
    dict[str, dict[str, int]],
]:
    required = (NODE_FILE, MEMBERSHIP_FILE, CLEAN_TEXT_FILE)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required upstream files: {missing}")

    node_rows = read_csv(NODE_FILE)
    eligible_nodes = {row["artist_label_id"]: row for row in node_rows if bool_field(row.get("graph_node_eligible"))}
    if len(eligible_nodes) != 204:
        raise RuntimeError(f"Expected 204 graph-eligible labels, found {len(eligible_nodes)}")

    memberships = [
        row
        for row in read_csv(MEMBERSHIP_FILE)
        if row["artist_label_id"] in eligible_nodes
        and bool_field(row.get("included_in_primary_centroid"))
        and bool_field(row.get("included_in_shared_text_exclusion_sensitivity"))
    ]
    membership_keys = {
        (row["song_id"], row["chunk_id"], row["canonical_lyric_text_sha256"], row["analysis_text_sha256"])
        for row in memberships
    }
    clean_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in read_csv(CLEAN_TEXT_FILE):
        key = (row["song_id"], row["chunk_id"], row["canonical_lyric_text_sha256"], row["analysis_text_sha256"])
        if key in membership_keys:
            if row.get("analysis_text_status") != "eligible_clean_text":
                raise RuntimeError(f"Membership joined non-eligible clean text at {key[:2]}")
            if key in clean_by_key:
                raise RuntimeError(f"Non-unique clean-text join key: {key[:2]}")
            clean_by_key[key] = row
    if len(clean_by_key) != len(membership_keys):
        raise RuntimeError(f"Clean-text join mismatch: {len(clean_by_key)} joined vs {len(membership_keys)} required")

    grouped_chunks: dict[tuple[str, str, str], tuple[str, str]] = {}
    song_to_labels: dict[str, set[str]] = defaultdict(set)
    for row in memberships:
        key = (row["song_id"], row["chunk_id"], row["canonical_lyric_text_sha256"], row["analysis_text_sha256"])
        clean = clean_by_key[key]
        sequence_key = (row["artist_label_id"], row["song_id"], row["chunk_id"])
        if sequence_key in grouped_chunks:
            raise RuntimeError(f"Non-unique source-label/song/chunk sequence: {sequence_key}")
        grouped_chunks[sequence_key] = (clean["analysis_text"], row["analysis_text_sha256"])
        song_to_labels[row["song_id"]].add(row["artist_label_id"])

    # Keep repeated lines and original line indices. Excluded lines remain gaps,
    # so later event construction can never bridge across them.
    sequences: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    exclusion_counts: Counter[str] = Counter()
    terminal_exclusions_by_song: dict[str, Counter[str]] = defaultdict(Counter)
    for (label_id, song_id, chunk_id), (text, analysis_hash) in sorted(
        grouped_chunks.items(), key=lambda item: (item[0][0], item[0][1], chunk_sort_key(item[0][2]))
    ):
        sequence: list[dict[str, Any]] = []
        for chunk_line_index, raw_line in enumerate(text.splitlines(), start=1):
            line = display_normalise(raw_line)
            if not line:
                exclusion_counts["empty_line"] += 1
                continue
            if is_header_line(line):
                exclusion_counts["section_or_credit_header"] += 1
                continue
            if not HAN_RE.search(line):
                exclusion_counts["no_han_character"] += 1
                continue
            terminal_kind = terminal_content_kind(line)
            if terminal_kind != "han":
                exclusion_counts[f"code_switch_or_non_han_terminal::{terminal_kind}"] += 1
                terminal_exclusions_by_song[song_id][terminal_kind] += 1
                continue
            duplicate_key = duplicate_normalise(line)
            if not duplicate_key:
                exclusion_counts["empty_after_normalisation"] += 1
                continue
            ending = written_ending_features(line)
            if ending is None:
                exclusion_counts["unclassified_written_ending"] += 1
                continue
            sequence.append(
                {
                    "artist_label_id": label_id,
                    "song_id": song_id,
                    "chunk_id": chunk_id,
                    "chunk_line_index": chunk_line_index,
                    "line_hash": sha256_text(duplicate_key),
                    "duplicate_key": duplicate_key,
                    "analysis_text_sha256": analysis_hash,
                    **ending,
                }
            )
        if sequence:
            for index, row in enumerate(sequence):
                row["sequence_index"] = index
                row["sequence_length"] = len(sequence)
            sequences[(label_id, song_id, chunk_id)] = sequence
        else:
            exclusion_counts["empty_label_song_chunk_after_filters"] += 1

    # Repetition is retained to preserve adjacency. Record multiplicity and a
    # first-occurrence flag for duplicate-balanced sensitivity evaluation.
    rows_by_label_song: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (label_id, song_id, _), sequence in sequences.items():
        rows_by_label_song[(label_id, song_id)].extend(sequence)
    for rows in rows_by_label_song.values():
        counts = Counter(row["duplicate_key"] for row in rows)
        seen: set[str] = set()
        for row in sorted(rows, key=lambda item: (chunk_sort_key(item["chunk_id"]), item["chunk_line_index"])):
            row["within_label_song_line_multiplicity"] = counts[row["duplicate_key"]]
            row["first_occurrence_within_label_song"] = row["duplicate_key"] not in seen
            row["within_label_song_repeat"] = row["duplicate_key"] in seen
            seen.add(row["duplicate_key"])
        exclusion_counts["retained_repeat_line_occurrences"] += sum(count - 1 for count in counts.values())

    return (
        sequences,
        eligible_nodes,
        song_to_labels,
        dict(exclusion_counts),
        {song_id: dict(counter) for song_id, counter in terminal_exclusions_by_song.items()},
    )


def desired_split_counts(n: int) -> dict[str, int]:
    if n < 3:
        raise RuntimeError("Every eligible label must have at least three songs")
    validation = max(1, int(round(n * 0.15)))
    test = max(1, int(round(n * 0.15)))
    if validation + test > n - 1:
        test = max(1, n - validation - 1)
    return {"train": n - validation - test, "validation": validation, "test": test}


def assign_song_splits(song_to_labels: dict[str, set[str]]) -> tuple[dict[str, str], dict[str, Any]]:
    label_songs: dict[str, set[str]] = defaultdict(set)
    for song_id, labels in song_to_labels.items():
        for label in labels:
            label_songs[label].add(song_id)
    targets = {label: desired_split_counts(len(songs)) for label, songs in label_songs.items()}
    counts: dict[str, Counter[str]] = {label: Counter() for label in label_songs}
    global_target = {
        "train": int(round(len(song_to_labels) * 0.70)),
        "validation": int(round(len(song_to_labels) * 0.15)),
    }
    global_target["test"] = len(song_to_labels) - global_target["train"] - global_target["validation"]
    global_counts: Counter[str] = Counter()

    ordered_songs = sorted(
        song_to_labels,
        key=lambda song: (
            min(len(label_songs[label]) for label in song_to_labels[song]),
            -len(song_to_labels[song]),
            stable_unit(f"split-order::{song}"),
        ),
    )
    assignment: dict[str, str] = {}
    split_order = ("train", "validation", "test")
    for song_id in ordered_songs:
        labels = song_to_labels[song_id]
        scored: list[tuple[float, float, str]] = []
        for split in split_order:
            label_score = 0.0
            for label in labels:
                target = targets[label][split]
                deficit = target - counts[label][split]
                label_score += max(deficit, 0) / max(target, 1)
            global_deficit = max(global_target[split] - global_counts[split], 0) / max(global_target[split], 1)
            jitter = stable_unit(f"split-choice::{song_id}::{split}") * 1e-6
            scored.append((label_score + 0.10 * global_deficit + jitter, -global_counts[split], split))
        split = max(scored)[2]
        assignment[song_id] = split
        global_counts[split] += 1
        for label in labels:
            counts[label][split] += 1

    # Conservative repair: a move is allowed only if every label attached to
    # the song keeps at least one song in the donor partition.
    for _ in range(10):
        missing = [(label, split) for label in sorted(label_songs) for split in split_order if counts[label][split] == 0]
        if not missing:
            break
        changed = False
        for label, needed_split in missing:
            candidates: list[tuple[float, str, str]] = []
            for song_id in label_songs[label]:
                donor = assignment[song_id]
                if donor == needed_split:
                    continue
                labels = song_to_labels[song_id]
                if any(counts[item][donor] <= 1 for item in labels):
                    continue
                before = sum((counts[item][part] - targets[item][part]) ** 2 for item in labels for part in split_order)
                after = 0.0
                for item in labels:
                    for part in split_order:
                        revised = counts[item][part] - (1 if part == donor else 0) + (1 if part == needed_split else 0)
                        after += (revised - targets[item][part]) ** 2
                candidates.append((after - before, song_id, donor))
            if not candidates:
                continue
            _, song_id, donor = min(candidates, key=lambda item: (item[0], stable_unit(f"repair::{item[1]}")))
            assignment[song_id] = needed_split
            global_counts[donor] -= 1
            global_counts[needed_split] += 1
            for item in song_to_labels[song_id]:
                counts[item][donor] -= 1
                counts[item][needed_split] += 1
            changed = True
        if not changed:
            break

    missing = [(label, split) for label in label_songs for split in split_order if counts[label][split] == 0]
    if missing:
        raise RuntimeError(f"Label-aware song split could not populate all partitions: {missing[:10]}")
    overlap = {
        a + "_" + b: len({song for song, split in assignment.items() if split == a} & {song for song, split in assignment.items() if split == b})
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    deviations = [abs(counts[label][split] - targets[label][split]) for label in label_songs for split in split_order]
    audit = {
        "algorithm": "deterministic multilabel-aware greedy assignment with non-empty-partition repair",
        "target_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "song_counts": {split: sum(value == split for value in assignment.values()) for split in split_order},
        "label_count": len(label_songs),
        "labels_with_nonempty_all_partitions": sum(all(counts[label][split] > 0 for split in split_order) for label in label_songs),
        "maximum_absolute_label_song_count_deviation_from_target": max(deviations),
        "mean_absolute_label_song_count_deviation_from_target": round_float(float(np.mean(deviations))),
        "cross_partition_song_overlap": overlap,
    }
    return assignment, audit


def position_bucket(index: int) -> str:
    if index < 4:
        return "opening_2_to_4"
    if index < 12:
        return "developing_5_to_12"
    return "later_13_plus"


def run_bucket(run_length: int) -> str:
    if run_length <= 1:
        return "1"
    if run_length == 2:
        return "2"
    if run_length == 3:
        return "3"
    return "4_plus"


def make_feature(sequence: list[dict[str, Any]], index: int) -> tuple[dict[str, str], int]:
    history = sequence[:index]
    previous = history[-1]
    current_run = 1
    for prior in reversed(history[:-1]):
        if prior["final_family"] == previous["final_family"]:
            current_run += 1
        else:
            break
    feature: dict[str, str] = {
        "artist_label": previous["artist_label_id"],
        "previous_final": previous["raw_final"],
        "previous_tone": previous["tone"],
        "previous_two_family": previous["two_family_pattern"] or "NONE",
        "run_bucket": run_bucket(current_run),
        "position_bucket": position_bucket(index),
        "history_family_diversity": str(min(len({row["final_family"] for row in history[-4:]}), 4)),
    }
    for lag in range(1, 5):
        feature[f"family_lag_{lag}"] = history[-lag]["final_family"] if len(history) >= lag else "BOS"
    if len(history) >= 2:
        feature["previous_transition"] = history[-2]["final_family"] + ">" + history[-1]["final_family"]
    else:
        feature["previous_transition"] = "BOS>" + history[-1]["final_family"]
    return feature, current_run


def contiguous_segments(sequence: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split retained ending rows at every excluded/missing original line."""
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in sorted(sequence, key=lambda item: item["chunk_line_index"]):
        if current and row["chunk_line_index"] != current[-1]["chunk_line_index"] + 1:
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


def build_events(
    sequences: dict[tuple[str, str, str], list[dict[str, Any]]], song_split: dict[str, str]
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for (label_id, song_id, chunk_id), sequence in sorted(sequences.items()):
        split = song_split[song_id]
        for segment in contiguous_segments(sequence):
            for index in range(1, len(segment)):
                feature, current_run = make_feature(segment, index)
                target = segment[index]
                previous = segment[index - 1]
                events.append(
                    {
                        "artist_label_id": label_id,
                        "song_id": song_id,
                        "chunk_id": chunk_id,
                        "split": split,
                        "sequence_index": target["sequence_index"],
                        "chunk_line_index": target["chunk_line_index"],
                        "previous_chunk_line_index": previous["chunk_line_index"],
                        "line_hash": target["line_hash"],
                        "duplicate_key": target["duplicate_key"],
                        "target_family": target["final_family"],
                        "target_raw_final": target["raw_final"],
                        "target_tone": target["tone"],
                        "previous_family": previous["final_family"],
                        "transition_type": "continuation" if target["final_family"] == previous["final_family"] else "switch",
                        "prior_run_bucket": run_bucket(current_run),
                        "position_bucket": position_bucket(index),
                        "target_is_repeat_within_label_song": target["within_label_song_repeat"],
                        "target_first_occurrence_within_label_song": target["first_occurrence_within_label_song"],
                        "feature": feature,
                        "leakage_status": "not_checked" if split != "train" else "training_reference",
                    }
                )
    return events


def character_ngrams(value: str, n: int = 3) -> frozenset[str]:
    if len(value) < n:
        return frozenset()
    return frozenset(value[index : index + n] for index in range(len(value) - n + 1))


class NearDuplicateIndex:
    """Exact trigram inverted index with exact Jaccard verification."""

    def __init__(self, values: Iterable[str]) -> None:
        self.values = sorted(set(values))
        self.exact = set(self.values)
        self.shingles: list[frozenset[str]] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(self.values):
            grams = character_ngrams(value)
            self.shingles.append(grams)
            for gram in grams:
                self.postings[gram].append(index)

    def classify(self, value: str, threshold: float = NEAR_DUPLICATE_JACCARD) -> tuple[str, float]:
        if value in self.exact:
            return "exact_train_line_duplicate", 1.0
        grams = character_ngrams(value)
        if not grams:
            return "eligible_no_train_duplicate", 0.0
        candidate_overlap: Counter[int] = Counter()
        for gram in grams:
            candidate_overlap.update(self.postings.get(gram, ()))
        minimum_overlap = int(math.ceil(threshold * len(grams)))
        maximum = 0.0
        for index, overlap in candidate_overlap.items():
            if overlap < minimum_overlap:
                continue
            candidate = self.shingles[index]
            score = overlap / len(grams | candidate)
            maximum = max(maximum, score)
            if score >= threshold:
                return "near_train_line_duplicate", score
        return "eligible_no_train_duplicate", maximum


def apply_leakage_filter(
    events: list[dict[str, Any]],
    sequences: dict[tuple[str, str, str], list[dict[str, Any]]],
    song_split: dict[str, str],
) -> dict[str, Any]:
    train_song_ids = {song_id for song_id, split in song_split.items() if split == "train"}
    validation_song_ids = {song_id for song_id, split in song_split.items() if split == "validation"}
    # Include isolated/first lines as leakage references even when they are not
    # prediction targets. Test is checked against both train and validation.
    train_lines = [
        row["duplicate_key"]
        for (_, song_id, _), sequence in sequences.items()
        if song_id in train_song_ids
        for row in sequence
    ]
    train_validation_lines = [
        row["duplicate_key"]
        for (_, song_id, _), sequence in sequences.items()
        if song_id in train_song_ids or song_id in validation_song_ids
        for row in sequence
    ]
    train_index = NearDuplicateIndex(train_lines)
    train_validation_index = NearDuplicateIndex(train_validation_lines)
    counts: Counter[str] = Counter()
    for event in events:
        if event["split"] == "train":
            counts["training_reference"] += 1
            continue
        reference_name = "train" if event["split"] == "validation" else "train_or_validation"
        reference_index = train_index if event["split"] == "validation" else train_validation_index
        status, score = reference_index.classify(event["duplicate_key"])
        if reference_name == "train_or_validation":
            status = status.replace("train", "train_or_validation", 1)
        event["leakage_status"] = status
        event["maximum_reference_line_trigram_jaccard"] = score
        counts[f"{event['split']}::{status}"] += 1
    return {
        "train_unique_normalised_written_lines": len(train_index.values),
        "train_or_validation_unique_normalised_written_lines": len(train_validation_index.values),
        "near_duplicate_definition": "character-trigram Jaccard >= 0.80 after NFKC, case-folding, and punctuation/space removal",
        "shorter_than_three_normalised_characters_policy": "exact-match check only",
        "validation_reference": "all retained strict-Han-ending train lines",
        "test_reference": "all retained strict-Han-ending train and validation lines",
        "counts": dict(sorted(counts.items())),
    }


def safe_events(events: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "train":
        return [event for event in events if event["split"] == split]
    return [
        event
        for event in events
        if event["split"] == split
        and event["leakage_status"]
        in {"eligible_no_train_duplicate", "eligible_no_train_or_validation_duplicate"}
    ]


def align_probabilities(probabilities: np.ndarray, source_classes: Iterable[str], classes: list[str]) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], len(classes)), dtype=np.float64)
    destination = {label: index for index, label in enumerate(classes)}
    for source_index, label in enumerate(source_classes):
        aligned[:, destination[str(label)]] = probabilities[:, source_index]
    row_sum = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(aligned, row_sum, out=np.full_like(aligned, 1.0 / len(classes)), where=row_sum > 0)
    return aligned


def sklearn_csr32(matrix: Any) -> Any:
    """Normalise sparse index dtype for sklearn/scipy version combinations."""
    matrix = matrix.tocsr()
    if matrix.indices.dtype != np.int32:
        matrix.indices = matrix.indices.astype(np.int32)
    if matrix.indptr.dtype != np.int32:
        matrix.indptr = matrix.indptr.astype(np.int32)
    return matrix


def global_probabilities(train_events: list[dict[str, Any]], n: int, classes: list[str]) -> np.ndarray:
    counts = Counter(event["target_family"] for event in train_events)
    denominator = sum(counts.values()) + len(classes)
    row = np.array([(counts[label] + 1.0) / denominator for label in classes], dtype=np.float64)
    return np.tile(row, (n, 1))


def markov_probabilities(train_events: list[dict[str, Any]], eval_events: list[dict[str, Any]], classes: list[str]) -> np.ndarray:
    global_counts = Counter(event["target_family"] for event in train_events)
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for event in train_events:
        transition_counts[event["previous_family"]][event["target_family"]] += 1
    global_total = sum(global_counts.values())
    prior = np.array([(global_counts[label] + 1.0) / (global_total + len(classes)) for label in classes])
    rows: list[np.ndarray] = []
    for event in eval_events:
        counter = transition_counts[event["previous_family"]]
        support = sum(counter.values())
        # Empirical-Bayes backoff: one pseudo-sample of every class distributed
        # according to the global prior, with total prior strength 20.
        row = np.array([counter[label] + 20.0 * prior[index] for index, label in enumerate(classes)])
        row /= support + 20.0
        rows.append(row)
    return np.vstack(rows) if rows else np.empty((0, len(classes)))


def target_indices(events: list[dict[str, Any]], classes: list[str]) -> np.ndarray:
    index = {label: position for position, label in enumerate(classes)}
    return np.array([index[event["target_family"]] for event in events], dtype=np.int64)


def metric_arrays(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="stable")
    matches = order == truth[:, None]
    ranks = np.argmax(matches, axis=1) + 1
    confidence = probabilities.max(axis=1)
    predicted = order[:, 0]
    return {
        "top1": (ranks <= 1).astype(float),
        "top3": (ranks <= min(3, probabilities.shape[1])).astype(float),
        "top5": (ranks <= min(5, probabilities.shape[1])).astype(float),
        "reciprocal_rank": 1.0 / ranks,
        "negative_log_likelihood": -np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1.0)),
        "confidence": confidence,
        "predicted": predicted.astype(float),
    }


def expected_calibration_error(probabilities: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    predicted = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == truth).astype(float)
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        selected = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(result)


def summary_metrics(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    arrays = metric_arrays(probabilities, truth)
    return {
        "top1_accuracy": float(arrays["top1"].mean()),
        "top3_accuracy": float(arrays["top3"].mean()),
        "top5_accuracy": float(arrays["top5"].mean()),
        "mrr": float(arrays["reciprocal_rank"].mean()),
        "negative_log_likelihood": float(arrays["negative_log_likelihood"].mean()),
        "ece_10_bins": expected_calibration_error(probabilities, truth),
    }


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    result = np.exp(logp)
    result /= result.sum(axis=1, keepdims=True)
    return result


def hierarchical_probabilities(
    switch_classifier: SGDClassifier,
    switched_family_classifier: SGDClassifier,
    matrix: Any,
    events: list[dict[str, Any]],
    classes: list[str],
) -> np.ndarray:
    """Combine P(switch) with P(new family | switch), excluding the prior family."""
    switch_column = list(switch_classifier.classes_).index(True)
    switch_probability = switch_classifier.predict_proba(matrix)[:, switch_column]
    conditional = align_probabilities(
        switched_family_classifier.predict_proba(matrix), switched_family_classifier.classes_, classes
    )
    class_index = {label: index for index, label in enumerate(classes)}
    for row_index, event in enumerate(events):
        conditional[row_index, class_index[event["previous_family"]]] = 0.0
    conditional /= conditional.sum(axis=1, keepdims=True)
    result = conditional * switch_probability[:, None]
    for row_index, event in enumerate(events):
        result[row_index, class_index[event["previous_family"]]] += 1.0 - switch_probability[row_index]
    return result


def bootstrap_metric_intervals(
    probabilities: np.ndarray,
    truth: np.ndarray,
    song_ids: list[str],
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, tuple[float, float]]:
    arrays = metric_arrays(probabilities, truth)
    unique_songs = sorted(set(song_ids))
    song_to_indices: dict[str, np.ndarray] = {
        song: np.flatnonzero(np.array(song_ids, dtype=object) == song) for song in unique_songs
    }
    additive_names = ("top1", "top3", "top5", "reciprocal_rank")
    song_counts = np.array([len(song_to_indices[song]) for song in unique_songs], dtype=float)
    song_sums = {
        name: np.array([arrays[name][song_to_indices[song]].sum() for song in unique_songs], dtype=float)
        for name in additive_names
    }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in additive_names}
    for _ in range(replicates):
        draw = rng.integers(0, len(unique_songs), size=len(unique_songs))
        denominator = song_counts[draw].sum()
        for name in additive_names:
            samples[name].append(float(song_sums[name][draw].sum() / denominator))
    rename = {
        "top1": "top1_accuracy",
        "top3": "top3_accuracy",
        "top5": "top5_accuracy",
        "reciprocal_rank": "mrr",
    }
    return {
        rename[name]: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for name, values in samples.items()
    }


def train_models(
    events: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    train_events = safe_events(events, "train")
    validation_events = safe_events(events, "validation")
    test_events = safe_events(events, "test")
    classes = sorted({event["target_family"] for event in train_events}, key=lambda item: FAMILY_ORDER.index(item))
    if set(classes) != set(FAMILY_ORDER):
        raise RuntimeError(f"Training partition does not cover all configured classes: {classes}")
    vectorizer = DictVectorizer(sparse=True, sort=True)
    train_matrix = sklearn_csr32(vectorizer.fit_transform([event["feature"] for event in train_events]))
    validation_matrix = sklearn_csr32(vectorizer.transform([event["feature"] for event in validation_events]))
    test_matrix = sklearn_csr32(vectorizer.transform([event["feature"] for event in test_events]))
    no_label_vectorizer = DictVectorizer(sparse=True, sort=True)
    no_label_train_matrix = sklearn_csr32(
        no_label_vectorizer.fit_transform(
            [{key: value for key, value in event["feature"].items() if key != "artist_label"} for event in train_events]
        )
    )
    no_label_validation_matrix = sklearn_csr32(
        no_label_vectorizer.transform(
            [{key: value for key, value in event["feature"].items() if key != "artist_label"} for event in validation_events]
        )
    )
    no_label_test_matrix = sklearn_csr32(
        no_label_vectorizer.transform(
            [{key: value for key, value in event["feature"].items() if key != "artist_label"} for event in test_events]
        )
    )
    train_target = np.array([event["target_family"] for event in train_events], dtype=object)
    validation_truth = target_indices(validation_events, classes)
    test_truth = target_indices(test_events, classes)

    flat_candidates: list[dict[str, Any]] = []
    flat_fitted: dict[float, SGDClassifier] = {}
    hierarchical_candidates: list[dict[str, Any]] = []
    hierarchical_fitted: dict[float, tuple[SGDClassifier, SGDClassifier]] = {}
    no_label_candidates: list[dict[str, Any]] = []
    no_label_fitted: dict[float, tuple[SGDClassifier, SGDClassifier]] = {}
    train_switch = np.array(
        [event["target_family"] != event["previous_family"] for event in train_events], dtype=bool
    )
    switch_indices = np.flatnonzero(train_switch)
    for alpha in (1e-5, 1e-4, 1e-3):
        flat_classifier = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=1000,
            tol=1e-4,
            random_state=SEED,
            average=True,
            n_jobs=-1,
        )
        flat_classifier.fit(train_matrix, train_target)
        flat_fitted[alpha] = flat_classifier
        flat_probability = align_probabilities(
            flat_classifier.predict_proba(validation_matrix), flat_classifier.classes_, classes
        )
        flat_metrics = summary_metrics(flat_probability, validation_truth)
        flat_candidates.append({"alpha": alpha, **{key: round_float(value) for key, value in flat_metrics.items()}})

        switch_classifier = SGDClassifier(
            loss="log_loss", alpha=alpha, max_iter=1000, tol=1e-4,
            random_state=SEED, average=True, n_jobs=-1,
        )
        switched_family_classifier = SGDClassifier(
            loss="log_loss", alpha=alpha, max_iter=1000, tol=1e-4,
            random_state=SEED + 1, average=True, n_jobs=-1,
        )
        switch_classifier.fit(train_matrix, train_switch)
        switched_family_classifier.fit(train_matrix[switch_indices], train_target[switch_indices])
        hierarchical_fitted[alpha] = (switch_classifier, switched_family_classifier)
        hierarchical_probability = hierarchical_probabilities(
            switch_classifier, switched_family_classifier, validation_matrix, validation_events, classes
        )
        hierarchical_metrics = summary_metrics(hierarchical_probability, validation_truth)
        hierarchical_candidates.append(
            {"alpha": alpha, **{key: round_float(value) for key, value in hierarchical_metrics.items()}}
        )

        no_label_switch_classifier = SGDClassifier(
            loss="log_loss", alpha=alpha, max_iter=1000, tol=1e-4,
            random_state=SEED, average=True, n_jobs=-1,
        )
        no_label_family_classifier = SGDClassifier(
            loss="log_loss", alpha=alpha, max_iter=1000, tol=1e-4,
            random_state=SEED + 1, average=True, n_jobs=-1,
        )
        no_label_switch_classifier.fit(no_label_train_matrix, train_switch)
        no_label_family_classifier.fit(no_label_train_matrix[switch_indices], train_target[switch_indices])
        no_label_fitted[alpha] = (no_label_switch_classifier, no_label_family_classifier)
        no_label_probability = hierarchical_probabilities(
            no_label_switch_classifier,
            no_label_family_classifier,
            no_label_validation_matrix,
            validation_events,
            classes,
        )
        no_label_metrics = summary_metrics(no_label_probability, validation_truth)
        no_label_candidates.append(
            {"alpha": alpha, **{key: round_float(value) for key, value in no_label_metrics.items()}}
        )

    flat_chosen = max(flat_candidates, key=lambda row: (row["mrr"], row["top3_accuracy"], -row["alpha"]))
    flat_classifier = flat_fitted[flat_chosen["alpha"]]
    validation_flat_raw = align_probabilities(
        flat_classifier.predict_proba(validation_matrix), flat_classifier.classes_, classes
    )
    hierarchy_chosen = max(
        hierarchical_candidates, key=lambda row: (row["mrr"], row["top3_accuracy"], -row["alpha"])
    )
    switch_classifier, switched_family_classifier = hierarchical_fitted[hierarchy_chosen["alpha"]]
    validation_hierarchy_raw = hierarchical_probabilities(
        switch_classifier, switched_family_classifier, validation_matrix, validation_events, classes
    )
    no_label_chosen = max(
        no_label_candidates, key=lambda row: (row["mrr"], row["top3_accuracy"], -row["alpha"])
    )
    no_label_switch_classifier, no_label_family_classifier = no_label_fitted[no_label_chosen["alpha"]]
    validation_no_label_raw = hierarchical_probabilities(
        no_label_switch_classifier,
        no_label_family_classifier,
        no_label_validation_matrix,
        validation_events,
        classes,
    )
    temperatures = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
    flat_temperature_scores = []
    hierarchy_temperature_scores = []
    no_label_temperature_scores = []
    for temperature in temperatures:
        flat_nll = summary_metrics(
            temperature_scale(validation_flat_raw, temperature), validation_truth
        )["negative_log_likelihood"]
        hierarchy_nll = summary_metrics(
            temperature_scale(validation_hierarchy_raw, temperature), validation_truth
        )["negative_log_likelihood"]
        flat_temperature_scores.append(
            {"temperature": temperature, "validation_negative_log_likelihood": flat_nll}
        )
        hierarchy_temperature_scores.append(
            {"temperature": temperature, "validation_negative_log_likelihood": hierarchy_nll}
        )
        no_label_nll = summary_metrics(
            temperature_scale(validation_no_label_raw, temperature), validation_truth
        )["negative_log_likelihood"]
        no_label_temperature_scores.append(
            {"temperature": temperature, "validation_negative_log_likelihood": no_label_nll}
        )
    flat_temperature = min(
        flat_temperature_scores, key=lambda row: row["validation_negative_log_likelihood"]
    )["temperature"]
    hierarchy_temperature = min(
        hierarchy_temperature_scores, key=lambda row: row["validation_negative_log_likelihood"]
    )["temperature"]
    no_label_temperature = min(
        no_label_temperature_scores, key=lambda row: row["validation_negative_log_likelihood"]
    )["temperature"]
    validation_flat = temperature_scale(validation_flat_raw, flat_temperature)
    test_flat = temperature_scale(
        align_probabilities(flat_classifier.predict_proba(test_matrix), flat_classifier.classes_, classes),
        flat_temperature,
    )
    validation_hierarchy = temperature_scale(validation_hierarchy_raw, hierarchy_temperature)
    test_hierarchy = temperature_scale(
        hierarchical_probabilities(
            switch_classifier, switched_family_classifier, test_matrix, test_events, classes
        ),
        hierarchy_temperature,
    )
    validation_no_label = temperature_scale(validation_no_label_raw, no_label_temperature)
    test_no_label = temperature_scale(
        hierarchical_probabilities(
            no_label_switch_classifier,
            no_label_family_classifier,
            no_label_test_matrix,
            test_events,
            classes,
        ),
        no_label_temperature,
    )
    probabilities = {
        "global_frequency": global_probabilities(train_events, len(test_events), classes),
        "first_order_markov": markov_probabilities(train_events, test_events, classes),
        "flat_sgd_logistic_context": test_flat,
        "hierarchical_sgd_context": test_hierarchy,
        "hierarchical_sgd_no_source_label": test_no_label,
        "validation_flat_sgd_logistic_context": validation_flat,
        "validation_hierarchical_sgd_context": validation_hierarchy,
        "validation_hierarchical_sgd_no_source_label": validation_no_label,
    }
    model_audit = {
        "feature_count": len(vectorizer.feature_names_),
        "feature_schema": [
            "source-credit label",
            "previous four written-rhyme classes",
            "previous raw pinyin final and tone",
            "previous two-syllable final-family pattern",
            "previous transition",
            "current same-class run bucket",
            "written-line position bucket",
            "recent class diversity",
        ],
        "flat_context_candidate_models": flat_candidates,
        "flat_context_selected_alpha": flat_chosen["alpha"],
        "flat_context_temperature_candidates": [
            {key: round_float(value) for key, value in row.items()} for row in flat_temperature_scores
        ],
        "flat_context_selected_validation_temperature": flat_temperature,
        "hierarchical_context_definition": (
            "binary continue-versus-switch probability multiplied by a switch-only family classifier, "
            "with the previous family excluded from the conditional switch distribution"
        ),
        "hierarchical_context_candidate_models": hierarchical_candidates,
        "hierarchical_context_selected_alpha": hierarchy_chosen["alpha"],
        "hierarchical_context_temperature_candidates": [
            {key: round_float(value) for key, value in row.items()} for row in hierarchy_temperature_scores
        ],
        "hierarchical_context_selected_validation_temperature": hierarchy_temperature,
        "no_source_label_ablation_definition": "identical hierarchical architecture with the source-credit-label feature removed",
        "no_source_label_candidate_models": no_label_candidates,
        "no_source_label_selected_alpha": no_label_chosen["alpha"],
        "no_source_label_temperature_candidates": [
            {key: round_float(value) for key, value in row.items()} for row in no_label_temperature_scores
        ],
        "no_source_label_selected_validation_temperature": no_label_temperature,
        "train_event_count": len(train_events),
        "leakage_safe_validation_event_count": len(validation_events),
        "leakage_safe_test_event_count": len(test_events),
        "classes": classes,
    }
    model_bundle = {
        "vectorizer": vectorizer,
        "flat_classifier": flat_classifier,
        "flat_temperature": flat_temperature,
        "switch_classifier": switch_classifier,
        "switched_family_classifier": switched_family_classifier,
        "hierarchy_alpha": hierarchy_chosen["alpha"],
        "hierarchy_temperature": hierarchy_temperature,
    }
    return model_audit, probabilities, classes, validation_events, test_events, model_bundle


def evaluate_models(
    probabilities: dict[str, np.ndarray],
    classes: list[str],
    test_events: list[dict[str, Any]],
    all_events: list[dict[str, Any]],
    all_label_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    truth = target_indices(test_events, classes)
    song_ids = [event["song_id"] for event in test_events]
    candidate_test_events = [event for event in all_events if event["split"] == "test"]
    metric_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(
        (
            "global_frequency", "first_order_markov", "flat_sgd_logistic_context",
            "hierarchical_sgd_context", "hierarchical_sgd_no_source_label",
        )
    ):
        probability = probabilities[model]
        metrics = summary_metrics(probability, truth)
        intervals = bootstrap_metric_intervals(probability, truth, song_ids, SEED + model_index)
        row: dict[str, Any] = {
            "model": model,
            "evaluation_split": "song_held_out_test",
            "candidate_event_count_before_leakage_filter": len(candidate_test_events),
            "leakage_excluded_event_count": len(candidate_test_events) - len(test_events),
            "leakage_safe_event_count": len(test_events),
            "predicted_event_count": len(test_events),
            "song_count": len(set(song_ids)),
            "model_coverage_on_leakage_safe_events": 1.0,
            "end_to_end_event_coverage": len(test_events) / len(candidate_test_events),
            **{key: round_float(value) for key, value in metrics.items()},
        }
        for metric, (lower, upper) in intervals.items():
            row[f"{metric}_ci95_low"] = round_float(lower)
            row[f"{metric}_ci95_high"] = round_float(upper)
        metric_rows.append(row)

        for label_id in all_label_ids:
            indices = np.array([index for index, event in enumerate(test_events) if event["artist_label_id"] == label_id])
            support = len(indices)
            candidate_support = sum(event["artist_label_id"] == label_id for event in candidate_test_events)
            if support >= MIN_PER_LABEL_EVAL_EVENTS:
                label_metrics = summary_metrics(probability[indices], truth[indices])
                status = "reported"
            else:
                label_metrics = {key: float("nan") for key in ("top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr")}
                status = "no_leakage_safe_test_events" if support == 0 else "suppressed_below_20_events"
            per_label_rows.append(
                {
                    "model": model,
                    "artist_label_id": label_id,
                    "candidate_event_count_before_leakage_filter": candidate_support,
                    "leakage_excluded_event_count": candidate_support - support,
                    "leakage_safe_event_count": support,
                    "predicted_event_count": support,
                    "end_to_end_event_coverage": support / candidate_support if candidate_support else 0.0,
                    "metric_status": status,
                    **{key: round_float(value) for key, value in label_metrics.items() if key in ("top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr")},
                }
            )

        stratum_specs = {
            "transition_type": sorted({event["transition_type"] for event in test_events}),
            "prior_run_bucket": ("1", "2", "3", "4_plus"),
            "position_bucket": ("opening_2_to_4", "developing_5_to_12", "later_13_plus"),
            "target_is_repeat_within_label_song": (False, True),
        }
        for field, values in stratum_specs.items():
            for value in values:
                indices = np.array([index for index, event in enumerate(test_events) if event[field] == value])
                if len(indices) == 0:
                    continue
                result = summary_metrics(probability[indices], truth[indices])
                stratified_rows.append(
                    {
                        "model": model,
                        "stratum_dimension": field,
                        "stratum_value": value,
                        "eligible_event_count": len(indices),
                        "song_count": len({test_events[index]["song_id"] for index in indices}),
                        "top1_accuracy": round_float(result["top1_accuracy"]),
                        "top3_accuracy": round_float(result["top3_accuracy"]),
                        "mrr": round_float(result["mrr"]),
                    }
                )

    ml_label_rows = [
        row
        for row in per_label_rows
        if row["model"] == "hierarchical_sgd_context" and row["metric_status"] == "reported"
    ]
    macro = {
        "model": "hierarchical_sgd_context",
        "evaluation_split": "song_held_out_test_macro_across_labels_with_at_least_20_events",
        "candidate_event_count_before_leakage_filter": sum(
            row["candidate_event_count_before_leakage_filter"] for row in ml_label_rows
        ),
        "leakage_excluded_event_count": sum(row["leakage_excluded_event_count"] for row in ml_label_rows),
        "leakage_safe_event_count": sum(row["leakage_safe_event_count"] for row in ml_label_rows),
        "predicted_event_count": sum(row["predicted_event_count"] for row in ml_label_rows),
        "song_count": "",
        "model_coverage_on_leakage_safe_events": 1.0,
        "end_to_end_event_coverage": round_float(
            sum(row["predicted_event_count"] for row in ml_label_rows)
            / sum(row["candidate_event_count_before_leakage_filter"] for row in ml_label_rows)
        ),
        "reported_label_share": len(ml_label_rows) / len(all_label_ids),
        "top1_accuracy": round_float(float(np.mean([row["top1_accuracy"] for row in ml_label_rows]))),
        "top3_accuracy": round_float(float(np.mean([row["top3_accuracy"] for row in ml_label_rows]))),
        "top5_accuracy": round_float(float(np.mean([row["top5_accuracy"] for row in ml_label_rows]))),
        "mrr": round_float(float(np.mean([row["mrr"] for row in ml_label_rows]))),
        "negative_log_likelihood": None,
        "ece_10_bins": None,
    }
    metric_rows.append(macro)
    return metric_rows, per_label_rows, stratified_rows


def paired_model_deltas(
    probabilities: dict[str, np.ndarray], classes: list[str], test_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Song-cluster paired uncertainty for the released model's improvements."""
    truth = target_indices(test_events, classes)
    song_ids = [event["song_id"] for event in test_events]
    songs = sorted(set(song_ids))
    index_by_song = {
        song: np.array([index for index, value in enumerate(song_ids) if value == song], dtype=int)
        for song in songs
    }
    released = metric_arrays(probabilities["hierarchical_sgd_context"], truth)
    rows: list[dict[str, Any]] = []
    name_map = {"top1": "top1_accuracy", "top3": "top3_accuracy", "top5": "top5_accuracy", "reciprocal_rank": "mrr"}
    for reference_offset, reference_name in enumerate(
        ("first_order_markov", "flat_sgd_logistic_context", "hierarchical_sgd_no_source_label")
    ):
        reference = metric_arrays(probabilities[reference_name], truth)
        for metric_offset, (array_name, metric_name) in enumerate(name_map.items()):
            difference = released[array_name] - reference[array_name]
            sums = np.array([difference[index_by_song[song]].sum() for song in songs])
            counts = np.array([len(index_by_song[song]) for song in songs])
            rng = np.random.default_rng(SEED + 500 + reference_offset * 10 + metric_offset)
            samples = []
            for _ in range(BOOTSTRAP_REPLICATES):
                draw = rng.integers(0, len(songs), size=len(songs))
                samples.append(float(sums[draw].sum() / counts[draw].sum()))
            rows.append(
                {
                    "released_model": "hierarchical_sgd_context",
                    "reference_model": reference_name,
                    "metric": metric_name,
                    "paired_difference_released_minus_reference": round_float(float(difference.mean())),
                    "song_cluster_bootstrap_ci95_low": round_float(float(np.percentile(samples, 2.5))),
                    "song_cluster_bootstrap_ci95_high": round_float(float(np.percentile(samples, 97.5))),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    return rows


def evaluate_abstention(
    probabilities: dict[str, np.ndarray],
    classes: list[str],
    validation_events: list[dict[str, Any]],
    test_events: list[dict[str, Any]],
    all_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation_probability = probabilities["validation_hierarchical_sgd_context"]
    test_probability = probabilities["hierarchical_sgd_context"]
    validation_confidence = validation_probability.max(axis=1)
    truth = target_indices(test_events, classes)
    test_confidence = test_probability.max(axis=1)
    candidate_test_count = sum(event["split"] == "test" for event in all_events)
    operating_points = (
        ("all_predictions", 0.0),
        ("validation_target_75pct_coverage", float(np.quantile(validation_confidence, 0.25))),
        ("validation_target_50pct_coverage", float(np.quantile(validation_confidence, 0.50))),
        ("validation_target_25pct_coverage", float(np.quantile(validation_confidence, 0.75))),
    )
    rows: list[dict[str, Any]] = []
    for name, threshold in operating_points:
        selected = test_confidence >= threshold
        if not selected.any():
            continue
        metrics = summary_metrics(test_probability[selected], truth[selected])
        rows.append(
            {
                "model": "hierarchical_sgd_context",
                "operating_point": name,
                "threshold_derived_from_validation": round_float(threshold),
                "test_coverage": round_float(float(selected.mean())),
                "coverage_against_all_primary_test_candidates": round_float(float(selected.sum() / candidate_test_count)),
                "accepted_event_count": int(selected.sum()),
                "abstained_event_count": int((~selected).sum()),
                "top1_accuracy_on_accepted": round_float(metrics["top1_accuracy"]),
                "top3_accuracy_on_accepted": round_float(metrics["top3_accuracy"]),
                "mrr_on_accepted": round_float(metrics["mrr"]),
            }
        )
    recommended = next(row for row in rows if row["operating_point"] == "validation_target_50pct_coverage")
    return rows, {
        "recommended_operating_point": recommended["operating_point"],
        "threshold": recommended["threshold_derived_from_validation"],
        "interpretation": "abstain when maximum ML class probability is below this validation-derived threshold",
        "warning": "confidence is model confidence for a written-ending class, not a judgment of lyrical quality or performed rhyme",
    }


def top_items(counter: Counter[str], total: int, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count, "share": round_float(count / total)}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def build_fingerprints(
    sequences: dict[tuple[str, str, str], list[dict[str, Any]]], nodes: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    corpus_families: Counter[str] = Counter()
    for (label_id, song_id, _), sequence in sequences.items():
        by_label[label_id].append((song_id, sequence))
        corpus_families.update(row["final_family"] for row in sequence)
    corpus_total = sum(corpus_families.values())
    detailed: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    for label_id in sorted(by_label):
        sequence_entries = by_label[label_id]
        family_counts: Counter[str] = Counter()
        raw_counts: Counter[str] = Counter()
        tone_counts: Counter[str] = Counter()
        two_counts: Counter[str] = Counter()
        transition_count = 0
        continuation_count = 0
        local_echo_count = 0
        local_echo_denominator = 0
        two_echo_count = 0
        two_echo_denominator = 0
        lines_in_runs = 0
        run_lengths: list[int] = []
        local_window_sizes: list[int] = []
        song_transition_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for song_id, sequence in sequence_entries:
            families_all = [row["final_family"] for row in sequence]
            patterns_all = [row["two_family_pattern"] for row in sequence]
            family_counts.update(families_all)
            raw_counts.update(row["raw_final"] for row in sequence)
            tone_counts.update(row["tone"] for row in sequence)
            two_counts.update(pattern for pattern in patterns_all if pattern)
            for segment in contiguous_segments(sequence):
                families = [row["final_family"] for row in segment]
                patterns = [row["two_family_pattern"] for row in segment]
                if len(segment) >= 2:
                    song_continuations = sum(families[index] == families[index - 1] for index in range(1, len(families)))
                    transition_count += len(segment) - 1
                    continuation_count += song_continuations
                    song_transition_counts[song_id][0] += song_continuations
                    song_transition_counts[song_id][1] += len(segment) - 1
                for index, family in enumerate(families):
                    if index > 0:
                        local_echo_denominator += 1
                        window = min(4, index)
                        local_window_sizes.append(window)
                        if family in families[index - window : index]:
                            local_echo_count += 1
                    if patterns[index] and index > 0:
                        two_echo_denominator += 1
                        window = min(4, index)
                        if patterns[index] in patterns[index - window : index]:
                            two_echo_count += 1
                start = 0
                while start < len(families):
                    end = start + 1
                    while end < len(families) and families[end] == families[start]:
                        end += 1
                    length = end - start
                    run_lengths.append(length)
                    if length >= 2:
                        lines_in_runs += length
                    start = end
        song_echo_rates = [same / total_count for same, total_count in song_transition_counts.values() if total_count]
        total = sum(family_counts.values())
        probabilities = np.array([family_counts[family] / total for family in FAMILY_ORDER])
        positive_probabilities = probabilities[probabilities > 0]
        entropy = -float(np.sum(positive_probabilities * np.log(positive_probabilities)))
        normalised_entropy = entropy / math.log(len(FAMILY_ORDER))
        expected_local_echo = float(np.mean([
            sum(probability * (1.0 - (1.0 - probability) ** window) for probability in probabilities)
            for window in local_window_sizes
        ])) if local_window_sizes else 0.0
        observed_local_echo = local_echo_count / max(local_echo_denominator, 1)
        distinctive = []
        for family in FAMILY_ORDER:
            label_rate = (family_counts[family] + 0.5) / (total + 0.5 * len(FAMILY_ORDER))
            corpus_rate = (corpus_families[family] + 0.5) / (corpus_total + 0.5 * len(FAMILY_ORDER))
            distinctive.append({"family": family, "log2_rate_ratio_vs_corpus": math.log2(label_rate / corpus_rate)})
        distinctive.sort(key=lambda row: (-row["log2_rate_ratio_vs_corpus"], row["family"]))
        node = nodes[label_id]
        profile = {
            "artist_label_id": label_id,
            "source_artist_label": node["source_artist_label"],
            "label_attribution_status": node["label_attribution_status"],
            "claim_boundary": "source-credit-labelled written lyric endings; not verified identity, performed rhyme, flow, or audio style",
            "eligible_song_count": len({song_id for song_id, _ in sequence_entries}),
            "eligible_written_line_occurrence_count": total,
            "top_written_rhyme_families": top_items(family_counts, total),
            "distinctive_written_rhyme_families_vs_corpus": [
                {"family": row["family"], "log2_rate_ratio_vs_corpus": round_float(row["log2_rate_ratio_vs_corpus"])}
                for row in distinctive[:5]
            ],
            "top_raw_pinyin_finals": top_items(raw_counts, total),
            "top_two_syllable_family_patterns": top_items(two_counts, sum(two_counts.values())) if two_counts else [],
            "final_tone_distribution": top_items(tone_counts, total, limit=5),
            "adjacent_same_family_rate": round_float(continuation_count / max(transition_count, 1)),
            "song_normalised_adjacent_same_family_rate": round_float(float(np.mean(song_echo_rates))) if song_echo_rates else None,
            "written_rhyme_switch_rate": round_float(1.0 - continuation_count / max(transition_count, 1)),
            "local_four_line_family_echo_rate": round_float(observed_local_echo),
            "iid_frequency_expectation_for_local_four_line_echo": round_float(expected_local_echo),
            "local_echo_lift_over_iid_frequency_expectation": round_float(observed_local_echo / expected_local_echo) if expected_local_echo else None,
            "two_syllable_pattern_local_echo_rate": round_float(two_echo_count / max(two_echo_denominator, 1)),
            "share_of_lines_in_adjacent_runs_of_two_or_more": round_float(lines_in_runs / total),
            "median_same_family_run_length": round_float(float(np.median(run_lengths))) if run_lengths else None,
            "maximum_same_family_run_length": max(run_lengths) if run_lengths else 0,
            "normalised_family_entropy": round_float(normalised_entropy),
            "family_concentration_hhi": round_float(float(np.sum(probabilities**2))),
        }
        detailed.append(profile)
        compact.append(
            {
                "artist_label_id": label_id,
                "source_artist_label": node["source_artist_label"],
                "label_attribution_status": node["label_attribution_status"],
                "eligible_song_count": len({song_id for song_id, _ in sequence_entries}),
                "eligible_written_line_occurrence_count": total,
                "dominant_written_rhyme_family": profile["top_written_rhyme_families"][0]["value"],
                "dominant_family_share": profile["top_written_rhyme_families"][0]["share"],
                "adjacent_same_family_rate": profile["adjacent_same_family_rate"],
                "local_four_line_family_echo_rate": profile["local_four_line_family_echo_rate"],
                "local_echo_lift_over_iid_frequency_expectation": profile["local_echo_lift_over_iid_frequency_expectation"],
                "two_syllable_pattern_local_echo_rate": profile["two_syllable_pattern_local_echo_rate"],
                "share_of_lines_in_adjacent_runs_of_two_or_more": profile["share_of_lines_in_adjacent_runs_of_two_or_more"],
                "normalised_family_entropy": profile["normalised_family_entropy"],
            }
        )
    return detailed, compact


def parse_section_lines(text: str) -> list[tuple[str | None, str]]:
    result: list[tuple[str | None, str]] = []
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if SECTION_RE.fullmatch(line):
            section = line
        else:
            result.append((section, line))
    return result


def confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return {
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matthews_correlation": (tp * tn - fp * fn) / denominator,
    }


def muchin_auxiliary_agreement() -> dict[str, Any]:
    if not MUCHIN1000_DIR.exists():
        return {
            "status": "not_run_missing_local_resource",
            "claim_boundary": "not used for training or primary evaluation",
        }
    per_song_confusion: list[tuple[int, int, int, int]] = []
    pair_count = 0
    aligned_count = 0
    excluded_no_han = 0
    for lyric_path in sorted(MUCHIN1000_DIR.rglob("str_lyric")):
        rhyme_path = lyric_path.with_name("str_rhyme")
        if not rhyme_path.exists():
            continue
        pair_count += 1
        lyric_rows = parse_section_lines(lyric_path.read_text(encoding="utf-8"))
        rhyme_rows = parse_section_lines(rhyme_path.read_text(encoding="utf-8"))
        if len(lyric_rows) != len(rhyme_rows):
            continue
        if any(left[0] != right[0] for left, right in zip(lyric_rows, rhyme_rows)):
            continue
        aligned_count += 1
        grouped: dict[str | None, list[tuple[str, bool]]] = defaultdict(list)
        for (section, lyric), (_, annotation) in zip(lyric_rows, rhyme_rows):
            ending = written_ending_features(lyric)
            if ending is None:
                excluded_no_han += 1
                continue
            grouped[section].append((ending["final_family"], "R" in annotation))
        tp = fp = fn = tn = 0
        for rows in grouped.values():
            counts = Counter(family for family, _ in rows)
            for family, annotated in rows:
                predicted = counts[family] >= 2
                if predicted and annotated:
                    tp += 1
                elif predicted:
                    fp += 1
                elif annotated:
                    fn += 1
                else:
                    tn += 1
        per_song_confusion.append((tp, fp, fn, tn))
    totals = tuple(sum(row[index] for row in per_song_confusion) for index in range(4))
    tp, fp, fn, tn = totals
    metrics = confusion_metrics(tp, fp, fn, tn)
    rng = np.random.default_rng(SEED + 991)
    f1_samples: list[float] = []
    mcc_samples: list[float] = []
    confusion_array = np.array(per_song_confusion, dtype=int)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = confusion_array[rng.integers(0, len(confusion_array), size=len(confusion_array))].sum(axis=0)
        result = confusion_metrics(*[int(value) for value in sampled])
        f1_samples.append(result["f1"])
        mcc_samples.append(result["matthews_correlation"])
    total_lines = tp + fp + fn + tn
    return {
        "status": "completed_auxiliary_partially_circular_agreement_check",
        "dataset": "MuChin V1 1000 publisher-recommended annotation folders included in the local MuChin-V2 repository",
        "use_in_primary_model": False,
        "use_in_model_selection": False,
        "paired_song_folders_found": pair_count,
        "exact_line_and_section_aligned_song_folders": aligned_count,
        "classified_written_lines": total_lines,
        "excluded_lines_without_classifiable_Han_ending": excluded_no_han,
        "manual_R_positive_rate": round_float((tp + fn) / total_lines),
        "derived_same_family_within_section_positive_rate": round_float((tp + fp) / total_lines),
        "confusion": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
        "metrics": {key: round_float(value) for key, value in metrics.items()},
        "song_cluster_bootstrap_95pct": {
            "f1": [round_float(float(np.percentile(f1_samples, 2.5))), round_float(float(np.percentile(f1_samples, 97.5)))],
            "matthews_correlation": [
                round_float(float(np.percentile(mcc_samples, 2.5))),
                round_float(float(np.percentile(mcc_samples, 97.5))),
            ],
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "comparison_rule": "within each annotated section, mark a line positive when at least one other line has the same derived written-ending family",
        "partial_circularity": (
            "MuChin's annotation UI automatically grouped and colour-highlighted line-final pinyin rhymes; annotators were instructed mainly "
            "to check polyphonic pronunciation. The exported R marker collapses non-empty group membership and does not retain class identity."
        ),
        "valid_claim": "narrow implementation agreement / polyphone-sensitive sanity check only",
        "invalid_claims": [
            "independent gold-standard validation",
            "Chinese-rap domain validation",
            "performed-rhyme, flow, cadence, or audio validation",
        ],
        "duplicate_policy": "used MuChin1000 folders only, as the publisher recommends for the 724 release overlaps; the combined 6066 JSON was not used",
        "provenance_hashes": {
            "root_LICENSE_sha256": sha256_file(MUCHIN_LICENSE) if MUCHIN_LICENSE.exists() else None,
            "annotation_UI_sha256": sha256_file(MUCHIN_UI_FILE) if MUCHIN_UI_FILE.exists() else None,
            "backend_character_util_sha256": sha256_file(MUCHIN_BACKEND_FILE) if MUCHIN_BACKEND_FILE.exists() else None,
            "overlap_list_sha256": sha256_file(MUCHIN_OVERLAP_FILE) if MUCHIN_OVERLAP_FILE.exists() else None,
        },
    }


def make_recommender_lookup(
    events: list[dict[str, Any]],
    classes: list[str],
    abstention: dict[str, Any],
    nodes: dict[str, dict[str, str]],
    model_bundle: dict[str, Any],
    label_ablation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train_events = safe_events(events, "train")
    global_counts = Counter(event["target_family"] for event in train_events)
    global_total = sum(global_counts.values())

    def recommendations(row: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
        order = np.argsort(-row, kind="stable")[:limit]
        return [{"written_rhyme_family": classes[index], "probability": round_float(row[index])} for index in order]

    global_row = np.array([(global_counts[label] + 1.0) / (global_total + len(classes)) for label in classes])
    markov_lookup: dict[str, Any] = {}
    for previous in classes:
        synthetic = [{"previous_family": previous}]
        row = markov_probabilities(train_events, synthetic, classes)[0]
        support = sum(event["previous_family"] == previous for event in train_events)
        markov_lookup[previous] = {"training_event_support": support, "top_5": recommendations(row)}

    # Average ML predictions for common observed, text-free contexts.  The
    # omitted raw-final/tone details are integrated over observed train events.
    train_matrix = sklearn_csr32(
        model_bundle["vectorizer"].transform([event["feature"] for event in train_events])
    )
    raw_probability = temperature_scale(
        hierarchical_probabilities(
            model_bundle["switch_classifier"],
            model_bundle["switched_family_classifier"],
            train_matrix,
            train_events,
            classes,
        ),
        model_bundle["hierarchy_temperature"],
    )
    groups: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for index, event in enumerate(train_events):
        feature = event["feature"]
        key = (
            event["artist_label_id"],
            feature["family_lag_2"],
            feature["family_lag_1"],
            feature["run_bucket"],
            feature["position_bucket"],
        )
        groups[key].append(index)
    context_rows: list[dict[str, Any]] = []
    for key, indices in groups.items():
        if len(indices) < MIN_PUBLIC_CONTEXT_SUPPORT:
            continue
        label_id, previous_2, previous_1, current_run, current_position = key
        mean_probability = raw_probability[np.array(indices)].mean(axis=0)
        context_rows.append(
            {
                "artist_label_id": label_id,
                "source_artist_label": nodes[label_id]["source_artist_label"],
                "previous_2_written_rhyme_family": previous_2,
                "previous_1_written_rhyme_family": previous_1,
                "current_same_family_run_bucket": current_run,
                "written_line_position_bucket": current_position,
                "training_event_support": len(indices),
                "top_5": recommendations(mean_probability),
            }
        )
    context_rows.sort(key=lambda row: (-row["training_event_support"], row["artist_label_id"], row["previous_1_written_rhyme_family"]))
    context_rows = context_rows[:MAX_PUBLIC_CONTEXTS]
    ablation_by_metric = {row["metric"]: row for row in label_ablation_rows}
    label_conditioning_supported = all(
        ablation_by_metric.get(metric, {}).get("song_cluster_bootstrap_ci95_low", -1.0) > 0
        for metric in ("top3_accuracy", "mrr")
    )
    return {
        "artifact_id": ARTIFACT_ID,
        "scope": "next written-line ending family recommendation from preceding written-ending context",
        "released_model": {
            "name": "hierarchical_sgd_context",
            "selected_alpha": model_bundle["hierarchy_alpha"],
            "selected_validation_temperature": model_bundle["hierarchy_temperature"],
        },
        "source_credit_label_conditioning": {
            "paired_ablation_rows": label_ablation_rows,
            "top3_and_mrr_intervals_support_positive_increment": label_conditioning_supported,
            "claim_policy": (
                "label conditioning may be described as incrementally useful on these held-out metrics"
                if label_conditioning_supported
                else "do not make a personalization-benefit claim; label-conditioned lookup is descriptive/experimental only"
            ),
        },
        "claim_boundary": "not lyric generation, performed rhyme, flow, cadence, audio style, quality judgment, or verified artist identity",
        "classes": [
            {"written_rhyme_family": family, "included_strict_pypinyin_finals": list(finals), "plain_description": description}
            for family, finals, description in RHYME_FAMILIES
        ],
        "global_frequency_top_5": recommendations(global_row),
        "markov_by_previous_family": markov_lookup,
        "ml_common_observed_contexts": context_rows,
        "ml_context_minimum_training_support": MIN_PUBLIC_CONTEXT_SUPPORT,
        "ml_context_rows_released": len(context_rows),
        "abstention": abstention,
        "public_safety": "aggregate probabilities only; no lyric text, written lines, song/chunk IDs, or content hashes",
    }


def validate_public_payloads(public_files: list[Path]) -> dict[str, Any]:
    forbidden_columns = {"song_id", "chunk_id", "line_hash", "analysis_text", "lyrics", "lyric_text", "duplicate_key"}
    forbidden_text = ("song_id", "chunk_id", "analysis_text_sha256", "canonical_lyric_text_sha256", "duplicate_key")
    checks: list[dict[str, Any]] = []
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        leaked_tokens = [token for token in forbidden_text if token in text]
        checks.append({"check": f"public_forbidden_token_scan::{path.name}", "passed": not leaked_tokens, "details": leaked_tokens})
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or [])
            bad_columns = sorted(columns & forbidden_columns)
            checks.append({"check": f"public_forbidden_column_scan::{path.name}", "passed": not bad_columns, "details": bad_columns})
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "checks": checks,
    }


def build() -> None:
    clear_build_directory(PUBLIC_DIR)
    clear_build_directory(PRIVATE_DIR)
    sequences, nodes, song_to_labels, line_exclusions, terminal_exclusions_by_song = load_corpus()
    song_split, split_audit = assign_song_splits(song_to_labels)
    terminal_exclusions_by_split: dict[str, Counter[str]] = {
        split: Counter() for split in ("train", "validation", "test")
    }
    for song_id, counter in terminal_exclusions_by_song.items():
        terminal_exclusions_by_split[song_split[song_id]].update(counter)
    events = build_events(sequences, song_split)
    leakage_audit = apply_leakage_filter(events, sequences, song_split)
    model_audit, probabilities, classes, validation_events, test_events, model_bundle = train_models(events)
    atomic_write_json(PRIVATE_DIR / "model_audit.json", model_audit)
    metric_rows, per_label_rows, stratified_rows = evaluate_models(
        probabilities, classes, test_events, events, sorted(nodes)
    )
    paired_delta_rows = paired_model_deltas(probabilities, classes, test_events)
    abstention_rows, abstention = evaluate_abstention(
        probabilities, classes, validation_events, test_events, events
    )
    fingerprints, fingerprint_rows = build_fingerprints(sequences, nodes)
    auxiliary = muchin_auxiliary_agreement()

    class_inventory = [
        {
            "written_rhyme_family": family,
            "included_strict_pypinyin_finals": " | ".join(finals),
            "plain_description": description,
            "tone_handling": "tone stored separately; family is tone-free",
        }
        for family, finals, description in RHYME_FAMILIES
    ]
    analysis_summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "central_result": "written-line ending families can be profiled and predicted under strict song-held-out, duplicate-filtered evaluation",
        "claim_boundary": (
            "All results concern pypinyin-derived endings of written lyric lines attached to unverified source-credit labels. "
            "They do not measure performed rhyme, flow, cadence, beat alignment, pronunciation variants, audio style, or real-world identity."
        ),
        "population": {
            "graph_eligible_source_credit_labels": len(nodes),
            "label_song_chunk_sequences": len(sequences),
            "unique_songs": len(song_split),
            "retained_strict_han_ending_written_line_occurrences": sum(len(sequence) for sequence in sequences.values()),
            "prediction_events_before_leakage_filter": len(events),
            "strict_terminal_han_primary": True,
            "line_exclusion_counts": line_exclusions,
            "non_han_terminal_exclusions_by_split": {
                split: dict(sorted(counter.items())) for split, counter in terminal_exclusions_by_split.items()
            },
        },
        "split": split_audit,
        "leakage_control": leakage_audit,
        "model_selection": model_audit,
        "primary_test_metrics": metric_rows,
        "paired_released_model_differences": paired_delta_rows,
        "source_credit_label_ablation": [
            row for row in paired_delta_rows if row["reference_model"] == "hierarchical_sgd_no_source_label"
        ],
        "abstention": {"operating_points": abstention_rows, "recommended": abstention},
        "auxiliary_MuChin_agreement": auxiliary,
        "public_privacy": "no lyric text, full written lines, song/chunk IDs, memberships, or content hashes",
    }

    atomic_write_json(PUBLIC_DIR / "analysis_summary.json", analysis_summary)
    atomic_write_json(PUBLIC_DIR / "label_written_rhyme_fingerprints.json", {"artifact_id": ARTIFACT_ID, "profiles": fingerprints})
    atomic_write_csv(
        PUBLIC_DIR / "label_written_rhyme_summary.csv",
        fingerprint_rows,
        [
            "artist_label_id", "source_artist_label", "label_attribution_status", "eligible_song_count",
            "eligible_written_line_occurrence_count", "dominant_written_rhyme_family", "dominant_family_share",
            "adjacent_same_family_rate", "local_four_line_family_echo_rate",
            "local_echo_lift_over_iid_frequency_expectation", "two_syllable_pattern_local_echo_rate",
            "share_of_lines_in_adjacent_runs_of_two_or_more", "normalised_family_entropy",
        ],
    )
    atomic_write_csv(
        PUBLIC_DIR / "rhyme_class_inventory.csv",
        class_inventory,
        ["written_rhyme_family", "included_strict_pypinyin_finals", "plain_description", "tone_handling"],
    )
    metric_fields = [
        "model", "evaluation_split", "candidate_event_count_before_leakage_filter", "leakage_excluded_event_count",
        "leakage_safe_event_count", "predicted_event_count", "song_count", "model_coverage_on_leakage_safe_events",
        "end_to_end_event_coverage", "reported_label_share", "top1_accuracy",
        "top1_accuracy_ci95_low", "top1_accuracy_ci95_high", "top3_accuracy", "top3_accuracy_ci95_low",
        "top3_accuracy_ci95_high", "top5_accuracy", "top5_accuracy_ci95_low", "top5_accuracy_ci95_high",
        "mrr", "mrr_ci95_low", "mrr_ci95_high", "negative_log_likelihood", "ece_10_bins",
    ]
    atomic_write_csv(PUBLIC_DIR / "model_metrics.csv", metric_rows, metric_fields)
    atomic_write_csv(
        PUBLIC_DIR / "paired_model_deltas.csv",
        paired_delta_rows,
        [
            "released_model", "reference_model", "metric", "paired_difference_released_minus_reference",
            "song_cluster_bootstrap_ci95_low", "song_cluster_bootstrap_ci95_high", "bootstrap_replicates",
        ],
    )
    atomic_write_csv(
        PUBLIC_DIR / "per_label_metrics.csv",
        per_label_rows,
        [
            "model", "artist_label_id", "candidate_event_count_before_leakage_filter", "leakage_excluded_event_count",
            "leakage_safe_event_count", "predicted_event_count", "end_to_end_event_coverage", "metric_status",
            "top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr",
        ],
    )
    atomic_write_csv(
        PUBLIC_DIR / "stratified_metrics.csv",
        stratified_rows,
        ["model", "stratum_dimension", "stratum_value", "eligible_event_count", "song_count", "top1_accuracy", "top3_accuracy", "mrr"],
    )
    atomic_write_csv(
        PUBLIC_DIR / "abstention_metrics.csv",
        abstention_rows,
        [
            "model", "operating_point", "threshold_derived_from_validation", "test_coverage", "accepted_event_count",
            "coverage_against_all_primary_test_candidates", "abstained_event_count", "top1_accuracy_on_accepted",
            "top3_accuracy_on_accepted", "mrr_on_accepted",
        ],
    )
    atomic_write_json(PUBLIC_DIR / "muchin_auxiliary_agreement.json", auxiliary)
    lookup = make_recommender_lookup(
        events,
        classes,
        abstention,
        nodes,
        model_bundle,
        [row for row in paired_delta_rows if row["reference_model"] == "hierarchical_sgd_no_source_label"],
    )
    atomic_write_json(PUBLIC_DIR / "recommender_lookup.json", lookup)

    method_text = f"""# Method: Chinese written-rhyme modelling V1

## Scope and claim boundary

The task predicts the tone-free pinyin-final family of the next **written lyric line ending**. It does not observe audio and therefore does not measure performed rhyme, flow, cadence, stress, beat alignment, regional pronunciation, or delivery. Corpus labels are source-credit labels whose real-world identities have not been externally verified.

## Population and line construction

The input population is the {len(nodes)} graph-eligible source-credit labels from `chinese-rap-lyrical-repertoire-graph-v2`. Only clean-text chunks retained by the graph's shared-clean-text exclusion sensitivity are used. Lines are defined by the cleaned source's written newline boundaries. The primary estimand requires the final non-space/non-punctuation written character itself to be Han; lines ending in Latin letters, digits, emoji, or other non-Han symbols are counted and excluded. Empty lines, explicit section/credit headers, lines without Han characters, and unclassifiable endings are also excluded.

Repeated written lines are retained in their original positions. Prediction events are created only when two retained lines were adjacent in the original same chunk (`current line index = previous line index + 1`). An excluded line or a chunk boundary breaks the sequence, so no synthetic bridged transition is formed. Exact within-song repeats are flagged, and non-repeat-target metrics are released as a sensitivity stratum without reconnecting the surrounding lines.

## Written-ending representation

`pypinyin 0.55.0` is run on the complete Han sequence of each line with `Style.FINALS_TONE3`, `strict=True`, and neutral tone 5 so phrase context can disambiguate some polyphones. The final and penultimate Han syllables are stored as strict pinyin finals, tones, a tone-free final family, and a two-syllable family pattern. The {len(RHYME_FAMILIES)} deterministic families are listed in `rhyme_class_inventory.csv`. They are transparent analytical normalisations, not an official or performance-sensitive rhyme standard.

## Descriptive fingerprints

For each source-credit label, the release reports dominant and distinctive ending families, raw finals and tones, adjacent continuation/switch rates, membership in same-family runs, four-line local echo, its independent-draw frequency expectation, and a two-syllable pattern echo statistic. These are corpus summaries, not intrinsic artist traits.

## Song-held-out prediction

Songs, including songs attached to more than one source-credit label, are assigned globally to train/validation/test partitions at approximately 70/15/15. A deterministic multilabel-aware greedy allocation plus repair guarantees every eligible label has at least one song in every partition. No song crosses partitions. The target is the ending family of the exact next originally adjacent strict-Han-ending written line; features include up to four preceding contiguous families, previous raw final/tone, prior two-syllable pattern, previous transition, current same-family run bucket, contiguous-segment position bucket, recent family diversity, and source-credit label.

Five systems are compared: (1) add-one global target frequency; (2) a first-order Markov model with a global empirical-Bayes backoff of strength 20; (3) flat one-hot context features with multinomial `SGDClassifier(loss='log_loss', average=True)`; (4) the released hierarchical context model, which combines a binary continue-versus-switch classifier with a switch-only family classifier that cannot recommend the previous family in its switch branch; and (5) an otherwise identical hierarchical ablation with the source-credit-label feature removed. The label-conditioned model is not called personalised unless paired held-out intervals against this ablation support that claim. SGD regularisation is selected from 1e-5, 1e-4, and 1e-3 using validation MRR, and probability temperature is selected using validation negative log likelihood.

## Duplicate leakage control

All normalised train written lines, including isolated/first lines that are not prediction targets, form the validation leakage reference. Test targets are checked against the union of all retained train and validation written lines. Validation/test events are excluded if their target line exactly matches the applicable reference or has character-trigram Jaccard similarity >= {NEAR_DUPLICATE_JACCARD:.2f}; candidates are found by an exact inverted index and verified with exact Jaccard. Lines shorter than three normalised characters receive the exact check only. Models never receive lyric text, but this target filter prevents copied/reissued written lines from inflating either validation or final test scores.

## Evaluation and uncertainty

Primary metrics are top-1/top-3/top-5 accuracy, mean reciprocal rank (MRR), negative log likelihood, ten-bin expected calibration error, and two explicit coverage denominators: model coverage among leakage-safe events and end-to-end coverage among all strict-primary test candidates before leakage filtering. Ninety-five-percent intervals for top-k accuracy and MRR use {BOOTSTRAP_REPLICATES} resamples of held-out songs, preserving within-song dependence. All {len(nodes)} labels receive a row for every model; metrics are suppressed below {MIN_PER_LABEL_EVAL_EVENTS} leakage-safe test events rather than silently dropping labels. Stratified results separate continuation/switch, prior run length, segment position, and repeated/non-repeated targets. The independent-draw expectation for local four-line echo uses the actual available predecessor window of 1, 2, 3, or 4 at each position rather than assuming four predecessors for opening lines.

## Abstention and recommendation

Selective thresholds are fixed from validation-confidence quantiles targeting 75%, 50%, and 25% coverage, then applied once to test. `recommender_lookup.json` releases global, Markov, and common aggregate ML contexts only. It contains no lyric line, song/chunk ID, or content hash. A recommendation is a likely next **class**, not a generated word, a quality score, or a performance prescription.

## MuChin auxiliary agreement check

MuChin V1 1000 is never used for training, model selection, or primary evaluation. The publisher-recommended 1,000 folders are used instead of the duplicate-contaminated combined release. For exactly line/section-aligned `str_lyric`/`str_rhyme` pairs, a line is marked by our rule if another line in the same section has the same derived family, and this is compared with exported `R` markers. This check is only partially independent: MuChin's UI automatically grouped and colour-highlighted line-final pinyin rhymes, while annotators mainly checked polyphonic pronunciations; exported `R` collapses class identity. The result is therefore an implementation sanity/agreement check, not independent gold validation, Chinese-rap domain validation, or audio/performance validation.
"""
    atomic_write_text(PUBLIC_DIR / "METHOD.md", method_text)
    readme_text = """# Chinese written-rhyme V1

This release turns strict Han-ending Chinese-rap written lyric lines into interpretable ending-family fingerprints and a song-held-out exact-next-adjacent-line recommender. Repeated lines remain in sequence; excluded lines and chunk boundaries never create bridged transitions.

Start with:

- `analysis_summary.json` for the result and all claim boundaries;
- `model_metrics.csv` and `paired_model_deltas.csv` for baseline-versus-context-model evaluation with song-bootstrap intervals;
- `label_written_rhyme_summary.csv` / `label_written_rhyme_fingerprints.json` for public-safe source-credit-label summaries;
- `recommender_lookup.json` for text-free aggregate next-class recommendations;
- `METHOD.md` for the complete protocol;
- `muchin_auxiliary_agreement.json` for the explicitly partially circular external sanity check.

The release does **not** contain lyrics, full written lines, song/chunk IDs, or content hashes. It does not claim performed rhyme, flow, cadence, audio style, or verified artist identity.
"""
    atomic_write_text(PUBLIC_DIR / "README.md", readme_text)

    # Private reproducibility/audit rows contain identifiers and content hashes,
    # but never full lines or lyric text.
    line_audit_rows: list[dict[str, Any]] = []
    for (label_id, song_id, chunk_id), sequence in sequences.items():
        for row in sequence:
            line_audit_rows.append(
                {
                    "artist_label_id": label_id,
                    "song_id": song_id,
                    "chunk_id": chunk_id,
                    "split": song_split[song_id],
                    "sequence_index": row["sequence_index"],
                    "chunk_line_index": row["chunk_line_index"],
                    "line_hash": row["line_hash"],
                    "analysis_text_sha256": row["analysis_text_sha256"],
                    "han_character_count": row["han_character_count"],
                    "raw_final": row["raw_final"],
                    "tone": row["tone"],
                    "final_family": row["final_family"],
                    "two_final_pattern": row["two_final_pattern"],
                    "two_family_pattern": row["two_family_pattern"],
                    "within_label_song_line_multiplicity": row["within_label_song_line_multiplicity"],
                    "first_occurrence_within_label_song": row["first_occurrence_within_label_song"],
                }
            )
    atomic_write_csv(
        PRIVATE_DIR / "line_feature_audit.csv",
        line_audit_rows,
        [
            "artist_label_id", "song_id", "chunk_id", "split", "sequence_index", "chunk_line_index", "line_hash",
            "analysis_text_sha256", "han_character_count", "raw_final", "tone", "final_family", "two_final_pattern",
            "two_family_pattern", "within_label_song_line_multiplicity", "first_occurrence_within_label_song",
        ],
    )
    event_audit_rows = [
        {
            "artist_label_id": event["artist_label_id"],
            "song_id": event["song_id"],
            "chunk_id": event["chunk_id"],
            "split": event["split"],
            "sequence_index": event["sequence_index"],
            "chunk_line_index": event["chunk_line_index"],
            "previous_chunk_line_index": event["previous_chunk_line_index"],
            "line_hash": event["line_hash"],
            "target_family": event["target_family"],
            "previous_family": event["previous_family"],
            "transition_type": event["transition_type"],
            "prior_run_bucket": event["prior_run_bucket"],
            "position_bucket": event["position_bucket"],
            "target_is_repeat_within_label_song": event["target_is_repeat_within_label_song"],
            "leakage_status": event["leakage_status"],
            "maximum_reference_line_trigram_jaccard": round_float(event.get("maximum_reference_line_trigram_jaccard", 0.0)),
        }
        for event in events
    ]
    atomic_write_csv(
        PRIVATE_DIR / "transition_split_audit.csv",
        event_audit_rows,
        [
            "artist_label_id", "song_id", "chunk_id", "split", "sequence_index", "chunk_line_index",
            "previous_chunk_line_index", "line_hash",
            "target_family", "previous_family", "transition_type", "prior_run_bucket", "position_bucket",
            "target_is_repeat_within_label_song", "leakage_status", "maximum_reference_line_trigram_jaccard",
        ],
    )
    atomic_write_json(PRIVATE_DIR / "split_audit.json", split_audit)
    atomic_write_json(PRIVATE_DIR / "leakage_audit.json", leakage_audit)
    atomic_write_json(PRIVATE_DIR / "muchin_auxiliary_agreement_audit.json", auxiliary)

    public_files_pre_manifest = sorted(path for path in PUBLIC_DIR.iterdir() if path.is_file())
    validation = validate_public_payloads(public_files_pre_manifest)
    validation["artifact_id"] = ARTIFACT_ID
    validation["version"] = VERSION
    validation["generated_at_utc"] = utc_now()
    validation["structural_checks"] = {
        "configured_rhyme_family_count": len(RHYME_FAMILIES),
        "graph_eligible_label_count": len(nodes),
        "fingerprint_count": len(fingerprints),
        "strict_song_partition_overlap_zero": all(value == 0 for value in split_audit["cross_partition_song_overlap"].values()),
        "all_labels_have_train_validation_test": split_audit["labels_with_nonempty_all_partitions"] == len(nodes),
        "test_event_count_matches_model_audit": len(test_events) == model_audit["leakage_safe_test_event_count"],
        "all_primary_models_have_full_coverage": all(
            row.get("model_coverage_on_leakage_safe_events") == 1.0
            for row in metric_rows if row["evaluation_split"] == "song_held_out_test"
        ),
        "all_204_labels_have_complete_rows_for_every_primary_model": len(per_label_rows) == 5 * len(nodes),
        "all_transitions_are_between_originally_adjacent_lines_in_one_chunk": all(
            event["chunk_line_index"] == event["previous_chunk_line_index"] + 1 for event in events
        ),
        "test_leakage_filter_uses_train_and_validation_reference": all(
            event["leakage_status"] in {
                "eligible_no_train_or_validation_duplicate",
                "exact_train_or_validation_line_duplicate",
                "near_train_or_validation_line_duplicate",
            }
            for event in events if event["split"] == "test"
        ),
        "bootstrap_replicates_at_least_2000": BOOTSTRAP_REPLICATES >= 2000,
        "released_recommender_model_matches_selected_primary_model": (
            lookup["released_model"]["name"] == "hierarchical_sgd_context"
            and lookup["released_model"]["selected_alpha"] == model_audit["hierarchical_context_selected_alpha"]
            and lookup["released_model"]["selected_validation_temperature"]
            == model_audit["hierarchical_context_selected_validation_temperature"]
        ),
        "public_output_contains_no_lyrics_or_private_identifiers": validation["status"] == "passed",
    }
    if not all(value for value in validation["structural_checks"].values()):
        validation["status"] = "failed"
    atomic_write_json(PUBLIC_DIR / "validation.json", validation)

    private_files = sorted(path for path in PRIVATE_DIR.iterdir() if path.is_file())
    private_validation = {
        "artifact_id": ARTIFACT_ID,
        "status": "passed" if validation["status"] == "passed" else "failed",
        "counts": {
            "line_audit_rows": len(line_audit_rows),
            "transition_audit_rows": len(event_audit_rows),
            "train_events": len(safe_events(events, "train")),
            "validation_events_after_leakage_filter": len(validation_events),
            "test_events_after_leakage_filter": len(test_events),
        },
        "full_lyric_text_present": False,
        "private_identifiers_present_by_design": ["song_id", "line_hash", "analysis_text_sha256"],
    }
    atomic_write_json(PRIVATE_DIR / "private_validation.json", private_validation)

    public_files = sorted(path for path in PUBLIC_DIR.iterdir() if path.is_file() and path.name != "manifest.json")
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "classification": "public-safe aggregate written-rhyme analysis",
        "claim_boundary": analysis_summary["claim_boundary"],
        "input_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (NODE_FILE, MEMBERSHIP_FILE, CLEAN_TEXT_FILE)
        },
        "builder": {
            "path": "work/build_chinese_rap_written_rhyme_v1.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "dependencies": dependency_fingerprints(),
        "output_files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in public_files
        },
    }
    atomic_write_json(PUBLIC_DIR / "manifest.json", manifest)
    private_files = sorted(path for path in PRIVATE_DIR.iterdir() if path.is_file() and path.name != "private_manifest.json")
    private_manifest = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "classification": "private local-only identifiers and hashed line audit; no full lyric text",
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "dependencies": dependency_fingerprints(),
        "files": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in private_files},
    }
    atomic_write_json(PRIVATE_DIR / "private_manifest.json", private_manifest)
    print(json.dumps({
        "artifact_id": ARTIFACT_ID,
        "status": validation["status"],
        "public_dir": str(PUBLIC_DIR),
        "private_dir": str(PRIVATE_DIR),
        "retained_lines": len(line_audit_rows),
        "test_events": len(test_events),
    }, ensure_ascii=False, indent=2))


def validate_existing() -> None:
    validation_path = PUBLIC_DIR / "validation.json"
    manifest_path = PUBLIC_DIR / "manifest.json"
    if not validation_path.exists() or not manifest_path.exists():
        raise RuntimeError("Build outputs before validation")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for name, metadata in manifest["output_files"].items():
        path = PUBLIC_DIR / name
        if not path.exists() or sha256_file(path) != metadata["sha256"]:
            mismatches.append(name)
    if validation.get("status") != "passed" or mismatches:
        raise RuntimeError(f"Validation failed; hash mismatches={mismatches}")
    print(json.dumps({"status": "passed", "manifest_files_checked": len(manifest["output_files"])}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate", "all"), nargs="?", default="all")
    args = parser.parse_args()
    if args.command in ("build", "all"):
        build()
    if args.command in ("validate", "all"):
        validate_existing()


if __name__ == "__main__":
    main()
