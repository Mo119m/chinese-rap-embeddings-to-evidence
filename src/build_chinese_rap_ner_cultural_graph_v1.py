#!/usr/bin/env python3
"""Build an auditable Chinese-rap NER candidate study and provisional network.

The public artifact contains aggregate evidence only.  Occurrence locators,
candidate contexts, and annotation forms stay in a private work directory.
No output is called gold unless two reviewers and adjudication have actually
been completed (the current corpus has no completed occurrence gold labels).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import regex
import torch
from transformers import AutoModelForTokenClassification, BertTokenizerFast
from scipy.stats import beta as beta_distribution
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = ROOT / "work/private-canonical-lyric-text-sidecar-v1/cleaned_analysis_chunks_v1.csv"
DEFAULT_SONGS = ROOT / "work/private-canonical-analysis-input-v1/canonical_analysis_songs_v1.csv"
DEFAULT_LEXICON = ROOT / "outputs/chinese-rap-curated-atlas-v3/safe_lexicon_catalog.csv"
DEFAULT_CORE_LEDGER = ROOT / "outputs/chinese-rap-ner-reference-ledger-v1/core_reference_ledger_v1.csv"
DEFAULT_GRAPH_LABELS = ROOT / "outputs/chinese-rap-lyrical-repertoire-graph-v2/artist_repertoire_nodes.csv"
DEFAULT_MODEL = ROOT / "work/private-ner-poc-cache/ckiplab-albert-tiny-chinese-ner-bcb5198"
DEFAULT_PUBLIC = ROOT / "outputs/chinese-rap-ner-cultural-graph-v1"
DEFAULT_PRIVATE = ROOT / "work/private-chinese-rap-ner-cultural-graph-v1"

ARTIFACT_ID = "chinese-rap-ner-cultural-graph-v1"
VERSION = "1.1.0"
SAMPLING_SEED = "chinese-rap-ner-cultural-graph-v1-audit-20260825"


LEXICON_TYPE_MAP = {
    "PERSON": "PERSON_REFERENCE",
    "ARTIST": "PERSON_REFERENCE",
    "ORG": "GROUP_CREW_OR_ORGANIZATION",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "FAC": "PLACE",
    "HOM": "PLACE",
    "PRODUCT": "BRAND_OR_PRODUCT",
    "CAR": "BRAND_OR_PRODUCT",
    "FASHION": "BRAND_OR_PRODUCT",
    "SUBSTANCES": "BRAND_OR_PRODUCT",
    "FOOD": "BRAND_OR_PRODUCT",
    "WORK_OF_ART": "WORK_OR_MEDIA",
    "LAW": "WORK_OR_MEDIA",
    "PLATFORM": "WORK_OR_MEDIA",
    "EVENT": "EVENT",
    "LANGUAGE": "LANGUAGE_OR_DIALECT_REFERENCE",
    "NORP": "ETHNOCULTURAL_GROUP_REFERENCE",
    "HHNL": "RAP_CULTURE_CONCEPT",
    "CULTURE": "RAP_CULTURE_CONCEPT",
    "REF": "OTHER_CULTURAL_REFERENCE",
    "TITLE": "OTHER_CULTURAL_REFERENCE",
    "ANIMAL": "OTHER_CULTURAL_REFERENCE",
}

MODEL_TYPE_MAP = {
    "PERSON": "PERSON_REFERENCE",
    "ORG": "GROUP_CREW_OR_ORGANIZATION",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "FAC": "PLACE",
    "PRODUCT": "BRAND_OR_PRODUCT",
    "WORK_OF_ART": "WORK_OR_MEDIA",
    "LAW": "WORK_OR_MEDIA",
    "EVENT": "EVENT",
    "LANGUAGE": "LANGUAGE_OR_DIALECT_REFERENCE",
    "NORP": "ETHNOCULTURAL_GROUP_REFERENCE",
}

PUBLIC_ENTITY_TYPES = {
    "PERSON_REFERENCE",
    "GROUP_CREW_OR_ORGANIZATION",
    "PLACE",
    "BRAND_OR_PRODUCT",
    "WORK_OR_MEDIA",
    "EVENT",
    "LANGUAGE_OR_DIALECT_REFERENCE",
    "ETHNOCULTURAL_GROUP_REFERENCE",
}

ANNOTATION_ENTITY_TYPES = [
    "PERSON_REFERENCE",
    "GROUP_CREW_OR_ORGANIZATION",
    "PLACE",
    "BRAND_OR_PRODUCT",
    "WORK_OR_MEDIA",
    "EVENT",
    "LANGUAGE_OR_DIALECT_REFERENCE",
    "ETHNOCULTURAL_GROUP_REFERENCE",
    "RAP_CULTURE_CONCEPT",
    "OTHER_CULTURAL_REFERENCE",
    "NOT_ENTITY",
    "UNCERTAIN",
]

# Surface-level abstentions discovered during the 48-pair release audit.  These
# strings can denote multiple or figurative places; without occurrence gold,
# a public place claim would be stronger than the evidence.
AMBIGUOUS_PUBLIC_SURFACE_DENYLIST = {
    ("中南海", "PLACE"),
    ("桃源", "PLACE"),
    ("西山", "PLACE"),
}

AMBIGUOUS_PUBLIC_SURFACE_REASONS = {
    ("中南海", "PLACE"): "WITHHOLD: the surface may denote a place, a political metonym, or a product brand; no occurrence gold resolves those senses.",
    ("桃源", "PLACE"): "WITHHOLD: the surface may be a literal place name or a figurative utopia; no occurrence gold resolves those senses.",
    ("西山", "PLACE"): "WITHHOLD: the surface names multiple places and can be a generic mountain reference; no occurrence gold resolves those senses.",
}

REVIEW_FIELDS = [
    "mention_valid",
    "entity_type",
    "referential_status",
    "normalized_surface",
    "linking_status",
    "resolved_entity_id",
    "confidence_1_to_5",
    "notes",
    "reviewer",
    "reviewed_at_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--songs", type=Path, default=DEFAULT_SONGS)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--core-ledger", type=Path, default=DEFAULT_CORE_LEDGER)
    parser.add_argument("--graph-labels", type=Path, default=DEFAULT_GRAPH_LABELS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--public-output", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private-output", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--audit-tasks", type=int, default=800)
    parser.add_argument("--background-lines", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\0".join(str(part) for part in parts)
    return f"{prefix}-{sha256_text(payload)[:24]}"


def reset_output_dir(path: Path, required_leaf: str) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved.name != required_leaf or root not in resolved.parents:
        raise ValueError(f"Refusing to reset unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=False)


def write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=convert) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if columns is not None:
        for column in columns:
            if column not in df.columns:
                df[column] = ""
        df = df[columns]
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return df


def ascii_word_char(value: str) -> bool:
    return bool(value) and bool(re.fullmatch(r"[A-Za-z0-9_]", value))


def boundary_ok(text: str, start: int, end: int, surface: str) -> bool:
    if not re.search(r"[A-Za-z0-9_]", surface):
        return True
    left = text[start - 1] if start else ""
    right = text[end] if end < len(text) else ""
    return not ascii_word_char(left) and not ascii_word_char(right)


def nonoverlap_longest_first(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for item in sorted(items, key=lambda row: (-(row["end_char"] - row["start_char"]), row["start_char"], row["surface"])):
        span = (int(item["start_char"]), int(item["end_char"]))
        if any(max(span[0], old[0]) < min(span[1], old[1]) for old in occupied):
            continue
        selected.append(item)
        occupied.append(span)
    return sorted(selected, key=lambda row: (row["start_char"], row["end_char"], row["surface"]))


def iter_lines_with_offsets(text: str) -> Iterable[tuple[int, int, str]]:
    start = 0
    for line in text.splitlines(keepends=True):
        clean = line.rstrip("\r\n")
        end = start + len(clean)
        if clean.strip():
            yield start, end, clean
        start += len(line)
    if text and not text.endswith(("\n", "\r")) and "\n" not in text and "\r" not in text:
        return


def line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, start) + 1
    next_newline = text.find("\n", end)
    line_end = len(text) if next_newline < 0 else next_newline
    if line_end > line_start and text[line_end - 1] == "\r":
        line_end -= 1
    return line_start, line_end


def compile_lexicon(lexicon: pd.DataFrame) -> tuple[regex.Pattern, dict[str, dict[str, str]]]:
    records: dict[str, dict[str, str]] = {}
    for row in lexicon.to_dict("records"):
        surface = str(row["entity"]).strip()
        effective = str(row["effective_label"]).strip()
        schema_type = LEXICON_TYPE_MAP.get(effective)
        if not surface or not schema_type or len(surface) < 2:
            continue
        if surface in records:
            raise ValueError(f"Safe lexicon surface is not unique: {surface}")
        records[surface] = {"lexicon_label": effective, "schema_type": schema_type}
    alternatives = sorted(records, key=lambda value: (-len(value), value))
    pattern = regex.compile("(?:" + "|".join(regex.escape(value) for value in alternatives) + ")")
    return pattern, records


def rank_key(*parts: Any) -> str:
    return sha256_text(SAMPLING_SEED + "\0" + "\0".join(str(part) for part in parts))


def load_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (
        args.chunks,
        args.songs,
        args.lexicon,
        args.core_ledger,
        args.graph_labels,
        args.model / "config.json",
        args.model / "pytorch_model.bin",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    chunks = pd.read_csv(
        args.chunks,
        dtype={"song_id": str, "chunk_id": str, "analysis_text_sha256": str},
        keep_default_na=False,
        usecols=[
            "song_id",
            "chunk_id",
            "analysis_text",
            "analysis_text_sha256",
            "analysis_text_status",
            "analysis_text_weight",
        ],
    )
    chunks = chunks[(chunks["analysis_text_status"] == "eligible_clean_text") & (chunks["analysis_text"].str.strip() != "")].copy()
    if chunks.duplicated(["song_id", "chunk_id"]).any():
        raise ValueError("Chunk key is not unique")

    songs = pd.read_csv(
        args.songs,
        dtype=str,
        keep_default_na=False,
        usecols=["song_id", "canonical_artist", "song_lyric_content_sha256", "downstream_eligibility"],
    )
    songs = songs[songs["downstream_eligibility"] == "eligible"].copy()
    if songs.duplicated("song_id").any():
        raise ValueError("Song key is not unique")
    chunks = chunks.merge(songs, on="song_id", how="inner", validate="many_to_one")
    chunks["analysis_text_weight"] = pd.to_numeric(chunks["analysis_text_weight"], errors="coerce").fillna(0.0)
    chunks = chunks[chunks["analysis_text_weight"] > 0].copy()

    lexicon = pd.read_csv(args.lexicon, dtype=str, keep_default_na=False)
    required = {"effective_label", "entity", "canonical_category"}
    if not required.issubset(lexicon.columns):
        raise ValueError(f"Lexicon missing columns: {sorted(required - set(lexicon.columns))}")
    core_ledger = pd.read_csv(args.core_ledger, dtype=str, keep_default_na=False)
    core_required = {"entity", "canonical_category", "evidence_tier"}
    if not core_required.issubset(core_ledger.columns):
        raise ValueError(f"Core ledger missing columns: {sorted(core_required - set(core_ledger.columns))}")
    graph_labels = pd.read_csv(args.graph_labels, dtype=str, keep_default_na=False)
    graph_required = {"artist_label_id", "source_artist_label", "graph_node_eligible"}
    if not graph_required.issubset(graph_labels.columns):
        raise ValueError(f"Graph label registry missing columns: {sorted(graph_required - set(graph_labels.columns))}")
    graph_labels = graph_labels[graph_labels["graph_node_eligible"].str.lower().eq("true")][
        ["artist_label_id", "source_artist_label"]
    ].copy()
    if len(graph_labels) != 204 or graph_labels["artist_label_id"].nunique() != 204 or graph_labels["source_artist_label"].nunique() != 204:
        raise ValueError("Expected exactly 204 unique graph-eligible source-credit labels")
    if not set(graph_labels["source_artist_label"]).issubset(set(songs["canonical_artist"])):
        raise ValueError("Graph-eligible label registry contains labels absent from eligible songs")
    return chunks, songs, lexicon, core_ledger, graph_labels


def build_lexicon_frame(
    chunks: pd.DataFrame,
    pattern: regex.Pattern,
    lexicon_records: dict[str, dict[str, str]],
    background_target: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    mentions: list[dict[str, Any]] = []
    candidate_line_instances: dict[str, dict[str, Any]] = {}
    background_by_text_hash: dict[str, dict[str, Any]] = {}

    for row in chunks.itertuples(index=False):
        text = str(row.analysis_text)
        raw: list[dict[str, Any]] = []
        for match in pattern.finditer(text, overlapped=True):
            surface = match.group(0)
            start, end = match.span()
            if boundary_ok(text, start, end, surface):
                metadata = lexicon_records[surface]
                raw.append(
                    {
                        "surface": surface,
                        "start_char": start,
                        "end_char": end,
                        "lexicon_label": metadata["lexicon_label"],
                        "schema_type": metadata["schema_type"],
                    }
                )
        selected = nonoverlap_longest_first(raw)
        line_has_candidate: set[tuple[int, int]] = set()
        for item in selected:
            ls, le = line_bounds(text, int(item["start_char"]), int(item["end_char"]))
            line_text = text[ls:le]
            line_text_hash = sha256_text(line_text)
            instance_id = stable_id("LINE", row.analysis_text_sha256, ls, le)
            line_has_candidate.add((ls, le))
            instance = {
                "line_instance_id": instance_id,
                "line_text_hash": line_text_hash,
                "line_text": line_text,
                "line_start_char": ls,
                "line_end_char": le,
                "song_id": row.song_id,
                "chunk_id": row.chunk_id,
                "analysis_text_sha256": row.analysis_text_sha256,
                "analysis_text_weight": float(row.analysis_text_weight),
                "source_credit_label": row.canonical_artist,
                "song_lyric_content_sha256": row.song_lyric_content_sha256,
            }
            candidate_line_instances[instance_id] = instance
            mentions.append(
                {
                    **instance,
                    **item,
                    "line_start_local": int(item["start_char"]) - ls,
                    "line_end_local": int(item["end_char"]) - ls,
                }
            )

        for ls, le, line_text in iter_lines_with_offsets(text):
            if (ls, le) in line_has_candidate:
                continue
            stripped = line_text.strip()
            if len(stripped) < 2:
                continue
            text_hash = sha256_text(line_text)
            current = background_by_text_hash.get(text_hash)
            candidate = {
                "line_instance_id": stable_id("LINE", row.analysis_text_sha256, ls, le),
                "line_text_hash": text_hash,
                "line_text": line_text,
                "line_start_char": ls,
                "line_end_char": le,
                "song_id": row.song_id,
                "chunk_id": row.chunk_id,
                "analysis_text_sha256": row.analysis_text_sha256,
                "analysis_text_weight": float(row.analysis_text_weight),
                "source_credit_label": row.canonical_artist,
                "song_lyric_content_sha256": row.song_lyric_content_sha256,
                "selection_rank": rank_key("background", text_hash),
            }
            if current is None or candidate["line_instance_id"] < current["line_instance_id"]:
                background_by_text_hash[text_hash] = candidate

    background_selected = dict(
        sorted(background_by_text_hash.items(), key=lambda pair: (pair[1]["selection_rank"], pair[0]))[:background_target]
    )
    return pd.DataFrame(mentions), candidate_line_instances, background_selected


def make_windows(text: str, window_chars: int = 180, overlap_chars: int = 40) -> list[tuple[int, str]]:
    if len(text) <= window_chars:
        return [(0, text)]
    windows: list[tuple[int, str]] = []
    start = 0
    stride = window_chars - overlap_chars
    while start < len(text):
        windows.append((start, text[start : start + window_chars]))
        if start + window_chars >= len(text):
            break
        start += stride
    return windows


def decode_bies(
    text: str,
    labels: list[str],
    offsets: list[tuple[int, int]],
    confidences: list[float],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def close() -> None:
        nonlocal current
        if current is not None and current["end"] > current["start"]:
            current["surface"] = text[current["start"] : current["end"]]
            current["confidence"] = float(np.mean(current.pop("token_confidences")))
            spans.append(current)
        current = None

    for label, (start, end), confidence in zip(labels, offsets, confidences):
        if start == end or label == "O":
            close()
            continue
        if "-" not in label:
            close()
            continue
        prefix, raw_type = label.split("-", 1)
        schema_type = MODEL_TYPE_MAP.get(raw_type)
        if schema_type is None:
            close()
            continue
        if prefix == "S":
            close()
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "surface": text[start:end],
                    "model_label": raw_type,
                    "schema_type": schema_type,
                    "confidence": float(confidence),
                }
            )
        elif prefix == "B":
            close()
            current = {
                "start": start,
                "end": end,
                "model_label": raw_type,
                "schema_type": schema_type,
                "token_confidences": [float(confidence)],
            }
        elif prefix in {"I", "E"} and current is not None and current["model_label"] == raw_type:
            current["end"] = end
            current["token_confidences"].append(float(confidence))
            if prefix == "E":
                close()
        else:
            close()
            current = {
                "start": start,
                "end": end,
                "model_label": raw_type,
                "schema_type": schema_type,
                "token_confidences": [float(confidence)],
            }
            if prefix == "E":
                close()
    close()
    return [span for span in spans if len(span["surface"].strip()) >= 2]


def run_transformer(
    line_texts: dict[str, str],
    model_path: Path,
    batch_size: int,
    torch_threads: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch.set_num_threads(max(1, torch_threads))
    tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()

    windows: list[tuple[str, int, str]] = []
    for line_hash, text in sorted(line_texts.items()):
        for start, window in make_windows(text):
            windows.append((line_hash, start, window))

    candidate_map: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for batch_start in range(0, len(windows), batch_size):
        batch = windows[batch_start : batch_start + batch_size]
        texts = [item[2] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1).values
        for idx, (line_hash, window_start, window_text) in enumerate(batch):
            labels = [model.config.id2label[int(value)] for value in prediction[idx].tolist()]
            decoded = decode_bies(
                window_text,
                labels,
                [tuple(map(int, value)) for value in offsets[idx].tolist()],
                [float(value) for value in confidence[idx].tolist()],
            )
            for item in decoded:
                start = int(item["start"]) + window_start
                end = int(item["end"]) + window_start
                key = (line_hash, start, end, item["model_label"])
                row = {
                    "line_text_hash": line_hash,
                    "start_local": start,
                    "end_local": end,
                    "surface": line_texts[line_hash][start:end],
                    "model_label": item["model_label"],
                    "schema_type": item["schema_type"],
                    "confidence": round(float(item["confidence"]), 6),
                }
                if key not in candidate_map or row["confidence"] > candidate_map[key]["confidence"]:
                    candidate_map[key] = row
    candidates = pd.DataFrame(sorted(candidate_map.values(), key=lambda row: (row["line_text_hash"], row["start_local"], row["end_local"], row["model_label"])))
    provenance = {
        "model_directory": model_path.name,
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "model_weights_sha256": sha256_file(model_path / "pytorch_model.bin"),
        "tokenizer_vocab_sha256": sha256_file(model_path / "vocab.txt"),
        "model_labels": sorted(set(MODEL_TYPE_MAP)),
        "window_chars": 180,
        "overlap_chars": 40,
        "max_tokens": 256,
        "batch_size": batch_size,
        "torch_threads": torch_threads,
    }
    return candidates, provenance


def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union else 0.0


def compare_candidates(lex_mentions: pd.DataFrame, model_candidates: pd.DataFrame) -> pd.DataFrame:
    by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_candidates.to_dict("records"):
        by_line[row["line_text_hash"]].append(row)
    rows: list[dict[str, Any]] = []
    for lex in lex_mentions.to_dict("records"):
        candidates = []
        for model in by_line.get(lex["line_text_hash"], []):
            iou = span_iou(int(lex["line_start_local"]), int(lex["line_end_local"]), int(model["start_local"]), int(model["end_local"]))
            if iou > 0:
                exact = int(lex["line_start_local"]) == int(model["start_local"]) and int(lex["line_end_local"]) == int(model["end_local"])
                type_agree = lex["schema_type"] == model["schema_type"]
                candidates.append((int(exact and type_agree), int(type_agree), exact, iou, float(model["confidence"]), model))
        if candidates:
            _, type_agree_int, exact, iou, confidence, model = max(candidates, key=lambda value: value[:5])
            type_agree = bool(type_agree_int)
            if exact and type_agree:
                state = "EXACT_SPAN_TYPE_AGREE"
            elif iou >= 0.5 and type_agree:
                state = "OVERLAP_TYPE_AGREE"
            else:
                state = "TYPE_OR_BOUNDARY_CONFLICT"
            model_surface = model["surface"]
            model_label = model["model_label"]
            model_schema = model["schema_type"]
        else:
            state = "LEXICON_ONLY"
            iou = 0.0
            confidence = 0.0
            model_surface = ""
            model_label = ""
            model_schema = ""
        rows.append(
            {
                **lex,
                "transformer_surface": model_surface,
                "transformer_label": model_label,
                "transformer_schema_type": model_schema,
                "transformer_confidence": round(float(confidence), 6),
                "span_iou": round(float(iou), 6),
                "agreement_state": state,
                "strict_high_consistency": bool(state == "EXACT_SPAN_TYPE_AGREE" and float(confidence) >= 0.80),
            }
        )
    return pd.DataFrame(rows)


def materialize_model_occurrences(
    line_instances: dict[str, dict[str, Any]],
    model_candidates: pd.DataFrame,
) -> pd.DataFrame:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_candidates.to_dict("records"):
        by_hash[row["line_text_hash"]].append(row)
    rows: list[dict[str, Any]] = []
    for instance in line_instances.values():
        for model in by_hash.get(instance["line_text_hash"], []):
            rows.append(
                {
                    **instance,
                    "surface": model["surface"],
                    "start_char": int(instance["line_start_char"]) + int(model["start_local"]),
                    "end_char": int(instance["line_start_char"]) + int(model["end_local"]),
                    "line_start_local": int(model["start_local"]),
                    "line_end_local": int(model["end_local"]),
                    "model_label": model["model_label"],
                    "schema_type": model["schema_type"],
                    "transformer_confidence": model["confidence"],
                }
            )
    return pd.DataFrame(rows)


def make_private_union(agreements: pd.DataFrame, model_occurrences: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    overlap_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for lex in agreements.to_dict("records"):
        overlap_index[lex["line_instance_id"]].append((int(lex["start_char"]), int(lex["end_char"])))
        rows.append(
            {
                **lex,
                "candidate_source": "LEXICON_WITH_TRANSFORMER_CHECK",
                "candidate_surface": lex["surface"],
                "candidate_schema_type": lex["schema_type"],
            }
        )
    for model in model_occurrences.to_dict("records"):
        spans = overlap_index.get(model["line_instance_id"], [])
        if any(max(int(model["start_char"]), start) < min(int(model["end_char"]), end) for start, end in spans):
            continue
        rows.append(
            {
                **model,
                "lexicon_label": "",
                "transformer_surface": model["surface"],
                "transformer_label": model["model_label"],
                "transformer_schema_type": model["schema_type"],
                "span_iou": 0.0,
                "agreement_state": "TRANSFORMER_ONLY",
                "strict_high_consistency": False,
                "candidate_source": "TRANSFORMER_ONLY",
                "candidate_surface": model["surface"],
                "candidate_schema_type": model["schema_type"],
            }
        )
    union = pd.DataFrame(rows)
    union["candidate_start_char"] = pd.to_numeric(union["start_char"], errors="coerce").astype(int)
    union["candidate_end_char"] = pd.to_numeric(union["end_char"], errors="coerce").astype(int)
    union["candidate_id"] = [
        stable_id(
            "CAND",
            row.song_id,
            row.chunk_id,
            row.analysis_text_sha256,
            row.candidate_start_char,
            row.candidate_end_char,
            row.candidate_surface,
            row.candidate_schema_type,
        )
        for row in union.itertuples(index=False)
    ]
    union = union.sort_values(["candidate_id", "agreement_state"]).drop_duplicates("candidate_id", keep="first").reset_index(drop=True)
    return union


def add_context(union: pd.DataFrame, chunks: pd.DataFrame, context_chars: int = 90) -> pd.DataFrame:
    text_by_key = {(row.song_id, row.chunk_id): row.analysis_text for row in chunks.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for item in union.to_dict("records"):
        text = text_by_key[(item["song_id"], item["chunk_id"])]
        start, end = int(item["candidate_start_char"]), int(item["candidate_end_char"])
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        # Preserve the exact substring so stored offsets remain directly auditable.
        # CSV quoting safely retains embedded newlines in this private-only file.
        context = text[context_start:context_end]
        item.update(
            {
                "context_start_char": context_start,
                "context_end_char": context_end,
                "surface_start_in_context": start - context_start,
                "surface_end_in_context": end - context_start,
                "context_snippet": context,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def stratified_sample(union: pd.DataFrame, target: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = union.copy()
    frame["stratum"] = frame["candidate_schema_type"] + "__" + frame["agreement_state"]
    frame["selection_rank_sha256"] = [rank_key(row.candidate_id, row.stratum) for row in frame.itertuples(index=False)]
    selected_parts: list[pd.DataFrame] = []
    nonempty_strata = max(1, frame["stratum"].nunique())
    first_quota = max(8, target // nonempty_strata)
    chosen_ids: set[str] = set()
    for _, group in frame.groupby("stratum", sort=True):
        part = group.sort_values(["selection_rank_sha256", "candidate_id"]).head(first_quota)
        selected_parts.append(part)
        chosen_ids.update(part["candidate_id"])
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else frame.head(0)
    if len(selected) < target:
        remainder = frame[~frame["candidate_id"].isin(chosen_ids)].sort_values(["selection_rank_sha256", "candidate_id"])
        selected = pd.concat([selected, remainder.head(target - len(selected))], ignore_index=True)
    selected = selected.sort_values(["stratum", "selection_rank_sha256", "candidate_id"]).head(target).reset_index(drop=True)
    if len(selected) < target:
        raise ValueError(f"Only {len(selected)} distinct candidates available for {target} requested audit tasks")
    selected["task_id"] = [stable_id("NERANN", value) for value in selected["candidate_id"]]
    summary = (
        frame.groupby("stratum", as_index=False)
        .size()
        .rename(columns={"size": "candidate_occurrences"})
        .merge(selected.groupby("stratum", as_index=False).size().rename(columns={"size": "selected_tasks"}), on="stratum", how="left")
        .fillna({"selected_tasks": 0})
    )
    summary["selected_tasks"] = summary["selected_tasks"].astype(int)
    summary["sampling_seed"] = SAMPLING_SEED
    return selected, summary


def reviewer_template(tasks: pd.DataFrame, reviewer_slot: str) -> pd.DataFrame:
    columns = [
        "task_id",
        "stratum",
        "source_credit_label",
        "song_id",
        "chunk_id",
        "candidate_surface",
        "candidate_schema_type",
        "context_snippet",
        "surface_start_in_context",
        "surface_end_in_context",
        "lexicon_label",
        "transformer_label",
        "transformer_schema_type",
        "transformer_confidence",
        "agreement_state",
    ]
    out = tasks[columns].copy()
    out["reviewer_slot"] = reviewer_slot
    for field in REVIEW_FIELDS:
        out[field] = reviewer_slot if field == "reviewer" else ""
    return out


def agreement_template(tasks: pd.DataFrame) -> pd.DataFrame:
    out = tasks[["task_id", "candidate_surface", "candidate_schema_type", "stratum"]].copy()
    for prefix in ("r1", "r2"):
        for field in ("mention_valid", "entity_type", "referential_status", "normalized_surface", "linking_status", "resolved_entity_id"):
            out[f"{prefix}_{field}"] = ""
    out["mention_valid_exact_agreement"] = ""
    out["entity_type_exact_agreement"] = ""
    out["referential_status_exact_agreement"] = ""
    out["adjudication_required"] = ""
    out["adjudicated_mention_valid"] = ""
    out["adjudicated_entity_type"] = ""
    out["adjudicated_referential_status"] = ""
    out["adjudicated_normalized_surface"] = ""
    out["adjudicated_linking_status"] = ""
    out["adjudicated_resolved_entity_id"] = ""
    out["adjudicator"] = ""
    out["adjudication_notes"] = ""
    out["adjudicated_at_utc"] = ""
    return out


def baseline_comparison(
    lex_mentions: pd.DataFrame,
    model_candidates: pd.DataFrame,
    agreements: pd.DataFrame,
    candidate_line_hashes: set[str],
    background_line_hashes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lex_unique = lex_mentions.drop_duplicates(["line_text_hash", "line_start_local", "line_end_local", "schema_type"])
    model_unique = model_candidates.drop_duplicates(["line_text_hash", "start_local", "end_local", "schema_type"])
    agreement_unique = agreements.drop_duplicates(
        ["line_text_hash", "line_start_local", "line_end_local", "schema_type"]
    )
    compare_rows = [
        {
            "baseline": "reviewed_lexicon_exact_match",
            "candidate_spans_on_common_frame": len(lex_unique),
            "distinct_surfaces": lex_unique["surface"].nunique(),
            "entity_types": lex_unique["schema_type"].nunique(),
            "common_frame_unique_lines": len(candidate_line_hashes | background_line_hashes),
            "candidate_bearing_lines": len(candidate_line_hashes),
            "deterministic_background_lines": len(background_line_hashes),
            "evaluation_status": "candidate_baseline_no_human_gold",
        },
        {
            "baseline": "ckip_albert_tiny_general_domain_ner",
            "candidate_spans_on_common_frame": len(model_unique),
            "distinct_surfaces": model_unique["surface"].nunique() if len(model_unique) else 0,
            "entity_types": model_unique["schema_type"].nunique() if len(model_unique) else 0,
            "common_frame_unique_lines": len(candidate_line_hashes | background_line_hashes),
            "candidate_bearing_lines": len(candidate_line_hashes),
            "deterministic_background_lines": len(background_line_hashes),
            "evaluation_status": "candidate_baseline_no_human_gold",
        },
    ]
    type_rows: list[dict[str, Any]] = []
    for entity_type in sorted(set(lex_unique["schema_type"]) | set(model_unique["schema_type"])):
        lex_type = lex_unique[lex_unique["schema_type"] == entity_type]
        model_type = model_unique[model_unique["schema_type"] == entity_type]
        agree_type = agreement_unique[agreement_unique["schema_type"] == entity_type]
        exact = int((agree_type["agreement_state"] == "EXACT_SPAN_TYPE_AGREE").sum())
        high = int(agree_type["strict_high_consistency"].sum())
        type_rows.append(
            {
                "entity_type": entity_type,
                "lexicon_candidate_spans_on_unique_line_frame": len(lex_type),
                "transformer_candidate_spans_on_unique_line_frame": len(model_type),
                "exact_span_type_agreements_on_unique_line_frame": exact,
                "strict_high_consistency_spans_on_unique_line_frame": high,
                "exact_agreement_per_lexicon_candidate_span": round(exact / len(lex_type), 6) if len(lex_type) else "",
                "exact_agreement_per_transformer_candidate_span": round(exact / len(model_type), 6) if len(model_type) else "",
                "metric_boundary": "unique-line candidate agreement only; not occurrence counts, precision, recall, or F1",
            }
        )
    return pd.DataFrame(compare_rows), pd.DataFrame(type_rows)


def compute_shared_text_hash_audit(chunks: pd.DataFrame) -> tuple[set[str], pd.DataFrame]:
    grouped = (
        chunks.groupby("analysis_text_sha256", as_index=False)
        .agg(
            canonical_label_count=("canonical_artist", "nunique"),
            source_membership_rows=("analysis_text_sha256", "size"),
            distinct_song_content_units=("song_lyric_content_sha256", "nunique"),
            canonical_labels=("canonical_artist", lambda values: " | ".join(sorted(set(map(str, values))))),
        )
    )
    audit = grouped[grouped["canonical_label_count"] > 1].copy()
    audit["exclusion_rule"] = "EXCLUDE_FROM_LABEL_ASSOCIATION_AND_CO_MENTION_PRIMARY_ANALYSIS"
    audit = audit.sort_values(["canonical_label_count", "source_membership_rows", "analysis_text_sha256"], ascending=[False, False, True])
    return set(audit["analysis_text_sha256"]), audit


def entity_quantitative_stats(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "entity", "entity_type", "strict_agreement_occurrences", "lexicon_candidate_occurrences",
        "strict_agreement_rate", "unique_song_units", "source_credit_labels",
        "mean_transformer_confidence", "quantitative_gate_pass",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    denominator = (
        frame.groupby(["surface", "schema_type"], as_index=False)
        .size()
        .rename(columns={"surface": "entity", "schema_type": "entity_type", "size": "lexicon_candidate_occurrences"})
    )
    strict = frame[frame["strict_high_consistency"]].copy()
    if strict.empty:
        return pd.DataFrame(columns=columns)
    out = (
        strict.groupby(["surface", "schema_type"], as_index=False)
        .agg(
            strict_agreement_occurrences=("surface", "size"),
            unique_song_units=("song_lyric_content_sha256", "nunique"),
            source_credit_labels=("source_credit_label", "nunique"),
            mean_transformer_confidence=("transformer_confidence", "mean"),
        )
        .rename(columns={"surface": "entity", "schema_type": "entity_type"})
        .merge(denominator, on=["entity", "entity_type"], how="left", validate="one_to_one")
    )
    out["strict_agreement_rate"] = out["strict_agreement_occurrences"] / out["lexicon_candidate_occurrences"]
    out["quantitative_gate_pass"] = (
        (out["strict_agreement_occurrences"] >= 5)
        & (out["unique_song_units"] >= 5)
        & (out["source_credit_labels"] >= 3)
        & (out["strict_agreement_rate"] >= 0.50)
        & (out["mean_transformer_confidence"] >= 0.80)
    )
    out["strict_agreement_rate"] = out["strict_agreement_rate"].round(4)
    out["mean_transformer_confidence"] = out["mean_transformer_confidence"].round(4)
    return out[columns].sort_values(
        ["quantitative_gate_pass", "unique_song_units", "strict_agreement_occurrences", "entity"],
        ascending=[False, False, False, True],
    )


def semantic_surface_decision(
    entity: str,
    entity_type: str,
    core_person_surfaces: set[str],
) -> tuple[bool, str, str]:
    key = (entity, entity_type)
    if entity_type == "PERSON_REFERENCE" and entity not in core_person_surfaces:
        return (
            False,
            "PERSON_SURFACE_NOT_IN_T1_CORE_NAMED_INDIVIDUAL_ALLOWLIST",
            "WITHHOLD: two automated methods agree on a person-shaped span, but the surface is absent from the independently screened T1 named-individual ledger; credits, fictional references, and figurative uses remain possible.",
        )
    if key in AMBIGUOUS_PUBLIC_SURFACE_DENYLIST:
        return False, "AMBIGUOUS_SURFACE_TYPE_WITHOUT_OCCURRENCE_GOLD", AMBIGUOUS_PUBLIC_SURFACE_REASONS[key]
    if entity_type == "PERSON_REFERENCE":
        return (
            True,
            "T1_CORE_NAMED_INDIVIDUAL_SURFACE_PLUS_CROSS_METHOD_SUPPORT_GATES",
            "Surface-level release as a named-person reference only; this does not establish rapper identity, authorship, or biography.",
        )
    if entity_type == "PLACE":
        return (
            True,
            "CONVENTIONAL_NAMED_GEOGRAPHY_SURFACE_PLUS_CROSS_METHOD_SUPPORT_GATES",
            "Surface-level release as a conventional geographic reference; individual occurrences remain unreviewed.",
        )
    if entity_type == "LANGUAGE_OR_DIALECT_REFERENCE":
        return (
            True,
            "DIRECT_LANGUAGE_SURFACE_PLUS_CROSS_METHOD_SUPPORT_GATES",
            "Surface-level release as a direct language reference; individual occurrences remain unreviewed.",
        )
    return (
        False,
        "NO_EXPLICIT_SURFACE_TYPE_RELEASE_RULE_IN_V1_1",
        "WITHHOLD: v1.1 has no explicit, defensible surface/type release rule for this candidate without occurrence-level human review.",
    )


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if not len(values):
        return values
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def graph_song_membership(
    chunks: pd.DataFrame,
    graph_labels: pd.DataFrame,
    shared_hashes: set[str],
    exclude_shared: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_id = dict(zip(graph_labels["source_artist_label"], graph_labels["artist_label_id"]))
    frame = chunks[chunks["canonical_artist"].isin(label_id)].copy()
    if exclude_shared:
        frame = frame[~frame["analysis_text_sha256"].isin(shared_hashes)].copy()
    membership = frame[["canonical_artist", "song_lyric_content_sha256"]].drop_duplicates().rename(
        columns={"canonical_artist": "source_credit_label", "song_lyric_content_sha256": "song_unit"}
    )
    membership["source_label_id"] = membership["source_credit_label"].map(label_id)
    membership["membership_unit_id"] = [
        stable_id("MEM", row.source_label_id, row.song_unit) for row in membership.itertuples(index=False)
    ]
    label_counts = membership.groupby("source_credit_label")["membership_unit_id"].nunique().to_dict()
    universe = graph_labels.rename(
        columns={"artist_label_id": "source_label_id", "source_artist_label": "source_credit_label"}
    ).copy()
    universe["eligible_song_units_after_shared_text_rule"] = universe["source_credit_label"].map(label_counts).fillna(0).astype(int)
    universe["universe_status"] = "RETRIEVAL_GRAPH_ELIGIBLE_SOURCE_CREDIT_LABEL"
    universe["identity_boundary"] = "source credit label; not externally identity-verified"
    return membership, universe


def unrestricted_song_membership(
    chunks: pd.DataFrame,
    shared_hashes: set[str],
    exclude_shared: bool,
) -> pd.DataFrame:
    frame = chunks.copy()
    if exclude_shared:
        frame = frame[~frame["analysis_text_sha256"].isin(shared_hashes)].copy()
    membership = frame[["canonical_artist", "song_lyric_content_sha256"]].drop_duplicates().rename(
        columns={"canonical_artist": "source_credit_label", "song_lyric_content_sha256": "song_unit"}
    )
    membership["membership_unit_id"] = [
        stable_id("MEMALL", row.source_credit_label, row.song_unit) for row in membership.itertuples(index=False)
    ]
    return membership


def association_test_table(
    entity_inventory: pd.DataFrame,
    strict_occurrences: pd.DataFrame,
    membership: pd.DataFrame,
    graph_labels: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "source_label_id", "source_credit_label", "entity", "entity_type", "label_song_units",
        "entity_song_units_within_label", "entity_song_units_corpus", "all_membership_song_units",
        "within_label_share", "corpus_share", "raw_lift", "label_rate_jeffreys_mean",
        "label_rate_ci95_low", "label_rate_ci95_high", "rest_rate_jeffreys_mean",
        "rest_rate_ci95_low", "rest_rate_ci95_high", "shrunken_risk_ratio",
        "shrunken_risk_ratio_ci95_low_conservative", "shrunken_risk_ratio_ci95_high_conservative",
        "p_value_one_sided_hypergeometric", "q_value_bh", "legacy_basic_gate_pass",
        "release_gate_pass", "reliability_class", "withhold_reason", "plain_meaning",
        "association_scope", "status",
    ]
    if entity_inventory.empty or membership.empty:
        return pd.DataFrame(columns=columns)
    label_id = dict(zip(graph_labels["source_artist_label"], graph_labels["artist_label_id"]))
    inventory_keys = set(zip(entity_inventory["entity"], entity_inventory["entity_type"]))
    strict = strict_occurrences[
        strict_occurrences.apply(lambda row: (row["surface"], row["schema_type"]) in inventory_keys, axis=1)
    ].copy()
    presence = strict[["source_credit_label", "song_lyric_content_sha256", "surface", "schema_type"]].drop_duplicates().rename(
        columns={"song_lyric_content_sha256": "song_unit", "surface": "entity", "schema_type": "entity_type"}
    )
    valid_memberships = set(zip(membership["source_credit_label"], membership["song_unit"]))
    if not presence.empty:
        presence = presence[
            presence.apply(lambda row: (row["source_credit_label"], row["song_unit"]) in valid_memberships, axis=1)
        ].copy()
    total_n = int(membership["membership_unit_id"].nunique())
    label_n = membership.groupby("source_credit_label")["membership_unit_id"].nunique().to_dict()
    global_k = presence.groupby(["entity", "entity_type"])["song_unit"].size().to_dict()
    label_k = presence.groupby(["source_credit_label", "entity", "entity_type"])["song_unit"].size().to_dict()
    rows: list[dict[str, Any]] = []
    for label in sorted(label_id):
        n = int(label_n.get(label, 0))
        for entity, entity_type in sorted(inventory_keys):
            k = int(label_k.get((label, entity, entity_type), 0))
            total_k = int(global_k.get((entity, entity_type), 0))
            rows.append(
                {
                    "source_label_id": label_id[label],
                    "source_credit_label": label,
                    "entity": entity,
                    "entity_type": entity_type,
                    "label_song_units": n,
                    "entity_song_units_within_label": k,
                    "entity_song_units_corpus": total_k,
                    "all_membership_song_units": total_n,
                }
            )
    out = pd.DataFrame(rows)
    n = out["label_song_units"].to_numpy(dtype=float)
    k = out["entity_song_units_within_label"].to_numpy(dtype=float)
    total_k = out["entity_song_units_corpus"].to_numpy(dtype=float)
    rest_n = total_n - n
    rest_k = total_k - k
    out["within_label_share"] = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    out["corpus_share"] = total_k / total_n
    out["raw_lift"] = np.divide(
        out["within_label_share"].to_numpy(dtype=float),
        out["corpus_share"].to_numpy(dtype=float),
        out=np.zeros_like(k),
        where=out["corpus_share"].to_numpy(dtype=float) > 0,
    )
    out["label_rate_jeffreys_mean"] = (k + 0.5) / (n + 1.0)
    out["rest_rate_jeffreys_mean"] = (rest_k + 0.5) / (rest_n + 1.0)
    out["label_rate_ci95_low"] = beta_distribution.ppf(0.025, k + 0.5, n - k + 0.5)
    out["label_rate_ci95_high"] = beta_distribution.ppf(0.975, k + 0.5, n - k + 0.5)
    out["rest_rate_ci95_low"] = beta_distribution.ppf(0.025, rest_k + 0.5, rest_n - rest_k + 0.5)
    out["rest_rate_ci95_high"] = beta_distribution.ppf(0.975, rest_k + 0.5, rest_n - rest_k + 0.5)
    out["shrunken_risk_ratio"] = out["label_rate_jeffreys_mean"] / out["rest_rate_jeffreys_mean"]
    out["shrunken_risk_ratio_ci95_low_conservative"] = out["label_rate_ci95_low"] / out["rest_rate_ci95_high"]
    out["shrunken_risk_ratio_ci95_high_conservative"] = out["label_rate_ci95_high"] / out["rest_rate_ci95_low"]
    out["p_value_one_sided_hypergeometric"] = hypergeom.sf(k - 1, total_n, total_k, n)
    out["q_value_bh"] = bh_fdr(out["p_value_one_sided_hypergeometric"])
    out["legacy_basic_gate_pass"] = (
        (out["label_song_units"] >= 10)
        & (out["entity_song_units_within_label"] >= 3)
        & (out["within_label_share"] >= 0.05)
        & (out["raw_lift"] >= 1.25)
    )
    out["release_gate_pass"] = (
        (out["label_song_units"] >= 10)
        & (out["entity_song_units_within_label"] >= 5)
        & (out["within_label_share"] >= 0.05)
        & (out["shrunken_risk_ratio"] >= 1.50)
        & (out["shrunken_risk_ratio_ci95_low_conservative"] > 1.0)
        & (out["q_value_bh"] <= 0.05)
    )
    reliability: list[str] = []
    reasons: list[str] = []
    meanings: list[str] = []
    for row in out.itertuples(index=False):
        if row.release_gate_pass and row.entity_song_units_within_label >= 10 and row.q_value_bh <= 0.01 and row.shrunken_risk_ratio_ci95_low_conservative >= 1.25:
            reliability.append("HIGH")
            reasons.append("")
        elif row.release_gate_pass:
            reliability.append("SUPPORTED")
            reasons.append("")
        else:
            failures = []
            if row.label_song_units < 10:
                failures.append("label_support_below_10")
            if row.entity_song_units_within_label < 5:
                failures.append("entity_support_below_5_song_units")
            if row.within_label_share < 0.05:
                failures.append("within_label_share_below_0.05")
            if row.shrunken_risk_ratio < 1.50:
                failures.append("shrunken_risk_ratio_below_1.50")
            if row.shrunken_risk_ratio_ci95_low_conservative <= 1.0:
                failures.append("uncertainty_interval_includes_no_enrichment")
            if row.q_value_bh > 0.05:
                failures.append("bh_fdr_q_above_0.05")
            reliability.append("WITHHOLD")
            reasons.append(" | ".join(failures))
        meanings.append(
            f"After shared-text exclusion, {row.entity} appears in {row.entity_song_units_within_label} of "
            f"{row.label_song_units} eligible {row.source_credit_label} song units ({row.within_label_share:.1%}); "
            f"shrunken enrichment {row.shrunken_risk_ratio:.2f}x, BH q={row.q_value_bh:.3g}."
        )
    out["reliability_class"] = reliability
    out["withhold_reason"] = reasons
    out["plain_meaning"] = meanings
    out["association_scope"] = "shared-text-excluded source-credit-label lyric repertoire; not biography, preference, or social relation"
    out["status"] = np.where(
        out["release_gate_pass"],
        "PROVISIONAL_STATISTICALLY_SUPPORTED_NOT_HUMAN_VALIDATED",
        "WITHHELD_ASSOCIATION_TEST_NOT_RELEASED",
    )
    numeric = [
        "within_label_share", "corpus_share", "raw_lift", "label_rate_jeffreys_mean",
        "label_rate_ci95_low", "label_rate_ci95_high", "rest_rate_jeffreys_mean",
        "rest_rate_ci95_low", "rest_rate_ci95_high", "shrunken_risk_ratio",
        "shrunken_risk_ratio_ci95_low_conservative", "shrunken_risk_ratio_ci95_high_conservative",
    ]
    for column in numeric:
        out[column] = out[column].astype(float).round(6)
    for column in ["p_value_one_sided_hypergeometric", "q_value_bh"]:
        out[column] = out[column].astype(float)
    return out[columns].sort_values(
        ["release_gate_pass", "q_value_bh", "entity_song_units_within_label", "source_credit_label", "entity"],
        ascending=[False, True, False, True, True],
    )


def legacy_basic_link_table(
    entity_inventory: pd.DataFrame,
    strict_occurrences: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "source_credit_label", "entity", "entity_type", "label_song_units",
        "entity_song_units_within_label", "within_label_share", "corpus_relative_lift",
    ]
    keys = set(zip(entity_inventory.get("entity", []), entity_inventory.get("entity_type", [])))
    if not keys or membership.empty:
        return pd.DataFrame(columns=columns)
    strict = strict_occurrences[
        strict_occurrences.apply(lambda row: (row["surface"], row["schema_type"]) in keys, axis=1)
    ].copy()
    valid = set(zip(membership["source_credit_label"], membership["song_unit"]))
    presence = strict[
        ["source_credit_label", "song_lyric_content_sha256", "surface", "schema_type"]
    ].drop_duplicates().rename(
        columns={"song_lyric_content_sha256": "song_unit", "surface": "entity", "schema_type": "entity_type"}
    )
    presence = presence[
        presence.apply(lambda row: (row["source_credit_label"], row["song_unit"]) in valid, axis=1)
    ]
    total_units = int(membership["song_unit"].nunique())
    label_n = membership.groupby("source_credit_label")["song_unit"].nunique().to_dict()
    entity_n = presence.groupby(["entity", "entity_type"])["song_unit"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    grouped = presence.groupby(["source_credit_label", "entity", "entity_type"])["song_unit"].nunique()
    for (label, entity, entity_type), k in grouped.items():
        n = int(label_n.get(label, 0))
        share = int(k) / n if n else 0.0
        corpus_share = int(entity_n.get((entity, entity_type), 0)) / total_units if total_units else 0.0
        lift = share / corpus_share if corpus_share else 0.0
        if n >= 10 and int(k) >= 3 and share >= 0.05 and lift >= 1.25:
            rows.append(
                {
                    "source_credit_label": label,
                    "entity": entity,
                    "entity_type": entity_type,
                    "label_song_units": n,
                    "entity_song_units_within_label": int(k),
                    "within_label_share": round(share, 6),
                    "corpus_relative_lift": round(lift, 6),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["entity_song_units_within_label", "corpus_relative_lift", "source_credit_label", "entity"],
        ascending=[False, False, True, True],
    )


def co_mention_test_table(
    entity_inventory: pd.DataFrame,
    strict_occurrences: pd.DataFrame,
    membership: pd.DataFrame,
    denominator_mode: str = "all_eligible_song_units",
) -> pd.DataFrame:
    columns = [
        "entity_a", "entity_a_type", "entity_b", "entity_b_type", "all_eligible_song_units",
        "entity_a_song_units", "entity_b_song_units", "unique_song_unit_co_mentions",
        "source_credit_labels", "lift", "npmi", "p_value_one_sided_hypergeometric",
        "q_value_bh", "legacy_basic_gate_pass", "release_gate_pass", "reliability_class",
        "plain_meaning", "relation_scope", "status",
    ]
    keys = sorted(set(zip(entity_inventory.get("entity", []), entity_inventory.get("entity_type", []))))
    if len(keys) < 2 or membership.empty:
        return pd.DataFrame(columns=columns)
    valid_song_units = set(membership["song_unit"])
    strict = strict_occurrences[
        strict_occurrences.apply(lambda row: (row["surface"], row["schema_type"]) in set(keys), axis=1)
    ].copy()
    strict = strict[strict["song_lyric_content_sha256"].isin(valid_song_units)]
    presence: dict[str, set[tuple[str, str]]] = defaultdict(set)
    labels_by_unit: dict[str, set[str]] = defaultdict(set)
    for row in strict.itertuples(index=False):
        presence[row.song_lyric_content_sha256].add((row.surface, row.schema_type))
        labels_by_unit[row.song_lyric_content_sha256].add(row.source_credit_label)
    if denominator_mode not in {"all_eligible_song_units", "entity_bearing_song_units"}:
        raise ValueError(f"Unexpected co-mention denominator mode: {denominator_mode}")
    total_n = len(valid_song_units) if denominator_mode == "all_eligible_song_units" else len(presence)
    support = Counter()
    for entities in presence.values():
        support.update(entities)
    rows: list[dict[str, Any]] = []
    for a, b in itertools.combinations(keys, 2):
        shared_units = [unit for unit, entities in presence.items() if a in entities and b in entities]
        pair_k = len(shared_units)
        a_k = int(support[a])
        b_k = int(support[b])
        label_support = len(set().union(*(labels_by_unit[unit] for unit in shared_units))) if shared_units else 0
        pa = a_k / total_n if total_n else 0.0
        pb = b_k / total_n if total_n else 0.0
        pab = pair_k / total_n if total_n else 0.0
        lift = pab / (pa * pb) if pa and pb else 0.0
        npmi = math.log(lift) / -math.log(pab) if lift > 0 and 0 < pab < 1 else 0.0
        p_value = float(hypergeom.sf(pair_k - 1, total_n, a_k, b_k)) if total_n and a_k and b_k else 1.0
        rows.append(
            {
                "entity_a": a[0],
                "entity_a_type": a[1],
                "entity_b": b[0],
                "entity_b_type": b[1],
                "all_eligible_song_units": total_n,
                "entity_a_song_units": a_k,
                "entity_b_song_units": b_k,
                "unique_song_unit_co_mentions": pair_k,
                "source_credit_labels": label_support,
                "lift": lift,
                "npmi": npmi,
                "p_value_one_sided_hypergeometric": p_value,
            }
        )
    out = pd.DataFrame(rows)
    out["q_value_bh"] = bh_fdr(out["p_value_one_sided_hypergeometric"])
    out["legacy_basic_gate_pass"] = (
        (out["unique_song_unit_co_mentions"] >= 5)
        & (out["source_credit_labels"] >= 3)
        & (out["lift"] >= 1.25)
        & (out["npmi"] > 0)
    )
    out["release_gate_pass"] = out["legacy_basic_gate_pass"] & (out["q_value_bh"] <= 0.05)
    out["reliability_class"] = np.where(out["release_gate_pass"], "SUPPORTED", "WITHHOLD")
    out["plain_meaning"] = [
        f"{row.entity_a} and {row.entity_b} co-occur in {row.unique_song_unit_co_mentions} of "
        f"{row.all_eligible_song_units} eligible song units after shared-text exclusion; NPMI={row.npmi:.2f}, BH q={row.q_value_bh:.3g}."
        for row in out.itertuples(index=False)
    ]
    out["relation_scope"] = (
        "same-song lyric-reference co-mention using all eligible song units as denominator; not a social relationship"
        if denominator_mode == "all_eligible_song_units"
        else "legacy sensitivity using entity-bearing song units as denominator; not a social relationship"
    )
    out["status"] = np.where(
        out["release_gate_pass"],
        "PROVISIONAL_STATISTICALLY_SUPPORTED_NOT_HUMAN_VALIDATED",
        "WITHHELD_CO_MENTION_TEST_NOT_RELEASED",
    )
    for column in ["lift", "npmi"]:
        out[column] = out[column].astype(float).round(6)
    for column in ["p_value_one_sided_hypergeometric", "q_value_bh"]:
        out[column] = out[column].astype(float)
    return out[columns].sort_values(
        ["release_gate_pass", "q_value_bh", "unique_song_unit_co_mentions", "entity_a", "entity_b"],
        ascending=[False, True, False, True, True],
    )


def public_aggregates(
    agreements: pd.DataFrame,
    chunks: pd.DataFrame,
    songs: pd.DataFrame,
    core_ledger: pd.DataFrame,
    graph_labels: pd.DataFrame,
    shared_hashes: set[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    del songs  # eligible song metadata are already joined onto chunks and agreements.
    graph_label_names = set(graph_labels["source_artist_label"])
    lex_public = agreements[agreements["schema_type"].isin(PUBLIC_ENTITY_TYPES)].copy()
    all_post = lex_public[~lex_public["analysis_text_sha256"].isin(shared_hashes)].copy()
    graph_pre = lex_public[lex_public["source_credit_label"].isin(graph_label_names)].copy()
    graph_post = graph_pre[~graph_pre["analysis_text_sha256"].isin(shared_hashes)].copy()

    corpus_stats = entity_quantitative_stats(lex_public)
    all_post_stats = entity_quantitative_stats(all_post)
    graph_pre_stats = entity_quantitative_stats(graph_pre)
    graph_post_stats = entity_quantitative_stats(graph_post)
    core_person_surfaces = set(
        core_ledger.loc[
            core_ledger["canonical_category"].eq("PERSON_NAMED")
            & core_ledger["evidence_tier"].str.startswith("T1"),
            "entity",
        ].astype(str)
    )

    candidate_keys = set(
        zip(
            corpus_stats.loc[corpus_stats["quantitative_gate_pass"], "entity"],
            corpus_stats.loc[corpus_stats["quantitative_gate_pass"], "entity_type"],
        )
    )
    semantic_decisions = {
        key: semantic_surface_decision(key[0], key[1], core_person_surfaces) for key in sorted(candidate_keys)
    }

    def released_inventory(stats: pd.DataFrame, status: str) -> pd.DataFrame:
        frame = stats[stats["quantitative_gate_pass"]].copy()
        if frame.empty:
            frame["status"] = pd.Series(dtype=str)
            return frame
        frame = frame[
            frame.apply(
                lambda row: semantic_decisions.get((row["entity"], row["entity_type"]), (False, "", ""))[0],
                axis=1,
            )
        ].copy()
        frame["status"] = status
        return frame.sort_values(["unique_song_units", "strict_agreement_occurrences", "entity"], ascending=[False, False, True])

    corpus_inventory = released_inventory(
        corpus_stats,
        "PROVISIONAL_CORPUSWIDE_SENSITIVITY_NOT_SHARED_TEXT_EXCLUDED_NOT_HUMAN_VALIDATED",
    )
    all_post_inventory = released_inventory(
        all_post_stats,
        "PROVISIONAL_SHARED_TEXT_EXCLUDED_ALL_LABELS_SENSITIVITY_NOT_HUMAN_VALIDATED",
    )
    primary_inventory = released_inventory(
        graph_post_stats,
        "PROVISIONAL_SHARED_TEXT_EXCLUDED_GRAPH_UNIVERSE_NOT_HUMAN_VALIDATED",
    )

    stat_maps = {}
    for name, frame in [
        ("corpus", corpus_stats),
        ("all_post", all_post_stats),
        ("graph_pre", graph_pre_stats),
        ("graph_post", graph_post_stats),
    ]:
        stat_maps[name] = {(row["entity"], row["entity_type"]): row for row in frame.to_dict("records")}

    audit_rows: list[dict[str, Any]] = []
    for key in sorted(candidate_keys):
        semantic_release, basis, human_reason = semantic_decisions[key]
        corpus_row = stat_maps["corpus"].get(key, {})
        all_post_row = stat_maps["all_post"].get(key, {})
        post_row = stat_maps["graph_post"].get(key, {})
        corpus_pass = bool(corpus_row.get("quantitative_gate_pass", False))
        post_pass = bool(post_row.get("quantitative_gate_pass", False))
        primary_release = semantic_release and post_pass
        if not semantic_release:
            decision = "WITHHOLD_SEMANTIC_TYPE"
        elif not post_pass:
            decision = "WITHHOLD_AFTER_SHARED_TEXT_AND_GRAPH_UNIVERSE_SUPPORT_GATE"
            human_reason = (
                "WITHHOLD FROM PRIMARY GRAPH: the surface/type is defensible, but it no longer meets the quantitative "
                "support gate after restricting to 204 labels and excluding cross-label shared cleaned text."
            )
        else:
            decision = "RELEASE_PROVISIONAL_PRIMARY"
        audit_rows.append(
            {
                "entity": key[0],
                "proposed_entity_type": key[1],
                "corpuswide_quantitative_gate_pass": corpus_pass,
                "shared_text_excluded_graph_quantitative_gate_pass": post_pass,
                "semantic_surface_type_pass": semantic_release,
                "primary_release_decision": decision,
                "decision_basis": basis,
                "human_readable_reason": human_reason,
                "allowed_public_claim": (
                    f"The shared-text-excluded 204-label graph universe contains provisional {key[1]} references to {key[0]}."
                    if primary_release
                    else "No primary graph entity or edge claim is released for this surface/type pair."
                ),
                "status": (
                    "PROVISIONAL_PRIMARY_RELEASE_NOT_OCCURRENCE_GOLD"
                    if primary_release
                    else "WITHHELD_FROM_PRIMARY_GRAPH_NOT_OCCURRENCE_GOLD"
                ),
            }
        )
    surface_audit = pd.DataFrame(audit_rows)

    sensitivity_rows: list[dict[str, Any]] = []
    corpus_keys = set(zip(corpus_inventory["entity"], corpus_inventory["entity_type"]))
    primary_keys = set(zip(primary_inventory["entity"], primary_inventory["entity_type"]))
    for key in sorted(corpus_keys):
        corpus_row = stat_maps["corpus"].get(key, {})
        all_post_row = stat_maps["all_post"].get(key, {})
        pre_row = stat_maps["graph_pre"].get(key, {})
        post_row = stat_maps["graph_post"].get(key, {})
        corpus_occ = int(corpus_row.get("strict_agreement_occurrences", 0))
        all_post_occ = int(all_post_row.get("strict_agreement_occurrences", 0))
        pre_occ = int(pre_row.get("strict_agreement_occurrences", 0))
        post_occ = int(post_row.get("strict_agreement_occurrences", 0))
        sensitivity_rows.append(
            {
                "entity": key[0],
                "entity_type": key[1],
                "corpuswide_strict_occurrences": corpus_occ,
                "all_labels_after_shared_exclusion_strict_occurrences": all_post_occ,
                "removed_cross_label_shared_text_occurrences_all_labels": corpus_occ - all_post_occ,
                "graph_universe_before_shared_exclusion_strict_occurrences": pre_occ,
                "graph_universe_after_shared_exclusion_strict_occurrences": post_occ,
                "removed_outside_204_label_universe_occurrences": corpus_occ - pre_occ,
                "removed_cross_label_shared_text_occurrences_within_graph_universe": pre_occ - post_occ,
                "shared_text_removed_share_within_graph_universe": round((pre_occ - post_occ) / pre_occ, 6) if pre_occ else 0.0,
                "corpuswide_unique_song_units": int(corpus_row.get("unique_song_units", 0)),
                "graph_after_exclusion_unique_song_units": int(post_row.get("unique_song_units", 0)),
                "corpuswide_source_credit_labels": int(corpus_row.get("source_credit_labels", 0)),
                "graph_after_exclusion_source_credit_labels": int(post_row.get("source_credit_labels", 0)),
                "primary_entity_release": key in primary_keys,
                "interpretation": "Sensitivity to the 204-label universe and cross-label exact cleaned-text exclusion; counts are repeated corpus occurrences, not independent samples.",
            }
        )
    entity_sensitivity = pd.DataFrame(sensitivity_rows)

    all_pre_membership = unrestricted_song_membership(chunks, shared_hashes, exclude_shared=False)
    all_post_membership = unrestricted_song_membership(chunks, shared_hashes, exclude_shared=True)
    pre_membership, _ = graph_song_membership(chunks, graph_labels, shared_hashes, exclude_shared=False)
    post_membership, graph_universe = graph_song_membership(chunks, graph_labels, shared_hashes, exclude_shared=True)
    strict_all_pre = lex_public[lex_public["strict_high_consistency"]].copy()
    strict_all_post = all_post[all_post["strict_high_consistency"]].copy()
    strict_graph_pre = graph_pre[graph_pre["strict_high_consistency"]].copy()
    strict_graph_post = graph_post[graph_post["strict_high_consistency"]].copy()
    legacy_all_pre_links = legacy_basic_link_table(corpus_inventory, strict_all_pre, all_pre_membership)
    legacy_all_post_links = legacy_basic_link_table(all_post_inventory, strict_all_post, all_post_membership)
    legacy_graph_pre_links = legacy_basic_link_table(corpus_inventory, strict_graph_pre, pre_membership)
    legacy_graph_post_links = legacy_basic_link_table(primary_inventory, strict_graph_post, post_membership)
    pre_tests = association_test_table(corpus_inventory, strict_graph_pre, pre_membership, graph_labels)
    post_tests = association_test_table(primary_inventory, strict_graph_post, post_membership, graph_labels)
    released_links = post_tests[post_tests["release_gate_pass"]].copy()
    released_links["status"] = "PROVISIONAL_STATISTICALLY_SUPPORTED_NOT_HUMAN_VALIDATED"

    pre_test_map = {
        (row["source_credit_label"], row["entity"], row["entity_type"]): row for row in pre_tests.to_dict("records")
    }
    post_test_map = {
        (row["source_credit_label"], row["entity"], row["entity_type"]): row for row in post_tests.to_dict("records")
    }
    link_sensitivity_rows: list[dict[str, Any]] = []
    link_keys = {
        key for key, row in pre_test_map.items() if row["legacy_basic_gate_pass"]
    } | {
        key for key, row in post_test_map.items() if row["legacy_basic_gate_pass"] or row["release_gate_pass"]
    }
    for key in sorted(link_keys):
        before = pre_test_map.get(key, {})
        after = post_test_map.get(key, {})
        link_sensitivity_rows.append(
            {
                "source_credit_label": key[0],
                "entity": key[1],
                "entity_type": key[2],
                "before_shared_exclusion_song_support": int(before.get("entity_song_units_within_label", 0)),
                "after_shared_exclusion_song_support": int(after.get("entity_song_units_within_label", 0)),
                "before_shared_exclusion_legacy_basic_gate": bool(before.get("legacy_basic_gate_pass", False)),
                "after_shared_exclusion_legacy_basic_gate": bool(after.get("legacy_basic_gate_pass", False)),
                "after_uncertainty_and_bh_release_gate": bool(after.get("release_gate_pass", False)),
                "after_shrunken_risk_ratio": after.get("shrunken_risk_ratio", ""),
                "after_conservative_rr_ci95_low": after.get("shrunken_risk_ratio_ci95_low_conservative", ""),
                "after_bh_q_value": after.get("q_value_bh", ""),
                "final_reliability_class": after.get("reliability_class", "WITHHOLD"),
                "final_withhold_reason": after.get("withhold_reason", "entity_not_in_primary_inventory"),
            }
        )
    link_sensitivity = pd.DataFrame(link_sensitivity_rows)

    legacy_all_pre_co_tests = co_mention_test_table(
        corpus_inventory,
        strict_all_pre,
        all_pre_membership,
        denominator_mode="entity_bearing_song_units",
    )
    legacy_all_post_co_tests = co_mention_test_table(
        all_post_inventory,
        strict_all_post,
        all_post_membership,
        denominator_mode="entity_bearing_song_units",
    )
    legacy_graph_post_co_tests = co_mention_test_table(
        primary_inventory,
        strict_graph_post,
        post_membership,
        denominator_mode="entity_bearing_song_units",
    )
    post_co_tests = co_mention_test_table(primary_inventory, strict_graph_post, post_membership)
    released_co = post_co_tests[post_co_tests["release_gate_pass"]].copy()
    released_co["status"] = "PROVISIONAL_STATISTICALLY_SUPPORTED_NOT_HUMAN_VALIDATED"
    pre_co_map = {
        (row["entity_a"], row["entity_a_type"], row["entity_b"], row["entity_b_type"]): row
        for row in legacy_all_pre_co_tests.to_dict("records")
    }
    post_co_map = {
        (row["entity_a"], row["entity_a_type"], row["entity_b"], row["entity_b_type"]): row
        for row in post_co_tests.to_dict("records")
    }
    co_sensitivity_rows: list[dict[str, Any]] = []
    co_keys = {
        key for key, row in pre_co_map.items() if row["legacy_basic_gate_pass"]
    } | {
        key for key, row in post_co_map.items() if row["legacy_basic_gate_pass"] or row["release_gate_pass"]
    }
    for key in sorted(co_keys):
        before = pre_co_map.get(key, {})
        after = post_co_map.get(key, {})
        co_sensitivity_rows.append(
            {
                "entity_a": key[0],
                "entity_a_type": key[1],
                "entity_b": key[2],
                "entity_b_type": key[3],
                "before_shared_exclusion_song_support": int(before.get("unique_song_unit_co_mentions", 0)),
                "after_shared_exclusion_song_support": int(after.get("unique_song_unit_co_mentions", 0)),
                "before_shared_exclusion_legacy_entity_bearing_gate": bool(before.get("legacy_basic_gate_pass", False)),
                "after_shared_exclusion_all_eligible_denominator_basic_gate": bool(after.get("legacy_basic_gate_pass", False)),
                "after_bh_release_gate": bool(after.get("release_gate_pass", False)),
                "after_npmi_all_eligible_denominator": after.get("npmi", ""),
                "after_bh_q_value": after.get("q_value_bh", ""),
                "final_reliability_class": after.get("reliability_class", "WITHHOLD"),
            }
        )
    co_sensitivity = pd.DataFrame(co_sensitivity_rows)

    release_sensitivity = pd.DataFrame(
        [
            {
                "stage": "v1_legacy_all_labels_before_shared_exclusion",
                "entities": len(corpus_inventory),
                "label_entity_links": len(legacy_all_pre_links),
                "co_mentions": int(legacy_all_pre_co_tests["legacy_basic_gate_pass"].sum()),
                "meaning": "Reconciliation of v1: all eligible labels, shared text included, support/lift-only links, entity-bearing co-mention denominator.",
            },
            {
                "stage": "all_labels_after_shared_exclusion_legacy_gates",
                "entities": len(all_post_inventory),
                "label_entity_links": len(legacy_all_post_links),
                "co_mentions": int(legacy_all_post_co_tests["legacy_basic_gate_pass"].sum()),
                "meaning": "Shared text excluded across all labels; legacy gates retained only to isolate that change.",
            },
            {
                "stage": "204_label_primary_universe_after_shared_exclusion_legacy_gates",
                "entities": len(primary_inventory),
                "label_entity_links": len(legacy_graph_post_links),
                "co_mentions": int(legacy_graph_post_co_tests["legacy_basic_gate_pass"].sum()),
                "meaning": "204 retrieval graph-eligible labels, shared text excluded, legacy gates retained only for sensitivity.",
            },
            {
                "stage": "v1_1_primary_all_song_denominator_uncertainty_fdr_release",
                "entities": len(primary_inventory),
                "label_entity_links": len(released_links),
                "co_mentions": len(released_co),
                "meaning": "Primary release after shared-text exclusion, shrinkage/interval gates, and BH-FDR.",
            },
        ]
    )

    node_rows: list[dict[str, Any]] = []
    for row in primary_inventory.to_dict("records"):
        node_rows.append(
            {
                "node_id": stable_id("ENT", row["entity_type"], row["entity"]),
                "node_label": row["entity"],
                "node_type": row["entity_type"],
                "support_song_units": int(row["unique_song_units"]),
                "status": row["status"],
            }
        )
    released_label_ids = set(released_links.get("source_label_id", []))
    for row in graph_universe[graph_universe["source_label_id"].isin(released_label_ids)].to_dict("records"):
        node_rows.append(
            {
                "node_id": row["source_label_id"],
                "node_label": row["source_credit_label"],
                "node_type": "SOURCE_CREDIT_LABEL",
                "support_song_units": int(row["eligible_song_units_after_shared_text_rule"]),
                "status": "SOURCE_LABEL_NOT_INDEPENDENTLY_IDENTITY_VERIFIED",
            }
        )
    network_nodes = pd.DataFrame(
        node_rows,
        columns=["node_id", "node_label", "node_type", "support_song_units", "status"],
    ).drop_duplicates("node_id")
    edge_rows: list[dict[str, Any]] = []
    for row in released_links.to_dict("records"):
        edge_rows.append(
            {
                "source_node_id": row["source_label_id"],
                "target_node_id": stable_id("ENT", row["entity_type"], row["entity"]),
                "edge_type": "SOURCE_LABEL_TO_LYRIC_REFERENCE",
                "support_song_units": int(row["entity_song_units_within_label"]),
                "effect_size": row["shrunken_risk_ratio"],
                "uncertainty_lower": row["shrunken_risk_ratio_ci95_low_conservative"],
                "q_value_bh": row["q_value_bh"],
                "reliability_class": row["reliability_class"],
                "meaning": row["plain_meaning"] + " This is not biography, preference, or a social relationship.",
                "status": row["status"],
            }
        )
    for row in released_co.to_dict("records"):
        edge_rows.append(
            {
                "source_node_id": stable_id("ENT", row["entity_a_type"], row["entity_a"]),
                "target_node_id": stable_id("ENT", row["entity_b_type"], row["entity_b"]),
                "edge_type": "SAME_SONG_LYRIC_REFERENCE_CO_MENTION",
                "support_song_units": int(row["unique_song_unit_co_mentions"]),
                "effect_size": row["npmi"],
                "uncertainty_lower": "",
                "q_value_bh": row["q_value_bh"],
                "reliability_class": row["reliability_class"],
                "meaning": row["plain_meaning"] + " This is not a social relationship.",
                "status": row["status"],
            }
        )
    network_edges = pd.DataFrame(
        edge_rows,
        columns=[
            "source_node_id", "target_node_id", "edge_type", "support_song_units", "effect_size",
            "uncertainty_lower", "q_value_bh", "reliability_class", "meaning", "status",
        ],
    )

    public_frames = {
        "entity_aggregate_provisional.csv": primary_inventory,
        "entity_inventory_corpuswide_sensitivity.csv": corpus_inventory,
        "entity_inventory_shared_text_excluded_all_labels_sensitivity.csv": all_post_inventory,
        "surface_type_release_audit.csv": surface_audit,
        "shared_text_exclusion_entity_sensitivity.csv": entity_sensitivity,
        "graph_label_universe.csv": graph_universe,
        "source_label_entity_association_tests.csv": post_tests,
        "source_label_entity_links_provisional.csv": released_links,
        "source_label_entity_link_sensitivity.csv": link_sensitivity,
        "entity_co_mentions_provisional.csv": released_co,
        "entity_co_mention_sensitivity.csv": co_sensitivity,
        "release_sensitivity_summary.csv": release_sensitivity,
        "cultural_network_nodes_provisional.csv": network_nodes,
        "cultural_network_edges_provisional.csv": network_edges,
    }
    diagnostics = {
        "shared_hashes": len(shared_hashes),
        "graph_labels": len(graph_labels),
        "corpus_inventory_entities": len(corpus_inventory),
        "all_labels_after_shared_exclusion_inventory_entities": len(all_post_inventory),
        "primary_inventory_entities": len(primary_inventory),
        "corpus_inventory_strict_occurrences": int(corpus_inventory["strict_agreement_occurrences"].sum()),
        "graph_pre_inventory_strict_occurrences": int(
            entity_sensitivity["graph_universe_before_shared_exclusion_strict_occurrences"].sum()
        ),
        "graph_post_inventory_strict_occurrences": int(
            entity_sensitivity["graph_universe_after_shared_exclusion_strict_occurrences"].sum()
        ),
        "removed_shared_inventory_occurrences": int(
            entity_sensitivity["removed_cross_label_shared_text_occurrences_within_graph_universe"].sum()
        ),
        "removed_shared_inventory_occurrences_all_labels": int(
            entity_sensitivity["removed_cross_label_shared_text_occurrences_all_labels"].sum()
        ),
        "legacy_all_label_pre_basic_label_links": len(legacy_all_pre_links),
        "legacy_all_label_post_basic_label_links": len(legacy_all_post_links),
        "graph_pre_basic_label_links": len(legacy_graph_pre_links),
        "post_basic_label_links": len(legacy_graph_post_links),
        "released_label_links": len(released_links),
        "legacy_all_label_pre_basic_co_mentions": int(legacy_all_pre_co_tests["legacy_basic_gate_pass"].sum()),
        "legacy_all_label_post_basic_co_mentions": int(legacy_all_post_co_tests["legacy_basic_gate_pass"].sum()),
        "graph_post_legacy_entity_bearing_co_mentions": int(legacy_graph_post_co_tests["legacy_basic_gate_pass"].sum()),
        "post_basic_co_mentions_all_song_denominator": int(post_co_tests["legacy_basic_gate_pass"].sum()),
        "released_co_mentions": len(released_co),
        "eligible_membership_song_units": len(post_membership),
        "eligible_global_song_units": int(post_membership["song_unit"].nunique()),
    }
    return public_frames, diagnostics


def schema_document(task_count: int) -> str:
    return f"""# Chinese Rap Entity Schema and Annotation Guideline v1.1

## Evidence status

This is an annotation protocol, not a gold dataset. The current release contains {task_count} unreviewed, stratified candidate occurrences for two independent reviewers. Gold labels may be created only after both reviews are complete, disagreements are adjudicated, and the final table is frozen with a manifest.

## Unit of annotation

One task is one character-span candidate in one private lyric context. Annotate the span as used in that context; do not infer a performer's biography, preference, identity, affiliation, collaboration, or social relationship.

## Entity types

| Code | Include | Exclude / boundary |
| --- | --- | --- |
| `PERSON_REFERENCE` | A surface that may name a real, stage-name, fictional, or mythic person in context | This label never by itself means “rapper”; exclude kinship terms, pronouns, roles, common nouns, credit residue, and figurative personification unless a reviewer confirms a named referent |
| `GROUP_CREW_OR_ORGANIZATION` | Named rap crew, collective, label, company, institution, or organized group | Generic “team”, “crew”, “company”, or fandom nouns without a specific name |
| `PLACE` | Named country, city, region, neighborhood, landmark, venue, street, or geographic feature | Generic “home”, “street”, “city”, “world”, or directional language without a specific referent |
| `BRAND_OR_PRODUCT` | Named brand, product line, vehicle brand/model, fashion house, drink/substance brand, or named consumer object | Generic objects or substances; annotate the shortest complete brand/product name |
| `WORK_OR_MEDIA` | Named song, album, film, book, program, platform, law, or other titled cultural work | Generic genres and untitled references |
| `EVENT` | Named historical, political, sporting, cultural, or rap event | Generic events such as “a battle” or “the show” without a specific name |
| `LANGUAGE_OR_DIALECT_REFERENCE` | Named language or dialect | Broad adjectives and scripts when no language is referred to |
| `ETHNOCULTURAL_GROUP_REFERENCE` | Named ethnic, national, or cultural group of people | Country/place names and broad adjectives when no people are referred to |
| `RAP_CULTURE_CONCEPT` | Domain-specific rap practice or culture concept useful for this corpus, such as a named technique or scene term | Ordinary English tokens and generic performance words; this class is not published as general-domain NER without human review |
| `OTHER_CULTURAL_REFERENCE` | A specific named reference that fits none of the above | Use sparingly and explain in notes |
| `NOT_ENTITY` | The proposed span is a common word, metaphor, extraction fragment, header residue, or otherwise not a named/cultural reference | — |
| `UNCERTAIN` | Context is genuinely insufficient after applying the rules | Do not use merely because external linking is unavailable |

## Span rule

Select the shortest complete surface that uniquely expresses the reference in context. Include internal English/Chinese characters and required name particles; exclude punctuation, surrounding titles, hashtags, and possessives unless part of the conventional name. When nested candidates occur, prefer the longer complete named expression and record the shorter candidate as invalid.

## Required reviewer fields

1. `mention_valid`: `VALID`, `INVALID`, or `UNCERTAIN`.
2. `entity_type`: one schema code above.
3. `referential_status`: `NAMED_REAL_WORLD`, `FICTIONAL_OR_MYTHIC`, `METAPHOR_OR_COMMON_WORD`, `AMBIGUOUS`, or `NOT_APPLICABLE`.
4. `normalized_surface`: spelling/case normalization only; do not invent a real-world identity.
5. `linking_status`: `SURFACE_ONLY`, `UNRESOLVED`, `RESOLVED`, or `NOT_APPLICABLE`.
6. `resolved_entity_id`: local or public stable ID only when evidence has been checked; otherwise blank.
7. Confidence from 1 (guess) to 5 (clear), plus a short note for invalid, uncertain, or corrected cases.

## Independent review and adjudication

R1 and R2 must work independently. Exact agreement is calculated separately for mention validity, entity type, and referential status. Any disagreement, either reviewer's `UNCERTAIN`, a corrected span/type, or conflicting entity link requires adjudication. The adjudicator sees both ratings and writes the final decision and rationale. Do not overwrite either original review.

## Gold-release gate

The package may be called gold only when: both reviewers completed every assigned task; adjudication is complete; no blank final labels remain; inter-annotator agreement and per-class support are reported; task IDs and input hashes match the frozen manifest; and a train/dev/test split is produced by song (never by occurrence) to prevent leakage.
"""


def method_document(
    chunks: pd.DataFrame,
    lexicon: pd.DataFrame,
    tasks: pd.DataFrame,
    model_provenance: dict[str, Any],
    existing_audit: dict[str, Any],
    diagnostics: dict[str, Any],
) -> str:
    return f"""# Method and Claim Boundaries

## Research question

Which named-person surfaces, groups or crews, places, brands/products, works/events, languages/dialects, and ethnocultural-group references can be recovered reproducibly from this Chinese-rap lyric corpus, and which aggregate co-mention patterns remain after duplicate control and conservative cross-method agreement gates? A `PERSON_REFERENCE` is never interpreted automatically as a rapper identity.

## Input audit

- Clean lyric sidecar: {len(chunks):,} eligible chunks; private text is never copied into the public artifact.
- Reviewed lexicon: {len(lexicon):,} screened surfaces. These are surface-level review decisions, not occurrence-level truth.
- Existing context audit v2: {existing_audit['context_tasks']:,} tasks found across evaluation and coverage frames; {existing_audit['context_review_form_tasks']:,} evaluation tasks have reviewer forms and {existing_audit['completed_context_reviews']:,} reviewer decisions are completed. A further {existing_audit['legacy_context_tasks']:,} v1 queue rows are also unreviewed.
- Existing lexicon occurrence audit: {existing_audit['occurrence_tasks']:,} tasks found, {existing_audit['completed_occurrence_reviews']:,} completed reviewer decisions.
- Existing verified factual entity registry rows: {existing_audit['verified_entity_rows']:,}; verified relation-evidence rows: {existing_audit['verified_relation_rows']:,}.

Because completed occurrence-level human gold is absent, this release does **not** report precision, recall, F1, or a train/dev/test benchmark.

## Baseline A — reviewed lexicon exact matcher

The first baseline maps the screened domain lexicon to the project schema and performs case-sensitive literal matching. ASCII/digit surfaces require non-ASCII-word boundaries. Overlaps are resolved globally by longest span, then earliest start. This baseline has domain coverage but can still confuse a named surface with a common word in a particular lyric context.

## Baseline B — general-domain Chinese transformer NER

The second baseline is the locally pinned `ckiplab/albert-tiny-chinese-ner` token classifier. Its config SHA-256 is `{model_provenance['model_config_sha256']}` and weights SHA-256 is `{model_provenance['model_weights_sha256']}`. It is a Traditional-Chinese, general-domain model; Chinese-rap slang and stage names are a domain-shift risk. Lines are processed in 180-character windows with 40-character overlap, and overlapping window predictions are deduplicated. The model is used as a reproducible independent candidate baseline, not as an oracle.

## Common comparison frame

Both baselines run on every unique lyric line containing a target-schema lexicon candidate plus a deterministic hash-ranked background sample of non-lexicon lines. `baseline_comparison.csv` and `cross_method_agreement_by_type.csv` count each identical lyric line/span/type once, so their agreement ratios use one consistent unique-line frame. Public entity support is separately counted over duplicate-controlled full-song lyric-content units. Agreement statistics are candidate overlap statistics only; they are not accuracy metrics.

The full source-occurrence frame contains {diagnostics['exact_span_type_agreement_occurrences']:,} exact span/type agreements, of which {diagnostics['strict_high_consistency_occurrences']:,} also clear the 0.80 confidence rule. After identical lyric line/span/type combinations are counted once, the corresponding counts are {diagnostics['exact_span_type_agreement_unique_line_spans']:,} and {diagnostics['strict_high_consistency_unique_line_spans']:,}. Occurrence counts are repeated corpus spans and must not be interpreted as independent samples.

## Private annotation sample

The private package contains {len(tasks):,} distinct candidate occurrences stratified by proposed entity type and agreement state. It includes exact-match agreement, overlap agreement, type/boundary conflicts, lexicon-only candidates, and transformer-only candidates. R1/R2 templates are independent, and a separate agreement/adjudication template preserves both ratings.

## Provisional public entity gate and surface/type audit

The corpuswide sensitivity inventory first requires exact span and schema-type agreement between both baselines; transformer confidence at least 0.80; at least five strict-agreement occurrences; at least five distinct full-song lyric-content units; at least three source-credit labels; strict agreement on at least 50% of that lexicon surface's candidate occurrences; and mean transformer confidence at least 0.80. The primary graph then reapplies those gates after restricting to the 204 retrieval graph-eligible source-credit labels and removing every exact cleaned-text hash observed under more than one canonical label. There are {diagnostics['shared_hashes']:,} such hashes. This changes the semantic-release inventory from {diagnostics['corpus_inventory_entities']:,} corpuswide sensitivity entities to {diagnostics['primary_inventory_entities']:,} primary graph entities.

Every surface/type pair receives an explicit decision in `surface_type_release_audit.csv`. A `PERSON_REFERENCE` is released only when the surface is also present in the independently screened T1 named-individual ledger; this supports a named-person surface, not rapper identity, authorship, or biography. Conventionally geographic place names and direct language names may be released as narrow surface types. Ambiguous surfaces such as `中南海`, `桃源`, and `西山` remain withheld. Only `RELEASE_PROVISIONAL_PRIMARY` rows can enter the primary graph.

## Shared-text exclusion and analysis universes

NER candidate discovery and the private annotation package cover all {len(chunks):,} eligible cleaned chunks. The primary association graph uses a separately named universe: the 204 source-credit labels frozen by the retrieval graph registry. A cleaned-text hash is excluded from label associations and co-mentions if it appears under more than one canonical label anywhere in the eligible corpus, including labels outside the 204-label graph universe. The private audit retains every source occurrence and the hash-level exclusion ledger; public outputs contain only aggregate sensitivity counts.

`entity_inventory_corpuswide_sensitivity.csv` is an explicitly non-primary corpuswide inventory. `shared_text_exclusion_entity_sensitivity.csv`, `source_label_entity_link_sensitivity.csv`, and `entity_co_mention_sensitivity.csv` show before/after consequences. Across all labels, shared-text exclusion removes {diagnostics['removed_shared_inventory_occurrences_all_labels']:,} of {diagnostics['corpus_inventory_strict_occurrences']:,} strict occurrences attached to the corpuswide semantic-release inventory and changes that inventory from {diagnostics['corpus_inventory_entities']:,} to {diagnostics['all_labels_after_shared_exclusion_inventory_entities']:,} surfaces. Restriction to the 204-label graph universe then yields {diagnostics['primary_inventory_entities']:,} primary surfaces. Inside that universe, {diagnostics['removed_shared_inventory_occurrences']:,} of {diagnostics['graph_pre_inventory_strict_occurrences']:,} strict occurrences are removed. These counts remain repeated corpus occurrences, not independent observations.

## Grounded cultural network

The public network has two bounded edge meanings:

1. `SOURCE_LABEL_TO_LYRIC_REFERENCE`: the entity must occur in at least five of at least ten eligible source-labelled song units, cover at least 5% of those units, have a Jeffreys-smoothed risk ratio of at least 1.50 versus the rest of the graph universe, have a conservative 95% risk-ratio interval lower bound above 1.0, and pass Benjamini–Hochberg FDR at q ≤ 0.05 across every 204-label × released-entity test. The table reports raw lift, Jeffreys rates and intervals, the shrunken risk ratio, p-value, q-value, and reliability class. It describes a source-labelled lyric repertoire, not biography or preference.
2. `SAME_SONG_LYRIC_REFERENCE_CO_MENTION`: two provisional entities must appear in at least five distinct full-song lyric-content units across at least three source-credit labels, with lift at least 1.25, positive normalized PMI, and BH q ≤ 0.05. The NPMI denominator is **all {diagnostics['eligible_global_song_units']:,} eligible shared-text-excluded full-song units**, including units with no released entity. It is a textual co-mention, not collaboration, influence, affiliation, identity, or a social relationship.

The association denominator has {diagnostics['eligible_membership_song_units']:,} source-label/song membership units; the co-mention denominator has {diagnostics['eligible_global_song_units']:,} distinct full-song-content units. Hashes, song IDs, chunk IDs, lyrics, contexts, and embeddings remain private.

For exact reconciliation with v1, the all-label, support/lift-only label-reference count changes from {diagnostics['legacy_all_label_pre_basic_label_links']:,} to {diagnostics['legacy_all_label_post_basic_label_links']:,} after shared-text exclusion. The legacy entity-bearing-denominator co-mention count changes from {diagnostics['legacy_all_label_pre_basic_co_mentions']:,} to {diagnostics['legacy_all_label_post_basic_co_mentions']:,}. Restricting to the 204-label graph universe leaves {diagnostics['post_basic_label_links']:,} legacy-gate label candidates and {diagnostics['graph_post_legacy_entity_bearing_co_mentions']:,} legacy-denominator co-mentions. The v1.1 all-eligible-song denominator yields {diagnostics['post_basic_co_mentions_all_song_denominator']:,} basic co-mention candidates. Uncertainty and BH-FDR produce the primary release of {diagnostics['released_label_links']:,} label-reference edges and {diagnostics['released_co_mentions']:,} co-mention edges.

## Limitations

The source-credit labels are not independently identity-verified. Literal lexicon matching is context-insensitive, the transformer faces domain and script shift, model confidence is not calibrated on rap lyrics, confidence/agreement gates are not substitutes for occurrence gold, and exact shared-text exclusion changes the estimand toward label-specific repertoire. Jeffreys intervals and BH-FDR quantify fixed-corpus song-unit uncertainty and multiplicity; they do not create external-population generalizability. Public results remain provisional until dual review and adjudication are complete.
"""


def inspect_existing_human_evidence() -> dict[str, int]:
    context_manifest_files = [
        ROOT / "work/private-chinese-rap-ner-context-audit-v2/evaluation_task_manifest_v2.csv",
        ROOT / "work/private-chinese-rap-ner-context-audit-v2/coverage_task_manifest_v2.csv",
    ]
    legacy_context_file = ROOT / "work/private-chinese-rap-ner-context-audit-v1/context_audit_queue_v1.csv"
    context_files = [
        ROOT / "work/private-chinese-rap-ner-context-reviewer-v1/reviewer_R1_ratings.csv",
        ROOT / "work/private-chinese-rap-ner-context-reviewer-v1/reviewer_R2_ratings.csv",
    ]
    occurrence_files = [
        ROOT / "outputs/private-lexicon-occurrence-review-v1/ratings/reviewer_R1_ratings.csv",
        ROOT / "outputs/private-lexicon-occurrence-review-v1/ratings/reviewer_R2_ratings.csv",
    ]
    completed_context = 0
    context_review_form_tasks = 0
    for path in context_files:
        if path.exists():
            data = pd.read_csv(path, dtype=str, keep_default_na=False)
            context_review_form_tasks = max(context_review_form_tasks, len(data))
            completed_context += int((data.get("decision", pd.Series(dtype=str)).astype(str).str.strip() != "").sum())
    context_tasks = sum(len(pd.read_csv(path, dtype=str, keep_default_na=False)) for path in context_manifest_files if path.exists())
    legacy_context_tasks = len(pd.read_csv(legacy_context_file, dtype=str, keep_default_na=False)) if legacy_context_file.exists() else 0
    completed_occurrence = 0
    occurrence_tasks = 0
    for path in occurrence_files:
        if path.exists():
            data = pd.read_csv(path, dtype=str, keep_default_na=False)
            occurrence_tasks = max(occurrence_tasks, len(data))
            completed_occurrence += int((data.get("match_valid", pd.Series(dtype=str)).astype(str).str.strip() != "").sum())
    entity_registry = ROOT / "work/data-drop/entity-linking-factual-relations-v1/entity_registry.csv"
    relation_registry = ROOT / "work/data-drop/entity-linking-factual-relations-v1/entity_relation_evidence.csv"
    entity_rows = len(pd.read_csv(entity_registry, dtype=str, keep_default_na=False)) if entity_registry.exists() else 0
    relation_rows = len(pd.read_csv(relation_registry, dtype=str, keep_default_na=False)) if relation_registry.exists() else 0
    return {
        "context_tasks": context_tasks,
        "context_review_form_tasks": context_review_form_tasks,
        "legacy_context_tasks": legacy_context_tasks,
        "completed_context_reviews": completed_context,
        "occurrence_tasks": occurrence_tasks,
        "completed_occurrence_reviews": completed_occurrence,
        "verified_entity_rows": entity_rows,
        "verified_relation_rows": relation_rows,
    }


def validate_public(
    public_dir: Path,
    public_frames: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    core_ledger: pd.DataFrame,
) -> dict[str, Any]:
    forbidden_exact = {
        "song_id", "chunk_id", "context_snippet", "analysis_text", "lyrics", "lyric_text",
        "analysis_text_sha256", "song_lyric_content_sha256", "embedding", "vector",
    }
    public_columns = {column for frame in public_frames.values() for column in frame.columns}
    files = [path for path in public_dir.iterdir() if path.is_file()]
    csv_cell_max = 0
    for frame in public_frames.values():
        for column in frame.columns:
            if len(frame):
                csv_cell_max = max(csv_cell_max, int(frame[column].astype(str).str.len().max()))

    entities = public_frames["entity_aggregate_provisional.csv"]
    corpus_inventory = public_frames["entity_inventory_corpuswide_sensitivity.csv"]
    surface_audit = public_frames["surface_type_release_audit.csv"]
    source_links = public_frames["source_label_entity_links_provisional.csv"]
    co_mentions = public_frames["entity_co_mentions_provisional.csv"]
    network_nodes = public_frames["cultural_network_nodes_provisional.csv"]
    network_edges = public_frames["cultural_network_edges_provisional.csv"]
    type_agreement = public_frames["cross_method_agreement_by_type.csv"]
    graph_universe = public_frames["graph_label_universe.csv"]
    association_tests = public_frames["source_label_entity_association_tests.csv"]
    entity_sensitivity = public_frames["shared_text_exclusion_entity_sensitivity.csv"]

    entity_keys = set(zip(entities["entity"], entities["entity_type"]))
    corpus_inventory_keys = set(zip(corpus_inventory["entity"], corpus_inventory["entity_type"]))
    audit_release_keys = set(
        zip(
            surface_audit.loc[surface_audit["primary_release_decision"].eq("RELEASE_PROVISIONAL_PRIMARY"), "entity"],
            surface_audit.loc[surface_audit["primary_release_decision"].eq("RELEASE_PROVISIONAL_PRIMARY"), "proposed_entity_type"],
        )
    )
    audit_withheld_keys = set(
        zip(
            surface_audit.loc[~surface_audit["primary_release_decision"].eq("RELEASE_PROVISIONAL_PRIMARY"), "entity"],
            surface_audit.loc[~surface_audit["primary_release_decision"].eq("RELEASE_PROVISIONAL_PRIMARY"), "proposed_entity_type"],
        )
    )
    core_person_surfaces = set(
        core_ledger.loc[
            core_ledger["canonical_category"].eq("PERSON_NAMED")
            & core_ledger["evidence_tier"].str.startswith("T1"),
            "entity",
        ].astype(str)
    )
    released_person_surfaces = set(
        entities.loc[entities["entity_type"].eq("PERSON_REFERENCE"), "entity"].astype(str)
    )
    source_link_keys = set(zip(source_links["entity"], source_links["entity_type"]))
    co_mention_keys = (
        set(zip(co_mentions["entity_a"], co_mentions["entity_a_type"]))
        | set(zip(co_mentions["entity_b"], co_mentions["entity_b_type"]))
    )
    entity_node_ids = set(network_nodes.loc[network_nodes["node_type"].ne("SOURCE_CREDIT_LABEL"), "node_id"])
    expected_entity_node_ids = {
        stable_id("ENT", entity_type, entity)
        for entity, entity_type in entity_keys
    }
    released_status_frames = [entities, source_links, co_mentions, network_edges]
    eligible_label_ids = set(graph_universe["source_label_id"])
    checks = [
        {"name": "no_completed_human_gold_claim", "passed": summary["human_gold_available"] is False},
        {"name": "two_reproducible_baselines_reported", "passed": summary["baseline_count"] >= 2},
        {"name": "audit_package_has_600_plus_occurrences", "passed": summary["private_audit_tasks"] >= 600},
        {"name": "public_columns_exclude_private_occurrence_fields", "passed": not bool(public_columns & forbidden_exact), "detail": sorted(public_columns & forbidden_exact)},
        {"name": "public_csv_cells_are_bounded_noncontext", "passed": csv_cell_max <= 320, "detail": csv_cell_max},
        {
            "name": "all_public_result_rows_are_provisional",
            "passed": all(
                frame.empty
                or "status" not in frame.columns
                or frame["status"].astype(str).str.startswith("PROVISIONAL_").all()
                for frame in released_status_frames
            ),
        },
        {
            "name": "surface_type_audit_covers_every_quantitative_candidate",
            "passed": len(surface_audit) == summary["public_surface_type_candidates"]
            and surface_audit[["entity", "proposed_entity_type"]].drop_duplicates().shape[0] == len(surface_audit),
        },
        {
            "name": "surface_type_audit_has_only_explicit_release_or_withhold_decisions",
            "passed": set(surface_audit["primary_release_decision"]).issubset(
                {
                    "RELEASE_PROVISIONAL_PRIMARY",
                    "WITHHOLD_SEMANTIC_TYPE",
                    "WITHHOLD_AFTER_SHARED_TEXT_AND_GRAPH_UNIVERSE_SUPPORT_GATE",
                }
            )
            and surface_audit["human_readable_reason"].astype(str).str.strip().ne("").all()
            and surface_audit["allowed_public_claim"].astype(str).str.strip().ne("").all(),
        },
        {"name": "all_public_entities_are_explicitly_released_by_surface_type_audit", "passed": entity_keys == audit_release_keys},
        {"name": "primary_entity_inventory_is_strict_subset_of_frozen_corpuswide_candidates", "passed": entity_keys.issubset(corpus_inventory_keys)},
        {"name": "withheld_surface_type_pairs_do_not_enter_public_entities", "passed": not bool(entity_keys & audit_withheld_keys)},
        {"name": "released_person_references_are_in_t1_named_individual_ledger", "passed": released_person_surfaces.issubset(core_person_surfaces)},
        {"name": "source_label_links_target_only_released_entities", "passed": source_link_keys.issubset(entity_keys)},
        {"name": "graph_label_universe_is_exactly_204_unique_labels", "passed": len(graph_universe) == 204 and graph_universe["source_label_id"].is_unique and graph_universe["source_credit_label"].is_unique},
        {"name": "source_label_links_use_only_204_label_universe", "passed": set(source_links["source_label_id"]).issubset(eligible_label_ids)},
        {"name": "co_mentions_target_only_released_entities", "passed": co_mention_keys.issubset(entity_keys)},
        {"name": "network_has_exactly_the_released_entity_nodes", "passed": entity_node_ids == expected_entity_node_ids},
        {"name": "source_label_entity_edges_are_unique", "passed": not source_links.duplicated(["source_credit_label", "entity", "entity_type"]).any()},
        {"name": "network_edges_are_unique", "passed": not network_edges.duplicated(["source_node_id", "target_node_id", "edge_type"]).any()},
        {
            "name": "released_label_links_pass_support_uncertainty_and_fdr_gates",
            "passed": source_links.empty
            or (
                source_links["release_gate_pass"].astype(bool).all()
                and (source_links["entity_song_units_within_label"] >= 5).all()
                and (source_links["shrunken_risk_ratio"] >= 1.5).all()
                and (source_links["shrunken_risk_ratio_ci95_low_conservative"] > 1.0).all()
                and (source_links["q_value_bh"] <= 0.05).all()
            ),
        },
        {
            "name": "association_test_q_values_are_bounded_and_complete",
            "passed": len(association_tests) == 204 * len(entities)
            and pd.to_numeric(association_tests["q_value_bh"], errors="coerce").notna().all()
            and pd.to_numeric(association_tests["q_value_bh"], errors="coerce").between(0, 1).all(),
        },
        {
            "name": "shared_text_sensitivity_counts_are_monotone",
            "passed": entity_sensitivity.empty
            or (
                (entity_sensitivity["corpuswide_strict_occurrences"] >= entity_sensitivity["graph_universe_before_shared_exclusion_strict_occurrences"]).all()
                and (entity_sensitivity["corpuswide_strict_occurrences"] >= entity_sensitivity["all_labels_after_shared_exclusion_strict_occurrences"]).all()
                and (
                    entity_sensitivity["graph_universe_before_shared_exclusion_strict_occurrences"]
                    >= entity_sensitivity["graph_universe_after_shared_exclusion_strict_occurrences"]
                ).all()
                and (entity_sensitivity["removed_cross_label_shared_text_occurrences_within_graph_universe"] >= 0).all()
            ),
        },
        {
            "name": "unique_line_agreement_counts_and_ratios_are_bounded",
            "passed": (
                (
                    type_agreement["exact_span_type_agreements_on_unique_line_frame"]
                    <= type_agreement["lexicon_candidate_spans_on_unique_line_frame"]
                ).all()
                and (
                    type_agreement["exact_span_type_agreements_on_unique_line_frame"]
                    <= type_agreement["transformer_candidate_spans_on_unique_line_frame"]
                ).all()
                and pd.to_numeric(
                    type_agreement["exact_agreement_per_lexicon_candidate_span"], errors="coerce"
                ).dropna().between(0, 1).all()
                and pd.to_numeric(
                    type_agreement["exact_agreement_per_transformer_candidate_span"], errors="coerce"
                ).dropna().between(0, 1).all()
            ),
        },
        {"name": "network_edges_define_bounded_meaning", "passed": network_edges.empty or network_edges["meaning"].astype(str).str.contains("not biography|not a social relationship", regex=True).all()},
        {"name": "public_artifact_contains_no_html_or_full_context_file", "passed": not any(path.suffix.lower() in {".html", ".htm"} or "context" in path.name.lower() for path in files)},
        {"name": "required_documents_present", "passed": all((public_dir / name).exists() for name in ["README.md", "SCHEMA_AND_ANNOTATION.md", "METHOD.md", "summary.json"])},
        {
            "name": "public_and_private_manifest_file_hashes_verified",
            "passed": summary["public_manifest_hashes_verified"] and summary["private_manifest_hashes_verified"],
            "detail": summary["manifest_hash_failures"],
        },
        {"name": "co_mentions_use_all_eligible_song_unit_denominator", "passed": co_mentions.empty or co_mentions["all_eligible_song_units"].nunique() == 1 and int(co_mentions["all_eligible_song_units"].iloc[0]) == summary["eligible_global_song_units"]},
        {"name": "released_co_mentions_pass_support_and_fdr_gates", "passed": co_mentions.empty or (co_mentions["release_gate_pass"].astype(bool).all() and (co_mentions["unique_song_unit_co_mentions"] >= 5).all() and (co_mentions["q_value_bh"] <= 0.05).all())},
        {"name": "entity_co_mentions_are_not_named_relations", "passed": co_mentions.empty or co_mentions["relation_scope"].astype(str).str.contains("all eligible song units as denominator; not a social relationship", regex=False).all()},
        {"name": "occurrence_and_unique_line_units_are_explicitly_separate", "passed": summary["strict_high_consistency_occurrences"] >= summary["strict_high_consistency_unique_line_spans"] and summary["exact_span_type_agreement_occurrences"] >= summary["exact_span_type_agreement_unique_line_spans"]},
    ]
    return {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "checks": checks,
    }


def validate_private(
    private_dir: Path,
    candidate_union: pd.DataFrame,
    lex_mentions: pd.DataFrame,
    tasks: pd.DataFrame,
    reviewer_r1: pd.DataFrame,
    reviewer_r2: pd.DataFrame,
    agreement: pd.DataFrame,
    shared_hash_audit: pd.DataFrame,
) -> dict[str, Any]:
    context_resolves = all(
        str(row.context_snippet)[int(row.surface_start_in_context) : int(row.surface_end_in_context)] == str(row.candidate_surface)
        for row in tasks.itertuples(index=False)
    )
    checks = [
        {"name": "all_candidate_occurrence_ids_are_source_row_unique", "passed": candidate_union["candidate_id"].is_unique},
        {
            "name": "all_lexicon_source_occurrences_are_preserved",
            "passed": int(candidate_union["candidate_source"].eq("LEXICON_WITH_TRANSFORMER_CHECK").sum()) == len(lex_mentions),
            "detail": {
                "private_lexicon_occurrence_rows": int(candidate_union["candidate_source"].eq("LEXICON_WITH_TRANSFORMER_CHECK").sum()),
                "source_lexicon_occurrence_rows": len(lex_mentions),
            },
        },
        {"name": "candidate_occurrences_have_song_and_chunk_locators", "passed": candidate_union[["song_id", "chunk_id"]].astype(str).apply(lambda column: column.str.strip().ne("").all()).all()},
        {"name": "shared_text_hash_audit_is_private_and_nonempty", "passed": len(shared_hash_audit) > 0 and "analysis_text_sha256" in shared_hash_audit.columns},
        {"name": "tasks_are_unique", "passed": tasks["task_id"].is_unique},
        {"name": "tasks_have_600_plus_occurrences", "passed": len(tasks) >= 600},
        {"name": "contexts_resolve_to_candidate_surface", "passed": context_resolves},
        {"name": "r1_rows_match_tasks", "passed": set(reviewer_r1["task_id"]) == set(tasks["task_id"])},
        {"name": "r2_rows_match_tasks", "passed": set(reviewer_r2["task_id"]) == set(tasks["task_id"])},
        {"name": "review_templates_are_blank", "passed": all((reviewer_r1[field].astype(str).str.strip() == "").all() and (reviewer_r2[field].astype(str).str.strip() == "").all() for field in REVIEW_FIELDS if field != "reviewer")},
        {"name": "agreement_rows_match_tasks", "passed": set(agreement["task_id"]) == set(tasks["task_id"])},
        {"name": "adjudication_fields_present", "passed": all(column in agreement.columns for column in ["adjudication_required", "adjudicated_mention_valid", "adjudicated_entity_type", "adjudicator", "adjudication_notes"])},
        {"name": "private_directory_is_under_work", "passed": (ROOT / "work").resolve() in private_dir.resolve().parents},
    ]
    return {
        "artifact_id": f"private-{ARTIFACT_ID}",
        "version": VERSION,
        "classification": "PRIVATE_LOCAL_ONLY_CONTAINS_LYRIC_CONTEXTS_AND_LOCATORS",
        "generated_at_utc": utc_now(),
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "checks": checks,
    }


def manifest_file_hashes_match(base_dir: Path, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for filename, metadata in manifest.get("files", {}).items():
        path = base_dir / filename
        if not path.exists() or sha256_file(path) != metadata.get("sha256") or path.stat().st_size != metadata.get("bytes"):
            failures.append(filename)
    return not failures, failures


def expected_input_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "builder_code_sha256": sha256_file(Path(__file__)),
        "cleaned_chunks_sha256": sha256_file(args.chunks),
        "songs_sha256": sha256_file(args.songs),
        "safe_lexicon_sha256": sha256_file(args.lexicon),
        "core_reference_ledger_sha256": sha256_file(args.core_ledger),
        "graph_label_registry_sha256": sha256_file(args.graph_labels),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "model_weights_sha256": sha256_file(args.model / "pytorch_model.bin"),
        "tokenizer_vocab_sha256": sha256_file(args.model / "vocab.txt"),
    }


def independent_reconciliation(
    args: argparse.Namespace,
    expected_lexicon_occurrences: int,
) -> dict[str, Any]:
    # This deliberately reloads raw inputs and persisted CSVs instead of using
    # the in-memory build frames, providing an independent grain reconciliation.
    source_chunks, _, _, _, graph_labels = load_inputs(args)
    shared_hashes, _ = compute_shared_text_hash_audit(source_chunks)
    candidates = pd.read_csv(args.private_output / "all_candidate_occurrences_private.csv", dtype=str, keep_default_na=False)
    entities = pd.read_csv(args.public_output / "entity_aggregate_provisional.csv", dtype=str, keep_default_na=False)
    links = pd.read_csv(args.public_output / "source_label_entity_links_provisional.csv", dtype=str, keep_default_na=False)
    co_mentions = pd.read_csv(args.public_output / "entity_co_mentions_provisional.csv", dtype=str, keep_default_na=False)
    graph_universe = pd.read_csv(args.public_output / "graph_label_universe.csv", dtype=str, keep_default_na=False)
    summary = json.loads((args.public_output / "summary.json").read_text(encoding="utf-8"))
    public_manifest = json.loads((args.public_output / "manifest.json").read_text(encoding="utf-8"))
    private_manifest = json.loads((args.private_output / "private_manifest.json").read_text(encoding="utf-8"))

    graph_label_names = set(graph_labels["source_artist_label"])
    entity_keys = set(zip(entities["entity"], entities["entity_type"]))
    strict = candidates[
        candidates["candidate_source"].eq("LEXICON_WITH_TRANSFORMER_CHECK")
        & candidates["strict_high_consistency"].str.lower().eq("true")
        & candidates["source_credit_label"].isin(graph_label_names)
        & ~candidates["analysis_text_sha256"].isin(shared_hashes)
    ].copy()
    strict = strict[
        strict.apply(
            lambda row: (row["candidate_surface"], row["candidate_schema_type"]) in entity_keys,
            axis=1,
        )
    ]
    post_membership, _ = graph_song_membership(
        source_chunks,
        graph_labels,
        shared_hashes,
        exclude_shared=True,
    )
    recon_occurrences = strict.rename(
        columns={"candidate_surface": "surface", "candidate_schema_type": "schema_type"}
    )
    recomputed_association_tests = association_test_table(
        entities,
        recon_occurrences,
        post_membership,
        graph_labels,
    )
    recomputed_links = recomputed_association_tests[recomputed_association_tests["release_gate_pass"]].copy()
    recomputed_link_map = {
        (row.source_credit_label, row.entity, row.entity_type): row
        for row in recomputed_links.itertuples(index=False)
    }
    persisted_link_map = {
        (row.source_credit_label, row.entity, row.entity_type): row
        for row in links.itertuples(index=False)
    }
    full_link_reconciliation = set(recomputed_link_map) == set(persisted_link_map) and all(
        int(recomputed_link_map[key].entity_song_units_within_label)
        == int(persisted_link_map[key].entity_song_units_within_label)
        and abs(float(recomputed_link_map[key].shrunken_risk_ratio) - float(persisted_link_map[key].shrunken_risk_ratio)) <= 1e-6
        and abs(float(recomputed_link_map[key].q_value_bh) - float(persisted_link_map[key].q_value_bh)) <= 1e-6
        for key in persisted_link_map
    )
    recomputed_co_tests = co_mention_test_table(entities, recon_occurrences, post_membership)
    recomputed_co = recomputed_co_tests[recomputed_co_tests["release_gate_pass"]].copy()
    recomputed_co_map = {
        (row.entity_a, row.entity_a_type, row.entity_b, row.entity_b_type): row
        for row in recomputed_co.itertuples(index=False)
    }
    persisted_co_map = {
        (row.entity_a, row.entity_a_type, row.entity_b, row.entity_b_type): row
        for row in co_mentions.itertuples(index=False)
    }
    full_co_reconciliation = set(recomputed_co_map) == set(persisted_co_map) and all(
        int(recomputed_co_map[key].unique_song_unit_co_mentions)
        == int(persisted_co_map[key].unique_song_unit_co_mentions)
        and abs(float(recomputed_co_map[key].npmi) - float(persisted_co_map[key].npmi)) <= 1e-6
        and abs(float(recomputed_co_map[key].q_value_bh) - float(persisted_co_map[key].q_value_bh)) <= 1e-6
        for key in persisted_co_map
    )
    presence = strict[
        ["source_credit_label", "song_lyric_content_sha256", "candidate_surface", "candidate_schema_type"]
    ].drop_duplicates()
    link_support = (
        presence.groupby(["source_credit_label", "candidate_surface", "candidate_schema_type"])["song_lyric_content_sha256"]
        .nunique()
        .to_dict()
    )
    link_support_matches = all(
        int(row.entity_song_units_within_label)
        == int(link_support.get((row.source_credit_label, row.entity, row.entity_type), 0))
        for row in links.itertuples(index=False)
    )

    unit_presence: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in presence.itertuples(index=False):
        unit_presence[row.song_lyric_content_sha256].add((row.candidate_surface, row.candidate_schema_type))
    co_support_matches = all(
        int(row.unique_song_unit_co_mentions)
        == sum(
            1
            for values in unit_presence.values()
            if (row.entity_a, row.entity_a_type) in values and (row.entity_b, row.entity_b_type) in values
        )
        for row in co_mentions.itertuples(index=False)
    )

    public_hash_ok, public_hash_failures = manifest_file_hashes_match(args.public_output, public_manifest)
    private_hash_ok, private_hash_failures = manifest_file_hashes_match(args.private_output, private_manifest)
    actual_inputs = expected_input_hashes(args)
    public_input_hash_ok = all(public_manifest.get("inputs", {}).get(key) == value for key, value in actual_inputs.items())
    private_input_hash_ok = all(private_manifest.get("inputs", {}).get(key) == value for key, value in actual_inputs.items())
    lexicon_rows_preserved = int(candidates["candidate_source"].eq("LEXICON_WITH_TRANSFORMER_CHECK").sum())
    checks = [
        {
            "name": "persisted_private_candidate_map_preserves_every_lexicon_occurrence",
            "passed": lexicon_rows_preserved == expected_lexicon_occurrences,
            "detail": {"persisted": lexicon_rows_preserved, "expected": expected_lexicon_occurrences},
        },
        {"name": "persisted_candidate_ids_are_unique", "passed": candidates["candidate_id"].is_unique},
        {"name": "recomputed_label_link_song_support_matches_public_release", "passed": link_support_matches},
        {"name": "recomputed_co_mention_song_support_matches_public_release", "passed": co_support_matches},
        {"name": "independent_full_association_release_rerun_matches_support_effect_and_q", "passed": full_link_reconciliation},
        {"name": "independent_full_co_mention_release_rerun_matches_support_npmi_and_q", "passed": full_co_reconciliation},
        {"name": "released_links_use_only_graph_eligible_labels", "passed": set(links["source_credit_label"]).issubset(graph_label_names)},
        {"name": "persisted_graph_label_universe_is_exactly_204", "passed": len(graph_universe) == 204 and graph_universe["source_label_id"].is_unique},
        {"name": "public_manifest_file_hashes_match", "passed": public_hash_ok, "detail": public_hash_failures},
        {"name": "private_manifest_file_hashes_match", "passed": private_hash_ok, "detail": private_hash_failures},
        {"name": "public_manifest_input_and_builder_hashes_match", "passed": public_input_hash_ok},
        {"name": "private_manifest_input_and_builder_hashes_match", "passed": private_input_hash_ok},
        {
            "name": "persisted_release_counts_match_summary",
            "passed": len(entities) == summary["counts"]["public_provisional_entities"]
            and len(links) == summary["counts"]["public_source_label_entity_links"]
            and len(co_mentions) == summary["counts"]["public_entity_co_mentions"],
        },
        {
            "name": "no_shared_hash_can_support_reconciled_primary_edges",
            "passed": not strict["analysis_text_sha256"].isin(shared_hashes).any(),
        },
    ]
    return {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "validation_mode": "independent_reload_of_raw_sources_and_persisted_private_occurrence_map",
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    if args.audit_tasks < 600:
        raise ValueError("--audit-tasks must be at least 600")
    reset_output_dir(args.public_output, "chinese-rap-ner-cultural-graph-v1")
    reset_output_dir(args.private_output, "private-chinese-rap-ner-cultural-graph-v1")

    existing_audit = inspect_existing_human_evidence()
    chunks, songs, lexicon, core_ledger, graph_labels = load_inputs(args)
    shared_hashes, shared_hash_audit = compute_shared_text_hash_audit(chunks)
    pattern, lexicon_records = compile_lexicon(lexicon)
    lex_mentions, candidate_instances, background_selected = build_lexicon_frame(
        chunks, pattern, lexicon_records, args.background_lines
    )
    if lex_mentions.empty:
        raise ValueError("Lexicon baseline produced no candidates")

    line_texts: dict[str, str] = {}
    for instance in candidate_instances.values():
        line_texts[instance["line_text_hash"]] = instance["line_text"]
    for text_hash, instance in background_selected.items():
        line_texts[text_hash] = instance["line_text"]
    model_candidates, model_provenance = run_transformer(
        line_texts, args.model, args.batch_size, args.torch_threads
    )

    all_instances = dict(candidate_instances)
    for instance in background_selected.values():
        all_instances[instance["line_instance_id"]] = instance
    model_occurrences = materialize_model_occurrences(all_instances, model_candidates)
    agreements = compare_candidates(lex_mentions, model_candidates)
    private_union = make_private_union(agreements, model_occurrences)
    private_union["cross_label_shared_cleaned_text"] = private_union["analysis_text_sha256"].isin(shared_hashes)
    private_union = add_context(private_union, chunks)
    tasks, sampling_summary = stratified_sample(private_union, args.audit_tasks)
    r1 = reviewer_template(tasks, "R1")
    r2 = reviewer_template(tasks, "R2")
    adjudication = agreement_template(tasks)

    comparison, agreement_by_type = baseline_comparison(
        lex_mentions,
        model_candidates,
        agreements,
        set(lex_mentions["line_text_hash"]),
        set(background_selected),
    )
    public_frames, public_diagnostics = public_aggregates(
        agreements,
        chunks,
        songs,
        core_ledger,
        graph_labels,
        shared_hashes,
    )
    public_frames["baseline_comparison.csv"] = comparison
    public_frames["cross_method_agreement_by_type.csv"] = agreement_by_type

    # Private outputs: raw contexts/locators and review instruments only.
    private_candidates_columns = [
        "candidate_id", "candidate_source", "candidate_surface", "candidate_schema_type", "agreement_state",
        "source_credit_label", "song_id", "chunk_id", "song_lyric_content_sha256", "analysis_text_sha256", "analysis_text_weight",
        "cross_label_shared_cleaned_text",
        "candidate_start_char", "candidate_end_char", "context_start_char", "context_end_char",
        "surface_start_in_context", "surface_end_in_context", "context_snippet", "lexicon_label",
        "transformer_surface", "transformer_label", "transformer_schema_type", "transformer_confidence",
        "span_iou", "strict_high_consistency",
    ]
    private_union[private_candidates_columns].to_csv(args.private_output / "all_candidate_occurrences_private.csv", index=False, encoding="utf-8-sig")
    task_columns = [
        "task_id", "candidate_id", "stratum", "source_credit_label", "song_id", "chunk_id",
        "analysis_text_sha256", "candidate_surface", "candidate_schema_type", "candidate_start_char",
        "candidate_end_char", "context_start_char", "context_end_char", "surface_start_in_context",
        "surface_end_in_context", "context_snippet", "lexicon_label", "transformer_surface",
        "transformer_label", "transformer_schema_type", "transformer_confidence", "span_iou",
        "agreement_state", "selection_rank_sha256",
    ]
    tasks[task_columns].to_csv(args.private_output / "annotation_tasks_private.csv", index=False, encoding="utf-8-sig")
    r1.to_csv(args.private_output / "reviewer_R1_private.csv", index=False, encoding="utf-8-sig")
    r2.to_csv(args.private_output / "reviewer_R2_private.csv", index=False, encoding="utf-8-sig")
    adjudication.to_csv(args.private_output / "agreement_adjudication_private.csv", index=False, encoding="utf-8-sig")
    sampling_summary.to_csv(args.private_output / "sampling_summary_private.csv", index=False, encoding="utf-8-sig")
    shared_hash_audit.to_csv(args.private_output / "shared_text_hash_audit_private.csv", index=False, encoding="utf-8-sig")

    private_validation = validate_private(
        args.private_output,
        private_union,
        lex_mentions,
        tasks,
        r1,
        r2,
        adjudication,
        shared_hash_audit,
    )
    write_json(args.private_output / "private_validation.json", private_validation)
    input_hashes = expected_input_hashes(args)
    private_manifest = {
        "artifact_id": f"private-{ARTIFACT_ID}",
        "version": VERSION,
        "classification": "PRIVATE_LOCAL_ONLY_CONTAINS_LYRIC_CONTEXTS_AND_SOURCE_LOCATORS",
        "generated_at_utc": utc_now(),
        "counts": {
            "all_candidate_occurrences": len(private_union),
            "preserved_lexicon_occurrence_rows": int(
                private_union["candidate_source"].eq("LEXICON_WITH_TRANSFORMER_CHECK").sum()
            ),
            "cross_label_shared_cleaned_text_hashes": len(shared_hashes),
            "annotation_tasks": len(tasks),
            "strata": int(tasks["stratum"].nunique()),
            "r1_completed": 0,
            "r2_completed": 0,
            "adjudicated": 0,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(args.private_output.iterdir()) if path.is_file()
        },
        "inputs": {**input_hashes, **model_provenance},
        "claim_boundary": "unreviewed candidates and reviewer instruments; not gold and not public",
    }
    write_json(args.private_output / "private_manifest.json", private_manifest)

    # Public outputs: aggregates and method only.
    for filename, frame in public_frames.items():
        frame.to_csv(args.public_output / filename, index=False, encoding="utf-8-sig")

    exact_unique_line_spans = int(agreement_by_type["exact_span_type_agreements_on_unique_line_frame"].sum())
    strict_unique_line_spans = int(agreement_by_type["strict_high_consistency_spans_on_unique_line_frame"].sum())
    public_diagnostics.update(
        {
            "exact_span_type_agreement_occurrences": int(
                (agreements["agreement_state"] == "EXACT_SPAN_TYPE_AGREE").sum()
            ),
            "strict_high_consistency_occurrences": int(agreements["strict_high_consistency"].sum()),
            "exact_span_type_agreement_unique_line_spans": exact_unique_line_spans,
            "strict_high_consistency_unique_line_spans": strict_unique_line_spans,
        }
    )
    summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "status": "PROVISIONAL_SHARED_TEXT_EXCLUDED_STATISTICALLY_SCREENED_NOT_HUMAN_VALIDATED",
        "human_gold_available": False,
        "baseline_count": 2,
        "counts": {
            "eligible_chunks": len(chunks),
            "safe_lexicon_surfaces_all_categories": len(lexicon),
            "target_schema_lexicon_surfaces": len(lexicon_records),
            "lexicon_candidate_occurrences": len(lex_mentions),
            "transformer_candidates_on_common_line_frame": len(model_candidates),
            "exact_span_type_agreement_occurrences": int((agreements["agreement_state"] == "EXACT_SPAN_TYPE_AGREE").sum()),
            "strict_high_consistency_occurrences": int(agreements["strict_high_consistency"].sum()),
            "exact_span_type_agreement_unique_line_spans": exact_unique_line_spans,
            "strict_high_consistency_unique_line_spans": strict_unique_line_spans,
            "private_all_candidate_occurrence_rows": len(private_union),
            "private_audit_tasks": len(tasks),
            "public_surface_type_candidates": len(public_frames["surface_type_release_audit.csv"]),
            "withheld_surface_type_candidates": int(
                (~public_frames["surface_type_release_audit.csv"]["primary_release_decision"].eq("RELEASE_PROVISIONAL_PRIMARY")).sum()
            ),
            "corpuswide_sensitivity_entities": len(public_frames["entity_inventory_corpuswide_sensitivity.csv"]),
            "public_provisional_entities": len(public_frames["entity_aggregate_provisional.csv"]),
            "public_source_label_entity_links": len(public_frames["source_label_entity_links_provisional.csv"]),
            "public_entity_co_mentions": len(public_frames["entity_co_mentions_provisional.csv"]),
        },
        "unit_boundary": {
            "occurrence_counts": "Repeated source-row span occurrences; descriptive and not statistically independent samples.",
            "unique_line_counts": "Identical lyric line/span/type combinations counted once; still not human-labelled evaluation units.",
            "association_units": "Distinct source-label/full-song-content membership units after cross-label exact cleaned-text exclusion.",
            "co_mention_denominator": "All distinct eligible full-song-content units in the shared-text-excluded 204-label graph universe, including units with no released entity.",
        },
        "graph_analysis": public_diagnostics,
        "existing_human_evidence_audit": existing_audit,
        "claim_boundary": "Shared-text-excluded aggregate lyric-reference evidence only; not gold, biography, preference, identity, collaboration, influence, affiliation, or social relationship.",
        "private_audit_pointer": "../../work/private-chinese-rap-ner-cultural-graph-v1 (local only; do not distribute)",
    }
    # Convenience keys used by validation, kept out of the serialized summary.
    validation_summary = {
        **summary,
        "private_audit_tasks": len(tasks),
        "public_surface_type_candidates": len(public_frames["surface_type_release_audit.csv"]),
        "eligible_global_song_units": public_diagnostics["eligible_global_song_units"],
        "strict_high_consistency_occurrences": int(agreements["strict_high_consistency"].sum()),
        "strict_high_consistency_unique_line_spans": strict_unique_line_spans,
        "exact_span_type_agreement_occurrences": int((agreements["agreement_state"] == "EXACT_SPAN_TYPE_AGREE").sum()),
        "exact_span_type_agreement_unique_line_spans": exact_unique_line_spans,
    }
    write_json(args.public_output / "summary.json", summary)
    (args.public_output / "SCHEMA_AND_ANNOTATION.md").write_text(schema_document(len(tasks)), encoding="utf-8")
    (args.public_output / "METHOD.md").write_text(
        method_document(chunks, lexicon, tasks, model_provenance, existing_audit, public_diagnostics),
        encoding="utf-8",
    )
    readme = """# Chinese Rap NER + Grounded Cultural Network v1.1

This release contains two reproducible NER candidate baselines, a private source-occurrence-complete 800-task dual-review package, a conservative entity inventory, and a shared-text-excluded cultural graph with song-unit uncertainty and BH-FDR control.

Start with `summary.json`, `release_sensitivity_summary.csv`, and `source_label_entity_links_provisional.csv`; then read `METHOD.md` and `SCHEMA_AND_ANNOTATION.md`. Public CSVs contain aggregate evidence only. No lyrics, contexts, song/chunk IDs, cleaned-text hashes, or embeddings are published.

**Evidence boundary:** there is no completed human occurrence gold. Occurrence counts are repeated corpus spans, not independent samples. Shared exact cleaned text is excluded from label associations and co-mentions. Every released result remains provisional; co-mention is a text pattern, never collaboration, influence, identity, or a social relationship.
"""
    (args.public_output / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "inputs": {**input_hashes, **model_provenance},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(args.public_output.iterdir()) if path.is_file()
        },
        "validation_contract": "validation.json and reconciliation_validation.json are written after this manifest and excluded from its non-self-referential file hash table",
    }
    write_json(args.public_output / "manifest.json", manifest)

    public_manifest_hash_ok, public_manifest_hash_failures = manifest_file_hashes_match(args.public_output, manifest)
    private_manifest_hash_ok, private_manifest_hash_failures = manifest_file_hashes_match(args.private_output, private_manifest)
    validation_summary["public_manifest_hashes_verified"] = public_manifest_hash_ok
    validation_summary["private_manifest_hashes_verified"] = private_manifest_hash_ok
    validation_summary["manifest_hash_failures"] = {
        "public": public_manifest_hash_failures,
        "private": private_manifest_hash_failures,
    }
    validation = validate_public(args.public_output, public_frames, validation_summary, core_ledger)
    write_json(args.public_output / "validation.json", validation)
    reconciliation = independent_reconciliation(args, len(lex_mentions))
    write_json(args.public_output / "reconciliation_validation.json", reconciliation)

    if validation["status"] != "pass" or private_validation["status"] != "pass" or reconciliation["status"] != "pass":
        raise SystemExit("Validation failed")
    print(
        json.dumps(
            {
                "artifact": ARTIFACT_ID,
                "status": "pass",
                "public_output": str(args.public_output),
                "private_output": str(args.private_output),
                "audit_tasks": len(tasks),
                "public_entities": len(public_frames["entity_aggregate_provisional.csv"]),
                "public_label_entity_links": len(public_frames["source_label_entity_links_provisional.csv"]),
                "public_co_mentions": len(public_frames["entity_co_mentions_provisional.csv"]),
                "public_network_edges": len(public_frames["cultural_network_edges_provisional.csv"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
