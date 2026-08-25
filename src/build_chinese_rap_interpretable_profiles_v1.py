#!/usr/bin/env python3
"""Build public-safe, interpretable profiles for the validated repertoire graph.

The graph edge remains a BGE-M3, duplicate-controlled stable-neighbour result.
This artifact adds *post-hoc* evidence that may help a reader interpret a link:

* stable characteristic vocabulary (weighted log-odds versus the corpus),
* dictionary-estimated written line-ending finals,
* song-level writing-form summaries, and
* calibrated pairwise overlap/similarity signals.

No lyric lines, song/chunk identifiers, embeddings, membership tables, or
unreviewed entity occurrences enter the public artifact.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import jieba
import jieba.posseg as pseg
import numpy as np
from pypinyin import Style, lazy_pinyin
from scipy.spatial.distance import jensenshannon


VERSION = "1.0.0"
ARTIFACT_ID = "chinese-rap-interpretable-profiles-v1"
ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "outputs" / "chinese-rap-lyrical-repertoire-graph-v2"
PRIVATE_GRAPH_DIR = ROOT / "work" / "private-chinese-rap-lyrical-repertoire-graph-v2"
CLEAN_DIR = ROOT / "work" / "private-canonical-lyric-text-sidecar-v1"
OUT_DIR = ROOT / "outputs" / ARTIFACT_ID
PRIVATE_OUT_DIR = ROOT / "work" / f"private-{ARTIFACT_ID}"

GRAPH_FILES = (
    "analysis_summary.json",
    "artist_label_registry.csv",
    "artist_repertoire_nodes.csv",
    "artist_repertoire_edges.csv",
    "manifest.json",
    "validation.json",
)

HAN_TOKEN_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]{2,6}$")
HAN_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")
NON_CONTENT_RE = re.compile(r"[^0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]+")

# POS filtering does most of the stop-word work.  This compact list removes
# high-frequency conversational/function items that Chinese rap segmentation
# frequently tags as content words but which are not useful profile labels.
STOP_TERMS = {
    "一个", "一些", "一样", "一直", "不会", "不是", "不能", "不用", "不要", "不过", "为了", "什么",
    "他们", "你们", "我们", "自己", "这个", "那个", "这样", "那样", "怎么", "为什么", "因为", "所以",
    "已经", "还是", "没有", "还有", "可以", "可能", "知道", "觉得", "看到", "听到", "告诉", "开始",
    "现在", "以后", "以前", "今天", "明天", "时候", "真的", "只是", "就是", "但是", "然后", "如果",
    "所有", "一切", "这里", "那里", "哪里", "每个", "每次", "一起", "一直", "一边", "这种", "那种",
    "起来", "出来", "进去", "回来", "下去", "上去", "变成", "成为", "让人", "别人", "有人", "没人",
    "东西", "事情", "感觉", "觉得", "生活", "时间", "世界", "身边", "里面", "外面", "一样", "一点",
    "就让", "看看", "不同", "没法", "感谢", "时刻", "做到", "人们", "确实", "类似", "抓住", "结束",
    "oh", "yeah", "yo", "hey", "ok", "okay",
}

ALLOWED_POS_PREFIXES = ("n", "v", "a", "i", "l", "j")
TERM_PRIOR_MASS = 1000.0
MIN_TERM_EFFECTIVE_COUNT = 4.0
MIN_TERM_SONG_SUPPORT = 5
MIN_TERM_Z = 2.0
MIN_LEAVE_ONE_SONG_STABILITY = 0.80
MAX_PROFILE_TERMS = 8
MAX_INTERNAL_TERMS = 40
MIN_RHYME_LINES = 100
MIN_RHYME_SONGS = 5
PAIR_SIGNAL_PERCENTILE_GATE = 90.0
BOOTSTRAP_REPLICATES = 400


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest_files(root: Path, manifest: dict[str, Any], key: str) -> None:
    records = manifest.get(key, {})
    if not records:
        raise RuntimeError(f"Manifest has no {key}: {root}")
    for name, record in records.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Manifest hash mismatch: {path}")


def require_inputs() -> dict[str, str]:
    missing = [str(GRAPH_DIR / name) for name in GRAPH_FILES if not (GRAPH_DIR / name).is_file()]
    missing += [
        str(path)
        for path in (
            PRIVATE_GRAPH_DIR / "artist_chunk_membership_v2.csv",
            PRIVATE_GRAPH_DIR / "private_manifest.json",
            PRIVATE_GRAPH_DIR / "private_validation.json",
            CLEAN_DIR / "cleaned_analysis_chunks_v1.csv",
            CLEAN_DIR / "private_manifest.json",
            CLEAN_DIR / "private_validation.json",
        )
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing required validated inputs: {missing}")

    graph_validation = load_json(GRAPH_DIR / "validation.json")
    private_graph_validation = load_json(PRIVATE_GRAPH_DIR / "private_validation.json")
    clean_validation = load_json(CLEAN_DIR / "private_validation.json")
    for name, payload in (
        ("public graph", graph_validation),
        ("private graph", private_graph_validation),
        ("clean sidecar", clean_validation),
    ):
        passed = payload.get("status") == "pass" or payload.get("passed") is True
        checks = payload.get("checks", [])
        if not passed or (checks and not all(bool(check.get("passed")) for check in checks)):
            raise RuntimeError(f"The {name} validation is not passing.")

    graph_manifest = load_json(GRAPH_DIR / "manifest.json")
    if graph_manifest.get("artifact_id") != "chinese-rap-lyrical-repertoire-graph-v2":
        raise RuntimeError("Unexpected public graph artifact.")
    validate_manifest_files(GRAPH_DIR, graph_manifest, "output_files")

    private_graph_manifest = load_json(PRIVATE_GRAPH_DIR / "private_manifest.json")
    clean_manifest = load_json(CLEAN_DIR / "private_manifest.json")
    validate_manifest_files(PRIVATE_GRAPH_DIR, private_graph_manifest, "files")
    validate_manifest_files(CLEAN_DIR, clean_manifest, "files")
    return {
        "graph_manifest_sha256": sha256_file(GRAPH_DIR / "manifest.json"),
        "graph_validation_sha256": sha256_file(GRAPH_DIR / "validation.json"),
        "private_graph_manifest_sha256": sha256_file(PRIVATE_GRAPH_DIR / "private_manifest.json"),
        "clean_manifest_sha256": sha256_file(CLEAN_DIR / "private_manifest.json"),
    }


def normalize_line(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return NON_CONTENT_RE.sub("", value)


def tokenize_content_words(value: str) -> list[str]:
    output: list[str] = []
    for token in pseg.cut(unicodedata.normalize("NFKC", value), HMM=False):
        word = token.word.strip()
        flag = token.flag or ""
        if not HAN_TOKEN_RE.fullmatch(word):
            continue
        if word in STOP_TERMS or not flag.startswith(ALLOWED_POS_PREFIXES):
            continue
        output.append(word)
    return output


def terminal_final(line: str) -> str:
    characters = HAN_CHAR_RE.findall(line)
    if not characters:
        return ""
    final = lazy_pinyin(characters[-1], style=Style.FINALS, strict=False)[0]
    return unicodedata.normalize("NFKC", final).lower().replace("ü", "v").strip()


def weighted_log_odds_z(
    term_count: float,
    label_total: float,
    background_count: float,
    background_total: float,
    prior_term: float,
    prior_total: float,
) -> float:
    a = term_count + prior_term
    b = max(1e-12, label_total + prior_total - a)
    c = background_count + prior_term
    d = max(1e-12, background_total + prior_total - c)
    delta = math.log(a / b) - math.log(c / d)
    variance = (1.0 / max(a, 1e-12)) + (1.0 / max(c, 1e-12))
    return delta / math.sqrt(variance)


def percentile_rank(value: float, sorted_values: list[float]) -> float:
    if not sorted_values:
        return 0.0
    left = bisect.bisect_left(sorted_values, value)
    right = bisect.bisect_right(sorted_values, value)
    rank = (left + right) / 2.0
    return 100.0 * rank / len(sorted_values)


def bootstrap_interval(values: list[float], seed_text: str) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    medians = np.median(array[indices], axis=1)
    return (float(np.quantile(medians, 0.05)), float(np.quantile(medians, 0.95)))


def descriptor(percentile: float) -> str:
    if percentile >= 80:
        return "high relative to this corpus"
    if percentile <= 20:
        return "low relative to this corpus"
    return "within the corpus middle range"


def pair_key(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


def support_bucket(minimum_songs: int) -> str:
    if minimum_songs < 10:
        return "5-9 songs"
    if minimum_songs < 20:
        return "10-19 songs"
    if minimum_songs < 40:
        return "20-39 songs"
    return "40+ songs"


def weighted_jaccard(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


def local_final_echo(finals: list[str], window: int = 4) -> tuple[float, float]:
    """Return observed and random-order expected local final echo.

    The expectation is analytic under a within-song permutation of the same
    final inventory, so it adds no simulation noise.
    """

    size = len(finals)
    if size < 2:
        return (0.0, 0.0)
    observed = sum(finals[index] in finals[max(0, index - window) : index] for index in range(1, size)) / (size - 1)
    counts = Counter(finals)
    expected_by_window: dict[int, float] = {}
    for width in range(1, min(window, size - 1) + 1):
        expected = 0.0
        for count in counts.values():
            none_probability = 1.0
            for step in range(width):
                numerator = size - count - step
                denominator = size - 1 - step
                if numerator <= 0:
                    none_probability = 0.0
                    break
                none_probability *= numerator / denominator
            expected += (count / size) * (1.0 - none_probability)
        expected_by_window[width] = expected
    expected = sum(expected_by_window[min(window, index)] for index in range(1, size)) / (size - 1)
    return (float(observed), float(expected))


def load_joined_rows(
    eligible_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int]]:
    clean_rows = read_csv(CLEAN_DIR / "cleaned_analysis_chunks_v1.csv")
    clean_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in clean_rows:
        key = (
            row["song_id"],
            row["chunk_id"],
            row["canonical_lyric_text_sha256"],
            row["analysis_text_sha256"],
        )
        if key in clean_by_key:
            raise RuntimeError("The clean text sidecar has a duplicate composite key.")
        clean_by_key[key] = row
    membership = read_csv(PRIVATE_GRAPH_DIR / "artist_chunk_membership_v2.csv")
    joined: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    counts = Counter()
    for row in membership:
        identifier = row["artist_label_id"]
        if identifier not in eligible_ids or row.get("included_in_primary_centroid") != "true":
            continue
        clean_key = (
            row["song_id"],
            row["chunk_id"],
            row["canonical_lyric_text_sha256"],
            row["analysis_text_sha256"],
        )
        clean = clean_by_key.get(clean_key)
        if clean is None or clean.get("analysis_text_status") != "eligible_clean_text":
            raise RuntimeError("Membership row does not rejoin the clean text sidecar exactly.")
        label = row["source_artist_label"]
        previous = labels.setdefault(identifier, label)
        if previous != label:
            raise RuntimeError("One label identifier maps to multiple source strings.")
        joined.append(
            {
                "id": identifier,
                "label": label,
                "song": row["song_id"],
                "text": clean["analysis_text"],
                "weight": float(row["comparison_text_weight"]),
                "sensitivity": row.get("included_in_shared_text_exclusion_sensitivity") == "true",
            }
        )
        counts[identifier] += 1
    if set(labels) != eligible_ids or any(counts[identifier] == 0 for identifier in eligible_ids):
        raise RuntimeError("Joined private membership does not cover all graph-eligible labels.")
    return joined, labels, dict(counts)


def compute_features() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lineage = require_inputs()
    node_rows = [row for row in read_csv(GRAPH_DIR / "artist_repertoire_nodes.csv") if row["graph_node_eligible"] == "true"]
    edge_rows = read_csv(GRAPH_DIR / "artist_repertoire_edges.csv")
    eligible_ids = {row["artist_label_id"] for row in node_rows}
    node_by_id = {row["artist_label_id"]: row for row in node_rows}
    joined, labels, joined_counts = load_joined_rows(eligible_ids)

    primary_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sensitivity_counts: dict[str, Counter[str]] = defaultdict(Counter)
    primary_totals: Counter[str] = Counter()
    sensitivity_totals: Counter[str] = Counter()
    primary_song_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    sensitivity_song_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    sensitivity_song_totals: dict[str, Counter[str]] = defaultdict(Counter)
    song_lines: dict[str, dict[str, list[tuple[str, str, float]]]] = defaultdict(lambda: defaultdict(list))

    for index, row in enumerate(joined, start=1):
        tokens = tokenize_content_words(row["text"])
        present_terms = set(tokens)
        weight = row["weight"]
        identifier = row["id"]
        song = row["song"]
        for term in present_terms:
            primary_song_counts[identifier][song][term] += weight
        if row["sensitivity"]:
            for term in present_terms:
                sensitivity_song_counts[identifier][song][term] += weight
            for raw_line in row["text"].splitlines():
                line = SPACE_RE.sub(" ", unicodedata.normalize("NFKC", raw_line).strip())
                normalized = normalize_line(line)
                if normalized:
                    song_lines[identifier][song].append((normalized, line, weight))
        if index % 2000 == 0:
            print(f"processed {index:,}/{len(joined):,} clean membership rows", flush=True)

    # Song-capped lexical presence prevents one long verse or repeated hook
    # from multiplying a term's evidence.  Reciprocal exact-text weights still
    # bound support contributed by duplicated units across songs.
    for identifier in eligible_ids:
        for song, counter in primary_song_counts[identifier].items():
            for term, value in list(counter.items()):
                counter[term] = min(1.0, float(value))
                primary_counts[identifier][term] += counter[term]
                primary_totals[identifier] += counter[term]
        for song, counter in sensitivity_song_counts[identifier].items():
            for term, value in list(counter.items()):
                counter[term] = min(1.0, float(value))
                sensitivity_counts[identifier][term] += counter[term]
                sensitivity_totals[identifier] += counter[term]
                sensitivity_song_totals[identifier][song] += counter[term]

    global_primary = Counter()
    global_sensitivity = Counter()
    for counter in primary_counts.values():
        global_primary.update(counter)
    for counter in sensitivity_counts.values():
        global_sensitivity.update(counter)
    global_primary_total = float(sum(global_primary.values()))
    global_sensitivity_total = float(sum(global_sensitivity.values()))
    label_document_frequency = Counter(
        term for identifier in eligible_ids for term in sensitivity_counts[identifier] if sensitivity_counts[identifier][term] > 0
    )

    internal_term_rows: list[dict[str, Any]] = []
    profile_terms: dict[str, list[dict[str, Any]]] = {}
    term_vectors: dict[str, dict[str, float]] = {}
    for identifier in sorted(eligible_ids):
        source_label_normalized = normalize_line(labels[identifier])
        candidates: list[dict[str, Any]] = []
        for term, sensitivity_count in sensitivity_counts[identifier].items():
            if sensitivity_count < MIN_TERM_EFFECTIVE_COUNT:
                continue
            if label_document_frequency[term] / len(eligible_ids) > 0.80:
                continue
            if source_label_normalized and (term in source_label_normalized or source_label_normalized in term):
                continue
            supporting_songs = sum(1 for counter in sensitivity_song_counts[identifier].values() if counter.get(term, 0.0) > 0)
            if supporting_songs < MIN_TERM_SONG_SUPPORT:
                continue
            primary_count = float(primary_counts[identifier].get(term, 0.0))
            prior_primary = max(0.01, TERM_PRIOR_MASS * global_primary[term] / global_primary_total)
            prior_sensitivity = max(0.01, TERM_PRIOR_MASS * global_sensitivity[term] / global_sensitivity_total)
            z_primary = weighted_log_odds_z(
                primary_count,
                float(primary_totals[identifier]),
                float(global_primary[term] - primary_count),
                global_primary_total - float(primary_totals[identifier]),
                prior_primary,
                TERM_PRIOR_MASS,
            )
            z_sensitivity = weighted_log_odds_z(
                float(sensitivity_count),
                float(sensitivity_totals[identifier]),
                float(global_sensitivity[term] - sensitivity_count),
                global_sensitivity_total - float(sensitivity_totals[identifier]),
                prior_sensitivity,
                TERM_PRIOR_MASS,
            )
            if min(z_primary, z_sensitivity) < MIN_TERM_Z:
                continue
            leave_one_scores: list[bool] = []
            for song, song_total in sensitivity_song_totals[identifier].items():
                reduced_count = float(sensitivity_count - sensitivity_song_counts[identifier][song].get(term, 0.0))
                reduced_total = float(sensitivity_totals[identifier] - song_total)
                if reduced_total <= 0:
                    leave_one_scores.append(False)
                    continue
                reduced_z = weighted_log_odds_z(
                    reduced_count,
                    reduced_total,
                    float(global_sensitivity[term] - sensitivity_count),
                    global_sensitivity_total - float(sensitivity_totals[identifier]),
                    prior_sensitivity,
                    TERM_PRIOR_MASS,
                )
                leave_one_scores.append(reduced_z > 0)
            stability = sum(leave_one_scores) / len(leave_one_scores) if leave_one_scores else 0.0
            if stability < MIN_LEAVE_ONE_SONG_STABILITY:
                continue
            record = {
                "term": term,
                "score": min(z_primary, z_sensitivity),
                "primary_z": z_primary,
                "sensitivity_z": z_sensitivity,
                "effective_count": float(sensitivity_count),
                "support_songs": supporting_songs,
                "leave_one_song_stability": stability,
            }
            candidates.append(record)
        candidates.sort(key=lambda item: (-item["score"], -item["support_songs"], item["term"]))
        profile_terms[identifier] = [
            {
                "text": item["term"],
                "score": round(item["score"], 2),
                "supportSongs": item["support_songs"],
                "stability": round(item["leave_one_song_stability"], 3),
            }
            for item in candidates[:MAX_PROFILE_TERMS]
        ]
        term_vectors[identifier] = {item["term"]: item["score"] for item in candidates[:MAX_INTERNAL_TERMS]}
        for rank, item in enumerate(candidates[:MAX_INTERNAL_TERMS], start=1):
            internal_term_rows.append(
                {
                    "artist_label_id": identifier,
                    "source_artist_label": labels[identifier],
                    "rank": rank,
                    "term": item["term"],
                    "stable_log_odds_z": f"{item['score']:.8f}",
                    "primary_z": f"{item['primary_z']:.8f}",
                    "sensitivity_z": f"{item['sensitivity_z']:.8f}",
                    "sensitivity_effective_count": f"{item['effective_count']:.8f}",
                    "support_songs": item["support_songs"],
                    "leave_one_song_stability": f"{item['leave_one_song_stability']:.8f}",
                }
            )

    song_form: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    rhyme_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rhyme_support_songs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    rhyme_line_counts: Counter[str] = Counter()
    rhyme_song_counts: Counter[str] = Counter()
    rhyme_echo_by_song: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for identifier in sorted(eligible_ids):
        for song, entries in song_lines[identifier].items():
            raw_lines = [item for item in entries if item[0]]
            if not raw_lines:
                continue
            normalized_lines = [item[0] for item in raw_lines]
            line_lengths = []
            mixed = []
            unique_by_normalized: dict[str, tuple[str, str, float]] = {}
            for normalized, line, weight in raw_lines:
                han_count = len(HAN_CHAR_RE.findall(line))
                ascii_tokens = len(re.findall(r"[A-Za-z]+", line))
                line_lengths.append(float(han_count + ascii_tokens))
                mixed.append(float(bool(han_count and ASCII_LETTER_RE.search(line))))
                unique_by_normalized.setdefault(normalized, (normalized, line, weight))
            repeated_share = 1.0 - (len(set(normalized_lines)) / len(normalized_lines))
            song_form[identifier][song] = {
                "line_length": float(statistics.median(line_lengths)),
                "repeated_share": repeated_share,
                "mixed_share": float(sum(mixed) / len(mixed)),
                "line_count": float(len(raw_lines)),
            }
            song_final_counts: Counter[str] = Counter()
            ordered_finals: list[str] = []
            for _normalized, line, weight in unique_by_normalized.values():
                final = terminal_final(line)
                if final:
                    song_final_counts[final] += weight
                    ordered_finals.append(final)
                    rhyme_support_songs[identifier][final].add(song)
                    rhyme_line_counts[identifier] += weight
            song_final_total = float(sum(song_final_counts.values()))
            if song_final_total:
                rhyme_song_counts[identifier] += 1
                rhyme_echo_by_song[identifier][song] = local_final_echo(ordered_finals)
                for final, count in song_final_counts.items():
                    # Normalize inside each song before aggregation so one
                    # unusually long transcription cannot dominate a label.
                    rhyme_counts[identifier][final] += float(count) / song_final_total

    form_raw: dict[str, dict[str, float]] = {}
    for identifier in sorted(eligible_ids):
        songs = list(song_form[identifier].values())
        if not songs:
            form_raw[identifier] = {"short": 0.0, "repeat": 0.0, "mix": 0.0}
            continue
        form_raw[identifier] = {
            "short": -float(statistics.median(item["line_length"] for item in songs)),
            "repeat": float(statistics.median(item["repeated_share"] for item in songs)),
            "mix": float(statistics.median(item["mixed_share"] for item in songs)),
        }
    form_sorted = {key: sorted(form_raw[identifier][key] for identifier in eligible_ids) for key in ("short", "repeat", "mix")}
    trait_meta = {
        "short": ("Short-line writing", "Higher means the label tends to use shorter written lines."),
        "repeat": ("Repeated-line use", "Higher means exact written lines recur more often within songs."),
        "mix": ("Chinese-English mixing", "Higher means more written lines contain both Chinese and English."),
    }
    form_traits: dict[str, list[dict[str, Any]]] = {}
    form_percentiles: dict[str, dict[str, float]] = defaultdict(dict)
    for identifier in sorted(eligible_ids):
        traits: list[dict[str, Any]] = []
        songs = song_form[identifier]
        for key in ("short", "repeat", "mix"):
            value = form_raw[identifier][key]
            percentile = percentile_rank(value, form_sorted[key])
            form_percentiles[identifier][key] = percentile
            raw_values = {
                "short": [-item["line_length"] for item in songs.values()],
                "repeat": [item["repeated_share"] for item in songs.values()],
                "mix": [item["mixed_share"] for item in songs.values()],
            }[key]
            low, high = bootstrap_interval(raw_values, f"{identifier}:{key}")
            if key == "short":
                raw_display = -value
                ci_low, ci_high = -high, -low
            else:
                raw_display = value
                ci_low, ci_high = low, high
            traits.append(
                {
                    "key": key,
                    "label": trait_meta[key][0],
                    "percentile": round(percentile, 1),
                    "descriptor": descriptor(percentile),
                    "raw": round(raw_display, 3),
                    "bootstrap90": [round(ci_low, 3), round(ci_high, 3)],
                    "definition": trait_meta[key][1],
                }
            )
        form_traits[identifier] = traits

    all_finals = sorted({final for counter in rhyme_counts.values() for final in counter})
    global_rhyme = Counter()
    for counter in rhyme_counts.values():
        global_rhyme.update(counter)
    global_rhyme_total = float(sum(global_rhyme.values()))
    rhyme_profiles: dict[str, dict[str, Any]] = {}
    rhyme_vectors: dict[str, np.ndarray] = {}
    for identifier in sorted(eligible_ids):
        total = float(sum(rhyme_counts[identifier].values()))
        vector = np.asarray([float(rhyme_counts[identifier][final]) + 0.5 for final in all_finals], dtype=np.float64)
        vector /= vector.sum()
        rhyme_vectors[identifier] = vector
        top = []
        rhyme_ready = rhyme_line_counts[identifier] >= MIN_RHYME_LINES and rhyme_song_counts[identifier] >= MIN_RHYME_SONGS
        echo_pairs = list(rhyme_echo_by_song[identifier].values())
        observed_echo = float(statistics.mean(item[0] for item in echo_pairs)) if echo_pairs else 0.0
        expected_echo = float(statistics.mean(item[1] for item in echo_pairs)) if echo_pairs else 0.0
        echo_lifts = [item[0] - item[1] for item in echo_pairs]
        echo_low, echo_high = bootstrap_interval(echo_lifts, f"{identifier}:ending-echo")
        if rhyme_ready:
            for final, count in rhyme_counts[identifier].most_common(5):
                share = float(count) / total
                global_share = float(global_rhyme[final]) / global_rhyme_total if global_rhyme_total else 0.0
                top.append(
                    {
                        "final": final,
                        "share": round(share, 4),
                        "corpusLift": round(share / global_share, 3) if global_share else 0.0,
                        "supportSongs": len(rhyme_support_songs[identifier][final]),
                    }
                )
        rhyme_profiles[identifier] = {
            "status": "ready" if rhyme_ready else "limited evidence",
            "analysedUniqueLines": int(round(rhyme_line_counts[identifier])),
            "supportSongs": int(rhyme_song_counts[identifier]),
            "top": top,
            "localEcho": {
                "observed": round(observed_echo, 4),
                "randomOrderExpected": round(expected_echo, 4),
                "lift": round(observed_echo - expected_echo, 4),
                "bootstrap90": [round(echo_low, 4), round(echo_high, 4)],
                "label": "Local written-ending echo",
                "definition": "A line ends with the same dictionary final as one of the previous four lines, compared with a random reordering of the same song endings.",
            },
            "boundary": "Dictionary-estimated Mandarin finals from written line endings; repeated identical lines within a song count once. Not performed rhyme or Flow.",
        }

    # Robust-standardize the three form dimensions before pair distances.
    form_matrix: dict[str, np.ndarray] = {}
    for key in ("short", "repeat", "mix"):
        values = np.asarray([form_raw[identifier][key] for identifier in sorted(eligible_ids)], dtype=np.float64)
        median = float(np.median(values))
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) or 1.0
        for identifier in eligible_ids:
            form_matrix.setdefault(identifier, np.zeros(3, dtype=np.float64))[("short", "repeat", "mix").index(key)] = (
                form_raw[identifier][key] - median
            ) / iqr

    identifiers = sorted(eligible_ids)
    pair_rows: list[dict[str, Any]] = []
    by_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for left_index, left in enumerate(identifiers):
        for right in identifiers[left_index + 1 :]:
            minimum_songs = min(int(node_by_id[left]["clean_song_count"]), int(node_by_id[right]["clean_song_count"]))
            bucket = support_bucket(minimum_songs)
            lexical = weighted_jaccard(term_vectors[left], term_vectors[right])
            rhyme = float(1.0 - jensenshannon(rhyme_vectors[left], rhyme_vectors[right], base=2.0))
            form_distance = float(np.mean((form_matrix[left] - form_matrix[right]) ** 2))
            form = math.exp(-0.5 * form_distance)
            record = {
                "key": pair_key(left, right),
                "a": left,
                "b": right,
                "bucket": bucket,
                "lexical": lexical,
                "rhyme": rhyme,
                "form": form,
            }
            pair_rows.append(record)
            by_bucket[bucket]["lexical"].append(lexical)
            by_bucket[bucket]["rhyme"].append(rhyme)
            by_bucket[bucket]["form"].append(form)
    for bucket in by_bucket.values():
        for values in bucket.values():
            values.sort()
    pair_by_key = {row["key"]: row for row in pair_rows}
    for row in pair_rows:
        for signal in ("lexical", "rhyme", "form"):
            row[f"{signal}_percentile"] = percentile_rank(row[signal], by_bucket[row["bucket"]][signal])

    explanations: list[dict[str, Any]] = []
    for edge in edge_rows:
        left = edge["artist_label_id_a"]
        right = edge["artist_label_id_b"]
        key = pair_key(left, right)
        pair = pair_by_key[key]
        shared_terms = sorted(
            set(term_vectors[left]) & set(term_vectors[right]),
            key=lambda term: (-min(term_vectors[left][term], term_vectors[right][term]), term),
        )[:4]
        left_final_shares = {
            item["final"]: item["share"] for item in rhyme_profiles[left]["top"]
        }
        right_final_shares = {
            item["final"]: item["share"] for item in rhyme_profiles[right]["top"]
        }
        shared_finals = sorted(
            set(left_final_shares) & set(right_final_shares),
            key=lambda final: (-min(left_final_shares[final], right_final_shares[final]), final),
        )[:3]
        form_matches = []
        for trait_key, title in (("short", "short-line writing"), ("repeat", "repeated-line use"), ("mix", "Chinese-English mixing")):
            lp = form_percentiles[left][trait_key]
            rp = form_percentiles[right][trait_key]
            if abs(lp - rp) <= 15 and (max(lp, rp) >= 70 or min(lp, rp) <= 30):
                direction = "high" if (lp + rp) / 2 >= 50 else "low"
                form_matches.append(f"{direction} {title}")
        signals: list[dict[str, Any]] = []
        if pair["lexical_percentile"] >= PAIR_SIGNAL_PERCENTILE_GATE and shared_terms:
            signals.append(
                {
                    "kind": "language",
                    "label": "Shared characteristic language",
                    "percentile": round(pair["lexical_percentile"], 1),
                    "items": shared_terms,
                    "summary": "Shared characteristic language: " + ", ".join(shared_terms),
                }
            )
        if (
            pair["rhyme_percentile"] >= PAIR_SIGNAL_PERCENTILE_GATE
            and rhyme_line_counts[left] >= MIN_RHYME_LINES
            and rhyme_line_counts[right] >= MIN_RHYME_LINES
            and rhyme_song_counts[left] >= MIN_RHYME_SONGS
            and rhyme_song_counts[right] >= MIN_RHYME_SONGS
            and shared_finals
        ):
            signals.append(
                {
                    "kind": "lineEnding",
                    "label": "Similar written line endings",
                    "percentile": round(pair["rhyme_percentile"], 1),
                    "items": [f"-{final}" for final in shared_finals],
                    "summary": "Similar written line endings: " + ", ".join(f"-{final}" for final in shared_finals),
                }
            )
        if pair["form_percentile"] >= PAIR_SIGNAL_PERCENTILE_GATE and form_matches:
            signals.append(
                {
                    "kind": "form",
                    "label": "Similar writing-form profile",
                    "percentile": round(pair["form_percentile"], 1),
                    "items": form_matches[:3],
                    "summary": "Similar writing-form profile: " + ", ".join(form_matches[:2]),
                }
            )
        signals.sort(key=lambda item: (-item["percentile"], item["kind"]))
        explanations.append(
            {
                "key": key,
                "a": left,
                "b": right,
                "dominantSignal": signals[0]["kind"] if signals else "semanticOnly",
                "interpretationState": "supported" if signals else "semantic-only",
                "signals": signals[:3],
                "calibration": {
                    "supportBucket": pair["bucket"],
                    "languagePercentile": round(pair["lexical_percentile"], 1),
                    "lineEndingPercentile": round(pair["rhyme_percentile"], 1),
                    "formPercentile": round(pair["form_percentile"], 1),
                },
                "boundary": "These are post-hoc concordant signals, not a causal decomposition of BGE-M3 and not a social relationship.",
            }
        )
    explanations.sort(key=lambda item: item["key"])

    profiles: list[dict[str, Any]] = []
    for identifier in sorted(eligible_ids, key=lambda item: (labels[item].casefold(), item)):
        terms = profile_terms[identifier]
        profile_status = "ready" if len(terms) >= 3 and rhyme_profiles[identifier]["status"] == "ready" else "limited evidence"
        profiles.append(
            {
                "id": identifier,
                "label": labels[identifier],
                "status": profile_status,
                "characteristicTerms": terms,
                "lineEndings": rhyme_profiles[identifier],
                "formTraits": form_traits[identifier],
                "references": {
                    "status": "awaiting human context validation",
                    "items": [],
                    "note": "Screened reference surfaces exist, but occurrence-level NER review is incomplete; no entity claims are displayed.",
                },
                "methodBoundary": "Text-derived profile of a source-credit-labelled corpus slice; not a verified biography, preference, performance style, or identity claim.",
            }
        )

    public_summary = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "counts": {
            "eligible_profiles": len(profiles),
            "ready_profiles": sum(profile["status"] == "ready" for profile in profiles),
            "limited_profiles": sum(profile["status"] != "ready" for profile in profiles),
            "stable_link_explanations": len(explanations),
            "links_with_interpretable_signal": sum(bool(item["signals"]) for item in explanations),
            "semantic_only_links": sum(not item["signals"] for item in explanations),
            "private_joined_membership_rows": len(joined),
            "pairwise_calibration_population": len(pair_rows),
        },
        "methods": {
            "characteristic_language": "Jieba content-word segmentation; duplicate-weighted informative-Dirichlet log-odds versus the corpus; must pass primary and shared-text-exclusion representations, multi-song support, and leave-one-song stability.",
            "written_line_endings": "pypinyin dictionary finals from the last Han character of each nonempty written line; exact repeated lines count once within a song; song-normalized final shares and local four-line echo versus a random-order expectation; no audio inference.",
            "form_traits": "Song-median line length, exact within-song repeated-line share, and mixed Chinese-English line share; corpus percentiles with deterministic song bootstrap intervals.",
            "pair_explanations": f"Post-hoc language, line-ending, and form signals calibrated within minimum-song support strata; public evidence requires at least the {PAIR_SIGNAL_PERCENTILE_GATE:.0f}th percentile.",
            "references": "Withheld until occurrence-level human context validation is complete.",
        },
        "claim_boundary": "Interpretable text evidence for stable lyrical-repertoire links; not verified identities, biography, preference, influence, collaboration, social relationship, performed rhyme, or Flow.",
        "lineage": lineage,
        "software": {
            "jieba": getattr(jieba, "__version__", "unknown"),
            "pypinyin": "runtime package",
            "numpy": np.__version__,
        },
    }
    private_payload = {
        "term_rows": internal_term_rows,
        "pair_rows": pair_rows,
        "joined_counts": joined_counts,
    }
    return {"profiles": profiles, "explanations": explanations, "summary": public_summary}, private_payload, lineage


def public_method() -> str:
    return f"""# Interpretable lyrical-profile method

## Purpose

This artifact explains, rather than replaces, the validated BGE-M3 repertoire
graph. A stable line is still defined only by reciprocal top-5 proximity in the
duplicate-weighted and exact-cross-label-shared-text-exclusion representations.
The language, written-ending, and form signals are post-hoc concordant evidence.

## Characteristic language

Chinese content words are segmented with Jieba and compared with the rest of
the corpus using weighted log-odds with an informative Dirichlet prior (prior
mass {TERM_PRIOR_MASS:.0f}). A displayed term must occur with at least
{MIN_TERM_EFFECTIVE_COUNT:.0f} effective count across at least
{MIN_TERM_SONG_SUPPORT} songs, have z >= {MIN_TERM_Z:.1f} in both the primary
and shared-text-exclusion representations, and remain positive after leaving
out one song in at least {MIN_LEAVE_ONE_SONG_STABILITY:.0%} of checks. These are
*characteristic terms*, not favourite words or beliefs.

## Written line endings

For every nonempty written line, the final Han character is mapped to a
dictionary Mandarin pinyin final. Exact repeated lines within a song count once.
Song-level final distributions are normalized before they are averaged. A local
echo statistic measures whether a line repeats a final found in the preceding
four lines and compares that rate with the analytic expectation after randomly
reordering the same song's endings. This gives an orthographic,
dictionary-estimated ending profile. Polyphones,
dialect pronunciation, delivery, timing, tone, beat, and Flow are not resolved.

## Writing-form traits

Three song-level quantities are summarized for each source-credit label:
written line length, exact line repetition within a song, and the share of lines
that contain both Chinese and English. Public bars are percentiles among the 204
eligible labels. Ninety-percent bootstrap intervals resample songs with a fixed,
label-specific seed.

## Pair explanations

Each of the 20,706 eligible-label pairs receives a language-overlap score, a
Jensen-Shannon similarity of written-ending distributions, and a robust-scaled
form similarity. Percentiles are calibrated within minimum-song support strata.
Only a signal at or above the {PAIR_SIGNAL_PERCENTILE_GATE:.0f}th percentile is
allowed to explain a stable graph edge. A stable semantic edge may correctly
remain `semantic-only` if no interpretable signal passes the gate.

## Entity/reference status

The existing 90-surface reference ledger is a screened candidate vocabulary,
not occurrence-level NER. Because the planned two-reviewer context annotation is
not complete, this artifact publishes no person, place, organization, material,
or event occurrence as a result.

## Public-data boundary

The public artifact contains aggregate source-credit-label profiles and stable
edge explanations only. It contains no lyric lines, song/chunk identifiers,
embeddings, membership tables, or private review contexts.
"""


def data_dictionary() -> str:
    return """# Public data dictionary

## source_label_profiles.json

- `id`, `label`: existing aggregate source-credit label identifier and string.
- `status`: `ready` or `limited evidence` under the declared support gates.
- `characteristicTerms`: stable corpus-distinctive terms with z score, song support, and leave-one-song stability.
- `lineEndings`: dictionary-estimated written final distribution and support.
- `formTraits`: corpus percentile, descriptor, raw aggregate, and song-bootstrap interval.
- `references`: intentionally empty until occurrence-level context annotation passes.

## stable_link_explanations.json

- `key`, `a`, `b`: exact stable graph edge key and endpoints.
- `dominantSignal`: strongest passing post-hoc signal or `semanticOnly`.
- `signals`: zero to three gated language, written-ending, or form signals.
- `calibration`: support stratum and within-stratum pair percentiles.
- `boundary`: interpretation limit repeated on every pair.

No field is a verified biographical, social, preference, audio, or performance claim.
"""


def validate_public_payload(root: Path, expected_counts: dict[str, int]) -> dict[str, Any]:
    profiles = load_json(root / "source_label_profiles.json")
    explanations = load_json(root / "stable_link_explanations.json")
    summary = load_json(root / "analysis_summary.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), **({"detail": detail} if detail else {})})

    check("profile_count_exact", len(profiles) == expected_counts["eligible_profiles"])
    check("profile_ids_unique", len({item["id"] for item in profiles}) == len(profiles))
    check("edge_explanation_count_exact", len(explanations) == expected_counts["stable_link_explanations"])
    check("edge_keys_unique", len({item["key"] for item in explanations}) == len(explanations))
    check("summary_counts_match", summary.get("counts", {}).get("eligible_profiles") == len(profiles) and summary.get("counts", {}).get("stable_link_explanations") == len(explanations))
    check(
        "term_gates_hold",
        all(
            len(term["text"]) <= 6
            and "\n" not in term["text"]
            and term["score"] >= MIN_TERM_Z
            and term["supportSongs"] >= MIN_TERM_SONG_SUPPORT
            and term["stability"] >= MIN_LEAVE_ONE_SONG_STABILITY
            for profile in profiles
            for term in profile["characteristicTerms"]
        ),
    )
    check(
        "trait_ranges_hold",
        all(
            0 <= trait["percentile"] <= 100 and len(trait["bootstrap90"]) == 2
            for profile in profiles
            for trait in profile["formTraits"]
        ),
    )
    check(
        "rhyme_profiles_are_bounded",
        all(
            sum(item["share"] for item in profile["lineEndings"]["top"]) <= 1.0002
            and all(item["supportSongs"] >= 1 for item in profile["lineEndings"]["top"])
            for profile in profiles
        ),
    )
    check(
        "pair_signal_gate_holds",
        all(
            signal["percentile"] >= PAIR_SIGNAL_PERCENTILE_GATE
            for item in explanations
            for signal in item["signals"]
        ),
    )
    check(
        "references_withheld",
        all(profile["references"]["status"] == "awaiting human context validation" and not profile["references"]["items"] for profile in profiles),
    )
    forbidden_keys = {"song_id", "chunk_id", "lyrics", "lyric_text", "analysis_text", "embedding", "membership"}

    def keys(value: Any) -> Iterable[str]:
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key).casefold()
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    public_keys = set(keys(profiles)) | set(keys(explanations))
    check("no_private_payload_keys", not (public_keys & forbidden_keys), detail=str(sorted(public_keys & forbidden_keys)))
    required = {
        "analysis_summary.json",
        "source_label_profiles.json",
        "source_label_profile_summary.csv",
        "stable_link_explanations.json",
        "stable_link_evidence_summary.csv",
        "method.md",
        "data_dictionary.md",
    }
    check("public_inventory_pre_manifest_exact", {path.name for path in root.iterdir() if path.is_file()} == required)
    status = "pass" if all(item["passed"] for item in checks) else "fail"
    return {"artifact_id": ARTIFACT_ID, "version": VERSION, "generated_at_utc": utc_now(), "status": status, "checks": checks}


def build() -> None:
    payload, private_payload, lineage = compute_features()
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    public_stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}-", dir=OUT_DIR.parent))
    private_stage = Path(tempfile.mkdtemp(prefix=f".private-{ARTIFACT_ID}-", dir=PRIVATE_OUT_DIR.parent))
    try:
        atomic_write_json(public_stage / "analysis_summary.json", payload["summary"])
        atomic_write_json(public_stage / "source_label_profiles.json", payload["profiles"])
        atomic_write_json(public_stage / "stable_link_explanations.json", payload["explanations"])
        profile_rows = []
        for profile in payload["profiles"]:
            profile_rows.append(
                {
                    "artist_label_id": profile["id"],
                    "source_artist_label": profile["label"],
                    "profile_status": profile["status"],
                    "characteristic_terms": " | ".join(item["text"] for item in profile["characteristicTerms"]),
                    "top_written_finals": " | ".join(item["final"] for item in profile["lineEndings"]["top"]),
                    "analysed_unique_written_lines": profile["lineEndings"]["analysedUniqueLines"],
                    "short_line_percentile": next(item["percentile"] for item in profile["formTraits"] if item["key"] == "short"),
                    "repeated_line_percentile": next(item["percentile"] for item in profile["formTraits"] if item["key"] == "repeat"),
                    "chinese_english_mixing_percentile": next(item["percentile"] for item in profile["formTraits"] if item["key"] == "mix"),
                    "reference_status": profile["references"]["status"],
                }
            )
        write_csv(
            public_stage / "source_label_profile_summary.csv",
            profile_rows,
            [
                "artist_label_id", "source_artist_label", "profile_status", "characteristic_terms",
                "top_written_finals", "analysed_unique_written_lines", "short_line_percentile",
                "repeated_line_percentile", "chinese_english_mixing_percentile", "reference_status",
            ],
        )
        explanation_rows = []
        for item in payload["explanations"]:
            explanation_rows.append(
                {
                    "edge_key": item["key"],
                    "artist_label_id_a": item["a"],
                    "artist_label_id_b": item["b"],
                    "dominant_signal": item["dominantSignal"],
                    "interpretation_state": item["interpretationState"],
                    "plain_evidence": " | ".join(signal["summary"] for signal in item["signals"]),
                    "language_percentile": item["calibration"]["languagePercentile"],
                    "line_ending_percentile": item["calibration"]["lineEndingPercentile"],
                    "form_percentile": item["calibration"]["formPercentile"],
                    "support_bucket": item["calibration"]["supportBucket"],
                }
            )
        write_csv(
            public_stage / "stable_link_evidence_summary.csv",
            explanation_rows,
            [
                "edge_key", "artist_label_id_a", "artist_label_id_b", "dominant_signal",
                "interpretation_state", "plain_evidence", "language_percentile",
                "line_ending_percentile", "form_percentile", "support_bucket",
            ],
        )
        atomic_write_text(public_stage / "method.md", public_method())
        atomic_write_text(public_stage / "data_dictionary.md", data_dictionary())
        validation = validate_public_payload(public_stage, payload["summary"]["counts"])
        if validation["status"] != "pass":
            raise RuntimeError(f"Public profile validation failed: {validation}")
        atomic_write_json(public_stage / "validation.json", validation)
        public_files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(public_stage.iterdir())
            if path.is_file()
        }
        manifest = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": utc_now(),
            "classification": "public_aggregate_profiles_and_post_hoc_stable_link_evidence",
            "lineage": lineage,
            "files": public_files,
        }
        atomic_write_json(public_stage / "manifest.json", manifest)

        write_csv(
            private_stage / "label_term_score_audit.csv",
            private_payload["term_rows"],
            [
                "artist_label_id", "source_artist_label", "rank", "term", "stable_log_odds_z",
                "primary_z", "sensitivity_z", "sensitivity_effective_count", "support_songs",
                "leave_one_song_stability",
            ],
        )
        private_pair_rows = [
            {
                "edge_key": item["key"], "artist_label_id_a": item["a"], "artist_label_id_b": item["b"],
                "support_bucket": item["bucket"], "language_overlap": f"{item['lexical']:.10f}",
                "written_ending_similarity": f"{item['rhyme']:.10f}", "form_similarity": f"{item['form']:.10f}",
                "language_percentile": f"{item['lexical_percentile']:.8f}",
                "written_ending_percentile": f"{item['rhyme_percentile']:.8f}",
                "form_percentile": f"{item['form_percentile']:.8f}",
            }
            for item in private_payload["pair_rows"]
        ]
        write_csv(
            private_stage / "pair_signal_calibration_audit.csv",
            private_pair_rows,
            [
                "edge_key", "artist_label_id_a", "artist_label_id_b", "support_bucket",
                "language_overlap", "written_ending_similarity", "form_similarity",
                "language_percentile", "written_ending_percentile", "form_percentile",
            ],
        )
        private_manifest = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "classification": "private_local_only_feature_and_pair_calibration_audit_without_lyric_text",
            "generated_at_utc": utc_now(),
            "lineage": lineage,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(private_stage.iterdir())
                if path.is_file()
            },
        }
        atomic_write_json(private_stage / "private_manifest.json", private_manifest)

        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        os.replace(public_stage, OUT_DIR)
        if PRIVATE_OUT_DIR.exists():
            shutil.rmtree(PRIVATE_OUT_DIR)
        os.replace(private_stage, PRIVATE_OUT_DIR)
    except Exception:
        shutil.rmtree(public_stage, ignore_errors=True)
        shutil.rmtree(private_stage, ignore_errors=True)
        raise


def validate_existing() -> None:
    required = {
        "analysis_summary.json", "source_label_profiles.json", "source_label_profile_summary.csv",
        "stable_link_explanations.json", "stable_link_evidence_summary.csv", "method.md",
        "data_dictionary.md", "validation.json", "manifest.json",
    }
    actual = {path.name for path in OUT_DIR.iterdir() if path.is_file()}
    if actual != required:
        raise RuntimeError(f"Unexpected public inventory: {sorted(actual ^ required)}")
    validation = load_json(OUT_DIR / "validation.json")
    if validation.get("status") != "pass" or not all(item.get("passed") for item in validation.get("checks", [])):
        raise RuntimeError("Persisted validation does not pass.")
    manifest = load_json(OUT_DIR / "manifest.json")
    validate_manifest_files(OUT_DIR, manifest, "files")
    print(json.dumps({"status": "pass", "artifact": ARTIFACT_ID, "checks": len(validation["checks"])}, indent=2))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("build", "validate", "all"), default="all")
    args = parser.parse_args()
    if args.command in {"build", "all"}:
        build()
    if args.command in {"validate", "all"}:
        validate_existing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
