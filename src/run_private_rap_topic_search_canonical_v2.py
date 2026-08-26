#!/usr/bin/env python3
"""Private canonical Chinese-rap topic search V2.

This is a *new* local-only search surface.  It does not reuse the old frozen
snapshot, old BGE embeddings, or V1 result cache.  Instead it reads the
canonical corpus contract and its private lyrics snapshot, verifies their
hashes and exact ``(song_id, chunk_id)`` joins, and admits only records with
``downstream_eligibility == 'eligible'``.

The static output deliberately contains no corpus text, artist names, song
titles, IDs, or search index.  Corpus data exists only in the local process
after it starts on 127.0.0.1 and is returned only for a submitted query.

Available modes:
  * exact  — literal phrase lookup in canonical eligible lyrics;
  * topic  — auditable BM25-style lexical retrieval over Chinese character
             bigrams and Latin/digit tokens, with canonical duplicate weights.
  * semantic — local BGE-M3 retrieval, enabled only after the private
               canonical semantic artifact verifies every contract gate.

The runtime never opens a frozen baseline source or its raw vector matrix.
It only loads the new private canonical artifact after it validates canonical
contract/private-content hashes, eligible row order, every row's provenance,
matrix shape/L2 norm, and the local BGE model fingerprint.

From the workspace root:

  & "work\\semantic-ml-venv\\Scripts\\python.exe" work\\run_private_rap_topic_search_canonical_v2.py build
  & "work\\semantic-ml-venv\\Scripts\\python.exe" work\\run_private_rap_topic_search_canonical_v2.py validate
  & "work\\semantic-ml-venv\\Scripts\\python.exe" work\\run_private_rap_topic_search_canonical_v2.py smoke
  & "work\\semantic-ml-venv\\Scripts\\python.exe" work\\run_private_rap_topic_search_canonical_v2.py serve --port 8787
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
import tempfile
import threading
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

from canonical_semantic_embeddings_v1 import (
    CanonicalSemanticArtifactError,
    encode_dense_with_local_model,
    load_local_bge_m3_model,
    load_verified_embedding_artifact,
)


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = ROOT / "outputs" / "chinese-rap-corpus-quality-v1"
CONTRACT_FILE = CANONICAL_DIR / "canonical_corpus_contract.json"
OUTPUT_DIR = ROOT / "outputs" / "private-rap-topic-search-canonical-v2"
INDEX_FILE = OUTPUT_DIR / "index.html"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
VALIDATION_FILE = OUTPUT_DIR / "validation.json"
SMOKE_FILE = OUTPUT_DIR / "smoke_test.json"

ARTIFACT_ID = "private-rap-topic-search-canonical-v2"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8787
MAX_QUERY_LENGTH = 120
MAX_RESULTS = 24
MAX_SNIPPET_CHARS = 260

SONG_REQUIRED_COLUMNS = {
    "canonical_corpus_id",
    "canonical_corpus_contract_version",
    "song_id",
    "canonical_artist",
    "canonical_song_title",
    "downstream_eligibility",
    "downstream_usage_status",
}
CHUNK_REQUIRED_COLUMNS = {
    "song_id",
    "chunk_id",
    "canonical_lyric_text_sha256",
    "downstream_eligibility",
    "downstream_usage_status",
    "is_cross_song_duplicate_text",
    "cross_song_duplicate_text_group_id",
    "cross_song_duplicate_text_song_count",
    "analysis_text_weight",
}
LYRIC_REQUIRED_COLUMNS = {
    "canonical_corpus_id",
    "canonical_corpus_contract_version",
    "canonical_artist",
    "canonical_song_title",
    "song_id",
    "chunk_id",
    "text",
    "canonical_lyric_text_sha256",
    "downstream_eligibility",
    "downstream_usage_status",
    "is_cross_song_duplicate_text",
    "cross_song_duplicate_text_group_id",
    "cross_song_duplicate_text_song_count",
    "analysis_text_weight",
}

WHITESPACE = re.compile(r"\s+")
CHINESE_RUN = re.compile(r"[\u3400-\u9fff]+")
LATIN_TOKEN = re.compile(r"[a-z0-9]+(?:['’._+-][a-z0-9]+)*", flags=re.IGNORECASE)


class CanonicalTopicSearchError(RuntimeError):
    """A fail-closed error that can be safely shown in the local UI."""


@dataclass(frozen=True)
class CanonicalRow:
    """One eligible canonical lyric chunk held only in process memory."""

    index: int
    song_id: str
    chunk_id: str
    artist: str
    title: str
    text: str
    folded_text: str
    canonical_text_sha256: str
    analysis_weight: float
    duplicate_group_id: str
    duplicate_song_count: int


@dataclass(frozen=True)
class CanonicalCorpus:
    corpus_id: str
    contract_version: str
    contract_sha256: str
    private_content_sha256: str
    total_private_chunks: int
    eligible_rows: tuple[CanonicalRow, ...]
    withheld_rows: int
    duplicate_group_count: int


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_display(value: Any) -> str:
    return str(value or "").strip()


def normalized_for_lookup(value: str) -> str:
    """Normalize only inside the retrieval index; never modify display fields."""

    return WHITESPACE.sub("", unicodedata.normalize("NFKC", value)).casefold()


def resolve_under_root(raw_path: str, *, base: Path) -> Path:
    candidate = (base / raw_path).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise CanonicalTopicSearchError("规范数据契约包含工作区外路径；已拒绝读取。") from exc
    return candidate


def read_csv_strict(path: Path, required: set[str], label: str) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise CanonicalTopicSearchError(f"找不到{label}。")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            missing = sorted(required - set(fields))
            raise CanonicalTopicSearchError(f"{label}的列结构不符合 canonical contract；缺少 {', '.join(missing) or '必要列'}。")
        return [
            {field: "" if value is None else str(value) for field, value in row.items()}
            for row in reader
        ], fields


def read_contract() -> tuple[dict[str, Any], str]:
    if not CONTRACT_FILE.is_file():
        raise CanonicalTopicSearchError("找不到 canonical_corpus_contract.json；拒绝回退到旧快照。")
    try:
        contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CanonicalTopicSearchError("canonical corpus contract 不是合法 JSON。") from exc
    required = {
        "canonical_corpus_id",
        "contract_version",
        "authoritative_inputs",
        "downstream_eligibility_definition",
        "cross_song_text_deduplication",
        "legacy_baseline",
    }
    if not isinstance(contract, dict) or not required.issubset(contract):
        raise CanonicalTopicSearchError("canonical corpus contract 缺少必要定义。")
    inputs = contract["authoritative_inputs"]
    if not isinstance(inputs, dict) or not {"song_registry", "chunk_registry", "lyric_content_snapshot"}.issubset(inputs):
        raise CanonicalTopicSearchError("canonical corpus contract 的输入定义不完整。")
    if contract["legacy_baseline"].get("role") != "frozen_lineage_baseline_only_not_a_downstream_canonical_input":
        raise CanonicalTopicSearchError("canonical corpus contract 未明确禁止旧基线作为下游输入。")
    return contract, sha256_file(CONTRACT_FILE)


def parse_positive_weight(value: str, *, song_id: str, chunk_id: str) -> float:
    try:
        weight = float(value)
    except ValueError as exc:
        raise CanonicalTopicSearchError(f"{song_id}/{chunk_id} 的 analysis_text_weight 不是数字。") from exc
    if not math.isfinite(weight) or weight <= 0 or weight > 1:
        raise CanonicalTopicSearchError(f"{song_id}/{chunk_id} 的 analysis_text_weight 不在 (0, 1]。")
    return weight


def load_canonical_corpus() -> CanonicalCorpus:
    """Read, reconcile, and filter only contract-approved canonical inputs."""

    contract, contract_sha = read_contract()
    inputs = contract["authoritative_inputs"]
    song_path = resolve_under_root(str(inputs["song_registry"].get("path", "")), base=CANONICAL_DIR)
    chunk_path = resolve_under_root(str(inputs["chunk_registry"].get("path", "")), base=CANONICAL_DIR)
    lyric_path = resolve_under_root(str(inputs["lyric_content_snapshot"].get("path", "")), base=ROOT)
    expected_private_sha = str(inputs["lyric_content_snapshot"].get("sha256", ""))
    if not expected_private_sha or sha256_file(lyric_path) != expected_private_sha:
        raise CanonicalTopicSearchError("私有 canonical 歌词快照哈希与 contract 不一致；已停止检索。")
    if inputs["lyric_content_snapshot"].get("classification") != "private_local_only_full_lyrics":
        raise CanonicalTopicSearchError("canonical 歌词快照没有标为 private local only；已停止检索。")

    song_rows, _ = read_csv_strict(song_path, SONG_REQUIRED_COLUMNS, "canonical 歌曲注册表")
    chunk_rows, _ = read_csv_strict(chunk_path, CHUNK_REQUIRED_COLUMNS, "canonical 歌词块注册表")
    lyric_rows, _ = read_csv_strict(lyric_path, LYRIC_REQUIRED_COLUMNS, "私有 canonical 歌词快照")

    corpus_id = str(contract["canonical_corpus_id"])
    contract_version = str(contract["contract_version"])
    song_by_id: dict[str, dict[str, str]] = {}
    for row in song_rows:
        song_id = normalize_display(row["song_id"])
        if not song_id or song_id in song_by_id:
            raise CanonicalTopicSearchError("canonical 歌曲注册表含空或重复 song_id。")
        if row["canonical_corpus_id"] != corpus_id or row["canonical_corpus_contract_version"] != contract_version:
            raise CanonicalTopicSearchError("歌曲注册表的 canonical corpus 标识不匹配。")
        song_by_id[song_id] = row

    chunk_by_key: dict[tuple[str, str], dict[str, str]] = {}
    duplicate_group_weight: dict[str, float] = defaultdict(float)
    for row in chunk_rows:
        key = (normalize_display(row["song_id"]), normalize_display(row["chunk_id"]))
        if not all(key) or key in chunk_by_key:
            raise CanonicalTopicSearchError("canonical 歌词块注册表含空或重复 (song_id, chunk_id)。")
        if key[0] not in song_by_id:
            raise CanonicalTopicSearchError("歌词块注册表含未在歌曲注册表中的 song_id。")
        parse_positive_weight(row["analysis_text_weight"], song_id=key[0], chunk_id=key[1])
        group = normalize_display(row["cross_song_duplicate_text_group_id"])
        is_duplicate = row["is_cross_song_duplicate_text"].lower() == "true"
        if bool(group) != is_duplicate:
            raise CanonicalTopicSearchError("跨歌重复文本字段在歌词块注册表中不一致。")
        if group:
            duplicate_group_weight[group] += float(row["analysis_text_weight"])
        chunk_by_key[key] = row
    for group, total_weight in duplicate_group_weight.items():
        if not math.isclose(total_weight, 1.0, abs_tol=1e-8):
            raise CanonicalTopicSearchError(f"重复文本组 {group} 的 analysis_text_weight 不等于 1。")

    active_rows: list[CanonicalRow] = []
    eligible_keys: set[tuple[str, str]] = set()
    seen_lyric_keys: set[tuple[str, str]] = set()
    withheld_rows = 0
    for raw in lyric_rows:
        key = (normalize_display(raw["song_id"]), normalize_display(raw["chunk_id"]))
        if not all(key) or key in seen_lyric_keys:
            raise CanonicalTopicSearchError("私有 canonical 歌词快照含空或重复 (song_id, chunk_id)。")
        seen_lyric_keys.add(key)
        chunk = chunk_by_key.get(key)
        song = song_by_id.get(key[0])
        if chunk is None or song is None:
            raise CanonicalTopicSearchError("私有 canonical 歌词未能以精确键连回注册表；已停止检索。")
        if raw["canonical_corpus_id"] != corpus_id or raw["canonical_corpus_contract_version"] != contract_version:
            raise CanonicalTopicSearchError("私有歌词快照的 canonical corpus 标识不匹配。")
        text = str(raw["text"])
        if not text.strip() or sha256_text(text) != raw["canonical_lyric_text_sha256"]:
            raise CanonicalTopicSearchError("私有 canonical 歌词文本或 SHA-256 不一致。")
        if raw["canonical_lyric_text_sha256"] != chunk["canonical_lyric_text_sha256"]:
            raise CanonicalTopicSearchError("私有 canonical 歌词与歌词块注册表的文本指纹不一致。")
        for field in (
            "downstream_eligibility",
            "downstream_usage_status",
            "is_cross_song_duplicate_text",
            "cross_song_duplicate_text_group_id",
            "cross_song_duplicate_text_song_count",
            "analysis_text_weight",
        ):
            if raw[field] != chunk[field]:
                raise CanonicalTopicSearchError(f"私有歌词与歌词块注册表的 {field} 不一致。")
        if raw["canonical_artist"] != song["canonical_artist"] or raw["canonical_song_title"] != song["canonical_song_title"]:
            raise CanonicalTopicSearchError("私有歌词与歌曲注册表的规范显示名不一致。")
        if raw["downstream_eligibility"] != song["downstream_eligibility"]:
            raise CanonicalTopicSearchError("私有歌词与歌曲注册表的下游资格不一致。")
        if raw["downstream_eligibility"] == "eligible":
            if not raw["canonical_artist"].strip() or not raw["canonical_song_title"].strip():
                raise CanonicalTopicSearchError("可用 canonical 歌词缺少规范歌手或歌名。")
            weight = parse_positive_weight(raw["analysis_text_weight"], song_id=key[0], chunk_id=key[1])
            active_rows.append(
                CanonicalRow(
                    index=len(active_rows),
                    song_id=key[0],
                    chunk_id=key[1],
                    artist=raw["canonical_artist"],
                    title=raw["canonical_song_title"],
                    text=text,
                    folded_text=normalized_for_lookup(text),
                    canonical_text_sha256=raw["canonical_lyric_text_sha256"],
                    analysis_weight=weight,
                    duplicate_group_id=normalize_display(raw["cross_song_duplicate_text_group_id"]),
                    duplicate_song_count=int(raw["cross_song_duplicate_text_song_count"] or "0"),
                )
            )
            eligible_keys.add(key)
        elif raw["downstream_eligibility"] == "withhold":
            withheld_rows += 1
        else:
            raise CanonicalTopicSearchError("私有歌词存在未定义的 downstream_eligibility。")

    if set(chunk_by_key) != seen_lyric_keys:
        raise CanonicalTopicSearchError("私有歌词与歌词块注册表的键集合不一致。")
    expected_eligible = {
        key for key, row in chunk_by_key.items()
        if row["downstream_eligibility"] == "eligible"
    }
    if eligible_keys != expected_eligible:
        raise CanonicalTopicSearchError("下游 eligible 过滤结果与歌词块注册表不一致。")
    if not active_rows:
        raise CanonicalTopicSearchError("canonical contract 没有可用于下游检索的歌词。")

    return CanonicalCorpus(
        corpus_id=corpus_id,
        contract_version=contract_version,
        contract_sha256=contract_sha,
        private_content_sha256=expected_private_sha,
        total_private_chunks=len(lyric_rows),
        eligible_rows=tuple(active_rows),
        withheld_rows=withheld_rows,
        duplicate_group_count=len(duplicate_group_weight),
    )


def tokens_for_bm25(value: str) -> list[str]:
    """Small transparent tokenizer: Chinese unigrams/bigrams + Latin tokens."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens: list[str] = []
    for match in CHINESE_RUN.finditer(normalized):
        run = match.group(0)
        tokens.extend(f"c1:{char}" for char in run)
        tokens.extend(f"c2:{run[index:index + 2]}" for index in range(len(run) - 1))
    tokens.extend(f"w:{match.group(0)}" for match in LATIN_TOKEN.finditer(normalized))
    return tokens


def safe_snippet(text: str, query: str, *, exact: bool) -> str:
    compact = WHITESPACE.sub(" ", text).strip()
    if len(compact) <= MAX_SNIPPET_CHARS:
        return compact
    start = 0
    query_compact = WHITESPACE.sub("", query)
    if exact and query_compact:
        without_spaces = WHITESPACE.sub("", compact)
        match = without_spaces.casefold().find(query_compact.casefold())
        if match >= 0:
            # Character offsets after removing whitespace are close enough to
            # bias the evidence window without rewriting/copying corpus text.
            start = max(0, match - 70)
    end = min(len(compact), start + MAX_SNIPPET_CHARS)
    return ("…" if start else "") + compact[start:end] + ("…" if end < len(compact) else "")


class LexicalIndex:
    """In-memory BM25 index with no on-disk corpus cache or external model."""

    def __init__(self, rows: tuple[CanonicalRow, ...]) -> None:
        self.rows = rows
        self.term_frequency: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for row in rows:
            counts = Counter(tokens_for_bm25(row.text))
            self.term_frequency.append(counts)
            length = max(1, sum(counts.values()))
            self.doc_lengths.append(length)
            for token, count in counts.items():
                self.postings[token].append((row.index, count))
                document_frequency[token] += 1
        self.document_frequency = document_frequency
        self.average_length = sum(self.doc_lengths) / len(self.doc_lengths)

    def topic_search(self, query: str) -> tuple[list[tuple[CanonicalRow, float, int]], int]:
        query_tokens = tokens_for_bm25(query)
        unique_tokens = list(dict.fromkeys(query_tokens))
        if not unique_tokens:
            return [], 0
        scores: defaultdict[int, float] = defaultdict(float)
        matched_token_counts: defaultdict[int, set[str]] = defaultdict(set)
        signal_token_matches: defaultdict[int, set[str]] = defaultdict(set)
        # Chinese single characters are useful for one-character queries, but
        # very common in a longer phrase.  When the query has a Chinese bigram
        # or a Latin/digit word, require at least one such higher-signal term
        # so a match on only “的/我/里” cannot enter the topic results.
        requires_signal = any(token.startswith(("c2:", "w:")) for token in unique_tokens)
        k1, b = 1.2, 0.75
        total_docs = len(self.rows)
        for token in unique_tokens:
            postings = self.postings.get(token, [])
            if not postings:
                continue
            df = self.document_frequency[token]
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for row_index, tf in postings:
                denominator = tf + k1 * (1 - b + b * self.doc_lengths[row_index] / self.average_length)
                scores[row_index] += idf * (tf * (k1 + 1) / denominator)
                matched_token_counts[row_index].add(token)
                if token.startswith(("c2:", "w:")):
                    signal_token_matches[row_index].add(token)
        query_folded = normalized_for_lookup(query)
        for row_index in list(scores):
            row = self.rows[row_index]
            # Exact phrase presence receives a documented small bonus; this
            # keeps a direct phrase above unrelated one-token overlaps.
            phrase_bonus = 2.0 if query_folded and query_folded in row.folded_text else 0.0
            scores[row_index] = (scores[row_index] + phrase_bonus) * row.analysis_weight
        ranked = sorted(
            (
                (self.rows[index], value, len(matched_token_counts[index]))
                for index, value in scores.items()
                if value > 0 and (not requires_signal or bool(signal_token_matches[index]))
            ),
            key=lambda item: (-item[1], item[0].song_id, item[0].chunk_id),
        )
        return ranked, len(ranked)


def unique_result_rows(candidates: Iterable[tuple[CanonicalRow, float, int]]) -> list[tuple[CanonicalRow, float, int]]:
    """One representative per canonical duplicate group and one block per song."""

    results: list[tuple[CanonicalRow, float, int]] = []
    seen_songs: set[str] = set()
    seen_content: set[str] = set()
    for row, score, matched in candidates:
        content_key = row.duplicate_group_id or row.canonical_text_sha256
        if row.song_id in seen_songs or content_key in seen_content:
            continue
        seen_songs.add(row.song_id)
        seen_content.add(content_key)
        results.append((row, score, matched))
        if len(results) >= MAX_RESULTS:
            break
    return results


def result_payload(row: CanonicalRow, *, query: str, mode: str, score: float | None, matched_terms: int) -> dict[str, Any]:
    duplicate_note = ""
    if row.duplicate_group_id and row.duplicate_song_count > 1:
        duplicate_note = f"这段标准化文本同时出现在 {row.duplicate_song_count} 首作品中；检索只展示一个代表，避免重复放大。"
    if mode == "exact":
        evidence_label = "歌词中包含你的原句"
    elif mode == "topic":
        evidence_label = f"与输入共享 {matched_terms} 个中文词片或英文词"
    elif mode == "semantic":
        evidence_label = "与输入的表达意思更接近；跨歌重复文本已降权且只展示一个代表"
    else:  # defensive: callers only admit explicit modes
        raise CanonicalTopicSearchError("未知的结果证据方式。")
    return {
        "artist": row.artist,
        "song_title": row.title,
        "snippet": safe_snippet(row.text, query, exact=mode == "exact"),
        "evidence_label": evidence_label,
        "duplicate_note": duplicate_note,
        "score": round(float(score), 4) if score is not None else None,
    }


class CanonicalSemanticIndex:
    """Local semantic retrieval gated by a complete canonical vector contract.

    The runtime does not open any frozen baseline CSV, old vector file, or old
    row mapping.  It asks the private canonical artifact loader to validate
    hashes, eligibility, exact row order, provenance, shape, and L2 norms
    before it will ever load the local BGE query encoder.
    """

    def __init__(self, corpus: CanonicalCorpus) -> None:
        self.corpus = corpus
        self._artifact: Any | None = None
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._status = "unavailable_contract_validation_failed"
        self._error: str | None = None
        try:
            self._artifact = load_verified_embedding_artifact(corpus)
            self._status = "verified_pending_local_query_encoder"
        except CanonicalSemanticArtifactError as exc:
            # Fail closed.  Exact and transparent lexical search remain useful.
            self._error = str(exc)

    @property
    def status(self) -> str:
        return self._status

    @property
    def verified(self) -> bool:
        return self._artifact is not None and self._status != "unavailable_contract_validation_failed"

    @property
    def artifact_id(self) -> str | None:
        return self._artifact.artifact_id if self._artifact is not None else None

    def _ensure_model(self) -> Any:
        if self._artifact is None:
            raise CanonicalTopicSearchError("按意思找尚未通过 canonical 向量契约验证；请改用主题线索或原句搜索。")
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            self._status = "loading_local_bge_m3"
            try:
                self._model = load_local_bge_m3_model()
                self._status = "ready"
                return self._model
            except CanonicalSemanticArtifactError as exc:
                self._error = str(exc)
                self._status = "unavailable_local_model_failed"
                self._model = None
                raise CanonicalTopicSearchError("本机 canonical 语义模型暂时不可用；请改用主题线索或原句搜索。") from exc

    def search(self, query: str) -> tuple[list[tuple[CanonicalRow, float, int]], int]:
        model = self._ensure_model()
        assert self._artifact is not None
        try:
            import numpy as np

            query_vector = encode_dense_with_local_model(model, [query], batch_size=1)[0]
            scores = np.asarray(self._artifact.vector_matrix @ query_vector, dtype=np.float32)
        except CanonicalSemanticArtifactError as exc:
            raise CanonicalTopicSearchError("本机 canonical 语义查询失败；请改用主题线索或原句搜索。") from exc
        # Apply the canonical duplicate-text analysis weight before ranking.
        # This prevents the same text copied across songs from gaining extra
        # exposure merely through repeated vector rows.
        weights = np.asarray([row.analysis_weight for row in self.corpus.eligible_rows], dtype=np.float32)
        scores *= weights
        if not bool(np.isfinite(scores).all()):
            raise CanonicalTopicSearchError("本机 canonical 语义查询返回了无效分数。")
        limit = min(len(self.corpus.eligible_rows), MAX_RESULTS * 35)
        candidate_indexes = np.argpartition(scores, -limit)[-limit:]
        candidate_indexes = candidate_indexes[np.argsort(scores[candidate_indexes])[::-1]]
        ranked = [
            (self.corpus.eligible_rows[int(index)], float(scores[int(index)]), 0)
            for index in candidate_indexes
        ]
        return ranked, len(ranked)


class CanonicalTopicApp:
    def __init__(self) -> None:
        self.corpus = load_canonical_corpus()
        self.index = LexicalIndex(self.corpus.eligible_rows)
        self.semantic_index = CanonicalSemanticIndex(self.corpus)

    def exact_search(self, query: str) -> tuple[list[dict[str, Any]], int]:
        folded = normalized_for_lookup(query)
        if not folded:
            return [], 0
        candidates: list[tuple[CanonicalRow, float, int]] = []
        for row in self.corpus.eligible_rows:
            occurrences = row.folded_text.count(folded)
            if occurrences:
                # Weight implements the contract's no-double-count rule.
                candidates.append((row, float(occurrences) * row.analysis_weight, 1))
        candidates.sort(key=lambda item: (-item[1], len(item[0].text), item[0].song_id, item[0].chunk_id))
        unique = unique_result_rows(candidates)
        return [result_payload(row, query=query, mode="exact", score=score, matched_terms=matched) for row, score, matched in unique], len(candidates)

    def topic_search(self, query: str) -> tuple[list[dict[str, Any]], int]:
        ranked, candidate_count = self.index.topic_search(query)
        unique = unique_result_rows(ranked)
        return [result_payload(row, query=query, mode="topic", score=score, matched_terms=matched) for row, score, matched in unique], candidate_count

    def semantic_search(self, query: str) -> tuple[list[dict[str, Any]], int]:
        ranked, candidate_count = self.semantic_index.search(query)
        unique = unique_result_rows(ranked)
        return [result_payload(row, query=query, mode="semantic", score=score, matched_terms=matched) for row, score, matched in unique], candidate_count

    def search(self, mode: str, query: str) -> dict[str, Any]:
        query = normalize_display(query)
        if not query or len(query) > MAX_QUERY_LENGTH:
            raise CanonicalTopicSearchError("请输入 1–120 个字符。")
        if mode == "exact":
            results, raw_candidates = self.exact_search(query)
        elif mode == "topic":
            results, raw_candidates = self.topic_search(query)
        elif mode == "semantic":
            results, raw_candidates = self.semantic_search(query)
        else:
            raise CanonicalTopicSearchError("未知的搜索方式。")
        return {
            "mode": mode,
            "results": results,
            "raw_candidate_count": raw_candidates,
            "corpus_note": "只从已清洗、核对并允许分析的本地歌词中寻找；跨歌重复的段落会合并处理，每次只展示一个代表。",
            "semantic_contract_status": self.semantic_index.status,
        }


def build_html() -> str:
    """A data-free static shell.  No corpus payload is interpolated here."""

    return r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>中文说唱 · 主题找歌</title>
  <style>
    :root{--bg:#080c13;--panel:#111a26;--panel2:#0c131d;--ink:#eef4fb;--muted:#a9b6c9;--line:#2b4054;--hot:#ffbd55;--mint:#77dfbf;--danger:#ff887e;--soft:#183040}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 86% -5%,#203c52 0,transparent 38%),radial-gradient(circle at 5% 20%,#31274c 0,transparent 31%),var(--bg);font:16px/1.55 "Microsoft YaHei UI","Noto Sans CJK SC",system-ui,sans-serif}main{width:min(1110px,calc(100% - 32px));margin:auto;padding:54px 0 92px}.eyebrow{color:var(--hot);font-size:12px;letter-spacing:.15em;font-weight:900}h1{font-size:clamp(39px,7vw,75px);letter-spacing:-.055em;line-height:1;margin:14px 0 17px;max-width:880px}.lede{max-width:770px;color:var(--muted);font-size:17px;line-height:1.75}.promise{border-left:4px solid var(--mint);padding:14px 18px;background:rgba(119,223,191,.08);border-radius:0 14px 14px 0;margin:27px 0;color:#dcecf0}.searchbox{margin:28px 0 12px;padding:19px;border:1px solid var(--line);border-radius:21px;background:rgba(17,26,38,.93);box-shadow:0 25px 70px rgba(0,0,0,.28)}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}input{flex:1 1 480px;min-width:0;border:1px solid #53677d;background:#091018;color:var(--ink);border-radius:14px;padding:17px 18px;font:inherit;font-size:18px;outline:none}input:focus{border-color:var(--hot);box-shadow:0 0 0 3px rgba(255,189,85,.16)}button{font:inherit;color:var(--ink);background:#203144;border:1px solid var(--line);border-radius:12px;padding:11px 14px;cursor:pointer}button:hover,button[aria-pressed="true"]{border-color:var(--hot);background:#2a3e52}.primary{background:var(--hot);color:#1a1408;border-color:var(--hot);font-weight:900;padding:15px 22px}.primary:hover{background:#ffd276}.modes{margin-top:12px}.modehelp{font-size:13px;line-height:1.5;color:var(--muted)}.examples{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.chip{font-size:13px;border-radius:999px;padding:7px 11px;color:#cad7e6}.status{min-height:28px;color:var(--muted);margin:24px 0 13px}.status.error{color:var(--danger)}.result-grid{display:grid;gap:13px}.card{border:1px solid var(--line);border-radius:18px;background:var(--panel);padding:20px}.top{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}.artist{font-size:18px;font-weight:900;color:var(--mint)}.song{color:#ced8e5}.tag{font-size:12px;color:var(--hot);border:1px solid rgba(255,189,85,.5);border-radius:999px;padding:4px 8px;white-space:nowrap}.snippet{margin:15px 0 0;white-space:pre-wrap;line-height:1.8;color:#e4ebf5}.evidence{margin:14px 0 0;color:var(--muted);font-size:13px}.duplicate{margin-top:8px;color:#bdcce0;font-size:12px;line-height:1.55}.empty{padding:31px 22px;border:1px dashed var(--line);border-radius:18px;color:var(--muted)}details{margin-top:35px;padding:16px 18px;border:1px solid var(--line);border-radius:16px;color:var(--muted);line-height:1.75}summary{cursor:pointer;color:var(--ink);font-weight:850}.foot{margin-top:16px;color:#8292a7;font-size:12px;line-height:1.55}@media(max-width:620px){main{width:min(100% - 22px,1110px);padding-top:32px}h1{font-size:43px}.controls{align-items:stretch}.primary{width:100%}}
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">私有本机规范语料工具 · 只在这台电脑运行</div>
    <h1>输入一个主题，<br>找到真正在唱它的歌。</h1>
    <p class="lede">从“城市里的孤独”到“想回家”，这里不是给歌曲贴主观主题标签，而是把你的表达和规范歌词中的证据段落连起来：你能看见是谁、哪首歌，以及为什么出现在结果里。</p>
    <div class="promise"><b>只用规范可用语料。</b> 未通过身份或歌词完整性门的记录不会进入搜索；跨歌重复段落也不会反复占满结果。</div>
    <section class="searchbox" aria-label="主题找歌">
      <div class="controls"><input id="query" maxlength="120" autocomplete="off" placeholder="例如：城市里的孤独 / 想回家 / growing up" aria-label="输入主题、意象或一句话"><button class="primary" id="go">开始找歌</button></div>
      <div class="controls modes" role="group" aria-label="搜索方式"><button class="mode" data-mode="topic" aria-pressed="true">按主题线索找</button><button class="mode" data-mode="semantic" aria-pressed="false">按意思找</button><button class="mode" data-mode="exact" aria-pressed="false">找原句</button><span class="modehelp" id="modehelp">按主题线索：按中文词片与英文词的相关度排序，不把结果假装成“官方主题”。</span></div>
      <div class="examples" aria-label="示例"><button class="chip" data-q="城市">城市</button><button class="chip" data-q="家乡">家乡</button><button class="chip" data-q="成长">成长</button><button class="chip" data-q="孤独">孤独</button><button class="chip" data-q="我要离开">我要离开</button></div>
    </section>
    <div id="status" class="status">输入一个主题、意象或一句想找的话。</div>
    <section id="results" class="result-grid" aria-live="polite"><div class="empty">页面本身没有预装歌名或歌词。你提交查询后，才会在本机按需显示一小段证据。</div></section>
    <details><summary>这三种找法有什么区别？</summary><p><b>按主题线索找</b>使用透明的词汇检索：中文按相邻词片、英文按词汇，结果按相关度排序；它能找到共享用词的表达，但不能理解同义句，更不能替音乐人宣布“这首歌的官方主题”。<b>按意思找</b>使用本机已验证的 canonical BGE-M3 向量，寻找表达较接近的片段；它也不是官方主题判定，且只有当向量契约、行顺序、文本指纹、去重权重和本地模型都通过检查时才会开启。<b>找原句</b>只找歌词中实际包含你输入字串的段落。</p></details>
    <p class="foot">私有本机研究工具。请不要分享运行页面、浏览器导出或 API 响应；它们可能包含受控歌词片段与作品信息。</p>
  </main>
<script>
(()=>{const $=id=>document.getElementById(id);const query=$('query'),go=$('go'),status=$('status'),results=$('results'),help=$('modehelp');let mode='topic';const esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));const setStatus=(text,kind='')=>{status.textContent=text;status.className='status '+kind};function draw(items,modeName){if(!items.length){results.innerHTML='<div class="empty">没有找到证据段落。试试更短的词，或切换到另一种找法。</div>';return}results.innerHTML=items.map(r=>`<article class="card"><div class="top"><div><span class="artist">来源署名：${esc(r.artist)}</span><span class="song"> · ${esc(r.song_title)}</span></div><span class="tag">${modeName}</span></div><p class="snippet">${esc(r.snippet)}</p><p class="evidence">${esc(r.evidence_label)}</p>${r.duplicate_note?`<p class="duplicate">${esc(r.duplicate_note)}</p>`:''}</article>`).join('')}async function search(){const q=query.value.trim();if(!q){setStatus('先输入一个主题或一句话。','error');return}go.disabled=true;setStatus(mode==='semantic'?'正在加载已验证的本机语义模型…':'正在在本机规范语料中寻找证据…');results.innerHTML='<div class="empty">正在读取本地结果…</div>';try{const response=await fetch('/api/search?mode='+encodeURIComponent(mode)+'&q='+encodeURIComponent(q),{cache:'no-store'});const data=await response.json().catch(()=>({}));if(!response.ok)throw new Error(data.error||'搜索暂时不可用。');const lead=mode==='topic'?'这是词汇相关的主题线索，不是自动主题判定。':mode==='semantic'?'这是表达意思更接近的候选，不是自动主题判定。':'这是实际包含该原句的证据段落。';setStatus(lead+' '+data.corpus_note);draw(data.results,mode==='topic'?'主题线索':mode==='semantic'?'意思接近':'原句命中')}catch(error){setStatus(error.message||'搜索暂时不可用。','error');results.innerHTML='<div class="empty">请检查本机规范语料是否完整，或换一个更短的输入。</div>'}finally{go.disabled=false}}document.querySelectorAll('.mode').forEach(button=>button.addEventListener('click',()=>{mode=button.dataset.mode;document.querySelectorAll('.mode').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));help.textContent=mode==='topic'?'按主题线索：按中文词片与英文词的相关度排序，不把结果假装成“官方主题”。':mode==='semantic'?'按意思找：只有本机向量材料通过完整核验时，才会按表达接近程度排序。':'找原句：只找歌词中包含这几个字/词的段落。'}));document.querySelectorAll('[data-q]').forEach(button=>button.addEventListener('click',()=>{query.value=button.dataset.q;search();}));go.addEventListener('click',search);query.addEventListener('keydown',event=>{if(event.key==='Enter')search()});})();
</script>
</body>
</html>'''


def build() -> None:
    corpus = load_canonical_corpus()
    semantic = CanonicalSemanticIndex(corpus)
    atomic_write_text(INDEX_FILE, build_html())
    atomic_write_text(
        OUTPUT_DIR / "README_CN.md",
        "# 中文说唱 · Canonical 主题找歌 V2（仅本机）\n\n"
        "这是一个只读取 canonical corpus contract 的本机检索工具。它不会回退到旧冻结快照或旧版 BGE 向量；"
        "只检索 `downstream_eligibility=eligible` 的私有规范歌词，并按 `analysis_text_weight` 处理跨歌重复文本。\n\n"
        "## 三种可用方法\n\n"
        "- **按主题线索找**：可审计的 BM25 式词汇检索。中文使用相邻词片，英文/数字使用词；它反映表达上的词汇重合，不宣称自动理解同义句或官方主题。\n"
        "- **按意思找**：只使用完整验证过的 canonical BGE-M3 向量。它在启动时核验 canonical contract/私有歌词 SHA、eligible 行顺序、每行来源、向量 shape/L2、去重权重和本地模型指纹；任何一项不通过，语义模式保持关闭，不会回退到旧向量。\n"
        "- **找原句**：只返回实际包含输入字串的规范歌词段落。\n\n"
        "## 语义层的严格边界\n\n"
        "旧 BGE-M3 矩阵不会被检索运行时直接读取。当前 canonical 向量工件只在 `(song_id, chunk_id)` 与文本 SHA"
        "都完全相同的行复用旧向量；文本变换行会在同一份本地 BGE-M3 上重算，withheld 行排除。\n\n"
        "## 运行\n\n"
        "```powershell\n"
        "& \"work\\semantic-ml-venv\\Scripts\\python.exe\" work\\run_private_rap_topic_search_canonical_v2.py build\n"
        "& \"work\\semantic-ml-venv\\Scripts\\python.exe\" work\\run_private_rap_topic_search_canonical_v2.py validate\n"
        "& \"work\\semantic-ml-venv\\Scripts\\python.exe\" work\\run_private_rap_topic_search_canonical_v2.py smoke\n"
        "& \"work\\semantic-ml-venv\\Scripts\\python.exe\" work\\run_private_rap_topic_search_canonical_v2.py serve --port 8787\n"
        "```\n\n"
        "静态目录无歌词、歌名、歌手、song_id 或索引。不要分享运行页面、API 响应或浏览器导出。\n",
    )
    atomic_write_text(
        OUTPUT_DIR / "PRIVATE_LOCAL_ONLY.md",
        "# Private local only\n\n"
        "静态页面不含语料。服务启动后，它会从本机私有 canonical 歌词读取受控片段，并仅向 127.0.0.1 的已提交查询返回歌手、歌名和短证据片段。"
        "不要上传、打包或分享运行页面、API 响应、浏览器导出或私有歌词输入。\n",
    )
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "privacy": "private_local_only_runtime_corpus_content",
        "canonical_input": {
            "canonical_corpus_id": corpus.corpus_id,
            "contract_version": corpus.contract_version,
            "contract_sha256": corpus.contract_sha256,
            "private_content_sha256": corpus.private_content_sha256,
            "old_frozen_baseline_used": False,
        },
        "counts": {
            "private_canonical_chunk_rows": corpus.total_private_chunks,
            "downstream_eligible_chunk_rows": len(corpus.eligible_rows),
            "withheld_chunk_rows": corpus.withheld_rows,
            "cross_song_duplicate_text_groups": corpus.duplicate_group_count,
        },
        "retrieval": {
            "exact": "literal phrase match in eligible canonical lyric chunks",
            "topic": "transparent BM25-style Chinese character bigram + Latin token lexical retrieval; analysis_text_weight applied",
            "semantic": {
                "status": "enabled_verified_canonical_embedding_artifact_v1" if semantic.verified else semantic.status,
                "artifact_id": semantic.artifact_id,
                "fail_closed": True,
                "runtime_rule": "load only after canonical contract, private content hash, eligible row order, row provenance, vector shape/L2, duplicate weight, and local model fingerprint verify.",
            },
        },
        "runtime": {"host": SERVER_HOST, "default_port": SERVER_PORT, "max_results": MAX_RESULTS, "full_song_lyrics_returned": False},
    }
    atomic_write_json(MANIFEST_FILE, manifest)


def validate() -> dict[str, Any]:
    corpus = load_canonical_corpus()
    if not INDEX_FILE.is_file() or not MANIFEST_FILE.is_file():
        raise CanonicalTopicSearchError("请先构建静态页面和 manifest。")
    page = INDEX_FILE.read_text(encoding="utf-8")
    static_files = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()} - {"validation.json", "smoke_test.json"}
    allowed = {"index.html", "README_CN.md", "PRIVATE_LOCAL_ONLY.md", "manifest.json"}
    # Test against actual corpus snippets without emitting them into logs.
    probe_texts = [row.text[: min(36, len(row.text))] for row in corpus.eligible_rows[:120] if row.text]
    leaked = any(probe and probe in page for probe in probe_texts)
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    semantic = CanonicalSemanticIndex(corpus)
    try:
        load_verified_embedding_artifact(replace(corpus, private_content_sha256="0" * 64))
        semantic_hash_tamper_fails_closed = False
    except CanonicalSemanticArtifactError:
        semantic_hash_tamper_fails_closed = True
    source_code = Path(__file__).read_text(encoding="utf-8")
    # The contract is deliberately read only to reject the legacy role.  The
    # runtime has no baseline CSV/model/embedding input constants or loaders.
    module = ast.parse(source_code)
    assigned_names = {
        target.id
        for node in ast.walk(module)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    legacy_runtime_inputs_absent = not ({"SOURCE_CSV", "EMBEDDING_FILE", "MODEL_DIR"} & assigned_names) and "BGEM3FlagModel" not in imported_names
    checks = [
        {"name": "canonical_contract_loaded", "passed": manifest["canonical_input"]["canonical_corpus_id"] == corpus.corpus_id},
        {"name": "private_content_hash_matches_contract", "passed": manifest["canonical_input"]["private_content_sha256"] == corpus.private_content_sha256},
        {"name": "only_downstream_eligible_rows_loaded", "passed": len(corpus.eligible_rows) + corpus.withheld_rows == corpus.total_private_chunks},
        {"name": "legacy_baseline_not_used", "passed": manifest["canonical_input"]["old_frozen_baseline_used"] is False and legacy_runtime_inputs_absent},
        {"name": "static_page_contains_no_probed_lyric_text", "passed": not leaked},
        {"name": "static_page_has_no_remote_assets", "passed": "https://" not in page and "http://" not in page},
        {"name": "static_allowlist", "passed": static_files == allowed, "actual": sorted(static_files)},
        {"name": "loopback_runtime_only", "passed": SERVER_HOST in Path(__file__).read_text(encoding="utf-8") and "is_loopback" in Path(__file__).read_text(encoding="utf-8")},
        {"name": "verified_canonical_semantic_artifact_required", "passed": semantic.verified and manifest["retrieval"]["semantic"]["status"] == "enabled_verified_canonical_embedding_artifact_v1"},
        {"name": "semantic_contract_hash_tamper_fails_closed", "passed": semantic_hash_tamper_fails_closed},
        {"name": "semantic_static_ui_is_contract_gated", "passed": 'data-mode="semantic"' in page and "本机向量材料通过完整核验" in page},
        {"name": "no_full_song_lyrics_in_runtime_payload_contract", "passed": manifest["runtime"]["full_song_lyrics_returned"] is False},
    ]
    result = {
        "artifact_id": ARTIFACT_ID,
        "passed": all(check["passed"] for check in checks),
        "canonical_corpus_id": corpus.corpus_id,
        "contract_version": corpus.contract_version,
        "eligible_chunk_rows": len(corpus.eligible_rows),
        "withheld_chunk_rows": corpus.withheld_rows,
        "checks": checks,
    }
    atomic_write_json(VALIDATION_FILE, result)
    if not result["passed"]:
        raise CanonicalTopicSearchError("Canonical V2 静态/契约验证失败。")
    return result


def smoke() -> dict[str, Any]:
    """Search without writing corpus values to evidence files or stdout."""

    app = CanonicalTopicApp()
    exact = app.search("exact", "城市")
    topic = app.search("topic", "城市里的孤独")
    semantic = app.search("semantic", "城市里的孤独")
    result = {
        "artifact_id": ARTIFACT_ID,
        "passed": bool(exact["results"]) and bool(topic["results"]) and bool(semantic["results"]),
        "canonical_corpus_id": app.corpus.corpus_id,
        "contract_version": app.corpus.contract_version,
        "tests": {
            "exact_query": {"query_length": 2, "result_count": len(exact["results"]), "raw_candidate_count": exact["raw_candidate_count"], "payload_contains_full_text": any("full_text" in item for item in exact["results"])},
            "topic_query": {"query_length": 6, "result_count": len(topic["results"]), "raw_candidate_count": topic["raw_candidate_count"], "payload_contains_full_text": any("full_text" in item for item in topic["results"])},
            "semantic_query": {
                "query_length": 6,
                "result_count": len(semantic["results"]),
                "raw_candidate_count": semantic["raw_candidate_count"],
                "payload_contains_full_text": any("full_text" in item for item in semantic["results"]),
                "contract_status": semantic["semantic_contract_status"],
            },
        },
    }
    result["passed"] = (
        bool(result["passed"])
        and not result["tests"]["exact_query"]["payload_contains_full_text"]
        and not result["tests"]["topic_query"]["payload_contains_full_text"]
        and not result["tests"]["semantic_query"]["payload_contains_full_text"]
        and result["tests"]["semantic_query"]["contract_status"] == "ready"
    )
    atomic_write_json(SMOKE_FILE, result)
    if not result["passed"]:
        raise CanonicalTopicSearchError("Canonical V2 smoke test 失败。")
    return result


def make_handler(app: CanonicalTopicApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PrivateCanonicalTopicSearch/2.0"

        def log_message(self, _format: str, *args: Any) -> None:
            # Do not write private query text, labels, or lyric snippets to the terminal log.
            return

        def is_loopback(self) -> bool:
            return self.client_address[0] in {"127.0.0.1", "::1"}

        def send_bytes(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            self.send_bytes(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if not self.is_loopback():
                self.send_bytes(HTTPStatus.FORBIDDEN, "text/plain; charset=utf-8", b"Loopback only")
                return
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self.send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", INDEX_FILE.read_bytes())
                return
            if parsed.path == "/api/health":
                self.send_json(HTTPStatus.OK, {
                    "artifact": ARTIFACT_ID,
                    "canonical_corpus_id": app.corpus.corpus_id,
                    "contract_version": app.corpus.contract_version,
                    "eligible_chunk_rows": len(app.corpus.eligible_rows),
                    "semantic_status": app.semantic_index.status,
                    "semantic_artifact_id": app.semantic_index.artifact_id,
                    "semantic_contract_verified": app.semantic_index.verified,
                    "bind": SERVER_HOST,
                })
                return
            if parsed.path == "/api/search":
                params = parse_qs(parsed.query, keep_blank_values=True)
                try:
                    self.send_json(HTTPStatus.OK, app.search(params.get("mode", [""])[0], params.get("q", [""])[0]))
                except CanonicalTopicSearchError as exc:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "未找到此页面。"})

    return Handler


def serve(port: int) -> None:
    if not INDEX_FILE.is_file() or not MANIFEST_FILE.is_file():
        build()
        validate()
    app = CanonicalTopicApp()
    server = ThreadingHTTPServer((SERVER_HOST, port), make_handler(app))
    print(f"PRIVATE_LOCAL_ONLY http://{SERVER_HOST}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "smoke", "serve", "all"))
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port 必须在 1024 至 65535 之间。")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.command in {"build", "all"}:
            build()
        if args.command in {"validate", "all"}:
            validation = validate()
            print(json.dumps({"validated": validation["passed"], "eligible_chunk_rows": validation["eligible_chunk_rows"]}, ensure_ascii=False))
        if args.command in {"smoke", "all"}:
            smoke_result = smoke()
            print(json.dumps({"smoke": smoke_result["passed"], "tests": smoke_result["tests"]}, ensure_ascii=False))
        if args.command == "serve":
            serve(args.port)
        return 0
    except CanonicalTopicSearchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
