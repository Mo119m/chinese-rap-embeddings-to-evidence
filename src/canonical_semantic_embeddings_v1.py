"""Contract-gated private canonical BGE-M3 embedding artifact helpers.

This module intentionally has no dependency on a legacy lyric snapshot.  It
loads a completed *canonical* embedding artifact only after checking the
canonical corpus contract hashes, eligible row order, row mapping, vector
shape, vector hash, and L2 normalization.  The legacy snapshot is consulted
only by the separate incremental builder, never by a search runtime.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


ROOT = Path(__file__).resolve().parent.parent
PRIVATE_DIR = ROOT / "work" / "private-canonical-semantic-embeddings-v1"
PRIVATE_CONTRACT_FILE = PRIVATE_DIR / "canonical_semantic_embedding_contract_v1.json"
VECTOR_FILE = PRIVATE_DIR / "canonical_bge_m3_embeddings_v1.npy"
ROW_MAP_FILE = PRIVATE_DIR / "canonical_embedding_row_map_v1.csv"
PRIVATE_VALIDATION_FILE = PRIVATE_DIR / "validation.json"

ARTIFACT_ID = "private-canonical-semantic-embeddings-v1"
ARTIFACT_VERSION = "1.0.0"
MODEL_DIR = ROOT / "work" / "hf-cache" / "hub" / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_BINARY = MODEL_DIR / "pytorch_model.bin"
MODEL_CONFIG = MODEL_DIR / "config.json"
EXPECTED_DIMENSIONS = 1024
L2_TOLERANCE = 2e-4

ROW_MAP_COLUMNS = (
    "canonical_row_index",
    "song_id",
    "chunk_id",
    "canonical_lyric_text_sha256",
    "analysis_text_weight",
    "embedding_provenance",
    "legacy_vector_index",
    "legacy_source_text_sha256",
    "reembed_reason",
)
REUSED_PROVENANCE = "reused_exact_song_chunk_and_source_text_sha"
REEMBEDDED_PROVENANCE = "reembedded_local_bge_m3"


class CanonicalSemanticArtifactError(RuntimeError):
    """Fail-closed error for malformed or unavailable canonical vectors."""


class CanonicalRowLike(Protocol):
    index: int
    song_id: str
    chunk_id: str
    canonical_text_sha256: str
    analysis_weight: float


class CanonicalCorpusLike(Protocol):
    corpus_id: str
    contract_version: str
    contract_sha256: str
    private_content_sha256: str
    eligible_rows: tuple[CanonicalRowLike, ...]
    withheld_rows: int


@dataclass(frozen=True)
class EmbeddingMapRow:
    canonical_row_index: int
    song_id: str
    chunk_id: str
    canonical_text_sha256: str
    analysis_text_weight: float
    embedding_provenance: str
    legacy_vector_index: int | None
    legacy_source_text_sha256: str
    reembed_reason: str


@dataclass(frozen=True)
class VerifiedCanonicalEmbeddingArtifact:
    artifact_id: str
    artifact_version: str
    contract_sha256: str
    vector_sha256: str
    row_map_sha256: str
    vector_matrix: Any
    row_map: tuple[EmbeddingMapRow, ...]
    reused_rows: int
    reembedded_rows: int
    dimensions: int
    model_binary_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_row_order_sha256(corpus: CanonicalCorpusLike) -> str:
    """Hash the exact eligible canonical order without writing any text."""

    digest = hashlib.sha256()
    for expected_index, row in enumerate(corpus.eligible_rows):
        if row.index != expected_index:
            raise CanonicalSemanticArtifactError("canonical eligible 行顺序不连续；拒绝加载语义向量。")
        encoded = (
            f"{expected_index}\t{row.song_id}\t{row.chunk_id}\t{row.canonical_text_sha256}\t"
            f"{float(row.analysis_weight):.17g}\n"
        ).encode("utf-8")
        digest.update(encoded)
    return digest.hexdigest()


def ensure_relative_filename(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CanonicalSemanticArtifactError(f"语义向量契约缺少 {field} 文件名。")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.parent != Path("."):
        raise CanonicalSemanticArtifactError(f"语义向量契约的 {field} 路径不安全。")
    return PRIVATE_DIR / candidate


def _read_contract() -> tuple[dict[str, Any], str]:
    if not PRIVATE_CONTRACT_FILE.is_file():
        raise CanonicalSemanticArtifactError("找不到已验证的 canonical 语义向量契约；语义搜索保持关闭。")
    try:
        contract = json.loads(PRIVATE_CONTRACT_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CanonicalSemanticArtifactError("canonical 语义向量契约不是合法 JSON；语义搜索保持关闭。") from exc
    required = {
        "artifact_id",
        "artifact_version",
        "complete_contract",
        "canonical_input",
        "canonical_eligible_row_order_sha256",
        "vector_file",
        "row_map_file",
        "incremental_provenance",
        "model",
    }
    if not isinstance(contract, dict) or not required.issubset(contract):
        raise CanonicalSemanticArtifactError("canonical 语义向量契约字段不完整；语义搜索保持关闭。")
    if contract["artifact_id"] != ARTIFACT_ID or contract["artifact_version"] != ARTIFACT_VERSION:
        raise CanonicalSemanticArtifactError("canonical 语义向量契约版本不匹配；语义搜索保持关闭。")
    if contract["complete_contract"] is not True:
        raise CanonicalSemanticArtifactError("canonical 语义向量契约未标为完整；语义搜索保持关闭。")
    return contract, sha256_file(PRIVATE_CONTRACT_FILE)


def _read_row_map(path: Path) -> tuple[EmbeddingMapRow, ...]:
    if not path.is_file():
        raise CanonicalSemanticArtifactError("找不到私有 canonical 向量行映射；语义搜索保持关闭。")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_MAP_COLUMNS:
            raise CanonicalSemanticArtifactError("私有 canonical 向量行映射列结构不匹配；语义搜索保持关闭。")
        rows: list[EmbeddingMapRow] = []
        for raw in reader:
            try:
                index = int(raw["canonical_row_index"])
                weight = float(raw["analysis_text_weight"])
                legacy_index_raw = raw["legacy_vector_index"]
                legacy_index = int(legacy_index_raw) if legacy_index_raw else None
            except (TypeError, ValueError) as exc:
                raise CanonicalSemanticArtifactError("私有 canonical 向量行映射包含无效数字。") from exc
            if not all(raw[field] for field in ("song_id", "chunk_id", "canonical_lyric_text_sha256", "embedding_provenance")):
                raise CanonicalSemanticArtifactError("私有 canonical 向量行映射包含空标识。")
            if not math.isfinite(weight) or not 0 < weight <= 1:
                raise CanonicalSemanticArtifactError("私有 canonical 向量行映射包含无效 duplicate weight。")
            rows.append(
                EmbeddingMapRow(
                    canonical_row_index=index,
                    song_id=raw["song_id"],
                    chunk_id=raw["chunk_id"],
                    canonical_text_sha256=raw["canonical_lyric_text_sha256"],
                    analysis_text_weight=weight,
                    embedding_provenance=raw["embedding_provenance"],
                    legacy_vector_index=legacy_index,
                    legacy_source_text_sha256=raw["legacy_source_text_sha256"],
                    reembed_reason=raw["reembed_reason"],
                )
            )
    return tuple(rows)


def _validate_map_against_corpus(
    mapping: tuple[EmbeddingMapRow, ...],
    corpus: CanonicalCorpusLike,
    contract: dict[str, Any],
) -> tuple[int, int]:
    if len(mapping) != len(corpus.eligible_rows):
        raise CanonicalSemanticArtifactError("canonical 向量行数与 eligible 语料不一致；语义搜索保持关闭。")
    reused = 0
    reembedded = 0
    for expected_index, (mapped, canonical) in enumerate(zip(mapping, corpus.eligible_rows, strict=True)):
        if mapped.canonical_row_index != expected_index or canonical.index != expected_index:
            raise CanonicalSemanticArtifactError("canonical 向量行顺序不匹配；语义搜索保持关闭。")
        if (
            mapped.song_id != canonical.song_id
            or mapped.chunk_id != canonical.chunk_id
            or mapped.canonical_text_sha256 != canonical.canonical_text_sha256
            or not math.isclose(mapped.analysis_text_weight, float(canonical.analysis_weight), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise CanonicalSemanticArtifactError("canonical 向量行映射与私有规范语料不一致；语义搜索保持关闭。")
        if mapped.embedding_provenance == REUSED_PROVENANCE:
            if mapped.legacy_vector_index is None or mapped.legacy_source_text_sha256 != mapped.canonical_text_sha256 or mapped.reembed_reason:
                raise CanonicalSemanticArtifactError("复用向量缺少严格的原始文本指纹证明；语义搜索保持关闭。")
            reused += 1
        elif mapped.embedding_provenance == REEMBEDDED_PROVENANCE:
            if mapped.legacy_vector_index is None or not mapped.legacy_source_text_sha256 or mapped.legacy_source_text_sha256 == mapped.canonical_text_sha256:
                raise CanonicalSemanticArtifactError("重嵌入行没有旧文本不匹配证明；语义搜索保持关闭。")
            if mapped.reembed_reason != "legacy_source_text_sha_mismatch":
                raise CanonicalSemanticArtifactError("重嵌入行原因不明确；语义搜索保持关闭。")
            reembedded += 1
        else:
            raise CanonicalSemanticArtifactError("canonical 向量行具有未知来源标记；语义搜索保持关闭。")
    declared = contract["incremental_provenance"]
    if not isinstance(declared, dict) or declared.get("reused_exact_rows") != reused or declared.get("reembedded_local_rows") != reembedded:
        raise CanonicalSemanticArtifactError("canonical 向量来源计数与行映射不一致；语义搜索保持关闭。")
    if reused + reembedded != len(mapping):
        raise CanonicalSemanticArtifactError("canonical 向量来源计数不完整；语义搜索保持关闭。")
    return reused, reembedded


def load_verified_embedding_artifact(corpus: CanonicalCorpusLike) -> VerifiedCanonicalEmbeddingArtifact:
    """Return an in-memory matrix only after every canonical check succeeds."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - supplied local runtime has numpy
        raise CanonicalSemanticArtifactError("本机缺少 numpy；语义搜索保持关闭。") from exc

    contract, contract_sha = _read_contract()
    canonical = contract["canonical_input"]
    if not isinstance(canonical, dict) or (
        canonical.get("canonical_corpus_id") != corpus.corpus_id
        or canonical.get("canonical_contract_version") != corpus.contract_version
        or canonical.get("canonical_contract_sha256") != corpus.contract_sha256
        or canonical.get("private_content_sha256") != corpus.private_content_sha256
        or canonical.get("eligible_chunk_rows") != len(corpus.eligible_rows)
        or canonical.get("withheld_chunk_rows") != corpus.withheld_rows
    ):
        raise CanonicalSemanticArtifactError("canonical 语义向量与当前 corpus contract 不匹配；语义搜索保持关闭。")
    if contract.get("canonical_eligible_row_order_sha256") != canonical_row_order_sha256(corpus):
        raise CanonicalSemanticArtifactError("canonical 语义向量的 eligible 行顺序不匹配；语义搜索保持关闭。")

    vector_spec = contract["vector_file"]
    row_map_spec = contract["row_map_file"]
    if not isinstance(vector_spec, dict) or not isinstance(row_map_spec, dict):
        raise CanonicalSemanticArtifactError("canonical 语义向量文件定义不正确；语义搜索保持关闭。")
    vector_path = ensure_relative_filename(vector_spec.get("filename"), field="vector")
    row_map_path = ensure_relative_filename(row_map_spec.get("filename"), field="row map")
    if not vector_path.is_file() or not row_map_path.is_file():
        raise CanonicalSemanticArtifactError("私有 canonical 语义向量文件缺失；语义搜索保持关闭。")
    vector_sha = sha256_file(vector_path)
    row_map_sha = sha256_file(row_map_path)
    if vector_sha != vector_spec.get("sha256") or row_map_sha != row_map_spec.get("sha256"):
        raise CanonicalSemanticArtifactError("私有 canonical 语义向量文件哈希不匹配；语义搜索保持关闭。")

    try:
        matrix = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise CanonicalSemanticArtifactError("无法读取私有 canonical 语义向量；语义搜索保持关闭。") from exc
    expected_shape = (len(corpus.eligible_rows), EXPECTED_DIMENSIONS)
    if tuple(matrix.shape) != expected_shape or str(matrix.dtype) != "float32":
        raise CanonicalSemanticArtifactError("canonical 语义向量 shape 或 dtype 不匹配；语义搜索保持关闭。")
    if not bool(np.isfinite(matrix).all()):
        raise CanonicalSemanticArtifactError("canonical 语义向量包含非有限值；语义搜索保持关闭。")
    norms = np.linalg.norm(matrix, axis=1)
    if not bool(np.all(np.abs(norms - 1.0) <= L2_TOLERANCE)):
        raise CanonicalSemanticArtifactError("canonical 语义向量未保持 L2 标准化；语义搜索保持关闭。")

    mapping = _read_row_map(row_map_path)
    reused, reembedded = _validate_map_against_corpus(mapping, corpus, contract)
    if vector_spec.get("rows") != expected_shape[0] or vector_spec.get("dimensions") != expected_shape[1] or vector_spec.get("dtype") != "float32":
        raise CanonicalSemanticArtifactError("canonical 语义向量契约 shape 声明不匹配；语义搜索保持关闭。")

    model = contract["model"]
    if not isinstance(model, dict) or model.get("offline_only") is not True:
        raise CanonicalSemanticArtifactError("canonical BGE 模型未标为离线本地；语义搜索保持关闭。")
    if model.get("local_model_relative_path") != str(MODEL_DIR.relative_to(ROOT)).replace("\\", "/"):
        raise CanonicalSemanticArtifactError("canonical BGE 模型路径不匹配；语义搜索保持关闭。")
    if not MODEL_BINARY.is_file() or not MODEL_CONFIG.is_file():
        raise CanonicalSemanticArtifactError("本地 BGE-M3 模型不完整；语义搜索保持关闭。")
    # The query encoder must be the same local model that encoded the changed
    # rows.  Hash it at load time rather than silently accepting a replacement.
    if model.get("pytorch_model_bin_sha256") != sha256_file(MODEL_BINARY) or model.get("config_sha256") != sha256_file(MODEL_CONFIG):
        raise CanonicalSemanticArtifactError("本地 BGE-M3 模型指纹与 canonical 向量不匹配；语义搜索保持关闭。")

    return VerifiedCanonicalEmbeddingArtifact(
        artifact_id=contract["artifact_id"],
        artifact_version=contract["artifact_version"],
        contract_sha256=contract_sha,
        vector_sha256=vector_sha,
        row_map_sha256=row_map_sha,
        vector_matrix=matrix,
        row_map=mapping,
        reused_rows=reused,
        reembedded_rows=reembedded,
        dimensions=EXPECTED_DIMENSIONS,
        model_binary_sha256=str(model["pytorch_model_bin_sha256"]),
    )


def prepare_offline_environment() -> None:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def load_local_bge_m3_model() -> Any:
    """Load the already-present local model; never permit a network fetch."""

    prepare_offline_environment()
    if not MODEL_DIR.is_dir() or not MODEL_BINARY.is_file():
        raise CanonicalSemanticArtifactError("本地 BGE-M3 模型不完整；无法计算 canonical 语义向量。")
    try:
        from FlagEmbedding import BGEM3FlagModel
        import torch
    except Exception as exc:  # pragma: no cover - environment setup failure
        raise CanonicalSemanticArtifactError("本机 BGE-M3 依赖不可用；无法计算 canonical 语义向量。") from exc
    try:
        return BGEM3FlagModel(str(MODEL_DIR), use_fp16=bool(torch.cuda.is_available()))
    except Exception as exc:
        raise CanonicalSemanticArtifactError("本地 BGE-M3 模型无法加载；拒绝回退到远程或旧向量。") from exc


def encode_dense_with_local_model(model: Any, texts: Iterable[str], *, batch_size: int = 2) -> Any:
    """Dense BGE-M3 encoding with the exact local-only configuration."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise CanonicalSemanticArtifactError("本机缺少 numpy；无法计算 canonical 语义向量。") from exc
    sequence = list(texts)
    if not sequence:
        return np.empty((0, EXPECTED_DIMENSIONS), dtype=np.float32)
    try:
        encoded = model.encode(
            sequence,
            batch_size=batch_size,
            max_length=2048,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        matrix = np.asarray(encoded["dense_vecs"], dtype=np.float32)
    except Exception as exc:
        raise CanonicalSemanticArtifactError("本地 BGE-M3 编码失败；拒绝使用不完整向量。") from exc
    if tuple(matrix.shape) != (len(sequence), EXPECTED_DIMENSIONS) or not bool(np.isfinite(matrix).all()):
        raise CanonicalSemanticArtifactError("本地 BGE-M3 返回了无效向量 shape。")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not bool(np.isfinite(norms).all()) or bool(np.any(norms <= 0)):
        raise CanonicalSemanticArtifactError("本地 BGE-M3 返回了零向量或非有限向量。")
    return (matrix / norms).astype(np.float32, copy=False)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_npy(path: Path, matrix: Any) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise CanonicalSemanticArtifactError("本机缺少 numpy；无法写入 canonical 语义向量。") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npy", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, matrix, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
