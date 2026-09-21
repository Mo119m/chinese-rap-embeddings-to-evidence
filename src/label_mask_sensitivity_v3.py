#!/usr/bin/env python3
"""Does the headline ordering survive the cues an author leaves on purpose? Label masking,
collaboration titles, near-identical rival chunks, and the leakage threshold.

QUESTION. On corpus v3 the three raw spaces order words (0.4963) > characters (0.4266) >
semantic (0.2997) MRR. Four things could carry that ordering without being style. Artists
name themselves and each other: the training-data audit found 6.6% of chunks naming their
own label, and a name is a word and a character string before it is anything else. Songs
whose title marks a collaboration (592 of the 7,220 queries by the audit's rule) put another
artist's verses under the label. A chunk whose nearest other-label chunk in BGE-M3 space has
cosine above 0.99 is the same text filed twice (the audit's p90 is 1.0), which the leakage
groups catch only when the whole songs are near-duplicates. And the leakage rule's own
near-duplicate half joins songs at trigram Jaccard 0.80; a looser or stricter threshold
moves songs between "held out with the query" and "in the profile". Each is an arm here,
and every arm reruns the three raw spaces under the unchanged protocol: one document and
one centroid per song, label profile = leave-group-out weighted sum, cosine rank of the
true label among the eligible labels.

ARMS
  baseline                       the protocol as published; must reproduce 0.2997 / 0.4266 /
                                 0.4963 (three_spaces.json) before any other arm is scored
  mask_all_labels                every label string of the corpus (every distinct
                                 source_credit_label, not only the query's own and not only
                                 the eligible ones) is removed from every chunk with
                                 train_identity_encoder_v3.mask_label (Latin labels as whole
                                 words, case-insensitive; other labels as substrings; labels
                                 under two characters skipped), longest label first, before the
                                 word and character spaces are fitted; occurrences are counted
                                 per label and split into own-label and other-label mentions;
                                 the public file carries totals and a distribution only, the
                                 per-label table goes to the private contract. The protocol's
                                 length rule is re-applied to the masked documents; the
                                 baseline leakage groups stand (the leakage unit is a property
                                 of the corpus text, not of the arm). The semantic column needs
                                 the masked chunks re-embedded: the CPU stage writes them to a
                                 private file and --stage embed (GPU) embeds them with
                                 FlagEmbedding under the recorded chunk run's configuration,
                                 after a cosine reproduction check on 256 UNMASKED chunks
                                 against the recorded vectors; until that run exists and
                                 passes, the column is "pending"
  exclude_collaboration_titles   songs whose title matches the audit's rule
                                 (training_data_audit_v3.FEAT: feat., ft., featuring, &, /, x,
                                 ×, or a list separator, case-insensitive) are removed as
                                 queries and as profile members. This rule NEEDS PRIVATE
                                 METADATA: the song_title column of the private corpus table,
                                 which no public artifact carries; no other metadata is read.
                                 Leakage groups are rebuilt on the remaining songs, the spaces
                                 are refitted on them and labels are re-checked for eligibility
  drop_near_identical_rival_queries
                                 a query is dropped (profiles unchanged) when any of its
                                 chunks has cosine above 0.99 with a chunk of another label in
                                 the recorded BGE-M3 chunk space. rival_anywhere is the arm as
                                 specified (the audit's definition: any chunk of another label,
                                 the query's own leakage group included). rival_outside_
                                 leakage_group is a companion, not the arm: the rival chunk
                                 must lie outside the query's own leakage group, so it counts
                                 only the near-identical text the groups do not already hold out
  leakage_threshold_0.70, _0.90  groups rebuilt with leakage_groups_v2.build_groups at trigram
                                 Jaccard 0.70 and 0.90 instead of 0.80; weights, scoring and
                                 the bootstrap follow the new groups; labels re-checked for
                                 at least two groups

SCORERS AND ESTIMANDS. Every space is scored by build_chinese_rap_downstream_retrieval_v1
.score_leave_group_out (the protocol's own scorer, dense and sparse branches) and ranked by
its rank_system; a sparse row with no feature takes the worst rank, as lexical_identity_
anatomy_v2.score_arm gives it. Per arm and space: MRR as the plain mean over the arm's
queries (the published estimand) and as the per-(group, label)-weighted mean (the
bootstrap's estimand), recall at 1 and 10. Every contrast words - characters,
characters - semantic and words - semantic carries a paired group bootstrap
(identity_spaces_v2.paired_group_bootstrap: 2000 replicates, seed 20260825, the arm's
leakage groups resampled with replacement, each (group, label) component weighted one).

CHECKS BEFORE ANY RESULT IS USED. (1) The baseline must reproduce the published MRR in every
space with gap <= 0.002, and the literals pinned here must equal three_spaces.json, which must
carry the current corpus digest. (2) The population rule must rebuild the protocol's query set
and label list exactly. (3) The dense scores from the two scorer calls of one arm must be
identical (gap <= 1e-6). (4) The collaboration count and the audit-style self-mention count
are compared with training_data_audit.json and the comparisons recorded. (5) The masked
semantic column is used only if the private embedding contract names the current masked-text
digest and corpus digest and its unmasked-chunk check reached median cosine 0.999. (6) The
synthetic test (test_label_mask_sensitivity.py) checks the population rule, the scorer wrapper
against a brute-force leave-group-out definition, the ranks and MRRs against a brute-force
rank, the masking counts, the nearest-rival search and the masked pipeline end to end.

READING RULE, fixed before the run. Under an arm, the ordering words > characters >
semantic is "robust to the arm" if every pairwise contrast (words - characters,
characters - semantic, words - semantic) keeps its sign (positive, as at baseline) with its
paired group-bootstrap interval clear of zero. If any contrast loses its sign or its
interval covers zero, the ordering is "not robust to the arm" and the failing contrast is
named. Where the semantic column is pending, the arm reads "partial" on words - characters
alone and decides nothing about the semantic space. The size of a shift in MRR under an arm
is reported as description; the rule reads only the ordering.

    CHINESE_RAP_CORPUS=v3 python src/label_mask_sensitivity_v3.py --private-root <ni-k>
    CHINESE_RAP_CORPUS=v3 python src/label_mask_sensitivity_v3.py --private-root <ni-k> --stage embed   (GPU)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import SEED, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap, weighted_mean  # noqa: E402
from leakage_groups_v2 import NEAR_DUPLICATE_JACCARD, build_groups, group_audit, normalise_document  # noqa: E402
from train_identity_encoder_v3 import mask_label  # noqa: E402
from training_data_audit_v3 import FEAT, LATIN  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "label_mask_sensitivity.json"
PRIVATE_DIR_NAME = "private-label-mask-v3"
SPACES = ("semantic", "characters", "words")
PAIRS = (("words", "characters"), ("characters", "semantic"), ("words", "semantic"))
# published v3 numbers: (file, system key, pinned literal); the file must agree with the literal
PUBLISHED = {"semantic": ("three_spaces.json", "semantic", 0.2997),
             "characters": ("three_spaces.json", "lexical_char_2_5", 0.4266),
             "words": ("three_spaces.json", "lexical_words", 0.4963)}
AUDIT_FILE = "training_data_audit.json"   # collaboration count and self-mention count, same rules
CHECK_GAP = 0.002
DENSE_GAP = 1e-6
RIVAL_COSINE = 0.99
THRESHOLDS = (0.70, 0.90)
MIN_GROUPS_PER_LABEL = 2
CHECK_CHUNKS = 256
MIN_CHECK_COSINE = 0.999
MAX_LENGTH, BATCH, BLOCK = 2048, 8, 512   # the recorded chunk run's configuration (embed_corpus_local_v3)
RIVAL_BLOCK = 500


def private_dir_of(private_root: Path) -> Path:
    return private_root / "work" / PRIVATE_DIR_NAME


def text_digest(texts) -> str:
    return hashlib.sha256("\x1f".join(texts).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ corpus
def setup(private_root: Path) -> dict:
    """The protocol's query population, as every v3 experiment builds it."""
    rows, vectors, state = load_v3(private_root)
    if state["vectors"] != "v3":
        raise SystemExit("this analysis needs the recorded v3 chunk embedding run")
    chunks_by_song, label_by_song, components_by_song, documents, centroids_by_song = build_songs(rows, vectors)
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {s for s in chunks_by_song
                   if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(l for l, m in songs_by_label.items()
                      if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
    normalised = {s: normalise_document(documents[s]) for s in songs}
    groups = build_groups(songs, components_by_song, normalised)
    return dict(rows=rows, vectors=vectors, state=state, chunks_by_song=chunks_by_song,
                label_by_song=label_by_song, components_by_song=components_by_song, documents=documents,
                centroids_by_song=centroids_by_song, eligible=eligible, songs=songs, normalised=normalised,
                groups=groups)


def population(songs, label_by_song, group_of, keep=None) -> dict:
    """The protocol's population rule on a song set: a label needs MINIMUM_SONGS_PER_LABEL songs
    and at least two leakage groups (leave-group-out must leave something); group ids and the
    per-(group, label) weights follow. `keep` maps song -> bool; None keeps every song."""
    kept = [s for s in songs if keep is None or keep[s]]
    by_label: dict[str, list[str]] = defaultdict(list)
    for s in kept:
        by_label[label_by_song[s]].append(s)
    labels = sorted(l for l, m in by_label.items()
                    if len(m) >= MINIMUM_SONGS_PER_LABEL and len({group_of[s] for s in m}) >= MIN_GROUPS_PER_LABEL)
    position = {l: i for i, l in enumerate(labels)}
    members = [s for s in kept if label_by_song[s] in position]
    label_index = np.asarray([position[label_by_song[s]] for s in members], dtype=np.int64)
    order = {g: i for i, g in enumerate(sorted({group_of[s] for s in members}))}
    group_ids = np.asarray([order[group_of[s]] for s in members], dtype=np.int64)
    size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    return dict(songs=members, labels=labels, label_index=label_index, group_ids=group_ids, weights=weights,
                label_count=len(labels), groups=len(order),
                largest_group=(max(Counter(group_ids.tolist()).values()) if members else 0),
                songs_offered=len(kept), songs_dropped_by_label_rule=len(kept) - len(members),
                labels_dropped_by_label_rule=len(by_label) - len(labels))


# ------------------------------------------------------------------ scoring
def score_spaces(pop: dict, dense: np.ndarray, char_matrix, word_matrix, with_semantic: bool = True):
    """The protocol's scorer on a population. `dense` (n, 1024) unit rows; the sparse matrices are
    CSR unit rows over the same songs. The dense branch is computed twice and must agree.
    Returns the score matrices per space and, per sparse space, the rows with no feature."""
    li, gi, L = pop["label_index"], pop["group_ids"], pop["label_count"]
    dense32 = np.ascontiguousarray(dense, dtype=np.float32)
    a = v1.score_leave_group_out(dense32, char_matrix, li, gi, L)
    b = v1.score_leave_group_out(dense32, word_matrix, li, gi, L)
    gap = float(np.abs(a.dense - b.dense).max())
    if gap > DENSE_GAP:
        raise RuntimeError(f"the dense scores differ between the two scorer calls ({gap:.2e})")
    scores = {"characters": a.lexical.astype(np.float64), "words": b.lexical.astype(np.float64)}
    empty = {"characters": np.asarray(char_matrix.getnnz(axis=1) == 0),
             "words": np.asarray(word_matrix.getnnz(axis=1) == 0)}
    if with_semantic:
        scores["semantic"] = a.dense.astype(np.float64)
    return scores, empty


def evaluate(pop: dict, scores: dict, empty: dict | None = None, query_mask=None):
    """Per-space MRR (plain and component-weighted), recall, and the paired contrasts over the
    arm's queries (all of them, or those in `query_mask`). Rows named in `empty` get the worst rank."""
    li, w, gi, L = pop["label_index"], pop["weights"], pop["group_ids"], pop["label_count"]
    mask = np.ones(len(li), dtype=bool) if query_mask is None else np.asarray(query_mask, dtype=bool)
    if mask.sum() == 0:
        raise RuntimeError("an arm has no queries")
    rr, systems = {}, {}
    for name in SPACES:
        if name not in scores:
            continue
        ranks = ranks_of(scores[name], li)
        empty_rows = np.zeros(len(li), dtype=bool) if not empty or name not in empty else empty[name]
        ranks = np.where(empty_rows, L, ranks)
        rr[name] = 1.0 / ranks
        systems[name] = {"mrr": round(float(np.mean(rr[name][mask])), 4),
                         "mrr_component_weighted": round(weighted_mean(rr[name], w, mask), 4),
                         "recall_at_1": round(float(np.mean(ranks[mask] <= 1)), 4),
                         "recall_at_10": round(float(np.mean(ranks[mask] <= 10)), 4),
                         "queries": int(mask.sum()), "rows_without_features": int(empty_rows[mask].sum())}
    pairs = [(a, b) for a, b in PAIRS if a in scores and b in scores]
    contrasts = paired_group_bootstrap(rr, w, gi, mask, pairs)
    return systems, contrasts


def reading_of(contrasts, spaces_present) -> str:
    expected = [(a, b) for a, b in PAIRS if a in spaces_present and b in spaces_present]
    failing = []
    for a, b in expected:
        c = next(c for c in contrasts if (c["system"], c["minus"]) == (a, b))
        if not (c["mrr_difference"] > 0 and c["ci95"][0] > 0):
            failing.append(f"{a} - {b}")
    if len(expected) < len(PAIRS):
        return ("partial: words > characters holds; semantic pending" if not failing
                else f"partial: not robust on {failing}; semantic pending")
    return ("robust to the arm" if not failing
            else f"not robust to the arm: {failing} (sign lost or interval covers zero)")


# ------------------------------------------------------------------ masking
def names_label(text: str, label: str) -> bool:
    """The training-data audit's self-mention test, on the original text."""
    label = label.strip()
    if len(label) < 2:
        return False
    if LATIN.match(label):
        return re.search(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])", text, re.I) is not None
    return label in text


def mask_all_labels(texts, labels, own=None):
    """mask_label for every label string applied to every text, longest label first (ties by the
    string), each label's occurrences counted before it is removed. `own` gives each text's own
    label position, so mentions split into own-label and other-label. Returns the masked texts,
    a count per label position, and aggregate counts."""
    ordered = sorted(((l.strip(), i) for i, l in enumerate(labels) if len(l.strip()) >= 2),
                     key=lambda t: (-len(t[0]), t[0]))
    latin = {i: re.compile(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])", re.I)
             for label, i in ordered if LATIN.match(label)}
    counts = np.zeros(len(labels), dtype=np.int64)
    tally = {"own_label_occurrences": 0, "other_label_occurrences": 0, "chunks_naming_own_label": 0,
             "chunks_naming_another_label": 0, "chunks_touched": 0}
    out = []
    for k, text in enumerate(texts):
        current, lower = text, text.lower()
        hit_own = hit_other = False
        for label, i in ordered:
            if i in latin:
                if label.lower() not in lower:
                    continue
                n = len(latin[i].findall(current))
            else:
                if label not in current:
                    continue
                n = current.count(label)
            if n:
                current = mask_label(current, label)
                lower = current.lower()
                counts[i] += n
                if own is not None and i == own[k]:
                    tally["own_label_occurrences"] += n
                    hit_own = True
                else:
                    tally["other_label_occurrences"] += n
                    hit_other = True
        out.append(current)
        tally["chunks_naming_own_label"] += int(hit_own)
        tally["chunks_naming_another_label"] += int(hit_other)
        tally["chunks_touched"] += int(hit_own or hit_other)
    return out, counts, tally


def masking_summary(counts: np.ndarray, tally: dict, chunks: int, label_lengths) -> dict:
    """Aggregate only: no label is named."""
    present = counts[counts > 0]
    lengths = np.asarray(label_lengths)
    by_length = {}
    for lo, hi, name in ((2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10 ** 6, "6+")):
        sel = (lengths >= lo) & (lengths <= hi)
        by_length[name] = {"labels": int(sel.sum()), "occurrences": int(counts[sel].sum())}
    return {"labels_masked": int((lengths >= 2).sum()), "labels_under_two_characters_skipped": int((lengths < 2).sum()),
            "labels_with_any_occurrence": int((counts > 0).sum()),
            "occurrences_total": int(counts.sum()), "chunks": int(chunks),
            **{k: int(v) for k, v in tally.items()},
            "share_of_chunks_touched": round(tally.get("chunks_touched", 0) / chunks, 4) if chunks else 0.0,
            "occurrences_per_label_quantiles": {str(p): int(np.percentile(counts, p)) for p in (50, 75, 90, 99, 100)},
            "occurrences_per_label_with_any_quantiles": ({str(p): int(np.percentile(present, p)) for p in (10, 50, 90)}
                                                         if len(present) else {}),
            "occurrences_by_label_length_in_characters": by_length}


# ------------------------------------------------------------------ rivals
def nearest_rival(chunk_vec: np.ndarray, chunk_label: np.ndarray, chunk_group: np.ndarray, block: int = RIVAL_BLOCK):
    """For every chunk: the highest cosine with a chunk of another label (the audit's definition),
    and with a chunk of another label outside the chunk's own leakage group."""
    n = len(chunk_label)
    anywhere, outside = np.full(n, -np.inf), np.full(n, -np.inf)
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = chunk_vec[start:stop] @ chunk_vec.T
        other = chunk_label[start:stop][:, None] != chunk_label[None, :]
        anywhere[start:stop] = np.where(other, sims, -np.inf).max(axis=1)
        far = other & (chunk_group[start:stop][:, None] != chunk_group[None, :])
        outside[start:stop] = np.where(far, sims, -np.inf).max(axis=1)
    return anywhere, outside


def song_maximum(values: np.ndarray, chunk_song: np.ndarray, song_count: int) -> np.ndarray:
    out = np.full(song_count, -np.inf)
    np.maximum.at(out, chunk_song, values)
    return out


# ------------------------------------------------------------------ spaces
def fit_spaces(documents: list[str]):
    char_matrix = v1.fit_tfidf(documents)
    word_matrix, _ = fit_words([" ".join(segment(d)) for d in documents])
    return char_matrix, word_matrix


def published_numbers(out_dir: Path) -> dict:
    expected = {}
    for space, (file_name, key, literal) in PUBLISHED.items():
        payload = json.loads((out_dir / file_name).read_text(encoding="utf-8"))
        if payload.get("corpus", {}).get("content_sha256") != V3_CONTENT_SHA256:
            raise SystemExit(f"{file_name} was computed on another corpus build")
        value = payload["systems"][key]["mrr"]
        if abs(value - literal) > 1e-9:
            raise SystemExit(f"{file_name} gives {value} for {key}, not the pinned {literal}; update PUBLISHED")
        expected[space] = literal
    return expected


def audit_numbers(out_dir: Path) -> dict:
    """The audit's collaboration and self-mention counts, for the recorded comparisons."""
    path = out_dir / AUDIT_FILE
    if not path.is_file():
        return {"collaborations": None, "self_mention_chunks": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"collaborations": payload.get("collaborations", {}).get("songs_with_a_collaboration_mark_in_the_title"),
            "self_mention_chunks": payload.get("self_mentions", {}).get("chunks")}


def check_published(systems: dict, expected: dict) -> dict:
    checks = {}
    for space, value in expected.items():
        got = systems[space]["mrr"]
        checks[space] = {"expected": value, "recomputed": got, "gap": round(abs(got - value), 4)}
        print(f"check {space}: recomputed {got:.4f} expected {value}", flush=True)
    if any(c["gap"] > CHECK_GAP for c in checks.values()):
        raise SystemExit(f"the baseline does not reproduce the published numbers: {checks}")
    return checks


# ------------------------------------------------------------------ private masked table
def write_masked_table(store: Path, rows, chunk_rows, masked_texts, check_rows, labels, summary,
                       counts_by_label=None) -> dict:
    """The masked chunk table and its contract, PRIVATE (lyric text and identifiers). The
    contract also holds the per-label occurrence table, which the public file never carries."""
    store.mkdir(parents=True, exist_ok=True)
    with (store / "masked_chunks_v3.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_index", "corpus_row", "song_id", "chunk_id", "source_order", "masked_text"])
        for k, i in enumerate(chunk_rows):
            writer.writerow([k, i, rows[i]["song_id"], rows[i]["chunk_id"], rows[i]["source_order"], masked_texts[k]])
    contract = {"corpus_content_sha256": V3_CONTENT_SHA256, "chunks": len(chunk_rows),
                "masked_text_sha256": text_digest(masked_texts),
                "labels_sha256": hashlib.sha256("\x1f".join(labels).encode("utf-8")).hexdigest(),
                "check_corpus_rows": [int(i) for i in check_rows], "masking": summary,
                "masked_occurrences_by_label_PRIVATE": ({str(l): int(c) for l, c in zip(labels, counts_by_label)}
                                                        if counts_by_label is not None else None),
                "embedding_configuration": {"model": "BAAI/bge-m3", "use_fp16": True, "max_length": MAX_LENGTH,
                                            "batch_size": BATCH, "block_size": BLOCK, "order": "table row order"},
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "warning": "private; derived from copyrighted lyric text. Never commit."}
    (store / "masked_table_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    return contract


def masked_vectors_if_ready(store: Path, masked_digest: str, chunk_count: int):
    """The masked chunk vectors, or (None, why) when the GPU stage has not run or is stale."""
    contract_path, vectors_path = store / "masked_embedding_contract.json", store / "masked_chunk_bge_m3.npy"
    if not contract_path.is_file() or not vectors_path.is_file():
        return None, "pending: the masked chunks have not been embedded (run --stage embed on the GPU)"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("corpus_content_sha256") != V3_CONTENT_SHA256:
        return None, "stale: the masked embedding was computed on another corpus build"
    if contract.get("masked_text_sha256") != masked_digest:
        return None, "stale: the masked embedding was computed on another masked text; rerun --stage embed"
    check = contract.get("check_unmasked_chunks", {})
    if check.get("median_cosine", 0.0) < MIN_CHECK_COSINE:
        return None, "failed: the embedding run did not reproduce the recorded vectors on unmasked chunks"
    vectors = np.load(vectors_path)
    if vectors.shape != (chunk_count, 1024):
        return None, "stale: the masked embedding has the wrong shape"
    return vectors, contract


# ------------------------------------------------------------------ GPU stage
def stage_embed(private_root: Path) -> int:
    """Embed the masked chunks exactly as the recorded chunk run and whole_song_embedding_v3 do
    (FlagEmbedding BGEM3FlagModel, fp16, CUDA, pinned weights), after reproducing the recorded
    vectors on 256 unmasked chunks. Blocks are saved as they finish so a killed run resumes."""
    import torch
    from huggingface_hub import snapshot_download
    from FlagEmbedding import BGEM3FlagModel
    from embed_corpus_local_v3 import MODEL_ID, MODEL_REVISION, PINNED_WEIGHTS_SHA256, sha256_file
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this run must match the recorded run's device class")
    store = private_dir_of(private_root)
    contract_path = store / "masked_table_contract.json"
    if not contract_path.is_file():
        raise SystemExit("no masked table yet; run the CPU stage first")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["corpus_content_sha256"] != V3_CONTENT_SHA256:
        raise SystemExit("the masked table was written for another corpus build")
    with (store / "masked_chunks_v3.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    texts = [r["masked_text"] for r in table]
    digest = text_digest(texts)
    if digest != contract["masked_text_sha256"] or len(texts) != contract["chunks"]:
        raise SystemExit("the masked table does not match its contract; rerun the CPU stage")
    rows, vectors, state = load_v3(private_root)
    if state["vectors"] != "v3":
        raise SystemExit("the check needs the recorded v3 chunk embedding run")
    for k, r in enumerate(table):
        i = int(r["corpus_row"])
        if int(r["row_index"]) != k or rows[i]["song_id"] != r["song_id"] or int(rows[i]["chunk_id"]) != int(r["chunk_id"]):
            raise SystemExit(f"the masked table diverges from the corpus at row {k}")
    check_rows = [int(i) for i in contract["check_corpus_rows"]]
    print(f"  {len(texts):,} masked chunks; {len(check_rows)} unmasked chunks for the check", flush=True)

    model_path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
    weights = next((model_path / n for n in ("pytorch_model.bin", "model.safetensors") if (model_path / n).is_file()), None)
    weights_sha = sha256_file(weights) if weights else None
    if weights_sha != PINNED_WEIGHTS_SHA256:
        raise SystemExit(f"weights sha256 {weights_sha} is not the pinned {PINNED_WEIGHTS_SHA256}")
    model = BGEM3FlagModel(str(model_path), use_fp16=True, devices="cuda")

    # the check first: unmasked text under this configuration must give the recorded vectors
    mine = unit_rows(np.asarray(model.encode([rows[i]["cleaned_text"] for i in check_rows], batch_size=BATCH,
                                             max_length=MAX_LENGTH)["dense_vecs"], dtype=np.float32))
    recorded = unit_rows(vectors[check_rows].astype(np.float32))
    agreement = np.sum(mine * recorded, axis=1)
    check = {"compared": len(check_rows), "min_cosine": round(float(agreement.min()), 6),
             "median_cosine": round(float(np.median(agreement)), 6),
             "below_0.999": int((agreement < MIN_CHECK_COSINE).sum())}
    print(f"  unmasked chunks against the recorded run: {check}", flush=True)
    if check["median_cosine"] < MIN_CHECK_COSINE:
        raise SystemExit("this run does not reproduce the recorded chunk vectors; the masked vectors would "
                         "not be comparable and are not written")

    blocks_dir = store / "blocks"
    blocks_dir.mkdir(exist_ok=True)
    stamp = blocks_dir / "masked_text_sha256.txt"
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() != digest:
        raise SystemExit(f"{blocks_dir} holds blocks of another masked text; delete it first")
    stamp.write_text(digest + "\n", encoding="utf-8")
    n_blocks = (len(texts) + BLOCK - 1) // BLOCK
    started = time.time()
    for b in range(n_blocks):
        target = blocks_dir / f"{b}.npy"
        if target.is_file():
            try:
                if np.load(target).shape[0] == min(BLOCK, len(texts) - b * BLOCK):
                    continue
            except Exception:
                pass
        block_vectors = model.encode(texts[b * BLOCK:(b + 1) * BLOCK], batch_size=BATCH, max_length=MAX_LENGTH)["dense_vecs"]
        np.save(target, np.asarray(block_vectors, dtype=np.float32))
        if (b + 1) % 5 == 0 or b + 1 == n_blocks:
            print(f"  block {b + 1}/{n_blocks}  {(time.time() - started) / 60:.1f} min", flush=True)
    matrix = np.concatenate([np.load(blocks_dir / f"{b}.npy") for b in range(n_blocks)], axis=0)
    if matrix.shape != (len(texts), 1024):
        raise SystemExit("the concatenated blocks do not cover the masked table")
    np.save(store / "masked_chunk_bge_m3.npy", matrix)
    embedding_contract = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": weights_sha,
                  "implementation": "FlagEmbedding.BGEM3FlagModel dense vectors"},
        "configuration": {"device": "cuda", "gpu": torch.cuda.get_device_name(0), "use_fp16": True,
                          "batch_size": BATCH, "max_length": MAX_LENGTH, "block_size": BLOCK,
                          "order": "table row order", "torch": torch.__version__},
        "chunks": len(texts), "dimension": 1024, "corpus_content_sha256": V3_CONTENT_SHA256,
        "masked_text_sha256": digest, "labels_sha256": contract["labels_sha256"],
        "embeddings_sha256": sha256_file(store / "masked_chunk_bge_m3.npy"),
        "check_unmasked_chunks": check, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "private; derived from copyrighted lyric text. Never commit."}
    (store / "masked_embedding_contract.json").write_text(
        json.dumps(embedding_contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"  wrote the masked chunk vectors to {store}; rerun the CPU stage to score the semantic column", flush=True)
    return 0


# ------------------------------------------------------------------ CPU stage
def stage_score(private_root: Path, out_dir: Path) -> int:
    started = time.time()
    import jieba
    jieba.setLogLevel(60)
    expected = published_numbers(out_dir)
    audit = audit_numbers(out_dir)
    print("loading corpus v3 and building the protocol's population", flush=True)
    d = setup(private_root)
    rows, vectors, songs, label_by_song, groups = d["rows"], d["vectors"], d["songs"], d["label_by_song"], d["groups"]
    base = population(songs, label_by_song, groups)
    if base["songs"] != songs or base["labels"] != d["eligible"]:
        raise SystemExit("the population rule does not reproduce the protocol's query set")
    n, L = len(base["songs"]), base["label_count"]
    print(f"  {n:,} queries, {L} labels, {base['groups']:,} groups (largest {base['largest_group']})", flush=True)
    song_pos = {s: i for i, s in enumerate(songs)}
    dense_all = v1.l2_normalize_dense(np.stack([d["centroids_by_song"][s] for s in songs]))
    documents = [d["documents"][s] for s in songs]

    # ---- baseline: the reproduction check before anything new
    print("baseline: fitting the character and word spaces", flush=True)
    char_matrix, word_matrix = fit_spaces(documents)
    print(f"  characters {char_matrix.shape[1]:,} features, words {word_matrix.shape[1]:,} features", flush=True)
    base_scores, base_empty = score_spaces(base, dense_all, char_matrix, word_matrix)
    base_systems, base_contrasts = evaluate(base, base_scores, base_empty)
    checks_published = check_published(base_systems, expected)
    arms = {"baseline": {"population": population_summary(base, n, L), "systems": base_systems,
                         "paired_contrasts": base_contrasts, "reading": reading_of(base_contrasts, base_scores)}}
    report_arm("baseline", arms["baseline"])
    print(f"  {(time.time() - started) / 60:.1f} min", flush=True)

    # ---- (a) mask every label string of the corpus
    print("mask_all_labels: masking every label string of the corpus in every chunk", flush=True)
    labels_all = sorted({r["source_credit_label"] for r in rows})
    label_pos = {l: i for i, l in enumerate(labels_all)}
    chunk_rows = [i for s in songs for i in sorted(d["chunks_by_song"][s], key=lambda i: int(rows[i]["source_order"]))]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    own = [label_pos[rows[i]["source_credit_label"]] for i in chunk_rows]
    masked_texts, counts, tally = mask_all_labels([rows[i]["cleaned_text"] for i in chunk_rows], labels_all, own)
    summary = masking_summary(counts, tally, len(chunk_rows), [len(l.strip()) for l in labels_all])
    summary["labels_in_corpus"] = len(labels_all)
    summary["eligible_labels"] = L
    # the audit's own-label test on the original text of every corpus row, for the recorded comparison
    self_all = sum(1 for r in rows if names_label(r["cleaned_text"], r["source_credit_label"]))
    summary["audit_style_self_mentions"] = {"chunks_all_rows": int(self_all), "audit": audit["self_mention_chunks"],
                                            "matches_audit": (None if audit["self_mention_chunks"] is None
                                                              else bool(self_all == audit["self_mention_chunks"]))}
    print(f"  {summary['occurrences_total']:,} occurrences of {summary['labels_with_any_occurrence']} of "
          f"{summary['labels_in_corpus']} label strings masked in {summary['chunks_touched']:,} of {summary['chunks']:,} "
          f"chunks ({summary['share_of_chunks_touched']:.1%}); own-label {summary['own_label_occurrences']:,}, "
          f"other-label {summary['other_label_occurrences']:,}", flush=True)
    print(f"  audit-style self-mentions over all rows: {self_all:,} (audit {audit['self_mention_chunks']}; "
          f"match {summary['audit_style_self_mentions']['matches_audit']})", flush=True)
    masked_doc = defaultdict(list)
    for k, i in enumerate(chunk_rows):
        masked_doc[rows[i]["song_id"]].append(masked_texts[k])
    masked_documents = {s: "".join(masked_doc[s]) for s in songs}
    keep = {s: len(v1.normalized_text(masked_documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS for s in songs}
    masked_pop = population(songs, label_by_song, groups, keep)
    print(f"  {n - masked_pop['songs_offered']} songs fall below the length rule after masking; "
          f"{len(masked_pop['songs']):,} queries, {masked_pop['label_count']} labels remain", flush=True)
    check_rows = sorted(np.random.default_rng(SEED).choice(len(chunk_rows), size=CHECK_CHUNKS, replace=False).tolist())
    store = private_dir_of(private_root)
    table_contract = write_masked_table(store, rows, chunk_rows, masked_texts, [chunk_rows[k] for k in check_rows],
                                        labels_all, summary, counts_by_label=counts)
    masked_vectors, why = masked_vectors_if_ready(store, table_contract["masked_text_sha256"], len(chunk_rows))
    semantic_status = {"status": "scored" if masked_vectors is not None else why.split(":")[0],
                       "detail": (why if masked_vectors is None else
                                  {k: why[k] for k in ("model", "configuration", "check_unmasked_chunks", "embeddings_sha256")}),
                       "private_inputs": f"work/{PRIVATE_DIR_NAME}/masked_chunks_v3.csv and masked_table_contract.json",
                       "how": "--stage embed embeds the masked chunks with FlagEmbedding under the recorded chunk run's "
                              f"configuration (fp16, max_length {MAX_LENGTH}, batch {BATCH}) after reproducing the recorded "
                              f"vectors on {CHECK_CHUNKS} unmasked chunks (median cosine >= {MIN_CHECK_COSINE}); the CPU "
                              "stage then scores the song centroids (plain mean of the masked chunk vectors, normalised, "
                              "as build_songs does) under the same protocol"}
    print(f"  semantic column: {semantic_status['status']}", flush=True)
    print("  fitting the masked character and word spaces", flush=True)
    m_char, m_word = fit_spaces([masked_documents[s] for s in masked_pop["songs"]])
    if masked_vectors is not None:
        acc = np.zeros((n, 1024))
        np.add.at(acc, chunk_song, masked_vectors.astype(np.float64))
        centroids = acc / np.bincount(chunk_song, minlength=n)[:, None]
        m_dense = v1.l2_normalize_dense(centroids[[song_pos[s] for s in masked_pop["songs"]]])
    else:
        m_dense = dense_all[[song_pos[s] for s in masked_pop["songs"]]]
    m_scores, m_empty = score_spaces(masked_pop, m_dense, m_char, m_word, with_semantic=masked_vectors is not None)
    m_systems, m_contrasts = evaluate(masked_pop, m_scores, m_empty)
    arms["mask_all_labels"] = {"population": population_summary(masked_pop, n, L), "masking": summary,
                               "semantic_column": semantic_status, "systems": m_systems,
                               "paired_contrasts": m_contrasts, "reading": reading_of(m_contrasts, m_scores)}
    report_arm("mask_all_labels", arms["mask_all_labels"])
    print(f"  {(time.time() - started) / 60:.1f} min", flush=True)

    # ---- (b) collaboration titles out
    print("exclude_collaboration_titles", flush=True)
    titles = {r["song_id"]: r["song_title"] for r in rows}
    collaboration = {s: bool(FEAT.search(titles[s])) for s in songs}
    found = sum(collaboration.values())
    print(f"  {found:,} of {n:,} queries carry a collaboration mark in the title "
          f"(audit: {audit['collaborations']}; match {found == audit['collaborations']})", flush=True)
    c_songs = [s for s in songs if not collaboration[s]]
    c_groups = build_groups(c_songs, d["components_by_song"], d["normalised"])
    c_pop = population(c_songs, label_by_song, c_groups)
    print(f"  {len(c_pop['songs']):,} queries, {c_pop['label_count']} labels, {c_pop['groups']:,} groups remain "
          f"({c_pop['songs_dropped_by_label_rule']} more songs and {L - c_pop['label_count']} labels lost to the label rule)", flush=True)
    c_char, c_word = fit_spaces([d["documents"][s] for s in c_pop["songs"]])
    c_scores, c_empty = score_spaces(c_pop, dense_all[[song_pos[s] for s in c_pop["songs"]]], c_char, c_word)
    c_systems, c_contrasts = evaluate(c_pop, c_scores, c_empty)
    arms["exclude_collaboration_titles"] = {
        "population": population_summary(c_pop, n, L),
        "collaboration_titles": {"found": int(found), "share": round(found / n, 4), "audit": audit["collaborations"],
                                 "matches_audit": (None if audit["collaborations"] is None else bool(found == audit["collaborations"])),
                                 "rule": "training_data_audit_v3.FEAT (feat., ft., featuring, &, /, x, ×, or a list separator; "
                                         "case-insensitive) on the song title",
                                 "needs_private_metadata": "yes: the song_title column of the private corpus table, which no "
                                                           "public artifact carries; no other metadata is read",
                                 "groups": "rebuilt with leakage_groups_v2.build_groups on the remaining songs"},
        "systems": c_systems, "paired_contrasts": c_contrasts, "reading": reading_of(c_contrasts, c_scores)}
    report_arm("exclude_collaboration_titles", arms["exclude_collaboration_titles"])
    print(f"  {(time.time() - started) / 60:.1f} min", flush=True)

    # ---- (c) near-identical rival queries dropped (profiles unchanged: baseline scores, restricted)
    print("drop_near_identical_rival_queries: nearest other-label chunk in the recorded space", flush=True)
    chunk_vec = unit_rows(vectors[chunk_rows].astype(np.float64))
    chunk_label = base["label_index"][chunk_song]
    chunk_group = base["group_ids"][chunk_song]
    anywhere, outside = nearest_rival(chunk_vec, chunk_label, chunk_group)
    del chunk_vec
    rival_arms = {}
    for variant, values in (("rival_anywhere", anywhere), ("rival_outside_leakage_group", outside)):
        song_values = song_maximum(values, chunk_song, n)
        mask = ~(song_values > RIVAL_COSINE)
        r_systems, r_contrasts = evaluate(base, base_scores, base_empty, mask)
        finite = values[np.isfinite(values)]
        rival_arms[variant] = {"role": ("the arm as specified: the audit's definition, any chunk of another label"
                                        if variant == "rival_anywhere" else
                                        "companion, not the arm: the rival chunk must lie outside the query's leakage group"),
                               "queries_dropped": int((~mask).sum()), "queries_kept": int(mask.sum()),
                               "share_dropped": round(float((~mask).mean()), 4),
                               "chunks_over_threshold": int((values > RIVAL_COSINE).sum()),
                               "chunk_nearest_rival_cosine_quantiles": {str(p): round(float(np.percentile(finite, p)), 3)
                                                                        for p in (10, 50, 90)},
                               "systems": r_systems, "paired_contrasts": r_contrasts,
                               "reading": reading_of(r_contrasts, base_scores)}
        print(f"  {variant}: {rival_arms[variant]['queries_dropped']:,} queries dropped "
              f"({rival_arms[variant]['share_dropped']:.1%}); chunk nearest-rival cosine quantiles "
              f"{rival_arms[variant]['chunk_nearest_rival_cosine_quantiles']}", flush=True)
        report_arm(variant, rival_arms[variant])
    arms["drop_near_identical_rival_queries"] = {"threshold": RIVAL_COSINE,
                                                 "profiles": "unchanged (baseline scores restricted to the kept queries)",
                                                 **rival_arms}
    print(f"  {(time.time() - started) / 60:.1f} min", flush=True)

    # ---- (d) leakage threshold
    for threshold in THRESHOLDS:
        name = f"leakage_threshold_{threshold:.2f}"
        print(f"{name}: rebuilding the leakage groups", flush=True)
        t_groups = build_groups(songs, d["components_by_song"], d["normalised"], threshold=threshold)
        t_audit = group_audit(t_groups, label_by_song, d["components_by_song"])
        t_pop = population(songs, label_by_song, t_groups)
        print(f"  {t_audit['groups']:,} groups, largest {t_audit['largest_group_songs']}, min per label "
              f"{t_audit['minimum_groups_in_a_label']}; {len(t_pop['songs']):,} queries, {t_pop['label_count']} labels", flush=True)
        if t_pop["songs"] == songs:
            t_char, t_word, t_dense = char_matrix, word_matrix, dense_all
        else:
            t_char, t_word = fit_spaces([d["documents"][s] for s in t_pop["songs"]])
            t_dense = dense_all[[song_pos[s] for s in t_pop["songs"]]]
        t_scores, t_empty = score_spaces(t_pop, t_dense, t_char, t_word)
        t_systems, t_contrasts = evaluate(t_pop, t_scores, t_empty)
        arms[name] = {"threshold": threshold, "population": population_summary(t_pop, n, L), "group_audit": t_audit,
                      "systems": t_systems, "paired_contrasts": t_contrasts, "reading": reading_of(t_contrasts, t_scores)}
        report_arm(name, arms[name])
        print(f"  {(time.time() - started) / 60:.1f} min", flush=True)

    readings = {name: arm["reading"] for name, arm in arms.items() if "reading" in arm}
    for variant in rival_arms:
        readings[f"drop_near_identical_rival_queries/{variant}"] = rival_arms[variant]["reading"]
    print("readings:", flush=True)
    for key, value in readings.items():
        print(f"  {key}: {value}", flush=True)

    payload = {
        "analysis": "sensitivity of the words > characters > semantic ordering to label masking, collaboration titles, "
                    "near-identical rival chunks and the leakage threshold",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n, "labels": L, "groups": base["groups"],
                   "chunks_of_queries": len(chunk_rows), "vectors": d["state"]["vectors"]},
        "design": {
            "protocol": "one document and one centroid per song; label profile = leave-group-out weighted sum; cosine rank "
                        "of the true label; build_chinese_rap_downstream_retrieval_v1.score_leave_group_out and rank_system; "
                        "a sparse row without features takes the worst rank",
            "spaces": {"semantic": "BGE-M3 song centroids (recorded v3 chunk run)", "characters": "character 2-5-gram TF-IDF",
                       "words": "jieba word 1-2-gram TF-IDF"},
            "arms": {
                "baseline": "the published configuration; reproduction check",
                "mask_all_labels": "every label string of the corpus removed from every chunk with "
                                   "train_identity_encoder_v3.mask_label, longest first, before fitting the word and "
                                   "character spaces; length rule re-applied; baseline groups kept; the semantic column "
                                   "needs the masked chunks embedded by --stage embed",
                "exclude_collaboration_titles": "songs with a collaboration mark in the title (training_data_audit_v3.FEAT on "
                                                "the private song_title column) removed as queries and profile members; groups "
                                                "rebuilt on the remaining songs; spaces refitted; labels re-checked",
                "drop_near_identical_rival_queries": f"queries with a chunk whose nearest other-label chunk exceeds cosine "
                                                     f"{RIVAL_COSINE} dropped as queries, profiles unchanged; rival_anywhere is the "
                                                     "arm as specified (the audit's definition); rival_outside_leakage_group is a "
                                                     "companion that excludes the query's own group",
                "leakage_threshold": f"groups rebuilt at trigram Jaccard {THRESHOLDS} instead of {NEAR_DUPLICATE_JACCARD}; "
                                     "weights, scoring and bootstrap follow"},
            "population_rule": f"a label needs {MINIMUM_SONGS_PER_LABEL} songs and {MIN_GROUPS_PER_LABEL} leakage groups in the arm",
            "estimands": "mrr = plain mean over the arm's queries (the published estimand); mrr_component_weighted = the "
                         "bootstrap's per-(group, label)-weighted estimand; contrasts are on the weighted estimand",
            "bootstrap": "identity_spaces_v2.paired_group_bootstrap: 2000 replicates, seed 20260825, the arm's leakage groups "
                         "resampled with replacement, each (group, label) component weighted one",
            "reading_rule": "the ordering words > characters > semantic is robust to the arm if every pairwise contrast "
                            "(words - characters, characters - semantic, words - semantic) keeps its sign (positive, as at "
                            "baseline) with its interval clear of zero; otherwise not robust and the failing contrast is named; "
                            "with the semantic column pending the arm reads partial on words - characters alone",
            "checks": f"baseline reproduces the published MRR in every space (gap <= {CHECK_GAP}) and the pinned literals equal "
                      f"three_spaces.json; the population rule rebuilds the protocol's query set; dense scores identical across "
                      f"the two scorer calls (gap <= {DENSE_GAP}); collaboration and self-mention counts compared with the audit; "
                      "masked semantic column used only under a contract naming the current masked-text digest with median "
                      f"unmasked-chunk cosine >= {MIN_CHECK_COSINE}"},
        "checks_against_published": checks_published,
        "arms": arms,
        "readings": readings,
        "privacy": "aggregate only; the masked chunk table and its vectors are private",
        "minutes": round((time.time() - started) / 60, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUT_NAME).write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"\nwrote {out_dir / OUT_NAME}  ({payload['minutes']} min)", flush=True)
    return 0


def population_summary(pop: dict, base_queries: int, base_labels: int) -> dict:
    return {"queries": len(pop["songs"]), "labels": pop["label_count"], "groups": pop["groups"],
            "largest_group": pop["largest_group"], "queries_removed_from_baseline": base_queries - len(pop["songs"]),
            "labels_removed_from_baseline": base_labels - pop["label_count"],
            "songs_dropped_by_label_rule": pop["songs_dropped_by_label_rule"]}


def report_arm(name: str, arm: dict) -> None:
    print(f"== {name}: " + "  ".join(f"{s}={v['mrr']:.4f}" for s, v in arm["systems"].items()), flush=True)
    for c in arm["paired_contrasts"]:
        print(f"    {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)
    print(f"    {arm['reading']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stage", choices=["score", "embed"], default="score",
                        help="score: the CPU analysis (default); embed: the GPU stage for the masked semantic column")
    args = parser.parse_args()
    if os.environ.get("CHINESE_RAP_CORPUS") != "v3":
        raise SystemExit("set CHINESE_RAP_CORPUS=v3; this analysis is a corpus v3 result")
    if args.stage == "embed":
        return stage_embed(args.private_root.resolve())
    return stage_score(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
