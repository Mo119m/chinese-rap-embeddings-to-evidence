#!/usr/bin/env python3
"""Whole-song embedding against the stanza mean: does averaging stanzas lose what the encoder
would read from the song as one text?

Every semantic number in this project represents a song as the mean of its stanza (chunk)
vectors. Half of the song records (3,649 of 7,379) are a single stanza, so for them the mean
is already the whole song; for the rest, what the mean discards is stanza order and whatever
the encoder would make of the song as a single sequence. The alternatives measured so
far all keep the stanza as the unit: a chunk on its own 0.188, max-sim over chunks 0.281, the
mean 0.300, whitening before the mean 0.427. This file measures the one not yet tried: the
whole song, stanzas joined by newlines in source order, embedded once by the same pinned
BGE-M3 (FlagEmbedding, fp16, max_length 8,192) and scored under the unchanged protocol.

Stage 1 (GPU) embeds every song record and runs one check before anything is scored: a song
with a single stanza is the same text as its stanza, so its whole-song vector must reproduce
the recorded chunk vector (cosine at least 0.999 for the median such song). Stage 2 (CPU)
scores both representations raw, after total and within-author whitening cross-fitted over
the five leakage-group folds, and fused with the word space, with paired group bootstraps of
every difference; it also checks that the stanza-mean numbers reproduce the recorded 0.2997
and 0.4164.

Reading, fixed before the run: the song representation changes to the whole-song embedding
only if it beats the stanza mean after within-author whitening with an interval excluding
zero and is not worse raw; otherwise the stanza mean stands and the README records that the
whole-song embedding was tried.

    python src/whole_song_embedding_v3.py --private-root <ni-k> [--stage 1|2|both]
"""

from __future__ import annotations

import argparse
import csv
import json
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
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from embed_corpus_local_v3 import MODEL_ID, MODEL_REVISION, PINNED_WEIGHTS_SHA256, sha256_file  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
MAX_LENGTH = 8192           # BGE-M3's limit
RECORDED_MAX_LENGTH = 2048  # the recorded chunk run's limit
BATCH = 4
RECORDED = {"cosine": ("three_spaces.json", ("systems", "semantic", "mrr")),
            "within_author_whitening": ("identity_probe.json", ("systems", "within_author_whitening", "mrr"))}


def private_dir_of(private_root: Path) -> Path:
    return private_root / "work" / "private-whole-song-v3"


def song_texts(rows, chunks_by_song):
    """One text per song record: its stanzas in source order, joined by newlines."""
    return {song: "\n".join(rows[i]["cleaned_text"] for i in sorted(indices, key=lambda i: int(rows[i]["source_order"])))
            for song, indices in chunks_by_song.items()}


# ------------------------------------------------------------------ stage 1
def stage_one(private_root: Path, rows, vectors) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from FlagEmbedding import BGEM3FlagModel
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this run must match the recorded run's device class")
    chunks_by_song, _, _, _, _ = build_songs(rows, vectors)
    song_ids = list(chunks_by_song)
    texts = song_texts(rows, chunks_by_song)
    model_path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
    weights = next((model_path / n for n in ("pytorch_model.bin", "model.safetensors") if (model_path / n).is_file()), None)
    weights_sha = sha256_file(weights) if weights else None
    if weights_sha != PINNED_WEIGHTS_SHA256:
        raise SystemExit(f"weights sha256 {weights_sha} is not the pinned {PINNED_WEIGHTS_SHA256}")
    print(f"  weights pin matched; {len(song_ids):,} song records", flush=True)
    model = BGEM3FlagModel(str(model_path), use_fp16=True, devices="cuda")
    lengths = np.asarray([len(model.tokenizer(texts[s], add_special_tokens=True, truncation=False)["input_ids"])
                          for s in song_ids])
    truncated = int((lengths > MAX_LENGTH).sum())
    print(f"  tokens per song: median {int(np.median(lengths))}, p90 {int(np.percentile(lengths, 90))}, "
          f"max {int(lengths.max())}; {truncated} over {MAX_LENGTH}", flush=True)
    started = time.time()
    dense = model.encode([texts[s] for s in song_ids], batch_size=BATCH, max_length=MAX_LENGTH)["dense_vecs"]
    dense = np.asarray(dense, dtype=np.float32)
    print(f"  embedded in {(time.time() - started) / 60:.1f} min", flush=True)

    # the check: a one-stanza song is its stanza, so its vector must be the recorded chunk vector.
    # The recorded run read at most 2,048 tokens, this one 8,192, so a one-stanza song longer
    # than 2,048 tokens is legitimately different and is counted, not compared.
    single_all = [k for k, s in enumerate(song_ids) if len(chunks_by_song[s]) == 1]
    single = [k for k in single_all if lengths[k] <= RECORDED_MAX_LENGTH]
    recorded = unit_rows(vectors[[chunks_by_song[song_ids[k]][0] for k in single]].astype(np.float32))
    mine = unit_rows(dense[single])
    agreement = np.sum(mine * recorded, axis=1)
    check = {"single_stanza_songs": len(single_all), "compared_at_most_2048_tokens": len(single),
             "min_cosine": round(float(agreement.min()), 6), "median_cosine": round(float(np.median(agreement)), 6),
             "songs_below_0.999": int((agreement < 0.999).sum())}
    print(f"  single-stanza songs against the recorded chunk run: {check}", flush=True)

    out = private_dir_of(private_root)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "whole_song_bge_m3.npy", dense)
    with (out / "row_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_index", "song_id", "chunks", "tokens"])
        for k, s in enumerate(song_ids):
            writer.writerow([k, s, len(chunks_by_song[s]), int(lengths[k])])
    contract = {"model": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": weights_sha,
                "implementation": "FlagEmbedding.BGEM3FlagModel dense vectors", "use_fp16": True,
                "max_length": MAX_LENGTH, "batch_size": BATCH, "songs": len(song_ids),
                "text": "stanzas in source order joined by newlines",
                "tokens_per_song": {"median": int(np.median(lengths)), "p90": int(np.percentile(lengths, 90)),
                                    "max": int(lengths.max()), "over_max_length": truncated,
                                    "over_recorded_max_length": int((lengths > RECORDED_MAX_LENGTH).sum())},
                "corpus_content_sha256": V3_CONTENT_SHA256, "check_single_stanza_songs": check,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "warning": "private; derived from copyrighted lyric text. Never commit."}
    (out / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if check["median_cosine"] < 0.999:
        raise SystemExit("the whole-song run does not reproduce the recorded chunk vectors on one-stanza songs; "
                         "stage 2 would compare two different pipelines")


# ------------------------------------------------------------------ stage 2
def stage_two(private_root: Path, out_dir: Path, rows, vectors) -> int:
    store = private_dir_of(private_root)
    contract = json.loads((store / "contract.json").read_text(encoding="utf-8"))
    if contract["corpus_content_sha256"] != V3_CONTENT_SHA256:
        raise SystemExit("the whole-song vectors were computed on another corpus build")
    if contract["check_single_stanza_songs"]["median_cosine"] < 0.999:
        raise SystemExit("the whole-song run failed its stage 1 check")
    whole = np.load(store / "whole_song_bge_m3.npy")
    position = {r["song_id"]: int(r["row_index"]) for r in csv.DictReader((store / "row_map.csv").open(encoding="utf-8"))}

    chunks_by_song, label_by_song, components_by_song, documents, centroids_by_song = build_songs(rows, vectors)
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {s for s in chunks_by_song
                   if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(l for l, m in songs_by_label.items()
                      if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
    label_index = np.asarray([eligible.index(label_by_song[s]) for s in songs], dtype=np.int64)
    groups = build_groups(songs, components_by_song, {s: normalise_document(documents[s]) for s in songs})
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    label_count = len(eligible)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    everything = np.arange(len(songs))
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)

    spaces = {"chunk_mean": v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs])).astype(np.float64),
              "whole_song": unit_rows(whole[[position[s] for s in songs]].astype(np.float64))}
    agreement = np.sum(spaces["chunk_mean"] * spaces["whole_song"], axis=1)
    print(f"  cosine between a song's two representations: median {np.median(agreement):.3f}, "
          f"p10 {np.percentile(agreement, 10):.3f}", flush=True)

    scores: dict[str, np.ndarray] = {}
    for name, song_vec in spaces.items():
        scores[f"{name}/cosine"] = dense_leave_group_out(song_vec, label_index, group_ids, weights, label_count, everything)
        for transform in ("total_whitening", "within_author_whitening"):
            out = np.zeros((len(songs), label_count))
            for k in range(FOLDS):
                train = fold != k
                queries = np.flatnonzero(~train)
                mean, matrix, _ = fit_transform(transform, song_vec[train], label_index[train], weights[train],
                                                np.random.default_rng(SEED + k))
                out[queries] = dense_leave_group_out(unit_rows((song_vec - mean) @ matrix.T), label_index, group_ids,
                                                     weights, label_count, queries)
            scores[f"{name}/{transform}"] = out
        print(f"  {name}: scored", flush=True)

    print("fitting the word space", flush=True)
    word_matrix, _ = fit_words([" ".join(segment(documents[s])) for s in songs])
    lexical = v1.score_leave_group_out(spaces["chunk_mean"].astype(np.float32), word_matrix, label_index, group_ids,
                                       label_count).lexical.astype(np.float64)
    scores["words"] = lexical
    for name in spaces:
        for transform in ("cosine", "within_author_whitening"):
            scores[f"{name}/fusion_{transform}_words"] = (v1.zscore_rows(scores[f"{name}/{transform}"])
                                                         + v1.zscore_rows(lexical)) / 2.0

    ranks = {name: v1.rank_system(s.astype(np.float32), label_index)[0].astype(np.int64) for name, s in scores.items()}
    systems = {name: {"mrr": round(float(np.mean(1.0 / r)), 4), "recall_at_1": round(float(np.mean(r <= 1)), 4),
                      "recall_at_10": round(float(np.mean(r <= 10)), 4)} for name, r in ranks.items()}
    for name, s in systems.items():
        print(f"  {name:48s} MRR {s['mrr']:.4f}  R@1 {s['recall_at_1']:.4f}  R@10 {s['recall_at_10']:.4f}", flush=True)

    checks = {}
    for transform, (file_name, path) in RECORDED.items():
        recorded = json.loads((out_dir / file_name).read_text(encoding="utf-8"))
        value = recorded
        for key in path:
            value = value[key]
        checks[transform] = {"recorded": value, "recomputed": systems[f"chunk_mean/{transform}"]["mrr"],
                             "gap": round(abs(systems[f"chunk_mean/{transform}"]["mrr"] - value), 4)}
    print(f"  stanza mean against the recorded results: {checks}", flush=True)
    if any(c["gap"] > 0.002 for c in checks.values()):
        raise SystemExit("the stanza mean does not reproduce the recorded MRRs; the protocol here differs")

    rr = {name: 1.0 / r for name, r in ranks.items()}
    pairs = [(f"whole_song/{t}", f"chunk_mean/{t}") for t in
             ("cosine", "total_whitening", "within_author_whitening", "fusion_cosine_words", "fusion_within_author_whitening_words")]
    pairs += [("whole_song/within_author_whitening", "words"), ("chunk_mean/within_author_whitening", "words")]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)
    main = next(c for c in contrasts if c["system"] == "whole_song/within_author_whitening" and c["minus"] == "chunk_mean/within_author_whitening")
    raw = next(c for c in contrasts if c["system"] == "whole_song/cosine" and c["minus"] == "chunk_mean/cosine")
    switch = bool(main["ci95"][0] > 0 and raw["ci95"][1] >= 0)
    verdict = ("the whole-song embedding beats the stanza mean after whitening and is not worse raw: switch the "
               "representation and recompute" if switch else
               "the stanza mean stands; the whole-song embedding was tried and does not beat it under the rule")
    print(f"  verdict: {verdict}", flush=True)

    # where the difference lives: a one-stanza song is the same text in both representations,
    # so any gap must come from songs with two or more stanzas
    stanzas = np.asarray([len(chunks_by_song[s]) for s in songs])
    buckets = {"1": stanzas == 1, "2": stanzas == 2, "3-5": (stanzas >= 3) & (stanzas <= 5), "6+": stanzas >= 6,
               "2+": stanzas >= 2}
    by_stanzas = {}
    for bucket, mask in buckets.items():
        entry = {"queries": int(mask.sum())}
        for name in ("chunk_mean/cosine", "whole_song/cosine", "chunk_mean/within_author_whitening",
                     "whole_song/within_author_whitening"):
            entry[name] = round(float(np.mean(1.0 / ranks[name][mask])), 4)
        entry["contrasts"] = paired_group_bootstrap(
            rr, weights, group_ids, mask,
            [("whole_song/cosine", "chunk_mean/cosine"),
             ("whole_song/within_author_whitening", "chunk_mean/within_author_whitening")])
        by_stanzas[bucket] = entry
        raw_c, wh_c = entry["contrasts"]
        print(f"  stanzas {bucket:4s} n {entry['queries']:5d}  raw {entry['chunk_mean/cosine']:.4f} -> "
              f"{entry['whole_song/cosine']:.4f} ({raw_c['mrr_difference']:+.4f} [{raw_c['ci95'][0]:+.4f}, {raw_c['ci95'][1]:+.4f}])  "
              f"whitened {entry['chunk_mean/within_author_whitening']:.4f} -> {entry['whole_song/within_author_whitening']:.4f} "
              f"({wh_c['mrr_difference']:+.4f} [{wh_c['ci95'][0]:+.4f}, {wh_c['ci95'][1]:+.4f}])", flush=True)

    payload = {
        "analysis": "one whole-song BGE-M3 embedding per song against the mean of its stanza vectors, unchanged protocol",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count, "groups": len(order)},
        "design": {"whole_song": {k: contract[k] for k in ("model", "revision", "implementation", "use_fp16", "max_length",
                                                            "batch_size", "text", "tokens_per_song")},
                   "stanza_mean": "the project's song vector: mean of the recorded chunk vectors, normalised",
                   "whitening": "total and within-author, Ledoit-Wolf shrunk, cross-fitted over the five leakage-group folds",
                   "fusion": "z-score mean of the dense and word (jieba TF-IDF) score rows",
                   "checks": {"single_stanza_songs_against_recorded_chunks": contract["check_single_stanza_songs"],
                              "stanza_mean_against_recorded_results": checks},
                   "representation_agreement": {"median_cosine": round(float(np.median(agreement)), 4),
                                                "p10_cosine": round(float(np.percentile(agreement, 10)), 4)},
                   "reading_rule": "switch to the whole-song embedding only if it beats the stanza mean after "
                                   "within-author whitening with an interval excluding zero and is not worse raw",
                   "verdict": verdict, "switch_representation": switch},
        "systems": systems,
        "by_stanza_count": {"note": "MRR is the plain mean over the bucket's queries; contrasts are the protocol's "
                                    "weighted estimand with a group bootstrap restricted to the bucket",
                            "buckets": by_stanzas},
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with replacement, "
                                       "each (group, label) component weighted one", "contrasts": contrasts},
        "privacy": "aggregate only; vectors are private",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "whole_song_embedding.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'whole_song_embedding.json'}")
    return 0


def build(private_root: Path, out_dir: Path, stage: str) -> int:
    print("loading corpus v3", flush=True)
    rows, vectors, state = load_v3(private_root)
    if state["vectors"] != "v3":
        raise SystemExit("this comparison needs the recorded v3 chunk embedding run")
    if stage in ("1", "both"):
        print("stage 1: one embedding per whole song", flush=True)
        stage_one(private_root, rows, vectors)
    if stage in ("2", "both"):
        print("stage 2: whole song against the stanza mean", flush=True)
        stage_two(private_root, out_dir, rows, vectors)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir, args.stage)


if __name__ == "__main__":
    sys.exit(main())
