#!/usr/bin/env python3
"""What did the 64-anchor fine-tune learn beyond whitening? Part B: which words its advantage
depends on, tested by masking the query's words.

In every fold the GradCache fine-tune (tag _gradcache64) beats the frozen encoder after
within-author whitening. Part A (finetune_learned_identity_v3.py) describes the extra
identity; this file tests causally what it rests on. The query songs' lyrics are re-encoded
with one kind of word replaced by the tokenizer's mask token, and, as the control, with the
same number of other words replaced at random. The reference label profiles stay unmasked.
If the tuned model's advantage over the frozen model shrinks more when a kind of word is
masked than when as many random words are, the advantage rests on that kind of word.

Kinds of word (jieba tokens of the label-masked chunk text, the text the adapter was trained
and scored on):
  latin        tokens containing a Latin letter: English words, ad-libs, abbreviations
  common_han   the 100 Han-character words found in the most songs: the common vocabulary
               whose relative use carries identity in the word space

For each fold k, over the fold's test songs and the 34 unseen labels' songs:
  stage 1 (GPU)  encode every query chunk under four conditions (latin masked, latin random,
                 common masked, common random) with the fold's adapter and with the adapter
                 disabled (the frozen encoder). Check first: 256 unmasked query chunks must
                 reproduce the saved vectors of both models (median cosine at least 0.999).
  stage 2 (CPU)  song vectors are chunk means; within-author whitening is fitted on the
                 fold's training songs (unmasked) as in analyse_identity_encoder_v3; queries
                 take their masked vectors, label profiles the unmasked ones. Check first: the
                 split scorer equals the protocol's scorer when queries are unmasked, and the
                 unmasked MRRs reproduce the recorded analysis.

Reading, fixed before the run. For a kind of word C and a scope, dependence = (tuned - frozen
under random masking) - (tuned - frozen under masking C), with a group bootstrap interval.
  - the advantage rests on C beyond its token count: dependence > 0 with the interval clear
    of zero in at least four of the five folds
  - it does not: the interval includes zero, or dependence < 0, in at least four folds
  - otherwise partly, reported fold by fold
Each model's own reliance on C (random masking minus masking C) is reported beside it. Masking
replaces a word by one mask token, so masked texts are shorter than the original in tokens;
the random control is matched in words replaced, not in characters.

Added after fold 0's stage 1 printed its word counts and before any masked MRR was computed:
in a chunk with more words of C than other words (mostly English chunks for latin) the random
control cannot mask as many words. In fold 0 this left the latin control 16% short in words
(264 of 1,187 test songs and 195 of 1,041 unseen-label songs contain such a chunk). The bias
this introduces has no fixed sign, so stage 2 recomputes each chunk's counts (checked against
stage 1's totals) and applies the same rule to the songs whose every chunk is count-matched.
A reading is stated only when all songs and matched songs give the same reading; otherwise it
is "depends on the songs whose control is short; read fold by fold".

Added after fold 0's result: masking the 100 common Han words replaces a third of all words,
and in fold 0 the tuned advantage was gone under the random control as well (test fold -0.003).
Dependence cannot show when the control alone removes the advantage, so a reading other than
"rests on" is marked uninformative unless the advantage under the random control is clear of
zero in at least four folds.

    python src/finetune_cue_masking_v3.py --private-root <ni-k> --stage 1 --fold 0
    python src/finetune_cue_masking_v3.py --private-root <ni-k> --stage 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of, song_level  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, fit_transform, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from train_identity_encoder_v3 import MODEL_ID, MODEL_REVISION, encode_all, mask_label  # noqa: E402
from word_identity_anatomy_v2 import segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
TAG = "_gradcache64"
HELD_OUT_SHARE = 0.15
MAX_LENGTH = 512
COMMON_K = 100
CHECK_CHUNKS = 256
CATEGORIES = ("latin", "common_han")
CONDITIONS = tuple(f"{c}_{v}" for c in CATEGORIES for v in ("masked", "random"))
MODELS = ("frozen", "tuned")
LATIN = re.compile(r"[A-Za-z]")
HAN = re.compile(r"[一-鿿]")
CHECK_GAP = 0.002


def setup(private_root: Path):
    """The protocol's songs, labels, groups, folds and chunk order, as every fine-tune file builds them."""
    rows, vectors, _ = load_v3(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, _ = build_songs(rows, vectors)
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
    song_pos = {s: i for i, s in enumerate(songs)}
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    held = np.zeros(label_count, dtype=bool)
    held[np.random.default_rng(SEED).permutation(label_count)[:int(round(HELD_OUT_SHARE * label_count))]] = True
    chunk_rows = [i for s in songs for i in sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"]))]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    chunk_text = [mask_label(rows[i]["cleaned_text"], rows[i]["source_credit_label"]) for i in chunk_rows]
    return dict(songs=songs, label_index=label_index, group_ids=group_ids, label_count=label_count, weights=weights,
                fold=fold, held=held, chunk_rows=chunk_rows, chunk_song=chunk_song, chunk_text=chunk_text,
                documents=documents)


def common_words(songs, documents) -> set[str]:
    df = Counter()
    for s in songs:
        df.update({t for t in segment(documents[s]) if HAN.search(t) and not LATIN.search(t)})
    return {w for w, _ in df.most_common(COMMON_K)}


def masked_versions(text: str, is_target, mask_token: str, rng: np.random.Generator):
    """(text with every target word masked, text with as many non-target words masked at
    random, number of target words, number of random words)."""
    import jieba
    tokens = list(jieba.cut(text))
    words = [i for i, t in enumerate(tokens) if t.strip()]
    target = [i for i in words if is_target(tokens[i])]
    others = [i for i in words if not is_target(tokens[i])]
    chosen = rng.choice(others, size=min(len(target), len(others)), replace=False).tolist() if target and others else []
    masked, random_masked = list(tokens), list(tokens)
    for i in target:
        masked[i] = mask_token
    for i in chosen:
        random_masked[i] = mask_token
    return "".join(masked), "".join(random_masked), len(target), len(chosen)


def mask_counts(d, query_chunks, common, mask_token: str):
    """Per chunk and category, (target words, random words), with stage 1's seeds and texts."""
    predicates = {"latin": lambda t: bool(LATIN.search(t)), "common_han": lambda t: t in common}
    counts = {c: np.zeros((len(query_chunks), 2), dtype=np.int64) for c in CATEGORIES}
    for row, j in enumerate(query_chunks):
        for ci, category in enumerate(CATEGORIES):
            rng = np.random.default_rng(SEED + 7919 * (ci + 1) + int(d["chunk_rows"][j]))
            _, _, n_target, n_random = masked_versions(d["chunk_text"][j], predicates[category], mask_token, rng)
            counts[category][row] = (n_target, n_random)
    return counts


def fold_scopes(d, k):
    return {"test_fold": np.flatnonzero((d["fold"] == k) & ~d["held"][d["label_index"]]),
            "unseen_labels": np.flatnonzero(d["held"][d["label_index"]])}


def private_dir_of(private_root: Path) -> Path:
    return private_root / "work" / "private-finetune-cue-masking-v3"


# ------------------------------------------------------------------ stage 1
def stage_one(private_root: Path, k: int) -> int:
    import jieba
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer
    jieba.setLogLevel(60)
    if not torch.cuda.is_available():
        raise SystemExit("stage 1 needs the CUDA device the fine-tune used")
    started = time.time()
    d = setup(private_root)
    scopes = fold_scopes(d, k)
    query_songs = np.union1d(scopes["test_fold"], scopes["unseen_labels"])
    query_chunks = np.flatnonzero(np.isin(d["chunk_song"], query_songs))
    common = common_words(d["songs"], d["documents"])
    predicates = {"latin": lambda t: bool(LATIN.search(t)), "common_han": lambda t: t in common}
    print(f"fold {k}: {len(query_songs):,} query songs, {len(query_chunks):,} query chunks", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    probe = tokenizer("x" + tokenizer.mask_token + "y")["input_ids"]
    if tokenizer.mask_token_id not in probe:
        raise SystemExit("the tokenizer does not read its mask token inside text")
    texts: dict[str, list[str]] = {c: [] for c in CONDITIONS}
    counts = {c: 0 for c in CONDITIONS}
    unmatched = {c: 0 for c in CATEGORIES}
    for j in query_chunks:
        text = d["chunk_text"][j]
        for ci, category in enumerate(CATEGORIES):
            # the same seed and call as mask_counts, which stage 2 uses to rebuild these counts
            rng = np.random.default_rng(SEED + 7919 * (ci + 1) + int(d["chunk_rows"][j]))
            masked, random_masked, n_target, n_random = masked_versions(text, predicates[category], tokenizer.mask_token, rng)
            texts[f"{category}_masked"].append(masked)
            texts[f"{category}_random"].append(random_masked)
            counts[f"{category}_masked"] += n_target
            counts[f"{category}_random"] += n_random
            unmatched[category] += int(n_random < n_target)
    word_total = sum(len([t for t in jieba.cut(d["chunk_text"][j]) if t.strip()]) for j in query_chunks)
    print(f"  words in query chunks {word_total:,}; masked {counts}; chunks where random could not match {unmatched}", flush=True)

    private = private_root / "work" / "private-identity-encoder-v3"
    adapter = private / f"lora_fold{k}{TAG}"
    adapter_sha = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(adapter.iterdir()) if p.is_file())).hexdigest()
    base = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter)).to("cuda")

    # check: unmasked query chunks reproduce the saved vectors of both models
    saved = {"tuned": np.load(private / f"fine_tuned_chunk_vectors_fold{k}{TAG}.npy"),
             "frozen": np.load(private / f"frozen_masked_chunk_vectors_fold{k}.npy")}
    sample = query_chunks[np.random.default_rng(SEED + k).choice(len(query_chunks), size=min(CHECK_CHUNKS, len(query_chunks)), replace=False)]
    check_texts = [d["chunk_text"][j] for j in sample]
    check = {}
    for name in MODELS:
        if name == "frozen":
            with model.disable_adapter():
                mine = encode_all(model, tokenizer, check_texts, "cuda", MAX_LENGTH)
        else:
            mine = encode_all(model, tokenizer, check_texts, "cuda", MAX_LENGTH)
        agreement = np.sum(mine * saved[name][sample], axis=1)
        check[name] = {"chunks": int(len(sample)), "min_cosine": round(float(agreement.min()), 6),
                       "median_cosine": round(float(np.median(agreement)), 6)}
    print(f"  check against the saved vectors: {check}", flush=True)
    if any(c["median_cosine"] < 0.999 for c in check.values()):
        raise SystemExit("the loaded models do not reproduce the saved vectors; masking results would compare other models")

    out = private_dir_of(private_root) / f"fold{k}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "query_chunks.npy", query_chunks)
    for condition in CONDITIONS:
        for name in MODELS:
            target = out / f"{name}_{condition}.npy"
            if target.is_file() and np.load(target).shape[0] == len(query_chunks):
                continue
            if name == "frozen":
                with model.disable_adapter():
                    vec = encode_all(model, tokenizer, texts[condition], "cuda", MAX_LENGTH)
            else:
                vec = encode_all(model, tokenizer, texts[condition], "cuda", MAX_LENGTH)
            np.save(target, vec.astype(np.float32))
            print(f"  {name} {condition} encoded  {(time.time() - started) / 60:.1f} min", flush=True)
    contract = {"fold": k, "tag": TAG, "model": f"{MODEL_ID}@{MODEL_REVISION}", "adapter_sha256": adapter_sha,
                "max_length": MAX_LENGTH, "query_songs": int(len(query_songs)), "query_chunks": int(len(query_chunks)),
                "words_in_query_chunks": int(word_total), "masked_words": counts, "unmatched_random_chunks": unmatched,
                "common_k": COMMON_K, "check_against_saved_vectors": check, "corpus_content_sha256": V3_CONTENT_SHA256,
                "minutes": round((time.time() - started) / 60, 1),
                "warning": "private; vectors of copyrighted lyric text. Never commit."}
    (out / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"  fold {k} stage 1 done in {contract['minutes']} min", flush=True)
    return 0


# ------------------------------------------------------------------ stage 2
def split_leave_group_out(profile_vec, query_vec, label_index, group_ids, weights, label_count, queries):
    """The protocol's dense scorer with label profiles from profile_vec and queries from query_vec."""
    sums = np.zeros((label_count, profile_vec.shape[1]))
    np.add.at(sums, label_index, profile_vec * weights[:, None])
    norms = np.linalg.norm(sums, axis=1)
    if np.any(norms <= 0):
        raise RuntimeError("an empty label profile")
    scores = query_vec[queries] @ (sums / norms[:, None]).T
    members_by_group: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids.tolist()):
        members_by_group[int(group)].append(index)
    for row, query in enumerate(queries.tolist()):
        members = members_by_group[int(group_ids[query])]
        for label in {int(label_index[m]) for m in members}:
            own = [m for m in members if int(label_index[m]) == label]
            leave = sums[label] - np.sum(profile_vec[own] * weights[own, None], axis=0)
            norm = float(np.linalg.norm(leave))
            if norm <= 1e-12:
                raise RuntimeError("a held-out group empties a label profile")
            scores[row, label] = float(query_vec[query] @ leave) / norm
    return scores


def stage_two(private_root: Path, out_dir: Path, folds: list[int]) -> int:
    started = time.time()
    d = setup(private_root)
    li, gi, w, L = d["label_index"], d["group_ids"], d["weights"], d["label_count"]
    n_songs = len(d["songs"])
    private = private_root / "work" / "private-identity-encoder-v3"
    import jieba
    jieba.setLogLevel(60)
    common = common_words(d["songs"], d["documents"])
    per_fold = {}
    for k in folds:
        store = private_dir_of(private_root) / f"fold{k}"
        contract = json.loads((store / "contract.json").read_text(encoding="utf-8"))
        if contract["corpus_content_sha256"] != V3_CONTENT_SHA256 or any(c["median_cosine"] < 0.999 for c in contract["check_against_saved_vectors"].values()):
            raise SystemExit(f"fold {k}: stage 1 store is from another build or failed its check")
        recorded = json.loads((out_dir / f"identity_encoder_analysis_fold{k}{TAG}.json").read_text(encoding="utf-8"))
        query_chunks = np.load(store / "query_chunks.npy")
        scopes = fold_scopes(d, k)
        expected_chunks = np.flatnonzero(np.isin(d["chunk_song"], np.union1d(scopes["test_fold"], scopes["unseen_labels"])))
        if not np.array_equal(query_chunks, expected_chunks):
            raise SystemExit(f"fold {k}: stage 1 encoded other chunks than this fold's queries")
        # rebuild each chunk's mask counts and check them against stage 1's totals
        per_chunk = mask_counts(d, query_chunks, common, "<mask>")
        rebuilt = {}
        for category in CATEGORIES:
            rebuilt[f"{category}_masked"] = int(per_chunk[category][:, 0].sum())
            rebuilt[f"{category}_random"] = int(per_chunk[category][:, 1].sum())
        if rebuilt != contract["masked_words"]:
            raise SystemExit(f"fold {k}: rebuilt mask counts {rebuilt} differ from stage 1's {contract['masked_words']}")
        matched_song = {}
        for category in CATEGORIES:
            ok = np.ones(n_songs, dtype=bool)
            short = per_chunk[category][:, 1] < per_chunk[category][:, 0]
            ok[np.unique(d["chunk_song"][query_chunks[short]])] = False
            matched_song[category] = ok
        train = (d["fold"] != k) & ~d["held"][li]
        full = {"tuned": np.load(private / f"fine_tuned_chunk_vectors_fold{k}{TAG}.npy").astype(np.float64),
                "frozen": np.load(private / f"frozen_masked_chunk_vectors_fold{k}.npy").astype(np.float64)}
        print(f"== fold {k}", flush=True)

        rr = {scope: {} for scope in scopes}
        systems = {scope: {} for scope in scopes}
        checks = {}
        for name in MODELS:
            profile_raw = song_level(full[name], d["chunk_song"], n_songs)
            mean, matrix, _ = fit_transform("within_author_whitening", profile_raw[train], li[train], w[train],
                                            np.random.default_rng(SEED + k))
            profile = unit_rows((profile_raw - mean) @ matrix.T)
            for scope, queries in scopes.items():
                reference = dense_leave_group_out(profile, li, gi, w, L, queries)
                split = split_leave_group_out(profile, profile, li, gi, w, L, queries)
                gap = float(np.abs(reference - split).max())
                if gap > 1e-9:
                    raise SystemExit(f"fold {k}: the split scorer differs from the protocol's by {gap:.2e}")
                ranks = ranks_of(reference, li[queries])
                systems[scope][f"{name}_none"] = round(float(np.mean(1.0 / ranks)), 4)
                arr = np.zeros(n_songs)
                arr[queries] = 1.0 / ranks
                rr[scope][f"{name}_none"] = arr
                theirs = "fine_tuned_within_author_whitening" if name == "tuned" else "frozen_masked_within_author_whitening"
                value = recorded["systems"][scope][theirs]["mrr"]
                checks[f"{scope}/{name}"] = {"recorded": value, "recomputed": systems[scope][f"{name}_none"],
                                             "gap": round(abs(systems[scope][f"{name}_none"] - value), 4)}
            for condition in CONDITIONS:
                chunks = full[name].copy()
                chunks[query_chunks] = np.load(store / f"{name}_{condition}.npy").astype(np.float64)
                query = unit_rows((song_level(chunks, d["chunk_song"], n_songs) - mean) @ matrix.T)
                for scope, queries in scopes.items():
                    ranks = ranks_of(split_leave_group_out(profile, query, li, gi, w, L, queries), li[queries])
                    systems[scope][f"{name}_{condition}"] = round(float(np.mean(1.0 / ranks)), 4)
                    arr = np.zeros(n_songs)
                    arr[queries] = 1.0 / ranks
                    rr[scope][f"{name}_{condition}"] = arr
        if any(c["gap"] > CHECK_GAP for c in checks.values()):
            raise SystemExit(f"fold {k}: unmasked MRRs do not reproduce the recorded analysis: {checks}")

        contrasts = {}
        for scope, queries in scopes.items():
            mask = np.zeros(n_songs, dtype=bool)
            mask[queries] = True
            derived = {"gain_none": rr[scope]["tuned_none"] - rr[scope]["frozen_none"], "zero": np.zeros(n_songs)}
            pairs = [("tuned_none", "frozen_none")]
            for category in CATEGORIES:
                derived[f"gain_{category}_masked"] = rr[scope][f"tuned_{category}_masked"] - rr[scope][f"frozen_{category}_masked"]
                derived[f"gain_{category}_random"] = rr[scope][f"tuned_{category}_random"] - rr[scope][f"frozen_{category}_random"]
                pairs += [(f"gain_{category}_random", f"gain_{category}_masked"),
                          (f"frozen_{category}_random", f"frozen_{category}_masked"),
                          (f"tuned_{category}_random", f"tuned_{category}_masked"),
                          (f"gain_{category}_random", "zero"),
                          (f"gain_{category}_masked", "zero")]
            contrasts[scope] = paired_group_bootstrap({**rr[scope], **derived}, w, gi, mask, pairs)
            print(f"  {scope:13s} " + "  ".join(f"{n}={v:.4f}" for n, v in systems[scope].items()), flush=True)
            for c in contrasts[scope]:
                print(f"    {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)
        matched_contrasts = {scope: {} for scope in scopes}
        matched_sizes = {scope: {} for scope in scopes}
        for scope, queries in scopes.items():
            for category in CATEGORIES:
                mask = np.zeros(n_songs, dtype=bool)
                mask[queries] = True
                mask &= matched_song[category]
                matched_sizes[scope][category] = {"songs": int(len(queries)), "matched_songs": int(mask.sum())}
                derived = {f"gain_{category}_{v}": rr[scope][f"tuned_{category}_{v}"] - rr[scope][f"frozen_{category}_{v}"]
                           for v in ("masked", "random")}
                pairs = [(f"gain_{category}_random", f"gain_{category}_masked"),
                         (f"frozen_{category}_random", f"frozen_{category}_masked"),
                         (f"tuned_{category}_random", f"tuned_{category}_masked")]
                matched_contrasts[scope][category] = paired_group_bootstrap({**rr[scope], **derived}, w, gi, mask, pairs)
                for c in matched_contrasts[scope][category]:
                    print(f"    matched {mask.sum()} songs  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
                          f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)
        per_fold[str(k)] = {"stage_one": {key: contract[key] for key in ("query_songs", "query_chunks", "words_in_query_chunks",
                                                                         "masked_words", "unmatched_random_chunks",
                                                                         "check_against_saved_vectors", "adapter_sha256")},
                            "rebuilt_mask_counts_equal_stage_one": True,
                            "systems": systems, "paired_contrasts": contrasts,
                            "count_matched_songs": {"sizes": matched_sizes, "paired_contrasts": matched_contrasts},
                            "checks_against_recorded_analysis": checks}

    def reading_of(found):
        clear = sum(1 for c in found if c["ci95"][0] > 0)
        n = len(found)
        need = 4 if n >= 5 else n
        reading = ("rests on" if clear >= need else "does not rest on" if n - clear >= need else "partly")
        return {"folds": n, "folds_clear_of_zero": clear, "reading": reading}

    verdicts = {}
    for category in CATEGORIES:
        for scope in ("test_fold", "unseen_labels"):
            pair = (f"gain_{category}_random", f"gain_{category}_masked")
            every = reading_of([c for f in per_fold.values() for c in f["paired_contrasts"][scope]
                                if (c["system"], c["minus"]) == pair])
            matched = reading_of([c for f in per_fold.values() for c in f["count_matched_songs"]["paired_contrasts"][scope][category]
                                  if (c["system"], c["minus"]) == pair])
            if every["reading"] == matched["reading"] and every["reading"] != "partly":
                stated = f"the tuned advantage {every['reading']} these words beyond their word count"
            elif every["reading"] == matched["reading"]:
                stated = "partly; read fold by fold"
            else:
                stated = "depends on the songs whose control is short; read fold by fold"
            # added after fold 0's result: dependence can only show if the advantage survives the control
            kept = [c for f in per_fold.values() for c in f["paired_contrasts"][scope]
                    if (c["system"], c["minus"]) == (f"gain_{category}_random", "zero")]
            kept_clear = sum(1 for c in kept if c["ci95"][0] > 0)
            if every["reading"] != "rests on" and kept_clear < (4 if len(kept) >= 5 else len(kept)):
                stated += (f"; uninformative about dependence: under the random control the advantage is not clear of "
                           f"zero in {len(kept) - kept_clear} of {len(kept)} folds")
            verdicts[f"{category}/{scope}"] = {"all_songs": every, "count_matched_songs": matched,
                                               "folds_where_the_advantage_survives_the_control": kept_clear,
                                               "reading": stated}
            print(f"reading {category} {scope}: {verdicts[f'{category}/{scope}']}", flush=True)

    payload = {
        "analysis": "which words the 64-anchor fine-tune's advantage over the frozen encoder rests on, by masking query words",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n_songs, "labels": L},
        "design": {"runs": f"fine-tune tag {TAG}, one launch per fold", "folds": folds, "max_length": MAX_LENGTH,
                   "categories": {"latin": "jieba tokens containing a Latin letter",
                                  "common_han": f"the {COMMON_K} Han-character words found in the most songs"},
                   "masking": "each target word replaced by the tokenizer's mask token in the query songs' label-masked "
                              "chunk text; the control replaces as many other words, chosen at random with a seed per chunk",
                   "profiles": "label profiles from the unmasked vectors; whitening fitted on the fold's unmasked training songs",
                   "dependence": "(tuned - frozen under random masking) - (tuned - frozen under masking the category)",
                   "reading_rule": "rests on the category if dependence clears zero in at least four of five folds; does not "
                                   "if it fails to in at least four; otherwise partly. Applied to all query songs and to the "
                                   "songs whose every chunk has a count-matched control; a reading is stated only when both agree",
                   "count_matched_songs": "added after fold 0's stage 1 word counts and before any masked MRR: where a chunk has "
                                          "more target words than other words, the random control masks fewer words",
                   "uninformative_flag": "added after fold 0's result: a reading other than 'rests on' is marked uninformative "
                                         "when the advantage under the random control is not clear of zero in at least four "
                                         "folds, since dependence cannot exceed what the control leaves",
                   "limits": "one mask token per word shortens the text; the control matches words, not characters"},
        "folds": per_fold,
        "verdicts": verdicts,
        "privacy": "aggregate only; vectors and texts are private",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finetune_cue_masking.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'finetune_cue_masking.json'}  ({(time.time() - started) / 60:.1f} min)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stage", choices=["1", "2"], required=True)
    parser.add_argument("--fold", type=int, help="stage 1: the fold to encode")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(FOLDS)), help="stage 2: folds to score")
    args = parser.parse_args()
    root = args.private_root.resolve()
    if args.stage == "1":
        if args.fold is None:
            raise SystemExit("--stage 1 needs --fold")
        return stage_one(root, args.fold)
    return stage_two(root, args.out_dir, args.folds)


if __name__ == "__main__":
    sys.exit(main())
