#!/usr/bin/env python3
"""What did the 64-anchor fine-tune learn beyond whitening? Part B: which words its advantage
depends on, tested by masking the query's words.

In every fold the GradCache fine-tune (tag _gradcache64) beats the frozen encoder after
within-author whitening. Part A (finetune_learned_identity_v3.py) describes the extra
identity; this file tests causally what it rests on. The query songs' lyrics are re-encoded
with one kind of word replaced by the tokenizer's mask token, and, as the control, with other
words replaced at random. The reference label profiles stay unmasked. If the tuned model's
advantage over the frozen model shrinks more when a kind of word is masked than when random
words are, the advantage rests on that kind of word.

Kinds of word (jieba tokens of the label-masked chunk text, the text the adapter was trained
and scored on):
  latin        tokens containing a Latin letter: English words, ad-libs, abbreviations
  common_han   the 100 Han-character words found in the most songs: the common vocabulary
               whose relative use carries identity in the word space

Controls (other words of the same chunk, chosen at random with a seed per chunk and category):
  random         as many words as the target words (word-matched)
  random_tokens  words drawn until their model tokens, each word tokenized alone, reach those
                 of the target words (token-matched)
  random_single  as many words as the target words, drawn only from words that are a single model
                 token alone (matched in words, so in mask tokens, and for one-token targets in
                 model tokens too)

For each fold k, over the fold's test songs and the 34 unseen labels' songs:
  stage 1 (GPU)  encode every query chunk under the six conditions with the fold's adapter and
                 with the adapter disabled (the frozen encoder). Checks first: 256 unmasked query
                 chunks must reproduce the saved vectors of both models (median cosine at least
                 0.999); vectors stored by an earlier run are kept only if 64 of their chunks,
                 re-encoded from the rebuilt texts, reproduce them to the same standard.
  stage 2 (CPU)  song vectors are chunk means; within-author whitening is fitted on the
                 fold's training songs (unmasked) as in analyse_identity_encoder_v3; queries
                 take their masked vectors, label profiles the unmasked ones. Checks first: the
                 rebuilt masked texts match stage 1's digests, the split scorer equals the
                 protocol's scorer, and the unmasked MRRs reproduce the recorded analysis.

Reading, fixed before the run. For a kind of word C, a control and a scope, dependence =
(tuned - frozen under the control) - (tuned - frozen under masking C), with a group bootstrap
interval.
  - the advantage rests on C beyond its amount: dependence > 0 with the interval clear of zero
    in at least four of the five folds
  - it does not: the interval includes zero, or dependence < 0, in at least four folds
  - otherwise partly, reported fold by fold
Each model's own reliance on C (control minus masking C) is reported beside it.

Additions, each dated against what had been seen:
  1. After fold 0's stage 1 word counts, before any masked MRR: in a chunk with more target words
     than other words (mostly English chunks for latin) the control cannot mask as many. In fold 0
     the word-matched latin control was 16% short in words. The same rule is also applied to the
     songs whose every chunk has a full control; a reading is stated only when all songs and those
     songs agree, otherwise "depends on the songs whose control is short".
  2. After fold 0's result: masking the 100 common Han words replaces a third of all words, and the
     tuned advantage was gone under the control as well. Dependence cannot show when the control
     alone removes the advantage, so a reading other than "rests on" is marked uninformative unless
     the advantage under the control is clear of zero in at least four folds.
  3. After folds 0-2's results with the word-matched control, before any token-matched MRR: masked
     words differ in length. Word-matched, latin masking removed about three times the characters
     of its control and common_han masking about half of its control's. The encoder reads model
     tokens, so the token-matched control was added. The overall reading is stated only when the
     word-matched and token-matched controls give the same reading; otherwise both are reported
     as disagreeing. Characters removed and encoded lengths are reported as description.
  4. After all five folds with the word- and token-matched controls, before any single-token MRR:
     every common Han word is one model token, so the token-matched control masks fewer words, and
     so places fewer mask tokens (268k against 373k per fold), than masking the common words. With
     it, the tuned advantage on test-fold songs rested on the common words in four folds; a tuned
     encoder that is merely more disturbed by many mask tokens would show the same. The single-token
     control matches words, mask tokens and model tokens at once for common_han. Reading, fixed
     now: that dependence is attributed to the common words rather than to the number of mask
     tokens only if the single-token control also gives "rests on", informative, by the same rule.
     The overall reading of addition 3 is left as it was; the single-token reading is reported
     beside it for every kind of word and scope.
Limits: a mask token starts a new tokenizer piece, so a masked text can be longer in model tokens
than the original; for chunks longer than the encoding length (512 tokens) masking changes how
much of the end the model reads.

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
STORED_CHECK_CHUNKS = 64
CATEGORIES = ("latin", "common_han")
CONTROLS = ("random", "random_tokens", "random_single")
VARIANTS = ("masked",) + CONTROLS
CONDITIONS = tuple(f"{c}_{v}" for c in CATEGORIES for v in VARIANTS)
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


def word_pieces(tokenizer, cache: dict):
    """Model tokens of a word tokenized alone, without the word-start piece the tokenizer adds
    in front of text that follows nothing."""
    def pieces(word: str) -> int:
        n = cache.get(word)
        if n is None:
            parts = tokenizer.tokenize(word)
            if parts and parts[0] == "▁":
                parts = parts[1:]
            n = cache[word] = max(len(parts), 1)
        return n
    return pieces


def masked_versions(tokens: list[str], is_target, mask_token: str, rng: np.random.Generator,
                    token_rng: np.random.Generator, single_rng: np.random.Generator, pieces):
    """One chunk's jieba tokens -> for each variant (masked, random, random_tokens, random_single)
    the masked text, the words masked, their model tokens and their characters. Each control has
    its own generator; the word-matched draw uses rng exactly as the first version of this file
    did, so its texts are unchanged."""
    words = [i for i, t in enumerate(tokens) if t.strip()]
    target = [i for i in words if is_target(tokens[i])]
    others = [i for i in words if not is_target(tokens[i])]
    chosen = rng.choice(others, size=min(len(target), len(others)), replace=False).tolist() if target and others else []
    goal = sum(pieces(tokens[i]) for i in target)
    drawn, total = [], 0
    if goal and others:
        for i in token_rng.permutation(others).tolist():
            if total >= goal:
                break
            drawn.append(i)
            total += pieces(tokens[i])
    single = [i for i in others if pieces(tokens[i]) == 1]
    singles = single_rng.choice(single, size=min(len(target), len(single)), replace=False).tolist() if target and single else []
    out = {}
    for variant, picked in (("masked", target), ("random", chosen), ("random_tokens", drawn), ("random_single", singles)):
        text = list(tokens)
        for i in picked:
            text[i] = mask_token
        out[variant] = {"text": "".join(text), "words": len(picked),
                        "pieces": sum(pieces(tokens[i]) for i in picked),
                        "characters": sum(len(tokens[i]) for i in picked)}
    return out


def build_conditions(d, query_chunks, common, tokenizer):
    """Masked texts of every query chunk under every condition, with per-chunk counts
    (words, model tokens, characters masked) and totals. Stage 1 and stage 2 both call this."""
    import jieba
    jieba.setLogLevel(60)
    predicates = {"latin": lambda t: bool(LATIN.search(t)), "common_han": lambda t: t in common}
    pieces = word_pieces(tokenizer, {})
    texts = {c: [] for c in CONDITIONS}
    per_chunk = {f"{c}/{field}": np.zeros(len(query_chunks), dtype=np.int64)
                 for c in CONDITIONS for field in ("words", "pieces", "characters")}
    words_total = characters_total = 0
    for row, j in enumerate(query_chunks):
        text = d["chunk_text"][j]
        tokens = list(jieba.cut(text))
        words_total += sum(1 for t in tokens if t.strip())
        characters_total += len(text)
        for ci, category in enumerate(CATEGORIES):
            rng = np.random.default_rng(SEED + 7919 * (ci + 1) + int(d["chunk_rows"][j]))
            token_rng = np.random.default_rng([SEED, 2, ci, int(d["chunk_rows"][j])])
            single_rng = np.random.default_rng([SEED, 3, ci, int(d["chunk_rows"][j])])
            versions = masked_versions(tokens, predicates[category], tokenizer.mask_token, rng, token_rng, single_rng, pieces)
            for variant, v in versions.items():
                condition = f"{category}_{variant}"
                texts[condition].append(v["text"])
                for field in ("words", "pieces", "characters"):
                    per_chunk[f"{condition}/{field}"][row] = v[field]
    encoded = {"original": [d["chunk_text"][j] for j in query_chunks], **texts}
    mean_tokens = {name: round(float(np.mean([min(len(ids), MAX_LENGTH) for ids in tokenizer(batch)["input_ids"]])), 1)
                   for name, batch in encoded.items()}
    at_max = int(sum(1 for ids in tokenizer(encoded["original"])["input_ids"] if len(ids) >= MAX_LENGTH))
    summary = {"words_in_query_chunks": int(words_total), "characters_in_query_chunks": int(characters_total),
               "chunks_at_max_length": at_max, "mean_encoded_tokens_per_chunk": mean_tokens,
               "masked": {c: {field: int(per_chunk[f"{c}/{field}"].sum()) for field in ("words", "pieces", "characters")}
                          for c in CONDITIONS},
               "chunks_with_a_short_control": {
                   f"{c}_{control}": int(np.sum(short_control(per_chunk, c, control))) for c in CATEGORIES for control in CONTROLS}}
    digests = {}
    for c in CONDITIONS:
        h = hashlib.sha256()
        for t in texts[c]:
            h.update(t.encode("utf-8") + b"\x00")
        digests[c] = h.hexdigest()
    return texts, per_chunk, summary, digests


def short_control(per_chunk, category: str, control: str) -> np.ndarray:
    """Chunks where the control could not reach the target words' amount: words for the word-matched
    and single-token controls, model tokens for the token-matched one."""
    field = "pieces" if control == "random_tokens" else "words"
    return per_chunk[f"{category}_{control}/{field}"] < per_chunk[f"{category}_masked/{field}"]


def fold_scopes(d, k):
    return {"test_fold": np.flatnonzero((d["fold"] == k) & ~d["held"][d["label_index"]]),
            "unseen_labels": np.flatnonzero(d["held"][d["label_index"]])}


def private_dir_of(private_root: Path) -> Path:
    return private_root / "work" / "private-finetune-cue-masking-v3"


def load_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    probe = tokenizer("x" + tokenizer.mask_token + "y")["input_ids"]
    if tokenizer.mask_token_id not in probe:
        raise SystemExit("the tokenizer does not read its mask token inside text")
    return tokenizer


# ------------------------------------------------------------------ stage 1
def stage_one(private_root: Path, k: int) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModel
    if not torch.cuda.is_available():
        raise SystemExit("stage 1 needs the CUDA device the fine-tune used")
    started = time.time()
    d = setup(private_root)
    scopes = fold_scopes(d, k)
    query_songs = np.union1d(scopes["test_fold"], scopes["unseen_labels"])
    query_chunks = np.flatnonzero(np.isin(d["chunk_song"], query_songs))
    common = common_words(d["songs"], d["documents"])
    print(f"fold {k}: {len(query_songs):,} query songs, {len(query_chunks):,} query chunks", flush=True)
    tokenizer = load_tokenizer()
    texts, _, summary, digests = build_conditions(d, query_chunks, common, tokenizer)
    print(f"  masking: {json.dumps(summary)}", flush=True)

    private = private_root / "work" / "private-identity-encoder-v3"
    adapter = private / f"lora_fold{k}{TAG}"
    adapter_sha = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(adapter.iterdir()) if p.is_file())).hexdigest()
    base = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter)).to("cuda")

    def encode(name, batch):
        if name == "frozen":
            with model.disable_adapter():
                return encode_all(model, tokenizer, batch, "cuda", MAX_LENGTH)
        return encode_all(model, tokenizer, batch, "cuda", MAX_LENGTH)

    def agreement_of(mine, stored):
        cos = np.sum(mine * stored, axis=1)
        return {"chunks": int(len(cos)), "min_cosine": round(float(cos.min()), 6), "median_cosine": round(float(np.median(cos)), 6)}

    # check: unmasked query chunks reproduce the saved vectors of both models
    saved = {"tuned": np.load(private / f"fine_tuned_chunk_vectors_fold{k}{TAG}.npy"),
             "frozen": np.load(private / f"frozen_masked_chunk_vectors_fold{k}.npy")}
    sample = query_chunks[np.random.default_rng(SEED + k).choice(len(query_chunks), size=min(CHECK_CHUNKS, len(query_chunks)), replace=False)]
    check = {name: agreement_of(encode(name, [d["chunk_text"][j] for j in sample]), saved[name][sample]) for name in MODELS}
    print(f"  check against the saved vectors: {check}", flush=True)
    if any(c["median_cosine"] < 0.999 for c in check.values()):
        raise SystemExit("the loaded models do not reproduce the saved vectors; masking results would compare other models")

    out = private_dir_of(private_root) / f"fold{k}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "query_chunks.npy", query_chunks)
    stored_checks = {}
    rows = np.random.default_rng(SEED + k + 1).choice(len(query_chunks), size=min(STORED_CHECK_CHUNKS, len(query_chunks)), replace=False)
    for condition in CONDITIONS:
        for name in MODELS:
            target = out / f"{name}_{condition}.npy"
            if target.is_file():
                stored = np.load(target)
                if stored.shape[0] != len(query_chunks):
                    raise SystemExit(f"{target.name} has {stored.shape[0]} rows for {len(query_chunks)} query chunks")
                agreement = agreement_of(encode(name, [texts[condition][r] for r in rows]), stored[rows])
                stored_checks[f"{name}_{condition}"] = agreement
                print(f"  stored {name} {condition}: {agreement}", flush=True)
                if agreement["median_cosine"] < 0.999:
                    raise SystemExit(f"stored {target.name} does not reproduce from the rebuilt texts")
                continue
            vec = encode(name, texts[condition])
            np.save(target, vec.astype(np.float32))
            print(f"  {name} {condition} encoded  {(time.time() - started) / 60:.1f} min", flush=True)
    contract = {"fold": k, "tag": TAG, "model": f"{MODEL_ID}@{MODEL_REVISION}", "adapter_sha256": adapter_sha,
                "max_length": MAX_LENGTH, "query_songs": int(len(query_songs)), "query_chunks": int(len(query_chunks)),
                "masking": summary, "text_sha256": digests, "common_k": COMMON_K,
                "check_against_saved_vectors": check, "stored_condition_checks": stored_checks,
                "corpus_content_sha256": V3_CONTENT_SHA256, "minutes": round((time.time() - started) / 60, 1),
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


def reading_of(found):
    clear = sum(1 for c in found if c["ci95"][0] > 0)
    n = len(found)
    need = 4 if n >= 5 else n
    reading = "rests on" if clear >= need else "does not rest on" if n - clear >= need else "partly"
    return {"folds": n, "folds_clear_of_zero": clear, "reading": reading}


def contrast(items, system, minus):
    found = [c for c in items if (c["system"], c["minus"]) == (system, minus)]
    if len(found) != 1:
        raise KeyError((system, minus))
    return found[0]


def stage_two(private_root: Path, out_dir: Path, folds: list[int]) -> int:
    started = time.time()
    d = setup(private_root)
    li, gi, w, L = d["label_index"], d["group_ids"], d["weights"], d["label_count"]
    n_songs = len(d["songs"])
    private = private_root / "work" / "private-identity-encoder-v3"
    tokenizer = load_tokenizer()
    common = common_words(d["songs"], d["documents"])
    per_fold = {}
    for k in folds:
        store = private_dir_of(private_root) / f"fold{k}"
        contract = json.loads((store / "contract.json").read_text(encoding="utf-8"))
        if contract["corpus_content_sha256"] != V3_CONTENT_SHA256 or any(c["median_cosine"] < 0.999 for c in contract["check_against_saved_vectors"].values()):
            raise SystemExit(f"fold {k}: stage 1 store is from another build or failed its check")
        if "text_sha256" not in contract or set(contract["text_sha256"]) != set(CONDITIONS):
            raise SystemExit(f"fold {k}: stage 1 store predates the current conditions; rerun stage 1 for this fold")
        recorded = json.loads((out_dir / f"identity_encoder_analysis_fold{k}{TAG}.json").read_text(encoding="utf-8"))
        query_chunks = np.load(store / "query_chunks.npy")
        scopes = fold_scopes(d, k)
        expected_chunks = np.flatnonzero(np.isin(d["chunk_song"], np.union1d(scopes["test_fold"], scopes["unseen_labels"])))
        if not np.array_equal(query_chunks, expected_chunks):
            raise SystemExit(f"fold {k}: stage 1 encoded other chunks than this fold's queries")
        # rebuild the masked texts and check them against the texts stage 1 encoded
        _, per_chunk, summary, digests = build_conditions(d, query_chunks, common, tokenizer)
        if digests != contract["text_sha256"] or summary != contract["masking"]:
            raise SystemExit(f"fold {k}: the rebuilt masked texts differ from the ones stage 1 encoded")
        print(f"== fold {k}  masking: {json.dumps(summary)}", flush=True)
        full_control = {}
        for category in CATEGORIES:
            for control in CONTROLS:
                ok = np.ones(n_songs, dtype=bool)
                ok[np.unique(d["chunk_song"][query_chunks[short_control(per_chunk, category, control)]])] = False
                full_control[(category, control)] = ok
        train = (d["fold"] != k) & ~d["held"][li]
        full = {"tuned": np.load(private / f"fine_tuned_chunk_vectors_fold{k}{TAG}.npy").astype(np.float64),
                "frozen": np.load(private / f"frozen_masked_chunk_vectors_fold{k}.npy").astype(np.float64)}

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

        contrasts, controlled, sizes = {}, {}, {}
        for scope, queries in scopes.items():
            in_scope = np.zeros(n_songs, dtype=bool)
            in_scope[queries] = True
            values = dict(rr[scope])
            values["gain_none"] = rr[scope]["tuned_none"] - rr[scope]["frozen_none"]
            values["zero"] = np.zeros(n_songs)
            for condition in CONDITIONS:
                values[f"gain_{condition}"] = rr[scope][f"tuned_{condition}"] - rr[scope][f"frozen_{condition}"]
            pairs = [("tuned_none", "frozen_none")]
            for category in CATEGORIES:
                pairs.append((f"gain_{category}_masked", "zero"))
                for control in CONTROLS:
                    pairs += [(f"gain_{category}_{control}", f"gain_{category}_masked"),
                              (f"frozen_{category}_{control}", f"frozen_{category}_masked"),
                              (f"tuned_{category}_{control}", f"tuned_{category}_masked"),
                              (f"gain_{category}_{control}", "zero")]
            contrasts[scope] = paired_group_bootstrap(values, w, gi, in_scope, pairs)
            controlled[scope], sizes[scope] = {}, {}
            for category in CATEGORIES:
                controlled[scope][category], sizes[scope][category] = {}, {}
                for control in CONTROLS:
                    mask = in_scope & full_control[(category, control)]
                    sizes[scope][category][control] = {"songs": int(in_scope.sum()), "songs_with_full_control": int(mask.sum())}
                    controlled[scope][category][control] = paired_group_bootstrap(
                        values, w, gi, mask, [(f"gain_{category}_{control}", f"gain_{category}_masked"),
                                              (f"frozen_{category}_{control}", f"frozen_{category}_masked"),
                                              (f"tuned_{category}_{control}", f"tuned_{category}_masked")])
            print(f"  {scope}: " + "  ".join(f"{n}={v:.4f}" for n, v in systems[scope].items()), flush=True)
            for c in contrasts[scope]:
                print(f"    {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)
        per_fold[str(k)] = {"stage_one": {key: contract[key] for key in ("query_songs", "query_chunks", "masking",
                                                                         "check_against_saved_vectors",
                                                                         "stored_condition_checks", "adapter_sha256")},
                            "rebuilt_texts_equal_stage_one": True,
                            "systems": systems, "paired_contrasts": contrasts,
                            "songs_with_full_control": {"sizes": sizes, "paired_contrasts": controlled},
                            "checks_against_recorded_analysis": checks}

    verdicts = {}
    for category in CATEGORIES:
        for scope in ("test_fold", "unseen_labels"):
            by_control = {}
            for control in CONTROLS:
                pair = (f"gain_{category}_{control}", f"gain_{category}_masked")
                every = reading_of([contrast(f["paired_contrasts"][scope], *pair) for f in per_fold.values()])
                with_full = reading_of([contrast(f["songs_with_full_control"]["paired_contrasts"][scope][category][control], *pair)
                                        for f in per_fold.values()])
                core = every["reading"] if every["reading"] == with_full["reading"] else "depends on the songs whose control is short"
                kept = [contrast(f["paired_contrasts"][scope], f"gain_{category}_{control}", "zero") for f in per_fold.values()]
                kept_clear = sum(1 for c in kept if c["ci95"][0] > 0)
                informative = core == "rests on" or kept_clear >= (4 if len(kept) >= 5 else len(kept))
                by_control[control] = {"all_songs": every, "songs_with_full_control": with_full, "reading": core,
                                       "folds_where_the_advantage_survives_the_control": kept_clear,
                                       "informative": informative}
            word, token = by_control["random"], by_control["random_tokens"]

            def phrase(v):
                text = v["reading"] if v["reading"] in ("partly", "depends on the songs whose control is short") else \
                    f"the tuned advantage {v['reading']} these words beyond their amount"
                return text if v["informative"] else f"{text}; uninformative: the advantage does not survive the control"

            if (word["reading"], word["informative"]) == (token["reading"], token["informative"]):
                stated = phrase(word) + " (word-matched and token-matched controls agree)"
            else:
                stated = f"the controls disagree: word-matched: {phrase(word)}; token-matched: {phrase(token)}"
            single = by_control["random_single"]
            single_reading = phrase(single)
            mask_count_check = ("the token-matched dependence is not explained by the number of mask tokens"
                                if (token["reading"], token["informative"]) == ("rests on", True) ==
                                (single["reading"], single["informative"]) else
                                "the token-matched dependence could come from the number of mask tokens"
                                if (token["reading"], token["informative"]) == ("rests on", True) else
                                "not applicable: the token-matched control does not give an informative 'rests on'")
            verdicts[f"{category}/{scope}"] = {"controls": by_control, "reading": stated,
                                               "single_token_control": single_reading, "mask_count_check": mask_count_check}
            print(f"reading {category} {scope}: {stated} | single-token: {single_reading} | {mask_count_check}", flush=True)

    payload = {
        "analysis": "which words the 64-anchor fine-tune's advantage over the frozen encoder rests on, by masking query words",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": n_songs, "labels": L},
        "design": {"runs": f"fine-tune tag {TAG}, one launch per fold", "folds": folds, "max_length": MAX_LENGTH,
                   "categories": {"latin": "jieba tokens containing a Latin letter",
                                  "common_han": f"the {COMMON_K} Han-character words found in the most songs"},
                   "masking": "each target word replaced by the tokenizer's mask token in the query songs' label-masked "
                              "chunk text",
                   "controls": {"random": "as many other words, chosen at random with a seed per chunk (word-matched)",
                                "random_tokens": "other words drawn at random until their model tokens, each word tokenized "
                                                 "alone, reach the target words' (token-matched)",
                                "random_single": "as many other words, drawn only from words that are one model token alone "
                                                 "(matched in words and mask tokens; for one-token targets also in model tokens)"},
                   "profiles": "label profiles from the unmasked vectors; whitening fitted on the fold's unmasked training songs",
                   "dependence": "(tuned - frozen under the control) - (tuned - frozen under masking the category)",
                   "reading_rule": "rests on the category if dependence clears zero in at least four of five folds; does not "
                                   "if it fails to in at least four; otherwise partly",
                   "additions": {
                       "songs_with_full_control": "added after fold 0's stage 1 word counts and before any masked MRR: the "
                                                  "rule is also applied to songs whose every chunk has a full control, and a "
                                                  "reading is kept only when both agree",
                       "uninformative_flag": "added after fold 0's result: a reading other than 'rests on' is uninformative "
                                             "unless the advantage under the control is clear of zero in at least four folds",
                       "token_matched_control": "added after folds 0-2's word-matched results and before any token-matched "
                                                "MRR, once characters removed showed masked words differ in length; the "
                                                "overall reading is stated only when both controls agree",
                       "single_token_control": "added after all five folds with the word- and token-matched controls, before "
                                               "any single-token MRR: masking the one-token common words places more mask "
                                               "tokens than the token-matched control; a token-matched 'rests on' is "
                                               "attributed to the words, not to the number of mask tokens, only if the "
                                               "single-token control also gives an informative 'rests on'. The overall "
                                               "reading is unchanged"},
                   "limits": "a mask token starts a new tokenizer piece, so masked texts can be longer in model tokens than "
                             "the original, and for chunks over the encoding length masking changes how much of the end "
                             "is read"},
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
