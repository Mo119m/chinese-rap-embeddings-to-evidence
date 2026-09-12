#!/usr/bin/env python3
"""Identity as word choice given context: where in the surprisal range does it sit?

The word anatomy says a label is identified by the relative use of common words. Sun,
Zemel and Xu (TACL 2021) write a speaker's word choice as inference over candidates given
the intended sense and the context, P(w | meaning, context); read for authorship, the same
form says an author is what is left of a word choice once the context has been accounted
for. This file makes that operational. A Chinese masked language model, trained on
standard written Mandarin and never on this corpus, is asked at every word of every lyric
line how probable the word the rapper actually used is with the word masked out --
whole-word masking, the model's own training regime. That number, the surprisal of the
choice given the context, is the model's estimate of how far the choice departs from
what the language expects there.

Stage 1 (GPU) records one surprisal per word instance. Stage 2 (CPU) rebuilds the word
space from the word instances in each surprisal band alone and scores every band under
the unchanged protocol, so the question becomes a curve: does identity live in the
expected choices (the common words used in their common places, at rates that differ by
rapper) or in the unexpected ones (dialect, slang, coinage, a personal turn of phrase)?
A companion table lists, per band, the share of tokens and how many of each are in the
605-surface catalogue and in the other-script set, so the bands can be read.

A global band mixes two things: which words sit in it (rare, non-standard and name-like
words are surprising wherever they occur) and how each word is used. Three controls pull
them apart. The catalogue-removed bands drop the reviewed name surfaces. The within-word
split orders every word's own instances by surprisal and halves them, so both halves hold
the same words at the same frequencies and differ only in how expected each use was; a
random split of the same blocks is its null. The within-word split inside strata of line
context repeats it among uses in equally unusual lines, so that a word's surprising uses
are not simply its uses in dialect, code-mixed or otherwise non-standard lines. Every arm is scored with unigram word TF-IDF,
because an arm keeps word instances without their neighbours.

    python src/lexical_choice_surprisal_v3.py --private-root <ni-k> [--stage 1|2|both]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import SEED  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from lexical_identity_anatomy_v2 import score_arm  # noqa: E402
from word_identity_anatomy_v2 import fit_words, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
MODEL_ID = "hfl/chinese-roberta-wwm-ext"
MODEL_REVISION = "5c58d0b8ec1d9014354d691c538661bf00bfdb44"   # pinned after the first local load, 2026-09-10
MAX_LINE_TOKENS = 64
BATCH = 384
TOKEN_BUDGET = 12288      # tokens per batch; keeps the masked-LM forward under 2 GB at any line length
BANDS = 4
CONTEXT_STRATA = 10
OTHER_SCRIPT = re.compile(r"[Ͱ-ϿЀ-ӿ֐-׿؀-ۿݐ-ݿऀ-ॿ฀-๿ༀ-࿿ᄀ-ᇿ᠀-᢯぀-ヿ㄰-㆏가-힯]")
HAN = re.compile(r"[一-鿿]")


# ------------------------------------------------------------------ stage 1: surprisal per word instance
def stage_one(private_root: Path, songs: list[str], documents: dict[str, str], private_dir: Path) -> Path:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
    if device == "cuda":
        model.half()
    revision = MODEL_REVISION
    mask_id = tokenizer.mask_token_id

    # every (song, line, word) instance with the word's character span in the line
    instances = []   # (song_index, word, char_start, char_end, line)
    for si, song in enumerate(songs):
        for line in documents[song].split("\n"):
            line = line.strip()
            if not line:
                continue
            pos = 0
            for word in segment(line):
                start = line.find(word, pos)
                if start < 0:
                    continue
                instances.append((si, word, start, start + len(word), line))
                pos = start + len(word)
    print(f"  {len(instances):,} word instances over {len(songs):,} songs", flush=True)

    surprisal = np.full(len(instances), np.nan, dtype=np.float32)
    pieces = np.zeros(len(instances), dtype=np.int16)
    started = time.time()
    order = sorted(range(len(instances)), key=lambda i: len(instances[i][4]))   # length-sorted batches
    # The first run ran out of GPU memory on the longest lines: a full [batch, tokens, vocab]
    # log-softmax is 2 GB at batch 384. Batches are now sized by a token budget, and the
    # log-softmax is taken only at the masked positions, after gathering their logits.
    batches, start = [], 0
    while start < len(order):
        longest = len(instances[order[min(start + BATCH, len(order)) - 1]][4])
        size = max(8, min(BATCH, TOKEN_BUDGET // max(min(longest + 2, MAX_LINE_TOKENS), 8)))
        batches.append(order[start:start + size])
        start += size
    done = 0
    with torch.no_grad():
        for bi, batch in enumerate(batches):
            lines = [instances[i][4] for i in batch]
            enc = tokenizer(lines, return_offsets_mapping=True, truncation=True, max_length=MAX_LINE_TOKENS,
                            padding=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping").numpy()
            input_ids = enc["input_ids"].clone()
            rows_idx, pos_idx, tok_idx, owner = [], [], [], []
            for r, i in enumerate(batch):
                _, _, cs, ce, _ = instances[i]
                hit = [t for t in range(offsets.shape[1]) if offsets[r, t, 1] > offsets[r, t, 0]
                       and offsets[r, t, 0] >= cs and offsets[r, t, 1] <= ce]
                for t in hit:
                    rows_idx.append(r)
                    pos_idx.append(t)
                    tok_idx.append(int(input_ids[r, t]))
                    owner.append(i)
                    input_ids[r, t] = mask_id
            enc["input_ids"] = input_ids
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            if rows_idx:
                picked = logits[torch.as_tensor(rows_idx, device=device), torch.as_tensor(pos_idx, device=device)].float()
                logp = torch.log_softmax(picked, dim=-1)
                chosen = logp[torch.arange(len(tok_idx), device=device), torch.as_tensor(tok_idx, device=device)].cpu().numpy()
                del logits, picked, logp
                per_instance: dict[int, list[float]] = defaultdict(list)
                for i, lp in zip(owner, chosen):
                    per_instance[i].append(-float(lp))
                for i, values in per_instance.items():
                    surprisal[i] = float(np.mean(values))     # per masked piece, so words of different lengths compare
                    pieces[i] = len(values)
            done += len(batch)
            if bi % 500 == 0:
                print(f"    {done:,} / {len(instances):,}  {(time.time() - started) / 60:.1f} min", flush=True)

    private_dir.mkdir(parents=True, exist_ok=True)
    np.save(private_dir / "surprisal.npy", surprisal)
    np.save(private_dir / "pieces.npy", pieces)
    np.save(private_dir / "song_index.npy", np.asarray([i[0] for i in instances], dtype=np.int32))
    (private_dir / "words.txt").write_text("\n".join(i[1] for i in instances), encoding="utf-8", newline="")
    (private_dir / "contract.json").write_text(json.dumps({
        "model": MODEL_ID, "revision": str(revision), "device": device, "max_line_tokens": MAX_LINE_TOKENS,
        "instances": len(instances), "scored": int(np.isfinite(surprisal).sum()),
        "corpus_content_sha256": V3_CONTENT_SHA256,
        "warning": "private; per-word surprisal over copyrighted lyric text. Never commit."}, indent=2), encoding="utf-8")
    print(f"  scored {int(np.isfinite(surprisal).sum()):,} instances in {(time.time() - started) / 60:.1f} min", flush=True)
    return private_dir


# ------------------------------------------------------------------ stage 2: identity by surprisal
def fit_unigrams(documents):
    """The protocol's word TF-IDF (fit_words) with unigrams only. An arm keeps word instances
    without their neighbours, so a bigram inside an arm would join words that were never
    adjacent in the lyric; the first pass of this stage used fit_words and had that defect."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), token_pattern=r"(?u)\S+",
                                 min_df=3, max_features=v1.TFIDF_MAX_FEATURES, sublinear_tf=True,
                                 norm="l2", dtype=np.float32)
    return vectorizer.fit_transform(documents).tocsr().astype(np.float32)


def within_word_midranks(type_id: np.ndarray, surprisal: np.ndarray, ok: np.ndarray, seed: int):
    """Where each scored instance sits among the instances of its own word, as a midrank in (0, 1).

    Instances of one word whose surprisal agrees to 0.01 nats form one block and share one
    midrank, so the repetitions of a hook line are never split between halves. A word with a
    single block has no within-word contrast and gets NaN. A block centred exactly on the
    median goes to a side by a seeded coin. The second return value is an independent seeded
    coin per block, which splits the same blocks at random: the null for the halving itself."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    t = type_id[idx]
    s = np.round(surprisal[idx].astype(np.float64), 2)
    order = np.lexsort((s, t))
    t_sorted, s_sorted = t[order], s[order]
    count_of_type = np.bincount(t_sorted)
    new_type = np.r_[True, np.diff(t_sorted) != 0]
    type_start = np.flatnonzero(new_type)
    run = np.diff(np.r_[type_start, len(t_sorted)])
    position = np.arange(len(t_sorted)) - np.repeat(type_start, run)
    new_block = new_type | np.r_[True, np.diff(s_sorted) != 0]
    block_id = np.cumsum(new_block) - 1
    block_start = np.flatnonzero(new_block)
    block_size = np.diff(np.r_[block_start, len(t_sorted)])
    mid = (position[block_start][block_id] + block_size[block_id] / 2.0) / count_of_type[t_sorted]
    blocks_of_type = np.bincount(t_sorted[block_start], minlength=len(count_of_type))
    mid[blocks_of_type[t_sorted] < 2] = np.nan
    tie_coin = rng.random(len(block_start))[block_id]
    split_coin = rng.random(len(block_start))[block_id]
    at_median = np.isclose(mid, 0.5)
    mid[at_median] = np.where(tie_coin[at_median] < 0.5, 0.49999, 0.50001)
    split_coin[~np.isfinite(mid)] = np.nan
    out_mid = np.full(len(surprisal), np.nan)
    out_coin = np.full(len(surprisal), np.nan)
    out_mid[idx[order]] = mid
    out_coin[idx[order]] = split_coin
    return out_mid, out_coin


def stage_two(private_root: Path, out_dir: Path, songs, label_index, group_ids, weights, label_count, dense,
              documents, private_dir: Path) -> int:
    import csv
    surprisal = np.load(private_dir / "surprisal.npy")
    song_index = np.load(private_dir / "song_index.npy")
    words = (private_dir / "words.txt").read_text(encoding="utf-8").split("\n")
    contract = json.loads((private_dir / "contract.json").read_text(encoding="utf-8"))
    if contract["corpus_content_sha256"] != V3_CONTENT_SHA256:
        raise SystemExit("the surprisal table was computed on another corpus build")
    if not (len(words) == len(surprisal) == len(song_index) == contract["instances"]):
        raise SystemExit("the surprisal table, the word list and the song index disagree in length")
    ok = np.isfinite(surprisal)
    print(f"  {int(ok.sum()):,} scored word instances; median surprisal {np.nanmedian(surprisal):.2f} nats", flush=True)

    vocab: dict[str, int] = {}
    type_id = np.fromiter((vocab.setdefault(w, len(vocab)) for w in words), dtype=np.int64, count=len(words))
    type_freq = np.bincount(type_id[ok], minlength=len(vocab))
    lexicon = private_root / "work" / "lexicon_arm_everything.csv"
    surfaces = {r["entity"].strip() for r in csv.DictReader(lexicon.open(encoding="utf-8-sig")) if r.get("entity", "").strip()}
    in_catalogue = np.fromiter((w in surfaces for w in words), dtype=bool, count=len(words))
    other_script = np.fromiter((bool(OTHER_SCRIPT.search(w)) for w in words), dtype=bool, count=len(words))
    latin = np.fromiter((not HAN.search(w) and bool(re.search(r"[A-Za-z]", w)) for w in words), dtype=bool, count=len(words))
    word_length = np.fromiter((len(w) for w in words), dtype=np.int32, count=len(words))

    # the line of every instance, rebuilt with stage 1's own enumeration and checked word by word
    line_id = np.full(len(words), -1, dtype=np.int64)
    rebuilt, line_count = 0, 0
    for si, song in enumerate(songs):
        for line in documents[song].split("\n"):
            line = line.strip()
            if not line:
                continue
            pos = 0
            for word in segment(line):
                start = line.find(word, pos)
                if start < 0:
                    continue
                if rebuilt >= len(words) or words[rebuilt] != word or int(song_index[rebuilt]) != si:
                    raise SystemExit(f"the rebuilt word instances diverge from stage 1 at instance {rebuilt}")
                line_id[rebuilt] = line_count
                rebuilt += 1
                pos = start + len(word)
            line_count += 1
    if rebuilt != len(words):
        raise SystemExit(f"rebuilt {rebuilt:,} word instances, stage 1 recorded {len(words):,}")
    # line context: the mean surprisal of the other scored words of the same line
    line_sum = np.bincount(line_id[ok], weights=surprisal[ok].astype(np.float64), minlength=line_count)
    line_n = np.bincount(line_id[ok], minlength=line_count)
    own = np.where(ok, surprisal, 0.0).astype(np.float64)
    others = line_n[line_id] - ok.astype(np.int64)
    context = np.full(len(words), np.nan)
    has_context = ok & (others >= 1)
    context[has_context] = (line_sum[line_id][has_context] - own[has_context]) / others[has_context]
    print(f"  rebuilt {rebuilt:,} instances over {line_count:,} lines; {int(has_context.sum()):,} have line context",
          flush=True)
    # the last word of a line: the rhyme slot, predicted with no context to its right
    line_final = np.r_[line_id[1:] != line_id[:-1], True]

    def docs_for(mask) -> list[str]:
        per_song: dict[int, list[str]] = defaultdict(list)
        for i in np.flatnonzero(mask):
            per_song[int(song_index[i])].append(words[i])
        return [" ".join(per_song.get(si, [])) for si in range(len(songs))]

    rr, systems = {}, {}

    def score(family: str, name: str, mask: np.ndarray, extra: dict | None = None, docs: list[str] | None = None) -> None:
        idx = np.flatnonzero(mask)
        info = {"tokens": int(len(idx)), "types": int(len(np.unique(type_id[idx]))),
                "median_corpus_frequency_of_its_tokens": int(np.median(type_freq[type_id[idx]])),
                "share_in_reviewed_catalogue": round(float(in_catalogue[idx].mean()), 4),
                "share_other_script": round(float(other_script[idx].mean()), 4),
                "share_latin": round(float(latin[idx].mean()), 4),
                "mean_word_length": round(float(word_length[idx].mean()), 3),
                "mean_line_context_nats": round(float(np.nanmean(context[idx])), 3),
                "share_line_final": round(float(line_final[idx].mean()), 4)}
        per_song = np.bincount(song_index[idx], minlength=len(songs))
        info["tokens_per_song"] = {"median": int(np.median(per_song)), "p10": int(np.percentile(per_song, 10)),
                                   "songs_below_20": int((per_song < 20).sum())}
        if extra:
            info.update(extra)
        ranks, error = score_arm(dense, fit_unigrams(docs if docs is not None else docs_for(mask)),
                                 label_index, group_ids, label_count)
        if error:
            info.update({"defined": False, "why": error})
            systems.setdefault(family, {})[name] = info
            print(f"  {family:36s} {name:28s} undefined ({error[:50]})", flush=True)
            return
        rr[f"{family}/{name}"] = 1.0 / ranks
        info.update({"defined": True, "mrr": round(float(np.mean(1.0 / ranks)), 4),
                     "recall_at_10": round(float(np.mean(ranks <= 10)), 4)})
        systems.setdefault(family, {})[name] = info
        print(f"  {family:36s} {name:28s} MRR {info['mrr']:.4f}  tokens {info['tokens']:>9,}  types {info['types']:>7,}  "
              f"median freq {info['median_corpus_frequency_of_its_tokens']:>7,}  catalogue {info['share_in_reviewed_catalogue']:.1%}",
              flush=True)

    # 1. global surprisal bands: the question as first asked
    edges = np.nanquantile(surprisal, np.linspace(0, 1, BANDS + 1))
    band = np.full(len(surprisal), -1, dtype=np.int8)
    for k in range(BANDS):
        lo, hi = edges[k], edges[k + 1]
        band[ok & (surprisal >= lo) & ((surprisal < hi) | ((k == BANDS - 1) & (surprisal <= hi)))] = k
    half = edges[BANDS // 2]
    for family, keep in (("global_surprisal", ok), ("global_surprisal_catalogue_removed", ok & ~in_catalogue)):
        score(family, "all_scored_words", keep)
        for k in range(BANDS):
            score(family, f"band_{k + 1}_of_{BANDS}", keep & (band == k),
                  {"surprisal_range_nats": [round(float(edges[k]), 3), round(float(edges[k + 1]), 3)]})
        score(family, "expected_half", keep & (surprisal < half))
        score(family, "unexpected_half", keep & (surprisal >= half))

    # 2. within each word: the same words at the same frequencies, split only by context
    mid, coin = within_word_midranks(type_id, surprisal, ok, SEED)
    contrastable = np.isfinite(mid)
    four = contrastable & (type_freq[type_id] >= 4)
    quartile = np.minimum(np.floor(np.nan_to_num(mid, nan=-1.0) * 4), 3)
    family = "within_word"
    score(family, "words_with_contrast", contrastable)
    score(family, "lower_half_for_its_word", contrastable & (mid < 0.5))
    score(family, "upper_half_for_its_word", contrastable & (mid > 0.5))
    score(family, "random_half_a", contrastable & (coin < 0.5))
    score(family, "random_half_b", contrastable & (coin >= 0.5))
    score(family, "words_with_four_or_more", four)
    for k in range(4):
        score(family, f"quartile_{k + 1}_for_its_word", four & (quartile == k))

    # 3. the same split inside deciles of line context: a word's more surprising uses must not
    #    simply be its uses in unusual lines (dialect, code-mixing, a non-standard register)
    #    (and at twice the resolution, to see whether the answer depends on the stratum width)
    #    and, last, inside line position too: the rhyme slot against the rest of the line
    for family, strata, by_position in (("within_word_and_line_context", CONTEXT_STRATA, False),
                                        ("within_word_and_line_context_fine", 2 * CONTEXT_STRATA, False),
                                        ("within_word_line_context_and_position", CONTEXT_STRATA, True)):
        cuts = np.quantile(context[has_context], np.linspace(0, 1, strata + 1)[1:-1])
        stratum = np.searchsorted(cuts, np.nan_to_num(context, nan=0.0), side="right")
        key = type_id * strata + stratum
        if by_position:
            key = key * 2 + line_final.astype(np.int64)
        mid_s, coin_s = within_word_midranks(key, surprisal, has_context, SEED + 1)
        contrastable_s = np.isfinite(mid_s)
        score(family, "words_with_contrast", contrastable_s, {"strata": strata})
        score(family, "lower_half_for_its_word", contrastable_s & (mid_s < 0.5), {"strata": strata})
        score(family, "upper_half_for_its_word", contrastable_s & (mid_s > 0.5), {"strata": strata})
        score(family, "random_half_a", contrastable_s & (coin_s < 0.5), {"strata": strata})
        score(family, "random_half_b", contrastable_s & (coin_s >= 0.5), {"strata": strata})

    # 5. is usage additive to vocabulary? On the strictest split (the loop's last family), each
    #    word instance also becomes a usage-tagged token, word + lower/upper half. Tagging splits
    #    counts and thins the space whatever the tags mean, so the comparison that isolates the
    #    usage information is against the same tagging by the random coin of the same blocks.
    lower_tag, upper_tag = "⟨L⟩", "⟨U⟩"

    def tagged_docs(mask, upper, keep_word):
        per_song: dict[int, list[str]] = defaultdict(list)
        for i in np.flatnonzero(mask):
            token = words[i] + (upper_tag if upper[i] else lower_tag)
            per_song[int(song_index[i])].extend((words[i], token) if keep_word else (token,))
        return [" ".join(per_song.get(si, [])) for si in range(len(songs))]

    family = "usage_tagged_words"
    by_usage, by_coin = mid_s > 0.5, coin_s >= 0.5
    score(family, "words_with_contrast", contrastable_s)
    score(family, "tagged_by_usage", contrastable_s, docs=tagged_docs(contrastable_s, by_usage, False))
    score(family, "tagged_at_random", contrastable_s, docs=tagged_docs(contrastable_s, by_coin, False))
    score(family, "words_plus_usage_tags", contrastable_s, docs=tagged_docs(contrastable_s, by_usage, True))
    score(family, "words_plus_random_tags", contrastable_s, docs=tagged_docs(contrastable_s, by_coin, True))

    g, c, w = "global_surprisal/", "global_surprisal_catalogue_removed/", "within_word/"
    pairs = [(g + "expected_half", g + "unexpected_half"), (g + "unexpected_half", g + "all_scored_words")]
    pairs += [(g + f"band_{k + 1}_of_{BANDS}", g + "all_scored_words") for k in range(BANDS)]
    pairs += [(c + "expected_half", c + "unexpected_half"), (c + "unexpected_half", c + "all_scored_words")]
    pairs += [(c + f"band_{k + 1}_of_{BANDS}", c + "all_scored_words") for k in range(BANDS)]
    pairs += [(w + "lower_half_for_its_word", w + "upper_half_for_its_word"),
              (w + "random_half_a", w + "random_half_b"),
              (w + "lower_half_for_its_word", w + "words_with_contrast"),
              (w + "upper_half_for_its_word", w + "words_with_contrast"),
              (w + "quartile_4_for_its_word", w + "quartile_1_for_its_word")]
    pairs += [(w + f"quartile_{k + 1}_for_its_word", w + "words_with_four_or_more") for k in range(4)]
    for x in ("within_word_and_line_context/", "within_word_and_line_context_fine/",
              "within_word_line_context_and_position/"):
        pairs += [(x + "lower_half_for_its_word", x + "upper_half_for_its_word"),
                  (x + "random_half_a", x + "random_half_b"),
                  (x + "lower_half_for_its_word", x + "words_with_contrast"),
                  (x + "upper_half_for_its_word", x + "words_with_contrast")]
    u = "usage_tagged_words/"
    pairs += [(u + "tagged_by_usage", u + "tagged_at_random"),
              (u + "words_plus_usage_tags", u + "words_plus_random_tags"),
              (u + "words_plus_usage_tags", u + "words_with_contrast"),
              (u + "tagged_by_usage", u + "words_with_contrast"),
              (u + "tagged_at_random", u + "words_with_contrast")]
    pairs = [(a, b) for a, b in pairs if a in rr and b in rr]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for item in contrasts:
        print(f"  {item['system']} - {item['minus']}: {item['mrr_difference']:+.4f} "
              f"[{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]", flush=True)

    # 4. who carries the within-word gap: per label (unnamed), leaving one label out at a time,
    #    and by the regional variety a label's lyrics lean to, with the lean rule and constants
    #    of dialect_marker_identity_v3 (checked against its published counts)
    from dialect_marker_identity_v3 import LEAN_RATIO, LEAN_SHARE, MARKERS
    base = "within_word_line_context_and_position/"
    lower_rr, upper_rr = rr[base + "lower_half_for_its_word"], rr[base + "upper_half_for_its_word"]
    delta = weights * (upper_rr - lower_rr)
    mass = np.bincount(label_index, weights=weights, minlength=label_count)
    queries_of = np.bincount(label_index, minlength=label_count)
    gain = np.bincount(label_index, weights=delta, minlength=label_count) / mass
    overall_gain = float(delta.sum() / weights.sum())
    leave_one_out = np.asarray([(delta.sum() - delta[label_index == l].sum()) / (weights.sum() - mass[l])
                                for l in range(label_count)])
    enough = queries_of >= 20

    marker_of: dict[str, set] = {}
    for variety, items in MARKERS.items():
        for item in items:
            marker_of.setdefault(item, set()).add(variety)
    varieties = list(MARKERS)
    tokens = [segment(documents[s]) for s in songs]
    label_share = np.zeros((label_count, len(varieties)))
    label_mass = np.zeros(label_count)
    for t, label, w_ in zip(tokens, label_index, weights):
        counts = Counter(v for token in t for v in marker_of.get(token, ()))
        for j, v in enumerate(varieties):
            label_share[int(label), j] += w_ * counts.get(v, 0)
        label_mass[int(label)] += w_ * len(t)
    label_share = label_share / np.maximum(label_mass, 1)[:, None]
    corpus_vec = (label_share * label_mass[:, None]).sum(axis=0) / label_mass.sum()
    lean = np.full(label_count, -1)
    for l in range(label_count):
        best = int(np.argmax(label_share[l] / np.maximum(corpus_vec, 1e-9)))
        if label_share[l, best] >= LEAN_SHARE and label_share[l, best] >= LEAN_RATIO * corpus_vec[best]:
            lean[l] = best
    lean_counts = Counter(varieties[k] if k >= 0 else "none" for k in lean)
    published = json.loads((out_dir / "dialect_marker_identity.json").read_text(encoding="utf-8"))["lean"]["labels_by_variety"]
    if dict(lean_counts) != published:
        raise SystemExit(f"the lean rule gives {dict(lean_counts)}, dialect_marker_identity.json has {published}")

    def masked_mrr(values, mask):
        return round(float((weights * values)[mask].sum() / weights[mask].sum()), 4)

    by_variety = {}
    for k, name in [(-1, "none")] + list(enumerate(varieties)):
        q = lean[label_index] == k
        if not q.any():
            continue
        by_variety[name] = {"labels": int((lean == k).sum()), "queries": int(q.sum()),
                            "lower_half_mrr": masked_mrr(lower_rr, q), "upper_half_mrr": masked_mrr(upper_rr, q),
                            "upper_minus_lower": round(masked_mrr(upper_rr, q) - masked_mrr(lower_rr, q), 4)}
    largest = max((k for k in range(len(varieties)) if (lean == k).any()), key=lambda k: int((lean == k).sum()))
    pair = {"upper": upper_rr, "lower": lower_rr}
    without_largest = paired_group_bootstrap(pair, weights, group_ids, lean[label_index] != largest, [("upper", "lower")])[0]
    no_lean_only = paired_group_bootstrap(pair, weights, group_ids, lean[label_index] == -1, [("upper", "lower")])[0]
    who = {
        "halves": base + "lower_half_for_its_word and upper_half_for_its_word; gains are upper minus lower",
        "overall_gain": round(overall_gain, 4),
        "labels_with_20_or_more_queries": int(enough.sum()),
        "share_of_those_labels_with_a_positive_gain": round(float((gain[enough] > 0).mean()), 4),
        "per_label_gain_quantiles_among_them": {q: round(float(np.quantile(gain[enough], p)), 4)
                                                 for q, p in (("p10", 0.1), ("p25", 0.25), ("median", 0.5),
                                                              ("p75", 0.75), ("p90", 0.9))},
        "leave_one_label_out_gain_range": [round(float(leave_one_out.min()), 4), round(float(leave_one_out.max()), 4)],
        "lean_rule": "dialect_marker_identity_v3 (markers counted in each label's lyrics; checked against its published counts)",
        "by_variety": by_variety,
        "without_the_largest_leaning_variety": {"variety": varieties[largest], **without_largest},
        "labels_leaning_to_no_variety_only": no_lean_only,
    }
    print(f"  who carries it: overall {overall_gain:+.4f}; {who['share_of_those_labels_with_a_positive_gain']:.0%} of "
          f"{int(enough.sum())} labels positive; leave-one-label-out {who['leave_one_label_out_gain_range']}; "
          f"without {varieties[largest]} {without_largest['mrr_difference']:+.4f} {without_largest['ci95']}; "
          f"no-lean labels only {no_lean_only['mrr_difference']:+.4f} {no_lean_only['ci95']}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "identity by the surprisal of each word choice under a standard-Mandarin masked language model",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count},
        "design": {
            "model": contract["model"], "revision": contract["revision"],
            "surprisal": "negative log-probability of the actual word, whole word masked, averaged over its word "
                         "pieces; one lyric line is the context; words past the 64-token line limit are unscored "
                         "and left out of every arm",
            "scoring": "word unigram TF-IDF (sublinear tf, min_df 3) rebuilt from the word instances of an arm "
                       "alone, the protocol unchanged; unigrams because an arm keeps instances without their "
                       "neighbours",
            "global_surprisal": f"{BANDS} quantile bands of surprisal over all scored word instances; the halves "
                                "split at the median",
            "catalogue_removed": f"the same bands without the tokens that are one of the {len(surfaces)} reviewed "
                                 "catalogue surfaces",
            "within_word": "each word's scored instances ordered by surprisal; instances within 0.01 nats form one "
                           "block with one midrank, so a repeated line is never split; words with a single block "
                           "are left out; halves by midrank, quartiles for words with four or more scored "
                           "instances; the random halves split the same blocks by a second seeded coin",
            "within_word_and_line_context": f"the within-word split inside {CONTEXT_STRATA} quantile strata of "
                                            "line context (the mean surprisal of the other scored words of the "
                                            "same line), so both halves of a word hold uses in equally unusual "
                                            "lines; lines are rebuilt with stage 1's enumeration and checked "
                                            "against its word list instance by instance; the _fine family repeats "
                                            f"it with {2 * CONTEXT_STRATA} strata",
            "usage_tagged_words": "on the strictest split, each instance becomes a token tagged with its half "
                                  "(tagged_by_usage) or added beside the plain word (words_plus_usage_tags); the "
                                  "same taggings by the random coin of the same blocks are the controls, since "
                                  "tagging thins the counts whatever the tags mean",
            "within_word_line_context_and_position": "the within-word, line-context split inside line position as "
                                                     "well: a line's last word (the rhyme slot, masked with no "
                                                     "context to its right) against every other position",
            "seed": SEED,
            "reference": "Sun, Zemel and Xu 2021: word choice as inference over candidates given meaning and context"},
        "systems": systems,
        "who_carries_the_within_word_gap": who,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with replacement, "
                                       "each (group, label) component weighted one", "contrasts": contrasts},
        "privacy": "aggregate only; no word is published",
    }
    (out_dir / "lexical_choice_surprisal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'lexical_choice_surprisal.json'}")
    return 0


def build(private_root: Path, out_dir: Path, stage: str) -> int:
    import jieba
    jieba.setLogLevel(60)
    print("loading corpus v3", flush=True)
    rows, vectors, _ = load_v3(private_root)
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
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs]))
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    print(f"  {len(songs):,} queries, {label_count} labels, {len(order):,} groups", flush=True)
    private_dir = private_root / "work" / "private-lexical-choice-surprisal-v3"
    if stage in ("1", "both"):
        print("stage 1: surprisal of every word choice", flush=True)
        stage_one(private_root, songs, documents, private_dir)
    if stage in ("2", "both"):
        print("stage 2: identity by surprisal band", flush=True)
        stage_two(private_root, out_dir, songs, label_index, group_ids, weights, label_count, dense, documents, private_dir)
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
