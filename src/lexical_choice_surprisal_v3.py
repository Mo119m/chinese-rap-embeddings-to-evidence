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
BANDS = 4
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
    with torch.no_grad():
        for b in range(0, len(order), BATCH):
            batch = order[b:b + BATCH]
            lines = [instances[i][4] for i in batch]
            enc = tokenizer(lines, return_offsets_mapping=True, truncation=True, max_length=MAX_LINE_TOKENS,
                            padding=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping").numpy()
            input_ids = enc["input_ids"].clone()
            targets = []   # per row: list of (token_position, original_id)
            for r, i in enumerate(batch):
                _, _, cs, ce, _ = instances[i]
                hit = [t for t in range(offsets.shape[1]) if offsets[r, t, 1] > offsets[r, t, 0]
                       and offsets[r, t, 0] >= cs and offsets[r, t, 1] <= ce]
                targets.append([(t, int(input_ids[r, t])) for t in hit])
                for t, _ in targets[-1]:
                    input_ids[r, t] = mask_id
            enc["input_ids"] = input_ids
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float()
            logp = torch.log_softmax(logits, dim=-1).cpu()
            for r, i in enumerate(batch):
                if not targets[r]:
                    continue
                s = -sum(float(logp[r, t, tok]) for t, tok in targets[r])
                surprisal[i] = s / len(targets[r])     # per masked piece, so words of different lengths compare
                pieces[i] = len(targets[r])
            if (b // BATCH) % 500 == 0:
                print(f"    {b + len(batch):,} / {len(instances):,}  {(time.time() - started) / 60:.1f} min", flush=True)

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


# ------------------------------------------------------------------ stage 2: identity by surprisal band
def stage_two(private_root: Path, out_dir: Path, songs, label_index, group_ids, weights, label_count, dense,
              documents, private_dir: Path) -> int:
    surprisal = np.load(private_dir / "surprisal.npy")
    song_index = np.load(private_dir / "song_index.npy")
    words = (private_dir / "words.txt").read_text(encoding="utf-8").split("\n")
    contract = json.loads((private_dir / "contract.json").read_text(encoding="utf-8"))
    if contract["corpus_content_sha256"] != V3_CONTENT_SHA256:
        raise SystemExit("the surprisal table was computed on another corpus build")
    ok = np.isfinite(surprisal)
    print(f"  {int(ok.sum()):,} scored word instances; median surprisal {np.nanmedian(surprisal):.2f} nats", flush=True)
    edges = np.nanquantile(surprisal, np.linspace(0, 1, BANDS + 1))
    band = np.full(len(surprisal), -1, dtype=np.int8)
    for k in range(BANDS):
        lo, hi = edges[k], edges[k + 1]
        band[ok & (surprisal >= lo) & ((surprisal < hi) | ((k == BANDS - 1) & (surprisal <= hi)))] = k

    lexicon = private_root / "work" / "lexicon_arm_everything.csv"
    import csv
    surfaces = {r["entity"].strip() for r in csv.DictReader(lexicon.open(encoding="utf-8-sig")) if r.get("entity", "").strip()}

    def docs_for(mask) -> list[str]:
        per_song: dict[int, list[str]] = defaultdict(list)
        for i in np.flatnonzero(mask):
            per_song[int(song_index[i])].append(words[i])
        return [" ".join(per_song.get(si, [])) for si in range(len(songs))]

    rr, report = {}, {}
    all_docs = docs_for(ok)
    matrix, _ = fit_words(all_docs)
    ranks, error = score_arm(dense, matrix, label_index, group_ids, label_count)
    if error:
        raise SystemExit(error)
    rr["all_scored_words"] = 1.0 / ranks
    report["all_scored_words"] = {"mrr": round(float(np.mean(rr["all_scored_words"])), 4), "tokens": int(ok.sum())}
    print(f"  all scored words: MRR {report['all_scored_words']['mrr']:.4f}", flush=True)
    for k in range(BANDS):
        mask = band == k
        docs = docs_for(mask)
        matrix, _ = fit_words(docs)
        ranks, error = score_arm(dense, matrix, label_index, group_ids, label_count)
        name = f"band_{k + 1}_of_{BANDS}"
        idx = np.flatnonzero(mask)
        band_words = [words[i] for i in idx]
        types = Counter(band_words)
        report[name] = {
            "surprisal_range_nats": [round(float(edges[k]), 3), round(float(edges[k + 1]), 3)],
            "tokens": int(mask.sum()), "types": len(types),
            "share_in_reviewed_catalogue": round(sum(1 for w in band_words if w in surfaces) / max(len(band_words), 1), 4),
            "share_other_script": round(sum(1 for w in band_words if OTHER_SCRIPT.search(w)) / max(len(band_words), 1), 4),
            "share_latin": round(sum(1 for w in band_words if not HAN.search(w) and re.search(r"[A-Za-z]", w)) / max(len(band_words), 1), 4),
            "mean_word_length": round(float(np.mean([len(w) for w in band_words])), 3) if band_words else None,
        }
        if error:
            report[name].update({"defined": False, "why": error})
            print(f"  {name}: undefined ({error[:50]})", flush=True)
            continue
        rr[name] = 1.0 / ranks
        report[name].update({"defined": True, "mrr": round(float(np.mean(rr[name])), 4),
                             "recall_at_10": round(float(np.mean(ranks <= 10)), 4)})
        print(f"  {name} [{edges[k]:.2f}, {edges[k + 1]:.2f}] nats: MRR {report[name]['mrr']:.4f}  "
              f"tokens {report[name]['tokens']:,}  types {report[name]['types']:,}  "
              f"catalogue {report[name]['share_in_reviewed_catalogue']:.1%}  other-script {report[name]['share_other_script']:.1%}",
              flush=True)
    # the low half against the high half, the cleanest statement of the question
    for name, mask in (("expected_half", ok & (surprisal < edges[BANDS // 2])), ("unexpected_half", ok & (surprisal >= edges[BANDS // 2]))):
        matrix, _ = fit_words(docs_for(mask))
        ranks, error = score_arm(dense, matrix, label_index, group_ids, label_count)
        if error:
            raise SystemExit(f"{name}: {error}")
        rr[name] = 1.0 / ranks
        report[name] = {"mrr": round(float(np.mean(rr[name])), 4), "tokens": int(mask.sum())}
        print(f"  {name}: MRR {report[name]['mrr']:.4f}", flush=True)
    pairs = [("expected_half", "unexpected_half"), ("expected_half", "all_scored_words"), ("unexpected_half", "all_scored_words")]
    pairs += [(f"band_{k + 1}_of_{BANDS}", "all_scored_words") for k in range(BANDS) if f"band_{k + 1}_of_{BANDS}" in rr]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool), pairs)
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "identity by the surprisal of each word choice under a standard-Mandarin masked language model",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count},
        "design": {"model": contract["model"], "revision": contract["revision"],
                   "surprisal": "negative log-probability of the actual word, whole word masked, averaged over its "
                                "word pieces; one lyric line is the context",
                   "bands": f"{BANDS} quantile bands of surprisal over all scored word instances",
                   "scoring": "jieba word TF-IDF rebuilt from the word instances of a band alone, the protocol unchanged",
                   "reference": "Sun, Zemel and Xu 2021: word choice as inference over candidates given meaning and context"},
        "systems": report,
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
