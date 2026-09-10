#!/usr/bin/env python3
"""How much of the common-word identity is regional variety rather than the individual?

The word anatomy puts a label's identity in the relative use of common words. Two
readings are open: the words are one rapper's habits, or they are the function words,
particles and pronouns of a regional variety of Chinese that many rappers share --
Cantonese 嘅/咗/唔/佢, Southwestern Mandarin 啥子/巴适/莫得, Northeastern 咋/整/嘎哈,
Beijing 您/甭/丫, Wu 侬/阿拉, Taiwan Mandarin 蛤/歹势. Both are 'common words'; only the
second is culture in the sense of a shared language variety. This file separates them
without any biography: a hand-compiled catalogue of marker words per variety, applied to
the text only.

  marker share      per label, the share of word tokens that are markers of each variety
                    and the variety it leans to (or none)
  markers only      the word space rebuilt from marker tokens alone: how far do the
                    markers by themselves identify a label
  markers removed   the word space with every marker token dropped: what the markers
                    were worth
  confusion         when the full word space names the wrong label first, how often the
                    wrong label leans to the same variety as the true one, against the
                    rate a random wrong label would -- the share of the word-space signal
                    that is regional

The catalogue is short, hand-written from standard dialect descriptions, and published
with the result; it is a first cut, not a lexicon of record, and every marker is a
plain string matched as a jieba word token.

    python src/dialect_marker_identity_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import sys
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
LEAN_SHARE = 0.004       # a label leans to a variety when at least 0.4% of its tokens are that variety's markers
LEAN_RATIO = 2.0         # and that share is at least twice the corpus-wide share of the same markers

# Function words, particles, pronouns and a few unmistakable everyday words, by variety.
# Chosen to be short, high-frequency in speech, and rarely used outside the variety; words
# that standard written Mandarin also uses freely are left out on purpose.
MARKERS = {
    "cantonese": ["嘅", "咗", "唔", "佢", "佢哋", "我哋", "你哋", "係", "喺", "嘢", "啲", "冇", "咁", "嗰", "呢度", "边度",
                  "点解", "乜", "乜嘢", "嘅话", "俾", "畀", "睇", "嚟", "翻", "嘞", "喇", "咩", "啩", "噉", "掂", "劲",
                  "好犀利", "犀利", "扑街", "仆街", "顶", "屋企", "捞", "唔使", "唔好", "而家", "仲", "仲有", "都係"],
    "southwestern_mandarin": ["啥子", "撒子", "巴适", "莫得", "莫", "要得", "晓得", "搞快", "勒是", "雄起", "瓜娃子", "瓜",
                              "耍", "耍朋友", "摆龙门阵", "龙门阵", "安逸", "扯把子", "不存在", "老子", "老汉", "幺儿",
                              "啷个", "咋个", "嘞", "嗦", "哈", "噻", "撒", "巴倒", "对头", "闷", "整", "板眼", "杀铁",
                              "撇脱", "假打", "锤子", "毛线", "宝器", "尔", "恁个", "几多"],
    "northeastern": ["咋", "咋地", "嘎哈", "干哈", "贼", "老铁", "唠嗑", "忽悠", "得瑟", "嘚瑟", "埋汰", "磕碜", "咱", "咱们",
                     "整", "整个", "老鼻子", "稀罕", "嘞", "波棱盖", "秃噜", "扬了二正", "老带劲", "带劲", "杠杠", "嘎",
                     "满血", "扒瞎", "膈应", "鼓捣", "尿性", "苞米", "嗯呐", "各应"],
    "beijing_northern": ["您", "甭", "丫", "局气", "倍儿", "得了", "姥姥", "哥们儿", "爷们儿", "老炮儿", "碎催", "颠儿",
                         "溜儿", "咱", "俺", "抠门儿", "牛逼", "大爷", "劲儿", "味儿", "事儿", "玩儿", "遛弯儿", "撒丫子",
                         "麻利", "利索", "怵", "熊", "拾掇"],
    "wu_shanghai": ["侬", "阿拉", "伊", "册那", "老卵", "交关", "覅", "呒没", "格", "辰光", "小赤佬", "赤佬", "十三点",
                    "戆", "戆大", "灵", "结棍", "巴子", "拎勿清", "邪气", "夯浜榔"],
    "min_taiwan": ["蛤", "齁", "系金欸", "金欸", "哇靠", "靠北", "靠腰", "歹势", "厚", "尬", "水", "袂", "阿嬷", "阿公",
                   "拍谢", "母汤", "母汤喔", "干嘛", "机车", "很难过", "夭寿", "呷", "呷饭", "冲三小", "三小", "郎", "肉脚",
                   "超", "超级", "蛮", "满"],
    "xiang_hunan": ["何解", "冇得", "策", "塑料", "嬲", "哦该", "满哥", "堂客", "妹坨", "伢子", "恩", "冇", "咯", "恰",
                    "恰饭", "霸蛮", "灵泛", "作古正经", "扎实", "晓得", "何事"],
}
# words that appear in more than one variety's list are kept in every list; the variety
# profile counts them for each, which only blurs the lean, never sharpens it


def build(private_root: Path, out_dir: Path) -> int:
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

    marker_of = {}
    for variety, items in MARKERS.items():
        for w in items:
            marker_of.setdefault(w, set()).add(variety)
    varieties = list(MARKERS)
    print("segmenting", flush=True)
    tokens = [segment(documents[s]) for s in songs]
    all_docs = [" ".join(t) for t in tokens]
    marker_docs = [" ".join(w for w in t if w in marker_of) for t in tokens]
    stripped_docs = [" ".join(w for w in t if w not in marker_of) for t in tokens]
    total_tokens = sum(len(t) for t in tokens)
    marker_tokens = sum(1 for t in tokens for w in t if w in marker_of)
    print(f"  {total_tokens:,} tokens, {marker_tokens:,} marker tokens ({marker_tokens / total_tokens:.2%})", flush=True)

    # per-label variety profile (weighted like the protocol) and lean
    corpus_share = {v: 0.0 for v in varieties}
    label_share = np.zeros((label_count, len(varieties)))
    label_mass = np.zeros(label_count)
    for t, label, w in zip(tokens, label_index, weights):
        counts = Counter(v for word in t for v in marker_of.get(word, ()))
        for j, v in enumerate(varieties):
            label_share[int(label), j] += w * counts.get(v, 0)
        label_mass[int(label)] += w * len(t)
    label_share = label_share / np.maximum(label_mass, 1)[:, None]
    corpus_vec = (label_share * label_mass[:, None]).sum(axis=0) / label_mass.sum()
    lean = np.full(label_count, -1)
    for l in range(label_count):
        best = int(np.argmax(label_share[l] / np.maximum(corpus_vec, 1e-9)))
        if label_share[l, best] >= LEAN_SHARE and label_share[l, best] >= LEAN_RATIO * corpus_vec[best]:
            lean[l] = best
    lean_counts = Counter(varieties[k] if k >= 0 else "none" for k in lean)
    print("  labels leaning by variety: " + ", ".join(f"{k} {v}" for k, v in lean_counts.most_common()), flush=True)

    rr, report, predicted = {}, {}, {}
    for name, docs in (("words_all", all_docs), ("markers_only", marker_docs), ("markers_removed", stripped_docs)):
        matrix, features = fit_words(docs)
        ranks, error = score_arm(dense, matrix, label_index, group_ids, label_count)
        if error:
            report[name] = {"defined": False, "why": error}
            print(f"  {name}: undefined ({error[:60]})", flush=True)
            continue
        rr[name] = 1.0 / ranks
        report[name] = {"defined": True, "features": int(matrix.shape[1]), "mrr": round(float(np.mean(rr[name])), 4),
                        "recall_at_1": round(float(np.mean(ranks <= 1)), 4), "recall_at_10": round(float(np.mean(ranks <= 10)), 4)}
        print(f"  {name:16s} MRR {report[name]['mrr']:.4f}  R@10 {report[name]['recall_at_10']:.4f}  ({matrix.shape[1]:,} features)", flush=True)
        top1 = v1.score_leave_group_out(dense, matrix, label_index, group_ids, label_count).lexical
        predicted[name] = np.argmax(top1, axis=1)

    # confusion: among wrong top-1 answers, same-variety rate against the random-wrong-label
    # rate -- for the full word space and, so the markers themselves cannot be what drives
    # it, for the space with every marker token removed
    def confusion_of(pred: np.ndarray) -> dict:
        wrong = np.flatnonzero(pred != label_index)
        leaning = [q for q in wrong if lean[label_index[q]] >= 0]
        same = [lean[pred[q]] == lean[label_index[q]] for q in leaning]
        expected_terms = []
        for q in leaning:
            others = np.delete(np.arange(label_count), label_index[q])
            expected_terms.append(float(np.mean(lean[others] == lean[label_index[q]])))
        observed = float(np.mean(same)) if same else float("nan")
        expected = float(np.mean(expected_terms)) if expected_terms else float("nan")
        return {"wrong_top1_queries": int(len(wrong)), "wrong_top1_queries_with_a_leaning_true_label": len(leaning),
                "same_variety_share_observed": round(observed, 4), "same_variety_share_if_random": round(expected, 4),
                "enrichment": round(observed / expected, 2) if expected and expected > 0 else None}
    confusion = {name: confusion_of(pred) for name, pred in predicted.items()}
    for name, c in confusion.items():
        print(f"  {name}: wrong top-1 answers sharing the true label's variety {c['same_variety_share_observed']:.1%} "
              f"observed vs {c['same_variety_share_if_random']:.1%} if random (x{c['enrichment']})", flush=True)

    # is the enrichment carried by a few labels? per leaning label (unnamed), the same-variety
    # share of its wrong answers, and the enrichment with each label left out in turn
    pred = predicted["markers_removed"]
    wrong = np.flatnonzero(pred != label_index)
    per_label = defaultdict(lambda: [0, 0])
    for q in wrong:
        l = int(label_index[q])
        if lean[l] < 0:
            continue
        per_label[l][0] += 1
        per_label[l][1] += int(lean[pred[q]] == lean[l])
    rows_out = sorted(({"variety": varieties[lean[l]], "wrong_answers": n, "same_variety": s,
                        "share": round(s / n, 3) if n else None} for l, (n, s) in per_label.items()),
                      key=lambda r: -r["wrong_answers"])
    leave_one_out = []
    for l in per_label:
        keep = [q for q in wrong if lean[label_index[q]] >= 0 and int(label_index[q]) != l]
        same = [lean[pred[q]] == lean[label_index[q]] for q in keep]
        exp = []
        for q in keep:
            others = np.delete(np.arange(label_count), label_index[q])
            exp.append(float(np.mean(lean[others] == lean[label_index[q]])))
        if same and exp and np.mean(exp) > 0:
            leave_one_out.append(round(float(np.mean(same)) / float(np.mean(exp)), 2))
    labels_with_share_above_random = sum(1 for r in rows_out if r["share"] is not None and r["wrong_answers"] >= 5
                                         and r["share"] > confusion["markers_removed"]["same_variety_share_if_random"])
    labels_with_5_or_more_wrong = sum(1 for r in rows_out if r["wrong_answers"] >= 5)
    per_label_report = {"leaning_labels_with_wrong_answers": len(rows_out),
                        "labels_with_at_least_5_wrong_answers": labels_with_5_or_more_wrong,
                        "of_which_same_variety_share_above_random": labels_with_share_above_random,
                        "same_variety_share_by_label": rows_out,
                        "enrichment_leave_one_label_out_min_max": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None}
    print(f"  per label (markers removed): {labels_with_share_above_random} of {labels_with_5_or_more_wrong} leaning labels with >=5 "
          f"wrong answers sit above the random share; leave-one-label-out enrichment "
          f"{per_label_report['enrichment_leave_one_label_out_min_max']}", flush=True)
    confusion["per_label_markers_removed"] = per_label_report

    pairs = [("markers_removed", "words_all"), ("markers_only", "words_all")]
    contrasts = paired_group_bootstrap(rr, weights, group_ids, np.ones(len(songs), dtype=bool),
                                       [(a, b) for a, b in pairs if a in rr and b in rr])
    for c in contrasts:
        print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "regional variety markers and the common-word identity signal",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count, "groups": len(order),
                   "tokens": total_tokens, "marker_tokens": marker_tokens},
        "catalogue": {v: sorted(items) for v, items in MARKERS.items()},
        "catalogue_note": "hand-compiled function words, particles, pronouns and unmistakable everyday words per variety; "
                          "a first cut matched as jieba word tokens, not a lexicon of record",
        "lean": {"rule": f"a label leans to the variety whose marker share is highest relative to the corpus share, if that "
                         f"share is at least {LEAN_SHARE:.1%} of its tokens and at least {LEAN_RATIO:g}x the corpus share",
                 "labels_by_variety": dict(lean_counts),
                 "corpus_marker_share_by_variety": {v: round(float(corpus_vec[j]), 5) for j, v in enumerate(varieties)}},
        "systems": report,
        "confusion": confusion,
        "paired_contrasts": {"design": "2000 replicates, seed 20260825, leakage groups resampled with replacement, "
                                       "each (group, label) component weighted one", "contrasts": contrasts},
        "privacy": "aggregate only; no label is named",
    }
    (out_dir / "dialect_marker_identity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'dialect_marker_identity.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
