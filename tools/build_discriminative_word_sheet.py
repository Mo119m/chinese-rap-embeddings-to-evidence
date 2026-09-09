#!/usr/bin/env python3
"""A review sheet of each label's most discriminative words, for a human to categorise.

The word experiments say that a label's identity lives in the relative use of common
words, content words and English tokens most of all. A statistic cannot say whether those
words are a rapper's own habit or the subject the rapper keeps returning to. A person who
knows the scene can. This tool writes, for the labels with the most songs, the twenty
words whose weighted TF-IDF use in the label's songs stands furthest above the corpus --
Burrows-style keyness: the label mean minus the mean over labels, divided by the standard
deviation over labels -- as a private HTML sheet where every word takes one category.

Categories: ad-lib or catchphrase, dialect or personal spelling, personal habitual word,
topic word, person or place name, transcription or formatting trace, uncertain. Answers
persist in the browser and export to JSON; nothing leaves the reviewer's machine until the
export is sent back. The sheet carries single words and counts, never a line of lyrics.
It is private: the output goes to the desktop task folder and is never committed.

    python tools/build_discriminative_word_sheet.py --private-root <ni-k> --out <dir>
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
from build_downstream_retrieval_v2 import (  # noqa: E402
    MINIMUM_SONGS_PER_LABEL,
    build_songs,
    load,
)
from common_word_curve_v2 import fit_unigrams  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from word_identity_anatomy_v2 import POS_GROUPS, pos_group, segment  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

LABELS_SHOWN = 30
WORDS_PER_LABEL = 20
CATEGORIES = [
    ("adlib", "口头禅 / ad-lib"),
    ("dialect", "方言或个人写法"),
    ("habit", "个人常用词（不是主题）"),
    ("topic", "主题词（他常唱的东西）"),
    ("name", "人名 / 地名 / 品牌"),
    ("artefact", "转写或格式痕迹"),
    ("unsure", "不确定"),
]
POS_LABEL = {"content": "实词", "person_or_place_name": "人名地名", "function": "虚词",
             "adverb": "副词", "english": "英文", "numeral": "数词", "punctuation_or_symbol": "标点"}


def build(private_root: Path, out: Path) -> int:
    import jieba
    import jieba.posseg as posseg
    jieba.setLogLevel(60)
    rows, vectors, _ = load(private_root)
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
    groups = build_groups(songs, components_by_song,
                          {s: normalise_document(documents[s]) for s in songs})
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])

    matrix, features = fit_unigrams([" ".join(segment(documents[s])) for s in songs])
    label_count = len(eligible)
    dense = matrix.toarray().astype(np.float64)
    presence = (dense > 0).astype(np.float64)
    label_share = np.zeros((label_count, matrix.shape[1]))
    np.add.at(label_share, label_index, presence)
    label_share /= np.bincount(label_index, minlength=label_count)[:, None]
    corpus_share = presence.mean(axis=0)

    # Neither keyness nor raw contribution. A word used by one label only saturates any
    # z-score, so keyness lists collaborators' names and one-off ad-lib spellings; raw
    # contribution to the label's own score is dominated by the words everyone uses (的,
    # 我, 你), because the profile dot product is mass before it is difference. What the
    # retrieval actually rewards is the word's contribution to the label's profile OVER
    # its contribution to a typical rival's: the mean over the label's songs of
    # x(song, word) x (profile(label, word) - mean profile(other labels, word)). A word
    # must also appear in at least 15% of the label's songs and in at least 2% of all
    # songs, so it is a habit and not a rarity.
    sums = np.zeros((label_count, matrix.shape[1]))
    mass = np.zeros(label_count)
    np.add.at(sums, label_index, dense * weights[:, None])
    np.add.at(mass, label_index, weights)
    profile = sums / np.linalg.norm(sums, axis=1, keepdims=True)
    others = (profile.sum(axis=0)[None, :] - profile) / (label_count - 1)
    advantage = profile - others
    contribution = np.zeros((label_count, matrix.shape[1]))
    np.add.at(contribution, label_index, dense * advantage[label_index])
    contribution /= np.bincount(label_index, minlength=label_count)[:, None]
    eligible_word = (label_share >= 0.15) & (corpus_share[None, :] >= 0.02) & (contribution > 0)

    counts = Counter(label_index.tolist())
    shown = [l for l, _ in sorted(counts.items(), key=lambda item: (-item[1], eligible[item[0]]))][:LABELS_SHOWN]
    entries = []
    for l in shown:
        ranked = np.argsort(-np.where(eligible_word[l], contribution[l], -1.0))[:WORDS_PER_LABEL]
        words = []
        for f in ranked:
            if not eligible_word[l, f]:
                continue
            word = features[f]
            pairs = list(posseg.cut(word))
            tag = pairs[0].flag if len(pairs) == 1 else "x"
            positive = contribution[l][contribution[l] > 0].sum()
            words.append({"word": word,
                          "score_share_percent": round(float(100 * contribution[l, f] / positive), 1),
                          "share_of_label_songs": round(float(label_share[l, f]), 2),
                          "share_of_all_songs": round(float(corpus_share[f]), 3),
                          "pos": POS_LABEL.get(pos_group(tag, word), "其他")})
        entries.append({"label": eligible[l], "songs": int(counts[l]), "words": words})

    out.mkdir(parents=True, exist_ok=True)
    (out / "discriminative_words_private.json").write_text(
        json.dumps({"labels": entries, "categories": CATEGORIES,
                    "warning": "private: single words per label; do not commit"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    radios = "".join(
        f'<label class="opt"><input type="radio" name="{{name}}" value="{code}"> {text}</label>'
        for code, text in CATEGORIES)
    blocks = []
    for e in entries:
        rows_html = []
        for i, w in enumerate(e["words"]):
            name = f"{e['label']}::{i}"
            rows_html.append(
                f'<tr data-key="{name}"><td class="w">{w["word"]}</td>'
                f'<td class="m">{w["pos"]}</td><td class="m">{w["score_share_percent"]}%</td>'
                f'<td class="m">{int(w["share_of_label_songs"]*100)}% / {round(w["share_of_all_songs"]*100,1)}%</td>'
                f'<td class="opts">{radios.replace("{name}", name)}</td></tr>')
        blocks.append(
            f'<section class="label"><h2>{e["label"]} <span class="n">{e["songs"]} 首</span></h2>'
            f'<table><thead><tr><th>词</th><th>词性</th><th>占他被认出的分数</th><th>他的歌里出现 / 全语料出现</th><th>这个词对他来说是什么</th></tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody></table></section>')
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>歌手高分辨力词标注</title>
<style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f5f2ea;color:#131820}}main{{max-width:1180px;margin:auto;padding:36px 16px 80px}}
h1{{font-size:1.8rem;margin:0 0 8px}}p{{line-height:1.5;color:#5b626a}}.label{{background:#fffefb;border:1px solid #cbc7bd;padding:18px 20px;margin:18px 0}}
h2{{margin:0 0 10px;font-size:1.25rem}}.n{{font-weight:400;color:#5b626a;font-size:.9rem}}table{{width:100%;border-collapse:collapse}}th{{text-align:left;font-size:.78rem;color:#5b626a;border-bottom:1px solid #cbc7bd;padding:6px 4px}}
td{{padding:6px 4px;border-bottom:1px solid #ece8de;vertical-align:top}}.w{{font-size:1.1rem;font-weight:700;white-space:nowrap}}.m{{color:#5b626a;font-size:.85rem;white-space:nowrap}}
.opts{{display:flex;flex-wrap:wrap;gap:6px}}.opt{{border:1px solid #cbc7bd;padding:3px 7px;font-size:.8rem;cursor:pointer;background:#fff}}.opt:has(input:checked){{background:#dff1fa;border-color:#0679b8}}
.opt input{{margin:0 3px 0 0}}.bar{{position:sticky;top:0;background:#f5f2ea;padding:10px 0;border-bottom:1px solid #cbc7bd;display:flex;gap:12px;align-items:center}}
button{{padding:8px 14px;font-weight:700;border:1px solid #0679b8;background:#0679b8;color:#fff;cursor:pointer}}#done{{color:#5b626a}}</style></head><body><main>
<h1>歌手高分辨力词标注</h1>
<p>每个歌手 20 个词，是模型把他和别的歌手分开时最依赖的词——他用得比别人多、而且用得够多的词（"占他被认出的分数"是这个词在"他比别人多的那部分得分"里的比例；"他的歌里出现 / 全语料出现"是含这个词的歌所占比例）。只列了在他至少 15% 的歌里、在全语料至少 2% 的歌里出现的词。请凭你对这个歌手的了解，判断这个词对他来说是什么。不用查资料，凭印象；拿不准选"不确定"。<br>
答案自动保存在浏览器里；全部做完点右上角导出，把下载的 json 发我。这个文件只有单个词和比例，没有歌词。</p>
<div class="bar"><button id="export">导出 JSON</button><span id="done"></span></div>
{"".join(blocks)}
</main><script>
const KEY='discriminative_word_labels_v1';let saved={{}};try{{saved=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{}}
const total=document.querySelectorAll('tr[data-key]').length;
function refresh(){{document.getElementById('done').textContent=Object.keys(saved).length+' / '+total+' 已标'}}
document.querySelectorAll('tr[data-key]').forEach(tr=>{{const k=tr.dataset.key;if(saved[k]){{const r=tr.querySelector('input[value="'+saved[k]+'"]');if(r)r.checked=true}}
tr.querySelectorAll('input').forEach(inp=>inp.addEventListener('change',()=>{{saved[k]=inp.value;try{{localStorage.setItem(KEY,JSON.stringify(saved))}}catch(e){{}};refresh()}}))}});
refresh();
document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{sheet:'discriminative_words_v1',labels:saved}},null,2)],{{type:'application/json'}});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='discriminative_word_labels.json';a.click()}};
</script></body></html>"""
    (out / "歌手高分辨力词标注.html").write_text(html, encoding="utf-8")
    print(f"{len(entries)} labels x {WORDS_PER_LABEL} words -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out.resolve())


if __name__ == "__main__":
    sys.exit(main())
