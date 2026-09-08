#!/usr/bin/env python3
"""Sources for Figure 5: the same songs in three spaces, and how often a song's neighbours share its author.

Three retrieval numbers say where identity lives -- 0.318 semantic, 0.450 lexical, and what
whitening recovers. A figure can show it, on one condition: the same songs must appear in
every panel, so that what changes between panels is the space and nothing else.

Panels A-C are t-SNE layouts of one fixed set of songs: the twelve source-credit labels with
the most songs, up to forty songs each, at most one per leakage group so that duplicates
cannot form artificial pairs. A is BGE-M3, B the character n-gram space, C BGE-M3 after
within-author whitening fitted on songs by every *other* label, so the transform has never
seen the authors it is drawn on. t-SNE is a picture, not a measurement: cluster sizes and
the distances between clusters mean nothing, only whether same-coloured points sit
together. One seed and one perplexity are used in every panel.

Panel D is the measurement the picture gestures at, over all 7,236 songs and without any
dimensionality reduction: for k from 1 to 20, the share of a song's k nearest neighbours
(cosine, neighbours from the song's own leakage group excluded) that carry its label. The
whitened curve is fold-wise, as in the probe experiment, so no song is scored in a space
fitted on itself.

This file computes; build_chinese_rap_journal_figures_v1.py draws, from the two public
source tables written here. The public layout table carries a label and two coordinates per
point and no song identifier; the table that ties points to songs stays private.

    python src/build_identity_spaces_figure_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import (  # noqa: E402
    MINIMUM_SONGS_PER_LABEL,
    build_songs,
    load,
)
from identity_probe_v2 import FOLDS, SEED, fit_transform, unit_rows  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v2"
FIGURE_DIR = ROOT / "figures"
AUTHORS_SHOWN = 12
SONGS_PER_AUTHOR = 40
K_MAX = 20
PERPLEXITY = 30.0


def purity_curve(sims: np.ndarray, queries: np.ndarray, label_index: np.ndarray,
                 group_ids: np.ndarray, members_by_group) -> np.ndarray:
    """Per-query cumulative same-label share of the k nearest, k = 1..K_MAX."""
    sims = np.array(sims, dtype=np.float64)
    for row, query in enumerate(queries.tolist()):
        sims[row, members_by_group[int(group_ids[query])]] = -np.inf
    order = np.argsort(-sims, axis=1, kind="stable")[:, :K_MAX]
    matches = (label_index[order] == label_index[queries][:, None]).astype(np.float64)
    return np.cumsum(matches, axis=1) / np.arange(1, K_MAX + 1)


def write_csv(path: Path, header: list[str], rows) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8", newline="")


def build(private_root: Path, out_dir: Path) -> int:
    print("loading", flush=True)
    rows, vectors, _ = load(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, centroids_by_song = build_songs(
        rows, vectors)
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
    dense = v1.l2_normalize_dense(np.stack([centroids_by_song[s] for s in songs])).astype(np.float64)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))]
                          for g, l in zip(group_ids, label_index)])
    members_by_group: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids.tolist()):
        members_by_group[int(group)].append(index)
    print(f"  {len(songs):,} songs, {len(eligible)} labels, {len(order):,} groups", flush=True)
    lexical = v1.fit_tfidf([documents[s] for s in songs])

    # ------------------------------------------------------------ panel D: purity
    print("neighbour purity over every song", flush=True)
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, FOLDS, size=len(order))[group_ids]  # the probe's own folds
    curves = {"semantic": np.zeros((len(songs), K_MAX)),
              "lexical": np.zeros((len(songs), K_MAX)),
              "semantic_whitened": np.zeros((len(songs), K_MAX))}
    for k in range(FOLDS):
        queries = np.flatnonzero(fold == k)
        train = fold != k
        curves["semantic"][queries] = purity_curve(
            dense[queries] @ dense.T, queries, label_index, group_ids, members_by_group)
        curves["lexical"][queries] = purity_curve(
            (lexical[queries] @ lexical.T).toarray(), queries, label_index, group_ids,
            members_by_group)
        mean, matrix, _ = fit_transform("within_author_whitening", dense[train],
                                        label_index[train], weights[train],
                                        np.random.default_rng(SEED + k))
        whitened = unit_rows((dense - mean) @ matrix.T)
        curves["semantic_whitened"][queries] = purity_curve(
            whitened[queries] @ whitened.T, queries, label_index, group_ids, members_by_group)
        print(f"  fold {k}: {len(queries):,} queries", flush=True)
    label_share = np.bincount(label_index) / len(songs)
    chance = float(np.sum(label_share ** 2))
    purity = {name: [round(float(c[:, k].mean()), 4) for k in range(K_MAX)]
              for name, c in curves.items()}
    for name, values in purity.items():
        print(f"  {name:18s} purity@1 {values[0]:.3f}  @10 {values[9]:.3f}  @20 {values[19]:.3f}",
              flush=True)

    # ------------------------------------------------------------ panels A-C: layouts
    print("layouts of one fixed song set", flush=True)
    counts = Counter(label_index.tolist())
    shown_labels = [l for l, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))][:AUTHORS_SHOWN]
    picker = np.random.default_rng(SEED)
    subset = []
    for label in shown_labels:
        by_group: dict[int, int] = {}
        for index in np.flatnonzero(label_index == label).tolist():
            by_group.setdefault(int(group_ids[index]), index)  # lowest index per group
        candidates = sorted(by_group.values())
        chosen = picker.choice(candidates, size=min(SONGS_PER_AUTHOR, len(candidates)),
                               replace=False)
        subset.extend(sorted(chosen.tolist()))
    subset = np.asarray(subset)
    outside = ~np.isin(label_index, shown_labels)
    mean, matrix, _ = fit_transform("within_author_whitening", dense[outside],
                                    label_index[outside], weights[outside],
                                    np.random.default_rng(SEED))
    whitened_subset = unit_rows((dense[subset] - mean) @ matrix.T)
    spaces = {
        "semantic": dense[subset] @ dense[subset].T,
        "lexical": (lexical[subset] @ lexical[subset].T).toarray().astype(np.float64),
        "semantic_whitened": whitened_subset @ whitened_subset.T,
    }
    layouts = {}
    for name, sims in spaces.items():
        distance = np.clip(1.0 - sims, 0.0, None)
        np.fill_diagonal(distance, 0.0)
        layouts[name] = TSNE(n_components=2, metric="precomputed", init="random",
                             perplexity=PERPLEXITY, random_state=SEED).fit_transform(distance)
    print(f"  {len(subset)} songs from {len(shown_labels)} labels", flush=True)

    # ------------------------------------------------------------ write
    out_dir.mkdir(parents=True, exist_ok=True)
    purity_rows = [[k, name, value] for name, values in purity.items()
                   for k, value in enumerate(values, start=1)]
    purity_rows += [[k, "chance", round(chance, 4)] for k in range(1, K_MAX + 1)]
    write_csv(FIGURE_DIR / "figure_5_author_purity_source.csv", ["k", "space", "purity"],
              purity_rows)
    write_csv(FIGURE_DIR / "figure_5_layout_source.csv", ["space", "label", "x", "y"],
              [[name, eligible[label_index[index]], round(float(points[row, 0]), 4),
                round(float(points[row, 1]), 4)]
               for name, points in layouts.items() for row, index in enumerate(subset.tolist())])

    private_dir = private_root / "work" / "private-identity-spaces-figure-v2"
    private_dir.mkdir(parents=True, exist_ok=True)
    write_csv(private_dir / "tsne_coordinates.csv", ["song_id", "label", "space", "x", "y"],
              [[songs[index], eligible[label_index[index]], name,
                round(float(points[row, 0]), 4), round(float(points[row, 1]), 4)]
               for name, points in layouts.items() for row, index in enumerate(subset.tolist())])

    payload = {
        "analysis": "neighbour purity by space, and the figure 5 layout set",
        "purity": {
            "definition": ("for each song, the share of its k nearest neighbours by cosine "
                           "that carry its label; neighbours from the song's own leakage "
                           "group are excluded; averaged over all songs"),
            "chance": round(chance, 4),
            "k_max": K_MAX,
            "curves": purity,
            "whitened_space": ("fold-wise as in the probe experiment: each song's neighbours "
                               "are found in the space fitted on the other four folds"),
        },
        "layout_set": {
            "labels_shown": len(shown_labels),
            "songs_per_label_at_most": SONGS_PER_AUTHOR,
            "songs": int(len(subset)),
            "rule": ("the labels with the most songs; songs drawn with seed 20260825, at "
                     "most one per leakage group"),
            "whitening_for_panel_c": "fitted on songs by every label not shown, so the "
                                     "transform never saw the authors it is drawn on",
            "tsne": {"perplexity": PERPLEXITY, "metric": "precomputed cosine distance",
                     "init": "random", "seed": SEED},
            "caution": ("cluster sizes and the distances between clusters in a t-SNE layout "
                        "carry no meaning; the measurement is panel D"),
        },
        "sources": ["figures/figure_5_author_purity_source.csv",
                    "figures/figure_5_layout_source.csv"],
        "renderer": "src/build_chinese_rap_journal_figures_v1.py",
        "privacy": ("purity curves and label-tagged layout coordinates are public; the table "
                    "tying coordinates to song identifiers is private"),
    }
    (out_dir / "knn_author_purity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"\nwrote {out_dir / 'knn_author_purity.json'} and the two figure 5 source tables")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
