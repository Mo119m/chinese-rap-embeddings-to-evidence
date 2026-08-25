#!/usr/bin/env python3
"""Build journal-ready figures and chart-source tables for the Chinese rap paper."""

from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patches
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from PIL import Image

import build_chinese_rap_research_atlas_v1 as atlas_builder


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs" / "chinese-rap-embeddings-to-evidence-paper-v1"
FIGURE_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"
ENCODER_SUMMARY = ROOT / "outputs" / "chinese-rap-encoder-sanity-benchmark-v1" / "analysis_summary.json"
BOOTSTRAP_CSV = ROOT / "outputs" / "chinese-rap-edge-bootstrap-v1" / "stable_edge_bootstrap.csv"

ARIAL = FontProperties(fname=r"C:\Windows\Fonts\arial.ttf")
ARIAL_BOLD = FontProperties(fname=r"C:\Windows\Fonts\arialbd.ttf")
CJK = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
CJK_BOLD = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")

INK = "#182536"
MUTED = "#647487"
GRID = "#DCE3EA"
BLUE = "#2F6B9A"
BLUE_OPEN = "#AFCADD"
GOLD = "#C78922"
GOLD_OPEN = "#F1D59B"
PURPLE = "#7857A4"
TEAL = "#208A86"
ORANGE = "#D67824"
GREY = "#7B8794"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str, stage_figures: Path) -> None:
    png = stage_figures / f"{stem}.png"
    tif = stage_figures / f"{stem}.tif"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(tif, dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def add_box(ax, xy, width, height, title, detail, face, edge, title_font=ARIAL_BOLD, detail_font=ARIAL):
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + 0.018, y + height - 0.04, title, va="top", ha="left", fontproperties=title_font, fontsize=10, color=INK)
    ax.text(x + 0.018, y + height - 0.105, detail, va="top", ha="left", fontproperties=detail_font, fontsize=8.3, color=MUTED, linespacing=1.3)
    return box


def arrow(ax, start, end, color=MUTED, width=1.1):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "lw": width, "color": color, "shrinkA": 3, "shrinkB": 3})


def figure_pipeline(stage_figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.965, "From lyric chunks to interpretable, uncertainty-audited evidence", fontproperties=ARIAL_BOLD, fontsize=14, va="top")
    ax.text(0.02, 0.915, "The encoder is an intermediate representation; downstream tests create the research outputs", fontproperties=ARIAL, fontsize=9.5, color=MUTED, va="top")

    add_box(ax, (0.02, 0.38), 0.16, 0.27, "Clean corpus", "7,129 songs\n21,346 lyric chunks\n240 credit labels", "#F3F6F8", GRID)
    add_box(ax, (0.235, 0.38), 0.18, 0.27, "Representations", "BGE-M3 dense\nCharacter 2-5 gram TF-IDF\nExact-text hashes", "#EAF2F8", BLUE)
    arrow(ax, (0.18, 0.515), (0.235, 0.515), BLUE)

    add_box(ax, (0.47, 0.68), 0.23, 0.19, "Retrieval benchmark", "1,000 low-overlap queries\nDense vs lexical vs fusion", "#FFF7E8", GOLD)
    add_box(ax, (0.47, 0.405), 0.23, 0.19, "Repertoire matching", "Duplicate-weighted centroids\nShared-text exclusion\nReciprocal top-five", "#F0ECF7", PURPLE)
    add_box(ax, (0.47, 0.13), 0.23, 0.19, "Readable evidence", "Distinctive words\nWritten ending sounds\nWriting habits", "#E9F6F4", TEAL)
    arrow(ax, (0.415, 0.56), (0.47, 0.775), BLUE)
    arrow(ax, (0.415, 0.515), (0.47, 0.50), BLUE)
    arrow(ax, (0.415, 0.47), (0.47, 0.225), BLUE)

    add_box(ax, (0.755, 0.68), 0.21, 0.19, "Hybrid benchmark result", "Fusion performs best\nMRR 0.278\nnDCG@10 0.208", "#FFF7E8", GOLD)
    add_box(ax, (0.755, 0.405), 0.21, 0.19, "Repeatable match core", "250 song bootstraps\n16 of 86 matches at >=0.50\n0 matches at >=0.80", "#F0ECF7", PURPLE)
    add_box(ax, (0.755, 0.13), 0.21, 0.19, "Lyrical profiles", "156 full-evidence profiles\n11 of 16 core matches\nhave a readable signal", "#E9F6F4", TEAL)
    arrow(ax, (0.70, 0.775), (0.755, 0.775), GOLD)
    arrow(ax, (0.70, 0.50), (0.755, 0.50), PURPLE)
    arrow(ax, (0.70, 0.225), (0.755, 0.225), TEAL)

    ax.text(0.86, 0.045, "PUBLIC OUTPUT: LYRICAL FINGERPRINTS ATLAS", ha="center", va="center", fontproperties=ARIAL_BOLD, fontsize=9.2, color=INK)
    arrow(ax, (0.86, 0.13), (0.86, 0.075), INK)
    save_figure(fig, "figure1_pipeline", stage_figures)


def figure_encoder(stage_figures: Path, stage_tables: Path) -> None:
    summary = json.loads(ENCODER_SUMMARY.read_text(encoding="utf-8"))
    models = summary["models"]
    metrics = [("mrr", "MRR"), ("recall_at_1", "Recall@1"), ("recall_at_5", "Recall@5"), ("recall_at_10", "Recall@10"), ("ndcg_at_10", "nDCG@10")]
    names = [item["model"].replace("character 2-5 gram", "Character 2-5 gram").replace("equal-weight", "Equal-weight") for item in models]
    values = np.array([[float(item[key]) for key, _ in metrics] for item in models])

    with (stage_tables / "table2_encoder_benchmark.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system"] + [label for _, label in metrics] + ["queries", "task_boundary"])
        for name, row, original in zip(names, values, models):
            writer.writerow([name] + [f"{x:.6f}" for x in row] + [original["queries"], summary["claim_boundary"]])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(metrics))
    offsets = [-0.24, 0.0, 0.24]
    colors = [GREY, BLUE, GOLD]
    hatches = ["//", "", ""]
    for i, (name, color, hatch, offset) in enumerate(zip(names, colors, hatches, offsets)):
        bars = ax.barh(y + offset, values[i], height=0.20, label=name, color=color, edgecolor=INK, linewidth=0.55, hatch=hatch)
        for bar, value in zip(bars, values[i]):
            ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="left", fontsize=7.8, color=INK, fontproperties=ARIAL)
    ax.set_yticks(y, [label for _, label in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.41)
    ax.set_xlabel("Score")
    ax.set_title("Corpus-internal encoder sanity benchmark", loc="left", pad=18)
    ax.text(0, 1.02, "1,000 low-character-overlap same-song queries; higher is better", transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.82, bottom=0.25)
    save_figure(fig, "figure2_encoder_benchmark", stage_figures)


def read_bootstrap_rows():
    with BOOTSTRAP_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def figure_bootstrap(stage_figures: Path, stage_tables: Path) -> None:
    rows = read_bootstrap_rows()
    probs = np.array(sorted(float(row["two_representation_edge_probability"]) for row in rows))
    with (stage_tables / "table_s1_edge_bootstrap.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["edge_key", "two_representation_edge_probability", "primary_mutual_probability", "sensitivity_mutual_probability", "bootstrap_band"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    x = np.arange(1, len(probs) + 1)
    colors = np.where(probs >= 0.50, GOLD, BLUE_OPEN)
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.vlines(x, 0, probs, color="#D9E1E8", linewidth=0.8, zorder=1)
    ax.scatter(x, probs, c=colors, edgecolors=INK, linewidths=0.45, s=28, zorder=2)
    ax.axhline(0.50, color=GOLD, linewidth=1.2, linestyle="--")
    ax.axhline(float(np.median(probs)), color=GREY, linewidth=1.0, linestyle=":")
    ax.text(86.5, 0.505, "Atlas default: 0.50", ha="right", va="bottom", fontsize=8.4, color=GOLD)
    ax.text(86.5, float(np.median(probs)) - 0.012, f"Median: {np.median(probs):.3f}", ha="right", va="top", fontsize=8.4, color=GREY)
    ax.annotate("Maximum 0.712", xy=(86, probs[-1]), xytext=(70, 0.79), arrowprops={"arrowstyle": "-", "color": MUTED}, fontsize=8.5, color=INK)
    ax.set_xlim(0, 88)
    ax.set_ylim(0, 0.82)
    ax.set_xlabel("Original matches, ordered by bootstrap frequency")
    ax.set_ylabel("Selection frequency")
    ax.set_title("Song-resampling repeatability of 86 original matches", loc="left", pad=18)
    ax.text(0, 1.02, "250 within-label song bootstraps; 16 matches reach 0.50 and none reaches 0.80", transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.82, bottom=0.15)
    save_figure(fig, "figure3_bootstrap_distribution", stage_figures)


def figure_core_network(stage_figures: Path, stage_tables: Path) -> None:
    payload, _ = atlas_builder.make_payload()
    nodes = {node["id"]: node for node in payload["nodes"]}
    edges = [edge for edge in payload["edges"] if edge["bootstrapSupported"]]
    graph = nx.Graph()
    for edge in edges:
        graph.add_edge(edge["a"], edge["b"], **edge)
    with (stage_tables / "table5_repeatable_core.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label_a", "label_b", "bootstrap_frequency", "dominant_explanation", "passing_signals"])
        for edge in sorted(edges, key=lambda item: -item["bootstrap"]["probability"]):
            signals = "+".join(signal["kind"] for signal in edge["explanation"]["signals"]) or "none"
            writer.writerow([nodes[edge["a"]]["label"], nodes[edge["b"]]["label"], f"{edge['bootstrap']['probability']:.3f}", edge["explanation"]["dominantSignal"], signals])

    components = sorted(
        (graph.subgraph(component).copy() for component in nx.connected_components(graph)),
        key=lambda component: -max(float(component.edges[a, b]["bootstrap"]["probability"]) for a, b in component.edges),
    )
    fig, axes = plt.subplots(5, 2, figsize=(10.2, 10.2))
    fig.suptitle("Moderately reselected lyrical-repertoire core", x=0.045, y=0.985, ha="left", fontproperties=ARIAL_BOLD, fontsize=15, color=INK)
    fig.text(0.045, 0.955, "Ten components among source-credit labels; edge labels show frequency across 250 song bootstraps", ha="left", va="top", fontproperties=ARIAL, fontsize=9.5, color=MUTED)
    kind_color = {"language": PURPLE, "lineEnding": ORANGE, "form": TEAL, "semanticOnly": GREY}
    kind_style = {"language": "-", "lineEnding": ":", "form": "-.", "semanticOnly": "--"}
    for component_index, (ax, component) in enumerate(zip(axes.flat, components), start=1):
        count = component.number_of_nodes()
        if count == 2:
            identifiers = sorted(component.nodes, key=lambda identifier: nodes[identifier]["label"].casefold())
            pos = {identifiers[0]: np.array([-0.62, 0.0]), identifiers[1]: np.array([0.62, 0.0])}
        else:
            # Keep node labels clear of the component header and panel border.
            pos = nx.spring_layout(component, seed=240825 + component_index, iterations=300, k=1.25, scale=0.68)
        max_probability = max(float(component.edges[a, b]["bootstrap"]["probability"]) for a, b in component.edges)
        ax.text(0.02, 0.96, f"Component {component_index}", transform=ax.transAxes, ha="left", va="top", fontsize=7.8, color=MUTED, fontproperties=ARIAL_BOLD)
        ax.text(0.98, 0.96, f"max {max_probability:.0%}", transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=MUTED, fontproperties=ARIAL)
        for a, b, edge in component.edges(data=True):
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            kind = edge["explanation"]["dominantSignal"]
            probability = float(edge["bootstrap"]["probability"])
            ax.plot([x1, x2], [y1, y2], color=kind_color[kind], linewidth=1.4 + 3.2 * (probability - 0.50) / 0.212, linestyle=kind_style[kind], alpha=0.95, zorder=1)
            midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
            ax.text(midpoint[0], midpoint[1], f"{probability:.0%}", ha="center", va="center", fontsize=7.2, color=INK, fontproperties=ARIAL_BOLD, zorder=4, bbox={"boxstyle": "round,pad=0.13", "facecolor": "white", "edgecolor": GRID, "linewidth": 0.6})
        xs = [pos[node][0] for node in component.nodes]
        ys = [pos[node][1] for node in component.nodes]
        ax.scatter(xs, ys, s=48, color="white", edgecolor=BLUE, linewidth=1.5, zorder=2)
        for identifier in component.nodes:
            x, y = pos[identifier]
            label = nodes[identifier]["label"]
            vertical = 0.13 if y >= -0.25 else -0.13
            ax.text(x, y + vertical, label, ha="center", va="bottom" if vertical > 0 else "top", fontsize=8.2, fontproperties=CJK_BOLD, zorder=3, bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.94})
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.02, 1.02)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID)
            spine.set_linewidth(0.8)
    legend = [
        Line2D([0], [0], color=PURPLE, lw=2.3, linestyle=kind_style["language"], label="Shared distinctive words"),
        Line2D([0], [0], color=TEAL, lw=2.3, linestyle=kind_style["form"], label="Similar writing form"),
        Line2D([0], [0], color=ORANGE, lw=2.3, linestyle=kind_style["lineEnding"], label="Similar written endings"),
        Line2D([0], [0], color=GREY, lw=2.3, linestyle=kind_style["semanticOnly"], label="No post-hoc signal passed"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.045, right=0.98, top=0.915, bottom=0.075, hspace=0.20, wspace=0.12)
    save_figure(fig, "figure4_repeatable_core", stage_figures)


def validate(stage_figures: Path, stage_tables: Path) -> dict:
    checks = []
    for index, stem in enumerate(("figure1_pipeline", "figure2_encoder_benchmark", "figure3_bootstrap_distribution", "figure4_repeatable_core"), start=1):
        png = stage_figures / f"{stem}.png"
        tif = stage_figures / f"{stem}.tif"
        image = Image.open(png)
        checks.append({"name": f"figure_{index}_png_exists_and_large", "passed": png.is_file() and png.stat().st_size > 50_000 and image.width >= 2000 and image.height >= 1200})
        checks.append({"name": f"figure_{index}_tif_exists", "passed": tif.is_file() and tif.stat().st_size > 100_000})
    checks.extend(
        [
            {"name": "encoder_table_exists", "passed": (stage_tables / "table2_encoder_benchmark.csv").is_file()},
            {"name": "bootstrap_table_has_86_rows", "passed": len(read_bootstrap_rows()) == 86},
            {"name": "repeatable_core_has_16_edges", "passed": sum(1 for row in read_bootstrap_rows() if float(row["two_representation_edge_probability"]) >= 0.50) == 16},
        ]
    )
    return {"status": "pass" if all(check["passed"] for check in checks) else "fail", "checks": checks}


def main() -> int:
    configure()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".paper-figures-", dir=OUT_ROOT))
    stage_figures = stage / "figures"
    stage_tables = stage / "tables"
    stage_figures.mkdir()
    stage_tables.mkdir()
    try:
        figure_pipeline(stage_figures)
        figure_encoder(stage_figures, stage_tables)
        figure_bootstrap(stage_figures, stage_tables)
        figure_core_network(stage_figures, stage_tables)
        result = validate(stage_figures, stage_tables)
        if result["status"] != "pass":
            raise RuntimeError(result)
        if FIGURE_DIR.exists():
            shutil.rmtree(FIGURE_DIR)
        if TABLE_DIR.exists():
            shutil.rmtree(TABLE_DIR)
        shutil.move(str(stage_figures), str(FIGURE_DIR))
        shutil.move(str(stage_tables), str(TABLE_DIR))
        (OUT_ROOT / "figure_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps({"status": "pass", "figures": 4, "tables": 3}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
