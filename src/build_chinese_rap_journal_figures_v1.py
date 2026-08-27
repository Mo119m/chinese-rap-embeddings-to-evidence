#!/usr/bin/env python3
"""Build strict journal variants of the four public Chinese-rap figures.

This builder is deliberately aggregate-only.  It reads the frozen source
tables already released with the figures and writes a compact, 6.5-inch-wide
journal canvas for each figure.  Captions, sources, and claim-boundary prose
remain in the manuscript rather than being duplicated inside the artwork.

Outputs
-------
* Descriptive PNG and TIFF filenames are retained for manuscript compatibility.
* ``fig1.tif`` ... ``fig4.tif`` are uncompressed 600-dpi RGB submission files.
* ``fig1.pdf``/``.svg`` ... ``fig4.pdf``/``.svg`` are vector alternatives.
* ``journal_figure_validation.json`` records typography and image QA.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
DPI = 600
MIN_FONT_PT = 7.0

PUBLIC_LINEAGE_PATHS = [
    "src/build_chinese_rap_downstream_figures_v1.py",
    "src/build_chinese_rap_figure_3_v1.py",
    "src/build_chinese_rap_journal_figures_v1.py",
    "results/input-audit-v1/analysis_summary.json",
    "results/retrieval-v1/analysis_summary.json",
    "results/retrieval-v1/metrics.csv",
    "results/retrieval-v1/uncertainty.csv",
    "results/ner-v1/entity_co_mentions_provisional.csv",
    "results/ner-v1/reconciliation_validation.json",
    "results/ner-v1/release_sensitivity_summary.csv",
    "results/ner-v1/source_label_entity_links_provisional.csv",
    "results/ner-v1/summary.json",
    "results/ner-v1/validation.json",
    "results/written-rhyme-v1/analysis_summary.json",
    "results/written-rhyme-v1/model_metrics.csv",
    "results/written-rhyme-v1/paired_model_deltas.csv",
    "results/written-rhyme-v1/stratified_metrics.csv",
    "methods/RESEARCH_CONTRACT.md",
]

ARIAL = Path("C:/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
CJK = Path("C:/Windows/Fonts/msyh.ttc")
CJK_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")

INK = "#202124"
MUTED = "#5F6772"
GRID = "#D7DCE2"
PAPER = "#FFFFFF"
SOFT = "#F5F6F7"
BLUE = "#0072B2"
BLUE_LIGHT = "#D9EEF8"
ORANGE = "#C75500"
ORANGE_LIGHT = "#F7DDCE"
PURPLE = "#8E5A82"
PURPLE_LIGHT = "#F0E6ED"
GRAY = "#6D737B"
LIGHT_GRAY = "#C5CAD0"

FIGURE_SPECS = {
    1: {"stem": "figure_1_research_design", "size": (6.5, 4.35)},
    2: {"stem": "figure_2_retrieval_benchmark", "size": (6.5, 3.55)},
    3: {"stem": "figure_3_cultural_reference_evidence", "size": (6.5, 5.55)},
    4: {"stem": "figure_4_written_rhyme_benchmark", "size": (6.5, 5.05)},
}

METRIC_ORDER_RETRIEVAL = ["mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10"]
METRIC_ORDER_RHYME = ["top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr"]
RHYME_MODELS = [
    "hierarchical_sgd_context",
    "hierarchical_sgd_no_source_label",
    "flat_sgd_logistic_context",
    "first_order_markov",
    "global_frequency",
]


def read_csv(name: str) -> list[dict[str, str]]:
    with (FIGURE_DIR / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_fonts() -> tuple[FontProperties, FontProperties]:
    for path in (ARIAL, ARIAL_BOLD, CJK, CJK_BOLD):
        if not path.is_file():
            raise FileNotFoundError(f"Required journal font not found: {path}")
        mpl.font_manager.fontManager.addfont(str(path))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 8.0,
            "font.weight": "normal",
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.3,
            "legend.frameon": False,
            "lines.linewidth": 1.05,
            "lines.markersize": 5.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    regular = FontProperties(family=["Arial", "Microsoft YaHei", "DejaVu Sans"])
    bold = FontProperties(family=["Arial", "Microsoft YaHei", "DejaVu Sans"], weight="bold")
    return regular, bold


def style_axis(ax: mpl.axes.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=2.6, width=0.65, color=GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.85, zorder=0)


def panel_title(ax: mpl.axes.Axes, letter: str, title: str, pad: float = 8.0) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=9.0, fontweight="bold", pad=pad)


def rounded_box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str = GRID,
    linewidth: float = 0.9,
    radius: float = 0.014,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.007,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.95,
            color=MUTED,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def integer(row: dict[str, str], key: str) -> int:
    return int(row[key])


def rounded_bbox_issue(fig: mpl.figure.Figure) -> list[dict[str, Any]]:
    """Return text objects that extend materially outside the fixed canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    issues: list[dict[str, Any]] = []
    for artist in fig.findobj(match=lambda obj: isinstance(obj, Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        tolerance = 3.0
        if (
            bbox.x0 < canvas.x0 - tolerance
            or bbox.y0 < canvas.y0 - tolerance
            or bbox.x1 > canvas.x1 + tolerance
            or bbox.y1 > canvas.y1 + tolerance
        ):
            issues.append(
                {
                    "text": artist.get_text(),
                    "bbox_pixels": [round(bbox.x0, 1), round(bbox.y0, 1), round(bbox.x1, 1), round(bbox.y1, 1)],
                }
            )
    return issues


def export_figure(fig: mpl.figure.Figure, number: int) -> dict[str, Any]:
    spec = FIGURE_SPECS[number]
    stem = spec["stem"]
    size = tuple(spec["size"])
    fig.canvas.draw()

    visible_text = [
        artist
        for artist in fig.findobj(match=lambda obj: isinstance(obj, Text))
        if artist.get_visible() and artist.get_text().strip()
    ]
    font_sizes = [float(artist.get_fontsize()) for artist in visible_text]
    minimum = min(font_sizes) if font_sizes else 0.0
    forbidden = [artist.get_text() for artist in visible_text if artist.get_text().strip().lower().startswith("figure ")]
    bbox_issues = rounded_bbox_issue(fig)
    if minimum < MIN_FONT_PT - 1e-9:
        raise AssertionError(f"Figure {number} contains {minimum:.2f} pt text; minimum is {MIN_FONT_PT:.1f} pt")
    if forbidden:
        raise AssertionError(f"Figure {number} duplicates a figure label inside the artwork: {forbidden}")
    if bbox_issues:
        raise AssertionError(f"Figure {number} has text outside the canvas: {bbox_issues}")

    png = FIGURE_DIR / f"{stem}.png"
    tiff = FIGURE_DIR / f"{stem}.tiff"
    submission_tiff = FIGURE_DIR / f"fig{number}.tif"
    submission_pdf = FIGURE_DIR / f"fig{number}.pdf"
    submission_svg = FIGURE_DIR / f"fig{number}.svg"

    fig.savefig(png, dpi=DPI, facecolor=PAPER, edgecolor="none", metadata={"Software": "Matplotlib"})
    fig.savefig(
        submission_pdf,
        facecolor=PAPER,
        edgecolor="none",
        metadata={"Title": f"Chinese rap study figure {number}", "Creator": "Matplotlib"},
    )
    fig.savefig(submission_svg, facecolor=PAPER, edgecolor="none", metadata={"Title": f"Chinese rap study figure {number}"})
    plt.close(fig)

    expected_pixels = (int(round(size[0] * DPI)), int(round(size[1] * DPI)))
    with Image.open(png) as image:
        if image.size != expected_pixels:
            raise AssertionError(f"Unexpected Figure {number} pixel size: {image.size}; expected {expected_pixels}")
        rgb = image.convert("RGB")
        rgb.save(tiff, format="TIFF", dpi=(DPI, DPI), compression="tiff_lzw")
        rgb.save(submission_tiff, format="TIFF", dpi=(DPI, DPI), compression="raw")

    return {
        "number": number,
        "print_size_inches": list(size),
        "pixels_at_600_dpi": list(expected_pixels),
        "minimum_visible_font_pt": round(minimum, 2),
        "visible_text_elements": len(visible_text),
        "font_policy": "Arial/Helvetica-style sans serif with Microsoft YaHei CJK fallback",
        "internal_figure_label_count": len(forbidden),
        "text_outside_canvas": bbox_issues,
        "files": [
            path.name
            for path in (png, tiff, submission_tiff, submission_pdf, submission_svg)
        ],
    }


def value_from_pipeline(rows: list[dict[str, str]], item: str) -> int:
    return integer(next(row for row in rows if row["item"] == item), "value")


def build_figure_1() -> mpl.figure.Figure:
    rows = read_csv("figure_1_pipeline_source.csv")
    songs = value_from_pipeline(rows, "songs")
    chunks = value_from_pipeline(rows, "canonical_chunks")
    clean = value_from_pipeline(rows, "eligible_clean_text_chunks")
    duplicate_groups = value_from_pipeline(rows, "exact_song_content_groups_spanning_songs")
    grouped_songs = value_from_pipeline(rows, "songs_in_spanning_exact_groups")

    fig = plt.figure(figsize=FIGURE_SPECS[1]["size"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(ax, 0.055, 0.815, 0.89, 0.125, SOFT, GRID, 0.9)
    ax.text(0.078, 0.902, "FROZEN CORPUS", transform=ax.transAxes, fontsize=7.4, fontweight="bold", color=MUTED, va="center")
    ax.text(
        0.078,
        0.855,
        f"{songs:,} songs   |   {chunks:,} chunks   |   {clean:,} analysis-eligible chunks",
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="center",
    )
    arrow(ax, (0.50, 0.815), (0.50, 0.772))
    rounded_box(ax, 0.055, 0.615, 0.89, 0.13, INK, INK, 0.9)
    ax.text(0.078, 0.708, "SHARED EVIDENCE CONTROLS", transform=ax.transAxes, fontsize=7.4, fontweight="bold", color=PAPER, va="center")
    ax.text(
        0.078,
        0.670,
        "Song-aware holdout  |  exact-content grouping  |  task-specific fitting boundaries",
        transform=ax.transAxes,
        fontsize=7.6,
        fontweight="bold",
        color=PAPER,
        va="center",
    )
    ax.text(
        0.078,
        0.635,
        f"{duplicate_groups:,} cross-song content groups ({grouped_songs:,} songs); no test-outcome tuning",
        transform=ax.transAxes,
        fontsize=7.3,
        color="#E1E5EA",
        va="center",
    )

    branches = [
        {
            "x": 0.045,
            "color": BLUE,
            "fill": BLUE_LIGHT,
            "title": "1  REPERTOIRE RETRIEVAL",
            "method": "BGE-M3 dense\nCharacter 2-5-gram TF-IDF\nUntuned z-score fusion",
            "output": "Held-out-song\nsource-label ranking",
        },
        {
            "x": 0.355,
            "color": ORANGE,
            "fill": ORANGE_LIGHT,
            "title": "2  CULTURAL REFERENCES",
            "method": "Lexicon/rule baseline\nContextual Chinese NER\nAgreement + BH-FDR gates",
            "output": "Typed references and\nsame-song co-mentions",
        },
        {
            "x": 0.665,
            "color": PURPLE,
            "fill": PURPLE_LIGHT,
            "title": "3  WRITTEN RHYME",
            "method": "Terminal-Han pinyin finals\nMarkov baseline\nHierarchical context model",
            "output": "Next written-ending\nfamily probabilities",
        },
    ]

    for branch in branches:
        x = branch["x"]
        width = 0.29
        arrow(ax, (0.50, 0.615), (x + width / 2, 0.565))
        rounded_box(ax, x, 0.105, width, 0.44, branch["fill"], branch["color"], 1.0)
        ax.add_patch(
            mpl.patches.Rectangle(
                (x, 0.482), width, 0.063, transform=ax.transAxes, facecolor=branch["color"], edgecolor="none", clip_on=False
            )
        )
        ax.text(x + 0.014, 0.513, branch["title"], transform=ax.transAxes, fontsize=7.2, fontweight="bold", color=PAPER, va="center")
        ax.text(x + 0.014, 0.447, "METHOD", transform=ax.transAxes, fontsize=7.1, fontweight="bold", color=MUTED, va="top")
        ax.text(x + 0.014, 0.410, branch["method"], transform=ax.transAxes, fontsize=7.3, color=INK, va="top", linespacing=1.32)
        ax.plot([x + 0.014, x + width - 0.014], [0.305, 0.305], transform=ax.transAxes, color=GRID, lw=0.75)
        ax.text(x + 0.014, 0.283, "OUTPUT", transform=ax.transAxes, fontsize=7.1, fontweight="bold", color=MUTED, va="top")
        ax.text(x + 0.014, 0.246, branch["output"], transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=INK, va="top", linespacing=1.25)

    return fig


def build_figure_2() -> mpl.figure.Figure:
    metrics = read_csv("figure_2_retrieval_benchmark_source.csv")
    deltas = read_csv("figure_2_retrieval_deltas_source.csv")
    fig = plt.figure(figsize=FIGURE_SPECS[2]["size"])
    grid = fig.add_gridspec(1, 2, left=0.115, right=0.985, bottom=0.18, top=0.76, width_ratios=[1.42, 1.0], wspace=0.32)
    ax_main = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1])

    systems = ["BGE-M3 dense (strict)", "character 2-5 gram TF-IDF (strict)", "equal-weight z-score fusion (strict)"]
    system_style = {
        systems[0]: {"color": GRAY, "marker": "s", "mfc": PAPER, "label": "BGE-M3 dense"},
        systems[1]: {"color": ORANGE, "marker": "^", "mfc": ORANGE_LIGHT, "label": "Character TF-IDF"},
        systems[2]: {"color": BLUE, "marker": "o", "mfc": BLUE, "label": "Dense + lexical fusion"},
    }
    offsets = {systems[0]: 0.19, systems[1]: 0.0, systems[2]: -0.19}
    y_base = {metric: 4 - index for index, metric in enumerate(METRIC_ORDER_RETRIEVAL)}
    metric_labels = {row["metric"]: row["metric_label"] for row in metrics}

    for system in systems:
        style = system_style[system]
        for row in (candidate for candidate in metrics if candidate["system"] == system):
            y = y_base[row["metric"]] + offsets[system]
            estimate = numeric(row, "estimate")
            lower = numeric(row, "ci95_lower")
            upper = numeric(row, "ci95_upper")
            ax_main.errorbar(
                estimate,
                y,
                xerr=[[estimate - lower], [upper - estimate]],
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markerfacecolor=style["mfc"],
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                markersize=5.3,
                elinewidth=1.0,
                capsize=2.1,
                zorder=3,
            )
            ax_main.text(upper + 0.012, y, f"{estimate:.3f}", va="center", fontsize=7.1, color=style["color"])

    handles = [
        mpl.lines.Line2D(
            [], [], color=system_style[system]["color"], marker=system_style[system]["marker"],
            markerfacecolor=system_style[system]["mfc"], linestyle="none", label=system_style[system]["label"]
        )
        for system in systems
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.98), ncol=3, columnspacing=1.2, handletextpad=0.45)

    ax_main.set_yticks([y_base[m] for m in METRIC_ORDER_RETRIEVAL], [metric_labels[m] for m in METRIC_ORDER_RETRIEVAL])
    ax_main.set_ylim(-0.55, 4.55)
    ax_main.set_xlim(0.0, 0.72)
    ax_main.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax_main.set_xlabel("Retrieval score")
    panel_title(ax_main, "A", "Model estimates")
    style_axis(ax_main, "x")

    comparisons = ["strict fusion minus strict BGE-M3", "strict fusion minus strict TF-IDF"]
    delta_style = {
        comparisons[0]: {"color": BLUE, "marker": "o", "mfc": BLUE, "label": "Fusion - BGE-M3"},
        comparisons[1]: {"color": ORANGE, "marker": "D", "mfc": PAPER, "label": "Fusion - TF-IDF"},
    }
    delta_offsets = {comparisons[0]: 0.14, comparisons[1]: -0.14}
    for comparison in comparisons:
        style = delta_style[comparison]
        for row in (candidate for candidate in deltas if candidate["comparison"] == comparison):
            y = y_base[row["metric"]] + delta_offsets[comparison]
            estimate = numeric(row, "estimate_delta")
            lower = numeric(row, "ci95_lower")
            upper = numeric(row, "ci95_upper")
            ax_delta.errorbar(
                estimate,
                y,
                xerr=[[estimate - lower], [upper - estimate]],
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markerfacecolor=style["mfc"],
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                markersize=5.1,
                elinewidth=1.0,
                capsize=2.0,
                zorder=3,
            )
            ax_delta.text(upper + 0.004, y, f"+{estimate:.3f}", va="center", fontsize=7.1, color=style["color"])

    ax_delta.axvline(0, color=INK, lw=0.85, zorder=1)
    ax_delta.set_yticks([y_base[m] for m in METRIC_ORDER_RETRIEVAL], [metric_labels[m] for m in METRIC_ORDER_RETRIEVAL])
    ax_delta.set_ylim(-0.55, 4.55)
    ax_delta.set_xlim(-0.006, 0.207)
    ax_delta.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax_delta.set_xlabel("Paired score difference")
    panel_title(ax_delta, "B", "Fusion advantage")
    style_axis(ax_delta, "x")
    delta_handles = [
        mpl.lines.Line2D(
            [], [], color=delta_style[comparison]["color"], marker=delta_style[comparison]["marker"],
            markerfacecolor=delta_style[comparison]["mfc"], linestyle="none", label=delta_style[comparison]["label"]
        )
        for comparison in comparisons
    ]
    fig.legend(
        handles=delta_handles,
        loc="upper center",
        bbox_to_anchor=(0.775, 0.885),
        ncol=2,
        fontsize=7.1,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    return fig


def build_figure_3(font_regular: FontProperties) -> mpl.figure.Figure:
    controls = read_csv("figure_3_evidence_controls_source.csv")
    links = sorted(read_csv("figure_3_label_place_links_source.csv"), key=lambda row: numeric(row, "shrunken_risk_ratio"), reverse=True)
    co_mentions = sorted(read_csv("figure_3_co_mentions_source.csv"), key=lambda row: numeric(row, "npmi"), reverse=True)
    fig = plt.figure(figsize=FIGURE_SPECS[3]["size"])

    ax_progress = fig.add_axes([0.25, 0.66, 0.72, 0.24])
    panel_title(ax_progress, "A", "Evidence-control progression", pad=10)
    stage_x = [0, 1, 2]
    ax_progress.set_xlim(-0.25, 2.25)
    ax_progress.set_ylim(-0.5, 3.5)
    ax_progress.set_xticks(stage_x, ["Candidate", "Shared text\nexcluded", "Primary\nrelease"])
    ax_progress.xaxis.tick_top()
    ax_progress.tick_params(axis="x", length=0, pad=3)
    ax_progress.set_yticks(
        [3, 2, 1, 0],
        ["Entity strings", "Label-to-PLACE pairs", "Co-mentions\n(all 5,681 songs)", "Co-mentions\n(entity-bearing only)"],
        fontproperties=font_regular,
    )
    progress_rows = [
        ([33, 23, 22], [0, 1, 2], BLUE, ["o", "o", "s"]),
        ([85, 40, 6], [0, 1, 2], ORANGE, ["o", "o", "s"]),
        ([5, 4], [1, 2], BLUE, ["o", "s"]),
        ([9, 1], [0, 1], GRAY, ["o", "s"]),
    ]
    for y, (values, xs, color, markers) in zip([3, 2, 1, 0], progress_rows):
        ax_progress.plot(xs, [y] * len(xs), color=LIGHT_GRAY, lw=1.4, zorder=1)
        for value, x, marker in zip(values, xs, markers):
            face = color if marker == "s" else PAPER
            ax_progress.scatter(x, y, s=35, marker=marker, facecolor=face, edgecolor=color, linewidth=1.1, zorder=3)
            ax_progress.text(x, y + 0.22, str(value), ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=color)
    ax_progress.spines[:].set_visible(False)
    ax_progress.set_yticklabels(ax_progress.get_yticklabels(), fontproperties=font_regular)
    ax_progress.tick_params(axis="y", length=0, pad=8)
    ax_progress.grid(axis="x", color=GRID, lw=0.55, zorder=0)

    ax_forest = fig.add_axes([0.245, 0.105, 0.43, 0.46])
    panel_title(ax_forest, "B", "Source-label-to-PLACE enrichment", pad=9)
    y_positions = list(range(len(links) - 1, -1, -1))
    for y, row in zip(y_positions, links):
        estimate = numeric(row, "shrunken_risk_ratio")
        lower = numeric(row, "ci95_lower_conservative")
        upper = numeric(row, "ci95_upper_conservative")
        high = row["reliability_class"] == "HIGH"
        color = BLUE if high else ORANGE
        marker = "o" if high else "s"
        face = color if high else PAPER
        ax_forest.errorbar(
            estimate,
            y,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt=marker,
            color=color,
            ecolor=color,
            markerfacecolor=face,
            markeredgecolor=color,
            markeredgewidth=1.0,
            markersize=5.6,
            elinewidth=1.05,
            capsize=2.2,
            zorder=3,
        )
    labels = [
        f"{row['source_credit_label']} → {row['entity']}  ({row['entity_song_units_within_label']}/{row['label_song_units']})"
        for row in links
    ]
    ax_forest.set_yticks(y_positions, labels, fontproperties=font_regular)
    ax_forest.set_xscale("log")
    ax_forest.set_xlim(1, 1000)
    ax_forest.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1,000"])
    ax_forest.minorticks_off()
    ax_forest.axvline(1, color=INK, lw=0.85, ls="--", zorder=1)
    ax_forest.set_ylim(-0.65, len(links) - 0.35)
    ax_forest.set_xlabel("Shrunken risk ratio (log scale)")
    style_axis(ax_forest, "x")
    high_handle = mpl.lines.Line2D([], [], color=BLUE, marker="o", markerfacecolor=BLUE, linestyle="none", label="High")
    supported_handle = mpl.lines.Line2D([], [], color=ORANGE, marker="s", markerfacecolor=PAPER, linestyle="none", label="Supported")
    ax_forest.legend(handles=[high_handle, supported_handle], loc="lower right", ncol=2, handletextpad=0.35, columnspacing=0.8)

    ax_co = fig.add_axes([0.740, 0.105, 0.240, 0.46])
    panel_title(ax_co, "C", "Same-song co-mentions", pad=9)
    co_y = list(range(len(co_mentions) - 1, -1, -1))
    for y, row in zip(co_y, co_mentions):
        value = numeric(row, "npmi")
        ax_co.hlines(y, 0, value, color=BLUE_LIGHT, linewidth=5.0, zorder=2)
        ax_co.scatter(value, y, s=28, marker="o", facecolor=BLUE, edgecolor=INK, linewidth=0.6, zorder=3)
        ax_co.text(
            0.0,
            y + 0.26,
            f"{row['entity_a']}-{row['entity_b']}",
            fontproperties=font_regular,
            fontsize=7.2,
            fontweight="bold",
            ha="left",
            va="center",
        )
        ax_co.text(value + 0.025, y, f"{value:.2f} (n={row['unique_song_unit_co_mentions']})", fontsize=7.0, ha="left", va="center")
    ax_co.set_xlim(0, 0.76)
    ax_co.set_xticks([0.0, 0.25, 0.50, 0.75], ["0", ".25", ".50", ".75"])
    ax_co.set_ylim(-0.65, len(co_mentions) - 0.35)
    ax_co.set_yticks([])
    ax_co.set_xlabel("NPMI")
    style_axis(ax_co, "x")

    # Keep the frozen source tables in the QA trail: every plotted release row
    # must still be present after the layout-only journal refactor.
    expected_control_counts = [33, 23, 22, 85, 40, 6, 9, 1, 5, 4]
    observed_control_counts = [integer(row, "value") for row in controls]
    if observed_control_counts != expected_control_counts:
        raise AssertionError(f"Figure 3 control-source drift: {observed_control_counts}")
    return fig


def build_figure_4() -> mpl.figure.Figure:
    benchmark = read_csv("figure_4_rhyme_benchmark_source.csv")
    transition = read_csv("figure_4_transition_source.csv")
    personalization = read_csv("figure_4_personalization_deltas_source.csv")
    fig = plt.figure(figsize=FIGURE_SPECS[4]["size"])

    grid = fig.add_gridspec(
        1,
        4,
        left=0.265,
        right=0.985,
        bottom=0.57,
        top=0.88,
        wspace=0.16,
    )
    top_axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    fig.text(0.035, 0.965, "A  Held-out model comparison", fontsize=9.0, fontweight="bold", va="top")

    model_short = {row["model"]: row["model_short"] for row in benchmark}
    metric_label = {row["metric"]: row["metric_label"] for row in benchmark}
    model_style = {
        "hierarchical_sgd_context": {"color": BLUE, "marker": "o", "mfc": BLUE},
        "hierarchical_sgd_no_source_label": {"color": BLUE, "marker": "o", "mfc": PAPER},
        "flat_sgd_logistic_context": {"color": INK, "marker": "D", "mfc": PAPER},
        "first_order_markov": {"color": ORANGE, "marker": "s", "mfc": ORANGE_LIGHT},
        "global_frequency": {"color": GRAY, "marker": "^", "mfc": PAPER},
    }
    y_positions = {model: 4 - index for index, model in enumerate(RHYME_MODELS)}
    for axis, metric in zip(top_axes, METRIC_ORDER_RHYME):
        for model in RHYME_MODELS:
            row = next(candidate for candidate in benchmark if candidate["model"] == model and candidate["metric"] == metric)
            estimate = numeric(row, "estimate")
            lower = numeric(row, "ci95_lower")
            upper = numeric(row, "ci95_upper")
            style = model_style[model]
            axis.errorbar(
                estimate,
                y_positions[model],
                xerr=[[estimate - lower], [upper - estimate]],
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markerfacecolor=style["mfc"],
                markeredgecolor=style["color"],
                markeredgewidth=0.95,
                markersize=4.7,
                elinewidth=0.95,
                capsize=1.9,
                zorder=3,
            )
        axis.set_xlim(0, 0.85)
        axis.set_xticks([0.0, 0.4, 0.8])
        axis.set_ylim(-0.55, 4.55)
        axis.set_title(metric_label[metric], fontsize=8.2, fontweight="bold", pad=7)
        style_axis(axis, "x")

    top_axes[0].set_yticks([y_positions[model] for model in RHYME_MODELS], [model_short[model] for model in RHYME_MODELS])
    for axis in top_axes[1:]:
        axis.set_yticks([y_positions[model] for model in RHYME_MODELS], [])

    ax_transition = fig.add_axes([0.105, 0.115, 0.43, 0.30])
    panel_title(ax_transition, "B", "Top-3 accuracy by transition type", pad=9)
    transition_types = ["continuation", "switch"]
    x = np.arange(2, dtype=float)
    width = 0.30
    transition_models = ["first_order_markov", "hierarchical_sgd_context"]
    for offset, model, color, face, hatch in [
        (-width / 2, transition_models[0], ORANGE, ORANGE_LIGHT, "///"),
        (width / 2, transition_models[1], BLUE, BLUE, ""),
    ]:
        values = [
            numeric(next(row for row in transition if row["model"] == model and row["transition_type"] == transition_type), "top3_accuracy")
            for transition_type in transition_types
        ]
        bars = ax_transition.bar(
            x + offset,
            values,
            width=width,
            color=face,
            edgecolor=color,
            linewidth=0.95,
            hatch=hatch,
            label=model_short[model],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax_transition.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.035,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=color,
            )
    ax_transition.set_xticks(x, ["Continuation\n(same family)", "Switch\n(different family)"])
    ax_transition.set_ylim(0, 1.10)
    ax_transition.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax_transition.set_ylabel("Top-3 accuracy")
    ax_transition.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.82),
        ncol=1,
        columnspacing=0.8,
        handlelength=1.5,
        fontsize=7.0,
    )
    style_axis(ax_transition, "y")

    ax_ablation = fig.add_axes([0.69, 0.115, 0.285, 0.30])
    panel_title(ax_ablation, "C", "Source-label feature ablation", pad=9)
    p_by_metric = {row["metric"]: row for row in personalization}
    ablation_y = {metric: 3 - index for index, metric in enumerate(METRIC_ORDER_RHYME)}
    for metric in METRIC_ORDER_RHYME:
        row = p_by_metric[metric]
        estimate = numeric(row, "estimate_delta")
        lower = numeric(row, "ci95_lower")
        upper = numeric(row, "ci95_upper")
        ax_ablation.errorbar(
            estimate,
            ablation_y[metric],
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            markerfacecolor=PAPER,
            markeredgecolor=BLUE,
            markeredgewidth=1.0,
            markersize=5.0,
            elinewidth=1.0,
            capsize=2.1,
            zorder=3,
        )
    ax_ablation.axvline(0, color=INK, lw=0.85, zorder=1)
    ax_ablation.set_xlim(-0.0023, 0.0024)
    ax_ablation.set_xticks([-0.002, 0.0, 0.002], ["-.002", "0", "+.002"])
    ax_ablation.set_yticks([ablation_y[m] for m in METRIC_ORDER_RHYME], [metric_label[m] for m in METRIC_ORDER_RHYME])
    ax_ablation.set_ylim(-0.55, 3.55)
    ax_ablation.set_xlabel("Score difference (+ = label helps)")
    style_axis(ax_ablation, "x")
    return fig


def verify_raster(number: int, name: str, expected_pixels: tuple[int, int], require_raw: bool) -> dict[str, Any]:
    path = FIGURE_DIR / name
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0, 0))
        compression = image.info.get("compression", "")
        detail = {
            "path": f"figures/{name}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "pixels": list(image.size),
            "mode": image.mode,
            "dpi": [round(float(dpi[0]), 3), round(float(dpi[1]), 3)] if isinstance(dpi, tuple) else [0.0, 0.0],
            "compression": str(compression),
        }
        if image.size != expected_pixels:
            raise AssertionError(f"Figure {number} raster dimensions failed: {detail}")
        if image.mode not in {"RGB", "RGBA"}:
            raise AssertionError(f"Figure {number} raster mode failed: {detail}")
        if min(detail["dpi"]) < 599.0:
            raise AssertionError(f"Figure {number} raster DPI failed: {detail}")
        if require_raw and compression not in {"raw", "none", None, 1}:
            raise AssertionError(f"Figure {number} submission TIFF is not uncompressed: {detail}")
        return detail


def update_chart_contracts(exports: list[dict[str, Any]]) -> None:
    path = FIGURE_DIR / "chart_contracts.json"
    contracts = json.loads(path.read_text(encoding="utf-8"))
    by_number = {item["number"]: item for item in exports}
    for number in range(1, 5):
        contract = contracts[f"figure_{number}"]
        contract["print_size_inches"] = by_number[number]["print_size_inches"]
        contract["journal_artwork_policy"] = (
            "Caption, source, figure number, and claim-boundary prose remain outside the canvas; "
            "panel labels, axes, concise legends, and evidence-bearing annotations remain."
        )
        contract["minimum_visible_font_pt"] = MIN_FONT_PT
        contract["journal_submission_files"] = [f"fig{number}.tif", f"fig{number}.pdf", f"fig{number}.svg"]
    write_json(path, contracts)


def update_primary_validation(journal_validation: dict[str, Any]) -> None:
    """Keep the original figure-pack validation aligned with journal exports."""
    path = FIGURE_DIR / "validation.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    stale_names = {
        "image_dimensions_and_tiff_600dpi",
        "figure_3_image_dimensions_and_tiff_600dpi",
        "figures_1_2_4_unchanged",
    }
    raster_details = [
        item
        for item in journal_validation["raster_audit"]
        if item["path"].endswith(".png") or item["path"].endswith(".tiff")
    ]
    journal_checks = (
        [
            {
                "name": "journal_descriptive_rasters_exact_600dpi",
                "passed": all(min(item["dpi"]) >= 599.0 for item in raster_details),
                "detail": raster_details,
            },
            {
                "name": "journal_submission_tiffs_uncompressed_rgb",
                "passed": next(check["passed"] for check in journal_validation["checks"] if check["name"] == "submission_tiffs_uncompressed_rgb"),
                "detail": [item for item in journal_validation["raster_audit"] if item["path"].endswith(".tif")],
            },
            {
                "name": "journal_typography_and_fixed_canvas",
                "passed": all(
                    item["minimum_visible_font_pt"] >= MIN_FONT_PT
                    and item["internal_figure_label_count"] == 0
                    and not item["text_outside_canvas"]
                    for item in journal_validation["figures"]
                ),
                "detail": journal_validation["figures"],
            },
            {
                "name": "journal_vector_pdf_and_svg_present",
                "passed": len(journal_validation["vector_audit"]) == 8,
                "detail": journal_validation["vector_audit"],
            },
        ]
    )
    # Replace, never append: re-running the builder must not stack a second copy of
    # these checks on top of the previous run's now-stale hashes.
    replaced_names = stale_names | {check["name"] for check in journal_checks}
    checks = [check for check in original.get("checks", []) if check.get("name") not in replaced_names]
    checks.extend(journal_checks)
    original["generated_at_utc"] = journal_validation["generated_at_utc"]
    original["status"] = "pass" if all(check.get("passed") for check in checks) else "fail"
    original["confidence"] = "ready_to_share" if original["status"] == "pass" else "needs_revision"
    original["checks"] = checks
    original["journal_artwork_policy"] = journal_validation["artwork_policy"]
    original["journal_validation"] = "figures/journal_figure_validation.json"
    write_json(path, original)


def update_manifest(exports: list[dict[str, Any]], validation_path: Path) -> None:
    path = FIGURE_DIR / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["version"] = "1.2.0-journal"
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["status"] = "pass"
    manifest["renderer"] = {
        "matplotlib": mpl.__version__,
        "pillow": Image.__version__,
        "raster_dpi": DPI,
        "descriptive_tiff_compression": "LZW (lossless)",
        "submission_tiff_compression": "none",
        "submission_tiff_mode": "RGB",
        "vector_formats": ["PDF", "SVG"],
        "minimum_visible_font_pt": MIN_FONT_PT,
    }
    manifest["figures"] = {
        f"figure_{item['number']}": {
            "print_size_inches": item["print_size_inches"],
            "pixels_at_600_dpi": item["pixels_at_600_dpi"],
            "minimum_visible_font_pt": item["minimum_visible_font_pt"],
            "files": [f"figures/{name}" for name in item["files"]],
        }
        for item in exports
    }
    if "historical_render_lineage" not in manifest:
        manifest["historical_render_lineage"] = {
            "status": "historical_build_workspace_not_fully_public",
            "note": "Preserved for provenance only. These original paths are not presented as public, checkout-verifiable lineage.",
            "records": manifest.get("lineage", {}),
        }
    manifest["lineage"] = {
        "status": "public_checkout_verifiable",
        "note": "Published sources are value-equivalent promotions of the aggregate inputs used for the historical render; paths, bytes, and hashes below resolve in this repository.",
        "public_files": [
            {
                "path": relative,
                "bytes": (ROOT / relative).stat().st_size,
                "sha256": sha256(ROOT / relative),
            }
            for relative in PUBLIC_LINEAGE_PATHS
        ],
    }
    manifest["files"] = [
        {"path": f"figures/{file.name}", "bytes": file.stat().st_size, "sha256": sha256(file)}
        for file in sorted(FIGURE_DIR.iterdir())
        if file.is_file() and file.name != path.name
    ]
    manifest["journal_validation"] = f"figures/{validation_path.name}"
    write_json(path, manifest)


def main() -> None:
    font_regular, _ = configure_fonts()
    figures = [
        build_figure_1(),
        build_figure_2(),
        build_figure_3(font_regular),
        build_figure_4(),
    ]
    exports = [export_figure(figure, number) for number, figure in enumerate(figures, start=1)]

    raster_audits: list[dict[str, Any]] = []
    vector_audits: list[dict[str, Any]] = []
    for item in exports:
        number = item["number"]
        expected_pixels = tuple(item["pixels_at_600_dpi"])
        stem = FIGURE_SPECS[number]["stem"]
        raster_audits.append(verify_raster(number, f"{stem}.png", expected_pixels, require_raw=False))
        raster_audits.append(verify_raster(number, f"{stem}.tiff", expected_pixels, require_raw=False))
        raster_audits.append(verify_raster(number, f"fig{number}.tif", expected_pixels, require_raw=True))
        for suffix in ("pdf", "svg"):
            path = FIGURE_DIR / f"fig{number}.{suffix}"
            if path.stat().st_size <= 0:
                raise AssertionError(f"Empty vector export: {path}")
            vector_audits.append(
                {
                    "path": f"figures/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "format": suffix.upper(),
                }
            )

    validation = {
        "artifact_id": "chinese-rap-journal-figures-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "journal_canvas_width_inches": 6.5,
        "minimum_visible_font_pt_required": MIN_FONT_PT,
        "typography": {
            "latin": "Arial (Helvetica-style sans serif)",
            "cjk_fallback": "Microsoft YaHei",
            "pdf_font_mode": "TrueType / Type 42",
            "svg_text_mode": "text retained as text",
        },
        "artwork_policy": {
            "internal_figure_numbers": False,
            "internal_headlines": False,
            "internal_captions": False,
            "internal_sources": False,
            "internal_claim_boundary_prose": False,
            "panel_labels_retained": True,
            "non_colour_redundancy": "marker shape, open/filled markers, hatch, row position, and direct labels",
        },
        "figures": exports,
        "raster_audit": raster_audits,
        "vector_audit": vector_audits,
        "checks": [
            {"name": "all_visible_text_at_least_7pt", "passed": all(item["minimum_visible_font_pt"] >= MIN_FONT_PT for item in exports)},
            {"name": "no_internal_figure_labels", "passed": all(item["internal_figure_label_count"] == 0 for item in exports)},
            {"name": "no_text_outside_fixed_canvas", "passed": all(not item["text_outside_canvas"] for item in exports)},
            {"name": "all_rasters_exact_600dpi", "passed": all(min(item["dpi"]) >= 599.0 for item in raster_audits)},
            {"name": "submission_tiffs_uncompressed_rgb", "passed": all(item["mode"] == "RGB" and item["compression"] == "raw" for item in raster_audits if item["path"].endswith(".tif"))},
            {"name": "vector_pdf_and_svg_present", "passed": len(vector_audits) == 8},
        ],
        "claim_boundary": (
            "The visual refactor changes only presentation. All numerical claims remain frozen corpus-internal aggregate results; "
            "the figures do not establish performer identity, authorship, biography, residence, preference, social relations, "
            "performed rhyme, flow, delivery, or audio properties."
        ),
    }
    if not all(check["passed"] for check in validation["checks"]):
        validation["status"] = "fail"
        raise AssertionError(json.dumps(validation, ensure_ascii=False, indent=2))

    validation_path = FIGURE_DIR / "journal_figure_validation.json"
    write_json(validation_path, validation)
    update_chart_contracts(exports)
    update_primary_validation(validation)
    update_manifest(exports, validation_path)
    print(json.dumps({"status": "pass", "figures": exports, "validation": str(validation_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
