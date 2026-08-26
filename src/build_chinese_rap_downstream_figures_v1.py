#!/usr/bin/env python3
"""Build publication figures for the Chinese Rap downstream study.

The builder is intentionally aggregate-only. It reads the frozen public audit,
retrieval, and written-rhyme artifacts; it never reads lyric text, song/chunk
identifiers, embeddings, or private audit rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "chinese-rap-downstream-figures-v1"

INPUT_AUDIT = ROOT / "outputs" / "chinese-rap-downstream-input-audit-v1" / "analysis_summary.json"
RESEARCH_CONTRACT = ROOT / "work" / "downstream_v1" / "RESEARCH_CONTRACT.md"
RETRIEVAL_METRICS = ROOT / "outputs" / "chinese-rap-downstream-retrieval-v1" / "metrics.csv"
RETRIEVAL_UNCERTAINTY = ROOT / "outputs" / "chinese-rap-downstream-retrieval-v1" / "uncertainty.csv"
RETRIEVAL_SUMMARY = ROOT / "outputs" / "chinese-rap-downstream-retrieval-v1" / "analysis_summary.json"
RHYME_METRICS = ROOT / "outputs" / "chinese-rap-written-rhyme-v1" / "model_metrics.csv"
RHYME_DELTAS = ROOT / "outputs" / "chinese-rap-written-rhyme-v1" / "paired_model_deltas.csv"
RHYME_STRATIFIED = ROOT / "outputs" / "chinese-rap-written-rhyme-v1" / "stratified_metrics.csv"
RHYME_SUMMARY = ROOT / "outputs" / "chinese-rap-written-rhyme-v1" / "analysis_summary.json"

SOURCE_PATHS = [
    INPUT_AUDIT,
    RESEARCH_CONTRACT,
    RETRIEVAL_METRICS,
    RETRIEVAL_UNCERTAINTY,
    RETRIEVAL_SUMMARY,
    RHYME_METRICS,
    RHYME_DELTAS,
    RHYME_STRATIFIED,
    RHYME_SUMMARY,
]

PNG_DPI = 600
TIFF_DPI = 600

# Okabe-Ito-derived, print-safe roots plus neutrals. Every data distinction also
# uses marker shape, fill, hatch, direct labelling, or panel position.
INK = "#202124"
MUTED = "#667085"
GRID = "#D7DCE2"
PAPER = "#FFFFFF"
SOFT = "#F5F6F7"
BLUE = "#0072B2"
BLUE_LIGHT = "#CFE8F3"
ORANGE = "#D55E00"
ORANGE_LIGHT = "#F7D9C7"
GOLD = "#E69F00"
PINK = "#CC79A7"
GRAY = "#7A8088"
LIGHT_GRAY = "#BEC4CC"

METRIC_LABELS = {
    "mrr": "MRR",
    "recall_at_1": "Recall@1",
    "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10",
    "ndcg_at_10": "nDCG@10",
    "top1_accuracy": "Top-1",
    "top3_accuracy": "Top-3",
    "top5_accuracy": "Top-5",
}

RETRIEVAL_SYSTEMS = [
    "BGE-M3 dense (strict)",
    "character 2-5 gram TF-IDF (strict)",
    "equal-weight z-score fusion (strict)",
]
RETRIEVAL_SYSTEM_SHORT = {
    "BGE-M3 dense (strict)": "BGE-M3 dense",
    "character 2-5 gram TF-IDF (strict)": "Character TF-IDF",
    "equal-weight z-score fusion (strict)": "Dense + lexical fusion",
}
RETRIEVAL_METRIC_ORDER = ["mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10"]

RHYME_MODELS = [
    "hierarchical_sgd_context",
    "hierarchical_sgd_no_source_label",
    "flat_sgd_logistic_context",
    "first_order_markov",
    "global_frequency",
]
RHYME_MODEL_SHORT = {
    "hierarchical_sgd_context": "Hierarchical + label",
    "hierarchical_sgd_no_source_label": "Hierarchical, no label",
    "flat_sgd_logistic_context": "Flat context",
    "first_order_markov": "First-order Markov",
    "global_frequency": "Global frequency",
}
RHYME_METRIC_ORDER = ["top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr"]


CHART_CONTRACTS = {
    "figure_1": {
        "analytical_question": "How does one frozen corpus support three bounded downstream tasks without treating BGE-M3 as a universal method?",
        "takeaway": "Shared evidence controls precede three task-specific branches; BGE-M3 is evaluated only in retrieval, while NER and written rhyme use their own methods and claim boundaries.",
        "family": "process / research-design diagram",
        "renderer": "Matplotlib static figure",
        "data_sufficiency": "Frozen corpus counts and method definitions are available in public audit and method contracts.",
        "palette_policy": "three branch roots plus neutrals; labels and position provide redundant distinction",
        "print_size_inches": [7.5, 5.8],
    },
    "figure_2": {
        "analytical_question": "Does untuned dense-lexical fusion improve held-out-song source-credit-label repertoire retrieval over each strict single representation?",
        "takeaway": "Fusion is higher than BGE-M3 and character TF-IDF on every requested macro metric; all paired 5,000-replicate intervals are above zero.",
        "family": "faceted dot-and-interval benchmark with paired-delta panel",
        "renderer": "Matplotlib static figure",
        "data_sufficiency": "15 primary benchmark rows and 10 paired contrasts, 5,455 queries across 204 labels.",
        "palette_policy": "hard two-root cap plus neutrals; marker shape/fill redundantly distinguishes systems and contrasts",
        "print_size_inches": [7.5, 5.2],
    },
    "figure_4": {
        "analytical_question": "How accurately can strict written line-ending families be predicted, where does the task remain difficult, and does adding source-credit labels help?",
        "takeaway": "Context models improve ranking, but family switches remain difficult and the source-credit-label feature has no supported held-out benefit.",
        "family": "small-multiple dot-and-interval benchmark plus grouped transition bars",
        "renderer": "Matplotlib static figure",
        "data_sufficiency": "Five models on 34,395 leakage-safe events in 787 held-out songs; two transition strata and four paired label-ablation contrasts.",
        "palette_policy": "hard two-root cap plus neutrals; open/filled markers and hatching provide non-colour distinction",
        "print_size_inches": [7.5, 6.2],
    },
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        raise ValueError(f"Missing numeric field {key}: {row}")
    return float(value)


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "lines.linewidth": 1.1,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: mpl.axes.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=2.5, width=0.6, color=GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.8, zorder=0)


def export_figure(fig: mpl.figure.Figure, stem: str, figsize: tuple[float, float]) -> dict[str, Any]:
    png_path = OUTPUT_DIR / f"{stem}.png"
    tiff_path = OUTPUT_DIR / f"{stem}.tiff"
    fig.savefig(png_path, dpi=PNG_DPI, facecolor=PAPER, edgecolor="none")
    plt.close(fig)

    with Image.open(png_path) as image:
        rgb = image.convert("RGB")
        rgb.save(tiff_path, format="TIFF", dpi=(TIFF_DPI, TIFF_DPI), compression="tiff_lzw")

    expected_px = [int(round(figsize[0] * PNG_DPI)), int(round(figsize[1] * PNG_DPI))]
    return {
        "png": rel(png_path),
        "tiff": rel(tiff_path),
        "print_size_inches": list(figsize),
        "expected_pixels_at_600_dpi": expected_px,
    }


def rounded_box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str = GRID,
    linewidth: float = 1.0,
    radius: float = 0.012,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def build_figure_1(audit: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    figsize = tuple(CHART_CONTRACTS["figure_1"]["print_size_inches"])
    songs = int(audit["grain"]["songs"])
    chunks = int(audit["grain"]["chunks"])
    clean_chunks = int(audit["coverage"]["eligible_clean_text_chunks"])
    exact_groups = int(audit["duplicate_risk"]["exact_song_content_groups_spanning_multiple_songs"])
    grouped_songs = int(audit["duplicate_risk"]["songs_in_those_groups"])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.045, 0.958, "Figure 1", fontsize=8.3, fontweight="bold", color=BLUE, va="top")
    fig.text(0.045, 0.925, "Research design and bounded evidence flow", fontsize=14.0, fontweight="bold", va="top")
    fig.text(
        0.045,
        0.887,
        "One frozen lyric corpus; shared leakage and privacy controls; three task-specific analytical branches.",
        fontsize=8.4,
        color=MUTED,
        va="top",
    )

    rounded_box(ax, 0.055, 0.735, 0.89, 0.105, SOFT, GRID, 0.9)
    ax.text(0.078, 0.807, "FROZEN CLEANED CORPUS", transform=ax.transAxes, fontsize=7.2, fontweight="bold", color=MUTED, va="center")
    ax.text(
        0.078,
        0.775,
        f"{songs:,} songs   |   {chunks:,} canonical chunks   |   {clean_chunks:,} eligible clean-text chunks",
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.078,
        0.748,
        f"Corpus source-credit labels remain provenance labels, not independently verified performer identities.",
        transform=ax.transAxes,
        fontsize=7.1,
        color=MUTED,
        va="center",
    )

    arrow(ax, (0.50, 0.735), (0.50, 0.695))

    rounded_box(ax, 0.055, 0.585, 0.89, 0.105, INK, INK, 1.0)
    ax.text(0.078, 0.657, "SHARED EVIDENCE CONTROLS", transform=ax.transAxes, fontsize=7.2, fontweight="bold", color="#DCE3EA", va="center")
    ax.text(
        0.078,
        0.625,
        "Song-level split  |  exact-content grouping  |  task-specific fitting boundaries  |  no test-outcome tuning",
        transform=ax.transAxes,
        fontsize=6.9,
        fontweight="bold",
        color=PAPER,
        va="center",
    )
    ax.text(
        0.078,
        0.599,
        f"{exact_groups:,} exact content groups span {grouped_songs:,} songs; public outputs contain aggregates only.",
        transform=ax.transAxes,
        fontsize=7.0,
        color="#DCE3EA",
        va="center",
    )

    branch_x = [0.045, 0.355, 0.665]
    branch_w = 0.29
    branch_y = 0.285
    branch_h = 0.245
    branch_colors = [BLUE, ORANGE, PINK]
    branch_fills = ["#EAF4FA", "#FCEFE8", "#F8EDF4"]
    branch_titles = ["1  REPERTOIRE RETRIEVAL", "2  CULTURAL REFERENCES", "3  WRITTEN RHYME"]
    branch_methods = [
        "BGE-M3 dense is evaluated here\nalongside character 2-5 gram TF-IDF;\nuntuned per-query z-score fusion.",
        "Lexicon/rule baseline plus contextual\nChinese NER candidate evidence;\nagreement gates and human-review queue.",
        "Strict terminal-Han pinyin-final\nfamilies; Markov and hierarchical\nmultinomial context models.",
    ]
    branch_outputs = [
        "OUTPUT\nHeld-out-song source-credit-label\nlyrical-repertoire ranking",
        "OUTPUT\nProvisional typed references and\naggregate lyric co-mentions",
        "OUTPUT\nNext written-ending family\nprobabilities and abstention",
    ]
    branch_bounds = [
        "Not identity or authorship",
        "Not identity or social relations",
        "Not performed rhyme or flow",
    ]

    for x, color, fill, title, method, output, bound in zip(
        branch_x, branch_colors, branch_fills, branch_titles, branch_methods, branch_outputs, branch_bounds
    ):
        arrow(ax, (0.50, 0.585), (x + branch_w / 2, branch_y + branch_h + 0.012), color=MUTED)
        rounded_box(ax, x, branch_y, branch_w, branch_h, fill, color, 1.15)
        ax.add_patch(
            FancyBboxPatch(
                (x, branch_y + branch_h - 0.047),
                branch_w,
                0.047,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                linewidth=0,
                facecolor=color,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.add_patch(
            mpl.patches.Rectangle(
                (x, branch_y + branch_h - 0.047),
                branch_w,
                0.022,
                linewidth=0,
                facecolor=color,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(x + 0.014, branch_y + branch_h - 0.024, title, transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=PAPER, va="center")
        ax.text(x + 0.014, branch_y + 0.172, method, transform=ax.transAxes, fontsize=6.5, color=INK, va="top", linespacing=1.27)
        ax.plot([x + 0.014, x + branch_w - 0.014], [branch_y + 0.106, branch_y + 0.106], transform=ax.transAxes, color=GRID, lw=0.7)
        ax.text(x + 0.014, branch_y + 0.095, output, transform=ax.transAxes, fontsize=6.25, color=INK, va="top", linespacing=1.2)
        ax.text(x + 0.014, branch_y + 0.016, bound, transform=ax.transAxes, fontsize=6.4, fontweight="bold", color=color, va="bottom")

    for x in branch_x:
        arrow(ax, (x + branch_w / 2, branch_y), (0.50, 0.224), color=MUTED)

    rounded_box(ax, 0.11, 0.105, 0.78, 0.105, PAPER, INK, 1.0)
    ax.text(0.50, 0.177, "UNIFYING QUESTION", transform=ax.transAxes, ha="center", fontsize=7.0, fontweight="bold", color=MUTED)
    ax.text(
        0.50,
        0.145,
        "How do Chinese rap lyrics form recognizable lyrical identities through language,\ncultural reference, and dictionary-estimated written rhyme?",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        linespacing=1.3,
    )
    fig.text(0.045, 0.035, "Source: frozen public input audit and preregistered research contract.", fontsize=6.5, color=MUTED)

    data_rows: list[dict[str, Any]] = [
        {"component": "frozen_corpus", "branch": "shared", "item": "songs", "value": songs, "unit": "songs", "source": rel(INPUT_AUDIT)},
        {"component": "frozen_corpus", "branch": "shared", "item": "canonical_chunks", "value": chunks, "unit": "chunks", "source": rel(INPUT_AUDIT)},
        {"component": "frozen_corpus", "branch": "shared", "item": "eligible_clean_text_chunks", "value": clean_chunks, "unit": "chunks", "source": rel(INPUT_AUDIT)},
        {"component": "evidence_control", "branch": "shared", "item": "exact_song_content_groups_spanning_songs", "value": exact_groups, "unit": "groups", "source": rel(INPUT_AUDIT)},
        {"component": "evidence_control", "branch": "shared", "item": "songs_in_spanning_exact_groups", "value": grouped_songs, "unit": "songs", "source": rel(INPUT_AUDIT)},
        {"component": "evidence_control", "branch": "shared", "item": "task-specific fitting boundaries; no test-outcome fitting or selection", "value": "", "unit": "design rule", "source": rel(RESEARCH_CONTRACT)},
        {"component": "task_method", "branch": "retrieval", "item": "BGE-M3 dense + character 2-5 gram TF-IDF + untuned z-score fusion", "value": "", "unit": "method", "source": rel(RETRIEVAL_SUMMARY)},
        {"component": "task_method", "branch": "ner", "item": "lexicon/rule baseline + contextual Chinese NER candidate evidence + agreement gates", "value": "", "unit": "method", "source": rel(RESEARCH_CONTRACT)},
        {"component": "task_method", "branch": "written_rhyme", "item": "terminal-Han dictionary pinyin-final families + Markov/hierarchical context prediction", "value": "", "unit": "method", "source": rel(RHYME_SUMMARY)},
    ]
    return export_figure(fig, "figure_1_research_design", figsize), data_rows


def select_retrieval_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_metrics = read_csv(RETRIEVAL_METRICS)
    selected_metrics: list[dict[str, Any]] = []
    for row in raw_metrics:
        if row.get("aggregation") != "source_credit_label_macro_duplicate_group_adjusted":
            continue
        if row.get("task_role") != "primary_comparison":
            continue
        if row.get("system") not in RETRIEVAL_SYSTEMS or row.get("metric") not in RETRIEVAL_METRIC_ORDER:
            continue
        selected_metrics.append(
            {
                "system": row["system"],
                "system_short": RETRIEVAL_SYSTEM_SHORT[row["system"]],
                "metric": row["metric"],
                "metric_label": METRIC_LABELS[row["metric"]],
                "estimate": as_float(row, "estimate"),
                "ci95_lower": as_float(row, "ci95_lower"),
                "ci95_upper": as_float(row, "ci95_upper"),
                "queries": int(row["queries"]),
                "source_credit_labels": int(row["source_credit_labels"]),
                "global_duplicate_components": int(row["global_duplicate_components"]),
                "aggregation": row["aggregation"],
                "bootstrap_replicates": 5000,
                "source": rel(RETRIEVAL_METRICS),
            }
        )

    contrasts = ["strict fusion minus strict BGE-M3", "strict fusion minus strict TF-IDF"]
    raw_uncertainty = read_csv(RETRIEVAL_UNCERTAINTY)
    selected_deltas: list[dict[str, Any]] = []
    for row in raw_uncertainty:
        if row.get("comparison") not in contrasts or row.get("metric") not in RETRIEVAL_METRIC_ORDER:
            continue
        selected_deltas.append(
            {
                "comparison": row["comparison"],
                "comparison_short": "Fusion - BGE-M3" if row["comparison"].endswith("BGE-M3") else "Fusion - TF-IDF",
                "metric": row["metric"],
                "metric_label": METRIC_LABELS[row["metric"]],
                "estimate_delta": as_float(row, "estimate_delta"),
                "ci95_lower": as_float(row, "ci95_lower"),
                "ci95_upper": as_float(row, "ci95_upper"),
                "interval_direction": row["interval_direction"],
                "bootstrap_replicates": int(row["paired_two_stage_bootstrap_replicates"]),
                "source": rel(RETRIEVAL_UNCERTAINTY),
            }
        )

    expected_metrics = len(RETRIEVAL_SYSTEMS) * len(RETRIEVAL_METRIC_ORDER)
    expected_deltas = 2 * len(RETRIEVAL_METRIC_ORDER)
    if len(selected_metrics) != expected_metrics:
        raise AssertionError(f"Expected {expected_metrics} retrieval rows, found {len(selected_metrics)}")
    if len(selected_deltas) != expected_deltas:
        raise AssertionError(f"Expected {expected_deltas} retrieval delta rows, found {len(selected_deltas)}")
    if any(row["bootstrap_replicates"] != 5000 for row in selected_deltas):
        raise AssertionError("Retrieval paired deltas are not all based on 5,000 replicates")
    return selected_metrics, selected_deltas


def build_figure_2(metrics: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> dict[str, Any]:
    figsize = tuple(CHART_CONTRACTS["figure_2"]["print_size_inches"])
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, left=0.10, right=0.975, bottom=0.16, top=0.73, width_ratios=[1.42, 1.0], wspace=0.28)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1])

    fig.text(0.04, 0.965, "Figure 2", fontsize=8.3, fontweight="bold", color=BLUE, va="top")
    fig.text(0.04, 0.925, "Held-out-song retrieval benchmark", fontsize=14.0, fontweight="bold", va="top")
    fig.text(
        0.04,
        0.882,
        "Source-credit-label macro means; duplicate-component adjusted; 5,455 queries across 204 labels.",
        fontsize=7.5,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.04,
        0.853,
        "Whiskers: 95% paired two-stage bootstrap intervals from 5,000 literal occurrence-wise replicates.",
        fontsize=7.2,
        color=MUTED,
        va="top",
    )

    system_style = {
        "BGE-M3 dense (strict)": {"color": GRAY, "marker": "s", "mfc": PAPER, "label": "BGE-M3 dense"},
        "character 2-5 gram TF-IDF (strict)": {"color": ORANGE, "marker": "^", "mfc": ORANGE, "label": "Character TF-IDF"},
        "equal-weight z-score fusion (strict)": {"color": BLUE, "marker": "o", "mfc": BLUE, "label": "Dense + lexical fusion"},
    }
    offsets = {RETRIEVAL_SYSTEMS[0]: 0.19, RETRIEVAL_SYSTEMS[1]: 0.0, RETRIEVAL_SYSTEMS[2]: -0.19}
    y_base = {metric: len(RETRIEVAL_METRIC_ORDER) - 1 - idx for idx, metric in enumerate(RETRIEVAL_METRIC_ORDER)}

    for system in RETRIEVAL_SYSTEMS:
        style = system_style[system]
        rows = sorted((r for r in metrics if r["system"] == system), key=lambda r: RETRIEVAL_METRIC_ORDER.index(r["metric"]))
        for row in rows:
            y = y_base[row["metric"]] + offsets[system]
            x = row["estimate"]
            ax_main.errorbar(
                x,
                y,
                xerr=[[x - row["ci95_lower"]], [row["ci95_upper"] - x]],
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markerfacecolor=style["mfc"],
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                markersize=5.1,
                elinewidth=1.0,
                capsize=2.1,
                zorder=3,
            )
            ax_main.text(row["ci95_upper"] + 0.012, y, f"{x:.3f}", va="center", fontsize=6.2, color=style["color"], fontweight="bold" if system == RETRIEVAL_SYSTEMS[2] else "normal")

    handles = []
    for system in RETRIEVAL_SYSTEMS:
        style = system_style[system]
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor=style["mfc"],
                markeredgecolor=style["color"],
                linestyle="none",
                label=style["label"],
            )
        )
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.096, 0.808), ncol=3, columnspacing=1.25, handletextpad=0.4, borderaxespad=0)

    ax_main.set_yticks([y_base[m] for m in RETRIEVAL_METRIC_ORDER], [METRIC_LABELS[m] for m in RETRIEVAL_METRIC_ORDER])
    ax_main.set_ylim(-0.55, 4.55)
    ax_main.set_xlim(0.0, 0.70)
    ax_main.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax_main.set_xlabel("Retrieval score (higher is better)")
    ax_main.set_title("A  Model estimates", loc="left", fontsize=9.0, fontweight="bold", pad=9)
    style_axis(ax_main, "x")

    delta_style = {
        "strict fusion minus strict BGE-M3": {"color": BLUE, "marker": "o", "mfc": BLUE, "label": "Fusion - BGE-M3"},
        "strict fusion minus strict TF-IDF": {"color": ORANGE, "marker": "D", "mfc": PAPER, "label": "Fusion - TF-IDF"},
    }
    delta_offsets = {"strict fusion minus strict BGE-M3": 0.13, "strict fusion minus strict TF-IDF": -0.13}
    for comparison, style in delta_style.items():
        rows = sorted((r for r in deltas if r["comparison"] == comparison), key=lambda r: RETRIEVAL_METRIC_ORDER.index(r["metric"]))
        for row in rows:
            y = y_base[row["metric"]] + delta_offsets[comparison]
            x = row["estimate_delta"]
            ax_delta.errorbar(
                x,
                y,
                xerr=[[x - row["ci95_lower"]], [row["ci95_upper"] - x]],
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markerfacecolor=style["mfc"],
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                markersize=4.8,
                elinewidth=1.0,
                capsize=2.0,
                zorder=3,
            )
            ax_delta.text(row["ci95_upper"] + 0.004, y, f"+{x:.3f}", va="center", fontsize=6.1, color=style["color"], fontweight="bold")

    ax_delta.axvline(0, color=INK, lw=0.8, zorder=1)
    ax_delta.set_yticks([y_base[m] for m in RETRIEVAL_METRIC_ORDER], [METRIC_LABELS[m] for m in RETRIEVAL_METRIC_ORDER])
    ax_delta.set_ylim(-0.55, 4.55)
    ax_delta.set_xlim(-0.005, 0.205)
    ax_delta.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax_delta.set_xlabel("Paired score difference")
    ax_delta.set_title("B  Fusion advantage", loc="left", fontsize=9.0, fontweight="bold", pad=9)
    style_axis(ax_delta, "x")

    delta_handles = [
        mpl.lines.Line2D([], [], color=delta_style[c]["color"], marker=delta_style[c]["marker"], markerfacecolor=delta_style[c]["mfc"], linestyle="none", label=delta_style[c]["label"])
        for c in delta_style
    ]
    ax_delta.legend(handles=delta_handles, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=1, borderaxespad=0, handletextpad=0.3)

    fig.text(
        0.04,
        0.055,
        "Bounded interpretation: retrieval consistency within this corpus; not identity, authorship, influence, collaboration, or a human semantic-similarity benchmark.",
        fontsize=6.4,
        color=MUTED,
    )
    return export_figure(fig, "figure_2_retrieval_benchmark", figsize)


def select_rhyme_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_metrics = read_csv(RHYME_METRICS)
    benchmark: list[dict[str, Any]] = []
    ci_columns = {
        "top1_accuracy": ("top1_accuracy_ci95_low", "top1_accuracy_ci95_high"),
        "top3_accuracy": ("top3_accuracy_ci95_low", "top3_accuracy_ci95_high"),
        "top5_accuracy": ("top5_accuracy_ci95_low", "top5_accuracy_ci95_high"),
        "mrr": ("mrr_ci95_low", "mrr_ci95_high"),
    }
    for row in raw_metrics:
        if row.get("evaluation_split") != "song_held_out_test" or row.get("model") not in RHYME_MODELS:
            continue
        for metric in RHYME_METRIC_ORDER:
            low_col, high_col = ci_columns[metric]
            benchmark.append(
                {
                    "model": row["model"],
                    "model_short": RHYME_MODEL_SHORT[row["model"]],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "estimate": as_float(row, metric),
                    "ci95_lower": as_float(row, low_col),
                    "ci95_upper": as_float(row, high_col),
                    "leakage_safe_event_count": int(row["leakage_safe_event_count"]),
                    "song_count": int(row["song_count"]),
                    "end_to_end_event_coverage": as_float(row, "end_to_end_event_coverage"),
                    "bootstrap_replicates": 2000,
                    "source": rel(RHYME_METRICS),
                }
            )

    raw_strata = read_csv(RHYME_STRATIFIED)
    transition: list[dict[str, Any]] = []
    for row in raw_strata:
        if row.get("stratum_dimension") != "transition_type":
            continue
        if row.get("stratum_value") not in {"continuation", "switch"}:
            continue
        if row.get("model") not in {"first_order_markov", "hierarchical_sgd_context"}:
            continue
        transition.append(
            {
                "model": row["model"],
                "model_short": RHYME_MODEL_SHORT[row["model"]],
                "transition_type": row["stratum_value"],
                "eligible_event_count": int(row["eligible_event_count"]),
                "song_count": int(row["song_count"]),
                "top1_accuracy": as_float(row, "top1_accuracy"),
                "top3_accuracy": as_float(row, "top3_accuracy"),
                "mrr": as_float(row, "mrr"),
                "source": rel(RHYME_STRATIFIED),
            }
        )

    raw_deltas = read_csv(RHYME_DELTAS)
    personalization: list[dict[str, Any]] = []
    for row in raw_deltas:
        if row.get("released_model") != "hierarchical_sgd_context":
            continue
        if row.get("reference_model") != "hierarchical_sgd_no_source_label":
            continue
        if row.get("metric") not in RHYME_METRIC_ORDER:
            continue
        personalization.append(
            {
                "released_model": row["released_model"],
                "reference_model": row["reference_model"],
                "metric": row["metric"],
                "metric_label": METRIC_LABELS[row["metric"]],
                "estimate_delta": as_float(row, "paired_difference_released_minus_reference"),
                "ci95_lower": as_float(row, "song_cluster_bootstrap_ci95_low"),
                "ci95_upper": as_float(row, "song_cluster_bootstrap_ci95_high"),
                "bootstrap_replicates": int(row["bootstrap_replicates"]),
                "source": rel(RHYME_DELTAS),
            }
        )

    if len(benchmark) != len(RHYME_MODELS) * len(RHYME_METRIC_ORDER):
        raise AssertionError(f"Expected 20 rhyme benchmark rows, found {len(benchmark)}")
    if len(transition) != 4:
        raise AssertionError(f"Expected 4 rhyme transition rows, found {len(transition)}")
    if len(personalization) != 4:
        raise AssertionError(f"Expected 4 personalization contrasts, found {len(personalization)}")
    if any(row["ci95_lower"] > 0 or row["ci95_upper"] < 0 for row in personalization):
        raise AssertionError("At least one source-credit-label ablation interval excludes zero; update the figure claim")
    if any(row["bootstrap_replicates"] != 2000 for row in personalization):
        raise AssertionError("Rhyme paired deltas are not all based on 2,000 replicates")
    return benchmark, transition, personalization


def build_figure_4(
    benchmark: list[dict[str, Any]], transition: list[dict[str, Any]], personalization: list[dict[str, Any]]
) -> dict[str, Any]:
    figsize = tuple(CHART_CONTRACTS["figure_4"]["print_size_inches"])
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        4,
        left=0.17,
        right=0.975,
        bottom=0.105,
        top=0.715,
        height_ratios=[1.02, 0.88],
        hspace=0.58,
        wspace=0.18,
    )
    top_axes = [fig.add_subplot(gs[0, idx]) for idx in range(4)]
    ax_transition = fig.add_subplot(gs[1, :3])
    ax_note = fig.add_subplot(gs[1, 3])

    fig.text(0.04, 0.965, "Figure 4", fontsize=8.3, fontweight="bold", color=BLUE, va="top")
    fig.text(0.04, 0.925, "Written-rhyme next-family benchmark", fontsize=14.0, fontweight="bold", va="top")
    fig.text(
        0.04,
        0.882,
        "Strict terminal-Han written endings; 34,395 leakage-safe events in 787 held-out songs.",
        fontsize=7.5,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.04,
        0.853,
        "Whiskers: song-cluster 95% bootstrap intervals from 2,000 held-out-song resamples.",
        fontsize=7.2,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.04,
        0.808,
        "A  Held-out model comparison",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
        va="top",
    )

    model_style = {
        "hierarchical_sgd_context": {"color": BLUE, "marker": "o", "mfc": BLUE},
        "hierarchical_sgd_no_source_label": {"color": BLUE, "marker": "o", "mfc": PAPER},
        "flat_sgd_logistic_context": {"color": INK, "marker": "D", "mfc": PAPER},
        "first_order_markov": {"color": ORANGE, "marker": "s", "mfc": ORANGE},
        "global_frequency": {"color": GRAY, "marker": "^", "mfc": PAPER},
    }
    y_positions = {model: len(RHYME_MODELS) - 1 - idx for idx, model in enumerate(RHYME_MODELS)}

    for ax, metric in zip(top_axes, RHYME_METRIC_ORDER):
        for model in RHYME_MODELS:
            row = next(r for r in benchmark if r["model"] == model and r["metric"] == metric)
            style = model_style[model]
            x = row["estimate"]
            y = y_positions[model]
            ax.errorbar(
                x,
                y,
                xerr=[[x - row["ci95_lower"]], [row["ci95_upper"] - x]],
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markerfacecolor=style["mfc"],
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                markersize=2.7,
                elinewidth=0.9,
                capsize=1.8,
                zorder=3,
            )
            ax.text(min(row["ci95_upper"] + 0.025, 0.805), y, f"{x:.3f}", va="center", fontsize=5.7, color=style["color"], fontweight="bold" if model == "hierarchical_sgd_context" else "normal")
        ax.set_xlim(0, 0.85)
        ax.set_xticks([0.0, 0.4, 0.8])
        ax.set_ylim(-0.55, 4.55)
        ax.set_title(METRIC_LABELS[metric], fontsize=8.0, fontweight="bold", pad=7)
        style_axis(ax, "x")

    top_axes[0].set_yticks([y_positions[m] for m in RHYME_MODELS], [RHYME_MODEL_SHORT[m] for m in RHYME_MODELS])
    for ax in top_axes[1:]:
        ax.set_yticks([y_positions[m] for m in RHYME_MODELS], [])

    fig.text(0.04, 0.425, "B  Top-3 accuracy by transition type", fontsize=9.0, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.398, "Descriptive strata; no separate uncertainty intervals were released for these rows.", fontsize=6.5, color=MUTED, va="top")

    transition_types = ["continuation", "switch"]
    x = np.arange(len(transition_types), dtype=float)
    width = 0.30
    for offset, model, color, hatch in [(-width / 2, "first_order_markov", ORANGE, "///"), (width / 2, "hierarchical_sgd_context", BLUE, "")]:
        values = [next(r for r in transition if r["model"] == model and r["transition_type"] == t)["top3_accuracy"] for t in transition_types]
        bars = ax_transition.bar(
            x + offset,
            values,
            width=width,
            color=color if model == "hierarchical_sgd_context" else ORANGE_LIGHT,
            edgecolor=color,
            linewidth=0.9,
            hatch=hatch,
            label=RHYME_MODEL_SHORT[model],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax_transition.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.1%}", ha="center", va="bottom", fontsize=7.0, color=color, fontweight="bold")

    ax_transition.set_xticks(x, ["Continuation\n(same family)", "Switch\n(different family)"])
    ax_transition.set_ylim(0, 1.08)
    ax_transition.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax_transition.set_ylabel("Top-3 accuracy")
    ax_transition.legend(loc="upper center", bbox_to_anchor=(0.50, 1.16), ncol=2, columnspacing=1.4, handlelength=1.6)
    style_axis(ax_transition, "y")

    ax_note.axis("off")
    mrr_delta = next(r for r in personalization if r["metric"] == "mrr")
    rounded_box(ax_note, 0.00, 0.44, 1.00, 0.52, "#EAF4FA", BLUE, 1.0, 0.025)
    ax_note.text(0.07, 0.89, "SOURCE-LABEL ABLATION", transform=ax_note.transAxes, fontsize=6.2, fontweight="bold", color=BLUE, va="top")
    ax_note.text(0.07, 0.80, "No supported gain", transform=ax_note.transAxes, fontsize=8.0, fontweight="bold", color=INK, va="top")
    ax_note.text(
        0.07,
        0.67,
        f"Delta MRR  {mrr_delta['estimate_delta']:+.4f}\n95% CI  {mrr_delta['ci95_lower']:+.4f} to {mrr_delta['ci95_upper']:+.4f}",
        transform=ax_note.transAxes,
        fontsize=6.3,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    ax_note.text(0.07, 0.49, "All four CIs cross zero.", transform=ax_note.transAxes, fontsize=5.4, color=MUTED, va="top")

    rounded_box(ax_note, 0.00, 0.02, 1.00, 0.36, SOFT, GRID, 0.9, 0.025)
    ax_note.text(0.07, 0.32, "STRICT BOUNDARY", transform=ax_note.transAxes, fontsize=6.2, fontweight="bold", color=MUTED, va="top")
    ax_note.text(
        0.07,
        0.23,
        "Dictionary-estimated written\nendings only. No audio, beat,\ndelivery, flow, or performed rhyme.",
        transform=ax_note.transAxes,
        fontsize=5.4,
        color=INK,
        va="top",
        linespacing=1.35,
    )

    return export_figure(fig, "figure_4_written_rhyme_benchmark", figsize)


def csv_float_map(rows: Iterable[dict[str, Any]], keys: list[str], values: list[str]) -> dict[tuple[str, ...], tuple[float, ...]]:
    out: dict[tuple[str, ...], tuple[float, ...]] = {}
    for row in rows:
        out[tuple(str(row[k]) for k in keys)] = tuple(float(row[v]) for v in values)
    return out


def validate_roundtrip(
    retrieval_metrics: list[dict[str, Any]],
    retrieval_deltas: list[dict[str, Any]],
    rhyme_benchmark: list[dict[str, Any]],
    rhyme_transition: list[dict[str, Any]],
    rhyme_personalization: list[dict[str, Any]],
    figure_exports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # Re-read generated source tables rather than trusting in-memory objects.
    generated_retrieval = read_csv(OUTPUT_DIR / "figure_2_retrieval_benchmark_source.csv")
    generated_retrieval_deltas = read_csv(OUTPUT_DIR / "figure_2_retrieval_deltas_source.csv")
    generated_rhyme = read_csv(OUTPUT_DIR / "figure_4_rhyme_benchmark_source.csv")
    generated_transition = read_csv(OUTPUT_DIR / "figure_4_transition_source.csv")
    generated_personalization = read_csv(OUTPUT_DIR / "figure_4_personalization_deltas_source.csv")

    expected = csv_float_map(retrieval_metrics, ["system", "metric"], ["estimate", "ci95_lower", "ci95_upper"])
    actual = csv_float_map(generated_retrieval, ["system", "metric"], ["estimate", "ci95_lower", "ci95_upper"])
    record("figure_2_benchmark_source_roundtrip", expected == actual, {"expected_rows": len(expected), "actual_rows": len(actual)})

    expected = csv_float_map(retrieval_deltas, ["comparison", "metric"], ["estimate_delta", "ci95_lower", "ci95_upper"])
    actual = csv_float_map(generated_retrieval_deltas, ["comparison", "metric"], ["estimate_delta", "ci95_lower", "ci95_upper"])
    record("figure_2_delta_source_roundtrip", expected == actual, {"expected_rows": len(expected), "actual_rows": len(actual)})

    expected = csv_float_map(rhyme_benchmark, ["model", "metric"], ["estimate", "ci95_lower", "ci95_upper"])
    actual = csv_float_map(generated_rhyme, ["model", "metric"], ["estimate", "ci95_lower", "ci95_upper"])
    record("figure_4_benchmark_source_roundtrip", expected == actual, {"expected_rows": len(expected), "actual_rows": len(actual)})

    expected = csv_float_map(rhyme_transition, ["model", "transition_type"], ["top1_accuracy", "top3_accuracy", "mrr"])
    actual = csv_float_map(generated_transition, ["model", "transition_type"], ["top1_accuracy", "top3_accuracy", "mrr"])
    record("figure_4_transition_source_roundtrip", expected == actual, {"expected_rows": len(expected), "actual_rows": len(actual)})

    expected = csv_float_map(rhyme_personalization, ["released_model", "reference_model", "metric"], ["estimate_delta", "ci95_lower", "ci95_upper"])
    actual = csv_float_map(generated_personalization, ["released_model", "reference_model", "metric"], ["estimate_delta", "ci95_lower", "ci95_upper"])
    record("figure_4_personalization_source_roundtrip", expected == actual, {"expected_rows": len(expected), "actual_rows": len(actual)})

    record("retrieval_primary_rows_complete", len(retrieval_metrics) == 15, {"rows": len(retrieval_metrics)})
    record("retrieval_paired_delta_rows_complete", len(retrieval_deltas) == 10, {"rows": len(retrieval_deltas)})
    record("retrieval_bootstrap_replicates_5000", all(r["bootstrap_replicates"] == 5000 for r in retrieval_deltas), {"replicates": sorted({r["bootstrap_replicates"] for r in retrieval_deltas})})
    record("retrieval_fusion_intervals_above_zero", all(r["ci95_lower"] > 0 for r in retrieval_deltas), {"minimum_lower_bound": min(r["ci95_lower"] for r in retrieval_deltas)})
    record("rhyme_benchmark_rows_complete", len(rhyme_benchmark) == 20, {"rows": len(rhyme_benchmark)})
    record(
        "rhyme_song_cluster_bootstrap_replicates_2000",
        all(r["bootstrap_replicates"] == 2000 for r in rhyme_benchmark),
        {"replicates": sorted({r["bootstrap_replicates"] for r in rhyme_benchmark})},
    )
    record("rhyme_personalization_intervals_include_zero", all(r["ci95_lower"] <= 0 <= r["ci95_upper"] for r in rhyme_personalization), {"contrasts": len(rhyme_personalization)})

    image_details: list[dict[str, Any]] = []
    image_ok = True
    for figure_id, export in figure_exports.items():
        expected_px = tuple(export["expected_pixels_at_600_dpi"])
        for format_key in ("png", "tiff"):
            path = ROOT / export[format_key]
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (0, 0))
                detail = {
                    "figure": figure_id,
                    "format": format_key,
                    "path": rel(path),
                    "pixels": list(image.size),
                    "mode": image.mode,
                    "dpi": [float(dpi[0]), float(dpi[1])] if isinstance(dpi, tuple) and len(dpi) >= 2 else [0.0, 0.0],
                }
                image_details.append(detail)
                format_ok = image.size == expected_px
                if format_key == "tiff":
                    format_ok = format_ok and image.mode == "RGB" and min(detail["dpi"]) >= 599.0
                image_ok = image_ok and format_ok
    record("image_dimensions_and_tiff_600dpi", image_ok, image_details)

    # Public source tables may contain only aggregate method/result fields.
    forbidden_fragments = ["song_id", "chunk_id", "lyric", "analysis_text", "content_hash", "embedding", "private"]
    generated_csvs = sorted(OUTPUT_DIR.glob("*_source.csv"))
    privacy_hits: list[dict[str, str]] = []
    for path in generated_csvs:
        rows = read_csv(path)
        headers = list(rows[0].keys()) if rows else []
        for header in headers:
            lowered = header.lower()
            if any(fragment in lowered for fragment in forbidden_fragments):
                privacy_hits.append({"path": rel(path), "header": header})
    record("aggregate_only_privacy_schema_scan", not privacy_hits, {"files_scanned": len(generated_csvs), "forbidden_header_hits": privacy_hits})

    return {
        "artifact_id": "chinese-rap-downstream-figures-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "confidence": "ready_to_share" if all(check["passed"] for check in checks) else "needs_revision",
        "claim_boundary": "Figures report corpus-internal aggregate evidence. They do not establish performer identity, authorship, biography, social relations, performed rhyme, flow, delivery, or audio properties.",
        "checks": checks,
    }


def caption_markdown() -> str:
    return """# Figure captions and alt text

## Figure 1 — Research design and bounded evidence flow

**Caption.** Figure 1. Study design. A frozen cleaned corpus (7,211 songs; 22,128 canonical chunks; 21,553 eligible clean-text chunks) enters shared song-level splitting, duplicate controls, task-specific fitting boundaries, a prohibition on using test outcomes for fitting or selection, and aggregate-only public release. The retrieval TF-IDF representation is transductively fit on the fixed unlabeled evaluation corpus, while labelled outcomes remain unused. The branches then use distinct task methods. BGE-M3 is evaluated only in held-out-song retrieval alongside character TF-IDF and untuned fusion; cultural-reference extraction combines lexicon and contextual-model candidate evidence and remains provisional without human gold; written-rhyme prediction uses dictionary-estimated terminal-Han pinyin-final families and task-specific context models. Every output is paired with its permitted interpretation.

**Alt text.** A top-to-bottom research pipeline begins with a frozen Chinese-rap lyric corpus and a shared evidence-control block. Three side-by-side branches follow: repertoire retrieval using BGE-M3, character TF-IDF, and fusion; provisional cultural-reference extraction using lexicon and contextual Chinese NER evidence; and written-rhyme prediction using terminal-Han pinyin-final families and Markov or hierarchical context models. The branches converge on one question about lyrical identity. Each branch states a boundary: retrieval is not identity or authorship, NER is not identity or a social relation, and written rhyme is not performed rhyme or flow.

**Takeaway.** The corpus and leakage controls are shared, but the downstream methods and valid claims are task-specific; BGE-M3 is a tested retrieval representation rather than a universal analytical engine.

## Figure 2 — Held-out-song retrieval benchmark

**Caption.** Figure 2. Strict held-out-song source-credit-label lyrical-repertoire retrieval. Panel A shows duplicate-component-adjusted macro MRR, Recall@1/5/10, and nDCG@10 for BGE-M3 dense representations, character 2–5-gram TF-IDF, and their untuned per-query z-score fusion. Panel B shows paired fusion-minus-baseline differences. Whiskers are 95% intervals from 5,000 literal occurrence-wise paired two-stage bootstrap replicates over 5,455 queries and 204 source-credit labels. Fusion is higher than both single representations on every requested metric, including MRR 0.447 (95% CI 0.414–0.481), but the result measures corpus-internal repertoire consistency rather than identity or authorship.

**Alt text.** Two dot-and-whisker panels compare five retrieval metrics. In the model panel, fusion is highest on all metrics, character TF-IDF is second, and BGE-M3 dense is lowest. In the paired-difference panel, all ten fusion-minus-baseline intervals lie to the right of zero; gains over TF-IDF are smaller than gains over BGE-M3.

**Takeaway.** Dense and character-form evidence are complementary: their untuned fusion produces a supported improvement over either representation alone across all five held-out metrics.

## Figure 4 — Written-rhyme next-family benchmark

**Caption.** Figure 4. Prediction of the next dictionary-estimated written line-ending family on the strict terminal-Han population. Panel A compares global frequency, first-order Markov, flat multinomial context, hierarchical context without source-credit labels, and hierarchical context with source-credit labels across 34,395 leakage-safe events in 787 held-out songs. Whiskers are 95% song-cluster bootstrap intervals from 2,000 replicates. Panel B descriptively separates continuation from family-switch events: top-3 accuracy is near ceiling for continuation but only 0.400 for switches under the hierarchical model (0.302 for Markov). Adding source-credit labels has no supported benefit: all four paired intervals include zero (MRR difference +0.0005, 95% CI −0.0001 to +0.0011). The task concerns written dictionary finals, not audio, flow, delivery, beat, or performed rhyme.

**Alt text.** Four aligned dot-and-whisker panels show Top-1, Top-3, Top-5, and MRR for five models. Context models dominate the global baseline; the flat and two hierarchical models are nearly overlapping. A grouped bar panel shows nearly perfect Top-3 accuracy when the next line continues the same rhyme family, but much lower accuracy when it switches. A callout states that adding the source-credit label has no supported gain and that the task is restricted to written endings.

**Takeaway.** Written-ending context is predictive, but most apparent success comes from continuation; switching remains difficult, and this evaluation does not support source-label personalization.
"""


def main() -> None:
    for source in SOURCE_PATHS:
        if not source.is_file():
            raise FileNotFoundError(source)

    expected_output = (ROOT / "outputs" / "chinese-rap-downstream-figures-v1").resolve()
    if OUTPUT_DIR.resolve() != expected_output:
        raise RuntimeError(f"Refusing to clean unexpected output path: {OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    setup_matplotlib()
    audit = read_json(INPUT_AUDIT)
    retrieval_summary = read_json(RETRIEVAL_SUMMARY)
    rhyme_summary = read_json(RHYME_SUMMARY)

    if audit.get("status") != "pass":
        raise AssertionError("Frozen input audit is not passing")
    if retrieval_summary.get("version") != "1.1.0":
        raise AssertionError(f"Retrieval output is not the frozen v1.1.0 artifact: {retrieval_summary.get('version')}")
    if int(retrieval_summary["uncertainty"]["replicates"]) != 5000:
        raise AssertionError("Retrieval uncertainty is not based on 5,000 replicates")
    if rhyme_summary.get("version") != "1.1.0":
        raise AssertionError(f"Rhyme output is not the corrected frozen v1.1.0 artifact: {rhyme_summary.get('version')}")

    retrieval_metrics, retrieval_deltas = select_retrieval_rows()
    rhyme_benchmark, rhyme_transition, rhyme_personalization = select_rhyme_rows()

    figure_exports: dict[str, dict[str, Any]] = {}
    figure_exports["figure_1"], figure_1_source = build_figure_1(audit)
    figure_exports["figure_2"] = build_figure_2(retrieval_metrics, retrieval_deltas)
    figure_exports["figure_4"] = build_figure_4(rhyme_benchmark, rhyme_transition, rhyme_personalization)

    write_csv(
        OUTPUT_DIR / "figure_1_pipeline_source.csv",
        figure_1_source,
        ["component", "branch", "item", "value", "unit", "source"],
    )
    write_csv(
        OUTPUT_DIR / "figure_2_retrieval_benchmark_source.csv",
        retrieval_metrics,
        ["system", "system_short", "metric", "metric_label", "estimate", "ci95_lower", "ci95_upper", "queries", "source_credit_labels", "global_duplicate_components", "aggregation", "bootstrap_replicates", "source"],
    )
    write_csv(
        OUTPUT_DIR / "figure_2_retrieval_deltas_source.csv",
        retrieval_deltas,
        ["comparison", "comparison_short", "metric", "metric_label", "estimate_delta", "ci95_lower", "ci95_upper", "interval_direction", "bootstrap_replicates", "source"],
    )
    write_csv(
        OUTPUT_DIR / "figure_4_rhyme_benchmark_source.csv",
        rhyme_benchmark,
        ["model", "model_short", "metric", "metric_label", "estimate", "ci95_lower", "ci95_upper", "leakage_safe_event_count", "song_count", "end_to_end_event_coverage", "bootstrap_replicates", "source"],
    )
    write_csv(
        OUTPUT_DIR / "figure_4_transition_source.csv",
        rhyme_transition,
        ["model", "model_short", "transition_type", "eligible_event_count", "song_count", "top1_accuracy", "top3_accuracy", "mrr", "source"],
    )
    write_csv(
        OUTPUT_DIR / "figure_4_personalization_deltas_source.csv",
        rhyme_personalization,
        ["released_model", "reference_model", "metric", "metric_label", "estimate_delta", "ci95_lower", "ci95_upper", "bootstrap_replicates", "source"],
    )

    write_json(OUTPUT_DIR / "chart_contracts.json", CHART_CONTRACTS)
    (OUTPUT_DIR / "figure_captions_and_alt_text.md").write_text(caption_markdown(), encoding="utf-8", newline="\n")

    validation = validate_roundtrip(
        retrieval_metrics,
        retrieval_deltas,
        rhyme_benchmark,
        rhyme_transition,
        rhyme_personalization,
        figure_exports,
    )
    write_json(OUTPUT_DIR / "validation.json", validation)
    if validation["status"] != "pass":
        raise AssertionError("Publication-figure validation failed; inspect validation.json")

    generated_files = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file())
    builder_path = Path(__file__).resolve()
    manifest = {
        "artifact_id": "chinese-rap-downstream-figures-v1",
        "version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": validation["status"],
        "renderer": {
            "matplotlib": mpl.__version__,
            "pillow": Image.__version__,
            "png_dpi": PNG_DPI,
            "tiff_dpi": TIFF_DPI,
            "tiff_mode": "RGB",
            "tiff_compression": "LZW",
        },
        "figures": figure_exports,
        "claim_boundary": validation["claim_boundary"],
        "lineage": {
            "builder": {"path": rel(builder_path), "sha256": sha256(builder_path)},
            "sources": [{"path": rel(path), "sha256": sha256(path)} for path in SOURCE_PATHS],
        },
        "files": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in generated_files],
        "privacy": "Aggregate-only figures and source tables; no lyric text, song/chunk IDs, content hashes, embeddings, or private audit rows.",
    }
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    print(json.dumps({"status": "pass", "output_dir": rel(OUTPUT_DIR), "figures": figure_exports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
