#!/usr/bin/env python3
"""Add the frozen NER/cultural-reference Figure 3 to the publication figure pack.

This builder is deliberately aggregate-only. It reads only the public v1.1 NER
release and updates Figure 3 plus figure-pack metadata. Existing Figures 1, 2,
and 4 are hashed before and after execution and must remain byte-identical.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NER_DIR = ROOT / "results" / "ner-v1"
OUTPUT_DIR = ROOT / "figures"

NER_SUMMARY = NER_DIR / "summary.json"
NER_SENSITIVITY = NER_DIR / "release_sensitivity_summary.csv"
NER_LABEL_LINKS = NER_DIR / "source_label_entity_links_provisional.csv"
NER_CO_MENTIONS = NER_DIR / "entity_co_mentions_provisional.csv"
NER_VALIDATION = NER_DIR / "validation.json"
NER_RECONCILIATION = NER_DIR / "reconciliation_validation.json"

FIGURE_STEM = "figure_3_cultural_reference_evidence"
FIGSIZE = (7.5, 6.6)
DPI = 600

INK = "#202124"
MUTED = "#667085"
GRID = "#D7DCE2"
PAPER = "#FFFFFF"
SOFT = "#F5F6F7"
BLUE = "#0072B2"
BLUE_LIGHT = "#CFE8F3"
ORANGE = "#D55E00"
ORANGE_LIGHT = "#F7D9C7"
GRAY = "#7A8088"
LIGHT_GRAY = "#BEC4CC"

CJK_FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
CJK_BOLD_FONT_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")

FIGURE3_CONTRACT = {
    "analytical_question": "What cultural-reference evidence remains after shared-text exclusion, uncertainty control, and BH-FDR screening?",
    "takeaway": "Evidence controls reduce a broad candidate inventory to 22 provisional entities, six source-label-to-place enrichments, and four same-song co-mentions; all remain lyric-reference claims rather than biographical or social claims.",
    "family": "stage-progression evidence controls plus log-scale forest plot and compact co-mention dot bars",
    "renderer": "Matplotlib static figure",
    "data_sufficiency": "Six released label-to-PLACE links with conservative intervals and four released co-mentions over 5,681 eligible song units; no completed human gold.",
    "palette_policy": "hard two-root cap plus neutrals; marker shape, fill, labels, and panel position provide non-colour distinction",
    "print_size_inches": list(FIGSIZE),
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


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def setup_matplotlib() -> FontProperties:
    if not CJK_FONT_PATH.is_file():
        raise FileNotFoundError(f"Required CJK font not found: {CJK_FONT_PATH}")
    mpl.font_manager.fontManager.addfont(str(CJK_FONT_PATH))
    if CJK_BOLD_FONT_PATH.is_file():
        mpl.font_manager.fontManager.addfont(str(CJK_BOLD_FONT_PATH))
    cjk = FontProperties(fname=str(CJK_FONT_PATH))
    family = cjk.get_name()
    mpl.rcParams.update(
        {
            "font.family": family,
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
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.1,
            "lines.markersize": 5.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return cjk


def style_axis(ax: mpl.axes.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=2.5, width=0.6, color=GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.8, zorder=0)


def rounded_box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str = PAPER,
    edgecolor: str = GRID,
    linewidth: float = 0.9,
    radius: float = 0.018,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def compact_q(value: float) -> str:
    if value < 0.001:
        exponent = int(math.floor(math.log10(value)))
        mantissa = value / (10**exponent)
        return f"{mantissa:.1f}e{exponent}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def load_and_validate_sources() -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    for path in [NER_SUMMARY, NER_SENSITIVITY, NER_LABEL_LINKS, NER_CO_MENTIONS, NER_VALIDATION, NER_RECONCILIATION]:
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = read_json(NER_SUMMARY)
    sensitivity = read_csv(NER_SENSITIVITY)
    links = read_csv(NER_LABEL_LINKS)
    co_mentions = read_csv(NER_CO_MENTIONS)
    validation = read_json(NER_VALIDATION)
    reconciliation = read_json(NER_RECONCILIATION)

    if summary.get("version") != "1.1.0":
        raise AssertionError(f"Expected frozen NER v1.1.0, found {summary.get('version')}")
    if validation.get("status") != "pass" or reconciliation.get("status") != "pass":
        raise AssertionError("NER validation or reconciliation is not passing")
    if summary.get("human_gold_available") is not False:
        raise AssertionError("Figure 3 assumes that no completed human gold is available")
    if len(links) != 6 or any(row["entity_type"] != "PLACE" for row in links):
        raise AssertionError("Expected exactly six released label-to-PLACE links")
    if len(co_mentions) != 4:
        raise AssertionError("Expected exactly four released co-mentions")
    if any(int(row["all_eligible_song_units"]) != 5681 for row in co_mentions):
        raise AssertionError("Co-mentions do not all use the corrected 5,681-song denominator")
    return summary, sensitivity, links, co_mentions


def evidence_control_rows(sensitivity: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_stage = {row["stage"]: row for row in sensitivity}
    expected = {
        "v1_legacy_all_labels_before_shared_exclusion": (33, 85, 9),
        "all_labels_after_shared_exclusion_legacy_gates": (23, 40, 1),
        "204_label_primary_universe_after_shared_exclusion_legacy_gates": (22, 40, 1),
        "v1_1_primary_all_song_denominator_uncertainty_fdr_release": (22, 6, 4),
    }
    for stage, counts in expected.items():
        row = by_stage.get(stage)
        if row is None:
            raise AssertionError(f"Missing sensitivity stage: {stage}")
        actual = (int(row["entities"]), int(row["label_entity_links"]), int(row["co_mentions"]))
        if actual != counts:
            raise AssertionError(f"Unexpected sensitivity counts for {stage}: {actual} != {counts}")

    return [
        {
            "dimension": "entity_inventory",
            "denominator_definition": "unique entity strings",
            "stage_order": 1,
            "stage": "corpuswide_candidates",
            "value": 33,
            "unit": "entities",
            "meaning": "Corpus-wide provisional inventory before shared-text exclusion.",
            "source": rel(NER_SENSITIVITY),
        },
        {
            "dimension": "entity_inventory",
            "denominator_definition": "unique entity strings",
            "stage_order": 2,
            "stage": "all_labels_after_shared_text_exclusion",
            "value": 23,
            "unit": "entities",
            "meaning": "All-label sensitivity inventory after cross-label exact shared-text exclusion.",
            "source": rel(NER_SENSITIVITY),
        },
        {
            "dimension": "entity_inventory",
            "denominator_definition": "unique entity strings",
            "stage_order": 3,
            "stage": "fixed_204_label_primary_universe",
            "value": 22,
            "unit": "entities",
            "meaning": "Primary release inventory in the fixed 204-label universe.",
            "source": rel(NER_SENSITIVITY),
        },
        {
            "dimension": "label_reference_links",
            "denominator_definition": "source-label/entity pairs",
            "stage_order": 1,
            "stage": "legacy_basic_gate_shared_text_included",
            "value": 85,
            "unit": "pairs",
            "meaning": "Legacy all-label support/lift gate before shared-text exclusion.",
            "source": rel(NER_SENSITIVITY),
        },
        {
            "dimension": "label_reference_links",
            "denominator_definition": "source-label/entity pairs",
            "stage_order": 2,
            "stage": "shared_text_excluded_legacy_gate",
            "value": 40,
            "unit": "pairs",
            "meaning": "Candidate pairs after shared-text exclusion with legacy gates retained for sensitivity.",
            "source": rel(NER_SENSITIVITY),
        },
        {
            "dimension": "label_reference_links",
            "denominator_definition": "source-label/entity pairs",
            "stage_order": 3,
            "stage": "uncertainty_and_bh_fdr_release",
            "value": 6,
            "unit": "pairs",
            "meaning": "Released pairs after shrinkage, conservative intervals, and BH-FDR.",
            "source": rel(NER_SENSITIVITY),
        },
        {
            "dimension": "co_mentions_legacy_denominator",
            "denominator_definition": "entity-bearing songs only (legacy sensitivity; not comparable to primary release)",
            "stage_order": 1,
            "stage": "shared_text_included_legacy_gate",
            "value": 9,
            "unit": "entity pairs",
            "meaning": "Legacy before-exclusion sensitivity count under the entity-bearing denominator.",
            "source": rel(NER_SENSITIVITY),
        },
        {
            "dimension": "co_mentions_legacy_denominator",
            "denominator_definition": "entity-bearing songs only (legacy sensitivity; not comparable to primary release)",
            "stage_order": 2,
            "stage": "shared_text_excluded_legacy_gate",
            "value": 1,
            "unit": "entity pairs",
            "meaning": "Legacy after-exclusion sensitivity count under the entity-bearing denominator.",
            "source": rel(NER_SENSITIVITY),
        },
        {
            "dimension": "co_mentions_corrected_denominator",
            "denominator_definition": "all 5,681 eligible shared-text-excluded song units",
            "stage_order": 1,
            "stage": "basic_gate",
            "value": 5,
            "unit": "entity pairs",
            "meaning": "Primary-universe candidates under the corrected all-song denominator.",
            "source": rel(NER_SENSITIVITY),
        },
        {
            "dimension": "co_mentions_corrected_denominator",
            "denominator_definition": "all 5,681 eligible shared-text-excluded song units",
            "stage_order": 2,
            "stage": "bh_fdr_release",
            "value": 4,
            "unit": "entity pairs",
            "meaning": "Released same-song co-mentions after BH-FDR.",
            "source": rel(NER_SENSITIVITY),
        },
    ]


def normalized_label_link_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "source_credit_label": row["source_credit_label"],
                "entity": row["entity"],
                "entity_type": row["entity_type"],
                "label_song_units": int(row["label_song_units"]),
                "entity_song_units_within_label": int(row["entity_song_units_within_label"]),
                "entity_song_units_corpus": int(row["entity_song_units_corpus"]),
                "all_membership_song_units": int(row["all_membership_song_units"]),
                "within_label_share": float(row["within_label_share"]),
                "corpus_share": float(row["corpus_share"]),
                "shrunken_risk_ratio": float(row["shrunken_risk_ratio"]),
                "ci95_lower_conservative": float(row["shrunken_risk_ratio_ci95_low_conservative"]),
                "ci95_upper_conservative": float(row["shrunken_risk_ratio_ci95_high_conservative"]),
                "q_value_bh": float(row["q_value_bh"]),
                "reliability_class": row["reliability_class"],
                "association_scope": row["association_scope"],
                "source": rel(NER_LABEL_LINKS),
            }
        )
    return sorted(normalized, key=lambda row: row["shrunken_risk_ratio"], reverse=True)


def normalized_co_mention_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "entity_a": row["entity_a"],
                "entity_a_type": row["entity_a_type"],
                "entity_b": row["entity_b"],
                "entity_b_type": row["entity_b_type"],
                "all_eligible_song_units": int(row["all_eligible_song_units"]),
                "unique_song_unit_co_mentions": int(row["unique_song_unit_co_mentions"]),
                "source_credit_labels": int(row["source_credit_labels"]),
                "npmi": float(row["npmi"]),
                "q_value_bh": float(row["q_value_bh"]),
                "reliability_class": row["reliability_class"],
                "relation_scope": row["relation_scope"],
                "source": rel(NER_CO_MENTIONS),
            }
        )
    return sorted(normalized, key=lambda row: row["npmi"], reverse=True)


def draw_stage_card(
    ax: mpl.axes.Axes,
    x: float,
    width: float,
    title: str,
    unit: str,
    stages: list[tuple[int, str, str]],
    color: str,
) -> None:
    rounded_box(ax, x, 0.04, width, 0.88, PAPER, GRID, 0.9, 0.025)
    ax.text(x + 0.018, 0.855, title, transform=ax.transAxes, fontsize=7.4, fontweight="bold", va="top")
    ax.text(x + 0.018, 0.775, unit, transform=ax.transAxes, fontsize=5.9, color=MUTED, va="top")
    maximum = max(value for value, _, _ in stages)
    row_y = [0.61, 0.39, 0.17]
    for idx, ((value, label, stage_note), y) in enumerate(zip(stages, row_y)):
        bar_w = (width - 0.055) * value / maximum
        fill = LIGHT_GRAY if idx == 0 else (BLUE_LIGHT if idx == 1 else color)
        edge = GRAY if idx < 2 else color
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.018, y),
                max(bar_w, 0.025),
                0.074,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.7,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(x + 0.028, y + 0.037, f"{value}", transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=INK, va="center")
        ax.text(x + 0.018, y - 0.018, label, transform=ax.transAxes, fontsize=5.55, color=INK, va="top")
        ax.text(x + 0.018, y - 0.066, stage_note, transform=ax.transAxes, fontsize=4.9, color=MUTED, va="top")


def draw_co_mention_control_card(ax: mpl.axes.Axes, x: float, width: float) -> None:
    rounded_box(ax, x, 0.04, width, 0.88, PAPER, GRID, 0.9, 0.025)
    ax.text(x + 0.018, 0.855, "Same-song co-mentions", transform=ax.transAxes, fontsize=7.4, fontweight="bold", va="top")
    ax.text(x + 0.018, 0.775, "two denominator definitions — keep separate", transform=ax.transAxes, fontsize=5.75, color=MUTED, va="top")

    ax.text(x + 0.018, 0.655, "LEGACY SENSITIVITY", transform=ax.transAxes, fontsize=5.2, color=GRAY, fontweight="bold", va="center")
    ax.text(x + 0.018, 0.575, "9  →  1", transform=ax.transAxes, fontsize=10.5, color=INK, fontweight="bold", va="center")
    ax.text(x + 0.018, 0.490, "entity-bearing-song denominator", transform=ax.transAxes, fontsize=5.2, color=MUTED, va="top")
    ax.text(x + 0.018, 0.438, "before → after shared-text exclusion", transform=ax.transAxes, fontsize=4.9, color=MUTED, va="top")

    ax.plot([x + 0.018, x + width - 0.018], [0.375, 0.375], transform=ax.transAxes, color=GRID, lw=0.7)

    ax.text(x + 0.018, 0.315, "PRIMARY RELEASE", transform=ax.transAxes, fontsize=5.2, color=BLUE, fontweight="bold", va="center")
    ax.text(x + 0.018, 0.235, "5  →  4", transform=ax.transAxes, fontsize=10.5, color=BLUE, fontweight="bold", va="center")
    ax.text(x + 0.018, 0.150, "all 5,681 eligible song units", transform=ax.transAxes, fontsize=5.2, color=INK, va="top")
    ax.text(x + 0.018, 0.098, "basic gate → BH-FDR release", transform=ax.transAxes, fontsize=4.9, color=MUTED, va="top")


def build_figure(
    controls: list[dict[str, Any]],
    links: list[dict[str, Any]],
    co_mentions: list[dict[str, Any]],
) -> dict[str, Any]:
    cjk = setup_matplotlib()
    fig = plt.figure(figsize=FIGSIZE)

    fig.text(0.04, 0.968, "Figure 3", fontsize=8.3, fontweight="bold", color=BLUE, va="top")
    fig.text(0.04, 0.930, "Cultural-reference evidence after leakage and uncertainty controls", fontsize=13.1, fontweight="bold", va="top")
    fig.text(
        0.04,
        0.890,
        "Provisional Chinese-rap NER evidence; cross-label exact shared text excluded; released associations BH-FDR screened (q ≤ .05).",
        fontsize=7.2,
        color=MUTED,
        va="top",
    )

    # Panel A: three evidence-control cards. Counts use visibly separated units.
    ax_cards = fig.add_axes([0.04, 0.595, 0.92, 0.245])
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    ax_cards.axis("off")
    ax_cards.text(0.0, 1.02, "A  Evidence-control progression", transform=ax_cards.transAxes, fontsize=8.7, fontweight="bold", va="bottom")
    draw_stage_card(
        ax_cards,
        0.00,
        0.30,
        "Entity inventory",
        "unique entity strings",
        [
            (33, "Corpus-wide", "candidate inventory"),
            (23, "Shared text excluded", "all-label sensitivity"),
            (22, "Fixed 204-label universe", "primary provisional inventory"),
        ],
        BLUE,
    )
    draw_stage_card(
        ax_cards,
        0.35,
        0.30,
        "Label → reference links",
        "source-label/entity pairs",
        [
            (85, "Legacy basic gate", "shared text included"),
            (40, "Shared text excluded", "legacy gate held fixed"),
            (6, "Uncertainty + BH-FDR", "released links"),
        ],
        ORANGE,
    )
    draw_co_mention_control_card(ax_cards, 0.70, 0.30)

    # Panel B: all released label-to-place enrichments, with uncertainty on a log axis.
    # Reserve explicit left space for mixed Chinese/Latin row labels so the
    # print export never clips the first characters.
    ax_forest = fig.add_axes([0.220, 0.145, 0.425, 0.355])
    ax_info = fig.add_axes([0.655, 0.145, 0.14, 0.355], sharey=ax_forest)
    y_positions = list(range(len(links) - 1, -1, -1))
    for y, row in zip(y_positions, links):
        estimate = row["shrunken_risk_ratio"]
        lower = row["ci95_lower_conservative"]
        upper = row["ci95_upper_conservative"]
        is_high = row["reliability_class"] == "HIGH"
        color = BLUE if is_high else ORANGE
        marker = "o" if is_high else "s"
        marker_fill = color if is_high else PAPER
        ax_forest.errorbar(
            estimate,
            y,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt=marker,
            color=color,
            ecolor=color,
            markerfacecolor=marker_fill,
            markeredgecolor=color,
            markeredgewidth=0.9,
            markersize=5.4,
            elinewidth=1.0,
            capsize=2.2,
            zorder=3,
        )

    y_labels = [
        f"{row['source_credit_label']} → {row['entity']}   {row['entity_song_units_within_label']}/{row['label_song_units']}"
        for row in links
    ]
    ax_forest.set_yticks(y_positions, y_labels, fontproperties=cjk)
    ax_forest.set_xscale("log")
    ax_forest.set_xlim(1, 1000)
    ax_forest.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1,000"])
    ax_forest.axvline(1, color=INK, lw=0.8, ls="--", zorder=1)
    ax_forest.set_ylim(-0.7, len(links) - 0.3)
    ax_forest.set_xlabel("Shrunken risk ratio (log scale; conservative 95% interval)")
    ax_forest.set_title("B  Source-label → PLACE enrichment", loc="left", fontsize=8.7, fontweight="bold", pad=28)
    ax_forest.text(
        0.0,
        1.055,
        "Row suffix = referenced songs / eligible songs for that label",
        transform=ax_forest.transAxes,
        fontsize=5.7,
        color=MUTED,
        va="bottom",
    )
    style_axis(ax_forest, "x")
    ax_forest.minorticks_off()

    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(ax_forest.get_ylim())
    ax_info.axis("off")
    ax_info.text(0.02, len(links) - 0.1, "Reliability", fontsize=5.5, color=MUTED, fontweight="bold", va="bottom")
    ax_info.text(0.61, len(links) - 0.1, "BH q", fontsize=5.5, color=MUTED, fontweight="bold", va="bottom")
    for y, row in zip(y_positions, links):
        color = BLUE if row["reliability_class"] == "HIGH" else ORANGE
        ax_info.text(0.02, y, row["reliability_class"].title(), fontsize=5.6, color=color, fontweight="bold", va="center")
        ax_info.text(0.61, y, compact_q(row["q_value_bh"]), fontsize=5.5, color=INK, va="center")

    # Panel C: released co-mentions under one corrected denominator.
    ax_co = fig.add_axes([0.815, 0.145, 0.155, 0.355])
    co_y = list(range(len(co_mentions) - 1, -1, -1))
    values = [row["npmi"] for row in co_mentions]
    ax_co.hlines(co_y, 0, values, color=BLUE_LIGHT, linewidth=5.5, zorder=2)
    ax_co.scatter(values, co_y, s=25, facecolor=BLUE, edgecolor=BLUE, linewidth=0.7, zorder=3)
    for y, row in zip(co_y, co_mentions):
        ax_co.text(
            0,
            y + 0.24,
            f"{row['entity_a']} — {row['entity_b']}",
            fontproperties=cjk,
            fontsize=5.7,
            color=INK,
            fontweight="bold",
            va="center",
            ha="left",
        )
        ax_co.text(
            row["npmi"] + 0.022,
            y,
            f"{row['npmi']:.2f} · n={row['unique_song_unit_co_mentions']}\nq={compact_q(row['q_value_bh'])}",
            fontsize=4.55,
            color=INK,
            va="center",
            linespacing=1.15,
        )
    ax_co.set_yticks([])
    ax_co.set_xlim(0, 0.76)
    ax_co.set_xticks([0.0, 0.25, 0.5], ["0", ".25", ".50"])
    ax_co.set_ylim(-0.7, len(co_mentions) - 0.15)
    ax_co.set_xlabel("NPMI", labelpad=2)
    ax_co.set_title("C  Same-song\nco-mentions", loc="left", fontsize=8.3, fontweight="bold", pad=28)
    ax_co.text(
        0.0,
        1.055,
        "All 5,681 eligible songs",
        transform=ax_co.transAxes,
        fontsize=5.3,
        color=MUTED,
        va="bottom",
    )
    style_axis(ax_co, "x")

    high_handle = mpl.lines.Line2D([], [], color=BLUE, marker="o", markerfacecolor=BLUE, linestyle="none", label="High reliability")
    supported_handle = mpl.lines.Line2D([], [], color=ORANGE, marker="s", markerfacecolor=PAPER, linestyle="none", label="Supported")
    ax_forest.legend(
        handles=[high_handle, supported_handle],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.105),
        ncol=2,
        handletextpad=0.35,
        columnspacing=1.1,
        borderaxespad=0,
    )

    fig.text(0.04, 0.066, "Human gold: 0 completed reviews. Precision, recall, and F1 are not reported.", fontsize=5.95, color=MUTED)
    fig.text(
        0.04,
        0.047,
        "Claim boundary: lyric-reference signals only — not residence, preference, biography, collaboration, influence, or social ties.",
        fontsize=5.95,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.021,
        "Source: frozen public Chinese rap NER/cultural-reference release v1.1.0; association units are distinct shared-text-excluded song memberships.",
        fontsize=5.7,
        color=MUTED,
    )

    png_path = OUTPUT_DIR / f"{FIGURE_STEM}.png"
    tiff_path = OUTPUT_DIR / f"{FIGURE_STEM}.tiff"
    fig.savefig(png_path, dpi=DPI, facecolor=PAPER, edgecolor="none")
    plt.close(fig)
    with Image.open(png_path) as image:
        image.convert("RGB").save(tiff_path, format="TIFF", dpi=(DPI, DPI), compression="tiff_lzw")

    expected_pixels = [int(round(FIGSIZE[0] * DPI)), int(round(FIGSIZE[1] * DPI))]
    return {
        "png": rel(png_path),
        "tiff": rel(tiff_path),
        "print_size_inches": list(FIGSIZE),
        "expected_pixels_at_600_dpi": expected_pixels,
    }


def figure3_caption_section() -> str:
    return """## Figure 3 — Cultural-reference evidence after leakage and uncertainty controls

**Caption.** Fig. 3 Provisional cultural-reference evidence after cross-label exact shared-text exclusion and statistical screening. Panel A keeps three analytical units separate. The candidate inventory falls from 33 corpus-wide entity strings to 23 after shared-text exclusion and to 22 in the fixed 204-label primary universe. Legacy source-label/entity links fall from 85 to 40 after shared-text exclusion; shrinkage, conservative interval, and Benjamini–Hochberg false-discovery-rate gates retain six source-label-to-PLACE enrichments. Legacy co-mention sensitivity used an entity-bearing-song denominator (9 before exclusion; 1 after) and is shown separately from the corrected primary denominator of all 5,681 eligible song units (5 basic-gate candidates; 4 BH-FDR releases). Panel B plots all six released shrunken risk ratios with conservative 95% intervals and BH-adjusted q values. Panel C plots the four released same-song reference co-mentions by normalized pointwise mutual information (NPMI). Human review remains incomplete (0 completed gold reviews), so precision, recall, and F1 are not reported. All findings concern lyric references, not residence, preference, biography, collaboration, influence, or social relationships.

**Alt text.** Three evidence-control cards show separate progressions for entity strings, source-label/entity pairs, and co-mention pairs. The entity inventory decreases from 33 to 23 to 22. Label-to-reference links decrease from 85 to 40 to 6. The co-mention card separates the legacy entity-bearing denominator, which changes from 9 to 1, from the corrected all-5,681-song denominator, which changes from 5 to 4. A log-scale forest plot shows six released source-label-to-place enrichments; every conservative interval is above one. The strongest point estimates are 泰格西 to 湖南 and 黑麦 to 天津, while GALI and 法老 both link to 上海 with smaller but supported enrichments. A compact dot-bar panel shows four released same-song co-mentions: 伦敦–巴黎, 中文–英文, 上海–新疆, and 上海–巴黎. A note states that human gold is incomplete and that the edges are lyric-reference evidence, not biographical or social relations.

**Takeaway.** Shared-text exclusion and uncertainty control materially reduce the candidate graph; the surviving edges are a small, interpretable set of corpus-internal cultural-reference signals with explicit claim boundaries.

"""


def upsert_caption(captions_path: Path) -> None:
    text = captions_path.read_text(encoding="utf-8")
    section = figure3_caption_section()
    if "## Figure 3 —" in text:
        text = re.sub(r"## Figure 3 —.*?(?=## Figure 4 —|\Z)", section, text, flags=re.DOTALL)
    elif "## Figure 4 —" in text:
        text = text.replace("## Figure 4 —", section + "## Figure 4 —", 1)
    else:
        text = text.rstrip() + "\n\n" + section
    captions_path.write_text(text, encoding="utf-8", newline="\n")


def validate_and_update_metadata(
    controls: list[dict[str, Any]],
    links: list[dict[str, Any]],
    co_mentions: list[dict[str, Any]],
    export: dict[str, Any],
    protected_hashes: dict[str, str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    actual_controls = read_csv(OUTPUT_DIR / "figure_3_evidence_controls_source.csv")
    actual_links = read_csv(OUTPUT_DIR / "figure_3_label_place_links_source.csv")
    actual_co_mentions = read_csv(OUTPUT_DIR / "figure_3_co_mentions_source.csv")

    record(
        "figure_3_evidence_control_counts_exact",
        [int(row["value"]) for row in actual_controls] == [33, 23, 22, 85, 40, 6, 9, 1, 5, 4],
        {"observed": [int(row["value"]) for row in actual_controls]},
    )
    record(
        "figure_3_six_released_label_place_links",
        len(actual_links) == 6 and all(row["entity_type"] == "PLACE" for row in actual_links),
        {"rows": len(actual_links), "types": sorted({row["entity_type"] for row in actual_links})},
    )
    record(
        "figure_3_label_link_intervals_and_q_pass",
        all(float(row["ci95_lower_conservative"]) > 1 and float(row["q_value_bh"]) <= 0.05 for row in actual_links),
        {"minimum_lower": min(float(row["ci95_lower_conservative"]) for row in actual_links), "maximum_q": max(float(row["q_value_bh"]) for row in actual_links)},
    )
    record(
        "figure_3_four_released_co_mentions_correct_denominator",
        len(actual_co_mentions) == 4 and all(int(row["all_eligible_song_units"]) == 5681 and float(row["q_value_bh"]) <= 0.05 for row in actual_co_mentions),
        {"rows": len(actual_co_mentions), "denominators": sorted({int(row["all_eligible_song_units"]) for row in actual_co_mentions})},
    )

    image_details: list[dict[str, Any]] = []
    image_ok = True
    expected_pixels = tuple(export["expected_pixels_at_600_dpi"])
    for format_key in ("png", "tiff"):
        path = ROOT / export[format_key]
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            detail = {
                "format": format_key,
                "path": rel(path),
                "pixels": list(image.size),
                "mode": image.mode,
                "dpi": [float(dpi[0]), float(dpi[1])] if isinstance(dpi, tuple) and len(dpi) >= 2 else [0.0, 0.0],
            }
            image_details.append(detail)
            format_ok = image.size == expected_pixels
            if format_key == "tiff":
                format_ok = format_ok and image.mode == "RGB" and min(detail["dpi"]) >= 599.0
            image_ok = image_ok and format_ok
    record("figure_3_image_dimensions_and_tiff_600dpi", image_ok, image_details)

    protected_after = {name: sha256(OUTPUT_DIR / name) for name in protected_hashes}
    record("figures_1_2_4_unchanged", protected_hashes == protected_after, {"before": protected_hashes, "after": protected_after})

    forbidden_fragments = ["song_id", "chunk_id", "lyric", "analysis_text", "content_hash", "embedding", "private"]
    privacy_hits: list[dict[str, str]] = []
    for path in sorted(OUTPUT_DIR.glob("figure_3_*_source.csv")):
        rows = read_csv(path)
        headers = list(rows[0].keys()) if rows else []
        for header in headers:
            if any(fragment in header.lower() for fragment in forbidden_fragments):
                privacy_hits.append({"path": rel(path), "header": header})
    record("figure_3_aggregate_only_privacy_schema_scan", not privacy_hits, {"forbidden_header_hits": privacy_hits})

    old_validation_path = OUTPUT_DIR / "validation.json"
    old_validation = read_json(old_validation_path)
    old_checks = [check for check in old_validation.get("checks", []) if not check.get("name", "").startswith("figure_3_") and check.get("name") != "figures_1_2_4_unchanged"]
    combined_checks = old_checks + checks
    validation = {
        "artifact_id": "chinese-rap-downstream-figures-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass" if all(check.get("passed") for check in combined_checks) else "fail",
        "confidence": "ready_to_share" if all(check.get("passed") for check in combined_checks) else "needs_revision",
        "claim_boundary": "Figures report corpus-internal aggregate evidence. They do not establish performer identity, authorship, biography, residence, preference, social relations, performed rhyme, flow, delivery, or audio properties.",
        "checks": combined_checks,
    }
    write_json(old_validation_path, validation)
    if validation["status"] != "pass":
        raise AssertionError("Figure-pack validation failed after adding Figure 3")
    return validation


def update_manifest(export: dict[str, Any], validation: dict[str, Any]) -> None:
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["version"] = "1.1.0"
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["status"] = validation["status"]
    figures = manifest.setdefault("figures", {})
    figures["figure_3"] = export
    manifest["figures"] = {key: figures[key] for key in ["figure_1", "figure_2", "figure_3", "figure_4"] if key in figures}
    manifest["claim_boundary"] = validation["claim_boundary"]
    lineage = manifest.setdefault("lineage", {})
    lineage["figure_3_builder"] = {"path": rel(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}
    old_sources = {entry["path"]: entry for entry in lineage.get("sources", [])}
    for path in [NER_SUMMARY, NER_SENSITIVITY, NER_LABEL_LINKS, NER_CO_MENTIONS, NER_VALIDATION, NER_RECONCILIATION]:
        old_sources[rel(path)] = {"path": rel(path), "sha256": sha256(path)}
    lineage["sources"] = [old_sources[key] for key in sorted(old_sources)]
    manifest["files"] = [
        {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(OUTPUT_DIR.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest["privacy"] = "Aggregate-only figures and source tables; no lyric text, song/chunk IDs, content hashes, embeddings, or private audit rows."
    write_json(manifest_path, manifest)


def main() -> None:
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(OUTPUT_DIR)
    protected_names = [
        "figure_1_research_design.png",
        "figure_1_research_design.tiff",
        "figure_2_retrieval_benchmark.png",
        "figure_2_retrieval_benchmark.tiff",
        "figure_4_written_rhyme_benchmark.png",
        "figure_4_written_rhyme_benchmark.tiff",
    ]
    protected_hashes = {name: sha256(OUTPUT_DIR / name) for name in protected_names}

    summary, sensitivity, raw_links, raw_co_mentions = load_and_validate_sources()
    controls = evidence_control_rows(sensitivity)
    links = normalized_label_link_rows(raw_links)
    co_mentions = normalized_co_mention_rows(raw_co_mentions)

    write_csv(
        OUTPUT_DIR / "figure_3_evidence_controls_source.csv",
        controls,
        ["dimension", "denominator_definition", "stage_order", "stage", "value", "unit", "meaning", "source"],
    )
    write_csv(
        OUTPUT_DIR / "figure_3_label_place_links_source.csv",
        links,
        [
            "source_credit_label",
            "entity",
            "entity_type",
            "label_song_units",
            "entity_song_units_within_label",
            "entity_song_units_corpus",
            "all_membership_song_units",
            "within_label_share",
            "corpus_share",
            "shrunken_risk_ratio",
            "ci95_lower_conservative",
            "ci95_upper_conservative",
            "q_value_bh",
            "reliability_class",
            "association_scope",
            "source",
        ],
    )
    write_csv(
        OUTPUT_DIR / "figure_3_co_mentions_source.csv",
        co_mentions,
        [
            "entity_a",
            "entity_a_type",
            "entity_b",
            "entity_b_type",
            "all_eligible_song_units",
            "unique_song_unit_co_mentions",
            "source_credit_labels",
            "npmi",
            "q_value_bh",
            "reliability_class",
            "relation_scope",
            "source",
        ],
    )

    export = build_figure(controls, links, co_mentions)

    contracts_path = OUTPUT_DIR / "chart_contracts.json"
    contracts = read_json(contracts_path)
    contracts["figure_3"] = FIGURE3_CONTRACT
    ordered_contracts = {key: contracts[key] for key in ["figure_1", "figure_2", "figure_3", "figure_4"] if key in contracts}
    write_json(contracts_path, ordered_contracts)
    upsert_caption(OUTPUT_DIR / "figure_captions_and_alt_text.md")

    validation = validate_and_update_metadata(controls, links, co_mentions, export, protected_hashes)
    update_manifest(export, validation)

    result = {
        "status": "pass",
        "output_dir": rel(OUTPUT_DIR),
        "figure": export,
        "source_counts": {
            "entities": summary["counts"]["public_provisional_entities"],
            "label_place_links": len(links),
            "co_mentions": len(co_mentions),
            "eligible_song_units": summary["graph_analysis"]["eligible_global_song_units"],
            "completed_human_gold": 0,
        },
        "protected_figures_unchanged": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
