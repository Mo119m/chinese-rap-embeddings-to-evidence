#!/usr/bin/env python3
"""Figure source tables for Figures 1, 2 and 4, read from the corpus v2 results.

The journal figure builder draws from small public tables under figures/. Those tables were
written from the v1 results by build_chinese_rap_downstream_figures_v1.py, which is bound to
the v1 artifacts by version checks and is kept as history. This file writes the same tables,
with the same columns and the same system and model names the renderer expects, from
results/retrieval-v2 and results/written-rhyme-v2, so that the figures say what the
manuscript says.

Nothing is computed here. Every number is copied from a published result file, and the
`source` column of every row names the file it came from.

    python src/build_downstream_figure_sources_v2.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "figures"
RETRIEVAL = ROOT / "results" / "retrieval-v2"
RHYME = ROOT / "results" / "written-rhyme-v2"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

METRIC_LABELS = {
    "mrr": "MRR", "recall_at_1": "Recall@1", "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10", "ndcg_at_10": "nDCG@10",
    "top1_accuracy": "Top-1", "top3_accuracy": "Top-3", "top5_accuracy": "Top-5",
}
RETRIEVAL_METRIC_ORDER = ["mrr", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10"]
# the renderer keys its styles on the v1 names; the v2 builder names the same three systems
RETRIEVAL_SYSTEM_NAMES = {
    "BGE-M3 dense": "BGE-M3 dense (strict)",
    "character 2-5 gram TF-IDF": "character 2-5 gram TF-IDF (strict)",
    "equal-weight z-score fusion": "equal-weight z-score fusion (strict)",
}
RETRIEVAL_SYSTEM_SHORT = {
    "BGE-M3 dense (strict)": "BGE-M3 dense",
    "character 2-5 gram TF-IDF (strict)": "Character TF-IDF",
    "equal-weight z-score fusion (strict)": "Dense + lexical fusion",
}
RETRIEVAL_COMPARISONS = {
    "equal-weight z-score fusion minus BGE-M3 dense": ("strict fusion minus strict BGE-M3", "Fusion - BGE-M3"),
    "equal-weight z-score fusion minus character 2-5 gram TF-IDF": ("strict fusion minus strict TF-IDF", "Fusion - TF-IDF"),
}
RHYME_MODELS = [
    "hierarchical_sgd_context", "hierarchical_sgd_no_source_label",
    "flat_sgd_logistic_context", "first_order_markov", "global_frequency",
]
RHYME_MODEL_SHORT = {
    "hierarchical_sgd_context": "Hierarchical + label",
    "hierarchical_sgd_no_source_label": "Hierarchical, no label",
    "flat_sgd_logistic_context": "Flat context",
    "first_order_markov": "First-order Markov",
    "global_frequency": "Global frequency",
}
RHYME_METRIC_ORDER = ["top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")


def figure_1_rows() -> list[dict]:
    summary = json.loads((RETRIEVAL / "analysis_summary.json").read_text(encoding="utf-8"))
    corpus, unit = summary["corpus"], summary["leakage_unit"]
    source = rel(RETRIEVAL / "analysis_summary.json")
    singleton_groups = unit["groups"] - unit["multi_song_groups"]
    items = [
        ("corpus", "songs", corpus["song_records"], "song records"),
        ("corpus", "canonical_chunks", corpus["chunks"], "lyric chunks"),
        ("corpus", "eligible_clean_text_chunks", corpus["chunks"], "lyric chunks retained; the duplicate control is a grouping, not a deletion"),
        ("controls", "exact_song_content_groups_spanning_songs", unit["multi_song_groups"], "leakage groups holding more than one song"),
        ("controls", "songs_in_spanning_exact_groups", corpus["queries_after_length_filter"] - singleton_groups, "query songs inside a multi-song leakage group"),
        ("controls", "leakage_groups", unit["groups"], "leakage groups over the query population"),
        ("controls", "query_songs", corpus["queries_after_length_filter"], "query songs after the length filter"),
        ("controls", "eligible_labels", corpus["eligible_labels"], "source-credit labels with at least five query songs"),
    ]
    return [{"component": component, "branch": "", "item": item, "value": value, "unit": unit_text,
             "source": source} for component, item, value, unit_text in items]


def figure_2_rows() -> tuple[list[dict], list[dict]]:
    metrics_path, deltas_path = RETRIEVAL / "metrics.csv", RETRIEVAL / "uncertainty.csv"
    metrics = []
    for row in read_csv(metrics_path):
        if row["aggregation"] != "source_credit_label_macro_duplicate_group_adjusted":
            continue
        if row["system"] not in RETRIEVAL_SYSTEM_NAMES or row["metric"] not in RETRIEVAL_METRIC_ORDER:
            continue
        system = RETRIEVAL_SYSTEM_NAMES[row["system"]]
        metrics.append({
            "system": system, "system_short": RETRIEVAL_SYSTEM_SHORT[system],
            "metric": row["metric"], "metric_label": METRIC_LABELS[row["metric"]],
            "estimate": row["estimate"], "ci95_lower": row["ci95_lower"], "ci95_upper": row["ci95_upper"],
            "queries": row["queries"], "source_credit_labels": row["source_credit_labels"],
            "global_duplicate_components": row["global_duplicate_components"],
            "aggregation": row["aggregation"], "bootstrap_replicates": row["bootstrap_replicates"],
            "source": rel(metrics_path),
        })
    deltas = []
    for row in read_csv(deltas_path):
        if row["comparison"] not in RETRIEVAL_COMPARISONS or row["metric"] not in RETRIEVAL_METRIC_ORDER:
            continue
        name, short = RETRIEVAL_COMPARISONS[row["comparison"]]
        deltas.append({
            "comparison": name, "comparison_short": short,
            "metric": row["metric"], "metric_label": METRIC_LABELS[row["metric"]],
            "estimate_delta": row["estimate_delta"], "ci95_lower": row["ci95_lower"],
            "ci95_upper": row["ci95_upper"], "interval_direction": row["interval_direction"],
            "bootstrap_replicates": row["paired_two_stage_bootstrap_replicates"],
            "source": rel(deltas_path),
        })
    if len(metrics) != 3 * len(RETRIEVAL_METRIC_ORDER):
        raise AssertionError(f"expected {3 * len(RETRIEVAL_METRIC_ORDER)} retrieval rows, found {len(metrics)}")
    if len(deltas) != 2 * len(RETRIEVAL_METRIC_ORDER):
        raise AssertionError(f"expected {2 * len(RETRIEVAL_METRIC_ORDER)} delta rows, found {len(deltas)}")
    return metrics, deltas


def figure_4_rows() -> tuple[list[dict], list[dict], list[dict]]:
    metrics_path = RHYME / "model_metrics.csv"
    strata_path = RHYME / "stratified_metrics.csv"
    deltas_path = RHYME / "paired_model_deltas.csv"
    benchmark = []
    for row in read_csv(metrics_path):
        if row["evaluation_split"] != "song_held_out_test" or row["model"] not in RHYME_MODELS:
            continue
        for metric in RHYME_METRIC_ORDER:
            benchmark.append({
                "model": row["model"], "model_short": RHYME_MODEL_SHORT[row["model"]],
                "metric": metric, "metric_label": METRIC_LABELS[metric],
                "estimate": row[metric], "ci95_lower": row[f"{metric}_ci95_low"],
                "ci95_upper": row[f"{metric}_ci95_high"],
                "leakage_safe_event_count": row["leakage_safe_event_count"],
                "song_count": row["song_count"],
                "end_to_end_event_coverage": row["end_to_end_event_coverage"],
                "bootstrap_replicates": 2000, "source": rel(metrics_path),
            })
    transition = []
    for row in read_csv(strata_path):
        if row["stratum_dimension"] != "transition_type" or row["stratum_value"] not in ("continuation", "switch"):
            continue
        if row["model"] not in ("first_order_markov", "hierarchical_sgd_context"):
            continue
        transition.append({
            "model": row["model"], "model_short": RHYME_MODEL_SHORT[row["model"]],
            "transition_type": row["stratum_value"],
            "eligible_event_count": row["eligible_event_count"], "song_count": row["song_count"],
            "top1_accuracy": row["top1_accuracy"], "top3_accuracy": row["top3_accuracy"],
            "mrr": row["mrr"], "source": rel(strata_path),
        })
    personalization = []
    for row in read_csv(deltas_path):
        if row["released_model"] != "hierarchical_sgd_context":
            continue
        if row["reference_model"] != "hierarchical_sgd_no_source_label" or row["metric"] not in RHYME_METRIC_ORDER:
            continue
        personalization.append({
            "released_model": row["released_model"], "reference_model": row["reference_model"],
            "metric": row["metric"], "metric_label": METRIC_LABELS[row["metric"]],
            "estimate_delta": row["paired_difference_released_minus_reference"],
            "ci95_lower": row["song_cluster_bootstrap_ci95_low"],
            "ci95_upper": row["song_cluster_bootstrap_ci95_high"],
            "bootstrap_replicates": row["bootstrap_replicates"], "source": rel(deltas_path),
        })
    if len(benchmark) != len(RHYME_MODELS) * len(RHYME_METRIC_ORDER):
        raise AssertionError(f"expected {len(RHYME_MODELS) * len(RHYME_METRIC_ORDER)} rhyme rows, found {len(benchmark)}")
    if len(transition) != 4 or len(personalization) != len(RHYME_METRIC_ORDER):
        raise AssertionError(f"rhyme strata {len(transition)} / personalization {len(personalization)} rows")
    return benchmark, transition, personalization


def main() -> int:
    write_csv(FIGURE_DIR / "figure_1_pipeline_source.csv", figure_1_rows(),
              ["component", "branch", "item", "value", "unit", "source"])
    metrics, deltas = figure_2_rows()
    write_csv(FIGURE_DIR / "figure_2_retrieval_benchmark_source.csv", metrics,
              ["system", "system_short", "metric", "metric_label", "estimate", "ci95_lower", "ci95_upper",
               "queries", "source_credit_labels", "global_duplicate_components", "aggregation",
               "bootstrap_replicates", "source"])
    write_csv(FIGURE_DIR / "figure_2_retrieval_deltas_source.csv", deltas,
              ["comparison", "comparison_short", "metric", "metric_label", "estimate_delta", "ci95_lower",
               "ci95_upper", "interval_direction", "bootstrap_replicates", "source"])
    benchmark, transition, personalization = figure_4_rows()
    write_csv(FIGURE_DIR / "figure_4_rhyme_benchmark_source.csv", benchmark,
              ["model", "model_short", "metric", "metric_label", "estimate", "ci95_lower", "ci95_upper",
               "leakage_safe_event_count", "song_count", "end_to_end_event_coverage", "bootstrap_replicates",
               "source"])
    write_csv(FIGURE_DIR / "figure_4_transition_source.csv", transition,
              ["model", "model_short", "transition_type", "eligible_event_count", "song_count",
               "top1_accuracy", "top3_accuracy", "mrr", "source"])
    write_csv(FIGURE_DIR / "figure_4_personalization_deltas_source.csv", personalization,
              ["released_model", "reference_model", "metric", "metric_label", "estimate_delta",
               "ci95_lower", "ci95_upper", "bootstrap_replicates", "source"])
    print("wrote figure 1, 2 and 4 source tables from the v2 results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
