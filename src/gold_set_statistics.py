"""Estimators for a stratified cluster sample scored line by line.

The MB-001 gold set draws chunks, scores lines, and weights by inverse sampling
probability. That makes almost every quantity of interest a *combined ratio* estimator whose
numerator and denominator are both weighted sums over clusters -- and combined ratios
estimated from a handful of informative clusters are biased, skewed, and badly served by a
naive percentile bootstrap. The functions here are separated from the reporting tool so the
arithmetic and, more importantly, the *calibration* can be exercised against cases whose
answers are known: no pandas, no numpy, no test framework.

Two things here are unusual and deliberate:

* ``bootstrap`` returns the whole draw distribution rather than an interval, so bias, spread
  and interval can be read off the same draws and cannot silently disagree.
* ``coverage_simulation`` measures what a nominal interval actually covers under this
  design. It exists because the answer for this sample is not 95 percent, and a number that
  is labelled 95 percent without being checked is worse than no number.
"""

from __future__ import annotations

import random
import statistics
from typing import Callable, Iterable, Mapping, Sequence

# a scored cluster: its inverse-probability weight and its confusion cells
Cluster = tuple[float, Mapping[str, int]]

CELLS = ("tp", "fp", "fn", "tn", "abstained")


def combine(clusters: Iterable[Cluster]) -> dict[str, float]:
    """Sum weighted confusion cells over clusters."""
    totals = {name: 0.0 for name in CELLS}
    for weight, cells in clusters:
        for name in CELLS:
            totals[name] += weight * cells[name]
    return totals


def line_rates(totals: Mapping[str, float]) -> dict[str, float | None]:
    """Precision, recall and F1 over lines.

    ``None`` rather than 0.0 when a denominator is empty: an empty denominator means the
    sample carries no information about that rate, and reporting ignorance as a measured
    zero is the specific error this guards against.
    """
    predicted_positive = totals["tp"] + totals["fp"]
    actual_positive = totals["tp"] + totals["fn"]
    precision = totals["tp"] / predicted_positive if predicted_positive else None
    recall = totals["tp"] / actual_positive if actual_positive else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def chunk_rates(clusters: Iterable[Cluster]) -> dict[str, float | None]:
    """Chunk-level view: is a metadata-bearing chunk noticed at all?

    Line recall answers "how much of the credit block gets removed". This answers "how often
    does the block get noticed", which is the number that matters when flagged chunks are
    routed to a human rather than deleted. A detector looks far better here, which is exactly
    why both belong in the report.
    """
    contains = detected = flagged = clean_flagged = 0.0
    for weight, cells in clusters:
        has_metadata = cells["tp"] + cells["fn"] > 0
        fired = cells["tp"] + cells["fp"] > 0
        if has_metadata:
            contains += weight
            if fired:
                detected += weight
        elif fired:
            clean_flagged += weight
        if fired:
            flagged += weight
    return {
        "detection_rate": detected / contains if contains else None,
        "flag_precision": detected / flagged if flagged else None,
        "chunks_containing_metadata": contains,
        "chunks_flagged": flagged,
        "clean_chunks_flagged": clean_flagged,
    }


def positive_share(clusters: Iterable[Cluster]) -> dict[str, float | None]:
    """Share of scored lines the rater called metadata.

    This is a property of the sampling frame, not of the corpus. The frame here is chunks of
    a bounded line count, which is a minority of corpus lines and is enriched in exactly the
    short pure-credit shape being counted, so calling the result a corpus rate overstates it.
    The caller is responsible for saying which frame; this function only measures it.
    """
    positive = scored = 0.0
    for weight, cells in clusters:
        positive += weight * (cells["tp"] + cells["fn"])
        scored += weight * (cells["tp"] + cells["fp"] + cells["fn"] + cells["tn"])
    return {
        "metadata_line_share": positive / scored if scored else None,
        "weighted_metadata_lines": positive,
        "weighted_scored_lines": scored,
    }


def all_metrics(clusters: Sequence[Cluster]) -> dict[str, float | None]:
    """Every published estimand, from one pass over the clusters."""
    metrics = dict(line_rates(combine(clusters)))
    metrics.update(chunk_rates(clusters))
    metrics.update(positive_share(clusters))
    return metrics


def informative_counts(strata: Mapping[str, Sequence[Cluster]]) -> dict[str, dict[str, int]]:
    """How many clusters per stratum actually carry signal.

    A recall estimate resting on four clusters out of forty is a different object from one
    resting on forty, and the confusion cells alone do not show which it is: ten missed lines
    are consistent with one cluster or with ten. Publishing this is what lets a reader see
    that the interval is wide for a structural reason.
    """
    counts: dict[str, dict[str, int]] = {}
    for name, clusters in strata.items():
        counts[name] = {
            "sampled": len(clusters),
            "containing_metadata": sum(1 for _, c in clusters if c["tp"] + c["fn"] > 0),
            "with_a_missed_line": sum(1 for _, c in clusters if c["fn"] > 0),
            "flagged": sum(1 for _, c in clusters if c["tp"] + c["fp"] > 0),
        }
    return counts


def resample(strata: Mapping[str, Sequence[Cluster]], rng: random.Random) -> list[Cluster]:
    """One stratified bootstrap draw: clusters with replacement, within each stratum."""
    drawn: list[Cluster] = []
    for clusters in strata.values():
        if clusters:
            drawn.extend(rng.choices(clusters, k=len(clusters)))
    return drawn


def bootstrap(
    strata: Mapping[str, Sequence[Cluster]],
    draws: int,
    seed: str,
    metrics: Sequence[str],
) -> dict[str, list[float]]:
    """The full draw distribution for each metric, not a summary of it.

    Clusters are the sampling unit, so clusters are what gets resampled. Resampling lines
    would treat a twelve-line credits block as twelve independent observations and produce
    intervals several times too narrow.
    """
    rng = random.Random(seed)
    collected: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(draws):
        drawn = all_metrics(resample(strata, rng))
        for name in metrics:
            value = drawn.get(name)
            if value is not None:
                collected[name].append(value)
    return collected


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(fraction * (len(ordered) - 1))))]


def summarise_draws(
    values: Sequence[float], point: float | None, low: float = 0.025, high: float = 0.975
) -> dict[str, float] | None:
    """Interval, spread and bias from one metric's draws.

    Bias is reported rather than removed. For a combined ratio on few clusters the bootstrap
    bias runs to a quarter of the estimate, and the bias-corrected point disagrees with
    itself depending on whether F1 is corrected directly or recomputed from corrected parts
    -- which is an argument for showing the reader the bias, not for quietly subtracting it.
    """
    if not values or point is None:
        return None
    mean = statistics.fmean(values)
    return {
        "low": percentile(values, low),
        "high": percentile(values, high),
        "median": statistics.median(values),
        "mean": mean,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "bias": mean - point,
        "relative_bias": (mean - point) / point if point else None,
        "draws_used": len(values),
    }


def coverage_simulation(
    strata: Mapping[str, Sequence[Cluster]],
    metrics: Sequence[str],
    outer: int,
    inner: int,
    seed: str,
    nominal_low: float = 0.025,
    nominal_high: float = 0.975,
) -> dict[str, dict[str, float]]:
    """Measure what the nominal interval actually covers, under this design.

    Plug-in: treat the observed clusters as the population, redraw a sample of the same
    stratum sizes, build the same percentile interval from it, and count how often the
    interval contains the population value. If the answer is not close to the nominal level
    then the interval must not be published at that level, which is the whole reason this
    runs. Failures are counted separately by direction, because an interval that misses
    upward and one that misses downward mislead a reader in opposite ways.
    """
    truth = all_metrics([c for clusters in strata.values() for c in clusters])
    rng = random.Random(seed)
    tally = {
        name: {"covered": 0, "attempts": 0, "entirely_above_truth": 0, "entirely_below_truth": 0}
        for name in metrics
    }
    for _ in range(outer):
        sample_strata = {
            name: rng.choices(clusters, k=len(clusters)) for name, clusters in strata.items()
        }
        drawn = bootstrap(sample_strata, inner, f"{seed}-{rng.random()}", metrics)
        for name in metrics:
            actual = truth.get(name)
            values = drawn[name]
            if actual is None or len(values) < inner // 2:
                continue
            low = percentile(values, nominal_low)
            high = percentile(values, nominal_high)
            tally[name]["attempts"] += 1
            if low <= actual <= high:
                tally[name]["covered"] += 1
            elif low > actual:
                tally[name]["entirely_above_truth"] += 1
            else:
                tally[name]["entirely_below_truth"] += 1
    return {
        name: {
            "achieved_coverage": counts["covered"] / counts["attempts"] if counts["attempts"] else None,
            "misses_entirely_above_truth": (
                counts["entirely_above_truth"] / counts["attempts"] if counts["attempts"] else None
            ),
            "misses_entirely_below_truth": (
                counts["entirely_below_truth"] / counts["attempts"] if counts["attempts"] else None
            ),
            "simulated_intervals": counts["attempts"],
        }
        for name, counts in tally.items()
    }


def leave_one_out(
    strata: Mapping[str, Sequence[Cluster]], metric: str
) -> list[dict[str, float | str]]:
    """Recompute one metric with each cluster dropped, reweighted to the smaller sample.

    A single dropped cluster moving the estimate by most of its own size is the plainest
    possible statement that the estimate rests on that cluster.
    """
    point = all_metrics([c for clusters in strata.values() for c in clusters]).get(metric)
    results: list[dict[str, float | str]] = []
    for name, clusters in strata.items():
        if len(clusters) < 2:
            continue
        scale = len(clusters) / (len(clusters) - 1)
        for index in range(len(clusters)):
            kept = [
                (weight * scale, cells)
                for position, (weight, cells) in enumerate(clusters)
                if position != index
            ]
            others = [c for other, group in strata.items() if other != name for c in group]
            value = all_metrics(kept + others).get(metric)
            if value is None or point is None:
                continue
            results.append(
                {"stratum": name, "index": index, "value": value, "ratio_to_point": value / point}
            )
    results.sort(key=lambda item: -abs(float(item["ratio_to_point"]) - 1.0))
    return results


def significant_figures(value: float | None, digits: int = 2) -> float | None:
    """Round to a number of significant figures the design can actually support."""
    if value is None or value == 0:
        return value
    from math import floor, log10

    return round(value, -int(floor(log10(abs(value)))) + (digits - 1))
