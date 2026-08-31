"""Checks for the stratified-cluster estimators behind the MB-001 gold set.

These functions turn 80 labelled chunks into numbers that go into a paper. The failure modes
that matter are not crashes: a weighting that silently cancels, a ratio whose bias is not
reported, an interval labelled 95 percent that covers 80, an estimate resting on four
clusters that reads like it rests on forty. Each of those is a case below with an answer
worked out by hand or forced by construction.

No private data, no third-party dependencies, no test framework:

    python tests/test_gold_set_statistics.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gs = load("gold_set_statistics")


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{': ' + detail if detail else ''}")
        FAILURES.append(label)


def cluster(weight: float, tp: int = 0, fp: int = 0, fn: int = 0, tn: int = 0, abstained: int = 0):
    return (weight, {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "abstained": abstained})


# ------------------------------------------------------------------------------- estimands


def test_line_rates() -> None:
    print("line_rates")
    perfect = gs.line_rates({"tp": 4.0, "fp": 0.0, "fn": 0.0, "tn": 6.0, "abstained": 0.0})
    check("perfect agreement is 1.0 throughout",
          perfect["precision"] == perfect["recall"] == perfect["f1"] == 1.0)
    half = gs.line_rates({"tp": 1.0, "fp": 1.0, "fn": 1.0, "tn": 1.0, "abstained": 0.0})
    check("precision and recall of one half each", half["precision"] == half["recall"] == 0.5)
    check("F1 is the harmonic mean", half["f1"] == 0.5)
    lopsided = gs.line_rates({"tp": 1.0, "fp": 0.0, "fn": 3.0, "tn": 0.0, "abstained": 0.0})
    check("F1 punishes a lopsided pair, unlike the arithmetic mean",
          abs(lopsided["f1"] - 0.4) < 1e-12 and lopsided["f1"] < (1.0 + 0.25) / 2,
          str(lopsided["f1"]))
    barren = gs.line_rates({"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 5.0, "abstained": 0.0})
    check("no positives anywhere reports None, never zero",
          barren["precision"] is None and barren["recall"] is None and barren["f1"] is None)


def test_weighting_actually_bites() -> None:
    print("weighting")
    # one light cluster the detector got right, one heavy cluster it missed entirely
    light = cluster(1.0, tp=1)
    heavy = cluster(9.0, fn=1)
    flat = gs.line_rates(gs.combine([(1.0, light[1]), (1.0, heavy[1])]))
    weighted = gs.line_rates(gs.combine([light, heavy]))
    check("a heavier stratum drags recall down",
          flat["recall"] == 0.5 and weighted["recall"] == 0.1,
          f"{flat['recall']} then {weighted['recall']}")
    # the property that made the tool's own docstring wrong: when every TP and FP sits in one
    # stratum, its common weight cancels and weighting cannot move precision at all
    single = [cluster(7.875, tp=56, fp=13), cluster(305.075, fn=10, tn=228)]
    flat_single = [(1.0, cells) for _, cells in single]
    check("weighting leaves precision untouched when all firings are in one stratum",
          gs.line_rates(gs.combine(single))["precision"]
          == gs.line_rates(gs.combine(flat_single))["precision"])
    check("but it does move recall in the same data",
          gs.line_rates(gs.combine(single))["recall"]
          != gs.line_rates(gs.combine(flat_single))["recall"])


def test_chunk_rates() -> None:
    print("chunk_rates")
    clusters = [
        cluster(1.0, tp=1, fn=1),   # metadata present, detector fired: detected
        cluster(1.0, fn=2),         # metadata present, detector silent: missed
        cluster(1.0, fp=1, tn=3),   # no metadata, detector fired: a false alarm
        cluster(1.0, tn=4),         # no metadata, detector silent
    ]
    result = gs.chunk_rates(clusters)
    check("detection rate counts noticed-at-all, not lines", result["detection_rate"] == 0.5)
    check("flag precision counts flagged chunks that really had metadata",
          result["flag_precision"] == 0.5)
    check("a clean chunk that fires is counted", result["clean_chunks_flagged"] == 1.0)
    check("chunks containing metadata is a weighted count", result["chunks_containing_metadata"] == 2.0)
    heavy = gs.chunk_rates([cluster(1.0, tp=1), cluster(50.0, fn=1)])
    check("an unnoticed heavy chunk collapses the detection rate",
          abs(heavy["detection_rate"] - 1 / 51) < 1e-12, str(heavy["detection_rate"]))


def test_positive_share() -> None:
    print("positive_share")
    share = gs.positive_share([cluster(1.0, tp=1, fn=1, tn=8)])
    check("share counts rater positives over scored lines", share["metadata_line_share"] == 0.2)
    check("abstentions are outside the denominator",
          gs.positive_share([cluster(1.0, tp=1, tn=1, abstained=8)])["metadata_line_share"] == 0.5)
    check("nothing scored gives None, not zero",
          gs.positive_share([cluster(1.0, abstained=3)])["metadata_line_share"] is None)


# ------------------------------------------------------------------------ what the sample rests on


def test_informative_counts() -> None:
    print("informative_counts")
    strata = {
        "negative": [cluster(305.0, fn=6), cluster(305.0, fn=2)] + [cluster(305.0, tn=5)] * 38,
        "positive": [cluster(7.9, tp=2, fp=1)] * 28 + [cluster(7.9, fp=1, tn=4)] * 12,
    }
    counts = gs.informative_counts(strata)
    check("counts the clusters that actually carry a missed line",
          counts["negative"]["with_a_missed_line"] == 2, str(counts["negative"]))
    check("ten missed lines from two clusters is visible as two",
          counts["negative"]["containing_metadata"] == 2)
    check("counts flagged clusters separately",
          counts["positive"]["flagged"] == 40 and counts["negative"]["flagged"] == 0)
    check("sampled size is reported alongside", counts["negative"]["sampled"] == 40)


def test_leave_one_out() -> None:
    print("leave_one_out")
    # one cluster carries almost all of the recall denominator
    strata = {
        "heavy": [cluster(100.0, fn=9)] + [cluster(100.0, tn=5)] * 3,
        "light": [cluster(1.0, tp=1)] * 4,
    }
    loo = gs.leave_one_out(strata, "recall")
    check("the dominating cluster is ranked first",
          loo[0]["stratum"] == "heavy" and loo[0]["index"] == 0, str(loo[0]))
    check("dropping it moves recall by most of its own size",
          loo[0]["ratio_to_point"] > 3.0, str(loo[0]["ratio_to_point"]))
    stable = gs.leave_one_out({"a": [cluster(1.0, tp=1, fn=1)] * 10}, "recall")
    check("a sample with no dominating cluster barely moves",
          all(abs(float(item["ratio_to_point"]) - 1.0) < 1e-9 for item in stable))


# ------------------------------------------------------------------------------- calibration


def test_bootstrap_and_summary() -> None:
    print("bootstrap / summarise_draws")
    # clusters must differ from each other, or every resample returns the same totals and
    # the seed cannot possibly matter -- an earlier version of this fixture made exactly that
    # mistake and the seed check failed against correct code
    strata = {
        "a": [cluster(1.0, tp=i % 3, fn=(i + 1) % 2, tn=i % 4) for i in range(10)],
        "b": [cluster(5.0, fn=i % 2, tn=1 + i % 3) for i in range(10)],
    }
    draws = gs.bootstrap(strata, 400, "test", ["recall", "precision"])
    check("every draw is collected", len(draws["recall"]) == 400)
    check("bootstrap is deterministic under a fixed seed",
          gs.bootstrap(strata, 400, "test", ["recall"])["recall"] == draws["recall"])
    check("a different seed gives a different distribution",
          gs.bootstrap(strata, 400, "other", ["recall"])["recall"] != draws["recall"])

    point = gs.all_metrics([c for group in strata.values() for c in group])["recall"]
    summary = gs.summarise_draws(draws["recall"], point)
    check("the point estimate sits inside its own interval",
          summary["low"] <= point <= summary["high"], f"{summary} vs {point}")
    check("bias is reported, not removed", "bias" in summary and "relative_bias" in summary)
    check("spread is reported", summary["sd"] >= 0)
    constant = gs.summarise_draws([0.5] * 50, 0.5)
    check("no variation gives a zero-width interval and zero bias",
          constant["low"] == constant["high"] == 0.5 and constant["bias"] == 0.0)
    check("no draws gives no summary", gs.summarise_draws([], 0.5) is None)
    check("no point estimate gives no summary", gs.summarise_draws([0.1], None) is None)


def test_coverage_simulation() -> None:
    print("coverage_simulation")
    # A well-behaved design: signal spread evenly over many equally weighted clusters.
    # The percentile interval should land near nominal here.
    spread = {"only": [cluster(1.0, tp=1, fn=1, tn=3) for _ in range(40)]}
    good = gs.coverage_simulation(spread, ["recall"], outer=60, inner=99, seed="spread")
    check("an evenly spread design covers near nominal",
          good["recall"]["achieved_coverage"] > 0.8, str(good["recall"]))

    # The pathological shape this project actually has: the whole recall denominator sits in
    # a couple of heavily weighted clusters. Coverage must visibly fall short of nominal,
    # which is the finding the simulation exists to surface.
    concentrated = {
        "negative": [cluster(305.0, fn=6), cluster(305.0, fn=2)] + [cluster(305.0, tn=5)] * 18,
        "positive": [cluster(7.9, tp=2, fn=1, tn=2)] * 20,
    }
    bad = gs.coverage_simulation(concentrated, ["recall"], outer=60, inner=99, seed="conc")
    check("a design resting on two clusters covers well below nominal",
          bad["recall"]["achieved_coverage"] < good["recall"]["achieved_coverage"],
          f"{bad['recall']['achieved_coverage']} vs {good['recall']['achieved_coverage']}")
    check("misses are counted by direction",
          bad["recall"]["misses_entirely_above_truth"] is not None
          and bad["recall"]["misses_entirely_below_truth"] is not None)
    check("coverage and both miss directions account for every interval",
          abs(bad["recall"]["achieved_coverage"]
              + bad["recall"]["misses_entirely_above_truth"]
              + bad["recall"]["misses_entirely_below_truth"] - 1.0) < 1e-9)
    check("the simulation is deterministic under a fixed seed",
          gs.coverage_simulation(concentrated, ["recall"], outer=60, inner=99, seed="conc") == bad)


def test_significant_figures() -> None:
    print("significant_figures")
    check("rounds to two significant figures", gs.significant_figures(0.108025) == 0.11)
    check("works below a tenth", gs.significant_figures(0.054453) == 0.054)
    check("works above one", gs.significant_figures(1440.8) == 1400.0)
    check("leaves zero alone", gs.significant_figures(0.0) == 0.0)
    check("passes None through", gs.significant_figures(None) is None)
    check("honours a different digit count", gs.significant_figures(0.108025, 3) == 0.108)


def main() -> int:
    for suite in (
        test_line_rates,
        test_weighting_actually_bites,
        test_chunk_rates,
        test_positive_share,
        test_informative_counts,
        test_leave_one_out,
        test_bootstrap_and_summary,
        test_coverage_simulation,
        test_significant_figures,
    ):
        suite()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
