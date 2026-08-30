"""Measure how much label-identifying signal cultural-reference strings carry.

The cultural-reference results currently need the extraction to be *correct*,
which is what the incomplete human review blocks. This asks a different question
that needs no annotation at all: remove the entity strings from the corpus and
see how much held-out retrieval degrades. Ground truth is the source-credit
label, exactly as in the released retrieval evaluation, so the measurement is
automatic. Individual extraction errors change the answer only in proportion to
how much signal they carry, instead of invalidating a claim.

Three conditions are compared:

  baseline  the corpus as released
  entities  every lexicon surface replaced by a neutral filler
  control   randomly chosen character n-grams of the same lengths and comparable
            corpus frequency replaced instead, so the "deleting characters lowers
            similarity" effect is subtracted from the entity condition

The control is imperfect and the output says so: a surface with no frequency-matched
candidate is skipped, so the control can remove fewer characters than the entity
condition. On the full lexicon it matched 559 of 605 surfaces and removed about
10.5% less. Report the result as an adjusted difference under that control, on one
seed and without an interval, not as an attributable effect.

The character TF-IDF arm is always reported. The dense arm's embeddings are
precomputed over the original text, so a dense or fusion number under ablation
would be meaningless unless the ablated corpus is re-encoded with the same model.
Pass --dense-npy with re-encoded vectors and those two arms are reported as well;
without it they are withheld rather than reported from stale embeddings.

Before trusting any delta the script re-derives the published baseline macro MRR
and refuses to continue if it does not reproduce.

    python tools/entity_ablation_retrieval.py
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
# The public release keeps the builders in src/; the private working tree the release
# was produced from keeps them in work/. Accept either so this runs in both places.
for _candidate in (ROOT / "src", ROOT / "work"):
    if (_candidate / "build_chinese_rap_downstream_retrieval_v1.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

PUBLISHED_TFIDF_MRR = 0.415771   # release_validation.json -> held_out_retrieval.character_tfidf_mrr
# (0.447084 is the fusion arm; this script reports the TF-IDF arm, so it checks that one)
FILLER = "\ue000"   # Unicode private use area: cannot collide with corpus characters


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def macro_point(components: list[np.ndarray], system_index: int, metric_index: int) -> float:
    """Label-balanced macro mean: average within label over groups, then over labels."""
    per_label = [float(np.mean(values[:, system_index, metric_index])) for values in components if len(values)]
    return float(np.mean(per_label))


def mask_surfaces(documents: list[str], surfaces: list[str]) -> tuple[list[str], int]:
    if not surfaces:
        return list(documents), 0
    pattern = re.compile("|".join(re.escape(s) for s in sorted(surfaces, key=len, reverse=True)))
    masked, replaced = [], 0
    for document in documents:
        out, count = pattern.subn(lambda m: FILLER * len(m.group(0)), document)
        masked.append(out)
        replaced += count
    return masked, replaced


def frequency_matched_controls(documents: list[str], surfaces: list[str],
                               seed: int) -> tuple[list[str], dict]:
    """Pick one non-entity control per surface, matched as closely as possible.

    Without a control the entity condition confounds "these strings carry label
    signal" with "removing this many characters lowers any similarity". An earlier
    version required the control's corpus frequency to fall inside a band and skipped
    the surface when nothing qualified, which left 46 of 605 surfaces uncontrolled and
    the control removing about 10.5% fewer characters than the entity condition. That
    residual is exactly the confound the control exists to remove.

    This matches every surface by choosing the unused non-entity n-gram of the same
    length whose corpus count is closest, so nothing is skipped, and reports the
    residual gap rather than leaving the caller to infer it.
    """
    rng = random.Random(seed)
    joined = "\n".join(documents)

    # Pool per length, sorted by corpus count once. Scanning the whole pool for every
    # surface is O(surfaces x pool) and does not finish on 605 surfaces; a sorted list
    # plus a bisect makes each match a short local scan.
    pools: dict[int, list[tuple[int, str]]] = {}
    for length in {len(surface) for surface in surfaces}:
        counter: Counter = Counter()
        for document in documents:
            for start in range(len(document) - length + 1):
                gram = document[start:start + length]
                if "\n" not in gram and FILLER not in gram:
                    counter[gram] += 1
        pools[length] = sorted((count, gram) for gram, count in counter.items())

    banned = set(surfaces)
    chosen: list[str] = []
    # Frequent surfaces first: they are the hard ones to match, and matching them
    # before the pool is depleted keeps the total occurrence gap small.
    for surface in sorted(surfaces, key=lambda s: -joined.count(s)):
        pool = pools[len(surface)]
        if not pool:
            continue
        target = joined.count(surface)
        position = bisect.bisect_left(pool, (target, ""))
        best, best_distance = None, None
        # walk outwards from the target count until an unused gram is found, then a
        # little further so ties are not always resolved in the same direction
        low, high = position - 1, position
        while low >= 0 or high < len(pool):
            for index in (high, low):
                if 0 <= index < len(pool):
                    count, gram = pool[index]
                    if gram not in banned:
                        distance = (abs(count - target), rng.random())
                        if best_distance is None or distance < best_distance:
                            best, best_distance = gram, distance
            if best is not None and min(
                    abs(pool[i][0] - target) for i in (low, high) if 0 <= i < len(pool)
            ) > best_distance[0]:
                break
            low -= 1
            high += 1
        if best is None:
            continue
        banned.add(best)
        chosen.append(best)

    entity_total = sum(joined.count(s) for s in surfaces)
    control_total = sum(joined.count(c) for c in chosen)
    report = {
        "surfaces": len(surfaces),
        "controls_matched": len(chosen),
        "entity_corpus_occurrences": entity_total,
        "control_corpus_occurrences": control_total,
        "occurrence_gap_fraction": (control_total - entity_total) / entity_total if entity_total else 0.0,
    }
    return chosen, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lexicon", type=Path,
                        default=ROOT / "outputs/chinese-rap-curated-atlas-v3/safe_lexicon_catalog.csv")
    parser.add_argument("--surfaces", type=Path,
                        default=ROOT / "results/ner-v1/entity_aggregate_provisional.csv",
                        help="restrict masking to these surfaces; omit --released-only to use the full lexicon")
    parser.add_argument("--released-only", action="store_true", default=True,
                        help="mask only the 22 released surfaces (default)")
    parser.add_argument("--full-lexicon", dest="released_only", action="store_false",
                        help="mask every lexicon surface instead")
    parser.add_argument("--types", default="", help="comma-separated schema types, e.g. PLACE")
    parser.add_argument("--dense-npy", type=Path, default=None,
                        help="re-encoded embeddings for the ablated corpus; unlocks dense and fusion arms")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--control-seeds", type=int, default=5,
                        help="independent control draws; the spread across them is the "
                             "uncertainty this design can report")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/entity-ablation")
    parser.add_argument("--skip-baseline-check", action="store_true",
                        help="continue even if the published baseline does not reproduce")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    try:
        import build_chinese_rap_downstream_retrieval_v1 as harness
    except ImportError as error:
        print(f"cannot import the retrieval harness: {error}", file=sys.stderr)
        print("this script reuses src/build_chinese_rap_downstream_retrieval_v1.py and needs "
              "its dependencies (numpy, scipy, scikit-learn) plus the private corpus.", file=sys.stderr)
        return 2

    if args.released_only:
        if not args.surfaces.is_file():
            print(f"missing input: {args.surfaces}", file=sys.stderr)
            return 2
        rows = read_rows(args.surfaces)
        surfaces = [r["entity"] for r in rows]
        types = {r["entity"]: r["entity_type"] for r in rows}
    else:
        if not args.lexicon.is_file():
            print(f"missing input: {args.lexicon}", file=sys.stderr)
            return 2
        rows = read_rows(args.lexicon)
        surfaces = [str(r["entity"]).strip() for r in rows if len(str(r["entity"]).strip()) >= 2]
        types = {}

    if args.types:
        wanted = {t.strip() for t in args.types.split(",") if t.strip()}
        surfaces = [s for s in surfaces if types.get(s) in wanted]
        if not surfaces:
            print(f"no surfaces of type(s) {sorted(wanted)}", file=sys.stderr)
            return 2

    print(f"masking {len(surfaces)} surface(s)")
    print("loading corpus through the release's own harness", flush=True)
    corpus = harness.load_corpus()

    # Reported arms. The dense and fusion entries are only meaningful when the
    # ablated corpus has been re-encoded, so they are withheld unless --dense-npy
    # is supplied rather than being computed from embeddings of the original text.
    ARMS = {"tfidf": harness.SYSTEM_TFIDF,
            "dense": harness.SYSTEM_BGE,
            "fusion": harness.SYSTEM_FUSION}
    reported = ["tfidf"] + (["dense", "fusion"] if args.dense_npy else [])
    metric_index = harness.METRICS.index("mrr")
    recall_index = harness.METRICS.index("recall_at_10")

    def evaluate(strict_docs: list[str], primary_docs: list[str], dense: np.ndarray | None):
        variant = replace(
            corpus,
            strict_docs=strict_docs,
            primary_docs=primary_docs,
            normalized_strict_docs=[harness.normalized_text(d) for d in strict_docs],
            **({"strict_dense": dense} if dense is not None else {}),
        )
        duplicate = harness.detect_duplicate_groups(variant)
        evaluation = harness.evaluate_models(variant, duplicate)
        components, _ = harness.component_metric_values(
            evaluation, variant.song_label_index, duplicate.strict_group_ids)
        return {name: {
            "mrr": macro_point(components, harness.SYSTEMS.index(ARMS[name]), metric_index),
            "recall_at_10": macro_point(components, harness.SYSTEMS.index(ARMS[name]), recall_index),
        } for name in reported}

    print("condition: baseline", flush=True)
    base = evaluate(corpus.strict_docs, corpus.primary_docs, None)
    base_mrr = base["tfidf"]["mrr"]
    drift = abs(base_mrr - PUBLISHED_TFIDF_MRR)
    print(f"  macro MRR {base_mrr:.6f} (published {PUBLISHED_TFIDF_MRR:.6f}, drift {drift:.2e})")
    if drift > 5e-4 and not args.skip_baseline_check:
        print("\nthe baseline does not reproduce the published figure, so any ablation delta",
              file=sys.stderr)
        print("computed here would not be comparable to the released numbers. Refusing to",
              file=sys.stderr)
        print("continue; pass --skip-baseline-check to override deliberately.", file=sys.stderr)
        return 1

    print("condition: entities masked", flush=True)
    strict_masked, replaced_strict = mask_surfaces(corpus.strict_docs, surfaces)
    primary_masked, _ = mask_surfaces(corpus.primary_docs, surfaces)
    dense = np.load(args.dense_npy) if args.dense_npy else None
    entities = evaluate(strict_masked, primary_masked, dense)
    print(f"  macro MRR {entities['tfidf']['mrr']:.6f}   ({replaced_strict} occurrences replaced)")

    control_runs, control_reports = [], []
    for offset in range(args.control_seeds):
        print(f"condition: frequency-matched control, draw {offset + 1}/{args.control_seeds}",
              flush=True)
        controls, report = frequency_matched_controls(corpus.strict_docs, surfaces,
                                                      args.seed + offset)
        strict_control, replaced_control = mask_surfaces(corpus.strict_docs, controls)
        primary_control, _ = mask_surfaces(corpus.primary_docs, controls)
        report["occurrences_replaced"] = replaced_control
        control_runs.append(evaluate(strict_control, primary_control, dense))
        control_reports.append(report)
        print(f"  macro MRR {control_runs[-1]['tfidf']['mrr']:.6f}   "
              f"({replaced_control} replaced, {report['controls_matched']}/{len(surfaces)} matched, "
              f"occurrence gap {report['occurrence_gap_fraction']:+.1%})")
    control = {name: {metric: sum(run[name][metric] for run in control_runs) / len(control_runs)
                      for metric in ("mrr", "recall_at_10")} for name in reported}
    attributable = {}
    for name in reported:
        entity_drop = base[name]["mrr"] - entities[name]["mrr"]
        control_drop = base[name]["mrr"] - control[name]["mrr"]
        attributable[name] = entity_drop - control_drop
        print(f"\n--- {ARMS[name]}, label-balanced macro ---")
        print(f"  baseline                  MRR {base[name]['mrr']:.4f}   R@10 {base[name]['recall_at_10']:.4f}")
        print(f"  entities masked           MRR {entities[name]['mrr']:.4f}   R@10 {entities[name]['recall_at_10']:.4f}   drop {entity_drop:+.4f}")
        print(f"  frequency-matched control MRR {control[name]['mrr']:.4f}   R@10 {control[name]['recall_at_10']:.4f}   drop {control_drop:+.4f}")
        spread = sorted(base[name]["mrr"] - run[name]["mrr"] for run in control_runs)
        print(f"  control drop across {len(spread)} draws: "
              f"{spread[0]:+.4f} to {spread[-1]:+.4f}")
        print(f"  adjusted difference (entity minus mean control): {attributable[name]:+.4f} MRR")
    if not args.dense_npy:
        print("\n  dense and fusion arms withheld: the stored embeddings describe the")
        print("  original text, so a number from them would not describe the ablation.")
        print("  Re-encode the masked corpus and pass --dense-npy to include them.")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "arms_reported": {name: ARMS[name] for name in reported},
        "estimator": "label-balanced macro over duplicate groups, as in the released evaluation",
        "masked_surfaces": len(surfaces),
        "control_surfaces_matched": len(controls),
        "occurrences_replaced": {"entities": replaced_strict,
                                 "control_per_draw": [r["occurrences_replaced"] for r in control_reports]},
        "control_matching": control_reports,
        "control_draws": args.control_seeds,
        "baseline": base,
        "entities_masked": entities,
        "frequency_matched_control": control,
        "adjusted_mrr_difference": attributable,
        "control_drop_range": {name: sorted(base[name]["mrr"] - run[name]["mrr"]
                                            for run in control_runs)[::len(control_runs) - 1]
                               if len(control_runs) > 1 else None
                               for name in reported},
        "dense_and_fusion_reported": bool(args.dense_npy),
        "estimator_note": ("The reported difference is entity drop minus the mean control "
                           "drop over independent draws. It is an adjusted difference, not an "
                           "attributable effect: controls match on surface length and corpus "
                           "frequency, and the residual occurrence gap is recorded above."),
        "claim_boundary": (
            "Measures how much label-identifying signal the masked strings carry in a "
            "fixed-corpus held-out retrieval task. Not an accuracy estimate for the "
            "entity extraction, and not evidence about what any reference means."),
    }
    (args.out / "entity_ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten to {display_path(args.out)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
