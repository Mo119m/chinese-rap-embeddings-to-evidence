"""Measure extraction reliability across several taggers instead of against a gold set.

The release retains a lexicon candidate when one transformer baseline,
`ckiplab/albert-tiny-chinese-ner`, proposes the same span with the same type. That
model is small and was trained on traditional-Chinese material while this corpus is
predominantly simplified, so a low retention rate cannot be told apart from a domain
mismatch in the baseline. Running further simplified-Chinese taggers separates them.

The output is inter-method reliability -- Krippendorff's alpha over methods, and the
share of candidates confirmed by at least k of n. That is a different quantity from
inter-annotator reliability and must be reported as such: it says how often
independent methods agreed, never how often any of them was right. It needs no
annotation, so it can be produced from compute alone.

Candidates come from the pipeline's own occurrence table, not from a fresh text scan.
An earlier version scanned the cleaned sidecar directly and scored a superset -- 334
candidates for 上海 where the release records 174. Each candidate is scored inside its
own stored context window, so the span the taggers are asked about is the span the
release recorded.

Population reproduction is checked per surface and reported. Restricting to
`cross_label_shared_cleaned_text == False` reproduces the published
`lexicon_candidate_occurrences` exactly for 13 of the 22 released surfaces; the
remaining aggregate filter has not been identified, so the headline statistics are
computed over the reproducing subset and the rest are listed separately rather than
silently pooled.

    pip install torch transformers
    python tools/multi_tagger_agreement.py --taggers hf:shibing624/bert4ner-base-chinese
    python tools/multi_tagger_agreement.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
csv.field_size_limit(10 ** 8)


def display_path(path: Path) -> str:
    """Repo-relative when the output lives inside the tree, absolute otherwise."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# Tagger label vocabularies differ; map each onto the release's schema types.
TYPE_ALIASES = {
    "PLACE": {"LOC", "GPE", "LOCATION", "NS", "ORG_LOC", "FAC", "地名", "PLACE",
              "ADDRESS", "SCENE"},                       # CLUENER: address, scene
    "PERSON_REFERENCE": {"PER", "PERSON", "NR", "人名", "PERSON_REFERENCE", "NAME"},
    "LANGUAGE_OR_DIALECT_REFERENCE": {"LANGUAGE", "NORP", "LANGUAGE_OR_DIALECT_REFERENCE"},
}


def canonical_type(raw: str) -> str | None:
    upper = str(raw).upper().removeprefix("B-").removeprefix("I-").removeprefix("E-").removeprefix("S-")
    for schema_type, aliases in TYPE_ALIASES.items():
        if upper in aliases:
            return schema_type
    return None


# --------------------------------------------------------------------------- stats

def krippendorff_alpha_nominal(ratings: list[list[str | None]]) -> float | None:
    """Nominal alpha over units x coders, ignoring None (a coder that abstained).

    Returns None when every rating carries the same value, where expected
    disagreement is zero and alpha is undefined.
    """
    pairable = [[v for v in unit if v is not None] for unit in ratings]
    pairable = [unit for unit in pairable if len(unit) >= 2]
    if not pairable:
        return None

    total = sum(len(unit) for unit in pairable)
    observed = 0.0
    for unit in pairable:
        counts = Counter(unit)
        disagreeing = sum(a * b for va, a in counts.items() for vb, b in counts.items() if va != vb)
        observed += disagreeing / (len(unit) - 1)
    observed /= total

    overall = Counter(v for unit in pairable for v in unit)
    expected = sum(a * b for va, a in overall.items() for vb, b in overall.items() if va != vb)
    expected /= total * (total - 1)

    if expected == 0:
        return None
    return 1.0 - observed / expected


# ------------------------------------------------------------------------- taggers

def load_tagger(spec: str):
    """Return (tag_fn, expressible_types).

    A tagger whose label inventory has no category for a schema type cannot vote on
    it either way. Counting that silence as rejection would be wrong: MSRA and
    CLUENER have no language-reference category at all, so every 中文 and 英文
    candidate would score zero agreement for a reason that has nothing to do with
    the corpus. Such a tagger abstains, and the alpha computation ignores it.
    """
    if spec == "hanlp":
        import hanlp  # noqa: PLC0415
        model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

        def tag(text: str):
            spans = set()
            for entity, label, start, end in model(text, tasks="ner*")["ner/msra"]:
                schema_type = canonical_type(label)
                if schema_type:
                    spans.add((start, end, schema_type))
            return spans
        return tag, {"PLACE", "PERSON_REFERENCE"}          # MSRA: LOC, PER, ORG

    if spec == "ltp":
        from ltp import LTP  # noqa: PLC0415
        model = LTP()

        def tag(text: str):
            output = model.pipeline([text], tasks=["cws", "ner"])
            spans = set()
            for label, entity, start, end in output.ner[0]:
                schema_type = canonical_type(label)
                if schema_type:
                    spans.add((start, end, schema_type))
            return spans
        return tag, {"PLACE", "PERSON_REFERENCE"}          # LTP: Ns, Nh, Ni

    if spec.startswith("hf:"):
        from transformers import pipeline  # noqa: PLC0415
        recogniser = pipeline("token-classification", model=spec[3:], aggregation_strategy="simple")
        # derive the expressible set from the model's own label inventory
        expressible = {canonical_type(label) for label in recogniser.model.config.id2label.values()}
        expressible.discard(None)

        def tag(text: str):
            spans = set()
            for item in recogniser(text):
                schema_type = canonical_type(item["entity_group"])
                if schema_type:
                    spans.add((int(item["start"]), int(item["end"]), schema_type))
            return spans
        return tag, expressible

    raise SystemExit(f"unknown tagger spec: {spec!r} (expected hanlp, ltp, or hf:<model-id>)")


# ---------------------------------------------------------------------------- main

def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def score_candidate(row, schema_type, taggers, expressible) -> list[str | None]:
    """One rating per tagger for a single stored candidate.

    The span asked about is the one the release recorded, inside the context window
    the release stored, so the taggers answer the same question the pipeline did.
    A tagger with no category for this schema type abstains (None) rather than
    voting no.
    """
    try:
        span = (int(row["surface_start_in_context"]), int(row["surface_end_in_context"]))
    except (TypeError, ValueError):
        return [None] * len(taggers)
    snippet = row["context_snippet"]
    vector: list[str | None] = []
    for spec, tag in taggers.items():
        if schema_type not in expressible[spec]:
            vector.append(None)
        else:
            vector.append("yes" if (span[0], span[1], schema_type) in tag(snippet) else "no")
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "work/private-chinese-rap-ner-cultural-graph-v1/all_candidate_occurrences_private.csv")
    parser.add_argument("--surfaces", type=Path,
                        default=ROOT / "results/ner-v1/entity_aggregate_provisional.csv")
    parser.add_argument("--taggers", default="hanlp",
                        help="comma-separated: hanlp, ltp, hf:<model-id>")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/multi-tagger-agreement")
    parser.add_argument("--self-test", action="store_true", help="check the statistics and exit")
    return parser.parse_args()


def self_test() -> int:
    # Two coders, four units, one disagreement. Nominal alpha for this case is
    # 8/15; Scott's pi on the same table is 7/15, alpha being higher because of
    # its small-sample correction.
    alpha = krippendorff_alpha_nominal([["1", "1"], ["1", "1"], ["0", "0"], ["0", "1"]])
    assert alpha is not None and abs(alpha - 8 / 15) < 1e-9, alpha

    assert abs(krippendorff_alpha_nominal([["1", "1"], ["0", "0"]]) - 1.0) < 1e-9
    assert krippendorff_alpha_nominal([["1", "1"], ["1", "1"]]) is None       # no variance
    assert krippendorff_alpha_nominal([["1", None], ["0", None]]) is None     # nothing pairable

    three = krippendorff_alpha_nominal([["1", "1", "1"], ["1", "1", "0"], ["0", "0", "0"], ["0", "1", "0"]])
    assert three is not None and 0.0 < three < 1.0, three

    assert canonical_type("B-LOC") == "PLACE"
    assert canonical_type("address") == "PLACE"
    assert canonical_type("nr") == "PERSON_REFERENCE"
    assert canonical_type("MISC") is None
    print("self-test passed: alpha, abstention handling, and label mapping behave as specified")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()

    for path in (args.candidates, args.surfaces):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    inventory = {row["entity"]: row for row in read_rows(args.surfaces)}
    published = {name: int(row["lexicon_candidate_occurrences"]) for name, row in inventory.items()}

    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(args.candidates):
        surface = row["candidate_surface"]
        if (surface in inventory
                and row["candidate_source"] == "LEXICON_WITH_TRANSFORMER_CHECK"
                and row["cross_label_shared_cleaned_text"] == "False"):
            candidates[surface].append(row)

    reproduces = {s: len(candidates[s]) == published[s] for s in inventory}
    print(f"population reproduces the published candidate count for "
          f"{sum(reproduces.values())} of {len(inventory)} surfaces")

    specs = [s.strip() for s in args.taggers.split(",") if s.strip()]
    loaded = {spec: load_tagger(spec) for spec in specs}
    taggers = {spec: fn for spec, (fn, _types) in loaded.items()}
    expressible = {spec: types for spec, (_fn, types) in loaded.items()}
    for spec, types in expressible.items():
        print(f"  {spec} expresses: {', '.join(sorted(types)) or '(nothing mappable)'}")

    records, summary_alphas = [], []
    print(f"\n{'surface':<8}{'cands':>7}{'pub':>6}{'alpha':>9}   confirmed by k of n")
    print("-" * 78)
    for surface in sorted(candidates, key=lambda s: -len(candidates[s])):
        schema_type = inventory[surface]["entity_type"]
        vectors, confirmations, abstained = [], Counter(), 0
        for row in candidates[surface]:
            vector = score_candidate(row, schema_type, taggers, expressible)
            vectors.append(vector)
            if all(v is None for v in vector):
                abstained += 1
            else:
                confirmations[sum(v == "yes" for v in vector)] += 1

        alpha = krippendorff_alpha_nominal(vectors)
        marker = "" if reproduces[surface] else "  *"
        if abstained == len(vectors):
            print(f"{surface:<8}{len(vectors):>7}{published[surface]:>6}{'--':>9}   "
                  f"no tagger has a category for {schema_type}; not comparable{marker}")
        else:
            breakdown = "  ".join(f"{k}:{confirmations.get(k, 0)}" for k in range(len(taggers) + 1))
            print(f"{surface:<8}{len(vectors):>7}{published[surface]:>6}"
                  f"{('n/a' if alpha is None else f'{alpha:.3f}'):>9}   {breakdown}{marker}")
            if alpha is not None and reproduces[surface]:
                summary_alphas.append((surface, alpha))

        record = {"entity": surface, "entity_type": schema_type,
                  "candidates_scored": len(vectors),
                  "published_lexicon_candidates": published[surface],
                  "population_reproduces_published": reproduces[surface],
                  "krippendorff_alpha_over_methods": "" if alpha is None else round(alpha, 6),
                  "all_taggers_abstained": abstained,
                  "confirmed_by_all": confirmations.get(len(taggers), 0),
                  "confirmed_by_none": confirmations.get(0, 0)}
        for index, spec in enumerate(taggers):
            votes = [v[index] for v in vectors if v[index] is not None]
            record[f"confirm_rate::{spec}"] = (
                round(sum(x == "yes" for x in votes) / len(votes), 6) if votes else "")
        records.append(record)

    print("\n* = candidate count does not reproduce the published figure for this surface;")
    print("  excluded from the summary below.")
    if summary_alphas:
        negative = [s for s, a in summary_alphas if a < 0]
        strong = [(s, a) for s, a in summary_alphas if a > 0.25]
        print(f"\nOver the {len(summary_alphas)} reproducing surfaces with a defined alpha: "
              f"{len(negative)} negative, {len(strong)} above 0.25"
              + (f" ({', '.join(f'{s} {a:.2f}' for s, a in strong)})" if strong else ""))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "multi_tagger_agreement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (args.out / "run.json").write_text(json.dumps({
        "taggers": list(taggers),
        "candidate_source": "pipeline occurrence table, LEXICON_WITH_TRANSFORMER_CHECK, "
                            "cross_label_shared_cleaned_text == False",
        "surfaces": len(records),
        "surfaces_reproducing_published_count": sum(reproduces.values()),
        "statistic": "Krippendorff alpha, nominal, over methods (not over annotators)",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten to {display_path(args.out)}/")
    print("Report this as inter-method reliability. It is not an accuracy estimate and")
    print("must not be described as human validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
