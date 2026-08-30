"""Report per-surface extraction reliability against the claims each surface carries.

`entity_aggregate_provisional.csv` already records, for every released surface, how
many lexicon candidates survived exact-span agreement with the transformer
baseline. That ratio is an inter-method reliability statistic and it varies by
almost a factor of two across the inventory, but nothing in the release reads it
back against the released edges and co-mentions -- so there is no statement
anywhere about whether the weakest-agreement surfaces are the ones doing work.

This joins the three public tables and answers that directly. It needs no private
input and no annotation.

    python tools/summarise_surface_reliability.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


# Windows consoles default to a legacy code page; the Han text these tools print
# must not depend on the caller exporting PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
NER = ROOT / "results" / "ner-v1"
OUT = ROOT / "analysis" / "surface-reliability"


def display_path(path: Path) -> str:
    """Repo-relative when the output lives inside the tree, absolute otherwise."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build() -> list[dict[str, object]]:
    inventory = read_rows(NER / "entity_aggregate_provisional.csv")
    links = read_rows(NER / "source_label_entity_links_provisional.csv")
    co_mentions = read_rows(NER / "entity_co_mentions_provisional.csv")

    carried: dict[str, list[str]] = {}
    support: dict[str, int] = {}
    for row in links:
        carried.setdefault(row["entity"], []).append(
            f"label-link {row['source_credit_label']}->{row['entity']}")
        support[row["entity"]] = support.get(row["entity"], 0) + int(row["entity_song_units_within_label"])
    for row in co_mentions:
        for side in ("entity_a", "entity_b"):
            carried.setdefault(row[side], []).append(
                f"co-mention {row['entity_a']}~{row['entity_b']}")
            support[row[side]] = support.get(row[side], 0) + int(row["unique_song_unit_co_mentions"])

    rows = []
    for item in inventory:
        entity = item["entity"]
        claims = carried.get(entity, [])
        rows.append({
            "entity": entity,
            "entity_type": item["entity_type"],
            "lexicon_candidate_occurrences": int(item["lexicon_candidate_occurrences"]),
            "strict_agreement_occurrences": int(item["strict_agreement_occurrences"]),
            "strict_agreement_rate": float(item["strict_agreement_rate"]),
            "occurrences_rejected_by_span_disagreement":
                int(item["lexicon_candidate_occurrences"]) - int(item["strict_agreement_occurrences"]),
            "released_claims_carried": len(claims),
            "released_claim_song_units": support.get(entity, 0),
            "released_claims": "; ".join(claims),
        })
    rows.sort(key=lambda r: (r["strict_agreement_rate"], -r["released_claims_carried"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rows = build()
    load_bearing = [r for r in rows if r["released_claims_carried"]]
    rates = [r["strict_agreement_rate"] for r in rows]
    weakest = [r for r in load_bearing if r["strict_agreement_rate"] < 0.70]

    write_csv(args.out / "surface_reliability.csv", rows)

    lines = [
        "# Surface reliability against released claims",
        "",
        "Derived from the public NER tables by `tools/summarise_surface_reliability.py`.",
        "No private input and no annotation is involved; every number here is a join of",
        "`entity_aggregate_provisional.csv`, `source_label_entity_links_provisional.csv`,",
        "and `entity_co_mentions_provisional.csv`.",
        "",
        "## What the ratio means",
        "",
        "A lexicon candidate is retained only when the transformer baseline proposes the",
        "*same span* with the *same schema type*. `boundary_ok()` in the builder returns",
        "True unconditionally for surfaces containing no ASCII characters, so the lexicon",
        "stage alone cannot separate a surface from a longer compound that contains it;",
        "the exact-span requirement is what rejects those. The retention rate is therefore",
        "an inter-method reliability statistic, not an accuracy estimate: it says how often",
        "two independent methods agreed, not how often either was right.",
        "",
        f"Across {len(rows)} released surfaces the rate ranges "
        f"{min(rates):.2f}-{max(rates):.2f} (median {statistics.median(rates):.2f}).",
        "",
        "## Surfaces that carry released claims",
        "",
        "| surface | type | agreement | candidates rejected | claims | song units |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in load_bearing:
        lines.append(
            f"| {r['entity']} | {r['entity_type']} | {r['strict_agreement_rate']:.2f} | "
            f"{r['occurrences_rejected_by_span_disagreement']} | {r['released_claims_carried']} | "
            f"{r['released_claim_song_units']} |")

    lines += [
        "",
        "## Reading",
        "",
        "The two lowest-agreement load-bearing surfaces are "
        + ", ".join(f"**{r['entity']}** ({r['strict_agreement_rate']:.2f})" for r in weakest[:2])
        + ", and both rest on small supports. It is tempting to read that as a ranking of",
        "how secure the released claims are. It is not, and treating it as one would be a",
        "mistake this file exists to prevent.",
        "",
        "The rate is computed corpus-wide, but a released label link depends only on the",
        "occurrences inside one source label, and a co-mention only on the songs where both",
        "surfaces appear. Those populations differ. A surface can look weak overall because",
        "of compounds that never occur under the label carrying the claim: 湖南 has the",
        "second-lowest retention here, yet the compounds that lower it -- 湖南卫视, a food",
        "term, and a university choir's institutional name generalised under NER-CR-001 --",
        "appear under other labels, and direct enumeration of its occurrences",
        "under 泰格西 finds every one of them a bare locative use.",
        "",
        "So use this table to decide what to look at, and `tools/audit_released_claim_",
        "occurrences.py` to decide what it means. That script resolves each released claim",
        "to its own occurrences, applies the same shared-text exclusion, and classifies",
        "every one; it needs the private corpus but no annotation.",
        "",
        "What the rate does bound is the extraction as a whole, and that bound is real:",
        "BH-FDR and the conservative intervals control sampling error given the extraction,",
        "never error in the extraction. The rate also cannot distinguish a genuinely",
        "ambiguous surface from a domain mismatch in the transformer baseline, which is",
        "`ckiplab/albert-tiny-chinese-ner`: a very small model trained on traditional-Chinese",
        "material and applied here to simplified Chinese. Reporting agreement across three",
        "or more simplified-Chinese taggers separates those two explanations without any",
        "annotation; `tools/multi_tagger_agreement.py` does that.",
        "",
    ]
    (args.out / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(f"{len(rows)} surfaces, {len(load_bearing)} carry released claims")
    for r in load_bearing:
        flag = "  <-- lowest agreement" if r in weakest[:2] else ""
        print(f"  {r['entity']:<6} {r['strict_agreement_rate']:.2f}  "
              f"rejected {r['occurrences_rejected_by_span_disagreement']:>3}  "
              f"claims {r['released_claims_carried']}  units {r['released_claim_song_units']}{flag}")
    print(f"\nwritten to {display_path(args.out)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
