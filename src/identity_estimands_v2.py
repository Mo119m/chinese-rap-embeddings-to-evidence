#!/usr/bin/env python3
"""One set of ranks, three ways of averaging them.

The identity-space and identity-probe experiments report a plain mean over queries, which
is the number a reader can check against a head-to-head count. The retrieval builder
reports a macro mean over source-credit labels, so that a label with 300 songs cannot
decide the conclusion for 225 others. Both are legitimate; they are different estimands
and the manuscript has to say which one it is quoting. This file reads the two private
per-query tables and reports every system under all three:

  micro       each query weight one
  component   each (leakage group, label) component weight one, the builder's weighting
  macro       component-weighted mean within each label, then the mean over labels

No new scoring happens here; the ranks are the ones the experiments wrote.

    python src/identity_estimands_v2.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402

OUT_DIR = ROOT / "results" / "retrieval-v2"

# contrasts the experiments did not report themselves, on the same ranks
CONTRASTS = {
    "identity_spaces": [
        ("fusion_dense_lexical", "lexical_character_ngrams"),
        ("fusion_dense_lexical", "dense_semantic"),
        ("lexical_character_ngrams", "dense_semantic"),
    ],
    "identity_probe": [
        ("fusion_lexical_within_author_whitening", "fusion_lexical_none"),
        ("within_author_whitening", "lexical_character_ngrams"),
    ],
}

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TABLES = {
    "identity_spaces": "private-identity-spaces-v2",
    "identity_probe": "private-identity-probe-v2",
}


def estimands(rows: list[dict], column: str, mask: np.ndarray) -> dict:
    ranks = np.asarray([int(r[column]) for r in rows], dtype=np.float64)
    labels = np.asarray([r["label"] for r in rows])
    groups = np.asarray([int(r["group"]) for r in rows])
    size = Counter(zip(groups.tolist(), labels.tolist()))
    weights = np.asarray([1.0 / size[(g, l)] for g, l in zip(groups.tolist(), labels.tolist())])
    rr = 1.0 / ranks
    hit10 = (ranks <= 10).astype(np.float64)
    out = {}
    for metric, values in (("mrr", rr), ("recall_at_10", hit10)):
        micro = float(values[mask].mean())
        component = float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))
        per_label = defaultdict(lambda: [0.0, 0.0])
        for value, weight, label, keep in zip(values, weights, labels, mask):
            if keep:
                per_label[label][0] += value * weight
                per_label[label][1] += weight
        macro = float(np.mean([num / den for num, den in per_label.values()]))
        out[metric] = {"micro": round(micro, 4), "component": round(component, 4),
                       "macro": round(macro, 4)}
    out["queries"] = int(mask.sum())
    out["labels"] = int(len({l for l, keep in zip(labels, mask) if keep}))
    return out


def build(private_root: Path, out_dir: Path) -> int:
    payload = {
        "analysis": "the identity experiments' ranks under micro, component and macro averaging",
        "definitions": {
            "micro": "each query weight one; the number the experiments print",
            "component": "each (leakage group, label) component weight one -- the retrieval "
                         "builder's per-query weighting",
            "macro": "component-weighted mean within each label, then the plain mean over "
                     "labels -- the retrieval builder's headline estimand",
        },
        "tables": {},
    }
    for name, folder in TABLES.items():
        path = private_root / "work" / folder / "per_query.csv"
        if not path.is_file():
            raise SystemExit(f"missing private table: {path}")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        columns = [c for c in rows[0] if c.startswith("rank_")]
        everything = np.ones(len(rows), dtype=bool)
        table = {"queries": len(rows), "systems": {}}
        for column in columns:
            system = column[len("rank_"):]
            table["systems"][system] = {"all_queries": estimands(rows, column, everything)}
            if "covered" in rows[0]:
                covered = np.asarray([r["covered"] == "1" for r in rows])
                table["systems"][system]["covered_subset"] = estimands(rows, column, covered)
        groups = np.asarray([int(r["group"]) for r in rows])
        labels = np.asarray([r["label"] for r in rows])
        size = Counter(zip(groups.tolist(), labels.tolist()))
        weights = np.asarray([1.0 / size[(g, l)] for g, l in zip(groups.tolist(), labels.tolist())])
        rr = {c[len("rank_"):]: 1.0 / np.asarray([int(r[c]) for r in rows], dtype=np.float64)
              for c in columns}
        table["paired_contrasts"] = {
            "design": ("2000 replicates, seed 20260825, leakage groups resampled with "
                       "replacement, each (group, label) component weighted one; all queries"),
            "contrasts": paired_group_bootstrap(rr, weights, groups, everything, CONTRASTS[name]),
        }
        payload["tables"][name] = table
        print(f"{name}: {len(rows):,} queries, {len(columns)} systems")
        for c in table["paired_contrasts"]["contrasts"]:
            print(f"  {c['system']} - {c['minus']}: {c['mrr_difference']:+.4f} "
                  f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
        for system, block in table["systems"].items():
            m = block["all_queries"]["mrr"]
            print(f"  {system:44s} micro {m['micro']:.4f}  component {m['component']:.4f}  "
                  f"macro {m['macro']:.4f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "identity_estimands.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")
    print(f"wrote {out_dir / 'identity_estimands.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
