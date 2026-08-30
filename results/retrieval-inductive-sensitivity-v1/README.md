# Retrieval inductive sensitivity v1

## Result

Fitting character TF-IDF on the complete unlabeled evaluation corpus gives a small matched advantage over fitting vocabulary and IDF on training folds only: macro MRR changes from **0.405** to **0.409** (delta +0.004). The central fusion result remains under the fully inductive design: inductive fusion reaches macro MRR **0.440**, +0.035 above inductive TF-IDF.

This supports a bounded conclusion: transductive vocabulary/IDF exposure contributes slight optimism but does not explain the semantic-lexical fusion gain.

## Files

- `metrics.csv`: duplicate-component-adjusted source-credit-label macro estimates and paired bootstrap intervals.
- `paired_deltas.csv`: paired system differences with 95% intervals.
- `fold_summary.csv`: aggregate six-fold coverage and feature counts.
- `analysis_summary.json`: machine-readable result and lineage.
- `METHOD.md`: complete estimand, split, scoring, aggregation, and uncertainty definitions.
- `validation.json` and `manifest.json`: release checks and byte-level inventory.

## Population and privacy

The analysis contains 5,455 length-qualified song queries, 5,430 global exact/near-duplicate components, 5,432 within-label component units, and 204 eligible source-credit labels. Public outputs contain no song identifiers, label names, lyric text, per-query scores, fold assignments, or vectors.

## Claim boundary

Grouped cross-fitted held-out-song retrieval of corpus source-credit-label lyrical-repertoire similarity. The comparison isolates lexical vocabulary/IDF exposure within this fixed corpus; it is not artist identification, biography, influence, collaboration, genre, social relation, or external-population performance.
