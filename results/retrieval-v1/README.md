# Explainable lyrical-repertoire retrieval

This release evaluates whether a held-out song's cleaned lyrical representation retrieves its corpus source-credit label's remaining lyrical repertoire. It compares BGE-M3 dense similarity, character 2–5 gram TF-IDF, and an equal-weight per-query z-score fusion.

## Read first

- `metrics.csv` contains macro and micro MRR, Recall@1/5/10, and nDCG@10.
- `uncertainty.csv` contains paired literal occurrence-wise two-stage bootstrap intervals for system differences.
- `coverage.csv` reports ground-truth-label and prediction-label coverage.
- `per_label_metrics.csv` contains aggregate source-credit-label results.
- `label_level_examples.csv` contains only aggregate label-level examples; it contains no lyrics or song/chunk identifiers.
- `METHOD.md` defines the leakage controls and claim boundary.

Headline strict-fusion macro MRR: **0.447** (95% bootstrap CI 0.414–0.481).

Population: 5,455 length-qualified held-out song queries and 204 eligible source-credit labels. Exact/near-duplicate grouping yields 5,430 global components, which map to 5,432 within-label component units for macro estimation and bootstrap resampling.

## Claim boundary

Held-out-song retrieval of corpus source-credit-label lyrical repertoire similarity; not artist identification, biography, influence, collaboration, genre, social relation, or a human semantic-similarity gold standard.
