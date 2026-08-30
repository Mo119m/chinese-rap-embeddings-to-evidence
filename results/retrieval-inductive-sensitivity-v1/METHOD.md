# Method: grouped cross-fitted inductive TF-IDF sensitivity

## Question and estimand

The analysis asks whether the retrieval-v1 lexical result and its equal-weight semantic-lexical fusion depend materially on fitting TF-IDF vocabulary and inverse-document-frequency weights on the complete unlabeled evaluation corpus. The estimand is the paired difference between two otherwise matched systems within the fixed, leakage-controlled corpus.

This is a sensitivity analysis, not a replacement for retrieval-v1. Absolute cross-fitted estimates use five-sixths of duplicate components to build each fold's label profiles and therefore should not be interpreted as a direct estimate of the original leave-one-component-out profile-size effect.

## Frozen population

The population is inherited from retrieval-v1: 5,455 length-qualified held-out songs from 204 eligible source-credit labels after shared-text exclusion. Exact and character-trigram-Jaccard >= 0.80 near duplicates use the frozen private retrieval-v1 component assignments. There are 5,430 global components and 5,432 within-label component units; a cross-label component contributes one unit to every label stratum that it intersects.

## Grouped six-fold split

Every global duplicate component is assigned wholly to one fold, so no exact/near-duplicate component crosses train and test. Components are ordered with multi-label components first, then larger components, then an unseeded SHA-256 tie-break of the frozen private component identifier. For a component touching labels L, the selected fold lexicographically minimizes:

1. the maximum current component count in that fold over labels in L;
2. the sum of each affected label's current fold count divided by its total component count;
3. the fold's current song count; and
4. the fold index.

All six test folds contain every source-credit label, and every corresponding training partition retains at least five components per label.

## Duplicate-component-weighted profiles

Within a training partition, song i in component g and label l receives weight

`w_i = 1 / n_train(g,l)`.

Weights therefore sum to one for every observed component-label unit. For normalized dense song vector d_i, the label profile is

`c_dense(l) = normalize(sum_i I[y_i=l] * w_i * d_i)`.

The lexical profile uses the same weighted sum and L2 normalization in sparse TF-IDF space. Test-to-label scores are cosine similarities.

## Lexical representations

Both lexical systems use character 2-5 gram TF-IDF with `min_df=3`, sublinear term frequency, L2 normalization, float32 values, and at most 150,000 features.

- **Inductive:** fit vocabulary and IDF only on the current fold's training documents, then transform training and test documents.
- **Matched transductive:** fit vocabulary and IDF once on all 5,455 unlabeled documents, while still constructing each fold's label profiles only from that fold's training rows and scoring only its test rows.

The paired comparison therefore changes lexical vocabulary/IDF exposure while holding query population, fold membership, label-profile membership, duplicate weights, candidate labels, score fusion, ranking, metrics, and resampling fixed.

## Fusion and ranking

Dense and lexical candidate-label score vectors are standardized separately within every query over the same 204 candidate labels. Fusion is the untrained average

`s_fusion(q,l) = 0.5 * z_l(s_dense(q,l)) + 0.5 * z_l(s_lexical(q,l))`.

Candidate ties use the frozen ascending label-ID order. Each query receives exactly one out-of-fold prediction.

## Metrics and aggregation

With one relevant source-credit label per query, the metrics are reciprocal rank, Recall@1, Recall@5, Recall@10, and nDCG@10. Query metrics are first averaged within every duplicate-component-by-label unit. Units are then averaged within labels, and the headline estimate is the unweighted macro mean over 204 labels.

## Uncertainty

Intervals use 5,000 fixed-seed paired two-stage bootstrap replicates. Each replicate samples 204 outer label occurrences with replacement. For every outer occurrence, that label's component units are independently sampled with replacement at the original stratum size. The same sampled units are applied to the complete system-by-metric tensor, preserving paired differences. Percentile 2.5% and 97.5% endpoints form the 95% intervals.

These intervals describe resampling variability within the fixed corpus strata. They do not quantify uncertainty for an external population of artists or songs.

## Interpretation rule

The matched-transductive-minus-inductive delta estimates the contribution of unlabeled evaluation-corpus vocabulary/IDF exposure under matched folds and profiles. It does not estimate causal generalization to a new corpus. A small positive delta is described as limited transductive optimism, not leakage-free evidence. The inductive-fusion-minus-inductive-TF-IDF comparison tests whether the fusion gain remains when lexical fitting is training-only.

## Public-release boundary

Only aggregate metrics, aggregate fold diagnostics, methods, validation, and hashes are released. Song identifiers, source-credit-label names, lyric text, per-query ranks or scores, fold assignments, and vector arrays remain private.

Grouped cross-fitted held-out-song retrieval of corpus source-credit-label lyrical-repertoire similarity. The comparison isolates lexical vocabulary/IDF exposure within this fixed corpus; it is not artist identification, biography, influence, collaboration, genre, social relation, or external-population performance.
