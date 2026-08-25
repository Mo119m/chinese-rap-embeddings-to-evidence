# Method and limitations

## Research question

For each song, can its cleaned lyric-based representation retrieve the corpus source-credit label attached to that song when the entire song—and every detected exact or near-duplicate variant—is removed from every candidate label repertoire?

The outcome is **corpus source-credit-label lyrical similarity**. It is not an identity classifier and does not imply artist style, biography, influence, collaboration, genre, social relation, or cultural causation.

## Why these three systems

BGE-M3 supplies a multilingual dense representation, but it was not assumed to be superior. The preceding low-overlap same-song continuation benchmark evaluated 1,000 queries and found MRR 0.223 for BGE-M3, 0.255 for character TF-IDF, and 0.278 for fusion. That benchmark motivated testing complementary dense and character-form evidence here; it is not reused as downstream gold data.

## Evaluation population

- Candidate labels: the 204 graph-eligible corpus source-credit labels.
- Query unit: one whole song assembled from its eligible cleaned chunks after exact cross-label shared-text exclusion.
- Minimum query length: 50 NFKC-normalized alphanumeric characters.
- Queries: 5,455 songs.
- Global exact/near-duplicate components: 5,430.
- Within-label component units: 5,432. A cross-label global component contributes one unit to each label stratum it intersects, so the within-label total can exceed the global total.
- Each query has one corpus source-credit label in this fixed analysis population.

## Representations

For BGE-M3, eligible cleaned chunk vectors are averaged within a song using the frozen comparison-text weights and then L2-normalized. No new model fitting occurs. For the lexical system, whole-song cleaned text is represented by character 2–5 gram TF-IDF (`min_df=3`, sublinear term frequency, L2 normalization, maximum 150,000 features). The strict matrix retained 150,000 features.

Each candidate label profile is an equal-component mean of its training-song representations. Songs in the same exact/near-duplicate component receive weights summing to one within a label, preventing repeated variants from dominating the profile.

## Strict held-out and leakage controls

1. **Whole-song holdout:** for every query, its complete song representation is removed from its true label profile.
2. **Exact shared-text exclusion:** chunks whose exact cleaned text occurs across source-credit labels are removed before strict representations are built. This removed 4,877 eligible membership rows overall and 522 rows in the evaluated song population; zero exact cleaned-text hashes remain shared across strict labels. Of 6,848 otherwise eligible songs, 1,229 had no non-shared chunk remaining and therefore cannot enter the strict query population.
3. **Exact and near-duplicate holdout:** normalized whole-song text is converted to character-trigram sets. A complete prefix-filtered all-pairs join finds every pair with Jaccard similarity at least 0.80; full sets verify candidates. All connected variants are held out together from every affected label profile. The audit found 1 exact pair and 25 additional near-duplicate pairs, including 3 cross-label pairs.
4. **Duplicate-adjusted aggregation:** every duplicate component contributes total weight one within a label in both profiles and macro evaluation.
5. **No query rows in public outputs:** song/chunk identifiers, pair evidence, and per-query ranks remain under the private audit directory. No lyric text or vectors are copied into either output directory.

TF-IDF vocabulary and IDF are estimated on the fixed unlabeled evaluation corpus. Candidate source-credit labels are not used during TF-IDF fitting; the label profiles themselves are strictly group-held-out. This is a corpus-internal transductive retrieval design, not an estimate for unseen future corpora.

## Fusion and ablations

Candidate-label scores are standardized separately within each query for BGE-M3 and TF-IDF, then averaged with equal weight. No fusion weight is tuned on the evaluation outcomes.

Three diagnostics are reported: averaging unstandardized cosine scores, removing the non-exact near-duplicate guard while retaining exact duplicate holdout, and rebuilding representations with cross-label shared text included. The shared-text diagnostic holds the strict 5,455-song query population fixed; it does not reintroduce the 1,229 songs that consist only of cross-label shared text. Only the systems explicitly marked `(strict)` are primary results.

## Metrics and uncertainty

With one relevant label per query, MRR is reciprocal true-label rank; Recall@k is whether the true label occurs in the first k candidates; nDCG@10 is `1/log2(rank+1)` for ranks at most 10 and zero otherwise. Query-micro values are descriptive. Primary inference uses source-credit-label macro means after duplicate-component adjustment.

Uncertainty uses 5,000 fixed-seed, literal occurrence-wise paired two-stage replicates. For each replicate, exactly 204 outer source-credit-label occurrences are sampled with replacement. For **every outer occurrence**, its within-label component units are independently sampled with replacement, using the original stratum size—even when the same label appears multiple times in the outer draw. One inner component-index draw is applied jointly to the complete system-by-metric tensor, preserving query/component pairing for system differences. The 204 occurrence means are averaged; percentile 2.5% and 97.5% endpoints form the reported 95% intervals. These intervals quantify resampling variability within the fixed corpus strata, not uncertainty for an external population of artists or songs.

## Interpretation limits

- Source-credit strings are corpus labels, not externally verified identity records.
- Correct retrieval shows consistency with a label's remaining lyric corpus, not authorship verification.
- The target label comes from corpus provenance rather than an independently annotated semantic gold standard.
- Strict leakage control changes the estimand: 1,229 otherwise eligible songs with no label-specific text are excluded, followed by 164 songs below the minimum effective length.
- Dense similarity remains difficult to explain at the individual token level; TF-IDF and the aggregate error summaries supply partial, not complete, interpretability.
- Small repertoires have wider effective uncertainty; the smallest strict profile has 5 independent training components after holdout.
- Chinese rap performance, delivery, beat, timing, and audio rhyme are outside this text-only task.

## Claim boundary

Held-out-song retrieval of corpus source-credit-label lyrical repertoire similarity; not artist identification, biography, influence, collaboration, genre, social relation, or a human semantic-similarity gold standard.
