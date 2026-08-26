# Network construction and interpretation contract

## Population and representations

The graph population is restricted to canonical artist-title comparison rows marked eligible after metadata cleaning. Each source-credit label must have at least five independent clean songs and at least 20 units of effective lyric-text mass in both representations. Exact duplicate clean text is weighted to a total mass of one within the comparison population.

For each eligible label, duplicate-weighted BGE-M3 lyric-chunk embeddings are averaged and normalized to create the primary repertoire centroid. A sensitivity centroid is recomputed after removing exact clean text that occurs under more than one source-credit label. The representation uses written lyrics only; it contains no audio, beat, voice, delivery, or flow information.

## Released edge rule

For each representation, cosine similarity is computed between eligible label centroids. An undirected edge is released only when both endpoints rank each other within their five nearest neighbours in **both** the primary and shared-text-exclusion sensitivity representations. This produces 86 edges among 204 eligible labels, with 93 labels connected.

The explanatory signals attached to an edge are calculated after edge selection. They may identify unusually concordant characteristic wording, dictionary-estimated written line endings, or writing-form measures. If none passes its support-stratified 90th-percentile evidence gate, the edge is labelled `semanticOnly`. These signals help interpret an already selected BGE-M3 edge; they are not a causal decomposition of the embedding model.

## Repeatability

The graph is recomputed for 250 within-label song-bootstrap resamples. The reported probability is the fraction of resamples in which the same two-representation edge is selected. Sixteen of 86 edges reappear in at least 50% of resamples; none reaches 80%. This is a repeatability diagnostic under corpus resampling, not a posterior confidence probability and not the probability that two people have a relationship.

## Global layout

The display coordinates are a deterministic two-dimensional PCA projection of normalized consensus primary-plus-sensitivity centroids across all 204 eligible labels. The two axes explain 26.2% of the representation variance. Nearby points therefore suggest approximate dense-semantic proximity, while only the stricter reciprocal two-representation rule defines released edges. A disconnected point means only that no pair involving that label passed the released edge rule.

## Reproducible public join

The release-site builder joins:

1. graph-eligible nodes with PCA coordinates;
2. all 86 retained graph edges;
3. one profile per eligible source-label key;
4. one post-hoc explanation and one bootstrap row per retained edge; and
5. the independently released retrieval, provisional NER, and written-rhyme artifacts.

The build fails on missing or duplicate keys, non-passing validations, manifest hash mismatches, unexpected population counts, or loss of a downstream join. Only the derived site IDs (`l001`, `l002`, ...) are exposed to the browser.
