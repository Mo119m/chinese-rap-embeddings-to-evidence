# Null baseline for the reciprocal top-five graph

Produced by `tools/null_baseline_reciprocal_edges.py` against the private embedding
and membership artefacts. The release reports 86 mutual top-five edges connecting 93
of 204 labels, but states nothing about how many such edges the rule yields when the
label-to-song assignment carries no information, so a reader cannot judge whether 86
is a lot.

## Method

Songs are redistributed among labels by permuting the label column of the label-song
incidence, which preserves every label's song count and every song's label count, so
label size and collaboration structure survive while the association between a label
and its lyrical content is destroyed. Centroids are rebuilt from the chunk
embeddings and the mutual top-five rule is reapplied under both the primary and the
shared-text-exclusion representations. 200 replicates.

Both centroids weight member chunks by `comparison_text_weight` and differ only in
which rows they include; the weight sum must equal the effective text mass recorded
in the node rowmap. The script rebuilds the observed graph first and refuses to
continue unless it reproduces the published 86 edges and 93 labels exactly.

## Result

| | edges | connected labels |
| --- | ---: | ---: |
| observed | 86 | 93 |
| null mean | 45.0 | 52.2 |
| null sd | 6.7 | – |
| null max over 200 replicates | 68 | 71 |
| null 95th percentile | 55.0 | – |

No replicate reached the observed count, giving a one-sided permutation
**p = 0.005**, the floor for 200 replicates. The observed graph is **1.91 times** the
null mean and sits about **6.1 standard deviations** above it.

## Reading

The observed graph is denser than chance by a wide margin, and that margin is the
result.

The null mean is also worth stating plainly, because it is not small: **an
uninformative label assignment still produces about 45 reciprocal top-five edges
connecting about 52 labels.** Reciprocity is not rare when 204 nodes each choose five
neighbours out of 203, and this corpus is generically similar — every document is
Chinese rap lyrics and centroid cosines sit high — so random groupings still yield
centroids that select each other. A reader who is told only that the rule produced 86
edges has no way to know that roughly 45 come free.

**What this does not license.** It does not say that any particular edge is spurious,
and it is not a false-discovery rate. The null describes how many edges the *rule*
yields on shuffled labels; it cannot be inverted to label individual observed edges as
real or not. An earlier version of this file said "roughly half the graph is at the
density chance alone supplies", which reads as though half the released edges were
chance artefacts. That inference is not available from this test and has been removed.

## Scope and what would strengthen it

The permutation holds two things fixed that the released pipeline derives from the
data: the shared-text mask and the 204-label eligibility set. Both were computed under
the true label assignment, so this is a **conditional** null — it asks whether the
label-to-content association matters *given* that masking and eligibility. Recomputing
both inside each replicate would be the unconditional version and is the obvious
sensitivity analysis to add.

## Relationship to the robustness null on the other branch

`codex/release-integrity-publication` adds a second null under
`results/repertoire-network-v1/robustness/`. **They are not duplicates and neither
replaces the other.**

That one holds the primary layer's mutual top-five edges fixed and rewires the
sensitivity layer with degree-preserving double-edge swaps, asking whether the
86-edge cross-treatment intersection is non-random. Its null mean is about 4.5 against
an observed 86 — overwhelming.

It cannot answer the question this one asks. Conditioning on the primary layer keeps
the real label-to-content association intact, so no amount of rewiring the second
layer can show how much of the edge count survives an uninformative labelling. Only
permuting the assignment itself shows that.

Report both. The cross-treatment result says the two text treatments agree far beyond
chance. The permutation result says the rule yields about 45 edges on shuffled labels
against 86 observed. A reader needs the second to size the first.

## Boundary

This tests whether the reciprocal top-five rule recovers more structure than an
uninformative label assignment. It is not evidence about artists, influence,
collaboration, or social relation, and it says nothing about whether any individual
edge is meaningful.
