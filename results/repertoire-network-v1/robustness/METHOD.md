# Method

## Cross-treatment graph-alignment null

The primary and shared-text-excluded BGE-M3 consensus layers contain 140 and 145 reciprocal top-5 edges. Their observed intersection contains 86. In each of 10,000 deterministic Monte Carlo replicates (seed 20260827), the sensitivity layer receives 1,450 successful double-edge swaps. This preserves every node's sensitivity-layer degree while randomizing its neighbours. The statistic is its edge intersection with the fixed primary layer. The add-one Monte Carlo p-value is `(exceedances + 1) / (replicates + 1)`.

An auxiliary 100,000-replicate source-label permutation null breaks layer correspondence entirely. The degree-preserving test is primary because it controls labels' different connection propensities. Neither null tests whether the semantic structure itself is random or gives an edge-specific p-value.

## Projection fidelity

The high-dimensional reference is the unit-normalized sum of each label's primary and sensitivity vectors, matching the public PCA builder. Trustworthiness and exact neighbourhood overlap are reported at k=5, 10, and 15. Pairwise Spearman correlation compares high-dimensional cosine order with negative two-dimensional Euclidean distance. Released-edge retention in the two-dimensional top-five graph is diagnostic only.

Private vectors and row maps are used only to compute aggregate statistics. They are not copied into this public artifact.
