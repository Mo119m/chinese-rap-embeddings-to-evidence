# Song-level edge bootstrap

The bootstrap tests whether each of the 86 existing stable semantic edges is
selected again when the observed songs inside every eligible source-credit
label are resampled with replacement.

For each of 250 deterministic replicates:

1. sample the label's observed songs with replacement;
2. reconstruct duplicate-weighted BGE-M3 centroids for the primary and
   exact-cross-label-shared-text-exclusion representations;
3. L2-normalize the centroids and recompute every cosine similarity;
4. recompute reciprocal top-5 edges separately in both representations;
5. record whether an original edge survives their intersection.

The reported probability is a selection frequency under this empirical
resampling scheme. It is not a Bayesian posterior, a p-value, or evidence of a
social, collaborative, geographic, or influence relationship.
