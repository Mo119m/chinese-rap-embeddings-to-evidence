# Data dictionary

- `primary_effective_text_mass`: sum of comparison-population exact-clean-text
  duplicate-control weights under the source label. The weights are recomputed
  after the metadata-eligibility filter so each retained exact-text group has
  total mass one.
- `shared_text_dropped_mass_share`: primary mass removed when exact cleaned text
  appearing under more than one source label is excluded.
- `primary_rank_*` / `sensitivity_rank_*`: reciprocal nearest-neighbour ranks
  among the eligible labels, not a social rank or popularity rank.
- `*_pair_percentile`: empirical percentile among all eligible label-pair
  cosine values for that representation.
- `stable_across_shared_text_exclusion`: retained only when both mutual-top-five
  tests pass.
- `x` / `y`: deterministic PCA coordinates for all graph-eligible labels;
  distance is an approximate semantic-proximity display, not an edge test.
- `projection_variance_explained_2d`: share of consensus-vector variation
  represented by the two displayed PCA axes.
- `component_id`: stable-edge connected component only. `0` means that an
  otherwise eligible source label has no retained edge; it does not mean that
  the label has no semantic neighbours.

Public files contain only aggregate/source-label graph data. Song IDs, chunk
IDs, lyric text, embeddings, and membership records stay private local-only.
