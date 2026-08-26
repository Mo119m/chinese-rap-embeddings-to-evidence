# Lyrical repertoire network v1

This directory is the publication-safe, aggregate source for the repertoire network shown by the result site. It replaces the earlier circular build in which the site-data builder read its own previously generated JSON.

## What the network contains

- **204 source-credit labels** meet the fixed support thresholds: at least 5 clean songs and 20 units of duplicate-controlled effective text mass in both representations.
- **86 released edges** satisfy the reciprocal top-5 rule in both the primary BGE-M3 representation and the shared-text-exclusion sensitivity representation.
- **93 labels** have at least one released edge. The other 111 labels remain visible in the global map but have no edge that passes the released rule; this does not mean that they have no lyrical neighbours.
- **16 of the 86 edges** reappear in at least 50% of 250 within-label song-bootstrap resamples. No edge reaches 80%.
- The two-dimensional display is a deterministic PCA projection of the normalized consensus representations. Its first two components explain **26.2%** of representation variance, so distance on the page is approximate and never creates an edge.

## Safe-publication boundary

The files here contain aggregate source-label profiles, coordinates, edge summaries, bootstrap frequencies, methods, manifests, and validation results. They contain no lyric text, song or chunk identifiers, membership rows, or embeddings. Hashed `ALBL-...` values are internal source-label join keys, not person identifiers.

The network describes **lyrical-repertoire proximity between source-credit-labelled corpus slices**. It is not evidence of biography, verified identity, popularity, genre membership, collaboration, influence, friendship, or any other social relationship. The source labels themselves have not yet received exhaustive external identity verification.

## Relationship to the retrieval benchmark

This map is deliberately based on duplicate-controlled **BGE-M3 lyric-chunk centroids**, because it is an explanatory map of dense semantic repertoires. It is not the fusion model evaluated in the held-out retrieval task. In that separate task, the BGE-M3 and character TF-IDF scores are fused, and the fusion system performs better than either component alone. The site and paper must keep these two uses separate.

## Directory layout

- `graph/`: eligible nodes, retained edges, PCA layout, robustness summary, protocol, manifest, and validation.
- `profiles/`: aggregate characteristic-language and writing-form profiles plus post-hoc edge explanations.
- `bootstrap/`: 250-resample edge repeatability estimates and their validation.
- `METHOD.md`: concise reconstruction and interpretation contract for downstream consumers.
- `manifest.json`: component identities, fixed expected counts, and hashes of the frozen component manifests and validations.

All three component `validation.json` files report `pass`. The site-data builder independently checks those statuses, verifies the component payload hashes listed by their manifests, reconstructs all joins, and enforces the fixed 204/86/93/16 counts before writing public site JSON.
