# Research protocol

## Representation

Each active clean lyric chunk has a validated BGE-M3 embedding. For every
source artist label, the primary centroid is the L2-normalized weighted mean of
chunk embeddings. Within the metadata-eligible comparison population, every
exact clean-text hash is assigned total mass one across its retained copies.
The weights are recomputed after the conservative song filter, so a surviving
copy is not penalized for a duplicate copy that was excluded from comparison.

## Mandatory sensitivity

An exact clean-text hash used under more than one source artist label is removed
entirely for the sensitivity centroid. A graph edge is displayed only when it
is mutual top-five in both representations. This removes apparent closeness
that can be driven solely by shared text across labels.

## Support, graph, and spatial-projection rule

A source label needs at least 5 clean songs and 20 effective clean-text mass
in **both** representations. Pairwise cosine is ranked within the eligible
corpus labels. No absolute cosine threshold is used. Edge membership is defined
only by the mutual-neighbour rule above.

For the public map, every graph-eligible label receives a deterministic 2D PCA
coordinate from its normalized consensus (primary plus sensitivity) centroid.
Near positions are an approximate visual summary of textual semantic proximity;
the two-dimensional map necessarily discards information. A line remains the
stricter result: it is drawn only for a retained mutual-top-five edge.

## Non-claims

Do not read an edge as collaboration, friendship, influence, affiliation,
location, genre, Flow, vocal technique, beat choice, or personal preference.
The data describe only this frozen, unevenly collected corpus slice.
