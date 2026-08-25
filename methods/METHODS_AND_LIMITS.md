# Methods and claim boundaries

This note explains what the project does after lyric chunks have been converted into vectors. It is a compact companion to the manuscript, not a substitute for it.

## The research object

The unit of analysis is a **source-credit label in the frozen corpus**, not a verified person. A label may denote an individual, group, collaboration, alias, or noisy credit string. All public outputs therefore describe observed lyric subsets.

## Why BGE-M3 is used—but not treated as the answer

BGE-M3 is a practical candidate because the corpus contains Chinese, English, mixed-script expressions, and passages of varied length. Its multilingual dense representation provides one view of textual similarity. The project does not assume that this view is automatically superior.

A 1,000-query low-character-overlap, same-song retrieval check compares three systems:

1. BGE-M3 dense cosine similarity.
2. Character 2–5-gram TF-IDF.
3. Equal-weight fusion after within-query score standardization.

Character TF-IDF outperforms BGE-M3 alone on all five reported metrics; fusion performs best. The result supports a multiview design. It is a weak-supervision sanity check, not a human semantic gold standard.

## How vectors become label-level representations

Each exact cleaned-text hash receives total weight one across retained copies. If identical cleaned text occurs several times, each copy receives a smaller share of that weight. Weighted chunk vectors are averaged within a label and normalized.

A second label representation removes every exact cleaned-text hash observed under more than one source-credit label. This sensitivity view asks whether identical cross-label text drives a match.

## How a candidate match is formed

A label must contain at least five clean songs and effective-text mass of at least 20 in both representations. Among the 204 eligible labels, a pair is retained only when:

- each label places the other in its top five neighbours;
- this reciprocal condition holds in the duplicate-weighted representation; and
- it also holds after exact cross-label shared text is removed.

This produces 86 two-representation candidate matches among 93 labels. The two-dimensional map is used only for orientation; it never determines an edge.

## How repeatability is measured

The project runs 250 deterministic within-label song bootstraps. Songs are sampled with replacement, both label representations are rebuilt, and the reciprocal-neighbour graph is recomputed each time. An edge's selection frequency is the share of bootstrap graphs in which it reappears.

Only 16 of the original 86 matches reappear in at least 50% of bootstraps; none reaches 80%. The atlas therefore shows these 16 by default and places the remaining 70 behind an exploratory control. Selection frequency is not a posterior probability or social-relationship confidence.

## How a label becomes understandable

Three independently computed evidence channels translate numerical proximity into readable outputs:

### Distinctive words

Jieba content-word segmentation is followed by duplicate-weighted informative-Dirichlet log-odds against the rest of the corpus. A public term must have sufficient effective count, appear across at least five songs, pass a z-score gate in both text representations, and remain positive in at least 80% of leave-one-song checks.

### Written ending sounds

The final Mandarin sound of the last Han character in each non-empty written line is estimated from a pronunciation dictionary. Exact repeated lines count once within a song. Song-level distributions are normalized before label aggregation. This describes transcribed written endings, not performed rhyme, pronunciation, delivery, beat alignment, or Flow.

### Writing habits

Three song-level measures are reported as corpus percentiles with 90% song-bootstrap intervals: typical effective line length, exact line repetition, and Chinese–English mixing.

## How a pair receives an explanation

All 20,706 eligible-label pairs receive lexical-overlap, ending-distribution, and writing-form similarity scores. Scores are calibrated within comparable minimum-song support strata. A readable signal appears only above the 90th percentile of its stratum.

Of the 16 default matches, 11 have at least one passing readable signal; five remain explicitly unexplained by the implemented channels. The interface does not invent a narrative for those five.

## What the project does not claim

It does not infer biography, hometown, identity, friendship, collaboration, influence, affiliation, genre, artistic preference, performed rhyme, Flow, voice, or beat. A place or person name in lyrics is not proof of a real-world relationship. Named-entity and cultural-reference results remain withheld until span-level human annotation and entity linking are complete.

## Public-release boundary

The public package includes aggregate profiles, pair evidence, figures, tables, validation summaries, and a self-contained atlas. It excludes full lyrics, song and chunk identifiers, embeddings, private membership rows, and unreviewed named-entity contexts.

