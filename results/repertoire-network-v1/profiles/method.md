# Interpretable lyrical-profile method

## Purpose

This artifact explains, rather than replaces, the validated BGE-M3 repertoire
graph. A stable line is still defined only by reciprocal top-5 proximity in the
duplicate-weighted and exact-cross-label-shared-text-exclusion representations.
The language, written-ending, and form signals are post-hoc concordant evidence.

## Characteristic language

Chinese content words are segmented with Jieba and compared with the rest of
the corpus using weighted log-odds with an informative Dirichlet prior (prior
mass 1000). A displayed term must occur with at least
4 effective count across at least
5 songs, have z >= 2.0 in both the primary
and shared-text-exclusion representations, and remain positive after leaving
out one song in at least 80% of checks. These are
*characteristic terms*, not favourite words or beliefs.

## Written line endings

For every nonempty written line, the final Han character is mapped to a
dictionary Mandarin pinyin final. Exact repeated lines within a song count once.
Song-level final distributions are normalized before they are averaged. A local
echo statistic measures whether a line repeats a final found in the preceding
four lines and compares that rate with the analytic expectation after randomly
reordering the same song's endings. This gives an orthographic,
dictionary-estimated ending profile. Polyphones,
dialect pronunciation, delivery, timing, tone, beat, and Flow are not resolved.

## Writing-form traits

Three song-level quantities are summarized for each source-credit label:
written line length, exact line repetition within a song, and the share of lines
that contain both Chinese and English. Public bars are percentiles among the 204
eligible labels. Ninety-percent bootstrap intervals resample songs with a fixed,
label-specific seed.

## Pair explanations

Each of the 20,706 eligible-label pairs receives a language-overlap score, a
Jensen-Shannon similarity of written-ending distributions, and a robust-scaled
form similarity. Percentiles are calibrated within minimum-song support strata.
Only a signal at or above the 90th percentile is
allowed to explain a stable graph edge. A stable semantic edge may correctly
remain `semantic-only` if no interpretable signal passes the gate.

## Entity/reference status

The existing 90-surface reference ledger is a screened candidate vocabulary,
not occurrence-level NER. Because the planned two-reviewer context annotation is
not complete, this artifact publishes no person, place, organization, material,
or event occurrence as a result.

## Public-data boundary

The public artifact contains aggregate source-credit-label profiles and stable
edge explanations only. It contains no lyric lines, song/chunk identifiers,
embeddings, membership tables, or private review contexts.
