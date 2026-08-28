# Protocol Amendment PD-002: Upstream chunk deduplication

## Status and timing

PD-002 is a post-freeze amendment recorded on 28 August 2026 after authorized project access to the private Drive source and inspection of the legacy cleaner. It does not pretend that the issue was anticipated. It preserves the V1 results as estimates conditional on their declared frozen snapshot, corrects overbroad repeat-retention wording, and defines the required duplicate-aware repair before a stronger corpus claim or predictive robustness claim is made.

## Live Drive reconciliation

The live native Google Sheet was compared row by row with the local raw CSV used below. Both contain 26,833 aligned records; song and chunk keys match exactly. Of 161 exact title differences, 132 are Unicode/whitespace-equivalent and 29 are native-sheet conversions of string-like titles to numbers, times, or percentages. Of 23,836 exact text differences, 23,832 are newline representation only, two are leading-apostrophe escape semantics, and two arise because DEL control characters cause the sheet import to remove both the control and its preceding character. Seven artist differences are Unicode/whitespace-equivalent. All 33 substantive cell differences are adjudicated and none remains unresolved.

The local CSV is therefore retained as the lineage source: it preserves intended string titles and leading apostrophes. Corpus v2 removes the two DEL characters while preserving their preceding characters; it does not copy the sheet's destructive import side effect. This comparison establishes aligned keys and documented cell-level differences, not byte identity between Drive and the local export.

## Discovery

The historical cleaner applies four ordered operations before the canonical corpus builder receives lyric chunks:

1. exclude titles marked Live, 伴奏, or Instrumental;
2. remove configured credit, structure, copyright, and speaker-label material;
3. remove chunks that become empty; and
4. keep the first exact `(source-credit label, cleaned chunk text)` occurrence.

The fourth operation controls copied text by deleting source rows. It conflates three different objects: duplicate ingestion records, legitimate repetition inside a song, and exact text reused across distinct song records. It can therefore erase song identity and original sequence before downstream leakage controls are applied.

## Reconciliation result

The aggregate-only builder `src/build_corpus_reconciliation_v1.py` exactly reconstructs the legacy pre-canonical 22,132-row, 7,214-song cleaned snapshot from the local Drive-matched raw chunk export. The subsequent canonical-reconciliation gates exclude four chunks across three song IDs, yielding the 22,128-chunk, 7,211-song canonical snapshot used in Table 1. The legacy cleaning stages are:

| Stage | Chunk rows | Song IDs | Rows removed at stage | Song IDs removed at stage |
| --- | ---: | ---: | ---: | ---: |
| Raw chunk export | 26,833 | 7,721 | 0 | 0 |
| Title exclusions | 25,279 | 7,420 | 1,554 | 301 |
| Line cleaning and empty removal | 25,026 | 7,391 | 253 | 29 |
| Artist + exact-cleaned-text keep-first | 22,132 | 7,214 | 2,894 | 177 |

The metadata discrepancy is fully reconciled: 118 of 7,839 nonblank metadata IDs never appear in the raw chunk export, and 507 raw-export IDs are removed by the three cleaning stages, yielding the 625 metadata-only rows already recorded in the corpus manifest.

Among the 177 song IDs erased entirely by chunk deduplication:

- 136 have the same complete cleaned chunk sequence as a retained song under the same source-credit label;
- 33 are a cleaned-chunk multiset subset of one retained song;
- 8 have chunks whose first retained occurrences are distributed across multiple songs; and
- 0 remain unreconciled.

Of the 136 exact-sequence cases, 131 also share the same normalized title and form the high-confidence duplicate-record stratum used by this amendment. The other 46 erased song records remain a review queue, not automatically verified duplicates. The primary rule is intentionally conservative and does not establish real-world work identity, reissue status, authorship, or performer identity.

## Effect on written-ending analysis

At the legacy full cleaned-source level, omitting the artist-level chunk deletion restores 16,623 strict-Han-ending line occurrences and 13,112 originally adjacent transitions. This broad counterfactual intentionally uses the exact legacy post-line-cleaning text so that it isolates the deduplication stage; it precedes the later canonical leading-header pass. The global 17-family distribution changes by total-variation distance 0.001767 and the switch rate changes by 0.001787 in absolute terms. These small aggregate shifts support stability of the corpuswide ending-family description, while the count and repetition losses remain material.

Inside the released 204-label, shared-text-exclusion rhyme universe, reconnecting eligible pre-snapshot chunks without retraining restores 7,033 strict-Han-ending occurrences, 4,938 adjacent transitions, and 6,472 additional within-label/song repeated-line occurrences beyond first occurrences. The family-distribution total-variation distance is 0.001789 and the switch-rate difference is 0.001390. This counterfactual does **not** establish that top-k accuracy, MRR, calibration, abstention, or paired model differences are stable, because those models have not yet been retrained on the repaired population.

The released phrase “repeated written lines are retained” is therefore restricted to repeats inside chunks that survived the legacy preprocessing. The fixed task universe contains 5,619 songs, of which 5,452 contribute at least one strict-Han-ending line and 5,347 contribute at least one adjacent transition. The reported 52,152 repeat count means additional occurrences beyond each normalized line's first within-label/song occurrence; it is not the number of all occurrences belonging to repeated groups.

## Replacement corpus rule

The next corpus version must preserve all non-empty cleaned source chunks and their original `(song ID, chunk ID, source order)` relation. Duplicate control is represented, not enacted by deleting chunks:

1. Assign a song-record duplicate group only under a declared song-level rule. The primary automatic stratum requires the same source-credit label, same normalized title, and exact complete cleaned chunk sequence. All other cases enter review.
2. Preserve repeated chunks and lines inside retained songs so hooks and sequence structure remain observable.
3. Assign exact-cleaned-text component IDs across songs. Keep component-linked records together across train, validation, and test or remove the query component from every candidate profile.
4. Give each component total weight one within a source-label aggregate so repeated imports cannot dominate centroids, vocabulary probes, or cultural-reference rates.
5. Report sensitivities for: high-confidence duplicate-record collapse versus preservation; exact-component weighting versus raw occurrence weighting; and the 46-record review queue included versus withheld.

## Downstream consequences

- **Retrieval:** V1 remains an internally valid frozen-corpus benchmark. V2 must rebuild the song universe, keep duplicate components together, and rerun paired metrics and intervals.
- **Repertoire graph:** Exact component weighting should preserve the intended centroid estimand, but song-resampling stability must be rerun because restored song units change the bootstrap population.
- **Cultural-reference evidence:** Support must be aggregated on duplicate-controlled song units. Released edges require a sensitivity to duplicate-record collapse, in addition to occurrence audit and extraction reliability.
- **Written endings:** Within-song repeats must remain in sequence. Cross-song duplicate components must not cross evaluation partitions, and all predictive metrics must be rerun before a robustness claim.

## Release boundary

Public files remain aggregate only. They contain no lyric text, titles, source-credit labels, song/chunk identifiers, embeddings, membership rows, or row-level hashes. The source export itself is not redistributed. `results/corpus-reconciliation-v1/` contains the public counts, input file hashes, method, and validation gates.
