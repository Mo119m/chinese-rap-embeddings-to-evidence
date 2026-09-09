# Retrieval on corpus v2: three spaces, one protocol

Every file here scores the same 7,236 query songs against the same 226 source-credit labels under the same leave-group-out protocol over 5,888 leakage groups (exact shared text ∪ near-duplicate songs). A difference between files is a difference in representation or in an experiment's controls, never in the query set.

| File | Built by | What it holds |
| --- | --- | --- |
| `analysis_summary.json` | `src/build_downstream_retrieval_v2.py` | Headline macro (per-label) metrics with paired two-stage bootstrap intervals for BGE-M3, character TF–IDF, their fusion, and the leakage-unit ablations; the embedding run contract; the corpus digest. |
| `metrics.csv`, `uncertainty.csv` | same | The same estimates and paired system differences in the table shape the figure pipeline reads. |
| `query_unit_comparison.json` | `src/retrieval_query_unit_v2.py` | Song centroid against best-matching chunk as the query unit, identical profiles. |
| `identity_spaces.json` | `src/identity_spaces_v2.py` | The rhyme-form space (no lexical content) beside the semantic and lexical spaces; fusions; paired leakage-group contrasts; coverage. |
| `lexical_identity_decomposition.json` | `src/lexical_identity_decomposition_v2.py` | The correct label's lexical score by script class; Latin-share quartiles; four named-reference neutralisation arms. |
| `identity_probe.json` | `src/identity_probe_v2.py` | Within-author whitening of the BGE-M3 vectors, five-fold by leakage group, with total-covariance, permuted-label, and centring controls; the matched-dimension lexical control; transfer to held-out authors. |
| `knn_author_purity.json` | `src/build_identity_spaces_figure_v2.py` | Same-label share of the k nearest neighbours in each space over all songs, and the figure 5 layout set description. |
| `identity_estimands.json` | `src/identity_estimands_v2.py` | Every system of the two experiments above under micro, component-weighted, and macro averaging, so the manuscript can say which it quotes. |
| `lexical_identity_anatomy.json` | `src/lexical_identity_anatomy_v2.py` | The character space taken apart by class of n-gram (script, length, line-boundary, function-character strings, digits and punctuation): score shares, remove-one and only-one arms with paired intervals. |
| `representation_unit.json` | `src/representation_unit_v2.py` | The song's representation changed one choice at a time: chunk MaxSim, chunk-level queries, whitening before or after averaging, single characters, short and long n-grams, jieba words. |
| `word_identity_anatomy.json` | `src/word_identity_anatomy_v2.py` | The word space taken apart by part of speech, document-frequency band and bigram status, plus the arm with the 605-surface catalogue stripped before segmentation. |
| `word_space_probe.json` | `src/word_space_probe_v2.py` | The word space reduced fold-wise to 1,024 dimensions and given the probe's transforms; fused with the chunk-whitened semantic space, the best system tried. |
| `common_word_curve.json` | `src/common_word_curve_v2.py` | Only the K most common words, for a ladder of K, original and neutralised. |

All numbers are aggregate. Per-query ranks, coordinates, and any table that names a song are private.

These results replace `results/retrieval-v1/`. They are not comparable to it point to point: the corpus population differs, and the v1 vectors were produced under an unrecorded configuration that `results/embedding-configuration-probe-v1/` bounds but does not identify. No difference between v1 and v2 numbers may be reported as an effect.
