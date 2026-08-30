# Method: Chinese written-rhyme modelling V1

## Scope and claim boundary

The task predicts the tone-free pinyin-final family of the next **written lyric line ending**. It does not observe audio and therefore does not measure performed rhyme, flow, cadence, stress, beat alignment, regional pronunciation, or delivery. Corpus labels are source-credit labels whose real-world identities have not been externally verified.

## Population and line construction

The input population is the 204 graph-eligible source-credit labels from `chinese-rap-lyrical-repertoire-graph-v2`. Only clean-text chunks retained by the graph's shared-clean-text exclusion sensitivity are used. Lines are defined by the cleaned source's written newline boundaries. The primary estimand requires the final non-space/non-punctuation written character itself to be Han; lines ending in Latin letters, digits, emoji, or other non-Han symbols are counted and excluded. Empty lines, explicit section/credit headers, lines without Han characters, and unclassifiable endings are also excluded.

Within the frozen chunk-deduplicated snapshot, repeated written lines are retained in their surviving positions. Prediction events are created only when two retained lines were adjacent in the original same chunk (`current line index = previous line index + 1`). An excluded line or a chunk boundary breaks the sequence, so no synthetic bridged transition is formed. Exact repeats within each source-label/song unit are flagged, and non-repeat-target metrics are released as a sensitivity stratum without reconnecting the surrounding lines. A post-freeze source-lineage audit found that legacy preprocessing had already removed exact duplicate chunks within source-credit labels; `results/corpus-reconciliation-v1/` quantifies that population change. Aggregate ending-family and switch-rate sensitivity is reported, but predictive robustness is withheld until duplicate-aware retraining.

## Written-ending representation

`pypinyin 0.55.0` is run on the complete Han sequence of each line with `Style.FINALS_TONE3`, `strict=True`, and neutral tone 5 so phrase context can disambiguate some polyphones. The final and penultimate Han syllables are stored as strict pinyin finals, tones, a tone-free final family, and a two-syllable family pattern. The 17 deterministic families are listed in `rhyme_class_inventory.csv`. They are transparent analytical normalisations, not an official or performance-sensitive rhyme standard.

## Descriptive fingerprints

For each source-credit label, the release reports dominant and distinctive ending families, raw finals and tones, adjacent continuation/switch rates, membership in same-family runs, four-line local echo, its independent-draw frequency expectation, and a two-syllable pattern echo statistic. These are corpus summaries, not intrinsic artist traits.

## Song-held-out prediction

Songs, including songs attached to more than one source-credit label, are assigned globally to train/validation/test partitions at approximately 70/15/15. A deterministic multilabel-aware greedy allocation plus repair guarantees every eligible label has at least one song in every partition. No song crosses partitions. The target is the ending family of the exact next originally adjacent strict-Han-ending written line; features include up to four preceding contiguous families, previous raw final/tone, prior two-syllable pattern, previous transition, current same-family run bucket, contiguous-segment position bucket, recent family diversity, and source-credit label.

Five systems are compared: (1) add-one global target frequency; (2) a first-order Markov model with a global empirical-Bayes backoff of strength 20; (3) flat one-hot context features with multinomial `SGDClassifier(loss='log_loss', average=True)`; (4) the released hierarchical context model, which combines a binary continue-versus-switch classifier with a switch-only family classifier that cannot recommend the previous family in its switch branch; and (5) an otherwise identical hierarchical ablation with the source-credit-label feature removed. The label-conditioned model is not called personalised unless paired held-out intervals against this ablation support that claim. SGD regularisation is selected from 1e-5, 1e-4, and 1e-3 using validation MRR, and probability temperature is selected using validation negative log likelihood.

## Duplicate leakage control

All normalised train written lines, including isolated/first lines that are not prediction targets, form the validation leakage reference. Test targets are checked against the union of all retained train and validation written lines. Validation/test events are excluded if their target line exactly matches the applicable reference or has character-trigram Jaccard similarity >= 0.80; candidates are found by an exact inverted index and verified with exact Jaccard. Lines shorter than three normalised characters receive the exact check only. Models never receive lyric text, but this target filter prevents copied/reissued written lines from inflating either validation or final test scores.

## Evaluation and uncertainty

Primary metrics are top-1/top-3/top-5 accuracy, mean reciprocal rank (MRR), negative log likelihood, ten-bin expected calibration error, and two explicit coverage denominators: model coverage among leakage-safe events and end-to-end coverage among all strict-primary test candidates before leakage filtering. Ninety-five-percent intervals for top-k accuracy and MRR use 2000 resamples of held-out songs, preserving within-song dependence. All 204 labels receive a row for every model; metrics are suppressed below 20 leakage-safe test events rather than silently dropping labels. Stratified results separate continuation/switch, prior run length, segment position, and repeated/non-repeated targets. The independent-draw expectation for local four-line echo uses the actual available predecessor window of 1, 2, 3, or 4 at each position rather than assuming four predecessors for opening lines.

## Abstention and recommendation

Selective thresholds are fixed from validation-confidence quantiles targeting 75%, 50%, and 25% coverage, then applied once to test. `recommender_lookup.json` releases global, Markov, and common aggregate ML contexts only. It contains no lyric line, song/chunk ID, or content hash. A recommendation is a likely next **class**, not a generated word, a quality score, or a performance prescription.

## MuChin auxiliary agreement check

MuChin V1 1000 is never used for training, model selection, or primary evaluation. The publisher-recommended 1,000 folders are used instead of the duplicate-contaminated combined release. For exactly line/section-aligned `str_lyric`/`str_rhyme` pairs, a line is marked by our rule if another line in the same section has the same derived family, and this is compared with exported `R` markers. This check is only partially independent: MuChin's UI automatically grouped and colour-highlighted line-final pinyin rhymes, while annotators mainly checked polyphonic pronunciations; exported `R` collapses class identity. The result is therefore an implementation sanity/agreement check, not independent gold validation, Chinese-rap domain validation, or audio/performance validation.
