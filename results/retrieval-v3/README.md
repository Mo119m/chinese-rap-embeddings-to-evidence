# Every headline experiment, rerun on corpus v3 (build 1.2.0)

Corpus v3 is corpus v2 with credit blocks, copied titles, track lists, HTML remnants and two
television-episode transcripts removed from the lyric text (`results/cleaned-corpus-v3/`):
0.75% of lines, 24,277 chunks, 7,379 songs, content digest `0798a37c…`. The semantic
vectors are a local BGE-M3 run over the v3 text (fp16, pinned revision; contract in the
private embedding run). Every file in this directory was produced on 2026-09-09 by the
same scripts as `results/retrieval-v2/`, run with `CHINESE_RAP_CORPUS=v3`, so the protocol
is unchanged: leave-group-out label profiles, 7,231 queries over 226 labels in 5,882
leakage groups, per-(group, label) weight one, paired group bootstrap (2,000 replicates,
seed 20260825). The two earlier v3 builds (1.0.0, 1.1.0) deleted lyrics and their numbers
were never published; nothing here was computed on them.

The v2 numbers are quoted for direction only: the population differs by 160 songs and the
text of 3,607 chunks.

## The three spaces (`three_spaces.json`, `identity_spaces.json`)

| system | v3 MRR | v2 MRR |
|---|---|---|
| semantic, BGE-M3 song mean | 0.3015 | 0.3181 |
| lexical, character 2–5-grams | 0.4275 | 0.4503 |
| lexical, jieba words | **0.4977** | 0.5188 |
| jieba words, 605-surface catalogue stripped | 0.4911 | 0.5113 |
| jieba words, other-script lines dropped | 0.5005 | — |
| rhyme form, strict | 0.0965 | 0.0967 |
| rhyme form with finals | 0.1118 | 0.1128 |
| label-frequency prior | 0.0188 | 0.0188 |
| fusion semantic + character | 0.4476 | 0.4756 |

Paired contrasts (v3): character − semantic +0.132 [+0.122, +0.142]; words − character
+0.072 [+0.065, +0.079]; catalogue-stripped − words −0.006 [−0.009, −0.004]; other-script
lines dropped − words +0.003 [+0.001, +0.004]; rhyme strict − prior +0.079 [+0.074, +0.084].

Reading. The ordering words > characters > semantic > rhyme form is unchanged. Every
text-bearing space lost about two points to the cleaning and the rhyme space lost nothing:
the credit lines were worth about two points of MRR in every space, which is exactly the
cheap signal the cleaning was meant to remove. The 1,300 lines in Uyghur, Tibetan, Korean,
Mongolian and kana (146 songs) are not a cheap identity signal in the word space: dropping
them raises the word MRR slightly, because jieba fragments them into noise.

## Is identity absent from the semantic space, or hidden? (`identity_probe.json`)

| transform of BGE-M3 song vectors | v3 MRR | v2 MRR |
|---|---|---|
| none | 0.3015 | 0.3181 |
| centred | 0.2986 | 0.3136 |
| total whitening | 0.4016 | 0.4316 |
| within-author whitening | **0.4196** | 0.4494 |
| within-author whitening, permuted labels | 0.4021 | 0.4306 |
| fusion with characters, no whitening | 0.4476 | 0.4756 |
| fusion with characters, within-author whitening | 0.5147 | 0.5436 |
| lexical SVD-1024, none / total / within | 0.3722 / 0.4840 / 0.4936 | 0.3933 / 0.5114 / 0.5183 |
| whitened lexical SVD + whitened semantic | 0.5555 | 0.5871 |

Held-out labels (15%): none 0.2707 → total whitening 0.4129 → within-author whitening
0.3833; seen labels under within-author whitening 0.3837. The label-free part of the gain
transfers to unseen labels; the label-specific part does not.

## Where the word signal sits (`word_identity_anatomy.json`, `common_word_curve.json`)

Unigrams alone 0.5133 (bigrams alone 0.4133); the commonest document-frequency quartile
alone 0.4962, the rarest quartile undefined (empty repertoires); English tokens alone
0.3296; numerals alone 0.1238; punctuation alone 0.1683. Common-word ladder (unigram-only
fit, all 49,424 unigrams 0.5132): top 50 words 0.1770, top 100 0.2319, top 200 0.2965,
top 500 0.3671, top 1,000 0.4133, top 2,000 0.4466, top 5,000 0.4753, top 10,000 0.4957,
top 20,000 0.5106. Stripping the 605-surface catalogue moves every rung by at most 0.007.

## Where the character signal sits (`lexical_identity_anatomy.json`)

Only 2-grams 0.4146, 3-grams 0.3919, 4-grams 0.3416, 5-grams 0.2928; only Han n-grams with
a less common character 0.4257; only function-character strings 0.2772; only Latin 0.3105;
only boundary-crossing 0.2573; only digit or punctuation 0.1901 (full model 0.4275).

## Representation unit (`representation_unit.json`, `word_space_probe.json`)

Whitening chunk vectors before averaging beats whitening the song mean, 0.4316 vs 0.4196
(+0.012 [+0.009, +0.015]); chunk max-sim is worse than the song mean (−0.016). Character
unigrams 0.4155, 2–3-grams 0.4180, 4–5-grams 0.3530, words 0.4977. Word SVD-1024: none
0.4428, total whitening 0.5490, within-author whitening 0.5446; **whitened words + chunk-
whitened semantic 0.5859**, the best system on v3 (v2 0.6082); the same fusion without
whitening 0.4899 (v2 0.5153).

## Chunk-level replication (`chunk_level_replication.json`)

At the chunk level (23,898 chunks, each scored on its own): semantic 0.1887, characters
0.2596, words 0.2963, fusion 0.3005; the ordering holds and the song mean of the chunk
scores recovers 0.2531 / 0.3472 / 0.3962.

## Label-permutation null (`label_permutation_null.json`)

Ten permutations of the label column: semantic 0.031–0.035, characters 0.033–0.040, words
0.035–0.040, against 0.3015 / 0.4275 / 0.4977 observed and a prior of 0.019.

## Semantic arm of the neutralisation test (`semantic_neutralisation.json`)

BGE-M3 vectors of the catalogue-stripped v3 text against the original: none 0.2932 vs
0.3015 (−0.008 [−0.012, −0.005]); total whitening 0.3888 vs 0.4016 (−0.013); within-author
whitening 0.4046 vs 0.4196 (−0.015). The identity the semantic model keeps is not the names.

## Two-covariance model, assumption ablation (`two_covariance_ablation.json`)

The model predicts the ordering of transforms and roughly twice their size. Four arms
resample songs as label mean + residual, with each component either Gaussian (from the
fitted covariances) or the real vectors reassigned across labels:

| arm | none | total whitening | within-author whitening |
|---|---|---|---|
| observed | 0.3015 | 0.4016 | 0.4196 |
| Gaussian means + Gaussian residuals | 0.636 | 0.999 | 1.000 |
| real means + Gaussian residuals | 0.628 | 0.993 | 0.993 |
| Gaussian means + real residuals | 0.622 | 0.971 | 0.992 |
| real means + real residuals, reassigned across labels | 0.600 | 0.944 | 0.948 |

No arm meets the observed value, so the error is in neither marginal. What every arm still
assumes is that a song's residual is exchangeable across labels: reassigning real residuals
to random labels is what all four share, and it doubles the MRR. Within-label variation is
label-specific — how far a rapper's songs scatter, and in which directions, depends on the
rapper — and a shared within-class covariance, which is also PLDA's assumption, cannot
represent it.

## Training-data audit for the identity encoder (`training_data_audit.json`)

Tokens per chunk median 76, p90 603, 3,676 over 512; all 226 labels have at least two
leakage groups; 593 songs (8.2%) carry a collaboration title; 1,636 chunks (6.7%) name their
own label; a chunk's nearest content rival (another label) is closer than its nearest
same-label chunk in 90.8% of cases (median cosine 0.725 vs 0.663).

## Privacy

Aggregate numbers only. Per-query tables, vectors and text stay under the private root.
