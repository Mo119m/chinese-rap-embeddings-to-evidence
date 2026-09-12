# Every headline experiment, rerun on corpus v3 (build 1.3.0)

Corpus v3 is corpus v2 with credit blocks, copied titles, track lists, HTML remnants,
production-credit residue and two television-episode transcripts removed from the lyric
text (`results/cleaned-corpus-v3/`): 24,237 chunks, 7,379 songs, content digest
`cd51bf69…`. The semantic vectors are a local BGE-M3 run over the v3 text (fp16, pinned
revision `5617a9f6`, weights `b5e0ce34`; contract in the private embedding run). Every
CPU experiment was produced on 2026-09-10 by the same scripts as `results/retrieval-v2/`,
run with `CHINESE_RAP_CORPUS=v3`; the GPU experiments (surprisal, layer-wise probe,
fine-tuning) followed on 2026-09-11 and 12. The protocol is unchanged throughout:
leave-group-out label profiles, 7,220 queries over 226 labels in 5,875 leakage groups,
per-(group, label) weight one, paired group bootstrap (2,000 replicates, seed 20260825).
`MANIFEST_1.3.0.json` records each file's digest and the build it was computed on.

Three earlier v3 builds existed. 1.0.0 and 1.1.0 deleted lyrics and their numbers were
never published. 1.2.0 was published for one day (commits `1310271`–`bda1d1b`); against
it every number here moves by 0.1–0.5 points in the same direction, and no contrast
changes sign or significance except one noted below. The v2 numbers are quoted for
direction only: the population differs by 171 songs and the text of 3,876 chunks.

## The three spaces (`three_spaces.json`, `identity_spaces.json`)

| system | v3 MRR | v2 MRR |
|---|---|---|
| semantic, BGE-M3 song mean | 0.2997 | 0.3181 |
| lexical, character 2–5-grams | 0.4266 | 0.4503 |
| lexical, jieba words | **0.4963** | 0.5188 |
| jieba words, 605-surface catalogue stripped | 0.4878 | 0.5113 |
| jieba words, other-script lines dropped | 0.4978 | — |
| rhyme form, strict | 0.0954 | 0.0967 |
| rhyme form with finals | 0.1105 | 0.1128 |
| label-frequency prior | 0.0188 | 0.0188 |
| fusion semantic + character | 0.4464 | 0.4756 |

Paired contrasts: character − semantic +0.133 [+0.123, +0.144]; words − character +0.071
[+0.064, +0.079]; catalogue-stripped − words −0.009 [−0.011, −0.007]; other-script lines
dropped − words +0.001 [−0.000, +0.003] (on 1.2.0 this was +0.003 with the interval
clear of zero; on 1.3.0 it is not — the one contrast the build changed); rhyme strict −
prior +0.077.

Reading. The ordering words > characters > semantic > rhyme form is unchanged. Every
text-bearing space lost about two points to the cleaning and the rhyme space lost
nothing: the credit lines were worth about two points of MRR in every space, which is
exactly the cheap signal the cleaning was meant to remove. The 1,300 lines in Uyghur,
Tibetan, Korean, Mongolian and kana are not a cheap identity signal in the word space:
dropping them changes nothing measurable.

## Is identity absent from the semantic space, or hidden? (`identity_probe.json`)

| transform of BGE-M3 song vectors | v3 MRR | v2 MRR |
|---|---|---|
| none | 0.2997 | 0.3181 |
| centred | 0.2967 | 0.3136 |
| total whitening | 0.3996 | 0.4316 |
| within-author whitening | **0.4164** | 0.4494 |
| within-author whitening, permuted labels | 0.3983 | 0.4306 |
| fusion with characters, no whitening | 0.4464 | 0.4756 |
| fusion with characters, within-author whitening | 0.5128 | 0.5436 |
| lexical SVD-1024, none / total / within | 0.3735 / 0.4856 / 0.4921 | 0.3933 / 0.5114 / 0.5183 |
| whitened lexical SVD + whitened semantic | 0.5551 | 0.5871 |

Held-out labels (15%): none 0.2658 → total whitening 0.4072 → within-author whitening
0.3813; seen labels under within-author whitening 0.3814. The label-free part of the gain
transfers to unseen labels; the label-specific part does not.

## Where the word signal sits (`word_identity_anatomy.json`, `common_word_curve.json`)

Unigrams alone 0.5117 (bigrams alone 0.4114); the commonest document-frequency quartile
alone 0.4951, the rarest quartile undefined (empty repertoires) and removable without loss
(0.4980 against 0.4963 with it); the second and third quartiles alone 0.2319 and 0.3449.
By part of speech: content words alone 0.3820, function words alone 0.2622, English tokens
alone 0.3251, person/place names alone 0.2382, numerals 0.1233, punctuation 0.1636;
removing English costs 0.033, removing content words 0.029, removing function words
0.004. Common-word ladder (unigram-only fit, all 49,424 unigrams 0.5112): top 50 words
0.1772, top 100 0.2320, top 200 0.2960, top 500 0.3670, top 1,000 0.4123, top 2,000
0.4447, top 5,000 0.4733, top 10,000 0.4929, top 20,000 0.5102. Stripping the 605-surface
catalogue moves every rung by at most 0.009.

## Where the character signal sits (`lexical_identity_anatomy.json`)

Only 2-grams 0.4131, 3-grams 0.3888, 4-grams 0.3401, 5-grams 0.2887; only Han n-grams with
a less common character 0.4238; only function-character strings 0.2777; only Latin 0.3070;
only boundary-crossing 0.2558; only digit or punctuation 0.1844 (full model 0.4266).

## Representation unit (`representation_unit.json`, `word_space_probe.json`)

Whitening chunk vectors before averaging beats whitening the song mean, 0.4271 vs 0.4164;
chunk max-sim is worse than the song mean (0.2805 vs 0.2997). Character unigrams 0.4141,
2–3-grams 0.4159, 4–5-grams 0.3504, words 0.4963. Word SVD-1024: none 0.4411, total
whitening 0.5479, within-author whitening 0.5426; **whitened words + chunk-whitened
semantic 0.5854** (R@1 0.479, R@10 0.780), the best system on v3 (v2 0.6082); the same
fusion without whitening 0.4894 (v2 0.5153).

## Chunk-level replication (`chunk_level_replication.json`)

At the chunk level (23,848 chunks, each scored on its own): semantic 0.1876, characters
0.2593, words 0.2960, fusion 0.3012; the ordering holds and the song mean of the chunk
scores recovers 0.2509 / 0.3462 / 0.3948.

## Label-permutation null (`label_permutation_null.json`)

Ten permutations of the label column: semantic 0.032–0.035, characters 0.035–0.040, words
0.036–0.040, against 0.2997 / 0.4266 / 0.4963 observed and a prior of 0.019.

## Semantic arm of the neutralisation test (`semantic_neutralisation.json`)

BGE-M3 vectors of the catalogue-stripped v3 text against the original: none 0.2917 vs
0.2997 (−0.008 [−0.011, −0.004]); total whitening 0.3853 vs 0.3996 (−0.014);
within-author whitening 0.4037 vs 0.4164 (−0.013). The identity the semantic model keeps
is not the names.

## Two-covariance model, assumption ablation (`two_covariance_ablation.json`)

The model predicts the ordering of transforms and roughly twice their size. Four arms
resample songs as label mean + residual, each component either Gaussian (from the fitted
covariances) or the real vectors reassigned across labels:

| arm | none | total whitening | within-author whitening |
|---|---|---|---|
| observed | 0.2997 | 0.3996 | 0.4164 |
| Gaussian means + Gaussian residuals | 0.632 | 0.999 | 1.000 |
| real means + Gaussian residuals | 0.629 | 0.993 | 0.993 |
| Gaussian means + real residuals | 0.621 | 0.971 | 0.993 |
| real means + real residuals, reassigned across labels | 0.595 | 0.944 | 0.949 |

No arm meets the observed value, so the error is in neither marginal. What every arm still
assumes is that a song's residual is exchangeable across labels; reassigning real residuals
to random labels is what all four share, and it doubles the MRR.

## What whitening presses, and label-specific scatter (`whitening_directions.json`)

**The pressed directions are topic and register.** The top eight eigenvectors of the
within-label covariance of BGE-M3 song vectors, with the jieba words (document frequency
≥ 50 songs) whose TF-IDF weight correlates most with a song's coordinate:

| direction | within-variance share | between share along it | one end | other end |
|---|---|---|---|---|
| 1 | 4.5% | 0.25 | 却 爱 回忆 你 离开 | 你们 说唱 兄弟 老子 他们 rapper |
| 2 | 2.5% | 0.15 | 啊 天下 江湖 天地 儿 剑 | i me you my |
| 3 | 2.2% | 0.31 | 人 也 是 自己 了 的 | i you me my the it |
| 4 | 2.0% | 0.21 | 天地 江湖 家乡 与 山河 天下 | i me 自己 他们 you |
| 5 | 1.6% | 0.20 | the to 世界 自己 梦想 | 你 说 爱 你 的 |
| 6 | 1.4% | 0.18 | 钱 他 她 买 吃 了 | 你 的 英雄 flow 说唱 |
| 7 | 1.3% | 0.15 | 钱 money 每天 烦恼 买 | 被 却 cypher 谁 了 |
| 8 | 1.2% | 0.14 | 说唱 自己 梦想 生活 音乐 | like 杀 my 地狱 敌人 |

Love song vs. braggadocio, wuxia register vs. English, Chinese vs. English, money vs.
solitude, aspiration vs. violence. Along them 69–86% of the variance is within-label,
against 90.6% over the whole space: these axes carry somewhat more label information than
average — some rappers favour a register — and whitening gives that up for the identity
in the small directions. On 1.2.0 direction 6 ended in production-credit residue; the
1.3.0 cleaning was written from that observation and the residue axis is gone.

**Label scatter is direction-specific.** For the 189 labels with ≥ 15 songs, the leading
SVD direction of each label's own residuals; the null reassigns the real residuals to
random labels (the two-covariance model's assumption), 20 draws:

| | observed | null (mean ± sd) |
|---|---|---|
| median \|cos\| between labels' leading directions | 0.271 | 0.456 ± 0.010 |
| share of a label's residual variance in the pooled top-10 directions | 0.193 | 0.192 ± 0.0005 |
| share in the label's own top-10 directions | 0.553 | 0.514 ± 0.0006 |

The directions along which a rapper's songs vary are the rapper's own; the pooled
covariance explains a real label's scatter no better than a random one's.

## PLDA scoring against whitened cosine (`plda_scoring.json`)

| scorer | MRR | R@1 | R@10 |
|---|---|---|---|
| cosine (reference) | 0.2997 | 0.2019 | 0.4911 |
| within-label whitening, cosine | **0.4164** | 0.3179 | 0.6026 |
| PLDA, simultaneously diagonalised | 0.3926 | 0.2974 | 0.5747 |
| PLDA, both covariances diagonal | 0.2961 | 0.1994 | 0.4889 |
| PLDA with a label-specific within scale | 0.2817 | 0.2061 | 0.4217 |

PLDA − cosine +0.095; PLDA − whitened cosine −0.021 [−0.026, −0.016]; diagonal − PLDA
−0.098; label-scale − PLDA −0.110. The between-label signal lives in about 166 of 1,024
whitened dimensions. The model's own scorer does not reach the heuristic, the independence
assumption alone costs the whole gain, and a per-label scale makes it worse: labels differ
in the directions of their scatter, not its size (measured above).

## Does 'not names' depend on NER recall? (`candidate_union_neutralisation.json`)

| stripped from the word space | surfaces | share of text | MRR | vs words |
|---|---|---|---|---|
| nothing | — | — | 0.4963 | — |
| the reviewed catalogue | 605 | 2.0% | 0.4881 | −0.008 [−0.011, −0.006] |
| + every proposed named surface in ≤ 2% of songs (could be names) | 5,419 | 5.4% | 0.4743 | −0.022 [−0.026, −0.019] |
| only proposed named surfaces in > 2% of songs (ordinary words) | 514 | 14.2% | 0.4766 | −0.020 |
| every proposed named surface | 5,915 | 16.8% | 0.4603 | −0.036 |
| + every proposed rap-culture term | 8,132 | 17.4% | 0.4583 | −0.038; terms alone −0.002 [−0.004, −0.000] |

Removing every name-shaped surface the NER ever considered — nine times the catalogue —
costs two points of fifty; what neither NER arm ever saw lives in the rare quartile, which
carries nothing. The finding does not rest on NER recall.

## One language model per label (`ngram_language_model_attribution.json`)

Interpolated character n-gram models per label (orders 1–5, fixed weights, add-0.5 unigram
floor, 1.1 M five-grams), the query scored by mean log-probability per character with its
leakage group subtracted: orders 1–3 0.3773, orders 1–5 0.3928 (R@1 0.297, R@10 0.575),
against 0.4266 for character TF-IDF (−0.033 [−0.041, −0.025]) and 0.4963 for words
(−0.104). The generative model class does not close the gap; the word space's lead is not
an artefact of TF-IDF weighting.

## A purpose-built style embedding as a fourth space (`style_embedding_space.json`)

mStyleDistance (xlm-roberta-base trained on GPT-4 sentence pairs that keep content and
change one of forty style features, Simplified Chinese among its nine languages; pinned
release `d66ed25e`; never trained on lyrics):

| space | MRR | R@1 | R@10 |
|---|---|---|---|
| style, cosine | 0.0536 | 0.0150 | 0.1119 |
| style, within-label whitening | 0.2073 | 0.1248 | 0.3720 |
| semantic, within-label whitening | 0.4164 | 0.3179 | 0.6026 |
| words | 0.4963 | 0.4029 | 0.6715 |
| whitened style + words | 0.4120 | | |
| whitened semantic + words | 0.5371 | | |
| whitened style + whitened semantic + words | 0.5178 | | |

Whitened style − whitened semantic −0.208 [−0.219, −0.197]; adding the style space to the
semantic + words fusion −0.022 [−0.028, −0.016]. The model covers Chinese, so language is
not the reason; its own authorship-verification scores are modest by design (ROC-AUC
0.60–0.73 on PAN languages), and its forty features are generic sentence-level dimensions.
Within those limits: the generic style dimensions such embeddings encode are not what
identifies a rapper.

## Regional variety inside the common-word signal (`dialect_marker_identity.json`)

A hand-compiled catalogue of function words, particles and pronouns for seven varieties
(published in the JSON) marks 0.55% of word tokens; 28 of 226 labels lean to a variety by
the rule in the file (16 Southwestern Mandarin, 6 Cantonese, 3 Northeastern, 3 Beijing).
Removing every marker token from the word space: 0.4957 vs 0.4963 (−0.001 [−0.002,
+0.001]); markers alone undefined. Among the word space's wrong top-1 answers whose true
label leans to a variety, the wrong label leans to the same variety 26.1% of the time
against 4.9% if wrong labels were random (×5.4); with every marker token removed, 24.7%
against 4.8% (×5.1). Per label, leaving any one out keeps the enrichment between 4.8× and
5.3×, but one variety carries it: the sixteen Southwestern Mandarin labels are confused with
each other 27–71% of the time in most cases, while the Cantonese, Northeastern and Beijing
labels (3–6 each) sit near zero. The regional variety is carried by the broad distribution
of ordinary words, not by a shortlist of dialect words; part of the common-word identity is
a shared language variety, established for the largest regional scene in the corpus and
open for the others. The lean is derived from the text alone, never from biography.

## Identity among content rivals (`content_rival_test.json`)

For each query the frozen semantic space names the K labels closest to it in content; every
space ranks the true label within that set. The frozen space's own row is a function of its
full rank and is the floor, not a test.

| space | all 226 labels | among 5 rivals (chance 0.408) | among 10 (0.275) | among 25 (0.148) |
|---|---|---|---|---|
| frozen semantic (defines the rivals) | 0.2997 | 0.3716 | 0.3296 | 0.3069 |
| semantic, within-label whitening | 0.4164 | 0.5897 | 0.5305 | 0.4755 |
| character 2–5-grams | 0.4266 | 0.6308 | 0.5593 | 0.4898 |
| jieba words | **0.4963** | **0.6765** | **0.6142** | 0.5530 |
| whitened semantic + words (fusion) | 0.5371 | 0.6571 | 0.6100 | **0.5677** |

Among 5 rivals: words − whitened semantic +0.095 [+0.086, +0.104]; words − characters
+0.048; fusion − words −0.023 [−0.030, −0.016]; among 25 rivals fusion − words +0.011
[+0.003, +0.018]. With subject held roughly constant the word space still names the label
first more than half the time; the fusion that wins over all labels loses to words alone
when every candidate shares the subject and wins again as candidates diversify: part of
what the semantic component contributes to identity is content.

## Word choice against a language model's expectation (`lexical_choice_surprisal.json`)

A Chinese masked language model trained on standard written Mandarin and never on this
corpus (`hfl/chinese-roberta-wwm-ext`, pinned) scored how expected every word the rapper
used was, with the whole word masked and its lyric line as context: 3,670,476 scored word
instances, median 4.17 nats. Each arm rebuilds the word space from a subset of those
instances and scores it under the unchanged protocol. Arms use unigram word TF-IDF, because
an arm keeps word instances without their neighbours. A first, uncommitted pass used the
protocol's unigram-plus-bigram vectorizer, whose bigrams inside a band joined words that
were never adjacent; it was replaced before any number here was written. On all scored
words the unigram space gives 0.5126, above the protocol's unigram-plus-bigram 0.4963. That
is a sensitivity observation; the headline system is not re-chosen on it.

**Global bands.** Quartiles of surprisal, equal token mass each:

| band | nats | MRR | word types | median corpus frequency of its tokens | catalogue share |
|---|---|---|---|---|---|
| 1, most expected | 0.00–1.30 | 0.2007 | 11,080 | 12,981 | 0.2% |
| 2 | 1.30–4.17 | 0.2852 | 29,610 | 3,071 | 0.5% |
| 3 | 4.17–6.85 | 0.3232 | 63,975 | 415 | 1.5% |
| 4, least expected | 6.85–29.1 | 0.4541 | 90,396 | 129 | 2.6% |

Expected half 0.3182, unexpected half 0.5025: −0.184 [−0.193, −0.174]. Read alone this says
identity lives in surprising choices, but a band mixes two things: which words sit in it
(rare, long, Latin-script and name-like words are surprising wherever they occur) and how
each word is used. Four controls pull them apart.

| control | lower half | upper half | lower − upper | random halves of the same blocks |
|---|---|---|---|---|
| global halves | 0.3182 | 0.5025 | −0.184 [−0.193, −0.174] | — |
| catalogue surfaces removed | 0.3171 | 0.4942 | −0.176 [−0.186, −0.167] | — |
| within each word | 0.3767 | 0.4440 | −0.065 [−0.073, −0.057] | +0.003 [−0.005, +0.011] |
| within each word and line-context decile | 0.3731 | 0.4200 | −0.047 [−0.055, −0.039] | −0.002 [−0.010, +0.006] |
| the same with 20 strata | 0.3668 | 0.4150 | −0.048 [−0.056, −0.039] | +0.005 [−0.003, +0.013] |
| within each word, line-context decile and line position | 0.3670 | 0.4156 | −0.048 [−0.056, −0.041] | +0.000 [−0.008, +0.008] |

- **Names do not carry it.** Removing the 605 reviewed catalogue surfaces moves the halves by
  less than 0.01.
- **Most of the global gap is vocabulary.** Split within each word, so both halves hold the
  same 62,413 words at the same frequencies, the gap falls from 0.184 to 0.065. The within-word
  quartiles still rise in order: 0.2578, 0.2981, 0.3130, 0.3415 (fourth − first +0.083
  [+0.074, +0.092]).
- **Part of the within-word gap is the line, not the word.** A word's surprising uses sat in
  more unusual lines (mean surprisal of the other words of the line 4.88 against 3.93 nats).
  Splitting inside deciles of that line context balances it (4.41 against 4.36) and leaves
  −0.047; twenty strata give the same.
- **Not the rhyme slot.** After the line-context split, the surprising uses still sat at a
  line's end more often (14.1% against 11.3%), where the model has no context to the right.
  Splitting inside line position as well balances that (11.3% both) and leaves −0.048.
- **Not document length.** Both stratified halves give a median song 202–203 tokens, and 30
  and 31 songs have fewer than 20.
- **Not a few rappers, and not dialect** (`who_carries_the_within_word_gap`, strictest split,
  upper minus lower). Of the 174 labels with at least 20 queries, 74% gain, median +0.049.
  Leaving any one label out keeps the gap between +0.047 and +0.049. By the variety a
  label's lyrics lean to (the rule of the dialect experiment, which reproduces its published
  counts): without the 16 Southwestern Mandarin labels +0.048 [+0.040, +0.056]; the 198
  labels leaning to no variety alone +0.046 [+0.038, +0.055].

Reading, at the tier it supports: the same word, at the same frequency and in an equally
unusual line, identifies the rapper better in the uses a standard-Mandarin model expects
least. That is a within-word effect of about 0.05 MRR between halves, spread across most
rappers, on top of a larger effect of which words a rapper uses. What it is not yet: a claim about intent or
creativity. The model's expectation is standard written Mandarin's, so "unexpected" means
unexpected for that register, and the context is one line.

## Identity at every layer of BGE-M3 (`layerwise_identity_probe.json`)

Every semantic number above reads one vector: the final layer's [CLS] state, which BGE-M3
was trained to make useful for retrieval by meaning. Probing work on BERT-style encoders
puts surface and lexical information low in the stack and more abstract information higher
(Jawahar, Sagot and Seddah 2019; Tenney, Das and Pavlick 2019). So the script was written
with a prediction: if identity here is word use, it should be most readable low in the
stack and fade toward the retrieval head. The pinned checkpoint was run once over every
chunk with all 25 hidden states, keeping each layer's [CLS] state and its attention-masked
token mean, and every layer was scored under the unchanged protocol, raw and after
within-author whitening cross-fitted over the five folds. Two checks come first. The final
[CLS], normalised, matches the recorded embedding run chunk by chunk (minimum cosine 0.9996,
none below 0.999). Scored, it reproduces the recorded MRRs (0.2998 against 0.2997, and
0.4164 against 0.4164).

| layer | [CLS] cosine | [CLS] whitened | mean-pooled cosine | mean-pooled whitened | mean cosine of random chunk pairs, [CLS] / mean |
|---|---|---|---|---|---|
| 0 | 0.0244 | 0.0244 | 0.1256 | 0.3621 | 1.000 / 0.694 |
| 4 | 0.1242 | 0.3478 | 0.1493 | 0.3795 | 0.991 / 0.930 |
| 8 | 0.1003 | 0.3354 | 0.1267 | 0.3883 | 0.986 / 0.945 |
| 12 | 0.0949 | 0.3454 | 0.1130 | 0.3806 | 0.982 / 0.907 |
| 16 | 0.0747 | 0.3370 | 0.1031 | 0.3427 | 0.982 / 0.859 |
| 18 | 0.0676 | 0.2711 | 0.0894 | 0.3313 | 0.969 / 0.838 |
| 20 | 0.1522 | 0.2904 | 0.0804 | 0.3331 | 0.977 / 0.846 |
| 22 | 0.2459 | 0.3741 | 0.0832 | 0.3624 | 0.876 / 0.892 |
| 23 | 0.2799 | 0.4061 | 0.1268 | 0.3643 | 0.706 / 0.905 |
| 24 | **0.2998** | **0.4164** | 0.2075 | 0.4013 | 0.496 / 0.719 |

Every other layer and pooling is below the final [CLS] under the same transform, each
interval excluding zero. The closest is the final layer's token mean after whitening,
−0.016 [−0.023, −0.010]. The best lower layer is the token mean at layer 8, −0.028
[−0.037, −0.018]. Both poolings dip between layers 16 and 20.

The prediction failed. Read by a linear map, BGE-M3 holds the most label identity at the
output it was trained to produce, not below it. Layer 0's token mean is a dense bag of
subword embeddings and whitens to 0.362, while a 1,024-dimensional SVD of the word TF-IDF
space whitens to 0.548. So the distance between the semantic encoder and the word space
is not a matter of which layer is read, and not of vector width. What the final layer does
is hide what it holds: its raw cosine is 0.30, and whitening lifts it to 0.42.

Limits: one encoder, a linear readout, and chunk vectors averaged into songs. The hidden
states are stored in float16, which limits the nearly constant [CLS] states of layers 0–2
and not the token means. The [CLS] state of a lower layer is not yet a summary of the
chunk, so for "low in the stack" the token mean is the fair reading.

## Contrastive fine-tuning, fold 0 (`identity_encoder_fold0*.json`, `identity_encoder_analysis_fold0*.json`)

**Baseline** (`identity_encoder_fold0.json`): LoRA on BGE-M3, eight anchors a step, a positive
from another leakage group, two content-controlled hard negatives per anchor, label strings
masked, lr 2e-4, two epochs (3,828 steps). Scored on the test fold (1,187 queries) and on the
34 labels never trained on (1,041 queries); whitening is fitted on the training folds.

| system | test fold | unseen labels |
|---|---|---|
| frozen BGE-M3, masked text | 0.2947 | 0.2419 |
| fine-tuned | 0.2565 | 0.1777 |
| frozen + within-author whitening | 0.4013 | 0.3361 |
| fine-tuned + within-author whitening | 0.4133 | 0.3365 |
| words | 0.5145 | 0.4233 |
| fine-tuned fused with words | 0.5198 | 0.4112 |

| contrast | test fold | unseen labels |
|---|---|---|
| fine-tuned − frozen | −0.041 [−0.061, −0.020] | −0.065 [−0.085, −0.045] |
| whitened fine-tuned − whitened frozen | +0.014 [−0.010, +0.036] | +0.002 [−0.019, +0.025] |
| fine-tuned fused with words − words | +0.006 [−0.005, +0.018] | −0.014 [−0.026, −0.001] |

The chunk vectors show what training did. The frozen space has a mean pairwise cosine of
0.495 and a participation ratio of 132; the tuned space 0.800 and 26, and a chunk's median
cosine to its own frozen vector is 0.62. Training moved the space a long way and narrowed
it, and after whitening it holds what the whitened frozen space already held.

Reading, and its limit: with eight anchors a step, contrastive fine-tuning added nothing
detectable beyond a within-author whitening of the frozen encoder, and on unseen labels it
made the word space slightly worse. That is a result about this configuration, not proof
that fine-tuning cannot help; batch size is the obvious untested variable (below). The 1.2.0
run's whitened lead (0.4189 vs 0.3839) does not replicate on 1.3.0. Fold-level numbers are
not paired across builds: a group's fold is drawn over the sorted group list, seven groups
left the corpus, and the 1.2.0 test fold had 1,205 queries.

**Momentum queue** (`identity_encoder_fold0_moco.json`): a 4,096-vector queue filled by an
exponential moving average of the LoRA weights (m 0.999), lr 5e-5. Test fold 0.1331, unseen
labels 0.0882; whitened 0.3091 and 0.2499, 0.090 and 0.081 below the whitened frozen space.
Its vectors have a mean pairwise cosine of 0.941 and a participation ratio of 7.0, and a
chunk's median cosine to its frozen vector is 0.20: a dimensional collapse. The training
guard missed it because it read the mean cosine of 32 vectors in train mode, which stayed
near 0.8. Review of the code then found the design flawed. MoCo compares each query with
keys that all come from the momentum encoder; here the positives came from the live encoder
and only the queue from the momentum encoder, so the loss could fall by moving live vectors
away from the queue. The learning rate also differed from the baseline by a factor of four.
The run therefore says nothing about whether more negatives help, and is kept only so the
failure stays on record, beside the earlier live-encoder queue that collapsed to one point
(`identity_encoder_fold0_queue4096.json`, 1.2.0).

**Next, with its reading fixed before it runs.** Both queue designs are replaced by an exact
64-anchor batch (GradCache, Gao et al. 2021): every vector in the denominator comes from the
current encoder and the gradient is computed in chunks. Before launch the chunked gradient
was checked against the direct one on a 32-sequence batch in chunks of 12
(`identity_encoder_grad_cache_check_gradcache.json`): with dropout off the two losses agree
to six decimals and the gradients have cosine 0.99996 (relative difference 0.9%); with
dropout on, the second pass reproduces the first pass's vectors exactly. The run uses 64
anchors a step in chunks of 32, lr 2e-4 and two epochs as before (so about 480 steps), and
the trainer now records an eval-mode probe's participation ratio every 50 steps. The
decision rule: the fine-tuning line shows learned identity beyond linear whitening only if
the whitened fine-tuned space beats the whitened frozen space on the unseen labels with an
interval excluding zero. Otherwise the conclusion is that, within this compute budget,
contrastive adaptation adds nothing detectable beyond a linear whitening of the frozen space.

## Training-data audit for the identity encoder (`training_data_audit.json`)

Tokens per chunk median 76, p90 603, 3,669 over 512; all 226 labels have at least two
leakage groups; 592 songs (8.2%) carry a collaboration title; 1,610 chunks (6.6%) name
their own label; a chunk's nearest content rival (another label) is closer than its nearest
same-label chunk in 91.0% of cases (median cosine 0.725 vs 0.663).

## Privacy

Aggregate numbers only. Per-query tables, vectors and text stay under the private root.
