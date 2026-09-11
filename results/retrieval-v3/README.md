# Every headline experiment, rerun on corpus v3 (build 1.3.0)

Corpus v3 is corpus v2 with credit blocks, copied titles, track lists, HTML remnants,
production-credit residue and two television-episode transcripts removed from the lyric
text (`results/cleaned-corpus-v3/`): 24,237 chunks, 7,379 songs, content digest
`cd51bf69…`. The semantic vectors are a local BGE-M3 run over the v3 text (fp16, pinned
revision `5617a9f6`, weights `b5e0ce34`; contract in the private embedding run). Every
file in this directory was produced on 2026-09-10 by the same scripts as
`results/retrieval-v2/`, run with `CHINESE_RAP_CORPUS=v3`, so the protocol is unchanged:
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

## Contrastive fine-tuning (1.2.0 numbers; 1.3.0 reruns in progress)

Fold 0 on build 1.2.0 (`identity_encoder_fold0.json`, `identity_encoder_analysis_fold0.json`):
LoRA fine-tuning of BGE-M3 with stanza positives from another leakage group, two
content-controlled hard negatives per anchor, labels masked, two epochs, scored 0.2698 on
the test fold against 0.2871 for the frozen model on the same masked text and 0.1952
against 0.2480 on the 34 labels never trained on; only after within-label whitening did
the tuned space lead (0.4189 vs 0.3839, a bound since the whitening is fitted on songs the
model saw), and it added nothing to the word space (0.5022 vs 0.4999). The vectors show
why: the fine-tune contracted the space (mean pairwise cosine 0.50 → 0.80, effective rank
265 → 138).

The same run with a 4,096-embedding cross-batch queue (`identity_encoder_fold0_queue4096.json`)
collapsed to a single point — mean pairwise cosine 1.000, effective rank 4, MRR 0.0212 at
chance, loss at ln K — the known failure of a stale queue without a momentum encoder. The
trainer now fills the queue from an EMA of the LoRA weights (MoCo), guards on the batch's
mean cosine, and records both. The fold-0 baseline and a MoCo run (lr 5e-5) are being
rerun on 1.3.0; this section is replaced when they finish.

## Training-data audit for the identity encoder (`training_data_audit.json`)

Tokens per chunk median 76, p90 603, 3,669 over 512; all 226 labels have at least two
leakage groups; 592 songs (8.2%) carry a collaboration title; 1,610 chunks (6.6%) name
their own label; a chunk's nearest content rival (another label) is closer than its nearest
same-label chunk in 91.0% of cases (median cosine 0.725 vs 0.663).

## Privacy

Aggregate numbers only. Per-query tables, vectors and text stay under the private root.
