# Every headline experiment, rerun on corpus v3 (build 1.3.0)

Corpus v3 is corpus v2 with credit blocks, copied titles, track lists, HTML remnants,
production-credit residue and two television-episode transcripts removed from the lyric
text (`results/cleaned-corpus-v3/`): 24,237 chunks, 7,379 songs, content digest
`cd51bf69…`. The semantic vectors are a local BGE-M3 run over the v3 text (fp16, pinned
revision `5617a9f6`, weights `b5e0ce34`; contract in the private embedding run). Every
CPU experiment was produced on 2026-09-10 by the same scripts as `results/retrieval-v2/`,
run with `CHINESE_RAP_CORPUS=v3`; the GPU experiments (surprisal, layer-wise probe,
fine-tuning) followed between 2026-09-11 and 2026-09-14. The protocol is unchanged throughout:
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

Reading. The ordering words > characters > semantic > rhyme form is unchanged. (Qualified 2026-09-21: words lead
characters under this prototype scorer only; under a linear classifier trained on the training folds characters lead,
0.541 against 0.482. See "A discriminative classifier against the prototype" and the label-size section.) Every
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

**Six held-out draws** (`heldout_label_redraw.json`, `src/heldout_label_redraw_v3.py`, 2026-09-20).
The transfer reading above rests on one draw of 34 labels, fixed by seed and shared by every
fold. Five further draws were taken as disjoint blocks of 34 labels each (seeds 20260918 to
20260922; 22 labels are never held out), each transform fitted on the seen labels' songs only
as the probe does, and the held-out labels' queries scored against all 226 profiles. Draw 0
reproduces the probe's numbers (gaps at most 0.00004). Rule, fixed before the run: the
label-free part transfers if total whitening minus none is positive with the interval clear
of zero in every draw; the label-specific part does not transfer if within-author minus total
whitening is not positive in at least five of six draws.

| draw (seed) | queries | none | total | within-author | seen, within-author | total − none | within − total | seen − unseen |
|---|---|---|---|---|---|---|---|---|
| 0 (20260825) | 1,041 | 0.2658 | 0.4072 | 0.3813 | 0.3814 | +0.138 [+0.114, +0.160] | -0.026 [-0.034, -0.018] | +0.002 [-0.005, +0.009] |
| 1 (20260918) | 932 | 0.2724 | 0.4614 | 0.4433 | 0.4323 | +0.187 [+0.163, +0.213] | -0.018 [-0.025, -0.010] | -0.011 [-0.018, -0.004] |
| 2 (20260919) | 1,211 | 0.3390 | 0.4679 | 0.4484 | 0.4499 | +0.124 [+0.103, +0.145] | -0.019 [-0.027, -0.013] | +0.001 [-0.006, +0.009] |
| 3 (20260920) | 1,156 | 0.3505 | 0.4719 | 0.4545 | 0.4446 | +0.114 [+0.093, +0.137] | -0.018 [-0.026, -0.010] | -0.009 [-0.016, -0.001] |
| 4 (20260921) | 1,158 | 0.2631 | 0.4202 | 0.4092 | 0.4016 | +0.151 [+0.130, +0.174] | -0.011 [-0.018, -0.004] | -0.008 [-0.016, +0.000] |
| 5 (20260922) | 983 | 0.2825 | 0.4312 | 0.4111 | 0.4042 | +0.149 [+0.124, +0.173] | -0.021 [-0.029, -0.012] | -0.008 [-0.016, +0.001] |

Spread over the six draws: unseen none 0.263–0.350,
total whitening 0.407–0.472,
within-author 0.381–0.455;
total − none +0.114 to +0.187
(sd 0.026); within − total
-0.026 to
-0.011 (sd
0.005). Readings:
the label-free part transfers (6 of 6 draws clear of zero) and the
label-specific part does not (6 of 6 negative and clear of zero). On unseen labels
the label-fitted whitening is below the label-free one in every draw, by 0.011 to 0.026. The
seen-label space is not reliably ahead of the unseen one on the same queries: the contrast is
clear of zero in 2 of 6 draws, both times negative. The single-draw levels vary
(none 0.263 to 0.351 across draws) but the two readings do not.

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

## Whole song against the stanza mean (`whole_song_embedding.json`)

Every semantic number represents a song as the mean of its stanza vectors. The stanza is the
source file's own unit: 3,649 of the 7,379 song records are a single stanza, the rest a
median of three. The alternative not tried before is to embed the song once as one text:
stanzas joined by newlines in source order, the same pinned BGE-M3 through FlagEmbedding at
fp16, `max_length` 8,192 (median song 574 tokens, two songs over the limit). Two checks come
first. A one-stanza song is the same text as its stanza, and its whole-song vector matches the
recorded chunk vector for all 3,649 (minimum cosine 0.99994). The stanza mean reproduces the
recorded 0.2997 and 0.4164 exactly. The two representations of a song agree at a median
cosine of 0.962 (10th percentile 0.867).

| | stanza mean | whole song | whole − mean |
|---|---|---|---|
| cosine | 0.2997 | 0.3218 | +0.022 [+0.016, +0.028] |
| total whitening | 0.3996 | 0.3943 | −0.006 [−0.012, +0.001] |
| within-author whitening | 0.4164 | 0.4122 | −0.005 [−0.011, +0.002] |
| fused with words, raw | 0.4894 | 0.4901 | +0.000 [−0.004, +0.004] |
| fused with words, within-author whitened | 0.5371 | 0.5323 | −0.004 [−0.009, +0.001] |

By the rule written before the run (switch only if the whole song beats the stanza mean after
within-author whitening with an interval clear of zero, and is not worse raw) the stanza mean
stands. By number of stanzas, which is where any difference must come from:

| stanzas | queries | cosine: mean → whole | whole − mean | whitened: mean → whole | whole − mean |
|---|---|---|---|---|---|
| 1 | 3,524 | 0.3035 → 0.3134 | +0.011 [+0.005, +0.017] | 0.3952 → 0.4043 | +0.008 [+0.001, +0.014] |
| 2 | 1,391 | 0.3291 → 0.3473 | +0.015 [−0.002, +0.032] | 0.4656 → 0.4575 | −0.009 [−0.026, +0.009] |
| 3–5 | 1,010 | 0.2979 → 0.3179 | +0.019 [+0.001, +0.039] | 0.4447 → 0.4186 | −0.025 [−0.046, −0.004] |
| 6 or more | 1,295 | 0.2594 → 0.3203 | +0.061 [+0.042, +0.080] | 0.3991 → 0.3802 | −0.017 [−0.038, +0.001] |
| 2 or more | 3,696 | 0.2962 → 0.3298 | +0.032 [+0.021, +0.043] | 0.4366 → 0.4198 | −0.016 [−0.027, −0.006] |

One-stanza queries move only because the label profiles and the whitening are built from
every song. For songs of two or more stanzas the two readings point opposite ways. Read by raw
cosine, averaging loses identity, the more stanzas the more (+0.061 for the whole song at six
or more): a mean of many unit vectors in an anisotropic space is pulled toward the shared
direction. Read after within-author whitening, the reading every semantic conclusion here
rests on, the stanza mean holds more identity than the whole-song embedding (−0.016 [−0.027,
−0.006]): the spread of stanza vectors that the mean keeps is what whitening uses, and one pass
over the whole song does not recover it. So averaging stanzas does not throw away identity the
encoder could otherwise give; it only hides part of it from raw cosine, which whitening
already corrects. The raw semantic number is conservative by 0.022, and the word space stays
ahead of either representation (whole song whitened − words −0.093 [−0.103, −0.083]). What this
does not measure is meaning for its own sake: the task is identity, and the two representations
agree closely enough (median cosine 0.962) that they are not different readings of the songs.

## A label as a centre, and the scorer's dependence on label size (`exemplar_vs_prototype.json`, `label_size_calibration.json`)

Every published number scores a held-out song against one profile per label, the normalised
weighted sum of the label's other songs: a prototype. Two questions about that choice were
asked on 2026-09-18, each with its reading rule fixed before the run.

**Exemplar scoring** (`src/exemplar_vs_prototype_v3.py`). Exemplar theory keeps every
instance: a query belongs with the label holding the song nearest to it. Four exemplar scorers
were run in four spaces, leave-group-out throughout: the nearest remaining song; the mean of
the three best near-duplicate components (a component is a (group, label) set scored by its
best member); the protocol-weighted mean cosine over the remaining songs; and the nearest song
among at most four components in a fixed seeded order, which removes the extra chances a large
label gets under a maximum. The prototype reproduces its published number in every space
(gaps 0.000) and wins everywhere:

| space | prototype | nearest song | top-3 components | mean cosine | nearest of at most 4 components |
|---|---|---|---|---|---|
| semantic, raw | 0.2997 | 0.1547 | 0.1618 | 0.1046 | 0.0825 |
| semantic, within-author whitened | 0.4164 | 0.3394 | 0.4048 | 0.3224 | 0.1846 |
| words, raw TF-IDF | 0.4963 | 0.3573 | 0.4404 | 0.3450 | 0.2247 |
| word SVD-1024, whitened | 0.5426 | 0.3408 | 0.4250 | 0.4218 | 0.2268 |

Nearest song minus prototype: −0.142 [−0.151, −0.133], −0.075 [−0.083, −0.067], −0.143
[−0.152, −0.134] and −0.207 [−0.216, −0.199]. By the rule (both the nearest song and the
size-matched nearest song must beat the prototype with intervals clear of zero) the prototype
stands in all four spaces. A label is closer to one centre than to a set of separable styles (but see the attention family
below, where a query-weighted mixture beats the centre); in the raw semantic space the nearest song is a topic neighbour. After
whitening the three best components come within 0.01 of the prototype (−0.010 [−0.017,
−0.003]): whitening is what makes the centre findable from few instances.

### Between a centre and a cloud: attention over a label's songs (`attention_exemplar.json`, `src/attention_exemplar_v3.py`)

The fixed exemplar rules above lost to the prototype; they are the ends of a family. Here a
query attends to each remaining song of a label with weight w_s·exp(β·cos(q, x_s)) (w_s the
protocol's component weight) and the label is scored by cosine with the attended profile. β = 0
is exactly the published prototype (checked against the protocol's scorer, gap 9e-16 dense,
4e-7 sparse, and reproducing every published MRR); as β grows the profile collapses onto the
query's nearest song of the label. β for each test fold is the grid value with the best MRR on
the other four folds' queries. Rule, fixed before the run: attention improves on the prototype
in a space if the fold-selected system minus the prototype is positive with the interval clear
of zero. The vectorised scorer equals a brute-force loop (gap ≤ 6e-16).

| space | β 0 (prototype) | 1 | 2 | 5 | 10 | 20 | 50 | 100 | β chosen | selected | selected − prototype |
|---|---|---|---|---|---|---|---|---|---|---|---|
| semantic, raw | 0.2997 | 0.3061 | 0.3138 | 0.3351 | 0.3557 | 0.3569 | 0.3095 | 0.2331 | 20 | 0.3569 | +0.058 [+0.051, +0.065] |
| semantic, within-label whitened | 0.4164 | 0.4289 | 0.4355 | 0.4496 | 0.4484 | 0.4357 | 0.3940 | 0.3709 | 5, 10 | 0.4481 | +0.034 [+0.029, +0.038] |
| jieba words | 0.4963 | 0.4992 | 0.5011 | 0.5047 | 0.5093 | 0.5139 | 0.4952 | 0.3611 | 20 | 0.5139 | +0.017 [+0.013, +0.021] |
| character 2–5-grams | 0.4266 | 0.4302 | 0.4333 | 0.4404 | 0.4436 | 0.4391 | 0.3997 | 0.3061 | 10 | 0.4436 | +0.017 [+0.013, +0.020] |
| word SVD-1024, whitened | 0.5426 | 0.5457 | 0.5456 | 0.5414 | 0.5209 | 0.5004 | 0.4431 | 0.3968 | 1, 2 | 0.5450 | +0.002 [+0.000, +0.004] |

Reading: attention improves on the prototype in all five spaces. In every space MRR rises from
β = 0 to an interior maximum and falls toward the nearest-song end, and the five folds choose
nearly the same β, so the optimum is a property of the space, not of a fold. A credited label is
identified best neither as one average nor as its nearest song, but as a mixture weighted toward
the songs that resemble the query. The gain is largest in the raw semantic space (+0.058),
where attention to a label's songs close in content to the query partly substitutes for
whitening; after whitening it is +0.034; in the surface spaces +0.017; in the whitened word SVD
+0.002. This is the one-parameter, unlearned form of attention pooling over a set; a learned
aggregator is the natural next step.

### Each label's own scatter: label-specific covariance (`label_specific_covariance.json`, `src/label_specific_covariance_v3.py`)

PLDA and the two-covariance model assume one within-label covariance for every label and lost to
the cosine prototype; the whitening-directions analysis showed labels scatter along their own
directions. A hierarchical model lets each label keep its own covariance, shrunk toward the
pooled one because a label has few songs (regularised discriminant analysis): Σ_l = λσ²I +
(1 − λ)S_l in the within-label whitened spaces, σ² the pooled within-label variance of the
training folds, μ_l and S_l the weighted mean and covariance of the label's songs outside the
query's leakage group, and the label scored by the Gaussian log-likelihood of the query. λ = 1
is one shared isotropic model (Euclidean distance to the label mean). λ is chosen per fold on
the other folds' queries. The computation is exact through the n_l × n_l Gram matrix; it equals
the explicit 1,024 × 1,024 computation (relative gap 4e-14), and the cosine prototype reproduces 0.4164 and
0.5426. Rule, fixed before the run: label-specific covariance helps if the selected λ minus
λ = 1 is positive with the interval clear of zero.

| space | cosine prototype | λ 1 (shared) | 0.9 | 0.7 | 0.5 | 0.3 | 0.1 | λ chosen | selected − shared | selected − cosine prototype |
|---|---|---|---|---|---|---|---|---|---|---|
| semantic, within-label whitened | 0.4164 | 0.4028 | 0.4272 | 0.4238 | 0.4192 | 0.4116 | 0.3970 | 0.9 | +0.025 [+0.019, +0.031] | +0.015 [+0.009, +0.021] |
| word SVD-1024, whitened | 0.5426 | 0.5429 | 0.5187 | 0.5084 | 0.5036 | 0.4973 | 0.4863 | 1 | +0.000 [+0.000, +0.000] | -0.000 [-0.004, +0.004] |

Readings: in the whitened semantic space label-specific covariance helps, and with λ = 0.9 in
every fold it is the first generative model in this study to beat the cosine prototype (PLDA
lost by 0.021). In the whitened word space a shared covariance is enough: λ = 1 in every fold,
level with the cosine prototype. In the semantic space each label's own directions of variation
carry identity beyond its mean; in the word space, after whitening, they do not.

**Label size** (`src/label_size_calibration_v3.py`). The exemplar run's breakdown by label size
showed that the prototype's word-over-semantic lead is not uniform. Under the protocol's
scorer, by songs per label (paired group bootstrap within the band):

| songs per label (labels, queries) | raw words | raw semantic | words − semantic | whitened word SVD | whitened semantic | words − semantic |
|---|---|---|---|---|---|---|
| 5–9 (19, 149) | 0.096 | 0.160 | −0.064 [−0.116, −0.014] | 0.325 | 0.340 | −0.015 [−0.074, +0.043] |
| 10–19 (33, 477) | 0.274 | 0.281 | −0.007 [−0.042, +0.029] | 0.453 | 0.415 | +0.038 [+0.005, +0.071] |
| 20–49 (171, 6,356) | 0.529 | 0.301 | +0.228 [+0.217, +0.238] | 0.557 | 0.414 | +0.143 [+0.132, +0.154] |
| 50 or more (3, 238) | 0.440 | 0.402 | +0.038 [−0.084, +0.158] | 0.556 | 0.507 | +0.049 [−0.054, +0.151] |

The cause is in the scorer. The prototype divides a query's dot product by the profile's norm,
and the norm of a mean of n unit vectors carries sampling noise: ||m||² = ||μ||² + s²/n, with
s² the within-label variance of one song. On the training folds s² is 0.36 in the raw semantic
space and 1.00 in the word space, whose sparse unit rows barely overlap, so a small label's word
profile is inflated far more than its semantic profile and its scores deflated accordingly. Two
scorers from the speaker-recognition back-end remove the dependence without a tuned parameter:
Z-normalisation, which standardises each label's scores by the impostor scores of training-fold
songs of other labels, and a noise-corrected prototype that divides by
sqrt(max(||m||² − s²/n, ||m||²/4)) instead of ||m||. The floor binds for no query–label pair in
the raw semantic space, 9% of pairs in the raw word space, 6–9% in the whitened semantic space
and 2–3% in the whitened word SVD. Rule, fixed before the run: an ordering is size-robust under
a scorer if the contrast is positive with the interval clear of zero in every band with at least
ten labels and one hundred queries (the three-label band is reported but does not count).

| scorer | raw words | raw semantic | whitened word SVD | whitened semantic | raw words − semantic | whitened words − semantic |
|---|---|---|---|---|---|---|
| prototype (the protocol) | 0.4963 | 0.2997 | 0.5426 | 0.4164 | size-dependent: fails at 5–9 and 10–19 | size-dependent: fails at 5–9 |
| Z-normalised | 0.4868 | 0.2100 | 0.5577 | 0.4144 | size-robust: +0.206 [+0.135, +0.278], +0.230, +0.283 | size-dependent: +0.004 [−0.058, +0.066] at 5–9, +0.032 [−0.003, +0.066] at 10–19 |
| noise-corrected | 0.5394 | 0.2635 | 0.5291 | 0.3902 | size-dependent: +0.029 [−0.045, +0.102] at 5–9, then +0.203, +0.285 | size-robust: +0.147 [+0.069, +0.225], +0.061, +0.148 |

Under Z-normalisation the raw word space identifies labels of every size alike (0.539, 0.551,
0.476 and 0.606 across the four bands), while the raw semantic space loses 0.092 [0.086, 0.098]
overall: its 20–49-song band falls from 0.301 to 0.193 as its small bands rise from 0.160 and
0.281 to 0.333 and 0.321, so part of the raw semantic score under the prototype is a
large-label advantage (averaged profiles sit nearer the centre of an anisotropic space and
score high against everything). Whitening removes that advantage on its own: Z-normalisation
moves the whitened spaces by −0.002 and +0.008. The noise-corrected prototype raises the raw
word space by +0.034 [+0.027, +0.041] to 0.539, the sampling inflation of sparse profile norms
having held it down, and lowers every other space.

Reading. The protocol's prototype scorer disadvantages labels with few songs, most in sparse
spaces. For the 52 labels with fewer than twenty songs (626 queries) the published raw word
lead does not hold under it, and at five to nine songs the raw semantic space leads. Under a
calibrated scorer the raw word lead holds in every band (Z-normalisation) or in every band but
the smallest (noise-corrected); the whitened word lead among small labels is established under
one calibration and not the other. Both calibrations leave the whitened spaces nearly where
they were. The headline ordering is therefore a result for labels with twenty songs or more
under the protocol's own scorer, and a result for all sizes only under calibrated scoring; the
paper should carry a label-size panel and a label-macro estimand beside the query-weighted one.
Both calibrations were added after the exemplar breakdown had been seen, with their reading rule
fixed before they were run; which of the two to headline is not pre-specified, and both are
reported.

Added 2026-09-21, after the classifier baseline put characters ahead of words, and run before
it was read: the character space joins the calibration. Character MRR under the prototype,
Z-normalisation and the noise-corrected prototype: 0.4266, 0.3266 and 0.4591. Words minus
characters by songs per label:

| songs per label | prototype | Z-normalised | noise-corrected |
|---|---|---|---|
| 5–9 | -0.017 [-0.044, +0.009] | +0.204 [+0.139, +0.266] | +0.041 [-0.006, +0.090] |
| 10–19 | +0.013 [-0.011, +0.037] | +0.128 [+0.099, +0.158] | +0.033 [+0.007, +0.060] |
| 20–49 | +0.077 [+0.070, +0.085] | +0.161 [+0.153, +0.169] | +0.087 [+0.080, +0.095] |
| 50 or more (3 labels) | +0.071 [+0.008, +0.138] | +0.201 [+0.138, +0.269] | +0.095 [+0.022, +0.167] |

By the rule, words over characters is size-dependent; fails in bands ['5-9', '10-19'] under the prototype,
size-robust under Z-normalisation and size-dependent; fails in bands ['5-9'] under the noise-corrected
prototype. Every profile-based scorer, calibrated or not, puts words ahead where labels have
twenty songs or more; the trained classifier is the one scorer that puts characters ahead.

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

**Correction, 2026-09-21.** These rivals are chosen by the semantic space, which handicaps the
semantic family. Under rivals chosen by word topics or by characters the whitened semantic space
leads the word space instead; see "Content rivals under other rival definitions" below. The
table stands as computed; the reading that the word space is the stronger one among content
rivals does not.

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
- **Real, but not a free feature** (`usage_tagged_words`). On the same split, every word
  instance was also written as a token tagged with its half. Tagged by usage beats the same
  tagging by the random coin, +0.011 [+0.007, +0.015], so the tags carry information. Added
  beside the plain words, usage tags beat random tags by only +0.003 [+0.001, +0.006]. Any
  tagging thins the counts, and the plain words stay ahead of both (words 0.4915, words
  plus usage tags 0.4682).

Reading, at the tier it supports: the same word, at the same frequency and in an equally
unusual line, identifies the rapper better in the uses a standard-Mandarin model expects
least. That is a within-word effect of about 0.05 MRR between halves, spread across most
rappers, on top of a larger effect of which words a rapper uses; it says where in a word's
uses identity concentrates, and adds almost nothing once the word counts are known. What it
is not yet: a claim about intent or
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

**A second encoder** (`layerwise_identity_probe_chinese_roberta_wwm_ext.json`):
`hfl/chinese-roberta-wwm-ext`, a Chinese BERT-base trained only as a masked language model,
never for retrieval. It reads at most 512 tokens, which truncates 4,773 chunks. It has no
recorded run, so neither check applies, and its contrasts are against its own final token
mean.

| layer | [CLS] whitened | mean-pooled cosine | mean-pooled whitened |
|---|---|---|---|
| 0 | 0.0244 | 0.1291 | 0.3564 |
| 3 | 0.3407 | 0.1285 | 0.3665 |
| 6 | 0.3514 | 0.1143 | 0.3651 |
| 9 | 0.3652 | 0.1373 | 0.3637 |
| 12 | 0.3634 | 0.1957 | **0.3744** |

Its whitened curve is nearly flat. Every layer's token mean lies between 0.356 and 0.374,
the final layer is the highest, and layers 4, 5, 8 and 11 are inside its interval. So the
prediction fails for an encoder never trained for retrieval as well: identity is about
equally readable by a linear map at every depth, and across these two encoders a whitened
token mean stays between 0.33 and 0.40. What BGE-M3 adds is its final [CLS] summary, 0.416
after whitening.

**A second retrieval encoder** (`layerwise_identity_probe_gte_large_zh.json`):
`thenlper/gte-large-zh`, a BERT-large trained for Chinese retrieval (Alibaba's GTE), read
officially as the final [CLS] state, normalised; its 512-token limit truncates 4,773 chunks.
It has no recorded run, so stage 1 checks its final [CLS] against the model's own
sentence-transformers pipeline on 64 seeded chunks: minimum cosine 0.9998, none below 0.999.
Contrasts are against its final [CLS].

| layer | [CLS] cosine | [CLS] whitened | mean-pooled cosine | mean-pooled whitened |
|---|---|---|---|---|
| 0 | 0.0244 | 0.0244 | 0.0864 | 0.3383 |
| 4 | 0.1569 | 0.3180 | 0.1596 | 0.3363 |
| 8 | 0.1875 | 0.3329 | 0.1547 | 0.3422 |
| 12 | 0.1872 | 0.3303 | 0.1573 | 0.3552 |
| 14 | 0.1782 | 0.3287 | 0.1571 | **0.3597** |
| 18 | 0.1317 | 0.3307 | 0.1157 | 0.2986 |
| 22 | 0.2039 | 0.3146 | 0.1219 | 0.2880 |
| 24 | **0.2270** | 0.3117 | 0.1636 | 0.2956 |

Here the curve has the shape the prediction expected. The official reading whitens to 0.312,
and most of the other readings beat it with intervals clear of zero: the token mean of layer
14 by +0.047 [+0.037, +0.056], the [CLS] of layer 17 by +0.031 [+0.022, +0.039], layer 0's
token mean by +0.026 [+0.016, +0.036]. The token means fall from layer 14 to the top (0.360
to 0.296): in this encoder the retrieval head is where the least identity is readable.

Across the three encoders, then. Raw cosine is low everywhere, never above 0.30 at any layer
of any model against 0.496 for the words. Within-author whitening lifts every reading, by
+0.12 for BGE-M3's final [CLS] and +0.09 for gte's. Where in the stack identity is most
readable is a property of the encoder, not of encoders: BGE-M3 holds the most at its
retrieval head, gte-large-zh holds the most in its middle token means and the least at its
head, and the masked-LM-only model is flat. So the earlier sentence that an encoder holds the
most identity "at the output it was trained to produce" is true of BGE-M3 and false of gte.
BGE-M3 is the stronger identity encoder at every reading (best whitened 0.416 against
0.360), and the word space whitened at the same width stays ahead of all three (0.548).
Limits: linear readouts, song means of chunk vectors, float16 stores, and two of the three
encoders truncate 4,773 chunks at 512 tokens.

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
detectable beyond a within-author whitening of the frozen encoder, and on unseen labels its
raw space made the word space slightly worse (the whitened fusions are in the GradCache
table below). That is a result about this configuration, not proof
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
(`identity_encoder_fold0_queue4096.json`, 1.2.0). Fused with the words, its whitened space
falls 0.062 below the whitened frozen fusion on both scopes.

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

**Run-to-run variation, found while adding checkpoints.** The first GradCache launch was
stopped at step 250 of 480 and left no output, so the trainer now saves and resumes
(`--checkpoint-every`, `--resume`). Testing that on 20-step runs showed something that bears
on every fine-tuning number here: two identical, uninterrupted launches are not reproducible
on this GPU. After 20 steps their learned LoRA update (the `lora_B` matrices, which start at
zero) differed by 49% in relative norm, and Adam's first moment by 67%, with the data order
and every RNG state identical. A run stopped at step 10 and resumed differed from an
uninterrupted one by 42% and 61%, inside that spread, so resuming adds nothing beyond it. That
first test ran from scratch scripts; the check is now in the repo
(`tools/check_trainer_resume_v3.py`, report `identity_encoder_resume_check.json`), and its run
on 2026-09-14 gave 44% and 59% between two uninterrupted runs and 45% and 59% for the resumed
one, with the data order and every RNG state identical, and passed. The bootstrap intervals
above cover the sampling of queries, not the variation between training runs. One rule is therefore added before any GradCache result exists,
and it only makes the reading stricter: a pass on one launch counts only once a second launch
repeats it.

**First GradCache launch** (`identity_encoder_fold0_gradcache64.json`,
`identity_encoder_analysis_fold0_gradcache64.json`): 480 steps without interruption; the
largest change of a vector between the two GradCache passes was 0.0.

| system | 8 anchors, test fold | 8 anchors, unseen labels | 64 anchors, test fold | 64 anchors, unseen labels |
|---|---|---|---|---|
| frozen, masked text | 0.2947 | 0.2419 | 0.2947 | 0.2419 |
| fine-tuned | 0.2565 | 0.1777 | 0.2802 | 0.2066 |
| frozen + within-author whitening | 0.4013 | 0.3361 | 0.4013 | 0.3361 |
| fine-tuned + within-author whitening | 0.4133 | 0.3365 | 0.4412 | 0.3721 |
| words | 0.5145 | 0.4233 | 0.5145 | 0.4233 |
| whitened frozen fused with words | 0.5361 | 0.4546 | 0.5361 | 0.4546 |
| whitened fine-tuned fused with words | 0.5448 | 0.4577 | 0.5513 | 0.4767 |

| contrast | 8 anchors, test fold | 8 anchors, unseen labels | 64 anchors, test fold | 64 anchors, unseen labels |
|---|---|---|---|---|
| fine-tuned − frozen | −0.041 [−0.061, −0.020] | −0.065 [−0.085, −0.045] | −0.015 [−0.033, +0.004] | −0.035 [−0.052, −0.017] |
| whitened fine-tuned − whitened frozen | +0.014 [−0.010, +0.036] | +0.002 [−0.019, +0.025] | +0.041 [+0.020, +0.062] | +0.039 [+0.019, +0.060] |
| whitened fine-tuned with words − whitened frozen with words | +0.009 [−0.009, +0.027] | +0.005 [−0.013, +0.024] | +0.016 [+0.001, +0.032] | +0.024 [+0.006, +0.042] |
| whitened fine-tuned with words − words | +0.026 [+0.007, +0.044] | +0.034 [+0.014, +0.054] | +0.033 [+0.015, +0.050] | +0.052 [+0.033, +0.073] |

By the rule written before the run, this launch passes: the whitened tuned space beats the
whitened frozen space on the 34 unseen labels, +0.039 [+0.019, +0.060], and on the test fold,
+0.041. With eight anchors the same contrast was +0.002, so batch size was the variable that
mattered. The gain carries into the kind of system this project builds: fused with the
words, the whitened tuned space beats the whitened frozen space fused with the words, +0.024
[+0.006, +0.042] on unseen labels. Read raw, the tuned space is still below the frozen one
(−0.035 on unseen labels), so what it learned, like what the frozen space holds, has to be
read through the whitening. Its participation ratio is 40.6, against 26 for the 8-anchor run
and 132 for the frozen space; mean pairwise cosine 0.798; a chunk's median cosine to its
frozen vector 0.66.

**Second launch, another training seed** (`identity_encoder_fold0_gradcache64_seed2.json`,
`identity_encoder_analysis_fold0_gradcache64_seed2.json`): the same configuration with
`--train-seed 20260826`, which moves the LoRA initialisation, batch order and dropout and
leaves the folds, held-out labels and probe unchanged. The first launch used the project seed,
20260825, before the flag existed. The second was stopped once for a game and resumed from its
step-90 checkpoint, and its process was suspended by standby for eleven hours and continued;
neither changes what is computed.

| | seed 20260825 | seed 20260826 |
|---|---|---|
| fine-tuned, test fold / unseen labels | 0.2802 / 0.2066 | 0.2890 / 0.2177 |
| fine-tuned + within-author whitening | 0.4412 / 0.3721 | 0.4443 / 0.3841 |
| whitened fine-tuned fused with words | 0.5513 / 0.4767 | 0.5610 / 0.4891 |
| whitened fine-tuned − whitened frozen, unseen labels | +0.039 [+0.019, +0.060] | +0.050 [+0.031, +0.070] |
| the same, test fold | +0.041 [+0.020, +0.062] | +0.045 [+0.024, +0.066] |
| whitened fine-tuned with words − whitened frozen with words, unseen labels | +0.024 [+0.006, +0.042] | +0.037 [+0.021, +0.054] |
| whitened fine-tuned with words − words, unseen labels | +0.052 [+0.033, +0.073] | +0.066 [+0.044, +0.087] |
| participation ratio; mean pairwise cosine | 40.6; 0.798 | 47.0; 0.773 |

Both launches pass, so by both rules written before the results the reading holds. With a
64-anchor batch, contrastive fine-tuning learns label identity beyond what a within-author
whitening of the frozen encoder reveals; it transfers to labels never trained on; and fused
with the words it adds to them more than the whitened frozen space does. Linear whitening is
not the ceiling of the semantic encoder. The two launches differ by 0.011 on the main
unseen-label contrast, the first measure of run-to-run spread, well inside both intervals.
Limits: fold 0 only and two seeds. Raw cosine on the tuned space is still below the frozen
space (−0.025 on unseen labels for the second seed), so the learned identity, like the frozen
space's, is read through the whitening, and the word space alone stays ahead of the whitened
tuned space. The analysis first fused only raw spaces with the words, which is why the
baseline reading above spoke of the word space getting worse; the whitened fusions were added
and every fold-0 run rescored, with every earlier number unchanged.

**The other folds, one launch each** (`identity_encoder_fold{k}_gradcache64.json`,
`identity_encoder_analysis_fold{k}_gradcache64.json`, project seed, same configuration; the
34 held-out labels are the same in every fold, the test fold changes). Rows are added as
folds finish; the rule is the one above.

| fold | test queries | whitened tuned − whitened frozen, test | the same, unseen labels | whitened tuned with words − whitened frozen with words, unseen | whitened tuned with words − words, unseen | participation ratio |
|---|---|---|---|---|---|---|
| 0 | 1,187 | +0.041 [+0.020, +0.062] | +0.039 [+0.019, +0.060] | +0.024 [+0.006, +0.042] | +0.052 [+0.033, +0.073] | 40.6 |
| 0, seed 2 | 1,187 | +0.045 [+0.024, +0.066] | +0.050 [+0.031, +0.070] | +0.037 [+0.021, +0.054] | +0.066 [+0.044, +0.087] | 47.0 |
| 1 | 1,239 | +0.048 [+0.030, +0.067] | +0.030 [+0.012, +0.048] | +0.014 [−0.001, +0.030] | +0.039 [+0.018, +0.060] | 50.5 |
| 2 | 1,284 | +0.049 [+0.030, +0.066] | +0.020 [+0.003, +0.039] | +0.014 [−0.002, +0.032] | +0.040 [+0.019, +0.062] | 43.0 |
| 3 | 1,237 | +0.047 [+0.027, +0.066] | +0.028 [+0.008, +0.048] | +0.017 [−0.002, +0.036] | +0.047 [+0.026, +0.069] | 46.1 |
| 4 | 1,232 | +0.025 [+0.006, +0.043] | +0.029 [+0.011, +0.049] | +0.016 [−0.001, +0.033] | +0.036 [+0.015, +0.058] | 35.7 |

Fold 1 (stopped three times for a game and resumed from its checkpoints at steps 190, 280 and
420) passes the rule: +0.030 on the unseen labels with the interval clear of zero. Its
whitened fusion with the words beats the whitened frozen fusion on the test fold (+0.026
[+0.010, +0.042]) but on the unseen labels its interval touches zero (+0.014 [−0.001,
+0.030]); against the words alone it is clear (+0.039). Fold 2 (450 steps; resumed once from
step 30 after the machine went into standby) passes too, with the smallest unseen-label margin
so far, +0.020 [+0.003, +0.039]; its test-fold gain is as large as the others (+0.049), and
its fusion pattern is fold 1's (+0.021 [+0.007, +0.035] on the test fold, +0.014 [−0.002,
+0.032] on the unseen labels, +0.040 against the words alone). Fold 3 (474 steps, no
interruption) passes, +0.028 [+0.008, +0.048]; it is the first fold where the raw tuned space
is not below the frozen one on the test fold (+0.005 [−0.012, +0.022]), and the first where
the fusion gain over the whitened frozen fusion misses on the test fold as well (+0.015
[−0.001, +0.031]; unseen +0.017 [−0.002, +0.036]; against the words alone +0.047). Fold 4
(478 steps; resumed from checkpoints at steps 50, 130 and 370 after two stops for a game and
one for battery power) passes, +0.029 [+0.011, +0.049], with the smallest test-fold gain,
+0.025 [+0.006, +0.043].

All five folds pass the rule written before the first run: unseen-label gains of +0.020 to
+0.039 with one launch per fold (+0.050 on fold 0's second seed), test-fold gains of +0.025 to
+0.049. Pooled over the five test folds, so that each of the 6,179 songs of the 192 seen
labels is scored once as an out-of-fold query under the adapter that never saw its fold:

| pooled over the five test folds | MRR |
|---|---|
| frozen, masked text | 0.284 |
| fine-tuned | 0.273 |
| frozen + within-author whitening | 0.384 |
| fine-tuned + within-author whitening | 0.425 |
| words | 0.509 |
| whitened frozen fused with words | 0.523 |
| whitened fine-tuned fused with words | 0.543 |

The pooled numbers are query-weighted means of the per-fold values and carry no interval:
the folds are disjoint query sets scored under different adapters. The fusion gain over the
whitened frozen fusion is significant in fold 0 (both scopes, both seeds) and on the test
folds of 1, 2 and 4, and not on the unseen labels of folds 1 to 4 (+0.014 to +0.017, each
interval touching zero); against the words alone it is significant everywhere, +0.036 to
+0.066 on the unseen labels. Reading: with a 64-anchor batch, contrastive adaptation learns
identity beyond within-author whitening in every fold, and the learned part transfers to
labels never trained on. Fused with the words it beats the words alone everywhere and the
whitened frozen fusion on seen labels; on unseen labels that last margin is small and not
established. The raw tuned space stays below the frozen one in four folds of five.

**What the fine-tune learned, part A: description** (`finetune_learned_identity.json`,
`src/finetune_learned_identity_v3.py`). For each fold the tuned and frozen song vectors are
whitened on the fold's training songs; the whitened MRRs reproduce the analysis files in every
fold. The new part of the tuned space is what a ridge regression from the whitened frozen
vectors, fitted on the training songs, does not predict.

| over the five folds | test fold | unseen labels |
|---|---|---|
| R² of the whitened tuned vectors from the whitened frozen ones | 0.29–0.31 | 0.32–0.34 |
| MRR of the new part alone | 0.196–0.215 | 0.230–0.250 |
| MRR of random vectors, same protocol | 0.025–0.029 | 0.026–0.029 |
| linear CKA with the word TF-IDF space: frozen raw → tuned raw | 0.39–0.41 → 0.26–0.29 | 0.39 → 0.25–0.28 |
| linear CKA with the word space: frozen whitened → tuned whitened | 0.66–0.69 → 0.66–0.69, +0.004 to +0.008 each fold | 0.66–0.67 → 0.67–0.68, +0.007 to +0.010 each fold |

Among the 50 words most correlated with each of the six leading directions of the new part,
42–55% are English (Latin-script) tokens and 6–9% function words; for the whitened frozen space
the figures are 37–45% and 13–16%. The English share is higher and the function-word share lower
in every fold.

Reading, descriptive only. Fine-tuning reorganised the space (about a third of it linear in the
frozen one) and added identity that no linear map of the whitened frozen space supplies. Its
leading new directions lean toward English tokens. After whitening, its geometry is very slightly
closer to the word space; raw, it is further from it. None of this says what the added identity
rests on: correlation with English tokens is not dependence on them, which part B tests by masking.

A first version of part A removed from each space its ridge prediction from the word space and
compared what was left. It was dropped after one fold, before its result was used: a regression
from words to vectors fitted on songs of the same labels learns "this label's words, this
label's direction" and so strips seen-label identity whether or not the spaces share information
(fold 0: whitened frozen 0.401 → 0.143 on the test fold, 0.336 → 0.337 on unseen labels, at an
out-of-sample R² of 0.01).

A related methods note found while checking it: a word SVD fitted on the training songs only
projects out-of-sample queries onto directions of other songs, while label profiles are built
mostly from in-sample songs. In fold 0 that lowers the whitened word SVD on the test fold from
0.582 (SVD fitted on all songs, which uses no labels) to 0.561 (fitted on the training folds, as
in `word_space_probe.json`). The published word-SVD numbers are therefore conservative.

**What the fine-tune learned, part B: masking query words** (`finetune_cue_masking.json`,
`src/finetune_cue_masking_v3.py`). Each fold's query songs (its test-fold songs and the 1,041
songs of the 34 unseen labels) are re-encoded by the fold's adapter and by the same model with the
adapter disabled, with one kind of word replaced by the tokenizer's mask token: English tokens
(jieba words containing a Latin letter) or the 100 Han words found in the most songs. A control
replaces other words of the same chunk, chosen at random. Label profiles stay unmasked, and
whitening is fitted on the unmasked training songs. Dependence is the tuned model's advantage over
the frozen one under the control minus that advantage under masking the kind of word, with a group
bootstrap interval per fold. The rule, written before the first run: the advantage rests on a kind
of word if dependence clears zero in at least four of five folds, and does not if it fails to in at
least four.

Checks in every fold: 256 unmasked query chunks reproduce the saved vectors of both models (minimum
cosine 1.000); every masked encoding stored by an earlier run reproduces from the rebuilt texts on 64
chunks (minimum cosine 1.000); stage 2 rebuilds every masked text and matches stage 1's digests; the unmasked MRRs
equal the analysis files (gap 0.000).

The design grew four times, each step recorded in the script before the result it could affect:

1. After fold 0's word counts and before any masked MRR: English-dominated chunks have too few
   other words for a full control. The rule is also applied to the songs whose every chunk has a
   full control, and a reading is kept only when all songs and those songs agree.
2. After fold 0's result: a control can remove the advantage by itself, which leaves nothing for
   dependence to show. A reading other than "rests on" is marked uninformative unless the advantage
   under the control clears zero in at least four folds.
3. After folds 0–2 with the word-matched control, before any token-matched MRR: the masked words
   differ in length from their controls (table below). A control matched in model tokens was added,
   and an overall reading is stated only when the word- and token-matched controls agree.
4. After all five folds with those two, before any single-token MRR: every common Han word is one
   model token, so the token-matched control masks fewer words and places fewer mask tokens. A
   control of as many words drawn only from one-token words matches words, mask tokens and, for the
   common words, model tokens. A token-matched "rests on" is attributed to the words only if this
   control also gives an informative "rests on".

| per fold (7,051–8,060 query chunks) | words masked | model tokens | characters |
|---|---|---|---|
| English tokens | 182,191–192,226 (16%) | 224,391–236,686 | 668,386–706,716 |
| word-matched control | 154,767–163,301 | 191,559–202,548 | 235,715–248,383 |
| token-matched control | 153,201–163,763 | 189,355–202,360 | 232,970–248,199 |
| single-token control | 150,182–158,166 | 150,182–158,166 | 200,754–210,645 |
| 100 common Han words | 373,387–386,235 (33%) | 373,387–386,235 | 422,053–436,122 |
| word-matched control | 368,992–381,454 | 519,861–536,556 | 807,411–833,687 |
| token-matched control | 267,649–277,427 | 374,539–387,603 | 586,782–607,762 |
| single-token control | 336,868–347,887 | 336,868–347,887 | 665,271–685,792 |

English controls fall short of their target mostly in English-dominated chunks.

**Both encoders lean on English tokens.** Masking them lowers MRR more than masking the control
words, in every fold, under every control, for both models: on the test folds by +0.097 to +0.132
for the frozen encoder and +0.115 to +0.151 for the tuned one, on the unseen labels by +0.061 to
+0.079 and +0.056 to +0.102. Masking the common Han words lowers MRR no more than the single-token
control (frozen, test folds: −0.024 to −0.000), which removes about 1.6 times their characters.

**Dependence of the tuned advantage** (unmasked advantage +0.025 to +0.049 on the test folds,
+0.020 to +0.039 on the unseen labels):

| test folds | all songs: dependence, folds clear of zero | songs with a full control |
|---|---|---|
| English, word-matched | +0.011 to +0.033, 3 of 5 | +0.016 to +0.041, 4 of 5 (923–1,004 songs) |
| English, token-matched | +0.014 to +0.040, 2 of 5 | +0.020 to +0.043, 4 of 5 (914–989 songs) |
| English, single-token | +0.009 to +0.030, 2 of 5 | +0.022 to +0.029, 5 of 5 (880–959 songs) |
| common Han, word-matched | −0.002 to +0.008, 0 of 5; the control alone removes the advantage in 4 folds | −0.003 to +0.012, 0 of 5 |
| common Han, token-matched | +0.011 to +0.029, 4 of 5; the advantage survives the control in 5 folds | +0.009 to +0.031, 4 of 5 |
| common Han, single-token | +0.005 to +0.012, 0 of 5; the control alone removes the advantage in 3 folds | −0.002 to +0.022, 0 of 5 (624–677 songs) |

Readings by the rules above:

- **English tokens, test folds:** "depends on the songs whose control is short", under all three
  controls. All 30 fold estimates are positive. Dependence clears zero in 4, 4 and 5 folds among the
  three quarters of test songs with a full control, and in 2–3 folds over all songs.
- **Common Han words, test folds:** the word- and token-matched controls disagree. The single-token
  control shows no dependence in any fold, so the token-matched dependence could come from the
  number of mask tokens rather than from the words.
- **Unseen labels:** uninformative for both kinds of word under all three controls. Masking a sixth
  of the words at random (the English controls) already takes the advantage below significance in two
  or three folds, and a third (the common-word controls) in four or five. English dependence clears
  zero only in folds 3 and 4, and not under every control. The unseen-label queries are the same 1,041
  songs in every fold, so these five results share their queries.

Reading. On labels it was trained on, the fine-tune's extra identity leans on English tokens beyond
their amount, whether amount is counted in words, model tokens or mask tokens, in the songs where a
full control can be built. This matches part A's English-leaning directions, but the rule does not
state it for all songs. There is no evidence that the extra identity rests on the common Han words:
a dependence that the token-matched control showed disappeared once mask tokens were matched as
well. What the part that transfers to new labels rests on, this test cannot say. Limits: a mask
token starts a new tokenizer piece, so masked texts can be longer in model tokens than the original
(masking the common words adds about 14 tokens per chunk). For the 1,091–1,148 chunks per fold
that reach the 512-token limit, masking changes how much of the end the model reads.

## Robustness of the headline ordering (2026-09-20 and 2026-09-21)

Analyses asked for by a simulated review of the manuscript plan. Each script reproduces the
published numbers before scoring anything new, fixes its reading rule in its docstring before
the run, and has a synthetic test under `tests/` that checks its scorers against brute-force
definitions without the corpus.

### Resampling the labels: the two-stage bootstrap and the label-macro estimand (`estimand_two_stage.json`, `src/estimand_two_stage_v3.py`)

Every interval published on corpus v3 resamples leakage groups and reads the
component-weighted MRR, which conditions on the 226 labels and lets a label with a hundred songs
count twenty times a label with five. The retrieval builder's own design, used on corpus v2 and
never run on v3, reads the label-macro MRR (component-weighted mean within a label, then the
plain mean over labels) under a paired two-stage bootstrap: labels drawn with replacement, then
each drawn label's components drawn with replacement (5,000 replicates, seed 20260825). The script
calls the builder's own `component_metric_values` and `run_bootstrap` through the constant
redirect `build_downstream_retrieval_v2` uses. Checks: all nine systems reproduce their
published query-weighted MRR (gap 0.0000) and, where published, their component-weighted MRR;
the three group-bootstrap contrasts already published on these ranks reproduce with their
intervals; the builder's component tensors equal the script's component means (gap 0) and its
point estimate equals the label-macro of the ranks (gap 1e-16). Rule, fixed before the run: a
contrast is established under label resampling if its two-stage 95% interval for the
label-macro MRR difference excludes zero.

| system | MRR, query-weighted | component-weighted | label-macro [two-stage 95%] | R@1, query-weighted | R@1, label-macro |
|---|---|---|---|---|---|
| semantic, raw | 0.2997 | 0.2979 | 0.2831 [0.258, 0.308] | 0.2019 | 0.1936 |
| characters, raw | 0.4266 | 0.4310 | 0.3765 [0.343, 0.410] | 0.3342 | 0.2947 |
| words, raw | 0.4963 | 0.5020 | 0.4316 [0.395, 0.468] | 0.4029 | 0.3481 |
| semantic, total whitening | 0.3996 | 0.3968 | 0.3924 [0.367, 0.417] | 0.3062 | 0.3006 |
| semantic, within-label whitening | 0.4164 | 0.4136 | 0.4062 [0.380, 0.431] | 0.3179 | 0.3101 |
| semantic, within-label whitening on permuted labels | 0.3983 | 0.3958 | 0.3919 [0.367, 0.416] | 0.3051 | 0.3011 |
| word SVD-1024, within-label whitening | 0.5426 | 0.5452 | 0.5121 [0.481, 0.543] | 0.4334 | 0.4039 |
| semantic, chunks whitened then averaged | 0.4271 | 0.4247 | 0.4174 [0.391, 0.443] | 0.3266 | 0.3205 |
| fusion of the two whitened spaces | 0.5854 | 0.5846 | 0.5608 [0.531, 0.589] | 0.4794 | 0.4581 |

| contrast | component-weighted MRR, group bootstrap | label-macro MRR, two-stage | label-macro R@1, two-stage | reading |
|---|---|---|---|---|
| words, raw − characters, raw (confirmatory) | +0.0711 [+0.0643, +0.0785] | +0.0551 [+0.0409, +0.0692] | +0.0534 [+0.0374, +0.0695] | established |
| characters, raw − semantic, raw (confirmatory) | +0.1331 [+0.1229, +0.1435] | +0.0935 [+0.0698, +0.1182] | +0.1011 [+0.0765, +0.1265] | established |
| words, raw − semantic, raw (confirmatory) | +0.2042 [+0.1944, +0.2142] | +0.1486 [+0.1201, +0.1767] | +0.1545 [+0.1255, +0.1836] | established |
| semantic, within-label whitening − semantic, total whitening (confirmatory) | +0.0168 [+0.0142, +0.0195] | +0.0137 [+0.0094, +0.0180] | +0.0095 [+0.0035, +0.0154] | established |
| semantic, within-label whitening − semantic, within-label whitening on permuted labels (confirmatory) | +0.0179 [+0.0153, +0.0205] | +0.0143 [+0.0093, +0.0191] | +0.0090 [+0.0017, +0.0159] | established |
| word SVD-1024, within-label whitening − semantic, within-label whitening (confirmatory) | +0.1316 [+0.1217, +0.1418] | +0.1059 [+0.0835, +0.1278] | +0.0938 [+0.0684, +0.1188] | established |
| fusion of the two whitened spaces − words, raw (confirmatory) | +0.0826 [+0.0746, +0.0898] | +0.1292 [+0.1067, +0.1525] | +0.1101 [+0.0872, +0.1334] | established |
| word SVD-1024, within-label whitening − semantic, chunks whitened then averaged (supplementary) | +0.1205 [+0.1108, +0.1305] | +0.0947 [+0.0720, +0.1164] | +0.0834 [+0.0574, +0.1084] | established |
| fusion of the two whitened spaces − word SVD-1024, within-label whitening (supplementary) | +0.0394 [+0.0329, +0.0461] | +0.0487 [+0.0360, +0.0615] | +0.0542 [+0.0378, +0.0711] | established |

All 9 of 9 contrasts are established under label resampling, the seven confirmatory ones
included; the two resamplings agree on every sign. The two-stage intervals are 1.6 to 3.0
times as wide as the group-bootstrap ones, which is the between-label variation the group
bootstrap holds fixed. Label-macro levels sit below the query-weighted ones in every space
(words 0.4316 against 0.4963, characters 0.3765 against 0.4266, semantic 0.2831 against
0.2997), most in the sparse spaces: labels with few songs score lower under the prototype
scorer, as the label-size section shows, and the label-macro estimand gives them equal weight.
The margins that were thin under group resampling stay clear of zero: within-label minus total
whitening +0.0137 [+0.0094, +0.0180], and within-label minus the permuted-label null +0.0143
[+0.0093, +0.0191]. The fusion's lead over the raw word space is larger under the label-macro
estimand (+0.129 against +0.083), consistent with the label-size section, where the whitened
spaces lose far less than the raw word space among labels with few songs.

### Fusion weights fitted inside the training folds (`fusion_weights_nested.json`, `src/fusion_weights_nested_v3.py`)

Every published fusion averages the two spaces' row-wise z-scores with weight 0.5, never tuned
and never tested. Here the weight w of the word space (grid 0.00 to 1.00 in steps of 0.05) is
chosen for each test fold on its training folds only, from inner scores whose SVD and whitening
were fitted on the three folds outside both the test fold and the training fold being scored,
and then applied to the test fold's published scores. Nine published numbers are reproduced
first (the four raw spaces under both published rank policies for rhyme, the fold-wise spaces
0.5426 and 0.4271, the fusions 0.4894 and 0.5854); fused(0.5) is bitwise the published
expression. Rules, fixed before the run: a fusion adds to its better component if the
fitted-weight fusion minus that component is positive with the interval clear of zero; rhyme
adds nothing beyond the surface if words + rhyme minus words is not; the published equal weight
"exceeds honest weighting", "is conservative" or "stands" by the sign of fitted minus equal.

| pair | better component | equal weight (published) | fitted weight | w chosen per fold | fitted − better component | fitted − equal | w chosen on the test fold itself − fitted (description) |
|---|---|---|---|---|---|---|---|
| whitened word SVD + chunk-whitened semantic (the published best system) | 0.5426 | 0.5854 | 0.5908 | 0.65, 0.60, 0.60, 0.60, 0.60 | +0.0455 [+0.0399, +0.0511] | +0.0061 [+0.0034, +0.0088] | +0.0011 [-0.0007, +0.0029] |
| raw words + raw semantic | 0.4963 | 0.4894 | 0.5040 | 0.75, 0.75, 0.75, 0.70, 0.70 | +0.0074 [+0.0039, +0.0107] | +0.0156 [+0.0119, +0.0195] | +0.0014 [-0.0002, +0.0030] |
| raw words + raw characters | 0.4963 | 0.4835 | 0.4964 | 0.85, 0.85, 0.85, 0.80, 0.95 | +0.0001 [-0.0019, +0.0022] | +0.0134 [+0.0100, +0.0171] | +0.0018 [+0.0001, +0.0033] |
| raw words + strict rhyme form (7,117 covered songs) | 0.4940 | 0.4170 | 0.4940 | 0.90, 0.85, 0.95, 0.85, 0.85 | +0.0002 [-0.0021, +0.0025] | +0.0776 [+0.0717, +0.0839] | +0.0029 [+0.0009, +0.0049] |

Readings. The published best system was not selected on the evaluation queries: a weight
fitted without them gives 0.5908 against the published 0.5854, so the published number is
conservative, and choosing the weight on the test fold itself would add +0.0011 [−0.0007,
+0.0029], an interval that includes zero. The fusion
adds 0.046 to the whitened word space. Raw words with raw semantic adds to the words under a
fitted weight (+0.007) where the published equal-weight fusion fell below them (−0.008). Words
with characters adds nothing. Rhyme adds nothing beyond the surface: words + rhyme minus words
is +0.0002 [−0.0021, +0.0025] with the weight fitted, so the redundancy reading no longer rests
on an untuned equal weight, under which the rhyme space simply dragged the fusion down (−0.077).
In every pair the fitted weight favours the word space (0.60 to 0.95).

### Cosine Delta and the TF-IDF configuration (`cosine_delta_and_tfidf_grid.json`, `src/cosine_delta_and_tfidf_grid_v3.py`)

Two questions a stylometry reviewer asks. (A) How does Delta, the field's standard
most-frequent-word method (Burrows 2002; Evert et al. 2017), compare with the word TF-IDF space
under the same leave-group-out protocol? MFW lists at six sizes fixed in advance are counted by
document frequency over the training-fold songs, relative frequencies of the word space's own
jieba unigrams (punctuation tokens included) are z-scored with means and deviations fitted on
the training folds, and each query is scored against its label's leave-group-out mean. (B) Does
the word-over-character ordering depend on the untuned TF-IDF configuration? A grid of
vocabulary caps, n-gram ranges and sublinear or raw term frequency is searched per test fold on
its training-fold queries only, and the chosen cell scores the test fold; the published cells
and an inductive variant (vocabulary and idf fitted on the training folds only) are reported
beside it. Checks first: the protocol code and the grid's own published cells reproduce 0.4266
and 0.4963; the count-matrix TF-IDF equals sklearn's within 6e-8; the vectorised scorers equal
the protocol's and a brute-force Burrows loop. Rules, fixed before the run: Delta matches the
word space if its best size reaches 0.4963 − 0.02; the ordering is configuration-robust if
inner-selected words minus inner-selected characters is positive with the interval clear of zero.

| MFW | 100 | 300 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| Cosine Delta (z-scored relative frequencies) | 0.2008 | 0.2622 | 0.2868 | 0.3116 | 0.3322 | 0.3543 |
| Cosine Delta, unit rows | 0.2117 | 0.2785 | 0.3015 | 0.3340 | 0.3519 | 0.3761 |
| Burrows's Delta | 0.1498 | 0.1271 | 0.0983 | 0.0837 | 0.0619 | 0.0427 |
| share of a song's tokens covered (fold 0) | 0.36 | 0.50 | 0.56 | 0.63 | 0.71 | 0.80 |

Reading A: Delta does not match the word space: cosine_delta is best at 5000 MFW with 0.3543, below 0.4763. Best size minus words: Cosine Delta
-0.150 [-0.159, -0.141], with unit rows -0.128 [-0.137, -0.119]. Both cosine
curves are still rising at 5,000 words, the largest size fixed, so the verdict holds for these
sizes only; Burrows's Delta falls as the list grows. Songs are a few hundred tokens long and
Delta was built for texts of thousands of words, where z-scores of rarer words are stable; the
comparison is for Delta under this protocol, on the word space's own tokens.

| system | characters | words | words − characters |
|---|---|---|---|
| published configuration | 0.4266 | 0.4963 | +0.0711 [+0.0643, +0.0785] |
| selected on training folds | 0.4388 | 0.5112 | +0.0708 [+0.0647, +0.0771] |
| published configuration, inductive idf | 0.4214 | 0.4928 | +0.0726 [+0.0657, +0.0800] |
| selected, inductive idf | 0.4356 | 0.5102 | +0.0729 [+0.0668, +0.0793] |
| best cell with raw term frequency (description) | 0.3244 | 0.3246 | +0.0009 [-0.0069, +0.0090] |

Reading B: configuration-robust (inner-selected words minus inner-selected characters +0.0708 [+0.0647, +0.0771]); the inductive contrast
agrees. The same cell was chosen in all five folds: `ngram 1-3 | max_features 150000 | sublinear_tf true` for characters and
`ngram 1 | max_features 150000 | sublinear_tf true` for words. The published configurations are not the best of their grids:
selection on training folds gains +0.0127 [+0.0084, +0.0170] for words and
+0.0129 [+0.0080, +0.0183] for characters, so the published word and character numbers are
slightly conservative. Fitting the idf inductively costs 0.001 to 0.005. Outside the rule, added
after a first run: with raw instead of sublinear term frequency both spaces lose about a third of
their MRR and the word lead disappears (+0.0009 [-0.0069, +0.0090]). Every sublinear word cell
scores above every sublinear character cell (0.4940 or more against 0.4388 or less), so the
ordering needs repeated words damped, which is what sublinear tf does.

### A discriminative classifier against the prototype (`discriminative_baseline.json`, `src/discriminative_baseline_v3.py`)

Every published system scores a query against a label's mean profile. The standard
authorship-attribution baseline is instead a linear classifier trained on the songs. Per space,
one-vs-rest linear SVM and logistic regression (liblinear primal solvers, the protocol's
component weights as sample weights) are trained on the four training folds and score the test
fold; C is chosen per fold from {0.1, 1, 10} on two halves of the training folds. Because a
classifier sees only training-fold songs while the published prototype builds profiles from all
other songs, the rule compares it with the prototype built from the training folds alone (fold
prototype). Checks first: the published prototype reproduces 0.4266 / 0.4963 / 0.2997 / 0.4164
with R@1; the fold prototype equals the protocol's scorer when all songs are available; the
one-vs-rest loop equals sklearn's native one on a 30-label subset. Rule, fixed before the run: a
space reads by its better classifier, which beats the prototype if its MRR minus the fold
prototype's is positive with the interval clear of zero. Added after a reviewer's inner-split
probe (no test fold scored in the TF-IDF spaces), description only: logistic regression on the
wider grid {10, 100, 1000}, and a qualifier when the selected C sits on a grid edge in every fold.

| space | published prototype (all songs) | fold prototype | linear SVM | logistic regression | logistic regression, wide grid (description) | SVM − fold prototype |
|---|---|---|---|---|---|---|
| character 2–5-grams | 0.4266 | 0.3993 | 0.5414 | 0.4558 | 0.4967 | +0.139 [+0.132, +0.147] |
| jieba words | 0.4963 | 0.3791 | 0.4823 | 0.4129 | 0.4534 | +0.099 [+0.093, +0.106] |
| semantic, raw | 0.2997 | 0.2883 | 0.3719 | 0.3443 | 0.3496 | +0.085 [+0.078, +0.092] |
| semantic, within-label whitened | 0.4164 | 0.3923 | 0.3699 | 0.3632 | 0.3560 | -0.020 [-0.025, -0.015] |

Readings: the SVM beats the fold prototype in the character (+0.139), word (+0.099) and raw
semantic (+0.085) spaces; in the whitened semantic space it loses (−0.020), with its C on the
lower grid edge in every fold, so that reading is one of the grid. The SVM chose C = 1 in every
fold of the other three spaces, inside the grid. Logistic regression chose the top of its grid in
every fold and space, and the top of the wide grid (1,000) in the two TF-IDF spaces, so it is not
at its optimum anywhere; the SVM is the classifier that carries the reading.

Space orderings (adjacent contrasts, paired group bootstrap):
- published prototype: words > characters > semantic_whitened > semantic_raw; words − characters +0.071 [+0.064, +0.079]; characters − semantic_whitened +0.017 [+0.007, +0.028]; semantic_whitened − semantic_raw +0.116 [+0.107, +0.124].
- fold prototype: characters > semantic_whitened > words > semantic_raw; characters − semantic_whitened +0.014 [+0.004, +0.025]; semantic_whitened − words +0.004 [-0.007, +0.015]; words − semantic_raw +0.100 [+0.090, +0.110].
- linear SVM: characters > words > semantic_raw > semantic_whitened; characters − words +0.058 [+0.050, +0.066]; words − semantic_raw +0.114 [+0.104, +0.124]; semantic_raw − semantic_whitened +0.001 [-0.005, +0.007].

Reading. Two headline statements depend on the scorer. The word space leads the character space
only under the published prototype; under a classifier trained on the training folds the
character space leads by 0.058, and under the fold prototype characters score 0.020 above words
(0.3993 against 0.3791; no interval was computed for that pair, the whitened semantic space
sitting between them). Building profiles from the training folds instead of all other songs
costs the word prototype 0.116 and the character prototype 0.027: the sparse word profile needs
many songs, as the label-size section found. Second, whitening is what lets cosine read identity
out of the semantic space, but a linear classifier reads it from the raw vectors directly (0.372
raw against 0.370 whitened, a difference of +0.001 [−0.005, +0.007]). What holds under every
scorer is that the lexical surface identifies the credited label far better than the semantic
content, and that the semantic space holds more identity than raw cosine shows.

### Repeated passages within a label (`within_label_repeats.json`, `src/within_label_repeats_v3.py`)

Leave-group-out removes the query's leakage group, not the label's other songs that share a
hook with it. A detector joins two same-label songs when their normalised texts share a
30-character run; it must find every whole-chunk match of the label-identity audit's rule
(0 of 319 missed). 601 same-label pairs share a passage, 351 of them already
inside one leakage group. Arm (a) merges the rest into leakage groups (5,875 → 5,691
groups, largest 27 → 42, no label lost). Arm (b) strips every line that recurs in two or more
leakage groups of a label from all of that label's songs (14,826 line occurrences, 2.2% of
characters, 2,254 songs; 20 songs fall below the length rule); the recorded chunk vectors cannot
be re-embedded on the CPU, so the semantic space is bracketed by dropping only emptied chunks
(favours the semantic space) or every touched chunk (favours the surface).

| | words | characters | semantic | words − characters | characters − semantic | words − semantic |
|---|---|---|---|---|---|---|
| published | 0.4963 | 0.4266 | 0.2997 | +0.071 [+0.064, +0.079] | +0.133 [+0.123, +0.143] | +0.204 [+0.194, +0.214] |
| (a) merged groups | 0.4780 | 0.4066 | 0.2918 | +0.074 [+0.067, +0.081] | +0.120 [+0.110, +0.130] | +0.193 [+0.183, +0.203] |
| (b) stripped lines, emptied chunks dropped | 0.4596 | 0.3864 | 0.2940 | +0.074 [+0.067, +0.081] | +0.099 [+0.088, +0.109] | +0.172 [+0.162, +0.183] |
| (b) stripped lines, touched chunks dropped | | | 0.2699 | | +0.122 [+0.112, +0.132] | +0.196 [+0.186, +0.207] |

Reading, by the rule: not driven by repeated passages. Repeats are worth 0.02 to 0.04 of MRR in the
surface spaces and under 0.01 in the semantic space, and the ordering does not move. Outside the
rule: with recurrent lines stripped the whitened semantic space (0.4069) passes the character
space (0.3864) under the bracket that favours it (-0.014 [-0.024, -0.003]) and not under the other
(+0.027 [+0.016, +0.038]), so the characters-over-whitened-semantic margin is within what repeated lines carry.

### Label names, collaborations, near-identical rivals and the leakage threshold (`label_mask_sensitivity.json`, `src/label_mask_sensitivity_v3.py`)

The fine-tuning pipeline masks a song's own label string; the headline spaces did not. Arms,
each rerunning the raw spaces under the unchanged protocol after reproducing 0.2997 / 0.4266 /
0.4963: every label string of the corpus (240 strings of two characters or more) removed from
every chunk with the trainer's `mask_label`, longest first (5,892 occurrences of 193 strings
in 3,277 of 23,848 chunks, 13.7%; 2,371 own-label and 3,521 other-label mentions; the
audit's 1,610 self-naming chunks reproduced exactly); the 592 songs whose title marks a collaboration
removed as queries and profile members, groups rebuilt (the audit's count reproduced; the rule
reads the private title column and nothing else); queries dropped when any of their chunks has
cosine above 0.99 with a chunk of another label (1,767 queries by the audit's definition, 39 when the
rival must lie outside the query's own leakage group); leakage groups rebuilt at trigram Jaccard
0.70 and 0.90 instead of 0.80. Rule, fixed before the run: the ordering is robust to an arm if
every pairwise contrast stays positive with its interval clear of zero.

| arm | queries | words | characters | semantic | words − characters | characters − semantic | words − semantic |
|---|---|---|---|---|---|---|---|
| published | 7,220 | 0.4963 | 0.4266 | 0.2997 | +0.071 [+0.064, +0.079] | +0.133 [+0.123, +0.143] | +0.204 [+0.194, +0.214] |
| every label string masked | 7,218 | 0.4803 | 0.4116 | 0.2842 | +0.070 [+0.063, +0.077] | +0.134 [+0.124, +0.144] | +0.204 [+0.194, +0.214] |
| collaboration-titled songs excluded | 6,625 | 0.4914 | 0.4209 | 0.2925 | +0.071 [+0.064, +0.079] | +0.133 [+0.123, +0.144] | +0.205 [+0.194, +0.215] |
| queries with a near-identical rival chunk dropped | 5,453 | 0.5333 | 0.4547 | 0.3182 | +0.078 [+0.070, +0.087] | +0.140 [+0.129, +0.151] | +0.218 [+0.206, +0.229] |
| the same, rival outside the query's leakage group | 7,181 | 0.4968 | 0.4272 | 0.2997 | +0.071 [+0.064, +0.078] | +0.134 [+0.124, +0.143] | +0.205 [+0.194, +0.215] |
| leakage threshold 0.70 | 7,220 | 0.4950 | 0.4252 | 0.2993 | +0.071 [+0.065, +0.079] | +0.132 [+0.122, +0.142] | +0.203 [+0.194, +0.213] |
| leakage threshold 0.90 | 7,220 | 0.4980 | 0.4281 | 0.2995 | +0.071 [+0.065, +0.078] | +0.135 [+0.125, +0.145] | +0.206 [+0.196, +0.216] |

Readings: robust to every arm. For the semantic column under label masking the masked chunks were
re-embedded on the GPU with the pinned BGE-M3 (2026-09-21), after the run reproduced the recorded
vectors on 256 unmasked chunks (minimum cosine 0.999985); masking every name costs the semantic space
0.016. It costs the word space 0.016 and the character space 0.015: names are a
small part of the surface signal, which agrees with the name-neutralisation arms above. The
near-identical-rival arm shows that most such chunks sit inside the query's own leakage group,
which the protocol already holds out.

### Content rivals under other rival definitions (`content_rivals_labelfree.json`, `src/content_rivals_labelfree_v3.py`)

The published content-rival test takes each query's rivals from the frozen semantic space, so
the semantic family is judged on the labels it finds hardest. The published file is reproduced
first (every level, top-1 and contrast with its interval, worst gap 0.0000); rivals are then
redefined by a fold-wise 50-topic NMF on word counts (vocabulary, idf and topics fitted on the
training folds, no label in the model) and by the character space (excluded from the contrasts
it judges). The three definitions pick different rivals (mean Jaccard overlap of the 10-rival
sets 0.15, 0.16 and 0.17). Rule, fixed before the run: the word space's advantage over
the whitened semantic space among rivals is "not an artefact of the rival definition" only if
it is positive with the interval clear of zero under both new definitions at K = 5, 10 and 25.

| rivals defined by | K | words | whitened semantic | characters | frozen semantic | whitened semantic + words | words − whitened semantic |
|---|---|---|---|---|---|---|---|
| frozen semantic space (published) | 5 | 0.676 | 0.590 | 0.631 | 0.372 | 0.657 | +0.095 [+0.086, +0.104] |
| frozen semantic space (published) | 10 | 0.614 | 0.530 | 0.559 | 0.330 | 0.610 | +0.092 [+0.082, +0.102] |
| frozen semantic space (published) | 25 | 0.553 | 0.475 | 0.490 | 0.307 | 0.568 | +0.086 [+0.076, +0.096] |
| 50-topic NMF on word counts, fold-wise, label-free | 5 | 0.648 | 0.711 | 0.638 | 0.575 | 0.726 | -0.055 [-0.064, -0.046] |
| 50-topic NMF on word counts, fold-wise, label-free | 10 | 0.585 | 0.633 | 0.560 | 0.483 | 0.663 | -0.040 [-0.049, -0.030] |
| 50-topic NMF on word counts, fold-wise, label-free | 25 | 0.532 | 0.545 | 0.491 | 0.392 | 0.599 | -0.005 [-0.015, +0.005] |
| character space | 5 | 0.581 | 0.678 | 0.483 | 0.545 | 0.663 | -0.091 [-0.099, -0.081] |
| character space | 10 | 0.539 | 0.611 | 0.450 | 0.464 | 0.616 | -0.064 [-0.073, -0.053] |
| character space | 25 | 0.512 | 0.537 | 0.432 | 0.384 | 0.576 | -0.017 [-0.027, -0.007] |

Reading, by the rule: definition-dependent; the contrast fails in all six cells of the two new
definitions, where the whitened semantic space leads the word space by 0.005 to 0.091 (the
interval reaches zero only among 25 topic rivals). A rival
test handicaps the family of evidence that chooses its rivals: semantic rivals are the labels
the semantic spaces confuse, and rivals chosen by topics of words or by characters are the labels
the surface spaces confuse. No definition available here is neutral, since the topic model is
itself built from word counts (note added to the script after the run). Read together, the three
definitions say the two families are complementary: each separates the labels the other
confuses, and the whitened fusion is the best or second-best system in every cell. The published
sentence that the word space "still names the label first more than half the time" with subject
held constant holds only for semantically chosen rivals and is withdrawn as evidence that the
word space is the stronger one among content rivals. Over all 226 labels nothing changes (words
0.4963, whitened semantic 0.4164). Text-distortion arm (Stamatatos 2018), as description: keeping only
the 500 most frequent training-fold words (55% of word tokens) and replacing the rest with one
placeholder leaves 0.3848 of the word space's 0.4963 over all labels.

## Training-data audit for the identity encoder (`training_data_audit.json`)

Tokens per chunk median 76, p90 603, 3,669 over 512; all 226 labels have at least two
leakage groups; 592 songs (8.2%) carry a collaboration title; 1,610 chunks (6.6%) name
their own label; a chunk's nearest content rival (another label) is closer than its nearest
same-label chunk in 91.0% of cases (median cosine 0.725 vs 0.663).

## A note on estimands, and what it does and does not change (2026-09-23)

A pre-submission audit of our own scripts found a reporting inconsistency that runs through
almost every arm in this directory, and it is recorded here rather than silently repaired,
because the numbers themselves are not wrong - only their juxtaposition is.

Three estimands are in play. **Query-weighted** MRR gives each of the 7,220 queries weight one.
**Component-weighted** MRR gives each (leakage group, label) component weight one, which is what
the protocol's per-song weights implement. **Label-macro** MRR takes the component-weighted mean
within each label and then the plain mean over the 226 labels.

The convention, now stated explicitly: in every JSON in this directory, a system's `mrr` field is
**query-weighted** unless its name says otherwise, while every interval produced by
`paired_group_bootstrap` - that is, every `paired_contrasts` entry, and its `mrr_difference`
point estimate - is **component-weighted**. Both are correct, and each `mrr_difference` is
consistent with its own interval. What a reader must not do is subtract two printed `mrr` levels
and expect the published `mrr_difference`: those are two different estimands.

`estimand_two_stage.json` records both for the nine headline systems, so the size of the
discrepancy is measurable rather than a worry:

| system | query-weighted | component-weighted | gap | label-macro | gap |
|---|---|---|---|---|---|
| fusion of the two whitened spaces | 0.5854 | 0.5846 | -0.0008 | 0.5608 | -0.0246 |
| word SVD-1024, within-label whitening | 0.5426 | 0.5452 | +0.0026 | 0.5121 | -0.0305 |
| words, raw | 0.4963 | 0.5020 | +0.0057 | 0.4316 | -0.0647 |
| semantic, chunks whitened then averaged | 0.4271 | 0.4247 | -0.0024 | 0.4174 | -0.0097 |
| characters, raw | 0.4266 | 0.4310 | +0.0044 | 0.3765 | -0.0501 |
| semantic, within-label whitening | 0.4164 | 0.4136 | -0.0028 | 0.4062 | -0.0102 |
| semantic, total whitening | 0.3996 | 0.3968 | -0.0028 | 0.3924 | -0.0072 |
| semantic, within-label whitening on permuted labels | 0.3983 | 0.3958 | -0.0025 | 0.3919 | -0.0064 |
| semantic, raw | 0.2997 | 0.2979 | -0.0018 | 0.2831 | -0.0166 |

| contrast | component-weighted difference | query-weighted difference | gap | role |
|---|---|---|---|---|
| words, raw - characters, raw | +0.0711 | +0.0697 | +0.0014 | confirmatory |
| characters, raw - semantic, raw | +0.1331 | +0.1269 | +0.0062 | confirmatory |
| words, raw - semantic, raw | +0.2042 | +0.1966 | +0.0076 | confirmatory |
| semantic, within-label whitening - semantic, total whitening | +0.0168 | +0.0168 | +0.0000 | confirmatory |
| semantic, within-label whitening - semantic, within-label whitening on permuted labels | +0.0179 | +0.0181 | -0.0002 | confirmatory |
| word SVD-1024, within-label whitening - semantic, within-label whitening | +0.1316 | +0.1262 | +0.0054 | confirmatory |
| fusion of the two whitened spaces - words, raw | +0.0826 | +0.0891 | -0.0065 | confirmatory |
| word SVD-1024, within-label whitening - semantic, chunks whitened then averaged | +0.1205 | +0.1155 | +0.0050 | supplementary |
| fusion of the two whitened spaces - word SVD-1024, within-label whitening | +0.0394 | +0.0427 | -0.0033 | supplementary |

So the two estimands differ by at most **0.0057** on a level and **0.0076** on a
difference, across the nine headline systems. For the large contrasts this is immaterial: the
word-minus-semantic gap of 0.0076 sits on an effect of 0.20. For a small effect it is not
immaterial at all, and three published claims are in that range and are flagged here:

- `label_specific_covariance.json` prints `mrr_selected` 0.4272 and `mrr_cosine_prototype` 0.4164,
  a query-weighted gap of 0.0108, beside the paired contrast +0.0149 [+0.0088, +0.0212], which is
  component-weighted. The 0.0041 between them is the estimand, not an error. The claim - that a
  label-specific covariance model beats cosine - holds under both.
- `attention_exemplar.json` reports +0.0022 [+0.0003, +0.0042] for the selected temperature in
  whitened word SVD. That effect is smaller than the estimand gap on the headline contrasts, so
  it must not be compared across estimands, and the whole beta curve rather than the selected
  arm is the object to read (see the note on selection below).
- `temperature_scale.json` declares a non-inferiority margin of 0.005 and applies it to
  component-weighted differences, which is internally consistent; its per-space `mrr` levels are
  query-weighted, so they must not be subtracted to reconstruct the margin test.

On selection. The per-fold "selected" hyper-parameter in these arms is chosen on the other four
folds' queries, but the folds are query folds over the same 226 labels, the same leave-group-out
profiles and the same candidate set, so the other-folds curve is nearly the full-sample curve:
in characters, fold 0's other-folds MRR at beta = 10 is 0.4436 against a full-sample 0.4436. A
selected arm is therefore descriptive, not held out, and the whole parameter curve is the
primary object. This costs nothing: every interior temperature already carries its own interval
clear of zero (characters +0.0135 at beta 5, +0.0167 at 10, +0.0121 at 20), so the attention
result stands at a pre-specified temperature and does not depend on the selection.

## Is the attention temperature a cosine scale? (`temperature_scale.json`, 2026-09-22)

`src/temperature_scale_v3.py`, with `tests/test_temperature_scale.py` for its scorers without the corpus.

The attention arm swept a fixed grid of temperatures in `a_s = w_s exp(beta cos(q, x_s))` and the
temperature that won differs by a factor of 20 across the five spaces. A temperature is not a free number: beta multiplies
a cosine, so it carries the units of one over a cosine. If the spread is only a units effect, the
dimensionless product `beta* sigma` should be far more nearly constant than `beta*`, and a scorer
that sets `beta(q, l) = c / sigma(q, l)` with one global `c` should do what the per-space tuned grid
does with no per-space tuning at all. `sigma(q, l)` is the component-weighted sd of the query's
cosines to label `l`'s songs outside the query's leakage group, so it needs no labels at scoring time.

Rules, fixed in the docstring before the run: the dimensionless reading holds if the spread of
`beta* * median sigma` across the five spaces is below 2, is *reduced but not constant* below 5, and
**fails** above 5; the one-constant scorer is non-inferior to the per-space tuned beta in a space if
the whole 95% interval of the difference lies above -0.005 (a margin declared in advance, so that
a scorer that simply lacks power cannot pass). `c` is chosen per fold on the other folds' queries
averaged over the five spaces; the tuned beta is chosen per fold the same way, on the same grid the
attention arm used. The per-pair rule is primary; a per-query variant that scales by the sd of the
query's cosines to the whole candidate pool, ignoring the label, is the secondary arm.

Checks: `c = 0` reproduces the published prototype in all five spaces (largest gap 3e-05) and is
bit-for-bit the prototype scorer (largest gap 0); the vectorised per-pair scorer equals a
brute-force loop on six query-label pairs per space (largest gap 2.2e-16), and `sigma(q, l)` from the
block loop equals a direct computation on the same pairs (gap 0).

| space | median sigma(q, l) | median own-label mean cosine | beta* | beta* x sigma |
|---|---|---|---|---|
| words, raw | 0.0118 | 0.0331 | 20 | 0.236 |
| characters, raw | 0.0165 | 0.0342 | 10 | 0.165 |
| semantic, within-label whitening | 0.0395 | 0.0194 | 5 | 0.198 |
| word SVD-1024, within-label whitening | 0.0508 | 0.0506 | 1 | 0.051 |
| semantic, raw | 0.0520 | 0.6507 | 20 | 1.040 |

**The pre-declared reading fails.** The product spans 20.39 times, which is not an improvement on
the 20 times that beta alone spans, so the temperature is not a pure scale parameter and the
one-constant law claimed in an earlier plan is withdrawn.

Post-hoc, and marked as post-hoc because the rule above was fixed first: the failure is carried by
one space. Four of the five have own-label cosines centred near zero (medians 0.0331, 0.0342, 0.0194, 0.0506), and raw BGE-M3 does not
(0.6507) - the known anisotropy of raw sentence embeddings. Among the four centred
spaces the product spans 4.6 times against 20 for beta, and beta* falls monotonically as sigma
rises. That is a reduced-but-not-constant reading on a subset chosen after seeing the numbers; it is
recorded as a hypothesis for the layer cells, where there are tens of replication units rather than five.

| space | best fixed beta | its MRR | best per-pair c | its MRR | one global c | tuned beta | one constant - tuned | non-inferior |
|---|---|---|---|---|---|---|---|---|
| words, raw | 20 | 0.5139 | 0.5 | 0.5112 | 0.5097 | 0.5139 | -0.0044 [-0.0065, -0.0022] | no |
| characters, raw | 10 | 0.4436 | 0.25 | 0.4422 | 0.4393 | 0.4436 | -0.0042 [-0.0066, -0.0018] | no |
| semantic, within-label whitening | 5 | 0.4496 | 0.25 | 0.4484 | 0.4475 | 0.4481 | -0.0006 [-0.0029, +0.0017] | yes |
| word SVD-1024, within-label whitening | 1 | 0.5457 | 0.25 | 0.5498 | 0.5429 | 0.5450 | -0.0031 [-0.0064, +0.0001] | no |
| semantic, raw | 20 | 0.3569 | 0.75 | 0.3584 | 0.3362 | 0.3569 | -0.0218 [-0.0275, -0.0165] | no |

One global `c` is non-inferior in 1 of five spaces (semantic, within-label whitening). It loses by
0.0218 in raw semantic and by 0.003 to 0.005 in the other three, so removing per-space tuning
costs a little everywhere and a lot in the one uncentred space. The dimensionless temperature is
nonetheless the more stable parameter: the per-space optimal `c` takes only the values 0.25, 0.5, 0.75
(a spread of 3) against 1 to 20 for beta. In word SVD-1024 the adaptive rule at its own
optimum reaches 0.5498, above the best fixed temperature's 0.5457 and above the
prototype's 0.5426 - the one space where the fixed grid had found almost nothing.

The label's own spread is what matters, not the query's. Replacing `sigma(q, l)` with the sd of the
query's cosines to the whole candidate pool, which ignores the label, costs -0.0062 (words, raw), -0.0054 (characters, raw), -0.0151 (word SVD-1024, within-label whitening),
is flat in semantic, within-label whitening, and gains +0.0037 in raw semantic, whose pool sd is almost constant across queries
(0.0483 to 0.0660 between the 5th and 95th percentiles). A repertoire has its own scale.

| space | effective sample size of the attention weights [5th, 50th, 95th] | largest single weight | songs in the profile |
|---|---|---|---|
| words, raw | 5.0, 36.7, 46.4 | 0.052 (95th 0.447) | 40 (5th 13, 95th 48) |
| characters, raw | 8.0, 38.1, 46.9 | 0.044 (95th 0.232) | 40 (5th 13, 95th 48) |
| semantic, within-label whitening | 10.6, 37.3, 46.4 | 0.050 (95th 0.207) | 40 (5th 13, 95th 48) |
| word SVD-1024, within-label whitening | 13.0, 39.4, 47.5 | 0.033 (95th 0.107) | 40 (5th 13, 95th 48) |
| semantic, raw | 6.7, 22.7, 35.1 | 0.141 (95th 0.401) | 40 (5th 13, 95th 48) |

**The recognition window is wide, and an earlier plan predicted the opposite.** At the temperature the
attention arm selected, the weights spread over 22.7 to 39.4 songs at the median, against a median
profile of 40 songs; the largest single weight is 0.033 to 0.141 at the median. A draft
plan of 2026-09-22 pre-registered a median effective sample size between 2 and 10; the measurement
refutes it, and it was withdrawn before the run on the arithmetic alone. The consequence is
substantive: attention at the selected temperature is not selecting one near-duplicate song, which is
the confound the repeated-passage arm was built to catch. Raw semantic concentrates most (median
22.7) and gains most (+0.0583); word SVD-1024 concentrates least (39.4) and gains least
(+0.0022). The upper tail is where concentration lives: at the 95th percentile of the largest weight,
raw words puts 0.447 of the profile on one song.

## Which labels are answered too often (`hubness_and_centrality.json`, 2026-09-23)

`src/hubness_and_centrality_v3.py`, with `tests/test_hubness_and_centrality.py`. In a fixed-corpus
retrieval task some candidates are returned for far more queries than their share, a geometry
effect rather than an accuracy one. Nothing here had measured it, and the size argument turns on
it: if a few labels absorb the top of every ranking, MRR is partly a statement about which labels
are central and how many songs they carry. k-occurrence N_k(l) counts the queries for which label
l is in the top k; it is a count over queries and has no component-weighted analogue, which the
payload says. Rules were fixed in the docstring before the run.

| space | N_10 skewness [label bootstrap] | one label's share of the top 10, over its mean | false-hub mass | top-1 entropy over labels |
|---|---|---|---|---|
| words, raw | 1.821 [1.385, 2.204] | 6.47x | 0.9329 | 0.8721 [0.8652, 0.8739] |
| characters, raw | 2.145 [1.603, 2.661] | 7.38x | 0.9394 | 0.8716 [0.8646, 0.8733] |
| semantic, raw | 0.836 [0.512, 1.111] | 3.38x | 0.9509 | 0.9353 [0.9292, 0.9355] |
| semantic, within-label whitening | 0.739 [0.324, 1.087] | 1.48x | 0.9397 | 0.9845 [0.9797, 0.9835] |
| word SVD-1024, within-label whitening | 0.834 [0.485, 1.118] | 2.85x | 0.9247 | 0.9528 [0.9471, 0.9526] |

**R1 holds, and its predicted ordering does not.** Hubness is present in all five spaces, every
interval clear of zero, but it is the LEXICAL spaces that are hubby, not raw BGE-M3: characters
2.15 and words 1.82 against raw semantic 0.84. An earlier plan predicted the
reverse ordering; it is refuted. Whitening reduces hubness, which is consistent with a large part
of what whitening buys being a de-hubbing correction - see the transform table below.

**R2, a prediction of ours, is refuted, and the reason is instructive.** The claim was that
kappa_l = r_l/E||m_l|| multiplies a label's whole score column, so a label with few songs has an
inflated ||m_l||, a smaller kappa_l and therefore fewer hub occurrences: ||m_l|| should carry a
NEGATIVE association. It carries a positive one in all five spaces +1.856, +0.957, +0.302, +0.075, +0.213,
every interval clear of zero. The diagnosis is a mis-specification, not a surprise about the
corpus: ||m_l||^2 = r_l^2 + s^2 W_l, so at fixed component count the profile norm IS the resultant
length. It measures how COHERENT a repertoire is, not how noisily its mean is estimated, and a
coherent label sits near more queries. The noise channel runs through the component count, and the
two cannot be separated in one linear model: they correlate -0.93, -0.81, -0.25, -0.92, -0.74.

**R2b, declared after R2 failed and therefore a re-specification rather than a confirmatory test,**
replaces the norm and the count with the two quantities they confound, which are algebraically
distinct: r_hat_l, the component-level U-statistic (coherence), and W_l = sum_c(1/k_c)/G_l^2, a
function of the group structure alone (estimation noise).

| space | coefficient on r_hat (coherence) | coefficient on W (noise) | their correlation | reading |
|---|---|---|---|---|
| words, raw | +0.6747 [+0.4895, +0.8714] | -0.8012 [-1.1843, -0.4539] | +0.23 | confirmed |
| characters, raw | +0.4241 [+0.2952, +0.5595] | -1.3027 [-1.5803, -1.0671] | +0.10 | confirmed |
| semantic, raw | +0.2775 [+0.1968, +0.3594] | -0.7208 [-0.8507, -0.5997] | -0.13 | confirmed |
| semantic, within-label whitening | +0.0215 [-0.0052, +0.0548] | -0.0706 [-0.0959, -0.0513] | +0.52 | noise only |
| word SVD-1024, within-label whitening | +0.1307 [+0.0458, +0.2254] | -0.2638 [-0.3444, -0.1916] | +0.30 | confirmed |

The noise term carries a negative association in all five spaces with every interval clear of
zero, and coherence a positive one in four; the exception is whitened semantic, where coherence
is flat. So the single scalar does decompose as the theory says - it was the profile norm, not the
theory, that mixed the two channels. This is a re-specification and is reported as one.

| space | scorer | query-weighted MRR | between-label variance of per-label MRR | false-hub mass | N_10 skewness |
|---|---|---|---|---|---|
| words, raw | prototype | 0.4963 | 0.07054 | 0.9329 | 1.821 |
| words, raw | znorm | 0.4859 | 0.04513 | 0.9304 | 1.349 |
| words, raw | csls | 0.5414 | 0.03966 | 0.9289 | 0.995 |
| words, raw | mutual_proximity | 0.5375 | 0.05609 | 0.9283 | 1.209 |
| characters, raw | prototype | 0.4266 | 0.05843 | 0.9394 | 2.145 |
| characters, raw | znorm | 0.3266 | 0.05305 | 0.9471 | 2.808 |
| characters, raw | csls | 0.4121 | 0.04782 | 0.9433 | 2.505 |
| characters, raw | mutual_proximity | 0.4257 | 0.05247 | 0.9377 | 2.092 |
| semantic, raw | prototype | 0.2997 | 0.02998 | 0.9509 | 0.836 |
| semantic, raw | znorm | 0.2105 | 0.04518 | 0.9607 | 1.374 |
| semantic, raw | csls | 0.2872 | 0.03796 | 0.9519 | 0.991 |
| semantic, raw | mutual_proximity | 0.2668 | 0.03978 | 0.9529 | 1.333 |
| semantic, within-label whitening | prototype | 0.4164 | 0.03272 | 0.9397 | 0.739 |
| semantic, within-label whitening | znorm | 0.4144 | 0.03272 | 0.9398 | -0.017 |
| semantic, within-label whitening | csls | 0.3741 | 0.01584 | 0.9446 | 0.794 |
| semantic, within-label whitening | mutual_proximity | 0.4157 | 0.03159 | 0.9399 | 0.412 |
| word SVD-1024, within-label whitening | prototype | 0.5426 | 0.05055 | 0.9247 | 0.834 |
| word SVD-1024, within-label whitening | znorm | 0.5577 | 0.04936 | 0.9234 | 0.267 |
| word SVD-1024, within-label whitening | csls | 0.5134 | 0.02759 | 0.9283 | 0.877 |
| word SVD-1024, within-label whitening | mutual_proximity | 0.5520 | 0.04418 | 0.9242 | -0.139 |

**A large finding, and not one this arm was built for.** CSLS - a standard de-hubbing correction
from cross-lingual retrieval - takes the raw word space from 0.4963 to **0.5414**, and mutual
proximity to 0.5375. That is +0.0451, two and a half times the attention gain, and it
is level with the whitened word SVD space's 0.5426, reached without fitting any transform on the
corpus at all. Both corrections also lower the between-label variance of per-label MRR and the
false-hub mass. So a substantial part of what within-label whitening buys in this corpus is
de-hubbing, which the repository already suspected from the label-free arm and can now show
directly. Neither correction reproduces Z-normalisation, which the same plan predicted they would:
in the raw word space Z-normalisation reaches only 0.4859.

**R3, fairness against accuracy.** A transform lowers query-weighted MRR while lowering both the
between-label variance and the false-hub mass in two of the five spaces (words under
Z-normalisation, characters under mutual proximity), so the estimand warning keeps its evidence,
but the split is small and does not appear in the whitened spaces.

## The profile-norm decomposition is algebra, and what is testable instead (`profile_norm_identity.json`, 2026-09-23)

`src/profile_norm_identity_v3.py`, with `tests/test_profile_norm_identity.py`. Every size claim in
this directory rests on E||m_l||^2 = r_l^2 + s^2 W_l. A draft analysis proposed to validate that
decomposition against the data. **It cannot be validated, because it is an identity.** With
r_hat^2 defined as the component-level U-statistic, the gap between the two sides has the closed
form

```
gap_l = [K_l(nu - 1) + sum_c (1 - 1/k_c)(chat_c - C_bar_l)] / (G_l(G_l - 1)), with nu the component-weighted mean squared row norm; a singleton component contributes exactly zero because 1 - 1/k_c = 0
```

and a singleton component contributes exactly zero, because 1 - 1/k_c = 0. This corpus is almost
all singletons: only 201 of its 6,889 components hold more than one song (2.92%), the
largest holds 10, and **142 of the 226 labels have no multi-song component at all**,
so for a clear majority of labels the gap is exactly zero by construction. Over the labels the
upper bound on |gap_l| has median 0.0000 and a 95th percentile of 0.0031 in the 20-49 band, against the 0.02 margin a test
would have declared. So this file reports no verdict on the decomposition and tests the four
questions that do have empirical content instead.

| space | pooling one s^2 | the published W = 1/G_l | the floor as the mechanism |
|---|---|---|---|
| words, raw | DEFENSIBLE | INADEQUATE | not established |
| characters, raw | DEFENSIBLE | INADEQUATE | not established |
| semantic, raw | DEFENSIBLE | INADEQUATE | not established |
| semantic, within-label whitening | DEFENSIBLE | INADEQUATE | not established |
| word SVD-1024, within-label whitening | DEFENSIBLE | INADEQUATE | not established |

**Pooling is defensible everywhere.** The interquartile ratio of a label's own s^2 to the pooled
value stays inside the declared [0.9, 1.1] in all five spaces, the widest being raw semantic at
[0.946, 1.072]. The published practice of estimating one s^2 per (space, fold) survives.

**The published W = 1/G_l is inadequate, and it does not matter much.** The weight-aware
W = sum_c(1/k_c)/G_l^2 is strictly smaller wherever a component holds several songs, and the third
quartile of the ratio is 1.0173 against a declared 1.01, so the published term over-subtracts.
Replacing it moves MRR by +0.0031 [+0.0006, +0.0056] in characters and +0.0011 [-0.0002, +0.0025]
in whitened word SVD, component-weighted - inside the 0.005 margin, so neither a gain nor a cost
is claimed. The correction is real and its consequence is not.

**The floor is a small-label boost, and it is not established as the mechanism of the published
change.** It is tempting to describe the published `noise_corrected` scorer as a cap rather than a
correction, and the measurement says that is too strong. The floor
`max(||m||^2 - s^2/n, 0.25||m||^2)` binds on whole label COLUMNS - 19 of them in raw words, 17 in
characters, 15 in whitened semantic, 5 in whitened word SVD and none at all in raw semantic - and
those columns carry 83 to 95 per cent of all floored pairs. Where a column is floored the score is
exactly twice the prototype, a constant per-label boost, and most of those labels are small (13 of
the 19 labels in the 5-9 band in raw words). Among the queries whose own label is floored the
scorer gains +0.2651 [+0.2176, +0.3136] in characters and +0.2474 [+0.1884, +0.3116] in whitened
word SVD, component-weighted - very large. But the overall change is a gain in the raw lexical
spaces (+0.0223 characters) and a COST in the whitened ones (-0.0167 whitened word SVD), so the
floor is not by itself the mechanism of the published number, and the pre-registered rule refuses
to say it is. The honest statement: the floor converts a bias correction into a fixed doubling for
a handful of small labels, that doubling is worth a quarter of a point of MRR to those labels'
queries, and whether it helps the corpus as a whole depends on the space.

Reproduction: every published s^2 and every published floor-hit count is reproduced exactly
(130,051 of 1,631,720 pairs in characters fold 1 against the published 130,051; 43,379 in whitened
word SVD fold 0 against 43,379), and every prototype and `noise_corrected` MRR, and all four band
MRRs per space, reproduce with a worst band gap of 0.00000.

## A calibrated profile scorer, and why no correction is uniformly best (`calibrated_profile_scorer.json`, 2026-09-23)

`src/calibrated_profile_scorer_v3.py`, with `tests/test_calibrated_profile_scorer.py`. The
prototype divides by ||m_l||, whose expectation carries the sampling inflation s^2 W_l; the
principled fix divides by an estimate of the label's true resultant length instead,
S_C = q.m_l / sqrt(r_tilde_l^2), with r_hat^2 the component-level U-statistic, empirical-Bayes
shrunk to a pooled prior and passed through a variance-stabilised positive part. Reading rules
were fixed before the run, with R4 (the pre-registered null) declared to OVERRIDE R1, and the
precedence implemented in the code rather than left to whoever writes the section.

| space | \|m\| band gradient, 20-49 minus 5-9, component-weighted | | | | |
|---|---|---|---|---|---|
| | prototype | exemplar_mean (= q.m, no norm) | znorm | noise_corrected | S_C |
| words, raw | +0.4332 | -0.1375 | -0.0623 | +0.1053 | **-0.1122** |
| characters, raw | +0.3394 | -0.1424 | -0.0193 | +0.0590 | **-0.1063** |
| semantic, raw | +0.1413 | -0.0620 | -0.1397 | -0.1505 | **-0.1046** |
| semantic, within-label whitening | +0.0736 | -0.1810 | +0.0640 | -0.0256 | **-0.0757** |
| word SVD-1024, within-label whitening | +0.2320 | -0.1477 | +0.2116 | -0.0248 | **-0.0098** |

**The size bias is large, and every calibration removes it - two of them by overshooting.** The
prototype's gradient is positive in every space and reaches +0.4332 in raw words: a label with 20
to 49 songs is answered far better than one with 5 to 9. S_C flattens it to between -0.11 and
-0.01, and the un-normalised dot (`exemplar_mean`) to between -0.06 and -0.18, both now favouring
SMALL labels. The flattest scorer in the lexical spaces is the simplest one in the literature: a
cohort Z-score, at -0.0193 in characters and -0.0623 in words.

| space | prototype | exemplar_mean | znorm | noise_corrected | S_C | label-macro penalty, prototype -> S_C |
|---|---|---|---|---|---|---|
| words, raw | 0.5020 | 0.3482 | 0.4840 | 0.5361 | 0.5097 | +0.0647 -> -0.0107 |
| characters, raw | 0.4310 | 0.2212 | 0.3237 | 0.4533 | 0.3975 | +0.0501 -> -0.0110 |
| semantic, raw | 0.2979 | 0.1050 | 0.2063 | 0.2611 | 0.2048 | +0.0166 -> -0.0206 |
| semantic, within-label whitening | 0.4136 | 0.3183 | 0.4112 | 0.3864 | 0.3803 | +0.0102 -> -0.0109 |
| word SVD-1024, within-label whitening | 0.5452 | 0.4240 | 0.5530 | 0.5285 | 0.5309 | +0.0305 -> +0.0034 |

All five MRRs are component-weighted, so they may be compared with the paired intervals.

**R1 is not met in any space, and R4 did not fire.** S_C is superior to the prototype only in raw
words; elsewhere it is not even non-inferior at the 0.005 margin. It does what it was built to do -
the label-macro penalty goes from +0.0647 to -0.0107 in raw words and from +0.0501 to -0.0110 in
characters, so the fairness gap between a random song and a random credited name essentially
closes - but it costs ranking accuracy, and against the published `noise_corrected` it is level in
whitened word SVD (+0.0024 [-0.0002, +0.0051]) and behind in the raw spaces.

The pre-registered null R4 - that small repertoires are under-determined rather than mis-scored -
did NOT fire: the share of own-label pairs with r_hat^2 <= 0 is 0.000 in the 5-9 band of every
space, and the mean empirical-Bayes shrinkage weight there is 0.695 in whitened word SVD. The
estimator's recovery simulation (in the test) shows r_hat^2 <= 0 in 47.9 per cent of labels at
n = 5 when r^2 = 0.03 in 1,024 dimensions, and unbiasedness at every n; the corpus's small labels
are better determined than that, because they hold at least five COMPONENTS and their r^2 is
larger. So the small band is read as mis-scored, and R1's verdict stands as the headline.

**What this section concludes.** The size bias is real, large, and removable, but no correction is
uniformly best: the principled estimator flattens the gradient furthest and loses MRR, the
published floor-capped scorer wins MRR in the raw lexical spaces and loses it in the whitened ones,
and a cohort Z-score lands closest to zero gradient while costing accuracy in three spaces. The
choice among them is a fairness choice about which estimand the study is answering, not a
statistical one, and the paper says so rather than naming a winner.

## The estimand 2x2, and a likelihood-ratio reading (`estimand_and_calibration.json`, 2026-09-23)

`src/estimand_and_calibration_v3.py`, with `tests/test_estimand_and_calibration.py`. Two gaps.

**Gap 1: the width factor is not a reusable number.** An earlier note here said the two-stage
intervals are 1.6 to 3.0 times as wide as the group-bootstrap ones and offered that as a
correction other leave-group-out studies could apply. Those two cells differ in BOTH the estimand
and the resampling, so the ratio conflated them. Filling the 2x2 over
14 contrasts: the total factor runs from 1.847 to
4.315 with median 2.936; the estimand factor alone runs
0.929 to 1.437, and the resampling factor
1.916 to 3.348. The share of the log total
attributable to the ESTIMAND has median 0.132 and can be negative
(-0.067), so roughly seven eighths of the widening is the
resampling and not the estimand. Neither factor is constant across contrasts. The number is
withdrawn as a reusable correction: another study must fill its own 2x2, and this one shows how.

**Gap 2: every metric here is a rank statistic, and the thesis is that scorers are estimators.**
A calibration that improved MRR while degrading the likelihood ratio would have been invisible. So
each system is also read as a detector: same-label against different-label trials, scored
leave-group-out exactly as retrieval is, with AUC, the cost of log-likelihood-ratio C_llr, and
min-C_llr after an optimal monotone (PAV) calibration, which separates discrimination from
calibration loss. The declared rule: a calibration counts as an improvement only where it lowers
min-C_llr as well as raising MRR, with both component-weighted intervals clear of the 0.005 band.

| space | calibration | dMRR, component-weighted | d(min-C_llr) | outcome |
|---|---|---|---|---|
| words, raw | znorm | -0.0180 | +0.0182 | no gain: MRR down and min_cllr up |
| words, raw | noise_corrected | +0.0340 | -0.0025 | a ranking gain with min_cllr unchanged within the declared band |
| characters, raw | znorm | -0.1073 | +0.0100 | no gain: MRR down and min_cllr up |
| characters, raw | noise_corrected | +0.0223 | +0.0041 | a ranking gain with min_cllr unchanged within the declared band |
| semantic, raw | znorm | -0.0916 | +0.0136 | no gain: MRR down and min_cllr up |
| semantic, raw | noise_corrected | -0.0367 | +0.0066 | no gain: MRR down and min_cllr up |
| semantic, within-label whitening | znorm | -0.0024 | +0.0008 | both unchanged within the declared band |
| semantic, within-label whitening | noise_corrected | -0.0272 | +0.0076 | no gain: MRR down and min_cllr up |
| word SVD-1024, within-label whitening | znorm | +0.0078 | -0.0069 | improvement: MRR up and min_cllr down |
| word SVD-1024, within-label whitening | noise_corrected | -0.0167 | -0.0011 | a ranking loss with min_cllr unchanged within the declared band |

**No calibration is an established improvement, and none is a ranking gain bought with a
calibration loss.** Z-normalisation raises min-C_llr - it is worse as a detector - in the raw
spaces where it also loses MRR, and improves both only in whitened word SVD. The published
`noise_corrected` scorer raises MRR in the two raw lexical spaces with min-C_llr unchanged inside
the band, and costs MRR in the whitened ones. The three estimands disagree about the verdict for
8 of the ten system-space pairs, which is the sharpest argument yet for stating the
estimand beside every number: the same calibration is an improvement under one weighting and not
under another.

## A local LLM re-ranking the protocol's retriever (`llm_rerank.json`, 2026-09-23)

`src/llm_rerank_v3.py`, with `tests/test_llm_rerank.py`. Zero-shot authorship attribution with an
instruction-tuned LLM is the current default in the literature, and its stated limitation is that it
does not scale to many candidate authors; this corpus has 226 credited labels. So the question is
asked in its scalable form: the protocol's own retriever - the raw jieba word TF-IDF prototype,
published MRR 0.4963, zero fitted parameters - proposes its top 10 labels, and a local
model chooses among them from example lyrics. The model is `Qwen/Qwen2.5-3B-Instruct` at revision
`aa8e72537993`, run offline in float16 on a NVIDIA GeForce RTX 5070 Laptop GPU with greedy decoding; every weight shard's
sha256 is in the payload. Candidates are shown as letters in an order shuffled per query, and no
artist name appears in any prompt, so the model cannot answer from what it knows about a rapper.
Each candidate is represented by 2 of its songs drawn at random from outside the query's leakage
group, truncated to 350 characters; the first prompt is 4,913 tokens. The subset is
1,000 queries drawn at random with seed 20260825; the true label is in the top 10 for
66.8% of them, which is the ceiling on what any re-ranking can reach. Rules were fixed in the
docstring before the run.

| | retriever | LLM re-ranked |
|---|---|---|
| MRR, query-weighted, subset | 0.4955 | 0.3251 |
| MRR, component-weighted, subset | 0.5008 | 0.3291 |
| recall@1 | 0.4090 | 0.0800 |
| top-1 correct among the 668 queries whose true label is offered | 0.6123 | 0.1198 |

Paired contrast, component-weighted: re-ranked minus retriever -0.1717 [-0.1891, -0.1541].
Abstentions (no candidate letter in the answer): main 3, reversed 2, blind 3.

**Controls, read before the result.**
C1, the same prompts with the candidates in reversed order, agree with the original choice for
20.3% of queries (floor 0.7). C2, the query replaced by a placeholder, is
correct for 6.9% against a chance level of 10% (ceiling 0.13). C3, the
memorisation probe - the query alone, the model asked to name the performer - matches the credited
label string for 1.8% of the subset.

**Reading.** R1 void: C1: agreement with the reversed order is 0.203, below 0.7: the model reads position

**The mechanism, in one table.** Which slot the model picks, per condition, against where the
true label actually sat (the shuffle puts it in every slot about equally, so a position reader
cannot do better than one in ten):

| slot | A | B | C | D | E | F | G | H | I | J | abstain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chosen, main | 396 | 49 | 21 | 26 | 4 | 156 | 5 | 2 | 2 | 336 | 3 |
| chosen, reversed | 413 | 40 | 17 | 32 | 5 | 150 | 5 | 0 | 2 | 334 | 2 |
| chosen, blind | 398 | 10 | 31 | 55 | 39 | 247 | 3 | 1 | 1 | 212 | 3 |
| true label's slot, main | 60 | 68 | 64 | 71 | 55 | 69 | 72 | 72 | 64 | 73 | - |

The model puts 888 of 1,000 choices on three slots (A, J, F: the first, the last and the middle of the list), and the blind
condition - no query at all - produces nearly the same histogram. That is a model reading the
shape of the prompt, not the lyrics, and it is why re-ranking with it costs the retriever a third
of its MRR. It is also the outcome the literature's own scaling caveat predicts for a model of
this size: the zero-shot results that motivate LLM attribution come from GPT-4-class models,
and a 3B model that must run locally so that no lyric leaves the machine is not one.

| band, songs per label | queries | retriever MRR | re-ranked MRR | retriever top-1 correct | LLM top-1 correct |
|---|---|---|---|---|---|
| 5-9 | 21 | 0.2568 | 0.1371 | 0.2381 | 0.0000 |
| 10-19 | 68 | 0.2579 | 0.1894 | 0.2059 | 0.0588 |
| 20-49 | 873 | 0.5238 | 0.3429 | 0.4318 | 0.0836 |
| 50-up | 38 | 0.4040 | 0.2623 | 0.3421 | 0.0789 |

Band MRRs are query-weighted over the subset's queries in that band and are description. The
prompt template, the candidate block and the probe template are recorded verbatim in the payload;
the prompts themselves and the model's answers stay in the private cache.

## Privacy

Aggregate numbers only. Per-query tables, vectors and text stay under the private root.
