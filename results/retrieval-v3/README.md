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

## PLDA scoring against whitened cosine (`plda_scoring.json`)

Whitened cosine is a heuristic reading of the two-covariance model; PLDA is its own
scorer, and cosine is PLDA with both covariances fixed to the identity. Fold-wise, same
folds and enrolment rule as the probe:

| scorer | MRR | R@1 | R@10 |
|---|---|---|---|
| cosine (reference) | 0.3015 | 0.2030 | 0.4947 |
| within-label whitening, cosine | **0.4196** | 0.3218 | 0.6071 |
| PLDA, simultaneously diagonalised | 0.3971 | 0.3047 | 0.5775 |
| PLDA, both covariances diagonal | 0.2986 | 0.2009 | 0.4933 |
| PLDA with a label-specific within scale | 0.2879 | 0.2108 | 0.4300 |

PLDA − cosine +0.096 [+0.088, +0.105]; PLDA − whitened cosine −0.021 [−0.026, −0.016];
diagonal PLDA − PLDA −0.100; label-scale PLDA − PLDA −0.110. The between-label signal
lives in about 167 of 1,024 whitened dimensions (ψ > 0.1). The model's own scorer does
not reach the heuristic, the independence assumption alone costs the whole gain, and the
smallest relaxation of the shared-scatter assumption — one scale per label, normalised to a
mean of one — makes it worse: what differs between labels is not how far their songs
scatter but in which directions, which no Gaussian with one shared shape represents. The
generative route ends here; the learned encoder is the next test.

## Contrastive fine-tuning, fold 0 (`identity_encoder_fold0.json`, `identity_encoder_analysis_fold0.json`)

BGE-M3 with LoRA (rank 16 on the attention projections, 7.1 M trainable parameters),
symmetric InfoNCE at temperature 0.05 over stanzas: positive = another stanza of the same
label from a different leakage group; negatives = the other seven labels in the batch plus
two content-controlled hard negatives per anchor (nearest other-label stanzas in the
frozen space); label strings masked from all text; collaboration-titled songs excluded
from training; two epochs, 3,774 steps, 4.6 h on the laptop GPU. Trained on four folds
and 192 labels; scored on fold 0 (1,205 songs) and on 34 labels never trained on (1,050
songs), against the frozen model on the same masked text.

| space, test fold | MRR | space, unseen labels | MRR |
|---|---|---|---|
| frozen, masked text | 0.2871 | frozen, masked text | 0.2480 |
| fine-tuned | 0.2698 | fine-tuned | 0.1952 |
| frozen + within-label whitening | 0.3839 | frozen + within-label whitening | 0.3400 |
| fine-tuned + within-label whitening | **0.4189** | fine-tuned + within-label whitening | 0.3434 |
| words | 0.4999 | words | 0.4268 |
| frozen + words (fusion) | 0.4843 | frozen + words | 0.4056 |
| fine-tuned + words (fusion) | 0.5022 | fine-tuned + words | 0.4111 |

Contrasts: fine-tuned − frozen −0.016 [−0.036, +0.004] on the test fold and −0.054
[−0.073, −0.036] on unseen labels; fine-tuned + whitening − frozen + whitening +0.036
[+0.015, +0.057] on the test fold (the whitening is fitted on the training folds, which the
tuned model has seen, so this is a bound) and +0.005 [−0.014, +0.025] on unseen labels;
fine-tuned + words − words +0.002 [−0.010, +0.013]. Chunk queries: 0.1775 vs 0.1800.
Songs that could be anchors and songs that could not lose alike.

Reading. The training loss fell from 3.66 to 2.65 against a chance level of about 3.2,
and what it learned shows only after whitening and only for the labels it was trained on:
the encoder moved those labels' identity into directions cosine does not read, learned
nothing that transfers to unseen labels, and adds nothing to the word space. The first
suspect is the batch: eight labels per step where LUAR saw 128 authors. The trainer now
takes a cross-batch memory of recent embeddings (`--queue-size`, XBM) so every step's
denominator holds thousands of stanzas across all labels; that run is the next test.

## Does 'not names' depend on NER recall? (`candidate_union_neutralisation.json`)

The published neutralisation strips the 605 reviewed surfaces. NER recall is unmeasured,
so the word space is also scored with everything either NER arm ever proposed for a
named-entity type stripped, whatever the gate decided, split by how many songs a surface
occurs in (a surface in more than 2% of songs is an ordinary word the transformer tagged
as a name, not a name):

| stripped from the word space | surfaces | share of text | MRR | vs words |
|---|---|---|---|---|
| nothing | — | — | 0.4977 | — |
| the reviewed catalogue | 605 | 2.0% | 0.4915 | −0.007 [−0.009, −0.005] |
| + every proposed named surface in ≤ 2% of songs (could be names) | 5,419 | 5.4% | 0.4775 | −0.020 [−0.023, −0.017] |
| only proposed named surfaces in > 2% of songs (ordinary words) | 514 | 14.2% | 0.4797 | −0.018 |
| every proposed named surface | 5,915 | 16.8% | 0.4608 | −0.037 |
| + every proposed rap-culture term (flow, beat, hook, ...) | 8,132 | 17.5% | 0.4600 | −0.038; terms alone −0.001 [−0.003, +0.001] |

Reading. Removing every name-shaped surface the NER ever considered — nine times the
catalogue — costs two points of fifty; the catalogue already holds a third of that. The
trade vocabulary of rap adds nothing. What remains open is only what neither NER arm ever
saw, and by the frequency-band anatomy that lives in the rare quartile, which carries
nothing. The finding does not rest on NER recall.

## One language model per label (`ngram_language_model_attribution.json`)

The generative counterpart of the TF-IDF spaces: an interpolated character n-gram model
per label (orders 1–5, fixed interpolation weights, add-0.5 unigram floor, n-grams seen
fewer than twice not modelled; 1.1 M five-grams), the query scored by its mean
log-probability per character with its own leakage group subtracted from every label it
touches.

| system | MRR | R@1 | R@10 |
|---|---|---|---|
| n-gram LM, orders 1–3 | 0.3805 | — | — |
| n-gram LM, orders 1–5 | 0.3952 | 0.3000 | 0.5781 |
| character 2–5-gram TF-IDF | 0.4275 | | |
| jieba words TF-IDF | 0.4977 | | |

Orders 1–5 − character TF-IDF −0.032 [−0.039, −0.024]; − words −0.103 [−0.111, −0.095].
Reading. Scoring a song by how well the label's own character model predicts it does not
reach the discriminative character space and sits ten points under the word space; the
interpolation weights are fixed, not tuned, so this is a floor for the model class rather
than its ceiling. The word space's lead is not an artefact of TF-IDF weighting.

## A purpose-built style embedding as a fourth space (`style_embedding_space.json`)

mStyleDistance (Patel et al. 2025): xlm-roberta-base trained by triplet loss on GPT-4-
written sentence pairs that keep the content and change one of about forty style features,
in nine languages including Simplified Chinese; never trained on lyrics or on this corpus.
Chunks embedded with the pinned release, song = mean of chunk vectors, protocol unchanged.

| space | MRR | R@1 | R@10 |
|---|---|---|---|
| style, cosine | 0.0541 | 0.0152 | 0.1112 |
| style, within-label whitening | 0.2088 | 0.1268 | 0.3706 |
| semantic (BGE-M3), cosine | 0.3015 | 0.2030 | 0.4947 |
| semantic, within-label whitening | 0.4196 | 0.3218 | 0.6071 |
| words | 0.4977 | 0.4048 | 0.6738 |
| whitened style + words (fusion) | 0.4147 | | |
| whitened semantic + words (fusion) | 0.5386 | | |
| whitened style + whitened semantic + words | 0.5186 | | |

Style − semantic (cosine) −0.246; whitened style − whitened semantic −0.210 [−0.221,
−0.199]; words − whitened style +0.297; adding the style space to the semantic+words
fusion −0.022 [−0.028, −0.017].

What could make this wrong: the model does cover Chinese, so language is not the reason;
its own authorship-verification scores in the paper are modest (ROC-AUC 0.60–0.73 on PAN
languages), so it is a weak authorship signal by design; and its forty features are
generic sentence-level dimensions (formality, emoji, contractions, ...) elicited from
GPT-4 paraphrases, not lyric style. Reading, within those limits: the generic style
dimensions a content-independent embedding encodes are not what identifies a rapper;
even whitened, the space holds half the identity the semantic space does and adds
nothing to the word space. What identifies a rapper is specific word usage, not style in
the sense such embeddings measure.

## Regional variety inside the common-word signal (`dialect_marker_identity.json`)

A hand-compiled catalogue of function words, particles and pronouns for seven varieties
(Cantonese, Southwestern Mandarin, Northeastern, Beijing, Wu, Min/Taiwan, Xiang; the list
is published in the JSON) is applied to the text only. Marker tokens are 0.55% of all word
tokens; 28 of 226 labels lean to a variety by the rule in the file (16 Southwestern
Mandarin, 6 Cantonese, 3 Northeastern, 3 Beijing).

| | MRR |
|---|---|
| word space | 0.4977 |
| word space, every marker token removed | 0.4972 (−0.0003 [−0.0018, +0.0012]) |
| markers alone | undefined (most labels have none) |

Among the word space's wrong top-1 answers whose true label leans to a variety, the wrong
label leans to the same variety 26.5% of the time against 4.9% if wrong labels were drawn
at random (×5.5); with every marker token removed from the space it is still 24.8%
against 4.8% (×5.1).

Per label (markers removed): 18 of the 27 leaning labels with at least five wrong answers
sit above the random share; leaving any one label out moves the enrichment only between
4.8× and 5.3×, so no single label carries it. But one variety does: the sixteen
Southwestern Mandarin labels are confused with each other 27–71% of the time in most
cases, while the Cantonese (6), Northeastern (3) and Beijing (3) labels are mostly at or
near zero — too few labels of each to say anything about them.

Reading. The marker words themselves carry no measurable identity, yet the space confuses
Southwestern Mandarin rappers with each other far above chance even when those words are
gone: the regional variety is carried by the broad distribution of ordinary words, not by
a shortlist of dialect words. Part of the common-word identity is therefore a shared
language variety rather than the individual, established for the largest regional scene
in the corpus and open for the others; the lean is derived from the text alone, never
from biography, and the catalogue is a first cut.

## Identity among content rivals (`content_rival_test.json`)

The retrieval version of Wegmann et al.'s style-or-content choice. For each query the
frozen semantic space names the K labels whose leave-group-out profiles are closest to
it in content (the true label excluded); every space then ranks the true label within
that set. The rival set is defined once, from the frozen space, so it is the same for all
spaces — which also means the frozen space's own row is not a test (its rank within the
set is a function of its full rank) and is shown only as the floor.

| space | all 226 labels | among 5 rivals (chance 0.408) | among 10 (0.275) | among 25 (0.148) |
|---|---|---|---|---|
| frozen semantic (defines the rivals) | 0.3015 | 0.3730 | 0.3311 | 0.3086 |
| semantic, within-label whitening | 0.4196 | 0.5881 | 0.5316 | 0.4783 |
| character 2–5-grams | 0.4275 | 0.6328 | 0.5606 | 0.4913 |
| jieba words | **0.4977** | **0.6776** | **0.6149** | 0.5546 |
| whitened semantic + words (fusion) | 0.5386 | 0.6557 | 0.6098 | **0.5692** |

Top-1 among 5 rivals: words 53%, characters 47%, whitened semantic 41%. Contrasts among
5 rivals: words − whitened semantic +0.097 [+0.088, +0.106]; words − characters +0.047;
fusion − words −0.025 [−0.032, −0.019]; among 25 rivals fusion − words +0.011 [+0.004,
+0.019].

Reading. With subject held roughly constant, the word space still names the label first
more than half the time; the whitened semantic space keeps most of its identity too but
less. The fusion that wins over all labels loses to the words alone when every candidate
is a content rival and wins again as the candidates diversify: part of what the semantic
component contributes to identity is content.

## Training-data audit for the identity encoder (`training_data_audit.json`)

Tokens per chunk median 76, p90 603, 3,676 over 512; all 226 labels have at least two
leakage groups; 593 songs (8.2%) carry a collaboration title; 1,636 chunks (6.7%) name their
own label; a chunk's nearest content rival (another label) is closer than its nearest
same-label chunk in 90.8% of cases (median cosine 0.725 vs 0.663).

## Privacy

Aggregate numbers only. Per-query tables, vectors and text stay under the private root.
