# Where Does a Lyrical Identity Live? Semantic, Lexical, and Written-Rhyme Evidence from Chinese Rap Lyrics

[Author name(s) — to be completed]

[Affiliation(s) — to be completed]

[Corresponding author email — to be completed]

## Structured Abstract

**Purpose:** This study asks where, in the text of a song, the identity of its source-credit label lives: in what the lyrics mean, in the characters they are written with, or in how their lines rhyme. It then asks whether a multilingual semantic embedding has discarded that identity or merely hidden it.

**Design/methodology/approach:** One held-out-song retrieval protocol scores the same 7,236 Chinese rap songs and 226 labels in three representation spaces: frozen BGE-M3 dense vectors, character 2–5-gram term frequency–inverse document frequency (TF–IDF), and a rhyme-form space built from written line endings that carries no lexical content. Leakage groups join exact shared text with near-duplicate songs; every label profile is leave-group-out; differences carry paired bootstrap intervals. The lexical advantage is decomposed by script class and by four named-reference neutralisation arms. A linear probe from speaker recognition, within-author covariance whitening, tests whether the semantic space holds identity information that cosine similarity cannot reach.

**Findings:** Character n-grams identify the label far better than the semantic model (macro mean reciprocal rank 0.398 versus 0.301; fusion 0.427). The rhyme form carries identity at 0.097 against a chance level of 0.027, but nothing the surface does not already hold. The lexical advantage is not code-switching and not names: removing a 605-entry catalogue of places, people, brands, slang, and English words leaves it at 0.444. Whitening lifts the semantic system from 0.318 to 0.449 and its fusion with the surface to 0.544. Almost all of that gain needs no labels — it is the anisotropy of the space — and the same correction lifts a 1,024-dimensional lexical space from 0.393 to 0.518, so the surface stays ahead under equal treatment; the two corrected spaces together reach 0.587. The label-fitted part of the correction is small in both spaces and does not transfer to unseen authors; the label-free part transfers in full.

**Originality:** The study replaces "which model wins" with "where the signal is", and shows that the standard reading of any embedding, cosine on raw vectors, underestimates what the space knows — for the neural model most of all.

**Contribution to the field of Digital Humanities:** It supplies a reusable protocol for asking a representational question of a culturally situated corpus without publishing copyrighted lyrics, and a warning that an embedding map is a measurement of the map's geometry before it is a measurement of the texts.

Keywords: Chinese rap; digital humanities; lyrics; authorship; text embeddings; BGE-M3; anisotropy; character n-grams; rhyme; leakage control

## 1. Introduction

A collection of lyrics carries, alongside what each song says, a great deal about who is saying it. Rappers return to their own subjects, but they also return to their own words, their own spellings, their own ways of closing a line. A computational study can ask which of these layers a listener-independent representation preserves, and the answer depends on the representation. A semantic embedding is trained to bring paraphrases together, and so to treat two songs about the same thing as alike whoever wrote them. A bag of character n-grams has no notion of meaning and keeps every idiosyncrasy of surface form. A rhyme scheme keeps neither the meaning nor the words.

This article asks one question of one corpus: where does a source-credit label's identity live in Chinese rap lyrics? "Identity" is used narrowly throughout. It means the corpus-relative, aggregate profile attached to a credit string, not a verified person, biography, hometown, belief, affiliation, influence, or collaboration. A label may be an alias, a group, or a metadata error, and nothing in this study resolves it further. What the study does establish is which kind of text evidence lets a held-out song be matched to the rest of its label's repertoire, and how much of that evidence each representation can reach.

The design is one protocol applied to three spaces. Every song is one vector in each space; every label is a profile built from its other songs with the held-out song's whole leakage group removed; every score is a cosine and every result is a rank of the true label among 226 candidates. Because the queries, labels, groups, and metrics are identical across spaces, a difference between spaces is a difference in representation and nothing else. This is what lets a rhyme scheme be compared with a language model on one scale.

Three results follow. First, the character surface carries far more identity than the semantic content: a character n-gram system outranks frozen BGE-M3 on 3,816 of 7,129 queries and loses on 2,132, while the rhyme form carries a small but real share and adds nothing beyond the surface. Second, the surface advantage lives nowhere convenient. It is not code-switching — the songs with the most Latin script are the ones where the semantic model does best — and it is not names: stripping every place, person, brand, slang term, and English word in a 605-entry catalogue, 168,004 characters in all, moves the lexical system from 0.450 to 0.444. Third, and against the working hypothesis of the study, the semantic model has not discarded the identity signal. A linear correction of its geometry, fitted on other songs, raises it to the level of the raw character surface, and the two spaces then fuse to a level neither reaches alone. Almost all of that correction needs no labels, and the same correction lifts the character surface too, so that under equal treatment the surface keeps its lead; the label-fitted remainder is small and does not transfer to authors the correction never saw, while the label-free part transfers in full.

The research questions are:

**RQ1.** Under one leave-group-out protocol, how much source-label identity does each of the semantic, lexical, and rhyme-form spaces carry, and are they complementary?

**RQ2.** Within the lexical space, is the advantage over the semantic model carried by script mixing or by named references, or is it diffuse in the character surface?

**RQ3.** Is the identity information absent from the semantic embedding, or present but misaligned with cosine similarity — and if the latter, how much of the recovery is generic geometry, how much is author-specific, and does the lexical space gain as much from the same correction?

**RQ4.** How can such a representational question be asked of a copyrighted, situated corpus while keeping every claim at the level of the label string, and what does the answer imply for the use of embedding maps in digital humanities?

## 2. Related Work and Analytical Position

### 2.1 Chinese Rap as Situated Repertoire

Ethnographic and lyric-centred studies show Chinese performers reworking global hip-hop through regional language, Putonghua, English, slang, local knowledge, and place (Barrett 2012; Wang 2013; Liu 2014). Work on Higher Brothers separates the written multilingual material from the vocal intelligibility and production through which identities are negotiated (Liu 2021). Named places and objects take part in stylised persona construction, but only together with stance and performance context (Baioud 2024). These studies motivate *lyrical repertoire* as situated corpus evidence and warn against reading any text-derived profile as an authenticity score or a fixed identity. They also motivate the second research question: if a label is recognisable, one wants to know whether it is recognisable by its places and references or by something less nameable.

### 2.2 Style and Content in Authorship

Authorship attribution has long known that surface features carry authorial signal, and that character n-grams are among the most robust of them (Stamatatos 2009; Sapkota *et al.* 2015). It has also known that attribution accuracy does not by itself separate style from content: common and proper nouns drive a good part of what attribution systems learn (Sundararajan and Woodard 2018), and neural authorship representations can succeed while capturing little that a stylometrician would call style (Wang *et al.* 2023). The present study takes both lessons as design constraints. It does not call any space "style"; it decomposes what each space uses, and it treats named references as a confound to be removed rather than as evidence of identity.

### 2.3 The Geometry of Sentence Embeddings

Contextual and sentence embeddings are anisotropic: their vectors occupy a narrow cone in which a few directions of large variance dominate every cosine similarity (Ethayarajh 2019; Timkey and van Schijndel 2021). Unsupervised corrections — flow-based (Li *et al.* 2020) or a plain whitening (Su *et al.* 2021) — improve semantic similarity tasks substantially without changing the model. Speaker recognition reached the same insight earlier and in supervised form: within-class covariance normalisation whitens an utterance embedding by the covariance of utterances *within* a speaker, shrinking the directions that vary from one recording to the next of the same person and keeping the ones that stay put (Hatch, Kajarekar and Stolcke 2006), a step that sits inside every i-vector back-end since (Dehak *et al.* 2011). The third research question applies this move to lyrics: within-author variation is what a song is about; whitening by it should expose who wrote it, if that information is there.

### 2.4 Named References and Cultural Networks

Chinese named-entity recognition (NER) is domain- and annotation-sensitive, and edited-text models cannot be assumed to transfer to informal language (Peng and Dredze 2015); dictionary matches are noisy candidates, not gold (Yang *et al.* 2018), and domain corpora require explicit annotation-quality reporting (Jiang *et al.* 2022). Unequal extraction error can become unequal cultural visibility (Lassen *et al.* 2024), and co-occurrence differs from embedding similarity as network evidence (Blessing *et al.* 2017; Rafaeli *et al.* 2026). The reference layer in this study is therefore released as description with its accuracy withheld, and its main analytical use is as the lexicon that the neutralisation arms remove.

### 2.5 Rhyme as Text-Derived Evidence

Computational work models phonetic, imperfect, internal, and artist-level rhyme (Hirjee and Brown 2009), and Chinese rap generation systems represent rhyme through vowel classes and decoding constraints (Xue *et al.* 2021; Sun *et al.* 2023). Sociolinguistic work on Chinese hip hop shows that rhyming style is part of persona and that imperfect rhyme is systematic rather than sloppy (Lin and Wang 2024; Wang and Lin 2024), while performed tone and flow exceed what a transcript can show (Liu *et al.* 2023). The rhyme-form space here is deliberately narrow: dictionary-estimated written endings, no audio, no performed pronunciation. Its role is to ask how much identity survives when every word is thrown away.

## 3. Corpus, Leakage Groups, and Evidence Boundary

The analysed corpus is a frozen private snapshot of 7,391 song records in 25,026 lyric chunks under 241 source-credit labels. It is the second corpus version of the project. The first applied a legacy cleaner that deleted chunks whose text recurred under more than one label; amendment PD-002 replaced deletion with representation, so that every chunk is retained and the duplicate control is carried by grouping. Song records, chunk order, and cleaned text are hashed as a content contract, and every builder refuses a corpus whose digest differs.

The leakage unit is the *leakage group*. Two songs share a group when they share a corpus text component — a connected set of chunks with byte-identical cleaned text — or when their whole-song trigram sets have Jaccard similarity of at least 0.80 after NFKC normalisation, case folding, and removal of punctuation and spaces. Neither rule contains the other: 404 of 799 multi-song text components would be split by the document rule alone, and 44 near-duplicate pairs share no text component, so the union is used and both halves are scored alone as ablations. Over the 7,236 query songs there are 5,888 groups, 798 of which hold more than one song, the largest 27. No text component is split across groups. Because shared text is how collaboration appears in this corpus, groups routinely span labels; a held-out group is removed from every label profile it touches.

The labels themselves were audited for the one failure that would corrupt every result: one artist filed under two strings. String similarity finds nothing usable — no pair is identical after normalisation, and every substring candidate is a stage-name particle. Shared text is the better signal. Among 511 label pairs whose songs share a passage of at least thirty characters, the highest mutual involvement is 48.5% and no pair exceeds 50%; one artist filed twice would show near-total mutual overlap. The 45 pairs at or above 10% were put to the author, who determined that each is a collaboration. No label merging is applied, and this determination is recorded as an input, not a computed result.

BGE-M3 vectors come from a single recorded run over corpus v2 on a Tesla T4 in half precision, batch size 8, maximum length 2,048, corpus order, with the checkpoint weights and the corpus digest pinned in the run's contract; a resume under any other device, precision, or corpus is refused. The earlier corpus's vectors had been produced under an unrecorded device and precision. A configuration probe found that three candidate configurations all reproduce those archived vectors within a cosine tolerance of 2.2 × 10⁻⁵, which bounds the concern without identifying the configuration; re-embedding in one recorded run removes the question rather than answering it.

The public release carries aggregate results only. It excludes lyric text, lines, song and chunk identifiers, per-song ranks and coordinates, embeddings, membership rows, and reviewer contexts. Every per-query table named below is private. Table 1 gives the populations.

| Population | Count | Role |
| --- | ---: | --- |
| Song records | 7,391 | Frozen song grain (corpus v2) |
| Lyric chunks | 25,026 | Retained in full; duplicate control by grouping |
| Source-credit labels | 241 | Corpus provenance strings |
| Labels with at least five songs | 226 | Candidate set in every space |
| Query songs (≥50 normalised alphanumeric characters) | 7,236 | The one query set |
| Leakage groups over the query set | 5,888 | Holdout and bootstrap unit; 798 hold >1 song |
| Label pairs sharing a ≥30-character passage | 511 | Identity audit; all reviewed pairs are collaborations |

Table 1 Frozen corpus and comparison populations. Every space in Sections 5–9 is scored on the same 7,236 queries, 226 labels, and 5,888 groups.

## 4. One Protocol

### 4.1 Query, Profile, and Rank

A query is a whole song. In each space it is one vector: the mean of its chunk vectors for BGE-M3, its TF–IDF row for the character space, and its TF–IDF row over rhyme tokens for the rhyme-form space. Using the mean of a song's chunks was itself tested rather than assumed. Within corpus v2 the median pair of chunks inside one song has cosine 0.572 and 58.8% of songs average below 0.6 internally, so the mean summarises things that are not alike; but scoring a song by its best-matching chunk against the same profiles, the standard passage-retrieval move, lowers MRR from 0.318 to 0.280. The song is the unit at which the estimand is defined, and the song-level vector serves it best.

A label profile is the weighted sum of its songs' vectors, normalised to unit length. Every leakage group contributes total weight one within a label, so a song filed five times counts once. For each query, the query's whole group is removed from every label the group touches before that label's profile is normalised; every other profile is the full one. The query's cosine against all 226 profiles is ranked, ties are broken against the true label, and the rank of the true label yields reciprocal rank, Recall@1/5/10, and nDCG@10. The relevance definition is deliberately narrow: a correct retrieval means the held-out song is consistent with the remaining text of its label, not that it is semantically similar by human judgement and not that anyone wrote it.

### 4.2 Two Systems Fused

Where two spaces are fused, each system's 226 candidate scores are standardised within the query and averaged with equal weight, fixed before any outcome was seen:

\[
z_{q\ell}^{D} = \frac{s_{q\ell}^{D}-\mu_q^{D}}{\sigma_q^{D}}, \qquad
z_{q\ell}^{L} = \frac{s_{q\ell}^{L}-\mu_q^{L}}{\sigma_q^{L}},
\]

\[
s_{q\ell}^{F}=\tfrac{1}{2}z_{q\ell}^{D}+\tfrac{1}{2}z_{q\ell}^{L}.
\]

Fusion weights are never tuned, so a fusion that falls below its better component is a finding about complementarity, not a failure of tuning.

### 4.3 Estimands and Uncertainty

Two averages of the same ranks are reported and named. The *macro* estimand is the retrieval builder's headline: the component-weighted mean within each label, then the plain mean over labels, so that a label with 300 songs cannot decide the conclusion for 225 others. Its 95% intervals come from a paired two-stage bootstrap of 5,000 fixed-seed replicates that draws labels with replacement and then leakage groups within each drawn label, applying one draw to every system so that system differences stay paired (Savoy 1997). The *plain* estimand is the mean over queries, the number a reader can check against a head-to-head count; it is used for the decomposition and probe experiments, whose contrasts carry 95% intervals from 2,000 paired resamples of leakage groups with the builder's per-(group, label) weighting. A published reconciliation table gives every system under both, and the ordering of spaces is the same under either.

Fig. 1 summarises the design.

[[FIGURE:figure_1_research_design.png]]

Fig. 1 Study design. One frozen corpus enters one protocol — leave-group-out label profiles over exact-text ∪ near-duplicate leakage groups, one query set, no test-outcome tuning — and is scored in three representation spaces. The semantic branch adds the whitening probe and its controls; the lexical branch adds the script-class decomposition and named-reference neutralisation; the rhyme branch adds the next-ending context model. Every branch answers the same question on the same scale.

Alt text: A top-to-bottom diagram. A corpus box gives 7,391 song records, 25,026 chunks, and 7,236 query songs across 226 labels. A dark controls box names the shared protocol and the leakage groups. Three coloured branches follow: semantic space with the whitening probe, lexical space with decomposition and neutralisation, and rhyme form with a next-ending model; each ends in a label ranking.

## 5. Three Spaces

### 5.1 Semantic

BGE-M3 is used through its dense head only, frozen, with no fine-tuning on the corpus and no import of its published benchmarks as evidence (Chen *et al.* 2024). It is the natural choice for a primarily Chinese corpus with mixed-script passages of varying length, and its training objective — retrieval of semantically related passages across languages — is exactly the property in question: a model built to find what a passage means has been given no reason to keep who wrote it.

### 5.2 Lexical

The character space is TF–IDF over character 2–5-grams of the whole song text, with document frequency at least three, sublinear term frequency, L2 rows, and 150,000 features: 94,714 Han, 48,319 Latin, 5,003 mixed-script, and 1,964 other. Character n-grams commit to no word segmentation and keep partial strings, spelling, mixed script, and punctuation-normalised surface form. The vocabulary and inverse document frequencies are estimated on the fixed unlabelled query corpus; labels never enter them, but query text does, and the estimand is fixed-corpus leave-group-out retrieval rather than deployment to a future corpus (amendment PD-001).

### 5.3 Rhyme Form

The rhyme-form space keeps no word. Every line that ends in a Han character and has a classifiable written final — the written-rhyme task's own line filter — is reduced to tokens: the rhyme family of its last syllable (seventeen tone-free classes of the strict `pypinyin` final), the tone of that syllable, the family pair across its last two syllables, the transition from the previous line's family, whether that transition continued or shifted, and one token per maximal run of same-family endings. A song is a bag of these tokens, weighted exactly as the lexical system weights its n-grams. A second tier adds the raw pinyin final, toned final, and final pair, which are closer to the syllable itself and so leak a little of which words end lines. The strict tier has 605 features and the second 1,997. Songs with fewer than four classifiable endings — 107 of 7,236, of which 72 have none — receive the worst rank in any system that uses this space and are excluded from the covered subset on which the spaces are compared; the median song has 51 endings.

### 5.4 Results

Table 2 gives the headline macro estimates for the two full-coverage spaces and their fusion (Fig. 2). Character n-grams outrank BGE-M3 on every metric, and fusion improves on both: macro MRR 0.427 (95% CI 0.396–0.458) against 0.398 (0.366–0.432) for TF–IDF and 0.301 (0.277–0.326) for the dense system. The paired fusion advantage over TF–IDF is +0.029 MRR (95% CI +0.019 to +0.039) and +0.037 Recall@10 (+0.024 to +0.051); over the dense system, +0.125 MRR (+0.106 to +0.145). TF–IDF's own margin over the dense system is +0.097 MRR (+0.073 to +0.121). Every fusion-versus-component interval on every metric is above zero.

| System | MRR | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 dense | 0.301 [0.277, 0.326] | 0.211 [0.187, 0.237] | 0.381 [0.353, 0.411] | 0.477 [0.449, 0.508] | 0.331 [0.307, 0.357] |
| Character 2–5-gram TF–IDF | 0.398 [0.366, 0.432] | 0.315 [0.283, 0.349] | 0.478 [0.441, 0.516] | 0.555 [0.517, 0.593] | 0.426 [0.393, 0.461] |
| Equal-weight z-score fusion | **0.427 [0.396, 0.458]** | **0.341 [0.310, 0.374]** | **0.513 [0.478, 0.547]** | **0.592 [0.558, 0.627]** | **0.457 [0.426, 0.489]** |

Table 2 Macro (source-label) retrieval metrics over 7,236 held-out songs and 226 labels under the leave-group-out protocol. Brackets are paired two-stage bootstrap 95% intervals (5,000 replicates, 6,902 within-label group units).

[[FIGURE:figure_2_retrieval_benchmark.png]]

Fig. 2 Held-out-song source-label retrieval in the semantic and lexical spaces. Panel A shows macro MRR, Recall@1/5/10, and nDCG@10 for BGE-M3, character TF–IDF, and their untuned per-query z-score fusion. Panel B shows paired fusion-minus-component differences. Whiskers are 95% intervals from 5,000 paired two-stage bootstrap replicates over 7,236 queries and 226 labels.

Alt text: Two dot-and-whisker panels compare five retrieval metrics. In the model panel, fusion is highest on every metric, character TF–IDF is second, and BGE-M3 is lowest. In the paired-difference panel every fusion-minus-component interval lies to the right of zero; gains over TF–IDF are smaller than gains over BGE-M3.

Table 3 places the rhyme-form space on the same scale, using the plain estimand on the 7,129 covered queries where every system is defined. Chance MRR over 226 candidates is 0.027; a frequency prior that ranks labels by how many groups they own reaches 0.019, below chance because the held-out group is subtracted from the true label alone. The strict rhyme-form space reaches 0.098 (Recall@10 0.196), +0.079 over the prior with interval [+0.074, +0.084]; adding raw finals lifts it to 0.114 (+0.017, [+0.014, +0.020]), so most of what the space carries is scheme rather than syllable. It is real and it is small: the lexical space beats it by 0.350 and the semantic space by 0.214. Nor does it add: equal-weight fusion with the lexical space falls from 0.447 to 0.391 (−0.056, [−0.062, −0.050]) and the three-way fusion trails dense-plus-lexical by 0.031 ([−0.037, −0.026]). Whatever the rhyme scheme knows about a label, the character surface already knows.

| Space | MRR | Recall@1 | Recall@10 |
| --- | ---: | ---: | ---: |
| Chance | 0.027 | 0.004 | 0.044 |
| Label-frequency prior | 0.019 | 0.000 | 0.014 |
| Rhyme form, strict | 0.098 | 0.041 | 0.196 |
| Rhyme form, with raw finals | 0.114 | 0.051 | 0.230 |
| Semantic (BGE-M3) | 0.313 | 0.214 | 0.510 |
| Lexical (character n-grams) | 0.447 | 0.353 | 0.627 |
| Fusion: semantic + rhyme | 0.270 | 0.171 | 0.467 |
| Fusion: lexical + rhyme | 0.391 | 0.295 | 0.577 |
| Fusion: semantic + lexical | 0.472 | 0.376 | 0.658 |
| Fusion: all three | 0.440 | 0.339 | 0.641 |

Table 3 Plain (per-query) metrics on the 7,129 covered queries, identical profiles and groups in every space. The rhyme form carries identity at nearly four times chance and adds nothing to the surface.

RQ1 is thus answered in order: lexical, then semantic, then rhyme form, with semantic and lexical complementary and rhyme redundant. The next three sections ask why, and Section 9 overturns the obvious reading of the first two.

## 6. Named References as Description and as Control

The lexical advantage could be the cheapest thing imaginable: rappers name their cities, their crews, and themselves. To test that, the study needs a catalogue of named references, and the project's cultural-reference layer supplies one. Its evidence status is stated here because the same numbers are also released as description.

The layer's schema types person, group or crew, place, brand, work, event, language or dialect, ethnocultural group, rap-culture concept, and other references, plus non-entity and uncertain. Two fallible candidate sources are compared: a screened 605-surface lexicon matched literally, and the pinned `ckiplab/albert-tiny-chinese-ner` token classifier run in overlapping windows (Wolf *et al.* 2020; Paszke *et al.* 2019). Exact span-and-type agreement occurs 3,566 times on a 23,177-line comparison frame. Agreement between two fallible methods is not accuracy: no occurrence-level human gold exists, an 800-occurrence dual-review package and a 157-occurrence claim audit remain unreviewed, and no precision, recall, or F1 is reported.

On the earlier corpus and a fixed 204-label universe, cross-method agreement, a release audit, and shared-text exclusion leave 22 provisional surfaces — 18 places, two language references, two named persons. Six source-label-to-place associations survive a Jeffreys-smoothed risk-ratio bound above one and Benjamini–Hochberg false-discovery control at *q* ≤ 0.05, and four same-song co-mentions survive a positive normalised pointwise mutual information and the same control over all 5,681 eligible song units (Fig. 3). For example, 杭州 appears in 10 of 38 eligible song units labelled Tangoz, a risk ratio of 74 against the rest of the universe. Every such edge is a concentrated lyric reference, not residence, preference, biography, or a social relation, and ambiguous surfaces such as 中南海 and 上帝 are withheld until occurrence review resolves their senses.

[[FIGURE:figure_3_cultural_reference_evidence.png]]

Fig. 3 Provisional cultural-reference evidence after cross-label shared-text exclusion and statistical screening, computed on the earlier corpus snapshot. Panel A keeps three analytical units separate: the candidate inventory falls from 33 corpus-wide entity strings to 23 after shared-text exclusion and to 22 in the fixed 204-label universe; legacy source-label/entity links fall from 85 to 40 and then to six after shrinkage, a conservative interval, and false-discovery control; the legacy co-mention denominator (9 before exclusion, 1 after) is shown separately from the corrected all-song denominator (5 basic-gate candidates; 4 releases). Panel B plots the six released shrunken risk ratios with conservative 95% intervals and adjusted *q* values. Panel C plots the four released same-song co-mentions by normalised pointwise mutual information. Human review is incomplete, so precision, recall, and F1 are not reported.

Alt text: Three evidence-control cards show separate progressions for entity strings, source-label/entity pairs, and co-mention pairs. The entity inventory decreases from 33 to 23 to 22. Label-to-reference links decrease from 85 to 40 to 6. The co-mention card separates the legacy denominator, which changes from 9 to 1, from the corrected denominator, which changes from 5 to 4. A log-scale forest plot shows six released source-label-to-place enrichments, every interval above one. A dot-bar panel shows four released same-song co-mentions.

For the present question the layer's value is as a lexicon. Three neutralisation arms are cut from it: place names (212 surfaces), all named entities (459), and the whole catalogue including curated hip-hop slang, English words, and language names (605); a fourth arm uses the 90-entity reference core. Each arm strips its surfaces from every song — Latin surfaces as whole words, Han surfaces as substrings — refits the lexical space, and rescores it against the unchanged semantic space.

## 7. Where in the Characters

### 7.1 Not Code-Switching

The first hypothesis was that the surface advantage is code-switching: Latin script is 33.7% of the corpus's characters, 74.5% of chunks contain some, and a semantic model trained mostly on monolingual text might mishandle it. The data say otherwise. Decomposing each query's lexical score for its correct label by the script class of the contributing n-grams, Han n-grams supply 48.7% and Latin 49.2%, against corpus-wide TF–IDF mass of 56.3% and 41.1%; Latin features are over-represented in the identity signal, but the queries on which the lexical system beats the semantic one are, if anything, *more* Han (50.0% Han) than those on which it loses (47.9%). By quartile of a song's Latin share, the lexical margin is widest where Latin is scarce — in the lowest quartile the semantic system scores 0.333 and the lexical 0.476 — and narrowest where Latin dominates, where the semantic system has its best quartile (0.371) and the lexical its worst (0.419). The Spearman correlation between Latin share and the rank advantage of the lexical system is −0.10. BGE-M3 handles the English; what it loses is in the Chinese.

### 7.2 Not Names

Table 4 gives the neutralisation arms. Removing the 212 place names moves the lexical MRR from 0.450 to 0.448; removing 459 named entities of every type, 42,742 characters over 4,962 songs, to 0.444; removing the whole 605-entry catalogue, 168,004 characters over 6,315 songs — seven times the named-entity arm, including the curated slang and English vocabulary that a critic would say *is* the rap register — also to 0.444. The last arm is no longer a topic control at all but the sharpest test of the claim, and the lexical system still beats the untouched semantic system by 0.126. The identity the surface carries is not a list of words that could be removed; it is diffuse in how the characters are put together.

| Arm | Surfaces removed | Characters removed | Songs touched | Lexical MRR | Still above semantic (0.318) |
| --- | ---: | ---: | ---: | ---: | --- |
| None | 0 | 0 | 0 | 0.450 | — |
| Reference core | 90 | 24,501 | 2,578 | 0.449 | yes |
| Place names | 212 | 11,534 | 2,583 | 0.448 | yes |
| All named entities | 459 | 42,742 | 4,962 | 0.444 | yes |
| Whole catalogue, incl. slang and English | 605 | 168,004 | 6,315 | 0.444 | yes |

Table 4 Named-reference neutralisation. Each arm strips one lexicon from every song, refits the character space, and rescores it against the unchanged semantic space. Plain MRR over 7,236 queries.

RQ2 therefore has a negative answer on both counts, and the answer sharpens RQ3. A semantic model that keeps English and drops something diffuse in the Chinese character surface is either discarding that something or storing it where cosine cannot see it.

## 8. The Rhyme Form Has Structure

Before the probe, the rhyme-form space needs one supporting fact: that written endings in this corpus are structured enough for "rhyme form" to name a layer at all. The project's written-rhyme task supplies it. Lines are the cleaned newlines of each chunk; those ending in a Han character with a classifiable final yield the same seventeen families as Section 5.3. Songs are assigned to train, validation, and test partitions (5,147 / 1,097 / 1,104 songs) by leakage group rather than individually — assigning songs one at a time would have put 433 of 799 multi-song text components on both sides of a boundary, touching 18.0% of the population — and a target line that exactly or near-duplicates any training or validation line is excluded, which removes 1,928 of 53,444 test candidates and leaves 51,516 leakage-safe adjacent-line events from 1,064 songs.

The task predicts the family of the next line's ending. A global-frequency baseline reaches top-3 accuracy 0.345; conditioning on the previous family alone lifts it to 0.649; a hierarchical context model that first decides whether to continue the current family and then, if not, which family to switch to, reaches 0.698 (95% CI 0.690–0.705) and MRR 0.630 (0.622–0.637), +0.049 over Markov on top-3 (0.044–0.054) and only +0.001 over a flat context classifier (Fig. 4). Continuation is nearly certain to be predicted (top-3 1.000 for both models) and switching remains hard (top-3 0.404 for the hierarchical model against 0.307 for Markov). Removing the source-credit label from the model's features changes top-3 by +0.0002 with an interval of −0.0008 to +0.0012 — which is the rhyme-form result of Section 5.4 seen from the other side: the label barely helps predict the next ending, and the endings barely identify the label. Written endings have grammar; they are not signature. An external check against the MuChin annotation export (Wang *et al.* 2024) agrees with the family markers at F1 0.931 but is partly circular and is not independent gold; details are in the supplement.

[[FIGURE:figure_4_written_rhyme_benchmark.png]]

Fig. 4 Prediction of the next dictionary-estimated written line-ending family on the strict terminal-Han population of corpus v2. Panel A compares global frequency, first-order Markov, flat context, and hierarchical context with and without the source-credit label across 51,516 leakage-safe events in 1,064 held-out songs; whiskers are 95% song-cluster bootstrap intervals from 2,000 replicates. Panel B separates continuation from family-switch events. Panel C plots four paired score differences for the model with versus without the label; three intervals cross zero and the largest difference, Top-5, is +0.0015.

Alt text: Four aligned dot-and-whisker panels show Top-1, Top-3, Top-5, and MRR for five models; context models dominate the global baseline, and the flat and hierarchical models nearly overlap. A grouped bar panel shows near-ceiling Top-3 accuracy when the next line continues the same family and much lower accuracy when it switches. A third panel shows four label-ablation differences on a scale of thousandths, three of them crossing zero.

## 9. Present but Hidden: The Whitening Probe

### 9.1 Method and Controls

If the semantic model has discarded the identity signal, no linear transform of its vectors will find it. If the signal is present but swamped by directions that vary within an author — what each song is about — then shrinking those directions will expose it. Within-class covariance normalisation does exactly that. Let Σ_w be the covariance of song vectors about their own label's mean, weighted so that each leakage group counts once within a label, with Ledoit–Wolf shrinkage towards a scaled identity (Ledoit and Wolf 2004); the transform maps a centred song vector x to Σ_w^(−1/2) x and re-normalises it, after which the protocol of Section 4 runs unchanged.

The transform must not see the songs it is scored on. Leakage groups are dealt into five folds; the transform for a fold is fitted on the other four and applied to every song; only that fold's queries are scored in it, against profiles built from all other songs exactly as before. Two controls fix what a gain means. *Total whitening* applies the same step to the total covariance with no labels used, and separates a generic geometric correction from an identity-specific one. *Permuted-label whitening* fits the within-author covariance after the training labels are shuffled, keeping class sizes and destroying author structure; it is the null. Centring alone is a third control, because every whitening also centres. The scorer reproduces the builder's dense scores to 7 × 10⁻⁷ and the untransformed space reproduces the builder's MRR exactly before any transformed space is trusted.

### 9.2 Results

Table 5 gives the probe. Within-author whitening lifts the semantic system from 0.318 to 0.449 (+0.128, 95% interval [+0.119, +0.137]), level with the raw character surface at 0.450 (difference +0.007, [−0.003, +0.017]) and better than it on 2,705 queries to 2,772. Fused with the raw lexical system it reaches 0.544 (Recall@1 0.443, Recall@10 0.737), +0.090 over the lexical system alone ([+0.083, +0.098]) and +0.067 over the original fusion ([+0.060, +0.073]). Semantic and lexical evidence are complementary once the semantic geometry is corrected (Fig. 5).

| Transform of the BGE-M3 song vectors | MRR | Recall@1 | Recall@10 |
| --- | ---: | ---: | ---: |
| None (raw cosine) | 0.318 | 0.219 | 0.514 |
| Centring only | 0.314 | 0.218 | 0.503 |
| Total whitening (no labels) | 0.432 | 0.338 | 0.606 |
| Within-author whitening, permuted labels | 0.431 | 0.337 | 0.606 |
| **Within-author whitening** | **0.449** | **0.352** | **0.634** |
| Character n-grams, for reference | 0.450 | 0.357 | 0.630 |
| Fusion: lexical + raw semantic | 0.476 | 0.380 | 0.661 |
| Fusion: lexical + whitened semantic | 0.544 | 0.443 | 0.737 |
| Lexical, 1,024-d SVD, within-author whitened | 0.518 | 0.406 | 0.737 |
| **Fusion: whitened lexical + whitened semantic** | **0.587** | **0.482** | **0.783** |

Table 5 The whitening probe. Five-fold by leakage group; each transform is fitted on four folds and scores the fifth. Plain MRR over 7,236 queries. The last two rows give the lexical space the same treatment (Section 9.2).

The controls say what the gain is made of. Total whitening, which uses no labels, already reaches 0.432 (+0.116 over centring, [+0.107, +0.124]); permuted-label whitening reaches the same 0.431. Most of the recovery is therefore the embedding's anisotropy: a few common directions dominate every raw cosine, and removing them exposes structure of every kind, identity included. The remaining +0.017 of the supervised transform over the total one ([+0.014, +0.020]), and +0.019 over the permutation null ([+0.016, +0.021]), is author structure and nothing else. Within-author variation accounts for 89.6% of the embedding's variance, against 96.0% under the null; the identity signal is a thin slice, but it is there, and it is where cosine does not look.

Two further checks change what may be claimed. First, "level with the lexical system" holds only against the *raw* surface. Applying the same transforms to a 1,024-dimensional truncated singular-value decomposition of the TF–IDF rows — the dense system's own width, keeping 44% of the variance and costing 0.057 of MRR untransformed — lifts that space from 0.393 to 0.511 with total whitening and 0.518 with within-author whitening, +0.113 and a further +0.008 ([+0.005, +0.011]). Anisotropy is not a property of the neural model alone; a bag of character n-grams has dominant common directions too, and under equal treatment the surface keeps a lead of 0.070 over the whitened semantic space ([+0.060, +0.081]). Fusing the two whitened spaces gives 0.587 (Recall@1 0.482, Recall@10 0.783), +0.042 over the fusion of raw lexical with whitened semantic ([+0.037, +0.047]) and +0.069 over the whitened lexical space alone ([+0.062, +0.075]); it is the strongest system in the study, and both of its parts are label-free in all but +0.008 and +0.017. Second, the label-fitted part of the transform does not transfer, and the label-free part does. Holding out 15% of labels entirely (34 labels), fitting the transforms on the other labels' songs only, and scoring the held-out labels' queries against the usual profiles, total whitening lifts those queries from 0.297 to 0.455 (+0.156, [+0.131, +0.180]) while within-author whitening fitted on other authors reaches 0.429, *below* the label-free transform by 0.027 ([−0.035, −0.019]) and no better than the fold-wise transform that had seen those authors (0.421; difference −0.008, [−0.016, −0.000]). What transfers is the geometry; the +0.017 of author structure is specific to the authors it was fitted on.

[[FIGURE:figure_5_identity_spaces.png]]

Fig. 5 The same songs in three representation spaces, and how often a song's nearest neighbours share its source-credit label. Panels A–C are t-SNE layouts of one fixed set of 442 songs from the twelve labels with the most songs, at most one song per leakage group, in the BGE-M3 space (A), the character 2–5-gram TF–IDF space (B), and the BGE-M3 space after within-author whitening fitted on songs by every other label (C). One seed and one perplexity are used throughout; cluster sizes and the distances between clusters carry no meaning (van der Maaten and Hinton 2008). Panel D is the measurement the layouts gesture at, over all 7,236 songs and without any projection: for k from 1 to 20, the share of a song's k nearest cosine neighbours, excluding its own leakage group, that carry its label. The whitened curve is fold-wise, so no song is scored in a space fitted on itself. The dotted line is the chance level implied by label sizes.

Alt text: Three square scatter panels show the same coloured and shaped points. In the semantic panel the colours are intermixed; in the lexical panel and the whitened semantic panel several labels form visible clusters. A line chart below plots same-label neighbour share against k for three spaces: the whitened semantic curve is highest, starting near 0.28 at k equal to 1, the lexical curve is next, starting near 0.20, and the raw semantic curve is lowest, starting near 0.10; all three decline towards a dotted chance line near zero. A legend lists twelve source-credit labels with their markers.

Neighbour purity tells the same story without a protocol: at k = 1, 10.2% of a song's nearest semantic neighbour shares its label, 19.7% in the raw character space, and 28.1% in the whitened semantic space, against a chance level of 0.5%. RQ3 is answered in three parts: the information was present in the semantic embedding; what hid it was almost entirely generic geometry, which hides identity in the lexical space as well and by a similar amount; and the small author-specific remainder is real within the corpus but is not a portable description of what identity looks like.

## 10. Discussion

### 10.1 What "Identity" Is Here

Every result in this article is about a credit string and the text filed under it. The strings were audited for splits and none was found, but they were not resolved to persons and cannot be. A label's identity, in the sense measured, is the property of a held-out song being consistent with the rest of its label's text; it lives in the character surface first, in the semantic content second once the geometry is corrected, and in the rhyme scheme only faintly. None of this says what a rapper is like, where they are from, or whom they know, and the neutralisation arms show that even the most nameable content of the surface — places and people — is not what makes the label recognisable.

### 10.2 The Geometry Lesson

The working hypothesis of the study, that a semantic objective erases authorial identity, was wrong, and the way it was wrong is the useful part. BGE-M3 keeps enough about who wrote a song to match the raw character surface, but it keeps it in a part of the space that cosine on raw vectors does not weight. Nearly all of the gap was closed by a correction that uses no labels at all — the anisotropy result of Ethayarajh (2019), Li *et al.* (2020), and Su *et al.* (2021), observed on a task those authors did not study — and the same correction lifts a bag of character n-grams by a similar amount, so the effect is a property of cosine over high-dimensional text representations rather than of the neural model. The within-class step of Hatch, Kajarekar and Stolcke (2006) adds a little on top in both spaces, but only for the authors it was fitted on: the transferable part of "identity-aware geometry" turned out to be the part that knows nothing about identity. A digital-humanities practice that reads an embedding map by its raw cosine distances is therefore reading the map's geometry before the texts. The distances are not wrong; they answer a question — what is this passage about — that the scholar may not have asked.

### 10.3 What the Surface Knows

The character surface carries identity that survives the removal of every listed name and slang term. Sapkota *et al.* (2015) found that the most useful character n-grams for attribution are those that capture affixes and punctuation rather than whole words; the present decomposition is consistent with that reading and adds two facts for a Chinese corpus: the signal is in the Han characters more than the Latin ones, and it is not reducible to a lexicon. What exactly it is — function characters, particle habits, orthographic choices, line shapes — is the next question, and the released per-arm results are the baseline any answer must beat.

### 10.4 For Digital Humanities Practice

Three practices follow. Ask a representational question — where is the signal — rather than a leaderboard question, because the leaderboard answer (TF–IDF beats BGE-M3) was true and misleading at once. Score every representation under one protocol on one query set with one leakage unit, so that a difference is a difference in representation. And before reading any embedding's cosine geometry as evidence, correct it with the cheapest available step and see what changes; if a label-free whitening moves a result by a third, the raw map was not measuring what it seemed to.

## 11. Limitations, Ethics, and Copyright

The corpus is a collected sample rather than a census of Chinese rap. Its source platforms, acquisition route and dates, temporal coverage, sampling frame, transcription origin, and rights basis are not documented in the export and must be established by the dataset owner before submission; no temporal split is possible because no record carries a date. Source-credit labels are not identity-resolved and may include aliases, groups, collaborations, or metadata errors. No text-derived result establishes hometown, belief, cultural membership, collaboration, influence, friendship, or affiliation.

The protocol is transductive to the fixed corpus: the lexical vocabulary and the whitening transforms are estimated on it, labels excluded, and the held-out-label transfer test bounds but does not remove that dependence. Near-duplicate detection uses one threshold and cannot find every paraphrase or lightly edited hook. The whitening probe is linear; a non-linear identity objective could recover more, and its absence here is a choice of the cheapest sufficient test, not a claim that the linear result is a ceiling. BGE-M3 remains hard to explain at token level, and the decomposition of the lexical space stops at script class and lexicon membership.

The cultural-reference layer has no completed occurrence-level human gold; literal matching is context-insensitive, the transformer is general-domain and Traditional-Chinese-oriented, and agreement gates favour conventional place and language names over creative spellings, so the released network may under-represent the innovative references that make the domain important. Written rhyme is inferred from dictionary pinyin applied to transcribed endings and cannot resolve every polyphone, dialectal pronunciation, or performed elision; it excludes code-switched line endings and must not be read as performed rhyme or flow.

Copyright and privacy constrain release. Public artifacts contain aggregate evidence and the figures' source tables — the Figure 5 layout table carries a label and two coordinates per point and no song identifier — and no lyrics, lines, identifiers, per-song ranks, vectors, membership rows, or reviewer contexts. Researchers with lawful access can reproduce every builder against file-level checksums; the lyrics are not redistributed.

## 12. Conclusion

One corpus, one protocol, three spaces. The character surface identifies a Chinese rap source label far better than a frozen multilingual semantic embedding, the rhyme scheme identifies it faintly and adds nothing, and the surface advantage is neither the English nor the names. Yet the embedding has not forgotten who wrote the song. A label-free correction of its geometry brings it level with the raw surface, the same correction lifts the surface in turn, and the two corrected spaces together identify labels better than anything else tried. The signal was present in both; the standard way of reading either could not see it. That is the finding, and it is a caution for any study that turns a vector space into a map of a culture.

## Data Availability

Copyright-safe aggregate outputs, the method contract and its amendments PD-001 and PD-002, corpus-reconciliation results, the embedding-configuration probe, the leakage-group module and its tests, every builder named in this article, figure-source tables, and validation manifests are maintained at https://github.com/Mo119m/chinese-rap-embeddings-to-evidence. The public package excludes lyric text, lines, song and chunk identifiers, per-song ranks and coordinates, embeddings, membership rows, and reviewer contexts. The private corpus is not redistributed. [Archival DOI to be added before submission.]

## Funding

[To be completed by the authors; state "None" if applicable.]

## Conflict of Interest

[To be completed and approved by all authors.]

## Ethics Statement

The study analyses collected lyric text and corpus metadata and releases only aggregate, copyright-safe evidence. No source-credit label is treated as a verified natural-person identity. The authors must confirm institutional requirements and any necessary ethics determination before submission.

## Author Contributions

[CRediT roles to be completed by the authors.]

## AI Disclosure Statement

This manuscript was prepared with the assistance of OpenAI Codex and Anthropic Claude Code [exact versions to be completed by the authors]. The tools were used for reproducible code development, data analysis, figure generation, and English-language drafting and editing. The human authors must independently check, reproduce, and approve all analyses, figures, claims, and references before submission and remain fully accountable for the work. The same use must be disclosed in the submission cover letter.

## References

Baioud, G. (2024) 'Constructing "corrupted village wives and urban men" through multilingual performances', *Language in Society*, 53/1: 25–45. https://doi.org/10.1017/S0047404522000665

Barrett, C. (2012) 'Hip-hopping across China: intercultural formulations of local identities', *Journal of Language, Identity and Education*, 11/4: 247–60. https://doi.org/10.1080/15348458.2012.706172

Blessing, A. *et al.* (2017) 'An end-to-end environment for research question-driven entity extraction and network analysis', *Proceedings of the Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature*, pp. 57–67. https://doi.org/10.18653/v1/W17-2208

Chen, J. *et al.* (2024) 'M3-Embedding: multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation', *Findings of the Association for Computational Linguistics: ACL 2024*, pp. 2318–35. https://doi.org/10.18653/v1/2024.findings-acl.137

Dehak, N. *et al.* (2011) 'Front-end factor analysis for speaker verification', *IEEE Transactions on Audio, Speech, and Language Processing*, 19/4: 788–98. https://doi.org/10.1109/TASL.2010.2064307

Ethayarajh, K. (2019) 'How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings', *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, pp. 55–65. https://doi.org/10.18653/v1/D19-1006

Hatch, A. O., Kajarekar, S. and Stolcke, A. (2006) 'Within-class covariance normalization for SVM-based speaker recognition', *Proceedings of Interspeech 2006*, pp. 1471–74.

Hirjee, H. and Brown, D. G. (2009) 'Automatic detection of internal and imperfect rhymes in rap lyrics', *Proceedings of the 10th International Society for Music Information Retrieval Conference*, pp. 711–16. https://archives.ismir.net/ismir2009/paper/000029.pdf

Jiang, H. *et al.* (2022) 'Annotating the Tweebank corpus on named entity recognition and building NLP models for social media analysis', *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pp. 7199–208. https://doi.org/10.18653/v1/2022.lrec-1.780

Lassen, I. M. S. *et al.* (2024) 'Epistemic consequences of unfair tools', *Digital Scholarship in the Humanities*, 39/1: 198–214. https://doi.org/10.1093/llc/fqad091

Ledoit, O. and Wolf, M. (2004) 'A well-conditioned estimator for large-dimensional covariance matrices', *Journal of Multivariate Analysis*, 88/2: 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4

Li, B. *et al.* (2020) 'On the sentence embeddings from pre-trained language models', *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, pp. 9119–30. https://doi.org/10.18653/v1/2020.emnlp-main.733

Lin, Y. and Wang, T. (2024) 'Rhyming style, persona, and the contested landscape of authentic Chinese hip hop', *Journal of Sociolinguistics*, 28/2: 22–41. https://doi.org/10.1111/josl.12635

Liu, J. (2014) 'Alternative voice and local youth identity in Chinese local-language rap music', *positions: asia critique*, 22/1: 263–92. https://doi.org/10.1215/10679847-2383840

Liu, J. (2021) 'Language, identity and unintelligibility: a case study of the rap group Higher Brothers', *East Asian Journal of Popular Culture*, 7/1: 43–59. https://doi.org/10.1386/eapc_00038_1

Liu, J. *et al.* (2023) 'Linguistic tone in Chinese rap: an interdisciplinary approach', *Journal of New Music Research*, 52/4: 265–84. https://doi.org/10.1080/09298215.2024.2329075

Paszke, A. *et al.* (2019) 'PyTorch: an imperative style, high-performance deep learning library', *Advances in Neural Information Processing Systems*, 32. https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

Peng, N. and Dredze, M. (2015) 'Named entity recognition for Chinese social media with jointly trained embeddings', *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pp. 548–54. https://doi.org/10.18653/v1/D15-1064

Rafaeli, O. *et al.* (2026) 'Mind the gap: word-embedding and multi-layered literary networks', *Digital Scholarship in the Humanities*, 41/Supplement 1: i213–29. https://doi.org/10.1093/llc/fqaf112

Sapkota, U. *et al.* (2015) 'Not all character n-grams are created equal: a study in authorship attribution', *Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pp. 93–102. https://doi.org/10.3115/v1/N15-1010

Savoy, J. (1997) 'Statistical inference in retrieval effectiveness evaluation', *Information Processing and Management*, 33/4: 495–512. https://doi.org/10.1016/S0306-4573(97)00027-7

Stamatatos, E. (2009) 'A survey of modern authorship attribution methods', *Journal of the American Society for Information Science and Technology*, 60/3: 538–56. https://doi.org/10.1002/asi.21001

Su, J. *et al.* (2021) 'Whitening sentence representations for better semantics and faster retrieval', arXiv preprint arXiv:2103.15316. https://doi.org/10.48550/arXiv.2103.15316

Sun, Y. *et al.* (2023) 'SongRewriter: a Chinese song rewriting system with controllable content and rhyme scheme', *Findings of the Association for Computational Linguistics: ACL 2023*, pp. 12863–80. https://doi.org/10.18653/v1/2023.findings-acl.814

Sundararajan, K. and Woodard, D. (2018) 'What represents "style" in authorship attribution?', *Proceedings of the 27th International Conference on Computational Linguistics*, pp. 2814–22. https://doi.org/10.18653/v1/C18-1238

Timkey, W. and van Schijndel, M. (2021) 'All bark and no bite: rogue dimensions in transformer language models obscure representational quality', *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pp. 4527–46. https://doi.org/10.18653/v1/2021.emnlp-main.372

van der Maaten, L. and Hinton, G. (2008) 'Visualizing data using t-SNE', *Journal of Machine Learning Research*, 9: 2579–605. https://jmlr.org/papers/v9/vandermaaten08a.html

Wang, A. *et al.* (2023) 'Can authorship representation learning capture stylistic features?', *Transactions of the Association for Computational Linguistics*, 11: 1416–31. https://doi.org/10.1162/tacl_a_00610

Wang, T. and Lin, Y. (2024) 'Variation is the way to perfection: imperfect rhyming in Chinese hip hop', *Linguistics Vanguard*, 10/1: 505–15. https://doi.org/10.1515/lingvan-2024-0093

Wang, X. (2013) '"I am not a qualified dialect rapper": constructing hip-hop authenticity in China', *Sociolinguistic Studies*, 6/2: 333–72. https://doi.org/10.1558/sols.v6i2.333

Wang, Z. *et al.* (2024) 'MuChin: a Chinese colloquial description benchmark for evaluating language models in the field of music', *Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence*, pp. 7771–79. https://doi.org/10.24963/ijcai.2024/860

Wolf, T. *et al.* (2020) 'Transformers: state-of-the-art natural language processing', *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 38–45. https://doi.org/10.18653/v1/2020.emnlp-demos.6

Xue, L. *et al.* (2021) 'DeepRapper: neural rap generation with rhyme and rhythm modeling', *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing*, pp. 69–81. https://doi.org/10.18653/v1/2021.acl-long.6

Yang, Y. *et al.* (2018) 'Distantly supervised NER with partial annotation learning and reinforcement learning', *Proceedings of the 27th International Conference on Computational Linguistics*, pp. 2159–69. https://doi.org/10.18653/v1/C18-1183
