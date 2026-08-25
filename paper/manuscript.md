# From Embeddings to Evidence: Interpretable and Uncertainty-Audited Mapping of Chinese Rap Lyrical Repertoires

[Author name(s)]

[Affiliation(s)]

[Corresponding author email]

## Structured Abstract

**Purpose:** This study asks how multilingual text embeddings can be converted into interpretable and uncertainty-audited evidence about Chinese rap lyrics, rather than treated as self-explanatory points on a map.

**Design/methodology/approach:** A frozen corpus of 7,129 comparison-eligible songs and 21,346 lyric chunks is represented with duplicate-weighted BGE-M3 dense vectors. A second representation excludes exact cleaned text shared across source-credit labels. Reciprocal top-five matches must occur in both representations. Characteristic words, dictionary-estimated written line endings, and writing-form measures provide separately calculated, support-stratum-calibrated evidence. A 250-replicate song bootstrap audits local match repeatability. A weakly supervised retrieval test compares BGE-M3, character n-gram TF-IDF, and their untuned fusion.

**Findings:** The two representations yield 86 reciprocal matches among 204 eligible labels, but only 16 reappear in at least half of song bootstraps and none reaches 0.80. Eleven of the 16 have at least one support-stratum-calibrated lexical, written-ending, or form signal. In 1,000 low-overlap same-song queries, character TF-IDF outperforms dense BGE-M3, while fusion performs best (MRR 0.278; nDCG@10 0.208).

**Originality:** The study separates representation stability, resampling repeatability, and human-readable concordance instead of presenting embedding proximity as a cultural or social relationship.

**Contribution to the field of Digital Humanities:** It offers a reproducible, claim-bounded workflow and public atlas for studying lyrical repertoires while preserving copyrighted text and exposing uncertainty.

Keywords: Chinese rap; digital humanities; lyrics; text embeddings; BGE-M3; corpus linguistics; lexical keyness; written line endings; robustness; information retrieval

## 1. Introduction

Computational work on song lyrics often moves quickly from a vector representation to a map, a cluster label, or a recommendation. That sequence can produce an attractive interface while leaving the main scholarly questions unanswered. What exactly distinguishes one lyrical repertoire from another? Why does a model place two repertoires close together? Would the match survive if repeated or shared text were controlled? Would it survive if a different set of songs had been observed? These questions are especially consequential for Chinese rap, where lyrics commonly combine Mandarin, English, regional expressions, artist and crew names, place references, phonetic play, repeated hooks, and transcribed performance cues.

This article presents an evidence-calibrated workflow for moving from chunk-level embeddings to four research outputs: a corpus-internal encoder benchmark, a duplicate-controlled repertoire graph, interpretable lyrical profiles, and a song-bootstrap repeatability audit of pairwise matches. The central object is not a verified performer identity. It is a source-credit-labelled slice of the corpus, which may correspond to an individual, group, collaboration, alias, or noisy metadata string. The word *repertoire* is used in that restricted corpus sense throughout.

The contribution is methodological and empirical. Methodologically, the workflow separates three questions that are often collapsed. Representation stability asks whether a match persists after an explicit text-processing perturbation. Resampling repeatability asks how often it returns when songs are sampled again within each label. Interpretive concordance asks whether independently calculated lexical, written-ending, or writing-form signals help a reader understand the match. Empirically, the results show why these distinctions matter. Global pairwise geometry changes little when exact cross-label shared text is removed, but local graph membership is more sensitive. Most of the 86 representation-stable matches have modest bootstrap frequency. Dense BGE-M3 is also not the strongest standalone method on the completed weak-supervision retrieval task; an untuned dense-lexical fusion performs best.

The study addresses four research questions:

RQ1. How does BGE-M3 dense retrieval compare with character n-gram lexical retrieval and their untuned fusion on a corpus-internal, low-lexical-overlap same-song retrieval task?

RQ2. How stable are source-label lyrical-repertoire matches under exact-text duplicate weighting, exclusion of exact text shared across labels, and within-label song resampling?

RQ3. Which corpus-observed lexical, written-line-ending, and writing-form profiles can be estimated under explicit multi-song support and stability gates?

RQ4. To what extent can representation-stable matches be accompanied by separately calculated, support-stratum-calibrated lexical, written-ending, or form evidence, and how often do they lack a passing signal from the implemented evidence channels?

The public-facing result is a self-contained atlas in which a reader can select a label, see its corpus-distinctive words and written-form tendencies, and inspect what independently measured evidence two matched profiles share. The page does not expose full lyrics, song or chunk identifiers, embeddings, private membership rows, or unreviewed named-entity occurrences. It therefore demonstrates how a digital-humanities interface can present research outputs without turning methods, dataset counts, or copyrighted text into the interface's main content.

## 2. Related Work and Analytical Position

Rap has long motivated computational work because meaning, vocabulary, repetition, rhyme, and line-to-line progression matter simultaneously. Malmi *et al.* (2016) combined semantic and rhyme-related features in a learning-to-rank system for rap-line continuation. Karsdorp, Manjavacas, and Kestemont (2019) showed that linguistic features including lexical diversity and rhyme density affect human judgements of generated rap authenticity. Work on large lyric corpora likewise treats lexical, structural, diversity, and rhyme descriptors as distinct analytical families rather than interchangeable proxies (Parada-Cabaleiro *et al.*, 2024).

Chinese lyric generation research reinforces the need to model meaning and sound separately. QiuNiu conditions generation on passage-level content (Zhang *et al.*, 2022), while SongRewriter models vowel information and explicit rhyme constraints (Sun *et al.*, 2023). These systems address generation rather than repertoire mapping, but they demonstrate why a text-only study should not treat a single semantic vector as a sufficient description of lyric form. A recent analysis of perceived lyric similarity also reports complementary roles for semantic, phonetic, and audio evidence (Kim and Akama, 2024). The present study is deliberately text-only, so it estimates dictionary-derived patterns at written line endings but does not claim performed rhyme, timing, delivery, beat alignment, or flow.

Research on Chinese rap itself sets an important evidentiary standard. Liu *et al.* (2024) combine lyric, musical, statistical, and interview evidence to study Mandarin lexical tone in rap, while Wang (2025) analyses how locally meaningful phonological variables and personae are enacted in Beijing rap performance. These multimodal and culturally situated studies show why a lyrics-only representation cannot establish pronunciation, performance practice, or social identity. The present corpus map is correspondingly narrower: it produces text-derived hypotheses and corpus-relative descriptions that require other evidence before they can be interpreted as performer-level cultural findings.

For lexical interpretation, raw word frequency is a poor answer to the question 'what is characteristic here?' because it rewards generally common vocabulary. Song lyrics also exhibit substantial lexical clumpiness when repeated hooks concentrate a term in one or a few songs (Langenhorst, Frommherz, and Meier-Vieracker, 2023). Informative-Dirichlet weighted log-odds provides a regularized comparison against a background corpus and is well suited to identifying relatively overrepresented lexical features (Monroe, Colaresi, and Quinn, 2008). The method is used here with additional multi-song and leave-one-song gates so that a term must characterize a repertoire rather than one song.

BGE-M3 was selected as a candidate encoder because the corpus includes Chinese, English, mixed-script expressions, and passages of varied length. The model is multilingual, supports inputs longer than many earlier sentence encoders, and exposes dense, sparse, and multi-vector retrieval functions (Chen *et al.*, 2024). This study uses only its 1,024-dimensional dense output, with a local maximum input length of 2,048. The model's published capabilities motivate evaluation; they do not prove that its dense head is optimal for Chinese rap. The local benchmark and the later fusion result are therefore integral to the study rather than a decorative model-selection paragraph.

## 3. Corpus and Claim Boundary

The frozen source snapshot contains 7,211 canonical song records and 22,128 lyric chunks. Deterministic text hygiene was applied through a derived clean-text sidecar without deleting canonical records. Metadata-only material was withheld, and records that failed the declared source-credit/title comparison rule were excluded rather than silently repaired. The final comparison population contains 7,129 songs, 21,346 active lyric chunks, and 240 source-credit labels (Table 1).

| Stage | Songs | Lyric chunks | Source-credit labels | Eligibility note |
| --- | ---: | ---: | ---: | --- |
| Frozen canonical snapshot | 7,211 | 22,128 | — | Preserved source snapshot |
| Comparison population | 7,129 | 21,346 | 240 | Passed the source-credit/title comparison rule |
| Graph-eligible labels | — | — | 204 | At least five clean songs and effective-text mass of at least 20 in both representations |

Table 1 Corpus populations used in the analysis

The unit of interpretation remains a source-credit-labelled corpus slice because external identity resolution is incomplete. A label may combine aliases, groups, collaborations, or metadata noise. Consequently, profile terms are described as 'distinctive in the observed corpus', not as favourite words or beliefs. A place name in a lyric is not evidence of hometown. Co-occurrence is not collaboration, friendship, affiliation, or influence. Dictionary-derived endings are properties of written transcriptions, not observed vocal performance.

Detailed transformation rules, hashes, and row-level audit material are retained in the reproducibility package rather than the main article. This keeps corpus hygiene visible but subordinate to the downstream analytical contribution.

## 4. Methods: From Embeddings to Evidence

### 4.1 Framework

Fig. 1 summarizes the workflow. Clean lyric chunks are encoded once, but the vectors then enter separate downstream paths: corpus-internal retrieval evaluation, duplicate-controlled repertoire aggregation, reciprocal-neighbour matching, interpretable profile estimation, pairwise evidence calibration, and song-level resampling. No two-dimensional projection decides a match.

[[FIGURE:figure1_pipeline.png]]

Fig. 1 Embedding-to-evidence workflow. The encoder is one component; downstream aggregation, robustness checks, evidence gates, and abstention create the reported outputs

Alt text: A left-to-right pipeline begins with cleaned lyric chunks and BGE-M3 or character n-gram representations. It branches into retrieval benchmarking and label-level aggregation. The aggregation branch applies duplicate weighting, shared-text exclusion, reciprocal-neighbour matching, interpretable lexical, written-ending, and form profiles, and song bootstrap checks. Outputs are a benchmark, profiles, bootstrap-audited lyric matches, and an atlas.

### 4.2 Encoder Sanity Benchmark

Dense vectors were generated offline with `FlagEmbedding.BGEM3FlagModel` from the fixed `BAAI/bge-m3` checkpoint revision `5617a9f61b028005a4858fdac845db406aefb181`. Inference used a maximum input length of 2,048, batch size two, 1,024-dimensional dense output, and L2 normalization. The packaged contract records the checkpoint-weight SHA-256 hash and vector-file hash.

The completed benchmark is a corpus-internal continuation and cohesion check, not a human semantic gold standard. Each query required at least 50 effective characters and at least one relevant chunk from the same song whose character-trigram Jaccard overlap with the query was no greater than 0.15. Exact text shared across source-credit labels was excluded. The eligible pool contained 7,849 queries and 15,760 candidate chunks from the 204 graph-eligible labels; 1,000 queries were sampled deterministically.

Three systems were compared. BGE-M3 dense retrieval used cosine similarity between L2-normalized vectors. The lexical baseline used character two- to five-gram TF-IDF, which is appropriate for Chinese text because it does not depend on one segmentation scheme and can capture partial strings. The fusion system standardized each system's scores within a query and averaged them with equal weight. No fusion weight was tuned on the test queries. Evaluation reports mean reciprocal rank (MRR), recall at 1, 5, and 10, and normalized discounted cumulative gain at 10 (nDCG@10).

### 4.3 Duplicate-Weighted Repertoire Representations

Let \(e_{ij}\) denote the L2-normalized BGE-M3 embedding for chunk \(j\) assigned to source label \(i\). Each exact cleaned-text hash receives total mass one across retained copies. If hash \(h\) occurs \(n_h\) times in the comparison population, each occurrence receives weight \(w_h = 1/n_h\). The primary centroid is

\[
c_i = \operatorname{norm}\left(\sum_j w_j e_{ij}\right).
\]

A sensitivity centroid is constructed after removing every exact cleaned-text hash observed under more than one source-credit label. This second representation tests whether proximity depends on identical text attributed across labels. It does not remove paraphrases, near duplicates, lightly edited hooks, or broader collection bias.

### 4.4 Representation-Stable Matches

A label is eligible when it contains at least five clean songs and effective-text mass of at least 20 in both representations. Pairwise cosine similarities are computed among the 204 eligible labels. A match is retained only when both endpoints rank one another among their top five neighbours in both the primary and shared-text-exclusion representations. The term *representation-stable* refers only to survival of these two declared text treatments.

For visual orientation, normalized consensus centroids are projected into two dimensions with deterministic principal component analysis. The first two components account for 26.2% of variance. Projection position is therefore approximate, and edge membership is never inferred from two-dimensional distance.

### 4.5 Interpretable Lyrical Profiles

Characteristic words are estimated with Jieba 0.42.1 content-word segmentation, duplicate weighting, and informative-Dirichlet log-odds against the rest of the corpus with prior mass 1,000. A released term requires effective count of at least four, support from at least five songs, log-odds z at least 2.0 in both text representations, and a positive leave-one-song result in at least 80% of checks. The last condition reduces the chance that one song supplies a label's apparent vocabulary.

Written line-ending profiles use the dictionary Mandarin final of the last Han character in each non-empty written line. Exact repeated lines count once within a song, and song-level ending distributions are normalized before aggregation. A local written-ending echo statistic asks whether a line shares a final with any of the previous four written lines more often than expected after randomly reordering the same song's endings. Deterministic song bootstrap intervals describe sampling variation. This is a written-text proxy. Polyphones, regional pronunciation, delivery, timing, melody, and internal rhyme are not resolved.

Writing form is summarized through three song-level measures: median effective line length, the share of exact repeated lines, and the share of lines containing both Chinese and English. Each is expressed as a percentile among the 204 eligible labels with a deterministic 90% song-bootstrap interval.

### 4.6 Pairwise Concordance Evidence

All 20,706 unordered eligible-label pairs receive three auxiliary comparison scores: weighted Jaccard overlap among characteristic words, Jensen-Shannon similarity between written-ending distributions, and robust-scaled similarity of writing-form profiles. Percentiles are calibrated within minimum-song support strata. A signal is displayed only when it reaches the 90th percentile of its support stratum.

These are post-hoc concordant signals. They do not decompose BGE-M3 causally. A representation-stable match can therefore remain without a passing auxiliary signal when none of the implemented measures reaches its evidence gate. Abstention is preferable to a fluent but unsupported explanation.

### 4.7 Song-Level Bootstrap

Sampling sensitivity is assessed with 250 deterministic bootstrap replicates. Within each source-credit label, observed songs are sampled with replacement. Primary and shared-text-exclusion centroids, cosine similarities, reciprocal top-five graphs, and their intersection are recomputed. For each original match, bootstrap selection frequency is the proportion of replicates in which that match reappears in the two-representation intersection. This quantity is a repeatability diagnostic, not a posterior probability and not confidence in a social or cultural relationship.

## 5. Results

### 5.1 Dense and Lexical Retrieval Are Complementary

Character two- to five-gram TF-IDF outperforms BGE-M3 dense retrieval on all five completed metrics (Table 2; Fig. 2). BGE-M3 obtains MRR 0.223 and nDCG@10 0.152, compared with 0.255 and 0.190 for TF-IDF. Equal-weight fusion performs best, reaching MRR 0.278, recall@10 0.363, and nDCG@10 0.208. Relative to BGE-M3, fusion improves MRR by 0.055 and recall@10 by 0.060.

| System | MRR | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 dense | 0.223 | 0.179 | 0.258 | 0.303 | 0.152 |
| Character 2-5 gram TF-IDF | 0.255 | 0.210 | 0.298 | 0.319 | 0.190 |
| Equal-weight z-score fusion | 0.278 | 0.230 | 0.319 | 0.363 | 0.208 |

Table 2 Corpus-internal low-overlap same-song retrieval, 1,000 queries

[[FIGURE:figure2_encoder_benchmark.png]]

Fig. 2 Encoder sanity benchmark across five retrieval metrics. Same-song membership supplies weak supervision and is not a human semantic relevance judgement

Alt text: Grouped horizontal bars compare BGE-M3 dense, character n-gram TF-IDF, and equal-weight fusion on MRR, Recall at 1, 5, and 10, and nDCG at 10. Fusion is highest on every metric; TF-IDF is second; BGE-M3 is lowest.

The result does not establish that TF-IDF is semantically superior. Same-song membership is a noisy relevance label, and character overlap can still signal topical or formulaic cohesion even under the trigram cap. It does establish that BGE-M3 should not be presented as self-evidently best for this corpus and that dense and lexical evidence are complementary.

### 5.2 Exact Shared Text Alters Local Matches More Than Global Geometry

The support rules retain 204 of 240 comparison labels. Reciprocal top-five intersection across the primary and shared-text-exclusion representations produces 86 matches connecting 93 labels. Across all 20,706 pairs, similarity scores remain strongly aligned after shared exact text is removed (Pearson r = 0.980; Spearman rho = 0.965). Local membership is less stable: the intersection retains 61.4% of primary mutual-neighbour edges and 59.3% of sensitivity mutual-neighbour edges, with graph Jaccard 0.432 (Table 3).

| Diagnostic | Result |
| --- | ---: |
| Eligible labels | 204 |
| Two-representation reciprocal top-five matches | 86 |
| Labels connected by those matches | 93 |
| Pairwise Pearson correlation | 0.980 |
| Pairwise Spearman correlation | 0.965 |
| Primary mutual-edge retention | 61.4% |
| Sensitivity mutual-edge retention | 59.3% |
| Edge-set Jaccard | 0.432 |

Table 3 Exact-cross-label-shared-text sensitivity

The distinction between global and local behaviour is substantive. A high correlation across all pairs does not imply that any particular nearest-neighbour decision is secure. Rank boundaries make local graphs sensitive even when the larger similarity field looks nearly unchanged.

### 5.3 Profiles Prefer Abstention to Filler

Profiles are generated for all 204 eligible labels. Of these, 156 (76.5%) pass the full evidence gates and 48 are marked limited. At least one characteristic word is available for 182 labels, while 157 have at least three. Across profiles, 1,136 characteristic-term instances are released, with a median of eight and a range of zero to eight per label. Written-ending profiles are available for 203 labels.

| Profile or explanation output | Result |
| --- | ---: |
| Full-evidence profiles | 156 / 204 (76.5%) |
| Limited profiles | 48 / 204 (23.5%) |
| Profiles with at least one characteristic word | 182 / 204 (89.2%) |
| Profiles with at least three characteristic words | 157 / 204 (77.0%) |
| Written-ending profiles | 203 / 204 (99.5%) |
| Matches with at least one gated explanation | 48 / 86 (55.8%) |
| Matches with no gated explanation | 38 / 86 (44.2%) |

Table 4 Coverage of interpretable profiles and pair evidence

Among all 86 representation-stable matches, 25 have a lexical signal, 16 have a written-ending signal, and 19 have a form signal; signals can overlap. No match passes all three gates. The 38 matches without a passing auxiliary signal are not filled with automatically generated stories. They remain visible only as semantic candidates with an explicit statement that none of the implemented evidence channels reaches the display gate.

### 5.4 Song Resampling Reveals a Small, Moderately Reselected Candidate Core

Bootstrap graphs contain between 46 and 90 two-representation matches per replicate, with median 66. Across the 86 original matches, selection frequency ranges from 0.072 to 0.712, with first quartile 0.192, median 0.280, and third quartile 0.427. Only 16 matches reach 0.50, nine reach 0.60, three reach 0.70, and none reaches 0.80 (Fig. 3).

[[FIGURE:figure3_bootstrap_distribution.png]]

Fig. 3 Song-bootstrap selection frequencies for the 86 original two-representation matches. The dashed line marks the atlas default threshold of 0.50

Alt text: A sorted dot plot shows selection frequency for 86 matches. Most dots lie below 0.50. Sixteen reach or exceed 0.50, three reach 0.70, and none reaches 0.80. The maximum is 0.712 and the median is 0.280.

The 16 matches at or above 0.50 connect 25 labels. Their mean selection frequency is 0.604 and median is 0.610. Eleven have at least one support-gated auxiliary signal; five lack a passing lexical, written-ending, or form signal. Lexical concordance is the most common signal in this core, appearing on nine matches, compared with three form signals and one written-ending signal.

[[FIGURE:figure4_repeatable_core.png]]

Fig. 4 Moderately reselected core of 16 lyric matches among 25 source-credit labels. Edge colour shows the strongest support-gated auxiliary signal; dashed grey edges have no passing signal. Width shows song-bootstrap selection frequency

Alt text: A network of 25 labelled nodes connected by 16 edges. Purple edges represent shared distinctive vocabulary, teal edges similar lyric structure, orange edges represent written-ending similarity, and dashed grey edges indicate matches without a passing auxiliary signal. Thicker edges were selected more frequently in 250 song bootstraps.

This is not a high-confidence cultural network. It is a moderately reselected candidate core under a specific corpus and graph rule. The absence of any match above 0.80 is a central result, not a defect to hide.

### 5.5 Worked Match Examples

Several core matches show how the evidence layers support interpretation without becoming biography. The source-credit-label pair 张雪飞–新街口组合 has the highest bootstrap frequency, 0.712. Its characteristic-word overlap is in the 98.3rd percentile and its form similarity in the 97.6th percentile within comparable support strata. The shared distinctive words are 怀念, 眼泪, 浪漫, and 爱情. These terms support a claim about shared corpus-relative vocabulary; they do not establish collaboration or influence between verified people.

The source-credit-label pair G Sauce–Rich4ever reappears with frequency 0.704 and shares the characteristic words 晓得, 肯定, 老子, and 不得 at the 98.5th percentile. The source-credit-label pair APMOZART–MULA SAKEE reappears with frequency 0.680 and shares 娃儿, 老子, 晓得, and 创造 at the 99.7th percentile. These examples also illustrate why a Chinese rap analysis should preserve Chinese expressions rather than translate every result into an abstract feature name.

The source-credit-label pair Jarstick–杀手耗 reappears with frequency 0.624 and has a written-ending similarity in the 99.4th percentile, especially for dictionary finals -i, -ai, and -e. This is the clearest written-ending example in the 16-match core, but the claim remains limited to transcribed written lines. The source-credit-label pair Capper–乃万 reappears with frequency 0.668 and shares an unusually high short-line-writing profile. Five other core matches have no auxiliary signal at the 90th percentile; for those, the interface states that no implemented evidence channel passes the display gate.

## 6. Research Atlas as a Result Surface

The public atlas is designed around four user questions: What distinguishes this label's observed lyrics? Which lyric matches reappear often enough to show by default? What do the two profiles measurably share? How repeatable is the match? The interface therefore presents characteristic words, common dictionary-estimated written endings, a short written-ending echo result, three writing-form indicators, and concise match evidence. Method details are collapsed.

By default, only the 16 matches selected in at least half of song bootstraps are shown. The remaining 70 two-representation-stable matches sit behind an explicit 'Show exploratory matches' control and appear dashed. Edge colour encodes the strongest gated explanation; edge width encodes bootstrap frequency. Selecting a node labels only that node and its visible neighbours. Selecting an edge shows the exact shared vocabulary, written-ending pattern, or form descriptor and the frequency with which the match reappeared.

The interface intentionally does not foreground the number of chunks, labels, or edge categories. Those quantities belong in the paper and methods package, not beside every node. It also avoids keyword occurrence search because locating a literal word can be done without a semantic model. The atlas instead exposes corpus-relative keyness, distributional matching, and resampling uncertainty - outputs that cannot be reproduced with a browser find command.

## 7. Discussion

### 7.1 An Embedding Is an Intermediate Representation

The main methodological lesson is that vectorization is not itself a downstream contribution. BGE-M3 transforms chunks into a shared space, but scholarly value appears only after the representation is evaluated, aggregated, perturbed, interpreted, and connected to a question. The local benchmark prevents model reputation from substituting for corpus evidence. The exact-text sensitivity test reveals whether duplicated attribution drives proximity. The bootstrap shows whether local matches depend on the observed song sample. The profile and pair-evidence layers translate numerical proximity into claims a reader can inspect.

The benchmark's negative result strengthens this position. Character TF-IDF beats dense BGE-M3 on the weakly supervised task, while fusion beats both. A defensible next retrieval system should therefore combine dense and lexical evidence and evaluate it against blinded human relevance judgements. The current result supports complementarity, not a final encoder-selection claim.

### 7.2 'Relationship' Is Too Strong for a Lyric-Matching Edge

A network edge is rhetorically powerful. Without a clear boundary, readers may treat it as friendship, collaboration, influence, regional affiliation, or shared genre. None of those claims follows from lyric-text similarity. This study therefore uses *match* or *proximity* for model edges and reserves *relationship* for externally verified real-world evidence.

The same discipline applies to profiles. Overrepresented words are not favourites. A frequently observed written final is not a preferred performed rhyme. A place reference is not an origin. A model needs external identity resolution, source validation, and human annotation before such claims become available.

### 7.3 Uncertainty Changes the Visual Argument

If all 86 representation-stable matches were drawn equally, the graph would imply a uniform level of support that the bootstrap contradicts. Showing only 16 by default makes the visual argument smaller but more honest. It also changes how an edge explanation should be read. A 99th-percentile lexical overlap can help explain a match, but it does not repair low resampling frequency. Conversely, a frequently reselected match can remain semantically unexplained. Reliability and interpretability are distinct axes.

### 7.4 Downstream Tasks Enabled but Not Yet Claimed

The same framework can support richer tasks after appropriate validation. Semantic concept retrieval should compare dense, lexical, and hybrid systems on human-labelled topic queries rather than literal word presence. Named-entity and cultural-reference analysis requires a complete span-level gold set, entity linking, literal-versus-metaphorical context decisions, and inter-annotator agreement. A current screened reference ledger contains candidates, but occurrence-level two-reviewer validation is incomplete, so no named-entity result is released here.

Rhyme analysis should add contextual pronunciation, polyphone resolution, internal rhyme, regional speech, and audio alignment before discussing performed rhyme or flow. A writing assistant or rap generator would require separate evaluation of topic fit, rhyme constraints, novelty, memorization, safety, and human usefulness, and should avoid imitation of a living artist. These tasks are a research programme, not extra buttons to place in the present interface.

## 8. Limitations

The corpus is a frozen and unevenly collected source snapshot rather than a representative census of Chinese rap. Source-credit strings are not complete identity records. The cleaning pipeline can detect declared structural problems and withhold ambiguous cases, but it cannot guarantee that every title and credit matches an authoritative discography without external verification.

BGE-M3 has not yet been compared with alternative multilingual encoders or blinded human judgements of Chinese rap semantic similarity. The completed benchmark uses same-song membership as weak supervision. Its fusion result is promising, but it is not a human-aligned retrieval benchmark.

The two representation conditions control exact hashes only. They do not capture paraphrase, near duplication, hooks with small edits, or source-collection bias. Reciprocal top-five membership is also a discrete decision around a rank boundary. The generally modest bootstrap frequencies demonstrate the resulting sampling sensitivity.

Characteristic words indicate relative overrepresentation in the observed corpus. Dictionary-derived written endings do not resolve polyphones, dialect, code-switched pronunciation, delivery, tone-melody interaction, internal rhyme, or beat alignment. Form indicators are intentionally simple and should not be called style in a comprehensive artistic sense.

The auxiliary evidence channels are calibrated against corpus pairs within support strata but are not human-rated explanations. A passing signal is evidence consistent with a match, not a causal explanation of the embedding. Human evaluation should test whether readers find the profiles accurate, understandable, and useful, and whether displayed evidence improves pairwise similarity judgements.

Copyright constraints prevent redistribution of full lyrics. The public artifact releases aggregates and short lexical items only. Reproducibility therefore depends on code, hashes, derived summaries, and controlled access to the underlying corpus rather than unrestricted lyric publication.

## 9. Conclusion

This study reframes Chinese rap lyric mapping as a movement from embeddings to evidence. BGE-M3 supplies a multilingual dense representation, but the completed findings come from what follows: corpus-internal benchmarking, duplicate-aware aggregation, exact-shared-text sensitivity, reciprocal-neighbour rules, corpus-relative lexical profiles, dictionary-estimated written-ending and form measures, calibrated pair evidence, and song-level bootstrap auditing.

The results are deliberately mixed. Global geometry is highly similar across two text treatments, yet local graph membership and song-resampling repeatability are much weaker. Only 16 of 86 representation-stable matches reappear in at least half of 250 bootstraps, and none reaches 0.80. Dense BGE-M3 trails character TF-IDF on the weak retrieval task, while their fusion performs best. These findings argue for multiview analysis and visible uncertainty rather than a single model score.

The resulting atlas makes the completed evidence usable: a reader can see what distinguishes a source-credit-labelled corpus slice, what independently measured evidence two matched profiles share, and how often the match returns. It also abstains when no implemented evidence channel passes the display gate. That combination - useful output, explicit uncertainty, and strict claim boundaries - is the project's principal contribution to computational lyric study and digital humanities.

## Data Availability

Full lyrics cannot be redistributed in the public research package because they may be copyrighted and were collected from heterogeneous sources. Publicly shareable code, aggregate results, manifests, cryptographic hashes, figure data, and the self-contained atlas have been prepared for deposit in an accompanying repository. Controlled access to non-public derived text may be considered subject to source terms, copyright review, and an approved data-use process. The verified repository URL or DOI, licence, and final access statement must be inserted before submission.

## AI Disclosure Statement

This manuscript was prepared with the assistance of OpenAI Codex [insert the exact deployed model/version and access date before submission]. The tool was used to inspect local analysis artifacts, draft and edit English prose, generate analysis and document-building code, and assist with figure production. The human authors remain responsible for verifying all calculations, interpretations, citations, figures, and statements before submission.

## Funding

[Insert the authors' verified funding statement, or: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.]

## Conflict of Interest

[Insert the authors' verified conflict-of-interest statement.]

## Author Contributions

[Insert verified CRediT roles for all human authors. AI systems must not be listed as authors.]

## References

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and Liu, Z. (2024). M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. In *Findings of the Association for Computational Linguistics: ACL 2024*, pp. 2318-2335. Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.findings-acl.137.

Karsdorp, F., Manjavacas, E., and Kestemont, M. (2019). Keepin' it real: Linguistic models of authenticity judgments for artificially generated rap lyrics. PLOS ONE, 14(10): e0224152. https://doi.org/10.1371/journal.pone.0224152.

Kim, H. and Akama, T. (2024). A Computational Analysis of Lyric Similarity Perception. arXiv:2404.02342. https://doi.org/10.48550/arXiv.2404.02342.

Langenhorst, J., Frommherz, Y., and Meier-Vieracker, S. (2023). Keyness in song lyrics: Challenges of highly clumpy data. *Journal for Language Technology and Computational Linguistics*, 36(1): 21-38. https://doi.org/10.21248/jlcl.36.2023.236.

Liu, J., Dong, H., Yuan, J., Ma, H., and She, A. (2024). Linguistic tone in Chinese rap: an interdisciplinary approach. *Journal of New Music Research*, 52(4): 265-284. https://doi.org/10.1080/09298215.2024.2329075.

Malmi, E., Takala, P., Toivonen, H., Raiko, T., and Gionis, A. (2016). DopeLearning: A Computational Approach to Rap Lyrics Generation. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 195-204. https://doi.org/10.1145/2939672.2939679.

Monroe, B. L., Colaresi, M. P., and Quinn, K. M. (2008). Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict. Political Analysis, 16(4): 372-403. https://doi.org/10.1093/pan/mpn018.

Parada-Cabaleiro, E., Mayerl, M., Brandl, S., Skowron, M., Schedl, M., Lex, E., and Zangerle, E. (2024). Song lyrics have become simpler and more repetitive over the last five decades. Scientific Reports, 14: 5531. https://doi.org/10.1038/s41598-024-55742-x.

Sun, Y., Li, L., Liu, Q., and Yeung, D.-Y. (2023). SongRewriter: A Chinese Song Rewriting System with Controllable Content and Rhyme Scheme. In Findings of the Association for Computational Linguistics: ACL 2023. Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.findings-acl.814.

Wang, T. (2025). The code of the streets in Beijing: Style-shifting and changing personae in the performance of Beijing male rappers. *Language in Society*, 54(1): 29-55. https://doi.org/10.1017/S0047404523000799.

Zhang, L., Zhang, R., Mao, X., and Chang, Y. (2022). QiuNiu: A Chinese Lyrics Generation System with Passage-Level Input. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 76-82. https://doi.org/10.18653/v1/2022.acl-demo.7.

## Figure Legends and Alt Text

Fig. 1 Embedding-to-evidence workflow. The encoder is one component; downstream aggregation, robustness checks, evidence gates, and abstention create the reported outputs

Alt text: A left-to-right pipeline begins with cleaned lyric chunks and BGE-M3 or character n-gram representations. It branches into retrieval benchmarking and label-level aggregation. The aggregation branch applies duplicate weighting, shared-text exclusion, reciprocal-neighbour matching, interpretable lexical, written-ending, and form profiles, and song bootstrap checks. Outputs are a benchmark, profiles, bootstrap-audited lyric matches, and an atlas.

Fig. 2 Encoder sanity benchmark across five retrieval metrics. Same-song membership supplies weak supervision and is not a human semantic relevance judgement

Alt text: Grouped horizontal bars compare BGE-M3 dense, character n-gram TF-IDF, and equal-weight fusion on MRR, Recall at 1, 5, and 10, and nDCG at 10. Fusion is highest on every metric; TF-IDF is second; BGE-M3 is lowest.

Fig. 3 Song-bootstrap selection frequencies for the 86 original two-representation matches. The dashed line marks the atlas default threshold of 0.50

Alt text: A sorted dot plot shows selection frequency for 86 matches. Most dots lie below 0.50. Sixteen reach or exceed 0.50, three reach 0.70, and none reaches 0.80. The maximum is 0.712 and the median is 0.280.

Fig. 4 Moderately reselected core of 16 lyric matches among 25 source-credit labels. Edge colour shows the strongest support-gated auxiliary signal; dashed grey edges have no passing signal. Width shows song-bootstrap selection frequency

Alt text: A network of 25 labelled nodes connected by 16 edges. Purple edges represent shared distinctive vocabulary, teal edges similar lyric structure, orange edges represent written-ending similarity, and dashed grey edges indicate matches without a passing auxiliary signal. Thicker edges were selected more frequently in 250 song bootstraps.
