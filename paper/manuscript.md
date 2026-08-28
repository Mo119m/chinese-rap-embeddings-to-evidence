# Language, Reference, and Written Rhyme: Evidence-Grounded Downstream Analysis of Chinese Rap Lyrics

[Author name(s) — to be completed]

[Affiliation(s) — to be completed]

[Corresponding author email — to be completed]

## Structured Abstract

**Purpose:** This study asks how Chinese rap lyrics form recognizable repertoires through language, cultural reference, and dictionary-estimated written rhyme, and how multilingual embeddings become evidence.

**Design/methodology/approach:** Three downstream tasks are constructed from a frozen corpus: held-out-song source-label retrieval, provisional named-entity recognition (NER) with typed cultural networks, and song-held-out prediction of the next written-ending family. BGE-M3 is compared with character 2–5-gram term frequency–inverse document frequency (TF–IDF) and untuned fusion. Exact and near-duplicate text is grouped or excluded; uncertainty uses paired label/component or song-cluster bootstrap. NER associations add Jeffreys bounds and false-discovery-rate control; extraction accuracy awaits dual review.

**Findings:** Dense–lexical fusion achieves macro mean reciprocal rank (MRR) 0.447 (95% confidence interval [CI] 0.414–0.481), exceeding TF–IDF by 0.031 (0.021–0.042); a sixfold training-only sensitivity preserves this advantage. Shared-text exclusion contracts the NER inventory from 33 to 22 provisional surfaces; six source-label/reference and four co-mention edges survive uncertainty and false-discovery-rate gates, but no precision, recall, or F1 score (the harmonic mean of precision and recall) is claimed. The written-ending context model reaches Top-3 0.695 (0.685–0.705), improving on Markov by 0.050 (0.044–0.055); source-label conditioning has no supported benefit.

**Originality:** The study makes post-representation task design, leakage control, typed explanation, uncertainty, and abstention the analytical contribution.

**Contribution to the field of Digital Humanities:** It supplies a reproducible framework for culturally situated lyric representations without publishing copyrighted lyrics or mistaking textual association for biography or social relation.

Keywords: Chinese rap; digital humanities; lyrics; information retrieval; BGE-M3; named entity recognition; cultural networks; written rhyme; uncertainty; abstention

## 1. Introduction

A text embedding is an intermediate representation, not a cultural finding. It becomes useful evidence only after researchers specify what the vector must do, define the unit on which it will be tested, compare it with simpler alternatives, control leakage, quantify uncertainty, and explain when the system should abstain. These requirements are easy to obscure in an interactive map: a screen can place two performers close together even when the underlying proximity is unstable, caused by repeated text, or semantically opaque. The visualization then appears to answer a question that the method never posed.

Chinese rap makes this problem unusually visible. Its lyrics combine Mandarin with English and other linguistic material, names and cultural references, repeated hooks, non-standard spellings, line-final phonetic patterning, and metadata that may conflate individuals, groups, aliases, and collaborations. No single representation can safely turn these heterogeneous signals into a claim about “style”. Work on authorship representation shows that topic and proper nouns can contribute strongly to apparently stylistic discrimination (Sundararajan and Woodard 2018; Wang *et al.* 2023). A source-label match may therefore reflect subject matter, named worlds, repeated expressions, writing form, or combinations of these—not a pure and stable personal essence.

We address this problem by organizing one corpus around a clear research theme: **how do Chinese rap lyrics form recognizable lyrical identities through language, cultural reference, and dictionary-estimated written rhyme?** “Identity” here means an aggregate, corpus-relative profile attached to a source-credit label. It does not mean a verified person, biography, hometown, belief, affiliation, influence, friendship, or collaboration. We use *lyrical repertoire* for this bounded object.

The article makes three linked contributions. First, strict fixed-corpus retrieval compares frozen BGE-M3, character 2–5-gram term frequency–inverse document frequency (TF–IDF), and untuned fusion after removing each query song and detected duplicates from candidate profiles. Second, a Chinese-rap named-entity recognition (NER) schema, two candidate baselines, an 800-occurrence dual-review package, and a provisional cultural-reference network make every edge textual and typed; incomplete human review precludes precision, recall, or an F1 score (the harmonic mean of precision and recall). Third, a 17-class model predicts strictly adjacent dictionary-estimated written endings, separates continuation from switching, tests the value of source-label input, and supports validation-defined abstention.

These tasks capture complementary dimensions. Retrieval asks what a held-out song resembles in the corpus. NER asks which named cultural worlds are referenced and how those references aggregate. Written-ending prediction asks how line-final form changes across a sequence. The companion results interface exposes these outputs as three actions—exploring a repertoire, a cultural reference, or a possible next written-ending class—but the interface does not substitute for the paper's methods or evidence.

The research questions are:

**RQ1.** Which representation best retrieves held-out source-label lyrical-repertoire evidence, and does dense–lexical fusion improve over either component under strict duplicate controls?

**RQ2.** Which cultural references can presently be released without conflating automated agreement with human validity, and what annotation work is required before model-performance or network-centrality claims are justified?

**RQ3.** How predictable is the next dictionary-estimated written-ending family, what sequential context improves on global-frequency and first-order-transition baselines, does source-label conditioning add predictive value, and how does abstention change reliability?

**RQ4.** As a methodological design synthesis, how can meaning, reference, and written form be joined into evidence-graded source-label profiles without turning textual association into claims about real-world persons or relationships?

The central argument is methodological: **representation → downstream task → leakage control → evaluation → explanation → abstention**. The value of BGE-M3 in this project is determined by what happens after embedding, not by the existence of the vectors or the model's published benchmarks.

## 2. Related Work and Analytical Position

### 2.1 Chinese Rap as Situated Repertoire

Chinese rap is not simply an English-language genre transferred into Chinese text. Ethnographic and lyric-centered studies show performers reworking global hip-hop resources through regional language, Putonghua, English, slang, local knowledge, and place to construct locally meaningful positions (Barrett 2012; Wang 2013; Liu 2014). Work on Higher Brothers further separates written multilingual material from the vocal intelligibility and production through which local, national, and global identities are negotiated (Liu 2021). These studies motivate *lyrical repertoire* as a situated bundle of recurring corpus evidence, while ruling out authenticity scores or a fixed real-world identity inferred from text alone.

Named places and objects can also participate in stylized persona construction, but only together with stance and performance context (Baioud 2024). A corpus-level place association is therefore culturally relevant without being biographical: it can show that retained lyrics invoke a place more than expected, not that a credited source lives there, belongs there, or has a social relationship with another source.

### 2.2 Representation Is Not Style

BGE-M3 was designed for multilingual, multi-function, and multi-granularity retrieval and supports a dense representation suitable for mixed-language passages (Chen *et al.* 2024). Those properties make it a plausible starting point for Chinese lyrics containing Chinese and English material of variable length. They do not validate it on Chinese rap. A model trained for broad retrieval may encode topical, lexical, and named-entity signals that are useful for finding related text while remaining unsuitable for claims about individual artistic style.

That distinction follows research on authorship representation. Sundararajan and Woodard (2018) demonstrate the influence of common and proper nouns in attribution and use masking to expose topic interference. Wang *et al.* (2023) similarly argue that success on authorship prediction does not establish disentanglement of style from content. We therefore evaluate a *lyrical-repertoire retrieval* task rather than “rapper identification”, compare dense representation with a strong surface-form baseline, and interpret fusion as complementarity rather than proof that either channel isolates style.

### 2.3 Named Entities and Cultural Networks in New Domains

Chinese NER is sensitive to domain, script, tokenization, and annotation conventions. Work on Chinese social-media NER shows why models built for edited text should not be assumed to transfer to informal language (Peng and Dredze 2015). Lexicon-enhanced architectures such as FLAT demonstrate that character and word information can be combined effectively (Li *et al.* 2020), while distantly supervised NER research shows that dictionary matches are incomplete and noisy rather than ready-made gold annotations (Yang *et al.* 2018). Domain-matched corpora also require explicit annotation-quality reporting (Jiang *et al.* 2022).

NER errors determine which actors and places become visible; unequal error can therefore distort cultural knowledge (Lassen *et al.* 2024). Literary-network work shows that nodes, windows, and relations must follow the research question (Blessing *et al.* 2017), and that co-occurrence and embedding similarity are distinct evidence layers (Rafaeli *et al.* 2026). In Chinese rap, language and place can localize a repertoire (Liu 2014), while named references gain meaning through stylization and performance (Baioud 2024). We therefore keep source-label/reference and same-song co-mention edges separate, preserve provenance and status, and call neither social nor biographical.

### 2.4 Rhyme as Text-Derived and Performance-Sensitive Evidence

Computational rhyme research has modeled phonetic similarity, imperfect rhyme, internal rhyme, and artist-level rhyme profiles (Hirjee and Brown 2009). Chinese lyric-generation systems likewise make rhyme representations explicit: DeepRapper models rhyme alongside beat symbols (Xue *et al.* 2021), while SongRewriter uses vowel modeling and decoding constraints (Sun *et al.* 2023). Chinese-rap scholarship shows why a text-only target must remain narrower. Rhyme participates in persona only within a multimodal analysis (Lin and Wang 2024); performers can create imperfect and Chinese–English rhymes through regional and phonetic variation that fixed pinyin classes miss (Wang and Lin 2024); and audio-based work links tonal realization and English use to flow (Liu *et al.* 2023). Without aligned audio, the present study cannot observe performed pronunciation, tone, stress, rhythm, cadence, beat alignment, or delivery. Its target is therefore named precisely: a **dictionary-estimated written-ending family**.

## 3. Corpus, Cleaning, and Evidence Boundary

The frozen private snapshot contains 7,211 songs and 22,128 lyric chunks. Deterministic metadata and text gates reconcile every canonical key to a clean-text sidecar, retain 21,553 eligible chunks, and withhold 575 metadata-only chunks. Analyses preserve one song record, split before expanding to chunks, lines, or entity occurrences, and keep identical-content groups within one partition or exclude cross-partition copies; 405 exact groups span 921 songs. Missing or doubtful metadata are never guessed.

PD-002 reconstructed the legacy cleaner: artist/text deduplication removed 2,894 chunks and 177 song IDs; 131 are conservative duplicate records and 46 require review. The live Drive sheet matched 26,833 keys; all 33 substantive cell differences were adjudicated. Canonical gates explain 7,214/22,132 upstream versus 7,211/22,128 analyzed. Results remain snapshot-conditional.

The export is a convenience snapshot rather than a census. It lacks stable per-track identifiers and verified records for source platforms, acquisition dates, temporal coverage, sampling, transcription origin, and rights or ethics basis. Those owner-supplied facts remain required; none is inferred from lyrics or filenames. Full reconciliation, exception totals, and checksums appear in the public supplement.

The source-credit label is corpus provenance, not verified identity; the study does not attempt global person resolution. Retrieval predicts that label, NER identifies lyric-reference surfaces, and rhyme fingerprints summarize label-attached lines. The public release excludes lyric text, full lines, song/chunk identifiers, membership rows, row-level lyric-content hashes, embeddings, and reviewer contexts. File-level checksums remain as non-content integrity metadata.

Table 1 summarizes the populations. Individual tasks apply stricter eligibility rules, so their denominators differ.

| Population | Count | Role |
| --- | ---: | --- |
| Canonical songs | 7,211 | Frozen song grain |
| Canonical / sidecar lyric chunks | 22,128 | Frozen chunk grain; exact key reconciliation |
| Eligible clean-text chunks | 21,553 | Model-eligible text |
| Withheld metadata-only chunks | 575 | Not analyzed as lyrics |
| Source-credit labels | 241 | Corpus provenance labels |
| Labels with at least five songs | 226 | Support audit |
| Exact song-content groups spanning multiple songs | 405 (921 songs) | Duplicate/leakage constraint |
| Graph-eligible labels used in all three downstream tasks | 204 | Fixed comparison frame |

Table 1 Frozen corpus and downstream comparison populations

## 4. Shared Representation and Experimental Design

### 4.1 Why BGE-M3, and What Is Frozen

BGE-M3 is selected because the primarily Chinese corpus contains mixed-script passages, chunk lengths vary, and the model is designed for retrieval. We use only its dense head. `FlagEmbedding.BGEM3FlagModel` generated vectors offline from pinned `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`, using FlagEmbedding 1.4.0, maximum length 2,048, batch size two, 1,024 dimensions, and L2 normalization (Chen *et al.* 2024). The embedding contract records checkpoint, configuration, row-map, and vector hashes. Supplementary Table S1 records the exact stack; historical device and realized mixed precision were not captured and are not retrospectively asserted.

The checkpoint is not fine-tuned on the corpus and its published benchmark performance is not imported as evidence. Instead, the vectors enter a downstream retrieval design with an explicit target, baseline, denominator, and uncertainty. This matters because a broadly capable multilingual encoder can still be weaker than character-form matching for a particular corpus-internal task.

### 4.2 Units, Duplicate Components, and Fitting Boundaries

All evaluation is song-aware. Retrieval holds out the complete song and any detected variants from every affected repertoire profile. Rhyme modeling assigns songs globally to train, validation, and test before line events are constructed. NER candidates are aggregated by duplicate-controlled full-song lyric-content units; its future gold split is also specified at song level.

Two normalization levels are used for leakage checks. Exact clean-text hashes catch identical chunks. For whole songs or written lines, NFKC-normalized, case-folded text with punctuation and spaces removed is represented as a character-trigram set. Candidate pairs with Jaccard similarity at least 0.80 are treated as near duplicates after exact verification of the set overlap. Connected items form one duplicate component. Where repeated variants remain inside a repertoire, the component receives total weight one so copies cannot dominate an aggregate.

Fitting and selection follow the task. BGE-M3 is frozen. Retrieval TF–IDF is transductive: vocabulary and inverse document frequencies are estimated on the fixed unlabeled evaluation corpus, never using source labels, and the label profiles remain group-held-out. Fusion weight is not tuned. Rhyme vocabularies, priors, classifiers, regularization, and abstention thresholds are learned from train or validation only; test is used once for final evaluation. The NER transformer is pinned and not trained on the corpus, while release thresholds are reproducibility gates rather than learned claims of correctness.

The implementation amends the frozen plan's generic train-only vectorizer rule. PD-001 records that character TF–IDF was estimated transductively on the fixed unlabeled corpus; the original contract remains preserved. Labels, relevance, and outcomes never enter vocabulary, inverse-document-frequency, profile, or fusion-weight selection, but query text influences corpuswide vocabulary and IDF. The estimand is fixed-corpus leave-one-song-out repertoire retrieval, not deployment to a future corpus. The departure is reported rather than retrospectively rewritten.

A sixfold sensitivity analysis fits the same TF–IDF specification only on training-fold texts and compares it with a matched all-corpus fit while holding duplicate-component folds and label profiles fixed (Supplementary Methods S2).

### 4.3 Metrics and Uncertainty

Each result names its inferential unit. Retrieval has one relevant source label per query and reports reciprocal rank, Recall@1/5/10, and normalized discounted cumulative gain at rank 10 (nDCG@10). Its primary estimates are duplicate-component-adjusted macro means across source labels, preventing high-volume labels from determining the conclusion. A paired two-stage bootstrap uses 5,000 fixed-seed replicates. Each replicate samples 204 source-label occurrences with replacement and then independently resamples that occurrence's duplicate components; a label selected twice therefore receives two independent inner draws. The same component draw indexes every system and metric, preserving pairing for model differences (Savoy 1997).

Rhyme prediction reports top-1/3/5 accuracy, mean reciprocal rank (MRR), negative log likelihood (NLL), ten-bin expected calibration error (ECE), and coverage. Its 95% intervals use 2,000 resamples of test songs, preserving dependence among lines from the same song. Paired model differences reuse the same song-cluster resamples. Per-label macro results require at least 20 leakage-safe test events.

The NER extraction layer reports counts and cross-method candidate agreement only. Without adjudicated occurrence truth, agreement is not precision, recall, or an F1 score. The downstream network is evaluated separately: source-label/reference rates receive Jeffreys intervals and all tested label/entity pairs are controlled by Benjamini–Hochberg false-discovery-rate (FDR) correction; co-mentions use hypergeometric tests and a separate FDR family. These intervals quantify fixed-corpus song-unit association, not entity-extraction accuracy or external-population generalization.

Fig. 1 summarizes the common evidence controls and the task-specific estimands and claim boundaries.

[[FIGURE:figure_1_research_design.png]]

Fig. 1 Study design. A frozen cleaned corpus enters shared song-aware partition or holdout rules, duplicate controls, task-specific fitting boundaries, a prohibition on using test outcomes for fitting or selection, and aggregate-only public release. Retrieval TF–IDF is fit transductively on the fixed unlabeled evaluation corpus while labelled outcomes remain unused. BGE-M3 is evaluated only in held-out-song retrieval; cultural-reference extraction aggregates duplicate-controlled song units and awaits a future gold split; written-rhyme prediction uses a fixed song-level split and terminal-Han pinyin-final families. Every output is paired with its permitted interpretation.

Alt text: A top-to-bottom research pipeline begins with a frozen Chinese-rap lyric corpus and a shared evidence-control block. Three branches follow: repertoire retrieval using BGE-M3, character TF–IDF, and fusion; provisional cultural-reference extraction using lexicon and contextual Chinese NER evidence; and written-rhyme prediction using terminal-Han pinyin-final families and context models. “Lyrical identity” is defined as a corpus-relative source-label repertoire; the branches reject verified-person, authorship, social-relation, and performed-rhyme claims.

## 5. Downstream Task 1: Explainable Lyrical-Repertoire Retrieval

### 5.1 Task Construction

For each eligible song \(q\), the task ranks the 204 candidate source-credit labels and asks where the song's observed label appears. The query is the complete song assembled from eligible clean chunks after exact text shared across labels has been removed. It must contain at least 50 NFKC-normalized alphanumeric characters. The final strict population contains 5,455 song queries forming 5,430 independent exact/near-duplicate components.

The relevance definition is intentionally narrow. A correct retrieval means that the held-out song is consistent with the remaining text assigned to its source label. It is not a human judgment of semantic similarity and not proof of authorship. Nevertheless, it supplies a reproducible fixed-corpus task in which every model must recover label-level repertoire evidence for a held-out song.

### 5.2 Strict Holdout and Profile Construction

For each query, the full song is removed from its true profile. Its entire exact/near-duplicate component is then removed from every source-label profile it touches, including cross-label variants. The audit found one exact whole-song pair and 25 additional near-duplicate pairs; three of these 26 pairs cross source labels. Each retained duplicate component contributes total weight one within a label. At least five independent training components must remain after holdout.

Cross-label exact shared text is excluded before representations are constructed. This removes 4,877 eligible membership rows overall and 522 within the evaluated-song population, leaving zero exact cleaned-text hashes shared between strict labels. Of 6,848 otherwise eligible songs, 1,229 have no label-specific chunk after this filter and 164 more fall below the length threshold. These exclusions make the task conservative and change its estimand: it evaluates label-specific residual evidence rather than shared hooks or copied text.

### 5.3 Dense, Lexical, and Fused Systems

For BGE-M3, chunk vectors are averaged within each song using frozen comparison-text weights and then normalized. Let sᴰ(q, ℓ) be cosine similarity between query song q and candidate label profile ℓ, where the profile is the equal-component mean of its eligible training-song vectors after the query group is removed.

The lexical system represents complete song text with character 2–5-gram TF–IDF using `min_df=3`, sublinear term frequency, L2 normalization, and a maximum of 150,000 features. Character n-grams avoid committing to one Chinese word-segmentation scheme and capture partial strings, morphology-like patterns, mixed script, and punctuation-normalized surface form. Candidate label profiles use the same strict group-held-out component aggregation. Their cosine scores are sᴸ(q, ℓ).

Because dense and lexical cosine distributions have different scales, fusion standardizes each system's 204 candidate-label scores within a query:

\[
z_{q\ell}^{D} = \frac{s_{q\ell}^{D}-\mu_q^{D}}{\sigma_q^{D}}, \qquad
z_{q\ell}^{L} = \frac{s_{q\ell}^{L}-\mu_q^{L}}{\sigma_q^{L}},
\]

\[
s_{q\ell}^{F}=\tfrac{1}{2}z_{q\ell}^{D}+\tfrac{1}{2}z_{q\ell}^{L}.
\]

The equal weight is fixed without looking at evaluation outcomes. A raw-cosine average is retained as a scale-sensitivity diagnostic. Two further diagnostics remove the non-exact near-duplicate guard or restore shared text while keeping the same query population.

### 5.4 Results

Character TF–IDF outperforms BGE-M3 dense retrieval on every primary metric, but their standardized fusion is better than either component (Table 2; Fig. 2). Fusion reaches macro MRR 0.447 (95% confidence interval [CI] 0.414–0.481) and Recall@10 0.611 (0.577–0.646). Its MRR advantage over TF–IDF is 0.031 (paired 95% CI 0.021–0.042); its Recall@10 advantage is 0.050 (0.033–0.066). Against BGE-M3, the fusion gains are larger: MRR +0.127 (0.106–0.147) and Recall@10 +0.103 (0.079–0.127). All five fusion-versus-component intervals are above zero.

Under sixfold training-only fitting, fusion MRR remains 0.440 versus 0.405 for TF–IDF (difference +0.035, 95% CI 0.024–0.046). Matched transductive exposure adds only 0.004 MRR to either TF–IDF or fusion (+0.003963 and +0.003681, respectively); this small optimism cannot account for the fusion gain.

| System | MRR | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 dense (strict) | 0.320 [0.295, 0.346] | 0.226 [0.199, 0.252] | 0.407 [0.377, 0.437] | 0.508 [0.478, 0.538] | 0.353 [0.326, 0.380] |
| Character 2–5-gram TF–IDF (strict) | 0.416 [0.381, 0.451] | 0.337 [0.301, 0.374] | 0.492 [0.453, 0.530] | 0.562 [0.523, 0.601] | 0.440 [0.405, 0.477] |
| Equal-weight z-score fusion (strict) | **0.447 [0.414, 0.481]** | **0.362 [0.328, 0.398]** | **0.528 [0.493, 0.564]** | **0.611 [0.577, 0.646]** | **0.476 [0.443, 0.511]** |

Table 2 Duplicate-component-adjusted source-label macro retrieval metrics across 5,455 held-out songs and 204 labels. Brackets show paired, occurrence-wise two-stage bootstrap 95% intervals (5,000 replicates).

Standardization improves MRR slightly over raw-cosine averaging (+0.005, 0.001–0.009); Recall@1/5 remain inconclusive and the Recall@10 interval barely excludes zero. Removing the near-duplicate guard produces small optimistic Recall@10 and nDCG@10 shifts. Restoring exact cross-label shared text similarly raises MRR by 0.005, Recall@10 by 0.008, and nDCG@10 by 0.006. These diagnostics support the strict controls; they do not show that a less guarded system generalizes better.

Performance varies substantially by repertoire: some source labels are retrieved nearly perfectly, whereas others have median true-label ranks below ten. Fusion is not a universal artist recognizer; semantic and surface-form signals rescue different queries. PO8 has the largest released label-level MRR lift over its better component (+0.224), while some easy labels gain nothing. The interface explains each aggregate recommendation and label-level repeatability without exposing query text or claiming a personal relationship.

[[FIGURE:figure_2_retrieval_benchmark.png]]

Fig. 2 Strict held-out-song source-credit-label lyrical-repertoire retrieval. Panel A shows duplicate-component-adjusted macro MRR, Recall@1/5/10, and nDCG@10 for BGE-M3, character TF–IDF, and their untuned per-query z-score fusion. Panel B shows paired fusion-minus-baseline differences. Whiskers are 95% intervals from 5,000 literal occurrence-wise paired two-stage bootstrap replicates over 5,455 queries and 204 source-credit labels.

Alt text: Two dot-and-whisker panels compare five retrieval metrics. In the model panel, fusion is highest on all metrics, character TF–IDF is second, and BGE-M3 is lowest. In the paired-difference panel, all ten fusion-minus-baseline intervals lie to the right of zero; gains over TF–IDF are smaller than gains over BGE-M3.

### 5.5 Corpus-Wide Descriptive Repertoire Companion

The interactive overview is a secondary descriptive analysis of Task 1, not another retrieval evaluation and not a visualization of the fusion scores in Table 2. It represents each of 204 support-eligible source-credit labels by a duplicate-component-weighted mean of its BGE-M3 lyric-chunk vectors. A sensitivity representation is recomputed after excluding every exact cleaned text shared across labels. A pair is retained only when each label ranks the other among its five closest cosine neighbours in both representations. Among 20,706 possible pairs, 86 satisfy this reciprocal rule, connecting 93 labels; the remaining 111 have no released reciprocal edge under the rule. A line therefore denotes corpus-internal BGE-M3 repertoire proximity, not collaboration, influence, genre, geography, biography, preference, popularity, or verified identity.

The 86 cross-treatment matches are far above a degree-controlled null. Holding the primary graph fixed and rewiring the sensitivity graph while preserving every label's degree produced a mean overlap of 4.52 edges (95% null range 1–9; maximum 13) across 10,000 replicates; none reached the observed 86 (add-one Monte Carlo *p* < 0.001). This tests specific adjacency agreement beyond degree structure, not whether each edge is independently significant or whether the graph recovers real-world cultural relations.

For the global layout, normalized centroids from both text treatments are combined and projected by deterministic principal-components analysis (PCA). The first two components retain 26.2% of profile variation. At five neighbours, trustworthiness is 0.785 and exact high-dimensional-neighbour retention is 0.173, versus a random expectation of 0.025. The map therefore preserves broad structure beyond chance but not every local neighbourhood: only a retained line defines a released match, while screen distance and the focused ego diagram's angle and radius remain non-analytical.

Repeatability is assessed with 250 fixed-seed resamples in which songs are sampled with replacement within each label and both reciprocal-top-five graphs are recomputed. Observed-edge selection frequency ranges from 0.072 to 0.712 (median 0.280); 16 of 86 edges, connecting 25 labels, meet the prespecified 0.50 display gate, and none reaches 0.80. This frequency is neither posterior probability nor statistical confidence. Auxiliary explanations are computed only after an edge exists: duplicate-controlled informative-Dirichlet distinctive wording, song-normalized written endings, and song-level writing-form probes. A signal is displayed only when it reaches the 90th percentile within its support stratum. Forty-eight edges have at least one such concordant signal; 38 remain semantic-only. These probes explain measurable common ground but do not causally decompose the embedding.

## 6. Downstream Task 2: Chinese-Rap NER and a Provisional Cultural-Reference Network

### 6.1 Schema and Evidence Status

The domain schema includes `PERSON_REFERENCE`, `GROUP_CREW_OR_ORGANIZATION`, `PLACE`, `BRAND_OR_PRODUCT`, `WORK_OR_MEDIA`, `EVENT`, `LANGUAGE_OR_DIALECT_REFERENCE`, `ETHNOCULTURAL_GROUP_REFERENCE`, `RAP_CULTURE_CONCEPT`, and `OTHER_CULTURAL_REFERENCE`, plus `NOT_ENTITY` and `UNCERTAIN`. A person reference may be real, stage-named, fictional, or mythic; it never automatically denotes a rapper. Reviewers label validity, type, referential status, normalized surface, linking status, confidence, and corrections in private context.

The current corpus has no completed occurrence-level human gold. The primary release provides two blank, complementary packages: an 800-occurrence corpuswide dual-review sample and a claim-directed audit containing all 157 unique occurrences supporting the six released label/reference and four co-mention claims. Both require independent dual review and adjudication; claim/control status is blinded in the claim-directed package, and zero decisions are complete. The claim audit completely covers its released-claim frame but is claim-conditioned, so neither package presently supports precision, recall, F1, inter-annotator agreement, or claim-confirmation rates.

### 6.2 Two Candidate Baselines

Baseline A is a screened 605-surface lexicon, of which 561 surfaces map to the target schema. Matching is case-sensitive and literal; ASCII/digit surfaces require non-ASCII-word boundaries. Overlap is resolved globally by longest span and then earliest start. Screening a surface establishes only that it is a plausible candidate, not that every occurrence is an entity in context.

Baseline B is the pinned `ckiplab/albert-tiny-chinese-ner` token classifier, executed through Transformers and PyTorch with configuration and weight hashes recorded in the release (Paszke *et al.* 2019; Wolf *et al.* 2020). It is a Traditional-Chinese general-domain model, used in 180-character windows with 40-character overlap. Overlapping window predictions are deduplicated. Because stage names, slang, creative spelling, and simplified Chinese create domain shift, model confidence is not treated as calibrated probability.

The comparison frame contains every unique line with a lexicon candidate plus a deterministic hash-ranked sample of 3,000 candidate-free lines: 23,177 lines. The lexicon produces 22,448 spans and the transformer 12,602; the full lexicon pass yields 37,983 source occurrences. Exact span/type agreement occurs 3,566 times, including 3,290 above confidence 0.80. Deduplicating identical line/span/type combinations gives 2,011 agreements and 1,888 high-consistency spans. Both units measure overlap between fallible methods, not correctness.

### 6.3 Human Review Design and Provisional Release Gate

The private annotation package samples 800 distinct occurrences stratified by proposed type and agreement state: exact agreement, overlapping agreement, type/boundary conflict, lexicon-only, and transformer-only. Two reviewers work independently; disagreements, uncertainty, corrected spans/types, or conflicting links require adjudication. A future gold release requires complete dual review, adjudication, agreement reporting, manifest reconciliation, and a song-level train/dev/test split.

Before review, corpuswide aggregates must pass exact span/type agreement, confidence ≥0.80, at least five strict occurrences in five duplicate-controlled songs and three labels, agreement on at least half of lexicon occurrences, and mean confidence ≥0.80. Every survivor receives a release audit. Named-person surfaces must also appear in a screened individual ledger; places and language surfaces retain only those narrow types. Ambiguous items and types without a release rule are withheld.

The primary graph then reapplies the same gate after two restrictions fixed independently of the observed association scores: the 204-label retrieval universe is used, and every exact cleaned-text hash observed under more than one canonical label anywhere in the eligible corpus is excluded. There are 2,187 such hashes. The primary inventory is required to be a strict subset of the frozen corpuswide candidate inventory so that denominator filtering cannot introduce a new surface merely by increasing its agreement rate.

### 6.4 Typed Network Construction

Only audited provisional entities enter the graph. Its two edge layers have different meanings and separate multiple-testing families:

1. `SOURCE_LABEL_TO_LYRIC_REFERENCE` tests every source-label/entity pair in the 204-label universe. Release requires the entity in at least five of at least ten eligible source-labelled song units, within-label coverage at least 5%, a Jeffreys-smoothed risk ratio of at least 1.50 against the rest of the graph universe, a conservative 95% risk-ratio interval lower bound above 1.0, and Benjamini–Hochberg FDR (q ≤ 0.05). If the label rate has a 95% Jeffreys interval [pᴸ⁻, pᴸ⁺] and the rest-of-corpus rate has [pᴿ⁻, pᴿ⁺], the reported constructed bound is [pᴸ⁻/pᴿ⁺, pᴸ⁺/pᴿ⁻]. It is a conservative ratio of marginal bounds, not a standard likelihood-based confidence interval for the risk ratio. The table retains raw lift, smoothed rates, this uncertainty bound, p, q, and a reliability class.
2. `SAME_SONG_LYRIC_REFERENCE_CO_MENTION` tests entity pairs over **all 5,681 eligible shared-text-excluded full-song units**, including songs with no released entity. Release requires at least five units across at least three source labels, lift at least 1.25, positive normalized PMI, and BH q ≤ 0.05.

Neither edge is called a collaboration, influence, affiliation, identity link, or social relation. Full-song content hashes define independent support but remain private.

### 6.5 Provisional Results and Required Abstention

The corpuswide inventory contains 33 provisional surfaces. Excluding exact cross-label shared text removes 539 of 1,474 strict occurrence rows and leaves 23 surfaces; the fixed 204-label universe leaves 22: 18 places, two language references, and two named-person surfaces. Legacy support/lift-only label/entity links fall from 85 to 40 after exclusion; uncertainty and BH-FDR retain six, all places. Legacy co-mentions fall from nine to one under the entity-bearing denominator; the corrected all-song denominator yields five basic-gate pairs and four BH-FDR releases. Table 3 and Figure 3 keep these units separate.

For example, 杭州 appears in 10 of 38 eligible shared-text-excluded song units labelled Tangoz (26.3%). Its Jeffreys-smoothed risk ratio against the rest of the graph universe is 74.12, with a conservative 95% interval of 26.87–186.44 and q=3.04 × 10⁻¹². This describes a concentrated lyric reference, not residence or preference. Two independently supported edges connect 上海 to GALI (8/36 units; risk ratio 13.17, q=0.00020) and 法老 (7/41; risk ratio 10.12, q=0.00496), enabling a shared-reference path without calling the labels socially related. Among entity pairs, 伦敦 and 巴黎 co-occur in five of 5,681 eligible song units across five labels (NPMI 0.500; q=4.21 × 10⁻⁵); this is a recurring textual frame, not a relation between places or persons.

Withholding is equally informative. 上帝 has abundant automated person-shaped agreement but is absent from the named-individual ledger and can include theological or figurative uses; it is not released as a person. 中南海 can denote a place, political metonym, or product brand; 桃源 can be geographic or figurative; 西山 has multiple possible referents. All are withheld until occurrence review resolves their senses. No label association involving a provisional person or language surface clears the song-unit uncertainty and multiplicity gates. The graph is therefore a small map of reproducible candidate evidence, not a completed NER benchmark.

| Stage | Count | Evidential meaning |
| --- | ---: | --- |
| Screened lexicon surfaces | 605 | Candidate inventory, not gold |
| Exact agreements | 3,566 occurrences / 2,011 unique-line spans | Cross-method agreement, not accuracy |
| Strict high-consistency agreements | 3,290 occurrences / 1,888 unique-line spans | Candidate evidence on two explicitly different units |
| Corpuswide provisional inventory | 33 surfaces | Frozen sensitivity inventory before shared-text exclusion |
| Shared-text-excluded primary inventory | 22 surfaces | 18 places, 2 language references, 2 named-person surfaces |
| Supported source-label/entity links | 6 | Jeffreys interval and BH-FDR gate; all are place references |
| Supported same-song co-mentions | 4 | All-song denominator, positive NPMI, and BH-FDR gate |
| Private dual-review tasks | 800 | Required path to human gold |
| Claim-directed audit | 157 unique occurrences | All support for ten released claims; 0 reviewed |

Table 3 NER readiness and statistically screened provisional cultural-network output. No precision, recall, or F1 is valid before dual review and adjudication.

[[FIGURE:figure_3_cultural_reference_evidence.png]]

Fig. 3 Provisional cultural-reference evidence after cross-label exact shared-text exclusion and statistical screening. Panel A keeps three analytical units separate. The candidate inventory falls from 33 corpus-wide entity strings to 23 after shared-text exclusion and to 22 in the fixed 204-label primary universe. Legacy source-label/entity links fall from 85 to 40 after shared-text exclusion; shrinkage, conservative interval, and Benjamini–Hochberg false-discovery-rate gates retain six source-label-to-place enrichments. Legacy co-mention sensitivity used an entity-bearing-song denominator (9 before exclusion; 1 after) and is shown separately from the corrected primary denominator of all 5,681 eligible song units (5 basic-gate candidates; 4 BH-FDR releases). Panel B plots all six released shrunken risk ratios with conservative 95% intervals and BH-adjusted q values. Panel C plots the four released same-song reference co-mentions by normalized pointwise mutual information (NPMI). Human review remains incomplete (0 completed gold reviews), so precision, recall, and F1 are not reported. All findings concern lyric references, not residence, preference, biography, collaboration, influence, or social relationships.

Alt text: Three evidence-control cards show separate progressions for entity strings, source-label/entity pairs, and co-mention pairs. The entity inventory decreases from 33 to 23 to 22. Label-to-reference links decrease from 85 to 40 to 6. The co-mention card separates the legacy entity-bearing denominator, which changes from 9 to 1, from the corrected all-5,681-song denominator, which changes from 5 to 4. A log-scale forest plot shows six released source-label-to-place enrichments; every conservative interval is above one. The strongest point estimates are 泰格西 to 湖南 and 黑麦 to 天津, while GALI and 法老 both link to 上海 with smaller but supported enrichments. A compact dot-bar panel shows four released same-song co-mentions: 伦敦–巴黎, 中文–英文, 上海–新疆, and 上海–巴黎. A note states that human gold is incomplete and that the edges are lyric-reference evidence, not biographical or social relations.

## 7. Downstream Task 3: Written-Rhyme Continuation

### 7.1 Line and Ending Construction

The rhyme task uses the same 204 source labels and the shared-text-exclusion population. Newline boundaries in cleaned source text define written lines. Empty lines, explicit section or credit headers, lines without Han characters, unclassifiable endings, and lines whose final non-space character is not Han are excluded from the primary estimand. This last rule prevents an earlier ambiguity in which the last Han character could be extracted from a line that actually ended in Latin script, a digit, or another symbol. The fixed task universe has 5,619 songs: 5,452 contribute 283,806 strict-Han-ending lines and 5,347 contribute 238,881 adjacent transitions. Its 52,152 repeat count is excess after first within label/song; repeats retain surviving positions and transitions preserve original adjacency. Reconnecting eligible pre-snapshot chunks adds 7,033 lines and 4,938 transitions, with task-universe total variation 0.001789 and switch-rate change 0.001390; prediction models were not retrained.

`pypinyin` 0.55.0 processes the complete Han sequence of each line using `Style.FINALS_TONE3`, `strict=True`, and neutral tone 5. Full-line context allows the dictionary system to disambiguate some polyphones. The last and penultimate Han syllables provide strict pinyin finals, tones, a two-syllable family pattern, and one of 17 tone-free ending families: A, O, E, IE/ÜE, AI, EI, AO, OU, AN, EN, ANG, ENG, ONG, I, U, Ü, and ER. Tone is retained as a feature but not as part of the target class. These deterministic normalizations are analytical conveniences, not an official or performance-sensitive rhyme standard.

### 7.2 Song-Level Split and Target Leakage Filter

Songs—including a song attached to more than one source label—are assigned globally to approximately 70/15/15 train/validation/test partitions by a deterministic multilabel-aware greedy algorithm followed by non-empty-partition repair. Every one of the 204 labels has at least one song in each split. The result is 3,951 training songs, 834 validation songs, and 834 test songs, with zero song overlap.

All normalized training lines, including first lines that are not prediction targets, form the validation leakage reference. Test targets are checked against the union of training and validation lines, so model-selection material cannot leak into final evaluation. An event is excluded when its target line exactly matches the relevant reference set or has verified character-trigram Jaccard similarity at least 0.80. Lines shorter than three normalized characters receive the exact check only. Validation filtering removes 944 targets (907 exact, 37 near), leaving 34,657 events. Test filtering removes 1,431 of 35,826 candidates (1,382 exact, 49 near), leaving 34,395 leakage-safe adjacent-line events from 787 songs. This is 96.01% end-to-end coverage of strict test candidates. Models never receive the target lyric text; filtering prevents copied or reissued lines from inflating sequence prediction.

### 7.3 Baselines and Context Models

The target yₜ is the family of the next written-line ending. The global-frequency baseline uses add-one-smoothed target frequencies estimated from training. The first-order Markov baseline conditions on yₜ₋₁ and uses a global empirical-Bayes backoff of strength 20.

The flat context model is multinomial logistic regression implemented with `SGDClassifier(loss='log_loss', average=True)` in scikit-learn (Pedregosa *et al.* 2011). One-hot features comprise the source-credit label; previous four ending families; previous raw pinyin final and tone; previous two-syllable family pattern; previous transition; current same-family run bucket; line-position bucket; and recent family diversity. The fitted feature space contains 922 features.

The released hierarchical model factorizes the next-class distribution into a continuation decision and a switch distribution. Let c = yₜ₋₁, let xₜ denote the context, and let g(xₜ) = P(yₜ = c | xₜ). A second classifier estimates hₖ(xₜ) = P(yₜ = k | yₜ ≠ c, xₜ) for k ≠ c. Then

\[
P(y_t=k\mid x_t)=
\begin{cases}
g(x_t), & k=c,\\
[1-g(x_t)]h_k(x_t), & k\neq c.
\end{cases}
\]

The switch classifier cannot return the previous family. This structure corresponds to a meaningful lyrical decision—continue an ending pattern or move away—while retaining a ranked choice among alternatives. For both flat and hierarchical SGD models, regularization is selected from 10⁻⁵, 10⁻⁴, and 10⁻³ by validation MRR. Probability temperature is selected from 0.5, 0.75, 1, 1.25, 1.5, and 2 by validation NLL. The hierarchical model selects regularization 10⁻³ and temperature 1.0; no test result enters selection. An otherwise identical hierarchical model removes the source-credit-label feature. This held-out ablation tests whether label conditioning contributes beyond the sequential written-ending context; it is necessary before any personalization claim.

### 7.4 Results

The global-frequency model achieves MRR 0.314 and top-5 accuracy 0.462. Conditioning only on the previous ending family produces the largest gain: the Markov baseline reaches MRR 0.608 and top-5 accuracy 0.729. The flat logistic context model reaches MRR 0.627 and top-5 0.773. The validation-selected hierarchical model reaches MRR 0.628 (95% CI 0.618–0.638), top-3 accuracy 0.695 (0.685–0.705), and top-5 accuracy 0.775 (0.767–0.782). Macro estimates across the 201 labels with at least 20 leakage-safe test events are similar (MRR 0.631, top-5 0.775), indicating that the result is not solely driven by the largest repertoires. Table 4 gives the complete benchmark, while Figure 4 makes the continuation-versus-switch asymmetry and the null source-label ablation visible.

| Model | Top-1 | Top-3 | Top-5 | MRR | NLL | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global frequency | 0.146 [0.137, 0.154] | 0.328 [0.318, 0.339] | 0.462 [0.450, 0.474] | 0.314 [0.307, 0.321] | 2.689 | 0.0008 |
| First-order Markov | 0.496 [0.483, 0.509] | 0.646 [0.634, 0.657] | 0.729 [0.718, 0.739] | 0.608 [0.597, 0.619] | 1.978 | 0.0123 |
| Flat SGD context | 0.498 [0.486, 0.511] | 0.694 [0.684, 0.704] | 0.773 [0.766, 0.781] | 0.627 [0.618, 0.637] | 1.895 | 0.0314 |
| Hierarchical SGD context | 0.498 [0.486, 0.511] | 0.695 [0.685, 0.705] | 0.775 [0.767, 0.782] | 0.628 [0.618, 0.638] | 1.869 | 0.0140 |
| Hierarchical SGD without source label | 0.498 [0.485, 0.511] | 0.695 [0.685, 0.704] | 0.775 [0.767, 0.783] | 0.628 [0.618, 0.638] | 1.872 | 0.0116 |

Table 4 Song-held-out results on 34,395 leakage-safe, originally adjacent test events from 787 songs. Brackets are song-cluster bootstrap 95% intervals (2,000 replicates). All five systems have full prediction coverage.

Relative to Markov, the hierarchical model improves MRR by 0.0198 (paired CI 0.0177–0.0220), top-3 by 0.0498 (0.0445–0.0551), top-5 by 0.0457 (0.0408–0.0509), and top-1 by 0.0022 (0.0008–0.0038). Its advantage over the flat context model is far smaller: only the top-3 interval is narrowly positive (+0.0017, 0.0001–0.0033); top-1, top-5, and MRR intervals include zero. The hierarchy is therefore useful chiefly as an interpretable continuation-versus-switch factorization, not as evidence of a large gain over a flat classifier.

Stratified results expose the task's main asymmetry. Continuation events are easy for the Markov model (top-1 0.992), while switch events remain difficult (0.016). The hierarchical model retains high continuation performance (top-1 0.985) and improves switch top-3 accuracy to 0.400, compared with 0.302 for Markov; its switch top-1 remains only 0.026. Repeated hooks do not solely explain the result: on non-repeated targets the hierarchical model still reaches top-1 0.490, top-3 0.683, and MRR 0.619. The application should therefore present multiple plausible next families, especially for an intended switch, rather than one supposedly correct rhyme.

### 7.5 Selective Abstention and Recommendation

Confidence thresholds are fixed from validation quantiles targeting 75%, 50%, and 25% coverage and then applied once to test. At full leakage-safe coverage, top-1 accuracy is 0.498. The nominal 75% threshold yields coverage 0.745 and accepted-event top-1 0.558; nominal 50% yields coverage 0.499 and top-1 0.611; nominal 25% yields coverage 0.244 and top-1 0.659. At the illustrative balanced 50% operating point, accepted-event top-3 is 0.736 and MRR is 0.701; coverage is 0.479 when the 1,431 leakage-filtered test candidates are retained in the denominator. Abstention therefore provides an explicit reliability–coverage trade-off rather than hiding weak contexts.

The public recommender releases global, Markov, and common aggregate ML contexts. It returns ending *families*, not lyric lines, words, or finished rap. A useful interface can show the top families, whether the previous class is being continued or switched, validation-defined confidence status, and descriptive label-level fingerprints. Crucially, removing the source-credit-label feature changes top-1 by only +0.0004, top-3 by +0.0007, and MRR by +0.0005 in favor of the labelled model, with all paired intervals crossing zero. The interface must not claim that rapper-label personalization improves prediction; it can only expose label-conditioned outputs as descriptive or experimental. It cannot prescribe quality or performed delivery.

An external MuChin implementation check (Wang *et al.* 2024) aligns 44,980 written lines and yields F1 0.931 and Matthews correlation 0.541 against exported same-family markers. Because the annotation interface itself grouped pinyin families and the export collapses class identity, this partially circular check is excluded from selection and is neither independent gold nor audio validation; full details appear in the supplement.

[[FIGURE:figure_4_written_rhyme_benchmark.png]]

Fig. 4 Prediction of the next dictionary-estimated written line-ending family on the strict terminal-Han population. Panel A compares global frequency, first-order Markov, flat context, hierarchical context without source-credit labels, and hierarchical context with source-credit labels across 34,395 leakage-safe events in 787 held-out songs. Whiskers are 95% song-cluster bootstrap intervals from 2,000 replicates. Panel B descriptively separates continuation from family-switch events; top-3 reaches 0.400 for switches under the hierarchical model versus 0.302 for Markov. Panel C plots four paired score differences for the model with versus without source-credit labels; every 95% interval crosses zero, so source-label conditioning has no supported benefit.

Alt text: Four aligned dot-and-whisker panels show Top-1, Top-3, Top-5, and MRR for five models. Context models dominate the global baseline, while the flat and two hierarchical models nearly overlap. A grouped bar panel shows near-ceiling Top-3 accuracy when the next line continues the same family but much lower accuracy when it switches. A third panel plots Top-1, Top-3, Top-5, and MRR differences for the model with versus without source-credit labels; all four 95% intervals cross zero. The task is restricted to written endings.

## 8. Integrated Lyrical-Repertoire Profiles

The three tasks should be joined only at the aggregate source-label level and only with their evidence statuses intact. A profile has three separate questions:

- **What does a held-out song resemble?** The strict dense–lexical retrieval model supplies ranked repertoire evidence and label-level reliability, not personal similarity.
- **Which named cultural worlds appear?** Reviewed or explicitly provisional entity links supply typed places, people, languages, groups, brands, works, or events, not biography or preference.
- **How do written endings tend to proceed?** Descriptive fingerprints and the continuation model supply recurring ending families and possible next-family transitions, not performed flow.

The present integration is asymmetric because the NER layer remains provisional. Retrieval and written-ending prediction have held-out test results; cultural entities have reproducibility gates and a path to gold but not accuracy. The profile must display that difference at the point of use. This is preferable to either omitting a promising cultural-reference task or overstating it as complete.

## 9. Discussion

### 9.1 What BGE-M3 Adds after Evaluation

RQ1 receives a clear but qualified answer. BGE-M3 is not the strongest standalone system for corpus source-label retrieval; character TF–IDF performs better. Dense representation is still useful because standardized fusion improves all primary metrics over TF–IDF with positive paired intervals. The result supports complementarity between distributional semantics and character-form evidence, not dense-model supremacy.

The post-embedding design is the main methodological contribution. Whole-song holdout converts vectors into a testable task. Cross-label shared-text exclusion and duplicate-component holdout prevent repeated material from masquerading as generalizable identity. Per-label macro aggregation changes the estimand from “average song in the corpus” to “average repertoire label”. The paired two-stage bootstrap attaches uncertainty to both scores and differences. Without these decisions, an embedding map could not answer whether a held-out song is recognizably consistent with a repertoire inside the frozen corpus.

The remaining interpretive limit is fundamental. Dense and lexical evidence may encode topic, names, register, formulaic language, or writing habits. We therefore avoid “pure style”. Future ablations can mask reviewed entities, remove high-keyness surface features, or isolate writing-form channels, but these would be probes of dependency, not causal decomposition of the encoder.

### 9.2 What NER Adds beyond Search

NER is useful when it normalizes spans into types, distinguishes ambiguous candidates, aggregates duplicate-controlled support, and creates edges with explicit meanings. Literal search alone can find a string; it cannot decide whether 中南海 is a location, political metonym, or brand in a particular context, nor can it validly aggregate the string into a cultural network. The current pilot demonstrates both the promise and the limit. Cross-method agreement and audit rules produce a small provisional layer, while ambiguous cases are withheld.

RQ2 is therefore answered in terms of readiness, leakage sensitivity, and statistically supported aggregate association rather than extraction accuracy. The frozen corpuswide inventory of 33 surfaces contracts to 22 after cross-label shared-text exclusion and the fixed graph universe are applied. The old support/lift-only 85 repertoire links contract first to 40 under the same shared-text control and then to six under song-unit uncertainty and BH-FDR; four entity co-mentions survive the corrected all-song denominator and multiplicity gate. None is human occurrence gold. The 800-item corpuswide design and unreviewed 157-occurrence released-claim audit are the next required empirical steps. Only after adjudication can strict-span micro precision/recall/F1, per-type support, boundary/type errors, inter-annotator agreement, and external uncertainty be reported. Network centrality should remain withheld until error by type and support is understood, because extraction bias can become cultural visibility bias (Lassen *et al.* 2024).

### 9.3 What Written-Ending Prediction Adds

RQ3 shows that immediate sequence context is powerful: first-order Markov modeling accounts for much of the gain over global frequency. Broader context then improves ranked multi-choice prediction by about five percentage points at top-3 relative to Markov. The hierarchical continuation/switch model performs almost identically to the flat context classifier on most paired metrics; its justification is an interpretable decision structure and favorable descriptive NLL/ECE, not a dramatic accuracy claim. The no-source-label ablation is equally consequential: the present data do not show that adding a corpus credit label improves prediction. A rapper-specific predictive claim would therefore be unsupported.

Selective abstention turns this model into a more honest instrument. A user can choose leakage-safe coverage near 0.50 and receive top-1 accuracy near 0.61 and top-3 near 0.74 on accepted events, or require greater selectivity and approach top-1 0.66 at coverage near 0.24. The severe switch-event gap remains visible at the point of use. The product is not a rap generator. It is an evidence-backed next-ending-family assistant that can support close reading or constrained writing while leaving lexical choice and performance to the user.

### 9.4 General Lesson for Digital Humanities

RQ4 is methodological rather than a single scalar result. A culturally meaningful computational profile is multi-layered and evidence-graded. Semantic retrieval, named-reference networks, and phonetic-form proxies should not be collapsed into one unexplained edge weight. Their disagreements are analytically useful: a pair may be retrievably similar without sharing provisional named references; two repertoires may reference the same places but differ in written-ending patterns. Such discrepancies can guide close reading, following the broader argument for multi-layer literary networks (Rafaeli *et al.* 2026).

The framework is transferable when representation, target, leakage control, baseline, inferential unit, typed explanation, and abstention are aligned with the scholarly question.

## 10. Limitations, Ethics, and Copyright

The corpus is a collected sample rather than a census of Chinese rap. Its source platforms, acquisition route and dates, temporal coverage, sampling frame, transcription origin, and rights or ethics basis are not documented in the supplied export and must be established by the dataset owner before submission. These unknowns, together with repertoire size and missing material, can bias which labels and references are visible. Source-credit labels are not globally identity-resolved and may include aliases, groups, collaborations, or metadata errors. No text-derived result establishes hometown, belief, cultural membership, collaboration, influence, friendship, or affiliation.

Retrieval relevance comes from corpus provenance, not independent human judgments of semantic similarity. TF–IDF is transductive to the fixed unlabeled evaluation corpus. Near-duplicate detection uses one threshold and cannot find every paraphrase or lightly edited hook. BGE-M3 remains difficult to explain at token level, and fusion does not disentangle topic from style.

The NER layer has the strongest current limitation: no completed occurrence-level human gold. Literal matching is context-insensitive, the CKIP model is general-domain and Traditional-Chinese-oriented, confidence is uncalibrated on Chinese rap, and agreement gates favor conventional place and language names over creative spellings, slang, brands, works, or rap-culture concepts. The provisional graph may therefore underrepresent precisely the innovative references that make the domain important. Dual review and per-type evaluation are prerequisites for stronger claims.

Written rhyme is inferred from dictionary pinyin applied to transcribed line endings. Contextual dictionary conversion cannot resolve every polyphone, dialectal or regional pronunciation, performance elision, internal rhyme, stress, cadence, or beat alignment. Regional and cross-language realization can make different dictionary families rhyme in performance, while audio-based tone and flow remain unobserved (Liu *et al.* 2023; Wang and Lin 2024). The strict estimand excludes code-switched and other non-Han terminal characters, so it does not describe every line in the corpus. Repeats retain original adjacency only inside chunks surviving legacy deduplication; the non-repeat sensitivity supports the frozen result. The counterfactual supports aggregate family shares and switch rate, not unretrained prediction metrics. Ending-family prediction must not be described as performed rhyme or flow.

Copyright and privacy constrain release. Public artifacts contain aggregate evidence and short analytic tokens, but no full lyrics, line text, private row identifiers, vectors, membership rows, or reviewer contexts. Researchers with lawful access can reproduce the builders against file-level checksums; copyrighted lyrics are not redistributed.

## 11. Conclusion

This study turns one Chinese-rap lyric corpus into three bounded downstream results. Strict held-out-song retrieval shows that BGE-M3 adds value when fused with character TF–IDF, even though it is weaker alone. The NER pilot produces a typed, conservative provisional cultural-reference layer while making the missing human gold impossible to overlook. Context models improve ranked next-written-ending prediction over a Markov baseline and expose a usable reliability–coverage trade-off through abstention, while a label ablation prevents an unsupported personalization claim.

The broader result is a standard for moving from representation to evidence. Embeddings become scholarly objects only after the downstream target, leakage controls, baselines, uncertainty, explanation, and claim boundary are specified. For Chinese rap, this yields a coherent account of lyrical repertoire through language, named cultural reference, and written-line form—without reducing the culture to an unexplained map or converting textual association into unsupported biography.

## Data Availability

Copyright-safe aggregate outputs, the original method contract and formal PD-001 protocol amendment, validation manifests, annotation schema and blank review templates, model builders, and figure-source tables are maintained at https://github.com/Mo119m/chinese-rap-embeddings-to-evidence. The public package excludes lyric text, full written lines, song/chunk identifiers, row-level lyric-content hashes, embeddings, private membership rows, and reviewer contexts. The frozen private corpus cannot be redistributed by this article. Its integrity is documented through file-level checksums and deterministic audit summaries so researchers with lawful access can reproduce the transformations against their copy. [Archival DOI to be added by the authors before submission.]

## Funding

[To be completed by the authors; state “None” if applicable.]

## Conflict of Interest

[To be completed and approved by all authors.]

## Ethics Statement

The study analyzes collected lyric text and corpus metadata and releases only aggregate, copyright-safe evidence. No source-credit label is treated as a verified natural-person identity. The authors must confirm institutional requirements and any necessary ethics determination before submission.

## Author Contributions

[CRediT roles to be completed by the authors.]

## AI Disclosure Statement

This manuscript was prepared with the assistance of OpenAI Codex [exact version/model to be completed by the authors]. The tool was used to assist with reproducible code development, preliminary data analysis, figure generation, and English-language drafting and editing. The human authors must independently check, reproduce, and approve all analyses, figures, claims, and references before submission and remain fully accountable for the work. The same use must be disclosed in the submission cover letter.

## References

Baioud, G. (2024) ‘Constructing “corrupted village wives and urban men” through multilingual performances’, *Language in Society*, 53/1: 25–45. https://doi.org/10.1017/S0047404522000665

Barrett, C. (2012) ‘Hip-hopping across China: intercultural formulations of local identities’, *Journal of Language, Identity and Education*, 11/4: 247–60. https://doi.org/10.1080/15348458.2012.706172

Blessing, A. *et al.* (2017) ‘An end-to-end environment for research question-driven entity extraction and network analysis’, *Proceedings of the Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature*, pp. 57–67. https://doi.org/10.18653/v1/W17-2208

Chen, J. *et al.* (2024) ‘M3-Embedding: multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation’, *Findings of the Association for Computational Linguistics: ACL 2024*, pp. 2318–35. https://doi.org/10.18653/v1/2024.findings-acl.137

Hirjee, H. and Brown, D. G. (2009) ‘Automatic detection of internal and imperfect rhymes in rap lyrics’, *Proceedings of the 10th International Society for Music Information Retrieval Conference*, pp. 711–16. https://archives.ismir.net/ismir2009/paper/000029.pdf

Jiang, H. *et al.* (2022) ‘Annotating the Tweebank corpus on named entity recognition and building NLP models for social media analysis’, *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pp. 7199–208. https://doi.org/10.18653/v1/2022.lrec-1.780

Lassen, I. M. S. *et al.* (2024) ‘Epistemic consequences of unfair tools’, *Digital Scholarship in the Humanities*, 39/1: 198–214. https://doi.org/10.1093/llc/fqad091

Li, X. *et al.* (2020) ‘FLAT: Chinese NER using Flat-Lattice Transformer’, *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 6836–42. https://doi.org/10.18653/v1/2020.acl-main.611

Lin, Y. and Wang, T. (2024) ‘Rhyming style, persona, and the contested landscape of authentic Chinese hip hop’, *Journal of Sociolinguistics*, 28/2: 22–41. https://doi.org/10.1111/josl.12635

Liu, J. (2014) ‘Alternative voice and local youth identity in Chinese local-language rap music’, *positions: asia critique*, 22/1: 263–92. https://doi.org/10.1215/10679847-2383840

Liu, J. (2021) ‘Language, identity and unintelligibility: a case study of the rap group Higher Brothers’, *East Asian Journal of Popular Culture*, 7/1: 43–59. https://doi.org/10.1386/eapc_00038_1

Liu, J. *et al.* (2023) ‘Linguistic tone in Chinese rap: an interdisciplinary approach’, *Journal of New Music Research*, 52/4: 265–84. https://doi.org/10.1080/09298215.2024.2329075

Paszke, A. *et al.* (2019) ‘PyTorch: an imperative style, high-performance deep learning library’, *Advances in Neural Information Processing Systems*, 32. https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

Pedregosa, F. *et al.* (2011) ‘Scikit-learn: machine learning in Python’, *Journal of Machine Learning Research*, 12: 2825–30. https://jmlr.org/papers/v12/pedregosa11a.html

Peng, N. and Dredze, M. (2015) ‘Named entity recognition for Chinese social media with jointly trained embeddings’, *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pp. 548–54. https://doi.org/10.18653/v1/D15-1064

Rafaeli, O. *et al.* (2026) ‘Mind the gap: word-embedding and multi-layered literary networks’, *Digital Scholarship in the Humanities*, 41/Supplement 1: i213–29. https://doi.org/10.1093/llc/fqaf112

Savoy, J. (1997) ‘Statistical inference in retrieval effectiveness evaluation’, *Information Processing and Management*, 33/4: 495–512. https://doi.org/10.1016/S0306-4573(97)00027-7

Sun, Y. *et al.* (2023) ‘SongRewriter: a Chinese song rewriting system with controllable content and rhyme scheme’, *Findings of the Association for Computational Linguistics: ACL 2023*, pp. 12863–80. https://doi.org/10.18653/v1/2023.findings-acl.814

Sundararajan, K. and Woodard, D. (2018) ‘What represents “style” in authorship attribution?’, *Proceedings of the 27th International Conference on Computational Linguistics*, pp. 2814–22. https://doi.org/10.18653/v1/C18-1238

Wang, A. *et al.* (2023) ‘Can authorship representation learning capture stylistic features?’, *Transactions of the Association for Computational Linguistics*, 11: 1416–31. https://doi.org/10.1162/tacl_a_00610

Wang, T. and Lin, Y. (2024) ‘Variation is the way to perfection: imperfect rhyming in Chinese hip hop’, *Linguistics Vanguard*, 10/1: 505–15. https://doi.org/10.1515/lingvan-2024-0093

Wang, X. (2013) ‘“I am not a qualified dialect rapper”: constructing hip-hop authenticity in China’, *Sociolinguistic Studies*, 6/2: 333–72. https://doi.org/10.1558/sols.v6i2.333

Wang, Z. *et al.* (2024) ‘MuChin: a Chinese colloquial description benchmark for evaluating language models in the field of music’, *Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence*, pp. 7771–79. https://doi.org/10.24963/ijcai.2024/860

Wolf, T. *et al.* (2020) ‘Transformers: state-of-the-art natural language processing’, *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 38–45. https://doi.org/10.18653/v1/2020.emnlp-demos.6

Xue, L. *et al.* (2021) ‘DeepRapper: neural rap generation with rhyme and rhythm modeling’, *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing*, pp. 69–81. https://doi.org/10.18653/v1/2021.acl-long.6

Yang, Y. *et al.* (2018) ‘Distantly supervised NER with partial annotation learning and reinforcement learning’, *Proceedings of the 27th International Conference on Computational Linguistics*, pp. 2159–69. https://doi.org/10.18653/v1/C18-1183
