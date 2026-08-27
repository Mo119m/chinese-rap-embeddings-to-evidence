# Supplementary Methods

## S1. Exact Computational Environment and Model Provenance

Table S1 records the executable environment used or recovered for the frozen analyses. The vector and result artifacts are hash-addressed. The historical device and realized mixed-precision setting used for the original BGE-M3 encoding were not captured; current hardware is not treated as evidence of that earlier run.

| Component | Exact record | Analysis-relevant settings | Evidence status |
| --- | --- | --- | --- |
| Execution environment | CPython 3.12.13; Windows 11 (`10.0.26200`) | Task-specific deterministic seeds; Python executable SHA-256 `e062889e…aea35` | Direct for BGE, named-entity recognition (NER), and rhyme; retrieval manifest lacks a contemporaneous runtime lock |
| Semantic encoder | `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`; FlagEmbedding 1.4.0; PyTorch 2.11.0+cu128 | Dense head; 1,024 dimensions; `max_length=2048`; batch 2; L2 normalization | Checkpoint/files and frozen vector hash verified; historical device and realized `use_fp16` unavailable |
| NER model | `ckiplab/albert-tiny-chinese-ner` revision `bcb519856ca93a666b1e48a9daef3f88c9b572a0`; Transformers 5.14.1; PyTorch 2.11.0+cu128 | 180-character windows; 40-character overlap; maximum 256 tokens; batch 96 | Runtime, checkpoint, weights, and vocabulary hashes verified |
| Written-ending extraction/model | pypinyin 0.55.0; scikit-learn 1.9.0 | `Style.FINALS_TONE3`; `strict=True`; neutral tone 5; averaged log-loss SGD | Direct package-version and module-hash evidence |
| Numerical/data stack | NumPy 2.4.6; pandas 3.0.5; SciPy 1.18.0; scikit-learn 1.9.0 | Sparse term frequency–inverse document frequency (TF–IDF), normalization, tests, tables | Direct for NER/rhyme; forensic rather than manifest-bound for retrieval |
| Transformer support stack | Transformers 5.14.1; tokenizers 0.22.2; huggingface-hub 1.27.0 | Local/offline model loading | Transformers direct for NER; remaining BGE support versions recovered locally |

The BGE checkpoint configuration contains `transformers_version=4.33.0`; that is upstream serialization metadata and is not reported as this project's runtime. The canonical environment-map fingerprint is SHA-256 `b290363fa776f50219cd16ee124fd88b7253d77658bdb9f0d2cef42ee2453152`. Full file hashes and official artifact URLs appear in `software_citation_recommendations.md` in the release audit directory.

## S2. PD-001 Retrieval Protocol Amendment

The original frozen research contract required vectorizers to be fitted on training data only. The released character TF–IDF builder instead calls `fit_transform` once on all document text in the fixed retrieval corpus before constructing group-held-out label profiles. Thus each query affects the unlabeled character vocabulary and inverse-document-frequency distribution, although its source label, relevance status, rank, and evaluation outcome are not supplied to the vectorizer. The equal fusion weight is also not outcome-tuned.

The reported retrieval estimand is therefore:

> Within the frozen corpus, when a song's source-credit membership and its contribution to candidate profiles are held out, how highly does the remaining source-label repertoire rank?

It is not a prospective estimate for later songs or another corpus. The original contract remains unchanged; `PROTOCOL_AMENDMENT_PD001_TRANSDUCTIVE_TFIDF.md` records the audit chronology and affected artifacts. The primary fixed-corpus estimates remain those of the frozen implementation.

A grouped sixfold sensitivity tests whether transductive vocabulary and IDF exposure explains the fusion result. Each of 5,430 global exact/near-duplicate components is assigned wholly to one fold; the 5,455 queries are distributed as 910 in one fold and 909 in each of five folds. Every test fold contains all 204 source-credit labels, and each corresponding training partition retains at least five components per label. Inductive TF–IDF is fitted only on the five training folds and then transforms their held-out fold. A matched transductive system fits vocabulary and IDF on all 5,455 unlabeled texts but uses the same fold-specific training profiles, test queries, duplicate weights, candidate labels, fusion rule, and rankings. Because cross-fitted profiles contain five-sixths of the components, their absolute scores are a sensitivity result rather than a direct replacement for leave-one-component-out Table 2.

Inductive fusion yields macro MRR 0.439712 versus 0.404763 for inductive TF–IDF, a paired difference of +0.034949 (95% CI 0.024373–0.046082). Matched transductive exposure increases TF–IDF MRR by +0.003963 (0.001020–0.006942) and fusion MRR by +0.003681 (0.001413–0.006416). Intervals use the same 5,000-replicate paired two-stage label/component bootstrap principle as the primary analysis. Thus evaluation-corpus exposure is mildly optimistic within this corpus, but its magnitude cannot explain the training-only fusion advantage. This sensitivity does not estimate prospective or external-population performance.

## S3. Repertoire-Graph Null and Projection Diagnostics

The published graph is the intersection of a 140-edge reciprocal-top-five network under primary duplicate control and a 145-edge network after cross-label shared-text exclusion. The observed intersection contains 86 edges. The primary null holds the first network fixed and applies 1,450 successful undirected double-edge swaps to the second network per replicate, preserving its complete node-degree sequence while randomizing endpoints. Across 10,000 fixed-seed replicates, the null intersection has mean 4.524, median 4, 95% range 1–9, and maximum 13; no replicate reaches 86, giving an add-one Monte Carlo *p* of 0.00010. An auxiliary 100,000-replicate source-label permutation gives mean 0.979 and *p* = 0.000010. These tests concern cross-treatment adjacency alignment, not edgewise significance, social relations, influence, or external cultural structure.

The deterministic two-dimensional display combines the normalized primary and sensitivity centroids, recentres them, and applies singular-value-decomposition PCA. Its first two components explain 26.20% of variation. Using cosine distance in the original 1,024-dimensional consensus space and Euclidean distance in the display, trustworthiness is 0.785, 0.795, and 0.804 at *k* = 5, 10, and 15. Exact-neighbour retention is 0.173, 0.268, and 0.347, compared with random expectations of 0.025, 0.049, and 0.074. Pairwise distance-rank correlation is 0.680. Only 18 of 86 released edges are mutual top-five neighbours in two dimensions. Accordingly, the PCA is a navigation overview; released graph lines, not screen distance, carry the relation claim.

## S4. External Written-Ending Implementation Check

The external MuChin resource (Wang *et al.* 2024; DOI `10.24963/ijcai.2024/860`) was used only as an implementation sensitivity check. Across 1,000 publisher-recommended folders, 44,980 written lines align exactly by line and section. Treating exported `R` markers as same-family-within-section indicators gives an F1 score (the harmonic mean of precision and recall) of 0.931 and a Matthews correlation of 0.541 for the project's deterministic same-family detection.

This is not independent gold. The MuChin interface automatically groups and colour-highlights pinyin rhyme families, annotators principally review polyphonic readings, and the exported `R` field collapses family identity. The check is therefore partially circular, excluded from model selection, and does not validate Chinese-rap rhyme, pronunciation, tone, cadence, beat alignment, or flow.

## S5. NER Annotation Path to Human Gold

The corpuswide private review package contains 800 distinct occurrences sampled by proposed type and agreement state: exact agreement, overlap agreement, type/boundary conflict, lexicon only, and transformer only. A separate claim-directed package contains every unique occurrence contributing to the ten currently released claims: 102 label/reference occurrence rows and 60 co-mention occurrence rows, with five rows overlapping, for 157 unique occurrences in total. Coverage is therefore 157/157 (100%) within the released-claim frame. Reviewer occurrence sheets add 20 real comparators and 16 synthetic boundary-attention controls; pair sheets add eight real non-released comparators to the 23 released co-mention support tasks.

For both packages, two reviewers work independently; in the claim-directed package, they are blind to claim/control status. Uncertainty, corrections, conflicting link decisions, and all disagreements require adjudication. At release, both reviewers and the adjudication sheets contain zero completed decisions. The claim-directed audit can ultimately confirm or reject the evidence underlying the released edges, but it is claim-conditioned and has no corpuswide negative frame; it cannot estimate corpuswide precision, recall, or F1. The 800-item package remains the route to those metrics and a song-level train/development/test split. Inter-annotator agreement, claim-confirmation rates, and all accuracy metrics remain withheld.

Two earlier, separate sets of 300 context forms and 300 occurrence tasks also contain no completed decisions and are superseded. No unreviewed package supports network centrality or demographic comparisons. The present 22-surface inventory and its six label-to-place and four co-mention edges are reproducibility-screened provisional evidence only.

## S6. Public/Private Release Boundary

The public supplement contains aggregate metrics, uncertainty intervals, typed entity surfaces, support summaries, written-ending classes, figure-source tables, code, contracts, and validation manifests. It excludes lyric text, full written lines, song/chunk identifiers, private membership rows, row-level lyric hashes, embeddings, and reviewer contexts. Short Chinese surfaces are included only when needed to interpret an aggregate entity or written-ending result.

The private frozen snapshot is required to rebuild row-level artifacts. Its original platforms, acquisition route and dates, temporal coverage, sampling frame, transcription origin, rights basis, and ethics determination are not established in the supplied export. `DATA_PROVENANCE_AND_AUTHOR_ACTIONS.md` separates verified local facts from the owner-supplied information required before journal submission.

## Software and Model Records

- Beijing Academy of Artificial Intelligence (2024) *BAAI/bge-m3*, revision `5617a9f61b028005a4858fdac845db406aefb181` [pretrained language model]. https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181
- CKIP Lab (2022) *ckiplab/albert-tiny-chinese-ner*, revision `bcb519856ca93a666b1e48a9daef3f88c9b572a0` [pretrained token-classification model]. https://huggingface.co/ckiplab/albert-tiny-chinese-ner/tree/bcb519856ca93a666b1e48a9daef3f88c9b572a0
- FlagOpen (2026) *FlagEmbedding*, version 1.4.0 [computer software]. https://github.com/FlagOpen/FlagEmbedding/releases/tag/v1.4.0
- mozillazg and contributors (2025) *pypinyin*, version 0.55.0 [computer software]. https://pypi.org/project/pypinyin/0.55.0/
- The scikit-learn developers (2026) *scikit-learn*, version 1.9.0 [computer software]. https://doi.org/10.5281/zenodo.20510517
- Gommers, R. *et al.* (2026) *SciPy*, version 1.18.0 [computer software]. https://doi.org/10.5281/zenodo.20764140
- The pandas development team (2026) *pandas*, version 3.0.5 [computer software]. https://doi.org/10.5281/zenodo.21500199
