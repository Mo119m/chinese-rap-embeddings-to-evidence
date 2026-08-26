# Supplementary Methods

## S1. Exact computational environment and model provenance

Table S1 records the executable environment used or recovered for the frozen analyses. The vector and result artifacts are hash-addressed. The historical device and realized mixed-precision setting used for the original BGE-M3 encoding were not captured; current hardware is not treated as evidence of that earlier run.

| Component | Exact record | Analysis-relevant settings | Evidence status |
| --- | --- | --- | --- |
| Execution environment | CPython 3.12.13; Windows 11 (`10.0.26200`) | Task-specific deterministic seeds; Python executable SHA-256 `e062889e…aea35` | Direct for BGE, NER, and rhyme; retrieval manifest lacks a contemporaneous runtime lock |
| Semantic encoder | `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`; FlagEmbedding 1.4.0; PyTorch 2.11.0+cu128 | Dense head; 1,024 dimensions; `max_length=2048`; batch 2; L2 normalization | Checkpoint/files and frozen vector hash verified; historical device and realized `use_fp16` unavailable |
| NER model | `ckiplab/albert-tiny-chinese-ner` revision `bcb519856ca93a666b1e48a9daef3f88c9b572a0`; Transformers 5.14.1; PyTorch 2.11.0+cu128 | 180-character windows; 40-character overlap; maximum 256 tokens; batch 96 | Runtime, checkpoint, weights, and vocabulary hashes verified |
| Written-ending extraction/model | pypinyin 0.55.0; scikit-learn 1.9.0 | `Style.FINALS_TONE3`; `strict=True`; neutral tone 5; averaged log-loss SGD | Direct package-version and module-hash evidence |
| Numerical/data stack | NumPy 2.4.6; pandas 3.0.5; SciPy 1.18.0; scikit-learn 1.9.0 | Sparse TF–IDF, normalization, tests, tables | Direct for NER/rhyme; forensic rather than manifest-bound for retrieval |
| Transformer support stack | Transformers 5.14.1; tokenizers 0.22.2; huggingface-hub 1.27.0 | Local/offline model loading | Transformers direct for NER; remaining BGE support versions recovered locally |

The BGE checkpoint configuration contains `transformers_version=4.33.0`; that is upstream serialization metadata and is not reported as this project's runtime. The canonical environment-map fingerprint is SHA-256 `b290363fa776f50219cd16ee124fd88b7253d77658bdb9f0d2cef42ee2453152`. Full file hashes and official artifact URLs appear in `software_citation_recommendations.md` in the release audit directory.

## S2. PD-001 retrieval protocol amendment

The original frozen research contract required vectorizers to be fitted on training data only. The released character TF–IDF builder instead calls `fit_transform` once on all document text in the fixed retrieval corpus before constructing group-held-out label profiles. Thus each query affects the unlabeled character vocabulary and inverse-document-frequency distribution, although its source label, relevance status, rank, and evaluation outcome are not supplied to the vectorizer. The equal fusion weight is also not outcome-tuned.

The reported retrieval estimand is therefore:

> Within the frozen corpus, when a song's source-credit membership and its contribution to candidate profiles are held out, how highly does the remaining source-label repertoire rank?

It is not a prospective estimate for later songs or another corpus. The original contract remains unchanged; `PROTOCOL_AMENDMENT_PD001_TRANSDUCTIVE_TFIDF.md` records the audit chronology, affected artifacts, and requirements for any future inductive version. No numerical result changed because the amendment corrects the declaration, not the frozen implementation.

## S3. External written-ending implementation check

The external MuChin resource (Wang *et al.*, 2024; DOI `10.24963/ijcai.2024/860`) was used only as an implementation sensitivity check. Across 1,000 publisher-recommended folders, 44,980 written lines align exactly by line and section. Treating exported `R` markers as same-family-within-section indicators gives F1 0.931 and Matthews correlation 0.541 for the project's deterministic same-family detection.

This is not independent gold. The MuChin interface automatically groups and colour-highlights pinyin rhyme families, annotators principally review polyphonic readings, and the exported `R` field collapses family identity. The check is therefore partially circular, excluded from model selection, and does not validate Chinese-rap rhyme, pronunciation, tone, cadence, beat alignment, or flow.

## S4. NER annotation path to human gold

The primary private review package contains 800 distinct occurrences sampled by proposed type and agreement state: exact agreement, overlap agreement, type/boundary conflict, lexicon only, and transformer only. Two reviewers work independently. Uncertainty, corrected spans or types, different link decisions, and all disagreements require adjudication. A future gold release requires complete dual review, adjudication, agreement reporting, manifest reconciliation, and a song-level train/development/test split.

Two earlier, separate sets of 300 context forms and 300 occurrence tasks contain no completed decisions and are superseded. Neither those empty forms nor the new unreviewed package supports precision, recall, F1, network centrality, or demographic comparisons. The present 22-surface inventory and its six label-to-place and four co-mention edges are reproducibility-screened provisional evidence only.

## S5. Public/private release boundary

The public supplement contains aggregate metrics, uncertainty intervals, typed entity surfaces, support summaries, written-ending classes, figure-source tables, code, contracts, and validation manifests. It excludes lyric text, full written lines, song/chunk identifiers, private membership rows, row-level lyric hashes, embeddings, and reviewer contexts. Short Chinese surfaces are included only when needed to interpret an aggregate entity or written-ending result.

The private frozen snapshot is required to rebuild row-level artifacts. Its original platforms, acquisition route and dates, temporal coverage, sampling frame, transcription origin, rights basis, and ethics determination are not established in the supplied export. `DATA_PROVENANCE_AND_AUTHOR_ACTIONS.md` separates verified local facts from the owner-supplied information required before journal submission.

## Software and model records

- Beijing Academy of Artificial Intelligence. (2024). *BAAI/bge-m3* (revision `5617a9f61b028005a4858fdac845db406aefb181`) [Pretrained language model]. https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181
- CKIP Lab. (2022). *ckiplab/albert-tiny-chinese-ner* (revision `bcb519856ca93a666b1e48a9daef3f88c9b572a0`) [Pretrained token-classification model]. https://huggingface.co/ckiplab/albert-tiny-chinese-ner/tree/bcb519856ca93a666b1e48a9daef3f88c9b572a0
- FlagOpen. (2026). *FlagEmbedding* (Version 1.4.0) [Computer software]. https://github.com/FlagOpen/FlagEmbedding/releases/tag/v1.4.0
- mozillazg and contributors. (2025). *pypinyin* (Version 0.55.0) [Computer software]. https://pypi.org/project/pypinyin/0.55.0/
- The scikit-learn developers. (2026). *scikit-learn* (Version 1.9.0) [Computer software]. https://doi.org/10.5281/zenodo.20510517
- Gommers, R., Virtanen, P., Haberland, M., *et al.* (2026). *SciPy* (Version 1.18.0) [Computer software]. https://doi.org/10.5281/zenodo.20764140
- The pandas development team. (2026). *pandas* (Version 3.0.5) [Computer software]. https://doi.org/10.5281/zenodo.21500199
