# Final independent manuscript audit

> **Historical snapshot — superseded for the current release.** This file audits the 25 August `work/paper_v3` snapshot. Its 8,600-word count and recorded hashes do not describe the current manuscript. The authoritative current records are `validation/release_validation.json` and `paper/derivative_provenance.json`.

## Overall assessment: NOT READY TO SUBMIT — empirical and methodological integrity PASS; author and release completion required

The revised manuscript is internally coherent and its scientific results are reconciled to the frozen artifacts. I found **zero numerical mismatches** across the corpus table, all retrieval values and intervals, the provisional NER/network funnel and released edges, the written-ending benchmark and abstention results, and all four figure captions. The transductive TF–IDF deviation is disclosed accurately and governed by a dated amendment. The NER risk-ratio bound is defined correctly, all figures and tables are called out in the prose, RQ4 is explicitly a design synthesis, and the citation list is closed. The final public-boundary correction is also coherent: row-level lyric-content hashes remain private, whereas whole-file SHA-256 checksums are retained solely as non-content integrity metadata.

The manuscript is not yet ready for journal submission because author-controlled metadata and corpus acquisition/rights/ethics facts remain unresolved, and the public repository has not yet been synchronized with everything claimed in the Data Availability statement. These are publication blockers, not hidden numerical defects.

## Audited snapshot

- Manuscript: `work/paper_v3/manuscript.md`
- SHA-256: `ed6e8d4d941bf8784d6bf98a6f3dc8e85484253d51f1c43aae06e99dc989a4bf`
- Size / modified: 70,217 bytes; 25 August 2026 22:45:11 America/Chicago
- Submission contract: `work/downstream_v1/DSH_SUBMISSION_CONTRACT_2026.md`, SHA-256 `8fed16aeca897842f142c73a433dae9e86f9f3d1af317d96d85bd657ad045c5a`
- Frozen method contract: `work/downstream_v1/RESEARCH_CONTRACT.md`, SHA-256 `ec5f36c787ff361e302cbdececb9e960975aa13915b327649d53cdc5e4a7d1d8`
- PD-001 amendment: `work/downstream_v1/PROTOCOL_AMENDMENT_PD001_TRANSDUCTIVE_TFIDF.md`, SHA-256 `e63e91b5f5b4064dad7ecfb3ad336154ae6baa642da5d0429ab530cd8e5800a6`
- Supplementary Methods: `work/paper_v3/supplementary_methods.md`, SHA-256 `ed76578ec395ccbc7247da41970d57c865d3f0243bbc1d00d919dc45d7bc9b3d`
- Local release validation: `work/chinese-rap-embeddings-to-evidence/validation/release_validation.json`, SHA-256 `b041a2a7d5a678660f17223ea537031291c9be2297f23f093d4108990a186fcc`
- Final render QA: `work/audits/paper_render_qa.json`, SHA-256 `ae014ff7716e55c5ceae74d28ce1d850c566475e12c619478e6dc0131f320fc0`

This is a hash-bound snapshot audit. A later manuscript change requires a targeted recheck.

## Gate summary

| Gate | Result | Evidence |
| --- | --- | --- |
| Corpus/result reconciliation | **PASS** | Zero mismatches against the frozen JSON/CSV artifacts. |
| Retrieval table and caption | **PASS** | All 15 Table 2 point estimates, all 30 displayed interval endpoints, reported paired deltas, ablations, 5,455-query/204-label denominators, and Fig. 2 statements reconcile. |
| NER/network table and caption | **PASS WITH NO-GOLD BOUNDARY** | The 33→23→22, 85→40→6, 9→1, and corrected 5→4 funnels; all six label/place edges; all four co-mentions; examples; denominators; and Fig. 3 reconcile. Human gold remains 0/800. |
| Written-ending table and caption | **PASS** | All 30 Table 4 displayed values, all interval endpoints, 34,395-event/787-song denominator, paired deltas, strata, abstention, label ablation, MuChin check, and Fig. 4 reconcile. |
| Frozen artifact validation | **PASS** | Input 12/12; retrieval 13/13; NER 31/31 plus reconciliation 14/14 and independent audit 19/19; rhyme 20/20; figures 21/21. |
| TF–IDF protocol integrity | **PASS** | §4.2, Fig. 1, Limitations, Supplement S2, and PD-001 all state fixed-corpus transductive vocabulary/IDF fitting, no label/outcome fitting, untuned fusion, and no future-corpus claim. The original contract is preserved. |
| NER interval language | **PASS** | §6.4 gives `[pL-low/pR-high, pL-high/pR-low]` and calls it a conservative ratio of marginal Jeffreys bounds, not a standard risk-ratio confidence interval. |
| RQ alignment | **PASS** | RQ1–RQ3 match the three evaluated tasks; RQ4 and §9.4 explicitly identify the integration as methodological design synthesis. |
| Figure/table callouts | **PASS** | Tables 1–4 and Figs 1–4 are each invoked in the prose before their legend. |
| Citation/reference closure | **PASS** | 26 unique author–year citations and 26 bibliography entries; no missing or orphan entry. The empirical MuChin check now cites Wang *et al.* (2024), verified against the official IJCAI record. |
| DSH abstract | **PASS** | Five required fields present; 204 whitespace-delimited words including field labels, below 250. |
| DSH main-text length | **PASS WITH MARGIN** | 8,600 whitespace-delimited words before References, below 9,000 by 400; final Word/portal count must be repeated after author additions. |
| Keywords / language / lyric quotation | **PASS** | 10 keywords; English exposition; Chinese appears as bounded analytic labels; no lyric passage is reproduced. |
| Figure files / alt text | **PASS AT SOURCE LEVEL** | Four RGB TIFFs at 600 dpi; every figure has a legend and adjacent alt text. |
| Public hash boundary | **PASS** | The manuscript, supplement, README, methods note, release validation, and site validation consistently exclude row-level lyric-content hashes while retaining whole-file checksums and aggregate join keys as non-content integrity metadata. No released row-level hash value was found. |
| Required author declarations | **FAIL — AUTHOR ACTION** | Identity, affiliation, email, DOI disposition, funding, conflict, CRediT, ethics outcome, and exact AI model/version remain unresolved. |
| Corpus provenance, rights, and ethics | **FAIL — OWNER ACTION** | Acquisition platform/mechanism/date, sampling, temporal scope, lyric origin, custody, rights/terms, ethics determination, bias, and controlled-access policy are not documented. |
| Data Availability snapshot | **FAIL — RELEASE ACTION** | At audited repo HEAD `58a3f5c`, the GitHub tree does not yet contain the revised v3 paper/supplement, the original method contract, PD-001, or the downstream input-audit package named by the statement. |
| Final rendered submission format | **PASS LOCALLY** | The hash-bound DOCX/PDF package passed layout, figure, table, Chinese-glyph, math-notation, and text-smoke QA. All 39 manuscript and four supplement pages were visually reviewed; pages affected by the boundary correction were rechecked. Final portal preview remains a submission-day human check. |

## Numerical reconciliation

### Corpus and cleaning — PASS

Verified: 7,211 songs; 22,128 canonical and sidecar chunks; 21,553 eligible chunks; 575 withheld metadata-only chunks; 241 source-credit labels; 226 labels with at least five songs; 7,134 artist/title-eligible and 77 ineligible songs; 102 title-semantic exclusions; four approved title overrides; 405 multi-song exact-content groups spanning 921 songs; and the fixed 204-label frame.

### Retrieval — PASS

Verified the 5,455-query, 5,430-component population and the complete exclusion trail; all Table 2 metrics and intervals; fusion-minus-TF–IDF and fusion-minus-BGE differences; score-standardization, near-duplicate, and shared-text diagnostics; and the PO8 +0.224 label-level example. The headline fusion MRR is 0.447084 [0.414490, 0.481482], and fusion-minus-TF–IDF MRR is +0.031312 [0.020975, 0.041780].

### NER and cultural-reference network — PASS WITH REQUIRED ABSTENTION

Verified the candidate and agreement counts, shared-hash exclusion, inventory types, every one of the six released label-to-place rows, every one of the four released co-mention rows, and the Tangoz/杭州, GALI/上海, 法老/上海, and 伦敦–巴黎 examples. The independent audit recomputed all 4,488 label/entity and 231 entity-pair tests with zero mismatches. The manuscript correctly reports no precision, recall, F1, identity, biography, residence, preference, or social-relation claim.

### Written-ending prediction — PASS

Verified 283,806 strict terminal-Han line occurrences, 5,619 songs, 52,152 retained repeats, the 3,951/834/834 song split, validation and test leakage exclusions, all Table 4 metrics and intervals, paired model differences, continuation/switch and non-repeat strata, four abstention rows, all source-label-ablation intervals, and the partially circular MuChin implementation check. The headline Top-3 is 0.695421 [0.685488, 0.704942], with a +0.049775 [0.044460, 0.055112] gain over Markov; all label-input ablation intervals include zero.

### Public hash boundary — PASS

The corrected wording distinguishes two different grains. A row-level lyric-content hash is a deterministic digest attached to a private song, chunk, line, or membership record and could function as a linkage token; those values remain private. A file-level checksum is one SHA-256 digest over an entire input, artifact, model file, or release file and is retained to verify bitwise integrity. The manuscript states this distinction in §3, §10, and Data Availability; Supplement S5 excludes row-level lyric hashes; the release README and methods note use the same boundary. Public manifests contain file-level checksums, and the input-audit summary names a row-hash validity check without exposing any row value. The reader-facing site reports zero hashes. These facts are complementary, not contradictory.

## Exact remaining blockers

1. **Author and declaration completion.** Replace the eight visible bracketed placeholders: author names, affiliations, corresponding email, archival-DOI disposition, funding, conflict of interest, CRediT roles, and exact Codex version/model. Record the institutional ethics outcome and use identical AI wording in the cover letter.
2. **Corpus owner evidence.** Complete the 12-field record in `DATA_PROVENANCE_AND_AUTHOR_ACTIONS.md` from documentary evidence. At minimum, resolve acquisition source/mechanism/dates, sampling and temporal scope, lyric/transcription origin, custody/freeze date, rights/terms basis, ethics/IRB determination, known bias/missingness, and private-access policy. If a fact remains unknown, retain `Unknown—not supplied` and narrow the claim; do not infer it.
3. **Public-release synchronization.** The local release tree and rendered files now pass validation, but repo HEAD and `origin/main` remain `58a3f5c` with the final package uncommitted. Before submission, commit and push the revised manuscript and supplement, contracts, PD-001 amendment, input audit, current validation records, and final figure manifest; verify the public commit and replace or remove the archival DOI placeholder.

No additional empirical rerun is required by this audit. A future inductive TF–IDF result would be a new estimand and version, not a correction to the present fixed-corpus transductive benchmark.

## Non-blocking caveats to preserve

- The 400-word source margin is adequate but not large; recount after author additions and final portal conversion.
- NER remains provisional until 800 dual reviews and adjudication are complete.
- Retrieval is snapshot-dependent because query text contributes to unlabeled TF–IDF vocabulary and IDF.
- Historical BGE device and realized mixed precision were not captured; the supplement discloses this rather than reconstructing it.
- Figure 1 now correctly says “song-aware partition or holdout rules,” identifies source-credit labels as provenance labels rather than verified performer identities, and avoids the earlier implication that all three tasks used the same fixed split. The regenerated PNG was visually inspected and the 600-dpi TIFF validation passes.
