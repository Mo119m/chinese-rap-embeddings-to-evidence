# Data Provenance: Verified Local Record and Author Actions

**Purpose:** This note separates facts verified in the local research workspace from corpus-acquisition information that only the data owner can supply. It is a submission-support document, not a substitute for the paper's final provenance, rights, or ethics statement.

## Verified local facts

- The project received and analyzes a **private, frozen supplied snapshot**. The local record does not establish who originally acquired the lyrics, from which platforms, by what collection mechanism, or on what dates.
- The historical cleaner's legacy frozen output contains **7,214 songs** and **22,132 lyric chunks**. These are lineage-stage counts before the later canonical identity and downstream-eligibility gates; they must not be presented as the current canonical analysis population.
- The downstream canonical input contains **7,211 songs** and **22,128 canonical lyric chunks**. Three songs and four chunks from the legacy cleaner output are withheld by the later identity/eligibility gates. The cleaned sidecar has the same 22,128 canonical chunk keys. These counts pass the 12 structural checks in `outputs/chinese-rap-downstream-input-audit-v1/validation.json`.
- The corresponding clean-text sidecar marks 21,553 chunks eligible for analysis and withholds 575 metadata-only chunks. These are processing statuses, not evidence about the original acquisition route.
- The source metadata export used for lineage checks has **no per-track source-URL column and no stable platform-ID column**. Internal artist/title agreement therefore is not independent discographic verification.
- The Drive comparison used by PD-002 is bounded to aggregate row-count, mismatch-class, and adjudication evidence from a row-aligned comparison of a live native Sheet with the local raw export. The public artifact does not contain Drive rows, lyrics, titles, labels, identifiers, or row-level hashes. It records equal row counts, a hash binding to the current local raw export, and zero unresolved substantive mismatches after declared adjudications, while **remote Drive object byte identity remains unverified**. This comparison does not establish acquisition provenance, custody before the supplied snapshot, rights, representativeness, or universal record accuracy.
- Exactly **four song-title fields** have approved external evidence in `work/data-drop/canonical-corpus-evidence-v1/approved_identity_overrides.csv`. Raw source and cleaned values are preserved in that evidence layer. These four field-level corrections do not verify the remaining corpus, performer identity, lyric accuracy, or collection provenance.
- The analysis treats source-credit strings as corpus labels, not independently verified performer identities.
- Public outputs are aggregate-only. They exclude lyric/full-line text, song and chunk identifiers, row-level lyric-content hashes, embeddings, private membership rows, and reviewer contexts. Short Chinese entity or rhyme-class surfaces may appear only as bounded analytic evidence. File-level checksums and deterministic aggregate join keys may remain as non-content integrity metadata.

## Local evidence trail

| Evidence | Verified fact | SHA-256 |
| --- | --- | --- |
| `outputs/chinese-rap-downstream-input-audit-v1/analysis_summary.json` | Frozen counts, coverage, identity-status totals, duplicate risk, and claim boundaries | `b8dbcf835e303d10cbec6dc68fd107a6c30a64ed948e1330c9c842ea0c15b898` |
| `outputs/chinese-rap-downstream-input-audit-v1/input_manifest.json` | Exact private input files and hashes used by the downstream audit | `52abc68a887b7faa52bc3d81c4396d57b7601eed53b1d79466aead05af92c1e0` |
| `outputs/chinese-rap-corpus-quality-v1/manifest.json` | Private-content classification, four approved title overrides, and missing track-level URL/stable-ID limitation | `265708db6d66a5359f7a085657a02944fed0d9af1f070f70b9ad73e22ae7d07e` |
| `work/data-drop/canonical-corpus-evidence-v1/approved_identity_overrides.csv` | Four approved, evidence-backed title-field decisions | `8a315bea65ef2316ba83f73b3f73e4de63e4c0cb98f6d0274004310255c4ec62` |
| `results/corpus-reconciliation-v1/analysis_summary.json` | Aggregate reconstruction of the legacy 7,214-song/22,132-chunk cleaner output, duplicate-loss diagnostics, task-aligned rhyme sensitivity, Drive-comparison boundary, and outstanding release action | `b44e5c7627f035f187088d690f655332151b1f40c7cd1ba9e7ec11c1923c6460` |
| `methods/PROTOCOL_AMENDMENT_PD002_UPSTREAM_CHUNK_DEDUPLICATION.md` | Post-freeze amendment separating legacy chunk deletion from duplicate-aware song/component control | `0e3676259f0caa7f3564dd2147be8c442eda4834a6a205851b29741bfac34270` |

Passing these local audits establishes structural lineage and the stated exception counts. It does **not** establish original acquisition provenance, legal permission, ethical approval, representativeness, or universal metadata accuracy.

## Corpus-lineage release action

The corpus reconciliation is a release audit, not a fourth downstream task. It exactly reconstructs the legacy cleaner and shows that artist-level exact-cleaned-text keep-first deletion removed 2,894 chunks and erased 177 song records before canonical analysis. The conservative duplicate-record stratum contains 131 records; 46 records remain review-required.

The current frozen-snapshot results remain reproducible within their declared 7,211-song/22,128-chunk population. However, aggregate counterfactual stability does not establish that held-out retrieval, repertoire-network stability, cultural-reference evidence, or written-ending prediction is unchanged after corpus repair. Before journal submission, the computational team must preserve song identity and within-song order, apply duplicate-aware song/component control, and rerun the affected downstream evaluations. Until then, repaired-corpus predictive-robustness claims are withheld.

## Information that remains owner-supplied

No verified local record currently resolves the following items. The data owner or responsible author must complete them before submission.

| Required author field | What the author must provide | Status |
| --- | --- | --- |
| Source platform(s) or source classes | Exact names where disclosure is permitted; otherwise a reasoned source-class description and why names cannot be disclosed | **Owner to supply** |
| Acquisition route and mechanism | Download/export/API/scrape/manual transfer method; tool or script where known; who performed it | **Owner to supply** |
| Acquisition dates | Start/end dates or best documented date range; distinguish collection date from song-release date | **Owner to supply** |
| Sampling frame | Inclusion and exclusion rules; how artists/tracks were selected; whether the collection was exhaustive, convenience-based, curated, or otherwise bounded | **Owner to supply** |
| Corpus temporal coverage | Earliest/latest release dates where known and the amount of missing date metadata | **Owner to supply** |
| Lyric provenance | Whether text was platform-supplied, user-contributed, transcribed, OCR-derived, or otherwise obtained; known corrections or normalization before receipt | **Owner to supply** |
| Track-level provenance | Any lawful source URLs, platform IDs, catalogue IDs, or archived manifests kept outside this export; coverage and missingness | **Owner to supply** |
| Snapshot custody | Original provider/custodian, date supplied to this project, freeze date/time, and any earlier versions | **Owner to supply** |
| Rights and terms | Copyright/terms-of-service/licence basis for collection, local analysis, quotation, retention, and aggregate publication | **Owner to supply** |
| Ethics determination | IRB/ethics-board reference or documented determination that review was not required; privacy and researcher-risk assessment | **Owner to supply** |
| Representation limits | Known genre, region, language, gender, era, platform, popularity, or repertoire biases; known missing artists/tracks | **Owner to supply** |
| Reproducibility access | Who may access the private snapshot, under what conditions, for how long, and whether a controlled-access route exists | **Owner to supply** |

## Concise author fill-in record

Complete this block from documentary evidence, not memory alone where records exist.

| Field | Author response | Supporting record / location | Confirmed by / date |
| --- | --- | --- | --- |
| Platform(s) / source class(es) |  |  |  |
| Acquisition mechanism |  |  |  |
| Acquisition date range |  |  |  |
| Song-release temporal range |  |  |  |
| Inclusion/exclusion rule |  |  |  |
| Lyric/transcription origin |  |  |  |
| Track-level URL/ID coverage |  |  |  |
| Provider, receipt date, freeze date |  |  |  |
| Rights/licence/terms basis |  |  |  |
| Ethics/IRB determination |  |  |  |
| Known sampling bias or missingness |  |  |  |
| Private-access/reproducibility policy |  |  |  |

## Do-not-invent rule

**Do not infer missing provenance from filenames, artist or title strings, lyric content, current search-engine results, or the present availability of a track on a platform. Do not convert an assumption into a date, platform, permission, sampling rule, or ethics determination.** If an item cannot be supported by an owner statement or contemporaneous record, write `Unknown—not supplied`, explain the limitation in the paper, and narrow the corresponding claim.

The four approved title overrides may be reported as four evidence-backed field corrections only. They must not be used to imply broad external verification of the dataset.

## Minimum insertion required before submission

The responsible author should use the completed record to add one compact corpus-provenance paragraph and, if space permits, a supplement table covering source, acquisition, sampling, temporal scope, lyric origin, missingness, rights, ethics, freeze/custody, and access restrictions. Until that owner-supplied information exists, the paper must describe the corpus only as a private, frozen supplied snapshot and explicitly state that representativeness, acquisition provenance, and the rights/ethics basis are unresolved in the local computational record.
