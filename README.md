# Language, Reference, and Written Rhyme

This repository is the public, copyright-safe research release for an evidence-grounded study of Chinese rap lyrics.

The central question is:

> How do Chinese rap lyrics form recognizable lyrical identities through language, cultural reference, and dictionary-estimated written rhyme?

## See the result first

Open [`index.html`](index.html). It is a self-contained results interface that works without a server and supports three research-backed actions:

1. start with a 204-label corpus overview, then inspect one label's evidence-graded lyrical-repertoire neighbours;
2. explore statistically screened, provisional cultural-reference links and co-mentions;
3. enter a Chinese line-final character to inspect its dictionary-estimated ending family and possible next-family transitions.

For the scholarly account, read [`paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf`](paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf). View the four questions and figures in [`figures/index.html`](figures/index.html). The exact computational environment, protocol amendment, and public/private boundary are in [`paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf`](paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf).

## Main results

- **Explainable repertoire retrieval:** 5,455 held-out songs across 204 source-credit labels. Untuned dense–lexical fusion reaches macro MRR **0.447** (95% CI 0.414–0.481) and Recall@10 **0.611** (0.577–0.646). Fusion improves over character TF–IDF by **0.031 MRR** (0.021–0.042).
- **Descriptive repertoire companion:** a separate BGE-M3-only map retains 86 reciprocal top-five matches under both duplicate-controlled text treatments, connecting 93 of 204 labels. Sixteen matches reappear in at least half of 250 within-label song resamples. Global PCA position is approximate (26.2% of profile variation); only a line defines a released match.
- **Cultural-reference evidence:** the provisional entity inventory contracts from 33 corpus-wide surfaces to 22 after shared-text exclusion and the fixed comparison universe. Six source-label-to-place associations and four same-song reference co-mentions survive uncertainty and BH-FDR screening. Human occurrence gold remains incomplete, so precision, recall, F1, biography, and social-relation claims are withheld.
- **Written-ending continuation:** 34,395 leakage-safe adjacent-line events from 787 held-out songs. The hierarchical context model reaches Top-3 **0.695** (0.685–0.705), improving on first-order Markov by **0.050** (0.044–0.055). Switch Top-1 remains only **0.026**, and source-credit-label conditioning has no supported benefit.

## Corpus-lineage audit — not a fourth downstream task

PD-002 reconstructs the historical raw-to-frozen cleaner as release lineage rather than adding another research task. The legacy cleaner produced **7,214 songs and 22,132 chunks** after artist-level exact-text keep-first deduplication; subsequent canonical identity and eligibility gates withheld three songs and four chunks, yielding the **7,211-song, 22,128-chunk** downstream input used by the current analyses.

The reconstruction shows that the legacy chunk rule removed 2,894 chunks and erased 177 song records. Of those records, 131 meet the conservative same-label, same-normalized-title, exact-sequence duplicate rule; 46 remain a review queue. Aggregate written-ending family shares are insensitive to a task-aligned restored-chunk counterfactual, but predictive metrics have not been retrained on a duplicate-aware repaired population. The frozen-snapshot results therefore remain reproducible while repaired-corpus predictive robustness is withheld.

See [`results/corpus-reconciliation-v1/`](results/corpus-reconciliation-v1/) for aggregate evidence and [`methods/PROTOCOL_AMENDMENT_PD002_UPSTREAM_CHUNK_DEDUPLICATION.md`](methods/PROTOCOL_AMENDMENT_PD002_UPSTREAM_CHUNK_DEDUPLICATION.md) for the replacement rule. The live-Drive comparison is retained publicly only as aggregate row-count, mismatch-class, and adjudication evidence; it does not verify remote-object byte identity, acquisition provenance, rights, or universal metadata accuracy.

## What happens after BGE-M3

BGE-M3 is a frozen representation, not the finding. It is evaluated inside a downstream task with:

- full-song holdout and duplicate-component removal;
- a strong character 2–5-gram TF–IDF baseline;
- per-query score standardization and untuned equal-weight fusion;
- label-balanced estimation and paired two-stage bootstrap uncertainty;
- explicit explanations and claim boundaries.

The retrieval TF–IDF vocabulary and IDF are estimated transductively on the fixed unlabeled evaluation corpus. Source labels and outcomes are not used for fitting, but the estimand is fixed-corpus retrieval rather than prospective performance. The original contract and formal amendment are both retained.

## Repository map

- `paper/` — final English manuscript and supplementary methods in Markdown, DOCX, and PDF.
- `figures/` — visual gallery plus four publication figures in PNG, 600-DPI TIFF, PDF, and SVG, with source tables and alt text.
- `results/` — aggregate input-audit, corpus-lineage reconciliation, retrieval, reproducible repertoire-network, NER/cultural-reference, and written-ending outputs.
- `methods/` — frozen research contract, protocol amendments, journal-format contract, pipeline explanation, public-release boundary, and author-owned provenance actions.
- `src/` — deterministic builders and validators.
- `site/` — source for the richer local results application.
- `validation/` — independent numerical, manuscript, accessibility, render, and release checks.
- `submission/dsh/` — technically prepared DSH upload bundle and the remaining author checklist.

## Evidence boundary

The public release contains no full lyrics, full written lines, song/chunk identifiers, row-level lyric-content hashes, embeddings, private membership rows, or reviewer contexts. Short Chinese entity and ending tokens are retained only where needed to interpret aggregate results. File-level SHA-256 checksums and deterministic aggregate join keys are retained as non-content integrity metadata.

Source-credit labels are corpus provenance, not verified natural-person identities. The outputs do not establish authorship, biography, hometown, preference, collaboration, influence, friendship, performed rhyme, flow, voice, or beat.

## Reproduction

The pipeline order and expected private inputs are documented in [`methods/METHODS_AND_LIMITS.md`](methods/METHODS_AND_LIMITS.md). Researchers must supply a lawfully accessible copy of the frozen corpus and private derived sidecars; the copyrighted lyric text is not redistributed here.

The frozen analysis artifacts record CPython 3.12.13. Cross-platform integrity checks run on Python 3.12, while the source syntax requires Python 3.10 or newer. The historical BGE-M3 device, CUDA, and half-precision state was not retained, so the release does not claim CPU/GPU or fp32/fp16 invariance.

## Licence

Copyright © 2026 Moshi Fu.

The manuscript, figures, methods, documentation, and aggregate result data are released under [CC BY 4.0](LICENSE). The build and validation code under `src/` and `site/` is released under the [MIT Licence](LICENSE-CODE).

Neither licence covers the underlying lyric corpus, which is not redistributed here. See [`methods/PUBLIC_RELEASE_BOUNDARY.md`](methods/PUBLIC_RELEASE_BOUNDARY.md).

## Integrity

Every published SHA-256 manifest hashes the bytes as committed, and `.gitattributes` disables line-ending translation so a checkout is byte-identical on Windows, macOS, and Linux. To verify a clone on any of those platforms:

```
python src/validate_public_release_integrity_v1.py
python tools/check_manuscript_derivatives.py
```

### Existing Windows checkouts

A Windows checkout created before the repository adopted the byte-exact `.gitattributes` policy can remain clean in `git status` while still containing historical CRLF working-tree bytes. After pulling the current release, first preserve any tracked work on another branch or in a backup and return to a clean checkout of the release commit. Then run:

```
python src/restore_committed_bytes_v1.py
python src/validate_public_release_integrity_v1.py
```

The restore helper reads every tracked file from its staged Git blob, writes those exact bytes atomically, verifies the result, and leaves untracked files alone. It refuses to run when tracked changes are present. Do not use `git add --renormalize .` for this migration: that changes staged content instead of restoring the published bytes.

## Submission status

The frozen-snapshot technical and analytical release passes the included validations and is shareable with the disclosed PD-002 boundary. It is not journal-submission-ready until the duplicate-aware corpus repair and affected predictive reruns are complete. The authors must also complete the factual items in [`methods/DATA_PROVENANCE_AND_AUTHOR_ACTIONS.md`](methods/DATA_PROVENANCE_AND_AUTHOR_ACTIONS.md), including authorship, funding, rights, acquisition provenance, ethics determination, and an archival DOI.
