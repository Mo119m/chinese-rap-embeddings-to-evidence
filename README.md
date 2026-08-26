# Language, Reference, and Written Rhyme

This repository is the public, copyright-safe research release for an evidence-grounded study of Chinese rap lyrics.

The central question is:

> How do Chinese rap lyrics form recognizable lyrical repertoires through language, cultural reference, and dictionary-estimated written rhyme?

## See the result first

Open [`index.html`](index.html). It is a self-contained results interface that works without a server and supports three research-backed actions:

1. inspect explainable held-out-song lyrical-repertoire relations;
2. explore statistically screened, provisional cultural-reference links and co-mentions;
3. enter a Chinese line-final character to inspect its dictionary-estimated ending family and possible next-family transitions.

For the scholarly account, read [`paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf`](paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf). The exact computational environment, protocol amendment, and public/private boundary are in [`paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf`](paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf).

## Main results

- **Explainable repertoire retrieval:** 5,455 held-out songs across 204 source-credit labels. Untuned dense–lexical fusion reaches macro MRR **0.447** (95% CI 0.414–0.481) and Recall@10 **0.611** (0.577–0.646). Fusion improves over character TF–IDF by **0.031 MRR** (0.021–0.042).
- **Cultural-reference evidence:** the provisional entity inventory contracts from 33 corpus-wide surfaces to 22 after shared-text exclusion and the fixed comparison universe. Six source-label-to-place associations and four same-song reference co-mentions survive uncertainty and BH-FDR screening. Human occurrence gold remains incomplete, so precision, recall, F1, biography, and social-relation claims are withheld.
- **Written-ending continuation:** 34,395 leakage-safe adjacent-line events from 787 held-out songs. The hierarchical context model reaches Top-3 **0.695** (0.685–0.705), improving on first-order Markov by **0.050** (0.044–0.055). Switch Top-1 remains only **0.026**, and source-credit-label conditioning has no supported benefit.

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
- `figures/` — four publication figures in PNG and 600-DPI TIFF, with source tables and alt text.
- `results/` — aggregate input-audit, retrieval, NER/cultural-reference, and written-ending outputs.
- `methods/` — frozen research contract, protocol amendment, journal-format contract, pipeline explanation, public-release boundary, and author-owned provenance actions.
- `src/` — deterministic builders and validators.
- `site/` — source for the richer local results application.
- `validation/` — independent numerical, manuscript, accessibility, render, and release checks.

## Evidence boundary

The public release contains no full lyrics, full written lines, song/chunk identifiers, row-level lyric-content hashes, embeddings, private membership rows, or reviewer contexts. Short Chinese entity and ending tokens are retained only where needed to interpret aggregate results. File-level SHA-256 checksums and deterministic aggregate join keys are retained as non-content integrity metadata.

Source-credit labels are corpus provenance, not verified natural-person identities. The outputs do not establish authorship, biography, hometown, preference, collaboration, influence, friendship, performed rhyme, flow, voice, or beat.

## Reproduction

The pipeline order and expected private inputs are documented in [`methods/METHODS_AND_LIMITS.md`](methods/METHODS_AND_LIMITS.md). Researchers must supply a lawfully accessible copy of the frozen corpus and private derived sidecars; the copyrighted lyric text is not redistributed here.

## Submission status

The technical and analytical release passes the included validations. Before journal submission, the authors must complete the factual items in [`methods/DATA_PROVENANCE_AND_AUTHOR_ACTIONS.md`](methods/DATA_PROVENANCE_AND_AUTHOR_ACTIONS.md), including authorship, funding, rights, acquisition provenance, ethics determination, licences, and an archival DOI.
