# Protocol Amendment PD-001: Transductive TF-IDF in Retrieval

**Amendment ID:** PD-001  
**Issued:** 25 August 2026 (America/Chicago)  
**Status:** Retrospective amendment issued after final-audit discovery  
**Scope:** Retrieval task only; `chinese-rap-downstream-retrieval-v1`, version 1.1.0  
**Numerical effect:** None. This amendment changes the declared estimand and claim boundary, not the frozen results.

## Decision

The current retrieval result is retained as a **fixed-corpus, transductive retrieval evaluation**. It must not be described as an inductive evaluation on text that was wholly unseen during representation fitting, and it does not estimate performance on newly arriving songs or a different corpus.

This amendment changes one fitting rule for the retrieval TF-IDF representation only. Every other rule in the original research contract remains in force, including whole-song holdout, exact/near-duplicate controls, group-held-out source-label profiles, outcome-free fusion, paired evaluation, aggregate-only release, and the prohibition on selecting the final model from test outcomes.

## Original frozen contract is preserved

The original contract remains unchanged at `work/downstream_v1/RESEARCH_CONTRACT.md` (SHA-256 `ec5f36c787ff361e302cbdececb9e960975aa13915b327649d53cdc5e4a7d1d8`). Its shared leakage-control clause states:

> Fit vectorizers, vocabularies, scaling parameters, entity priors, rhyme priors, and thresholds on training data only.

The released retrieval implementation does not satisfy that clause for character TF-IDF vocabulary and inverse-document-frequency estimation. The conflict was identified in the final publication-integrity audit on 25 August 2026 and recorded as PRA-003 in `work/audits/paper_reference_audit.md`.

The original file is not edited, replaced, or presented as if it had anticipated this design. PD-001 is the dated audit trail for the deviation.

## Observed implementation and exact estimand

The character baseline uses one whole-song document per member of the fixed retrieval evaluation corpus. `TfidfVectorizer.fit_transform(documents)` estimates the character 2–5-gram vocabulary and IDF values from those fixed **unlabelled texts**, with `min_df=3`, sublinear term frequency, L2 normalization, and a 150,000-feature cap. Consequently, each query song contributes to the unsupervised corpus distribution used to define the lexical feature space and IDF weights.

After representation fitting, the query song and every detected exact or near-duplicate component linked to it are removed from the relevant candidate repertoire profiles. Candidate profiles are means of the remaining group-controlled song representations. Evaluation then asks:

> Within this one fixed corpus, when a song's source-credit label is withheld from the representation-fitting step and the song plus detected duplicate variants are excluded from candidate profiles, how highly does its corpus source-credit label rank among 204 label profiles?

This is a corpus-internal, transductive estimand. The held-out object is the query's labelled profile membership and its contribution to candidate profiles—not the query text's contribution to TF-IDF vocabulary/IDF estimation.

## What was and was not used in fitting

Verified from the released implementation and method record:

- TF-IDF fitting receives document text only; source-credit labels and retrieval ranks are not inputs to `fit_transform`.
- Source-credit labels are used after representation fitting to construct group-held-out repertoire profiles and to score retrieval outcomes.
- BGE-M3 vectors are frozen and are not fitted on this corpus in the retrieval builder.
- Fusion is the equal-weight mean of per-query standardized dense and lexical candidate scores. No fusion weight is tuned on evaluation outcomes.
- The released primary metrics and paired intervals are computed after model construction; the amendment does not use them to alter the representation or select a replacement system.

These controls rule out label-supervised TF-IDF fitting and outcome-tuned fusion. They do **not** turn the design into an inductive evaluation, because the unlabelled query texts still influence the lexical vocabulary and IDF.

## Consequences and limitations

1. **No prospective-corpus claim.** Results apply to retrieval inside the frozen corpus. They do not quantify performance for a later song, an expanded catalogue, or another Chinese-rap collection.
2. **No wholly unseen-text wording.** A query may be called *profile-held-out* or *label-held-out*, but not unseen to TF-IDF fitting.
3. **Asymmetric representation histories.** BGE-M3 is a previously trained frozen encoder, whereas TF-IDF learns an unlabelled lexical distribution from the fixed evaluation corpus. Their comparison remains valid for the stated transductive benchmark, not as a general comparison of inductive encoders.
4. **Corpus-composition dependence.** Adding or removing documents can change the TF-IDF vocabulary and IDF values, so the lexical and fusion rankings are snapshot-specific.
5. **No correction to the reported values.** The audited retrieval values remain the results for this estimand. An inductive estimate would require a new query-safe or fold-safe TF-IDF run and separately versioned artifacts.

## Affected artifacts

The amendment governs interpretation of these existing artifacts and every derivative that reports their retrieval result:

| Artifact | Role | Required treatment |
| --- | --- | --- |
| `work/build_chinese_rap_downstream_retrieval_v1.py` | Frozen builder; SHA-256 `340564076f8a9cbb1b4b90eabab9d82636b5dbaba52333117cecee5deef9b016` | Preserve; do not relabel as train-only TF-IDF. |
| `outputs/chinese-rap-downstream-retrieval-v1/` | Version 1.1.0 retrieval result | Retain values; state the transductive fixed-corpus estimand. |
| `outputs/chinese-rap-downstream-figures-v1/figure_1_research_design.*` | Design disclosure | Must identify fixed-unlabelled-corpus TF-IDF fitting. |
| `outputs/chinese-rap-downstream-figures-v1/figure_2_retrieval_benchmark.*` | Numerical retrieval comparison | Interpret only under PD-001's fixed-corpus boundary. |
| `work/paper_v3/manuscript.md` | Article draft | Use *transductive*, qualify held-out language, and state no future-corpus generalization. |
| Companion result site and release package | Public interpretation | Preserve the same limitation wherever retrieval results appear. |

## Why this amendment is not backdated

The inconsistency was discovered after the retrieval artifacts existed, during final publication-integrity review. Backdating PD-001—or silently rewriting the frozen contract—would falsely imply that the deviation had been specified before results were known. This document therefore records the real chronology. It is a transparent retrospective governance correction, **not** a preregistration and not evidence that the transductive choice was outcome-independent when originally made.

## Requirements for any later inductive result

A future inductive version must receive a new artifact/version identifier, fit TF-IDF vocabulary and IDF without each evaluated query or test partition, repeat all metrics and paired uncertainty, regenerate dependent figures/site data, and report the result alongside—not in place of—the present transductive benchmark.

