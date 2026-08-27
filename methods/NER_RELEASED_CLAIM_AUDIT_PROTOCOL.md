# Released-claim NER occurrence audit protocol

Version 1.0.0 · 27 August 2026

## Purpose

The public Chinese-rap cultural-reference layer currently releases six statistically screened source-credit-label/entity associations and four same-song entity co-mentions. Those claims pass the frozen automated support, uncertainty, shared-text-exclusion, and BH-FDR gates, but their contributing entity occurrences have not yet been reviewed by humans.

This protocol adds a deliberately narrow audit:

> Do the private lyric occurrences that directly support every currently released NER claim express the proposed named reference, with the proposed span and entity type, in their local context?

The audit covers the released claims themselves rather than a representative sample of the whole NER candidate population. It can support an occurrence-confirmation statement about the released claims. It cannot estimate corpus-wide NER precision, recall, or F1.

## Current frozen scope

The builder reconciles the current released tables to the private occurrence ledger before creating any review task. The v1 package contains:

- 6 released source-credit-label/entity claims;
- 4 released same-song entity co-mention claims;
- 102 occurrence rows contributing to the six label/entity claims;
- 60 occurrence rows contributing to the four co-mention claims;
- 5 occurrence rows shared by both claim families;
- 157 unique released-claim occurrence rows in total;
- 23 released co-mention claim × supporting-song review tasks;
- 20 real non-contributing comparator candidates; and
- 16 synthetic boundary-corruption attention controls;
- 8 real non-contributing same-song pair comparators.

The 193 occurrence tasks therefore contain all 157 released-claim occurrences plus 36 blinded occurrence controls. The co-mention sheets contain the 23 released support tasks plus 8 blinded pair comparators, for 31 pair-review rows. No released-claim occurrence or supporting song unit is sampled away or deduplicated. Exact current counts and hashes are recorded in `results/ner-v1/released_claim_audit_status.json` and the private manifest.

## Evidence and unit boundaries

### Occurrence task

One occurrence task is one machine-proposed character span in one private lyric context. Source-row occurrences remain separate even when their displayed text repeats. This preserves exact lineage to the rows used by the released analysis.

### Label/entity support unit

The inferential support unit for a source-credit-label/entity claim is a distinct shared-text-excluded full-song-content unit within that source-credit label. Several reviewed occurrence rows may collapse to one support unit.

### Co-mention support unit

The inferential support unit for a co-mention claim is a distinct shared-text-excluded full-song-content unit containing at least one occurrence of each proposed entity. Code verifies song co-location from the private hashes. Human reviewers assess whether each side is a valid reference in context; they do not infer collaboration, influence, affiliation, identity, or any other social relation.

### Claim-conditioned scope

The released-claim audit has no corpus-wide negative frame and does not measure missed entities. It must not be described as a general NER test set, a random sample, or completed human gold for the corpus.

## Reproducible selection

Run from the repository root:

```powershell
python src/build_ner_released_claim_audit_v1.py
```

The builder reads:

1. the private `all_candidate_occurrences_private.csv` ledger from the frozen NER build;
2. the six public `source_label_entity_links_provisional.csv` rows;
3. the four public `entity_co_mentions_provisional.csv` rows;
4. the private and public manifests that authenticate those inputs.

An occurrence is claim-contributing only if it:

1. has strict cross-method span/type consistency;
2. is not attached to a cleaned-text hash shared across source-credit labels;
3. matches the released entity surface and entity type; and
4. belongs to the released source label for a label/entity claim, or to a song unit containing both released entities for a co-mention claim.

The builder independently recomputes each released claim's support count and stops if any count differs from the public table. Stable hashed task IDs and reviewer-specific deterministic orderings use the fixed seed stored in the private manifest.

The private output target is exactly `work/private-chinese-rap-ner-released-claim-audit-v1/` at the workspace level. The builder refuses to place lyric contexts inside the repository. It also refuses to overwrite an existing package unless `--replace-empty` is supplied and every review and adjudication decision remains blank.

## Review design

The design is independent dual review, blinded to claim status and control status, followed by adjudication.

1. A coordinator retains all admin manifests and the negative-control key.
2. Reviewer 1 receives only the R1 occurrence sheet, the R1 co-mention sheet, and this protocol.
3. Reviewer 2 receives only the corresponding R2 files and this protocol.
4. The reviewers work independently and do not compare decisions, search the public result for task identity, or consult one another.
5. Each reviewer locks and returns both sheets before adjudication begins.
6. The coordinator transfers both decisions into the adjudication templates.
7. The adjudicator resolves disagreements while still blinded to the negative-control key.
8. Control identities are revealed only after adjudication is locked.

“Blinded” here means that reviewers cannot see whether a row contributes to a released claim, is a real comparator, or is a synthetic attention control, and cannot see the other reviewer's response. Reviewers necessarily see the proposed span and type because those are the objects being evaluated.

## Occurrence annotation

The target is marked inside the private context as `⟦TARGET⟧…⟦/TARGET⟧`. Review the marked use, not the surface in isolation and not any presumed biography of the credited source.

### Required fields and allowed values

| Field | Allowed values | Decision rule |
|---|---|---|
| `mention_valid` | `YES`, `NO`, `UNCERTAIN` | Does the marked text function as a referential expression in this lyric context? |
| `boundary_valid` | `YES`, `NO`, `UNCERTAIN` | Is the marked span the shortest complete expression, without missing or extra characters? |
| `referential_status` | `NAMED_REFERENCE`, `LANGUAGE_REFERENCE`, `GENERIC_OR_COMMON`, `FIGURATIVE_OR_METONYMIC`, `CREDIT_OR_METADATA`, `OTHER`, `UNCERTAIN` | What kind of contextual use is present? |
| `entity_type_decision` | one schema type, `NOT_ENTITY`, or `UNCERTAIN` | What type does the marked use express? |
| `normalized_surface` | free text or blank | Enter the shortest complete surface when the proposed boundary is wrong; otherwise repeat the target or leave blank consistently. |
| `confidence_1_to_5` | integer 1–5 | 1 = highly uncertain; 5 = highly certain. |
| `exclusion_reason` | controlled short reason or blank | Required when `mention_valid=NO` or `boundary_valid=NO`. |
| `notes` | concise free text | Explain ambiguity or evidence needed for adjudication. Do not add unsupported real-world facts. |
| `reviewed_at_utc` | ISO 8601 UTC timestamp | Record when the row was completed. |

Schema types available to the released claims are `PLACE` and `LANGUAGE_OR_DIALECT_REFERENCE`. If the context clearly supports another project schema type, enter that type rather than forcing the proposed type. Use `UNCERTAIN` when the local text cannot resolve the use.

An adjudicated occurrence is valid for its proposed claim only when:

1. `mention_valid=YES`;
2. `boundary_valid=YES`;
3. `referential_status` is compatible with the proposed claim; and
4. the adjudicated entity type equals the proposed released type.

Figurative or metonymic uses are not automatically invalid. The reviewer must decide whether the marked surface still functions as the proposed named reference in context and explain the choice. The adjudicator applies the same rule consistently across occurrences.

## Co-mention review

Each co-mention row bundles every selected occurrence of entity A and entity B in one code-verified supporting song unit. The reviewer records:

- `entity_a_has_valid_reference`: `YES`, `NO`, or `UNCERTAIN`;
- `entity_b_has_valid_reference`: `YES`, `NO`, or `UNCERTAIN`;
- `pair_semantically_supported`: `YES`, `NO`, or `UNCERTAIN`;
- confidence and a concise note.

Set `pair_semantically_supported=YES` only when at least one occurrence on each side is a valid reference of the proposed type. This decision confirms a same-song textual co-occurrence only. It does not name a relationship between people, places, languages, groups, or credited sources.

## Controls

### Real non-contributing occurrence comparators

Twenty real candidate contexts are drawn deterministically from shared-text-excluded candidates that do not contribute to the released claims and lack strict cross-method consistency. They are matched approximately to the released occurrence type distribution and diversified across surfaces and source labels.

These rows have no assumed gold answer. Their adjudicated acceptance rate is a descriptive comparator for indiscriminate approval and anchoring; it is not a false-positive rate and is not included in corpus-wide precision, recall, or F1.

### Real non-contributing co-mention comparators

Eight additional pair rows are drawn from strict, shared-text-excluded references that occur in the same private song unit but do not belong to the released co-mention set. Pair surfaces and supporting songs are diversified deterministically. These rows mask released pair status in the reviewer sheets.

They also have no assumed gold answer: a non-released pair may contain two valid references but fail a support, uncertainty, or multiplicity gate elsewhere in the corpus. Use their adjudicated acceptance descriptively; never call them false co-mentions or use them to estimate specificity.

### Synthetic boundary attention controls

Sixteen released-claim contexts are duplicated with the marked boundary deliberately expanded by one adjacent character. The corrupted span is rejected if it equals any known candidate surface or released surface. The expected outcome is `boundary_valid=NO`.

These controls test whether reviewers inspect span boundaries. They are synthetic quality checks, not natural negative examples. Exclude them from scientific occurrence-confirmation rates and inter-annotator agreement for real tasks. Report their pass rate separately and investigate any miss before unblinding or scoring the released claims.

## Adjudication and claim re-evaluation

The coordinator first verifies that both reviewers completed every required field. Report raw agreement and Cohen's κ for the principal categorical decisions on real tasks only, accompanied by the contingency table and prevalence. Because the audit is claim-conditioned, κ describes protocol reliability on this audit set, not model performance in the corpus.

After adjudication:

1. exclude invalid occurrences according to the frozen decision rule;
2. collapse remaining valid occurrences to the original full-song support units;
3. recompute each released label/entity and co-mention support count;
4. rerun the original support, uncertainty, enrichment, and multiplicity gates rather than editing public counts by hand;
5. withdraw or downgrade any claim that no longer passes its original gate; and
6. update the public audit status with completed counts, agreement, control results, and an exact manifest hash.

A label/entity claim is not “occurrence-audited” until all of its contributing occurrence tasks are adjudicated. A co-mention support unit remains valid only when at least one adjudicated-valid occurrence of each entity survives in that song unit.

## Permitted reporting

Before completion, report only:

- package coverage;
- the number of blank dual-review and adjudication tasks; and
- hash/validation status.

After completion, the project may report:

- the adjudicated confirmation rate among released-claim occurrences;
- per-claim valid occurrence and support-unit counts;
- the number of released claims that retain their original gate;
- reviewer agreement on real audit tasks; and
- the synthetic boundary-control pass rate, separately.

The project must continue to withhold corpus-wide NER precision, recall, and F1 until the separately planned representative 800-item dual-review benchmark is completed. The targeted audit cannot estimate recall because it contains no exhaustive entity annotation frame, and its claim-conditioned composition cannot be treated as a random precision sample.

## Privacy and rights boundary

Reviewer sheets, adjudication sheets, task manifests, control keys, source locators, lyric hashes, and context snippets are private local research material. They must not be committed to Git, placed in the desktop public release, posted to an issue, or shared through an unapproved service.

The repository contains only:

- this protocol;
- the deterministic builder; and
- `results/ner-v1/released_claim_audit_status.json`, which contains aggregate counts, states, and hashes but no lyric text, context, song/chunk locator, or absolute private path.

Reviewers must have lawful access to the source material and follow the project's corpus-rights, ethics, and data-custody requirements. Delete working copies according to the approved retention plan after adjudication and archival verification.

## Validation and integrity

The builder records SHA-256 hashes for the private occurrence source, released public claim tables, source manifests, every private output file, the private validation record, and the private package manifest. It checks:

- input-manifest integrity;
- unique source occurrence IDs;
- exact context/target offset resolution;
- complete released-claim occurrence coverage;
- exact reconciliation of all public support counts;
- one pair task per released co-mention support unit;
- requested occurrence- and pair-control counts and control semantics;
- independent reviewer row orders;
- absence of coordinator-only fields in reviewer templates;
- blank review and adjudication decisions;
- the exact private output boundary; and
- absence of private schema/path tokens from the public status file.

Package validation passing means that the audit instrument is complete and internally consistent. It does not mean that any human decision, released occurrence, or NER accuracy metric has been validated.
