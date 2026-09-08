# Where Does a Lyrical Identity Live?

This repository is the public, copyright-safe research release for a study of Chinese rap lyrics that asks one representational question of one corpus.

The central question is:

> Where, in the text of a song, does the identity of its source-credit label live — in what the lyrics mean, in the characters they are written with, or in how their lines rhyme — and has a multilingual semantic embedding discarded that identity or merely hidden it?

"Identity" means the corpus-relative profile attached to a credit string. It never means a verified person, biography, hometown, belief, affiliation, influence, or collaboration.

## See the result first

Read [`paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf`](paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf) for the scholarly account, and view the five figures with their source tables in [`figures/index.html`](figures/index.html). The exact computational environment, protocol amendments, leakage groups, label audit, and the whitening probe's details are in [`paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf`](paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf).

[`index.html`](index.html) is a self-contained companion interface built on the earlier corpus snapshot: a 204-label repertoire overview, the provisional cultural-reference links, and a written-ending lookup. It is an interface, not evidence; the article does not cite its repertoire graph.

## Main results

One held-out-song retrieval protocol scores the same **7,236 songs and 226 source-credit labels** in three representation spaces. Label profiles are leave-group-out over **5,888 leakage groups** (exact shared text ∪ near-duplicate songs); nothing is tuned on outcomes.

- **Three spaces, one scale** ([`results/retrieval-v2/`](results/retrieval-v2/)): character 2–5-gram TF–IDF reaches macro MRR **0.398** (95% CI 0.366–0.432) against **0.301** (0.277–0.326) for frozen BGE-M3; their untuned fusion reaches **0.427** (0.396–0.458). A rhyme-form space built from written line endings, with no lexical content, carries identity at plain MRR **0.097** against a chance level of 0.027 — real, small, and redundant with the character surface (`identity_spaces.json`).
- **Not code-switching, not names** (`lexical_identity_decomposition.json`): the songs with the most Latin script are where BGE-M3 does best, and the lexical margin is widest on Han-dominant songs. Stripping a 605-entry catalogue of places, people, brands, slang, and English words — 168,004 characters — moves the lexical system from 0.450 to **0.444**. Whatever the surface carries is diffuse in the characters, not a lexicon.
- **Present but hidden** (`identity_probe.json`): whitening the BGE-M3 song vectors by within-author covariance, fitted on four folds of leakage groups and applied to the fifth, lifts the semantic system from **0.318 to 0.449**, level with the raw lexical one; fused with it, they reach **0.544**. Whitening by the total covariance with no labels already gives 0.432 and permuted labels give 0.431, so almost all of the recovery is the embedding's anisotropy and only +0.017 (interval +0.014 to +0.020) is author structure. The same correction lifts a 1,024-dimensional lexical space from 0.393 to **0.518**, so the surface keeps its lead under equal treatment; and on 34 held-out labels the label-free whitening transfers in full (+0.156) while the label-fitted one falls short of it (−0.027).
- **Written-ending continuation** ([`results/written-rhyme-v2/`](results/written-rhyme-v2/)): 51,516 leakage-safe adjacent-line events from 1,064 held-out songs, partitioned by leakage group. The hierarchical context model reaches Top-3 **0.698** (0.690–0.705), +0.049 over first-order Markov; family switches remain hard (Top-3 0.404) and the source-credit label adds nothing (+0.0002, interval crossing zero).
- **Cultural-reference evidence** ([`results/ner-v1/`](results/ner-v1/), earlier snapshot): the provisional entity inventory contracts from 33 corpus-wide surfaces to 22 after shared-text exclusion and the fixed comparison universe; six source-label-to-place associations and four same-song co-mentions survive uncertainty and BH-FDR screening. Human occurrence gold remains incomplete, so precision, recall, F1, biography, and social-relation claims are withheld. In the article this layer is the lexicon the neutralisation arms remove.
- **Label identity** ([`results/label-identity-v1/`](results/label-identity-v1/)): among 511 label pairs sharing a passage of at least thirty characters, none shows the near-total overlap that one artist filed twice would produce; the 45 pairs at or above 10% mutual involvement were reviewed by the author and are collaborations. No merging is applied.

## One protocol

BGE-M3 is a frozen representation, not the finding. Every space is evaluated inside the same task:

- one query set, one candidate set, one leakage unit;
- leave-group-out label profiles with each (group, label) component weighted one;
- a strong character 2–5-gram TF–IDF comparison and a rhyme-form space that keeps no word;
- per-query score standardisation and untuned equal-weight fusion;
- a macro (per-label) headline with paired two-stage bootstrap intervals, and paired leakage-group bootstrap intervals for every contrast between spaces ([`identity_estimands.json`](results/retrieval-v2/identity_estimands.json) reconciles the two averages);
- controls before conclusions: script-class decomposition, four neutralisation arms, total-covariance and permuted-label whitening, a matched-dimension lexical control, and a held-out-author transfer test.

The retrieval TF–IDF vocabulary and IDF are estimated transductively on the fixed unlabelled query corpus; labels and outcomes never enter them (amendment PD-001). The estimand is fixed-corpus retrieval, not prospective performance.

## Corpus lineage

Amendment PD-002 replaced the legacy cleaner, which deleted chunks whose text recurred under a label, with representation: corpus v2 retains all **25,026 cleaned chunks across 7,391 song records** and carries duplicate structure as **6,025 exact-text components**, of which 799 span more than one song. The reconstruction of the legacy cleaner, the 46-record review queue, and the live-Drive comparison are in [`results/corpus-reconciliation-v1/`](results/corpus-reconciliation-v1/) and [`methods/PROTOCOL_AMENDMENT_PD002_UPSTREAM_CHUNK_DEDUPLICATION.md`](methods/PROTOCOL_AMENDMENT_PD002_UPSTREAM_CHUNK_DEDUPLICATION.md). The retrieval and written-rhyme results above are built on corpus v2 with a single recorded BGE-M3 run (Tesla T4, half precision; contract in [`results/retrieval-v2/analysis_summary.json`](results/retrieval-v2/analysis_summary.json)); the earlier snapshot's vectors, whose device and precision were never recorded, were bounded by a configuration probe ([`results/embedding-configuration-probe-v1/`](results/embedding-configuration-probe-v1/)) and replaced. The v1 results remain in the repository as frozen history and are not comparable point to point.

The `results/retrieval-v1/` and `results/written-rhyme-v1/` artifacts are superseded by their v2 counterparts; the NER and cultural-reference layer has not been rebuilt on corpus v2.

## Repository map

- `paper/` — manuscript and supplementary methods in Markdown, DOCX, and PDF.
- `figures/` — gallery plus five publication figures in PNG, 600-DPI TIFF, PDF, and SVG, with source tables and alt text.
- `results/` — aggregate outputs: `retrieval-v2/` (three spaces, decomposition, probe, purity, estimands), `written-rhyme-v2/`, `label-identity-v1/`, `embedding-configuration-probe-v1/`, `repaired-corpus-v2/`, `corpus-reconciliation-v1/`, `ner-v1/`, and the frozen v1 history.
- `methods/` — frozen research contract, protocol amendments PD-001 and PD-002, journal-format contract, public-release boundary, and author-owned provenance actions.
- `src/` — deterministic builders and validators; `leakage_groups_v2.py` carries the leakage unit and its tests.
- `tools/` — standalone checks that need no build step, including release-integrity verification, the prose-number check, and the compound-resolution gate.
- `tests/` — checks for the logic behind every number the tools report; runnable with `python tests/test_tools.py` and run in CI.
- `analysis/` — post-hoc analysis derived from the released tables.
- `site/` — source for the companion interface.
- `validation/` — independent numerical, manuscript, accessibility, render, and release checks.
- `submission/dsh/` — technically prepared DSH upload bundle and the remaining author checklist.

## Known defect under repair

The lexicon stage of the NER build cannot separate a surface from a longer compound
containing it, so a province name inside a broadcaster name and a city name inside a
company name were counted as mentions of the place. Protocol
[NER-CR-001](methods/NER_CR_001_COMPOUND_RESOLUTION.md) freezes the flag stage and a
blinded per-occurrence verdict for each of the 91 affected occurrences, folded from
retained raw ballots by script: 42 short mentions stand, 49 do not. Longer spans the reference taggers proposed are recorded as new
candidates that must pass the standard release gate; none is published as an entity here. The released NER outputs are unchanged: the repair is sequenced behind an upstream
text-cleaning amendment, and the frozen table is published now so the method can be
reviewed against evidence rather than against a summary.

The adjudication is AI-assisted and independent human review is pending. It is not an
NER accuracy estimate; precision, recall and F1 remain withheld.

```bash
python tools/verify_compound_resolution.py
```

## Evidence boundary

The public release contains no full lyrics, full written lines, song/chunk identifiers, per-song ranks or layout coordinates, row-level lyric-content hashes, embeddings, private membership rows, or reviewer contexts. Short Chinese entity and ending tokens are retained only where needed to interpret aggregate results. File-level SHA-256 checksums and deterministic aggregate join keys are retained as non-content integrity metadata.

Source-credit labels are corpus provenance, not verified natural-person identities. The outputs do not establish authorship, biography, hometown, preference, collaboration, influence, friendship, performed rhyme, flow, voice, or beat.

## Reproduction

The pipeline order and expected private inputs are documented in [`methods/METHODS_AND_LIMITS.md`](methods/METHODS_AND_LIMITS.md). Researchers must supply a lawfully accessible copy of the corpus and private derived sidecars; the copyrighted lyric text is not redistributed here. Every v2 builder refuses a corpus or vector file whose digest differs from the published contract.

The analysis artifacts record CPython 3.12.13. Cross-platform integrity checks run on Python 3.12, while the source syntax requires Python 3.10 or newer.

## Licence

Copyright © 2026 Moshi Fu.

The manuscript, figures, methods, documentation, and aggregate result data are released under [CC BY 4.0](LICENSE). The build and validation code under `src/` and `site/` is released under the [MIT Licence](LICENSE-CODE).

Neither licence covers the underlying lyric corpus, which is not redistributed here. See [`methods/PUBLIC_RELEASE_BOUNDARY.md`](methods/PUBLIC_RELEASE_BOUNDARY.md).

## Integrity

Every published SHA-256 manifest hashes the bytes as committed, and `.gitattributes` disables line-ending translation so a checkout is byte-identical on Windows, macOS, and Linux. To verify a clone on any of those platforms:

```
python tools/verify_release_integrity.py
python src/validate_public_release_integrity_v1.py
python tools/check_manuscript_derivatives.py
```

It runs on every push against Linux, macOS, and Windows, together with the tool tests,
the manuscript-derivative staleness check, and an assertion that no path is
line-ending translated. `src/validate_public_release_integrity_v1.py` performs the
same verification from inside the builder tree.

### Existing Windows checkouts

A Windows checkout created before the repository adopted the byte-exact `.gitattributes` policy can remain clean in `git status` while still containing historical CRLF working-tree bytes. After pulling the current release, first preserve any tracked work on another branch or in a backup and return to a clean checkout of the release commit. Then run:

```
python src/restore_committed_bytes_v1.py
python tools/verify_release_integrity.py
```

The restore helper reads every tracked file from its staged Git blob, writes those exact bytes atomically, verifies the result, and leaves untracked files alone. It refuses to run when tracked changes are present. Do not use `git add --renormalize .` for this migration: that changes staged content instead of restoring the published bytes.

## Submission status

This is a reproducible release that passes the included validations. The retrieval and written-rhyme results are built on the repaired corpus v2 with a recorded embedding run; the cultural-reference layer remains provisional pending independent dual human review of its 800-occurrence package. Before journal submission the authors must supply the author-owned facts listed in `validation/RELEASE_READINESS_V4.md`.
