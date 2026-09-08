# Pipeline, methods, and claim boundaries

This document is the plain-language map of what the builders do, in what order, and what each output may and may not be taken to mean. The frozen research contract and its amendments PD-001 and PD-002 are the authority where they and this document differ.

## Pipeline

1. **Freeze and audit the corpus.** Reconcile the raw export to one row per chunk, record the four historical cleaning stages, and row-align the live Drive sheet with the raw CSV (26,833 exact song/chunk keys; 33/33 substantive cell differences adjudicated). Amendment PD-002 replaces the legacy cleaner's deletion of repeated `(label, chunk text)` rows with representation: corpus v2 retains all 25,026 cleaned chunks over 7,391 song records and carries duplicate structure as 6,025 exact-text components. Nothing is guessed for missing metadata.
2. **Embed once, and record the run.** Generate BGE-M3 dense vectors for every corpus v2 chunk in a single recorded run (Tesla T4, half precision, batch 8, maximum length 2,048, corpus order) whose contract pins the checkpoint weights, the corpus digest, the device, and the precision; a resume under any other configuration is refused. The earlier vectors' unrecorded configuration is bounded by `results/embedding-configuration-probe-v1/` and not used.
3. **Build the leakage groups.** Over the query population, join songs that share an exact-text component with songs whose whole-song trigram Jaccard is at least 0.80; neither rule contains the other. `src/leakage_groups_v2.py` carries the rule and its tests. Audit the labels for one artist filed under two strings (`results/label-identity-v1/`).
4. **Score three spaces under one protocol** (`results/retrieval-v2/`). Represent each song once in the semantic space (BGE-M3 centroid), the lexical space (character 2–5-gram TF–IDF), and the rhyme-form space (TF–IDF over written-ending tokens, no words). Build leave-group-out label profiles with each (group, label) component weighted one, rank the true label among 226, and report a macro headline with paired two-stage bootstrap intervals and paired leakage-group intervals for every contrast between spaces.
5. **Decompose the lexical advantage.** Split each query's correct-label lexical score by script class, stratify by Latin share, and strip four named-reference lexicons — places, all named entities, the whole 605-surface catalogue, and the 90-entity reference core — before refitting the lexical space.
6. **Probe the semantic space.** Whiten the BGE-M3 song vectors by within-author covariance (Ledoit–Wolf shrinkage), fitted on four folds of leakage groups and applied to the fifth, with total-covariance, permuted-label, and centring controls, a matched-dimension lexical control, and a held-out-author transfer test.
7. **Build the written-ending continuation task on corpus v2** (`results/written-rhyme-v2/`). Convert strict terminal-Han lines to 17 tone-free pinyin-final families, partition songs by leakage group, exclude exact and near-duplicate target leakage, and compare global, Markov, flat-context, and hierarchical continuation/switch models with and without the source-credit label. Select hyperparameters and abstention thresholds on validation only.
8. **Keep the cultural-reference layer as description and as a control.** The screened lexicon and the pinned contextual Chinese NER baseline, cross-method agreement, shared-text exclusion, and the uncertainty and BH-FDR gates are unchanged from the earlier snapshot (`results/ner-v1/`); their accuracy is withheld pending the 800-occurrence dual review, and the lexicon supplies the neutralisation arms of step 5.
9. **Render evidence.** Write figure-source tables from the v2 results, render the five publication figures, build the DOCX/PDF derivatives from `paper/manuscript.md`, and run the numerical, privacy, accessibility, render, and release-integrity checks.

## Why BGE-M3

The corpus is primarily Chinese with mixed-script passages and variable-length chunks. BGE-M3 supplies a multilingual dense retrieval representation and is frozen rather than fine-tuned. It is the object of study, not the method: the question is what its geometry keeps of a source label's identity. Its raw cosine ranks labels far below the character surface (macro MRR 0.301 against 0.398); a label-free whitening recovers most of the gap and a within-author whitening the rest, so the information is present and misaligned rather than absent.

## Retrieval estimand and PD-001

For each query song the system ranks the 226 labels with at least five qualifying songs after removing the song's whole leakage group from every profile it touches. The true corpus label supplies relevance; it is not an external human semantic judgement.

The character TF–IDF vocabulary and inverse document frequencies are fitted once on the fixed unlabelled query corpus. No source label, rank, relevance judgement, evaluation outcome, or tuned fusion weight enters that fit. Because query text influences the unlabelled vocabulary and IDF, the estimand is **transductive fixed-corpus leave-group-out retrieval**, not prospective performance (amendment PD-001). The whitening transforms of step 6 are fitted fold-wise with labels, and the held-out-author test bounds their dependence on the corpus.

Two averages of the same ranks are named. The macro estimand averages within labels first; the plain estimand averages over queries. `results/retrieval-v2/identity_estimands.json` gives every system under both.

The companion interface's repertoire map is separate and descriptive. It uses BGE-M3 centroids from the earlier snapshot and is not cited as evidence.

## NER and network estimand

Candidate extraction is not treated as validated NER accuracy. The two candidate methods are fallible and the 800-item package has zero completed human gold decisions. Public outputs therefore report:

- candidate counts and cross-method agreement;
- a 22-surface shared-text-excluded provisional inventory;
- six source-label-to-place enrichment edges with Jeffreys-based bounds and BH-FDR control;
- four same-song reference co-mentions with an all-song denominator, positive NPMI, and BH-FDR control.

A source-label-to-reference edge means that eligible lyrics assigned to one corpus label invoke that reference more often than the rest of the comparison universe. A co-mention edge means that two reference surfaces recur in the same eligible songs more than expected. Neither is a biography, residence, preference, influence, collaboration, or social relation.

## Written-ending estimand

The target is the next **dictionary-estimated written-ending family** among 17 pinyin-final classes. Lines must end in a Han character, and transitions must join originally adjacent lines in the same chunk. Partitions are fixed by leakage group before line events are built. Test targets that exactly or nearly match training or validation lines are excluded.

The hierarchical model separates continuation from switching and ranks alternatives. It does not generate lyrics. Performance is strong for continuing the same family and weak for switches. Source-credit-label conditioning has no supported predictive benefit, and the rhyme-form space of step 4 carries only a small share of label identity; label-level ending summaries are descriptive, not evidence of intrinsic rapper preference.

## Public/private boundary

Public: aggregate metrics, uncertainty intervals, typed short entity surfaces, support summaries, ending-family classes, lookup tables, figure-source tables (including label-tagged layout coordinates without song identifiers), figures, code, contracts, and validation manifests.

Private: full lyrics, full written lines, song/chunk identifiers, per-query ranks, tables tying coordinates to songs, row-level lyric-content hashes, embeddings, membership rows, and reviewer contexts. Public manifests may retain file-level SHA-256 checksums, and aggregate tables may retain deterministic join keys; neither is lyric text.

## Required private inputs for full reruns

- the corpus v2 chunk table (`work/private-repaired-corpus-v2/`) and its content digest;
- the recorded BGE-M3 embedding matrix, row map, and contract (`work/private-repaired-corpus-v2-embeddings/`);
- the screened entity lexicon, the reference core, and the three lexicon arms used by the neutralisation experiment;
- the reference ledger and fixed retrieval-label registry used by the NER build;
- any future completed NER reviewer files.

The code deliberately fails closed when required private inputs or recorded hashes do not match.
