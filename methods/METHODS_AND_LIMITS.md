# Pipeline, methods, and claim boundaries

This note is the compact execution map. The manuscript gives the full scholarly argument and the task-specific `METHOD.md` files give exact estimands and settings.

## Pipeline

1. **Freeze and audit the corpus.** Reconcile one canonical song row per song and one clean-text sidecar row per chunk. Withhold metadata-only chunks. Record eligibility, exceptions, and content-group structure without silently guessing missing metadata. The post-freeze PD-002 audit row-aligns the live Drive sheet with the raw CSV (26,833 exact song/chunk keys; 33/33 substantive cell differences adjudicated), reconstructs the earlier raw-to-snapshot cleaner, distinguishes 131 high-confidence duplicate records from 46 review-required records, and forbids artist-level chunk deletion as the future duplicate-control mechanism.
2. **Create frozen representations.** Generate normalized BGE-M3 dense vectors from eligible clean chunks and record the checkpoint revision, row map, dimensions, and hashes.
3. **Build downstream Task 1: repertoire retrieval.** Hold out the complete query song and its duplicate component from every affected label profile. Compare BGE-M3, character TF–IDF, and untuned standardized fusion. Estimate label-balanced metrics and paired two-stage bootstrap intervals.
4. **Build the Task 1 descriptive repertoire companion.** Form duplicate-component-weighted BGE-M3 centroids for 204 supported source-credit labels in a primary and a shared-text-excluded representation. Retain only reciprocal top-five pairs that appear in both treatments; quantify observed-edge repeatability with 250 within-label song resamples. Use deterministic two-dimensional PCA only for the global overview and state that its 26.2% retained variation makes position approximate. Apply vocabulary, written-ending, and writing-form probes after edge selection as auxiliary explanations, never as causal edge definitions.
5. **Build downstream Task 2: cultural-reference evidence.** Combine a screened Chinese-rap lexicon with a pinned contextual Chinese NER baseline. Exclude cross-label shared text, aggregate at duplicate-controlled song units, and release only typed associations that pass support, uncertainty, and BH-FDR gates. Preserve the 800-occurrence dual-review path to future human gold.
6. **Build downstream Task 3: written-ending continuation.** Convert strict terminal-Han written lines to 17 tone-free pinyin-final families, split songs before constructing adjacent-line events, exclude exact and near-duplicate target leakage, and compare global, Markov, flat-context, and hierarchical continuation/switch models. Select hyperparameters and abstention thresholds on validation only. Repeat retention is conditional on the frozen chunk-deduplicated snapshot; the PD-002 counterfactual quantifies restored line/transition counts and withholds predictive robustness until a duplicate-aware retraining is complete.
7. **Render evidence.** Generate publication figures, aggregate tables, the single-file results interface, and journal-oriented DOCX/PDF files. Run numerical, privacy, accessibility, and visual-render checks.

## Why BGE-M3

The corpus is primarily Chinese but includes mixed-script passages and variable-length chunks. BGE-M3 supplies a multilingual dense retrieval representation and is frozen rather than fine-tuned. Its relevance is tested empirically: BGE-M3 alone is weaker than character TF–IDF, while their untuned standardized fusion is better than either component. The contribution is therefore the post-representation design, not the mere existence of embeddings.

## Retrieval estimand and PD-001

For each eligible song, the system ranks the 204 fixed source-credit labels after removing the query song and all detected duplicate variants from candidate profiles. The true corpus label supplies relevance; it is not an external human semantic judgment.

The character TF–IDF vocabulary and inverse document frequencies were fit once on the fixed unlabeled retrieval corpus. No source label, rank, relevance judgment, evaluation outcome, or tuned fusion weight enters that fit. Because query text influences the unlabeled vocabulary and IDF, the reported estimand is **transductive fixed-corpus leave-one-song-out retrieval**, not prospective performance on later songs. `PROTOCOL_AMENDMENT_PD001_TRANSDUCTIVE_TFIDF.md` records the departure from the original generic train-only rule.

The global and local repertoire maps are a separate descriptive companion. They use BGE-M3 centroids only—not retrieval fusion—and do not contain held-out queries. A released line means that both labels rank one another among their five closest profiles under both duplicate-controlled text treatments. Of 86 retained lines, 16 reappear in at least half of 250 repeated within-label song samples. That threshold is a display gate, not confidence or posterior probability. Detailed aggregate lineage, method, and validation are frozen under `results/repertoire-network-v1/`.

## NER and network estimand

Candidate extraction is not treated as validated NER accuracy. The two candidate methods are fallible and the current 800-item package has zero completed human gold decisions. Public outputs therefore report:

- candidate counts and cross-method agreement;
- a 22-surface shared-text-excluded provisional inventory;
- six source-label-to-place enrichment edges with Jeffreys-based bounds and BH-FDR control;
- four same-song reference co-mentions with an all-song denominator, positive NPMI, and BH-FDR control.

A source-label-to-reference edge means that eligible lyrics assigned to one corpus label invoke that reference more often than the rest of the comparison universe. A co-mention edge means that two reference surfaces recur in the same eligible songs more than expected. Neither is a biography, residence, preference, influence, collaboration, or social relation.

## Written-ending estimand

The target is the next **dictionary-estimated written-ending family** among 17 pinyin-final classes. Lines must end in a Han character, and transitions must join originally adjacent lines in the same chunk. All song partitions are fixed before line events are built. Test targets that exactly or nearly match training/validation lines are excluded.

The hierarchical model separates continuation from switching and ranks alternatives. It does not generate lyrics. Performance is strong for continuing the same family and weak for exact switches; the interface therefore shows several possible next families and an abstention/reliability status. Source-credit-label conditioning has no supported predictive benefit, so label-level ending summaries are descriptive rather than evidence of intrinsic rapper preference.

## Public/private boundary

Public: aggregate metrics, uncertainty intervals, typed short entity surfaces, support summaries, ending-family classes, lookup tables, figures, code, contracts, and validation manifests.

Private: full lyrics, full written lines, song/chunk identifiers, row-level lyric-content hashes, embeddings, membership rows, and reviewer contexts. Public manifests may retain file-level SHA-256 checksums, and detailed aggregate tables may retain deterministic join keys; neither is lyric text.

## Required private inputs for full reruns

- frozen canonical song and chunk exports;
- cleaned-text sidecar and its reconciliation manifest;
- frozen BGE-M3 embedding matrix and row map;
- screened entity lexicon, reference ledger, and fixed retrieval-label registry used by the NER build;
- any future completed NER reviewer files.

The code deliberately fails closed when required private inputs or recorded hashes do not match.
