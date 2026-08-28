# Corpus reconciliation v1

The live Drive sheet and the local raw export contain the same **26,833 row-aligned records** with exact song and chunk keys. All **33 substantive cell differences are adjudicated**: **29** title type coercions, **2** leading-apostrophe escapes, and **2** control-character import side effects. No substantive mismatch remains unresolved.

The historical cleaner is exactly reproducible, but its artist-level exact-text deduplication removed **2,894 chunks** and erased **177 songs** before downstream modeling. Of those lost songs, **131** satisfy a conservative high-confidence duplicate-record rule; **46** require review rather than automatic deletion.

At full cleaned-source scope, the global written-ending distribution changes little, but the frozen input omits **16,623 strict-Han-ending line occurrences** and **13,112 adjacent transitions** present before chunk deduplication. In the exact released 204-label task frame, restoring eligible chunks adds **7,033 lines**, **4,938 transitions**, and **6,472 additional repeat occurrences beyond first occurrences**; family-distribution total variation is **0.001789**. Therefore the released repeat-retention wording is true only inside surviving chunks. Aggregate sensitivity is complete, while predictive metrics remain untested on the repaired population.

Status: **pass_with_release_action**. Structural reconstruction and aggregate written-ending sensitivity pass; duplicate-aware corpus repair and predictive reruns remain release actions.
