# Repaired corpus v2

Corpus v2 implements the PD-002 replacement rule. All **25,026 cleaned chunk rows** across **7,391 song records** are retained with their original `(song ID, chunk ID, source order)` relation, restoring the **2,894 chunks** and **177 song records** the legacy keep-first rule deleted. No chunk is removed for being a duplicate.

Duplicate structure is represented instead. The primary automatic stratum — same source-credit label, same normalized title, exact complete cleaned chunk sequence — forms **84 groups** over **227 song records**, of which **143** are flagged as non-representative rather than deleted. Songs sharing any exact cleaned chunk text form **6,025 components**; **799** span more than one song and cover **2,165 song records**, the largest reaching **27**. Each component carries total weight one inside a source-label aggregate.

The PD-002 review queue is reproduced and not narrowed: of the **177** song records the legacy rule erased, **131** meet the high-confidence duplicate-record rule and **46** remain queued for author adjudication. **12** of those queued records are additionally grouped by the v2 primary rule, because their exact-sequence twin was itself erased; they stay in the queue and are reported as their own stratum.

Status: **pass_pending_author_review**. Structure, preservation, and duplicate representation pass their gates. The duplicate groups do not establish work identity, reissue status, authorship, or performer identity, and no downstream predictive metric has been rerun on this population.
