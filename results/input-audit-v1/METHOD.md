# Downstream input audit method

The audit freezes the exact song, chunk, cleaned-text, and metadata sidecars used by the three downstream tasks. It verifies grain, key uniqueness, join coverage, non-empty eligible text, hash validity, label/title completeness, metadata-flag coverage, and snapshot counts. Exact-content groups are reported so every task can keep duplicate-linked songs and texts in one split.

Passing this audit means the files satisfy the declared structural contract. It does not mean every source-credit label, title, lyric line, entity, or pronunciation has been manually or externally verified. The task builders must preserve the claim boundaries in `analysis_summary.json` and publish aggregate-only outputs.
