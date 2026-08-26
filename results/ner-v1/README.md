# Chinese Rap NER + Grounded Cultural Network v1.1

This release contains two reproducible NER candidate baselines, a private source-occurrence-complete 800-task dual-review package, a conservative entity inventory, and a shared-text-excluded cultural graph with song-unit uncertainty and BH-FDR control.

Start with `summary.json`, `release_sensitivity_summary.csv`, and `source_label_entity_links_provisional.csv`; then read `METHOD.md` and `SCHEMA_AND_ANNOTATION.md`. Public CSVs contain aggregate evidence only. No lyrics, contexts, song/chunk IDs, cleaned-text hashes, or embeddings are published.

**Evidence boundary:** there is no completed human occurrence gold. Occurrence counts are repeated corpus spans, not independent samples. Shared exact cleaned text is excluded from label associations and co-mentions. Every released result remains provisional; co-mention is a text pattern, never collaboration, influence, identity, or a social relationship.
