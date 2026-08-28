# Chinese written-rhyme V1

This release turns strict Han-ending Chinese-rap written lyric lines into interpretable ending-family fingerprints and a song-held-out exact-next-adjacent-line recommender. Within the frozen chunk-deduplicated snapshot, repeated lines remain in surviving sequences; excluded lines and chunk boundaries never create bridged transitions. The post-freeze reconciliation in `../corpus-reconciliation-v1/` documents the legacy upstream chunk removal and the duplicate-aware repair action.

Start with:

- `analysis_summary.json` for the result and all claim boundaries;
- `model_metrics.csv` and `paired_model_deltas.csv` for baseline-versus-context-model evaluation with song-bootstrap intervals;
- `label_written_rhyme_summary.csv` / `label_written_rhyme_fingerprints.json` for public-safe source-credit-label summaries;
- `recommender_lookup.json` for text-free aggregate next-class recommendations;
- `METHOD.md` for the complete protocol;
- `muchin_auxiliary_agreement.json` for the explicitly partially circular external sanity check.

The release does **not** contain lyrics, full written lines, song/chunk IDs, or content hashes. It does not claim performed rhyme, flow, cadence, audio style, or verified artist identity.
