# Duplicate-aware corpus repair method

Corpus v2 replaces one stage of the historical cleaner and nothing else. Title exclusions, configured line cleaning, and empty-chunk removal are replayed unchanged from `src/build_corpus_reconciliation_v1.py`, and the builder refuses to run unless those stages still reconstruct the frozen snapshot exactly. Only the fourth stage — keep the first exact `(source-credit label, cleaned chunk text)` occurrence — is withdrawn.

Every non-empty cleaned chunk is retained together with `song_id`, `chunk_id`, a global `source_order`, and a per-song `within_song_order`. Repetition inside a song is preserved, so hooks and refrains remain observable in sequence, and no song record disappears.

Duplicate control is represented rather than enacted. A song-record duplicate group requires the same source-credit label, the same normalized title, and the exact complete cleaned chunk sequence; the earliest record in source order is the representative and the others are flagged, not deleted. This rule is deliberately conservative and establishes only that two ingestion records carry identical cleaned content under one label and title. It does not establish real-world work identity, reissue status, authorship, or performer identity.

Cross-song text reuse is carried separately. Songs that share any exact cleaned chunk text are unioned into a text component; repetition inside one song is not a link. Components are the leakage unit: a component must stay together across train, validation, and test partitions, or be removed from every candidate profile. Each component carries total weight one inside a source-label aggregate, so repeated imports cannot dominate a label centroid, vocabulary probe, or cultural-reference rate.

The DEL control characters adjudicated in PD-002 are removed while their preceding characters are kept, which is the source semantics; the live sheet's destructive import behaviour is not copied.

Song records the legacy rule erased that no automatic rule resolves stay in a review queue for author adjudication. The queue is reproduced at the size PD-002 published. Where the v2 primary rule happens to group a queued record — which occurs when that record's exact-sequence twin was itself erased and so is retained only in v2 — the record remains in the queue and is reported as its own stratum, because a published safeguard is not narrowed by a later rebuild.

Two output trees are written. The private tree carries the repaired corpus itself, including lyric text, titles, source-credit labels, and identifiers, and is never published. The public artifact carries aggregate counts, input and software hashes, this method text, and validation results, and carries no text, titles, labels, identifiers, embeddings, or row-level hashes.
