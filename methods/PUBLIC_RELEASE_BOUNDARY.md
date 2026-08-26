# Public release boundary

This repository separates research integrity metadata from private lyric-level records.

## Excluded

- full lyrics and full written lines;
- song IDs, chunk IDs, line IDs, and private membership rows;
- row-level hashes or fingerprints of lyric content;
- dense embeddings and row maps;
- reviewer contexts, ratings, and adjudication records;
- credentials, personal email addresses, and private storage paths.

## Retained

- aggregate metrics, confidence intervals, support counts, short entity surfaces, and short characteristic tokens;
- file-level SHA-256 checksums for reproducibility and integrity verification;
- `ALBL-*` and `ENT-*` deterministic aggregate join keys in detailed result tables.

The `ALBL-*` and `ENT-*` values are not song, chunk, line, or lyric-content hashes. They join aggregate rows whose source-credit label or entity surface is already published in the same result layer. They do not expose lyric text or a hidden natural-person identity. The reader-facing `index.html` replaces them with short sequential interface IDs and does not expose the hashed join keys.

Source-credit labels remain corpus provenance rather than verified artist identities. Cultural-reference edges are lyric-reference associations rather than biography or social relations. Written-ending results are dictionary-pinyin estimates rather than performed rhyme or flow.
