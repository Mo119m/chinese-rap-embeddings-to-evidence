# Chinese Rap Lyrical Repertoire Graph v2

## The single question

Which **artist-labelled corpus slices** have similar duplicate-controlled lyric
repertoires, and which neighbours remain after removing exact cleaned text
shared across source labels?

An edge is deliberately narrow: the two labels are mutual top-five semantic
neighbours in both the primary duplicate-weighted chunk representation and the
shared-text-exclusion sensitivity representation. It is a relationship between
two corpus slices, not a real-world relationship between rappers.

The current artist strings are source labels. They remain labelled as not
externally identity-verified until an evidence-backed identity registry exists.

Only songs whose canonical `artist_title_comparison_eligible` field is `true`
enter this source-label comparison. This is a conservative metadata-quality
filter, not a correction or renaming of source labels.
