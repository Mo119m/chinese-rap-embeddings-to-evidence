# Protocol Amendment PD-003: metadata lines removed from the lyric text (corpus v3)

## Status and timing

PD-003 is a post-freeze amendment. It was opened on 9 September 2026 and closed with corpus v3
build 1.3.0 on 10 September 2026. This document was written on 22 September 2026, after the
fact, from the builder (`src/build_cleaned_corpus_v3.py`), its public audit
(`results/cleaned-corpus-v3/analysis_summary.json`) and the git history. It records what was
decided when, including the decisions made after results had been seen.

## Why

An audit of corpus v2 on 9 September 2026 found that the lyric text still carried what the
source pages carried around the lyrics: production credits in 5.8% of chunks, section markers
in 3.0%, HTML remnants in 0.5%, a few URLs, and lines of nothing but digits or dashes. None of
that is lyric, and every identity experiment had been run over it. Protocol MB-001 already held
a frozen, self-tested detector for credit blocks, written after the author's instruction that
such blocks be removed from the input rather than labelled; it had been sequenced behind
PD-002. PD-003 applies it, together with a small set of line rules.

## Rule

The builder removes a line, and counts it by rule, when it is:

- a credit block labelled by the MB-001.1 detector, classified over the whole song in source
  order so that a block straddling two chunks is seen whole;
- a section header, an English-form role line or a `作词：` shape, a line carrying a bracketed
  production tag, a numbered album-page entry, a URL, a line of digits and punctuation only, or
  a whole-line bracket marker;
- one of the credit shapes added in build 1.3.0 (handle, production, role-by, feat,
  publisher or domain, pipe-separated credit) on a short line without sentence punctuation;
- the song's own title, but only next to a credit or header line (inside a copied page header).

A song that is a whole television episode is dropped whole. HTML tags are stripped and entities
unescaped; whitespace is normalised. Chunks left empty are dropped. Every other column of
corpus v2 is carried unchanged, so leakage groups, components and weights still resolve, and a
content digest over the v3 rows is recorded so that every downstream builder refuses any other
table.

Kept on purpose: bracketed short spans such as ad-libs and backing vocals, a song's mention of
its own artist, and lines in scripts the corpus does not model (Uyghur, Tibetan, Korean,
Mongolian). Each of these carries identity and has its own control arm in the experiments
instead of being removed.

## Builds, and what was seen before each

| build | date | change | seen before the change |
|---|---|---|---|
| 1.0.0 | 2026-09-09 | first application of the rule | the corpus v2 audit only |
| 1.1.0 | 2026-09-09 | title-equal lines kept when lyric lines surround them; fuzzy detector rules honoured only inside anchored blocks; inline angle-bracket spans left alone | a sample of deleted lines: about 2,800 of 3,023 deleted title-equal lines were hook lines |
| 1.2.0 | 2026-09-09 | track-list lines; a television episode filed as a song | the author's reading of a line sample |
| 1.3.0 | 2026-09-10 | six credit shapes on short lines | a whitening-directions analysis on build 1.2.0 in which the credit residue formed a direction of the semantic space |

Builds 1.0.0 and 1.1.0 deleted lyrics and their numbers were never published. Build 1.2.0 was
published for one day. The 1.3.0 rule was written after an identity result on 1.2.0 had been
seen, which is disclosed here and in the results README; its effect on every published number
is 0.1 to 0.5 points in the same direction, with one contrast changing significance.

## Effect (build 1.3.0, from the public audit)

| quantity | before | after |
|---|---|---|
| songs | 7,391 | 7,379 |
| chunks | 25,026 | 24,237 |
| lines | 549,646 | 544,748 |

4,898 lines (0.89%) were removed, touching 3,876 chunks (15.5%) and 2,788 songs (37.7%); 733
chunks were left empty and dropped, and 2 songs were dropped as television transcripts. Lines
kept on purpose that a rule would otherwise have caught: 2,977 title-equal hook lines, 837
fuzzy detector hits judged lyric, 2 bracketed lines with sentence punctuation. Counts by rule
are in the audit file.

## Validation, and what is missing

The MB-001 detector has a single-rater gold set on the corpus v2 frame
(`results/metadata-block-gold-v1/`); it scores the detector's block rule only, not the v3 line
rules. No removed-line validation of the v3 builder as a whole exists. A stratified sample of
removed and kept lines, coded independently by two people, is the planned validation; until it
exists the cleaning is described by its rules and counts, not by an accuracy.
