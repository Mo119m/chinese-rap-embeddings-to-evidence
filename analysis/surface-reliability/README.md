# Surface reliability against released claims

Derived from the public NER tables by `tools/summarise_surface_reliability.py`.
No private input and no annotation is involved; every number here is a join of
`entity_aggregate_provisional.csv`, `source_label_entity_links_provisional.csv`,
and `entity_co_mentions_provisional.csv`.

## What the ratio means

A lexicon candidate is retained only when the transformer baseline proposes the
*same span* with the *same schema type*. `boundary_ok()` in the builder returns
True unconditionally for surfaces containing no ASCII characters, so the lexicon
stage alone cannot separate a surface from a longer compound that contains it;
the exact-span requirement is what rejects those. The retention rate is therefore
an inter-method reliability statistic, not an accuracy estimate: it says how often
two independent methods agreed, not how often either was right.

Across 22 released surfaces the rate ranges 0.55-1.00 (median 0.74).

## Surfaces that carry released claims

| surface | type | agreement | candidates rejected | claims | song units |
| --- | --- | ---: | ---: | ---: | ---: |
| 伦敦 | PLACE | 0.55 | 10 | 1 | 5 |
| 湖南 | PLACE | 0.60 | 10 | 1 | 5 |
| 中文 | LANGUAGE_OR_DIALECT_REFERENCE | 0.65 | 94 | 1 | 6 |
| 台北 | PLACE | 0.68 | 11 | 1 | 6 |
| 天津 | PLACE | 0.72 | 17 | 1 | 10 |
| 上海 | PLACE | 0.82 | 32 | 4 | 27 |
| 杭州 | PLACE | 0.83 | 11 | 1 | 10 |
| 英文 | LANGUAGE_OR_DIALECT_REFERENCE | 0.86 | 6 | 1 | 6 |
| 巴黎 | PLACE | 0.87 | 17 | 2 | 12 |
| 新疆 | PLACE | 0.89 | 6 | 1 | 5 |

## Reading

The two lowest-agreement load-bearing surfaces are **伦敦** (0.55), **湖南** (0.60), and both rest on small supports. It is tempting to read that as a ranking of
how secure the released claims are. It is not, and treating it as one would be a
mistake this file exists to prevent.

The rate is computed corpus-wide, but a released label link depends only on the
occurrences inside one source label, and a co-mention only on the songs where both
surfaces appear. Those populations differ. A surface can look weak overall because
of compounds that never occur under the label carrying the claim, or because a
compound that does occur there is rare. 湖南 has the second-lowest retention here;
the compounds that lower it are 湖南卫视, a food term, and a university choir's
institutional name generalised under NER-CR-001.

This file cannot say how those resolve inside a claim, and it previously did. It
asserted that enumerating 湖南 under 泰格西 finds every occurrence a bare locative
use. That was wrong twice over. `tools/audit_released_claim_occurrences.py` states
that `bare surface` means "not one of the compounds listed", never "verified
locative" -- so no tool here establishes a locative reading. And the enumeration
does not return all-bare: one accepted 湖南卫视 sits inside the 泰格西--湖南 link,
recorded in paper/PENDING_MANUSCRIPT_INSERTS.md and HANDOFF.md, and corroborated by
analysis/compound-resolution/resolution_table.csv, whose 湖南/湖南卫视 row carries
flagged_strict_occurrences 1. That link's support is 5 song units against a release
gate of at least 5, so it sits on its floor with a compound occurrence inside it.
This file also cannot check that: it joins public tables only, while the enumeration
needs the private candidate table.

So use this table to decide what to look at, and `tools/audit_released_claim_
occurrences.py` to decide what it means. That script resolves each released claim
to its own occurrences, applies the same shared-text exclusion, and classifies
every one; it needs the private corpus but no annotation.

What the rate does bound is the extraction as a whole, and that bound is real:
BH-FDR and the conservative intervals control sampling error given the extraction,
never error in the extraction. The rate also cannot distinguish a genuinely
ambiguous surface from a domain mismatch in the transformer baseline, which is
`ckiplab/albert-tiny-chinese-ner`: a very small model trained on traditional-Chinese
material and applied here to simplified Chinese. Reporting agreement across three
or more simplified-Chinese taggers separates those two explanations without any
annotation; `tools/multi_tagger_agreement.py` does that.
