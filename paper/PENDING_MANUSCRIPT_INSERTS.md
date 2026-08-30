# Pending manuscript inserts

**Not yet in `manuscript.md`, and not all of them are ready.** An independent review
found substantive problems in three of the four supporting analyses. All five are verified and ready
to paste. Each was rerun after an independent review found the first drafts scoped or
counted wrongly; the corrections are listed at the end.

The manuscript's DOCX and PDF derivatives are built by
`src/build_chinese_rap_paper_docx_v1.py`, and the PDF step is not part of the tracked
pipeline, so editing `manuscript.md` alone would leave four binary derivatives
disagreeing with their source while every checksum still validates.
`tools/check_manuscript_derivatives.py` catches that; rerun it with `--record` after
rebuilding, and delete this file once everything here has landed.

---

## Insert 5 — Results, repertoire-network subsection — READY

Verified. The graph rebuilds to exactly the published 86 edges and 93 labels before
any permutation, and the numbers below come from that same code path.

> The reciprocal top-five rule is permissive by construction, and the released edge
> count should be read against what it yields on an uninformative label assignment. We
> permuted the label column of the label-song incidence, preserving every label's song
> count and every song's label count, rebuilt the centroids and reapplied the rule
> under both representations, over 200 replicates. The null yields a mean of 45.0
> edges (sd 6.7, maximum 68) connecting a mean of 52.2 labels. No replicate reached
> the observed count, giving a one-sided permutation p of 0.005; the observed graph is
> 1.91 times the null mean and about 6.1 standard deviations above it.
>
> The excess is therefore large, and the null mean is also worth stating: reciprocity
> is common when 204 nodes each select five of 203, and a corpus in which every
> document is Chinese rap lyrics produces high centroid cosines even under random
> grouping, so about 45 edges arise before any label information is used. The test
> does not identify which observed edges are spurious and is not a false-discovery
> rate. The permutation holds the shared-text mask and the eligibility set fixed at
> their values under the true assignment, so it is a null conditional on those.

## Insert 2 — Methods, NER subsection, after the baseline description — READY

Needs no data beyond reading the builder.

> The exact-span requirement carries more weight than the schema alone suggests. The
> boundary test applied to lexicon matches passes unconditionally for surfaces
> containing no ASCII characters, which is every surface in this inventory, so the
> lexicon stage cannot by itself distinguish a surface from a longer compound that
> contains it. Compound absorption is instead rejected downstream, when the
> transformer proposes a longer span than the lexicon match and the two fail to
> agree. Retention rates are consistent with that mechanism: the largest proportional
> rejections fall on surfaces with common compound extensions.

---

## Insert 1 — Results, cultural-reference subsection — READY

Rerun on the accepted set. All ten released claims reproduce their published support
exactly, so the classification below covers the occurrences that carry them.

> Every occurrence supporting a released cultural-reference claim was resolved from
> the accepted extraction set — lexicon candidates that agreed with the transformer on
> exact span and schema type, after shared-text exclusion — and classified by the
> characters immediately following the surface. The six label links rest on 102 such
> occurrences and the four co-mentions on their own. Two occurrences read as a
> compound rather than the entity: one 巴黎世家, the fashion house, inside the 伦敦–巴黎
> co-mention, and one 湖南卫视, a broadcaster, inside the 泰格西–湖南 link, which has the
> smallest support of the six. A further four occurrences under 黑麦–天津 are 天津卫, a
> historical name for the city itself, and one under 中文–英文 is the genre term
> 中文说唱 rather than a reference to the language.
>
> This is a compound check, not a semantic one. A surface not followed by a known
> compound is recorded as bare, which does not establish that it is used locatively;
> a metaphorical use of a bare surface would not be detected. Dual human review
> remains the only route to precision, recall, and F1.

## Insert 3 — Limitations, replacing the single-baseline caveat — READY

Rerun on the release's own candidate occurrences.

> The second baseline is `ckiplab/albert-tiny-chinese-ner`, a small model trained on
> traditional-Chinese material and applied here to a predominantly simplified-Chinese
> corpus, which raises the question of whether retention rates reflect genuine
> ambiguity or a domain mismatch. We scored the release's own lexicon candidates,
> each inside its recorded context window and at its recorded span, against two
> further taggers trained on simplified Chinese: `shibing624/bert4ner-base-chinese`
> and a CLUENER-finetuned RoBERTa. The candidate population reproduces the published
> counts exactly for thirteen of the twenty-two surfaces; statistics are reported over
> those thirteen. Krippendorff's alpha between the two comparison taggers, computed
> over methods rather than annotators and on the same exact-span criterion, is
> negative for nine of them and exceeds 0.25 for two, 台北 at 0.62 and 昆明 at 0.30.
>
> Independent taggers therefore disagree with one another about place references in
> this material at roughly chance, which is not specific to the baseline we used.
> Adding methods yields a reliability statistic rather than an accuracy estimate, and
> that statistic is poor. We read this as supporting the decision to withhold
> precision, recall, and F1 until dual human review is complete. Neither comparison
> tagger has a language-reference category, so 中文 and 英文 admit no cross-method
> comparison at all, and exact-span scoring counts a tagger that returns 天津卫 against
> a lexicon match on 天津 as disagreeing.

## Insert 4 — Results, new paragraph, or Discussion — READY

Rerun with a control that matches every surface and with three independent draws.

> Because the correctness of individual extractions cannot yet be established, we also
> measured the entity inventory in a way that does not depend on it. Masking every
> curated lexicon surface in the corpus and re-running the held-out retrieval
> evaluation gives an automatic ground truth — the source-credit label — and asks how
> much label-identifying signal those strings carry rather than whether each was
> correctly typed. Removing all 605 surfaces (52,106 occurrences) costs 0.0069 macro
> MRR on the character TF-IDF arm. A control that pairs every surface with a
> non-entity character n-gram of the same length and closest corpus frequency, drawn
> three times independently and matching the entity condition's total occurrences to
> within 0.1%, costs a mean of 0.0009 with individual draws at −0.0008, +0.0009 and
> +0.0020. The adjusted difference is 0.0061, and the entity condition's drop exceeds
> every control draw.
>
> Masking only the twenty-two released surfaces produces no measurable change, which
> is a statement about statistical power: those occurrences are about 0.05% of the
> corpus by character. Three draws give a range rather than a confidence interval, and
> the comparison is matched rather than decomposed, so the observation that 0.0061 is
> roughly a fifth of the published 0.031 fusion gain places the magnitude without
> apportioning it.

---

## Corrections to numbers that appeared in earlier drafts

| claimed | correct |
| --- | --- |
| median retention 0.82 | **0.74** (the tool also computed 0.75; both were wrong) |
| alpha negative for 12 of 20, on a superset frame | **9 of 13** on the release's own candidates |
| alpha above 0.25 for three | **two** (台北 0.62, 昆明 0.30) on the corrected frame |
| control replaced ~51,000 occurrences | the band-matched control replaced 46,619 and skipped 46 surfaces; the rerun matches 605 of 605 to within 0.1% |
| observed graph 87 edges / 94 labels | **86 / 93** — the earlier run used the wrong centroid weight |
| observed nearly 3 sd above the null | about **6.1 sd** |

## Two claims to consider weakening at the same time

1. **String-level framing.** "Six source labels are associated with six places"
   requires the extraction to be right. "Six source labels over-use six curated
   place-name strings, after shared-text exclusion and relative to a fixed comparison
   universe" is true by construction and needs no precision estimate. With the
   occurrence audit blocked, this framing is currently the only one fully supported.

2. **Figure 1's alt text** reads "not identity or authorship" while the figure's own
   question line uses *identity*. The manuscript resolves this by defining the term
   and narrowing it to *lyrical repertoire*, but the figure carries no such
   scaffolding. Either add one clause to the caption or re-render the figure.
