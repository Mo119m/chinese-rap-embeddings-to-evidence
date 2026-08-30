# Agreement across independent taggers

Produced by `tools/multi_tagger_agreement.py`. Two simplified-Chinese taggers were
run over the release's own lexicon candidates for the twenty-two released surfaces,
independently of the release's baseline:

- `shibing624/bert4ner-base-chinese` (MSRA-style: PER, LOC, ORG)
- `uer/roberta-base-finetuned-cluener2020-chinese` (CLUENER: address, name, scene, …)

Agreement is scored on *exact span and schema type*, the same criterion the release
applies. This is inter-method reliability, not inter-annotator reliability: it
records how often independent methods agreed, never how often any of them was right.

## Population

Candidates come from the pipeline's own occurrence table, filtered to
`LEXICON_WITH_TRANSFORMER_CHECK` with `cross_label_shared_cleaned_text == False`, and
each is scored inside its own stored context window at its own recorded span. An
earlier version scanned the cleaned sidecar directly and scored a superset — 334
candidates for 上海 where the release records 174, 436 for 中文 against 267 — which
made its numbers incomparable to the release's own retention rate.

That filter reproduces the published `lexicon_candidate_occurrences` **exactly for 13
of the 22 surfaces**, and lands within a few candidates on most of the rest (191
against 174 for 上海, 273 against 267 for 中文). The remaining aggregate filter has not
been identified. Rather than pool the two groups, the headline is computed over the
thirteen that reproduce and the others are marked in the table.

## Result

Over the **13 surfaces whose population reproduces exactly** and that have a defined
alpha: **9 are negative**, and **2 exceed 0.25** — 台北 at 0.62 and 昆明 at 0.30.
太平洋 reaches −0.84 across eighteen candidates. 潘玮柏 sits at 0.00 with both taggers
confirming 24 of its 25 candidates, so the alpha is uninformative rather than bad.

The frame matters: 上海 scores 0.47 here against 0.26 on the earlier superset, so the
corrected population is more favourable to the release, not less.

## Reading

The hypothesis this was built to test was that the release's baseline,
`ckiplab/albert-tiny-chinese-ner`, might be depressing retention rates because it is
small and trained on traditional-Chinese material. The result does not support that
reading, and does not support the opposite one either. Two further taggers, both
trained on simplified Chinese and both larger, disagree with *each other* at or below
chance on most place surfaces, on the release's own candidates.

The defensible statement is that **these two models, on these candidates, agree
poorly.** Going further and calling the disagreement inherent to the task in this
domain would need more than two models.

Two consequences hold. Adding taggers is not a route to a precision estimate here —
it produces a reliability statistic, and that statistic is poor. And the release's
decision to withhold precision, recall, and F1 pending human review is supported
rather than undermined.

## What the taggers cannot vote on

Neither label inventory has a language-reference category, so both abstain on 中文 and
英文. Those rows read `not comparable`. Counting the silence as rejection would have
produced a spurious zero-agreement result for the two surfaces behind the 中文–英文
co-mention, for a reason having nothing to do with this corpus. Abstentions are
excluded from alpha.

## Caveat in the other direction

Exact-span matching is strict: where one model tags 天津卫 as a unit and the other tags
天津, they count as disagreeing even though both located the same reference. A
span-overlap criterion would very likely score higher. How much higher is not
established here, so these values should not be called a bound.
