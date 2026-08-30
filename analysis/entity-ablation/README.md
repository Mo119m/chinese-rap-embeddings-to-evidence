# Entity ablation in held-out retrieval

Produced by `tools/entity_ablation_retrieval.py` against the private corpus. Ground
truth is the source-credit label, so the measurement is automatic: no annotation is
involved, and an individual extraction error changes the answer only in proportion to
the signal it carries.

The script reuses the release's own retrieval harness rather than reimplementing it,
and refuses to report a delta unless it first reproduces the published TF-IDF macro
MRR. Every run reproduced it to `0.415771` against a published `0.415771`, a drift of
3.9e-07.

Only the character TF-IDF arm is reported. The dense embeddings are precomputed over
the original text, so dense and fusion numbers are withheld rather than computed from
vectors that describe the unmasked corpus.

## Control

Each masked surface is paired with a non-entity character n-gram of the same length
whose corpus frequency is closest, so the "deleting characters lowers any similarity"
effect can be subtracted. An earlier version required the frequency to fall inside a
band and skipped a surface when nothing qualified, which left 46 of 605 surfaces
uncontrolled and the control removing about 10.5% fewer characters than the entity
condition — exactly the confound the control exists to remove.

Closest-frequency matching now pairs **every** surface: 605 of 605, with the control's
total corpus occurrences within ±0.1% of the entity condition's. The control is drawn
three times with independent seeds and the spread across draws is reported, because a
single draw gives a point estimate with no sense of its own variability.

## Results, full curated lexicon

| condition | macro MRR | drop |
| --- | ---: | ---: |
| baseline | 0.4158 | – |
| all 605 surfaces masked (52,106 occurrences) | 0.4088 | **+0.0069** |
| frequency-matched control, mean of 3 draws | 0.4149 | +0.0009 |

Control drop across the three draws: **−0.0008, +0.0009, +0.0020**.

**Adjusted difference: +0.0061 macro MRR.**

The entity condition's drop of 0.0069 exceeds every control draw, the largest of which
is 0.0020. That is the substantive point: the effect is not reproduced by removing
comparable amounts of arbitrary text.

## Released inventory alone

Masking only the twenty-two released surfaces changes nothing measurable. That is a
statement about statistical power rather than about cultural reference: 1,088
occurrences of two- and three-character surfaces are roughly 0.05% of the corpus by
character, which cannot move a macro MRR. It is reported so a reader does not have to
infer it.

## What this does and does not license

Three draws give a range, not a confidence interval, and the design is a matched
comparison rather than a decomposition. The published dense-lexical fusion gain over
character TF-IDF is 0.031; noting that 0.0061 is about a fifth of it places the
magnitude and nothing more. It does not mean entity strings supply a fifth of what
fusion supplies.

More draws would narrow the range. Matching is on surface length and corpus frequency,
not on position or collocational context, so a residual difference between entity
strings and their controls remains possible.

## Boundary

This measures how much label-identifying signal the masked strings carry in a
fixed-corpus held-out retrieval task. It is not an accuracy estimate for the entity
extraction, it is not evidence about what any reference means, and it says nothing
about artists.
