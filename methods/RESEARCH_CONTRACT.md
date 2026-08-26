# Chinese Rap Downstream Study V1 — Research Contract

## Unifying question

How do Chinese rap lyrics form recognizable lyrical identities through language, cultural reference, and dictionary-estimated written rhyme?

The study answers that question with three downstream tasks:

1. **Explainable lyrical-repertoire retrieval:** Which source-credit-labelled repertoires are nearest to a held-out song or another repertoire, what independently measured features do they share, and how repeatable is the match?
2. **Chinese-rap entity recognition and grounded cultural networks:** Which people/groups, crews/labels, places, brands/products, and works/events populate each lyrical repertoire, and which lyric co-mentions or externally sourced relations can be stated without conflating the two?
3. **Written-rhyme modelling and recommendation:** Which dictionary-estimated written endings recur within each repertoire, how do ending classes change across lines, and which ending class is plausible next in held-out songs?

## Frozen data contract

- Canonical song grain: one row per `song_id`.
- Canonical chunk grain: one row per `(song_id, chunk_id)`.
- Modelling text: `analysis_text` only where `analysis_text_status=eligible_clean_text`.
- Labels: immutable corpus source-credit labels; not independently verified performer identities.
- Titles: title-semantic and artist/title uses require the relevant metadata eligibility flags.
- Input snapshot and structural integrity are governed by `outputs/chinese-rap-downstream-input-audit-v1`.

## Leakage controls shared by every task

- Split by song before expanding songs into chunks, lines, entity occurrences, or ending events.
- Keep identical song-content groups in one split.
- Keep exact cleaned-text hash groups in one split or remove cross-split copies.
- Fit vectorizers, vocabularies, scaling parameters, entity priors, rhyme priors, and thresholds on training data only.
- Report label and song eligibility rules, coverage, abstentions, and exclusions.
- Never select the final model on the test set.

## Evaluation contract

### Task 1 — retrieval

- Baselines: BGE-M3 dense; character 2–5 TF–IDF.
- Main model: untuned standardized dense–lexical fusion unless validation data supports a different preregistered weighting.
- Metrics: MRR, Recall@1/5/10, nDCG@10, macro label coverage, stratification by label support, and bootstrap confidence intervals.
- Interpretation: lexical keyness, written-ending concordance, and writing-form concordance are post-hoc evidence; they do not causally decompose the dense encoder.

### Task 2 — NER and networks

- Baselines: reproducible lexicon/rule system and at least one contextual Chinese NER system when available.
- Human evidence: no candidate or model agreement set is called gold. A gold benchmark requires occurrence-level human review, double annotation, and adjudication.
- Metrics when gold exists: strict-span micro and macro precision/recall/F1, per-class results, boundary/type errors, and inter-annotator agreement.
- Networks: lyric co-mention edges and externally verified real-world edges are separate layers with separate labels and provenance.

### Task 3 — written rhyme

- Baselines: global ending-class frequency and training-only transition/Markov context.
- Main model: a context model that predicts the next dictionary-estimated written-ending class with explicit abstention.
- Metrics: Top-1/3/5 accuracy, MRR, coverage, macro/per-support results, bootstrap confidence intervals, and calibration or selective-risk curves when possible.
- Boundary: written Mandarin character pronunciations are not performed rhyme, Flow, voice, or beat.

## Paper contract

The English paper must provide enough detail to reconstruct every task:

- source snapshot, eligibility and cleaning;
- exact split and leakage-control rules;
- representation, model, feature and threshold definitions;
- baselines and ablations;
- metrics, uncertainty, stratified results and error analysis;
- claim boundaries, copyright/privacy protection and withheld analyses;
- a direct mapping from every paper table/figure to a saved aggregate artifact.

The paper must distinguish completed results, provisional outputs, and future work. A visually appealing artifact cannot substitute for an evaluated downstream task.

## Result-site contract

The site has only three primary actions: **Explore a repertoire**, **Explore a cultural entity**, and **Explore or predict a written rhyme**.

Every visible sentence, number, control, card, tooltip, node, or edge must answer at least one of these questions:

1. What is distinctive here?
2. Why are these items connected or recommended?
3. How reliable is the result?
4. What can the visitor do next?

Anything that only describes pipeline mechanics, counts records without interpretation, repeats a heading, or uses unexplained technical jargon is removed from the public interface. Method details belong in the paper. Long lyric passages, private contexts, identifiers, embeddings, and unreviewed NER occurrences never enter the site.

## Publication gate

A task can appear as a paper result only when its validation artifact passes and the claimed evidence level matches the available review. A task may appear as a clearly labelled prototype only when its uncertainty and missing validation are visible at the point of use. No relation is promoted from lyrical similarity or co-mention to collaboration, influence, affiliation, biography, or identity without external evidence.
