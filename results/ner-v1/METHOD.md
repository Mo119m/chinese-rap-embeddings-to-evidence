# Method and Claim Boundaries

## Research question

Which named-person surfaces, groups or crews, places, brands/products, works/events, languages/dialects, and ethnocultural-group references can be recovered reproducibly from this Chinese-rap lyric corpus, and which aggregate co-mention patterns remain after duplicate control and conservative cross-method agreement gates? A `PERSON_REFERENCE` is never interpreted automatically as a rapper identity.

## Input audit

- Clean lyric sidecar: 21,553 eligible chunks; private text is never copied into the public artifact.
- Reviewed lexicon: 605 screened surfaces. These are surface-level review decisions, not occurrence-level truth.
- Existing context audit v2: 600 tasks found across evaluation and coverage frames; 300 evaluation tasks have reviewer forms and 0 reviewer decisions are completed. A further 289 v1 queue rows are also unreviewed.
- Existing lexicon occurrence audit: 300 tasks found, 0 completed reviewer decisions.
- Existing verified factual entity registry rows: 0; verified relation-evidence rows: 0.

Because completed occurrence-level human gold is absent, this release does **not** report precision, recall, F1, or a train/dev/test benchmark.

## Baseline A — reviewed lexicon exact matcher

The first baseline maps the screened domain lexicon to the project schema and performs case-sensitive literal matching. ASCII/digit surfaces require non-ASCII-word boundaries. Overlaps are resolved globally by longest span, then earliest start. This baseline has domain coverage but can still confuse a named surface with a common word in a particular lyric context.

## Baseline B — general-domain Chinese transformer NER

The second baseline is the locally pinned `ckiplab/albert-tiny-chinese-ner` token classifier. Its config SHA-256 is `4828a80f3c7eb67514f3d9c76ed2544149bad4e498f6b74e4978b48a542b5b5b` and weights SHA-256 is `03f6e38f92ada4b59b88ae9122a50a0c98b85f07722ff1234048928e55ed10d3`. It is a Traditional-Chinese, general-domain model; Chinese-rap slang and stage names are a domain-shift risk. Lines are processed in 180-character windows with 40-character overlap, and overlapping window predictions are deduplicated. The model is used as a reproducible independent candidate baseline, not as an oracle.

## Common comparison frame

Both baselines run on every unique lyric line containing a target-schema lexicon candidate plus a deterministic hash-ranked background sample of non-lexicon lines. `baseline_comparison.csv` and `cross_method_agreement_by_type.csv` count each identical lyric line/span/type once, so their agreement ratios use one consistent unique-line frame. Public entity support is separately counted over duplicate-controlled full-song lyric-content units. Agreement statistics are candidate overlap statistics only; they are not accuracy metrics.

The full source-occurrence frame contains 3,566 exact span/type agreements, of which 3,290 also clear the 0.80 confidence rule. After identical lyric line/span/type combinations are counted once, the corresponding counts are 2,011 and 1,888. Occurrence counts are repeated corpus spans and must not be interpreted as independent samples.

## Private annotation sample

The private package contains 800 distinct candidate occurrences stratified by proposed entity type and agreement state. It includes exact-match agreement, overlap agreement, type/boundary conflicts, lexicon-only candidates, and transformer-only candidates. R1/R2 templates are independent, and a separate agreement/adjudication template preserves both ratings.

## Provisional public entity gate and surface/type audit

The corpuswide sensitivity inventory first requires exact span and schema-type agreement between both baselines; transformer confidence at least 0.80; at least five strict-agreement occurrences; at least five distinct full-song lyric-content units; at least three source-credit labels; strict agreement on at least 50% of that lexicon surface's candidate occurrences; and mean transformer confidence at least 0.80. The primary graph then reapplies those gates after restricting to the 204 retrieval graph-eligible source-credit labels and removing every exact cleaned-text hash observed under more than one canonical label. There are 2,187 such hashes. This changes the semantic-release inventory from 33 corpuswide sensitivity entities to 22 primary graph entities.

Every surface/type pair receives an explicit decision in `surface_type_release_audit.csv`. A `PERSON_REFERENCE` is released only when the surface is also present in the independently screened T1 named-individual ledger; this supports a named-person surface, not rapper identity, authorship, or biography. Conventionally geographic place names and direct language names may be released as narrow surface types. Ambiguous surfaces such as `中南海`, `桃源`, and `西山` remain withheld. Only `RELEASE_PROVISIONAL_PRIMARY` rows can enter the primary graph.

## Shared-text exclusion and analysis universes

NER candidate discovery and the private annotation package cover all 21,553 eligible cleaned chunks. The primary association graph uses a separately named universe: the 204 source-credit labels frozen by the retrieval graph registry. A cleaned-text hash is excluded from label associations and co-mentions if it appears under more than one canonical label anywhere in the eligible corpus, including labels outside the 204-label graph universe. The private audit retains every source occurrence and the hash-level exclusion ledger; public outputs contain only aggregate sensitivity counts.

`entity_inventory_corpuswide_sensitivity.csv` is an explicitly non-primary corpuswide inventory. `shared_text_exclusion_entity_sensitivity.csv`, `source_label_entity_link_sensitivity.csv`, and `entity_co_mention_sensitivity.csv` show before/after consequences. Across all labels, shared-text exclusion removes 539 of 1,474 strict occurrences attached to the corpuswide semantic-release inventory and changes that inventory from 33 to 23 surfaces. Restriction to the 204-label graph universe then yields 22 primary surfaces. Inside that universe, 532 of 1,420 strict occurrences are removed. These counts remain repeated corpus occurrences, not independent observations.

## Grounded cultural network

The public network has two bounded edge meanings:

1. `SOURCE_LABEL_TO_LYRIC_REFERENCE`: the entity must occur in at least five of at least ten eligible source-labelled song units, cover at least 5% of those units, have a Jeffreys-smoothed risk ratio of at least 1.50 versus the rest of the graph universe, have a conservative 95% risk-ratio interval lower bound above 1.0, and pass Benjamini–Hochberg FDR at q ≤ 0.05 across every 204-label × released-entity test. The table reports raw lift, Jeffreys rates and intervals, the shrunken risk ratio, p-value, q-value, and reliability class. It describes a source-labelled lyric repertoire, not biography or preference.
2. `SAME_SONG_LYRIC_REFERENCE_CO_MENTION`: two provisional entities must appear in at least five distinct full-song lyric-content units across at least three source-credit labels, with lift at least 1.25, positive normalized PMI, and BH q ≤ 0.05. The NPMI denominator is **all 5,681 eligible shared-text-excluded full-song units**, including units with no released entity. It is a textual co-mention, not collaboration, influence, affiliation, identity, or a social relationship.

The association denominator has 5,681 source-label/song membership units; the co-mention denominator has 5,681 distinct full-song-content units. Hashes, song IDs, chunk IDs, lyrics, contexts, and embeddings remain private.

For exact reconciliation with v1, the all-label, support/lift-only label-reference count changes from 85 to 40 after shared-text exclusion. The legacy entity-bearing-denominator co-mention count changes from 9 to 1. Restricting to the 204-label graph universe leaves 40 legacy-gate label candidates and 1 legacy-denominator co-mentions. The v1.1 all-eligible-song denominator yields 5 basic co-mention candidates. Uncertainty and BH-FDR produce the primary release of 6 label-reference edges and 4 co-mention edges.

## Limitations

The source-credit labels are not independently identity-verified. Literal lexicon matching is context-insensitive, the transformer faces domain and script shift, model confidence is not calibrated on rap lyrics, confidence/agreement gates are not substitutes for occurrence gold, and exact shared-text exclusion changes the estimand toward label-specific repertoire. Jeffreys intervals and BH-FDR quantify fixed-corpus song-unit uncertainty and multiplicity; they do not create external-population generalizability. Public results remain provisional until dual review and adjudication are complete.
