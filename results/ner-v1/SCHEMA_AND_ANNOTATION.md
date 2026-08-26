# Chinese Rap Entity Schema and Annotation Guideline v1.1

## Evidence status

This is an annotation protocol, not a gold dataset. The current release contains 800 unreviewed, stratified candidate occurrences for two independent reviewers. Gold labels may be created only after both reviews are complete, disagreements are adjudicated, and the final table is frozen with a manifest.

## Unit of annotation

One task is one character-span candidate in one private lyric context. Annotate the span as used in that context; do not infer a performer's biography, preference, identity, affiliation, collaboration, or social relationship.

## Entity types

| Code | Include | Exclude / boundary |
| --- | --- | --- |
| `PERSON_REFERENCE` | A surface that may name a real, stage-name, fictional, or mythic person in context | This label never by itself means “rapper”; exclude kinship terms, pronouns, roles, common nouns, credit residue, and figurative personification unless a reviewer confirms a named referent |
| `GROUP_CREW_OR_ORGANIZATION` | Named rap crew, collective, label, company, institution, or organized group | Generic “team”, “crew”, “company”, or fandom nouns without a specific name |
| `PLACE` | Named country, city, region, neighborhood, landmark, venue, street, or geographic feature | Generic “home”, “street”, “city”, “world”, or directional language without a specific referent |
| `BRAND_OR_PRODUCT` | Named brand, product line, vehicle brand/model, fashion house, drink/substance brand, or named consumer object | Generic objects or substances; annotate the shortest complete brand/product name |
| `WORK_OR_MEDIA` | Named song, album, film, book, program, platform, law, or other titled cultural work | Generic genres and untitled references |
| `EVENT` | Named historical, political, sporting, cultural, or rap event | Generic events such as “a battle” or “the show” without a specific name |
| `LANGUAGE_OR_DIALECT_REFERENCE` | Named language or dialect | Broad adjectives and scripts when no language is referred to |
| `ETHNOCULTURAL_GROUP_REFERENCE` | Named ethnic, national, or cultural group of people | Country/place names and broad adjectives when no people are referred to |
| `RAP_CULTURE_CONCEPT` | Domain-specific rap practice or culture concept useful for this corpus, such as a named technique or scene term | Ordinary English tokens and generic performance words; this class is not published as general-domain NER without human review |
| `OTHER_CULTURAL_REFERENCE` | A specific named reference that fits none of the above | Use sparingly and explain in notes |
| `NOT_ENTITY` | The proposed span is a common word, metaphor, extraction fragment, header residue, or otherwise not a named/cultural reference | — |
| `UNCERTAIN` | Context is genuinely insufficient after applying the rules | Do not use merely because external linking is unavailable |

## Span rule

Select the shortest complete surface that uniquely expresses the reference in context. Include internal English/Chinese characters and required name particles; exclude punctuation, surrounding titles, hashtags, and possessives unless part of the conventional name. When nested candidates occur, prefer the longer complete named expression and record the shorter candidate as invalid.

## Required reviewer fields

1. `mention_valid`: `VALID`, `INVALID`, or `UNCERTAIN`.
2. `entity_type`: one schema code above.
3. `referential_status`: `NAMED_REAL_WORLD`, `FICTIONAL_OR_MYTHIC`, `METAPHOR_OR_COMMON_WORD`, `AMBIGUOUS`, or `NOT_APPLICABLE`.
4. `normalized_surface`: spelling/case normalization only; do not invent a real-world identity.
5. `linking_status`: `SURFACE_ONLY`, `UNRESOLVED`, `RESOLVED`, or `NOT_APPLICABLE`.
6. `resolved_entity_id`: local or public stable ID only when evidence has been checked; otherwise blank.
7. Confidence from 1 (guess) to 5 (clear), plus a short note for invalid, uncertain, or corrected cases.

## Independent review and adjudication

R1 and R2 must work independently. Exact agreement is calculated separately for mention validity, entity type, and referential status. Any disagreement, either reviewer's `UNCERTAIN`, a corrected span/type, or conflicting entity link requires adjudication. The adjudicator sees both ratings and writes the final decision and rationale. Do not overwrite either original review.

## Gold-release gate

The package may be called gold only when: both reviewers completed every assigned task; adjudication is complete; no blank final labels remain; inter-annotator agreement and per-class support are reported; task IDs and input hashes match the frozen manifest; and a train/dev/test split is produced by song (never by occurrence) to prevent leakage.
