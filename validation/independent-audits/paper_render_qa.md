# Final DOCX/PDF render QA

Status: **PASS**  
Checked: 2026-08-26 UTC

## Artifacts

- Main manuscript: 39 Letter pages; tagged PDF; title metadata present; author metadata intentionally blank pending author completion.
- Supplementary Methods: 4 Letter pages; tagged PDF; title metadata present; author metadata intentionally blank pending author completion.
- Main DOCX SHA-256: `bd1231028e8c4b47020e0b6254ae801f3a67b3cb551b80e60512d46fa4afd2a1`
- Main PDF SHA-256: `e071e65ca2106bff36c5b0de068d6f7d0ce4009a4f8a925be2b8c22df594c4bf`
- Supplement DOCX SHA-256: `c9d1115fc8046ccb2fdd28e3eb25dc92093951ceb83e35c80fb22845d4769f7c`
- Supplement PDF SHA-256: `9d8309de2e2e9a70212e7530b711f74f9d0d545a1f5e203d6095caa259a5ce43`

## Visual review

All 39 manuscript pages and all 4 supplement pages were rendered to PNG and visually inspected. After the final public-boundary wording correction, manuscript pages 8, 33, and 34 and supplement pages 3 and 4 were re-rendered and re-inspected with unchanged pagination. The review checked:

- no clipped, overlapping, or off-page text;
- readable four-figure placement and captions;
- readable four main tables and Supplementary Table S1;
- intact Chinese characters and mathematical symbols;
- correct fusion equation, NER interval notation, and hierarchical written-ending equation;
- clean title, abstract, declaration, and reference pages;
- stable final pagination after the last content edits.

## Structural and text checks

- `Structured Abstract` and `References` each occur once in extracted PDF text.
- No stale `p_L`, `p_R`, `s_q`, LaTeX thin-space escape, or “preregistered research contract” string remains in the final PDF.
- Independent manuscript audit found zero frozen-result numerical mismatches, 26/26 citation closure, and a 204-word abstract; a final text-only re-audit is recorded separately.

## Remaining human-owned submission fields

The files intentionally retain author, affiliation, corresponding-email, funding, conflict-of-interest, CRediT, archival DOI, provenance/rights, and ethics-confirmation actions. These are facts that cannot be inferred from the corpus or analysis.
