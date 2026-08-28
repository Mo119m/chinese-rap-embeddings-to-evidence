# DSH upload bundle

Prepared for *Digital Scholarship in the Humanities* technical requirements checked 25 August 2026.

## Upload-ready files

- `manuscript.docx` — double-spaced English manuscript, under 9,000 words excluding references, with structured abstract, keywords, data-availability statement, AI-disclosure placeholder, and figure legends/alt text collected at the end. Figures are not embedded.
- `supplementary_methods.docx` — reproducibility and public/private-boundary supplement.
- `fig1.pdf`–`fig4.pdf` and `fig1.svg`–`fig4.svg` — vector submission artwork.
- `fig1.tif`–`fig4.tif` — 600-dpi, 6.5-inch-wide, uncompressed RGB submission artwork. Upload the canonical files from the release-root Figures directory (`figures/` in the repository; `Figures/` in the desktop package). They are not duplicated here because the four files total about 130 MB. Their checksums are recorded in `journal_figure_validation.json`.
- PDF files are previews for author checking; upload policy should follow the journal portal.

## Stop before submission

The frozen-snapshot public release is reproducible and its repository licences are fixed, but PD-002 records an outstanding computational release action. Before journal submission, rebuild the corpus with duplicate-aware song/component control and rerun the affected predictive downstream tasks; the aggregate corpus-lineage sensitivity does not establish predictive robustness on that repaired population. The responsible authors must also enter factual affiliations, corresponding-author email, funding, conflict of interest, CRediT roles, corpus acquisition/provenance, corpus-rights basis, ethics determination, exact AI-tool/model disclosure, and archival DOI. NER precision/recall/F1 must remain unreported until dual human review is complete.
