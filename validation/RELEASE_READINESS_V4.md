# Release Readiness V4

## Done

- **Clear research theme:** language, cultural reference, and dictionary-estimated written rhyme are three branches of one lyrical-repertoire question.
- **Global → local network:** the application first shows all 204 eligible source-credit labels, 86 released reciprocal edges, 93 connected labels, and a 16-edge ≥50% repeatability view; clicking any node opens its focused network below.
- **Useful relationship explanation:** every local edge states the mutual-top-five rule, its auxiliary vocabulary/written-ending/writing-form signal when gated, and its return count across 250 song-level resamples.
- **Meaningful downstream evaluation:** retrieval uses held-out songs and paired uncertainty; cultural-reference links use shared-text exclusion, support, conservative intervals, and BH-FDR; written-ending prediction uses song-held-out evaluation, baselines, ablation, calibration, and switch diagnostics.
- **Released-claim audit prepared:** a private, blinded dual-review package covers all 157 occurrences supporting the 10 released cultural-reference claims; the public protocol and aggregate status expose coverage and hashes without lyric contexts or locators.
- **Corpus lineage reconstructed:** the aggregate-only PD-002 audit exactly reconstructs the legacy 7,214-song/22,132-chunk cleaner output, separates it from the later 7,211-song/22,128-chunk canonical downstream input, and publishes the duplicate-loss diagnostics without lyric text or row identifiers. This is a release-lineage audit, not a fourth downstream task.
- **No Command-F-style output:** the release does not expose a generic word-occurrence search. Search is limited to choosing a source label or supplying a written ending to an evaluated model/table.
- **Academic presentation:** the manuscript is English, double-spaced, under 9,000 words before references, uses a structured abstract and Oxford HUMSOC citations, and separates upload figures. The four figures are 6.5 inches wide, 600 dpi, and at least 7 pt at print size.
- **Claim boundaries:** source-credit labels are not verified people; textual proximity is not friendship/collaboration/influence; cultural references are not biography/residence/preference; dictionary pinyin is not audio rhyme/flow/beat.

## Partial by design

- The global PCA position is an approximate overview explaining 26.2% of profile variation. Only a released line defines the stricter reciprocal relation.
- Sixteen of 86 repertoire edges return in at least 50% of resamples; the remaining 70 are displayed as lower-repeatability candidates, not as equally stable facts.
- Cultural-reference extraction is statistically screened but still provisional because human occurrence gold is 0/800.
- The rhyme tool can rank plausible written-ending families, but exact switches remain difficult and source-label conditioning did not improve held-out prediction.
- PD-002 has aggregate sensitivity results but remains `pass_with_release_action`: the duplicate-aware corpus repair and affected retrieval, graph, reference, and rhyme reruns are not yet complete, so repaired-corpus predictive robustness is withheld.

## Human required before journal submission

- Complete author names, affiliations, corresponding-author email, funding, conflict of interest, and CRediT roles.
- Supply documented corpus acquisition, sampling, dates, temporal coverage, lyric origin, custody, rights/licence basis, ethics determination, and access policy.
- Mint an archival DOI, finalize the exact AI-tool/model disclosure, and inspect the journal portal upload preview.
- Complete dual human NER review before reporting precision, recall, or F1.

The frozen-snapshot computational and public-share package passes its V4 checks with the disclosed PD-002 boundary. It is **not** marked journal-submission-ready until the duplicate-aware repair and affected predictive reruns are complete and the author-owned facts above are supplied.
