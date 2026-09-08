# Figure captions and alt text

## Figure 1 — Research design: one protocol, three spaces

**Caption.** Fig. 1 Study design. One frozen corpus (7,391 song records; 25,026 chunks; 7,236 query songs across 226 labels) enters one protocol — leave-group-out label profiles over exact-text or near-duplicate leakage groups, one query set, no test-outcome tuning — and is scored in three representation spaces. The semantic branch adds the whitening probe and its controls; the lexical branch adds the script-class decomposition and named-reference neutralisation; the rhyme branch adds the next-ending context model. Every branch answers the same question on the same scale.

**Alt text.** A top-to-bottom diagram. A corpus box gives 7,391 song records, 25,026 chunks, and 7,236 query songs across 226 labels. A dark controls box names the shared protocol and the 5,888 leakage groups. Three coloured branches follow: semantic space with the whitening probe, lexical space with decomposition and neutralisation, and rhyme form with a next-ending model; each ends in a label ranking.

**Takeaway.** The corpus, the query set, the leakage unit, and the protocol are shared, so a difference between branches is a difference in representation and nothing else.

## Figure 2 — Held-out-song retrieval in the semantic and lexical spaces

**Caption.** Fig. 2 Held-out-song source-label retrieval in the semantic and lexical spaces. Panel A shows macro MRR, Recall@1/5/10, and nDCG@10 for BGE-M3, character TF–IDF, and their untuned per-query z-score fusion. Panel B shows paired fusion-minus-component differences. Whiskers are 95% intervals from 5,000 paired two-stage bootstrap replicates over 7,236 queries and 226 labels.

**Alt text.** Two dot-and-whisker panels compare five retrieval metrics. In the model panel, fusion is highest on every metric, character TF–IDF is second, and BGE-M3 is lowest. In the paired-difference panel every fusion-minus-component interval lies to the right of zero; gains over TF–IDF are smaller than gains over BGE-M3.

**Takeaway.** The character surface carries more label identity than the raw semantic embedding, and the two are complementary: their untuned fusion is above either on every metric with every paired interval above zero.

## Figure 3 — Cultural-reference evidence after leakage and uncertainty controls

**Caption.** Fig. 3 Provisional cultural-reference evidence after cross-label shared-text exclusion and statistical screening, computed on the earlier corpus snapshot. Panel A keeps three analytical units separate: the candidate inventory falls from 33 corpus-wide entity strings to 23 after shared-text exclusion and to 22 in the fixed 204-label universe; legacy source-label/entity links fall from 85 to 40 and then to six after shrinkage, a conservative interval, and false-discovery control; the legacy co-mention denominator (9 before exclusion, 1 after) is shown separately from the corrected all-song denominator (5 basic-gate candidates; 4 releases). Panel B plots the six released shrunken risk ratios with conservative 95% intervals and adjusted *q* values. Panel C plots the four released same-song co-mentions by normalised pointwise mutual information. Human review is incomplete, so precision, recall, and F1 are not reported.

**Alt text.** Three evidence-control cards show separate progressions for entity strings, source-label/entity pairs, and co-mention pairs. The entity inventory decreases from 33 to 23 to 22. Label-to-reference links decrease from 85 to 40 to 6. The co-mention card separates the legacy denominator, which changes from 9 to 1, from the corrected denominator, which changes from 5 to 4. A log-scale forest plot shows six released source-label-to-place enrichments, every interval above one. A dot-bar panel shows four released same-song co-mentions.

**Takeaway.** Shared-text exclusion and uncertainty control reduce the candidate graph to a small, interpretable set of corpus-internal lyric-reference signals; in the article the layer's lexicon serves as the control that the neutralisation arms remove.

## Figure 4 — Written-ending continuation on corpus v2

**Caption.** Fig. 4 Prediction of the next dictionary-estimated written line-ending family on the strict terminal-Han population of corpus v2. Panel A compares global frequency, first-order Markov, flat context, and hierarchical context with and without the source-credit label across 51,516 leakage-safe events in 1,064 held-out songs; whiskers are 95% song-cluster bootstrap intervals from 2,000 replicates. Panel B separates continuation from family-switch events. Panel C plots four paired score differences for the model with versus without the label; three intervals cross zero and the largest difference, Top-5, is +0.0015.

**Alt text.** Four aligned dot-and-whisker panels show Top-1, Top-3, Top-5, and MRR for five models; context models dominate the global baseline, and the flat and hierarchical models nearly overlap. A grouped bar panel shows near-ceiling Top-3 accuracy when the next line continues the same family and much lower accuracy when it switches. A third panel shows four label-ablation differences on a scale of thousandths, three of them crossing zero.

**Takeaway.** Written endings have grammar — the previous family predicts the next — but they are not signature: the source-credit label adds nothing measurable, which is the rhyme-form result seen from the other side.

## Figure 5 — Identity across semantic, lexical, and whitened semantic spaces

**Caption.** Fig. 5 The same songs in three representation spaces, and how often a song's nearest neighbours share its source-credit label. Panels A–C are t-SNE layouts of one fixed set of 442 songs from the twelve labels with the most songs, at most one song per leakage group, in the BGE-M3 space (A), the character 2–5-gram TF–IDF space (B), and the BGE-M3 space after within-author whitening fitted on songs by every other label (C). One seed and one perplexity are used throughout; cluster sizes and the distances between clusters carry no meaning. Panel D is the measurement the layouts gesture at, over all 7,236 songs and without any projection: for k from 1 to 20, the share of a song's k nearest cosine neighbours, excluding its own leakage group, that carry its label. The whitened curve is fold-wise, so no song is scored in a space fitted on itself. The dotted line is the chance level implied by label sizes.

**Alt text.** Three square scatter panels show the same coloured and shaped points. In the semantic panel the colours are intermixed; in the lexical panel and the whitened semantic panel several labels form visible clusters. A line chart below plots same-label neighbour share against k for three spaces: the whitened semantic curve is highest, starting near 0.28 at k equal to 1, the lexical curve is next, starting near 0.20, and the raw semantic curve is lowest, starting near 0.10; all three decline towards a dotted chance line near zero. A legend lists twelve source-credit labels with their markers.

**Takeaway.** The identity information is in the BGE-M3 vectors, but raw cosine does not expose it; a linear correction of the geometry, almost all of it label-free, makes the semantic space as label-pure as the raw character surface. The same correction lifts the character space too, so the surface keeps its lead under equal treatment.
