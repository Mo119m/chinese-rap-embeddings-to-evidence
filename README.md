# From Embeddings to Evidence: Chinese Rap Lyrical Repertoires

This repository contains the paper, aggregate results, publication figures, analysis/build scripts, and a self-contained interactive Atlas for a claim-bounded study of Chinese rap lyrics.

## Start with the result

- Inspect the independently audited [held-out-song retrieval release](results/retrieval-v1/README.md), including metrics, paired uncertainty intervals, leakage controls, and aggregate label-level diagnostics.
- Open [`index.html`](index.html) to inspect 204 corpus credit-label profiles and the 16 lyric matches that reappeared in at least half of 250 within-label song resamples.
- Read the [paper](paper/From_Embeddings_to_Evidence.pdf) for the research questions, methods, results, and limitations.
- Read [Methods and limits](methods/METHODS_AND_LIMITS.md) for a compact reproducibility contract.

The Atlas is not a keyword-search demo. A label profile reports characteristic words, dictionary-estimated written endings, and writing habits. A pair view reports exactly which independently measured signal the profiles share and the match's song-bootstrap repeatability.

## Main empirical results

- In the frozen downstream evaluation, equal-weight standardized fusion retrieves the correct source-credit label's remaining repertoire with macro MRR **0.447** (95% hierarchical-bootstrap CI 0.414–0.481) and Recall@10 **0.611** across 5,455 held-out songs and 204 labels.
- Fusion improves over character TF–IDF by +0.031 MRR (95% CI 0.021–0.042) and +0.050 Recall@10 (0.033–0.066); all five paired metric intervals are above zero. BGE-M3 alone reaches MRR 0.320, while character TF–IDF alone reaches 0.416.
- An earlier 1,000-query encoder sanity check showed the same ordering and motivated the dense–lexical design; it is not the paper's primary downstream estimate.
- Two duplicate-control representations agreed on 86 reciprocal top-five matches among 204 eligible labels.
- Only 16 of those 86 matches reappeared in at least 50% of 250 within-label song resamples; none reached 80%.
- Eleven of the 16 repeatable matches have a separately calculated lexical, written-ending, or writing-form signal. Five are overall wording matches with no single post-hoc trait passing the calibration gate.

## Repository layout

- `paper/` — English manuscript in PDF, DOCX, and Markdown.
- `figures/` — four publication figures in PNG and 600-DPI TIFF.
- `results/` — independently audited downstream retrieval release, encoder benchmark, all 86 bootstrap-audited matches, and the 16-match core.
- `src/` — research and artifact builders.
- `validation/` — automated artifact checks.

## Reproduction boundary

The scripts expect an access-controlled lyric corpus, cleaned-text sidecars, and frozen BGE-M3 embeddings in the directory contracts documented in the source. Those inputs are not included because of copyright and privacy constraints. Public outputs contain no full lyrics, song/chunk identifiers, embeddings, private membership rows, or unreviewed NER contexts.

Source credit labels have not been independently identity-verified. Results describe written lyrical repertoires and do not establish collaboration, influence, friendship, biography, hometown, preferred genre, performed rhyme, flow, voice, or beat. NER results are withheld pending occurrence-level two-reviewer validation.

## Status

The technical package passes the included validations. Before journal submission, the human authors must fill the author/funding/CRediT fields, select licences for code and research outputs, provide the final repository DOI and lyric-access statement, and complete the exact AI-use disclosure.
