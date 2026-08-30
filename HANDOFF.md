# Handoff: what changed on `fix/release-integrity`, and what to do next

This branch is not merged. `main` is still at `57ab3fd`, exactly as published. Every
change described here lives on `fix/release-integrity` and is waiting on review.

```
git fetch origin
git log --oneline main..origin/fix/release-integrity
python tools/verify_release_integrity.py        # must print "all 215 ... verified"
```

CI runs that verifier on Linux, macOS, and Windows for every push, plus an assertion
that no path is line-ending translated. It has been green on every commit here.

---

## 1. What was wrong, and what fixed it

### 1.1 Published checksums only validated on Windows

The builders wrote CSV records with the `csv` module's default CRLF terminator and
wrote JSON and Markdown through Windows text-mode translation, then recorded SHA-256
over those CRLF bytes. Git normalised the same files to LF on commit. **69 of the
published checksums therefore validated only on a Windows checkout** and failed on
Linux, macOS, and `raw.githubusercontent`.

Fixed at the source: `lineterminator="\n"` on every `csv.DictWriter` and
`DataFrame.to_csv`, `newline="\n"` on every `write_text`, and `.gitattributes`
extended from `results/repertoire-network-v1/** -text` to `* -text` so the bytes that
are hashed are the bytes that are committed. All manifests were then re-derived.

Verified by cloning the branch three times under `core.autocrlf` of `true`, `input`,
and `false` and running the verifier in each: 215/215 in-repository claims pass in
all three. Before the fix, two of those three failed 85 claims.

### 1.2 Two different hashes for the same file

`sync_chinese_rap_portable_site_v1.py` wrote `index.html` through Windows
translation (CRLF) but hashed `html.encode("utf-8")` (LF), one line apart. That is
why `validation/release_validation.json` and `validation/portable_site_manifest.json`
disagreed about `index.html`. Both now agree with the committed bytes.

### 1.3 `figures/validation.json` accumulated three runs of the same checks

`update_primary_validation()` filtered three retired check names out of the existing
file but not the four it was about to append, so every builder run stacked another
copy. The file carried 30 records under 22 names, with three mutually inconsistent
hash sets for `fig1-4.pdf/.svg`; only the last matched the committed files. The
builder now filters on the names it is actually writing, and the file was deduplicated
keeping the current run.

### 1.4 No licence

The repository was public with no licence, which defaults to all rights reserved.
Now `LICENSE` (CC BY 4.0, verbatim legal code so GitHub detects it) covers the
manuscript, figures, methods, documentation, and aggregate results; `LICENSE-CODE`
(MIT) covers `src/`, `site/`, and `tools/`. Neither covers the lyric corpus, which is
not redistributed.

### 1.5 The central question was worded two ways

`README.md` and `validation/release_validation.json` said *lyrical repertoires*;
`methods/RESEARCH_CONTRACT.md`, the manuscript, Figure 1, and the application said
*lyrical identities*. The V4 builder had switched wording without propagating it.
Aligned on *identities*, which required editing two strings rather than re-rendering
Figure 1, and which matches the frozen contract.

The application's `h1` carried *lyrical identity* with none of the scaffolding the
manuscript gives the term. Both renderings now carry the definition inline.

### 1.6 156 MB of duplicated TIFFs

`submission/dsh/fig1-4.tif` were byte-identical to `figures/fig1-4.tif`. Removed;
`README_BEFORE_SUBMISSION.md` now says to upload those four from `figures/`. The
bundle went from 131 MB to 1.2 MB. History still contains the old blobs, so `.git`
did not shrink; rewriting history was deliberately **not** done.

---

## 2. What the analysis found

**Read this section as amended.** An independent review checked all four supporting
analyses and found real problems in three of them, plus several numbers in earlier
drafts of these documents that contradicted the data committed alongside them. The
corrections are folded in below; `paper/PENDING_MANUSCRIPT_INSERTS.md` marks which
manuscript passages are consequently blocked.

Numbers corrected since the first draft of this file:

| earlier claim | correct |
| --- | --- |
| observed graph 87 edges / 94 labels | **86 / 93** — the run used the wrong centroid weight |
| "one edge from float32/float64 at the rank-five boundary" | a real bug, not float noise |
| observed nearly 3 sd above the null | about **6.1 sd** |
| median retention 0.82 | **0.74** (the tool also computed 0.75; both wrong) |
| alpha negative for 12 of 20 surfaces | **15 of 19** |
| alpha above 0.25 for three surfaces | **two** (台北 0.64, 上海 0.26) |
| ablation control replaced ~51,000 | **46,619**, matching 559 of 605 surfaces |

### 2.1 Null baseline for the repertoire graph — VERIFIED

`tools/null_baseline_reciprocal_edges.py` rebuilds the 204 label centroids from the
chunk embeddings, permutes the label column of the label-song incidence — preserving
every label's song count and every song's label count — and reapplies the mutual
top-five rule under both representations.

Both centroids weight members by `comparison_text_weight`; the weight sum must equal
the effective text mass in the node rowmap. The first version used
`frozen_analysis_text_weight` for the primary layer, which reproduces that centroid to
a cosine of 0.9999983 rather than 1.0 — enough to move one pair across the rank-five
boundary and yield 87 edges instead of 86. I saw the imperfect cosine, called it float
noise, and did not check the mass column. It was a bug. With the correct weight the
graph rebuilds to **exactly 86 edges and 93 labels** and the gate now demands that.

**Result: observed 86, null mean 45.0 (sd 6.7, max 68 over 200 replicates), one-sided
p = 0.005, 1.91 times the null mean, about 6.1 sd above it.**

The null mean is not small — an uninformative label assignment still yields about 45
edges — but that does **not** license saying half the released graph is chance. The
test says nothing about which observed edges are spurious and is not an FDR. It also
holds the shared-text mask and eligibility set fixed at their true-assignment values,
so it is a conditional null; recomputing both inside each replicate is the obvious
sensitivity analysis.

### 2.2 Occurrence audit — VERIFIED on the accepted set

`tools/audit_released_claim_occurrences.py` now resolves each claim from the private
candidate table (`strict_high_consistency`, excluding `cross_label_shared_cleaned_text`)
and refuses to report until the per-claim support equals the published figure. **All
ten released claims reproduce exactly.** An earlier version enumerated raw string
matches, a wider pool, and found 12 songs for 黑麦–天津 where the release publishes 10.

The six label links rest on 102 accepted occurrences. Two read as a compound rather
than the entity: one 巴黎世家 in the 伦敦–巴黎 co-mention, and one 湖南卫视 in the
泰格西–湖南 link, which has the smallest support of the six. Four occurrences under
黑麦–天津 are 天津卫, a historical name for the city itself; one under 中文–英文 is the
genre term 中文说唱.

This is a compound check, not a semantic one: `bare surface` means "not followed by a
known compound", and a figurative use of a bare surface would not be detected.

### 2.3 Multi-tagger agreement — RERUN on the release's candidate frame

`tools/multi_tagger_agreement.py` now scores the pipeline's own lexicon candidates,
each inside its recorded context window at its recorded span. An earlier version
scanned the sidecar text and scored a superset: 334 candidates for 上海 against the
release's 174.

The corrected population reproduces the published candidate count exactly for 13 of
the 22 surfaces and lands within a few on most of the rest; the headline is computed
over the thirteen. Of those, **nine have a negative alpha and two exceed 0.25** (台北
0.62, 昆明 0.30). The frame matters and favours the release: 上海 scores 0.47 here
against 0.26 on the superset.

Defensible: these two models, on these candidates, agree poorly, so adding taggers is
not a route to a precision estimate. Not defensible from two models: that the
disagreement is inherent to the task in this domain.

### 2.4 Entity ablation — RERUN with a matched control

The control now pairs **every** surface with the closest-frequency non-entity n-gram —
605 of 605, total occurrences within ±0.1% of the entity condition — and is drawn
three times. The earlier band-matched control skipped 46 surfaces and removed 10.5%
fewer characters, which was the confound it existed to remove.

Masking all 605 surfaces costs 0.0069 macro MRR. The control costs a mean of 0.0009,
with draws at −0.0008, +0.0009 and +0.0020. **Adjusted difference: +0.0061**, and the
entity drop exceeds every control draw. Masking only the 22 released surfaces gives
+0.0001 against a control range of −0.0003 to +0.0001 — a power statement, since those
occurrences are 0.05% of the corpus by character.

Three draws give a range, not a confidence interval, and the comparison is matched
rather than decomposed: 0.0061 being about a fifth of the published 0.031 fusion gain
places the magnitude without apportioning it.

## 3. Tool inventory

| tool | needs | status |
| --- | --- | --- |
| `verify_release_integrity.py` | nothing | run; in CI on three platforms |
| `summarise_surface_reliability.py` | public tables only | run; output in `analysis/` |
| `audit_surface_collocations.py` | private corpus | run |
| `audit_released_claim_occurrences.py` | private corpus | run |
| `multi_tagger_agreement.py` | private corpus, torch, transformers | run with two taggers |
| `entity_ablation_retrieval.py` | private corpus, scikit-learn | run, both variants |
| `null_baseline_reciprocal_edges.py` | private embeddings and membership | run |

Everything in `analysis/` is aggregate counts only. The tree was scanned before
commit: the longest contiguous Chinese run in any committed analysis file is five
characters (a compound named in prose). No lyric lines are present.

### Environment notes for whoever runs these next

- The builders live in `src/` in this repository and in `work/` in the private
  working tree. `entity_ablation_retrieval.py` and the null baseline accept either.
- `torch` fails to import in a venv created with `--system-site-packages` on top of a
  conda base: the conda MKL/OpenMP DLLs conflict. Use a clean venv.
- `pip install torch` also fails if the venv path is long — Windows long-path support.
  Put the venv somewhere short.
- The retrieval harness needs `numpy`, `scipy`, `scikit-learn`; not `pandas`.
- The NER builder additionally needs `pandas`, `regex`, and `torch`.

---

## 4. Recommended plan

**In priority order.** The first item is the only one that cannot be automated.

1. **Paste `paper/PENDING_MANUSCRIPT_INSERTS.md` into `manuscript.md` and rebuild the
   DOCX and PDF.** Four passages are drafted and reflect the evidence above. This was
   deliberately not done here: the PDF step is not part of the tracked pipeline, so
   editing the Markdown alone would leave four binary derivatives disagreeing with
   their source while every checksum still validated. Delete that file afterwards.

2. **Merge and tag.** NOT a fast-forward: `codex/release-integrity-publication` is
   not an ancestor of this branch, so `git merge --ff-only` from main would silently
   discard that branch's PD-002 amendment, corpus reconciliation, repaired-population
   boundaries and superseded-audit markers. The required sequence is a true merge of
   `fix/release-integrity` with `codex/release-integrity-publication` (see AGENT_BRIEF.md for the measured conflict list, the nine-step order, and the
   regenerate-don't-hand-merge rule). Merging is the last step, after the repaired corpus,
   the audit, the cleaning policy, the downstream rebuild, the paper update, green CI and
   author approval. After all of that, then
   `git tag -a v4.0.0`, then enable the repository in Zenodo *before* publishing the
   GitHub release — the order matters or no DOI is minted. This closes the
   archival-DOI item in `validation/RELEASE_READINESS_V4.md`.

3. **Consider reframing the cultural-reference claim from entities to strings.**
   "Six source labels over-use six curated place-name strings, after shared-text
   exclusion and relative to a fixed comparison universe" is true by construction and
   needs no precision estimate. The occurrence audit makes the entity reading
   defensible, so this is now a choice rather than a necessity — but the string-level
   statement is the one that survives a hostile reading.

4. **Re-encode the ablated corpus with BGE-M3 and rerun the ablation with
   `--dense-npy`** to get the dense and fusion arms. Currently only the lexical arm is
   reported, because the stored embeddings describe the original text.

5. **Manuscript-derivative staleness is now guarded.**
   `tools/check_manuscript_derivatives.py` records which Markdown hash each of the ten
   DOCX and PDF derivatives was built from and fails when the source moves ahead of
   them. CI runs it. Rerun with `--record` immediately after a rebuild, never to
   silence a failure. This closes the one failure mode the checksum manifests cannot
   see: they describe each file individually, so a stale derivative validates
   perfectly while contradicting its source.

6. **Smaller items, none blocking:** UTF-8 BOM is inconsistent across `results/` CSVs
   (25 of 139 text files carry one) because writers mix `utf-8` and `utf-8-sig`;
   fixing it moves 30-odd hashes. `requirements.txt` pins no Python version and does
   not record the BGE device/CUDA/fp16 state, which will block reproduction.
   `site/.openai/hosting.json` is deployment scaffolding unrelated to the research but
   is imported by `site/vite.config.ts`, so it cannot simply be deleted; splitting
   `site/` into its own repository is the clean fix. The name "Verseprint" appears only
   in the page `<title>` and the social preview image, nowhere in the README or paper.

### What not to do

- **Do not validate the label-place edges against real artists' biographies.** The
  release explicitly disclaims biography, residence, and social relation. Confirming
  黑麦–天津 against an artist's actual hometown is exactly the inference the claim
  boundary forbids, and it converts a corpus-internal statistic into an assertion about
  a real person.
- **Do not rewrite history to shrink the repository.** It is public; rewriting breaks
  every existing clone for a saving that a research repository shipping 600-dpi
  submission artwork does not need.
- **Do not report any model-assisted annotation as human validation.** Inter-method
  reliability and inter-annotator reliability are different quantities. Without a human
  anchor there is no way to estimate a model annotator's own error rate, so no
  precision or recall figure can be derived from one.
