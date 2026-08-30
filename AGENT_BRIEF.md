# Brief for the next agent

Repository: `Mo119m/chinese-rap-embeddings-to-evidence` (public). It is the release
package for a Chinese-rap lyric study being prepared for *Digital Scholarship in the
Humanities*. Read `HANDOFF.md` on `fix/release-integrity` first; this brief is the
short version plus the one job that is actually blocked.

## Current state

| ref | sha | what it is |
| --- | --- | --- |
| `main` | `57ab3fd` | exactly as published; untouched |
| `fix/release-integrity` | `16ac655` | 10 commits at time of writing; CI green on Linux/macOS/Windows across five checks. Check `git log` for the current tip. |
| `codex/release-integrity-publication` | `978a28c` | 4 commits; branched off `c64473b` |

Both feature branches share the first three commits (`6d04ad1`, `b15cbf1`,
`c64473b`) and then diverge. Merge base is `c64473b`.

## The blocking job: reconcile the two branches

A real simulated merge (`git merge-tree --write-tree HEAD
origin/codex/release-integrity-publication`) produces **26 conflicts**. This branch is
`fix/release-integrity`; the Codex branch is `codex/release-integrity-publication` at
`5fe4e41`. An earlier brief said six conflicts, measured before the privacy history
rewrites removed the merge commit `2666d85`/`978a28c` from this branch, so the branches
diverge again over every file both sides regenerated.

The order of work is not negotiable, and merging is the LAST step, not the first:

1. agree the scientific boundary between the two branches
2. generate and freeze the PD-002 repaired corpus, with its manifest and hash
3. run the MB audit over the repaired corpus, and build its independent gold set
4. decide and apply the cleaning policy
5. rebuild the NER candidate frame and every affected downstream result
6. update the manuscript, the site and all derived files
7. full-matrix CI green
8. author approval
9. only then merge and tag

Conflicting paths:

```
.github/workflows/release-integrity.yml
README.md
figures/manifest.json
paper/Chinese_Rap_Evidence_Grounded_Manuscript.docx
paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf
paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx
paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf
paper/Chinese_Rap_Evidence_Grounded_Supplement.docx
paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf
paper/derivative_provenance.json
paper/manuscript.md
paper/supplementary_methods.md
src/build_chinese_rap_release_v4.py
src/normalize_public_text_v1.py
src/update_public_result_manifests_v1.py
src/validate_public_release_integrity_v1.py
submission/dsh/MANIFEST.json
submission/dsh/README_BEFORE_SUBMISSION.md
submission/dsh/manuscript.docx
submission/dsh/manuscript_preview.pdf
submission/dsh/supplementary_methods.docx
submission/dsh/supplementary_methods_preview.pdf
tools/check_manuscript_derivatives.py
validation/MANIFEST.json
validation/RELEASE_READINESS_V4.md
validation/release_validation.json
```

Most are regenerated artefacts (manifests, DOCX/PDF derivatives, validation
records): resolve those by REGENERATING on the merged tree, never by hand-merging
hunks. The content conflicts that need real decisions are README.md, manuscript.md,
supplementary_methods.md, the release builder, the validator, the derivative checker
and the CI workflow — on each, keep the union of both branches' checks; do not drop
the Codex branch's PD-002 amendment, corpus reconciliation, repaired-population
boundaries, or superseded-audit markers.

The three `validation/*.json` are manifests that get re-derived anyway, so resolve
them by regenerating rather than by hand-merging hunks. The other three are real
content overlaps.

**Unique to `codex/release-integrity-publication`:**
- rebuilt `paper/*.docx` and `paper/*.pdf` derivatives, and re-rendered `figures/`
- `results/retrieval-inductive-sensitivity-v1/` — inductive TF-IDF sensitivity
- `results/repertoire-network-v1/robustness/` — a degree-preserving null and
  projection-fidelity diagnostics
- `results/ner-v1/released_claim_audit_status.json` + protocol + builder
- `.python-version`, `requirements.txt` updates
- BOM and line-ending normalisation across `results/` — verified byte-identical to
  the other branch after normalising, so **no data was re-run**
- `.github/workflows/release-integrity.yml`

**Unique to `fix/release-integrity`:**
- `tools/` — eight standalone checks, all runnable, most already run
- `tests/test_tools.py` — checks the pure functions behind every reported number (numpy only, no framework); CI runs it on all three platforms
- `analysis/` — four result sets with READMEs
- `HANDOFF.md`, `paper/PENDING_MANUSCRIPT_INSERTS.md` (five drafted passages)
- the `hero-note` qualifier in `index.html` and `site/app/page.tsx`
- `.github/workflows/verify-release.yml`

Both branches added a CI workflow doing the same thing under different filenames.
Keep one. Whichever survives must still assert **both** halves of the integrity
property: that every published checksum matches the checkout, and that no path is
line-ending translated (`git ls-files --eol`, index eol equals worktree eol and every
path carries `attr/-text`).

## Think about this before deduplicating anything

**The two null baselines are not duplicates.** Deleting either would be a mistake.

- `results/repertoire-network-v1/robustness/` holds the primary layer's 140 edges
  fixed and rewires the sensitivity layer with degree-preserving swaps. It asks
  whether the 86-edge cross-treatment intersection is non-random. Null mean 4.5,
  observed 86, p ≈ 1e-4.
- `analysis/null-baseline/` permutes the label-to-song assignment, rebuilds the 204
  centroids from the chunk embeddings, and reapplies the whole rule. It asks whether
  the label carries information at all. Null mean **45.0** edges, observed 87,
  p = 0.005 at 200 replicates.

The first conditions on the primary layer, which already encodes the real
label-to-content association, so it cannot detect that roughly half the released
graph sits at the density any label assignment would supply. Report both. Reporting
only the first overstates the evidence.

## Two things I checked so you do not have to

`src/update_public_result_manifests_v1.py` on the codex branch walks an explicit list
of paths under `results/`, not a blind glob over every JSON, so it will **not** clobber
`paper/derivative_provenance.json` — the record the manuscript staleness guard depends
on. That was the obvious way for the two branches to break each other and it does not
happen.

`site/vite.config.ts` imports `.openai/hosting.json`, so that deployment scaffolding
cannot simply be deleted. Splitting `site/` into its own repository is the clean fix.
I did not touch the vite config because there is no node toolchain on the machine I
worked from and I will not ship a build change I cannot build.

## Review outcome — all four analyses now rerun and verified

A review on the codex side found real problems in three of four supporting analyses.
Its numerical criticisms were verified independently and were correct. All three have
since been rerun on the correct populations and every manuscript insert is unblocked.

| analysis | status |
| --- | --- |
| null baseline | **verified** after a centroid-weight fix; observed 86/93 exactly |
| occurrence audit | **verified**; all ten claims reproduce their published support |
| multi-tagger agreement | **rerun** on the release's candidate frame, 13 of 22 reproducing |
| entity ablation | **rerun** with a control matching 605 of 605 surfaces, three draws |

Corrections made, all confirmed against the data:

| earlier claim | correct |
| --- | --- |
| observed graph 87 edges / 94 labels | **86 / 93**; the primary centroid used `frozen_analysis_text_weight` where the release uses `comparison_text_weight` |
| "one edge from float32/float64 rounding" | a real bug, not float noise |
| observed nearly 3 sd above the null | about **6.1 sd** |
| median retention 0.82 | **0.74** (the tool also computed 0.75) |
| alpha negative for 12 of 20 on a superset frame | **9 of 13** on the release's own candidates |
| alpha above 0.25 for three | **two** (台北 0.62, 昆明 0.30) |
| six links rest on 131 bare occurrences | raw matches, not the accepted set; the accepted set is **102 occurrences**, of which one is 湖南卫视 |
| control replaced ~51,000 | band matching skipped 46 surfaces; closest-frequency matching now pairs **605 of 605** within ±0.1% |
| attributable +0.0067 | **adjusted difference +0.0061** against a control range of −0.0008 to +0.0020 |

Two test fixtures in `tests/test_tools.py` were real lyric fragments, written while
reading collocation output. They are replaced with stilted constructions each verified
absent from the corpus. If you add Chinese fixtures, verify them the same way.

`HANDOFF.md` section 2 carries the detail. Every tool states its own population in its
docstring. `paper/PENDING_MANUSCRIPT_INSERTS.md` now has all five inserts ready.

## Ordered plan

1. **Fix the three blocked analyses first**, on whichever branch they end up living.
   Reconciling before the content settles means re-deriving every manifest twice.
2. **Reconcile the branches** into one, resolving the 26 conflicts (regenerating, not hand-merging, the derived ones) and keeping one CI
   workflow. Manifest re-derivation and the three-mode clone verification come *after*
   the manuscript, derivatives and results have stopped changing, not before.
3. **Verify** with `python tools/verify_release_integrity.py`, then clone the result
   three times under `core.autocrlf` of `true`, `input`, and `false` and run it in
   each. All three must report the same count with zero failures. This is the
   property the whole integrity effort exists to guarantee; a merge that breaks it
   silently is worse than no merge.
4. **Paste `paper/PENDING_MANUSCRIPT_INSERTS.md` into `manuscript.md`** — five drafted
   passages covering extraction reliability, the exact-span mechanism, the
   multi-tagger limitation, the ablation, and the graph null — then rebuild the DOCX
   and PDF and delete that file. If the other branch's derivatives are being kept,
   these passages still need to go in and the derivatives rebuilt again.
5. **The staleness guard already exists** — `tools/check_manuscript_derivatives.py`
   with `paper/derivative_provenance.json`, run in CI. It fails when `manuscript.md`
   or `supplementary_methods.md` changes without the ten derivatives being rebuilt,
   which every individual checksum would otherwise happily validate. After rebuilding
   them, rerun it with `--record`. Do not run `--record` to silence a failure.
6. **Merge to `main`, tag `v4.0.0`, then Zenodo** — enable the repository in Zenodo
   *before* publishing the GitHub release, or no DOI is minted.
7. Optional: raise the label-permutation null from 200 to 2000 replicates (the p is
   currently floored by the replicate count; the informative number, the null mean,
   is already stable). Re-encode the ablated corpus with BGE-M3 and rerun
   `tools/entity_ablation_retrieval.py --dense-npy` to get the dense and fusion arms.

## Constraints

- **Do not validate the label-place edges against real artists' biographies.** The
  release explicitly disclaims biography, residence, and social relation. Checking
  黑麦–天津 against an artist's actual hometown is the exact inference the claim
  boundary forbids and turns a corpus statistic into an assertion about a person.
- **Do not rewrite git history to shrink the repository.** It is public.
- **Do not report model-assisted annotation as human validation.** Inter-method and
  inter-annotator reliability are different quantities; without a human anchor there
  is no way to estimate a model annotator's own error rate, so no precision or recall
  figure can come from one. `global_ner_benchmark.precision_recall_f1` must stay
  `WITHHELD` until the dual review is done.
- **Do not commit lyric text.** Everything under `analysis/` is aggregate counts; the
  longest contiguous Chinese run in any committed analysis file is five characters.
  Tools that print context windows write to paths outside the repository.
- **Do not push to `main` without the author's review.** These changes alter published
  checksums on a public research release.

## Environment notes

- Builders live in `src/` here and in `work/` in the private working tree; the
  ablation and null tools accept either.
- `torch` will not import in a venv created with `--system-site-packages` over a conda
  base — the conda MKL/OpenMP DLLs conflict. Use a clean venv, and put it at a short
  path or `pip install torch` hits the Windows long-path limit.
- The retrieval harness needs `numpy`, `scipy`, `scikit-learn`. It does not need
  `pandas`. The NER builder additionally needs `pandas`, `regex`, `torch`.
- Both the ablation and the null baseline self-check against a published figure before
  reporting anything (`0.415771` macro MRR, and 86 edges respectively). Keep those
  gates. The null's gate legitimately reports a one-edge drift from float32/float64
  arithmetic at the rank-five boundary and compares against its own observed value.
