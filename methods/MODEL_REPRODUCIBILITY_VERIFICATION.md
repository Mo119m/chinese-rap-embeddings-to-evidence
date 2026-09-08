# Model reproducibility verification

## Why this exists

Three artifacts carry the paper's headline model numbers: retrieval macro MRR and Recall@10, written-ending Top-3, and the repertoire graph's edge count. Until 31 August 2026 none of them had ever been rebuilt from its inputs.

Nor could a reader have checked. Unlike `results/corpus-reconciliation-v1/` and `results/repaired-corpus-v2/`, the model artifacts record neither their input hashes nor the code that produced them. There was no fingerprint block to go stale, and so nothing to verify against — the gap was invisible rather than flagged.

`tools/verify_model_reproducibility.py` closes that by making the rebuild a procedure anyone with the private corpus can rerun.

## What the procedure does

Each artifact is rebuilt in a sandbox and compared against the committed copy, file bytes first and then every leaf value of the summary.

The sandbox is the part that needed care. Two of the builders write beside their inputs, so pointing them at the private tree makes a verification run a write into the author's data. Both refused to clear a directory they did not expect, which is how this was noticed rather than discovered afterwards. The sandbox now links the private inputs read-only and gives every build target an ordinary directory inside itself, so a build cannot reach the private tree even if a guard were missing.

Differences are classified, not counted. A timestamp differs on every rebuild and means nothing. A recorded builder hash that names a superseded version means the artifact predates a code change and that change must be examined. Anything else is a real divergence.

## Result, 31 August 2026

| artifact | files identical | leaf values | differing |
| --- | ---: | ---: | ---: |
| repertoire-network robustness | 8 of 8 | 47 | 0 |
| retrieval | 7 of 10 | 136 | 2 |
| written-rhyme | 10 of 15 | 521 | 1 |

**All three reproduce.** Every published metric — MRR 0.447, Recall@10 0.611, Top-3 0.695, 86 edges across 93 labels — recomputed to the same value from the committed builders and the private inputs.

Two differences beyond timestamps, both resolved.

**Retrieval records a superseded builder hash.** The artifact's `lineage.builder_code_sha256` is `340564…`, the version at `cf8c2fa`; the builder has been `7ccb3fa…` since `6b77c95`. That commit changed exactly two things, `utf-8-sig` to `utf-8` and an explicit LF line terminator, so it changed CSV bytes and no value. The rebuilt CSVs are byte-identical to the committed ones, which settles it: the results follow from the current code and the recorded hash simply names an older version of it.

**Written-rhyme was missing a withholding statement from its builder.** The committed artifact carries `repaired_population_predictive_metrics_retrained: false` and a `repaired_population_predictive_claim` of *WITHHELD pending duplicate-aware corpus reconstruction … under PD-002*. The builder emitted neither. That safeguard existed only as bytes on disk: rebuilding the artifact dropped it silently, and nothing in the repository would have noticed. It now lives in the builder, and a rebuild reproduces 520 of 521 leaf values with only the timestamp differing.

## What is still open

The model artifacts still record no input or software fingerprints. This document and the tool substitute for that, but they are not equivalent: a fingerprint block travels with the artifact, whereas a procedure has to be rerun.

The natural moment to add fingerprints is the rebuild these artifacts are already scheduled for — step 5 of the PD-002 sequencing, on the repaired population. Adding them now would mean rebuilding published results, and a rebuild would stamp today's date onto an analysis that ran on 25 August 2026. That is a worse defect than the one it fixes, so it was not done.

## Running it

    python tools/verify_model_reproducibility.py \
        --private-root <directory containing work/ and outputs/> \
        --sandbox <a directory outside the repository> \
        --report verification.json

Needs the private corpus, the model dependencies in `requirements.txt`, and roughly half an hour. It is not in CI for those reasons. Rerun it whenever a model builder changes or a published number is questioned.
