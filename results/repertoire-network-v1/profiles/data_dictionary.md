# Public data dictionary

## source_label_profiles.json

- `id`, `label`: existing aggregate source-credit label identifier and string.
- `status`: `ready` or `limited evidence` under the declared support gates.
- `characteristicTerms`: stable corpus-distinctive terms with z score, song support, and leave-one-song stability.
- `lineEndings`: dictionary-estimated written final distribution and support.
- `formTraits`: corpus percentile, descriptor, raw aggregate, and song-bootstrap interval.
- `references`: intentionally empty until occurrence-level context annotation passes.

## stable_link_explanations.json

- `key`, `a`, `b`: exact stable graph edge key and endpoints.
- `dominantSignal`: strongest passing post-hoc signal or `semanticOnly`.
- `signals`: zero to three gated language, written-ending, or form signals.
- `calibration`: support stratum and within-stratum pair percentiles.
- `boundary`: interpretation limit repeated on every pair.

No field is a verified biographical, social, preference, audio, or performance claim.
