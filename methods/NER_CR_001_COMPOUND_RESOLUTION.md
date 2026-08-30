# NER-CR-001 — compound resolution for lexicon candidates

Status: **post-release protocol amendment, frozen before affected outcomes were
recomputed.** No downstream quantity has been recomputed under it, no manuscript passage
written from it, and the released NER outputs are unchanged.

An earlier draft numbered this PD-002. That number belongs to the upstream chunk
deduplication amendment; this is renumbered and the draft withdrawn.

## The defect

`boundary_ok()` in `src/build_chinese_rap_ner_cultural_graph_v1.py` returns `True`
unconditionally for surfaces with no ASCII character, which is every surface here, so the
lexicon stage cannot separate a surface from a longer compound containing it.
`nonoverlap_longest_first` only helps when the longer form is itself a lexicon entry, and
it usually is not: of 巴黎世家, 湖南卫视, 中文说唱, 加勒比海盗, 加州旅馆, 麦克风, 天津卫,
嘉陵江, 太平洋保险 and 香港脚, none is in the 605-entry lexicon, though 上海滩 is. The
release's own transformer agreed at `span_iou` 1.0 on every compound occurrence found
here, so nothing in the released artefacts signals the problem.

## The question each occurrence is asked

> Does the short surface, **at this occurrence**, refer to what its release schema type
> says, or is it only a substring of a different referent?

Three rules settle it, applied uniformly:

- A place name inside a **larger proper name** is not a mention of the place, even when
  the larger name is transparently derived from it and even when the thing named sits in
  that place. Brands, companies, clubs, stations, monuments, streets, universities,
  choirs, films, song titles.
- A place name inside a **fuller co-referential form of the same place** is a mention: a
  city suffix, a province-plus-city administrative form, a traditional or colloquial name,
  an affectionate intensifier, a bilingual gloss.
- A place name used as an **external locative modifier** on a name that stands without it
  is a mention: a museum or a venue identified by the city it sits in.

A tagger span that merged across a line break, a language boundary or a list of separate
names is an artefact; the surface is then judged on its own.

## Flag stage — automatic, high recall, decides nothing

Frame, fixed before flagging: `candidate_source == LEXICON_WITH_TRANSFORMER_CHECK`,
`cross_label_shared_cleaned_text == False`, `source_credit_label` in the 204-label graph
universe, `candidate_surface` in the released inventory. **1,098 candidates**, reproducing
the published `lexicon_candidate_occurrences` for **22 of 22** surfaces.

A candidate is flagged when either pinned reference tagger proposes an entity span
strictly containing it. **91 flagged occurrences → 63 combinations.**

The winning span is chosen deterministically: longest, then leftmost, then shortest, then
by tagger label, then by model name. Each occurrence records the models that proposed the
winning span (`proposing_models`) and the models that proposed any containing span
(`flagging_models`).

**20 of 63 combinations were flagged by both taggers; none had the same winning span from
both.** The flag stage is a union of two disagreeing views, not a consensus.

## Adjudication — blinded, one verdict per occurrence

Every one of the **91 occurrences** was adjudicated separately against its own context.
The adjudicator sees a random blind identifier, the surface, its schema type, the
containing span, the tagger's label, and the context. It does **not** see the
strict-consistency flag, the source credit label, the song identifier, any occurrence or
label count, or any earlier decision, and **no two occurrences of one combination share a
batch** — earlier batch assignments violated this (one five-occurrence combination sat
three-in-one-batch), so the whole set was re-batched under that constraint and every
occurrence re-adjudicated, not only the affected ones.

The instructions, including the escalation procedure, are frozen as one file and hashed.
Every record whose first ruling was not high-confidence, or was DEFER, received two
further rulings; the recorded verdict is the majority of three. In the frozen run, **4
records escalated** and one resolved 2–1. The rulings are **AI-assisted repeated rulings
from separate model invocations; independence between sessions is procedural, not
demonstrated**, and they are recorded as such.

All raw ballots are retained and hashed; `tools/rekey_blinded_ballots.py` — a script,
not a judgement — validates the ballot set against the frozen procedure, maps blind
identifiers back to canonical occurrence uids through a separately frozen map, and folds
the majority. The frozen provenance set (instructions, blinded input, batch manifest,
blind-id map, ballot manifest) is pinned by hash in the freeze record and in the offline
gate.

Blinding was not free, and its effect is recorded rather than assumed: removing the one
leaked pipeline flag changed one verdict of 91, and the batch-isolation re-run changed
one more — a landmark name that resolved 2–1 on escalation. Both margins are visible in
the retained ballots.

## Two orthogonal judgements, never one variable

`span_relation` is the relation of the longer span to the surface. `text_region` is
whether the occurrence sits in lyric text or in a production-credit block. They are
adjudicated separately, folded separately, and published separately.

An earlier version folded them into one field. The result was 8 credit-block occurrences
carrying a non-credit label, one lyric occurrence carrying a credit label, and a public
row whose rationale described its occurrences as credits while its own count said
otherwise.

## Actions

The action set decides one thing: whether the short mention stands.

| action | retain short | rows | occ | strict |
| --- | --- | ---: | ---: | ---: |
| `RETAIN_SHORT` | yes | 33 | 42 | 21 |
| `DROP_SHORT` | no | 30 | 49 | 18 |
| `DEFER` | withheld | 0 | 0 | 0 |
| `SPLIT_NOT_FOLDED` | split | 0 | 0 | 0 |
| | | **63** | **91** | **39** |

It does **not** emit the longer span as an entity. An earlier version did, under the
reference tagger's own label — `address`, `company`, `movie`, `scene` — which is neither
this project's schema nor a defensible type assignment, and let one span acquire four
types. `longer_span_disposition` follows from `span_relation` alone:

| disposition | rows |
| --- | ---: |
| `NEW_CANDIDATE_REQUIRES_STANDARD_GATE` | 38 |
| `NOT_A_CANDIDATE_TAGGER_ARTEFACT` | 13 |
| `NOT_A_CANDIDATE_SAME_REFERENT` | 11 |
| `NOT_ASSESSED_RELATION_MIXED` | 1 |

One combination's occurrences agree on the verdict but were assigned different span
relations; that row publishes `MIXED` and `NOT_ASSESSED_RELATION_MIXED` rather than a
modal value.

Nothing here publishes a new entity. A span marked a new candidate must enter the standard
candidate pipeline and pass the normal release gate before anything is published about it.

**21 of the 91 occurrences sit in production-credit blocks rather than lyric text**, which
is an upstream text-cleaning problem, not an NER problem.

## Adjudication provenance

Blinded AI per-occurrence contextual adjudication by `claude-opus-5` under frozen
instructions, consulting external sources for institutional referents.
**`independent_human_review_status: pending`.** No human review has been completed. An
earlier revision of this file described an author spot-check as partial human review; that
claim is withdrawn. Comments made in conversation, an author's observations about corpus
cleaning, and this model's own re-reading of its output are none of them an adjudication,
and attributing them to a reviewer overstated the evidence.

Human review, when it happens, requires an append-only per-occurrence ledger carrying the
full occurrence uid, the sampling scheme, each judgement, its timestamp and the reviewer's
signature, with that ledger's SHA-256 bound into this freeze. Until such a ledger exists,
this adjudication is AI-only.

This is not human gold, not a pre-specified analysis, and not an NER accuracy estimate.
NER precision, recall and F1 remain **WITHHELD**.

## Public and private tiers

The public table carries **no verbatim occurrence context, no song locator and no private
identifier**. It is not context-free in any stronger sense: the entity names that carry
the boundary decision are published, and they occur in the corpus.

- `combination_id` is a **frozen opaque UUID4** drawn from a private map. A truncated hash
  of the private key, which an earlier version published, can be recomputed by anyone who
  guesses the key, so it confirmed guessed organisation names instead of withholding them.
- A span is published verbatim only if it appears in `public_name_allowlist.json` with
  `publish_verbatim` true. That list is a **disclosure-safety gate only**: three
  AI-assisted repeated rulings, unanimity required, on whether the STRING is a complete,
  conventional public name that is safe to print. It does not validate what any
  occurrence refers to and establishes no entity boundary in any context — contextual
  reference is decided per occurrence in the adjudication ledger, never there. **4 of 14
  candidate spans failed the ruling** and are generalised instead. An earlier version
  used a contiguous-Han regular expression, which cannot tell a complete name from a
  one-character-short tagger boundary artefact, and published one: a three-character
  prefix of a four-character brand whose full form appears on four other rows of the
  same table. The allowlist's raw ballots are retained and their manifest hash recorded
  inside the list itself.
- Production-credit organisations and work titles become `<PRODUCTION_CREDIT_COMPANY>`,
  `<PRODUCTION_CREDIT_CHOIR>`, `<PERSONNEL_AFFILIATION_ORG>`, `<PUBLISHING_CREDIT_ORG>`,
  `<SAMPLE_SOURCE_WORK_TITLE>`. That map is keyed by hash and kept private: even a
  hash-keyed map in the repository would let a guessed name be confirmed.
- Everything else keeps only the target surface, with every other run replaced by
  `<OTHER_PLACE>`, `<HAN>`, `<LATIN_RUN>` or `<BREAK>`, and the exact span length replaced
  by a bucket. Line breaks are detected on their own; 6 rows carry one.
- **Rationales are generated from a fixed phrase table** keyed on the action and the span
  relation. Free text written while reading an occurrence is a fingerprint even after the
  Chinese is removed.

## Reproducing and gating

```
python tools/verify_compound_resolution.py                        # offline, in CI
python tests/test_compound_resolution_gate.py                     # tries to defeat it
python tests/test_release_boundary.py                             # repository checkout only
python tools/build_compound_resolution_table.py --check ...       # private inputs, both models
python tools/publish_compound_resolution.py --check ...           # re-derives the public table
```

The first two run inside the desktop package as well as in a checkout. The third builds
and tampers with real packages, so it needs a clean repository working tree and is not
shipped in a form that runs from inside the package.

The offline gate compares against values hardcoded **in the gate**, never read from the
artefacts under test, including a second copy of the rationale phrase table. It pins the
whole `resolution_table.csv`, the whole canonical `freeze.json` and the whole allowlist by
hash, so a field the gate does not individually understand cannot be edited either. It
recomputes the totals, the per-action counts, the production-credit occurrence count and
the not-folded and mixed-state counts from the table and compares them against both the
pinned values and the record. It checks the column whitelist, that every id is an opaque
UUID4 and unique, non-negative counts, `strict ≤ occurrences`, `credit ≤ occurrences`,
`labels ≤ occurrences`, that `retain_short` follows the action and the disposition follows
the relation, that a credit-region row accounts for all its occurrences and a lyric-region
row for none, that every proposing model is a pinned reference model, and every redaction
rule. It hashes the committed graph universe and released inventory on disk rather than
trusting the record.

The publisher fails closed: the credits map, the id map, the freeze inputs and the
allowlist are required and pinned by whole-file hash; the derived table AND the composed
canonical freeze record are compared against their pins **before** anything is written;
and the two outputs are replaced with rollback, so a failure part-way restores what was
already replaced rather than leaving the table and the record disagreeing.

The builder pins and compares the flag table, the occurrence ledger, the fold state
(rendered in memory on every run, so the comparison does not depend on writing the
private directory) and the adjudication file, and checks the adjudication's structure:
the embedded uid equals the key, votes are legal and number one or three, and the
recorded verdict is the majority of its votes. The frozen blinded-provenance hashes are
emitted into the freeze inputs and pinned in the offline gate.

Reference models, pinned by revision:

- `shibing624/bert4ner-base-chinese` at `5d660ed2aa9da482bf2d99c6bc8cf2ce66758f6a`
- `uer/roberta-base-finetuned-cluener2020-chinese` at `cddd8fc233e373855a8c0a7f4b7eb83acb686a2b`

## Release boundary

The desktop package is built from **committed bytes**: the builder refuses to run over a
working tree that differs from HEAD, verifies every copied file against its HEAD blob
sha1, and records a `SOURCE_PROVENANCE.json` binding every packaged file to the
committed blob it came from. The validator re-derives that binding from git itself, so
editing a packaged file and regenerating the package's own manifest cannot pass — the
authority is the repository, not anything the package carries. Merely checking that a
path was tracked, as an earlier version did, still packaged working-tree bytes, so an
unstaged edit to a tracked methods file reached the package and validated.

The final archive checksum is pinned in the committed `validation/desktop_zip.sha256`,
outside the archive and outside every generated manifest, and the archive is
byte-reproducible, so CI compares the built ZIP against a committed expectation.

Only tools that actually run inside the package are shipped in it:
`Reproducibility/tools/verify_compound_resolution.py`,
`Reproducibility/tests/test_compound_resolution_gate.py` and
`Reproducibility/tests/test_tools.py` (which skips its repository-only suite there). The
repository-only checksum verifier and the package-attacking boundary suite are not
packaged; the checksum verifier fails closed if it finds zero claims to verify, and the
repository validator refuses with a clear message when run outside a checkout.

## Sequencing — this amendment is provisional

Twenty-one of the 91 occurrences are not lyrics. They are production credit blocks inside
`analysis_text`. The cause is upstream — text cleaning does not strip metadata blocks, the
same cause that produced the performer annotations found earlier. Two further occurrences
share the same lyric content and character span across different chunks, which is the
upstream duplication.

This must not be patched row by row through the NER tables. The correct order is:

1. complete the upstream chunk deduplication amendment
2. run a pre-specified corpus-wide metadata-block detection audit and report its scope
3. decide and apply a global stripping policy
4. rebuild the NER candidate frame
5. re-run this compound resolution on the rebuilt frame
6. only then re-run the cultural-reference network

**This freeze will therefore be superseded.** It is published now so the method and the
adjudications can be reviewed against evidence rather than against a summary.

## What this addresses, and what it does not

| failure mode | example | addressed |
| --- | --- | --- |
| boundary / compound ambiguity | a province name inside a broadcaster name | yes |
| entity-type mismatch in the lexicon | a product type on a surface whose compound is a common noun | no |
| lexical-sense ambiguity | a classical name used as a boastful first person | no |
| reference-tagger coverage limits | no tagger has a language class, so 0 of 123 language compounds flagged | no |
