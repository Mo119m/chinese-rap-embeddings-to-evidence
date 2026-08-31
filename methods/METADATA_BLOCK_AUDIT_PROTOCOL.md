# MB-001 — pilot-informed metadata-block CANDIDATE detector

Status: **executed once against a single-rater gold set; still not a source of truth.** See the amendment at the end of this file, which supersedes the original status line and the prohibitions that went with it. Everything the amendment does not change stands.

This is a **candidate detector**, not a metadata ground truth. Its rules were written from
a pilot look at a handful of real credit blocks, so they are informed by data they were
then measured against — which means any figure computed from them is a candidate count,
not an accuracy. **No VALIDATED recall, precision or F-measure is claimed anywhere**, and none may be
claimed until an independently constructed, multi-rater gold set exists. A single-rater stratified estimate is now published under the amendment below and is not a validated figure. An earlier revision of this
file said the detector "implements exactly" the rules and reported a recall figure; both
claims are withdrawn. This protocol is step 2 of the
sequencing declared in `NER_CR_001_COMPOUND_RESOLUTION.md`. It runs only after the
upstream chunk-deduplication amendment (PD-002, on `codex/release-integrity-publication`)
has landed, and its detector is committed here, with its self-tests, so the rules cannot
drift between specification and execution.

## Why

The NER-CR-001 adjudication found that **21 of 91 flagged occurrences sit in
production-credit blocks rather than lyric text** — publishing credits, personnel
affiliations, choirs, universities, a sample attribution. Earlier work found performer
annotations surviving cleaning. Both have one upstream cause: text cleaning does not
strip metadata blocks from `analysis_text`. The corpus-wide scope of that cause is
unknown, and until it is measured no global stripping policy can be justified.

## The author's instruction

The project author, reviewing real occurrence contexts, stated the requirement directly:
NER was run over production-credit text as though it were lyrics, and those blocks should
be **removed from the input**, not merely labelled. This audit measures how much there is
to remove; step 3 decides the removal policy. A provisional pre-deduplication scoping run
has been performed and is held privately under the sequencing guard below — its figures
are not the audit result and are not published, because the deduplication repair changes
the population they are measured over.

## Question

For every song unit in the deduplicated corpus: which character spans of its
`analysis_text` are metadata (credits, personnel, publishing, copyright, tracklist,
annotation) rather than lyric text, and what fraction of text and of NER candidate
occurrences fall inside them?

## Frozen decision rules

`tools/detect_metadata_blocks.py` implements these rules in priority order, and the code
is the specification: where prose and code differ, the code is what ran. Several rules were
revised during a pilot pass over real credit blocks, so they are **fitted to examples they
were then tested on** and their apparent agreement with those examples is not evidence of
accuracy. A LINE is the unit of classification; a BLOCK is a maximal run of consecutive
metadata lines.

1. **Role-prefix line**: an optional bracketed tag, then a role term, then optionally
   further role terms and Latin label text (so a chained bilingual label such as
   `企划统筹A&R COORDINATOR：` is caught in full), then a separator (：, :, ／, /, or
   two-plus spaces), then content. Han text that is not itself a role term ends the run,
   which keeps lyric lines that merely begin with a role word out. The role lexicon is frozen in the
   detector: 作词, 作曲, 编曲, 制作, 制作人, 监制, 出品, 出品人, 出品方, 发行, 发行方,
   录音, 录音师, 混音, 混音师, 母带, 母带工程师, 和声, 和音, 吉他, 贝斯, 鼓, 键盘, 弦乐,
   合唱, 合唱团, 伴唱, 封面, 封面设计, 视觉, 企划, 统筹, 宣推, 营销, 版权, 词, 曲, OP, SP,
   Producer, Prod, Beat, Mix, Master, Mastering, Arranged, Arranger, Composer, Lyrics,
   Lyricist, Vocal, Backing, Artwork, Design, Label, Publishing.
2. **Sample attribution line**: begins with `sample from`, `sampled from`, `samples:`,
   `interpolates`, 采样自, 采样于, 采样：, or 取样自. Author review of the NER-CR-001
   adjudication found one such line labelled lyric text; it names a borrowed work and its
   performer, so it is a credit.
3. **Copyright line**: contains ©, ℗, 版权所有, 保留所有权利, All Rights Reserved, or
   未经许可 followed within the line by 使用/复制/翻录.
4. **Organisation-registration line**: contains 有限公司, 传媒, 文化传播, 唱片, 工作室,
   娱乐, 影业, or 集团 AND (a separator structure from rule 1 OR the line is ≤ 30
   characters with no sentence-final punctuation 。？！).
5. **Bracketed annotation line**: the entire line, whitespace-trimmed, is enclosed in
   （）, (), 【】, or [] and contains no rhyme-bearing final particle — this catches
   (女声), 【副歌】, (Hook) style annotations.
6. **Block extension**, in two bounded forms. A name-only line carries no verb, is ≤ 20
   characters, and is otherwise name-like.
   - *bridge*: a run of name-only lines flanked on **both** sides by rule hits joins the
     block. This is the personnel list between two credit labels.
   - *tail*: a run of name-only lines adjacent to a block with ≥ 2 rule hits joins it
     **only when the run reaches the start or the end of the text**.

   The boundary condition is deliberate and conservative. Extending into the middle of a
   text swallowed the first lyric line after a credits header, because a short Latin
   refrain is indistinguishable from a personnel name by shape alone. A single name-only
   line sitting between a credits header and the lyrics is therefore left classified as
   lyric: this audit measures scope, and over-detection would inflate the estimate of
   what a later stripping policy should remove.

A line matching none of these is lyric text. **No model, no lexicon lookup against the
entity inventory, and no per-case judgement is involved** — the audit must be mechanical
so that running it twice is running the same audit.

## What must exist before this is run for record

None of these is in place yet. The protocol is committed now so the rules cannot drift;
running it for record requires all of the following, and the tool refuses to emit a
non-provisional summary without them:

- the PD-002 repaired-corpus manifest and its hash, bound automatically into the summary
- a non-empty input and a schema check on every song-unit record read
- character spans for every detected block, and a mapping from those spans to both the
  1,098-candidate frame and the full private candidate table
- a reproducible worksheet: fixed random seed, recorded sampling frame, regenerable from
  the committed detector and the corpus hash alone
- a pre-frozen positive AND negative sample, drawn before any figure is computed, with a
  two-reviewer scheme declared in advance, for the independent gold set that any accuracy
  statement would need

## Outputs (aggregate only; nothing per-song leaves the private tier)

- corpus totals: song units, lines, characters; candidate metadata lines, characters, blocks
- character spans of every candidate block, and their mapping onto both candidate frames
- per-rule hit counts
- fraction of NER candidate occurrences (the 1,098-candidate frame and the full private
  candidate table separately) whose context midpoint falls inside a detected block
- the distribution of metadata-block position (leading / trailing / interior)
- a stratified 40-block sample worksheet (private) for human spot-verification of the
  detector's precision, with the same agree/disagree/unsure format as earlier
  spot-checks

Public artefact: one `results/metadata-audit-v1/summary.json` with the aggregates, the
detector version hash, and the corpus content hash it ran over. No lyric text, no song
identifiers.

## What the audit does NOT decide

It measures scope. The stripping policy (step 3) is a separate decision taken after the
numbers exist, and the candidate-frame rebuild (step 4), the compound-resolution re-run
(step 5) and the network re-run (step 6) follow it in that order. Nothing in this audit
edits the corpus.

## Sequencing guard

The detector refuses to produce the public summary unless it is told, explicitly and
truthfully, whether PD-002 deduplication has been applied to its input
(`--dedup-state applied|not-applied`), and stamps that state into the summary. A
pre-dedup run is permitted for private scoping but its summary is marked
`provisional_pre_dedup` and must not be published as the audit result.

## Gold-set labelling (MB-001-GOLD-001)

The detector carries no precision, recall, or F-measure because nothing independent has labelled what a metadata line is. `tools/build_metadata_gold_sheet.py` renders a sample of the repaired corpus for labelling, blind: the rater is not shown which lines fired, which stratum a chunk came from, or any rule name.

The sample is stratified by detector output -- half from chunks where at least one line fires, half from chunks where none does -- because a uniform sample of a corpus this size contains too few positives to estimate precision. That makes the sample non-representative by construction. The per-stratum populations and draw sizes are written into the key file, and every estimate computed from these labels must be weighted by the sampling probabilities and reported as stratified. It is not a random sample of the corpus and may not be described as one.

Every sampled chunk requires an affirmative answer, including "no metadata line here". A chunk left untouched is an unread chunk, not a negative, and is excluded rather than counted as clean; recall is estimated from the affirmative negatives, so the distinction is not cosmetic.

Labels collected this way are author labels from a single rater. They license a stratified precision and recall estimate attributed to one rater. They do not establish inter-rater reliability, they are not an independent human review, and no F-measure may be reported as validated until a genuine second independent labelling exists.

## Amendment MB-001-A, 30 August 2026: single-rater execution

This amendment records that the protocol was executed, and reconciles the result with the prohibitions written above. It is recorded after the fact and does not pretend the execution was anticipated in this form.

**What changed.** The original status line said "pre-specified, not executed", and the paragraph beneath it said no recall, precision or F-measure is claimed anywhere. A gold set has now been collected under `MB-001-GOLD-001` and scored, and the artifact in `results/metadata-block-gold-v1/` publishes line precision and recall, a chunk-level detection rate and flag precision, and a share of lines labelled metadata. Those sentences and this artifact could not both stand, and leaving the contradiction in a file that ships inside the public package would have handed a reader "not executed" and a precision figure in the same download.

**What is now permitted.** Single-rater stratified estimates over the declared frame, each published with its measured interval coverage, its bootstrap bias and spread, and the count of clusters actually carrying signal.

**What remains forbidden, unchanged.** A validated precision, recall or F-measure. Any inter-rater reliability figure. Any accuracy statement — the original text made a two-reviewer scheme a precondition for that, and one rater does not meet it, so no accuracy metric appears in the artifact at all. Any claim that detector output is metadata truth. `independent_human_review_status` stays `pending`.

**The eligibility filter was never specified here and should have been.** The sample was drawn only from chunks of 2 to 14 lines. That filter existed only as a command-line default in `tools/build_metadata_gold_sheet.py`. It matters more than a default should: those chunks are about half the corpus's chunks but only ~13.5% of its lines, and short chunks are where whole credit blocks live, so the frame is enriched in the very shape being counted. Every rate is a rate over that frame. A corpus-wide contamination rate is **not** estimated and may not be quoted from this artifact.

**The intervals are not 95% intervals.** Most of the recall denominator rests on four chunks out of forty. A percentile bootstrap on a combined ratio in that situation undercovers: measured coverage is about 0.80 for recall, F1 and the metadata line share, and about 0.90 for the detection rate, against a nominal 0.95. Only precision, which is a within-stratum quantity, holds up at about 0.94. The tool measures this by simulation on every run and publishes the measured figure; the word "95%" does not appear on any interval.

**Recall, F1 and the detection rate are biased upward** by roughly a fifth to a quarter of their own size, as combined-ratio estimators on few informative clusters. The bias is published beside each interval and is deliberately not subtracted.

**The labels are not adjudicated.** Two false-positive lines match categories the frozen instruction text itself names as non-lyric. They are reported as conflicts and the labels stand exactly as returned; only the rater may revise their own labels. Published precision is therefore a lower bound, with the bound published alongside. The false negatives have not been adjudicated at all.

**Not deviations.** The `results/metadata-audit-v1/summary.json` artifact and the 40-block precision worksheet named elsewhere in this file are a different, still-unproduced deliverable. They are not superseded by this amendment and are not recorded as deviations from it.
