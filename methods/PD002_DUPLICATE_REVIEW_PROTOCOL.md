# PD-002 duplicate-record review protocol

## Purpose and scope

PD-002 established that the legacy cleaner erased 177 song records outright. Of those, 131 satisfy a conservative automatic rule and 46 do not. This protocol governs how those 46 are adjudicated. It covers nothing else: it does not decide what the corpus contains, which is settled by the replacement rule in `src/build_repaired_corpus_v2.py`, and it does not decide anything about the 131.

The adjudication is not required for corpus v2 to exist. Corpus v2 already retains all 46 records, because the replacement rule deletes nothing. The adjudication decides only how those records are *weighted and reported* in the duplicate-collapse sensitivity, and until it exists the sensitivity is reported with the queue withheld.

## What is being asked

One question per record: are these two (occasionally three or four) song records the same recorded work?

Three answers are admissible. `same` means the records are repeated ingestion of one recording, including reissues and retitled imports. `different` means they are not one recording; identical or overlapping lyrics are compatible with this, since a cover, a remix, or a hook reused across two songs all produce shared text. `cannot_tell` is a full answer and is recorded as such — the queue exists because an automatic rule could not decide, and a forced choice would manufacture precision that is not there.

## Admissible evidence

The rater sees the source-credit label, the title, and the cleaned chunk text of each record, and rules on those alone.

Biographical evidence is inadmissible. An artist's birthplace, hometown, residence, or social relationships may not be used, here or anywhere else in this project, to decide a question about the corpus. Prior impressions of an artist and any source outside the review sheet are likewise inadmissible.

## Blinding

`tools/build_duplicate_review_sheet.py` renders the queue with two facts withheld from the rater.

The rater is not told which record the legacy cleaner erased and which it retained. The sides of each comparison are shuffled under a fixed presentation seed, so the queued record appears first in roughly half the cards and the ordering is reproducible without being informative.

The rater is not shown the automatic reason code, nor whether the v2 primary rule groups the record. A ruling that merely ratifies the classifier would add nothing; withholding the verdict makes the ruling independent of it. The mapping from review ID to reason code is written to a separate key file at generation time, so the rulings can be analysed against the automatic classification afterwards without having been anchored to it beforehand.

Each card does carry a plain structural readout of how the paragraph sets line up -- identical sequence, same set reordered, one contained in the other, or a multi-way overlap with counts. This is a diff of the two panels the rater is already reading, and stating it saves the labour of doing that diff by hand. It is recorded here because it is not costless: the queue's reason codes partition it almost exactly, so a rater who knows this protocol can infer the category. What the readout does not supply is the thing being asked. It says how the text overlaps; it does not say whether that overlap means one recording or two, which is the entire question and the reason these 46 records could not be resolved automatically.

The paragraphs the records have in common are marked in both panels, and where the whole overlap is a single line the readout says so, including when that line restates a title. This was added after inspecting the queue: ten of the forty-six cards overlap in nothing but one title-restating line, which is a header artefact of the source export rather than a shared verse, and a rater skimming two long panels would not reliably notice that the common text is one line long. Marking it is a description of the panels. Whether a shared header means one recording or two is still the rater's call, and it is the call that decides the record.

Changing the instruction text or the structural readout changes the frozen instruction hash. Rulings are only ever scored against the hash they were collected under; a sheet regenerated with different wording invalidates nothing already returned, it simply belongs to a different frozen question.

## A second fact the pair question cannot carry

A reviewer working the queue observed that on some cards the portion of a record which is *not* shared with the other record is itself another song's content. Checking that against the corpus confirmed it: on 29 of the 46 cards at least one paragraph also belongs to a song that is not displayed on the card, because the card renders only the records the automatic rule related while paragraph ownership is a property of the whole corpus. A card can therefore present a paragraph as unique to one of two records when it is not.

Two changes follow, and neither moves a judgement from the rater to the tool.

Every paragraph now states how many songs off the card also carry it, and the structural line says when some of the apparently unique paragraphs are not unique. This is a count over the corpus, not a claim about what it means.

A checkbox records "one of these records has another song's content mixed into it", alongside and independent of the three-way ruling. It is a separate fact with a separate consequence: a contaminated record is not merged with its neighbour and not left as a distinct work, it is split and the foreign paragraph removed. Folding that into `same` or `different` would have destroyed the distinction, and the three-way question alone could not express it. The ruling still gates completion; the checkbox is additional information and never a substitute for the ruling.

This was found by a human reading the cards, after the automatic classifier, the detector, and two of my own structural hypotheses had all missed it.

## Recording

The instruction text is frozen and hashed. The sheet, the key file, and any returned rulings all carry that hash, so a ruling can be tied to the exact question that was asked. The key file also carries the content digest of the repaired corpus and the digest of the review queue it was generated from; a ruling collected against one corpus state cannot be silently reused against another.

Rulings are recorded as returned, including notes and `cannot_tell` answers. Partial returns are valid and are recorded as partial.

## Attribution

Rulings collected under this protocol are author adjudications and are attributed as such. They are one rater.

They are not an independent human review, they do not establish inter-rater reliability, and they may not be reported as either. Nothing produced in conversation with an AI assistant — a spot-check, a comment on an example, a summary agreed to in passing — is a ruling under this protocol; only a ruling recorded against the frozen instruction hash is.

`independent_human_review_status` in the corpus v2 artifact stays `pending` regardless of how many of the 46 are ruled on. It changes only if and when a genuine second independent review exists.

## What a completed adjudication does and does not license

It licenses reporting the duplicate-collapse sensitivity with the queue included, under the declared strata, attributed to a single author rater.

It does not establish real-world work identity, reissue status, authorship, or performer identity for any record. It does not license retiring the review queue from the published record, and it does not license any predictive claim: no model has been retrained on the repaired population, so top-k accuracy, MRR, calibration, abstention, and paired model differences remain untested there.
