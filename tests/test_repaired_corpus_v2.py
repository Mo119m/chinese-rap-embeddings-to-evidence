"""Checks for the PD-002 duplicate-control rules.

These rules decide which song records get called duplicates of each other, which records
share text and therefore may not be split across evaluation partitions, and which records a
human still has to rule on. All three are easy to get wrong in a way that looks right in the
aggregate: an order-insensitive comparison quietly merges a song with its own remix, a
within-song hook accidentally unions a song with itself, a weighting rule sums to something
other than one, or an automatic rule silently empties the review queue it was supposed to
feed. Each of those is a case below with an answer worked out by hand.

Fixtures are synthetic ASCII, so no corpus text can reach a public file through this suite.
The one exception is the title-normalisation case, which needs characters that NFKC actually
folds; those are punctuation and full-width Latin, not lyric text.

No private data, no third-party dependencies, no test framework:

    python tests/test_repaired_corpus_v2.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dc = load("duplicate_control_v2")


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{': ' + detail if detail else ''}")
        FAILURES.append(label)


def rows(*specs: tuple[str, str, str, str]) -> list[dict[str, object]]:
    """(song_id, label, title, text) tuples numbered in the order given."""
    return [
        {"song_id": song, "artist": label, "song_title": title, "text": text, "source_row": index}
        for index, (song, label, title, text) in enumerate(specs)
    ]


# ------------------------------------------------------------------- text-level corrections

def test_strip_del() -> None:
    print("strip_del")
    delete = chr(0x7F)
    check("removes the control and keeps its neighbours", dc.strip_del(f"ab{delete}cd") == "abcd")
    check("removes every occurrence", dc.strip_del(f"{delete}a{delete}b{delete}") == "ab")
    check("leaves text without controls alone", dc.strip_del("abc") == "abc")
    # the live sheet drops the preceding character too; corpus v2 deliberately does not
    check("does not copy the destructive sheet import", dc.strip_del(f"ab{delete}c") != "ac")


def test_normalise_title() -> None:
    print("normalise_title")
    check("case-folds", dc.normalise_title("Same Song") == dc.normalise_title("same song"))
    check("drops spaces and punctuation", dc.normalise_title("a b-c!") == "abc")
    check("NFKC folds full-width Latin", dc.normalise_title("ＡＢ") == "ab")
    check("NFKC folds full-width digits", dc.normalise_title("１２") == "12")
    check("keeps genuinely different titles apart",
          dc.normalise_title("song one") != dc.normalise_title("song two"))
    check("empty and punctuation-only collapse together",
          dc.normalise_title("") == dc.normalise_title("---") == "")
    check("tolerates a missing title", dc.normalise_title(None) == "")


# ------------------------------------------------------------------------- record assembly

def test_build_song_records() -> None:
    print("build_song_records")
    # deliberately fed out of order, and the second song interleaved with the first
    unordered = [
        {"song_id": "s1", "artist": "L", "song_title": "T1", "text": "b", "source_row": 1},
        {"song_id": "s2", "artist": "L", "song_title": "T2", "text": "z", "source_row": 3},
        {"song_id": "s1", "artist": "L", "song_title": "T1", "text": "a", "source_row": 0},
        {"song_id": "s1", "artist": "L", "song_title": "T1", "text": "b", "source_row": 2},
    ]
    records = dc.build_song_records(unordered)
    check("sequence follows source order, not input order",
          records["s1"]["sequence"] == ("a", "b", "b"), str(records["s1"]["sequence"]))
    check("within-song repetition is kept in the sequence", records["s1"]["chunk_rows"] == 3)
    check("counter reflects the repetition", records["s1"]["counter"]["b"] == 2)
    check("first_source_row is the earliest row", records["s1"]["first_source_row"] == 0)
    check("each song is one record", sorted(records) == ["s1", "s2"])


# ---------------------------------------------------------------------- duplicate grouping

def test_duplicate_groups() -> None:
    print("assign_duplicate_groups")

    def group(specs):
        records = dc.build_song_records(rows(*specs))
        return records, dc.assign_duplicate_groups(records)

    records, (group_of, is_rep, membership) = group([
        ("s1", "L", "Title", "a"),
        ("s2", "L", "title", "a"),
    ])
    check("same label, same folded title, same sequence groups", len(membership) == 1)
    check("the earlier record represents the group", is_rep["s1"] and not is_rep["s2"])
    check("both members stay addressable", set(group_of) == {"s1", "s2"})

    _, (_, _, membership) = group([("s1", "L1", "Title", "a"), ("s2", "L2", "Title", "a")])
    check("a different source-credit label does not group", membership == {})

    _, (_, _, membership) = group([("s1", "L", "One", "a"), ("s2", "L", "Two", "a")])
    check("a different title does not group", membership == {})

    # the same chunks in a different order are a different arrangement, not a duplicate import
    records = dc.build_song_records([
        {"song_id": "s1", "artist": "L", "song_title": "T", "text": "a", "source_row": 0},
        {"song_id": "s1", "artist": "L", "song_title": "T", "text": "b", "source_row": 1},
        {"song_id": "s2", "artist": "L", "song_title": "T", "text": "b", "source_row": 2},
        {"song_id": "s2", "artist": "L", "song_title": "T", "text": "a", "source_row": 3},
    ])
    _, _, membership = dc.assign_duplicate_groups(records)
    check("a reordered chunk multiset does not group", membership == {})

    # group ids must not depend on which order the songs happened to be inserted
    forward = dc.build_song_records(rows(("s1", "L", "T", "a"), ("s2", "L", "T", "a")))
    backward = dict(reversed(list(forward.items())))
    check("group ids are independent of dict order",
          dc.assign_duplicate_groups(forward)[0] == dc.assign_duplicate_groups(backward)[0])

    _, (_, is_rep, membership) = group([
        ("s1", "L", "T", "a"), ("s2", "L", "T", "a"), ("s3", "L", "T", "a"),
    ])
    check("a group of three has exactly one representative",
          sum(1 for s in membership["SDG-00001"] if is_rep[s]) == 1)


# ----------------------------------------------------------------------- text components

def test_text_components() -> None:
    print("assign_text_components")

    def components(pairs_and_rows):
        records = dc.build_song_records(pairs_and_rows)
        pairs = [(r["song_id"], r["text"]) for r in pairs_and_rows]
        return dc.assign_text_components(pairs, records)

    # a hook repeated inside one song must not union that song with anything
    solo = [
        {"song_id": "s1", "artist": "L", "song_title": "T", "text": "hook", "source_row": 0},
        {"song_id": "s1", "artist": "L", "song_title": "T", "text": "verse", "source_row": 1},
        {"song_id": "s1", "artist": "L", "song_title": "T", "text": "hook", "source_row": 2},
        {"song_id": "s2", "artist": "L", "song_title": "U", "text": "other", "source_row": 3},
    ]
    component_of, membership, shared = components(solo)
    check("within-song repetition is not a cross-song link", shared == 0)
    check("unshared songs stay singletons", all(len(v) == 1 for v in membership.values()))
    check("every song lands in exactly one component",
          sorted(component_of) == ["s1", "s2"] and len(membership) == 2)

    # sharing is transitive: s1-s2 through "x", s2-s3 through "y", so all three are one unit
    chain = [
        {"song_id": "s1", "artist": "L", "song_title": "A", "text": "x", "source_row": 0},
        {"song_id": "s2", "artist": "L", "song_title": "B", "text": "x", "source_row": 1},
        {"song_id": "s2", "artist": "L", "song_title": "B", "text": "y", "source_row": 2},
        {"song_id": "s3", "artist": "L", "song_title": "C", "text": "y", "source_row": 3},
        {"song_id": "s4", "artist": "L", "song_title": "D", "text": "z", "source_row": 4},
    ]
    component_of, membership, shared = components(chain)
    check("shared text unions transitively",
          component_of["s1"] == component_of["s2"] == component_of["s3"])
    check("an unrelated song is not pulled in", component_of["s4"] != component_of["s1"])
    check("two distinct texts are counted as shared", shared == 2)
    check("the first component is the earliest in source order",
          membership["CMP-000001"] == ["s1", "s2", "s3"], str(membership))

    # components must cross label boundaries: leakage does not respect who is credited
    across = [
        {"song_id": "s1", "artist": "L1", "song_title": "A", "text": "x", "source_row": 0},
        {"song_id": "s2", "artist": "L2", "song_title": "B", "text": "x", "source_row": 1},
    ]
    component_of, _, _ = components(across)
    check("components span source-credit labels", component_of["s1"] == component_of["s2"])


# ------------------------------------------------------------------------------- weighting

def test_component_weights() -> None:
    print("component_weights")

    def weigh(row_list):
        records = dc.build_song_records(row_list)
        pairs = [(r["song_id"], r["text"]) for r in row_list]
        component_of, _, _ = dc.assign_text_components(pairs, records)
        return records, component_of, dc.component_weights(records, component_of)

    same_label = [
        {"song_id": "s1", "artist": "L", "song_title": "A", "text": "x", "source_row": 0},
        {"song_id": "s2", "artist": "L", "song_title": "B", "text": "x", "source_row": 1},
        {"song_id": "s3", "artist": "L", "song_title": "C", "text": "y", "source_row": 2},
    ]
    records, component_of, weights = weigh(same_label)
    check("two imports of one component split weight inside a label",
          weights["s1"] == weights["s2"] == 0.5)
    check("an unshared song keeps full weight", weights["s3"] == 1.0)
    total = sum(weights.values())
    distinct = len({component_of[s] for s in records})
    check("label weight totals its component count", abs(total - distinct) < 1e-12,
          f"{total} vs {distinct}")

    # a component spanning two labels carries weight one in each; it is not split between them
    two_labels = [
        {"song_id": "s1", "artist": "L1", "song_title": "A", "text": "x", "source_row": 0},
        {"song_id": "s2", "artist": "L2", "song_title": "B", "text": "x", "source_row": 1},
    ]
    _, _, weights = weigh(two_labels)
    check("a shared component weighs one in each label it appears in",
          weights["s1"] == 1.0 and weights["s2"] == 1.0)

    check("weights are never zero and never exceed one",
          all(0 < w <= 1 for w in weights.values()))


# ---------------------------------------------------------------------------- review queue

def test_review_queue() -> None:
    print("classify_review_queue")

    def queue_for(row_list, retained):
        records = dc.build_song_records(row_list)
        group_of, is_rep, _ = dc.assign_duplicate_groups(records)
        return dc.classify_review_queue(records, set(retained), group_of, is_rep)

    # an exact-sequence duplicate of a retained song under the same title is high confidence
    exact = rows(("s1", "L", "Title", "a"), ("s2", "L", "title", "a"))
    queue, diagnostics = queue_for(exact, {"s1"})
    check("an exact duplicate under the same title is not queued", queue == [])
    check("it is counted as high confidence", diagnostics["high_confidence_duplicate_records"] == 1)

    # the same content under a different title is not automatically a duplicate record
    retitled = rows(("s1", "L", "One", "a"), ("s2", "L", "Two", "a"))
    queue, _ = queue_for(retitled, {"s1"})
    check("an exact duplicate under a different title is queued", len(queue) == 1)
    check("and is labelled by why",
          queue[0]["reason"] == "exact_cleaned_sequence_but_different_normalized_title",
          queue[0]["reason"])

    # a strict subset of one retained song
    subset = [
        {"song_id": "s1", "artist": "L", "song_title": "A", "text": "a", "source_row": 0},
        {"song_id": "s1", "artist": "L", "song_title": "A", "text": "b", "source_row": 1},
        {"song_id": "s2", "artist": "L", "song_title": "B", "text": "a", "source_row": 2},
    ]
    queue, _ = queue_for(subset, {"s1"})
    check("a subset of one retained song is queued",
          len(queue) == 1 and queue[0]["reason"] == "chunk_multiset_subset_of_one_retained_song",
          queue[0]["reason"] if queue else "empty")

    # chunks whose first occurrences sit in two different retained songs
    distributed = [
        {"song_id": "s1", "artist": "L", "song_title": "A", "text": "a", "source_row": 0},
        {"song_id": "s2", "artist": "L", "song_title": "B", "text": "b", "source_row": 1},
        {"song_id": "s3", "artist": "L", "song_title": "C", "text": "a", "source_row": 2},
        {"song_id": "s3", "artist": "L", "song_title": "C", "text": "b", "source_row": 3},
    ]
    queue, _ = queue_for(distributed, {"s1", "s2"})
    check("chunks spread over several retained songs are queued",
          len(queue) == 1
          and queue[0]["reason"] == "chunks_distributed_across_multiple_retained_songs",
          queue[0]["reason"] if queue else "empty")
    check("both related retained songs are recorded",
          queue and queue[0]["related_song_ids"] == ["s1", "s2"])

    # a record with no relation to anything retained must surface, not vanish
    unrelated = rows(("s1", "L", "A", "a"), ("s2", "L", "B", "b"))
    queue, _ = queue_for(unrelated, {"s1"})
    check("an unreconciled record is still queued",
          len(queue) == 1 and queue[0]["reason"] == "unreconciled_other",
          queue[0]["reason"] if queue else "empty")

    # THE safeguard: a queued record that the v2 primary rule happens to group stays queued
    twins = [
        {"song_id": "s1", "artist": "L", "song_title": "A", "text": "a", "source_row": 0},
        {"song_id": "s2", "artist": "L", "song_title": "B", "text": "a", "source_row": 1},
        {"song_id": "s3", "artist": "L", "song_title": "B", "text": "a", "source_row": 2},
    ]
    queue, diagnostics = queue_for(twins, {"s1"})
    check("automatic grouping does not empty the review queue", len(queue) == 2, str(len(queue)))
    check("the grouped member is flagged rather than removed",
          diagnostics["queue_records_also_grouped_by_v2_primary_rule"] == 1,
          str(diagnostics))
    check("the queue tally matches the queue itself",
          diagnostics["manual_review_queue_records"] == len(queue))
    check("erased-record count covers the whole queue plus the high-confidence stratum",
          diagnostics["legacy_erased_song_records"]
          == diagnostics["manual_review_queue_records"]
          + diagnostics["high_confidence_duplicate_records"])


# ------------------------------------------------------------------------- content digest

def test_content_digest() -> None:
    print("corpus_content_sha256")
    base = [
        {"source_credit_label": "L", "song_id": "s1", "song_title": "T",
         "chunk_id": 1, "source_order": 0, "cleaned_text": "a"},
        {"source_credit_label": "L", "song_id": "s1", "song_title": "T",
         "chunk_id": 2, "source_order": 1, "cleaned_text": "b"},
    ]
    digest = dc.corpus_content_sha256(base)
    check("is stable across calls", dc.corpus_content_sha256(base) == digest)

    for field, replacement in (
        ("source_credit_label", "M"),
        ("song_id", "s9"),
        ("song_title", "U"),
        ("chunk_id", 7),
        ("source_order", 5),
        ("cleaned_text", "c"),
    ):
        mutated = [dict(base[0]), dict(base[1])]
        mutated[0][field] = replacement
        check(f"changes when {field} changes", dc.corpus_content_sha256(mutated) != digest)

    reordered = [base[1], base[0]]
    check("changes when row order changes", dc.corpus_content_sha256(reordered) != digest)

    # a field the digest does not cover must not move it: the digest pins content, not labels
    annotated = [dict(row, text_component_id="CMP-000001") for row in base]
    check("ignores fields outside the content contract",
          dc.corpus_content_sha256(annotated) == digest)


def main() -> int:
    for suite in (
        test_strip_del,
        test_normalise_title,
        test_build_song_records,
        test_duplicate_groups,
        test_text_components,
        test_component_weights,
        test_review_queue,
        test_content_digest,
    ):
        suite()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
