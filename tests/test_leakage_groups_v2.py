"""The union grouping must catch both kinds of leakage, and must be able to fail.

Corpus v2's exact-text components and the retrieval builder's near-duplicate rule each see
leakage the other does not. A test that only checked the union produces *some* grouping
would pass for either rule alone, so each case below is built so that exactly one of the two
rules can catch it, and then asserts the union catches both.

Fixtures are synthetic. No corpus text appears here -- that is a release-boundary rule, and
a detector fixture drawn from the real corpus has already been found in this repository once.

    python tests/test_leakage_groups_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from leakage_groups_v2 import (  # noqa: E402
    build_groups,
    group_audit,
    near_duplicate_pairs,
    normalise_document,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}{': ' + detail if detail else ''}")
        print(f"  FAIL {name}{': ' + detail if detail else ''}")


# A shared verse, long enough to be a real chunk, and two songs whose other content differs
# completely. The whole documents are nothing alike, so only the component rule links them.
SHARED_VERSE = "aaaaabbbbbcccccdddddeeeee"
ONLY_A = "fffffggggghhhhhiiiiijjjjj"
ONLY_B = "kkkkkllllmmmmmnnnnnooooo"

# Two songs that share almost all of their text and differ only in a short tail. At trigram
# level that is Jaccard ~0.9, well above the 0.80 threshold, while no chunk of either need be
# byte-identical to a chunk of the other -- so the component rule cannot link them and only
# the near-duplicate rule can. A difference spread thinly through the document instead would
# corrupt three trigrams at every edit and fall below the threshold, which is why the tail is
# where the two differ.
_BODY = "".join(chr(0x61 + (index * 7 + index // 5) % 26) for index in range(180))
NEAR_1 = _BODY + "qqqqqqqqqq"
NEAR_2 = _BODY + "vvvvvvvvvv"

UNRELATED = "zzzzz00000zzzzz11111zzzzz22222"


def scenario():
    documents = {
        "s1": normalise_document(SHARED_VERSE + ONLY_A),
        "s2": normalise_document(SHARED_VERSE + ONLY_B),
        "s3": normalise_document(NEAR_1),
        "s4": normalise_document(NEAR_2),
        "s5": normalise_document(UNRELATED),
    }
    components = {
        "s1": {"C1"}, "s2": {"C1"},          # share the verse
        "s3": {"C3"}, "s4": {"C4"},          # no shared chunk at all
        "s5": {"C5"},
    }
    return documents, components


print("the two rules each catch what the other misses")
documents, components = scenario()

component_only = build_groups(list(documents), components, documents, threshold=1.01)
check("without near-duplicate edges, s3 and s4 are NOT grouped",
      component_only["s3"] != component_only["s4"])
check("without near-duplicate edges, s1 and s2 ARE grouped (shared component)",
      component_only["s1"] == component_only["s2"])

no_components = build_groups(list(documents), {k: set() for k in components}, documents)
check("without components, s1 and s2 are NOT grouped",
      no_components["s1"] != no_components["s2"],
      "the documents are too different for the near-duplicate rule")
check("without components, s3 and s4 ARE grouped (near-duplicates)",
      no_components["s3"] == no_components["s4"])

print()
print("the union catches both, and nothing else")
groups = build_groups(list(documents), components, documents)
check("s1 and s2 grouped", groups["s1"] == groups["s2"])
check("s3 and s4 grouped", groups["s3"] == groups["s4"])
check("s1 and s3 are not grouped", groups["s1"] != groups["s3"])
check("s5 stands alone", len({s for s, g in groups.items() if g == groups["s5"]}) == 1)

print()
print("determinism and the invariant")
reversed_input = build_groups(list(reversed(list(documents))), components, documents)
check("input order does not change the result", groups == reversed_input)

labels = {"s1": "L1", "s2": "L1", "s3": "L2", "s4": "L2", "s5": "L2"}
audit = group_audit(groups, labels, components)
check("no text component is split across groups",
      audit["text_components_split_across_groups"] == 0)
check("audit counts the groups it produced", audit["groups"] == len(set(groups.values())))

print()
print("the invariant can actually fail")
# Hand-build a grouping that splits a component, and confirm the audit reports it. An
# invariant that no input can violate would tell a reader nothing.
split = dict(groups)
split["s2"] = "deliberately-separated"
broken = group_audit(split, labels, components)
check("a component split across groups is reported",
      broken["text_components_split_across_groups"] == 1,
      f"got {broken['text_components_split_across_groups']}")

print()
print("the near-duplicate join is exact, not a sample")
pairs = near_duplicate_pairs({"a": documents["s3"], "b": documents["s4"]})
check("the near-duplicate pair is found", ("a", "b") in pairs)
far = near_duplicate_pairs({"a": documents["s1"], "b": documents["s5"]})
check("unrelated documents are not paired", not far)

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed")
    sys.exit(1)
print("all leakage-grouping checks passed")
