"""Self-contained checks for the logic in tools/.

Every tool that reports a number does so from a handful of pure functions, and those
are what can be wrong in a way no reviewer would notice: a macro that silently pools
instead of balancing, a top-k that includes self-matches, a mask that changes document
length. This exercises them against cases whose answers are known by hand.

Chinese fixtures are deliberately stilted constructions, each verified absent from the
corpus. Two earlier fixtures turned out to be real lyric fragments -- they were written
while reading collocation output -- which put corpus text into a public repository.

No private data, no third-party dependencies beyond numpy, no test framework:

    python tests/test_tools.py
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{': ' + detail if detail else ''}")
        FAILURES.append(label)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------- null baseline maths

def test_null_baseline() -> None:
    print("null_baseline_reciprocal_edges")
    nb = load("null_baseline_reciprocal_edges")

    # Four unit vectors in a plane: 0 and 1 close together, 2 and 3 close together.
    angles = np.array([0.00, 0.05, 1.50, 1.55])
    matrix = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    adjacency = nb.mutual_top_k(matrix, k=1)
    check("mutual_top_k finds the two obvious pairs",
          bool(adjacency[0, 1] and adjacency[2, 3] and not adjacency[0, 2]))
    check("mutual_top_k excludes self-matches", not adjacency.diagonal().any())
    check("mutual_top_k is symmetric", bool((adjacency == adjacency.T).all()))

    # A pair present in one layer but not the other must not survive the intersection.
    other = np.stack([np.cos(np.array([0.0, 1.5, 0.05, 1.55])),
                      np.sin(np.array([0.0, 1.5, 0.05, 1.55]))], axis=1)
    edges, connected = nb.graph_stats(matrix, other, k=1)
    check("graph_stats intersects the two layers", edges == 0 and connected == 0,
          f"got {edges} edges, {connected} connected")
    edges, connected = nb.graph_stats(matrix, matrix, k=1)
    check("graph_stats counts each edge once", edges == 2 and connected == 4,
          f"got {edges} edges, {connected} connected")

    vectors = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
    result = nb.centroids(vectors, [[(0, 1.0), (1, 3.0)], [], [(2, 5.0)]])
    expected = np.array([0.75, 3.0]) / np.linalg.norm([0.75, 3.0])
    check("centroids weight members", np.allclose(result[0], expected),
          f"got {result[0]}")
    check("centroids leave an empty group at zero", np.allclose(result[1], 0.0))
    check("centroids are unit length", abs(np.linalg.norm(result[2]) - 1.0) < 1e-12)

    # k must stay below the node count; argpartition would otherwise raise
    check("mutual_top_k rejects k >= n rather than misbehaving",
          _raises(lambda: nb.mutual_top_k(matrix, k=4)))
    check("mutual_top_k keeps the diagonal clear at large k",
          not nb.mutual_top_k(matrix, k=3).diagonal().any())


def _raises(call) -> bool:
    try:
        call()
    except Exception:
        return True
    return False


# ------------------------------------------------------------------ ablation maths

def test_entity_ablation() -> None:
    print("entity_ablation_retrieval")
    ab = load("entity_ablation_retrieval")

    documents = ["甲乙丙巴黎世家丁戊", "甲乙巴黎丙丁", "甲乙上海丙巴黎", "甲乙丙丁戊己庚"]
    masked, replaced = ab.mask_surfaces(documents, ["巴黎", "上海"])
    check("mask_surfaces counts every occurrence", replaced == 4, f"got {replaced}")
    check("mask_surfaces preserves document length",
          all(len(a) == len(b) for a, b in zip(documents, masked)))
    check("mask_surfaces leaves untouched documents alone", masked[3] == documents[3])
    check("mask_surfaces uses a filler absent from any corpus",
          ord(ab.FILLER) == 0xE000)

    longest_first, _ = ab.mask_surfaces(["巴黎世家"], ["巴黎", "巴黎世家"])
    check("mask_surfaces prefers the longest match",
          longest_first[0] == ab.FILLER * 4, f"got {longest_first[0]!r}")

    controls, report = ab.frequency_matched_controls(documents, ["巴黎", "上海"], seed=1)
    check("controls never reuse an entity surface",
          set(controls).isdisjoint({"巴黎", "上海"}), f"got {controls}")
    check("controls match surface length",
          all(len(c) == 2 for c in controls), f"got {controls}")
    check("every surface gets a control, none skipped",
          report["controls_matched"] == report["surfaces"] == 2, f"got {report}")
    check("the report exposes the residual occurrence gap",
          "occurrence_gap_fraction" in report)
    again, _ = ab.frequency_matched_controls(documents, ["巴黎", "上海"], seed=1)
    check("controls are deterministic under a seed", controls == again)
    other, _ = ab.frequency_matched_controls(documents, ["巴黎", "上海"], seed=999)
    check("a different seed is still a valid control",
          set(other).isdisjoint({"巴黎", "上海"}), f"got {other}")

    # label A averages 0.75, label B averages 0.0; balanced macro is 0.375, pooled 0.5
    components = [np.array([[[0.5]], [[1.0]]]), np.array([[[0.0]]])]
    check("macro_point balances by label, not by group",
          abs(ab.macro_point(components, 0, 0) - 0.375) < 1e-12,
          f"got {ab.macro_point(components, 0, 0)}")


# ------------------------------------------------------------------- tagger maths

def test_multi_tagger() -> None:
    print("multi_tagger_agreement")
    mt = load("multi_tagger_agreement")

    alpha = mt.krippendorff_alpha_nominal([["1", "1"], ["1", "1"], ["0", "0"], ["0", "1"]])
    check("alpha matches the hand-computed 8/15", abs(alpha - 8 / 15) < 1e-12, f"got {alpha}")
    check("alpha is 1 on perfect agreement",
          abs(mt.krippendorff_alpha_nominal([["1", "1"], ["0", "0"]]) - 1.0) < 1e-12)
    check("alpha is undefined without variance",
          mt.krippendorff_alpha_nominal([["1", "1"], ["1", "1"]]) is None)
    check("alpha ignores abstentions",
          mt.krippendorff_alpha_nominal([["1", None], ["0", None]]) is None)

    check("canonical_type strips BIO prefixes", mt.canonical_type("B-LOC") == "PLACE")
    check("canonical_type maps CLUENER address", mt.canonical_type("address") == "PLACE")
    check("canonical_type maps person labels", mt.canonical_type("nr") == "PERSON_REFERENCE")
    check("canonical_type refuses unmappable labels", mt.canonical_type("company") is None)

    # a candidate is scored on its own stored span, inside its own stored context
    row = {"context_snippet": "甲乙上海丙丁戊",
           "surface_start_in_context": "4", "surface_end_in_context": "6"}

    def strict(text):
        return {(4, 6, "PLACE")}

    def shifted(text):
        return {(0, 2, "PLACE")}

    taggers = {"a": strict, "b": shifted}
    expressible = {"a": {"PLACE"}, "b": {"PLACE"}}
    check("score_candidate confirms an exact span match",
          mt.score_candidate(row, "PLACE", taggers, expressible) == ["yes", "no"])
    check("score_candidate abstains where the type is inexpressible",
          mt.score_candidate(row, "PLACE", taggers, {"a": set(), "b": {"PLACE"}}) == [None, "no"])
    check("score_candidate abstains on an unusable span",
          mt.score_candidate({"context_snippet": "x", "surface_start_in_context": "",
                              "surface_end_in_context": ""}, "PLACE", taggers, expressible)
          == [None, None])


# ------------------------------------------------------------- claim audit logic

def test_claim_audit() -> None:
    print("audit_released_claim_occurrences")
    ca = load("audit_released_claim_occurrences")

    def row(surface, snippet, start, end):
        return {"candidate_surface": surface, "context_snippet": snippet,
                "surface_start_in_context": str(start), "surface_end_in_context": str(end)}

    kind, tail = ca.classify(row("巴黎", "甲乙巴黎世家丙", 2, 4))
    check("classify names a known compound", kind.startswith("Balenciaga"), f"got {kind}")
    check("classify reports the following characters", tail.startswith("世家"), f"got {tail!r}")

    kind, _ = ca.classify(row("巴黎", "甲乙巴黎丙丁", 2, 4))
    check("classify passes the same surface used bare", kind == "bare surface", f"got {kind}")

    kind, _ = ca.classify(row("湖南", "甲乙湖南卫视丙", 2, 4))
    check("classify catches a broadcaster compound", kind.startswith("Hunan TV"), f"got {kind}")

    kind, _ = ca.classify(row("上海", "x", 9, 99))
    check("classify tolerates a span past the snippet", kind == "bare surface", f"got {kind}")

    kind, _ = ca.classify(row("上海", "上海", "", ""))
    check("classify tolerates an unusable span", kind == "bare surface", f"got {kind}")


# -------------------------------------------------------- reliability join, on disk

def test_surface_reliability() -> None:
    print("summarise_surface_reliability")
    sr = load("summarise_surface_reliability")

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        sr.NER = tmp
        write_csv(tmp / "entity_aggregate_provisional.csv", [
            {"entity": "甲", "entity_type": "PLACE", "strict_agreement_occurrences": 6,
             "lexicon_candidate_occurrences": 10, "strict_agreement_rate": 0.6},
            {"entity": "乙", "entity_type": "PLACE", "strict_agreement_occurrences": 9,
             "lexicon_candidate_occurrences": 10, "strict_agreement_rate": 0.9},
        ])
        write_csv(tmp / "source_label_entity_links_provisional.csv", [
            {"source_credit_label": "L", "entity": "甲", "entity_song_units_within_label": 5},
        ])
        write_csv(tmp / "entity_co_mentions_provisional.csv", [
            {"entity_a": "甲", "entity_a_type": "PLACE", "entity_b": "乙",
             "entity_b_type": "PLACE", "unique_song_unit_co_mentions": 3},
        ])
        rows = sr.build()

    by_entity = {row["entity"]: row for row in rows}
    check("build orders by agreement, weakest first", rows[0]["entity"] == "甲")
    check("build counts every claim a surface carries",
          by_entity["甲"]["released_claims_carried"] == 2,
          f"got {by_entity['甲']['released_claims_carried']}")
    check("build sums support across claim types",
          by_entity["甲"]["released_claim_song_units"] == 8,
          f"got {by_entity['甲']['released_claim_song_units']}")
    check("build derives the rejected count",
          by_entity["乙"]["occurrences_rejected_by_span_disagreement"] == 1)
    check("a surface carrying no claim is still reported",
          by_entity["乙"]["released_claims_carried"] == 1)


# ------------------------------------------------------------ manifest resolution

def test_verifier() -> None:
    print("verify_release_integrity")
    vr = load("verify_release_integrity")
    check("resolve finds a repo-relative path", vr.resolve("README.md", "") == "README.md")
    check("resolve prefers a manifest-relative path",
          vr.resolve("METHOD.md", "results/retrieval-v1") == "results/retrieval-v1/METHOD.md")
    check("resolve returns None for a private build input",
          vr.resolve("work/private-canonical-corpus-v1/x.csv", "") is None)
    check("resolve ignores a non-string hint", vr.resolve(None, "") is None)


def main() -> int:
    suites = [test_null_baseline, test_entity_ablation, test_multi_tagger,
              test_claim_audit, test_surface_reliability]
    if (ROOT / "tools" / "verify_release_integrity.py").is_file():
        suites.append(test_verifier)
    else:
        print("\n  (verify_release_integrity.py is repository-only and absent here; "
              "its suite is skipped)")
    for suite in suites:
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
