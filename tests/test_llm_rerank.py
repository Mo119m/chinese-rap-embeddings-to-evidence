"""Synthetic checks for src/llm_rerank_v3.py, without the corpus and without the model:

    python tests/test_llm_rerank.py

Everything that does not need the LLM is checked against a literal definition: the candidate
sets, the leave-group-out exemplar draw, the letter map and the shuffle, the answer parser, the
re-ranking rule against a brute-force recomputation of MRR, and the two limiting choosers -- one
that always takes the retriever's first choice (must leave MRR unchanged) and one that always
takes the true label when it is offered (must lift recall@1 to recall@K on those queries).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import llm_rerank_v3 as lr  # noqa: E402
from analyse_identity_encoder_v3 import ranks_of  # noqa: E402


def fixture(seed=5, n=400, L=30, groups=150, dim=32):
    rng = np.random.default_rng(seed)
    li = rng.integers(0, L, size=n)
    gi = rng.integers(0, groups, size=n)
    x = rng.normal(size=(n, dim)) + 1.2 * rng.normal(size=(L, dim))[li]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    centres = np.stack([x[li == l].mean(axis=0) for l in range(L)])
    scores = x @ centres.T + 0.05 * rng.normal(size=(n, L))
    return rng, li, gi, scores


def check_candidates_and_exemplars():
    rng, li, gi, scores = fixture()
    cands = lr.candidate_sets(scores, k=lr.K)
    for q in range(0, len(li), 37):
        top = np.argsort(-scores[q])[:lr.K]
        assert list(cands[q]) == list(top), (q, cands[q], top)
    for q in range(0, len(li), 23):
        for l in cands[q]:
            ex = lr.exemplars_for(int(l), int(q), li, gi, rng, n=lr.EXEMPLARS)
            assert 1 <= len(ex) <= lr.EXEMPLARS
            for s in ex:
                assert li[s] == l, "an exemplar is not the candidate's song"
                assert gi[s] != gi[q], "an exemplar shares the query's leakage group"
                assert s != q
    print("ok: the top-k candidates are the retriever's best k, and exemplars never share the query's group")


def check_prompt_and_letters():
    cands = [17, 3, 25, 8, 11, 0, 29, 14, 6, 21]
    texts = {c: [f"song {c} one", f"song {c} two"] for c in cands}
    order = [9, 0, 4, 2, 7, 1, 8, 3, 6, 5]
    prompt, letter_of = lr.build_prompt("the query", cands, texts, order)
    assert set(letter_of) == set(lr.LETTERS[:lr.K]) and len(set(letter_of.values())) == lr.K
    assert letter_of["A"] == cands[9] and letter_of["B"] == cands[0] and letter_of["J"] == cands[5]
    assert "the query" in prompt and prompt.index("【候选歌手 A】") < prompt.index("【候选歌手 J】")
    assert "song 21 one" in prompt.split("【候选歌手 A】")[1].split("【候选歌手 B】")[0]
    for name in ("artist", "歌手名"):
        assert name not in lr.TEMPLATE.replace("候选歌手", ""), "no name channel in the template"
    # the true label's letter position is uniform under the per-query shuffle
    pos = Counter()
    for q in range(3000):
        rng = np.random.default_rng(lr.EXEMPLAR_SEED * 100003 + q)
        order = list(rng.permutation(lr.K))
        _, lo = lr.build_prompt("q", cands, texts, order)
        pos[[k for k, v in lo.items() if v == cands[0]][0]] += 1
    assert max(pos.values()) < 3000 / lr.K * 1.35 and min(pos.values()) > 3000 / lr.K * 0.65, pos
    prompt_b, _ = lr.build_prompt(lr.PLACEHOLDER, cands, texts, order)
    assert "the query" not in prompt_b and lr.PLACEHOLDER in prompt_b
    print("ok: letters map one-to-one, the shuffle is uniform over positions, and the blind prompt has no query")


def check_parse_and_truncate():
    assert lr.parse_letter("C") == "C" and lr.parse_letter(" 答案：b。") == "B" and lr.parse_letter("J") == "J"
    assert lr.parse_letter("K") is None and lr.parse_letter("") is None and lr.parse_letter("不知道") is None
    assert lr.parse_letter("A或B") == "A", "the first letter wins"
    t = "x" * 700
    assert lr.truncate(t, 500).endswith("…") and len(lr.truncate(t, 500)) == 501
    assert lr.truncate("short", 500) == "short"
    assert lr.normalise_name(" A-B c_1 ") == "abc1"
    print("ok: the parser abstains on anything that is not a candidate letter; truncation and name normalisation behave")


def check_rerank_rule():
    rng, li, gi, scores = fixture(seed=9)
    cands = lr.candidate_sets(scores)
    base = 1.0 / ranks_of(scores, li)
    # chooser 1: always the retriever's first choice -> MRR identical
    same = scores.copy()
    for q in range(len(li)):
        same[q] = lr.rerank(scores[q], cands[q], int(cands[q][0]))
    assert np.array_equal(ranks_of(same, li), ranks_of(scores, li))
    # chooser 2: the true label whenever offered -> rank 1 there, unchanged elsewhere
    oracle = scores.copy()
    for q in range(len(li)):
        chosen = int(li[q]) if li[q] in cands[q] else None
        oracle[q] = lr.rerank(scores[q], cands[q], chosen)
    r = ranks_of(oracle, li)
    offered = np.asarray([li[q] in cands[q] for q in range(len(li))])
    assert np.all(r[offered] == 1)
    assert np.array_equal(r[~offered], ranks_of(scores, li)[~offered])
    assert abs(float((1.0 / r).mean()) - (offered.mean() + (1.0 / ranks_of(scores, li))[~offered].sum() / len(li))) < 1e-12
    # a wrong choice: the chosen candidate goes to rank 1 and the true label loses exactly one
    # place if it was among the candidates and ahead of nothing else changes
    q = int(np.flatnonzero(offered)[0])
    wrong = int([c for c in cands[q] if c != li[q]][0])
    row = lr.rerank(scores[q], cands[q], wrong)
    order_before = list(np.argsort(-scores[q]))
    order_after = list(np.argsort(-row))
    assert order_after[0] == wrong
    rest_before = [c for c in order_before if c != wrong]
    assert order_after[1:] == rest_before, "everything except the chosen label keeps its order"
    assert base.mean() > 0
    print("ok: the re-ranking rule leaves MRR unchanged under the retriever's own choice, lifts the oracle to rank 1, and moves nothing else")


def main() -> int:
    check_candidates_and_exemplars()
    check_prompt_and_letters()
    check_parse_and_truncate()
    check_rerank_rule()
    print("all llm-rerank checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
