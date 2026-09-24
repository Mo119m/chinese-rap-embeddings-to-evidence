#!/usr/bin/env python3
"""Does a local instruction-tuned LLM improve on the protocol's retriever when asked to choose
among its top candidates?

WHY THIS ARM. Zero-shot authorship attribution with instruction-tuned LLMs is the current default
in the literature (GPT-4 Turbo reaches 84% with 10 candidate authors on blog posts against 42% for
a supervised BERT, Huang et al. 2024), and its stated limitation is that it does not scale to many
candidates. This corpus has 226 credited labels, so the honest form of the question is the
retrieve-and-rerank one: the protocol's own retriever proposes its top K labels, and the LLM
chooses among them from example lyrics. Nothing leaves this machine: the model is a pinned local
checkpoint run offline, and the candidates are shown as letters, never as artist names, so the
model cannot answer from what it knows about a rapper -- only from the lyrics in front of it.

THE RETRIEVER RERANKED. The protocol's raw jieba word TF-IDF prototype (published MRR 0.4963):
zero fitted parameters, exactly reproducible, and the scorer the review called "protocol". The
gate reproduces 0.4963 on all 7,220 queries before any subset is drawn.

THE PROMPT. One query song and K = 10 candidate labels, each shown by EXEMPLARS songs drawn at
random (seeded) from the label's songs OUTSIDE the query's leakage group -- the same leave-group-out
rule every scorer here obeys, so a candidate's exemplars can never be the query's own twin. Songs
are truncated to TRUNCATE characters (median song 955 characters, p90 1,919). Candidate order is
shuffled per query and the retriever's rank is not shown. The model answers with one letter;
greedy decoding; an unparseable answer abstains and leaves the retriever's ranking unchanged.
The template is recorded in the payload verbatim; it contains no lyric.

RERANKING RULE. The chosen candidate moves to rank 1; the other candidates keep the retriever's
order; labels outside the top K keep their ranks. MRR is over the full 226-label ranking.

CONTROLS, all pre-registered, all on the same subset.
  C1 reversed order: the same query, exemplars and candidates in reversed order. Agreement with
     the main choice below AGREEMENT_FLOOR means the model reads position, and R1 is void.
  C2 blind: the query text is replaced by a fixed placeholder, candidates unchanged. Accuracy
     above BLIND_CEILING (chance is 1/K) means the presentation leaks the answer, and R1 is void.
  C3 memorisation probe: the query alone, no candidates, "who performs this?"; the answer is
     compared with the credited label string (normalised exact match). This does not void R1 but
     bounds how much of any gain could be recognition of memorised text rather than attribution.
     Only the match RATE is published; the model's guesses stay in the private cache.
  C4 by songs-per-label band, descriptive.

READING RULE R1, fixed before the run. On the subset, MRR(reranked) minus MRR(retriever),
component-weighted with a paired group-bootstrap interval: "improves" if the whole interval lies
above zero; "no gain" if the whole interval lies below +0.005 (the margin used elsewhere here);
"undecided" otherwise. Reported beside it: the change in recall@1, and among queries whose true
label is in the top K, how often the model's choice is correct against how often the retriever's
first choice is. C1 and C2 are read first; if either voids R1, R1 is reported as void with the
control's number, not as a result.

CLEANLINESS RECORDS. Model id and pinned revision, sha256 of every weight shard and of the
tokenizer, HF offline mode, torch and transformers versions, GPU, decoding parameters, the corpus
content digest, the subset seed, the exemplar seed, the prompt template. Per-query rows (song
index, letters, answers) go to --cache-dir only and never to results/; the payload is aggregate.

GAMING. Before every batch the script looks for VALORANT-Win64-Shipping.exe; if it is running the
model is unloaded and the GPU released, and work resumes ten minutes after the game closes. The
per-query cache means a kill loses at most one batch.

    CHINESE_RAP_CORPUS=v3 python src/llm_rerank_v3.py --private-root <root> --model-dir <dir> --cache-dir <dir>
    ... --model-dir <dir> --cache-dir <dir> --smoke   (synthetic text only: VRAM, throughput, token
                                                      ratio; touches no lyric and needs no corpus)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analyse_identity_encoder_v3 import ranks_of  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256  # noqa: E402
from identity_probe_v2 import SEED  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from label_size_calibration_v3 import BANDS, lgo_parts  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
OUT_NAME = "llm_rerank.json"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_RETRIEVER = 0.4963
TOLERANCE = 5e-4
K = 10
EXEMPLARS = 2                 # two songs per candidate show a label's own variation
TRUNCATE = 350                # about the first third of a median song: 2 x 350 characters over 10
                              # candidates is a median 4,726-token prompt on real lyrics (p90 5,209,
                              # measured on 200 queries at 0.617 tokens per character)
SUBSET = 1000
SUBSET_SEED = SEED
EXEMPLAR_SEED = SEED + 7
LETTERS = "ABCDEFGHIJ"
MARGIN = 0.005
AGREEMENT_FLOOR = 0.7
BLIND_CEILING = 0.13          # chance is 0.10; three binomial sd at n = 1,000 is 0.028
BATCH = 1                     # one prompt at a time: no padding, no mask, so SDPA keeps its
                              # memory-efficient kernel; a batch of 8 ten-thousand-token prompts
                              # asked the math kernel for 127 GiB on this 8 GiB card
MAX_NEW_TOKENS = 4
SMOKE_TOKENS = 4800           # the measured median real prompt is 4,726 tokens, p90 5,209


def _attention_backends():
    from torch.nn.attention import SDPBackend
    return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]


try:
    ATTENTION_BACKENDS = _attention_backends()
except Exception:                                   # torch absent: the pure functions still import
    ATTENTION_BACKENDS = []
GAME = "VALORANT-Win64-Shipping.exe"
GAME_COOLDOWN_S = 600
PLACEHOLDER = "（此处没有歌词）"
OOM_ABSTAIN = "<out of memory>"   # never contains a candidate letter, so it parses as an abstention
YIELD_FREE_GIB = 0.6              # a 5k-token prompt needs about 0.55 GiB beyond the loaded weights
YIELD_LIMIT_S = 1800              # yield to the other GPU job for at most half an hour per prompt

TEMPLATE = (
    "下面是一首中文说唱歌曲的歌词（可能被截断），以及 {k} 位候选歌手各自的 {n} 首其他歌曲的歌词片段。"
    "候选歌手只用字母表示。请仅根据用词、句式、押韵和主题等文本特征判断，这首歌最可能出自哪位候选歌手。"
    "只回答一个字母。\n\n"
    "【待判断的歌曲】\n{query}\n\n"
    "{candidates}"
    "答案（只写字母）："
)
CANDIDATE_BLOCK = "【候选歌手 {letter}】\n{songs}\n\n"
SONG_SEP = "\n----\n"
PROBE_TEMPLATE = "下面是一首中文说唱歌曲的歌词（可能被截断）。这首歌的演唱者是谁？只回答歌手名字，不要解释。\n\n{query}\n\n答案："


# ------------------------------------------------------------------ pure functions (tested)
def truncate(text: str, limit: int = TRUNCATE) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def candidate_sets(scores: np.ndarray, k: int = K) -> np.ndarray:
    """The retriever's top-k labels per query, in rank order (best first)."""
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def exemplars_for(label: int, query: int, label_index, group_ids, rng, n: int = EXEMPLARS) -> list[int]:
    """n songs of the label outside the query's leakage group, drawn at random."""
    pool = np.flatnonzero((label_index == label) & (group_ids != group_ids[query]))
    if len(pool) == 0:
        raise RuntimeError("a candidate label has no song outside the query's group")
    take = min(n, len(pool))
    return [int(s) for s in rng.choice(pool, size=take, replace=False)]


def build_prompt(query_text: str, candidates: list[int], exemplar_texts: dict[int, list[str]],
                 order: list[int], k: int = K, n: int = EXEMPLARS) -> tuple[str, dict[str, int]]:
    """Prompt text plus the letter -> label map, candidates shown in `order` (a permutation)."""
    blocks, letter_of = [], {}
    for pos, idx in enumerate(order):
        label = candidates[idx]
        letter = LETTERS[pos]
        letter_of[letter] = int(label)
        blocks.append(CANDIDATE_BLOCK.format(letter=letter, songs=SONG_SEP.join(exemplar_texts[label])))
    prompt = TEMPLATE.format(k=k, n=n, query=query_text, candidates="".join(blocks))
    return prompt, letter_of


def parse_letter(answer: str, k: int = K) -> str | None:
    """The first candidate letter in the answer, or None (abstain)."""
    m = re.search(rf"[{LETTERS[:k]}]", answer.strip().upper())
    return m.group(0) if m else None


def rerank(scores_row: np.ndarray, candidates: np.ndarray, chosen: int | None) -> np.ndarray:
    """A score row in which the chosen candidate outranks every other candidate and nothing else
    moves: the chosen label takes the top candidate's score plus a margin that no other label can
    reach, so labels outside the top K keep their ranks exactly."""
    row = scores_row.copy()
    if chosen is None:
        return row
    top = float(np.max(row[candidates]))
    row[chosen] = top + 1.0
    return row


def normalise_name(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.strip().lower())


def game_running() -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {GAME}"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:
        return False
    return GAME.lower().split(".exe")[0] in out.lower()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ the model
class Chooser:
    """Greedy one-letter answers from the pinned local model; unloads itself for the game."""

    def __init__(self, model_path: Path):
        self.path = model_path
        self.model = None
        self.tokenizer = None
        self.oom_abstains = 0
        self.cuda_failures = 0

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.float16, device_map="cuda")
        self.model.eval()

    def unload(self):
        self.model = None
        self._safe_empty_cache()                  # an asynchronous CUDA error can surface here too

    def wait_for_game(self):
        if not game_running():
            return
        print("  the game is running: releasing the GPU", flush=True)
        self.unload()
        clear_since = None
        while True:
            time.sleep(30)
            if game_running():
                clear_since = None
                continue
            clear_since = clear_since or time.time()
            if time.time() - clear_since >= GAME_COOLDOWN_S:
                break
        print("  the game has been closed for ten minutes: reloading", flush=True)
        self.load()

    def answer(self, prompts: list[str], max_new_tokens: int = MAX_NEW_TOKENS) -> list[str]:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        self.load()
        chats = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                    add_generation_prompt=True) for p in prompts]
        enc = self.tokenizer(chats, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        # Measured on this card: the default backend choice ran a 4.8k-token prompt through the
        # math path in 49 s at a 9.4 GiB peak (over the 8 GiB physical, spilling to system memory);
        # the fused kernels do the same prompt in 1.6 s at 6.3 GiB. So the fused kernels are
        # required, and a spill is treated as an error rather than a slowdown.
        # This card is shared with another project's GPU job that allocates intermittently. Two
        # runs died to it: once in the KV-cache concatenation, once inside empty_cache() itself
        # (an asynchronous CUDA error surfacing late). So before each prompt the driver's free
        # memory is checked and the job YIELDS while it is short, and a CUDA failure climbs a
        # ladder -- clear and retry, then unload / wait / reload and retry -- before abstaining.
        self.yield_while_vram_is_short()
        out = None
        for attempt in range(3):
            try:
                torch.cuda.reset_peak_memory_stats()
                with sdpa_kernel(ATTENTION_BACKENDS), torch.no_grad():
                    out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                              pad_token_id=self.tokenizer.eos_token_id)
                torch.cuda.synchronize()
                break
            except (torch.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as failure:
                if "CUDA" not in str(failure) and "out of memory" not in str(failure).lower():
                    raise
                self.cuda_failures += 1
                print(f"  CUDA failure {self.cuda_failures} on attempt {attempt + 1}: {str(failure)[:70]}", flush=True)
                if attempt == 0:
                    self._safe_empty_cache()
                    time.sleep(20)
                elif attempt == 1:
                    self.unload()
                    time.sleep(60)
                    self.yield_while_vram_is_short()
                    self.load()
                    enc = enc.to("cuda")
                else:
                    self.oom_abstains += 1
                    print("  three CUDA failures on one prompt: abstaining on it", flush=True)
                    self._safe_empty_cache()
                    return [OOM_ABSTAIN] * len(prompts)
        peak = torch.cuda.max_memory_allocated()
        physical = torch.cuda.get_device_properties(0).total_memory
        self._safe_empty_cache()                      # return every transient block between prompts
        if peak > physical:
            raise RuntimeError(f"peak allocation {peak / 2**30:.2f} GiB exceeds the card's {physical / 2**30:.2f} GiB: "
                               "the run spilled to system memory")
        new = out[:, enc["input_ids"].shape[1]:]
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in new]

    def _safe_empty_cache(self):
        import torch
        try:
            torch.cuda.empty_cache()
        except Exception as failure:              # an asynchronous error can surface here too
            print(f"  empty_cache raised ({str(failure)[:60]}); continuing", flush=True)

    def yield_while_vram_is_short(self, need_gib: float = YIELD_FREE_GIB, limit_s: int = YIELD_LIMIT_S):
        """Sleep while the driver reports less free memory than one prompt needs, up to a limit."""
        import torch
        waited = 0
        while waited < limit_s:
            try:
                free, _ = torch.cuda.mem_get_info()
            except Exception:
                return
            if free / 2**30 >= need_gib:
                if waited:
                    print(f"  VRAM free again after {waited}s", flush=True)
                return
            if not waited:
                print(f"  only {free / 2**30:.2f} GiB of VRAM free: yielding to the other GPU job", flush=True)
            time.sleep(15)
            waited += 15
        print(f"  VRAM still short after {limit_s}s; trying anyway", flush=True)

    def token_count(self, text: str) -> int:
        if self.tokenizer is None:                # the tokenizer alone: no weights, no VRAM
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        return len(self.tokenizer(text)["input_ids"])


# ------------------------------------------------------------------ the run
def run_condition(name, chooser, jobs, cache: Path, batch: int = BATCH) -> dict[int, str]:
    """jobs: list of (query_index, prompt). Answers cached per query in a private JSONL."""
    done: dict[int, str] = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[int(row["q"])] = row["a"]
    todo = [(q, p) for q, p in jobs if q not in done]
    print(f"  {name}: {len(done):,} cached, {len(todo):,} to run", flush=True)
    t0 = time.time()
    with cache.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), batch):
            chooser.wait_for_game()
            chunk = todo[start:start + batch]
            answers = chooser.answer([p for _, p in chunk])
            for (q, _), a in zip(chunk, answers):
                done[q] = a
                handle.write(json.dumps({"q": int(q), "a": a}, ensure_ascii=False) + "\n")
            handle.flush()
            if (start // batch) % 10 == 0:
                rate = (start + len(chunk)) / max(time.time() - t0, 1e-9)
                print(f"    {name}: {start + len(chunk):,}/{len(todo):,} ({rate * 60:.0f} per minute)", flush=True)
    return done


def smoke(chooser: Chooser, out: Path, batch: int = BATCH) -> int:
    """VRAM, throughput and the token ratio on synthetic text that is not lyrics."""
    import torch
    rng = random.Random(0)
    # Common characters only: text drawn from the whole CJK block tokenises at 1.5+ tokens per
    # character against 0.62 for real lyrics, and a prompt three times too long in tokens is what
    # a smoke test must not measure. The first 600 code points of the block are everyday
    # characters; the lines are short like lyric lines. It is still not lyrics.
    han = [chr(c) for c in range(0x4E00, 0x4E00 + 600)]
    fake = lambda n: "".join(rng.choice(han) if rng.random() < 0.92 else "\n" for _ in range(n))
    query = fake(TRUNCATE)
    cands = list(range(K))
    texts = {c: [fake(TRUNCATE) for _ in range(EXEMPLARS)] for c in cands}
    prompt, _ = build_prompt(query, cands, texts, list(range(K)))
    chooser.load()
    tokens = chooser.token_count(prompt)
    print(f"synthetic prompt: {len(prompt):,} characters -> {tokens:,} tokens ({tokens / len(prompt):.2f} per character)", flush=True)
    # Synthetic characters do not form words, so they tokenise at twice the rate of real lyrics
    # (0.62 tokens per character, measured). What this test must reproduce is the real prompt's
    # LENGTH IN TOKENS, so the synthetic prompt is cut to the measured real median and re-decoded.
    ids = chooser.tokenizer(prompt)["input_ids"][:SMOKE_TOKENS]
    prompt = chooser.tokenizer.decode(ids)
    tokens = chooser.token_count(prompt)
    print(f"cut to the real prompt length: {tokens:,} tokens", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    answers = []
    for _ in range(4):                     # four prompts, one call each: the real run's shape
        answers += chooser.answer([prompt] * batch)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"4 x batch {batch}: {dt:.1f}s ({4 * batch / dt * 60:.0f} per minute), peak VRAM {peak:.2f} GiB, "
          f"answers {answers}", flush=True)
    out.write_bytes((json.dumps({"characters": len(prompt), "tokens": tokens, "batch": batch,
                                 "seconds": round(dt, 2), "peak_vram_gib": round(peak, 2),
                                 "answers": answers}, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="a plain directory holding the pinned snapshot; fetched on first use, offline after")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--subset", type=int, default=SUBSET)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.time()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this arm must not fall back to CPU")
    model_path = args.model_dir.resolve()
    if not (model_path / "config.json").is_file():
        # first use only: the pinned revision goes into a plain directory, because the hub cache
        # wants symlinks and Windows refuses them without a privilege; every later run is offline
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=model_path,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise SystemExit("no weight shards under the pinned snapshot")
    model_record = {"id": MODEL_ID, "revision": MODEL_REVISION, "license": "Qwen Research License (see LICENSE in the snapshot)",
                    "weights_sha256": {s.name: sha256_file(s) for s in shards},
                    "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
                    "offline": os.environ.get("HF_HUB_OFFLINE"), "dtype": "float16",
                    "attention_backends": [b.name for b in ATTENTION_BACKENDS],
                    "decoding": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS},
                    "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                    "gpu": torch.cuda.get_device_name(0), "python": platform.python_version()}
    print(f"model {MODEL_ID}@{MODEL_REVISION[:12]} from {model_path}", flush=True)
    chooser = Chooser(model_path)
    if args.smoke:
        return smoke(chooser, args.cache_dir / "smoke.json", batch=args.batch)
    if args.private_root is None:
        raise SystemExit("--private-root is required for the real run")

    import jieba
    jieba.setLogLevel(60)
    from exemplar_vs_prototype_v3 import setup
    from word_identity_anatomy_v2 import fit_words, segment
    d = setup(args.private_root.resolve())
    li, gi, w, L = d["label_index"], d["group_ids"], d["weights"], d["label_count"]
    songs = d["songs"]
    docs = [d["documents"][s] for s in songs]
    labels_text = d["labels"]                      # the credited label strings, by label index
    n = len(li)
    everything = np.arange(n)
    words, _ = fit_words([" ".join(segment(doc)) for doc in docs])
    words = words.astype(np.float64).tocsr()
    dots, norm2, _ = lgo_parts(words, li, gi, w, L, everything)
    scores = dots / np.sqrt(norm2)
    del dots, norm2
    got = float(np.mean(1.0 / ranks_of(scores, li)))
    if abs(got - EXPECTED_RETRIEVER) > TOLERANCE:
        raise SystemExit(f"the retriever does not reproduce the published MRR ({got:.4f})")
    print(f"gate: raw word prototype {got:.4f} against the published {EXPECTED_RETRIEVER}", flush=True)

    # the subset and the candidate sets
    sub = np.sort(np.random.default_rng(SUBSET_SEED).choice(n, size=min(args.subset, n), replace=False))
    cands = candidate_sets(scores)
    in_top = np.asarray([li[q] in cands[q] for q in sub])
    print(f"subset {len(sub):,} queries; true label in the top {K} for {in_top.mean():.4f}", flush=True)

    # exemplars and prompts, one exemplar rng stream per query so the draw is reproducible
    jobs_main, jobs_rev, jobs_blind, jobs_probe, maps = [], [], [], [], {}
    for q in sub:
        rng = np.random.default_rng(EXEMPLAR_SEED * 100003 + int(q))
        texts = {int(l): [truncate(docs[s]) for s in exemplars_for(int(l), int(q), li, gi, rng)] for l in cands[q]}
        order = list(rng.permutation(K))
        prompt, letter_of = build_prompt(truncate(docs[q]), [int(l) for l in cands[q]], texts, order)
        prompt_rev, letter_rev = build_prompt(truncate(docs[q]), [int(l) for l in cands[q]], texts, order[::-1])
        prompt_blind, _ = build_prompt(PLACEHOLDER, [int(l) for l in cands[q]], texts, order)
        maps[int(q)] = (letter_of, letter_rev)
        jobs_main.append((int(q), prompt))
        jobs_rev.append((int(q), prompt_rev))
        jobs_blind.append((int(q), prompt_blind))
        jobs_probe.append((int(q), PROBE_TEMPLATE.format(query=truncate(docs[q]))))
    sample_tokens = chooser.token_count(jobs_main[0][1])
    print(f"first prompt: {len(jobs_main[0][1]):,} characters, {sample_tokens:,} tokens", flush=True)

    tag = f"{MODEL_REVISION[:8]}_k{K}_n{EXEMPLARS}_t{TRUNCATE}_s{SUBSET_SEED}"
    ans_main = run_condition("main", chooser, jobs_main, args.cache_dir / f"main_{tag}.jsonl", batch=args.batch)
    ans_rev = run_condition("reversed", chooser, jobs_rev, args.cache_dir / f"reversed_{tag}.jsonl", batch=args.batch)
    ans_blind = run_condition("blind", chooser, jobs_blind, args.cache_dir / f"blind_{tag}.jsonl", batch=args.batch)
    ans_probe = run_condition("probe", chooser, jobs_probe, args.cache_dir / f"probe_{tag}.jsonl", batch=args.batch)
    chooser.unload()

    # read the answers; the letter histogram per condition is the mechanism evidence -- a model
    # that reads content spreads its letters, a model that reads position piles them on a few
    # slots -- and it is an aggregate over letters, naming no song or label
    chosen_main, chosen_rev, chosen_blind = {}, {}, {}
    abstain = {"main": 0, "reversed": 0, "blind": 0}
    histogram = {name: {letter: 0 for letter in LETTERS[:K]} | {"abstain": 0} for name in abstain}
    for q in sub:
        letter_of, letter_rev = maps[int(q)]
        for name, ans, mp, store in (("main", ans_main, letter_of, chosen_main),
                                     ("reversed", ans_rev, letter_rev, chosen_rev),
                                     ("blind", ans_blind, letter_of, chosen_blind)):
            letter = parse_letter(ans[int(q)])
            if letter is None:
                abstain[name] += 1
                histogram[name]["abstain"] += 1
                store[int(q)] = None
            else:
                histogram[name][letter] += 1
                store[int(q)] = mp[letter]
    # the true label's slot is uniform by construction (the shuffle), so a position reader cannot
    # beat 1/K on it; recorded so the reader can check that the shuffle did its job
    true_slot = {letter: 0 for letter in LETTERS[:K]}
    for q in sub:
        letter_of, _ = maps[int(q)]
        for letter, label in letter_of.items():
            if label == li[q]:
                true_slot[letter] += 1

    reranked = scores.copy()
    for q in sub:
        reranked[q] = rerank(scores[q], cands[q], chosen_main[int(q)])
    rr_base_all = 1.0 / ranks_of(scores, li)
    rr_new_all = 1.0 / ranks_of(reranked, li)
    mask = np.zeros(n, dtype=bool)
    mask[sub] = True
    rr = {"retriever": rr_base_all, "llm_reranked": rr_new_all}
    contrast = paired_group_bootstrap(rr, w, gi, mask, [("llm_reranked", "retriever")])[0]

    true_sub = li[sub]
    correct_main = np.asarray([chosen_main[int(q)] == li[q] for q in sub])
    correct_blind = np.asarray([chosen_blind[int(q)] == li[q] for q in sub])
    retr_top1 = np.asarray([cands[q][0] == li[q] for q in sub])
    agree = np.asarray([chosen_main[int(q)] is not None and chosen_main[int(q)] == chosen_rev[int(q)] for q in sub])

    # the memorisation probe: normalised exact match against the credited label string
    probe_rate = None
    if labels_text is not None:
        names = {i: normalise_name(str(t)) for i, t in enumerate(labels_text)}
        hits = [normalise_name(ans_probe[int(q)]) == names[int(li[q])] and names[int(li[q])] != "" for q in sub]
        probe_rate = float(np.mean(hits))

    bands = {}
    spl = np.bincount(li, minlength=L)
    for name, lo, hi in BANDS:
        here = np.asarray([(spl[li[q]] >= lo) and (spl[li[q]] <= hi) for q in sub])
        if here.sum() == 0:
            continue
        bands[name] = {"queries": int(here.sum()),
                       "retriever_mrr_query_weighted": round(float(rr_base_all[sub][here].mean()), 4),
                       "reranked_mrr_query_weighted": round(float(rr_new_all[sub][here].mean()), 4),
                       "llm_top1_correct": round(float(correct_main[here].mean()), 4),
                       "retriever_top1_correct": round(float(retr_top1[here].mean()), 4)}

    agreement = float(agree.mean())
    blind_acc = float(correct_blind.mean())
    void = []
    if agreement < AGREEMENT_FLOOR:
        void.append(f"C1: agreement with the reversed order is {agreement:.3f}, below {AGREEMENT_FLOOR}: the model reads position")
    if blind_acc > BLIND_CEILING:
        void.append(f"C2: blind accuracy is {blind_acc:.3f}, above {BLIND_CEILING}: the presentation leaks the answer")
    if void:
        verdict = "R1 void: " + "; ".join(void)
    elif contrast["ci95"][0] > 0:
        verdict = "R1 improves: the LLM re-ranking raises MRR over the retriever, whole interval above zero"
    elif contrast["ci95"][1] < MARGIN:
        verdict = f"R1 no gain: the whole interval lies below +{MARGIN}"
    else:
        verdict = "R1 undecided"

    out = {"analysis": "llm_rerank_v3",
           "question": "does a local instruction-tuned LLM improve on the protocol's retriever when it chooses among the top candidates?",
           "corpus": {"content_sha256": V3_CONTENT_SHA256, "songs": n, "labels": L},
           "model": model_record,
           "design": {"retriever": "raw jieba word TF-IDF prototype, leave-group-out", "k": K,
                      "exemplars_per_candidate": EXEMPLARS, "truncate_characters": TRUNCATE,
                      "subset": int(len(sub)), "subset_seed": SUBSET_SEED, "exemplar_seed": EXEMPLAR_SEED,
                      "candidates_shown_as": "letters, shuffled per query; artist names never appear",
                      "prompt_template": TEMPLATE, "candidate_block": CANDIDATE_BLOCK,
                      "probe_template": PROBE_TEMPLATE, "placeholder": PLACEHOLDER,
                      "first_prompt_tokens": int(sample_tokens),
                      "reading_rule": ("C1 and C2 first; then MRR(reranked) - MRR(retriever), component-weighted, "
                                       "paired group bootstrap: improves if the whole interval is above zero, no gain "
                                       f"if the whole interval is below +{MARGIN}, undecided otherwise"),
                      "controls": {"C1_agreement_floor": AGREEMENT_FLOOR, "C2_blind_ceiling": BLIND_CEILING}},
           "checks": {"retriever_mrr": {"expected": EXPECTED_RETRIEVER, "recomputed": round(got, 4)}},
           "subset": {"queries": int(len(sub)), "true_label_in_top_k": round(float(in_top.mean()), 4),
                      "abstained": abstain, "abstained_for_out_of_memory": int(chooser.oom_abstains),
                      "cuda_failures_recovered_or_abstained": int(chooser.cuda_failures)},
           "results": {"retriever_mrr_query_weighted": round(float(rr_base_all[sub].mean()), 4),
                       "reranked_mrr_query_weighted": round(float(rr_new_all[sub].mean()), 4),
                       "retriever_mrr_component_weighted": round(float(np.sum(rr_base_all[sub] * w[sub]) / np.sum(w[sub])), 4),
                       "reranked_mrr_component_weighted": round(float(np.sum(rr_new_all[sub] * w[sub]) / np.sum(w[sub])), 4),
                       "paired_contrast_component_weighted": contrast,
                       "recall_at_1": {"retriever": round(float(retr_top1.mean()), 4), "llm": round(float(correct_main.mean()), 4)},
                       "among_queries_with_true_label_in_top_k": {
                           "queries": int(in_top.sum()),
                           "retriever_top1_correct": round(float(retr_top1[in_top].mean()), 4),
                           "llm_choice_correct": round(float(correct_main[in_top].mean()), 4)},
                       "by_band_songs_per_label": bands},
           "controls": {"C1_reversed_order_agreement": round(agreement, 4),
                        "C2_blind_accuracy": round(blind_acc, 4), "C2_chance": round(1.0 / K, 4),
                        "C3_memorisation_probe_exact_match_rate": None if probe_rate is None else round(probe_rate, 4),
                        "letter_histogram_by_condition": histogram,
                        "true_label_slot_histogram_main": true_slot},
           "reading": verdict,
           "minutes": round((time.time() - started) / 60.0, 1),
           "privacy": ("aggregate only: no lyric, song, label or artist name; the prompt template is recorded, "
                       "the prompts themselves and the model's answers stay in the private cache")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / OUT_NAME
    path.write_bytes((json.dumps(out, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"\n{verdict}\nC1 agreement {agreement:.3f}, C2 blind {blind_acc:.3f}, C3 probe {probe_rate}", flush=True)
    print(f"wrote {path} in {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
