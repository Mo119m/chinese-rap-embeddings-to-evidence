#!/usr/bin/env python3
"""Teach the encoder who wrote it: contrastive fine-tuning of BGE-M3 on corpus v3.

Everything so far took BGE-M3 as it came and read its vectors with cosine or a linear
map. This trains the encoder on the corpus itself, with an objective that says nothing
about meaning and everything about authorship: two stanzas by one label should sit close,
two stanzas by different labels should not. The design follows what the training-data
audit measured, and each decision below names the fact that forced it.

  unit            the chunk -- a stanza -- because the audit found the recorded 2,048-token
                  budget never reached; chunks are truncated at --max-length tokens
  positive        another chunk by the same label from a DIFFERENT leakage group, so the
                  model cannot succeed by recognising a repeated hook (LUAR's episode idea,
                  Rivera-Soto et al. 2021, with the corpus's own leakage unit)
  negatives       every other label in the batch, plus --hard-negatives chunks per anchor
                  drawn from the anchor's nearest neighbours by BGE-M3 cosine among OTHER
                  labels: content-controlled negatives in the sense of Wegmann et al.
                  2022, which force style over subject; and, with --queue-size, a
                  cross-batch memory of recent embeddings (XBM, Wang et al. 2020), because
                  eight anchors give eight labels per step where LUAR saw 128 authors --
                  the first run (fold 0, no queue) lost 0.016 on the test fold and 0.054 on
                  unseen labels against the frozen model
  masking         the label string is removed from every chunk, train and test, because
                  7.5% of chunks name their own artist and that is the first thing a
                  model would learn
  exclusions      songs whose title marks a collaboration are not used as anchors or
                  positives, because their verses may be someone else's
  splits          the five leakage-group folds every experiment shares: train on four,
                  test on --test-fold; and --held-out-label-share of labels are removed
                  from training entirely and evaluated as unseen authors
  loss            InfoNCE over cosine similarities at temperature --temperature, symmetric
                  anchor <-> positive, with the in-batch and hard negatives in the
                  denominator
  parameters      LoRA on the attention projections (peft), base weights frozen, so the
                  run fits an 8 GB card and the result is a small delta on the recorded
                  checkpoint
  batch size      eight anchors per step is what fits the card with gradients. Two ways of
                  putting more labels into the denominator failed. A queue of the live
                  encoder's recent vectors (XBM) collapsed to one point. A queue filled by a
                  momentum encoder contracted the space to a participation ratio of 7, and
                  that design was flawed besides: MoCo (He et al. 2020) compares queries with
                  keys that all come from the momentum encoder, while here the positives came
                  from the live encoder and only the queue from the momentum encoder, so the
                  loss could fall by moving live vectors away from the queue. Both options are
                  kept only to reproduce the reported runs. --grad-cache-chunk replaces them
                  with an exact large batch (GradCache, Gao et al. 2021): every vector in the
                  denominator comes from the current encoder, the gradient is computed in
                  chunks that fit the card, and --check-grad-cache compares the chunked
                  gradient with the direct one before a run is trusted
  geometry        every 50 steps a fixed probe of 512 training chunks is encoded in eval mode
                  and its participation ratio and mean pairwise cosine are recorded; the
                  batch cosine of the first runs, taken in train mode over one small batch,
                  read 0.8 while the finished space sat at 0.94

Evaluation is the protocol every number in this project uses: song = mean of its chunk
vectors, label profile = leave-group-out weighted mean, cosine rank of the true label
among the 226. The frozen model is scored on the same masked text under the same protocol
so the comparison is one variable. Aggregate metrics are public; vectors are private.

    python src/train_identity_encoder_v3.py --private-root <ni-k> --test-fold 0
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from build_downstream_retrieval_v2 import MINIMUM_SONGS_PER_LABEL, build_songs  # noqa: E402
from corpus_v3 import V3_CONTENT_SHA256, load_v3  # noqa: E402
from identity_probe_v2 import FOLDS, SEED, dense_leave_group_out, unit_rows  # noqa: E402
from identity_spaces_v2 import paired_group_bootstrap  # noqa: E402
from leakage_groups_v2 import build_groups, normalise_document  # noqa: E402
from training_data_audit_v3 import FEAT, LATIN  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "results" / "retrieval-v3"
MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def mask_label(text: str, label: str) -> str:
    label = label.strip()
    if len(label) < 2:
        return text
    if LATIN.match(label):
        return re.sub(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])", " ", text, flags=re.I)
    return text.replace(label, " ")


def encode_all(model, tokenizer, texts, device, max_length, batch_size=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                hidden = model(**batch).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def geometry(vectors: np.ndarray, seed: int = 0) -> dict:
    """Mean pairwise cosine (over a seeded sample of at most 4,000 rows) and the participation
    ratio of the centred covariance, (sum of eigenvalues)^2 / sum of squared eigenvalues."""
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    sample = unit[np.random.default_rng(seed).choice(len(unit), size=min(4000, len(unit)), replace=False)]
    sims = sample @ sample.T
    off = sims[~np.eye(len(sims), dtype=bool)]
    values = np.clip(np.linalg.eigvalsh(np.cov((unit - unit.mean(axis=0)).T)), 0, None)
    return {"mean_pairwise_cosine": round(float(off.mean()), 4),
            "participation_ratio": round(float(values.sum() ** 2 / (values ** 2).sum()), 1)}


def contrastive_loss(emb, n, labels_a, labels_all, queue_e, queue_labels, temperature):
    """Symmetric InfoNCE over [anchors | positives | negatives], plus an optional queue.

    Every same-label column other than the assigned positive is masked out of the denominator."""
    import torch
    anchor_e, pos_e, neg_e = emb[:n], emb[n:2 * n], emb[2 * n:]
    arange = torch.arange(n, device=emb.device)
    logits = anchor_e @ torch.cat([pos_e, neg_e, queue_e]).T / temperature
    same = labels_a[:, None] == torch.cat([labels_all, queue_labels])[None, :]
    same[arange, arange] = False
    loss = torch.nn.functional.cross_entropy(logits.masked_fill(same, float("-inf")), arange)
    logits_b = pos_e @ torch.cat([anchor_e, neg_e, queue_e]).T / temperature
    same_b = labels_a[:, None] == torch.cat([labels_a, labels_all[n:], queue_labels])[None, :]
    same_b[arange, arange] = False
    loss_b = torch.nn.functional.cross_entropy(logits_b.masked_fill(same_b, float("-inf")), arange)
    return (loss + loss_b) / 2


def build(args) -> int:
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu:
        raise SystemExit("no CUDA device; training on CPU is not realistic (pass --allow-cpu to insist)")
    torch.manual_seed(SEED)

    print("loading corpus v3", flush=True)
    rows, vectors, state = load_v3(args.private_root, allow_interim_v2_vectors=True)
    chunks_by_song, label_by_song, components_by_song, documents, _ = build_songs(rows, vectors)
    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)
    long_enough = {s for s in chunks_by_song
                   if len(v1.normalized_text(documents[s])) >= v1.MIN_EFFECTIVE_CHARACTERS}
    eligible = sorted(l for l, m in songs_by_label.items()
                      if sum(1 for s in m if s in long_enough) >= MINIMUM_SONGS_PER_LABEL)
    songs = sorted(s for s in long_enough if label_by_song[s] in set(eligible))
    label_index = np.asarray([eligible.index(label_by_song[s]) for s in songs], dtype=np.int64)
    groups = build_groups(songs, components_by_song, {s: normalise_document(documents[s]) for s in songs})
    order = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    group_ids = np.asarray([order[groups[s]] for s in songs], dtype=np.int64)
    label_count = len(eligible)
    component_size = Counter(zip(group_ids.tolist(), label_index.tolist()))
    weights = np.asarray([1.0 / component_size[(int(g), int(l))] for g, l in zip(group_ids, label_index)])
    song_pos = {s: i for i, s in enumerate(songs)}
    fold = np.random.default_rng(SEED).integers(0, FOLDS, size=len(order))[group_ids]
    held = np.zeros(label_count, dtype=bool)
    held[np.random.default_rng(SEED).permutation(label_count)[:int(round(args.held_out_label_share * label_count))]] = True
    titles = {r["song_id"]: r["song_title"] for r in rows}
    print(f"  {len(songs):,} query songs, {label_count} labels, {len(order):,} groups; test fold {args.test_fold}; "
          f"{int(held.sum())} labels held out", flush=True)

    # chunk-level tables over the query songs, masked
    chunk_rows = [i for s in songs for i in sorted(chunks_by_song[s], key=lambda i: int(rows[i]["source_order"]))]
    chunk_song = np.asarray([song_pos[rows[i]["song_id"]] for i in chunk_rows])
    chunk_text = [mask_label(rows[i]["cleaned_text"], rows[i]["source_credit_label"]) for i in chunk_rows]
    chunk_label = label_index[chunk_song]
    chunk_group = group_ids[chunk_song]
    chunk_fold = fold[chunk_song]
    collaboration = np.asarray([bool(FEAT.search(titles[rows[i]["song_id"]])) for i in chunk_rows])
    trainable = (chunk_fold != args.test_fold) & ~held[chunk_label] & ~collaboration
    print(f"  {len(chunk_rows):,} chunks; {int(trainable.sum()):,} usable for training "
          f"({int(collaboration.sum()):,} excluded as collaborations)", flush=True)

    # hard negatives from the frozen space: nearest chunks of other labels, other groups
    print("mining content-controlled negatives from the frozen space", flush=True)
    base_vec = unit_rows(vectors[chunk_rows].astype(np.float32))
    train_idx = np.flatnonzero(trainable)
    hard = {}
    for start in range(0, len(train_idx), 512):
        block = train_idx[start:start + 512]
        sims = base_vec[block] @ base_vec[train_idx].T
        same_label = chunk_label[block][:, None] == chunk_label[train_idx][None, :]
        sims[same_label] = -np.inf
        top = np.argpartition(-sims, args.hard_negative_pool, axis=1)[:, :args.hard_negative_pool]
        for row, anchor in enumerate(block):
            hard[int(anchor)] = train_idx[top[row]].tolist()

    # positives: same label, different leakage group, trainable
    by_label_train: dict[int, list[int]] = defaultdict(list)
    for c in train_idx.tolist():
        by_label_train[int(chunk_label[c])].append(c)
    anchors = [c for c in train_idx.tolist()
               if any(chunk_group[o] != chunk_group[c] for o in by_label_train[int(chunk_label[c])])]
    print(f"  {len(anchors):,} anchors with a positive in another leakage group", flush=True)

    # model
    print(f"loading {MODEL_ID}@{MODEL_REVISION[:8]} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device)
    if device == "cuda":
        model.gradient_checkpointing_enable()
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
                      target_modules=["query", "key", "value", "dense"], bias="none")
    model = get_peft_model(model, lora)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters {trainable_params:,}", flush=True)

    # frozen baseline on the same masked text, same protocol
    print("scoring the frozen model on the masked text", flush=True)
    with model.disable_adapter():
        frozen = encode_all(model, tokenizer, chunk_text, device, args.max_length)

    def song_level(chunk_vectors):
        acc = np.zeros((len(songs), chunk_vectors.shape[1]))
        np.add.at(acc, chunk_song, chunk_vectors)
        return unit_rows(acc / np.bincount(chunk_song)[:, None])

    def evaluate(chunk_vectors, name):
        song_vec = song_level(chunk_vectors)
        test = np.flatnonzero((fold == args.test_fold) & ~held[label_index])
        unseen = np.flatnonzero(held[label_index])
        scores_test = dense_leave_group_out(song_vec, label_index, group_ids, weights, label_count, test)
        scores_unseen = dense_leave_group_out(song_vec, label_index, group_ids, weights, label_count, unseen)
        r_test = v1.rank_system(scores_test.astype(np.float32), label_index[test])[0]
        r_unseen = v1.rank_system(scores_unseen.astype(np.float32), label_index[unseen])[0]
        out = {"test_fold": {"queries": int(len(test)), "mrr": round(float(np.mean(1.0 / r_test)), 4),
                             "recall_at_1": round(float(np.mean(r_test <= 1)), 4),
                             "recall_at_10": round(float(np.mean(r_test <= 10)), 4)},
               "unseen_labels": {"queries": int(len(unseen)), "mrr": round(float(np.mean(1.0 / r_unseen)), 4),
                                 "recall_at_10": round(float(np.mean(r_unseen <= 10)), 4)}}
        print(f"  {name:22s} test-fold MRR {out['test_fold']['mrr']:.4f}  R@10 {out['test_fold']['recall_at_10']:.4f}  |  "
              f"unseen-label MRR {out['unseen_labels']['mrr']:.4f}", flush=True)
        full = np.ones(len(songs))
        full[test] = 1.0 / r_test
        full[unseen] = 1.0 / r_unseen
        return out, full, test, unseen

    results = {}
    results["frozen_masked"], rr_frozen, test, unseen = evaluate(frozen, "frozen, masked text")

    # training
    print("training", flush=True)
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(anchors) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs if not args.dry_run else 20
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / max(total_steps, 1)))
    rng = np.random.default_rng(SEED + args.test_fold)
    model.train()
    step = 0
    losses = []
    started = time.time()
    # cross-batch memory: a fixed-size queue of recent embeddings and their labels. Eight
    # anchors give eight labels per step; LUAR saw 128 authors per batch. The queue puts the
    # last --queue-size chunks of every label into the denominator at no memory cost beyond
    # the vectors themselves (they carry no gradient, so they are slightly stale; XBM).
    queue_e = torch.zeros((0, 1024), device=device)
    queue_labels = torch.zeros((0,), dtype=torch.long, device=device)
    # Momentum encoder for the queue (MoCo, He et al. 2020): the first queue run collapsed
    # to a single point (mean pairwise cosine 1.000, loss at ln K) because the queue held
    # vectors of an encoder that no longer existed. With --momentum m > 0 the queue is
    # filled by an exponential moving average of the LoRA weights, which drifts slowly
    # enough for its old entries to stay comparable. The EMA is a second copy of the
    # trainable parameters only; encoding with it swaps the weights in and out.
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ema = {n: p.detach().clone() for n, p in trainable_named} if args.momentum > 0 else None

    def encode_with_ema(encoded):
        live = {n: p.detach().clone() for n, p in trainable_named}
        with torch.no_grad():
            for n, p in trainable_named:
                p.copy_(ema[n])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                keys = torch.nn.functional.normalize(model(**encoded).last_hidden_state[:, 0].float(), dim=-1)
            for n, p in trainable_named:
                p.copy_(live[n])
        return keys

    autocast = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")

    def grad_cache_backward(texts, n, labels_a, labels_all):
        """GradCache (Gao et al. 2021): the loss over the whole batch, its gradient taken with
        respect to the vectors, then pushed through the encoder chunk by chunk. The RNG state
        is restored before each chunk's second pass so dropout draws the same masks; the
        largest change of a vector between the two passes is returned as the check."""
        chunk = args.grad_cache_chunk
        encs = [tokenizer(texts[i:i + chunk], padding=True, truncation=True, max_length=args.max_length,
                          return_tensors="pt").to(device) for i in range(0, len(texts), chunk)]
        states, reps = [], []
        with torch.no_grad():
            for e in encs:
                states.append((torch.get_rng_state(), torch.cuda.get_rng_state() if device == "cuda" else None))
                with torch.autocast(**autocast):
                    hidden = model(**e).last_hidden_state[:, 0]
                reps.append(torch.nn.functional.normalize(hidden.float(), dim=-1))
        cached = torch.cat(reps).requires_grad_(True)
        empty_e = torch.zeros((0, cached.shape[1]), device=device)
        empty_l = torch.zeros((0,), dtype=torch.long, device=device)
        loss = contrastive_loss(cached, n, labels_a, labels_all, empty_e, empty_l, args.temperature)
        loss.backward()
        drift = 0.0
        for e, (cpu_state, cuda_state), grad, rep in zip(encs, states, cached.grad.split([len(r) for r in reps]), reps):
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)
            with torch.autocast(**autocast):
                hidden = model(**e).last_hidden_state[:, 0]
            live = torch.nn.functional.normalize(hidden.float(), dim=-1)
            drift = max(drift, float((live.detach() - rep).abs().max()))
            torch.dot(live.flatten(), grad.flatten()).backward()
        return loss.detach(), cached.detach(), drift

    def sample_batch(batch_anchor, generator):
        batch_pos, batch_neg = [], []
        for a in batch_anchor:
            candidates = [o for o in by_label_train[int(chunk_label[a])] if chunk_group[o] != chunk_group[a]]
            batch_pos.append(int(generator.choice(candidates)))
            pool = [x for x in hard[a] if chunk_group[x] != chunk_group[a]]
            batch_neg.extend(int(x) for x in generator.choice(pool, size=min(args.hard_negatives, len(pool)), replace=False))
        return batch_pos, batch_neg

    if args.check_grad_cache:
        if args.grad_cache_chunk <= 0:
            raise SystemExit("--check-grad-cache needs --grad-cache-chunk")
        batch_anchor = anchors[:args.batch_size]
        batch_pos, batch_neg = sample_batch(batch_anchor, np.random.default_rng(SEED))
        texts = [chunk_text[i] for i in batch_anchor + batch_pos + batch_neg]
        n = len(batch_anchor)
        labels_a = torch.as_tensor(chunk_label[batch_anchor], device=device)
        labels_all = torch.as_tensor(np.concatenate([chunk_label[batch_pos], chunk_label[batch_neg]]), device=device)
        params = [p for _, p in trainable_named]

        def flat_grad():
            return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
                              for p in params])
        # 1. dropout off, gradient checkpointing on (train mode): chunked gradient against direct
        dropouts = [(m, m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        for m, _ in dropouts:
            m.p = 0.0
        optimiser.zero_grad(set_to_none=True)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
        with torch.autocast(**autocast):
            hidden = model(**enc).last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(hidden.float(), dim=-1)
        loss_direct = contrastive_loss(emb, n, labels_a, labels_all, queue_e, queue_labels, args.temperature)
        loss_direct.backward()
        direct = flat_grad()
        optimiser.zero_grad(set_to_none=True)
        loss_cached, _, drift_off = grad_cache_backward(texts, n, labels_a, labels_all)
        cached_grad = flat_grad()
        for m, p in dropouts:
            m.p = p
        # 2. dropout on: the second pass must reproduce the first pass's vectors
        optimiser.zero_grad(set_to_none=True)
        _, _, drift_on = grad_cache_backward(texts, n, labels_a, labels_all)
        optimiser.zero_grad(set_to_none=True)
        report = {"sequences": len(texts), "chunk": args.grad_cache_chunk,
                  "loss_direct": round(float(loss_direct), 6), "loss_cached": round(float(loss_cached), 6),
                  "gradient_cosine": round(float(torch.nn.functional.cosine_similarity(direct, cached_grad, dim=0)), 6),
                  "gradient_relative_difference": round(float((direct - cached_grad).norm() / direct.norm()), 6),
                  "gradient_norm_direct": round(float(direct.norm()), 6),
                  "vector_drift_dropout_off": round(drift_off, 6), "vector_drift_dropout_on": round(drift_on, 6)}
        print(f"  GradCache check: {report}", flush=True)
        passed = (report["gradient_cosine"] > 0.999 and report["gradient_relative_difference"] < 0.05
                  and report["vector_drift_dropout_on"] < 0.02)
        (args.out_dir / f"identity_encoder_grad_cache_check{args.tag}.json").write_text(
            json.dumps({"check": report, "passed": passed}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
        if not passed:
            raise SystemExit("GradCache check failed; do not train with it")
        print("  GradCache check passed", flush=True)
        return 0

    if args.grad_cache_chunk > 0 and args.queue_size > 0:
        raise SystemExit("GradCache replaces the queue; use one or the other")
    probe_idx = np.random.default_rng(SEED).choice(train_idx, size=min(512, len(train_idx)), replace=False)
    probe_text = [chunk_text[i] for i in probe_idx]
    probe_geometry = [dict(step=0, **geometry(encode_all(model, tokenizer, probe_text, device, args.max_length, 32)))]
    model.train()
    print(f"  probe at step 0: {probe_geometry[0]}", flush=True)
    rep_drift = []

    cohesion = []   # mean pairwise cosine of the batch embeddings, every 50 steps: the collapse guard
    for epoch in range(args.epochs):
        rng.shuffle(anchors)
        for start in range(0, len(anchors), args.batch_size):
            if step >= total_steps:
                break
            batch_anchor = anchors[start:start + args.batch_size]
            batch_pos, batch_neg = [], []
            for a in batch_anchor:
                candidates = [o for o in by_label_train[int(chunk_label[a])] if chunk_group[o] != chunk_group[a]]
                batch_pos.append(int(rng.choice(candidates)))
                pool = [n for n in hard[a] if chunk_group[n] != chunk_group[a]]
                batch_neg.extend(int(x) for x in rng.choice(pool, size=min(args.hard_negatives, len(pool)), replace=False))
            texts = [chunk_text[i] for i in batch_anchor + batch_pos + batch_neg]
            n = len(batch_anchor)
            labels_a = torch.as_tensor(chunk_label[batch_anchor], device=device)
            labels_all = torch.as_tensor(np.concatenate([chunk_label[batch_pos], chunk_label[batch_neg]]), device=device)
            optimiser.zero_grad(set_to_none=True)
            if args.grad_cache_chunk > 0:
                loss, emb, drift = grad_cache_backward(texts, n, labels_a, labels_all)
                rep_drift.append(drift)
                if drift > 0.02:
                    raise SystemExit(f"the GradCache second pass differs from the first by {drift:.4f} at step {step}; "
                                     "the dropout masks were not reproduced")
            else:
                enc = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    hidden = model(**enc).last_hidden_state[:, 0]
                emb = torch.nn.functional.normalize(hidden.float(), dim=-1)
                loss = contrastive_loss(emb, n, labels_a, labels_all, queue_e, queue_labels, args.temperature)
                if args.queue_size > 0:
                    keys = encode_with_ema(enc) if ema is not None else emb.detach()
                    queue_e = torch.cat([queue_e, keys])[-args.queue_size:]
                    queue_labels = torch.cat([queue_labels, torch.cat([labels_a, labels_all])])[-args.queue_size:]
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            losses.append(float(loss))
            if ema is not None:
                with torch.no_grad():
                    for n, p in trainable_named:
                        ema[n].mul_(args.momentum).add_(p.detach(), alpha=1.0 - args.momentum)
            step += 1
            if step % 50 == 0 or step == total_steps:
                with torch.no_grad():
                    sims = emb.detach() @ emb.detach().T
                    off = sims[~torch.eye(len(sims), dtype=torch.bool, device=sims.device)]
                    cohesion.append(round(float(off.mean()), 4))
                probe = dict(step=step, **geometry(encode_all(model, tokenizer, probe_text, device, args.max_length, 32)))
                model.train()
                probe_geometry.append(probe)
                print(f"  step {step}/{total_steps}  loss {np.mean(losses[-50:]):.4f}  batch cosine {cohesion[-1]:.3f}  "
                      f"probe PR {probe['participation_ratio']:.1f} cos {probe['mean_pairwise_cosine']:.3f}  "
                      f"{(time.time() - started) / 60:.1f} min", flush=True)
                if len(cohesion) >= 2 and min(cohesion[-2:]) > args.collapse_guard:
                    raise SystemExit(f"representation collapsed: batch cosine {cohesion[-2:]} above {args.collapse_guard} "
                                     f"at step {step}; stopping rather than scoring a degenerate space")
                if (args.min_participation > 0 and len(probe_geometry) >= 3
                        and max(p["participation_ratio"] for p in probe_geometry[-2:]) < args.min_participation):
                    raise SystemExit(f"dimensional collapse: probe participation ratio below {args.min_participation} "
                                     f"at two checks, step {step}; stopping rather than scoring a degenerate space")
        if step >= total_steps:
            break

    print("scoring the fine-tuned model", flush=True)
    tuned = encode_all(model, tokenizer, chunk_text, device, args.max_length)
    results["fine_tuned"], rr_tuned, _, _ = evaluate(tuned, "fine-tuned")
    mask_test = np.zeros(len(songs), dtype=bool); mask_test[test] = True
    mask_unseen = np.zeros(len(songs), dtype=bool); mask_unseen[unseen] = True
    contrasts = {
        "test_fold": paired_group_bootstrap({"fine_tuned": rr_tuned, "frozen_masked": rr_frozen}, weights, group_ids,
                                            mask_test, [("fine_tuned", "frozen_masked")]),
        "unseen_labels": paired_group_bootstrap({"fine_tuned": rr_tuned, "frozen_masked": rr_frozen}, weights, group_ids,
                                                mask_unseen, [("fine_tuned", "frozen_masked")]),
    }
    for scope, cs in contrasts.items():
        for c in cs:
            print(f"  {scope}: fine-tuned - frozen {c['mrr_difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]", flush=True)

    private = args.private_root / "work" / "private-identity-encoder-v3"
    private.mkdir(parents=True, exist_ok=True)
    np.save(private / f"fine_tuned_chunk_vectors_fold{args.test_fold}{args.tag}.npy", tuned.astype(np.float32))
    np.save(private / f"frozen_masked_chunk_vectors_fold{args.test_fold}.npy", frozen.astype(np.float32))
    model.save_pretrained(private / f"lora_fold{args.test_fold}{args.tag}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "contrastive fine-tuning of BGE-M3 for label identity on corpus v3",
        "corpus": {"content_sha256": V3_CONTENT_SHA256, "queries": len(songs), "labels": label_count, "groups": len(order)},
        "design": {"unit": "chunk (stanza)", "max_length_tokens": args.max_length,
                   "positive": "same label, different leakage group", "hard_negatives_per_anchor": args.hard_negatives,
                   "hard_negative_pool": f"{args.hard_negative_pool} nearest other-label chunks in the frozen space ({state['vectors']} vectors)",
                   "label_string_masked_in_all_text": True, "collaboration_titles_excluded_from_training": int(collaboration.sum()),
                   "test_fold": args.test_fold, "held_out_labels": int(held.sum()), "epochs": args.epochs, "steps": step,
                   "batch_anchors": args.batch_size, "temperature": args.temperature, "learning_rate": args.learning_rate,
                   "lora_rank": args.lora_rank, "trainable_parameters": trainable_params,
                   "queue_size": args.queue_size, "momentum": args.momentum, "collapse_guard": args.collapse_guard,
                   "batch_cosine_every_50_steps": cohesion, "tag": args.tag,
                   "grad_cache_chunk": args.grad_cache_chunk,
                   "grad_cache_max_vector_drift": round(max(rep_drift), 6) if rep_drift else None,
                   "probe_geometry_every_50_steps": probe_geometry,
                   "min_participation_guard": args.min_participation,
                   "model": f"{MODEL_ID}@{MODEL_REVISION}", "device": device, "dry_run": args.dry_run},
        "training_loss": {"first_50_mean": round(float(np.mean(losses[:50])), 4) if losses else None,
                          "last_50_mean": round(float(np.mean(losses[-50:])), 4) if losses else None},
        "results": results,
        "geometry": {"frozen_masked": geometry(frozen), "fine_tuned": geometry(tuned),
                     "median_cosine_of_each_chunk_to_its_frozen_vector": round(float(np.median(
                         np.sum(unit_rows(tuned) * unit_rows(frozen), axis=1))), 4)},
        "paired_contrasts": contrasts,
        "reading": ("the frozen row is BGE-M3 on the same masked text under the same protocol, so the difference "
                    "is what the corpus taught the encoder; the unseen-label row is whether what it learned is "
                    "identity in general or these authors in particular"),
        "privacy": "aggregate only; vectors and adapter weights are private",
    }
    (args.out_dir / f"identity_encoder_fold{args.test_fold}{args.tag}{'_dryrun' if args.dry_run else ''}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    print(f"\nwrote {args.out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--held-out-label-share", type=float, default=0.15)
    # the audit: median chunk 76 tokens, 90th percentile 600; 512 keeps 85% of chunks whole
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8, help="anchors per step")
    parser.add_argument("--hard-negatives", type=int, default=2, help="per anchor")
    parser.add_argument("--hard-negative-pool", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--queue-size", type=int, default=0,
                        help="cross-batch memory of recent chunk embeddings used as extra negatives "
                             "(Wang et al. 2020, XBM); 0 keeps only in-batch and hard negatives")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="EMA coefficient of a momentum encoder that fills the queue (MoCo); 0 = the live encoder")
    parser.add_argument("--collapse-guard", type=float, default=0.98,
                        help="stop when the batch's mean pairwise cosine stays above this at two consecutive checks")
    parser.add_argument("--grad-cache-chunk", type=int, default=0,
                        help="GradCache: sequences per chunk of an exact large batch (Gao et al. 2021); 0 = off")
    parser.add_argument("--check-grad-cache", action="store_true",
                        help="compare the GradCache gradient with the direct one on one batch, write the check, exit")
    parser.add_argument("--min-participation", type=float, default=0.0,
                        help="stop when the eval-mode probe's participation ratio stays below this at two checks; 0 = off")
    parser.add_argument("--tag", default="", help="suffix for this run's output names, e.g. _queue4096")
    parser.add_argument("--dry-run", action="store_true", help="20 steps, to prove the pipeline")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    args.private_root = args.private_root.resolve()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
