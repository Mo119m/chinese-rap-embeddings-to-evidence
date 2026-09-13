#!/usr/bin/env python3
"""Check that a resumed fine-tune is as good as an uninterrupted one, against the noise floor.

train_identity_encoder_v3.py saves LoRA weights, optimiser, schedule, data order and every RNG
state with --checkpoint-every and continues from them with --resume. Training on the GPU is
not bit-reproducible, so "resumed equals uninterrupted" cannot be tested for equality. This
runs three 20-step dry runs with the real code path (GradCache chunks, checkpoints):

  B   20 steps straight
  D   20 steps straight again: B against D is the run-to-run noise floor
  C   10 steps, exit, --resume to 20: B against C is uninterrupted against resumed

and compares the step-20 checkpoints. lora_B starts at zero, so its norm is the learned update;
Adam's moments would differ grossly if the optimiser state were not restored. Resume passes when
B-C lies within the spread of B-D (at most 1.5 times it) and the data order and every RNG state
agree exactly. The report is written to results/retrieval-v3/identity_encoder_resume_check.json;
the test checkpoints are deleted afterwards.

    python tools/check_trainer_resume_v3.py --private-root <ni-k>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "retrieval-v3" / "identity_encoder_resume_check.json"
COMMON = ["--test-fold", "0", "--dry-run", "--batch-size", "8", "--grad-cache-chunk", "12", "--checkpoint-every", "10"]


def run(private_root: Path, extra: list[str], log: Path) -> None:
    cmd = [sys.executable, str(ROOT / "src" / "train_identity_encoder_v3.py"), "--private-root", str(private_root)]
    env = dict(os.environ, CHINESE_RAP_CORPUS="v3", PYTHONIOENCODING="utf-8")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(cmd + COMMON + extra, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(extra)} failed; see {log}")


def relative(x: dict, y: dict, keys) -> float:
    num = sum(float((x[k] - y[k]).float().norm() ** 2) for k in keys) ** 0.5
    den = sum(float(x[k].float().norm() ** 2) for k in keys) ** 0.5
    return num / den


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path(os.environ.get("TEMP", "/tmp")))
    args = parser.parse_args()
    import torch

    private = args.private_root.resolve() / "work" / "private-identity-encoder-v3"
    dirs = {tag: private / f"checkpoint_fold0_resumecheck_{tag}_dryrun" for tag in ("b", "c", "d")}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
    run(args.private_root, ["--stop-after-step", "20", "--tag", "_resumecheck_b"], args.log_dir / "resumecheck_b.log")
    run(args.private_root, ["--stop-after-step", "20", "--tag", "_resumecheck_d"], args.log_dir / "resumecheck_d.log")
    run(args.private_root, ["--stop-after-step", "10", "--tag", "_resumecheck_c"], args.log_dir / "resumecheck_c1.log")
    run(args.private_root, ["--stop-after-step", "20", "--resume", "--tag", "_resumecheck_c"], args.log_dir / "resumecheck_c2.log")

    states = {tag: torch.load(d / "state.pt", map_location="cpu", weights_only=False) for tag, d in dirs.items()}
    b = states["b"]
    keys_b = [k for k in b["trainable"] if "lora_B" in k]
    keys_a = [k for k in b["trainable"] if "lora_A" in k]
    idx = list(b["optimiser"]["state"])
    report = {}
    for name, other in (("uninterrupted_vs_uninterrupted", states["d"]), ("uninterrupted_vs_resumed", states["c"])):
        lb, lo = np.asarray(b["losses"]), np.asarray(other["losses"])
        ob, oo = b["optimiser"]["state"], other["optimiser"]["state"]
        report[name] = {
            "resumed_at": other["resumed_at"],
            "loss_max_abs_diff_steps_1_10": round(float(np.abs(lb[:10] - lo[:10]).max()), 5),
            "loss_max_abs_diff_steps_11_20": round(float(np.abs(lb[10:] - lo[10:]).max()), 5),
            "lora_B_relative_diff": round(relative(b["trainable"], other["trainable"], keys_b), 5),
            "lora_A_relative_diff": round(relative(b["trainable"], other["trainable"], keys_a), 6),
            "adam_exp_avg_relative_diff": round(relative({i: ob[i]["exp_avg"] for i in idx}, {i: oo[i]["exp_avg"] for i in idx}, idx), 5),
            "adam_exp_avg_sq_relative_diff": round(relative({i: ob[i]["exp_avg_sq"] for i in idx}, {i: oo[i]["exp_avg_sq"] for i in idx}, idx), 5),
            "data_order_equal": other["anchors"] == b["anchors"],
            "numpy_rng_equal": other["numpy_rng"] == b["numpy_rng"],
            "torch_rng_equal": bool(torch.equal(other["torch_rng"], b["torch_rng"])),
            "cuda_rng_equal": bool(other["cuda_rng"] is None or torch.equal(other["cuda_rng"], b["cuda_rng"])),
            "schedule_equal": other["schedule"] == b["schedule"],
        }
    noise, resumed = report["uninterrupted_vs_uninterrupted"], report["uninterrupted_vs_resumed"]
    exact = all(resumed[k] for k in ("data_order_equal", "numpy_rng_equal", "torch_rng_equal", "cuda_rng_equal", "schedule_equal"))
    within = all(resumed[k] <= 1.5 * noise[k] for k in ("lora_B_relative_diff", "adam_exp_avg_relative_diff"))
    payload = {
        "analysis": "resume of train_identity_encoder_v3 against run-to-run nondeterminism",
        "design": {"runs": "B and D uninterrupted 20 steps; C stopped at 10 and resumed to 20",
                   "arguments": COMMON, "pass_rule": "data order and RNG states equal, and the resumed lora_B and "
                   "Adam first-moment differences at most 1.5 times the uninterrupted pair's"},
        "comparisons": report,
        "passed": bool(exact and within),
        "privacy": "aggregate only",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
