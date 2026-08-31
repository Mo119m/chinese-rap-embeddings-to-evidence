#!/usr/bin/env python3
"""Rebuild the model artifacts from the private inputs and compare against what is published.

The three model results carry the paper's headline numbers -- retrieval MRR and Recall@10,
written-ending Top-3, the repertoire graph's edge count -- and until 31 August 2026 none of
them had ever been rebuilt from their inputs. They also record neither their inputs nor the
code that produced them, so there was nothing to go stale and nothing to check: a reader
could not tell whether the published numbers still followed from the committed builders.

This makes that check repeatable. It builds each artifact in a sandbox and compares file
bytes and every leaf value in the summary against the committed copy.

The sandbox matters more than it looks. Two of the builders write beside their inputs, so a
naive setup points them at the author's private tree; both refused to clear a directory they
did not expect, which is how that was caught. Here the private inputs are linked read-only
and the build directories are ordinary directories inside the sandbox, so a build cannot
reach the private tree even if a guard were missing.

Differences are expected and are classified rather than counted:

* a timestamp differs on every rebuild and is inert;
* a recorded builder hash may name a superseded version, which says the artifact predates a
  code change and needs checking against what that change did;
* anything else is a real divergence.

    python tools/verify_model_reproducibility.py --private-root <ni-k> --sandbox <dir>

Needs the private corpus, the model dependencies from requirements.txt, and roughly half an
hour. It is not part of CI for those reasons; it is the procedure to rerun whenever a model
builder changes or a result is questioned.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories the builders read. Linked read-only so a build cannot write through them.
READ_ONLY = [
    ("work/private-chinese-rap-lyrical-repertoire-graph-v2", True),
    ("work/private-canonical-lyric-text-sidecar-v1", True),
    ("work/private-canonical-clean-text-embeddings-v1", True),
    ("work/external_sources", False),
    ("outputs/chinese-rap-lyrical-repertoire-graph-v2", True),
    ("outputs/chinese-rap-encoder-sanity-benchmark-v1", True),
]
# The rhyme builder clears and rewrites its own private directory, so it gets a copy.
COPIED = ["work/private-chinese-rap-written-rhyme-v1"]

JOBS = [
    {
        "name": "repertoire-network robustness",
        "command": ["src/build_repertoire_robustness_inference_v1.py",
                    "--private-graph", "work/private-chinese-rap-lyrical-repertoire-graph-v2"],
        "published": "results/repertoire-network-v1/robustness",
        "rebuilt": "results/repertoire-network-v1/robustness",
    },
    {
        "name": "retrieval",
        "command": ["src/build_chinese_rap_downstream_retrieval_v1.py"],
        "published": "results/retrieval-v1",
        "rebuilt": "outputs/chinese-rap-downstream-retrieval-v1",
    },
    {
        "name": "written-rhyme",
        "command": ["src/build_chinese_rap_written_rhyme_v1.py", "build"],
        "published": "results/written-rhyme-v1",
        "rebuilt": "outputs/chinese-rap-written-rhyme-v1",
    },
]

# leaf paths that differ on every rebuild without meaning anything
INERT = ("/generated_at_utc", "/generated_at", "/timestamp")


def leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{prefix}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaves(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


def junction(link: Path, target: Path) -> bool:
    link.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True)
        return result.returncode == 0
    link.symlink_to(target, target_is_directory=True)
    return True


def prepare(private_root: Path, sandbox: Path) -> list[str]:
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    for name in ("src", "results", "tools", "methods", "analysis", "validation"):
        source = REPO_ROOT / name
        if source.is_dir():
            shutil.copytree(source, sandbox / name)
    shutil.copy2(REPO_ROOT / "requirements.txt", sandbox / "requirements.txt")
    # real directories, so every build target stays inside the sandbox
    (sandbox / "work").mkdir()
    (sandbox / "outputs").mkdir()

    missing = []
    for relative, required in READ_ONLY:
        target = private_root / relative
        if not target.exists():
            (missing if required else [])[:0] = [relative]
            continue
        if not junction(sandbox / relative, target):
            missing.append(f"{relative} (link failed)")
    for relative in COPIED:
        target = private_root / relative
        if not target.exists():
            missing.append(relative)
            continue
        shutil.copytree(target, sandbox / relative)
    return missing


def compare(published: Path, rebuilt: Path) -> dict:
    if not rebuilt.is_dir():
        return {"error": f"nothing rebuilt at {rebuilt}"}
    identical = differing = 0
    differing_names = []
    for path in sorted(p for p in published.iterdir() if p.is_file()):
        other = rebuilt / path.name
        if not other.is_file():
            differing_names.append(f"{path.name} (missing)")
            differing += 1
        elif path.read_bytes() == other.read_bytes():
            identical += 1
        else:
            differing_names.append(path.name)
            differing += 1

    summary = {"files_identical": identical, "files_differing": differing,
               "differing_files": differing_names}
    a, b = published / "analysis_summary.json", rebuilt / "analysis_summary.json"
    if a.is_file() and b.is_file():
        left = dict(leaves(json.loads(a.read_text(encoding="utf-8"))))
        right = dict(leaves(json.loads(b.read_text(encoding="utf-8"))))
        changed = [(k, left[k], right.get(k)) for k in left if left[k] != right.get(k)]
        summary["leaf_values"] = len(left)
        summary["leaf_values_differing"] = len(changed)
        summary["differences"] = [
            {"path": k, "published": str(x)[:120], "rebuilt": str(y)[:120],
             "classification": ("inert timestamp" if any(m in k for m in INERT)
                                else "recorded builder hash names a superseded version"
                                if "builder_code_sha256" in k or "code_sha256" in k
                                else "REAL DIVERGENCE")}
            for k, x, y in changed]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True,
                        help="directory containing work/ and outputs/")
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only", nargs="*", help="run only these jobs by name")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.sandbox.resolve().is_relative_to(REPO_ROOT):
        raise SystemExit("--sandbox must not resolve inside the repository")

    print(f"preparing sandbox at {args.sandbox}")
    missing = prepare(args.private_root.resolve(), args.sandbox.resolve())
    if missing:
        for name in missing:
            print(f"  missing private input: {name}", file=sys.stderr)
        return 2
    print("  private inputs linked read-only; build directories are inside the sandbox")

    results = {}
    for job in JOBS:
        if args.only and job["name"] not in args.only:
            continue
        print(f"\n=== {job['name']} ===")
        completed = subprocess.run([args.python, *job["command"]], cwd=args.sandbox,
                                   capture_output=True, text=True)
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-4:]
            print("  build failed:")
            for line in tail:
                print(f"    {line}")
            results[job["name"]] = {"build_failed": True, "tail": tail}
            continue
        outcome = compare(REPO_ROOT / job["published"], args.sandbox / job["rebuilt"])
        results[job["name"]] = outcome
        print(f"  files identical {outcome.get('files_identical')} / "
              f"differing {outcome.get('files_differing')}")
        if "leaf_values" in outcome:
            print(f"  leaf values {outcome['leaf_values']} / "
                  f"differing {outcome['leaf_values_differing']}")
        for difference in outcome.get("differences", []):
            print(f"    [{difference['classification']}] {difference['path']}")

    real = [(name, d) for name, outcome in results.items()
            for d in outcome.get("differences", [])
            if d["classification"] == "REAL DIVERGENCE"]
    failed = [name for name, outcome in results.items() if outcome.get("build_failed")]

    print()
    if failed:
        print(f"{len(failed)} build(s) failed: {failed}")
    if real:
        print(f"{len(real)} real divergence(s):")
        for name, difference in real:
            print(f"  {name}: {difference['path']}")
    if not failed and not real:
        print("every model artifact reproduces; all differences are inert or explained")

    if args.report:
        args.report.write_text(
            json.dumps({"private_root": str(args.private_root), "results": results},
                       ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="")
        print(f"wrote {args.report}")
    return 1 if (failed or real) else 0


if __name__ == "__main__":
    sys.exit(main())
