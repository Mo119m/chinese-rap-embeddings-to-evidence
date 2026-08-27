#!/usr/bin/env python3
"""Restore tracked working-tree files to their exact committed blob bytes.

This is for an existing Windows checkout created before the repository disabled
Git line-ending translation. It refuses to run when the index or tracked
working tree contains changes, leaves untracked files alone, and never edits
the Git metadata directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def tracked_changes() -> list[str]:
    output = git("status", "--porcelain=v1", "--untracked-files=no").decode("utf-8")
    return [line for line in output.splitlines() if line]


def index_entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw in git("ls-files", "--stage", "-z").split(b"\0"):
        if not raw:
            continue
        metadata, encoded_path = raw.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        if stage != "0" or mode == "160000":
            continue
        entries.append((encoded_path.decode("utf-8"), oid))
    return entries


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    changes = tracked_changes()
    if changes:
        print(json.dumps({"status": "refused", "tracked_changes": changes}, indent=2))
        raise SystemExit("Refusing to overwrite a checkout with tracked changes.")

    restored = 0
    verified = 0
    for relative, oid in index_entries():
        path = (ROOT / relative).resolve()
        if ROOT.resolve() not in path.parents:
            raise RuntimeError(f"Tracked path escapes repository root: {relative}")
        payload = git("cat-file", "blob", oid)
        if not path.is_file() or path.read_bytes() != payload:
            atomic_write(path, payload)
            restored += 1
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(payload).digest():
            raise RuntimeError(f"Blob verification failed after restore: {relative}")
        verified += 1

    print(json.dumps({"status": "pass", "restored": restored, "verified": verified}, indent=2))


if __name__ == "__main__":
    main()
