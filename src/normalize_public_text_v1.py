#!/usr/bin/env python3
"""Normalize every tracked public text file to UTF-8 without BOM and LF.

This is an explicit release-maintenance step. It never touches untracked files,
binary formats, or private data outside the repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".css",
    ".html",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_NAMES = {".gitattributes", ".gitignore", ".python-version", "LICENSE", "LICENSE-CODE"}
UTF8_BOM = b"\xef\xbb\xbf"


def tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]


def is_text_path(path: Path) -> bool:
    return path.name in TEXT_NAMES or path.suffix.lower() in TEXT_SUFFIXES


def normalized_bytes(payload: bytes) -> bytes:
    if payload.startswith(UTF8_BOM):
        payload = payload[len(UTF8_BOM) :]
    text = payload.decode("utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite nonconforming tracked text files. Without this flag, only check.",
    )
    args = parser.parse_args()

    changed: list[str] = []
    for path in tracked_paths():
        if not is_text_path(path):
            continue
        original = path.read_bytes()
        normalized = normalized_bytes(original)
        if normalized == original:
            continue
        changed.append(path.relative_to(ROOT).as_posix())
        if args.fix:
            path.write_bytes(normalized)

    result = {
        "status": "fixed" if args.fix else ("fail" if changed else "pass"),
        "nonconforming_files": changed,
        "count": len(changed),
        "contract": "UTF-8 without BOM; LF line endings",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if changed and not args.fix:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
