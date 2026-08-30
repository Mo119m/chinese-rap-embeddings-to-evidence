"""Real tamper scenarios against the desktop release boundary.

Searching the builder's source for a string proves nothing about what it packages. Each
scenario here builds an actual desktop package, attacks it the way a real mistake or a real
adversary would, and asserts the validator refuses it.

Slow -- it builds the package several times -- so it is a separate suite from the gate
tests and runs once in CI rather than on every platform.

    python tests/test_release_boundary.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
NAME = "Chinese_Rap_Research_Release_V4"
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{': ' + detail if detail else ''}")
        FAILURES.append(label)


def build(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "build_chinese_rap_release_v4.py"),
         "--reuse-generated-at", "--desktop", str(target)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")


def validate(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "validate_public_release_integrity_v1.py"),
         "--desktop", str(target)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")


def rewrite_manifest(target: Path) -> None:
    """Regenerate the package's own manifest, the way a tamperer would cover their tracks."""
    manifest_path = target / "Validation" / "RELEASE_PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = target / record["path"]
        record["bytes"] = path.stat().st_size
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8", newline="\n")


def test_untracked_files_are_not_packaged() -> None:
    print("\nuntracked files")
    probe = ROOT / "results" / "ner-v1" / "synthetic_probe_lyrics.csv"
    workspace = Path(tempfile.mkdtemp(prefix="boundary-"))
    try:
        probe.write_text("song_id,lyrics\nprobe,synthetic\n", encoding="utf-8", newline="\n")
        ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", str(probe)])
        check("the probe is ignored by git, so only the index can exclude it",
              ignored.returncode == 0)
        result = build(workspace / NAME)
        check("the build still succeeds", result.returncode == 0, result.stderr[-400:])
        packaged = list((workspace / NAME).rglob("synthetic_probe_lyrics.csv"))
        check("an ignored file is not packaged", not packaged,
              f"{len(packaged)} copies packaged")
        with zipfile.ZipFile((workspace / NAME).with_suffix(".zip")) as package:
            in_zip = [n for n in package.namelist() if "synthetic_probe" in n]
        check("and does not reach the ZIP", not in_zip)
    finally:
        probe.unlink(missing_ok=True)
        shutil.rmtree(workspace, ignore_errors=True)


def test_zip_member_tampering_is_caught() -> None:
    print("\nZIP member content")
    workspace = Path(tempfile.mkdtemp(prefix="boundary-"))
    try:
        target = workspace / NAME
        build(target)
        check("a clean package validates", validate(target).returncode == 0)

        archive = target.with_suffix(".zip")
        member = f"{NAME}/Results/compound-resolution-ner-cr-001/resolution_table.csv"
        with zipfile.ZipFile(archive) as package:
            entries = [(info, package.read(info.filename)) for info in package.infolist()]
        rebuilt = archive.with_name("tampered.zip")
        with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED) as package:
            for info, payload in entries:
                if info.filename == member:
                    payload = payload + b"TAMPERED\n"
                # writestr recomputes a correct CRC, so testzip() cannot see this
                package.writestr(info, payload)
        rebuilt.replace(archive)
        with zipfile.ZipFile(archive) as package:
            check("the tampered ZIP has an entirely valid CRC", package.testzip() is None)
        check("a modified ZIP member is rejected", validate(target).returncode != 0)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_packaged_tool_tampering_is_caught() -> None:
    print("\npackaged tools")
    workspace = Path(tempfile.mkdtemp(prefix="boundary-"))
    try:
        target = workspace / NAME
        build(target)
        gate = target / "Reproducibility" / "tools" / "verify_compound_resolution.py"
        source = gate.read_text(encoding="utf-8")
        gate.write_text(source.replace("return 1\n", "return 0\n", 1), encoding="utf-8", newline="\n")
        rewrite_manifest(target)
        subprocess.run([sys.executable, "-c",
                        "import shutil,sys,zipfile,pathlib;"
                        "t=pathlib.Path(sys.argv[1]);a=t.with_suffix('.zip');a.unlink();"
                        "z=zipfile.ZipFile(a,'w',zipfile.ZIP_DEFLATED);"
                        "[z.write(p, f'{t.name}/{p.relative_to(t).as_posix()}') "
                        " for p in sorted(t.rglob('*')) if p.is_file()];z.close()",
                        str(target)], check=True)
        check("a modified packaged gate is rejected even after the manifest is regenerated",
              validate(target).returncode != 0)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_unstaged_edits_are_refused() -> None:
    print("\nunstaged edits to tracked files")
    target = ROOT / "methods" / "NER_CR_001_COMPOUND_RESOLUTION.md"
    workspace = Path(tempfile.mkdtemp(prefix="boundary-"))
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n<!-- SYNTHETIC_UNSTAGED_MARKER -->\n")
        result = build(workspace / NAME)
        check("a build over an unstaged edit to a tracked file is refused",
              result.returncode != 0 and "working tree" in (result.stderr + result.stdout))
        check("nothing was packaged", not (workspace / NAME / "Methods").exists())
    finally:
        target.write_bytes(original)
        shutil.rmtree(workspace, ignore_errors=True)


def test_provenance_binds_every_member() -> None:
    print("\nsource provenance")
    workspace = Path(tempfile.mkdtemp(prefix="boundary-"))
    try:
        target = workspace / NAME
        build(target)
        victim = target / "Results" / "retrieval-v1" / "metrics.csv"
        victim.write_bytes(victim.read_bytes() + b"tampered,row\n")
        rewrite_manifest(target)
        subprocess.run([sys.executable, "-c",
                        "import sys,zipfile,pathlib;"
                        "t=pathlib.Path(sys.argv[1]);a=t.with_suffix('.zip');a.unlink();"
                        "z=zipfile.ZipFile(a,'w',zipfile.ZIP_DEFLATED);"
                        "[z.write(p, f'{t.name}/{p.relative_to(t).as_posix()}') "
                        " for p in sorted(t.rglob('*')) if p.is_file()];z.close()",
                        str(target)], check=True)
        check("tampering with a file OUTSIDE the 13-file byte map is still rejected, because "
              "provenance binds every member to a committed blob",
              validate(target).returncode != 0)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_archive_is_byte_reproducible() -> None:
    print("\narchive determinism")
    first = Path(tempfile.mkdtemp(prefix="boundary-"))
    second = Path(tempfile.mkdtemp(prefix="boundary-"))
    try:
        build(first / NAME)
        build(second / NAME)
        digests = [hashlib.sha256((base / NAME).with_suffix(".zip").read_bytes()).hexdigest()
                   for base in (first, second)]
        check("two builds of one tree produce the same archive bytes", digests[0] == digests[1],
              f"{digests[0][:16]} vs {digests[1][:16]}")
        print(f"       archive sha256 {digests[0]}")
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)


def main() -> int:
    probe = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-dir"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        print("this suite builds and attacks real packages, which requires a repository "
              "checkout with git; it does not run inside the desktop package")
        return 2
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("working tree is not clean; these scenarios add and remove files in it")
        print(dirty)
        return 2
    for suite in (test_untracked_files_are_not_packaged, test_unstaged_edits_are_refused,
                  test_zip_member_tampering_is_caught, test_provenance_binds_every_member,
                  test_packaged_tool_tampering_is_caught, test_archive_is_byte_reproducible):
        suite()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  {name}")
        return 1
    print("all release-boundary checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
