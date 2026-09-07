#!/usr/bin/env python3
"""Build the manuscript PDFs, and pin the part of them that is reproducible.

Until now this step lived outside the repository. The DOCX derivatives are built by
`src/build_chinese_rap_paper_docx_v1.py` and their bytes are pinned; the PDFs beside
them were produced by hand, so an edit to `paper/manuscript.md` left them stale with nothing
to say so. `tools/check_manuscript_derivatives.py` reports that they need rebuilding but
cannot rebuild them, which is where the trail ended.

Why the PDFs are not pinned by hash. Word writes a creation timestamp and a document id into
every conversion, so the same DOCX converted twice gives two different files -- measured
here, byte digests differ while the size is identical to the byte. Pinning those bytes would
produce a check that fails on every run for no reason, which is worse than no check.

What is stable is the content. Across repeated conversions of one DOCX the extracted text
and the page count reproduce exactly, so those are what this records and what `--check`
verifies. That answers the question the byte hash was being asked for -- is this PDF the
current manuscript -- without pretending to a determinism the converter does not have.

    python tools/build_manuscript_pdfs.py            # convert, then record
    python tools/build_manuscript_pdfs.py --check    # verify without converting

The two PDFs under submission/dsh are copies the release builder makes; this tool
checks that they are still copies rather than converting them a second time.

`--check` needs only pypdf and is safe in CI. Conversion needs Word (via docx2pdf) or
LibreOffice, so it runs where those exist and says plainly when they do not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "paper" / "pdf_provenance.json"

# The PDFs this tool builds, and the DOCX each comes from.
#
# Only the three under paper/. The two under submission/dsh are copies of two of these,
# made by src/build_chinese_rap_release_v4.py, and its own manifest covers them. Converting
# them here as well would have this tool and the release builder overwrite each other in
# turn, each leaving the other's record stale.
PAIRS = [
    ("paper/Chinese_Rap_Evidence_Grounded_Manuscript.docx",
     "paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf"),
    ("paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx",
     "paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf"),
    ("paper/Chinese_Rap_Evidence_Grounded_Supplement.docx",
     "paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf"),
]

# Copied by the release builder from the PDFs above; verified as copies, not reconverted.
COPIES = {
    "submission/dsh/manuscript_preview.pdf":
        "paper/Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf",
    "submission/dsh/supplementary_methods_preview.pdf":
        "paper/Chinese_Rap_Evidence_Grounded_Supplement.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_signature(pdf: Path) -> dict:
    """What survives the converter's non-determinism: the text and how it is paged."""
    import pypdf

    reader = pypdf.PdfReader(str(pdf))
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return {
        "pages": len(reader.pages),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_characters": len(text),
    }


def converter():
    """(name, callable) for the first converter available, or (None, None)."""
    try:
        import docx2pdf  # noqa: F401
        if sys.platform == "win32":
            def convert(src: Path, dst: Path) -> None:
                import docx2pdf
                docx2pdf.convert(str(src), str(dst))
            return "Microsoft Word via docx2pdf", convert
    except ImportError:
        pass

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        def convert(src: Path, dst: Path) -> None:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", "--outdir",
                 str(dst.parent), str(src)],
                check=True, capture_output=True)
            produced = dst.parent / (src.stem + ".pdf")
            if produced != dst:
                produced.replace(dst)
        return f"LibreOffice ({soffice})", convert

    return None, None


def check() -> int:
    if not RECORD.is_file():
        print(f"no record at {RECORD.relative_to(ROOT)}; run without --check first",
              file=sys.stderr)
        return 2
    recorded = json.loads(RECORD.read_text(encoding="utf-8"))["pdfs"]
    failures = []
    for docx_rel, pdf_rel in PAIRS:
        docx, pdf = ROOT / docx_rel, ROOT / pdf_rel
        entry = recorded.get(pdf_rel)
        if entry is None:
            failures.append(f"{pdf_rel}: not in the record")
            continue
        if not pdf.is_file():
            failures.append(f"{pdf_rel}: missing")
            continue
        if docx.is_file() and sha256(docx) != entry["source_docx_sha256"]:
            failures.append(
                f"{pdf_rel}: built from an older {docx_rel}; reconvert")
            continue
        actual = content_signature(pdf)
        for field in ("pages", "text_sha256"):
            if actual[field] != entry[field]:
                failures.append(
                    f"{pdf_rel}: {field} is {actual[field]}, recorded "
                    f"{entry[field]}")
                break
        else:
            print(f"  ok   {pdf_rel}  ({actual['pages']} pages)")

    for copy_rel, source_rel in COPIES.items():
        copy, source = ROOT / copy_rel, ROOT / source_rel
        if not copy.is_file():
            failures.append(f"{copy_rel}: missing")
        elif not source.is_file():
            failures.append(f"{copy_rel}: its source {source_rel} is missing")
        elif sha256(copy) != sha256(source):
            failures.append(
                f"{copy_rel}: no longer a copy of {source_rel}; rerun the release build")
        else:
            print(f"  ok   {copy_rel}  (copy of {Path(source_rel).name})")

    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"\n{len(failures)} PDF(s) do not match the record")
        return 1
    print(f"\nall {len(PAIRS)} PDFs match their source DOCX and recorded content")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify against the record without converting")
    args = parser.parse_args()

    if args.check:
        return check()

    name, convert = converter()
    if convert is None:
        print("no PDF converter found. Install one of:\n"
              "  pip install docx2pdf     (needs Microsoft Word, Windows or macOS)\n"
              "  LibreOffice              (any platform; provides `soffice`)",
              file=sys.stderr)
        return 2
    print(f"converting with {name}")

    record = {
        "how_this_is_verified": (
            "The converter writes a creation timestamp and a document id, so PDF bytes "
            "differ between two conversions of one DOCX and cannot be pinned. The extracted "
            "text and page count do reproduce, and are what is recorded here."
        ),
        "pdfs": {},
    }
    for docx_rel, pdf_rel in PAIRS:
        docx, pdf = ROOT / docx_rel, ROOT / pdf_rel
        if not docx.is_file():
            print(f"  skip {pdf_rel}: {docx_rel} does not exist")
            continue
        convert(docx, pdf)
        signature = content_signature(pdf)
        record["pdfs"][pdf_rel] = {
            "source_docx": docx_rel,
            "source_docx_sha256": sha256(docx),
            **signature,
        }
        print(f"  ok   {pdf_rel}  ({signature['pages']} pages, "
              f"{signature['text_characters']:,} characters)")

    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8", newline="")
    print(f"\nrecorded {len(record['pdfs'])} PDF(s) in {RECORD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
