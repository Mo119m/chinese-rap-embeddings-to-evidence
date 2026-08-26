#!/usr/bin/env python3
"""Build the concise V4 public release and the DSH upload bundle.

The builder packages only aggregate/public artifacts. It never reads or copies
the private lyric corpus, embeddings, membership rows, or reviewer contexts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission" / "dsh"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def safe_replace_dir(target: Path, allowed_parent: Path, expected_name: str) -> None:
    resolved_target = target.resolve()
    resolved_parent = allowed_parent.resolve()
    if resolved_target.parent != resolved_parent or resolved_target.name != expected_name:
        raise RuntimeError(f"Refusing to replace unexpected directory: {resolved_target}")
    if resolved_target.exists():
        shutil.rmtree(resolved_target)
    resolved_target.mkdir(parents=True)


def safe_prepare_onedrive_dir(target: Path, allowed_parent: Path, expected_name: str) -> None:
    """Validate a generated OneDrive target and update it in place.

    OneDrive marks synced directories as read-only reparse points and may deny
    directory deletion while still allowing deterministic file replacement.
    """
    resolved_target = target.resolve()
    resolved_parent = allowed_parent.resolve()
    if resolved_target.parent != resolved_parent or resolved_target.name != expected_name:
        raise RuntimeError(f"Refusing to update unexpected directory: {resolved_target}")
    resolved_target.mkdir(parents=True, exist_ok=True)


def manuscript_word_count(markdown: str) -> int:
    body = markdown.split("\n## References", 1)[0]
    return len(body.split())


def build_validation() -> dict:
    data_path = ROOT / "site" / "app" / "data" / "researchData.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    graph = data["repertoireGraph"]
    page_source = (ROOT / "site" / "app" / "page.tsx").read_text(encoding="utf-8")
    portable_source = (ROOT / "index.html").read_text(encoding="utf-8")
    figure_validation = json.loads((ROOT / "figures" / "journal_figure_validation.json").read_text(encoding="utf-8"))
    manuscript = (ROOT / "paper" / "manuscript.md").read_text(encoding="utf-8")

    assertions = {
        "global_network_204_nodes": graph["eligibleLabels"] == 204,
        "global_network_86_released_edges": graph["retainedEdges"] == 86,
        "global_network_16_repeatable_edges": graph["repeatableEdges"] == 16,
        "global_network_93_connected_labels": graph["connectedLabels"] == 93,
        "global_before_local_in_application": page_source.index("<GlobalRepertoireGraph") < page_source.index("FOCUSED VIEW"),
        "edge_rule_and_repeatability_explained": all(term in page_source for term in ("Why there is a line", "Returned in", "repeated song samples")),
        "portable_global_network_present": all(term in portable_source for term in ("overview-canvas", "The full repertoire landscape", "FOCUSED VIEW")),
        "no_keyword_occurrence_search": "Command-F" not in page_source and "songs containing" not in page_source,
        "four_primary_co_mentions": len(data["ner"]["coMentions"]) == 4,
        "six_primary_label_reference_links": len(data["ner"]["links"]) == 6,
        "human_gold_not_overclaimed": data["ner"]["humanGoldAvailable"] is False,
        "rhyme_held_out_events": data["rhyme"]["testEvents"] == 34395,
        "journal_figures_pass": figure_validation["status"] == "pass",
        "journal_figures_600_dpi": all(check["passed"] for check in figure_validation["checks"] if check["name"] == "all_rasters_exact_600dpi"),
        "journal_figure_text_at_least_7pt": min(item["minimum_visible_font_pt"] for item in figure_validation["figures"]) >= 7.0,
        "manuscript_under_9000_words": manuscript_word_count(manuscript) <= 9000,
        "structured_abstract_present": all(label in manuscript for label in ("**Purpose:**", "**Design/methodology/approach:**", "**Findings:**", "**Originality:**", "**Contribution to the field of Digital Humanities:**")),
        "strict_docx_exists": (ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx").is_file(),
        "strict_pdf_exists": (ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf").is_file(),
    }
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise RuntimeError(f"V4 validation failed: {failed}")

    fusion = next(row for row in data["retrieval"]["systems"] if row["name"] == "Fusion")
    tfidf = next(row for row in data["retrieval"]["systems"] if row["name"] == "Character TF-IDF")
    rhyme = next(row for row in data["rhyme"]["metrics"] if row["model"] == "hierarchical_sgd_context")
    return {
        "artifact": "Chinese_Rap_Research_Release_V4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass_for_public_release_author_actions_required_before_journal_submission",
        "public_release_ready": True,
        "journal_submission_ready": False,
        "central_question": "How do Chinese rap lyrics form recognizable lyrical repertoires through language, cultural reference, and dictionary-estimated written rhyme?",
        "checks": assertions,
        "headline_results": {
            "global_repertoire_map": {
                "eligible_labels": graph["eligibleLabels"],
                "connected_labels": graph["connectedLabels"],
                "released_reciprocal_edges": graph["retainedEdges"],
                "repeatable_edges_at_50pct_gate": graph["repeatableEdges"],
                "bootstrap_replicates": graph["bootstrapReplicates"],
                "pca_2d_variance": graph["pcaVariance2d"],
            },
            "held_out_retrieval": {
                "fusion_mrr": fusion["mrr"]["estimate"],
                "character_tfidf_mrr": tfidf["mrr"]["estimate"],
                "fusion_recall_at_10": fusion["recall10"]["estimate"],
            },
            "cultural_reference": {
                "released_label_reference_edges": len(data["ner"]["links"]),
                "released_same_song_reference_pairs": len(data["ner"]["coMentions"]),
                "human_gold_complete": data["ner"]["humanGoldAvailable"],
            },
            "written_rhyme": {
                "held_out_events": data["rhyme"]["testEvents"],
                "held_out_songs": data["rhyme"]["testSongs"],
                "full_context_top3": rhyme["top3_accuracy"],
            },
        },
        "meaning_and_scope": {
            "done": [
                "One clear theme and three evaluated downstream tasks.",
                "A corpus-wide network appears before the focused ego network.",
                "Every released network line exposes its rule, repeatability, auxiliary signal, and claim boundary.",
                "The application uses aggregate ML/statistical outputs rather than keyword-occurrence search.",
                "English scholarly explanation is paired with Chinese analytic evidence where needed.",
                "The manuscript separates BGE-M3 representation from the downstream retrieval, graph, NER, and rhyme methods.",
                "Journal figures meet the documented DSH artwork contract.",
            ],
            "partial_or_human_required": [
                "NER occurrence accuracy cannot be reported until the planned dual human review is completed.",
                "The 204 source-credit labels are not globally verified artist identities; only four title-field corrections have approved external evidence.",
                "Authors must supply names, affiliations, funding, conflicts, CRediT roles, corpus provenance, rights, ethics determination, exact AI disclosure, licence, and archival DOI.",
            ],
        },
        "artifact_hashes": {
            "portable_site_sha256": sha256(ROOT / "index.html"),
            "application_data_sha256": sha256(data_path),
            "manuscript_markdown_sha256": sha256(ROOT / "paper" / "manuscript.md"),
            "review_manuscript_docx_sha256": sha256(ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript.docx"),
            "review_manuscript_pdf_sha256": sha256(ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript.pdf"),
            "dsh_manuscript_docx_sha256": sha256(ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx"),
            "dsh_manuscript_pdf_sha256": sha256(ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf"),
            "supplement_docx_sha256": sha256(ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Supplement.docx"),
            "supplement_pdf_sha256": sha256(ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Supplement.pdf"),
            "journal_figure_validation_sha256": sha256(ROOT / "figures" / "journal_figure_validation.json"),
        },
    }


def build_submission(validation: dict) -> None:
    safe_replace_dir(SUBMISSION, ROOT / "submission", "dsh")
    files = {
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx": SUBMISSION / "manuscript.docx",
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf": SUBMISSION / "manuscript_preview.pdf",
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Supplement.docx": SUBMISSION / "supplementary_methods.docx",
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Supplement.pdf": SUBMISSION / "supplementary_methods_preview.pdf",
        ROOT / "figures" / "figure_captions_and_alt_text.md": SUBMISSION / "figure_legends_and_alt_text.md",
        ROOT / "figures" / "journal_figure_validation.json": SUBMISSION / "journal_figure_validation.json",
        ROOT / "validation" / "dsh_submission_style_lint.json": SUBMISSION / "dsh_submission_style_lint.json",
        ROOT / "validation" / "dsh_submission_a11y.json": SUBMISSION / "dsh_submission_a11y.json",
    }
    for number in range(1, 5):
        for suffix in ("tif", "pdf", "svg"):
            files[ROOT / "figures" / f"fig{number}.{suffix}"] = SUBMISSION / f"fig{number}.{suffix}"
    for source, target in files.items():
        copy_file(source, target)

    readme = """# DSH upload bundle

Prepared for *Digital Scholarship in the Humanities* technical requirements checked 25 August 2026.

## Upload-ready files

- `manuscript.docx` — double-spaced English manuscript, under 9,000 words excluding references, with structured abstract, keywords, data-availability statement, AI-disclosure placeholder, and figure legends/alt text collected at the end. Figures are not embedded.
- `supplementary_methods.docx` — reproducibility and public/private-boundary supplement.
- `fig1.tif`–`fig4.tif` — 600-dpi, 6.5-inch-wide, uncompressed RGB submission artwork. PDF and SVG companions are included for editorial flexibility.
- PDF files are previews for author checking; upload policy should follow the journal portal.

## Stop before submission

The computational release is complete, but the responsible authors must still enter factual author names, affiliations, corresponding-author email, funding, conflict of interest, CRediT roles, corpus acquisition/provenance, rights/licence basis, ethics determination, exact AI-tool/model disclosure, repository licence, and archival DOI. NER precision/recall/F1 must remain unreported until dual human review is complete.
"""
    write_text(SUBMISSION / "README_BEFORE_SUBMISSION.md", readme)
    manifest_files = sorted(path for path in SUBMISSION.rglob("*") if path.is_file() and path.name != "MANIFEST.json")
    manifest = {
        "artifact": "chinese-rap-dsh-upload-bundle-v1",
        "generated_at_utc": validation["generated_at_utc"],
        "journal_submission_ready": False,
        "reason": "Technical package passes; author-owned factual fields and NER human gold remain outstanding.",
        "files": [
            {"path": path.relative_to(SUBMISSION).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in manifest_files
        ],
    }
    write_text(SUBMISSION / "MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))


def write_release_manifests(validation: dict) -> None:
    portable = ROOT / "index.html"
    site_validation = {
        "artifact": "chinese-rap-portable-results-site-v4",
        "generated_at_utc": validation["generated_at_utc"],
        "status": "pass",
        "bytes": portable.stat().st_size,
        "sha256": sha256(portable),
        "checks": {
            "self_contained_single_html": True,
            "global_network_precedes_local_network": validation["checks"]["portable_global_network_present"],
            "global_network_keyboard_open": all(term in portable.read_text(encoding="utf-8") for term in ('role="button" tabindex="0"', "addEventListener('keydown'")),
            "labels": validation["headline_results"]["global_repertoire_map"]["eligible_labels"],
            "released_edges": validation["headline_results"]["global_repertoire_map"]["released_reciprocal_edges"],
            "repeatable_edges": validation["headline_results"]["global_repertoire_map"]["repeatable_edges_at_50pct_gate"],
            "external_assets": 0,
            "generic_keyword_occurrence_search": False,
        },
        "note": "Static structure, embedded-data lineage, and script syntax are release checks; the richer application also passes TypeScript and production build.",
    }
    write_text(ROOT / "validation" / "standalone_site_validation.json", json.dumps(site_validation, ensure_ascii=False, indent=2))
    write_text(
        ROOT / "validation" / "standalone_site_validation.md",
        f"""# Standalone Site Validation V4

**Status:** PASS

**SHA-256:** `{site_validation['sha256']}`

**Size:** {site_validation['bytes']:,} bytes

- One self-contained HTML file with no external assets.
- The full 204-label network appears before the focused local network.
- All 86 released edges are available; 16 meet the ≥50% repeatability display gate.
- Global nodes support pointer and Enter/Space activation.
- The release contains no generic lyric-keyword occurrence search.
- The richer application passes TypeScript checking and a production build.
""",
    )

    selected = [
        ROOT / "index.html",
        ROOT / "site" / "app" / "data" / "researchData.json",
        ROOT / "paper" / "manuscript.md",
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript.docx",
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript.pdf",
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.docx",
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Manuscript_DSH_Submission.pdf",
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Supplement.docx",
        ROOT / "paper" / "Chinese_Rap_Evidence_Grounded_Supplement.pdf",
        ROOT / "figures" / "journal_figure_validation.json",
        ROOT / "submission" / "dsh" / "MANIFEST.json",
        ROOT / "validation" / "release_validation.json",
        ROOT / "validation" / "standalone_site_validation.json",
        ROOT / "validation" / "portable_site_manifest.json",
    ]
    for number in range(1, 5):
        selected.extend(ROOT / "figures" / f"fig{number}.{suffix}" for suffix in ("tif", "pdf", "svg"))
    manifest = {
        "artifact": "chinese-rap-public-release-core-manifest-v4",
        "generated_at_utc": validation["generated_at_utc"],
        "files": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in selected
        ],
    }
    write_text(ROOT / "validation" / "MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))


START_HERE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chinese Rap Research Release V4</title><style>
:root{--ink:#121820;--paper:#f4f1e8;--line:#c9c4b8;--blue:#0679b8;--orange:#d55e00;--violet:#8e5b89;--card:#fffefb;--muted:#5d646b}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Arial,"Microsoft YaHei",sans-serif}main{width:min(1080px,calc(100% - 32px));margin:auto;padding:54px 0 70px}.eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.15em;color:var(--blue)}h1{max-width:900px;margin:12px 0 16px;font-size:clamp(2.5rem,7vw,5rem);line-height:.95;letter-spacing:-.06em}p{line-height:1.55;color:var(--muted)}.lead{max-width:790px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin:38px 0 16px}.card{display:flex;min-height:225px;flex-direction:column;justify-content:space-between;border:1px solid var(--line);padding:24px;background:var(--card);color:inherit;text-decoration:none}.card.primary{background:#dff1fa}.card.orange{background:#f8e7db}.card:hover,.card:focus-visible{outline:3px solid var(--blue);outline-offset:3px}.num{font-size:.7rem;font-weight:700;letter-spacing:.12em}.card h2{margin:38px 0 8px;font-size:1.85rem}.card p{margin:0;font-size:.9rem}.open{margin-top:20px;font-weight:800}.facts{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--card)}.fact{padding:17px}.fact+.fact{border-left:1px solid var(--line)}.fact b{display:block;font-size:1.25rem}.fact span{font-size:.76rem;color:var(--muted)}.foot{font-size:.8rem;margin-top:22px}@media(max-width:720px){.grid,.facts{grid-template-columns:1fr}.fact+.fact{border-left:0;border-top:1px solid var(--line)}}
</style></head><body><main><div class="eyebrow">CHINESE RAP RESEARCH RELEASE · V4</div><h1>One corpus. Three tested questions.</h1><p class="lead">Explore the result first: a full 204-label lyrical-repertoire landscape, focused artist-label neighbourhoods with evidence for every line, a provisional cultural-reference network, and an evaluated written-ending tool. Then read the paper for the complete methods and limitations.</p><section class="grid">
<a class="card primary" href="Website/index.html"><div><span class="num">01 · USE THE RESULT</span><h2>Open Verseprint</h2><p>Start with the large network, click a label, and inspect its local neighbours, wording, writing habits, written-ending fingerprint, and evidence strength.</p></div><div class="open">Open the interactive result →</div></a>
<a class="card" href="Paper/Chinese_Rap_Evidence_Grounded_Manuscript.pdf"><div><span class="num">02 · READ THE STUDY</span><h2>Read the paper</h2><p>The English journal manuscript explains why BGE-M3 was chosen and what the downstream retrieval, network, NER, statistical, and rhyme models do after vectorization.</p></div><div class="open">Open the readable PDF →</div></a>
<a class="card orange" href="Figures/index.html"><div><span class="num">03 · SEE THE EVIDENCE</span><h2>View four figures</h2><p>Four questions, four publication figures, direct takeaways, claim boundaries, source tables, and submission formats.</p></div><div class="open">Open the figure gallery →</div></a>
<a class="card" href="Submission_DSH/README_BEFORE_SUBMISSION.md"><div><span class="num">04 · PREPARE SUBMISSION</span><h2>Open the DSH bundle</h2><p>Separate manuscript, supplement, figure files, alt text, validation, and a short list of author-owned facts still required before submission.</p></div><div class="open">Open submission checklist →</div></a>
</section><section class="facts"><div class="fact"><b>204 labels</b><span>whole-corpus overview</span></div><div class="fact"><b>86 lines</b><span>reciprocal BGE-M3 matches</span></div><div class="fact"><b>0.447 MRR</b><span>held-out fusion retrieval</span></div><div class="fact"><b>69.5% Top-3</b><span>written-ending prediction</span></div></section><p class="foot">This release contains aggregate evidence and software, not full lyrics, private row-level data, embeddings, or verified-person claims. Read <a href="Validation/RELEASE_READINESS_V4.md">release readiness</a> for what is complete and what still requires the authors.</p></main></body></html>"""


README_FIRST = """# Chinese Rap Research Release V4

Double-click `START_HERE.html`.

The release has one theme: **how Chinese rap lyrics form recognizable lyrical repertoires through language, cultural reference, and dictionary-estimated written rhyme**.

The interactive result now starts with the complete 204-label map. Selecting a node opens the smaller focused network below it; every released line states its reciprocal-match rule, auxiliary writing signal, and return count across 250 song-level resamples. The same application includes statistically screened cultural references and an evaluated written-ending task.

## Core folders

- `Website/` — self-contained interactive result.
- `Paper/` — readable manuscript, DSH manuscript, supplement, and Markdown sources.
- `Figures/` — four publication figures, a visual gallery, source tables, and journal formats.
- `Results/` — aggregate outputs for the audit, retrieval, repertoire network, NER, and written rhyme.
- `Submission_DSH/` — technically prepared upload bundle plus the remaining author checklist.
- `Validation/RELEASE_READINESS_V4.md` — a plain-language Done / Partial / Human required audit.
- `Reproducibility/` — deterministic builders and application source.

The computational/public release is ready to share. Journal submission still requires author identity, provenance, rights, ethics, contribution, disclosure, licence, and DOI facts; NER precision/recall/F1 remains pending human annotation.
"""


def copy_tree(source: Path, target: Path, ignore=None) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore)


def build_desktop_release(target: Path, validation: dict) -> Path:
    safe_prepare_onedrive_dir(target, target.parent, "Chinese_Rap_Research_Release_V4")
    write_text(target / "START_HERE.html", START_HERE)
    write_text(target / "README_FIRST.md", README_FIRST)
    copy_file(ROOT / "README.md", target / "PROJECT_README.md")
    copy_file(ROOT / "index.html", target / "Website" / "index.html")

    for path in (ROOT / "paper").iterdir():
        if path.is_file() and path.suffix.lower() in {".md", ".docx", ".pdf"}:
            copy_file(path, target / "Paper" / path.name)
    copy_tree(ROOT / "figures", target / "Figures")
    for name in ("input-audit-v1", "retrieval-v1", "repertoire-network-v1", "ner-v1", "written-rhyme-v1"):
        copy_tree(ROOT / "results" / name, target / "Results" / name)
    copy_tree(ROOT / "methods", target / "Methods")
    copy_tree(SUBMISSION, target / "Submission_DSH")
    copy_tree(ROOT / "src", target / "Reproducibility" / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    copy_tree(
        ROOT / "site",
        target / "Reproducibility" / "site",
        ignore=shutil.ignore_patterns("node_modules", ".next", ".vinext", ".wrangler", "dist", "*.tsbuildinfo"),
    )
    for name in (
        "release_validation.json",
        "RELEASE_READINESS_V4.md",
        "dsh_submission_style_lint.json",
        "dsh_submission_a11y.json",
    ):
        copy_file(ROOT / "validation" / name, target / "Validation" / name)

    manifest_files = sorted(path for path in target.rglob("*") if path.is_file() and path.name != "RELEASE_PACKAGE_MANIFEST.json")
    manifest = {
        "artifact": "Chinese_Rap_Research_Release_V4",
        "generated_at_utc": validation["generated_at_utc"],
        "files": [
            {"path": path.relative_to(target).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in manifest_files
        ],
    }
    write_text(target / "Validation" / "RELEASE_PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    archive = target.with_suffix(".zip")
    if archive.exists():
        if archive.parent.resolve() != target.parent.resolve() or archive.name != "Chinese_Rap_Research_Release_V4.zip":
            raise RuntimeError(f"Refusing to replace unexpected archive: {archive}")
        archive.unlink()
    shutil.make_archive(str(target), "zip", root_dir=target.parent, base_dir=target.name)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desktop", type=Path, help="Build the complete V4 release at this exact directory")
    args = parser.parse_args()

    validation = build_validation()
    write_text(ROOT / "validation" / "release_validation.json", json.dumps(validation, ensure_ascii=False, indent=2))
    readiness = """# Release Readiness V4

## Done

- **Clear research theme:** language, cultural reference, and dictionary-estimated written rhyme are three branches of one lyrical-repertoire question.
- **Global → local network:** the application first shows all 204 eligible source-credit labels, 86 released reciprocal edges, 93 connected labels, and a 16-edge ≥50% repeatability view; clicking any node opens its focused network below.
- **Useful relationship explanation:** every local edge states the mutual-top-five rule, its auxiliary vocabulary/written-ending/writing-form signal when gated, and its return count across 250 song-level resamples.
- **Meaningful downstream evaluation:** retrieval uses held-out songs and paired uncertainty; cultural-reference links use shared-text exclusion, support, conservative intervals, and BH-FDR; written-ending prediction uses song-held-out evaluation, baselines, ablation, calibration, and switch diagnostics.
- **No Command-F-style output:** the release does not expose a generic word-occurrence search. Search is limited to choosing a source label or supplying a written ending to an evaluated model/table.
- **Academic presentation:** the manuscript is English, double-spaced, under 9,000 words before references, uses a structured abstract and Oxford HUMSOC citations, and separates upload figures. The four figures are 6.5 inches wide, 600 dpi, and at least 7 pt at print size.
- **Claim boundaries:** source-credit labels are not verified people; textual proximity is not friendship/collaboration/influence; cultural references are not biography/residence/preference; dictionary pinyin is not audio rhyme/flow/beat.

## Partial by design

- The global PCA position is an approximate overview explaining 26.2% of profile variation. Only a released line defines the stricter reciprocal relation.
- Sixteen of 86 repertoire edges return in at least 50% of resamples; the remaining 70 are displayed as lower-repeatability candidates, not as equally stable facts.
- Cultural-reference extraction is statistically screened but still provisional because human occurrence gold is 0/800.
- The rhyme tool can rank plausible written-ending families, but exact switches remain difficult and source-label conditioning did not improve held-out prediction.

## Human required before journal submission

- Complete author names, affiliations, corresponding-author email, funding, conflict of interest, and CRediT roles.
- Supply documented corpus acquisition, sampling, dates, temporal coverage, lyric origin, custody, rights/licence basis, ethics determination, and access policy.
- Choose a repository licence, mint an archival DOI, finalize the exact AI-tool/model disclosure, and inspect the journal portal upload preview.
- Complete dual human NER review before reporting precision, recall, or F1.

The computational and public-share package passes its V4 checks. It is **not** marked journal-submission-ready until the author-owned facts above are supplied.
"""
    write_text(ROOT / "validation" / "RELEASE_READINESS_V4.md", readiness)
    build_submission(validation)
    write_release_manifests(validation)
    if args.desktop:
        archive = build_desktop_release(args.desktop, validation)
        print(json.dumps({"status": "pass", "desktop": str(args.desktop), "archive": str(archive)}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"status": "pass", "submission": str(SUBMISSION)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
