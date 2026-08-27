#!/usr/bin/env python3
"""Synchronize the portable single-file reader with validated site data.

The portable reader keeps a compact, name-keyed schema so it can open directly
from disk. This script updates its repertoire nodes, edges, global coordinates,
and graph method metadata from the canonical public site bundle while preserving
the portable-only cultural co-mention and rhyme lookup fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "index.html"
SITE_DATA_PATH = ROOT / "site" / "app" / "data" / "researchData.json"
CHARACTER_MAP_PATH = ROOT / "site" / "app" / "data" / "characterToRhymeFamily.json"
MANIFEST_PATH = ROOT / "validation" / "portable_site_manifest.json"


def embedded_payload(html: str, element_id: str) -> dict:
    pattern = rf'(<script type="application/json" id="{re.escape(element_id)}">)(.*?)(</script>)'
    match = re.search(pattern, html, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"Missing embedded JSON element: {element_id}")
    return json.loads(match.group(2))


def replace_payload(html: str, element_id: str, payload: dict) -> str:
    pattern = rf'(<script type="application/json" id="{re.escape(element_id)}">)(.*?)(</script>)'
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return re.sub(pattern, lambda match: match.group(1) + encoded + match.group(3), html, count=1, flags=re.DOTALL)


def main() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    portable = embedded_payload(html, "research-data")
    site = json.loads(SITE_DATA_PATH.read_text(encoding="utf-8"))
    character_map = json.loads(CHARACTER_MAP_PATH.read_text(encoding="utf-8"))

    id_to_name = {row["id"]: row["label"] for row in site["labels"]}
    if len(id_to_name) != 204 or len(site["lyricalEdges"]) != 86:
        raise RuntimeError("Unexpected canonical repertoire population")

    portable["labels"] = [
        {
            "name": row["label"],
            "x": row["x"],
            "y": row["y"],
            "independentSongs": row["independentSongs"],
            "terms": row.get("terms", []),
            "traits": row.get("traits", []),
            "rhyme": row.get("rhyme"),
        }
        for row in site["labels"]
    ]
    portable["lyricalEdges"] = [
        {
            "a": id_to_name[row["a"]],
            "b": id_to_name[row["b"]],
            "repeatability": row["repeatability"],
            "status": row["status"],
            "dominantSignal": row["dominantSignal"],
            "reasons": row.get("reasons", []),
        }
        for row in site["lyricalEdges"]
    ]
    portable["repertoireGraph"] = site["repertoireGraph"]
    portable["question"] = site["question"]
    portable["publicBoundary"] = site["publicBoundary"]

    linked = {
        endpoint
        for edge in portable["lyricalEdges"]
        for endpoint in (edge["a"], edge["b"])
    }
    repeatable = [edge for edge in portable["lyricalEdges"] if edge["status"] == "repeatable"]
    if (len(portable["labels"]), len(portable["lyricalEdges"]), len(linked), len(repeatable)) != (204, 86, 93, 16):
        raise RuntimeError("Portable graph reconciliation failed")
    if any(not isinstance(row.get("x"), (int, float)) or not isinstance(row.get("y"), (int, float)) for row in portable["labels"]):
        raise RuntimeError("Portable graph is missing PCA coordinates")

    html = replace_payload(html, "research-data", portable)
    html = replace_payload(html, "character-map", character_map)
    HTML_PATH.write_text(html, encoding="utf-8", newline="\n")

    html_bytes = html.encode("utf-8")
    manifest = {
        "artifact": "chinese-rap-portable-results-site-v2",
        "labels": len(portable["labels"]),
        "lyricalEdges": len(portable["lyricalEdges"]),
        "connectedLabels": len(linked),
        "repeatableEdges": len(repeatable),
        "characterMapEntries": len(character_map),
        "researchDataSourceSha256": hashlib.sha256(SITE_DATA_PATH.read_bytes()).hexdigest(),
        "portableHtmlSha256": hashlib.sha256(html_bytes).hexdigest(),
        "status": "pass",
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
