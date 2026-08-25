#!/usr/bin/env python3
"""Build a self-contained, public-safe explorer for the v2 repertoire graph.

The explorer deliberately exposes only the aggregate graph.  It does not load
lyrics, song/chunk identifiers, embeddings, or the private neighbour audit.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "1.1.0"
ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "outputs" / "chinese-rap-lyrical-repertoire-graph-v2"
OUT_DIR = ROOT / "outputs" / "chinese-rap-lyrical-repertoire-explorer-v1"
ARTIFACT_ID = "chinese-rap-lyrical-repertoire-explorer-v1"
REQUIRED_GRAPH_FILES = (
    "analysis_summary.json",
    "artist_label_registry.csv",
    "artist_repertoire_nodes.csv",
    "artist_repertoire_edges.csv",
    "artist_repertoire_layout.csv",
    "robustness_summary.csv",
    "manifest.json",
    "validation.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value: str | int | float) -> int:
    return int(float(value))


def as_float(value: str | int | float) -> float:
    return float(value)


def require_graph() -> dict[str, Any]:
    missing = [name for name in REQUIRED_GRAPH_FILES if not (GRAPH_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing validated graph inputs: {missing}")

    validation = json.loads((GRAPH_DIR / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "pass" or not all(bool(check.get("passed")) for check in validation.get("checks", [])):
        raise RuntimeError("The source graph validation is not passing; refusing to build explorer.")

    manifest = json.loads((GRAPH_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_id") != "chinese-rap-lyrical-repertoire-graph-v2":
        raise RuntimeError("Unexpected source graph artifact.")
    for name, record in manifest.get("output_files", {}).items():
        path = GRAPH_DIR / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Validated graph payload hash is stale or missing: {name}")
    return manifest


def make_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    graph_manifest = require_graph()
    summary = json.loads((GRAPH_DIR / "analysis_summary.json").read_text(encoding="utf-8"))
    registry_rows = read_csv(GRAPH_DIR / "artist_label_registry.csv")
    node_rows = read_csv(GRAPH_DIR / "artist_repertoire_nodes.csv")
    edge_rows = read_csv(GRAPH_DIR / "artist_repertoire_edges.csv")
    layout_rows = read_csv(GRAPH_DIR / "artist_repertoire_layout.csv")
    robustness_rows = read_csv(GRAPH_DIR / "robustness_summary.csv")

    registry_by_id = {row["artist_label_id"]: row for row in registry_rows}
    layout_by_id = {row["artist_label_id"]: row for row in layout_rows}
    if len(registry_by_id) != len(registry_rows) or len(layout_by_id) != len(layout_rows):
        raise RuntimeError("The public graph has duplicate source-label registry or layout identifiers.")
    eligible = [row for row in node_rows if row.get("graph_node_eligible") == "true"]
    eligible_ids = {row["artist_label_id"] for row in eligible}
    if len(eligible_ids) != len(eligible):
        raise RuntimeError("The public graph has duplicate graph-eligible node identifiers.")
    if int(summary["counts"]["graph_eligible_labels"]) != len(eligible):
        raise RuntimeError("Graph-eligible node count does not match the source graph summary.")
    if set(layout_by_id) != eligible_ids:
        raise RuntimeError("The semantic layout must cover exactly all graph-eligible nodes.")
    connected = [row for row in eligible if as_int(row["stable_graph_degree"]) > 0]
    connected_ids = {row["artist_label_id"] for row in connected}
    if not connected_ids:
        raise RuntimeError("Source graph has no connected nodes.")
    if len(connected_ids) != len(connected):
        raise RuntimeError("The public graph has duplicate connected node identifiers.")
    if int(summary["counts"]["connected_stable_graph_nodes"]) != len(connected):
        raise RuntimeError("Connected-node count does not match the source graph summary.")

    projection_rows = {(row["projection_population"], row["projection_variance_explained_2d"], row["layout_note"]) for row in layout_rows}
    if len(projection_rows) != 1:
        raise RuntimeError("Public semantic layout rows disagree about projection provenance.")
    projection_population, projection_variance, projection_note = next(iter(projection_rows))
    if as_int(projection_population) != len(eligible):
        raise RuntimeError("Semantic-layout population does not match graph-eligible node count.")
    if abs(as_float(projection_variance) - as_float(summary["model"].get("spatial_projection_variance_explained_2d", -1))) > 1e-8:
        raise RuntimeError("Semantic-layout explained variance does not match the source graph summary.")

    public_nodes: list[dict[str, Any]] = []
    for row in eligible:
        layout = layout_by_id.get(row["artist_label_id"])
        registry = registry_by_id.get(row["artist_label_id"])
        if layout is None:
            raise RuntimeError(f"Graph-eligible node has no semantic-layout row: {row['artist_label_id']}")
        if registry is None or registry.get("source_artist_label") != row.get("source_artist_label"):
            raise RuntimeError("A graph node does not exactly rejoin its source-label registry row.")
        if registry.get("label_attribution_status") != row.get("label_attribution_status"):
            raise RuntimeError("A graph node and registry disagree on source-label attribution status.")
        if registry.get("external_identity_verified") not in {"true", "false"}:
            raise RuntimeError("The source-label registry has an invalid identity-verification state.")
        if registry.get("external_identity_verified") != "false":
            raise RuntimeError("A positive identity claim requires a cited public identity registry; this explorer will not infer one.")
        public_nodes.append(
            {
                "id": row["artist_label_id"],
                "label": row["source_artist_label"],
                "identityVerified": registry["external_identity_verified"] == "true",
                "labelStatus": registry["label_attribution_status"],
                "songs": as_int(row["clean_song_count"]),
                "independentSongs": as_int(row["independent_clean_song_count"]),
                "primaryMass": round(as_float(row["primary_effective_text_mass"]), 4),
                "sensitivityMass": round(as_float(row["sensitivity_effective_text_mass"]), 4),
                "sharedDropShare": round(as_float(row["shared_text_dropped_mass_share"]), 6),
                "degree": as_int(row["stable_graph_degree"]),
                "hasStableLink": as_int(row["stable_graph_degree"]) > 0,
                "componentId": as_int(layout["component_id"]),
                "x": round(as_float(layout["x"]), 8),
                "y": round(as_float(layout["y"]), 8),
            }
        )
    public_nodes.sort(key=lambda row: (row["label"].casefold(), row["id"]))

    public_edges: list[dict[str, Any]] = []
    for row in edge_rows:
        left, right = row["artist_label_id_a"], row["artist_label_id_b"]
        if left not in connected_ids or right not in connected_ids:
            raise RuntimeError("Retained edge does not join two connected nodes.")
        if row.get("stable_across_shared_text_exclusion") != "true":
            raise RuntimeError("Explorer refuses an edge that did not pass the shared-text sensitivity gate.")
        if row.get("source_artist_label_a") != registry_by_id[left]["source_artist_label"] or row.get("source_artist_label_b") != registry_by_id[right]["source_artist_label"]:
            raise RuntimeError("An edge label does not exactly match the source-label registry.")
        ranks = [as_int(row[column]) for column in ("primary_rank_a_to_b", "primary_rank_b_to_a", "sensitivity_rank_a_to_b", "sensitivity_rank_b_to_a")]
        if any(rank < 1 or rank > as_int(summary["model"]["top_k"]) for rank in ranks):
            raise RuntimeError("A retained edge has a mutual-neighbour rank outside the source graph protocol.")
        public_edges.append(
            {
                "a": left,
                "b": right,
                "primaryRanks": [as_int(row["primary_rank_a_to_b"]), as_int(row["primary_rank_b_to_a"])],
                "sensitivityRanks": [as_int(row["sensitivity_rank_a_to_b"]), as_int(row["sensitivity_rank_b_to_a"])],
                "primaryPercentile": round(as_float(row["primary_pair_percentile"]), 3),
                "sensitivityPercentile": round(as_float(row["sensitivity_pair_percentile"]), 3),
            }
        )
    public_edges.sort(key=lambda row: (row["a"], row["b"]))
    visible_degree = Counter(identifier for edge in public_edges for identifier in (edge["a"], edge["b"]))
    if any(visible_degree[node["id"]] != node["degree"] for node in public_nodes):
        raise RuntimeError("Node degrees do not recompute from the retained public edge set.")
    if int(summary["counts"]["stable_retained_edges"]) != len(public_edges):
        raise RuntimeError("Stable-edge count does not match the source graph summary.")
    robustness = {row["metric"]: row["value"] for row in robustness_rows}
    if robustness.get("comparison_population_policy") != summary["model"].get("comparison_population_policy"):
        raise RuntimeError("Source graph comparison-population policy is not internally consistent.")

    payload = {
        "title": "Chinese Rap Lyrical Repertoire Map",
        "subtitle": "Stable lyric-repertoire proximity after duplicate control and removal of exact text shared across source labels.",
        "boundary": summary["claim_boundary"],
        "provenance": {
            "sourceArtifact": graph_manifest["artifact_id"],
            "sourceGraphVersion": summary["version"],
            "comparisonPopulationPolicy": summary["model"]["comparison_population_policy"],
        },
        "metrics": {
            "eligibleLabels": summary["counts"]["graph_eligible_labels"],
            "projectionLabels": summary["counts"]["graph_eligible_labels"],
            "stableLinkedLabels": summary["counts"]["connected_stable_graph_nodes"],
            "stableEdges": summary["counts"]["stable_retained_edges"],
            "cleanSongs": summary["counts"]["clean_songs"],
            "cleanChunks": summary["counts"]["clean_lyric_chunks"],
            "unlinkedEligibleLabels": summary["counts"]["graph_eligible_labels"] - summary["counts"]["connected_stable_graph_nodes"],
            "projectionVariance2d": round(as_float(projection_variance), 6),
            "topK": summary["model"]["top_k"],
            "minSongs": summary["model"]["minimum_clean_songs"],
            "minMass": summary["model"]["minimum_effective_text_mass"],
        },
        "projection": {
            "method": summary["model"].get("spatial_projection_method"),
            "population": as_int(projection_population),
            "varianceExplained2d": round(as_float(projection_variance), 6),
            "note": projection_note,
        },
        "nodes": public_nodes,
        "edges": public_edges,
    }
    lineage = {
        "source_graph_artifact": graph_manifest["artifact_id"],
        "source_graph_manifest_sha256": sha256_file(GRAPH_DIR / "manifest.json"),
        "source_graph_validation_sha256": sha256_file(GRAPH_DIR / "validation.json"),
        "required_public_input_sha256": {name: sha256_file(GRAPH_DIR / name) for name in REQUIRED_GRAPH_FILES},
        "source_graph_payload_hashes": graph_manifest["output_files"],
    }
    return payload, lineage


def html_document(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A duplicate-controlled Chinese rap lyrical-repertoire graph.">
  <title>Chinese Rap Lyrical Repertoire Map</title>
  <style>
    :root { --ink:#eff5ff; --muted:#a6b4c9; --bg:#07111f; --panel:#0d1b2f; --line:#213753; --accent:#5eead4; --gold:#f6c667; --danger:#ff8c86; --shadow:0 18px 50px rgba(0,0,0,.28); }
    * { box-sizing:border-box; }
    html { background:var(--bg); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
    body { margin:0; min-width:320px; background:radial-gradient(circle at 8% -10%,#17345a 0,transparent 32rem),radial-gradient(circle at 100% 0,#173d3f 0,transparent 30rem),var(--bg); }
    .shell { width:min(1440px,calc(100% - 32px)); margin:0 auto; padding:44px 0 54px; }
    .eyebrow { color:var(--accent); text-transform:uppercase; letter-spacing:.12em; font-weight:760; font-size:.71rem; }
    .visually-hidden { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
    h1 { max-width:900px; margin:10px 0 10px; font-size:clamp(2.05rem,5vw,4.6rem); letter-spacing:-.058em; line-height:.96; }
    .lede { max-width:830px; color:#cfdbec; font-size:clamp(1rem,1.75vw,1.25rem); line-height:1.55; margin:0; }
    .metrics { display:flex; flex-wrap:wrap; gap:10px; margin:27px 0 26px; }
    .metric { min-width:140px; padding:12px 14px; background:rgba(13,27,47,.8); border:1px solid rgba(147,181,222,.2); border-radius:12px; }
    .metric b { display:block; color:var(--accent); font-size:1.23rem; letter-spacing:-.03em; }
    .metric span { display:block; margin-top:3px; color:var(--muted); font-size:.73rem; }
    .frame { display:grid; grid-template-columns:minmax(0,1.54fr) minmax(315px,.76fr); border:1px solid rgba(147,181,222,.24); background:rgba(9,21,38,.77); border-radius:22px; overflow:hidden; box-shadow:var(--shadow); }
    .map-pane { min-height:670px; padding:20px 14px 14px; position:relative; border-right:1px solid rgba(147,181,222,.22); }
    .map-toolbar { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:0 8px 14px; }
    .map-toolbar strong { font-size:.93rem; }
    .map-toolbar span { color:var(--muted); font-size:.77rem; }
    .map-tools { display:flex; align-items:center; justify-content:flex-end; flex-wrap:wrap; gap:8px; }
    .focus-select { max-width:210px; min-height:33px; border:1px solid rgba(147,181,222,.24); border-radius:8px; padding:5px 8px; background:#10243d; color:#dae7f6; font:inherit; font-size:.74rem; }
    .focus-select:focus { border-color:var(--accent); outline:2px solid rgba(94,234,212,.24); }
    .graph-wrap { position:relative; height:604px; border-radius:16px; overflow:hidden; background:linear-gradient(135deg,rgba(22,47,78,.55),rgba(5,14,26,.74)); border:1px solid rgba(147,181,222,.14); }
    svg { width:100%; height:100%; display:block; }
    .edge { stroke:#617c9c; stroke-opacity:.34; stroke-width:1.6; transition:stroke .15s,stroke-opacity .15s,stroke-width .15s; cursor:pointer; }
    .edge.hit { stroke:transparent; stroke-width:14; }
    .axis-guide { stroke:rgba(147,181,222,.16); stroke-width:1; }
    .axis-label { fill:#8fa6c0; font-size:12px; letter-spacing:.04em; }
    .node { stroke:#e9f5ff; stroke-width:1.2; cursor:pointer; transition:r .16s,fill .16s,stroke-width .16s,opacity .16s; outline:none; }
    .node.context { opacity:.46; stroke-width:.8; }
    .node.context.selected,.node.context:hover,.node.context:focus { opacity:1; }
    .node:hover,.node:focus,.node.selected { stroke:#fff; stroke-width:3; }
    .node.dim,.edge.dim { opacity:.10; }
    .edge.selected { stroke:var(--gold); stroke-opacity:1; stroke-width:3.1; }
    .halo { fill:none; stroke:var(--gold); stroke-width:1.4; stroke-opacity:.95; pointer-events:none; }
    .tip { position:absolute; z-index:3; pointer-events:none; max-width:220px; padding:8px 10px; border:1px solid rgba(255,255,255,.18); border-radius:9px; background:rgba(5,13,25,.95); color:#f6fbff; font-size:.76rem; line-height:1.35; opacity:0; transform:translate(-50%,-118%); transition:opacity .12s; box-shadow:0 10px 20px rgba(0,0,0,.35); }
    .tip.show { opacity:1; }
    .legend { position:absolute; left:27px; bottom:27px; display:flex; align-items:center; gap:12px; padding:8px 10px; border-radius:9px; background:rgba(4,11,21,.68); color:#d8e6f5; font-size:.71rem; pointer-events:none; }
    .legend i { display:inline-block; width:8px; height:8px; background:#77A9D8; border-radius:50%; margin-right:4px; }
    .legend i.context-key { background:#6c7c91; opacity:.72; }
    aside { padding:26px 24px; min-height:670px; display:flex; flex-direction:column; gap:16px; }
    .panel-title { color:var(--muted); font-size:.71rem; text-transform:uppercase; letter-spacing:.12em; font-weight:750; }
    .selection-title { font-size:clamp(1.45rem,2.1vw,2rem); line-height:1.06; overflow-wrap:anywhere; margin:2px 0 0; }
    .source-note { color:var(--gold); font-size:.76rem; line-height:1.4; }
    .rule { height:1px; background:rgba(147,181,222,.18); margin:2px 0; }
    .facts { display:grid; grid-template-columns:1fr 1fr; gap:9px; }
    .fact { padding:10px; border-radius:10px; background:rgba(27,49,77,.52); }
    .fact b { display:block; font-size:1.05rem; color:#f5fbff; }
    .fact span { display:block; margin-top:3px; color:var(--muted); font-size:.69rem; line-height:1.25; }
    .neighbors { display:grid; gap:8px; }
    .neighbor { width:100%; text-align:left; color:inherit; border:1px solid rgba(147,181,222,.19); background:rgba(18,38,61,.55); padding:10px 11px; border-radius:10px; cursor:pointer; font:inherit; }
    .neighbor:hover,.neighbor:focus { border-color:var(--accent); background:rgba(28,65,76,.65); outline:none; }
    .neighbor b { display:block; font-size:.93rem; overflow-wrap:anywhere; }
    .neighbor small { display:block; color:var(--muted); font-size:.71rem; line-height:1.35; margin-top:4px; }
    .notice { padding:12px 13px; color:#c6d3e3; border-left:2px solid var(--accent); background:rgba(32,82,80,.14); border-radius:0 9px 9px 0; font-size:.77rem; line-height:1.48; }
    .empty { padding:20px 0; color:var(--muted); line-height:1.55; }
    .method { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:20px; }
    .method div { min-height:119px; padding:16px; border:1px solid rgba(147,181,222,.17); border-radius:13px; background:rgba(12,28,47,.64); }
    .method b { display:block; color:var(--accent); font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; }
    .method p { color:#bfcee1; font-size:.82rem; line-height:1.45; margin:9px 0 0; }
    .provenance { color:#8397b1; font-size:.69rem; line-height:1.45; padding:15px 0 0; }
    .provenance code { overflow-wrap:anywhere; color:#aabdd2; }
    button.reset { appearance:none; border:1px solid rgba(147,181,222,.24); border-radius:8px; padding:7px 9px; background:#132641; color:#dae7f6; cursor:pointer; font:inherit; font-size:.74rem; }
    button.reset:hover,button.reset:focus { border-color:var(--accent); color:white; outline:none; }
    @media (max-width:980px) { .frame { grid-template-columns:1fr; } .map-pane { min-height:570px; border-right:0; border-bottom:1px solid rgba(147,181,222,.22); } .graph-wrap { height:495px; } aside { min-height:0; } }
    @media (max-width:620px) { .shell { width:min(100% - 22px,1440px); padding-top:28px; } .map-pane { padding:12px 8px 8px; } .graph-wrap { height:430px; } aside { padding:20px 16px; } .method { grid-template-columns:1fr; } .metric { flex:1 1 130px; } .legend { display:none; } }
  </style>
</head>
<body>
  <main class="shell">
    <div class="eyebrow">Duplicate-controlled semantic analysis · English interface / Chinese-rap corpus</div>
    <h1 id="title"></h1>
    <p class="lede" id="subtitle"></p>
    <div class="metrics" id="metrics" aria-label="Corpus and graph summary"></div>
    <section class="frame" aria-label="Interactive lyrical-repertoire graph">
      <div class="map-pane">
        <div class="map-toolbar"><strong>Semantic projection + stable links</strong><div class="map-tools"><span>Click a dot or a stable line.</span><label class="visually-hidden" for="focusSelect">Focus a source corpus label</label><select class="focus-select" id="focusSelect"><option value="">Focus a corpus label</option></select><button class="reset" id="reset" type="button">Reset view</button></div></div>
        <div class="graph-wrap" id="graphWrap"><svg id="graph" viewBox="0 0 1000 680" role="img" aria-label="Semantic projection of source-labelled lyrical repertoires with stable proximity links"></svg><div class="tip" id="tip"></div><div class="legend"><span><i></i>blue = stable-linked label</span><span><i class="context-key"></i>grey = eligible, no retained line</span><span>nearer = approximate semantic proximity</span><span>line = stricter stable result</span></div></div>
      </div>
      <aside aria-live="polite">
        <div class="panel-title" id="detail-kicker">How to use the map</div>
        <h2 class="selection-title" id="detail-title">Start with a node</h2>
        <div id="detail-body" class="empty">Every dot is a source artist label with enough clean lyric material for comparison. Nearness is a two-dimensional semantic projection; lines are the stricter result that passed both duplicate-control representations.</div>
        <div class="notice">This is not a social, collaboration, identity, hometown, genre, beat, voice, or Flow graph. A line only says that two labelled lyric corpora were mutual top-5 semantic neighbours both before and after exact text shared across labels was removed.</div>
      </aside>
    </section>
    <section class="method" aria-label="How to read this result">
      <div><b>1 · Representation</b><p>Each label is represented by duplicate-weighted clean lyric-chunk embeddings. More duplicated text receives less weight.</p></div>
      <div><b>2 · Robustness gate</b><p>Exact clean text appearing under more than one label is removed, and the mutual-neighbour test is repeated.</p></div>
      <div><b>3 · Spatial meaning</b><p>Dot positions are a 2D PCA of the consensus lyric representation. Nearer dots are approximately semantically closer; a line is the stricter two-representation result.</p></div>
    </section>
    <footer class="provenance" id="provenance"></footer>
  </main>
  <script id="graph-data" type="application/json">__PAYLOAD__</script>
  <script>
  (() => {
    const G = JSON.parse(document.getElementById('graph-data').textContent);
    const $ = (id) => document.getElementById(id);
    const graph = $('graph'), wrap = $('graphWrap'), tip = $('tip');
    const nodeById = new Map(G.nodes.map(n => [n.id,n]));
    const edgeByKey = new Map();
    const k = (a,b) => [a,b].sort().join('::');
    G.edges.forEach(e => edgeByKey.set(k(e.a,e.b), e));
    const margin = 54, W = 1000, H = 680;
    const xs = G.nodes.map(n => n.x), ys = G.nodes.map(n => n.y);
    const range = (arr) => [Math.min(...arr),Math.max(...arr)];
    const [minX,maxX] = range(xs), [minY,maxY] = range(ys);
    const scale = (v,lo,hi,targetLo,targetHi) => hi === lo ? (targetLo+targetHi)/2 : targetLo + ((v-lo)/(hi-lo))*(targetHi-targetLo);
    const pos = (n) => ({ x:scale(n.x,minX,maxX,margin,W-margin), y:scale(n.y,minY,maxY,H-margin,margin) });
    const P = new Map(G.nodes.map(n => [n.id,pos(n)]));
    const esc = (value) => String(value).replace(/[&<>\"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[ch]));
    const showNumber = (v, digits=0) => Number(v).toLocaleString('en-US',{maximumFractionDigits:digits});
    const pct = (v) => (Number(v)*100).toFixed(1) + '%';
    const neighborEdges = (id) => G.edges.filter(e => e.a === id || e.b === id).sort((x,y) => y.sensitivityPercentile-x.sensitivityPercentile || y.primaryPercentile-x.primaryPercentile);
    const title = $('title'), subtitle = $('subtitle'), metrics = $('metrics');
    title.textContent = G.title; subtitle.textContent = G.subtitle;
    [['projectionLabels','labels in projection'],['stableLinkedLabels','stable-linked labels'],['stableEdges','stable links'],['unlinkedEligibleLabels','eligible, no stable line'],['cleanSongs','clean songs']].forEach(([key,label]) => {
      const el = document.createElement('div'); el.className='metric'; el.innerHTML=`<b>${showNumber(G.metrics[key])}</b><span>${label}</span>`; metrics.appendChild(el);
    });
    const ns = 'http://www.w3.org/2000/svg';
    const el = (tag,attrs={}) => { const x=document.createElementNS(ns,tag); Object.entries(attrs).forEach(([key,value])=>x.setAttribute(key,String(value))); return x; };
    const guideLayer = el('g',{'aria-hidden':'true'}), edgeLayer = el('g',{'aria-hidden':'true'}), hitLayer = el('g'), nodeLayer = el('g'), haloLayer = el('g',{'pointer-events':'none'});
    const zeroX=scale(0,minX,maxX,margin,W-margin), zeroY=scale(0,minY,maxY,H-margin,margin);
    guideLayer.append(el('line',{x1:zeroX,y1:margin,x2:zeroX,y2:H-margin,class:'axis-guide'}),el('line',{x1:margin,y1:zeroY,x2:W-margin,y2:zeroY,class:'axis-guide'}));
    const xLabel=el('text',{x:W-margin,y:H-18,class:'axis-label','text-anchor':'end'}); xLabel.textContent='PCA axis 1';
    const yLabel=el('text',{x:margin+7,y:margin+16,class:'axis-label'}); yLabel.textContent='PCA axis 2';
    guideLayer.append(xLabel,yLabel);
    graph.append(guideLayer,edgeLayer,hitLayer,haloLayer,nodeLayer);
    const nodeEls = new Map(), edgeEls = new Map(), hitEls = new Map();
    G.edges.forEach(e => { const a=P.get(e.a), b=P.get(e.b), key=k(e.a,e.b); const visual=el('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'edge','data-key':key}); const hit=el('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'edge hit','data-key':key,tabindex:'0',role:'button','aria-label':`Compare ${nodeById.get(e.a).label} and ${nodeById.get(e.b).label}`}); edgeLayer.appendChild(visual); hitLayer.appendChild(hit); edgeEls.set(key,visual); hitEls.set(key,hit); });
    G.nodes.slice().sort((a,b)=>Number(a.hasStableLink)-Number(b.hasStableLink) || a.label.localeCompare(b.label,'zh-Hans-CN')).forEach(n => { const p=P.get(n.id), radius=n.hasStableLink?Math.max(5,Math.min(15,3.5+1.12*Math.sqrt(n.primaryMass))):Math.max(3.2,Math.min(8.5,2.3+.78*Math.sqrt(n.primaryMass))); const lineState=n.hasStableLink?`${n.degree} stable lyrical-repertoire neighbours`:'no retained stable line; still shown as semantic context'; const circle=el('circle',{cx:p.x,cy:p.y,r:radius,class:`node ${n.hasStableLink?'stable':'context'}`,'data-id':n.id,fill:n.hasStableLink?'#77A9D8':'#6c7c91',tabindex:'0',role:'button','aria-label':`Source corpus label ${n.label}; identity not externally verified; ${n.songs} clean songs; ${lineState}`}); nodeLayer.appendChild(circle); nodeEls.set(n.id,circle); });
    G.nodes.slice().sort((a,b)=>a.label.localeCompare(b.label,'zh-Hans-CN')).forEach(n => { const option=document.createElement('option'); option.value=n.id; option.textContent=n.label; $('focusSelect').appendChild(option); });
    let selectedNode = null, selectedEdge = null;
    function clearEmphasis() { nodeEls.forEach(x=>x.classList.remove('dim','selected')); edgeEls.forEach(x=>x.classList.remove('dim','selected')); haloLayer.replaceChildren(); }
    function addHalo(id) { const n=nodeById.get(id), p=P.get(id), r=Number(nodeEls.get(id).getAttribute('r'))+5; haloLayer.appendChild(el('circle',{cx:p.x,cy:p.y,r,class:'halo'})); }
    function edgeSummary(e, focal) { const otherId=e.a===focal?e.b:e.a, other=nodeById.get(otherId); const primary = e.a===focal ? e.primaryRanks : [e.primaryRanks[1],e.primaryRanks[0]]; const sensitivity = e.a===focal ? e.sensitivityRanks : [e.sensitivityRanks[1],e.sensitivityRanks[0]]; return `<button class="neighbor" type="button" data-edge="${esc(k(e.a,e.b))}"><b>${esc(other.label)}</b><small>Mutual ranks: ${primary[0]} / ${primary[1]} (primary) · ${sensitivity[0]} / ${sensitivity[1]} (shared-text removal)<br>Pair percentile: ${e.primaryPercentile.toFixed(1)} → ${e.sensitivityPercentile.toFixed(1)}</small></button>`; }
    function selectNode(id) { selectedNode=id; selectedEdge=null; $('focusSelect').value=id; clearEmphasis(); const n=nodeById.get(id); const adjacent=neighborEdges(id); const near=new Set([id,...adjacent.flatMap(e=>[e.a,e.b])]); nodeEls.forEach((x,key)=>{if(!near.has(key))x.classList.add('dim');}); edgeEls.forEach((x,key)=>{if(!adjacent.some(e=>k(e.a,e.b)===key))x.classList.add('dim');}); nodeEls.get(id).classList.add('selected'); addHalo(id); const linkState=n.hasStableLink?`<div class="panel-title" style="margin-top:4px">Stable lyrical-repertoire neighbours</div><div class="neighbors">${adjacent.map(e=>edgeSummary(e,id)).join('')}</div>`:'<div class="notice">No stable line was retained for this label. This is not evidence of isolation: the dot remains in the semantic projection, but no pair met the stricter mutual-top-five rule in both duplicate-control representations.</div>'; $('detail-kicker').textContent='Selected source corpus label'; $('detail-title').textContent=n.label; $('detail-body').innerHTML=`<div class="source-note">Source corpus label · identity not externally verified.</div><div class="rule"></div><div class="facts"><div class="fact"><b>${showNumber(n.songs)}</b><span>clean songs</span></div><div class="fact"><b>${showNumber(n.degree)}</b><span>stable-line neighbours</span></div><div class="fact"><b>${showNumber(n.primaryMass,1)}</b><span>duplicate-controlled lyric-evidence mass</span></div><div class="fact"><b>${pct(n.sharedDropShare)}</b><span>primary mass removed as exact cross-label text</span></div></div><div class="notice">Dot position: a two-dimensional summary of consensus lyric semantics. Nearness is approximate; the line test is stricter.</div>${linkState}`; $('detail-body').querySelectorAll('[data-edge]').forEach(button=>button.addEventListener('click',()=>selectEdge(button.dataset.edge))); }
    function selectEdge(key) { const e=edgeByKey.get(key); if(!e)return; selectedNode=null; selectedEdge=key; $('focusSelect').value=''; clearEmphasis(); const a=nodeById.get(e.a), b=nodeById.get(e.b); nodeEls.forEach((x,id)=>{if(id!==e.a&&id!==e.b)x.classList.add('dim');}); edgeEls.forEach((x,id)=>{if(id!==key)x.classList.add('dim');}); edgeEls.get(key).classList.add('selected'); nodeEls.get(e.a).classList.add('selected'); nodeEls.get(e.b).classList.add('selected'); addHalo(e.a); addHalo(e.b); $('detail-kicker').textContent='Stable pairwise proximity'; $('detail-title').textContent=`${a.label} ↔ ${b.label}`; $('detail-body').innerHTML=`<div class="source-note">Two source corpus labels · neither is an externally verified identity or a claimed social relationship.</div><div class="rule"></div><div class="facts"><div class="fact"><b>${e.primaryRanks[0]} ↔ ${e.primaryRanks[1]}</b><span>mutual ranks in duplicate-weighted representation</span></div><div class="fact"><b>${e.sensitivityRanks[0]} ↔ ${e.sensitivityRanks[1]}</b><span>mutual ranks after shared text removal</span></div><div class="fact"><b>${e.primaryPercentile.toFixed(1)}</b><span>primary pair percentile</span></div><div class="fact"><b>${e.sensitivityPercentile.toFixed(1)}</b><span>sensitivity pair percentile</span></div></div><div class="notice">Why this line appears: each label places the other within its top-${G.metrics.topK} lyrical-repertoire neighbours in both representations. The percentiles compare this pair with all eligible label pairs; they are not probabilities.</div><div class="neighbors"><button class="neighbor" type="button" data-node="${esc(e.a)}"><b>Inspect ${esc(a.label)}</b><small>See its complete retained neighbourhood.</small></button><button class="neighbor" type="button" data-node="${esc(e.b)}"><b>Inspect ${esc(b.label)}</b><small>See its complete retained neighbourhood.</small></button></div>`; $('detail-body').querySelectorAll('[data-node]').forEach(button=>button.addEventListener('click',()=>selectNode(button.dataset.node))); }
    function reset() { selectedNode=null; selectedEdge=null; $('focusSelect').value=''; clearEmphasis(); $('detail-kicker').textContent='How to use the map'; $('detail-title').textContent='Semantic position, then strict evidence'; $('detail-body').className='empty'; $('detail-body').textContent=`All ${showNumber(G.metrics.projectionLabels)} dots are graph-eligible source labels. Nearness summarizes a 2D semantic projection (${pct(G.projection.varianceExplained2d)} of consensus-vector variation); a line is a separate, stricter stable-proximity result.`; }
    nodeEls.forEach((circle,id)=> { const n=nodeById.get(id); circle.addEventListener('click',()=>selectNode(id)); circle.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectNode(id);}}); circle.addEventListener('pointermove',event=>{if(selectedNode||selectedEdge)return; const box=wrap.getBoundingClientRect(); const linkText=n.hasStableLink?`${n.degree} stable neighbour${n.degree===1?'':'s'}`:'no retained stable line'; tip.innerHTML=`<b>${esc(n.label)}</b><br>${linkText} · ${showNumber(n.songs)} clean songs`; tip.style.left=(event.clientX-box.left)+'px';tip.style.top=(event.clientY-box.top)+'px';tip.classList.add('show');}); circle.addEventListener('pointerleave',()=>tip.classList.remove('show')); });
    hitEls.forEach((line,key)=> { const e=edgeByKey.get(key); line.addEventListener('click',()=>selectEdge(key)); line.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectEdge(key);}}); line.addEventListener('pointermove',event=>{if(selectedNode||selectedEdge)return; const box=wrap.getBoundingClientRect(),a=nodeById.get(e.a),b=nodeById.get(e.b); tip.innerHTML=`<b>${esc(a.label)} ↔ ${esc(b.label)}</b><br>stable mutual top-${G.metrics.topK} pair`;tip.style.left=(event.clientX-box.left)+'px';tip.style.top=(event.clientY-box.top)+'px';tip.classList.add('show');}); line.addEventListener('pointerleave',()=>tip.classList.remove('show')); });
    $('reset').addEventListener('click',reset);
    $('focusSelect').addEventListener('change',event=>{ if(event.target.value) selectNode(event.target.value); else reset(); });
    document.addEventListener('keydown',event=>{ if(event.key==='Escape') reset(); });
    $('provenance').innerHTML=`Built from the validated <code>chinese-rap-lyrical-repertoire-graph-v2</code> public graph. The map shows a deterministic consensus semantic PCA across ${showNumber(G.projection.population)} eligible labels (${pct(G.projection.varianceExplained2d)} variance represented in two dimensions); retained lines use the separate mutual-top-${G.metrics.topK} stability rule. It contains ${showNumber(G.metrics.cleanChunks)} clean lyric chunks from ${showNumber(G.metrics.cleanSongs)} songs. No lyric text, song/chunk IDs, membership records, or embeddings are included in this page.`;
  })();
  </script>
</body>
</html>
""".replace("__PAYLOAD__", payload_json)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temp_path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def public_readme() -> str:
    return """# Chinese Rap Lyrical Repertoire Map

Open `index.html` in a modern browser. The page is self-contained and can be
opened from disk; it makes no network requests.

Each dot is a **source artist label**, not an externally verified performer
identity. All graph-eligible labels are placed in a deterministic two-dimensional
PCA projection of their consensus (primary plus shared-text-exclusion) lyric
representations. Thus, nearby dots are an approximate semantic display; the
two-dimensional projection does not itself establish a graph relationship.

A visible line is a conservative **lyrical-repertoire proximity**:
the labels are mutual top-five semantic neighbours both (1) using
duplicate-weighted clean lyric chunks and (2) after exact clean text shared
across source labels is removed. Both representations must also retain at least
five clean songs and 20 effective-text mass. It is not evidence of a social relationship,
collaboration, genre, hometown, beat, voice, Flow, or real-world identity.

The source graph includes only songs whose canonical
`artist_title_comparison_eligible` field is `true`; it does not silently rename
or externally correct a source label.

The source graph was validated before this derivative artifact was built. This
page intentionally contains no lyric text, song/chunk IDs, membership data, or
embeddings.
"""


def method_text() -> str:
    return """# Explorer method note

## Research question

Which source-labelled Chinese-rap lyric corpora retain nearby semantic
repertoires once repeated text and exact text shared across labels are
controlled?

## Display rule

The map shows every graph-eligible node from the validated v2 graph. Its
coordinates are a deterministic two-dimensional PCA of normalized consensus
(primary plus sensitivity) lyric-repertoire centroids. Spatial proximity is an
approximate semantic display; the two dimensions do not fully preserve the
original representation. An edge is retained iff it is a reciprocal top-five
neighbour pair under both the duplicate-weighted representation and the
exact-shared-text-exclusion sensitivity representation. Node size represents
duplicate-controlled lyric-evidence mass, not popularity or importance. A gray
node without a line remains eligible and projected; it simply has no pair that
passed the stricter stable-edge rule.

## Interpretation boundary

This explorer makes corpus-level text claims only. It does not infer who a
person is, whether two performers know or collaborate with one another, their
style/genre, place, voice, beat, or Flow. Source artist labels are retained as
labels because external identity verification is outside this artifact.
"""


def write_artifact() -> None:
    payload, lineage = make_payload()
    stage = Path(tempfile.mkdtemp(prefix=f".{OUT_DIR.name}.stage-", dir=OUT_DIR.parent))
    try:
        atomic_write_text(stage / "index.html", html_document(payload))
        atomic_write_text(stage / "README.md", public_readme())
        atomic_write_text(stage / "method.md", method_text())
        manifest = {
            "artifact_id": ARTIFACT_ID,
            "version": VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "claim_boundary": "interactive display of validated source-label lyrical-repertoire proximity only; not a real-world rapper relationship, identity, affiliation, style, genre, or audio-performance claim",
            "source_lineage": lineage,
            "embedded_graph_payload_sha256": sha256_json(payload),
            "counts": {
                "visible_projection_labels": len(payload["nodes"]),
                "visible_stable_linked_labels": sum(1 for node in payload["nodes"] if node["hasStableLink"]),
                "visible_stable_edges": len(payload["edges"]),
            },
            "privacy": "self-contained page contains aggregate source-label graph data only; no lyric text, song/chunk IDs, membership records, or embeddings",
        }
        atomic_write_text(stage / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        validate_artifact(stage, payload, manifest, allow_missing_validation=True)
        if OUT_DIR.exists():
            if not OUT_DIR.is_dir():
                raise RuntimeError(f"Explorer target is not a directory: {OUT_DIR}")
            existing_manifest = OUT_DIR / "manifest.json"
            if not existing_manifest.is_file() or json.loads(existing_manifest.read_text(encoding="utf-8")).get("artifact_id") != ARTIFACT_ID:
                raise RuntimeError("Refusing to replace a directory that is not this explorer artifact.")
            shutil.rmtree(OUT_DIR)
        os.replace(stage, OUT_DIR)
        validate_artifact(OUT_DIR, payload, manifest, allow_missing_validation=False)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_artifact(root: Path, expected: dict[str, Any], manifest: dict[str, Any], *, allow_missing_validation: bool) -> None:
    required = {"index.html", "README.md", "method.md", "manifest.json"}
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if not required.issubset(actual):
        raise RuntimeError(f"Explorer inventory incomplete: {actual}")
    unexpected = actual - required - {"validation.json"}
    if unexpected:
        raise RuntimeError(f"Explorer inventory contains unexpected files: {sorted(unexpected)}")

    page = (root / "index.html").read_text(encoding="utf-8")
    embedded_marker = '<script id="graph-data" type="application/json">'
    if embedded_marker not in page:
        raise RuntimeError("Explorer page does not have an embedded graph payload.")
    payload_text = page.split(embedded_marker, 1)[1].split("</script>", 1)[0]
    actual_payload = json.loads(payload_text)
    if actual_payload != expected:
        raise RuntimeError("Embedded graph payload does not match current graph inputs.")
    actual_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if actual_manifest != manifest:
        raise RuntimeError("Explorer manifest does not match the current deterministic build payload.")
    if actual_manifest.get("embedded_graph_payload_sha256") != sha256_json(actual_payload):
        raise RuntimeError("Explorer manifest does not authenticate the embedded graph payload.")
    expected_docs = {"README.md": public_readme(), "method.md": method_text()}
    actual_docs = {name: (root / name).read_text(encoding="utf-8") for name in expected_docs}
    if actual_docs != expected_docs:
        raise RuntimeError("Explorer documentation does not recompute exactly.")

    forbidden = ("song_id", "chunk_id", "analysis_text", "lyric_text", "embedding", "artist_chunk_membership")
    payload_keys = json.dumps(actual_payload, ensure_ascii=False).casefold()
    hits = [token for token in forbidden if token in payload_keys]
    if hits:
        raise RuntimeError(f"Public explorer payload exposes forbidden fields: {hits}")

    source_manifest_hash = sha256_file(GRAPH_DIR / "manifest.json")
    if manifest["source_lineage"]["source_graph_manifest_sha256"] != source_manifest_hash:
        raise RuntimeError("Source graph manifest changed during explorer build.")
    for name, expected_hash in manifest["source_lineage"]["required_public_input_sha256"].items():
        if sha256_file(GRAPH_DIR / name) != expected_hash:
            raise RuntimeError(f"Source graph public input changed during explorer build: {name}")

    report = {
        "artifact_id": ARTIFACT_ID,
        "status": "pass",
        "checks": [
            {"name": "source_graph_validation_passing", "passed": True},
            {"name": "embedded_payload_exactly_matches_derivative_public_data", "passed": True},
            {"name": "manifest_and_documentation_recompute_exactly", "passed": True},
            {"name": "public_payload_has_no_lyric_or_identifier_fields", "passed": True},
            {"name": "source_graph_manifest_hash_current", "passed": True},
            {"name": "static_single_file_explorer_has_no_remote_dependencies", "passed": True},
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
    }
    if allow_missing_validation:
        return
    atomic_write_text(root / "validation.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    write_artifact()
    print(f"Built {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
