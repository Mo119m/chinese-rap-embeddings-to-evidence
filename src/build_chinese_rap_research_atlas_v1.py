#!/usr/bin/env python3
"""Build the result-first Chinese Rap Research Atlas centerpiece."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_chinese_rap_lyrical_repertoire_explorer_v1 as graph_explorer


VERSION = "1.0.0"
ARTIFACT_ID = "chinese-rap-research-atlas-v1"
ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = ROOT / "outputs" / "chinese-rap-interpretable-profiles-v1"
BOOTSTRAP_DIR = ROOT / "outputs" / "chinese-rap-edge-bootstrap-v1"
OUT_DIR = ROOT / "outputs" / ARTIFACT_ID
BOOTSTRAP_DISPLAY_GATE = 0.50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def require_profiles() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    required = {
        "analysis_summary.json",
        "source_label_profiles.json",
        "stable_link_explanations.json",
        "validation.json",
        "manifest.json",
    }
    missing = [name for name in required if not (PROFILE_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing interpretable-profile inputs: {missing}")
    validation = json.loads((PROFILE_DIR / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "pass" or not all(item.get("passed") for item in validation.get("checks", [])):
        raise RuntimeError("The profile artifact validation is not passing.")
    manifest = json.loads((PROFILE_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_id") != "chinese-rap-interpretable-profiles-v1":
        raise RuntimeError("Unexpected profile artifact.")
    for name, record in manifest.get("files", {}).items():
        path = PROFILE_DIR / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Profile artifact hash mismatch: {name}")
    profiles = json.loads((PROFILE_DIR / "source_label_profiles.json").read_text(encoding="utf-8"))
    explanations = json.loads((PROFILE_DIR / "stable_link_explanations.json").read_text(encoding="utf-8"))
    summary = json.loads((PROFILE_DIR / "analysis_summary.json").read_text(encoding="utf-8"))
    lineage = {
        "profile_manifest_sha256": sha256_file(PROFILE_DIR / "manifest.json"),
        "profile_validation_sha256": sha256_file(PROFILE_DIR / "validation.json"),
    }
    return profiles, explanations, summary, lineage


def require_bootstrap() -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, str]]:
    required = {"analysis_summary.json", "stable_edge_bootstrap.csv", "validation.json", "manifest.json"}
    missing = [name for name in required if not (BOOTSTRAP_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing edge-bootstrap inputs: {missing}")
    validation = json.loads((BOOTSTRAP_DIR / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "pass" or not all(item.get("passed") for item in validation.get("checks", [])):
        raise RuntimeError("The edge-bootstrap validation is not passing.")
    manifest = json.loads((BOOTSTRAP_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_id") != "chinese-rap-edge-bootstrap-v1":
        raise RuntimeError("Unexpected edge-bootstrap artifact.")
    for name, record in manifest.get("files", {}).items():
        path = BOOTSTRAP_DIR / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Edge-bootstrap artifact hash mismatch: {name}")
    import csv
    with (BOOTSTRAP_DIR / "stable_edge_bootstrap.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {
        row["edge_key"]: {
            "probability": float(row["two_representation_edge_probability"]),
            "primaryProbability": float(row["primary_mutual_probability"]),
            "sensitivityProbability": float(row["sensitivity_mutual_probability"]),
            "band": row["bootstrap_band"],
        }
        for row in rows
    }
    if len(by_key) != 86:
        raise RuntimeError("Unexpected bootstrap edge count.")
    summary = json.loads((BOOTSTRAP_DIR / "analysis_summary.json").read_text(encoding="utf-8"))
    lineage = {
        "bootstrap_manifest_sha256": sha256_file(BOOTSTRAP_DIR / "manifest.json"),
        "bootstrap_validation_sha256": sha256_file(BOOTSTRAP_DIR / "validation.json"),
    }
    return by_key, summary, lineage


def key(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


def make_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    graph, graph_lineage = graph_explorer.make_payload()
    profiles, explanations, profile_summary, profile_lineage = require_profiles()
    bootstrap_by_key, bootstrap_summary, bootstrap_lineage = require_bootstrap()
    profile_by_id = {item["id"]: item for item in profiles}
    explanation_by_key = {item["key"]: item for item in explanations}
    graph_ids = {item["id"] for item in graph["nodes"]}
    edge_keys = {key(item["a"], item["b"]) for item in graph["edges"]}
    if set(profile_by_id) != graph_ids or len(profile_by_id) != len(profiles):
        raise RuntimeError("Profiles do not map one-to-one to graph nodes.")
    if set(explanation_by_key) != edge_keys or len(explanation_by_key) != len(explanations):
        raise RuntimeError("Explanations do not map one-to-one to stable graph edges.")
    for node in graph["nodes"]:
        node["profile"] = profile_by_id[node["id"]]
    for edge in graph["edges"]:
        edge["explanation"] = explanation_by_key[key(edge["a"], edge["b"])]
        edge["bootstrap"] = bootstrap_by_key[key(edge["a"], edge["b"])]
        edge["bootstrapSupported"] = edge["bootstrap"]["probability"] >= BOOTSTRAP_DISPLAY_GATE
    supported_degree: dict[str, int] = {identifier: 0 for identifier in graph_ids}
    for edge in graph["edges"]:
        if edge["bootstrapSupported"]:
            supported_degree[edge["a"]] += 1
            supported_degree[edge["b"]] += 1
    for node in graph["nodes"]:
        node["bootstrapSupportedDegree"] = supported_degree[node["id"]]
        node["hasBootstrapSupportedLink"] = supported_degree[node["id"]] > 0
    candidates = sorted(
        graph["nodes"],
        key=lambda item: (
            -int(item["hasBootstrapSupportedLink"]),
            -item["bootstrapSupportedDegree"],
            -len(item["profile"]["characteristicTerms"]),
            -item["degree"],
            item["label"].casefold(),
        ),
    )
    default_node = candidates[0]["id"]
    payload = {
        "artifact": ARTIFACT_ID,
        "version": VERSION,
        "title": "Chinese Rap Lyrical Fingerprints",
        "subtitle": "Choose a rapper or group label to see its distinctive words, common ending sounds, writing habits, and closest repeatable lyric matches.",
        "defaultNode": default_node,
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "projection": graph["projection"],
        "metrics": {
            **graph["metrics"],
            "readyProfiles": profile_summary["counts"]["ready_profiles"],
            "linksWithInterpretiveSignal": profile_summary["counts"]["links_with_interpretable_signal"],
            "semanticOnlyLinks": profile_summary["counts"]["semantic_only_links"],
            "bootstrapDisplayGate": BOOTSTRAP_DISPLAY_GATE,
            "bootstrapSupportedEdges": bootstrap_summary["counts"]["edges_probability_at_least_0_50"],
            "candidateEdges": len(graph["edges"]) - bootstrap_summary["counts"]["edges_probability_at_least_0_50"],
            "bootstrapReplicates": bootstrap_summary["counts"]["replicates"],
        },
        "method": {
            "relation": "A line is retained only when two source-credit-label repertoires are reciprocal top-5 neighbours in both the duplicate-weighted BGE-M3 representation and the exact-cross-label-shared-text-exclusion representation.",
            "profiles": profile_summary["methods"],
            "boundary": profile_summary["claim_boundary"],
            "bootstrap": bootstrap_summary["method"],
        },
    }
    lineage = {**graph_lineage, **profile_lineage, **bootstrap_lineage}
    return payload, lineage


def html_document(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>Chinese Rap Lyrical Fingerprints</title>
  <style>
    :root{--bg:#07101c;--panel:#0d1928;--panel2:#111f31;--line:#23364c;--ink:#f2f6fb;--muted:#9fb0c4;--blue:#6fb8ff;--teal:#5eead4;--purple:#c19cff;--orange:#ffb45b;--grey:#708198;--focus:#f9dc75;--shadow:0 24px 70px #0008}
    *{box-sizing:border-box}html{background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}body{margin:0;min-height:100vh;background:radial-gradient(circle at 6% -10%,#16496d52,transparent 30%),radial-gradient(circle at 94% 0,#43377735,transparent 26%),var(--bg)}button,input{font:inherit}.shell{width:min(1560px,calc(100% - 32px));margin:auto;padding:28px 0 42px}.mast{display:flex;align-items:end;justify-content:space-between;gap:28px;margin-bottom:20px}.eyebrow{font-size:.72rem;text-transform:uppercase;letter-spacing:.15em;color:var(--teal);font-weight:800;margin:0 0 9px}.mast h1{font-size:clamp(2rem,4vw,3.8rem);line-height:.98;letter-spacing:-.055em;margin:0}.lede{color:var(--muted);line-height:1.55;max-width:730px;margin:13px 0 0;font-size:.96rem}.search{display:flex;gap:8px;min-width:min(430px,40vw)}.search input{width:100%;background:#081321;color:var(--ink);border:1px solid #36506c;border-radius:11px;padding:11px 13px;outline:none}.search input:focus{border-color:var(--focus);box-shadow:0 0 0 3px #f9dc7522}.search button,.icon-button{border:1px solid #38526e;background:#102238;color:var(--ink);border-radius:10px;padding:10px 13px;cursor:pointer}.search button:hover,.icon-button:hover{border-color:var(--teal)}button:focus-visible,summary:focus-visible,input:focus-visible{outline:3px solid var(--focus);outline-offset:2px}.layout{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(380px,.78fr);gap:16px;align-items:stretch}.card{border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#101d2deb,#091522f2);box-shadow:var(--shadow);overflow:hidden}.map-head{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:14px 16px;border-bottom:1px solid var(--line)}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:.72rem}.legend span{display:inline-flex;align-items:center;gap:5px}.swatch{width:20px;height:3px;border-radius:4px;background:var(--grey)}.swatch.language{background:var(--purple)}.swatch.lineEnding{background:var(--orange)}.swatch.form{background:var(--teal)}.swatch.semanticOnly{height:0;border-top:2px dashed var(--grey);background:none}.map-tools{display:flex;gap:6px}.icon-button{padding:7px 10px;font-size:.78rem}.network{position:relative;height:min(720px,72vh);min-height:560px;overflow:hidden;background:radial-gradient(circle at 50% 50%,#1b304d48,transparent 52%),#071321}.network svg{display:block;width:100%;height:100%;touch-action:none;cursor:grab}.network svg.dragging{cursor:grabbing}.edge{fill:none;stroke-width:1.5;opacity:.68;vector-effect:non-scaling-stroke;cursor:pointer}.edge.language{stroke:var(--purple)}.edge.lineEnding{stroke:var(--orange)}.edge.form{stroke:var(--teal)}.edge.semanticOnly{stroke:var(--grey);stroke-dasharray:5 5;opacity:.46}.edge.dim,.node.dim{opacity:.07}.edge.selected{stroke-width:4;opacity:1}.node{cursor:pointer;transition:opacity .15s}.node circle{vector-effect:non-scaling-stroke;stroke-width:1.5}.node.linked circle{fill:#79bfff;stroke:#d5edff}.node.unlinked circle{fill:#071321;stroke:#6f8299}.node:hover circle,.node.selected circle{stroke:var(--focus);stroke-width:3}.node-label{fill:#f7fbff;paint-order:stroke;stroke:#07101c;stroke-width:4px;stroke-linejoin:round;font-size:12px;font-weight:750;pointer-events:none}.tooltip{position:absolute;pointer-events:none;z-index:4;max-width:260px;padding:9px 11px;border:1px solid #425a75;border-radius:10px;background:#07101eee;box-shadow:0 10px 32px #0009;font-size:.76rem;line-height:1.45;opacity:0;transform:translate(10px,10px)}.tooltip b{display:block;color:#fff;font-size:.88rem;margin-bottom:3px}.panel{height:min(778px,calc(72vh + 58px));min-height:618px;overflow:auto;padding:22px 22px 18px}.panel::-webkit-scrollbar{width:9px}.panel::-webkit-scrollbar-thumb{background:#31465f;border-radius:8px}.kicker{color:var(--teal);font-size:.69rem;font-weight:800;text-transform:uppercase;letter-spacing:.13em}.panel h2{font-size:clamp(1.65rem,3vw,2.6rem);line-height:1.02;letter-spacing:-.04em;margin:7px 0 6px;overflow-wrap:anywhere}.quick-summary{color:#c4cfdd;font-size:.87rem;line-height:1.5;margin:0 0 19px}.section{border-top:1px solid var(--line);padding-top:17px;margin-top:17px}.section h3{margin:0 0 10px;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:#b9c8d9}.chips{display:flex;flex-wrap:wrap;gap:7px}.chip{border:1px solid #3b506c;background:#132239;border-radius:999px;padding:6px 9px;color:#f0f6ff;font-size:.82rem}.empty-evidence{color:var(--muted);font-size:.82rem;line-height:1.5}.ending-list{display:grid;gap:9px}.ending-row{display:grid;grid-template-columns:42px minmax(0,1fr) 44px;align-items:center;gap:9px;font-size:.78rem}.ending-row b{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--orange);font-size:.9rem}.bar{height:8px;border-radius:8px;background:#1d2e43;overflow:hidden}.bar i{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--orange),#ffd086)}.ending-row span{text-align:right;color:var(--muted)}.echo{margin-top:12px;padding:11px 12px;border-left:3px solid var(--orange);background:#ffb45b0c;color:#dce4ee;font-size:.81rem;line-height:1.5}.traits{display:grid;gap:12px}.trait-head{display:flex;justify-content:space-between;gap:12px;font-size:.8rem}.trait-head span{color:var(--muted)}.trait .bar i{background:linear-gradient(90deg,#4f7fff,var(--teal))}.neighbors{display:grid;gap:8px}.neighbor{width:100%;text-align:left;border:1px solid #2e445e;border-radius:12px;background:#0b1727;color:var(--ink);padding:11px 12px;cursor:pointer}.neighbor:hover{border-color:var(--blue);background:#10233a}.neighbor b{display:block;font-size:.91rem}.neighbor span{display:block;color:var(--muted);font-size:.76rem;line-height:1.4;margin-top:3px}.signal-card{border:1px solid #334a63;border-radius:13px;padding:13px;margin:9px 0;background:#0a1726}.signal-card.language{border-left:4px solid var(--purple)}.signal-card.lineEnding{border-left:4px solid var(--orange)}.signal-card.form{border-left:4px solid var(--teal)}.signal-card h3{font-size:.86rem;margin:0 0 7px;text-transform:none;letter-spacing:0;color:#edf5ff}.signal-card p{margin:0;color:#bbc9d9;font-size:.8rem;line-height:1.45}.pair-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.mini{border:1px solid #2e435c;border-radius:12px;padding:12px;background:#0a1625}.mini h3{font-size:1rem;text-transform:none;letter-spacing:0;color:#fff;overflow-wrap:anywhere}.mini .chips{gap:5px}.mini .chip{font-size:.72rem;padding:4px 7px}.mini p{font-size:.72rem;color:var(--muted);line-height:1.45}.mini button{margin-top:9px;width:100%;border:1px solid #405a76;background:#102238;color:#fff;border-radius:8px;padding:7px;cursor:pointer}.method{border-top:1px solid var(--line);margin-top:20px;padding-top:14px;color:var(--muted);font-size:.74rem;line-height:1.55}.method summary{cursor:pointer;color:#c7d4e4;font-weight:700}.method p{margin:9px 0 0}.foot{color:#72849a;font-size:.7rem;line-height:1.5;margin:13px 2px 0}.status-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:1050px){.mast{display:block}.search{margin-top:17px;min-width:0;width:min(560px,100%)}.layout{grid-template-columns:1fr}.panel{height:auto;min-height:0}.network{height:620px}}@media(max-width:560px){.shell{width:min(100% - 18px,1560px);padding-top:20px}.mast h1{font-size:2.45rem}.map-head{align-items:flex-start}.legend{gap:8px}.network{height:520px;min-height:480px}.panel{padding:18px 15px}.pair-grid{grid-template-columns:1fr}.ending-row{grid-template-columns:38px minmax(0,1fr) 38px}.lede{font-size:.87rem}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
  </style>
  <style>
    .edge.semanticOnly{stroke-dasharray:5 5;opacity:.58}
    .edge.candidate{stroke-dasharray:2 7;opacity:.23}
    .edge.hidden{display:none}
    .node.hidden{display:none}
    .node.supported circle{fill:#79bfff;stroke:#d5edff}
    .node.candidateOnly circle{fill:#19314b;stroke:#7090af}
    .node.unlinked circle{fill:#071321;stroke:#52677d}
    .candidate-button[aria-pressed="true"]{border-color:var(--orange);color:#ffe0b5;background:#2b2117}
    .support-pill{display:inline-flex;align-items:center;border:1px solid #45617e;border-radius:999px;background:#10243a;color:#ddecff;padding:6px 9px;font-size:.76rem;font-weight:750;margin:2px 0 14px}
    .support-pill.candidate{border-color:#6b5a43;background:#271e15;color:#ffdbab}
    .search-error{display:none;grid-column:1/-1;color:#ffbd9b;font-size:.76rem;line-height:1.3}
    .search.has-error{display:grid;grid-template-columns:minmax(0,1fr) auto}
    .search.has-error .search-error{display:block}
    @media(max-width:760px){.map-head{display:grid}.map-tools{flex-wrap:wrap}.candidate-button{order:2;width:100%}}
  </style>
</head>
<body>
  <main class="shell">
    <header class="mast">
      <div><p class="eyebrow">Chinese Rap Lyrics Atlas</p><h1>Lyrical Fingerprints</h1><p class="lede">Choose a rapper or group label to see its distinctive words, common ending sounds, writing habits, and closest repeatable lyric matches. Click a line to see what the two profiles share.</p></div>
      <form class="search" id="search-form"><input id="search" list="label-list" autocomplete="off" placeholder="Search a rapper or group…" aria-label="Search a rapper or group label"><datalist id="label-list"></datalist><button type="submit">View profile</button><small class="search-error" id="search-error">No matching label. Try fewer characters or choose a suggestion.</small></form>
    </header>
    <section class="layout">
      <div class="card">
        <div class="map-head"><div class="legend" aria-label="Lyric-match explanation legend"><span><i class="swatch language"></i>shared distinctive words</span><span><i class="swatch lineEnding"></i>similar written endings</span><span><i class="swatch form"></i>same writing habit</span><span><i class="swatch semanticOnly"></i>overall wording match only</span></div><div class="map-tools"><button class="icon-button candidate-button" id="candidates" type="button" aria-pressed="false">Show exploratory matches</button><button class="icon-button" id="zoom-out" type="button" aria-label="Zoom out">−</button><button class="icon-button" id="fit" type="button">Reset view</button><button class="icon-button" id="zoom-in" type="button" aria-label="Zoom in">+</button></div></div>
        <div class="network" id="network"><svg id="svg" viewBox="0 0 1000 680" role="img" aria-label="Interactive lyric-similarity map. Select a point for a profile or a line to compare two profiles."><g id="viewport"><g id="edge-layer"></g><g id="node-layer"></g><g id="label-layer"></g></g></svg><div class="tooltip" id="tooltip"></div></div>
      </div>
      <aside class="card panel" id="panel"></aside>
    </section>
    <div class="status-only" id="announce" aria-live="polite"></div>
    <p class="foot">Names come from corpus credits and have not been independently verified. No full lyrics are shown.</p>
  </main>
  <script id="atlas-data" type="application/json">__PAYLOAD__</script>
  <script>
  (()=>{
    const G=JSON.parse(document.getElementById('atlas-data').textContent),$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
    const nodeById=new Map(G.nodes.map(n=>[n.id,n])),edgeKey=(a,b)=>[a,b].sort().join('|'),edgeByKey=new Map(G.edges.map(e=>[edgeKey(e.a,e.b),e]));
    const edgeEls=new Map(),nodeEls=new Map();let selectedNode=null,selectedEdge=null,showCandidates=false,transform={x:0,y:0,k:1},drag=null;
    const X=n=>70+(n.x+1)*430,Y=n=>45+(1-(n.y+1)/2)*590;
    const pct=x=>`${Math.round(Number(x)*100)}%`,plainDescriptor=(key,p)=>key==='short'?(p>=70?'mostly shorter lines':p<=30?'mostly longer lines':'mixed line lengths'):key==='repeat'?(p>=70?'frequent line repetition':p<=30?'rare line repetition':'typical line repetition'):(p>=70?'frequent Chinese–English mixing':p<=30?'rare Chinese–English mixing':'typical Chinese–English mixing'),traitLabel=key=>({short:'Line length',repeat:'Line repetition',mix:'Chinese–English mix'}[key]||key);
    const adjacency=id=>G.edges.filter(e=>(showCandidates||e.bootstrapSupported)&&(e.a===id||e.b===id)).sort((a,b)=>b.bootstrap.probability-a.bootstrap.probability||Math.max(...b.explanation.signals.map(s=>s.percentile),0)-Math.max(...a.explanation.signals.map(s=>s.percentile),0));
    const formCopy=item=>({'high Chinese-English mixing':'Both mix Chinese and English frequently.','low Chinese-English mixing':'Both rarely mix Chinese and English.','high repeated-line use':'Both repeat exact lines frequently.','low repeated-line use':'Both rarely repeat exact lines.','high short-line writing':'Both use mostly shorter written lines.','low short-line writing':'Both use mostly longer written lines.'}[item]||item),signalTitle=s=>({language:'Shared distinctive words',lineEnding:'Similar written endings',form:'Same writing habit'}[s.kind]||'Shared signal'),signalBody=s=>s.kind==='language'?s.items.join(' · '):s.kind==='lineEnding'?`Both often end written lines with ${s.items.join(', ')}.`:s.items.map(formCopy).join(' '),summary=e=>e.explanation.signals[0]?`${signalTitle(e.explanation.signals[0])}: ${signalBody(e.explanation.signals[0])}`:'Overall wording is similar, but no single shared word, written ending, or writing habit dominates.';
    function create(tag,attrs={}){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));return el}
    function renderGraph(){const edges=$('edge-layer'),nodes=$('node-layer');G.edges.forEach(e=>{const a=nodeById.get(e.a),b=nodeById.get(e.b),kind=e.explanation.dominantSignal,state=e.bootstrapSupported?'supported':'candidate hidden',line=create('line',{x1:X(a),y1:Y(a),x2:X(b),y2:Y(b),class:`edge ${kind} ${state}`,'data-key':edgeKey(e.a,e.b)});line.style.strokeWidth=String(1.2+e.bootstrap.probability*3);line.addEventListener('click',ev=>{ev.stopPropagation();selectEdge(edgeKey(e.a,e.b))});edges.append(line);edgeEls.set(edgeKey(e.a,e.b),line)});G.nodes.forEach(n=>{const state=n.hasBootstrapSupportedLink?'supported':n.hasStableLink?'candidateOnly hidden':'unlinked hidden',g=create('g',{class:`node ${state}`,transform:`translate(${X(n)} ${Y(n)})`});const circle=create('circle',{r:n.hasBootstrapSupportedLink?5.8:n.hasStableLink?4.3:3.5});g.append(circle);g.addEventListener('click',ev=>{ev.stopPropagation();selectNode(n.id)});g.addEventListener('pointerenter',ev=>showTip(ev,n));g.addEventListener('pointermove',ev=>moveTip(ev));g.addEventListener('pointerleave',hideTip);nodes.append(g);nodeEls.set(n.id,g)});$('svg').addEventListener('click',()=>selectNode(G.defaultNode));}
    function syncNodeVisibility(activeIds=[]){const active=new Set(activeIds);nodeEls.forEach((el,id)=>{const n=nodeById.get(id),visible=n.hasBootstrapSupportedLink||(showCandidates&&n.hasStableLink)||active.has(id);el.classList.toggle('hidden',!visible)})}
    function showTip(ev,n){const t=$('tooltip'),terms=n.profile.characteristicTerms.slice(0,2).map(x=>x.text).join(' · '),ending=n.profile.lineEndings.top[0]?.final;t.innerHTML=`<b>${esc(n.label)}</b>${terms?`Distinctive words: ${esc(terms)}<br>`:''}${ending?`Most common ending sound: -${esc(ending)}<br>`:''}Profile based on ${n.independentSongs} independent songs`;t.style.opacity='1';moveTip(ev)}function moveTip(ev){const box=$('network').getBoundingClientRect(),t=$('tooltip');let x=ev.clientX-box.left+10,y=ev.clientY-box.top+10;x=Math.min(x,box.width-t.offsetWidth-8);y=Math.min(y,box.height-t.offsetHeight-8);t.style.left=`${Math.max(6,x)}px`;t.style.top=`${Math.max(6,y)}px`}function hideTip(){$('tooltip').style.opacity='0'}
    function clear(){nodeEls.forEach(el=>el.classList.remove('dim','selected'));edgeEls.forEach(el=>el.classList.remove('dim','selected'));$('label-layer').replaceChildren()}
    function addLabel(id,primary=false){const n=nodeById.get(id),text=create('text',{x:X(n)+8,y:Y(n)-8,class:'node-label'});text.textContent=n.label;if(primary)text.setAttribute('style','fill:#f9dc75;font-size:14px');$('label-layer').append(text)}
    function termChips(profile,limit=8){const terms=profile.characteristicTerms.slice(0,limit);return terms.length?`<div class="chips">${terms.map(t=>`<span class="chip" title="Supported by ${t.supportSongs} songs">${esc(t.text)}</span>`).join('')}</div>`:'<div class="empty-evidence">No word is distinctive across enough songs yet.</div>'}
    function endingRows(profile,limit=5){const endings=profile.lineEndings.top.slice(0,limit);return endings.length?`<div class="ending-list">${endings.map(x=>`<div class="ending-row"><b>-${esc(x.final)}</b><div class="bar"><i style="width:${Math.min(100,x.share*400)}%"></i></div><span>${pct(x.share)}</span></div>`).join('')}</div>`:'<div class="empty-evidence">Too few independent lines to estimate ending sounds.</div>'}
    function echoCopy(profile){const e=profile.lineEndings.localEcho;if(!e)return'';const points=Math.round(e.lift*1000)/10,clear=e.bootstrap90[0]>0||e.bootstrap90[1]<0;if(!clear)return '<div class="echo"><b>Nearby ending-sound reuse:</b> uncertain; the difference may be chance.</div>';return `<div class="echo"><b>Nearby ending-sound reuse:</b> clear — the same ending sound recurs within four written lines ${points>=0?'more':'less'} often than expected (${points>=0?'+':''}${points.toFixed(1)} percentage points).</div>`}
    function traits(profile){return `<div class="traits">${profile.formTraits.map(t=>`<div class="trait"><div class="trait-head"><b>${esc(traitLabel(t.key))}</b><span>${esc(plainDescriptor(t.key,t.percentile))}</span></div><div class="bar"><i style="width:${t.percentile}%"></i></div></div>`).join('')}</div>`}
    function profileSummary(n){const marked=[...n.profile.formTraits].sort((a,b)=>Math.abs(b.percentile-50)-Math.abs(a.percentile-50))[0],ending=n.profile.lineEndings.top[0]?.final,parts=[];if(marked)parts.push(`Standout writing habit: ${esc(plainDescriptor(marked.key,marked.percentile))}.`);if(ending)parts.push(`Most common ending sound: -${esc(ending)}.`);return parts.join(' ')||'Not enough independent-song evidence for a concise profile yet.'}
    function reliabilityDetails({node=null,edge=null}={}){if(edge){const support=Math.round(edge.bootstrap.probability*100);return `<details class="method"><summary>Reliability & limits</summary><p>This lyric match reappeared in ${support}% of ${G.metrics.bootstrapReplicates} song resamples and remained after repeated and cross-label shared lyrics were controlled. The listed words, ending sounds, and writing habits help interpret the match; they did not create it.</p><p>This is not evidence of collaboration, influence, identity, or a social relationship.</p></details>`}return `<details class="method"><summary>Reliability & limits</summary><p>This profile is based on ${node.independentSongs} independent songs. Only words and writing patterns supported across several songs are shown. Repeated or cross-label shared lyrics are controlled so they cannot dominate the profile.</p><p>Ending sounds are estimated from written Mandarin characters, not audio. The profile describes the name attached to songs in this corpus; it does not verify the person's identity, biography, preferences, performed rhymes, or flow.</p></details>`}
    function selectNode(id){const n=nodeById.get(id);if(!n)return;selectedNode=id;selectedEdge=null;syncNodeVisibility([id]);clear();const adjacent=adjacency(id),near=new Set([id,...adjacent.flatMap(e=>[e.a,e.b])]);nodeEls.forEach((el,key)=>{if(!near.has(key))el.classList.add('dim')});edgeEls.forEach((el,key)=>{if(!adjacent.some(e=>edgeKey(e.a,e.b)===key))el.classList.add('dim')});nodeEls.get(id).classList.add('selected');addLabel(id,true);adjacent.forEach(e=>addLabel(e.a===id?e.b:e.a));const neighbors=adjacent.length?`<div class="neighbors">${adjacent.map(e=>{const other=nodeById.get(e.a===id?e.b:e.a);return `<button class="neighbor" data-edge="${esc(edgeKey(e.a,e.b))}"><b>${esc(other.label)}</b><span>${Math.round(e.bootstrap.probability*100)}% repeatability · ${esc(summary(e))}</span></button>`}).join('')}</div>`:`<div class="empty-evidence">${n.hasStableLink&&!showCandidates?'No lyric match reappeared often enough to show. Use “Show exploratory matches” to inspect candidates.':'No repeatable lyric match is available for this profile.'}</div>`;const neighborHeading=showCandidates?'Exploratory lyric matches':'Closest repeatable matches';$('panel').innerHTML=`<div class="kicker">Lyric profile</div><h2>${esc(n.label)}</h2><p class="quick-summary">${profileSummary(n)}</p><section class="section"><h3>Distinctive words</h3>${termChips(n.profile)}</section><section class="section"><h3>Common ending sounds</h3>${endingRows(n.profile)}${echoCopy(n.profile)}</section><section class="section"><h3>Writing habits</h3>${traits(n.profile)}</section><section class="section"><h3>${neighborHeading}</h3>${neighbors}</section>${reliabilityDetails({node:n})}`;$('panel').querySelectorAll('[data-edge]').forEach(b=>b.addEventListener('click',()=>selectEdge(b.dataset.edge)));$('search').value=n.label;$('search-form').classList.remove('has-error');$('announce').textContent=`Selected ${n.label}`;}
    function miniProfile(n){return `<article class="mini"><h3>${esc(n.label)}</h3>${termChips(n.profile,4)}<p>Common ending sounds: ${n.profile.lineEndings.top.slice(0,3).map(x=>'-'+esc(x.final)).join(' · ')||'not enough written lines'}</p><button data-node="${esc(n.id)}">View full profile</button></article>`}
    function selectEdge(k){const e=edgeByKey.get(k);if(!e)return;selectedEdge=k;selectedNode=null;syncNodeVisibility([e.a,e.b]);clear();nodeEls.forEach((el,id)=>{if(id!==e.a&&id!==e.b)el.classList.add('dim')});edgeEls.forEach((el,key)=>{if(key!==k)el.classList.add('dim')});edgeEls.get(k).classList.add('selected');nodeEls.get(e.a).classList.add('selected');nodeEls.get(e.b).classList.add('selected');addLabel(e.a,true);addLabel(e.b,true);const a=nodeById.get(e.a),b=nodeById.get(e.b),signals=e.explanation.signals.length?e.explanation.signals.map(s=>`<article class="signal-card ${s.kind}"><h3>${esc(signalTitle(s))}</h3><p>${esc(signalBody(s))}</p></article>`).join(''):'<article class="signal-card"><h3>Overall wording match only</h3><p>The two lyric profiles remain close after repeated and cross-label shared lyrics were controlled, but no single shared word, written ending, or writing habit dominates the match.</p></article>',support=Math.round(e.bootstrap.probability*100),status=e.bootstrapSupported?'Repeatable lyric match':'Exploratory lyric match';$('panel').innerHTML=`<div class="kicker">${status}</div><h2>${esc(a.label)} ↔ ${esc(b.label)}</h2><div class="support-pill ${e.bootstrapSupported?'':'candidate'}">Reappeared in ${support}% of ${G.metrics.bootstrapReplicates} song resamples</div><section class="section"><h3>What they share</h3>${signals}</section><section class="section"><h3>Compare their lyric profiles</h3><div class="pair-grid">${miniProfile(a)}${miniProfile(b)}</div></section>${reliabilityDetails({edge:e})}`;$('panel').querySelectorAll('[data-node]').forEach(btn=>btn.addEventListener('click',()=>selectNode(btn.dataset.node)));$('announce').textContent=`Selected lyric match between ${a.label} and ${b.label}, ${support}% repeatability`;}
    function applyTransform(){const v=$('viewport');v.setAttribute('transform',`translate(${transform.x} ${transform.y}) scale(${transform.k})`)}function zoom(factor,cx=500,cy=340){const next=Math.max(.55,Math.min(5,transform.k*factor));transform.x=cx-(cx-transform.x)*(next/transform.k);transform.y=cy-(cy-transform.y)*(next/transform.k);transform.k=next;applyTransform()}function fit(){transform={x:0,y:0,k:1};applyTransform()}
    $('candidates').textContent=`Show ${G.metrics.candidateEdges} exploratory matches`;$('candidates').addEventListener('click',()=>{showCandidates=!showCandidates;$('candidates').setAttribute('aria-pressed',String(showCandidates));$('candidates').textContent=showCandidates?'Hide exploratory matches':`Show ${G.metrics.candidateEdges} exploratory matches`;edgeEls.forEach((el,k)=>{const e=edgeByKey.get(k);el.classList.toggle('hidden',!showCandidates&&!e.bootstrapSupported)});syncNodeVisibility(selectedNode?[selectedNode]:[]);const activeEdge=selectedEdge?edgeByKey.get(selectedEdge):null;if(activeEdge&&(showCandidates||activeEdge.bootstrapSupported))selectEdge(selectedEdge);else selectNode(selectedNode||G.defaultNode)});
    $('zoom-in').addEventListener('click',()=>zoom(1.25));$('zoom-out').addEventListener('click',()=>zoom(.8));$('fit').addEventListener('click',fit);$('svg').addEventListener('wheel',ev=>{ev.preventDefault();const box=$('svg').getBoundingClientRect(),cx=(ev.clientX-box.left)*1000/box.width,cy=(ev.clientY-box.top)*680/box.height;zoom(ev.deltaY<0?1.12:.89,cx,cy)},{passive:false});$('svg').addEventListener('pointerdown',ev=>{if(ev.target.closest('.node,.edge'))return;drag={x:ev.clientX,y:ev.clientY,tx:transform.x,ty:transform.y};$('svg').setPointerCapture(ev.pointerId);$('svg').classList.add('dragging')});$('svg').addEventListener('pointermove',ev=>{if(!drag)return;const box=$('svg').getBoundingClientRect();transform.x=drag.tx+(ev.clientX-drag.x)*1000/box.width;transform.y=drag.ty+(ev.clientY-drag.y)*680/box.height;applyTransform()});$('svg').addEventListener('pointerup',()=>{drag=null;$('svg').classList.remove('dragging')});
    const list=$('label-list');G.nodes.forEach(n=>{const o=document.createElement('option');o.value=n.label;list.append(o)});$('search-form').addEventListener('submit',ev=>{ev.preventDefault();const q=$('search').value.trim().casefold?.()??$('search').value.trim().toLowerCase(),matches=G.nodes.filter(n=>n.label.toLowerCase()===q||n.label.toLowerCase().includes(q));if(matches.length)selectNode(matches[0].id);else{$('search-form').classList.add('has-error');$('announce').textContent='No matching label. Try fewer characters or choose a suggestion.'}});
    renderGraph();selectNode(G.defaultNode);
  })();
  </script>
</body>
</html>""".replace("__PAYLOAD__", encoded)


def readme() -> str:
    return """# Chinese Rap Research Atlas V1

Open `index.html` in a modern browser. The page is self-contained and focuses
on interpretable results: characteristic language, written-ending fingerprints,
writing-form profiles, resampling-supported lyrical neighbours, and concise
edge evidence. The default graph shows only edges reselected in at least 50%
of 250 song-level bootstrap replicates. A toggle reveals the remaining
two-representation-stable but lower-repeatability candidates.

The public page contains no lyric lines, song/chunk identifiers, embeddings,
membership rows, or unreviewed NER occurrences.
"""


def method_text() -> str:
    return """# Atlas method boundary

The network topology comes unchanged from the validated duplicate-controlled
BGE-M3 repertoire graph. The profile layer comes from the separately validated
interpretable-profile artifact. Edge colors show the strongest auxiliary signal
that passed its within-support-stratum 90th-percentile gate: shared
characteristic language, written-ending distribution, writing form, or
semantic-only.

Each original edge was re-estimated over 250 within-label song bootstrap
replicates. The public Atlas defaults to the 16 edges with selection frequency
of at least 0.50; the other 70 edges remain available as dashed exploratory
candidates. Bootstrap selection frequency is a repeatability diagnostic, not a
posterior probability.

Auxiliary signals are post-hoc concordant evidence. They are not a causal
decomposition of BGE-M3. A line is not a collaboration, influence, friendship,
biography, hometown, genre, Flow, voice, or beat claim.

NER/reference results are not shown because occurrence-level two-reviewer
context validation has not been completed.
"""


def validate(stage: Path, payload: dict[str, Any]) -> dict[str, Any]:
    page = (stage / "index.html").read_text(encoding="utf-8")
    checks = [
        {"name": "all_graph_nodes_have_profiles", "passed": len(payload["nodes"]) == 204 and all("profile" in item for item in payload["nodes"])},
        {"name": "all_stable_edges_have_explanations", "passed": len(payload["edges"]) == 86 and all("explanation" in item for item in payload["edges"])},
        {"name": "all_edges_have_bootstrap_support", "passed": all("bootstrap" in item and "bootstrapSupported" in item for item in payload["edges"])},
        {"name": "bootstrap_default_matches_validated_count", "passed": sum(bool(item["bootstrapSupported"]) for item in payload["edges"]) == payload["metrics"]["bootstrapSupportedEdges"] == 16},
        {"name": "candidate_toggle_present", "passed": "id=\"candidates\"" in page and "Show ${G.metrics.candidateEdges} exploratory matches" in page},
        {"name": "result_first_sections_present", "passed": all(term in page for term in ("Distinctive words", "Common ending sounds", "Writing habits", "What they share", "song resamples"))},
        {"name": "pca_axes_not_presented", "passed": "PCA axis 1" not in page and "PCA axis 2" not in page},
        {"name": "method_is_collapsed", "passed": "<details class=\"method\">" in page},
        {"name": "no_remote_assets", "passed": "<script src=" not in page and "<link rel=\"stylesheet\"" not in page and "url(http" not in page},
        {"name": "no_unreviewed_reference_items", "passed": all(not item["profile"]["references"]["items"] for item in payload["nodes"])},
        {"name": "default_profile_has_content", "passed": any(item["id"] == payload["defaultNode"] and len(item["profile"]["characteristicTerms"]) >= 3 for item in payload["nodes"])},
        {"name": "default_graph_hides_noncore_nodes", "passed": ".node.hidden{display:none}" in page and "candidateOnly hidden" in page and "unlinked hidden" in page},
        {"name": "vague_result_copy_removed", "passed": all(phrase not in page.lower() for phrase in ("reason not yet clear", "latin character", "deduplicated text"))},
        {"name": "meaningful_result_explanations_present", "passed": all(phrase in page for phrase in ("What they share", "Reappeared in", "Distinctive words", "Reliability & limits", "overall wording match only"))},
    ]
    return {"artifact_id": ARTIFACT_ID, "version": VERSION, "generated_at_utc": utc_now(), "status": "pass" if all(item["passed"] for item in checks) else "fail", "checks": checks}


def build() -> None:
    payload, lineage = make_payload()
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}-", dir=OUT_DIR.parent))
    try:
        atomic_write_text(stage / "index.html", html_document(payload))
        atomic_write_text(stage / "README.md", readme())
        atomic_write_text(stage / "method.md", method_text())
        result = validate(stage, payload)
        if result["status"] != "pass":
            raise RuntimeError(f"Atlas validation failed: {result}")
        atomic_write_json(stage / "validation.json", result)
        files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(stage.iterdir()) if path.is_file()}
        manifest = {"artifact_id": ARTIFACT_ID, "version": VERSION, "generated_at_utc": utc_now(), "classification": "public_self_contained_interpretable_research_atlas", "lineage": lineage, "files": files}
        atomic_write_json(stage / "manifest.json", manifest)
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        os.replace(stage, OUT_DIR)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_existing() -> None:
    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((OUT_DIR / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "pass" or not all(item.get("passed") for item in validation.get("checks", [])):
        raise RuntimeError("Persisted Atlas validation failed.")
    for name, record in manifest.get("files", {}).items():
        path = OUT_DIR / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Atlas payload hash mismatch: {name}")
    print(json.dumps({"artifact": ARTIFACT_ID, "status": "pass", "checks": len(validation["checks"])}, indent=2))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("build", "validate", "all"), default="all")
    args = parser.parse_args()
    if args.command in {"build", "all"}:
        build()
    if args.command in {"validate", "all"}:
        validate_existing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
