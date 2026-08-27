#!/usr/bin/env python3
"""Build the public, aggregate-only data bundle for the Chinese rap result site.

The script joins four already validated public artifacts. It intentionally does
not read lyric text, song/chunk identifiers, embeddings, or private audit rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from pypinyin import Style, lazy_pinyin
except ModuleNotFoundError:  # Integrity-only rebuilds may reuse the frozen map.
    Style = None
    lazy_pinyin = None


ROOT = Path(__file__).resolve().parents[1]
REPERTOIRE_DIR = ROOT / "results/repertoire-network-v1"
REPERTOIRE_GRAPH_DIR = REPERTOIRE_DIR / "graph"
REPERTOIRE_PROFILE_DIR = REPERTOIRE_DIR / "profiles"
REPERTOIRE_BOOTSTRAP_DIR = REPERTOIRE_DIR / "bootstrap"
REPERTOIRE_ROBUSTNESS_DIR = REPERTOIRE_DIR / "robustness"
RETRIEVAL_DIR = ROOT / "results/retrieval-v1"
RETRIEVAL_INDUCTIVE_DIR = ROOT / "results/retrieval-inductive-sensitivity-v1"
NER_DIR = ROOT / "results/ner-v1"
NER_CO_MENTION_PATH = NER_DIR / "entity_co_mentions_provisional.csv"
NER_RELEASED_CLAIM_AUDIT_PATH = NER_DIR / "released_claim_audit_status.json"
RHYME_DIR = ROOT / "results/written-rhyme-v1"
SITE_DATA_DIR = ROOT / "site/app/data"
PUBLIC_DATA_DIR = ROOT / "site/public/data"


FAMILY_ORDER = [
    "A", "O", "E", "IE_VE", "AI", "EI", "AO", "OU", "AN",
    "EN", "ANG", "ENG", "ONG", "I", "U", "V", "ER",
]
FAMILY_FINALS = {
    "A": {"a", "ia", "ua"}, "O": {"o", "uo"}, "E": {"e"},
    "IE_VE": {"ie", "ve"}, "AI": {"ai", "uai"},
    "EI": {"ei", "ui"}, "AO": {"ao", "iao"}, "OU": {"ou", "iu"},
    "AN": {"an", "ian", "uan", "van"},
    "EN": {"en", "in", "un", "vn"},
    "ANG": {"ang", "iang", "uang"},
    "ENG": {"eng", "ing", "ueng"}, "ONG": {"ong", "iong"},
    "I": {"i"}, "U": {"u"}, "V": {"v"}, "ER": {"er"},
}
FINAL_TO_FAMILY = {ending: family for family, endings in FAMILY_FINALS.items() for ending in endings}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_passing_validation(path: Path) -> dict[str, Any]:
    validation = read_json(path)
    if str(validation.get("status", "")).lower() not in {"pass", "passed"}:
        raise RuntimeError(f"Validation is not passing: {path}")
    failed_checks = [
        check.get("name", "unnamed")
        for check in validation.get("checks", [])
        if not check.get("passed", False)
    ]
    if failed_checks:
        raise RuntimeError(f"Validation contains failed checks in {path}: {failed_checks}")
    return validation


def verify_component_payloads(directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "manifest.json")
    files = manifest.get("files", manifest.get("output_files", {}))
    if not files:
        raise RuntimeError(f"No public payload inventory in {directory / 'manifest.json'}")
    for name, expected in files.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != integer(expected.get("bytes"), -1):
            raise RuntimeError(
                f"Payload byte-size mismatch for {path}: {actual_bytes} != {expected.get('bytes')}"
            )
        if actual_sha256 != str(expected.get("sha256", "")).lower():
            raise RuntimeError(f"Payload hash mismatch for {path}")
    return manifest


def unique_by(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise RuntimeError(f"Duplicate {label} key: {value}")
        result[value] = row
    return result


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_repertoire_source() -> dict[str, Any]:
    """Reconstruct the public repertoire graph from frozen aggregate artifacts."""
    root_manifest = read_json(REPERTOIRE_DIR / "manifest.json")
    root_validation_path = REPERTOIRE_DIR / root_manifest["validation"]["file"]
    require_passing_validation(root_validation_path)
    if root_validation_path.stat().st_size != root_manifest["validation"]["bytes"]:
        raise RuntimeError("Frozen repertoire-network validation byte-size mismatch")
    if sha256_file(root_validation_path) != root_manifest["validation"]["sha256"]:
        raise RuntimeError("Frozen repertoire-network validation hash mismatch")
    component_dirs = {
        "graph": REPERTOIRE_GRAPH_DIR,
        "profiles": REPERTOIRE_PROFILE_DIR,
        "bootstrap": REPERTOIRE_BOOTSTRAP_DIR,
        "robustness": REPERTOIRE_ROBUSTNESS_DIR,
    }
    component_manifests: dict[str, dict[str, Any]] = {}
    for name, directory in component_dirs.items():
        component_manifests[name] = verify_component_payloads(directory)
        require_passing_validation(directory / "validation.json")
        expected = root_manifest["components"][name]
        if component_manifests[name].get("artifact_id") != expected["artifact_id"]:
            raise RuntimeError(f"Unexpected {name} component artifact")
        if sha256_file(directory / "manifest.json") != expected["manifest_sha256"]:
            raise RuntimeError(f"Frozen {name} manifest hash mismatch")
        if sha256_file(directory / "validation.json") != expected["validation_sha256"]:
            raise RuntimeError(f"Frozen {name} validation hash mismatch")

    graph_summary = read_json(REPERTOIRE_GRAPH_DIR / "analysis_summary.json")
    bootstrap_summary = read_json(REPERTOIRE_BOOTSTRAP_DIR / "analysis_summary.json")
    robustness_summary = read_json(REPERTOIRE_ROBUSTNESS_DIR / "analysis_summary.json")
    expected_counts = root_manifest["expected_counts"]

    eligible_node_rows = [
        row
        for row in read_csv(REPERTOIRE_GRAPH_DIR / "artist_repertoire_nodes.csv")
        if row.get("graph_node_eligible", "").lower() == "true"
    ]
    node_by_id = unique_by(eligible_node_rows, "artist_label_id", "eligible graph node")
    if len(node_by_id) != expected_counts["eligible_labels"]:
        raise RuntimeError(f"Expected 204 eligible graph nodes, found {len(node_by_id)}")
    if len({row["source_artist_label"] for row in eligible_node_rows}) != len(node_by_id):
        raise RuntimeError("Eligible graph source labels are not unique")

    layout_rows = read_csv(REPERTOIRE_GRAPH_DIR / "artist_repertoire_layout.csv")
    layout_by_id = unique_by(layout_rows, "artist_label_id", "layout")
    if set(layout_by_id) != set(node_by_id):
        raise RuntimeError("The PCA layout does not join one-to-one to eligible graph nodes")

    profile_rows = read_json(REPERTOIRE_PROFILE_DIR / "source_label_profiles.json")
    if not isinstance(profile_rows, list):
        raise RuntimeError("source_label_profiles.json must contain a list")
    profile_by_id = unique_by(profile_rows, "id", "profile")
    if set(profile_by_id) != set(node_by_id):
        raise RuntimeError("Profiles do not join one-to-one to eligible graph nodes")
    for artist_label_id, node in node_by_id.items():
        if profile_by_id[artist_label_id]["label"] != node["source_artist_label"]:
            raise RuntimeError(f"Profile/source-label mismatch for {artist_label_id}")

    graph_edges = read_csv(REPERTOIRE_GRAPH_DIR / "artist_repertoire_edges.csv")
    if len(graph_edges) != expected_counts["released_edges"]:
        raise RuntimeError(f"Expected 86 retained graph edges, found {len(graph_edges)}")
    graph_edge_by_key: dict[str, dict[str, str]] = {}
    for row in graph_edges:
        a = row["artist_label_id_a"]
        b = row["artist_label_id_b"]
        key = pair_key(a, b)
        if key in graph_edge_by_key:
            raise RuntimeError(f"Duplicate retained graph edge: {key}")
        if a not in node_by_id or b not in node_by_id:
            raise RuntimeError(f"Retained graph edge has an ineligible endpoint: {key}")
        if row.get("stable_across_shared_text_exclusion", "").lower() != "true":
            raise RuntimeError(f"Retained edge fails the two-representation rule: {key}")
        if row["source_artist_label_a"] != node_by_id[a]["source_artist_label"]:
            raise RuntimeError(f"Edge/source-label mismatch for {a}")
        if row["source_artist_label_b"] != node_by_id[b]["source_artist_label"]:
            raise RuntimeError(f"Edge/source-label mismatch for {b}")
        graph_edge_by_key[key] = row

    explanation_rows = read_json(REPERTOIRE_PROFILE_DIR / "stable_link_explanations.json")
    if not isinstance(explanation_rows, list):
        raise RuntimeError("stable_link_explanations.json must contain a list")
    explanation_by_key = unique_by(explanation_rows, "key", "edge explanation")
    bootstrap_rows = read_csv(REPERTOIRE_BOOTSTRAP_DIR / "stable_edge_bootstrap.csv")
    bootstrap_by_key = unique_by(bootstrap_rows, "edge_key", "edge bootstrap")
    graph_keys = set(graph_edge_by_key)
    if set(explanation_by_key) != graph_keys:
        raise RuntimeError("Edge explanations do not join one-to-one to retained graph edges")
    if set(bootstrap_by_key) != graph_keys:
        raise RuntimeError("Bootstrap rows do not join one-to-one to retained graph edges")

    nodes = []
    for artist_label_id, row in node_by_id.items():
        layout = layout_by_id[artist_label_id]
        profile = profile_by_id[artist_label_id]
        nodes.append(
            {
                "id": artist_label_id,
                "label": row["source_artist_label"],
                "x": number(layout["x"]),
                "y": number(layout["y"]),
                "independentSongs": integer(row["independent_clean_song_count"]),
                "profile": profile,
            }
        )

    edges = []
    for key, graph_edge in graph_edge_by_key.items():
        explanation = explanation_by_key[key]
        bootstrap = bootstrap_by_key[key]
        probability = number(bootstrap["two_representation_edge_probability"])
        edges.append(
            {
                "a": graph_edge["artist_label_id_a"],
                "b": graph_edge["artist_label_id_b"],
                "bootstrap": {"probability": probability},
                "bootstrapSupported": probability >= 0.50,
                "explanation": explanation,
            }
        )

    connected_ids = {endpoint for edge in edges for endpoint in (edge["a"], edge["b"])}
    repeatable_edges = sum(edge["bootstrapSupported"] for edge in edges)
    if len(connected_ids) != expected_counts["connected_labels"]:
        raise RuntimeError(f"Expected 93 connected graph labels, found {len(connected_ids)}")
    if repeatable_edges != expected_counts["edges_repeatability_at_least_0_50"]:
        raise RuntimeError(f"Expected 16 repeatable edges, found {repeatable_edges}")
    if bootstrap_summary["counts"]["replicates"] != expected_counts["bootstrap_replicates"]:
        raise RuntimeError("Unexpected bootstrap replicate count")
    pca_variance = graph_summary["model"]["spatial_projection_variance_explained_2d"]
    if abs(pca_variance - root_manifest["projection_variance_explained_2d"]) > 1e-12:
        raise RuntimeError("Unexpected two-dimensional PCA variance")

    return {
        "artifact": root_manifest["artifact_id"],
        "componentArtifacts": {
            name: manifest["artifact_id"] for name, manifest in component_manifests.items()
        },
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "representation": "BGE-M3 lyric-chunk centroids under primary duplicate control and shared-text-exclusion sensitivity",
            "eligibleLabels": len(nodes),
            "connectedLabels": len(connected_ids),
            "retainedEdges": len(edges),
            "repeatableEdges": repeatable_edges,
            "bootstrapReplicates": bootstrap_summary["counts"]["replicates"],
            "repeatabilityGate": 0.50,
            "pcaVariance2d": pca_variance,
            "alignmentNull": robustness_summary["graph_alignment_null"],
            "projectionFidelity": robustness_summary["projection_fidelity"],
            "edgeRule": "Both source-label profiles rank each other among their five closest matches in both duplicate-controlled text representations.",
            "layoutMeaning": "The global position is an approximate two-dimensional PCA summary of BGE-M3 repertoire profiles; only a released line represents the stricter reciprocal match rule.",
            "claimBoundary": root_manifest["claim_boundary"],
        },
    }


def compact_retrieval() -> dict[str, Any]:
    summary = read_json(RETRIEVAL_DIR / "analysis_summary.json")
    inductive = read_json(RETRIEVAL_INDUCTIVE_DIR / "analysis_summary.json")
    systems = []
    display_names = {
        "BGE-M3 dense (strict)": "BGE-M3",
        "character 2-5 gram TF-IDF (strict)": "Character TF-IDF",
        "equal-weight z-score fusion (strict)": "Fusion",
    }
    for source_name, values in summary["primary_macro_metrics"].items():
        systems.append(
            {
                "name": display_names[source_name],
                "mrr": values["mrr"],
                "recall1": values["recall_at_1"],
                "recall5": values["recall_at_5"],
                "recall10": values["recall_at_10"],
                "ndcg10": values["ndcg_at_10"],
            }
        )
    inductive_metrics = inductive["macro_metrics"]
    inductive_deltas = inductive["paired_deltas"]
    tfidf_exposure = inductive_deltas[
        "TF-IDF matched transductive minus inductive train-only"
    ]["mrr"]
    fusion_exposure = inductive_deltas[
        "fusion matched transductive minus inductive train-only"
    ]["mrr"]
    inductive_fusion_gain = inductive_deltas[
        "inductive fusion minus inductive TF-IDF"
    ]["mrr"]
    return {
        "systems": systems,
        "queries": summary["population"]["length_qualified_song_queries"],
        "labels": summary["population"]["eligible_source_credit_labels"],
        "groups": summary["population"]["global_strict_duplicate_components"],
        "fusionMrrCi": summary["headline"]["fusion_macro_mrr_ci"],
        "inductiveSanity": {
            "folds": inductive["population"]["folds"],
            "queries": inductive["population"]["queries"],
            "inductiveTfidfMrr": inductive_metrics["TF-IDF inductive train-only"]["mrr"]["estimate"],
            "inductiveFusionMrr": inductive_metrics["fusion inductive train-only"]["mrr"]["estimate"],
            "inductiveFusionGainMrr": inductive_fusion_gain,
            "tfidfExposureEffectMrr": tfidf_exposure,
            "fusionExposureEffectMrr": fusion_exposure,
            "interpretation": inductive["interpretation"]["fusion_robustness"],
            "claimBoundary": inductive["claim_boundary"],
        },
        "claimBoundary": summary["claim_boundary"],
    }


def compact_rhyme(label_id_by_name: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = read_json(RHYME_DIR / "analysis_summary.json")
    fingerprints = read_json(RHYME_DIR / "label_written_rhyme_fingerprints.json")["profiles"]
    lookup = read_json(RHYME_DIR / "recommender_lookup.json")
    label_profiles: dict[str, Any] = {}
    family_label_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in fingerprints:
        site_id = label_id_by_name.get(profile["source_artist_label"])
        if not site_id:
            continue
        written_line_occurrences = integer(profile.get("eligible_written_line_occurrence_count"))
        label_profiles[site_id] = {
            "dominantFamily": profile["top_written_rhyme_families"][0]["value"],
            "dominantShare": profile["top_written_rhyme_families"][0]["share"],
            "topFamilies": profile["top_written_rhyme_families"][:5],
            "distinctiveFamilies": profile["distinctive_written_rhyme_families_vs_corpus"][:4],
            "adjacentSameFamilyRate": profile["song_normalised_adjacent_same_family_rate"],
            "echoLift": profile["local_echo_lift_over_iid_frequency_expectation"],
            "medianRun": profile["median_same_family_run_length"],
            "lines": written_line_occurrences,
            "songs": profile["eligible_song_count"],
        }
        top_family = {
            item["value"]: item for item in profile["top_written_rhyme_families"]
        }
        for item in profile["distinctive_written_rhyme_families_vs_corpus"]:
            family = item["family"]
            observed = top_family.get(family)
            if (
                observed
                and integer(observed.get("count")) >= 15
                and integer(profile.get("eligible_song_count")) >= 5
                and written_line_occurrences >= 100
            ):
                family_label_candidates[family].append(
                    {
                        "labelId": site_id,
                        "count": integer(observed.get("count")),
                        "share": number(observed.get("share")),
                        "log2RateRatio": number(item.get("log2_rate_ratio_vs_corpus")),
                        "songs": integer(profile.get("eligible_song_count")),
                        "lines": written_line_occurrences,
                    }
                )

    labels_by_family = {
        family: sorted(
            family_label_candidates.get(family, []),
            key=lambda row: (row["log2RateRatio"], row["count"], row["songs"]),
            reverse=True,
        )[:8]
        for family in FAMILY_ORDER
    }

    contexts = []
    for row in lookup["ml_common_observed_contexts"]:
        site_id = label_id_by_name.get(row["source_artist_label"])
        if not site_id:
            continue
        contexts.append(
            {
                "labelId": site_id,
                "previous2": row["previous_2_written_rhyme_family"],
                "previous1": row["previous_1_written_rhyme_family"],
                "run": row["current_same_family_run_bucket"],
                "position": row["written_line_position_bucket"],
                "support": row["training_event_support"],
                "top5": row["top_5"],
            }
        )

    metrics = [
        row for row in summary["primary_test_metrics"]
        if row["evaluation_split"] == "song_held_out_test"
    ]
    stratified_path = RHYME_DIR / "stratified_metrics.csv"
    stratified = read_csv(stratified_path)
    switch_rows = [
        row for row in stratified
        if row.get("model") == "hierarchical_sgd_context"
        and row.get("stratum_dimension") == "transition_type"
    ]
    rhyme = {
        "classes": lookup["classes"],
        "familyOrder": FAMILY_ORDER,
        "globalTop5": lookup["global_frequency_top_5"],
        "markov": lookup["markov_by_previous_family"],
        "contexts": contexts,
        "labelsByFamily": labels_by_family,
        "releasedModel": lookup["released_model"],
        "sourceLabelConditioning": lookup["source_credit_label_conditioning"],
        "abstention": summary["abstention"],
        "metrics": metrics,
        "pairedDeltas": summary["paired_released_model_differences"],
        "switchDiagnostic": switch_rows,
        "testEvents": summary["model_selection"]["leakage_safe_test_event_count"],
        "strictCandidateCoverage": next(
            row["end_to_end_event_coverage"]
            for row in metrics
            if row["model"] == "hierarchical_sgd_context"
        ),
        "testSongs": next(row["song_count"] for row in metrics if row["model"] == "hierarchical_sgd_context"),
        "claimBoundary": summary["claim_boundary"],
    }
    return rhyme, label_profiles


def compact_ner(label_id_by_name: dict[str, str]) -> dict[str, Any]:
    summary = read_json(NER_DIR / "summary.json")
    released_claim_audit = read_json(NER_RELEASED_CLAIM_AUDIT_PATH)
    ner_manifest = read_json(NER_DIR / "manifest.json")
    primary_co_mention_manifest = ner_manifest["files"][NER_CO_MENTION_PATH.name]
    if NER_CO_MENTION_PATH.stat().st_size != integer(primary_co_mention_manifest["bytes"]):
        raise RuntimeError("Primary NER co-mention payload byte-size mismatch")
    if sha256_file(NER_CO_MENTION_PATH) != primary_co_mention_manifest["sha256"]:
        raise RuntimeError("Primary NER co-mention payload hash mismatch")
    if summary.get("version") != "1.1.0" or ner_manifest.get("version") != "1.1.0":
        raise RuntimeError("The corrected v1.1 NER release is required for co-mentions")
    entities = read_csv(NER_DIR / "entity_aggregate_provisional.csv")
    links = read_csv(NER_DIR / "source_label_entity_links_provisional.csv")
    entity_rows = sorted(entities, key=lambda row: (row["entity_type"], row["entity"]))
    entity_id = {
        (row["entity"], row["entity_type"]): f"e{index:03d}"
        for index, row in enumerate(entity_rows, start=1)
    }
    compact_entities = [
        {
            "id": entity_id[(row["entity"], row["entity_type"])],
            "name": row["entity"],
            "type": row["entity_type"],
            "songs": integer(row.get("unique_song_units")),
            "labels": integer(row.get("source_credit_labels")),
            "agreementRate": number(row.get("strict_agreement_rate")),
            "status": row.get("status", "PROVISIONAL"),
        }
        for row in entity_rows
    ]
    compact_links = []
    for row in links:
        site_label_id = label_id_by_name.get(row["source_credit_label"])
        site_entity_id = entity_id.get((row["entity"], row["entity_type"]))
        if not site_label_id or not site_entity_id:
            continue
        compact_links.append(
            {
                "labelId": site_label_id,
                "entityId": site_entity_id,
                "entityType": row["entity_type"],
                "songs": integer(row.get("entity_song_units_within_label")),
                "labelSongs": integer(row.get("label_song_units")),
                "share": number(row.get("within_label_share")),
                "lift": number(row.get("shrunken_risk_ratio")),
                "liftLow": number(row.get("shrunken_risk_ratio_ci95_low_conservative")),
                "liftHigh": number(row.get("shrunken_risk_ratio_ci95_high_conservative")),
                "qValue": number(row.get("q_value_bh")),
                "agreementOccurrences": 0,
                "reliability": row.get("reliability_class", "SUPPORTED"),
                "plainMeaning": row.get("plain_meaning", ""),
                "status": row.get("status", "PROVISIONAL"),
            }
        )
    compact_links.sort(key=lambda row: (-row["songs"], -row["lift"], row["labelId"]))

    # The primary v1.1 co-mention release uses all 5,681 eligible song units as
    # its denominator. Never substitute the legacy entity-bearing-denominator
    # sensitivity table, which produces a different and non-primary result.
    if NER_CO_MENTION_PATH.name != "entity_co_mentions_provisional.csv":
        raise RuntimeError("NER co-mentions must come from the corrected primary release")
    co_mention_rows = read_csv(NER_CO_MENTION_PATH)
    expected_co_mentions = integer(summary["counts"]["public_entity_co_mentions"])
    if expected_co_mentions != 4 or len(co_mention_rows) != expected_co_mentions:
        raise RuntimeError(
            f"Expected exactly 4 corrected primary NER co-mentions, found {len(co_mention_rows)}"
        )
    if integer(summary["graph_analysis"]["released_co_mentions"]) != expected_co_mentions:
        raise RuntimeError("NER co-mention release count disagrees with the analysis summary")
    all_song_denominator = integer(summary["graph_analysis"]["eligible_global_song_units"])
    if all_song_denominator != 5681:
        raise RuntimeError(f"Unexpected primary NER co-mention denominator: {all_song_denominator}")

    sensitivity_rows = read_csv(NER_DIR / "release_sensitivity_summary.csv")
    primary_stage = [
        row
        for row in sensitivity_rows
        if row.get("stage") == "v1_1_primary_all_song_denominator_uncertainty_fdr_release"
    ]
    if len(primary_stage) != 1 or integer(primary_stage[0].get("co_mentions")) != 4:
        raise RuntimeError("Corrected v1.1 primary co-mention release is not uniquely identified")
    legacy_stage_counts = {
        integer(row.get("co_mentions"))
        for row in sensitivity_rows
        if "legacy" in row.get("stage", "")
    }
    if legacy_stage_counts != {1, 9}:
        raise RuntimeError(f"Unexpected legacy co-mention sensitivity counts: {legacy_stage_counts}")

    released_entity_pairs = {(row["entity"], row["entity_type"]) for row in entities}
    compact_co_mentions = []
    seen_co_mention_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for row in co_mention_rows:
        endpoint_a = (row["entity_a"], row["entity_a_type"])
        endpoint_b = (row["entity_b"], row["entity_b_type"])
        key = tuple(sorted((endpoint_a, endpoint_b)))
        if key in seen_co_mention_pairs:
            raise RuntimeError(f"Duplicate primary NER co-mention pair: {key}")
        seen_co_mention_pairs.add(key)
        if endpoint_a not in released_entity_pairs or endpoint_b not in released_entity_pairs:
            raise RuntimeError(f"NER co-mention targets an unreleased entity: {key}")
        if integer(row.get("all_eligible_song_units")) != all_song_denominator:
            raise RuntimeError(f"NER co-mention does not use the all-song denominator: {key}")
        if row.get("release_gate_pass", "").lower() != "true":
            raise RuntimeError(f"Unreleased NER co-mention entered the primary table: {key}")
        if number(row.get("npmi")) <= 0 or number(row.get("q_value_bh"), 1.0) > 0.05:
            raise RuntimeError(f"NER co-mention fails the corrected NPMI/FDR release gate: {key}")
        relation_boundary = row.get("relation_scope", "")
        if "all eligible song units as denominator" not in relation_boundary.lower():
            raise RuntimeError(f"NER co-mention has a legacy or missing relation boundary: {key}")
        compact_co_mentions.append(
            {
                "a": row["entity_a"],
                "aType": row["entity_a_type"],
                "b": row["entity_b"],
                "bType": row["entity_b_type"],
                "allEligibleSongUnits": all_song_denominator,
                "songUnits": integer(row.get("unique_song_unit_co_mentions")),
                "labels": integer(row.get("source_credit_labels")),
                "lift": number(row.get("lift")),
                "npmi": number(row.get("npmi")),
                "qValue": number(row.get("q_value_bh")),
                "reliability": row.get("reliability_class", "SUPPORTED"),
                "relationBoundary": relation_boundary,
                "status": row.get("status", "PROVISIONAL"),
            }
        )
    return {
        "entities": compact_entities,
        "links": compact_links,
        "coMentions": compact_co_mentions,
        "status": summary["status"],
        "humanGoldAvailable": summary["human_gold_available"],
        "releasedClaimAudit": {
            "status": released_claim_audit["status"],
            "releasedClaims": released_claim_audit["scope"]["released_claims_total"],
            "uniqueSupportingOccurrences": released_claim_audit["scope"]["unique_contributing_occurrence_rows"],
            "coverage": released_claim_audit["validation"]["released_claim_occurrence_coverage"],
            "occurrenceDecisionsCompleted": released_claim_audit["review_progress"]["occurrence_tasks_adjudicated"],
            "coMentionDecisionsCompleted": released_claim_audit["review_progress"]["co_mention_tasks_adjudicated"],
            "metrics": released_claim_audit["targeted_audit_metrics"],
            "evidenceBoundary": released_claim_audit["evidence_boundary"],
            "nextAction": released_claim_audit["next_action"],
        },
        "counts": summary["counts"],
        "claimBoundary": summary["claim_boundary"],
    }


def pinyin_family(character: str) -> str | None:
    if lazy_pinyin is None or Style is None:
        raise RuntimeError("pypinyin is required to recompute the character-to-family map")
    token = lazy_pinyin(
        character,
        style=Style.FINALS,
        strict=True,
        errors=lambda item: [""] * len(item),
    )[0]
    token = token.lower().replace("ü", "v")
    token = {"uei": "ui", "iou": "iu", "uen": "un"}.get(token, token)
    return FINAL_TO_FAMILY.get(token)


def build_character_map() -> dict[str, str]:
    if lazy_pinyin is None:
        frozen_path = SITE_DATA_DIR / "characterToRhymeFamily.json"
        if not frozen_path.is_file():
            raise RuntimeError("pypinyin is unavailable and no frozen character map can be reused")
        frozen = read_json(frozen_path)
        if not isinstance(frozen, dict) or len(frozen) < 20_000:
            raise RuntimeError("Frozen character-to-family map is invalid")
        return {str(key): str(value) for key, value in frozen.items()}
    mapping: dict[str, str] = {}
    for codepoint in range(0x4E00, 0xA000):
        character = chr(codepoint)
        family = pinyin_family(character)
        if family:
            mapping[character] = family
    return mapping


def main() -> None:
    for path in (
        REPERTOIRE_DIR / "manifest.json",
        RETRIEVAL_DIR / "validation.json",
        RETRIEVAL_INDUCTIVE_DIR / "validation.json",
        NER_DIR / "validation.json",
        NER_DIR / "reconciliation_validation.json",
        NER_CO_MENTION_PATH,
        NER_RELEASED_CLAIM_AUDIT_PATH,
        RHYME_DIR / "validation.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    validations = {
        "retrieval": read_json(RETRIEVAL_DIR / "validation.json"),
        "retrievalInductive": read_json(RETRIEVAL_INDUCTIVE_DIR / "validation.json"),
        "ner": read_json(NER_DIR / "validation.json"),
        "nerReconciliation": read_json(NER_DIR / "reconciliation_validation.json"),
        "rhyme": read_json(RHYME_DIR / "validation.json"),
    }
    if any(str(value.get("status", "")).lower() not in {"pass", "passed"} for value in validations.values()):
        raise RuntimeError("All public downstream artifacts must pass validation before site data can be built")
    released_claim_audit = read_json(NER_RELEASED_CLAIM_AUDIT_PATH)
    if released_claim_audit.get("validation", {}).get("package_generation") != "pass":
        raise RuntimeError("The released-claim NER audit package must pass generation validation")
    if released_claim_audit.get("status") != "PENDING_DUAL_HUMAN_REVIEW_AND_ADJUDICATION":
        raise RuntimeError("The public site must not imply that the released-claim NER audit is complete")

    repertoire = load_repertoire_source()
    old_nodes = sorted(repertoire["nodes"], key=lambda row: row["label"].casefold())
    old_node_by_name = {row["label"]: row for row in old_nodes}
    ner_link_rows = read_csv(NER_DIR / "source_label_entity_links_provisional.csv")
    rhyme_rows = read_json(RHYME_DIR / "label_written_rhyme_fingerprints.json")["profiles"]
    all_label_names = sorted(
        set(old_node_by_name)
        | {row["source_credit_label"] for row in ner_link_rows}
        | {row["source_artist_label"] for row in rhyme_rows},
        key=str.casefold,
    )
    if set(all_label_names) != set(old_node_by_name):
        extra = sorted(set(all_label_names) - set(old_node_by_name), key=str.casefold)
        missing = sorted(set(old_node_by_name) - set(all_label_names), key=str.casefold)
        raise RuntimeError(
            f"Downstream/source-label population mismatch; extra={extra}, missing={missing}"
        )
    label_id_by_name = {
        name: f"l{index:03d}" for index, name in enumerate(all_label_names, start=1)
    }
    old_to_site = {
        row["id"]: label_id_by_name[row["label"]] for row in old_nodes
    }

    rhyme, rhyme_profiles = compact_rhyme(label_id_by_name)
    ner_song_support: dict[str, int] = defaultdict(int)
    for row in ner_link_rows:
        ner_song_support[row["source_credit_label"]] = max(
            ner_song_support[row["source_credit_label"]],
            integer(row.get("label_song_units")),
        )
    nodes = []
    for label_name in all_label_names:
        row = old_node_by_name.get(label_name)
        site_id = label_id_by_name[label_name]
        if row is None:
            nodes.append(
                {
                    "id": site_id,
                    "label": label_name,
                    "x": 0.0,
                    "y": 0.0,
                    "independentSongs": ner_song_support.get(label_name, 0),
                    "terms": [],
                    "traits": [],
                    "rhyme": rhyme_profiles.get(site_id),
                }
            )
            continue
        profile = row["profile"]
        nodes.append(
            {
                "id": site_id,
                "label": row["label"],
                "x": row["x"],
                "y": row["y"],
                "independentSongs": row["independentSongs"],
                "terms": profile["characteristicTerms"][:8],
                "traits": [
                    {
                        "key": item["key"],
                        "percentile": item["percentile"],
                        "raw": item["raw"],
                    }
                    for item in profile["formTraits"]
                ],
                "rhyme": rhyme_profiles.get(site_id),
            }
        )

    edges = []
    for row in repertoire["edges"]:
        reasons = [
            {
                "kind": signal["kind"],
                "label": signal["label"],
                "items": signal["items"],
                "percentile": signal["percentile"],
            }
            for signal in row["explanation"]["signals"]
        ]
        edges.append(
            {
                "a": old_to_site[row["a"]],
                "b": old_to_site[row["b"]],
                "repeatability": row["bootstrap"]["probability"],
                "status": "repeatable" if row["bootstrapSupported"] else "exploratory",
                "dominantSignal": row["explanation"]["dominantSignal"],
                "reasons": reasons,
            }
        )

    ner = compact_ner(label_id_by_name)
    if len(ner["links"]) != len(ner_link_rows):
        raise RuntimeError(
            f"Site join lost NER links: retained {len(ner['links'])} of {len(ner_link_rows)}"
        )
    payload = {
        "artifact": "chinese-rap-results-site-data-v1",
        "question": "How do Chinese rap lyrics form recognizable lyrical identities through language, cultural reference, and dictionary-estimated written rhyme?",
        "constructDefinition": "Here, lyrical identity means a corpus-relative writing profile attached to a source-credit label—not a verified person or an authorship claim.",
        "labels": nodes,
        "lyricalEdges": edges,
        "repertoireGraph": repertoire["metadata"],
        "retrieval": compact_retrieval(),
        "ner": ner,
        "rhyme": rhyme,
        "lineage": {
            "repertoireNetwork": repertoire["artifact"],
            "repertoireGraph": repertoire["componentArtifacts"]["graph"],
            "repertoireProfiles": repertoire["componentArtifacts"]["profiles"],
            "repertoireBootstrap": repertoire["componentArtifacts"]["bootstrap"],
            "repertoireRobustness": repertoire["componentArtifacts"]["robustness"],
            "retrieval": validations["retrieval"]["artifact_id"],
            "retrievalInductive": validations["retrievalInductive"]["artifact_id"],
            "ner": validations["ner"]["artifact_id"],
            "nerReleasedClaimAudit": released_claim_audit["artifact_id"],
            "nerReconciliation": validations["nerReconciliation"]["validation_mode"],
            "rhyme": validations["rhyme"]["artifact_id"],
        },
        "publicBoundary": "Aggregate evidence only; no lyric lines, song/chunk identifiers, embeddings, or verified-person claims.",
    }
    character_map = build_character_map()

    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    characters_encoded = json.dumps(character_map, ensure_ascii=False, separators=(",", ":"))
    (SITE_DATA_DIR / "researchData.json").write_text(encoded, encoding="utf-8", newline="\n")
    (SITE_DATA_DIR / "characterToRhymeFamily.json").write_text(characters_encoded, encoding="utf-8", newline="\n")
    (PUBLIC_DATA_DIR / "researchData.json").write_text(encoded, encoding="utf-8", newline="\n")

    manifest = {
        "artifact": "chinese-rap-results-site-data-v1",
        "labels": len(nodes),
        "lyricalEdges": len(edges),
        "connectedLyricalLabels": repertoire["metadata"]["connectedLabels"],
        "repeatableLyricalEdges": repertoire["metadata"]["repeatableEdges"],
        "repertoireBootstrapReplicates": repertoire["metadata"]["bootstrapReplicates"],
        "repertoireNullReplicates": repertoire["metadata"]["alignmentNull"]["null_replicates"],
        "retrievalInductiveFolds": payload["retrieval"]["inductiveSanity"]["folds"],
        "retrievalInductiveQueries": payload["retrieval"]["inductiveSanity"]["queries"],
        "nerEntities": len(payload["ner"]["entities"]),
        "nerLinks": len(payload["ner"]["links"]),
        "nerCoMentions": len(payload["ner"]["coMentions"]),
        "nerReleasedClaimAuditOccurrences": payload["ner"]["releasedClaimAudit"]["uniqueSupportingOccurrences"],
        "nerReleasedClaimAuditStatus": payload["ner"]["releasedClaimAudit"]["status"],
        "rhymeContexts": len(rhyme["contexts"]),
        "dictionaryCharacters": len(character_map),
        "researchDataSha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "status": "pass",
    }
    (PUBLIC_DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
