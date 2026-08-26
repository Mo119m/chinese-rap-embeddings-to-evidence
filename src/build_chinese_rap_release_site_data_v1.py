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

from pypinyin import Style, lazy_pinyin


ROOT = Path(__file__).resolve().parents[1]
REPERTOIRE_DATA = ROOT / "site/app/data/researchData.json"
RETRIEVAL_DIR = ROOT / "results/retrieval-v1"
NER_DIR = ROOT / "results/ner-v1"
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
    """Translate the released site data into the compact repertoire source schema.

    The first private build used an earlier exploratory HTML artifact as this
    source. The publication repository instead carries the exact aggregate
    repertoire coordinates, traits, terms, and edges needed for a deterministic
    rebuild, so no removed Atlas file or private corpus path is required here.
    """
    payload = read_json(REPERTOIRE_DATA)
    nodes = [
        {
            "id": row["id"],
            "label": row["label"],
            "x": row["x"],
            "y": row["y"],
            "independentSongs": row["independentSongs"],
            "profile": {
                "characteristicTerms": row.get("terms", []),
                "formTraits": row.get("traits", []),
            },
        }
        for row in payload["labels"]
    ]
    edges = [
        {
            "a": row["a"],
            "b": row["b"],
            "bootstrap": {"probability": row["repeatability"]},
            "bootstrapSupported": row["status"] == "repeatable",
            "explanation": {
                "dominantSignal": row["dominantSignal"],
                "signals": row.get("reasons", []),
            },
        }
        for row in payload["lyricalEdges"]
    ]
    return {
        "artifact": payload.get("lineage", {}).get("atlas", payload["artifact"]),
        "nodes": nodes,
        "edges": edges,
    }


def compact_retrieval() -> dict[str, Any]:
    summary = read_json(RETRIEVAL_DIR / "analysis_summary.json")
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
    return {
        "systems": systems,
        "queries": summary["population"]["length_qualified_song_queries"],
        "labels": summary["population"]["eligible_source_credit_labels"],
        "groups": summary["population"]["global_strict_duplicate_components"],
        "fusionMrrCi": summary["headline"]["fusion_macro_mrr_ci"],
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
    return {
        "entities": compact_entities,
        "links": compact_links,
        "status": summary["status"],
        "humanGoldAvailable": summary["human_gold_available"],
        "counts": summary["counts"],
        "claimBoundary": summary["claim_boundary"],
    }


def pinyin_family(character: str) -> str | None:
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
    mapping: dict[str, str] = {}
    for codepoint in range(0x4E00, 0xA000):
        character = chr(codepoint)
        family = pinyin_family(character)
        if family:
            mapping[character] = family
    return mapping


def main() -> None:
    for path in (
        REPERTOIRE_DATA,
        RETRIEVAL_DIR / "validation.json",
        NER_DIR / "validation.json",
        RHYME_DIR / "validation.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    validations = {
        "retrieval": read_json(RETRIEVAL_DIR / "validation.json"),
        "ner": read_json(NER_DIR / "validation.json"),
        "rhyme": read_json(RHYME_DIR / "validation.json"),
    }
    if any(str(value.get("status", "")).lower() not in {"pass", "passed"} for value in validations.values()):
        raise RuntimeError("All public downstream artifacts must pass validation before site data can be built")

    atlas = load_repertoire_source()
    old_nodes = sorted(atlas["nodes"], key=lambda row: row["label"].casefold())
    old_node_by_name = {row["label"]: row for row in old_nodes}
    ner_link_rows = read_csv(NER_DIR / "source_label_entity_links_provisional.csv")
    rhyme_rows = read_json(RHYME_DIR / "label_written_rhyme_fingerprints.json")["profiles"]
    all_label_names = sorted(
        set(old_node_by_name)
        | {row["source_credit_label"] for row in ner_link_rows}
        | {row["source_artist_label"] for row in rhyme_rows},
        key=str.casefold,
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
    for row in atlas["edges"]:
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
        "labels": nodes,
        "lyricalEdges": edges,
        "retrieval": compact_retrieval(),
        "ner": ner,
        "rhyme": rhyme,
        "lineage": {
            "atlas": atlas["artifact"],
            "retrieval": validations["retrieval"]["artifact_id"],
            "ner": validations["ner"]["artifact_id"],
            "rhyme": validations["rhyme"]["artifact_id"],
        },
        "publicBoundary": "Aggregate evidence only; no lyric lines, song/chunk identifiers, embeddings, or verified-person claims.",
    }
    character_map = build_character_map()

    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    characters_encoded = json.dumps(character_map, ensure_ascii=False, separators=(",", ":"))
    (SITE_DATA_DIR / "researchData.json").write_text(encoded, encoding="utf-8")
    (SITE_DATA_DIR / "characterToRhymeFamily.json").write_text(characters_encoded, encoding="utf-8")
    (PUBLIC_DATA_DIR / "researchData.json").write_text(encoded, encoding="utf-8")

    manifest = {
        "artifact": "chinese-rap-results-site-data-v1",
        "labels": len(nodes),
        "lyricalEdges": len(edges),
        "nerEntities": len(payload["ner"]["entities"]),
        "nerLinks": len(payload["ner"]["links"]),
        "rhymeContexts": len(rhyme["contexts"]),
        "dictionaryCharacters": len(character_map),
        "researchDataSha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "status": "pass",
    }
    (PUBLIC_DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
