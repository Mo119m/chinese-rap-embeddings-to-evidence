#!/usr/bin/env python3
"""Held-out-song retrieval on the PD-002 repaired corpus.

The v1 result stands as a frozen-corpus benchmark and is not edited: its bytes are pinned by
results/retrieval-inductive-sensitivity-v1/manifest.json, and the amendment records it as
internally valid on the population it used. This is a separate artifact on corpus v2.

What changes, and only this:

* Population. 25,026 chunks over 7,391 song records, from the repaired corpus, replacing the
  22,132-chunk snapshot the legacy cleaner produced by deleting rows.
* Vectors. The single recorded BGE-M3 run over corpus v2, replacing a vector set whose
  device and precision were never captured and which step 1 established cannot be identified.
* Leakage unit. The union of corpus v2's exact-text components and this task's own
  near-duplicate rule, from leakage_groups_v2 -- neither contains the other, and the
  measurements are in that module.
* Duplicate control by representation, not deletion. v1 deleted chunks whose exact cleaned
  text occurred under more than one source-credit label. PD-002 replaces that: "Duplicate
  control is represented, not enacted by deleting chunks." Every chunk is kept and the
  grouping carries the control.

What does not change is imported from the v1 builder rather than restated here, so the two
results cannot drift apart in their definitions: the TF-IDF fit, the leave-group-out
geometry, the ranking and its tie-breaking, the metric definitions, the occurrence-wise
paired two-stage bootstrap, the random seed and the replicate count.

    python src/build_downstream_retrieval_v2.py --private-root <ni-k>

The v2 numbers REPLACE the v1 numbers. They are not comparable point to point: the corpus
population differs, and the embedding contract records that the v1 vector space cannot be
identified. Nothing here should be read as a delta against a published v1 figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_chinese_rap_downstream_retrieval_v1 as v1  # noqa: E402
from leakage_groups_v2 import build_groups, group_audit, normalise_document  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

csv.field_size_limit(10 ** 9)

ARTIFACT_ID = "chinese-rap-downstream-retrieval-v2"
VERSION = "2.0.0"
OUT_DIR = ROOT / "results" / "retrieval-v2"

# What the corpus must be. A build against a different corpus is a different result and
# should stop rather than quietly produce numbers under this artifact's name.
CORPUS_CONTENT_SHA256 = "8102da32084496fac0fb51c7b78485dec9e216696ff98e21b922fa1708b4313b"
EMBEDDINGS_SHA256 = "247e961918c5d8c3f13967e8a882e2df2222ca1d452c72bd681665db41e51da1"
EXPECTED_CHUNKS = 25026
EXPECTED_SONGS = 7391

# A label needs enough songs that leave-group-out leaves something behind. v1 took this from
# the repertoire graph's node table, which also required an "effective text mass" defined on
# the graph's own weighting; the graph is no longer a published result, so the rule is stated
# here in terms of the corpus alone.
MINIMUM_SONGS_PER_LABEL = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_content_sha256(rows) -> str:
    """The published content contract, recomputed here so a swapped corpus cannot pass."""
    digest = hashlib.sha256()
    for row in rows:
        canonical = json.dumps(
            [str(row["source_credit_label"]), str(row["song_id"]), str(row["song_title"]),
             int(row["chunk_id"]), int(row["source_order"]), str(row["cleaned_text"])],
            ensure_ascii=False, separators=(",", ":"))
        digest.update((canonical + "\n").encode("utf-8"))
    return digest.hexdigest()


def corpus_version() -> str:
    """Which corpus the downstream experiments read: v2 (default) or the cleaned v3.

    Set CHINESE_RAP_CORPUS=v3 to rerun any experiment on corpus v3 with its own recorded
    vectors; pass --out-dir results/retrieval-v3 so the v2 results are not overwritten.
    """
    import os
    version = os.environ.get("CHINESE_RAP_CORPUS", "v2")
    if version not in ("v2", "v3"):
        raise SystemExit(f"CHINESE_RAP_CORPUS must be v2 or v3, not {version!r}")
    return version


def expected(v2_value, v3_value):
    """A self-check value keyed to the corpus in use; None means the check is skipped."""
    return v2_value if corpus_version() == "v2" else v3_value


def load(private_root: Path):
    if corpus_version() == "v3":
        from corpus_v3 import load_v3
        rows, vectors, state = load_v3(private_root)
        return rows, vectors, {"contract_version": state["contract"].get("contract_version"),
                               "configuration": state["contract"].get("configuration"),
                               "corpus": "v3", "embeddings_sha256": state["sha256"]}
    corpus_path = private_root / "work" / "private-repaired-corpus-v2" / "repaired_lyric_chunks_v2.csv"
    embed_dir = private_root / "work" / "private-repaired-corpus-v2-embeddings"
    vectors_path = embed_dir / "repaired_corpus_v2_bge_m3_embeddings.npy"
    row_map_path = embed_dir / "repaired_corpus_v2_embedding_row_map.csv"
    contract_path = embed_dir / "repaired_corpus_v2_embedding_contract.json"
    for path in (corpus_path, vectors_path, row_map_path, contract_path):
        if not path.is_file():
            raise SystemExit(f"missing private input: {path}")

    rows = list(csv.DictReader(corpus_path.open(encoding="utf-8")))
    digest = corpus_content_sha256(rows)
    if digest != CORPUS_CONTENT_SHA256:
        raise SystemExit(
            f"corpus content digest {digest} is not the published corpus v2 "
            f"{CORPUS_CONTENT_SHA256}")
    if len(rows) != EXPECTED_CHUNKS:
        raise SystemExit(f"expected {EXPECTED_CHUNKS} chunks, read {len(rows)}")

    vectors = np.load(vectors_path)
    vectors_digest = sha256_file(vectors_path)
    if vectors_digest != EMBEDDINGS_SHA256:
        raise SystemExit(
            f"embedding file digest {vectors_digest} is not the recorded run "
            f"{EMBEDDINGS_SHA256}")
    if vectors.shape != (EXPECTED_CHUNKS, 1024):
        raise SystemExit(f"expected ({EXPECTED_CHUNKS}, 1024) vectors, got {vectors.shape}")

    row_map = list(csv.DictReader(row_map_path.open(encoding="utf-8")))
    if len(row_map) != len(rows):
        raise SystemExit(f"row map has {len(row_map)} rows for {len(rows)} chunks")
    # Positional, and checked. The row map carries no text, so an off-by-one would be
    # invisible in the outputs and would silently attach every vector to the wrong chunk.
    for index, (mapped, row) in enumerate(zip(row_map, rows)):
        if (int(mapped["row_index"]) != index
                or mapped["song_id"] != row["song_id"]
                or int(mapped["chunk_id"]) != int(row["chunk_id"])
                or int(mapped["source_order"]) != int(row["source_order"])):
            raise SystemExit(f"row map diverges from the corpus at row {index}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return rows, vectors, contract


def build_songs(rows, vectors):
    """One document and one centroid per song record, in corpus order."""
    chunks_by_song: dict[str, list[int]] = defaultdict(list)
    label_by_song: dict[str, str] = {}
    components_by_song: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        song = row["song_id"]
        chunks_by_song[song].append(index)
        label_by_song[song] = row["source_credit_label"]
        components_by_song[song].add(row["text_component_id"])

    documents: dict[str, str] = {}
    centroids: dict[str, np.ndarray] = {}
    for song, indices in chunks_by_song.items():
        ordered = sorted(indices, key=lambda i: int(rows[i]["source_order"]))
        documents[song] = "".join(rows[i]["cleaned_text"] for i in ordered)
        # A plain mean. v1 weighted chunks by a graph-derived comparison weight; corpus v2
        # has no chunk weight, and PD-002 keeps repetition inside a song rather than
        # discounting it, so repeated lines move the song's own vector -- which is what the
        # song is.
        centroids[song] = vectors[ordered].mean(axis=0)
    return chunks_by_song, label_by_song, components_by_song, documents, centroids


def build(private_root: Path, out_dir: Path) -> int:
    print("loading corpus v2 and its recorded embedding run", flush=True)
    rows, vectors, contract = load(private_root)
    chunks_by_song, label_by_song, components_by_song, documents, centroids = build_songs(
        rows, vectors)
    print(f"  {len(rows):,} chunks over {len(chunks_by_song):,} song records", flush=True)
    if len(chunks_by_song) != EXPECTED_SONGS:
        raise SystemExit(f"expected {EXPECTED_SONGS} songs, built {len(chunks_by_song)}")

    songs_by_label: dict[str, list[str]] = defaultdict(list)
    for song, label in label_by_song.items():
        songs_by_label[label].append(song)

    # length filter first, then eligibility, so a label is judged on songs that can be queries
    long_enough = {
        song for song in chunks_by_song
        if len(v1.normalized_text(documents[song])) >= v1.MIN_EFFECTIVE_CHARACTERS
    }
    eligible_labels = sorted(
        label for label, members in songs_by_label.items()
        if sum(1 for song in members if song in long_enough) >= MINIMUM_SONGS_PER_LABEL
    )
    songs = sorted(song for song in long_enough
                   if label_by_song[song] in set(eligible_labels))
    print(f"  {len(songs):,} length-qualified queries over "
          f"{len(eligible_labels)} eligible labels "
          f"(>= {MINIMUM_SONGS_PER_LABEL} qualifying songs)", flush=True)

    print("grouping: exact-text components unioned with near-duplicates", flush=True)
    normalised = {song: normalise_document(documents[song]) for song in songs}
    groups = build_groups(songs, components_by_song, normalised)
    audit = group_audit(groups, label_by_song, components_by_song)
    print(f"  {audit['groups']:,} groups, largest {audit['largest_group_songs']}, "
          f"min per label {audit['minimum_groups_in_a_label']}", flush=True)
    if audit["text_components_split_across_groups"]:
        raise SystemExit("a text component is split across groups; PD-002 rule 3 is violated")
    if audit["minimum_groups_in_a_label"] < 2:
        raise SystemExit("a label has fewer than two groups; leave-group-out has nothing left")

    label_index = {label: index for index, label in enumerate(eligible_labels)}
    song_labels = np.asarray([label_index[label_by_song[song]] for song in songs], dtype=np.int64)
    dense = v1.l2_normalize_dense(np.stack([centroids[song] for song in songs]))
    corpus_documents = [documents[song] for song in songs]

    def group_array(mapping):
        order = {group: index for index, group in enumerate(sorted(set(mapping.values())))}
        return np.asarray([order[mapping[song]] for song in songs], dtype=np.int64)

    # The two halves of the union, kept as ablations. v1's ablation set had one arm for
    # "near-duplicate guard removed" and one for "shared-text exclusion removed"; the second
    # has no meaning here, because v2 does not exclude shared text at all. Since neither half
    # of the union contains the other, the informative pair is each half alone.
    components_only = build_groups(songs, components_by_song, normalised, threshold=1.01)
    near_only = build_groups(songs, {song: set() for song in songs}, normalised)

    print("fitting character 2-5 gram TF-IDF", flush=True)
    lexical = v1.fit_tfidf(corpus_documents)
    print(f"  {lexical.shape[1]:,} features", flush=True)

    print("scoring group-held-out label profiles (union, components-only, near-only)",
          flush=True)
    profiles = {
        "union": v1.score_leave_group_out(dense, lexical, song_labels,
                                          group_array(groups), len(eligible_labels)),
        "components_only": v1.score_leave_group_out(dense, lexical, song_labels,
                                                    group_array(components_only),
                                                    len(eligible_labels)),
        "near_only": v1.score_leave_group_out(dense, lexical, song_labels,
                                              group_array(near_only), len(eligible_labels)),
    }

    def fusion(entry):
        return (v1.zscore_rows(entry.dense) + v1.zscore_rows(entry.lexical)) / 2.0

    SYSTEM_SCORES = {
        "BGE-M3 dense": profiles["union"].dense,
        "character 2-5 gram TF-IDF": profiles["union"].lexical,
        "equal-weight z-score fusion": fusion(profiles["union"]),
        "raw-score equal fusion (score-scale ablation)":
            (profiles["union"].dense + profiles["union"].lexical) / 2.0,
        "fusion, near-duplicate half of the guard removed": fusion(profiles["components_only"]),
        "fusion, component half of the guard removed": fusion(profiles["near_only"]),
    }

    print("ranking", flush=True)
    metric_arrays = {}
    for name, scores in SYSTEM_SCORES.items():
        ranks, _ = v1.rank_system(scores, song_labels)
        metric_arrays[name] = v1.metrics_from_ranks(ranks)

    # component_metric_values and run_bootstrap read the module-level SYSTEMS and METRICS at
    # call time, and the bootstrap is PAIRED: one resample drives every system at once, which
    # is what makes system differences comparable. Bootstrapping each system separately would
    # silently discard that pairing while still producing plausible intervals -- so the
    # constants are redirected here rather than the paired logic being restated.
    v1_systems, v1_metrics = v1.SYSTEMS, v1.METRICS
    v1.SYSTEMS = tuple(SYSTEM_SCORES)
    v1.METRICS = tuple(next(iter(metric_arrays.values())))
    if len(v1.SYSTEMS) != len(SYSTEM_SCORES) or set(v1.METRICS) != set(
            next(iter(metric_arrays.values()))):
        raise SystemExit("system/metric redirection did not take effect")
    try:
        evaluation = v1.Evaluation(
            ranks={}, top_labels={}, metric_arrays=metric_arrays,
            strict_profiles=profiles["union"],
            exact_only_profiles=profiles["components_only"],
            shared_profiles=profiles["near_only"],
            strict_tfidf_features=int(lexical.shape[1]),
            shared_tfidf_features=int(lexical.shape[1]))
        print("bootstrapping (paired across all systems)", flush=True)
        components, label_group_counts = v1.component_metric_values(
            evaluation, song_labels, group_array(groups))
        outcome = v1.run_bootstrap(components)
        systems, metrics = list(v1.SYSTEMS), list(v1.METRICS)
    finally:
        v1.SYSTEMS, v1.METRICS = v1_systems, v1_metrics

    summary = {}
    for system_offset, name in enumerate(systems):
        summary[name] = {
            metric: {
                "point": float(outcome.point_macro[system_offset, metric_offset]),
                "interval_low": float(outcome.lower[system_offset, metric_offset]),
                "interval_high": float(outcome.upper[system_offset, metric_offset]),
            }
            for metric_offset, metric in enumerate(metrics)
        }
        entry = summary[name]["mrr"]
        print(f"  {name[:46]:46s} MRR {entry['point']:.3f} "
              f"[{entry['interval_low']:.3f}, {entry['interval_high']:.3f}]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_id": ARTIFACT_ID,
        "version": VERSION,
        "claim_boundary": v1.CLAIM_BOUNDARY,
        "corpus": {
            "content_sha256": CORPUS_CONTENT_SHA256,
            "chunks": len(rows),
            "song_records": len(chunks_by_song),
            "queries_after_length_filter": len(songs),
            "eligible_labels": len(eligible_labels),
            "minimum_songs_per_label": MINIMUM_SONGS_PER_LABEL,
        },
        "embeddings": {
            "sha256": EMBEDDINGS_SHA256,
            "contract_version": contract.get("contract_version"),
            "configuration": contract.get("configuration"),
        },
        "leakage_unit": {
            "rule": ("connected components of (shares a corpus v2 exact-text component) "
                     "union (whole-song trigram Jaccard >= 0.80)"),
            "ablations": ("each half of the union is scored alone, because neither half "
                          "contains the other: 404 of 799 multi-song components would be "
                          "split by the document rule, and 44 near-duplicate pairs share no "
                          "component"),
            **audit,
        },
        "metrics": summary,
        "bootstrap": {
            "replicates": v1.BOOTSTRAP_REPLICATES,
            "design": ("paired two-stage cluster resample: labels drawn with replacement, "
                       "then components within each drawn label, applied to every system at "
                       "once so system differences stay comparable"),
            "within_label_component_units": int(label_group_counts.sum()),
        },
        "reused_from_v1_builder": [
            "fit_tfidf", "score_leave_group_out", "rank_system", "metrics_from_ranks",
            "component_metric_values", "run_bootstrap", "zscore_rows", "l2_normalize_dense",
        ],
        "seed": v1.RANDOM_SEED,
        "not_comparable_to": (
            "results/retrieval-v1. The corpus population differs (25,026 chunks over 7,391 "
            "song records against 22,132 over 7,214), and the v1 vectors were produced under "
            "a device and precision that were never recorded and that the configuration probe "
            "established cannot be identified. These numbers replace the v1 numbers; no "
            "difference between them may be reported as an effect."
        ),
        "duplicate_control": (
            "Represented, not enacted by deletion. v1 removed chunks whose exact cleaned text "
            "occurred under more than one source-credit label; every chunk is retained here "
            "and the grouping carries the control, per PD-002's replacement rule."
        ),
        "privacy": "aggregate only; no lyric text, song or chunk identifiers, or vectors",
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")

    # The tables v1 published, in v1's shape, so the figure pipeline can read v2 without
    # a second code path. Differences come from the same replicate tensor as the
    # intervals above, so they are paired: one component draw indexes every system.
    primary = ("BGE-M3 dense", "character 2-5 gram TF-IDF", "equal-weight z-score fusion")
    metric_lines = []
    for name in primary:
        for metric_offset, metric in enumerate(metrics):
            entry = summary[name][metric]
            metric_lines.append({
                "system": name, "task_role": "primary_comparison",
                "aggregation": "source_credit_label_macro_duplicate_group_adjusted",
                "metric": metric, "estimate": f"{entry['point']:.6f}",
                "ci95_lower": f"{entry['interval_low']:.6f}",
                "ci95_upper": f"{entry['interval_high']:.6f}",
                "queries": len(songs), "source_credit_labels": len(eligible_labels),
                "global_duplicate_components": audit["groups"],
                "label_stratum_component_units": int(label_group_counts.sum()),
                "bootstrap_replicates": v1.BOOTSTRAP_REPLICATES,
            })
    difference_lines = []
    for left, right in (("equal-weight z-score fusion", "BGE-M3 dense"),
                        ("equal-weight z-score fusion", "character 2-5 gram TF-IDF"),
                        ("character 2-5 gram TF-IDF", "BGE-M3 dense")):
        l, r = systems.index(left), systems.index(right)
        for metric_offset, metric in enumerate(metrics):
            draws = outcome.replicates[:, l, metric_offset] - outcome.replicates[:, r, metric_offset]
            low, high = np.quantile(draws, [0.025, 0.975])
            point = float(outcome.point_macro[l, metric_offset] - outcome.point_macro[r, metric_offset])
            direction = "left_higher" if low > 0 else ("right_higher" if high < 0 else "inconclusive")
            difference_lines.append({
                "comparison": f"{left} minus {right}", "metric": metric,
                "estimate_delta": f"{point:.6f}", "ci95_lower": f"{float(low):.6f}",
                "ci95_upper": f"{float(high):.6f}", "interval_direction": direction,
                "paired_two_stage_bootstrap_replicates": v1.BOOTSTRAP_REPLICATES,
            })
    for filename, lines in (("metrics.csv", metric_lines), ("uncertainty.csv", difference_lines)):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(lines[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)
        (out_dir / filename).write_text(buffer.getvalue(), encoding="utf-8", newline="")
    print(f"\nwrote {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True,
                        help="directory containing work/private-repaired-corpus-v2 and "
                             "work/private-repaired-corpus-v2-embeddings")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.out_dir.resolve().is_relative_to(ROOT / "work"):
        raise SystemExit("--out-dir must not be inside work/")
    return build(args.private_root.resolve(), args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
