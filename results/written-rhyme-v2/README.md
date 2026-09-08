# Written-ending continuation on corpus v2

Prediction of the next line's written-ending family (seventeen tone-free classes of the strict `pypinyin` final) from the sequence so far, on corpus v2, with songs partitioned by leakage group rather than individually. Built by `src/build_written_rhyme_v2.py`, which imports the v1 builder's line filter, event construction, leakage filter, models, and evaluation so the two cannot drift in definition.

| File | What it holds |
| --- | --- |
| `analysis_summary.json` | Split audit (5,147 / 1,097 / 1,104 songs; 0 text components crossing a partition), leakage filter counts, model metrics with song-cluster bootstrap intervals, paired deltas, abstention thresholds, line-frame exclusions. |
| `model_metrics.csv` | Five models on the 51,516 leakage-safe test events from 1,064 songs. |
| `paired_model_deltas.csv` | Hierarchical model against Markov, flat context, and the no-label ablation, 2,000 song-cluster replicates. |
| `stratified_metrics.csv` | Continuation against switch events, and other strata, per model. |
| `per_label_metrics.csv` | Per-label results for labels with enough leakage-safe events. |

These results replace `results/written-rhyme-v1/`, whose predictive metrics were withheld pending exactly this rerun. They are not comparable point to point: the population differs and the split unit differs. The target is a dictionary-estimated written ending; nothing here measures performed rhyme, flow, delivery, or audio.
