# Independent final audit — Chinese Rap NER + Cultural Graph v1.1

**Verdict: PASS — 19/19 checks passed.**

Audit time: 2026-08-26T02:01:24Z  
Frozen artifact: `outputs/chinese-rap-ner-cultural-graph-v1`  
Production builder SHA-256: `8260592777d8fdddf79ca5c82c0b6e884c7c0efb4757e1824ffdd5c957c0ea75`  
Independent recomputation script SHA-256: `79e3404df0b8e2f5d518d187cf2b126b29b792ef064dfddda3a75f1b4b2ac186`

## Audit boundary

This was a read-only audit. The recomputation script does not import the production builder and did not modify the frozen public or private outputs. It independently rebuilt the documented fixed-corpus estimands from:

- the eligible cleaned-chunk sidecar;
- the eligible-song table;
- the frozen 204-label registry;
- the curated lexicon and T1 named-person ledger; and
- the persisted private candidate-occurrence map.

The public and private manifests, all input hashes, and the current builder hash match exactly.

## Population and exclusion checks

| Check | Independently recomputed result |
|---|---:|
| Eligible cleaned chunks | 21,553 |
| Cross-label shared cleaned-text hashes | 2,187 |
| Graph-eligible source-credit labels | 204 |
| Post-exclusion label/song membership units | 5,681 |
| Post-exclusion distinct full-song-content units | 5,681 |
| Song units attached to multiple graph labels | 0 |
| Private candidate rows | 44,970 |
| Preserved lexicon occurrence rows | 37,983 |
| Unique private candidate IDs | 44,970 |

The recomputed shared-hash set is identical to the private exclusion ledger. No strict occurrence supporting a released label–entity edge uses a shared cleaned-text hash.

## Candidate-agreement and inventory checks

The audit reconstructed line boundaries from the raw cleaned chunks, rather than trusting the reported line-frame totals.

| Unit | Exact span/type agreement | Strict high-consistency agreement |
|---|---:|---:|
| Repeated corpus occurrences | 3,566 | 3,290 |
| Unique line/span/type units | 2,011 | 1,888 |

The entity inventory was also reconstructed independently. It matches the frozen release exactly:

- 33 defensible corpuswide sensitivity entities;
- 23 after cross-label shared-text exclusion across all labels; and
- 22 in the primary shared-text-excluded 204-label universe.

Every persisted count, support value, agreement rate, and mean transformer confidence for the 22 primary entities matches the recomputation.

## Label–entity association audit

All **4,488** tests (204 labels × 22 entities) were independently recomputed. Song-unit counts, Jeffreys posterior means, beta intervals, conservative risk-ratio bounds, one-sided hypergeometric p-values, and BH-adjusted q-values match the frozen table within its published rounding. The same six rows pass the support, effect, uncertainty, and FDR gates:

| Source-credit label | Lyric reference | Support | Eligible label units | Shrunken RR | Conservative 95% interval | BH q | Release class |
|---|---|---:|---:|---:|---:|---:|---|
| 黑麦 | 天津 | 10 | 33 | 151.699 | [49.780, 455.252] | 1.47e-14 | HIGH |
| Tangoz | 杭州 | 10 | 38 | 74.124 | [26.874, 186.439] | 3.04e-12 | HIGH |
| 泰格西 | 湖南 | 5 | 25 | 159.556 | [33.222, 693.820] | 1.25e-06 | SUPPORTED |
| 国蛋 | 台北 | 6 | 36 | 63.991 | [17.032, 200.534] | 2.38e-06 | SUPPORTED |
| GALI | 上海 | 8 | 36 | 13.168 | [5.285, 26.485] | 1.98e-04 | SUPPORTED |
| 法老 | 上海 | 7 | 41 | 10.124 | [3.757, 21.313] | 4.96e-03 | SUPPORTED |

These are source-credit-label **lyrical repertoire associations** in the frozen corpus. They are not evidence of residence, origin, biography, personal preference, identity, affiliation, or a social relationship.

## Co-mention audit

All **231** possible entity-pair tests were independently recomputed using all 5,681 eligible full-song-content units as the denominator, including units with no released entity. Five pairs pass the pre-FDR support/lift/NPMI gate; the same four frozen pairs remain after BH-FDR:

| Entity pair | Co-mention units | Source-credit labels | NPMI | BH q |
|---|---:|---:|---:|---:|
| 伦敦 — 巴黎 | 5 | 5 | 0.500 | 4.21e-05 |
| 中文 — 英文 | 6 | 5 | 0.324 | 4.01e-03 |
| 上海 — 巴黎 | 7 | 5 | 0.251 | 1.70e-02 |
| 上海 — 新疆 | 5 | 5 | 0.270 | 3.10e-02 |

These edges mean only that two provisional references occur in the same song-content unit. They do not encode a named semantic relation, collaboration, influence, affiliation, or social connection.

## Sensitivity reconciliation

The published funnel is internally and independently consistent:

| Stage | Entities | Label–entity links | Co-mentions |
|---|---:|---:|---:|
| Legacy all-label, shared text included | 33 | 85 | 9 |
| All labels, shared text excluded, legacy gates | 23 | 40 | 1 |
| Primary 204-label universe, shared text excluded, legacy gates | 22 | 40 | 1 |
| v1.1 uncertainty + all-song denominator + BH-FDR release | 22 | 6 | 4 |

The apparent rise from one to four co-mention edges at the last stage is not a relaxation of evidence control: the first value uses the legacy entity-bearing-song denominator, while the final value uses the corrected denominator of all eligible song units and then applies BH-FDR. The output documents this denominator change explicitly.

## Privacy and claim audit

- All 16 public CSVs contain aggregate evidence only.
- No public CSV exposes song IDs, chunk IDs, lyric-content hashes, cleaned-text hashes, raw text, context snippets, candidate IDs, or embeddings.
- No public CSV contains a bare SHA-256 value or a cell longer than 216 characters.
- The public network has 28 nodes and 10 edges: 22 entity nodes, 6 released source-label nodes, 6 label→reference edges, and 4 same-song co-mention edges.
- All result rows remain explicitly provisional.
- The 800-task dual-review package is present, but R1, R2, and adjudication fields are blank. Completed human gold remains zero.
- Consequently, the release correctly reports no precision, recall, or F1.

## Final decision

The frozen NER/cultural-graph artifact is **technically consistent, reproducible from the persisted evidence, privacy-bounded, and ready to integrate into the paper, figures, website, and public repository as a provisional corpus-analysis result**.

It is not a validated general-purpose Chinese-rap NER benchmark. Publication language must preserve the current boundary: automated, cross-method, fixed-corpus lyric-reference evidence pending dual human annotation and adjudication.

