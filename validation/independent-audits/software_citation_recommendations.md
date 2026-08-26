# Software and model provenance audit

**Audit date:** 2026-08-25 (America/Chicago)  
**Scope:** analysis-critical model and software provenance for the final paper and release  
**Decision:** **Conditionally ready.** The model checkpoints and the NER and written-rhyme environments are sufficiently pinned to support the reported analyses. Two reproducibility gaps should be repaired in the release metadata: the retrieval artifact does not record its runtime package versions, and the original BGE-M3 embedding manifest does not record the device or the realized `use_fp16` value. Neither gap invalidates the frozen downstream results because the resulting vectors and artifacts are hash-addressed.

This audit did not modify the manuscript, data, models, or results.

## Executive finding

The strongest provenance chain is:

1. The exact **BAAI/bge-m3** snapshot is locally resolved to commit `5617a9f61b028005a4858fdac845db406aefb181`.
2. Its local configuration and weight hashes match the clean-text embedding contract exactly.
3. The clean-text embedding matrix is frozen as 21,553 × 1,024 `float32` L2-normalized dense vectors with SHA-256 `90a6d1680c9dba20d53c6e6bebcd3eef24b9343746bf73c10b742db3eabc0c6a`.
4. The exact **CKIP Lab ALBERT-tiny Chinese NER** snapshot is resolved to commit `bcb519856ca93a666b1e48a9daef3f88c9b572a0`; its configuration, weights, and vocabulary hashes all match the NER manifest.
5. The written-rhyme manifest pins Python, NumPy, pypinyin, and scikit-learn versions and module hashes.

The current analysis environment contains the exact versions listed below. Its canonical package-map fingerprint is `b290363fa776f50219cd16ee124fd88b7253d77658bdb9f0d2cef42ee2453152` (SHA-256 of compact, key-sorted UTF-8 JSON). This environment is direct evidence for the NER and written-rhyme runs where versions are manifest-bound; for the earlier BGE run and the retrieval artifact, it is strong local forensic evidence but not a substitute for a run-time lock embedded in those original manifests.

## Recommended compact reproducibility table

This six-row table is compact enough for the manuscript or a primary supplement. Keep hashes in an online appendix if journal space is tight.

| Component | Exact record used | Analysis-relevant settings | Evidence status |
|---|---|---|---|
| Execution environment | CPython 3.12.13, Anaconda build; Windows 11 (`10.0.26200`); Python executable SHA-256 `e062889e…aea35` | Deterministic seeds/settings remain task-specific in each method | Directly recorded for BGE, NER, and rhyme; OS not recorded in retrieval manifest |
| Semantic encoder | `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`; FlagEmbedding 1.4.0; PyTorch 2.11.0+cu128 | Dense head only; 1,024 dimensions; `max_length=2048`; batch size 2; L2-normalized output | Model revision and files verified; device/realized mixed precision absent from original run manifest |
| NER model | `ckiplab/albert-tiny-chinese-ner` revision `bcb519856ca93a666b1e48a9daef3f88c9b572a0`; Transformers 5.14.1; PyTorch 2.11.0+cu128 | `BertTokenizerFast`; 256-token windows; 40-character overlap; batch size 96 | Full runtime and model-file hashes verified |
| Written-rhyme extraction/model | pypinyin 0.55.0; pinyin-data 0.15.0; phrase-pinyin-data 0.19.0; scikit-learn 1.9.0 | `Style.FINALS_TONE3`, `strict=True`, neutral tone `5`; `SGDClassifier(loss="log_loss", average=True)` | Direct version/module-hash evidence for pypinyin, NumPy, and scikit-learn; dictionary-data versions derive from the official 0.55.0 changelog |
| Numerical/data stack | NumPy 2.4.6; pandas 3.0.5; SciPy 1.18.0; scikit-learn 1.9.0 | Sparse TF–IDF, normalization, statistical tests, tables | Directly manifest-bound for NER (NumPy/pandas/SciPy) and rhyme (NumPy/scikit-learn); retrieval versions need binding |
| Transformer support stack | Transformers 5.14.1; tokenizers 0.22.2; huggingface-hub 1.27.0; sentence-transformers 5.7.0 | Local/offline model loading | Transformers is manifest-bound for NER; the rest are current/forensic environment evidence for the BGE run |

**Important wording rule:** the BGE checkpoint's `config.json` contains `"transformers_version": "4.33.0"`. That field describes the library version that serialized the upstream checkpoint; it is **not** the Transformers runtime used by this project. The locally executed stack is Transformers 5.14.1, where directly recorded by the NER manifest, and is strongly supported for the BGE environment by local package metadata. Do not report 4.33.0 as the project runtime.

## Model provenance

### BGE-M3 / FlagEmbedding

- **Official model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), MIT license.
- **Exact model revision:** [`5617a9f61b028005a4858fdac845db406aefb181`](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181).
- **Official model metadata:** 1,024-dimensional output and a model-supported maximum sequence length of 8,192 tokens. This project deliberately used `max_length=2048`; do not imply that 2,048 is the architecture limit.
- **Executed implementation:** `FlagEmbedding.BGEM3FlagModel`, dense-vector head only. Sparse lexical and ColBERT multi-vector outputs were not used.
- **FlagEmbedding release:** [v1.4.0](https://github.com/FlagOpen/FlagEmbedding/releases/tag/v1.4.0), release commit `7ed43d6`.
- **Verified local files:** configuration SHA-256 `26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa`; weights SHA-256 `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`.
- **Frozen analysis vector:** SHA-256 `90a6d1680c9dba20d53c6e6bebcd3eef24b9343746bf73c10b742db3eabc0c6a`.

The project loader sets `use_fp16=bool(torch.cuda.is_available())`, but the 2026-08-08 embedding manifest does not save the realized Boolean or GPU identity. The current machine reports an NVIDIA GeForce RTX 5070 Laptop GPU, CUDA runtime 12.8, and cuDNN 91900, but those current observations are **not proof of the historical embedding device**. The paper should therefore avoid naming historical hardware unless a separate contemporaneous record is found. The frozen vector hash makes all current downstream analyses reproducible from the released/private vector artifact without rerunning the encoder.

### CKIP NER

- **Official model:** [ckiplab/albert-tiny-chinese-ner](https://huggingface.co/ckiplab/albert-tiny-chinese-ner), GPL-3.0 license.
- **Exact model revision:** [`bcb519856ca93a666b1e48a9daef3f88c9b572a0`](https://huggingface.co/ckiplab/albert-tiny-chinese-ner/tree/bcb519856ca93a666b1e48a9daef3f88c9b572a0).
- **Verified local files:** configuration SHA-256 `4828a80f3c7eb67514f3d9c76ed2544149bad4e498f6b74e4978b48a542b5b5b`; weights SHA-256 `03f6e38f92ada4b59b88ae9122a50a0c98b85f07722ff1234048928e55ed10d3`; vocabulary SHA-256 `45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c`.
- **Runtime:** Transformers 5.14.1 and PyTorch 2.11.0+cu128, recorded in the public NER manifest.
- **Loading route:** `AutoModelForTokenClassification` with `BertTokenizerFast`.

The project does **not** import or execute the separate `ckip-transformers` Python package. Cite the CKIP model card and its exact revision, plus Hugging Face Transformers and PyTorch. Do not invent a `ckip-transformers` package version. Because this is a general-domain Traditional Chinese model applied to Chinese rap, the paper should retain the existing domain-shift caveat and human-validation design.

## Software provenance by downstream task

| Task | Direct imports or execution dependency | Exact project version | Provenance quality |
|---|---|---:|---|
| BGE-M3 encoding | FlagEmbedding | 1.4.0 | Version recorded in run manifest |
| BGE-M3 encoding / CKIP NER | PyTorch | 2.11.0+cu128 | Recorded in both relevant manifests |
| CKIP NER | Transformers | 5.14.1 | Recorded in NER manifest |
| BGE support stack | Transformers / tokenizers / huggingface-hub | 5.14.1 / 0.22.2 / 1.27.0 | Installed before the embedding run and present unchanged; not named in original BGE manifest |
| Written-rhyme extraction | pypinyin | 0.55.0 | Version and module-file hash recorded |
| Written-rhyme classifier | scikit-learn | 1.9.0 | Version and module-file hash recorded |
| Retrieval baseline | NumPy / SciPy / scikit-learn | 2.4.6 / 1.18.0 / 1.9.0 | Current/forensic environment only; retrieval manifest omits versions |
| NER aggregation/tests | NumPy / pandas / SciPy | 2.4.6 / 3.0.5 / 1.18.0 | Recorded in NER manifest |

The installed `sentence-transformers` 5.7.0 package is part of the BGE environment but is not a direct analytical method invoked by the project builders. It belongs in the environment lock, not necessarily in the paper's prose or references.

## Reproducibility issues and smallest remediations

| Severity | Confidence | Finding | Why it matters | Smallest remediation |
|---|---|---|---|---|
| Medium | High | The retrieval manifest lacks Python/platform and NumPy/SciPy/scikit-learn versions. | A future rebuild could differ through tokenization, sparse operations, or library defaults. | Before release, add a non-result-changing environment sidecar or rebuild only the manifest so it records Python 3.12.13, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0, the Python executable hash, and the environment fingerprint. |
| Medium | High | The BGE run manifest omits device, CUDA/cuDNN, and the realized `use_fp16` setting. | Exact raw re-encoding may vary slightly by precision/backend. | Preserve the existing vector hash; for any future encoding run, record `torch.cuda.is_available()`, device name, CUDA/cuDNN, and `use_fp16`. Do not retrospectively assert current hardware was used. |
| Low | High | The BGE reference in the current manuscript points to the arXiv record although a peer-reviewed ACL Findings version exists. | Journal bibliography should prefer the archival venue. | Replace the arXiv-form entry with DOI `10.18653/v1/2024.findings-acl.137`. |
| Low | High | Exact software/model records are not yet represented consistently in the bibliography. | Readers cannot distinguish an algorithm paper from the precise executable artifact. | Cite the method paper in the main bibliography and put exact version/revision records in this environment table or data/software availability statement. |
| Low | High | The NER manifest abbreviates the model revision in the directory name (`bcb5198`). | Seven characters are human-readable but not ideal archival provenance. | Publish the full revision SHA and the three verified file hashes in the supplement. |

No result rerun is required solely to correct the citations. If the retrieval manifest is regenerated, preserve every existing output hash and state explicitly that only provenance metadata changed.

## Recommended citation policy

Use two layers:

- **Scholarly method citations in the manuscript:** BGE-M3, Transformers, PyTorch, scikit-learn, SciPy, and NumPy.
- **Executable artifact citations in the reproducibility supplement:** exact BGE and CKIP model revisions; FlagEmbedding 1.4.0; pypinyin 0.55.0; and version-specific Zenodo/PyPI/GitHub records for the numerical stack.

This avoids an overlong methods bibliography while retaining enough information to reproduce the exact environment.

## Normalized bibliography entries

The entries below use a compact author–year style and official/primary records.

1. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D. and Liu, Z. (2024). M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. *Findings of the Association for Computational Linguistics: ACL 2024*, 2318–2335. https://doi.org/10.18653/v1/2024.findings-acl.137
2. FlagOpen. (2026). *FlagEmbedding* (Version 1.4.0) [Computer software]. GitHub. https://github.com/FlagOpen/FlagEmbedding/releases/tag/v1.4.0
3. Beijing Academy of Artificial Intelligence. (2024). *BAAI/bge-m3* (revision `5617a9f61b028005a4858fdac845db406aefb181`) [Pretrained language model]. Hugging Face. https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181
4. CKIP Lab. (2022). *ckiplab/albert-tiny-chinese-ner* (revision `bcb519856ca93a666b1e48a9daef3f88c9b572a0`) [Pretrained token-classification model]. Hugging Face. https://huggingface.co/ckiplab/albert-tiny-chinese-ner/tree/bcb519856ca93a666b1e48a9daef3f88c9b572a0
5. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q. and Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38–45. https://doi.org/10.18653/v1/2020.emnlp-demos.6
6. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J. and Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*. https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html
7. mozillazg and contributors. (2025). *pypinyin* (Version 0.55.0) [Computer software]. PyPI. https://pypi.org/project/pypinyin/0.55.0/
8. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
9. Virtanen, P., Gommers, R., Oliphant, T. E., et al. and SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*, 261–272. https://doi.org/10.1038/s41592-019-0686-2
10. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362. https://doi.org/10.1038/s41586-020-2649-2
11. The pandas development team. (2026). *pandas* (Version 3.0.5) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21500199

Optional exact-version software citations for an appendix:

- The scikit-learn developers. (2026). *scikit-learn* (Version 1.9.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20510517
- Gommers, R., Virtanen, P., Haberland, M., et al. (2026). *SciPy* (Version 1.18.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20764140
- The pandas development team. (2026). *pandas* (Version 3.0.5) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21500199

For pandas, a journal that prefers a scholarly method reference may additionally accept: McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. https://doi.org/10.25080/Majora-92bf1922-00a

## Official source register

| Record | Official/primary URL | Use |
|---|---|---|
| BGE-M3 paper | https://aclanthology.org/2024.findings-acl.137/ | Scholarly model-method citation |
| BGE-M3 model card | https://huggingface.co/BAAI/bge-m3 | Architecture/license/usage metadata |
| BGE-M3 pinned snapshot | https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181 | Exact executable model artifact |
| FlagEmbedding 1.4.0 | https://github.com/FlagOpen/FlagEmbedding/releases/tag/v1.4.0 | Exact inference-library release |
| CKIP model card | https://huggingface.co/ckiplab/albert-tiny-chinese-ner | Model task/license metadata |
| CKIP pinned snapshot | https://huggingface.co/ckiplab/albert-tiny-chinese-ner/tree/bcb519856ca93a666b1e48a9daef3f88c9b572a0 | Exact executable NER artifact |
| Transformers paper | https://aclanthology.org/2020.emnlp-demos.6/ | Scholarly library citation |
| Transformers 5.14.1 | https://github.com/huggingface/transformers/releases/tag/v5.14.1 | Exact release record |
| PyTorch paper | https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html | Scholarly framework citation |
| PyTorch 2.11.0 | https://github.com/pytorch/pytorch/releases/tag/v2.11.0 | Exact release record |
| pypinyin 0.55.0 | https://pypi.org/project/pypinyin/0.55.0/ | Exact package record |
| pypinyin usage | https://pypinyin.readthedocs.io/zh-cn/master/usage.html | `FINALS_TONE3`/strict behavior |
| pypinyin changelog | https://github.com/mozillazg/python-pinyin/blob/master/CHANGELOG.rst | Bundled dictionary-data versions |
| scikit-learn paper | https://jmlr.org/papers/v12/pedregosa11a.html | Scholarly ML citation |
| scikit-learn 1.9.0 | https://doi.org/10.5281/zenodo.20510517 | Exact software version |
| SciPy paper | https://doi.org/10.1038/s41592-019-0686-2 | Scholarly scientific-computing citation |
| SciPy 1.18.0 | https://doi.org/10.5281/zenodo.20764140 | Exact software version |
| NumPy paper | https://doi.org/10.1038/s41586-020-2649-2 | Scholarly array-computing citation |
| NumPy citation guide | https://numpy.org/citing-numpy/ | Official citation guidance |
| NumPy 2.4.6 | https://github.com/numpy/numpy/releases/tag/v2.4.6 | Exact release record |
| pandas citation guide | https://pandas.pydata.org/about/citing.html | Official citation guidance |
| pandas 3.0.5 | https://doi.org/10.5281/zenodo.21500199 | Exact software version |

## Local audit evidence

| Local record | SHA-256 at audit time |
|---|---|
| `outputs/embedding-benchmark-v1/bge-m3-corpus-v1/model_manifest.json` | `193c99da281dae521736de9c5a12961faa4a5d36cb0ae8e927a635aa53cfef52` |
| `work/private-canonical-clean-text-embeddings-v1/canonical_clean_text_embedding_contract_v1.json` | `d65a446dc8ac3cb9fb5f513c63e9d10a3f6a94a6c1cebd28953b045170600c5b` |
| `outputs/chinese-rap-downstream-retrieval-v1/manifest.json` | `facd87827a50767e7507d7dc039d4b16dbb97870af5fb26c44e05bcd54fd3f3c` |
| `outputs/chinese-rap-ner-cultural-graph-v1/manifest.json` | `d60986eb4cb484cfba9e5957e2eb20292e93dffc64516e93b837b65a491096af` |
| `outputs/chinese-rap-written-rhyme-v1/manifest.json` | `55131351ba5b1d1b8f49c60f3750ebf930df3b2199cc974ca5314d93f975f7e3` |
| `work/canonical_semantic_embeddings_v1.py` | `e506c70c35f893b3386760c002972ece31b8082b2d9392765048fd0c3409946e` |
| Retrieval builder | `340564076f8a9cbb1b4b90eabab9d82636b5dbaba52333117cecee5deef9b016` |
| NER builder | `8260592777d8fdddf79ca5c82c0b6e884c7c0efb4757e1824ffdd5c957c0ea75` |
| Written-rhyme builder | `57e1b53bf505bde9ba02906db92d3aa4c6f666c7afac5a1cb31d7d356255e47d` |

The companion JSON file contains the same recommendations in machine-readable form.
