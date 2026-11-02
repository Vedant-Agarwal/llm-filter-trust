# When Can You Trust LLM-Extracted Filters? Reproduction code

Code, saved extractor outputs and results for "When Can You Trust LLM-Extracted Filters? A Label-Free Catalog Check for Product Search" (Vedant Agarwal and Sumit Kumar, ISIR-eCom 2026).

The paper applies attributes extracted by an LLM as hard filters on two product-search benchmarks with human labels (Amazon ESCI, Wayfair WANDS), measures the loss of relevance and its causes, and proposes a label-free safety score (catalog coverage times title-field agreement) that tells which attributes a catalog can support.

## What was rerun and what was not

| Step | Hardware | Status |
|---|---|---|
| Data preparation, candidate index, all analysis, tables, figures | CPU | Rerun from a fresh clone; see `results/` |
| Extraction (three models) and LoRA training | One GPU | The paper uses the original run, saved in `extractions/` (log: `provenance/gpu_run.log`). A separate re-run validated it; see below |

Every paper number depends on the saved extractions. Rows marked `G` in `results/check_numbers.txt` are read directly from a saved GPU output or log.

## Setup

Python 3.12. Tested on macOS arm64 (Apple M4 Pro, 12 cores, 24 GB) with no GPU.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

The first run downloads `BAAI/bge-small-en-v1.5` (revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`). All paths come from two variables:

```bash
export ISIR_DATA=$PWD/data      # candidates and prepared query files
export ISIR_WANDS=$PWD/WANDS    # clone of the WANDS repository
mkdir -p "$ISIR_DATA"
```

## Data

No dataset copies are in this repository. The steps below fetch them from their sources.

```bash
git clone https://github.com/wayfair/WANDS "$ISIR_WANDS"
git -C "$ISIR_WANDS" checkout 3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5
python src/prepare_data.py       # ESCI sample and query lists (downloads about 2.5 GB)
python gpu/queryner_prep.py      # QueryNER splits; needs no GPU
python src/build_index.py wands  # about 2 minutes
python src/build_index.py esci   # about 9 minutes
```

**The ESCI sample.** `prepare_data.py` takes US-locale test queries with at least 8 judged rows (which is all of them), sorts them, shuffles with `random.Random(0)` and keeps the first 4,000. It reads the Hugging Face mirror [`tasksource/esci`](https://huggingface.co/datasets/tasksource/esci) at revision `8113b17`, which is what the paper's run used. The rule selects the same 4,000 queries on the [official files](https://github.com/amazon-science/esci-data): every US test query has at least 8 judged products, so the rule removes nothing.

**A known issue in the ESCI mirror.** The mirror joins examples to products on product ID without locale, so a US example whose product also exists in the Spanish or Japanese catalog appears once per locale. In our sample this affects 752 queries and adds 1,474 of 77,461 rows (19.4 rows per query, 19.0 distinct products). The paper's numbers use the mirror. See "Official ESCI check" below for the effect.

Expected SHA-256 of the outputs:

```
esci_test_us_4000.jsonl  548255133615c468171ea3e887dbf7a381e60162b392dcac3ab6de6b56e3c8fd
esci_queries.jsonl       5cf806b49c5d2f59e288400d8924d32a5100da89fd8485bbe8786c5d51694d5d
wands_queries.jsonl      f79c46a66594f565d68309aaf2e561c55a43d0a2a8bcb5579b754013b0063988
queryner_test.jsonl      50266654a43c74e0ee628d9b67cf7d0f65789434aef44942cc69fa2e36bcd0d4
```

## Run

```bash
./run_all.sh     # about 4 minutes on the machine above
```

This writes `results/` and `figures/`, then runs `src/check_numbers.py`. That script lists every number in the paper with its line in the paper source, expected value, tolerance and PASS, FAIL or MISSING. `results/check_numbers.txt` holds the committed run. Bootstrap and sign-flip use fixed seeds in the code. `python src/check_numbers.py` alone needs only `results/`.

## Where each number comes from

| Paper item | Command (in `run_all.sh`) | Output |
|---|---|---|
| Table I (extraction quality) | `src/table1.py` | `results/table1.txt`, `.json` |
| Dataset counts, 8% overlap (316 of 4,000), attribute ratios | `src/stats.py` | `results/stats.txt`, `.json` |
| Headline nDCG, CIs, p-values, title fallback, gating, oracle; Fig. 1 | `src/depth.py EXTRACT` | `results/depth_<extractor>.txt` |
| Table IV loss avoided at cut-offs 0.4, 0.6 | `src/depth.py EXTRACT --cut C` | `results/depth_qwen3b_lora_cut<C>.txt` |
| Loss causes by type; Fig. 2 | `src/loss_decomp.py EXTRACT` | `results/loss_<extractor>.txt` |
| Table IV wording shares, lenient labels, lenient-out drops | `src/robust.py EXTRACT` | `results/robust_qwen3b_lora.txt` |
| Table III (two decimals) | `src/safety_check.py EXTRACTS` | `results/safety_check.txt` |
| 38 cells, Spearman (0.98, 0.90, 0.66, 0.82), rule counts 34/36/38 and lenient-out, Table III raw values, Fig. 3 | `src/rules.py` | `results/rules.txt`, `.json` |
| 44-64 drop, 78-98% avoided, oracle 1.5 and 2.7, 16-20 and 13-31, relaxation, shared queries | `src/grid.py` | `results/grid.txt`, `.json` |
| Learned policy across retailers | `src/policy_exp.py EXTRACT --base hybrid` | `results/policy_<extractor>.txt` |
| Table II and worked examples | `src/table2.py` | `results/table2.txt`, `.json` |
| Other examples (random draw, seed 1) | `src/examples.py` | `results/examples.txt` |
| Official ESCI check, line 55 disclosure | `./run_official.sh`, `src/esci_official_report.py` | `results/esci_official/` |
| Separate GPU re-run (not the paper's numbers) | `gpu/run_gpu.sh`, `src/gpu_rerun_compare.py` | `provenance/gpu_rerun_summary.txt`, `results/gpu_rerun/` |
| Figures 1, 2, 3 | `src/figs.py` | `figures/fig_{ndcg,loss,safety}.{pdf,png}` |

Fixed-order relaxation drops attributes until ten products survive. The relaxation p-values in the paper come from `grid.py`.

## GPU steps

```bash
uv pip install -r gpu/requirements-gpu.txt
ISIR_DATA=$PWD/data OUT=$PWD/out gpu/run_gpu.sh
```

The original run used one NVIDIA A100 80GB (RunPod image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`), took about 43 minutes (`provenance/gpu_timing.txt`) and recorded only vLLM 0.32.0 and PEFT 0.21.2. The LoRA adapter is not included.

**Separate re-run (2026-11-02), not used for any paper number.** From a clean clone, `gpu/run_gpu.sh` ran unmodified in about 36 minutes on the same GPU type and image. Extractions are not byte-identical to the original:

| | Original | Re-run |
|---|---|---|
| LoRA P / R / F1 | 0.748 / 0.714 / 0.731 | 0.740 / 0.710 / 0.725 |
| 72B P / R / F1 | 0.785 / 0.337 / 0.472 | 0.784 / 0.335 / 0.470 |
| 3B P / R / F1 | 0.592 / 0.265 / 0.366 | 0.589 / 0.265 / 0.366 |
| Same greedy attributes per query (LoRA / 72B / 3B) | | 98.47% / 99.89% / 99.42% |
| Largest change in hard-filter drop, 18 runs | | 0.75 points |
| Rule correct at 0.4 / 0.5 / 0.6 | 34 / 36 / 38 | 34 / 37 / 38 |

The rule-count change comes from 3B prompted WANDS color, whose safety sits at the cut-off (0.506). Across the re-run, loss avoided by gating moves for that cell from 78% to 94%, so that end of the paper's range depends on one borderline cell. Full comparison: `provenance/gpu_rerun_summary.txt`.

## Official ESCI check

`./run_official.sh` downloads Amazon's official parquet files (about 1.2 GB, not committed), keeps the same 4,000 queries, joins on product ID and locale, rebuilds the ESCI candidates and reruns the analysis. Not used for the paper's numbers.

- A join on product ID alone reproduces all 77,461 mirror rows in the sample and no others, so that is the cause of the repeats. The locale-aware join gives 75,987 rows, a subset of the mirror's, with identical labels.
- The mirror's Japanese brand strings for "wd" queries do not occur in the official US rows (13 rows to 0).
- Every ESCI hard-filter drop changes by at most 0.2 points (LoRA hybrid 54.1 to 54.3), and rule counts stay 34 / 36 / 38. Full comparison: `results/esci_official/report.txt`.

## Licenses

Code: MIT (`LICENSE`). Third-party data and model terms are in `NOTICE`.

| Item | License | Source | Here |
|---|---|---|---|
| ESCI | Apache-2.0 | https://github.com/amazon-science/esci-data/blob/main/LICENSE | Query strings in `extractions/` and product text in a few result files; not the dataset |
| WANDS | MIT | https://github.com/wayfair/WANDS/blob/main/LICENSE | Same |
| QueryNER | CC BY 4.0 (dataset card; the code repository is MIT) | https://huggingface.co/datasets/bltlab/queryner | Test queries and gold lists in `extractions/`, with attribution in `NOTICE` |
| Qwen2.5-3B-Instruct | Qwen Research License (non-commercial) | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE | Outputs only |
| Qwen2.5-72B-Instruct-AWQ | Qwen License | https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ/blob/main/LICENSE | Outputs only |
| bge-small-en-v1.5 | MIT | https://huggingface.co/BAAI/bge-small-en-v1.5 | Downloaded at run time |

## Citation

ISIR-eCom 2026 (workshop at IEEE BigData). Citation details will follow when available.
