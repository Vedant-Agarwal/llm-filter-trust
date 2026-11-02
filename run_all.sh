#!/bin/bash
# Everything that runs on CPU from the candidate pickles and the saved extractions. Run from the repository root.
# Expects ISIR_DATA to hold *_cands.pkl and queryner_{train,test}.jsonl.
set -euo pipefail
export ISIR_DATA=${ISIR_DATA:-data} ISIR_WANDS=${ISIR_WANDS:-WANDS} PYTHONPATH=src
mkdir -p results figures logs
X=extractions/extract_
py() { python "$@" 2>>logs/stderr.log; }
py src/table1.py                      > results/table1.txt
py src/stats.py                       > results/stats.txt
for e in qwen3b_lora qwen72 qwen3b_base; do
  py src/depth.py ${X}$e.jsonl        > results/depth_$e.txt
  py src/loss_decomp.py ${X}$e.jsonl  > results/loss_$e.txt
  py src/policy_exp.py ${X}$e.jsonl --base hybrid > results/policy_$e.txt
done
for c in 0.4 0.6; do py src/depth.py ${X}qwen3b_lora.jsonl --cut $c > results/depth_qwen3b_lora_cut$c.txt; done
py src/robust.py ${X}qwen3b_lora.jsonl > results/robust_qwen3b_lora.txt
py src/safety_check.py ${X}qwen72.jsonl ${X}qwen3b_lora.jsonl ${X}qwen3b_base.jsonl > results/safety_check.txt
py src/rules.py                       > results/rules.txt
py src/grid.py                        > results/grid.txt
py src/table2.py                      > results/table2.txt
py src/examples.py                    > results/examples.txt
py src/figs.py                        > results/figs.txt
python src/check_numbers.py > results/check_numbers.txt || true
grep -E "^(TOTAL|FAIL|MISSING)" results/check_numbers.txt
