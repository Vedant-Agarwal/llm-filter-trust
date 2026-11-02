#!/bin/bash
# Official ESCI check (separate from the paper's numbers). Run from the repository root after the README's data steps and run_all.sh.
# Downloads about 1.2 GB into $ISIR_OFFICIAL; nothing from it is committed. About 12 minutes on CPU.
set -euo pipefail
export ISIR_DATA ISIR_WANDS PYTHONPATH=src ISIR_OFFICIAL=${ISIR_OFFICIAL:-official}
OUT=results/esci_official; X=extractions/extract_; mkdir -p "$ISIR_OFFICIAL" $OUT logs
for f in examples products; do
  [ -s "$ISIR_OFFICIAL/shopping_queries_dataset_$f.parquet" ] || curl -sSL -o "$ISIR_OFFICIAL/shopping_queries_dataset_$f.parquet" "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_$f.parquet"
done
ISIR_RESULTS=$OUT python src/esci_official.py > $OUT/join_check.txt 2>>logs/stderr.log
ln -sf "$ISIR_DATA/wands_cands.pkl" "$ISIR_DATA/official/wands_cands.pkl"
O="$ISIR_DATA/official"
ISIR_DATA="$O" python src/build_index.py esci > /dev/null 2>>logs/stderr.log
ISIR_DATA="$O" ISIR_RESULTS=$OUT python src/rules.py > $OUT/rules.txt 2>>logs/stderr.log
ISIR_DATA="$O" ISIR_RESULTS=$OUT python src/grid.py > $OUT/grid.txt 2>>logs/stderr.log
ISIR_DATA="$O" python src/loss_decomp.py ${X}qwen3b_lora.jsonl > $OUT/loss_qwen3b_lora.txt 2>>logs/stderr.log
for e in qwen3b_lora qwen72 qwen3b_base; do ISIR_DATA="$O" python src/depth.py ${X}$e.jsonl > $OUT/depth_$e.txt 2>>logs/stderr.log; done
ISIR_DATA="$O" python src/robust.py ${X}qwen3b_lora.jsonl > $OUT/robust_qwen3b_lora.txt 2>>logs/stderr.log
python src/esci_official_report.py > $OUT/report.txt
python src/check_numbers.py > results/check_numbers.txt || true
grep -E "^(TOTAL|FAIL|MISSING)" results/check_numbers.txt
echo official done
