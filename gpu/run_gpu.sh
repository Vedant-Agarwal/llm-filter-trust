#!/bin/bash
# Extraction and LoRA training. Needs one GPU (paper: A100 80GB, under one hour). Not rerun for the check; see README.
# Usage: ISIR_DATA=/path/to/data OUT=/path/to/out gpu/run_gpu.sh   (ISIR_DATA must already hold *_queries.jsonl)
set -ex
: "${ISIR_DATA:?set ISIR_DATA}"; OUT=${OUT:-out}; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
export HF_HUB_DISABLE_XET=1 ISIR_DATA
cd "$HERE"
python -c "import vllm, peft; print('vllm', vllm.__version__, 'peft', peft.__version__)"
python queryner_prep.py
Q="$ISIR_DATA/wands_queries.jsonl $ISIR_DATA/esci_queries.jsonl $ISIR_DATA/queryner_test.jsonl"
python extract.py Qwen/Qwen2.5-72B-Instruct-AWQ "$OUT/extract_qwen72.jsonl" $Q 2>&1 | grep -E "^DONE|Traceback|Error"
python lora_train.py "$OUT/lora_qwen3b_queryner" 2>&1 | grep -E "step .*0/|^DONE|Traceback|Error"
python extract.py Qwen/Qwen2.5-3B-Instruct "$OUT/extract_qwen3b_lora.jsonl" --lora "$OUT/lora_qwen3b_queryner" $Q 2>&1 | grep -E "^DONE|Traceback|Error"
python extract.py Qwen/Qwen2.5-3B-Instruct "$OUT/extract_qwen3b_base.jsonl" $Q 2>&1 | grep -E "^DONE|Traceback|Error"
