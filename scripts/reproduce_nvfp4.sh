#!/usr/bin/env bash
# NVFP4 rows for Qwen3-8B: BF16 reference, RTN, Standard QAT and QUASAR. Each quantized model is
# packed into a compressed-tensors NVFP4 (W4A16) checkpoint; KL / top-1 against the BF16 model is
# measured on held-out real conversations, and the packed checkpoint is evaluated in vLLM on
# HMMT'26, AIME'25 and SuperGPQA (settings: TABLES["nvfp4"] in quasar/eval/settings.py).
#
#   scripts/reproduce_nvfp4.sh
#
# Environment (default):
#   NPROC       GPUs per training run (8)
#   TRAIN_ARGS  extra trainer flags, e.g. "--grad_accum 8" to keep the global batch on 4 GPUs
#   DATA_DIR    directory with train.jsonl and eval.jsonl (required)
#   OUT_ROOT    output directory (runs)
#   METHODS     QAT methods ("quasar standard")
# Finished steps are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC=${NPROC:-8}
TRAIN_ARGS=${TRAIN_ARGS:-}
DATA_DIR=${DATA_DIR:?set DATA_DIR to a directory with train.jsonl and eval.jsonl}
OUT_ROOT=${OUT_ROOT:-runs}
METHODS=${METHODS:-"quasar standard"}

MODEL=Qwen/Qwen3-8B
RECIPE=configs/qad_nvfp4_qwen3_8b.yaml
ROOT=$OUT_ROOT/nvfp4_qwen3_8b

evaluate() {  # evaluate WEIGHTS SERVED OUT_DIR: KL / top-1 on the bf16 weights, generation on the served model
  python -m quasar.eval.suite --table nvfp4 --family qwen3_8b --model "$1" --out_dir "$3" --tasks heldout \
    --heldout "$DATA_DIR/eval.jsonl"
  python -m quasar.eval.suite --table nvfp4 --family qwen3_8b --model "$2" --out_dir "$3"
}

pack() {  # pack RUN_DIR: RUN_DIR/materialized -> RUN_DIR/w4a16
  [[ -f $1/w4a16/receipt.json ]] || python -m quasar.export.nvfp4 --materialized "$1/materialized" --out_dir "$1/w4a16"
}

evaluate "$MODEL" "$MODEL" "$ROOT/bf16/eval"

[[ -f $ROOT/rtn/materialized/receipt.json ]] || python -m quasar.export.rtn --model "$MODEL" --format nvfp4 \
  --out_dir "$ROOT/rtn/materialized" --device cuda
pack "$ROOT/rtn"
evaluate "$ROOT/rtn/materialized" "$ROOT/rtn/w4a16" "$ROOT/rtn/eval"

for method in $METHODS; do
  run=$ROOT/$method
  if [[ ! -f $run/materialized/receipt.json ]]; then
    # shellcheck disable=SC2086  # TRAIN_ARGS is a list of flags
    torchrun --standalone --nproc_per_node "$NPROC" -m quasar.train --recipe "$RECIPE" --method "$method" \
      --out_dir "$run" --train_data "$DATA_DIR/train.jsonl" --eval_data "$DATA_DIR/eval.jsonl" $TRAIN_ARGS
  fi
  pack "$run"
  evaluate "$run/materialized" "$run/w4a16" "$run/eval"
done
