#!/usr/bin/env bash
# Adaptation table: Qwen3-4B-Base fine-tuned on OpenMathReasoning in full precision (FP-SFT),
# RTN of the FP-SFT model, and five QAT methods at INT4/3/2, each evaluated with the adaptation
# suite (held-out perplexity and five math benchmarks).
#
#   scripts/reproduce_adaptation.sh
#
# Environment (default):
#   NPROC       GPUs per training run (8)
#   TRAIN_ARGS  extra trainer flags, e.g. "--grad_accum 2" to keep the global batch on 4 GPUs
#   DATA_DIR    directory with train.jsonl and eval.jsonl (required)
#   OUT_ROOT    output directory (runs)
#   BITS        bit widths ("4 3 2");  METHODS  QAT methods ("quasar standard lsq denoising bitdistiller")
# Finished runs and evaluation tasks are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC=${NPROC:-8}
TRAIN_ARGS=${TRAIN_ARGS:-}
DATA_DIR=${DATA_DIR:?set DATA_DIR to a directory with train.jsonl and eval.jsonl}
OUT_ROOT=${OUT_ROOT:-runs}
BITS=${BITS:-"4 3 2"}
METHODS=${METHODS:-"quasar standard lsq denoising bitdistiller"}

RECIPE=configs/adapt_qwen3_4b_base_omr.yaml
ROOT=$OUT_ROOT/adapt_qwen

train() {  # train OUT_DIR METHOD BITS
  [[ -f $1/materialized/receipt.json ]] && return
  # shellcheck disable=SC2086  # TRAIN_ARGS is a list of flags
  torchrun --standalone --nproc_per_node "$NPROC" -m quasar.train --recipe "$RECIPE" \
    --method "$2" --bits "$3" --out_dir "$1" \
    --train_data "$DATA_DIR/train.jsonl" --eval_data "$DATA_DIR/eval.jsonl" $TRAIN_ARGS
}

evaluate() {  # evaluate MODEL OUT_DIR
  python -m quasar.eval.suite --table adaptation --family qwen --model "$1" --out_dir "$2" --heldout "$DATA_DIR/eval.jsonl"
}

train "$ROOT/fp_sft" none 4
evaluate "$ROOT/fp_sft/materialized" "$ROOT/fp_sft/eval"
for bits in $BITS; do
  rtn=$ROOT/w${bits}_rtn
  [[ -f $rtn/receipt.json ]] || python -m quasar.export.rtn --model "$ROOT/fp_sft/materialized" --bits "$bits" \
    --out_dir "$rtn" --device cuda
  evaluate "$rtn" "$rtn/eval"
  for method in $METHODS; do
    train "$ROOT/w${bits}_$method" "$method" "$bits"
    evaluate "$ROOT/w${bits}_$method/materialized" "$ROOT/w${bits}_$method/eval"
  done
done
