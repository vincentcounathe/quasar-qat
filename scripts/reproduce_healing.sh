#!/usr/bin/env bash
# Healing table for one model: BF16 reference, RTN, and five QAT methods at INT4/3/2, each
# trained with the recipe and evaluated with the healing suite.
#
#   FAMILY=qwen scripts/reproduce_healing.sh     # Qwen3-4B-Thinking-2507
#   FAMILY=llama scripts/reproduce_healing.sh    # Llama-3.1-8B-Instruct
#
# Environment (default):
#   FAMILY      qwen | llama (qwen)
#   NPROC       GPUs per training run (8)
#   TRAIN_ARGS  extra trainer flags, e.g. on 4 GPUs "--grad_accum 2" (qwen) / "--grad_accum 16" (llama)
#               to keep the global batch
#   DATA_DIR    directory with train.jsonl and eval.jsonl (required)
#   OUT_ROOT    output directory (runs)
#   BITS        bit widths ("4 3 2");  METHODS  QAT methods ("quasar standard lsq denoising bitdistiller")
#   LCB_REPO, LCB_PYTHON  LiveCodeBench checkout and its Python, to score LiveCodeBench (optional)
# Finished runs and evaluation tasks are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

FAMILY=${FAMILY:-qwen}
NPROC=${NPROC:-8}
TRAIN_ARGS=${TRAIN_ARGS:-}
DATA_DIR=${DATA_DIR:?set DATA_DIR to a directory with train.jsonl and eval.jsonl}
OUT_ROOT=${OUT_ROOT:-runs}
BITS=${BITS:-"4 3 2"}
METHODS=${METHODS:-"quasar standard lsq denoising bitdistiller"}

case $FAMILY in
  qwen) RECIPE=configs/heal_qwen3_4b_thinking.yaml MODEL=Qwen/Qwen3-4B-Thinking-2507 ;;
  llama) RECIPE=configs/heal_llama31_8b_instruct.yaml MODEL=meta-llama/Llama-3.1-8B-Instruct ;;
  *) echo "FAMILY must be qwen or llama" >&2; exit 1 ;;
esac
ROOT=$OUT_ROOT/heal_$FAMILY

train() {  # train OUT_DIR METHOD BITS
  [[ -f $1/materialized/receipt.json ]] && return
  # shellcheck disable=SC2086  # TRAIN_ARGS is a list of flags
  torchrun --standalone --nproc_per_node "$NPROC" -m quasar.train --recipe "$RECIPE" \
    --method "$2" --bits "$3" --out_dir "$1" \
    --train_data "$DATA_DIR/train.jsonl" --eval_data "$DATA_DIR/eval.jsonl" $TRAIN_ARGS
}

evaluate() {  # evaluate MODEL OUT_DIR
  python -m quasar.eval.suite --table healing --family "$FAMILY" --model "$1" --out_dir "$2" \
    --heldout "$DATA_DIR/eval.jsonl" --ruler_data "$OUT_ROOT/ruler_$FAMILY" \
    ${LCB_REPO:+--lcb_repo "$LCB_REPO" --lcb_python "${LCB_PYTHON:-python}"}
}

evaluate "$MODEL" "$ROOT/bf16/eval"
for bits in $BITS; do
  rtn=$ROOT/w${bits}_rtn
  [[ -f $rtn/receipt.json ]] || python -m quasar.export.rtn --model "$MODEL" --bits "$bits" --out_dir "$rtn" --device cuda
  evaluate "$rtn" "$rtn/eval"
  for method in $METHODS; do
    train "$ROOT/w${bits}_$method" "$method" "$bits"
    evaluate "$ROOT/w${bits}_$method/materialized" "$ROOT/w${bits}_$method/eval"
  done
done
