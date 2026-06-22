#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_FILE="${DATA_FILE:-data/math_step_dpo_train.parquet}"

export HF_HUB_DISABLE_XET=1

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Data file not found: $DATA_FILE"
  echo "Download it first or set DATA_FILE=/path/to/train.parquet"
  exit 1
fi

run_sft() {
  "$PYTHON_BIN" -m src.train_sft \
    --model "$MODEL" \
    --data_file "$DATA_FILE" \
    --max_length "${SFT_MAX_LENGTH:-1024}" \
    --batch_size "${SFT_BATCH_SIZE:-16}" \
    --gradient_accumulation_steps "${SFT_GRAD_ACCUM:-1}" \
    --epochs "${SFT_EPOCHS:-2}" \
    --rank "${LORA_RANK:-32}" \
    --alpha "${LORA_ALPHA:-64}" \
    --lr "${SFT_LR:-2e-5}" \
    --log_dir "${SFT_LOG_DIR:-runs/sft}"
}

run_dpo() {
  "$PYTHON_BIN" -m src.train_dpo \
    --model "$MODEL" \
    --init_adapter_dir "${INIT_ADAPTER_DIR:-outputs/sft_lora}" \
    --output_dir "${DPO_OUTPUT_DIR:-outputs/dpo_lora}" \
    --data_file "$DATA_FILE" \
    --max_length "${DPO_MAX_LENGTH:-1024}" \
    --batch_size "${DPO_BATCH_SIZE:-8}" \
    --gradient_accumulation_steps "${DPO_GRAD_ACCUM:-1}" \
    --epochs "${DPO_EPOCHS:-2}" \
    --rank "${LORA_RANK:-32}" \
    --alpha "${LORA_ALPHA:-64}" \
    --lr "${DPO_LR:-1e-5}" \
    --beta "${DPO_BETA:-0.1}" \
    --log_dir "${DPO_LOG_DIR:-runs/dpo}"
}

case "$MODE" in
  sft)
    run_sft
    ;;
  dpo)
    run_dpo
    ;;
  all)
    run_sft
    run_dpo
    ;;
  *)
    echo "Usage: $0 [sft|dpo|all]"
    exit 1
    ;;
esac
