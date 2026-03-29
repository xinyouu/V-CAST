#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME=${HF_HOME:-"/data/250010029/Datasets"}

pretrained_model="qwen3_vl"
pretrained="${pretrained:-Qwen/Qwen3-VL-8B-Instruct}"
method="v_cast"

TASK_NAMES=("mlvu_dev" "mvbench" "videomme" "egoschema" "longvideobench_val_v")

model_args="pretrained=${pretrained},max_num_frames=64,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False"

CUDA_DEVICES="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8

BASE_LOG_DIR="/data/250010029/Code/logs/${pretrained_model}/64/${method}"
mkdir -p "${BASE_LOG_DIR}"

for task_name in "${TASK_NAMES[@]}"; do
  ts=$(date +"%m-%d-%H-%M-%S")
  TAG="rr0.25_default"
  JOB_NAME="${pretrained_model}_${method}_${task_name}_${TAG}"
  output_path="${BASE_LOG_DIR}/${ts}_${JOB_NAME}"
  log_file="${BASE_LOG_DIR}/${ts}_${JOB_NAME}.log"

  echo "========================================"
  echo "Running evaluation:"
  echo "  Task: ${task_name}"
  echo "  Model: ${pretrained}"
  echo "  Method: ${method}"
  echo "  Output: ${output_path}"
  echo "  Log: ${log_file}"
  echo "========================================"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  method="${method}" \
  accelerate launch \
    --main_process_port $((30000 + RANDOM % 10000)) \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model "${pretrained_model}" \
    --model_args "${model_args}" \
    --tasks "${task_name}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path "${output_path}" 2>&1 | tee "${log_file}"

  echo "Completed evaluation for task: ${task_name}"
  echo "----------------------------------------"
done
