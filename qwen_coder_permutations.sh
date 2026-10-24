#!/bin/bash
#SBATCH -J cl-qwen-permutations
#SBATCH -o cl-qwen-permutation-%A_%a.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --mem 128G
#SBATCH --gres=gpu:a100-sxm4-80gb:2
#SBATCH --array=1-5

set -euo pipefail

# `sbatch qwen_coder_permutations.sh` runs all five permutations as an array.
# For a single local run, use: `bash qwen_coder_permutations.sh <1-5>`.
PERMUTATION_ID="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ ! "$PERMUTATION_ID" =~ ^[1-5]$ ]]; then
    echo "Usage: bash $0 <permutation-id: 1-5>" >&2
    echo "       sbatch $0  # submits the five-permutation SLURM array" >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-./.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./.cache}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-1.5B}"
NUM_GPUS="${NUM_GPUS:-2}"
DS_CONFIG="${DS_CONFIG:-./configs/ds_configs/stage2.config}"
DATA_DIR="${DATA_DIR:-CODETASK_Benchmark}"
GEN_DATA_DIR="${GEN_DATA_DIR:-generated_data/lora_gen_superni_llama}"
OUTPUT_BASE="${OUTPUT_BASE:-logs_and_outputs}"

# Keep the common part of the output/LoRA path in one place. The permutation
# suffix prevents array jobs from reading or overwriting each other's adapters.
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-test_qwen_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0}"
RUN_NAME="${RUN_NAME_PREFIX}_permutation_${PERMUTATION_ID}"
OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_NAME}/outputs"

# IDs from the experiment design:
# A=CONCODE, B=CodeTrans, C=CodeSearchNet, D=BFP, E=KodCode,
# F=RunBugRun, G=TheVault_Csharp, H=CoST.
case "$PERMUTATION_ID" in
    1) TASKS=(CONCODE CodeTrans CoST CodeSearchNet TheVault_Csharp BFP RunBugRun KodCode) ;;
    2) TASKS=(CodeSearchNet BFP CodeTrans KodCode CONCODE RunBugRun CoST TheVault_Csharp) ;;
    3) TASKS=(KodCode RunBugRun BFP TheVault_Csharp CodeSearchNet CoST CodeTrans CONCODE) ;;
    4) TASKS=(TheVault_Csharp CoST RunBugRun CONCODE KodCode CodeTrans BFP CodeSearchNet) ;;
    5) TASKS=(CoST CONCODE TheVault_Csharp CodeTrans RunBugRun CodeSearchNet KodCode BFP) ;;
esac

printf -v TASK_ORDER '%s,' "${TASKS[@]}"
TASK_ORDER="${TASK_ORDER%,}"

# Some task configs are generated (and ignored by git) in this repository.
# Serialize generation because the five SLURM array jobs may start together.
task_configs_ready() {
    local task split
    for task in "${TASKS[@]}"; do
        for split in train dev test; do
            [[ -f "configs/CodeTask/${task}/${split}_tasks.json" ]] || return 1
        done
    done
}

generate_task_configs() {
    if task_configs_ready; then
        return
    fi

    if command -v flock >/dev/null 2>&1; then
        (
            flock -x 9
            task_configs_ready || python configs/CodeTask/create_codetask_config.py
        ) 9< configs/CodeTask/create_codetask_config.py
        return
    fi

    # Portable fallback for systems without util-linux/flock.
    local lock_dir="configs/CodeTask/.config_generation.lock"
    until mkdir "$lock_dir" 2>/dev/null; do
        task_configs_ready && return
        sleep 1
    done
    trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT INT TERM
    task_configs_ready || python configs/CodeTask/create_codetask_config.py
    rmdir "$lock_dir"
    trap - EXIT INT TERM
}

generate_task_configs

echo "Permutation ${PERMUTATION_ID}: ${TASK_ORDER}"
echo "Outputs: ${OUTPUT_ROOT}"

PREVIOUS_LORAS=()

for TASK_INDEX in "${!TASKS[@]}"; do
    STEP=$((TASK_INDEX + 1))
    TASK="${TASKS[$TASK_INDEX]}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${STEP}-${TASK}"

    # Preserve the per-task settings from qwen_coder_full.sh regardless of
    # where the task occurs in a permutation.
    case "$TASK" in
        CONCODE)
            TRAIN_BATCH=16; EVAL_BATCH=8; GRAD_ACCUM=1
            MAX_SOURCE=320; MAX_TARGET=150
            ;;
        CodeTrans)
            TRAIN_BATCH=16; EVAL_BATCH=16; GRAD_ACCUM=1
            MAX_SOURCE=320; MAX_TARGET=256
            ;;
        CodeSearchNet|RunBugRun|TheVault_Csharp|CoST)
            TRAIN_BATCH=16; EVAL_BATCH=16; GRAD_ACCUM=1
            MAX_SOURCE=256; MAX_TARGET=128
            ;;
        BFP)
            TRAIN_BATCH=16; EVAL_BATCH=16; GRAD_ACCUM=1
            MAX_SOURCE=130; MAX_TARGET=120
            ;;
        KodCode)
            TRAIN_BATCH=8; EVAL_BATCH=8; GRAD_ACCUM=2
            MAX_SOURCE=512; MAX_TARGET=300
            ;;
        *)
            echo "Unknown task: ${TASK}" >&2
            exit 3
            ;;
    esac

    PREVIOUS_LORA_ARGS=()
    if ((${#PREVIOUS_LORAS[@]} > 0)); then
        printf -v PREVIOUS_LORA_PATH '%s,' "${PREVIOUS_LORAS[@]}"
        PREVIOUS_LORA_PATH="${PREVIOUS_LORA_PATH%,}"
        PREVIOUS_LORA_ARGS=(
            --previous_lora_path "$PREVIOUS_LORA_PATH"
            --previous_lora_distribution_path "$PREVIOUS_LORA_PATH"
        )
    fi

    PREDICT_ARGS=()
    if [[ "$TASK" == "BFP" || "$TASK" == "CoST" ]]; then
        PREDICT_ARGS=(--do_predict)
    fi

    echo "[$STEP/${#TASKS[@]}] Training ${TASK}"
    if ((${#PREVIOUS_LORAS[@]} > 0)); then
        echo "Using ${#PREVIOUS_LORAS[@]} previous LoRA adapter(s)"
    fi

    deepspeed --num_gpus="$NUM_GPUS" src/run_qwen_new.py \
        --do_train \
        --do_eval \
        "${PREDICT_ARGS[@]}" \
        --predict_with_generate \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        "${PREVIOUS_LORA_ARGS[@]}" \
        --data_dir "$DATA_DIR" \
        --task_order "$TASK_ORDER" \
        --gen_data_dir "$GEN_DATA_DIR" \
        --task_config_dir "configs/CodeTask/${TASK}" \
        --output_dir "$OUTPUT_DIR" \
        --per_device_train_batch_size "$TRAIN_BATCH" \
        --per_device_eval_batch_size "$EVAL_BATCH" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate 1e-04 \
        --attn_lr 0.0 \
        --num_train_epochs 3 \
        --bf16 \
        --run_name "$RUN_NAME" \
        --distances_temperature 1.0 \
        --distances_way L2 \
        --max_source_length "$MAX_SOURCE" \
        --max_target_length "$MAX_TARGET" \
        --generation_max_length "$MAX_TARGET" \
        --add_task_name False \
        --add_dataset_name False \
        --overwrite_output_dir \
        --overwrite_cache \
        --lr_scheduler_type constant \
        --warmup_steps 0 \
        --logging_strategy steps \
        --logging_steps 50 \
        --evaluation_strategy no \
        --save_strategy no \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.0 \
        --data_replay_freq -1 \
        --replay_after_n_epoch 0 \
        --kl_ratio 1 \
        --attn_temperature 1 \
        --train_key_weight_top 1 \
        --test_key_weight_top 1 \
        --train_key_weight_top_p -1.0 \
        --test_key_weight_top_p -1.0 \
        --successor N \
        --deepspeed "$DS_CONFIG" \
        --num_eval 200

    # The next task needs every earlier adapter, in chronological order.
    PREVIOUS_LORAS+=("${OUTPUT_DIR}/saved_weights")

    # Match the old script's cleanup without repeating or globbing a long root.
    shopt -s nullglob
    CHECKPOINTS=("${OUTPUT_DIR}"/checkpoint*)
    if ((${#CHECKPOINTS[@]} > 0)); then
        rm -rf -- "${CHECKPOINTS[@]}"
    fi
    shopt -u nullglob
done

echo "Permutation ${PERMUTATION_ID} finished successfully."
