#!/bin/bash
#SBATCH -J cl_eval_1_python
#SBATCH -o cl_eval_1_python-%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --mem 128G
#SBATCH --gres=gpu:a100-sxm4-80gb:2
set -o pipefail

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
fuser -k /dev/nvidia*

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=0
DS_CONFIG="./configs/ds_configs/stage2.config"

TASK_ORDER="python,cpp,swift,rust,csharp,java,php,typescript,shell"
RUN_NAME="test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0"
BASE_DIR="logs_and_outputs/${RUN_NAME}/outputs"

k_task="python"
k_num=1
LORA_WEIGHTS_DIR="${BASE_DIR}/${k_num}-${k_task}/saved_weights"
PREV_LORA_PATH=""
EVAL_TASKS=(python)

run_eval() {
    local j_task="$1"
    local eval_output_dir="${BASE_DIR}/${k_num}-${k_task}/eval_on_${j_task}"
    echo "================================================================"
    echo "Model trained through task ${k_num}-${k_task}, evaluating on: ${j_task}"
    echo "================================================================"
    local cmd=(deepspeed --num_gpus=1 src/run_qwen_new.py
        --do_predict
        --predict_with_generate
        --model_name_or_path Qwen/Qwen2.5-Coder-1.5B
        --lora_weights_dir "${LORA_WEIGHTS_DIR}"
        --data_dir Executable_Benchmark
        --task_order "${TASK_ORDER}"
        --task_config_dir "configs/Executable/${j_task}"
        --output_dir "${eval_output_dir}"
        --per_device_train_batch_size 1
        --per_device_eval_batch_size 8
        --gradient_accumulation_steps 1
        --learning_rate 1e-04
        --attn_lr 0.0
        --num_train_epochs 1
        --run_name "${RUN_NAME}"
        --distances_temperature 1.0
        --distances_way L2
        --max_source_length 1024
        --max_target_length 2048
        --generation_max_length 2048
        --add_task_name False
        --add_dataset_name False
        --overwrite_output_dir
        --overwrite_cache
        --lr_scheduler_type constant
        --warmup_steps 0
        --logging_strategy steps
        --logging_steps 50
        --evaluation_strategy no
        --save_strategy no
        --lora_r 16
        --lora_alpha 32
        --lora_dropout 0.0
        --data_replay_freq -1
        --replay_after_n_epoch 0
        --kl_ratio 1
        --attn_temperature 1
        --train_key_weight_top 1
        --test_key_weight_top 1
        --train_key_weight_top_p -1.0
        --test_key_weight_top_p -1.0
        --successor N
        --deepspeed "${DS_CONFIG}"
        --fp16
    )
    if [ -n "${PREV_LORA_PATH}" ]; then
        cmd+=(--previous_lora_path "${PREV_LORA_PATH}"
              --previous_lora_distribution_path "${PREV_LORA_PATH}")
    fi
    "${cmd[@]}" || { echo "ERROR: eval failed for ${k_num}-${k_task} on ${j_task}"; exit 1; }
}

for j_task in "${EVAL_TASKS[@]}"; do
    run_eval "${j_task}"
done

echo "Done: model ${k_num}-${k_task} evaluated on all tasks."
