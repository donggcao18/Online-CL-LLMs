
#!/bin/bash
#SBATCH -J cl_eval
#SBATCH -o cl_eval-%j.out
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

TASKS=(python cpp swift rust csharp java php typescript shell)
TASK_ORDER="python,cpp,swift,rust,csharp,java,php,typescript,shell"
RUN_NAME="test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0"
BASE_DIR="logs_and_outputs/${RUN_NAME}/outputs"

# For each training checkpoint (after training on task k), evaluate on all
# previously seen tasks (tasks 1..k).
for k_idx in $(seq 0 8); do
    k_task="${TASKS[$k_idx]}"
    k_num=$((k_idx + 1))
    lora_weights_dir="${BASE_DIR}/${k_num}-${k_task}/saved_weights"

    # Skip if this task's trained weights don't exist yet
    if [ ! -f "${lora_weights_dir}/lora_weights_A.pt" ]; then
        echo "Skipping task ${k_num}-${k_task}: no trained weights found at ${lora_weights_dir}"
        continue
    fi

    # Build comma-separated previous_lora_path for tasks 1..k-1
    prev_lora_path=""
    for j in $(seq 0 $((k_idx - 1))); do
        j_task="${TASKS[$j]}"
        j_num=$((j + 1))
        path="${BASE_DIR}/${j_num}-${j_task}/saved_weights"
        if [ -z "$prev_lora_path" ]; then
            prev_lora_path="${path}"
        else
            prev_lora_path="${prev_lora_path},${path}"
        fi
    done

    # Evaluate the task-k model on each of the tasks seen so far (1..k)
    for j_idx in $(seq 0 $k_idx); do
        j_task="${TASKS[$j_idx]}"
        eval_output_dir="${BASE_DIR}/${k_num}-${k_task}/eval_on_${j_task}"

        echo "================================================================"
        echo "Model trained through task ${k_num}-${k_task}, evaluating on: ${j_task}"
        echo "================================================================"

        # Base command
        cmd=(deepspeed --num_gpus=1 src/run_qwen_new.py
            --do_predict
            --predict_with_generate
            --model_name_or_path Qwen/Qwen2.5-Coder-1.5B
            --lora_weights_dir "${lora_weights_dir}"
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
            --bf16
        )

        # Add previous LoRA paths only when they exist (task 1 has none)
        if [ -n "$prev_lora_path" ]; then
            cmd+=(--previous_lora_path "${prev_lora_path}"
                  --previous_lora_distribution_path "${prev_lora_path}")
        fi

        "${cmd[@]}"

        if [ $? -ne 0 ]; then
            echo "ERROR: evaluation failed for model ${k_num}-${k_task} on task ${j_task}"
            exit 1
        fi
    done
done

echo "================================================================"
echo "All evaluations complete."
echo "Results are in: ${BASE_DIR}/<k>-<task>/eval_on_<task>/predict_eval_predictions.jsonl"
echo "================================================================"
