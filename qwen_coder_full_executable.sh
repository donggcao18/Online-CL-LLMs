
#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
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
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
port=$(shuf -i25000-30000 -n1)
DS_CONFIG="./configs/ds_configs/stage2.config"

# # Task 1: python
# deepspeed --num_gpus=1 src/run_qwen_new.py \
#    --do_eval \
#    --predict_with_generate \
#    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
#    --data_dir Executable_Benchmark \
#    --task_order python,cpp,swift,rust,csharp,java,php,typescript,shell \
#    --task_config_dir configs/Executable/python \
#    --output_dir logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python \
#    --per_device_train_batch_size 2 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 16 \
#    --learning_rate 1e-04 \
#    --attn_lr 0.0 \
#    --num_train_epochs 3 \
#    --run_name test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
#    --distances_temperature 1.0 \
#    --distances_way L2 \
#    --max_source_length 1024 \
#    --max_target_length 2048 \
#    --generation_max_length 2048 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy steps \
#    --logging_steps 50 \
#    --evaluation_strategy no \
#    --save_strategy no \
#    --lora_r 16 \
#    --lora_alpha 32 \
#    --lora_dropout 0.0 \
#    --data_replay_freq -1 \
#    --replay_after_n_epoch 0 \
#    --kl_ratio 1 \
#    --attn_temperature 1 \
#    --train_key_weight_top 1 \
#    --test_key_weight_top 1 \
#    --train_key_weight_top_p -1.0 \
#    --test_key_weight_top_p -1.0 \
#    --successor N \
#    --deepspeed $DS_CONFIG \
#    --bf16
   


# rm -rf logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/checkpoint*

# Task 2: cpp
deepspeed --num_gpus=1 src/run_qwen_new.py \
   --do_train \
   --do_eval \
   --predict_with_generate \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --previous_lora_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights \
   --data_dir Executable_Benchmark \
   --task_order python,cpp,swift,rust,csharp,java,php,typescript,shell \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/Executable/cpp \
   --output_dir logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --per_device_train_batch_size 16 \
   --learning_rate 1e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 3 \
   --run_name test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 2048 \
   --generation_max_length 2048 \
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
   --deepspeed $DS_CONFIG \
   --bf16
   

rm -rf logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/checkpoint*


# Task 3: swift
deepspeed --num_gpus=1 src/run_qwen_new.py \
   --do_train \
   --do_eval \
   --predict_with_generate \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --previous_lora_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights \
   --data_dir Executable_Benchmark \
   --task_order python,cpp,swift,rust,csharp,java,php,typescript,shell \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/Executable/swift \
   --output_dir logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --per_device_train_batch_size 16 \
   --learning_rate 1e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 3 \
   --run_name test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 2048 \
   --generation_max_length 2048 \
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
   --deepspeed $DS_CONFIG \
   --bf16
   

rm -rf logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/checkpoint*


# Task 4: rust
deepspeed --num_gpus=1 src/run_qwen_new.py \
   --do_train \
   --do_eval \
   --predict_with_generate \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --previous_lora_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights \
   --data_dir Executable_Benchmark \
   --task_order python,cpp,swift,rust,csharp,java,php,typescript,shell \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/Executable/rust \
   --output_dir logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --per_device_train_batch_size 16 \
   --learning_rate 1e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 3 \
   --run_name test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 2048 \
   --generation_max_length 2048 \
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
   --deepspeed $DS_CONFIG \
   --bf16
   

rm -rf logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/checkpoint*



# Task 5: csharp
deepspeed --num_gpus=1 src/run_qwen_new.py \
   --do_train \
   --do_eval \
   --predict_with_generate \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --previous_lora_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights \
   --data_dir Executable_Benchmark \
   --task_order python,cpp,swift,rust,csharp,java,php,typescript,shell \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/Executable/csharp \
   --output_dir logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --per_device_train_batch_size 16 \
   --learning_rate 1e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 3 \
   --run_name test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 2048 \
   --generation_max_length 2048 \
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
   --deepspeed $DS_CONFIG \
   --bf16
   

rm -rf logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp/checkpoint*

# Task 6: java
deepspeed --num_gpus=1 src/run_qwen_new.py \
   --do_train \
   --do_eval \
   --predict_with_generate \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --previous_lora_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp/saved_weights \
   --data_dir Executable_Benchmark \
   --task_order python,cpp,swift,rust,csharp,java,php,typescript,shell \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/Executable/java \
   --output_dir logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-java \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --per_device_train_batch_size 16 \
   --learning_rate 1e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 3 \
   --run_name test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 2048 \
   --generation_max_length 2048 \
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
   --deepspeed $DS_CONFIG \
   --bf16
   

rm -rf logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-java/checkpoint*

# Task 7: php
deepspeed --num_gpus=1 src/run_qwen_new.py \
   --do_train \
   --do_eval \
   --predict_with_generate \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --previous_lora_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-java/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-java/saved_weights \
   --data_dir Executable_Benchmark \
   --task_order python,cpp,swift,rust,csharp,java,php,typescript,shell \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/Executable/php \
   --output_dir logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-php \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --per_device_train_batch_size 16 \
   --learning_rate 1e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 3 \
   --run_name test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 2048 \
   --generation_max_length 2048 \
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
   --deepspeed $DS_CONFIG \
   --bf16
   

rm -rf logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-php/checkpoint*

# Task 8: typescript
deepspeed --num_gpus=1 src/run_qwen_new.py \
   --do_train \
   --do_eval \
   --predict_with_generate \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --previous_lora_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-java/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-php/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-java/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-php/saved_weights \
   --data_dir Executable_Benchmark \
   --task_order python,cpp,swift,rust,csharp,java,php,typescript,shell \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/Executable/typescript \
   --output_dir logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-typescript \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --per_device_train_batch_size 16 \
   --learning_rate 1e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 3 \
   --run_name test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 2048 \
   --generation_max_length 2048 \
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
   --deepspeed $DS_CONFIG \
   --bf16
   

rm -rf logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-typescript/checkpoint*

# Task 9: shell
deepspeed --num_gpus=1 src/run_qwen_new.py \
   --do_train \
   --do_eval \
   --predict_with_generate \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --previous_lora_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-java/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-php/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-typescript/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-python/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cpp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-swift/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-rust/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-csharp/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-java/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-php/saved_weights,logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-typescript/saved_weights \
   --data_dir Executable_Benchmark \
   --task_order python,cpp,swift,rust,csharp,java,php,typescript,shell \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/Executable/shell \
   --output_dir logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-shell \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 4 \
   --per_device_train_batch_size 16 \
   --learning_rate 1e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 3 \
   --run_name test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 2048 \
   --generation_max_length 2048 \
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
   --deepspeed $DS_CONFIG \
   --bf16
   

rm -rf logs_and_outputs/test_qwen_executable_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-shell/checkpoint*

