
#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
#SBATCH -p compute 
#SBATCH -N 1                           
#SBATCH -t 20:00:00   
#SBATCH --mem 128G 
#SBATCH --gres=gpu:a100-sxm4-80gb:2
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
fuser -k /dev/nvidia*

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
port=$(shuf -i25000-30000 -n1)
DS_CONFIG="configs/ds_configs/stage2_without_offload.config"


python src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-770m \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
   --task_config_dir configs/CodeTask/CONCODE \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 320 \
   --max_target_length 150 \
   --generation_max_length 150 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 100 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 1 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N \
   --bf16 \
   # --deepspeed $DS_CONFIG \

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/checkpoint*


python src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-770m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/CodeTrans \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 320 \
   --max_target_length 256 \
   --generation_max_length 256 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_CodeTrans \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
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
   # --deepspeed $DS_CONFIG \

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/checkpoint*



python src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-770m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/CodeSearchNet \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 256 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_CodeSearchNet \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
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
   # --deepspeed $DS_CONFIG \

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/checkpoint*



python src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-770m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/BFP \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 130 \
   --max_target_length 120 \
   --generation_max_length 120 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_BFP \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
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
      # --deepspeed $DS_CONFIG 

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/checkpoint*




python src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-770m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/KodCode \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-KodCode \
   --per_device_eval_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 130 \
   --max_target_length 120 \
   --generation_max_length 120 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_KodCode \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
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
      # --deepspeed $DS_CONFIG 

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-KodCode/checkpoint*

python src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-770m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-KodCode/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-KodCode/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/RunBugRun \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-RunBugRun \
   --per_device_eval_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 130 \
   --max_target_length 120 \
   --generation_max_length 120 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_RunBugRun \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
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
      # --deepspeed $DS_CONFIG 

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-RunBugRun/checkpoint*

python src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-770m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-KodCode/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-RunBugRun/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-KodCode/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-RunBugRun/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/TheVault_Csharp \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-TheVault_Csharp \
   --per_device_eval_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 130 \
   --max_target_length 120 \
   --generation_max_length 120 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_TheVault_Csharp \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
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
      # --deepspeed $DS_CONFIG 

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-TheVault_Csharp/checkpoint*

python src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-770m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-KodCode/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-RunBugRun/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-TheVault_Csharp/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-KodCode/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-RunBugRun/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-TheVault_Csharp/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/CoST \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-CoST \
   --per_device_eval_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 130 \
   --max_target_length 120 \
   --generation_max_length 120 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_CoST \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
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
      # --deepspeed $DS_CONFIG 

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-CoST/checkpoint*



# python3 src/run_t5_new_eval.py \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path Salesforce/codet5p-770m \
#    --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights \
#    --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights \
#    --data_dir CODETASK_Benchmark \
#    --task_order CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
#    --gen_data_dir generated_data/lora_gen_superni_llama \
#    --task_config_dir configs/CodeTask/BFP \
#    --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/15-task875_emotion_classification \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate 3e-04 \
#    --attn_lr 0.0 \
#    --num_train_epochs 5 \
#    --bf16 \
#    --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
#    --distances_temperature 1.0 \
#    --distances_way L2 \
#    --max_source_length 1024 \
#    --max_target_length 256 \
#    --generation_max_length 256 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy steps \
#    --logging_steps 10 \
#    --metric_for_best_model eval_rougeL_for_BFP \
#    --evaluation_strategy steps \
#    --save_strategy steps \
#    --save_total_limit 1 \
#    --load_best_model_at_end \
#    --lora_r 8 \
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
#    --successor N
