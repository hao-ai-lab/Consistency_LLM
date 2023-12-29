datapath=$1
dataset_name=$2
sample=$3
kl=$4
max_new_tokens=$5
consistency_loss=$6

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export WANDB_PROJECT=consistency_llm

torchrun --nnodes=1 --nproc_per_node=5 --rdzv_id=101 --rdzv_endpoint='localhost:5667' \
    --master_port 1000 \
    cllm/train_fsdp_dataset.py \
    --student_model_path $datapath/deepseek-coder-6.7b-instruct \
    --teacher_model_path $datapath/deepseek-coder-6.7b-instruct \
    --data_path /liymai24/sjtu/siqi/Consistency_LLM-master/data/raw_data/spider_jacobian16_augTrue_labels_True_max_seq_len_256_part1.json \
    --max_new_tokens $max_new_tokens \
    --max_new_seq_len 256 \
    --bf16 True \
    --tf32 True \
    --output_dir $datapath/ar_labels_weight20_coder_${dataset_name}_aug_${max_new_tokens}/$consistency_loss \
    --logging_dir /liymai24/sjtu/siqi/experiments/training_logs/ar_labels_weight20_coder_${dataset_name}_aug_${max_new_tokens}/$consistency_loss \
    --logging_steps 1 \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 256 \
    --lazy_preprocess True \
    --mode offline \
    --sample_source $sample \
    --kl_method $kl \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --consistency_loss $consistency_loss \

# bash ./offline_fsdp_dataset.sh /liymai24/sjtu/siqi/llm-model spider teacher forward 16 global

# torchrun --nnodes=1 --nproc_per_node=5 --rdzv_id=101 --rdzv_endpoint='localhost:5668' \
#     --master_port 1002 \
#     cllm/train_fsdp_dataset.py \
#     --student_model_path $datapath/vicuna-7b-v1.5 \
#     --teacher_model_path $datapath/vicuna-7b-v1.5 \
#     --data_path /liymai24/sjtu/siqi/wizard_dataset/WizardLM_evol_instruct_V2_143k_jacobian16_augTrue_labels_True_max_seq_len_256.json \
#     --max_new_tokens $max_new_tokens \
#     --max_new_seq_len 256 \
#     --bf16 True \
#     --tf32 True \
#     --output_dir $datapath/new_ar_labels_weight20_vicuna_${dataset_name}_aug_${max_new_tokens}/$consistency_loss \
#     --logging_dir /liymai24/sjtu/siqi/experiments/training_logs/new_ar_labels_weight20_vicuna_${dataset_name}_aug_${max_new_tokens}/$consistency_loss \
#     --logging_steps 1 \
#     --do_train \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "epoch" \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 100 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --model_max_length 256 \
#     --lazy_preprocess True \
#     --mode offline \
#     --sample_source $sample \
#     --kl_method $kl \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --consistency_loss $consistency_loss \

# bash ./offline_fsdp_dataset.sh /liymai24/sjtu/siqi/llm-model wizard teacher forward 16 global
