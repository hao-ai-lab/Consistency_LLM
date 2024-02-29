export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=consistency_llm

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint='localhost:5667' \
    --master_port 10005 \
    train_cllm_global.py \
    --target_model_path your_target_model_path \
    --data_path ../data/collected_jacobi_trajectory/your_jacobi_trajectory_name \
    --output_dir your_local_path_to_save_cllm \
    --max_new_tokens n_token_seq_size \
    --bf16 True \
    --tf32 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
