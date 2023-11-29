datapath=$1
dataset_name=$2
sample=$3
kl=$4

export CUDA_VISIBLE_DEVICES=1,2,3,4
export WANDB_PROJECT=consistency_llm

# data/raw_data/spider_train_with_answer.json

torchrun --nproc_per_node=4 cllm/train.py \
    --student_model_path JackFram/llama-160m \
    --teacher_model_path lmsys/vicuna-7b-v1.3 \
    --data_path ${dataset_name} \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir $datapath/spider_${sample}_${kl} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 256 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name llama_spider_consistency_distill_${sample}_${kl} \
    --mode offline \
    --sample_source $sample \
    --kl_method $kl
