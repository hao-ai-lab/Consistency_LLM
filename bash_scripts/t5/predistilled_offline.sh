datapath=$1
dataset_name=$2
sample_source=$3
kl=$4

export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT=consistency_llm

python cllm/train_mlm.py \
    --student_model_path models/t5-small-spider-teacher-forward  \
    --teacher_model_path google/flan-t5-xl \
    --dataset_name ${dataset_name} \
    --mode offline \
    --sample_source ${sample_source} \
    --kl_method ${kl} \
    --do_train \
    --do_eval \
    --fast_eval \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --source_max_length 1024 \
    --train_target_max_length 512 \
    --val_target_max_length 512 \
    --test_target_max_length 512 \
    --run_name t5_${dataset_name}_consistency_distill_${sample_source}_${kl} \
    --output_dir $datapath/distilled_t5_${dataset_name}_consistency_${sample_source}_${kl}
