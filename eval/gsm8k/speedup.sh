model_path=$1
target_model_path=$2
max_new_tokens=$3

# test model is tested and we use the tokenizer of teacher model because the tokenizer of test model has something to fix
python3 eval/gsm8k/speedup.py \
    --test_model_path ${model_path} \
    --teacher_model_path ${target_model_path} \
    --max_new_tokens ${max_new_tokens}