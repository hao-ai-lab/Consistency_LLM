model_path=$1
target_model_path=$2
test_file_path=$3
max_new_tokens=$4

# test model is tested and we use the tokenizer of teacher model because the tokenizer of test model has something to fix
python3 eval/gsm8k/speedup.py \
    --filename ${test_file_path} \
    --test_model_path ${model_path} \
    --teacher_model_path ${target_model_path} \
    --max_new_tokens ${max_new_tokens}