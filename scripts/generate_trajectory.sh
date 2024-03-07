filename=$1
model_path=$2
max_new_tokens=$3
max_new_seq_len=$4

python3 data/generate_trajectory.py \
	--filename ${filename} \
	--model ${model_path} \
	--max_new_tokens ${max_new_tokens} \
	 --max_new_seq_len ${max_new_seq_len}
