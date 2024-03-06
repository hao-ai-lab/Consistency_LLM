filename=$1
model_path=$2

python3 data/generate_trajectory.py \
	--filename ${filename} \
	--model ${model_path}
