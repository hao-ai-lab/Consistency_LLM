model_path=$1
cllm_type=$2

python3 applications/chat_cli_cllm.py --model_path ${model_path} --cllm_type ${cllm_type} --chat --debug
