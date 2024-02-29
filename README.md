# Consistency Large Language Models: A Family of Efficient Parallel Decoders
## Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
## Introduction

## Installation
1. Environment setup:
```
conda create -n cllm python=3.9
conda activate cllm
```
2. Clone this repository and build from source:
```
git clone git@github.com:snyhlxde1/Consistency_LLM.git
cd Consistency_LLM
```
3. Install dependency:
```
pip install -r requirements.txt
```
## Model Weights
#### Target Model
| Size | Dataset |  Hugging Face Repo                             |
| ---- | -------- | --------------------------------------------- | 
| 7B   | ShareGPT |  [cllm/consistency-llm-sharegpt48k](https://huggingface.co/cllm/consistency-llm-sharegpt48k)   |
| 7B  | GSM8K | [FasterDecoding/medusa-vicuna-13b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-13b-v1.3) |
| 7B  | Spider | [FasterDecoding/medusa-vicuna-33b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-33b-v1.3) |
| 7B  | Code-Search-Net Python | [FasterDecoding/medusa-vicuna-33b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-33b-v1.3) |
#### CLLM
| Size | Dataset |  Hugging Face Repo                             |
| ---- | -------- | --------------------------------------------- | 
| 7B   | ShareGPT |  [cllm/consistency-llm-sharegpt48k](https://huggingface.co/cllm/consistency-llm-sharegpt48k)   |
| 7B  | GSM8K | [FasterDecoding/medusa-vicuna-13b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-13b-v1.3) |
| 7B  | Spider | [FasterDecoding/medusa-vicuna-33b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-33b-v1.3) |
| 7B  | Code-Search-Net Python | [FasterDecoding/medusa-vicuna-33b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-33b-v1.3) |
## Usage
### Inference 
```
python -m medusa.inference.cli --model FasterDecoding/medusa-vicuna-33b-v1.3`
```
### Training
#### Collect Jacobi trajectory
- Method 1: Directly download Jacobi trajectory in hugging face to `./data/collected_jacobi_trajectory/`.
- Method 2(Generate trajectory suitable to your own target model and dataset): Download raw dataset ([ShareGPT](https://huggingface.co/datasets/cllm/sharegpt_20230521_2k_clean_lang_split_identity_gpt4), [Spider](https://huggingface.co/datasets/cllm/spider)) in `./data/raw_data`. Then run the `generate_trajectory_{dataset_name}.py` and the training dataset for a CLLM will be saved in  `./data/collected_jacobi_trajectory/`. For example,
```
# for gsm8k dataset generation, max_new_tokens corresponds to the size of n_token_sequence
cd data
CUDA_VISIBLE_DEVICES=0 python generate_trajectory_gsm8k.py --model path_to_target_model --filename ./raw_data/gsm8k_train.jsonl --use_aug --use_labels --max_new_tokens 16 --max_new_seq_len 512
```
#### Refine target model to a CLLM
Please adjust `train_cllm.sh` to match your local file path.
```
cd cllm
bash train_cllm.sh
```
#### Evaluation
The throughput speed and generation quality can be evaluated in `eval` folder. Take GSM8K dataset for example, 
```
# for gsm8k dataset evaluation
cd eval
cd gsm8k
# compare throughput speed of CLLM using AR decoding and Jacobi decoding
CUDA_VISIBLE_DEVICES=0 python speedup.py --test_model path_to_cllm --max_new_tokens 16 --filename ./test.jsonl
# test accuracy in gsm8k
CUDA_VISIBLE_DEVICES=0 acc.py --model_dir path_to_cllm --max_new_tokens_for_consistency 16 --temperature 0.0 --top_p 1.0 \
--output_file_name 'cllm_generated_gsm8k.jsonl' --dev_set "gsm8k" --prompt_type math-single --max_tokens 1024 --use_consistency_decoding
```
## Citation
## Acknowledgements
