# Consistency_LLM

## Install
1. Environment setup:
```
conda create -n cllm python=3.9
conda activate cllm
```
2. Clone this repository and build from source:
```
git clone 
cd Consistency_LLM
```
3. Install dependency:
```
pip install -r requirements.txt
```

## Usage
### Prepare data
```
cd data
mkdir raw_data
python clean_{dataset}.py
```
dataset can take the value of `spider`, `finance`, `code_search`, `gsm8k`.

### LLaMA
1. Customized offline distillation:
```
bash bash_scripts/{dataset_name}/offline.sh {your_datapath} {sample_source} {distillation_method}
```
3. Customized online distillation:
```
bash bash_scripts/{dataset_name}/offline.sh {your_datapath} {sample_source} {distillation_method}
```

### T5
1. Customized offline distillation:
```
bash bash_scripts/t5/offline.sh {your_datapath} {dataset_name} {sample_source} {distillation_method}
```
2. Customized online distillation:
```
bash bash_scripts/t5/onine.sh {your_datapath} {dataset_name} {sample_source} {distillation_method}
```

### Command options
```
--student_model_path: path to the student (small) model
--teacher_model_path: path to the teacher (big) model
--mode: distillation mode. Select one from {online, offline} \
--sample_source: sampling methods. Select one from {teacher, student, mix_token, mix_request} \
--kl_method: distillation methods. Select one from {forward, reverse, jsd} \
```

### Datasets
This repo currently supports distillation and evaluation on the following datasets:

Models | GSM8K | Spider | Finance-Alpaca | CSN Python | PIQA | Starcode | Arena | CNN Dailymail | Xsum |
:---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: | :---: | :---: |
 LLaMA | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |  |
T5 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |  |  | :heavy_check_mark: | :heavy_check_mark: |
