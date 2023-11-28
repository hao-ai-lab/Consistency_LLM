
import os
import json
import random

DATASET_NAMES = {
    "spider" : "spider_train.json",
    "gsm8k" : "gsm8k_train.json",
    "finance" : "gbharti_finance-alpaca_train.json"
}

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)

def get_sample_idx(cases, i, total_size):
    def gen_prob(category_num, i):
        if category_num == 2:
            p1 = 1 - i * 1.0 / total_size
            p2 = 1 - p1
            return [p1, p2]
        elif category_num == 3:
            p1 = i / total_size * 0.7 + 0.1 # 0.1 ~ 0.8
            p2 = 0.8 - i / total_size * 0.7 # 0.8 ~ 0.1
            p3 = 1 - p1 - p2
            return [p1, p2, p3]
        else:
            raise ValueError()       
        
    category_num = len(cases)
    probs = gen_prob(category_num, i)
    return random.choices(range(category_num), probs)[0]
    
def load_case(dataset):
    cases = json.load(open(DATASET_NAMES[dataset], "r"))
    return cases

def sample_cases(cases, total_size):
    sampled = []
    idx_in_category = [0 for _ in range(len(cases))]
    for i in range(total_size):
        category_id = get_sample_idx(cases, i, total_size)
        sample = cases[category_id][idx_in_category[category_id]]
        # print(sample)
        sampled.append(sample)
        idx_in_category[category_id] += 1
        idx_in_category[category_id] %= len(cases[category_id])
    return sampled
        
if __name__ == "__main__":
    datasets = ["gsm8k", "spider", "finance"]
    cases = []
    for dataset in datasets:
        # run_cmd(f"python clean_{dataset}.py")
        cases.append(load_case(dataset))

    total_size = 6000
    sampled = sample_cases(cases, total_size)
    with open(f'smooth.json', 'w') as f:
        json.dump(sampled, f)
    ten_percent_sample = []
    for i in range(len(datasets)):
        ten_percent_sample += cases[i][:int(total_size//len(datasets)*0.1)]
    with open(f'online_sample.json', 'w') as f:
        json.dump(sampled, f)

    
    sharp_mix = []
    start = 0
    for i in range(len(datasets)):
        sharp_mix += cases[i][start: start + total_size // len(datasets)]
        start += total_size // len(datasets)
    with open(f'sharp.json', 'w') as f:
        json.dump(sharp_mix, f)

