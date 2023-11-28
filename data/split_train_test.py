import json
import random

def sample_from_interval(data, start_time, end_time):
    selected_data =  [d for d in data if d['tstamp'] >= start_time and d['tstamp'] < end_time]
    sample_idx = random.randint(0, len(selected_data) - 1)
    return selected_data[sample_idx]

def extract_first_prompt(d):
    assert d['conversation'][0]['role'] == 'user'
    d['conversation'] = [d['conversation'][0]]
    return d

filename = "clean_chat_clean_conv_20230809_10k.json"
with open(filename, "r") as f:
    data = json.load(f)

data = sorted(data, key=lambda x: x['tstamp'])
min_time = data[0]['tstamp']
max_time = data[-1]['tstamp']

eval_size = 200
interval = (max_time - min_time) / eval_size
eval_data = []
start_time = min_time
for i in range(eval_size):
    eval_data.append(sample_from_interval(data, start_time, start_time + interval))
    start_time += interval

train_data = [d for d in data if d not in eval_data]
eval_data = [extract_first_prompt(d) for d in eval_data]

# only get the first prompt for each eval
print(eval_data[0]['conversation'])
print(f"{len(train_data)}/{len(eval_data)}")

with open('train.json', 'w') as f:
    json.dump(train_data, f, ensure_ascii=False)
    
with open('eval.json', 'w') as f:
    json.dump(eval_data, f, ensure_ascii=False)