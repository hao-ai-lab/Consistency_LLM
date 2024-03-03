import json
from transformers import AutoTokenizer, LlamaForCausalLM
from fastchat.model.model_adapter import get_conversation_template
import torch
from tqdm import tqdm
import random
import argparse
from datasets import load_dataset
import datasets
import transformers
import sqlite3
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import copy
import numpy as np
import os

####### Preprocessing spider dataset #######
IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def build_instruction_prompt(instruction: str):
    return '''### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    sources_input_ids = sources_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
    #     label[:source_len] = IGNORE_INDEX

    return dict(sources_input_ids=sources_input_ids, sources_len=sources_tokenized["input_ids_lens"], labels_ids=labels)

def train_tokenize_function(database_path, examples, tokenizer):
    db_ids = [id for id in examples['db_id']]

    prompts = []
    for db_name in db_ids:
        db_path = os.join(database_path, f"{db_name}/{db_name}.sqlite")
        con = sqlite3.connect(db_path)
        cursor = con.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table";')
        curr_table = cursor.fetchall()

        table_rows = {}
        for table in curr_table:
            table_name = str(table[0])

            cursor_t = con.execute(f"SELECT * from {table_name}")
            names = list(map(lambda x: x[0], cursor_t.description))
            table_rows[table_name] = names
            cursor_t.close()

        cursor.close()
        con.close()

        database_info = "The SQL database has "
        for k, v in table_rows.items():
            database_info = database_info + f"table named {k} with columns {v}, "

        prefix= "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer. "
        prompt = prefix + database_info + "Question: "
        prompts.append(prompt)

    sources = [
        build_instruction_prompt(prompt+instruction)
        for prompt, instruction in zip(prompts, examples['question'])
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['query']]

    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

####### Get jacobian trajectory #######
@torch.inference_mode()
def get_jacobian_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens
    ):

    bsz = input_ids.shape[0]
    prompt_len = [torch.sum(t) for t in attention_mask]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # initialize the first point of jacobian trajectory
    tokens = torch.full((bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device="cuda")

    for i in range(bsz):
        tokens[i, :] = torch.tensor(random.choices(input_ids[i][attention_mask[i]==1], k=total_len)).to(dtype=torch.long, device="cuda")
        tokens[i, : prompt_len[i]] = torch.tensor(input_ids[i][: prompt_len[i]], dtype=torch.long, device="cuda")
    trajectory = []
    logits_trajectory = []
    next_generation = tokens
    generate_attention_mask = torch.full_like(next_generation, 1).to(model.device)
    trajectory.append(tokens)
    itr=0
    while True:
        
        current_generation = next_generation
        logits = model(current_generation, generate_attention_mask).logits
        logits_trajectory.append(logits)
        next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)

        # hold prompt unchanged and update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i]-1:total_len-1]), dim=0)
        trajectory.append(next_generation)
        if torch.all(torch.eq(next_generation, current_generation)).item():
            eos_reached = len(torch.where(trajectory[-1] == tokenizer.eos_token_id)[0])>0
            return trajectory[:-1], logits_trajectory[-1], eos_reached # right generation is saved twice so we delete the last element of trajectory list
        itr+=1

def detect_repetitive_patterns(prompt_ids, repeat_ngram_size):

    if len(prompt_ids.shape)==2:
        prompt_ids = prompt_ids[0]
    elif len(prompt_ids.shape)==3:
        prompt_ids = prompt_ids[0][0]
    else:
        print(f'Unexpected shape {prompt_ids.shape}! Please check prompt ids')
        assert False
        
    count = 1
    for i in range(1, len(prompt_ids)):
        if prompt_ids[i] == prompt_ids[i - 1]:
            count += 1
            if count == repeat_ngram_size:
                return True
        else:
            count = 1

    return False

def jacobian_generated_data_postprocessed(generated_data, model_path):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    low_quality_data_id_lst = []
    # delete low quality data with repetitive pattern
    for i, d in enumerate(generated_data):
        if detect_repetitive_patterns(np.array(d['prompt_ids']), repeat_ngram_size=10):
            prompt_ids = np.array(d['prompt_ids'])
            if len(prompt_ids.shape)==2:
                prompt_ids = prompt_ids[0]
            elif len(prompt_ids.shape)==3:
                prompt_ids = prompt_ids[0][0]
            print(f'Low quality generation detected: {tokenizer.decode(prompt_ids)}')
            low_quality_data_id_lst.append(i)
    print(f'{len(low_quality_data_id_lst)} low quality data detected. {len(low_quality_data_id_lst)/len(generated_data)} percent of low quality data.')

    # add complete teacher outputs
    teacher_output_inspector = {}
    for d in generated_data:
        data_id = d["data_id"]
        if data_id in teacher_output_inspector.keys():
            all_teacher_output_map = teacher_output_inspector[data_id]
        else:
            all_teacher_output_map = {}
            #print(data_id)
        itr = d["jacobian_itr_id"]
        # handle bsz=1 case only
        all_teacher_output_map[itr] = d["teacher_output_ids"][0]
        teacher_output_inspector[data_id] = all_teacher_output_map

    teacher_output_collector = {}
    for d_id in teacher_output_inspector.keys():
        all_teacher_output_map = teacher_output_inspector[d_id]
        all_itr = [int(s.split('_')[1]) for s in all_teacher_output_map.keys()]
        print(all_itr)
        max_itr = max(all_itr)
        max_itr_s = "itr_" + str(max_itr)
        complete_teacher_output = all_teacher_output_map[max_itr_s]
        teacher_output_collector[d_id] = complete_teacher_output

    f_result = []
    for d in generated_data:
        data_id = d["data_id"]
        complete_teacher_output = teacher_output_collector[data_id]
        d["complete_teacher_output_ids"] = complete_teacher_output
        # print(tokenizer.decode(complete_teacher_output))
        f_result.append(d)
    
    cleaned_f_result = []
    for i, d in enumerate(generated_data):
        if i in low_quality_data_id_lst:
            continue
        cleaned_f_result.append(d)

    # print(len(cleaned_f_result))

    return cleaned_f_result

def main(database_path, model, tokenizer, max_new_tokens, max_new_seq_len, use_aug, use_labels):
    
    raw_train_datasets = load_dataset("spider")["train"]
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={ "database_path": database_path, "tokenizer": tokenizer}
    )
    new_data = []
    for i in tqdm(range(len(train_dataset))):
        d = train_dataset[i]
        jacobian_prompt = tokenizer.decode(d['sources_input_ids'])[21:]
        itr = 0
        eos_reached=False
        while itr * max_new_tokens < max_new_seq_len and eos_reached==False:
        # while eos_reached==False:
            dic = {}
            dic['data_id']=f'data_{i}'
            dic['jacobian_itr_id']=f'itr_{itr}'
            dic['prompt_ids_len'] = d['sources_len']
            dic['jacobian_prompt'] = jacobian_prompt
            # d['sources_len'] == tokenizer(jacobian_prompt, return_tensors="pt")['input_ids'].shape[1]
            inputs = tokenizer(jacobian_prompt, return_tensors="pt").to(model.device)
            jacobian_trajectory_ids, teacher_logits, eos_reached = get_jacobian_trajectory(model, tokenizer, inputs['input_ids'], inputs['attention_mask'], max_new_tokens)
            dic["answer_trajectory_ids"] = []
            for _, id in enumerate(jacobian_trajectory_ids): 
                # only support batch size=1 now
                dic["answer_trajectory_ids"].append(id[0][-max_new_tokens:].tolist()) 

            if use_aug:
                for j in range(len(dic["answer_trajectory_ids"])-3, -1, -1):
                    correct_positions = torch.where(torch.tensor(dic["answer_trajectory_ids"][j]!=dic["answer_trajectory_ids"][-1]))[0]
                    for correct_id in random.choices(correct_positions, k=8):
                        aug_trajectory = dic["answer_trajectory_ids"][j].copy()
                        aug_trajectory[correct_id] = dic["answer_trajectory_ids"][-1][correct_id]
                    dic["answer_trajectory_ids"].insert(0, aug_trajectory)

            if use_labels:
                dic['labels_ids'] = d['labels_ids']
            jacobian_prompt = tokenizer.decode(jacobian_trajectory_ids[-1][0])[21:] # 21: is to remove the <｜begin▁of▁sentence｜>(bos) generated by tokenizer
            print(jacobian_prompt)
            new_data.append(dic)
            itr+=1
                
    print('Jacobi trajectory has been collected. Now delete low-quality generation as post processing.')
    save_path = './collected_jacobi_trajectory/'    
    cleaned_data = jacobian_generated_data_postprocessed(new_data, model_path)
    new_file_name = "cleaned_" + f"spider_jacobi_max_new_tokens{max_new_tokens}_aug{use_aug}_labels_{use_labels}_max_seq_len_{max_new_seq_len}.json"
    new_file_path = os.path.join(save_path, new_file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(new_file_path, 'w') as f_merged:
        json.dump(cleaned_data, f_merged)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str,
                        default="sharegpt/sharegpt_20230521_2k_clean_lang_split_identity_gpt4.json")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_new_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str,
                        default="llm-model/vicuna-7b-sharegpt-gpt4-48k")
    parser.add_argument("--use_aug", action="store_true")
    parser.add_argument("--use_labels", action="store_true")
    args = parser.parse_args()

    database_path = args.database_path
    model_path = args.model
    max_new_tokens = args.max_new_tokens
    max_new_seq_len = args.max_new_seq_len
    model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map='cuda', 
                                             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", use_fast=True)
    main(database_path, model, tokenizer, max_new_tokens, max_new_seq_len, args.use_aug, args.use_labels)

    # for spider dataset generation
    # CUDA_VISIBLE_DEVICES=7 python generate_trajectory_spider.py --model path_to_target_model --database_path_to_db_path ./raw_data/spider/database --use_aug --use_labels --max_new_tokens 16 --max_new_seq_len 512
