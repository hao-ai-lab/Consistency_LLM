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
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from cllm.utils import jacobian_generated_data_postprocessed

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

    return dict(sources_input_ids=sources_input_ids, sources_len=sources_tokenized["input_ids_lens"], labels_ids=labels)

def preprocess_sharegpt(data, tokenizer):
    
    train_dataset = []
    for i in tqdm(range(len(data))):
        d = data[i]
        #if len(d["conversations"]) > 2:
        #    continue
        prompt = d["conversations"][0]["value"]
        
        if len(prompt) > 1024:
            # exclude prompts that are too long
            continue

        conv = get_conversation_template(model_path)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt_with_template = conv.get_prompt()

        #jacobian_prompt = prompt_with_template
        prompt_with_template_ids = tokenizer(prompt_with_template, return_tensors="pt")['input_ids']
        inputs = torch.Tensor(prompt_with_template_ids).unsqueeze(0).to(dtype=torch.int)

        labels = tokenizer(prompt_with_template + d["conversations"][1]["value"], return_tensors="pt")['input_ids'][0]
        labels_ids = torch.concat((labels, torch.tensor([tokenizer.eos_token_id])), dim=-1).to(dtype=torch.int)
        
        train_dataset.append(dict(sources_input_ids=inputs, sources_len=[
            input.ne(tokenizer.pad_token_id).sum().item() for input in inputs], labels_ids=labels_ids))
        

    return train_dataset

def train_tokenize_function_spider(examples, tokenizer):
    db_ids = [id for id in examples['db_id']]

    prompts = []
    for db_name in db_ids:
        db_path = f"data/raw_data/spider/database/{db_name}/{db_name}.sqlite"
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

def preprocess_gsm8k(
    processed_prompts,
    answers,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    train_dataset = []
    for processed_prompt, answer in zip(processed_prompts, answers):
        # Tokenize conversations
        inputs = tokenizer(
            processed_prompt,
            return_tensors="pt",
        ).input_ids
        labels_ids = tokenizer(
            processed_prompt+answer,
            return_tensors="pt",
        ).input_ids
        train_dataset.append(dict(sources_input_ids=inputs, sources_len=[
            input.ne(tokenizer.pad_token_id).sum().item() for input in inputs], labels_ids=labels_ids))

    return train_dataset

def train_tokenize_function_code_search_net(examples, tokenizer):
    prompt = "Please generate code based on the following doc:\n"

    sources = [
        build_instruction_prompt(prompt+instruction) for instruction in examples['func_documentation_string']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['func_code_string']]

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
    genearte_idx = 0
    while True:
        current_generation = next_generation
        logits = model(current_generation, generate_attention_mask).logits
        logits_trajectory.append(logits)
        
        # greedy decoding
        # next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)
        
        # top-k sampling
        topk_k = 2
        topk_values, topk_indices = torch.topk(torch.nn.functional.softmax(logits, dim=-1) / 0.01, k=topk_k, dim=-1)
        topk_prob = topk_values / torch.sum(topk_values, dim=-1, keepdim=True)
        next_tokens = torch.multinomial(topk_prob.view(-1, topk_k), 1).view(bsz, -1)
        next_generation = torch.gather(topk_indices, -1, next_tokens.unsqueeze(-1)).squeeze(-1)
        
        # hold prompt unchanged and update generated tokens
        for i in range(bsz):
            # greedy decoding convergence
            # next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i]-1:total_len-1]), dim=0)
            
            # top-k sampling convergence, jump to the next token if the same token is generated
            compare_start_idx = prompt_len[i]-1 + genearte_idx
            step = 1
            while torch.all(torch.eq(current_generation[i, compare_start_idx:compare_start_idx+step], next_generation[i, compare_start_idx:compare_start_idx + step])).item() and compare_start_idx + step < total_len:
                step += 1
            genearte_idx += step 
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], current_generation[i, prompt_len[i]-1:min(prompt_len[i]-1+genearte_idx, total_len-1)], next_generation[i, prompt_len[i]-1+genearte_idx:total_len-1]), dim=0)

            
        trajectory.append(next_generation)
        if torch.all(torch.eq(next_generation, current_generation)).item() or prompt_len[0]-1+genearte_idx >= total_len-1:
            eos_reached = len(torch.where(trajectory[-1] == tokenizer.eos_token_id)[0])>0
            return trajectory[:-1], logits_trajectory[-1], eos_reached # converged generation is saved twice so we delete the last element of trajectory list
        itr+=1

def main(filename, model, tokenizer, max_new_tokens, max_new_seq_len, use_aug, use_labels, data_size):

    if 'sharegpt' in filename.lower():
        with open(filename) as f:
            data = json.load(f)
        
        train_dataset = preprocess_sharegpt(data, tokenizer)
    elif 'spider' in filename.lower(): #use another preprocess method when training with spider dataset
        raw_train_datasets = datasets.load_dataset('spider', split='train')

        train_dataset = raw_train_datasets.map(
            train_tokenize_function_spider,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True, # not args.overwrite_cache
            desc="Running Encoding",
            fn_kwargs={"tokenizer": tokenizer}
        )
    elif 'code_search_net' in filename.lower(): #use another preprocess method when training with spider dataset
        raw_train_datasets = datasets.load_dataset('code_search_net', 'python', split='train')

        train_dataset = raw_train_datasets.map(
            train_tokenize_function_code_search_net,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True, # not args.overwrite_cache
            desc="Running Encoding",
            fn_kwargs={"tokenizer": tokenizer}
        )
    elif 'gsm8k' in filename.lower():
        data = []
        with open(filename, 'r') as file:
            for line in file:
                data.append(json.loads(line))

        prompt_mapping = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        processed_prompts = [prompt_mapping.format(input=query['question']) for query in data]
        answers = [query['answer'] for query in data]
        
        train_dataset = preprocess_gsm8k(processed_prompts, answers, tokenizer)
    else:
        raise NotImplementedError('Jacobi trajectory collection for dataset: {filename.lower()} is not currently supported.')
        
    prompt_size = min(len(train_dataset), data_size)

    counter = 0
    new_data = []

    for i in tqdm(range(prompt_size)):
        d = train_dataset[i]
        inputs = torch.Tensor(d['sources_input_ids']).unsqueeze(0).to(device=model.device, dtype=torch.int)

        itr = 0
        eos_reached=False
        while itr * max_new_tokens < max_new_seq_len and eos_reached==False:
            dic = {}
            dic['data_id']=f'data_{i}'
            dic['jacobian_itr_id']=f'itr_{itr}'
            dic['prompt_ids_len'] = d['sources_len']

            attention_mask = torch.full_like(inputs, 1, dtype=torch.int).to(model.device)
            dic['prompt_ids'] = inputs.tolist()

            print('retrieving one Jacobian trajectory...')
            jacobian_trajectory_ids, teacher_logits, eos_reached = get_jacobian_trajectory(model, tokenizer, inputs, attention_mask, max_new_tokens)
            dic["answer_trajectory_ids"] = []
            for _, id in enumerate(jacobian_trajectory_ids):
                # only support batch size=1 now
                dic["answer_trajectory_ids"].append(id[0][-max_new_tokens:].tolist())

            if use_aug:
                for j in range(len(dic["answer_trajectory_ids"])-3, 1, -1):
                    correct_positions = torch.where(torch.tensor(dic["answer_trajectory_ids"][j])!= torch.tensor(dic["answer_trajectory_ids"][-1]))[0]
                    if correct_positions.shape[0] > 1:
                        corrected_size = random.sample(range(1, correct_positions.shape[0]), k=1)
                    else:
                        continue
                    for correct_id in random.choices(correct_positions, k=corrected_size[0]):
                        aug_trajectory = dic["answer_trajectory_ids"][j].copy()
                        aug_trajectory[correct_id] = dic["answer_trajectory_ids"][-1][correct_id]
                    dic["answer_trajectory_ids"].insert(0, aug_trajectory)

            if use_labels:
                dic['labels_ids'] = d['labels_ids']

            inputs = jacobian_trajectory_ids[-1]

            dic['teacher_output_ids'] = jacobian_trajectory_ids[-1].tolist()
            new_data.append(dic)
            itr+=1

            print(f'writing counter = {counter}...')
            counter += 1
    
    print('Jacobi trajectory has been collected. Now delete low-quality generation as post processing.')
    save_path = 'data/collected_jacobi_trajectory/'    
    cleaned_data = jacobian_generated_data_postprocessed(new_data, model_path)
    new_file_name = "cleaned_" + f"{filename.lower()}_jacobi_max_new_tokens{max_new_tokens}_aug{use_aug}_labels_{use_labels}_max_seq_len_{max_new_seq_len}.json"
    new_file_path = os.path.join(save_path, new_file_name)
    
    # create directory for a path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(new_file_path, 'w') as f_merged:
        json.dump(cleaned_data, f_merged)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str,
                        default="data/raw_data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_new_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str,
                        default="models/vicuna-7b-v1.5")
    parser.add_argument("--data_size", default=5000)
    parser.add_argument("--use_aug", default=True)
    parser.add_argument("--use_labels", default=True)
    args = parser.parse_args()
    filename = args.filename
    model_path = args.model
    max_new_tokens = args.max_new_tokens
    max_new_seq_len = args.max_new_seq_len
    model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map='cuda', 
                                             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", use_fast=True)

    main(filename, model, tokenizer, max_new_tokens, max_new_seq_len, args.use_aug, args.use_labels, args.data_size)
