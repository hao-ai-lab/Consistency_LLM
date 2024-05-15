import json
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm
import random
import argparse
import transformers
import json
from typing import Optional, Dict, Sequence
import os, sys
import json
import argparse
import numpy as np
from torch.nn import functional as F

def get_default_question(cllm_type):
    if cllm_type == 'sharegpt':
        return "Which methods did Socrates employ to challenge the prevailing thoughts of his time?"
    elif cllm_type == 'spider':
        return "The SQL database has table named vehicle with columns ['Vehicle_ID', 'Model', 'Build_Year', 'Top_Speed', 'Power', 'Builder', 'Total_Production'], table named driver with columns ['Driver_ID', 'Name', 'Citizenship', 'Racing_Series'], table named vehicle_driver with columns ['Driver_ID', 'Vehicle_ID'], Question: What are the vehicle ids and models which have been driven by more than 2 drivers or been driven by the driver named 'Jeff Gordon'?"
    elif cllm_type == 'python':
        return "Implement the Conway's Game of Life. You should start with a 2D grid initialized with some configuration of live and dead cells. 1 for live cell and -1 for dead cell. The simulation should update the grid state by applying the rules for each cell simultaneously: any live cell with fewer than two live neighbors dies, as if by underpopulation. Any live cell with two or three live neighbors lives on to the next generation. Any live cell with more than three live neighbors dies, as if by overpopulation. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction. initial_grid = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]"
    elif cllm_type == 'gsm8k':
        return "Poppy is solving a 1000-piece jigsaw puzzle. She places a quarter of the pieces on the board, then her mom places a third of the remaining pieces. How many jigsaw pieces are left to be placed?"
    else:
        return "Tell me a short story."

def get_system_prompt(cllm_type):
    if cllm_type == 'sharegpt':
        return "Answer in English unless other language is used. A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    elif cllm_type == 'spider':
        return "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer.\n"
    elif cllm_type == 'python':
        return "Please generate code based on the following doc:\n"
    elif cllm_type == 'gsm8k':
        return ""
    else:
        return "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

def get_instruction_template(system_prompt, roles, model_input, cllm_type):
    if cllm_type == 'sharegpt':
        return system_prompt + f"{roles[0]}: " + f"{model_input}\n{roles[1]}: "
    if cllm_type == 'spider' or 'python':
        return f"### Instruction:\n" + system_prompt + f"{model_input}\n" + f"### Response:\n"
    if cllm_type == 'gsm8k':
        prompt_mapping = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        return prompt_mapping.format(input=model_input)
    else:
        return system_prompt + f"{roles[0]}: " + f"{model_input}\n{roles[1]}: "
    

def detect_repetitive_patterns(tokenizer, prompt_ids, repeat_ngram_size):

    if len(prompt_ids.shape)==1:
        prompt_ids = prompt_ids
    elif len(prompt_ids.shape)==2:
        prompt_ids = prompt_ids[0]
    elif len(prompt_ids.shape)==3:
        prompt_ids = prompt_ids[0][0]
    else:
        print(f'Unexpected shape {prompt_ids.shape}! Please check prompt ids')
        assert False

    count = 1
    for i in range(1, len(prompt_ids)):
        if prompt_ids[i] == tokenizer.eos_token_id:
            break
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
        if detect_repetitive_patterns(tokenizer, np.array(d['prompt_ids']), repeat_ngram_size=10):
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
        f_result.append(d)
    
    cleaned_f_result = []
    for i, d in enumerate(generated_data):
        if i in low_quality_data_id_lst:
            continue
        cleaned_f_result.append(d)


    return cleaned_f_result

# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

# copy norm_logits, sample, max_fn from https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/utils.py 
def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs : torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next

def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True) 
    return x_max / x_max_sum