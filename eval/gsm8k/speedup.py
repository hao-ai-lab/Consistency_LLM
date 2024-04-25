from dataclasses import dataclass, field
import json
import math
import pathlib
import functools
from typing import Dict, Optional, Sequence, List, Tuple
import random
from tqdm import tqdm
import torch.nn.functional as F
import sqlite3
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
from fastchat.model.model_adapter import get_conversation_template
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers import LlamaModel, LlamaForCausalLM, GenerationConfig
import argparse

import os

import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from cllm.utils import detect_repetitive_patterns
from cllm.cllm_llama_modeling import delete_false_key_value, jacobi_forward_profiling

DynamicCache.delete_false_key_value = delete_false_key_value
LlamaForCausalLM.jacobi_forward = jacobi_forward_profiling

def jacobi_generate(inputs, model, tokenizer, max_new_tokens, max_new_seq_len):
    converge_step = []
    forward_times = 0

    all_jacobian_trajectory = []
    prompt_len = torch.sum(inputs['attention_mask'], dim=-1)
    generation = inputs['input_ids']
    ### prefill the kv-cache
    past_key_values, first_correct_token = model.jacobi_forward(input_ids=inputs['input_ids'], max_new_tokens=max_new_tokens, past_key_values=None, use_cache = True, prefill_phase = True)
    ### generation phase
    itr = 0
    eos_reached = False
    while True:
        itr+=1
        bsz = 1 # only support batch_size = 1 now
        # randomly initialize the first point of jacobian trajectory
        random_point = torch.tensor(random.choices(generation[0], k=(max_new_tokens-1)), device="cuda").view(1,-1)
        input_ids = torch.cat((first_correct_token.view(1,-1), random_point),dim=-1)
        jacobian_trajectory, n_gram_generation, first_correct_token, iter_steps = model.jacobi_forward(input_ids=input_ids, max_new_tokens=max_new_tokens, past_key_values=past_key_values, use_cache = True, prefill_phase = False)
        forward_times += iter_steps
        all_jacobian_trajectory.append(jacobian_trajectory)
        eos_positions = torch.where(n_gram_generation[0]==tokenizer.eos_token_id)[0]

        if len(eos_positions)>0:
            eos_reached = True
        
        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_id 
        generation = torch.cat((generation, n_gram_generation), dim=-1)
        if eos_reached or itr*max_new_tokens > max_new_seq_len:
            break
    
    # to support bsz > 1
    converge_step.append(forward_times / itr)

    return generation[0, prompt_len:], converge_step, all_jacobian_trajectory

def jacobian_speed_evaluate(processed_prompt, model, tokenizer, max_new_tokens, max_new_seq_len):

    time_speed = []
    eos_reached = False
    inputs = tokenizer([processed_prompt], return_tensors="pt").to(model.device)
    t1 = torch.cuda.Event(enable_timing=True)
    t2 = torch.cuda.Event(enable_timing=True)
    t1.record()
    jacobi_generation, converge_step, all_jacobian_trajectory = jacobi_generate(inputs, model, tokenizer, max_new_tokens, max_new_seq_len)
    t2.record()
    torch.cuda.synchronize()
    
    t = t1.elapsed_time(t2) / 1000
    prompt_token_len = torch.sum(inputs['attention_mask'], dim=-1)
    eos_positions = torch.where(jacobi_generation==tokenizer.eos_token_id)[0]
    if len(eos_positions)>0:
        eos_reached = True
        total_generation_len = jacobi_generation[:int(eos_positions[0])].shape[0]
        decoded_generation = tokenizer.decode(jacobi_generation[:int(eos_positions[0])])
    else:
        total_generation_len = jacobi_generation.shape[0]
        decoded_generation = tokenizer.decode(jacobi_generation)
    time_speed.append(total_generation_len/t)

    return eos_reached, time_speed, converge_step, jacobi_generation, decoded_generation, all_jacobian_trajectory
    
def speed_compare(args):
    # Load model and tokenizer
    model = transformers.LlamaForCausalLM.from_pretrained(args.test_model_path, low_cpu_mem_usage=True, device_map='auto', 
                                             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.teacher_model_path,
        padding_side="right",
        use_fast=False,
    )
    ##### compare speed of jacobian and ar #####
    converge_step = []
    ar_time_speed = []
    jacobian_time_speed = []
    filename = args.filename
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    
    per_request_meta_trajectory_records = []
    data_lst = range(args.data_size)
    # only support batch size ==1 now
    for i in tqdm(data_lst): 
        d = data[i]
        prompt_mapping = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        processed_prompt = prompt_mapping.format(input=d['question'])
        max_new_tokens = args.max_new_tokens
        inputs = tokenizer([processed_prompt], return_tensors="pt").to(model.device)
        ar_begin = time.time()
        ar_generated = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)[0][inputs['input_ids'].shape[-1]:-1]
        ar_end = time.time()
        print(f'ar generated length: {len(ar_generated)}')
        eos_reached, jacobian_time_speed_lst, jacobian_itr_step_lst, decoded_ids, decoded_result, all_jacobian_trajectory = jacobian_speed_evaluate(processed_prompt, model, tokenizer, max_new_tokens, args.max_new_seq_len)
        
        if not detect_repetitive_patterns(tokenizer, decoded_ids, repeat_ngram_size=10):
            per_request_meta_trajectory_records.append(all_jacobian_trajectory)

            jacobian_time_speed.append(*jacobian_time_speed_lst)
            converge_step.append(*jacobian_itr_step_lst)

            inputs = tokenizer([processed_prompt], return_tensors="pt").to(model.device)

            gen_cfg = GenerationConfig.from_model_config(model.config)

            ar_begin = torch.cuda.Event(enable_timing=True)
            ar_end = torch.cuda.Event(enable_timing=True)
            ar_begin.record()
            ar_generated = model.generate(**inputs, use_cache=True, max_new_tokens=512)[0][inputs.input_ids.shape[-1]:-1]
            ar_end.record()
            torch.cuda.synchronize()
            
            #print(ar_generated)
            print(f'ar generated length: {len(ar_generated)}')
            ar_time = ar_begin.elapsed_time(ar_end) / 1000
            print(f'ar time: {len(ar_generated)/(ar_time)}')
            ar_time_speed.append(len(ar_generated)/ar_time)
    
    # all trajectory analsis for speedup interpretability
    fast_forward_and_fix_points_statistics = {}
    # initialize dict for all stats
    fast_forward_and_fix_points_statistics['fix_points'] = []
    fast_forward_and_fix_points_statistics['fast_forward'] = []
    fast_forward_and_fix_points_statistics['fix_points_per_gram'] = []

    # iterate over all requests
    for all_generation_trajectory in per_request_meta_trajectory_records:
        fast_forward_metrics = []

        fix_points_metrics = 0

        effective_trajectory_length = args.max_new_tokens
        # iterate over all n-grams, across the sequence length dimension
        # last trajectory contains eos, we need to keep track
        last_traj_flag = False
        for n_gram_id in range(len(all_generation_trajectory)):
            # initialize fix_points_tracker
            fix_points_tracker = {}
            for pos_ind in range(args.max_new_tokens):
                # to track how many times right token is predicted right
                fix_points_tracker[pos_ind] = 0

            # initialize single_fast_forward_metrics
            single_fast_forward_metrics = []

            generation_trajectory = all_generation_trajectory[n_gram_id]

            if n_gram_id == len(all_generation_trajectory) - 1:
                last_traj_flag = True

            correct_token_cnt = 0
            fix_points_per_gram = 0
            # look at a single n-gram trajectory
            # iterate over all points in the trajectory (with the same dimension)
            eos_reached = False
            eos_pos = None
            steps_to_convergence = 0
            for id, generation_ids in enumerate(generation_trajectory):
                # skip initialiation
                if id == 0:
                    continue
                if eos_reached == True:
                    break
                assert len(generation_ids[0]) == args.max_new_tokens

                # iterate over all tokens
                fast_forward_cnt = 0

                contiguous_correct_flag = True

                for i in range(len(generation_ids[0])):
                    token_generated = generation_ids[0][i]
                    if generation_ids[0][i] == generation_trajectory[-1][0][i]:
                        #print(BLUE + tokenizer.decode([token_generated]) + RESET, end=" ")  # print blue token
                        # update fix point tracker
                        fix_points_tracker[i] += 1

                        # update fast-forward tracker
                        # first (i + 1) is to offset index
                        if (i + 1) > correct_token_cnt and contiguous_correct_flag:
                            fast_forward_cnt += 1

                        # check whether eos has been reached as a contiguous sentence
                        if last_traj_flag and token_generated == tokenizer.eos_token_id and contiguous_correct_flag:
                            effective_trajectory_length = i + 1

                            eos_reached = True
                            eos_pos = i

                            # before break out of the loop, uppdate values
                            correct_token_cnt += fast_forward_cnt

                            break
                    else:
                        #print(RED + tokenizer.decode([token_generated]) + RESET, end=" ")  # print red token
                        if fix_points_tracker[i] > 0:
                                fix_points_tracker[i] = 0

                        if contiguous_correct_flag:
                            correct_token_cnt += fast_forward_cnt
                            contiguous_correct_flag = False

                single_fast_forward_metrics.append(fast_forward_cnt)

                steps_to_convergence += 1

            ff_baseline_cnt = {}
            for pos_ind in range(effective_trajectory_length):
                # to track how many times right token should be predicted right, if there is only fast_forward
                ff_baseline_cnt[pos_ind] = 0

            fast_forward_ptr = 0
            next_ff_flag = True
            for pos_ind in range(effective_trajectory_length):
                if next_ff_flag:
                    fast_forward_offset = single_fast_forward_metrics[fast_forward_ptr]
                    next_ff_flag = False

                ff_baseline_cnt[pos_ind] = steps_to_convergence - fast_forward_ptr

                fast_forward_offset -= 1
                if fast_forward_offset == 0:
                    next_ff_flag = True
                    fast_forward_ptr += 1

            for pos_ind in fix_points_tracker.keys():
                cnt = fix_points_tracker[pos_ind]
                ff_baseline = ff_baseline_cnt[pos_ind]
                if cnt > ff_baseline:
                    fix_points_metrics += 1
                    fix_points_per_gram += 1

                if last_traj_flag and pos_ind == eos_pos:
                    break

            # record avg fast forward count over a single n-gram
            fast_forward_metrics.append(np.average(single_fast_forward_metrics))
            fast_forward_and_fix_points_statistics['fix_points_per_gram'].append(fix_points_per_gram)


        all_fast_forward = fast_forward_and_fix_points_statistics['fast_forward']
        all_fix_points = fast_forward_and_fix_points_statistics['fix_points']

        avg_fast_forward = np.average(fast_forward_metrics)
        all_fast_forward.append(avg_fast_forward)
        all_fix_points.append(fix_points_metrics)


    print(f"global average fast forward cnt: {np.average(fast_forward_and_fix_points_statistics['fast_forward'])}")
    print(f"global average fix-point cnt: {np.average(fast_forward_and_fix_points_statistics['fix_points'])}")
    print(f"global average fix-point per gram cnt: {np.average(fast_forward_and_fix_points_statistics['fix_points_per_gram'])}")
    
    save_path = 'data/speedup_profiling_results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    new_file_path= f'gsm8k_speedup_profiling_results_{args.max_new_tokens}_{args.max_new_seq_len}_{args.data_size}_stats.json'
    fast_forward_and_fix_points_statistics_file = os.path.join(save_path, new_file_path)

    with open(fast_forward_and_fix_points_statistics_file, 'w') as f:
        json.dump(fast_forward_and_fix_points_statistics, f, indent=4)
    
    ar_time_speed = ar_time_speed[1:]
    jacobian_time_speed = jacobian_time_speed[1:]
    print(f'ar speed: {ar_time_speed}')
    print(f'jacobian speed: {jacobian_time_speed}')
    print(f'The max speed of model {args.test_model_path} using jacobian iteration (max_new_tokens: {max_new_tokens}) is {max(jacobian_time_speed)}')
    print(f'The min speed of model {args.test_model_path} using jacobian iteration (max_new_tokens: {max_new_tokens}) is {min(jacobian_time_speed)}')
    print(f'The avg speed of model {args.test_model_path} using jacobian iteration (max_new_tokens: {max_new_tokens}) is {sum(jacobian_time_speed)/len(jacobian_time_speed)}')
    print(f'The max speed of model {args.test_model_path} using ar is {max(ar_time_speed)}')
    print(f'The min speed of model {args.test_model_path} using ar is {min(ar_time_speed)}')
    print(f'The avg speed of model {args.test_model_path} using ar is {sum(ar_time_speed)/len(ar_time_speed)}')

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str,
                        default="eval/gsm8k/test.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_new_seq_len", type=int, default=1024)
    parser.add_argument("--test_model_path", type=str,
                        default="models/vicuna-7b-sharegpt-gpt4-48k")
    parser.add_argument("--teacher_model_path", type=str,
                        default="cllm/consistency-llm-7b-sharegpt48k")
    parser.add_argument("--data_size", type=str,
                        default=500)
    args = parser.parse_args() 
    speed_compare(args)