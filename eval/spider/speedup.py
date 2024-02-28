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
from transformers import LlamaModel,LlamaForCausalLM
import argparse

def build_instruction_prompt(instruction: str):
    return '''### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

def delete_false_key_value(
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]

DynamicCache.delete_false_key_value = delete_false_key_value

@torch.inference_mode()
def jacobi_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
):
    
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids)
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0
        next_point = input_ids
        while True:

            current_point = next_point
            inputs_embeds = self.model.embed_tokens(current_point)
            attention_mask = None
            position_ids = None
            seq_length = current_point.shape[1]
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length) 
            # print(past_key_values_length) # return previous_seq_length
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.model._use_sdpa :
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            # embed positions
            hidden_states = inputs_embeds

            # decoder layers            
            for decoder_layer in self.model.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
            next_point= torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)
            first_false_index = torch.where(torch.eq(current_point[0], next_point[0]) == False)[0]
            if len(first_false_index) > 0:
                fast_forward_cnt = first_false_index[0].item()
                past_key_values.delete_false_key_value(seq_length - fast_forward_cnt) # delete the false keys & values
            else:
                fast_forward_cnt = torch.sum(torch.eq(current_point, next_point)).item()
                accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]         
                first_correct_token = all_shift_one_token[:,-1]   
                break 
            accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]
            accurate_length += fast_forward_cnt
            next_point = next_point[0, fast_forward_cnt:].view(1,-1) # only false tokens should be re-generated

        return accurate_n_gram, first_correct_token
        

LlamaForCausalLM.jacobi_forward = jacobi_forward

def jacobi_generate(inputs, model, tokenizer, max_new_tokens, max_new_seq_len):

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
        n_gram_generation, first_correct_token = model.jacobi_forward(input_ids=input_ids, max_new_tokens=max_new_tokens, past_key_values=past_key_values, use_cache = True, prefill_phase = False)
        eos_positions = torch.where(n_gram_generation[0]==tokenizer.eos_token_id)[0]
        if len(eos_positions)>0:
            eos_reached = True
        
        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_id 
        generation = torch.cat((generation, n_gram_generation), dim=-1)

        if eos_reached or itr*max_new_tokens > max_new_seq_len:
            break

    return generation[0, prompt_len:]


def jacobian_speed_evaluate(processed_prompt, model, tokenizer, max_new_tokens, max_new_seq_len):

    time_speed = []
    eos_reached = False
    inputs = tokenizer([processed_prompt], return_tensors="pt").to(model.device)
    t1 = time.time() 
    jacobi_generation = jacobi_generate(inputs, model, tokenizer, max_new_tokens, max_new_seq_len)
    t2 = time.time()
    t = t2-t1
    # prompt_token_len = torch.sum(inputs['attention_mask'], dim=-1)
    eos_positions = torch.where(jacobi_generation==tokenizer.eos_token_id)[0]
    if len(eos_positions)>0:
        eos_reached = True
        total_generation_len = jacobi_generation[:int(eos_positions[0])].shape[0]
        time_speed.append(total_generation_len/t)
    else:
        eos_reached = False

    return time_speed, eos_reached
    

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
    ar_time_speed = []
    jacobian_time_speed = []
    filename = args.filename
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    data_lst = range(501)
    # only support batch size ==1 now
    filename = '../../data/raw_data/spider/database/test_data/dev.json'
    with open(filename) as f:
        data = json.load(f)
    for d in tqdm(data[:500]):

        #### generate suitable prompt (data preprocess) for spider dataset
        db_name = d["db_id"]
        db_path = f"../../data/raw_data/spider/test_database/{db_name}/{db_name}.sqlite"
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
        prompt = prefix + database_info + "Question: " + d["question"]
        processed_prompt = build_instruction_prompt(prompt)
        max_new_tokens = args.max_new_tokens
        inputs = tokenizer([processed_prompt], return_tensors="pt").to(model.device)
        ar_begin = time.time()
        ar_generated = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)[0][inputs['input_ids'].shape[-1]:-1]
        ar_end = time.time()
        print(f'ar generated length: {len(ar_generated)}')
        time_speed_lst, eos_reached = jacobian_speed_evaluate(processed_prompt, model, tokenizer, max_new_tokens, args.max_new_seq_len)
        if eos_reached:
            jacobian_time_speed.append(*time_speed_lst)
            ar_time_speed.append(len(ar_generated)/(ar_end - ar_begin))
        else:
            continue
    
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
                        default="./test.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_new_seq_len", type=int, default=1024)
    parser.add_argument("--test_model_path", type=str,
                        default="llm-model/vicuna-7b-sharegpt-gpt4-48k")
    args = parser.parse_args() 
    speed_compare(args)

    # CUDA_VISIBLE_DEVICES=7 python speedup.py --test_model path_to_cllm --max_new_tokens 16