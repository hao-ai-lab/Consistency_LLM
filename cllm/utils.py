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

def get_instruction_template(system_prompt, roles, model_input):
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

@torch.inference_mode()
def jacobi_forward_profiling(
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
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0
        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0
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
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)
            next_point= torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)
            jacobian_trajectory.append(next_point)
            
            if torch.all(torch.eq(current_point, next_point)).item():    
                print('Successfully break!')
                print(next_point)
                first_correct_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)[:,-1]
                break
            past_key_values.delete_false_key_value(seq_length)

            iter_counter += 1

        return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter