import collections
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

import torch.nn.functional as F
from transformers import LlamaModel,LlamaForCausalLM
import argparse

def delete_false_key_value(
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]
            
def tree_update_key_value(
        self,
        preserve_idx,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    tree_width = self.key_cache[0].shape[0]
    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx] = self.key_cache[layer_idx][preserve_idx, ...].repeat(tree_width, 1, 1, 1)
        self.value_cache[layer_idx] = self.value_cache[layer_idx][preserve_idx, ...].repeat(tree_width, 1, 1, 1)
        
@torch.inference_mode()
def jacobi_forward(
    self,
    input_ids: torch.LongTensor = None,
    tokenizer=None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
    chat: Optional[bool] = False,
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

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001,  dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0

        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0

        prev_len = 0
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
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001, dim=-1)

            next_point = torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)

            first_false_index = torch.where(torch.eq(current_point[0], next_point[0]) == False)[0]
            
            jacobian_trajectory.append(next_point)

            if len(first_false_index) > 0:
                fast_forward_cnt = first_false_index[0].item()

                past_key_values.delete_false_key_value(seq_length - fast_forward_cnt) # delete the false keys & values
            else:
                fast_forward_cnt = torch.sum(torch.eq(current_point, next_point)).item()

                accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]         
                first_correct_token = all_shift_one_token[:,-1]   
                if chat:
                    if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length + fast_forward_cnt]:
                        eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                        eos_position = eos_positions[0]
                        generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
                    else:
                        generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length + fast_forward_cnt], skip_special_tokens=True)

                    print(generated_str[prev_len:], flush=True, end="")
                    prev_len = len(generated_str)
                break 

            accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]
            accurate_length += fast_forward_cnt
            next_point = next_point[0, fast_forward_cnt:].view(1,-1) # only false tokens should be re-generated

            if chat:
                if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length]:
                    eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                    eos_position = eos_positions[0]

                    generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
                else:
                    generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length], skip_special_tokens=True)

                print(generated_str[prev_len:], flush=True, end="")
                prev_len = len(generated_str)

            iter_counter += 1

        return accurate_n_gram, first_correct_token, iter_counter, accurate_length


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
                # print('Successfully break!')
                # print(next_point)
                first_correct_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)[:,-1]
                break
            past_key_values.delete_false_key_value(seq_length)

            iter_counter += 1

        return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter


@torch.inference_mode()
def jacobi_forward_tree_profiling(
    self,
    input_ids: torch.LongTensor = None,
    topk_prob: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
    tree_width: Optional[int] = None,
):

    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if prefill_phase:  # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids.repeat(tree_width, 1))
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            # position_ids = position_ids.unsqueeze(0)
            position_ids = position_ids.repeat(tree_width, 1)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self.model._use_sdpa:
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
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
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
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        # TODO need the following line?

        # logits = logits[0, :, :].unsqueeze(0)
        # predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        topk_values, topk_indices = torch.topk(
            torch.nn.functional.softmax(logits, dim=-1) / 0.01, k=tree_width, dim=-1
        )
        first_correct_tokens = topk_indices[0, -1, :].reshape(tree_width, -1)
        first_correct_tokens_prob = torch.diag(topk_values[0, -1, :])

        return next_decoder_cache, first_correct_tokens, first_correct_tokens_prob
    else:  # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0
        generation_idx = 0
        topk_accumulated = topk_prob
        last_minimal_sorted_topk_indice = None
        topk_indices = None
        while True:
            iter_counter += 1
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
            if position_ids is None:
                device = (
                    input_ids.device if input_ids is not None else inputs_embeds.device
                )
                position_ids = torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                # position_ids = position_ids.unsqueeze(0)
                position_ids = position_ids.repeat(tree_width, 1)

            if self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = (
                    attention_mask
                    if (attention_mask is not None and 0 in attention_mask)
                    else None
                )
            elif self.model._use_sdpa:
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
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
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
                lm_head_slices = self.lm_head.weight.split(
                    self.vocab_size // self.config.pretraining_tp, dim=0
                )
                logits = [
                    F.linear(hidden_states, lm_head_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            
            # all_shift_one_token_greedy = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)

            # two problems
            # 1. getting the next token and compare is easy, but this will only proceed the generation by one
            # 2. if we want to compare the prev and advance more(which is the purpose), we need to compare
            # so the first solution is that, when one sub tree matched, directly advance it to replace the largest?
            #       e.g. the second candidate got the forward_num, but
            # the second solution will be, advance another string, and keep it

            topk_values, topk_indices = torch.topk(
                torch.nn.functional.softmax(logits, dim=-1) / 0.01, k=tree_width, dim=-1
            )

            topk_dot_product = torch.matmul(
                topk_accumulated, topk_values[:, generation_idx, :]
            )
            topk_accumulated, sorted_topk_indice = torch.topk(
                topk_dot_product.flatten(), k=tree_width
            )
            topk_accumulated = torch.diag(topk_accumulated)
            sorted_topk_indice = np.array(
                np.unravel_index(sorted_topk_indice.cpu().numpy(), topk_dot_product.shape)
            ).T
            
            topk_tokens = None
            for indice in sorted_topk_indice:
                # if handle_rest_token == "greedy":
                # TODO accelerate this process
                new_topk_tokens = torch.cat(
                    (
                        current_point[indice[0], : generation_idx + 1].view(1, -1),
                        topk_indices[
                            indice[0], generation_idx, indice[1]
                        ].view(1, -1),
                        topk_indices[
                            indice[0], generation_idx + 1 : seq_length - 1, 0
                        ].view(1, -1),
                    ),
                    dim=-1,
                )
                # elif handle_rest_token == "sampling":
                # new_topk_tokens = torch.cat((current_point[indice[0], :generation_idx+1].view(1,-1), topk_indices[indice[0], change here, indice[1]].view(1,-1)), dim=-1)
                # elif handle_rest_token == "random":
                # new_topk_tokens = torch.cat((current_point[indice[0], :generation_idx+1].view(1,-1), topk_indices[indice[0], change here, indice[1]].view(1,-1)), dim=-1)
                # else:
                # raise ValueError("handle_rest_token should be one of greedy, sampling, random")

                if topk_tokens is None:
                    topk_tokens = new_topk_tokens
                else:
                    topk_tokens = torch.cat((topk_tokens, new_topk_tokens), dim=0)

            next_point = topk_tokens[:, :seq_length]
            generation_idx += 1
            
            if generation_idx >= seq_length:
                break
            # if forward_num

            current_point_corresponding_topk = current_point[
                sorted_topk_indice[:, 0], :
            ]
            
            forward_num = 1
            max_agreement_idx = torch.where(
                torch.eq(
                    current_point_corresponding_topk[:, : generation_idx + forward_num],
                    next_point[:, : generation_idx + forward_num],
                ).sum(dim=-1)
                == generation_idx + forward_num
            )[0]
            
            
            #do more here about what to select for the rest
            
            while len(max_agreement_idx) > 0 and generation_idx + forward_num <= seq_length:
                potential_forward_num = forward_num + 1
                potential_max_agreement_idx = torch.where(
                    torch.eq(
                        current_point_corresponding_topk[:, : generation_idx + potential_forward_num],
                        next_point[:, : generation_idx + potential_forward_num],
                    ).sum(dim=-1)
                    == generation_idx + potential_forward_num
                )[0]
                if len(potential_max_agreement_idx) > 0:
                    max_agreement_idx = potential_max_agreement_idx
                    forward_num = potential_forward_num
                else:
                    break
           
            # if we allow multiple forward_num
            # TODO we do nothing
            
            # elif we just allow one forward_num
            if len(max_agreement_idx) > 0:
                # if gatekeeper == "max":
                # minimal_idx = max_agreement_idx[0]
                # minimal_sorted_topk_indice = sorted_topk_indice[minimal_idx]
                # elif gatekeeper == "random":
                # rescale_topk_accumulated = torch.diag(topk_accumulated) / torch.diag(topk_accumulated).sum()
                # rescale with softmax
                rescale_topk_accumulated = torch.nn.functional.softmax(
                    torch.diag(topk_accumulated), dim=0
                )
                random_threshold = torch.rand(rescale_topk_accumulated.shape).to(rescale_topk_accumulated.device)
                
                selected_topk_advance = random_threshold < rescale_topk_accumulated
                
                selected_topk_advance = torch.where(
                    selected_topk_advance==True
                )[0]
                
                candidate = np.intersect1d(max_agreement_idx.cpu().numpy(), selected_topk_advance.cpu().numpy())
                
                # minimal_idx = random.choice(potential_list, p=topk_accumulated[potential_list]).item()


                if len(candidate) > 0:
                    # minimal_idx = 0 if 0 in max_agreement_idx else candidate[0] 
                    minimal_idx = candidate[0]
                    minimal_sorted_topk_indice = sorted_topk_indice[minimal_idx]
                    next_point = torch.cat(
                        (
                            current_point_corresponding_topk[
                                minimal_idx, : generation_idx + 1
                            ].repeat(tree_width, 1),
                            topk_indices[
                                minimal_sorted_topk_indice[0],
                                generation_idx: generation_idx + forward_num,
                                :,
                            ].view(tree_width, -1),
                            topk_indices[
                                minimal_sorted_topk_indice[0],
                                generation_idx + forward_num : seq_length - 1,
                                0,
                            ].repeat(tree_width, 1),
                        ),
                        dim=-1,
                    )
                    next_point = next_point[:, :seq_length]

                    topk_accumulated = torch.diag(topk_values[
                        minimal_sorted_topk_indice[0], generation_idx, :
                    ])
                    generation_idx += forward_num
                    if generation_idx >= seq_length:
                        last_minimal_sorted_topk_indice = minimal_sorted_topk_indice
                        break
            
            past_key_values.delete_false_key_value(seq_length)

            # next_point= torch.cat((current_point[:, 0].view(1,-1), all_shift_one_token[:, :seq_length-1].view(1,-1)), dim=-1)
            
            jacobian_trajectory.append(next_point)
            
        assert generation_idx == seq_length

        # just return the best one and update cache
        try:
            selected_subsentence_idx = last_minimal_sorted_topk_indice[0]
        except:
            selected_subsentence_idx = 0
        next_point = next_point[selected_subsentence_idx, :].view(1, -1)
        first_correct_tokens = topk_indices[selected_subsentence_idx, -1, :].reshape(tree_width, -1)
        first_correct_tokens_prob = torch.diag(topk_values[selected_subsentence_idx, -1, :])
        past_key_values.tree_update_key_value(selected_subsentence_idx)
        
        return (
            jacobian_trajectory[:-1],
            next_point,
            first_correct_tokens,
            first_correct_tokens_prob,
            iter_counter
        )
