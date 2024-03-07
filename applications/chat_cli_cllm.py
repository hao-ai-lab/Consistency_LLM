import torch
import argparse
import subprocess

import time, os
import random
from typing import Dict, Optional, Sequence, List, Tuple
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
from fastchat.model.model_adapter import get_conversation_template
from transformers.cache_utils import Cache, DynamicCache
from transformers import LlamaModel,LlamaForCausalLM
from transformers.generation import GenerationConfig

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from cllm.utils import get_default_question, get_system_prompt, get_instruction_template
from cllm.cllm_llama_modeling import delete_false_key_value, jacobi_forward, jacobi_forward_profiling
        
DynamicCache.delete_false_key_value = delete_false_key_value
LlamaForCausalLM.jacobi_forward = jacobi_forward

def jacobi_generate(inputs, model, tokenizer, max_new_tokens, max_new_seq_len):
    #converge_step = []
    CHAT = int(os.environ.get("CHAT", 0))
    if CHAT:
        chat = True
    else:
        chat = False
    forward_times = 0

    #all_jacobian_trajectory = []

    prompt_len = torch.sum(inputs['attention_mask'], dim=-1)
    generation = inputs['input_ids']
    ### prefill the kv-cache

    past_key_values, first_correct_token = model.jacobi_forward(input_ids=inputs['input_ids'], tokenizer=tokenizer, max_new_tokens=max_new_tokens, past_key_values=None, use_cache = True, prefill_phase = True, chat=chat)
    ### generation phase
    itr = 0
    global_accurate_length = 0
    eos_reached = False
    while True:
        itr+=1
        bsz = 1 # only support batch_size = 1 now
        # randomly initialize the first point of jacobian trajectory
        random_point = torch.tensor(random.choices(generation[0], k=(max_new_tokens-1)), device="cuda").view(1,-1)
        input_ids = torch.cat((first_correct_token.view(1,-1), random_point),dim=-1)
        n_gram_generation, first_correct_token, iter_steps, accurate_length = model.jacobi_forward(input_ids=input_ids, tokenizer=tokenizer, max_new_tokens=max_new_tokens, past_key_values=past_key_values, use_cache = True, prefill_phase = False, chat=chat)
        forward_times += iter_steps
        global_accurate_length += accurate_length
        #all_jacobian_trajectory.append(jacobian_trajectory)
        
        eos_positions = torch.where(n_gram_generation[0]==tokenizer.eos_token_id)[0]

        if len(eos_positions)>0:
            eos_reached = True
        
        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_id 
        generation = torch.cat((generation, n_gram_generation), dim=-1)

        if eos_reached or itr*max_new_tokens > max_new_seq_len:
            break

    return generation, global_accurate_length / forward_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0) 
    parser.add_argument("--model_path", type=str, help="model path", default="meta-llama/Llama-2-7b-chat-hf") #tiiuae/falcon-7b-instruct #"TheBloke/Falcon-180B-Chat-GPTQ" 
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--cllm_type", type=str, default="sharegpt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="n-token sequence size",
    )
    parser.add_argument(
        "--max_new_seq_len",
        type=int,
        default=1024,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    
    if args.dtype == "float16":
        args.dtype = torch.float16
    elif args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    
    #if args.use_ds:
    config = transformers.AutoConfig.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='cuda',
        attn_implementation="flash_attention_2",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        model_max_length=2048,
        padding_side="right",
    )

    user_input = ""
    num_rounds = 0
    if args.model_type == "llama":  
        roles = ("USER", "ASSISTANT") #support vicuna
    else:
        assert False 

    user_input = ""
    if args.model_type == "llama":  
        system_prompt = get_system_prompt(args.cllm_type)
    else:
        raise NotImplementedError('Only LLaMA or LLaMA2 architecture is supported.')

    while True:
        num_rounds += 1
        if args.chat:
            model_input = input("USER: ")
        else:
            model_input = get_default_question(args.cllm_type)
            print("USER: " + model_input)

        new_inputs = get_instruction_template(system_prompt, roles, model_input, args.cllm_type)
        user_input += new_inputs

        print("ASSISTANT: " , flush=True, end="")
        inputs = tokenizer(user_input, return_tensors="pt").to(args.device)

        if not args.chat:
            tmp_greedy_output, _ = jacobi_generate(inputs, model, tokenizer, args.max_new_tokens, args.max_new_seq_len) #warmup

        os.environ["CHAT"] = "1"
        t0 = time.time()
        greedy_output, avg_fast_forwward_count = jacobi_generate(inputs, model, tokenizer, args.max_new_tokens, args.max_new_seq_len)
        
        t1 = time.time()
        
        os.environ["CHAT"] = "0"
        output = tokenizer.decode(greedy_output[0], skip_special_tokens=False)

        # re-initialize user input
        # TODO: support multi-turn conversation
        user_input = ""
        
        if args.debug:
            generated_tokens = len(greedy_output[0]) - inputs.input_ids.numel()
            print()
            print("======================================SUMMARY=======================================================")
            print("Generated tokens: ", generated_tokens,"Time: ", round(t1 - t0, 2), "s Throughput: ", round(generated_tokens / (t1 - t0), 2), "tokens/s", "Fast forwarding: ", round(avg_fast_forwward_count, 2), "tokens/step")
            print("====================================================================================================")
        if not args.chat:
            break

