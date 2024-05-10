import transformers
import os
import re
import json
import jsonlines
import argparse
import torch
from tqdm import tqdm
import sys
import pdb
import random
from math_normalization import *

def consistency_generate(
    model,
    tokenizer,
    inputs,
    max_new_tokens,
    max_new_seq_len
    ):
    itr = 0
    while True:
        if itr == 0:
            input_ids = inputs['input_ids']
            input_masks = inputs['attention_mask']
        else:
            input_masks = torch.ones_like(input_ids).to(input_ids.device)
            for j in range(bsz):
                input_masks[j][torch.sum(inputs["attention_mask"], dim=-1)[j] + itr*max_new_tokens:] = 0

        bsz = input_ids.shape[0]
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        generation = get_jacobian_trajectory(model, tokenizer, input_ids, input_masks, max_new_tokens)
        ### tokens generated after <eos> are set to <pad>
        for j in range(bsz):
            prompt_len = torch.sum(input_masks, dim=-1)
            eos_positions = torch.where(generation[j]==tokenizer.eos_token_id)[0]
            if len(eos_positions)==0:
                # no EOS, continue to the next item in the batch
                total_token_len = prompt_len + max_new_tokens
                continue
            # otherwise, set tokens coming after EOS as pad 
            eos_reached[j] = True
            total_token_len = int(eos_positions[0])

        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_ids
        itr+=1      
        if all(eos_reached) or itr*max_new_tokens >= max_new_seq_len:
            return generation[0, :total_token_len]
        input_ids = generation

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
        tokens[i, :] = torch.tensor(random.choices(input_ids[i][attention_mask[i]==1], k=total_len), dtype=torch.long, device="cuda")
        tokens[i, : prompt_len[i]] = torch.tensor(input_ids[i][: prompt_len[i]], dtype=torch.long, device="cuda")
    itr = 0
    next_generation = tokens
    generate_attention_mask = torch.full_like(next_generation, 1).to(tokens.device)
    while True:

        current_generation = next_generation
        with torch.no_grad():
            logits = model(current_generation, generate_attention_mask).logits
        next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)

        # hold prompt unchanged and update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i]-1:total_len-1]), dim=0)
        if torch.all(torch.eq(next_generation, current_generation)).item():
            print(f"Iteration steps: {itr}")
            return next_generation # right generation is saved twice so we delete the last element of trajectory list
        itr+=1

def get_results(pred_file, dev_set):
    def test_answer(pred_str, ans_str):
        pattern = "#### (.*)$"

        if "Question" in pred_str:
            pred_str = pred_str.split("Question")[0]

        preds = re.findall(pattern, pred_str)
        pred = preds[-1] if len(preds) >= 1 else ""
        if "</s>" in pred:
            pred = pred[:-4]

        gold = ans_str
        pred = normalize_final_answer(pred)
        gold = normalize_final_answer(gold)
        return check_sympy_equivalence(gold, pred), pred, gold

    def parse_pred_ans(preds_str, golds_str, properties_list):
        num_q = 0
        acc = 0
        results = []
        preds = []
        golds = []
        correct_table = {}
        cnt_table = {}
        source_set = set()
        for pred_str, gold_str, properties in tqdm(zip(preds_str, golds_str, properties_list), total=len(preds_str)):
            num_q += 1
            result, pred, gold = test_answer(pred_str, gold_str)
            results.append(result)
            preds.append(pred)
            golds.append(gold)
            if result:
                acc += 1
            source = properties['source']
            tag = properties['tag']
            source_set.add(source)
            if source not in correct_table.keys():
                correct_table[source] = 1 if result else 0
                cnt_table[source] = 1
            else:
                correct_table[source] = (correct_table[source] + 1) if result else correct_table[source]
                cnt_table[source] += 1
            for key in tag.keys():
                value = tag[key]
                value = source+","+key+"__"+value
                if value not in correct_table.keys():
                    correct_table[value] = 1 if result else 0
                    cnt_table[value] = 1
                else:
                    correct_table[value] = (correct_table[value] + 1) if result else correct_table[value]
                    cnt_table[value] += 1
        print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
        acc_table = {}
        for key in correct_table.keys():
            acc_table[key] = correct_table[key] / cnt_table[key]
        acc_table = list(zip(acc_table.keys(), acc_table.values()))
        acc_table.sort(key=lambda x: x[0])
        for key, acc in acc_table:
            if key in source_set:
                print(key+" : "+str(acc))
            else:
                print("    " + key.split(",")[-1]+ " : " + str(acc))
        return results, preds, golds

    if dev_set in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        golds_str = []
        properties = []
        with open(f'test.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if dev_set != "all":
                    if json.loads(line)['source'].lower() == dev_set:
                        golds_str.append(json.loads(line)['target'])
                        properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
                else:
                    golds_str.append(json.loads(line)['target'])
                    properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
        preds_str = []
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                preds_str.append(json.loads(line)['response'])
        results, preds, golds = parse_pred_ans(preds_str, golds_str, properties)
        with open(pred_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        for i, line in enumerate(data):
            line['pred'] = preds[i]
            line['gold'] = golds[i]
            line['result'] = results[i]

        # Save the updated list of dictionaries back to the jsonl file
        with open(pred_file, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')

    else:
        raise NotImplementedError("Evaluation not supported.")


def get_raw_inputs(dev_set):
    # in this function, we will get the raw queries for a target dev set
    data = []
    if dev_set in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        with open(f'test.jsonl') as f:
            for line in jsonlines.Reader(f):
                data.append(line)
        if dev_set != 'all':
            data = [line for line in data if line['source'].lower() == dev_set]
    else:
        raise ValueError

    prompt_list = [line['question'] for line in data]
    return prompt_list


prompt_mapping = {
    "math-single": "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
}

if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--output_file_name', type=str, default='output.json')
    parser.add_argument('--stop', type=str, nargs='+', default=[], help="you can pass one or multiple stop strings to halt the generation process.")
    parser.add_argument('--dev_set', type=str, default='all')
    parser.add_argument('--prompt_type', type=str, default='math-single')
    parser.add_argument('--sample_num', type=int, default=-1, )
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--max_num_batched_tokens', type=int, default=2048)
    parser.add_argument(
        "--use_consistency_decoding",
        action="store_true",
        help="Whether to use consistency decoding",
    )
    parser.add_argument(
        "--max_new_tokens_for_consistency",
        type=int,
        default=16,
        help="The n-gram for consistency decoding.",
    ) 
    args = parser.parse_args()
    max_new_token = args.max_tokens
    if args.eval_only == False:
        # part 1 we set the model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='cuda',
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_dir,
            padding_side="right",
            use_fast=False,
        )
        print('>>>>>> model and tokenizer loaded')

        # part 2 we prepare raw queries and wrap them with target prompt
        raw_queries = get_raw_inputs(args.dev_set)
        prompt = prompt_mapping[args.prompt_type]
        processed_prompts = [prompt.format(input=query) for query in raw_queries]
        processed_prompts = processed_prompts[:args.sample_num] if args.sample_num > 0 else processed_prompts

        outputs = []
        # part 3 we generate answers
        for processed_prompt in tqdm(processed_prompts):

            input_ids = tokenizer([processed_prompt]).input_ids
            if args.use_consistency_decoding:
                output_ids = consistency_generate(
                    model,
                    tokenizer,
                    tokenizer([processed_prompt], return_tensors="pt").to(model.device),
                    max_new_tokens=args.max_new_tokens_for_consistency,
                    max_new_seq_len=max_new_token,
                )
                output_ids = output_ids.unsqueeze(dim=0)
            else:
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=False,
                    # temperature=args.temperature,
                    max_new_tokens=max_new_token,
                )
            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]) :]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()
            outputs.append({'prompt': processed_prompt, 'answer': output})
        print('>>>>>> generation done')

        # part 5 we save the results, always be {'id':id,'response':response}
        # if dir of output file is not exist, it will be created automatically
        with open(args.output_file_name, "w") as f:
            for id, output in enumerate(outputs):
            # note that `prompt`s are the wrapped ones
                f.write(json.dumps({'id': id, 'prompt': output['prompt'], 'response': output['answer']}) + '\n')
        print('>>>>>> writing prediction done')

    # part 6 evaluate, I guess this should be done in a separate script
    get_results(args.output_file_name, args.dev_set)
    print('>>>>>> evaluation done')


# CUDA_VISIBLE_DEVICES=0 acc.py --model_dir path_to_cllm --temperature 0.0 --top_p 1.0 --output_file_name 'cllm_generated_gsm8k.jsonl' --dev_set "gsm8k" --prompt_type math-single --max_new_tokens_for_consistency 16 --max_tokens 1024 --use_consistency_decoding