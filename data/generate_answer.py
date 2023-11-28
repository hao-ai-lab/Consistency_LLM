import json
from transformers import AutoTokenizer, LlamaForCausalLM
from fastchat.model.model_adapter import get_conversation_template
import torch
from tqdm import tqdm
import argparse
    
def generate_answer(prompt, model, tokenizer):
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt_with_template = conv.get_prompt()
    max_new_tokens = 128
    inputs = tokenizer([prompt_with_template], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)[0][inputs.input_ids.shape[-1]:-1]
    generated_str = tokenizer.decode(generated)
    return generated_str


def main(filename, model, tokenizer):
    with open(filename) as f:
        data = json.load(f)

    for d in tqdm(data):
        assert len(d["conversation"]) == 1
        
        prompt = d["conversation"][0]["content"]
        answer = generate_answer(prompt, model, tokenizer)
        d["conversation"].append(
            {
                "role" : "assistant",
                "content" : answer
            }
        )
        
    with open(f"{filename.split('.')[0]}_with_answer.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str,
                        default="data/raw_data/spider_train.json")
    parser.add_argument("--model", type=str,
                        default="lmsys/vicuna-7b-v1.3")
    args = parser.parse_args()
    filename = args.filename
    model_path = args.model
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', 
                                             torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    main(filename, model, tokenizer)
