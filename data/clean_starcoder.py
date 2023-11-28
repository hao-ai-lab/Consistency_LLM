from collector import Collector
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/rscratch/zhendong/lily/starcoderbase-1b/")

def transform(i, case, need_label=False):
    case["id"] = f"identity_{i}"
    input_ids = tokenizer(case['content'])["input_ids"]
    prompt_len = 50
    if len(input_ids) < prompt_len + 100:
        return None
    prompt = tokenizer.decode(input_ids[100:100+prompt_len])
    label = tokenizer.decode(input_ids[100+prompt_len:])
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": label
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "bigcode/starcoderdata"
    language = "assembly"
    c = Collector(data_name, data_dir=language)
    c.collect("train", transform, size=10000, 
              prefix=language, split_train=True)