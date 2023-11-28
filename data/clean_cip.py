from collector import Collector


def transform(i, case, need_label=False):
    case["id"] = f"identity_{i}"
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content":  case['prompt']
            },
            {
                "role": "assistant",
                "content": case['response']
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content": case['prompt']
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "alespalla/chatbot_instruction_prompts"
    c = Collector(data_name)
    c.collect("train", transform)
    c.collect("test", transform, 200)