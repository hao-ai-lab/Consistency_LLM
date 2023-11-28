from collector import Collector


def transform(i, case, need_label=False):
    code_prompt = " Please only include Python code in your answer, don't include any explanation."
    case["id"] = f"identity_{i}"
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content":  case['text'] + code_prompt
            },
            {
                "role": "assistant",
                "content": case['code']
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content": case['text']
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "mbpp"
    c = Collector(data_name)
    c.collect(["train", "validation", "test"], transform)