from collector import Collector


def transform(i, case, need_label=False):
    case["id"] = f"identity_{i}"
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content":  case['question']
            },
            {
                "role": "assistant",
                "content": case['answer']
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content": case['question']
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "gsm8k"
    c = Collector(data_name, 'main')
    c.collect("train", transform)
    c.collect("test", transform, 200)