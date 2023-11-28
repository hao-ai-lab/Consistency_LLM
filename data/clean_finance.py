from collector import Collector


def transform(i, case, need_label=False):
    case["id"] = f"identity_{i}"
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content":  case['instruction']
            },
            {
                "role": "assistant",
                "content": case['output']
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content":  case['instruction']
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "gbharti/finance-alpaca"
    c = Collector(data_name)
    c.collect("train", transform, split_train=True)
