from collector import Collector


def transform(i, case, need_label=False):
    case["id"] = f"identity_{i}"
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content":  case['goal']
            },
            {
                "role": "assistant",
                "content": case[f"sol{1 + int(case['label'])}"]
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content": case['goal']
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "piqa"
    c = Collector(data_name)
    c.collect("train", transform)
    c.collect("test", transform, 200)