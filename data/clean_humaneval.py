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
                "content": case['canonical_solution']
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
    data_name = "openai_humaneval"
    c = Collector(data_name)
    c.collect("test", transform)