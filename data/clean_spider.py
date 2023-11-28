from collector import Collector


def transform(i, case, need_label=False):
    SQL_prompt = "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer. "
    case["id"] = f"identity_{i}"
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content": SQL_prompt + case['question']
            },
            {
                "role": "assistant",
                "content": " ".join(case['query_toks_no_value'])
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content": SQL_prompt + case['question']
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "spider"
    c = Collector(data_name)
    c.collect("train", transform)
    c.collect("validation", transform, 200)
