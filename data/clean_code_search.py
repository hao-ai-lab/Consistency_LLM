from collector import Collector


def transform(i, case, need_label=False):
    code_prompt = "Please generate code based on the following doc:\n"
    case["id"] = f"identity_{i}"
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content":  code_prompt + case['func_documentation_string']
            },
            {
                "role": "assistant",
                "content": case['func_code_string']
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content":  code_prompt + case['func_documentation_string']
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "code_search_net"
    language = "python"
    c = Collector(data_name, language)
    c.collect("train", transform, split_train=False)
    c.collect("test", transform, split_train=False, size=200)
