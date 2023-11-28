from collector import Collector


def transform(i, case):
    case["id"] = f"identity_{i}"
    case["conversation"] = [
            {
                "role": "user",
                "content": case['prompt']
            }
    ]
    return case


if __name__ == "__main__":
    data_name = "nampdn-ai/tiny-codes"
    c = Collector(data_name)
    c.collect("train", transform, split_train=True)
