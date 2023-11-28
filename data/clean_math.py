from collector import Collector


def transform(i, case):
    case["id"] = f"identity_{i}"
    case["conversation"] = [
            {
                "role": "user",
                "content": case['question']
            }
    ]
    return case


if __name__ == "__main__":
    data_name = "math_dataset"
    c = Collector(data_name, "algebra__linear_1d")
    c.collect("train", transform)
    c.collect("test", transform, size=200)
