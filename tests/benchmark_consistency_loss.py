"""Run from repository root: PYTHONPATH=. python tests/benchmark_consistency_loss.py.

Reports allocated-memory peaks and elapsed time for loss + backward, keeping
the same preallocated input logits in both cases. No model downloads required.
"""

import argparse
import json
import time

import torch

from cllm.losses import soft_cross_entropy


def dense(student, teacher, mask):
    # Match the original caller's FP32 inputs and in-place masking.
    student = student.float()
    teacher = teacher.detach().float()
    predict_log_prob = student.log_softmax(-1)
    targets_prob = teacher.softmax(-1)
    values = -targets_prob * predict_log_prob
    values.masked_fill_(mask.unsqueeze(-1), 0)
    return values.sum() / (~mask).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(0)
    student = torch.randn(1, args.length, args.vocab, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True)
    teacher = torch.randn_like(student)
    mask = torch.zeros(1, args.length, device="cuda", dtype=torch.bool)
    results = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "length": args.length, "vocab": args.vocab, "measurements": {}}
    for name, fn in (("dense", dense), ("chunked", soft_cross_entropy)):
        fn(student, teacher, mask).backward()
        timings, peaks = [], []
        for _ in range(args.repeats):
            student.grad = None
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss = fn(student, teacher, mask)
            loss.backward()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            peaks.append(torch.cuda.max_memory_allocated() - baseline)
        results["measurements"][name] = {"loss": loss.item(), "seconds": timings,
                                         "incremental_peak_bytes": peaks}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
