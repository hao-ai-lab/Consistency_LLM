"""Single-GPU benchmark of the actual CllmTrainer consistency training step.

Randomly initialized Llama and synthetic tokens measure execution/memory only.
This is not a trained model or a task-quality evaluation.
"""

import argparse
from contextlib import nullcontext, redirect_stdout
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, default=Path("cllm/cllm_trainer_global.py"))
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Load exactly the trainer file supplied, including the untouched upstream
    # version for the baseline. Sibling imports work in both script modes.
    sys.path.insert(0, str(args.trainer.resolve().parent))
    spec = importlib.util.spec_from_file_location("benchmark_trainer", args.trainer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=32000, hidden_size=512, intermediate_size=2048,
                         num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=8,
                         attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).train()
    first = torch.randint(16, config.vocab_size, (1, args.length), device="cuda")
    last = first.clone()
    last[:, -16:] = torch.randint(16, config.vocab_size, (1, 16), device="cuda")
    trainer = object.__new__(module.CllmTrainer)
    trainer.max_new_tokens = 16
    trainer.use_gt_labels = False
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    trainer.args = SimpleNamespace(qlora=False, local_rank=-1,
                                   consistency_loss_chunk_size=args.chunk_size)
    trainer.accelerator = SimpleNamespace(accumulate=lambda _: nullcontext(),
                                          backward=lambda loss: loss.backward())
    torch.distributed.barrier = lambda: None
    inputs = {"jacobian_trajectory": [first, last],
              "attention_mask": torch.ones_like(first),
              "teacher_output_ids": last.cpu().tolist()}
    measurements = []
    for repeat in range(args.repeats + 1):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(100)
        random.seed(100)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with redirect_stdout(sys.stderr):
            loss = trainer.consistency_training_step(model, inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        if repeat:
            measurements.append({"seconds": elapsed, "peak_allocated_bytes": peak,
                                 "incremental_peak_bytes": peak - baseline,
                                 "loss": loss.item()})
    grad_norm = sum(parameter.grad.float().square().sum().item()
                    for parameter in model.parameters() if parameter.grad is not None) ** 0.5
    result = {"trainer_path": str(args.trainer.resolve()),
              "trainer_sha256": hashlib.sha256(args.trainer.read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "transformers": transformers.__version__, "dtype": "bfloat16",
              "model_config": config.to_dict(), "length": args.length,
              "chunk_size": args.chunk_size, "measurements": measurements,
              "gradient_norm": grad_norm,
              "scope": "Synthetic-token random-Llama single-GPU actual trainer backward; no optimizer, FSDP, task-quality claim"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("trainer_path", "length", "measurements", "gradient_norm")}, indent=2))


if __name__ == "__main__":
    main()
