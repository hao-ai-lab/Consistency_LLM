"""Compare actual trainer parameter gradients on controlled synthetic data."""

import argparse
from contextlib import nullcontext, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def load_trainer(path, name):
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CllmTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixed_path = Path("cllm/cllm_trainer_global.py")
    classes = {"baseline": load_trainer(args.baseline, "original_trainer"),
               "fixed": load_trainer(fixed_path, "fixed_trainer")}
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=1024, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "eager"
    initial = LlamaForCausalLM(config).state_dict()
    first = torch.randint(16, 1024, (1, 128), device="cuda")
    last = first.clone()
    last[:, -16:] = torch.randint(16, 1024, (1, 16), device="cuda")
    torch.distributed.barrier = lambda: None
    tensors, results = {}, {}
    for dtype in (torch.float32, torch.bfloat16):
        dtype_name = str(dtype).split(".")[-1]
        variants = [("baseline", None), ("fixed", 7), ("fixed", 128), ("fixed", 512)]
        measurements = {}
        for name, chunk in variants:
            model = LlamaForCausalLM(config).to(device="cuda", dtype=dtype).train()
            model.load_state_dict(initial)
            trainer = object.__new__(classes[name])
            trainer.max_new_tokens = 16
            trainer.use_gt_labels = False
            trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
            trainer.args = SimpleNamespace(qlora=False, local_rank=-1,
                                           consistency_loss_chunk_size=chunk)
            trainer.accelerator = SimpleNamespace(accumulate=lambda _: nullcontext(),
                                                  backward=lambda loss: loss.backward())
            inputs = {"jacobian_trajectory": [first.clone(), last.clone()],
                      "attention_mask": torch.ones_like(first),
                      "teacher_output_ids": last.cpu().tolist()}
            torch.manual_seed(100)
            random.seed(100)
            with redirect_stdout(sys.stderr):
                loss = trainer.consistency_training_step(model, inputs)
            key = name if chunk is None else f"{name}_{chunk}"
            gradients = {param_name: parameter.grad.detach().cpu()
                         for param_name, parameter in model.named_parameters()}
            tensors[f"{dtype_name}/{key}"] = gradients
            measurements[key] = {"loss": loss.item()}
            if name == "baseline":
                reference = gradients
                continue
            per_parameter = {}
            all_ref, all_actual = [], []
            for param_name, grad in gradients.items():
                ref = reference[param_name].double().flatten()
                actual = grad.double().flatten()
                delta = actual - ref
                per_parameter[param_name] = {
                    "max_abs": delta.abs().max().item(),
                    "relative_l2": (delta.norm() / ref.norm().clamp_min(1e-30)).item(),
                    "finite": bool(torch.isfinite(actual).all()),
                }
                all_ref.append(ref)
                all_actual.append(actual)
            ref, actual = torch.cat(all_ref), torch.cat(all_actual)
            measurements[key].update({
                "max_abs": (actual - ref).abs().max().item(),
                "relative_l2": ((actual - ref).norm() / ref.norm()).item(),
                "cosine": torch.nn.functional.cosine_similarity(actual, ref, dim=0).item(),
                "finite": bool(torch.isfinite(actual).all()),
                "per_parameter": per_parameter,
            })
        results[dtype_name] = measurements
    tensor_path = args.output.with_suffix(".pt")
    torch.save(tensors, tensor_path)
    sources = [args.baseline, fixed_path, Path("cllm/losses.py"), Path(__file__)]
    artifact = {"scope": "Deterministic synthetic Llama eager single-GPU trainer backward; not task quality",
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "model_config": config.to_dict(), "sequence_length": 128,
                "tensor_path": str(tensor_path.resolve()), "results": results,
                "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sources}}
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({dtype: {name: {key: value for key, value in row.items() if key != "per_parameter"}
                             for name, row in variants.items()}
                      for dtype, variants in results.items()}, indent=2))


if __name__ == "__main__":
    main()
