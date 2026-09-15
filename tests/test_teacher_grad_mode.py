from contextlib import nullcontext
from types import SimpleNamespace

import torch
from transformers.modeling_outputs import CausalLMOutput

from cllm.cllm_trainer_global import CllmTrainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)
        self.grad_modes = []

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, input_ids, attention_mask=None):
        self.grad_modes.append(torch.is_grad_enabled())
        return CausalLMOutput(logits=self.head(self.embed(input_ids)))


def test_training_step_disables_only_teacher_graph(monkeypatch):
    model = TinyModel()
    trainer = object.__new__(CllmTrainer)
    trainer.max_new_tokens = 2
    trainer.use_gt_labels = False
    trainer.tokenizer = SimpleNamespace(eos_token_id=14, pad_token_id=15)
    trainer.args = SimpleNamespace(qlora=False, local_rank=-1)
    trainer.accelerator = SimpleNamespace(accumulate=lambda _: nullcontext(),
                                          backward=lambda loss: loss.backward())
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    trajectory = [torch.tensor([[1, 2, 3, 4, 5, 6]]),
                  torch.tensor([[1, 2, 3, 4, 7, 8]])]
    inputs = {"jacobian_trajectory": trajectory,
              "attention_mask": torch.ones_like(trajectory[0]),
              "teacher_output_ids": [[1, 2, 3, 4, 7, 8]]}
    loss = trainer.consistency_training_step(model, inputs)
    assert model.grad_modes == [True, False, True]
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
