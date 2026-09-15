"""Memory-bounded consistency distillation loss."""

import torch
from torch.utils.checkpoint import checkpoint


def soft_cross_entropy(predicts, targets, padding_mask, chunk_size=128):
    """Token-mean cross entropy with stop-gradient targets and FP32 reductions.

    Checkpoint token chunks so full-vocabulary probabilities are neither kept
    for backward nor materialized for the entire sequence at once.
    """
    if predicts.shape != targets.shape or predicts.shape[:-1] != padding_mask.shape:
        raise ValueError("logits and padding mask must have matching batch/token dimensions")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if padding_mask.dtype != torch.bool:
        raise TypeError("padding_mask must be boolean")
    if predicts.numel() == 0:
        return predicts.float().sum() * 0
    if not (~padding_mask).any():
        return predicts[..., 0, 0].float().sum() * 0

    def chunk_loss(student, teacher, mask):
        log_prob = torch.nn.functional.log_softmax(student.float(), dim=-1)
        target_prob = torch.nn.functional.softmax(teacher.float(), dim=-1)
        per_token = -(target_prob * log_prob).sum(dim=-1)
        return per_token.masked_fill(mask, 0).sum()

    # Split along sequence, preserving batch dimensions and avoiding a copy of
    # noncontiguous shifted logits into a second full-sized tensor.
    total = predicts.new_zeros((), dtype=torch.float32)
    targets = targets.detach()
    for start in range(0, predicts.shape[-2], chunk_size):
        student = predicts[..., start:start + chunk_size, :]
        teacher = targets[..., start:start + chunk_size, :]
        mask = padding_mask[..., start:start + chunk_size]
        if torch.is_grad_enabled() and student.requires_grad:
            total = total + checkpoint(chunk_loss, student, teacher, mask,
                                       use_reentrant=False)
        else:
            total = total + chunk_loss(student, teacher, mask)
    return total / (~padding_mask).sum().clamp_min(1)
