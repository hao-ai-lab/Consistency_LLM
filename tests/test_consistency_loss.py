import pytest
import torch

from cllm.losses import soft_cross_entropy


def reference(student, teacher, mask):
    values = -teacher.detach().float().softmax(-1) * student.float().log_softmax(-1)
    return values.masked_fill(mask.unsqueeze(-1), 0).sum() / (~mask).sum().clamp_min(1)


@pytest.mark.parametrize("batch,length,vocab", [(1, 1, 7), (2, 19, 71), (3, 131, 257)])
@pytest.mark.parametrize("mask_mode", ["none", "mixed", "all"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_value_and_student_gradient_match_reference(batch, length, vocab, mask_mode, dtype):
    torch.manual_seed(123)
    student = torch.randn(batch, length + 1, vocab, dtype=dtype, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    mask = torch.rand(batch, length) < 0.3
    if mask_mode == "none":
        mask.fill_(False)
    elif mask_mode == "all":
        mask.fill_(True)
    # Shifted views are deliberately noncontiguous for batch > 1.
    actual = soft_cross_entropy(student[:, :-1], teacher[:, :-1], mask, chunk_size=7)
    expected = reference(student[:, :-1], teacher[:, :-1], mask)
    actual_grad, = torch.autograd.grad(actual, student, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, student)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-3)
    assert teacher.grad is None
    assert torch.isfinite(actual_grad).all()
    if mask_mode == "all":
        assert actual.item() == 0
        assert torch.count_nonzero(actual_grad) == 0


def test_target_has_no_autograd_edge():
    student = torch.randn(2, 4, 11, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    loss = soft_cross_entropy(student, teacher, torch.zeros(2, 4, dtype=torch.bool), 2)
    teacher_grad, = torch.autograd.grad(loss, teacher, allow_unused=True)
    assert teacher_grad is None


def test_inference_matches_training_value():
    student = torch.randn(2, 8, 11, requires_grad=True)
    teacher = torch.randn_like(student)
    mask = torch.rand(2, 8) < 0.4
    expected = soft_cross_entropy(student, teacher, mask, 3)
    with torch.no_grad():
        actual = soft_cross_entropy(student, teacher, mask, 3)
    torch.testing.assert_close(actual, expected)
