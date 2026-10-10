from pathlib import Path
from types import SimpleNamespace
import sys

import torch


COMP_ROOT = Path(__file__).resolve().parents[1] / "ComP"
sys.path.insert(0, str(COMP_ROOT))

from model import ComP, PromptFormer, PromptFormer_2  # noqa: E402
from modules.Attention import Attention  # noqa: E402


def make_args(seq_len=6, drop_rate=0.0):
    return SimpleNamespace(
        seq_len=seq_len,
        drop_rate=drop_rate,
        device=torch.device("cpu"),
        no_cuda=True,
    )


def assert_prefix_equal(left, right, cutoff, atol=1e-6):
    torch.testing.assert_close(left[:, : cutoff + 1], right[:, : cutoff + 1], atol=atol, rtol=0)


def test_attention_cannot_read_future_positions():
    torch.manual_seed(1)
    attention = Attention(dim=8, num_heads=2).eval()
    x = torch.randn(2, 6, 8)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 20
    modality_mask = torch.ones(2, 18)

    original = attention(x, "a", modality_mask)
    perturbed = attention(changed, "a", modality_mask)

    assert_prefix_equal(original, perturbed, cutoff=3)


def test_prefix_prototypes_are_causal_and_match_full_projection_at_end():
    torch.manual_seed(2)
    prompt_former = PromptFormer(6, 10, 8, 4, make_args()).eval()
    x = torch.randn(2, 6, 8)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 20

    prefix = prompt_former._prefix_prototypes(x)
    changed_prefix = prompt_former._prefix_prototypes(changed)
    full = prompt_former.proj_n(x.permute(0, 2, 1)).permute(0, 2, 1)

    assert_prefix_equal(prefix, changed_prefix, cutoff=3)
    torch.testing.assert_close(prefix[:, -1], full, atol=1e-6, rtol=0)


def test_both_prompt_stages_cannot_read_future_positions():
    torch.manual_seed(3)
    args = make_args()
    stage1 = PromptFormer(6, 10, 8, 4, args).eval()
    stage2 = PromptFormer_2(6, 10, 8, 4, args).eval()
    x = torch.randn(2, 6, 8)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 20
    availability = torch.ones(6, 2)

    prompt, prototypes, _ = stage1(x, availability)
    changed_prompt, changed_prototypes, _ = stage1(changed, availability)
    prompt2 = stage2(x, availability, prototypes)
    changed_prompt2 = stage2(changed, availability, changed_prototypes)

    assert_prefix_equal(prompt, changed_prompt, cutoff=3)
    assert_prefix_equal(prompt2, changed_prompt2, cutoff=3)


def test_end_to_end_predictions_are_causal_with_missing_modalities():
    torch.manual_seed(4)
    args = make_args()
    model = ComP(
        args,
        adim=4,
        tdim=4,
        vdim=4,
        D_e=8,
        n_classes=1,
        depth=4,
        num_heads=2,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        no_cuda=True,
        lbd=0.3,
    ).eval()
    features = torch.randn(6, 2, 12)
    changed = features.clone()
    changed[4:] = torch.randn_like(changed[4:]) * 20
    labels = torch.zeros(2, 6)
    modality_mask = torch.ones(6, 2, 3)
    modality_mask[1, 0, 0] = 0
    modality_mask[2, 1, 2] = 0
    utterance_mask = torch.ones(2, 6)

    original = model(features, labels, modality_mask, utterance_mask, first_stage=False)[1]
    perturbed = model(changed, labels, modality_mask, utterance_mask, first_stage=False)[1]

    assert_prefix_equal(original, perturbed, cutoff=3, atol=2e-6)


def test_causal_path_backpropagates_and_updates_prototype_weights():
    torch.manual_seed(5)
    args = make_args()
    model = ComP(
        args,
        adim=4,
        tdim=4,
        vdim=4,
        D_e=8,
        n_classes=1,
        depth=4,
        num_heads=2,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        no_cuda=True,
        lbd=0.3,
    ).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.randn(6, 2, 12)
    labels = torch.randn(2, 6)
    modality_mask = torch.ones(6, 2, 3)
    utterance_mask = torch.ones(2, 6)
    before = model.prop_former_a_1.proj_n.fc1.weight.detach().clone()

    output = model(features, labels, modality_mask, utterance_mask, first_stage=False)[1]
    loss = torch.nn.functional.mse_loss(output.squeeze(-1), labels)
    loss.backward()
    gradient = model.prop_former_a_1.proj_n.fc1.weight.grad
    optimizer.step()

    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert not torch.equal(before, model.prop_former_a_1.proj_n.fc1.weight.detach())


if __name__ == "__main__":
    tests = [
        test_attention_cannot_read_future_positions,
        test_prefix_prototypes_are_causal_and_match_full_projection_at_end,
        test_both_prompt_stages_cannot_read_future_positions,
        test_end_to_end_predictions_are_causal_with_missing_modalities,
        test_causal_path_backpropagates_and_updates_prototype_weights,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
