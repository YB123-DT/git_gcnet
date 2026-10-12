import pickle
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch


COMP = Path(__file__).resolve().parents[1] / "external_repos/ComP_causal/ComP"
sys.path.insert(0, str(COMP))


def _write_meld(path):
    ids = [0, 1, 2]
    speakers = {i: np.eye(9, dtype=np.int64)[[i % 9, (i + 1) % 9]] for i in ids}
    labels = {i: [i, i + 1] for i in ids}
    text = {i: np.ones((2, 4), np.float32) * i for i in ids}
    audio = {i: np.ones((2, 3), np.float32) * i for i in ids}
    visual = {i: np.ones((2, 2), np.float32) * i for i in ids}
    value = (speakers, labels, labels, text, text, text, text, text, text, text,
             text, audio, visual, {i: ["x", "y"] for i in ids}, [0], [2], [1])
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def test_comp_meld_loader_uses_official_train_and_test(tmp_path):
    from dataloader_meld import MELDDataset

    source = tmp_path / "meld.pkl"
    _write_meld(source)
    dataset = MELDDataset(source)
    assert dataset.trainVids == {0}
    assert dataset.valVids == {1}
    assert dataset.testVids == {2}
    assert dataset.get_featDim() == (3, 4, 2)
    assert dataset[0][6].tolist() == [0.0, 1.0]


def test_comp_meld_loader_duplicates_legacy_speaker_slots(tmp_path):
    from dataloader_meld import MELDDataset

    source = tmp_path / "meld.pkl"
    _write_meld(source)
    dataset = MELDDataset(source)
    item = dataset[1]
    for host, guest in zip(item[:3], item[3:6]):
        torch.testing.assert_close(host, guest)


def test_attention_mode_controls_future_access():
    from modules.Attention import Attention

    torch.manual_seed(4)
    x = torch.randn(1, 5, 8)
    changed = x.clone()
    changed[:, 3:] += 20
    mask = torch.ones(1, 15)
    causal = Attention(8, num_heads=2, temporal_causal=True).eval()
    noncausal = Attention(8, num_heads=2, temporal_causal=False).eval()
    noncausal.load_state_dict(causal.state_dict())
    torch.testing.assert_close(causal(x, "a", mask)[:, :3], causal(changed, "a", mask)[:, :3])
    assert not torch.allclose(noncausal(x, "a", mask)[:, :3], noncausal(changed, "a", mask)[:, :3])


def test_prompt_mode_controls_prefix_prototypes():
    from model import PromptFormer

    common = dict(seq_len=5, drop_rate=0.0, device=torch.device("cpu"), no_cuda=True)
    causal = PromptFormer(5, 3, 8, 4, SimpleNamespace(**common, temporal_causal=True)).eval()
    noncausal = PromptFormer(5, 3, 8, 4, SimpleNamespace(**common, temporal_causal=False)).eval()
    noncausal.load_state_dict(causal.state_dict())
    x = torch.randn(1, 5, 8)
    changed = x.clone(); changed[:, 3:] += 20
    mask = torch.ones(1, 5)
    torch.testing.assert_close(causal(x, mask)[0][:, :3], causal(changed, mask)[0][:, :3])
    assert not torch.allclose(noncausal(x, mask)[0][:, :3], noncausal(changed, mask)[0][:, :3])
