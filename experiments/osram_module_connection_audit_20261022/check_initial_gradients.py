"""CPU topology check only; random synthetic latents, no data or checkpoint.

Exercise the real OSRAM+V3 forward to distinguish a delayed gradient from a
disconnected branch. Four ephemeral Adam updates are NOT a task experiment.
"""
import argparse
import json
from pathlib import Path

import torch

from gcnet_missing_m3.osram import OSRAMBackbone


def norm(parameter):
    return None if parameter.grad is None else float(parameter.grad.norm())


def check():
    torch.set_num_threads(1)
    torch.manual_seed(66)
    model = OSRAMBackbone(
        latent_dim=256, output_dim=1600, num_heads=8, key_dim=64,
        value_dim=64, n_speakers=1, dropout=0., bidirectional=False,
        write_step=.6, osram_source_role_v3=True,
    )
    head = torch.nn.Linear(1600, 1)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=3e-4,
        weight_decay=0.,
    )
    length, batch, dim = 4, 3, 256
    availability = torch.eye(3).unsqueeze(0).repeat(length, 1, 1)
    availability[0] = 1
    node = torch.randn(length, batch, dim)
    latents = {name: torch.randn(length, batch, dim)
               for name in ('audio', 'text', 'visual')}
    speaker = torch.zeros(batch, length, dtype=torch.long)
    valid = torch.ones(batch, length)
    labels = torch.randn(length, batch, 1)
    watched = {
        'flat_final': model.emotion_adapter[-1].weight,
        'v3_base_decoder': model.source_role_v3.decoders[0].weight,
        'v3_pair_mlp': model.source_role_v3.pair_mlp[1].weight,
        'v3_attention': model.source_role_v3.source_attention.in_proj_weight,
        'v3_query': model.source_role_v3.role_queries[0].weight,
        'writer_value_audio': model.value_projectors['audio'].weight,
    }
    rows = []
    for step in range(1, 5):
        optimizer.zero_grad(set_to_none=True)
        hidden, _ = model(node, latents, availability, speaker, valid, [length] * batch)
        loss = (head(hidden) - labels).square().mean()
        loss.backward()
        gradients = {name: norm(parameter) for name, parameter in watched.items()}
        assert all(value is not None and torch.isfinite(torch.tensor(value))
                   for value in gradients.values())
        rows.append({'backward': step, 'loss': float(loss.detach()),
                     'gradient_norms': gradients})
        optimizer.step()
    assert rows[0]['gradient_norms']['flat_final'] > 0
    assert rows[0]['gradient_norms']['v3_base_decoder'] == 0
    assert rows[1]['gradient_norms']['v3_base_decoder'] > 0
    assert rows[1]['gradient_norms']['v3_pair_mlp'] == 0
    assert rows[2]['gradient_norms']['v3_pair_mlp'] > 0
    return {
        'scope': 'Synthetic CPU topology check, not performance or training-trajectory evidence',
        'torch_version': torch.__version__, 'seed': 66,
        'differences_from_experiment': ['synthetic latents/labels', 'dropout=0',
                                        'weight_decay=0', 'four ephemeral updates'],
        'flat_adapter_parameters': sum(p.numel() for p in model.emotion_adapter.parameters()),
        'v3_parameters': sum(p.numel() for p in model.source_role_v3.parameters()),
        'steps': rows,
        'conclusion': 'Stacked zero-initialized layers delay gradients; they do not permanently disconnect V3.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    payload = json.dumps(check(), indent=2, ensure_ascii=False) + '\n'
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(payload)
    print(payload, end='')
