"""Build the shared frozen PAM-E utility cache for one seed/rate/split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_utility_text_retrieval_20260918.utility_common import (
    build_reader, query_utility_from_batch,
)

ROOT = Path('/data2/yb/remote_experiments/osram_utility_text_retrieval_20260918')


def rate_tag(rate: float) -> str:
    return f'{rate:.1f}'.replace('.', 'p')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'utility_cache')
    args = parser.parse_args()
    device = torch.device(args.device)
    model, cfg, dims, shape, loaders = build_reader(args.seed, args.rate, device)
    if args.split == 'train':
        loader = loaders[0][cfg.fold - 1]
    elif args.split == 'validation':
        loader = loaders[1][cfg.fold - 1]
    else:
        loader = loaders[2][cfg.fold - 1]
    schedule = tr._build_schedule(cfg, args.split, args.rate)
    pieces = []
    for batch_index, raw in enumerate(loader):
        data = tr._move_batch(raw, device)
        view = tr._prepare_view(data, schedule, 0, dims)
        with torch.no_grad():
            out = query_utility_from_batch(model, cfg, view, batch_index)
        if out is not None:
            pieces.append(out)
    if pieces:
        max_c = max(int(piece['values'].shape[1]) for piece in pieces)
        padded = []
        for piece in pieces:
            item = {}
            for key, value in piece.items():
                if key in {'values', 'mask', 'scores', 'losses', 'utility'}:
                    if value.shape[1] < max_c:
                        pad_shape = (value.shape[0], max_c - value.shape[1]) + tuple(value.shape[2:])
                        pad = torch.zeros(pad_shape, dtype=value.dtype)
                        if key == 'mask':
                            pad = pad.to(dtype=torch.bool)
                        value = torch.cat((value, pad), dim=1)
                item[key] = value
            padded.append(item)
        merged = {
            key: torch.cat([piece[key] for piece in padded], dim=0)
            for key in padded[0]
        }
    else:
        merged = {
            'context': torch.zeros(0, 1),
            'values': torch.zeros(0, 1, 1),
            'mask': torch.zeros(0, 1, dtype=torch.bool),
            'scores': torch.zeros(0, 1),
            'losses': torch.zeros(0, 1),
            'utility': torch.zeros(0, 1),
            'labels': torch.zeros(0),
            'patterns': torch.zeros(0, 3),
        }
    output = args.output_root / f'seed_{args.seed}' / f'rate_{rate_tag(args.rate)}' / f'{args.split}.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'tensors': merged,
        'seed': args.seed,
        'rate': args.rate,
        'split': args.split,
        'n_queries': int(merged['labels'].shape[0]),
        'max_candidates': int(merged['values'].shape[1]) if merged['values'].ndim == 3 else 0,
        'reader_checkpoint': str(__import__('experiments.osram_utility_text_retrieval_20260918.utility_common', fromlist=['reader_path']).reader_path(args.seed, f'{args.rate:.1f}')),
    }, output)
    print(json.dumps({'output': str(output), 'n_queries': int(merged['labels'].shape[0])}))


if __name__ == '__main__':
    main()
