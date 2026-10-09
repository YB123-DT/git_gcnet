"""Frozen cached OSRAM decision correction; train-only updates, fixed last epoch.

Source backbones and historical probes were test-oracle selected. All resulting
scores remain INTERNAL DIAGNOSTIC ONLY, including fixed-epoch residual scores.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('audit_probe', ROOT / 'experiments/osram_frozen_memory_audit_20261009/probe.py')
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)
GROUPS = ('A_local', 'A_donor', 'A_real', 'B_local_history', 'B_donor_history', 'B_real_history', 'B_gold_history')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ResidualHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, original):
        return original + self.net(features).squeeze(-1)


def residual_features(original, local, availability, extra, has_history=None):
    parts = [np.asarray(original)[:, None], local, availability]
    if has_history is not None:
        parts.append(np.asarray(has_history)[:, None])
    parts.append(extra)
    result = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite residual inputs')
    return result


def donor_memory(train_memory, donors, target):
    result = np.zeros((len(donors), 4, 512), dtype=np.float32)
    valid = donors >= 0
    result[valid] = train_memory[donors[valid]]
    return np.where(audit.active_slots(target)[..., None], result, 0)


def paired_metrics(labels, prediction, original):
    result = audit.metrics(labels, prediction)
    valid = labels != 0
    old_correct = (original > 0) == (labels > 0)
    new_correct = (prediction > 0) == (labels > 0)
    result.update(corrections=int(np.sum(valid & ~old_correct & new_correct)),
                  harms=int(np.sum(valid & old_correct & ~new_correct)))
    return result


def fit_head(features, offsets, labels, seed, epochs, device, destination):
    """Test labels enter no optimizer computation and never select an epoch."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'last.pt').exists():
        raise FileExistsError('Refusing to overwrite existing head: ' + str(destination))
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    model = ResidualHead(features['train'].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    convert = lambda values: {k: torch.as_tensor(v, dtype=torch.float32, device=device).detach() for k, v in values.items()}
    x, y0, targets = convert(features), convert(offsets), convert(labels)
    with torch.no_grad():
        for split in x:
            if not torch.equal(model(x[split], y0[split]), y0[split]):
                raise AssertionError('Zero-init differs from original prediction')
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        for ids in torch.randperm(len(x['train']), generator=generator).split(128):
            ids = ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x['train'][ids], y0['train'][ids]) - targets['train'][ids]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = {s: model(x[s], y0[s]).cpu().numpy() for s in x}
        if not all(np.isfinite(value).all() for value in prediction.values()):
            raise ValueError('Nonfinite residual prediction')
        curve.append(dict(epoch=epoch, **{s: paired_metrics(labels[s], prediction[s], offsets[s]) for s in x}))
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(dict(state_dict=state, optimizer=optimizer.state_dict(), epoch=epoch,
                        input_dim=x['train'].shape[1], hidden_dim=32, seed=seed,
                        selection='fixed_last_epoch', gradient_split='train',
                        torch_rng=torch.get_rng_state(), permutation_rng=generator.get_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None), destination / 'last.pt')
        dump(destination / 'curve.json', curve)
    return state, prediction


def load_probe(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = nn.Sequential(nn.Linear(checkpoint['input_dim'], checkpoint['hidden_dim']), nn.GELU(), nn.Linear(checkpoint['hidden_dim'], 1))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return model


def decode(model, local, availability, memory=None, projection=None):
    parts = [local]
    if memory is not None:
        parts.append((memory @ projection).reshape(len(local), 256))
    parts.append(availability)
    with torch.no_grad():
        return model(torch.as_tensor(np.concatenate(parts, axis=1), dtype=torch.float32)).squeeze(-1).numpy()


def prepare(source, output):
    """Reuse exact frozen audit preprocessing, donors and fixed seed66 probes."""
    files = [source / f'{s}_features.npz' for s in ('train', 'test')]
    files += [source / 'RESULT.json', source / 'probes/preprocessing.npz', source / 'probes/donors_seed66.json']
    for kind in ('A', 'B'):
        files += [source / f'probes/seed66/preceding3_mean/{kind}/{name}' for name in ('best.pt', 'predictions.npz')]
    hashes = {str(path): sha(path) for path in files}
    splits = {s: dict(np.load(source / f'{s}_features.npz', allow_pickle=False)) for s in ('train', 'test')}
    if set(splits['train']['conversation_ids']) & set(splits['test']['conversation_ids']):
        raise ValueError('Train/test conversation overlap')
    normal = dict(np.load(source / 'probes/preprocessing.npz', allow_pickle=False))
    projection = normal['projection']
    processed = {s: audit.transform(data, normal) for s, data in splits.items()}
    donor_record = json.loads((source / 'probes/donors_seed66.json').read_text())
    probes = {kind: load_probe(source / f'probes/seed66/preceding3_mean/{kind}/best.pt') for kind in ('A', 'B')}
    features, masks, histories, parity = {}, {}, {}, {}
    for split, data in splits.items():
        local, memory = processed[split]
        records = donor_record[split]
        if len(records) != len(local):
            raise ValueError('Donor record length mismatch')
        donors = np.array([r['donor_train_row'] for r in records])
        keep = np.array([r['included'] for r in records], dtype=bool)
        history = data['utterance_indices'] > 0
        for i, record in enumerate(records):
            if str(data['conversation_ids'][i]) != record['target_conversation'] or int(data['utterance_indices'][i]) != record['target_utterance']:
                raise ValueError('Donor target ID mismatch')
            if keep[i] and history[i]:
                j = donors[i]
                train = splits['train']
                if j < 0 or train['conversation_ids'][j] == data['conversation_ids'][i] or train['utterance_indices'][j] <= 0 or not np.array_equal(train['availability'][j], data['availability'][i]):
                    raise ValueError('Invalid matched training donor')
        shuffled = donor_memory(processed['train'][1], donors, data)
        availability = data['availability'].astype(np.float32)
        histories[split] = dict(local=decode(probes['A'], local, availability),
                                real=decode(probes['B'], local, availability, memory, projection),
                                donor=decode(probes['B'], local, availability, shuffled, projection),
                                gold=audit.make_targets(data)['preceding3_mean'])
        for kind, key in [('A', 'local'), ('B', 'real')]:
            prior = np.load(source / f'probes/seed66/preceding3_mean/{kind}/predictions.npz', allow_pickle=False)
            rows = prior[f'{split}_row']
            if (not np.issubdtype(rows.dtype, np.integer) or rows.ndim != 1
                    or len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= len(local)).any()):
                raise AssertionError('Invalid saved historical probe row indices')
            if (not np.array_equal(prior[f'{split}_conversation'].astype(str), data['conversation_ids'][rows].astype(str))
                    or not np.array_equal(prior[f'{split}_utterance'], data['utterance_indices'][rows])
                    or not np.array_equal(rows, np.flatnonzero(keep & history))
                    or not np.allclose(prior[f'{split}_target'], histories[split]['gold'][rows], atol=1e-7, rtol=0)):
                raise AssertionError('Historical probe sample IDs/targets differ from cache')
            error = float(np.max(np.abs(histories[split][key][rows] - prior[f'{split}_prediction'])))
            parity[f'{split}_{kind}'] = error
            if error > 1e-5:
                raise AssertionError('Frozen historical probe parity failure')
        if not all(np.isfinite(v[keep & history]).all() for v in histories[split].values()):
            raise ValueError('Missing historical targets or predictions')
        features[split] = {}
        for group, extra in [('A_local', np.zeros((len(local), 256))), ('A_donor', (shuffled @ projection).reshape(len(local), 256)), ('A_real', (memory @ projection).reshape(len(local), 256))]:
            features[split][group] = residual_features(data['prediction'], local, availability, extra)
        masks[split] = keep
    selected = masks['train'] & (splits['train']['utterance_indices'] > 0)
    scalar_mean = float(histories['train']['real'][selected].mean())
    scalar_scale = max(float(histories['train']['real'][selected].std()), 1e-6)
    for split, data in splits.items():
        history = data['utterance_indices'] > 0
        for key in ('local', 'donor', 'real', 'gold'):
            value = np.where(history, (histories[split][key] - scalar_mean) / scalar_scale, 0).astype(np.float32)
            features[split][f'B_{key}_history'] = residual_features(data['prediction'], processed[split][0], data['availability'], value[:, None], history)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'PREPROCESSING.npz', **normal, history_mean=scalar_mean, history_scale=scalar_scale)
    dump(output / 'donors.json', donor_record)
    dump(output / 'PROVENANCE.json', dict(source=str(source), input_sha256=hashes, parity_max_abs=parity,
        fixed_probe_seed=66, source_selection='test-oracle', normalization='reused training-only audit; shared B scalar normalization from train real-history',
        coverage={s: dict(total=len(m), included=int(m.sum())) for s, m in masks.items()}))
    return splits, features, masks, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rates', type=float, nargs='+', default=[i / 10 for i in range(8)])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[66, 67, 68])
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('Positive epochs required')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / 'STATUS.json', dict(status='running', rates=args.rates, seeds=args.seeds, epochs=args.epochs))
    summary = dict(protocol=dict(caveat='INTERNAL DIAGNOSTIC ONLY; source models test-oracle',
        selection='fixed_last_epoch', epochs=args.epochs, seeds=args.seeds, rates=args.rates,
        hidden_dim=32, optimizer='Adam', lr=.001, weight_decay=1e-5, batch_size=128,
        loss='original MOSI MSE', gradient_split='train', fixed_probe_seed=66,
        gold='nondeployable diagnostic comparator, not guaranteed numerical upper bound'), runs=[])
    for rate in args.rates:
        tag = f'rate_{rate:.1f}'.replace('.', 'p')
        candidates = [p for p in args.audit_root.rglob(tag) if (p / 'train_features.npz').exists()]
        if len(candidates) != 1:
            raise ValueError(f'Expected unique audit source for {tag}, found {candidates}')
        destination = args.output / tag
        splits, features, masks, hashes = prepare(candidates[0], destination)
        offsets = {s: data['prediction'][masks[s]].astype(np.float32) for s, data in splits.items()}
        labels = {s: data['labels'][masks[s]].astype(np.float32) for s, data in splits.items()}
        for seed in args.seeds:
            for group in GROUPS:
                folder = destination / f'seed{seed}' / group
                result_file = folder / 'RESULT.json'
                if result_file.exists():
                    record = json.loads(result_file.read_text())
                    if record['epoch'] != args.epochs or record['input_sha256'] != hashes:
                        raise ValueError('Completed run provenance/config mismatch')
                    summary['runs'].append(record)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                dump(folder / 'STATUS.json', dict(status='running', rate=rate, seed=seed, group=group))
                x = {s: features[s][group][masks[s]] for s in splits}
                state, predictions = fit_head(x, offsets, labels, seed, args.epochs, args.device, folder)
                arrays = {}
                for split, data in splits.items():
                    arrays.update({f'{split}_prediction': predictions[split], f'{split}_original': offsets[split],
                        f'{split}_target': labels[split], f'{split}_row': np.flatnonzero(masks[split]),
                        f'{split}_conversation': data['conversation_ids'][masks[split]].astype(str),
                        f'{split}_utterance': data['utterance_indices'][masks[split]]})
                np.savez_compressed(folder / 'predictions.npz', **arrays)
                record = dict(rate=rate, seed=seed, group=group, epoch=args.epochs, selection='fixed_last_epoch',
                    input_dim=x['train'].shape[1], parameter_count=sum(v.numel() for v in state.values()),
                    input_sha256=hashes, train_only_updates=True,
                    **{s: paired_metrics(labels[s], predictions[s], offsets[s]) for s in splits},
                    **{f'baseline_{s}': paired_metrics(labels[s], offsets[s], offsets[s]) for s in splits})
                dump(result_file, record)
                dump(folder / 'STATUS.json', dict(status='completed', epoch=args.epochs))
                summary['runs'].append(record)
                dump(args.output / 'SUMMARY.json', summary)
                print(f'{tag} seed={seed} {group}: WF1={record["test"]["weighted_f1_nonzero"] * 100:.4f}', flush=True)
        if any(sha(path) != expected for path, expected in hashes.items()):
            raise AssertionError('Frozen source artifacts changed during residual training')
    dump(args.output / 'SUMMARY.json', summary)
    dump(args.output / 'STATUS.json', dict(status='completed', runs=len(summary['runs'])))


if __name__ == '__main__':
    main()
