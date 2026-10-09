"""Frozen-memory diagnostic probes; labels are used only as regression targets."""
import json
from pathlib import Path

import numpy as np


def active_slots(data):
    history = np.asarray(data['utterance_indices']) > 0
    return np.column_stack((history, (1 - data['availability']) * history[:, None])).astype(bool)


def fit_normalizer(train):
    local = np.asarray(train['local'], dtype=np.float32)
    memory = np.asarray(train['memory'], dtype=np.float32)
    active = active_slots(train)
    mean = np.zeros((4, 512), dtype=np.float32)
    scale = np.ones_like(mean)
    for slot in range(4):
        values = memory[active[:, slot], slot]
        if len(values):
            mean[slot] = values.mean(0)
            scale[slot] = np.maximum(values.std(0), 1e-6)
    return dict(local_mean=local.mean(0), local_scale=np.maximum(local.std(0), 1e-6),
                memory_mean=mean, memory_scale=scale)


def transform(data, normalizer):
    local = (data['local'] - normalizer['local_mean']) / normalizer['local_scale']
    memory = (data['memory'] - normalizer['memory_mean']) / normalizer['memory_scale']
    memory = np.where(active_slots(data)[..., None], memory, 0)
    return local.astype(np.float32), memory.astype(np.float32)


def choose_donors(train, target, seed):
    """Indices reference TRAIN exclusively; -1 denotes retained zero first-turn memory."""
    rng = np.random.default_rng(seed)
    donors = np.full(len(target['labels']), -1, dtype=np.int64)
    keep = np.ones(len(donors), dtype=bool)
    train_history = train['utterance_indices'] > 0
    for i in range(len(donors)):
        if target['utterance_indices'][i] == 0:
            continue
        eligible = (train_history & np.all(train['availability'] == target['availability'][i], axis=1)
                    & (train['conversation_ids'] != target['conversation_ids'][i]))
        candidates = np.flatnonzero(eligible)
        if len(candidates):
            donors[i] = rng.choice(candidates)
        else:
            keep[i] = False
    return donors, keep


def make_targets(data):
    labels = np.asarray(data['labels'], dtype=np.float32)
    previous = np.full(len(labels), np.nan, dtype=np.float32)
    mean3 = previous.copy()
    lookup = {(str(c), int(t)): float(y) for c, t, y in
              zip(data['conversation_ids'], data['utterance_indices'], labels)}
    if len(lookup) != len(labels):
        raise ValueError('Duplicate conversation/utterance IDs')
    for i, (conversation, position) in enumerate(zip(data['conversation_ids'], data['utterance_indices'])):
        position = int(position)
        if position > 0:
            previous[i] = lookup.get((str(conversation), position - 1), np.nan)
            values = [lookup.get((str(conversation), t), np.nan)
                      for t in range(max(0, position - 3), position)]
            mean3[i] = np.mean(values)
    return dict(current=labels.copy(), previous=previous, preceding3_mean=mean3,
                current_minus_previous=labels - previous)


def metrics(labels, predictions):
    result = {'n': int(len(labels)), 'mse': float(np.mean((labels - predictions) ** 2))}
    valid = labels != 0
    truth, predicted = labels[valid] > 0, predictions[valid] > 0
    result['nonneutral_n'] = int(valid.sum())
    result['acc_nonzero'] = float(np.mean(truth == predicted)) if len(truth) else None
    weighted = 0.
    for value in (False, True):
        tp = int(np.sum((truth == value) & (predicted == value)))
        fp = int(np.sum((truth != value) & (predicted == value)))
        fn = int(np.sum((truth == value) & (predicted != value)))
        score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        weighted += score * int(np.sum(truth == value))
    result['weighted_f1_nonzero'] = weighted / len(truth) if len(truth) else None
    return result


def _fit(features, targets, hidden, seed, epochs, device, destination):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    model = nn.Sequential(nn.Linear(features['train'].shape[1], hidden), nn.GELU(), nn.Linear(hidden, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    x = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in features.items()}
    y = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in targets.items()}
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, state = float('inf'), None, None
    curve = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x['train']), generator=generator).to(device)
        for indices in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(x['train'][indices]).squeeze(-1) - y['train'][indices]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite probe training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Non-finite probe gradient')
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(((model(x['test']).squeeze(-1) - y['test']) ** 2).mean())
        curve.append(val)
        if np.isfinite(val) and val < best:
            best, best_epoch = val, epoch + 1
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if state is None:
        raise ValueError('No finite test-selection checkpoint')
    model.load_state_dict(state)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / 'best.pt'
    torch.save({'state_dict': state, 'epoch': best_epoch, 'selection_mse': best, 'selection_split': 'test',
                'input_dim': features['train'].shape[1], 'hidden_dim': hidden, 'seed': seed}, weights_path)
    with torch.no_grad():
        predictions = {k: model(v).squeeze(-1).cpu().numpy() for k, v in x.items()}
    return dict(best_epoch=best_epoch, selection_split='test', test_selection_curve=curve, weights=str(weights_path),
                parameter_count=sum(p.numel() for p in model.parameters())), predictions


def run_probes(splits, output: Path, device='cpu', seeds=(66, 67, 68), epochs=100):
    """TEST-ORACLE INTERNAL DIAGNOSTIC: test-selected epochs are not independent evaluation."""
    import torch

    if epochs < 1:
        raise ValueError('epochs must be positive')
    torch.set_num_threads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ('train', 'test')
    overlap = set(map(str, splits['train']['conversation_ids'])) & set(map(str, splits['test']['conversation_ids']))
    if overlap:
        raise ValueError(f'Train/test conversation overlap: {len(overlap)} IDs')
    for name in names:
        data = splits[name]
        n = len(data['labels'])
        if n == 0:
            raise ValueError(f'{name}: empty split')
        for key in ('labels', 'conversation_ids', 'utterance_indices', 'speaker', 'prediction'):
            if np.asarray(data[key]).shape != (n,):
                raise ValueError(f'{name}/{key}: expected one scalar per row')
        for key, shape in [('local', (n, 256)), ('memory', (n, 4, 512)), ('availability', (n, 3))]:
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f'{name}/{key}: invalid shape or nonfinite features')
        if (not np.isin(data['availability'], [0, 1]).all()
                or not np.all(data['availability'].sum(axis=1) > 0)
                or not np.isfinite(data['labels']).all()):
            raise ValueError(f'{name}: invalid availability or labels')
    normalizer = fit_normalizer(splits['train'])
    projection = (np.random.default_rng(66).normal(size=(512, 64)) / np.sqrt(512)).astype(np.float32)
    np.savez(output / 'preprocessing.npz', projection=projection, **normalizer)
    processed = {name: transform(splits[name], normalizer) for name in names}
    targets = {name: make_targets(splits[name]) for name in names}
    summary = dict(parameter_counts={'A': 33409, 'B': 33089, 'C': 33089},
                   protocol=dict(seeds=list(seeds), epochs=epochs, optimizer='Adam', lr=1e-3,
                                 weight_decay=1e-5, batch_size=128, projection_seed=66,
                                 selection='minimum test MSE; all target labels', selection_split='test',
                                 caveat='TEST-ORACLE INTERNAL DIAGNOSTIC: test selection is not independent generalization; gradients use train only',
                                 slot_order=['Base', 'Gap-A', 'Gap-T', 'Gap-V'],
                                 normalization='training-only active-slot zscore, inactive slots re-zeroed',
                                 donor_policy='train only; same availability/has-history; other conversation; replacement',
                                 budget_relative_difference=(33409 - 33089) / 33409),
                   coverage={}, runs=[])
    for seed in seeds:
        donor_record, coverage, features = {}, {}, {p: {} for p in ('A', 'B', 'C')}
        masks = {}
        for split_number, name in enumerate(names):
            data = splits[name]
            donors, keep = choose_donors(splits['train'], data, seed + 1009 * split_number)
            masks[name] = keep
            local, memory = processed[name]
            shuffled = np.zeros_like(memory)
            valid = donors >= 0
            shuffled[valid] = processed['train'][1][donors[valid]]
            availability = data['availability'].astype(np.float32)
            features['A'][name] = np.concatenate((local, availability), axis=1)
            for probe, source in [('B', memory), ('C', shuffled)]:
                features[probe][name] = np.concatenate((local, (source @ projection).reshape(len(local), 256), availability), axis=1)
            coverage[name] = dict(total=len(keep), included=int(keep.sum()), excluded=int((~keep).sum()))
            donor_record[name] = [dict(target_conversation=str(data['conversation_ids'][i]),
                                      target_utterance=int(data['utterance_indices'][i]),
                                      included=bool(keep[i]), donor_train_row=int(j),
                                      donor_conversation=str(splits['train']['conversation_ids'][j]) if j >= 0 else None,
                                      donor_utterance=int(splits['train']['utterance_indices'][j]) if j >= 0 else None)
                                  for i, j in enumerate(donors)]
        (output / f'donors_seed{seed}.json').write_text(json.dumps(donor_record, indent=2))
        summary['coverage'][str(seed)] = coverage
        for task in targets['train']:
            selected = {name: masks[name] & np.isfinite(targets[name][task]) for name in names}
            task_counts = {name: int(selected[name].sum()) for name in names}
            if not all(task_counts.values()):
                summary['runs'].append(dict(seed=seed, task=task, status='skipped_empty_split', coverage=task_counts))
                continue
            target = {name: targets[name][task][selected[name]] for name in names}
            for probe in ('A', 'B', 'C'):
                feature = {name: features[probe][name][selected[name]] for name in names}
                destination = output / f'seed{seed}' / task / probe
                run, predictions = _fit(feature, target, 128 if probe == 'A' else 64,
                                        seed, epochs, device, destination)
                arrays = {}
                for name in names:
                    arrays.update({f'{name}_prediction': predictions[name], f'{name}_target': target[name],
                                   f'{name}_row': np.flatnonzero(selected[name]),
                                   f'{name}_conversation': splits[name]['conversation_ids'][selected[name]].astype(str),
                                   f'{name}_utterance': splits[name]['utterance_indices'][selected[name]]})
                prediction_path = destination / 'predictions.npz'
                np.savez_compressed(prediction_path, **arrays)
                run.update(seed=seed, task=task, probe=probe, coverage=task_counts,
                           predictions=str(prediction_path),
                           **{name: metrics(target[name], predictions[name]) for name in names})
                summary['runs'].append(run)
                (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
            print(f'Probe seed={seed} task={task}: A/B/C complete (test-oracle selection)', flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary
