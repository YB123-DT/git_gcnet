"""MELD frozen original Query/decision diagnostic. INTERNAL DIAGNOSTIC ONLY.

Fit B heads on original reads, then freeze their exact parameters and train C
query corrections. Source CHM/Local/Flat weights never receive an update.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score

from experiments.osram_meld_query_decision_20261014.learning import (
    LightReadout, QueryCorrector, forward_reads,
)
from experiments.osram_meld_query_decision_20261014.trace import extract_split, make_donor_indices
from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model, _sha256, _write
from experiments.osram_meld_write_retention_20261014.task_replay import validate_source
from experiments.osram_meld_write_retention_20261014.run import guard_gpu, state_digest


def module_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bind_source(source, weights, manifest, rate, checkpoint, config):
    """Bind exact weights to recovered winning masks, not merely same classes."""
    sources = json.loads(Path(manifest).read_text())['sources']
    try:
        record = sources[str(source)]['rates'][str(rate)]
    except KeyError as error:
        raise ValueError('No canonical source/rate identity in selection manifest') from error
    normalized = json.loads(json.dumps(checkpoint.get('config')))
    if (record['checkpoint_sha256'] != _sha256(weights)
            or record['epoch'] != checkpoint.get('epoch')
            or checkpoint.get('selection_rate') != rate or normalized != config):
        raise ValueError('Checkpoint identity/config/epoch differs from canonical source')
    return record


def metrics(labels, logits, num_classes=7):
    labels, logits = np.asarray(labels), np.asarray(logits)
    if len(labels) == 0 or logits.shape != (len(labels), num_classes):
        raise ValueError('Require nonempty aligned labels and logits')
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite prediction')
    predicted = logits.argmax(1)
    recalls = recall_score(labels, predicted, labels=np.arange(num_classes),
                           average=None, zero_division=0)
    shifted = logits - logits.max(1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]
    return dict(N=len(labels), ACC=100. * accuracy_score(labels, predicted),
                UA=100. * float(recalls.mean()),
                WF1=100. * f1_score(labels, predicted, average='weighted', zero_division=0),
                CE=float(ce.mean()), recall=(100. * recalls).tolist(),
                support=np.bincount(labels, minlength=num_classes).tolist())


def paired_counts(labels, before, after):
    a, b = np.asarray(before).argmax(1), np.asarray(after).argmax(1)
    y = np.asarray(labels)
    if a.shape != y.shape or b.shape != y.shape:
        raise ValueError('Paired predictions must share the same IDs and labels')
    return dict(N=len(y), corrections=int(((a != y) & (b == y)).sum()),
                harms=int(((a == y) & (b != y)).sum()),
                both_correct=int(((a == y) & (b == y)).sum()),
                both_wrong=int(((a != y) & (b != y)).sum()), flips=int((a != b).sum()))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Torch1.13 NVML reports device_count()==0 with a visible GPU UUID on
        # this host although the CUDA runtime can use logical cuda:0. Seed the
        # explicitly admitted single device, never enumerate it through NVML.
        torch.cuda.manual_seed(seed)


def read_inputs(data, ids, mode, corrector, read_ridge, residual_strength, query_capture=None):
    if mode not in ('real', 'local', 'donor'):
        raise ValueError(mode)
    if mode == 'local':
        width = data['memory'].shape[1] * data['memory'].shape[2]
        return (data['local'].new_zeros(len(ids), width),
                data['local'].new_zeros(len(ids), 3, width))
    cache = data.get('_reads_' + mode)
    if corrector is None and cache is not None:
        return cache[0][ids], cache[1][ids]
    source = ids
    if mode == 'donor':
        source = data['donors'][ids]
        if (source < 0).any():
            raise ValueError('Cannot read unmatched donor rows')
    queries = data['queries'][ids]
    if corrector is not None:
        queries = corrector(data['local'][ids], queries, data['availability'][ids])
        if query_capture is not None:
            queries.retain_grad()
            query_capture.append(queries)
    return forward_reads(data['memory'][source], queries, data['keys'][ids],
                         data['availability'][ids], read_ridge, residual_strength)


@torch.no_grad()
def predict(head, corrector, data, ids, mode, batch_size, ridge, strength):
    head.eval()
    if corrector is not None:
        corrector.eval()
    result = []
    for selected in ids.split(batch_size):
        base, gap = read_inputs(data, selected, mode, corrector, ridge, strength)
        result.append(head(data['local'][selected], base, gap, data['availability'][selected]).cpu())
    return torch.cat(result).numpy()


def fit_stage(head, corrector, train, test, train_ids, test_ids, mode, output,
              *, epochs=30, batch_size=256, seed=66, lr=1e-3, weight_decay=1e-5,
              clip=1., read_ridge=.001, residual_strength=1., num_classes=7):
    """Epoch0 included; earliest best ACC wins. Test-selected diagnostic only.

    C optimizer owns only QueryCorrector. B's normalizer, LayerNorm, decoder
    and all its gradients are immutable in C; input gradients remain enabled.
    """
    if len(train_ids) == 0 or len(test_ids) == 0 or epochs < 1 or batch_size < 1:
        raise ValueError('Require nonempty common populations and a positive budget')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    head.zero_grad(set_to_none=True)
    head.requires_grad_(corrector is None)
    owner = head if corrector is None else corrector
    owner.requires_grad_(True)
    parameters = list(owner.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    frozen_before = module_digest(head) if corrector is not None else None
    shuffle = torch.Generator().manual_seed(seed)
    settings = dict(epochs=epochs, batch_size=batch_size, seed=seed, lr=lr,
                    weight_decay=weight_decay, clip=clip, read_ridge=read_ridge,
                    residual_strength=residual_strength, mode=mode, num_classes=num_classes,
                    optimizer_owner='query_only' if corrector is not None else 'readout_only',
                    selection='earliest maximum test ACC including epoch0; INTERNAL DIAGNOSTIC ONLY')
    labels = test['labels'][test_ids].cpu().numpy()
    report = dict(settings=settings, train_N=len(train_ids), test_N=len(test_ids),
                  parameters=sum(p.numel() for p in parameters), curve=[],
                  frozen_decoder_sha256_before=frozen_before,
                  gradient=dict(steps=0, finite_steps=0, nonzero_steps=0, norm_sum=0., norm_max=0.),
                  query_gradient={name: dict(eligible_heads=0, finite_heads=0, nonzero_heads=0)
                                  for name in ('Base', 'Gap-A', 'Gap-T', 'Gap-V')})
    best_acc, best_logits = -1., None
    started = time.monotonic()
    for epoch in range(epochs + 1):
        total, updates = 0., 0
        if epoch:
            owner.train()
            if corrector is not None:
                head.eval()
            order = torch.randperm(len(train_ids), generator=shuffle).to(train_ids.device)
            for selected in train_ids[order].split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                captured = [] if corrector is not None else None
                base, gap = read_inputs(train, selected, mode, corrector, read_ridge, residual_strength, captured)
                logits = head(train['local'][selected], base, gap, train['availability'][selected])
                loss = F.cross_entropy(logits, train['labels'][selected])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                if captured:
                    query_grad = captured[0].grad
                    if query_grad is None or not torch.isfinite(query_grad).all():
                        raise RuntimeError('Missing/nonfinite corrected-query gradients')
                    sources = train['donors'][selected] if mode == 'donor' else selected
                    has_history = train['memory'][sources].flatten(1).ne(0).any(1)
                    active = torch.cat((torch.ones_like(has_history[:, None]),
                                        ~train['availability'][selected].bool()), 1) & has_history[:, None]
                    for slot, name in enumerate(('Base', 'Gap-A', 'Gap-T', 'Gap-V')):
                        norms = query_grad[:, slot].norm(dim=-1)[active[:, slot]]
                        coverage = report['query_gradient'][name]
                        coverage['eligible_heads'] += norms.numel()
                        coverage['finite_heads'] += int(torch.isfinite(norms).sum())
                        coverage['nonzero_heads'] += int((norms > 0).sum())
                gradients = [p.grad for p in parameters if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError('Missing/nonfinite owner gradients')
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip))
                stats = report['gradient']
                stats['steps'] += 1
                stats['finite_steps'] += 1
                stats['nonzero_steps'] += int(norm > 0.)
                stats['norm_sum'] += norm
                stats['norm_max'] = max(stats['norm_max'], norm)
                optimizer.step()
                if corrector is not None and any(p.grad is not None for p in head.parameters()):
                    raise RuntimeError('Frozen decoder received a gradient')
                total += float(loss.detach()) * len(selected)
                updates += len(selected)
        logits = predict(head, corrector, test, test_ids, mode, batch_size, read_ridge, residual_strength)
        score = metrics(labels, logits, num_classes)
        report['curve'].append(dict(epoch=epoch, train_CE=None if not updates else total / updates, **score))
        state = dict(epoch=epoch, readout=head.state_dict(),
                     corrector=None if corrector is None else corrector.state_dict(),
                     optimizer=optimizer.state_dict(), settings=settings,
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                     python_rng=random.getstate(), shuffle_rng=shuffle.get_state(),
                     cuda_rng=[torch.cuda.get_rng_state(0)] if torch.cuda.is_available() else [],
                     selected_test_metrics=score)
        torch.save(state, output / 'last.pt')
        if score['ACC'] > best_acc:
            best_acc, best_logits = score['ACC'], logits.copy()
            report.update(best_epoch=epoch, best=score)
            torch.save(state, output / 'best.pt')
        print(f'{output.name} epoch={epoch}/{epochs} ACC={score["ACC"]:.3f} '
              f'UA={score["UA"]:.3f} WF1={score["WF1"]:.3f}', flush=True)
        _write(output / 'SUMMARY.json', report)
    report.update(last=score, seconds=time.monotonic() - started,
                  frozen_decoder_sha256_after=module_digest(head) if corrector is not None else None,
                  frozen_decoder_unchanged=None if corrector is None else frozen_before == module_digest(head))
    if corrector is not None and not report['frozen_decoder_unchanged']:
        raise RuntimeError('C changed its frozen B decoder')
    report['gradient']['mean_norm'] = report['gradient']['norm_sum'] / max(1, report['gradient']['steps'])
    # Load storages on CPU and copy through load_state_dict: direct CUDA
    # deserialization relies on the broken Torch1.13/NVML UUID device count.
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    head.load_state_dict(checkpoint['readout'], strict=True)
    if corrector is not None:
        corrector.load_state_dict(checkpoint['corrector'], strict=True)
    _write(output / 'SUMMARY.json', report)
    return report, best_logits


def to_device(arrays, device):
    fields = ('local', 'memory', 'queries', 'keys', 'availability', 'labels')
    return {name: torch.as_tensor(arrays[name], device=device) for name in fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('source', 'selected-masks', 'dataset-root', 'meld-pickle', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--rate', type=float, required=True, choices=[i / 10 for i in range(8)])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--max-batches', type=int, help='Partial real-batch smoke ONLY')
    parser.add_argument('--selection-manifest', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.batch_size < 1 or args.seed != 66:
        parser.error('First diagnostic requires seed66 and positive budgets')
    source_config = json.loads((args.source / 'config.json').read_text())
    validate_source(source_config)
    suffix = f'{args.rate:.1f}'.replace('.', 'p')
    checkpoint_path = args.source / f'best_miss_{suffix}.pt'
    mask_path = args.selected_masks / f'predictions_miss_{suffix}.npz'
    for path in (checkpoint_path, mask_path, args.meld_pickle):
        if not path.is_file():
            raise FileNotFoundError(path)
    inventory = json.loads(Path(__file__).with_name('CANONICAL_INPUTS.json').read_text())
    mask_sha = _sha256(mask_path)
    dataset_sha = _sha256(args.meld_pickle)
    if (str(args.source) != inventory['source'] or mask_sha != inventory['mask_sha256'][str(args.rate)]
            or dataset_sha != inventory['dataset_sha256']):
        raise ValueError('Canonical mask/source/data artifact identity mismatch; no fallback')
    gpu = guard_gpu(args.gpu_uuid)
    os.environ['GCNET_DATASET_ROOT'] = str(args.dataset_root)
    os.environ['GCNET_MELD_PICKLE'] = str(args.meld_pickle)
    torch.set_num_threads(2)
    seed_all(args.seed)
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from gcnet_modality_jepa.train_gcnet import get_loaders
    config = replace(TrainConfig(**source_config), device='cuda:0')
    roots = [args.dataset_root / 'MELD/features' / name for name in
             ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    trains, _, tests, ad, td, vd = get_loaders(audio_root=str(roots[0]), text_root=str(roots[1]),
        video_root=str(roots[2]), num_folder=1, dataset='MELD', batch_size=config.batch_size,
        num_workers=0, seed=config.seed, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)
    device = torch.device('cuda:0')
    model = _build_model(config, (ad, td, vd), device)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    manifest = args.selection_manifest or args.selected_masks.parent / 'SUMMARY.json'
    source_selection = bind_source(args.source, checkpoint_path, manifest, args.rate, state, source_config)
    model.load_state_dict(state['model'], strict=True)
    source_epoch = state.get('epoch')
    del state
    source_digest = state_digest(model)
    args.output.mkdir(parents=True)
    summary = dict(status='running', protocol='INTERNAL DIAGNOSTIC ONLY; per-rate test ACC selection',
        training='small readout/query fits only; source CHM never trains', seed=args.seed,
        rate=args.rate, source=str(args.source), selected_masks=str(args.selected_masks),
        checkpoint_epoch=source_epoch, checkpoint_sha256=_sha256(checkpoint_path),
        selection_manifest_sha256=_sha256(manifest), source_selection=source_selection,
        selected_masks_sha256=mask_sha, dataset_sha256=dataset_sha,
        source_config=source_config, gpu=gpu, command=sys.argv,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_diff=subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT, text=True),
        code_sha256={str(path.relative_to(ROOT)): _sha256(path) for path in
                     (Path(__file__), Path(__file__).with_name('learning.py'), Path(__file__).with_name('trace.py'),
                      ROOT / 'gcnet_missing_m3/osram.py', ROOT / 'gcnet_missing_m3/model.py')},
        torch_version=torch.__version__, stages={}, mask_seed=args.seed + round(args.rate * 10),
        donor_seed=args.seed, state_cache='RAM/GPU only; masks/IDs/donors preserved; no disk state cache')
    _write(args.output / 'SUMMARY.json', summary)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        train_arrays, train_checks = extract_split(model, trains[0], config, (ad, td, vd),
            args.rate, 'train', mask_seed=summary['mask_seed'], max_batches=args.max_batches)
        with np.load(mask_path) as saved:
            test_arrays, test_checks = extract_split(model, tests[0], config, (ad, td, vd),
                args.rate, 'test', saved=saved, max_batches=args.max_batches)
        summary.update(trace_checks=dict(train=train_checks, test=test_checks),
                       A_full=metrics(test_arrays['labels'], test_arrays['logits']))
        if not args.max_batches and any(abs(summary['A_full'][k] - source_selection[k]) > 1e-8
                                       for k in ('ACC', 'WF1', 'UA')):
            raise RuntimeError('Winning masks do not reproduce canonical source selected scores')
        ridge = model.osram.read_ridge
        strength = 0. if model.osram.osram_gap_read == 'raw' else model.osram.gap_residual_strength
        data = {}
        populations = {}
        for split, arrays in [('train', train_arrays), ('test', test_arrays)]:
            donors, included = make_donor_indices(arrays, args.seed)
            ids = np.flatnonzero(included)
            if not len(ids):
                raise RuntimeError(f'No eligible donor population: {split}')
            metadata = {name: value for name, value in arrays.items()
                        if name not in ('memory', 'queries', 'keys', 'base', 'gap')}
            np.savez_compressed(args.output / f'{split}_metadata.npz', **metadata,
                                donors=donors, included=included)
            data[split] = to_device(arrays, device)
            data[split]['donors'] = torch.as_tensor(donors, device=device)
            populations[split] = torch.as_tensor(ids, device=device)
            summary[split + '_coverage'] = dict(total=len(donors), matched=len(ids),
                unmatched=len(donors) - len(ids), fraction=float(included.mean()))
            # Precompute original-query reads. Donor state is evaluated with
            # recipient Q/keys, not the donor's unrelated current utterance.
            for mode in ('real', 'donor'):
                base = arrays['base'].copy() if mode == 'real' else np.zeros_like(arrays['base'])
                gap = arrays['gap'].copy() if mode == 'real' else np.zeros_like(arrays['gap'])
                if mode == 'donor':
                    with torch.no_grad():
                        for selected in populations[split].split(args.batch_size):
                            b, g = read_inputs(data[split], selected, mode, None, ridge, strength)
                            base[selected.cpu().numpy()] = b.cpu().numpy()
                            gap[selected.cpu().numpy()] = g.cpu().numpy()
                data[split]['_reads_' + mode] = (torch.as_tensor(base, device=device),
                                                torch.as_tensor(gap, device=device))
        # Source stays alive for the final immutable hash but its trace arrays
        # need not occupy duplicated host RAM throughout small-head training.
        train_arrays.clear()
        ids = populations['test'].cpu().numpy()
        labels = test_arrays['labels'][ids]
        predictions = {'A': test_arrays['logits'][ids]}
        summary['A_paired'] = metrics(labels, predictions['A'])
        for mode in ('real', 'local', 'donor'):
            seed_all(args.seed)
            head = LightReadout(local_dim=data['train']['local'].shape[-1],
                                context_dim=data['train']['memory'].shape[1] * data['train']['memory'].shape[2]).to(device)
            with torch.no_grad():
                base, gap = read_inputs(data['train'], populations['train'], mode, None, ridge, strength)
                head.fit_normalizer(data['train']['local'][populations['train']], base, gap,
                                    data['train']['availability'][populations['train']])
            name = 'B_' + mode
            result, predictions[name] = fit_stage(head, None, data['train'], data['test'],
                populations['train'], populations['test'], mode, args.output / name,
                epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                read_ridge=ridge, residual_strength=strength)
            summary['stages'][name] = result
            if mode != 'local':
                seed_all(args.seed)
                query = QueryCorrector(local_dim=data['train']['local'].shape[-1],
                    key_dim=data['train']['memory'].shape[-1], num_heads=data['train']['memory'].shape[1]).to(device)
                initial = predict(head, query, data['test'], populations['test'], mode,
                                  args.batch_size, ridge, strength)
                max_initial_error = float(np.abs(initial - predictions[name]).max())
                if max_initial_error > 5e-5 or not np.array_equal(initial.argmax(1), predictions[name].argmax(1)):
                    raise RuntimeError('Zero-corrector C does not reproduce selected B')
                result, predictions['C_' + mode] = fit_stage(head, query, data['train'], data['test'],
                    populations['train'], populations['test'], mode, args.output / ('C_' + mode),
                    epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                    read_ridge=ridge, residual_strength=strength)
                result['initial_B_logit_max_error'] = max_initial_error
                with torch.no_grad():
                    q = query(data['test']['local'][populations['test']],
                              data['test']['queries'][populations['test']],
                              data['test']['availability'][populations['test']])
                    result['query_delta_norm_mean'] = float((q - data['test']['queries'][populations['test']]).norm(dim=-1).mean())
                    active = torch.cat((torch.ones(len(q), 1, device=device, dtype=torch.bool),
                                        ~data['test']['availability'][populations['test']].bool()), 1)
                    delta = (q - data['test']['queries'][populations['test']]).norm(dim=-1)
                    result['query_delta_by_slot'] = {name: float(delta[:, slot][active[:, slot]].mean())
                        if active[:, slot].any() else None for slot, name in enumerate(('Base', 'Gap-A', 'Gap-T', 'Gap-V'))}
                summary['stages']['C_' + mode] = result
            _write(args.output / 'SUMMARY.json', summary)
        comparisons = [('A', 'B_real'), ('B_local', 'B_real'), ('B_donor', 'B_real'),
                       ('B_real', 'C_real'), ('B_donor', 'C_donor'), ('C_donor', 'C_real'), ('A', 'C_real')]
        summary['paired_changes'] = {a + '->' + b: paired_counts(labels, predictions[a], predictions[b])
                                     for a, b in comparisons}
        summary['availability_groups'] = {}
        av = test_arrays['availability'][ids].astype(int)
        for pattern in np.unique(av, axis=0):
            selected = np.all(av == pattern, axis=1)
            name = ''.join(m for m, keep in zip('ATV', pattern) if keep)
            summary['availability_groups'][name] = dict(N=int(selected.sum()),
                metrics={k: metrics(labels[selected], v[selected]) for k, v in predictions.items()},
                changes={a + '->' + b: paired_counts(labels[selected], predictions[a][selected], predictions[b][selected])
                         for a, b in comparisons})
        np.savez_compressed(args.output / 'predictions.npz', labels=labels,
            conversation_ids=test_arrays['conversation_ids'][ids], utterance_indices=test_arrays['utterance_indices'][ids],
            availability=av, included_indices=ids, **predictions)
        if state_digest(model) != source_digest or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Source CHM changed during diagnostic fits')
        summary.update(status='partial_smoke' if args.max_batches else 'complete',
                       source_unchanged=True, source_state_sha256=source_digest)
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        summary.update(seconds=time.monotonic() - started,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        _write(args.output / 'SUMMARY.json', summary)


if __name__ == '__main__':
    main()
