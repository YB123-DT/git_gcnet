"""Frozen paired MELD seed66 sensitivity; no fitting or mask generation.

Large utterance and spectral artifacts belong in the remote output directory.
The driver is current; numerical model/loader packages come from b4a3138.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HISTORICAL_COMMIT = 'b4a313841f098487a6482d2712bdac94aa8e319e'
FEATURE_SHA256 = 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'
RATES = (0., .4, .7)
CONDITIONS = {'baseline': {}, **{f'beta_{v:g}': {'beta': v} for v in (.05, .5, .95)},
              **{f'alpha_{v:g}': {'alpha': v} for v in (.95, .98, .995)}}


def sample_rows(ids, umask):
    if len(ids) != umask.shape[0] or umask.ndim != 2:
        raise ValueError('Conversation IDs must match [B,L] utterance mask')
    return [(str(ids[b]), t) for b in range(len(ids))
            for t in range(umask.shape[1]) if bool(umask[b, t])]


def history_bin(prior):
    return ('first' if prior == 0 else '1-3' if prior < 4 else
            '4-7' if prior < 8 else '8-15' if prior < 16 else '16+')


def metrics(labels, logits, baseline_predictions):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    labels, logits = np.asarray(labels), np.asarray(logits)
    if logits.shape != (len(labels), 7) or not np.isfinite(logits).all() or not len(labels):
        raise ValueError('Require finite aligned seven-class logits')
    pred = logits.argmax(-1)
    ce = F.cross_entropy(torch.as_tensor(logits), torch.as_tensor(labels).long()).item()
    correct, original = pred == labels, np.asarray(baseline_predictions) == labels
    return dict(N=len(labels), accuracy=float(accuracy_score(labels, pred)),
                unweighted_accuracy=float(recall_score(labels, pred, labels=list(range(7)),
                    average='macro', zero_division=0)),
                weighted_f1=float(f1_score(labels, pred, average='weighted', zero_division=0)),
                cross_entropy=float(ce), corrections=int((correct & ~original).sum()),
                harms=int((~correct & original).sum()),
                prediction_changes=int((pred != baseline_predictions).sum()),
                class_recall=recall_score(labels, pred, labels=list(range(7)), average=None,
                    zero_division=0).astype(float).tolist(),
                class_support=np.bincount(labels.astype(int), minlength=7).tolist())


def readout_logits(model, local, flat_input, valid):
    osram = model.osram
    hidden = osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(flat_input))
    return model.smax_fc(torch.where(valid[..., None], hidden, torch.zeros_like(hidden)))


def relative_per_head(value, reference):
    """Euclidean relative change; denominator floor is recorded in metadata."""
    return (value - reference).flatten(2).norm(dim=-1) / reference.flatten(2).norm(dim=-1).clamp_min(1e-12)


def write_json(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + '\n')


def load_runtime(model_source):
    # Import driver surfaces while this snapshot has priority. These helpers
    # defer numerical model/loader imports until invoked below.
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model, _sha256, _write
    from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture, validate_source, replay_task_logits
    from experiments.osram_meld_write_retention_20261014.run import WriteCapture, guard_gpu, state_digest
    from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
        scatter_saved_availability, replay_view_from_availability)
    from experiments.osram_meld_memory_sensitivity_20261017.sensitivity import (
        parameter_override, mask_head, spectral_rows, alpha_forks)
    for prefix in ('gcnet_missing_m3', 'gcnet_modality_jepa'):
        if any(name == prefix or name.startswith(prefix + '.') for name in sys.modules):
            raise RuntimeError('Run in a fresh process: numerical packages were already imported')
    sys.path.insert(0, str(model_source))
    numerical = {name: importlib.import_module(name) for name in (
        'gcnet_missing_m3.train_gcnet', 'gcnet_missing_m3.model',
        'gcnet_missing_m3.osram', 'gcnet_modality_jepa.train_gcnet')}
    for module in numerical.values():
        if not Path(module.__file__).resolve().is_relative_to(model_source.resolve()):
            raise RuntimeError(f'Historical numerical import escaped source: {module.__file__}')
    return dict(build=_build_model, sha=_sha256, write=_write, capture=TaskCapture,
                validate=validate_source, writer=WriteCapture, gpu=guard_gpu, digest=state_digest,
                scatter=scatter_saved_availability, view=replay_view_from_availability,
                override=parameter_override, mask_head=mask_head, spectral=spectral_rows,
                alpha_forks=alpha_forks, task_logits=replay_task_logits,
                train=numerical['gcnet_missing_m3.train_gcnet'],
                loaders=numerical['gcnet_modality_jepa.train_gcnet'].get_loaders,
                code_sha256={str(path): _sha256(path) for path in (
                    Path(__file__), Path(__file__).with_name('sensitivity.py'),
                    ROOT / 'experiments/osram_meld_write_retention_20261014/task_replay.py',
                    ROOT / 'experiments/osram_meld_write_retention_20261014/run.py',
                    ROOT / 'experiments/osram_meld_flat_source_combinations_20261013/evaluate.py',
                    ROOT / 'experiments/osram_meld_gcnet_random_no_skip_evidence_20261014/evaluate.py',
                    *[Path(module.__file__) for module in numerical.values()],
                    model_source / 'config.py')},
                numerical_files={name: module.__file__ for name, module in numerical.items()})


def capture_forward(model, view, runtime):
    with runtime['capture'](model.osram) as capture, runtime['writer'](model.osram, view['umask']) as writer:
        logits, _, _, completion = model([view['incomplete']], view['availability'], view['qmask'],
            view['umask'], view['lengths'], predict_missing=False)
    if completion is not None or capture.calls != 1 or capture.flat.local_calls != 1 or capture.flat.adapter_calls != 1:
        raise RuntimeError('Expected one unchanged causal Flat forward')
    if len(writer.snapshots) != len(writer.times):
        raise RuntimeError('Incomplete causal write capture')
    return logits, capture, writer.snapshots


def assert_current_inputs_same(current, baseline, writes, original_writes):
    for name in ('local', 'queries', 'availability', 'valid'):
        if not torch.equal(getattr(current, name), getattr(baseline, name)):
            raise RuntimeError(f'Parameter intervention changed fixed current {name}')
    if len(writes) != len(original_writes):
        raise RuntimeError('Parameter intervention changed scan length')
    for actual, expected in zip(writes, original_writes):
        if actual[0] != expected[0] or any(not torch.equal(actual[i], expected[i]) for i in (2, 3, 4)):
            raise RuntimeError('Parameter intervention changed original keys/values/availability')


def assert_first_state_zero(capture, writes):
    """Each conversation's first read has no past; interventions cannot invent it."""
    unseen = torch.ones(capture.valid.shape[1], dtype=torch.bool, device=capture.valid.device)
    for t, pre, *_ in writes:
        first = capture.valid[t] & unseen
        for value in (pre[first], capture.base[t, first], capture.gap[t, first]):
            if bool(value.ne(0).any()):
                raise RuntimeError('First valid utterance acquired synthetic history')
        unseen &= ~capture.valid[t]


def evaluate_rate(model, loader, config, dimensions, saved, runtime, output, rate, max_batches=None):
    train, osram = runtime['train'], model.osram
    cursor = 0
    all_ids, all_labels, all_patterns, all_bins = [], [], [], []
    logits_by_condition = {}
    lagged = {}
    max_parity = 0.
    prefix = f'miss_{rate:.1f}'.replace('.', 'p')
    streams = [gzip.open(output / f'{prefix}_{name}.jsonl.gz', 'wt')
               for name in ('utterances', 'state_changes', 'spectra', 'direct_beta', 'lagged_alpha')]
    utterances, states_stream, spectral_stream, direct_stream, lag_stream = streams
    frozen = runtime['digest'](model)
    cuda_rng = torch.cuda.get_rng_state().clone()
    batches = 0
    try:
        with torch.no_grad():
            for raw in loader:
                if max_batches is not None and batches >= max_batches:
                    break
                data = train._move_batch(raw, next(model.parameters()).device)
                start = cursor
                availability, cursor = runtime['scatter'](saved['availability'], cursor, data[7])
                view = runtime['view'](data, availability, dimensions)
                ids = sample_rows(data[-1], view['umask'])
                valid = view['umask'].T.bool()
                if not torch.equal(availability.bool().any(-1), valid):
                    raise ValueError('Canonical saved availability must mark every valid utterance with an observed modality')
                labels = view['labels'][view['umask'].bool()].cpu().numpy()
                baseline_logits, baseline, original_writes = capture_forward(model, view, runtime)
                assert_first_state_zero(baseline, original_writes)
                original, actual_labels, _ = train._collect_predictions('MELD', baseline_logits,
                    view['labels'], view['umask'], config.mosi_task_mode)
                np.testing.assert_array_equal(actual_labels, saved['labels'][start:cursor])
                np.testing.assert_array_equal(original, saved['predictions'][start:cursor])
                np.testing.assert_array_equal(actual_labels, labels)
                full_replay = readout_logits(model, baseline.local, baseline.flat.flat_input, valid)
                torch.testing.assert_close(full_replay[valid], baseline_logits[valid], rtol=2e-5, atol=2e-5)
                if not torch.equal(full_replay[valid].argmax(-1), baseline_logits[valid].argmax(-1)):
                    raise RuntimeError('Readout replay changed classes')
                max_parity = max(max_parity, float((full_replay[valid] - baseline_logits[valid]).abs().max()))
                base_np = baseline_logits.transpose(0, 1)[view['umask'].bool()].cpu().numpy()
                base_ce = F.cross_entropy(torch.as_tensor(base_np), torch.as_tensor(labels), reduction='none').numpy()
                patterns = [''.join(str(int(v)) for v in row) for row in availability.transpose(0, 1)[view['umask'].bool()].cpu().numpy()]
                prior = (valid.long().cumsum(0) - 1).transpose(0, 1)[view['umask'].bool()].cpu().tolist()
                bins = [history_bin(i) for i in prior]
                all_ids.extend(ids); all_labels.extend(labels.tolist())
                all_patterns.extend(patterns); all_bins.extend(bins)

                def record(condition, logits):
                    values = logits.transpose(0, 1)[view['umask'].bool()].cpu().numpy()
                    logits_by_condition.setdefault(condition, []).append(values)
                    ce = F.cross_entropy(torch.as_tensor(values), torch.as_tensor(labels), reduction='none').numpy()
                    for i, (conversation, t) in enumerate(ids):
                        prediction, base_prediction = int(values[i].argmax()), int(original[i])
                        write_json(utterances, dict(condition=condition, rate=rate, conversation_id=conversation,
                            utterance_index=t, prior_valid_utterances=prior[i], history_bin=bins[i],
                            availability=patterns[i], label=int(labels[i]), prediction=prediction,
                            baseline_prediction=base_prediction, ce=float(ce[i]), baseline_ce=float(base_ce[i]),
                            delta_ce=float(ce[i] - base_ce[i]), correction=bool(prediction == labels[i] and base_prediction != labels[i]),
                            harm=bool(prediction != labels[i] and base_prediction == labels[i])))

                record('baseline', baseline_logits)
                beta = osram.beta_logits.sigmoid()
                for t, before, keys, values, available, after in original_writes:
                    for row in runtime['spectral'](keys, available, valid[t], beta, osram.write_ridge):
                        b = row.pop('batch')
                        write_json(spectral_stream, dict(rate=rate, conversation_id=str(data[-1][b]), utterance_index=t, **row))
                    # Same pre-write state/current keys/values: isolate beta's
                    # direct algebraic effect from accumulated history changes.
                    for value in (.05, .5, .95):
                        edited = osram.block_write(before, keys, values, available, beta=torch.full_like(beta, value))
                        difference = relative_per_head(edited - before, after - before).cpu().numpy()
                        absolute = (edited - after).flatten(2).norm(dim=-1).cpu().numpy()
                        for b in range(len(data[-1])):
                            if bool(valid[t, b]):
                                write_json(direct_stream, dict(rate=rate, conversation_id=str(data[-1][b]),
                                    utterance_index=t, beta=value, relative_write_delta_change=difference[b].tolist(),
                                    absolute_poststate_change=absolute[b].tolist(), availability=available[b].int().cpu().tolist()))

                for name, override in CONDITIONS.items():
                    if name == 'baseline':
                        continue
                    with runtime['override'](osram, **override):
                        logits, capture, writes = capture_forward(model, view, runtime)
                    assert_current_inputs_same(capture, baseline, writes, original_writes)
                    assert_first_state_zero(capture, writes)
                    record(name, logits)
                    for actual, expected in zip(writes, original_writes):
                        t = actual[0]
                        pre = relative_per_head(actual[1], expected[1]).cpu().numpy()
                        post = relative_per_head(actual[5], expected[5]).cpu().numpy()
                        base = relative_per_head(capture.base[t].reshape(-1, osram.num_heads, osram.value_dim),
                            baseline.base[t].reshape(-1, osram.num_heads, osram.value_dim)).cpu().numpy()
                        gap = relative_per_head(capture.gap[t].reshape(-1, 3, osram.num_heads, osram.value_dim).permute(0, 2, 1, 3),
                            baseline.gap[t].reshape(-1, 3, osram.num_heads, osram.value_dim).permute(0, 2, 1, 3)).cpu().numpy()
                        for b in range(len(data[-1])):
                            if bool(valid[t, b]):
                                write_json(states_stream, dict(condition=name, rate=rate, conversation_id=str(data[-1][b]),
                                    utterance_index=t, availability=actual[4][b].int().cpu().tolist(),
                                    prestate_relative_change=pre[b].tolist(), poststate_relative_change=post[b].tolist(),
                                    base_relative_change=base[b].tolist(), gap_relative_change=gap[b].tolist(),
                                    active_gap=(~actual[4][b].bool()).cpu().tolist()))
                    if runtime['digest'](model) != frozen:
                        raise RuntimeError('Parameter intervention failed to restore original state')
                for kind in ('base', 'gap'):
                    for head in range(osram.num_heads):
                        flat = runtime['mask_head'](baseline.flat.flat_input, local_dim=config.latent_dim,
                            context_dim=osram.context_dim, head=head, value_dim=osram.value_dim, kind=kind)
                        record(f'remove_{kind}_head_{head}', readout_logits(model, baseline.local, flat, valid))
                by_time = {row[0]: row for row in original_writes}
                for alpha_name, alpha in [('learned', osram.alpha_logits.sigmoid()),
                                           *[(str(value), value) for value in (.95, .98, .995)]]:
                    for source_t, target_t, memory in runtime['alpha_forks'](osram, original_writes, alpha):
                        selected = valid[source_t] & valid[target_t]
                        if not bool(selected.any()):
                            continue
                        logits = runtime['task_logits'](model, baseline.local[target_t], memory,
                            baseline.queries[target_t], by_time[target_t][2], baseline.availability[target_t],
                            valid[target_t])
                        original_target = baseline_logits[target_t]
                        if alpha_name == 'learned':
                            torch.testing.assert_close(memory[selected], by_time[target_t][1][selected], rtol=2e-5, atol=2e-5)
                            torch.testing.assert_close(logits[selected], original_target[selected], rtol=2e-5, atol=2e-5)
                            if not torch.equal(logits[selected].argmax(-1), original_target[selected].argmax(-1)):
                                raise RuntimeError('Learned-alpha lag replay changed predicted classes')
                        y = view['labels'][:, target_t][selected].cpu().numpy()
                        values = logits[selected].cpu().numpy()
                        original_values = original_target[selected].cpu().numpy()
                        key = (alpha_name, target_t - source_t)
                        entry = lagged.setdefault(key, {'labels': [], 'logits': [], 'original': []})
                        entry['labels'].append(y); entry['logits'].append(values); entry['original'].append(original_values.argmax(-1))
                        changed_ce = F.cross_entropy(torch.as_tensor(values), torch.as_tensor(y), reduction='none').numpy()
                        original_ce = F.cross_entropy(torch.as_tensor(original_values), torch.as_tensor(y), reduction='none').numpy()
                        for index, b in enumerate(selected.nonzero().flatten().tolist()):
                            write_json(lag_stream, dict(rate=rate, alpha=alpha_name, source_t=source_t, target_t=target_t,
                                distance=target_t-source_t, conversation_id=str(data[-1][b]), label=int(y[index]),
                                prediction=int(values[index].argmax()), baseline_prediction=int(original_values[index].argmax()),
                                ce=float(changed_ce[index]), baseline_ce=float(original_ce[index]),
                                delta_ce=float(changed_ce[index] - original_ce[index])))
                if any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Frozen diagnostics accumulated gradients')
                if not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                    raise RuntimeError('Frozen diagnostics consumed CUDA RNG')
                for stream in streams:
                    stream.flush()
                batches += 1
                print(f'rate={rate} batches={batches} utterances={cursor}', flush=True)
        if max_batches is None and cursor != len(saved['labels']):
            raise ValueError('Selected masks not fully consumed')
        if len(set(all_ids)) != len(all_ids):
            raise ValueError('Duplicate conversation/utterance sample identity')
        if runtime['digest'](model) != frozen:
            raise RuntimeError('Frozen model changed')
    finally:
        for stream in streams:
            stream.close()
    labels = np.asarray(all_labels)
    predictions = {name: np.concatenate(parts) for name, parts in logits_by_condition.items()}
    original = predictions['baseline'].argmax(-1)
    grouped = []
    for condition, logits in predictions.items():
        for field, groups in (('all', ['all'] * len(labels)), ('history_bin', all_bins), ('availability', all_patterns)):
            for group in sorted(set(groups)):
                selected = np.asarray(groups) == group
                grouped.append(dict(condition=condition, grouping=field, group=group,
                    **metrics(labels[selected], logits[selected], original[selected])))
    np.savez_compressed(output / f'{prefix}_predictions.npz', labels=labels,
        conversation_ids=np.asarray([v[0] for v in all_ids]), utterance_indices=np.asarray([v[1] for v in all_ids]),
        availability_patterns=np.asarray(all_patterns), history_bins=np.asarray(all_bins), **predictions)
    lagged_metrics = [dict(alpha=alpha, distance=distance, cohort='paired source-target occurrences; repeated targets',
        **metrics(np.concatenate(entry['labels']), np.concatenate(entry['logits']), np.concatenate(entry['original'])))
        for (alpha, distance), entry in lagged.items()]
    return dict(batches=batches, utterances=cursor, full_coverage=cursor == len(saved['labels']),
                sample_labels_sha256=hashlib.sha256(labels.astype(np.int64).tobytes()).hexdigest(),
                sample_ids_sha256=hashlib.sha256(json.dumps(all_ids).encode()).hexdigest(),
                saved_predictions_exact=True, saved_labels_exact=True, max_readout_logit_error=max_parity,
                state_sha256_before=frozen, state_sha256_after=runtime['digest'](model),
                cuda_rng_unchanged=True, no_gradients=True, metrics=grouped, lagged_metrics=lagged_metrics), all_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'mask-root', 'model-source', 'dataset-root', 'meld-pickle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--max-batches', type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max-batches must be positive')
    if args.mask_root.resolve() == args.source.resolve():
        raise ValueError('Exact selected-mask root must be separate from final-mask checkpoint root')
    os.environ['GCNET_DATASET_ROOT'] = str(args.dataset_root)
    os.environ['GCNET_MELD_PICKLE'] = str(args.meld_pickle)
    runtime = load_runtime(args.model_source)
    config_payload = json.loads((args.source / 'config.json').read_text())
    runtime['validate'](config_payload)
    exact = {'osram_num_heads': 8, 'osram_value_dim': 64, 'osram_key_dim': 64,
             'osram_write_step': .6, 'beta_mode': 'embedded', 'osram_write_ridge': .001}
    if any(config_payload.get(name) != value for name, value in exact.items()):
        raise ValueError('Require original eight 64-dimensional embedded-beta CHM with step.6/ridge.001')
    commit = subprocess.check_output(['git', '-C', str(args.model_source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != HISTORICAL_COMMIT:
        raise ValueError('Numerical model source must be original b4a3138')
    subprocess.run(['git', '-C', str(args.model_source), 'diff', '--exit-code', 'HEAD', '--',
                    'gcnet_missing_m3', 'gcnet_modality_jepa', 'config.py'], check=True, capture_output=True)
    if runtime['sha'](args.meld_pickle) != FEATURE_SHA256:
        raise ValueError('MELD feature pickle SHA256 does not match accepted baseline')
    for rate in RATES:
        suffix = f'{rate:.1f}'.replace('.', 'p')
        for path in (args.source / f'best_miss_{suffix}.pt', args.mask_root / f'predictions_miss_{suffix}.npz'):
            if not path.is_file():
                raise FileNotFoundError(path)
    gpu = runtime['gpu'](args.gpu_uuid)
    # Torch1.13's NVML UUID parser can report zero visible devices while the
    # CUDA runtime correctly sees the explicitly restricted single device.
    runtime_count = torch._C._cuda_getDeviceCount()
    if runtime_count != 1:
        raise RuntimeError('Require exactly one CUDA runtime device after UUID restriction')
    torch.cuda.device_count = lambda: runtime_count
    torch.cuda.random.device_count = torch.cuda.device_count
    train = runtime['train']
    config = replace(train.TrainConfig(**config_payload), device='cuda:0')
    torch.set_num_threads(2)
    _, _, test, ad, td, vd = runtime['loaders']('MELD-pickle', 'MELD-pickle', 'MELD-pickle',
        1, 'MELD', config.batch_size, 0, config.seed, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)
    model = runtime['build'](config, (ad, td, vd), torch.device('cuda:0'))
    args.output.mkdir(parents=True)
    summary = dict(status='running', training=False, internal_diagnostic_only=True, seed=66,
        rates={}, conditions=CONDITIONS, gpu=gpu, source=str(args.source), mask_root=str(args.mask_root),
        model_source=str(args.model_source), model_commit=commit, numerical_files=runtime['numerical_files'],
        code_sha256=runtime['code_sha256'],
        feature_sha256=FEATURE_SHA256, source_config=config_payload, source_config_sha256=runtime['sha'](args.source / 'config.json'),
        protocol='ACC-selected original per-rate checkpoint and exact winning-epoch masks; frozen intervention',
        relative_norm_denominator_floor=1e-12, command=sys.argv, max_batches=args.max_batches,
        driver_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        driver_sha256=runtime['sha'](Path(__file__)), started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    runtime['write'](args.output / 'SUMMARY.json', summary)
    tick = time.monotonic()
    reference_ids = None
    reference_labels = None
    try:
        for rate in RATES:
            suffix = f'{rate:.1f}'.replace('.', 'p')
            checkpoint_path = args.source / f'best_miss_{suffix}.pt'
            mask_path = args.mask_root / f'predictions_miss_{suffix}.npz'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=True)
            with np.load(mask_path) as saved:
                if len(saved['labels']) != len(saved['predictions']) or len(saved['availability']) != len(saved['labels']):
                    raise ValueError('Selected artifact counts disagree')
                if saved['availability'].shape != (len(saved['labels']), 3) or not np.isin(saved['availability'], (0, 1)).all():
                    raise ValueError('Require binary [N,3] saved selected availability')
                if not (saved['availability'].sum(-1) >= 1).all():
                    raise ValueError('Canonical masks must preserve at least one observed modality')
                result, ids = evaluate_rate(model, test[config.fold - 1], config, (ad, td, vd), saved,
                    runtime, args.output, rate, args.max_batches)
            if reference_ids is not None and reference_ids != ids:
                raise RuntimeError('Sample identity/order changed across rates')
            if reference_labels is not None and reference_labels != result['sample_labels_sha256']:
                raise RuntimeError('Sample labels changed across rates')
            reference_ids = ids
            reference_labels = result['sample_labels_sha256']
            baseline = next(row for row in result['metrics'] if row['condition'] == 'baseline' and row['grouping'] == 'all')
            if args.max_batches is None:
                for metric, field in (('accuracy', 'selection_accuracy'), ('weighted_f1', 'selection_weighted_f1')):
                    if abs(baseline[metric] - checkpoint[field]) > 1e-10:
                        raise RuntimeError(f'Selected score parity failed: {field}')
            summary['rates'][str(rate)] = dict(**result, checkpoint_epoch=checkpoint['epoch'],
                checkpoint_sha256=runtime['sha'](checkpoint_path), selected_masks_sha256=runtime['sha'](mask_path))
            runtime['write'](args.output / 'SUMMARY.json', summary)
            temporary_csv = args.output / 'METRICS.csv.tmp'
            with temporary_csv.open('w', newline='') as stream:
                rows = [dict(rate=key, **row) for key, entry in summary['rates'].items() for row in entry['metrics']]
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            temporary_csv.replace(args.output / 'METRICS.csv')
        summary['status'] = 'partial_smoke' if args.max_batches else 'complete'
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        summary['seconds'] = time.monotonic() - tick
        summary['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        runtime['write'](args.output / 'SUMMARY.json', summary)


if __name__ == '__main__':
    main()
