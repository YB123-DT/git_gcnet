"""Two fixed-checkpoint diagnostics; no optimizer or parameter update."""
import argparse
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

HISTORY_RATE = '0.3'
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003')


def choose_targets(view):
    """One label-independent later target per eligible conversation."""
    valid = view['umask'].detach().cpu().numpy().astype(bool)
    result = []
    for b, name in enumerate(view['conversation_ids']):
        eligible = np.flatnonzero(valid[b])
        eligible = eligible[eligible > 0]
        if len(eligible):
            digest = hashlib.sha256(f'current-pattern:66:{name}'.encode()).digest()
            result.append((b, int(eligible[int.from_bytes(digest[:8], 'big') % len(eligible)])))
    return result


def select_donor(pool, recipient_id, length):
    """Match the whole strict prefix, not labels or a favorable donor score."""
    if length <= 1:
        return None
    eligible = sorted(name for name, features in pool.items()
                      if str(name) != str(recipient_id) and len(features) >= length-1)
    if not eligible:
        return None
    digest = hashlib.sha256(f'history-donor:66:{recipient_id}:{length}'.encode()).digest()
    return eligible[int.from_bytes(digest[:8], 'big') % len(eligible)]


def _incomplete(complete, availability, dimensions):
    import torch
    mask = torch.repeat_interleave(availability.bool(),
        torch.tensor(dimensions, device=availability.device), dim=-1)
    return torch.where(mask, complete, torch.zeros_like(complete))


def replaced_view(background, pool, donor_ids, dimensions):
    complete = background['complete'].clone()
    for b, donor in enumerate(donor_ids):
        length = int(background['umask'][b].sum())
        if donor is not None:
            if donor == str(background['conversation_ids'][b]) or len(pool[donor]) < length-1:
                raise ValueError('Donor must be another sufficiently long conversation')
            complete[:length-1, b] = pool[donor][:length-1].to(complete)
    return dict(background, complete=complete,
        incomplete=_incomplete(complete, background['availability'], dimensions))


def current_pattern_view(background, dimensions, pattern):
    import torch
    from .core import pattern_bits
    valid = background['umask'].T.bool()
    support = torch.tensor(pattern_bits(pattern), device=valid.device,
                           dtype=background['availability'].dtype)
    availability = valid[..., None].to(support.dtype)*support
    return dict(background, availability=availability,
        incomplete=_incomplete(background['complete'], availability, dimensions))


def load_train_and_donors(bundle, output):
    import torch
    from experiments.osram_mosi_gcnet_router_3seed_20261019.worker import load_manifest
    from experiments.osram_mosi_recent_history_20261018 import run as data_source
    manifest, _ = load_manifest(data_source.DATA, data_source.DATA_SHA)
    cfg, trainer = bundle['cfg'], bundle['trainer']
    train, dev, test, *dimensions = trainer.get_loaders(*manifest['feature_roots'],
        dataset='CMUMOSI', num_folder=1, batch_size=cfg.batch_size, num_workers=0,
        seed=66, validation_fraction=cfg.validation_fraction,
        evaluation_protocol=cfg.evaluation_protocol)
    if tuple(dimensions) != bundle['dimensions']:
        raise ValueError('Train/donor feature dimensions changed')
    for name, loader in (('train', train[0]), ('validation', dev[0]), ('test', test[0])):
        if loader.protocol_metadata != bundle['identity']['split'][name]:
            raise ValueError('Train/donor split changed: '+name)
    loader = train[0]
    loader.sampler.set_epoch(0)
    pool = {}
    for raw in loader:
        data = trainer._move_batch(raw, torch.device('cpu'))
        full_mask = data[7].T[..., None].expand(-1, -1, 3).clone()
        view = trainer._prepare_view_from_primary_masks(data, full_mask, full_mask,
                                                       bundle['dimensions'])
        for b, name in enumerate(view['conversation_ids']):
            name = str(name)
            if name in pool:
                raise ValueError('Duplicate donor training conversation')
            n = int(view['umask'][b].sum())
            pool[name] = view['complete'][:n, b].detach().cpu().clone()
    return loader, pool


def _write_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False)+'\n')


def gradient_audit(bundle, loader, output, *, smoke=False):
    import torch
    from .core import PATTERNS, prefix_for_pattern, write_parameters, target_gradient, geometry
    from experiments.osram_tasr_20261018.runner_helpers import isolated_rng, write_json
    model, trainer = bundle['model'], bundle['trainer']
    model.eval().requires_grad_(False)
    named = write_parameters(model)
    for _, param in named:
        param.requires_grad_(True)
    totals = {p: [torch.zeros_like(v, device='cpu', dtype=torch.float64) for _, v in named]
              for p in PATTERNS}
    count = 0
    digest = hashlib.sha256()
    pattern_rows = []
    max_memory_error = max_past_error = 0.
    schedule = trainer._schedules(bundle['cfg'], 'train')[float(HISTORY_RATE)]
    loader.sampler.set_epoch(0)
    try:
        with isolated_rng(66+271923), gzip.open(output/'GRADIENT_PAIRS.jsonl.gz', 'wt') as stream:
            for index, raw in enumerate(loader):
                data = trainer._move_batch(raw, next(model.parameters()).device)
                background = trainer._prepare_view(data, schedule, 0, bundle['dimensions'])
                digest.update(background['availability'].detach().cpu().contiguous().numpy().tobytes())
                for b, t in choose_targets(background):
                    records = {}
                    first = None
                    for pattern in PATTERNS:
                        view = prefix_for_pattern(background, b, t, bundle['dimensions'], pattern)
                        value = target_gradient(model, view, named)
                        if first is None:
                            first = value
                        else:
                            torch.testing.assert_close(value['memory'], first['memory'], rtol=0, atol=0)
                            torch.testing.assert_close(value['past_prediction'], first['past_prediction'], rtol=0, atol=0)
                            max_memory_error = max(max_memory_error, float((value['memory']-first['memory']).abs().max()))
                            max_past_error = max(max_past_error, float((value['past_prediction']-first['past_prediction']).abs().max()))
                        if value['calls'] != t+1:
                            raise RuntimeError('Expected one original causal prefix scan')
                        gradients = value['gradients']
                        for total, grad in zip(totals[pattern], gradients):
                            if grad is not None:
                                total.add_(grad.double())
                        pattern_rows.append(dict(conversation_id=str(background['conversation_ids'][b]),
                            target_t=t, pattern=pattern, loss=value['loss'], prediction=value['prediction'],
                            residual=value['residual'], label=float(view['labels'][0,t]),
                            gradient_defined=sum(g is not None for g in gradients),
                            gradient_nonzero=sum(g is not None and bool(g.ne(0).any()) for g in gradients)))
                        records[pattern] = value
                    for pa, pb in itertools.combinations(PATTERNS, 2):
                        va, vb = records[pa], records[pb]
                        measured = geometry(named, va['gradients'], vb['gradients'])
                        for group, geo in [('Writer', measured['overall']),
                                           ('WriteKey', measured['groups']['WriteKey']),
                                           ('WriteValue', measured['groups']['WriteValue'])]:
                            product = va['residual']*vb['residual']
                            cosine = geo['cosine']
                            jacobian = (cosine*(1 if product > 0 else -1)
                                if cosine is not None and abs(va['residual'])>1e-8 and abs(vb['residual'])>1e-8 else None)
                            _write_row(stream, dict(conversation_id=str(background['conversation_ids'][b]),
                                target_t=t, split='train', history_rate=HISTORY_RATE,
                                pattern_a=pa, pattern_b=pb, label=float(background['labels'][b,t]),
                                residual_a=va['residual'], residual_b=vb['residual'],
                                residual_sign_opposed=product<0, group=group, loss_cosine=cosine,
                                loss_dot=geo['dot'], grad_norm_a=geo['visible_norm'],
                                grad_norm_b=geo['missing_norm'], jacobian_cosine=jacobian))
                    count += 1
                    stream.flush()
                    print(json.dumps(dict(event='gradient_anchor', count=count,
                        conversation_id=str(background['conversation_ids'][b]), target=t)), flush=True)
                    del records, first
                    if smoke and count >= 2:
                        break
                if smoke and count >= 2:
                    break
        aggregate = []
        for pa, pb in itertools.combinations(PATTERNS, 2):
            measured = geometry(named, [g/max(1,count) for g in totals[pa]],
                                [g/max(1,count) for g in totals[pb]])
            aggregate.append(dict(pattern_a=pa, pattern_b=pb,
                overall=measured['overall'], WriteKey=measured['groups']['WriteKey'],
                WriteValue=measured['groups']['WriteValue']))
        write_json(output/'GRADIENT_PATTERNS.json', pattern_rows)
        write_json(output/'AGGREGATE_GRADIENTS.json', aggregate)
        return dict(anchors=count, pattern_predictions=len(pattern_rows),
            pair_rows=count*21*3, max_prewrite_memory_error=max_memory_error,
            max_past_prediction_error=max_past_error, sampled_history_mask_sha256=digest.hexdigest(),
            historical_writer_only=True, parameter_names=[n for n,_ in named],
            zero_optimizer_steps=True, aggregate_gradient_file='AGGREGATE_GRADIENTS.json')
    finally:
        model.requires_grad_(False)


def _capture_states(model, view):
    import torch
    from experiments.osram_meld_write_retention_20261014.run import WriteCapture
    from experiments.osram_tasr_20261018.runner_helpers import forward
    with torch.no_grad(), WriteCapture(model.osram, view['umask']) as capture:
        prediction = forward(model, view)
    length = view['availability'].shape[0]
    states = prediction.new_zeros(length, view['availability'].shape[1],
        model.osram.num_heads, model.osram.value_dim, model.osram.key_dim)
    for t, decayed, *_ in capture.snapshots:
        states[t] = decayed
    return states, prediction


def _fixed_prediction(model, view, states):
    import torch
    from .core import fixed_history_reads
    from experiments.osram_tasr_20261018.runner_helpers import forward
    local = []
    hook = model.osram.local_skip.register_forward_pre_hook(
        lambda _, args: local.append(args[0].detach().clone()))
    try:
        with torch.no_grad(), fixed_history_reads(model.osram, states):
            prediction = forward(model, view)
    finally:
        hook.remove()
    if len(local) != 1:
        raise RuntimeError('Expected one original Local Skip readout')
    return prediction, local[0]


def history_audit(bundle, pool, output, *, smoke=False):
    import torch
    from .core import PATTERNS
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018.runner_helpers import write_json
    model, trainer, loader = bundle['model'], bundle['trainer'], bundle['loader']
    model.eval().requires_grad_(False)
    loader.sampler.set_epoch(0)
    selected = bundle['summary']['best_by_rate'][HISTORY_RATE]
    with np.load(bundle['reference']/selected['predictions'], allow_pickle=False) as original:
        saved = {k:original[k].copy() for k in original.files}
    rows = cursor = 0
    max_saved_error = max_replay_error = max_empty_boundary_error = 0.
    donors = []
    with torch.no_grad(), gzip.open(output/'HISTORY_ROWS.jsonl.gz', 'wt') as stream:
        for index, raw in enumerate(loader):
            if smoke and index >= 1:
                break
            data = trainer._move_batch(raw, next(model.parameters()).device)
            meta = metadata(data)
            record = bundle['masks'][HISTORY_RATE][index]
            for k, value in meta.items():
                np.testing.assert_array_equal(value, record[k])
                np.testing.assert_array_equal(value, saved[k][cursor:cursor+len(meta['labels'])])
            availability = record['availability'].to(data[7].device)
            background = trainer._prepare_view_from_primary_masks(data, availability,
                availability, bundle['dimensions'])
            factual, normal = _capture_states(model, background)
            valid = data[7].bool()
            normal_array = normal.transpose(0,1)[valid].cpu().numpy().reshape(-1)
            expected = saved['prediction'][cursor:cursor+len(normal_array)].reshape(-1)
            np.testing.assert_allclose(normal_array, expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(normal_array>0, expected>0)
            np.testing.assert_array_equal(availability.transpose(0,1)[valid].cpu().numpy(),
                                          saved['availability'][cursor:cursor+len(normal_array)])
            max_saved_error = max(max_saved_error, float(np.abs(normal_array-expected).max()))
            replay, _ = _fixed_prediction(model, background, factual)
            torch.testing.assert_close(replay, normal, rtol=2e-5, atol=2e-5)
            if not torch.equal(replay.transpose(0,1)[valid]>0, normal.transpose(0,1)[valid]>0):
                raise RuntimeError('Original fixed-state replay changed a polarity')
            max_replay_error = max(max_replay_error, float((normal-replay).abs().max()))
            donor_ids = [select_donor(pool, str(name), int(background['umask'][b].sum()))
                         for b,name in enumerate(background['conversation_ids'])]
            for b, donor in enumerate(donor_ids):
                donors.append(dict(recipient_id=str(background['conversation_ids'][b]),
                    recipient_length=int(background['umask'][b].sum()), donor_id=donor,
                    donor_split='train' if donor else None,
                    donor_length=len(pool[donor]) if donor else None,
                    donor_features_sha256=hashlib.sha256(pool[donor].contiguous().numpy().tobytes()).hexdigest() if donor else None))
            replacement, _ = _capture_states(model,
                replaced_view(background, pool, donor_ids, bundle['dimensions']))
            empty = torch.zeros_like(factual)
            for pattern in PATTERNS:
                current = current_pattern_view(background, bundle['dimensions'], pattern)
                results, current_local = {}, None
                for condition, state in (('correct', factual), ('empty', empty), ('replaced', replacement)):
                    prediction, local = _fixed_prediction(model, current, state)
                    if current_local is None:
                        current_local = local
                    else:
                        torch.testing.assert_close(local, current_local, rtol=0, atol=0)
                    if not torch.isfinite(prediction).all():
                        raise RuntimeError('Nonfinite fixed-history task output')
                    results[condition] = prediction[:,:,0].cpu().numpy()
                for b, name in enumerate(background['conversation_ids']):
                    n = int(background['umask'][b].sum())
                    for t in range(n):
                        if t == 0:
                            errors = [abs(results[c][t,b]-results['correct'][t,b]) for c in ('empty','replaced')]
                            max_empty_boundary_error = max(max_empty_boundary_error, *map(float, errors))
                            if max(errors)>1e-6:
                                raise RuntimeError('No-history boundary created artificial history influence')
                        _write_row(stream, dict(conversation_id=str(name), target_t=t,
                            split='test', pattern=pattern, history_rate=HISTORY_RATE,
                            label=float(background['labels'][b,t]),
                            pred_correct=float(results['correct'][t,b]),
                            pred_empty=float(results['empty'][t,b]),
                            pred_replaced=float(results['replaced'][t,b]) if donor_ids[b] is not None else None,
                            history_length=t, donor_id=donor_ids[b]))
                        rows += 1
                stream.flush()
            cursor += len(normal_array)
            print(json.dumps(dict(event='history_batch', batch=index+1,
                utterances=cursor, rows=rows, factual_replay_error=max_replay_error)), flush=True)
    if not smoke and cursor != len(saved['labels']):
        raise RuntimeError('Full original test cohort was not completely consumed')
    write_json(output/'DONORS.json', donors)
    return dict(utterances=cursor, rows=rows, full_test_coverage=not smoke,
        donor_conversations=sum(d['donor_id'] is not None for d in donors),
        unavailable_donors=sum(d['donor_id'] is None for d in donors),
        max_saved_prediction_error=max_saved_error, max_fixed_state_replay_error=max_replay_error,
        max_nohistory_prediction_error=max_empty_boundary_error,
        current_local_identical_across_history_conditions=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    from experiments.osram_kv_task_utility_20261022.run import setup
    from experiments.osram_tasr_20261018.runner_helpers import backbone_digest, write_json, sha, isolated_rng
    from .report import generate
    if args.output.exists():
        raise FileExistsError('Fresh output required; never overwrite original evidence')
    args.output.mkdir(parents=True)
    tick = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    files = [str(p.relative_to(root)) for package in ('gcnet_missing_m3', 'gcnet_modality_jepa')
             for p in (root/package).rglob('*.py')]
    files += ['experiments/osram_current_pattern_history_audit_20261022/'+n
              for n in ('run.py','core.py','report.py')]
    files += ['experiments/osram_kv_task_utility_20261022/run.py',
        'experiments/osram_text_condition_optimization_audit_20261022/run.py',
        'experiments/osram_text_condition_optimization_audit_20261022/core.py',
        'experiments/osram_meld_write_retention_20261014/task_replay.py',
        'experiments/osram_meld_write_retention_20261014/run.py',
        'experiments/osram_tasr_20261018/runner_helpers.py']
    state = dict(status='preflight', source_commit=args.source_commit, pid=os.getpid(),
        source_hashes={name:sha(root/name) for name in files}, server='biggpu',
        new_training_steps=0, optimizer_created=False, smoke=args.smoke,
        history_rate=HISTORY_RATE, label='INTERNAL DIAGNOSTIC ONLY',
        protocol=dict(dataset='CMUMOSI', seed=66, checkpoint_rate='0.3',
            history_rate=HISTORY_RATE, mask_protocol='gcnet-random-mask-v1',
            current_patterns=['A','T','V','AT','AV','TV','ATV'],
            current_intervention='one target at a time; not Persistent Missing',
            gradient_split='train', history_evaluation_split='test',
            gradient_path='past writes only; current Key output detached',
            donor_split='train', donor_mask_and_speakers='recipient trajectory unchanged',
            optimizer_steps=0, smoke_only=args.smoke))
    write_json(args.output/'STATUS.json', state)
    try:
        torch.set_num_threads(1)
        bundle = setup(REFERENCE, args.gpu_uuid, args.output)
        model = bundle['model']
        before = backbone_digest(model)
        train_loader, donors = load_train_and_donors(bundle, args.output)
        state.update(status='running', identity=bundle['identity'], model_digest=before)
        write_json(args.output/'STATUS.json', state)
        torch.cuda.reset_peak_memory_stats()
        with isolated_rng(66+76591):
            state['gradient_audit'] = gradient_audit(bundle, train_loader, args.output, smoke=args.smoke)
            write_json(args.output/'STATUS.json', state)
            state['history_audit'] = history_audit(bundle, donors, args.output, smoke=args.smoke)
        if backbone_digest(model) != before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Diagnostic changed model state or populated parameter.grad')
        for name, digest in state['source_hashes'].items():
            if sha(root/name) != digest:
                raise RuntimeError('Immutable numerical source changed: '+name)
        state.update(collection_complete=True, weights_unchanged_exact=True,
            parameter_grad_unmodified=True, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
            seconds=time.monotonic()-tick)
        write_json(args.output/'STATUS.json', state)
        generate(args.output)
        state.update(status='complete')
    except BaseException as error:
        state.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        write_json(args.output/'STATUS.json', state)


if __name__ == '__main__':
    main()
