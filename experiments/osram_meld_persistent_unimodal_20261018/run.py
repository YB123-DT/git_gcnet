"""Frozen A/T/V whole-conversation evaluation of the accepted MELD baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

PATTERNS = {'A': (1, 0, 0), 'T': (0, 1, 0), 'V': (0, 0, 1)}
HELPERS = Path('/home/qinjinghui/remote_experiments/osram_meld_related_history_diagnostic_20261018/source_470e653')
GPU = 'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'


def fixed_availability(umask, pattern):
    import torch
    if pattern not in PATTERNS or umask.ndim != 2 or not torch.isin(umask, umask.new_tensor([0, 1])).all():
        raise ValueError('Require A/T/V and binary [B,T] padding mask')
    bits = torch.tensor(PATTERNS[pattern], dtype=torch.float32, device=umask.device)
    return torch.where(umask.T.bool().unsqueeze(-1), bits, torch.zeros_like(bits))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(model, loader, dimensions, saved, trainer, pattern):
    import numpy as np
    import torch
    from experiments.osram_meld_small_lr_readout_20261018.evaluate import _metadata, _numpy
    from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import scatter_saved_availability
    from experiments.osram_meld_write_tuning_20261017.capture import _metrics
    fields = ('labels', 'conversation_ids', 'utterance_indices', 'speakers')
    parts = {k: [] for k in (*fields, 'availability', 'logits')}
    cursor = scans = 0
    with torch.no_grad():
        for raw in loader:
            data = trainer._move_batch(raw, next(model.parameters()).device)
            metadata = _metadata(data)
            count = len(metadata['labels'])
            for key in fields:
                np.testing.assert_array_equal(metadata[key], saved[key][cursor:cursor+count])
            if pattern is None:
                availability, stop = scatter_saved_availability(saved['availability'], cursor, data[7])
                assert stop == cursor + count
            else:
                availability = fixed_availability(data[7], pattern)
                np.testing.assert_array_equal(_numpy(availability)[_numpy(data[7].T).astype(bool)],
                    np.broadcast_to(PATTERNS[pattern], (count, 3)))
                assert not bool(availability[~data[7].T.bool()].any())
            # Host and guest both receive this fixed pattern; original speaker
            # selection and all numerical model paths remain untouched.
            view = trainer._prepare_view_from_primary_masks(data, availability, availability, dimensions)
            np.testing.assert_array_equal(_numpy(view['availability']), _numpy(availability))
            logits, _, _, completion = model([view['incomplete']], view['availability'],
                view['qmask'], view['umask'], view['lengths'], predict_missing=False)
            assert completion is None
            current = _numpy(logits.transpose(0, 1)[view['umask'].bool()])
            assert current.shape == (count, 7) and np.isfinite(current).all()
            if pattern is None:
                np.testing.assert_allclose(current, saved['logits'][cursor:cursor+count], rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(current.argmax(1), saved['predictions'][cursor:cursor+count])
            for key in fields:
                parts[key].append(metadata[key])
            parts['logits'].append(current)
            parts['availability'].append(_numpy(availability.transpose(0, 1))[data[7].cpu().numpy().astype(bool)])
            cursor += count
            scans += 1
    assert cursor == len(saved['labels']) == 2610
    artifacts = {key: np.concatenate(value) for key, value in parts.items()}
    artifacts['predictions'] = artifacts['logits'].argmax(1)
    metrics = _metrics(artifacts)
    return dict(metrics=metrics, scans=scans, metadata_exact=True,
        availability_sha256=hashlib.sha256(artifacts['availability'].tobytes()).hexdigest()), artifacts


def render(result):
    lines = ['# Current MELD baseline: persistent unimodal evaluation', '',
        'INTERNAL DIAGNOSTIC ONLY. Evaluation-only; no training or checkpoint reselection.',
        'Every conversation keeps only A, T(Text), or V throughout, not random-mask subgroups.',
        'Accepted Flat1600 + Local Skip, lr1e-5 / wd1e-5, seeds66/67/68, original cmerc-repro runtime.',
        'Each seed uses its preserved random-miss0.7 ACC-selected BEST for all three patterns.', '',
        '| Seed | Persistent visible | ACC | UA | W-F1 | CE |', '|---:|---|---:|---:|---:|---:|']
    for seed, row in result['seeds'].items():
        for name, value in row['patterns'].items():
            lines.append(f'| {seed} | {name} | '+' | '.join(f'{value["metrics"][m]:.6f}' for m in ('ACC','UA','WF1','CE'))+' |')
    lines += ['', '| Pattern | mean ACC | mean UA | mean W-F1 | mean CE |', '|---|---:|---:|---:|---:|']
    for name, row in result['aggregate'].items():
        lines.append('| '+name+' | '+' | '.join(f'{row["mean"][m]:.6f}' for m in ('ACC','UA','WF1','CE'))+' |')
    lines += ['', 'UA is seven-class macro recall; class recalls/support and sample standard deviations are in RESULTS.json.',
        'This is a new fixed-availability intervention, not a change to training GCNet missing or to the accepted random baseline.',
        'No weights, thresholds or hyperparameters selected using these new pattern scores.',
        'Original selected-mask replay passes before each seed; all source checkpoint/prediction hashes stay unchanged.',
        'No single-modality models were trained; Memory contains only the currently evaluated modality throughout the conversation.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Require a new independent evaluation output')
    sys.path.insert(0, str(HELPERS))
    from experiments.osram_meld_related_history_diagnostic_20261018.run import check_sources
    from experiments.osram_meld_optimizer_screen_20261018 import run as screen
    check_sources()
    baseline = json.loads(args.baseline.read_text())
    cfg0 = baseline['config_without_seed']
    assert cfg0['learning_rate'] == cfg0['weight_decay'] == 1e-5
    assert cfg0['missing_mask_protocol'] == 'gcnet-random-mask-v1'
    assert baseline['aggregate']['seeds'] == [66, 67, 68]
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU, GCNET_MELD_PICKLE=str(screen.original.DATASET),
        GCNET_DATASET_ROOT=str(screen.MODEL_SOURCE/'dataset'), GCNET_CACHE_ROOT=str(args.output/'feature_cache'))
    import numpy as np
    import torch
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment, repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu, state_digest
    gpu = guard_gpu(GPU)
    assert gpu['host_index'] == 3 and gpu['uuid'] == GPU
    environment = require_replication_environment(torch)
    repair_uuid_rng(torch)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert sha(screen.original.DATASET) == screen.original.DATA_SHA
    trainer = screen.original.load_historical(screen.MODEL_SOURCE)
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
    args.output.mkdir(parents=True)
    result = dict(status='running', training=False, seeds={}, baseline_id=baseline['baseline_id'],
        baseline_sha256=sha(args.baseline), source_commit=args.commit, source_sha256=sha(__file__),
        helper_snapshot=str(HELPERS), environment=environment, gpu=gpu, pid=os.getpid())
    tick = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    hashes = {}
    try:
        for run in baseline['runs']:
            seed = run['seed']; ref = run['artifacts']['0.7']
            cp, pred = Path(ref['checkpoint']), Path(ref['selected_predictions'])
            for path, digest in ((cp, ref['checkpoint_sha256']), (pred, ref['prediction_sha256'])):
                assert sha(path) == digest
                hashes[str(path)] = digest
            state = torch.load(cp, map_location='cpu', weights_only=False)
            assert state['config'] == dict(cfg0, seed=seed) and state['epoch'] == ref['epoch']
            with np.load(pred, allow_pickle=False) as file:
                saved = {key: file[key].copy() for key in file.files}
            cfg = trainer.TrainConfig(**state['config'])
            _, _, tests, ad, td, vd = get_loaders(audio_root='MELD-pickle', text_root='MELD-pickle',
                video_root='MELD-pickle', dataset='MELD', num_folder=1, batch_size=32, num_workers=0,
                seed=seed, validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
            dimensions = (ad, td, vd)
            model = _build_model(cfg, dimensions, torch.device('cuda:0'))
            model.load_state_dict(state['model'], strict=True)
            model.eval().requires_grad_(False)
            before = state_digest(model)
            replay, _ = evaluate(model, tests[0], dimensions, saved, trainer, None)
            assert abs(replay['metrics']['ACC'] - run['testbest']['rates']['0.7']['ACC']) < 1e-10
            row = dict(source=ref, original_selected_replay=replay, patterns={})
            result['seeds'][str(seed)] = row
            for pattern in PATTERNS:
                value, artifacts = evaluate(model, tests[0], dimensions, saved, trainer, pattern)
                row['patterns'][pattern] = value
                np.savez_compressed(args.output/f'seed{seed}_{pattern}.npz', **artifacts)
                print(seed, pattern, value['metrics'], flush=True)
                write_json(args.output/'RESULTS.json', result)
            assert state_digest(model) == before and all(p.grad is None for p in model.parameters())
            del model, state
        result['aggregate'] = {pattern: {scope: {metric: fn(
            row['patterns'][pattern]['metrics'][metric] for row in result['seeds'].values())
            for metric in ('ACC', 'UA', 'WF1', 'CE')} for scope, fn in
            (('mean', statistics.mean), ('sample_std', statistics.stdev))} for pattern in PATTERNS}
        assert all(sha(path) == digest for path, digest in hashes.items())
        result.update(status='complete', source_inputs_unchanged=True, source_hashes=hashes,
            seconds=time.monotonic()-tick, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        (args.output/'RESULT.md').write_text(render(result))
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(args.output/'RESULTS.json', result)


if __name__ == '__main__':
    main()
