"""Two approved MOSI66 source-MoE arms; INTERNAL TEST-ORACLE SCREEN."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2')
RATES = tuple(str(i / 10) for i in range(8))


def validate_mode(mode):
    if mode not in ('replace', 'residual'):
        raise ValueError('Only the two approved connection arms are allowed')


def learning_rate(epoch):
    if not 0 <= epoch < 150:
        raise ValueError('Exactly 150 epochs authorized')
    return 3e-4 if epoch < 100 else 3e-4 / 10


def setup(mode, uuid, output):
    import numpy as np
    import torch
    from experiments.osram_mosi_cmerc_20261023 import worker as control
    from experiments.osram_mosi_gcnet_router_3seed_20261019.worker import fresh_backbone
    from experiments.osram_tasr_20261018 import runner_helpers as h
    validate_mode(mode)
    prior = json.loads((REFERENCE / 'SUMMARY.json').read_text())
    if prior['status'] != 'complete' or prior['epoch'] != 150:
        raise ValueError('Completed accepted cmerc Flat reference required')
    # Reuse original data-loader/config construction, not its immutable-source
    # guard: the intended change in this experiment is the new opt-in module.
    original = json.loads((control.REFERENCE / 'SUMMARY.json').read_text())
    bundle = control.setup(uuid, output, original)
    baseline, base = bundle['model'], bundle['cfg']
    if h.backbone_digest(baseline) != prior['initial_backbone_sha256']:
        raise ValueError('Current disabled model initialization differs from accepted Flat')
    cfg = replace(base, osram_source_moe_mode=mode)
    model = fresh_backbone(cfg, bundle['dimensions'], torch.device('cuda:0'))
    if mode == 'replace':
        model.osram.emotion_adapter.requires_grad_(False)
    old = baseline.state_dict()
    for key, value in model.state_dict().items():
        if key.startswith('osram.source_moe.'):
            continue
        if key not in old or not torch.equal(old[key], value):
            raise ValueError('Original initialization altered: ' + key)
    bundle.update(cfg=cfg, model=model)
    banks = torch.load(REFERENCE / 'fixed_masks.pt', map_location='cpu', weights_only=False)
    test_bank = banks['test']
    hashes = {r: hashlib.sha256(b''.join(x['availability'].numpy().tobytes()
              for x in records)).hexdigest() for r, records in test_bank.items()}
    if hashes != prior['identity']['mask_hashes']['test']:
        raise ValueError('Accepted exact test mask hashes differ')
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    checks = {}
    for rate in ('0.0', '0.7'):
        _, a = scalar.evaluate(bundle, baseline, 'flat', 'test', test_bank[rate], max_batches=1)
        _, b = scalar.evaluate(bundle, model, 'flat', 'test', test_bank[rate], max_batches=1)
        np.testing.assert_array_equal(a['prediction'], b['prediction'])
        checks[rate] = dict(initial_predictions_exact=True, N=len(a['labels']))
    del baseline
    torch.cuda.empty_cache()
    return bundle, prior, test_bank, hashes, checks


def source_hashes():
    """Capture all imported project source families, not weights or feature caches."""
    paths = []
    for folder in ('gcnet_missing_m3', 'gcnet_modality_jepa', 'gcnet_plci_jepa', 'experiments'):
        paths.extend((ROOT / folder).rglob('*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def report(output, summary, prior):
    rows = ['# Source–Role MoE MOSI', '', 'INTERNAL TEST-ORACLE SCREEN', '',
            f"Mode: {summary['mode']}; status: {summary['status']}; epoch: {summary.get('epoch', 0)}/150.",
            '', 'Same cmerc environment, GCNet random missing and accepted exact test masks.',
            'Per-rate test-best selection is exploratory and may select different epochs.', '']
    if summary.get('best_by_rate'):
        rows += ['| Rate | Flat WF1 | MoE ACC | MoE WF1 | MSE | Epoch |',
                 '|---|---:|---:|---:|---:|---:|']
        for rate, best in summary['best_by_rate'].items():
            m = best['metrics']['router']
            old = prior['best_by_rate'][rate]['metrics']['router']['WF1']
            rows.append(f"|{rate}|{old:.4f}|{m['ACC']:.4f}|{m['WF1']:.4f}|{m['MSE']:.4f}|{best['epoch']}|")
    rows += ['', 'Initialization and smoke checks are implementation evidence, not performance evidence.',
             'Source slots track joint-write Value contributions, not pure modal causal attribution.',
             'Replace and residual differ in active Flat capacity; this is not a capacity-matched ablation.', '']
    (output / 'RESULT.md').write_text('\n'.join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('replace', 'residual'), required=True)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # config.py resolves label paths at import time, before loader setup. Use
    # the same manifest root that setup subsequently checks by SHA256.
    manifest = json.loads((ROOT / 'experiments/osram_complete150_20261011/MANIFEST.json').read_text())
    os.environ['GCNET_DATASET_ROOT'] = manifest['data']['CMUMOSI']['dataset_root']
    import fcntl
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore, checkpoint_payload
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_mosi_cmerc_20261023.worker import check_training_parity
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_iemocap_b6_transfer_20261015.worker import (
        require_replication_environment, repair_uuid_rng)
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    guard_gpu(args.gpu_uuid)
    environment = require_replication_environment(torch)
    rng_repair = repair_uuid_rng(torch)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output / 'SUMMARY.json'
    if path.exists() and json.loads(path.read_text()).get('status') in ('complete', 'smoke_passed'):
        raise ValueError('Do not rerun a completed output')
    tick = time.monotonic()
    summary = dict(status='preflight', pid=os.getpid(), mode=args.mode, phase=args.phase,
        seed=66, dataset='CMUMOSI', source_commit=args.source_commit,
        label='INTERNAL TEST-ORACLE SCREEN', environment=environment, rng_repair=rng_repair)
    prior = None
    try:
        h.write_json(path, summary)
        bundle, prior, bank, masks, checks = setup(args.mode, args.gpu_uuid, args.output)
        model, cfg = bundle['model'], bundle['cfg']
        sources = source_hashes()
        identity = dict(source_commit=args.source_commit, source_hashes=sources,
            config=asdict(cfg), environment=environment, mode=args.mode, seed=66,
            mask_hashes={'test': masks}, split=bundle['split'], seed_protocol=bundle['seed_protocol'],
            data_manifest=bundle['data_manifest'], reference=str(REFERENCE),
            reference_summary_sha256=h.sha(REFERENCE / 'SUMMARY.json'),
            old_weights_loaded=False, rng_repair=rng_repair)
        optimizer = torch.optim.Adam([dict(params=[p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate, name='backbone')], weight_decay=cfg.weight_decay)
        summary.update(status='running', identity=identity, config=asdict(cfg),
            gpu=bundle['gpu'], initialization=checks,
            total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            moe_parameters=sum(p.numel() for p in model.osram.source_moe.parameters()))
        h.write_json(args.output / 'config.json', asdict(cfg))
        torch.save({'test': bank}, args.output / 'fixed_masks.pt')
        observed = {}
        handles = []
        def observe(name):
            def hook(grad):
                if not torch.isfinite(grad).all():
                    raise RuntimeError('Nonfinite MoE gradient: ' + name)
                observed[name] = observed.get(name, False) or bool(grad.ne(0).any())
            return hook
        for name, p in model.osram.source_moe.named_parameters():
            handles.append(p.register_hook(observe(name)))
        scans = [0]
        original_scan = model.osram._scan
        def counted_scan(*a, **kw):
            scans[0] += 1
            return original_scan(*a, **kw)
        model.osram._scan = counted_scan
        history, best, start = [], {}, 0
        reset_training_stochasticity(66, cfg.fold)
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            selected.validate_selected(args.output, best)
        ref_history = json.loads((REFERENCE / 'history.json').read_text())
        torch.cuda.reset_peak_memory_stats()
        # Three full MOSI epochs open the chained zero-initialized output layers
        # and check the router/experts, rather than merely observing a zero grad tensor.
        budget = 3 if args.phase == 'smoke' else 150
        for epoch in range(start, budget):
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(epoch)
            scans[0] = 0
            row = scalar.train_epoch(bundle, model, optimizer, epoch, 'flat')
            if scans[0] != row['optimizer_steps']:
                raise RuntimeError('Expected exactly one causal Memory scan per training batch')
            check_training_parity(row, ref_history[epoch])
            row['source_moe'] = model.osram.source_moe.last_diagnostics
            row['diagnostics_scope'] = 'last training batch, not full-test distribution'
            if epoch >= 2:
                missing = [name for name in observed if not observed[name]]
                expected = set(dict(model.osram.source_moe.named_parameters()))
                missing += sorted(expected - set(observed))
                if missing:
                    raise RuntimeError('No nonzero MoE training gradient: ' + str(missing))
            if args.phase == 'train':
                evaluated = {r: scalar.evaluate(bundle, model, 'flat', 'test', bank[r]) for r in RATES}
                row['test'] = {r: v[0] for r, v in evaluated.items()}
                history.append(row)
                selected.persist_epoch(args.output, model, optimizer, epoch=epoch+1,
                    identity=identity, history=history, best=best, evaluated=evaluated)
            else:
                history.append(row)
            summary.update(epoch=epoch+1, best_by_rate=best, seconds=time.monotonic()-tick,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                nonzero_moe_gradients=observed, training_inputs_exact=True)
            h.write_json(path, summary)
            print(json.dumps(dict(epoch=epoch+1, mode=args.mode, task_MSE=row['task_MSE'],
                seconds=summary['seconds'], best_by_rate={r:v['metrics']['router'] for r,v in best.items()})), flush=True)
        for handle in handles:
            handle.remove()
        if args.phase == 'smoke':
            _, before = scalar.evaluate(bundle, model, 'flat', 'test', bank['0.7'], max_batches=1)
            payload = checkpoint_payload(model, optimizer, epoch=budget, identity=identity,
                history=history, best_key=None)
            h.atomic_torch_save(args.output / 'smoke.pt', payload)
            state = torch.load(args.output / 'smoke.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            _, after = scalar.evaluate(bundle, model, 'flat', 'test', bank['0.7'], max_batches=1)
            import numpy as np
            np.testing.assert_array_equal(before['prediction'], after['prediction'])
            if len(state['rng']['cuda']) != 1:
                raise RuntimeError('Missing UUID-bound CUDA RNG recovery state')
            summary.update(status='smoke_passed', strict_reload_exact=True, cuda_rng_saved=True)
        else:
            summary['evaluation'] = selected.publish_selected(args.output, best)
            summary['aggregate'] = scalar.aggregate(summary['evaluation'])
            summary.update(status='complete')
        if source_hashes() != sources:
            raise RuntimeError('Source snapshot changed during run')
        h.write_json(args.output / 'history.json', history)
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        summary['seconds'] = time.monotonic()-tick
        h.write_json(path, summary)
        if prior is not None:
            report(args.output, summary, prior)


if __name__ == '__main__':
    main()
