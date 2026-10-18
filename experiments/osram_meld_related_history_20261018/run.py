"""Two approved MELD Local-history blocks on unchanged O6 Flat training."""
from __future__ import annotations

import argparse
import fcntl
from functools import wraps
import hashlib
import importlib.util
import inspect
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.osram_meld_optimizer_screen_20261018 import run as screen
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler, write_json

CONTROL = Path('/home/qinjinghui/remote_experiments/osram_meld_optimizer_screen_20261018/runs_v1/O6')
GPU = 0
UUID = 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45'


def branch_config(kind):
    if kind not in ('B1', 'B2'):
        raise ValueError('Only approved B1/B2 are authorized')
    return dict(kind=kind, window=8, attention_dim=128,
        input_dim=2435 if kind == 'B1' else 2564,
        relation_weight=0. if kind == 'B1' else .05)


def make_config(kind):
    branch_config(kind)
    return screen.make_config(screen.source_config(), 'O6')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_source():
    screen.check_snapshot()
    manifest = json.loads(Path(__file__).with_name('SOURCE_MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        if screen.original.sha(ROOT / name) != digest:
            raise ValueError('Branch implementation changed: ' + name)
    return manifest


def original_hash(model):
    """Same byte serialization as TrainingCapture, excluding only the new branch."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('osram.related_history.'):
            continue
        array = value.detach().cpu().contiguous().numpy().copy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def observe_training(trainer, output, integration, *, smoke=False):
    """One RNG-restored parity check, then observe actual backward gradients."""
    import torch
    original = trainer.train_epoch
    signature = inspect.signature(original)
    parity_done = (output / 'last_training.pt').exists()
    if parity_done and not (output / 'ZERO_INIT_PARITY.json').exists():
        raise ValueError('Resume requires recorded original zero-init parity')

    def train(*args, **kwargs):
        nonlocal parity_done
        bound = signature.bind(*args, **kwargs)
        model, optimizer = bound.arguments['model'], bound.arguments['optimizer']
        epoch = bound.arguments['epoch'] + 1
        if smoke:
            bound.arguments['loader'] = list(itertools.islice(bound.arguments['loader'], 2))
        branch = model.osram.related_history
        parameter_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        if any(id(p) not in parameter_ids for p in branch.parameters()):
            raise ValueError('Branch parameters missing from original optimizer')
        forward, clip = model.forward, torch.nn.utils.clip_grad_norm_
        observations = []
        auxiliary_values = []

        def checked_forward(*inputs, **keywords):
            nonlocal parity_done
            if not parity_done:
                cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                try:
                    integration.set_enabled(model, False)
                    with torch.no_grad():
                        off = forward(*inputs, **keywords)[0]
                    off_cpu, off_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                    torch.set_rng_state(cpu); torch.cuda.set_rng_state_all(cuda)
                    integration.set_enabled(model, True)
                    with torch.no_grad():
                        on = forward(*inputs, **keywords)[0]
                    if not torch.equal(on, off):
                        raise ValueError('Real-batch zero-init prediction parity failed')
                    if (not torch.equal(off_cpu, torch.get_rng_state()) or
                            any(not torch.equal(a, b) for a, b in zip(off_cuda, torch.cuda.get_rng_state_all()))):
                        raise ValueError('Branch changed original Dropout/mask RNG stream')
                    write_json(output / 'ZERO_INIT_PARITY.json', dict(passed=True,
                        real_batch=True, logits_exact=True, rng_exact=True))
                    parity_done = True
                finally:
                    torch.set_rng_state(cpu); torch.cuda.set_rng_state_all(cuda)
                    integration.set_enabled(model, True)
                    branch.clear_cache()
            return forward(*inputs, **keywords)

        def checked_clip(parameters, max_norm, *args, **kwargs):
            gradients = {name: p.grad for name, p in branch.named_parameters() if p.grad is not None}
            if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
                raise ValueError('Branch backward missing or nonfinite')
            observations.append(dict(gradient_norm=float(torch.stack([g.norm() for g in gradients.values()]).norm()),
                nonzero_tensors=sum(bool(g.abs().sum()) for g in gradients.values()),
                gradient_tensors=len(gradients)))
            if model.osram.related_history_last_aux is not None:
                auxiliary_values.append(model.osram.related_history_last_aux)
            return clip(parameters, max_norm, *args, **kwargs)

        model.forward = checked_forward
        torch.nn.utils.clip_grad_norm_ = checked_clip
        try:
            result = original(*bound.args, **bound.kwargs)
        finally:
            model.forward = forward
            torch.nn.utils.clip_grad_norm_ = clip
            branch.clear_cache()
        write_json(output / 'diagnostics' / f'epoch_{epoch:03d}.json', dict(epoch=epoch,
            steps=observations, auxiliary_relation_losses=auxiliary_values,
            branch_output_weight_norm=float(branch.output.weight.detach().norm()),
            optimizer=screen.constant_optimizer_check(
                screen.audit_optimizer_groups(model, optimizer.param_groups), make_config(branch.kind))))
        return result

    trainer.train_epoch = train
    if smoke:
        evaluate = trainer.evaluate_rate
        eval_signature = inspect.signature(evaluate)

        @wraps(evaluate)
        def small_eval(*args, **kwargs):
            bound = eval_signature.bind(*args, **kwargs)
            bound.arguments['loader'] = list(itertools.islice(bound.arguments['loader'], 1))
            return evaluate(*bound.args, **bound.kwargs)
        trainer.evaluate_rate = small_eval


def worker(args):
    manifest = check_source()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,
        GCNET_MELD_PICKLE=str(screen.original.DATASET),
        GCNET_DATASET_ROOT=str(screen.MODEL_SOURCE / 'dataset'),
        GCNET_CACHE_ROOT=str(args.output / 'feature_cache'))
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment, repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    import torch
    gpu = guard_gpu(args.gpu_uuid)
    if gpu['host_index'] != GPU or gpu['uuid'] != UUID:
        raise ValueError('Require authorized healthy physical GPU0 UUID')
    environment = require_replication_environment(torch); repair_uuid_rng(torch)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    config = json.loads(args.config.read_text())
    if config != make_config(args.kind) or screen.original.sha(screen.original.DATASET) != screen.original.DATA_SHA:
        raise ValueError('Unapproved config or MELD features')
    control = json.loads((CONTROL / 'SUMMARY.json').read_text())
    if control['status'] != 'complete' or control['config'] != config:
        raise ValueError('Matching completed O6 seed66 control required')
    trainer = screen.original.load_historical(screen.MODEL_SOURCE)
    from gcnet_missing_m3.training_resume import TrainingState
    from experiments.osram_meld_write_tuning_20261017.capture import TrainingCapture
    from experiments.osram_meld_fixed_rates_20261018.run import selected_result
    module = load_file('meld_related_history', ROOT / 'gcnet_missing_m3/related_history.py')
    integration = load_file('meld_related_integration', Path(__file__).with_name('integration.py'))
    integration.install(trainer, kind=args.kind, module_class=module.RelatedHistoryResidual)
    observe_training(trainer, args.output, integration, smoke=args.smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'RUN.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(numerical_commit=screen.original.HISTORICAL, extension=manifest,
            branch=branch_config(args.kind), dataset_sha256=screen.original.DATA_SHA,
            config_sha256=screen.original.sha(args.config), smoke=args.smoke)
        path = args.output / 'SUMMARY.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['identity'] != identity:
                raise ValueError('Source/config changed on resume')
            if previous['status'] in ('complete', 'smoke-complete'):
                return
        summary = dict(status='running', kind=args.kind, config=config, branch=branch_config(args.kind),
            identity=identity, environment=environment, gpu=gpu, pid=os.getpid(), started_unix=time.time(),
            control=str(CONTROL), from_scratch=True, weights_loaded_from_control=False,
            protocol='INTERNAL TEST-ORACLE SCREEN; ACC-best corresponding UA/W-F1', smoke=args.smoke)
        write_json(path, summary)
        capture = TrainingCapture(trainer, args.output)
        Parent = screen.original.state_class(TrainingState, capture, args.output, 'W0', args.output)
        reference = json.loads((CONTROL / 'MASK_TRACE.json').read_text())

        class BranchState(Parent):
            def bind(self, model, optimizer, **kwargs):
                super().bind(model, optimizer, **kwargs)
                if original_hash(model) != control['initial_state_sha256']:
                    raise ValueError('Original Flat initialization changed')
                branch = model.osram.related_history
                added = sum(p.numel() for p in branch.parameters())
                if capture.modelparameters - added != 13290795:
                    raise ValueError('Original Flat parameter set changed')
                write_json(args.output / 'PARAMETERS.json', dict(total=capture.modelparameters,
                    added=added, original=13290795, original_initialization_equal=True))
                return self

            def save_best(self, **kwargs):
                kwargs['checkpoint'] = dict(kwargs['checkpoint'], related_history=branch_config(args.kind))
                super().save_best(**kwargs)

            def commit_epoch(self, **kwargs):
                if not args.smoke:
                    screen.assert_epoch_masks(capture.trace, reference, kwargs['next_epoch'])
                super().commit_epoch(**kwargs)
                status_path = args.output / 'EPOCH_STATUS.json'
                status = json.loads(status_path.read_text())
                status.update(kind=args.kind, original_initialization_equal=True,
                    mask_stream_equal=not args.smoke)
                write_json(status_path, status)

        tick = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        try:
            effective = dict(config, epochs=1) if args.smoke else config
            with capture:
                trainer.run_experiment(trainer.TrainConfig(**effective), 'MELD-pickle', 'MELD-pickle',
                    'MELD-pickle', args.output, training_state=BranchState(args.output, identity=identity))
                capture.write_trace()
            parameters = json.loads((args.output / 'PARAMETERS.json').read_text())
            summary.update(parameters=parameters['total'], added_parameters=parameters['added'],
                original_initialization_equal=True,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            if args.smoke:
                summary.update(status='smoke-complete', performance_evidence=False)
            else:
                if len(json.loads((args.output / 'history.json').read_text())) != 50:
                    raise ValueError('Incomplete formal training')
                selected = {str(i/10): selected_result(args.output, i/10) for i in range(8)}
                summary.update(status='complete', testbest=screen.original.aggregate(selected))
        except BaseException as error:
            summary.update(status='failed', error=repr(error)); raise
        finally:
            summary.update(seconds=time.monotonic()-tick, finished_unix=time.time())
            write_json(path, summary)


def render(rows):
    lines = ['# MELD B1/B2 Local-history residual', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        'Seed66,50epochs,Adam lr1e-5 wd2e-4; ACC-best same-forward UA/W-F1.',
        'Original O6 Flat reused. All full forward Base/Gap slots preserved.', '',
        '| Model | ACC | UA | W-F1 | high ACC | high UA |', '|---|---:|---:|---:|---:|---:|']
    for kind, row in rows.items():
        scores = row['testbest']
        values = [scores['mean'][m] for m in ('ACC', 'UA', 'WF1')]
        values += [scores['high'][m] for m in ('ACC', 'UA')]
        lines.append('| ' + kind + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', '| Model | rate | epoch | ACC | UA | W-F1 |', '|---|---:|---:|---:|---:|---:|']
    for kind, row in rows.items():
        for rate, result in row['testbest']['rates'].items():
            lines.append(f'| {kind} | {rate} | {result["epoch"]} | ' +
                ' | '.join(f'{result[m]:.3f}' for m in ('ACC', 'UA', 'WF1')) + ' |')
    return '\n'.join(lines) + '\n'


class BranchScheduler(Scheduler):
    def __init__(self, output, costs):
        super().__init__(output / 'queue', gpus=(GPU,), per_gpu=2, code_root=ROOT, smoke_costs=costs)

    def command(self, job, config_path, gpu_uuid):
        return [self.python, '-u', str(Path(__file__)), '--worker', '--kind', job['method'],
            '--config', str(config_path), '--output', job['output'], '--gpu-uuid', gpu_uuid]


def controller(args):
    check_source()
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use --resume only for same-source queue')
    if shutil.disk_usage(args.output.parent).free < 10*2**30:
        raise ValueError('Need10GiB output space')
    costs = {}
    for kind in ('B1', 'B2'):
        smoke = json.loads((args.smoke_root / kind / 'SUMMARY.json').read_text())
        if smoke['status'] != 'smoke-complete' or smoke['branch'] != branch_config(kind):
            raise ValueError('Each approved block needs a real-batch smoke')
        costs[kind] = dict(peak_memory_mib=max(4096, smoke['peak_memory_mib']*1.5), disk_bytes=2*2**30)
    scheduler = BranchScheduler(args.output, costs)
    if scheduler.gpu_snapshot()[GPU]['uuid'] != UUID:
        raise ValueError('GPU0 identity changed')
    jobs = [dict(id=kind, method=kind, config=make_config(kind), output=str(args.output / kind))
        for kind in ('B1', 'B2')]
    completed = scheduler.run_jobs(jobs)
    rows = {'Flat': json.loads((CONTROL / 'SUMMARY.json').read_text())}
    rows.update({row['method']: row['summary'] for row in completed if row['status'] == 'complete'})
    result = dict(status='complete' if len(rows) == 3 else 'failed', models=rows,
        failures={row['id']: row.get('failures') for row in completed if row['status'] != 'complete'})
    write_json(args.output / 'RESULTS.json', result)
    (args.output / 'RESULT.md').write_text(render(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--resume', action='store_true')
    p.add_argument('--worker', action='store_true'); p.add_argument('--kind', choices=('B1', 'B2'))
    p.add_argument('--config', type=Path); p.add_argument('--gpu-uuid')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--smoke-root', type=Path)
    args = p.parse_args()
    if args.worker and any(getattr(args, name) is None for name in ('kind', 'config', 'gpu_uuid')):
        p.error('Worker requires kind/config/gpu-uuid')
    if not args.worker and args.smoke_root is None:
        p.error('Controller requires --smoke-root')
    (worker if args.worker else controller)(args)


if __name__ == '__main__':
    main()
