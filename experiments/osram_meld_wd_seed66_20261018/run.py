"""Complete four WD configurations with seed66; reuse all completed67/68 runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.osram_meld_optimizer_screen_20261018 import run as screen
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler, write_json

JOBS = {f'S66_WD{i}': wd for i, wd in enumerate((0., 1e-5, 1e-4, 5e-4))}
CONTROL = Path('/home/qinjinghui/remote_experiments/osram_meld_optimizer_screen_20261018/runs_v1/O6')
PREVIOUS = Path('/home/qinjinghui/remote_experiments/osram_meld_fixed_lr_wd_20261018/runs_v1')
GPU_UUIDS = {2: 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
_CHECK_SHARED = screen.check_snapshot


def control_config():
    return dict(screen.source_config(), learning_rate=1e-5, weight_decay=2e-4)


def make_config(job):
    if job not in JOBS:
        raise ValueError('Only the four missing seed66 configurations are authorized')
    return dict(control_config(), weight_decay=JOBS[job])


def check_control(path):
    summary = json.loads((path / 'SUMMARY.json').read_text())
    trace = json.loads((path / 'MASK_TRACE.json').read_text())
    initial = json.loads((path / 'INITIAL_STATE.json').read_text())
    expected = control_config()
    if (summary['status'] != 'complete' or summary['config'] != expected
            or summary['parameters'] != 13290795
            or summary['identity']['numerical_commit'] != screen.original.HISTORICAL
            or summary['identity']['dataset_sha256'] != screen.original.DATA_SHA
            or initial != dict(sha256=summary['initial_state_sha256'], parameters=13290795)):
        raise ValueError('Need unchanged completed O6 seed66 control')
    epochs = {str(i) for i in range(1, 51)}
    if set(trace['train']) != epochs or set(trace['selection']) != epochs:
        raise ValueError('Require all50 actual baseline mask traces')
    for epoch in epochs:
        screen.assert_epoch_masks(trace, trace, int(epoch))
    return summary


def check_source(commit):
    manifest = _CHECK_SHARED()
    extension = json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())
    for name, digest in extension['files'].items():
        if screen.original.sha(ROOT / name) != digest:
            raise ValueError('Immutable seed66 driver changed: ' + name)
    if not commit or commit == 'unrecorded':
        raise ValueError('Actual committed driver identity required')
    return dict(manifest, git_sha=commit, seed66_extension=extension)


def worker(args):
    expected = make_config(args.job)
    if json.loads(args.config.read_text()) != expected:
        raise ValueError('Unexpected training configuration')
    # Reuse the identical audited training worker, numerical model, sampler,
    # exact-selection capture, optimizer/RNG recovery and epoch mask checks.
    # These adaptations affect only orchestration in this child interpreter.
    screen.JOB_SPECS = {job: (1e-5, wd) for job, wd in JOBS.items()}
    screen.GPU_UUIDS = dict(GPU_UUIDS)
    screen.validate_control = check_control
    screen.check_snapshot = lambda: check_source(args.driver_commit)
    args.control = CONTROL
    args.model_source = screen.MODEL_SOURCE
    screen.worker(args)


def aggregate_three(rows, wd):
    if len(rows) != 3 or {r['config']['seed'] for r in rows} != {66, 67, 68}:
        raise ValueError('Require exactly completed seeds66/67/68')
    expected = control_config()
    for row in rows:
        if row['status'] != 'complete' or row['config'] != dict(expected,
                seed=row['config']['seed'], weight_decay=wd):
            raise ValueError('Incomplete or protocol-mismatched three-seed result')
    result = dict(N_seeds=3, seeds=[66, 67, 68], weight_decay=wd)
    for scope in ('mean', 'high'):
        for suffix, func in (('', statistics.mean), ('_sample_std', statistics.stdev)):
            result[scope + suffix] = {metric: func(row['testbest'][scope][metric] for row in rows)
                for metric in ('ACC', 'UA', 'WF1')}
    return result


def render(result):
    lines = ['# MELD fixed LR WD: completed three-seed confirmation', '',
        'INTERNAL TEST-ORACLE SCREEN; per-rate ACC-best same-forward UA/W-F1.',
        'Only four missing seed66 runs are new; seeds67/68 and WD2e-4 controls reused.',
        'Original Flat1600 + Local Skip,50epochs,batch32,Adam lr1e-5,GCNet random-mask-v1.', '',
        '| WD | three-seed ACC | UA | W-F1 | high ACC | high UA |',
        '|---:|---:|---:|---:|---:|---:|']
    for wd, row in sorted(result['aggregate'].items(), key=lambda item: float(item[0])):
        cells = [f'{row[scope][metric]:.3f} ± {row[scope+"_sample_std"][metric]:.3f}'
            for scope, metric in (('mean','ACC'), ('mean','UA'), ('mean','WF1'), ('high','ACC'), ('high','UA'))]
        lines.append(f'| {float(wd):g} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| Seed | WD | ACC | UA | W-F1 |', '|---:|---:|---:|---:|---:|']
    for row in sorted(result['runs'], key=lambda r: (r['config']['weight_decay'], r['config']['seed'])):
        cfg = row['config']
        lines.append(f'| {cfg["seed"]} | {cfg["weight_decay"]:g} | ' +
            ' | '.join(f'{row["testbest"]["mean"][m]:.3f}' for m in ('ACC','UA','WF1')) + ' |')
    return '\n'.join(lines) + '\n'


class Seed66Scheduler(Scheduler):
    def __init__(self, args):
        self.args = args
        super().__init__(args.output / 'queue', gpus=(2,), per_gpu=4, code_root=ROOT,
            smoke_costs={'Flat': dict(peak_memory_mib=4096, disk_bytes=2*2**30)})

    def command(self, job, config_path, gpu_uuid):
        return [self.python, '-u', str(Path(__file__)), '--worker', '--job', job['id'],
            '--config', str(config_path), '--output', job['output'], '--gpu-uuid', gpu_uuid,
            '--driver-commit', self.args.driver_commit]


def controller(args):
    check_source(args.driver_commit)
    control = check_control(CONTROL)
    previous = json.loads((PREVIOUS / 'RESULTS.json').read_text())
    if previous['status'] != 'complete' or len(previous['results']) != 8:
        raise ValueError('Require all eight completed previous67/68 runs')
    reused = list(previous['controls'].values()) + list(previous['results'].values())
    for row in reused:
        if row['status'] != 'complete':
            raise ValueError('Cannot reuse an incomplete run')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Existing output requires explicit same-source --resume')
    if shutil.disk_usage(args.output.parent).free < 12*2**30:
        raise ValueError('Need12GiB output space')
    scheduler = Seed66Scheduler(args)
    if scheduler.gpu_snapshot()[2]['uuid'] != GPU_UUIDS[2]:
        raise ValueError('Physical GPU2 UUID changed')
    jobs = [dict(id=job, method='Flat', config=make_config(job), output=str(args.output / job)) for job in JOBS]
    finished = scheduler.run_jobs(jobs)
    runs = reused + [control] + [r['summary'] for r in finished if r['status'] == 'complete']
    aggregates = {}
    for wd in (0., 1e-5, 1e-4, 2e-4, 5e-4):
        rows = [r for r in runs if r['config']['weight_decay'] == wd]
        if len(rows) == 3:
            aggregates[str(wd)] = aggregate_three(rows, wd)
    result = dict(status='complete' if len(aggregates) == 5 else 'failed', runs=runs, aggregate=aggregates,
        previous_output=str(PREVIOUS), control_output=str(CONTROL),
        previous_result_sha256=screen.original.sha(PREVIOUS / 'RESULTS.json'),
        failures={r['id']: r.get('failures') for r in finished if r['status'] != 'complete'})
    write_json(args.output / 'RESULTS.json', result)
    (args.output / 'RESULT.md').write_text(render(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--driver-commit', required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--worker', action='store_true')
    p.add_argument('--job', choices=tuple(JOBS)); p.add_argument('--config', type=Path); p.add_argument('--gpu-uuid')
    args = p.parse_args()
    if args.worker and any(getattr(args, n) is None for n in ('job','config','gpu_uuid')):
        p.error('Worker needs job/config/gpu-uuid')
    (worker if args.worker else controller)(args)


if __name__ == '__main__':
    main()
