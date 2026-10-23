"""Finite cmerc Nested screen: six variants and one matched Flat per dataset.

This controller does not kill workers, silently retry, or expand the 14-job plan.
It is intended to run from an immutable committed worktree under tmux.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.osram_mosi_gcnet_router_3seed_20261019 import run as base
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib, write

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/data2/yb/conda_envs/cmerc-repro/bin/python'
MODULE = 'experiments.osram_nested_cmerc_tuning_20261023.worker'
GPU = 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'
GPU1 = 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'
BAD_UUID = base.BAD_UUID
METHODS = ('none', 'nested_gnn_rooted_evidence', 'nested_dim32', 'nested_dim128',
           'mosei_nested_depth2', 'mosei_nested_mlp128', 'mosei_nested_groups4')
RATES = tuple(f'{i / 10:.1f}' for i in range(8))


def validate_gpu_uuid(uuid):
    if uuid not in (GPU, GPU1):
        raise ValueError('Only the verified healthy GPU7/GPU1 UUID whitelist is authorized; GPU4 forbidden')


def plan_jobs(output, source_commit, gpu_uuid):
    validate_gpu_uuid(gpu_uuid)
    jobs = []
    for method in METHODS:
        for dataset, fold, epochs, learning_rate in (
                ('CMUMOSEI', 1, 60, 1e-4), ('IEMOCAPSix', 5, 100, 1e-3)):
            path = Path(output) / 'runs' / dataset / f'fold_{fold}' / 'seed_66' / method
            command = [PYTHON, '-u', '-m', MODULE, '--dataset', dataset, '--method', method,
                       '--seed', '66', '--fold', str(fold), '--phase', 'train',
                       '--gpu-uuid', gpu_uuid, '--source-commit', source_commit, '--output', str(path)]
            jobs.append(dict(dataset=dataset, method=method, seed=66, fold=fold, epochs=epochs,
                             learning_rate=learning_rate, phase='train', source_commit=source_commit,
                             gpu_uuid=gpu_uuid, output=str(path), log=str(path) + '.log',
                             status='pending', command=command, mask_protocol='gcnet-random-mask-v1'))
    return jobs


def can_admit(*, free_gpu_gib, ram_gib, disk_gib, active_count, per_gpu=4, reserve_mib=6144):
    if not 1 <= per_gpu <= 4:
        raise ValueError('Concurrency must be between one and four')
    if reserve_mib < 6144:
        raise ValueError('Per-worker reserve cannot be below six GiB')
    # Intentionally count existing allocations twice if already visible. This
    # protects against workers still importing/loading data before CUDA allocates.
    required_gib = (active_count + 1) * reserve_mib / 1024 + 2
    return (active_count < per_gpu and free_gpu_gib >= required_gib and ram_gib >= 64 and disk_gib >= 25)


def verify_source_snapshot(root, expected_commit):
    root = Path(root)
    snapshot_path = root / 'SOURCE_SNAPSHOT.json'
    if not (root / '.git').exists():
        snapshot = json.loads(snapshot_path.read_text())
        commit = snapshot['source_commit']
        files = snapshot['files']
        if not isinstance(files, dict) or not files:
            raise ValueError('Archive snapshot must enumerate committed Python/AGENTS source')
        for relative, expected_hash in files.items():
            path = root / relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Archive manifest source path escapes snapshot')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError('Archive source bytes differ from committed manifest: ' + relative)
        for directory in ('gcnet_missing_m3', 'gcnet_modality_jepa',
                          'experiments/osram_nested_cmerc_tuning_20261023'):
            for path in (root / directory).rglob('*.py'):
                if str(path.relative_to(root)) not in files:
                    raise ValueError('Unlisted Python source contaminates immutable archive: ' + str(path))
    else:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip():
            raise ValueError('Tracked source changes would contaminate this experiment')
        tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).split(b'\0')
        files = {}
        for raw in tracked:
            if not raw:
                continue
            relative = os.fsdecode(raw)
            if relative.endswith('.py') or Path(relative).name == 'AGENTS.md':
                files[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if commit != expected_commit:
        raise ValueError('Controller source commit does not match the immutable snapshot')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not (root / '.git').exists() and digest != snapshot['manifest_sha256']:
        raise ValueError('Archive manifest canonical digest differs')
    return dict(source_commit=commit, source_root=str(root), tracked_python_and_agents_sha256=files,
                manifest_sha256=digest)


def validate_summary(job):
    summary = json.loads((Path(job['output']) / 'SUMMARY.json').read_text())
    expected = dict(status='complete', phase='train', dataset=job['dataset'], method=job['method'],
                    seed=job['seed'], fold=job['fold'], source_commit=job['source_commit'], epoch=job['epochs'])
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f'Child summary {key} mismatch for {job["output"]}')
    if set(summary.get('evaluation', {})) != set(RATES) or 'router' not in summary.get('aggregate', {}):
        raise ValueError('Completion requires all eight selected evaluations and aggregate')
    return summary


def recover_jobs(state):
    """Only explicit recovery; living workers and failed jobs must not repeat."""
    for job in state['jobs']:
        if job['status'] == 'complete':
            validate_summary(job)
        elif job['status'] == 'running':
            try:
                os.kill(job['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Recorded child still alive; refuse duplicate submission')
            try:
                validate_summary(job)
                job['status'] = 'complete'
                continue
            except (OSError, ValueError):
                pass
            if not (Path(job['output']) / 'last.pt').exists():
                raise RuntimeError('Interrupted worker has no recovery checkpoint')
            job.update(status='pending', previous_pid=job['pid'])
            if '--resume' not in job['command']:
                job['command'].append('--resume')
        elif job['status'] in ('failed', 'blocked'):
            raise RuntimeError('Failed jobs are not automatically retried')
        elif job['status'] != 'pending':
            raise ValueError('Unknown persisted child status')


def poll(process, job):
    code = process.poll()
    if code is None:
        return False
    try:
        if code != 0:
            raise RuntimeError(f'Child exit status {code}')
        validate_summary(job)
        job['status'] = 'complete'
    except (OSError, ValueError, RuntimeError) as error:
        job.update(status='failed', error=repr(error))
    job.update(exit_code=code, finished_at=time.time())
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuid', default=GPU)
    parser.add_argument('--per-gpu', type=int, default=4)
    parser.add_argument('--reserve-mib', type=int, default=6144)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.per_gpu <= 4 or args.poll_seconds <= 0 or args.reserve_mib < 6144:
        raise ValueError('Require concurrency 1–4 and a positive polling interval')
    validate_gpu_uuid(args.gpu_uuid)
    args.output = args.output.resolve()
    manifest = verify_source_snapshot(ROOT, args.source_commit)
    jobs = plan_jobs(args.output, args.source_commit, args.gpu_uuid)
    # Verify the physical identity before creating a queue, not just CUDA's ordinal.
    devices = base.devices([args.gpu_uuid])
    if args.output.exists() and not args.resume:
        raise FileExistsError('Independent output required; existing queue must not submit twice')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output / 'QUEUE.json'
    identity = dict(source_commit=args.source_commit, gpu_uuid=args.gpu_uuid, maximum_parallel=args.per_gpu,
                    reserve_mib=args.reserve_mib, source_manifest_sha256=manifest['manifest_sha256'],
                    planned_jobs=copy.deepcopy(jobs))
    if args.resume:
        state = json.loads(path.read_text())
        if state.get('identity') != identity or state['status'] == 'complete':
            raise ValueError('Queue identity differs or queue already completed')
        recover_jobs(state)
        state.update(pid=os.getpid(), recovery_started_at=time.time(), status='resuming')
        state.pop('error', None)
    else:
        state = dict(status='preflight', identity=identity, source_commit=args.source_commit,
                     gpu_uuid=args.gpu_uuid, maximum_parallel=args.per_gpu, pid=os.getpid(), jobs=jobs,
                     started_at=time.time(), label='INTERNAL TEST-ORACLE SCREEN', maximum_jobs=14,
                     minimum_ram_gib=64, minimum_disk_gib=25, reserved_worker_gib=args.reserve_mib / 1024,
                     admission_margin_gib=2, gpu_preflight=devices)
        write(args.output / 'SOURCE_MANIFEST.json', manifest)
    children = {}
    base.ROOT = ROOT

    def record():
        state['updated_at'] = time.time()
        write(path, state)

    def drain():
        # No kills: if admission/control fails, already admitted peers may finish.
        for index, process in list(children.items()):
            process.wait()
            poll(process, state['jobs'][index])
            del children[index]
            record()

    try:
        record()
        while True:
            for index, process in list(children.items()):
                if poll(process, state['jobs'][index]):
                    del children[index]
            failed = any(job['status'] == 'failed' for job in state['jobs'])
            pending = [i for i, job in enumerate(state['jobs']) if job['status'] == 'pending']
            if failed:
                state['status'] = 'failure_draining_active'
                if not children:
                    for index in pending:
                        state['jobs'][index].update(status='blocked', error='Earlier worker failed; no retry/expansion')
                    raise RuntimeError('Worker failed; admitted peers finished, remaining jobs blocked')
            elif not children and not pending:
                from .report import write_report
                write_report(args.output, state)
                state.update(status='complete', finished_at=time.time())
                record()
                return
            else:
                found = base.devices([args.gpu_uuid])[args.gpu_uuid]
                resources = dict(free_gpu_gib=(found['total_mib'] - found['used_mib']) / 1024,
                                 ram_gib=available_gib(), disk_gib=shutil.disk_usage(args.output).free / 2**30,
                                 active_count=len(children), per_gpu=args.per_gpu, reserve_mib=args.reserve_mib)
                state['resources'] = resources
                if pending and can_admit(**resources):
                    # Revalidate immutable source before each new process.
                    if verify_source_snapshot(ROOT, args.source_commit) != manifest:
                        raise ValueError('Source changed while this queue was active')
                    index = pending[0]
                    children[index] = base.launch(state['jobs'][index])
                    state['status'] = 'running'
                    record()
                    time.sleep(4)
                    continue
                state['status'] = 'running' if children else 'waiting_for_resources'
            record()
            time.sleep(args.poll_seconds)
    except BaseException as error:
        state.update(status='failure_draining_active', error=repr(error))
        record()
        drain()
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     finished_at=time.time())
        record()
        raise


if __name__ == '__main__':
    main()
