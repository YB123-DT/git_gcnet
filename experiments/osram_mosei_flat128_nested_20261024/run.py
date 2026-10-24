"""Finite Flat128 control / Nested screen: 23 fresh jobs and one reused result.

Physical GPU0/1 each admit at most six recorded workers, including the old
capacity queue. Measured reservations and live resources may reduce concurrency.
No process is killed and no failed job is automatically retried.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from experiments.osram_iemocap6_nested128_3seed_20261024.run import own_gpu_memory_mib
from experiments.osram_nested_cmerc_tuning_20261023 import run as common
from . import config

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/data2/yb/conda_envs/cmerc-repro/bin/python'
MODULE = 'experiments.osram_mosei_flat128_nested_20261024.worker'
GPU0 = 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45'
GPU1 = 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'
BAD_UUID = common.BAD_UUID
OLD_ROOT = Path('/data2/yb/remote_experiments/osram_mosei_capacity_cmerc_20261024')
OLD_QUEUE = OLD_ROOT / 'execution_v1/QUEUE.json'
REUSED_SOURCE = OLD_ROOT / 'source_v1'
MARGIN_MIB = 2048
MAX_PER_GPU = 6
MAX_GLOBAL = 12
INITIAL0 = ('F67', 'F68', 'N19', 'N20', 'N01', 'N02')
INITIAL1 = ('N03', 'N04', 'N05', 'N06', 'N07', 'N08')
ORDER = INITIAL0 + INITIAL1 + tuple(f'N{i:02d}' for i in range(9, 19)) + ('N21',)


def validate_gpu_uuid(uuid):
    if uuid not in (GPU0, GPU1):
        raise ValueError('Only biggpu physical GPU0/1 are authorized; GPU4 forbidden')


def _reserve(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Resource proof must contain numeric MiB reservations')
    if not math.isfinite(value) or value < 4096 or int(value) != value:
        raise ValueError('Resource reservations must be finite integral MiB, at least 4096')
    return int(value)


def load_resource_proof(path):
    """Require measured small/grouped, default and largest-variant anchors."""
    data = Path(path).read_bytes()
    proof = json.loads(data)
    if proof.get('status') != 'passed':
        raise ValueError('Resource measurements must pass before queue submission')
    _reserve(proof.get('default_reserve_mib'))
    variants = proof.get('variants', {})
    if not isinstance(variants, dict) or not {'N01', 'N20', 'N21'} <= set(variants):
        raise ValueError('Resource proof requires N01, N20 and N21 measurements')
    for variant, value in variants.items():
        if variant not in ORDER or not isinstance(value, dict):
            raise ValueError('Unknown variant or malformed resource measurement')
        _reserve(value.get('reserve_mib'))
    return proof, hashlib.sha256(data).hexdigest()


def plan_jobs(output, source_commit, resource_proof):
    jobs = []
    default_reserve = _reserve(resource_proof['default_reserve_mib'])
    for variant in ORDER:
        spec = config.run_spec(variant)
        preferred = GPU0 if variant in INITIAL0 else GPU1 if variant in INITIAL1 else None
        uuid = preferred or GPU0
        path = Path(output) / 'runs' / variant.lower()
        reserve = max(default_reserve, _reserve(resource_proof['variants'].get(
            variant, {'reserve_mib': default_reserve})['reserve_mib']))
        command = [PYTHON, '-u', '-m', MODULE, '--variant', variant,
                   '--seed', str(spec['seed']), '--phase', 'train', '--gpu-uuid', uuid,
                   '--source-commit', source_commit, '--output', str(path)]
        jobs.append(dict(slot_id=variant, variant=variant, method=spec['method'],
            dataset='CMUMOSEI', seed=spec['seed'], fold=1, epochs=60, learning_rate=1e-4,
            phase='train', source_commit=source_commit, output=str(path), log=str(path) + '.log',
            command=command, gpu_uuid=uuid, preferred_gpu_uuid=preferred, reserve_mib=reserve,
            status='pending', mask_protocol='gcnet-random-mask-v1'))
    return jobs


def bind_gpu(job, uuid):
    """Bind both launch environment and worker argv to the physical UUID."""
    validate_gpu_uuid(uuid)
    job['gpu_uuid'] = uuid
    job['command'][job['command'].index('--gpu-uuid') + 1] = uuid


def validate_persisted_plan(jobs, planned):
    """Recovery may change status and GPU binding, never the finite protocol."""
    if len(jobs) != len(planned):
        raise ValueError('Persisted queue does not contain the exact planned runs')
    for job, expected in zip(jobs, planned):
        if any(job.get(key) != value for key, value in expected.items()
               if key not in ('status', 'gpu_uuid', 'command')):
            raise ValueError('Persisted job protocol or reservation differs from the plan')
        rebound = copy.deepcopy(expected)
        bind_gpu(rebound, job['gpu_uuid'])
        command = job.get('command')
        if command not in (rebound['command'], rebound['command'] + ['--resume']):
            raise ValueError('Persisted worker command differs from the plan')


def can_admit(*, total_mib, used_mib, own_used_mib, active_reserve_mib,
              proposed_mib, active_count, global_active_count, ram_gib, disk_gib):
    values = (total_mib, used_mib, own_used_mib, active_reserve_mib, proposed_mib,
              active_count, global_active_count, ram_gib, disk_gib)
    if not all(math.isfinite(float(v)) and v >= 0 for v in values) or proposed_mib <= 0:
        return False
    external = max(0, used_mib - min(own_used_mib, used_mib))
    reserved = max(active_reserve_mib, own_used_mib)
    return (active_count < MAX_PER_GPU and global_active_count < MAX_GLOBAL
            and ram_gib >= 64 and disk_gib >= 25
            and external + reserved + proposed_mib + MARGIN_MIB <= total_mib)


def recorded_worker_alive(job):
    """Avoid treating a recycled PID's unrelated allocation as our reservation."""
    try:
        os.kill(job['pid'], 0)
    except ProcessLookupError:
        return False
    try:
        actual = [os.fsdecode(v) for v in Path(f'/proc/{job["pid"]}/cmdline').read_bytes().split(b'\0') if v]
    except FileNotFoundError:
        return False
    if actual != job['command']:
        raise RuntimeError('Recorded old worker PID has a different command; refuse memory reclamation')
    return True


def external_workers(path=None):
    """Read the old controller each admission; count every still-live worker."""
    queue = json.loads(Path(path or OLD_QUEUE).read_text())
    workers = []
    for job in queue['jobs']:
        if job['status'] != 'running':
            continue
        validate_gpu_uuid(job['gpu_uuid'])
        _reserve(job['reserve_mib'])
        if recorded_worker_alive(job):
            workers.append(job)
            continue
        summary_path = Path(job['output']) / 'SUMMARY.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if summary.get('status') != 'complete' or summary.get('epoch') != 60:
            raise RuntimeError('Old capacity worker exited without completed 60-epoch output')
    return workers


def _devices():
    found = common.base.devices([GPU0, GPU1])
    if found[GPU0]['index'] != 0 or found[GPU1]['index'] != 1:
        raise ValueError('Physical GPU UUID mapping changed')
    return found


def _resources(uuid, active, output):
    device = _devices()[uuid]
    workers = [job for job in active if job['gpu_uuid'] == uuid]
    # Reserve each worker's unseen peak separately; an unusually large worker
    # cannot consume another still-loading worker's reservation.
    allocations = [own_gpu_memory_mib(uuid, [job['pid']]) for job in workers]
    return dict(total_mib=device['total_mib'], used_mib=device['used_mib'],
        own_used_mib=sum(allocations),
        active_reserve_mib=sum(max(job['reserve_mib'], used) for job, used in zip(workers, allocations)),
        active_count=len(workers), global_active_count=len(active),
        ram_gib=common.available_gib(), disk_gib=shutil.disk_usage(output).free / 2**30)


def main(argv=None, *, reporter=None, reused_validator=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--resource-proof', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.output.exists() and not args.resume:
        raise FileExistsError('Independent output required; an existing queue needs explicit --resume')
    proof, proof_sha256 = load_resource_proof(args.resource_proof)
    manifest = common.verify_source_snapshot(ROOT, args.source_commit)
    extension_proof = config.verify_registry_extension(ROOT)
    if reporter is None or reused_validator is None:
        from .report import summarize, validate_reused_control
        reporter = reporter or summarize
        reused_validator = reused_validator or validate_reused_control
    reused_proof = reused_validator(config.REUSED_CONTROL)
    reused = dict(slot_id='F66', variant='F66', seed=66, method='none', status='reused_complete',
                  output=str(config.REUSED_CONTROL))
    jobs = plan_jobs(args.output, args.source_commit, proof)
    devices = _devices()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue_path = args.output / 'QUEUE.json'
    identity = dict(source_commit=args.source_commit, source_manifest_sha256=manifest['manifest_sha256'],
        resource_proof_sha256=proof_sha256, registry_extension_proof=extension_proof,
        gpus=[GPU0, GPU1], maximum_per_gpu=MAX_PER_GPU, maximum_parallel=MAX_GLOBAL,
        margin_mib=MARGIN_MIB, reused=reused, reused_control_proof=reused_proof,
        planned_jobs=copy.deepcopy(jobs))
    if args.resume:
        state = json.loads(queue_path.read_text())
        if state.get('identity') != identity or state['status'] == 'complete':
            raise ValueError('Resume identity differs or this finite queue is already complete')
        validate_persisted_plan(state['jobs'], jobs)
        common.recover_jobs(state)
        state.update(status='resuming', pid=os.getpid(), recovery_started_at=time.time())
        state.pop('error', None)
    else:
        state = dict(status='preflight', pid=os.getpid(), source_commit=args.source_commit,
            identity=identity, jobs=jobs, reused=reused, reused_source=str(REUSED_SOURCE),
            reused_control_proof=reused_proof, source_extension_proof=extension_proof,
            resource_proof=proof, resource_proof_sha256=proof_sha256, gpu_preflight=devices,
            maximum_new_jobs=23, total_result_slots=24, maximum_per_gpu=MAX_PER_GPU,
            maximum_parallel=MAX_GLOBAL, minimum_ram_gib=64, minimum_disk_gib=25,
            admission_margin_mib=MARGIN_MIB, old_queue=str(OLD_QUEUE),
            started_at=time.time(), label='INTERNAL TEST-ORACLE SCREEN')
        common.write(args.output / 'SOURCE_MANIFEST.json', manifest)
        common.write(args.output / 'RESOURCE_PROOF.json', proof)
    children = {}
    common.base.ROOT = ROOT

    def record():
        state['updated_at'] = time.time()
        common.write(queue_path, state)

    def block_pending():
        for job in state['jobs']:
            if job['status'] == 'pending':
                job.update(status='blocked', error='Queue failed; no automatic retry or expansion')

    def drain():
        for index, process in list(children.items()):
            process.wait()
            common.poll(process, state['jobs'][index])
            del children[index]
            record()

    try:
        record()
        while True:
            for index, process in list(children.items()):
                if common.poll(process, state['jobs'][index]):
                    del children[index]
            pending = [i for i, job in enumerate(state['jobs']) if job['status'] == 'pending']
            failed = any(job['status'] == 'failed' for job in state['jobs'])
            if failed:
                state['status'] = 'failure_draining_active'
                if not children:
                    raise RuntimeError('A worker failed; remaining unstarted runs are blocked')
            elif not pending and not children:
                if len(state['jobs']) != 23 or any(job['status'] != 'complete' for job in state['jobs']):
                    raise RuntimeError('Completion requires all 23 fresh jobs and reused F66')
                reporter(args.output, state, final=True)
                state.update(status='complete', finished_at=time.time())
                record()
                return
            else:
                external = external_workers()
                live = [job for job in state['jobs'] if job['status'] == 'running'] + external
                state['external_active_workers'] = external
                state['resources'] = {}
                admitted = False
                for uuid in (GPU0, GPU1):
                    resources = _resources(uuid, live, args.output)
                    state['resources'][uuid] = resources
                    # Preserve the requested first assignments; later candidates
                    # can use either authorized physical GPU as a slot opens.
                    candidates = [i for i in pending if state['jobs'][i]['preferred_gpu_uuid'] in (None, uuid)]
                    index = next((i for i in candidates if can_admit(
                        **resources, proposed_mib=state['jobs'][i]['reserve_mib'])), None)
                    if index is None:
                        continue
                    if common.verify_source_snapshot(ROOT, args.source_commit) != manifest:
                        raise ValueError('Immutable source changed while queue was active')
                    if config.verify_registry_extension(ROOT) != extension_proof:
                        raise ValueError('Verified registry extension changed')
                    if load_resource_proof(args.resource_proof)[1] != proof_sha256:
                        raise ValueError('Measured resource proof changed')
                    bind_gpu(state['jobs'][index], uuid)
                    children[index] = common.base.launch(state['jobs'][index])
                    state['status'] = 'running'
                    record()
                    reporter(args.output, state, final=False)
                    time.sleep(4)
                    admitted = True
                    break  # Refresh resources and old-worker status before another launch.
                if admitted:
                    continue
                state['status'] = 'running' if children else 'waiting_for_resources'
            record()
            reporter(args.output, state, final=False)
            time.sleep(15)
    except BaseException as error:
        state.update(status='failure_draining_active', error=repr(error))
        if any(job['status'] == 'failed' for job in state['jobs']):
            block_pending()
        record()
        drain()
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     finished_at=time.time())
        record()
        reporter(args.output, state, final=False)
        raise


if __name__ == '__main__':
    main()
