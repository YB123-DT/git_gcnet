"""Adopt live workers without restart after the user's IEMOCAP4 cancellation.

Controller code may change; every retained worker keeps the OLD immutable source,
command, model/optimizer/RNG and output. Cancelled jobs remain in the audit ledger.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

from . import run


def active_jobs(state):
    return [j for j in state['jobs'] if j['dataset'] != 'IEMOCAPFour']


def reduce_state(state):
    result = copy.deepcopy(state)
    active = active_jobs(result)
    if (len(active) != 14 or {(j['dataset'], j['method']) for j in active}
            != {(d, m) for d in ('CMUMOSEI', 'IEMOCAPSix') for m in run.METHODS}):
        raise ValueError('Expected exactly14 retained tasks')
    for job in result['jobs']:
        if job['dataset'] == 'IEMOCAPFour':
            job.update(status_before_cancellation=job['status'], status='cancelled_by_user',
                cancellation_reason='User explicitly requested IEMOCAP6 only; no IEMOCAP4')
    result.update(maximum_jobs=14, scope_revision=dict(excluded_dataset='IEMOCAPFour',
        allowed_datasets=['CMUMOSEI', 'IEMOCAPSix'], original_ledger_jobs=len(result['jobs']), active_jobs=14))
    return result


def command_of(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()


class AdoptedProcess:
    def __init__(self, job):
        self.job = job

    def poll(self):
        pid = self.job['pid']
        try:
            status = Path(f'/proc/{pid}/status').read_text()
            if '\nState:\tZ' not in status:
                command = command_of(pid)
                if run.MODULE not in command or self.job['output'] not in command:
                    raise RuntimeError('Adopted PID no longer belongs to this task')
                return None
        except FileNotFoundError:
            pass
        try:
            run.validate_summary(self.job)
            return 0
        except (OSError, ValueError):
            return 1

    def wait(self):
        while self.poll() is None:
            time.sleep(5)
        return self.poll()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker-root', type=Path, required=True)
    p.add_argument('--source-commit', required=True, help='New supervisor commit, NOT retained worker SHA')
    args = p.parse_args()
    root = args.output.resolve()
    state = json.loads((root / 'QUEUE.json').read_text())
    receipt = json.loads((root / 'IEMOCAP4_SCOPE_CHANGE.json').read_text())
    run.verify_source_snapshot(run.ROOT, args.source_commit)
    manifest = run.verify_source_snapshot(args.worker_root, state['source_commit'])
    if manifest['manifest_sha256'] != state['identity']['source_manifest_sha256']:
        raise ValueError('Retained immutable worker source changed')
    reduced = reduce_state(state)
    old_pid = state['pid']
    if old_pid != receipt['controller_pid']:
        raise ValueError('Cancellation receipt targets a different controller')
    for job in active_jobs(state):
        if job['status'] == 'running':
            cmd = command_of(job['pid'])
            if run.MODULE not in cmd or job['output'] not in cmd:
                raise RuntimeError('Refuse to adopt unrelated process')
    # The operator already paused this exact parent before signalling only IEMOCAP4.
    cmd = command_of(old_pid)
    if 'experiments.osram_nested_cmerc_tuning_20261023.run' not in cmd or str(root) not in cmd:
        raise RuntimeError('Refuse to terminate an unrelated controller')
    if '\nState:\tT' not in Path(f'/proc/{old_pid}/status').read_text():
        raise RuntimeError('Old controller must be paused for atomic handoff')
    with (root / 'QUEUE_before_iemocap4_cancel.json').open('x') as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
    os.kill(old_pid, signal.SIGTERM)
    os.kill(old_pid, signal.SIGCONT)
    lock = (root / 'QUEUE.lock').open('a')
    for attempt in range(60):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(.1)
    else:
        raise RuntimeError('Old controller did not release its lock')
    state = reduced
    state.update(status='running', pid=os.getpid(), previous_controller_pid=old_pid,
        controller_source_commit=args.source_commit, scope_updated_at=time.time())
    run.base.ROOT = args.worker_root.resolve()
    children = {i: AdoptedProcess(j) for i, j in enumerate(state['jobs']) if j['status'] == 'running'}
    receipt.update(status='scope_reduced', controller_source_commit=args.source_commit,
        new_controller_pid=os.getpid(), retained_worker_source_commit=state['source_commit'],
        original_jobs=21, active_jobs=14, cancelled_jobs=7, retained_workers_restarted=False)
    run.write(root / 'IEMOCAP4_SCOPE_CHANGE.json', receipt)

    def record():
        state['updated_at'] = time.time()
        run.write(root / 'QUEUE.json', state)

    record()
    try:
        while True:
            for i, proc in list(children.items()):
                if run.poll(proc, state['jobs'][i]):
                    del children[i]
            jobs = active_jobs(state)
            pending = [i for i, j in enumerate(state['jobs']) if j['dataset'] != 'IEMOCAPFour' and j['status'] == 'pending']
            if any(j['status'] == 'failed' for j in jobs):
                raise RuntimeError('A retained worker failed; no automatic retry')
            if not children and not pending:
                from .report import write_report
                write_report(root, dict(state, jobs=jobs))
                state.update(status='complete', finished_at=time.time())
                record(); return
            device = run.base.devices([state['gpu_uuid']])[state['gpu_uuid']]
            resources = dict(free_gpu_gib=(device['total_mib']-device['used_mib'])/1024,
                ram_gib=run.available_gib(), disk_gib=shutil.disk_usage(root).free/2**30,
                active_count=len(children), per_gpu=state['maximum_parallel'],
                reserve_mib=state['identity']['reserve_mib'])
            state['resources'] = resources
            if pending and run.can_admit(**resources):
                if run.verify_source_snapshot(args.worker_root, state['source_commit']) != manifest:
                    raise RuntimeError('Retained worker source changed')
                i = pending[0]
                children[i] = run.base.launch(state['jobs'][i])
                record(); time.sleep(4); continue
            state['status'] = 'running' if children else 'waiting_for_resources'
            record(); time.sleep(20)
    except BaseException as error:
        state.update(status='failure_draining_active', error=repr(error)); record()
        for i, proc in children.items():
            proc.wait(); run.poll(proc, state['jobs'][i])
        state.update(status='failed', finished_at=time.time()); record()
        raise


if __name__ == '__main__':
    main()
