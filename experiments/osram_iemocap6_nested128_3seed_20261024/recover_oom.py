"""One audited resource-only recovery; import numerical code from original archive.

The five-worker admission underestimated CUDA context/reservation overhead.
Keep every training identity unchanged, cap admission at three, and restore the
one failed worker from its completed epoch checkpoint. Never retry twice.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def prepare(state, check_failed):
    if state.get('oom_recovery'):
        raise ValueError('OOM recovery already attempted; no repeated retries')
    failed = [job for job in state['jobs'] if job['status'] == 'failed']
    if len(failed) != 1 or (failed[0]['seed'], failed[0]['fold']) != (68, 5):
        raise ValueError('Unexpected failed trajectory; inspect before recovery')
    if any(job['status'] not in ('complete', 'failed', 'blocked') for job in state['jobs']):
        raise ValueError('Unexpected active or pending queue; refuse duplicate submission')
    check_failed(failed[0])
    repaired = copy.deepcopy(state)
    for job in repaired['jobs']:
        if job['status'] == 'failed':
            job['previous_failure'] = dict(pid=job.get('pid'), error=job.get('error'))
            job['status'] = 'pending'
            if '--resume' not in job['command']:
                job['command'].append('--resume')
        elif job['status'] == 'blocked':
            job['status'] = 'pending'
        if job['status'] == 'pending':
            job.pop('error', None)
    repaired['oom_recovery'] = dict(attempt=1, maximum_parallel=3,
        reserve_mib=8192, margin_mib=2048, recorded_at=time.time(),
        reason='Five concurrent CUDA contexts exhausted GPU0; training unchanged')
    return repaired


def can_admit(**resources):
    # Do not alter the persisted numerical training/seed/mask identity.
    if resources['active_count'] >= 3:
        return False
    if resources['ram_gib'] < 64 or resources['disk_gib'] < 25:
        return False
    external = max(0, resources['used_mib'] - resources['own_used_mib'])
    reserved = max(resources['active_count'] * 8192, resources['own_used_mib'])
    return external + reserved + 8192 + 2048 <= resources['total_mib']


def check_failed(job):
    import torch
    from experiments.osram_iemocap6_gcnet_testbest_20261020.worker import validate_selected
    output = Path(job['output'])
    summary = json.loads((output / 'SUMMARY.json').read_text())
    if 'CUDA out of memory' not in summary.get('error', ''):
        raise ValueError('Not the diagnosed CUDA OOM; refuse automatic recovery')
    state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    # SUMMARY is JSON (tuples become lists); keep checkpoint identity untouched
    # for the original worker's later strict in-memory comparison.
    if json.loads(json.dumps(state['identity'])) != summary['identity'] or state['identity']['seed'] != 68:
        raise ValueError('Recovery checkpoint identity differs')
    if state['epoch'] != 11 or len(state['history']) != state['epoch']:
        raise ValueError('Expected complete epoch11, not a partially updated model')
    if len(state['rng']['cuda']) != 1 or not state['optimizer']['state']:
        raise ValueError('Require full CUDA RNG and Adam state')
    validate_selected(output, state['best_by_rate'])
    return dict(epoch=state['epoch'], cuda_rng_states=1,
        checkpoint_sha256=hashlib.sha256((output / 'last.pt').read_bytes()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.source_root.resolve()))
    from experiments.osram_iemocap6_nested128_3seed_20261024 import run
    original_recover = run.recover_jobs
    ledger = json.loads((args.output / 'QUEUE.json').read_text())
    try:
        os.kill(ledger['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Previous controller PID is still alive; refuse duplicates')

    def recover_locked(state):
        # Original controller owns QUEUE.lock before this callback.
        checked = {}
        def validate(job):
            checked.update(check_failed(job))
        candidate = prepare(state, validate)
        original_recover(candidate)  # validates all completed jobs unchanged
        for job in candidate['jobs']:
            if job['status'] == 'pending' and '--resume' not in job['command']:
                if Path(job['output']).exists():
                    raise ValueError('Blocked fresh job unexpectedly has output')
        failed = next(job for job in state['jobs'] if job['status'] == 'failed')
        candidate['oom_recovery'].update(checked,
            recovery_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_source_commit=state['source_commit'])
        for name, payload in (
            ('OOM_FAILURE_QUEUE.json', state),
            ('OOM_FAILED_SUMMARY.json', json.loads((Path(failed['output']) / 'SUMMARY.json').read_text())),
        ):
            with (args.output / name).open('x') as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
        state.clear()
        state.update(candidate)
    run.recover_jobs = recover_locked
    run.can_admit = can_admit
    run.main(['--output', str(args.output), '--source-commit', ledger['source_commit'],
              '--gpu-uuid', ledger['gpu_uuid'], '--parallel', '6', '--resume'])


if __name__ == '__main__':
    main()
