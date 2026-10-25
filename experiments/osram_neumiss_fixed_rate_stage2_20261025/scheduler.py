"""Finite frozen-parent fixed-rate NeuMiss/Signature stage-two queue."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_three_block_joint_mosi_20261025.scheduler import (
    guard_gpu, now, resources, save)


MODES = ('neumiss', 'signature')
RATES = tuple(f'{i / 10:.1f}' for i in range(8))
MODULE = 'experiments.osram_neumiss_fixed_rate_stage2_20261025.run'
PARENT = Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2')


def capacity(snapshot):
    if snapshot['ram_available_gib'] < 32 or snapshot['disk_free_gib'] < 16:
        raise RuntimeError('Need 32 GiB available RAM and 16 GiB free disk')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def jobs_for(phase):
    rates = ('0.7',) if phase == 'smoke' else RATES
    return [dict(id=f'{mode}_{rate}', mode=mode, rate=rate, status='pending')
            for rate in rates for mode in MODES]


def launch(args, job, uuid, phase):
    output = args.root / phase / job['id']
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', '-m', MODULE, '--mode', job['mode'],
               '--rate', job['rate'], '--gpu-uuid', uuid, '--output', str(output),
               '--source-commit', args.source_commit]
    if phase == 'smoke':
        command.append('--smoke')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    log_path = output / 'worker.log'
    log = log_path.open('a', buffering=1)
    try:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    except BaseException:
        log.close()
        raise
    job.update(status='running', pid=process.pid, command=command, gpu_uuid=uuid,
               output=str(output), log=str(log_path), started_at=now(),
               environment={key: environment.get(key) for key in (
                   'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS',
                   'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    return process, log, job


def finish(job, code, phase):
    job.update(returncode=code, finished_at=now())
    try:
        if code:
            raise RuntimeError(f'Worker exited {code}; see {job["log"]}')
        path = Path(job['output']) / 'SUMMARY.json'
        summary = json.loads(path.read_text())
        expected = 'smoke_passed' if phase == 'smoke' else 'complete'
        if summary.get('status') != expected:
            raise ValueError(f'Expected SUMMARY status {expected}')
        peak = float(summary['peak_memory_mib'])
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError('Missing positive measured GPU peak')
        job.update(status=expected, summary=str(path), peak_memory_mib=peak)
    except Exception as error:
        job.update(status='failed', error=str(error))


def run_phase(args, state, phase, peaks=None):
    jobs = state[phase] = jobs_for(phase)
    active = []
    try:
        while any(job['status'] in ('pending', 'running') for job in jobs):
            for process, log, job in active[:]:
                code = process.poll()
                if code is not None:
                    log.close()
                    finish(job, code, phase)
                    active.remove((process, log, job))
            # Failed jobs are retained; never retry or admit new work after failure.
            failed = any(job['status'] == 'failed' for job in jobs)
            if failed:
                for job in jobs:
                    if job['status'] == 'pending':
                        job.update(status='blocked', error='Another finite-queue job failed')
            else:
                snapshot = resources(args.root, [args.gpu0_uuid, args.gpu3_uuid])
                state['latest_resources'] = snapshot
                capacity(snapshot)
                for job in jobs:
                    if job['status'] != 'pending':
                        continue
                    candidates = [args.gpu3_uuid] if phase == 'smoke' else [args.gpu0_uuid, args.gpu3_uuid]
                    for uuid in candidates:
                        peers = [entry for _, _, entry in active if entry['gpu_uuid'] == uuid]
                        limit = 2 if phase == 'smoke' else 4
                        if len(peers) >= limit:
                            continue
                        if phase != 'smoke':
                            # Already-resident allocations reduce free memory; reserve each peer's
                            # full smoke peak as an additional conservative transient margin.
                            required = peaks[job['mode']] + sum(peaks[p['mode']] for p in peers) + 4096
                            if snapshot['gpus'][uuid]['free_mib'] < required:
                                continue
                        active.append(launch(args, job, uuid, phase))
                        if phase != 'smoke':
                            snapshot['gpus'][uuid]['free_mib'] -= peaks[job['mode']]
                        break
            state['updated_at'] = now()
            save(args.root / 'QUEUE.json', state)
            if not active and any(job['status'] == 'pending' for job in jobs):
                raise RuntimeError('No job fits measured GPU capacity; hyperparameters unchanged')
            if active:
                time.sleep(5)
    finally:
        for _, log, _ in active:
            log.close()
    if any(job['status'] != ('smoke_passed' if phase == 'smoke' else 'complete') for job in jobs):
        raise RuntimeError(f'{phase} contains failed or blocked jobs; see QUEUE.json')


def collect(args, state):
    accepted = json.loads((PARENT / 'SUMMARY.json').read_text())
    rows, results = [], {}
    for job in state['train']:
        summary = json.loads(Path(job['summary']).read_text())
        identity = summary['identity']
        parent = identity.get('parent', identity.get('parent_identity', identity))
        rate, mode = job['rate'], job['mode']
        selected = accepted['best_by_rate'][rate]
        expected_hash = selected['checkpoint_sha256']
        actual_hash = digest(PARENT / selected['checkpoint'])
        expected_epoch = 0 if mode == 'neumiss' and rate == '0.0' else 150
        if summary.get('status') != 'complete' or summary['epoch'] != expected_epoch:
            raise ValueError(f'{job["id"]}: expected complete epoch {expected_epoch}')
        if (summary['parent_id'] != expected_hash or actual_hash != expected_hash
                or parent.get('checkpoint_sha256', parent.get('parent_id')) != expected_hash):
            raise ValueError(f'{job["id"]}: accepted parent checkpoint hash mismatch')
        if identity['training_rate'] != rate or identity['evaluation_rate'] != rate:
            raise ValueError(f'{job["id"]}: training/evaluation must match its own rate')
        baseline, best = summary['baseline'], summary['best']['metrics']
        row = dict(method=mode, rate=rate, ACC=best['ACC'], WF1=best['WF1'], MSE=best['MSE'],
                   parent_ACC=baseline['ACC'], parent_WF1=baseline['WF1'], parent_MSE=baseline['MSE'],
                   delta_ACC=best['ACC']-baseline['ACC'], delta_WF1=best['WF1']-baseline['WF1'],
                   delta_MSE=best['MSE']-baseline['MSE'])
        rows.append(row)
        results[job['id']] = dict(row, parent_id=expected_hash, epoch=summary['epoch'],
                                  summary=job['summary'])
    if len(results) != 16:
        raise ValueError('Exactly sixteen unique logical results are required')
    aggregate = []
    for mode in MODES:
        for scope, subset in (('mean', RATES), ('high', RATES[5:])):
            selected_rows = [row for row in rows if row['method'] == mode and row['rate'] in subset]
            aggregate.append(dict(method=mode, rate=scope, **{
                key: sum(row[key] for row in selected_rows)/len(subset)
                for key in rows[0] if key not in ('method', 'rate')}))
    save(args.root / 'RESULT.json', dict(status='complete', source_commit=args.source_commit,
         completed_at=now(), protocol='fixed-rate frozen-parent internal Test-oracle exploration',
         results=results, aggregate=aggregate))
    all_rows = rows + aggregate
    with (args.root / 'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    lines = ['# Fixed-rate frozen-parent stage-two MOSI66', '',
             f'Source: `{args.source_commit}`. Internal Test-oracle exploration.', '',
             'NeuMiss rate 0.0 is an epoch-zero replay/no-op; the other fifteen runs train 150 epochs.', '',
             '| Method | Rate/scope | ACC | WF1 | MSE | Parent WF1 | Δ ACC | Δ WF1 | Δ MSE |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in aggregate + rows:
        fields = ('ACC', 'WF1', 'MSE', 'parent_WF1', 'delta_ACC', 'delta_WF1', 'delta_MSE')
        lines.append(f'| {row["method"]} | {row["rate"]} | ' +
                     ' | '.join(f'{row[key]:.4f}' for key in fields) + ' |')
    (args.root / 'RESULT.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu0-uuid', required=True)
    parser.add_argument('--gpu3-uuid', required=True)
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), started_at=now(),
                 source_commit=args.source_commit, command=[sys.executable]+sys.argv,
                 python=sys.executable, cwd=os.getcwd(), max_per_gpu=4,
                 gpu0_uuid=args.gpu0_uuid, gpu3_uuid=args.gpu3_uuid)
    save(args.root / 'QUEUE.json', state)
    try:
        previous = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            for index, uuid in ((0, args.gpu0_uuid), (3, args.gpu3_uuid)):
                os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                guard_gpu(uuid, index)
        finally:
            if previous is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = previous
        state['preflight'] = resources(args.root, [args.gpu0_uuid, args.gpu3_uuid])
        capacity(state['preflight'])
        state['status'] = 'smoke'
        run_phase(args, state, 'smoke')
        peaks = {job['mode']: job['peak_memory_mib'] for job in state['smoke']}
        state.update(status='train', admission_smoke_peaks_mib=peaks)
        run_phase(args, state, 'train', peaks)
        collect(args, state)
        state.update(status='complete', completed_at=now())
        save(args.root / 'QUEUE.json', state)
        return 0
    except Exception as error:
        state.update(status='failed', error=str(error), finished_at=now())
        save(args.root / 'QUEUE.json', state)
        print(f'Queue failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
