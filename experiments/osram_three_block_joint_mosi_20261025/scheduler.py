"""Finite four-arm MOSI queue; all real smoke runs precede formal training."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


MODES = ('flat', 'neumiss', 'signature', 'csdi')
MODULE = 'experiments.osram_three_block_joint_mosi_20261025.run'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_inventory():
    listing = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    return {row[1].strip(): {'host_index': int(row[0]), 'free_mib': float(row[2])}
            for line in listing.splitlines() if (row := line.split(','))}


def guard_gpu(uuid, expected_index):
    """Validate physical identity before any CUDA/torch import."""
    if not uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must match the explicit GPU UUID')
    record = gpu_inventory().get(uuid)
    if not record or record['host_index'] == 4 or record['host_index'] != expected_index:
        raise ValueError(f'{uuid} must identify healthy physical GPU {expected_index}')
    return dict(record, uuid=uuid)


def resources(root, uuids):
    inventory = gpu_inventory()
    available = next(int(line.split()[1]) * 1024 for line in
                     Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return {'at': now(), 'gpus': {u: inventory[u] for u in uuids},
            'ram_available_gib': available / 2**30,
            'disk_free_gib': shutil.disk_usage(root).free / 2**30}


def require_capacity(snapshot, needs):
    if snapshot['ram_available_gib'] < 24 or snapshot['disk_free_gib'] < 16:
        raise RuntimeError('Need at least 24 GiB available RAM and 16 GiB free disk')
    for uuid, required_mib in needs.items():
        if snapshot['gpus'][uuid]['free_mib'] < required_mib:
            raise RuntimeError(f'Insufficient free memory on {uuid}: need {required_mib:.1f} MiB')


def run_phase(args, state, phase):
    jobs = state[phase] = []
    active = []
    try:
        for mode in MODES:
            uuid = args.gpu1_uuid if mode == 'csdi' else args.gpu0_uuid
            output = args.root / phase / mode
            output.mkdir(parents=True)
            command = [sys.executable, '-u', '-m', MODULE, '--mode', mode,
                       '--output', str(output), '--gpu-uuid', uuid,
                       '--source-commit', args.source_commit]
            if phase == 'smoke':
                command.append('--smoke')
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid,
                               CUDA_DEVICE_ORDER='PCI_BUS_ID')
            log_path = output / 'worker.log'
            log = log_path.open('a', buffering=1)
            try:
                process = subprocess.Popen(command, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            job = dict(mode=mode, output=str(output), gpu_uuid=uuid,
                       command=command, pid=process.pid, status='running',
                       started_at=now(), log=str(log_path),
                       environment={key: environment.get(key) for key in
                                    ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                                     'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
            jobs.append(job)
            active.append((process, log, job))
            save(args.root / 'QUEUE.json', state)
        while active:
            for process, log, job in active[:]:
                code = process.poll()
                if code is None:
                    continue
                log.close()
                job.update(returncode=code, finished_at=now())
                try:
                    if code != 0:
                        raise RuntimeError(f'worker exited {code}; see {job["log"]}')
                    summary_path = Path(job['output']) / 'SUMMARY.json'
                    summary = json.loads(summary_path.read_text())
                    expected = 'smoke_passed' if phase == 'smoke' else 'complete'
                    if summary.get('status') != expected:
                        raise RuntimeError(f'Expected SUMMARY status {expected}')
                    for key, expected_value in (('mode', job['mode']),
                                                ('source_commit', args.source_commit)):
                        if key in summary and summary[key] != expected_value:
                            raise RuntimeError(f'SUMMARY {key} mismatch')
                    if phase == 'smoke':
                        peak = float(summary['peak_memory_mib'])
                        if not math.isfinite(peak) or peak <= 0:
                            raise RuntimeError('Smoke must report a positive measured GPU peak')
                    job.update(status=expected, summary=str(summary_path),
                               peak_memory_mib=summary.get('peak_memory_mib'))
                except Exception as error:
                    job.update(status='failed', error=str(error))
                active.remove((process, log, job))
            state['updated_at'] = now()
            save(args.root / 'QUEUE.json', state)
            if active:
                time.sleep(5)
    finally:
        # Never signal other processes. If this controller fails, retain child IDs.
        for _, log, _ in active:
            log.close()
    failed = [job['mode'] for job in jobs if job['status'] == 'failed']
    if failed:
        raise RuntimeError(f'{phase} failed: {", ".join(failed)}')


def collect(args, state):
    import csv

    summaries = {job['mode']: json.loads(Path(job['summary']).read_text())
                 for job in state['train']}
    if set(summaries) != set(MODES):
        raise ValueError('All four formal summaries are required')
    reference_trace = reference_masks = None
    compact = {}
    rates = tuple(f'{i / 10:.1f}' for i in range(8))
    for mode in MODES:
        summary = summaries[mode]
        output = Path(next(job['output'] for job in state['train'] if job['mode'] == mode))
        history = json.loads((output / 'history.json').read_text())
        if (summary.get('status') != 'complete' or summary.get('epoch') != 150
                or len(history) != 150):
            raise ValueError(f'{mode}: expected 150 completed epochs and history rows')
        trace = [(row['epoch'], row['optimizer_steps'], row['mask_trace_sha256'])
                 for row in history]
        if ([row[0] for row in trace] != list(range(1, 151))
                or any(row[1] <= 0 or not row[2] for row in trace)):
            raise ValueError(f'{mode}: invalid epoch/optimizer/mask trace')
        masks = summary['identity']['mask_hashes']
        if set(masks) != set(rates) or any(not value for value in masks.values()):
            raise ValueError(f'{mode}: eight exact test mask hashes required')
        if reference_trace is None:
            reference_trace, reference_masks = trace, masks
        elif trace != reference_trace or masks != reference_masks:
            raise ValueError(f'{mode}: optimizer steps or training/test masks differ across arms')
        if set(summary['best_by_rate']) != set(rates):
            raise ValueError(f'{mode}: eight selected rate results required')
        compact[mode] = dict(summary=str(output / 'SUMMARY.json'),
                             aggregate=summary['aggregate'],
                             per_rate={r: summary['best_by_rate'][r]['metrics']['router']
                                       for r in rates})
    result = dict(status='complete', source_commit=args.source_commit,
                  completed_at=now(), protocol='internal per-rate Test-oracle exploration',
                  verification=dict(epochs=150, matched_optimizer_steps=True,
                                    matched_training_mask_traces=True,
                                    matched_test_mask_hashes=reference_masks), methods=compact)
    save(args.root / 'RESULT.json', result)
    lines = ['# Four-arm MOSI joint-block result', '',
             f'Source commit: `{args.source_commit}`', '',
             'Internal per-rate Test-oracle exploration; four matched seed66 arms.', '',
             'Verified 150 epochs per arm, identical epoch optimizer counts and training mask traces, '
             'and identical eight-rate test mask hashes.', '',
             '| Method | Mean ACC | Mean WF1 | High WF1 | Mean MSE | Δ WF1 vs Flat |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    baseline = compact['flat']['aggregate']['router']['mean']['WF1']
    rows = []
    for mode in MODES:
        aggregate = compact[mode]['aggregate']['router']
        mean, high = aggregate['mean'], aggregate['high']
        values = [mean['ACC'], mean['WF1'], high['WF1'], mean['MSE'], mean['WF1'] - baseline]
        lines.append('| ' + mode + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
        rows.append(dict(method=mode, rate='mean', ACC=mean['ACC'], WF1=mean['WF1'],
                         high_WF1=high['WF1'], MSE=mean['MSE'], delta_WF1=mean['WF1']-baseline))
    lines.extend(['', '| Method | Missing rate | ACC | WF1 | MSE | Δ WF1 vs Flat |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for mode in MODES:
        for rate in rates:
            metrics = compact[mode]['per_rate'][rate]
            delta = metrics['WF1'] - compact['flat']['per_rate'][rate]['WF1']
            lines.append(f'| {mode} | {rate} | {metrics["ACC"]:.4f} | '
                         f'{metrics["WF1"]:.4f} | {metrics["MSE"]:.4f} | {delta:.4f} |')
            rows.append(dict(method=mode, rate=rate, ACC=metrics['ACC'], WF1=metrics['WF1'],
                             high_WF1='', MSE=metrics['MSE'], delta_WF1=delta))
    (args.root / 'RESULT.md').write_text('\n'.join(lines))
    with (args.root / 'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu0-uuid', required=True)
    parser.add_argument('--gpu1-uuid', required=True)
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), started_at=now(),
                 source_commit=args.source_commit, command=[sys.executable] + sys.argv,
                 python=sys.executable, cwd=os.getcwd(), modes=list(MODES),
                 gpu0_uuid=args.gpu0_uuid, gpu1_uuid=args.gpu1_uuid)
    save(args.root / 'QUEUE.json', state)
    try:
        prior = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            for index, uuid in ((0, args.gpu0_uuid), (1, args.gpu1_uuid)):
                os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                guard_gpu(uuid, index)
        finally:
            if prior is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = prior
        uuids = [args.gpu0_uuid, args.gpu1_uuid]
        state['preflight'] = resources(args.root, uuids)
        require_capacity(state['preflight'], {args.gpu0_uuid: 16 * 1024,
                                              args.gpu1_uuid: 24 * 1024})
        state['status'] = 'smoke'
        run_phase(args, state, 'smoke')
        peaks = {job['mode']: job['peak_memory_mib'] for job in state['smoke']}
        needs = {args.gpu0_uuid: sum(peaks[m] for m in MODES[:3]) + 4096,
                 args.gpu1_uuid: peaks['csdi'] + 4096}
        state['formal_admission'] = dict(resources=resources(args.root, uuids),
                                         required_mib=needs, smoke_peaks_mib=peaks)
        require_capacity(state['formal_admission']['resources'], needs)
        state['status'] = 'train'
        run_phase(args, state, 'train')
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
