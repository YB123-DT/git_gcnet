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
    summaries = {job['mode']: json.loads(Path(job['summary']).read_text())
                 for job in state['train']}
    result = dict(status='complete', source_commit=args.source_commit,
                  completed_at=now(), protocol='internal per-rate Test-oracle exploration',
                  methods=summaries)
    save(args.root / 'RESULT.json', result)
    lines = ['# Four-arm MOSI joint-block result', '',
             f'Source commit: `{args.source_commit}`', '',
             'Internal per-rate Test-oracle exploration; four matched seed66 arms.', '']
    for mode, summary in summaries.items():
        metrics = {key: value for key, value in summary.items()
                   if key in ('metrics', 'aggregate', 'mean_wf1', 'high_missing_wf1',
                              'mean_f1', 'best_by_rate', 'evaluation')}
        lines.extend([f'## {mode}', '', f'Summary: `{args.root / "train" / mode / "SUMMARY.json"}`',
                      '', '```json', json.dumps(metrics, indent=2), '```', ''])
    (args.root / 'RESULT.md').write_text('\n'.join(lines))


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
