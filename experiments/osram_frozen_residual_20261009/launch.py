"""Persistent CPU-only execution of seven frozen-residual conditions, two shards."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def commands(root, audit_root, python):
    root = Path(root)
    return [(name, [python, '-u', '-m', 'experiments.osram_frozen_residual_20261009.run',
                    '--audit-root', str(audit_root), '--output', str(root / 'runs' / name),
                    '--device', 'cpu', '--epochs', '100', '--seeds', '66', '67', '68',
                    '--rates', *[str(i / 10) for i in indices]])
            for name, indices in [('low', range(4)), ('high', range(4, 8))]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2]
    lock = (root / 'LAUNCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'LAUNCH.json').exists():
        raise FileExistsError('Existing launch: inspect rather than silently restart')
    if not (source / 'SNAPSHOT.json').is_file():
        raise ValueError('Sealed source snapshot required')
    now = lambda: datetime.now(timezone.utc).isoformat()
    state = dict(status='running', started_utc=now(), pid=os.getpid(), source=str(source),
                 snapshot=json.loads((source / 'SNAPSHOT.json').read_text()),
                 device='cpu', threads_per_process=2, runs=[])
    def save():
        (root / 'LAUNCH.json').write_text(json.dumps(state, indent=2))
    save()
    children = []
    for name, command in commands(root, args.audit_root.resolve(), sys.executable):
        log = (root / f'{name}.log').open('x')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['runs'].append(dict(name=name, pid=process.pid, command=command,
                                  output=str(root / 'runs' / name), log=str(root / f'{name}.log')))
        children.append((process, log))
        save()
    for (process, log), record in zip(children, state['runs']):
        record['exit_code'] = process.wait()
        log.close()
        save()
    with (root / 'analysis.log').open('x') as log:
        analysis = subprocess.run([sys.executable, '-m', 'experiments.osram_frozen_residual_20261009.analyze',
                                   '--inputs', str(root / 'runs/low'), str(root / 'runs/high'),
                                   '--output', str(root / 'summary')], cwd=source,
                                  stdout=log, stderr=subprocess.STDOUT)
    state.update(analysis_exit_code=analysis.returncode, finished_utc=now())
    state['status'] = 'complete' if analysis.returncode == 0 and all(
        row['exit_code'] == 0 for row in state['runs']) else 'failed'
    save()
    if state['status'] != 'complete':
        raise RuntimeError('Frozen residual run failed; inspect preserved logs')


if __name__ == '__main__':
    main()
