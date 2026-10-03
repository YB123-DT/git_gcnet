"""Feed the existing training queue using one shared short-run template."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'source'
    # This coordinator lives outside the immutable model snapshot.
    sys.path.insert(0, str(source))
    from experiments.osram_meaningful20_20261003.manifest import read, write, sha, verify_snapshot
    from experiments.osram_meaningful20_20261003.preflight import query_gpus, validate_gpu
    snapshot = verify_snapshot(source)
    cards = read(source / 'experiments/osram_meaningful20_20261003/ROUND1.json')['cards']
    if len(cards) != 20:
        raise ValueError('Only the frozen first round of twenty is authorized here')
    cpu = root / 'cpu_template.log'
    if not cpu.is_file() or '\nOK\n' not in cpu.read_text():
        raise ValueError('Shared CPU template has not passed')
    with (root / 'advance.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for card in cards:
            name = card['id']
            ready = root / 'readiness' / name / 'READY.json'
            if ready.exists():
                continue
            check = root / 'checks' / name
            if check.exists():
                print(f'{name}: existing attempt requires inspection; not duplicated', flush=True)
                continue
            while True:
                resources = query_gpus()
                mapping = {index: row['uuid'] for index, row in resources.items()}
                eligible = []
                for index, row in resources.items():
                    if index == '4':
                        continue
                    validate_gpu(index, row['uuid'], mapping)
                    if row['free_mib'] >= 16384 and row['utilization'] < 90 and row['temperature'] < 85:
                        eligible.append((row['utilization'], -row['free_mib'], index))
                if eligible:
                    index = min(eligible)[-1]
                    break
                write(root / 'ADVANCE.json', {'status': 'waiting_resources', 'next': name,
                                             'max_distinct_methods': 60, 'round': 1})
                time.sleep(30)
            command = [sys.executable, '-u', '-m', 'experiments.osram_meaningful20_20261003.cuda_check',
                       '--candidate', name, '--reference', str(args.reference.resolve()),
                       '--dataset', str(root / 'DATA.json'), '--output', str(check),
                       '--gpu-index', index, '--gpu-uuid', mapping[index]]
            write(root / 'ADVANCE.json', {'status': 'checking', 'candidate': name, 'gpu': index})
            print(f'{name}: shared CUDA template on host GPU{index}', flush=True)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=mapping[index], OMP_NUM_THREADS='2',
                               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONPATH=str(source))
            with (root / f'preflight_{name}.log').open('x') as log:
                result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f'{name}: failed; preserved log, continuing other methods', flush=True)
                continue
            profile = read(check / 'profile.json')
            if profile['status'] != 'passed':
                continue
            write(ready, dict(status='ready', candidate=name, snapshot_root=str(source),
                design_sha256=card['design_sha256'], source_sha256=snapshot['source_sha256'],
                cpu=dict(command='python -m unittest tests.test_meaningful_block_integration tests.test_meaningful_blocks',
                         returncode=0, log=str(cpu), sha256=sha(cpu)),
                cuda=dict(command=command, returncode=0, log=profile['log'], sha256=profile['log_sha256'],
                          gpu_index=index, gpu_uuid=mapping[index]), profile=profile['profile']))
            print(f'{name}: ready; existing queue can dispatch immediately', flush=True)
        write(root / 'ADVANCE.json', {'status': 'first_round_checks_finished', 'max_distinct_methods': 60})


if __name__ == '__main__':
    main()
