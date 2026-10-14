"""Launch the authorized eight independent rate diagnostics, never other jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
GPU_UUIDS = ('GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
             'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse duplicate launch/output reuse: ' + str(args.output))
    listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
    available = {}
    for row in listing.splitlines():
        index, uuid, total, used = (value.strip() for value in row.split(','))
        available[uuid] = dict(index=int(index), free_MiB=int(total) - int(used))
    for uuid in GPU_UUIDS:
        if uuid not in available or available[uuid]['index'] == 4 or available[uuid]['free_MiB'] < 16000:
            raise RuntimeError('GPU admission failed; do not kill other jobs or alter batch protocol')
    args.output.mkdir(parents=True)
    inventory = json.loads(Path(__file__).with_name('CANONICAL_INPUTS.json').read_text())
    launch = dict(created_unix=time.time(), code_root=str(ROOT), python=sys.executable,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  gpu_admission=available, runs=[])
    for i in range(8):
        uuid = GPU_UUIDS[0 if i < 4 else 1]
        output = args.output / f'rate_0p{i}'
        command = [sys.executable, str(Path(__file__).with_name('run.py')),
            '--source', inventory['source'], '--selected-masks', inventory['selected_masks'],
            '--dataset-root', '/data2/yb/paper/remote_runs_20261012/data', '--meld-pickle',
            '/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl',
            '--output', str(output), '--gpu-uuid', uuid, '--rate', str(i / 10), '--epochs', '30']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2')
        log_path = args.output / f'rate_0p{i}.log'
        with log_path.open('xb') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launch['runs'].append(dict(rate=i / 10, pid=process.pid, gpu_uuid=uuid,
                                   host_gpu=available[uuid]['index'], output=str(output),
                                   log=str(log_path), command=command))
        (args.output / 'LAUNCH.json').write_text(json.dumps(launch, indent=2) + '\n')
        print(f'rate={i / 10} pid={process.pid} hostGPU={available[uuid]["index"]}', flush=True)


if __name__ == '__main__':
    main()
