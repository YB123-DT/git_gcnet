"""Launch the explicitly authorized ten additions; keep the existing two jobs."""
import json
from pathlib import Path
import shlex
import subprocess
import time

from experiments.osram_meaningful20_20261003.manifest import write


def main():
    here = Path(__file__).resolve().parent
    if (here / 'LAUNCH_GPU2_FILL12.json').exists():
        raise FileExistsError('Already launched; never redeploy over live controllers')
    root = '/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/attempt2'
    snapshot = '/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0'
    manifest = root + '/control/SELECTED_GPU2_FILL12.json'
    controller = root + '/control/occupied_lane_fill12.py'
    python = '/data2/yb/reproduction_workspace/envs/s0/bin/python'
    gpu_uuid = 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'
    subprocess.run(['rsync', '-a', str(here.parents[1] / 'experiments/osram_meaningful20_round2_20261004/occupied_lane.py'),
                    'biggpu:' + controller], check=True)
    subprocess.run(['rsync', '-a', str(here / 'SELECTED_GPU2_FILL12.json'), 'biggpu:' + manifest], check=True)
    old = {c['id'] for c in json.loads((here / 'SELECTED_GPU0123.json').read_text())['cards']}
    cards = json.loads((here / 'SELECTED_GPU2_FILL12.json').read_text())['cards']
    records = []
    for card in cards:
        candidate = card['id']
        if candidate in old:
            continue
        command = [python, '-u', controller, '--root', root, '--snapshot', snapshot,
                   '--manifest', manifest, '--reference-root', '/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full',
                   '--baseline-audit', snapshot + '/experiments/osram_meaningful20_20261003/BASELINE_AUDIT.json',
                   '--data-manifest', '/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json',
                   '--cpu-log', '/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/cpu_template.log',
                   '--cpu-command', "CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_meaningful_input tests.test_meaningful_block_integration tests.test_meaningful_occupied_lane",
                   '--candidate', candidate, '--gpu', '2', '--gpu-uuid', gpu_uuid,
                   '--max-per-gpu', '12', '--defer-cuda-smoke-by-user', '--estimated-peak-mib', '1700']
        session = 'osram_fill12_' + candidate
        log = root + '/launch_fill12_' + candidate + '.log'
        shell = ('cd ' + shlex.quote(snapshot) + ' && export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=' +
                 gpu_uuid + '; ' + shlex.join(command) + ' >> ' + shlex.quote(log) + ' 2>&1')
        subprocess.run(['ssh', 'biggpu', shlex.join(['tmux', 'new-session', '-d', '-s', session, shell])], check=True)
        records.append(dict(candidate=candidate, gpu=2, gpu_uuid=gpu_uuid, session=session,
                            command=command, log=log, status='controller_submitted_not_yet_verified_training'))
        write(here / 'LAUNCH_GPU2_FILL12.json', dict(server='biggpu', model_snapshot=snapshot,
              controller=controller, cuda_smoke='explicitly deferred by user; not passed',
              existing_gpu2_jobs_retained=2, new_jobs=records))
        print(candidate, 'submitted', flush=True)
        time.sleep(3)


if __name__ == '__main__':
    main()
