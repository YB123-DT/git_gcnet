"""One-off explicitly authorized GPU6 batch; no CUDA smoke is claimed."""
import json
from pathlib import Path
import shlex
import subprocess
import time

from experiments.osram_meaningful20_20261003.manifest import write


def main():
    here = Path(__file__).resolve().parent
    output = here / 'LAUNCH_GPU6_ELEVEN.json'
    if output.exists():
        raise FileExistsError('Inspect already recorded jobs instead of relaunching')
    root = '/data2/yb/remote_experiments/osram_new40_gpu6_eleven_20261004'
    snapshot = '/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0'
    controller = root + '/control/occupied_lane.py'
    manifest = root + '/control/SELECTED_GPU6_ELEVEN.json'
    python = '/data2/yb/reproduction_workspace/envs/s0/bin/python'
    gpu_uuid = 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'
    subprocess.run(['ssh', 'biggpu', shlex.join(['mkdir', '-p', root + '/control'])], check=True)
    subprocess.run(['rsync', '-a', str(here.parents[1] / 'experiments/osram_meaningful20_round2_20261004/occupied_lane.py'), 'biggpu:' + controller], check=True)
    subprocess.run(['rsync', '-a', str(here / 'SELECTED_GPU6_ELEVEN.json'), 'biggpu:' + manifest], check=True)
    jobs = []
    for card in json.loads((here / 'SELECTED_GPU6_ELEVEN.json').read_text())['cards']:
        candidate = card['id']
        command = [python, '-u', controller, '--root', root, '--snapshot', snapshot,
            '--manifest', manifest, '--reference-root', '/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full',
            '--baseline-audit', snapshot + '/experiments/osram_meaningful20_20261003/BASELINE_AUDIT.json',
            '--data-manifest', '/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json',
            '--cpu-log', '/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/cpu_template.log',
            '--cpu-command', "CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_meaningful_input tests.test_meaningful_block_integration tests.test_meaningful_occupied_lane",
            '--candidate', candidate, '--gpu', '6', '--gpu-uuid', gpu_uuid,
            '--max-per-gpu', '11', '--defer-cuda-smoke-by-user', '--estimated-peak-mib', '1700']
        session = 'osram_gpu6_11_' + candidate
        log = root + '/launch_' + candidate + '.log'
        shell = ('cd ' + shlex.quote(snapshot) + ' && export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=' +
                 gpu_uuid + '; ' + shlex.join(command) + ' >> ' + shlex.quote(log) + ' 2>&1')
        subprocess.run(['ssh', 'biggpu', shlex.join(['tmux', 'new-session', '-d', '-s', session, shell])], check=True)
        jobs.append(dict(candidate=candidate, command=command, log=log, session=session,
                         status='submitted_not_yet_verified_training'))
        write(output, dict(server='biggpu', gpu_index=6, gpu_uuid=gpu_uuid, jobs=jobs,
            model_snapshot=snapshot, model_commit='ad211c0', controller_commit='b1ae9b3',
            cuda_smoke='explicitly deferred by user; resources estimated, not measured',
            global_method_budget=69, concurrent_gpu_cap=11))
        print(candidate, 'submitted', flush=True)
        time.sleep(3)


if __name__ == '__main__':
    main()
