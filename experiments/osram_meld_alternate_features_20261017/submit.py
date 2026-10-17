"""Bounded same-server submission retry, never local training or environment changes."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import time

PYTHON = '/data2/yb/conda_envs/cmerc-repro/bin/python'
SOURCE = '/home/qinjinghui/remote_experiments/osram_meld_b3_code_737fdb1'
REMOTE = '/home/qinjinghui/remote_experiments/osram_meld_alternate_features_20261017'
DATA_SHA = '59eb11b079fc02eea2a38f81bc1132ede29077810697d3d9384c1796450ecb13'
SESSION = 'meld_flat_magtkd_seed66_1017'
CODE = REMOTE + '/code/worker.py'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'biggpu']


def command(uuid, data_sha, output, smoke=False):
    args = [PYTHON, '-u', CODE, '--data', REMOTE + '/data/magtkd_meld.pkl',
            '--data-sha256', data_sha, '--output', str(output), '--gpu-uuid', uuid,
            '--source-root', SOURCE, '--source-config', REMOTE + '/Flat66.json']
    return args + (['--smoke'] if smoke else [])


def choose_gpu(inventory, processes):
    counts = Counter(line.split(',')[0].strip() for line in processes.splitlines() if line.strip())
    rows = {}
    for line in inventory.splitlines():
        index, uuid, free = (part.strip() for part in line.split(','))
        if int(index) in (0, 2, 3) and uuid.startswith('GPU-') and float(free) >= 6000 and counts[uuid] < 4:
            rows[int(index)] = uuid
    return next((rows[i] for i in (3, 2, 0) if i in rows), None)


def ssh(text, timeout=30):
    return subprocess.run(SSH + [text], check=True, capture_output=True, text=True, timeout=timeout).stdout


def main():
    global CODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--controller-sha', required=True)
    parser.add_argument('--retry-seconds', type=int, default=60)
    parser.add_argument('--max-attempts', type=int, default=1440)
    args = parser.parse_args()
    if args.retry_seconds < 30 or not 1 <= args.max_attempts <= 1440:
        raise ValueError('Bounded submission retries only, at least30seconds apart')
    root = Path(__file__).resolve().parents[2]
    worker = Path(__file__).with_name('worker.py')
    worker_hash = hashlib.sha256(worker.read_bytes()).hexdigest()
    CODE = REMOTE + '/code/' + worker_hash + '/worker.py'
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError('Prepared MAGTKD data hash changed')
    base = root / 'experiments/osram_meld_block2_auto_20261014'
    methods = runpy.run_path(str(base / 'methods.py'))
    config = methods['build_config'](json.loads((base / 'BASE_CONFIG.json').read_text()), 'Flat', 'managed')
    args.state.mkdir(parents=True, exist_ok=True)
    config_path = args.state / 'Flat66.json'
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
    state_path = args.state / 'SUBMISSION.json'
    state = dict(status='awaiting_ssh', controller_git_sha=args.controller_sha,
                 worker_sha256=worker_hash,
                 data_sha256=DATA_SHA, remote_output=REMOTE, server='biggpu',
                 formal_training_started=False, other_release='MM-Align blocked: incomplete train and missing dev110_7')
    with (args.state / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for attempt in range(1, args.max_attempts + 1):
            state.update(attempt=attempt, checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            try:
                probe = "import json,subprocess,sys,torch,numpy; assert torch.__version__=='1.13.1+cu117'; assert numpy.__version__=='1.26.4'; print(json.dumps(dict(inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True))))"
                info = json.loads(ssh(shlex.join([PYTHON, '-c', probe])))
                uuid = choose_gpu(info['inventory'], info['processes'])
                if uuid is None:
                    state['status'] = 'waiting_for_healthy_gpu'
                else:
                    ssh(shlex.join(['mkdir', '-p', str(Path(CODE).parent), REMOTE + '/data']))
                    for local, target in ((worker, CODE), (config_path, REMOTE + '/Flat66.json'),
                                          (args.data, REMOTE + '/data/magtkd_meld.pkl')):
                        subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                            str(local), 'biggpu:' + target], check=True, capture_output=True, text=True, timeout=180)
                    # Recheck after transfer instead of relying on stale free memory.
                    info = json.loads(ssh(shlex.join([PYTHON, '-c', probe])))
                    uuid = choose_gpu(info['inventory'], info['processes'])
                    if uuid is None:
                        state['status'] = 'waiting_for_healthy_gpu'
                    else:
                        shell = shlex.join(command(uuid, DATA_SHA, REMOTE + '/smoke/MAGTKD', True))
                        shell += ' && ' + shlex.join(command(uuid, DATA_SHA, REMOTE + '/runs/Flat_MAGTKD_seed66'))
                        shell = '(' + shell + ') > ' + shlex.quote(REMOTE + '/pipeline.log') + ' 2>&1'
                        guard = (f"if test -f {REMOTE}/runs/Flat_MAGTKD_seed66/SUMMARY.json || "
                                 f"tmux has-session -t {SESSION} 2>/dev/null; then exit 0; else "
                                 f"tmux new-session -d -s {SESSION} {shlex.quote(shell)}; fi")
                        ssh(guard)
                        state.update(status='submitted_remote', gpu_uuid=uuid, remote_tmux=SESSION,
                            formal_training_started=False,
                            note='Submitted smoke-then-train chain; inspect remote SUMMARY before claiming training')
                        state_path.write_text(json.dumps(state, indent=2) + '\n')
                        print(json.dumps(state), flush=True)
                        return
            except (subprocess.SubprocessError, json.JSONDecodeError) as error:
                state.update(status='awaiting_ssh_or_transfer', error=str(error))
                if isinstance(error, subprocess.CalledProcessError):
                    state['error_detail'] = (error.stderr or '')[-2000:]
            state_path.write_text(json.dumps(state, indent=2) + '\n')
            print(json.dumps(state), flush=True)
            if attempt != args.max_attempts:
                time.sleep(args.retry_seconds)
        state['status'] = 'blocked_submission_deadline'
        state_path.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
